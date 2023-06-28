/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"

#include <graph/utils/utils.hpp>

#include "graph/interface/backend.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/fusion_info.hpp"
#include "graph/backend/dnnl/internal_attrs.hpp"
#include "graph/backend/dnnl/op_executable.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

const indices_t::type_t input = indices_t::type_t::input;
const indices_t::type_t output = indices_t::type_t::output;

conv_fwd_executable_t::desc_t conv_fwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::convolution_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    // prepare the operator attributes
    auto strides = op->get_attr<dims>(op_attr::strides);
    auto dilates = op->get_attr<dims>(op_attr::dilations);
    auto pads_begin = op->get_attr<dims>(op_attr::pads_begin);
    auto pads_end = op->get_attr<dims>(op_attr::pads_end);
    dilates = get_compatible_dilates(dilates);

    dnnl::primitive_attr prm_attr;
    fusion_info_t fusion_info;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        fusion_info = mgr.get_info(key);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    prm_attr.set_fpmath_mode(
            static_cast<dnnl::fpmath_mode>(mgr.get_fpmath_mode()));
    const bool can_use_blocked_layout = mgr.get_use_blocked_layout();

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());

    // assume constant weight is for inference scenario
    const auto &wei_lt = op->get_input_value(1)->get_logical_tensor();
    auto pkind = (logical_tensor_wrapper_t(wei_lt).property_type()
                         == property_type::constant)
            ? prop_kind::forward_inference
            : prop_kind::forward_training;
    auto weight = make_dnnl_memory_desc(wei_lt);
    weight = to_format_any(weight);

    auto base_conv_dst_lt = op->get_output_value(0)->get_logical_tensor();
    if (fusion_info.has_post_dw_conv()) {
        // when fused post depthwise conv, onednn required to use the base conv
        // dst md to create the conv primitive. in the subgraph, the base conv
        // dst is a intermediate output which has been fused away, so here we
        // get it from fusion info
        const auto &dw_conv = fusion_info.get_post_dw_conv();
        base_conv_dst_lt
                = dw_conv->get_op()->get_input_value(0)->get_logical_tensor();
    }
    auto dst = make_dnnl_memory_desc(base_conv_dst_lt);
    auto create_pd = [&](const dnnl::memory::desc &src_md,
                             const dnnl::memory::desc &dst_md) {
        if (op->has_attr(op_attr::with_bias)
                && op->get_attr<bool>(op_attr::with_bias)) {
            auto bias = make_dnnl_memory_desc(
                    op->get_input_value(2)->get_logical_tensor());
            bias = to_format_any(bias);
            return dnnl::convolution_forward::primitive_desc(p_engine, pkind,
                    algorithm::convolution_direct, src_md, weight, bias, dst_md,
                    strides, dilates, pads_begin, pads_end, prm_attr);
        } else {
            return dnnl::convolution_forward::primitive_desc(p_engine, pkind,
                    algorithm::convolution_direct, src_md, weight, dst_md,
                    strides, dilates, pads_begin, pads_end, prm_attr);
        }
    };

    if (!can_use_blocked_layout) {
        src = to_nxc_format(src);
        dst = to_nxc_format(dst);
    } else {
        // If the dst has been explicitly set to nxc layout or the data_format
        // has been defined as NXC by users, we prefer to directly use optimal
        // blocked src and plain dst to create conv pd. In the following, we
        // will first query out the optimal src.
        bool permute_nxc_dst = false;
        if (op->get_output_value(0)->get_consumers().size() == 1) {
            const auto &next_op
                    = op->get_output_value(0)->get_consumers()[0].get_op();
            if (next_op.get_kind() == op_kind::dnnl_permute) {
                auto permute_dst_lt
                        = next_op.get_output_value(0)->get_logical_tensor();
                auto perm = get_permutation(permute_dst_lt.ndims, "NCX", "NXC");
                if (next_op.get_attr<std::vector<int64_t>>(op_attr::permutation)
                        == perm) {
                    auto inverse_perm = get_permutation(
                            permute_dst_lt.ndims, "NXC", "NCX");
                    auto perm_dst = make_dnnl_memory_desc(permute_dst_lt);
                    dst = perm_dst.permute_axes(
                            dnnl_impl::utils::cast_to_int32(inverse_perm));
                    permute_nxc_dst = true;
                }
            }
        }
        if (!is_format(dst, "nxc") && !permute_nxc_dst) {
            src = to_format_any(src);
            dst = to_format_any(dst);
        } else {
            auto tmp_src = to_format_any(src);
            auto tmp_dst = to_format_any(dst);
            dnnl::convolution_forward::primitive_desc tmp_pd
                    = create_pd(tmp_src, tmp_dst);
            src = tmp_pd.src_desc();
        }
    }

    dnnl::convolution_forward::primitive_desc pd = create_pd(src, dst);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

deconv_fwd_executable_t::desc_t deconv_fwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::deconvolution_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    // prepare the operator attributes
    auto strides = op->get_attr<dims>(op_attr::strides);
    auto dilates = op->get_attr<dims>(op_attr::dilations);
    auto pads_begin = op->get_attr<dims>(op_attr::pads_begin);
    auto pads_end = op->get_attr<dims>(op_attr::pads_end);
    dilates = get_compatible_dilates(dilates);

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    prm_attr.set_fpmath_mode(
            static_cast<dnnl::fpmath_mode>(mgr.get_fpmath_mode()));

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    src = to_format_any(src);
    auto weight = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    weight = to_format_any(weight);
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    dst = to_format_any(dst);

    dnnl::deconvolution_forward::primitive_desc pd;
    if (op->has_attr(op_attr::with_bias)
            && op->get_attr<bool>(op_attr::with_bias)) {
        auto bias = make_dnnl_memory_desc(
                op->get_input_value(2)->get_logical_tensor());
        bias = to_format_any(bias);
        pd = dnnl::deconvolution_forward::primitive_desc(p_engine,
                prop_kind::forward_inference, algorithm::deconvolution_direct,
                src, weight, bias, dst, strides, dilates, pads_begin, pads_end,
                prm_attr);
    } else {
        pd = dnnl::deconvolution_forward::primitive_desc(p_engine,
                prop_kind::forward_inference, algorithm::deconvolution_direct,
                src, weight, dst, strides, dilates, pads_begin, pads_end,
                prm_attr);
    }

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

deconv_bwd_data_executable_t::desc_t deconv_bwd_data_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::deconvolution_backward_data::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    // prepare the operator attributes
    auto strides = op->get_attr<dims>(op_attr::strides);
    auto dilates = op->get_attr<dims>(op_attr::dilations);
    auto pads_begin = op->get_attr<dims>(op_attr::pads_begin);
    auto pads_end = op->get_attr<dims>(op_attr::pads_end);
    dilates = get_compatible_dilates(dilates);

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    prm_attr.set_fpmath_mode(
            static_cast<dnnl::fpmath_mode>(mgr.get_fpmath_mode()));

    auto diff_dst = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    diff_dst = to_format_any(diff_dst);
    auto weight = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    weight = to_format_any(weight);
    auto diff_src = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    diff_src = to_format_any(diff_src);

    auto fwd_hints = dnnl::deconvolution_forward::primitive_desc(p_engine,
            prop_kind::forward_training, algorithm::deconvolution_direct,
            diff_src, weight, diff_dst, strides, dilates, pads_begin, pads_end,
            prm_attr);

    dnnl::deconvolution_backward_data::primitive_desc pd(p_engine,
            dnnl::algorithm::deconvolution_direct, diff_src, weight, diff_dst,
            strides, pads_begin, pads_end, fwd_hints);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

deconv_bwd_weights_executable_t::desc_t
deconv_bwd_weights_executable_t::create_desc(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::deconvolution_backward_weights::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    // prepare the operator attributes
    auto strides = op->get_attr<dims>(op_attr::strides);
    auto dilates = op->get_attr<dims>(op_attr::dilations);
    auto pads_begin = op->get_attr<dims>(op_attr::pads_begin);
    auto pads_end = op->get_attr<dims>(op_attr::pads_end);
    dilates = get_compatible_dilates(dilates);

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_fpmath_mode(
            static_cast<dnnl::fpmath_mode>(mgr.get_fpmath_mode()));

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    src = to_format_any(src);
    auto diff_dst = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    diff_dst = to_format_any(diff_dst);
    auto diff_weight = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    diff_weight = to_format_any(diff_weight);

    auto fwd_hints = dnnl::deconvolution_forward::primitive_desc(p_engine,
            dnnl::prop_kind::forward_training,
            dnnl::algorithm::deconvolution_direct, src, diff_weight, diff_dst,
            strides, dilates, pads_begin, pads_end);

    dnnl::deconvolution_backward_weights::primitive_desc pd(p_engine,
            dnnl::algorithm::deconvolution_direct, src, diff_weight, diff_dst,
            strides, dilates, pads_begin, pads_end, fwd_hints);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

matmul_executable_t::desc_t matmul_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::matmul::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    prm_attr.set_fpmath_mode(
            static_cast<dnnl::fpmath_mode>(mgr.get_fpmath_mode()));

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    // For non-constant activation, create primitive desc with strided layout
    // when:
    // 1) activation has 4 dimensions and layout is acbd since oneDNN has
    //    optimized kernel
    // 2) activation has 2/3 dimensions and device kind is gpu for avoiding
    //    blocked activation. This can reduce the cost for the reorder between
    //    plain and block layout, especially for users who compile partition
    //    with plain layout. The performance of strided primitive on GPU will be
    //    optimized by oneDNN.
    bool const_activation
            = logical_tensor_wrapper_t(
                      op->get_input_value(0)->get_logical_tensor())
                      .is_constant()
            && is_constant_cache_enabled();
    const bool use_strided_src = !const_activation
            && ((src.get_ndims() == 4
                        && is_format(src, dnnl::memory::format_tag::acbd))
                    || ((src.get_ndims() == 2 || src.get_ndims() == 3)
                            && p_engine.get_kind() == dnnl::engine::kind::gpu));
    // convert src memory desc to any when:
    // 1) not the situation mentioned above
    // 2) the given md is blocked and convert to queried layout is necessary
    if (!use_strided_src || !is_plain(src)) { src = to_format_any(src); }
    auto wei = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    // For non-constant weight, create primitive desc with strided layout when:
    // 1) weight has 4 dimensions and layout is adbc/abdc/acbd since oneDNN has
    //    optimized kernel
    bool const_weight = logical_tensor_wrapper_t(
                                op->get_input_value(1)->get_logical_tensor())
                                .is_constant()
            && is_constant_cache_enabled();
    const bool use_strided_wei = !const_weight
            && (wei.get_ndims() == 4
                    && (is_format(wei, dnnl::memory::format_tag::adbc)
                            || is_format(wei, dnnl::memory::format_tag::abdc)
                            || is_format(wei, dnnl::memory::format_tag::acbd)));
    if (!use_strided_wei) { wei = to_format_any(wei); }
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    const bool keep_dst_layout = op->has_attr(op_attr::keep_dst_layout)
            && op->get_attr<bool>(op_attr::keep_dst_layout);
    const bool use_strided_dst
            = ((src.get_ndims() == 2 || src.get_ndims() == 3)
                      && p_engine.get_kind() == dnnl::engine::kind::gpu)
            || keep_dst_layout;
    if (!use_strided_dst) {
        dst = to_format_any(dst);
    } else if (dst.get_format_kind() == dnnl::memory::format_kind::any
            && !keep_dst_layout) {
        // convert to strided for avoiding blocked activation. The format kind
        // of dst is possible to be any when:
        // 1) It is created with internal logical tensor
        // 2) It is the partition output and defined by user
        dst = to_ncx_format(dst);
    } else {
        // do nothing
    }

    dnnl::matmul::primitive_desc pd;
    if (op->has_attr(op_attr::with_bias)
            && op->get_attr<bool>(op_attr::with_bias)) {
        auto bias = make_dnnl_memory_desc(
                op->get_input_value(2)->get_logical_tensor());
        bias = to_format_any(bias);
        pd = dnnl::matmul::primitive_desc(
                p_engine, src, wei, bias, dst, prm_attr);
    } else {
        pd = dnnl::matmul::primitive_desc(p_engine, src, wei, dst, prm_attr);
    }

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

pool_executable_t::desc_t pool_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::pooling_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dims strides = op->get_attr<dims>(op_attr::strides);
    dims kernel = op->get_attr<dims>(op_attr::kernel);
    dims pads_begin = op->get_attr<dims>(op_attr::pads_begin);
    dims pads_end = op->get_attr<dims>(op_attr::pads_end);
    dims dilations(strides.size(), 1);
    if (op->has_attr(op_attr::dilations)
            && (op->get_attr<std::string>(op_attr::kind) == "maxpool")) {
        dilations = op->get_attr<dims>(op_attr::dilations);
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    dst = to_format_any(dst);

    // infer dnnl explicit padding
    dims new_pads_end(pads_end);
    bool adj_pad = false;
    std::string rounding_type = "floor";
    if (op->has_attr(op_attr::rounding_type)) {
        rounding_type = op->get_attr<std::string>(op_attr::rounding_type);
    }

    // oneDNN pooling primitive doesn't support ceil mode, so we need to add
    // additional padding right to simulate the ceil mode by using floor mode,
    // and then exclude those additional paddings when doing average.
    if (rounding_type == "ceil") {
        dims src_sp = src.get_dims();
        src_sp.erase(src_sp.begin(), src_sp.begin() + 2);
        dims output_sp = dst.get_dims();
        output_sp.erase(output_sp.begin(), output_sp.begin() + 2);
        for (size_t i = 0; i < kernel.size(); ++i) {
            dim_t dilated = dilations[i] * (kernel[i] - 1) + 1;
            // calculate the expected padded input size according to floor mode
            // formula: output = (padded - dilated) / strides + 1
            dim_t expected_padded = (output_sp[i] - 1) * strides[i] + dilated;
            dim_t cur_pads_end = expected_padded - src_sp[i] - pads_begin[i];
            new_pads_end[i] = cur_pads_end;
        }
        adj_pad = true;
    }

    algorithm algo = algorithm::undef;
    prop_kind prop = prop_kind::forward_inference;
    if (op->get_attr<std::string>(op_attr::kind) == "maxpool") {
        algo = algorithm::pooling_max;
        dilations = get_compatible_dilates(dilations, src.get_ndims());
        if (op->num_outputs() == 3) {
            prop = prop_kind::forward_training;
            op->set_attr<bool>(op_attr::is_training, true);
        }
    } else if (op->get_attr<std::string>(op_attr::kind) == "avgpool") {
        const bool exclude_pad = op->get_attr<bool>(op_attr::exclude_pad);
        dilations = dims(src.get_ndims(), 0);
        algo = (exclude_pad || adj_pad)
                ? algorithm::pooling_avg_exclude_padding
                : algorithm::pooling_avg_include_padding;
    } else {
        BACKEND_DNNL_ENFORCE(
                0, "Currently only int8 MaxPool/AvgPool is supported.");
    }

    dnnl::pooling_forward::primitive_desc pd(p_engine, prop, algo, src, dst,
            strides, kernel, dilations, pads_begin, new_pads_end, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

pool_bwd_executable_t::desc_t pool_bwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::pooling_backward::primitive_desc>(pd_cache.at(op.get()));
        return {pd, true};
    }

    dims strides = op->get_attr<dims>(op_attr::strides);
    dims kernel = op->get_attr<dims>(op_attr::kernel);
    dims pads_begin = op->get_attr<dims>(op_attr::pads_begin);
    dims pads_end = op->get_attr<dims>(op_attr::pads_end);
    dims dilations(strides.size(), 0);
    if (op->has_attr(op_attr::dilations)) {
        dilations = op->get_attr<dims>(op_attr::dilations);
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto diff_dst = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto diff_src = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());

    auto src = op->get_attr<std::string>(op_attr::kind) == "maxpool"
            ? make_dnnl_memory_desc(
                    op->get_input_value(2)->get_logical_tensor())
            : dnnl::memory::desc(diff_src.get_dims(), diff_src.get_data_type(),
                    get_ncx_format(diff_src.get_dims()));

    // infer dnnl explicit pad
    dims new_pads_end(pads_end);
    bool adj_pad = false;
    std::string rounding_type = "floor";
    if (op->has_attr(op_attr::rounding_type)) {
        rounding_type = op->get_attr<std::string>(op_attr::rounding_type);
    }
    if (rounding_type == "ceil") {
        dims src_sp = src.get_dims();
        src_sp.erase(src_sp.begin(), src_sp.begin() + 2);
        dims output_sp = diff_dst.get_dims();
        output_sp.erase(output_sp.begin(), output_sp.begin() + 2);
        for (size_t i = 0; i < kernel.size(); ++i) {
            dim_t dilated = dilations[i] * (kernel[i] - 1) + 1;
            if (op->get_attr<std::string>(op_attr::kind) == "avgpool")
                dilated += 1;
            dim_t cur_pads_end = (output_sp[i] - 1) * strides[i] + dilated
                    - src_sp[i] - pads_begin[i];
            new_pads_end[i] = cur_pads_end;
        }
        adj_pad = true;
    }

    algorithm algo = algorithm::undef;
    if (op->get_attr<std::string>(op_attr::kind) == "maxpool") {
        algo = algorithm::pooling_max;
        dilations = get_compatible_dilates(dilations, src.get_ndims());
    } else if (op->get_attr<std::string>(op_attr::kind) == "avgpool") {
        const bool exclude_pad = op->get_attr<bool>(op_attr::exclude_pad);
        algo = (exclude_pad || adj_pad)
                ? algorithm::pooling_avg_exclude_padding
                : algorithm::pooling_avg_include_padding;
    } else {
        BACKEND_DNNL_ENFORCE(0,
                "Currently only MaxPoolBackprop/AvgPoolBackprop is "
                "supported.");
    }

    if (op->get_attr<std::string>(op_attr::kind) == "maxpool") {
        diff_dst = to_format_any(diff_dst);
    }

    dnnl::pooling_forward::primitive_desc forward_hints
            = dnnl::pooling_forward::primitive_desc(p_engine,
                    prop_kind::forward_training, algo, src, diff_dst, strides,
                    kernel, dilations, pads_begin, new_pads_end);

    dnnl::pooling_backward::primitive_desc pd(p_engine, algo, diff_src,
            diff_dst, strides, kernel, dilations, pads_begin, new_pads_end,
            forward_hints);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

batchnorm_executable_t::desc_t batchnorm_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::batch_normalization_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    float epsilon = op->get_attr<float>(op_attr::epsilon);

    auto flags = dnnl::normalization_flags::none;
    // for inference
    if (!op->get_attr<bool>(op_attr::is_training)) {
        flags |= dnnl::normalization_flags::use_global_stats;
        flags |= dnnl::normalization_flags::use_scale;
        flags |= dnnl::normalization_flags::use_shift;
    } else {
        // for training, inputs: [src, mean, variance, gamma, beta]
        if (op->num_inputs() > 3) {
            flags |= dnnl::normalization_flags::use_scale;
            flags |= dnnl::normalization_flags::use_shift;
        }

        if (op->has_attr(op_attr::fuse_relu)
                && op->get_attr<bool>(op_attr::fuse_relu))
            flags |= dnnl::normalization_flags::fuse_norm_relu;
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    dst = to_format_any(dst);

    if (src.get_inner_nblks() == 1 && src.get_inner_idxs()[0] == 1
            && src.get_inner_blks()[0] == 4) {
        // to default format
        src = to_ncx_format(src);
    }

    auto pkind = op->get_attr<bool>(op_attr::is_training)
            ? prop_kind::forward_training
            : prop_kind::forward_inference;

    dnnl::batch_normalization_forward::primitive_desc pd(
            p_engine, pkind, src, dst, epsilon, flags, prm_attr);

    pd_cache.insert({op.get(), pd});
    return {pd, false};
}

batchnorm_bwd_executable_t::desc_t batchnorm_bwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::batch_normalization_backward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    float epsilon = op->get_attr<float>(op_attr::epsilon);

    auto flags = dnnl::normalization_flags::none;
    // [diff_src, diff_scale, diff_shift, scratchpad]
    if (op->num_outputs() > 2) {
        flags |= dnnl::normalization_flags::use_scale;
        flags |= dnnl::normalization_flags::use_shift;
    } else {
        // [diff_src, scratchpad]
        flags |= dnnl::normalization_flags::use_global_stats;
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());

    if (src.get_inner_nblks() == 1 && src.get_inner_idxs()[0] == 1
            && src.get_inner_blks()[0] == 4) {
        // to default format
        src = to_ncx_format(src);
    }

    auto forward_hints = dnnl::batch_normalization_forward::primitive_desc(
            p_engine, prop_kind::forward_training, src, src, epsilon, flags);

    dnnl::batch_normalization_backward::primitive_desc pd(p_engine,
            prop_kind::backward, src, forward_hints.dst_desc(), src, epsilon,
            flags, forward_hints);

    pd_cache.insert({op.get(), pd});
    return {pd, false};
}

layernorm_executable_t::desc_t layernorm_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::layer_normalization_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }

    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    float epsilon = 1e-5f;
    if (op->has_attr(op_attr::epsilon))
        epsilon = op->get_attr<float>(op_attr::epsilon);
    bool keep_stats = true;
    if (op->has_attr(op_attr::keep_stats))
        keep_stats = op->get_attr<bool>(op_attr::keep_stats);
    bool use_affine = true;
    if (op->has_attr(op_attr::use_affine))
        use_affine = op->get_attr<bool>(op_attr::use_affine);

    auto flags = dnnl::normalization_flags::none;
    if (use_affine)
        flags |= (dnnl::normalization_flags::use_scale
                | dnnl::normalization_flags::use_shift);

    prop_kind pkind = keep_stats ? prop_kind::forward_training
                                 : prop_kind::forward_inference;

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());

    dnnl::layer_normalization_forward::primitive_desc pd(
            p_engine, pkind, src, dst, epsilon, flags, prm_attr);

    pd_cache.insert({op.get(), pd});
    return {pd, false};
}

layernorm_bwd_executable_t::desc_t layernorm_bwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::layer_normalization_backward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto epsilon = op->get_attr<float>(op_attr::epsilon);
    auto flags = dnnl::normalization_flags::none;
    const bool use_affine = op->get_attr<bool>(op_attr::use_affine);
    if (use_affine) {
        flags |= dnnl::normalization_flags::use_scale;
        flags |= dnnl::normalization_flags::use_shift;
    }

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto diff_dst = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    auto diff_src = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    dnnl::layer_normalization_forward::primitive_desc fwd_hints(p_engine,
            prop_kind::forward_training, src, diff_dst, epsilon, flags);

    dnnl::layer_normalization_backward::primitive_desc pd(p_engine,
            prop_kind::backward, diff_src, diff_dst, src, epsilon, flags,
            fwd_hints, prm_attr);

    pd_cache.insert({op.get(), pd});
    return {pd, false};
}

conv_bwd_data_executable_t::desc_t conv_bwd_data_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::convolution_backward_data::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    // prepare the operator attributes
    auto strides = op->get_attr<dims>(op_attr::strides);
    auto dilates = op->get_attr<dims>(op_attr::dilations);
    auto pads_begin = op->get_attr<dims>(op_attr::pads_begin);
    auto pads_end = op->get_attr<dims>(op_attr::pads_end);
    dilates = get_compatible_dilates(dilates);

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    prm_attr.set_fpmath_mode(
            static_cast<dnnl::fpmath_mode>(mgr.get_fpmath_mode()));
    const bool can_use_blocked_layout = mgr.get_use_blocked_layout();

    auto diff_dst = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    if (!can_use_blocked_layout)
        diff_dst = to_nxc_format(diff_dst);
    else
        diff_dst = to_format_any(diff_dst);
    auto weight = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    weight = to_format_any(weight);
    auto diff_src = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    if (!can_use_blocked_layout)
        diff_src = to_nxc_format(diff_src);
    else
        diff_src = to_format_any(diff_src);

    auto fwd_hints = dnnl::convolution_forward::primitive_desc(p_engine,
            dnnl::prop_kind::forward_training,
            dnnl::algorithm::convolution_direct, diff_src, weight, diff_dst,
            strides, dilates, pads_begin, pads_end);

    dnnl::convolution_backward_data::primitive_desc pd(p_engine,
            dnnl::algorithm::convolution_direct, diff_src, weight, diff_dst,
            strides, dilates, pads_begin, pads_end, fwd_hints);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

conv_bwd_weights_executable_t::desc_t
conv_bwd_weights_executable_t::create_desc(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::convolution_backward_weights::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    // prepare the operator attributes
    auto strides = op->get_attr<dims>(op_attr::strides);
    auto dilates = op->get_attr<dims>(op_attr::dilations);
    auto pads_begin = op->get_attr<dims>(op_attr::pads_begin);
    auto pads_end = op->get_attr<dims>(op_attr::pads_end);
    dilates = get_compatible_dilates(dilates);

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    prm_attr.set_fpmath_mode(
            static_cast<dnnl::fpmath_mode>(mgr.get_fpmath_mode()));
    const bool can_use_blocked_layout = mgr.get_use_blocked_layout();

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    if (!can_use_blocked_layout)
        src = to_nxc_format(src);
    else
        src = to_format_any(src);
    auto diff_dst = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    if (!can_use_blocked_layout)
        diff_dst = to_nxc_format(diff_dst);
    else
        diff_dst = to_format_any(diff_dst);
    auto diff_weight = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    diff_weight = to_format_any(diff_weight);

    auto fwd_hints = dnnl::convolution_forward::primitive_desc(p_engine,
            dnnl::prop_kind::forward_training,
            dnnl::algorithm::convolution_direct, src, diff_weight, diff_dst,
            strides, dilates, pads_begin, pads_end);

    dnnl::convolution_backward_weights::primitive_desc pd(p_engine,
            dnnl::algorithm::convolution_direct, src, diff_weight, diff_dst,
            strides, dilates, pads_begin, pads_end, fwd_hints);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

eltwise_executable_t::desc_t eltwise_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::eltwise_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    float alpha = 0.f, beta = 0.f;
    if (op->has_attr(op_attr::alpha)) {
        alpha = op->get_attr<float>(op_attr::alpha);
    }
    if (op->has_attr(op_attr::beta)) {
        beta = op->get_attr<float>(op_attr::beta);
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    dst = to_format_any(dst);

    const algorithm algo = static_cast<dnnl::algorithm>(
            op->get_attr<int64_t>(op_attr::alg_kind));
    if (algo == algorithm::undef) {
        BACKEND_DNNL_ENFORCE(0, "Unsupported eltwise op.");
    }

    dnnl::eltwise_forward::primitive_desc pd;
    pd = dnnl::eltwise_forward::primitive_desc(p_engine, prop_kind::forward,
            algo, src, dst, alpha, beta, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

eltwise_bwd_executable_t::desc_t eltwise_bwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::eltwise_backward::primitive_desc>(pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    const float alpha = op->has_attr(op_attr::alpha)
            ? op->get_attr<float>(op_attr::alpha)
            : 0.f;
    const float beta = op->has_attr(op_attr::beta)
            ? op->get_attr<float>(op_attr::beta)
            : 0.f;
    const auto bwd_algo = static_cast<dnnl::algorithm>(
            op->get_attr<int64_t>(op_attr::alg_kind));
    const auto fwd_algo = static_cast<dnnl::algorithm>(
            op->get_attr<int64_t>(op_attr::fwd_alg_kind));

    auto forward_data = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    dnnl::eltwise_forward::primitive_desc fwd_hints(p_engine,
            prop_kind::forward_training, fwd_algo, forward_data, forward_data,
            alpha, beta, prm_attr);

    auto diff_dst = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    auto diff_src = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    diff_dst = to_format_any(diff_dst);
    diff_src = to_format_any(diff_src);
    dnnl::eltwise_backward::primitive_desc pd(p_engine, bwd_algo, diff_src,
            diff_dst, forward_data, alpha, beta, fwd_hints, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

sum_executable_t::desc_t sum_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::sum::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    std::vector<dnnl::memory::desc> src_descs;
    src_descs.reserve(op->num_inputs());
    for (const auto &in_val : op->get_input_values()) {
        src_descs.emplace_back(
                make_dnnl_memory_desc(in_val->get_logical_tensor()));
    }

    auto dst_desc = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());

    // create default scales
    std::vector<float> scales(op->num_inputs(), 1.f);

    dnnl::sum::primitive_desc pd(p_engine, dst_desc, scales, src_descs);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

concat_executable_t::desc_t concat_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {graph::utils::any_cast<dnnl::concat::primitive_desc>(
                        pd_cache.at(op.get())),
                false};
    }

    // Here we force to use plain-in-plain-out (acdb) for 4D case to make
    // sure good performance of DenseNet121 (reducing reorder overhead).
    // But for other cases like 2D/3D (e.g. DLRM), we just use default
    // format since there may be followed by a non-DNNL op which requires an
    // input with default format. Anyway it looks like a bit tricky.
    auto get_forced_format_tag = [](const dims &in_dims) -> format_tag {
        if (in_dims.size() == 4)
            return format_tag::acdb;
        else
            return get_ncx_format(in_dims);
    };

    const auto rank = op->get_output_value(0)->get_logical_tensor().ndims;
    const auto res = utils::try_reverse_axis(
            op->get_attr<int64_t>(op_attr::axis), rank);
    assertm(res.first, "Incorrect axis value.");
    const auto axis = res.second;

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    std::vector<memory::desc> src_mds;
    src_mds.reserve(op->num_inputs());
    for (const auto &in_val : op->get_input_values()) {
        const auto tmp_desc
                = make_dnnl_memory_desc(in_val->get_logical_tensor());
        src_mds.emplace_back(
                memory::desc {tmp_desc.get_dims(), tmp_desc.get_data_type(),
                        get_forced_format_tag(tmp_desc.get_dims())});
    }
    const auto tmp_desc = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    auto dst = memory::desc {tmp_desc.get_dims(), tmp_desc.get_data_type(),
            get_forced_format_tag(tmp_desc.get_dims())};

    dnnl::concat::primitive_desc pd(
            p_engine, dst, static_cast<int>(axis), src_mds, prm_attr);
    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

resampling_executable_t::desc_t resampling_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::resampling_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    // resampling src doesn't support any
    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    dst = to_format_any(dst);

    std::string mode = op->get_attr<std::string>(op_attr::mode);
    algorithm algo = algorithm::undef;
    if (mode == "nearest") {
        algo = algorithm::resampling_nearest;
    } else if (mode == "linear" || mode == "bilinear" || mode == "trilinear") {
        algo = algorithm::resampling_linear;
    } else {
        BACKEND_DNNL_ENFORCE(0, "Unsupported resampling mode.");
    }

    dnnl::resampling_forward::primitive_desc pd;
    pd = dnnl::resampling_forward::primitive_desc(
            p_engine, prop_kind::forward_inference, algo, src, dst, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

resampling_bwd_executable_t::desc_t resampling_bwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::resampling_backward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto mode = op->get_attr<std::string>(op_attr::mode);
    auto algo = algorithm::undef;
    if (mode == "nearest") {
        algo = algorithm::resampling_nearest;
    } else if (mode == "linear" || mode == "bilinear" || mode == "trilinear") {
        algo = algorithm::resampling_linear;
    } else {
        BACKEND_DNNL_ENFORCE(0, "Unsupported resampling mode.");
    }

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto diff_dst = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    dnnl::resampling_forward::primitive_desc fwd_hints(p_engine,
            prop_kind::forward_training, algo, src, to_format_any(diff_dst),
            prm_attr);

    auto diff_src = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    diff_src = to_format_any(diff_src);
    dnnl::resampling_backward::primitive_desc pd(
            p_engine, algo, diff_src, diff_dst, fwd_hints, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

binary_executable_t::desc_t binary_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::binary::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src0 = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto src1 = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    auto tmp_dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());

    // For binary, if we set dst memory tag any, it will deduce strange format
    // for dst when src0 shape is 1x1x1x1, such as cdab. It will cause binary
    // performance poor, and the post matmul pattern performance is poor.
    // So we force dst format to src0 format.
    auto format_tag = get_format_tag_str(src0);
    const auto &dims = tmp_dst.get_dims();
    const auto &dtype = tmp_dst.get_data_type();
    dnnl_memory_desc_t dst_c;
    dnnl_memory_desc_create_with_string_tag(&dst_c,
            static_cast<int>(dims.size()), dims.data(),
            static_cast<dnnl_data_type_t>(dtype), format_tag.data());
    dnnl::memory::desc dst;
    dst.reset(dst_c);

    const algorithm algo = static_cast<dnnl::algorithm>(
            op->get_attr<int64_t>(op_attr::alg_kind));

    dnnl::binary::primitive_desc pd;
    pd = dnnl::binary::primitive_desc(
            p_engine, algo, src0, src1, dst, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

prelu_executable_t::desc_t prelu_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::prelu_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto wei = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    wei = to_format_any(wei);
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    dst = to_format_any(dst);

    dnnl::prelu_forward::primitive_desc pd(
            p_engine, prop_kind::forward, src, wei, dst, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

prelu_bwd_executable_t::desc_t prelu_bwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::prelu_backward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto forward_data = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto wei = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    wei = to_format_any(wei);
    auto diff_dst = make_dnnl_memory_desc(
            op->get_input_value(2)->get_logical_tensor());

    auto diff_src = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    auto diff_wei = make_dnnl_memory_desc(
            op->get_output_value(1)->get_logical_tensor());
    diff_wei = to_format_any(diff_wei);

    auto hint_fwd_pd = dnnl::prelu_forward::primitive_desc(p_engine,
            prop_kind::forward, forward_data, wei, diff_dst, prm_attr);

    dnnl::prelu_backward::primitive_desc pd(p_engine, forward_data, wei,
            diff_src, diff_wei, diff_dst, hint_fwd_pd, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

softmax_executable_t::desc_t softmax_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::softmax_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());

    int64_t axis = op->get_attr<int64_t>(op_attr::axis);
    if (axis < 0) { axis += src.get_ndims(); }

    const dnnl::algorithm algo
            = op->get_kind() == dnnl_impl::op_kind::dnnl_logsoftmax
            ? dnnl::algorithm::softmax_log
            : dnnl::algorithm::softmax_accurate;

    dnnl::softmax_forward::primitive_desc pd;
    pd = dnnl::softmax_forward::primitive_desc(p_engine,
            prop_kind::forward_inference, algo, src, dst,
            static_cast<int>(axis), prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

softmax_bwd_executable_t::desc_t softmax_bwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::softmax_backward::primitive_desc>(pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto diff_dst = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    diff_dst = to_format_any(diff_dst);

    auto diff_src_lt = op->get_output_value(0)->get_logical_tensor();
    auto diff_src = make_dnnl_memory_desc(diff_src_lt);

    const auto rank = op->get_output_value(0)->get_logical_tensor().ndims;
    const auto res = utils::try_reverse_axis(
            op->get_attr<int64_t>(op_attr::axis), rank);
    assertm(res.first, "Incorrect axis value.");
    const auto axis = res.second;

    // construct src with layout information from dst and data type information
    // from diff_src.
    auto dst_lt = op->get_input_value(1)->get_logical_tensor();
    dst_lt.data_type = diff_src_lt.data_type;
    auto dst = make_dnnl_memory_desc(dst_lt);
    const dnnl::memory::desc &src = dst;

    const dnnl::algorithm algo
            = op->get_kind() == dnnl_impl::op_kind::dnnl_logsoftmax_bwd
            ? dnnl::algorithm::softmax_log
            : dnnl::algorithm::softmax_accurate;

    auto hint_fwd_pd = dnnl::softmax_forward::primitive_desc(p_engine,
            prop_kind::forward_training, algo, src, dst, static_cast<int>(axis),
            prm_attr);

    auto pd = dnnl::softmax_backward::primitive_desc(p_engine, algo, diff_src,
            diff_dst, dst, static_cast<int>(axis), hint_fwd_pd, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

shuffle_executable_t::desc_t shuffle_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::shuffle_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    const int group = static_cast<int>(op->get_attr<int64_t>(op_attr::groups));
    const int axis = static_cast<int>(op->get_attr<int64_t>(op_attr::axis));

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    dst = to_format_any(dst);

    dnnl::shuffle_forward::primitive_desc pd(p_engine,
            prop_kind::forward_inference, src, dst, axis, group, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

reduction_executable_t::desc_t reduction_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::reduction::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    const algorithm alg = static_cast<dnnl::algorithm>(
            op->get_attr<int64_t>(op_attr::alg_kind));
    if (alg == algorithm::undef) {
        BACKEND_DNNL_ENFORCE(0, "Unsupported reduction op.");
    }
    float p = op->has_attr(op_attr::p) ? op->get_attr<float>(op_attr::p) : 0.f;

    float eps = 0.0f;

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    dst = to_format_any(dst);

    dnnl::reduction::primitive_desc pd(
            p_engine, alg, src, dst, p, eps, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

reorder_executable_t::desc_t reorder_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::reorder::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }

    // generate mask
    int mask = 0;
    if (op->has_attr(op_attr::axis) && op->has_attr(op_attr::qtype)) {
        int64_t axis = op->get_attr<int64_t>(op_attr::axis);
        std::string qtype = op->get_attr<std::string>(op_attr::qtype);
        mask = qtype == "per_tensor" ? 0 : 1 << axis;
    }

    if (op->has_attr(op_attr::with_runtime_src_zps)
            && op->get_attr<bool>(op_attr::with_runtime_src_zps)) {
        // runtime src zps
        prm_attr.set_zero_points_mask(DNNL_ARG_FROM, mask);
    } else if (op->has_attr(op_attr::src_zps)) {
        assertm(false, "only support runtime src zero points.\n");
    }

    if (op->has_attr(op_attr::with_runtime_scales)
            && op->get_attr<bool>(op_attr::with_runtime_scales)) {
        // runtime arg scales
        prm_attr.set_scales_mask(DNNL_ARG_SRC, mask);
    } else if (op->has_attr(op_attr::scales)) {
        assertm(false, "only support runtime arg scales.\n");
    }

    if (op->has_attr(op_attr::with_runtime_dst_zps)
            && op->get_attr<bool>(op_attr::with_runtime_dst_zps)) {
        // runtime dst zps
        prm_attr.set_zero_points_mask(DNNL_ARG_TO, mask);
    } else if (op->has_attr(op_attr::dst_zps)) {
        assertm(false, "only support runtime dst zero points.\n");
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto in_md = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto out_md = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());

    auto pd = dnnl::reorder::primitive_desc(
            p_engine, in_md, p_engine, out_md, prm_attr);
    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

bn_folding_t::desc_t bn_folding_t::create_desc(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache) {
    UNUSED(mgr);
    UNUSED(pd_cache);

    desc_t desc;

    desc.epsilon_ = op->get_attr<float>(op_attr::epsilon);
    desc.data_format_ = op->get_attr<std::string>(op_attr::data_format);
    desc.filter_format_ = op->get_attr<std::string>(op_attr::weights_format);
    desc.with_bias_ = op->get_attr<bool>(op_attr::with_bias);

    size_t in_idx = 0;
    auto weights = make_dnnl_memory_desc(
            op->get_input_value(in_idx++)->get_logical_tensor());
    auto bias = desc.with_bias_ ? make_dnnl_memory_desc(
                        op->get_input_value(in_idx++)->get_logical_tensor())
                                : memory::desc();
    auto scale = make_dnnl_memory_desc(
            op->get_input_value(in_idx++)->get_logical_tensor());
    auto shift = make_dnnl_memory_desc(
            op->get_input_value(in_idx++)->get_logical_tensor());
    auto mean = make_dnnl_memory_desc(
            op->get_input_value(in_idx++)->get_logical_tensor());
    auto variance = make_dnnl_memory_desc(
            op->get_input_value(in_idx++)->get_logical_tensor());

    // 1. sqrt_variance = sqrt(variance + epsilon)

    // temp = variance + epsilon
    memory::dims epsilon_dims(variance.get_ndims(), 1);
    desc.epsilon_desc_ = memory::desc(
            epsilon_dims, memory::data_type::f32, memory::format_tag::a);

    post_ops add_post_ops;
    // sqrt_variance = sqrt(temp)
    add_post_ops.append_eltwise(algorithm::eltwise_sqrt, 0.0f, 0.0f);

    primitive_attr add_attr;
    add_attr.set_post_ops(add_post_ops);
    desc.add_pd_ = dnnl::binary::primitive_desc(p_engine, algorithm::binary_add,
            variance, desc.epsilon_desc_, variance, add_attr);

    // 2. updated_weight = weights * scale / sqrt_variance

    // expand 1D scale and variance to same ndims with weights
    desc.new_scale_desc_ = expand(scale, weights.get_ndims());
    desc.new_variance_desc_ = expand(variance, weights.get_ndims());

    // after expand, the c channel is on the last dimension, which
    // meet the requirement of NXC format. But for NCX format, we
    // need permute c channel to the second dimension
    if (desc.filter_format_ == "NCX") { // matmul case
        auto perm = dnnl_impl::utils::cast_to_int32(get_permutation(
                desc.new_scale_desc_.get_ndims(), "NXC", "NCX"));
        desc.new_scale_desc_ = desc.new_scale_desc_.permute_axes(perm);
        desc.new_variance_desc_ = desc.new_variance_desc_.permute_axes(perm);
    }

    // after expand, the c channel is on the last dimension, which
    // meet the requirement of XIO format. But for OIX format, we
    // need permute c channel to the first dimension
    if (desc.filter_format_ == "OIX") { // conv case
        auto perm = dnnl_impl::utils::cast_to_int32(get_permutation(
                desc.new_scale_desc_.get_ndims(), "XIO", "OIX"));
        desc.new_scale_desc_ = desc.new_scale_desc_.permute_axes(perm);
        desc.new_variance_desc_ = desc.new_variance_desc_.permute_axes(perm);
    }

    // temp = weights * scale
    post_ops mul_post_ops;
    // updated_weight = temp / sqrt_variance
    mul_post_ops.append_binary(algorithm::binary_div, desc.new_variance_desc_);

    primitive_attr mul_attr;
    mul_attr.set_post_ops(mul_post_ops);
    desc.mul_pd_ = dnnl::binary::primitive_desc(p_engine, algorithm::binary_mul,
            weights, desc.new_scale_desc_, weights, mul_attr);

    // 3. updated_bias = (bias - mean) * scale / sqrt_variance + shift

    // temp = bias - mean
    memory::desc valid_bias = bias.is_zero() ? mean : bias;

    post_ops sub_post_ops;
    // temp = temp * scale
    sub_post_ops.append_binary(algorithm::binary_mul, scale);
    // temp = temp / sqrt_variance
    sub_post_ops.append_binary(algorithm::binary_div, variance);
    // temp = temp + shift
    sub_post_ops.append_binary(algorithm::binary_add, shift);

    primitive_attr sub_attr;
    sub_attr.set_post_ops(sub_post_ops);
    desc.sub_pd_ = dnnl::binary::primitive_desc(p_engine, algorithm::binary_sub,
            valid_bias, mean, valid_bias, sub_attr);

    memory::dims scratchpad_dims = variance.get_dims();
    // sqrt_variance, zero_bias and others (like epsilon),
    // or no need to alloc bias
    size_t factor = bias.is_zero() ? 3 : 2;
    scratchpad_dims[0] *= factor;
    desc.scratchpad_desc_ = memory::desc(
            scratchpad_dims, variance.get_data_type(), memory::format_tag::a);

    return desc;
}

static void get_arg_indices_for_post_ops(const op_t *op, fusion_info_mgr_t &mgr,
        arg_indices_t &indices, size_t &base_index) {
    const fusion_info_t &fusion_info
            = (op->has_attr(op_attr::fusion_info_key)
                      && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1)
            ? mgr.get_info(op->get_attr<int64_t>(op_attr::fusion_info_key))
            : fusion_info_t();
    const auto &pops = fusion_info.get_post_ops();
    for (size_t i = 0; i < pops.size(); i++) {
        if (pops[i]->is_post_sum()) {
            indices.insert(
                    {DNNL_GRAPH_ARG_POST_SRC, indices_t {input, base_index++}});
        } else if (pops[i]->get_op()->get_kind() == op_kind::dnnl_binary) {
            indices.insert(
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP((int)i) | DNNL_ARG_SRC_1,
                            indices_t {input, base_index++}});
        } else if (pops[i]->get_op()->get_kind() == op_kind::dnnl_convolution) {
            indices.insert({DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS,
                    indices_t {input, base_index++}});
        } else {
        }
    }
}

static arg_indices_t get_arg_indices_for_conv_and_matmul(
        const op_t *op, fusion_info_mgr_t &mgr) {
    arg_indices_t arg_indices;

    // add input args
    size_t index = 0;
    arg_indices.insert({DNNL_ARG_SRC, indices_t {input, index++}});
    arg_indices.insert({DNNL_ARG_WEIGHTS, indices_t {input, index++}});
    if (op->has_attr(op_attr::with_bias)
            && op->get_attr<bool>(op_attr::with_bias)) {
        arg_indices.insert({DNNL_ARG_BIAS, indices_t {input, index++}});
    }

    const fusion_info_t &fusion_info
            = (op->has_attr(op_attr::fusion_info_key)
                      && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1)
            ? mgr.get_info(op->get_attr<int64_t>(op_attr::fusion_info_key))
            : fusion_info_t();

    if (fusion_info.with_runtime_scales(true, 0)) {
        arg_indices.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
                indices_t {input, index++}});
    }

    if (fusion_info.with_runtime_scales(true, 1)) {
        arg_indices.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
                indices_t {input, index++}});
    }

    if (fusion_info.with_runtime_zero_points(true, 0)) {
        arg_indices.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC,
                indices_t {input, index++}});
    }

    if (fusion_info.with_runtime_zero_points(true, 1)) {
        arg_indices.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
                indices_t {input, index++}});
    }

    get_arg_indices_for_post_ops(op, mgr, arg_indices, index);

    if (fusion_info.with_runtime_scales(false, 0)) {
        arg_indices.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
                indices_t {input, index++}});
    }

    if (fusion_info.with_runtime_zero_points(false, 0)) {
        arg_indices.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST,
                indices_t {input, index++}});
    }

    // add output args
    arg_indices.insert({DNNL_ARG_DST, indices_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indices_t {output, 1}});

    return arg_indices;
}

arg_indices_t conv_fwd_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_conv_and_matmul(op, mgr);
}

arg_indices_t deconv_fwd_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_conv_and_matmul(op, mgr);
}

arg_indices_t matmul_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_conv_and_matmul(op, mgr);
}

arg_indices_t binary_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    arg_indices_t arg_indices;

    // add input args
    size_t index = 0;
    arg_indices.insert({DNNL_ARG_SRC_0, indices_t {input, index++}});
    arg_indices.insert({DNNL_ARG_SRC_1, indices_t {input, index++}});

    get_arg_indices_for_post_ops(op, mgr, arg_indices, index);

    // add output args
    arg_indices.insert({DNNL_ARG_DST, indices_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indices_t {output, 1}});

    return arg_indices;
}

arg_indices_t prelu_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    // add input args
    size_t index = 0;
    arg_indices.insert({DNNL_ARG_SRC, indices_t {input, index++}});
    arg_indices.insert({DNNL_ARG_WEIGHTS, indices_t {input, index++}});

    // add output args
    arg_indices.insert({DNNL_ARG_DST, indices_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indices_t {output, 1}});

    return arg_indices;
}

arg_indices_t prelu_bwd_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    // add input args
    arg_indices.insert({DNNL_ARG_SRC, indices_t {input, 0}});
    arg_indices.insert({DNNL_ARG_WEIGHTS, indices_t {input, 1}});
    arg_indices.insert({DNNL_ARG_DIFF_DST, indices_t {input, 2}});

    // add output args
    arg_indices.insert({DNNL_ARG_DIFF_SRC, indices_t {output, 0}});
    arg_indices.insert({DNNL_ARG_DIFF_WEIGHTS, indices_t {output, 1}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indices_t {output, 2}});

    return arg_indices;
}

arg_indices_t memory_reparser_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;
    arg_indices.insert({DNNL_ARG_FROM, indices_t {input, 0}});
    arg_indices.insert({DNNL_ARG_TO, indices_t {output, 0}});
    return arg_indices;
}

// for single-input-single-output op
static arg_indices_t get_arg_indices_for_siso_op(
        const op_t *op, fusion_info_mgr_t &mgr) {
    arg_indices_t arg_indices;

    // add input args
    size_t index = 0;
    arg_indices.insert({DNNL_ARG_FROM, indices_t {input, index++}});

    const fusion_info_t &fusion_info
            = (op->has_attr(op_attr::fusion_info_key)
                      && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1)
            ? mgr.get_info(op->get_attr<int64_t>(op_attr::fusion_info_key))
            : fusion_info_t();

    if (fusion_info.with_runtime_scales(false, 0)) {
        arg_indices.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
                indices_t {input, index++}});
    }

    get_arg_indices_for_post_ops(op, mgr, arg_indices, index);

    // add output args
    arg_indices.insert({DNNL_ARG_TO, indices_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indices_t {output, 1}});

    const bool is_training = op->has_attr(op_attr::is_training)
            ? op->get_attr<bool>(op_attr::is_training)
            : false;
    if (is_training) {
        arg_indices.insert({DNNL_ARG_WORKSPACE, indices_t {output, 2}});
    }

    return arg_indices;
}

arg_indices_t pool_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_siso_op(op, mgr);
}

arg_indices_t softmax_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_siso_op(op, mgr);
}

arg_indices_t eltwise_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_siso_op(op, mgr);
}

arg_indices_t shuffle_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_siso_op(op, mgr);
}

arg_indices_t reduction_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_siso_op(op, mgr);
}

arg_indices_t resampling_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_siso_op(op, mgr);
}

arg_indices_t pool_bwd_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    arg_indices_t arg_indices;

    // add input args
    arg_indices.insert({DNNL_ARG_DIFF_DST, indices_t {input, 0}});
    if (op->get_attr<std::string>(op_attr::kind) == "maxpool") {
        // maxpool bwd op must need workspace input
        arg_indices.insert({DNNL_ARG_WORKSPACE, indices_t {input, 1}});
    }

    arg_indices.insert({DNNL_ARG_DIFF_SRC, indices_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indices_t {output, 1}});
    return arg_indices;
}

static arg_indices_t get_arg_indices_for_miso_op(
        const op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    for (size_t i = 0; i < op->num_inputs(); ++i) {
        arg_indices.insert({DNNL_ARG_MULTIPLE_SRC + (int)i,
                indices_t {input, static_cast<size_t>(i)}});
    }

    arg_indices.insert({DNNL_ARG_DST, indices_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indices_t {output, 1}});
    return arg_indices;
}

arg_indices_t concat_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_miso_op(op, mgr);
}

arg_indices_t sum_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_miso_op(op, mgr);
}

arg_indices_t bn_folding_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    size_t in_idx = 0;
    arg_indices.insert({DNNL_ARG_WEIGHTS, indices_t {input, in_idx++}});
    if (op->get_attr<bool>(op_attr::with_bias)) {
        arg_indices.insert({DNNL_ARG_BIAS, indices_t {input, in_idx++}});
    }
    arg_indices.insert(
            {DNNL_ARG_WEIGHTS_1, indices_t {input, in_idx++}}); // scale
    arg_indices.insert(
            {DNNL_ARG_WEIGHTS_2, indices_t {input, in_idx++}}); // shift
    arg_indices.insert({DNNL_ARG_MEAN, indices_t {input, in_idx++}}); // mean
    arg_indices.insert(
            {DNNL_ARG_VARIANCE, indices_t {input, in_idx++}}); // variance

    // bind output memory
    size_t out_idx = 0;
    arg_indices.insert(
            {DNNL_ARG_DST_0, indices_t {output, out_idx++}}); // updated weight
    arg_indices.insert(
            {DNNL_ARG_DST_1, indices_t {output, out_idx++}}); // updated bias
    arg_indices.insert(
            {DNNL_ARG_SCRATCHPAD, indices_t {output, out_idx++}}); // scratchpad

    return arg_indices;
}

arg_indices_t conv_bwd_data_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    arg_indices.insert({DNNL_ARG_DIFF_DST, indices_t {input, 0}});
    arg_indices.insert({DNNL_ARG_WEIGHTS, indices_t {input, 1}});

    arg_indices.insert({DNNL_ARG_DIFF_SRC, indices_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indices_t {output, 1}});

    return arg_indices;
}

arg_indices_t deconv_bwd_data_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    return conv_bwd_data_executable_t::get_arg_indices(op, mgr);
}

arg_indices_t conv_bwd_weights_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    arg_indices.insert({DNNL_ARG_SRC, indices_t {input, 0}});
    arg_indices.insert({DNNL_ARG_DIFF_DST, indices_t {input, 1}});

    arg_indices.insert({DNNL_ARG_DIFF_WEIGHTS, indices_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indices_t {output, 1}});

    return arg_indices;
}

arg_indices_t deconv_bwd_weights_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    return conv_bwd_weights_executable_t::get_arg_indices(op, mgr);
}

arg_indices_t batchnorm_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    size_t in_index = 0;
    arg_indices.insert({DNNL_ARG_SRC, indices_t {input, in_index++}});
    if (!op->get_attr<bool>(op_attr::is_training)) { // inference
        arg_indices.insert({DNNL_ARG_SCALE, indices_t {input, in_index++}});
        arg_indices.insert({DNNL_ARG_SHIFT, indices_t {input, in_index++}});
        arg_indices.insert({DNNL_ARG_MEAN, indices_t {input, in_index++}});
        arg_indices.insert({DNNL_ARG_VARIANCE, indices_t {input, in_index++}});
    } else { // training
        // running_mean/running_variance of last iteration
        arg_indices.insert({DNNL_ARG_SRC_1, indices_t {input, in_index++}});
        arg_indices.insert({DNNL_ARG_SRC_2, indices_t {input, in_index++}});

        if (op->num_inputs() > 3) {
            arg_indices.insert({DNNL_ARG_SCALE, indices_t {input, in_index++}});
            arg_indices.insert({DNNL_ARG_SHIFT, indices_t {input, in_index++}});
        }
    }

    size_t out_index = 0;
    arg_indices.insert({DNNL_ARG_DST, indices_t {output, out_index++}});
    if (op->get_attr<bool>(op_attr::is_training)) {
        // running_mean
        arg_indices.insert({DNNL_ARG_DST_1, indices_t {output, out_index++}});
        // running_variance
        arg_indices.insert({DNNL_ARG_DST_2, indices_t {output, out_index++}});
        // batch_mean
        arg_indices.insert({DNNL_ARG_MEAN, indices_t {output, out_index++}});
        // batch_variance
        arg_indices.insert(
                {DNNL_ARG_VARIANCE, indices_t {output, out_index++}});
    }

    if (op->num_outputs() > out_index) {
        arg_indices.insert(
                {DNNL_ARG_SCRATCHPAD, indices_t {output, out_index++}});
    }

    // workspace (for BatchNormForwardTraining with ReLU)
    if (op->num_outputs() > out_index) {
        arg_indices.insert(
                {DNNL_ARG_WORKSPACE, indices_t {output, out_index++}});
    }

    return arg_indices;
}

arg_indices_t batchnorm_bwd_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    arg_indices_t arg_indices;
    size_t index = 0;

    arg_indices.insert({DNNL_ARG_SRC, indices_t {input, index++}});
    arg_indices.insert({DNNL_ARG_DIFF_DST, indices_t {input, index++}});

    arg_indices.insert({DNNL_ARG_MEAN, indices_t {input, index++}});
    arg_indices.insert({DNNL_ARG_VARIANCE, indices_t {input, index++}});

    if (op->num_outputs() > 2) {
        // oneDNN only need the scales now
        arg_indices.insert({DNNL_ARG_SCALE, indices_t {input, index++}});
    }

    index = 0;
    arg_indices.insert({DNNL_ARG_DIFF_SRC, indices_t {output, index++}});
    // check if has diff_scale and diff_shift outputs
    if (op->num_outputs() > 2) {
        arg_indices.insert({DNNL_ARG_DIFF_SCALE, indices_t {output, index++}});
        arg_indices.insert({DNNL_ARG_DIFF_SHIFT, indices_t {output, index++}});
    }

    if (op->num_outputs() > index) {
        arg_indices.insert({DNNL_ARG_SCRATCHPAD, indices_t {output, index++}});
    }

    return arg_indices;
}

arg_indices_t layernorm_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    size_t in_index = 0;
    arg_indices.insert({DNNL_ARG_SRC, indices_t {input, in_index++}});
    if (!op->has_attr(op_attr::use_affine)
            || op->get_attr<bool>(op_attr::use_affine)) {
        arg_indices.insert({DNNL_ARG_SCALE, indices_t {input, in_index++}});
        arg_indices.insert({DNNL_ARG_SHIFT, indices_t {input, in_index++}});
    }

    const fusion_info_t &fusion_info
            = (op->has_attr(op_attr::fusion_info_key)
                      && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1)
            ? mgr.get_info(op->get_attr<int64_t>(op_attr::fusion_info_key))
            : fusion_info_t();

    if (fusion_info.with_runtime_scales(false, 0)) {
        arg_indices.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
                indices_t {input, in_index++}});
    }

    size_t out_index = 0;
    arg_indices.insert({DNNL_ARG_DST, indices_t {output, out_index++}});
    if (!op->has_attr(op_attr::keep_stats)
            || op->get_attr<bool>(op_attr::keep_stats)) {
        arg_indices.insert({DNNL_ARG_MEAN, indices_t {output, out_index++}});
        arg_indices.insert(
                {DNNL_ARG_VARIANCE, indices_t {output, out_index++}});
    }

    if (op->num_outputs() > out_index) {
        arg_indices.insert(
                {DNNL_ARG_SCRATCHPAD, indices_t {output, out_index++}});
    }

    return arg_indices;
}

arg_indices_t layernorm_bwd_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    arg_indices_t arg_indices;

    arg_indices.insert({DNNL_ARG_SRC, indices_t {input, 0}});
    arg_indices.insert({DNNL_ARG_DIFF_DST, indices_t {input, 1}});
    arg_indices.insert({DNNL_ARG_MEAN, indices_t {input, 2}});
    arg_indices.insert({DNNL_ARG_VARIANCE, indices_t {input, 3}});

    if (op->num_inputs() > 4) {
        arg_indices.insert({DNNL_ARG_SCALE, indices_t {input, 4}});

        if (op->num_inputs() > 5) {
            arg_indices.insert({DNNL_ARG_SHIFT, indices_t {input, 5}});
        } else {
            // use scale mem for fake shift
            arg_indices.insert({DNNL_ARG_SHIFT, indices_t {input, 4}});
        }
    }

    size_t out_index = 0;
    arg_indices.insert({DNNL_ARG_DIFF_SRC, indices_t {output, out_index++}});
    if (op->get_attr<bool>(op_attr::use_affine)) {
        arg_indices.insert(
                {DNNL_ARG_DIFF_SCALE, indices_t {output, out_index++}});
        arg_indices.insert(
                {DNNL_ARG_DIFF_SHIFT, indices_t {output, out_index++}});
    }
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indices_t {output, out_index++}});
    return arg_indices;
}

arg_indices_t reorder_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    arg_indices_t arg_indices;

    size_t index = 0;
    arg_indices.insert({DNNL_ARG_FROM, indices_t {input, index++}});

    const fusion_info_t &fusion_info
            = (op->has_attr(op_attr::fusion_info_key)
                      && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1)
            ? mgr.get_info(op->get_attr<int64_t>(op_attr::fusion_info_key))
            : fusion_info_t();

    if ((op->has_attr(op_attr::with_runtime_scales)
                && op->get_attr<bool>(op_attr::with_runtime_scales))
            || fusion_info.with_runtime_scales(true, 0)) {
        arg_indices.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
                indices_t {input, index++}});
    }

    if (fusion_info.with_runtime_scales(false, 0)) {
        arg_indices.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
                indices_t {input, index++}});
    }

    if ((op->has_attr(op_attr::with_runtime_src_zps)
                && op->get_attr<bool>(op_attr::with_runtime_src_zps))
            || fusion_info.with_runtime_zero_points(true, 0)) {
        arg_indices.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC,
                indices_t {input, index++}});
    }

    get_arg_indices_for_post_ops(op, mgr, arg_indices, index);

    if ((op->has_attr(op_attr::with_runtime_dst_zps)
                && op->get_attr<bool>(op_attr::with_runtime_dst_zps))
            || fusion_info.with_runtime_zero_points(false, 0)) {
        arg_indices.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST,
                indices_t {input, index++}});
    }

    arg_indices.insert({DNNL_ARG_TO, indices_t {output, 0}});
    if (op->num_outputs() > 1) {
        arg_indices.insert({DNNL_ARG_SCRATCHPAD, indices_t {output, 1}});
    }
    return arg_indices;
}

arg_indices_t softmax_bwd_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    arg_indices.insert({DNNL_ARG_DIFF_DST, indices_t {input, 0}});
    arg_indices.insert({DNNL_ARG_DST, indices_t {input, 1}});

    arg_indices.insert({DNNL_ARG_DIFF_SRC, indices_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indices_t {output, 1}});

    return arg_indices;
}

arg_indices_t resampling_bwd_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    arg_indices.insert({DNNL_ARG_DIFF_DST, indices_t {input, 1}});

    arg_indices.insert({DNNL_ARG_DIFF_SRC, indices_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indices_t {output, 1}});

    return arg_indices;
}

arg_indices_t eltwise_bwd_executable_t::get_arg_indices(
        const op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    if (op->get_attr<bool>(op_attr::use_dst)) {
        arg_indices.insert({DNNL_ARG_DST, indices_t {input, 0}});
    } else {
        arg_indices.insert({DNNL_ARG_SRC, indices_t {input, 0}});
    }
    arg_indices.insert({DNNL_ARG_DIFF_DST, indices_t {input, 1}});

    arg_indices.insert({DNNL_ARG_DIFF_SRC, indices_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indices_t {output, 1}});

    return arg_indices;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
