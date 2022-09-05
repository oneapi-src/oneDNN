/*******************************************************************************
 * Copyright 2022 Intel Corporation
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

#include "dnnl.hpp"

#include <utils/utils.hpp>

#include "interface/backend.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/fusion_info.hpp"
#include "backend/dnnl/internal_attrs.hpp"
#include "backend/dnnl/op_executable.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

const indice_t::type_t input = indice_t::type_t::input;
const indice_t::type_t output = indice_t::type_t::output;

conv_fwd_executable_t::desc_t conv_fwd_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<
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
    src = to_nxc_format(src);
    // assume constant weight is for inference scenario
    const auto &wei_lt = op->get_input_value(1)->get_logical_tensor();
    auto pkind = (impl::logical_tensor_wrapper_t(wei_lt).property_type()
                         == property_type::constant)
            ? prop_kind::forward_inference
            : prop_kind::forward_training;
    auto weight = make_dnnl_memory_desc(wei_lt);
    weight = to_format_any(weight);
    size_t dst_offset = 0;
    if (op->get_kind() == op_kind::dnnl_conv_depthwise) {
        // at this stage conv_depthwise op should have 3 outputs: the dst, the
        // scratchpad and the intermediate out
        assertm(op->num_outputs() == 3,
                "conv_depthwise op should have 3 outputs.");
        // we want to take 3nd output as it represent base conv output
        // (needed to create pd)
        dst_offset = 2;
    }
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(dst_offset)->get_logical_tensor());
    dst = to_nxc_format(dst);

    dnnl::convolution_forward::primitive_desc pd;
    if (op->has_attr(op_attr::with_bias)
            && op->get_attr<bool>(op_attr::with_bias)) {
        auto bias = make_dnnl_memory_desc(
                op->get_input_value(2)->get_logical_tensor());
        bias = to_format_any(bias);
        pd = dnnl::convolution_forward::primitive_desc(
                {pkind, algorithm::convolution_direct, src, weight, bias, dst,
                        strides, dilates, pads_begin, pads_end},
                prm_attr, p_engine);
    } else {
        pd = dnnl::convolution_forward::primitive_desc(
                {pkind, algorithm::convolution_direct, src, weight, dst,
                        strides, dilates, pads_begin, pads_end},
                prm_attr, p_engine);
    }

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

deconv_fwd_executable_t::desc_t deconv_fwd_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<
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
        pd = dnnl::deconvolution_forward::primitive_desc(
                {prop_kind::forward_inference, algorithm::deconvolution_direct,
                        src, weight, bias, dst, strides, dilates, pads_begin,
                        pads_end},
                prm_attr, p_engine);
    } else {
        pd = dnnl::deconvolution_forward::primitive_desc(
                {prop_kind::forward_inference, algorithm::deconvolution_direct,
                        src, weight, dst, strides, dilates, pads_begin,
                        pads_end},
                prm_attr, p_engine);
    }

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

deconv_bwd_data_executable_t::desc_t deconv_bwd_data_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<
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

    auto fwd_hints = dnnl::deconvolution_forward::primitive_desc(
            {prop_kind::forward_training, algorithm::deconvolution_direct,
                    diff_src, weight, diff_dst, strides, dilates, pads_begin,
                    pads_end},
            prm_attr, p_engine);

    dnnl::deconvolution_backward_data::primitive_desc pd(
            {dnnl::algorithm::deconvolution_direct, diff_src, weight, diff_dst,
                    strides, pads_begin, pads_end},
            p_engine, fwd_hints);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

deconv_bwd_weights_executable_t::desc_t
deconv_bwd_weights_executable_t::create_desc(std::shared_ptr<impl::op_t> &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<
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

    auto fwd_hints = dnnl::deconvolution_forward::primitive_desc(
            {dnnl::prop_kind::forward_training,
                    dnnl::algorithm::deconvolution_direct, src, diff_weight,
                    diff_dst, strides, dilates, pads_begin, pads_end},
            dnnl::primitive_attr(), p_engine);

    dnnl::deconvolution_backward_weights::primitive_desc pd(
            {dnnl::algorithm::deconvolution_direct, src, diff_weight, diff_dst,
                    strides, dilates, pads_begin, pads_end},
            p_engine, fwd_hints);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

matmul_executable_t::desc_t matmul_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<dnnl::matmul::primitive_desc>(
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
            && ((src.dims().size() == 4
                        && is_format(src, dnnl::memory::format_tag::acbd))
                    || ((src.dims().size() == 2 || src.dims().size() == 3)
                            && p_engine.get_kind() == dnnl::engine::kind::gpu));
    if (!use_strided_src) { src = to_format_any(src); }
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
            && (wei.dims().size() == 4
                    && (is_format(wei, dnnl::memory::format_tag::adbc)
                            || is_format(wei, dnnl::memory::format_tag::abdc)
                            || is_format(wei, dnnl::memory::format_tag::acbd)));
    if (!use_strided_wei) { wei = to_format_any(wei); }
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    if (!((src.dims().size() == 2 || src.dims().size() == 3)
                && p_engine.get_kind() == dnnl::engine::kind::gpu)) {
        dst = to_format_any(dst);
    }

    dnnl::matmul::primitive_desc pd;
    if (op->has_attr(op_attr::with_bias)
            && op->get_attr<bool>(op_attr::with_bias)) {
        auto bias = make_dnnl_memory_desc(
                op->get_input_value(2)->get_logical_tensor());
        bias = to_format_any(bias);
        pd = dnnl::matmul::primitive_desc(
                {src, wei, bias, dst}, prm_attr, p_engine);
    } else {
        pd = dnnl::matmul::primitive_desc({src, wei, dst}, prm_attr, p_engine);
    }

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

pool_executable_t::desc_t pool_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<
                dnnl::pooling_v2_forward::primitive_desc>(
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

    // infer dnnl expilicit pad
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
        dims src_sp = src.dims();
        src_sp.erase(src_sp.begin(), src_sp.begin() + 2);
        dims output_sp = dst.dims();
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
        dilations = get_compatible_dilates(dilations, src.dims().size());
        if (op->num_outputs() == 3) {
            prop = prop_kind::forward_training;
            op->set_attr<bool>(op_attr::is_training, true);
        }
    } else if (op->get_attr<std::string>(op_attr::kind) == "avgpool") {
        const bool exclude_pad = op->get_attr<bool>(op_attr::exclude_pad);
        dilations = dims(src.dims().size(), 0);
        algo = (exclude_pad || adj_pad)
                ? algorithm::pooling_avg_exclude_padding
                : algorithm::pooling_avg_include_padding;
    } else {
        BACKEND_DNNL_ENFORCE(
                0, "Currently only int8 MaxPool/AvgPool is supported.");
    }

    dnnl::pooling_v2_forward::primitive_desc pd(
            {prop, algo, src, dst, strides, kernel, dilations, pads_begin,
                    new_pads_end},
            prm_attr, p_engine);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

pool_bwd_executable_t::desc_t pool_bwd_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<
                dnnl::pooling_v2_backward::primitive_desc>(
                pd_cache.at(op.get()));
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
            : dnnl::memory::desc(diff_src.dims(), diff_src.data_type(),
                    get_ncx_format(diff_src.dims()));

    // infer dnnl explicit pad
    dims new_pads_end(pads_end);
    bool adj_pad = false;
    std::string rounding_type = "floor";
    if (op->has_attr(op_attr::rounding_type)) {
        rounding_type = op->get_attr<std::string>(op_attr::rounding_type);
    }
    if (rounding_type == "ceil") {
        dims src_sp = src.dims();
        src_sp.erase(src_sp.begin(), src_sp.begin() + 2);
        dims output_sp = diff_dst.dims();
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
        dilations = get_compatible_dilates(dilations, src.dims().size());
    } else if (op->get_attr<std::string>(op_attr::kind) == "avgpool") {
        const bool exclude_pad = op->get_attr<bool>(op_attr::exclude_pad);
        algo = (exclude_pad || adj_pad)
                ? algorithm::pooling_avg_exclude_padding
                : algorithm::pooling_avg_include_padding;
    } else {
        BACKEND_DNNL_ENFORCE(0,
                "Currently only MaxPoolBackprop/AvgPoolBackprop is supported.");
    }

    if (op->get_attr<std::string>(op_attr::kind) == "maxpool") {
        diff_dst = to_format_any(diff_dst);
    }

    dnnl::pooling_v2_forward::primitive_desc forward_hints
            = dnnl::pooling_v2_forward::primitive_desc(
                    {prop_kind::forward_training, algo, src, diff_dst, strides,
                            kernel, dilations, pads_begin, new_pads_end},
                    p_engine);

    dnnl::pooling_v2_backward::primitive_desc pd(
            {algo, diff_src, diff_dst, strides, kernel, dilations, pads_begin,
                    new_pads_end},
            p_engine, forward_hints);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

batchnorm_executable_t::desc_t batchnorm_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<
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

    const auto &blk = src.data.format_desc.blocking;
    if (blk.inner_nblks == 1 && blk.inner_idxs[0] == 1
            && blk.inner_blks[0] == 4) {
        // to default format
        src = to_ncx_format(src);
    }

    auto pkind = op->get_attr<bool>(op_attr::is_training)
            ? prop_kind::forward_training
            : prop_kind::forward_inference;

    dnnl::batch_normalization_forward::primitive_desc pd(
            {pkind, src, epsilon, flags}, prm_attr, p_engine);

    pd_cache.insert({op.get(), pd});
    return {pd, false};
}

batchnorm_bwd_executable_t::desc_t batchnorm_bwd_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<
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

    const auto &blk = src.data.format_desc.blocking;
    if (blk.inner_nblks == 1 && blk.inner_idxs[0] == 1
            && blk.inner_blks[0] == 4) {
        // to default format
        src = to_ncx_format(src);
    }

    auto forward_hints = dnnl::batch_normalization_forward::primitive_desc(
            {prop_kind::forward_training, src, epsilon, flags}, p_engine);

    dnnl::batch_normalization_backward::primitive_desc pd(
            {prop_kind::backward, forward_hints.dst_desc(), src, epsilon,
                    flags},
            p_engine, forward_hints);

    pd_cache.insert({op.get(), pd});
    return {pd, false};
}

layernorm_executable_t::desc_t layernorm_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<
                dnnl::layer_normalization_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    float epsilon = 1e-5;
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

    dnnl::layer_normalization_forward::primitive_desc pd(
            {pkind, src, epsilon, flags}, p_engine);

    pd_cache.insert({op.get(), pd});
    return {pd, false};
}

layernorm_bwd_executable_t::desc_t layernorm_bwd_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<
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

    auto data = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    dnnl::layer_normalization_forward::primitive_desc fwd_hints(
            {prop_kind::forward_training, data, epsilon, flags}, p_engine);

    dnnl::layer_normalization_backward::primitive_desc pd(
            {prop_kind::backward, fwd_hints.dst_desc(), data, epsilon, flags},
            prm_attr, p_engine, fwd_hints);

    pd_cache.insert({op.get(), pd});
    return {pd, false};
}

conv_bwd_data_executable_t::desc_t conv_bwd_data_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<
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

    auto diff_dst = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    diff_dst = to_nxc_format(diff_dst);
    auto weight = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    weight = to_format_any(weight);
    auto diff_src = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    diff_src = to_nxc_format(diff_src);

    auto fwd_hints = dnnl::convolution_forward::primitive_desc(
            {dnnl::prop_kind::forward_training,
                    dnnl::algorithm::convolution_direct, diff_src, weight,
                    diff_dst, strides, dilates, pads_begin, pads_end},
            dnnl::primitive_attr(), p_engine);

    dnnl::convolution_backward_data::primitive_desc pd(
            {dnnl::algorithm::convolution_direct, diff_src, weight, diff_dst,
                    strides, dilates, pads_begin, pads_end},
            p_engine, fwd_hints);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

conv_bwd_weights_executable_t::desc_t
conv_bwd_weights_executable_t::create_desc(std::shared_ptr<impl::op_t> &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<
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

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    src = to_nxc_format(src);
    auto diff_dst = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    diff_dst = to_nxc_format(diff_dst);
    auto diff_weight = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    diff_weight = to_format_any(diff_weight);

    auto fwd_hints = dnnl::convolution_forward::primitive_desc(
            {dnnl::prop_kind::forward_training,
                    dnnl::algorithm::convolution_direct, src, diff_weight,
                    diff_dst, strides, dilates, pads_begin, pads_end},
            dnnl::primitive_attr(), p_engine);

    dnnl::convolution_backward_weights::primitive_desc pd(
            {dnnl::algorithm::convolution_direct, src, diff_weight, diff_dst,
                    strides, dilates, pads_begin, pads_end},
            p_engine, fwd_hints);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

eltwise_executable_t::desc_t eltwise_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<dnnl::eltwise_forward::primitive_desc>(
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

    const algorithm algo = static_cast<dnnl::algorithm>(
            op->get_attr<int64_t>(op_attr::alg_kind));
    if (algo == algorithm::undef) {
        BACKEND_DNNL_ENFORCE(0, "Unsupported eltwise op.");
    }

    dnnl::eltwise_forward::primitive_desc pd;
    pd = dnnl::eltwise_forward::primitive_desc(
            {prop_kind::forward, algo, src, alpha, beta}, prm_attr, p_engine);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

eltwise_bwd_executable_t::desc_t eltwise_bwd_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<dnnl::eltwise_backward::primitive_desc>(
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
    dnnl::eltwise_forward::primitive_desc fwd_hints(
            {prop_kind::forward_training, fwd_algo, forward_data, alpha, beta},
            prm_attr, p_engine);

    auto diff_src = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    dnnl::eltwise_backward::primitive_desc pd(
            {bwd_algo, diff_src, forward_data, alpha, beta}, prm_attr, p_engine,
            fwd_hints);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

sum_executable_t::desc_t sum_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<dnnl::sum::primitive_desc>(
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

    dnnl::sum::primitive_desc pd(dst_desc, scales, src_descs, p_engine);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

concat_executable_t::desc_t concat_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {impl::utils::any_cast<dnnl::concat::primitive_desc>(
                        pd_cache.at(op.get())),
                false};
    }

    // Here we force to use plain-in-plain-out (acdb) for 4D case to make
    // sure good performance of DensenNet121 (reducing reorder overhead).
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
        src_mds.emplace_back(memory::desc {tmp_desc.dims(),
                tmp_desc.data_type(), get_forced_format_tag(tmp_desc.dims())});
    }
    const auto tmp_desc = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    auto dst = memory::desc {tmp_desc.dims(), tmp_desc.data_type(),
            get_forced_format_tag(tmp_desc.dims())};

    dnnl::concat::primitive_desc pd(
            dst, static_cast<int>(axis), src_mds, p_engine, prm_attr);
    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

resampling_executable_t::desc_t resampling_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<
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
            {prop_kind::forward_inference, algo, src, dst}, prm_attr, p_engine);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

resampling_bwd_executable_t::desc_t resampling_bwd_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<
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
    } else if (mode == "linear") {
        algo = algorithm::resampling_linear;
    } else {
        BACKEND_DNNL_ENFORCE(0, "Unsupported resampling mode.");
    }

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto diff_dst = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    dnnl::resampling_forward::primitive_desc fwd_hints(
            {prop_kind::forward_training, algo, src, to_format_any(diff_dst)},
            prm_attr, p_engine);

    auto diff_src = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    diff_src = to_format_any(diff_src);
    dnnl::resampling_backward::primitive_desc pd(
            {algo, diff_src, diff_dst}, prm_attr, p_engine, fwd_hints);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

binary_executable_t::desc_t binary_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<dnnl::binary::primitive_desc>(
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
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    dst = to_format_any(dst);

    const algorithm algo = static_cast<dnnl::algorithm>(
            op->get_attr<int64_t>(op_attr::alg_kind));

    dnnl::binary::primitive_desc pd;
    pd = dnnl::binary::primitive_desc(
            {algo, src0, src1, dst}, prm_attr, p_engine);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

prelu_executable_t::desc_t prelu_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<dnnl::prelu_forward::primitive_desc>(
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

    dnnl::prelu_forward::primitive_desc pd(
            {prop_kind::forward, src, wei}, prm_attr, p_engine);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

prelu_bwd_executable_t::desc_t prelu_bwd_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<dnnl::prelu_backward::primitive_desc>(
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

    auto diff_data = make_dnnl_memory_desc(
            op->get_input_value(2)->get_logical_tensor());
    auto diff_wei = make_dnnl_memory_desc(
            op->get_output_value(1)->get_logical_tensor());
    diff_wei = to_format_any(diff_wei);

    auto hint_fwd_pd = dnnl::prelu_forward::primitive_desc(
            {prop_kind::forward, forward_data, wei}, prm_attr, p_engine);

    dnnl::prelu_backward::primitive_desc pd(
            {forward_data, wei, diff_data, diff_wei}, prm_attr, p_engine,
            hint_fwd_pd);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

softmax_executable_t::desc_t softmax_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<
                dnnl::softmax_v2_forward::primitive_desc>(
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
    if (axis < 0) { axis += src.data.ndims; }

    const dnnl::algorithm algo
            = op->get_kind() == dnnl_impl::op_kind::dnnl_logsoftmax
            ? dnnl::algorithm::softmax_log
            : dnnl::algorithm::softmax_accurate;

    dnnl::softmax_v2_forward::primitive_desc pd;
    pd = dnnl::softmax_v2_forward::primitive_desc(
            {prop_kind::forward_inference, algo, src, dst,
                    static_cast<int>(axis)},
            prm_attr, p_engine);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

softmax_bwd_executable_t::desc_t softmax_bwd_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<
                dnnl::softmax_v2_backward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto diff_dst = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    diff_dst = to_format_any(diff_dst);

    auto dst = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());

    auto diff_src = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());

    const auto rank = op->get_output_value(0)->get_logical_tensor().ndims;
    const auto res = utils::try_reverse_axis(
            op->get_attr<int64_t>(op_attr::axis), rank);
    assertm(res.first, "Incorrect axis value.");
    const auto axis = res.second;

    // construct src with layout information from dst and data type information
    // from diff_src.
    dnnl::memory::desc src = dst;
    src.data.data_type = diff_src.data.data_type;

    const dnnl::algorithm algo
            = op->get_kind() == dnnl_impl::op_kind::dnnl_logsoftmax_bwd
            ? dnnl::algorithm::softmax_log
            : dnnl::algorithm::softmax_accurate;

    auto hint_fwd_pd = dnnl::softmax_v2_forward::primitive_desc(
            {prop_kind::forward_training, algo, src, dst,
                    static_cast<int>(axis)},
            prm_attr, p_engine);

    auto pd = dnnl::softmax_v2_backward::primitive_desc(
            {algo, diff_src, diff_dst, dst, static_cast<int>(axis)}, prm_attr,
            p_engine, hint_fwd_pd);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

shuffle_executable_t::desc_t shuffle_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<dnnl::shuffle_forward::primitive_desc>(
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

    auto common = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());

    dnnl::shuffle_forward::primitive_desc pd(
            {prop_kind::forward_inference, common, axis, group}, p_engine,
            prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

reduction_executable_t::desc_t reduction_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<dnnl::reduction::primitive_desc>(
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
            {alg, src, dst, p, eps}, prm_attr, p_engine);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

reorder_executable_t::desc_t reorder_executable_t::create_desc(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = impl::utils::any_cast<dnnl::reorder::primitive_desc>(
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
        prm_attr.set_zero_points(DNNL_ARG_FROM, mask, {DNNL_RUNTIME_S32_VAL});
    } else if (op->has_attr(op_attr::src_zps)) {
        auto zps = op->get_attr<std::vector<int64_t>>(op_attr::src_zps);
        std::vector<int32_t> int32_zps = utils::cast_to_int32(zps);
        prm_attr.set_zero_points(DNNL_ARG_FROM, mask, int32_zps);
    }

    if (op->has_attr(op_attr::with_runtime_scales)
            && op->get_attr<bool>(op_attr::with_runtime_scales)) {
        // runtime scales
        prm_attr.set_output_scales(mask, {DNNL_RUNTIME_F32_VAL});
    } else if (op->has_attr(op_attr::scales)) {
        auto scales = op->get_attr<std::vector<float>>(op_attr::scales);
        prm_attr.set_output_scales(mask, scales);
    }

    if (op->has_attr(op_attr::with_runtime_dst_zps)
            && op->get_attr<bool>(op_attr::with_runtime_dst_zps)) {
        // runtime dst zps
        prm_attr.set_zero_points(DNNL_ARG_TO, mask, {DNNL_RUNTIME_S32_VAL});
    } else if (op->has_attr(op_attr::dst_zps)) {
        auto zps = op->get_attr<std::vector<int64_t>>(op_attr::dst_zps);
        std::vector<int32_t> int32_zps = utils::cast_to_int32(zps);
        prm_attr.set_zero_points(DNNL_ARG_TO, mask, int32_zps);
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

bn_folding_t::desc_t bn_folding_t::create_desc(std::shared_ptr<impl::op_t> &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache) {
    UNUSED(mgr);
    UNUSED(pd_cache);

    desc_t desc;

    desc.epsilon_ = op->get_attr<float>(op_attr::epsilon);
    desc.data_format_ = op->get_attr<std::string>(op_attr::data_format);
    desc.filter_format_ = op->get_attr<std::string>(op_attr::filter_format);
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
    memory::dims epsilon_dims(variance.data.ndims, 1);
    desc.epsilon_desc_ = memory::desc(
            epsilon_dims, memory::data_type::f32, memory::format_tag::a);
    dnnl::binary::desc add_d(
            algorithm::binary_add, variance, desc.epsilon_desc_, variance);

    post_ops add_post_ops;
    // sqrt_variance = sqrt(temp)
    add_post_ops.append_eltwise(1.0f, algorithm::eltwise_sqrt, 0.0f, 0.0f);

    primitive_attr add_attr;
    add_attr.set_post_ops(add_post_ops);
    desc.add_pd_ = dnnl::binary::primitive_desc(add_d, add_attr, p_engine);

    // 2. updated_weight = weights * scale / sqrt_variance

    // expand 1D scale and variance to same ndims with weights
    desc.new_scale_desc_ = expand(scale, weights.data.ndims);
    desc.new_variance_desc_ = expand(variance, weights.data.ndims);

    // after expand, the c channel is on the last dimension, which
    // meet the requirement of NXC format. But for NCX format, we
    // need permute c channel to the second dimension
    if (desc.filter_format_ == "NCX") { // matmul case
        desc.new_scale_desc_ = permute_NXC2NCX(desc.new_scale_desc_);
        desc.new_variance_desc_ = permute_NXC2NCX(desc.new_variance_desc_);
    }

    // after expand, the c channel is on the last dimension, which
    // meet the requirement of XIO format. But for OIX format, we
    // need permute c channel to the first dimension
    if (desc.filter_format_ == "OIX") { // conv case
        desc.new_scale_desc_ = permute_XIO2OIX(desc.new_scale_desc_);
        desc.new_variance_desc_ = permute_XIO2OIX(desc.new_variance_desc_);
    }

    // temp = weights * scale
    dnnl::binary::desc mul_d(
            algorithm::binary_mul, weights, desc.new_scale_desc_, weights);

    post_ops mul_post_ops;
    // updated_weight = temp / sqrt_variance
    mul_post_ops.append_binary(algorithm::binary_div, desc.new_variance_desc_);

    primitive_attr mul_attr;
    mul_attr.set_post_ops(mul_post_ops);
    desc.mul_pd_ = dnnl::binary::primitive_desc(mul_d, mul_attr, p_engine);

    // 3. updated_bias = (bias - mean) * scale / sqrt_variance + shift

    // temp = bias - mean
    memory::desc valid_bias = bias.is_zero() ? mean : bias;
    dnnl::binary::desc sub_d(
            algorithm::binary_sub, valid_bias, mean, valid_bias);

    post_ops sub_post_ops;
    // temp = temp * scale
    sub_post_ops.append_binary(algorithm::binary_mul, scale);
    // temp = temp / sqrt_variance
    sub_post_ops.append_binary(algorithm::binary_div, variance);
    // temp = temp + shift
    sub_post_ops.append_binary(algorithm::binary_add, shift);

    primitive_attr sub_attr;
    sub_attr.set_post_ops(sub_post_ops);
    desc.sub_pd_ = dnnl::binary::primitive_desc(sub_d, sub_attr, p_engine);

    memory::dims scratchpad_dims = variance.dims();
    // sqrt_variance, zero_bias and others (like epslion),
    // or no need to alloc bias
    size_t factor = bias.is_zero() ? 3 : 2;
    scratchpad_dims[0] *= factor;
    desc.scratchpad_desc_ = memory::desc(
            scratchpad_dims, variance.data_type(), memory::format_tag::a);

    return desc;
}

static void get_arg_indices_for_post_ops(const impl::op_t *op,
        fusion_info_mgr_t &mgr, arg_indices_t &indices, size_t &base_indice) {
    const fusion_info_t &fusion_info
            = (op->has_attr(op_attr::fusion_info_key)
                      && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1)
            ? mgr.get_info(op->get_attr<int64_t>(op_attr::fusion_info_key))
            : fusion_info_t();
    const auto &pops = fusion_info.get_post_ops();
    for (int i = 0; i < pops.size(); i++) {
        if (pops[i]->is_post_sum()) {
            indices.insert(
                    {DNNL_GRAPH_ARG_POST_SRC, indice_t {input, base_indice++}});
        } else if (pops[i]->get_op()->get_kind() == op_kind::dnnl_binary) {
            indices.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1,
                    indice_t {input, base_indice++}});
        } else if (pops[i]->get_op()->get_kind() == op_kind::dnnl_convolution) {
            indices.insert({DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS,
                    indice_t {input, base_indice++}});
        } else {
        }
    }
}

static arg_indices_t get_arg_indices_for_conv_and_matmul(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    arg_indices_t arg_indices;

    // add input args
    size_t indice = 0;
    arg_indices.insert({DNNL_ARG_SRC, indice_t {input, indice++}});
    arg_indices.insert({DNNL_ARG_WEIGHTS, indice_t {input, indice++}});
    if (op->has_attr(op_attr::with_bias)
            && op->get_attr<bool>(op_attr::with_bias)) {
        arg_indices.insert({DNNL_ARG_BIAS, indice_t {input, indice++}});
    }

    get_arg_indices_for_post_ops(op, mgr, arg_indices, indice);

    // add output args
    arg_indices.insert({DNNL_ARG_DST, indice_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indice_t {output, 1}});

    return arg_indices;
}

arg_indices_t conv_fwd_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_conv_and_matmul(op, mgr);
}

arg_indices_t deconv_fwd_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_conv_and_matmul(op, mgr);
}

arg_indices_t matmul_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_conv_and_matmul(op, mgr);
}

arg_indices_t binary_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    arg_indices_t arg_indices;

    // add input args
    size_t indice = 0;
    arg_indices.insert({DNNL_ARG_SRC_0, indice_t {input, indice++}});
    arg_indices.insert({DNNL_ARG_SRC_1, indice_t {input, indice++}});

    get_arg_indices_for_post_ops(op, mgr, arg_indices, indice);

    // add output args
    arg_indices.insert({DNNL_ARG_DST, indice_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indice_t {output, 1}});

    return arg_indices;
}

arg_indices_t prelu_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    // add input args
    size_t indice = 0;
    arg_indices.insert({DNNL_ARG_SRC, indice_t {input, indice++}});
    arg_indices.insert({DNNL_ARG_WEIGHTS, indice_t {input, indice++}});

    // add output args
    arg_indices.insert({DNNL_ARG_DST, indice_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indice_t {output, 1}});

    return arg_indices;
}

arg_indices_t prelu_bwd_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    // add input args
    arg_indices.insert({DNNL_ARG_SRC, indice_t {input, 0}});
    arg_indices.insert({DNNL_ARG_WEIGHTS, indice_t {input, 1}});
    arg_indices.insert({DNNL_ARG_DIFF_DST, indice_t {input, 2}});

    // add output args
    arg_indices.insert({DNNL_ARG_DIFF_SRC, indice_t {output, 0}});
    arg_indices.insert({DNNL_ARG_DIFF_WEIGHTS, indice_t {output, 1}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indice_t {output, 2}});

    return arg_indices;
}

arg_indices_t memory_reparser_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;
    arg_indices.insert({DNNL_ARG_FROM, indice_t {input, 0}});
    arg_indices.insert({DNNL_ARG_TO, indice_t {output, 0}});
    return arg_indices;
}

// for single-input-single-output op
static arg_indices_t get_arg_indices_for_siso_op(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    arg_indices_t arg_indices;

    // add input args
    size_t indice = 0;
    arg_indices.insert({DNNL_ARG_FROM, indice_t {input, indice++}});

    get_arg_indices_for_post_ops(op, mgr, arg_indices, indice);

    // add output args
    arg_indices.insert({DNNL_ARG_TO, indice_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indice_t {output, 1}});

    const bool is_training = op->has_attr(op_attr::is_training)
            ? op->get_attr<bool>(op_attr::is_training)
            : false;
    if (is_training) {
        arg_indices.insert({DNNL_ARG_WORKSPACE, indice_t {output, 2}});
    }

    return arg_indices;
}

arg_indices_t pool_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_siso_op(op, mgr);
}

arg_indices_t softmax_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_siso_op(op, mgr);
}

arg_indices_t eltwise_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_siso_op(op, mgr);
}

arg_indices_t shuffle_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_siso_op(op, mgr);
}

arg_indices_t reduction_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_siso_op(op, mgr);
}

arg_indices_t resampling_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_siso_op(op, mgr);
}

arg_indices_t pool_bwd_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    arg_indices_t arg_indices;

    // add input args
    arg_indices.insert({DNNL_ARG_DIFF_DST, indice_t {input, 0}});
    if (op->get_attr<std::string>(op_attr::kind) == "maxpool") {
        // maxpool bwd op must need workspace input
        arg_indices.insert({DNNL_ARG_WORKSPACE, indice_t {input, 1}});
    }

    arg_indices.insert({DNNL_ARG_DIFF_SRC, indice_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indice_t {output, 1}});
    return arg_indices;
}

static arg_indices_t get_arg_indices_for_miso_op(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    for (int i = 0; i < op->num_inputs(); ++i) {
        arg_indices.insert({DNNL_ARG_MULTIPLE_SRC + i,
                indice_t {input, static_cast<size_t>(i)}});
    }

    arg_indices.insert({DNNL_ARG_DST, indice_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indice_t {output, 1}});
    return arg_indices;
}

arg_indices_t concat_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_miso_op(op, mgr);
}

arg_indices_t sum_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    return get_arg_indices_for_miso_op(op, mgr);
}

arg_indices_t bn_folding_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    size_t in_idx = 0;
    arg_indices.insert({DNNL_ARG_WEIGHTS, indice_t {input, in_idx++}});
    if (op->get_attr<bool>(op_attr::with_bias)) {
        arg_indices.insert({DNNL_ARG_BIAS, indice_t {input, in_idx++}});
    }
    arg_indices.insert(
            {DNNL_ARG_WEIGHTS_1, indice_t {input, in_idx++}}); // scale
    arg_indices.insert(
            {DNNL_ARG_WEIGHTS_2, indice_t {input, in_idx++}}); // shift
    arg_indices.insert({DNNL_ARG_MEAN, indice_t {input, in_idx++}}); // mean
    arg_indices.insert(
            {DNNL_ARG_VARIANCE, indice_t {input, in_idx++}}); // variance

    // bind output memory
    size_t out_idx = 0;
    arg_indices.insert(
            {DNNL_ARG_DST_0, indice_t {output, out_idx++}}); // updated weight
    arg_indices.insert(
            {DNNL_ARG_DST_1, indice_t {output, out_idx++}}); // updated bias
    arg_indices.insert(
            {DNNL_ARG_SCRATCHPAD, indice_t {output, out_idx++}}); // scratchpad

    return arg_indices;
}

arg_indices_t conv_bwd_data_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    arg_indices.insert({DNNL_ARG_DIFF_DST, indice_t {input, 0}});
    arg_indices.insert({DNNL_ARG_WEIGHTS, indice_t {input, 1}});

    arg_indices.insert({DNNL_ARG_DIFF_SRC, indice_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indice_t {output, 1}});

    return arg_indices;
}

arg_indices_t deconv_bwd_data_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    return conv_bwd_data_executable_t::get_arg_indices(op, mgr);
}

arg_indices_t conv_bwd_weights_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    arg_indices.insert({DNNL_ARG_SRC, indice_t {input, 0}});
    arg_indices.insert({DNNL_ARG_DIFF_DST, indice_t {input, 1}});

    arg_indices.insert({DNNL_ARG_DIFF_WEIGHTS, indice_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indice_t {output, 1}});

    return arg_indices;
}

arg_indices_t deconv_bwd_weights_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    return conv_bwd_weights_executable_t::get_arg_indices(op, mgr);
}

arg_indices_t batchnorm_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    size_t in_indice = 0;
    arg_indices.insert({DNNL_ARG_SRC, indice_t {input, in_indice++}});
    if (!op->get_attr<bool>(op_attr::is_training)) { // inference
        arg_indices.insert({DNNL_ARG_SCALE, indice_t {input, in_indice++}});
        arg_indices.insert({DNNL_ARG_SHIFT, indice_t {input, in_indice++}});
        arg_indices.insert({DNNL_ARG_MEAN, indice_t {input, in_indice++}});
        arg_indices.insert({DNNL_ARG_VARIANCE, indice_t {input, in_indice++}});
    } else { // training
        // running_mean/running_variance of last iteration
        arg_indices.insert({DNNL_ARG_SRC_1, indice_t {input, in_indice++}});
        arg_indices.insert({DNNL_ARG_SRC_2, indice_t {input, in_indice++}});

        if (op->num_inputs() > 3) {
            arg_indices.insert({DNNL_ARG_SCALE, indice_t {input, in_indice++}});
            arg_indices.insert({DNNL_ARG_SHIFT, indice_t {input, in_indice++}});
        }
    }

    size_t out_indice = 0;
    arg_indices.insert({DNNL_ARG_DST, indice_t {output, out_indice++}});
    if (op->get_attr<bool>(op_attr::is_training)) {
        // running_mean
        arg_indices.insert({DNNL_ARG_DST_1, indice_t {output, out_indice++}});
        // running_variance
        arg_indices.insert({DNNL_ARG_DST_2, indice_t {output, out_indice++}});
        // batch_meam
        arg_indices.insert({DNNL_ARG_MEAN, indice_t {output, out_indice++}});
        // batch_variance
        arg_indices.insert(
                {DNNL_ARG_VARIANCE, indice_t {output, out_indice++}});
    }

    if (op->num_outputs() > out_indice) {
        arg_indices.insert(
                {DNNL_ARG_SCRATCHPAD, indice_t {output, out_indice++}});
    }

    // workspace (for BatchNormForwardTraining with ReLU)
    if (op->num_outputs() > out_indice) {
        arg_indices.insert(
                {DNNL_ARG_WORKSPACE, indice_t {output, out_indice++}});
    }

    return arg_indices;
}

arg_indices_t batchnorm_bwd_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    arg_indices_t arg_indices;
    size_t indice = 0;

    arg_indices.insert({DNNL_ARG_SRC, indice_t {input, indice++}});
    arg_indices.insert({DNNL_ARG_DIFF_DST, indice_t {input, indice++}});

    arg_indices.insert({DNNL_ARG_MEAN, indice_t {input, indice++}});
    arg_indices.insert({DNNL_ARG_VARIANCE, indice_t {input, indice++}});

    if (op->num_outputs() > 2) {
        // DNNL_ARG_SCALE and DNNL_ARG_SHIFT use the same memory
        arg_indices.insert({DNNL_ARG_SCALE, indice_t {input, indice}});
        arg_indices.insert({DNNL_ARG_SHIFT, indice_t {input, indice++}});
    }

    indice = 0;
    arg_indices.insert({DNNL_ARG_DIFF_SRC, indice_t {output, indice++}});
    // check if has diff_scale and diff_shift outputs
    if (op->num_outputs() > 2) {
        arg_indices.insert({DNNL_ARG_DIFF_SCALE, indice_t {output, indice++}});
        arg_indices.insert({DNNL_ARG_DIFF_SHIFT, indice_t {output, indice++}});
    }
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indice_t {output, indice++}});

    return arg_indices;
}

arg_indices_t layernorm_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    size_t in_indice = 0;
    arg_indices.insert({DNNL_ARG_SRC, indice_t {input, in_indice++}});
    if (!op->has_attr(op_attr::use_affine)
            || op->get_attr<bool>(op_attr::use_affine)) {
        arg_indices.insert({DNNL_ARG_SCALE, indice_t {input, in_indice++}});
        arg_indices.insert({DNNL_ARG_SHIFT, indice_t {input, in_indice++}});
    }

    size_t out_indice = 0;
    arg_indices.insert({DNNL_ARG_DST, indice_t {output, out_indice++}});
    if (!op->has_attr(op_attr::keep_stats)
            || op->get_attr<bool>(op_attr::keep_stats)) {
        arg_indices.insert({DNNL_ARG_MEAN, indice_t {output, out_indice++}});
        arg_indices.insert(
                {DNNL_ARG_VARIANCE, indice_t {output, out_indice++}});
    }

    if (op->num_outputs() > out_indice) {
        arg_indices.insert(
                {DNNL_ARG_SCRATCHPAD, indice_t {output, out_indice++}});
    }

    return arg_indices;
}

arg_indices_t layernorm_bwd_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    arg_indices_t arg_indices;

    arg_indices.insert({DNNL_ARG_SRC, indice_t {input, 0}});
    arg_indices.insert({DNNL_ARG_DIFF_DST, indice_t {input, 1}});
    arg_indices.insert({DNNL_ARG_MEAN, indice_t {input, 2}});
    arg_indices.insert({DNNL_ARG_VARIANCE, indice_t {input, 3}});

    if (op->num_inputs() > 4) {
        arg_indices.insert({DNNL_ARG_SCALE, indice_t {input, 4}});

        if (op->num_inputs() > 5) {
            arg_indices.insert({DNNL_ARG_SHIFT, indice_t {input, 5}});
        } else {
            // use scale mem for fake shift
            arg_indices.insert({DNNL_ARG_SHIFT, indice_t {input, 4}});
        }
    }

    size_t out_indice = 0;
    arg_indices.insert({DNNL_ARG_DIFF_SRC, indice_t {output, out_indice++}});
    if (op->get_attr<bool>(op_attr::use_affine)) {
        arg_indices.insert(
                {DNNL_ARG_DIFF_SCALE, indice_t {output, out_indice++}});
        arg_indices.insert(
                {DNNL_ARG_DIFF_SHIFT, indice_t {output, out_indice++}});
    }
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indice_t {output, out_indice++}});
    return arg_indices;
}

arg_indices_t reorder_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    arg_indices_t arg_indices;

    size_t indice = 0;
    arg_indices.insert({DNNL_ARG_FROM, indice_t {input, indice++}});

    // we always insert the input belonging to input fusion before the input
    // belonging to output fusion. So, src_zps must be before scales if it
    // exists
    if (op->has_attr(op_attr::with_runtime_src_zps)
            && op->get_attr<bool>(op_attr::with_runtime_src_zps)) {
        arg_indices.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC,
                indice_t {input, indice}});
        auto src_zps = op->get_input_value(indice++);
        assertm(src_zps->get_logical_tensor().data_type == impl::data_type::s32,
                "oneDNN runtime zps must be s32 type");
    }

    if (op->has_attr(op_attr::with_runtime_scales)
            && op->get_attr<bool>(op_attr::with_runtime_scales)) {
        arg_indices.insert(
                {DNNL_ARG_ATTR_OUTPUT_SCALES, indice_t {input, indice++}});
    }

    if (op->has_attr(op_attr::with_runtime_dst_zps)
            && op->get_attr<bool>(op_attr::with_runtime_dst_zps)) {
        arg_indices.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST,
                indice_t {input, indice}});
        auto dst_zps = op->get_input_value(indice++);
        assertm(dst_zps->get_logical_tensor().data_type == impl::data_type::s32,
                "oneDNN runtime zps must be s32 type");
    }

    get_arg_indices_for_post_ops(op, mgr, arg_indices, indice);

    arg_indices.insert({DNNL_ARG_TO, indice_t {output, 0}});
    if (op->num_outputs() > 1) {
        arg_indices.insert({DNNL_ARG_SCRATCHPAD, indice_t {output, 1}});
    }
    return arg_indices;
}

arg_indices_t softmax_bwd_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    arg_indices.insert({DNNL_ARG_DIFF_DST, indice_t {input, 0}});
    arg_indices.insert({DNNL_ARG_DST, indice_t {input, 1}});

    arg_indices.insert({DNNL_ARG_DIFF_SRC, indice_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indice_t {output, 1}});

    return arg_indices;
}

arg_indices_t resampling_bwd_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    arg_indices.insert({DNNL_ARG_DIFF_DST, indice_t {input, 1}});

    arg_indices.insert({DNNL_ARG_DIFF_SRC, indice_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indice_t {output, 1}});

    return arg_indices;
}

arg_indices_t eltwise_bwd_executable_t::get_arg_indices(
        const impl::op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    arg_indices_t arg_indices;

    if (op->get_attr<bool>(op_attr::use_dst)) {
        arg_indices.insert({DNNL_ARG_DST, indice_t {input, 0}});
    } else {
        arg_indices.insert({DNNL_ARG_SRC, indice_t {input, 0}});
    }
    arg_indices.insert({DNNL_ARG_DIFF_DST, indice_t {input, 1}});

    arg_indices.insert({DNNL_ARG_DIFF_SRC, indice_t {output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, indice_t {output, 1}});

    return arg_indices;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
