/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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
#ifndef GRAPH_BACKEND_DNNL_PASSES_OP_EXECUTABLE_HPP
#define GRAPH_BACKEND_DNNL_PASSES_OP_EXECUTABLE_HPP

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include <CL/sycl.hpp>
#endif

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/internal_attrs.hpp"
#include "graph/backend/dnnl/passes/fusion_info.hpp"
#include "graph/backend/dnnl/passes/lower_down.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"

#include "graph/utils/utils.hpp"

#define DNNL_GRAPH_ARG_POST_SRC (-1)

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

// return arg: std::pair<dnnl::convolution_forward::primitive_desc, bool>
//      -> {pd, the flag indicating if this is the first time to create}
inline std::pair<dnnl::convolution_forward::primitive_desc, bool>
create_conv_pd(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<dnnl::convolution_forward::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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
    auto pkind = (logical_tensor_wrapper_t(wei_lt).property_type()
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

    return {pd, true};
}

inline std::pair<dnnl::deconvolution_forward::primitive_desc, bool>
create_deconv_pd(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<dnnl::deconvolution_forward::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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

    return {pd, true};
}

inline std::pair<dnnl::deconvolution_backward_data::primitive_desc, bool>
create_deconv_bwd_data_pd(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<
                        dnnl::deconvolution_backward_data::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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

    return {pd, true};
}

inline std::pair<dnnl::deconvolution_backward_weights::primitive_desc, bool>
create_deconv_bwd_weights_pd(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<
                        dnnl::deconvolution_backward_weights::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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

    return {pd, true};
}

inline std::pair<dnnl::matmul::primitive_desc, bool> create_matmul_pd(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<dnnl::matmul::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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
    // create primitive desc with strided activation when:
    // 1) activation has 4 dimensions and layout is acbd since oneDNN has
    //    optimized kernel
    // 2) activation has 2/3 dimensions and device kind is gpu for avoiding
    //    blocked activation. This can reduce the cost for the reorder between
    //    plain and block layout, especially for users who compile partition
    //    with plain layout. The performance of strided primitive on GPU will be
    //    optimized by oneDNN.
    const bool use_strided_src
            = (src.dims().size() == 4
                      && is_format(src, dnnl::memory::format_tag::acbd))
            || ((src.dims().size() == 2 || src.dims().size() == 3)
                    && p_engine.get_kind() == dnnl::engine::kind::gpu);
    if (!use_strided_src) { src = to_format_any(src); }
    auto wei = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    if (!(wei.dims().size() == 4
                && is_format(wei, dnnl::memory::format_tag::adbc))) {
        wei = to_format_any(wei);
    }
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

    return {pd, true};
}

inline std::pair<dnnl::pooling_v2_forward::primitive_desc, bool> create_pool_pd(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<dnnl::pooling_v2_forward::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
    }

    dims strides = op->get_attr<dims>(op_attr::strides);
    dims kernel = op->get_attr<dims>(op_attr::kernel);
    dims pads_begin = op->get_attr<dims>(op_attr::pads_begin);
    dims pads_end = op->get_attr<dims>(op_attr::pads_end);
    dims dilations(strides.size(), 0);
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
    if (rounding_type == "ceil") {
        dims src_sp = src.dims();
        src_sp.erase(src_sp.begin(), src_sp.begin() + 2);
        dims output_sp = dst.dims();
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

    return {pd, true};
}

inline std::pair<dnnl::pooling_v2_backward::primitive_desc, bool>
create_pool_bwd_pd(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<dnnl::pooling_v2_backward::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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

    return {pd, true};
}

inline std::pair<dnnl::batch_normalization_forward::primitive_desc, bool>
create_batchnorm_pd(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<
                        dnnl::batch_normalization_forward::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
    }

    float epsilon = op->get_attr<float>(op_attr::epsilon);

    auto flags = normalization_flag::none;
    // for inference
    if (!op->get_attr<bool>(op_attr::is_training)) {
        flags |= normalization_flag::use_global_stats;
        flags |= normalization_flag::use_scale;
        flags |= normalization_flag::use_shift;
    } else {
        // for training, inputs: [src, gamma, beta, mean, variance]
        if (op->num_inputs() > 3) {
            flags |= normalization_flag::use_scale;
            flags |= normalization_flag::use_shift;
        }

        if (op->has_attr(op_attr::fuse_relu)
                && op->get_attr<bool>(op_attr::fuse_relu))
            flags |= normalization_flag::fuse_norm_relu;
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
    return {pd, true};
}

inline std::pair<dnnl::batch_normalization_backward::primitive_desc, bool>
create_batchnorm_bwd_pd(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<
                        dnnl::batch_normalization_backward::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
    }

    float epsilon = op->get_attr<float>(op_attr::epsilon);

    auto flags = normalization_flag::use_scale | normalization_flag::use_shift;

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
    return {pd, true};
}

inline std::pair<dnnl::layer_normalization_forward::primitive_desc, bool>
create_layernorm_pd(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<
                        dnnl::layer_normalization_forward::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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

    auto flags = normalization_flag::none;
    if (use_affine)
        flags |= (normalization_flag::use_scale
                | normalization_flag::use_shift);

    prop_kind pkind = keep_stats ? prop_kind::forward_training
                                 : prop_kind::forward_inference;

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());

    dnnl::layer_normalization_forward::primitive_desc pd(
            {pkind, src, epsilon, flags}, p_engine);

    pd_cache.insert({op.get(), pd});
    return {pd, true};
}

inline std::pair<dnnl::layer_normalization_backward::primitive_desc, bool>
create_layernorm_bwd_pd(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<
                        dnnl::layer_normalization_backward::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        prm_attr = make_dnnl_primitive_attr(op, mgr.get_info(key));
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto epsilon = op->get_attr<float>(op_attr::epsilon);
    auto flags = normalization_flag::none;
    const bool use_affine = op->get_attr<bool>(op_attr::use_affine);
    if (use_affine) {
        flags |= normalization_flag::use_scale;
        flags |= normalization_flag::use_shift;
    }

    auto data = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    dnnl::layer_normalization_forward::primitive_desc fwd_hints(
            {prop_kind::forward_training, data, epsilon, flags}, p_engine);

    dnnl::layer_normalization_backward::primitive_desc pd(
            {prop_kind::backward, fwd_hints.dst_desc(), data, epsilon, flags},
            prm_attr, p_engine, fwd_hints);

    pd_cache.insert({op.get(), pd});
    return {pd, true};
}

inline std::pair<dnnl::convolution_backward_data::primitive_desc, bool>
create_conv_bwd_data_pd(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<dnnl::convolution_backward_data::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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

    return {pd, true};
}

inline std::pair<dnnl::convolution_backward_weights::primitive_desc, bool>
create_conv_bwd_weights_pd(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<
                        dnnl::convolution_backward_weights::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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

    return {pd, true};
}

inline std::pair<dnnl::eltwise_forward::primitive_desc, bool> create_eltwise_pd(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<dnnl::eltwise_forward::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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

    return {pd, true};
}

inline std::pair<dnnl::eltwise_backward::primitive_desc, bool>
create_eltwise_bwd_pd(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<dnnl::eltwise_backward::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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

    return {pd, true};
}

inline dnnl::sum::primitive_desc create_dnnl_sum_pd(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
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

    return pd;
}

inline dnnl::concat::primitive_desc create_concat_pd(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
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

    return pd;
}

inline std::pair<dnnl::resampling_forward::primitive_desc, bool>
create_resampling_pd(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<dnnl::resampling_forward::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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

    return {pd, true};
}

inline std::pair<dnnl::resampling_backward::primitive_desc, bool>
create_resampling_bwd_pd(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<dnnl::resampling_backward::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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

    return {pd, true};
}

inline std::pair<dnnl::binary::primitive_desc, bool> create_binary_pd(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<dnnl::binary::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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

    return {pd, true};
}

inline std::pair<dnnl::prelu_forward::primitive_desc, bool> create_prelu_pd(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<dnnl::prelu_forward::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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

    return {pd, true};
}

inline std::pair<dnnl::prelu_backward::primitive_desc, bool>
create_prelu_bwd_pd(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<dnnl::prelu_backward::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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

    return {pd, true};
}

inline std::pair<dnnl::softmax_v2_forward::primitive_desc, bool>
create_softmax_pd(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache, dnnl::algorithm algo) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<dnnl::softmax_v2_forward::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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

    dnnl::softmax_v2_forward::primitive_desc pd;
    pd = dnnl::softmax_v2_forward::primitive_desc(
            {prop_kind::forward_inference, algo, src, dst,
                    static_cast<int>(axis)},
            prm_attr, p_engine);

    pd_cache.insert({op.get(), pd});

    return {pd, true};
}

inline std::pair<dnnl::softmax_v2_backward::primitive_desc, bool>
create_softmax_bwd_pd(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache, dnnl::algorithm algo) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<dnnl::softmax_v2_backward::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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

    auto hint_fwd_pd = dnnl::softmax_v2_forward::primitive_desc(
            {prop_kind::forward_training, algo, src, dst,
                    static_cast<int>(axis)},
            prm_attr, p_engine);

    auto pd = dnnl::softmax_v2_backward::primitive_desc(
            {algo, diff_src, diff_dst, dst, static_cast<int>(axis)}, prm_attr,
            p_engine, hint_fwd_pd);

    pd_cache.insert({op.get(), pd});

    return {pd, true};
}

inline std::pair<dnnl::shuffle_forward::primitive_desc, bool> create_shuffle_pd(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<dnnl::shuffle_forward::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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

    return {pd, true};
}

inline std::pair<dnnl::reduction::primitive_desc, bool> create_reduction_pd(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {static_cast<dnnl::reduction::primitive_desc &>(
                        pd_cache.at(op.get())),
                false};
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

    return {pd, true};
}

inline dnnl::reorder::primitive_desc create_reorder_pd(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr) {
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
        std::vector<int32_t> neg_zps = dnnl_impl::utils::fmap(
                zps, [](int64_t zp) { return static_cast<int32_t>(-zp); });
        prm_attr.set_zero_points(DNNL_ARG_FROM, mask, neg_zps);
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
        std::vector<int32_t> int32_zps = dnnl_impl::utils::fmap(
                zps, [](int64_t zp) { return static_cast<int32_t>(zp); });
        prm_attr.set_zero_points(DNNL_ARG_TO, mask, int32_zps);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto in_md = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto out_md = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());

    auto pd = dnnl::reorder::primitive_desc(
            p_engine, in_md, p_engine, out_md, prm_attr);
    return pd;
}

struct op_executable_t {
    virtual ~op_executable_t() = default;
    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const = 0;
#ifdef DNNL_WITH_SYCL
    virtual ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const = 0;
#endif
};

struct memory_reparser_t : public op_executable_t {
    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        UNUSED(stream);
        assertm(args.find(DNNL_ARG_FROM)->second.get_data_handle()
                        == args.find(DNNL_ARG_TO)->second.get_data_handle(),
                "memory_parser must be inplaced");
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        UNUSED(stream);
        assertm(args.find(DNNL_ARG_FROM)->second.get_data_handle()
                        == args.find(DNNL_ARG_TO)->second.get_data_handle(),
                "memory_parser must be inplaced");

        // Fast path: if only one event, return it.
        if (deps.size() == 1) return deps[0];

        // Otherwise, we run a trivial kernel to gather all deps. The
        // dummy task is needed to not get an error related to empty
        // kernel.
        auto q = dnnl::sycl_interop::get_queue(stream);
        auto e = q.submit([&](::sycl::handler &cgh) {
            cgh.depends_on(deps);
            cgh.single_task<class dnnl_graph_dummy_kernel>([]() {});
        });
        return e;
    }
#endif
};

template <typename attr_dt, typename target_dt>
struct const_memory_filler_t : public op_executable_t {
    const_memory_filler_t(std::shared_ptr<op_t> &op, op_attr_t attr_name) {
        attr_data_
                = get_attr_data(op->get_attr<std::vector<attr_dt>>(attr_name),
                        std::is_same<attr_dt, target_dt>());
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        void *data_handle = static_cast<void *>(
                const_cast<target_dt *>(attr_data_.data()));
        const memory &dst_mem = args.find(DNNL_ARG_TO)->second;

        auto is_cpu = dst_mem.get_engine().get_kind() == engine::kind::cpu;
        // handle cross-engine case
        auto src_eng = (is_cpu) ? dst_mem.get_engine()
                                : engine(dflt_eng_kind, dflt_eng_idx);

        const memory src_mem
                = make_dnnl_memory(dst_mem.get_desc(), src_eng, data_handle);
        dnnl::reorder(src_mem, dst_mem)
                .execute(stream, const_cast<memory &>(src_mem),
                        const_cast<memory &>(dst_mem));
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        void *data_handle = static_cast<void *>(
                const_cast<target_dt *>(attr_data_.data()));
        const memory &dst_mem = args.find(DNNL_ARG_TO)->second;
        auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
        auto e = sycl_queue
                .memcpy(dst_mem.get_data_handle(), data_handle,
                        dst_mem.get_desc().get_size());
        return e;
    }
#endif

private:
    std::vector<target_dt> get_attr_data(
            const std::vector<attr_dt> &orig_data, std::true_type) {
        return orig_data;
    }
    std::vector<target_dt> get_attr_data(
            const std::vector<attr_dt> &orig_data, std::false_type) {
        return std::vector<target_dt>(orig_data.begin(), orig_data.end());
    }

    const engine::kind dflt_eng_kind = engine::kind::cpu;
    const size_t dflt_eng_idx = 0;
    std::vector<target_dt> attr_data_;
};

using fvec_to_fvec_filler = const_memory_filler_t<float, float>;
using i64vec_to_i32vec_filler = const_memory_filler_t<int64_t, int32_t>;

struct conv_fwd_executable_t : public op_executable_t {
    conv_fwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_conv_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::convolution_forward(pd_);
        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                // psrc_mem and dst_mem may have different data type bug same
                // buffer size(u8 and s8) for such case, need to reorder
                // psrc_mem to dst_mem with original data type
                if (psrc_mem.get_desc().data_type()
                                == dnnl::memory::data_type::s8
                        && dst_mem.get_desc().data_type()
                                == dnnl::memory::data_type::u8) {
                    dnnl::memory::desc to_desc = dst_mem.get_desc();
                    to_desc.data.data_type = psrc_mem.get_desc().data.data_type;
                    const memory to_mem
                            = dnnl::memory(to_desc, psrc_mem.get_engine());
                    to_mem.set_data_handle(dst_mem.get_data_handle());
                    dnnl::reorder(psrc_mem, to_mem)
                            .execute(stream, const_cast<memory &>(psrc_mem),
                                    const_cast<memory &>(to_mem));
                } else {
                    dnnl::reorder(psrc_mem, dst_mem)
                            .execute(stream, const_cast<memory &>(psrc_mem),
                                    const_cast<memory &>(dst_mem));
                }
            }
        }

        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto sycl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                // psrc_mem and dst_mem may have different data type bug same
                // buffer size(u8 and s8) for such case, need to reorder
                // psrc_mem to dst_mem with original data type
                if (psrc_mem.get_desc().data_type()
                                == dnnl::memory::data_type::s8
                        && dst_mem.get_desc().data_type()
                                == dnnl::memory::data_type::u8) {
                    dnnl::memory::desc to_desc = dst_mem.get_desc();
                    to_desc.data.data_type = psrc_mem.get_desc().data.data_type;
                    const memory to_mem
                            = dnnl::memory(to_desc, psrc_mem.get_engine());
                    to_mem.set_data_handle(dst_mem.get_data_handle());
                    auto prim = dnnl::reorder(psrc_mem, to_mem);
                    auto e = dnnl::sycl_interop::execute(prim, stream,
                            {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                    {DNNL_ARG_TO,
                                            const_cast<memory &>(to_mem)}},
                            sycl_deps);
                    sycl_deps = {e};
                    if (stream.get_engine().get_kind() == engine::kind::cpu)
                        e.wait();
                } else {
                    auto prim = dnnl::reorder(psrc_mem, dst_mem);
                    auto e = dnnl::sycl_interop::execute(prim, stream,
                            {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                    {DNNL_ARG_TO,
                                            const_cast<memory &>(dst_mem)}},
                            sycl_deps);
                    sycl_deps = {e};
                }
            }
        }
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::convolution_forward::primitive_desc pd_;
    dnnl::convolution_forward prim_;
    bool with_sum_ {false};
};

struct deconv_fwd_executable_t : public op_executable_t {
    deconv_fwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_deconv_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::deconvolution_forward(pd_);
        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, const_cast<memory &>(psrc_mem),
                                const_cast<memory &>(dst_mem));
            }
        }

        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto sycl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::sycl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        sycl_deps);
                sycl_deps = {e};
            }
        }
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::deconvolution_forward::primitive_desc pd_;
    dnnl::deconvolution_forward prim_;
    bool with_sum_ {false};
};

struct deconv_bwd_data_executable_t : public op_executable_t {
    deconv_bwd_data_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_deconv_bwd_data_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::deconvolution_backward_data(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::deconvolution_backward_data::primitive_desc pd_;
    dnnl::deconvolution_backward_data prim_;
};

struct deconv_bwd_weights_executable_t : public op_executable_t {
    deconv_bwd_weights_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_deconv_bwd_weights_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::deconvolution_backward_weights(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::deconvolution_backward_weights::primitive_desc pd_;
    dnnl::deconvolution_backward_weights prim_;
};

struct matmul_executable_t : public op_executable_t {
    matmul_executable_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
        pd_ = create_matmul_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::matmul(pd_);

        // The scratchpad size of pd created by using any format tag may be
        // different from the scratchpad size of pd created by using queried
        // optimal format tag
        dnnl::memory::desc stored = make_dnnl_memory_desc(
                op->get_output_value(1)->get_logical_tensor());
        dnnl::memory::desc real = pd_.scratchpad_desc();
        if (stored != real) {
            auto scratchpad_val = op->get_output_value(1);
            scratchpad_val->set_layout_type(layout_type::any);
            fill_layout_info(scratchpad_val, real);
        }

        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (with_sum_) {
            memory &dst_mem
                    = const_cast<memory &>(args.find(DNNL_ARG_DST)->second);
            memory &psrc_mem = const_cast<memory &>(
                    args.find(DNNL_GRAPH_ARG_POST_SRC)->second);

            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, psrc_mem, dst_mem);
            }
        }
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto sycl_deps = deps;
        if (with_sum_) {
            memory &dst_mem
                    = const_cast<memory &>(args.find(DNNL_ARG_DST)->second);
            memory &psrc_mem = const_cast<memory &>(
                    args.find(DNNL_GRAPH_ARG_POST_SRC)->second);

            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::sycl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        sycl_deps);
                sycl_deps = {e};
            }
        }
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::matmul::primitive_desc pd_;
    dnnl::matmul prim_;
    bool with_sum_ {false};
};

struct eltwise_executable_t : public op_executable_t {
    eltwise_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_eltwise_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::eltwise_forward(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::eltwise_forward::primitive_desc pd_;
    dnnl::eltwise_forward prim_;
};

struct eltwise_bwd_executable_t : public op_executable_t {
    eltwise_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_eltwise_bwd_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::eltwise_backward(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::eltwise_backward::primitive_desc pd_;
    dnnl::eltwise_backward prim_;
};

struct binary_executable_t : public op_executable_t {
    binary_executable_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
        pd_ = create_binary_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::binary(pd_);

        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (with_sum_) {
            memory &dst_mem
                    = const_cast<memory &>(args.find(DNNL_ARG_DST)->second);
            memory &psrc_mem = const_cast<memory &>(
                    args.find(DNNL_GRAPH_ARG_POST_SRC)->second);

            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, psrc_mem, dst_mem);
            }
        }

        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto sycl_deps = deps;
        if (with_sum_) {
            memory &dst_mem
                    = const_cast<memory &>(args.find(DNNL_ARG_DST)->second);
            memory &psrc_mem = const_cast<memory &>(
                    args.find(DNNL_GRAPH_ARG_POST_SRC)->second);

            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::sycl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        sycl_deps);
                sycl_deps = {e};
            }
        }
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::binary::primitive_desc pd_;
    dnnl::binary prim_;
    bool with_sum_ {false};
};

struct concat_executable_t : public op_executable_t {
    concat_executable_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            fusion_info_mgr_t &mgr) {
        pd_ = create_concat_pd(op, p_engine, mgr);
        prim_ = dnnl::concat(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::concat::primitive_desc pd_;
    dnnl::concat prim_;
};

struct shuffle_executable_t : public op_executable_t {
    shuffle_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_shuffle_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::shuffle_forward(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::shuffle_forward::primitive_desc pd_;
    dnnl::shuffle_forward prim_;
};

struct pool_executable_t : public op_executable_t {
    pool_executable_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
        pd_ = create_pool_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::pooling_v2_forward(pd_);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::pooling_v2_forward::primitive_desc pd_;
    dnnl::pooling_v2_forward prim_;
};

struct pool_bwd_executable_t : public op_executable_t {
    pool_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_pool_bwd_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::pooling_v2_backward(pd_);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::pooling_v2_backward::primitive_desc pd_;
    dnnl::pooling_v2_backward prim_;
};

struct prelu_executable_t : public op_executable_t {
    prelu_executable_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
        pd_ = create_prelu_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::prelu_forward(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::prelu_forward::primitive_desc pd_;
    dnnl::prelu_forward prim_;
};

struct prelu_bwd_executable_t : public op_executable_t {
    prelu_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_prelu_bwd_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::prelu_backward(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::prelu_backward::primitive_desc pd_;
    dnnl::prelu_backward prim_;
};

struct reorder_executable_t : public op_executable_t {
    reorder_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
        pd_ = create_reorder_pd(op, p_engine, mgr);
        prim_ = dnnl::reorder(pd_);
        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, const_cast<memory &>(psrc_mem),
                                const_cast<memory &>(dst_mem));
            }
        }
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto sycl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::sycl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        sycl_deps);
                sycl_deps = {e};
            }
        }
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::reorder::primitive_desc pd_;
    dnnl::reorder prim_;
    bool with_sum_ {false};
};

struct bn_folding_t : public op_executable_t {
    bn_folding_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine) {
        epsilon_ = op->get_attr<float>(op_attr::epsilon);
        data_format_ = op->get_attr<std::string>(op_attr::data_format);
        filter_format_ = op->get_attr<std::string>(op_attr::filter_format);
        with_bias_ = op->get_attr<bool>(op_attr::with_bias);

        size_t in_idx = 0;
        auto weights = make_dnnl_memory_desc(
                op->get_input_value(in_idx++)->get_logical_tensor());
        auto bias = with_bias_ ? make_dnnl_memory_desc(
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
        epsilon_desc_ = memory::desc(
                epsilon_dims, memory::data_type::f32, memory::format_tag::a);
        dnnl::binary::desc add_d(
                algorithm::binary_add, variance, epsilon_desc_, variance);

        post_ops add_post_ops;
        // sqrt_variance = sqrt(temp)
        add_post_ops.append_eltwise(1.0f, algorithm::eltwise_sqrt, 0.0f, 0.0f);

        primitive_attr add_attr;
        add_attr.set_post_ops(add_post_ops);
        dnnl::binary::primitive_desc add_pd(add_d, add_attr, p_engine);
        add_prim_ = dnnl::binary(add_pd);

        // 2. updated_weight = weights * scale / sqrt_variance

        // expand 1D scale and variance to same ndims with weights
        new_scale_desc_ = expand(scale, weights.data.ndims);
        new_variance_desc_ = expand(variance, weights.data.ndims);

        // after expand, the c channel is on the last dimension, which
        // meet the requirement of NXC format. But for NCX format, we
        // need permute c channel to the second dimension
        if (filter_format_ == "NCX") { // matmul case
            new_scale_desc_ = permute_NXC2NCX(new_scale_desc_);
            new_variance_desc_ = permute_NXC2NCX(new_variance_desc_);
        }

        // after expand, the c channel is on the last dimension, which
        // meet the requirement of XIO format. But for OIX format, we
        // need permute c channel to the first dimension
        if (filter_format_ == "OIX") { // conv case
            new_scale_desc_ = permute_XIO2OIX(new_scale_desc_);
            new_variance_desc_ = permute_XIO2OIX(new_variance_desc_);
        }

        // temp = weights * scale
        dnnl::binary::desc mul_d(
                algorithm::binary_mul, weights, new_scale_desc_, weights);

        post_ops mul_post_ops;
        // updated_weight = temp / sqrt_variance
        mul_post_ops.append_binary(algorithm::binary_div, new_variance_desc_);

        primitive_attr mul_attr;
        mul_attr.set_post_ops(mul_post_ops);
        dnnl::binary::primitive_desc mul_pd(mul_d, mul_attr, p_engine);
        mul_prim_ = dnnl::binary(mul_pd);

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
        dnnl::binary::primitive_desc sub_pd(sub_d, sub_attr, p_engine);
        sub_prim_ = dnnl::binary(sub_pd);

        memory::dims scratchpad_dims = variance.dims();
        // sqrt_variance, zero_bias and others (like epslion),
        // or no need to alloc bias
        size_t factor = bias.is_zero() ? 3 : 2;
        scratchpad_dims[0] *= factor;
        scratchpad_desc_ = memory::desc(
                scratchpad_dims, variance.data_type(), memory::format_tag::a);
    }

    const memory::desc &scratchpad_desc() const { return scratchpad_desc_; }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        UNUSED(args);

        auto weights = args.find(DNNL_ARG_WEIGHTS)->second;
        auto bias = with_bias_ ? args.find(DNNL_ARG_BIAS)->second : memory();
        auto scale = args.find(DNNL_ARG_WEIGHTS_1)->second;
        auto shift = args.find(DNNL_ARG_WEIGHTS_2)->second;
        auto mean = args.find(DNNL_ARG_MEAN)->second;
        auto variance = args.find(DNNL_ARG_VARIANCE)->second;
        auto scratchpad = args.find(DNNL_ARG_SCRATCHPAD)->second;

        auto updated_weights = args.find(DNNL_ARG_DST_0)->second;
        auto updated_bias = args.find(DNNL_ARG_DST_1)->second;

        // 0. split scratchpad buffer to specific intermediate memory
        // sqrt_variance
        char *buf_start = (char *)scratchpad.get_data_handle();
        memory sqrt_variance = make_dnnl_memory(variance.get_desc(),
                scratchpad.get_engine(), (void *)buf_start);
        buf_start += variance.get_desc().get_size();
        // zero_bias
        memory valid_bias = bias;
        if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
            valid_bias = make_dnnl_memory(variance.get_desc(),
                    scratchpad.get_engine(), (void *)buf_start);
            buf_start += valid_bias.get_desc().get_size();
        }
        // epslion
        memory epsilon_mem = make_dnnl_memory(
                epsilon_desc_, scratchpad.get_engine(), (void *)buf_start);

        // 1. sqrt_variance = sqrt(variance + epsilon)
        if (variance.get_engine().get_kind() == engine::kind::cpu) {
            float *ptr = (float *)epsilon_mem.get_data_handle();
            *ptr = epsilon_;
        } else {
            engine cpu_eng(engine::kind::cpu, 0);
            memory cpu_mem = make_dnnl_memory(
                    epsilon_desc_, cpu_eng, (void *)&epsilon_);
            dnnl::reorder(cpu_mem, epsilon_mem)
                    .execute(stream, cpu_mem, epsilon_mem);
        }

        add_prim_.execute(stream,
                {{DNNL_ARG_SRC_0, variance}, {DNNL_ARG_SRC_1, epsilon_mem},
                        {DNNL_ARG_DST, sqrt_variance}});

        // 2. updated_weight = weights * scale / sqrt_variance
        memory new_scale(
                new_scale_desc_, scale.get_engine(), scale.get_data_handle());
        memory new_sqrt_variance(new_variance_desc_, sqrt_variance.get_engine(),
                sqrt_variance.get_data_handle());
        mul_prim_.execute(stream,
                {{DNNL_ARG_SRC_0, weights}, {DNNL_ARG_SRC_1, new_scale},
                        {DNNL_ARG_DST, updated_weights},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                new_sqrt_variance}});

        // 3. updated_bias = (bias - mean) * scale / sqrt_variance + shift
        if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
            // initialize the bias with zero value
            std::vector<float> zero(
                    graph::utils::prod(variance.get_desc().dims()), 0.0f);
            if (mean.get_engine().get_kind() == engine::kind::cpu) {
                std::memcpy(valid_bias.get_data_handle(), zero.data(),
                        valid_bias.get_desc().get_size());
            } else {
                engine cpu_eng(engine::kind::cpu, 0);
                memory cpu_mem = make_dnnl_memory(
                        variance.get_desc(), cpu_eng, zero.data());
                dnnl::reorder(cpu_mem, valid_bias)
                        .execute(stream, cpu_mem, valid_bias);
            }
        }

        sub_prim_.execute(stream,
                {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                        {DNNL_ARG_DST, updated_bias},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                scale},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                sqrt_variance},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                                shift}});
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        UNUSED(args);

        auto weights = args.find(DNNL_ARG_WEIGHTS)->second;
        auto bias = with_bias_ ? args.find(DNNL_ARG_BIAS)->second : memory();
        auto scale = args.find(DNNL_ARG_WEIGHTS_1)->second;
        auto shift = args.find(DNNL_ARG_WEIGHTS_2)->second;
        auto mean = args.find(DNNL_ARG_MEAN)->second;
        auto variance = args.find(DNNL_ARG_VARIANCE)->second;
        auto scratchpad = args.find(DNNL_ARG_SCRATCHPAD)->second;

        auto updated_weights = args.find(DNNL_ARG_DST_0)->second;
        auto updated_bias = args.find(DNNL_ARG_DST_1)->second;

        // 0. split scratchpad buffer to specific intermediate memory
        // sqrt_variance
        char *buf_start = (char *)scratchpad.get_data_handle();
        memory sqrt_variance = make_dnnl_memory(variance.get_desc(),
                scratchpad.get_engine(), (void *)buf_start);
        buf_start += variance.get_desc().get_size();
        // zero_bias
        memory valid_bias = bias;
        if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
            valid_bias = make_dnnl_memory(variance.get_desc(),
                    scratchpad.get_engine(), (void *)buf_start);
            buf_start += valid_bias.get_desc().get_size();
        }
        // epslion
        memory epsilon_mem = make_dnnl_memory(
                epsilon_desc_, scratchpad.get_engine(), (void *)buf_start);

        // 1. sqrt_variance = sqrt(variance + epsilon)
        auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
        sycl_queue
                .memcpy(epsilon_mem.get_data_handle(), &epsilon_,
                        epsilon_mem.get_desc().get_size())
                .wait();

        auto sycl_deps = dnnl::sycl_interop::execute(add_prim_, stream,
                {{DNNL_ARG_SRC_0, variance}, {DNNL_ARG_SRC_1, epsilon_mem},
                        {DNNL_ARG_DST, sqrt_variance}},
                deps);

        // 2. updated_weight = weights * scale / sqrt_variance
        memory new_scale(
                new_scale_desc_, scale.get_engine(), scale.get_data_handle());
        memory new_sqrt_variance(new_variance_desc_, sqrt_variance.get_engine(),
                sqrt_variance.get_data_handle());

        auto sycl_deps2 = dnnl::sycl_interop::execute(mul_prim_, stream,
                {{DNNL_ARG_SRC_0, weights}, {DNNL_ARG_SRC_1, new_scale},
                        {DNNL_ARG_DST, updated_weights},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                new_sqrt_variance}},
                {sycl_deps});

        // 3. updated_bias = (bias - mean) * scale / sqrt_variance + shift
        if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
            // initialize the bias with zero value
            std::vector<float> zero(
                    graph::utils::prod(variance.get_desc().dims()), 0.0f);
            sycl_queue
                    .memcpy(valid_bias.get_data_handle(), zero.data(),
                            valid_bias.get_desc().get_size())
                    .wait();
            auto sycl_deps3 = dnnl::sycl_interop::execute(sub_prim_, stream,
                    {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                            {DNNL_ARG_DST, updated_bias},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                    scale},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                    sqrt_variance},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                                    shift}},
                    {sycl_deps2});
            if (stream.get_engine().get_kind() == engine::kind::cpu)
                sycl_deps3.wait();
            return sycl_deps3;
        }

        auto sycl_deps3 = dnnl::sycl_interop::execute(sub_prim_, stream,
                {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                        {DNNL_ARG_DST, updated_bias},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                scale},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                sqrt_variance},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                                shift}},
                {sycl_deps2});
        if (stream.get_engine().get_kind() == engine::kind::cpu)
            sycl_deps3.wait();
        return sycl_deps3;
    }
#endif

private:
    memory::desc scratchpad_desc_;

    dnnl::binary add_prim_;
    dnnl::binary mul_prim_;
    dnnl::binary sub_prim_;

    float epsilon_;
    std::string data_format_;
    std::string filter_format_;

    memory::desc epsilon_desc_;
    memory::desc new_scale_desc_;
    memory::desc new_variance_desc_;

    bool with_bias_ {false};
};

struct conv_bwd_data_executable_t : public op_executable_t {
    conv_bwd_data_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_conv_bwd_data_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::convolution_backward_data(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::convolution_backward_data::primitive_desc pd_;
    dnnl::convolution_backward_data prim_;
};

struct conv_bwd_weights_executable_t : public op_executable_t {
    conv_bwd_weights_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_conv_bwd_weights_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::convolution_backward_weights(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::convolution_backward_weights::primitive_desc pd_;
    dnnl::convolution_backward_weights prim_;
};

struct batchnorm_executable_t : public op_executable_t {
    batchnorm_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        is_training_ = op->get_attr<bool>(op_attr::is_training);
        float momentum = 0.5;
        if (op->has_attr(op_attr::momentum))
            momentum = op->get_attr<float>(op_attr::momentum);
        scales_ = {momentum, 1 - momentum};
        pd_ = create_batchnorm_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::batch_normalization_forward(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (!is_training_) {
            prim_.execute(stream, args);
            return;
        }

        std::unordered_map<int, memory> exe_args = args;
        exe_args.erase(DNNL_ARG_SRC_1);
        exe_args.erase(DNNL_ARG_SRC_2);
        exe_args.erase(DNNL_ARG_DST_1);
        exe_args.erase(DNNL_ARG_DST_2);

        prim_.execute(stream, exe_args);

        // calculate running_mean and running_variance
        auto batch_mean = args.find(DNNL_ARG_MEAN)->second;
        auto batch_variance = args.find(DNNL_ARG_VARIANCE)->second;
        auto old_running_mean = args.find(DNNL_ARG_SRC_1)->second;
        auto old_running_variance = args.find(DNNL_ARG_SRC_2)->second;
        auto new_running_mean = args.find(DNNL_ARG_DST_1)->second;
        auto new_running_variance = args.find(DNNL_ARG_DST_2)->second;

        dnnl::engine p_engine = stream.get_engine();
        // new_running_mean = momentum * old_running_mean +
        //                                      (1 - momentum) * batch_mean
        dnnl::sum(
                {scales_, {old_running_mean.get_desc(), batch_mean.get_desc()},
                        p_engine})
                .execute(stream,
                        {{DNNL_ARG_MULTIPLE_SRC, old_running_mean},
                                {DNNL_ARG_MULTIPLE_SRC + 1, batch_mean},
                                {DNNL_ARG_DST, new_running_mean}});
        // new_running_variance = momentum * old_running_variance +
        //                                  (1 - momentum) * batch_variance
        dnnl::sum({scales_,
                          {old_running_variance.get_desc(),
                                  batch_variance.get_desc()},
                          p_engine})
                .execute(stream,
                        {{DNNL_ARG_MULTIPLE_SRC, old_running_variance},
                                {DNNL_ARG_MULTIPLE_SRC + 1, batch_variance},
                                {DNNL_ARG_DST, new_running_variance}});
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        if (!is_training_) {
            auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
            if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
            return e;
        }

        std::unordered_map<int, memory> exe_args = args;
        exe_args.erase(DNNL_ARG_SRC_1);
        exe_args.erase(DNNL_ARG_SRC_2);
        exe_args.erase(DNNL_ARG_DST_1);
        exe_args.erase(DNNL_ARG_DST_2);

        auto e0 = dnnl::sycl_interop::execute(prim_, stream, exe_args, deps);

        // calculate running_mean and running_variance
        auto batch_mean = args.find(DNNL_ARG_MEAN)->second;
        auto batch_variance = args.find(DNNL_ARG_VARIANCE)->second;
        auto old_running_mean = args.find(DNNL_ARG_SRC_1)->second;
        auto old_running_variance = args.find(DNNL_ARG_SRC_2)->second;
        auto new_running_mean = args.find(DNNL_ARG_DST_1)->second;
        auto new_running_variance = args.find(DNNL_ARG_DST_2)->second;

        dnnl::engine p_engine = stream.get_engine();
        // new_running_mean = momentum * old_running_mean +
        //                                      (1 - momentum) * batch_mean
        auto sum_prim_0 = dnnl::sum(
                {scales_, {old_running_mean.get_desc(), batch_mean.get_desc()},
                        p_engine});
        auto e1 = dnnl::sycl_interop::execute(sum_prim_0, stream,
                {{DNNL_ARG_MULTIPLE_SRC, old_running_mean},
                        {DNNL_ARG_MULTIPLE_SRC + 1, batch_mean},
                        {DNNL_ARG_DST, new_running_mean}},
                {e0});
        // new_running_variance = momentum * old_running_variance +
        //                                  (1 - momentum) * batch_variance
        dnnl::sum({scales_,
                          {old_running_variance.get_desc(),
                                  batch_variance.get_desc()},
                          p_engine})
                .execute(stream,
                        {{DNNL_ARG_MULTIPLE_SRC, old_running_variance},
                                {DNNL_ARG_MULTIPLE_SRC + 1, batch_variance},
                                {DNNL_ARG_DST, new_running_variance}});

        auto sum_prim_1 = dnnl::sum({scales_,
                {old_running_variance.get_desc(), batch_variance.get_desc()},
                p_engine});
        auto e2 = dnnl::sycl_interop::execute(sum_prim_0, stream,
                {{DNNL_ARG_MULTIPLE_SRC, old_running_variance},
                        {DNNL_ARG_MULTIPLE_SRC + 1, batch_variance},
                        {DNNL_ARG_DST, new_running_variance}},
                {e1});
        if (stream.get_engine().get_kind() == engine::kind::cpu) e2.wait();
        return e2;
    }
#endif

private:
    dnnl::batch_normalization_forward::primitive_desc pd_;
    dnnl::batch_normalization_forward prim_;
    bool is_training_ {false};
    std::vector<float> scales_;
};

struct batchnorm_bwd_executable_t : public op_executable_t {
    batchnorm_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_batchnorm_bwd_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::batch_normalization_backward(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::batch_normalization_backward::primitive_desc pd_;
    dnnl::batch_normalization_backward prim_;
};

struct resampling_executable_t : public op_executable_t {
    resampling_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_resampling_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::resampling_forward(pd_);
        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, const_cast<memory &>(psrc_mem),
                                const_cast<memory &>(dst_mem));
            }
        }
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto sycl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::sycl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        sycl_deps);
                sycl_deps = {e};
            }
        }
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::resampling_forward::primitive_desc pd_;
    dnnl::resampling_forward prim_;
    bool with_sum_ {false};
};

struct resampling_bwd_executable_t : public op_executable_t {
    resampling_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_resampling_bwd_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::resampling_backward(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::resampling_backward::primitive_desc pd_;
    dnnl::resampling_backward prim_;
};

struct layernorm_executable_t : public op_executable_t {
    layernorm_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_layernorm_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::layer_normalization_forward(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::layer_normalization_forward::primitive_desc pd_;
    dnnl::layer_normalization_forward prim_;
};

struct layernorm_bwd_executable_t : public op_executable_t {
    layernorm_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_layernorm_bwd_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::layer_normalization_backward(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::layer_normalization_backward::primitive_desc pd_;
    dnnl::layer_normalization_backward prim_;
};

struct sum_executable_t : public op_executable_t {
    sum_executable_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            fusion_info_mgr_t &mgr) {
        pd_ = create_dnnl_sum_pd(op, p_engine, mgr);
        prim_ = dnnl::sum(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::sum::primitive_desc pd_;
    dnnl::sum prim_;
};

struct softmax_executable_t : public op_executable_t {
    softmax_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_softmax_pd(
                op, p_engine, mgr, pd_cache, dnnl::algorithm::softmax_accurate)
                      .first;
        prim_ = dnnl::softmax_v2_forward(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::softmax_v2_forward::primitive_desc pd_;
    dnnl::softmax_v2_forward prim_;
};

struct softmax_bwd_executable_t : public op_executable_t {
    softmax_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_softmax_bwd_pd(
                op, p_engine, mgr, pd_cache, dnnl::algorithm::softmax_accurate)
                      .first;
        prim_ = dnnl::softmax_v2_backward(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::softmax_v2_backward::primitive_desc pd_;
    dnnl::softmax_v2_backward prim_;
};

struct logsoftmax_executable_t : public op_executable_t {
    logsoftmax_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_softmax_pd(
                op, p_engine, mgr, pd_cache, dnnl::algorithm::softmax_log)
                      .first;
        prim_ = dnnl::softmax_v2_forward(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::softmax_v2_forward::primitive_desc pd_;
    dnnl::softmax_v2_forward prim_;
};

struct logsoftmax_bwd_executable_t : public op_executable_t {
    logsoftmax_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_softmax_bwd_pd(
                op, p_engine, mgr, pd_cache, dnnl::algorithm::softmax_log)
                      .first;
        prim_ = dnnl::softmax_v2_backward(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::softmax_v2_backward::primitive_desc pd_;
    dnnl::softmax_v2_backward prim_;
};

struct reduction_executable_t : public op_executable_t {
    reduction_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_reduction_pd(op, p_engine, mgr, pd_cache).first;
        prim_ = dnnl::reduction(pd_);

        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, const_cast<memory &>(psrc_mem),
                                const_cast<memory &>(dst_mem));
            }
        }

        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto sycl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::sycl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        sycl_deps);
                sycl_deps = {e};
            }
        }

        auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

private:
    dnnl::reduction::primitive_desc pd_;
    dnnl::reduction prim_;
    bool with_sum_ {false};
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
