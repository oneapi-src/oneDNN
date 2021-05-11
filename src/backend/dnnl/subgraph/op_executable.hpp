/*******************************************************************************
 * Copyright 2021 Intel Corporation
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
#ifndef BACKEND_DNNL_SUBGRAPH_OP_EXECUTABLE_HPP
#define BACKEND_DNNL_SUBGRAPH_OP_EXECUTABLE_HPP

#include <memory>
#include <string>
#include <unordered_map>

#include "dnnl.hpp"

#include "backend/dnnl/subgraph/passes.hpp"

#define DNNL_GRAPH_ARG_POST_SRC -1

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

inline dnnl::convolution_forward::primitive_desc create_conv_pd(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        primitive_attr_mgr &prm_attr_mgr) {
    // prepare the operator attributes
    auto strides = op->get_attr<dims>("strides");
    auto dilates = op->get_attr<dims>("dilations");
    auto pads_begin = op->get_attr<dims>("pads_begin");
    auto pads_end = op->get_attr<dims>("pads_end");
    dilates = get_compatible_dilates(dilates);

    int64_t key = op->get_attr<int64_t>("primitive_attr_key");
    dnnl::primitive_attr prm_attr = prm_attr_mgr.get_attr(key);
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto weight = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());

    if (op->has_attr("output_format")
            && op->get_attr<std::string>("output_format") == "NXC") {
        dst = permute_NXC2NCX(dst);
    }

    dnnl::convolution_forward::primitive_desc pd;
    if (op->has_attr("with_bias") && op->get_attr<bool>("with_bias")) {
        auto bias = make_dnnl_memory_desc(
                op->get_input_value(2)->get_logical_tensor());
        pd = dnnl::convolution_forward::primitive_desc(
                {prop_kind::forward, algorithm::convolution_direct, src, weight,
                        bias, dst, strides, dilates, pads_begin, pads_end},
                prm_attr, p_engine);
    } else {
        pd = dnnl::convolution_forward::primitive_desc(
                {prop_kind::forward, algorithm::convolution_direct, src, weight,
                        dst, strides, dilates, pads_begin, pads_end},
                prm_attr, p_engine);
    }

    return pd;
}

inline dnnl::matmul::primitive_desc create_matmul_pd(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        primitive_attr_mgr &prm_attr_mgr) {
    int64_t key = op->get_attr<int64_t>("primitive_attr_key");
    dnnl::primitive_attr prm_attr = prm_attr_mgr.get_attr(key);
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto wei = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());

    dnnl::matmul::primitive_desc pd;
    if (op->has_attr("with_bias") && op->get_attr<bool>("with_bias")) {
        auto bias = make_dnnl_memory_desc(
                op->get_input_value(2)->get_logical_tensor());
        pd = dnnl::matmul::primitive_desc(
                {src, wei, bias, dst}, prm_attr, p_engine);
    } else {
        pd = dnnl::matmul::primitive_desc({src, wei, dst}, prm_attr, p_engine);
    }
    return pd;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
