/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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
#ifndef BACKEND_DNNL_PATTERNS_UTILS_HPP
#define BACKEND_DNNL_PATTERNS_UTILS_HPP

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "interface/c_types_map.hpp"
#include "interface/graph.hpp"
#include "interface/value.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pattern {

template <int64_t N>
bool check_zps_values(op_t *op) {
    auto zps = op->get_attr<std::vector<int64_t>>(op_attr::zps);
    return std::all_of(
            zps.begin(), zps.end(), [](int64_t i) { return i == N; });
}

template <size_t N>
bool check_input_num(op_t *op) {
    return op->num_inputs() == N;
}

template <size_t N>
bool check_output_num(op_t *op) {
    return op->num_outputs() == N;
}

template <impl::data_type_t DTYPE>
bool check_input_dtype(op_t *op) {
    for (size_t i = 0; i < op->num_inputs(); ++i) {
        const logical_tensor_t &iport
                = op->get_input_value(i)->get_logical_tensor();
        if (iport.data_type != DTYPE) return false;
    }

    return true;
}

template <impl::data_type_t DTYPE>
bool check_output_dtype(op_t *op) {
    for (size_t i = 0; i < op->num_outputs(); ++i) {
        const logical_tensor_t &oport
                = op->get_output_value(i)->get_logical_tensor();
        if (oport.data_type != DTYPE) return false;
    }

    return true;
}

template <size_t N>
bool check_producer_input_num(op_t *op) {
    op_t *producer = op->get_input_op(0);
    return producer->num_inputs() == N;
}

inline bool check_qtype_equal_to_per_tensor(op_t *op) {
    std::string qtype = op->get_attr<std::string>(op_attr::qtype);
    return qtype == "per_tensor";
}

inline const std::vector<impl::op_kind_t> &get_unary_ops() {
    const static std::vector<impl::op_kind_t> unary = {impl::op_kind::Abs,
            impl::op_kind::Clamp, impl::op_kind::Elu, impl::op_kind::Exp,
            impl::op_kind::GELU, impl::op_kind::HardSwish,
            impl::op_kind::LeakyReLU, impl::op_kind::Log, impl::op_kind::Mish,
            impl::op_kind::Sigmoid, impl::op_kind::SoftPlus, impl::op_kind::Pow,
            impl::op_kind::ReLU, impl::op_kind::Round, impl::op_kind::Sqrt,
            impl::op_kind::Square, impl::op_kind::Tanh};

    return unary;
}

inline const std::vector<impl::op_kind_t> &get_binary_ops() {
    const static std::vector<impl::op_kind_t> binary
            = {impl::op_kind::Add, impl::op_kind::Multiply,
                    impl::op_kind::Maximum, impl::op_kind::Minimum,
                    impl::op_kind::Divide, impl::op_kind::Subtract};

    return binary;
}

inline const std::vector<impl::op_kind_t> &get_unary_binary_ops() {
    const static std::vector<impl::op_kind_t> unary_binary = {
            impl::op_kind::Abs, impl::op_kind::Clamp, impl::op_kind::Elu,
            impl::op_kind::Exp, impl::op_kind::GELU, impl::op_kind::HardSwish,
            impl::op_kind::LeakyReLU, impl::op_kind::Log, impl::op_kind::Mish,
            impl::op_kind::Sigmoid, impl::op_kind::SoftPlus, impl::op_kind::Pow,
            impl::op_kind::ReLU, impl::op_kind::Round, impl::op_kind::Sqrt,
            impl::op_kind::Square, impl::op_kind::Tanh, impl::op_kind::Add,
            impl::op_kind::Multiply, impl::op_kind::Maximum,
            impl::op_kind::Minimum, impl::op_kind::Divide,
            impl::op_kind::Subtract};

    return unary_binary;
}

// Optional Quantize for weight will only be fused
// when:
// 1. input logical tensor has constant property type
// 2. the optional Quantize has a Wildcard producer
// 3. the optional Quantize has no producer
inline bool check_if_constant_weight(op_t *op) {
    const auto &in_value = op->get_input_value(0);
    if (in_value->get_logical_tensor().property
            == impl::property_type::constant) {
        return true;
    }
    if (in_value->has_producer()) {
        return in_value->get_producer().get_kind() == impl::op_kind::Wildcard;
    } else {
        return true;
    }
}

// Optional BiasAdd after operator like Conv/ConvTranspose/Matmul. If
// `maybe_typecase` is true, there will also be an optional TypeCast before the
// 2nd input of BiasAdd.
inline impl::utils::pm::repetition_t *optional_bias_add(
        const std::shared_ptr<impl::utils::pm::pb_graph_t> &pgraph,
        impl::utils::pm::pb_op_t *input, bool maybe_typecast = false) {
    auto popt_bias_graph
            = std::make_shared<impl::utils::pm::pb_graph_t>("poptional_bias");
    impl::utils::pm::pb_op_t *pbias = nullptr;
    if (maybe_typecast) {
        auto popt_tc_graph
                = std::make_shared<impl::utils::pm::pb_graph_t>("poptional_tc");
        impl::utils::pm::pb_op_t *typecast_bias
                = popt_tc_graph->append_op(impl::op_kind::TypeCast, "tc_bias");
        typecast_bias->append_decision_function(
                check_output_dtype<impl::data_type::bf16>);
        popt_tc_graph->create_input_port(0, typecast_bias, 0);
        popt_tc_graph->create_output_port(0, typecast_bias, 0);
        auto popt_tc
                = popt_bias_graph->append_optional(popt_tc_graph, "popt_tc");
        pbias = popt_bias_graph->append_op(impl::op_kind::BiasAdd,
                impl::utils::pm::in_edges_t {in_edge(1, popt_tc, 0)}, "pbias");
    } else {
        pbias = popt_bias_graph->append_op(impl::op_kind::BiasAdd, "pbias");
    }
    pbias->append_decision_function(check_producer_input_num<2>);
    popt_bias_graph->create_input_port(0, pbias, 0);
    popt_bias_graph->create_output_port(0, pbias, 0);
    auto popt_bias = pgraph->append_optional(popt_bias_graph,
            impl::utils::pm::in_edges_t {in_edge(0, input, 0)}, "popt_bias");
    return popt_bias;
};

} // namespace pattern
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
