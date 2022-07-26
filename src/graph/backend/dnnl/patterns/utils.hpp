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
#ifndef GRAPH_BACKEND_DNNL_PATTERNS_UTILS_HPP
#define GRAPH_BACKEND_DNNL_PATTERNS_UTILS_HPP

#include <memory>
#include <vector>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/graph.hpp"
#include "graph/interface/value.hpp"

namespace dnnl {
namespace impl {
namespace graph {
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

template <data_type_t DTYPE>
bool check_input_dtype(op_t *op) {
    for (size_t i = 0; i < op->num_inputs(); ++i) {
        const logical_tensor_t &iport
                = op->get_input_value(i)->get_logical_tensor();
        if (iport.data_type != DTYPE) return false;
    }

    return true;
}

template <data_type_t DTYPE>
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

inline const std::vector<op_kind_t> &get_unary_ops() {
    const static std::vector<op_kind_t> unary = {graph::op_kind::Abs,
            graph::op_kind::Clamp, graph::op_kind::Elu, graph::op_kind::Exp,
            graph::op_kind::GELU, graph::op_kind::HardSwish,
            graph::op_kind::LeakyReLU, graph::op_kind::Log,
            graph::op_kind::Mish, graph::op_kind::Sigmoid,
            graph::op_kind::SoftPlus, graph::op_kind::Pow, graph::op_kind::ReLU,
            graph::op_kind::Round, graph::op_kind::Sqrt, graph::op_kind::Square,
            graph::op_kind::Tanh};

    return unary;
}

inline const std::vector<op_kind_t> &get_binary_ops() {
    const static std::vector<op_kind_t> binary
            = {graph::op_kind::Add, graph::op_kind::Multiply,
                    graph::op_kind::Maximum, graph::op_kind::Minimum,
                    graph::op_kind::Divide, graph::op_kind::Subtract};

    return binary;
}

inline const std::vector<op_kind_t> &get_unary_binary_ops() {
    const static std::vector<op_kind_t> unary_binary = {graph::op_kind::Abs,
            graph::op_kind::Clamp, graph::op_kind::Elu, graph::op_kind::Exp,
            graph::op_kind::GELU, graph::op_kind::HardSwish,
            graph::op_kind::LeakyReLU, graph::op_kind::Log,
            graph::op_kind::Mish, graph::op_kind::Sigmoid,
            graph::op_kind::SoftPlus, graph::op_kind::Pow, graph::op_kind::ReLU,
            graph::op_kind::Round, graph::op_kind::Sqrt, graph::op_kind::Square,
            graph::op_kind::Tanh, graph::op_kind::Add, graph::op_kind::Multiply,
            graph::op_kind::Maximum, graph::op_kind::Minimum,
            graph::op_kind::Divide, graph::op_kind::Subtract};

    return unary_binary;
}

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
