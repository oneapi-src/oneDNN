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

#include <memory>
#include <vector>

#include "interface/c_types_map.hpp"
#include "interface/graph.hpp"
#include "interface/value.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pattern {

template <size_t N>
bool check_input_num(op_t *op) {
    return op->num_inputs() == N;
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

template <impl::op_kind_t KIND>
bool check_successor_op_kind(op_t *op) {
    auto out_value = op->get_output_value(0);
    if (out_value->get_consumers().empty()) return false;
    auto &successor = out_value->get_consumers()[0].get_op();
    return successor.get_kind() == KIND;
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

} // namespace pattern
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
