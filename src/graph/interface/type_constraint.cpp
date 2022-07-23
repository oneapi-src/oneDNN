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

#include "graph/interface/type_constraint.hpp"

namespace dnnl {
namespace impl {
namespace graph {

bool check_bn_fwd_data_type(const op_t *n) {
    const logical_tensor_t &T1_lt = n->get_input_value(0)->get_logical_tensor();
    const logical_tensor_t &T2_lt = n->get_input_value(1)->get_logical_tensor();

    // only when data is bf16, gamma/beta/mean/var can be bf16.
    // If data is bf16, gamma/beta/mean/var can be f32 or bf16.
    if (T1_lt.data_type != data_type::bf16
            && T2_lt.data_type == data_type::bf16)
        return false;
    else
        return true;
}

bool check_bn_bwd_data_type(const op_t *n) {
    const logical_tensor_t &T1_lt = n->get_input_value(0)->get_logical_tensor();
    const logical_tensor_t &T2_lt = n->get_input_value(2)->get_logical_tensor();

    // only when data is bf16, gamma/beta/mean/var can be bf16.
    // If data is bf16, gamma/beta/mean/var can be f32 or bf16.
    if (T1_lt.data_type != data_type::bf16
            && T2_lt.data_type == data_type::bf16)
        return false;
    else
        return true;
}

bool check_ln_data_type(const op_t *n) {
    auto input_values = n->get_input_values();
    auto output_values = n->get_output_values();

    const logical_tensor_t &T1_lt = input_values[0]->get_logical_tensor();
    logical_tensor_t T2_lt;
    // check if optional input /output exists
    if (input_values.size() > 1) {
        T2_lt = input_values[1]->get_logical_tensor();
    } else if (output_values.size() > 1) {
        T2_lt = output_values[1]->get_logical_tensor();
    } else {
        return true;
    }
    // only when data is bf16, gamma/beta/mean/var can be bf16.
    // If data is bf16, gamma/beta/mean/var can be f32 or bf16.
    if (T1_lt.data_type != data_type::bf16
            && T2_lt.data_type == data_type::bf16)
        return false;
    else
        return true;
}

bool check_typecast_data_type(const op_t *n) {
    const logical_tensor_t &T1_lt = n->get_input_value(0)->get_logical_tensor();
    const logical_tensor_t &T2_lt
            = n->get_output_value(0)->get_logical_tensor();
    // for TypeCast, input & output should not have the same dtype
    if (T1_lt.data_type == T2_lt.data_type) return false;
    if (T1_lt.data_type == data_type::f16 && T2_lt.data_type == data_type::bf16)
        return false;
    if (T1_lt.data_type == data_type::bf16 && T2_lt.data_type == data_type::f16)
        return false;
    return true;
}

} // namespace graph
} // namespace impl
} // namespace dnnl
