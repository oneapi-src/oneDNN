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
#ifndef BACKEND_DNNL_PATTERNS_FUSIONS_HPP
#define BACKEND_DNNL_PATTERNS_FUSIONS_HPP

#include "backend/dnnl/patterns/transformation_pattern.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pass {

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

template <impl::op_kind_t OPKIND>
bool check_post_ops_only_one_add(op_t *op) {
    size_t num_add = 0;
    op_t *current_op = op;
    while (current_op->get_kind() != OPKIND) {
        if (current_op->get_kind() == impl::op_kind::Add) {
            num_add += 1;
            if (!current_op->get_input_value(0)->has_producer()
                    || !current_op->get_input_value(1)->has_producer())
                return false;
            if (current_op->get_input_op(0)->get_kind()
                            != impl::op_kind::Dequantize
                    && current_op->get_input_op(1)->get_kind()
                            != impl::op_kind::Dequantize)
                return false;
            if (current_op->get_input_op(0)->get_kind()
                    == impl::op_kind::Dequantize)
                current_op = current_op->get_input_op(1);
            else
                current_op = current_op->get_input_op(0);
        } else {
            current_op = current_op->get_input_op(0);
        }
    }
    return num_add == 1;
}

DNNL_BACKEND_REGISTER_PASSES_DECLARE(conv_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(matmul_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(binary_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(bn_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(convtranspose_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(eltwise_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(gelu_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(interpolate_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(pool_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(quantize_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(reduction_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(reorder_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(shuffle_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(single_op_pass)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(sum_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(concat_fusion)

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
#endif
