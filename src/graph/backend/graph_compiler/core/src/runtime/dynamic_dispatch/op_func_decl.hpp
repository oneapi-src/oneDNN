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
#include <stdint.h>
#include <runtime/dynamic_dispatch/ops/runtime_op_info.hpp>

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_OP_FUNC_DECL_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_OP_FUNC_DECL_HPP

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

extern "C" {
SC_API void infer_shape_matmul_op(void *out, void *data, void *weight);
SC_API void infer_shape_conv_fwd_op(void *out, void *data, void *weight,
        dyn_conv_fwd_runtime_info_t &op_info);
SC_API void infer_shape_unary_fusible_op(void *out, void *in);
SC_API void infer_shape_binary_fusible_op(void *out, void *in0, void *in1);
SC_API void infer_shape_reduce_op(
        void *out, void *in, int *rd_axis, int num_axis);
SC_API void infer_shape_transpose_op(
        void *out, void *in, int *tr_axis, int num_axis);
SC_API void infer_shape_tensor_view_op(void *out, void *in, int64_t *old_axis,
        int num_old_axis, int64_t *new_axis, int num_new_axis);
SC_API void infer_shape_select_op(void *out, void *in0, void *in1, void *in2);

SC_API void query_format_matmul_core_op(void *table, void *out, void *data,
        void *weight, uint64_t *out_fmt, uint64_t *data_fmt,
        uint64_t *weight_fmt, uint64_t *out_size, void *kernel, int *impl_alg);
SC_API void query_format_managed_matmul_core_op(void *table, void *out,
        void *data, void *weight, uint64_t *out_fmt, uint64_t *data_fmt,
        uint64_t *weight_fmt, uint64_t *out_size, void *kernel, int *impl_alg);
SC_API void query_format_conv_fwd_core_op(void *table, void *out, void *data,
        void *weight, uint64_t *out_fmt, uint64_t *data_fmt,
        uint64_t *weight_fmt, uint64_t *out_size, void *kernel, int *impl_alg);
SC_API void query_format_unary_fusible_op(void *table, void *out, void *in,
        uint64_t *out_fmt, uint64_t *in_fmt, uint64_t *out_size, void *kernel);
SC_API void query_format_binary_fusible_op(void *table, void *out, void *in0,
        void *in1, uint64_t *out_fmt, uint64_t *in0_fmt, uint64_t *in1_fmt,
        uint64_t *out_size, void *kernel);
SC_API void query_format_reorder_op(void *table, void *out, void *in,
        uint64_t *out_fmt, uint64_t *in_fmt, uint64_t *out_size, void *kernel,
        int *impl_alg);
SC_API void query_format_padding_op(void *table, void *out, void *in,
        uint64_t *out_fmt, uint64_t *in_fmt, uint64_t *out_size, void *kernel,
        int *impl_alg);
SC_API void query_format_reduce_op(void *table, void *out, void *in,
        uint64_t *out_fmt, uint64_t *in_fmt, uint64_t *out_size, void *kernel);
SC_API void query_format_tensor_view_op(void *table, void *out, void *in,
        uint64_t *out_fmt, uint64_t *in_fmt, uint64_t *out_size, void *kernel);
SC_API void query_format_dynamic_reshape_op(void *table, void *out, void *in1,
        void *in2, uint64_t *out_fmt, uint64_t *in_fmt1, uint64_t *in_fmt2,
        uint64_t *out_size, void *kernel);
SC_API void query_format_select_op(void *table, void *out, void *in0, void *in1,
        void *in2, uint64_t *out_fmt, uint64_t *in0_fmt, uint64_t *in1_fmt,
        uint64_t *in2_fmt, uint64_t *out_size, void *kernel);
SC_API void query_combined_fused_op(void *table, uint64_t **combined_keys,
        int *combined_algs, int *each_op_num_key, int op_num, void *kernel);
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
