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
#include <stdint.h>

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_OP_FUNC_DECL_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_OP_FUNC_DECL_HPP

namespace sc {

extern "C" {

SC_API void query_format_matmul_core_op(void *table, void *out, void *data,
        void *weight, uint64_t *out_fmt, uint64_t *data_fmt,
        uint64_t *weight_fmt, uint64_t *out_size, void *kernel);
SC_API void query_format_unary_fusible_op(void *table, void *out, void *in,
        uint64_t *out_fmt, uint64_t *in_fmt, uint64_t *out_size, void *kernel);
SC_API void query_format_binary_fusible_op(void *table, void *out, void *in0,
        void *in1, uint64_t *out_fmt, uint64_t *in0_fmt, uint64_t *in1_fmt,
        uint64_t *out_size, void *kernel);
SC_API void query_format_reorder_op(void *table, void *out, void *in,
        uint64_t *out_fmt, uint64_t *in_fmt, uint64_t *out_size, void *kernel);
SC_API void query_format_reduce_op(void *table, void *out, void *in,
        uint64_t *out_fmt, uint64_t *in_fmt, uint64_t *out_size, void *kernel);
SC_API void query_format_tensor_view_op(void *table, void *out, void *in,
        uint64_t *out_fmt, uint64_t *in_fmt, uint64_t *out_size, void *kernel);
SC_API void calculate_shape_of_tensor_op(void *out, void *in, uint64_t *out_fmt,
        uint64_t *in_fmt, int *shape_idxs, int shape_size);
};

} // namespace sc

#endif
