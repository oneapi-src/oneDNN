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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRANSFORM_TRANSFORM_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRANSFORM_TRANSFORM_HPP

#include "../graph.hpp"
namespace sc {

SC_INTERNAL_API void elemwise_bcast_swap(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void elemwise_dimension_alignment(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void layout_propagation(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void batchwise_merge(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

void fuse_ops(sc_graph_t &g, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void graph_inline(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

/**
 * Optimize the constant_optimizable_t ops.
 * */
SC_INTERNAL_API void constant_optimization(
        sc_graph_t &graph, const context_ptr &ctx);

void permute_propagation(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void tensor_view_transform(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void graph_simplify(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

void horizontal_merge(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void global_reschedule(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void inplace_transform(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

void brgemm_fusion_transform(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());
namespace quantize {
SC_INTERNAL_API void quantize_info_propagation(
        sc_graph_t &mgr, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void graph_reschedule(
        sc_graph_t &mgr, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void calculate_op_compensation(
        sc_graph_t &mgr, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void quantize_inline(
        sc_graph_t &mgr, const context_ptr &ctx = get_default_context());
} // namespace quantize

} // namespace sc

#endif
