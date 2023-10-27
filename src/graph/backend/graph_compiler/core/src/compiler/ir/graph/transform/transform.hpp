/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRANSFORM_TRANSFORM_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRANSFORM_TRANSFORM_HPP

#include <vector>
#include "../graph.hpp"
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_INTERNAL_API void elemwise_bcast_swap(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void broadcast_transform(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void elemwise_dimension_alignment(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void dynamic_graph_transform(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void layout_propagation(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void pre_padding(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void mixed_partition(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void annotate_config(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void graph_inline(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void partial_reduce_replace(
        sc_graph_t &graph, const context_ptr &ctx);
/**
 * Optimize the constant_optimizable_t ops.
 * */
SC_INTERNAL_API void constant_optimization(
        sc_graph_t &graph, const context_ptr &ctx);

void permute_propagation(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void tensor_view_transform(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void flatten_conv(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API
void graph_simplify(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void global_reschedule(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void inplace_transform(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void div_bcast_transform(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

void brgemm_fusion_transform(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void shape_relationship_binding(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void merge_concats(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void graph_concat_memory_planning(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void rl_conv_weight_transform(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

namespace quantize {
SC_INTERNAL_API void annotate_fusion_break(
        sc_graph_t &mgr, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void quantize_info_propagation(
        sc_graph_t &mgr, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void graph_reschedule(
        sc_graph_t &mgr, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void calculate_op_compensation(
        sc_graph_t &mgr, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void quantize_inline(
        sc_graph_t &mgr, const context_ptr &ctx = get_default_context());
} // namespace quantize

SC_INTERNAL_API void fpmath_mode(
        sc_graph_t &mgr, const context_ptr &ctx = get_default_context());

SC_INTERNAL_API void eliminate_zero_shaped_tensors(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
