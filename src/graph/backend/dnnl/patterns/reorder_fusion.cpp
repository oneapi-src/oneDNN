/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include "graph/backend/dnnl/internal_ops.hpp"
#include "graph/backend/dnnl/kernels/reorder.hpp"
#include "graph/backend/dnnl/patterns/fusions.hpp"
#include "graph/backend/dnnl/patterns/pattern_matcher_pass.hpp"
#include "graph/backend/dnnl/patterns/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

namespace pm = graph::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph = pm::pb_graph_t;
using FCreatePattern = graph::pass::FCreatePattern;

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(reorder_fusion)

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, reorder_sum_fusion)
        .set_priority(10.1f)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op_t *reorder
                            = pgraph->append_op(graph::op_kind::Reorder);
                    pm::pb_op_t *add = pgraph->append_op(
                            graph::op_kind::Add, {in_edge(0, reorder, 0)});
                    add->append_decision_function([](op_t *graph_op) -> bool {
                        return !graph_op->has_attr(op_attr::auto_broadcast)
                                || graph_op->get_attr<std::string>(
                                           op_attr::auto_broadcast)
                                == "none";
                    });
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_reorder>();
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, int8_reorder_fusion)
        .set_priority(10.1f)
        .set_kind(partition_kind_t::misc_quantized_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op_t *dequant
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    pm::pb_op_t *reorder = pgraph->append_op(
                            graph::op_kind::Reorder, {in_edge(0, dequant, 0)});
                    pgraph->append_op(
                            graph::op_kind::Quantize, {in_edge(0, reorder, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_reorder>();
        });

/*
Currently DNNL Backend doesn't support Post-sum/binary with zero points
on GPU, while CPU supports.
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, int8_reorder_sum_fusion_cpu)
        .set_priority(10.2f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(partition_kind_t::misc_quantized_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op_t *dequant
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    pm::pb_op_t *dequant_other
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    pm::pb_op_t *reorder = pgraph->append_op(
                            graph::op_kind::Reorder, {in_edge(0, dequant, 0)});
                    pm::pb_op_t *add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, reorder, 0),
                                    in_edge(1, dequant_other, 0)});
                    add->append_decision_function([](op_t *graph_op) -> bool {
                        return !graph_op->has_attr(op_attr::auto_broadcast)
                                || graph_op->get_attr<std::string>(
                                           op_attr::auto_broadcast)
                                == "none";
                    });
                    pgraph->append_op(
                            graph::op_kind::Quantize, {in_edge(0, add, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_reorder>();
        });

/*
Currently DNNL Backend doesn't support Post-sum/binary with zero points
on GPU, while CPU supports.
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, int8_reorder_sum_fusion_gpu)
        .set_priority(10.2f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(partition_kind_t::misc_quantized_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op_t *dequant
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    pm::pb_op_t *dequant_other
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    dequant_other->append_decision_function(
                            check_zps_values<0>);
                    pm::pb_op_t *reorder = pgraph->append_op(
                            graph::op_kind::Reorder, {in_edge(0, dequant, 0)});
                    pm::pb_op_t *add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, reorder, 0),
                                    in_edge(1, dequant_other, 0)});
                    add->append_decision_function([](op_t *graph_op) -> bool {
                        return !graph_op->has_attr(op_attr::auto_broadcast)
                                || graph_op->get_attr<std::string>(
                                           op_attr::auto_broadcast)
                                == "none";
                    });
                    pgraph->append_op(
                            graph::op_kind::Quantize, {in_edge(0, add, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_reorder>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
