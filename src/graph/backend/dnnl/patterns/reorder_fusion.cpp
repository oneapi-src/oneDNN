/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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
#include "graph/backend/dnnl/patterns/transformation_pattern.hpp"
#include "graph/backend/dnnl/patterns/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

namespace pm = graph::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph = pm::pb_graph_t;
using FCreateV2Pattern = graph::pass::FCreateV2Pattern;

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(reorder_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, reorder_sum_fusion)
        .set_priority(10.1f)
        .set_kind(partition_kind::misc_post_ops)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op_t *reorder = pgraph->append_op(
                            graph::op_kind::Reorder, "preorder");
                    pm::pb_op_t *add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, reorder, 0)}, "padd");
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

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, int8_reorder_fusion)
        .set_priority(10.1f)
        .set_kind(partition_kind::misc_quantized_post_ops)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op_t *dequant = pgraph->append_op(
                            graph::op_kind::Dequantize, "pdequant");
                    pm::pb_op_t *reorder
                            = pgraph->append_op(graph::op_kind::Reorder,
                                    {in_edge(0, dequant, 0)}, "preorder");
                    pgraph->append_op(graph::op_kind::Quantize,
                            {in_edge(0, reorder, 0)}, "pquant");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_reorder>();
        });

/*
Currently DNNL Backend doesn't support Post-sum/binary with zero points
on GPU, while CPU supports.
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, int8_reorder_sum_fusion_cpu)
        .set_priority(10.2f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(partition_kind::misc_quantized_post_ops)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op_t *dequant = pgraph->append_op(
                            graph::op_kind::Dequantize, "pdequant");
                    pm::pb_op_t *dequant_other = pgraph->append_op(
                            graph::op_kind::Dequantize, "pdequant_other");
                    pm::pb_op_t *reorder
                            = pgraph->append_op(graph::op_kind::Reorder,
                                    {in_edge(0, dequant, 0)}, "preorder");
                    pm::pb_op_t *add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, reorder, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");
                    add->append_decision_function([](op_t *graph_op) -> bool {
                        return !graph_op->has_attr(op_attr::auto_broadcast)
                                || graph_op->get_attr<std::string>(
                                           op_attr::auto_broadcast)
                                == "none";
                    });
                    pgraph->append_op(graph::op_kind::Quantize,
                            {in_edge(0, add, 0)}, "pquant");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_reorder>();
        });

/*
Currently DNNL Backend doesn't support Post-sum/binary with zero points
on GPU, while CPU supports.
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, int8_reorder_sum_fusion_gpu)
        .set_priority(10.2f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(partition_kind::misc_quantized_post_ops)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op_t *dequant = pgraph->append_op(
                            graph::op_kind::Dequantize, "pdequant");
                    pm::pb_op_t *dequant_other = pgraph->append_op(
                            graph::op_kind::Dequantize, "pdequant_other");
                    dequant_other->append_decision_function(
                            check_zps_values<0>);
                    pm::pb_op_t *reorder
                            = pgraph->append_op(graph::op_kind::Reorder,
                                    {in_edge(0, dequant, 0)}, "preorder");
                    pm::pb_op_t *add = pgraph->append_op(graph::op_kind::Add,
                            {in_edge(0, reorder, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");
                    add->append_decision_function([](op_t *graph_op) -> bool {
                        return !graph_op->has_attr(op_attr::auto_broadcast)
                                || graph_op->get_attr<std::string>(
                                           op_attr::auto_broadcast)
                                == "none";
                    });
                    pgraph->append_op(graph::op_kind::Quantize,
                            {in_edge(0, add, 0)}, "pquant");
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
