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

#include "backend/dnnl/patterns/fusions.hpp"

#include "utils/pm/pbuilder.hpp"
namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pass {

namespace pm = impl::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreateV2FusedOp = impl::pass::FCreateV2FusedOp;
using FCreateV2Pattern = impl::pass::FCreateV2Pattern;

/*!
 * \brief This provides pool fusion.
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */
DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(pool_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, pool_binary_fusion)
        .set_priority(9.9f)
        .set_kind(impl::partition_kind::pooling_post_ops)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *ppool = pgraph->append_alternation(
                            {impl::op_kind::AvgPool, impl::op_kind::MaxPool},
                            "peltwise");
                    pgraph->append_alternation(
                            {impl::op_kind::Add, impl::op_kind::Multiply,
                                    impl::op_kind::Maximum,
                                    impl::op_kind::Minimum,
                                    impl::op_kind::Divide,
                                    impl::op_kind::Subtract},
                            {in_edge(0, ppool, 0)}, "pbinary");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::pool_binary);
                    fused_op->set_attr<std::string>(op_attr::backend, "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_pool_binary_fusion)
        .set_priority(10.0f)
        .set_kind(impl::partition_kind::quantized_pooling_post_ops)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto pdequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "pdequnt_data");

                    auto ppool = pgraph->append_alternation(
                            {impl::op_kind::AvgPool, impl::op_kind::MaxPool},
                            {in_edge(0, pdequant_data, 0)}, "ppool");

                    auto padd_subgraph
                            = std::make_shared<pb_graph_t>("padd_subgraph");
                    auto pdequant_other = padd_subgraph->append_op(
                            impl::op_kind::Dequantize, "pdequnt_other");
                    auto padd = padd_subgraph->append_op(impl::op_kind::Add,
                            {in_edge(1, pdequant_other, 0)}, "padd");
                    padd_subgraph->create_input_port(0, padd, 0);
                    padd_subgraph->create_input_port(1, pdequant_other, 0);
                    padd_subgraph->create_output_port(0, padd, 0);
                    auto pbinary = pgraph->append_optional(
                            padd_subgraph, {in_edge(0, ppool, 0)}, "pbinary");

                    pgraph->append_op(impl::op_kind::Quantize,
                            {in_edge(0, pbinary, 0)}, "pquantize");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::int8_pool_binary);
                    fused_op->set_attr<std::string>(op_attr::backend, "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
