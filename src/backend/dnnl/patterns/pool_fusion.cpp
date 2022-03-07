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

using pattern = impl::pass::pattern;
using FCreatePattern = impl::pass::FCreatePattern;
using FCreateOptPattern = impl::pass::FCreateOptPattern;

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
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *ppool = pgraph->append_alternation(
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
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_avgpool_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *pool = apattern->create_op(impl::op_kind::AvgPool);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    pool->fill_and_connect_input(0, *dequant_data, 0);
                    quant->fill_and_connect_input(0, *pool, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_avgpool);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_avgpool_add_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_other
                            = apattern->create_op(impl::op_kind::Dequantize);

                    op_t *pool = apattern->create_op(impl::op_kind::AvgPool);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);

                    pool->fill_and_connect_input(0, *dequant_data, 0);
                    add->fill_and_connect_input(0, *pool, 0);
                    add->fill_and_connect_input(1, *dequant_other, 0);
                    quant->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_avgpool_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_maxpool_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *pool = apattern->create_op(impl::op_kind::MaxPool);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    pool->fill_and_connect_input(0, *dequant_data, 0);
                    quant->fill_and_connect_input(0, *pool, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_maxpool);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_maxpool_add_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_other
                            = apattern->create_op(impl::op_kind::Dequantize);

                    op_t *pool = apattern->create_op(impl::op_kind::MaxPool);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);

                    pool->fill_and_connect_input(0, *dequant_data, 0);
                    add->fill_and_connect_input(0, *pool, 0);
                    add->fill_and_connect_input(1, *dequant_other, 0);
                    quant->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_maxpool_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
