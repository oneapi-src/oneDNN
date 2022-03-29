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

#include "backend/dnnl/internal_ops.hpp"
#include "backend/dnnl/patterns/fusions.hpp"

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
 * \brief This provides GELU fusion.
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */
DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(gelu_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, gelu_fusion)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto pow = pgraph->append_op(impl::op_kind::Pow, "pow");
                    auto multiply_1 = pgraph->append_op(impl::op_kind::Multiply,
                            {in_edge(0, pow, 0)}, "multiply_1");
                    auto add_1 = pgraph->append_op(impl::op_kind::Add,
                            {in_edge(0, multiply_1, 0)}, "add_1");
                    auto multiply_2 = pgraph->append_op(impl::op_kind::Multiply,
                            {in_edge(0, add_1, 0)}, "multiply_2");
                    auto tanh = pgraph->append_op(impl::op_kind::Tanh,
                            {in_edge(0, multiply_2, 0)}, "tanh");
                    auto add_2 = pgraph->append_op(
                            impl::op_kind::Add, {in_edge(0, tanh, 0)}, "add_2");
                    auto multiply_3 = pgraph->append_op(impl::op_kind::Multiply,
                            {in_edge(0, add_2, 0)}, "multiply_3");
                    pgraph->append_op(impl::op_kind::Multiply,
                            {in_edge(0, multiply_3, 0)}, "multiply_4");
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto div = pgraph->append_op(impl::op_kind::Divide, "div");
                    auto erf = pgraph->append_op(
                            impl::op_kind::Erf, {in_edge(0, div, 0)}, "erf");
                    auto add = pgraph->append_op(
                            impl::op_kind::Add, {in_edge(0, erf, 0)}, "add");
                    auto multiply_1 = pgraph->append_op(impl::op_kind::Multiply,
                            {in_edge(0, add, 0)}, "multiply_1");
                    pgraph->append_op(impl::op_kind::Multiply,
                            {in_edge(0, multiply_1, 0)}, "multiply_2");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    auto fused_op = std::make_shared<op_t>(impl::op_kind::GELU);
                    fused_op->set_attr("backend", std::string("dnnl"));
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
