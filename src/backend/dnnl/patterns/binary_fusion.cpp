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
 * \brief This provides binary-related fusion, i.e.
 *        binary-relu fusion
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */

DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(binary_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, reciprocal_multiply_fusion)
        .set_priority(8.2f)
        .set_kind(impl::partition_kind::binary_post_ops)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto reciprocal = pgraph->append_op(
                            impl::op_kind::Reciprocal, "reciprocal");
                    pgraph->append_op(impl::op_kind::Multiply,
                            in_edges_t {in_edge(0, reciprocal, 0)}, "multiply");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(impl::op_kind::Divide);
                    fused_op->set_attr<std::string>(op_attr::backend, "dnnl");
                    return fused_op;
                });

// TODO(zitian): wait for the implementation of comparison ops:
//      Gt, Ge, Le, Lt, Eq, Ne
DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, binary_post_ops_fusion)
        .set_priority(8.3f)
        .set_kind(impl::partition_kind::binary_post_ops)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto binary_op = pgraph->append_alternation(
                            {impl::op_kind::Add, impl::op_kind::Divide,
                                    impl::op_kind::Maximum,
                                    impl::op_kind::Minimum,
                                    impl::op_kind::Multiply,
                                    impl::op_kind::Subtract},
                            "binary_op");

                    auto post_subgraph
                            = std::make_shared<pb_graph_t>("post_subgraph");
                    auto alternative_post_op
                            = post_subgraph->append_alternation(
                                    {impl::op_kind::Abs, impl::op_kind::Add,
                                            impl::op_kind::Clamp,
                                            impl::op_kind::Divide,
                                            impl::op_kind::Elu,
                                            impl::op_kind::Exp,
                                            impl::op_kind::GELU,
                                            impl::op_kind::HardSwish,
                                            impl::op_kind::Log,
                                            impl::op_kind::Maximum,
                                            impl::op_kind::Minimum,
                                            impl::op_kind::Multiply,
                                            impl::op_kind::Pow,
                                            impl::op_kind::ReLU,
                                            impl::op_kind::Round,
                                            impl::op_kind::Sigmoid,
                                            impl::op_kind::SoftPlus,
                                            impl::op_kind::Sqrt,
                                            impl::op_kind::Square,
                                            impl::op_kind::Subtract,
                                            impl::op_kind::Tanh},
                                    "alternative_post_op");
                    post_subgraph->create_input_port(0, alternative_post_op, 0);
                    post_subgraph->create_output_port(
                            0, alternative_post_op, 0);

                    pgraph->append_repetition(post_subgraph, {0, 0}, 1,
                            MAX_REPETITION, {in_edge(0, binary_op, 0)}, "palt");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::binary_post_ops_fusion);
                    fused_op->set_attr<std::string>(op_attr::backend, "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
