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

#include "backend/dnnl/kernels/binary.hpp"
#include "backend/dnnl/patterns/fusions.hpp"
#include "backend/dnnl/patterns/transformation_pattern.hpp"
#include "backend/dnnl/patterns/utils.hpp"

#include "utils/pm/pbuilder.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pattern {

namespace pm = impl::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreatePattern = impl::pass::FCreatePattern;

/*!
 * \brief This provides binary-related fusion, i.e.
 *        binary-relu fusion
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(binary_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, reciprocal_multiply_fusion)
        .set_priority(8.2f)
        .set_kind(impl::partition_kind::binary_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto reciprocal = pgraph->append_op(
                            impl::op_kind::Reciprocal, "reciprocal");
                    pgraph->append_op(impl::op_kind::Multiply,
                            in_edges_t {in_edge(0, reciprocal, 0)}, "multiply");
                })
        .set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr { return std::make_shared<binary_t>(); });

// TODO(zitian): wait for the implementation of comparison ops:
//      Gt, Ge, Le, Lt, Eq, Ne
DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, binary_post_ops_fusion)
        .set_priority(8.3f)
        .set_kind(impl::partition_kind::binary_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto binary_op = pgraph->append_alternation(
                            get_binary_ops(), "binary_op");

                    auto post_subgraph
                            = std::make_shared<pb_graph_t>("post_subgraph");
                    auto alternative_post_op
                            = post_subgraph->append_alternation(
                                    get_unary_binary_ops(),
                                    "alternative_post_op");
                    post_subgraph->create_input_port(0, alternative_post_op, 0);
                    post_subgraph->create_output_port(
                            0, alternative_post_op, 0);

                    pgraph->append_repetition(post_subgraph, {0, 0}, 1,
                            MAX_REPETITION, {in_edge(0, binary_op, 0)}, "palt");
                })
        .set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr { return std::make_shared<binary_t>(); });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
