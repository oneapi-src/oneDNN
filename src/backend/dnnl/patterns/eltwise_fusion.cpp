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
#include "backend/dnnl/patterns/transformation_pattern.hpp"
#include "backend/dnnl/patterns/utils.hpp"

#include "backend/dnnl/kernels/eltwise.hpp"
#include "backend/dnnl/kernels/large_partition.hpp"

#include "utils/pm/pbuilder.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pattern {

namespace pm = impl::utils::pm;

using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreateV2FusedOp = impl::pass::FCreateV2FusedOp;
using FCreateV2Pattern = impl::pass::FCreateV2Pattern;

/*!
 * \brief This provides eltwise-related fusion, i.e.
 *        relu-add fusion
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(eltwise_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, eltwise_binary_fusion)
        .set_priority(8.2f)
        .set_kind(impl::partition_kind::unary_post_ops)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *peltwise = pgraph->append_alternation(
                            get_unary_ops(), "peltwise");
                    auto pbinary_graph
                            = std::make_shared<pb_graph_t>("pbinary_graph");
                    pm::pb_op_t *pbinary_op = pbinary_graph->append_alternation(
                            get_binary_ops(), "pbinary_op");
                    pbinary_graph->create_input_port(0, pbinary_op, 0);
                    pbinary_graph->create_input_port(1, pbinary_op, 1);
                    pbinary_graph->create_output_port(0, pbinary_op, 0);

                    pgraph->append_repetition(pbinary_graph, {0, 0}, 1,
                            MAX_REPETITION,
                            in_edges_t {in_edge(0, peltwise, 0)},
                            "prepetition");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_eltwise_fwd>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, chained_relu_fusion)
        .set_priority(5.0f)
        .set_kind(impl::partition_kind::unary_post_ops)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto chained_relu = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *relu
                            = chained_relu->append_op(impl::op_kind::ReLU);
                    chained_relu->create_input_port(0, relu, 0);
                    chained_relu->create_output_port(0, relu, 0);

                    pgraph->append_repetition(
                            chained_relu, {0, 0}, 1, MAX_REPETITION);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
