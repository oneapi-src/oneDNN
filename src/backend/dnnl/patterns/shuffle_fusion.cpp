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

#include "backend/dnnl/internal_ops.hpp"
#include "backend/dnnl/kernels/shuffle.hpp"
#include "backend/dnnl/patterns/fusions.hpp"
#include "backend/dnnl/patterns/transformation_pattern.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pattern {

namespace pm = impl::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph = pm::pb_graph_t;
using FCreateV2FusedOp = impl::pass::FCreateV2FusedOp;
using FCreateV2Pattern = impl::pass::FCreateV2Pattern;

/*!
 * \brief This provides shuffle-related fusion
 *        The process includes following steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(shuffle_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, shuffle_fusion)
        .set_priority(8.2f)
        .set_kind(impl::partition_kind::misc_post_ops)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op_t *reshape0
                            = pgraph->append_op(impl::op_kind::StaticReshape);
                    pm::pb_op_t *transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, reshape0, 0)});
                    pgraph->append_op(impl::op_kind::StaticReshape,
                            in_edges_t {in_edge(0, transpose, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<shuffle_fwd_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
