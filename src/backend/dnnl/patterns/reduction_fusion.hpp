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
#ifndef BACKEND_DNNL_PATTERNS_REDUCTION_FUSION_HPP
#define BACKEND_DNNL_PATTERNS_REDUCTION_FUSION_HPP

#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "backend/dnnl/internal_ops.hpp"
#include "backend/dnnl/patterns/transformation_pattern.hpp"
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
 * \brief This provides reduction fusion.
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */
DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(reduction_fusion)

#define ADD_POST_OP_PATTERN_FUNC(akind) \
    [](const std::shared_ptr<pb_graph_t> &pgraph) -> void { \
        pm::pb_op *reduction = pgraph->append_op((akind)); \
        reduction->append_decision_function([](op_t *graph_op) -> bool { \
            if (graph_op->has_attr("axes") \
                    && graph_op->get_attr<std::vector<int64_t>>("axes") \
                               .empty()) \
                return false; \
            return true; \
        }); \
        pm::pb_op *add = pgraph->append_op( \
                impl::op_kind::Add, in_edges_t {in_edge(0, reduction, 0)}); \
        add->allow_internal_inputs({0, 1}); \
    }

#define RELU_POST_OP_PATTERN_FUNC(akind) \
    [](const std::shared_ptr<pb_graph_t> &pgraph) -> void { \
        pm::pb_op *reduction = pgraph->append_op((akind)); \
        reduction->append_decision_function([](op_t *graph_op) -> bool { \
            if (graph_op->has_attr("axes") \
                    && graph_op->get_attr<std::vector<int64_t>>("axes") \
                               .empty()) \
                return false; \
            return true; \
        }); \
        pgraph->append_op( \
                impl::op_kind::ReLU, in_edges_t {in_edge(0, reduction, 0)}); \
    }

#define FUSED_PATTERN_FUNC(akind) \
    []() -> std::shared_ptr<op_t> { \
        std::shared_ptr<op_t> fused_op \
                = std::make_shared<op_t>(op_kind::reduction_fusion); \
        fused_op->set_attr<int64_t>( \
                "alg_kind", static_cast<int64_t>((akind))); \
        fused_op->set_attr<std::string>("backend", "dnnl"); \
        return fused_op; \
    }

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, reducel1_add_fusion)
        .set_priority(8.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                ADD_POST_OP_PATTERN_FUNC(impl::op_kind::ReduceL1))
        .set_attr<FCreateV2FusedOp>("FCreateV2FusedOp",
                FUSED_PATTERN_FUNC(impl::op_kind::ReduceL1));

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, reducel1_relu_fusion)
        .set_priority(8.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                RELU_POST_OP_PATTERN_FUNC(impl::op_kind::ReduceL1))
        .set_attr<FCreateV2FusedOp>("FCreateV2FusedOp",
                FUSED_PATTERN_FUNC(impl::op_kind::ReduceL1));

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, reducel2_add_fusion)
        .set_priority(8.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                ADD_POST_OP_PATTERN_FUNC(impl::op_kind::ReduceL2))
        .set_attr<FCreateV2FusedOp>("FCreateV2FusedOp",
                FUSED_PATTERN_FUNC(impl::op_kind::ReduceL2));

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, reducel2_relu_fusion)
        .set_priority(8.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                RELU_POST_OP_PATTERN_FUNC(impl::op_kind::ReduceL2))
        .set_attr<FCreateV2FusedOp>("FCreateV2FusedOp",
                FUSED_PATTERN_FUNC(impl::op_kind::ReduceL2));

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, reducemax_add_fusion)
        .set_priority(8.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                ADD_POST_OP_PATTERN_FUNC(impl::op_kind::ReduceMax))
        .set_attr<FCreateV2FusedOp>("FCreateV2FusedOp",
                FUSED_PATTERN_FUNC(impl::op_kind::ReduceMax));

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, reducemax_relu_fusion)
        .set_priority(8.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                RELU_POST_OP_PATTERN_FUNC(impl::op_kind::ReduceMax))
        .set_attr<FCreateV2FusedOp>("FCreateV2FusedOp",
                FUSED_PATTERN_FUNC(impl::op_kind::ReduceMax));

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, reducemean_add_fusion)
        .set_priority(8.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                ADD_POST_OP_PATTERN_FUNC(impl::op_kind::ReduceMean))
        .set_attr<FCreateV2FusedOp>("FCreateV2FusedOp",
                FUSED_PATTERN_FUNC(impl::op_kind::ReduceMean));

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, reducemean_relu_fusion)
        .set_priority(8.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                RELU_POST_OP_PATTERN_FUNC(impl::op_kind::ReduceMean))
        .set_attr<FCreateV2FusedOp>("FCreateV2FusedOp",
                FUSED_PATTERN_FUNC(impl::op_kind::ReduceMean));

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, reducemin_add_fusion)
        .set_priority(8.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                ADD_POST_OP_PATTERN_FUNC(impl::op_kind::ReduceMin))
        .set_attr<FCreateV2FusedOp>("FCreateV2FusedOp",
                FUSED_PATTERN_FUNC(impl::op_kind::ReduceMin));

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, reducemin_relu_fusion)
        .set_priority(8.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                RELU_POST_OP_PATTERN_FUNC(impl::op_kind::ReduceMin))
        .set_attr<FCreateV2FusedOp>("FCreateV2FusedOp",
                FUSED_PATTERN_FUNC(impl::op_kind::ReduceMin));

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, reduceprod_add_fusion)
        .set_priority(8.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                ADD_POST_OP_PATTERN_FUNC(impl::op_kind::ReduceProd))
        .set_attr<FCreateV2FusedOp>("FCreateV2FusedOp",
                FUSED_PATTERN_FUNC(impl::op_kind::ReduceProd));

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, reduceprod_relu_fusion)
        .set_priority(8.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                RELU_POST_OP_PATTERN_FUNC(impl::op_kind::ReduceProd))
        .set_attr<FCreateV2FusedOp>("FCreateV2FusedOp",
                FUSED_PATTERN_FUNC(impl::op_kind::ReduceProd));

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, reducesum_add_fusion)
        .set_priority(8.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                ADD_POST_OP_PATTERN_FUNC(impl::op_kind::ReduceSum))
        .set_attr<FCreateV2FusedOp>("FCreateV2FusedOp",
                FUSED_PATTERN_FUNC(impl::op_kind::ReduceSum));

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, reducesum_relu_fusion)
        .set_priority(8.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                RELU_POST_OP_PATTERN_FUNC(impl::op_kind::ReduceSum))
        .set_attr<FCreateV2FusedOp>("FCreateV2FusedOp",
                FUSED_PATTERN_FUNC(impl::op_kind::ReduceSum));

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
