/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#ifndef BACKEND_DNNL_PATTERNS_SHUFFLE_FUSION_HPP
#define BACKEND_DNNL_PATTERNS_SHUFFLE_FUSION_HPP

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backend/dnnl/internal_ops.hpp"
#include "backend/dnnl/patterns/transformation_pattern.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pass {

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

DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(shuffle_fusion)

// verify whether original and new shapes are equal at all positions
// except at the 'channel' axis. The original shape is expected to be
// the one from the input or output of the whole pattern, while the
// new shape is the one coming from intermediate OPs.
#define VERIFY_RESHAPE(orig_shape, new_shape, c_axis) \
    if ((orig_shape).size() < 3 \
            || (orig_shape).size() + 1 != (new_shape).size()) \
        return false; \
    for (size_t i = 0, j = 0; i < (orig_shape).size(); ++i) { \
        if (i == (c_axis)) { \
            if ((orig_shape)[i] != (new_shape)[j] * (new_shape)[j + 1]) \
                return false; \
            j += 2; \
        } else { \
            if ((orig_shape)[i] != (new_shape)[j++]) return false; \
        } \
    }

// find the first axis at which dims0 and dims1 differ. This axis is
// expected to be a 'channel' axis.
#define FIND_FIRST_NOT_COMMON_DIM(dims0, dims1) \
    [](const dims &d0, const dims &d1) -> size_t { \
        const auto res = std::mismatch(d0.cbegin(), d0.cend(), d1.cbegin()); \
        const auto dist = std::distance(d0.cbegin(), res.first); \
        return dist; \
    }((dims0), (dims1))

// get n-th (marked by todo) predecessor OP. It is needed to compare
// the data between the OPs.
#define GET_BACK_N_OPS(agraph_op, todo) \
    [](op_t *o, int t) mutable -> std::pair<op_t *, bool> { \
        op_t *cur_op = o; \
        while (t != 0) { \
            const auto &input_val = cur_op->get_input_value(0); \
            if (!input_val->has_producer()) \
                return std::make_pair(cur_op, false); \
            cur_op = &input_val->get_producer(); \
            --t; \
        } \
        return std::make_pair(cur_op, true); \
    }((agraph_op), (todo))

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, shuffle_fusion)
        .set_priority(8.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *reshape0
                            = pgraph->append_op(impl::op_kind::StaticReshape);
                    reshape0->append_decision_function(
                            [](op_t *graph_op) -> bool {
                                const logical_tensor_t src_port
                                        = graph_op->get_input_value(0)
                                                  ->get_logical_tensor();
                                const auto lt_shape
                                        = impl::logical_tensor_wrapper_t(
                                                src_port)
                                                  .vdims();
                                const auto shape
                                        = graph_op->get_attr<dims>("shape");
                                const auto axis = FIND_FIRST_NOT_COMMON_DIM(
                                        lt_shape, shape);
                                if (axis >= lt_shape.size()) return false;
                                VERIFY_RESHAPE(lt_shape, shape, axis);

                                return true;
                            });

                    pm::pb_op *transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, reshape0, 0)});
                    transpose->append_decision_function([](op_t *graph_op)
                                                                -> bool {
                        const auto order = graph_op->get_attr<dims>("order");
                        if (order.size() < 3) return false;

                        int steps = 1;
                        const auto res = GET_BACK_N_OPS(graph_op, steps);
                        if (!res.second) return false;

                        const op_t *predecessor_op = res.first;
                        const auto lt_shape = impl::logical_tensor_wrapper_t(
                                predecessor_op->get_input_value(0)
                                        ->get_logical_tensor())
                                                      .vdims();
                        const auto shape
                                = predecessor_op->get_attr<dims>("shape");
                        const auto axis
                                = FIND_FIRST_NOT_COMMON_DIM(lt_shape, shape);
                        std::vector<dim> exp_o(order.size());
                        std::iota(exp_o.begin(), exp_o.end(), 0);
                        std::swap(exp_o[axis], exp_o[axis + 1]);
                        if (!std::equal(
                                    order.begin(), order.end(), exp_o.begin()))
                            return false;

                        return true;
                    });

                    pm::pb_op *reshape1
                            = pgraph->append_op(impl::op_kind::StaticReshape,
                                    in_edges_t {in_edge(0, transpose, 0)});
                    reshape1->append_decision_function([](op_t *graph_op)
                                                               -> bool {
                        const logical_tensor_t dst_port
                                = graph_op->get_output_value(0)
                                          ->get_logical_tensor();
                        int steps = 2;
                        const auto res = GET_BACK_N_OPS(graph_op, steps);
                        if (!res.second) return false;

                        const op_t *predecessor_op = res.first;
                        const auto src_shape = impl::logical_tensor_wrapper_t(
                                predecessor_op->get_input_value(0)
                                        ->get_logical_tensor())
                                                       .vdims();
                        const auto dst_shape
                                = impl::logical_tensor_wrapper_t(dst_port)
                                          .vdims();
                        if (src_shape.size() != dst_shape.size()) return false;

                        const auto axis = FIND_FIRST_NOT_COMMON_DIM(
                                src_shape, dst_shape);
                        if (axis != src_shape.size()) return false;

                        return true;
                    });
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::dnnl_shuffle);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
