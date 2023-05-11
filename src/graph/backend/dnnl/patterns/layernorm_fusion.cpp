/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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
#include "graph/backend/dnnl/kernels/layernorm.hpp"
#include "graph/backend/dnnl/patterns/fusions.hpp"
#include "graph/backend/dnnl/patterns/pattern_matcher_pass.hpp"
#include "graph/backend/dnnl/patterns/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

namespace pm = graph::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreatePattern = graph::pass::FCreatePattern;

/*!
 * \brief This provides layernorm-related fusion
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 * 
 * \brief This pattern can match the target graph as shown below:
 *  
 *           |                             
 *       layernorm          |              |     
 *           |          layernorm      layernorm              |     
 *        typecast  or      |      or      |        --->   layernorm
 *           |           typecast       quantize              |
 *        quantize          |              |
 *           |
 *  
 */
DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(layernorm_fusion)

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, layernorm_post_ops_fusion_cpu)
        .set_priority(8.2f)
        .set_kind(graph::partition_kind_t::misc_post_ops)
        .set_engine_kind(engine_kind::cpu)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *layernorm_base
                            = pgraph->append_op(graph::op_kind::LayerNorm);
                    layernorm_base->append_decision_function(
                            check_input_dtype_from_offset<impl::data_type::f32,
                                    1>);
                    // Alt0: Typecast + Quantize
                    auto ptcq_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *ptc
                            = ptcq_graph->append_op(graph::op_kind::TypeCast);
                    pm::pb_op_t *pquant
                            = ptcq_graph->append_op(graph::op_kind::Quantize,
                                    in_edges_t {in_edge(0, ptc, 0)});
                    pquant->append_decision_function(check_zps_values<0>);
                    ptcq_graph->create_input_port(0, ptc, 0);
                    ptcq_graph->create_output_port(0, pquant, 0);

                    // Alt1: Typecast
                    auto ptypecast_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *ptypecast = ptypecast_graph->append_op(
                            graph::op_kind::TypeCast);
                    // For layernorm+tc+quant case, if the quant's zp is not
                    // zero, then layernorm+tc will be matched and the tc+quant
                    // fusion will be broken. To avoid this, we make the
                    // layernorm+tc fusion only happen when tc's consumer is not
                    // quant.
                    ptypecast->append_decision_function([](op_t *op) -> bool {
                        auto &csms = op->get_output_value(0)->get_consumers();
                        return std::none_of(csms.begin(), csms.end(),
                                [](const graph::value_t::consumer_t &csm) {
                                    return csm.get_op().get_kind()
                                            == graph::op_kind::Quantize;
                                });
                    });
                    ptypecast_graph->create_input_port(0, ptypecast, 0);
                    ptypecast_graph->create_output_port(0, ptypecast, 0);

                    // Alt2: Quantize
                    auto pquantize_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquantize = pquantize_graph->append_op(
                            graph::op_kind::Quantize);
                    pquantize->append_decision_function(check_zps_values<0>);
                    pquantize_graph->create_input_port(0, pquantize, 0);
                    pquantize_graph->create_output_port(0, pquantize, 0);

                    // It will be mathced in priority order of Alt0, Alt1, Alt2.
                    pgraph->append_alternation(
                            {ptcq_graph, ptypecast_graph, pquantize_graph},
                            in_edges_t {in_edge(0, layernorm_base, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<layernorm_fwd_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
