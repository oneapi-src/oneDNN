/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/quantization/quantize_info.hpp>
#include <compiler/ir/graph/quantization/quantize_op.hpp>
#include <compiler/ir/graph/traits.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <compiler/ir/graph/visitor.hpp>
#include <util/math_utils.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace quantize {
void calculate_op_compensation(sc_graph_t &mgr, const context_ptr &ctx) {
    if (!mgr.attrs_.get_or_else(sc_graph_t::attr_key_t::quantize, false))
        return;
    op_visitor_t vis = op_visitor_t::dfs();
    vis.visit_graph(mgr, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (auto qnode = node->dyn_cast<op_traits::may_quantize_t>()) {
            if (qnode->is_quantized_ && qnode->need_compensation_) {
                std::vector<std::pair<int, sc_op_weak_ptr_t>> uses;
                for (auto &out : node->get_outputs()) {
                    uses.insert(
                            uses.end(), out->uses_.begin(), out->uses_.end());
                }
                sc_op_ptr compensation_qnode
                        = qnode->do_compensations(mgr, ctx);
                auto new_out = compensation_qnode->get_outputs()[0];
                for (auto &use : uses) {
                    use.second->replace_input(use.first, new_out);
                }
                vis->update_state_for_visited(compensation_qnode);
            }
        }
/* I want to keep this comment now in case we do bias add in s32
 * dtype.*/
#if 0
            else if (auto bst_node = node->dyn_cast<broadcast_op_t>()) {
                if (bst_node->is_quantized_ && bst_node->need_compensation_) {
                    std::vector<float> data_scales
                            = bst_node->attrs_.get<std::vector<float>>(
                                    "data_scales");
                    std::vector<float> weight_scales
                            = bst_node->attrs_.get<std::vector<float>>(
                                    "weight_scales");
                    std::vector<float> mul_scales
                            = math_utils::vector_mul(data_scales,
                            weight_scales);
                    std::vector<union_val> union_scales(
                            mul_scales.begin(), mul_scales.end());
                    sc_op_ptr mul_scales_const = mgr.make("constant", {}, {},
                            {{"values", union_scales}, {"dtype",
                            datatypes::f32}});
                    sc_op_ptr bst_div = mgr.make("div",
                            {bst_node->get_inputs()[1],
                                    mul_scales_const->get_outputs()[0]},
                            {}, {});
                    sc_op_ptr bst_cast = mgr.make("cast",
                    bst_div->get_outputs(),
                            {}, {{"dtype", datatypes::s32}});
                    sc_op_ptr bst_new = mgr.make("broadcast",
                            {bst_node->get_inputs()[0],
                            bst_cast->get_outputs()[0]},
                            {}, node->attrs_);

                    bst_node->need_compensation_ = false;
                    bst_node->replace_uses_with_and_remove(bst_new);
                    vis->update_state_for_visited(bst_new);
                }
            }
        }
#endif
    });
    mgr.reset_op_ids();
}
} // namespace quantize
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
