/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#include <vector>
#include "../fusible_op.hpp"
#include "../graph_op.hpp"
#include "../pass/pass.hpp"
#include "../visitor.hpp"
#include "transform.hpp"
#include <compiler/ir/graph/quantization/quantize_op.hpp>
#include <ops/fusible/reduce.hpp>
#include <ops/reduce_mean.hpp>
#include <util/math_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

// Give annotation to graph_op which contains tunable op. The same tunable op
// with a given default config will perform differently in different graphs,
// considering the potential impact of fusion. Attributes are given in this pass
// to tell tunable op use proper config (or even format) accordingly. Currently
// only managed_matmul_core_op benefits from this pass.
void annotate_config(sc_graph_t &graph, const context_ptr &ctx) {
    // annotate graph with matmul+reduce fusion
    auto vis0 = op_visitor_t::bfs();
    bool is_int8 = false;
    vis0.visit_graph(graph, [&](op_visitor_t *vis0, const sc_op_ptr &node) {
        // consider mixed-datatype
        if (node->isa<quantize::dequantize_op_t>()) { is_int8 = true; }
        if (node->op_name_ == "matmul"
                || node->op_name_ == "managed_matmul_core") {
            // currently only consider static case
            if (node->is_dynamic()) { return; }
            size_t use_size = node->get_outputs()[0]->uses_.size();
            auto next_node
                    = node->get_outputs()[0]->uses_.at(0).second.get_shared();
            while (!next_node->isa<tunable_op_t>()
                    && !next_node->isa<output_op>()
                    && next_node->op_name_ != "matmul") {
                if (next_node->isa<reduce_op_t>()
                        || next_node->isa<reduce_mean_op_t>()
                        || next_node->op_name_ == "layernorm") {
                    auto data_dims
                            = node->get_inputs()[0]->details_.get_plain_dims();
                    auto weight_dims
                            = node->get_inputs()[1]->details_.get_plain_dims();
                    auto out_dims_size
                            = std::max(data_dims.size(), weight_dims.size());
                    bool transpose_b
                            = node->attrs_.get_or_else("transpose_b", false);
                    bool transpose_a
                            = node->attrs_.get_or_else("transpose_a", false);
                    if (next_node->attrs_.has_key("rd_axis")
                            && !next_node->attrs_.get_or_else(
                                    op_attr_key::break_pre_fuse, false)) {
                        std::vector<int> rd_axis
                                = next_node->attrs_.get<std::vector<int>>(
                                        "rd_axis");
                        if ((weight_dims.size() == 2 || data_dims.size() == 2)
                                && rd_axis.size() == 1
                                && rd_axis.at(0)
                                        == static_cast<int>(out_dims_size)
                                                - 1) {
                            auto K = transpose_b ? weight_dims.at(1)
                                                 : weight_dims.at(0);
                            auto M = transpose_a
                                    ? data_dims.at(1)
                                    : math_utils::get_dims_product(data_dims)
                                            / data_dims.back();
                            if (((K >= 640 && K < 4096) || M < 12288)
                                    && utils::is_one_of(
                                            node->get_inputs()[0]
                                                    ->details_.dtype_,
                                            datatypes::bf16, datatypes::f16)
                                    && !is_int8) {
                                next_node->attrs_.set(
                                        op_attr_key::break_pre_fuse, true);
                                break;
                            }
                        }
                        node->attrs_.set("post_rd_axis", rd_axis);
                        break;
                    }
                }
                if (next_node->attrs_.get_or_else(
                            op_attr_key::break_pre_fuse, false)) {
                    break;
                }
                if (next_node->get_outputs()[0]->uses_.size() > 1) { break; }
                if (next_node->isa<binary_elementwise_op_t>()
                        && !(next_node->get_inputs()[1]
                                        ->producer_owner_->isa<input_op>())
                        && node->get_inputs()[0]->details_.dtype_
                                == datatypes::f32
                        && !is_int8) {
                    // consider residual cases
                    break;
                }
                next_node = next_node->get_outputs()[0]
                                    ->uses_.at(0)
                                    .second.get_shared();
            }
        }
    });

    // annotate graph with matmul+binary(no broadcast)
    auto vis1 = op_visitor_t::bfs();
    vis1.visit_graph(graph, [&](op_visitor_t *vis1, const sc_op_ptr &node) {
        if (node->op_name_ == "matmul"
                || node->op_name_ == "managed_matmul_core") {
            size_t use_size = node->get_outputs()[0]->uses_.size();
            auto next_node
                    = node->get_outputs()[0]->uses_.at(0).second.get_shared();
            while (!next_node->isa<tunable_op_t>()
                    && !next_node->isa<output_op>()
                    && next_node->op_name_ != "matmul") {
                if (next_node->attrs_.get_or_else(
                            op_attr_key::break_pre_fuse, false)) {
                    break;
                }
                if (next_node->isa<binary_elementwise_op_t>()) {
                    node->attrs_.set("post_binary",
                            next_node->get_inputs()[0]
                                            ->details_.get_plain_dims()
                                            .size()
                                    == next_node->get_inputs()[1]
                                               ->details_.get_plain_dims()
                                               .size());
                    break;
                }
                if (next_node->get_outputs()[0]->uses_.size() > 1) { break; }
                next_node = next_node->get_outputs()[0]
                                    ->uses_.at(0)
                                    .second.get_shared();
            }
        }
    });
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
