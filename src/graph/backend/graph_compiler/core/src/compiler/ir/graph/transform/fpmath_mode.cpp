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

#include "transform.hpp"
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/visitor.hpp>
#include <ops/fusible/unary_elemwise.hpp>
#include <runtime/env_vars.hpp>
#include <unordered_map>

SC_MODULE(fpmath_mode)

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

bool down_converted_op_exit(const context_ptr &ctx, const sc_op_ptr &node,
        const sc_data_type_t &target_type,
        std::vector<int> &onedim_const) { // whether the op support
    // converting to the target type
    if (node->isa<input_op>()) return false;
    if (node->isa<cast_op_t>()) return false;
    // avoid cast epsilon to bf16
    if (node->isa<constant_op_t>()) {
        auto const_op = node->dyn_cast<constant_op_t>();
        if (const_op->get_constant_plain_dims() == sc_dims {1}) {
            onedim_const.push_back(node->logical_op_id_);
            return false;
        }
    };
    // check inputs data types.
    for (auto &in : node->get_inputs()) {
        if (std::find(onedim_const.begin(), onedim_const.end(),
                    in->producer_owner_->logical_op_id_)
                != onedim_const.end())
            return false;
        if (!utils::is_one_of(in->details_.dtype_, datatypes::f32, target_type))
            return false;
    }
    // checkout outputs data types.
    for (auto &out : node->get_outputs()) {
        if (!utils::is_one_of(
                    out->details_.dtype_, datatypes::f32, target_type))
            return false;
    }
    return true;
}

bool check_is_quantized(sc_graph_t &graph) {
    for (auto &op : graph.ops_) {
        if (op->op_name_.find("quantize") != std::string::npos) return true;
    }
    return false;
}

void fpmath_mode(sc_graph_t &graph, const context_ptr &ctx) {
    // checks for open fpmath_mode
    int fpmath_mode_attr
            = graph.attrs_.get_or_else(sc_graph_t::attr_key_t::fpmath_mode, 0);
    int fpmath_mode_env = utils::getenv_int("SC_FPMATH_MODE", 0);
    if (!fpmath_mode_attr && !fpmath_mode_env) return;
    if (!ctx->machine_.cpu_flags_.fAVX512AMXBF16
            && !ctx->machine_.cpu_flags_.fAVX512BF16)
        return;
    if (check_is_quantized(graph)) return;
    auto vis = op_visitor_t::bfs_topology_sort(graph.ops_.size());
    std::unordered_map<int, sc_data_type_t> dtype_backup;
    std::vector<int> onedim_const;
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (node->isa<output_op>()) {
            bool downconvert_output
                    = dtype_backup.count(
                              node->get_inputs()[0]
                                      ->producer_owner_->logical_op_id_)
                    > 0;
            if (downconvert_output) {
                // add the cast op: BF16 -> F32
                sc_op_ptr cast_node = graph.make("cast", node->get_inputs(), {},
                        {{"dtype", datatypes::f32}});
                node->replace_input(0, cast_node->get_outputs()[0]);
            }
        } else {
            if (down_converted_op_exit(
                        ctx, node, datatypes::bf16, onedim_const)) {
                for (auto in_id = 0; in_id < int(node->get_inputs().size());
                        ++in_id) {
                    if (node->get_inputs()[in_id]->details_.dtype_
                            == datatypes::f32) {
                        // add the cast op: F32 -> BF16
                        auto cur_owner_op
                                = node->get_inputs()[in_id]->producer_owner_;
                        auto is_constant_op
                                = (cur_owner_op->isa<constant_op_t>()
                                        || cur_owner_op->attrs_.get_or_else(
                                                "constant",
                                                const_kind::not_const));
                        sc_op_ptr cast_node = graph.make("cast",
                                {node->get_inputs()[in_id]}, {},
                                {{"dtype", datatypes::bf16}});
                        cast_node->get_outputs()[0]->details_.dtype_
                                = datatypes::bf16;
                        if (is_constant_op) {
                            cast_node->attrs_.set(
                                    "constant", const_kind::local_const);
                        }
                        node->replace_input(in_id, cast_node->get_outputs()[0]);
                    }
                }
                // tunable ops can infer output type automatically, do not
                // need to set out_dtype
                if (!node->get_outputs().empty()
                        && !node->isa<tunable_op_t>()) {
                    if (node->get_outputs()[0]->details_.dtype_
                            == datatypes::f32) {
                        dtype_backup[node->logical_op_id_]
                                = node->get_outputs()[0]->details_.dtype_;
                        for (auto &out : node->get_outputs()) {
                            out->details_.dtype_ = datatypes::bf16;
                        }
                    }
                }
            } else if (!node->isa<input_op>() && !node->isa<cast_op_t>()) {
                for (auto in_id = 0; in_id < int(node->get_inputs().size());
                        ++in_id) {
                    if (node->get_inputs()[in_id]->details_.dtype_
                            == datatypes::bf16) {
                        // add the cast op: BF16 -> F32
                        auto cur_owner_op
                                = node->get_inputs()[in_id]->producer_owner_;
                        auto is_constant_op
                                = (cur_owner_op->isa<constant_op_t>()
                                        || cur_owner_op->attrs_.get_or_else(
                                                "constant",
                                                const_kind::not_const));
                        sc_op_ptr cast_node = graph.make("cast",
                                {node->get_inputs()[in_id]}, {},
                                {{"dtype", datatypes::f32}});
                        cast_node->get_outputs()[0]->details_.dtype_
                                = datatypes::f32;
                        if (is_constant_op) {
                            cast_node->attrs_.set(
                                    "constant", const_kind::local_const);
                        }
                        node->replace_input(in_id, cast_node->get_outputs()[0]);
                    }
                }
            }
        }
    });
    graph.reset_op_ids();
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
