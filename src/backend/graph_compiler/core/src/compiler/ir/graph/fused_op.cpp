/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

#include "fused_op.hpp"
#include <algorithm>
#include <atomic>
#include <utility>
#include "fusible_op_utils.hpp"
#include "fusion_mgr.hpp"
#include "lowering.hpp"
#include "outer_loop_generator.hpp"
#include "pass/pass.hpp"
#include "tunable_op.hpp"
#include "visitor.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/func_inline.hpp>
#include <compiler/ir/transform/loop_transform.hpp>
#include <compiler/ir/transform/scope_flatten.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <runtime/config.hpp>
#include <unordered_map>
#include <unordered_set>

namespace sc {

fusion_partition_t *fusion_partition_t::get_root() const {
    if (merged_to) {
        assert(ops.empty());
        return merged_to->get_root();
    }
    return const_cast<fusion_partition_t *>(this);
}

bool fusion_partition_t::is_ok_to_add(
        sc_op *op, const op_dep_matrix_t &g) const {
    if (merged_to) { return get_root()->is_ok_to_add(op, g); }
    for (auto &in : op->get_inputs()) {
        // if the input is in the partition, don't need to check
        if (contains(in->producer_owner_)) { continue; }
        // if "in" depends on an op in the partition
        if (main_tunable_op
                && g.lookup(main_tunable_op->logical_op_id_,
                           in->producer_owner_->logical_op_id_)
                        == 1) {
            return false;
        }
        for (auto &op_in_set : ops) {
            auto result = g.lookup(op_in_set->logical_op_id_,
                    in->producer_owner_->logical_op_id_);
            // if "in" depends on an op in the partition
            if (result == 1) { return false; }
        }
    }
    return true;
}

bool fusion_partition_t::contains(sc_op *op) const {
    if (merged_to) { return get_root()->contains(op); }
    return op == main_tunable_op.get()
            || ops.find(op->shared_from_this()) != ops.end();
}

// merge the ops in "other" to "this"
void fusion_partition_t::merge(const ptr &other) const {
    auto ths_root = get_root();
    auto other_root = other->get_root();
    if (ths_root == other_root) { return; }
    ths_root->ops.insert(other_root->ops.begin(), other_root->ops.end());
    assert(other_root->main_tunable_op == nullptr);
    other_root->ops.clear();
    other_root->merged_to = ths_root->shared_from_this();
}

namespace graph {
void mark_read_or_write_buffers(std::vector<expr> &args, bool is_read) {
    const char *name = is_read ? "read_buffer" : "write_buffer";
    for (auto &tsr : args) {
        tsr->attr()[name] = true;
    }
}

func_t create_func_decl_for_op(
        sc_op *op, std::vector<expr> &ins, std::vector<expr> &outs) {
    ins = graph::tensor_detail_to_ir_tensor("__ins_", op->get_inputs());
    outs = graph::tensor_detail_to_ir_tensor("__outs_", op->get_outputs());
    graph::mark_read_or_write_buffers(ins, true);
    graph::mark_read_or_write_buffers(outs, false);
    std::vector<expr> args = outs;
    args.insert(args.end(), ins.begin(), ins.end());
    auto func_name = op->op_name_;
    func_name += "__";
    func_name += std::to_string(op->logical_op_id_);
    auto func = builder::make_func(func_name, args,
            make_stmt<stmts_node_t>(std::vector<stmt>()), datatypes::boolean);
    // func->attr()["Gflop"] = gen_ptr->get_gflop();
    return func;
}
} // namespace graph
static std::atomic<int> out_idx(0);

void collect_shrinked_graph_lt_map(
        const sc_graph_t &graph, gt2gt_map &lt_map, int shrink_size) {
    op_visitor_t pre_vis = op_visitor_t::bfs();
    pre_vis.visit_graph(graph, [&](const sc_op_ptr &op) {
        if (op->isa<input_op>() || op->isa<output_op>()
                || op->isa<constant_op_t>()) {
            return;
        } else if (auto bw_op
                = op->dyn_cast<op_traits::batchwise_shrinkable_t>()) {
            bw_op->collect_shrinked_lt_map(shrink_size, lt_map);
        } else {
            COMPILE_ASSERT(0, "Unexpected op kind found: " << op->op_name_)
        }
    });
}

void collect_shrinked_graph_axes_map(
        const sc_graph_t &graph, gt2axes_map &axes_map, int shrink_size) {
    op_visitor_t pre_vis = op_visitor_t::bfs();
    pre_vis.visit_graph(graph, [&](const sc_op_ptr &op) {
        if (op->isa<input_op>() || op->isa<output_op>()
                || op->isa<constant_op_t>()) {
            return;
        } else if (auto bw_op
                = op->dyn_cast<op_traits::batchwise_shrinkable_t>()) {
            bw_op->collect_shrinked_axes_map(shrink_size, axes_map);
        } else {
            COMPILE_ASSERT(0, "Unexpected op kind found: " << op->op_name_)
        }
    });
}

// shrink graph by shrink size
sc_graph_t shrink_graph(const sc_graph_t &graph, gt2gt_map &lt_map) {
    sc_graph_t shrinked_graph;
    op_visitor_t vis(op_visitor_t::dequeue_selector,
            op_visitor_t::create_DAG_updater(graph.ops_.size()));
    std::unordered_map<sc_op_ptr, int> op_id_map;
    vis.visit_graph(graph, [&](const sc_op_ptr &node) {
        sc_op_ptr new_node;
        if (node->dyn_cast<input_op>()) {
            new_node = shrinked_graph.make_input(
                    {lt_map.get(node->get_outputs()[0])});
            new_node->attrs_ = node->attrs_;
        } else if (node->dyn_cast<output_op>()) {
            new_node = new_node = shrinked_graph.make_output(
                    {lt_map.get(node->get_inputs()[0])});
            new_node->attrs_ = node->attrs_;
        } else if (node->dyn_cast<constant_op_t>()) {
            new_node = node->dyn_cast<op_traits::copyable_t>()->copy(
                    {}, {lt_map.get(node->get_outputs()[0])}, shrinked_graph);
        } else if (auto bw_node
                = node->dyn_cast<op_traits::batchwise_shrinkable_t>()) {
            new_node = bw_node->bw_shrinked_copy(lt_map, shrinked_graph);
        }
        op_id_map[new_node] = node->logical_op_id_;
    });
    shrinked_graph.attrs_ = graph.attrs_;
    shrinked_graph.resort_op_ids(op_id_map);
    return shrinked_graph;
}

fusion_mgr_ptr shrink_fmgr(const fusion_mgr_ptr &fmgr, gt2gt_map &lt_map) {
    auto &graph = fmgr->get_graph();
    auto new_fmgr = std::make_shared<fusion_manager>();
    op_visitor_t vis(op_visitor_t::dequeue_selector,
            op_visitor_t::create_DAG_updater(graph.ops_.size()));
    vis.visit_graph(graph, [&](const sc_op_ptr &node) {
        sc_op_ptr new_node;
        if (node->dyn_cast<input_op>()) {
            new_node = new_fmgr->make_input(
                    {lt_map.get(node->get_outputs()[0])});
            new_node->attrs_ = node->attrs_;
        } else if (node->dyn_cast<output_op>()) {
            const auto &outtsr = lt_map.get(node->get_inputs()[0]);
            new_node = new_node = new_fmgr->make<output_op>(outtsr);
            new_node->attrs_ = node->attrs_;
        } else if (node->dyn_cast<constant_op_t>()) {
            new_node = node->dyn_cast<op_traits::copyable_t>()->copy({},
                    {lt_map.get(node->get_outputs()[0])},
                    new_fmgr->get_graph());
        } else if (auto bw_node
                = node->dyn_cast<op_traits::batchwise_shrinkable_t>()) {
            new_node = bw_node->bw_shrinked_copy(lt_map, new_fmgr->get_graph());
        }
    });
    new_fmgr->get_graph().attrs_ = graph.attrs_;
    return new_fmgr;
}

op_traits::post_fusion_acceptable_t *fused_op_t::get_main_op() const {
    COMPILE_ASSERT(main_op_.ops_.size() == 2, "Bad internal graph");
    auto op = main_op_.ops_[1]->dyn_cast<op_traits::post_fusion_acceptable_t>();
    COMPILE_ASSERT(op, "The main op is not post_fusion_acceptable_t");
    return op;
}

bool fused_op_t::is_valid(const context_ptr &ctx) {
    if (main_op_.empty()) { return true; }
    return main_op_.ops_[1]->is_valid(ctx);
}

std::shared_ptr<sc_graph_t> fused_op_t::get_graph_impl() {
    throw std::runtime_error("fused_op_t::get_graph Not implemented");
    return nullptr;
}

fused_op_t::fused_op_t(const std::string &name, sc_graph_t &&main_op,
        std::shared_ptr<fusion_manager> fuse_mgr,
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : mgr_(std::move(fuse_mgr)), main_op_(std::move(main_op)) {
    info_.inputs_ = ins;
    info_.outputs_ = outs;
    attrs_ = attrs;
    if (!main_op_.ops_.empty()) {
        attrs_.set("horizontal_merge",
                main_op_.ops_[1]->attrs_.get_or_else(
                        "horizontal_merge", horizontal_merge_type::no_merge));
    }
    op_name_ = name;
}

sc_op_ptr fused_op_t::copy(const std::vector<graph_tensor_ptr> &ins, // NOLINT
        const std::vector<graph_tensor_ptr> &outs, sc_graph_t &graph) {
    sc_graph_t new_main_op;
    fusion_mgr_ptr new_fmgr;
    if (!main_op_.ops_.empty()) {
        auto base_op = dynamic_cast<sc_op *>(get_main_op());
        auto copyable = base_op->dyn_cast<op_traits::copyable_t>();
        COMPILE_ASSERT(copyable, base_op->op_name_ << " is not copyable");

        auto dummy_inop = new_main_op.make_input(
                copy_logical_tsr(base_op->get_inputs()));
        auto copied = copyable->copy(dummy_inop->get_outputs(),
                copy_logical_tsr(base_op->get_outputs()), new_main_op);
        assert(new_main_op.ops_.size() == 2);
    }
    if (mgr_) new_fmgr = mgr_->copy();
    auto new_fused_op = std::make_shared<fused_op_t>(op_name_,
            std::move(new_main_op), new_fmgr,
            /*ins*/ ins,
            /*outs*/
            outs, attrs_);
    graph.add(new_fused_op);
    return new_fused_op;
}

static sc_dims get_common_dims(const sc_dims &A, const sc_dims &B) {
    sc_dims ret;
    size_t common_size = std::min(A.size(), B.size());
    for (size_t i = 0; i < common_size; i++) {
        if (A[i] != B[i]) return ret;
        ret.emplace_back(A[i]);
    }
    return ret;
}

sc_dims fused_op_t::get_bwise_fuse_shrink_dims() {
    sc_dims bw_dims;
    if (main_op_.ops_.empty()) {
        // outer_loop_generator
        auto base_inp = const_cast<sc_op *>(mgr_->get_first_input())
                                ->dyn_cast<fusible_op_t>();
        for (auto &user : base_inp->get_outputs()[0]->uses_) {
            if (auto bw_op
                    = user.second
                              ->dyn_cast<op_traits::batchwise_shrinkable_t>()) {
                bw_dims = bw_op->get_bwise_fuse_shrink_dims();
                break;
            }
        }
    } else {
        auto base_op = dynamic_cast<sc_op *>(get_main_op());
        if (auto bw_op
                = base_op->dyn_cast<op_traits::batchwise_shrinkable_t>()) {
            bw_dims = bw_op->get_bwise_fuse_shrink_dims();
        }
    }
    // double check fused graph
    sc_dims common_bw_dims = bw_dims;
    sc_dims common_no_strided_dims_pre, common_no_strided_dims_post,
            common_no_strided_dims;
    bool no_strided_dims_pre_init = false, no_strided_dims_post_init = false;
    for (auto &op : mgr_->get_graph().ops_) {
        if (common_bw_dims.empty()) break;
        if (op->isa<input_op>() || op->isa<output_op>()
                || op->isa<constant_op_t>())
            continue;
        else if (auto bw_op
                = op->dyn_cast<op_traits::batchwise_shrinkable_t>()) {
            auto cur_bw_dims = bw_op->get_bwise_fuse_shrink_dims();
            COMPILE_ASSERT(
                    !op->attrs_.has_key(op_attr_key::bwise_no_strided_dims)
                            || (op->attrs_.get_or_else(
                                        op_attr_key::bwise_break_pre_fuse,
                                        false)
                                    || op->attrs_.get_or_else(
                                            op_attr_key::bwise_break_post_fuse,
                                            false)),
                    "If the op has been set batch-wise no stride dims, it "
                    "means it "
                    "may cause bwise break fusion");
            if (op->isa<reorder_op_t>()
                    && op->attrs_.has_key(op_attr_key::bwise_no_strided_dims)) {
                sc_dims no_strided_dims = op->attrs_.get<sc_dims>(
                        op_attr_key::bwise_no_strided_dims);
                if (op->attrs_.get_or_else(
                            op_attr_key::bwise_break_post_fuse, false)) {
                    if (op->is_single_output_single_use()
                            && op->get_outputs()[0]
                                       ->uses_[0]
                                       .second->isa<output_op>()) {
                        attrs_.set(op_attr_key::bwise_break_post_fuse, true);
                    } else {
                        cur_bw_dims = no_strided_dims;
                    }
                    // cache common no strided dims for fall-back
                    if (!no_strided_dims_post_init) {
                        common_no_strided_dims_post = no_strided_dims;
                        no_strided_dims_post_init = true;
                    } else {
                        common_no_strided_dims_post = get_common_dims(
                                common_no_strided_dims_post, no_strided_dims);
                    }
                }
                if (op->attrs_.get_or_else(
                            op_attr_key::bwise_break_pre_fuse, false)) {
                    if (op->get_inputs()[0]->producer_owner_->isa<input_op>()) {
                        attrs_.set(op_attr_key::bwise_break_pre_fuse, true);
                    } else {
                        cur_bw_dims = no_strided_dims;
                    }
                    // cache common no strided dims for fall-back
                    if (!no_strided_dims_pre_init) {
                        common_no_strided_dims_pre = no_strided_dims;
                        no_strided_dims_pre_init = true;
                    } else {
                        common_no_strided_dims_pre = get_common_dims(
                                common_no_strided_dims_pre, no_strided_dims);
                    }
                }
                op->attrs_.remove(op_attr_key::bwise_no_strided_dims);
                if (op->attrs_.has_key(op_attr_key::bwise_break_pre_fuse))
                    op->attrs_.remove(op_attr_key::bwise_break_pre_fuse);
                if (op->attrs_.has_key(op_attr_key::bwise_break_post_fuse))
                    op->attrs_.remove(op_attr_key::bwise_break_post_fuse);
            }
            common_bw_dims = get_common_dims(common_bw_dims, cur_bw_dims);
        } else {
            common_bw_dims = {};
        }
    }
    // double-check bwise break fuse attr
    if (attrs_.get_or_else(op_attr_key::bwise_break_pre_fuse, false)
            && common_bw_dims.size() <= common_no_strided_dims_pre.size()) {
        if (attrs_.has_key(op_attr_key::bwise_break_pre_fuse)) {
            attrs_.remove(op_attr_key::bwise_break_pre_fuse);
        }
    }
    if (attrs_.get_or_else(op_attr_key::bwise_break_post_fuse, false)
            && common_bw_dims.size() <= common_no_strided_dims_post.size()) {
        if (attrs_.has_key(op_attr_key::bwise_break_post_fuse)) {
            attrs_.remove(op_attr_key::bwise_break_post_fuse);
        }
    }
    // set bwise_no_strided_dims if necessary
    if (attrs_.get_or_else(op_attr_key::bwise_break_pre_fuse, false)
            || attrs_.get_or_else(op_attr_key::bwise_break_post_fuse, false)) {
        COMPILE_ASSERT(no_strided_dims_pre_init || no_strided_dims_post_init,
                "pre/post no strided dims should be set")
        if (no_strided_dims_pre_init && no_strided_dims_post_init) {
            common_no_strided_dims = get_common_dims(
                    common_no_strided_dims_pre, common_no_strided_dims_post);
            common_no_strided_dims
                    = get_common_dims(common_bw_dims, common_no_strided_dims);
        } else if (no_strided_dims_pre_init) {
            common_no_strided_dims = get_common_dims(
                    common_bw_dims, common_no_strided_dims_pre);
        } else {
            common_no_strided_dims = get_common_dims(
                    common_bw_dims, common_no_strided_dims_post);
        }
        attrs_.set(op_attr_key::bwise_no_strided_dims, common_no_strided_dims);
    }
    return common_bw_dims;
}

sc_op_ptr fused_op_t::bw_shrinked_copy(
        gt2gt_map &bw_lt_map, sc_graph_t &shrinked_graph) {
    sc_graph_t new_main_op;
    fusion_mgr_ptr new_fmgr;
    auto ths = this;
    auto shrink_logical_tsr
            = [&bw_lt_map, &ths](const std::vector<graph_tensor_ptr> &old_gt) {
                  std::vector<graph_tensor_ptr> new_gt(old_gt.size());
                  std::transform(old_gt.begin(), old_gt.end(), new_gt.begin(),
                          [&bw_lt_map, &ths](const graph_tensor_ptr &gt) {
                              COMPILE_ASSERT(bw_lt_map.haskey(gt),
                                      ths->op_name_
                                              << ": new input graph tensor not "
                                                 "found in map");
                              return bw_lt_map.get(gt);
                          });
                  return new_gt;
              };
    if (!main_op_.ops_.empty()) {
        auto base_op = dynamic_cast<sc_op *>(get_main_op());
        COMPILE_ASSERT(base_op->isa<op_traits::batchwise_shrinkable_t>(),
                "Please check whether " << base_op->op_name_
                                        << " is the batchwise shrinkable op")
        auto dummy_inop = new_main_op.make_input(
                shrink_logical_tsr(base_op->get_inputs()));
        auto bw_op = base_op->dyn_cast<op_traits::batchwise_shrinkable_t>();
        auto new_base_op = bw_op->bw_shrinked_copy(bw_lt_map, new_main_op);

        assert(new_main_op.ops_.size() == 2);
    }
    new_fmgr = shrink_fmgr(mgr_, bw_lt_map);

    auto old_ins = ths->get_inputs(), old_out = ths->get_outputs();
    auto new_fused_ins = shrink_logical_tsr(old_ins),
         new_fused_out = shrink_logical_tsr(old_out);

    auto new_fused_op = std::make_shared<fused_op_t>(op_name_,
            std::move(new_main_op), new_fmgr,
            /*ins*/ new_fused_ins,
            /*outs*/
            new_fused_out, attrs_);
    shrinked_graph.add(new_fused_op);
    return new_fused_op;
}

void fused_op_t::collect_shrinked_lt_map(int bw_size, gt2gt_map &bw_lt_map) {
    std::vector<graph_tensor_ptr> fused_ins = get_inputs(),
                                  fused_out = get_outputs();
    size_t base_op_out_size = 0;
    if (!main_op_.ops_.empty()) {
        auto base_op = dynamic_cast<sc_op *>(get_main_op());
        COMPILE_ASSERT(base_op->isa<op_traits::batchwise_shrinkable_t>(),
                "Please check whether " << base_op->op_name_
                                        << " is the batchwise shrinkable op")
        if (auto bw_op
                = base_op->dyn_cast<op_traits::batchwise_shrinkable_t>()) {
            bw_op->collect_shrinked_lt_map(bw_size, bw_lt_map);
        }
        auto base_ins = base_op->get_inputs();
        auto base_out = base_op->get_outputs();
        base_op_out_size += base_out.size();
        for (size_t i = 0; i < base_ins.size(); i++) {
            COMPILE_ASSERT(
                    bw_lt_map.haskey(base_ins[i]), "Unexpected cases found");
            auto &plain_dims
                    = bw_lt_map.get(base_ins[i])->details_.get_plain_dims();
            op_traits::batchwise_shrinkable_t::record_shrinked_gt(
                    bw_lt_map, fused_ins[i], plain_dims);
        }
    }
    // collect fusion graph
    collect_shrinked_graph_lt_map(mgr_->get_graph(), bw_lt_map, bw_size);

    // collect fused_op self
    auto fmgr_inop = mgr_->get_graph().get_input_ops();
    auto fmgr_outop = mgr_->get_graph().get_output_ops();
    std::vector<graph_tensor_ptr> fmgr_ins(fmgr_inop.size()),
            fmgr_out(fmgr_outop.size());
    std::transform(fmgr_inop.begin(), fmgr_inop.end(), fmgr_ins.begin(),
            [&](const sc_op_ptr &op) { return op->get_outputs()[0]; });
    std::transform(fmgr_outop.begin(), fmgr_outop.end(), fmgr_out.begin(),
            [&](const sc_op_ptr &op) { return op->get_inputs()[0]; });
    for (size_t i = 0; i < fmgr_ins.size(); i++) {
        COMPILE_ASSERT(bw_lt_map.haskey(fmgr_ins[i]), "Unexpected cases found");
        auto &plain_dims
                = bw_lt_map.get(fmgr_ins[i])->details_.get_plain_dims();
        op_traits::batchwise_shrinkable_t::record_shrinked_gt(
                bw_lt_map, fused_ins[i + base_op_out_size], plain_dims);
    }
    for (size_t i = 0; i < fmgr_out.size(); i++) {
        COMPILE_ASSERT(bw_lt_map.haskey(fmgr_out[i]), "Unexpected cases found");
        auto &plain_dims
                = bw_lt_map.get(fmgr_out[i])->details_.get_plain_dims();
        op_traits::batchwise_shrinkable_t::record_shrinked_gt(
                bw_lt_map, fused_out[i], plain_dims);
    }
}

void fused_op_t::collect_shrinked_axes_map(
        int bw_size, gt2axes_map &bw_axes_map) {
    std::vector<graph_tensor_ptr> fused_ins = get_inputs(),
                                  fused_out = get_outputs();
    size_t base_op_out_size = 0;
    if (!main_op_.ops_.empty()) {
        auto base_op = dynamic_cast<sc_op *>(get_main_op());
        COMPILE_ASSERT(base_op->isa<op_traits::batchwise_shrinkable_t>(),
                "Please check whether " << base_op->op_name_
                                        << " is the batchwise shrinkable op")
        if (auto bw_op
                = base_op->dyn_cast<op_traits::batchwise_shrinkable_t>()) {
            bw_op->collect_shrinked_axes_map(bw_size, bw_axes_map);
        }
        auto base_ins = base_op->get_inputs();
        auto base_out = base_op->get_outputs();
        base_op_out_size += base_out.size();
        for (size_t i = 0; i < base_ins.size(); i++) {
            COMPILE_ASSERT(
                    bw_axes_map.haskey(base_ins[i]), "Unexpected cases found");
            op_traits::batchwise_shrinkable_t::record_shrinked_axes(
                    bw_axes_map, fused_ins[i], bw_axes_map.get(base_ins[i]));
        }
    }
    // collect fusion graph
    collect_shrinked_graph_axes_map(mgr_->get_graph(), bw_axes_map, bw_size);

    // collect fused_op self
    auto fmgr_inop = mgr_->get_graph().get_input_ops();
    auto fmgr_outop = mgr_->get_graph().get_output_ops();
    std::vector<graph_tensor_ptr> fmgr_ins(fmgr_inop.size()),
            fmgr_out(fmgr_outop.size());
    std::transform(fmgr_inop.begin(), fmgr_inop.end(), fmgr_ins.begin(),
            [&](const sc_op_ptr &op) { return op->get_outputs()[0]; });
    std::transform(fmgr_outop.begin(), fmgr_outop.end(), fmgr_out.begin(),
            [&](const sc_op_ptr &op) { return op->get_inputs()[0]; });
    for (size_t i = 0; i < fmgr_ins.size(); i++) {
        COMPILE_ASSERT(
                bw_axes_map.haskey(fmgr_ins[i]), "Unexpected cases found");
        op_traits::batchwise_shrinkable_t::record_shrinked_axes(bw_axes_map,
                fused_ins[i + base_op_out_size], bw_axes_map.get(fmgr_ins[i]));
    }
    for (size_t i = 0; i < fmgr_out.size(); i++) {
        COMPILE_ASSERT(
                bw_axes_map.haskey(fmgr_out[i]), "Unexpected cases found");
        op_traits::batchwise_shrinkable_t::record_shrinked_axes(
                bw_axes_map, fused_out[i], bw_axes_map.get(fmgr_out[i]));
    }
}

bool fused_op_t::compare_contents(const sc_op *other) const {
    if (!sc_op::compare_contents(other)) { return false; }
    if (auto other_fused = other->dyn_cast<const fused_op_t>()) {
        if (main_op_.empty() != other_fused->main_op_.empty()) { return false; }
        if (!main_op_.empty()) {
            auto mainop = dynamic_cast<sc_op *>(get_main_op());
            auto other_mainop
                    = dynamic_cast<sc_op *>(other_fused->get_main_op());
            if (!mainop->compare_contents(other_mainop)) { return false; }
        }
        return compare_graph(mgr_->get_graph(), other_fused->mgr_->get_graph());
    }
    return false;
}

// may need refactor when enable graph hash
size_t fused_op_t::hash_contents() const {
    size_t seed = 0;
    hash_combine(seed, sc_op::hash_contents());
    if (!main_op_.empty()) {
        auto mainop = dynamic_cast<sc_op *>(get_main_op());
        hash_combine(seed, mainop->hash_contents());
    }
    return seed;
}

ir_module_ptr fused_op_t::get_func(context_ptr ctx) {
    std::vector<sc_op_ptr> out_failed;
    auto ret = try_get_func(ctx, false, out_failed);
    COMPILE_ASSERT(ret && out_failed.empty(),
            "Fusion failed. Fallback not implemented");
    return ret;
}

ir_module_ptr fused_op_t::try_get_func(const context_ptr &ctx, bool just_check,
        std::vector<sc_op_ptr> &out_failed) {
    auto modu = std::make_shared<ir_module_t>(ctx);
    // if no base Op, do fusion on fusible ops
    if (main_op_.empty()) {
        auto &graph = mgr_->get_graph();
        op_dep_matrix_t dep(graph);
        // find correct base input idx for generator
        size_t inp_idx = 0;
        // collect ops which results to broadcast op
        std::vector<sc_op *> bc_dep_ops;
        for (auto &op : graph.ops_) {
            if (auto be_op = op->dyn_cast<binary_elementwise_op_t>()) {
                int bc_idx = be_op->get_broadcast_input();
                if (bc_idx >= 0) { bc_dep_ops.emplace_back(op.get()); }
            }
        }

        for (size_t i = 0; i < graph.get_input_ops().size(); i++) {
            auto ins = graph.get_input_ops()[i];
            // sucessfully found flag
            bool found = true;
            for (auto &op : bc_dep_ops) {
                if (dep.lookup(ins->logical_op_id_, op->logical_op_id_) == 1) {
                    if (auto be_op = op->dyn_cast<binary_elementwise_op_t>()) {
                        int bc_idx = be_op->get_broadcast_input();
                        // ensued by above check
                        COMPILE_ASSERT(bc_idx >= 0,
                                "Implicit broadcast op is expected.");
                        auto bc_arg = op->get_inputs()[bc_idx]->producer_owner_;
                        auto non_bc_arg
                                = op->get_inputs()[1 - bc_idx]->producer_owner_;
                        if (dep.lookup(ins->logical_op_id_,
                                    non_bc_arg->logical_op_id_)
                                        == 1
                                || ins.get() == non_bc_arg) {
                            continue;
                        }
                        if (dep.lookup(
                                    ins->logical_op_id_, bc_arg->logical_op_id_)
                                        == 1
                                || ins.get() == bc_arg) {
                            found = false;
                            break;
                        }
                    } else {
                        COMPILE_ASSERT(
                                0, "Unexpected Op named: " << op->op_name_);
                    }
                }
            }
            if (found) {
                inp_idx = i;
                break;
            }
        }
        // reset input_idx
        mgr_->put_input_first(
                graph.get_input_ops()[inp_idx]->dyn_cast<input_op>());
        outer_loop_generator_t gen(inp_idx);
        return try_lower_fusion_manager(
                ctx, &gen, this, mgr_.get(), true, just_check, out_failed);
    }
    auto mainop = dynamic_cast<tunable_op_t *>(get_main_op());
    COMPILE_ASSERT(mainop, "Expecting tunable_op");

    COMPILE_ASSERT(mainop->get_outputs().size() == 1,
            "Expecting single output tunable op");
    auto gen_ptr = mainop->create_generator();
    mainop->set_config_if_empty(ctx, gen_ptr.get());

    std::vector<expr> ins;
    // real_outs are the output tensors in the function arguments
    std::vector<expr> real_outs;
    auto func = graph::create_func_decl_for_op(this, ins, real_outs);
    assert(mainop->get_outputs().size() <= real_outs.size());
    assert(mainop->get_outputs().size() == 1);
    assert(keep_outputs_.size() == 1);
    // finds if an output can be computed in-place on an "input" of the fusion
    // graph
    auto inplacemap = mgr_->query_inplace();
    // outs are the output tensors for the original outputs of the main op
    std::vector<expr> outs;
    if (keep_outputs_[0]) {
        outs = {real_outs[0]};
    } else if (!inplacemap.at(0).empty() && inplacemap[0][0] == 0) {
        // a really naive inplace optimization. We currently only allow
        // output of fusion replace output of the original output of tunable
        // Op
        outs = {real_outs[0]};
    } else {
        // no in-place available, define the output args independently
        outs = graph::tensor_detail_to_ir_tensor(
                "__origouts_" + std::to_string(out_idx++),
                mainop->get_outputs());
        assert(outs.size() == 1);
    }
    auto main_op_input_size = mainop->get_inputs().size();
    COMPILE_ASSERT(get_inputs().size() >= main_op_input_size,
            "Input tsr number of Fused op should >= Input tsr number of "
            "original op");
    // additional inputs arg tensors for fusion
    std::vector<expr> additional_ins;
    std::vector<expr> origin_ins(ins.begin(), ins.begin() + main_op_input_size);
    for (size_t i = 0; i < get_inputs().size() - main_op_input_size; i++) {
        additional_ins.emplace_back(ins[i + main_op_input_size]);
    }

    // =======================
    // Start of building function body
    // =======================
    builder::ir_builder_t bld;
    bld.push_scope();
    // for each original outputs
    for (auto &local_out : outs) {
        // if the output is in final output args, do not need to define the
        // output buffer as local tensor
        bool need_define_local = true;
        for (auto &v : real_outs) {
            if (v.ptr_same(local_out)) {
                need_define_local = false;
                break;
            }
        }
        if (need_define_local) {
            local_out->attr()[tensor_shrinker_attrs::may_shrink] = true;
            bld.push_var_tensor_def(local_out);
        }
    }
    std::vector<for_loop> loops;
    bool status = gen_ptr->generate(ctx, mainop->get_config().data_.get(),
            mgr_.get(), origin_ins, outs, loops);
    assert(status);
    bld.push_returns(true);
    auto body = bld.pop_scope();

    // =======================
    // End of building function body
    // =======================
    fuse_state_t fstate;
    std::vector<expr> fuse_outs;
    if (keep_outputs_[0]) {
        assert(real_outs.size() > 1);
        fuse_outs = std::vector<expr>(real_outs.begin() + 1, real_outs.end());
    } else {
        fuse_outs = real_outs;
    }
    if (!just_check) { mgr_->transform_graph(ctx, true); }
    out_failed = mgr_->prepare_and_check(ctx, fstate);
    if (!out_failed.empty()) {
        mgr_->clear_anchor();
        return nullptr;
    }
    if (just_check) {
        mgr_->clear_anchor();
        return nullptr;
    }
    bool can_in_brg = ctx->flags_.brgemm_backend_ == scflags_t::brgemm_t::dnnl
            && mgr_->can_register_brgemm_fusion(body);
    if (!can_in_brg) { mgr_->break_brgemm_fusion(); }
    mgr_->commit(modu, fstate, fuse_outs, additional_ins);
    // register fusion in brgemm.
    if (can_in_brg) {
        body = mgr_->get_brgemm_fusion_register()
                       .remake_brgemm_intrinsic_by_fusion(body);
    }
    func->body_ = std::move(body);
    gen_ptr->schedule_loops(
            ctx, mainop->get_config().data_.get(), func->body_, loops);
    modu->add_func({func});
    modu->set_entry_func_idx(0);
    return modu;
}

std::shared_ptr<sc_graph_t> horizontal_fused_op_t::get_graph_impl() {
    throw std::runtime_error("horiaontal_fused_op::get_graph Not implemented");
    return nullptr;
}

horizontal_fused_op_t::horizontal_fused_op_t(const std::string &name,
        const std::vector<sc_op_ptr> &ops_to_merge,
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : ops_to_merge_(ops_to_merge) {
    info_.inputs_ = ins;
    info_.outputs_ = outs;
    op_name_ = name;
    attrs_ = attrs;
}

static std::vector<graph_tensor_ptr> select_graph_tensor_by_idx(
        const std::vector<graph_tensor_ptr> &ins, const std::vector<int> &idx) {
    std::vector<graph_tensor_ptr> outs;
    outs.reserve(idx.size());
    for (auto &i : idx) {
        outs.push_back(ins[i]);
    }
    return outs;
}

static std::vector<expr> select_expr_by_idx(
        const std::vector<expr> &ins, const std::vector<int> &idx) {
    std::vector<expr> outs;
    outs.reserve(idx.size());
    for (auto &i : idx) {
        outs.push_back(ins[i]);
    }
    return outs;
}

void horizontal_fused_op_t::schedule_loops(const stmt &body) {
    COMPILE_ASSERT(body.isa<stmts>(), "body has only one stmt.");
    scope_flatten(body.checked_as<stmts>(), -1);
    std::vector<stmt> &body_seq = body.checked_as<stmts>()->seq_;
    std::vector<for_loop> loops;
    std::vector<stmt> not_loops;
    stmt return_stmt;
    for (auto &st : body_seq) {
        if (st.isa<for_loop>()) {
            loops.push_back(st.checked_as<for_loop>());
        } else if (!st.isa<returns>()) {
            not_loops.push_back(st);
        } else {
            return_stmt = st;
        }
    }
    std::vector<stmt> new_seq(not_loops.begin(), not_loops.end());
    new_seq.insert(new_seq.end(), loops.begin(), loops.end());
    new_seq.push_back(return_stmt);
    body_seq = std::move(new_seq);
    COMPILE_ASSERT(loops.size() > 1,
            "No need to horizontal fuse as parallel loop number is less than "
            "2.");
    for (size_t i = 1; i < loops.size(); i++) {
        loops[0]->parallel_merge(body, loops[i]);
    }
}

ir_module_ptr horizontal_fused_op_t::get_func(context_ptr ctx) {
    auto modu = std::make_shared<ir_module_t>(ctx);
    std::vector<expr> ins, outs;
    auto func = graph::create_func_decl_for_op(this, ins, outs);
    func_inliner_t inliner;
    builder::ir_builder_t bld;
    bld.push_scope();
    for (auto &op : ops_to_merge_) {
        auto &ins_idx = op->attrs_.get<std::vector<int>>("op_ins_idx");
        auto &outs_idx = op->attrs_.get<std::vector<int>>("op_outs_idx");
        op->info_.inputs_ = select_graph_tensor_by_idx(info_.inputs_, ins_idx);
        op->info_.outputs_
                = select_graph_tensor_by_idx(info_.outputs_, outs_idx);
        auto mod_to_merge = op->get_func(ctx);
        auto &global_vars = mod_to_merge->get_module_vars();
        for (auto &def_v : global_vars) {
            modu->add_global_var(def_v);
        }
        auto f = mod_to_merge->get_entry_func();
        tensor_shrinker_t pass;
        f = std::const_pointer_cast<func_base>(pass(f));
        std::vector<expr> op_in_args = select_expr_by_idx(ins, ins_idx);
        std::vector<expr> op_out_args = select_expr_by_idx(outs, outs_idx);
        op_out_args.insert(
                op_out_args.end(), op_in_args.begin(), op_in_args.end());
        auto callf = make_expr<call_node>(f, op_out_args);
        inliner.inline_at(callf, bld.get_current_scope().body, 0, global_vars);
    }
    bld.push_returns(true);
    func->body_ = bld.pop_scope();
    schedule_loops(func->body_);
    modu->add_func({func});
    modu->set_entry_func_idx(0);
    return modu;
}

batchwise_fused_op_t::batchwise_fused_op_t(const std::string &name,
        const sc_dims &bw_dims, const sc_graph_t &graph,
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : bw_dims_(bw_dims) {
    info_.inputs_ = ins;
    info_.outputs_ = outs;
    bw_graph_ = copy_graph(graph);
    op_name_ = name;
    attrs_ = attrs;
}

ir_module_ptr batchwise_fused_op_t::get_func(context_ptr ctx) {
    gt2gt_map lt_map;
    gt2axes_map axes_map;
    collect_shrinked_graph_lt_map(bw_graph_, lt_map, bw_dims_.size());
    collect_shrinked_graph_axes_map(bw_graph_, axes_map, bw_dims_.size());
    auto sub_graph = shrink_graph(bw_graph_, lt_map);
    // print_graph(sub_graph, std::cout, 1);

    std::vector<sc_op_ptr> sub_args;
    auto orig_inp_ops = bw_graph_.get_input_ops(),
         orig_out_ops = bw_graph_.get_output_ops();
    auto inp_ops = sub_graph.get_input_ops(),
         out_ops = sub_graph.get_output_ops();
    sub_args.insert(sub_args.end(), out_ops.begin(), out_ops.end());
    sub_args.insert(sub_args.end(), inp_ops.begin(), inp_ops.end());
    auto sub_modu = lower_graph(ctx, sub_graph, sub_args);
    auto &sub_vars = sub_modu->get_module_vars();

    for (auto &f : sub_modu->get_contents()) {
        remove_parallel(f);
        f->attr().set("skip_trace", true);
    }
    // std::cout << sub_modu->get_entry_func() << "\n";
    std::vector<expr> ins, outs;
    auto func = graph::create_func_decl_for_op(this, ins, outs);
    auto func_body = builder::make_stmts_unattached({}).checked_as<stmts>();
    func->body_ = func_body;
    auto modu = ir_module_t::from_entry_func(ctx, func);
    modu->merge(*sub_modu);

    std::vector<expr> loop_vars;
    sc_dims loop_ranges(bw_dims_.size());
    for (size_t i = 0; i < loop_ranges.size(); i++) {
        loop_vars.emplace_back(builder::make_var(datatypes::index,
                std::string("__batchwise_iter_") + std::to_string(i)));
    }

    std::unordered_map<expr, expr> strided_in_tsr_map, strided_out_tsr_map;
    std::vector<expr> args(ins.size() + outs.size());
    auto transform_new_args = [&](const expr &tsr, const sc_op_ptr &op,
                                      bool is_output) {
        auto dims = get_expr_to_dims(tsr.checked_as<tensor>()->dims_);
        auto bw_axes = axes_map.get(
                is_output ? op->get_inputs()[0] : op->get_outputs()[0]);
        COMPILE_ASSERT(bw_axes.size() == bw_dims_.size(),
                "batchwise axes size should be equal to bw dims")
        std::vector<expr> offset(dims.size(), 0);
        constant_folder_t f;
        bool strided = false;
        sc_dims shrink_dims = dims;
        for (size_t i = 0; i < bw_dims_.size(); i++) {
            if (bw_axes[i] == -1) continue;
            if (shrink_dims[bw_axes[i]] == 1) {
                offset[bw_axes[i]] = 0;
            } else {
                shrink_dims[bw_axes[i]] /= bw_dims_[i];
                offset[bw_axes[i]]
                        = dim2unsigned(shrink_dims[bw_axes[i]]) * loop_vars[i];
            }
            if ((i > 0 && shrink_dims[bw_axes[i]] != 1)
                    || bw_axes[i] != static_cast<int>(i))
                strided = true;
        }
        if (strided) {
            auto &tsr_map
                    = is_output ? strided_out_tsr_map : strided_in_tsr_map;
            // TODO(xxx): consider making a strided tensor here
            auto shrinked_tsr = builder::make_tensor(std::string("strided_")
                            + (is_output ? std::string("out_")
                                         : std::string("in_"))
                            + std::to_string(tsr_map.size()),
                    dims_to_expr(shrink_dims),
                    tsr.checked_as<tensor>()->elem_dtype_);
            shrinked_tsr->attr().set("temp.bw_axes", bw_axes);
            tsr_map[tsr] = shrinked_tsr;
            return shrinked_tsr;
        } else {
            return builder::tensor_ptr(tsr, offset, {}, true);
        }
    };
    std::transform(outs.begin(), outs.end(), orig_out_ops.begin(), args.begin(),
            [&](const expr &tsr, const sc_op_ptr &out_op) {
                return transform_new_args(tsr, out_op, true);
            });

    std::transform(ins.begin(), ins.end(), orig_inp_ops.begin(),
            args.begin() + outs.size(),
            [&](const expr &tsr, const sc_op_ptr &in_op) {
                return transform_new_args(tsr, in_op, false);
            });

    auto declare_strided_tsr_ir
            = [](stmt &body, std::unordered_map<expr, expr> &strided_tsr_map) {
                  for (auto &m : strided_tsr_map) {
                      auto shrinked_tsr = m.second.checked_as<tensor>();
                      body.checked_as<stmts>()->seq_.emplace_back(
                              builder::make_var_tensor_def_unattached(
                                      shrinked_tsr));
                  }
              };

    auto gen_copy_strided_tsr_ir = [&ctx](stmt &body,
                                           std::unordered_map<expr, expr>
                                                   &strided_tsr_map,
                                           const std::vector<expr> &lpvars,
                                           bool orig2shrink) {
        for (auto &m : strided_tsr_map) {
            auto orig_tsr = m.first.checked_as<tensor>(),
                 shrinked_tsr = m.second.checked_as<tensor>();
            std::vector<expr> loop_idx, shrinked_idx, orig_idx;
            auto orig_dims = get_expr_to_dims(orig_tsr->dims_);
            auto shriked_dims = get_expr_to_dims(shrinked_tsr->dims_);
            COMPILE_ASSERT(shrinked_tsr->attr().has_key("temp.bw_axes"),
                    "bw axes could not be found");
            auto bw_axes = shrinked_tsr->attr().get<std::vector<int>>(
                    "temp.bw_axes");
            shrinked_tsr->attr().remove("temp.bw_axes");
            int step = static_cast<int>(ctx->get_max_vector_lanes(
                    shrinked_tsr->elem_dtype_.type_code_));
            bool vectorized = ((shriked_dims.back() % step == 0)
                                      && (shriked_dims.back() >= step))
                    && ((orig_dims.back() % step == 0)
                            && (orig_dims.back() >= step));
            for (size_t i = 0; i < shriked_dims.size(); i++) {
                loop_idx.emplace_back(builder::make_var(datatypes::index,
                        std::string("_strided_cpy_iter") + std::to_string(i)));
                shrinked_idx.emplace_back(loop_idx.back());
                auto iter = std::find(
                        bw_axes.begin(), bw_axes.end(), static_cast<int>(i));
                if (iter != bw_axes.end())
                    orig_idx.emplace_back(lpvars[iter - bw_axes.begin()]
                                    * dim2unsigned(shriked_dims[i])
                            + loop_idx.back());
                else
                    orig_idx.emplace_back(loop_idx.back());
            }
            auto shrink_indexing = builder::make_indexing(
                    shrinked_tsr, shrinked_idx, vectorized ? step : 1);
            auto orig_indexing = builder::make_indexing(
                    orig_tsr, orig_idx, vectorized ? step : 1);
            auto cur = builder::make_stmts_unattached(
                    {builder::make_assign_unattached(
                            orig2shrink ? shrink_indexing : orig_indexing,
                            orig2shrink ? orig_indexing : shrink_indexing)});
            for (int i = shriked_dims.size() - 1; i >= 0; i--) {
                cur = builder::make_for_loop_unattached(loop_idx[i], 0,
                        dim2unsigned(shriked_dims[i]),
                        (i == static_cast<int>(shriked_dims.size() - 1))
                                        && vectorized
                                ? step
                                : 1,
                        make_stmt<stmts_node_t>(std::vector<stmt> {cur}), true,
                        for_type::NORMAL);
            }
            body.checked_as<stmts>()->seq_.emplace_back(cur);
        }
    };

    func_inliner_t inliner;
    stmt cur = builder::make_stmts_unattached({});
    declare_strided_tsr_ir(cur, strided_in_tsr_map);
    declare_strided_tsr_ir(cur, strided_out_tsr_map);
    gen_copy_strided_tsr_ir(cur, strided_in_tsr_map, loop_vars, true);

    auto the_call = builder::make_call(sub_modu->get_entry_func(), args)
                            .checked_as<call>();
    inliner.inline_at(the_call, cur.checked_as<stmts>()->seq_,
            cur.checked_as<stmts>()->seq_.size(), sub_vars);
    gen_copy_strided_tsr_ir(cur, strided_out_tsr_map, loop_vars, false);
    // std::cout << cur << "\n";
    std::reverse_copy(bw_dims_.begin(), bw_dims_.end(), loop_ranges.begin());
    std::reverse(loop_vars.begin(), loop_vars.end());
    for (size_t i = 0; i < loop_ranges.size(); i++) {
        cur = builder::make_for_loop_unattached(loop_vars[i], 0,
                dim2unsigned(loop_ranges[i]), 1,
                make_stmt<stmts_node_t>(std::vector<stmt> {cur}), true,
                for_type::NORMAL);
    }
    schedule_loops(cur);
    func_body->seq_.emplace_back(cur);
    auto ret = builder::make_returns_unattached(true);
    func_body->seq_.emplace_back(ret);
    // std::cout << func << "\n";

    return modu;
}

void batchwise_fused_op_t::schedule_loops(const stmt &body) {
    COMPILE_ASSERT(body.isa<for_loop>(), "for loop node is expected");
    for_loop outer_most_loop = body.checked_as<for_loop>();
    outer_most_loop->kind_ = for_type::PARALLEL;
    assert(outer_most_loop.defined());
    const int run_threads = runtime_config_t::get().get_num_threads();
    for_loop cur_loop = outer_most_loop;
    std::vector<for_loop> loops;
    auto fused_number = 1;
    while (true) {
        fused_number *= (get_expr_as_int(cur_loop->iter_end_)
                - get_expr_as_int(cur_loop->iter_begin_));
        if (fused_number >= run_threads && (fused_number % run_threads) == 0)
            break;
        cur_loop = get_inner_for_loop(cur_loop.get());
        if (cur_loop.defined())
            outer_most_loop->fuse(cur_loop);
        else
            break;
    }
}

std::shared_ptr<sc_graph_t> batchwise_fused_op_t::get_graph_impl() {
    throw std::runtime_error("batchwise_fused_op_t::get_graph Not implemented");
    return nullptr;
}

} // namespace sc
