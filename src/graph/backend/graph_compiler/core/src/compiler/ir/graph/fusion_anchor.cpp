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

#include "fusion_anchor.hpp"
#include "fusible_op_utils.hpp"
#include "mixed_partition.hpp"
#include "tunable_op.hpp"
#include <util/optional_find.hpp>

SC_MODULE(graph.fusion_anchor);

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

// get parent loop for the given node, if not found, return the top parent node
static stmt get_parent_loop_or_node(const stmt_c &node) {
    stmt cur_node = node.remove_const();
    stmt parent_node = get_parent_node(cur_node);
    while (parent_node.defined()) {
        if (parent_node.isa<for_loop>()) { return parent_node; }
        cur_node = parent_node;
        parent_node = get_parent_node(parent_node);
    }
    return cur_node;
}

stmt fuse_anchor_map_t::get_parent_loop() const {
    return get_parent_loop_or_node(anchor_position_);
}

void fuse_anchor_map_t::append_content(anchor_content_t content) {
    std::vector<anchor_content_t> contents;
    // if content is fusion anchor, need to append all of its contents
    if (auto fanchor = content.as_or_null<fuse_anchor_map_t *>()) {
        contents = (*fanchor)->get_contents();
    }
    contents.emplace_back(content);
    append_contents(contents, content_number_map_.size());

    // recursively attach to parent anchor
    auto root = this;
    while (root->parent_) {
        auto num_id = root->parent_->content_number_map_[root];
        root = root->parent_.get();
        root->append_contents(contents, num_id);
    }
}

void fuse_anchor_map_t::attach_parent_anchor(
        const std::shared_ptr<fuse_anchor_map_t> &parent,
        const std::shared_ptr<fuse_anchor_map_t> &repl_parent) {
    if (!parent) return;
    auto root = this;
    while (root->parent_ && (root->parent_ != repl_parent)) {
        COMPILE_ASSERT(root != root->parent_.get(),
                "Ring parent anchor relationship found");
        root = root->parent_.get();
    }
    if (root == parent.get()) return;
    root->parent_ = parent;
    parent->append_content(root);
}

void fuse_anchor_map_t::merge(const std::shared_ptr<fuse_anchor_map_t> &other) {
    fsmap_.datamap_.insert(
            other->fsmap_.datamap_.begin(), other->fsmap_.datamap_.end());
    blocked_gt_set_.insert(
            other->blocked_gt_set_.begin(), other->blocked_gt_set_.end());
    borrowed_fanchor_map_.insert(other->borrowed_fanchor_map_.begin(),
            other->borrowed_fanchor_map_.end());
    auto contents_size = content_number_map_.size();
    for (auto &cont_numb_pair : other->content_number_map_) {
        content_number_map_.insert(std::make_pair(
                cont_numb_pair.first, cont_numb_pair.second + contents_size));
    }
}

bool fuse_anchor_map_t::has_view_of(sc_op *op) {
    auto op_anchor = binded_mxp_->op_anchor_map_[op];
    if (!op_anchor) return true;
    while (true) {
        if (op_anchor.get() == this) return true;
        if (op_anchor->content_number_map_.find(this)
                == op_anchor->content_number_map_.end()) {
            if (op_anchor->parent_) {
                op_anchor = op_anchor->parent_;
            } else
                break;
        } else if (*utils::find_map_value(op_anchor->content_number_map_, this)
                            .get()
                > *utils::find_map_value(op_anchor->content_number_map_, op)
                           .get()) {
            return true;
        } else {
            return false;
        }
    }
    auto ths_root_scope = get_root()->get_parent_scope()->seq_;
    stmt ths_anchor_ss = get_root()->anchor_position_;
    auto ths_pos = std::find_if(ths_root_scope.begin(), ths_root_scope.end(),
            [&ths_anchor_ss](
                    const stmt &s) { return ths_anchor_ss.ptr_same(s); });
    auto op_anchor_loop = op_anchor->get_parent_loop();
    while (true) {
        auto op_anchor_pos = std::find_if(ths_root_scope.begin(),
                ths_root_scope.end(), [&op_anchor_loop](const stmt &s) {
                    return op_anchor_loop.ptr_same(s);
                });
        if (op_anchor_pos != ths_root_scope.end()) {
            return op_anchor_pos < ths_pos;
        }
        op_anchor_loop = get_parent_node(op_anchor_loop);
        if (!op_anchor_loop.defined()) return false;
    }
}

bool fuse_anchor_map_t::check_dep_for_op(const sc_op *op) {
    if (op->isa<input_op>() || op->isa<constant_op_t>()) return true;
    auto parti = binded_mxp_;
    auto dep_m = parti->dep_m_;
    for (auto &inp : op->get_inputs()) {
        // check dep ops
        for (auto &cur : parti->ops) {
            if (std::any_of(cur->get_outputs().begin(),
                        cur->get_outputs().end(),
                        [&parti](const graph_tensor_ptr &gt) {
                            return parti->is_parti_out(gt);
                        })
                    && (dep_m->lookup(cur.get(), inp->producer_owner_) == 1
                            || cur.get() == inp->producer_owner_)) {
                // this fanchor should has view of depent op
                if (!this->has_view_of(cur.get())) return false;
            }
        }
    }
    return true;
}

bool fuse_anchor_map_t::validate_input_for_op(
        const sc_op *op, const std::unordered_set<graph_tensor_ptr> &known_gt) {
    auto fanchor = this;
    return std::all_of(op->get_inputs().begin(), op->get_inputs().end(),
            [&fanchor, &known_gt](const graph_tensor_ptr &gt) {
                auto parti = fanchor->binded_mxp_;
                auto dep_op = gt->producer_owner_;
                // if the producer owner of gt is successfully inferred
                if (known_gt.find(gt) != known_gt.end()) return true;
                // if the producer owner of gt is excluded in current
                // partition
                if (!parti->contains(dep_op)) {
                    return fanchor->check_dep_for_op(dep_op);
                }

                // fanchor should has view of depent op
                if (!fanchor->has_view_of(dep_op)) return false;

                slice_range_list inferred_slice = fanchor->fsmap_.get(gt);
                fuse_anchor_map_t *cur = fanchor;

                COMPILE_ASSERT(!dep_op->get_inputs().empty(),
                        dep_op->op_name_ << " op has no input")
                // find the borrowed scope
                auto dep_inp = dep_op->get_inputs()[0];
                while (cur->parent_) {
                    cur = cur->parent_.get();
                    // if input is ready on cur anchor, its input slice range
                    // should not be empty
                    if ((!(dep_op->isa<tunable_op_t>()
                                 && parti->is_parti_inp(dep_inp))
                                && !cur->fsmap_.hasvalue(dep_inp))
                            || !cur->fsmap_.hasvalue(gt))
                        continue;
                    slice_range_list cur_slice = cur->fsmap_.get(gt);
                    auto res = cmp_slice_range(inferred_slice, cur_slice);
                    if (res != cmp_res::l_larger_r) {
                        fanchor->borrowed_fanchor_map_[gt]
                                = cur->shared_from_this();
                        return true;
                    }
                }
                return false;
            });
}

void fuse_anchor_map_t::forbid_op(
        const sc_op *op, const std::unordered_set<graph_tensor_ptr> &known_gt) {
    auto fanchor = this;
    std::for_each(op->get_inputs().begin(), op->get_inputs().end(),
            [&known_gt, &fanchor](const graph_tensor_ptr &gt) {
                if (known_gt.find(gt) == known_gt.end()) {
                    fanchor->fsmap_.datamap_.erase(gt.get());
                }
            });
    std::for_each(op->get_outputs().begin(), op->get_outputs().end(),
            [&known_gt, &fanchor](const graph_tensor_ptr &gt) {
                if (known_gt.find(gt) == known_gt.end()) {
                    fanchor->fsmap_.datamap_.erase(gt.get());
                    fanchor->blocked_gt_set_.insert(gt);
                }
            });
}

bool fuse_anchor_map_t::check_input_for_op(
        const sc_op *op, std::unordered_set<graph_tensor_ptr> &known_gt) {
    auto fanchor = this;
    bool input_blocked = false;
    for (auto &gt : op->get_inputs()) {
        if (blocked_gt_set_.find(gt) != blocked_gt_set_.end()) {
            std::for_each(op->get_outputs().begin(), op->get_outputs().end(),
                    [&fanchor](const graph_tensor_ptr &gt) {
                        fanchor->blocked_gt_set_.insert(gt);
                    });
            input_blocked = true;
            break;
        }
        if (fanchor->fsmap_.hasvalue(gt)) {
            if (fanchor->borrowed_fanchor_map_.find(gt)
                    != fanchor->borrowed_fanchor_map_.end())
                continue;
            known_gt.insert(gt);
        }
    }
    return !(input_blocked || known_gt.empty());
}

void fuse_iter_anchor_map_t::commit(const stmt &s) {
    if (cached_iter_anchor_.empty()) {
        if (dispatch_helper_.isa<stmts>()) {
            anchor_position_->seq_.insert(anchor_position_->seq_.end(),
                    dispatch_helper_.static_as<stmts>()->seq_.begin(),
                    dispatch_helper_.static_as<stmts>()->seq_.end());
        } else {
            anchor_position_->seq_.emplace_back(dispatch_helper_);
        }
    }
    // create cached_iter_anchor_ if necessary
    if (cached_iter_anchor_.size() < iter_size_) {
        stmts ss = s.isa<stmts>()
                ? s.static_as<stmts>()
                : builder::make_stmts_unattached({s}).checked_as<stmts>();
        anchor_position_->seq_.emplace_back(
                make_stmt<if_else_node_t>(iter_ == iter_cnt_, ss, stmt()));
        cached_iter_anchor_.emplace_back(ss);
    }
    // commit into cached_iter_anchor_
    else {
        auto cached_anchor = cached_iter_anchor_.at(iter_cnt_);
        if (s.isa<stmts>()) {
            cached_anchor->seq_.insert(cached_anchor->seq_.end(),
                    s.static_as<stmts>()->seq_.begin(),
                    s.static_as<stmts>()->seq_.end());
        } else {
            cached_anchor->seq_.emplace_back(s);
        }
    }
    iter_cnt_++;
    if (iter_cnt_ == iter_size_) iter_cnt_ = 0;
}

void fuse_grouped_anchor_map_t::commit(const stmt &s) {
    auto pos = anchor_position_->seq_.at(group_cnt_).checked_as<stmts>();
    if (s.isa<stmts>()) {
        pos->seq_.insert(pos->seq_.end(), s.static_as<stmts>()->seq_.begin(),
                s.static_as<stmts>()->seq_.end());
    } else {
        pos->seq_.emplace_back(s);
    }
    group_cnt_++;
    if (group_cnt_ == group_size_) group_cnt_ = 0;
}

stmt fuse_grouped_anchor_map_t::get_parent_loop() const {
    // use last group anchor which is always the largest anchor
    stmt cur_node = anchor_position_->seq_.front();
    if (group_size_ == 1) {
        return get_parent_loop_or_node(cur_node);
    } else {
        stmt common_parent;
        for (size_t i = 1; i < anchor_position_->seq_.size(); i++) {
            auto next_node = anchor_position_->seq_[i];
            common_parent = get_common_parent_node(cur_node, next_node);
            if (!common_parent.defined()) return common_parent;
            cur_node = common_parent;
        }
        return get_parent_loop_or_node(common_parent);
    }
}

void fusion_anchor_mgr_t::create_fusion_anchor(const slice_map &fsmap,
        const fuse_anchor_map_ptr &parent, bool is_input_anchor) {
    COMPILE_ASSERT(!fsmap.empty(), "fusion anchor init slice not found")
    // create anchor placeholder in IR
    auto bld = builder::get_current_builder();
    auto pos = bld->push_anchor();
    // append to fanchor list
    fanchor_list_.emplace_back(std::make_shared<fuse_anchor_map_t>(
            pos, fslice_map(fsmap), parent, is_input_anchor));
}

void fusion_anchor_mgr_t::create_fusion_anchor(const expr &iter_var,
        const slice_map &fsmap, const stmt &dispatch_helper,
        const fuse_anchor_map_ptr &parent, bool is_input_anchor) {
    // create anchor placeholder in IR
    auto bld = builder::get_current_builder();
    auto pos = bld->push_anchor();
    // append to fanchor list
    fanchor_list_.emplace_back(std::make_shared<fuse_iter_anchor_map_t>(
            iter_var, pos, fslice_map(fsmap), dispatch_helper, parent,
            is_input_anchor));
}

void fusion_anchor_mgr_t::create_fusion_anchor(int group_id,
        const slice_map &fsmap, const fuse_anchor_map_ptr &parent,
        bool is_input_anchor) {
    COMPILE_ASSERT(!fsmap.empty(), "grouped fusion anchor init slice not found")
    COMPILE_ASSERT(
            std::all_of(fsmap.begin(), fsmap.end(),
                    [](const std::pair<graph_tensor *, slice_range_list> &p) {
                        return p.second.size() == 1;
                    }),
            "all init slice size of grouped fusion anchor should be 1")
    // create anchor placeholder in IR
    auto bld = builder::get_current_builder();
    auto pos = bld->push_anchor();
    // search grouped anchor map
    auto res = grouped_id_map_.find(group_id);
    // if group id already exist, update cached grouped anchor
    if (res != grouped_id_map_.end()) {
        auto &group_anchor = res->second;
        auto &datamap = group_anchor->fsmap_.datamap_;
        // update fsmap
        for (auto &new_kv : fsmap) {
            auto gt = new_kv.first;
            auto slice = new_kv.second[0];
            auto cache_kv = datamap.find(gt);
            COMPILE_ASSERT(cache_kv != datamap.end(),
                    "grouped anchor must have same graph tensor key")
            // update cached slice range list
            cache_kv->second.emplace_back(slice);
        }
        // update anchor position
        group_anchor->anchor_position_->seq_.emplace_back(pos);
        // update group size
        group_anchor->group_size_++;
    } else {
        auto new_grouped_anchor = std::make_shared<fuse_grouped_anchor_map_t>(
                builder::make_stmts_unattached({pos}).checked_as<stmts>(),
                fslice_map(fsmap), parent, is_input_anchor);
        // append to fanchor list
        fanchor_list_.emplace_back(new_grouped_anchor);
        // cache to grouped anchor map
        grouped_id_map_.insert(std::make_pair(group_id, new_grouped_anchor));
    }
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
