/*******************************************************************************
 * Copyright 2022 Intel Corporation
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

#include "mixed_partition.hpp"
#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include "fusible_op.hpp"
#include "graph_op.hpp"
#include "pass/pass.hpp"
#include "transform/transform.hpp"
#include "utils.hpp"
#include "visitor.hpp"
#include <compiler/ir/graph/fused_op.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/loop_transform.hpp>
#include <compiler/ir/transform/scope_flatten.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <compiler/ir/visitor.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/reduce.hpp>
#include <runtime/config.hpp>
#include <unordered_map>
#include <unordered_set>

namespace sc {

SC_MODULE(graph.mixed_partition);

static constexpr const char *mixed_attr_key_partition
        = "mixed_fuse_op.partition";
static constexpr const char *mixed_attr_key_orig_op
        = "mixed_fuse_op.original_op";
static constexpr const char *mixed_attr_key_cut_buffer
        = "mixed_fuse_op.cut_buffer";

void do_mixed_partition(const context_ptr &ctx, sc_graph_t &graph);

// add ret to parent node of s.
static void set_parent_node(const stmt &s, const stmt &parent) {
    std::weak_ptr<stmt_base_t> owner = parent.impl;
    s->attr()["builder.parent_node"] = owner;
}

// This function can find the parent node in IR, if the node has no parent
// node, return itself.
static stmt get_parent_node(stmt node) {
    if (!node->attr().has_key("builder.parent_node")) return node;
    stmt parent {node->attr()["builder.parent_node"]
                         .get<std::weak_ptr<stmt_base_t>>()
                         .lock()};
    COMPILE_ASSERT(parent.defined(),
            "parent node could not be found for stmts: " << node);
    return parent;
}

/** This function will find the nearest parent 'for_loop' node body for tensor
 *  declaration. If no loop found, it will just push statements in outer-most
 *  stmt.
 *  @param anchor: tensor should belong to which anchor
 * */
static stmt get_parent_loop_body(stmt anchor) {
    auto node = std::move(anchor);
    while (node->attr().has_key("builder.parent_node")) {
        if (node.isa<for_loop>()) {
            auto ret = node.static_as<for_loop>();
            return ret->body_;
        }
        node = get_parent_node(node);
    }
    if (node.isa<for_loop>()) {
        auto ret = node.static_as<for_loop>();
        return ret->body_;
    }
    return node;
}

static tensor get_real_tensor(const expr &buffer) {
    COMPILE_ASSERT(buffer.isa<tensor>() || buffer.isa<tensorptr>(),
            "Only tensor or tensorptr is accepted")
    auto tsr = buffer;
    if (tsr.isa<tensorptr>()) {
        auto base = tsr.static_as<tensorptr>()->base_;
        COMPILE_ASSERT(base.isa<indexing>(),
                "tensor_ptr base should be indexing, but got: " << base);
        tsr = base.static_as<indexing>()->ptr_;
    }
    COMPILE_ASSERT(tsr.isa<tensor>(), "Tensor type is expected")
    return tsr.static_as<tensor>();
}

void mxp_replacer_t::replace_anchor(
        const std::vector<fuse_anchor_map_ptr> &fanchors) {
    auto replace_fsmap = [&](const fuse_anchor_map_ptr &cur) {
        for (auto &fs_pair : cur->fsmap_.datamap_) {
            for (auto &slice : fs_pair.second) {
                for (auto &range : slice) {
                    range.first = dispatch_impl(range.first);
                    range.second = dispatch_impl(range.second);
                }
            }
        }
    };
    for (auto &anmap : fanchors) {
        replace_fsmap(anmap);
    }
}

void mxp_buffer_allocator::allocate_buffer(sc_op *op) {
    graph::tensor_detail_to_ir_tensor(
            op->op_name_ + "_" + std::to_string(op->logical_op_id_) + "_ins_",
            op->get_inputs(), g2b_map_);

    // inter-op inplace inferring
    auto query_inpalce = [&](const graph_tensor_ptr &out,
                                 const graph_tensor_ptr &in) -> bool {
        return (!op->isa<tunable_op_t>()) && (in->uses_.size() == 1)
                && (out != in)
                && (out->details_.get_blocking_dims()
                        == in->details_.get_blocking_dims())
                && (out->details_.dtype_ == in->details_.dtype_)
                && (out->details_.get_format() == in->details_.get_format())
                && (!binded_mxp_->is_parti_inp(
                        in)) // inputs of partition should not be inplaced
                && (!in->producer_owner_->isa<tunable_op_t>())
                && (!(g2b_map_.get(in).isa<tensor>()
                        && g2b_map_.get(in)
                                   .static_as<tensor>()
                                   ->init_value_)); // TODO(XXX): inplace inited
        // tensor
    };

    for (auto &out : op->get_outputs()) {
        // query input
        for (auto &inp : op->get_inputs()) {
            if (query_inpalce(out, inp)) {
                g2b_map_.get(out) = g2b_map_.get(inp);
                break;
            }
        }
    }

    if (auto collc_op = op->dyn_cast<reduce_collect_op_t>()) {
        if (collc_op->is_place_holder_op()) {
            g2b_map_.get(op->get_outputs()[0])
                    = g2b_map_.get(op->get_inputs()[0]);
        }
    }

    if (auto tv_op = op->dyn_cast<tensor_view_op_t>()) {
        auto base_tsr = get_real_tensor(g2b_map_.get(op->get_inputs()[0]));
        g2b_map_.get(op->get_outputs()[0]) = builder::tensor_ptr(base_tsr,
                std::vector<expr>(base_tsr->dims_.size(), 0),
                dims_to_expr(tv_op->get_shapes()));
    }

    graph::tensor_detail_to_ir_tensor(
            op->op_name_ + "_" + std::to_string(op->logical_op_id_) + "_outs_",
            op->get_outputs(), g2b_map_);

    if (op->isa<reduce_compute_op_t>()) {
        auto buf = g2b_map_.get(op->get_outputs()[0]);
        COMPILE_ASSERT(buf.isa<tensor>(),
                "output of reduce_compute_op_t should be tensor type")
        buf.checked_as<tensor>()->init_value_
                = tensor_node::get_zero_tensor_initializer();
    }
}

std::tuple<std::vector<expr>, std::vector<expr>>
mxp_buffer_allocator::get_buffer(sc_op *op) {
    std::vector<expr> inputs, outputs;
    std::for_each(op->get_inputs().begin(), op->get_inputs().end(),
            [&](const graph_tensor_ptr &gt) {
                COMPILE_ASSERT(
                        g2b_map_.haskey(gt), "please allocate buffer first")
                inputs.emplace_back(g2b_map_.get(gt));
            });
    std::for_each(op->get_outputs().begin(), op->get_outputs().end(),
            [&](const graph_tensor_ptr &gt) {
                COMPILE_ASSERT(
                        g2b_map_.haskey(gt), "please allocate buffer first")
                outputs.emplace_back(g2b_map_.get(gt));
            });
    return std::make_tuple(inputs, outputs);
}

static cmp_res cmp_op_anchor(sc_op *op, fuse_anchor_map_ptr cur_anchor,
        fuse_anchor_map_ptr new_anchor) {
    cmp_res res = cmp_res::unknown;
    std::for_each(op->get_inputs().begin(), op->get_inputs().end(),
            [&](const graph_tensor_ptr &gt) {
                if (utils::is_one_of(res, cmp_res::unknown, cmp_res::equal)
                        && cur_anchor->fsmap_.haskey(gt)
                        && new_anchor->fsmap_.haskey(gt)) {
                    res = cmp_slice_range(cur_anchor->fsmap_.get(gt),
                            new_anchor->fsmap_.get(gt));
                }
            });
    COMPILE_ASSERT(res != cmp_res::unknown, "Unknown comparision result")
    return res;
}

void mxp_buffer_allocator::update_input_buffer_info(
        sc_op *op, mixed_parti_t *parti) {
    auto commited_anchor_map = parti->lookup_anchor_map(op);
    auto update_ins_tensor_info_by_gt = [&](const graph_tensor_ptr &gt) {
        auto buf = g2b_map_.get(gt);
        if (b2g_map_.find(buf) == b2g_map_.end()) b2g_map_[buf] = gt;
        auto tsr = get_real_tensor(buf);
        auto real_anchor_map
                = commited_anchor_map->borrowed_fanchor_map_.find(gt)
                        != commited_anchor_map->borrowed_fanchor_map_.end()
                ? commited_anchor_map->borrowed_fanchor_map_[gt]
                : commited_anchor_map;
        if (tsr_anch_map_.find(tsr) != tsr_anch_map_.end()) {
            COMPILE_ASSERT(b2g_map_.find(buf) != b2g_map_.end(),
                    "base tensor should be visited")
            auto cur_slice = tsr_anch_map_[tsr]->fsmap_.get(b2g_map_[buf]);
            auto new_slice = real_anchor_map->fsmap_.get(gt);
            auto res = cmp_slice_range(cur_slice, new_slice);
            if (res == cmp_res::l_less_r) {
                tsr_anch_map_[tsr] = real_anchor_map;
                if (real_anchor_map != commited_anchor_map) b2g_map_[buf] = gt;
            } else if (res == cmp_res::equal) {
                if (tsr_anch_map_[tsr] != real_anchor_map->parent_) {
                    tsr_anch_map_[tsr] = real_anchor_map;
                    if (real_anchor_map != commited_anchor_map)
                        b2g_map_[buf] = gt;
                }
            }
        } else {
            tsr_anch_map_[tsr] = real_anchor_map;
            if (real_anchor_map != commited_anchor_map) b2g_map_[buf] = gt;
        }
    };

    std::for_each(op->get_inputs().begin(), op->get_inputs().end(),
            [&](const graph_tensor_ptr &gt) {
                update_ins_tensor_info_by_gt(gt);
            });
}

void mxp_buffer_allocator::update_output_buffer_info(
        sc_op *op, mixed_parti_t *parti) {
    auto commited_anchor_map = parti->lookup_anchor_map(op);
    auto sub_of_commited_anchor_map
            = parti->lookup_sub_anchor_map(commited_anchor_map);

    auto update_outs_tensor_info_by_gt = [&](const graph_tensor_ptr &gt) {
        auto buf = g2b_map_.get(gt);
        if (b2g_map_.find(buf) == b2g_map_.end()) b2g_map_[buf] = gt;
        auto tsr = get_real_tensor(buf);
        if (tsr_anch_map_.find(tsr) != tsr_anch_map_.end()) {
            // do nothing
            return;
        } else {
            fuse_anchor_map_ptr min_anchor_map = nullptr;
            for (auto &sub_anchor : sub_of_commited_anchor_map) {
                if (!sub_anchor->fsmap_.haskey(gt)) continue;
                if (!min_anchor_map)
                    min_anchor_map = sub_anchor;
                else {
                    auto min_slice = min_anchor_map->fsmap_.get(gt);
                    auto cur_slice = sub_anchor->fsmap_.get(gt);
                    if (cmp_slice_range(min_slice, cur_slice)
                            == cmp_res::l_larger_r) {
                        min_anchor_map = sub_anchor;
                    }
                }
            }
            tsr_anch_map_[tsr]
                    = min_anchor_map ? min_anchor_map : commited_anchor_map;
        }
    };

    std::for_each(op->get_outputs().begin(), op->get_outputs().end(),
            [&](const graph_tensor_ptr &gt) {
                update_outs_tensor_info_by_gt(gt);
            });
}

void mxp_buffer_allocator::declare_and_shrink_tensor(
        std::unordered_map<sc_op *, fuse_anchor_map_ptr> &op_anchor_map_) {
    // define real tensor in fdata, and put it at the beginning of ss
    auto declare_tensor_ = [&](const expr &tsr,
                                   const fuse_anchor_map_ptr &fanchor) {
        auto gt = b2g_map_[tsr];
        auto max_slice_range_list = fanchor->fsmap_.get(gt);
        // recurrsively find parent fanchor
        stmts decl_body = get_parent_loop_body(fanchor->anchor_position_)
                                  .checked_as<stmts>();
        auto &ss = decl_body->seq_;
        ss.emplace(ss.begin(), builder::make_var_tensor_def_unattached(tsr));
    };

    auto set_shrink_info_ = [](const expr &buffer,
                                    const slice_range_list &range_list) {
        if (range_list.size() != 1) return;
        buffer->attr()[tensor_shrinker_attrs::should_shrink]
                = tensor_shrinker_t::shrink_info_t {
                        /*base*/ get_slice_idx(range_list[0]),
                        /*shape*/ get_slice_shape(range_list[0]), stmts()};
    };

    for (auto &tsr2def : tsr_anch_map_) {
        if (tsr2def.first->attr().has_key(mixed_attr_key_cut_buffer)) continue;
        declare_tensor_(tsr2def.first, tsr2def.second);
    }
    for (auto &buf2shr : b2g_map_) {
        auto buf = buf2shr.first;
        if (buf->attr().has_key(mixed_attr_key_cut_buffer)) continue;
        auto anch = tsr_anch_map_[get_real_tensor(buf)];
        set_shrink_info_(buf, anch->fsmap_.get(buf2shr.second));
    }
}

void mxp_buffer_allocator::merge(mxp_buffer_allocator &other,
        std::unordered_map<expr, expr> &buffer_map,
        const std::pair<fuse_anchor_map_ptr, fuse_anchor_map_ptr>
                &common_buffer_anchor_pair) {
    buffer_map.clear();
    auto common_buffer_anchor = common_buffer_anchor_pair.first,
         common_other_buffer_anchor = common_buffer_anchor_pair.second;
    for (auto &other_g2b : other.g2b_map_.datamap_) {
        // if other tensor has conflict in current tensor, redirect it to common
        // buffer anchor
        if (g2b_map_.haskey(other_g2b.first)) {
            auto existed_buf = g2b_map_.get(other_g2b.first);
            buffer_map[other_g2b.second] = existed_buf;
            if (binded_mxp_->is_parti_cut(other_g2b.first)
                    && other.binded_mxp_->is_parti_cut(other_g2b.first))
                continue;
            COMPILE_ASSERT(!common_buffer_anchor,
                    "Conflict buffer: "
                            << existed_buf
                            << " is detected but no common buffer anchor "
                               "is found for redirection")
            tsr_anch_map_[get_real_tensor(existed_buf)] = common_buffer_anchor;
        } else {
            auto &buffer = other.g2b_map_.get(other_g2b.first);
            g2b_map_.get(other_g2b.first) = buffer;
            if (other.b2g_map_.find(buffer) != other.b2g_map_.end()) {
                b2g_map_[buffer] = other.b2g_map_[buffer];
            }
            if (other.tsr_anch_map_.find(get_real_tensor(buffer))
                    != other.tsr_anch_map_.end()) {
                auto other_anchor
                        = other.tsr_anch_map_[get_real_tensor(buffer)];
                tsr_anch_map_[get_real_tensor(buffer)]
                        = (other_anchor == common_other_buffer_anchor)
                        ? common_buffer_anchor
                        : other_anchor;
            }
        }
    }
}

void mxp_buffer_allocator::clear() {
    binded_mxp_ = nullptr;
    g2b_map_.clear();
    tsr_anch_map_.clear();
    b2g_map_.clear();
}

void extract_anchor_from_fmgr_to_parti(fusion_manager *fmgr,
        mixed_parti_t *parti, std::vector<expr> ir_tsrs,
        std::vector<graph_tensor_ptr> gtsrs,
        const fuse_anchor_map_ptr &parent_fanchor) {
    auto anchor_map_list = fmgr->unpack_src_anchor();
    COMPILE_ASSERT(ir_tsrs.size() == gtsrs.size(),
            "IR tensor-to-graph tensor mapping is not expected")
    size_t extracted_anchor_size = 0;
    for (auto &anchor_map : anchor_map_list) {
        fslice_map fsmap;
        for (size_t i = 0; i < ir_tsrs.size(); i++) {
            if (anchor_map.second.find(ir_tsrs[i]) != anchor_map.second.end()) {
                fsmap.get(gtsrs[i])
                        = anchor_map.second.find(ir_tsrs[i])->second;
                // usually used for extract tunable op inner anchor, due to
                // current template style, it need to handle fsmap base offset
                if (parent_fanchor) {
                    COMPILE_ASSERT(parent_fanchor->fsmap_.haskey(gtsrs[i])
                                    && parent_fanchor->fsmap_.get(gtsrs[i])
                                                    .size()
                                            == 1,
                            "inherited fsmap should have recorded current gt "
                            "single "
                            "slice range")
                    auto inherited_slice_range
                            = parent_fanchor->fsmap_.get(gtsrs[i])[0];
                    auto inherited_offset
                            = get_slice_idx(inherited_slice_range);
                    size_t outer_anchor_loop_size = 0;
                    for (auto &off : inherited_offset) {
                        if (!off.isa<constant_c>()
                                || get_expr_as_int(off) != 0) {
                            outer_anchor_loop_size++;
                        } else {
                            break;
                        }
                    }
                    auto &cur_slice_range_list = fsmap.get(gtsrs[i]);
                    for (auto &cur_slice_range : cur_slice_range_list) {
                        COMPILE_ASSERT(cur_slice_range.size()
                                        >= outer_anchor_loop_size,
                                "inner anchor range should be larger than "
                                "outer anchor loop size")
                        for (size_t i = 0; i < outer_anchor_loop_size; i++) {
                            cur_slice_range[i].first = cur_slice_range[i].first
                                    + inherited_slice_range[i].first;
                        }
                    }
                }
            }
        }
        if (!fsmap.empty()) {
            parti->fanchors_.emplace_back(std::make_shared<fuse_anchor_map_t>(
                    anchor_map.first, fsmap, parent_fanchor));
            extracted_anchor_size++;
        }
    }
    if (anchor_map_list.size() != extracted_anchor_size)
        SC_MODULE_INFO
                << "Could not extract fusion anchor from fmgr, please check";
}

void search_op_anchor_in_parti(sc_op *op, mixed_parti_t *parti) {
    if (parti->merged_to) {
        search_op_anchor_in_parti(op, parti->get_root());
        return;
    }
    auto check_inp_anchor
            = [&parti](const sc_op *op, const fuse_anchor_map_ptr &fanchor,
                      std::unordered_set<graph_tensor_ptr> &known_gt) -> bool {
        return std::all_of(op->get_inputs().begin(), op->get_inputs().end(),
                [&fanchor, &parti, &known_gt](const graph_tensor_ptr &gt) {
                    if (!parti->contains(gt->producer_owner_)) return true;
                    if (known_gt.find(gt) != known_gt.end()) return true;
                    slice_range_list inferred_slice = fanchor->fsmap_.get(gt);
                    fuse_anchor_map_ptr cur = fanchor;
                    while (cur->parent_) {
                        auto cached_cur = cur;
                        cur = cur->parent_;
                        if (cur->content_number_map_.find(gt->producer_owner_)
                                != cur->content_number_map_.end()) {
                            COMPILE_ASSERT(cur->fsmap_.haskey(gt)
                                            && !cur->fsmap_.get(gt).empty(),
                                    "Could not find slice range for output of "
                                            << gt->producer_owner_)
                            COMPILE_ASSERT(cur->content_number_map_.find(
                                                   cached_cur.get())
                                            != cur->content_number_map_.end(),
                                    "Could not find sub fanchor in parent "
                                    "fanchor content")
                            if (cur->content_number_map_[gt->producer_owner_]
                                    > cur->content_number_map_[cached_cur
                                                                       .get()])
                                continue;
                            slice_range_list cur_slice = cur->fsmap_.get(gt);
                            auto res = cmp_slice_range(
                                    inferred_slice, cur_slice);
                            if (res != cmp_res::l_larger_r) {
                                fanchor->borrowed_fanchor_map_[gt] = cur;
                                return true;
                            }
                        }
                    }
                    return false;
                });
    };

    auto erase_and_block_gt =
            [](const sc_op *op, std::unordered_set<graph_tensor_ptr> &known_gt,
                    const fuse_anchor_map_ptr &fanchor) {
                std::for_each(op->get_inputs().begin(), op->get_inputs().end(),
                        [&known_gt, &fanchor](const graph_tensor_ptr &gt) {
                            if (known_gt.find(gt) == known_gt.end())
                                fanchor->fsmap_.datamap_.erase(gt.get());
                        });
                std::for_each(op->get_outputs().begin(),
                        op->get_outputs().end(),
                        [&fanchor](const graph_tensor_ptr &gt) {
                            fanchor->fsmap_.datamap_.erase(gt.get());
                            fanchor->blocked_gt_set_.insert(gt);
                        });
            };

    // querry current parallelism
    float current_parallelism
            = evaluate_loop_parallel_balance(parti->get_outer_loops());

    for (auto &fanchor : parti->fanchors_) {
        infer_status_map_t stat_map(parti->ctx_, false);
        std::unordered_set<graph_tensor_ptr> known_gt;
        if (parti->empty() && op->isa<reorder_op_t>()
                && !fanchor->fsmap_.haskey(op->get_inputs()[0])) {
            auto reo = op->stc_cast<reorder_op_t>();
            if (reo->get_output_format().is_vnni_format()) {
                auto output_dims
                        = op->get_outputs()[0]->details_.get_blocking_dims();
                COMPILE_ASSERT(output_dims.size() > 3,
                        "Unexpected VNNI format kind: "
                                << reo->get_output_format())
                std::vector<int> required_axes;
                for (size_t i = output_dims.size() - 3; i < output_dims.size();
                        i++) {
                    required_axes.emplace_back(i);
                }
                if (!slice_full_on_axes(output_dims,
                            fanchor->fsmap_.get(op->get_outputs()[0])[0],
                            required_axes)) {
                    stat_map.append_ops_by_status(op, infer_status_code::RETRY);
                    continue;
                }
            }
            fanchor->fsmap_.get(op->get_inputs()[0])
                    = slice_range_list {gen_slice_by_dims(
                            op->get_inputs()[0]->details_.get_blocking_dims())};
        } else {
            bool input_blocked = false;
            for (auto &gt : op->get_inputs()) {
                if (fanchor->blocked_gt_set_.find(gt)
                        != fanchor->blocked_gt_set_.end()) {
                    std::for_each(op->get_outputs().begin(),
                            op->get_outputs().end(),
                            [&fanchor](const graph_tensor_ptr &gt) {
                                fanchor->blocked_gt_set_.insert(gt);
                            });
                    input_blocked = true;
                    break;
                }
                if (fanchor->fsmap_.haskey(gt)
                        && !fanchor->fsmap_.get(gt).empty()) {
                    if (fanchor->borrowed_fanchor_map_.find(gt)
                            != fanchor->borrowed_fanchor_map_.end())
                        continue;
                    known_gt.insert(gt);
                }
            }
            if (input_blocked || known_gt.empty()) continue;
            // TODO(XXX): merge
            if (auto fusible = op->dyn_cast<fusible_op_t>()) {
                fusible->infer_slice_ranges(fanchor->fsmap_, stat_map);
            } else if (auto tunable = op->dyn_cast<tunable_op_t>()) {
                tunable->infer_slice_ranges(fanchor->fsmap_, stat_map);
            } else {
                COMPILE_ASSERT(0, "Unexpected op type found: " << op->op_name_)
            }
        }
        if (stat_map.is_ok()) {
            if (!check_inp_anchor(op, fanchor, known_gt)) {
                erase_and_block_gt(op, known_gt, fanchor);
                continue;
            }
            // check parallelism: it should be ensured new commited anchor would
            // not break parallelism
            if (parti->ctx_->flags_.use_cost_model_
                    && (parti->contain_tunable_op() || op->isa<tunable_op_t>())
                    && (current_parallelism > evaluate_loop_parallel_balance(
                                parti->get_outer_loops(fanchor)))) {
                erase_and_block_gt(op, known_gt, fanchor);
                continue;
            }
            fanchor->append_op(op);
            parti->set_anchor_for_op(op, fanchor);
            continue;
        }
        if (stat_map.is_fail()) {
            auto &fail_list
                    = stat_map.get_ops_by_status(infer_status_code::FAIL);
            if (fail_list.find(op->shared_from_this()) == fail_list.end()) {
                fanchor->append_op(op);
                parti->set_anchor_for_op(op, fanchor);
                continue;
            }
        }
        erase_and_block_gt(op, known_gt, fanchor);
    }
}

static std::string print_loops_range(const std::vector<for_loop> &loops) {
    std::stringstream os;
    int cnt = 0;
    for (auto &l : loops) {
        if (cnt != 0) { os << "X"; }
        os << (get_expr_as_int(l->iter_end_) - get_expr_as_int(l->iter_begin_));
        cnt++;
    }
    return os.str();
}

enum class parti_dep : int {
    no_dep = 0,
    l_dep_r = 1,
    r_dep_l = 2,
    inter_dep = 3,
};

/**
 * Check two partition dependency
 * */
parti_dep check_parti_dep(
        mixed_parti_t *A, mixed_parti_t *B, const op_dep_matrix_t &g) {
    auto A_ops = A->ops, B_ops = B->ops;
    bool A_dep_B = false, B_dep_A = false;
    for (auto &op_a : A_ops) {
        for (auto &op_b : B_ops) {
            auto dep_flag = g.lookup(op_a, op_b);
            if (dep_flag == 1)
                B_dep_A = true;
            else if (dep_flag == -1)
                A_dep_B = true;
        }
    }
    if (A_dep_B && !B_dep_A) {
        return parti_dep::l_dep_r;
    } else if (B_dep_A && !A_dep_B) {
        return parti_dep::r_dep_l;
    } else if (!A_dep_B && !B_dep_A) {
        return parti_dep::no_dep;
    } else {
        return parti_dep::inter_dep;
    }
}

/**
 * Check two partition connectionship
 * */
bool check_parti_connectionship(mixed_parti_t *A, mixed_parti_t *B) {
    auto A_ops = A->ops, B_ops = B->ops;
    for (auto &op_a : A_ops) {
        std::unordered_set<graph_tensor_ptr> gt_set;
        std::for_each(op_a->get_inputs().begin(), op_a->get_inputs().end(),
                [&gt_set](const graph_tensor_ptr &gt) { gt_set.insert(gt); });
        std::for_each(op_a->get_outputs().begin(), op_a->get_outputs().end(),
                [&gt_set](const graph_tensor_ptr &gt) { gt_set.insert(gt); });
        for (auto &op_b : B_ops) {
            for (auto &inp : op_b->get_inputs()) {
                if (gt_set.find(inp) != gt_set.end()) return true;
            }
            for (auto &out : op_b->get_outputs()) {
                if (gt_set.find(out) != gt_set.end()) return true;
            }
        }
    }
    return false;
}

bool try_merge_mixed_parti_horizontally(
        mixed_parti_t *A, mixed_parti_t *B, const op_dep_matrix_t &g) {
    if (A->get_root() == B->get_root()) return false;
    if (!A->func_.get() || !B->func_.get()) return false;
    if (!A->contain_tunable_op() || !B->contain_tunable_op()) return false;
    if (!check_parti_connectionship(A, B)) return false;
    if (check_parti_dep(A, B, g) != parti_dep::no_dep) return false;

    auto outer_loops_A = A->get_outer_loops(),
         outer_loops_B = B->get_outer_loops();
    if (outer_loops_A.empty() || outer_loops_B.empty()) return false;

    if (A->ctx_->flags_.use_cost_model_) {
        auto A_parallelism = evaluate_loop_parallel_balance(outer_loops_A),
             B_parallelism = evaluate_loop_parallel_balance(outer_loops_B);
        if (A_parallelism == 1.f || B_parallelism == 1.f) return false;
    }

    // evaluate two partition by cost model
    float score_A = A->evaluate_perf(), score_B = B->evaluate_perf();

    SC_MODULE_INFO << "horizontally merging two partition:";
    SC_MODULE_INFO << A->func_;
    SC_MODULE_INFO << B->func_;

    /* * * * * * * * * * * * * * * * *
     * Step 0: Fuse func_
     * * * * * * * * * * * * * * * * */
    std::unordered_map<expr, expr> expr_map;
    schedule_loop_body(A->func_->body_, &expr_map);
    schedule_loop_body(B->func_->body_, &expr_map);

    auto new_body = make_stmt<stmts_node_t>(
            std::vector<stmt> {A->func_->body_, B->func_->body_});

    /* * * * * * * * * * * * * * * * *
     * Step 1: Merge func_
     * * * * * * * * * * * * * * * * */
    A->func_->body_ = std::move(new_body);

    /* * * * * * * * * * * * * * * * *
     * Step 2: Merge fanchor_
     * * * * * * * * * * * * * * * * */
    A->fanchors_.insert(
            A->fanchors_.end(), B->fanchors_.begin(), B->fanchors_.end());

    /* * * * * * * * * * * * * * * * *
     * Step 3: Merge buffer_
     * * * * * * * * * * * * * * * * */
    std::unordered_map<expr, expr> buffer_map;
    A->buf_alloc_.merge(
            B->buf_alloc_, buffer_map, std::make_pair(nullptr, nullptr));

    /* * * * * * * * * * * * * * * * *
     * Step 4: Merge buffer_
     * * * * * * * * * * * * * * * * */
    expr_map.insert(buffer_map.begin(), buffer_map.end());
    mxp_replacer_t expr_reper(expr_map);
    // 1. func->body
    expr_reper.replace_func(A->func_);
    // 2. fanchor->fsmap->slice_range
    expr_reper.replace_anchor(A->fanchors_);

    /* * * * * * * * * * * * * * * * *
     * Step 5: Merge op_anchor_map_
     * * * * * * * * * * * * * * * * */
    A->op_anchor_map_.insert(
            B->op_anchor_map_.begin(), B->op_anchor_map_.end());

    // call base merge
    A->fusion_partition_t::merge(
            static_cast<fusion_partition_t *>(B)->shared_from_this());

    auto &body = A->func_->body_;
    /* * * * * * * * * * * * * * * * *
     * Step 6: Same to Horizontal Merge
     * * * * * * * * * * * * * * * * */
    COMPILE_ASSERT(body.isa<stmts>(), "body has only one stmt.");
    scope_flatten(body.checked_as<stmts>(), -1);
    std::vector<stmt> &body_seq = body.checked_as<stmts>()->seq_;
    std::vector<for_loop> loops;
    std::vector<stmt> not_loops;
    for (auto &st : body_seq) {
        if (st.isa<for_loop>()) {
            loops.push_back(st.checked_as<for_loop>());
        } else if (!st.isa<returns>()) {
            not_loops.push_back(st);
        }
    }
    std::vector<stmt> new_seq(not_loops.begin(), not_loops.end());
    new_seq.insert(new_seq.end(), loops.begin(), loops.end());
    body_seq = std::move(new_seq);
    COMPILE_ASSERT(loops.size() > 1,
            "No need to horizontal fuse as parallel loop number is less "
            "than "
            "2.");

    constant_folder_t cf;
    auto_caster_t ac;
    for (size_t i = 1; i < loops.size(); i++) {
        loops[0]->parallel_merge(body, loops[i]);
        set_parent_node(loops[i]->body_, loops[0]);
        loops[0]->iter_end_ = cf(ac(loops[0]->iter_end_)).remove_const();
    }

    A->func_->name_ += "_horizontal_merge_" + B->func_->name_;

    SC_MODULE_INFO << "horizontally merging result:";
    SC_MODULE_INFO << A->func_;

    // clear merged parti
    B->clear();

    float score_C = A->evaluate_perf();

    if (score_C < std::max(score_A, score_B)) {
        SC_MODULE_WARN << "Merging these two partition may cause performance "
                          "drop, no fall-back strategy found";
    }

    return true;
}

bool try_merge_mixed_parti_vertically(
        mixed_parti_t *A, mixed_parti_t *B, const op_dep_matrix_t &g) {
    if (A->get_root() == B->get_root()) return false;
    if (!A->func_.get() || !B->func_.get()) return false;
    auto dep_flag = check_parti_dep(A, B, g);
    // if two partition inter-depends each other, could not merge them
    if (dep_flag == parti_dep::inter_dep) return false;
    mixed_parti_t *pa_to_merge = (dep_flag == parti_dep::l_dep_r) ? B : A,
                  *parti_be_merged = (dep_flag == parti_dep::l_dep_r) ? A : B;
    auto outer_loops_to_merge = pa_to_merge->get_outer_loops(),
         outer_loops_be_merged = parti_be_merged->get_outer_loops();

    auto cmp_loop_range = [](const for_loop &A, const for_loop &B) -> int64_t {
        if (!(A->iter_begin_.isa<constant_c>() && A->iter_end_.isa<constant_c>()
                    && B->iter_begin_.isa<constant_c>()
                    && B->iter_end_.isa<constant_c>())) {
            return 0;
        }
        auto A_begin = get_expr_as_int(A->iter_begin_),
             A_end = get_expr_as_int(A->iter_end_),
             B_begin = get_expr_as_int(B->iter_begin_),
             B_end = get_expr_as_int(B->iter_end_);
        return (A_begin == B_begin && A_end == B_end) ? (A_end - A_begin) : 0;
    };

    sc_dims loop_range_vec;
    // great common size
    auto gcs = std::min(
            outer_loops_to_merge.size(), outer_loops_be_merged.size());
    size_t merged_loop_size = 0;
    for (; merged_loop_size < gcs; merged_loop_size++) {
        auto common_range
                = cmp_loop_range(outer_loops_to_merge[merged_loop_size],
                        outer_loops_be_merged[merged_loop_size]);
        if (!common_range) break;
        loop_range_vec.emplace_back(common_range);
    }

    if (!merged_loop_size) return false;
    if (pa_to_merge->ctx_->flags_.use_cost_model_) {
        auto new_parallelism = evaluate_loop_parallel_balance(loop_range_vec),
             to_merge_parallelism
                = evaluate_loop_parallel_balance(outer_loops_to_merge),
             be_merged_parallelism
                = evaluate_loop_parallel_balance(outer_loops_be_merged);
        if (new_parallelism < to_merge_parallelism
                || new_parallelism < be_merged_parallelism) {
            return false;
        }
    }

    // evaluate two partition by cost model
    float score_A = A->evaluate_perf(), score_B = B->evaluate_perf();

    SC_MODULE_INFO << "merging two partition:";
    SC_MODULE_INFO << A->func_;
    SC_MODULE_INFO << B->func_;

    /**
     * for_loop(){
     *   // body
     *   ...
     *   // anchor
     *   {}
     * }
     * */
    auto get_anchor_inside_loop
            = [](mixed_parti_t *parti, const for_loop &loop) -> stmts {
        auto &body = loop->body_;
        if (body.isa<stmts>()) {
            auto ss = body.static_as<stmts>();
            if (ss->seq_.size() == 1 && ss->seq_[0].isa<stmts>()) {
                auto inner_ss = ss->seq_[0].static_as<stmts>();
                auto anchor_map = parti->lookup_anchor_map(inner_ss);
                if (anchor_map) return inner_ss;
            } else if (ss->seq_.size() == 2 && ss->seq_[1].isa<stmts>()) {
                auto inner_ss = ss->seq_[1].static_as<stmts>();
                auto anchor_map = parti->lookup_anchor_map(inner_ss);
                if (anchor_map) return inner_ss;
            }
        }
        return stmts();
    };

    /* * * * * * * * * * * * * * * * *
     * Step 1: Merge func_
     * * * * * * * * * * * * * * * * */
    auto max_to_merge_anchor = get_anchor_inside_loop(
            pa_to_merge, outer_loops_to_merge[merged_loop_size - 1]);
    auto max_be_merged_anchor = get_anchor_inside_loop(
            parti_be_merged, outer_loops_be_merged[merged_loop_size - 1]);
    auto max_to_merge_anchor_map
            = pa_to_merge->lookup_anchor_map(max_to_merge_anchor);
    auto max_be_merged_anchor_map
            = parti_be_merged->lookup_anchor_map(max_be_merged_anchor);
    auto max_be_merged_ss = outer_loops_be_merged[merged_loop_size - 1]->body_;
    auto max_to_merge_ss = outer_loops_to_merge[merged_loop_size - 1]->body_;
    // insert be_merged_ss to the back of to_merged_ss
    if (max_to_merge_anchor_map) {
        max_to_merge_anchor_map->commit_stmt(max_be_merged_ss);
        // redirect parent node
        set_parent_node(max_be_merged_ss, max_to_merge_anchor);
    } else {
        if (dep_flag != parti_dep::no_dep) return false;
        max_to_merge_ss.checked_as<stmts>()->seq_.emplace_back(
                max_be_merged_ss);
        // redirect parent node
        set_parent_node(max_be_merged_ss, max_to_merge_ss);
    }

    // var and tensor replace map
    std::unordered_map<expr, expr> expr_map, buffer_map;

    /* * * * * * * * * * * * * * * * *
     * Step 2: Merge fanchor_
     * * * * * * * * * * * * * * * * */
    // erase inferred but not allocated gt
    for (auto &to_merge_anchor_map : pa_to_merge->fanchors_) {
        for (auto iter = to_merge_anchor_map->fsmap_.datamap_.begin();
                iter != to_merge_anchor_map->fsmap_.datamap_.end();) {
            if (!pa_to_merge->buf_alloc_.g2b_map_.haskey(iter->first)) {
                iter = to_merge_anchor_map->fsmap_.datamap_.erase(iter);
            } else {
                iter++;
            }
        }
    }

    for (auto &be_merged_anchor_map : parti_be_merged->fanchors_) {
        for (auto iter = be_merged_anchor_map->fsmap_.datamap_.begin();
                iter != be_merged_anchor_map->fsmap_.datamap_.end();) {
            if (!parti_be_merged->buf_alloc_.g2b_map_.haskey(iter->first)) {
                iter = be_merged_anchor_map->fsmap_.datamap_.erase(iter);
            } else {
                iter++;
            }
        }
    }

    // merge outer loop anchor
    std::unordered_set<fuse_anchor_map_ptr> be_merged_anchor_masks;
    for (size_t i = 0; i < merged_loop_size; i++) {
        expr_map[outer_loops_be_merged[i]->var_]
                = outer_loops_to_merge[i]->var_;
        auto be_merged_anchor = get_anchor_inside_loop(
                     parti_be_merged, outer_loops_be_merged[i]),
             to_merge_anchor
                = get_anchor_inside_loop(pa_to_merge, outer_loops_to_merge[i]);
        fuse_anchor_map_ptr be_merged_anchor_map, to_merge_anchor_map;
        if (be_merged_anchor.defined()) {
            be_merged_anchor_map
                    = parti_be_merged->lookup_anchor_map(be_merged_anchor);
            be_merged_anchor_masks.insert(be_merged_anchor_map);
        }
        if (to_merge_anchor.defined()) {
            to_merge_anchor_map
                    = pa_to_merge->lookup_anchor_map(to_merge_anchor);
        }
        if (be_merged_anchor_map && to_merge_anchor_map) {
            to_merge_anchor_map->merge(be_merged_anchor_map);
        }
    }

    // append inner loop anchor
    for (auto &be_merged_anchor_map : parti_be_merged->fanchors_) {
        if (be_merged_anchor_masks.find(be_merged_anchor_map)
                != be_merged_anchor_masks.end())
            continue;
        if (max_to_merge_anchor_map) {
            be_merged_anchor_map->attach_parent_anchor(
                    max_to_merge_anchor_map, max_be_merged_anchor_map);
        }
        pa_to_merge->fanchors_.emplace_back(be_merged_anchor_map);
    }

    /* * * * * * * * * * * * * * * * *
     * Step 3: Merge buf_alloc_
     * * * * * * * * * * * * * * * * */
    pa_to_merge->buf_alloc_.merge(parti_be_merged->buf_alloc_, buffer_map,
            std::make_pair(max_to_merge_anchor_map, max_be_merged_anchor_map));

    /* * * * * * * * * * * * * * * * * *
     * Step 4: Replace IR node involving:
     *  1. func->body
     *  2. fanchor->fsmap->slice_range
     * * * * * * * * * * * * * * * * * */
    expr_map.insert(buffer_map.begin(), buffer_map.end());
    // create mxp inplace replacer
    mxp_replacer_t expr_reper(expr_map);
    // 1. func->body
    expr_reper.replace_func(pa_to_merge->func_);
    // 2. fanchor->fsmap->slice_range
    expr_reper.replace_anchor(pa_to_merge->fanchors_);

    /* * * * * * * * * * * * * * * * *
     * Step 5: Merge op_anchor_map_
     * * * * * * * * * * * * * * * * */
    pa_to_merge->op_anchor_map_.insert(parti_be_merged->op_anchor_map_.begin(),
            parti_be_merged->op_anchor_map_.end());

    // erase joint op in op_anchor_map
    for (auto iter = pa_to_merge->op_anchor_map_.begin();
            iter != pa_to_merge->op_anchor_map_.end();) {
        if (!pa_to_merge->contains(iter->first)) {
            iter = pa_to_merge->op_anchor_map_.erase(iter);
        } else {
            iter++;
        }
    }

    /* * * * * * * * * * * * * * * * *
     * Step 6: Merge op_
     * * * * * * * * * * * * * * * * */
    // call base merge
    pa_to_merge->fusion_partition_t::merge(
            static_cast<fusion_partition_t *>(parti_be_merged)
                    ->shared_from_this());

    pa_to_merge->func_->name_ += "_merge_" + parti_be_merged->func_->name_;

    SC_MODULE_INFO << "Merging result:";
    SC_MODULE_INFO << pa_to_merge->func_;

    // clear merged parti
    parti_be_merged->clear();

    float score_C = pa_to_merge->evaluate_perf();

    if (score_C < std::max(score_A, score_B)) {
        SC_MODULE_WARN << "Merging these two partition may cause performance "
                          "drop, no fall-back strategy found";
    }

    return true;
}

void mixed_parti_t::merge(const ptr &other, const op_dep_matrix_t &g) const {
    try_merge_mixed_parti_vertically(get_root(), other->get_root(), g);
}

mixed_parti_t::mixed_parti_t(const sc_op_ptr &op, const context_ptr &ctx)
    : ctx_(ctx) {
    buf_alloc_.binded_mxp_ = this;
    if (!op->isa<constant_op_t>() && !op->isa<tensor_view_op_t>()) {
        SC_MODULE_INFO << "================  create new partition: "
                       << op->op_name_ << "_" << op->logical_op_id_
                       << " ================";
        auto mixed_op = op->dyn_cast<op_traits::mixed_partition_acceptable>();
        mixed_op->create_mixed_partition(this);
        func_->name_ = op->op_name_ + std::to_string(op->logical_op_id_);
        SC_MODULE_INFO << func_;
    }
    ops.insert(op);
}

bool mixed_parti_t::is_ok_to_add(sc_op *op, const op_dep_matrix_t &g) {
    if (merged_to) { return get_root()->is_ok_to_add(op, g); }
    if (empty()) { return false; }
    if (!fusion_partition_t::is_ok_to_add(op, g)) {
        SC_MODULE_INFO << op->op_name_ << "_" << op->logical_op_id_
                       << " fail to add partition: " << func_->name_
                       << ", due to potential graph "
                          "dependency ring risk";
        return false;
    }
    if (auto reo = op->dyn_cast<reorder_op_t>()) {
        if (reo->get_output_format().is_vnni_format()) return false;
    }
    auto mixed_op = op->dyn_cast<op_traits::mixed_partition_acceptable>();
    mixed_op->search_anchor(this);
    if (!ready_for_op(op)) {
        SC_MODULE_INFO << op->op_name_ << "_" << op->logical_op_id_
                       << " fail to add partition: " << func_->name_
                       << ", due to not suitable anchor found";
        return false;
    }
    return true;
}

bool mixed_parti_t::ready_for_op(sc_op *op) const {
    if (merged_to) { return get_root()->ready_for_op(op); }
    return op_anchor_map_.find(op) != op_anchor_map_.end();
}

void mixed_parti_t::set_anchor_for_op(
        sc_op *op, const fuse_anchor_map_ptr &fanchor_map) {
    if (merged_to) {
        get_root()->set_anchor_for_op(op, fanchor_map);
        return;
    }
    auto iter = op_anchor_map_.find(op);
    if (iter != op_anchor_map_.end()) {
        // if new anchor is more smaller than the current one
        if (cmp_op_anchor(op, iter->second, fanchor_map)
                == cmp_res::l_larger_r) {
            // overwrite new anchor
            op_anchor_map_[op] = fanchor_map;
        }
    } else {
        op_anchor_map_[op] = fanchor_map;
    }
}

void mixed_parti_t::add(const sc_op_ptr &op) {
    if (merged_to) { return get_root()->add(op); }
    SC_MODULE_INFO << "================  adding op: " << op->op_name_ << "_"
                   << op->logical_op_id_ << " to partition: " << func_->name_
                   << " ================";
    auto mixed_op = op->dyn_cast<op_traits::mixed_partition_acceptable>();
    mixed_op->append_mixed_partition(this);
    func_->name_ += "_" + op->op_name_ + std::to_string(op->logical_op_id_);
    SC_MODULE_INFO << func_;
    ops.insert(op);
}

fuse_anchor_map_ptr mixed_parti_t::lookup_anchor_map(sc_op *op) {
    if (merged_to) { return get_root()->lookup_anchor_map(op); }
    auto iter = op_anchor_map_.find(op);
    COMPILE_ASSERT(iter != op_anchor_map_.end(),
            "No dispatched fusion anchor map found for "
                    << op->op_name_
                    << " in this partition, please try to search it firstly");
    return iter->second;
}

fuse_anchor_map_ptr mixed_parti_t::lookup_anchor_map(const stmts &ss) {
    if (merged_to) { return get_root()->lookup_anchor_map(ss); }
    auto iter = std::find_if(fanchors_.begin(), fanchors_.end(),
            [&ss](const fuse_anchor_map_ptr &amap) {
                return ss.ptr_same(amap->anchor_position_);
            });
    return (iter != fanchors_.end()) ? (*iter) : nullptr;
}

std::vector<fuse_anchor_map_ptr> mixed_parti_t::lookup_sub_anchor_map(
        const fuse_anchor_map_ptr &parent_fanchor) {
    if (merged_to) { return get_root()->lookup_sub_anchor_map(parent_fanchor); }
    std::vector<fuse_anchor_map_ptr> subs;
    for (auto &fanc : fanchors_) {
        if (fanc->parent_ == parent_fanchor) { subs.emplace_back(fanc); }
    }
    return subs;
}

void mixed_parti_t::clear_fanchor(fuse_anchor_map_ptr &fanchor) {
    auto anchor = fanchor->anchor_position_;
    COMPILE_ASSERT(anchor->seq_.empty(),
            "Could not remove this fanchor, due to it is not empty")
    stmt parent = get_parent_node(anchor);
    auto &ss_parent = parent.checked_as<stmts>()->seq_;
    // find anchor iter
    std::vector<sc::stmt>::iterator anchor_iter
            = std::find_if(ss_parent.begin(), ss_parent.end(),
                    [anchor](stmt &s) { return s.ptr_same(anchor); });
    COMPILE_ASSERT(anchor_iter != ss_parent.end(),
            "Could not found anchor in current parent stmts");
    // remove anchor
    ss_parent.erase(anchor_iter);
    // reset fanchor
    fanchor->anchor_position_ = stmts();
    fanchor->fsmap_.clear();
}

void mixed_parti_t::clear_fanchors() {
    for (auto iter = fanchors_.begin(); iter != fanchors_.end();) {
        auto fanchor = *iter;
        if (!fanchor->anchor_position_->size()) {
            clear_fanchor(fanchor);
            iter = fanchors_.erase(iter);
        } else {
            iter++;
        }
    }
}

std::vector<for_loop> mixed_parti_t::get_outer_loops(
        fuse_anchor_map_ptr fanchor) {
    if (merged_to) { return get_root()->get_outer_loops(); }
    auto body = func_->body_;
    std::vector<for_loop> outer_loops;
    auto ths = this;
    fuse_anchor_map_ptr target_fanchor = std::move(fanchor);
    while (target_fanchor && target_fanchor->parent_) {
        target_fanchor = target_fanchor->parent_;
    }
    auto get_next_inner_loop_with_anchor = [&ths, &target_fanchor](
                                                   const for_loop &cur_loop) {
        if (cur_loop->body_.isa<for_loop>()) {
            return cur_loop->body_.checked_as<for_loop>();
        } else if (cur_loop->body_.isa<stmts>()) {
            auto ss = cur_loop->body_.static_as<stmts>();
            if (ss->seq_.size() == 1 && ss->seq_[0].isa<for_loop>()) {
                return cur_loop->body_.checked_as<stmts>()
                        ->seq_[0]
                        .checked_as<for_loop>();
            } else if (ss->seq_.size() == 2 && ss->seq_[0].isa<for_loop>()
                    && ss->seq_[1].isa<stmts>()) {
                auto inner_ss = ss->seq_[1].static_as<stmts>();
                if (inner_ss->seq_.empty()) {
                    auto anchor_map = ths->lookup_anchor_map(inner_ss);
                    if (anchor_map) {
                        return (anchor_map == target_fanchor)
                                ? for_loop()
                                : cur_loop->body_.checked_as<stmts>()
                                          ->seq_[0]
                                          .checked_as<for_loop>();
                    }
                }
            }
        }
        return for_loop();
    };
    if (body.isa<stmts>() && body.checked_as<stmts>()->seq_.size() == 1) {
        auto st = body.checked_as<stmts>()->seq_[0];
        if (st.isa<for_loop>()) {
            auto loop = st.static_as<for_loop>();
            while (loop.defined()) {
                outer_loops.emplace_back(loop);
                loop = get_next_inner_loop_with_anchor(loop);
            }
        }
    }
    return outer_loops;
}

void mixed_parti_t::buffer_schedule() {
    if (merged_to) {
        get_root()->buffer_schedule();
        return;
    }
    buf_alloc_.declare_and_shrink_tensor(op_anchor_map_);
}

bool mixed_parti_t::is_parti_inp(const graph_tensor *gt) {
    if (merged_to) { return get_root()->is_parti_inp(gt); }
    return !contains(gt->producer_owner_);
}

bool mixed_parti_t::is_parti_inp(const graph_tensor_ptr &gt) {
    return is_parti_inp(gt.get());
}

bool mixed_parti_t::is_parti_out(const graph_tensor *gt) {
    if (merged_to) { return get_root()->is_parti_out(gt); }
    for (auto &use : gt->uses_) {
        if (!contains(use.second.get())) { return true; }
    }
    return false;
}

bool mixed_parti_t::is_parti_out(const graph_tensor_ptr &gt) {
    return is_parti_out(gt.get());
}

bool mixed_parti_t::contain_tunable_op() const {
    if (merged_to) { return get_root()->contain_tunable_op(); }
    for (auto &op : ops) {
        if (op->isa<tunable_op_t>()) return true;
    }
    return false;
}

void mixed_parti_t::clear() {
    // Graph-related
    ops.clear();

    // IR-related
    func_ = func_t();
    fanchors_.clear();
    buf_alloc_.clear();
    op_anchor_map_.clear();
}

float mixed_parti_t::evaluate_perf() {
    if (merged_to) { return get_root()->evaluate_perf(); }
    return cost_.evaluate(this);
}

static bool do_partition(const context_ptr &ctx, sc_graph_t &g,
        const op_dep_matrix_t &dep,
        std::vector<mixed_parti_t::ptr> &op_2_partition,
        const std::vector<bool> &op_mask) {
    // a DFS visitor which only visits masked ops and skips unmasked ones
    op_visitor_t visitor
            = op_visitor_t::dfs_topology_speculative_sort(g.ops_.size());
    visitor.visit_graph(g, [&](const sc_op_ptr &op) {
        if (op->isa<input_op>() || op->isa<output_op>()
                || op->attrs_.get_or_else(op_attr_key::no_fuse, false)
                || op_mask[op->logical_op_id_])
            return;

        mixed_parti_t::ptr parent_partition;
        if (!op->attrs_.get_or_else(op_attr_key::break_pre_fuse, false)) {
            // merge the partitons of all inputs
            for (auto &in : op->get_inputs()) {
                auto &cur_in_partition
                        = op_2_partition[in->producer_owner_->logical_op_id_];
                // if an input is fusible and is not "break_post_fuse"
                if (cur_in_partition
                        && !in->producer_owner_->attrs_.get_or_else(
                                op_attr_key::break_post_fuse, false)
                        && cur_in_partition->is_ok_to_add(op.get(), dep)
                        && in->producer_owner_->attrs_.get_or_else(
                                   "constant", const_kind::not_const)
                                == const_kind::not_const) {
                    if (parent_partition) {
                        // support matmul to fuse m_o axis
                        if (op->isa<tunable_op_t>()) {
                            cur_in_partition->merge(parent_partition, dep);
                        } else {
                            if (parent_partition->contain_tunable_op()) {
                                cur_in_partition->merge(parent_partition, dep);
                            } else {
                                parent_partition->merge(cur_in_partition, dep);
                            }
                        }
                        parent_partition
                                = std::static_pointer_cast<mixed_parti_t>(
                                        parent_partition->get_root()
                                                ->shared_from_this());
                    } else {
                        parent_partition = cur_in_partition;
                    }
                }
            }
        }
        if (!parent_partition) {
            parent_partition = std::make_shared<mixed_parti_t>(op, ctx);
        } else {
            parent_partition->get_root()->add(op);
        }
        op_2_partition[op->logical_op_id_] = parent_partition;
    });

    // legalize
    auto run_threads = runtime_config_t::get().get_num_threads();

    auto check_partition = [&run_threads](mixed_parti_t::ptr &parti,
                                   std::unordered_set<sc_op_ptr> &retry_ops) {
        if (!parti || parti->ops.empty() || parti->ops.size() < 2) return;
        // check tensorview in edge of partition
        for (auto &op : parti->ops) {
            if (op->isa<tensor_view_op_t>()) {
                if (parti->is_parti_out(op->get_outputs()[0])
                        || parti->is_parti_out(op->get_inputs()[0])) {
                    op->attrs_[op_attr_key::no_fuse] = true;
                    retry_ops.insert(op);
                }
            }
        }
        // check reorder in partition which includes reduce op but exclude
        // tunable op
        if (std::all_of(parti->ops.begin(), parti->ops.end(),
                    [](const sc_op_ptr &op) {
                        return !op->isa<tunable_op_t>();
                    })
                && std::any_of(parti->ops.begin(), parti->ops.end(),
                        [](const sc_op_ptr &op) {
                            return op->isa<reduce_op_t>();
                        })
                && std::any_of(parti->ops.begin(), parti->ops.end(),
                        [](const sc_op_ptr &op) {
                            return op->isa<movement_op_t>();
                        })) {
            bool forced_reorder_axis = false;
            std::unordered_set<sc_op_ptr> movement_op_set;
            for (auto &op : parti->ops) {
                if (auto rd_op = op->dyn_cast<reduce_op_t>()) {
                    int outer_rd_axis_size = 1;
                    auto reduce_axis = rd_op->get_rd_axis();
                    auto shape = rd_op->get_inputs()[0]
                                         ->details_.get_blocking_dims();
                    for (int i = 0; i < *reduce_axis.begin(); i++) {
                        outer_rd_axis_size *= shape[i];
                    }
                    if (outer_rd_axis_size < run_threads)
                        forced_reorder_axis = true;
                } else if (op->isa<movement_op_t>()) {
                    movement_op_set.insert(op);
                }
            }
            if (forced_reorder_axis) {
                std::for_each(movement_op_set.begin(), movement_op_set.end(),
                        [&retry_ops](const sc_op_ptr &op) {
                            op->attrs_[op_attr_key::no_fuse] = true;
                            retry_ops.insert(op);
                        });
            }
        }
    };

    std::unordered_set<sc_op_ptr> retry_ops;
    for (auto &parti : op_2_partition) {
        check_partition(parti, retry_ops);
    }
    if (!retry_ops.empty()) return false;

    for (auto &parti : op_2_partition) {
        if (parti) {
            parti = std::static_pointer_cast<mixed_parti_t>(
                    parti->get_root()->shared_from_this());
        }
    }
    return true;
}

bool try_optimize_reduce(const context_ptr &ctx, sc_graph_t &g,
        std::unordered_map<graph_tensor_ptr, graph_tensor_ptr> &graph2orig) {
    bool reduce_contained = false;
    std::unordered_set<sc_op_ptr> tunable_op_set;
    for (auto &op : g.ops_) {
        if (op->isa<tunable_op_t>()) {
            tunable_op_set.insert(op);
        } else if (op->isa<reduce_op_t>()) {
            reduce_contained = true;
        }
    }

    if (!reduce_contained) return false;

    if (!tunable_op_set.empty()) {
        auto ops = g.ops_;
        op_dep_matrix_t dep(g);
        std::unordered_set<reduce_op_t *> splited_reduce_set;
        std::for_each(ops.begin(), ops.end(),
                [&dep, &tunable_op_set, &splited_reduce_set](
                        const sc_op_ptr &op) {
                    if (auto red_op = op->dyn_cast<reduce_op_t>()) {
                        if (std::any_of(tunable_op_set.begin(),
                                    tunable_op_set.end(),
                                    [&dep, &op](const sc_op_ptr &tun) {
                                        return dep.lookup(tun, op) == 1;
                                    })) {
                            splited_reduce_set.insert(red_op);
                        }
                    }
                });

        if (!splited_reduce_set.empty()) {
            std::for_each(splited_reduce_set.begin(), splited_reduce_set.end(),
                    [&g, &ctx, &graph2orig](reduce_op_t *red_op) {
                        if (red_op->can_split_op()) {
                            auto orig_out = red_op->get_outputs()[0];
                            auto new_out = red_op->split_op(ctx, g, 1);
                            auto iter = graph2orig.find(orig_out);
                            if (iter != graph2orig.end()) {
                                auto orig_v = iter->second;
                                graph2orig.erase(iter);
                                graph2orig.insert(
                                        std::make_pair(new_out, orig_v));
                            }
                        }
                    });
            return true;
        } else {
            return false;
        }
    } else if (runtime_config_t::get().get_num_threads() == 1) {
        auto old_ops = g.ops_;
        for (auto &op : old_ops) {
            if (auto rd = op->dyn_cast<reduce_op_t>()) {
                if (rd->can_split_op()) { rd->split_op(ctx, g, 1); }
            }
        }
    }

    if (std::all_of(g.ops_.begin(), g.ops_.end(), [](const sc_op_ptr &op) {
            return op->isa<input_op>() || op->isa<output_op>()
                    || op->isa<constant_op_t>()
                    || op->isa<unary_elementwise_op_t>()
                    || op->isa<binary_elementwise_op_t>()
                    || op->isa<reduce_op_t>() || op->isa<reduce_impl_op_t>();
        })) {
        std::for_each(g.ops_.begin(), g.ops_.end(), [&g](const sc_op_ptr &op) {
            if (op->isa<unary_elementwise_op_t>()
                    || op->isa<binary_elementwise_op_t>()
                    || op->isa<reduce_op_t>() || op->isa<reduce_impl_op_t>()) {
                op->attrs_["temp.mixed_partition_hint.sub_graph_ptr"] = &g;
            }
        });
        return true;
    }

    return false;
}

/**
 * Copies the partition to the graph in new sub graph.
 * @param graph
 * @param partition
 * @param op_name it will append the fused op names to this string
 * @param out_output_tsr outputs the out tensors for the new fused op
 * @return the additional inputs besides the inputs of the original base op for
 * the fused op
 * */
static std::shared_ptr<mixed_fuse_op_t> transform_pa_to_mixed_op(
        const context_ptr &ctx, sc_graph_t &g,
        const std::shared_ptr<mixed_parti_t> &partition) {
    sc_graph_t sub_graph;
    std::vector<graph_tensor_ptr> fused_op_in, fused_op_out;
    std::vector<expr> arg_ins, arg_out;
    std::string op_name;
    auto &parti = *partition;
    // the mapping for original LT in original ops to fuse => the LT in the
    // sub_graph.
    std::unordered_map<graph_tensor_ptr, graph_tensor_ptr> orig_2_graph;
    // the mapping for the LT in the sub_graph  => original LT in original ops
    // to fuse.
    std::unordered_map<graph_tensor_ptr, graph_tensor_ptr> graph_2_orig;

    auto get_or_create_graph_tsr = [&](const graph_tensor_ptr &orig_lr) {
        auto itr = orig_2_graph.find(orig_lr);
        if (itr != orig_2_graph.end()) { return itr->second; }
        auto ret = std::make_shared<graph_tensor>(nullptr, orig_lr->details_);
        orig_2_graph.insert(std::make_pair(orig_lr, ret));
        graph_2_orig.insert(std::make_pair(ret, orig_lr));
        return ret;
    };

    auto visitor = op_visitor_t::dfs_topology_sort(g.ops_.size());
    std::unordered_set<graph_tensor_ptr> input_tsr_set;
    visitor.visit_graph(g, [&](const sc_op_ptr &op) {
        if (parti.ops.find(op) == parti.ops.end()) { return; }
        std::vector<graph_tensor_ptr> new_graph_in, new_graph_ou;
        for (auto &in : op->get_inputs()) {
            new_graph_in.emplace_back(get_or_create_graph_tsr(in));
            if (parti.is_parti_inp(in)
                    && input_tsr_set.find(in) == input_tsr_set.end()) {
                // if the input is not included in the parti, make an input
                // node
                sub_graph.make_input({new_graph_in.back()});
                // add the input in the args of the fused op in orig sub_graph
                fused_op_in.emplace_back(in);
                input_tsr_set.insert(in);
                COMPILE_ASSERT(parti.buf_alloc_.g2b_map_.haskey(in),
                        "No buffer allocated for "
                                << op->op_name_ << "_"
                                << std::to_string(op->logical_op_id_)
                                << " inputs")
                arg_ins.emplace_back(parti.buf_alloc_.g2b_map_.get(in));
            }
        }
        for (auto &out : op->get_outputs()) {
            new_graph_ou.emplace_back(get_or_create_graph_tsr(out));
            // if the output is a "cut" - an edge across the parti and
            // outside of the parti
            if (parti.is_parti_out(out)) {
                // if there is a use outside of the parti, the tensor should
                // be marked "output"
                const auto &outtsr = new_graph_ou.back();
                sub_graph.make_output({outtsr});
                // make a new output tensor for the fused_op_t in the original
                // sub_graph
                fused_op_out.emplace_back(
                        std::make_shared<graph_tensor>(nullptr, out->details_));
                // save the mapping of the tensor to be replaced => new tensor
                parti.output_replace_map[out] = fused_op_out.back();
                COMPILE_ASSERT(parti.buf_alloc_.g2b_map_.haskey(out),
                        "No buffer allocated for "
                                << op->op_name_ << "_"
                                << std::to_string(op->logical_op_id_)
                                << " outputs")
                arg_out.emplace_back(parti.buf_alloc_.g2b_map_.get(out));
            }
        }
        auto copyable = op->dyn_cast<op_traits::copyable_t>();
        assert(copyable);
        auto copied = copyable->copy(new_graph_in, new_graph_ou, sub_graph);
        copied->attrs_[mixed_attr_key_orig_op] = op;

        // build the  fused op name
        if (!op_name.empty()) op_name += '_';
        op_name += copied->op_name_;
    });

    std::shared_ptr<mixed_fuse_op_t> fused_op;

    if (!g.attrs_.get_or_else("temp.mixed_partition_hint.retried_graph", false)
            && try_optimize_reduce(ctx, sub_graph, graph_2_orig)) {
        sub_graph.attrs_["temp.mixed_partition_hint.retried_graph"] = true;
        SC_MODULE_INFO << "Optimizing reduce op in current partition...";
        // redo mixed partition with setting hint
        do_mixed_partition(ctx, sub_graph);
        for (auto &op : sub_graph.ops_) {
            if (auto mx_op = op->dyn_cast<mixed_fuse_op_t>()) {
                COMPILE_ASSERT(!fused_op,
                        "It is expected that only one mixed fused op after "
                        "retry")
                std::vector<graph_tensor_ptr> new_ins(
                        mx_op->get_inputs().size()),
                        new_outs(mx_op->get_outputs().size());
                std::transform(mx_op->get_inputs().begin(),
                        mx_op->get_inputs().end(), new_ins.begin(),
                        [&graph_2_orig](const graph_tensor_ptr &gt) {
                            COMPILE_ASSERT(
                                    graph_2_orig.find(gt) != graph_2_orig.end(),
                                    "Could not find gt in sub graph")
                            return graph_2_orig[gt];
                        });
                std::transform(mx_op->get_outputs().begin(),
                        mx_op->get_outputs().end(), new_outs.begin(),
                        [&graph_2_orig, &mx_op, &parti](
                                const graph_tensor_ptr &gt) {
                            for (auto &kv : mx_op->parti_->output_replace_map) {
                                if (kv.second == gt) {
                                    COMPILE_ASSERT(graph_2_orig.find(kv.first)
                                                    != graph_2_orig.end(),
                                            "Could not find gt in sub graph")
                                    return parti.output_replace_map
                                            [graph_2_orig[kv.first]];
                                }
                            }
                            COMPILE_ASSERT(0,
                                    "Could not find gt in output_replace_map")
                            return gt;
                        });
                fused_op = std::make_shared<mixed_fuse_op_t>(mx_op->op_name_,
                        mx_op->parti_, mx_op->sub_graph_,
                        /*ins*/ new_ins,
                        /*outs*/
                        new_outs, any_map_t {});
            } else {
                COMPILE_ASSERT(op->isa<input_op>() || op->isa<output_op>()
                                || op->isa<constant_op_t>(),
                        "Unexpected op type found after retry: "
                                << op->op_name_)
            }
        }
        COMPILE_ASSERT(fused_op, "No mixed fused op found");
    } else {
        // mark read/write buffer
        graph::mark_read_or_write_buffers(arg_ins, true);
        graph::mark_read_or_write_buffers(arg_out, false);
        // build up final func name and param
        std::vector<expr> args = arg_out;
        args.insert(args.end(), arg_ins.begin(), arg_ins.end());
        std::for_each(args.begin(), args.end(), [](const expr &arg) {
            arg->attr()[mixed_attr_key_cut_buffer] = true;
        });
        parti.func_->params_ = args;
        // set function name
        parti.func_->name_ = op_name;
        std::string prefix;
        if (parti.ops.size() > 1) {
            prefix = "partition_";
            auto outer_loops = parti.get_outer_loops();
            if (!outer_loops.empty()) {
                prefix = "outerloop_" + print_loops_range(outer_loops) + "_"
                        + prefix;
            }
        }
        parti.func_->name_ = prefix + parti.func_->name_;
        parti.func_->decl_->params_ = args;
        parti.func_->decl_->name_ = parti.func_->name_;
        // remove all parallel flag
        remove_parallel(parti.func_, false);

        // push return to the end of body
        auto ret = builder::make_returns_unattached(true);
        parti.func_->body_.checked_as<stmts>()->seq_.emplace_back(ret);

        parti.buffer_schedule();

        // clear unused fanchor
        partition->clear_fanchors();

        fused_op = std::make_shared<mixed_fuse_op_t>(parti.func_->name_,
                partition, sub_graph,
                /*ins*/ fused_op_in,
                /*outs*/
                fused_op_out, any_map_t {});
    }

    if (!g.attrs_.get_or_else(
                "temp.mixed_partition_hint.retried_graph", false)) {
        SC_MODULE_INFO << "mixed partition result:";
        SC_MODULE_INFO << fused_op->parti_->func_;
    }

    return fused_op;
}

using crossover_alg = std::function<void(
        std::vector<mixed_parti_t::ptr> &op_2_partition, op_dep_matrix_t &dep)>;

void brute_crossover(
        std::vector<mixed_parti_t::ptr> &op_2_partition, op_dep_matrix_t &dep) {
    SC_MODULE_INFO << "Using brute forced crossover Alg...";
    auto op_size = op_2_partition.size();
    for (size_t i = 0; i < op_size; i++) {
        auto parti_A = op_2_partition[i];
        if (!parti_A) continue;
        for (size_t j = i; j < op_size; j++) {
            auto parti_B = op_2_partition[j];
            if (!parti_B) continue;
            try_merge_mixed_parti_horizontally(
                    parti_A.get(), parti_B.get(), dep);
        }
    }
}

static void crossover_partition(std::vector<mixed_parti_t::ptr> &op_2_partition,
        op_dep_matrix_t &dep, const std::vector<crossover_alg> &algs) {
    for (auto &al : algs) {
        al(op_2_partition, dep);
    }
    for (auto &parti : op_2_partition) {
        if (parti) {
            parti = std::static_pointer_cast<mixed_parti_t>(
                    parti->get_root()->shared_from_this());
        }
    }
}

void do_mixed_partition(const context_ptr &ctx, sc_graph_t &graph) {
    op_dep_matrix_t dep(graph);
    auto op_size = graph.ops_.size();
    // mapping from op id => partition
    std::vector<mixed_parti_t::ptr> op_2_partition;
    // initial partitioning
    std::vector<bool> op_mask(op_size, false);

    constexpr int maxiter = 10;
    for (int i = 0; i < maxiter; i++) {
        op_2_partition.clear();
        op_2_partition.resize(op_size);
        if (do_partition(ctx, graph, dep, op_2_partition, op_mask)) break;
    }

    std::vector<crossover_alg> algs = {brute_crossover};
    crossover_partition(op_2_partition, dep, algs);

    std::vector<sc_op_ptr> fused_ops;
    for (auto &parti : op_2_partition) {
        if (!parti || !parti->output_replace_map.empty()
                || (parti->ops.size() <= 1)) {
            // if a partition has been processed or it is a single op, skip
            continue;
        }

        auto fused_op = transform_pa_to_mixed_op(ctx, graph, parti);

        fused_op->attrs_[mixed_attr_key_partition]
                = std::weak_ptr<mixed_parti_t>(parti);
        fused_ops.emplace_back(fused_op);
    }

    std::unordered_map<graph_tensor_ptr, graph_tensor_ptr> tsr_replace_map;
    for (auto &fused_op : fused_ops) {
        auto partition = fused_op->attrs_[mixed_attr_key_partition]
                                 .get<std::weak_ptr<mixed_parti_t>>()
                                 .lock();
        assert(partition);
        fused_op->attrs_.remove(mixed_attr_key_partition);
        for (auto &old_new : partition->output_replace_map) {
            auto &old = old_new.first;
            auto &newv = old_new.second;
            old->replace_with(newv);
            assert(tsr_replace_map.find(old) == tsr_replace_map.end());
            tsr_replace_map.insert(old_new);
        }
        for (auto &in : fused_op->info_.inputs_) {
            // if an input is replaced by other fused_op node, update it
            auto itr = tsr_replace_map.find(in);
            if (itr != tsr_replace_map.end()) { in = itr->second; }
        }
        graph.add(fused_op);
        // remove the original op mapping tag
        auto fused_op_ptr = fused_op->dyn_cast<::sc::mixed_fuse_op_t>();
        for (auto &op : fused_op_ptr->sub_graph_.ops_) {
            if (op->attrs_.has_key(mixed_attr_key_orig_op)) {
                op->attrs_.remove(mixed_attr_key_orig_op);
            }
        }
        for (auto &op : partition->ops) {
            op->remove();
        }
        if (partition->main_tunable_op) {
            partition->main_tunable_op->remove();
        }
    }
    graph.reset_op_ids();
}

void mixed_partition(sc_graph_t &graph, const context_ptr &ctx) {
    if (!graph.attrs_.get_or_else("temp.fuse", 1)) { return; }
    SC_MODULE_INFO << "Starting Mixed Partition...";
    do_mixed_partition(ctx, graph);
    // print_graph(graph, std::cout, 1);
}

} // namespace sc
