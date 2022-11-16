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

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include "fusible_op.hpp"
#include "graph_op.hpp"
#include "mixed_partition.hpp"
#include "pass/pass.hpp"
#include "transform/transform.hpp"
#include "utils.hpp"
#include "visitor.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fused_op.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/loop_transform.hpp>
#include <compiler/ir/transform/pointer_alias_info.hpp>
#include <compiler/ir/transform/scope_flatten.hpp>
#include <compiler/ir/transform/tensor_inplace_info.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <compiler/ir/visitor.hpp>
#include <ops/convolution.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/padding.hpp>
#include <ops/fusible/reduce.hpp>
#include <ops/managed_matmul_core.hpp>
#include <runtime/config.hpp>
#include <unordered_map>
#include <unordered_set>

namespace sc {

SC_MODULE(graph.mixed_partition);

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
    auto &graph = op->get_owner_graph();
    graph::tensor_detail_to_ir_tensor(graph,
            op->op_name_ + "_" + std::to_string(op->logical_op_id_) + "_ins_",
            op->get_inputs(), g2b_map_);

    // inter-op inplace inferring
    auto query_inplace = [&](const graph_tensor_ptr &out,
                                 const graph_tensor_ptr &in) -> bool {
        return (!op->isa<tunable_op_t>()) && (in->uses_.size() == 1)
                && (out != in)
                && (out->details_.get_blocking_dims()
                        == in->details_.get_blocking_dims())
                && (out->details_.dtype_ == in->details_.dtype_)
                && (out->details_.get_format() == in->details_.get_format())
                && (binded_mxp_->contains(
                        in->producer_owner_)) // inputs of partition should not
                // be inplaced
                && (!in->producer_owner_->isa<tunable_op_t>())
                && (!(g2b_map_.get(in).isa<tensor>()
                        && g2b_map_.get(in)
                                   .static_as<tensor>()
                                   ->init_value_)); // TODO(XXX): inplace inited
        // tensor
    };

    for (auto &out : op->get_outputs()) {
        if (out->attrs_.get_or_else(mixed_partition_hint::no_inplace, false)) {
            out->attrs_.remove(mixed_partition_hint::no_inplace);
            continue;
        }
        // query input
        for (auto &inp : op->get_inputs()) {
            if (query_inplace(out, inp)) {
                g2b_map_.get(out) = g2b_map_.get(inp);
                break;
            }
        }
    }

    if (auto tv_op = op->dyn_cast<tensor_view_op_t>()) {
        auto inp = tv_op->get_inputs()[0];
        if ((inp->uses_.size() == 1)
                && (inp->details_.get_blocking_dims()
                        == tv_op->get_outputs()[0]
                                   ->details_.get_blocking_dims())
                && (binded_mxp_->contains(inp->producer_owner_))
                && (!(g2b_map_.get(inp).isa<tensor>()
                        && g2b_map_.get(inp)
                                   .static_as<tensor>()
                                   ->init_value_))) {
            g2b_map_.get(op->get_outputs()[0]) = g2b_map_.get(inp);
        } else {
            auto base_tsr = get_real_tensor(g2b_map_.get(inp));
            g2b_map_.get(op->get_outputs()[0]) = builder::tensor_ptr(base_tsr,
                    std::vector<expr>(base_tsr->dims_.size(), 0),
                    graph.dims_to_expr(tv_op->get_shapes()));
        }
    }

    graph::tensor_detail_to_ir_tensor(graph,
            op->op_name_ + "_" + std::to_string(op->logical_op_id_) + "_outs_",
            op->get_outputs(), g2b_map_);

    if (op->isa<reduce_compute_op_t>()) {
        auto buf = g2b_map_.get(op->get_outputs()[0]);
        COMPILE_ASSERT(buf.isa<tensor>(),
                "output of reduce_compute_op_t should be tensor type")
        buf.checked_as<tensor>()->init_value_
                = tensor_node::get_zero_tensor_initializer();
    }

    if (auto collc_op = op->dyn_cast<reduce_collect_op_t>()) {
        if (!collc_op->is_place_holder_op()) {
            auto buf = g2b_map_.get(op->get_outputs()[0]);
            COMPILE_ASSERT(buf.isa<tensor>(),
                    "output of reduce_collect_op_t should be tensor type")
            buf.checked_as<tensor>()->init_value_
                    = tensor_node::get_zero_tensor_initializer();
        } else {
            // inplace reduce_compute_op output
            COMPILE_ASSERT(
                    op->get_inputs()[0]
                            ->producer_owner_->isa<reduce_compute_op_t>(),
                    "reduce collect op is expected to follow reduce compute "
                    "op, but got "
                            << op->get_inputs()[0]->producer_owner_->op_name_)
            g2b_map_.get(op->get_outputs()[0])
                    = g2b_map_.get(op->get_inputs()[0]);
        }
    }

    if (op->isa<padding_op_t>() && op->get_inputs()[0]->uses_.size() == 1
            && !binded_mxp_->empty()) {
        auto out = op->get_outputs()[0];
        auto ins = op->get_inputs()[0];
        auto old_input = g2b_map_.get(ins);
        if (old_input.isa<tensor>()) {
            op->attrs_.set<bool>("temp.inplace_padding", true);
            auto new_input = builder::tensor_ptr(g2b_map_.get(out),
                    op->dyn_cast<padding_op_t>()->get_padding_offsets_exprs(),
                    {}, true);
            std::unordered_map<expr, expr> buffer_map
                    = {{old_input, new_input}};
            // IR replace
            mxp_replacer_t(buffer_map).replace_func(binded_mxp_->func_);
            // Buffer replace
            replace_buffer(ins.get(), new_input);
        }
    }
}

size_t mxp_buffer_allocator::get_total_allocated_buffer_size() const {
    size_t already_allocated_size = 0;
    for (auto &tsr_anch_pair : tsr_anch_map_) {
        auto tsr = tsr_anch_pair.first.checked_as<tensor>();
        already_allocated_size
                += (get_dims_product(get_expr_to_dims(tsr->dims_))
                        * utils::get_sizeof_etype(tsr->elem_dtype_.type_code_));
    }
    return already_allocated_size;
}

void mxp_buffer_allocator::replace_buffer(graph_tensor *gt, expr &new_buffer) {
    // get old buffer
    auto old_buffer = g2b_map_.get(gt);
    // assert new buffer
    COMPILE_ASSERT(b2g_map_.find(new_buffer) == b2g_map_.end(),
            "Current, it is only expected to replace with new buffer which "
            "never appear in mixed IR, but got "
                    << new_buffer)
    if (tsr_anch_map_.find(old_buffer) != tsr_anch_map_.end()) {
        tsr_anch_map_.erase(old_buffer);
    }
    if (b2g_map_.find(old_buffer) != b2g_map_.end()) {
        auto gt = b2g_map_[old_buffer];
        b2g_map_.erase(old_buffer);
        b2g_map_[new_buffer] = gt;
    }
    g2b_map_.get(gt) = new_buffer;
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

void mxp_buffer_allocator::update_input_buffer_info(sc_op *op) {
    auto commited_anchor_map = binded_mxp_->lookup_anchor_map(op);
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
            // auto skip
            if (tsr_anch_map_[tsr] == real_anchor_map) return;
            auto cur_slice = tsr_anch_map_[tsr]->fsmap_.get(b2g_map_[buf]);
            auto new_slice = real_anchor_map->fsmap_.get(gt);
            auto res = cmp_slice_range(cur_slice, new_slice);
            if (res == cmp_res::l_less_r) {
                tsr_anch_map_[tsr] = real_anchor_map;
                if (real_anchor_map != commited_anchor_map) b2g_map_[buf] = gt;
            } else if (res == cmp_res::equal) {
                if (tsr_anch_map_[tsr]->is_cousin_for(real_anchor_map)) {
                    // set common parent anchor
                    tsr_anch_map_[tsr]
                            = real_anchor_map->get_root()->shared_from_this();
                } else if (real_anchor_map->is_parent_for(tsr_anch_map_[tsr])
                        || real_anchor_map->is_sibling_for(
                                tsr_anch_map_[tsr])) {
                    // usually occurs in op is reduce or reduce collect op
                    tsr_anch_map_[tsr] = real_anchor_map;
                } else {
                    // dont overwrite
                    return;
                }
                if (real_anchor_map != commited_anchor_map) b2g_map_[buf] = gt;
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

void mxp_buffer_allocator::update_output_buffer_info(sc_op *op) {
    auto commited_anchor_map = binded_mxp_->lookup_anchor_map(op);
    auto sub_of_commited_anchor_map
            = binded_mxp_->lookup_sub_anchor_map(commited_anchor_map);

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

void mxp_buffer_allocator::tensor_initialize() {
    for (auto &pair : g2b_map_.datamap_) {
        auto op = pair.first->producer_owner_;
        // zero out padding area
        if (auto padding = op->dyn_cast<padding_op_t>()) {
            if (b2g_map_.find(pair.second) == b2g_map_.end()) continue;
            stmts decl_body;
            auto pad_tsr = get_real_tensor(pair.second);
            slice_range_list range_list = {};
            if (pair.second->attr().has_key(mixed_partition_hint::cut_buffer)) {
                decl_body = binded_mxp_->func_->body_.checked_as<stmts>();
            } else {
                COMPILE_ASSERT(
                        tsr_anch_map_.find(pad_tsr) != tsr_anch_map_.end(),
                        "Could not find padding tensor: "
                                << pad_tsr << " in tsr2anchor map")
                auto anchor = tsr_anch_map_[pad_tsr];
                range_list = anchor->fsmap_.get(b2g_map_[pair.second]);
                decl_body = anchor->get_parent_scope();
            }
            auto ret = padding->get_zero_out_stmt(pad_tsr, range_list);
            decl_body->seq_.insert(decl_body->seq_.begin(), ret);
        }
    }
}

inline bool is_elementwise_producer(const graph_tensor *gt) {
    return gt->producer_owner_->isa<unary_elementwise_op_t>()
            || gt->producer_owner_->isa<binary_elementwise_op_t>();
}

// max step to explore preview ops
static constexpr int EXPLORE_INPLACE_MAX_STEP = 8;

static void collect_inplace_info(graph_tensor *cur_gt, graph_tensor *ref_gt,
        std::unordered_set<graph_tensor *> &inplace_set,
        std::unordered_set<expr> &visited_set,
        const std::unordered_set<graph_tensor *> &valid_set,
        mxp_buffer_allocator *alloc, int step) {
    // increment by 1 recursive depth and auto skip
    if (EXPLORE_INPLACE_MAX_STEP == (step++)) return;
    // skip repeated
    if (inplace_set.find(cur_gt) != inplace_set.end()) return;
    // return when producer is not elementwise op
    if (!is_elementwise_producer(cur_gt)) { return; }
    // use buffer as key for map
    auto cur_buf = alloc->g2b_map_.get(cur_gt);
    // check inplace condition
    if ((visited_set.find(cur_buf) == visited_set.end())
            && (valid_set.find(cur_gt) != valid_set.end()) && (cur_gt != ref_gt)
            && (cur_gt->details_.get_blocking_dims()
                    == ref_gt->details_.get_blocking_dims())
            && (cur_gt->details_.dtype_ == ref_gt->details_.dtype_)
            && (cur_gt->details_.get_format() == ref_gt->details_.get_format())
            && std::all_of(cur_gt->uses_.begin(), cur_gt->uses_.end(),
                    [&alloc, &ref_gt](
                            const std::pair<int, sc::sc_op_weak_ptr_t> &user) {
                        return alloc->binded_mxp_->dep_m_->lookup(
                                       user.second.get(),
                                       ref_gt->producer_owner_)
                                == 1;
                    })) {
        inplace_set.insert(cur_gt);
        // reset step
        step = 0;
        // reset ref_gt
        ref_gt = cur_gt;
    }
    // mark visited
    visited_set.insert(cur_buf);
    // get cur op
    auto elem_op = cur_gt->producer_owner_;
    // recursively collect inplace information
    for (auto &inp : elem_op->get_inputs()) {
        collect_inplace_info(inp.get(), ref_gt, inplace_set, visited_set,
                valid_set, alloc, step);
    }
}

void mxp_buffer_allocator::query_buffer_inplace() {
    SC_MODULE_INFO << "Query buffer inplace hint...";
    // step 0: get outer loop
    auto outer_loops = binded_mxp_->get_outer_loops();
    if (outer_loops.empty()) return;
    auto batch_anchor = binded_mxp_->get_anchor_inside_loop(outer_loops.back());
    if (!batch_anchor) return;

    // step 1: search gt which defined inside outer loop
    std::vector<graph_tensor *> ref_gt_list;
    std::unordered_set<graph_tensor *> valid_set;
    for (auto &tsr2anch : tsr_anch_map_) {
        auto anchor = tsr2anch.second;
        auto tsr = tsr2anch.first;
        if (tsr->attr().has_key(mixed_partition_hint::cut_buffer)) continue;
        if (b2g_map_.find(tsr) == b2g_map_.end()) continue;
        auto shrink_gt = b2g_map_[tsr];
        auto slice_on_batch_anchor = batch_anchor->fsmap_.get(shrink_gt);
        auto slice_on_cur_anchor = slice_range_list {get_shrinked_info(tsr)};
        if (slice_on_cur_anchor.empty() || slice_on_batch_anchor.empty())
            continue;
        if (cmp_slice_range(slice_on_batch_anchor, slice_on_cur_anchor)
                == cmp_res::equal) {
            ref_gt_list.emplace_back(b2g_map_[tsr].get());
            valid_set.insert(b2g_map_[tsr].get());
        }
    }
    // skip
    if (ref_gt_list.empty()) return;
    // sort by op id
    std::sort(ref_gt_list.begin(), ref_gt_list.end(),
            [](const graph_tensor *gt1, const graph_tensor *gt2) {
                return gt1->producer_owner_->logical_op_id_
                        > gt2->producer_owner_->logical_op_id_;
            });

    std::unordered_set<expr> replaced_buffer, visited_buffer;
    for (auto &ref_gt : ref_gt_list) {
        // auto skip
        if (replaced_buffer.find(g2b_map_.get(ref_gt)) != replaced_buffer.end())
            continue;
        // step 2: collect inplace mapping for each gt
        std::unordered_set<graph_tensor *> inplace_gt_set;
        collect_inplace_info(ref_gt, ref_gt, inplace_gt_set, visited_buffer,
                valid_set, this, /*init_step*/ 0);
        if (inplace_gt_set.empty()) continue;
        // step 3: transform map to vector sorted by op committing order
        std::vector<graph_tensor *> inplace_gt_list;
        for (auto &commit_op : binded_mxp_->committed_ops_) {
            if (commit_op.get() == ref_gt->producer_owner_) {
                inplace_gt_list.emplace_back(ref_gt);
            } else {
                for (auto &out : commit_op->get_outputs()) {
                    if (inplace_gt_set.find(out.get())
                            != inplace_gt_set.end()) {
                        inplace_gt_list.emplace_back(out.get());
                    }
                }
            }
        }
        // step 4: mark inplace attr for each buffer in list with the previous
        // one
        for (size_t i = 1; i < inplace_gt_list.size(); i++) {
            // get target gt
            auto target_gt = inplace_gt_list[i - 1];
            auto target_buf = g2b_map_.get(target_gt);
            // get inplace gt
            auto inplace_gt = inplace_gt_list[i];
            auto inplace_buf = g2b_map_.get(inplace_gt);
            // skip repeated replaced
            if (replaced_buffer.find(inplace_buf) != replaced_buffer.end())
                continue;

            auto target_id
                    = alias_info::get_or_create_alias_info(*target_buf.get());

            SC_MODULE_INFO << "Mark inplace hint for buffer: " << inplace_buf
                           << " ==> " << target_buf;

            inplace_buf->attr()[attr_keys::tensor_inplace_hint]
                    = std::vector<temp_tensor_inplace_info_t> {
                            {target_id, inplace_kind::ZERO_OFFSET}};
            replaced_buffer.insert(inplace_buf);
        }
    }
}

fuse_anchor_map_ptr mxp_buffer_allocator::get_real_anchor_for_buffer(
        const expr &buffer) const {
    auto tsr = get_real_tensor(buffer);
    COMPILE_ASSERT(tsr_anch_map_.find(tsr) != tsr_anch_map_.end(),
            "Could ont find " << tsr << " in tsr2anchor map")
    auto anch = tsr_anch_map_.find(tsr)->second;
    auto parent_loop = anch->get_parent_loop();
    // use outer anchor for cases that calculation is partially done. (e.g.
    // calculating part of K for matmul)
    if ((b2g_map_.find(buffer) == b2g_map_.end()
                || b2g_map_.find(buffer)
                           ->second->producer_owner_->isa<tunable_op_t>())
            && parent_loop.isa<for_loop>()
            && parent_loop->attr().has_key(stmt_attr_key::reduce_root_loop)) {
        auto raw = parent_loop->attr()
                           .get<std::weak_ptr<stmt_base_t>>(
                                   stmt_attr_key::reduce_root_loop)
                           .lock();
        COMPILE_ASSERT(raw, "reduce_root_loop weak ptr invalidated");
        anch = binded_mxp_->get_anchor_inside_loop(
                stmt(raw).checked_as<for_loop>());
        COMPILE_ASSERT(anch,
                "No anchor found under reduce root loop, please create it "
                "otherwise, it may cause correctness issue")
    }
    return anch;
}

slice_range mxp_buffer_allocator::get_shrinked_info(const expr &buffer) const {
    auto anchor = get_real_anchor_for_buffer(buffer);
    COMPILE_ASSERT(b2g_map_.find(buffer) != b2g_map_.end(),
            "Could not find " << buffer << " b2g map")
    auto range_list = anchor->fsmap_.get(b2g_map_.find(buffer)->second);
    if (range_list.size() != 1) {
        if (range_list.empty() || !anchor->isa<fuse_iter_anchor_map_t>())
            return {};
        else
            return *std::max_element(range_list.begin(), range_list.end(),
                    [&](const slice_range &A, const slice_range &B) {
                        return cmp_slice_range({A}, {B}) == cmp_res::l_less_r;
                    });

    } else {
        return range_list[0];
    }
}

void mxp_buffer_allocator::declare_and_shrink_tensor() {
    // define real tensor, and put it at the beginning of ss
    auto declare_tensor_ = [&](const expr &tsr,
                                   const fuse_anchor_map_ptr &fanchor) {
        // recurrsively find parent fanchor
        auto &ss = fanchor->get_parent_scope()->seq_;
        ss.emplace(ss.begin(), builder::make_var_tensor_def_unattached(tsr));
    };
    // set shrink info
    auto set_shrink_info_ = [&](const expr &buffer) {
        auto shrink_range = get_shrinked_info(buffer);
        if (shrink_range.empty()) return;
        buffer->attr()[tensor_shrinker_attrs::should_shrink]
                = tensor_shrinker_t::shrink_info_t {
                        /*base*/ get_slice_idx(shrink_range),
                        /*shape*/ get_slice_shape(shrink_range), stmts()};
    };

    for (auto &tsr2def : tsr_anch_map_) {
        if (tsr2def.first->attr().has_key(mixed_partition_hint::cut_buffer))
            continue;
        declare_tensor_(
                tsr2def.first, get_real_anchor_for_buffer(tsr2def.first));
    }
    for (auto &buf2shr : b2g_map_) {
        if (buf2shr.first->attr().has_key(mixed_partition_hint::cut_buffer))
            continue;
        set_shrink_info_(buf2shr.first);
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
            if ((binded_mxp_->is_parti_inp(other_g2b.first)
                        && other.binded_mxp_->is_parti_inp(other_g2b.first))
                    || (binded_mxp_->is_parti_out(other_g2b.first)
                            && other.binded_mxp_->is_parti_out(
                                    other_g2b.first)))
                continue;
            COMPILE_ASSERT(common_buffer_anchor,
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

void outerloop_axis_binder::run(int real_axis_size) {
    // reset
    reset();
    if (!base_gt_ || init_axis_.empty()) return;
    // set start node
    bd_ax_map_.get(base_gt_) = bound_axis {init_axis_.begin(),
            init_axis_.begin()
                    + std::min(static_cast<int64_t>(real_axis_size),
                            static_cast<int64_t>(init_axis_.size()))};
    // call start node user recursively infer binding axis
    COMPILE_ASSERT(
            !base_gt_->uses_.empty(), "no user found for base graph tensor")
    for (auto &user : base_gt_->uses_) {
        auto user_op = user.second;
        if (!user_op->isa<output_op>()) {
            COMPILE_ASSERT(
                    user_op->isa<op_traits::mixed_partition_acceptable>(),
                    user_op->op_name_
                            << " is not mixed partition acceptable op")
            user_op->dyn_cast<op_traits::mixed_partition_acceptable>()
                    ->infer_binding_axis(bd_ax_map_);
            break;
        }
    }
}

int outerloop_axis_binder::align_with(
        outerloop_axis_binder &other, int check_axis_size) {
    // start running auto-infer binding axis
    run(check_axis_size);
    other.run(check_axis_size);

    bound_axis cur_axis, other_axis;
    if (bd_ax_map_.haskey(base_gt_) && other.bd_ax_map_.haskey(base_gt_)) {
        cur_axis = bd_ax_map_.get(base_gt_);
        other_axis = other.bd_ax_map_.get(base_gt_);
    } else if (bd_ax_map_.haskey(other.base_gt_)
            && other.bd_ax_map_.haskey(other.base_gt_)) {
        cur_axis = bd_ax_map_.get(other.base_gt_);
        other_axis = other.bd_ax_map_.get(other.base_gt_);
    } else {
        SC_MODULE_INFO
                << "Could not validate axis due to no binding hint found";
        return 0;
    }
    COMPILE_ASSERT(!cur_axis.empty() && !other_axis.empty(),
            "binding axis could not be empty, but got "
                    << utils::print_nested_vector(cur_axis) << " and "
                    << utils::print_nested_vector(other_axis))
    COMPILE_ASSERT(check_axis_size <= static_cast<int64_t>(cur_axis.size())
                    && check_axis_size
                            <= static_cast<int64_t>(other_axis.size()),
            "check axis size should not be larger than binding axis size, but "
            "got " << check_axis_size
                   << " for " << utils::print_nested_vector(cur_axis) << " and "
                   << utils::print_nested_vector(other_axis))
    int aligned_num = 0;
    for (int i = 0; i < check_axis_size; i++) {
        if (cur_axis[i] == other_axis[i])
            aligned_num++;
        else
            break;
    }
    return aligned_num;
}

void extract_anchor_from_fmgr_to_parti(fusion_manager *fmgr,
        mixed_parti_t *parti, std::vector<expr> ir_tsrs,
        std::vector<graph_tensor_ptr> gtsrs,
        const fuse_anchor_map_ptr &parent_fanchor) {
    if (!fmgr->iter_anchor_list_.empty()) {
        for (auto &iter_anchor : fmgr->iter_anchor_list_) {
            fslice_map fsmap;
            for (size_t i = 0; i < ir_tsrs.size(); i++) {
                if (iter_anchor.tsr_.ptr_same(ir_tsrs[i])) {
                    fsmap.get(gtsrs[i]) = iter_anchor.slice_list_;
                    parti->append_fusion_anchor(
                            std::make_shared<fuse_iter_anchor_map_t>(
                                    iter_anchor.iter_,
                                    iter_anchor.anchor_position_, fsmap,
                                    iter_anchor.slice_list_.size(),
                                    iter_anchor.dispatch_helper_));
                }
            }
        }
    }

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
            parti->append_fusion_anchor(std::make_shared<fuse_anchor_map_t>(
                    anchor_map.first, fsmap, parent_fanchor));
            extracted_anchor_size++;
        }
    }
    if (anchor_map_list.size() != extracted_anchor_size)
        SC_MODULE_INFO
                << "Could not extract fusion anchor from fmgr, please check";
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
        } else if (op_anchor->content_number_map_.find(this)->second
                > op_anchor->content_number_map_.find(op)->second) {
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
        if (op_anchor_loop->attr().has_key("builder.parent_node")) {
            op_anchor_loop = get_parent_node(op_anchor_loop);
        } else {
            return false;
        }
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

bool fuse_anchor_map_t::check_input_for_op(
        const sc_op *op, std::unordered_set<graph_tensor_ptr> &known_gt) {
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
                while (cur->parent_) {
                    cur = cur->parent_.get();
                    // if input is ready on cur anchor, its input slice range
                    // should not be empty
                    if (!cur->fsmap_.haskey(dep_op->get_inputs()[0])
                            || cur->fsmap_.get(dep_op->get_inputs()[0]).empty()
                            || !cur->fsmap_.haskey(gt)
                            || cur->fsmap_.get(gt).empty())
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

void fuse_anchor_map_t::forbid_op_in_anchor(
        const sc_op *op, std::unordered_set<graph_tensor_ptr> &known_gt) {
    auto fanchor = this->shared_from_this();
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

void search_op_anchor_in_parti(sc_op *op, mixed_parti_t *parti) {
    if (parti->merged_to) {
        search_op_anchor_in_parti(op, parti->get_root());
        return;
    }

    // querry current parallelism
    float current_parallelism
            = evaluate_loop_parallel_balance(parti->get_outer_loops());

    for (auto &fanchor : parti->fanchors_) {
        infer_status_map_t stat_map(parti->ctx_, false);
        std::unordered_set<graph_tensor_ptr> known_gt;
        if (parti->empty() && op->isa<reorder_op_t>()
                && !fanchor->fsmap_.haskey(op->get_inputs()[0])) {
            auto reo = op->stc_cast<reorder_op_t>();
            if (reo->support_optmized_kernel(parti->ctx_)) {
                auto output_dims
                        = op->get_outputs()[0]->details_.get_blocking_dims();
                size_t required_axis_from_end
                        = reo->get_output_format().is_vnni_format() ? 3 : 2;
                COMPILE_ASSERT(output_dims.size() > required_axis_from_end,
                        "Unexpected output format kind: "
                                << reo->get_output_format())
                std::vector<int> required_axis;
                for (size_t i = output_dims.size() - required_axis_from_end;
                        i < output_dims.size(); i++) {
                    required_axis.emplace_back(i);
                }
                if (!slice_full_on_axis(output_dims,
                            fanchor->fsmap_.get(op->get_outputs()[0])[0],
                            required_axis)) {
                    stat_map.append_ops_by_status(op, infer_status_code::RETRY);
                    fanchor->forbid_op_in_anchor(op, known_gt);
                    continue;
                }
            }
            fanchor->fsmap_.get(op->get_inputs()[0])
                    = slice_range_list {gen_slice_by_dims(
                            op->get_inputs()[0]->details_.get_blocking_dims())};
        } else {
            if (op->attrs_.get_or_else(op_attr_key::break_pre_fuse, false)
                    && !parti->empty()) {
                // if op is marked as break_pre_fuse, only pre-op fusion is
                // accepted
                if (!fanchor->fsmap_.haskey(op->get_outputs()[0])
                        || fanchor->fsmap_.get(op->get_outputs()[0]).empty()) {
                    continue;
                }
            }
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
            // check for iter fuse anchor
            if (fanchor->isa<fuse_iter_anchor_map_t>()) {
                auto expected_slice_size
                        = fanchor->dyn_cast<fuse_iter_anchor_map_t>()
                                  ->iter_size_;
                if (std::any_of(op->get_outputs().begin(),
                            op->get_outputs().end(),
                            [&fanchor, &expected_slice_size](
                                    const graph_tensor_ptr &out) {
                                return fanchor->fsmap_.get(out).size()
                                        != expected_slice_size;
                            })) {
                    fanchor->forbid_op_in_anchor(op, known_gt);
                    continue;
                }
            }
            if (!fanchor->check_input_for_op(op, known_gt)) {
                fanchor->forbid_op_in_anchor(op, known_gt);
                continue;
            }
            // check parallelism: it should be ensured new commited anchor would
            // not break parallelism
            if (parti->ctx_->flags_.use_cost_model_
                    && (parti->contain_tunable_op() || op->isa<tunable_op_t>())
                    && (current_parallelism > evaluate_loop_parallel_balance(
                                parti->get_outer_loops(fanchor)))) {
                SC_MODULE_INFO << "Cost model rejects current inferring result "
                                  "for op: "
                               << op->op_name_ << op->logical_op_id_
                               << " under current fusion anchor from "
                                  "perspective of parallellism";
                fanchor->forbid_op_in_anchor(op, known_gt);
                continue;
            }
            parti->set_anchor_for_op(op, fanchor);
            continue;
        }
        if (stat_map.is_fail()) {
            auto &fail_list
                    = stat_map.get_ops_by_status(infer_status_code::FAIL);
            if (fail_list.find(op->shared_from_this()) == fail_list.end()) {
                parti->set_anchor_for_op(op, fanchor);
                continue;
            }
        }
        fanchor->forbid_op_in_anchor(op, known_gt);
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
static parti_dep check_parti_dep(mixed_parti_t *A, mixed_parti_t *B) {
    A = A->get_root(), B = B->get_root();
    auto dep_m = A->dep_m_;
    auto A_ops = A->ops, B_ops = B->ops;
    bool A_dep_B = false, B_dep_A = false;
    for (auto &op_a : A_ops) {
        for (auto &op_b : B_ops) {
            auto dep_flag = dep_m->lookup(op_a, op_b);
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
static bool check_parti_connectionship(mixed_parti_t *A, mixed_parti_t *B) {
    A = A->get_root(), B = B->get_root();
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

/**
 * Check two partition ring risk
 * */
static bool check_parti_ring_risk(mixed_parti_t *A, mixed_parti_t *B) {
    A = A->get_root(), B = B->get_root();
    auto dep = check_parti_dep(A, B);
    if (dep == parti_dep::inter_dep)
        return true;
    else if (dep == parti_dep::no_dep)
        return false;

    auto append_parti = (dep == parti_dep::l_dep_r) ? A : B,
         target_parti = (dep == parti_dep::l_dep_r) ? B : A;
    auto append_ops = append_parti->ops;
    for (auto &op_a : append_ops) {
        for (auto &inp : op_a->get_inputs()) {
            if (append_parti->is_parti_inp(inp)
                    && !target_parti->contains(inp->producer_owner_)) {
                for (auto &op_in_set : target_parti->ops) {
                    auto result = target_parti->dep_m_->lookup(
                            op_in_set->logical_op_id_,
                            inp->producer_owner_->logical_op_id_);
                    if (result == 1) { return true; }
                }
            }
        }
    }
    return false;
}

/**
 * Check two partition is forked, like
 *            input
 *           /     \
 *      parti A   parti B
 * */
static bool check_parti_forked(mixed_parti_t *A, mixed_parti_t *B) {
    A = A->get_root(), B = B->get_root();
    auto dep_m = A->dep_m_;
    // for all op depends on A, should not depends on B
    for (auto &op_in_A : A->ops) {
        auto A_id = op_in_A->logical_op_id_;
        auto related_ids = dep_m->lookup_ops_depend_on(A_id);
        for (auto &op_in_B : B->ops) {
            auto B_id = op_in_B->logical_op_id_;
            if (std::any_of(related_ids.begin(), related_ids.end(),
                        [&B_id, &dep_m](const int &rid) {
                            return dep_m->lookup(B_id, rid) == 1;
                        })) {
                return false;
            }
        }
    }
    return true;
}

/**
 * Check two partition axis binding
 * */
static int check_parti_loop_axis_binding(
        mixed_parti_t *A, mixed_parti_t *B, int check_loop_size) {
    // skip conv workload until all conv ops implement axis binding infer
    if (A->is_conv_workload() || B->is_conv_workload()) {
        // limit to at most one outer loop (batch-dimension)
        return std::min(1, check_loop_size);
    }
    // auto skip when A and B have no dependency
    if (check_parti_forked(A, B)) return check_loop_size;
    A = A->get_root(), B = B->get_root();
    return A->ax_binder_.align_with(B->ax_binder_, check_loop_size);
}

static bool try_merge_mixed_parti_parallel(mixed_parti_t *A, mixed_parti_t *B) {
    A = A->get_root(), B = B->get_root();
    if (A == B) return false;
    if (!A->func_.get() || !B->func_.get()) return false;
    if (!check_parti_connectionship(A, B)) return false;
    if (check_parti_ring_risk(A, B)) return false;

    auto dep = check_parti_dep(A, B);
    COMPILE_ASSERT(
            dep != parti_dep::inter_dep, "inter-dependency is not expected");

    auto append_parti = (dep == parti_dep::l_dep_r) ? A : B,
         target_parti = (dep == parti_dep::l_dep_r) ? B : A;

    auto outer_loops_target = target_parti->get_outer_loops(),
         outer_loops_append = append_parti->get_outer_loops();
    if (outer_loops_target.empty() || outer_loops_append.empty()) return false;

    auto outermost_loop_target = outer_loops_target[0],
         outermost_loop_append = outer_loops_append[0];

    // check parallel loops attr.
    if (!(outermost_loop_target->kind_ == for_type::PARALLEL
                && (outermost_loop_target->num_threads_ > 0))
            || !(outermost_loop_append->kind_ == for_type::PARALLEL
                    && (outermost_loop_append->num_threads_ > 0))) {
        return false;
    }
    // check axis binding.
    if (!check_parti_loop_axis_binding(append_parti, target_parti,
                1 /*due to current parallel merge only for outer most loop*/)) {
        return false;
    }
    // cannot do parallel merge when loop split granularity are different
    if (outermost_loop_target->attr().get_or_else(
                stmt_attr_key::parallel_merge_loop_granularity, 1)
            != outermost_loop_append->attr().get_or_else(
                    stmt_attr_key::parallel_merge_loop_granularity, 1)) {
        return false;
    }

    if (outermost_loop_target->iter_begin_.isa<constant_c>()
            && outermost_loop_target->iter_end_.isa<constant_c>()
            && outermost_loop_target->step_.isa<constant_c>()
            && outermost_loop_append->iter_begin_.isa<constant_c>()
            && outermost_loop_append->iter_end_.isa<constant_c>()
            && outermost_loop_append->step_.isa<constant_c>()) {
        auto target_begin = get_expr_as_int(outermost_loop_target->iter_begin_),
             target_end = get_expr_as_int(outermost_loop_target->iter_end_),
             target_step = get_expr_as_int(outermost_loop_target->step_),
             append_begin = get_expr_as_int(outermost_loop_append->iter_begin_),
             append_end = get_expr_as_int(outermost_loop_append->iter_end_),
             append_step = get_expr_as_int(outermost_loop_append->step_);
        auto target_thread = outermost_loop_target->num_threads_,
             append_thread = outermost_loop_append->num_threads_;
        if (!(target_begin == append_begin && target_end == append_end
                    && target_step == append_step
                    && target_thread == append_thread))
            return false;
    } else {
        return false;
    }

    // cannot do parallel merge when loop split granularity are different
    if (outermost_loop_target->attr().get_or_else(
                stmt_attr_key::parallel_merge_loop_granularity, 1)
            != outermost_loop_append->attr().get_or_else(
                    stmt_attr_key::parallel_merge_loop_granularity, 1)) {
        return false;
    }

    SC_MODULE_INFO << "parallel merging two partition:";
    SC_MODULE_INFO << target_parti->func_;
    SC_MODULE_INFO << append_parti->func_;

    // remove barrier if necessary
    if (dep == parti_dep::no_dep) {
        auto last_loop = get_last_loop_in_body(outermost_loop_target->body_);
        if (last_loop.defined()) {
            last_loop->attr()[stmt_attr_key::no_post_barrier] = true;
        }
    }

    /* * * * * * * * * * * * * * * * *
     * Step 1: Merge func_
     * * * * * * * * * * * * * * * * */
    std::unordered_map<expr, expr> expr_map;
    expr_map[outermost_loop_append->var_] = outermost_loop_target->var_;

    auto common_fanchor
            = target_parti->get_anchor_inside_loop(outermost_loop_target);
    auto common_other_fanchor
            = append_parti->get_anchor_inside_loop(outermost_loop_append);
    // create common anchor if necessary
    if (!common_fanchor) {
        auto s = builder::make_stmts_unattached({}).checked_as<stmts>();
        add_parent_node(s, outermost_loop_target);
        outermost_loop_target->body_.checked_as<stmts>()->seq_.emplace_back(s);
        // dummy fsmap, the tensor belongs to this scope will not be shrinked
        fslice_map fsmap;
        common_fanchor = std::make_shared<fuse_anchor_map_t>(s, fsmap);
        target_parti->append_fusion_anchor(common_fanchor);
    } else if (common_other_fanchor) {
        common_fanchor->merge(common_other_fanchor);
    }

    common_fanchor->fuse_anchor_map_t::commit_stmt(
            outermost_loop_append->body_);

    /* * * * * * * * * * * * * * * * *
     * Step 2: Merge fanchor_
     * * * * * * * * * * * * * * * * */
    target_parti->append_fusion_anchor(append_parti->fanchors_);

    /* * * * * * * * * * * * * * * * *
     * Step 3: Merge buffer_
     * * * * * * * * * * * * * * * * */
    std::unordered_map<expr, expr> buffer_map;
    target_parti->buf_alloc_.merge(append_parti->buf_alloc_, buffer_map,
            std::make_pair(common_fanchor, common_other_fanchor));

    /* * * * * * * * * * * * * * * * *
     * Step 4: Replace expr
     * * * * * * * * * * * * * * * * */
    expr_map.insert(buffer_map.begin(), buffer_map.end());
    mxp_replacer_t expr_reper(expr_map);
    // 1. func->body
    expr_reper.replace_func(target_parti->func_);
    // 2. fanchor->fsmap->slice_range
    expr_reper.replace_anchor(target_parti->fanchors_);

    /* * * * * * * * * * * * * * * * *
     * Step 5: Merge op_anchor_map_
     * * * * * * * * * * * * * * * * */
    target_parti->op_anchor_map_.insert(append_parti->op_anchor_map_.begin(),
            append_parti->op_anchor_map_.end());

    // call base merge
    target_parti->fusion_partition_t::merge(
            static_cast<fusion_partition_t *>(append_parti)
                    ->shared_from_this());

    // Merge commited ops
    target_parti->committed_ops_.insert(target_parti->committed_ops_.end(),
            append_parti->committed_ops_.begin(),
            append_parti->committed_ops_.end());

    target_parti->func_->name_
            += "_parallel_merge_" + append_parti->func_->name_;

    SC_MODULE_INFO << "parallel merging result:";
    SC_MODULE_INFO << target_parti->func_;

    // clear merged parti
    append_parti->clear();

    return true;
}

static bool try_merge_mixed_parti_horizontally(
        mixed_parti_t *A, mixed_parti_t *B) {
    A = A->get_root(), B = B->get_root();
    if (A == B) return false;
    if (!A->func_.get() || !B->func_.get()) return false;
    if (!A->contain_tunable_op() || !B->contain_tunable_op()) return false;
    if (!check_parti_connectionship(A, B)) return false;
    if (check_parti_dep(A, B) != parti_dep::no_dep) return false;

    auto outer_loops_A = A->get_outer_loops(),
         outer_loops_B = B->get_outer_loops();
    if (outer_loops_A.empty() || outer_loops_B.empty()) return false;
    if (outer_loops_A[0]->num_threads_ > 0
            || outer_loops_B[0]->num_threads_ > 0)
        return false;

    if (A->ctx_->flags_.use_cost_model_) {
        auto A_parallelism = evaluate_loop_parallel_balance(outer_loops_A),
             B_parallelism = evaluate_loop_parallel_balance(outer_loops_B);
        if (A_parallelism == 1.f || B_parallelism == 1.f) {
            SC_MODULE_INFO << "Cost model rejects to horizontally merge two "
                              "partition: "
                           << A->func_->name_ << " and " << B->func_->name_
                           << " from perspective of parallellism";
            return false;
        }
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
    A->append_fusion_anchor(B->fanchors_);

    /* * * * * * * * * * * * * * * * *
     * Step 3: Merge buffer_
     * * * * * * * * * * * * * * * * */
    std::unordered_map<expr, expr> buffer_map;
    A->buf_alloc_.merge(
            B->buf_alloc_, buffer_map, std::make_pair(nullptr, nullptr));

    /* * * * * * * * * * * * * * * * *
     * Step 4: Replace expr
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

    // Merge commited ops
    A->committed_ops_.insert(A->committed_ops_.end(), B->committed_ops_.begin(),
            B->committed_ops_.end());

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
        add_parent_node(loops[i]->body_, loops[0]);
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

static sc_dim get_loops_range(const for_loop &loop) {
    if (!(loop->iter_begin_.isa<constant_c>()
                && loop->iter_end_.isa<constant_c>())) {
        return (int64_t)0;
    }
    return get_expr_as_int(loop->iter_end_)
            - get_expr_as_int(loop->iter_begin_);
}

static sc_dim get_loops_range_prod(const std::vector<for_loop> &loops) {
    sc_dim prod_res = 1;
    for (auto &l : loops) {
        prod_res *= get_loops_range(l);
    }
    return prod_res;
}

static int64_t cmp_loop_range(
        const for_loop &to_merge, const for_loop &be_merged) {
    if (!(to_merge->iter_begin_.isa<constant_c>()
                && to_merge->iter_end_.isa<constant_c>()
                && to_merge->step_.isa<constant_c>()
                && be_merged->iter_begin_.isa<constant_c>()
                && be_merged->iter_end_.isa<constant_c>()
                && be_merged->step_.isa<constant_c>())) {
        return 0;
    }
    auto to_merge_begin = get_expr_as_int(to_merge->iter_begin_),
         to_merge_end = get_expr_as_int(to_merge->iter_end_),
         to_merge_step = get_expr_as_int(to_merge->step_),
         be_merged_begin = get_expr_as_int(be_merged->iter_begin_),
         be_merged_end = get_expr_as_int(be_merged->iter_end_),
         be_merge_step = get_expr_as_int(be_merged->step_);
    return (to_merge_begin == be_merged_begin && to_merge_end == be_merged_end
                   && to_merge_step == be_merge_step)
            ? (to_merge_end - to_merge_begin)
            : 0;
}

static void try_align_parti_outer_loops(mixed_parti_t *A, mixed_parti_t *B) {
    auto outer_loops_A = A->get_outer_loops(),
         outer_loops_B = B->get_outer_loops();
    if (outer_loops_A.empty() || outer_loops_B.empty()) return;
    if (outer_loops_A.size() == outer_loops_B.size()) return;

    auto outermost_loop_A_range = get_loops_range(outer_loops_A[0]),
         outermost_loop_B_range = get_loops_range(outer_loops_B[0]);
    if (outermost_loop_A_range == outermost_loop_B_range) {
        return;
    } else if (outermost_loop_A_range <= outermost_loop_B_range) {
        B->try_split_outermost_loop(outermost_loop_A_range);
    } else {
        A->try_split_outermost_loop(outermost_loop_B_range);
    }
}

// check brgemm pre-op fusion
static bool try_merge_brgemm_and_preop_parti(
        mixed_parti_t *A, mixed_parti_t *B, bool check_connection = true) {
    A = A->get_root(), B = B->get_root();
    if (A == B) return false;
    if (!A->func_.get() || !B->func_.get()) return false;
    if (check_connection && !check_parti_connectionship(A, B)) return false;
    if (check_parti_ring_risk(A, B)) return false;
    auto dep_flag = check_parti_dep(A, B);
    if (dep_flag == parti_dep::inter_dep) return false;

    mixed_parti_t *brgemm_parti, *preop_parti;
    if (A->contain_tunable_op() && B->contain_elemwise_op_only()) {
        brgemm_parti = A;
        preop_parti = B;
    } else if (B->contain_tunable_op() && A->contain_elemwise_op_only()) {
        brgemm_parti = B;
        preop_parti = A;
    } else {
        return false;
    }

    if (check_parti_dep(brgemm_parti, preop_parti) == parti_dep::l_dep_r)
        return false;

    SC_MODULE_INFO << "pre-op merging two partition:";
    SC_MODULE_INFO << A->func_;
    SC_MODULE_INFO << B->func_;

    /* * * * * * * * * * * * * * * * *
     * Step 1: pre infer elementwise op using brgemm parti fusion anchor
     * * * * * * * * * * * * * * * * */
    for (auto &brgemm_parti_anchor : brgemm_parti->fanchors_) {
        fslice_map tmp_fsmap;
        for (auto iter = brgemm_parti_anchor->fsmap_.datamap_.begin();
                iter != brgemm_parti_anchor->fsmap_.datamap_.end();) {
            if (!brgemm_parti->buf_alloc_.g2b_map_.haskey(iter->first)) {
                auto brgemm_parti_cut_op = iter->first->producer_owner_;
                if (preop_parti->contains(brgemm_parti_cut_op)) {
                    if (brgemm_parti_cut_op
                            != preop_parti->committed_ops_.back().get()) {
                        SC_MODULE_INFO << "brgemm_parti_cut_op "
                                       << brgemm_parti_cut_op->op_name_
                                       << brgemm_parti_cut_op->logical_op_id_
                                       << " is not the end of preop_parti "
                                          "commited_ops.";
                    }
                    bool hit_cut_op = false;
                    infer_status_map_t stat_map(brgemm_parti->ctx_, false);
                    // set known slice range and prepare for pre-infer slice
                    // range
                    tmp_fsmap.get(iter->first) = iter->second;
                    // only ops before cut_op_iter will be infered with
                    // pre_slice
                    for (auto op_iter = preop_parti->committed_ops_.rbegin();
                            op_iter != preop_parti->committed_ops_.rend();
                            op_iter++) {
                        COMPILE_ASSERT((*op_iter)->isa<fusible_op_t>(),
                                "Only fusible op is expected on pre-op "
                                "partiion. "
                                "but got "
                                        << (*op_iter)->op_name_);
                        if ((*op_iter).get() == brgemm_parti_cut_op) {
                            hit_cut_op = true;
                        }
                        if (!hit_cut_op) continue;
                        (*op_iter)->stc_cast<fusible_op_t>()->pre_slice_ranges(
                                tmp_fsmap, stat_map);
                        COMPILE_ASSERT(stat_map.is_ok(),
                                "Elementwise ops are expected to infer "
                                "successfully")
                    }
                    iter++;
                } else {
                    iter = brgemm_parti_anchor->fsmap_.datamap_.erase(iter);
                }
            } else {
                iter++;
            }
        }
        // move fsmap
        brgemm_parti_anchor->fsmap_.datamap_.insert(
                tmp_fsmap.datamap_.begin(), tmp_fsmap.datamap_.end());
    }

    /* * * * * * * * * * * * * * * * *
     * Step 2: Commit ops in pre-op parti into brgemm parti by original order
     * * * * * * * * * * * * * * * * */
    brgemm_parti->func_->name_ += "_preop_merge";
    for (auto &op_in_preop_parti : preop_parti->committed_ops_) {
        brgemm_parti->add(op_in_preop_parti);
    }

    // erase joint op in op_anchor_map
    for (auto iter = brgemm_parti->op_anchor_map_.begin();
            iter != brgemm_parti->op_anchor_map_.end();) {
        if (!brgemm_parti->contains(iter->first)) {
            iter = brgemm_parti->op_anchor_map_.erase(iter);
        } else {
            iter++;
        }
    }

    /* * * * * * * * * * * * * * * * *
     * Step 3: Merge op_
     * * * * * * * * * * * * * * * * */
    // call base merge
    brgemm_parti->fusion_partition_t::merge(
            static_cast<fusion_partition_t *>(preop_parti)->shared_from_this());

    // Merge commited ops
    brgemm_parti->committed_ops_.insert(brgemm_parti->committed_ops_.end(),
            preop_parti->committed_ops_.begin(),
            preop_parti->committed_ops_.end());

    SC_MODULE_INFO << "pre-op merging result:";
    SC_MODULE_INFO << brgemm_parti->func_;

    // clear merged parti
    preop_parti->clear();

    return true;
}

static bool try_merge_mixed_parti_vertically(
        mixed_parti_t *A, mixed_parti_t *B, bool check_connection = true) {
    A = A->get_root(), B = B->get_root();
    if (A == B) return false;
    if (!A->func_.get() || !B->func_.get()) return false;
    if (check_connection && !check_parti_connectionship(A, B)) return false;
    if (check_connection && check_parti_ring_risk(A, B)) return false;
    auto dep_flag = check_parti_dep(A, B);
    // if two partition inter-depends each other, could not merge them
    if (dep_flag == parti_dep::inter_dep) return false;
    mixed_parti_t *pa_to_merge = nullptr, *parti_be_merged = nullptr;

    pa_to_merge = (dep_flag == parti_dep::l_dep_r) ? B : A;
    parti_be_merged = (dep_flag == parti_dep::l_dep_r) ? A : B;

    auto outer_loops_to_merge = pa_to_merge->get_outer_loops(),
         outer_loops_be_merged = parti_be_merged->get_outer_loops();

    if (outer_loops_to_merge.empty() || outer_loops_be_merged.empty())
        return false;

    // great common size
    auto gcs = std::min(
            outer_loops_to_merge.size(), outer_loops_be_merged.size());

    sc_dims parallel_loop_range;
    size_t merged_loop_size = 0;
    for (; merged_loop_size < gcs; merged_loop_size++) {
        auto common_range
                = cmp_loop_range(outer_loops_to_merge[merged_loop_size],
                        outer_loops_be_merged[merged_loop_size]);
        if (!common_range) break;
        parallel_loop_range.emplace_back(common_range);
    }

    if (!merged_loop_size) return false;

    // validate axis binding
    merged_loop_size = check_parti_loop_axis_binding(
            parti_be_merged, pa_to_merge, merged_loop_size);
    // reset parallel_loop_range
    parallel_loop_range = {parallel_loop_range.begin(),
            parallel_loop_range.begin() + merged_loop_size};

    if (!merged_loop_size) return false;

    if (pa_to_merge->ctx_->flags_.use_cost_model_) {
        auto new_parallelism
                = evaluate_loop_parallel_balance(parallel_loop_range),
                to_merge_parallelism
                = evaluate_loop_parallel_balance(outer_loops_to_merge),
                be_merged_parallelism
                = evaluate_loop_parallel_balance(outer_loops_be_merged);
        if (new_parallelism < to_merge_parallelism
                || new_parallelism < be_merged_parallelism) {
            SC_MODULE_INFO << "Cost model rejects to vertically merge two "
                              "partition: "
                           << A->func_->name_ << " and " << B->func_->name_
                           << " from perspective of parallellism";
            return false;
        }
    }

    // pre loaded buffer size tolerance in per core
    constexpr size_t pre_loaded_buffer_size_tol = 1024 * 1024;

    if (pa_to_merge->ctx_->flags_.use_cost_model_
            && (pa_to_merge->ops.size() != 1
                    && (pa_to_merge->buf_alloc_
                                       .get_total_allocated_buffer_size()
                               / runtime_config_t::get().get_num_threads())
                            > pre_loaded_buffer_size_tol)) {
        // TODO(xxx): enhance the logic here
        if (pa_to_merge->contain_op_with_type<reduce_impl_op_t>()) {
            SC_MODULE_INFO
                    << "Cost model ignores partition with reduce_compute: "
                    << A->func_->name_ << " and " << B->func_->name_
                    << " in cache inefficiencent case";
        } else if (pa_to_merge
                           ->contain_op_with_type<ops::conv_fwd_core_op_t>()) {
            SC_MODULE_INFO
                    << "Cost model ignores partition for convolution workload: "
                    << A->func_->name_ << " and " << B->func_->name_
                    << " in cache inefficiencent case";
        } else {
            SC_MODULE_INFO << "Cost model rejects to vertically merge two "
                              "partition: "
                           << A->func_->name_ << " and " << B->func_->name_
                           << " from perspective of cache efficiency";
            return false;
        }
    }

    // evaluate two partition by cost model
    float score_A = A->evaluate_perf(), score_B = B->evaluate_perf();

    SC_MODULE_INFO << "merging two partition:";
    SC_MODULE_INFO << A->func_;
    SC_MODULE_INFO << B->func_;

    /* * * * * * * * * * * * * * * * *
     * Step 1: Merge func_
     * * * * * * * * * * * * * * * * */
    auto max_to_merge_anchor_map = pa_to_merge->get_anchor_inside_loop(
            outer_loops_to_merge[merged_loop_size - 1]);
    auto max_be_merged_anchor_map = parti_be_merged->get_anchor_inside_loop(
            outer_loops_be_merged[merged_loop_size - 1]);
    auto max_be_merged_ss = outer_loops_be_merged[merged_loop_size - 1]->body_;
    auto max_to_merge_ss = outer_loops_to_merge[merged_loop_size - 1]->body_;
    // insert be_merged_ss to the back of to_merged_ss
    if (max_to_merge_anchor_map) {
        max_to_merge_anchor_map->fuse_anchor_map_t::commit_stmt(
                max_be_merged_ss);
    } else {
        return false;
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
        auto be_merged_anchor_map = parti_be_merged->get_anchor_inside_loop(
                     outer_loops_be_merged[i]),
             to_merge_anchor_map
                = pa_to_merge->get_anchor_inside_loop(outer_loops_to_merge[i]);
        if (be_merged_anchor_map) {
            be_merged_anchor_masks.insert(be_merged_anchor_map);
            if (to_merge_anchor_map) {
                to_merge_anchor_map->merge(be_merged_anchor_map);
            }
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
        pa_to_merge->append_fusion_anchor(be_merged_anchor_map);
    }

    /* * * * * * * * * * * * * * * * *
     * Step 3: Merge buf_alloc_
     * * * * * * * * * * * * * * * * */
    pa_to_merge->buf_alloc_.merge(parti_be_merged->buf_alloc_, buffer_map,
            std::make_pair(max_to_merge_anchor_map, max_be_merged_anchor_map));

    /* * * * * * * * * * * * * * * * * *
     * Step 4: Replace expr involving:
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

    // Merge commited ops
    pa_to_merge->committed_ops_.insert(pa_to_merge->committed_ops_.end(),
            parti_be_merged->committed_ops_.begin(),
            parti_be_merged->committed_ops_.end());

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

/**
 * @param check_connection: if the value is False, the connection ship should be
 * ensured by joint op.
 * */
void mixed_parti_t::merge(const ptr &other) {
    if (!try_merge_brgemm_and_preop_parti(this, other.get(), false)) {
        try_merge_mixed_parti_vertically(this, other.get(), false);
    }
}

mixed_parti_t::mixed_parti_t(
        const context_ptr &ctx, const sc_op_ptr &op, const dep_mat_ptr &dep_m)
    : dep_m_(dep_m), ctx_(ctx) {
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
    committed_ops_.emplace_back(op);
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
    // currently, disable tunable op commit into parallel for_loop.
    if (op->isa<tunable_op_t>()
            && contain_op_with_type<ops::managed_matmul_core_op_t>())
        return false;
    if (auto reo = op->dyn_cast<reorder_op_t>()) {
        if (reo->get_inputs()[0]->uses_.size() > 1) return false;
    }
    auto mixed_op = op->dyn_cast<op_traits::mixed_partition_acceptable>();
    mixed_op->search_anchor(this);
    if (!ready_for_op(op)) {
        SC_MODULE_INFO << op->op_name_ << "_" << op->logical_op_id_
                       << " fail to add partition: " << func_->name_
                       << ", due to no suitable anchor found";
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
        // auto skip
        if (iter->second == fanchor_map) return;
        auto res = cmp_op_anchor(op, iter->second, fanchor_map);
        // if new anchor is more smaller than the current one
        if (res == cmp_res::l_larger_r) {
            // overwrite new anchor
            op_anchor_map_[op] = fanchor_map;
        } else if (res == cmp_res::equal) {
            // Except for reduce_collect_op_t with COPY kind due to this mode
            // will elimiate first required axis
            if (auto red_coll = op->dyn_cast<reduce_collect_op_t>()) {
                if (red_coll->op_ == reduce_collect_op_t::kind::COPY) return;
            }
            // if sub anchor is equal to parent one, overwrite it
            if (iter->second->is_parent_for(fanchor_map)) {
                // overwrite inner anchor
                op_anchor_map_[op] = fanchor_map;
            }
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
    committed_ops_.emplace_back(op);
}

fuse_anchor_map_ptr mixed_parti_t::lookup_anchor_map(sc_op *op) const {
    if (merged_to) { return get_root()->lookup_anchor_map(op); }
    auto iter = op_anchor_map_.find(op);
    COMPILE_ASSERT(iter != op_anchor_map_.end(),
            "No dispatched fusion anchor map found for "
                    << op->op_name_
                    << " in this partition, please try to search it firstly");
    return iter->second;
}

fuse_anchor_map_ptr mixed_parti_t::lookup_anchor_map(const stmts &ss) const {
    if (merged_to) { return get_root()->lookup_anchor_map(ss); }
    auto iter = std::find_if(fanchors_.begin(), fanchors_.end(),
            [&ss](const fuse_anchor_map_ptr &amap) {
                return ss.ptr_same(amap->anchor_position_);
            });
    return (iter != fanchors_.end()) ? (*iter) : nullptr;
}

std::vector<fuse_anchor_map_ptr> mixed_parti_t::lookup_sub_anchor_map(
        const fuse_anchor_map_ptr &parent_fanchor) const {
    if (merged_to) { return get_root()->lookup_sub_anchor_map(parent_fanchor); }
    std::vector<fuse_anchor_map_ptr> subs;
    for (auto &fanc : fanchors_) {
        if (fanc->parent_ == parent_fanchor) { subs.emplace_back(fanc); }
    }
    return subs;
}

void mixed_parti_t::clear_fanchor(fuse_anchor_map_ptr &fanchor) {
    stmt anchor = fanchor->anchor_position_;
    COMPILE_ASSERT(anchor.checked_as<stmts>()->seq_.empty(),
            "Could not remove this fanchor, due to it is not empty")
    stmt parent = get_parent_node(anchor);
    auto ss_parent = parent.checked_as<stmts>();
    // clear empty if_node outside iter anchor if necessary
    if (fanchor->isa<fuse_iter_anchor_map_t>() && ss_parent->seq_.size() == 1
            && get_parent_node(parent).isa<if_else>()) {
        auto if_node = get_parent_node(parent);
        // redirect
        parent = get_parent_node(if_node);
        ss_parent = parent.checked_as<stmts>();
        anchor = if_node;
    }
    // find anchor iter
    std::vector<sc::stmt>::iterator anchor_iter
            = std::find_if(ss_parent->seq_.begin(), ss_parent->seq_.end(),
                    [anchor](stmt &s) { return s.ptr_same(anchor); });
    COMPILE_ASSERT(anchor_iter != ss_parent->seq_.end(),
            "Could not found anchor in current parent stmts");
    // remove anchor
    ss_parent->seq_.erase(anchor_iter);
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
    if (merged_to) { return get_root()->get_outer_loops(fanchor); }
    auto body = func_->body_;
    std::vector<for_loop> outer_loops;

    fuse_anchor_map_ptr target_fanchor = std::move(fanchor);
    while (target_fanchor && target_fanchor->parent_) {
        target_fanchor = target_fanchor->parent_;
    }

    if (body.isa<stmts>()
            && (body.checked_as<stmts>()->seq_.size() == 1
                    || body.checked_as<stmts>()->seq_.size() == 2)) {
        if (body.checked_as<stmts>()->seq_.size() == 2) {
            if (!body.checked_as<stmts>()->seq_[1].isa<returns>())
                return outer_loops;
        }
        auto st = body.checked_as<stmts>()->seq_[0];
        if (st.isa<for_loop>()) {
            auto loop = st.static_as<for_loop>();
            while (loop.defined()) {
                outer_loops.emplace_back(loop);
                loop = get_next_inner_loop_with_anchor(loop, target_fanchor);
            }
        }
    }
    return outer_loops;
}

/**
 * for_loop(){
 *   // body
 *   ...
 *   // anchor
 *   {}
 * }
 * */
fuse_anchor_map_ptr mixed_parti_t::get_anchor_inside_loop(
        const for_loop &loop) const {
    auto &body = loop->body_;
    if (body.isa<stmts>()) {
        auto ss = body.static_as<stmts>();
        for (auto s : ss->seq_) {
            // find anchor inside if-node
            if (s.isa<if_else>()) {
                auto if_node = s.static_as<if_else>();
                if (!if_node->then_case_.defined()
                        || if_node->else_case_.defined())
                    continue;
                auto then_stmts = if_node->then_case_.static_as<stmts>();
                if (then_stmts->seq_.size() == 1)
                    s = then_stmts->seq_[0];
                else
                    continue;
            }
            if (!s.isa<stmts>()) continue;
            auto inner_ss = s.static_as<stmts>();
            auto anchor_map = lookup_anchor_map(inner_ss);
            if (anchor_map) return anchor_map;
        }
    }
    return nullptr;
}

for_loop mixed_parti_t::get_next_inner_loop_with_anchor(
        const for_loop &cur_loop,
        const fuse_anchor_map_ptr &target_fanchor) const {
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
                auto anchor_map = lookup_anchor_map(inner_ss);
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
}

void mixed_parti_t::try_split_outermost_loop(int64_t block) {
    auto outer_loops = get_outer_loops();
    if (outer_loops.empty()) return;
    auto outermost_loop = outer_loops[0];
    auto outermost_loop_range = get_expr_as_int(outermost_loop->iter_end_)
            - get_expr_as_int(outermost_loop->iter_begin_);
    if (outermost_loop_range % block != 0) return;
    if (!outermost_loop->step_.isa<constant>()
            || get_expr_as_int(outermost_loop->step_) != 1)
        return;

    std::unordered_map<expr, expr> remap;
    // change IR and record replace map
    outermost_loop->split(outermost_loop_range / block, &remap);
    // split outer loop axis binder
    ax_binder_.split_init_axis(0);
    mxp_replacer_t(remap).replace_anchor(fanchors_);
}

void mixed_parti_t::buffer_schedule() {
    if (merged_to) {
        get_root()->buffer_schedule();
        return;
    }
    buf_alloc_.tensor_initialize();
    buf_alloc_.declare_and_shrink_tensor();
    buf_alloc_.query_buffer_inplace();
}

bool mixed_parti_t::is_parti_inp(const graph_tensor *gt) {
    if (merged_to) { return get_root()->is_parti_inp(gt); }
    auto ths = this;
    return std::any_of(gt->uses_.begin(), gt->uses_.end(),
                   [&ths](const std::pair<int, sc::sc_op_weak_ptr_t> &user) {
                       return ths->contains(user.second.get());
                   })
            && !contains(gt->producer_owner_);
}

bool mixed_parti_t::is_parti_inp(const graph_tensor_ptr &gt) {
    return is_parti_inp(gt.get());
}

bool mixed_parti_t::is_parti_out(const graph_tensor *gt) {
    if (merged_to) { return get_root()->is_parti_out(gt); }
    auto ths = this;
    return contains(gt->producer_owner_)
            && std::any_of(gt->uses_.begin(), gt->uses_.end(),
                    [&ths](const std::pair<int, sc::sc_op_weak_ptr_t> &user) {
                        return !ths->contains(user.second.get());
                    });
}

bool mixed_parti_t::is_parti_out(const graph_tensor_ptr &gt) {
    return is_parti_out(gt.get());
}

template <typename T>
size_t mixed_parti_t::count_op_with_type() const {
    if (merged_to) { return get_root()->count_op_with_type<T>(); }
    size_t cnt = 0;
    for (auto &op : ops) {
        if (op->isa<T>()) cnt++;
    }
    return cnt;
}

template <typename T>
bool mixed_parti_t::contain_op_with_type() const {
    if (merged_to) { return get_root()->contain_op_with_type<T>(); }
    return (count_op_with_type<T>() != 0);
}

bool mixed_parti_t::is_conv_workload() const {
    return contain_op_with_type<ops::conv_fwd_core_op_t>()
            || contain_op_with_type<ops::conv_bwd_data_core_op_t>()
            || contain_op_with_type<ops::conv_bwd_weight_core_op_t>();
}

bool mixed_parti_t::contain_tunable_op() const {
    return contain_op_with_type<tunable_op_t>();
}

bool mixed_parti_t::contain_elemwise_op_only() const {
    if (merged_to) { return get_root()->contain_elemwise_op_only(); }
    for (auto &op : ops) {
        if (!op->isa<unary_elementwise_op_t>()
                && !op->isa<binary_elementwise_op_t>())
            return false;
    }
    return true;
}

void mixed_parti_t::clear() {
    // Graph-related
    ops.clear();
    committed_ops_.clear();
    ax_binder_.clear();

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

static bool check_partition(const mixed_parti_t::ptr &parti,
        std::unordered_set<sc_op_ptr> &retry_ops) {
    auto run_threads = runtime_config_t::get().get_num_threads();
    if (!parti || parti->ops.empty() || parti->ops.size() < 2) return true;
    bool legal = true;
    // check tensorview in edge of partition
    for (auto &op : parti->ops) {
        if (op->isa<tensor_view_op_t>()) {
            if ((parti->is_parti_out(op->get_outputs()[0])
                        && parti->buf_alloc_.g2b_map_.get(op->get_outputs()[0])
                                   .isa<tensorptr>())
                    || parti->is_parti_out(op->get_inputs()[0])) {
                op->attrs_[op_attr_key::no_fuse] = true;
                retry_ops.insert(op);
                legal = false;
            }
        }
    }
    // check reorder in partition which includes reduce op but exclude
    // tunable op
    if (std::all_of(parti->ops.begin(), parti->ops.end(),
                [](const sc_op_ptr &op) { return !op->isa<tunable_op_t>(); })
            && std::any_of(parti->ops.begin(), parti->ops.end(),
                    [](const sc_op_ptr &op) { return op->isa<reduce_op_t>(); })
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
                auto shape
                        = rd_op->get_inputs()[0]->details_.get_blocking_dims();
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
                    [&retry_ops, &legal](const sc_op_ptr &op) {
                        op->attrs_[op_attr_key::no_fuse] = true;
                        retry_ops.insert(op);
                        legal = false;
                    });
        }
    }
    return legal;
}

static bool do_partition(const context_ptr &ctx, sc_graph_t &g,
        const op_dep_matrix_t &dep,
        std::vector<mixed_parti_t::ptr> &op_2_partition) {
    // a speculative DFS visitor
    op_visitor_t visitor
            = op_visitor_t::dfs_topology_speculative_sort(g.ops_.size());
    visitor.visit_graph(g, [&](const sc_op_ptr &op) {
        if (op->isa<input_op>() || op->isa<output_op>()
                || op->attrs_.get_or_else(op_attr_key::no_fuse, false))
            return;

        mixed_parti_t::ptr parent_partition = nullptr;
        if (!op->attrs_.get_or_else(op_attr_key::break_pre_fuse, false)) {
            // try to preop merge the partitons of all inputs in advance
            for (auto &in : op->get_inputs()) {
                if (!in->producer_owner_->attrs_.get_or_else(
                            op_attr_key::break_post_fuse, false)
                        && in->producer_owner_->attrs_.get_or_else(
                                   "constant", const_kind::not_const)
                                == const_kind::not_const) {
                    auto &cur_in_partition
                            = op_2_partition[in->producer_owner_
                                                     ->logical_op_id_];
                    if (!parent_partition) {
                        parent_partition = cur_in_partition;
                    } else if (!parent_partition->empty() && cur_in_partition
                            && !cur_in_partition->empty()) {
                        COMPILE_ASSERT(
                                op->isa<op_traits::
                                                mixed_partition_acceptable>(),
                                "Unexpected op type find: " << op->op_name_)
                        op->dyn_cast<op_traits::mixed_partition_acceptable>()
                                ->search_anchor(parent_partition.get());
                        op->dyn_cast<op_traits::mixed_partition_acceptable>()
                                ->search_anchor(cur_in_partition.get());
                        if (parent_partition->ready_for_op(op.get())
                                && cur_in_partition->ready_for_op(op.get())) {
                            // the difference between later partition merge is
                            // that current merge need to check partition
                            // connection to ensure legality.
                            try_merge_brgemm_and_preop_parti(
                                    parent_partition.get(),
                                    cur_in_partition.get(),
                                    /*check connection*/ true);
                        }
                    }
                }
            }
            // reset nullptr
            parent_partition = nullptr;

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
                            cur_in_partition->merge(parent_partition);
                        } else {
                            if (parent_partition->contain_tunable_op()) {
                                cur_in_partition->merge(parent_partition);
                            } else {
                                parent_partition->merge(cur_in_partition);
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
            parent_partition = std::make_shared<mixed_parti_t>(
                    ctx, op, std::make_shared<op_dep_matrix_t>(dep));
        } else {
            parent_partition->get_root()->add(op);
        }
        op_2_partition[op->logical_op_id_] = parent_partition;
    });

    // legalize
    std::unordered_set<sc_op_ptr> retry_ops;
    for (auto &parti : op_2_partition) {
        if (!check_partition(parti, retry_ops)) {
            SC_MODULE_INFO
                    << "partition: " << parti->func_->name_
                    << " will do repartition due to illegal pattern detected";
        }
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

static bool validate_optimized_reduce(
        const std::shared_ptr<mixed_parti_t> &parti) {
    for (auto &op : parti->ops) {
        if (auto coll = op->dyn_cast<reduce_collect_op_t>()) {
            if (coll->op_ == reduce_collect_op_t::kind::COPY) {
                auto gt = coll->get_inputs()[0];
                COMPILE_ASSERT(gt->producer_owner_->isa<reduce_compute_op_t>(),
                        "reduce_collect_op is only expected follow "
                        "reduce_compute_op");
                COMPILE_ASSERT(parti->buf_alloc_.g2b_map_.haskey(gt),
                        "Could not found allocated buffer for "
                                << coll->op_name_)
                auto buf = parti->buf_alloc_.g2b_map_.get(gt);
                auto shrinked_info = parti->buf_alloc_.get_shrinked_info(buf);
                if (shrinked_info.empty()) continue;
                auto shape = get_slice_shape(shrinked_info);
                auto prod = get_dims_product(get_expr_to_dims(shape));
                auto tsr_simd_len = vectorize_step(parti->ctx_,
                        get_real_tensor(buf)->elem_dtype_.type_code_);
                constexpr int max_register_tol = 16;
                if (prod % tsr_simd_len != 0
                        || (prod / tsr_simd_len) >= max_register_tol)
                    return false;
            }
        }
    }
    return true;
}

static bool try_optimize_reduce(const context_ptr &ctx, sc_graph_t &g,
        std::unordered_map<graph_tensor_ptr, graph_tensor_ptr> &graph2orig) {
    bool reduce_contained = false;
    std::unordered_set<sc_op_ptr> tunable_op_set;
    for (auto &op : g.ops_) {
        if (op->isa<tunable_op_t>()) {
            tunable_op_set.insert(op);
        } else if (op->isa<op_traits::maybe_split_optimized_t>()) {
            reduce_contained = true;
        }
    }

    if (!reduce_contained) return false;

    bool redo = false;

    auto split_rd_op = [&redo](const context_ptr &ctx, sc_graph_t &g, sc_op *op,
                               std::unordered_map<graph_tensor_ptr,
                                       graph_tensor_ptr> &graph2orig) {
        COMPILE_ASSERT(op->isa<op_traits::maybe_split_optimized_t>(),
                op->op_name_ << " could not be split")
        auto orig_out = op->get_outputs()[0];
        auto new_out
                = op->dyn_cast<op_traits::maybe_split_optimized_t>()->split_op(
                        ctx, g, 1);
        auto iter = graph2orig.find(orig_out);
        if (iter != graph2orig.end()) {
            if (orig_out->attrs_.get_or_else("temp.mixed_partition_hint.no_"
                                             "inplace",
                        false)) {
                new_out->attrs_
                        ["temp.mixed_partition_hint.no_"
                         "inplace"]
                        = true;
            }
            auto orig_v = iter->second;
            graph2orig.erase(iter);
            graph2orig.insert(std::make_pair(new_out, orig_v));
        }
        redo = true;
    };

    if (!tunable_op_set.empty()) {
        auto ops = g.ops_;
        op_dep_matrix_t dep(g);
        std::unordered_set<op_traits::maybe_split_optimized_t *>
                splited_reduce_set;

        std::for_each(ops.begin(), ops.end(),
                [&dep, &tunable_op_set, &splited_reduce_set](
                        const sc_op_ptr &op) {
                    if (auto red_op = op->dyn_cast<
                                      op_traits::maybe_split_optimized_t>()) {
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
                    [&g, &ctx, &graph2orig, &split_rd_op](
                            op_traits::maybe_split_optimized_t *red_op) {
                        auto op = dynamic_cast<sc_op *>(red_op);
                        // pre-check
                        if (auto rd_op = op->dyn_cast<reduce_compute_op_t>()) {
                            auto rd_axis = rd_op->get_rd_axis();
                            rd_axis.erase(rd_axis.begin());
                            if (!rd_axis.empty()) {
                                sc_dim prod = 1;
                                for (auto &ax : rd_axis) {
                                    prod *= rd_op->get_inputs()[0]
                                                    ->details_
                                                    .get_blocking_dims()[ax];
                                }
                                auto tsr_simd_len = vectorize_step(ctx,
                                        rd_op->get_inputs()[0]
                                                ->details_.dtype_.type_code_);
                                constexpr int max_register_tol = 16;
                                if (prod % tsr_simd_len != 0
                                        || (prod / tsr_simd_len)
                                                >= max_register_tol)
                                    return;
                            }
                        }
                        if (red_op->can_split_op()) {
                            split_rd_op(ctx, g, op, graph2orig);
                        }
                    });
            return redo;
        } else {
            return false;
        }
    } else if (runtime_config_t::get().get_num_threads() == 1) {
        auto old_ops = g.ops_;
        for (auto &op : old_ops) {
            if (auto rd = op->dyn_cast<reduce_op_t>()) {
                if (rd->can_split_op()) {
                    split_rd_op(ctx, g, op.get(), graph2orig);
                }
            }
        }
    }

    // try to give sub graph hint for outer loop generator if necessary
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
                op->attrs_[mixed_partition_hint::sub_graph_ptr] = &g;
            }
        });
        return true;
    }

    return redo;
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
    bool redo_flag = false;
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
                auto out_buffer = parti.buf_alloc_.g2b_map_.get(out);
                if (out_buffer.isa<tensorptr>()) {
                    redo_flag = true;
                    outtsr->attrs_[mixed_partition_hint::no_inplace] = true;
                }
                arg_out.emplace_back(out_buffer);
            }
        }
        auto copyable = op->dyn_cast<op_traits::copyable_t>();
        assert(copyable);
        auto copied = copyable->copy(new_graph_in, new_graph_ou, sub_graph);

        // build the  fused op name
        if (!op_name.empty()) op_name += '_';
        op_name += copied->op_name_;
    });

    std::shared_ptr<mixed_fuse_op_t> fused_op = nullptr;

    // prepare for redo fall-back case
    auto copied_grpah = copy_graph(sub_graph);

    // currently, only redo one time
    redo_flag = (!g.attrs_.get_or_else(
                        mixed_partition_hint::retried_graph, false))
            && (redo_flag || try_optimize_reduce(ctx, sub_graph, graph_2_orig));

    if (redo_flag) {
        sub_graph.attrs_[mixed_partition_hint::retried_graph] = true;
        SC_MODULE_INFO << "Redo mixed partition for current pattern: "
                       << partition->func_->name_;
        // redo mixed partition with setting hint
        do_mixed_partition(ctx, sub_graph);
        for (auto &op : sub_graph.ops_) {
            if (auto mx_op = op->dyn_cast<mixed_fuse_op_t>()) {
                if (fused_op) {
                    // reset
                    fused_op = nullptr;
                    SC_MODULE_WARN << "It is expected that only one mixed "
                                      "fused op after "
                                      "mixed partition redo, fall-back to "
                                      "original pattern "
                                      "which may cause performance dropping, "
                                      "please check it ";
                    break;
                }
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
                if (!op->isa<input_op>() && !op->isa<output_op>()
                        && !op->isa<constant_op_t>()) {
                    // reset
                    fused_op = nullptr;
                    SC_MODULE_WARN
                            << "Unexpected op type found after mixed partition "
                               "redo: "
                            << op->op_name_
                            << ", fall-back to original pattern which may "
                               "cause performance dropping, please check it";
                    break;
                }
            }
        }
        if (fused_op) {
            // redo validation stage
            if (!validate_optimized_reduce(fused_op->parti_)) {
                // reset
                fused_op = nullptr;
                SC_MODULE_WARN << "invalid optmized reduce detected, fall-back "
                                  "to original pattern";
            }
        }
    }
    if (!fused_op) {
        // mark read/write buffer
        graph::mark_read_or_write_buffers(arg_ins, true);
        graph::mark_read_or_write_buffers(arg_out, false);
        // build up final func name and param
        std::vector<expr> args = arg_out;
        args.insert(args.end(), arg_ins.begin(), arg_ins.end());
        std::for_each(args.begin(), args.end(), [](const expr &arg) {
            COMPILE_ASSERT(arg.isa<tensor>(),
                    "Only tensor node is expected for function argument, but "
                    "got " << arg)
            arg->attr()[mixed_partition_hint::cut_buffer] = true;
        });
        parti.func_->params_ = args;
        parti.func_->decl_->params_ = args;

        fused_op = std::make_shared<mixed_fuse_op_t>(op_name, partition,
                copied_grpah,
                /*ins*/ fused_op_in,
                /*outs*/
                fused_op_out, any_map_t {});
    }

    if (!g.attrs_.get_or_else(mixed_partition_hint::retried_graph, false)) {
        /** Freeze IR **/
        auto &fparti = *(fused_op->parti_);

        // buffer schedule: declare, set shrink info and tensor initilize.
        fparti.buffer_schedule();

        // set function name
        std::string prefix;
        if (fparti.ops.size() > 1) {
            prefix = "partition_";
            auto outer_loops = fparti.get_outer_loops();
            if (!outer_loops.empty()) {
                prefix = "outerloop_" + print_loops_range(outer_loops) + "_"
                        + prefix;
            }
        }
        fparti.func_->name_ = prefix + fused_op->op_name_;
        fparti.func_->decl_->name_ = fparti.func_->name_;
        // link op name
        fused_op->op_name_ = fparti.func_->name_;

        // clear unused fanchor, inavoid of loop fuse break
        fparti.clear_fanchors();

        // push return to the end of body
        auto ret = builder::make_returns_unattached(true);
        fparti.func_->body_.checked_as<stmts>()->seq_.emplace_back(ret);

        // remove all parallel flag
        remove_parallel(fparti.func_, false);

        SC_MODULE_INFO << "mixed partition result:";
        SC_MODULE_INFO << fparti.func_;
    } else {
        g.attrs_.remove(mixed_partition_hint::retried_graph);
    }

    return fused_op;
}

using crossover_alg
        = std::function<void(const std::vector<mixed_parti_t::ptr> &parti_vec)>;

static void horizontal_crossover(
        const std::vector<mixed_parti_t::ptr> &parti_vec) {
    SC_MODULE_INFO << "Applying horizontal merge crossover algorithm...";
    auto op_size = parti_vec.size();
    for (size_t i = 0; i < op_size; i++) {
        auto parti_A = parti_vec[i];
        if (!parti_A) continue;
        for (size_t j = i; j < op_size; j++) {
            auto parti_B = parti_vec[j];
            if (!parti_B) continue;
            try_merge_mixed_parti_horizontally(parti_A.get(), parti_B.get());
        }
    }
}

static void parallel_crossover(
        const std::vector<mixed_parti_t::ptr> &parti_vec) {
    SC_MODULE_INFO << "Applying parallel merge crossover algorithm...";
    auto op_size = parti_vec.size();
    for (size_t i = 0; i < op_size; i++) {
        auto parti_A = parti_vec[i];
        if (!parti_A) continue;
        for (size_t j = i; j < op_size; j++) {
            auto parti_B = parti_vec[j];
            if (!parti_B) continue;
            try_merge_mixed_parti_parallel(parti_A.get(), parti_B.get());
        }
    }
}

static void vertical_crossover(
        const std::vector<mixed_parti_t::ptr> &parti_vec) {
    SC_MODULE_INFO << "Applying vertical merge crossover algorithm...";
    auto op_size = parti_vec.size();
    for (size_t i = 0; i < op_size; i++) {
        auto parti_A = parti_vec[i];
        if (!parti_A) continue;
        for (size_t j = i; j < op_size; j++) {
            auto parti_B = parti_vec[j];
            if (!parti_B) continue;
            // currently only apply on conv image affine
            if (parti_A->contain_op_with_type<ops::conv_fwd_core_op_t>()
                    && parti_B->contain_op_with_type<ops::conv_fwd_core_op_t>())
                try_merge_mixed_parti_vertically(parti_A.get(), parti_B.get());
        }
    }
}

static void crossover_partition(std::vector<mixed_parti_t::ptr> &op_2_partition,
        const std::vector<crossover_alg> &algs) {
    std::unordered_set<mixed_parti_t::ptr> parti_set;
    for (auto &parti : op_2_partition) {
        // auto skip const parti
        if (parti && parti->ops.size() == 1
                && parti->committed_ops_[0]->attrs_.get_or_else(
                           "constant", const_kind::not_const)
                        != const_kind::not_const)
            continue;
        parti_set.insert(parti);
    }
    std::vector<mixed_parti_t::ptr> parti_vec {
            parti_set.begin(), parti_set.end()};
    for (auto &al : algs) {
        al(parti_vec);
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

    constexpr int maxiter = 10;
    for (int i = 0; i < maxiter; i++) {
        op_2_partition.clear();
        op_2_partition.resize(op_size);
        if (do_partition(ctx, graph, dep, op_2_partition)) break;
    }

    std::vector<crossover_alg> algs
            = {horizontal_crossover, parallel_crossover, vertical_crossover};
    crossover_partition(op_2_partition, algs);

    std::vector<sc_op_ptr> fused_ops;
    for (auto &parti : op_2_partition) {
        if (!parti || !parti->output_replace_map.empty() || (parti->ops.empty())
                || (parti->ops.size() == 1
                        && !parti->ops.begin()->get()->isa<padding_op_t>())) {
            // if a partition has been processed or it is a single op, skip
            continue;
        }

        auto fused_op = transform_pa_to_mixed_op(ctx, graph, parti);

        fused_op->attrs_[mixed_partition_hint::parti]
                = std::weak_ptr<mixed_parti_t>(parti);
        fused_ops.emplace_back(fused_op);
    }

    std::unordered_map<graph_tensor_ptr, graph_tensor_ptr> tsr_replace_map;
    for (auto &fused_op : fused_ops) {
        auto partition = fused_op->attrs_[mixed_partition_hint::parti]
                                 .get<std::weak_ptr<mixed_parti_t>>()
                                 .lock();
        assert(partition);
        fused_op->attrs_.remove(mixed_partition_hint::parti);
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
            if (op->attrs_.has_key(mixed_partition_hint::sub_graph_ptr)) {
                op->attrs_.remove(mixed_partition_hint::sub_graph_ptr);
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
}

} // namespace sc
