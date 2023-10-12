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

#include "mixed_partition.hpp"
#include <algorithm>
#include <string>
#include "pass/pass.hpp"
#include "transform/transform.hpp"
#include "tunable_op.hpp"
#include "utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/ir/transform/concat_memory_planning.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/dyn_tsr_transform.hpp>
#include <compiler/ir/transform/index_flatten.hpp>
#include <compiler/ir/transform/loop_transform.hpp>
#include <compiler/ir/transform/pointer_alias_info.hpp>
#include <compiler/ir/transform/scope_flatten.hpp>
#include <compiler/ir/transform/tensor2var.hpp>
#include <compiler/ir/transform/tensor_inplace_info.hpp>
#include <compiler/ir/viewer.hpp>
#include <compiler/ir/visitor.hpp>
#include <ops/convolution.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/padding.hpp>
#include <ops/fusible/pooling.hpp>
#include <ops/fusible/reduce.hpp>
#include <ops/managed_matmul_core.hpp>
#include <runtime/config.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

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

namespace graph {
void tensor_detail_to_ir_tensor(sc_graph_t &graph, const std::string &name,
        const graph_tensor_ptr &gt, mxp_buffer_allocator *buf_alloc) {
    COMPILE_ASSERT(buf_alloc, "No buffer allocator found")
    bool is_cost_model_enabled
            = buf_alloc->get_binded_mxp()->cost_->is_enabled();
    if (!buf_alloc->g2b_map_.haskey(gt)) {
        auto tsr = graph::tensor_detail_to_ir_tensor(graph, name, gt->details_);
        buf_alloc->g2b_map_.get(gt) = tsr;
        if (!graph.is_dynamic() && is_cost_model_enabled) {
            auto dim_prod = get_dims_product(
                    get_expr_to_dims(tsr.checked_as<tensor>()->dims_));
            auto dtype_size = utils::get_sizeof_etype(
                    tsr.checked_as<tensor>()->elem_dtype_.type_code_);
            // first touch
            buf_alloc->mem_trace_.emplace_back(
                    memory_optim::memory_alloc_trace_t {(uintptr_t)tsr.get(),
                            (size_t)dim_prod * dtype_size});
            // last use
            buf_alloc->mem_trace_.emplace_back(
                    memory_optim::memory_alloc_trace_t {
                            (uintptr_t)tsr.get(), (size_t)0});
        }
    } else if (!graph.is_dynamic() && is_cost_model_enabled) {
        auto tsr = get_real_tensor(buf_alloc->g2b_map_.get(gt));
        // update last use trace
        auto last_trace = std::remove_if(buf_alloc->mem_trace_.begin(),
                buf_alloc->mem_trace_.end(),
                [&tsr](const memory_optim::memory_alloc_trace_t &trace) {
                    return ((uintptr_t)tsr.get() == trace.buffer_id_)
                            && (trace.size_ == 0);
                });
        COMPILE_ASSERT((last_trace + 1) == buf_alloc->mem_trace_.end(),
                "Not found last use trace for: " << tsr);
        (*last_trace) = memory_optim::memory_alloc_trace_t {
                (uintptr_t)tsr.get(), (size_t)0};
    }
}

void tensor_detail_to_ir_tensor(sc_graph_t &graph,
        const std::string &name_prefix,
        const std::vector<graph_tensor_ptr> &tsrs,
        mxp_buffer_allocator *buf_alloc) {
    COMPILE_ASSERT(buf_alloc, "No buffer allocator found")
    for (size_t i = 0; i < tsrs.size(); i++) {
        tensor_detail_to_ir_tensor(
                graph, name_prefix + std::to_string(i), tsrs[i], buf_alloc);
    }
}
} // namespace graph

mixed_fuse_op_t *get_mixed_op_from_graph(sc_graph_t &graph) {
    mixed_fuse_op_t *mixed_op = nullptr;
    for (auto &op : graph.ops_) {
        if (auto mx_op = op->dyn_cast<mixed_fuse_op_t>()) {
            COMPILE_ASSERT(!mixed_op, "Only one fused op is expected")
            mixed_op = mx_op;
        }
    }
    return mixed_op;
}

void mxp_buffer_allocator::set_buffer_inplace_hint(
        const expr &target_buf, const expr &inplace_buf) {
    // skip dynamic cases
    // Buffer inplace currently does not support dynamic buffers
    if (binded_mxp_->get_host_graph().is_dynamic()) { return; }
    COMPILE_ASSERT(target_buf.defined() && inplace_buf.defined(),
            "Both buffer should be defined")
    // skip same buffer
    if (inplace_buf.ptr_same(target_buf)) return;
    auto target_id = alias_info::get_or_create_alias_info(*target_buf.get());
    SC_MODULE_INFO << "Mark inplace hint for buffer: " << inplace_buf << " ==> "
                   << target_buf;
    inplace_buf->attr()[attr_keys::tensor_inplace_hint]
            = std::vector<temp_tensor_inplace_info_t> {
                    {target_id, inplace_kind::ZERO_OFFSET}};
    // update inner inaplce map
    inplace_map_[(uintptr_t)inplace_buf.get()]
            = std::vector<std::pair<uintptr_t, inplace_kind>> {
                    {(uintptr_t)target_buf.get(), inplace_kind::ZERO_OFFSET}};
}

void mxp_buffer_allocator::allocate_buffer(sc_op *op) {
    auto &graph = op->get_owner_graph();
    // allocate input buffer
    graph::tensor_detail_to_ir_tensor(graph,
            op->op_name_ + "_" + std::to_string(op->logical_op_id_) + "_ins_",
            op->get_inputs(), this);

    /* deal with special ops: explict inplace input */
    // tensorview op
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
            if (graph.is_dynamic()) {
                // reset plain dims hint as do inplacement here.
                auto &tsr = g2b_map_.get(inp);
                tsr->attr().set(attr_keys::plain_dims,
                        graph.dims_to_expr(
                                op->get_outputs()[0]
                                        ->details_.get_plain_dims()));
            }
            g2b_map_.get(op->get_outputs()[0]) = g2b_map_.get(inp);
        } else {
            auto base_tsr = get_real_tensor(g2b_map_.get(inp));
            g2b_map_.get(op->get_outputs()[0]) = builder::tensor_ptr(base_tsr,
                    std::vector<expr>(base_tsr->dims_.size(), 0),
                    tv_op->get_shapes_expr());
        }
    }

    // reduce collect op
    if (auto collc_op = op->dyn_cast<reduce_collect_op_t>()) {
        if (collc_op->is_place_holder_op()) {
            // inplace reduce_compute_op output
            COMPILE_ASSERT(
                    op->get_inputs()[0]
                            ->producer_owner_->isa<reduce_compute_op_t>(),
                    "reduce collect op is expected to follow reduce compute "
                    "op, but got "
                            << op->get_inputs()[0]->producer_owner_->op_name_)
            // no code generated
            g2b_map_.get(op->get_outputs()[0])
                    = g2b_map_.get(op->get_inputs()[0]);
        }
    }

    // allocate output buffer
    graph::tensor_detail_to_ir_tensor(graph,
            op->op_name_ + "_" + std::to_string(op->logical_op_id_) + "_outs_",
            op->get_outputs(), this);

    // reorder op
    if (auto reo_op = op->dyn_cast<reorder_op_t>()) {
        if (reo_op->check_padding()) {
            op->get_outputs()[0]->attrs_.set(
                    mixed_partition_hint::no_inplace, true);
        }
    }

    /* infer post-op inplace */
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
            continue;
        }
        // query input
        for (auto &inp : op->get_inputs()) {
            if (inp->attrs_.get_or_else(
                        mixed_partition_hint::no_inplace, false)) {
                continue;
            }
            if (query_inplace(out, inp)) {
                // set buffer inplace hint for output buffer
                set_buffer_inplace_hint(g2b_map_.get(inp), g2b_map_.get(out));
                break;
            }
        }
    }

    /* deal with special ops: set tensor initial value */
    // reduce collect and compute op
    if (auto rd_impl_op = op->dyn_cast<reduce_impl_op_t>()) {
        auto buf = g2b_map_.get(op->get_outputs()[0]);
        COMPILE_ASSERT(buf.isa<tensor>(),
                "output of reduce_impl op should be tensor type")
        rd_impl_op->set_reduce_buffer(buf.checked_as<tensor>());
    }

    /* infer pre-op inplace */
    if (op->isa<padding_op_t>() && op->get_inputs()[0]->uses_.size() == 1
            && !binded_mxp_->empty()) {
        auto out = op->get_outputs()[0];
        auto ins = op->get_inputs()[0];
        auto old_input = g2b_map_.get(ins);
        if (old_input.isa<tensor>()
                || (old_input.isa<tensorptr>() && ins->uses_.size() == 1
                        && utils::is_one_of(
                                old_input.static_as<tensorptr>()->base_->dtype_,
                                sc_data_type_t::u8(), sc_data_type_t::s8()))) {
            if (old_input.isa<tensorptr>()) {
                auto &producer = ins->producer_owner_;
                if (!producer->isa<tensor_view_op_t>()) return;
                auto &tv_inp = producer->get_inputs()[0];
                if (tv_inp->attrs_.get_or_else(
                            mixed_partition_hint::no_inplace, false))
                    return;
                if (producer->share_gt_with_op<tensor_view_op_t>(tv_inp)
                        || producer->share_gt_with_op<tunable_op_t>(tv_inp))
                    return;
            }
            op->attrs_.set<bool>(
                    mixed_partition_hint::inplace_optimized_op, true);
            auto pad_op = op->dyn_cast<padding_op_t>();
            auto new_input = builder::tensor_ptr(g2b_map_.get(out),
                    pad_op->get_padding_offsets_exprs(),
                    pad_op->get_inputs()[0]->details_.get_blocking_dims_expr(
                            graph),
                    true);
            if (old_input.isa<tensorptr>()) {
                auto parent_tsr = get_real_tensor(old_input);
                auto shape = parent_tsr->dims_;
                new_input = builder::tensor_ptr(new_input,
                        std::vector<expr>(
                                new_input.static_as<tensorptr>()->shape_.size(),
                                0),
                        shape, false);
                old_input = parent_tsr;
            }
            // Buffer replace
            replace_buffer(old_input, new_input);
        }
    }
}

std::vector<memory_optim::memory_alloc_trace_t>
mxp_buffer_allocator::get_real_mem_trace(
        const std::unordered_set<graph_tensor *> &keep_cut_set) const {
    std::vector<memory_optim::memory_alloc_trace_t> shrink_trace = mem_trace_;
    std::unordered_set<expr> ignore_buf_set;
    for (auto &g2b : g2b_map_.datamap_) {
        if (binded_mxp_->is_parti_cut(g2b.first)
                && keep_cut_set.find(g2b.first) == keep_cut_set.end()) {
            ignore_buf_set.insert(get_real_tensor(g2b.second));
        }
    }
    for (auto iter = shrink_trace.begin(); iter != shrink_trace.end();) {
        auto tsr = ((expr_base *)iter->buffer_id_)->node_ptr_from_this();
        if (ignore_buf_set.find(tsr) != ignore_buf_set.end()) {
            iter = shrink_trace.erase(iter);
        } else {
            iter++;
        }
    }
    std::transform(shrink_trace.begin(), shrink_trace.end(),
            shrink_trace.begin(),
            [&](const memory_optim::memory_alloc_trace_t &t) {
                if (t.size_ == 0) return t;
                auto tsr = ((expr_base *)t.buffer_id_)->node_ptr_from_this();
                auto shrink_info = get_shrinked_info(tsr);
                auto shape = get_dims_product(get_expr_to_dims(
                        shrink_info.empty() ? tsr.checked_as<tensor>()->dims_
                                            : get_slice_shape(shrink_info)));
                auto dtype_size = utils::get_sizeof_etype(
                        tsr.checked_as<tensor>()->elem_dtype_.type_code_);
                return memory_optim::memory_alloc_trace_t {
                        t.buffer_id_, shape * dtype_size};
            });
    return shrink_trace;
}

size_t get_buffer_usage(const context_ptr &ctx,
        const std::vector<memory_optim::memory_alloc_trace_t> &mem_trace,
        const memory_optim::inplace_info_map &inplace_map) {
    std::unordered_map<uintptr_t, std::size_t> out_schedule;
    std::unordered_map<uintptr_t, std::vector<uintptr_t>> out_inplace_selection;
    return schedule_memory_allocations(mem_trace, /*alignment*/ 64,
            ctx->flags_.buffer_schedule_ == attr_keys::BUF_SCHED_HOT,
            inplace_map, out_schedule, out_inplace_selection);
}

size_t mxp_buffer_allocator::get_real_buffer_usage() const {
    return get_buffer_usage(
            binded_mxp_->ctx_, get_real_mem_trace(), inplace_map_);
}

void mxp_buffer_allocator::replace_buffer(
        const expr &old_buffer, const expr &new_buffer) {
    // assert new buffer
    COMPILE_ASSERT(b2g_map_.find(new_buffer) == b2g_map_.end(),
            "Currently, it is only expected to replace with new buffer which "
            "never appear in mixed IR, but got "
                    << new_buffer)
    // get old buffer
    COMPILE_ASSERT(old_buffer.isa<tensor>(),
            "Replace target is expected to be Tensor node")
    if (tsr2anch_map_.find(old_buffer) != tsr2anch_map_.end()) {
        tsr2anch_map_.erase(old_buffer);
    }
    if (b2g_map_.find(old_buffer) != b2g_map_.end()) {
        auto old_gt = b2g_map_[old_buffer];
        b2g_map_.erase(old_buffer);
        b2g_map_[new_buffer] = old_gt;
    }
    // get real tsr
    auto old_tsr = get_real_tensor(old_buffer),
         new_tsr = get_real_tensor(new_buffer);
    // update trace
    bool new_tsr_already_in_trace = false;
    std::for_each(mem_trace_.begin(), mem_trace_.end(),
            [&old_tsr, &new_tsr, &new_tsr_already_in_trace](
                    memory_optim::memory_alloc_trace_t &t) {
                if (t.buffer_id_ == (uintptr_t)old_tsr.get()) {
                    t.buffer_id_ = (uintptr_t)new_tsr.get();
                } else if (t.buffer_id_ == (uintptr_t)new_tsr.get()) {
                    new_tsr_already_in_trace = true;
                }
            });
    if (new_tsr_already_in_trace) {
        int cnt = 0;
        // erase middle two times
        for (auto iter = mem_trace_.begin(); iter != mem_trace_.end();) {
            if (iter->buffer_id_ == (uintptr_t)new_tsr.get()) {
                cnt++;
                if (cnt == 2 || cnt == 3) {
                    iter = mem_trace_.erase(iter);
                    continue;
                }
            }
            iter++;
        }
        COMPILE_ASSERT(
                cnt == 4, "Unexpected buffer occurs time in trace: " << cnt)
    }
    // update inplace map if necessary
    for (auto iter = inplace_map_.begin(); iter != inplace_map_.end();) {
        auto buf1 = ((expr_base *)iter->first)->node_ptr_from_this();
        auto buf2 = get_inplaced_buffer(buf1);
        if (buf1.ptr_same(old_buffer) || buf2.ptr_same(old_buffer)) {
            iter = inplace_map_.erase(iter);
        } else {
            iter++;
        }
    }
    // replace g2b map
    for (auto &g2b : g2b_map_.datamap_) {
        auto &buf = g2b.second;
        auto tsr = get_real_tensor(buf);
        if (tsr.ptr_same(old_tsr.static_as<tensor>())) {
            set_base_tensor(buf, new_buffer);
        }
    }
    // TIR replace
    node_ptr_map buffer_map = {{old_buffer.impl, new_buffer.impl}};
    mxp_replacer_t(buffer_map).replace_func(binded_mxp_->func_);
}

std::tuple<std::vector<expr>, std::vector<expr>>
mxp_buffer_allocator::get_buffer(sc_op *op) const {
    std::vector<expr> inputs(op->get_inputs().size()),
            outputs(op->get_outputs().size());
    std::transform(op->get_inputs().begin(), op->get_inputs().end(),
            inputs.begin(), [&](const graph_tensor_ptr &gt) {
                COMPILE_ASSERT(
                        g2b_map_.haskey(gt), "please allocate buffer first")
                return g2b_map_.datamap_.find(gt.get())->second;
            });
    std::transform(op->get_outputs().begin(), op->get_outputs().end(),
            outputs.begin(), [&](const graph_tensor_ptr &gt) {
                COMPILE_ASSERT(
                        g2b_map_.haskey(gt), "please allocate buffer first")
                return g2b_map_.datamap_.find(gt.get())->second;
            });
    return std::make_tuple(inputs, outputs);
}

static cmp_res cmp_op_anchor(sc_op *op, fuse_anchor_map_ptr cur_anchor,
        fuse_anchor_map_ptr new_anchor) {
    cmp_res res = cmp_res::unknown;
    auto cmper = [&](const std::vector<graph_tensor_ptr> &gt_vec) {
        std::for_each(
                gt_vec.begin(), gt_vec.end(), [&](const graph_tensor_ptr &gt) {
                    if (utils::is_one_of(res, cmp_res::unknown, cmp_res::equal)
                            && cur_anchor->fsmap_.hasvalue(gt)
                            && new_anchor->fsmap_.hasvalue(gt)) {
                        res = cmp_slice_range(cur_anchor->fsmap_.get(gt),
                                new_anchor->fsmap_.get(gt));
                    }
                });
    };
    // compare input
    cmper(op->get_inputs());
    // if result is unknown, continue compare output
    if (res == cmp_res::unknown) { cmper(op->get_outputs()); }
    COMPILE_ASSERT(res != cmp_res::unknown, "Unknown comparision result")
    return res;
}

void mxp_buffer_allocator::update_input_buffer_info(sc_op *op) {
    auto commited_anchor_map = binded_mxp_->lookup_anchor_map(op);
    auto update_inp_tensor_info = [&](const graph_tensor_ptr &inp) {
        auto buf = g2b_map_.get(inp);
        if (b2g_map_.find(buf) == b2g_map_.end()) b2g_map_[buf] = inp;
        auto tsr = get_real_tensor(buf);
        bool is_borrowed = (commited_anchor_map->borrowed_fanchor_map_.find(inp)
                != commited_anchor_map->borrowed_fanchor_map_.end());
        // update b2g map if necessary
        if (is_borrowed) b2g_map_[buf] = inp;
        auto real_anchor_map = is_borrowed
                ? commited_anchor_map->borrowed_fanchor_map_[inp]
                : commited_anchor_map;
        if (tsr2anch_map_.find(tsr) != tsr2anch_map_.end()) {
            COMPILE_ASSERT(b2g_map_.find(buf) != b2g_map_.end(),
                    "base tensor should be visited")
            // current anchor map
            auto cur_anchor_map = tsr2anch_map_[tsr];
            // auto skip
            if (cur_anchor_map == real_anchor_map) return;
            // redirect to common parent anchor if new anchor is cousin
            // relationship of current anchor in avoid of use before define
            if (cur_anchor_map->is_cousin_for(real_anchor_map)) {
                // set common parent anchor no matter which anchor is larger
                tsr2anch_map_[tsr]
                        = real_anchor_map->get_root()->shared_from_this();
            } else {
                auto cur_slice = cur_anchor_map->fsmap_.get(b2g_map_[buf]);
                auto new_slice = real_anchor_map->fsmap_.get(inp);
                auto res = cmp_slice_range(cur_slice, new_slice);
                bool need_overwrite = false;
                if (res == cmp_res::l_less_r) {
                    need_overwrite = true;
                } else if (res == cmp_res::equal) {
                    // usually occurs in op is reduce or reduce collect op
                    need_overwrite
                            = real_anchor_map->is_parent_for(cur_anchor_map)
                            || real_anchor_map->is_sibling_for(cur_anchor_map);
                }
                // update latest anchor map
                if (need_overwrite) { tsr2anch_map_[tsr] = real_anchor_map; }
            }
        } else {
            tsr2anch_map_[tsr] = real_anchor_map;
        }
    };

    for (auto &inp : op->get_inputs()) {
        update_inp_tensor_info(inp);
    }
}

void mxp_buffer_allocator::update_output_buffer_info(sc_op *op) {
    auto commited_anchor_map = binded_mxp_->lookup_anchor_map(op);
    auto sub_of_commited_anchor_map
            = binded_mxp_->lookup_sub_anchor_map(commited_anchor_map);

    auto update_out_tensor_info = [&](const graph_tensor_ptr &out) {
        auto buf = g2b_map_.get(out);
        if (b2g_map_.find(buf) == b2g_map_.end()) b2g_map_[buf] = out;
        auto tsr = get_real_tensor(buf);
        if (tsr2anch_map_.find(tsr) != tsr2anch_map_.end()) {
            // only input anchor is expected for below logic
            if (!commited_anchor_map->is_input_anchor()) return;
            COMPILE_ASSERT(b2g_map_.find(buf) != b2g_map_.end(),
                    "base tensor should be visited")
            // auto skip
            if (tsr2anch_map_[tsr] == commited_anchor_map) return;
            auto cur_slice = tsr2anch_map_[tsr]->fsmap_.get(b2g_map_[buf]);
            auto new_slice = commited_anchor_map->fsmap_.get(out);
            auto res = cmp_slice_range(cur_slice, new_slice);
            if (res == cmp_res::l_less_r) {
                tsr2anch_map_[tsr] = commited_anchor_map;
            }
        } else {
            fuse_anchor_map_ptr min_anchor_map = nullptr;
            for (auto &sub_anchor : sub_of_commited_anchor_map) {
                if (!sub_anchor->fsmap_.hasvalue(out)) continue;
                if (!min_anchor_map)
                    min_anchor_map = sub_anchor;
                else {
                    auto min_slice = min_anchor_map->fsmap_.get(out);
                    auto cur_slice = sub_anchor->fsmap_.get(out);
                    if (cmp_slice_range(min_slice, cur_slice)
                            == cmp_res::l_larger_r) {
                        min_anchor_map = sub_anchor;
                    }
                }
            }
            tsr2anch_map_[tsr]
                    = min_anchor_map ? min_anchor_map : commited_anchor_map;
        }
    };

    for (auto &out : op->get_outputs()) {
        update_out_tensor_info(out);
    }
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
                        tsr2anch_map_.find(pad_tsr) != tsr2anch_map_.end(),
                        "Could not find padding tensor: "
                                << pad_tsr << " in tsr2anchor map")
                auto anchor = tsr2anch_map_[pad_tsr];
                range_list = anchor->fsmap_.get(b2g_map_[pair.second]);
                decl_body = anchor->get_parent_scope();
            }
            auto ret = padding->get_zero_out_stmt(pad_tsr, range_list);
            decl_body->seq_.insert(decl_body->seq_.begin(), ret);
        }
    }
}

void mxp_buffer_allocator::copy_concat_memory_attrs_tsr2buf() {
    for (auto &op : binded_mxp_->committed_ops_) {
        if (!op->isa<concat_op_t>()) { continue; }
        auto concat = op->stc_cast<concat_op_t>();
        for (auto &input_tsr : concat->get_inputs()) {
            if (input_tsr->attrs_.has_key(
                        concat_optim_attr_keys::graph_memory_offset)) {
                auto &offset = input_tsr->attrs_.get<std::vector<expr>>(
                        concat_optim_attr_keys::graph_memory_offset);

                auto &buf = binded_mxp_->buf_alloc_.g2b_map_.get(input_tsr);
                COMPILE_ASSERT(buf.isa<tensor>(),
                        "Buffer with memory_offset should be a tensor")
                buf->attr()[concat_optim_attr_keys::pass_memory_offset]
                        = offset;

                auto &final_tsr = input_tsr->attrs_.get<graph_tensor_ptr>(
                        concat_optim_attr_keys::graph_memory_offset_to);
                COMPILE_ASSERT(
                        binded_mxp_->buf_alloc_.g2b_map_.haskey(final_tsr),
                        "No buffer allocated for concat outputs")
                auto &out_buffer
                        = binded_mxp_->buf_alloc_.g2b_map_.get(final_tsr);
                buf->attr()[concat_optim_attr_keys::pass_memory_offset_to]
                        = out_buffer;
                SC_MODULE_INFO
                        << "Buffer: " << buf
                        << " has memory offset to buffer: " << out_buffer;
            }
        }
    }
}

inline bool is_elementwise_op(const sc_op *op) {
    return op->isa<unary_elementwise_op_t>()
            || op->isa<binary_elementwise_op_t>();
}

inline bool is_elementwise_producer(const graph_tensor *gt) {
    return is_elementwise_op(gt->producer_owner_);
}

// If last gt depends on all users of cur gt, return true
inline bool check_last_use_for_gt(const graph_tensor *cur_gt,
        const graph_tensor *last_gt, const mxp_buffer_allocator *alloc) {
    return std::all_of(cur_gt->uses_.begin(), cur_gt->uses_.end(),
            [&alloc, &last_gt](const std::pair<int, sc_op_weak_ptr_t> &user) {
                return alloc->get_binded_mxp()->dep_m_->lookup(
                               user.second.get(), last_gt->producer_owner_)
                        == 1;
            });
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
            && check_last_use_for_gt(cur_gt, ref_gt, alloc)) {
        inplace_set.insert(cur_gt);
        // reset step
        step = 0;
        // reset ref_gt
        ref_gt = cur_gt;
    }

    // if not mark visited
    if (visited_set.find(cur_buf) == visited_set.end()) {
        // recursively mark visited
        while (cur_buf.defined()) {
            visited_set.insert(cur_buf);
            cur_buf = alloc->get_inplaced_buffer(cur_buf);
        }
    }
    // get cur op
    auto elem_op = cur_gt->producer_owner_;
    // recursively collect inplace information
    for (auto &inp : elem_op->get_inputs()) {
        collect_inplace_info(inp.get(), ref_gt, inplace_set, visited_set,
                valid_set, alloc, step);
    }
}

expr mxp_buffer_allocator::get_inplaced_buffer(const expr &buf) const {
    auto iter = inplace_map_.find((uintptr_t)buf.get());
    if (iter != inplace_map_.end()) {
        COMPILE_ASSERT(iter->second.size() == 1,
                "Unexpected inplace info size during partition")
        return ((expr_base *)iter->second[0].first)->node_ptr_from_this();
    }
    return expr();
}

void mxp_buffer_allocator::query_inplace() {
    SC_MODULE_INFO << "Query buffer inplace hint...";
    // step 0: get outer loop
    auto outer_loops = binded_mxp_->get_outer_loops();
    if (outer_loops.empty()) return;
    auto batch_anchor = binded_mxp_->get_anchor_inside_loop(outer_loops.back());
    if (!batch_anchor) return;

    // step 1: search gt which defined inside outer loop
    std::vector<graph_tensor *> ref_gt_list;
    std::unordered_set<graph_tensor *> valid_set;
    for (auto &tsr2anch : tsr2anch_map_) {
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
        // step 4: validate inplace chain to ensure all of gt satisfy last use
        graph_tensor *next_gt = nullptr;
        for (auto iter = inplace_gt_list.rbegin();
                iter != inplace_gt_list.rend();) {
            if (!next_gt)
                next_gt = (*iter);
            else {
                if (check_last_use_for_gt(*iter, next_gt, this)) {
                    next_gt = (*iter);
                } else {
                    iter = std::vector<graph_tensor *>::reverse_iterator(
                            inplace_gt_list.erase((++iter).base()));
                    continue;
                }
            }
            ++iter;
        }
        // step 5: mark inplace attr for each buffer in list with the previous
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
            // set inplace hint
            set_buffer_inplace_hint(target_buf, inplace_buf);
            // in avoid of repeat try
            replaced_buffer.insert(inplace_buf);
        }
    }
}

void mxp_buffer_allocator::calibrate_info() {
    // collect cut buffer
    std::unordered_set<expr> cut_buffer_set;
    for (auto iter = inplace_map_.begin(); iter != inplace_map_.end();) {
        auto out_buf = ((expr_base *)iter->first)->node_ptr_from_this();
        if (!out_buf->attr().has_key(mixed_partition_hint::cut_buffer)) {
            ++iter;
            continue;
        }
        // temp buffer set
        std::unordered_set<expr> temp_set;
        // record whether tensorptr would be inplaced by cut buffer
        bool inplace_tptr = false;
        auto buf = out_buf;
        while (buf.defined()) {
            // if tensorptr found
            if (buf.isa<tensorptr>()) {
                inplace_tptr = true;
                break;
            }
            temp_set.insert(buf);
            buf = get_inplaced_buffer(buf);
        }
        if (!inplace_tptr) {
            // remove tensor shrink attr for those shared with cut buffer
            for (auto &tb : temp_set) {
                tb->attr().remove(tensor_shrinker_attrs::should_shrink);
            }
            cut_buffer_set.insert(temp_set.begin(), temp_set.end());
            ++iter;
        } else {
            // cut off inplace hint for the output buffer
            out_buf->attr().remove(attr_keys::tensor_inplace_hint);
            // remove inplace map
            iter = inplace_map_.erase(iter);
        }
    }

    // validate inplace hint for shrink info
    for (auto iter = inplace_map_.begin(); iter != inplace_map_.end();) {
        auto buf1 = ((expr_base *)iter->first)->node_ptr_from_this();
        auto buf2 = get_inplaced_buffer(buf1);
        // auto skip cut buffer
        if (cut_buffer_set.find(buf1) != cut_buffer_set.end()) {
            COMPILE_ASSERT(cut_buffer_set.find(buf2) != cut_buffer_set.end(),
                    "inplaced buffer should also be set cut buffer")
            ++iter;
        } else {
            auto shrink_info1 = get_shrinked_info(buf1);
            auto shrink_info2 = get_shrinked_info(buf2);
            // if shrink info is not equal, remove inplace hint
            if ((shrink_info1.empty() ^ shrink_info2.empty())
                    || (!shrink_info1.empty() && !shrink_info2.empty()
                            && cmp_slice_range({shrink_info1}, {shrink_info2})
                                    != cmp_res::equal)) {
                SC_MODULE_INFO << "removing tensor inplace hint: " << buf1
                               << " ==> " << buf2 << " for safety";
                // remove inplace hint to ensure correctness
                buf1->attr().remove(attr_keys::tensor_inplace_hint);
                // remove inplace map
                iter = inplace_map_.erase(iter);
            } else {
                ++iter;
            }
        }
    }
}

inline bool check_tsr_len_under_resigter_size(
        size_t tsr_len, uint16_t simd_len, uint16_t max_register_tol = 16) {
    return (tsr_len % simd_len == 0 && (tsr_len / simd_len) < max_register_tol);
}

bool mxp_buffer_allocator::validate_tsr2var() const {
    for (auto &tsr2def : tsr2anch_map_) {
        auto tsr = tsr2def.first;
        if (!tsr->attr().has_key(attr_keys::must_tensor2var)) continue;
        auto shrinked_info = get_shrinked_info(tsr);
        if (shrinked_info.empty()) continue;
        auto shape = get_slice_shape(shrinked_info);
        auto prod = get_dims_product(get_expr_to_dims(shape));
        auto tsr_simd_len = vectorize_step(binded_mxp_->ctx_,
                tsr.checked_as<tensor>()->elem_dtype_.type_code_);
        if (!check_tsr_len_under_resigter_size(prod, tsr_simd_len))
            return false;
    }
    return true;
}

int mxp_buffer_allocator::use_count(const expr &buffer) const {
    int cnt = 0;
    for (auto &g2b : g2b_map_.datamap_) {
        if (g2b.second.ptr_same(buffer)) cnt++;
    }
    return cnt;
}

fuse_anchor_map_ptr mxp_buffer_allocator::get_real_anchor_for_buffer(
        const expr &buffer) const {
    auto tsr = get_real_tensor(buffer);
    if (tsr2anch_map_.find(tsr) == tsr2anch_map_.end()) return nullptr;
    auto anch = tsr2anch_map_.find(tsr)->second;
    auto parent_loop = anch->get_parent_loop();
    // use outer anchor for cases that calculation is partially done. (e.g.
    // calculating part of K for matmul)
    if ((b2g_map_.find(tsr) == b2g_map_.end()
                || b2g_map_.find(tsr)
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
    if (!anchor) return {};
    COMPILE_ASSERT(b2g_map_.find(buffer) != b2g_map_.end(),
            "Could not find " << buffer << " in b2g map");
    slice_range ret;
    auto range_list = anchor->fsmap_.get(b2g_map_.find(buffer)->second);
    if (range_list.size() != 1) {
        if (range_list.empty()
                || !(anchor->isa<fuse_iter_anchor_map_t>()
                        || anchor->isa<fuse_grouped_anchor_map_t>()))
            return {};
        else
            ret = *std::max_element(range_list.begin(), range_list.end(),
                    [&](const slice_range &A, const slice_range &B) {
                        return cmp_slice_range({A}, {B}) == cmp_res::l_less_r;
                    });

    } else {
        ret = range_list[0];
    }

    return ret;
}

void mxp_buffer_allocator::declare_tensor() const {
    // define real tensor, and put it at the `def_pos` of ss
    auto declare_tensor_ = [&](const expr &tsr) {
        auto fanchor = get_real_anchor_for_buffer(tsr);
        if (!fanchor) return;
        // search insert position in scope of fusion anchor
        auto &ss = fanchor->get_parent_scope()->seq_;
        size_t def_pos = 0;
        for (; def_pos < ss.size(); def_pos++) {
            // skip var def node
            if (!ss[def_pos].cast<define>()
                            .filter([](const define &d) {
                                return d->var_.isa<var>();
                            })
                            .has_value())
                break;
        }
        ss.emplace(ss.begin() + def_pos,
                builder::make_var_tensor_def_unattached(tsr));
    };

    for (auto &tsr2def : tsr2anch_map_) {
        if (tsr2def.first->attr().has_key(mixed_partition_hint::cut_buffer))
            continue;
        declare_tensor_(tsr2def.first);
    }
}

void mxp_buffer_allocator::set_shrink_info() const {
    // set shrink info
    auto set_shrink_info_ = [&](const expr &buffer) {
        auto shrink_range = get_shrinked_info(buffer);
        if (shrink_range.empty()) return;
        buffer->attr()[tensor_shrinker_attrs::should_shrink]
                = tensor_shrinker_t::shrink_info_t {
                        /*base*/ get_slice_idx(shrink_range),
                        /*shape*/ get_slice_shape(shrink_range), stmts()};
    };
    for (auto &buf2shr : b2g_map_) {
        if (buf2shr.first->attr().has_key(mixed_partition_hint::cut_buffer))
            continue;
        set_shrink_info_(buf2shr.first);
    }
}

std::vector<memory_optim::memory_alloc_trace_t> merge_mem_trace(
        const std::vector<memory_optim::memory_alloc_trace_t> &mem_trace1,
        const std::vector<memory_optim::memory_alloc_trace_t> &mem_trace2,
        const std::unordered_map<expr, expr> &buffer_map) {
    auto ret = mem_trace1;
    for (auto &trace : mem_trace2) {
        auto tsr = ((expr_base *)trace.buffer_id_)->node_ptr_from_this();
        auto trace_itr = buffer_map.find(tsr);
        bool is_replaced = trace_itr != buffer_map.end();
        auto buf_id = is_replaced ? (uintptr_t)(trace_itr->second.get())
                                  : trace.buffer_id_;
        if (trace.size_ > 0) {
            auto last_use = std::find_if(ret.begin(), ret.end(),
                    [&buf_id](const memory_optim::memory_alloc_trace_t &tr) {
                        return (tr.buffer_id_ == buf_id) && (tr.size_ == 0);
                    });
            if (last_use != ret.end()) {
                ret.erase(last_use);
                continue;
            }
        }
        if (is_replaced) {
            ret.emplace_back(
                    memory_optim::memory_alloc_trace_t {buf_id, trace.size_});
        } else {
            ret.emplace_back(trace);
        }
    }
    return ret;
}

memory_optim::inplace_info_map merge_inplace_map(
        const memory_optim::inplace_info_map &inplace_map1,
        const memory_optim::inplace_info_map &inplace_map2,
        const std::unordered_map<expr, expr> &buffer_map) {
    auto ret = inplace_map1;
    for (auto &inplace_pair : inplace_map2) {
        COMPILE_ASSERT(inplace_pair.second.size() == 1,
                "Unexpected inplace info size during partition")
        memory_optim::inplace_info info = inplace_pair.second[0];
        auto tsr = ((expr_base *)info.first)->node_ptr_from_this();
        auto iter = buffer_map.find(tsr);
        ret[inplace_pair.first]
                = std::vector<std::pair<uintptr_t, inplace_kind>> {
                        {iter != buffer_map.end()
                                        ? (uintptr_t)iter->second.get()
                                        : info.first,
                                info.second}};
    }
    return ret;
}

mxp_mem_info merge_real_mem_info(const mxp_buffer_allocator &alloc1,
        const mxp_buffer_allocator &alloc2) {
    // get buffer replace map
    std::unordered_map<expr, expr> buffer_map;
    std::unordered_set<graph_tensor *> keep_cut_set;
    for (auto &g2b : alloc2.g2b_map_.datamap_) {
        auto gt = g2b.first;
        if (alloc1.g2b_map_.haskey(gt)) {
            buffer_map[g2b.second] = alloc1.g2b_map_.datamap_.find(gt)->second;
            if (alloc1.get_binded_mxp()->is_parti_out(gt)
                    && alloc2.get_binded_mxp()->is_parti_inp(gt)) {
                keep_cut_set.insert(gt);
            }
        }
    }
    return std::make_pair(
            merge_mem_trace(alloc1.get_real_mem_trace(keep_cut_set),
                    alloc2.get_real_mem_trace(keep_cut_set), buffer_map),
            merge_inplace_map(
                    alloc1.inplace_map_, alloc2.inplace_map_, buffer_map));
}

void mxp_buffer_allocator::merge(mxp_buffer_allocator &other,
        std::unordered_map<expr, expr> &buffer_map,
        const std::pair<fuse_anchor_map_ptr, fuse_anchor_map_ptr>
                &common_buffer_anchor_pair) {
    buffer_map.clear();
    auto common_buffer_anchor = common_buffer_anchor_pair.first,
         common_other_buffer_anchor = common_buffer_anchor_pair.second;
    for (auto &other_g2b : other.g2b_map_.datamap_) {
        auto other_gt = other_g2b.first;
        // if other tensor has conflict in current tensor, redirect it to common
        // buffer anchor
        if (g2b_map_.haskey(other_gt)) {
            auto existed_buf = g2b_map_.get(other_gt);
            buffer_map[other_g2b.second] = existed_buf;
            if ((binded_mxp_->is_parti_inp(other_gt)
                        && other.binded_mxp_->is_parti_inp(other_gt))
                    || (binded_mxp_->is_parti_out(other_gt)
                            && other.binded_mxp_->is_parti_out(other_gt)))
                continue;
            COMPILE_ASSERT(common_buffer_anchor,
                    "Conflict buffer: "
                            << existed_buf
                            << " is detected but no common buffer anchor "
                               "is found for redirection")
            tsr2anch_map_[get_real_tensor(existed_buf)] = common_buffer_anchor;
            // the existed buffer will become intermediate buffer, which may be
            // shrinked
            if (b2g_map_.find(existed_buf) == b2g_map_.end()) {
                b2g_map_[existed_buf] = other_gt->shared_from_this();
            }
        } else {
            auto buffer = other.g2b_map_.get(other_gt);
            g2b_map_.get(other_gt) = buffer;
            if (other.b2g_map_.find(buffer) != other.b2g_map_.end()) {
                b2g_map_[buffer] = other.b2g_map_[buffer];
            }
            if (other.tsr2anch_map_.find(get_real_tensor(buffer))
                    != other.tsr2anch_map_.end()) {
                auto other_anchor
                        = other.tsr2anch_map_[get_real_tensor(buffer)];
                tsr2anch_map_[get_real_tensor(buffer)]
                        = (other_anchor == common_other_buffer_anchor)
                        ? common_buffer_anchor
                        : other_anchor;
            }
        }
    }
    // merge mem trace
    mem_trace_ = merge_mem_trace(mem_trace_, other.mem_trace_, buffer_map);
    // merge inplace map
    inplace_map_
            = merge_inplace_map(inplace_map_, other.inplace_map_, buffer_map);
}

void mxp_buffer_allocator::clear() {
    binded_mxp_ = nullptr;
    g2b_map_.clear();
    tsr2anch_map_.clear();
    b2g_map_.clear();
    mem_trace_.clear();
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
    COMPILE_ASSERT(ir_tsrs.size() == gtsrs.size(),
            "IR tensor-to-graph tensor mapping is not expected")
    // extract input anchor
    for (auto &anchor_map : fmgr->unpack_dst_anchor()) {
        fslice_map fsmap;
        for (size_t i = 0; i < ir_tsrs.size(); i++) {
            if (anchor_map.second.find(ir_tsrs[i]) != anchor_map.second.end()) {
                fsmap.get(gtsrs[i])
                        = anchor_map.second.find(ir_tsrs[i])->second;
            }
        }
        if (!fsmap.empty()) {
            parti->append_fusion_anchor(std::make_shared<fuse_anchor_map_t>(
                    anchor_map.first, fsmap, parent_fanchor, true));
        }
    }

    // extract output anchor
    for (auto &anchor_map : fmgr->unpack_src_anchor()) {
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
        }
    }

    // extract output iterated anchor
    if (!fmgr->iter_anchor_list_.empty()) {
        for (auto &iter_anchor : fmgr->iter_anchor_list_) {
            fslice_map fsmap;
            for (size_t i = 0; i < ir_tsrs.size(); i++) {
                if (iter_anchor.tsr_.ptr_same(ir_tsrs[i])) {
                    fsmap.get(gtsrs[i]) = iter_anchor.slice_list_;
                }
            }
            if (!fsmap.empty()) {
                parti->append_fusion_anchor(
                        std::make_shared<fuse_iter_anchor_map_t>(
                                iter_anchor.iter_, iter_anchor.anchor_position_,
                                fsmap, iter_anchor.dispatch_helper_));
            }
        }
    }

    // extract output grouped anchor
    if (!fmgr->grouped_anchor_map_.empty()) {
        for (auto &grouped_anchor_map_ : fmgr->grouped_anchor_map_) {
            auto grouped_anchor = grouped_anchor_map_.second;
            // extract anchor position
            auto anchor_ss = std::move(grouped_anchor.anchor_position_);
            // extract anchor fsmap
            auto anchor_es_map = grouped_anchor.expr_slice_map_;
            fslice_map fsmap;
            for (size_t i = 0; i < ir_tsrs.size(); i++) {
                auto iter = anchor_es_map.find(ir_tsrs[i]);
                if (iter != anchor_es_map.end()) {
                    fsmap.get(gtsrs[i]) = std::move(iter->second);
                }
            }
            if (!fsmap.empty()) {
                parti->append_fusion_anchor(
                        std::make_shared<fuse_grouped_anchor_map_t>(
                                anchor_ss, fsmap));
            }
        }
    }
}

void search_op_anchor_in_parti(sc_op *op, mixed_parti_t *parti) {
    if (parti->merged_to) {
        search_op_anchor_in_parti(op, parti->get_root());
        return;
    }

    for (auto &fanchor : parti->fanchors_) {
        // if op is marked as break_pre_fuse, only pre-op fusion is accepted
        if (!parti->empty()
                && op->attrs_.get_or_else(
                        mixed_partition_hint::pre_fuse_begin_op, false)) {
            if (fanchor->fsmap_.hasvalue(op->get_outputs()[0])) {
                parti->set_anchor_for_op(op, fanchor);
            }
            // in avoid of preop fusion select unexpected anchor
            continue;
        }
        infer_status_map_t stat_map(parti->ctx_, false);
        std::unordered_set<graph_tensor_ptr> known_gt;
        // deal with fusible op which use output loop mode to create partition
        if (parti->empty()
                && std::any_of(op->get_outputs().begin(),
                        op->get_outputs().end(),
                        [&fanchor](const graph_tensor_ptr &gt) {
                            return fanchor->fsmap_.haskey(gt);
                        })) {
            COMPILE_ASSERT(op->isa<fusible_op_t>(),
                    "Only fusible op is expected, but got " << op->op_name_)
            // use pre-infer slice range method
            op->dyn_cast<fusible_op_t>()->pre_slice_ranges(
                    fanchor->fsmap_, stat_map);
        } else { /* most common cases */
            // check input slice firstly
            if (!fanchor->check_input_for_op(op, known_gt)) continue;
            // TODO(XXX): merge
            if (auto fusible = op->dyn_cast<fusible_op_t>()) {
                fusible->infer_slice_ranges(fanchor->fsmap_, stat_map);
            } else if (auto tunable = op->dyn_cast<tunable_op_t>()) {
                tunable->infer_slice_ranges(fanchor->fsmap_, stat_map);
            } else {
                COMPILE_ASSERT(0, "Unexpected op type found: " << op->op_name_)
            }
        }
        bool success_flag = true;
        // check status map
        if (stat_map.is_ok()) {
            /** doule check list
             * 1. validate input slice
             * 2. validate output slice
             * 3. check cost model
             * */
            success_flag = fanchor->validate_input_for_op(op, known_gt)
                    && fanchor->validate_output_for_op(op)
                    && parti->cost_->make_decision_for_op(op, fanchor);
        }
        // TODO(yunfei): remove this check when old fmgr is totally deprecated
        else if (stat_map.is_fail()) {
            auto &fail_list
                    = stat_map.get_ops_by_status(infer_status_code::FAIL);
            success_flag = (fail_list.find(op->shared_from_this())
                    == fail_list.end());
        } else {
            success_flag = false;
        }
        if (success_flag) {
            // set op anchor, and auto select smallest one
            parti->set_anchor_for_op(op, fanchor);
        } else {
            // forbid op in current anchor
            fanchor->forbid_op(op, known_gt);
        }
    }
}

static std::string print_loops_range(const std::vector<for_loop> &loops) {
    std::stringstream os;
    int cnt = 0;
    for (auto &l : loops) {
        if (cnt != 0) { os << "X"; }
        if (l->iter_begin_.isa<constant>() && l->iter_end_.isa<constant>()) {
            os << (get_expr_as_int(l->iter_end_)
                    - get_expr_as_int(l->iter_begin_));
        } else {
            os << "var";
        }
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

static stmts find_last_parallel_for(const stmts &scope, int64_t &out_index) {
    auto &outer_body = scope.static_as<stmts>()->seq_;
    int64_t prefetch_insertion_point = -1;
    if (!outer_body.empty()) {
        for (int64_t i = outer_body.size() - 1; i >= 0; i--) {
            auto &s = outer_body[i];
            if (s.isa<stmts>()) {
                auto ret = find_last_parallel_for(
                        s.static_as<stmts>(), out_index);
                if (ret.defined()) { return ret; }
            }
            if (s.cast<for_loop>()
                            .filter([](const for_loop &v) {
                                return v->kind_ == for_type::PARALLEL
                                        && (!v->attr_
                                                || !v->attr_->get_or_else(
                                                        "dont_prefetch",
                                                        false));
                            })
                            .has_value()) {
                out_index = i;
                return scope;
            }
        }
    }
    return stmts();
}

/**
 * Check two partition is forked, like
 *            input
 *           /     \
 *      parti A   parti B
 *         |        |
 *       output    output
 * Note that: this is different relationship from no dependency
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
    A = A->get_root();
    B = B->get_root();
    // skip conv workload until all conv ops implement axis binding infer
    if (A->contain_convolution() || B->contain_convolution()) {
        // limit to at most one outer loop (batch-dimension)
        return std::min(1, check_loop_size);
    }
    // auto skip when A and B are forked
    if (check_parti_forked(A, B)) return check_loop_size;
    return A->ax_binder_.align_with(B->ax_binder_, check_loop_size);
}

static void merge_parti_impl(mixed_parti_t *pa_to_merge,
        mixed_parti_t *parti_be_merged, size_t merged_loop_size,
        const sc_op_ptr &joint_op = nullptr) {
    pa_to_merge = pa_to_merge->get_root(),
    parti_be_merged = parti_be_merged->get_root();
    auto outer_loops_to_merge = pa_to_merge->get_outer_loops(),
         outer_loops_be_merged = parti_be_merged->get_outer_loops();
    auto outer_loops_merged_target = outer_loops_to_merge[merged_loop_size - 1];
    auto outer_loops_merged_append
            = outer_loops_be_merged[merged_loop_size - 1];
    auto max_to_merge_anchor_map
            = pa_to_merge->get_anchor_inside_loop(outer_loops_merged_target);
    auto max_be_merged_anchor_map = parti_be_merged->get_anchor_inside_loop(
            outer_loops_merged_append);
    /* * * * * * * * * * * * * * * * *
     * Step 1: Merge func_
     * * * * * * * * * * * * * * * * */
    COMPILE_ASSERT(
            max_to_merge_anchor_map, "max-to-merge fusion anchor not found")
    max_to_merge_anchor_map->fuse_anchor_map_t::commit_stmt(
            outer_loops_merged_append->body_);

    // var and tensor replace map
    node_ptr_map node_remap;
    std::unordered_map<expr, expr> buffer_map;

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

    // append inner loop anchor
    for (auto &be_merged_anchor_map : parti_be_merged->fanchors_) {
        // skip outer anchor
        if (std::any_of(outer_loops_be_merged.begin(),
                    outer_loops_be_merged.begin() + merged_loop_size,
                    [&be_merged_anchor_map, &parti_be_merged](
                            const for_loop &loop) {
                        return parti_be_merged->get_anchor_inside_loop(loop,
                                       be_merged_anchor_map->is_input_anchor())
                                == be_merged_anchor_map;
                    })) {
            continue;
        }
        be_merged_anchor_map->attach_parent_anchor(
                max_to_merge_anchor_map, max_be_merged_anchor_map);
        pa_to_merge->append_fusion_anchor(be_merged_anchor_map);
    }

    // merge outer loop anchor
    for (size_t i = 0; i < merged_loop_size; i++) {
        node_remap[outer_loops_be_merged[i]->var_.impl]
                = outer_loops_to_merge[i]->var_.impl;
        node_remap[outer_loops_be_merged[i].impl]
                = outer_loops_to_merge[i].impl;
        auto be_merged_anchor_map = parti_be_merged->get_anchor_inside_loop(
                     outer_loops_be_merged[i]),
             to_merge_anchor_map
                = pa_to_merge->get_anchor_inside_loop(outer_loops_to_merge[i]);
        if (be_merged_anchor_map && to_merge_anchor_map) {
            to_merge_anchor_map->merge(be_merged_anchor_map);
            // reset op anchor map if neccesary
            if (i == merged_loop_size - 1) {
                for (auto &op_anchor_pair : parti_be_merged->op_anchor_map_) {
                    if (op_anchor_pair.second == be_merged_anchor_map) {
                        op_anchor_pair.second = to_merge_anchor_map;
                    }
                }
            }
        }
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
    for (auto &buf_pair : buffer_map) {
        node_remap.insert(
                std::make_pair(buf_pair.first.impl, buf_pair.second.impl));
    }
    // create mxp inplace replacer
    mxp_replacer_t mxp_reper(node_remap);
    // 1. func->body
    mxp_reper.replace_func(pa_to_merge->func_);
    // 2. fanchor->fsmap->slice_range
    mxp_reper.replace_anchor(pa_to_merge->fanchors_);

    /* * * * * * * * * * * * * * * * *
     * Step 5: Merge op_anchor_map_
     * * * * * * * * * * * * * * * * */
    for (auto &op_anchor_pair : parti_be_merged->op_anchor_map_) {
        // override existed ones
        pa_to_merge->op_anchor_map_[op_anchor_pair.first]
                = op_anchor_pair.second;
    }

    // erase joint op in op_anchor_map
    pa_to_merge->op_anchor_map_.erase(joint_op.get());

    /* * * * * * * * * * * * * * * * *
     * Step 6: Merge op_
     * * * * * * * * * * * * * * * * */
    // call base merge
    // move 'ops' from src to target and set 'merged_to' of src to be target
    pa_to_merge->fusion_partition_t::merge(
            static_cast<fusion_partition_t *>(parti_be_merged)
                    ->shared_from_this());

    // Merge commited ops
    pa_to_merge->committed_ops_.insert(pa_to_merge->committed_ops_.end(),
            parti_be_merged->committed_ops_.begin(),
            parti_be_merged->committed_ops_.end());

    // clear merged parti
    parti_be_merged->clear();
}

static size_t get_great_common_loop_size(const std::vector<for_loop> &loop_A,
        const std::vector<for_loop> &loop_B) {
    // great common size
    auto gcs = std::min(loop_A.size(), loop_B.size());

    size_t merged_loop_size = 0;
    for (; merged_loop_size < gcs; merged_loop_size++) {
        auto &to_merge = loop_A[merged_loop_size];
        auto &be_merged = loop_B[merged_loop_size];
        if (!(to_merge->iter_begin_.isa<constant_c>()
                    && to_merge->step_.isa<constant_c>()
                    && be_merged->iter_begin_.isa<constant_c>()
                    && be_merged->step_.isa<constant_c>())) {
            break;
        }
        if (!to_merge->iter_end_.isa<constant_c>()
                && slice_expr_equals(
                        to_merge->iter_end_, be_merged->iter_end_)) {
            continue; // for dynamic
        } else if (!(to_merge->iter_end_.isa<constant_c>()
                           && be_merged->iter_end_.isa<constant_c>())) {
            break;
        }
        auto A_begin = get_expr_as_int(to_merge->iter_begin_),
             A_end = get_expr_as_int(to_merge->iter_end_),
             A_step = get_expr_as_int(to_merge->step_),
             B_begin = get_expr_as_int(be_merged->iter_begin_),
             B_end = get_expr_as_int(be_merged->iter_end_),
             B_step = get_expr_as_int(be_merged->step_);
        auto A_num_threads = to_merge->num_threads_;
        auto B_num_threads = be_merged->num_threads_;
        if (A_begin != B_begin || A_end != B_end || A_step != B_step
                || A_num_threads != B_num_threads)
            break;
    }
    return merged_loop_size;
}

/**
 * find nested parallel for. E.g.
 * pfor(){
 *  tensor a; // optional
 *  pfor(){
 *  }
 * }
 * */
class nested_pfor_finder_t : public ir_viewer_t {
public:
    using ir_viewer_t::dispatch;
    using ir_viewer_t::view;
    // return whether nested pfor exist
    bool operator()(for_loop_c v) {
        ir_viewer_t::dispatch(std::move(v));
        return pfor_cnt_ > 1;
    }
    expr_c dispatch(expr_c v) override { return v; }

    void view(for_loop_c f) override {
        // check pfor
        if (f->kind_ == for_type::PARALLEL && f->num_threads_ > 0) {
            pfor_cnt_++;
        }
        // to be faster
        if (pfor_cnt_ > 1) return;
        ir_viewer_t::view(f);
    }

private:
    int pfor_cnt_ = 0;
};

static bool try_merge_mixed_parti_parallel_inners(
        mixed_parti_t *pa_to_merge, mixed_parti_t *parti_be_merged) {
    pa_to_merge = pa_to_merge->get_root(),
    parti_be_merged = parti_be_merged->get_root();

    auto outer_loops_to_merge = pa_to_merge->get_outer_loops(),
         outer_loops_be_merged = parti_be_merged->get_outer_loops();

    if (outer_loops_to_merge.empty() || outer_loops_be_merged.empty())
        return false;

    auto merged_loop_size = get_great_common_loop_size(
            outer_loops_to_merge, outer_loops_be_merged);
    if (!merged_loop_size) return false; // no outer loop can be merged

    // validate axis binding
    merged_loop_size = check_parti_loop_axis_binding(
            parti_be_merged, pa_to_merge, merged_loop_size);
    SC_MODULE_INFO << "After axis binding, num loops to merge: "
                   << merged_loop_size;
    if (!merged_loop_size) return false; // no outer loop can be merged

    // change the loop var name of the first for.
    // first for: for m_s, for n_s. second for: for m_s_o, m_s_i, n_s.
    // merge directly: for m_s, for n_s, for n_s.
    // add suffix and merge: for m_s_0, for n_s_0, for n_s.
    for (size_t i = 0; i < merged_loop_size; ++i) {
        std::string &varname
                = outer_loops_to_merge[i]->var_.static_as<var>()->name_;
        varname += "_0";
    }

    SC_MODULE_INFO << "parallel merging two partition:";
    SC_MODULE_INFO << pa_to_merge->func_;
    SC_MODULE_INFO << parti_be_merged->func_;

    if (check_parti_dep(pa_to_merge, parti_be_merged) == parti_dep::no_dep) {
        auto &to_merge_body = outer_loops_to_merge[merged_loop_size - 1]->body_;
        /**
         * The thread-shared buffer of the previous pfor and the thread-shared
         * buffer of the next pfor may share the same memory location after
         * buffer scheduling and hoist. After removal of barrier, some threads
         * may work on the previous pfor and others may work on the next pfor,
         * and they share the same memory localtion. This will cause race
         * condition. To avoid that, we need to check that:
         *  1. there are no tensors defined in the immediate body of the merged
         * pfor.
         *  2. the buffers inside of the child pfor of the merged pfor must be
         * the most inner loop (i.e. merged pfor must be second most inner
         * loop). This is to avoid hoisting of the tensors. */
        bool barrier_can_remove = true;
        if (to_merge_body.isa<stmts>()) {
            for (auto &s : to_merge_body.static_as<stmts>()->seq_) {
                if (s.isa<define>()) {
                    // Case 1: if tensor node is defined
                    if (s.static_as<define>()->var_.isa<tensor>()) {
                        barrier_can_remove = false;
                        break;
                    }
                } else if (s.isa<for_loop>()) {
                    // Case 2: nested pfor and potential hoist buffer
                    if (nested_pfor_finder_t()(s.static_as<for_loop>())) {
                        barrier_can_remove = false;
                        break;
                    }
                }
            }
        }
        if (barrier_can_remove) {
            auto last_for = get_last_loop_in_body(to_merge_body);
            if (last_for.defined()) {
                last_for->attr()[stmt_attr_key::no_post_barrier] = true;
            }
        }
    }

    // try to add prefetch code
    if (pa_to_merge->ctx_->flags_.prefetch_) {
        auto op_first = pa_to_merge->committed_ops_[0];
        auto op_second = parti_be_merged->committed_ops_[0];
        if (auto second_prefetch
                = op_second->dyn_cast<op_traits::may_prefetch_t>()) {
            int64_t prefetch_insertion_point = -1;
            auto last_for_parent = find_last_parallel_for(
                    outer_loops_to_merge[merged_loop_size - 1]
                            ->body_.static_as<stmts>(),
                    prefetch_insertion_point);

            std::vector<tensor_slice> in_slice;
            in_slice.reserve(op_second->get_inputs().size());
            for (auto &inp : op_second->get_inputs()) {
                auto second_in = parti_be_merged->buf_alloc_.g2b_map_.get(inp);
                in_slice.emplace_back(second_in);
            }
            auto prefetch_idx = second_prefetch->query_prefetch(
                    pa_to_merge->ctx_, false, in_slice);
            for (auto itr = prefetch_idx.begin(); itr != prefetch_idx.end();) {
                auto input = op_second->get_inputs()[*itr]->producer_owner_;
                if (pa_to_merge->contains(input)) {
                    itr = prefetch_idx.erase(itr);
                } else {
                    ++itr;
                }
            }
            if (prefetch_insertion_point != -1 && !prefetch_idx.empty()) {
                std::vector<stmt> out_seq;
                second_prefetch->generate_prefetcher_and_set_idle(
                        pa_to_merge->ctx_, false, in_slice, prefetch_idx,
                        out_seq);
                auto &outer_body = last_for_parent->seq_;
                outer_body.insert(outer_body.begin() + prefetch_insertion_point,
                        out_seq.begin(), out_seq.end());
            }
        }
    }

    auto outer_loops_merged_target = outer_loops_to_merge[merged_loop_size - 1];
    auto max_to_merge_anchor_map
            = pa_to_merge->get_anchor_inside_loop(outer_loops_merged_target);
    if (!max_to_merge_anchor_map) {
        auto s = builder::make_stmts_unattached({}).checked_as<stmts>();
        add_parent_node(s, outer_loops_merged_target->body_);
        outer_loops_merged_target->body_.checked_as<stmts>()->seq_.emplace_back(
                s);
        // dummy fsmap, the tensor belongs to this scope will not be shrinked
        fslice_map fsmap;
        max_to_merge_anchor_map = std::make_shared<fuse_anchor_map_t>(s, fsmap);
        pa_to_merge->append_fusion_anchor(max_to_merge_anchor_map);
    }

    pa_to_merge->func_->name_
            += "_parallel_merge_" + parti_be_merged->func_->name_;

    merge_parti_impl(pa_to_merge, parti_be_merged, merged_loop_size);

    SC_MODULE_INFO << "parallel merging result:";
    SC_MODULE_INFO << pa_to_merge->func_;

    return true;
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
    SC_MODULE_INFO << "Start try_merge_mixed_parti_parallel: "
                   << "Target: " << target_parti->func_->name_
                   << ", Append: " << append_parti->func_->name_;

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

    // check axis binding on the outmost axis
    if (!check_parti_loop_axis_binding(append_parti, target_parti, 1)) {
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
        // start and step must be the same
        if (!(target_begin == append_begin && target_step == append_step)) {
            return false;
        }
        if (target_end
                != append_end) { // if the end is not same, then we try to split
            // the first on num_threads_ to merge, but we
            // require num_iters == num_threads
            if (append_parti->contain_convolution()
                    || target_parti->contain_convolution()) {
                // currently do not support split num_threads_ on conv
                return false;
            }

            if (target_begin != 0 || target_step != 1 || append_begin != 0
                    || append_step != 1) {
                // Only support begin is 0 and step is 1
                return false;
            }
            if (target_end == outermost_loop_target->num_threads_
                    && append_end == outermost_loop_append->num_threads_) {
                // For the two fors, same start, same step, different end =>
                // different num_iters.
                // For each for, num_iters == num_threads. So We split the for
                // on num_threads.
                // can not split the outermost loop in imbalanced cases
                if (!outermost_loop_target->attr().get_or_else(
                            stmt_attr_key::parallel_loop_balanced, true)
                        || !outermost_loop_append->attr().get_or_else(
                                stmt_attr_key::parallel_loop_balanced, true)) {
                    SC_MODULE_INFO << "The outermost loop is imbalanced, can "
                                      "not be split";
                    return false;
                }
                if (outermost_loop_target->num_threads_
                                > outermost_loop_append->num_threads_
                        && outermost_loop_target->num_threads_
                                        % outermost_loop_append->num_threads_
                                == 0) {
                    if (outermost_loop_append->num_threads_ == 1) {
                        // in this case, after split, outermost_loop_target
                        // num_threads will be 1
                        return false;
                    }
                    int64_t num_groups = outermost_loop_target->num_threads_
                            / outermost_loop_append->num_threads_;
                    target_parti->try_split_outermost_loop_on_num_threads(
                            num_groups);
                    return try_merge_mixed_parti_parallel_inners(
                            target_parti, append_parti);
                } else if (outermost_loop_append->num_threads_
                                > outermost_loop_target->num_threads_
                        && outermost_loop_append->num_threads_
                                        % outermost_loop_target->num_threads_
                                == 0) {
                    if (outermost_loop_target->num_threads_ == 1) {
                        // in this case, after split, outermost_loop_append
                        // num_threads will be 1
                        return false;
                    }
                    int64_t num_groups = outermost_loop_append->num_threads_
                            / outermost_loop_target->num_threads_;
                    append_parti->try_split_outermost_loop_on_num_threads(
                            num_groups);
                    return try_merge_mixed_parti_parallel_inners(
                            target_parti, append_parti);
                } else {
                    return false;
                }
            } else { // do not support num_iters != num_threads case for now
                return false;
            }
        } else { // the two fors have same (start, end, step)
            // if num_threads are the same, merge them directly
            if (outermost_loop_target->num_threads_
                    == outermost_loop_append->num_threads_) {
                return try_merge_mixed_parti_parallel_inners(
                        target_parti, append_parti);
            } else {
                return false;
            }
        }
    } else { // if (start, end, step) is not constant
        return false;
    }
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
    // skip parallel merge
    if (outer_loops_A[0]->num_threads_ > 0
            || outer_loops_B[0]->num_threads_ > 0)
        return false;
    // check cost model
    if (!A->cost_->make_decision_for_parti(
                B, 1, parti_merge_kind::horizontal)) {
        return false;
    }

    SC_MODULE_INFO << "horizontally merging two partition:";
    SC_MODULE_INFO << A->func_;
    SC_MODULE_INFO << B->func_;

    /* * * * * * * * * * * * * * * * *
     * Step 0: Fuse func_
     * * * * * * * * * * * * * * * * */
    node_ptr_map node_remap;
    schedule_loop_body(A->func_->body_, &node_remap);
    schedule_loop_body(B->func_->body_, &node_remap);

    /* * * * * * * * * * * * * * * * *
     * Step 1: Merge func_
     * * * * * * * * * * * * * * * * */
    auto new_body = make_stmt<stmts_node_t>(
            std::vector<stmt> {A->func_->body_, B->func_->body_});

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
    for (auto &buf_pair : buffer_map) {
        node_remap.insert(
                std::make_pair(buf_pair.first.impl, buf_pair.second.impl));
    }
    mxp_replacer_t expr_reper(node_remap);
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
        loops[0]->iter_end_ = cf(ac(loops[0]->iter_end_)).remove_const();
    }

    add_parent_node(loops[0], stmt());

    A->func_->name_ += "_horizontal_merge_" + B->func_->name_;

    SC_MODULE_INFO << "horizontally merging result:";
    SC_MODULE_INFO << A->func_;

    // clear merged parti
    B->clear();

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

static void try_align_parti_outer_loops(mixed_parti_t *A, mixed_parti_t *B) {
    auto outer_loops_A = A->get_outer_loops(),
         outer_loops_B = B->get_outer_loops();
    if (outer_loops_A.empty() || outer_loops_B.empty()) return;
    if (outer_loops_A.size() == outer_loops_B.size()) return;

    auto outermost_loop_A_range = get_loops_range(outer_loops_A[0]),
         outermost_loop_B_range = get_loops_range(outer_loops_B[0]);
    if (outermost_loop_A_range == 0 || outermost_loop_B_range == 0
            || outermost_loop_A_range == outermost_loop_B_range) {
        return;
    } else if (outermost_loop_A_range <= outermost_loop_B_range) {
        B->try_split_outermost_loop(outermost_loop_A_range);
    } else {
        A->try_split_outermost_loop(outermost_loop_B_range);
    }
}

// check brgemm pre-op fusion
static bool try_merge_brgemm_and_preop_parti(mixed_parti_t *A, mixed_parti_t *B,
        const sc_op_ptr &joint_op = nullptr) {
    A = A->get_root(), B = B->get_root();
    if (A == B) return false;
    if (!A->func_.get() || !B->func_.get()) return false;
    if (!joint_op && !check_parti_connectionship(A, B)) return false;
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
    // set pre-op fusion attr
    preop_parti->committed_ops_.front()->attrs_.set(
            mixed_partition_hint::pre_fuse_begin_op, true);
    for (auto &op_in_preop_parti : preop_parti->committed_ops_) {
        brgemm_parti->add(op_in_preop_parti);
    }
    // remove pre-op fusion attr
    preop_parti->committed_ops_.front()->attrs_.remove(
            mixed_partition_hint::pre_fuse_begin_op);

    // erase joint op in op_anchor_map
    brgemm_parti->op_anchor_map_.erase(joint_op.get());

    /* * * * * * * * * * * * * * * * *
     * Step 3: Merge op_
     * * * * * * * * * * * * * * * * */
    // call base merge
    brgemm_parti->fusion_partition_t::merge(
            static_cast<fusion_partition_t *>(preop_parti)->shared_from_this());

    SC_MODULE_INFO << "pre-op merging result:";
    SC_MODULE_INFO << brgemm_parti->func_;

    // clear merged parti
    preop_parti->clear();

    return true;
}

static bool try_merge_mixed_parti_vertically(mixed_parti_t *A, mixed_parti_t *B,
        const sc_op_ptr &joint_op, bool keep_outerloop_size = false) {
    A = A->get_root(), B = B->get_root();
    if (A == B) return false;
    if (!A->func_.get() || !B->func_.get()) return false;
    // in avoid conflict with parallel merge
    if (A->contain_nested_parallel_for() || B->contain_nested_parallel_for())
        return false;
    if (!joint_op && !check_parti_connectionship(A, B)) return false;
    if (!joint_op && check_parti_ring_risk(A, B)) return false;
    auto dep_flag = check_parti_dep(A, B);
    // if two partition inter-depends each other, could not merge them
    if (dep_flag == parti_dep::inter_dep) return false;
    // if both two partitions have input anchor, could not merge them
    if (A->contain_input_anchor() && B->contain_input_anchor()) return false;
    mixed_parti_t *pa_to_merge = nullptr, *parti_be_merged = nullptr;

    pa_to_merge = (dep_flag == parti_dep::l_dep_r) ? B : A;
    parti_be_merged = (dep_flag == parti_dep::l_dep_r) ? A : B;

    auto outer_loops_to_merge = pa_to_merge->get_outer_loops(),
         outer_loops_be_merged = parti_be_merged->get_outer_loops();

    auto merged_loop_size = get_great_common_loop_size(
            outer_loops_to_merge, outer_loops_be_merged);

    if (!merged_loop_size) return false;

    // validate axis binding
    merged_loop_size = check_parti_loop_axis_binding(
            parti_be_merged, pa_to_merge, merged_loop_size);

    if (!merged_loop_size
            || (keep_outerloop_size
                    && (merged_loop_size != outer_loops_to_merge.size()
                            || merged_loop_size
                                    != outer_loops_be_merged.size())))
        return false;

    // check cost model
    if (!pa_to_merge->cost_->make_decision_for_parti(
                parti_be_merged, merged_loop_size, parti_merge_kind::vertical))
        return false;

    if (auto max_to_merge_anchor_map = pa_to_merge->get_anchor_inside_loop(
                outer_loops_to_merge[merged_loop_size - 1])) {
        if (joint_op
                && std::any_of(joint_op->get_inputs().begin(),
                        joint_op->get_inputs().end(),
                        [&max_to_merge_anchor_map, &pa_to_merge](
                                const graph_tensor_ptr &inp) {
                            return pa_to_merge->contains(inp->producer_owner_)
                                    && max_to_merge_anchor_map->blocked_gt_set_
                                               .find(inp)
                                    != max_to_merge_anchor_map->blocked_gt_set_
                                               .end();
                        }))
            return false;
    } else {
        return false;
    }

    SC_MODULE_INFO << "merging two partition:";
    SC_MODULE_INFO << A->func_;
    SC_MODULE_INFO << B->func_;

    pa_to_merge->func_->name_ += "_merge_" + parti_be_merged->func_->name_;

    merge_parti_impl(pa_to_merge, parti_be_merged, merged_loop_size, joint_op);

    SC_MODULE_INFO << "Merging result:";
    SC_MODULE_INFO << pa_to_merge->func_;

    return true;
}

// usually used by crossover dispatcher
static bool try_merge_mixed_parti_vertically(
        mixed_parti_t *A, mixed_parti_t *B) {
    A = A->get_root(), B = B->get_root();
    if (A == B) return false;
    // if A and B are forked, do not merge them
    if (check_parti_forked(A, B)) return false;
    // skip single op partition
    if (A->is_single_op_parti() || B->is_single_op_parti()) return false;
    bool image_affinity = A->contain_convolution() && B->contain_convolution();
    return try_merge_mixed_parti_vertically(A, B, nullptr, !image_affinity);
}

static bool try_merge_mixed_parti_with_joint_op(const mixed_parti_t::ptr &A,
        const mixed_parti_t::ptr &B, const sc_op_ptr &joint_op) {
    mixed_parti_t *default_lhs, *default_rhs;
    // If no dependence, first execute the partition with more tunable ops
    if (joint_op->isa<tunable_op_t>()
            || (A->contain_tunable_op()
                    && (!B->contain_tunable_op()
                            || (B->count_op_with_type<tunable_op_t>()
                                    > A->count_op_with_type<
                                            tunable_op_t>())))) {
        default_lhs = B.get();
        default_rhs = A.get();
    } else {
        default_lhs = A.get();
        default_rhs = B.get();
    }
    auto &ctx = A->ctx_;
    if (ctx->flags_.opt_level_ == sc_opt_level::lv1) {
        return try_merge_mixed_parti_vertically(
                default_lhs, default_rhs, joint_op);
    }
    // if pre-op fusion succeed, skip vertical merge
    return try_merge_brgemm_and_preop_parti(default_lhs, default_rhs, joint_op)
            || try_merge_mixed_parti_vertically(
                    default_lhs, default_rhs, joint_op);
}

mixed_parti_t::mixed_parti_t(
        const context_ptr &ctx, const sc_op_ptr &op, const dep_mat_ptr &dep_m)
    : dep_m_(dep_m), ctx_(ctx) {
    auto &graph = op->get_owner_graph();
    if (graph.is_dynamic()) {
        cost_ = std::make_shared<dynamic_fusion_cost_model_t>(this,
                graph.attrs_.get_or_else("temp.dynamic_fusion_policy",
                        dynamic_fusion_policy_t::max_fusion));
    } else {
        cost_ = std::make_shared<static_fusion_cost_model_t>(this);
    }
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

mixed_parti_t::mixed_parti_t(const context_ptr &ctx, const func_t &func,
        const fusion_anchor_mgr_t &fmgr, const dep_mat_ptr &dep_m)
    : dep_m_(dep_m), ctx_(ctx), func_(func) {
    cost_ = std::make_shared<static_fusion_cost_model_t>(this);
    append_fusion_anchor(fmgr.fanchor_list_);
}

bool mixed_parti_t::is_ok_to_add(sc_op *op) {
    if (merged_to) { return get_root()->is_ok_to_add(op); }
    if (empty()) { return false; }
    if (!fusion_partition_t::is_ok_to_add(op, (*dep_m_))) {
        SC_MODULE_INFO << op->op_name_ << "_" << op->logical_op_id_
                       << " fail to add partition: " << func_->name_
                       << ", due to potential graph "
                          "dependency ring risk";
        return false;
    }
    // TODO(yunfei): when all tunable ops completely refactored to managed
    // threads mode, remove this if.
    if (op->isa<tunable_op_t>() && contain_nested_parallel_for()) {
        return false;
    }
    // search suitable anchor for op
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
    // update op_anchor_map_
    if (iter != op_anchor_map_.end()) {
        // auto skip
        if (iter->second == fanchor_map) return;
        auto res = cmp_op_anchor(op, iter->second, fanchor_map);
        // if new anchor is more smaller than the current one
        if (res == cmp_res::l_larger_r) {
            // overwrite new anchor
            op_anchor_map_[op] = fanchor_map;
        } else if (res == cmp_res::equal) {
            // Except for reduce_collect_op_t with COPY kind because this mode
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

bool mixed_parti_t::add(const sc_op_ptr &op) {
    if (merged_to) { return get_root()->add(op); }
    // pre-check anchor for op in avoid of throw assert during following
    // appending stage
    search_op_anchor_in_parti(op.get(), this);
    // if no anchor is ready, return false
    if (!ready_for_op(op.get())) return false;
    /* adding op to partition */
    SC_MODULE_INFO << "================  adding op: " << op->op_name_ << "_"
                   << op->logical_op_id_ << " to partition: " << func_->name_
                   << " ================";
    if (op->need_dynamic_internal_query()) {
        dyn_inter_ = std::make_shared<mixed_dyn_internal_info_t>(ctx_);
    }
    auto mixed_op = op->dyn_cast<op_traits::mixed_partition_acceptable>();
    mixed_op->append_mixed_partition(this);
    func_->name_ += "_" + op->op_name_ + std::to_string(op->logical_op_id_);
    SC_MODULE_INFO << func_;
    ops.insert(op);
    committed_ops_.emplace_back(op);
    return true;
}

void mixed_parti_t::append_fusion_anchor(const fuse_anchor_map_ptr &fanchor) {
    auto cur = try_convert_anchor(ctx_, fanchor);
    cur->binded_mxp_ = this;
    fanchors_.emplace_back(cur);
}

fuse_anchor_map_ptr mixed_parti_t::lookup_anchor_map(
        sc_op *op, bool throw_assert) const {
    if (merged_to) { return get_root()->lookup_anchor_map(op); }
    auto iter = op_anchor_map_.find(op);
    auto ret = (iter != op_anchor_map_.end()) ? iter->second : nullptr;
    if (throw_assert) {
        COMPILE_ASSERT(ret,
                "No dispatched fusion anchor map found for "
                        << op->op_name_
                        << " in this partition, please try to search it "
                           "firstly");
    }
    return ret;
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
    if (!fanchor->isa<fuse_grouped_anchor_map_t>()) {
        COMPILE_ASSERT(anchor.checked_as<stmts>()->seq_.empty(),
                "Could not remove this fanchor, because it is not empty");
    } else {
        for (auto &sub_group : anchor.checked_as<stmts>()->seq_) {
            COMPILE_ASSERT(sub_group.checked_as<stmts>()->seq_.empty(),
                    "Could not remove this fanchor, because it is not empty");
        }
    }
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
    std::vector<stmt>::iterator anchor_iter
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
        fuse_anchor_map_ptr fanchor) const {
    if (merged_to) { return get_root()->get_outer_loops(fanchor); }
    std::vector<for_loop> outer_loops;
    if (!func_) return outer_loops;
    auto body = func_->body_;

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
 *   // input anchor
 *   {}
 *   // body
 *   ...
 *   // output anchor
 *   {}
 * }
 * */
fuse_anchor_map_ptr mixed_parti_t::get_anchor_inside_loop(
        const for_loop &loop, bool input_anchor) const {
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
            if (anchor_map) {
                // check anchor type
                if (input_anchor != anchor_map->is_input_anchor()) continue;
                return anchor_map;
            }
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
        bool target_anchor_found = false;
        for_loop next_loop = for_loop();
        for (auto &s : ss->seq_) {
            if (s.isa<stmts>()) {
                auto inner_ss = s.static_as<stmts>();
                if (inner_ss->seq_.empty()) {
                    auto anchor_map = lookup_anchor_map(inner_ss);
                    if (anchor_map) {
                        if (anchor_map == target_fanchor) {
                            target_anchor_found = true;
                            break;
                        }
                        continue;
                    } else {
                        next_loop = for_loop();
                        break;
                    }
                } else {
                    next_loop = for_loop();
                    break;
                }
            } else if (s.isa<for_loop>() && !next_loop.defined()) {
                next_loop = s.checked_as<for_loop>();
            } else if (s.cast<evaluate>()
                               .map([](const evaluate &v) {
                                   return v->value_.as<intrin_call>();
                               })
                               .filter([](const intrin_call &v) {
                                   return v->type_
                                           == intrin_type::set_thread_idle_func;
                               })
                               .has_value()) {
                // When prefetch is enabled, a prefetcher function call
                // `evaluate{set_thread_idle_func(...)}` would have been added
                // to the loop. We should ignore it when finding next anchor.
                continue;
            } else {
                next_loop = for_loop();
                break;
            }
        }
        return target_anchor_found ? for_loop() : next_loop;
    }
    return for_loop();
}

class var_replacer_t : public ir_visitor_t {
public:
    var_replacer_t(const std::unordered_map<expr_c, expr_c> &remap)
        : remap_(remap) {}
    std::unordered_map<expr_c, expr_c> remap_;
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;
    expr_c visit(var_c v) override {
        auto itr = remap_.find(v);
        if (itr != remap_.end()) { return itr->second; }
        return v;
    }
};

void mixed_parti_t::try_split_outermost_loop_on_num_threads(
        int64_t num_groups) {
    auto outer_loops = get_outer_loops();
    if (outer_loops.empty()) return;
    auto outermost_loop = outer_loops[0];

    // cache original largest anchor inside outermost loop
    auto origin_large_anchor = get_anchor_inside_loop(outermost_loop);
    // node ptr replace map
    node_ptr_map remap;
    // change IR and record replace map
    auto split_inner_loop
            = outermost_loop->split_on_num_threads(num_groups, &remap);
    // split outer loop axis binder
    ax_binder_.split_init_axis(0);
    mxp_replacer_t(remap).replace_anchor(fanchors_);
    // if original largest anchor not null, create new largest anchor under new
    // outermost loop
    if (origin_large_anchor) {
        auto s = builder::make_stmts_unattached({}).checked_as<stmts>();
        add_parent_node(s, outermost_loop->body_);
        outermost_loop->body_.checked_as<stmts>()->seq_.emplace_back(s);
        // dummy fsmap, the tensor belongs to this scope will not be shrinked
        fslice_map new_fsmap;
        // copy from original fsmap
        new_fsmap.datamap_ = origin_large_anchor->fsmap_.datamap_;
        // create var inplacer
        std::unordered_map<expr_c, expr_c> vmap = {{split_inner_loop->var_,
                make_expr<constant_node>(
                        UINT64_C(0), split_inner_loop->var_->dtype_)}};
        var_replacer_t repl(vmap);
        //  modify slice range
        for (auto &fspair : new_fsmap.datamap_) {
            for (auto &srange : fspair.second) {
                for (auto &r : srange) {
                    auto new_v = repl.dispatch(r.first);
                    // if necesary
                    if (!new_v.ptr_same(r.first)) {
                        // set new offset
                        r.first = new_v.remove_const();
                        // set new range
                        r.second = dim2unsigned(get_expr_as_int(r.second)
                                * (get_expr_as_int(split_inner_loop->iter_end_)
                                        - get_expr_as_int(
                                                split_inner_loop
                                                        ->iter_begin_)));
                    }
                }
            }
        }
        // append new anchor into parti
        append_fusion_anchor(std::make_shared<fuse_anchor_map_t>(s, new_fsmap));
    }
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

    node_ptr_map remap;
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
    if (ctx_->flags_.concat_optimization_) {
        buf_alloc_.copy_concat_memory_attrs_tsr2buf();
    }
    buf_alloc_.tensor_initialize();
    buf_alloc_.declare_tensor();
    buf_alloc_.set_shrink_info();
    buf_alloc_.query_inplace();
    buf_alloc_.calibrate_info();
}

bool mixed_parti_t::is_parti_inp(const graph_tensor *gt) const {
    if (merged_to) { return get_root()->is_parti_inp(gt); }
    auto ths = this;
    return std::any_of(gt->uses_.begin(), gt->uses_.end(),
                   [&ths](const std::pair<int, sc_op_weak_ptr_t> &user) {
                       return ths->contains(user.second.get());
                   })
            && !contains(gt->producer_owner_);
}

bool mixed_parti_t::is_parti_inp(const graph_tensor_ptr &gt) const {
    return is_parti_inp(gt.get());
}

bool mixed_parti_t::is_parti_out(const graph_tensor *gt) const {
    if (merged_to) { return get_root()->is_parti_out(gt); }
    auto ths = this;
    return contains(gt->producer_owner_)
            && std::any_of(gt->uses_.begin(), gt->uses_.end(),
                    [&ths](const std::pair<int, sc_op_weak_ptr_t> &user) {
                        return !ths->contains(user.second.get());
                    });
}

bool mixed_parti_t::is_parti_out(const graph_tensor_ptr &gt) const {
    return is_parti_out(gt.get());
}

bool mixed_parti_t::is_const_parti() const {
    return (get_ops_size() == 1
            && get_ith_op(0)->attrs_.get_or_else(
                       "constant", const_kind::not_const)
                    != const_kind::not_const);
}

bool mixed_parti_t::contain_convolution() const {
    return contain_op_with_type<ops::conv_fwd_core_op_t>()
            || contain_op_with_type<ops::conv_bwd_data_core_op_t>()
            || contain_op_with_type<ops::conv_bwd_weight_core_op_t>();
}

bool mixed_parti_t::contain_nested_parallel_for() const {
    auto outer_loops = get_outer_loops();
    if (outer_loops.empty()) return false;
    if (outer_loops.size() == 1) {
        if (outer_loops[0]->num_threads_ == 0) return false;
        if (outer_loops[0]->body_.isa<stmts>()) {
            auto &ss = outer_loops[0]->body_.static_as<stmts>()->seq_;
            for (auto &s : ss) {
                if (s.isa<for_loop>()) {
                    if (s.static_as<for_loop>()->num_threads_ > 0) return true;
                }
            }
        }
        return false;
    }
    return (outer_loops[0]->num_threads_ > 0)
            && (outer_loops[1]->num_threads_ > 0);
}

bool mixed_parti_t::contain_tunable_op() const {
    return contain_op_with_type<tunable_op_t>();
}

bool mixed_parti_t::contain_elemwise_op_only() const {
    if (merged_to) { return get_root()->contain_elemwise_op_only(); }
    for (auto &op : ops) {
        if (!is_elementwise_op(op.get())) return false;
    }
    return true;
}

bool mixed_parti_t::is_single_op_parti() const {
    if (merged_to) { return get_root()->is_single_op_parti(); }
    return ops.size() == 1;
}

bool mixed_parti_t::is_optimized() const {
    if (merged_to) { return get_root()->is_optimized(); }
    if (committed_ops_.empty()) return false;
    return is_optimized_sub_graph(get_host_graph());
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

    // Cost Model
    cost_ = nullptr;
}

float mixed_parti_t::evaluate_perf() {
    if (merged_to) { return get_root()->evaluate_perf(); }
    return cost_->evaluate();
}

bool mixed_parti_t::is_small_workload() const {
    if (merged_to) { return get_root()->is_small_workload(); }
    // skip partition owning more than one ops
    if (committed_ops_.size() != 1) return false;
    // get sinlge op
    auto single_op = committed_ops_[0].get();
    // skip tunable op
    if (single_op->isa<tunable_op_t>()) return false;
    // get committed anchor
    auto committed_anchor = lookup_anchor_map(single_op);
    COMPILE_ASSERT(committed_anchor, "No committed anchor found")
    // query small op workload
    return committed_anchor->is_small_op_workload(single_op);
}

std::vector<mixed_parti_t::ptr> collect_parti_set(
        const std::vector<mixed_parti_t::ptr> &op_2_partition,
        bool ignore_const) {
    std::unordered_set<mixed_parti_t::ptr> parti_set;
    std::vector<mixed_parti_t::ptr> parti_vec;
    for (auto &parti : op_2_partition) {
        if (parti_set.count(parti)) { continue; }
        // auto skip nullptr or const parti
        if (!parti || (ignore_const && parti->is_const_parti())) continue;
        parti_set.insert(parti);
        parti_vec.push_back(parti);
    }
    return parti_vec;
}

static bool check_repartition(const mixed_parti_t::ptr &parti) {
    if (!parti || parti->get_ops_size() < 2) return false;
    // check tensorview in edge of partition
    bool repartition = false;
    // check buffer is tensorptr
    auto check_parti_out_tptr = [&repartition, &parti](
                                        const graph_tensor_ptr &out) {
        auto producer = out->producer_owner_;
        COMPILE_ASSERT(parti->buf_alloc_.g2b_map_.haskey(out),
                "No buffer allocated found for output "
                "of " << producer->op_name_
                      << producer->logical_op_id_)
        if (parti->buf_alloc_.g2b_map_.get(out).isa<tensorptr>()) {
            if (producer->isa<tensor_view_op_t>()) {
                auto tv_op = producer;
                graph_tensor *tv_inp;
                while (tv_op->isa<tensor_view_op_t>()) {
                    tv_op->attrs_[op_attr_key::no_fuse] = true;
                    tv_inp = tv_op->get_inputs()[0].get();
                    tv_op = tv_inp->producer_owner_;
                }
                if (parti->buf_alloc_.g2b_map_.get(tv_inp).isa<tensorptr>()) {
                    tv_inp->attrs_[mixed_partition_hint::no_inplace] = true;
                }
            } else {
                out->attrs_[mixed_partition_hint::no_inplace] = true;
            }
            repartition = true;
        }
    };
    for (auto &op : parti->ops) {
        // 1. check partition output buffer should not be tensoptr
        for (auto &out : op->get_outputs()) {
            if (parti->is_parti_out(out)) check_parti_out_tptr(out);
        }
        // 2. check reorder
        if (auto reo = op->isa<reorder_op_t>()) {
            if (!op->dyn_cast<reorder_op_t>()->support_output_loop()
                    || op->get_inputs()[0]->uses_.size() == 1) {
                continue;
            }
            auto tunable_users = search_tuneop_linearly(op);
            // mark this kind of reorder to do pre-op fusion during
            // repartition stage
            if (tunable_users && !parti->contains(tunable_users.get())) {
                op->attrs_[op_attr_key::break_pre_fuse] = true;
                repartition = true;
                // the input of this reorder will be the new output of
                // partition, check tptr in advance to reduce repartition times
                check_parti_out_tptr(op->get_inputs()[0]);
            }
        }
    }
    return repartition;
}

bool mixed_parti_t::contain_input_anchor() const {
    if (merged_to) { return get_root()->contain_input_anchor(); }
    return std::any_of(fanchors_.begin(), fanchors_.end(),
            [](const fuse_anchor_map_ptr &anchor) {
                return anchor->is_input_anchor();
            });
}

static mixed_parti_t::ptr try_execute_pre_op_fusion(const context_ptr &ctx,
        const sc_op_ptr &op,
        std::vector<mixed_parti_t::ptr> &avaliable_input_parti) {
    mixed_parti_t::ptr parent_partition = nullptr;
    // only suitable for tunable op
    if (!op->isa<tunable_op_t>()) return parent_partition;
    if (ctx->flags_.opt_level_ == sc_opt_level::lv1) {
        return parent_partition;
    }
    // collect reorder parti
    std::vector<mixed_parti_t *> reo_parti;
    for (auto &inp_parti : avaliable_input_parti) {
        if (inp_parti->get_ops_size() == 1
                && inp_parti->contain_op_with_type<reorder_op_t>()) {
            reo_parti.emplace_back(inp_parti->get_root());
        }
    }
    if (reo_parti.empty()) return parent_partition;
    // create tunable partition for possible input fusion anchor
    parent_partition = std::make_shared<mixed_parti_t>(
            reo_parti[0]->ctx_, op, reo_parti[0]->dep_m_);
    if (!parent_partition->contain_input_anchor()) {
        parent_partition->clear();
        return nullptr;
    }
    // search input anchor for reorder op
    for (auto &rparti : reo_parti) {
        auto reo = rparti->committed_ops_[0];
        bool input_anchor_found = false;
        for (auto &fanchor : parent_partition->fanchors_) {
            // only search input anchor
            if (!fanchor->is_input_anchor()) continue;
            // check output of reorder
            auto reo_out = reo->get_outputs()[0], reo_in = reo->get_inputs()[0];
            if (!fanchor->fsmap_.hasvalue(reo_out)) continue;
            fslice_map tmp_fsmap;
            // copy to tmp fsmap
            tmp_fsmap.get(reo_out) = fanchor->fsmap_.get(reo_out);
            infer_status_map_t stat_map(parent_partition->ctx_, false);
            reo->stc_cast<fusible_op_t>()->pre_slice_ranges(
                    tmp_fsmap, stat_map);
            if (stat_map.is_ok()) {
                input_anchor_found = true;
                fanchor->fsmap_.get(reo_in) = tmp_fsmap.get(reo_in);
            }
        }
        if (input_anchor_found) {
            // set pre-op fusion attr
            reo->attrs_.set(mixed_partition_hint::pre_fuse_begin_op, true);
            // pre fuse reorder
            parent_partition->add(reo);
            // remove pre-op fusion attr
            reo->attrs_.remove(mixed_partition_hint::pre_fuse_begin_op);
            // clear origin parti
            rparti->clear();
            // reset root pointer
            rparti->merged_to = parent_partition;
        }
    }
    if (parent_partition->get_ops_size() == 1) {
        parent_partition->clear();
        // reset
        parent_partition = nullptr;
    } else {
        auto &ops = parent_partition->committed_ops_;
        auto old_op = std::move(ops.front());
        ops.erase(ops.begin());
        ops.emplace_back(std::move(old_op));
    }
    return parent_partition;
}

static mixed_parti_t::ptr try_execute_post_op_fusion(const context_ptr &ctx,
        const sc_op_ptr &op,
        std::vector<mixed_parti_t::ptr> &avaliable_input_parti) {
    mixed_parti_t::ptr parent_partition = nullptr;
    // try to preop merge the partitons of all inputs in advance
    for (auto &inp_parti : avaliable_input_parti) {
        if (!parent_partition) {
            parent_partition = inp_parti;
        } else if (!parent_partition->empty() && !inp_parti->empty()) {
            COMPILE_ASSERT(op->isa<op_traits::mixed_partition_acceptable>(),
                    "Unexpected op type find: " << op->op_name_)
            op->dyn_cast<op_traits::mixed_partition_acceptable>()
                    ->search_anchor(parent_partition.get());
            op->dyn_cast<op_traits::mixed_partition_acceptable>()
                    ->search_anchor(inp_parti.get());
            if (ctx->flags_.opt_level_ == sc_opt_level::lv1) { continue; }
            if (parent_partition->ready_for_op(op.get())
                    && inp_parti->ready_for_op(op.get())) {
                // the difference between later partition merge with
                // joint op is that current merge need to check
                // partition connection to ensure legality.
                try_merge_brgemm_and_preop_parti(
                        parent_partition.get(), inp_parti.get(), nullptr);
            }
        }
    }
    // reset parent partition
    parent_partition = nullptr;
    // try to add op to input partition
    for (auto &inp_parti : avaliable_input_parti) {
        if (inp_parti->is_ok_to_add(op.get())) {
            if (parent_partition) {
                try_merge_mixed_parti_with_joint_op(
                        parent_partition, inp_parti, op);
                parent_partition = std::static_pointer_cast<mixed_parti_t>(
                        parent_partition->get_root()->shared_from_this());
            } else {
                parent_partition = inp_parti;
                // Do not merge input partition according hint
                if (op->attrs_.get_or_else(
                            mixed_partition_hint::no_gather_op, false))
                    break;
            }
        }
    }
    if (parent_partition) {
        if (!parent_partition->get_root()->add(op)) {
            // set no gather input partition hint
            op->attrs_.set(mixed_partition_hint::no_gather_op, true);
        }
    }
    return parent_partition;
}

bool do_partition(const context_ptr &ctx, sc_graph_t &g,
        std::vector<mixed_parti_t::ptr> &op_2_partition) {
    // validate partition
    bool repartition = false;
    // a speculative DFS visitor
    op_visitor_t visitor
            = op_visitor_t::dfs_topology_speculative_sort(g.ops_.size());
    visitor.visit_graph(g, [&](op_visitor_t *visitor, const sc_op_ptr &op) {
        if (op->isa<input_op>() || op->isa<output_op>()) return;
        mixed_parti_t::ptr parent_partition = nullptr;
        if (op->attrs_.get_or_else(op_attr_key::no_fuse, false)) {
            SC_MODULE_INFO << op->op_name_ << "_" << op->logical_op_id_
                           << " is marked as no fuse";
        } else {
            if (!op->attrs_.get_or_else(op_attr_key::break_pre_fuse, false)) {
                std::vector<mixed_parti_t::ptr> avaliable_input_parti;
                // collect avaliable input partition
                auto sorted_inputs = get_sorted_inputs_by_layout_input(op);
                for (auto &in : sorted_inputs) {
                    // if an input is fusible and is not "break_post_fuse"
                    if (!in->producer_owner_->attrs_.get_or_else(
                                op_attr_key::break_post_fuse, false)
                            && !in->producer_owner_->attrs_.get_or_else(
                                    op_attr_key::no_fuse, false)
                            && in->producer_owner_->attrs_.get_or_else(
                                       "constant", const_kind::not_const)
                                    == const_kind::not_const) {
                        auto &cur_in_partition
                                = op_2_partition[in->producer_owner_
                                                         ->logical_op_id_];
                        if (cur_in_partition)
                            avaliable_input_parti.emplace_back(
                                    cur_in_partition);
                    } else if (in->producer_owner_->attrs_.get_or_else(
                                       op_attr_key::break_post_fuse, false)) {
                        SC_MODULE_INFO << op->op_name_ << "_"
                                       << op->logical_op_id_
                                       << " fail to add partition because it "
                                          "is marked as break post fuse";
                    }
                }
                // try pre op fusion
                parent_partition = try_execute_pre_op_fusion(
                        ctx, op, avaliable_input_parti);
                if (!parent_partition) {
                    // try post op fusion
                    parent_partition = try_execute_post_op_fusion(
                            ctx, op, avaliable_input_parti);
                }
            } else {
                SC_MODULE_INFO << op->op_name_ << "_" << op->logical_op_id_
                               << " fail to add partition because it is marked "
                                  "as break pre fuse";
            }
        }
        if (!parent_partition || !parent_partition->contains(op.get())) {
            // op was not added into parent partition, usually after unexpected
            // input partition merge, as the result, set repatition flag to
            // trigger next round of partition from the view of performance.
            if (parent_partition && !parent_partition->contains(op.get())) {
                repartition = true;
            }
            parent_partition = std::make_shared<mixed_parti_t>(
                    ctx, op, std::make_shared<op_dep_matrix_t>(g));
        }
        op_2_partition[op->logical_op_id_] = parent_partition;
    });

    for (auto &parti : op_2_partition) {
        if (parti) {
            parti = std::static_pointer_cast<mixed_parti_t>(
                    parti->get_root()->shared_from_this());
        }
    }

    // collect parti set
    auto parti_set = collect_parti_set(op_2_partition);
    // assign checker
    auto checker = &check_repartition;
    std::for_each(parti_set.begin(), parti_set.end(),
            [&checker, &repartition](const mixed_parti_t::ptr &parti) {
                if ((*checker)(parti)) repartition = true;
            });
    if (repartition) {
        SC_MODULE_INFO << "================ Repartition the whole graph "
                          "================";
    }
    return !repartition;
}

bool mixed_parti_t::validate_optimization() const {
    return buf_alloc_.validate_tsr2var();
}

bool mixed_parti_t::can_optimize_outer_loop(bool allow_tensorview) const {
    if (merged_to) {
        return get_root()->can_optimize_outer_loop(allow_tensorview);
    }
    bool for_reduce = contain_op_with_type<op_traits::maybe_split_optimized_t>()
            && std::all_of(ops.begin(), ops.end(),
                    [&](const sc_op_ptr &op) {
                        if (op->isa<movement_op_t>()) {
                            if (op->isa<tensor_view_op_t>()
                                    && allow_tensorview) {
                                return is_parti_out(op->get_outputs()[0])
                                        || is_parti_inp(op->get_inputs()[0]);
                            } else if (op->isa<reorder_op_t>()) {
                                return op->attrs_.get_or_else(
                                        op_attr_key::no_fuse, false);
                            } else {
                                return false;
                            }
                        } else {
                            return is_elementwise_op(op.get())
                                    || op->isa<reduce_op_t>()
                                    || op->isa<reduce_collect_op_t>()
                                    || (op->isa<reduce_compute_op_t>()
                                            && !op->stc_cast<
                                                          reduce_compute_op_t>()
                                                        ->is_partial_reduce());
                        }
                    })
            && std::any_of(ops.begin(), ops.end(), [&](const sc_op_ptr &op) {
                   std::vector<int> rd_axis;
                   if (auto rd_op = op->dyn_cast<reduce_op_t>()) {
                       rd_axis = rd_op->get_rd_axis();
                   } else if (auto rd_op
                           = op->dyn_cast<reduce_compute_op_t>()) {
                       rd_axis = rd_op->get_rd_axis();
                   } else {
                       return false;
                   }
                   std::sort(rd_axis.begin(), rd_axis.end());
                   int cur = (op->get_inputs()[0]
                                      ->details_.get_blocking_dims()
                                      .size()
                           - rd_axis.size());
                   /** E.g
                    * - reduce input dims: [32,64,16,16]
                    * - rd_axis: [2,3]
                    *  It is unecessary to optimize outer loop for this kind
                    * of reduce op
                    * */
                   for (auto &ax : rd_axis) {
                       if (ax != cur) return true;
                       cur++;
                   }
                   return false;
               });
    bool for_pooling = contain_op_with_type<pooling_op_t>()
            && std::all_of(ops.begin(), ops.end(), [&](const sc_op_ptr &op) {
                   return (!op->isa<movement_op_t>() || op->isa<padding_op_t>())
                           && !op->isa<tunable_op_t>();
               });
    return !is_optimized() && (for_reduce || for_pooling);
}

static bool try_optimize_reduce(mixed_parti_t *parti, sc_graph_t &sub_graph,
        const std::unordered_map<sc_op_ptr, sc_op_ptr> &graph2orig_ops) {
    // currently disable reduce optimization in dynamic
    if (!parti->contain_op_with_type<op_traits::maybe_split_optimized_t>()
            || sub_graph.is_dynamic())
        return false;

    bool redo = false;
    auto ctx = parti->ctx_;

    auto outer_loops = parti->get_outer_loops();
    // if parti contains nested parallel for, it could not be ensured that the
    // inner loop is not parallel
    bool nested_parallel_found = parti->contain_nested_parallel_for();
    // calculate least loop size which satisfies parallelism
    size_t parallel_least_size = 0;
    if (!outer_loops.empty()) {
        for (parallel_least_size = 1; parallel_least_size < outer_loops.size();
                parallel_least_size++) {
            if (evaluate_loop_parallel_balance({outer_loops.begin(),
                        outer_loops.begin() + parallel_least_size})
                    == 1.0f) {
                break;
            }
        }
    }
    // If parallel loop can be ensured in advanced
    if (parallel_least_size > 0) {
        std::unordered_set<op_traits::maybe_split_optimized_t *>
                splited_reduce_set;
        parti->ax_binder_.run(parallel_least_size);
        for (auto &op : sub_graph.ops_) {
            if (auto red_op
                    = op->dyn_cast<op_traits::maybe_split_optimized_t>()) {
                if (!red_op->can_split_op()) continue;
                if (auto rd_op = op->dyn_cast<reduce_op_t>()) {
                    if (!nested_parallel_found)
                        splited_reduce_set.insert(red_op);
                    continue;
                } else if (auto rd_op = op->dyn_cast<reduce_compute_op_t>()) {
                    COMPILE_ASSERT(rd_op->is_partial_reduce(),
                            "Only partial reduce is expected")
                    if (nested_parallel_found) {
                        splited_reduce_set.insert(red_op);
                        continue;
                    }
                    auto rd_axis = rd_op->get_rd_axis();
                    // transform to plain rd axis
                    rd_axis = transform_axis_blocking2plain(
                            op->get_inputs()[0]->details_, rd_axis);
                    rd_axis.erase(rd_axis.begin());
                    // find original reduce op in partition
                    auto orig_iter = graph2orig_ops.find(op);
                    if (orig_iter == graph2orig_ops.end()) continue;
                    auto &rd_binding_axis = parti->ax_binder_.bd_ax_map_.get(
                            orig_iter->second->get_inputs()[0]);
                    if (rd_binding_axis.empty()) continue;
                    // If all of `rd_axis` would not appear on parallel
                    // outer loops
                    if (std::all_of(rd_binding_axis.begin(),
                                rd_binding_axis.end(),
                                [&rd_axis](const std::vector<int> &bd_ax) {
                                    return std::all_of(bd_ax.begin(),
                                            bd_ax.end(),
                                            [&rd_axis](const int &ax) {
                                                return std::all_of(
                                                        rd_axis.begin(),
                                                        rd_axis.end(),
                                                        [&ax](const int &
                                                                        rd_ax) {
                                                            return ax != rd_ax;
                                                        });
                                            });
                                })) {
                        splited_reduce_set.insert(red_op);
                    }
                } else {
                    COMPILE_ASSERT(
                            0, "Unexpected kind of op found: " << op->op_name_)
                }
            }
        }
        // If split reduce op exist
        for (auto &red_op : splited_reduce_set) {
            auto op = dynamic_cast<sc_op *>(red_op);
            reduce_operator rd_type;
            // check padding except for reduce add
            if (auto rd_op = op->dyn_cast<reduce_op_t>()) {
                rd_type = rd_op->get_rd_op();
            } else if (auto rd_op = op->dyn_cast<reduce_compute_op_t>()) {
                rd_type = rd_op->get_rd_op();
            } else {
                COMPILE_ASSERT(
                        0, "Unexpected kind of op found: " << op->op_name_)
            }
            if (rd_type != reduce_operator::add) {
                auto &plain_dims
                        = op->get_inputs()[0]->details_.get_plain_dims();
                auto &fmt = op->get_inputs()[0]->details_.get_format();
                auto blocking_dims = sc_data_format_t::get_blocking_shapes(
                        plain_dims, fmt);
                auto padded_dims = sc_data_format_t::get_padded_plain_shapes(
                        blocking_dims, fmt);
                // currently, do not support split with padding
                if (plain_dims != padded_dims) continue;
            }
            // pre-check
            if (op->isa<reduce_compute_op_t>()) {
                // find original op in partition
                auto orig_iter = graph2orig_ops.find(op->shared_from_this());
                if (orig_iter == graph2orig_ops.end()) continue;
                // get shrink info
                auto slice_info = parti->buf_alloc_.get_shrinked_info(
                        parti->buf_alloc_.g2b_map_.get(
                                orig_iter->second->get_outputs()[0]));
                if (slice_info.empty()) continue;
                sc_dim prod = get_dims_product(
                        get_expr_to_dims(get_slice_shape(slice_info)));
                auto tsr_simd_len = vectorize_step(
                        ctx, op->get_inputs()[0]->details_.dtype_.type_code_);
                if (!check_tsr_len_under_resigter_size(prod, tsr_simd_len))
                    continue;
            }
            // Do split
            red_op->split_op(ctx, sub_graph, 1);
            redo = true;
        }
    }

    return redo;
}

static bool try_optimize_concat(mixed_parti_t *parti, sc_graph_t &sub_graph) {
    return parti->ctx_->flags_.concat_optimization_
            && concat_memory_planning_on_graph(sub_graph);
}

static bool try_optimize_outer_loop(mixed_parti_t *parti, sc_graph_t &sub_graph,
        const std::unordered_map<sc_op_ptr, sc_op_ptr> &graph2orig_ops) {
    // prepare stage
    auto &ops = sub_graph.ops_;
    // check reorder in partition which includes reduce op but
    // exclude tunable op
    if (parti->contain_op_with_type<op_traits::maybe_split_optimized_t>()
            && !parti->contain_tunable_op()
            && parti->contain_op_with_type<reorder_op_t>()) {
        bool forced_reorder_axis = false;
        std::unordered_set<sc_op_ptr> reo_op_set;
        auto run_threads = runtime_config_t::get().get_num_threads();
        for (auto &op : ops) {
            if (op->is_removed_) continue;
            if (op->isa<op_traits::maybe_split_optimized_t>()) {
                std::vector<int> rd_axis;
                if (auto rd_op = op->dyn_cast<reduce_op_t>()) {
                    rd_axis = rd_op->get_rd_axis();
                } else if (auto rd_op = op->dyn_cast<reduce_compute_op_t>()) {
                    if (rd_op->is_partial_reduce()) {
                        forced_reorder_axis = false;
                        break;
                    }
                    rd_axis = rd_op->get_rd_axis();
                }
                auto shape = op->get_inputs()[0]->details_.get_blocking_dims();
                int outer_rd_axis_size = 1;
                for (int i = 0; i < *rd_axis.begin(); i++) {
                    outer_rd_axis_size *= shape[i];
                }
                if (outer_rd_axis_size < run_threads)
                    forced_reorder_axis = true;
            } else if (op->isa<reorder_op_t>()) {
                reo_op_set.insert(op);
            }
        }
        if (forced_reorder_axis) {
            std::for_each(reo_op_set.begin(), reo_op_set.end(),
                    [&graph2orig_ops](const sc_op_ptr &op) {
                        op->attrs_[op_attr_key::no_fuse] = true;
                        // sync origin op in partition
                        auto orig_iter = graph2orig_ops.find(op);
                        COMPILE_ASSERT(orig_iter != graph2orig_ops.end(),
                                "Could not find original op in partition")
                        orig_iter->second->attrs_[op_attr_key::no_fuse] = true;
                    });
        }
    }

    // optimize loop order
    if (parti->can_optimize_outer_loop()) {
        for (auto &op : ops) {
            if (is_elementwise_op(op.get())
                    || op->isa<op_traits::maybe_split_optimized_t>()
                    || op->isa<pooling_op_t>()) {
                // set optimized outer loop hint
                op->attrs_.set(
                        mixed_partition_hint::optimized_outer_loop, true);
            }
        }
        return true;
    } else {
        return false;
    }
}

static bool try_optimize_fusion_anchor(mixed_parti_t *parti,
        sc_graph_t &sub_graph,
        const std::unordered_map<sc_op_ptr, sc_op_ptr> &graph2orig_ops) {
    // auto skip
    if (parti->committed_ops_.size() < 2) return false;
    auto &ctx = parti->ctx_;
    // check the op whether is the elementwise op with last dim undividable
    auto elem_op_with_last_dim_undividable
            = [&ctx, &parti](const sc_op_ptr &op) -> bool {
        if (!is_elementwise_op(op.get())) return false;
        auto committed_anchor = parti->lookup_anchor_map(op.get(), false);
        if (!committed_anchor) return false;
        auto gt = op->get_outputs()[0];
        COMPILE_ASSERT(committed_anchor->fsmap_.hasvalue(gt),
                "Unexpected case for elementwise op: " << op->op_name_ << "_"
                                                       << op->logical_op_id_)
        auto &slice_info = committed_anchor->fsmap_.get(gt);
        if (slice_info.size() != 1) return false;
        auto compute_shape = get_slice_shape(slice_info[0]);
        if (!compute_shape.back().isa<constant>()) return false;
        auto last_dim = get_expr_as_int(compute_shape.back());
        // get max lanes
        auto dtype = gt->details_.dtype_;
        auto lanes = vectorize_step(ctx, dtype.type_code_);
        return ((last_dim > lanes) && (last_dim % lanes != 0));
    };
    // query partition op on graph
    auto query_op_on_graph = [&graph2orig_ops](const sc_op_ptr &op) {
        auto iter = std::find_if(graph2orig_ops.begin(), graph2orig_ops.end(),
                [&op](const std::pair<sc_op_ptr, sc_op_ptr> &kv) {
                    return kv.second == op;
                });
        COMPILE_ASSERT(
                iter != graph2orig_ops.end(), "Could not find mapping op")
        return iter->first;
    };
    // set hint about fusion anchor
    auto set_hint = [](std::vector<sc_op_ptr> &ops) -> bool {
        bool flag = false;
        if (ops.size() > 1) {
            for (auto &op : ops) {
                op->attrs_.set(mixed_partition_hint::split_anchor_op, true);
            }
            flag = true;
        }
        ops.clear();
        return flag;
    };

    auto dep_m = op_dep_matrix_t(sub_graph);
    bool redo = false;
    std::vector<sc_op_ptr> target_ops;
    // visit sorted commit ops
    for (auto &op : parti->committed_ops_) {
        // get mapping op on sub graph
        auto sub_graph_op = query_op_on_graph(op);
        if (target_ops.empty() && elem_op_with_last_dim_undividable(op)) {
            // lookup commited anchor
            auto &committed_anchor = *parti->lookup_anchor_map(op.get());
            if (typeid(committed_anchor) == typeid(fuse_anchor_map_t)) {
                target_ops.emplace_back(sub_graph_op);
            }
        } else if (!target_ops.empty()) {
            auto first_op
                    = graph2orig_ops.find(target_ops.front())->second.get();
            // successive elementwise op and same anchor with first op and have
            // dependency with last target op
            if (elem_op_with_last_dim_undividable(op)
                    && (parti->lookup_anchor_map(first_op)
                            == parti->lookup_anchor_map(op.get()))
                    && dep_m.lookup(target_ops.back(), sub_graph_op) == 1) {
                target_ops.emplace_back(sub_graph_op);
            } else if (parti->lookup_anchor_map(first_op)
                            == parti->lookup_anchor_map(op.get(), false)
                    && !target_ops.empty()) {
                target_ops.clear();
                // replace first op
                if (elem_op_with_last_dim_undividable(op)) {
                    target_ops.emplace_back(sub_graph_op);
                }
            } else {
                // check redo flag
                redo |= set_hint(target_ops);
            }
        }
    }
    // if the first op still exists utils loop ends
    redo |= set_hint(target_ops);

    return redo;
}

bool try_optimize_parti(mixed_parti_t *parti, sc_graph_t &sub_graph,
        const std::unordered_map<sc_op_ptr, sc_op_ptr> &graph2orig_ops) {
    if (sub_graph.is_dynamic()) { return false; }
    // skip already optimized parti
    if (parti->is_optimized()) return false;
    bool need_optimize = false;
    // optimize reduce
    need_optimize |= try_optimize_reduce(parti, sub_graph, graph2orig_ops);
    // optimize concat
    need_optimize |= try_optimize_concat(parti, sub_graph);
    // optimize loop order
    need_optimize |= try_optimize_outer_loop(parti, sub_graph, graph2orig_ops);
    // optimize fusion anchor
    need_optimize
            |= try_optimize_fusion_anchor(parti, sub_graph, graph2orig_ops);

    if (need_optimize) {
        sub_graph.attrs_.set(mixed_partition_hint::optimized_sub_graph, true);
    }
    return need_optimize;
}

static std::string get_graph_name(const sc_graph_t &graph) {
    std::string ret;
    for (auto &op : graph.ops_) {
        if (op->isa<input_op>() || op->isa<output_op>()
                || op->isa<constant_op_t>())
            continue;
        if (!ret.empty()) ret += "_";
        ret += op->op_name_;
    }
    return ret;
}

static std::string get_parti_prefix(const mixed_parti_t &parti) {
    std::string prefix;
    if (parti.ops.size() > 1) {
        prefix = "partition_";
        auto outer_loops = parti.get_outer_loops();
        if (!outer_loops.empty()) {
            prefix = "outerloop_" + print_loops_range(outer_loops) + "_"
                    + prefix;
        }
    }
    return prefix;
}

static void search_first_prefetch_op(mixed_parti_t &parti) {
    for (const auto &op : parti.committed_ops_) {
        if (op->isa<tunable_op_t>() && op->isa<op_traits::may_prefetch_t>()) {
            op->attrs_[mixed_partition_hint::first_prefetch_op] = true;
            break;
        }
    }
}

std::shared_ptr<mixed_fuse_op_t> mixed_parti_t::transform_to_mixed_op() {
    COMPILE_ASSERT(!empty(), "Could not transform empty partition")
    // Get original graph
    auto &g = get_host_graph();
    // Make sub graph
    sc_graph_t sub_graph;
    sub_graph.sync_dynamic_info_with_graph(g);
    std::vector<graph_tensor_ptr> fused_op_in, fused_op_out;
    std::vector<expr> arg_ins, arg_out;
    std::string op_name;
    // the mapping for original op to op in sub graph
    std::unordered_map<sc_op_ptr, sc_op_ptr> graph2orig_ops;
    // the mapping for original LT in original ops to fuse => the LT in the
    // sub_graph.
    std::unordered_map<graph_tensor_ptr, graph_tensor_ptr> orig_2_graph;
    auto get_or_create_graph_tsr = [&](const graph_tensor_ptr &orig_lr) {
        auto itr = orig_2_graph.find(orig_lr);
        if (itr != orig_2_graph.end()) { return itr->second; }
        auto ret = std::make_shared<graph_tensor>(nullptr, orig_lr->details_);
        orig_2_graph.insert(std::make_pair(orig_lr, ret));
        return ret;
    };
    auto visitor = op_visitor_t::dfs_topology_sort(g.ops_.size());
    std::unordered_set<graph_tensor_ptr> input_tsr_set;
    // search first prefetch op of original graph and set attr on it
    search_first_prefetch_op(*this);
    // visit original graph
    visitor.visit_graph(g, [&](op_visitor_t *visitor, const sc_op_ptr &op) {
        if (ops.find(op) == ops.end()) { return; }
        std::vector<graph_tensor_ptr> new_graph_in, new_graph_ou;
        for (auto &in : op->get_inputs()) {
            new_graph_in.emplace_back(get_or_create_graph_tsr(in));
            if (is_parti_inp(in)
                    && input_tsr_set.find(in) == input_tsr_set.end()) {
                // if the input is not included in the parti, make an input
                // node
                auto new_input_op = sub_graph.make_input({new_graph_in.back()});
                // inherit constant attr for input if necessary
                if (in->producer_owner_->attrs_.has_key("constant")) {
                    new_input_op->attrs_.set("constant",
                            in->producer_owner_->attrs_.get<int>("constant"));
                }
                // add the input in the args of the fused op in orig
                // sub_graph
                fused_op_in.emplace_back(in);
                input_tsr_set.insert(in);
                COMPILE_ASSERT(buf_alloc_.g2b_map_.haskey(in),
                        "No buffer allocated for " << op->op_name_ << "_"
                                                   << op->logical_op_id_
                                                   << " inputs")
                arg_ins.emplace_back(buf_alloc_.g2b_map_.get(in));
            }
        }
        for (auto &out : op->get_outputs()) {
            new_graph_ou.emplace_back(get_or_create_graph_tsr(out));
            // if the output is a "cut" - an edge across the parti and
            // outside of the parti; or if concat optimization is enabled:
            if (is_parti_out(out)
                    || (ctx_->flags_.concat_optimization_
                            && op->isa<concat_op_t>()
                            && op->attrs_.get_or_else(
                                    concat_optim_attr_keys::is_final_concat,
                                    false))) {
                // if there is a use outside of the parti, the tensor should
                // be marked "output"
                const auto &outtsr = new_graph_ou.back();
                auto new_out_op = sub_graph.make_output({outtsr});
                // make a new output tensor for the fused_op_t in the
                // original sub_graph
                fused_op_out.emplace_back(
                        std::make_shared<graph_tensor>(nullptr, out->details_));
                // save the mapping of the tensor to be replaced => new
                // tensor
                output_replace_map[out] = fused_op_out.back();
                COMPILE_ASSERT(buf_alloc_.g2b_map_.haskey(out),
                        "No buffer allocated for " << op->op_name_ << "_"
                                                   << op->logical_op_id_
                                                   << " outputs")
                auto out_buffer = buf_alloc_.g2b_map_.get(out);
                // if outbuffer is already reused, set attr on output op
                if (out_buffer->attr().has_key(
                            attr_keys::tensor_inplace_hint)) {
                    new_out_op->attrs_.set("buffer_already_reused", true);
                }
                arg_out.emplace_back(out_buffer);
            }
        }
        auto copyable = op->dyn_cast<op_traits::copyable_t>();
        assert(copyable);
        auto copied = copyable->copy(new_graph_in, new_graph_ou, sub_graph);
        copied->copy_dispatch_key_set_from_op(op);
        graph2orig_ops.insert(std::make_pair(copied, op));
        // build the fused op name
        if (!op_name.empty()) op_name += '_';
        op_name += copied->op_name_;
    });

    // copy graph in avoid of redo fall-back case
    auto copied_grpah = copy_graph(sub_graph);

    if (try_optimize_parti(this, sub_graph, graph2orig_ops)) {
        SC_MODULE_INFO << "Optimizing mixed partition for current pattern: "
                       << func_->name_;
        // copy optimized sub graph
        auto copied_opt_grpah = copy_graph(sub_graph);
        // redo mixed partition with setting hint
        do_mixed_partition(ctx_, sub_graph);
        bool fall_back = false;
        std::vector<mixed_parti_t::ptr> parti_list;
        std::string mx_op_name;
        bool non_mixed_op_exist = false;
        // redo validation stage
        for (auto &op : sub_graph.ops_) {
            if (op->isa<input_op>() || op->isa<output_op>()
                    || op->isa<constant_op_t>())
                continue;
            if (!mx_op_name.empty()) mx_op_name += "_";
            if (auto mx_op = op->dyn_cast<mixed_fuse_op_t>()) {
                COMPILE_ASSERT(mx_op->parti_list_.size() == 1,
                        "Unexpected partition size found: "
                                << mx_op->parti_list_.size())
                if (!mx_op->parti_list_[0]->validate_optimization()) {
                    // reset
                    fall_back = true;
                    SC_MODULE_INFO << "invalid optimized reduce detected, "
                                      "fall-back "
                                      "to original pattern";
                    break;
                }
                parti_list.emplace_back(mx_op->parti_list_[0]);
                mx_op_name += get_graph_name(mx_op->sub_graph_);
            } else {
                if (op->isa<reduce_collect_op_t>()) {
                    fall_back = true;
                    SC_MODULE_INFO << "reduce collect op must be fused with "
                                      "reduce compute op, fall-back "
                                      "to original pattern";
                    break;
                }
                if (!op->isa<tensor_view_op_t>()) non_mixed_op_exist = true;
                mx_op_name += op->op_name_;
            }
        }
        if (!fall_back) {
            if (parti_list.size() == 1 && !non_mixed_op_exist) {
                mx_op_name = get_parti_prefix(*parti_list[0]) + mx_op_name;
            } else {
                mx_op_name = "multi_partitions_" + mx_op_name;
            }
            std::vector<sc_op_ptr> lower_args(sub_graph.get_output_ops());
            auto input_ops = sub_graph.get_input_ops();
            lower_args.insert(
                    lower_args.end(), input_ops.begin(), input_ops.end());
            auto modu = lower_graph(ctx_, sub_graph, lower_args, false);
            auto main_func = modu->get_entry_func();
            main_func->name_ = mx_op_name;
            main_func->decl_->name_ = mx_op_name;
            return std::make_shared<mixed_fuse_op_t>(mx_op_name, parti_list,
                    modu, copied_opt_grpah,
                    /*ins*/ fused_op_in,
                    /*outs*/
                    fused_op_out, any_map_t {});
        } else {
            // fall-back
            parti_list.clear();
            mx_op_name.clear();
        }
    }

    // mark read/write buffer
    graph::mark_read_or_write_buffers(arg_ins, true);
    graph::mark_read_or_write_buffers(arg_out, false);
    // build up final func name and param
    std::vector<expr> args = arg_out, buffer_args;
    args.insert(args.end(), arg_ins.begin(), arg_ins.end());
    std::for_each(args.begin(), args.end(), [&g](const expr &arg) {
        COMPILE_ASSERT(arg.isa<tensor>(),
                "Only tensor node is expected for function argument, but "
                "got " << arg)
        arg->attr()[mixed_partition_hint::cut_buffer] = true;
        if (g.is_dynamic()
                || g.attrs_.get_or_else("temp.parent_graph_dynamic", false)) {
            arg->attr()[attr_keys::always_trans] = true;
        }
    });
    if (dyn_inter_) {
        buffer_args = args;
        args.emplace_back(dyn_inter_->inter_funcs_param_);
    }
    func_->params_ = args;
    func_->decl_->params_ = args;
    if (dyn_inter_) {
        assert(dyn_inter_->inter_call_.defined());
        // replace output buffer of single core/internal func.
        auto reset_args = [&buffer_args](std::vector<expr> &target_args,
                                  const std::vector<expr> &extra_args) {
            target_args = buffer_args;
            target_args.insert(
                    target_args.end(), extra_args.begin(), extra_args.end());
        };
        reset_args(dyn_inter_->inter_call_->args_,
                dyn_inter_->inter_call_extra_args_);
        reset_args(dyn_inter_->inter_func_->params_,
                dyn_inter_->inter_func_extra_args_);
        dyn_inter_->inter_func_->decl_->params_
                = dyn_inter_->inter_func_->params_;
        reset_args(dyn_inter_->single_core_func_->params_,
                dyn_inter_->single_core_func_extra_args_);
        dyn_inter_->single_core_func_->decl_->params_
                = dyn_inter_->single_core_func_->params_;
    }
    // buffer schedule: declare, set shrink info, tensor initilize and query
    // inplace
    buffer_schedule();

    // clear unused fanchor, in avoid of loop fuse break
    clear_fanchors();

    // remove all parallel flag
    remove_parallel(func_, true);

    // set function name
    func_->name_ = get_parti_prefix(*this) + op_name;
    func_->decl_->name_ = func_->name_;
    // link op name
    op_name = func_->name_;

    SC_MODULE_INFO << "mixed partition result:";
    SC_MODULE_INFO << func_;

    auto fused_op = std::make_shared<mixed_fuse_op_t>(op_name,
            std::vector<mixed_parti_t::ptr> {
                    std::static_pointer_cast<mixed_parti_t>(
                            shared_from_this())},
            nullptr, copied_grpah,
            /*ins*/ fused_op_in,
            /*outs*/
            fused_op_out, any_map_t {});

    return fused_op;
}

using crossover_alg
        = std::function<void(const std::vector<mixed_parti_t::ptr> &parti_vec)>;

static void crossover_dispatcher(
        const std::vector<mixed_parti_t::ptr> &parti_vec,
        parti_merge_kind merge_kind) {
    // select merger by merge kind
    bool (*merger)(mixed_parti_t * A, mixed_parti_t * B);
    switch (merge_kind) {
        case parti_merge_kind::vertical: {
            merger = try_merge_mixed_parti_vertically;
            break;
        }
        case parti_merge_kind::horizontal: {
            merger = try_merge_mixed_parti_horizontally;
            break;
        }
        case parti_merge_kind::parallel: {
            merger = try_merge_mixed_parti_parallel;
            break;
        }
        default: COMPILE_ASSERT(0, "Unknown partition merge kind found")
    }
    auto op_size = parti_vec.size();
    for (size_t i = 0; i < op_size; i++) {
        auto parti_A = parti_vec[i];
        if (!parti_A) continue;
        for (size_t j = i; j < op_size; j++) {
            auto parti_B = parti_vec[j];
            if (!parti_B) continue;
            merger(parti_A.get(), parti_B.get());
        }
    }
}

static void horizontal_crossover(
        const std::vector<mixed_parti_t::ptr> &parti_vec) {
    SC_MODULE_INFO << "Applying horizontal merge crossover algorithm...";
    crossover_dispatcher(parti_vec, parti_merge_kind::horizontal);
}

static void parallel_crossover(
        const std::vector<mixed_parti_t::ptr> &parti_vec) {
    SC_MODULE_INFO << "Applying parallel merge crossover algorithm...";
    crossover_dispatcher(parti_vec, parti_merge_kind::parallel);
}

static void vertical_crossover(
        const std::vector<mixed_parti_t::ptr> &parti_vec) {
    SC_MODULE_INFO << "Applying vertical merge crossover algorithm...";
    crossover_dispatcher(parti_vec, parti_merge_kind::vertical);
}

static void crossover_partition(std::vector<mixed_parti_t::ptr> &op_2_partition,
        const std::vector<crossover_alg> &algs) {
    std::vector<mixed_parti_t::ptr> parti_vec
            = collect_parti_set(op_2_partition);
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

static expr merge_fusion_condition_by_parti_list(
        const std::vector<mixed_parti_t::ptr> &partis) {
    auto parti_set = collect_parti_set(partis);
    expr ret = false;
    for (auto &parti : parti_set) {
        if (parti) { ret = ret || parti->get_fusion_policy_condition(); }
    }
    return ret;
}

void do_mixed_partition(const context_ptr &ctx, sc_graph_t &graph) {
    auto op_size = graph.ops_.size();
    // mapping from op id => partition
    std::vector<mixed_parti_t::ptr> op_2_partition;
    // set max iter times
    constexpr int maxiter = 3;
    // dynamic policy condition
    expr fusion_policy_condition = false;
    for (int i = 0; i < maxiter; i++) {
        op_2_partition.clear();
        op_2_partition.resize(op_size);
        bool ret = do_partition(ctx, graph, op_2_partition);
        auto cur_cond = merge_fusion_condition_by_parti_list(op_2_partition);
        fusion_policy_condition = fusion_policy_condition || cur_cond;
        if (ret)
            break;
        else if (i == maxiter - 1) {
            SC_MODULE_INFO << "mixed partition exceed max iteration times, "
                              "please enlarge limitation and try again";
            return;
        }
    }
    graph.attrs_.set("temp.fusion_policy_condition", fusion_policy_condition);
    if (ctx->flags_.opt_level_ >= sc_opt_level::lv3) {
        std::vector<crossover_alg> algs = {
                horizontal_crossover, parallel_crossover, vertical_crossover};
        crossover_partition(op_2_partition, algs);
    }
    std::vector<sc_op_ptr> fused_ops;
    for (auto &parti : op_2_partition) {
        if (!parti || !parti->output_replace_map.empty() || parti->empty()
                || parti->ops.size() == 1) {
            // if a partition has been processed or it is empty or single op,
            // skip it.
            continue;
        }

        auto fused_op = parti->transform_to_mixed_op();

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

        for (auto &op : partition->ops) {
            op->remove();
        }

        // no main op is expected
        assert(!partition->main_tunable_op);
    }
    graph.reset_op_ids();
}

void mixed_partition(sc_graph_t &graph, const context_ptr &ctx) {
    if (!graph.attrs_.get_or_else("temp.fuse", 1)) { return; }
    SC_MODULE_INFO << "Starting Mixed Partition...";
    do_mixed_partition(ctx, graph);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
