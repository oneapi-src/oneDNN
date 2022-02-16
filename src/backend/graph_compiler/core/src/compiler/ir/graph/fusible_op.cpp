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

#include <assert.h>
#include <atomic>
#include <numeric>
#include <unordered_map>

#include <algorithm>
#include <utility>
#include "fusible_op.hpp"
#include "fusion_mgr.hpp"
#include "outer_loop_generator.hpp"
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/loop_transform.hpp>
#include <microkernel/builtin.hpp>
#include <util/exceptions.hpp>
#include <util/reflection.hpp>
#include <util/string_utils.hpp>
#include <util/utils.hpp>

namespace sc {
// incrementor for loop
static std::atomic<int> idx = {0};
// incrementor for temp var
static std::atomic<int> var_idx = {0};
// helper function to get current var_idx
std::string fusion_create_var_idx() {
    return std::string("_") + std::to_string(var_idx++);
}

// workload penalty coefficient for transpose/reorder measured by
// for(i, 0, 128){
//     for(j, 0, 256){
//         B[j, i] = A[i, j];
//     }
// }
// TODO(xxx): currently we mark this penalty on op, we will add loop analysis
// pass for tensor sequential access analysis in future
static const size_t workload_penalty_coefficient = 16UL;

static std::vector<tensor_slice *> convert_t(std::vector<tensor_slice> &src) {
    std::vector<tensor_slice *> dst(src.size());
    std::transform(src.begin(), src.end(), dst.begin(),
            [](tensor_slice &t) { return &t; });
    return dst;
}

static std::vector<const tensor_slice *> convert_const_t(
        std::vector<tensor_slice> &src) {
    std::vector<const tensor_slice *> dst(src.size());
    std::transform(src.begin(), src.end(), dst.begin(),
            [](tensor_slice &t) { return &t; });
    return dst;
}

static std::vector<tensor_slice> make_tensor_slice(
        const std::vector<graph_tensor_ptr> &data,
        const std::string &tensor_name, std::vector<expr> &flattened) {
    std::vector<tensor_slice> expected;
    for (size_t i = 0; i < data.size(); ++i) {
        std::vector<expr> dims
                = dims_to_expr(data[i]->details_.get_blocking_dims());
        auto aexpr = builder::make_tensor(tensor_name + std::to_string(i), dims,
                data[i]->details_.dtype_);
        flattened.emplace_back(aexpr);
        expected.emplace_back(tensor_slice(aexpr));
    }
    return expected;
}

static ir_module_ptr fusible_op_get_func(fusible_op_t *op,
        outer_loop_generator_t &gen, const context_ptr &ctx,
        bool check_parallel) {
    fusion_manager fmgr;
    std::vector<graph_tensor_ptr> ins;
    std::vector<graph_tensor_ptr> outs;
    for (auto &in : op->get_inputs()) {
        ins.emplace_back(fmgr.make<input_op>(in->details_)->get_outputs()[0]);
    }
    for (auto &out : op->get_outputs()) {
        outs.emplace_back(
                std::make_shared<graph_tensor>(nullptr, out->details_));
    }
    auto copyable = op->dyn_cast<op_traits::copyable_t>();
    COMPILE_ASSERT(copyable, "The fusible op should be copyable");
    auto copied = copyable->copy(ins, outs, fmgr.get_graph());
    COMPILE_ASSERT(copied->get_outputs().size() == 1,
            "Currently only support 1 output only");
    fmgr.make<output_op>(copied->get_outputs()[0]);
    return lower_fusion_manager(ctx, &gen, op, &fmgr, check_parallel);
}

sc_dims get_expr_to_dims(const std::vector<expr> &dim) {
    sc_dims dim_int;
    dim_int.reserve(dim.size());
    for (const expr &d : dim) {
        COMPILE_ASSERT(d.isa<constant_c>(), "non-constant value found.");
        dim_int.emplace_back(get_const_as_int(d.static_as<constant_c>()));
    }
    return dim_int;
}

ir_module_ptr fusible_op_t::get_func(context_ptr ctx) {
    outer_loop_generator_t gen;
    return fusible_op_get_func(this, gen, ctx, true);
}

ir_module_ptr reorder_op_t::get_func(context_ptr ctx) {
    top_level_anchor_generator_t gen;
    attrs_.set(op_attr_key::no_fuse, true);
    auto ret = fusible_op_get_func(this, gen, ctx, false);
    auto func = ret->get_entry_func();
    auto body = func->body_.as<stmts>();
    COMPILE_ASSERT(body.defined(), "Expecting a body");
    COMPILE_ASSERT(body->seq_.size() == 2, "Expecting 2 stmt in reorder body");
    auto loop = body->seq_[0].as<for_loop>();
    COMPILE_ASSERT(loop.defined(), "Expecting a for loop in reorder body");
    loop->kind_ = for_type::PARALLEL;
    return ret;
}

ir_module_ptr reshape_op_t::get_func(context_ptr ctx) {
    top_level_anchor_generator_t gen;
    attrs_.set(op_attr_key::no_fuse, true);
    auto ret = fusible_op_get_func(this, gen, ctx, true);
    return ret;
}

void fusible_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {
    if (this->isa<constant_op_t>()) {
        out_formats.push_back({info_.outputs_[0]->details_.get_format()});
    } else {
        out_formats.push_back({info_.inputs_[0]->details_.get_format()});
    }
}

size_t fusible_op_t::compute_workload(const std::vector<shape_dtype_pair> &ins,
        const std::vector<shape_dtype_pair> &outs) {
    size_t wkld = 0UL;
    auto accumulate_workload
            = [&wkld](size_t weight, const shape_dtype_pair &v) {
                  auto &dtype = v.second;
                  wkld += utils::get_sizeof_type(dtype) * weight;
              };
    std::for_each(ins.begin(), ins.end(),
            std::bind(accumulate_workload,
                    static_cast<size_t>(
                            op_traits::workload_computable_t::read_weight),
                    std::placeholders::_1));
    std::for_each(outs.begin(), outs.end(),
            std::bind(accumulate_workload,
                    static_cast<size_t>(
                            op_traits::workload_computable_t::write_weight),
                    std::placeholders::_1));
    return wkld;
}

size_t fusible_op_t::compute_fusible_workload(const context_ptr &ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    std::vector<shape_dtype_pair> wkld_ins, wkld_outs;
    wkld_ins.resize(inputs.size());
    wkld_outs.resize(dst.size());
    auto get_shape_dtype_pair = [](const tensor_slice *v) {
        return std::make_pair(get_expr_to_dims(v->shape_), v->get_base_dtype());
    };
    std::transform(inputs.begin(), inputs.end(), wkld_ins.begin(),
            get_shape_dtype_pair);
    std::transform(
            dst.begin(), dst.end(), wkld_outs.begin(), get_shape_dtype_pair);
    return compute_workload(wkld_ins, wkld_outs);
}

sc_op_ptr fusible_op_t::copy(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, sc_graph_t &mgr) {
    auto new_op = op_traits::auto_copyable_t::copy(ins, outs, mgr);
    new_op->stc_cast<fusible_op_t>()->fuse_in_brgemm_ = fuse_in_brgemm_;
    new_op->stc_cast<fusible_op_t>()->alg_kind_ = alg_kind_;
    return new_op;
}

// check input size
static const std::vector<graph_tensor_ptr> &check_size(
        const std::vector<graph_tensor_ptr> &ins) {
    COMPILE_ASSERT(ins.size() == 1, "input size should be equal to one");
    return ins;
}

static inline uint32_t vectorize_step(
        const context_ptr &ctx, sc_data_etype detype) {
    return std::min(16U, ctx->get_max_vector_lanes(detype));
}

static bool slice_full_on_axes(
        const sc_dims &dim, slice_range ranges, const std::vector<int> &axes) {
    for (auto &ax : axes) {
        if (!ranges[ax].first.isa<constant>()
                || !ranges[ax].second.isa<constant>()) {
            return false;
        }
        if (get_const_as_int(ranges[ax].first.checked_as<constant>()) != 0
                || get_const_as_int(ranges[ax].second.checked_as<constant>())
                        != dim[ax]) {
            return false;
        }
    }
    return true;
}

// todo use uint64_t instead of mask count
expr make_select_by_mask(expr lhs_vec, int mask_count, uint32_t vector_lanes) {
    if (mask_count == -1) { return lhs_vec; }
    expr rhs_vec = make_expr<constant_node>(
            std::vector<union_val>(vector_lanes, UINT64_C(0)),
            sc_data_type_t(lhs_vec->dtype_.type_code_, vector_lanes));
    if (mask_count == 0) { return rhs_vec; }
    std::vector<union_val> mask_const(vector_lanes, 1.f);
    for (uint32_t i = mask_count; i < vector_lanes; i++) {
        mask_const[i] = 0.f;
    }
    expr mask_vec = make_expr<constant_node>(
            mask_const, sc_data_type_t::f32(vector_lanes));
    expr zero_vec = make_expr<constant_node>(
            std::vector<union_val>(vector_lanes, 0.f),
            sc_data_type_t::f32(vector_lanes));
    return builder::make_select(mask_vec > zero_vec, lhs_vec, rhs_vec);
}

struct mask_compute_func_t {
    mask_compute_func_t(const std::function<stmt(const std::vector<expr> &,
                    std::vector<expr::lvalue_proxy_t> &, int, float)> &func)
        : impl_(func) {}
    stmt operator()(const std::vector<expr> &in,
            std::vector<expr::lvalue_proxy_t> &out, int mask_count = -1,
            float mask_value = 0.f) const {
        return impl_(in, out, mask_count, mask_value);
    }
    std::function<stmt(const std::vector<expr> &,
            std::vector<expr::lvalue_proxy_t> &, int, float)>
            impl_;
};

/** Determine whether masks are needed during elementwise computation and
 * generate conditional expressions for the mask
 * @param src input slice
 * @param plain_dims plain shapes
 * @param format input format
 * @param iter_vars input loop vars
 * @param lanes simd lanes
 * @param condition key is related iter var, value is two conditions:first means
 * in the condition, all elements should be all computed,second means only
 * `mask_count` elements will be computed
 * @param last_axis_mask mask count, how many elements should be computed in
 * this time. -1 means all.
 * */
static void compute_mask_and_generate_condition(
        const std::vector<const tensor_slice *> &src, const sc_dims &plain_dims,
        sc_data_format_t format, const std::vector<expr> &iter_vars, int lanes,
        std::unordered_map<expr, std::pair<expr, expr>> &conditions,
        int &last_axis_mask) {
    auto blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, format);
    auto padded_dims
            = sc_data_format_t::get_padded_plain_shapes(blocking_dims, format);
    auto &format_code = format.format_code_;
    if (plain_dims == padded_dims) { return; }
    auto offset = src[0]->get_offset();
    auto shapes = src[0]->get_shape();
    size_t ndims = format_code.ndims();
    assert(offset.size() == ndims && shapes.size() == ndims
            && iter_vars.size() == ndims);
    auto plain2block = format_code.collect_p2b_mapping();
    for (size_t i = 0; i < plain2block.size(); i++) {
        auto &orig_dim = i;
        if (plain_dims[orig_dim] == padded_dims[orig_dim]
                || plain2block[i].size() == 1) {
            continue;
        }
        auto &block_dim = plain2block[i][plain2block[i].size() - 1];
        auto blocks = format_code.collect_blocking_index(orig_dim);
        int padding_count = 0;
        conditions[iter_vars[block_dim]].first
                = lanes + (iter_vars[block_dim] + offset[block_dim]);
        conditions[iter_vars[block_dim]].second
                = iter_vars[block_dim] + offset[block_dim];
        for (int b = static_cast<int>(blocks.size()) - 1; b >= 0; b--) {
            if (b > 0 && blocks[b - 1] % blocks[b] != 0) { padding_count++; }
            conditions[iter_vars[block_dim]].first
                    = conditions[iter_vars[block_dim]].first
                    + (iter_vars[plain2block[i][b]] + offset[plain2block[i][b]])
                            * format.blocks_[blocks[b]];
            conditions[iter_vars[block_dim]].second
                    = conditions[iter_vars[block_dim]].second
                    + (iter_vars[plain2block[i][b]] + offset[plain2block[i][b]])
                            * format.blocks_[blocks[b]];
        }
        conditions[iter_vars[block_dim]].first
                = conditions[iter_vars[block_dim]].first
                < dim2unsigned(plain_dims[orig_dim]);
        conditions[iter_vars[block_dim]].second
                = conditions[iter_vars[block_dim]].second
                < dim2unsigned(plain_dims[orig_dim]);
        COMPILE_ASSERT(padding_count < 2,
                "Currently we don't support multi-level padding mask.");
        if (block_dim == format_code.ndims() - 1) {
            assert(lanes > 1);
            last_axis_mask = plain_dims[orig_dim] % lanes;
        }
    }
}

void create_fusible_output_anchor(std::vector<stmt> &parent,
        const tensor_slice &dst, const std::vector<expr> &loop_vars,
        const std::vector<int> &anchor_pos_in_loop,
        const vectorized_info_t &vx_info, any_map_t &attrs) {
    if (attrs.has_key(op_attr_key::inner_anchor)) {
        // insert inner anchor (cache-level)
        auto tsr = dst.get_real_tensor();
        auto range = dst.get_ranges();
        if (range.size() != loop_vars.size()) return;
        COMPILE_ASSERT(std::all_of(anchor_pos_in_loop.begin(),
                               anchor_pos_in_loop.end(),
                               [&loop_vars](int pos) {
                                   return pos >= 0
                                           && pos <= static_cast<int>(
                                                      loop_vars.size());
                               }),
                "Could not create fusible output anchor at loop position: "
                        << utils::print_vector(anchor_pos_in_loop)
                        << ", due to only " << loop_vars.size()
                        << " loops found")
        // reset offset
        for (size_t j = 0; j < loop_vars.size(); j++) {
            if (anchor_pos_in_loop.end()
                    != std::find(anchor_pos_in_loop.begin(),
                            anchor_pos_in_loop.end(), static_cast<int>(j)))
                continue;
            if (!range[j].second.isa<constant>()) return;
            if (get_expr_as_int(range[j].second) == 1) continue;
            range[j].first = loop_vars[j];
            range[j].second = ((static_cast<int>(j) == vx_info.axis)
                            ? expr(int(vx_info.lanes))
                            : expr(1));
        }
        auto s = make_stmt<stmts_node_t>(std::vector<stmt> {});
        auto fanchor = fuse_anchor_t(s,
                std::make_pair(std::vector<tensor_slice> {tensor_slice(
                                       tsr, std::move(range))},
                        std::vector<tensor_slice> {}));
        // redirect gen_fanchor
        attrs[op_attr_key::inner_anchor] = fanchor;
        parent.emplace_back(s);
    }
}

static void create_fusible_output_anchor(stmt &parent, const tensor_slice &dst,
        const std::vector<expr> &loop_vars,
        const std::vector<int> &anchor_pos_in_loop,
        const vectorized_info_t &vx_info, any_map_t &attrs) {
    std::vector<stmt> ss = parent.isa<stmts>() ? parent.static_as<stmts>()->seq_
                                               : std::vector<stmt> {parent};
    create_fusible_output_anchor(
            ss, dst, loop_vars, anchor_pos_in_loop, vx_info, attrs);
    parent = make_stmt<stmts_node_t>(std::move(ss));
}

static void compute_vectorized_op(const std::vector<const tensor_slice *> &src,
        const tensor_slice &dst, sc_op_info_t &info,
        const vectorized_info_t &vx_info,
        const mask_compute_func_t &compute_lanes,
        const mask_compute_func_t &compute_scalar, any_map_t &attrs,
        size_t wkld = 0UL, bool use_mask = false) {
    // nested loop vars
    std::vector<expr> iter_vars;
    // the indices for multiple inputs. First dim: the input, Second dim:
    // the dimemsions in the tensor
    std::vector<std::vector<expr>> src_indices_floor(src.size());
    std::vector<std::vector<expr>> src_indices_tail(src.size());
    // the indices for the output tensor
    std::vector<expr> dst_idx_floor;
    std::vector<expr> dst_idx_tail;
    for (unsigned i = 0; i < dst.nslice_dims(); i++) {
        // make the loop var for the for-loop
        iter_vars.emplace_back(builder::make_var(datatypes::index,
                std::string("_fuseiter") + std::to_string(idx++)));
        // for each input tensor
        for (size_t j = 0; j < src.size(); j++) {
            auto &src_idx_floor = src_indices_floor.at(j);
            auto &src_idx_tail = src_indices_tail.at(j);
            // push an index
            src_idx_floor.emplace_back(iter_vars.back());
            src_idx_tail.emplace_back(iter_vars.back());
        }
        // push an index for output tensor
        dst_idx_floor.emplace_back(iter_vars.back());
        dst_idx_tail.emplace_back(iter_vars.back());
    }
    auto tail_var = builder::make_var(
            datatypes::index, std::string("_fuseiter") + std::to_string(idx++));
    for (size_t j = 0; j < src.size(); j++) {
        auto &src_idx_tail = src_indices_tail.at(j);
        src_idx_tail[vx_info.axis] = tail_var;
    }
    dst_idx_tail[vx_info.axis] = tail_var;
    expr indexed_target_floor
            = builder::make_indexing(dst.tptr_, dst_idx_floor, vx_info.lanes);
    expr indexed_target_tail = builder::make_indexing(dst.tptr_, dst_idx_tail);
    std::vector<expr> indexed_input_floor, indexed_input_tail;
    for (unsigned j = 0; j < src.size(); j++) {
        indexed_input_floor.emplace_back(builder::make_indexing(
                src.at(j)->tptr_, src_indices_floor.at(j), vx_info.lanes));
        indexed_input_tail.emplace_back(builder::make_indexing(
                src.at(j)->tptr_, src_indices_tail.at(j)));
    }

    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder is set");
    std::vector<expr::lvalue_proxy_t> target_floor
            = {expr::lvalue_proxy_t(indexed_target_floor, false)};
    std::vector<expr::lvalue_proxy_t> target_tail
            = {expr::lvalue_proxy_t(indexed_target_tail, false)};
    auto slice_len = get_const_as_int(
            dst.get_shape().at(vx_info.axis).static_as<constant>());
    int floor = slice_len / vx_info.lanes * vx_info.lanes;
    int tail = slice_len % vx_info.lanes;
    int last_axis_mask = -1;
    std::unordered_map<expr, std::pair<expr, expr>> conditions;
    if (use_mask) {
        compute_mask_and_generate_condition(src,
                info.inputs_[0]->details_.get_plain_dims(),
                info.inputs_[0]->details_.get_format(), iter_vars,
                vx_info.lanes, conditions, last_axis_mask);
    }
    if (last_axis_mask != -1) {
        COMPILE_ASSERT(tail == 0,
                "Currently we only support mask in vectorize compute not "
                "tail.");
    }
    std::vector<stmt> tcur;
    stmt cur;
    // recover schedule loop
    for (int i = static_cast<int>(dst.get_shape().size() - 1); i >= 0; i--) {
        stmt body;
        // currently vx_axis should be last axis
        if (static_cast<int>(dst.get_shape().size()) == vx_info.axis + 1
                && i == vx_info.axis) {
            if (floor) {
                bld->push_scope();
                if (conditions.find(iter_vars[i]) != conditions.end()) {
                    assert(last_axis_mask != -1);
                    stmt no_mask = builder::make_stmts_unattached(
                            {compute_lanes(indexed_input_floor, target_floor)});
                    stmt semi_mask = builder::make_stmts_unattached(
                            {compute_lanes(indexed_input_floor, target_floor,
                                    last_axis_mask)});
                    stmt all_mask
                            = builder::make_stmts_unattached({compute_lanes(
                                    indexed_input_floor, target_floor, 0)});
                    cur = builder::make_if_else_unattached(
                            conditions[iter_vars[i]].first, no_mask,
                            builder::make_stmts_unattached(
                                    {builder::make_if_else_unattached(
                                            conditions[iter_vars[i]].second,
                                            semi_mask, all_mask)}));
                } else {
                    cur = compute_lanes(indexed_input_floor, target_floor);
                }
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                stmt s = bld->pop_scope();
                auto ss = std::vector<stmt> {s};
                if (!tail) // create fusible output anchor as demand
                    create_fusible_output_anchor(
                            ss, dst, iter_vars, {i + 1}, vx_info, attrs);
                cur = make_stmt<for_loop_node_t>(iter_vars.at(i), expr(0),
                        expr(floor), expr(int(vx_info.lanes)),
                        ss.size() > 1 ? make_stmt<stmts_node_t>(std::move(ss))
                                      : s,
                        true, for_type::NORMAL);
                tcur.emplace_back(cur);
            }
            if (tail) {
                bld->push_scope();
                cur = compute_scalar(indexed_input_tail, target_tail);
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                cur = make_stmt<for_loop_node_t>(tail_var, expr(floor),
                        expr(floor + tail), expr(1), bld->pop_scope(), true,
                        for_type::NORMAL);
                tcur.emplace_back(cur);
                // create fusible output anchor as demand
                create_fusible_output_anchor(
                        tcur, dst, iter_vars, {i}, vx_info, attrs);
            }

        } else {
            if (!tcur.empty() && tcur[0].defined()) {
                body = make_stmt<stmts_node_t>(std::move(tcur));
                tcur.clear();
                // address special condition, like temp_buffer is used
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0), dst.get_shape().at(i), expr(1),
                        std::move(body), true, for_type::NORMAL);
            } else if (cur.defined()) {
                body = make_stmt<stmts_node_t>(
                        std::vector<stmt> {std::move(cur)});
                // address special condition, like temp_buffer is used
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0), dst.get_shape().at(i), expr(1),
                        std::move(body), true, for_type::NORMAL);
            } else {
                // if cur not defined, means last axis of tensor slice has range
                // 1, e.g. tensor_slice{{i, 100},{0, 1}}
                indexed_target_floor
                        = builder::make_indexing(dst.tptr_, dst_idx_floor);
                for (unsigned j = 0; j < src.size(); j++) {
                    indexed_input_floor[j] = builder::make_indexing(
                            src.at(j)->tptr_, src_indices_floor.at(j));
                }

                std::vector<expr::lvalue_proxy_t> target_floor
                        = {expr::lvalue_proxy_t(indexed_target_floor, false)};
                bld->push_scope();
                cur = compute_scalar(indexed_input_floor, target_floor);
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                cur = make_stmt<for_loop_node_t>(iter_vars.at(i), expr(0),
                        dst.get_shape().at(i), expr(1), bld->pop_scope(), true,
                        for_type::NORMAL);
            }
        }
    }
    if (!tcur.empty() && tcur[0].defined()) {
        assert(dst.get_shape().size() == 1UL);
        // TODO(xxx): currenly we don't add merge_loop attribute for this
        // special case, need stronger loop analysis.
        for (auto &it : tcur) {
            bld->emit(it);
        }
    } else {
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    }
}

void compute_block_broadcast(const std::vector<const tensor_slice *> &src,
        const tensor_slice &dst, int bc_input_idx,
        const std::vector<int> &bc_axis, const vectorized_info_t &vx_info,
        const std::function<expr(expr, expr)> &compute,
        sc_data_type_t dtype = datatypes::f32, size_t wkld = 0UL) {
    // nested loop vars
    std::vector<expr> iter_vars;
    // the indices for multiple inputs. First dim: the input, Second dim: the
    // dimemsions in the tensor
    std::vector<expr> in_idx, in_bc_idx;
    // the indices for the output tensor
    std::vector<expr> dst_idx;

    COMPILE_ASSERT(bc_input_idx == 0 || bc_input_idx == 1,
            "bc_input_idx is expected to be 0 or 1")
    const tensor_slice *in_tsl = src[1 - bc_input_idx],
                       *in_bc_tsl = src[bc_input_idx];
    bool keep_dims = in_tsl->get_base_dims().size()
            == in_bc_tsl->get_base_dims().size();
    // add output type check, manual downcast
    sc_data_etype out_etype
            = dst.tptr_->dtype_.get_pointer_element().as_etype();
    // use src_indices.at(0) as default
    for (unsigned i = 0; i < dst.nslice_dims(); i++) {
        // make the loop var for the for-loop
        iter_vars.emplace_back(builder::make_var(datatypes::index,
                std::string("_fuseiter") + std::to_string(idx++)));
        in_idx.emplace_back(iter_vars.back());
        if (std::find(bc_axis.begin(), bc_axis.end(), i) != bc_axis.end()) {
            in_bc_idx.emplace_back(iter_vars.back());
        } else if (keep_dims) {
            in_bc_idx.emplace_back(0);
        }
        /** push an index for output tensor **/
        dst_idx.emplace_back(iter_vars.back());
    }
    // For empty bc_axis
    if (in_bc_idx.empty()) in_bc_idx = {0};
    std::vector<expr> in_idx_tail = in_idx, in_bc_idx_tail = in_bc_idx,
                      dst_idx_tail = dst_idx;
    auto tail_var = builder::make_var(
            datatypes::index, std::string("_fuseiter") + std::to_string(idx++));
    in_idx_tail[vx_info.axis] = tail_var;
    dst_idx_tail[vx_info.axis] = tail_var;

    expr indexed_target
            = builder::make_indexing(dst.tptr_, dst_idx, vx_info.lanes);
    expr indexed_input
            = builder::make_indexing(in_tsl->tptr_, in_idx, vx_info.lanes);

    expr indexed_target_tail = builder::make_indexing(dst.tptr_, dst_idx_tail);
    expr indexed_input_tail
            = builder::make_indexing(in_tsl->tptr_, in_idx_tail);
    if (!in_tsl->tptr_->dtype_.get_pointer_element().is_etype(out_etype)) {
        indexed_input = builder::make_cast(
                sc_data_type_t(out_etype, indexed_input->dtype_.lanes_),
                indexed_input);
        indexed_input_tail = builder::make_cast(
                sc_data_type_t(out_etype, indexed_input_tail->dtype_.lanes_),
                indexed_input);
    }
    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder is set");
    auto slice_len = get_const_as_int(
            dst.get_shape().at(vx_info.axis).static_as<constant>());
    int floor = slice_len / vx_info.lanes * vx_info.lanes;
    int tail = slice_len % vx_info.lanes;
    std::vector<stmt> tcur;
    stmt cur;
    bool bc_input_cast
            = !in_bc_tsl->tptr_->dtype_.get_pointer_element().is_etype(
                    out_etype);
    // recover schedule loop
    for (int i = static_cast<int>(dst.get_shape().size() - 1); i >= 0; i--) {
        stmt body;
        // move broadcast op to body
        if (static_cast<int>(dst.get_shape().size()) == vx_info.axis + 1
                && i == vx_info.axis) {
            // IF last dim is included in bc_axis.
            if (floor) {
                expr indexed_bc_input;
                if (bc_axis.back() == static_cast<int64_t>(vx_info.axis)) {
                    indexed_bc_input = builder::make_indexing(
                            in_bc_tsl->tptr_, in_bc_idx, vx_info.lanes);
                }
                // IF last dim is excluded in bc_axis.
                else {
                    indexed_bc_input = builder::make_broadcast(
                            builder::make_indexing(in_bc_tsl->tptr_, in_bc_idx),
                            static_cast<int>(vx_info.lanes));
                }
                if (bc_input_cast) {
                    indexed_bc_input = builder::make_cast(
                            sc_data_type_t(
                                    out_etype, indexed_bc_input->dtype_.lanes_),
                            indexed_bc_input);
                }
                bld->push_scope();
                cur = make_stmt<assign_node_t>(indexed_target,
                        bc_input_idx == 1
                                ? compute(indexed_input, indexed_bc_input)
                                : compute(indexed_bc_input, indexed_input));
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                cur = make_stmt<for_loop_node_t>(iter_vars.at(i), expr(0),
                        expr(floor), expr(int(vx_info.lanes)), bld->pop_scope(),
                        true, for_type::NORMAL);
                tcur.emplace_back(cur);
            }
            if (tail) {
                auto res_it = std::find(
                        bc_axis.begin(), bc_axis.end(), vx_info.axis);
                if (res_it != bc_axis.end()) {
                    in_bc_idx_tail[keep_dims ? vx_info.axis
                                             : (res_it - bc_axis.begin())]
                            = tail_var;
                }
                expr indexed_bc_input_tail = builder::make_indexing(
                        in_bc_tsl->tptr_, in_bc_idx_tail);
                if (bc_input_cast) {
                    indexed_bc_input_tail = builder::make_cast(
                            sc_data_type_t(out_etype,
                                    indexed_bc_input_tail->dtype_.lanes_),
                            indexed_bc_input_tail);
                }
                bld->push_scope();
                cur = make_stmt<assign_node_t>(indexed_target_tail,
                        bc_input_idx == 1 ? compute(
                                indexed_input_tail, indexed_bc_input_tail)
                                          : compute(indexed_bc_input_tail,
                                                  indexed_input_tail));
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                cur = make_stmt<for_loop_node_t>(tail_var, expr(floor),
                        expr(floor + tail), expr(1), bld->pop_scope(), true,
                        for_type::NORMAL);
                tcur.emplace_back(cur);
            }
        } else {
            if (!tcur.empty() && tcur[0].defined()) {
                body = make_stmt<stmts_node_t>(std::move(tcur));
                tcur.clear();
                // address special condition, like temp_buffer is used
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0), dst.get_shape().at(i), expr(1),
                        std::move(body), true, for_type::NORMAL);
            } else if (cur.defined()) {
                body = make_stmt<stmts_node_t>(
                        std::vector<stmt> {std::move(cur)});
                // address special condition, like temp_buffer is used
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0), dst.get_shape().at(i), expr(1),
                        std::move(body), true, for_type::NORMAL);
            } else {
                // if cur not defined, means last axis of tensor slice has range
                // 1, e.g. tensor_slice{{i, 100},{0, 1}}
                indexed_target = builder::make_indexing(dst.tptr_, dst_idx);

                indexed_input = builder::make_indexing(in_tsl->tptr_, in_idx);

                expr indexed_bc_input
                        = builder::make_indexing(in_bc_tsl->tptr_, in_bc_idx);
                if (bc_input_cast) {
                    indexed_bc_input = builder::make_cast(
                            sc_data_type_t(
                                    out_etype, indexed_bc_input->dtype_.lanes_),
                            indexed_bc_input);
                }
                bld->push_scope();
                cur = make_stmt<assign_node_t>(indexed_target,
                        bc_input_idx == 1
                                ? compute(indexed_input, indexed_bc_input)
                                : compute(indexed_bc_input, indexed_input));
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                cur = make_stmt<for_loop_node_t>(iter_vars.at(i), expr(0),
                        dst.get_shape().at(i), expr(1), bld->pop_scope(), true,
                        for_type::NORMAL);
            }
        }
    }
    if (!tcur.empty() && tcur[0].defined()) {
        assert(dst.get_shape().size() == 1UL);
        // TODO(xxx): currenly we don't add merge_loop attribute for this
        // special case, need stronger loop analysis.
        for (auto &it : tcur) {
            bld->emit(it);
        }
    } else {
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    }
}

void compute_block_concat(const std::vector<const tensor_slice *> &src,
        const tensor_slice &dst, int dim, size_t wkld = 0UL) {
    // outer nested loop vars
    std::vector<expr> outer_iter(dim);
    // inner nested loop vars
    std::vector<std::vector<expr>> inner_iter(dst.nslice_dims() - dim);
    // the indices for multiple inputs. First dim: the input, Second dim:
    // the dimemsions in the tensor
    std::vector<std::vector<expr>> src_idx(src.size());
    // the indices for the output tensor. Cause concat is a assign op, we
    // need number of src indexes.
    std::vector<std::vector<expr>> dst_idx(src.size());
    for (unsigned i = 0; i < dst.nslice_dims(); i++) {
        if (i < static_cast<unsigned>(dim)) { // outer loop
            // make the loop var for the for-loop
            outer_iter[i] = builder::make_var(datatypes::index,
                    std::string("_fuseiter") + std::to_string(idx++));
            for (unsigned j = 0; j < src.size(); j++) {
                src_idx[j].emplace_back(outer_iter[i]);
                dst_idx[j].emplace_back(outer_iter[i]);
            }
        } else { // inner loop
            expr cur = 0;
            for (unsigned j = 0; j < src.size(); j++) {
                inner_iter[i - dim].emplace_back(builder::make_var(
                        datatypes::index,
                        std::string("_fuseiter") + std::to_string(idx++)));
                src_idx[j].emplace_back(inner_iter[i - dim][j]);
                if (static_cast<int>(i) == dim) {
                    if (j > 0) { cur = cur + src[j - 1]->get_shape()[i]; }
                    dst_idx[j].emplace_back(inner_iter[i - dim][j] + cur);
                } else {
                    dst_idx[j].emplace_back(inner_iter[i - dim][j]);
                }
            }
        }
    }
    expr indexed_target;
    expr indexed_input;
    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder is set");
    std::vector<stmt> tcur;
    for (unsigned j = 0; j < src.size(); j++) {
        indexed_target = builder::make_indexing(dst.tptr_, dst_idx[j]);
        indexed_input = builder::make_indexing(src[j]->tptr_, src_idx[j]);
        stmt cur = make_stmt<assign_node_t>(indexed_target, indexed_input);
        cur->attr()[op_traits::workload_computable_t::workload_number] = wkld;
        for (int64_t i = static_cast<int64_t>(dst.nslice_dims()) - 1; i >= dim;
                i--) {
            auto body = make_stmt<stmts_node_t>(
                    std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(inner_iter[i - dim][j], expr(0),
                    src[j]->get_shape()[i], expr(1), std::move(body), true,
                    for_type::NORMAL);
        }
        tcur.emplace_back(std::move(cur));
    }
    if (dim) {
        stmt cur = make_stmt<stmts_node_t>(std::move(tcur));
        for (int i = dim - 1; i >= 0; i--) {
            stmt body;
            if (cur.isa<for_loop>()) {
                body = make_stmt<stmts_node_t>(
                        std::vector<stmt> {std::move(cur)});
            } else {
                body = cur;
            }
            cur = make_stmt<for_loop_node_t>(outer_iter[i], expr(0),
                    src[0]->get_shape()[i], expr(1), std::move(body), true,
                    for_type::NORMAL);
        }
        bld->emit(cur);
    } else {
        for (auto &cur : tcur) {
            bld->emit(cur);
        }
    }
}

void check_concat_validity(
        const std::vector<graph_tensor_ptr> &candidates, unsigned concat_dim) {
    COMPILE_ASSERT(candidates.size() > 1,
            "Number of candidates for concat op must be larger than 1!\n");
    auto firstShape = candidates[0]->details_.get_blocking_dims();
    COMPILE_ASSERT(firstShape.size(),
            "First candidate of concat op has empty dimensions!\n");

    for (unsigned i = 1; i < candidates.size(); i++) {
        auto curShape = candidates[i]->details_.get_blocking_dims();
        if (curShape.size() != firstShape.size()) {
            COMPILE_ASSERT(
                    0, "Input shapes are not matched in concat fusion op!\n");
        }
        for (unsigned dim = 0; dim < firstShape.size(); dim++) {
            if (concat_dim == dim && curShape[dim]) { continue; }
            COMPILE_ASSERT(curShape[dim] == firstShape[dim],
                    "Input shapes: "
                            << utils::print_vector(curShape) << " and "
                            << utils::print_vector(firstShape)
                            << " are not matched in concat fusion op!\n");
        }
    }
}

input_op::input_op(const sc_dims &dims, sc_data_type_t dtype) {
    op_name_ = "input";
    info_.outputs_.emplace_back(std::make_shared<graph_tensor>(
            this, sc_data_format_t(), dims, dtype));
}

input_op::input_op(const logical_tensor_t &lt) {
    op_name_ = "input";
    info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this, lt));
}

input_op::input_op(const std::vector<graph_tensor_ptr> &outs) {
    info_.outputs_ = outs;
    for (auto &out : info_.outputs_) {
        out->producer_owner_ = this;
    }
    op_name_ = "input";
}

size_t get_dims_product(const sc_dims &dims) {
    sc_dim ret = 1;
    for (unsigned i = 0; i < dims.size(); ++i) {
        ret *= dims[i];
    }
    assert(ret > 0 && "Overflow or non-constant shape detected");
    return ret;
}

std::vector<int> binary_elementwise_op_t::infer_broadcast_axis() const {
    int bc_input_idx = get_broadcast_input();
    if (bc_input_idx == -1) return {};

    sc_dims lhs_dims, rhs_dims;
    lhs_dims = get_inputs()[0]->details_.get_plain_dims();
    rhs_dims = get_inputs()[1]->details_.get_plain_dims();

    sc_dims elt_dims, bc_dims;
    if (bc_input_idx == 1) {
        elt_dims = lhs_dims;
        bc_dims = rhs_dims;
    } else {
        elt_dims = rhs_dims;
        bc_dims = lhs_dims;
    }
    if (bc_dims.size() == 1 && bc_dims[0] == 1) {
        return std::vector<int> {-1};
    }
    std::vector<int> bc_axis;
    // broad-cast conditions 1: the shape of lhs and rhs not match
    if (elt_dims.size() != bc_dims.size()) {
        std::vector<int> common_axes(elt_dims.size(), 0);
        // from right to left
        int64_t i = elt_dims.size() - 1;
        for (int64_t j = bc_dims.size() - 1; j >= 0; j--) {
            for (; i >= 0; i--) {
                if (elt_dims.at(i) == bc_dims.at(j)) {
                    common_axes.at(i) = 1;
                    break;
                }
            }
            if (i == -1) {
                COMPILE_ASSERT(0,
                        "illegal elementwise operand found. "
                                << utils::print_vector(elt_dims) << " , "
                                << utils::print_vector(bc_dims));
            }
        }
        for (size_t j = 0; j < common_axes.size(); ++j)
            if (common_axes.at(j) == 1) bc_axis.emplace_back(j);
    }
    // broad-cast conditions 2: the shape of lhs and rhs match,
    // but length=1 in dims
    else {
        bool double_check_broadcast = false;
        for (size_t i = 0; i < elt_dims.size(); ++i) {
            if (elt_dims.at(i) != bc_dims.at(i)) {
                if (bc_dims.at(i) == 1) {
                    double_check_broadcast = true;
                } else {
                    COMPILE_ASSERT(0,
                            "illegal elementwise operand found: "
                                    << utils::print_vector(elt_dims) << " , "
                                    << utils::print_vector(bc_dims));
                }
            }
        }
        if (double_check_broadcast) {
            for (size_t i = 0; i < elt_dims.size(); ++i) {
                if (elt_dims.at(i) == bc_dims.at(i)) {
                    bc_axis.emplace_back(i);
                }
            }
            if (bc_axis.empty()) { bc_axis.emplace_back(-1); }
        } else
            bc_axis = {};
    }
    return bc_axis;
}

slice_range_map search_known_slice_ranges(
        fusible_op_t *cur, fslice_map &fsmap) {
    slice_range_map known_ranges_map;
    auto input_size = cur->get_inputs().size();
    COMPILE_ASSERT(input_size > 0,
            "We could not infer slice ranges for op without input.");
    for (size_t i = 0; i < input_size; i++) {
        auto &input = cur->get_inputs()[i];
        if (!fsmap.get(input).empty()) {
            known_ranges_map[i] = fsmap.get(input);
        }
    }
    COMPILE_ASSERT(!known_ranges_map.empty(),
            "No original slice of inputs can be searched for op: "
                    << cur->op_name_ << "\n");
    return known_ranges_map;
}

void set_unknown_slice_ranges(fusible_op_t *cur,
        slice_range_map known_ranges_map, fslice_map &fsmap,
        infer_status_map_t &stat_map) {
    // set other unknown ranges.
    auto input_size = cur->get_inputs().size();
    for (size_t i = 0; i < input_size; i++) {
        auto input = cur->get_inputs()[i];
        auto &inp_slice = fsmap.get(input);
        if (input->producer_owner_->isa<input_op>()
                && input->producer_owner_->dyn_cast<input_op>()
                           ->is_arg_input()) {
            inp_slice = known_ranges_map[i];
        } else {
            if (inp_slice.empty()) {
                inp_slice = known_ranges_map[i];
                input->producer_owner_->dyn_cast<fusible_op_t>()
                        ->pre_slice_ranges(fsmap, stat_map);
            }
        }
    }
}

void infer_unary_slice_ranges(fusible_op_t *cur, fslice_map &fsmap) {
    COMPILE_ASSERT(cur->get_inputs().size() == 1, "unary op is expected");
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map = search_known_slice_ranges(cur, fsmap);
    // set outputs slice range
    fsmap.get(cur->get_outputs()[0]) = known_ranges_map[0];
}

void infer_binary_slice_ranges(
        fusible_op_t *cur, fslice_map &fsmap, infer_status_map_t &stat_map) {
    COMPILE_ASSERT(cur->get_inputs().size() == 2, "binary op is expected");
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map = search_known_slice_ranges(cur, fsmap);
    auto &outslice = fsmap.get(cur->get_outputs()[0]);
    // if unkown slice ranges exist.
    if (known_ranges_map.size() < cur->get_inputs().size()) {
        int unknown_idx
                = known_ranges_map.find(0) != known_ranges_map.end() ? 1 : 0;
        known_ranges_map[unknown_idx] = known_ranges_map[1 - unknown_idx];
        // set the other unknown slice range by achieved known_ranges_list
        set_unknown_slice_ranges(cur, known_ranges_map, fsmap, stat_map);
    }
    // set outputs slice range
    outslice = known_ranges_map[0];
}

static slice_range_list infer_broadcast_arg_slice(
        slice_range_list known_range_list, std::vector<int> bc_axis,
        bool keep_dims) {
    slice_range_list bc_arg_range_list(known_range_list.size());
    for (size_t i = 0; i < bc_arg_range_list.size(); i++) {
        auto &known_range = known_range_list[i];
        for (size_t j = 0; j < known_range.size(); j++) {
            if (bc_axis.end() != std::find(bc_axis.begin(), bc_axis.end(), j)) {
                bc_arg_range_list[i].emplace_back(known_range.at(j));
            } else {
                if (keep_dims) {
                    bc_arg_range_list[i].emplace_back(
                            std::make_pair(expr(0), expr(1)));
                }
            }
        }
        if (bc_arg_range_list[i].empty())
            bc_arg_range_list[i].emplace_back(std::make_pair(0, 1));
    }
    return bc_arg_range_list;
}

static slice_range_list infer_broadcast_slice(slice_range_list known_range_list,
        std::vector<int> bc_axis, sc_dims bc_dim) {
    slice_range_list bc_range_list(known_range_list.size());
    for (size_t i = 0; i < bc_range_list.size(); i++) {
        auto &known_range = known_range_list[i];
        COMPILE_ASSERT(
                known_range.size() == bc_dim.size(), "Unexpected cases found")
        for (size_t j = 0; j < known_range.size(); j++) {
            if (bc_axis.end() != std::find(bc_axis.begin(), bc_axis.end(), j)) {
                bc_range_list[i].emplace_back(known_range.at(j));
            } else {
                bc_range_list[i].emplace_back(
                        std::make_pair(expr(0), dim2unsigned(bc_dim[j])));
            }
        }
    }
    return bc_range_list;
}

void pre_unary_slice_ranges(
        fusible_op_t *cur, fslice_map &fsmap, infer_status_map_t &stat_map) {
    auto &input = cur->get_inputs()[0];
    auto &out_ranges = fsmap.get(cur->get_outputs()[0]);
    auto &in_ranges = fsmap.get(input);
    if (in_ranges.empty()) {
        in_ranges = out_ranges;
        input->producer_owner_->dyn_cast<fusible_op_t>()->pre_slice_ranges(
                fsmap, stat_map);
    }
}

void input_op::prepare_fusion_data(fdata_map &fdmap) {
    COMPILE_ASSERT(info_.outputs_.size() == 1, "Wrong op output size.\n");
}

output_op::output_op(const graph_tensor_ptr &v) {
    info_.inputs_.emplace_back(v);
    op_name_ = "output";
}

output_op::output_op(const std::vector<graph_tensor_ptr> &in) {
    info_.inputs_ = in;
    op_name_ = "output";
}

void output_op::prepare_fusion_data(fdata_map &fdmap) {
    assert(info_.outputs_.empty() && "Wrong op output size.\n");
    auto &inputs = info_.inputs_[0];
    auto &outdetail = fdmap.get(inputs);
    outdetail.need_alloc_ = false;
}

binary_elementwise_op_t::binary_elementwise_op_t(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    // TODO(xxx): do not cache vectorized_ or inplace_
    assert(ins.size() == 2);
    info_.inputs_ = ins;
    if (outs.empty()) {
        // fixme: correctly infer the shape for broadcast
        auto lhs_const = dynamic_cast<constant_op_t *>(
                info_.inputs_.at(0)->producer_owner_);
        auto rhs_const = dynamic_cast<constant_op_t *>(
                info_.inputs_.at(1)->producer_owner_);
        if (!lhs_const && rhs_const) {
            info_.outputs_.emplace_back(
                    std::make_shared<graph_tensor>(this, ins[0]->details_));
        } else if (lhs_const && !rhs_const) {
            info_.outputs_.emplace_back(
                    std::make_shared<graph_tensor>(this, ins[1]->details_));
        } else {
            int bc_input_idx = get_broadcast_input();
            int ref_idx = bc_input_idx < 0 ? 0 : 1 - bc_input_idx;
            info_.outputs_.emplace_back(std::make_shared<graph_tensor>(
                    this, ins[ref_idx]->details_));
        }
    } else {
        info_.outputs_ = outs;
    }

    int bc_idx = get_broadcast_input();
    int non_bc_idx = bc_idx < 0 ? 0 : 1 - bc_idx;

    info_.outputs_[0]->details_.dtype_
            = info_.inputs_[non_bc_idx]->details_.dtype_;

    attrs_ = attrs;
    plain_bc_axis_ = attrs.get_or_else("bc_axis", std::vector<int> {});

    if (plain_bc_axis_.empty()) { plain_bc_axis_ = infer_broadcast_axis(); }

    inplace_ = attrs.get_or_else("inplace", 1);
}

binary_elementwise_op_t::binary_elementwise_op_t(graph_tensor_ptr lhs,
        graph_tensor_ptr rhs, elt_operator elt_op, int inplace)
    : binary_elementwise_op_t(
            {std::move(lhs), std::move(rhs)}, {}, {{"inplace", inplace}}) {
    elt_op_ = elt_op;
    switch (elt_op) {
        case elt_operator::ADD: op_name_ = "add"; break;
        case elt_operator::SUB: op_name_ = "sub"; break;
        case elt_operator::MUL: op_name_ = "mul"; break;
        case elt_operator::DIV: op_name_ = "div"; break;
        case elt_operator::MIN: op_name_ = "min"; break;
        case elt_operator::MAX: op_name_ = "max"; break;
        case elt_operator::SQD_DIFF: op_name_ = "sqd_diff"; break;
        default: break;
    }
}

int binary_elementwise_op_t::get_broadcast_input() const {
    const sc_dims &lhs_dims = info_.inputs_[0]->details_.get_plain_dims();
    const sc_dims &rhs_dims = info_.inputs_[1]->details_.get_plain_dims();
    if (lhs_dims == rhs_dims) {
        return -1;
    } else {
        auto lhs_dp = get_dims_product(lhs_dims);
        auto rhs_dp = get_dims_product(rhs_dims);
        if (lhs_dp == rhs_dp) {
            COMPILE_ASSERT(lhs_dims.size() != rhs_dims.size(),
                    "Unexpected dims of bianry elementwise inputs are found: "
                            << utils::print_vector(lhs_dims) << " and "
                            << utils::print_vector(rhs_dims))
            return lhs_dims.size() > rhs_dims.size() ? 1 : 0;
        } else {
            return lhs_dp > rhs_dp ? 1 : 0;
        }
    }
}

/**
 * @param a: the input's 0th index dims.
 * @param b: the input's 1th index dims.
 * @param bc_axis: default b(input's 1th index)need to do broadcast.
 * @return common_axis: record all common axis pair {{a axis, b axis}, ...}
 * for plain dims*/
static std::vector<std::array<int, 2>> get_common_axis(const sc_dims &a,
        const sc_dims &b, const std::vector<int> &bc_axis = {}) {
    std::vector<std::array<int, 2>> common_axis;
    if (!bc_axis.empty() && bc_axis != std::vector<int> {-1}
            && a.size() != b.size()) {
        COMPILE_ASSERT(
                bc_axis.size() == b.size(), "Unexpected bc axis size found")
        for (int i = 0; i < (int)b.size(); ++i) {
            if (b[i] == a[bc_axis[i]]) {
                common_axis.push_back({bc_axis[i], i});
            }
        }
    } else {
        int i = a.size() - 1, j = b.size() - 1;
        while (i >= 0 && j >= 0) {
            if (a[i] == b[j]) {
                common_axis.push_back({i, j});
            } else {
                if ((a[i] == 1 || b[j] == 1)) {
                    common_axis.push_back({i, j});
                } else {
                    COMPILE_ASSERT(0,
                            "No common axis: "
                                    << i << " th axis doesn't have same value: "
                                    << a[i] << " and " << b[j]
                                    << " expected having 1 value");
                }
            }
            --i;
            --j;
        }
    }
    return common_axis;
}

static sc_data_format_t infer_blocking_format(
        const logical_tensor_t &blocking_lt, const logical_tensor_t &plain_lt,
        int blocking_in_index, int plain_in_index,
        const std::vector<std::array<int, 2>> &common_axis) {
    sc_data_format_t::blocking_t blocks;
    blocks.fill(0);
    std::vector<int> storage_args(plain_lt.get_plain_dims().size());
    std::iota(storage_args.begin(), storage_args.end(), 0);
    auto blocked_axis = blocking_lt.get_format().get_blocked_axis();

    size_t pos = 0;
    for (int i = common_axis.size() - 1; i >= 0; --i) {
        // for {1,..}
        int bs = plain_lt.get_format().format_code_.is_batch_format()
                ? plain_lt.get_plain_dims().size() - 2
                : 0;
        if (plain_lt.get_plain_dims()[common_axis[i][plain_in_index] + bs] == 1
                && blocked_axis.find(common_axis[i][blocking_in_index])
                        != blocked_axis.end()) {
            // consider padding condition
            if (get_dims_product(blocking_lt.get_plain_dims())
                    != get_dims_product(blocking_lt.get_blocking_dims())) {
                for (auto block :
                        blocked_axis[common_axis[i][blocking_in_index]]) {
                    storage_args.push_back(common_axis[i][plain_in_index]);
                    blocks[pos++] = block;
                }
                continue;
            }
            storage_args.push_back(common_axis[i][plain_in_index]);
            blocks[pos++] = 1;
        } else {
            if (blocked_axis.find(common_axis[i][blocking_in_index])
                    != blocked_axis.end()) {
                for (auto block :
                        blocked_axis[common_axis[i][blocking_in_index]]) {
                    storage_args.push_back(common_axis[i][plain_in_index]);
                    blocks[pos++] = block;
                }
            }
        }
    }
    // temporary fix for MHA case
    // TODO(xxx): add an extension pass to align lhs and rhs dimension for
    // broadcast/binary elementwise op
    if (plain_lt.get_plain_dims().size()
            == blocking_lt.get_plain_dims().size()) {
        return sc_data_format_t(blocking_lt.get_format().format_code_, blocks);
    }
    return sc_data_format_t(
            plain_lt.get_format().format_code_.is_batch_format(), storage_args,
            blocks);
}

void binary_elementwise_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {
    auto in0_format = info_.inputs_[0]->details_.get_format();
    auto in1_format = info_.inputs_[1]->details_.get_format();
    int bc_input_idx = get_broadcast_input();
    std::vector<std::array<int, 2>> common_axis = get_common_axis(
            info_.inputs_[bc_input_idx == -1 ? 0 : (1 - bc_input_idx)]
                    ->details_.get_plain_dims(),
            info_.inputs_[bc_input_idx == -1 ? 1 : bc_input_idx]
                    ->details_.get_plain_dims(),
            plain_bc_axis_);
    // swap common axis
    if (bc_input_idx == 0) {
        std::for_each(common_axis.begin(), common_axis.end(),
                [](std::array<int, 2> &arr) { std::swap(arr[0], arr[1]); });
    }

    COMPILE_ASSERT(!common_axis.empty(),
            "binary elementwise op doesn't support two shape : "
                    << utils::print_vector(
                               info_.inputs_[0]->details_.get_plain_dims())
                    << ", "
                    << utils::print_vector(
                               info_.inputs_[1]->details_.get_plain_dims())
                    << "to query format, consider to use broadcast op");

    if (in0_format.get_format_category() != in1_format.get_format_category()) {
        // plain+block combination.
        if (in0_format.is_blocking()) {
            in_formats.push_back({in0_format});
            // for {1} shape
            if (info_.inputs_[1]->details_.get_plain_dims().size() == 1
                    && info_.inputs_[1]->details_.get_plain_dims()[0] == 1
                    && plain_bc_axis_ == std::vector<int> {-1}) {
                in_formats.push_back({in1_format});
            } else {
                in_formats.push_back({infer_blocking_format(
                        info_.inputs_[0]->details_, info_.inputs_[1]->details_,
                        0, 1, common_axis)});
            }
            out_formats.push_back({in0_format});
        } else {
            // for {1} shape
            if (info_.inputs_[0]->details_.get_plain_dims().size() == 1
                    && info_.inputs_[0]->details_.get_plain_dims()[0] == 1
                    && plain_bc_axis_ == std::vector<int> {-1}) {
                in_formats.push_back({in0_format});
            } else {
                in_formats.push_back({infer_blocking_format(
                        info_.inputs_[1]->details_, info_.inputs_[0]->details_,
                        1, 0, common_axis)});
            }
            in_formats.push_back({in1_format});
            out_formats.push_back({in1_format});
        }
    } else {
        // plain+plain
        if ((in0_format.is_plain() && in1_format.is_plain())) {
            in_formats.push_back({in0_format});
            in_formats.push_back({in1_format});
            get_dims_product(info_.inputs_[0]->details_.get_plain_dims())
                            >= get_dims_product(
                                    info_.inputs_[1]->details_.get_plain_dims())
                    ? out_formats.push_back({in0_format})
                    : out_formats.push_back({in1_format});
        } else if (info_.inputs_[0]->details_.get_plain_dims().size()
                == info_.inputs_[1]->details_.get_plain_dims().size()) {
            auto in0_blocked_axis = in0_format.get_blocked_axis();
            auto in1_blocked_axis = in1_format.get_blocked_axis();
            int index = get_dims_product(
                                info_.inputs_[0]->details_.get_plain_dims())
                            >= get_dims_product(
                                    info_.inputs_[1]->details_.get_plain_dims())
                    ? 0
                    : 1;

            auto base_format_code
                    = (index ? in1_format : in0_format).format_code_;
            auto base_blocked_axis
                    = (index ? in1_blocked_axis : in0_blocked_axis);
            size_t base_dims = base_format_code.norig_dims();
            size_t max_dims = base_format_code.ndims();
            sc_data_format_t::blocking_t blocks
                    = (index ? in1_format : in0_format).blocks_;
            for (size_t i = base_dims; i < max_dims; ++i) {
                int axis = base_format_code.get(i);
                if (info_.inputs_[0]->details_.get_plain_dims()[axis]
                        != info_.inputs_[1]->details_.get_plain_dims()[axis]) {
                    blocks[i - base_dims] = 1;
                }
            }
            // blocks = {1, 64, 0, 0};
            auto mixed_format = sc_data_format_t(base_format_code, blocks);
            index ? in_formats.push_back({mixed_format})
                  : in_formats.push_back({in0_format});
            index ? in_formats.push_back({in1_format})
                  : in_formats.push_back({mixed_format});
            index ? out_formats.push_back({in1_format})
                  : out_formats.push_back({in0_format});
        } else {
            // block+ block
            auto in0_blocked_axis = in0_format.get_blocked_axis();
            auto in1_blocked_axis = in1_format.get_blocked_axis();
            int index = in0_blocked_axis.size() >= in1_blocked_axis.size() ? 0
                                                                           : 1;
            bool same_blocked = true;
            for (int i = common_axis.size() - 1; i >= 0; --i) {
                if (info_.inputs_[0]->details_
                                        .get_plain_dims()[common_axis[i][0]]
                                == 1
                        || info_.inputs_[1]
                                        ->details_
                                        .get_plain_dims()[common_axis[i][1]]
                                == 1)
                    continue;
                if (in0_blocked_axis.find(common_axis[i][0])
                                != in0_blocked_axis.end()
                        && in1_blocked_axis.find(common_axis[i][1])
                                != in1_blocked_axis.end()) {
                    if (in0_blocked_axis[common_axis[i][0]]
                            != in1_blocked_axis[common_axis[i][1]]) {
                        same_blocked = false;
                    }
                }
            }

            if (same_blocked) {
                in_formats.push_back({in0_format});
                in_formats.push_back({in1_format});
                index ? out_formats.push_back({in1_format})
                      : out_formats.push_back({in0_format});
            } else {
                index ? in_formats.push_back({in1_format})
                      : in_formats.push_back({in0_format});
                index ? in_formats.push_back({in1_format})
                      : in_formats.push_back({in0_format});
                index ? out_formats.push_back({in1_format})
                      : out_formats.push_back({in0_format});
            }
        }
    }
}

void binary_elementwise_op_t::prepare_fusion_data(fdata_map &fdmap) {
    COMPILE_ASSERT(!op_name_.empty(), "op_name or elt_operator is not set.\n");
    COMPILE_ASSERT(info_.outputs_.size() == 1, "Wrong op output size.\n");
    auto &output = info_.outputs_[0];
    auto &outdetail = fdmap.get(output);
    auto &in_detail0 = fdmap.get(info_.inputs_[0]);
    auto &in_detail1 = fdmap.get(info_.inputs_[1]);

    in_detail0.use_count_++;
    in_detail1.use_count_++;
    auto lhs_const = dynamic_cast<constant_op_t *>(
            info_.inputs_.at(0)->producer_owner_);
    auto rhs_const = dynamic_cast<constant_op_t *>(
            info_.inputs_.at(1)->producer_owner_);
    // no inplace, need to create a new buffer
    if (inplace_ == 0) {
        info_.tensor_share_info_ = {};
    }
    // inplace 1-th input
    else if (inplace_ == 1) {
        if (lhs_const
                || (info_.inputs_[0]->details_.get_blocking_dims()
                        != output->details_.get_blocking_dims())) {
            info_.tensor_share_info_ = {};
        } else {
            info_.tensor_share_info_ = {{0, {0}}};
        }
    }
    // inplace 2-th input
    else if (inplace_ == 2) {
        if (rhs_const
                || (info_.inputs_[1]->details_.get_blocking_dims()
                        != output->details_.get_blocking_dims())) {
            info_.tensor_share_info_ = {};
        } else {
            info_.tensor_share_info_ = {{0, {1}}};
        }
    } else {
        COMPILE_ASSERT(0,
                "binary op only have two inputs, but got "
                        << inplace_ << "-th input to be inplaced.");
    }
}

// The logic below might be suitable for most fusible op, which has same
// slice ranges on inputs and outputs
void binary_elementwise_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    COMPILE_ASSERT(get_inputs().size() == 2, "binary op is expected");
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map = search_known_slice_ranges(this, fsmap);
    auto &outslice = fsmap.get(get_outputs()[0]);
    // if unkown slice ranges exist.
    if (known_ranges_map.size() < get_inputs().size()) {
        int unknown_idx
                = known_ranges_map.find(0) != known_ranges_map.end() ? 1 : 0;
        // check broadcast
        int bc_input_idx = get_broadcast_input();
        if (bc_input_idx >= 0) {
            bool keep_dims = get_inputs()[bc_input_idx]
                                     ->details_.get_blocking_dims()
                                     .size()
                    == get_inputs()[1 - bc_input_idx]
                               ->details_.get_blocking_dims()
                               .size();
            auto bc_axis = get_bc_axis();
            if (unknown_idx != bc_input_idx) {
                slice_range_list bc_range_list = infer_broadcast_slice(
                        known_ranges_map[1 - unknown_idx], bc_axis,
                        get_inputs()[1 - bc_input_idx]
                                ->details_.get_blocking_dims());
                known_ranges_map[unknown_idx] = bc_range_list;
            } else {
                slice_range_list bc_arg_range_list = infer_broadcast_arg_slice(
                        known_ranges_map[1 - unknown_idx], bc_axis, keep_dims);
                known_ranges_map[unknown_idx] = bc_arg_range_list;
            }
            // set the other unknown slice range by achieved
            // known_ranges_list
            set_unknown_slice_ranges(this, known_ranges_map, fsmap, stat_map);
            // set outputs slice range
            outslice = known_ranges_map[1 - bc_input_idx];
            return;
        } else {
            known_ranges_map[unknown_idx] = known_ranges_map[1 - unknown_idx];
        }
        // set the other unknown slice range by achieved known_ranges_list
        set_unknown_slice_ranges(this, known_ranges_map, fsmap, stat_map);
    }
    // set outputs slice range
    outslice = known_ranges_map[inplace_ == 2 ? 1 : 0];
}

void binary_elementwise_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    auto &outslice = fsmap.get(get_outputs()[0]);
    // check broadcast
    int bc_input_idx = get_broadcast_input();
    for (size_t i = 0; i < get_inputs().size(); i++) {
        auto &input = get_inputs()[i];
        auto &inpslice = fsmap.get(input);
        if (inpslice.empty()) {
            if (bc_input_idx == static_cast<int>(i)) {
                auto bc_axis = get_bc_axis();
                inpslice = infer_broadcast_arg_slice(outslice, bc_axis,
                        get_inputs()[bc_input_idx]
                                        ->details_.get_blocking_dims()
                                        .size()
                                == get_inputs()[1 - bc_input_idx]
                                           ->details_.get_blocking_dims()
                                           .size());
            } else {
                inpslice = outslice;
            }
            input->producer_owner_->dyn_cast<fusible_op_t>()->pre_slice_ranges(
                    fsmap, stat_map);
        }
    }
}

std::vector<int> transform_axis_plain2blocking(sc_data_format_t fmt,
        const std::vector<int> &plain_axis, int bs_offset = 0) {
    // If format is any, just return.
    if (fmt.is_any()) { return plain_axis; }
    std::vector<int> real_axis;
    auto p2bmp = fmt.format_code_.collect_p2b_mapping();
    for (auto &i : plain_axis) {
        if (i < bs_offset) {
            real_axis.emplace_back(i);
        } else {
            std::vector<int> res;
            res.resize(p2bmp[i - bs_offset].size());
            std::transform(p2bmp[i - bs_offset].begin(),
                    p2bmp[i - bs_offset].end(), res.begin(),
                    [&bs_offset](const int &v) { return v + bs_offset; });
            real_axis.insert(real_axis.end(), res.begin(), res.end());
        }
    }
    std::sort(real_axis.begin(), real_axis.end());
    return real_axis;
}

std::vector<int> binary_elementwise_op_t::get_bc_axis() const {
    int bc_input_idx = get_broadcast_input();
    if (bc_input_idx == -1) return {};
    if (plain_bc_axis_ == std::vector<int> {-1}) return plain_bc_axis_;
    auto fmt = info_.inputs_[1 - bc_input_idx]->details_.get_format();
    int bs_ndim = 0;
    if (fmt.format_code_.is_batch_format()) {
        bs_ndim = static_cast<int>(info_.inputs_[1 - bc_input_idx]
                                           ->details_.get_blocking_dims()
                                           .size())
                - fmt.format_code_.ndims();
    }
    return transform_axis_plain2blocking(fmt, plain_bc_axis_, bs_ndim);
}

bool binary_elementwise_op_t::register_brgemm_fusion(const context_ptr &ctx,
        const std::vector<tensor_slice *> &outputs,
        const std::vector<const tensor_slice *> &inputs,
        brgemm_fusion_register &brg_reg) {
    if (!fuse_in_brgemm_) { return false; }
    int bc_input_idx = get_broadcast_input();
    // input 0 broadcast, can not be processed in brgemm
    if (bc_input_idx == 0) { return false; }
    return brg_reg.register_op_infos(shared_from_this(),
            outputs[0]->get_tensor_ptr(), inputs[1]->get_tensor_ptr(),
            inputs[1]->get_shape());
}

void binary_elementwise_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);
    // set default vectorized information
    vx_info_.axis = dst[0]->get_shape().size() - 1;

    for (int64_t i = dst[0]->nslice_dims() - 1; i >= 0; --i) {
        int cur_dim = get_const_as_int(
                dst[0]->get_shape()[i].checked_as<constant>());
        if (1 != cur_dim) {
            vx_info_.axis = i;
            break;
        }
    }
    vx_info_.lanes
            = vectorize_step(ctx, info_.inputs_[0]->details_.dtype_.type_code_);

    // use broad-cast
    int bc_input_idx = get_broadcast_input();
    if (bc_input_idx != -1) {
        // reuse broadcast op
        compute_block_broadcast(
                inputs, *dst[0], bc_input_idx, get_bc_axis(), vx_info_,
                [&](const expr &in_0, const expr &in_1) -> expr {
                    switch (elt_op_) {
                        case elt_operator::ADD: return (in_0 + in_1);
                        case elt_operator::SUB: return (in_0 - in_1);
                        case elt_operator::MUL: return (in_0 * in_1);
                        case elt_operator::DIV: return (in_0 / in_1);
                        case elt_operator::MIN:
                            return builder::make_min(in_0, in_1);
                        case elt_operator::MAX:
                            return builder::make_max(in_0, in_1);
                        case elt_operator::SQD_DIFF:
                            return (in_0 - in_1) * (in_0 - in_1);
                        default:
                            COMPILE_ASSERT(
                                    false, "Unsupport elementwise op found.\n");
                            return expr();
                    }
                },
                info_.outputs_[0]->details_.dtype_, wkld);
    } else {
        auto func = [&](const std::vector<expr> &in,
                            std::vector<expr::lvalue_proxy_t> &out,
                            int mask_count, float mask_value) -> stmt {
            auto out_dtype = out[0]->dtype_;
            expr in0 = in[0], in1 = in[1];
            if (in[0]->dtype_ != out_dtype) {
                in0 = builder::make_cast(out_dtype, in[0]);
            }
            if (in[1]->dtype_ != out_dtype) {
                in1 = builder::make_cast(out_dtype, in[1]);
            }
            switch (elt_op_) {
                case elt_operator::ADD:
                    return builder::make_assign_unattached(out[0], in0 + in1);
                case elt_operator::SUB:
                    return builder::make_assign_unattached(out[0], in0 - in1);
                case elt_operator::MUL:
                    return builder::make_assign_unattached(out[0], in0 * in1);
                case elt_operator::DIV:
                    return builder::make_assign_unattached(out[0],
                            make_select_by_mask(
                                    in0 / in1, mask_count, vx_info_.lanes));
                case elt_operator::MIN:
                    return builder::make_assign_unattached(
                            out[0], builder::make_min(in0, in1));
                case elt_operator::MAX:
                    return builder::make_assign_unattached(
                            out[0], builder::make_max(in0, in1));
                case elt_operator::SQD_DIFF:
                    return builder::make_assign_unattached(
                            out[0], (in0 - in1) * (in0 - in1));
                default:
                    COMPILE_ASSERT(false,
                            "Unsupport elementwise op "
                            "found.\n");
                    return stmt();
            }
        };
        // todo: currently we only support mask for div.
        bool use_mask = elt_op_ == elt_operator::DIV;
        compute_vectorized_op(inputs, *dst[0], info_, vx_info_,
                mask_compute_func_t(func), mask_compute_func_t(func), attrs_,
                wkld, use_mask);
    }
}

// special handling for union values
bool constant_op_t::compare_contents(const sc_op *other) const {
    COMPILE_ASSERT(attrs_.has_key("values") && attrs_.has_key("dtype"),
            "expecting values and dtype in attr");
    COMPILE_ASSERT(
            other->attrs_.has_key("values") && other->attrs_.has_key("dtype"),
            "expecting values and dtype in attr");
    auto dtype = attrs_.get<sc_data_type_t>("dtype");
    if (other->attrs_.get<sc_data_type_t>("dtype") != dtype) { return false; }
    if (attrs_.has_key("format")) {
        if (!other->attrs_.has_key("format")) { return false; }
        if (other->attrs_.get<sc_data_format_t>("format")
                != attrs_.get<sc_data_format_t>("format")) {
            return false;
        }
    }
    auto &vals = attrs_.get<std::shared_ptr<static_data_t>>("values");
    auto &vals2 = other->attrs_.get<std::shared_ptr<static_data_t>>("values");
    if (vals->size_ != vals2->size_) { return false; }

    switch (get_type_category_nothrow(dtype)) {
        case CATE_FLOAT:
            for (size_t i = 0; i < vals->size_ / 4; i++) {
                if (static_cast<float *>(vals->data_)[i]
                        != static_cast<float *>(vals2->data_)[i]) {
                    return false;
                }
            }
            break;
        case CATE_INT:
        case CATE_UINT:
            for (size_t i = 0; i < vals->size_ / 4; i++) {
                if (static_cast<uint32_t *>(vals->data_)[i]
                        != static_cast<uint32_t *>(vals2->data_)[i]) {
                    return false;
                }
            }
            break;
        default:
            throw std::runtime_error("Met unexpected dtype for constant");
            break;
    }
    return true;
}

size_t constant_op_t::hash_contents() const {
    size_t seed = 0;
    COMPILE_ASSERT(attrs_.has_key("values") && attrs_.has_key("dtype"),
            "expecting values and dtype in attr");
    if (attrs_.has_key("format")) {
        hash_combine(seed, attrs_.get<sc_data_format_t>("format"));
    }
    auto &vals = attrs_.get<std::shared_ptr<static_data_t>>("values");

    for (size_t i = 0; i < vals->size_; i++) {
        hash_combine(seed, static_cast<char *>(vals->data_)[i]);
    }

    return seed;
}

constant_op_t::constant_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    COMPILE_ASSERT(ins.empty(), "No op input.\n");
    COMPILE_ASSERT(attrs.has_key("values") && attrs.has_key("dtype")
                    && attrs.has_key("plain_dims"),
            "expecting values, format and dtype in attr");
    op_name_ = "constant";
    sc_data_format_t format
            = attrs.get_or_else("format", sc_data_format_t(format_kinds::A));
    attrs_ = attrs;
    const_values_ = attrs.get<std::shared_ptr<static_data_t>>("values");
    sc_data_type_t dtype = attrs.get<sc_data_type_t>("dtype");
    sc_dims plain_dims = attrs.get<sc_dims>("plain_dims");

    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(
                this, logical_tensor_t {format, plain_dims, dtype}));
    } else {
        COMPILE_ASSERT(outs.size() == 1, "Wrong op output size.\n");
        info_.outputs_ = outs;
    }
}

// todo: support tensor expr
constant_op_t::constant_op_t(std::shared_ptr<static_data_t> v,
        sc_data_type_t dtype, const sc_dims &plain_dims,
        const sc_data_format_t &format) {
    const_values_ = std::move(v);
    info_.outputs_.emplace_back(
            std::make_shared<graph_tensor>(this, format, plain_dims, dtype));
    info_.outputs_[0]->details_.dtype_ = dtype;
    info_.outputs_[0]->details_.set_plain_dims(plain_dims);
    attrs_.set("dtype", dtype);
    attrs_.set("values", const_values_);
    attrs_.set("plain_dims", plain_dims);
    attrs_.set("format", format);
    op_name_ = "constant";
}

void constant_op_t::prepare_fusion_data(fdata_map &fdmap) {
    COMPILE_ASSERT(info_.outputs_.size() == 1, "Wrong op output size.\n");
    auto &output = info_.outputs_[0];
    auto &outdetail = fdmap.get(output);
    auto blocking_dims = get_constant_blocking_dims();
    outdetail.need_alloc_ = true;
}

unary_elementwise_op_t::unary_elementwise_op_t(
        graph_tensor_ptr v, const std::string &op_name)
    : unary_elementwise_op_t(op_name, {std::move(v)}, {}, {}) {}

unary_elementwise_op_t::unary_elementwise_op_t(const std::string &op_name,
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    COMPILE_ASSERT(ins.size() == 1, "Wrong op input size.\n");
    op_name_ = op_name;
    info_.inputs_ = ins;
    attrs_ = attrs;
    if (outs.empty()) {
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, ins[0]->details_));
    } else {
        COMPILE_ASSERT(outs.size() == 1, "Wrong op output size.\n");
        COMPILE_ASSERT(outs[0]->details_.get_blocking_dims()
                        == ins[0]->details_.get_blocking_dims(),
                "Wrong op output shapes.\n");
        info_.outputs_ = outs;
    }
    info_.tensor_share_info_ = {{0, {0}}};
    attrs_ = attrs;
}

void unary_elementwise_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);
    // set default vectorized information
    vx_info_.axis = dst[0]->get_shape().size() - 1;
    for (int64_t i = dst[0]->nslice_dims() - 1; i >= 0; --i) {
        int cur_dim = get_const_as_int(
                dst.at(0)->get_shape().at(i).checked_as<constant_c>());
        if (1 != cur_dim) {
            vx_info_.axis = i;
            break;
        }
    }
    vx_info_.lanes
            = vectorize_step(ctx, info_.inputs_[0]->details_.dtype_.type_code_);
    auto func = [&](const std::vector<expr> &in,
                        std::vector<expr::lvalue_proxy_t> &out, int mask_count,
                        float mask_value) -> stmt {
        return builder::make_assign_unattached(
                out[0], compute_element(in[0], mask_count, mask_value));
    };
    // Currenly only support for exp
    bool use_mask = op_name_ == "exp";
    compute_vectorized_op(inputs, *dst[0], info_, vx_info_,
            mask_compute_func_t(func), mask_compute_func_t(func), attrs_, wkld,
            use_mask);
}

void unary_elementwise_op_t::prepare_fusion_data(fdata_map &fdmap) {
    COMPILE_ASSERT(info_.outputs_.size() == 1, "Wrong op output size.\n");
    auto &in_detail0 = fdmap.get(info_.inputs_[0]);
    in_detail0.use_count_++;
}

void unary_elementwise_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    infer_unary_slice_ranges(this, fsmap);
}

void unary_elementwise_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    pre_unary_slice_ranges(this, fsmap, stat_map);
}

bool unary_elementwise_op_t::register_brgemm_fusion(const context_ptr &ctx,
        const std::vector<tensor_slice *> &outputs,
        const std::vector<const tensor_slice *> &inputs,
        brgemm_fusion_register &brg_reg) {
    if (!fuse_in_brgemm_) { return false; }
    return brg_reg.register_op_infos(
            shared_from_this(), outputs[0]->get_tensor_ptr());
}

transpose_op_t::transpose_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : axes_(attrs.get<std::vector<int>>("axes")) {
    info_.inputs_ = ins;
    info_.outputs_ = outs;
    assert(info_.inputs_.size() == 1);
    assert(axes_.size() == 2);
    if (info_.outputs_.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        auto dims = info_.inputs_[0]->details_.get_plain_dims();
        std::swap(dims[axes_[0]], dims[axes_[1]]);
        info_.outputs_[0]->details_.set_plain_dims(dims);
        info_.outputs_[0]->details_.dtype_ = ins[0]->details_.dtype_;
    }
    attrs_ = attrs;
    op_name_ = "transpose";
}

transpose_op_t::transpose_op_t(graph_tensor_ptr v, std::vector<int> &axes)
    : axes_(axes) {
    info_.inputs_.emplace_back(std::move(v));
    info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
    op_name_ = "transpose";
}

void transpose_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {
    auto axes = attrs_.get<std::vector<int>>("axes");
    COMPILE_ASSERT(!info_.inputs_[0]->details_.get_format().is_any(),
            "cannot infer output format with any input format");
    std::vector<int> storage_args;
    bool is_batch = info_.inputs_[0]
                            ->details_.get_format()
                            .format_code_.is_batch_format();

    auto in_format_code = info_.inputs_[0]->details_.get_format().format_code_;
    for (int i = 0; i < sc_data_format_kind_t::MAX_DIMS; ++i) {
        int axis = in_format_code.get(i);
        if (axis == axes[0]) {
            axis = axes[1];
        } else if (axis == axes[1]) {
            axis = axes[0];
        }
        storage_args.push_back(axis);
    }
    sc_data_format_t::blocking_t blocks
            = info_.inputs_[0]->details_.get_format().blocks_;

    auto output_format = sc_data_format_t(is_batch, storage_args, blocks);

    in_formats.push_back(std::vector<sc_data_format_t> {
            info_.inputs_[0]->details_.get_format()});
    out_formats.push_back(std::vector<sc_data_format_t> {output_format});
}

void transpose_op_t::prepare_fusion_data(fdata_map &fdmap) {
    COMPILE_ASSERT(info_.inputs_.size() == 1, "Wrong op input size.\n");
    COMPILE_ASSERT(info_.outputs_.size() == 1, "Wrong op output size.\n");
    auto &in_detail0 = fdmap.get(info_.inputs_[0]);
    in_detail0.use_count_++;
}

void transpose_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map = search_known_slice_ranges(this, fsmap);
    // transpose_op_t need to reorder it to new axes order.
    size_t slice_size = known_ranges_map[0].size();
    slice_range_list transpose_ranges_list(slice_size);
    for (size_t i = 0; i < slice_size; i++) {
        transpose_ranges_list[i] = known_ranges_map[0][i];
        std::swap(transpose_ranges_list[i][axes_[0]],
                transpose_ranges_list[i][axes_[1]]);
    }
    fsmap.get(get_outputs()[0]) = std::move(transpose_ranges_list);
}

void transpose_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {}

void compute_block_transpose(const std::vector<const tensor_slice *> &src,
        const tensor_slice &dst, const std::vector<int> &axes, size_t wkld) {
    std::vector<expr> iters(src[0]->nslice_dims());
    std::vector<expr> src_idx(src[0]->nslice_dims());
    std::vector<expr> dst_idx(src[0]->nslice_dims());

    for (unsigned i = 0; i < src[0]->nslice_dims(); i++) {
        iters[i] = builder::make_var(datatypes::index,
                std::string("_fuseiter") + std::to_string(idx++));
        src_idx[i] = iters[i];
    }
    dst_idx = src_idx;
    std::swap(dst_idx[axes[0]], dst_idx[axes[1]]);
    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder is set");

    stmt cur;
    expr indexed_target = builder::make_indexing(dst.tptr_, dst_idx);
    expr indexed_input = builder::make_indexing(src[0]->tptr_, src_idx);
    cur = make_stmt<assign_node_t>(indexed_target, indexed_input);
    for (int64_t i = src[0]->nslice_dims() - 1; i >= 0; i--) {
        auto body = make_stmt<stmts_node_t>(std::vector<stmt> {std::move(cur)});
        cur = make_stmt<for_loop_node_t>(iters[i], expr(0),
                src[0]->get_shape()[i], expr(1), std::move(body), true,
                for_type::NORMAL);
    }
    bld->emit(cur);
}

void transpose_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);
    compute_block_transpose(inputs, *dst[0], axes_, wkld);
}

size_t transpose_op_t::compute_workload(
        const std::vector<shape_dtype_pair> &ins,
        const std::vector<shape_dtype_pair> &outs) {
    return fusible_op_t::compute_workload(ins, outs)
            * workload_penalty_coefficient;
}

sc_dims tensor_view_op_t::get_shapes() const {
    // if `shapes_` is empty, it needs to get dynamically in need.
    return shapes_.empty() ? info_.outputs_[0]->details_.get_blocking_dims()
                           : shapes_;
}

tensor_view_op_t::tensor_view_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    op_name_ = "tensor_view";
    COMPILE_ASSERT(ins.size() == 1, "Reshape takes 1 input");
    info_.inputs_ = ins;
    auto cache_input_format = ins[0]->details_.get_format();
    attrs_ = attrs;
    auto &shapes = attrs_.get<sc_dims>("shape");
    auto format = attrs.get_or_else("format", sc_data_format_t());
    int total_shape1 = 1, total_shape2 = 1;
    for (auto &dim : sc_data_format_t::get_padded_plain_shapes(
                 ins[0]->details_.get_blocking_dims(), cache_input_format)) {
        total_shape1 *= dim;
    }
    for (auto &dim : shapes) {
        total_shape2 *= dim;
    }
    COMPILE_ASSERT(total_shape1 == total_shape2,
            "Wrong total size of input shapes, can not do reshape plain dims "
            "from " << utils::print_vector(ins[0]->details_.get_plain_dims())
                    << " to " << utils::print_vector(shapes));
    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        info_.outputs_[0]->details_.dtype_ = ins[0]->details_.dtype_;
        info_.outputs_[0]->details_.set_plain_dims(
                sc_data_format_t::get_padded_plain_shapes(shapes, format));
        info_.outputs_[0]->details_.set_format(format);
        shapes_ = shapes;
    } else {
        COMPILE_ASSERT(outs.size() == 1, "Wrong op output size.\n");
        info_.outputs_ = outs;
        // changed to get dynamically in need.
        // shapes_ = outs[0]->details_.get_blocking_dims();
    }
    if (cache_input_format.is_any()) {
        cache_input_format
                = sc_data_format_t(sc_data_format_kind_t::get_plain_by_dims(
                        ins[0]->details_.get_plain_dims().size()));
    }
    attrs_["cache_input_format"] = cache_input_format;
}

tensor_view_op_t::tensor_view_op_t(graph_tensor_ptr v, const sc_dims &shapes)
    : shapes_(shapes) {
    info_.inputs_.emplace_back(std::move(v));
    info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
    op_name_ = "tensor_view";
    attrs_["shape"] = shapes;
}

bool tensor_view_op_t::try_penetrate(
        sc_data_format_t &new_output_format) const {
    auto input_plain_shapes = info_.inputs_[0]->details_.get_plain_dims();
    auto input_format = info_.inputs_[0]->details_.get_format();
    // if it is batch format return false
    if (input_format.format_code_.is_batch_format()
            || info_.outputs_[0]
                       ->details_.get_format()
                       .format_code_.is_batch_format()) {
        return false;
    }
    auto output_plain_shapes = info_.outputs_[0]->details_.get_plain_dims();
    auto input_size = input_plain_shapes.size();
    auto output_size = output_plain_shapes.size();
    bool inp_short = input_size < output_size;
    auto &short_plain_shapes
            = inp_short ? input_plain_shapes : output_plain_shapes;
    auto &long_plain_shapes
            = inp_short ? output_plain_shapes : input_plain_shapes;
    auto &short_size = inp_short ? input_size : output_size;
    auto &long_size = inp_short ? output_size : input_size;
    bool can_penetrate = true;
    std::unordered_map<size_t, size_t> long_to_short;
    // if inp_short, inp blk idx to out blk idx
    std::unordered_map<size_t, size_t> inp_blk_map;
    size_t short_idx = 0, long_idx = 0;
    while (short_idx < short_size) {
        int64_t acc_shape = long_plain_shapes[long_idx];
        long_to_short[long_idx] = short_idx;
        long_idx++;
        while (long_idx < long_size
                && (acc_shape < short_plain_shapes[short_idx]
                        || long_plain_shapes[long_idx] == 1)) {
            acc_shape *= long_plain_shapes[long_idx];
            long_to_short[long_idx] = short_idx;
            long_idx++;
        }

        if (acc_shape != short_plain_shapes[short_idx]) {
            can_penetrate = false;
            break;
        }
        // blocking of short format is big than corresponding long plain dims.
        if (inp_short) {
            auto blk_idx_list
                    = input_format.format_code_.collect_blocking_index(
                            short_idx);
            if (!blk_idx_list.empty()) {
                inp_blk_map[short_idx] = long_idx - 1;
                if (input_format.blocks_[blk_idx_list[0]]
                        > long_plain_shapes[long_idx - 1]) {
                    can_penetrate = false;
                    break;
                }
            }
        }
        short_idx++;
    }

    if (can_penetrate) {
        if (!inp_short) {
            auto &input_code = input_format.format_code_;
            sc_data_format_t new_format;
            auto &new_code = new_format.format_code_;
            int out_count[sc_data_format_kind_t::MAX_DIMS] = {0};
            size_t blk_idx = 0;
            for (int i = 0; i < input_code.ndims(); i++) {
                new_code.set(i, long_to_short[input_code.get(i)]);
                out_count[new_code.get(i)]++;
                if (out_count[new_code.get(i)] > 1
                        && blk_idx < input_size - output_size) {
                    new_format.blocks_[blk_idx++]
                            = input_plain_shapes[input_code.get(i)];
                }
            }
            new_code.set(sc_data_format_kind_t::MAX_DIMS,
                    input_format.format_code_.get(
                            sc_data_format_kind_t::MAX_DIMS));
            size_t inp_blk_idx = 0;
            while (inp_blk_idx < 4 && blk_idx < 4
                    && input_format.blocks_[inp_blk_idx] > 0) {
                new_format.blocks_[blk_idx++]
                        = input_format.blocks_[inp_blk_idx++];
            }
            new_output_format = new_format;
            return true;
        } else {
            sc_data_format_t new_format;
            auto &new_code = new_format.format_code_;
            for (int i = 0; i < static_cast<int>(output_plain_shapes.size());
                    i++) {
                new_code.set(i, i);
            }
            int inp_plain_idx = input_format.format_code_.norig_dims();
            int inp_blk_size = input_format.format_code_.ndims()
                    - input_format.format_code_.norig_dims();
            for (int i = 0; i < inp_blk_size; i++) {
                new_code.set(i + static_cast<int>(output_plain_shapes.size()),
                        inp_blk_map[input_format.format_code_.get(
                                i + inp_plain_idx)]);
            }
            new_code.set(sc_data_format_kind_t::MAX_DIMS,
                    input_format.format_code_.get(
                            sc_data_format_kind_t::MAX_DIMS));
            new_format.blocks_ = input_format.blocks_;
            new_output_format = new_format;
            return true;
        }
    }
    new_output_format = info_.outputs_[0]->details_.get_format();
    return false;
}

void tensor_view_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {
    sc_data_format_t output_format;
    bool can_penetrate = try_penetrate(output_format);
    if (can_penetrate) {
        out_formats.push_back({output_format});
        in_formats.push_back({info_.inputs_[0]->details_.get_format()});
    } else {
        out_formats.push_back({info_.outputs_[0]->details_.get_format()});
        in_formats.push_back(
                {attrs_.get<sc_data_format_t>("cache_input_format")});
    }
}

void tensor_view_op_t::prepare_fusion_data(fdata_map &fdmap) {
    auto &in_detail0 = fdmap.get(info_.inputs_[0]);
    in_detail0.use_count_++;
}

slice_range_list infer_tensor_view_slice(
        const slice_range_list &known_ranges_list, const sc_dims &src_dims,
        const sc_dims &dst_dims) {
    slice_range_list ret;
    // auto skip
    if (src_dims == dst_dims) {
        ret = known_ranges_list;
        return ret;
    }
    for (auto &known_ranges : known_ranges_list) {
        bool slice_stop = false;
        // flatten index
        expr flatten_idx = 0;
        // total length
        sc_dim total_len = 1;
        // accumulater src dims
        sc_dim acc_src_dim = 1;
        for (int i = src_dims.size() - 1; i >= 0; i--) {
            if (slice_stop) {
                // check whether slice is full on last several dims
                if (get_const_as_int(
                            known_ranges[i].second.checked_as<constant_c>())
                        != 1)
                    // if tensor_view deals with inconsequence slice, it will
                    // return empty slice range list to tell fusion manager not
                    // to fuse it
                    return slice_range_list {};
            }
            if (!(known_ranges[i].first.isa<constant_c>()
                        && get_const_as_int(
                                   known_ranges[i]
                                           .first.checked_as<constant_c>())
                                == 0
                        && get_const_as_int(
                                   known_ranges[i]
                                           .second.checked_as<constant_c>())
                                == src_dims[i])) {
                slice_stop = true;
            }
            total_len *= get_const_as_int(
                    known_ranges[i].second.checked_as<constant_c>());
            flatten_idx = flatten_idx
                    + known_ranges[i].first * expr(dim2unsigned(acc_src_dim));
            acc_src_dim *= src_dims[i];
        }
        // deflatten to new shape
        slice_range reshape_ranges;
        sc_dims acc_dst_dim;
        sc_dim tmp_acc = 1;
        for (int64_t i = static_cast<int64_t>(dst_dims.size()) - 1; i >= 0;
                i--) {
            tmp_acc *= dst_dims[i];
            acc_dst_dim.emplace_back(tmp_acc);
        }
        std::reverse(acc_dst_dim.begin(), acc_dst_dim.end());
        std::vector<expr> dst_idx;
        for (unsigned i = 0; i < dst_dims.size() - 1; i++) {
            expr cur_idx = flatten_idx / expr(dim2unsigned(acc_dst_dim[i + 1]));
            dst_idx.emplace_back(cur_idx);
            flatten_idx = flatten_idx % expr(dim2unsigned(acc_dst_dim[i + 1]));
        }
        slice_stop = false;
        for (int64_t i = static_cast<int64_t>(dst_dims.size()) - 1; i >= 0;
                i--) {
            if (!slice_stop && total_len >= acc_dst_dim[i]) {
                reshape_ranges.emplace_back(std::make_pair(
                        expr(0), expr(dim2unsigned(dst_dims[i]))));
                if (total_len == acc_dst_dim[i]) slice_stop = true;
            } else {
                if (!slice_stop) slice_stop = true;
                if (i == static_cast<int64_t>(dst_dims.size()) - 1) {
                    reshape_ranges.emplace_back(std::make_pair(
                            flatten_idx, expr(dim2unsigned(total_len))));
                } else {
                    reshape_ranges.emplace_back(std::make_pair(dst_idx[i],
                            expr(std::max(UINT64_C(1),
                                    dim2unsigned(
                                            total_len / acc_dst_dim[i + 1])))));
                }
            }
        }
        std::reverse(reshape_ranges.begin(), reshape_ranges.end());
        constant_folder_t f;
        auto_caster_t ca;
        for (auto &r : reshape_ranges) {
            r.first = f.expand_polynomial(ca(r.first)).remove_const();
        }
        ret.emplace_back(reshape_ranges);
    }
    return ret;
}

void tensor_view_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map = search_known_slice_ranges(this, fsmap);
    slice_range_list known_ranges_list = known_ranges_map[0];

    if (fsmap.get(get_outputs()[0]).empty()) {
        // src
        auto src_dims = info_.inputs_[0]->details_.get_blocking_dims();
        // dst
        auto shapes = get_shapes();

        auto tv_slice
                = infer_tensor_view_slice(known_ranges_list, src_dims, shapes);
        if (tv_slice.empty()) {
            stat_map.append_ops_by_status(this, infer_status_code::FAIL);
            return;
        }
        fsmap.get(get_outputs()[0]) = tv_slice;
    }
}

void tensor_view_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    if (fsmap.get(get_inputs()[0]).empty()) {
        slice_range_list known_ranges_list = fsmap.get(get_outputs()[0]);
        // src
        auto src_dims = info_.inputs_[0]->details_.get_blocking_dims();
        // dst
        auto shapes = get_shapes();
        // NOTE: pre_slice_ranges use shapes as src_dims
        auto tv_slice
                = infer_tensor_view_slice(known_ranges_list, shapes, src_dims);
        if (tv_slice.empty()) {
            stat_map.append_ops_by_status(this, infer_status_code::FAIL);
            return;
        }
        fsmap.get(get_inputs()[0]) = tv_slice;
        // recursively pre-infer
        info_.inputs_[0]
                ->producer_owner_->dyn_cast<fusible_op_t>()
                ->pre_slice_ranges(fsmap, stat_map);
    }
}

void tensor_view_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {}

reshape_op_t::reshape_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    op_name_ = "reshape";
    COMPILE_ASSERT(ins.size() == 1, "Reshape copy takes 1 input");
    info_.inputs_ = ins;
    attrs_ = attrs;
    auto &shapes = attrs_.get<sc_dims>("shape");
    int total_shape1 = 1, total_shape2 = 1;
    for (auto &dim : ins[0]->details_.get_plain_dims()) {
        total_shape1 *= dim;
    }
    for (auto &dim : shapes) {
        total_shape2 *= dim;
    }
    COMPILE_ASSERT(total_shape1 == total_shape2,
            "Wrong total size of input shapes, can not do reshape plain dims "
            "from " << utils::print_vector(ins[0]->details_.get_plain_dims())
                    << " to " << utils::print_vector(shapes));
    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        info_.outputs_[0]->details_.dtype_ = ins[0]->details_.dtype_;
        info_.outputs_[0]->details_.set_plain_dims(shapes);
        shapes_ = shapes;
    } else {
        COMPILE_ASSERT(outs.size() == 1, "Wrong op output size.\n");
        info_.outputs_ = outs;
        shapes_ = outs[0]->details_.get_plain_dims();
    }
}
void reshape_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {}
void reshape_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    // fake infer slice
    std::vector<std::pair<expr, expr>> ranges;
    auto &shapes = info_.outputs_[0]->details_.get_plain_dims();
    ranges.reserve(shapes.size());
    for (size_t i = 0; i < shapes.size(); i++) {
        ranges.emplace_back(expr(0), expr(dim2unsigned(shapes[i])));
    }
    fsmap.get(get_outputs()[0]).push_back(ranges);
}
void reshape_op_t::prepare_fusion_data(fdata_map &fdmap) {
    auto &in_detail0 = fdmap.get(info_.inputs_[0]);
    in_detail0.use_count_++;
}
void reshape_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {
    out_formats.push_back({sc_data_format_kind_t::get_plain_by_dims(
            info_.outputs_[0]->details_.get_plain_dims().size())});
}
void reshape_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dsts,
        const std::vector<const tensor_slice *> &inputs) {
    auto *src = inputs[0];
    auto *dst = dsts[0];
    // accumulate src tensor size
    std::vector<expr> src_accsize {expr(dim2unsigned(1))};
    expr src_size = expr(dim2unsigned(1));
    // accumulate dst tensor size
    std::vector<expr> dst_accsize {expr(dim2unsigned(1))};
    expr dst_size = expr(dim2unsigned(1));
    // outer nested loop vars
    expr iters = builder::make_var(
            datatypes::index, std::string("_fuseiter") + std::to_string(idx++));
    // the indices for the input tensor.
    std::vector<expr> src_idx(src->nslice_dims());
    // the indices for the output tensor.
    std::vector<expr> dst_idx(dst->nslice_dims());
    uint64_t total_size = 1;
    for (auto it : shapes_) {
        total_size *= it;
    }
    for (int64_t i = inputs.at(0)->nslice_dims() - 1; i > 0; i--) {
        src_size = src_size * src->get_shape()[i];
        src_accsize.emplace_back(src_size);
    }

    for (auto i = dst->nslice_dims() - 1; i > 0; i--) {
        dst_size = dst_size * dst->get_shape()[i];
        dst_accsize.emplace_back(dst_size);
    }
    std::reverse(src_accsize.begin(), src_accsize.end());
    std::reverse(dst_accsize.begin(), dst_accsize.end());

    for (int i = 0; i < (int)src->nslice_dims(); i++) {
        if (i == 0) {
            src_idx[i] = iters / src_accsize[i] + src->get_offset()[i];
        } else {
            src_idx[i] = iters % src_accsize[i - 1] / src_accsize[i]
                    + src->get_offset()[i];
        }
    }
    for (int i = 0; i < (int)dst->nslice_dims(); i++) {
        if (i == 0) {
            dst_idx[i] = iters / dst_accsize[i] + dst->get_offset()[i];
        } else {
            dst_idx[i] = iters % dst_accsize[i - 1] / dst_accsize[i]
                    + dst->get_offset()[i];
        }
    }
    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder is set");

    expr indexed_target = builder::make_indexing(dst->tptr_, {dst_idx});
    expr indexed_input = builder::make_indexing(src->tptr_, src_idx);
    stmt_c cur = builder::make_assign_unattached(indexed_target, indexed_input);
    auto body = builder::make_stmts_unattached(
            std::vector<stmt_c> {std::move(cur)});
    cur = builder::make_for_loop_unattached(iters, expr(0), expr(total_size),
            expr(1), std::move(body), true, for_type::NORMAL);
    constant_folder_t folder;
    cur = folder(cur);
    bld->emit(cur.remove_const());
}

split_op_t::split_op_t(graph_tensor_ptr v, int dim, const sc_dims &shapes)
    : dim_(dim), shapes_(shapes) {
    attrs_.set(op_attr_key::no_fuse, true);
    info_.inputs_.emplace_back(std::move(v));
    for (unsigned i = 0; i < shapes_.size(); i++) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
    }
}

void split_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {
    out_formats.reserve(info_.outputs_.size());
    for (size_t i = 0; i < out_formats.size(); ++i) {
        out_formats[i].push_back({info_.inputs_[0]->details_.get_format()});
    }
}

void split_op_t::prepare_fusion_data(fdata_map &fdmap) {
    auto &in_detail0 = info_.inputs_[0];
    fdmap.get(in_detail0).use_count_++;
    COMPILE_ASSERT(info_.outputs_.size() > 1,
            "Split op output size should bigger than 1.\n");
    auto dims = in_detail0->details_.get_blocking_dims();
    auto dims_size = dims.size();
    COMPILE_ASSERT(dims_size > dim_, "Split dim is not available.\n");
    sc_dim total_split = 0;
    for (auto num : shapes_) {
        total_split += num;
    }
    COMPILE_ASSERT(total_split == dims[dim_],
            "Split shapes are not matched with input.\n");
    for (unsigned i = 0; i < info_.outputs_.size(); i++) {
        auto &output = info_.outputs_[i];
        sc_dims out_dims(dims_size);
        std::vector<expr> tmp_shape;
        auto &outdetail = fdmap.get(output);
        for (unsigned j = 0; j < dims_size; j++) {
            if (j != dim_) {
                tmp_shape.emplace_back(dim2unsigned(dims[j]));
                out_dims.emplace_back(
                        info_.inputs_[0]->details_.get_blocking_dims()[j]);
            } else {
                tmp_shape.emplace_back(dim2unsigned(shapes_[i]));
                out_dims.emplace_back(shapes_[i]);
            }
        }
        output->details_.set_blocking_dims(out_dims);
    }
}

void split_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map = search_known_slice_ranges(this, fsmap);
    size_t slice_size = known_ranges_map[0].size();
    slice_range_list split_ranges_list = known_ranges_map[0];
    for (size_t i = 0; i < get_outputs().size(); i++) {
        fsmap.get(get_outputs()[i]).resize(slice_size);
        for (size_t n = 0; n < slice_size; n++) {
            for (size_t j = 0; j < split_ranges_list.at(n).size(); j++) {
                if (j == dim_) {
                    // Due to query stage, split shapes should be matched
                    // with input.
                    fsmap.get(get_outputs()[i])
                            .at(n)
                            .emplace_back(std::make_pair(
                                    expr(0), dim2unsigned(shapes_.at(i))));
                } else {
                    fsmap.get(get_outputs()[i])
                            .at(n)
                            .emplace_back(std::make_pair(
                                    split_ranges_list.at(n).at(j).first,
                                    split_ranges_list.at(n).at(j).second));
                }
            }
        }
    }
}

void split_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {}

void compute_block_split(const std::vector<const tensor_slice *> &src,
        const std::vector<tensor_slice *> &dst, unsigned dim,
        const sc_dims &shapes, size_t wkld = 0UL) {
    // outer nested loop vars
    std::vector<expr> outer_iter(dim);
    // inner nested loop vars
    std::vector<std::vector<expr>> inner_iter(
            static_cast<uint64_t>(src[0]->nbase_dims()) - dim);
    // the indices for multiple inputs. First dim: the input, Second
    // dim: the dimemsions in the tensor
    std::vector<std::vector<expr>> src_idx(dst.size());
    // the indices for the output tensor. Cause concat is a assign op,
    // we need number of src indexes.
    std::vector<std::vector<expr>> dst_idx(dst.size());
    for (int64_t i = 0; i < src[0]->nbase_dims(); i++) {
        if (i < dim) { // outer loop
            // make the loop var for the for-loop
            outer_iter[i] = builder::make_var(datatypes::index,
                    std::string("_fuseiter") + std::to_string(idx++));
            for (unsigned j = 0; j < dst.size(); j++) {
                src_idx[j].emplace_back(outer_iter[i]);
                dst_idx[j].emplace_back(outer_iter[i]);
            }
        } else { // inner loop
            expr cur;
            for (unsigned j = 0; j < dst.size(); j++) {
                inner_iter[i - dim].emplace_back(builder::make_var(
                        datatypes::index,
                        std::string("_fuseiter") + std::to_string(idx++)));
                dst_idx[j].emplace_back(inner_iter[i - dim][j]);
                if (i == dim) {
                    if (j == 0) {
                        cur = 0;
                    } else {
                        cur = cur + dst[j - 1]->get_shape()[i];
                    }
                    src_idx[j].emplace_back(inner_iter[i - dim][j] + cur);
                } else {
                    src_idx[j].emplace_back(inner_iter[i - dim][j]);
                }
            }
        }
    }
    expr indexed_target;
    expr indexed_input;
    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder is set");
    std::vector<stmt> tcur;
    for (unsigned j = 0; j < dst.size(); j++) {
        indexed_target = builder::make_indexing(dst[j]->tptr_, dst_idx[j]);
        indexed_input = builder::make_indexing(src[0]->tptr_, src_idx[j]);
        stmt cur = make_stmt<assign_node_t>(indexed_target, indexed_input);
        cur->attr()[op_traits::workload_computable_t::workload_number] = wkld;
        for (int64_t i = src[0]->nslice_dims() - 1; i >= dim; i--) {
            auto body = make_stmt<stmts_node_t>(
                    std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(inner_iter[i - dim][j], expr(0),
                    dst[j]->get_shape()[i], expr(1), std::move(body), true,
                    for_type::NORMAL);
        }
        tcur.emplace_back(std::move(cur));
    }
    if (dim) {
        stmt cur = make_stmt<stmts_node_t>(std::move(tcur));
        for (int i = dim - 1; i >= 0; i--) {
            stmt body;
            if (cur.isa<for_loop>()) {
                body = make_stmt<stmts_node_t>(
                        std::vector<stmt> {std::move(cur)});
            } else {
                body = cur;
            }
            cur = make_stmt<for_loop_node_t>(outer_iter[i], expr(0),
                    src[0]->get_shape()[i], expr(1), std::move(body), true,
                    for_type::NORMAL);
        }
        bld->emit(cur);
    } else {
        for (auto &cur : tcur) {
            bld->emit(cur);
        }
    }
}

void split_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);
    compute_block_split(inputs, dst, dim_, shapes_, wkld);
}

void reorder_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {
    if (!attrs_.get_or_else("internal", false)) {
        in_formats.push_back(std::vector<sc_data_format_t> {
                info_.inputs_[0]->details_.get_format()});
        out_formats.push_back(std::vector<sc_data_format_t> {
                info_.inputs_[0]->details_.get_format()});
    }
}

reorder_op_t::reorder_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    for (auto &in : ins) {
        info_.inputs_.emplace_back(in);
    }
    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(
                this, info_.inputs_[0]->details_));
        info_.outputs_[0]->details_.set_format(
                attrs.get<sc_data_format_t>("out_format"));
    } else {
        info_.outputs_ = outs;
    }
    op_name_ = "reorder";
    attrs_ = attrs;
    plain_dims_ = ins[0]->details_.get_plain_dims();
    input_format_ = info_.inputs_[0]->details_.get_format();
    output_format_ = info_.outputs_[0]->details_.get_format();
    COMPILE_ASSERT(info_.inputs_[0]->details_.get_format().is_convertible(
                           info_.outputs_[0]->details_.get_format()),
            "input format " << info_.inputs_[0]->details_.get_format()
                            << " can not convert to "
                            << info_.outputs_[0]->details_.get_format() << ".");
}

reorder_op_t::reorder_op_t(graph_tensor_ptr v, sc_data_format_t input_format,
        sc_data_format_t output_format)
    : input_format_(input_format), output_format_(output_format) {
    COMPILE_ASSERT(input_format.is_convertible(output_format),
            "input format " << input_format << " can not convert to "
                            << output_format << ".");
    plain_dims_ = v->details_.get_plain_dims();
    info_.inputs_.emplace_back(std::move(v));
    info_.outputs_.emplace_back(
            std::make_shared<graph_tensor>(this, info_.inputs_[0]->details_));
    info_.outputs_[0]->details_.set_format(output_format);
    op_name_ = "reorder";
}

void reorder_op_t::prepare_fusion_data(fdata_map &fdmap) {
    auto &in_detail0 = fdmap.get(info_.inputs_[0]);
    in_detail0.use_count_++;
}

// This function will try to merge multi slice range
static void merge_multi_slice(slice_range &src_slice) {
    bool all_const_start = true;
    for (auto &r : src_slice) {
        if (!r.first.isa<constant_c>()) {
            all_const_start = false;
            break;
        }
    }
    if (all_const_start) {
        std::sort(src_slice.begin(), src_slice.end(),
                [](const std::pair<expr, expr> &a,
                        const std::pair<expr, expr> &b) {
                    return get_const_as_int(a.first.checked_as<constant_c>())
                            < get_const_as_int(
                                    b.first.checked_as<constant_c>());
                });
    }
    slice_range dst_slice;
    std::pair<int, int> temp_range = {0, 0};
    for (auto &r : src_slice) {
        if (r.first.isa<constant_c>() && r.second.isa<constant_c>()) {
            int start = get_const_as_int(r.first.checked_as<constant_c>());
            int length = get_const_as_int(r.second.checked_as<constant_c>());
            // concat
            if (temp_range.second == 0
                    || start == temp_range.first + temp_range.second) {
                if (temp_range.second == 0) temp_range.first = start;
                temp_range.second += length;
            }
            // push and reset
            else {
                dst_slice.emplace_back(temp_range);
                temp_range.first = start;
                temp_range.second = length;
            }
        } else {
            if (temp_range.second > 0) {
                dst_slice.emplace_back(temp_range);
                temp_range.second = 0;
            }
            dst_slice.emplace_back(std::move(r));
        }
    }
    if (temp_range.second > 0) {
        dst_slice.emplace_back(temp_range);
        temp_range.second = 0;
    }
    src_slice = std::move(dst_slice);
}

// Get block to plain ranges
slice_range get_block2plain_ranges(const expr &block_num_start,
        const expr &block_num_length, const expr &block_size_start,
        const expr &block_size_length, int blocks) {
    COMPILE_ASSERT(block_num_length.isa<constant_c>()
                    && block_size_length.isa<constant_c>(),
            "constant length is expected, but got "
                    << block_num_length << " and " << block_size_length);
    int block_num_length_int
            = get_const_as_int(block_num_length.checked_as<constant_c>());
    int block_size_length_int
            = get_const_as_int(block_size_length.checked_as<constant_c>());

    std::vector<std::pair<expr, expr>> plain_range_list;
    if (block_size_length_int == blocks) {
        // when block size is equal to blocks, reorder will generate
        // consequent slice in output
        auto plain_range
                = std::make_pair(block_num_start * blocks + block_size_start,
                        expr(block_num_length_int * block_size_length_int));
        plain_range_list = {plain_range};
    } else {
        // multi plain ranges
        for (int i = 0; i < block_num_length_int; i++) {
            constant_folder_t f;
            auto_caster_t ca;
            auto plain_range
                    = std::make_pair(f(ca((block_num_start + expr(i)) * blocks
                                               + block_size_start))
                                             .remove_const(),
                            block_size_length);
            plain_range_list.emplace_back(plain_range);
        }
    }
    // try to merge multi slice
    merge_multi_slice(plain_range_list);
    return plain_range_list;
}

// get greatest common divisor of block_in and block_out
inline int get_gcd(int a, int b) {
    COMPILE_ASSERT(a * b != 0, "non-zero number is expected");
    int i = std::min(a, b);
    while (a % i != 0 || b % i != 0) {
        i--;
        if (i == 0) return 1;
    }
    return i;
}

// Get plain to block ranges
std::vector<std::pair<std::pair<expr, expr>, std::pair<expr, expr>>>
get_plain2block_ranges(const expr &start, const expr &length, int blocks) {
    std::vector<std::pair<std::pair<expr, expr>, std::pair<expr, expr>>> ret;
    COMPILE_ASSERT(length.isa<constant_c>(),
            "constant length is expected, but got " << length);
    int ilength = get_const_as_int(length.checked_as<constant_c>());
    expr folded_start
            = constant_folder_t()(auto_caster_t()(start)).remove_const();

    // Case 1: the most commone case.
    if (folded_start.isa<constant>() && get_expr_as_int(folded_start) == 0) {
        if (ilength >= blocks) {
            auto block_num_range = std::make_pair(0, ilength / blocks);
            auto block_size_range = std::make_pair(0, blocks);
            ret.emplace_back(std::make_pair(
                    std::move(block_num_range), std::move(block_size_range)));
        }
        if (ilength % blocks != 0) {
            auto block_num_range = std::make_pair(ilength / blocks, 1);
            auto block_size_range = std::make_pair(0, ilength % blocks);
            ret.emplace_back(std::make_pair(
                    std::move(block_num_range), std::move(block_size_range)));
        }
    } else {
        // Case 2: gcd case.
        if (folded_start->node_type_ == sc_expr_type::mul) {
            auto r = constant_folding::get_operand_from_binary(folded_start)
                             .second;
            if (r.isa<constant>()) {
                auto multiple = get_expr_as_int(r);
                if (multiple % blocks == 0) {
                    if (ilength >= blocks) {
                        auto block_num_range = std::make_pair(
                                folded_start / blocks, ilength / blocks);
                        auto block_size_range = std::make_pair(0, blocks);
                        ret.emplace_back(
                                std::make_pair(std::move(block_num_range),
                                        std::move(block_size_range)));
                    }
                    if (ilength % blocks != 0) {
                        auto block_num_range = std::make_pair(
                                folded_start / blocks + ilength / blocks, 1);
                        auto block_size_range
                                = std::make_pair(0, ilength % blocks);
                        ret.emplace_back(
                                std::make_pair(std::move(block_num_range),
                                        std::move(block_size_range)));
                    }
                } else {
                    int gcd = get_gcd(multiple, blocks);
                    for (int i = 0; i < ilength / gcd; i++) {
                        auto block_num_range = std::make_pair(
                                (folded_start + i * gcd) / blocks, 1);
                        auto block_size_range = std::make_pair(
                                (folded_start + i * gcd) % blocks, gcd);
                        ret.emplace_back(
                                std::make_pair(std::move(block_num_range),
                                        std::move(block_size_range)));
                    }
                    if (ilength % gcd != 0) {
                        auto block_num_range = std::make_pair(
                                (folded_start + ilength / gcd) / blocks, 1);
                        auto block_size_range = std::make_pair(
                                (folded_start + ilength / gcd) % blocks,
                                ilength % gcd);
                        ret.emplace_back(
                                std::make_pair(std::move(block_num_range),
                                        std::move(block_size_range)));
                    }
                }
            }
        }
    }

    // Case 3: fallback to multi one-length slice
    if (ret.empty()) {
        for (int i = 0; i < ilength; i++) {
            auto block_num_range
                    = std::make_pair((folded_start + i) / blocks, 1);
            auto block_size_range
                    = std::make_pair((folded_start + i) % blocks, 1);
            ret.emplace_back(std::make_pair(
                    std::move(block_num_range), std::move(block_size_range)));
        }
    }
    return ret;
}

void infer_stride2plain_reorder(slice_range &input_slice,
        sc_data_format_t input_format, sc_data_format_t output_format,
        slice_range &output_slice) {
    int ndim_begin = static_cast<int>(input_slice.size())
            - input_format.format_code_.norig_dims();
    output_slice = input_slice;
    for (int i = 0; i < input_format.format_code_.norig_dims(); i++) {
        int plain_axis = input_format.format_code_.get(i);
        output_slice[plain_axis + ndim_begin] = input_slice[i + ndim_begin];
    }
}

void infer_plain2stride_reorder(slice_range &input_slice,
        sc_data_format_t input_format, sc_data_format_t output_format,
        slice_range &output_slice) {
    int ndim_begin = static_cast<int>(input_slice.size())
            - input_format.format_code_.norig_dims();
    output_slice = input_slice;
    for (int i = 0; i < output_format.format_code_.norig_dims(); i++) {
        int plain_axis = output_format.format_code_.get(i);
        output_slice[i + ndim_begin] = input_slice[plain_axis + ndim_begin];
    }
}

void infer_stride2stride_reorder(slice_range_list &input_slice_list,
        sc_data_format_t input_format, sc_data_format_t output_format,
        slice_range_list &output_slice_list) {
    for (auto &input_slice : input_slice_list) {
        slice_range plain_slice, reorder_ranges;
        infer_stride2plain_reorder(
                input_slice, input_format, output_format, plain_slice);
        infer_plain2stride_reorder(
                plain_slice, input_format, output_format, reorder_ranges);
        output_slice_list.emplace_back(reorder_ranges);
    }
}

/**
 * For [AxBxCxD], if B and D, have more than one slice ranges, such as two, it
 * needs to dispatch them to four (2x2) slice ranges. Note that: if B and D come
 * from same plain axis, it only generate two slice ranges instead of four.
 * */
void dispatch_reorder_ranges(slice_range_list &total_ranges_list,
        slice_range_list &reorder_ranges_list,
        sc_data_format_t output_format = sc_data_format_t()) {
    std::vector<int> acc_size_list;
    int total_size = 1;
    for (size_t i = 0; i < total_ranges_list.size(); i++) {
        total_size *= total_ranges_list.at(i).size();
        acc_size_list.emplace_back(total_size);
    }

    // this check is aim to avoid too many compile time, it will also break
    // this kind reorder post fusion
    const int max_slice_size = 16;

    for (int i = 0; i < total_size; i++) {
        // set remainder
        int rmd = i;
        std::vector<int> index_list;
        bool valid = true;
        for (size_t j = 0; j < acc_size_list.size(); j++) {
            int size = (total_size / acc_size_list[j]);
            index_list.emplace_back(rmd / size);
            if (!output_format.is_any()
                    && output_format.format_code_.get(j)
                            < static_cast<int>(index_list.size())
                    && index_list[output_format.format_code_.get(j)]
                            != index_list[j]) {
                valid = false;
                break;
            }
            rmd = rmd % size;
        }
        if (!valid) continue;
        slice_range reorder_range;
        for (size_t j = 0; j < index_list.size(); j++) {
            reorder_range.emplace_back(total_ranges_list[j][index_list[j]]);
        }
        reorder_ranges_list.emplace_back(reorder_range);

        if (reorder_ranges_list.size() > max_slice_size) {
            reorder_ranges_list.clear();
            return;
        }
    }
}

// infer plain format to blocking format generally
void infer_stride2block_reorder(slice_range_list &input_slice_list,
        sc_data_format_t input_format, sc_data_format_t output_format,
        slice_range_list &output_slice_list) {
    auto plain_format = input_format.to_plain();
    auto out_kind = output_format.format_code_;
    for (auto &input_slice : input_slice_list) {
        slice_range_list reorder_ranges_list;
        // stride->plain
        slice_range plain_slice;
        infer_stride2plain_reorder(input_slice, input_format,
                output_format.to_plain(), plain_slice);

        /**
         * block_slice_dict is the map, in which:
         * 1. key: represents plain pos
         * 2. value: is the vector of slice_range, e.g.
         *                               block_1_1  block_1_1
         *             block_1  block_1
         *   plain_1                     block_1_2  block_1_2
         *             block_2  block_2
         *                               block_2_1  block_2_2
         *
         *  the vector may look like as below:
         *  {block_1, block_1_1, block1_1}
         *  {block_1, block_1_2, block1_2}
         *  {block_2, block_2_1, block2_1}
         * */
        std::unordered_map<int, slice_range_list> block_slice_dict;

        // the first is plain index, the second is block cnt
        std::unordered_map<int, std::vector<int>> block_cnt_dict;
        int ndim_begin = static_cast<int>(plain_slice.size())
                - plain_format.format_code_.ndims();

        for (int i = 0; i < out_kind.ndims(); i++) {
            int plain_pos = out_kind.get(i);
            block_cnt_dict[plain_pos].emplace_back(i);
            if (block_slice_dict[plain_pos].empty()) {
                block_slice_dict[plain_pos].emplace_back(
                        slice_range {plain_slice[ndim_begin + plain_pos]});
            } else {
                slice_range_list update_block_slice;
                for (auto &block_range_list : block_slice_dict[plain_pos]) {
                    auto cur_plain_range = block_range_list.back();
                    auto cur_block
                            = output_format.blocks_
                                      [out_kind.collect_blocking_index(
                                                       plain_pos)
                                                      .at(block_cnt_dict[plain_pos] // NOLINT
                                                                      .size()
                                                              - 2)];
                    auto cur_block_ranges
                            = get_plain2block_ranges(cur_plain_range.first,
                                    cur_plain_range.second, cur_block);

                    for (auto &range_pair : cur_block_ranges) {
                        slice_range cpy_last_range_list = block_range_list;
                        cpy_last_range_list.back() = range_pair.first;
                        cpy_last_range_list.emplace_back(range_pair.second);
                        update_block_slice.emplace_back(cpy_last_range_list);
                    }
                }
                if (!update_block_slice.empty())
                    block_slice_dict[plain_pos] = update_block_slice;
            }
        }

        std::vector<slice_range> total_range_list(out_kind.ndims());
        // collect all blocking slice
        for (auto &mp : block_slice_dict) {
            int plain_pos = mp.first;
            for (auto &range_list : mp.second) {
                for (size_t i = 0; i < range_list.size(); i++) {
                    int block_pos = block_cnt_dict[plain_pos].at(i);
                    total_range_list[block_pos].emplace_back(range_list[i]);
                }
            }
        }

        dispatch_reorder_ranges(
                total_range_list, reorder_ranges_list, output_format);

        if (ndim_begin) {
            for (auto &range : reorder_ranges_list) {
                range.insert(range.begin(), input_slice.begin(),
                        input_slice.begin() + ndim_begin);
            }
        }

        output_slice_list.insert(output_slice_list.end(),
                reorder_ranges_list.begin(), reorder_ranges_list.end());
    }
}

// infer blocking format to plain format generally
void infer_block2stride_reorder(slice_range_list &input_slice_list,
        sc_data_format_t input_format, sc_data_format_t output_format,
        slice_range_list &output_slice_list) {
    auto out_kind = output_format.format_code_.to_plain();
    auto in_kind = input_format.format_code_;
    for (auto &input_slice : input_slice_list) {
        slice_range_list reorder_ranges_list;
        std::unordered_map<int, slice_range_list> plain_slice_dict;
        int ndim_begin = static_cast<int>(input_slice.size())
                - input_format.format_code_.ndims();
        // from right to left
        for (int i = in_kind.ndims() - 1; i >= 0; i--) {
            int plain_pos = in_kind.get(i);
            if (plain_slice_dict[plain_pos].empty()) {
                plain_slice_dict[plain_pos].emplace_back(
                        slice_range {input_slice[i + ndim_begin]});
            } else {
                std::pair<expr, expr> cur_block_num_range
                        = input_slice[i + ndim_begin];
                slice_range res;
                for (auto &cur_block_size_range :
                        plain_slice_dict[plain_pos].back()) {
                    auto cur_blocks = in_kind.collect_blocking_index(plain_pos);
                    int cur_block = input_format.blocks_[cur_blocks.at(
                            cur_blocks.size()
                            - plain_slice_dict[plain_pos].size())];
                    slice_range cur_plain_ranges_list
                            = get_block2plain_ranges(cur_block_num_range.first,
                                    cur_block_num_range.second,
                                    cur_block_size_range.first,
                                    cur_block_size_range.second, cur_block);
                    res.insert(res.end(), cur_plain_ranges_list.begin(),
                            cur_plain_ranges_list.end());
                }
                plain_slice_dict[plain_pos].emplace_back(std::move(res));
            }
        }

        std::vector<slice_range> total_range_list;
        for (int i = 0; i < out_kind.ndims(); i++) {
            total_range_list.emplace_back(plain_slice_dict[i].back()); // NOLINT
        }

        dispatch_reorder_ranges(total_range_list, reorder_ranges_list,
                output_format.to_plain());

        if (ndim_begin) {
            for (auto &range : reorder_ranges_list) {
                // plain -> stride
                slice_range stride_range;
                infer_plain2stride_reorder(range, output_format.to_plain(),
                        output_format, stride_range);
                stride_range.insert(stride_range.begin(), input_slice.begin(),
                        input_slice.begin() + ndim_begin);
                range = std::move(stride_range);
            }
        }
        output_slice_list.insert(output_slice_list.end(),
                reorder_ranges_list.begin(), reorder_ranges_list.end());
    }
}

// infer blocking format to blocking format generally
void infer_block2block_reorder(slice_range_list &input_slice_list,
        sc_data_format_t input_format, sc_data_format_t output_format,
        slice_range_list &output_slice_list) {
    slice_range_list plain_ranges_list;
    infer_block2stride_reorder(input_slice_list, input_format,
            input_format.to_plain(), plain_ranges_list);

    infer_stride2block_reorder(plain_ranges_list, input_format.to_plain(),
            output_format, output_slice_list);
}

static inline bool is_not_blocking(sc_data_format_t format) {
    return !format.is_blocking() && !format.is_any();
}

// generally infer reorder slice for any format (except padding cases)
void infer_reorder_slice(slice_range_list &input_slice_list,
        sc_data_format_t input_format, sc_data_format_t output_format,
        slice_range_list &output_slice_list) {
    if (is_not_blocking(input_format) && is_not_blocking(output_format)) {
        infer_stride2stride_reorder(input_slice_list, input_format,
                output_format, output_slice_list);
    } else if (is_not_blocking(input_format) && output_format.is_blocking()) {
        infer_stride2block_reorder(input_slice_list, input_format,
                output_format, output_slice_list);
    } else if (input_format.is_blocking() && is_not_blocking(output_format)) {
        infer_block2stride_reorder(input_slice_list, input_format,
                output_format, output_slice_list);
    } else if (input_format.is_blocking() && output_format.is_blocking()) {
        infer_block2block_reorder(input_slice_list, input_format, output_format,
                output_slice_list);
    } else {
        std::ostringstream ss;
        ss << "Unsupported data format. in = " << input_format
           << ", out = " << output_format;
        SC_WARN << ss.str();
        throw tuner_recoverable_exception_t(ss.str());
    }
    constant_folder_t f;
    auto_caster_t ca;
    for (auto &reorder_range : output_slice_list) {
        for (auto &r : reorder_range) {
            r.first = f.expand_polynomial(ca(r.first)).remove_const();
        }
    }
}

// infer reorder slice according input_slice
void reorder_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    // has been pre-inferred, skip
    if (!fsmap.get(get_outputs()[0]).empty()) return;
    COMPILE_ASSERT(input_format_.is_convertible(output_format_),
            "Can not convert input format "
                    << input_format_ << " to output format " << output_format_
                    << ".");
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map = search_known_slice_ranges(this, fsmap);
    auto input_slice_list = known_ranges_map[0];
    slice_range_list reorder_ranges_list;

    infer_reorder_slice(input_slice_list, input_format_, output_format_,
            reorder_ranges_list);
    if (reorder_ranges_list.empty()) {
        for (auto &user : get_outputs()[0]->uses_) {
            if (user.second->isa<output_op>()) {
                continue;
            } else {
                user.second->attrs_.set(op_attr_key::fused_mode_hint,
                        op_attr_key::break_pre_fuse);
                stat_map.get_ops_by_status(infer_status_code::FAIL)
                        .emplace_back(user.second);
                return;
            }
        }
    }
    fsmap.get(get_outputs()[0]) = reorder_ranges_list;
}

// pre-infer reorder slice according output_slice
void reorder_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    if (fsmap.get(get_inputs()[0]).empty()) {
        slice_range_list known_ranges_list = fsmap.get(get_outputs()[0]);
        slice_range_list input_slice_list;

        infer_reorder_slice(known_ranges_list, output_format_, input_format_,
                input_slice_list);
        if (input_slice_list.empty()) {
            stat_map.append_ops_by_status(this, infer_status_code::FAIL);
            return;
        }
        fsmap.get(get_inputs()[0]) = input_slice_list;
        // recursively pre-infer
        info_.inputs_[0]
                ->producer_owner_->dyn_cast<fusible_op_t>()
                ->pre_slice_ranges(fsmap, stat_map);
    }
}

static std::vector<expr> get_reorder_stride2stride_indexes(
        const std::vector<expr> &in_indexes, const sc_data_format_t &in_format,
        const sc_data_format_t &out_format, const sc_dims &plain_dims) {
    if (in_indexes.empty()) { return std::vector<expr>(); }
    COMPILE_ASSERT(in_format.format_code_ != format_kinds::any
                    && out_format.format_code_ != format_kinds::any,
            "format can not be any in reorder op, please check it in layout "
            "propagation.");
    size_t base_out_dim = 0;
    assert(in_format.format_code_.ndims() == out_format.format_code_.ndims());
    size_t num_plain_dims = in_format.format_code_.norig_dims();
    size_t num_out_dims = num_plain_dims;
    std::vector<expr> ret(num_out_dims, 0);
    if (in_format.format_code_.is_batch_format()) {
        COMPILE_ASSERT(in_indexes.size() >= num_plain_dims,
                "Wrong number of dimensions for batch format: "
                        << in_format << ", real shape = "
                        << utils::print_vector(in_indexes));
        base_out_dim = in_indexes.size() - num_plain_dims;
        num_out_dims = base_out_dim + num_plain_dims;
        ret.resize(num_out_dims, 0);
        for (size_t i = 0; i < base_out_dim; i++) {
            ret[i] = (in_indexes[i]);
        }
    } else {
        COMPILE_ASSERT(in_indexes.size() == num_plain_dims,
                "Wrong number of dimensions for format: "
                        << in_format << ", real shape = "
                        << utils::print_vector(in_indexes));
    };

    COMPILE_ASSERT(in_indexes.size() <= sc_data_format_kind_t::MAX_DIMS,
            "Too many dims in plain shapes");
    std::vector<std::vector<int>> out_axis_map
            = out_format.format_code_.collect_p2b_mapping();
    // format index
    for (auto inp_idx = base_out_dim; inp_idx < num_out_dims; inp_idx++) {
        auto orig_dim = in_format.format_code_.get(inp_idx - base_out_dim);
        assert(orig_dim < static_cast<int>(out_axis_map.size())
                && out_axis_map[orig_dim].size() == 1);
        auto out_idx = out_axis_map[orig_dim][0];
        assert(base_out_dim + out_idx < ret.size());
        ret[base_out_dim + out_idx] = in_indexes[inp_idx];
    }
    return ret;
}

static std::vector<expr> get_reorder_block2plain_indexes(
        const std::vector<expr> &in_indexes, const sc_data_format_t &format,
        const sc_dims &plain_dims, expr &condition) {
    if (in_indexes.empty()) { return std::vector<expr>(); }
    COMPILE_ASSERT(format.format_code_ != format_kinds::any,
            "format can not be any_t in reorder op, please check it in layout "
            "propagation.");
    size_t base_out_dim = 0;
    size_t num_plain_dims = format.format_code_.norig_dims();
    size_t num_format_dims = format.format_code_.ndims();
    size_t num_out_dims = num_plain_dims;
    std::vector<expr> ret(num_out_dims, 0);
    if (format.format_code_.is_batch_format()) {
        COMPILE_ASSERT(in_indexes.size() >= num_format_dims,
                "Wrong number of dimensions for batch format: "
                        << format << ", real shape = "
                        << utils::print_vector(in_indexes));
        base_out_dim = in_indexes.size() - num_format_dims;
        num_out_dims = base_out_dim + num_plain_dims;
        ret.resize(num_out_dims, 0);
        for (size_t i = 0; i < base_out_dim; i++) {
            ret[i] = (in_indexes[i]);
        }
    } else {
        COMPILE_ASSERT(in_indexes.size() == num_format_dims,
                "Wrong number of dimensions for format: "
                        << format << ", real shape = "
                        << utils::print_vector(in_indexes));
    };

    COMPILE_ASSERT(in_indexes.size() <= sc_data_format_kind_t::MAX_DIMS,
            "Too many dims in plain shapes");
    condition = true;
    std::unordered_map<int, int>
            axis2blocks; // plain_axis to block idx, idx++ after an access
    // format index
    for (int inp_idx = static_cast<int>(num_format_dims + base_out_dim) - 1;
            inp_idx >= static_cast<int>(base_out_dim); inp_idx--) {
        auto orig_axis = format.format_code_.get(inp_idx - base_out_dim);
        auto blocks = format.format_code_.collect_blocking_index(orig_axis);
        if (axis2blocks.find(orig_axis) == axis2blocks.end()) {
            axis2blocks[orig_axis] = static_cast<int>(blocks.size());
        }
        if (axis2blocks[orig_axis] == static_cast<int>(blocks.size())) {
            ret[base_out_dim + orig_axis]
                    = ret[base_out_dim + orig_axis] + in_indexes[inp_idx];
        } else {
            ret[base_out_dim + orig_axis] = ret[base_out_dim + orig_axis]
                    + in_indexes[inp_idx]
                            * format.blocks_[blocks[axis2blocks[orig_axis]]];
            if (axis2blocks[orig_axis] == 0) {
                condition = condition
                        && ret[base_out_dim + orig_axis] < dim2unsigned(
                                   plain_dims[base_out_dim + orig_axis]);
            } else {
                condition = condition
                        && ret[base_out_dim + orig_axis]
                                < format.blocks_[blocks[axis2blocks[orig_axis]
                                        - 1]];
            }
        }
        axis2blocks[orig_axis]--; // next block
    }
    return ret;
}

static std::vector<expr> get_reorder_plain2block_indexes(
        const std::vector<expr> &in_indexes, const sc_data_format_t &format) {
    if (in_indexes.empty()) { return std::vector<expr>(); }
    COMPILE_ASSERT(format.format_code_ != format_kinds::any,
            "format can not be any in reorder op, please check it in layout "
            "propagation.");
    size_t base_out_dim = 0;
    size_t num_plain_dims = format.format_code_.norig_dims();
    size_t num_format_dims = format.format_code_.ndims();
    size_t num_out_dims = num_format_dims;
    std::vector<expr> ret(num_out_dims, 0);
    if (format.format_code_.is_batch_format()) {
        COMPILE_ASSERT(in_indexes.size() >= num_plain_dims,
                "Wrong number of dimensions for batch format: "
                        << format << ", real shape = "
                        << utils::print_vector(in_indexes));
        base_out_dim = in_indexes.size() - num_plain_dims;
        num_out_dims = base_out_dim + num_format_dims;
        ret.resize(num_out_dims, 0);
        for (size_t i = 0; i < base_out_dim; i++) {
            ret[i] = in_indexes[i];
        }
    } else {
        COMPILE_ASSERT(in_indexes.size() == num_plain_dims,
                "Wrong number of dimensions for format: "
                        << format << ", real shape = "
                        << utils::print_vector(in_indexes));
    };

    COMPILE_ASSERT(in_indexes.size() <= sc_data_format_kind_t::MAX_DIMS,
            "Too many dims in plain shapes");
    std::unordered_map<int, int>
            axis2blocks; // plain_axis to block idx, idx++ after an access
    std::unordered_map<int, expr> axis2index; // current index of blocking
    for (auto inp_idx = base_out_dim; inp_idx < num_plain_dims + base_out_dim;
            inp_idx++) {
        axis2index[inp_idx - base_out_dim] = in_indexes[inp_idx];
    }
    // format index
    for (auto out_idx = base_out_dim; out_idx < num_format_dims + base_out_dim;
            out_idx++) {
        auto orig_axis = format.format_code_.get(out_idx - base_out_dim);
        if (axis2blocks.find(orig_axis) == axis2blocks.end()) {
            axis2blocks[orig_axis] = 0;
        }
        auto blocks = format.format_code_.collect_blocking_index(orig_axis);
        int cur_block = 0;
        if (axis2blocks[orig_axis] >= (int)blocks.size()) {
            ret[out_idx] = axis2index[orig_axis];
        } else {
            cur_block = format.blocks_[blocks[axis2blocks[orig_axis]]];
            ret[out_idx] = axis2index[orig_axis] / cur_block;
            axis2index[orig_axis] = axis2index[orig_axis] % cur_block;
            axis2blocks[orig_axis]++; // next block
        }
    }
    return ret;
}

void compute_reorder_stride2stride(const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, any_map_t &attrs, size_t wkld = 0UL) {
    auto input = src.get_real_tensor();
    auto output = dst.get_real_tensor();
    int step = static_cast<int>(vectorize_step(ctx, dtype.type_code_));
    auto bld = builder::get_current_builder();
    std::vector<expr> iter_vars;
    std::vector<expr> in_indexes;
    std::vector<expr> loop_indexes;
    for (size_t i = 0; i < plain_dims.size(); i++) {
        iter_vars.emplace_back(builder::make_var(datatypes::index,
                std::string("_fuseiter") + std::to_string(idx++)));
        loop_indexes.emplace_back(iter_vars[i]);
        in_indexes.emplace_back(iter_vars[i] + src.get_offset()[i]);
    }
    std::vector<expr> out_indexes = get_reorder_stride2stride_indexes(
            in_indexes, input_format, output_format, plain_dims);

    auto cur = builder::make_stmts_unattached({builder::make_assign_unattached(
            builder::make_indexing(output, out_indexes),
            builder::make_indexing(src.tptr_, loop_indexes))});
    cur->attr()[op_traits::workload_computable_t::workload_number] = wkld;
    stmt body;
    std::vector<stmt> loops;
    for (int i = static_cast<int>(plain_dims.size()) - 1; i >= 0; i--) {
        body = cur.isa<stmts>()
                ? cur
                : make_stmt<stmts_node_t>(std::vector<stmt> {std::move(cur)});
        cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)), expr(0),
                src.get_shape()[i], expr(1), std::move(body), true,
                for_type::NORMAL);
        loops.push_back(cur);
    }
    std::reverse(loops.begin(), loops.end());
    for (size_t i = 0; i < plain_dims.size() - 2; i++) {
        loops[0].checked_as<for_loop>()->fuse(loops[i].checked_as<for_loop>());
    }
    bld->emit(cur);
}

void compute_reorder_block2stride(const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, any_map_t &attrs, size_t wkld = 0UL) {
    auto input = src.get_real_tensor();
    auto output = dst.get_real_tensor();
    int step = static_cast<int>(vectorize_step(ctx, dtype.type_code_));
    auto bld = builder::get_current_builder();
    auto input_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, input_format);
    auto output_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, output_format);
    assert(output_format.format_code_.ndims()
            == output_format.format_code_.norig_dims());
    auto output_last_origin_axis = output_format.format_code_.get(
            output_format.format_code_.ndims() - 1);
    auto block_axis = input_format.format_code_.collect_blocking_index(
            output_last_origin_axis); // {block_idx1, block_idx2,...}
    bool can_vectorize = !block_axis.empty()
            && block_axis.at(block_axis.size() - 1)
                    == input_format.get_blocks_size() - 1
            && input_blocking_dims[input_blocking_dims.size() - 1] % step == 0
            && plain_dims[plain_dims.size() - 1] % step == 0;
    if (can_vectorize && attrs.get_or_else(op_attr_key::no_fuse, false)) {
        int max_step = ctx->get_max_vector_lanes(dtype.type_code_);
        while (step < max_step
                && input_blocking_dims[input_blocking_dims.size() - 1]
                                % (2 * step)
                        == 0
                && plain_dims[plain_dims.size() - 1] % (2 * step) == 0) {
            step = 2 * step;
        }
    }
    bool no_padding = sc_data_format_t::get_padded_plain_shapes(
                              input_blocking_dims, input_format)
            == sc_data_format_t::get_padded_plain_shapes(
                    output_blocking_dims, output_format);

    // For no padding case, vectorize judgement should be more strictly for src
    // due to input maybe fused by previous op
    if (no_padding)
        can_vectorize = can_vectorize && src.get_shape().back().isa<constant>()
                && get_expr_as_int(src.get_shape().back()) % step == 0;

    step = can_vectorize ? step : 1;
    std::vector<expr> iter_vars;
    std::vector<expr> in_indexes;
    std::vector<expr> loop_indexes;
    for (size_t i = 0; i < input_blocking_dims.size(); i++) {
        iter_vars.emplace_back(builder::make_var(datatypes::index,
                std::string("_fuseiter") + std::to_string(idx++)));
        in_indexes.emplace_back(iter_vars[i] + src.get_offset()[i]);
        loop_indexes.emplace_back(iter_vars[i]);
    }
    expr condition;
    std::vector<expr> tmp_out_indexes = get_reorder_block2plain_indexes(
            in_indexes, input_format, plain_dims, condition);
    std::vector<expr> out_indexes
            = get_reorder_stride2stride_indexes(tmp_out_indexes,
                    output_format.to_plain(), output_format, plain_dims);

    auto assign
            = builder::make_stmts_unattached({builder::make_assign_unattached(
                    builder::make_indexing(output, out_indexes, step),
                    // here, use src.tptr instead of input is aimed to avoid
                    // input is tensor_view_op. Oherwisw, it will throw
                    // illegal exception in index_flatten
                    builder::make_indexing(
                            expr(src.tptr_), loop_indexes, step))});
    assign->attr()[op_traits::workload_computable_t::workload_number] = wkld;
    auto cur = no_padding
            ? assign
            : builder::make_if_else_unattached(condition, assign, stmt());
    stmt body;
    for (int i = static_cast<int>(input_blocking_dims.size()) - 1; i >= 0;
            i--) {
        body = cur.isa<stmts>()
                ? cur
                : make_stmt<stmts_node_t>(std::vector<stmt> {std::move(cur)});
        cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)), expr(0),
                src.get_shape()[i],
                i == static_cast<int>(input_blocking_dims.size()) - 1
                                && can_vectorize
                        ? expr(static_cast<int>(step))
                        : expr(1),
                std::move(body), true, for_type::NORMAL);
    }
    cur->attr()[stmt_attr_key::merge_loop] = true;
    bld->emit(cur);
}

void compute_reorder_stride2block(const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        size_t wkld = 0UL) {
    auto input = src.get_real_tensor();
    auto output = dst.get_real_tensor();
    int step = static_cast<int>(vectorize_step(ctx, dtype.type_code_));
    auto bld = builder::get_current_builder();
    auto input_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, input_format);
    auto output_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, output_format);
    assert(input_format.format_code_.ndims()
            == input_format.format_code_.norig_dims());
    auto input_last_origin_axis = input_format.format_code_.get(
            input_format.format_code_.ndims() - 1);
    auto block_axis = output_format.format_code_.collect_blocking_index(
            input_last_origin_axis); // {block_idx1, block_idx2,...}
    bool can_vectorize = !block_axis.empty()
            && block_axis.at(block_axis.size() - 1)
                    == output_format.get_blocks_size() - 1
            && output_blocking_dims[output_blocking_dims.size() - 1] % step == 0
            && plain_dims[plain_dims.size() - 1] % step == 0;
    bool no_padding = sc_data_format_t::get_padded_plain_shapes(
                              output_blocking_dims, output_format)
            == sc_data_format_t::get_padded_plain_shapes(
                    input_blocking_dims, input_format);

    // For no padding case, vectorize judgement should be more strictly for src
    // due to input maybe fused by previous op
    if (no_padding) {
        can_vectorize = can_vectorize && src.get_shape().back().isa<constant>()
                && get_expr_as_int(src.get_shape().back()) % step == 0;
    }

    if (can_vectorize && attrs.get_or_else(op_attr_key::no_fuse, false)) {
        int max_step = ctx->get_max_vector_lanes(dtype.type_code_);
        while (step < max_step
                && output_blocking_dims[output_blocking_dims.size() - 1]
                                % (2 * step)
                        == 0
                && plain_dims[plain_dims.size() - 1] % (2 * step) == 0) {
            step = 2 * step;
        }
    }

    step = can_vectorize ? step : 1;
    std::vector<expr> iter_vars;
    std::vector<expr> loop_indexes;
    if (!output_loop) {
        std::vector<expr> in_indexes;
        for (size_t i = 0; i < input_blocking_dims.size(); i++) {
            iter_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("_fuseiter") + std::to_string(idx++)));
            in_indexes.emplace_back(iter_vars[i] + src.get_offset()[i]);
            loop_indexes.emplace_back(iter_vars[i]);
        }
        expr condition;
        std::vector<expr> tmp_out_indexes = get_reorder_stride2stride_indexes(
                in_indexes, input_format, input_format.to_plain(), plain_dims);
        std::vector<expr> out_indexes = get_reorder_plain2block_indexes(
                tmp_out_indexes, output_format);

        auto assign = builder::make_stmts_unattached(
                {builder::make_assign_unattached(
                        builder::make_indexing(output, out_indexes, step),
                        // here, use src.tptr instead of input is aimed to avoid
                        // input is tensor_view_op. Oherwisw, it will throw
                        // illegal exception in index_flatten
                        builder::make_indexing(
                                src.tptr_, loop_indexes, step))});
        assign->attr()[op_traits::workload_computable_t::workload_number]
                = wkld;
        auto cur = assign;
        stmt body;
        for (int i = static_cast<int>(input_blocking_dims.size()) - 1; i >= 0;
                i--) {
            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0), src.get_shape()[i],
                    i == static_cast<int>(input_blocking_dims.size()) - 1
                                    && can_vectorize
                            ? expr(static_cast<int>(step))
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
        }
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    } else {
        std::vector<expr> out_indexes;
        for (size_t i = 0; i < output_blocking_dims.size(); i++) {
            iter_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("_fuseiter") + std::to_string(idx++)));
            out_indexes.emplace_back(iter_vars[i] + dst.get_offset()[i]);
        }
        expr condition;
        std::vector<expr> tmp_in_indexes = get_reorder_block2plain_indexes(
                out_indexes, output_format, plain_dims, condition);
        std::vector<expr> in_indexes
                = get_reorder_stride2stride_indexes(tmp_in_indexes,
                        input_format.to_plain(), input_format, plain_dims);

        auto assign = builder::make_stmts_unattached(
                {builder::make_assign_unattached(
                        builder::make_indexing(expr(output), out_indexes, step),
                        builder::make_indexing(
                                expr(input), in_indexes, step))});
        assign->attr()[op_traits::workload_computable_t::workload_number]
                = wkld;
        auto padding = builder::make_stmts_unattached(
                {builder::make_assign_unattached(
                        builder::make_indexing(output, out_indexes, step),
                        builder::make_constant({0UL},
                                sc_data_type_t(dtype.type_code_, step)))});
        auto cur = no_padding
                ? assign
                : builder::make_if_else_unattached(condition, assign, padding);
        stmt body;
        std::vector<stmt> loops;
        for (int i = static_cast<int>(output_blocking_dims.size()) - 1; i >= 0;
                i--) {
            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0),
                    no_padding ? dst.get_shape()[i]
                               : dim2unsigned(output_blocking_dims[i]),
                    i == static_cast<int>(output_blocking_dims.size()) - 1
                                    && can_vectorize
                            ? expr(static_cast<int>(step))
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
            loops.push_back(cur);
        }
        bld->emit(cur);
    }
}

void compute_reorder_block2block(const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims1, bool output_loop, any_map_t &attrs,
        size_t wkld = 0UL) {
    auto input = src.get_real_tensor();
    auto output = dst.get_real_tensor();
    int step = static_cast<int>(vectorize_step(ctx, dtype.type_code_));
    auto bld = builder::get_current_builder();
    // walk around for bert fusion
    sc_dims plain_dims(plain_dims1);
    if (plain_dims.empty()) {
        sc_dims dst_blocking_dims;
        for (int64_t i = 0; i < dst.nslice_dims(); i++) {
            dst_blocking_dims.push_back(get_const_as_int(
                    dst.get_shape()[i].checked_as<constant>()));
        }
        plain_dims = sc_data_format_t::get_padded_plain_shapes(
                dst_blocking_dims, output_format);
    }
    auto input_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, input_format);
    auto output_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, output_format);
    auto input_padded_plain_dims = sc_data_format_t::get_padded_plain_shapes(
            input_blocking_dims, input_format);
    auto output_padded_plain_dims = sc_data_format_t::get_padded_plain_shapes(
            output_blocking_dims, output_format);
    // plain axis of last block
    auto input_block_axis = input_format.format_code_.get(
            input_format.format_code_.ndims() - 1);
    auto output_block_axis = output_format.format_code_.get(
            output_format.format_code_.ndims() - 1);
    bool no_padding = input_padded_plain_dims == output_padded_plain_dims;
    bool can_vectorize = input_block_axis == output_block_axis
            && output_blocking_dims[output_blocking_dims.size() - 1] % step == 0
            && input_blocking_dims[input_blocking_dims.size() - 1] % step == 0
            && no_padding;

    // For no padding case, vectorize judgement should be more strictly for src
    // due to input maybe fused by previous op
    if (no_padding)
        can_vectorize = can_vectorize && src.get_shape().back().isa<constant>()
                && get_expr_as_int(src.get_shape().back()) % step == 0;

    step = can_vectorize ? step : 1;
    std::vector<expr> iter_vars;
    // for input loops
    if (!output_loop) {
        std::vector<expr> in_indexes;
        std::vector<expr> loop_indexes;
        for (size_t i = 0; i < input_blocking_dims.size(); i++) {
            iter_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("_fuseiter") + std::to_string(idx++)));
            in_indexes.emplace_back(iter_vars[i] + src.get_offset()[i]);
            loop_indexes.emplace_back(iter_vars[i]);
        }
        expr condition;
        std::vector<expr> tmp_indexes = get_reorder_block2plain_indexes(
                in_indexes, input_format, plain_dims, condition);
        std::vector<expr> out_indexes
                = get_reorder_plain2block_indexes(tmp_indexes, output_format);
        auto cur = builder::make_stmts_unattached(
                {builder::make_assign_unattached(
                        builder::make_indexing(output, out_indexes, step),
                        // here, use src.tptr instead of input is aimed to
                        // avoid input is tensor_view_op. Oherwisw, it will
                        // throw illegal exception in index_flatten
                        builder::make_indexing(
                                src.tptr_, loop_indexes, step))});
        cur->attr()[op_traits::workload_computable_t::workload_number] = wkld;
        stmt body;
        for (int i = static_cast<int>(input_blocking_dims.size()) - 1; i >= 0;
                i--) {
            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0), src.get_shape()[i],
                    i == static_cast<int>(input_blocking_dims.size()) - 1
                                    && can_vectorize
                            ? expr(static_cast<int>(step))
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
        }
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    } else {
        std::vector<expr> out_indexes;
        for (size_t i = 0; i < output_blocking_dims.size(); i++) {
            iter_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("_fuseiter") + std::to_string(idx++)));
            out_indexes.emplace_back(iter_vars[i] + dst.get_offset()[i]);
        }
        expr condition;
        std::vector<expr> tmp_indexes = get_reorder_block2plain_indexes(
                out_indexes, output_format, plain_dims, condition);
        std::vector<expr> in_indexes
                = get_reorder_plain2block_indexes(tmp_indexes, input_format);
        auto assign = builder::make_stmts_unattached(
                {builder::make_assign_unattached(
                        builder::make_indexing(output, out_indexes, step),
                        builder::make_indexing(input, in_indexes, step))});
        assign->attr()[op_traits::workload_computable_t::workload_number]
                = wkld;
        auto padding = builder::make_stmts_unattached(
                {builder::make_assign_unattached(
                        builder::make_indexing(output, out_indexes),
                        builder::make_constant({0UL},
                                sc_data_type_t(dtype.type_code_, step)))});
        auto cur = no_padding
                ? assign
                : builder::make_if_else_unattached(condition, assign, padding);
        stmt body;
        std::vector<stmt> loops;
        for (int i = static_cast<int>(output_blocking_dims.size()) - 1; i >= 0;
                i--) {
            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0),
                    no_padding ? dst.get_shape()[i]
                               : dim2unsigned(output_blocking_dims[i]),
                    i == static_cast<int>(output_blocking_dims.size()) - 1
                                    && can_vectorize
                            ? expr(static_cast<int>(step))
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
            loops.push_back(cur);
        }
        std::reverse(loops.begin(), loops.end());
        for (size_t i = 0; i < output_blocking_dims.size() - 2; i++) {
            loops[0].checked_as<for_loop>()->fuse(
                    loops[i].checked_as<for_loop>());
        }
        bld->emit(cur);
    }
}

void compute_reorder_block(const context_ptr &ctx, const tensor_slice &src,
        tensor_slice &dst, const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        size_t wkld = 0UL) {
    COMPILE_ASSERT(input_format.is_convertible(output_format),
            "Can not convert input format "
                    << input_format << " to output format " << output_format
                    << ".");
    if (is_not_blocking(input_format) && is_not_blocking(output_format)) {
        compute_reorder_stride2stride(ctx, src, dst, input_format,
                output_format, dtype, plain_dims, attrs, wkld);
    } else if (is_not_blocking(input_format) && output_format.is_blocking()) {
        compute_reorder_stride2block(ctx, src, dst, input_format, output_format,
                dtype, plain_dims, output_loop, attrs, wkld);
    } else if (input_format.is_blocking() && is_not_blocking(output_format)) {
        compute_reorder_block2stride(ctx, src, dst, input_format, output_format,
                dtype, plain_dims, attrs, wkld);
    } else if (input_format.is_blocking() && output_format.is_blocking()) {
        compute_reorder_block2block(ctx, src, dst, input_format, output_format,
                dtype, plain_dims, output_loop, attrs, wkld);
    } else {
        std::ostringstream ss;
        ss << "Unsupported data format. in = " << input_format
           << ", out = " << output_format;
        SC_WARN << ss.str();
        throw tuner_recoverable_exception_t(ss.str());
    }
}

size_t reorder_op_t::compute_workload(const std::vector<shape_dtype_pair> &ins,
        const std::vector<shape_dtype_pair> &outs) {
    return fusible_op_t::compute_workload(ins, outs)
            * workload_penalty_coefficient;
}

bool reorder_op_t::check_padding() const {
    auto input_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims_, input_format_);
    auto output_blocking_dims = sc_data_format_t::get_blocking_shapes(
            plain_dims_, output_format_);
    return sc_data_format_t::get_padded_plain_shapes(
                   input_blocking_dims, input_format_)
            != sc_data_format_t::get_padded_plain_shapes(
                    output_blocking_dims, output_format_);
}

bool reorder_op_t::use_output_loop() const {
    if (check_padding()) return true;
    if (attrs_.get_or_else(op_attr_key::no_fuse, false)) {
        if (!get_input_format().is_blocking()) return true;
    }
    if (attrs_.get_or_else(op_attr_key::break_pre_fuse, false)) return true;
    return false;
}

void reorder_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);
    compute_reorder_block(ctx, *inputs[0], *dst[0], input_format_,
            output_format_, info_.inputs_[0]->details_.dtype_, plain_dims_,
            use_output_loop(), attrs_, wkld);
}

// compute the output data format after reduction given the plain reduction
// axes
static sc_data_format_t get_reduced_format(
        const sc_data_format_t &in_fmt, const std::vector<int> &rd_axis) {
    auto base_fmt = in_fmt;
    // we should set the blocking of the reduce axies to 1
    for (int ax : rd_axis) {
        for (int blocking_idx :
                in_fmt.format_code_.collect_blocking_index(ax)) {
            base_fmt.blocks_[blocking_idx] = 1;
        }
    }
    return base_fmt;
}

reduce_op_t::reduce_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    COMPILE_ASSERT(ins.size() == 1, "Expecting 1 input for reduce_op_t");
    info_.inputs_ = ins;
    COMPILE_ASSERT(attrs.has_key("rd_axis") && attrs.has_key("rd_op"),
            "attrs should have reduce axis info.");
    plain_rd_axis_ = attrs.get<std::vector<int>>("rd_axis");
    rd_op_ = reduce_operator(attrs.get<int>("rd_op"));
    keep_dims_ = attrs.get_or_else("keep_dims", true);
    need_mean_ = attrs.get_or_else("need_mean", false);

    auto &old_reduce_dims = ins[0]->details_.get_plain_dims();
    std::sort(plain_rd_axis_.begin(), plain_rd_axis_.end());
    assert(plain_rd_axis_[plain_rd_axis_.size() - 1]
            < static_cast<int64_t>(old_reduce_dims.size()));
    // check duplicates
    bool duplicate
            = std::adjacent_find(plain_rd_axis_.begin(), plain_rd_axis_.end())
            != plain_rd_axis_.end();
    COMPILE_ASSERT(!duplicate, "duplicate axis found in rd_axis");
    sc_dims new_reduce_dims;
    new_reduce_dims.reserve(keep_dims_
                    ? old_reduce_dims.size()
                    : old_reduce_dims.size() - plain_rd_axis_.size());
    for (unsigned i = 0; i < old_reduce_dims.size(); i++) {
        bool is_reduction = std::find(plain_rd_axis_.begin(),
                                    plain_rd_axis_.end(), static_cast<int>(i))
                != plain_rd_axis_.end();
        if (is_reduction) {
            if (keep_dims_) { new_reduce_dims.push_back(1); }
        } else {
            new_reduce_dims.push_back(old_reduce_dims[i]);
        }
    }
    if (new_reduce_dims.empty()) new_reduce_dims.push_back(1);
    if (outs.empty()) {
        logical_tensor_t out;
        out = logical_tensor_t(get_reduced_format(ins[0]->details_.get_format(),
                                       plain_rd_axis_),
                new_reduce_dims, ins[0]->details_.dtype_);
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this, out));
    } else {
        COMPILE_ASSERT(outs.size() == 1, "Wrong op output size.\n");
        info_.outputs_ = outs;
    }
    auto &output = info_.outputs_[0];
    output->details_.dtype_ = info_.inputs_[0]->details_.dtype_;
    attrs_ = attrs;
    op_name_ = "reduce";
}

reduce_op_t::reduce_op_t(graph_tensor_ptr v, const std::string &rd_name,
        const std::vector<int> &rd_axis, reduce_operator rd_op, bool keep_dims,
        bool need_mean)
    : reduce_op_t({std::move(v)}, {},
            {{"rd_axis", rd_axis}, {"rd_op", static_cast<int>(rd_op)},
                    {"keep_dims", keep_dims}, {"need_mean", need_mean}}) {
    // default is need_allocate
    info_.tensor_share_info_ = {};
    rd_name_ = rd_name;
}

void reduce_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {
    if (keep_dims_) {
        const auto &in_fmt = info_.inputs_[0]->details_.get_format();
        out_formats.push_back({get_reduced_format(in_fmt, plain_rd_axis_)});
    } else {
        auto out_shape_size = info_.inputs_[0]->details_.get_plain_dims().size()
                - plain_rd_axis_.size();
        if (out_shape_size == 0) out_shape_size = 1;
        out_formats.push_back(
                {sc_data_format_t::get_plain_by_dims(out_shape_size)});
    }
}

void reduce_op_t::prepare_fusion_data(fdata_map &fdmap) {
    fdmap.get(info_.inputs_[0]).use_count_++;
    COMPILE_ASSERT(info_.inputs_.size() == 1, "Wrong op input size.\n");
    COMPILE_ASSERT(info_.outputs_.size() == 1, "Wrong op output size.\n");
    auto real_rd_axis = get_rd_axis();
    auto dim_size = info_.inputs_[0]->details_.get_blocking_dims().size();
    // check reduction axis legal
    COMPILE_ASSERT(real_rd_axis.size() <= dim_size,
            "reduction axis length should be less than input shape");
    COMPILE_ASSERT((*std::max_element(real_rd_axis.begin(), real_rd_axis.end())
                           <= static_cast<int64_t>(dim_size)),
            "Unexpected reduction axis found");
}

void reduce_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map = search_known_slice_ranges(this, fsmap);
    // set the other unknown slice range by achieved known_ranges_list
    slice_range_list known_ranges_list = known_ranges_map[0];
    // COMPILE_ASSERT(known_ranges_list.size() == 1,
    //         "Reduce Op should not accept inconsequent or irruglar
    //         slice");
    slice_range_list reduce_ranges_list;
    auto real_rd_axis = get_rd_axis();
    auto &src_dim = get_inputs()[0]->details_.get_blocking_dims();
    // check the slice range whether meet the least demand of reduce op
    for (auto &src_range : fsmap.get(get_inputs()[0])) {
        if (!slice_full_on_axes(src_dim, src_range, real_rd_axis)) {
            attrs_.set(
                    op_attr_key::fused_mode_hint, op_attr_key::break_pre_fuse);
            stat_map.append_ops_by_status(this, infer_status_code::RETRY);
        }
    }
    for (auto &known_ranges : known_ranges_list) {
        slice_range reduce_range;
        // additional process is needed.
        for (size_t i = 0; i < known_ranges.size(); i++) {
            if (real_rd_axis.end()
                    != std::find(real_rd_axis.begin(), real_rd_axis.end(), i)) {
                if (keep_dims_) {
                    reduce_range.emplace_back(std::pair<expr, expr> {0, 1});
                }
            } else {
                reduce_range.emplace_back(known_ranges.at(i));
            }
        }
        // reduce all and keep_dims = false;
        if (reduce_range.empty())
            reduce_range.emplace_back(std::pair<expr, expr> {0, 1});
        reduce_ranges_list.emplace_back(reduce_range);
    }

    fsmap.get(get_outputs()[0]) = std::move(reduce_ranges_list);
}

void reduce_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    auto &input = get_inputs()[0];
    auto &out_ranges = fsmap.get(get_outputs()[0]);
    auto &in_ranges = fsmap.get(input);
    auto real_rd_axis = get_rd_axis();
    if (in_ranges.empty()) {
        slice_range_list reduce_ranges_list;
        for (auto &range : out_ranges) {
            slice_range reduce_range;
            // additional process is needed.
            auto real_dims = get_inputs()[0]->details_.get_blocking_dims();
            // idx record real idx in range, used to skip range {0,1} when
            // keep_dims=false
            int idx = 0;
            for (size_t i = 0; i < real_dims.size(); i++) {
                if (real_rd_axis.end()
                        != std::find(
                                real_rd_axis.begin(), real_rd_axis.end(), i)) {
                    reduce_range.emplace_back(std::pair<expr, expr> {
                            0, dim2unsigned(real_dims.at(i))});
                    if (keep_dims_) idx++;
                } else {
                    reduce_range.emplace_back(range.at(idx++));
                }
            }
            reduce_ranges_list.emplace_back(reduce_range);
        }
        in_ranges = reduce_ranges_list;
        if (!this->isa<input_op>()) {
            input->producer_owner_->dyn_cast<fusible_op_t>()->pre_slice_ranges(
                    fsmap, stat_map);
        }
    }
}

// reduce all tensor_slice into sum, NOTE here src is a common
// tensor_slice but dst maybe whole temp_buffer because output shape of
// reduction is not equal to src, so it will allocate a new buffer
static void compute_block_reduce(const std::vector<const tensor_slice *> &src,
        const tensor_slice &dst, reduce_operator rd_op,
        std::vector<int> rd_axis, bool keep_dims, bool need_mean,
        const std::string &rd_name, const vectorized_info_t &vx_info,
        sc_data_type_t dtype, any_map_t &attrs, size_t wkld = 0UL) {
    // nested loop vars
    std::vector<expr> iter_vars;
    // the indices for multiple inputs. First dim: the input, Second
    // dim: the dimemsions in the tensor
    std::vector<std::vector<expr>> src_indices(src.size());
    // the indices for the output tensor
    std::vector<expr> dst_idx;
    // If last_axis can be reduce, we use current logic, but if last axis is
    // not in `rd_axis`, we can still use vectorization but not use
    // reduce_add.
    std::sort(rd_axis.begin(), rd_axis.end());
    bool last_axis_reduce = *rd_axis.rbegin()
            == static_cast<int>(src.at(0)->nslice_dims() - 1);

    /*** Unlike compute_xxx, compute_reduce only use src.ranges_
     * The final IR may look like below:
     * _for_(_fuseiter_i, 0, 1)
     *  sum = 0;
     *  _for_(_fuseiter_j, 0, 1)
     *   _for_(_fuseiter_k, 0, 1)
     *     sum += src[src_idx];
     *  dst[dst_idx] = sum(/num);
     * */
    // use src_indices.at(0) as default
    auto &src_idx = src_indices.at(0);
    int reduce_num = 1;
    for (unsigned i = 0; i < src.at(0)->nslice_dims(); i++) {
        iter_vars.emplace_back(builder::make_var(datatypes::index,
                std::string("_fuseiter") + std::to_string(idx++)));
        src_idx.emplace_back(iter_vars.back());
        if (rd_axis.end()
                != std::find(
                        rd_axis.begin(), rd_axis.end(), static_cast<int>(i))) {
            reduce_num *= get_const_as_int(
                    src.at(0)->get_shape().at(i).checked_as<constant>());
        }
    }
    // dst.ranges_ is equal to dst.tptr_->dims() in this case, because it
    // will be newly allocated.
    for (unsigned i = 0; i < src.at(0)->nslice_dims(); i++) {
        if (rd_axis.end() != std::find(rd_axis.begin(), rd_axis.end(), i)) {
            if (keep_dims) dst_idx.emplace_back(expr(0));
        } else {
            dst_idx.emplace_back(iter_vars.at(i));
        }
    }
    dst_idx = !dst_idx.empty() ? dst_idx : std::vector<expr> {expr {0}};
    expr indexed_target = builder::make_indexing(
            dst.tptr_, dst_idx, !last_axis_reduce ? vx_info.lanes : 1);
    expr indexed_input = builder::make_indexing(
            src.at(0)->tptr_, src_indices.at(0), vx_info.lanes);

    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder is set");
    stmt body, cur;
    auto reduce_value
            = builder::make_var(sc_data_type_t(dtype.type_code_, vx_info.lanes),
                    "reduce_" + rd_name + fusion_create_var_idx());
    auto asnode = make_stmt<assign_node_t>(reduce_value,
            make_expr<constant_node>((int64_t)0,
                    sc_data_type_t(dtype.type_code_, vx_info.lanes)));
    auto define_reduce
            = make_stmt<define_node_t>(reduce_value, linkage::local, expr());

    // because reduce_op_t use temp register to add up, for rightly write
    // back it may need to reorder reduction for-loop into inner-most
    // loop
    std::vector<int> new_loop_order = rd_axis;
    for (int64_t i = src.at(0)->nslice_dims() - 1; i >= 0; i--) {
        if (rd_axis.end() != std::find(rd_axis.begin(), rd_axis.end(), i))
            continue;
        else
            new_loop_order.insert(new_loop_order.begin(), i);
    }
    std::reverse(new_loop_order.begin(), new_loop_order.end());
    bool loop_reorder = false;
    int pre_ax = -1;
    for (auto ax : rd_axis) {
        if (pre_ax != -1) {
            if (ax != pre_ax + 1) {
                loop_reorder = true;
                break;
            }
        }
        pre_ax = ax;
    }
    for (auto i : new_loop_order) {
        if (i == new_loop_order.front()) {
            if (rd_op == reduce_operator::add) {
                if (need_mean) {
                    cur = make_stmt<assign_node_t>(reduce_value,
                            builder::make_fmadd(indexed_input,
                                    make_expr<constant_node>(1.0f / reduce_num,
                                            sc_data_type_t::f32(vx_info.lanes)),
                                    reduce_value));
                } else {
                    cur = make_stmt<assign_node_t>(reduce_value,
                            builder::make_add(indexed_input, reduce_value));
                }
            } else if (rd_op == reduce_operator::mul) {
                cur = make_stmt<assign_node_t>(reduce_value,
                        builder::make_mul(indexed_input, reduce_value));
            }
        }
        body = cur.isa<stmts>()
                ? cur
                : make_stmt<stmts_node_t>(std::vector<stmt> {std::move(cur)});
        cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)), expr(0),
                src.at(0)->get_shape().at(i),
                i == static_cast<int>(src.at(0)->nslice_dims() - 1)
                        ? expr(static_cast<int>(vx_info.lanes))
                        : expr(1),
                std::move(body), true, for_type::NORMAL);
        // the outer-most reduction axis
        if (i == rd_axis.front()) {
            if (rd_op == reduce_operator::add) {
                cur = make_stmt<stmts_node_t>(std::vector<stmt> {define_reduce,
                        asnode, std::move(cur),
                        make_stmt<assign_node_t>(indexed_target,
                                vx_info.lanes > 1 && last_axis_reduce
                                        ? builder::make_reduce_add(reduce_value)
                                        : reduce_value)});
            } else if (rd_op == reduce_operator::mul) {
                cur = make_stmt<stmts_node_t>(std::vector<stmt> {define_reduce,
                        asnode, std::move(cur),
                        make_stmt<assign_node_t>(indexed_target,
                                need_mean
                                        ? (vx_info.lanes > 1 && last_axis_reduce
                                                          ? builder::make_reduce_mul( // NOLINT
                                                                  reduce_value)
                                                          : reduce_value)
                                                / reduce_num
                                        : (vx_info.lanes > 1
                                                  && last_axis_reduce)
                                                ? builder::make_reduce_mul(
                                                        reduce_value)
                                                : reduce_value)});
            }
            cur->attr()[op_traits::workload_computable_t::workload_number]
                    = wkld;
            // try to create inner anchor for reduce op
            create_fusible_output_anchor(
                    cur, dst, iter_vars, {rd_axis}, vx_info, attrs);
        }
    }
    // set merge_loop attr
    if (!loop_reorder) cur->attr()[stmt_attr_key::merge_loop] = true;
    bld->emit(cur);
}

std::vector<int> reduce_op_t::get_rd_axis() const {
    auto fmt = info_.inputs_[0]->details_.get_format();
    int bs_ndim = 0;
    if (fmt.format_code_.is_batch_format()) {
        bs_ndim = static_cast<int>(
                          info_.inputs_[0]->details_.get_blocking_dims().size())
                - fmt.format_code_.ndims();
    }
    return transform_axis_plain2blocking(fmt, plain_rd_axis_, bs_ndim);
}

void reduce_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);
    // original rd_axis may be modified during layout_propagation pass, so
    // we need to call `get_rd_axis()` to get real reduce axis
    auto real_rd_axis = get_rd_axis();
    // set default vectorized information
    vx_info_.axis = dst[0]->get_shape().size() - 1;
    vx_info_.lanes = 1;
    // TODO(xxx): need more detailed judgement for `last_dim = 1` case
    int last_dim = get_const_as_int(
            inputs[0]->get_shape().back().checked_as<constant_c>());
    auto vector_lanes
            = vectorize_step(ctx, info_.inputs_[0]->details_.dtype_.type_code_);
    if (last_dim / vector_lanes && last_dim % vector_lanes == 0) {
        vx_info_.lanes = vector_lanes;
    }

    compute_block_reduce(inputs, *dst[0], rd_op_, real_rd_axis, keep_dims_,
            need_mean_, rd_name_, vx_info_, info_.inputs_[0]->details_.dtype_,
            attrs_, wkld);
}

size_t reduce_op_t::compute_workload(const std::vector<shape_dtype_pair> &ins,
        const std::vector<shape_dtype_pair> &outs) {
    auto &shape = ins[0].first;
    auto &dtype = ins[0].second;
    auto real_rd_axis = get_rd_axis();
    size_t wkld = utils::get_sizeof_type(dtype) * read_weight;
    for (auto &rd_axis : real_rd_axis) {
        wkld *= shape[rd_axis];
    }
    wkld += utils::get_sizeof_type(dtype) * write_weight;
    wkld *= workload_penalty_coefficient;
    return wkld;
}

OP_REGISTER(transpose_op_t, transpose)
OP_REGISTER(add_op_t, add)
OP_REGISTER(mul_op_t, mul)
OP_REGISTER(sub_op_t, sub)
OP_REGISTER(div_op_t, div)
OP_REGISTER(min_op_t, min)
OP_REGISTER(max_op_t, max)
OP_REGISTER(tensor_view_op_t, tensor_view)
OP_REGISTER(reshape_op_t, reshape)
OP_REGISTER(reorder_op_t, reorder)
OP_REGISTER(constant_op_t, constant)
OP_REGISTER(reduce_op_t, reduce)

} // namespace sc
