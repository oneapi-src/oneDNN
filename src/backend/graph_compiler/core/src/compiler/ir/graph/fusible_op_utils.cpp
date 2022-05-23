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

#include <assert.h>
#include <atomic>
#include <unordered_map>

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include "fusible_op.hpp"
#include "fusion_mgr.hpp"
#include "outer_loop_generator.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
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

std::string fusion_create_idx() {
    return std::string("_") + std::to_string(idx++);
}

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
        std::vector<expr> strides
                = dims_to_expr(data[i]->details_.get_strides());
        expr aexpr = builder::make_stensor(tensor_name + std::to_string(i),
                dims, strides, data[i]->details_.dtype_);
        flattened.emplace_back(aexpr);
        expected.emplace_back(tensor_slice(aexpr));
    }
    return expected;
}

ir_module_ptr fusible_op_get_func(fusible_op_t *op, outer_loop_generator_t &gen,
        const context_ptr &ctx, bool check_parallel) {
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

stmt mask_compute_func_t::operator()(const std::vector<expr> &in,
        std::vector<expr::lvalue_proxy_t> &out, const expr &cur_idx,
        const expr &upper_bound, uint32_t lanes) const {
    auto ret = impl_(in, out);
    if (cur_idx.defined() && upper_bound.defined()) {
        auto bld = builder::get_current_builder();
        bld->emit(ret);
        return builder::make_assign_unattached(out[0],
                make_select_by_mask(out[0], cur_idx, upper_bound, lanes));
    }
    return ret;
}

expr make_select_by_mask(const expr &lhs_vec, const expr &cur_index,
        const expr &upper_bound, uint32_t lanes) {
    auto bld = builder::get_current_builder();
    auto offset = builder::make_cast(datatypes::s32, upper_bound)
            - builder::make_cast(datatypes::s32, cur_index);
    offset = static_cast<int>(lanes)
            - builder::make_max(
                    0, builder::make_min(static_cast<int>(lanes), offset));
    sc_data_type_t var_dtype;
    uint64_t init_value;
    switch (lanes) {
        case 4: {
            var_dtype = datatypes::u8;
            init_value = std::numeric_limits<uint8_t>::max();
            break;
        }
        case 8: {
            var_dtype = datatypes::u8;
            init_value = std::numeric_limits<uint8_t>::max();
            break;
        }
        case 16: {
            var_dtype = datatypes::u16;
            init_value = std::numeric_limits<uint16_t>::max();
            break;
        }
        case 32: {
            var_dtype = datatypes::s32;
            init_value = std::numeric_limits<uint32_t>::max();
            break;
        }
        case 64: {
            var_dtype = datatypes::index;
            init_value = std::numeric_limits<uint64_t>::max();
            break;
        }
        default: COMPILE_ASSERT(false, "invalid lanes: " << lanes);
    }
    auto mask = builder::make_var(
            var_dtype, "__mask_" + std::to_string(var_idx++));
    auto def = builder::make_var_tensor_def_unattached(mask, linkage::local,
            builder::make_constant({init_value}, var_dtype));
    auto assign = builder::make_assign_unattached(
            mask, mask >> builder::make_cast(var_dtype, offset));
    bld->emit(def);
    bld->emit(assign);
    expr rhs_vec = make_expr<constant_node>(
            std::vector<union_val>(lanes, UINT64_C(0)),
            sc_data_type_t(lhs_vec->dtype_.type_code_, lanes));
    return builder::make_select(mask, lhs_vec, rhs_vec);
}

/** Determine whether masks are needed during elementwise computation and
 * generate conditional expressions for the mask
 * @param src input slice
 * @param plain_dims plain shapes
 * @param format input format
 * @param iter_vars input loop vars
 * @param lanes simd lanes
 * @param condition key is related iter var, value is two exprs: first is
 * current accumulated index, second is its plain shape upperbound.
 * @param last_axis_mask mask count, how many elements should be computed in
 * this time. -1 means all.
 * */
void compute_mask_and_generate_condition(
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
                = iter_vars[block_dim] + offset[block_dim];
        for (int b = static_cast<int>(blocks.size()) - 1; b >= 0; b--) {
            if (b > 0 && blocks[b - 1] % blocks[b] != 0) { padding_count++; }
            conditions[iter_vars[block_dim]].first
                    = conditions[iter_vars[block_dim]].first
                    + (iter_vars[plain2block[i][b]] + offset[plain2block[i][b]])
                            * format.blocks_[blocks[b]];
        }
        conditions[iter_vars[block_dim]].second
                = dim2unsigned(plain_dims[orig_dim]);
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
            if (!range[j].first.isa<constant>()) return;
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

void create_fusible_output_anchor(stmt &parent, const tensor_slice &dst,
        const std::vector<expr> &loop_vars,
        const std::vector<int> &anchor_pos_in_loop,
        const vectorized_info_t &vx_info, any_map_t &attrs) {
    std::vector<stmt> ss = parent.isa<stmts>() ? parent.static_as<stmts>()->seq_
                                               : std::vector<stmt> {parent};
    create_fusible_output_anchor(
            ss, dst, loop_vars, anchor_pos_in_loop, vx_info, attrs);
    parent = make_stmt<stmts_node_t>(std::move(ss));
}

void compute_vectorized_op(const std::vector<const tensor_slice *> &src,
        const tensor_slice &dst, sc_op_info_t &info,
        const vectorized_info_t &vx_info,
        const mask_compute_func_t &compute_lanes,
        const mask_compute_func_t &compute_scalar, any_map_t &attrs,
        size_t wkld, bool use_mask, const tensor_slice *expand_loop_by) {
    if (!expand_loop_by) { expand_loop_by = &dst; }
    // nested loop vars
    std::vector<expr> iter_vars;
    // the indices for multiple inputs. First dim: the input, Second dim:
    // the dimemsions in the tensor
    std::vector<std::vector<expr>> src_indices_floor(src.size());
    std::vector<std::vector<expr>> src_indices_tail(src.size());
    // the indices for the output tensor
    std::vector<expr> dst_idx_floor;
    std::vector<expr> dst_idx_tail;
    for (unsigned i = 0; i < expand_loop_by->nslice_dims(); i++) {
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
            expand_loop_by->get_shape().at(vx_info.axis).static_as<constant>());
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
    if (last_axis_mask != -1 && floor > 0) {
        COMPILE_ASSERT(tail == 0,
                "Currently we only support mask in vectorize compute not "
                "tail.");
    }
    std::vector<stmt> tcur;
    stmt cur;
    int loop_size = static_cast<int>(expand_loop_by->get_shape().size());
    // recover schedule loop
    for (int i = loop_size - 1; i >= 0; i--) {
        stmt body;
        // currently vx_axis should be last axis
        if (loop_size == vx_info.axis + 1 && i == vx_info.axis) {
            if (floor) {
                bld->push_scope();
                auto cond_it = conditions.find(iter_vars[i]);
                if (cond_it != conditions.end()) {
                    assert(last_axis_mask != -1);
                    cur = compute_lanes(indexed_input_floor, target_floor,
                            cond_it->second.first, cond_it->second.second,
                            vx_info.lanes);
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
                        expr(0), expand_loop_by->get_shape().at(i), expr(1),
                        std::move(body), true, for_type::NORMAL);
            } else if (cur.defined()) {
                body = make_stmt<stmts_node_t>(
                        std::vector<stmt> {std::move(cur)});
                // address special condition, like temp_buffer is used
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0), expand_loop_by->get_shape().at(i), expr(1),
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
                        expand_loop_by->get_shape().at(i), expr(1),
                        bld->pop_scope(), true, for_type::NORMAL);
            }
        }
    }
    if (!tcur.empty() && tcur[0].defined()) {
        assert(expand_loop_by->get_shape().size() == 1UL);
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

size_t get_dims_product(const sc_dims &dims) {
    sc_dim ret = 1;
    for (unsigned i = 0; i < dims.size(); ++i) {
        ret *= dims[i];
    }
    assert(ret > 0 && "Overflow or non-constant shape detected");
    return ret;
}

bool loop_can_be_fused(const for_loop &loop) {
    return get_expr_as_int(loop->step_) == INT64_C(1);
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
        const slice_range_map &known_ranges_map, fslice_map &fsmap,
        infer_status_map_t &stat_map) {
    // set other unknown ranges.
    auto input_size = cur->get_inputs().size();
    for (size_t i = 0; i < input_size; i++) {
        auto input = cur->get_inputs()[i];
        auto &inp_slice = fsmap.get(input);
        if (input->producer_owner_->isa<input_op>()
                && input->producer_owner_->dyn_cast<input_op>()
                           ->is_arg_input()) {
            inp_slice = known_ranges_map.find(i)->second;
        } else {
            if (inp_slice.empty()) {
                inp_slice = known_ranges_map.find(i)->second;
                input->producer_owner_->dyn_cast<fusible_op_t>()
                        ->pre_slice_ranges(fsmap, stat_map);
            }
        }
    }
}

} // namespace sc
