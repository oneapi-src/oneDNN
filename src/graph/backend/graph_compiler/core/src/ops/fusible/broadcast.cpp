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

#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "compiler/ir/builder.hpp"
#include "compiler/ir/graph/fusible_op_utils.hpp"
#include "compiler/ir/graph/trait/may_broadcast.hpp"
#include "ops/fusible/broadcast.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/*
Check whether it strictly follows auto_broadcast rule
e.g. [1, 2, 3, 4] & [1, 4] ==> OK
e.g. [1, 2, 3, 4] & [2] ==> FAIL
e.g. [1, 2, 3, 4] & [2, 3, 1] ==> OK
*/
static bool is_auto_broadcast(
        const sc_dims &input_shape, const sc_dims &output_shape) {
    const size_t input_rank = input_shape.size();
    const size_t output_rank = output_shape.size();
    const size_t offset = output_rank - input_rank;
    for (int i = input_rank - 1; i >= 0; --i) {
        // TODO(yifei): consider whether input_shape[i] != 1 is necessary
        // here
        if (input_shape[i] != 1 && input_shape[i] != output_shape[i + offset]) {
            return false;
        }
    }
    return true;
}

broadcast_op_t::broadcast_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    info_.inputs_ = ins;
    op_name_ = "broadcast";
    attrs_ = attrs;
    COMPILE_ASSERT(attrs_.has_key("output_shape"),
            "output_shape must be specified for broadcast op.");
    output_shape_ = attrs_.has_key("output_shape")
            ? attrs_.get<sc_dims>("output_shape")
            : outs[0]->details_.get_plain_dims();
    // when bc_axis is explicitly specified by users, it will overshadow
    // auto_broadcast rules
    plain_bc_axis_ = attrs_.get_or_else("bc_axis", std::vector<int> {});
    const auto &input_shape = info_.inputs_[0]->details_.get_plain_dims();
    if (plain_bc_axis_.empty()) {
        plain_bc_axis_ = op_traits::may_broadcast_t::get_auto_broadcast_bc_axis(
                input_shape, output_shape_);
    }
    if (outs.empty()) {
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, sc_data_format_t(),
                        output_shape_, info_.inputs_[0]->details_.dtype_));
    } else {
        info_.outputs_ = outs;
        COMPILE_ASSERT(output_shape_ == outs[0]->details_.get_plain_dims(),
                "output_shape attribute shall be consistent with specified "
                "output.");
    }
}

broadcast_op_t::broadcast_op_t(graph_tensor_ptr v,
        std::vector<int> &output_shape, std::vector<int> &bc_axis)
    : broadcast_op_t({std::move(v)}, {},
            {{"output_shape", output_shape}, {"bc_axis", bc_axis}}) {}

shape_rl_vec broadcast_op_t::get_dynamic_shape_relations() const {
    shape_rl_vec ret;
    auto &in_plain_dims = get_inputs()[0]->details_.get_plain_dims();
    auto &out_plain_dims = get_outputs()[0]->details_.get_plain_dims();
    assert(in_plain_dims.size() == out_plain_dims.size()
            || in_plain_dims.size() == 1);
    if (in_plain_dims.size() == out_plain_dims.size()) {
        for (size_t i = 0; i < in_plain_dims.size(); i++) {
            // maybe broadcast
            if (is_dynamic_dim(in_plain_dims[i])) {
                ret.emplace_back(in_plain_dims[i], out_plain_dims[i]);
            }
        }
    }
    return ret;
}

void broadcast_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    const auto &in_format = info_.inputs_[0]->details_.get_format();
    auto input_dims = info_.inputs_[0]->details_.get_plain_dims();
    auto output_dims = info_.outputs_[0]->details_.get_plain_dims();
    sc_data_format_t out_format;
    if (input_dims.size() != output_dims.size()) {
        COMPILE_ASSERT(in_format == sc_data_format_t(format_kinds::A)
                        && input_dims == sc_dims {1},
                "Unsupported format encountered in broadcast op's query "
                "format.");
        std::vector<int> storage_args(output_dims.size(), -1);
        std::iota(storage_args.begin(), storage_args.end(), 0);
        out_format = sc_data_format_t(storage_args, in_format.blocks_);
    } else {
        out_format = in_format;
    }
    in_formats.push_back({in_format});
    out_formats.push_back({out_format});
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

void broadcast_op_t::prepare_fusion_data(fdata_map &fdmap) {}

std::vector<int> broadcast_op_t::get_bc_axis() const {
    if (plain_bc_axis_ == std::vector<int> {-1}) return plain_bc_axis_;
    return transform_axis_plain2blocking(info_.outputs_[0], plain_bc_axis_);
}

void broadcast_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map
            = search_known_slice_ranges(this, fsmap, stat_map);
    if (known_ranges_map.empty()) return;

    slice_range_list known_ranges_list = known_ranges_map[0];
    // derive outputs slice range
    auto out_dims = this->get_outputs()[0]->details_.get_blocking_dims();
    auto in_dims = this->get_inputs()[0]->details_.get_blocking_dims();
    auto blocking_bc_axis = get_bc_axis();
    slice_range_list ranges_list(known_ranges_list.size());
    size_t dim_diff = out_dims.size() - in_dims.size();
    for (size_t i = 0; i < known_ranges_list.size(); i++) {
        auto &known_range = known_ranges_list[i];
        COMPILE_ASSERT(known_range.size() == in_dims.size(),
                "Input's known_range shall have same length as in_dims.")
        for (size_t j = 0; j < out_dims.size(); j++) {
            if (std::find(blocking_bc_axis.begin(), blocking_bc_axis.end(), j)
                    != blocking_bc_axis.end()) {
                ranges_list[i].emplace_back(known_range[j - dim_diff]);
            } else {
                ranges_list[i].emplace_back(
                        expr(0), expr(dim2unsigned(out_dims[j])));
            }
        }
    }
    fsmap.get(this->get_outputs()[0]) = ranges_list;
}

void broadcast_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    auto &outslice = fsmap.get(get_outputs()[0]);
    if (outslice.empty()) {
        stat_map.append_ops_by_status(this, infer_status_code::RETRY);
        return;
    }
    auto &input_slice = fsmap.get(get_inputs()[0]);
    if (input_slice.empty()) {
        auto out_dims = this->get_outputs()[0]->details_.get_blocking_dims();
        auto in_dims = this->get_inputs()[0]->details_.get_blocking_dims();
        slice_range_list ranges_list(in_dims.size());
        size_t dim_diff = out_dims.size() - in_dims.size();
        for (size_t i = 0; i < in_dims.size(); i++) {
            if (out_dims[i + dim_diff] == in_dims[i]) {
                ranges_list[i] = outslice[i + dim_diff];
            } else {
                ranges_list[i].emplace_back(expr(0), expr(1));
            }
        }
        if (stat_map.is_recursive_mode()) {
            get_inputs()[0]
                    ->producer_owner_->dyn_cast<fusible_op_t>()
                    ->pre_slice_ranges(fsmap, stat_map);
        }
    }
}

void broadcast_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);
    // set default vectorized information
    vx_info_.axis = dst[0]->nslice_dims() - 1;
    for (int64_t i = dst[0]->nslice_dims() - 1; i >= 0; --i) {
        auto cur_dim = dst[0]->get_shape()[i];
        if (!cur_dim.isa<constant>()
                || get_const_as_int(cur_dim.checked_as<constant>()) > 1) {
            vx_info_.axis = i;
            break;
        }
    }
    vx_info_.lanes
            = vectorize_step(ctx, info_.inputs_[0]->details_.dtype_.type_code_);

    // get blocking dims
    auto out_dims = get_outputs()[0]->details_.get_blocking_dims();
    auto in_dims = get_inputs()[0]->details_.get_blocking_dims();
    auto bc_axis = get_bc_axis();
    // validate whether out_dims and in_dims match
    COMPILE_ASSERT(is_auto_broadcast(in_dims, out_dims),
            "Broadcast op's compute_block requires the blocking shape of input "
            "& output strictly follow the auto-broadcast rules; if the "
            "assertion here failed, please double check specified layouts.");
    COMPILE_ASSERT(bc_axis == std::vector<int> {-1}
                    || in_dims.size() == out_dims.size(),
            "Broadcast op's compute_block encounter unsupported case.");
    // define assign
    auto assign = [&](const std::vector<expr> &in,
                          std::vector<expr::lvalue_proxy_t> &out) -> stmt {
        return builder::make_assign_unattached(out[0], in[0]);
    };
    // auto assign = mask_compute_func_t(func);

    //  enable vectorized code
    bool use_vectorized = false;
    vec_backend_require(ctx, use_vectorized);
    // nested loop vars
    std::vector<expr> iter_vars;
    // the indices for in & dst
    std::vector<expr> in_idx, dst_idx;
    const tensor_slice *in_tsl = inputs[0], *dst_tsl = dst[0];

    // use src_indices.at(0) as default
    for (unsigned i = 0; i < dst_tsl->nslice_dims(); i++) {
        // make the loop var for the for-loop
        iter_vars.emplace_back(range_from_outer_loop(dst_tsl->get_ranges()[i])
                        ? expr(0)
                        : builder::make_var(datatypes::index,
                                std::string("_fuseiter")
                                        + fusion_create_idx()));
        dst_idx.emplace_back(iter_vars.back());
        if (std::find(bc_axis.begin(), bc_axis.end(), i) != bc_axis.end()) {
            in_idx.emplace_back(iter_vars.back());
        } else if (bc_axis != std::vector<int> {-1}) {
            in_idx.emplace_back(0);
        }
    }
    // for empty bc_axis
    if (in_idx.empty()) in_idx = {0};

    std::vector<expr> in_idx_tail = in_idx, dst_idx_tail = dst_idx;
    auto tail_var = builder::make_var(
            datatypes::index, std::string("_fuseiter") + fusion_create_idx());
    dst_idx_tail[vx_info_.axis] = tail_var;

    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder set");
    auto slice_len = dst_tsl->get_shape().at(vx_info_.axis);
    int lanes = static_cast<int>(vx_info_.lanes);
    auto floor = do_cast_and_fold(slice_len / lanes * lanes);
    auto tail = do_cast_and_fold(slice_len % lanes);
    int floor_int = 0;
    int tail_int = 0;
    int floor_len
            = get_const_as_int(slice_len.static_as<constant>()) / lanes * lanes;
    int tail_len = get_const_as_int(slice_len.static_as<constant>()) % lanes;
    if (floor.isa<constant>()) {
        floor_int = get_expr_as_int(floor);
        tail_int = get_expr_as_int(tail);
        COMPILE_ASSERT(
                (floor_int + tail_int), "Don't support shape len == 0 cases.");
    }

    auto last_axis = expr(floor + tail);

    // threshold logic is the same as binary op
    bool tail_threshold = tail.isa<constant>() && tail_int <= 1;
    bool last_dim_eq_1 = tail.isa<constant>() && tail_int == 1;
    bool use_scalar = tail_threshold || !use_vectorized || lanes == 1;

    auto func_indexing_input = [&](std::vector<expr> &in_idx,
                                       expr &indexed_input, expr &iter_var,
                                       bool use_scalar = false,
                                       bool has_tail = false) {
        if (bc_axis.back() == static_cast<int64_t>(vx_info_.axis)) {
            indexing_from_diff_cond(use_scalar, has_tail, *in_tsl, in_idx,
                    lanes, indexed_input, slice_len, iter_var, floor);
        } else {
            if (use_scalar) {
                indexed_input = builder::make_indexing(in_tsl->tptr_, in_idx);
            } else {
                indexed_input = builder::make_broadcast(
                        builder::make_indexing(in_tsl->tptr_, in_idx), lanes);
            }
        }
    };

    std::vector<stmt> tcur;
    stmt cur;
    // recover schedule loop
    for (int i = static_cast<int>(dst_tsl->get_shape().size() - 1); i >= 0;
            i--) {
        stmt body;
        // do vectorize on the last dim
        if (static_cast<int>(dst_tsl->get_shape().size()) == vx_info_.axis + 1
                && i == vx_info_.axis) {
            if (!floor.isa<constant>() || floor_int) {
                expr indexed_input, indexed_target;
                indexing_from_diff_cond(false, false, *dst_tsl, dst_idx, lanes,
                        indexed_target, slice_len, iter_vars.at(i), floor);
                func_indexing_input(in_idx, indexed_input, iter_vars.at(i));
                bld->push_scope();
                std::vector<expr::lvalue_proxy_t> target_vec {
                        expr::lvalue_proxy_t(indexed_target, false)};
                cur = assign(std::vector<expr> {indexed_input}, target_vec);
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                cur = bld->pop_scope();
                if (iter_vars.at(i).isa<var>()) {
                    cur = make_stmt<for_loop_node_t>(iter_vars.at(i), expr(0),
                            expr(floor), expr(lanes), cur, true,
                            for_type::NORMAL);
                }
                tcur.emplace_back(cur);
            }
            if (!tail.isa<constant>() || tail_int) {
                expr indexed_target_tail, indexed_input_tail;
                if (std::find(bc_axis.begin(), bc_axis.end(), i)
                        != bc_axis.end()) {
                    in_idx_tail[vx_info_.axis] = tail_var;
                }
                indexing_from_diff_cond(use_scalar, true, *dst_tsl,
                        dst_idx_tail, lanes, indexed_target_tail, slice_len,
                        tail_var, floor, true);
                func_indexing_input(in_idx_tail, indexed_input_tail, tail_var,
                        use_scalar, true);
                std::vector<expr::lvalue_proxy_t> target_vec_tail {
                        expr::lvalue_proxy_t(indexed_target_tail, false)};
                bld->push_scope();
                cur = assign(std::vector<expr> {indexed_input_tail},
                        target_vec_tail);
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                cur = make_stmt<for_loop_node_t>(tail_var, expr(floor),
                        do_cast_and_fold(floor + tail),
                        use_scalar ? expr(1) : lanes, bld->pop_scope(), true,
                        for_type::NORMAL);
                tcur.emplace_back(cur);
            }
        } else if (iter_vars.at(i).isa<var>()) {
            if (!tcur.empty() && tcur[0].defined()) {
                body = make_stmt<stmts_node_t>(std::move(tcur));
                tcur.clear();
                // address special condition, like temp_buffer is used
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0), dst_tsl->get_shape().at(i), expr(1),
                        std::move(body), true, for_type::NORMAL);
            } else if (cur.defined()) {
                body = make_stmt<stmts_node_t>(
                        std::vector<stmt> {std::move(cur)});
                // address special condition, like temp_buffer is used
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0), dst_tsl->get_shape().at(i), expr(1),
                        std::move(body), true, for_type::NORMAL);
            } else {
                // if cur not defined, means last axis of tensor slice has range
                // 1, e.g. tensor_slice{{i, 100},{0, 1}}
                expr indexed_target, indexed_input;
                indexed_target
                        = builder::make_indexing(dst_tsl->tptr_, dst_idx);
                indexed_input = builder::make_indexing(in_tsl->tptr_, in_idx);
                std::vector<expr::lvalue_proxy_t> target_vec {
                        expr::lvalue_proxy_t(indexed_target, false)};
                bld->push_scope();
                cur = assign(std::vector<expr> {indexed_input}, target_vec);
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0), dst_tsl->get_shape().at(i), expr(1),
                        bld->pop_scope(), true, for_type::NORMAL);
            }
        }
    }
    if (!tcur.empty() && tcur[0].defined()) {
        // TODO(xxx): currenly we don't add merge_loop attribute for this
        // special case, need stronger loop analysis.
        for (auto &it : tcur) {
            bld->emit(it);
        }
        // TODO(yifei): analyze whether this is safe enough
        cur->attr()[stmt_attr_key::merge_loop] = true;
    } else {
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    }
}

size_t broadcast_op_t::compute_workload(
        const std::vector<shape_dtype_pair> &ins,
        const std::vector<shape_dtype_pair> &outs) {
    // compute_workload(outs, outs) instead of compute_workload(ins, outs)
    // because broadcast op involves reading single in for multiple times
    // while writing the result to outs
    return fusible_op_t::compute_workload(outs, outs)
            * workload_penalty_coefficient;
}

OP_REGISTER(broadcast_op_t, broadcast)

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
