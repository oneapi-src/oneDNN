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

#include <assert.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include "ternary_elemwise.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <unordered_map>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

std::vector<std::pair<int, std::vector<tensor_inplace_info_t>>>
select_op_t::get_inplace_map() {
    std::vector<tensor_inplace_info_t> ret;
    auto &inp = get_inputs();
    auto &out_dim = get_outputs()[0]->details_.get_plain_dims();
    auto &out_dtype = get_outputs()[0]->details_.dtype_;
    for (size_t i = 0; i < inp.size(); i++) {
        if (inp[i]->details_.get_plain_dims() == out_dim
                && inp[i]->details_.dtype_ == out_dtype) {
            ret.emplace_back(tensor_inplace_info_t {
                    static_cast<int>(i), inplace_kind::ZERO_OFFSET});
        }
    }
    if (ret.empty()) { return {}; }
    return {{0, std::move(ret)}};
}

static slice_range_list infer_broadcast_slice(slice_range_list known_range_list,
        const std::vector<int> &bc_axis, const std::vector<expr> &bc_dim) {
    slice_range_list bc_range_list(known_range_list.size());
    for (size_t i = 0; i < bc_range_list.size(); i++) {
        auto &known_range = known_range_list[i];
        COMPILE_ASSERT(known_range.size() == bc_dim.size()
                        || bc_axis == std::vector<int> {-1},
                "Unexpected cases found")
        for (size_t j = 0; j < known_range.size(); j++) {
            if (bc_axis.end() != std::find(bc_axis.begin(), bc_axis.end(), j)) {
                bc_range_list[i].emplace_back(known_range.at(j));
            } else if (bc_dim.size() != 1) {
                bc_range_list[i].emplace_back(
                        std::make_pair(expr(0), bc_dim[j]));
            }
        }
    }
    return bc_range_list;
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

static bound_axis infer_broadcast_axis_binding(
        bound_axis known_axis_list, const std::vector<int> &bc_axis) {
    bound_axis bc_axis_list(known_axis_list.size());
    for (size_t i = 0; i < bc_axis_list.size(); i++) {
        auto &known_ax = known_axis_list[i];
        for (size_t j = 0; j < known_ax.size(); j++) {
            auto &ax = known_ax[i];
            bc_axis_list[i].emplace_back(bc_axis[ax]);
        }
    }
    return bc_axis_list;
}

static bound_axis infer_broadcast_arg_axis_binding(
        bound_axis known_axis_list, const std::vector<int> &bc_axis) {
    bound_axis bc_arg_axis_list(known_axis_list.size());
    for (size_t i = 0; i < bc_arg_axis_list.size(); i++) {
        auto &known_ax = known_axis_list[i];
        for (size_t j = 0; j < known_ax.size(); j++) {
            auto iter = std::find(bc_axis.begin(), bc_axis.end(), known_ax[j]);
            if (iter != bc_axis.end()) {
                auto offset = std::distance(bc_axis.begin(), iter);
                bc_arg_axis_list[i].emplace_back(offset);
            }
        }
    }
    return bc_arg_axis_list;
}

static std::vector<int> fill_auto_broadcast_bc_axis(
        const sc_dims &input_shape, const sc_dims &output_shape) {
    if (input_shape.size() == 1 && input_shape[0] == 1) { return {-1}; }
    // following auto_broadcast semantics
    const size_t input_rank = input_shape.size();
    const size_t output_rank = output_shape.size();
    COMPILE_ASSERT(output_rank >= input_rank,
            "Incorrect input or output shape for broadcastable op.");
    const size_t offset = output_rank - input_rank;
    std::vector<int> bc_axis;
    for (size_t i = 0; i < input_rank; ++i) {
        // TODO(yifei): consider whether input_shape[i] != 1 is
        // necessary here
        if (input_shape[i] == output_shape[i + offset]) {
            bc_axis.emplace_back(i + offset);
        }
    }
    if (bc_axis.empty()) { bc_axis.emplace_back(-1); }
    return bc_axis;
}

select_op_t::select_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    op_name_ = "select";
    COMPILE_ASSERT(ins.size() == 3, "Select op shall have 3 inputs.");
    info_.inputs_ = ins;
    std::string auto_broadcast
            = attrs.get_or_else("auto_broadcast", std::string("numpy"));
    COMPILE_ASSERT(auto_broadcast == "numpy" || get_max_input() == -1,
            "Select op's inputs should have the same size when auto_broadcast "
            "is none.");
    int maxtensor_idx = get_max_input() < 0 ? 1 : get_max_input();
    const auto &output_shape
            = info_.inputs_[maxtensor_idx]->details_.get_plain_dims();
    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(
                this, ins[maxtensor_idx]->details_));
        info_.outputs_[0]->details_.dtype_ = info_.inputs_[1]->details_.dtype_;
    } else {
        info_.outputs_ = outs;
    }
    COMPILE_ASSERT(info_.outputs_[0]->details_.get_plain_dims() == output_shape,
            "Select op's output doesn't have the correct shape");
    COMPILE_ASSERT(info_.outputs_[0]->details_.dtype_
                    == info_.inputs_[1]->details_.dtype_,
            "Select op's output doesn't have the correct data type");

    attrs_ = attrs;

    plain_bc_axis_.reserve(3);
    for (size_t i = 0; i < info_.inputs_.size(); ++i) {
        plain_bc_axis_.emplace_back(fill_auto_broadcast_bc_axis(
                info_.inputs_[i]->details_.get_plain_dims(), output_shape));
    }
}

select_op_t::select_op_t(
        graph_tensor_ptr cond, graph_tensor_ptr then, graph_tensor_ptr els)
    : select_op_t({std::move(cond), std::move(then), std::move(els)}, {}, {}) {}

int select_op_t::get_broadcast_input(const int l, const int r) const {
    const graph_tensor_ptr &lhs = info_.inputs_[l];
    const graph_tensor_ptr &rhs = info_.inputs_[r];
    const sc_dims &lhs_dims = info_.inputs_[l]->details_.get_plain_dims();
    const sc_dims &rhs_dims = info_.inputs_[r]->details_.get_plain_dims();
    if (lhs_dims == rhs_dims) {
        return -1;
    } else {
        int side_needs_broadcast = -1;
        bool multi_directional = false;
        const size_t lhs_rank = lhs_dims.size();
        const size_t rhs_rank = rhs_dims.size();
        const size_t max_rank = std::max(lhs_rank, rhs_rank);

        const size_t lhs_offset = max_rank - lhs_rank;
        const size_t rhs_offset = max_rank - rhs_rank;
        for (size_t i = 0; i < max_rank; ++i) {
            sc_dim l = 1, r = 1;
            if (i >= lhs_offset) l = lhs_dims[i - lhs_offset];
            if (i >= rhs_offset) r = rhs_dims[i - rhs_offset];
            if (l == 1 && r != 1) {
                if (side_needs_broadcast == 1) multi_directional = true;
                side_needs_broadcast = 0;
            } else if (l != 1 && r == 1) {
                if (side_needs_broadcast == 0) multi_directional = true;
                side_needs_broadcast = 1;
            }
        }
        if (multi_directional) { return -2; }
        if (side_needs_broadcast == -1) {
            if (lhs_dims.size() == rhs_dims.size()) {
                COMPILE_ASSERT(lhs->is_dynamic() && rhs->is_dynamic(),
                        "Unexpected shape condition in get_broadcast_input.");
            }
            return lhs_dims.size() > rhs_dims.size() ? r : l;
        } else {
            return side_needs_broadcast ? r : l;
        }
    }
}

static bool shape_equal(const sc_dims &shape1, const sc_dims &shape2) {
    if (shape1.size() != shape2.size()) return false;
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (!is_dynamic_dim(shape1[i]) && !is_dynamic_dim(shape2[i])
                && shape1[i] != shape2[i]) {
            return false;
        }
    }
    return true;
}

int select_op_t::get_max_input() const {
    const sc_dims &cond_dims = info_.inputs_[0]->details_.get_plain_dims();
    const sc_dims &then_dims = info_.inputs_[1]->details_.get_plain_dims();
    const sc_dims &else_dims = info_.inputs_[2]->details_.get_plain_dims();
    if (cond_dims == then_dims && then_dims == else_dims) {
        return -1;
    } else {
        for (size_t i = 0; i < info_.inputs_.size(); ++i) {
            bool is_i_max_input = true;
            for (size_t j = 0; j < info_.inputs_.size(); ++j) {
                if (i != j) {
                    int ret = get_broadcast_input(i, j);
                    if (ret == static_cast<int>(i) || ret == -2) {
                        // if ret indicates i is broadcast input
                        is_i_max_input = false;
                        break;
                    }
                }
            }
            if (is_i_max_input) { return i; }
        }
    }
    COMPILE_ASSERT(0, "Cannot find select op's max input.");
    return -2;
}

static sc_data_format_t infer_broadcast_format(
        const logical_tensor_t &target_lt, const logical_tensor_t &bc_lt) {
    COMPILE_ASSERT(
            bc_lt.get_plain_dims().size() == target_lt.get_plain_dims().size(),
            "infer_blocking_format only support plain dimension aligned cases");
    sc_data_format_kind_t target_lt_format_code
            = target_lt.get_format().format_code_;
    sc_data_format_t::blocking_t blocks = target_lt.get_format().blocks_;
    sc_data_format_kind_t bc_lt_format_code = bc_lt.get_format().format_code_;
    // start infer the blocks
    sc_dims bc_plain_dim = bc_lt.get_plain_dims();
    sc_dims target_plain_dim = target_lt.get_plain_dims();
    int block_dim = target_lt_format_code.ndims()
            - target_lt_format_code.norig_dims();
    int target_batch_dim = target_lt.get_plain_dims().size()
            - target_lt_format_code.norig_dims();
    for (int i = 0; i < target_lt_format_code.norig_dims(); ++i) {
        if (bc_plain_dim[target_batch_dim + i] == 1
                && target_plain_dim[target_batch_dim + i] != 1) {
            // if bc_plain_dim is 1 and this axis is with broadcast semantics
            auto axis = target_lt_format_code.collect_blocking_index(i);
            for (auto ax : axis) {
                blocks[ax] = 1;
            }
        }
    }
    // start infer the format code
    // if both batch OR both non-batch
    // smaller side's format code == larger side's format code
    COMPILE_ASSERT(target_lt_format_code.norig_dims()
                    == bc_lt_format_code.norig_dims(),
            "Unsupported case for select op's query format.");
    return sc_data_format_t(target_lt.get_format().format_code_, blocks);
}

void select_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    int max_input_idx = get_max_input();
    if (max_input_idx == -1) {
        // only consider then and else branch, similar to binary_elementwise
        if (is_dynamic()) {
            max_input_idx
                    = info_.inputs_[1]->details_.get_format_candidates().size()
                            >= info_.inputs_[2]
                                       ->details_.get_format_candidates()
                                       .size()
                    ? 1
                    : 2;
        } else {
            max_input_idx = 1;
            if (!info_.inputs_[1]->details_.get_format().is_blocking()
                    && info_.inputs_[2]->details_.get_format().is_blocking()) {
                max_input_idx = 2;
            }
        }
    }
    if (attrs_.has_key(op_attr_key::layout_input_index)) {
        max_input_idx = attrs_.get<int>(op_attr_key::layout_input_index);
    }
    attrs_.set<int>(op_attr_key::layout_input_index, max_input_idx);

    size_t max_rank
            = info_.inputs_[max_input_idx]->details_.get_plain_dims().size();
    auto ref_format = info_.inputs_[max_input_idx]->details_.get_format();

    for (size_t i = 0; i < info_.inputs_.size(); ++i) {
        size_t input_rank = info_.inputs_[i]->details_.get_plain_dims().size();
        COMPILE_ASSERT((input_rank == 1
                               && info_.inputs_[i]->details_.get_plain_dims()
                                       == sc_dims {1}
                               && info_.inputs_[i]->details_.get_format()
                                       == sc_data_format_t(format_kinds::A))
                        || input_rank == max_rank,
                "Invalid shape or format encountered in select op's query "
                "format.");
        if (static_cast<int>(i) == max_input_idx) {
            in_formats.push_back({ref_format});
        } else if (input_rank == 1) {
            in_formats.push_back({info_.inputs_[i]->details_.get_format()});
        } else {
            auto target_format = infer_broadcast_format(
                    info_.inputs_[max_input_idx]->details_,
                    info_.inputs_[i]->details_);
            in_formats.push_back({target_format});
        }
    }
    out_formats.push_back({ref_format});

    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

// The logic below might be suitable for most fusible op, which has same
// slice ranges on inputs and outputs
void select_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    COMPILE_ASSERT(
            get_inputs().size() == 3, "Select op is expected to have 3 inputs");
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map
            = search_known_slice_ranges(this, fsmap, stat_map);
    if (known_ranges_map.empty()) return;
    auto &outslice = fsmap.get(get_outputs()[0]);
    // if unkown slice ranges exist.
    int maxtensor_idx = get_max_input();
    if (known_ranges_map.size() < get_inputs().size()) {
        std::vector<int> known_idx(info_.inputs_.size(), 0);
        for (size_t i = 0; i < info_.inputs_.size(); ++i) {
            known_idx[i] = (known_ranges_map.find(i) != known_ranges_map.end());
        }
        // check broadcast
        if (maxtensor_idx >= 0) {
            if (known_idx[maxtensor_idx] == 1) {
                for (int i = 0; i < 3; i++) {
                    if (known_idx[i] == 0) {
                        bool keep_dims = get_inputs()[i]
                                                 ->details_.get_blocking_dims()
                                                 .size()
                                == get_inputs()[maxtensor_idx]
                                           ->details_.get_blocking_dims()
                                           .size();
                        auto bc_axis = get_bc_axis(maxtensor_idx, i);
                        slice_range_list bc_arg_range_list
                                = infer_broadcast_arg_slice(
                                        known_ranges_map[maxtensor_idx],
                                        bc_axis, keep_dims);
                        known_ranges_map[i] = bc_arg_range_list;
                    }
                }
            } else {
                auto it = std::find(known_idx.begin(), known_idx.end(), 1);
                COMPILE_ASSERT(it != known_idx.end(), "No known idx found.");
                int known_tensor_idx = std::distance(known_idx.begin(), it);
                auto bc_axis = get_bc_axis(maxtensor_idx, known_tensor_idx);
                slice_range_list bc_range_list = infer_broadcast_slice(
                        known_ranges_map[known_tensor_idx], bc_axis,
                        get_inputs()[maxtensor_idx]
                                ->details_.get_blocking_dims_expr(
                                        get_owner_graph()));
                known_ranges_map[maxtensor_idx] = bc_range_list;
                known_idx[maxtensor_idx] = 1;
                // deal with the remaining unkown slice range
                it = std::find(known_idx.begin(), known_idx.end(), 0);
                if (it != known_idx.end()) {
                    int remaining_idx = std::distance(known_idx.begin(), it);
                    bool keep_dims = get_inputs()[remaining_idx]
                                             ->details_.get_blocking_dims()
                                             .size()
                            == get_inputs()[maxtensor_idx]
                                       ->details_.get_blocking_dims()
                                       .size();
                    bc_axis = get_bc_axis(maxtensor_idx, remaining_idx);
                    bc_range_list = infer_broadcast_arg_slice(
                            known_ranges_map[maxtensor_idx], bc_axis,
                            keep_dims);
                    known_ranges_map[remaining_idx] = bc_range_list;
                }
            }
            // set the other unknown slice range by achieved
            // known_ranges_list
            set_unknown_slice_ranges(this, known_ranges_map, fsmap, stat_map);
            // set outputs slice range
            outslice = known_ranges_map[maxtensor_idx];
            return;
        } else {
            auto it = std::find(known_idx.begin(), known_idx.end(), 1);
            COMPILE_ASSERT(it != known_idx.end(), "No known idx found.");
            int known_tensor = std::distance(known_idx.begin(), it);
            for (int i = 0; i < 3; i++) {
                if (i != known_tensor) {
                    known_ranges_map[i] = known_ranges_map[known_tensor];
                }
            }
        }
        // set the other unknown slice range by achieved known_ranges_list
        set_unknown_slice_ranges(this, known_ranges_map, fsmap, stat_map);
    }
    // set outputs slice range
    outslice = known_ranges_map[maxtensor_idx > -1 ? maxtensor_idx : 1];
}

void select_op_t::infer_binding_axis(bound_axis_map &bdax_map) {
    COMPILE_ASSERT(
            get_inputs().size() == 3, "Select op is expected to have 3 inputs");
    // search known axis from any input of cur fusbile op
    auto known_axis_map = search_known_bound_axis(this, bdax_map);
    if (!bdax_map.get(get_outputs()[0]).empty()) return;

    // if unkown slice ranges exist.
    if (known_axis_map.size() < get_inputs().size()) {
        std::vector<int> known_idx(3, 0);
        known_idx[0] = known_axis_map.find(0) != known_axis_map.end() ? 1 : 0;
        known_idx[1] = known_axis_map.find(1) != known_axis_map.end() ? 1 : 0;
        known_idx[2] = known_axis_map.find(2) != known_axis_map.end() ? 1 : 0;
        // check broadcast
        int maxtensor_idx = get_max_input();
        if (maxtensor_idx >= 0) {
            if (known_idx[maxtensor_idx] == 1) {
                for (int i = 0; i < 3; i++) {
                    if (known_idx[i] == 0) {
                        bool keep_dims = get_inputs()[i]
                                                 ->details_.get_blocking_dims()
                                                 .size()
                                == get_inputs()[maxtensor_idx]
                                           ->details_.get_blocking_dims()
                                           .size();
                        if (keep_dims) {
                            known_axis_map[i] = known_axis_map[maxtensor_idx];
                        } else {
                            COMPILE_ASSERT(
                                    get_inputs()[i]->details_.get_plain_dims()
                                            == sc_dims {1},
                                    "Select op's infer binding axis "
                                    "encountered unaligned input shapes.");
                            bound_axis bc_arg_axis_list(
                                    known_axis_map[maxtensor_idx].size());
                            known_axis_map[i] = bc_arg_axis_list;
                        }
                    }
                }
            } else {
                auto it = std::find(known_idx.begin(), known_idx.end(), 1);
                COMPILE_ASSERT(it != known_idx.end(), "No known idx found.");
                int known_tensor_idx = std::distance(known_idx.begin(), it);
                bool keep_dims = get_inputs()[known_tensor_idx]
                                         ->details_.get_blocking_dims()
                                         .size()
                        == get_inputs()[maxtensor_idx]
                                   ->details_.get_blocking_dims()
                                   .size();
                if (keep_dims) {
                    known_axis_map[maxtensor_idx]
                            = known_axis_map[known_tensor_idx];
                } else {
                    COMPILE_ASSERT(get_inputs()[known_tensor_idx]
                                            ->details_.get_plain_dims()
                                    == sc_dims {1},
                            "Select op's infer binding axis encountered "
                            "unaligned input shapes.");
                    auto plain_bc_axis = fill_auto_broadcast_bc_axis(
                            get_inputs()[known_tensor_idx]
                                    ->details_.get_plain_dims(),
                            get_inputs()[maxtensor_idx]
                                    ->details_.get_plain_dims());
                    if (plain_bc_axis == std::vector<int> {-1}) {
                        plain_bc_axis[0] = get_inputs()[maxtensor_idx]
                                                   ->details_.get_plain_dims()
                                                   .size()
                                - 1;
                    }
                    auto bc_axis_list = infer_broadcast_axis_binding(
                            known_axis_map[known_tensor_idx], plain_bc_axis);
                    known_axis_map[maxtensor_idx] = bc_axis_list;
                }
                known_idx[maxtensor_idx] = 1;
                // deal with the remaining unknown binding axis
                it = std::find(known_idx.begin(), known_idx.end(), 0);
                if (it != known_idx.end()) {
                    int remaining_idx = std::distance(known_idx.begin(), it);
                    bool keep_dims = get_inputs()[remaining_idx]
                                             ->details_.get_blocking_dims()
                                             .size()
                            == get_inputs()[maxtensor_idx]
                                       ->details_.get_blocking_dims()
                                       .size();
                    if (keep_dims) {
                        known_axis_map[remaining_idx]
                                = known_axis_map[maxtensor_idx];
                    } else {
                        COMPILE_ASSERT(get_inputs()[remaining_idx]
                                                ->details_.get_plain_dims()
                                        == sc_dims {1},
                                "Select op's infer binding axis encountered "
                                "unaligned input shapes.");
                        bound_axis bc_arg_axis_list(
                                known_axis_map[maxtensor_idx].size());
                        known_axis_map[remaining_idx] = bc_arg_axis_list;
                    }
                }
            }
        } else {
            auto it = std::find(known_idx.begin(), known_idx.end(), 1);
            COMPILE_ASSERT(it != known_idx.end(), "No known idx found.");
            int known_tensor = std::distance(known_idx.begin(), it);
            for (int i = 0; i < 3; i++) {
                if (i != known_tensor) {
                    known_axis_map[i] = known_axis_map[known_tensor];
                }
            }
        }
    }
    // set outputs axis binding
    int maxtensor_idx = get_max_input();
    bdax_map.get(get_outputs()[0])
            = known_axis_map[maxtensor_idx > -1 ? maxtensor_idx : 1];
    // set the other unknown axis binding by achieved known_axis_map
    set_unknown_axis_binding(this, known_axis_map, bdax_map);
}

void select_op_t::pre_binding_axis(bound_axis_map &bdax_map) {}

std::vector<int> select_op_t::get_bc_axis(const int l, const int r) const {
    auto lhs_shape = info_.inputs_[l]->details_.get_plain_dims();
    auto rhs_shape = info_.inputs_[r]->details_.get_plain_dims();
    int bc_input_idx = get_broadcast_input(l, r);
    COMPILE_ASSERT(bc_input_idx != -2,
            "get_bc_axis shall be called with uni-directional broadcastable "
            "inputs.");
    if (bc_input_idx == -1) {
        auto temp = get_inputs()[l]->details_.get_blocking_dims();
        std::vector<int> blocking_dims(temp.size());
        std::iota(blocking_dims.begin(), blocking_dims.end(), 0);
        return blocking_dims;
    }
    std::vector<int> plain_axis = bc_input_idx == l
            ? fill_auto_broadcast_bc_axis(lhs_shape, rhs_shape)
            : fill_auto_broadcast_bc_axis(rhs_shape, lhs_shape);
    if (plain_axis == std::vector<int> {-1}) return plain_axis;
    return transform_axis_plain2blocking(
            info_.inputs_[bc_input_idx == l ? r : l], plain_axis);
}

shape_rl_vec select_op_t::get_dynamic_shape_relations() const {
    shape_rl_vec ret;
    auto &out_plain_dims = get_outputs()[0]->details_.get_plain_dims();
    for (size_t i = 0; i < get_inputs().size(); ++i) {
        const auto &in_plain_dims = get_inputs()[i]->details_.get_plain_dims();
        assert(in_plain_dims.size() == out_plain_dims.size()
                || in_plain_dims.size() == 1);
    }
    for (size_t i = 0; i < get_inputs().size(); ++i) {
        const auto &dims1 = get_inputs()[i]->details_.get_plain_dims();
        for (size_t j = i + 1; j < get_inputs().size(); ++j) {
            const auto &dims2 = get_inputs()[j]->details_.get_plain_dims();
            if (dims1.size() == dims2.size()) {
                for (size_t idx = 0; idx < dims1.size(); ++idx) {
                    // maybe broadcast
                    if ((is_dynamic_dim(dims1[idx])
                                || is_dynamic_dim(dims2[idx]))
                            && dims1[idx] != 1 && dims2[idx] != 1) {
                        ret.emplace_back(dims1[idx], dims2[idx]);
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < out_plain_dims.size(); ++i) {
        if (is_dynamic_dim(out_plain_dims[i])) {
            for (size_t j = 0; j < get_inputs().size(); ++j) {
                const auto &in_plain_dims
                        = get_inputs()[j]->details_.get_plain_dims();
                if (i < in_plain_dims.size() && in_plain_dims[i] != 1) {
                    ret.emplace_back(in_plain_dims[i], out_plain_dims[i]);
                }
            }
        }
    }
    return ret;
}

void compute_block_select(const std::vector<const tensor_slice *> &src,
        const tensor_slice &dst, sc_op_info_t &info, const int maxtensor_idx,
        const std::vector<std::vector<int>> &blocking_bc_axis,
        const vectorized_info_t &vx_info, const mask_compute_func_t &compute,
        sc_data_type_t dtype = datatypes::f32, size_t wkld = 0UL) {
    // nested loop vars
    std::vector<expr> iter_vars;
    // the indices for multiple inputs. First dim: the input, Second dim: the
    // dimensions in the tensor
    std::vector<expr> in_idx, in_bc_idx_1, in_bc_idx_2;
    // the indices for the output tensor
    std::vector<expr> dst_idx;

    COMPILE_ASSERT(maxtensor_idx >= 0, "maxtensor_idx shall be determined.")
    bool is_blocking_shape = is_op_input_blocking_shape(info);
    std::vector<int> bc;
    for (int i = 0; i < 3; i++) {
        if (i != maxtensor_idx) { bc.emplace_back(i); }
    }

    const tensor_slice *in_tsl = src[maxtensor_idx], *in_bc_tsl_1 = src[bc[0]],
                       *in_bc_tsl_2 = src[bc[1]];
    bool keep_dims_1 = in_tsl->get_base_dims().size()
            == in_bc_tsl_1->get_base_dims().size();
    bool keep_dims_2 = in_tsl->get_base_dims().size()
            == in_bc_tsl_2->get_base_dims().size();
    auto bc_axis_1 = blocking_bc_axis[bc[0]];
    auto bc_axis_2 = blocking_bc_axis[bc[1]];
    // add output type check, manual downcast
    sc_data_etype out_etype
            = dst.tptr_->dtype_.get_pointer_element().as_etype();
    // use src_indices.at(0) as default
    for (unsigned i = 0; i < dst.nslice_dims(); i++) {
        // make the loop var for the for-loop
        iter_vars.emplace_back(builder::make_var(datatypes::index,
                std::string("_fuseiter") + fusion_create_idx()));
        in_idx.emplace_back(iter_vars.back());
        if (std::find(bc_axis_1.begin(), bc_axis_1.end(), i)
                != bc_axis_1.end()) {
            in_bc_idx_1.emplace_back(iter_vars.back());
        } else if (keep_dims_1) {
            in_bc_idx_1.emplace_back(0);
        }
        if (std::find(bc_axis_2.begin(), bc_axis_2.end(), i)
                != bc_axis_2.end()) {
            in_bc_idx_2.emplace_back(iter_vars.back());
        } else if (keep_dims_2) {
            in_bc_idx_2.emplace_back(0);
        }
        /** push an index for output tensor **/
        dst_idx.emplace_back(iter_vars.back());
    }

    // For empty bc_axis
    if (in_bc_idx_1.empty()) in_bc_idx_1 = {0};
    if (in_bc_idx_2.empty()) in_bc_idx_2 = {0};

    expr indexed_target;
    expr indexed_input;
    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder is set");
    auto slice_len = dst.get_shape().at(vx_info.axis);
    int lanes = static_cast<int>(vx_info.lanes);
    auto floor = do_cast_and_fold(slice_len / lanes * lanes);
    auto tail = do_cast_and_fold(slice_len % lanes);
    int floor_int = 0;
    int tail_int = 0;
    if (floor.isa<constant>()) { floor_int = get_expr_as_int(floor); }
    if (tail.isa<constant>()) { tail_int = get_expr_as_int(tail); }
    expr last_axis = slice_len;
    std::vector<stmt> tcur;
    stmt cur;

    // recover schedule loop
    for (int i = static_cast<int>(dst.get_shape().size() - 1); i >= 0; i--) {
        stmt body;
        // move broadcast op to body
        if (static_cast<int>(dst.get_shape().size()) == vx_info.axis + 1
                && i == vx_info.axis) {
            if ((!floor.isa<constant>() || floor_int)
                    || (!tail.isa<constant>() || tail_int)) {
                int vec_len = vx_info.lanes;
                stmt mask_def;
                expr mask;
                bld->push_scope();
                // In the dynamic scene, when the input shapes are blocking,
                // there is no tail.
                if ((!tail.isa<constant>() && !is_blocking_shape) || tail_int) {
                    auto last_axis_offset
                            = cast_to_s32(last_axis - iter_vars.at(i));
                    // mask = min(max(0, last_dim_len -
                    // last_dim_idx),real_step) To choose [0 ~
                    // step] mask
                    auto cur_step = builder::make_min(
                            builder::make_max(builder::make_constant(0),
                                    last_axis_offset),
                            vec_len);
                    auto cur_step_var = builder::make_var(
                            sc_data_type_t::s32(1), "cur_step_var");
                    auto cur_step_var_assign
                            = builder::make_var_tensor_def_unattached(
                                    cur_step_var, linkage::local, cur_step);
                    bld->emit(cur_step_var_assign);
                    // mask = other_dims_condition ? mask : 0;
                    mask = generate_mask_var_by_step(
                            mask_def, cur_step_var, vec_len);
                    bld->emit(mask_def);
                }
                expr indexed_target = builder::make_indexing(
                        dst.tptr_, dst_idx, vx_info.lanes, mask);
                expr indexed_input = builder::make_indexing(
                        in_tsl->tptr_, in_idx, vx_info.lanes, mask);

                if (!in_tsl->tptr_->dtype_.get_pointer_element().is_etype(
                            out_etype)) {
                    indexed_input = builder::make_cast(
                            sc_data_type_t(
                                    out_etype, indexed_input->dtype_.lanes_),
                            indexed_input);
                }

                expr indexed_bc_input_1, indexed_bc_input_2;
                // IF last dim is included in bc_axis_1.
                if (bc_axis_1.back() == static_cast<int64_t>(vx_info.axis)) {
                    indexed_bc_input_1
                            = builder::make_indexing(in_bc_tsl_1->tptr_,
                                    in_bc_idx_1, vx_info.lanes, mask);
                }
                // IF last dim is excluded in bc_axis_1.
                else {
                    indexed_bc_input_1 = builder::make_broadcast(
                            builder::make_indexing(
                                    in_bc_tsl_1->tptr_, in_bc_idx_1),
                            static_cast<int>(vx_info.lanes));
                }
                // IF last dim is excluded in bc_axis_2.
                if (bc_axis_2.back() == static_cast<int64_t>(vx_info.axis)) {
                    indexed_bc_input_2
                            = builder::make_indexing(in_bc_tsl_2->tptr_,
                                    in_bc_idx_2, vx_info.lanes, mask);
                }
                // IF last dim is excluded in bc_axis_2.
                else {
                    indexed_bc_input_2 = builder::make_broadcast(
                            builder::make_indexing(
                                    in_bc_tsl_2->tptr_, in_bc_idx_2),
                            static_cast<int>(vx_info.lanes));
                }
                std::vector<expr::lvalue_proxy_t> target_vec {
                        expr::lvalue_proxy_t(indexed_target, false)};
                std::vector<expr> inputs(3);
                if (maxtensor_idx == 0) {
                    inputs = {indexed_input, indexed_bc_input_1,
                            indexed_bc_input_2};
                } else if (maxtensor_idx == 1) {
                    inputs = {indexed_bc_input_1, indexed_input,
                            indexed_bc_input_2};
                } else {
                    inputs = {indexed_bc_input_1, indexed_bc_input_2,
                            indexed_input};
                }
                cur = compute(inputs, target_vec);
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                cur = make_stmt<for_loop_node_t>(iter_vars.at(i), expr(0),
                        last_axis, expr(int(vx_info.lanes)), bld->pop_scope(),
                        true, for_type::NORMAL);
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
                indexed_target = builder::make_indexing(dst.tptr_, dst_idx);

                indexed_input = builder::make_indexing(in_tsl->tptr_, in_idx);

                expr indexed_bc_input_1 = builder::make_indexing(
                        in_bc_tsl_1->tptr_, in_bc_idx_1);
                expr indexed_bc_input_2 = builder::make_indexing(
                        in_bc_tsl_2->tptr_, in_bc_idx_2);

                std::vector<expr::lvalue_proxy_t> target_vec {
                        expr::lvalue_proxy_t(indexed_target, false)};
                bld->push_scope();
                std::vector<expr> inputs(3);
                if (maxtensor_idx == 0) {
                    inputs = {indexed_input, indexed_bc_input_1,
                            indexed_bc_input_2};
                } else if (maxtensor_idx == 1) {
                    inputs = {indexed_bc_input_1, indexed_input,
                            indexed_bc_input_2};
                } else {
                    inputs = {indexed_bc_input_1, indexed_bc_input_2,
                            indexed_input};
                }
                cur = compute(inputs, target_vec);
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
        for (auto &it : tcur) {
            bld->emit(it);
        }
    } else {
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    }
}

void select_op_t::compute_block(context_ptr ctx,
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
            = vectorize_step(ctx, info_.inputs_[1]->details_.dtype_.type_code_);
    // use broad-cast
    int maxtensor_idx = get_max_input();
    maxtensor_idx = (maxtensor_idx == -1) ? 0 : maxtensor_idx;
    auto func = [&](const std::vector<expr> &ins,
                        std::vector<expr::lvalue_proxy_t> &outs) -> stmt {
        return builder::make_assign_unattached(outs[0],
                builder::make_select(
                        ins[0] > make_expr<constant_node>(
                                static_cast<uint64_t>(0), ins[0]->dtype_),
                        ins[1], ins[2]));
        // Here we use "ins[0] >
        // make_expr<constant_node>(static_cast<uint64_t>(0),
        // ins[0]->dtype_)" instead of "ins[0]", because _mm_cmp_epi8_mask
        // intrinsic is the optimal instruction to cast bool tensor to
        // bitmap
    };
    std::vector<std::vector<int>> blocking_bc_axis(info_.inputs_.size());
    for (size_t i = 0; i < info_.inputs_.size(); i++) {
        blocking_bc_axis[i] = get_bc_axis(maxtensor_idx, i);
    }
    compute_block_select(inputs, *dst[0], info_, maxtensor_idx,
            blocking_bc_axis, vx_info_, mask_compute_func_t(func),
            info_.outputs_[0]->details_.dtype_, wkld);
}

// Pure virtual function in fusible_op_t class.
void select_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {}

// Pure virtual function in fusible_op_t class.
void select_op_t::prepare_fusion_data(fdata_map &fdmap) {}

OP_REGISTER(select_op_t, select)

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
