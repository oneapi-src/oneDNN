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

static sc_dims infer_select_output_shape(const sc_dims &cond_shape,
        const sc_dims &then_shape, const sc_dims &else_shape) {
    sc_dims output_shape
            = op_traits::may_broadcast_t::infer_auto_broadcast_output_shape(
                    then_shape, else_shape);
    output_shape
            = op_traits::may_broadcast_t::infer_auto_broadcast_output_shape(
                    output_shape, cond_shape);
    return output_shape;
}

std::vector<int> select_op_t::get_non_broadcast_input_index(
        bool assert_non_empty) const {
    const sc_dims &cond_dims = info_.inputs_[0]->details_.get_plain_dims();
    const sc_dims &then_dims = info_.inputs_[1]->details_.get_plain_dims();
    const sc_dims &else_dims = info_.inputs_[2]->details_.get_plain_dims();
    auto output_dims
            = infer_select_output_shape(cond_dims, then_dims, else_dims);
    std::vector<int> ret;
    for (size_t i = 0; i < info_.inputs_.size(); ++i) {
        if (may_broadcast_t::broadcastable_shape_equal(
                    info_.inputs_[i]->details_.get_plain_dims(), output_dims)) {
            ret.emplace_back(i);
        }
    }
    if (assert_non_empty) {
        // non-broadcast input means input no need to be broadcasted, whose
        // shape is the same as the output
        COMPILE_ASSERT(!ret.empty(),
                "Select op is required to have at least one non-broadcast "
                "input at this stage.");
    }
    return ret;
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

select_op_t::select_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    op_name_ = "select";
    COMPILE_ASSERT(ins.size() == 3, "Select op shall have 3 inputs.");
    info_.inputs_ = ins;
    auto cond_shape = info_.inputs_[0]->details_.get_plain_dims();
    auto then_shape = info_.inputs_[1]->details_.get_plain_dims();
    auto else_shape = info_.inputs_[2]->details_.get_plain_dims();
    auto output_shape
            = infer_select_output_shape(cond_shape, then_shape, else_shape);
    auto non_bc_indices = get_non_broadcast_input_index(false);
    std::string auto_broadcast
            = attrs.get_or_else("auto_broadcast", std::string("numpy"));
    COMPILE_ASSERT(auto_broadcast == "numpy" || non_bc_indices.size() == 3,
            "Select op's all three inputs should have the same size when "
            "auto_broadcast is none.");
    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        info_.outputs_[0]->details_.set_plain_dims(output_shape);
        int ref_idx = get_ref_input_index(true);
        if (ref_idx == may_broadcast_t::NOT_DETERMINED) {
            ref_idx = then_shape.size() >= else_shape.size() ? 1 : 2;
        }
        info_.outputs_[0]->details_.set_format(
                info_.inputs_[ref_idx]->details_.get_format());
        info_.outputs_[0]->details_.dtype_ = info_.inputs_[1]->details_.dtype_;
    } else {
        info_.outputs_ = outs;
    }
    COMPILE_ASSERT(info_.outputs_[0]->details_.get_plain_dims() == output_shape,
            "Select op's output doesn't have the correct shape");

    attrs_ = attrs;
    plain_bc_axis_.reserve(3);
    for (size_t i = 0; i < info_.inputs_.size(); ++i) {
        plain_bc_axis_.emplace_back(
                op_traits::may_broadcast_t::get_auto_broadcast_bc_axis(
                        info_.inputs_[i]->details_.get_plain_dims(),
                        output_shape));
    }
}

select_op_t::select_op_t(
        graph_tensor_ptr cond, graph_tensor_ptr then, graph_tensor_ptr els)
    : select_op_t({std::move(cond), std::move(then), std::move(els)}, {}, {}) {}

int select_op_t::get_ref_input_index(bool assert_determined) const {
    auto non_bc_index = get_non_broadcast_input_index(assert_determined);
    if (!assert_determined && non_bc_index.empty())
        return may_broadcast_t::NOT_DETERMINED;
    int max_input_idx = non_bc_index[0];
    bool is_cond_non_bc = std::find(non_bc_index.begin(), non_bc_index.end(), 0)
            != non_bc_index.end();
    bool is_then_non_bc = std::find(non_bc_index.begin(), non_bc_index.end(), 1)
            != non_bc_index.end();
    bool is_else_non_bc = std::find(non_bc_index.begin(), non_bc_index.end(), 2)
            != non_bc_index.end();
    if (is_then_non_bc && is_else_non_bc) {
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
    if (is_cond_non_bc) {
        COMPILE_ASSERT(non_bc_index.size() > 1,
                "Select op's cond input shall not be the only non-broadcast "
                "input.");
        max_input_idx = non_bc_index[1];
    }
    if (attrs_.has_key(op_attr_key::layout_input_index)) {
        max_input_idx = attrs_.get<int>(op_attr_key::layout_input_index);
    }
    return max_input_idx;
}

void select_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;

    int max_input_idx = get_ref_input_index(true);
    attrs_[op_attr_key::layout_input_index] = max_input_idx;

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
    int maxtensor_idx = get_ref_input_index(true);
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
                        known_ranges_map[i] = std::move(bc_arg_range_list);
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
        int maxtensor_idx = get_ref_input_index(true);
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
                    auto plain_bc_axis = op_traits::may_broadcast_t::
                            get_auto_broadcast_bc_axis(
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
    int maxtensor_idx = get_ref_input_index(true);
    bdax_map.get(get_outputs()[0])
            = known_axis_map[maxtensor_idx > -1 ? maxtensor_idx : 1];
    // set the other unknown axis binding by achieved known_axis_map
    set_unknown_axis_binding(this, known_axis_map, bdax_map);
}

void select_op_t::pre_binding_axis(bound_axis_map &bdax_map) {}

std::vector<int> select_op_t::get_bc_axis(
        const int axis1, const int axis2) const {
    auto shape1 = info_.inputs_[axis1]->details_.get_plain_dims();
    auto shape2 = info_.inputs_[axis2]->details_.get_plain_dims();
    auto non_bc_indices = get_non_broadcast_input_index(true);
    int ref_axis
            = std::find(non_bc_indices.begin(), non_bc_indices.end(), axis1)
                    != non_bc_indices.end()
            ? axis1
            : axis2;
    std::vector<int> plain_axis = ref_axis == axis1
            ? op_traits::may_broadcast_t::get_auto_broadcast_bc_axis(
                    shape2, shape1)
            : op_traits::may_broadcast_t::get_auto_broadcast_bc_axis(
                    shape1, shape2);
    if (plain_axis == std::vector<int> {-1}) return plain_axis;
    return transform_axis_plain2blocking(info_.inputs_[ref_axis], plain_axis);
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

void compute_block_select(const context_ptr &ctx,
        const std::vector<const tensor_slice *> &src, const tensor_slice &dst,
        sc_op_info_t &info, const int maxtensor_idx,
        const std::vector<std::vector<int>> &blocking_bc_axis,
        const vectorized_info_t &vx_info, const mask_compute_func_t &compute,
        sc_data_type_t dtype = datatypes::f32, size_t wkld = 0UL) {
    bool use_vectorize = false;
    vec_backend_require(ctx, use_vectorize);
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
        iter_vars.emplace_back(range_from_outer_loop(dst.get_ranges()[i])
                        ? expr(0)
                        : builder::make_var(datatypes::index,
                                std::string("_fuseiter")
                                        + fusion_create_idx()));
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
    std::vector<expr> in_idx_tail = in_idx, in_bc_idx_1_tail = in_bc_idx_1,
                      in_bc_idx_2_tail = in_bc_idx_2, dst_idx_tail = dst_idx;
    auto tail_var = builder::make_var(
            datatypes::index, std::string("_fuseiter") + fusion_create_idx());
    in_idx_tail[vx_info.axis] = tail_var;
    dst_idx_tail[vx_info.axis] = tail_var;
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
    std::vector<stmt> tcur;
    stmt cur;
    int vec_len = vx_info.lanes;
    bool tail_threshold = tail.isa<constant>() && tail_int <= 1;
    bool use_scalar = !use_vectorize || tail_threshold || lanes == 1;
    auto find_bc_input_index = [&](bool tail_threshold,
                                       tensor_slice const *in_bc_tsl,
                                       std::vector<expr> const &in_bc_idx,
                                       std::vector<int> &bc_axis,
                                       expr &indexed_bc_input, expr &mask) {
        // IF last dim is included in bc_axis_1.
        if (bc_axis.back() == static_cast<int64_t>(vx_info.axis)) {
            indexed_bc_input = builder::make_indexing(in_bc_tsl->tptr_,
                    in_bc_idx, tail_threshold ? 1 : vx_info.lanes, mask);
        }
        // IF last dim is excluded in bc_axis_1.
        else {
            if (!tail_threshold) {
                indexed_bc_input = builder::make_broadcast(
                        builder::make_indexing(in_bc_tsl->tptr_, in_bc_idx),
                        static_cast<int>(vx_info.lanes));
            } else {
                indexed_bc_input
                        = builder::make_indexing(in_bc_tsl->tptr_, in_bc_idx);
            }
        }
    };
    // recover schedule loop
    for (int i = static_cast<int>(dst.get_shape().size() - 1); i >= 0; i--) {
        stmt body;
        // move broadcast op to body
        if (static_cast<int>(dst.get_shape().size()) == vx_info.axis + 1
                && i == vx_info.axis) {
            if ((!floor.isa<constant>() || floor_int)) {
                expr mask;
                bld->push_scope();
                expr indexed_target = builder::make_indexing(
                        dst.tptr_, dst_idx, vx_info.lanes);
                expr indexed_input = builder::make_indexing(
                        in_tsl->tptr_, in_idx, vx_info.lanes);

                if (!in_tsl->tptr_->dtype_.get_pointer_element().is_etype(
                            out_etype)) {
                    indexed_input = builder::make_cast(
                            sc_data_type_t(
                                    out_etype, indexed_input->dtype_.lanes_),
                            indexed_input);
                }

                expr indexed_bc_input_1, indexed_bc_input_2;

                find_bc_input_index(false, in_bc_tsl_1, in_bc_idx_1, bc_axis_1,
                        indexed_bc_input_1, mask);
                find_bc_input_index(false, in_bc_tsl_2, in_bc_idx_2, bc_axis_2,
                        indexed_bc_input_2, mask);

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
                cur = bld->pop_scope();
                if (iter_vars.at(i).isa<var>()) {
                    cur = make_stmt<for_loop_node_t>(iter_vars.at(i), expr(0),
                            expr(floor), expr(int(vx_info.lanes)), cur, true,
                            for_type::NORMAL);
                }
                tcur.emplace_back(cur);
            }
            if ((!tail.isa<constant>() && !is_blocking_shape) || tail_int) {
                auto func_tail_var_pos = [&](std::vector<expr> &in_bc_idx,
                                                 std::vector<int> &bc_axis,
                                                 bool keep_dims,
                                                 expr &tail_var) {
                    auto res_it = std::find(
                            bc_axis.begin(), bc_axis.end(), vx_info.axis);
                    if (res_it != bc_axis.end()) {
                        in_bc_idx[keep_dims ? vx_info.axis
                                            : (res_it - bc_axis.begin())]
                                = tail_var;
                    }
                };
                func_tail_var_pos(
                        in_bc_idx_1_tail, bc_axis_1, keep_dims_1, tail_var);
                func_tail_var_pos(
                        in_bc_idx_2_tail, bc_axis_2, keep_dims_2, tail_var);
                expr mask;
                if (!use_scalar) {
                    mask = last_dim_generate_mask(
                            tail_var, floor, slice_len, lanes, true);
                }
                expr indexed_bc_input_1_tail, indexed_bc_input_2_tail;
                find_bc_input_index(use_scalar, in_bc_tsl_1, in_bc_idx_1_tail,
                        bc_axis_1, indexed_bc_input_1_tail, mask);
                find_bc_input_index(use_scalar, in_bc_tsl_2, in_bc_idx_2_tail,
                        bc_axis_2, indexed_bc_input_2_tail, mask);

                expr indexed_target_tail = builder::make_indexing(
                        dst.tptr_, dst_idx_tail, use_scalar ? 1 : lanes, mask);
                expr indexed_input_tail = builder::make_indexing(in_tsl->tptr_,
                        in_idx_tail, use_scalar ? 1 : lanes, mask);
                std::vector<expr::lvalue_proxy_t> target_vec_tail {
                        expr::lvalue_proxy_t(indexed_target_tail, false)};
                bld->push_scope();
                std::vector<expr> inputs_tail(3);
                if (maxtensor_idx == 0) {
                    inputs_tail = {indexed_input_tail, indexed_bc_input_1_tail,
                            indexed_bc_input_2_tail};
                } else if (maxtensor_idx == 1) {
                    inputs_tail = {indexed_bc_input_1_tail, indexed_input_tail,
                            indexed_bc_input_2_tail};
                } else {
                    inputs_tail = {indexed_bc_input_1_tail,
                            indexed_bc_input_2_tail, indexed_input_tail};
                }
                cur = compute(inputs_tail, target_vec_tail);
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                cur = make_stmt<for_loop_node_t>(tail_var, expr(floor),
                        slice_len, use_scalar ? expr(1) : lanes,
                        bld->pop_scope(), true, for_type::NORMAL);
                tcur.emplace_back(cur);
            }
        } else if (iter_vars.at(i).isa<var>()) {
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
        for (auto &it : tcur) {
            bld->emit(it);
        }
        cur->attr()[stmt_attr_key::merge_loop] = true;
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
    int maxtensor_idx = get_ref_input_index(true);

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
    compute_block_select(ctx, inputs, *dst[0], info_, maxtensor_idx,
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
