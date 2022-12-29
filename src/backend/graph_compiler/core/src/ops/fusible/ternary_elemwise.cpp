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

namespace sc {

// Now we only support unidirectional broadcast. We assume one of the three
// tensors(cond,then,else) is the largest, the remaining two tensors are
// unidirectional broadcast to the largest one.
//  (In select spec: 1. then and else are broadcasted to each other; 2. the cond
//  will be one-way broadcasted to the resulting shape of broadcasted then and
//  else)
std::vector<int> select_op_t::infer_broadcast_axis(
        const int l, const int r) const {
    int bc_input_idx = get_broadcast_input(l, r);
    if (bc_input_idx == -1) return {};

    sc_dims lhs_dims, rhs_dims;
    lhs_dims = get_inputs()[l]->details_.get_plain_dims();
    rhs_dims = get_inputs()[r]->details_.get_plain_dims();

    sc_dims elt_dims, bc_dims;
    if (bc_input_idx == r) {
        elt_dims = lhs_dims;
        bc_dims = rhs_dims;
    } else if (bc_input_idx == l) {
        elt_dims = rhs_dims;
        bc_dims = lhs_dims;
    }
    if (bc_dims.size() == 1 && bc_dims[0] == 1) {
        return std::vector<int> {-1};
    }
    std::vector<int> bc_axis;
    // broad-cast conditions 1: the shape of lhs and rhs not match
    if (elt_dims.size() != bc_dims.size()) {
        std::vector<int> common_axis(elt_dims.size(), 0);
        // from right to left
        int64_t i = elt_dims.size();
        for (int64_t j = bc_dims.size() - 1; j >= 0; j--) {
            while (i >= 1) {
                i--;
                if (elt_dims.at(i) == bc_dims.at(j)) {
                    common_axis.at(i) = 1;
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
        for (size_t j = 0; j < common_axis.size(); ++j)
            if (common_axis.at(j) == 1) bc_axis.emplace_back(j);
    }
    // broad-cast conditions 2: the shape of lhs and rhs match,
    // but length=1 in dims
    else {
        bool double_check_broadcast = false;
        for (size_t i = 0; i < elt_dims.size(); ++i) {
            if (elt_dims.at(i) != bc_dims.at(i)) {
                if (bc_dims.at(i) == 1) {
                    double_check_broadcast = true;
                } else if (!is_dynamic_dim(elt_dims.at(i))
                        && !is_dynamic_dim(bc_dims.at(i))) {
                    COMPILE_ASSERT(0,
                            "illegal elementwise operand found: "
                                    << utils::print_vector(elt_dims) << " , "
                                    << utils::print_vector(bc_dims));
                }
            }
        }
        if (double_check_broadcast) {
            for (size_t i = 0; i < elt_dims.size(); ++i) {
                if (elt_dims.at(i) == bc_dims.at(i)
                        || (is_dynamic_dim(elt_dims.at(i))
                                && is_dynamic_dim(bc_dims.at(i)))) {
                    bc_axis.emplace_back(i);
                }
            }
            if (bc_axis.empty()) { bc_axis.emplace_back(-1); }
        } else
            bc_axis = {};
    }
    return bc_axis;
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

select_op_t::select_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    op_name_ = "select";
    assert(ins.size() == 3);
    COMPILE_ASSERT(ins[1]->details_.get_plain_dims() == sc_dims {1},
            "shape of then tensor should be 1 for now");
    info_.inputs_ = ins;
    int maxtensor_idx = get_max_input() < 0 ? 1 : get_max_input();
    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(
                this, ins[maxtensor_idx]->details_));
    } else {
        info_.outputs_ = outs;
    }
    COMPILE_ASSERT(info_.outputs_[0]->details_.get_plain_dims()
                    == info_.inputs_[maxtensor_idx]->details_.get_plain_dims(),
            "output doesn't have the correct shape");

    info_.outputs_[0]->details_.dtype_ = info_.inputs_[1]->details_.dtype_;

    attrs_ = attrs;
    // TODO(shihao): improve the logic of plain_bc_axis to better satisfy
    // ternary cases
    plain_bc_axis_ = attrs.get_or_else("bc_axis", std::vector<int> {});
    if (plain_bc_axis_.empty()) { plain_bc_axis_ = infer_broadcast_axis(0, 2); }

    inplace_ = attrs.get_or_else("inplace", 2);
    // TODO(shihao): improve the logic of legelize inplace to better satisfy
    // ternary cases
    if (inplace_ == 1 && maxtensor_idx != 1) {
        inplace_ = -1;
    } else if (inplace_ == 2 && maxtensor_idx != 2) {
        inplace_ = -1;
    }

    info_.tensor_share_info_ = (inplace_ <= 0)
            ? std::unordered_map<int, std::vector<int>> {}
            : std::unordered_map<int, std::vector<int>> {{0, {inplace_}}};
}

select_op_t::select_op_t(graph_tensor_ptr cond, graph_tensor_ptr then,
        graph_tensor_ptr els, int inplace)
    : select_op_t({std::move(cond), std::move(then), std::move(els)}, {},
            {{"inplace", inplace}}) {
    inplace_ = inplace;
}

int select_op_t::get_broadcast_input(const int l, const int r) const {
    const sc_dims &lhs_dims = info_.inputs_[l]->details_.get_plain_dims();
    const sc_dims &rhs_dims = info_.inputs_[r]->details_.get_plain_dims();
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
            return lhs_dims.size() > rhs_dims.size() ? r : l;
        } else {
            return lhs_dp > rhs_dp ? r : l;
        }
    }
}

// TODO(shihao): improve the logic to make select more general
int select_op_t::get_max_input() const {
    const sc_dims &cond_dims = info_.inputs_[0]->details_.get_plain_dims();
    const sc_dims &then_dims = info_.inputs_[1]->details_.get_plain_dims();
    const sc_dims &else_dims = info_.inputs_[2]->details_.get_plain_dims();
    if (cond_dims == then_dims && then_dims == else_dims) {
        return -1;
    } else {
        auto cond_dp = get_dims_product(cond_dims);
        auto then_dp = get_dims_product(then_dims);
        auto else_dp = get_dims_product(else_dims);
        if (then_dp >= cond_dp && then_dp >= else_dp) {
            return 1;
        } else if (else_dp >= then_dp && else_dp >= cond_dp) {
            return 2;
        } else {
            return 0;
        }
    }
    return 1;
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
            "Unsupported case for binary_elementwise query format.");
    return sc_data_format_t(target_lt.get_format().format_code_, blocks);
}

void select_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    auto in0_format = info_.inputs_[0]->details_.get_format();
    auto in1_format = info_.inputs_[1]->details_.get_format();
    auto in2_format = info_.inputs_[2]->details_.get_format();
    COMPILE_ASSERT(info_.inputs_[1]->details_.get_plain_dims().size() == 1
                    && in1_format == sc_data_format_t(format_kinds::A),
            "The shape length and format of then lt shall confirm with the "
            "shape constraint ({1}).");

    int bc_input_idx = get_broadcast_input(0, 2);

    if (info_.inputs_[0]->details_.get_plain_dims().size()
            != info_.inputs_[2]->details_.get_plain_dims().size()) {
        COMPILE_ASSERT(in0_format == sc_data_format_t(format_kinds::A)
                        || in2_format == sc_data_format_t(format_kinds::A),
                "Unsupported format encountered in select query format.");
        in_formats.push_back({in0_format});
        in_formats.push_back({in1_format});
        in_formats.push_back({in2_format});
        out_formats.push_back({!bc_input_idx ? in2_format : in0_format});
    } else {
        if (!bc_input_idx) {
            auto target_format = infer_broadcast_format(
                    info_.inputs_[2]->details_, info_.inputs_[0]->details_);
            in_formats.push_back({target_format});
            in_formats.push_back({in1_format});
            in_formats.push_back({in2_format});
            out_formats.push_back({in2_format});
        } else {
            auto target_format = infer_broadcast_format(
                    info_.inputs_[0]->details_, info_.inputs_[2]->details_);
            in_formats.push_back({in0_format});
            in_formats.push_back({in1_format});
            in_formats.push_back({target_format});
            out_formats.push_back({in0_format});
        }
    }
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

// The logic below might be suitable for most fusible op, which has same
// slice ranges on inputs and outputs
void select_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    COMPILE_ASSERT(get_inputs().size() == 3, "select op is expected 3 inputs");
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map
            = search_known_slice_ranges(this, fsmap, stat_map);
    if (known_ranges_map.empty()) return;
    auto &outslice = fsmap.get(get_outputs()[0]);
    // if unkown slice ranges exist.
    if (known_ranges_map.size() < get_inputs().size()) {
        std::vector<int> known_idx(3, 0);
        known_idx[0]
                = known_ranges_map.find(0) != known_ranges_map.end() ? 1 : 0;
        known_idx[1]
                = known_ranges_map.find(1) != known_ranges_map.end() ? 1 : 0;
        known_idx[2]
                = known_ranges_map.find(2) != known_ranges_map.end() ? 1 : 0;
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
                        auto bc_axis = get_bc_axis(maxtensor_idx, i);
                        slice_range_list bc_arg_range_list
                                = infer_broadcast_arg_slice(
                                        known_ranges_map[maxtensor_idx],
                                        bc_axis, keep_dims);
                        known_ranges_map[i] = bc_arg_range_list;
                    }
                }
            } else {
                COMPILE_ASSERT(known_idx[0] || known_idx[2],
                        "input0 and input2 can't both be unknown");
                COMPILE_ASSERT(maxtensor_idx != 1,
                        "maxtensor_idx shouldn't be input1");
                // call get_bc_axis for input[0] && input[2]
                auto bc_axis = get_bc_axis(maxtensor_idx, 2 - maxtensor_idx);
                slice_range_list bc_range_list = infer_broadcast_slice(
                        known_ranges_map[2 - maxtensor_idx], bc_axis,
                        get_inputs()[maxtensor_idx]
                                ->details_.get_blocking_dims_expr(
                                        get_owner_graph()));
                known_ranges_map[maxtensor_idx] = bc_range_list;
                // deal with input[1]
                bc_axis = get_bc_axis(2 - maxtensor_idx, 1);
                bc_range_list = infer_broadcast_slice(
                        known_ranges_map[2 - maxtensor_idx], bc_axis,
                        get_inputs()[1]->details_.get_blocking_dims_expr(
                                get_owner_graph()));
                known_ranges_map[1] = bc_range_list;
            }
            // set the other unknown slice range by achieved
            // known_ranges_list
            set_unknown_slice_ranges(this, known_ranges_map, fsmap, stat_map);
            // set outputs slice range
            outslice = known_ranges_map[maxtensor_idx];
            return;
        } else {
            int known_tensor = -1;
            for (int i = 0; i < 3; i++) {
                if (known_idx[i] == 1) {
                    known_tensor = i;
                    break;
                }
            }
            COMPILE_ASSERT(
                    known_tensor >= 0, "At least one slice shall be known.");
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
    int maxtensor_idx = get_max_input();
    outslice = known_ranges_map[maxtensor_idx > -1
                    ? maxtensor_idx
                    : (inplace_ >= 1 ? inplace_ : 1)];
}

// l is always the larger side when passing parameters.
std::vector<int> select_op_t::get_bc_axis(const int l, const int r) const {
    COMPILE_ASSERT(l == get_max_input() || r == 1,
            "l should be the larger side when passing parameters");
    int bc_input_idx = get_broadcast_input(l, r);
    if (bc_input_idx == -1) {
        auto temp = get_inputs()[l]->details_.get_blocking_dims();
        std::vector<int> blocking_dims;
        for (size_t i = 0; i < temp.size(); i++) {
            blocking_dims.emplace_back(i);
        }
        return blocking_dims;
    }
    std::vector<int> plain_axis_ = infer_broadcast_axis(l, r);
    if (plain_axis_ == std::vector<int> {-1}) return plain_axis_;
    return transform_axis_plain2blocking(info_.inputs_[l], plain_axis_);
}

shape_rl_vec select_op_t::get_dynamic_shape_relations() const {
    shape_rl_vec ret;
    auto &cond_dims = get_inputs()[0]->details_.get_plain_dims();
    auto &inp2_dims = get_inputs()[2]->details_.get_plain_dims();
    auto &out_dims = get_outputs()[0]->details_.get_plain_dims();
    for (size_t i = 0; i < cond_dims.size(); i++) {
        if (is_dynamic_dim(cond_dims[i])) {
            ret.emplace_back(cond_dims[i], inp2_dims[i]);
        }
    }
    for (size_t i = 0; i < out_dims.size(); i++) {
        if (is_dynamic_dim(out_dims[i])) {
            ret.emplace_back(inp2_dims[i], out_dims[i]);
        }
    }
    return ret;
}

void compute_block_broadcast(const std::vector<const tensor_slice *> &src,
        const tensor_slice &dst, sc_op_info_t &info, const int maxtensor_idx,
        std::vector<int> bc_axis_1, std::vector<int> bc_axis_2,
        const vectorized_info_t &vx_info, const mask_compute_func_t &compute,
        sc_data_type_t dtype = datatypes::f32, size_t wkld = 0UL) {
    // nested loop vars
    std::vector<expr> iter_vars;
    // the indices for multiple inputs. First dim: the input, Second dim: the
    // dimensions in the tensor
    std::vector<expr> in_idx, in_bc_idx_1, in_bc_idx_2;
    // the indices for the output tensor
    std::vector<expr> dst_idx;

    COMPILE_ASSERT(maxtensor_idx >= 0 && maxtensor_idx != 1,
            "maxtensor_idx is expected to be 0 or 2")

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
    std::vector<expr> in_idx_tail = in_idx, in_bc_idx_1_tail = in_bc_idx_1,
                      in_bc_idx_2_tail = in_bc_idx_2, dst_idx_tail = dst_idx;
    auto tail_var = builder::make_var(
            datatypes::index, std::string("_fuseiter") + fusion_create_idx());
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

    // recover schedule loop
    for (int i = static_cast<int>(dst.get_shape().size() - 1); i >= 0; i--) {
        stmt body;
        // move broadcast op to body
        if (static_cast<int>(dst.get_shape().size()) == vx_info.axis + 1
                && i == vx_info.axis) {
            if (floor) {
                expr indexed_bc_input_1, indexed_bc_input_2;
                // IF last dim is included in bc_axis_1.
                if (bc_axis_1.back() == static_cast<int64_t>(vx_info.axis)) {
                    indexed_bc_input_1 = builder::make_indexing(
                            in_bc_tsl_1->tptr_, in_bc_idx_1, vx_info.lanes);
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
                    indexed_bc_input_2 = builder::make_indexing(
                            in_bc_tsl_2->tptr_, in_bc_idx_2, vx_info.lanes);
                }
                // IF last dim is excluded in bc_axis_2.
                else {
                    indexed_bc_input_2 = builder::make_broadcast(
                            builder::make_indexing(
                                    in_bc_tsl_2->tptr_, in_bc_idx_2),
                            static_cast<int>(vx_info.lanes));
                }
                bld->push_scope();
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
                        expr(floor), expr(int(vx_info.lanes)), bld->pop_scope(),
                        true, for_type::NORMAL);
                tcur.emplace_back(cur);
            }
            if (tail) {
                auto res_it_1 = std::find(
                        bc_axis_1.begin(), bc_axis_1.end(), vx_info.axis);
                if (res_it_1 != bc_axis_1.end()) {
                    in_bc_idx_1_tail[keep_dims_1
                                    ? vx_info.axis
                                    : (res_it_1 - bc_axis_1.begin())]
                            = tail_var;
                }
                expr indexed_bc_input_1_tail = builder::make_indexing(
                        in_bc_tsl_1->tptr_, in_bc_idx_1_tail);
                auto res_it_2 = std::find(
                        bc_axis_2.begin(), bc_axis_2.end(), vx_info.axis);
                if (res_it_2 != bc_axis_2.end()) {
                    in_bc_idx_2_tail[keep_dims_2
                                    ? vx_info.axis
                                    : (res_it_2 - bc_axis_2.begin())]
                            = tail_var;
                }
                expr indexed_bc_input_2_tail = builder::make_indexing(
                        in_bc_tsl_2->tptr_, in_bc_idx_2_tail);
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

// TODO(shihao): improve the logic to fix cond_dims == then_dims == else_dims
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
    if (maxtensor_idx != -1) {
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
        std::vector<int> bc;
        for (int i = 0; i < 3; i++) {
            if (i != maxtensor_idx) { bc.emplace_back(i); }
        }
        COMPILE_ASSERT(maxtensor_idx != 1, "maxtensor_idx can't be input1");
        std::vector<int> bc_axis_1 = get_bc_axis(maxtensor_idx, bc[0]);
        std::vector<int> bc_axis_2 = get_bc_axis(maxtensor_idx, bc[1]);
        // reuse broadcast op
        compute_block_broadcast(inputs, *dst[0], info_, maxtensor_idx,
                bc_axis_1, bc_axis_2, vx_info_, mask_compute_func_t(func),
                info_.outputs_[0]->details_.dtype_, wkld);
    } else {
        COMPILE_ASSERT(
                0, "Select op does not support non-broadcast cases for now.");
    }
}

// Pure virtual function in fusible_op_t class.
void select_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {}

// Pure virtual function in fusible_op_t class.
void select_op_t::prepare_fusion_data(fdata_map &fdmap) {}

OP_REGISTER(select_op_t, select)

} // namespace sc
