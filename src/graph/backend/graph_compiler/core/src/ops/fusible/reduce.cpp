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
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "reduce.hpp"
#include "util/bf16.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/transform/tensor2var.hpp>
#include <runtime/config.hpp>
#include <unordered_map>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

static inline any_map_t add_key(const any_map_t &attrs, int rd_op_) {
    auto ret = attrs;
    ret["rd_op"] = rd_op_;
    return ret;
}

reduce_sum_op_t::reduce_sum_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : reduce_op_t(ins, outs,
            add_key(attrs, static_cast<int>(reduce_operator::add))) {}

reduce_prod_op_t::reduce_prod_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : reduce_op_t(ins, outs,
            add_key(attrs, static_cast<int>(reduce_operator::mul))) {}

reduce_max_op_t::reduce_max_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : reduce_op_t(ins, outs,
            add_key(attrs, static_cast<int>(reduce_operator::max))) {}

reduce_min_op_t::reduce_min_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : reduce_op_t(ins, outs,
            add_key(attrs, static_cast<int>(reduce_operator::min))) {}

// compute the output data format after reduction given the plain reduction
// axis
static sc_data_format_t get_reduced_format(const sc_data_format_t &in_fmt,
        const std::vector<int> &rd_axis, size_t nlogical_dims) {
    auto base_fmt = in_fmt;
    // we should set the blocking of the reduce axies to 1
    int ax_offset = 0;
    for (int ax : rd_axis) {
        for (int blocking_idx :
                in_fmt.format_code_.collect_blocking_index(ax - ax_offset)) {
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
            "attrs of reduce op should have both reduce axis and operand "
            "information.");
    plain_rd_axis_ = attrs.get<std::vector<int>>("rd_axis");
    rd_op_ = reduce_operator(attrs.get<int>("rd_op"));
    keep_dims_ = attrs.get_or_else("keep_dims", true);

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
        if (keep_dims_) {
            out = logical_tensor_t(
                    get_reduced_format(ins[0]->details_.get_format(),
                            plain_rd_axis_,
                            ins[0]->details_.get_plain_dims().size()),
                    new_reduce_dims, ins[0]->details_.dtype_);
        } else {
            out = logical_tensor_t(
                    sc_data_format_t::get_plain_by_dims(new_reduce_dims.size()),
                    new_reduce_dims, ins[0]->details_.dtype_);
        }

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

reduce_op_t::reduce_op_t(graph_tensor_ptr v, const std::vector<int> &rd_axis,
        reduce_operator rd_op, bool keep_dims)
    : reduce_op_t({std::move(v)}, {},
            {{"rd_axis", rd_axis}, {"rd_op", static_cast<int>(rd_op)},
                    {"keep_dims", keep_dims}}) {
    // default is need_allocate
    info_.tensor_share_info_ = {};
}

void reduce_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    const auto &in_fmt = info_.inputs_[0]->details_.get_format();
    in_formats.push_back({in_fmt});
    if (keep_dims_) {
        out_formats.push_back({get_reduced_format(in_fmt, plain_rd_axis_,
                info_.inputs_[0]->details_.get_plain_dims().size())});
    } else {
        auto out_shape_size = info_.inputs_[0]->details_.get_plain_dims().size()
                - plain_rd_axis_.size();
        if (out_shape_size == 0) out_shape_size = 1;
        if (!in_fmt.is_blocking()) {
            out_formats.push_back(
                    {sc_data_format_t::get_plain_by_dims(out_shape_size)});
        } else {
            if (static_cast<int>(plain_rd_axis_.size())
                    == in_fmt.format_code_.norig_dims()) {
                // all reduced
                out_formats.push_back({sc_data_format_t(format_kinds::A)});
            } else {
                sc_data_format_t new_format;
                auto &new_code = new_format.format_code_;
                std::vector<int> old_code_idx, ordered_idx;
                // plain part
                for (int i = 0; i < in_fmt.format_code_.norig_dims(); i++) {
                    if (std::all_of(plain_rd_axis_.begin(),
                                plain_rd_axis_.end(), [&](int j) {
                                    return j != in_fmt.format_code_.get(i);
                                })) {
                        old_code_idx.push_back(in_fmt.format_code_.get(i));
                    }
                }
                for (int i = 0; i < static_cast<int>(old_code_idx.size());
                        i++) {
                    ordered_idx.push_back(i);
                }
                std::sort(ordered_idx.begin(), ordered_idx.end(),
                        [&old_code_idx](int p, int q) -> bool {
                            return old_code_idx[p] < old_code_idx[q];
                        });
                std::vector<int> new_code_idx(old_code_idx.size(), 0);
                for (int i = 0; i < static_cast<int>(old_code_idx.size());
                        i++) {
                    new_code_idx[ordered_idx[i]] = i;
                }
                // remained blocking part
                for (int i = in_fmt.format_code_.norig_dims();
                        i < in_fmt.format_code_.ndims(); i++) {
                    for (int j = 0; j < static_cast<int>(old_code_idx.size());
                            j++) {
                        if (old_code_idx[j] == in_fmt.format_code_.get(i)) {
                            new_code_idx.push_back(j);
                            break;
                        }
                    }
                }
                // infer new_format.format_code_ accoring to new_code_idx
                for (int i = 0; i < static_cast<int>(new_code_idx.size());
                        i++) {
                    new_code.set(i, new_code_idx[i]);
                }
                // copy blocks_ to new_format
                if (std::all_of(plain_rd_axis_.begin(), plain_rd_axis_.end(),
                            [&](int i) {
                                return !in_fmt.format_code_
                                                .collect_blocking_index(i)
                                                .empty();
                            })) {
                    int blocks_idx = 0;
                    for (int i = in_fmt.format_code_.norig_dims();
                            i < in_fmt.format_code_.ndims(); i++) {
                        if (std::none_of(plain_rd_axis_.begin(),
                                    plain_rd_axis_.end(), [&](int j) {
                                        return in_fmt.format_code_.get(i) == j;
                                    })) {
                            new_format.blocks_[blocks_idx] = in_fmt.blocks_[i
                                    - in_fmt.format_code_.norig_dims()];
                            blocks_idx++;
                        }
                    }
                } else {
                    new_format.blocks_[0] = in_fmt.blocks_[0];
                    new_format.blocks_[1] = in_fmt.blocks_[1];
                    new_format.blocks_[2] = in_fmt.blocks_[2];
                    new_format.blocks_[3] = in_fmt.blocks_[3];
                }
                out_formats.push_back({new_format});
            }
        }
    }
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
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

static slice_range_list infer_output_slice_range(bool is_reduce_compute,
        uint64_t vec_step, const slice_range_list &known_ranges_list,
        const std::vector<int> &real_rd_axis, bool keep_dims,
        sc_dim num_threads) {
    slice_range_list reduce_ranges_list;
    for (auto &known_ranges : known_ranges_list) {
        slice_range reduce_range;
        if (num_threads > 1) {
            reduce_range.emplace_back(std::pair<expr, expr> {0, 1});
        }
        // additional process is needed.
        for (size_t i = 0; i < known_ranges.size(); i++) {
            if (real_rd_axis.end()
                    != std::find(real_rd_axis.begin(), real_rd_axis.end(), i)) {
                if (keep_dims) {
                    reduce_range.emplace_back(std::pair<expr, expr> {0, 1});
                }
                // last-axis reduce
                if (is_reduce_compute && i == known_ranges.size() - 1) {
                    reduce_range.emplace_back(
                            std::pair<expr, expr> {0, vec_step});
                }
            } else {
                reduce_range.emplace_back(known_ranges.at(i));
            }
        }
        // reduce all and keep_dims = false;
        if ((known_ranges.size() == real_rd_axis.size()) && !keep_dims)
            reduce_range.emplace(
                    reduce_range.begin(), std::pair<expr, expr> {0, 1});
        reduce_ranges_list.emplace_back(reduce_range);
    }
    return reduce_ranges_list;
}

void update_reduce_op_fsmap(sc_op *ths, const graph_tensor_ptr &input,
        fslice_map &fsmap, infer_status_map_t &stat_map,
        const std::vector<int> &real_rd_axis) {
    auto required_axis = real_rd_axis;
    if (auto red_coll = ths->dyn_cast<reduce_collect_op_t>()) {
        if (red_coll->op_ == reduce_collect_op_t::kind::COPY) {
            required_axis.erase(required_axis.begin());
        }
    }
    auto &src_dim = input->details_.get_blocking_dims();
    // check the slice range whether meet the least demand of reduce op
    for (auto &src_range : fsmap.get(input)) {
        if (!slice_full_on_axis(src_dim, src_range, required_axis)) {
            ths->attrs_.set(
                    op_attr_key::fused_mode_hint, op_attr_key::break_pre_fuse);
            stat_map.append_ops_by_status(ths, infer_status_code::RETRY);
        }
    }
}

void reduce_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map
            = search_known_slice_ranges(this, fsmap, stat_map);
    // set the other unknown slice range by achieved known_ranges_list
    slice_range_list &known_ranges_list = known_ranges_map[0];
    // COMPILE_ASSERT(known_ranges_list.size() == 1,
    //         "Reduce Op should not accept inconsequent or irruglar
    //         slice");
    auto real_rd_axis = get_rd_axis();
    update_reduce_op_fsmap(
            this, get_inputs()[0], fsmap, stat_map, real_rd_axis);
    if (!stat_map.is_recursive_mode() && stat_map.is_retry()) return;
    fsmap.get(get_outputs()[0]) = infer_output_slice_range(
            false, 0, known_ranges_list, real_rd_axis, keep_dims_, 1);
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
            auto real_dims = get_inputs()[0]->details_.get_blocking_dims_expr(
                    get_owner_graph());
            // idx record real idx in range, used to skip range {0,1} when
            // keep_dims=false
            int idx = 0;
            for (size_t i = 0; i < real_dims.size(); i++) {
                if (real_rd_axis.end()
                        != std::find(
                                real_rd_axis.begin(), real_rd_axis.end(), i)) {
                    reduce_range.emplace_back(
                            std::pair<expr, expr> {0, real_dims.at(i)});
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

void infer_reduce_binding_axis(fusible_op_t *cur, bound_axis_map &bdax_map,
        const std::vector<int> &plain_rd_axis, bool keep_dims) {
    auto known_axis_map = search_known_bound_axis(cur, bdax_map);
    if (!bdax_map.get(cur->get_outputs()[0]).empty()) return;
    if (keep_dims) {
        bdax_map.get(cur->get_outputs()[0]) = known_axis_map[0];
    } else {
        std::vector<int> non_rd_axis;
        auto plain_dims = cur->get_inputs()[0]->details_.get_plain_dims();
        for (size_t i = 0; i < plain_dims.size(); i++) {
            if (plain_rd_axis.end()
                    != std::find(plain_rd_axis.begin(), plain_rd_axis.end(),
                            static_cast<int>(i)))
                continue;
            else
                non_rd_axis.emplace_back(i);
        }
        bound_axis out_axis;
        for (auto &bd_ax : known_axis_map[0]) {
            std::vector<int> ret;
            for (auto &ax : bd_ax) {
                auto iter
                        = std::find(non_rd_axis.begin(), non_rd_axis.end(), ax);
                if (iter != non_rd_axis.end()) {
                    ret.emplace_back(iter - non_rd_axis.begin());
                }
            }
            out_axis.emplace_back(ret);
        }
        bdax_map.get(cur->get_outputs()[0]) = out_axis;
    }
    // auto expand for partial reduce compute
    if (auto red_comp = cur->dyn_cast<reduce_compute_op_t>()) {
        if (red_comp->is_partial_reduce()) {
            for (auto &bd_ax : bdax_map.get(cur->get_outputs()[0])) {
                for (auto &ax : bd_ax)
                    ax++;
            }
        }
    }
    set_unknown_axis_binding(cur, known_axis_map, bdax_map);
}

void pre_reduce_binding_axis(fusible_op_t *cur, bound_axis_map &bdax_map,
        const std::vector<int> &plain_rd_axis, bool keep_dims) {
    auto outaxis = bdax_map.get(cur->get_outputs()[0]);
    COMPILE_ASSERT(!outaxis.empty(),
            "Unknown output axis found, could not pre bind axis")
    // auto shrink for partial reduce compute
    if (auto red_comp = cur->dyn_cast<reduce_compute_op_t>()) {
        if (red_comp->is_partial_reduce()) {
            for (auto &bd_ax : outaxis) {
                for (auto &ax : bd_ax)
                    ax--;
            }
        }
    }
    auto &input = cur->get_inputs()[0];
    auto &inpaxis = bdax_map.get(input);

    if (inpaxis.empty()) {
        if (keep_dims) {
            inpaxis = outaxis;
        } else {
            std::vector<int> non_rd_axis;
            auto plain_dims = cur->get_inputs()[0]->details_.get_plain_dims();
            for (size_t i = 0; i < plain_dims.size(); i++) {
                if (plain_rd_axis.end()
                        != std::find(plain_rd_axis.begin(), plain_rd_axis.end(),
                                static_cast<int>(i)))
                    continue;
                else
                    non_rd_axis.emplace_back(i);
            }
            for (auto &bd_ax : outaxis) {
                std::vector<int> ret;
                ret.reserve(bd_ax.size());
                for (auto &ax : bd_ax) {
                    ret.emplace_back(non_rd_axis[ax]);
                }
                inpaxis.emplace_back(ret);
            }
        }
        if (auto bd_op
                = input->producer_owner_
                          ->dyn_cast<op_traits::mixed_partition_acceptable>()) {
            bd_op->pre_binding_axis(bdax_map);
        }
    }
}

void reduce_op_t::infer_binding_axis(bound_axis_map &bdax_map) {
    infer_reduce_binding_axis(this, bdax_map, plain_rd_axis_, keep_dims_);
}
void reduce_op_t::pre_binding_axis(bound_axis_map &bdax_map) {
    pre_reduce_binding_axis(this, bdax_map, plain_rd_axis_, keep_dims_);
}

shape_rl_vec reduce_op_t::get_dynamic_shape_relations() const {
    shape_rl_vec ret;
    auto &in_dims = get_inputs()[0]->details_.get_plain_dims();
    auto &out_dims = get_outputs()[0]->details_.get_plain_dims();
    auto rd_axis = get_rd_axis();
    for (size_t i = 0; i < out_dims.size(); i++) {
        if (is_dynamic_dim(out_dims[i])) {
            ret.emplace_back(in_dims[i], out_dims[i]);
        }
    }
    return ret;
}

static union_val get_init_val_for_reduce(
        reduce_operator rd_op, sc_data_type_t dtype) {
    variant<float, int64_t> init_value;
    bool is_int = utils::is_one_of(dtype.type_code_, sc_data_etype::U8,
            sc_data_etype::U32, sc_data_etype::S8, sc_data_etype::S32);
    if (rd_op == reduce_operator::mul) {
        if (is_int) {
            init_value = int64_t(1);
        } else {
            init_value = 1.f;
        }
    } else if (rd_op == reduce_operator::add) {
        if (is_int) {
            init_value = int64_t(0);
        } else {
            init_value = 0.f;
        }
    } else if (rd_op == reduce_operator::min) {
        init_value = numeric_limits_maximum(dtype.type_code_);
    } else {
        COMPILE_ASSERT(rd_op == reduce_operator::max, "wrong reduce kind");
        init_value = numeric_limits_minimum(dtype.type_code_);
    }
    return init_value.cast<union_val>();
}

using binary_tir_gen_f = expr (*)(const expr_c &, const expr_c &);
static binary_tir_gen_f get_binary_by_reduce_op(reduce_operator rdop) {
    switch (rdop) {
        case reduce_operator::add: return builder::make_add; break;
        case reduce_operator::mul: return builder::make_mul; break;
        case reduce_operator::max: return builder::make_max; break;
        case reduce_operator::min: return builder::make_min; break;
    }
    return nullptr;
}

using unary_tir_gen_f = expr (*)(const expr_c &);
static unary_tir_gen_f get_binary_reduce_by_reduce_op(reduce_operator rdop) {
    switch (rdop) {
        case reduce_operator::add: return builder::make_reduce_add; break;
        case reduce_operator::mul: return builder::make_reduce_mul; break;
        case reduce_operator::max: return builder::make_reduce_max; break;
        case reduce_operator::min: return builder::make_reduce_min; break;
    }
    return nullptr;
}

// reduce all tensor_slice into sum, NOTE here src is a common
// tensor_slice but dst maybe whole temp_buffer because output shape of
// reduction is not equal to src, so it will allocate a new buffer
static void compute_block_reduce(sc_graph_t &graph, const sc_op_info_t &info,
        const std::vector<const tensor_slice *> &src, const tensor_slice &dst,
        reduce_operator rd_op, std::vector<int> rd_axis, bool keep_dims,
        const vectorized_info_t &vx_info, sc_data_type_t dtype,
        any_map_t &attrs, size_t wkld = 0UL, bool is_dynamic = false) {
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
    // TODO(xxx): need more detailed judgement for `last_dim = 1` case
    int last_dim = -1;
    auto &dim_tmp = src[0]->get_shape().back();
    if (dim_tmp.isa<constant>()) {
        last_dim = get_const_as_int(dim_tmp.checked_as<constant_c>());
    }

    for (unsigned i = 0; i < src.at(0)->nslice_dims(); i++) {
        iter_vars.emplace_back(range_from_outer_loop(src.at(0)->get_ranges()[i])
                        ? expr(0)
                        : builder::make_var(datatypes::index,
                                std::string("_fuseiter")
                                        + fusion_create_idx()));
        src_idx.emplace_back(iter_vars.back());
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
    const int INVALID_AXIS_MASK = -64;
    int last_axis_mask = INVALID_AXIS_MASK;
    std::unordered_map<expr, std::pair<expr, expr>> conditions;
    compute_mask_and_generate_condition(graph, src,
            info.inputs_[0]->details_.get_plain_dims(),
            info.inputs_[0]->details_.get_format(), iter_vars, vx_info.lanes,
            conditions, last_axis_mask);
    // need mask
    expr mask_directly;
    auto slice_len = src[0]->get_shape().back();
    int lanes = static_cast<int>(vx_info.lanes);
    auto floor = do_cast_and_fold(slice_len / lanes * lanes);
    auto tail = do_cast_and_fold(slice_len % lanes);
    int floor_int = 0;
    int tail_int = 0;
    if (floor.isa<constant>()) {
        floor_int = get_expr_as_int(floor);
        tail_int = get_expr_as_int(tail);
        COMPILE_ASSERT((floor_int + tail_int), "Don't support shape len = 0.");
    }
    bool is_lastdim_meet_require
            = tail.isa<constant>() && tail_int == 1 && floor_int == 0;
    if (is_lastdim_meet_require) {
        lanes = 1;
    } else if (last_dim % lanes) {
        if (rd_op == reduce_operator::add) {
            mask_directly = last_dim_generate_mask(
                    src_idx.back(), floor, slice_len, lanes);
        } else {
            lanes = 1;
        }
    }
    dst_idx = !dst_idx.empty() ? dst_idx : std::vector<expr> {expr {0}};
    expr indexed_target = builder::make_indexing(dst.tptr_, dst_idx,
            !last_axis_reduce ? lanes : 1,
            !last_axis_reduce ? mask_directly : expr());
    expr indexed_input = builder::make_indexing(
            src.at(0)->tptr_, src_indices.at(0), lanes, mask_directly);

    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder is set");
    stmt body, cur;
    auto reduce_value
            = builder::make_var(sc_data_type_t(dtype.type_code_, lanes),
                    "reduce_" + fusion_create_var_idx());
    auto inter_padding_var
            = builder::make_var(sc_data_type_t(dtype.type_code_, lanes),
                    "reduce_" + fusion_create_var_idx());

    union_val value = get_init_val_for_reduce(rd_op, dtype);
    stmt asnode = make_stmt<assign_node_t>(reduce_value,
            make_expr<constant_node>(
                    value, sc_data_type_t(dtype.type_code_, lanes)));
    auto define_reduce
            = make_stmt<define_node_t>(reduce_value, linkage::local, expr());
    stmt padding_middle_asnode = make_stmt<assign_node_t>(inter_padding_var,
            make_expr<constant_node>(
                    value, sc_data_type_t(dtype.type_code_, lanes)));
    auto define_padding_reduce = make_stmt<define_node_t>(
            inter_padding_var, linkage::local, expr());

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

    bool is_padding_ir = false;
    for (auto i : new_loop_order) {
        if (i == new_loop_order.front()) {
            auto cond_it = conditions.find(iter_vars[i]);
            is_padding_ir = cond_it != conditions.end()
                    && rd_op != reduce_operator::add;
            // reduce add don't need to consider padding case
            if (is_padding_ir) {
                assert(last_axis_mask != INVALID_AXIS_MASK);
                std::vector<stmt_c> cur_list;
                cur_list.emplace_back(builder::make_assign_unattached(
                        inter_padding_var, indexed_input));

                // calculate mask, upper_bound - cur_index
                auto upper_bound_int = builder::make_cast(
                        datatypes::s32, cond_it->second.second);
                auto cur_index_int = builder::make_cast(
                        datatypes::s32, cond_it->second.first);
                auto cur_step = builder::make_min(
                        builder::make_max(builder::make_sub(upper_bound_int,
                                                  cur_index_int),
                                0),
                        lanes);
                stmt mask_def;
                auto mask
                        = generate_mask_var_by_step(mask_def, cur_step, lanes);
                cur_list.emplace_back(mask_def);

                cur_list.emplace_back(builder::make_assign_unattached(
                        reduce_value,
                        builder::make_select(mask,
                                get_binary_by_reduce_op(rd_op)(
                                        reduce_value, inter_padding_var),
                                reduce_value)));
                cur = builder::make_stmts_unattached(cur_list);
            } else {
                cur = make_stmt<assign_node_t>(reduce_value,
                        get_binary_by_reduce_op(rd_op)(
                                indexed_input, reduce_value));
            }
            cur->attr()[op_traits::workload_computable_t::workload_number]
                    = wkld;
        }
        // Do not generate those dummy loops
        if (iter_vars.at(i).isa<var>()) {
            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            // insert mask define.
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0), src.at(0)->get_shape().at(i),
                    i == static_cast<int>(src.at(0)->nslice_dims() - 1)
                            ? expr(static_cast<int>(lanes))
                            : expr(1),
                    std::move(body), true, for_type::NORMAL);
        }
        // the outer-most reduction axis
        if (i == rd_axis.front()) {
            std::vector<stmt_c> res;
            std::vector<stmt_c> rd_no_padding_assign {define_reduce, asnode,
                    std::move(cur),
                    make_stmt<assign_node_t>(indexed_target,
                            lanes > 1 && last_axis_reduce
                                    ? get_binary_reduce_by_reduce_op(rd_op)(
                                            reduce_value)
                                    : reduce_value)};
            if (is_padding_ir) {
                std::vector<stmt_c> rd_padding_assign {
                        define_padding_reduce, padding_middle_asnode};
                rd_padding_assign.insert(rd_padding_assign.end(),
                        rd_no_padding_assign.begin(),
                        rd_no_padding_assign.end());
                res = std::move(rd_padding_assign);
            } else {
                res = std::move(rd_no_padding_assign);
            }

            cur = builder::make_stmts_unattached(res);
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
    return transform_axis_plain2blocking(info_.inputs_[0], plain_rd_axis_);
}

int reduce_op_t::get_compressed_rd_axis_int() const {
    auto rd_axis = get_rd_axis();
    int ret = 0;
    for (auto &rd : rd_axis) {
        ret |= (1 << rd);
    }
    return ret;
}

sc_dims reduce_op_t::get_bwise_fuse_shrink_dims() {
    if (!keep_dims_) return {};
    auto real_rd_axis = get_rd_axis();
    auto input_dims = info_.outputs_[0]->details_.get_blocking_dims();
    int offset = op_traits::batchwise_shrinkable_t::get_shrinkable_offset(
            info_.outputs_[0]);
    int min_rd_axis
            = (*std::min_element(real_rd_axis.begin(), real_rd_axis.end()));
    return {input_dims.begin(),
            input_dims.begin() + std::min(offset, min_rd_axis + 1)};
}

void reduce_op_t::collect_shrinked_lt_map(int bw_size, gt2gt_map &bw_lt_map) {
    auto rd_axis = get_rd_axis();
    int invalid_size = 0;
    for (auto &ax : rd_axis) {
        if (ax < bw_size)
            invalid_size++;
        else
            break;
    }
    op_traits::batchwise_shrinkable_t::record_shrinked_gt(
            bw_lt_map, get_inputs()[0], bw_size);
    op_traits::batchwise_shrinkable_t::record_shrinked_gt(bw_lt_map,
            get_outputs()[0], keep_dims_ ? bw_size : (bw_size - invalid_size));
}

void reduce_op_t::collect_shrinked_axis_map(
        int bw_size, gt2axis_map &bw_axis_map) {
    auto rd_axis = get_rd_axis();
    std::vector<int> bw_axis;
    int valid_cnt = 0;
    for (int i = 0; i < bw_size; i++) {
        auto iter = std::find(rd_axis.begin(), rd_axis.end(), i);
        if (iter != rd_axis.end()) {
            bw_axis.emplace_back(-1);
        } else {
            bw_axis.emplace_back(valid_cnt++);
        }
    }
    op_traits::batchwise_shrinkable_t::record_shrinked_axis(
            bw_axis_map, get_inputs()[0], bw_size);
    if (keep_dims_) {
        op_traits::batchwise_shrinkable_t::record_shrinked_axis(
                bw_axis_map, get_outputs()[0], bw_size);
    } else {
        op_traits::batchwise_shrinkable_t::record_shrinked_axis(
                bw_axis_map, get_outputs()[0], bw_axis);
    }
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
    auto vector_lanes
            = vectorize_step(ctx, info_.inputs_[0]->details_.dtype_.type_code_);
    vx_info_.lanes = vector_lanes;

    compute_block_reduce(get_owner_graph(), info_, inputs, *dst[0], rd_op_,
            real_rd_axis, keep_dims_, vx_info_,
            info_.inputs_[0]->details_.dtype_, attrs_, wkld, is_dynamic());
}

size_t reduce_op_t::compute_workload(const std::vector<shape_dtype_pair> &ins,
        const std::vector<shape_dtype_pair> &outs) {
    auto &shape = ins[0].first;
    auto &dtype = ins[0].second;
    auto real_rd_axis = get_rd_axis();
    size_t wkld = utils::get_sizeof_type(dtype) * read_weight;
    wkld += utils::get_sizeof_type(dtype) * write_weight;
    return wkld;
}

// assume that the first axis is parallel. we currently can use
// reduce_compute+reduce_collect when reduction axis is not outside of the
// parallel axis and is not last axis reduction(for performance)
bool reduce_op_t::can_split_op() const {
    if (runtime_config_t::get().get_num_threads() == 1) { return true; }
    auto ax = get_rd_axis();
    int last_dim = get_inputs()[0]->details_.get_blocking_dims().size() - 1;
    for (auto i : ax) {
        if (i == 0) return false;
    }
    return true;
}

graph_tensor_ptr reduce_op_t::split_op(
        const context_ptr &ctx, sc_graph_t &graph, int num_threads) {
    auto rd_ax = get_rd_axis();

    int last_dim = get_inputs()[0]->details_.get_blocking_dims().size() - 1;
    bool last_axis = false;
    for (auto i : rd_ax) {
        if (last_dim == i) {
            last_axis = true;
            break;
        }
    }

    auto first_out = get_outputs()[0]->copy();
    first_out->producer_owner_ = nullptr;
    auto second_out = get_outputs()[0]->copy();
    second_out->producer_owner_ = nullptr;

    bool is_bf16 = get_inputs()[0]->details_.dtype_ == datatypes::bf16
            && rd_op_ != reduce_operator::max && rd_op_ != reduce_operator::min;
    if (is_bf16) {
        first_out->details_.dtype_ = datatypes::f32;
        second_out->details_.dtype_ = datatypes::f32;
    }

    if (last_axis) {
        auto vec_step
                = vectorize_step(ctx, first_out->details_.dtype_.type_code_);
        auto new_dims = first_out->details_.get_blocking_dims();
        if (num_threads > 1) { new_dims.insert(new_dims.begin(), num_threads); }
        new_dims.push_back(vec_step);
        first_out->details_.set_blocking_dims(new_dims);
    } else {
        // if partial reduce, the output has a leading dimension of thread id
        auto new_dims = first_out->details_.get_blocking_dims();
        if (num_threads > 1) { new_dims.insert(new_dims.begin(), num_threads); }
        first_out->details_.set_blocking_dims(new_dims);
    }

    auto first = graph.make<reduce_compute_op_t>(get_inputs()[0], first_out,
            rd_ax, rd_op_, keep_dims_, /*local_mode*/ false);

    sc_op_ptr second;
    if (num_threads > 1) {
        std::vector<int> rx_ax {0};
        if (last_axis) {
            rx_ax.push_back(first_out->details_.get_blocking_dims().size() - 1);
        }
        // add a standalone reduce op after partial reduce
        second = graph.make("reduce", {first_out}, {second_out},
                {
                        {"rd_axis", std::move(rx_ax)},
                        {"rd_op", static_cast<int>(rd_op_)},
                        {"keep_dims", false},
                });
    } else {
        second = graph.make<reduce_collect_op_t>(first_out, second_out, rd_ax,
                rd_op_, keep_dims_,
                last_axis ? reduce_collect_op_t::LAST_AXIS_COLLECT
                          : reduce_collect_op_t::NOOP);
    }
    if (is_bf16) {
        auto out_tsr = second_out->copy();
        out_tsr->details_.dtype_ = datatypes::bf16;
        out_tsr->producer_owner_ = nullptr;
        second = graph.make(
                "cast", {second_out}, {out_tsr}, {{"dtype", datatypes::bf16}});
        second_out = out_tsr;
    }

    get_outputs()[0]->replace_with(second_out);
    remove();
    return second_out;
}

OP_REGISTER(reduce_op_t, reduce)

void reduce_impl_op_t::prepare_fusion_data(fdata_map &fdmap) {
    fdmap.get(info_.inputs_[0]).use_count_++;
}
void reduce_impl_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    throw std::runtime_error("Cannot query_format for this internal op");
}
void reduce_impl_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    throw std::runtime_error("Cannot pre_slice_ranges for this internal op");
}

void reduce_impl_op_t::infer_binding_axis(bound_axis_map &bdax_map) {
    infer_reduce_binding_axis(this, bdax_map,
            transform_axis_blocking2plain(
                    get_inputs()[0]->details_, real_rd_axis_),
            keep_dims_);
}

void reduce_impl_op_t::pre_binding_axis(bound_axis_map &bdax_map) {
    pre_reduce_binding_axis(this, bdax_map,
            transform_axis_blocking2plain(
                    get_inputs()[0]->details_, real_rd_axis_),
            keep_dims_);
}

reduce_impl_op_t::reduce_impl_op_t(const graph_tensor_ptr &in,
        const graph_tensor_ptr &old_out, const std::vector<int> &rd_axis,
        reduce_operator rd_op, bool keep_dims)
    : real_rd_axis_(rd_axis), rd_op_(rd_op), keep_dims_(keep_dims) {
    info_.inputs_ = {in};
    info_.outputs_ = {old_out};
    std::sort(real_rd_axis_.begin(), real_rd_axis_.end());
}
// get real reduce axis, generaly, you should set rd_axis on plain format
// semantics.
const std::vector<int> &reduce_impl_op_t::get_rd_axis() const {
    return real_rd_axis_;
}

bool reduce_compute_op_t::can_split_op() const {
    if (runtime_config_t::get().get_num_threads() == 1) { return false; }
    auto last_axis = get_inputs()[0]->details_.get_blocking_dims().size() - 1;
    bool last_axis_reduce
            = static_cast<unsigned>(real_rd_axis_.back()) == last_axis;
    return is_partial_reduce() && !last_axis_reduce;
}

graph_tensor_ptr reduce_compute_op_t::split_op(
        const context_ptr &ctx, sc_graph_t &graph, int num_threads) {
    assert(can_split_op());

    auto first_out = get_outputs()[0]->copy();
    first_out->producer_owner_ = nullptr;
    auto second_out = get_outputs()[0]->copy();
    second_out->producer_owner_ = nullptr;

    // remove the thread-id dimension
    auto new_first_dims = first_out->details_.get_blocking_dims();
    new_first_dims.erase(new_first_dims.begin());
    first_out->details_.set_blocking_dims(new_first_dims);

    auto first = graph.make<reduce_compute_op_t>(get_inputs()[0], first_out,
            real_rd_axis_, rd_op_, keep_dims_, /*local_mode*/ true);
    auto second = graph.make<reduce_collect_op_t>(first_out, second_out,
            real_rd_axis_, rd_op_, keep_dims_, reduce_collect_op_t::COPY);
    get_outputs()[0]->replace_with(second_out);
    remove();
    return second_out;
}

sc_op_ptr reduce_compute_op_t::copy(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, sc_graph_t &mgr) {
    auto ret = mgr.make<reduce_compute_op_t>(ins.at(0), outs.at(0),
            real_rd_axis_, rd_op_, keep_dims_, local_mode_);
    ret->copy_dispatch_key_set_from_op(shared_from_this());
    ret->attrs_ = attrs_;
    return ret;
}

reduce_compute_op_t::reduce_compute_op_t(const graph_tensor_ptr &in,
        const graph_tensor_ptr &old_out, const std::vector<int> &rd_axis,
        reduce_operator rd_op, bool keep_dims, bool local_mode)
    : reduce_impl_op_t(in, old_out, rd_axis, rd_op, keep_dims)
    , local_mode_(local_mode) {
    op_name_ = "reduce_compute";

    size_t in_dims = in->details_.get_blocking_dims().size();
    size_t expected_dims = in_dims;
    // if no keep dims
    if (!keep_dims_) {
        expected_dims
                = std::max((size_t)1, (expected_dims - real_rd_axis_.size()));
    }
    if (is_partial_reduce()) {
        expected_dims++;
        attrs_[op_attr_key::break_post_fuse] = true;
    }
    // if last axis reduction
    if (real_rd_axis_.back() == static_cast<int>(in_dims) - 1) {
        expected_dims += 1;
    }
    COMPILE_ASSERT(
            expected_dims == old_out->details_.get_blocking_dims().size(),
            "Bad output dims for reduce_compute op:"
                    << expected_dims << " v.s. "
                    << old_out->details_.get_blocking_dims().size());
}

bool reduce_compute_op_t::is_partial_reduce() const {
    // single thread can do first axis reduction without partial reduce
    return !local_mode_ && real_rd_axis_.front() == 0
            && runtime_config_t::get().get_num_threads() != 1;
}

void reduce_compute_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    slice_range_map known_ranges_map
            = search_known_slice_ranges(this, fsmap, stat_map);
    // set the other unknown slice range by achieved known_ranges_list
    slice_range_list &known_ranges_list = known_ranges_map[0];

    auto &real_rd_axis = get_rd_axis();
    sc_dim num_threads = 1;
    if (is_partial_reduce()) {
        num_threads = get_outputs()[0]->details_.get_blocking_dims()[0];
    }
    // if is last axis reduce, the last dim is the vec step
    auto vec_step = get_outputs()[0]->details_.get_blocking_dims().back();
    fsmap.get(get_outputs()[0]) = infer_output_slice_range(true, vec_step,
            known_ranges_list, real_rd_axis, keep_dims_, num_threads);
}

void reduce_compute_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);
    // set default vectorized information
    auto &real_rd_axis = get_rd_axis();
    auto last_axis = get_inputs()[0]->details_.get_blocking_dims().size() - 1;
    bool last_axis_reduce
            = static_cast<unsigned>(real_rd_axis.back()) == last_axis;
    vx_info_.axis = inputs[0]->get_shape().size() - 1;
    vx_info_.lanes
            = vectorize_step(ctx, info_.inputs_[0]->details_.dtype_.type_code_);
    bool is_partial = is_partial_reduce();
    auto ths = this;
    auto func = [&](const std::vector<expr> &in,
                        std::vector<expr::lvalue_proxy_t> &out) -> stmt {
        indexing indexing_nd = out[0].get().checked_as<indexing>();
        auto lanes = indexing_nd->dtype_.lanes_;
        // if keep dims, set reduction axis to 0, else remove the axis from
        // indexing node
        if (ths->keep_dims_) {
            for (auto ax : real_rd_axis) {
                indexing_nd->idx_.at(ax) = 0;
            }
            if (is_partial) {
                indexing_nd->idx_.insert(indexing_nd->idx_.begin(),
                        builtin::get_thread_id_func()());
            }
            if (last_axis_reduce) { indexing_nd->idx_.emplace_back(0); }
        } else {
            std::vector<expr> new_idx;
            if (is_partial) {
                // for partial reduce, the first axis should be thread id
                new_idx.emplace_back(builtin::get_thread_id_func()());
            }
            for (auto itr = indexing_nd->idx_.begin();
                    itr != indexing_nd->idx_.end(); ++itr) {
                bool remove = false;
                auto axis_id = itr - indexing_nd->idx_.begin();
                for (auto ax : real_rd_axis) {
                    // if the axis is reduced and is not last axis, remove
                    if (axis_id == ax) {
                        if (ax == static_cast<int>(last_axis)) {
                            // if is last axis reduction, set index to 0
                            *itr = 0;
                        } else {
                            remove = true;
                            break;
                        }
                    }
                }
                if (!remove) { new_idx.emplace_back(std::move(*itr)); }
            }
            indexing_nd->idx_ = std::move(new_idx);
        }
        expr result = get_binary_by_reduce_op(ths->rd_op_)(indexing_nd, in[0]);
        return builder::make_assign_unattached(indexing_nd, result);
    };

    compute_vectorized_op(ctx, get_owner_graph(), inputs, *dst[0], info_,
            vx_info_, mask_compute_func_t(func), mask_compute_func_t(func),
            attrs_, wkld, false, inputs[0], /*unroll*/ local_mode_);
}

void reduce_compute_op_t::set_reduce_buffer(const tensor &buf) {
    buf->init_value_ = tensor_node::make_tensor_initializer(
            get_init_val_for_reduce(rd_op_, buf->elem_dtype_));
    if (local_mode_) buf->attr()[attr_keys::must_tensor2var] = true;
}

reduce_collect_op_t::reduce_collect_op_t(const graph_tensor_ptr &in,
        const graph_tensor_ptr &old_out, const std::vector<int> &rd_axis,
        reduce_operator rd_op, bool keep_dims, reduce_collect_op_t::kind op)
    : reduce_impl_op_t(in, old_out, rd_axis, rd_op, keep_dims), op_(op) {
    op_name_ = "reduce_collect";
    if (in->details_.get_blocking_dims()
            == old_out->details_.get_blocking_dims()) {
        info_.tensor_share_info_[0] = {0};
    } else {
        info_.tensor_share_info_ = {};
    }
}

void reduce_collect_op_t::set_reduce_buffer(const tensor &buf) {
    if (!is_place_holder_op()) {
        buf->init_value_ = tensor_node::make_tensor_initializer(
                get_init_val_for_reduce(rd_op_, buf->elem_dtype_));
    }
}

sc_op_ptr reduce_collect_op_t::copy(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, sc_graph_t &mgr) {
    auto ret = mgr.make<reduce_collect_op_t>(
            ins.at(0), outs.at(0), real_rd_axis_, rd_op_, keep_dims_, op_);
    ret->copy_dispatch_key_set_from_op(shared_from_this());
    ret->attrs_ = attrs_;
    return ret;
}

void reduce_collect_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    slice_range_map known_ranges_map
            = search_known_slice_ranges(this, fsmap, stat_map);
    // set the other unknown slice range by achieved known_ranges_list
    slice_range_list &known_ranges_list = known_ranges_map[0];
    // get producer
    auto &producer = get_inputs()[0]->producer_owner_;
    COMPILE_ASSERT(producer->isa<reduce_compute_op_t>(),
            "reduce_collect_op_t can only be placed after "
            "reduce_compute_op_t, but got "
                    << producer->op_name_);
    auto &input = producer->get_inputs().at(0);
    auto &real_rd_axis = get_rd_axis();
    update_reduce_op_fsmap(this, input, fsmap, stat_map, real_rd_axis);
    if (!stat_map.is_recursive_mode() && stat_map.is_retry()) return;
    if (op_ == LAST_AXIS_COLLECT) {
        // if is not placeholder op, and don't keep dims, we will add an
        // additional axis at the end, when in reduce_compute. need to drop
        // the last axis
        for (auto &range : known_ranges_list) {
            range.pop_back();
        }
    } else if (op_ == COPY) {
        // if is copy-mode, the output has an additional dimension for thread-id
        for (auto &range : known_ranges_list) {
            range.insert(range.begin(), std::pair<expr, expr> {0, 1});
        }
    }
    fsmap.get(get_outputs()[0]) = known_ranges_list;
}

void reduce_collect_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    if (op_ == COPY) {
        vx_info_.axis = dst[0]->get_shape().size() - 1;
        auto vec_lanes = vectorize_step(
                ctx, info_.inputs_[0]->details_.dtype_.type_code_);
        vx_info_.lanes = vec_lanes;
        auto ths = this;
        auto func = [&](const std::vector<expr> &in,
                            std::vector<expr::lvalue_proxy_t> &out) -> stmt {
            indexing out_nd = out[0].get().checked_as<indexing>();
            //  add the axis to indexing node
            out_nd->idx_.front()
                    = out_nd->idx_.front() + builtin::get_thread_id_func()();
            return builder::make_assign_unattached(
                    out[0], get_binary_by_reduce_op(rd_op_)(out[0], in[0]));
        };
        compute_vectorized_op(ctx, get_owner_graph(), inputs, *dst[0], info_,
                vx_info_, mask_compute_func_t(func), mask_compute_func_t(func),
                attrs_, 0, false, dst[0], /*unroll*/ true);
    } else if (op_ == LAST_AXIS_COLLECT) {
        // set default vectorized information
        auto &real_rd_axis = get_rd_axis();
        auto last_axis
                = get_inputs()[0]->details_.get_blocking_dims().size() - 1;
        vx_info_.axis = dst[0]->get_shape().size() - 1;
        auto vec_lanes = vectorize_step(
                ctx, info_.inputs_[0]->details_.dtype_.type_code_);
        vx_info_.lanes = 1;
        auto ths = this;
        auto func = [&](const std::vector<expr> &in,
                            std::vector<expr::lvalue_proxy_t> &out) -> stmt {
            indexing in_nd = in[0].checked_as<indexing>();
            out[0]->dtype_.lanes_ = 1;
            auto lanes = vec_lanes;
            in_nd->dtype_.lanes_ = lanes;
            //  add the axis to indexing node
            in_nd->idx_.emplace_back(0);
            expr result = get_binary_reduce_by_reduce_op(rd_op_)(in_nd);
            return builder::make_assign_unattached(out[0], result);
        };

        compute_vectorized_op(ctx, get_owner_graph(), inputs, *dst[0], info_,
                vx_info_, mask_compute_func_t(func), mask_compute_func_t(func),
                attrs_, 0, false, dst[0]);
    } else {
        builder::get_current_builder()->emit(
                builder::make_stmts_unattached({}));
    }
}
OP_REGISTER(reduce_sum_op_t, reduce_sum)
OP_REGISTER(reduce_prod_op_t, reduce_prod)
OP_REGISTER(reduce_max_op_t, reduce_max)
OP_REGISTER(reduce_min_op_t, reduce_min)
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
