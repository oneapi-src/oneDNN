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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "reduce.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <util/utils.hpp>

namespace sc {

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

std::vector<int> transform_axis_plain2blocking(
        const graph_tensor_ptr &gt, const std::vector<int> &plain_axis) {
    auto fmt = gt->details_.get_format();
    int bs_ndim = 0;
    if (fmt.format_code_.is_batch_format()) {
        bs_ndim = static_cast<int>(gt->details_.get_blocking_dims().size())
                - fmt.format_code_.ndims();
    }
    // If format is any, just return.
    if (fmt.is_any()) { return plain_axis; }
    std::vector<int> real_axis;
    auto p2bmp = fmt.format_code_.collect_p2b_mapping();
    for (auto &i : plain_axis) {
        if (i < bs_ndim) {
            real_axis.emplace_back(i);
        } else {
            std::vector<int> res;
            res.resize(p2bmp[i - bs_ndim].size());
            std::transform(p2bmp[i - bs_ndim].begin(), p2bmp[i - bs_ndim].end(),
                    res.begin(),
                    [&bs_ndim](const int &v) { return v + bs_ndim; });
            real_axis.insert(real_axis.end(), res.begin(), res.end());
        }
    }
    std::sort(real_axis.begin(), real_axis.end());
    return real_axis;
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
        if (keep_dims_) {
            out = logical_tensor_t(
                    get_reduced_format(
                            ins[0]->details_.get_format(), plain_rd_axis_),
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
    const auto &in_fmt = info_.inputs_[0]->details_.get_format();
    if (keep_dims_) {
        out_formats.push_back({get_reduced_format(in_fmt, plain_rd_axis_)});
    } else {
        auto out_shape_size = info_.inputs_[0]->details_.get_plain_dims().size()
                - plain_rd_axis_.size();
        if (out_shape_size == 0) out_shape_size = 1;
        if (!in_fmt.is_blocking()) {
            out_formats.push_back(
                    {sc_data_format_t::get_plain_by_dims(out_shape_size)});
        } else {
            COMPILE_ASSERT(plain_rd_axis_.size() == 1,
                    "Currently we only support 1 reduce axis when input is "
                    "blocking with keep_dims_=false.")
            sc_data_format_t new_format;
            auto &new_code = new_format.format_code_;
            int new_code_i = 0;
            for (int i = 0; i < in_fmt.format_code_.ndims(); i++) {
                if (in_fmt.format_code_.get(i) < plain_rd_axis_[0]) {
                    new_code.set(new_code_i, in_fmt.format_code_.get(i));
                    new_code_i++;
                } else if (in_fmt.format_code_.get(i) == plain_rd_axis_[0]) {
                    continue;
                } else {
                    new_code.set(new_code_i, in_fmt.format_code_.get(i) - 1);
                    new_code_i++;
                }
            }
            if (!in_fmt.format_code_.collect_blocking_index(plain_rd_axis_[0])
                            .empty()) {
                if (in_fmt.format_code_
                                .collect_blocking_index(plain_rd_axis_[0])
                                .at(0)
                        == 0) {
                    new_format.blocks_[0] = in_fmt.blocks_[1];
                } else {
                    new_format.blocks_[0] = in_fmt.blocks_[0];
                }
            }
            out_formats.push_back({new_format});
        }
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
                std::string("_fuseiter") + fusion_create_idx()));
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
    return transform_axis_plain2blocking(info_.inputs_[0], plain_rd_axis_);
}

sc_dims reduce_op_t::get_bwise_fuse_shrink_dims() const {
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

void reduce_op_t::collect_shrinked_axes_map(
        int bw_size, gt2axes_map &bw_axes_map) {
    auto rd_axis = get_rd_axis();
    std::vector<int> bw_axes;
    int valid_cnt = 0;
    for (int i = 0; i < bw_size; i++) {
        auto iter = std::find(rd_axis.begin(), rd_axis.end(), i);
        if (iter != rd_axis.end()) {
            bw_axes.emplace_back(-1);
        } else {
            bw_axes.emplace_back(valid_cnt++);
        }
    }
    op_traits::batchwise_shrinkable_t::record_shrinked_axes(
            bw_axes_map, get_inputs()[0], bw_size);
    if (keep_dims_) {
        op_traits::batchwise_shrinkable_t::record_shrinked_axes(
                bw_axes_map, get_outputs()[0], bw_size);
    } else {
        op_traits::batchwise_shrinkable_t::record_shrinked_axes(
                bw_axes_map, get_outputs()[0], bw_axes);
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

OP_REGISTER(reduce_op_t, reduce)

} // namespace sc
