/*******************************************************************************
 * Copyright 2022-2024 Intel Corporation
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
#include <vector>

#include "memory_movement.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/anchor_loop_generator.hpp>
#include <compiler/ir/graph/dynamic_dispatch_key.hpp>
#include <compiler/ir/graph/dynamic_utils.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <unordered_map>
#include <util/math_utils.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
ir_module_ptr reshape_op_t::get_func(context_ptr ctx) {
    return fusible_op_get_func(this, ctx);
}

static void check_concat_validity(
        const std::vector<graph_tensor_ptr> &candidates, unsigned axis) {
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
            if (axis == dim && curShape[dim]) { continue; }
            COMPILE_ASSERT(curShape[dim] == firstShape[dim],
                    "Input shapes: "
                            << utils::print_vector(curShape) << " and "
                            << utils::print_vector(firstShape)
                            << " are not matched in concat fusion op!\n");
        }
    }
}

static void compute_block_concat(const context_ptr &ctx,
        const std::vector<const tensor_slice *> &src, const tensor_slice &dst,
        int64_t axis, size_t wkld = 0UL) {
    // outer nested loop vars
    std::vector<expr> outer_iter(axis);
    // inner nested loop vars
    std::vector<std::vector<expr>> inner_iter(dst.nslice_dims() - axis);
    // the indices for multiple inputs. First dim: the input, Second dim:
    // the dimemsions in the tensor
    std::vector<std::vector<expr>> src_idx(src.size());
    // the indices for the output tensor. Cause concat is a assign op, we
    // need number of src indexes.
    std::vector<std::vector<expr>> dst_idx(src.size());
    for (unsigned i = 0; i < dst.nslice_dims(); i++) {
        if (i < static_cast<unsigned>(axis)) { // outer loop
            // make the loop var for the for-loop
            outer_iter[i] = range_from_outer_loop(dst.get_ranges()[i])
                    ? expr(0)
                    : builder::make_var(datatypes::index,
                            std::string("_fuseiter") + fusion_create_idx());
            for (unsigned j = 0; j < src.size(); j++) {
                src_idx[j].emplace_back(outer_iter[i]);
                dst_idx[j].emplace_back(outer_iter[i]);
            }
        } else { // inner loop
            expr cur = 0;
            for (unsigned j = 0; j < src.size(); j++) {
                inner_iter[i - axis].emplace_back(builder::make_var(
                        datatypes::index,
                        std::string("_fuseiter") + fusion_create_idx()));
                src_idx[j].emplace_back(inner_iter[i - axis][j]);
                if (static_cast<int64_t>(i) == axis) {
                    if (j > 0) { cur = cur + src[j - 1]->get_shape()[i]; }
                    dst_idx[j].emplace_back(inner_iter[i - axis][j] + cur);
                } else {
                    dst_idx[j].emplace_back(inner_iter[i - axis][j]);
                }
            }
        }
    }
    expr indexed_dst;
    expr indexed_src;
    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder is set");
    std::vector<stmt> tcur;
    for (unsigned j = 0; j < src.size(); j++) {
        size_t last_axis = dst.nslice_dims() - 1;
        auto slice_len = do_cast_and_fold(src[j]->get_shape().at(last_axis));
        bool is_static = slice_len.isa<constant>();
        if (is_static) { // static shape case
            // for the inner-most axis, use vectorization
            auto dtype = dst.get_base_dtype();
            auto vec_lanes = vectorize_step(ctx, dtype.type_code_);
            auto floor = do_cast_and_fold(slice_len / vec_lanes * vec_lanes);
            auto tail = do_cast_and_fold(slice_len % vec_lanes);
            int floor_int = get_expr_as_int(floor);
            int tail_int = get_expr_as_int(tail);
            COMPILE_ASSERT(
                    (floor_int + tail_int), "Don't support shape len = 0.");
            auto cur = builder::make_stmts_unattached({});
            if (tail_int) { // tail part
                auto tail_part = builder::make_stmts_unattached({});
                auto mask = last_dim_generate_mask(
                        inner_iter[last_axis - axis][j], floor, slice_len,
                        vec_lanes, true);
                indexed_dst = builder::make_indexing(
                        dst.tptr_, dst_idx[j], vec_lanes, mask);
                indexed_src = builder::make_indexing(
                        src[j]->tptr_, src_idx[j], vec_lanes, mask);
                auto assign
                        = make_stmt<assign_node_t>(indexed_dst, indexed_src);
                assign->attr()
                        [op_traits::workload_computable_t::workload_number]
                        = wkld;
                tail_part.static_as<stmts>()->seq_.emplace_back(assign);
                tail_part = make_stmt<for_loop_node_t>(
                        inner_iter[last_axis - axis][j], floor,
                        src[j]->get_shape()[last_axis], vec_lanes,
                        std::move(tail_part), true, for_type::NORMAL);
                cur.static_as<stmts>()->seq_.emplace_back(tail_part);
            }

            if (floor_int) { // divisible part
                auto divisible_part = builder::make_stmts_unattached({});
                indexed_dst = builder::make_indexing(
                        dst.tptr_, dst_idx[j], vec_lanes);
                indexed_src = builder::make_indexing(
                        src[j]->tptr_, src_idx[j], vec_lanes);
                auto assign
                        = make_stmt<assign_node_t>(indexed_dst, indexed_src);
                assign->attr()
                        [op_traits::workload_computable_t::workload_number]
                        = wkld;
                divisible_part.static_as<stmts>()->seq_.emplace_back(assign);
                divisible_part = make_stmt<for_loop_node_t>(
                        inner_iter[last_axis - axis][j], expr(0), floor,
                        vec_lanes, std::move(divisible_part), true,
                        for_type::NORMAL);
                cur.static_as<stmts>()->seq_.emplace_back(divisible_part);
            }

            // for other inner axes
            for (int64_t i = static_cast<int64_t>(dst.nslice_dims()) - 2;
                    i >= axis; i--) {
                auto body = cur.isa<stmts>()
                        ? std::move(cur)
                        : make_stmt<stmts_node_t>(
                                std::vector<stmt> {std::move(cur)});
                cur = make_stmt<for_loop_node_t>(inner_iter[i - axis][j],
                        expr(0), src[j]->get_shape()[i], expr(1),
                        std::move(body), true, for_type::NORMAL);
            }
            tcur.emplace_back(std::move(cur));
        } else { // dynamic case, use step = 1
            indexed_dst = builder::make_indexing(dst.tptr_, dst_idx[j]);
            indexed_src = builder::make_indexing(src[j]->tptr_, src_idx[j]);
            stmt cur = make_stmt<assign_node_t>(indexed_dst, indexed_src);
            cur->attr()[op_traits::workload_computable_t::workload_number]
                    = wkld;
            for (int64_t i = static_cast<int64_t>(dst.nslice_dims()) - 1;
                    i >= axis; i--) {
                auto body = make_stmt<stmts_node_t>(
                        std::vector<stmt> {std::move(cur)});
                cur = make_stmt<for_loop_node_t>(inner_iter[i - axis][j],
                        expr(0), src[j]->get_shape()[i], expr(1),
                        std::move(body), true, for_type::NORMAL);
            }
            tcur.emplace_back(std::move(cur));
        }
    }

    if (axis) {
        stmt cur = make_stmt<stmts_node_t>(std::move(tcur));
        for (int i = axis - 1; i >= 0; i--) {
            // Do not generate those dummy loops
            if (!outer_iter[i].isa<var>()) continue;
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

static bool check_slice_on_non_concat_axis_equal(
        const slice_range_map &known_ranges_map, int64_t axis) {
    std::vector<slice_range_list> slices;
    for (auto &id_sr_pair : known_ranges_map) {
        slices.push_back(id_sr_pair.second);
    }
    COMPILE_ASSERT(slices.size() > 1,
            "Only check if slices of multiple inputs are given");
    for (size_t i = 1; i < slices.size(); ++i) { // number of known slices
        COMPILE_ASSERT(slices[0].size() == slices[i].size(),
                "The multi-slice number should be equal");
        for (size_t j = 0; j < slices[0].size(); ++j) { // number of multi-slice
            // input tensor rank
            COMPILE_ASSERT(slices[0][j].size() == slices[i][j].size(),
                    "The rank of inout tensors should be equal");
            for (size_t k = 0; k < slices[0][j].size(); ++k) {
                // if pair offset and range is not equal on non-concat axis
                if (int64_t(k) != axis
                        && slices[0][j][k].first.isa<constant_c>()
                        && slices[i][j][k].first.isa<constant_c>()
                        && slices[0][j][k].second.isa<constant_c>()
                        && slices[i][j][k].second.isa<constant_c>()) {
                    auto input0_offset = get_expr_as_int(slices[0][j][k].first);
                    auto inputi_offset = get_expr_as_int(slices[i][j][k].first);
                    auto input0_range = get_expr_as_int(slices[0][j][k].second);
                    auto inputi_range = get_expr_as_int(slices[i][j][k].second);
                    if (input0_offset != inputi_offset
                            || input0_range != inputi_range) {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

static sc_dims infer_concat_output_shape(
        const std::vector<graph_tensor_ptr> &inputs, const int &axis) {
    sc_dims ref_output_shape = inputs[0]->details_.get_plain_dims();
    sc_data_type_t ref_dtype = inputs[0]->details_.dtype_;
    sc_dim accumulated_dim = ref_output_shape[axis];
    for (size_t i = 1; i < inputs.size(); i++) {
        const auto &input_shape = inputs[i]->details_.get_plain_dims();
        COMPILE_ASSERT(ref_output_shape.size() == input_shape.size(),
                "The rank of all inputs of concat op shall match.");
        COMPILE_ASSERT(inputs[i]->details_.dtype_ == ref_dtype,
                "The data type of all inputs of concat op shall match.");
        for (size_t d = 0; d < input_shape.size(); ++d) {
            if (d == static_cast<size_t>(axis)) {
                accumulated_dim += input_shape[d];
            } else {
                COMPILE_ASSERT(input_shape[d] == ref_output_shape[d],
                        "The shape of concat inputs on not concated dim shall "
                        "match.");
            }
        }
    }
    ref_output_shape[axis] = accumulated_dim;
    return ref_output_shape;
}

concat_op_t::concat_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    op_name_ = "concat";
    COMPILE_ASSERT(!ins.empty(), "Inputs to concat should be non-empty");
    COMPILE_ASSERT(attrs.has_key("axis"), "Concat axis should be provided.");
    attrs_ = attrs;
    for (auto &in : ins) {
        info_.inputs_.emplace_back(in);
    }
    is_input_valid_ = std::vector<bool>(info_.inputs_.size(), true);
    set_format_and_axis();
    // inferring output plain shape
    sc_dims output_plain_dim
            = infer_concat_output_shape(info_.inputs_, plain_axis_);

    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        info_.outputs_[0]->details_.dtype_ = info_.inputs_[0]->details_.dtype_;
        info_.outputs_[0]->details_.set_plain_dims(output_plain_dim);
        info_.outputs_[0]->details_.set_format(
                info_.inputs_[0]->details_.get_format());
    } else {
        COMPILE_ASSERT(
                outs.size() == 1, "Only one output is supported for concat op");
        COMPILE_ASSERT(
                gc::graph::check_shape_equal(
                        outs[0]->details_.get_plain_dims(), output_plain_dim),
                "Concat op's output shape is not correct.");
        info_.outputs_.emplace_back(outs.front());
    }
}

void concat_op_t::set_format_and_axis() {
    // find the largest input index
    auto max_buffer = std::max_element(info_.inputs_.begin(),
            info_.inputs_.end(),
            [](const graph_tensor_ptr &a, const graph_tensor_ptr &b) {
                return get_dims_product(a->details_.get_plain_dims())
                        < get_dims_product(b->details_.get_plain_dims());
            });
    if ((*max_buffer)->details_.get_format().get_format_category()
            == sc_format_category::non_blocking) {
        ori_format_ = (*max_buffer)->details_.get_format();
    } else {
        // if the input has any/block/vnni format, use plain format when concat
        ori_format_ = sc_data_format_t::get_plain_by_dims(
                (int)(*max_buffer)->details_.get_plain_dims().size());
    }
    // here axis_ is in plain format (because it is copied from llga bridge)
    // we need to transform it to blocking format
    axis_ = attrs_.get<int>("axis");
    // We accept negative axis_, but keep it non-negative internally
    int64_t rank = info_.inputs_[0]->details_.get_plain_dims().size();
    COMPILE_ASSERT(axis_ >= -rank && axis_ <= rank - 1,
            "Concat axis should be in range [" << -rank << ", " << rank - 1
                                               << "], but get: " << axis_);
    if (axis_ < 0) { axis_ += rank; }
    plain_axis_ = axis_;
    std::vector<int> blocking_axes
            = ori_format_.format_code_.collect_p2b_mapping()[axis_];
    COMPILE_ASSERT(
            blocking_axes.size() == 1, "The concat axis should not be blocked");
    axis_ = blocking_axes[0];
}

concat_op_t::concat_op_t(
        const std::vector<graph_tensor_ptr> &candidates, int axis)
    : concat_op_t(candidates, {}, any_map_t({{"axis", axis}})) {}

void concat_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    set_format_and_axis();
    std::vector<std::vector<sc_data_format_t>> in_formats(
            info_.inputs_.size(), {ori_format_});
    std::vector<std::vector<sc_data_format_t>> out_formats(
            info_.outputs_.size(), {ori_format_});
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
    for (size_t i = 0; i < info_.inputs_.size(); ++i) {
        if (info_.inputs_[i]->details_.get_format() == ori_format_) {
            supported_ins[i][0].second
                    = info_.inputs_[i]->details_.get_strides();
        }
    }
}

infer_status_code concat_op_t::infer_slice_ranges(
        const context_ptr &ctx, fslice_map &fsmap) {
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map = search_known_input_slice(this, fsmap);
    if (known_ranges_map.empty()) return infer_status_code::RETRY;
    if (known_ranges_map.size() > 1) {
        // slice of multiple inputs are given
        if (!check_slice_on_non_concat_axis_equal(known_ranges_map, axis_)) {
            return infer_status_code::RETRY;
        }
    }
    auto known_id = known_ranges_map.begin()->first; // input id
    slice_range_list sr = known_ranges_map[known_id];

    size_t slice_size = sr.size(); // multi-slice
    for (size_t i = 0; i < get_inputs().size(); ++i) {
        if (known_ranges_map.find(i) == known_ranges_map.end()) {
            fsmap.get(get_inputs()[i]).clear();
            fsmap.get(get_inputs()[i]).resize(slice_size);
        }
    }
    fsmap.get(get_outputs()[0]).clear();
    fsmap.get(get_outputs()[0]).resize(slice_size);

    std::vector<int> required_axis = {int(axis_)};
    for (size_t n = 0; n < slice_size; n++) { // multi-slice index
        // slice at concat axis should be full
        if (!slice_full_on_axis(
                    info_.inputs_[known_id]->details_.get_blocking_dims(),
                    sr[n], {int(axis_)})) {
            return infer_status_code::RETRY;
        }

        // slice_ranges of inputs and output only differ at concat axis.
        // Since we have already checked the slice_range is full, now we can
        // safely set its offset to 0 and range to shape.
        for (size_t i = 0; i < get_inputs().size(); ++i) {
            slice_range sr_i = sr[n];
            sr_i[axis_].first = 0;
            sr_i[axis_].second = int(dim2unsigned(
                    info_.inputs_[i]->details_.get_blocking_dims()[axis_]));
            fsmap.get(get_inputs()[i]).at(n) = sr_i;
        }
        slice_range sr_o = sr[n];
        sr_o[axis_].first = 0;
        sr_o[axis_].second = int(dim2unsigned(
                info_.outputs_[0]->details_.get_blocking_dims()[axis_]));
        fsmap.get(get_outputs()[0]).at(n) = sr_o;
    }
    return infer_status_code::OK;
}

infer_status_code concat_op_t::pre_infer_slice_ranges(
        const context_ptr &ctx, fslice_map &fsmap) {
    throw std::runtime_error("Not implemented");
}

void concat_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);
    compute_block_concat(ctx, inputs, *dst[0], axis_, wkld);
}

static sc_dims infer_transpose_output_shape(
        const sc_dims &input_shape, const std::vector<int> &order) {
    sc_dims output_shape(input_shape.size());
    for (size_t i = 0; i < input_shape.size(); ++i) {
        output_shape[i] = input_shape[order[i]];
    }
    return output_shape;
}

transpose_op_t::transpose_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : order_(attrs.get<std::vector<int>>("order")) {
    info_.inputs_ = ins;
    info_.outputs_ = outs;
    assert(info_.inputs_.size() == 1);
    COMPILE_ASSERT(
            order_.size() == info_.inputs_[0]->details_.get_plain_dims().size(),
            "Attribute order shall have the same length as input.");
    auto output_shape = infer_transpose_output_shape(
            info_.inputs_[0]->details_.get_plain_dims(), order_);
    auto out_format = attrs.get_or_else("out_format", sc_data_format_t());
    if (info_.outputs_.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        info_.outputs_[0]->details_.set_plain_dims(output_shape);
        info_.outputs_[0]->details_.dtype_ = ins[0]->details_.dtype_;
        info_.outputs_[0]->details_.set_format(out_format);
    } else {
        COMPILE_ASSERT(info_.outputs_.size() == 1,
                "Transpose op shall only have 1 output.");
        COMPILE_ASSERT(gc::graph::check_shape_equal(
                               info_.outputs_[0]->details_.get_plain_dims(),
                               output_shape),
                "Specified transpose op's output shape is incorrect.");
        COMPILE_ASSERT(info_.outputs_[0]->details_.dtype_
                        == info_.inputs_[0]->details_.dtype_,
                "Specified transpose op's output dtype is incorrect.");
    }
    attrs_ = attrs;
    op_name_ = "transpose";
}

transpose_op_t::transpose_op_t(graph_tensor_ptr v, std::vector<int> &order)
    : order_(order) {
    info_.inputs_.emplace_back(std::move(v));
    info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
    op_name_ = "transpose";
}

shape_rl_vec transpose_op_t::get_dynamic_shape_relations() const {
    shape_rl_vec ret;
    auto &in_dims = get_inputs()[0]->details_.get_plain_dims();
    auto &out_dims = get_outputs()[0]->details_.get_plain_dims();
    for (size_t i = 0; i < in_dims.size(); i++) {
        if (is_dynamic_dim(in_dims[order_[i]])) {
            assert(is_dynamic_dim(out_dims[i]));
            ret.emplace_back(in_dims[order_[i]], out_dims[i]);
        }
    }
    return ret;
}

void transpose_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    COMPILE_ASSERT(!info_.inputs_[0]->details_.get_format().is_any(),
            "cannot infer output format with any input format");
    auto in_format = info_.inputs_[0]->details_.get_format();
    auto in_strides = info_.inputs_[0]->details_.get_strides();
    auto in_format_code = in_format.format_code_;
    int batch_dims = info_.inputs_[0]->details_.get_plain_dims().size()
            - in_format_code.norig_dims();
    std::unordered_map<int, int> order_map;
    for (size_t i = 0; i < order_.size(); ++i) {
        COMPILE_ASSERT((order_[i] >= batch_dims)
                        == (static_cast<int>(i) >= batch_dims),
                "Permutation on batch dims is not supported.")
        if (order_[i] >= batch_dims && static_cast<int>(i) >= batch_dims) {
            order_map[order_[i] - batch_dims] = i - batch_dims;
        }
    }

    std::vector<int> storage_args;
    for (int i = 0; i < in_format_code.ndims(); ++i) {
        int axis = in_format_code.get(i);
        storage_args.push_back(order_map[axis]);
    }
    auto out_format = sc_data_format_t(storage_args, in_format.blocks_);

    supported_ins.resize(1);
    supported_outs.resize(1);
    supported_ins[0].emplace_back(std::make_pair(in_format, in_strides));
    supported_outs[0].emplace_back(std::make_pair(out_format, in_strides));
}

infer_status_code transpose_op_t::infer_slice_ranges(
        const context_ptr &ctx, fslice_map &fsmap) {
    throw std::runtime_error("Not implemented");
}

infer_status_code transpose_op_t::pre_infer_slice_ranges(
        const context_ptr &ctx, fslice_map &fsmap) {
    throw std::runtime_error("Not implemented");
}

void compute_block_transpose(const std::vector<const tensor_slice *> &src,
        const tensor_slice &dst, const std::vector<int> &axis, size_t wkld) {
    std::vector<expr> iters(src[0]->nslice_dims());
    std::vector<expr> src_idx(src[0]->nslice_dims());
    std::vector<expr> dst_idx(src[0]->nslice_dims());

    for (unsigned i = 0; i < src[0]->nslice_dims(); i++) {
        iters[i] = builder::make_var(datatypes::index,
                std::string("_fuseiter") + fusion_create_idx());
        src_idx[i] = iters[i];
    }
    dst_idx = src_idx;
    std::swap(dst_idx[axis[0]], dst_idx[axis[1]]);
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
    throw std::runtime_error("Not implemented");
}

size_t transpose_op_t::compute_workload(
        const std::vector<shape_dtype_pair> &ins,
        const std::vector<shape_dtype_pair> &outs) {
    return fusible_op_t::compute_workload(ins, outs)
            * workload_penalty_coefficient;
}

sc_dims tensor_view_op_t::get_shapes() const {
    return info_.outputs_[0]->details_.get_blocking_dims();
}

std::vector<expr> tensor_view_op_t::get_shapes_expr() {
    return info_.outputs_[0]->details_.get_blocking_dims_expr(
            get_owner_graph());
}

tensor_view_op_t::tensor_view_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    op_name_ = "tensor_view";
    COMPILE_ASSERT(ins.size() == 1, "Reshape takes 1 input");
    info_.inputs_ = ins;
    auto cache_input_format = ins[0]->details_.get_format();
    attrs_ = attrs;
    auto shapes = attrs_.get<sc_dims>("shape");
    auto format = attrs_.get_or_else("format", sc_data_format_t());
    int total_shape1 = 1, total_shape2 = 1, total_shape3 = 1;
    for (auto &dim : sc_data_format_t::get_padded_plain_shapes(
                 ins[0]->details_.get_blocking_dims(), cache_input_format)) {
        total_shape1 *= dim;
    }
    for (auto &dim : shapes) {
        total_shape2 *= dim;
    }
    if (!outs.empty()) {
        for (auto &dim : sc_data_format_t::get_padded_plain_shapes(
                     outs[0]->details_.get_blocking_dims(),
                     outs[0]->details_.get_format())) {
            total_shape3 *= dim;
        }
    }
    COMPILE_ASSERT(is_dynamic() || total_shape1 == total_shape2
                    || (!outs.empty() && total_shape1 == total_shape3),
            "Wrong total size of input shapes, can not do reshape plain dims "
            "from " << utils::print_vector(ins[0]->details_.get_plain_dims())
                    << " to " << utils::print_vector(shapes));
    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        info_.outputs_[0]->details_.dtype_ = ins[0]->details_.dtype_;
        info_.outputs_[0]->details_.set_plain_dims(shapes);
        info_.outputs_[0]->details_.set_format(format);
        shapes_ = shapes;
    } else {
        COMPILE_ASSERT(outs.size() == 1, "Wrong op output size.\n");
        info_.outputs_ = outs;
        format = info_.outputs_[0]->details_.get_format();
        // changed to get dynamically in need.
        // shapes_ = outs[0]->details_.get_blocking_dims();
    }
    if (cache_input_format.is_any()) {
        cache_input_format
                = sc_data_format_t(sc_data_format_kind_t::get_plain_by_dims(
                        ins[0]->details_.get_plain_dims().size()));
    }
    if (!attrs_.has_key("cache_input_format")) {
        attrs_["cache_input_format"] = cache_input_format;
    }
    if (format.is_any()) {
        format = sc_data_format_t(sc_data_format_kind_t::get_plain_by_dims(
                info_.outputs_[0]->details_.get_plain_dims().size()));
        attrs_["format"] = format;
    } else if (!attrs_.has_key("format")) {
        attrs_["format"] = format;
    }
    if (is_dynamic()
            && count_dynamic_dims(get_inputs()[0]->details_.get_plain_dims())
                    != count_dynamic_dims(
                            get_outputs()[0]->details_.get_plain_dims())) {
        attrs_.set(op_attr_key::no_fuse, true);
    }
}

bool tensor_view_op_t::is_only_expand_or_penetrate() const {
    const auto &in_tensor = this->get_inputs()[0]->details_;
    auto in_shape = in_tensor.get_plain_dims();
    auto in_real_shape = in_tensor.get_blocking_dims();
    auto out_tensor = this->get_outputs()[0]->details_;
    auto out_shape = out_tensor.get_plain_dims();
    auto out_real_shape = out_tensor.get_blocking_dims();
    auto erase_element_one = [&](sc_dims &dim) {
        dim.erase(std::remove_if(dim.begin(), dim.end(),
                          [&](sc_dim v) { return v == 1; }),
                dim.end());
    };
    erase_element_one(in_shape);
    erase_element_one(in_real_shape);
    erase_element_one(out_shape);
    erase_element_one(out_real_shape);
    if (in_real_shape == out_real_shape && in_shape == out_shape) {
        return true;
    }
    return false;
}

tensor_view_op_t::tensor_view_op_t(graph_tensor_ptr v, const sc_dims &shapes)
    : tensor_view_op_t({std::move(v)}, {}, {{"shape", shapes}, {}}) {}

bool tensor_view_op_t::try_penetrate(
        sc_data_format_t &new_output_format) const {
    if (attrs_.get_or_else("forbid_penetrate", false)) { return false; }
    auto input_plain_shapes = info_.inputs_[0]->details_.get_plain_dims();
    auto input_blocking_shapes = info_.inputs_[0]->details_.get_blocking_dims();
    auto input_format = info_.inputs_[0]->details_.get_format();
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
        COMPILE_ASSERT(long_idx < long_plain_shapes.size(),
                "long_idx shall be within the valid range.");
        int64_t acc_shape = long_plain_shapes[long_idx];
        long_to_short[long_idx] = short_idx;
        long_idx++;
        while (long_idx < long_size
                && (long_size - long_idx) >= (short_size - short_idx)
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
            // cannot penetrate, if the number of blocking dims will exceed 4
            // in the penetrated output format
            if (input_size - output_size + input_format.get_blocks_size() > 4)
                return false;
            // cannot penetrate, if long_to_short involves any permutation
            // e.g. {a, b, c, d} --> {a*b, c, d}
            // ABCD --> AaBC is OK
            // BACD --> AaBC is incorrect, since B and A are permuted
            auto &input_code = input_format.format_code_;
            auto input_p2b = input_code.collect_p2b_mapping();
            for (size_t out_idx = 0; out_idx < short_size; ++out_idx) {
                std::vector<int> concat_axis;
                for (size_t inp_idx = 0; inp_idx < long_size; ++inp_idx) {
                    if (long_to_short[inp_idx] == out_idx) {
                        concat_axis.insert(concat_axis.end(),
                                input_p2b[inp_idx].begin(),
                                input_p2b[inp_idx].end());
                    }
                }
                if (!std::is_sorted(concat_axis.begin(), concat_axis.end()))
                    return false;
            }
            // start penetrated format inferring
            sc_data_format_t new_format;
            auto &new_code = new_format.format_code_;
            int out_count[sc_data_format_kind_t::MAX_DIMS] = {0};
            size_t blk_idx = 0;
            auto remain_blocks = output_plain_shapes;
            for (int i = 0; i < input_code.ndims(); i++) {
                auto new_idx = long_to_short[input_code.get(i)];
                new_code.set(i, new_idx);
                out_count[new_code.get(i)]++;
                if (out_count[new_code.get(i)] > 1
                        && blk_idx < input_size - output_size) {
                    new_format.blocks_[blk_idx++] = remain_blocks[new_idx];
                }
                remain_blocks[new_idx] /= input_plain_shapes[input_code.get(i)];
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
            if (!is_dynamic()) {
                if (math_utils::get_dims_product(input_blocking_shapes)
                        != math_utils::get_dims_product(
                                sc_data_format_t::get_blocking_shapes(
                                        output_plain_shapes,
                                        new_output_format))) {
                    return false;
                }
            }
            return true;
        } else {
            sc_data_format_t new_format;
            auto &new_code = new_format.format_code_;
            // reconstruct long_to_short to short_to_long
            std::unordered_map<size_t, std::vector<size_t>> short_to_long;
            for (const auto &pair : long_to_short) {
                short_to_long[pair.second].push_back(pair.first);
            }
            // sort short_to_long mapping
            for (size_t i = 0; i < input_size; i++) {
                std::sort(short_to_long[i].begin(), short_to_long[i].end());
            }
            // setting output format code according to input format code
            // and input-output-dimension-mapping
            int format_code_offset = 0;
            int axis_count[sc_data_format_kind_t::MAX_DIMS] = {0};
            for (int i = 0;
                    i < static_cast<int>(input_format.format_code_.ndims());
                    i++) {
                int input_dim = input_format.format_code_.get(i);
                axis_count[input_dim]++;
                std::vector<size_t> output_dims = short_to_long[input_dim];
                if (axis_count[input_dim] > 1) {
                    // blocking dimension
                    new_code.set(i + format_code_offset,
                            static_cast<int>(output_dims.back()));
                } else {
                    // plain dimension
                    for (int idx = 0;
                            idx < static_cast<int>(output_dims.size()); ++idx) {
                        new_code.set(i + format_code_offset + idx,
                                static_cast<int>(output_dims[idx]));
                    }
                    format_code_offset += short_to_long[input_dim].size() - 1;
                }
            }
            new_code.set(sc_data_format_kind_t::MAX_DIMS,
                    input_format.format_code_.get(
                            sc_data_format_kind_t::MAX_DIMS));
            new_format.blocks_ = input_format.blocks_;
            new_output_format = new_format;
            if (!is_dynamic()) {
                if (math_utils::get_dims_product(input_blocking_shapes)
                        != math_utils::get_dims_product(
                                sc_data_format_t::get_blocking_shapes(
                                        output_plain_shapes,
                                        new_output_format))) {
                    return false;
                }
            }
            return true;
        }
    }
    new_output_format = info_.outputs_[0]->details_.get_format();
    return false;
}

shape_rl_vec tensor_view_op_t::get_dynamic_shape_relations() const {
    auto rl_axis_pair = attrs_.get_or_else(
            "temp.rl_axis_pair", std::vector<std::pair<int, int>>());
    auto in_dims = get_inputs()[0]->details_.get_plain_dims();
    auto out_dims = get_outputs()[0]->details_.get_plain_dims();
    shape_rl_vec ret;
    for (auto &it : rl_axis_pair) {
        if (is_dynamic_dim(in_dims[it.first])
                || is_dynamic_dim(out_dims[it.second])) {
            ret.emplace_back(in_dims[it.first], out_dims[it.second]);
        }
    }
    return ret;
}

void tensor_view_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    sc_data_format_t output_format;
    // temp workaround
    assert(!attrs_.get<sc_data_format_t>("format").is_any());
    bool query_by_dispatch_key = false;
    if (is_dynamic()) {
        auto key_set = get_dispatch_key_set();
        key_set->for_each_key_process([&](const op_dispatch_key_base_t *key) {
            auto dkey = static_cast<const op_dispatch_key_t *>(key);
            if (!query_by_dispatch_key
                    && dkey->in_out_formats_[0]
                            == info_.inputs_[0]->details_.get_format()) {
                query_by_dispatch_key = true;
                output_format = dkey->in_out_formats_[1];
            }
        });
    }
    if (query_by_dispatch_key) {
        out_formats.push_back({output_format});
        in_formats.push_back({info_.inputs_[0]->details_.get_format()});
    } else if (attrs_.has_key("expand_dim")
            && info_.inputs_[0]->details_.get_format()
                    == attrs_.get<sc_data_format_t>("cache_input_format")) {
        out_formats.push_back({attrs_.get<sc_data_format_t>("format")});
        in_formats.push_back({info_.inputs_[0]->details_.get_format()});
    } else {
        bool can_penetrate = try_penetrate(output_format);
        if (can_penetrate) {
            out_formats.push_back({output_format});
            in_formats.push_back({info_.inputs_[0]->details_.get_format()});
        } else {
            out_formats.push_back({attrs_.get<sc_data_format_t>("format")});
            in_formats.push_back(
                    {attrs_.get<sc_data_format_t>("cache_input_format")});
        }
    }
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

slice_range_list infer_tensor_view_slice(sc_graph_t &graph,
        const slice_range_list &known_ranges_list,
        const std::vector<expr> &src_tv_dims,
        const std::vector<expr> &dst_tv_dims) {
    slice_range_list ret;
    for (auto known_ranges : known_ranges_list) {
        slice_range consistent_tv_slice;
        auto src_dims = src_tv_dims, dst_dims = dst_tv_dims;
        while (!dst_dims.empty() && !src_dims.empty()) {
            if (!slice_expr_equals(dst_dims.back(), src_dims.back())) break;
            consistent_tv_slice.insert(
                    consistent_tv_slice.begin(), known_ranges.back());
            known_ranges.pop_back();
            src_dims.pop_back();
            dst_dims.pop_back();
        }

        if (consistent_tv_slice.size() == dst_tv_dims.size()) {
            ret.emplace_back(consistent_tv_slice);
            continue;
        }
        // search continuous slice
        bool continuous_slice_stop = false;
        // flatten index
        expr flatten_idx = 0;
        // total length of static dim
        sc_dim total_len = 1;
        // accumulater src dims
        expr acc_src_dim_expr = 1;
        const int dyn_len = -2;
        // check whether complex case exist
        bool complex_case = false;
        for (int i = src_dims.size() - 1; i >= 0; i--) {
            auto slice_expr = do_cast_and_fold(known_ranges[i].second);
            auto src_expr = src_dims[i];
            // continuous slice check
            if (continuous_slice_stop) {
                // check whether slice is full on last several dims
                if (!slice_expr.isa<constant_c>()
                        || get_expr_as_int(slice_expr) != 1)
                    // if tensor_view deals with inconsequence slice, it will
                    // return empty slice range list to tell fusion manager not
                    // to fuse it
                    return slice_range_list {};
            } else {
                if (!(known_ranges[i].first.isa<constant_c>()
                            && get_expr_as_int(known_ranges[i].first) == 0
                            && slice_expr_equals(slice_expr, src_expr))) {
                    // if the last dim is already non-full
                    if (i == static_cast<int>(src_dims.size()) - 1) {
                        // last dim of dst
                        auto dst_expr = dst_dims.back();
                        // double-check legality
                        if (slice_expr.isa<constant>()
                                && dst_expr.isa<constant>()) {
                            auto slice_int = get_expr_as_int(slice_expr);
                            auto dst_int = get_expr_as_int(dst_expr);
                            // skip too complex cases to analyze tensorview
                            // slice range mapping relationship
                            if ((slice_int > dst_int
                                        && slice_int % dst_int != 0)
                                    || (dst_int > slice_int
                                            && dst_int % slice_int != 0)) {
                                return slice_range_list {};
                            }
                        }
                    }
                    if (!slice_expr.isa<constant_c>()
                            || get_expr_as_int(slice_expr) != 1) {
                        complex_case = true;
                    }
                    continuous_slice_stop = true;
                }
            }
            if (slice_expr.isa<constant_c>()) {
                total_len *= get_expr_as_int(slice_expr);
            } else {
                total_len *= dyn_len;
            }
            flatten_idx
                    = flatten_idx + known_ranges[i].first * acc_src_dim_expr;
            acc_src_dim_expr = acc_src_dim_expr * src_expr;
        }
        // deflatten to new shape
        slice_range reshape_ranges;
        sc_dims acc_dst_dim;
        std::vector<expr> acc_dst_dim_expr;
        sc_dim tmp_acc = 1;
        expr tmp_acc_expr = 1;
        for (int64_t i = static_cast<int64_t>(dst_dims.size()) - 1; i >= 0;
                i--) {
            tmp_acc *= !dst_dims[i].isa<constant>()
                    ? dyn_len
                    : get_expr_as_int(dst_dims[i]);
            tmp_acc_expr = tmp_acc_expr * dst_dims[i];
            acc_dst_dim.emplace_back(tmp_acc);
            acc_dst_dim_expr.emplace_back(tmp_acc_expr);
        }
        std::reverse(acc_dst_dim.begin(), acc_dst_dim.end());
        std::reverse(acc_dst_dim_expr.begin(), acc_dst_dim_expr.end());
        std::vector<expr> dst_idx;
        for (unsigned i = 0; i < dst_dims.size() - 1; i++) {
            expr cur_idx = flatten_idx / acc_dst_dim_expr[i + 1];
            dst_idx.emplace_back(cur_idx);
            flatten_idx = flatten_idx % acc_dst_dim_expr[i + 1];
        }
        // mapping input slice ranges to output
        for (int64_t i = static_cast<int64_t>(dst_dims.size()) - 1; i >= 0;
                i--) {
            if (abs(total_len) > abs(acc_dst_dim[i])) {
                reshape_ranges.emplace_back(
                        std::make_pair(expr(0), dst_dims[i]));
            } else {
                if (i == static_cast<int64_t>(dst_dims.size()) - 1) {
                    reshape_ranges.emplace_back(std::make_pair(
                            flatten_idx, expr(dim2unsigned(total_len))));
                } else {
                    if (!complex_case && abs(total_len) == abs(acc_dst_dim[i])
                            && acc_dst_dim[i] != acc_dst_dim[i + 1]) {
                        // simplify offset of ranges for easy case in avoid of
                        // potential fuse break for following post ops
                        reshape_ranges.emplace_back(
                                std::make_pair(expr(0), dst_dims[i]));
                    } else {
                        reshape_ranges.emplace_back(std::make_pair(dst_idx[i],
                                expr(std::max(UINT64_C(1),
                                        dim2unsigned(total_len
                                                / acc_dst_dim[i + 1])))));
                    }
                }
            }
        }
        std::reverse(reshape_ranges.begin(), reshape_ranges.end());
        for (auto &r : reshape_ranges) {
            r.first = do_cast_and_fold(r.first);
        }
        reshape_ranges.insert(reshape_ranges.end(), consistent_tv_slice.begin(),
                consistent_tv_slice.end());
        ret.emplace_back(reshape_ranges);
    }
    return ret;
}

infer_status_code tensor_view_op_t::infer_slice_ranges(
        const context_ptr &ctx, fslice_map &fsmap) {
    if (share_gt_with_op<output_op>(get_inputs()[0])) {
        return infer_status_code::FAIL;
    }
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map = search_known_input_slice(this, fsmap);
    if (known_ranges_map.empty()) return infer_status_code::RETRY;
    slice_range_list known_ranges_list = known_ranges_map[0];

    if (fsmap.get(get_outputs()[0]).empty()) {
        auto &graph = get_owner_graph();
        // src
        auto src_dims
                = info_.inputs_[0]->details_.get_blocking_dims_expr(graph);
        // dst
        auto dst_dims
                = info_.outputs_[0]->details_.get_blocking_dims_expr(graph);

        auto tv_slice = infer_tensor_view_slice(
                graph, known_ranges_list, src_dims, dst_dims);

        if (tv_slice.empty()) { return infer_status_code::RETRY; }
        fsmap.get(get_outputs()[0]) = tv_slice;
    }
    return infer_status_code::OK;
}

infer_status_code tensor_view_op_t::pre_infer_slice_ranges(
        const context_ptr &ctx, fslice_map &fsmap) {
    if (share_gt_with_op<output_op>(get_inputs()[0])) {
        return infer_status_code::FAIL;
    }
    if (fsmap.get(get_inputs()[0]).empty()) {
        slice_range_list known_ranges_list = fsmap.get(get_outputs()[0]);
        auto &graph = get_owner_graph();
        // src
        auto src_dims
                = info_.inputs_[0]->details_.get_blocking_dims_expr(graph);
        // dst
        auto dst_dims
                = info_.outputs_[0]->details_.get_blocking_dims_expr(graph);
        // NOTE: pre_infer_slice_ranges use shapes as src_dims
        auto tv_slice = infer_tensor_view_slice(
                graph, known_ranges_list, dst_dims, src_dims);
        if (tv_slice.empty()) { return infer_status_code::RETRY; }
        fsmap.get(get_inputs()[0]) = tv_slice;
    }
    return infer_status_code::OK;
}

// transpose_axis_map stores the transpose relation of src_axis --> dst_axis
binding_axis infer_tensor_view_binding_axis(const binding_axis &src_axis,
        const sc_dims &src_dims, const sc_dims &dst_dims,
        const std::vector<int> &expand_dims = {},
        const std::vector<int> &transpose_axis_map = {}) {
    binding_axis dst_axis, tv_axis_map;

    if (!transpose_axis_map.empty()) {
        binding_axis real_src_axis;
        COMPILE_ASSERT(src_dims.size() == dst_dims.size()
                        && src_dims.size() == transpose_axis_map.size(),
                "src dims, dst dims, and transpose_axis_map shall have the "
                "same length.")
        for (auto &bd_ax : src_axis) {
            std::vector<int> ret;
            for (auto &ax : bd_ax) {
                COMPILE_ASSERT(ax < static_cast<int>(transpose_axis_map.size()),
                        "ax should be less then transpose_axis_map size")
                ret.emplace_back(transpose_axis_map[ax]);
            }
            real_src_axis.emplace_back(ret);
        }
        return real_src_axis;
    }

    sc_dims acc_src_dims(src_dims.size()), acc_dst_dims(dst_dims.size());
    sc_dim tmp_acc = 1;
    std::transform(src_dims.begin(), src_dims.end(), acc_src_dims.begin(),
            [&tmp_acc](const sc_dim &d) {
                tmp_acc *= d;
                return tmp_acc;
            });
    tmp_acc = 1;
    std::transform(dst_dims.begin(), dst_dims.end(), acc_dst_dims.begin(),
            [&tmp_acc](const sc_dim &d) {
                tmp_acc *= d;
                return tmp_acc;
            });
    // compare src and dst
    size_t j = 0;
    assert(!acc_dst_dims.empty());
    for (size_t i = 0; i < acc_src_dims.size(); i++) {
        std::vector<int> axis;
        while (j < acc_dst_dims.size()) {
            axis.emplace_back(j);
            if (std::abs(acc_src_dims[i]) <= std::abs(acc_dst_dims[j])) {
                if (std::abs(acc_src_dims[i]) == std::abs(acc_dst_dims[j])) {
                    j++;
                }
                break;
            }
            j++;
        }
        tv_axis_map.emplace_back(axis);
    }

    for (auto &bd_ax : src_axis) {
        std::vector<int> ret;
        for (auto &ax : bd_ax) {
            if (expand_dims.end()
                    != std::find(expand_dims.begin(), expand_dims.end(), ax)) {
                continue;
            } else {
                ret.insert(ret.end(), tv_axis_map[ax].begin(),
                        tv_axis_map[ax].end());
            }
        }
        // check if empty to make g++12 happy
        if (!ret.empty()) {
            // remove duplicated axis.
            std::sort(ret.begin(), ret.end());
            ret.erase(std::unique(ret.begin(), ret.end()), ret.end());
        }
        dst_axis.emplace_back(ret);
    }
    return dst_axis;
}

void tensor_view_op_t::infer_binding_axis(binding_axis_map &bdax_map) {
    auto known_axis_map = search_known_input_axis(this, bdax_map);
    if (!bdax_map.get(get_outputs()[0]).empty()) return;
    // src
    auto src_plain_dims = info_.inputs_[0]->details_.get_plain_dims();
    // dst
    auto dst_plain_dims = info_.outputs_[0]->details_.get_plain_dims();
    auto ths = this;
    auto order = attrs_.get_or_else("order", std::vector<int> {});
    std::vector<int> axis_mapping(order.size(), 0);
    for (size_t i = 0; i < order.size(); ++i) {
        axis_mapping[order[i]] = i;
    }
    auto plain_bd_axis = infer_tensor_view_binding_axis(known_axis_map[0],
            src_plain_dims, dst_plain_dims, std::vector<int> {}, axis_mapping);
    bdax_map.get(get_outputs()[0]) = plain_bd_axis;
    set_unknown_binding_axis(this, known_axis_map, bdax_map);
}

void tensor_view_op_t::pre_infer_binding_axis(binding_axis_map &bdax_map) {
    auto &outaxis = bdax_map.get(get_outputs()[0]);
    COMPILE_ASSERT(!outaxis.empty(),
            "Unknown output axis found, could not pre infer binding axis")
    auto &input = get_inputs()[0];
    auto &inpaxis = bdax_map.get(input);

    if (inpaxis.empty()) {
        // src
        auto src_plain_dims = info_.inputs_[0]->details_.get_plain_dims();
        // dst
        auto dst_plain_dims = info_.outputs_[0]->details_.get_plain_dims();
        auto ths = this;
        auto plain_bd_axis = infer_tensor_view_binding_axis(outaxis,
                dst_plain_dims, src_plain_dims,
                attrs_.get_or_else("expand_dim", std::vector<int> {}),
                attrs_.get_or_else("order", std::vector<int> {}));
        inpaxis = plain_bd_axis;
        if (auto bd_op
                = input->producer_owner_
                          ->dyn_cast<op_traits::mixed_partition_acceptable>()) {
            bd_op->pre_infer_binding_axis(bdax_map);
        }
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
infer_status_code reshape_op_t::pre_infer_slice_ranges(
        const context_ptr &ctx, fslice_map &fsmap) {
    throw std::runtime_error("Not implemented");
}

infer_status_code reshape_op_t::infer_slice_ranges(
        const context_ptr &ctx, fslice_map &fsmap) {
    slice_range_map known_ranges_map = search_known_input_slice(this, fsmap);
    if (known_ranges_map.empty() || known_ranges_map[0].size() != 1)
        return infer_status_code::RETRY;
    auto blocking_dims = info_.inputs_[0]->details_.get_blocking_dims();
    std::vector<int> axis(blocking_dims.size());
    std::iota(axis.begin(), axis.end(), 0);
    if (!slice_full_on_axis(blocking_dims, known_ranges_map[0][0], axis)) {
        return infer_status_code::RETRY;
    }
    // fake infer slice
    std::vector<std::pair<expr, expr>> ranges;
    auto &shapes = info_.outputs_[0]->details_.get_plain_dims();
    ranges.reserve(shapes.size());
    for (size_t i = 0; i < shapes.size(); i++) {
        ranges.emplace_back(expr(0), expr(dim2unsigned(shapes[i])));
    }
    fsmap.get(get_outputs()[0]).push_back(ranges);
    return infer_status_code::OK;
}

void reshape_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    in_formats.push_back({info_.inputs_[0]->details_.get_format()});
    out_formats.push_back({sc_data_format_kind_t::get_plain_by_dims(
            info_.outputs_[0]->details_.get_plain_dims().size())});
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
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
            datatypes::index, std::string("_fuseiter") + fusion_create_idx());
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
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    out_formats.reserve(info_.outputs_.size());
    for (size_t i = 0; i < out_formats.size(); ++i) {
        out_formats[i].push_back({info_.inputs_[0]->details_.get_format()});
    }
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

infer_status_code split_op_t::infer_slice_ranges(
        const context_ptr &ctx, fslice_map &fsmap) {
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map = search_known_input_slice(this, fsmap);
    if (known_ranges_map.empty()) return infer_status_code::RETRY;
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
    return infer_status_code::OK;
}

infer_status_code split_op_t::pre_infer_slice_ranges(
        const context_ptr &ctx, fslice_map &fsmap) {
    throw std::runtime_error("Not implemented");
}

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
                    std::string("_fuseiter") + fusion_create_idx());
            for (unsigned j = 0; j < dst.size(); j++) {
                src_idx[j].emplace_back(outer_iter[i]);
                dst_idx[j].emplace_back(outer_iter[i]);
            }
        } else { // inner loop
            expr cur;
            for (unsigned j = 0; j < dst.size(); j++) {
                inner_iter[i - dim].emplace_back(builder::make_var(
                        datatypes::index,
                        std::string("_fuseiter") + fusion_create_idx()));
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

OP_REGISTER(concat_op_t, concat)
OP_REGISTER(transpose_op_t, transpose)
OP_REGISTER(tensor_view_op_t, tensor_view)
OP_REGISTER(reshape_op_t, reshape)
OP_REGISTER(reorder_op_t, reorder)
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
