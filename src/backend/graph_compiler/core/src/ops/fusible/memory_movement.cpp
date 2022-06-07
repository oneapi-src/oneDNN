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
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "memory_movement.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/graph/outer_loop_generator.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <unordered_map>
#include <util/utils.hpp>

namespace sc {
ir_module_ptr reshape_op_t::get_func(context_ptr ctx) {
    top_level_anchor_generator_t gen;
    attrs_.set(op_attr_key::no_fuse, true);
    auto ret = fusible_op_get_func(this, gen, ctx, true);
    return ret;
}

transpose_op_t::transpose_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : order_(attrs.get<std::vector<int>>("order")) {
    info_.inputs_ = ins;
    info_.outputs_ = outs;
    assert(info_.inputs_.size() == 1);
    assert(order_.size() == info_.inputs_[0]->details_.get_plain_dims().size());
    if (info_.outputs_.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        auto in_dims = info_.inputs_[0]->details_.get_plain_dims();
        sc_dims out_dims(in_dims.size());
        for (size_t i = 0; i < in_dims.size(); ++i) {
            out_dims[i] = in_dims[order_[i]];
        }
        info_.outputs_[0]->details_.set_plain_dims(out_dims);
        info_.outputs_[0]->details_.dtype_ = ins[0]->details_.dtype_;
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

void transpose_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    COMPILE_ASSERT(!info_.inputs_[0]->details_.get_format().is_any(),
            "cannot infer output format with any input format");
    auto in_format = info_.inputs_[0]->details_.get_format();
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

    in_formats.push_back(std::vector<sc_data_format_t> {in_format});
    out_formats.push_back(std::vector<sc_data_format_t> {out_format});
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

void transpose_op_t::prepare_fusion_data(fdata_map &fdmap) {
    throw std::runtime_error("Not implemented");
}

void transpose_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    throw std::runtime_error("Not implemented");
}

void transpose_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    throw std::runtime_error("Not implemented");
}

void compute_block_transpose(const std::vector<const tensor_slice *> &src,
        const tensor_slice &dst, const std::vector<int> &axes, size_t wkld) {
    std::vector<expr> iters(src[0]->nslice_dims());
    std::vector<expr> src_idx(src[0]->nslice_dims());
    std::vector<expr> dst_idx(src[0]->nslice_dims());

    for (unsigned i = 0; i < src[0]->nslice_dims(); i++) {
        iters[i] = builder::make_var(datatypes::index,
                std::string("_fuseiter") + fusion_create_idx());
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
    throw std::runtime_error("Not implemented");
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
        format = info_.outputs_[0]->details_.get_format();
        // changed to get dynamically in need.
        // shapes_ = outs[0]->details_.get_blocking_dims();
    }
    if (cache_input_format.is_any()) {
        cache_input_format
                = sc_data_format_t(sc_data_format_kind_t::get_plain_by_dims(
                        ins[0]->details_.get_plain_dims().size()));
    }
    attrs_["cache_input_format"] = cache_input_format;
    if (format.is_any()) {
        format = sc_data_format_t(sc_data_format_kind_t::get_plain_by_dims(
                info_.outputs_[0]->details_.get_plain_dims().size()));
    }
    attrs_["format"] = format;
}

tensor_view_op_t::tensor_view_op_t(graph_tensor_ptr v, const sc_dims &shapes)
    : tensor_view_op_t({std::move(v)}, {}, {{"shape", shapes}, {}}) {}

bool tensor_view_op_t::try_penetrate(
        sc_data_format_t &new_output_format) const {
    auto input_plain_shapes = info_.inputs_[0]->details_.get_plain_dims();
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
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    sc_data_format_t output_format;
    // temp workaround
    assert(!attrs_.get<sc_data_format_t>("format").is_any());
    if (attrs_.get_or_else<bool>("expand_dim", false)
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
    if (share_gt_with_op<output_op>(get_inputs()[0])) {
        stat_map.append_ops_by_status(this, infer_status_code::FAIL);
        return;
    }
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
    if (share_gt_with_op<output_op>(get_inputs()[0])) {
        stat_map.append_ops_by_status(this, infer_status_code::FAIL);
        return;
    }
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

sc_dims tensor_view_op_t::get_bwise_fuse_shrink_dims() {
    auto old_dims = info_.inputs_[0]->details_.get_blocking_dims();
    auto new_dims = get_shapes();
    sc_dims bw_dims;
    int offset
            = std::min(op_traits::batchwise_shrinkable_t::get_shrinkable_offset(
                               info_.inputs_[0]),
                    op_traits::batchwise_shrinkable_t::get_shrinkable_offset(
                            info_.outputs_[0]));
    int common_size = std::min(old_dims.size(), new_dims.size());
    for (int i = 0; i < std::min(common_size, offset); i++) {
        if (old_dims[i] == new_dims[i])
            bw_dims.emplace_back(new_dims[i]);
        else
            break;
    }
    return bw_dims;
}

sc_op_ptr tensor_view_op_t::bw_shrinked_copy(
        gt2gt_map &bw_lt_map, sc_graph_t &shrinked_graph) {
    auto ins = get_inputs()[0];
    auto cache_input_format = ins->details_.get_format();
    COMPILE_ASSERT(bw_lt_map.haskey(ins),
            "tensor_view_op: new input graph tensor not found in map")
    auto plain_shape = sc_data_format_t::get_padded_plain_shapes(
            bw_lt_map.get(ins)->details_.get_blocking_dims(),
            cache_input_format);
    return op_traits::batchwise_shrinkable_t::bw_shrinked_copy(
            bw_lt_map, shrinked_graph, {{"shape", plain_shape}});
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
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
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

OP_REGISTER(transpose_op_t, transpose)
OP_REGISTER(tensor_view_op_t, tensor_view)
OP_REGISTER(reshape_op_t, reshape)
OP_REGISTER(reorder_op_t, reorder)
} // namespace sc
