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

#include <string>
#include <utility>
#include "padding.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>

namespace sc {

padding_op_t::padding_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    COMPILE_ASSERT(ins.size() == 1, "padding expects 1 input");
    const int ndims = ins[0]->details_.get_plain_dims().size();

    COMPILE_ASSERT(utils::is_one_of(static_cast<int>(ndims), 4, 5),
            "wrong input dims, expected to be 4D or 5D input, but got "
                    << ins.size() << "D.");

    info_.inputs_ = ins;
    attrs_ = attrs;

    COMPILE_ASSERT(attrs_.has_key("pads_begin") && attrs_.has_key("pads_end"),
            "padding op shall have pads_begin & pads_end attributes");

    auto &pads_begin = attrs_.get<sc_dims>("pads_begin");
    auto &pads_end = attrs_.get<sc_dims>("pads_end");

    COMPILE_ASSERT(pads_begin == pads_end,
            "Current padding op only supports symmetric padding.");

    if (pads_begin.size() == 1) {
        pads_begin = sc_dims(ndims - 2, pads_begin[0]);
        pads_end = sc_dims(ndims - 2, pads_end[0]);
    }

    COMPILE_ASSERT((ndims - 2) == static_cast<int>(pads_begin.size()),
            "wrong padding dims, " << ndims - 2 << "D input, but got"
                                   << pads_begin.size() << "D paddings.");

    sc_dims expected_out_shape = infer_out_dims(
            info_.inputs_[0]->details_.get_plain_dims(), pads_begin, pads_end);

    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this,
                info_.inputs_[0]->details_.get_format(), expected_out_shape,
                info_.inputs_[0]->details_.dtype_));
    } else {
        COMPILE_ASSERT(outs.size() == 1, "padding expects 1 output");
        COMPILE_ASSERT(outs[0]->details_.get_plain_dims() == expected_out_shape,
                "Bad output shape for padding");
        info_.outputs_ = outs;
    }
    op_name_ = "padding";
}

padding_op_t::padding_op_t(
        graph_tensor_ptr v, sc_dims &pads_begin, sc_dims &pads_end)
    : padding_op_t({std::move(v)}, {},
            any_map_t({{"pads_begin", pads_begin}, {"pads_end", pads_end}})) {}

sc_dims padding_op_t::infer_out_dims(const sc_dims &input_dims,
        const sc_dims &pads_begin, const sc_dims &pads_end) {
    int ndims = input_dims.size();
    auto out_dims = input_dims;
    for (int i = 2; i < ndims; i++) {
        out_dims[i] += pads_begin[i - 2] + pads_end[i - 2];
    }
    return out_dims;
}

void padding_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    out_formats.push_back({info_.inputs_[0]->details_.get_format()});
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

void padding_op_t::prepare_fusion_data(fdata_map &fdmap) {}

void padding_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {}

void padding_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map
            = search_known_slice_ranges(this, fsmap, stat_map);

    size_t slice_size = known_ranges_map[0].size();
    slice_range_list ranges_list(slice_size);
    // update the input slice range offset with additional padding area, while
    // keep the size unchanged
    const size_t ndims = info_.inputs_[0]->details_.get_plain_dims().size();
    const auto &pads_begin = attrs_.get<sc_dims>("pads_begin");

    // if format is channel_last, the spatial_dims_offset should set to 1
    size_t spatial_dims_offset
            = info_.outputs_[0]->details_.get_format().is_channel_last() ? 1
                                                                         : 2;
    for (size_t i = 0; i < slice_size; i++) {
        ranges_list[i] = known_ranges_map[0][i];
        for (size_t j = 0; j < pads_begin.size(); ++j) {
            auto &offset = ranges_list[i][j + spatial_dims_offset].first;
            offset = offset + static_cast<int>(pads_begin[j]);
        }
    }
    fsmap.get(get_outputs()[0]) = std::move(ranges_list);
}

void padding_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &src) {
    size_t wkld = compute_fusible_workload(ctx, dst, src);
    const size_t ndims = dst[0]->nslice_dims();
    auto dst_shape = get_expr_to_dims(dst[0]->get_shape());

    std::vector<expr> iter_vars;
    std::vector<expr> src_idx;
    std::vector<expr> dst_idx;
    for (size_t i = 0; i < ndims; ++i) {
        iter_vars.emplace_back(builder::make_var(datatypes::index,
                std::string("_fuseiter") + fusion_create_idx()));
        src_idx.emplace_back(iter_vars.back());
        dst_idx.emplace_back(iter_vars.back());
    }
    auto bld = builder::get_current_builder();

    int step = static_cast<int>(
            vectorize_step(ctx, info_.inputs_[0]->details_.dtype_.type_code_));
    auto can_vectorize
            = get_expr_as_int(src[0]->get_shape()[ndims - 1]) % step == 0;

    expr indexed_src = builder::make_indexing(
            src[0]->tptr_, src_idx, can_vectorize ? step : 1);
    expr indexed_dst = builder::make_indexing(
            dst[0]->tptr_, dst_idx, can_vectorize ? step : 1);

    stmt cur = make_stmt<assign_node_t>(indexed_dst, indexed_src);
    cur->attr()[op_traits::workload_computable_t::workload_number] = wkld;

    for (int64_t i = static_cast<int64_t>(ndims) - 1; i >= 0; --i) {
        auto body = make_stmt<stmts_node_t>(std::vector<stmt> {std::move(cur)});
        cur = make_stmt<for_loop_node_t>(std::move(iter_vars[i]), expr(0),
                src[0]->get_shape()[i],
                i == static_cast<int64_t>(ndims) - 1 && can_vectorize
                        ? expr(step)
                        : expr(1),
                std::move(body), true, for_type::NORMAL);
    }

    bld->emit(cur);
}

size_t padding_op_t::compute_workload(const std::vector<shape_dtype_pair> &ins,
        const std::vector<shape_dtype_pair> &outs) {
    return fusible_op_t::compute_workload(ins, outs)
            * workload_penalty_coefficient;
}

std::vector<int> padding_op_t::get_real_padding_axis() {
    const int padding_dims_size = attrs_.get<sc_dims>("pads_begin").size();
    const int offset
            = info_.outputs_[0]->details_.get_format().is_channel_last() ? 1
                                                                         : 2;
    std::vector<int> padding_axis(padding_dims_size, 0);
    for (int i = 0; i < padding_dims_size; i++) {
        padding_axis[i] = i + offset;
    }
    return padding_axis;
}

stmt padding_op_t::get_zero_out_stmt(
        const tensor &out, const slice_range_list &range_list) {
    COMPILE_ASSERT(attrs_.has_key("pads_begin") && attrs_.has_key("pads_end"),
            "padding op shall have pads_begin & pads_end attributes");

    COMPILE_ASSERT(range_list.size() <= 1, "Multi-slice is not expected")

    // Support 4d or 5d output blocking format, e.g NCHW, NHWc, NCHWc
    // Todo (xurui) add support for output with D dim, such as NCDHWc
    COMPILE_ASSERT(get_inputs()[0]->details_.get_plain_dims().size() == 4,
            "padding op input was expected to be 4D");

    auto out_dtype = out->dtype_.is_pointer()
            ? out->dtype_.get_pointer_element()
            : out->dtype_;

    auto range = range_list.empty() ? slice_range {} : range_list[0];
    auto out_tsl = range.empty() ? tensor_slice(out)
                                 : tensor_slice(out, std::move(range));

    int N = get_expr_as_int(out_tsl.get_shape()[0]);
    auto real_padding_axis = get_real_padding_axis();

    auto is_channel_last
            = get_outputs()[0]->details_.get_format().is_channel_last();

    // All the format will be treated as NKHWc
    const int K = is_channel_last ? 1 : get_expr_as_int(out->dims_[1]);

    int c = 1;
    auto ndims = out->dims_.size();
    auto is_4d_out = ndims == 4;

    for (size_t i = real_padding_axis.back() + 1; i < ndims; i++) {
        c *= get_expr_as_int(out->dims_[i]);
    }

    const auto pads_begin = attrs_.get<sc_dims>("pads_begin");
    const auto pads_end = attrs_.get<sc_dims>("pads_end");

    // input plain format must be NCHW in conv_fwd_core
    auto input_plain_dims = get_inputs()[0]->details_.get_plain_dims();
    int plain_ndims_ = input_plain_dims.size();
    int w = input_plain_dims[plain_ndims_ - 1] + pads_begin[1] + pads_end[1];
    int oh_ = input_plain_dims[plain_ndims_ - 2];
    int ow_ = input_plain_dims[plain_ndims_ - 1];
    int ph1_ = pads_begin[0], ph2_ = pads_end[0];
    int pw1_ = pads_begin[1], pw2_ = pads_end[1];

    auto out_tptr = out_tsl.tptr_;

    for_loop ln, lk;
    builder::ir_builder_t bld;
    bld.push_scope();
    _named_for_(ln, n, 0, N, 1, for_type::PARALLEL) {
        _named_for_(lk, k, 0, K) {
            auto ptr = is_4d_out
                    ? (is_channel_last ? builder::tensor_ptr(
                               out_tptr, {n, 0, 0, 0})
                                       : builder::tensor_ptr(
                                               out_tptr, {n, k, 0, 0}))
                    : builder::tensor_ptr(out_tptr, {n, k, 0, 0, 0});
            sc::builtin::mem_zero(ptr, ph1_ * w * c, out_dtype);

            _for_(p1, 0, oh_) {
                sc::builtin::mem_zero(is_4d_out
                                ? (is_channel_last
                                                ? builder::tensor_ptr(out_tptr,
                                                        {n, p1 + ph1_, 0, 0})
                                                : builder::tensor_ptr(out_tptr,
                                                        {n, k, p1 + ph1_, 0}))
                                : builder::tensor_ptr(
                                        out_tptr, {n, k, p1 + ph1_, 0, 0}),

                        pw1_ * c, out_dtype);

                sc::builtin::mem_zero(
                        is_4d_out ? (is_channel_last
                                        ? builder::tensor_ptr(out_tptr,
                                                {n, p1 + ph1_, ow_ + pw1_, 0})
                                        : builder::tensor_ptr(out_tptr,
                                                {
                                                        n,
                                                        k,
                                                        p1 + ph1_,
                                                        ow_ + pw1_,
                                                }))
                                  : builder::tensor_ptr(out_tptr,
                                          {n, k, p1 + ph1_, ow_ + pw1_, 0}),

                        pw2_ * c, out_dtype);
            }

            sc::builtin::mem_zero(is_4d_out
                            ? (is_channel_last ? builder::tensor_ptr(
                                       out_tptr, {n, ph1_ + oh_, 0, 0})
                                               : builder::tensor_ptr(out_tptr,
                                                       {n, k, ph1_ + oh_, 0}))
                            : builder::tensor_ptr(
                                    out_tptr, {n, k, ph1_ + oh_, 0, 0}),
                    ph2_ * w * c, out_dtype);
        }
    }

    auto ret = bld.pop_scope();
    return ret;
}
std::vector<expr> padding_op_t::get_padding_offsets_exprs() {
    COMPILE_ASSERT(attrs_.has_key("pads_begin"),
            "padding op shall have pads_begin attribute")
    auto pads_begin = attrs_.get<sc_dims>("pads_begin");

    int ndims = get_outputs()[0]->details_.get_blocking_dims().size();
    auto real_padding_axis = get_real_padding_axis();

    COMPILE_ASSERT(pads_begin.size() == real_padding_axis.size(),
            "padding op shall have the same size of pads_begin and adding "
            "axis");

    std::vector<expr> offsets(ndims, 0);
    for (size_t i = 0; i < pads_begin.size(); i++) {
        offsets[real_padding_axis[i]] = (int)pads_begin[i];
    }
    return offsets;
}

OP_REGISTER(padding_op_t, padding)
} // namespace sc
