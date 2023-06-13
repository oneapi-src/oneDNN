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
#include <string>
#include <utility>
#include <vector>
#include "compiler/ir/graph/fusible_op_utils.hpp"
#include "reorder.hpp"
#include "util/math_utils.hpp"
#include "util/utils.hpp"
#include <compiler/ir/builder.hpp>
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

bool whole_buffer_reorder(const tensor_slice &src) {
    for (auto &offset : src.get_offset()) {
        if (!offset.isa<constant>() || get_expr_as_int(offset) != 0) {
            return false;
        }
    }
    return true;
}
// currently only support f32 8x8 and bf16 32x8
const int trans_lanesx8 = 8;
const int trans_lanesx16 = 16;
const int trans_lanes_bf16x8 = 32;
// [..., a, ... , b] <=> [..., b, ..., a]
bool can_be_fast_transpose(const context_ptr &ctx, std::vector<int> &inp_a_axis,
        std::vector<int> &inp_b_axis, std::vector<int> &out_a_axis,
        std::vector<int> &out_b_axis, const sc_dims &plain_dims,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, const tensor_slice &src,
        const tensor_slice &dst, const sc_data_type_t &dtype, bool is_dynamic,
        bool dynamic_no_padding, bool &use_lanesx16) {
    if (!ctx->machine_.cpu_flags_.fAVX2) { return false; }
    if (!dtype.is_etype(sc_data_etype::F32)
            && !dtype.is_etype(sc_data_etype::BF16)) {
        return false;
    }

    bool is_bf16 = dtype == datatypes::bf16;
    inp_a_axis.clear();
    inp_b_axis.clear();
    out_a_axis.clear();
    out_b_axis.clear();
    int inp_idx = 0, out_idx = 0;
    auto &inp_code = input_format.format_code_;
    auto &out_code = output_format.format_code_;
    int input_ndims = input_format.format_code_.ndims();
    int output_ndims = output_format.format_code_.ndims();
    auto input_blocking_shapes
            = sc_data_format_t::get_blocking_shapes(plain_dims, input_format);
    auto output_blocking_shapes
            = sc_data_format_t::get_blocking_shapes(plain_dims, output_format);
    auto inp_b_idx = inp_code.get(input_ndims - 1);
    auto out_a_idx = out_code.get(output_ndims - 1);
    if (inp_b_idx == out_a_idx) { return false; }
    while (inp_idx < input_ndims && out_idx < output_ndims) {
        while (inp_idx < input_ndims
                && utils::is_one_of(
                        inp_code.get(inp_idx), out_a_idx, inp_b_idx)) {
            if (inp_code.get(inp_idx) == out_a_idx) {
                inp_a_axis.push_back(inp_idx);
            } else {
                inp_b_axis.push_back(inp_idx);
            }
            inp_idx++;
        }
        while (inp_idx + 1 < input_ndims
                && inp_code.get(inp_idx + 1) == inp_code.get(inp_idx)) {
            inp_idx++;
        }
        while (out_idx < output_ndims
                && utils::is_one_of(
                        out_code.get(out_idx), out_a_idx, inp_b_idx)) {
            if (out_code.get(out_idx) == out_a_idx) {
                out_a_axis.push_back(out_idx);
            } else {
                out_b_axis.push_back(out_idx);
            }
            out_idx++;
        }
        while (out_idx + 1 < output_ndims
                && out_code.get(out_idx + 1) == out_code.get(out_idx)) {
            out_idx++;
        }
        auto orig_inp_idx = inp_code.get(inp_idx);
        auto orig_out_idx = out_code.get(out_idx);
        // other axis should be in same order.
        if (orig_inp_idx != orig_out_idx) { return false; }
        inp_idx++;
        out_idx++;
    }
    // input or output not end
    if (inp_idx < input_ndims || out_idx < output_ndims) { return false; }
    // number of non-transpose axis should be equal
    if (static_cast<size_t>(input_ndims) - inp_a_axis.size() - inp_b_axis.size()
            != static_cast<size_t>(output_ndims) - out_a_axis.size()
                    - out_b_axis.size()) {
        return false;
    }
    if (input_format.is_blocking() && output_format.is_blocking()) {
        // The transpose dimension in the output shape should be an integer
        // multiple of the input shape in block2block case.
        int ix = inp_a_axis[inp_a_axis.size() - 1];
        int iy = inp_b_axis[inp_b_axis.size() - 1];
        int ox = out_a_axis[out_a_axis.size() - 1];
        int oy = out_b_axis[out_b_axis.size() - 1];
        // can't do it in dynamic cases
        if (!src.shape_[ix].isa<constant>() || !src.shape_[iy].isa<constant>()
                || !dst.shape_[ox].isa<constant>()
                || !dst.shape_[oy].isa<constant>())
            return false;
        int inp_x = get_expr_as_int(src.shape_[ix]);
        int inp_y = get_expr_as_int(src.shape_[iy]);
        int out_x = get_expr_as_int(dst.shape_[ox]);
        int out_y = get_expr_as_int(dst.shape_[oy]);

        if (out_x % inp_x != 0 || out_y % inp_y != 0) { return false; }
    }
    auto satisfy_dim_lanes = [&]() {
        int trans_lanes1 = is_bf16 ? trans_lanes_bf16x8 : trans_lanesx8;
        int trans_lanes2 = trans_lanesx8;
        return plain_dims[inp_b_idx] % trans_lanes2 == 0
                && plain_dims[out_a_idx] % trans_lanes1 == 0
                && get_expr_as_int(
                           src.shape_[inp_a_axis[inp_a_axis.size() - 1]])
                        % trans_lanes1
                == 0
                && get_expr_as_int(
                           dst.shape_[out_b_axis[out_b_axis.size() - 1]])
                        % trans_lanes2
                == 0
                && get_expr_as_int(src.shape_[input_blocking_shapes.size() - 1])
                        % trans_lanes2
                == 0
                && get_expr_as_int(
                           dst.shape_[output_blocking_shapes.size() - 1])
                        % trans_lanes1
                == 0;
    };
    auto meet_bf16kernel_require = [&](int threshold) {
        int total = threshold;
        return get_expr_as_int(src.shape_[inp_a_axis[inp_a_axis.size() - 1]])
                        * get_expr_as_int(
                                src.shape_[inp_b_axis[inp_b_axis.size() - 1]])
                >= total
                && get_expr_as_int(
                           dst.shape_[out_a_axis[out_a_axis.size() - 1]])
                        * get_expr_as_int(
                                dst.shape_[out_b_axis[out_b_axis.size() - 1]])
                >= total;
    };
    // Currently bf16 calculation needs to be larger than
    // the number of elements threshold, otherwise the performance may
    // regression.
    int bf16_threshold = trans_lanesx8 * trans_lanes_bf16x8 / 2;
    if (is_bf16
            && (dynamic_no_padding
                    || (!is_dynamic
                            && !meet_bf16kernel_require(bf16_threshold)))) {
        return false;
    }
    // currently does not support tensor slice with padding.
    if (!whole_buffer_reorder(src) && (is_dynamic || !satisfy_dim_lanes())) {
        return false;
    }
    if (ctx->machine_.cpu_flags_.fAVX512F
            && plain_dims[inp_b_idx] > trans_lanesx8
            && plain_dims[out_a_idx] > trans_lanesx8) {
        // Currently we don't use f32x16 kernel.But we keep it for the
        // convenience of future performance comparison test on new machines.
        use_lanesx16 = false;
    }
    return true;
}

#define TRANS2D_ASSIGN(dst, src) \
    cur_list.emplace_back(builder::make_assign_unattached( \
            rows[((dst)-1)], rows[((src)-1)]));
// unpack and interleave
#define TRANS2D_UNPACK_ASSIGN(option, dst, src1, src2, elem_bits) \
    cur_list.emplace_back(builder::make_assign_unattached(rows[((dst)-1)], \
            builder::make_unpack_##option( \
                    rows[((src1)-1)], rows[((src2)-1)], elem_bits)));
#define TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32( \
        command, dst, src1, src2, imm, elem_bits) \
    cur_list.emplace_back(builder::make_assign_unattached(rows[((dst)-1)], \
            builder::make_##command( \
                    rows[((src1)-1)], rows[((src2)-1)], imm, elem_bits)));

#define PERMUTEX_ASSIGN_F32(dst, src1, src2, imm, mask) \
    cur_list.emplace_back(builder::make_assign_unattached(rows[((dst)-1)], \
            builder::make_permute( \
                    rows[((src1)-1)], rows[((src2)-1)], imm, mask)));

#define TRANS2D_REG_CALCULATION_F32(type_bits) \
    TRANS2D_UNPACK_ASSIGN(low, 9, 1, 2, 32) \
    TRANS2D_UNPACK_ASSIGN(high, 1, 1, 2, 32) \
    TRANS2D_UNPACK_ASSIGN(low, 10, 3, 4, 32) \
    TRANS2D_UNPACK_ASSIGN(high, 2, 3, 4, 32) \
    TRANS2D_UNPACK_ASSIGN(low, 11, 5, 6, 32) \
    TRANS2D_UNPACK_ASSIGN(high, 3, 5, 6, 32) \
    TRANS2D_UNPACK_ASSIGN(low, 12, 7, 8, 32) \
    TRANS2D_UNPACK_ASSIGN(high, 4, 7, 8, 32) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(shuffle, 5, 9, 10, 68, 32) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(shuffle, 6, 9, 10, 238, 32) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(shuffle, 7, 1, 2, 68, 32) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(shuffle, 8, 1, 2, 238, 32) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(shuffle, 9, 11, 12, 68, 32) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(shuffle, 10, 11, 12, 238, 32) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(shuffle, 11, 3, 4, 68, 32) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(shuffle, 12, 3, 4, 238, 32) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(permute, 1, 5, 9, 32, type_bits) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(permute, 2, 6, 10, 32, type_bits) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(permute, 3, 7, 11, 32, type_bits) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(permute, 4, 8, 12, 32, type_bits) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(permute, 5, 5, 9, 49, type_bits) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(permute, 6, 6, 10, 49, type_bits) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(permute, 7, 7, 11, 49, type_bits) \
    TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(permute, 8, 8, 12, 49, type_bits)

#define TRANS2D_REG_CALCULATION_BF16() \
    TRANS2D_UNPACK_ASSIGN(low, 9, 1, 2, 16) \
    TRANS2D_UNPACK_ASSIGN(high, 10, 1, 2, 16) \
    TRANS2D_UNPACK_ASSIGN(low, 11, 3, 4, 16) \
    TRANS2D_UNPACK_ASSIGN(high, 12, 3, 4, 16) \
    TRANS2D_UNPACK_ASSIGN(low, 13, 5, 6, 16) \
    TRANS2D_UNPACK_ASSIGN(high, 14, 5, 6, 16) \
    TRANS2D_UNPACK_ASSIGN(low, 15, 7, 8, 16) \
    TRANS2D_UNPACK_ASSIGN(high, 16, 7, 8, 16) \
    TRANS2D_UNPACK_ASSIGN(low, 1, 9, 11, 32) \
    TRANS2D_UNPACK_ASSIGN(high, 2, 9, 11, 32) \
    TRANS2D_UNPACK_ASSIGN(low, 3, 10, 12, 32) \
    TRANS2D_UNPACK_ASSIGN(high, 4, 10, 12, 32) \
    TRANS2D_UNPACK_ASSIGN(low, 5, 13, 15, 32) \
    TRANS2D_UNPACK_ASSIGN(high, 6, 13, 15, 32) \
    TRANS2D_UNPACK_ASSIGN(low, 7, 14, 16, 32) \
    TRANS2D_UNPACK_ASSIGN(high, 8, 14, 16, 32) \
    TRANS2D_UNPACK_ASSIGN(low, 9, 1, 5, 64) \
    TRANS2D_UNPACK_ASSIGN(high, 10, 1, 5, 64) \
    TRANS2D_UNPACK_ASSIGN(low, 11, 2, 6, 64) \
    TRANS2D_UNPACK_ASSIGN(high, 12, 2, 6, 64) \
    TRANS2D_UNPACK_ASSIGN(low, 13, 3, 7, 64) \
    TRANS2D_UNPACK_ASSIGN(high, 14, 3, 7, 64) \
    TRANS2D_UNPACK_ASSIGN(low, 15, 4, 8, 64) \
    TRANS2D_UNPACK_ASSIGN(high, 16, 4, 8, 64)

void compute_fast_transpose(sc_graph_t &graph, const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        const std::vector<int> &inp_a_axis, const std::vector<int> &inp_b_axis,
        const std::vector<int> &out_a_axis, const std::vector<int> &out_b_axis,
        size_t wkld, bool is_dynamic, bool dynamic_no_padding,
        bool use_lanesx16) {
    auto input = src.get_real_tensor();
    auto output = dst.get_real_tensor();
    int step = use_lanesx16 && dtype == datatypes::f32
            ? trans_lanesx16
            : trans_lanesx8; // fixed f32x8
    auto bld = builder::get_current_builder();
    auto input_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, input_format);
    auto output_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, output_format);
    auto input_blocking_dims_expr
            = get_blocking_shapes_expr(graph, plain_dims, input_format);
    auto output_blocking_dims_expr
            = get_blocking_shapes_expr(graph, plain_dims, output_format);
    auto input_format_code = input_format.format_code_;
    std::vector<expr> rows;
    std::vector<expr> iter_vars;
    std::vector<stmt_c> cur_list;
    stmt cur, body;
    bool is_padding = false;
    int inp_a_step = dtype == datatypes::f32
            ? (use_lanesx16 ? trans_lanesx16 : trans_lanesx8)
            : trans_lanes_bf16x8;
    int inp_b_step = use_lanesx16 && dtype == datatypes::f32 ? trans_lanesx16
                                                             : trans_lanesx8;
    const int type_bits = utils::get_sizeof_type(sc_data_type_t::f32(4)) * 8;
    if ((!is_dynamic
                && (input_blocking_dims[inp_a_axis.back()] % inp_a_step
                        || input_blocking_dims[inp_b_axis.back()] % inp_b_step
                        || output_blocking_dims[out_a_axis.back()] % inp_a_step
                        || output_blocking_dims[out_b_axis.back()] % inp_b_step
                        || math_utils::get_dims_product(input_blocking_dims)
                                != math_utils::get_dims_product(
                                        output_blocking_dims)))
            || (is_dynamic && !dynamic_no_padding)) {
        is_padding = true;
    }
    if (dtype == datatypes::f32) {
        rows.resize(step + step / 2);
        for (auto i = 0; i < step + step / 2; i++) {
            rows[i] = builder::make_var(sc_data_type_t::f32(step),
                    "row" + std::to_string(i + 1) + fusion_create_var_idx());
            cur_list.emplace_back(
                    builder::make_var_tensor_def_unattached(rows[i]));
        }
    } else if (dtype == datatypes::bf16) {
        rows.resize(16); // bf16 uses 16 zmms.
        for (auto i = 0; i < 16; i++) {
            rows[i] = builder::make_var(sc_data_type_t::bf16(32),
                    "row" + std::to_string(i + 1) + fusion_create_var_idx());
            cur_list.emplace_back(
                    builder::make_var_tensor_def_unattached(rows[i]));
        }
    }

    auto determine_cur_step = [&](const std::vector<expr> &blocking_dims_expr,
                                      const std::vector<expr> &tmp_in_indexes,
                                      const std::vector<expr> &plain_indexes,
                                      expr &cur_step, expr &sup_condition,
                                      int in_axis, bool use_output_loop,
                                      int step) {
        if (!use_output_loop) {
            cur_step = builder::make_min(
                    builder::make_max(cast_to_s32(blocking_dims_expr.back())
                                    - cast_to_s32(tmp_in_indexes.back()),
                            0),
                    step);
            sup_condition
                    = tmp_in_indexes[in_axis] < blocking_dims_expr[in_axis];
        } else {
            auto tmp_plain = graph.dims_to_expr(plain_dims);
            auto input_last_dim
                    = input_format_code.get(input_format_code.ndims() - 1);
            auto input_other_dim = input_format_code.get(in_axis);
            cur_step = builder::make_min(
                    builder::make_max(cast_to_s32(tmp_plain[input_last_dim])
                                    - cast_to_s32(
                                            plain_indexes[input_last_dim]),
                            0),
                    step);
            sup_condition = plain_indexes[input_other_dim]
                    < tmp_plain[input_other_dim];
        }
    };

    auto compute_transpose_f32 = [&](const std::vector<expr> &in_indexes,
                                         const std::vector<expr> &out_indexes,
                                         const std::vector<expr>
                                                 &plain_indexes) {
        expr mask;
        for (int i = 0; i < step; i++) {
            auto tmp_in_indexes = in_indexes;
            auto in_axis = inp_a_axis[inp_a_axis.size() - 1];
            tmp_in_indexes[in_axis]
                    = tmp_in_indexes[in_axis] + static_cast<uint64_t>(i);
            auto tmp_plain_indexes = plain_indexes;
            tmp_plain_indexes[input_format_code.get(in_axis)]
                    = tmp_plain_indexes[input_format_code.get(in_axis)]
                    + static_cast<uint64_t>(i);
            expr tmp_in = src.tptr_;
            if (output_loop) { tmp_in = input; }
            if (is_padding) {
                expr cur_step, sup_condition;
                determine_cur_step(input_blocking_dims_expr, tmp_in_indexes,
                        tmp_plain_indexes, cur_step, sup_condition, in_axis,
                        output_loop, step);
                stmt mask_def;
                mask = generate_mask_var_by_step(
                        mask_def, cur_step, step, sup_condition);
                cur_list.emplace_back(mask_def);
            }
            auto assign = builder::make_assign_unattached(rows[i],
                    // here, use src.tptr instead of input is aimed to
                    // avoid input is tensor_view_op. Otherwise, it will
                    // throw illegal exception in tensor_shrink
                    builder::make_indexing(tmp_in, tmp_in_indexes, step, mask));
            assign->attr()[op_traits::workload_computable_t::workload_number]
                    = wkld;
            cur_list.emplace_back(assign);
        }
        TRANS2D_REG_CALCULATION_F32(type_bits);
        for (int i = 0; i < step; i++) {
            auto tmp_out_indexes = out_indexes;
            auto out_axis = out_b_axis[out_b_axis.size() - 1];
            tmp_out_indexes[out_axis]
                    = tmp_out_indexes[out_axis] + static_cast<uint64_t>(i);
            if (is_padding) {
                auto cur_step = builder::make_min(
                        builder::make_max(
                                cast_to_s32(output_blocking_dims_expr.back())
                                        - cast_to_s32(tmp_out_indexes.back()),
                                0),
                        step);
                auto sup_condition = tmp_out_indexes[out_axis]
                        < output_blocking_dims_expr[out_axis];
                // ABba(4, 4, 16, 4) => AB(16,64)
                if (input_format.is_blocking()) {
                    sup_condition = sup_condition
                            && in_indexes.back() + i
                                    < input_blocking_dims_expr.back();
                }
                stmt mask_def;
                mask = generate_mask_var_by_step(
                        mask_def, cur_step, step, sup_condition);
                cur_list.emplace_back(mask_def);
            }
            auto assign = builder::make_assign_unattached(
                    builder::make_indexing(output, tmp_out_indexes, step, mask),
                    rows[i]);
            cur_list.emplace_back(assign);
        }
    };

    auto compute_transpose_bf16 = [&](const std::vector<expr> &in_indexes,
                                          const std::vector<expr> &out_indexes,
                                          const std::vector<expr>
                                                  &plain_indexes) {
        expr mask;
        for (int i = 0; i < step; i++) {
            for (int p = 0; p < 4; p++) {
                auto tmp_in_indexes = in_indexes;
                auto in_axis = inp_a_axis[inp_a_axis.size() - 1];
                tmp_in_indexes[in_axis] = tmp_in_indexes[in_axis]
                        + (static_cast<uint64_t>(i) + p * 8);
                auto tmp_plain_indexes = plain_indexes;
                tmp_plain_indexes[input_format_code.get(in_axis)]
                        = tmp_plain_indexes[input_format_code.get(in_axis)]
                        + (static_cast<uint64_t>(i) + p * 8);
                expr tmp_in = src.tptr_;
                if (output_loop) { tmp_in = input; }
                if (is_padding) {
                    expr cur_step, sup_condition;
                    determine_cur_step(input_blocking_dims_expr, tmp_in_indexes,
                            tmp_plain_indexes, cur_step, sup_condition, in_axis,
                            output_loop, trans_lanesx8);
                    stmt mask_def;
                    mask = generate_mask_var_by_step(
                            mask_def, cur_step, trans_lanesx8, sup_condition);
                    cur_list.emplace_back(mask_def);
                }
                auto brct_src = builder::make_broadcast(
                        builder::make_indexing(
                                tmp_in, tmp_in_indexes, step, mask),
                        trans_lanes_bf16x8);
                auto assign = builder::make_assign_unattached(rows[i],
                        // here, use src.tptr instead of input is aimed
                        // to avoid input is tensor_view_op. Otherwise,
                        // it will throw illegal exception in
                        // tensor_shrink
                        p > 0 ? builder::make_select(
                                0xff << (p * step), brct_src, rows[i])
                              : brct_src);
                assign->attr()
                        [op_traits::workload_computable_t::workload_number]
                        = wkld;
                cur_list.emplace_back(assign);
            }
        }

        TRANS2D_REG_CALCULATION_BF16();
        for (int i = 0; i < step; i++) {
            auto tmp_out_indexes = out_indexes;
            auto out_axis = out_b_axis[out_b_axis.size() - 1];
            tmp_out_indexes[out_axis]
                    = tmp_out_indexes[out_axis] + static_cast<uint64_t>(i);
            if (is_padding) {
                auto cur_step = builder::make_min(
                        builder::make_max(
                                cast_to_s32(output_blocking_dims_expr.back())
                                        - cast_to_s32(tmp_out_indexes.back()),
                                0),
                        trans_lanes_bf16x8);
                auto sup_condition = tmp_out_indexes[out_axis]
                        < output_blocking_dims_expr[out_axis];
                // ABba(4, 4, 16, 4) => AB(16,64)
                if (input_format.is_blocking()) {
                    sup_condition = sup_condition
                            && in_indexes.back() + i
                                    < input_blocking_dims_expr.back();
                }
                stmt mask_def;
                mask = generate_mask_var_by_step(
                        mask_def, cur_step, trans_lanes_bf16x8, sup_condition);
                cur_list.emplace_back(mask_def);
            }
            auto assign = builder::make_assign_unattached(
                    builder::make_indexing(
                            output, tmp_out_indexes, trans_lanes_bf16x8, mask),
                    rows[i + 8]);
            cur_list.emplace_back(assign);
        }
    };

    auto compute_loops = [&](const std::vector<expr> &blocking_dims,
                                 const std::vector<int> &a_axis,
                                 const std::vector<int> &b_axis,
                                 const tensor_slice &tsr) {
        for (int i = static_cast<int>(blocking_dims.size()) - 1; i >= 0; i--) {
            if (utils::is_one_of(i, a_axis.back(), b_axis.back())) {
                body = cur.isa<stmts>()
                        ? cur
                        : make_stmt<stmts_node_t>(
                                std::vector<stmt> {std::move(cur)});
                int cur_step = i == a_axis.back() ? inp_a_step : inp_b_step;
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0),
                        (!is_padding || !tsr.get_offset()[i].isa<constant>()
                                || !output_loop)
                                ? tsr.get_shape()[i]
                                : blocking_dims[i],
                        expr(cur_step), std::move(body), true,
                        for_type::NORMAL);
            }
        }
        for (int i = static_cast<int>(blocking_dims.size()) - 1; i >= 0; i--) {
            if (!utils::is_one_of(i, a_axis.back(), b_axis.back())) {
                body = cur.isa<stmts>()
                        ? cur
                        : make_stmt<stmts_node_t>(
                                std::vector<stmt> {std::move(cur)});
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0),
                        (!is_padding || !tsr.get_offset()[i].isa<constant>()
                                || !output_loop)
                                ? tsr.get_shape()[i]
                                : blocking_dims[i],
                        expr(1), std::move(body), true, for_type::NORMAL);
            }
        }
    };
    if (!output_loop) {
        std::vector<expr> in_indexes, loop_indexes;
        for (size_t i = 0; i < input_blocking_dims_expr.size(); i++) {
            iter_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("_fuseiter") + fusion_create_idx()));
            in_indexes.emplace_back(iter_vars[i] + src.get_offset()[i]);
            loop_indexes.emplace_back(iter_vars[i]);
        }
        expr condition;
        expr last_axis_offset, other_axis_condition;
        std::vector<expr> tmp_indexes = get_reorder_block2plain_indexes(graph,
                in_indexes, input_format, plain_dims, condition,
                last_axis_offset, other_axis_condition,
                input_format.format_code_.get(
                        static_cast<int>(input_format.format_code_.ndims())
                        - 1));
        std::vector<expr> out_indexes
                = get_reorder_plain2block_indexes(tmp_indexes, output_format);
        if (dtype == datatypes::f32) {
            compute_transpose_f32(loop_indexes, out_indexes, tmp_indexes);
        } else {
            compute_transpose_bf16(loop_indexes, out_indexes, tmp_indexes);
        }
        cur = builder::make_stmts_unattached(cur_list);
        compute_loops(input_blocking_dims_expr, inp_a_axis, inp_b_axis, src);
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    } else {
        std::vector<expr> out_indexes;
        for (size_t i = 0; i < output_blocking_dims_expr.size(); i++) {
            iter_vars.emplace_back(builder::make_var(datatypes::index,
                    std::string("_fuseiter") + fusion_create_idx()));
            out_indexes.emplace_back(iter_vars[i] + dst.get_offset()[i]);
        }
        expr condition;
        expr last_axis_offset, other_axis_condition;
        std::vector<expr> tmp_indexes = get_reorder_block2plain_indexes(graph,
                out_indexes, output_format, plain_dims, condition,
                last_axis_offset, other_axis_condition);
        std::vector<expr> in_indexes
                = get_reorder_plain2block_indexes(tmp_indexes, input_format);
        if (dtype == datatypes::f32) {
            compute_transpose_f32(in_indexes, out_indexes, tmp_indexes);
        } else {
            compute_transpose_bf16(in_indexes, out_indexes, tmp_indexes);
        }
        cur = builder::make_stmts_unattached(cur_list);
        compute_loops(output_blocking_dims_expr, out_a_axis, out_b_axis, dst);
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    }
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
