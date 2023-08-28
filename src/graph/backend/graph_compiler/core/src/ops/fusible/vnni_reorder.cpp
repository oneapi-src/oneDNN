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
#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include "compiler/config/context.hpp"
#include "compiler/ir/graph/fusible_op_utils.hpp"
#include "reorder.hpp"
#include "util/math_utils.hpp"
#include <compiler/ir/builder.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

static sc_vnni_kernel get_vnni_kernel_type(sc_data_etype etype) {
    switch (etype) {
        case sc_data_etype::BF16: {
            return sc_vnni_kernel::X16_REORDER_VNNI;
        } break;
        case sc_data_etype::U8: {
            return sc_vnni_kernel::X16_REORDER_VNNI;
        } break;
        case sc_data_etype::S8: {
            return sc_vnni_kernel::X16_REORDER_VNNI;
        } break;
        default: {
            COMPILE_ASSERT(false, "Do not support dtype: " << etype);
        } break;
    }
    return sc_vnni_kernel::NO_VNNI;
}

bool can_be_vnni_reorder(const context_ptr &ctx, std::vector<int> &inp_n_axis,
        std::vector<int> &inp_k_axis, std::vector<int> &out_n_axis,
        std::vector<int> &out_k_axis, const sc_dims &plain_dims,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, const tensor_slice &src,
        const tensor_slice &dst, const sc_data_type_t &dtype,
        bool &is_vnni_reorder, bool is_dynamic, bool dynamic_no_padding,
        sc_vnni_kernel &vnni_kernel_used) {
    // VNNI reorder only support NK2NKknk-liked format.
    // Last axis should be 2 if dytpe is bf16 and 4 if dytpe is u8/s8
    // eg. 384N 64K -> 12N 4K 8k 32n 2k
    //     384N 64K -> 12N 2K 8k 32n 4k
    //     128A 16B 32C -> 128A 2B 2C 4c 8b 4c
    // dynamic cases can't check in current condition
    if (!ctx->machine_.cpu_flags_.fAVX2) { return false; }
    auto input_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, input_format);
    auto output_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, output_format);
    bool is_padding = false;
    if ((!is_dynamic
                && math_utils::get_dims_product(input_blocking_dims)
                        != math_utils::get_dims_product(output_blocking_dims))
            || (is_dynamic && !dynamic_no_padding)) {
        is_padding = true;
    }

    if (input_format.is_blocking()) {
        vnni_kernel_used = sc_vnni_kernel::NO_VNNI;
        return false;
    }
    bool is_bf16 = dtype.as_etype() == sc_data_etype::BF16;
    inp_n_axis.clear();
    inp_k_axis.clear();
    out_n_axis.clear();
    out_k_axis.clear();
    if (!utils::is_one_of(dtype.as_etype(), sc_data_etype::U8,
                sc_data_etype::S8, sc_data_etype::BF16)) {
        vnni_kernel_used = sc_vnni_kernel::NO_VNNI;
        return false;
    }
    int inp_idx = 0, out_idx = 0;
    auto &inp_code = input_format.format_code_;
    auto &out_code = output_format.format_code_;
    int input_ndims = input_format.format_code_.ndims();
    int output_ndims = output_format.format_code_.ndims();
    int vectorized_last_dims = output_ndims - 1;
    int out_dim_counts[sc_data_format_kind_t::MAX_DIMS] = {0};
    output_format.format_code_.collect_dim_count(out_dim_counts);
    int vectorized_original_axis
            = output_format.format_code_.get(vectorized_last_dims);
    // Relax the restriction that the irrelevant dimension in the output is 1
    for (int i = vectorized_last_dims; i >= 0; i--) {
        auto tmp = output_format.format_code_.get(i);
        auto target_axis_dim = dst.shape_[i];
        // can't do vnni in dynamic cases
        if (!target_axis_dim.isa<constant>()) {
            vnni_kernel_used = sc_vnni_kernel::NO_VNNI;
            return false;
        }
        if (out_dim_counts[tmp] > 1 && (get_expr_as_int(target_axis_dim) > 1)) {
            vectorized_last_dims = i;
            vectorized_original_axis = tmp;
            break;
        }
    }
    if (out_dim_counts[vectorized_original_axis] < 2) {
        vnni_kernel_used = sc_vnni_kernel::NO_VNNI;
        return false;
    }

    if (!dst.get_shape().at(vectorized_last_dims).isa<constant>()
            || get_expr_as_int(
                       dst.get_real_tensor()->strides_[vectorized_last_dims])
                    != 1) {
        vnni_kernel_used = sc_vnni_kernel::NO_VNNI;
        return false;
    }

    auto out_k2_pos = vectorized_last_dims,
         out_n_pos = vectorized_last_dims - 1, out_k_pos = -1, out_K_pos = -1,
         out_N_pos = -1, in_K_pos = -1, in_N_pos = -1;
    auto k_idx = out_code.get(out_k2_pos);
    auto n_idx = out_code.get(out_n_pos);

    for (auto i = vectorized_last_dims - 1; i >= 0; --i) {
        if (out_code.get(i) == k_idx) {
            if (out_k_pos == -1) {
                out_k_pos = i;
            } else if (out_K_pos == -1) {
                out_K_pos = i;
            }
        }
    }

    for (auto i = vectorized_last_dims - 2; i >= 0; --i) {
        if (out_code.get(i) == n_idx) {
            if (out_N_pos == -1) { out_N_pos = i; }
        }
    }

    for (auto i = input_ndims - 1; i >= 0; --i) {
        if (inp_code.get(i) == k_idx) {
            if (in_K_pos == -1) { in_K_pos = i; }
        }
    }
    for (auto i = input_ndims - 1; i >= 0; --i) {
        if (inp_code.get(i) == n_idx) {
            if (in_N_pos == -1) { in_N_pos = i; }
        }
    }
    // our K and N dims need to be constant
    if (!(src.get_shape().at(in_K_pos).isa<constant>()
                || src.get_shape().at(in_N_pos).isa<constant>())) {
        vnni_kernel_used = sc_vnni_kernel::NO_VNNI;
        return false;
    }
    // Relax the restriction that the irrelevant dimension in the input is 1
    // eg: ABC[16,16,1] -> ABCbab[1,1,1,8,16,2] C dim is 1, still use vnni
    // reorder
    if (!(inp_code.get(input_ndims - 1) == k_idx
                || inp_code.get(input_ndims - 1) == n_idx)) {
        int input_lastdim_max_idx = std::max(in_K_pos, in_N_pos);
        if (get_expr_as_int(
                    src.get_real_tensor()->strides_[input_lastdim_max_idx])
                != 1) {
            vnni_kernel_used = sc_vnni_kernel::NO_VNNI;
            return false;
        }
    }

    if ((in_N_pos > in_K_pos && out_n_pos > out_k_pos)
            || (in_N_pos < in_K_pos && out_n_pos < out_k_pos)) {
        is_vnni_reorder = true;
    }
    // find axie of N K and N K k n k
    out_n_axis.emplace_back(out_N_pos);
    out_n_axis.emplace_back(out_n_pos);
    out_k_axis.emplace_back(out_K_pos);
    out_k_axis.emplace_back(out_k_pos);
    out_k_axis.emplace_back(out_k2_pos);
    inp_n_axis.emplace_back(in_N_pos);
    inp_k_axis.emplace_back(in_K_pos);
    // VNNI reorder kernel shape is 4x16 for u8/s8 and 4x8 for bf16.
    vnni_kernel_used = get_vnni_kernel_type(dtype.as_etype());
    if (!is_padding) {
        if (get_expr_as_int(dst.shape_[out_k2_pos]) != (is_bf16 ? 2 : 4)) {
            vnni_kernel_used = sc_vnni_kernel::NO_VNNI;
            return false;
        }
        if (!is_vnni_reorder) {
            if (get_expr_as_int(dst.shape_[out_n_pos]) % 4 != 0) {
                vnni_kernel_used = sc_vnni_kernel::NO_VNNI;
                return false;
            }
            if (get_expr_as_int(dst.shape_[out_k_pos]) % 4 != 0) {
                vnni_kernel_used = sc_vnni_kernel::NO_VNNI;
                return false;
            }
            vnni_kernel_used = is_bf16 ? sc_vnni_kernel::BF16_TRANSPOSE_VNNI
                                       : sc_vnni_kernel::U8S8_TRANSPOSE_VNNI;
        } else {
            if (get_expr_as_int(dst.shape_[out_n_pos]) % 16 == 0) {
                vnni_kernel_used = sc_vnni_kernel::X16_REORDER_VNNI;
            } else if (get_expr_as_int(dst.shape_[out_n_pos])
                            % (is_bf16 ? 8 : 16)
                    == 0) {
                vnni_kernel_used = sc_vnni_kernel::X8_REORDER_VNNI;
            } else {
                vnni_kernel_used = sc_vnni_kernel::NO_VNNI;
                return false;
            }

            if ((get_expr_as_int(dst.shape_[out_k_pos])
                        * get_expr_as_int(dst.shape_[out_k2_pos]))
                            % 4
                    != 0) {
                vnni_kernel_used = sc_vnni_kernel::NO_VNNI;
                return false;
            }
        }
    } else {
        if (output_blocking_dims[out_k2_pos] != (is_bf16 ? 2 : 4)) {
            vnni_kernel_used = sc_vnni_kernel::NO_VNNI;
            return false;
        }
        if (!is_vnni_reorder) {
            if (output_blocking_dims[out_n_pos] % 4 != 0) {
                vnni_kernel_used = sc_vnni_kernel::NO_VNNI;
                return false;
            }
            if (output_blocking_dims[out_k_pos] % 4 != 0) {
                vnni_kernel_used = sc_vnni_kernel::NO_VNNI;
                return false;
            }
            vnni_kernel_used = is_bf16 ? sc_vnni_kernel::BF16_TRANSPOSE_VNNI
                                       : sc_vnni_kernel::U8S8_TRANSPOSE_VNNI;
        } else {
            if (output_blocking_dims[out_n_pos] % 16 == 0) {
                vnni_kernel_used = sc_vnni_kernel::X16_REORDER_VNNI;
            } else if (output_blocking_dims[out_n_pos] % (is_bf16 ? 8 : 16)
                    == 0) {
                vnni_kernel_used = sc_vnni_kernel::X8_REORDER_VNNI;
            } else {
                vnni_kernel_used = sc_vnni_kernel::NO_VNNI;
                return false;
            }
            if (output_blocking_dims[out_k_pos]
                            * output_blocking_dims[out_k2_pos] % 4
                    != 0) {
                vnni_kernel_used = sc_vnni_kernel::NO_VNNI;
                return false;
            }
        }
    }

    return true;
}

void do_vnni_reorder(std::vector<stmt_c> &cur_list, std::vector<expr> &rows,
        sc_data_type_t &rows_dtype, const bool is_vnni_reorder,
        const int bf16_step) {
    bool is_bf16 = rows_dtype.type_code_ == sc_data_etype::BF16;
    bool is_u8 = rows_dtype.type_code_ == sc_data_etype::U8;
    // reorder on a kernel of 4x16(u8/s8) or 4x8(bf16)
    // registers to perform reorder, should reinterpret data to u8 due to
    // intrinsic limitation
    any_map_t reinterpret_attr;
    expr xmm0, xmm1, xmm2, xmm3, xmm_tmp;
#define PARAM(X) X
#define MAKE_VAR(name, type) \
    PARAM(name) = builder::make_var(sc_data_type_t::type, std::string(#name)); \
    cur_list.emplace_back(builder::make_var_tensor_def_unattached(name));

    stmt assign;
#define MAKE_ASSIGN(dst, src) \
    assign = builder::make_assign_unattached(dst, src); \
    cur_list.emplace_back(assign);

#define MAKE_INTERPRET(dst, src, attr) \
    MAKE_ASSIGN(dst, \
            make_expr<intrin_call_node>( \
                    intrin_type::reinterpret, std::vector<expr> {src}, attr));

#define MAKE_UNPACK_HIGH(dst, src1, src2, elem_bit) \
    MAKE_ASSIGN(dst, builder::make_unpack_high(src1, src2, elem_bit))

#define MAKE_UNPACK_LOW(dst, src1, src2, elem_bit) \
    MAKE_ASSIGN(dst, builder::make_unpack_low(src1, src2, elem_bit))

#define MAKE_PERMUTE(dst, src1, src2, imm, elem_bits) \
    MAKE_ASSIGN(dst, builder::make_permute(src1, src2, imm, elem_bits))

    if (!is_vnni_reorder) {
        MAKE_VAR(xmm0, u32(4))
        MAKE_VAR(xmm1, u32(4))
        MAKE_VAR(xmm2, u32(4))
        MAKE_VAR(xmm3, u32(4))
        reinterpret_attr[intrin_attr::out_dtype] = sc_data_type_t::u32(4);
        MAKE_INTERPRET(xmm0, rows[0].remove_const(), reinterpret_attr)
        MAKE_INTERPRET(xmm1, rows[1].remove_const(), reinterpret_attr)
        MAKE_INTERPRET(xmm2, rows[2].remove_const(), reinterpret_attr)
        MAKE_INTERPRET(xmm3, rows[3].remove_const(), reinterpret_attr)
        expr xmm_tmp0, xmm_tmp1, xmm_tmp2, xmm_tmp3, xmm_tmp4, xmm_tmp5;
        MAKE_VAR(xmm_tmp0, u32(4))
        MAKE_VAR(xmm_tmp1, index(2))
        MAKE_VAR(xmm_tmp2, index(2))
        MAKE_VAR(xmm_tmp3, index(2))
        MAKE_VAR(xmm_tmp4, index(2))
        MAKE_VAR(xmm_tmp5, index(2))
        MAKE_UNPACK_LOW(xmm_tmp0, xmm0, xmm1, 32)
        MAKE_UNPACK_HIGH(xmm1, xmm0, xmm1, 32)
        MAKE_ASSIGN(xmm0, xmm_tmp0)
        MAKE_UNPACK_LOW(xmm_tmp0, xmm2, xmm3, 32)
        MAKE_UNPACK_HIGH(xmm3, xmm2, xmm3, 32)
        reinterpret_attr[intrin_attr::out_dtype] = sc_data_type_t::index(2);
        MAKE_INTERPRET(xmm_tmp1, xmm0, reinterpret_attr)
        MAKE_INTERPRET(xmm_tmp2, xmm1, reinterpret_attr)
        MAKE_INTERPRET(xmm_tmp3, xmm_tmp0, reinterpret_attr)
        MAKE_INTERPRET(xmm_tmp4, xmm3, reinterpret_attr)
        MAKE_UNPACK_LOW(xmm_tmp5, xmm_tmp1, xmm_tmp3, 64)
        MAKE_UNPACK_HIGH(xmm_tmp3, xmm_tmp1, xmm_tmp3, 64)
        MAKE_ASSIGN(xmm_tmp1, xmm_tmp5)
        MAKE_UNPACK_LOW(xmm_tmp5, xmm_tmp2, xmm_tmp4, 64)
        MAKE_UNPACK_HIGH(xmm_tmp4, xmm_tmp2, xmm_tmp4, 64)
        reinterpret_attr[intrin_attr::out_dtype] = rows_dtype;
        MAKE_INTERPRET(rows[0], xmm_tmp1, reinterpret_attr)
        MAKE_INTERPRET(rows[1], xmm_tmp3, reinterpret_attr)
        MAKE_INTERPRET(rows[2], xmm_tmp5, reinterpret_attr)
        MAKE_INTERPRET(rows[3], xmm_tmp4, reinterpret_attr)
        return;
    } else if (is_bf16) {
        if (bf16_step == 16) {
            MAKE_VAR(xmm0, bf16(bf16_step))
            MAKE_VAR(xmm1, bf16(bf16_step))
            MAKE_UNPACK_LOW(xmm0, rows[0], rows[1], 16)
            MAKE_UNPACK_HIGH(rows[1], rows[0], rows[1], 16)
            MAKE_UNPACK_LOW(xmm1, rows[2], rows[3], 16)
            MAKE_UNPACK_HIGH(rows[3], rows[2], rows[3], 16)
            MAKE_PERMUTE(rows[0], xmm0, rows[1], 0x20, 128)
            MAKE_PERMUTE(rows[1], xmm0, rows[1], 0x31, 128)
            MAKE_PERMUTE(rows[2], xmm1, rows[3], 0x20, 128)
            MAKE_PERMUTE(rows[3], xmm1, rows[3], 0x31, 128)
        } else {
            MAKE_VAR(xmm0, bf16(8))
            MAKE_VAR(xmm1, bf16(8))
            MAKE_UNPACK_LOW(xmm0, rows[0], rows[1], 16)
            MAKE_UNPACK_HIGH(rows[1], rows[0], rows[1], 16)
            MAKE_UNPACK_LOW(xmm1, rows[2], rows[3], 16)
            MAKE_UNPACK_HIGH(rows[3], rows[2], rows[3], 16)
            MAKE_ASSIGN(rows[0], xmm0)
            MAKE_ASSIGN(rows[2], xmm1)
        }
    } else {
        if (is_u8) {
            MAKE_VAR(xmm0, u8(16))
            MAKE_VAR(xmm1, u8(16))
            MAKE_VAR(xmm2, u8(16))
            MAKE_VAR(xmm3, u8(16))
        } else {
            MAKE_VAR(xmm0, s8(16))
            MAKE_VAR(xmm1, s8(16))
            MAKE_VAR(xmm2, s8(16))
            MAKE_VAR(xmm3, s8(16))
        }
        MAKE_UNPACK_LOW(xmm0, rows[0], rows[2], 8)
        MAKE_UNPACK_HIGH(rows[2], rows[0], rows[2], 8)
        MAKE_UNPACK_LOW(xmm1, rows[1], rows[3], 8)
        MAKE_UNPACK_HIGH(rows[3], rows[1], rows[3], 8)
        MAKE_UNPACK_LOW(rows[0], xmm0, xmm1, 8)
        MAKE_UNPACK_HIGH(xmm2, xmm0, xmm1, 8)
        MAKE_UNPACK_LOW(xmm3, rows[2], rows[3], 8)
        MAKE_UNPACK_HIGH(rows[3], rows[2], rows[3], 8)
        MAKE_ASSIGN(rows[1], xmm2)
        MAKE_ASSIGN(rows[2], xmm3)
    }
}

// Keep this flag to test insert kernel performance in the future.

void compute_vnni_reorder(sc_graph_t &graph, const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        std::vector<int> &inp_n_axis, std::vector<int> &inp_k_axis,
        std::vector<int> &out_n_axis, std::vector<int> &out_k_axis, size_t wkld,
        const bool &is_vnni_reorder, bool is_dynamic, bool dynamic_no_padding,
        const sc_vnni_kernel vnni_kernel_used) {
    bool is_bf16 = dtype.as_etype() == sc_data_etype::BF16;
    bool is_u8 = dtype.as_etype() == sc_data_etype::U8;
    auto input = src.get_real_tensor();
    auto output = dst.get_real_tensor();
    int step = 4;
    auto bld = builder::get_current_builder();
    auto input_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, input_format);
    auto input_blocking_shape_expr
            = get_blocking_shapes_expr(graph, plain_dims, input_format);
    auto output_blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, output_format);
    auto input_blocking_dims_expr
            = get_blocking_shapes_expr(graph, plain_dims, input_format);
    auto output_blocking_dims_expr
            = get_blocking_shapes_expr(graph, plain_dims, output_format);
    // plain axis of last block
    auto input_last_origin_axis = input_format.format_code_.get(
            input_format.format_code_.ndims() - 1);
    auto output_last_origin_axis = output_format.format_code_.get(
            output_format.format_code_.ndims() - 1);
    int input_origin_axis_vectorized = input_format.format_code_.ndims() - 1;
    int output_origin_axis_vectorized = output_format.format_code_.ndims() - 1;
    find_vectorized_axis(input_blocking_dims_expr, input_format,
            input_last_origin_axis, input_origin_axis_vectorized);
    find_vectorized_axis(output_blocking_dims_expr, output_format,
            output_last_origin_axis, output_origin_axis_vectorized);
    bool is_padding = false;
    if ((!is_dynamic
                && math_utils::get_dims_product(input_blocking_dims)
                        != math_utils::get_dims_product(output_blocking_dims))
            || (is_dynamic && !dynamic_no_padding)) {
        is_padding = true;
    }
    bool use_x16step = vnni_kernel_used == sc_vnni_kernel::X16_REORDER_VNNI;
    int u8_step = 16, bf16_step = use_x16step ? 16 : 8;

    std::vector<expr> rows(step);
    std::vector<expr> iter_vars;
    std::vector<stmt_c> cur_list;
    auto rows_dtype = dtype;
    if (use_x16step) {
        rows_dtype.lanes_ = 16;
    } else {
        rows_dtype.lanes_ = is_bf16 ? 8 : 16;
    }

#define PARAM(X) X
#define MAKE_VAR_ASSIGN_VNNI(name, type, alias) \
    PARAM(name) = builder::make_var(sc_data_type_t::type, \
            #alias + std::string("_vnni_reorder_") + std::to_string(i + 1) \
                    + fusion_create_var_idx());

    auto func_bf16_forloop_step = [&](int step,
                                          std::vector<expr> &tmp_out_indexes,
                                          int loop_step) {
        switch (step) {
            case 1:
                tmp_out_indexes[out_n_axis[1]] = tmp_out_indexes[out_n_axis[1]]
                        + static_cast<uint64_t>(loop_step);
                break;
            case 2:
                tmp_out_indexes[out_k_axis[1]] = tmp_out_indexes[out_k_axis[1]]
                        + static_cast<uint64_t>(1);
                break;
            case 3:
                tmp_out_indexes[out_k_axis[1]] = tmp_out_indexes[out_k_axis[1]]
                        + static_cast<uint64_t>(1);
                tmp_out_indexes[out_n_axis[1]] = tmp_out_indexes[out_n_axis[1]]
                        + static_cast<uint64_t>(loop_step);
                break;
        }
    };

    for (auto i = 0; i < step; i++) {
        if (is_bf16) {
            MAKE_VAR_ASSIGN_VNNI(rows[i], bf16(bf16_step), row)
        } else {
            if (is_u8) {
                MAKE_VAR_ASSIGN_VNNI(rows[i], u8(u8_step), row)
            } else {
                MAKE_VAR_ASSIGN_VNNI(rows[i], s8(u8_step), row)
            }
        }

        // skip bf16 elimination pass on rows. Otherwise it will be promote
        // to f32.
        rows[i]->attr()["can_promote_to_f32"] = false;
        cur_list.emplace_back(builder::make_var_tensor_def_unattached(rows[i]));
    }
    if (!output_loop) {
        std::vector<expr> in_indexes, loop_indexes;
        for (size_t i = 0; i < input_blocking_dims.size(); i++) {
            iter_vars.emplace_back(range_from_outer_loop(src.get_ranges()[i])
                            ? expr(0)
                            : builder::make_var(datatypes::index,
                                    std::string("_fuseiter")
                                            + fusion_create_idx()));
            in_indexes.emplace_back((iter_vars[i] + src.get_offset()[i]));
            loop_indexes.emplace_back(iter_vars[i]);
        }
        expr condition;
        expr last_axis_offset, other_axis_condition;
        std::vector<expr> tmp_indexes
                = get_reorder_block2plain_indexes(graph, in_indexes,
                        input_format, plain_dims, condition, last_axis_offset,
                        other_axis_condition, input_origin_axis_vectorized);
        std::vector<expr> out_indexes
                = get_reorder_plain2block_indexes(tmp_indexes, output_format);
        for (int i = 0; i < step; i++) {
            auto tmp_in_indexes = loop_indexes;
            if (!is_vnni_reorder) {
                tmp_in_indexes[inp_n_axis[0]] = tmp_in_indexes[inp_n_axis[0]]
                        + static_cast<uint64_t>(i);
            } else {
                tmp_in_indexes[inp_k_axis[0]] = tmp_in_indexes[inp_k_axis[0]]
                        + static_cast<uint64_t>(i);
            }
            stmt assign = builder::make_assign_unattached(rows[i],
                    // here, use src.tptr instead of input is aimed to
                    // avoid input is tensor_view_op. Otherwise, it will
                    // throw illegal exception in tensor_shrink
                    builder::make_indexing(src.tptr_, tmp_in_indexes,
                            is_bf16 ? bf16_step : u8_step));

            assign->attr()[op_traits::workload_computable_t::workload_number]
                    = wkld;
            cur_list.emplace_back(assign);
        }
        do_vnni_reorder(cur_list, rows, rows_dtype, is_vnni_reorder, bf16_step);

        for (int i = 0; i < step; i++) {
            auto tmp_out_indexes = out_indexes;
            if (!is_vnni_reorder) { // vnni transpose
                tmp_out_indexes[out_k_axis[1]] = tmp_out_indexes[out_k_axis[1]]
                        + static_cast<uint64_t>(i);
            } else {
                if (is_bf16) {
                    if (use_x16step) {
                        func_bf16_forloop_step(i, tmp_out_indexes, 8);
                    } else {
                        func_bf16_forloop_step(i, tmp_out_indexes, 4);
                    }
                } else {
                    tmp_out_indexes[out_n_axis[1]]
                            = tmp_out_indexes[out_n_axis[1]]
                            + static_cast<uint64_t>(i) * 4;
                }
            }
            stmt assign;

            assign = builder::make_assign_unattached(
                    builder::make_indexing(output, tmp_out_indexes,
                            is_bf16 ? bf16_step : u8_step),
                    rows[i]);
            cur_list.emplace_back(assign);
        }
        stmt cur = builder::make_stmts_unattached(cur_list);
        stmt body;
        expr_c iter_end;
        // for-loop-transforms only support step=1
        // we can only divide iter_end_ rather than multiply step_
        for (int i = static_cast<int>(input_blocking_dims.size()) - 1; i >= 0;
                i--) {
            // Do not generate those dummy loops
            if (!iter_vars.at(i).isa<var>()) continue;
            iter_end = src.get_shape()[i];
            expr cur_step = 1;
            if (!is_vnni_reorder) {
                if (i == inp_n_axis[0]) {
                    cur_step = 4;
                } else if (i == inp_k_axis[0]) {
                    cur_step = is_bf16 ? 8 : 16;
                }
            } else {
                if (i == inp_k_axis[0]) {
                    cur_step = 4;
                } else if (i == inp_n_axis[0]) {
                    cur_step = is_bf16 ? bf16_step : u8_step;
                }
            }
            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0), iter_end.remove_const(), cur_step, std::move(body),
                    true, for_type::NORMAL);
        }
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    } else { // use output loop
        std::vector<expr> out_indexes;
        // create iter variable, and make index
        for (size_t i = 0; i < output_blocking_dims.size(); i++) {
            iter_vars.emplace_back(range_from_outer_loop(dst.get_ranges()[i])
                            ? expr(0)
                            : builder::make_var(datatypes::index,
                                    std::string("_fuseiter")
                                            + fusion_create_idx()));
            out_indexes.emplace_back(iter_vars[i] + dst.get_offset()[i]);
        }

        // calculate the input index according to the output index
        expr condition, vnni_condition;
        expr last_axis_offset, other_axis_condition;
        std::vector<expr> tmp_indexes
                = get_reorder_block2plain_indexes(graph, out_indexes,
                        output_format, plain_dims, condition, last_axis_offset,
                        other_axis_condition, output_origin_axis_vectorized);
        std::vector<expr> in_indexes
                = get_reorder_plain2block_indexes(tmp_indexes, input_format);
        expr cur_step_var;
        stmt cur_step_var_assign;
        // load data to register
        for (int i = 0; i < step; i++) {
            auto tmp_in_indexes = in_indexes;
            if (!is_vnni_reorder) {
                tmp_in_indexes[inp_n_axis[0]] = tmp_in_indexes[inp_n_axis[0]]
                        + static_cast<uint64_t>(i);
            } else {
                tmp_in_indexes[inp_k_axis[0]] = tmp_in_indexes[inp_k_axis[0]]
                        + static_cast<uint64_t>(i);
            }
            expr mask;
            stmt mask_def;
            int real_step = is_bf16 ? bf16_step : u8_step;
            if (is_padding) {
                int tmp_in_last_dim
                        = is_vnni_reorder ? inp_n_axis[0] : inp_k_axis[0];
                int tmp_in_other_dim
                        = is_vnni_reorder ? inp_k_axis[0] : inp_n_axis[0];
                last_axis_offset
                        = cast_to_s32(
                                  input_blocking_shape_expr[tmp_in_last_dim])
                        - cast_to_s32(tmp_in_indexes[tmp_in_last_dim]);
                other_axis_condition = tmp_in_indexes[tmp_in_other_dim]
                        < input_blocking_shape_expr[tmp_in_other_dim];
                // The cur_step corresponding to each step is the same,
                // so we only need to count the first time and others
                // can be reused.
                if (i == 0) {
                    // mask = min(max(0, last_dim_len -
                    // last_dim_idx),real_step) To choose [0 ~
                    // step] mask
                    auto cur_step = builder::make_min(
                            builder::make_max(builder::make_constant(0),
                                    last_axis_offset),
                            real_step);
                    cur_step_var = builder::make_var(
                            sc_data_type_t::s32(1), "cur_step_var");
                    cur_step_var_assign
                            = builder::make_var_tensor_def_unattached(
                                    cur_step_var, linkage::local, cur_step);
                    cur_list.emplace_back(cur_step_var_assign);
                }
                // mask = other_dims_condition ? mask : 0;
                mask = generate_mask_var_by_step(mask_def, cur_step_var,
                        real_step, other_axis_condition);
                cur_list.emplace_back(mask_def);
            }

            // here, we use input is to fix the out-of-bound exception.
            // Using tptr will make the address calculation wrong in
            // some cases (479x1024..).
            auto assign = builder::make_assign_unattached(rows[i],
                    builder::make_indexing(input, tmp_in_indexes,
                            is_bf16 ? bf16_step : u8_step, mask));
            assign->attr()[op_traits::workload_computable_t::workload_number]
                    = wkld;
            cur_list.emplace_back(assign);
        }

        do_vnni_reorder(cur_list, rows, rows_dtype, is_vnni_reorder, bf16_step);

        // store data from register
        for (int i = 0; i < step; i++) {
            auto tmp_out_indexes = out_indexes;
            if (!is_vnni_reorder) { // vnni transpose
                tmp_out_indexes[out_k_axis[1]] = tmp_out_indexes[out_k_axis[1]]
                        + static_cast<uint64_t>(i);
            } else {
                if (is_bf16) {
                    if (use_x16step) {
                        func_bf16_forloop_step(i, tmp_out_indexes, 8);
                    } else {
                        func_bf16_forloop_step(i, tmp_out_indexes, 4);
                    }
                } else {
                    tmp_out_indexes[out_n_axis[1]]
                            = tmp_out_indexes[out_n_axis[1]]
                            + static_cast<uint64_t>(i) * 4;
                }
            }
            stmt assign = builder::make_assign_unattached(
                    builder::make_indexing(output, tmp_out_indexes,
                            is_bf16 ? bf16_step : u8_step),
                    rows[i]);
            cur_list.emplace_back(assign);
        }
        stmt cur = builder::make_stmts_unattached(cur_list);
        stmt body;

        expr iter_end;
        // for-loop-transforms only support step=1
        // we can only divide iter_end_ rather than multiply step_
        for (int i = static_cast<int>(output_blocking_dims_expr.size()) - 1;
                i >= 0; i--) {
            // Do not generate those dummy loops
            if (!iter_vars.at(i).isa<var>()) continue;
            // if the offset of dst is given(commit op)
            if (!is_padding || !dst.get_offset()[i].isa<constant>()) {
                iter_end = dst.get_shape()[i];
            } else {
                iter_end = output_blocking_dims_expr[i];
            }
            expr cur_step = 1;
            if (!is_vnni_reorder) { // vnni transpose
                if (i == out_n_axis[1] || i == out_k_axis[1]) {
                    cur_step = 4;
                } else if (i == out_k_axis[2]) {
                    cur_step = is_bf16 ? 2 : 4;
                }
            } else { // vnni reorder
                if (i == out_n_axis[1]) {
                    cur_step = is_bf16 ? bf16_step : u8_step;
                } else if (i == out_k_axis[2]
                        || (is_bf16 && i == out_k_axis[1])) {
                    cur_step = is_bf16 ? 2 : 4;
                }
            }

            body = cur.isa<stmts>() ? cur
                                    : make_stmt<stmts_node_t>(
                                            std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                    expr(0), iter_end.remove_const(), cur_step, std::move(body),
                    true, for_type::NORMAL);
        }
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    }
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
