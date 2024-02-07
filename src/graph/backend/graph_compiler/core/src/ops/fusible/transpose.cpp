/*******************************************************************************
 * Copyright 2023-2024 Intel Corporation
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

static sc_trans_kernel get_trans_kernel_type(const sc_data_etype &etype) {
    switch (etype) {
        case sc_data_etype::U16:
        case sc_data_etype::F16:
        case sc_data_etype::BF16: {
            return sc_trans_kernel::BIT16_32X8_TRANS;
        }; break;
        case sc_data_etype::F32: {
            return sc_trans_kernel::F32_8X8_TRANS;
        }; break;
        case sc_data_etype::U8: {
            return sc_trans_kernel::U8S8_16X16_TRANS;
        }; break;
        case sc_data_etype::S8: {
            return sc_trans_kernel::U8S8_16X16_TRANS;
        }; break;
        default: {
            COMPILE_ASSERT(false, "Do not support this dtype :" << etype);
        } break;
    }
    return sc_trans_kernel::NO_TRANS;
}

// currently only support f32 8x8 and bf16 32x8
const int trans_lanesx8 = 8;
const int trans_lanesx16 = 16;
const int trans_lanes_16bitx8 = 32;
// [..., a, ... , b] <=> [..., b, ..., a]
bool can_be_fast_transpose(const sc_graph_t &graph, const context_ptr &ctx,
        std::vector<int> &inp_a_axis, std::vector<int> &inp_b_axis,
        std::vector<int> &out_a_axis, std::vector<int> &out_b_axis,
        const sc_dims &plain_dims, const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, const tensor_slice &src,
        const tensor_slice &dst, const sc_data_type_t &dtype, bool is_dynamic,
        bool dynamic_no_padding, sc_trans_kernel &trans_kernel_used) {
    if (!ctx->machine_.cpu_flags_.fAVX2) {
        trans_kernel_used = sc_trans_kernel::NO_TRANS;
        return false;
    }
    if (!dtype.is_etype(sc_data_etype::F32)
            && !dtype.is_etype(sc_data_etype::BF16)
            && !dtype.is_etype(sc_data_etype::S8)
            && !dtype.is_etype(sc_data_etype::U8)) {
        trans_kernel_used = sc_trans_kernel::NO_TRANS;
        return false;
    }

    bool is_16bit = (utils::get_sizeof_etype(dtype.type_code_) * 8) == 16;
    bool is_float = dtype.is_etype(sc_data_etype::F32);
    bool is_s8u8 = dtype.is_etype(sc_data_etype::S8)
            || dtype.is_etype(sc_data_etype::U8);
    inp_a_axis.clear();
    inp_b_axis.clear();
    out_a_axis.clear();
    out_b_axis.clear();
    int trans_inp_a_axis = 0, trans_inp_b_axis = 0, trans_out_a_axis = 0,
        trans_out_b_axis = 0;
    int inp_idx = 0, out_idx = 0;
    auto &inp_code = input_format.format_code_;
    auto &out_code = output_format.format_code_;
    int input_ndims = input_format.format_code_.ndims();
    int output_ndims = output_format.format_code_.ndims();
    auto input_blocking_shapes
            = sc_data_format_t::get_blocking_shapes(plain_dims, input_format);
    auto output_blocking_shapes
            = sc_data_format_t::get_blocking_shapes(plain_dims, output_format);
    auto input_blocking_shapes_expr = get_blocking_shapes_expr(
            const_cast<sc_graph_t &>(graph), plain_dims, input_format);
    auto output_blocking_shapes_expr = get_blocking_shapes_expr(
            const_cast<sc_graph_t &>(graph), plain_dims, output_format);
    auto inp_b_idx = inp_code.get(input_ndims - 1);
    int vec_inp_lastdim = input_ndims - 1;
    auto out_a_idx = out_code.get(output_ndims - 1);
    int vec_out_lastdim = output_ndims - 1;
    find_vectorized_axis(src, input_format, inp_b_idx, vec_inp_lastdim);
    find_vectorized_axis(dst, output_format, out_a_idx, vec_out_lastdim);
    if (inp_b_idx == out_a_idx) {
        trans_kernel_used = sc_trans_kernel::NO_TRANS;
        return false;
    }
    while (inp_idx < input_ndims || out_idx < output_ndims) {
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
        if (orig_inp_idx != orig_out_idx) {
            trans_kernel_used = sc_trans_kernel::NO_TRANS;
            return false;
        }
        inp_idx++;
        out_idx++;
    }
    // number of non-transpose axis should be equal
    if (static_cast<size_t>(input_ndims) - inp_a_axis.size() - inp_b_axis.size()
            != static_cast<size_t>(output_ndims) - out_a_axis.size()
                    - out_b_axis.size()) {
        trans_kernel_used = sc_trans_kernel::NO_TRANS;
        return false;
    }
    auto find_another_vec_axis
            = [&](const sc_data_format_t &format, int &vec_axis,
                      const tensor_slice &tsl, const int ori_idx) {
                  auto format_code = format.format_code_;
                  auto all_dims = format_code.ndims();
                  for (int i = all_dims - 1; i >= 0; i--) {
                      if (!tsl.get_shape()[i].isa<constant>()) { break; }
                      if (format_code.get(i) == ori_idx
                              && get_expr_as_int(tsl.get_shape()[i]) > 1) {
                          vec_axis = i;
                          break;
                      }
                  }
              };
    // find last dim of input a axis length not 1
    int ori_inp_a = inp_code.get(inp_a_axis[inp_a_axis.size() - 1]);
    int ori_out_b = out_code.get(out_b_axis[out_b_axis.size() - 1]);
    int vec_inp_a = inp_a_axis[inp_a_axis.size() - 1];
    int vec_out_b = out_b_axis[out_b_axis.size() - 1];
    find_another_vec_axis(input_format, vec_inp_a, src, ori_inp_a);
    find_another_vec_axis(output_format, vec_out_b, dst, ori_out_b);
    trans_inp_a_axis = vec_inp_a;
    trans_inp_b_axis = vec_inp_lastdim;
    trans_out_a_axis = vec_out_lastdim;
    trans_out_b_axis = vec_out_b;
    // remove axis don't use
    auto delete_axis = [&](std::vector<int> &inp_axis, const int last) {
        // assume transpose axis is a and b, b1 dim len is 1
        auto itlast = std::find(inp_axis.begin(), inp_axis.end(), last);
        // [a, b, b1] -> [a, b]
        inp_axis.erase(++itlast, inp_axis.end());
    };
    delete_axis(inp_a_axis, vec_inp_a);
    delete_axis(inp_b_axis, vec_inp_lastdim);
    delete_axis(out_a_axis, vec_out_lastdim);
    delete_axis(out_b_axis, vec_out_b);
    bool is_dynamic_axis = !src.shape_[trans_inp_a_axis].isa<constant>()
            || !src.shape_[trans_inp_b_axis].isa<constant>()
            || !dst.shape_[trans_out_a_axis].isa<constant>()
            || !dst.shape_[trans_out_b_axis].isa<constant>();
    if (input_format.is_blocking() && output_format.is_blocking()) {
        // The transpose dimension in the output shape should be an integer
        // multiple of the input shape in block2block case.
        int ix = vec_inp_a;
        int iy = vec_inp_lastdim;
        int ox = vec_out_lastdim;
        int oy = vec_out_b;
        // can't do it in dynamic cases
        if (is_dynamic_axis) {
            trans_kernel_used = sc_trans_kernel::NO_TRANS;
            return false;
        }

        int inp_x = get_expr_as_int(src.shape_[ix]);
        int inp_y = get_expr_as_int(src.shape_[iy]);
        int out_x = get_expr_as_int(dst.shape_[ox]);
        int out_y = get_expr_as_int(dst.shape_[oy]);

        if (out_x % inp_x != 0 || out_y % inp_y != 0) {
            trans_kernel_used = sc_trans_kernel::NO_TRANS;
            return false;
        }
        // example: ABCD4c16d2c -> ACBD8c16d, Note that the
        // block format has been reconstructed in this example due to the
        // multiple blocks and is not suitable for transpose. The number of data
        // in the input and output format blocks should be the same, otherwise
        // the data position may cause errors due to format changes.
        auto input_block_axis = input_format.get_blocked_axis();
        auto output_block_axis = output_format.get_blocked_axis();
        size_t block_count_inp = input_block_axis.size();
        size_t block_count_out = output_block_axis.size();
        if (block_count_inp > 1 || block_count_out > 1) {
            for (auto iter = input_block_axis.begin();
                    iter != input_block_axis.end(); iter++) {
                if (iter->second[0] != output_block_axis[iter->first][0]) {
                    trans_kernel_used = sc_trans_kernel::NO_TRANS;
                    return false;
                }
            }
        }
    }
    auto satisfy_dim_lanes = [&]() {
        int trans_lanes1 = is_16bit ? trans_lanes_16bitx8
                : is_s8u8           ? trans_lanesx16
                                    : trans_lanesx8;
        int trans_lanes2 = is_s8u8 ? trans_lanesx16 : trans_lanesx8;
        return plain_dims[inp_b_idx] % trans_lanes2 == 0
                && plain_dims[out_a_idx] % trans_lanes1 == 0
                && get_expr_as_int(src.shape_[vec_inp_a]) % trans_lanes1 == 0
                && get_expr_as_int(dst.shape_[vec_out_b]) % trans_lanes2 == 0
                && get_expr_as_int(src.shape_[vec_inp_lastdim]) % trans_lanes2
                == 0
                && get_expr_as_int(dst.shape_[vec_out_lastdim]) % trans_lanes1
                == 0;
    };
    auto meet_kernel_require = [&](int threshold) {
        int total = threshold;
        return get_expr_as_int(src.shape_[vec_inp_a])
                        * get_expr_as_int(src.shape_[vec_inp_lastdim])
                >= total
                && get_expr_as_int(dst.shape_[vec_out_lastdim])
                        * get_expr_as_int(dst.shape_[vec_out_b])
                >= total;
    };

    // Currently bf16 calculation needs to be larger than
    // the number of elements threshold, otherwise the performance may
    // regression.
    // Multithread threashold is 35 - 88 (threads 56 - 4). We just div 6
    // directly in u8s8 and div 3 in bf16.
    int bit16_threshold = trans_lanesx8 * trans_lanes_16bitx8 / 3;
    int s8u8_threshold = trans_lanesx16 * trans_lanesx16 / 6;
    int f32_threshold = 1;
    auto cur_run_thread = runtime_config_t::get().get_num_threads();
    // Single threashold is 128.
    if (cur_run_thread == 1) {
        bit16_threshold = trans_lanesx8 * trans_lanes_16bitx8 / 2;
        s8u8_threshold = trans_lanesx16 * trans_lanesx16 / 2;
    }
    // transpose axis should not be dynamic
    if (is_16bit
            && (dynamic_no_padding
                    || (!is_dynamic
                            && !meet_kernel_require(bit16_threshold)))) {
        trans_kernel_used = sc_trans_kernel::NO_TRANS;
        return false;
    } else if (is_s8u8
            && (dynamic_no_padding
                    || (!is_dynamic && !meet_kernel_require(s8u8_threshold)))) {
        trans_kernel_used = sc_trans_kernel::NO_TRANS;
        return false;
    } else if (is_float
            && (!is_dynamic && !meet_kernel_require(f32_threshold))) {
        trans_kernel_used = sc_trans_kernel::NO_TRANS;
        return false;
    }

    // According to the current experimental results, when the number of
    // shape elements is about 200 times the size of the L1cache, the
    // performance will decline due to the drop in the L1cache hit rate.
    // test shape example: [56, 256, 56, 56], [56, 64, 56, 56], [56, 256,
    // 31, 22]
    // Note the shape sizes that appear around the threshold.
    if (!is_dynamic) {
        auto shape_number = math_utils::get_dims_product(input_blocking_shapes)
                * utils::get_sizeof_etype(dtype.type_code_);
        int cache_multiplier = 200;
        auto buffer_size_threshold
                = ctx->machine_.cpu_flags_.getDCacheSize(1) * cache_multiplier;
        // It seems that in the case of multi-threading, the threshold
        // setting is similar.
        if (cur_run_thread == 1) {
            cache_multiplier = 125;
            buffer_size_threshold = ctx->machine_.cpu_flags_.getDCacheSize(1)
                    * cache_multiplier;
        }
        if (!whole_buffer_reorder(src)
                && ((!satisfy_dim_lanes()
                        && ((uint64_t)shape_number > buffer_size_threshold)))) {
            return false;
        }
    } else {
        // currently does not support tensor slice in dynamic
        if (!whole_buffer_reorder(src)) { return false; }
    }

    trans_kernel_used = get_trans_kernel_type(dtype.as_etype());

    if (is_float && ctx->machine_.cpu_flags_.fAVX512F
            && plain_dims[inp_b_idx] > trans_lanesx8
            && plain_dims[out_a_idx] > trans_lanesx8) {
        // Currently we don't use f32x16 kernel (keep using f32x8, due to avx512
        // frequency problem).But we keep it for the convenience of future
        // performance comparison test on new machines.
        trans_kernel_used = sc_trans_kernel::F32_8X8_TRANS;
    }
    if (is_16bit && !ctx->machine_.cpu_flags_.fAVX512F
            && ctx->machine_.cpu_flags_.fAVX2) {
        trans_kernel_used = sc_trans_kernel::BIT16_16X16_TRANS;
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

#define TRANS2D_REG_CALCULATION_BIT16x16(type_bits) \
    for (int iter_i = 0; iter_i < 16; iter_i += 2) { \
        TRANS2D_UNPACK_ASSIGN(low, iter_i + 17, iter_i + 1, iter_i + 2, 16) \
        TRANS2D_UNPACK_ASSIGN(high, iter_i + 18, iter_i + 1, iter_i + 2, 16) \
    } \
    for (int iter_i = 0; iter_i < 16; iter_i += 4) { \
        TRANS2D_UNPACK_ASSIGN(low, iter_i + 1, iter_i + 17, iter_i + 19, 32) \
        TRANS2D_UNPACK_ASSIGN(high, iter_i + 2, iter_i + 17, iter_i + 19, 32) \
        TRANS2D_UNPACK_ASSIGN(low, iter_i + 3, iter_i + 18, iter_i + 20, 32) \
        TRANS2D_UNPACK_ASSIGN(high, iter_i + 4, iter_i + 18, iter_i + 20, 32) \
    } \
    for (int iter_i = 0; iter_i < 16; iter_i += 8) { \
        for (int iter_j = 0, offset = 0; iter_j < 8; iter_j += 2, offset++) { \
            TRANS2D_UNPACK_ASSIGN(low, iter_i + 17 + iter_j, \
                    iter_i + offset + 1, iter_i + offset + 5, 64) \
            TRANS2D_UNPACK_ASSIGN(high, iter_i + 18 + iter_j, \
                    iter_i + offset + 1, iter_i + offset + 5, 64) \
        } \
    } \
    for (int iter_i = 0; iter_i < 8; iter_i += 1) { \
        TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32( \
                permute, iter_i + 1, iter_i + 17, iter_i + 25, 32, type_bits) \
    } \
    for (int iter_i = 8; iter_i < 16; iter_i += 1) { \
        TRANS2D_SHUFFLE_PERMUTE_ASSIGN_F32(permute, iter_i + 1, \
                iter_i + 17 - 8, iter_i + 25 - 8, 49, type_bits) \
    }

#define TRANS2D_REG_CALCULATION_BIT16() \
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

#define TRANS2D_REG_CALCULATION_U8S8X16() \
    for (int iter_i = 0; iter_i < 16; iter_i += 2) { \
        TRANS2D_UNPACK_ASSIGN(low, iter_i + 17, iter_i + 1, iter_i + 2, 8) \
        TRANS2D_UNPACK_ASSIGN(high, iter_i + 18, iter_i + 1, iter_i + 2, 8) \
    } \
    for (int iter_i = 0; iter_i < 16; iter_i += 4) { \
        TRANS2D_UNPACK_ASSIGN(low, iter_i + 1, iter_i + 17, iter_i + 19, 16) \
        TRANS2D_UNPACK_ASSIGN(high, iter_i + 2, iter_i + 17, iter_i + 19, 16) \
        TRANS2D_UNPACK_ASSIGN(low, iter_i + 3, iter_i + 18, iter_i + 20, 16) \
        TRANS2D_UNPACK_ASSIGN(high, iter_i + 4, iter_i + 18, iter_i + 20, 16) \
    } \
    for (int iter_i = 0; iter_i < 16; iter_i += 8) { \
        for (int iter_j = 0, offset = 0; iter_j < 8; iter_j += 2, offset++) { \
            TRANS2D_UNPACK_ASSIGN(low, iter_i + 17 + iter_j, \
                    iter_i + offset + 1, iter_i + offset + 5, 32) \
            TRANS2D_UNPACK_ASSIGN(high, iter_i + 18 + iter_j, \
                    iter_i + offset + 1, iter_i + offset + 5, 32) \
        } \
    } \
    for (int iter_i = 0, offset = 0; iter_i < 16; iter_i += 2, offset++) { \
        TRANS2D_UNPACK_ASSIGN(low, iter_i + 1, offset + 17, offset + 25, 64) \
        TRANS2D_UNPACK_ASSIGN(high, iter_i + 2, offset + 17, offset + 25, 64) \
    }

void compute_fast_transpose(sc_graph_t &graph, const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        const std::vector<int> &vec_inp_a_axis,
        const std::vector<int> &vec_inp_b_axis,
        const std::vector<int> &vec_out_a_axis,
        const std::vector<int> &vec_out_b_axis,
        const graph_tensor_ptr &expand_gt, size_t wkld, bool is_dynamic,
        bool dynamic_no_padding, const sc_trans_kernel trans_kernel_used) {
    auto input = src.get_real_tensor();
    auto output = dst.get_real_tensor();
    bool is_s8u8 = dtype.is_etype(sc_data_etype::S8)
            || dtype.is_etype(sc_data_etype::U8);
    bool is_16bit = (utils::get_sizeof_etype(dtype.type_code_) * 8) == 16;
    int step = (sc_trans_kernel::F32_16X16_TRANS == trans_kernel_used
                       && dtype == datatypes::f32)
                    || is_s8u8
            ? trans_lanesx16
            : trans_lanesx8; // fixed f32x8
    const int lanesx16 = 16;
    bool direct_use_bit16x16 = is_16bit
            && trans_kernel_used == sc_trans_kernel::BIT16_16X16_TRANS;
    step = direct_use_bit16x16 ? lanesx16 : step;
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
    const int inp_a_axis = vec_inp_a_axis[vec_inp_a_axis.size() - 1];
    const int inp_b_axis = vec_inp_b_axis[vec_inp_b_axis.size() - 1];
    const int out_a_axis = vec_out_a_axis[vec_out_a_axis.size() - 1];
    const int out_b_axis = vec_out_b_axis[vec_out_b_axis.size() - 1];
    std::vector<expr> rows;
    std::vector<expr> iter_vars, iter_vars_tail;
    std::vector<stmt_c> cur_list, cur_list_floor, var_define_list,
            cur_list_lastdim_floor;
    stmt cur, body, cur_tail, body_tail;
    bool need_mask = false;
    int inp_a_step = dtype == datatypes::f32
            ? (sc_trans_kernel::F32_16X16_TRANS == trans_kernel_used
                            ? trans_lanesx16
                            : trans_lanesx8)
            : direct_use_bit16x16 ? lanesx16
                                  : trans_lanes_16bitx8;
    inp_a_step = is_s8u8 ? trans_lanesx16 : inp_a_step;
    int inp_b_step = (sc_trans_kernel::F32_16X16_TRANS == trans_kernel_used
                             && dtype == datatypes::f32)
                    || is_s8u8
            ? trans_lanesx16
            : direct_use_bit16x16 ? lanesx16
                                  : trans_lanesx8;
    const int type_bits = utils::get_sizeof_type(sc_data_type_t::f32(4)) * 8;
    expr inputloop_otheraxis_condition, outputloop_axis_condition,
            inputloop_lastdim_condition;
    bool use_split_loop = true, is_padding = false;
    auto is_shapesize_need_mask = [&](const int64_t inp_a_size,
                                          const int64_t inp_b_size,
                                          const int64_t out_a_size,
                                          const int64_t out_b_size) {
        assert(!is_dynamic);
        if ((!is_dynamic
                    && (inp_a_size % inp_a_step || inp_b_size % inp_b_step
                            || out_a_size % inp_a_step
                            || out_b_size % inp_b_step))) {
            need_mask = true;
        }
    };
    if (is_dynamic && !dynamic_no_padding) { need_mask = true; }
    if (!is_dynamic) {
        is_shapesize_need_mask(input_blocking_dims[inp_a_axis],
                input_blocking_dims[inp_b_axis],
                output_blocking_dims[out_a_axis],
                output_blocking_dims[out_b_axis]);
        // tensor slice check
        is_shapesize_need_mask(get_expr_as_int(src.get_shape()[inp_a_axis]),
                get_expr_as_int(src.get_shape()[inp_b_axis]),
                get_expr_as_int(dst.get_shape()[out_a_axis]),
                get_expr_as_int(dst.get_shape()[out_b_axis]));
    }

    if (!is_dynamic
            && math_utils::get_dims_product(input_blocking_dims)
                    != math_utils::get_dims_product(output_blocking_dims)) {
        need_mask = true;
        is_padding = true;
    }

    auto make_rows_var = [&](std::vector<expr> &rows,
                                 std::vector<stmt_c> &cur_list,
                                 const sc_data_type_t &dtype, const int var_len,
                                 bool skip_promote = true) {
        rows.resize(var_len); // bf16 uses 16 zmms.
        for (auto i = 0; i < var_len; i++) {
            rows[i] = builder::make_var(dtype,
                    "row" + std::to_string(i + 1) + fusion_create_var_idx());
            cur_list.emplace_back(
                    builder::make_var_tensor_def_unattached(rows[i]));
            if (skip_promote) {
                // skip bf16 elimination pass on rows. Otherwise it will be
                // promote to f32.
                rows[i]->attr()["can_promote_to_f32"] = false;
            }
        }
    };
    if (dtype == datatypes::f32 || is_s8u8 || direct_use_bit16x16) {
        int var_len = is_s8u8 ? step * 2 : step + step / 2;
        var_len = direct_use_bit16x16 ? 32 : var_len;
        sc_data_type_t cur_dtype = sc_data_type_t(dtype.type_code_, step);
        make_rows_var(rows, cur_list, cur_dtype, var_len);
    } else if (is_16bit) {
        const int var_len = 16;
        auto cur_dtype = sc_data_type_t::bf16(32);
        make_rows_var(rows, cur_list, cur_dtype, var_len, false);
    }
    var_define_list.assign(cur_list.begin(), cur_list.end());

    auto determine_cur_step = [&](const std::vector<expr> &blocking_dims_expr,
                                      const std::vector<expr> &tmp_in_indexes,
                                      const std::vector<expr> &plain_indexes,
                                      expr &cur_step, expr &sup_condition,
                                      int in_axis, int in_baxis,
                                      bool use_output_loop, int step,
                                      sc_data_format_kind_t &format_code,
                                      bool dst_condition = false) {
        auto slice_condition = [&]() {
            cur_step = builder::make_min(
                    builder::make_max(cast_to_s32(blocking_dims_expr[in_baxis])
                                    - cast_to_s32(tmp_in_indexes[in_baxis]),
                            0),
                    step);
            sup_condition
                    = tmp_in_indexes[in_axis] >= blocking_dims_expr[in_axis];
        };
        auto plain_condition = [&]() {
            auto tmp_plain = graph.dims_to_expr(plain_dims);
            auto input_last_dim = format_code.get(in_baxis);
            auto input_other_dim = format_code.get(in_axis);
            cur_step = builder::make_min(
                    builder::make_max(cast_to_s32(tmp_plain[input_last_dim])
                                    - cast_to_s32(
                                            plain_indexes[input_last_dim]),
                            0),
                    step);
            sup_condition = plain_indexes[input_other_dim]
                    >= tmp_plain[input_other_dim];
        };
        if (!use_output_loop || dst_condition) {
            slice_condition();
        } else {
            plain_condition();
        }
    };

    auto src_opt_mask = [&](expr &cur_step, expr &sup_condition,
                                const int step) {
        auto src_shape
                = output_loop ? input_blocking_dims_expr : src.get_shape();
        auto src_mask_can_empty = [&](const int src_mask_axis,
                                          const int dst_mask_axis,
                                          const int mask_step) {
            if (is_dynamic) return false;
            bool is_srcshape_multiple_step
                    = (get_expr_as_int(src_shape[src_mask_axis]) % mask_step)
                    == 0;
            bool is_dstshape_multiple_step = (output_loop
                    && (get_expr_as_int(dst.get_shape()[dst_mask_axis])
                               % mask_step)
                            == 0
                    && !is_padding);
            return !is_dynamic && (is_srcshape_multiple_step)
                    && (!output_loop || is_dstshape_multiple_step);
        };
        if (src_mask_can_empty(inp_a_axis, out_a_axis, inp_a_step)) {
            sup_condition = expr();
        } else {
            inputloop_otheraxis_condition = sup_condition;
        }
        if (src_mask_can_empty(inp_b_axis, out_b_axis, inp_b_step)) {
            cur_step = step;
            inputloop_lastdim_condition = expr();
        }
    };
    auto dst_opt_mask = [&](expr &cur_step, expr &sup_condition,
                                const std::vector<expr> &in_indexes,
                                const int i, const int step) {
        auto dst_shape
                = output_loop ? dst.get_shape() : output_blocking_dims_expr;
        auto dst_mask_can_empty = [&](const int dst_mask_axis,
                                          const int src_mask_axis,
                                          const int mask_step) {
            if (is_dynamic) return false;
            bool is_dstshape_multiple_step
                    = (get_expr_as_int(dst_shape[dst_mask_axis]) % mask_step)
                    == 0;
            bool is_srcshape_multiple_step = (!output_loop
                    && (get_expr_as_int(src.get_shape()[src_mask_axis])
                               % mask_step)
                            == 0
                    && !is_padding);
            return !is_dynamic && (is_dstshape_multiple_step)
                    && (output_loop || is_srcshape_multiple_step);
        };
        if (dst_mask_can_empty(out_b_axis, inp_b_axis, inp_b_step)) {
            sup_condition = expr();
        } else {
            outputloop_axis_condition = sup_condition;
        }
        // ABba(4, 4, 16, 4) => AB(16,64)
        if (input_format.is_blocking() || !output_loop) {
            auto src_condition_shape = output_loop
                    ? input_blocking_dims_expr[inp_b_axis]
                    : src.get_shape()[inp_b_axis];
            expr outof_srcshape_bound
                    = in_indexes[inp_b_axis] + i >= src_condition_shape;
            if (is_dynamic) {
                sup_condition = sup_condition || (outof_srcshape_bound);
            } else {
                if ((get_expr_as_int(src_condition_shape) % inp_b_step) != 0) {
                    if (sup_condition.defined()) {
                        sup_condition = sup_condition || (outof_srcshape_bound);
                    } else {
                        sup_condition = (outof_srcshape_bound);
                    }
                    outputloop_axis_condition = sup_condition;
                }
            }
        }
        if (dst_mask_can_empty(out_a_axis, inp_a_axis, inp_a_step)) {
            cur_step = step;
        }
    };
    auto inputloop_split_ir_directly = [&](const std::vector<expr> &in_indexes,
                                               const std::vector<expr>
                                                       &out_indexes,
                                               const std::vector<expr>
                                                       &plain_indexes,
                                               const std::vector<expr>
                                                       &out_indexes_slice) {
        assert(!is_dynamic);
        expr mask, mask_floor, mask_floor_lastdim;
        int lastdim_bound = get_expr_as_int(src.get_shape()[inp_b_axis]) % step;
        for (int i = 0; i < (direct_use_bit16x16 ? lanesx16 : step); i++) {
            auto tmp_in_indexes = in_indexes;
            auto in_axis = inp_a_axis;
            tmp_in_indexes[in_axis]
                    = tmp_in_indexes[in_axis] + static_cast<uint64_t>(i);
            auto tmp_plain_indexes = plain_indexes;
            tmp_plain_indexes[input_format_code.get(in_axis)]
                    = tmp_plain_indexes[input_format_code.get(in_axis)]
                    + static_cast<uint64_t>(i);
            expr tmp_in = src.tptr_;
            if (output_loop) { tmp_in = input; }
            if (need_mask) {
                expr cur_step, sup_condition;
                determine_cur_step(output_loop ? input_blocking_dims_expr
                                               : src.get_shape(),
                        tmp_in_indexes, tmp_plain_indexes, cur_step,
                        sup_condition, in_axis, inp_b_axis, output_loop, step,
                        input_format_code);
                src_opt_mask(cur_step, sup_condition, step);
                if (i == 0 && (lastdim_bound != 0)) {
                    inputloop_lastdim_condition
                            = cast_to_s32(src.get_shape()[inp_b_axis])
                                    - cast_to_s32(step)
                                    - cast_to_s32(tmp_in_indexes[inp_b_axis])
                            < 0;
                }

                stmt mask_def, mask_def_floor;
                mask = generate_mask_var_by_step(
                        mask_def, cur_step, step, sup_condition, true);
                cur_list.emplace_back(mask_def);
                // if sup_condition is defined, means this axis can
                // split loop to avoid other axis mask
                if (inputloop_otheraxis_condition.defined()) {
                    mask_floor = generate_mask_var_by_step(
                            mask_def_floor, cur_step, step, expr(), true);
                    cur_list_floor.emplace_back(mask_def_floor);
                }
            }

            auto assign = builder::make_assign_unattached(rows[i],
                    builder::make_indexing(tmp_in, tmp_in_indexes, step, mask));
            assign->attr()[op_traits::workload_computable_t::workload_number]
                    = wkld;
            cur_list.emplace_back(assign);
            if (inputloop_otheraxis_condition.defined()) {
                auto assign_floor = builder::make_assign_unattached(rows[i],
                        builder::make_indexing(
                                tmp_in, tmp_in_indexes, step, mask_floor));
                assign_floor->attr()
                        [op_traits::workload_computable_t::workload_number]
                        = wkld;
                cur_list_floor.emplace_back(assign_floor);
            }
            if (inputloop_lastdim_condition.defined()) {
                stmt assign_floor_lastdim;
                assign_floor_lastdim = builder::make_assign_unattached(rows[i],
                        builder::make_indexing(tmp_in, tmp_in_indexes, step));
                assign_floor_lastdim->attr()
                        [op_traits::workload_computable_t::workload_number]
                        = wkld;
                cur_list_lastdim_floor.emplace_back(assign_floor_lastdim);
            }
        }
        auto tmp_cur_list = cur_list;
        if (inputloop_otheraxis_condition.defined()
                || inputloop_lastdim_condition.defined()) {
            cur_list.clear();
        }
        if (is_s8u8) {
            TRANS2D_REG_CALCULATION_U8S8X16();
        } else if (direct_use_bit16x16) {
            TRANS2D_REG_CALCULATION_BIT16x16(type_bits);
        } else {
            if (sc_trans_kernel::F32_16X16_TRANS == trans_kernel_used) {
                TRANS2D_REG_CALCULATION_F32(type_bits);
            } else {
                TRANS2D_REG_CALCULATION_F32(type_bits);
            }
        }
        if (inputloop_otheraxis_condition.defined()
                || inputloop_lastdim_condition.defined()) {
            cur_list_floor.insert(
                    cur_list_floor.end(), cur_list.begin(), cur_list.end());
            cur_list_lastdim_floor.insert(cur_list_lastdim_floor.end(),
                    cur_list.begin(), cur_list.end());
            tmp_cur_list.insert(
                    tmp_cur_list.end(), cur_list.begin(), cur_list.end());
            cur_list.clear();
            cur_list = std::move(tmp_cur_list);
        }
        for (int i = 0; i < (direct_use_bit16x16 ? lanesx16 : step); i++) {
            auto tmp_out_indexes = out_indexes;
            auto out_axis = out_b_axis;
            tmp_out_indexes[out_axis]
                    = tmp_out_indexes[out_axis] + static_cast<uint64_t>(i);
            auto tmp_out_indexes_slice
                    = output_loop ? out_indexes_slice : out_indexes;
            tmp_out_indexes_slice[out_axis] = tmp_out_indexes_slice[out_axis]
                    + static_cast<uint64_t>(i);
            if (need_mask) {
                expr cur_step, sup_condition;
                determine_cur_step(output_loop ? dst.get_shape()
                                               : output_blocking_dims_expr,
                        tmp_out_indexes_slice, plain_indexes, cur_step,
                        sup_condition, out_axis, out_a_axis, output_loop, step,
                        input_format_code, true);
                dst_opt_mask(cur_step, sup_condition, in_indexes, i, step);
                stmt mask_def, mask_def_floor, mask_def_floor_lastdim;
                mask = generate_mask_var_by_step(
                        mask_def, cur_step, step, sup_condition, true);
                cur_list.emplace_back(mask_def);

                if (inputloop_otheraxis_condition.defined()) {
                    mask_floor = generate_mask_var_by_step(mask_def_floor,
                            cur_step, step, sup_condition, true);
                    cur_list_floor.emplace_back(mask_def_floor);
                }
                if (inputloop_lastdim_condition.defined()) {
                    expr sup_condition_lastdim = tmp_out_indexes_slice[out_axis]
                            >= output_blocking_dims_expr[out_axis];
                    mask_floor_lastdim = generate_mask_var_by_step(
                            mask_def_floor_lastdim, expr(step), step,
                            sup_condition_lastdim, true);
                    cur_list_lastdim_floor.emplace_back(mask_def_floor_lastdim);
                }
            }
            stmt assign;
            assign = builder::make_assign_unattached(
                    builder::make_indexing(output, tmp_out_indexes, step, mask),
                    rows[i]);

            cur_list.emplace_back(assign);
            if (inputloop_otheraxis_condition.defined()) {
                auto assign_floor = builder::make_assign_unattached(
                        builder::make_indexing(
                                output, tmp_out_indexes, step, mask_floor),
                        rows[i]);
                cur_list_floor.emplace_back(assign_floor);
            }
            if (inputloop_lastdim_condition.defined()) {
                stmt assign_floor_lastdim;
                assign_floor_lastdim = builder::make_assign_unattached(
                        builder::make_indexing(output, tmp_out_indexes, step,
                                mask_floor_lastdim),
                        rows[i]);

                cur_list_lastdim_floor.emplace_back(assign_floor_lastdim);
            }
        }
    };

    auto outputloop_split_ir_directly = [&](const std::vector<expr> &in_indexes,
                                                const std::vector<expr>
                                                        &out_indexes,
                                                const std::vector<expr>
                                                        &plain_indexes,
                                                const std::vector<expr>
                                                        &out_indexes_slice) {
        assert(cur_list_floor.empty() && cur_list.empty());
        expr mask, mask_floor;
        for (int i = 0; i < (direct_use_bit16x16 ? lanesx16 : step); i++) {
            auto tmp_in_indexes = in_indexes;
            auto in_axis = inp_a_axis;
            tmp_in_indexes[in_axis]
                    = tmp_in_indexes[in_axis] + static_cast<uint64_t>(i);
            auto tmp_plain_indexes = plain_indexes;
            tmp_plain_indexes[input_format_code.get(in_axis)]
                    = tmp_plain_indexes[input_format_code.get(in_axis)]
                    + static_cast<uint64_t>(i);
            expr tmp_in = src.tptr_;
            if (output_loop) { tmp_in = input; }
            if (need_mask) {
                expr cur_step, sup_condition;
                determine_cur_step(output_loop ? input_blocking_dims_expr
                                               : src.get_shape(),
                        tmp_in_indexes, tmp_plain_indexes, cur_step,
                        sup_condition, in_axis, inp_b_axis, output_loop, step,
                        input_format_code);
                src_opt_mask(cur_step, sup_condition, step);
                stmt mask_def, mask_def_floor;
                mask = generate_mask_var_by_step(
                        mask_def, cur_step, step, sup_condition, true);
                cur_list.emplace_back(mask_def);
                mask_floor = generate_mask_var_by_step(
                        mask_def_floor, cur_step, step, sup_condition, true);
                cur_list_floor.emplace_back(mask_def_floor);
            }

            auto assign = builder::make_assign_unattached(rows[i],
                    builder::make_indexing(tmp_in, tmp_in_indexes, step, mask));
            assign->attr()[op_traits::workload_computable_t::workload_number]
                    = wkld;
            cur_list.emplace_back(assign);
            auto assign_floor = builder::make_assign_unattached(rows[i],
                    builder::make_indexing(
                            tmp_in, tmp_in_indexes, step, mask_floor));
            assign_floor
                    ->attr()[op_traits::workload_computable_t::workload_number]
                    = wkld;
            cur_list_floor.emplace_back(assign_floor);
        }
        auto tmp_cur_list = cur_list;
        if (!cur_list_floor.empty()) { cur_list.clear(); }
        if (is_s8u8) {
            TRANS2D_REG_CALCULATION_U8S8X16();
        } else if (direct_use_bit16x16) {
            TRANS2D_REG_CALCULATION_BIT16x16(type_bits);
        } else {
            if (sc_trans_kernel::F32_16X16_TRANS == trans_kernel_used) {
                TRANS2D_REG_CALCULATION_F32(type_bits);
            } else {
                TRANS2D_REG_CALCULATION_F32(type_bits);
            }
        }
        if (!cur_list_floor.empty()) {
            cur_list_floor.insert(
                    cur_list_floor.end(), cur_list.begin(), cur_list.end());
            tmp_cur_list.insert(
                    tmp_cur_list.end(), cur_list.begin(), cur_list.end());
            cur_list.clear();
            cur_list = std::move(tmp_cur_list);
        }
        for (int i = 0; i < (direct_use_bit16x16 ? lanesx16 : step); i++) {
            auto tmp_out_indexes = out_indexes;
            auto out_axis = out_b_axis;
            tmp_out_indexes[out_axis]
                    = tmp_out_indexes[out_axis] + static_cast<uint64_t>(i);
            auto tmp_out_indexes_slice
                    = output_loop ? out_indexes_slice : out_indexes;
            tmp_out_indexes_slice[out_axis] = tmp_out_indexes_slice[out_axis]
                    + static_cast<uint64_t>(i);
            if (need_mask) {
                expr cur_step, sup_condition;
                determine_cur_step(output_loop ? dst.get_shape()
                                               : output_blocking_dims_expr,
                        tmp_out_indexes_slice, plain_indexes, cur_step,
                        sup_condition, out_axis, out_a_axis, output_loop, step,
                        input_format_code, true);
                dst_opt_mask(cur_step, sup_condition, in_indexes, i, step);
                stmt mask_def, mask_def_floor;
                mask = generate_mask_var_by_step(
                        mask_def, cur_step, step, sup_condition, true);
                cur_list.emplace_back(mask_def);
                if (outputloop_axis_condition.defined()) {
                    mask_floor = generate_mask_var_by_step(
                            mask_def_floor, cur_step, step, expr(), true);
                    cur_list_floor.emplace_back(mask_def_floor);
                }
            }
            auto assign = builder::make_assign_unattached(
                    builder::make_indexing(output, tmp_out_indexes, step, mask),
                    rows[i]);
            cur_list.emplace_back(assign);
            if (outputloop_axis_condition.defined()) {
                auto assign_floor = builder::make_assign_unattached(
                        builder::make_indexing(
                                output, tmp_out_indexes, step, mask_floor),
                        rows[i]);
                cur_list_floor.emplace_back(assign_floor);
            }
        }
    };

    auto inputloop_split_ir_bit16 = [&](const std::vector<expr> &in_indexes,
                                            const std::vector<expr>
                                                    &out_indexes,
                                            const std::vector<expr>
                                                    &plain_indexes,
                                            const std::vector<expr>
                                                    &out_indexes_slice) {
        expr mask, mask_floor;
        for (int i = 0; i < step; i++) {
            for (int p = 0; p < 4; p++) {
                auto tmp_in_indexes = in_indexes;
                auto in_axis = inp_a_axis;
                tmp_in_indexes[in_axis] = tmp_in_indexes[in_axis]
                        + (static_cast<uint64_t>(i) + p * 8);
                auto tmp_plain_indexes = plain_indexes;
                tmp_plain_indexes[input_format_code.get(in_axis)]
                        = tmp_plain_indexes[input_format_code.get(in_axis)]
                        + (static_cast<uint64_t>(i) + p * 8);
                expr tmp_in = src.tptr_;
                if (output_loop) { tmp_in = input; }
                if (need_mask) {
                    expr cur_step, sup_condition;
                    determine_cur_step(output_loop ? input_blocking_dims_expr
                                                   : src.get_shape(),
                            tmp_in_indexes, tmp_plain_indexes, cur_step,
                            sup_condition, in_axis, inp_b_axis, output_loop,
                            trans_lanesx8, input_format_code);
                    src_opt_mask(cur_step, sup_condition, trans_lanesx8);
                    stmt mask_def, mask_def_tail;
                    mask = generate_mask_var_by_step(mask_def, cur_step,
                            trans_lanesx8, sup_condition, true);
                    cur_list.emplace_back(mask_def);
                    if (inputloop_otheraxis_condition.defined()) {
                        mask_floor = generate_mask_var_by_step(
                                mask_def_tail, cur_step, step, expr(), true);
                        cur_list_floor.emplace_back(mask_def_tail);
                    }
                }
                auto brct_src = builder::make_broadcast(
                        builder::make_indexing(
                                tmp_in, tmp_in_indexes, step, mask),
                        trans_lanes_16bitx8);
                auto assign = builder::make_assign_unattached(rows[i],
                        p > 0 ? builder::make_select(
                                0xff << (p * step), brct_src, rows[i])
                              : brct_src);
                assign->attr()
                        [op_traits::workload_computable_t::workload_number]
                        = wkld;
                cur_list.emplace_back(assign);
                if (inputloop_otheraxis_condition.defined()) {
                    auto brct_src_floor = builder::make_broadcast(
                            builder::make_indexing(
                                    tmp_in, tmp_in_indexes, step, mask_floor),
                            trans_lanes_16bitx8);
                    auto assign_floor = builder::make_assign_unattached(rows[i],
                            p > 0 ? builder::make_select(
                                    0xff << (p * step), brct_src_floor, rows[i])
                                  : brct_src_floor);
                    assign_floor->attr()
                            [op_traits::workload_computable_t::workload_number]
                            = wkld;
                    cur_list_floor.emplace_back(assign_floor);
                }
            }
        }
        auto tmp_cur_list = cur_list;
        if (inputloop_otheraxis_condition.defined()) { cur_list.clear(); }
        TRANS2D_REG_CALCULATION_BIT16();
        if (inputloop_otheraxis_condition.defined()) {
            cur_list_floor.insert(
                    cur_list_floor.end(), cur_list.begin(), cur_list.end());
            tmp_cur_list.insert(
                    tmp_cur_list.end(), cur_list.begin(), cur_list.end());
            cur_list.clear();
            cur_list = std::move(tmp_cur_list);
        }
        for (int i = 0; i < step; i++) {
            auto tmp_out_indexes = out_indexes;
            auto out_axis = out_b_axis;
            tmp_out_indexes[out_axis]
                    = tmp_out_indexes[out_axis] + static_cast<uint64_t>(i);
            auto tmp_out_indexes_slice
                    = output_loop ? out_indexes_slice : out_indexes;
            tmp_out_indexes_slice[out_axis] = tmp_out_indexes_slice[out_axis]
                    + static_cast<uint64_t>(i);
            if (need_mask) {
                expr cur_step, sup_condition;
                determine_cur_step(output_loop ? dst.get_shape()
                                               : output_blocking_dims_expr,
                        tmp_out_indexes_slice, plain_indexes, cur_step,
                        sup_condition, out_axis, out_a_axis, output_loop,
                        trans_lanes_16bitx8, input_format_code, true);
                dst_opt_mask(cur_step, sup_condition, in_indexes, i,
                        trans_lanes_16bitx8);
                stmt mask_def, mask_def_floor;
                mask = generate_mask_var_by_step(mask_def, cur_step,
                        trans_lanes_16bitx8, sup_condition, true);
                cur_list.emplace_back(mask_def);
                if (inputloop_otheraxis_condition.defined()) {
                    mask_floor = generate_mask_var_by_step(mask_def_floor,
                            cur_step, trans_lanes_16bitx8, sup_condition, true);
                    cur_list_floor.emplace_back(mask_def_floor);
                }
            }
            auto assign = builder::make_assign_unattached(
                    builder::make_indexing(
                            output, tmp_out_indexes, trans_lanes_16bitx8, mask),
                    rows[i + 8]);
            cur_list.emplace_back(assign);
            if (inputloop_otheraxis_condition.defined()) {
                auto assign_floor = builder::make_assign_unattached(
                        builder::make_indexing(output, tmp_out_indexes,
                                trans_lanes_16bitx8, mask_floor),
                        rows[i + 8]);
                cur_list_floor.emplace_back(assign_floor);
            }
        }
    };

    auto compute_transpose_direct = [&](const std::vector<expr> &in_indexes,
                                            const std::vector<expr>
                                                    &out_indexes,
                                            const std::vector<expr>
                                                    &plain_indexes,
                                            const std::vector<expr>
                                                    &out_indexes_slice) {
        expr mask;
        for (int i = 0; i < (direct_use_bit16x16 ? lanesx16 : step); i++) {
            auto tmp_in_indexes = in_indexes;
            auto in_axis = inp_a_axis;
            tmp_in_indexes[in_axis]
                    = tmp_in_indexes[in_axis] + static_cast<uint64_t>(i);
            auto tmp_plain_indexes = plain_indexes;
            tmp_plain_indexes[input_format_code.get(in_axis)]
                    = tmp_plain_indexes[input_format_code.get(in_axis)]
                    + static_cast<uint64_t>(i);
            expr tmp_in = src.tptr_;
            if (output_loop) { tmp_in = input; }
            if (need_mask) {
                expr cur_step, sup_condition;
                determine_cur_step(output_loop ? input_blocking_dims_expr
                                               : src.get_shape(),
                        tmp_in_indexes, tmp_plain_indexes, cur_step,
                        sup_condition, in_axis, inp_b_axis, output_loop, step,
                        input_format_code);
                src_opt_mask(cur_step, sup_condition, step);
                stmt mask_def;
                mask = generate_mask_var_by_step(
                        mask_def, cur_step, step, sup_condition, true);
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
        if (is_s8u8) {
            TRANS2D_REG_CALCULATION_U8S8X16();
        } else if (direct_use_bit16x16) {
            TRANS2D_REG_CALCULATION_BIT16x16(type_bits);
        } else {
            if (sc_trans_kernel::F32_16X16_TRANS == trans_kernel_used) {
                TRANS2D_REG_CALCULATION_F32(type_bits);
            } else {
                TRANS2D_REG_CALCULATION_F32(type_bits);
            }
        }
        for (int i = 0; i < (direct_use_bit16x16 ? lanesx16 : step); i++) {
            auto tmp_out_indexes = out_indexes;
            auto out_axis = out_b_axis;
            tmp_out_indexes[out_axis]
                    = tmp_out_indexes[out_axis] + static_cast<uint64_t>(i);
            auto tmp_out_indexes_slice
                    = output_loop ? out_indexes_slice : out_indexes;
            tmp_out_indexes_slice[out_axis] = tmp_out_indexes_slice[out_axis]
                    + static_cast<uint64_t>(i);
            if (need_mask) {
                expr cur_step, sup_condition;
                determine_cur_step(output_loop ? dst.get_shape()
                                               : output_blocking_dims_expr,
                        tmp_out_indexes_slice, plain_indexes, cur_step,
                        sup_condition, out_axis, out_a_axis, output_loop, step,
                        input_format_code, true);
                dst_opt_mask(cur_step, sup_condition, in_indexes, i, step);
                stmt mask_def;
                mask = generate_mask_var_by_step(
                        mask_def, cur_step, step, sup_condition, true);
                cur_list.emplace_back(mask_def);
            }
            auto assign = builder::make_assign_unattached(
                    builder::make_indexing(output, tmp_out_indexes, step, mask),
                    rows[i]);
            cur_list.emplace_back(assign);
        }
    };

    auto compute_transpose_bit16 = [&](const std::vector<expr> &in_indexes,
                                           const std::vector<expr> &out_indexes,
                                           const std::vector<expr>
                                                   &plain_indexes,
                                           const std::vector<expr>
                                                   &out_indexes_slice) {
        expr mask;
        for (int i = 0; i < step; i++) {
            for (int p = 0; p < 4; p++) {
                auto tmp_in_indexes = in_indexes;
                auto in_axis = inp_a_axis;
                tmp_in_indexes[in_axis] = tmp_in_indexes[in_axis]
                        + (static_cast<uint64_t>(i) + p * 8);
                auto tmp_plain_indexes = plain_indexes;
                tmp_plain_indexes[input_format_code.get(in_axis)]
                        = tmp_plain_indexes[input_format_code.get(in_axis)]
                        + (static_cast<uint64_t>(i) + p * 8);
                expr tmp_in = src.tptr_;
                if (output_loop) { tmp_in = input; }
                if (need_mask) {
                    expr cur_step, sup_condition;
                    determine_cur_step(output_loop ? input_blocking_dims_expr
                                                   : src.get_shape(),
                            tmp_in_indexes, tmp_plain_indexes, cur_step,
                            sup_condition, in_axis, inp_b_axis, output_loop,
                            trans_lanesx8, input_format_code);
                    src_opt_mask(cur_step, sup_condition, trans_lanesx8);
                    stmt mask_def;
                    mask = generate_mask_var_by_step(mask_def, cur_step,
                            trans_lanesx8, sup_condition, true);
                    cur_list.emplace_back(mask_def);
                }
                auto brct_src = builder::make_broadcast(
                        builder::make_indexing(
                                tmp_in, tmp_in_indexes, step, mask),
                        trans_lanes_16bitx8);
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

        TRANS2D_REG_CALCULATION_BIT16();
        for (int i = 0; i < step; i++) {
            auto tmp_out_indexes = out_indexes;
            auto out_axis = out_b_axis;
            tmp_out_indexes[out_axis]
                    = tmp_out_indexes[out_axis] + static_cast<uint64_t>(i);
            auto tmp_out_indexes_slice
                    = output_loop ? out_indexes_slice : out_indexes;
            tmp_out_indexes_slice[out_axis] = tmp_out_indexes_slice[out_axis]
                    + static_cast<uint64_t>(i);
            if (need_mask) {
                expr cur_step, sup_condition;
                determine_cur_step(output_loop ? dst.get_shape()
                                               : output_blocking_dims_expr,
                        tmp_out_indexes_slice, plain_indexes, cur_step,
                        sup_condition, out_axis, out_a_axis, output_loop,
                        trans_lanes_16bitx8, input_format_code, true);
                dst_opt_mask(cur_step, sup_condition, in_indexes, i,
                        trans_lanes_16bitx8);
                stmt mask_def;
                mask = generate_mask_var_by_step(mask_def, cur_step,
                        trans_lanes_16bitx8, sup_condition, true);
                cur_list.emplace_back(mask_def);
            }
            auto assign = builder::make_assign_unattached(
                    builder::make_indexing(
                            output, tmp_out_indexes, trans_lanes_16bitx8, mask),
                    rows[i + 8]);
            cur_list.emplace_back(assign);
        }
    };

    auto compute_loops = [&](const std::vector<expr> &blocking_dims,
                                 const int a_axis, const int b_axis,
                                 const tensor_slice &tsr) {
        for (int i = static_cast<int>(blocking_dims.size()) - 1; i >= 0; i--) {
            if (utils::is_one_of(i, a_axis, b_axis)
                    && iter_vars.at(i).isa<var>()) {
                body = cur.isa<stmts>()
                        ? cur
                        : make_stmt<stmts_node_t>(
                                std::vector<stmt> {std::move(cur)});
                int cur_step = i == a_axis ? inp_a_step : inp_b_step;
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0), (tsr.get_shape()[i]), expr(cur_step),
                        std::move(body), true, for_type::NORMAL);
                bind_loop_axis(expand_gt, cur, i, true);
            }
        }
        for (int i = static_cast<int>(blocking_dims.size()) - 1; i >= 0; i--) {
            if (!utils::is_one_of(i, a_axis, b_axis)
                    && iter_vars.at(i).isa<var>()) {
                body = cur.isa<stmts>()
                        ? cur
                        : make_stmt<stmts_node_t>(
                                std::vector<stmt> {std::move(cur)});
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0), tsr.get_shape()[i], expr(1), std::move(body),
                        true, for_type::NORMAL);
                bind_loop_axis(expand_gt, cur, i, true);
            }
        }
    };

    bool use_direct = dtype == datatypes::f32 || is_s8u8 || direct_use_bit16x16;
    if (!output_loop) {
        std::vector<expr> in_indexes, loop_indexes, in_indexes_tail,
                loop_indexes_tail;
        auto indexvar_or_zero = [&src](size_t idx) {
            return range_from_outer_loop(src.get_ranges()[idx])
                    ? expr(0)
                    : builder::make_var(datatypes::index,
                            std::string("_fuseiter") + fusion_create_idx());
        };
        auto make_iter_vars = [&](std::vector<expr> &iter_vars,
                                      bool is_tail = false) {
            for (size_t i = 0; i < input_blocking_dims_expr.size(); i++) {
                iter_vars.emplace_back(indexvar_or_zero(i));
                in_indexes.emplace_back(iter_vars[i] + src.get_offset()[i]);
                loop_indexes.emplace_back(iter_vars[i]);
            }
        };
        make_iter_vars(iter_vars);
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
        if (use_split_loop && !is_dynamic) {
            cur_list.clear();
            cur_list_floor.clear();
            cur_list_lastdim_floor.clear();
        }
        if (dtype == datatypes::f32 || is_s8u8 || direct_use_bit16x16) {
            if (use_split_loop && !is_dynamic) {
                inputloop_split_ir_directly(
                        loop_indexes, out_indexes, tmp_indexes, out_indexes);
            } else {
                compute_transpose_direct(
                        loop_indexes, out_indexes, tmp_indexes, out_indexes);
            }
        } else {
            if (use_split_loop && !is_dynamic) {
                inputloop_split_ir_bit16(
                        loop_indexes, out_indexes, tmp_indexes, out_indexes);
            } else {
                compute_transpose_bit16(
                        loop_indexes, out_indexes, tmp_indexes, out_indexes);
            }
        }
        cur = builder::make_stmts_unattached(cur_list);
        if (use_split_loop && !is_dynamic) {
            if (inputloop_otheraxis_condition.defined()
                    && inputloop_lastdim_condition.defined()) {
                auto cur_floor_lastdim = builder::make_stmts_unattached(
                        cur_list_lastdim_floor);
                auto cur_floor = builder::make_stmts_unattached(cur_list_floor);
                // other axis out of bound
                auto other_axis_meet = builder::make_if_else_unattached(
                        inputloop_lastdim_condition, cur_floor,
                        cur_floor_lastdim);
                cur = builder::make_if_else_unattached(
                        inputloop_otheraxis_condition, cur, other_axis_meet);
            } else if (inputloop_lastdim_condition.defined()) {
                auto cur_floor_lastdim = builder::make_stmts_unattached(
                        cur_list_lastdim_floor);
                cur = builder::make_if_else_unattached(
                        inputloop_lastdim_condition, cur, cur_floor_lastdim);
            } else if (inputloop_otheraxis_condition.defined()) {
                auto cur_floor = builder::make_stmts_unattached(cur_list_floor);
                cur = builder::make_if_else_unattached(
                        inputloop_otheraxis_condition, cur, cur_floor);
            }
            var_define_list.emplace_back(cur);
            cur = builder::make_stmts_unattached(var_define_list);
        }
        compute_loops(input_blocking_dims_expr, inp_a_axis, inp_b_axis, src);
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    } else {
        std::vector<expr> out_indexes, out_indexes_slice;
        for (size_t i = 0; i < output_blocking_dims_expr.size(); i++) {
            iter_vars.emplace_back(range_from_outer_loop(dst.get_ranges()[i])
                            ? expr(0)
                            : builder::make_var(datatypes::index,
                                    std::string("_fuseiter")
                                            + fusion_create_idx()));
            out_indexes.emplace_back(iter_vars[i] + dst.get_offset()[i]);
            out_indexes_slice.emplace_back(iter_vars[i]);
        }
        expr condition;
        expr last_axis_offset, other_axis_condition;
        std::vector<expr> tmp_indexes = get_reorder_block2plain_indexes(graph,
                out_indexes, output_format, plain_dims, condition,
                last_axis_offset, other_axis_condition);
        std::vector<expr> in_indexes
                = get_reorder_plain2block_indexes(tmp_indexes, input_format);
        if (use_split_loop && !is_dynamic) {
            cur_list.clear();
            cur_list_floor.clear();
        }
        if (use_direct) {
            if (use_split_loop && !is_dynamic) {
                outputloop_split_ir_directly(in_indexes, out_indexes,
                        tmp_indexes, out_indexes_slice);
            } else {
                compute_transpose_direct(
                        in_indexes, out_indexes, tmp_indexes, out_indexes);
            }
        } else {
            compute_transpose_bit16(
                    in_indexes, out_indexes, tmp_indexes, out_indexes_slice);
        }
        cur = builder::make_stmts_unattached(cur_list);
        if (use_split_loop && !is_dynamic) {
            if (outputloop_axis_condition.defined() && use_direct) {
                auto cur_floor = builder::make_stmts_unattached(cur_list_floor);
                cur = builder::make_if_else_unattached(
                        outputloop_axis_condition, cur, cur_floor);
            }
            var_define_list.emplace_back(cur);
            cur = builder::make_stmts_unattached(var_define_list);
        }
        compute_loops(output_blocking_dims_expr, out_a_axis, out_b_axis, dst);
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    }
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
