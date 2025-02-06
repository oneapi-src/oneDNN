/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_OCL_POST_OPS_H
#define GPU_INTEL_OCL_OCL_POST_OPS_H

#if WITH_POST_OP

#if !WITH_ELTWISE
#undef WITH_ELTWISE
#define WITH_ELTWISE 1
#endif

#include "gpu/intel/ocl/ocl_conversion.h"
#include "gpu/intel/ocl/ocl_eltwise.h"
#include "gpu/intel/ocl/ocl_io.h"

float fwd_Xnary(unsigned kind, unsigned algorithm, float x, float y,
        float alpha, float beta, float scale) {
    if (kind == PO_BINARY) {
        switch (algorithm) {
            // binary
            case BINARY_ADD: return x + y; break;
            case BINARY_MUL: return x * y; break;
            case BINARY_MIN: return x < y ? x : y; break;
            case BINARY_MAX: return x > y ? x : y; break;
            case BINARY_DIV: return x / y; break;
            case BINARY_SUB: return x - y; break;
            case BINARY_GE: return x >= y; break;
            case BINARY_GT: return x > y; break;
            case BINARY_LE: return x <= y; break;
            case BINARY_LT: return x < y; break;
            case BINARY_EQ: return x == y; break;
            case BINARY_NE: return x != y; break;
            case RELU: // binary && relu = prelu
                return fwd_eltwise_common(RELU, x, y, beta, scale);
                break;
            default: return 0.f;
        }
    } else { // eltwise kind
        return fwd_eltwise_common(algorithm, x, alpha, beta, scale);
    }
}

#define FWD_XNARY_GENERIC_DT(po_kind, algorithm, res_ptr, arg0_ptr, arg0_len, \
        arg1_ptr, arg1_len, alpha, beta, scale) \
    { \
        auto ty = arg0_len + arg1_len; \
        const typeof(ty) out_len \
                = max((typeof(ty))arg0_len, (typeof(ty))arg1_len); \
        unroll_for(typeof(out_len + 0) idx = 0; idx < out_len; ++idx) { \
            const int arg0_idx = arg0_len == 1 ? 0 : idx; \
            const int arg1_idx = arg1_len == 1 ? 0 : idx; \
            res_ptr[arg0_len == 1 && arg1_len == 1 ? 0 : idx] = fwd_Xnary( \
                    po_kind, algorithm, into_float(arg0_ptr[arg0_idx]), \
                    into_float(arg1_ptr[arg1_idx]), alpha, beta, scale); \
        } \
    }

#define FMA_BLOCK( \
        block_size, nof_elems, acc_ptr, acc_elem_dt, a_ptr, a_elem_dt, b, c) \
    unroll_for(; nof_elems >= block_size; acc_ptr += block_size, \
               a_ptr += block_size, nof_elems -= block_size) { \
        CONCAT2(acc_elem_dt, block_size) \
        a_conv = CONCAT3(convert_, acc_elem_dt, block_size)( \
                *((CONCAT2(a_elem_dt, block_size) *)a_ptr)); \
        *((CONCAT2(acc_elem_dt, block_size) *)acc_ptr) = fma(a_conv - c, b, \
                *((CONCAT2(acc_elem_dt, block_size) *)acc_ptr)); \
    }

#define FMA_MIXED(acc_nof_elems, a, a_elem_dt, b, acc_ptr, acc_elem_dt, c) \
    { \
        auto nof_elems = acc_nof_elems; \
        a_elem_dt *a_ptr = (a_elem_dt *)(&a); \
        FMA_BLOCK(8, nof_elems, acc_ptr, acc_elem_dt, a_ptr, a_elem_dt, b, c); \
        FMA_BLOCK(4, nof_elems, acc_ptr, acc_elem_dt, a_ptr, a_elem_dt, b, c); \
        FMA_BLOCK(2, nof_elems, acc_ptr, acc_elem_dt, a_ptr, a_elem_dt, b, c); \
        if (nof_elems == 1) { *acc_ptr += (*a_ptr - c) * b; } \
    }

#define po_dt(idx) CONCAT3(PO_, idx, _BIN_ARG_ACTUAL_DATA_T)
#define po_buf(idx) ((__global po_dt(idx) *)(CONCAT3(po_, idx, _binary_arg)))

#define FILL_BIN_ARG_SERIAL(idx, dest_ptr, x0, x0_s, x1, x1_s, x1_incr, x2, \
        x2_s, x3, x3_s, x4, x4_s, x5, x5_s) \
    unroll_for(typeof(x0 + x0_s) x0_idx = x0, bin_arg_offset = 0; \
               x0_idx < x0 + x0_s; ++x0_idx) { \
        unroll_for(typeof(x1 + x1_s + x1_incr) x1_idx = x1; \
                   x1_idx < x1 + x1_s; x1_idx += x1_incr) { \
            unroll_for(typeof(x2 + x2_s) x2_idx = x2; x2_idx < x2 + x2_s; \
                       ++x2_idx) { \
                unroll_for(typeof(x3 + x3_s) x3_idx = x3; x3_idx < x3 + x3_s; \
                           ++x3_idx) { \
                    unroll_for(typeof(x4 + x4_s) x4_idx = x4; \
                               x4_idx < x4 + x4_s; ++x4_idx) { \
                        unroll_for(typeof(x5 + x5_s) x5_idx = x5; \
                                   x5_idx < x5 + x5_s; \
                                   ++x5_idx, ++bin_arg_offset) { \
                            const auto bin_arg_glob_off = OFF_MD( \
                                    CONCAT3(PO_, idx, _BIN_ARG), \
                                    x0_idx % CONCAT3(PO_, idx, _BIN_ARG_D0), \
                                    x1_idx % CONCAT3(PO_, idx, _BIN_ARG_D1), \
                                    x2_idx % CONCAT3(PO_, idx, _BIN_ARG_D2), \
                                    x3_idx % CONCAT3(PO_, idx, _BIN_ARG_D3), \
                                    x4_idx % CONCAT3(PO_, idx, _BIN_ARG_D4), \
                                    x5_idx % CONCAT3(PO_, idx, _BIN_ARG_D5)); \
                            dest_ptr[bin_arg_offset] = into_float( \
                                    po_buf(idx)[bin_arg_glob_off]); \
                        } \
                    } \
                } \
            } \
        } \
    }

#define FILL_WITH_BLOCK_READ(src_ptr, dst_ptr, nelem) \
    unroll_for(typeof(nelem + 0) load_idx = 0; load_idx < nelem; ++load_idx) { \
        block_load(&dst_ptr[load_idx], \
                src_ptr + load_idx * get_sub_group_size()); \
    }

#define X_NELEMS(x) (x / get_sub_group_size())

#define CONDITIONAL_FILL(blocked_coord, nelem, src_ptr, dst_ptr) \
    if (blocked_coord / get_sub_group_size() == nelem) \
        FILL_WITH_BLOCK_READ(src_ptr, dst_ptr, nelem);

#define FILL_BIN_ARG_TRY_BLOCK(idx, dest_ptr, dest_size, x0, x0_s, x1, x1_s, \
        x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s) \
    { \
        unroll_for(typeof(x0 + x0_s) x0_idx = x0, arg_off = 0; \
                   x0_idx < x0 + x0_s; ++x0_idx, arg_off += X_NELEMS(x1_s)) { \
            const auto bin_arg_glob_off = OFF_MD(CONCAT3(PO_, idx, _BIN_ARG), \
                    x0_idx % CONCAT3(PO_, idx, _BIN_ARG_D0), \
                    x1 % CONCAT3(PO_, idx, _BIN_ARG_D1), \
                    x2 % CONCAT3(PO_, idx, _BIN_ARG_D2), \
                    x3 % CONCAT3(PO_, idx, _BIN_ARG_D3), \
                    x4 % CONCAT3(PO_, idx, _BIN_ARG_D4), \
                    x5 % CONCAT3(PO_, idx, _BIN_ARG_D5)); \
\
            CONDITIONAL_FILL(x1_s, 1, (po_buf(idx) + bin_arg_glob_off), \
                    (dest_ptr + arg_off)); \
            CONDITIONAL_FILL(x1_s, 2, (po_buf(idx) + bin_arg_glob_off), \
                    (dest_ptr + arg_off)); \
            CONDITIONAL_FILL(x1_s, 4, (po_buf(idx) + bin_arg_glob_off), \
                    (dest_ptr + arg_off)); \
        } \
    }

#define REPLICATE_DATA( \
        dest_ptr, dest_size, x0_s, x1_s, x2_s, x3_s, x4_s, x5_s) \
    { \
        const auto copy_size = x0_s * x1_s * x2_s * x3_s * x4_s * x5_s; \
        unroll_for(typeof(dest_size + 0) fid = copy_size; fid < dest_size; \
                   ++fid) { \
            *(dest_ptr + fid) = *(dest_ptr + (fid % copy_size)); \
        } \
    }

#define IS_BURSTABLE(idx, x0, x0_s, x1, x1_s, x2, x2_s, x3, x3_s, x4, x4_s, \
        x5, x5_s, is_burst) \
    ({ \
        bool is_burstable = is_burst; \
        if (x0_s > CONCAT3(PO_, idx, _BIN_ARG_D0) && x0_s > 1) \
            is_burstable = false; \
        if (x1_s > CONCAT3(PO_, idx, _BIN_ARG_D1) && x1_s > 1) \
            is_burstable = false; \
        if (x2_s > CONCAT3(PO_, idx, _BIN_ARG_D2) && x2_s > 1) \
            is_burstable = false; \
        if (x3_s > CONCAT3(PO_, idx, _BIN_ARG_D3) && x3_s > 1) \
            is_burstable = false; \
        if (x4_s > CONCAT3(PO_, idx, _BIN_ARG_D4) && x4_s > 1) \
            is_burstable = false; \
        if (x5_s > CONCAT3(PO_, idx, _BIN_ARG_D5) && x5_s > 1) \
            is_burstable = false; \
        if ((CONCAT3(PO_, idx, _BIN_ARG_D0) == 1) \
                && (CONCAT3(PO_, idx, _BIN_ARG_D1) == 1) \
                && (CONCAT3(PO_, idx, _BIN_ARG_D2) == 1) \
                && (CONCAT3(PO_, idx, _BIN_ARG_D2) == 1) \
                && (CONCAT3(PO_, idx, _BIN_ARG_D3) == 1) \
                && (CONCAT3(PO_, idx, _BIN_ARG_D4) == 1) \
                && (CONCAT3(PO_, idx, _BIN_ARG_D5) == 1)) \
            is_burstable = false; \
\
        is_burstable; \
    })

#define APPLY_PO_BINARY(idx, accumulator, x0, x0_s, x1, x1_s, x1_incr, x2, \
        x2_s, x3, x3_s, x4, x4_s, x5, x5_s, is_burst, bin_arg_size) \
    { \
        float bin_arg[bin_arg_size]; \
        __private float *bin_arg_ptr = &bin_arg[0]; \
        const bool use_burst_read = IS_BURSTABLE(idx, x0, x0_s, x1, x1_s, x2, \
                x2_s, x3, x3_s, x4, x4_s, x5, x5_s, is_burst); \
        if (use_burst_read) { \
            FILL_BIN_ARG_TRY_BLOCK(idx, bin_arg_ptr, bin_arg_size, x0, x0_s, \
                    x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                    x5_s); \
            REPLICATE_DATA(bin_arg_ptr, bin_arg_size, x0_s, X_NELEMS(x1_s), \
                    x2_s, x3_s, x4_s, x5_s); \
        } else { \
            const auto x1_jump = is_burst ? (int)get_sub_group_size() : 1; \
            const auto x1_size = x1_s / x1_jump; \
            FILL_BIN_ARG_SERIAL(idx, bin_arg_ptr, x0, x0_s, (x1 + x1_incr), \
                    x1_s, x1_jump, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s); \
            REPLICATE_DATA(bin_arg_ptr, bin_arg_size, x0_s, x1_size, x2_s, \
                    x3_s, x4_s, x5_s); \
        } \
        FWD_XNARY_GENERIC_DT(PO_BINARY, CONCAT3(PO_, idx, _ALG), accumulator, \
                accumulator, bin_arg_size, bin_arg_ptr, bin_arg_size, 0.0f, \
                0.0f, 1.0f); \
    }

#define APPLY_PO_SUM( \
        idx, accumulator, acc_size, acc_elem_dt, sum_src, sum_elem_dt) \
    FMA_MIXED(acc_size, sum_src, sum_elem_dt, CONCAT3(PO_, idx, _SUM_SCALE), \
            accumulator, acc_elem_dt, CONCAT3(PO_, idx, _SUM_ZP));

#define APPLY_PO_ELTWISE(idx, accumulator, nelems) \
    FWD_XNARY_GENERIC_DT(PO_ELTWISE, CONCAT3(PO_, idx, _ALG), accumulator, \
            accumulator, nelems, accumulator, nelems, \
            CONCAT3(PO_, idx, _ELTWISE_ALPHA), \
            CONCAT3(PO_, idx, _ELTWISE_BETA), \
            CONCAT3(PO_, idx, _ELTWISE_SCALE));

#define APPLY_PO_STAGE(idx, acc, acc_elem_dt, nelems, sum_src, sum_elem_dt, \
        x0, x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    switch (CONCAT3(PO_, idx, _KIND)) { \
        case PO_BINARY: \
            APPLY_PO_BINARY(idx, acc, x0, x0_s, x1, x1_s, x1_incr, x2, x2_s, \
                    x3, x3_s, x4, x4_s, x5, x5_s, is_burst, nelems); \
            break; \
        case PO_ELTWISE: APPLY_PO_ELTWISE(idx, acc, nelems); break; \
        case PO_SUM: \
            APPLY_PO_SUM(idx, acc, nelems, acc_elem_dt, sum_src, sum_elem_dt); \
            break; \
    }

// clang-format off
#define APPLY_PO_STAGE_0(...)
#define APPLY_PO_STAGE_1(...) APPLY_PO_STAGE(0, __VA_ARGS__)
#define APPLY_PO_STAGE_2(...) APPLY_PO_STAGE_1(__VA_ARGS__) APPLY_PO_STAGE(1, __VA_ARGS__)
#define APPLY_PO_STAGE_3(...) APPLY_PO_STAGE_2(__VA_ARGS__) APPLY_PO_STAGE(2, __VA_ARGS__)
#define APPLY_PO_STAGE_4(...) APPLY_PO_STAGE_3(__VA_ARGS__) APPLY_PO_STAGE(3, __VA_ARGS__)
#define APPLY_PO_STAGE_5(...) APPLY_PO_STAGE_4(__VA_ARGS__) APPLY_PO_STAGE(4, __VA_ARGS__)
#define APPLY_PO_STAGE_6(...) APPLY_PO_STAGE_5(__VA_ARGS__) APPLY_PO_STAGE(5, __VA_ARGS__)
#define APPLY_PO_STAGE_7(...) APPLY_PO_STAGE_6(__VA_ARGS__) APPLY_PO_STAGE(6, __VA_ARGS__)
#define APPLY_PO_STAGE_8(...) APPLY_PO_STAGE_7(__VA_ARGS__) APPLY_PO_STAGE(7, __VA_ARGS__)
#define APPLY_PO_STAGE_9(...) APPLY_PO_STAGE_8(__VA_ARGS__) APPLY_PO_STAGE(8, __VA_ARGS__)
#define APPLY_PO_STAGE_10(...) APPLY_PO_STAGE_9(__VA_ARGS__) APPLY_PO_STAGE(9, __VA_ARGS__)
#define APPLY_PO_STAGE_11(...) APPLY_PO_STAGE_10(__VA_ARGS__) APPLY_PO_STAGE(10, __VA_ARGS__)
#define APPLY_PO_STAGE_12(...) APPLY_PO_STAGE_11(__VA_ARGS__) APPLY_PO_STAGE(11, __VA_ARGS__)
#define APPLY_PO_STAGE_13(...) APPLY_PO_STAGE_12(__VA_ARGS__) APPLY_PO_STAGE(12, __VA_ARGS__)
#define APPLY_PO_STAGE_14(...) APPLY_PO_STAGE_13(__VA_ARGS__) APPLY_PO_STAGE(13, __VA_ARGS__)
#define APPLY_PO_STAGE_15(...) APPLY_PO_STAGE_14(__VA_ARGS__) APPLY_PO_STAGE(14, __VA_ARGS__)
#define APPLY_PO_STAGE_16(...) APPLY_PO_STAGE_15(__VA_ARGS__) APPLY_PO_STAGE(15, __VA_ARGS__)
#define APPLY_PO_STAGE_17(...) APPLY_PO_STAGE_16(__VA_ARGS__) APPLY_PO_STAGE(16, __VA_ARGS__)
#define APPLY_PO_STAGE_18(...) APPLY_PO_STAGE_17(__VA_ARGS__) APPLY_PO_STAGE(17, __VA_ARGS__)
#define APPLY_PO_STAGE_19(...) APPLY_PO_STAGE_18(__VA_ARGS__) APPLY_PO_STAGE(18, __VA_ARGS__)
#define APPLY_PO_STAGE_20(...) APPLY_PO_STAGE_19(__VA_ARGS__) APPLY_PO_STAGE(19, __VA_ARGS__)
#define APPLY_PO_STAGE_21(...) APPLY_PO_STAGE_20(__VA_ARGS__) APPLY_PO_STAGE(20, __VA_ARGS__)
#define APPLY_PO_STAGE_22(...) APPLY_PO_STAGE_21(__VA_ARGS__) APPLY_PO_STAGE(21, __VA_ARGS__)
#define APPLY_PO_STAGE_23(...) APPLY_PO_STAGE_22(__VA_ARGS__) APPLY_PO_STAGE(22, __VA_ARGS__)
#define APPLY_PO_STAGE_24(...) APPLY_PO_STAGE_23(__VA_ARGS__) APPLY_PO_STAGE(23, __VA_ARGS__)
#define APPLY_PO_STAGE_25(...) APPLY_PO_STAGE_24(__VA_ARGS__) APPLY_PO_STAGE(24, __VA_ARGS__)
#define APPLY_PO_STAGE_26(...) APPLY_PO_STAGE_25(__VA_ARGS__) APPLY_PO_STAGE(25, __VA_ARGS__)
#define APPLY_PO_STAGE_27(...) APPLY_PO_STAGE_26(__VA_ARGS__) APPLY_PO_STAGE(26, __VA_ARGS__)
#define APPLY_PO_STAGE_28(...) APPLY_PO_STAGE_27(__VA_ARGS__) APPLY_PO_STAGE(27, __VA_ARGS__)
#define APPLY_PO_STAGE_29(...) APPLY_PO_STAGE_28(__VA_ARGS__) APPLY_PO_STAGE(28, __VA_ARGS__)
#define APPLY_PO_STAGE_30(...) APPLY_PO_STAGE_29(__VA_ARGS__) APPLY_PO_STAGE(29, __VA_ARGS__)
#define APPLY_PO_STAGE_31(...) APPLY_PO_STAGE_30(__VA_ARGS__) APPLY_PO_STAGE(30, __VA_ARGS__)
#define APPLY_PO_STAGE_32(...) APPLY_PO_STAGE_31(__VA_ARGS__) APPLY_PO_STAGE(31, __VA_ARGS__)
// clang-format on

#define APPLY_ALL_PO_STAGES(accumulator, acc_elem_dt, ...) \
    { \
        const int nelems = sizeof(accumulator) / sizeof(acc_elem_dt); \
        acc_elem_dt *acc = &accumulator; \
        CONCAT2(APPLY_PO_STAGE_, POST_OP_CHAIN_LENGTH) \
        (acc, acc_elem_dt, nelems, __VA_ARGS__) \
    }

#define APPLY_POST_OPS_BL(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    APPLY_ALL_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
            x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
            is_burst);

#define APPLY_POST_OPS_TRY_BURST(accumulator, acc_elem_dt, sum_src, \
        sum_elem_dt, mb_start, mb_size, oc_start, oc_size, oc_serial_incr) \
    APPLY_POST_OPS_BL(accumulator, acc_elem_dt, sum_src, sum_elem_dt, \
            mb_start, mb_size, oc_start, oc_size, oc_serial_incr, 0, 1, 0, 1, \
            0, 1, 0, 1, true)

#define APPLY_POST_OPS_SERIAL(accumulator, acc_elem_dt, sum_src, sum_elem_dt, \
        mb_start, mb_size, oc_start, oc_size, d2_start, d2_size, d3_start, \
        d3_size, d4_start, d4_size, d5_start, d5_size) \
    APPLY_POST_OPS_BL(accumulator, acc_elem_dt, sum_src, sum_elem_dt, \
            mb_start, mb_size, oc_start, oc_size, 0, d2_start, d2_size, \
            d3_start, d3_size, d4_start, d4_size, d5_start, d5_size, false)

#define APPLY_POST_OPS_SERIAL_BINARY_2D(accumulator, acc_elem_dt, sum_src, \
        sum_elem_dt, mb_start, mb_size, oc_start, oc_size) \
    APPLY_POST_OPS_BL(accumulator, acc_elem_dt, sum_src, sum_elem_dt, \
            mb_start, mb_size, oc_start, oc_size, 0, 0, 1, 0, 1, 0, 1, 0, 1, \
            false)
#else

#define APPLY_POST_OPS_SERIAL(...)
#define APPLY_POST_OPS_SERIAL_BINARY_2D(...)
#define APPLY_POST_OPS_TRY_BURST(...)

#endif // WITH_POST_OP

#endif
