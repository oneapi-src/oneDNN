/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

#if defined(IS_MAX)
#define INIT_ACC TO_DEF_ACC_DATA_T(DATA_MIN)
#elif defined(IS_MIN)
#define INIT_ACC TO_DEF_ACC_DATA_T(DATA_MAX)
#elif defined(IS_MUL)
#define INIT_ACC TO_DEF_ACC_DATA_T(DATA_ONE)
#else
#define INIT_ACC TO_DEF_ACC_DATA_T(DATA_ZERO)
#endif

#if defined(IS_MAX)
#if defined(SRC_DT_S8) || defined(SRC_DT_U8)
#define ACCUMULATE(x, y) max(x, y)
#else
#define ACCUMULATE(x, y) fmax(x, y)
#endif
#elif defined(IS_MIN)
#if defined(SRC_DT_S8) || defined(SRC_DT_U8)
#define ACCUMULATE(x, y) min(x, y)
#else
#define ACCUMULATE(x, y) fmin(x, y)
#endif
#elif defined(IS_MEAN) || defined(IS_SUM)
#define ACCUMULATE(x, y) (x + y)
#elif defined(IS_MUL)
#define ACCUMULATE(x, y) (x * y)
#else
#define ACCUMULATE(x, y) (x + pow(fabs(y), POWER))
#endif

// We want to use some acc algorithms (like pow) only once
// for a given element
#if defined(IS_MAX) || defined(IS_MIN) || defined(IS_MUL)
#define ACCUMULATE_AGAIN(x, y) ACCUMULATE(x, y)
#else
#define ACCUMULATE_AGAIN(x, y) (x + y)
#endif

#if defined(IS_MEAN)
#define FINALIZE_REDUCTION(x) (x / REDUCTION_SIZE)
#elif defined(IS_LP_MAX)
#define FINALIZE_REDUCTION(x) rootn(fmax(x, EPS), POWER)
#elif defined(IS_LP_SUM)
#define FINALIZE_REDUCTION(x) rootn(x + EPS, POWER)
#elif defined(IS_P_MAX)
#define FINALIZE_REDUCTION(x) fmax(x, EPS)
#elif defined(IS_P_SUM)
#define FINALIZE_REDUCTION(x) (x + EPS)
#else
#define FINALIZE_REDUCTION(x) (x)
#endif

#if WITH_SUM
#define INIT_SUM(sum_data) DST_TO_REF(sum_data)
#else
#define INIT_SUM(sum_data) 0.0f
#endif //WITH_SUM

#if defined(IS_MAX)
#define SUB_GROUP_REDUCE(x, c_block) sub_group_reduce_max(x)
#elif defined(IS_MIN)
#define SUB_GROUP_REDUCE(x, c_block) sub_group_reduce_min(x)
#elif defined(IS_MUL)
#define SUB_GROUP_REDUCE(x, c_block) \
    ({ \
        int cid_end \
                = (INITIAL_C % SUB_GROUP_SIZE == 0 ? SUB_GROUP_SIZE \
                                                   : (INITIAL_C - c_block)); \
        DEF_ACC_DATA_T sub_group_acc = 1.0; \
        for (int channel_id = 0; channel_id < cid_end; channel_id++) { \
            sub_group_acc *= intel_sub_group_shuffle(c_acc, channel_id); \
        } \
        sub_group_acc; \
    })
#else
#define SUB_GROUP_REDUCE(x, c_block) sub_group_reduce_add(x)
#endif

#if INITIAL_C_CHUNKS == 1
#define C_BLOCK_READ BLOCK_READ
#define AS_C_BLOCK_DATA_T AS_DATA_T
#define CONVERT_C_BLOCK_T TO_DEF_ACC_DATA_T
#define C_BLOCK_T DEF_ACC_DATA_T
#elif INITIAL_C_CHUNKS == 2
#define C_BLOCK_READ BLOCK_READ2
#define AS_C_BLOCK_DATA_T AS_DATA2_T
#define CONVERT_C_BLOCK_T TO_DEF_ACC_DATA2_T
#define C_BLOCK_T DEF_ACC_DATA2_T
#elif INITIAL_C_CHUNKS == 4
#define C_BLOCK_READ BLOCK_READ4
#define AS_C_BLOCK_DATA_T AS_DATA4_T
#define CONVERT_C_BLOCK_T TO_DEF_ACC_DATA4_T
#define C_BLOCK_T DEF_ACC_DATA4_T
#elif INITIAL_C_CHUNKS == 8
#define C_BLOCK_READ BLOCK_READ8
#define AS_C_BLOCK_DATA_T AS_DATA8_T
#define CONVERT_C_BLOCK_T TO_DEF_ACC_DATA8_T
#define C_BLOCK_T DEF_ACC_DATA8_T
#endif

#define ROUND_DOWN(a, b) ((a) - ((a) % (b)))
#undef ROUND_UP
#define ROUND_UP(a, b) ROUND_DOWN((a + b - 1), (b))

// clang-format off
// C blocked or N,C blocked
#define INITIAL_SRC_OFFSET(n, c, hwd) \
    (((n) / N_BLOCK_SIZE) * INITIAL_HWD_DIM * N_BLOCK_SIZE  * ROUND_UP(INITIAL_C, C_BLOCK_SIZE) + \
     ((c) / C_BLOCK_SIZE) * INITIAL_HWD_DIM * N_BLOCK_SIZE  * C_BLOCK_SIZE + \
     (hwd)                                  * N_BLOCK_SIZE  * C_BLOCK_SIZE + \
     ((n)                                   % N_BLOCK_SIZE) * C_BLOCK_SIZE + \
     ((c)                                                   % C_BLOCK_SIZE))

#define INITIAL_DST_OFFSET(n, c, hwd) \
     ((n  / N_BLOCK_SIZE) * FINAL_HWD_DIM * N_BLOCK_SIZE  * ROUND_UP(FINAL_C_DIM, C_BLOCK_SIZE) + \
     ((c) / C_BLOCK_SIZE) * FINAL_HWD_DIM * N_BLOCK_SIZE  * C_BLOCK_SIZE + \
     (hwd)                                * N_BLOCK_SIZE  * C_BLOCK_SIZE + \
     ((n)                                 % N_BLOCK_SIZE) * C_BLOCK_SIZE + \
     ((c)                                                 % C_BLOCK_SIZE))

#define FINAL_SRC_OFFSET(n, c, hwd) INITIAL_DST_OFFSET(n, c, hwd)

#define FINAL_DST_OFFSET(n, c, hwd) \
     ((n) / N_BLOCK_SIZE) * (FINAL_HWD_DIM / FINAL_HWD_CHUNK_SIZE) * N_BLOCK_SIZE  * ROUND_UP(FINAL_C_DIM / FINAL_C_CHUNK_SIZE, C_BLOCK_SIZE) + \
     ((c) / C_BLOCK_SIZE) * (FINAL_HWD_DIM / FINAL_HWD_CHUNK_SIZE) * N_BLOCK_SIZE  * C_BLOCK_SIZE + \
     (hwd)                                                         * N_BLOCK_SIZE  * C_BLOCK_SIZE + \
     ((n)                                                          % N_BLOCK_SIZE) * C_BLOCK_SIZE + \
     ((c)                                                                          % C_BLOCK_SIZE)
// clang-format on

#if WITH_POST_OP
// Compute H, W, D indices before passing them into the post op macro
#define APPLY_POST_OPS(sum_data, data, n_idx, c_idx, hwd_idx) \
    { \
        float sum_init_val = INIT_SUM(sum_data); \
        const int D = hwd_start / (DST_H_DIM * DST_W_DIM); \
        const int H = (hwd_start % (DST_H_DIM * DST_W_DIM)) / DST_W_DIM; \
        const int W = hwd_start % DST_W_DIM; \
        APPLY_POST_OPS_SERIAL(data, float, sum_init_val, float, n_idx, 1, \
                c_idx, 1, D, 1, H, 1, W, 1, 0, 1); \
    }
#else
#define APPLY_POST_OPS(sum_data, data, n_idx, c_idx, hwd_idx) \
    {}
#endif // WITH_POST_OP

#define WRITE_FINAL_RESULT(dst_elem, acc, n_start, c_start, hwd_start) \
    { \
        float acc_float = FINALIZE_REDUCTION(convert_float(acc)); \
        APPLY_POST_OPS(dst_elem, acc_float, n_start, c_start, hwd_start); \
        dst_elem = TO_DST(acc_float); \
    }

#if SKIP_FINAL_PHASE
#define WRITE_INITIAL_RESULT WRITE_FINAL_RESULT
#define INITIAL_DST_DTYPE DST_DATA_T
#else
#define WRITE_INITIAL_RESULT(dst_elem, data, n_start, c_start, hwd_start) \
    { dst_elem = data; }
#define INITIAL_DST_DTYPE DEF_ACC_DATA_T
#endif

// Reduces only chunks of reduction dimensions
// in order to create more threads and increase precision
NAMED_KERNEL_ATTR(INITIAL)
__kernel void gen9_initial_reduce(__global SRC_DATA_T *src,
        __global INITIAL_DST_DTYPE *dst
#if SKIP_FINAL_PHASE
                POST_OP_ARGS
#endif
) {
    const int n_chunk_idx = GWS_GET_INITIAL_N();
    const int c = GWS_GET_INITIAL_C();
    const int c_block_idx = c / C_BLOCK_SIZE;
    const int hwd_chunk_idx = GWS_GET_INITIAL_HWD_CHUNK_ID();
    const int hwd_start = hwd_chunk_idx * INITIAL_HWD_CHUNK_SIZE;
    const int current_hwd_chunk = min(INITIAL_HWD_CHUNK_SIZE,
            INITIAL_HWD_DIM - hwd_chunk_idx * INITIAL_HWD_CHUNK_SIZE);

    // Limit the chunk size to stop early at a vector boundary
    const int aligned_hwd_chunk = ROUND_DOWN(current_hwd_chunk, VECT_DT_N);

    const int n_start = n_chunk_idx * INITIAL_N_CHUNK_SIZE;
    const int n_end = min(n_start + INITIAL_N_CHUNK_SIZE, INITIAL_N);

#if SKIP_FINAL_PHASE
    // zero pad dst memory
    for (int n_idx = n_start; n_idx < n_start + INITIAL_N_CHUNK_SIZE; n_idx++) {
        for (int c_idx = c; c_idx < c + INITIAL_C_CHUNKS * SUB_GROUP_SIZE;
                c_idx++) {
            if (n_idx >= DST_N && n_idx < DST_N_PADDED
                    || c_idx >= DST_C && c_idx < DST_C_PADDED) {
                for (int hwd_idx = hwd_start;
#if IS_HWD_REDUCED
                        hwd_idx < hwd_start + FINAL_HWD_CHUNK_SIZE;
#else
                        hwd_idx < hwd_start + INITIAL_HWD_CHUNK_SIZE;
#endif
                        hwd_idx++) {
                    int n = (!IS_C_REDUCED && IS_N_REDUCED && NDIMS == 3
                                    && DST_N_PADDED == 1)
                            ? 0
                            : n_idx;
                    const int dst_off = FINAL_DST_OFFSET(n, c_idx, hwd_idx);
                    dst[dst_off] = TO_DST(0.0f);
                }
            }
        }
    }
#endif

    int channel_id = c + get_sub_group_local_id();
    if (channel_id >= INITIAL_C || n_start >= INITIAL_N) { return; }

    VECT_DEF_ACC_DATA_T vector_acc = INIT_ACC;
    for (int n = n_start; n < n_end; n++) {
        for (int hwd_id = 0; hwd_id < aligned_hwd_chunk; hwd_id += VECT_DT_N) {
            for (int c_chunk = 0; c_chunk < INITIAL_C_CHUNKS; c_chunk++) {
                // It will always read from the beginning of c block
                const int off = INITIAL_SRC_OFFSET(n, c,
                        hwd_start + hwd_id
                                + c_chunk * VECT_DT_N / INITIAL_C_CHUNKS);
                VECT_DEF_ACC_DATA_T data
                        = AS_VECT_DEF_ACC_DATA_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                                (const __global BLOCK_DATA_T *)&src[off])));
                vector_acc = ACCUMULATE(vector_acc, data);
            }
        }

        for (int hwd_id = aligned_hwd_chunk; hwd_id < current_hwd_chunk;
                hwd_id++) {
            const int off = INITIAL_SRC_OFFSET(n, c, hwd_start + hwd_id);
            C_BLOCK_T data = CONVERT_C_BLOCK_T(AS_C_BLOCK_DATA_T(
                    C_BLOCK_READ((const __global BLOCK_DATA_T *)&src[off])));

#if VECT_DT_N == 1
            vector_acc = ACCUMULATE(vector_acc, data);
#else // VECT_DT_N == 1
#if INITIAL_C_CHUNKS == 1
            vector_acc[0] = ACCUMULATE(vector_acc[0], data);
#else
            // data[0] and data[i] must be accumulated separately as they contain different C
            for (int i = 0; i < INITIAL_C_CHUNKS; i++) {
                vector_acc[i] = ACCUMULATE(vector_acc[i], data[i]);
            }
#endif // INITIAL_C_CHUNKS == 1
#endif // VECT_DT_N == 1
        }
    }
#if VECT_DT_N == 1
    VECT_DEF_ACC_DATA_T acc = vector_acc;
#else // VECT_DT_N == 1
    const int elems_to_accumulate = aligned_hwd_chunk > 0 ? VECT_DT_N : 1;

#if INITIAL_C_CHUNKS == 1
    DEF_ACC_DATA_T acc = INIT_ACC;
    for (int i = 0; i < elems_to_accumulate; i++) {
        acc = ACCUMULATE_AGAIN(acc, vector_acc[i]);
    }
#else
    C_BLOCK_T acc = INIT_ACC;
    for (int i = 0; i < elems_to_accumulate; i += INITIAL_C_CHUNKS) {
        unroll_for(int j = 0; j < INITIAL_C_CHUNKS; j++) {
            acc[j] = ACCUMULATE_AGAIN(acc[j], vector_acc[i + j]);
        }
    }
#endif // INITIAL_C_CHUNKS == 1
#endif // VECT_DT_N == 1
    const int local_id = get_sub_group_local_id();
#if IS_C_REDUCED
#if INITIAL_C_CHUNKS == 1
    DEF_ACC_DATA_T c_acc = acc;
#else
    DEF_ACC_DATA_T c_acc = acc[0];
    for (int i = 1; i < INITIAL_C_CHUNKS; i++) {
        c_acc += acc[i];
    }
#endif // INITIAL_C_CHUNKS == 1
    const int dst_off
            = INITIAL_DST_OFFSET(n_chunk_idx, c_block_idx, hwd_chunk_idx);
    c_acc = SUB_GROUP_REDUCE(c_acc, c);
    if (local_id == 0) {
        WRITE_INITIAL_RESULT(dst[dst_off], c_acc, n_start, c, hwd_start);
    }
#else // IS_C_REDUCED
    const int dst_c = c + local_id;
#if INITIAL_C_CHUNKS == 1
    WRITE_INITIAL_RESULT(
            dst[INITIAL_DST_OFFSET(n_chunk_idx, dst_c, hwd_chunk_idx)], acc,
            n_start, dst_c, hwd_start);
#else // INITIAL_C_CHUNKS == 1
    for (int i = 0; i < INITIAL_C_CHUNKS; i++) {
        int c_off = i * SUB_GROUP_SIZE;
        WRITE_INITIAL_RESULT(dst[INITIAL_DST_OFFSET(n_chunk_idx, dst_c + c_off,
                                     hwd_chunk_idx)],
                acc[i], n_start, c, hwd_start);
    }
#endif // INITIAL_C_CHUNKS == 1
#endif // IS_C_REDUCED
}

#if !SKIP_FINAL_PHASE

// Finalizes reduction by reducing results of initial reduction
NAMED_KERNEL_ATTR(FINAL)
__kernel void gen9_final_reduce(
        __global DEF_ACC_DATA_T *src, __global DST_DATA_T *dst POST_OP_ARGS) {
    const int n_start = GWS_GET_FINAL_N() * FINAL_N_CHUNK_SIZE;
    const int c_start = GWS_GET_FINAL_C() * FINAL_C_CHUNK_SIZE;
    const int hwd_start = GWS_GET_FINAL_HWD() * FINAL_HWD_CHUNK_SIZE;

    DEF_ACC_DATA_T acc = INIT_ACC;
    const int max_n = max(DST_N_PADDED, FINAL_N_DIM);
    const int max_c = max(DST_C_PADDED, FINAL_C_DIM);
    const int n_end = min(max_n, n_start + FINAL_N_CHUNK_SIZE);
    const int c_end = min(max_c, c_start + FINAL_C_CHUNK_SIZE);
    const int hwd_end = min(FINAL_HWD_DIM, hwd_start + FINAL_HWD_CHUNK_SIZE);
    for (int n = n_start; n < n_end; n++) {
        for (int c = c_start; c < c_end; c++) {
            for (int hwd = hwd_start; hwd < hwd_end; hwd++) {
                // zero pad dst memory
                if ((n >= DST_N && n < DST_N_PADDED)
                        || (c >= DST_C && c < DST_C_PADDED)) {
#if NDIMS == 2 && DST_N_PADDED == 1 // all reduced case for 2D
                    const int dst_off = FINAL_DST_OFFSET(0, c, hwd);
#elif NDIMS >= 3 && DST_N_PADDED == 1 && IS_HWD_REDUCED // hwd, n reduction
                    const int dst_off = FINAL_DST_OFFSET(0, c, 0);
#elif IS_HWD_REDUCED // 4D, 5D, 3D cases with hwd reduced
                    const int dst_off = FINAL_DST_OFFSET(n, c, 0);
#else
                    const int dst_off = FINAL_DST_OFFSET(n, c, hwd);
#endif
                    dst[dst_off] = TO_DST(0.0f);
                }
                if (n < FINAL_N_DIM && c < FINAL_C_DIM) {
                    const int off = FINAL_SRC_OFFSET(n, c, hwd);
                    const DEF_ACC_DATA_T data = src[off];
                    acc = ACCUMULATE_AGAIN(acc, data);
                }
            }
        }
    }
    if (n_start < DST_N && c_start < DST_C) {
        const int off = FINAL_DST_OFFSET(n_start, c_start, hwd_start);

        WRITE_FINAL_RESULT(dst[off], acc, n_start, c_start, hwd_start);
    }
}
#endif // !SKIP_FINAL_PHASE
