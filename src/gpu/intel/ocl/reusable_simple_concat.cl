/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#if USE_LARGE_INDEX == 1
typedef long idx_t;
#else
typedef int idx_t;
#endif

#include "gpu/intel/ocl/concat_common.h"
#define unroll_for __attribute__((opencl_unroll_hint)) for

#define SRC_PARAM(n) \
    CS_PARAM(__global const DATA_T *src##n, const idx_t src_ext_offset##n, \
            const idx_t offset##n, const idx_t padded_offset##n, \
            const idx_t src_concat_axis##n)
#define SRC_PARAMS REDUCE(N_INPUTS, JOIN_COMMA, SRC_PARAM)

idx_t get_concat_idx(idx_t inner_idx, idx_t inner_offset) {
    idx_t block = 1;
    idx_t idx = 0;
#define HANDLE(n) \
    idx += block * ((inner_idx / (BLOCK_S##n)) % (BLOCK_B##n)); \
    block *= BLOCK_B##n

    REDUCE(BLOCK_DEPTH, JOIN_SEMICOLON, HANDLE);
#undef HANDLE
    return idx + block * (inner_idx / inner_offset);
}

idx_t get_concat_offset(idx_t concat_idx, idx_t inner_offset) {
    idx_t block = 1;
    idx_t idx = 0;
#define HANDLE(n) \
    idx += (BLOCK_S##n) * ((concat_idx / block) % (BLOCK_B##n)); \
    block *= BLOCK_B##n

    REDUCE(BLOCK_DEPTH, JOIN_SEMICOLON, HANDLE);
#undef HANDLE
    return idx + inner_offset * (concat_idx / block);
}

struct write_info_t {
    idx_t idx;
    bool write;
};

struct write_info_t get_write_info(const idx_t src_idx,
        const idx_t src_ext_offset, const idx_t concat_offset,
        const idx_t dst_ext_offset, const idx_t thread_offset,
        const idx_t concat_axis, const idx_t last_concat_axis,
        const idx_t zero_pad_offset, const idx_t zero_pad_concat_axis,
        const idx_t inner_offset, bool must_compute_ext_idx) {

    idx_t inner_idx, ext_idx;
    if (must_compute_ext_idx) {
        // short circuit logic avoids significant assembly bloat
        // in the special case
        ext_idx = (src_ext_offset == 1) ? src_idx : src_idx / src_ext_offset;
        inner_idx = (src_ext_offset == 1) ? 0 : src_idx % src_ext_offset;
    } else {
        ext_idx = 0;
        inner_idx = src_idx;
    }

    struct write_info_t info;

#if BLOCK_DEPTH > 0
    idx_t concat_idx = get_concat_idx(inner_idx, inner_offset);
    bool write_value = concat_offset + concat_idx < concat_axis;

    idx_t write_offset;
    if (last_concat_axis < zero_pad_concat_axis) {
        idx_t zero_pad_start = zero_pad_offset - concat_axis + thread_offset;
        bool write_zeropad = zero_pad_start + concat_idx < zero_pad_concat_axis;

        write_offset = write_value ? concat_offset : zero_pad_start;
        info.write = write_value || write_zeropad;
    } else {
        write_offset = concat_offset;
        info.write = write_value;
    }
    info.idx = ext_idx * dst_ext_offset + inner_idx
            + get_concat_offset(concat_idx + write_offset, inner_offset)
            - get_concat_offset(concat_idx, inner_offset);
#else

    info.write = true;
    info.idx = ext_idx * dst_ext_offset + inner_offset * concat_offset
            + inner_idx;

#endif
    return info;
}

#if SIMD != 1
__attribute__((intel_reqd_sub_group_size(SIMD)))
#endif
__kernel void
reusable_simple_concat(__global DATA_T *dst, const ulong dst_offset0,
        const ulong dst_ext_offset, SRC_PARAMS, const idx_t zero_pad_offset,
        const idx_t zero_pad_concat_axis, const idx_t read_overlap,
        const idx_t gws0_block, const idx_t inner_offset,
        const uchar must_compute_ext_idx) {
    __global const DATA_T *src;
    idx_t src_ext_offset, input_offset;
    idx_t input_padded_offset, concat_axis_size;

#define CHECK_AND_GET(n) \
    if (get_global_id(2) >= padded_offset##n) { \
        src = src##n; \
        input_offset = offset##n; \
        input_padded_offset = padded_offset##n; \
        concat_axis_size = src_concat_axis##n; \
        src_ext_offset = src_ext_offset##n; \
    }
    REDUCE(N_INPUTS, JOIN_ELSE, CHECK_AND_GET);
#undef CHECK_AND_GET

    const idx_t block_offset
            = gws0_block * (get_global_id(2) - input_padded_offset)
            + (get_global_id(0) / SIMD) * READ_BLOCK;
    const idx_t thr_elems = READ_BLOCK / SIMD;
    src += read_overlap * get_global_id(1) * src_ext_offset + block_offset;

#define CONCAT_AXIS(n) src_concat_axis##n
#define RIGHT(x, y) y
#define WRITE_INFO(idx) \
    get_write_info(block_offset + (idx), src_ext_offset, input_offset, \
            dst_ext_offset, input_padded_offset, concat_axis_size, \
            REDUCE(N_INPUTS, RIGHT, CONCAT_AXIS), zero_pad_offset, \
            zero_pad_concat_axis, inner_offset, must_compute_ext_idx)

#if SIMD == 1
    dst += dst_offset0 + read_overlap * get_global_id(1) * dst_ext_offset;
    for (int i = 0; i < thr_elems; ++i) {
        struct write_info_t info = WRITE_INFO(i);
        if (info.write) dst[info.idx] = src[i];
    }
#else
    const uint lane = get_sub_group_local_id() % SIMD;
    buffer_t buf;

    if (READ_BLOCK * DATA_TYPE_SIZE % 4 != 0) {
        for (int i = 0; i < thr_elems; ++i)
            buf.v1[i] = src[i * SIMD + lane];
    } else {
#define MAYBE_BLOCK_READ(n, read, elems) \
    do { \
        if ((elems) & (n)) { \
            const uint rel = (elems) & ~(((n) << 1) - 1); \
            buf.v##n[rel / (n)] = read(&src[rel * SIMD]); \
        } \
    } while (0)

        for (int i = 0; i < thr_elems / 16; ++i)
            buf.v16[i] = BLOCK_READ16(&src[16 * i * SIMD]);
        MAYBE_BLOCK_READ(8, BLOCK_READ8, thr_elems);
        MAYBE_BLOCK_READ(4, BLOCK_READ4, thr_elems);
        MAYBE_BLOCK_READ(2, BLOCK_READ2, thr_elems);
        MAYBE_BLOCK_READ(1, BLOCK_READ, thr_elems);
#undef MAYBE_BLOCK_READ
    }
    if (lane < READ_BLOCK % SIMD)
        buf.v1[thr_elems] = src[thr_elems * SIMD + lane];

    dst += dst_offset0 + read_overlap * get_global_id(1) * dst_ext_offset;
    if (WRITE_BLOCK * DATA_TYPE_SIZE % 16 != 0) {
        for (int i = 0; i < thr_elems; ++i) {
            struct write_info_t info = WRITE_INFO(i * SIMD + lane);
            if (info.write) dst[info.idx] = buf.v1[i];
        }
        if (lane < READ_BLOCK % SIMD) {
            struct write_info_t info = WRITE_INFO(thr_elems * SIMD + lane);
            if (info.write) dst[info.idx] = buf.v1[thr_elems];
        }
    } else {
        // Break up the data that was read into several blocks that may be written
        // sequentially. If the block size is not divisible by the subgroup size,
        // borrow values from the next block(s), if available, to fill a scattered
        // write.
        const uint elems_per_iteration = MAX(SIMD, WRITE_BLOCK);
        const uint iterations = DIV_UP(READ_BLOCK, elems_per_iteration);

        unroll_for(int j = 0; j < iterations; ++j) {
            const uint buf_off = DIV_UP(j * elems_per_iteration, SIMD);
            const uint block_off = buf_off * SIMD;
            // Accounting for any values borrowed from the last iteration, this
            // block only has `iter_elems` values to write:
            const uint iter_elems = WRITE_BLOCK - (block_off % WRITE_BLOCK);
            const uint thr_iter_elems = iter_elems / SIMD;
            struct write_info_t info = WRITE_INFO(block_off);
            __global DATA_T *iter_dst = dst + info.idx;
#define MAYBE_BLOCK_WRITE(n, write, elems) \
    do { \
        if ((elems) & (n)) { \
            const uint rel = (elems) & ~(((n) << 1) - 1); \
            write(&iter_dst[rel * SIMD], load_vec##n(&buf, buf_off + rel)); \
        } \
    } while (0)

            if (info.write) {
                for (int i = 0; i < thr_iter_elems / 16; ++i)
                    BLOCK_WRITE16(&iter_dst[16 * i * SIMD],
                            load_vec16(&buf, buf_off + i * 16));
                MAYBE_BLOCK_WRITE(8, BLOCK_WRITE8, thr_iter_elems);
                MAYBE_BLOCK_WRITE(4, BLOCK_WRITE4, thr_iter_elems);
                MAYBE_BLOCK_WRITE(2, BLOCK_WRITE2, thr_iter_elems);
                MAYBE_BLOCK_WRITE(1, BLOCK_WRITE, thr_iter_elems);
            }
#undef MAYBE_BLOCK_WRITE

            // Write tail elements + the leading elements of the next block
            if (iter_elems % SIMD) {
                const uint written = block_off + thr_iter_elems * SIMD;
                struct write_info_t info = WRITE_INFO(written + lane);
                if (info.write && lane < MIN(SIMD, READ_BLOCK - written))
                    dst[info.idx] = buf.v1[buf_off + thr_iter_elems];
            }
        }
    }
#endif // SIMD == 1
}
