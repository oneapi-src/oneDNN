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

#if LARGE_INDEX_T == 1
#define IDX_T long
#else
#define IDX_T uint
#endif

#define ZERO_PAD_OFFSET CAT(offset, N_INPUTS)
#define ZERO_PAD_CONCAT_DIM CAT(padded_offset, N_INPUTS)

#include "gpu/intel/ocl/concat_common.h"

#define SRC_PARAM(n) \
    CS_PARAM(__global const DATA_T *src##n, const IDX_T src_ext_offset##n, \
            const IDX_T offset##n, const IDX_T padded_offset##n, \
            const IDX_T src_concat_axis##n)
#define SRC_PARAMS REDUCE(N_INPUTS, JOIN_COMMA, SRC_PARAM)

#if BLOCK_DEPTH > 0

#define BLOCK_LAYOUT_ARG(n) JOIN_COMMA(block_b##n, block_s##n)
#define BLOCK_LAYOUT_PARAM(n) \
    JOIN_COMMA(const IDX_T block_b##n, const IDX_T block_s##n)

#define BLOCK_LAYOUT_PARAMS REDUCE(BLOCK_DEPTH, JOIN_COMMA, BLOCK_LAYOUT_PARAM)
#define BLOCK_LAYOUT_ARGS REDUCE(BLOCK_DEPTH, JOIN_COMMA, BLOCK_LAYOUT_ARG)

#define CONCAT_AXIS(n) src_concat_axis##n
#define RIGHT(x, y) y

#endif

IDX_T get_concat_idx(IDX_T inner_idx, IDX_T inner_offset
#if BLOCK_DEPTH > 0
        ,
        BLOCK_LAYOUT_PARAMS
#endif
) {
    uint block = 1;
    uint idx = 0;
#define HANDLE(n) \
    idx += block * ((inner_idx / (block_s##n)) % (block_b##n)); \
    block *= block_b##n

    REDUCE(BLOCK_DEPTH, JOIN_SEMICOLON, HANDLE);
#undef HANDLE
    return idx + block * (inner_idx / inner_offset);
}

IDX_T get_concat_offset(IDX_T concat_idx, IDX_T inner_offset
#if BLOCK_DEPTH > 0
        ,
        BLOCK_LAYOUT_PARAMS
#endif
) {
    int block = 1;
    int idx = 0;
#define HANDLE(n) \
    idx += (block_s##n) * ((concat_idx / block) % (block_b##n)); \
    block *= block_b##n

    REDUCE(BLOCK_DEPTH, JOIN_SEMICOLON, HANDLE);
#undef HANDLE
    return idx + inner_offset * (concat_idx / block);
}

#define PARAM_OFFSET_INPUT_N(n) const IDX_T offset##n
#define ARG_OFFSET_INPUT_N(n) offset##n

#define EXPAND_N_ARGS(EXPAND_MACRO, n) \
    JOIN_COMMA(REDUCE(n, JOIN_COMMA, EXPAND_MACRO), EXPAND_MACRO(n))

#define EXPANDED_OFFSET_PARAMS EXPAND_N_ARGS(PARAM_OFFSET_INPUT_N, N_INPUTS)
#define EXPANDED_OFFSET_ARGS EXPAND_N_ARGS(ARG_OFFSET_INPUT_N, N_INPUTS)

struct write_info_t {
    IDX_T idx;
    bool write;
};

struct write_info_t get_write_info(
        const IDX_T src_idx, const IDX_T src_ext_offset,
        const IDX_T concat_offset, const IDX_T inner_offset,
        const IDX_T read_overlap, const IDX_T gws0_block,
        const IDX_T dst_ext_offset, EXPANDED_OFFSET_PARAMS
#if BLOCK_DEPTH > 0
        ,
        const IDX_T REDUCE(N_INPUTS, RIGHT, CONCAT_AXIS),
        const IDX_T ZERO_PAD_CONCAT_DIM, const IDX_T thread_offset,
        const IDX_T concat_axis, BLOCK_LAYOUT_PARAMS
#endif
) {
#define LOGICAL_OR(x, y) (x) || (y)
#define HANDLE(n) CAT(offset, n) % read_overlap != 0
#define CUTOFF REDUCE(N_INPUTS, LOGICAL_OR, HANDLE) || HANDLE(N_INPUTS)

    IDX_T inner_idx, ext_idx;
    if (read_overlap * gws0_block > inner_offset || CUTOFF) {
        // short circuit logic avoids significant assembly bloat
        // in the special case
        ext_idx = (src_ext_offset == 1) ? src_idx : src_idx / src_ext_offset;
        inner_idx = (src_ext_offset == 1) ? 0 : src_idx % src_ext_offset;
    } else {
        ext_idx = 0;
        inner_idx = src_idx;
    }

#undef CUTOFF
#undef HANDLE
#undef LOGICAL_OR

    struct write_info_t info;

#if BLOCK_DEPTH > 0
    IDX_T concat_idx
            = get_concat_idx(inner_idx, inner_offset, BLOCK_LAYOUT_ARGS);
    bool write_value = concat_offset + concat_idx < concat_axis;

    IDX_T write_offset;
    if (REDUCE(N_INPUTS, RIGHT, CONCAT_AXIS) < ZERO_PAD_CONCAT_DIM) {
        IDX_T zero_pad_offset = ZERO_PAD_OFFSET - concat_axis + thread_offset;
        bool write_zeropad = zero_pad_offset + concat_idx < ZERO_PAD_CONCAT_DIM;

        write_offset = write_value ? concat_offset : zero_pad_offset;
        info.write = write_value || write_zeropad;
    } else {
        write_offset = concat_offset;
        info.write = write_value;
    }
    info.idx = ext_idx * dst_ext_offset + inner_idx
            + get_concat_offset(
                    concat_idx + write_offset, inner_offset, BLOCK_LAYOUT_ARGS)
            - get_concat_offset(concat_idx, inner_offset, BLOCK_LAYOUT_ARGS);
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
reusable_simple_concat(
        __global DATA_T *dst, const long dst_offset0, const long dst_ext_offset,
        SRC_PARAMS, const IDX_T CAT(offset, N_INPUTS),
        const IDX_T CAT(padded_offset, N_INPUTS), const IDX_T inner_offset,
        const IDX_T read_overlap, const IDX_T gws0_block
#if BLOCK_DEPTH > 0
        ,
        BLOCK_LAYOUT_PARAMS
#endif
) {
    __global const DATA_T *src;
    IDX_T src_ext_offset, input_offset;
    IDX_T input_padded_offset, concat_axis_size;

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

    const IDX_T block_offset
            = gws0_block * (get_global_id(2) - input_padded_offset)
            + (get_global_id(0) / SIMD) * READ_BLOCK;
    const uint thr_elems = READ_BLOCK / SIMD;
    src += read_overlap * get_global_id(1) * src_ext_offset + block_offset;

#if BLOCK_DEPTH > 0
#define WRITE_INFO(idx) \
    get_write_info(block_offset + (idx), src_ext_offset, input_offset, \
            inner_offset, read_overlap, gws0_block, dst_ext_offset, \
            EXPANDED_OFFSET_ARGS, REDUCE(N_INPUTS, RIGHT, CONCAT_AXIS), \
            ZERO_PAD_CONCAT_DIM, input_padded_offset, concat_axis_size, \
            BLOCK_LAYOUT_ARGS)
#else
#define WRITE_INFO(idx) \
    get_write_info(block_offset + (idx), src_ext_offset, input_offset, \
            inner_offset, read_overlap, gws0_block, dst_ext_offset, \
            EXPANDED_OFFSET_ARGS)
#endif

#if SIMD == 1
    dst += dst_offset0 + read_overlap * get_global_id(1) * dst_ext_offset;
    for (int i = 0; i < thr_elems; ++i) {
        struct write_info_t info = WRITE_INFO(i);
        if (info.write) dst[info.idx] = src[i];
    }
#else
    const uint lane = get_sub_group_local_id() % SIMD;
    buffer_t buf;

#if (READ_BLOCK * DATA_TYPE_SIZE % 4 != 0)
    for (int i = 0; i < thr_elems; ++i)
        buf.v1[i] = src[i * SIMD + lane];
#else
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
#endif // aligned for block reads
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

#pragma unroll
        for (int j = 0; j < iterations; ++j) {
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
