/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#if DATA_TYPE_SIZE == 8
#define DATA_T ulong
#define BLOCK_READ intel_sub_group_block_read_ul
#define BLOCK_WRITE intel_sub_group_block_write_ul
#elif DATA_TYPE_SIZE == 4
#define DATA_T uint
#define BLOCK_READ intel_sub_group_block_read
#define BLOCK_WRITE intel_sub_group_block_write
#elif DATA_TYPE_SIZE == 2
#define DATA_T ushort
#define BLOCK_READ intel_sub_group_block_read_us
#define BLOCK_WRITE intel_sub_group_block_write_us
#elif DATA_TYPE_SIZE == 1
#define DATA_T uchar
#define BLOCK_READ intel_sub_group_block_read_uc
#define BLOCK_WRITE intel_sub_group_block_write_uc
#endif

#define INTERNAL_CAT(a, b) a##b
#define CAT(a, b) INTERNAL_CAT(a, b)
#define DIV_UP(a, b) ((a) + ((b)-1)) / (b)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define DATA2_T CAT(DATA_T, 2)
#define BLOCK_READ2 CAT(BLOCK_READ, 2)
#define BLOCK_WRITE2 CAT(BLOCK_WRITE, 2)

#define DATA4_T CAT(DATA_T, 4)
#define BLOCK_READ4 CAT(BLOCK_READ, 4)
#define BLOCK_WRITE4 CAT(BLOCK_WRITE, 4)

#define DATA8_T CAT(DATA_T, 8)
#define BLOCK_READ8 CAT(BLOCK_READ, 8)
#define BLOCK_WRITE8 CAT(BLOCK_WRITE, 8)

#define DATA16_T CAT(DATA_T, 16)
#if DATA_TYPE_SIZE == 1
#define BLOCK_READ16 CAT(BLOCK_READ, 16)
#define BLOCK_WRITE16 CAT(BLOCK_WRITE, 16)
#else
#define BLOCK_READ16(ptr) \
    (DATA16_T)((DATA8_T)(BLOCK_READ8(ptr)), \
            (DATA8_T)(BLOCK_READ8((ptr) + 8 * SIMD)))
#define BLOCK_WRITE16(ptr, v) \
    do { \
        DATA16_T tmp = v; \
        BLOCK_WRITE8((ptr), tmp.s01234567); \
        BLOCK_WRITE8((ptr) + 8 * SIMD, tmp.s89abcdef); \
    } while (0)
#endif

#define RW_RATIO DIV_UP(BLOCK, INNER_OFFSET)

#include "gpu/ocl/concat_common.h"

#define SRC_PTR(n) __global const DATA_T *src##n
#define SRC_PTRS REDUCE(N_INPUTS, JOIN_COMMA, SRC_PTR)
#define JOIN_ELSE(x, y) y else x

typedef union {
    DATA16_T v16[2];
    DATA8_T v8[4];
    DATA4_T v4[8];
    DATA2_T v2[16];
    DATA_T v1[32];
} buffer_t;

DATA_T load_vec1(const buffer_t *buf, size_t offset) {
    return buf->v1[offset];
}

DATA2_T load_vec2(const buffer_t *buf, size_t offset) {
    return offset & 0x1
            ? (DATA2_T)(load_vec1(buf, offset), load_vec1(buf, offset + 1))
            : buf->v2[offset / 2];
}

// XXX: Consider handling the special cases
//   offset & 0x1 -> (DATA4_T)(load_vec1(buf, offset),
//          load_vec2(buf, offset + 1), load_vec1(buf, offset + 3))
// (and similar for vec8/16)
DATA4_T load_vec4(const buffer_t *buf, size_t offset) {
    return offset & 0x3
            ? (DATA4_T)(load_vec2(buf, offset), load_vec2(buf, offset + 2))
            : buf->v4[offset / 4];
}

DATA8_T load_vec8(const buffer_t *buf, size_t offset) {
    return offset & 0x7
            ? (DATA8_T)(load_vec4(buf, offset), load_vec4(buf, offset + 4))
            : buf->v8[offset / 8];
}

DATA16_T load_vec16(const buffer_t *buf, size_t offset) {
    return offset & 0xf
            ? (DATA16_T)(load_vec8(buf, offset), load_vec8(buf, offset + 8))
            : buf->v16[offset / 16];
}

#if SIMD != 1
__attribute__((intel_reqd_sub_group_size(SIMD)))
#endif
__kernel void
simple_concat(__global DATA_T *dst, long dst_offset0, SRC_PTRS) {
    const uint x = (get_global_id(0) / SIMD) * BLOCK;
    __global const DATA_T *src;
    size_t src_ext_offset, src_offset, input_offset;

#define CHECK_AND_GET(n) \
    if (get_global_id(2) >= OFFSET##n) { \
        src_ext_offset = SRC##n##_EXT_OFFSET; \
        src_offset = RW_RATIO * (get_global_id(2) - OFFSET##n) * INNER_OFFSET; \
        src = src##n + RW_RATIO * get_global_id(1) * src_ext_offset + x \
                + src_offset; \
        input_offset = OFFSET##n; \
    }
    REDUCE(N_INPUTS, JOIN_ELSE, CHECK_AND_GET);
#undef CHECK_AND_GET

    const size_t thr_elems = BLOCK / SIMD;

#if BLOCK > INNER_OFFSET
#define DST_IDX(idx) \
    ((src_offset + idx) / src_ext_offset) * DST_EXT_OFFSET \
            + ((src_offset + idx) % src_ext_offset)
#else
#define DST_IDX(idx) src_offset + idx
#endif
#if SIMD == 1
    dst += dst_offset0 + RW_RATIO * get_global_id(1) * DST_EXT_OFFSET + x
            + input_offset * INNER_OFFSET;

    for (int i = 0; i < thr_elems; ++i)
        dst[DST_IDX(i)] = src[i];
#else
    const size_t lane = get_sub_group_local_id();
    buffer_t buf;

#if (BLOCK * DATA_TYPE_SIZE % 4 != 0)
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
    if (lane < BLOCK % SIMD) buf.v1[thr_elems] = src[thr_elems * SIMD + lane];

    dst += dst_offset0 + RW_RATIO * get_global_id(1) * DST_EXT_OFFSET + x
            + input_offset * INNER_OFFSET;

#if (MIN(BLOCK, INNER_OFFSET) * DATA_TYPE_SIZE % 16 != 0)
    for (int i = 0; i < thr_elems; ++i)
        dst[DST_IDX(i * SIMD + lane)] = buf.v1[i];
    if (lane < BLOCK % SIMD)
        dst[DST_IDX(thr_elems * SIMD + lane)] = buf.v1[thr_elems];
#else
    // Break up the data that was read into several blocks that may be written
    // sequentially. If the block size is not divisible by the subgroup size,
    // borrow values from the next block(s), if available, to fill a scattered
    // write.
    const size_t block_elems = MIN(BLOCK, INNER_OFFSET);
    const size_t elems_per_iteration = MAX(SIMD, block_elems);
    const size_t iterations = DIV_UP(BLOCK, elems_per_iteration);

#pragma unroll
    for (int j = 0; j < iterations; ++j) {
        const size_t buf_off = DIV_UP(j * elems_per_iteration, SIMD);
        const size_t block_off = buf_off * SIMD;
        // Accounting for any values borrowed from the last iteration, this
        // block only has `iter_elems` values to write:
        const size_t iter_elems = block_elems - (block_off % block_elems);
        const size_t thr_iter_elems = iter_elems / SIMD;
        __global DATA_T *iter_dst = dst + DST_IDX(block_off);
#define MAYBE_BLOCK_WRITE(n, write, elems) \
    do { \
        if ((elems) & (n)) { \
            const uint rel = (elems) & ~(((n) << 1) - 1); \
            write(&iter_dst[rel * SIMD], load_vec##n(&buf, buf_off + rel)); \
        } \
    } while (0)

        for (int i = 0; i < thr_iter_elems / 16; ++i)
            BLOCK_WRITE16(&iter_dst[16 * i * SIMD],
                    load_vec16(&buf, buf_off + i * 16));
        MAYBE_BLOCK_WRITE(8, BLOCK_WRITE8, thr_iter_elems);
        MAYBE_BLOCK_WRITE(4, BLOCK_WRITE4, thr_iter_elems);
        MAYBE_BLOCK_WRITE(2, BLOCK_WRITE2, thr_iter_elems);
        MAYBE_BLOCK_WRITE(1, BLOCK_WRITE, thr_iter_elems);
#undef MAYBE_BLOCK_WRITE

        // Write tail elements + the leading elements of the next block
        if (iter_elems % SIMD) {
            const size_t written = block_off + thr_iter_elems * SIMD;
            if (lane < MIN(SIMD, BLOCK - written))
                dst[DST_IDX(written + lane)] = buf.v1[buf_off + thr_iter_elems];
        }
    }
#endif // aligned for block writes
#endif // SIMD == 1
}
