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
#define DIV_UP(a, b) (((a) + ((b)-1)) / (b))
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

#define ZERO_PAD_OFFSET CAT(OFFSET, N_INPUTS)
#define ZERO_PAD_CONCAT_DIM CAT(PADDED_OFFSET, N_INPUTS)

#include "gpu/ocl/concat_common.h"

#define SRC_PTR(n) __global const DATA_T *src##n
#define SRC_PTRS REDUCE(N_INPUTS, JOIN_COMMA, SRC_PTR)
#define JOIN_ELSE(x, y) y else x
#define JOIN_SEMICOLON(x, y) \
    x; \
    y

#define DATA1_T DATA_T
#define VECTOR(n) DATA##n##_T v##n[DIV_UP(READ_BLOCK, n)]
typedef union {
    VECTOR(16);
    VECTOR(8);
    VECTOR(4);
    VECTOR(2);
    VECTOR(1);
} buffer_t;
#undef VECTOR
#undef DATA1_T

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

size_t get_concat_idx(size_t inner_idx) {
    size_t block = 1;
    size_t idx = 0;
#define HANDLE(n) \
    idx += block * ((inner_idx / (BLOCK_S##n)) % (BLOCK_B##n)); \
    block *= BLOCK_B##n

    REDUCE(BLOCK_DEPTH, JOIN_SEMICOLON, HANDLE);
#undef HANDLE
    return idx + block * (inner_idx / INNER_OFFSET);
}

size_t get_concat_offset(size_t concat_idx) {
    size_t block = 1;
    size_t idx = 0;
#define HANDLE(n) \
    idx += (BLOCK_S##n) * ((concat_idx / block) % (BLOCK_B##n)); \
    block *= BLOCK_B##n

    REDUCE(BLOCK_DEPTH, JOIN_SEMICOLON, HANDLE);
#undef HANDLE
    return idx + INNER_OFFSET * (concat_idx / block);
}

struct write_info_t {
    size_t idx;
    bool write;
};

struct write_info_t get_write_info(size_t src_idx, size_t src_ext_offset,
        size_t concat_offset, size_t thread_offset, size_t concat_axis) {
#define LOGICAL_OR(x, y) (x) || (y)
#define HANDLE(n) CAT(OFFSET, n) % READ_OVERLAP != 0
#define CUTOFF REDUCE(N_INPUTS, LOGICAL_OR, HANDLE) || HANDLE(N_INPUTS)
#if READ_OVERLAP * GWS0_BLOCK > INNER_OFFSET || CUTOFF
    const size_t ext_idx = src_idx / src_ext_offset;
    const size_t inner_idx = src_idx % src_ext_offset;
#else
    const size_t ext_idx = 0;
    const size_t inner_idx = src_idx;
#endif
#undef CUTOFF
#undef HANDLE
#undef LOGICAL_OR

    struct write_info_t info;
#if BLOCK_DEPTH > 0
    size_t concat_idx = get_concat_idx(inner_idx);
    bool write_value = concat_offset + concat_idx < concat_axis;

#define CONCAT_AXIS(n) SRC##n##_CONCAT_AXIS
#define RIGHT(x, y) y
#if REDUCE(N_INPUTS, RIGHT, CONCAT_AXIS) < ZERO_PAD_CONCAT_DIM
    size_t zero_pad_offset = ZERO_PAD_OFFSET - concat_axis + thread_offset;
    bool write_zeropad = zero_pad_offset + concat_idx < ZERO_PAD_CONCAT_DIM;

    size_t write_offset = write_value ? concat_offset : zero_pad_offset;
    info.write = write_value || write_zeropad;
#else
    size_t write_offset = concat_offset;
    info.write = write_value;
#endif
#undef RIGHT
#undef CONCAT_AXIS

    info.idx = ext_idx * DST_EXT_OFFSET + inner_idx
            + get_concat_offset(concat_idx + write_offset)
            - get_concat_offset(concat_idx);
#else
    info.write = true;
    info.idx = ext_idx * DST_EXT_OFFSET + INNER_OFFSET * concat_offset
            + inner_idx;
#endif
    return info;
}

#if SIMD != 1
__attribute__((intel_reqd_sub_group_size(SIMD)))
#endif
__kernel void
simple_concat(__global DATA_T *dst, long dst_offset0, SRC_PTRS) {
    __global const DATA_T *src;
    size_t src_ext_offset, input_offset;
    size_t input_padded_offset, concat_axis_size;

#define CHECK_AND_GET(n) \
    if (get_global_id(2) >= PADDED_OFFSET##n) { \
        src = src##n; \
        input_offset = OFFSET##n; \
        input_padded_offset = PADDED_OFFSET##n; \
        concat_axis_size = SRC##n##_CONCAT_AXIS; \
        src_ext_offset = SRC##n##_EXT_OFFSET; \
    }
    REDUCE(N_INPUTS, JOIN_ELSE, CHECK_AND_GET);
#undef CHECK_AND_GET

    const size_t block_offset
            = GWS0_BLOCK * (get_global_id(2) - input_padded_offset)
            + (get_global_id(0) / SIMD) * READ_BLOCK;
    const size_t thr_elems = READ_BLOCK / SIMD;
    src += READ_OVERLAP * get_global_id(1) * src_ext_offset + block_offset;

#define WRITE_INFO(idx) \
    get_write_info(block_offset + (idx), src_ext_offset, input_offset, \
            input_padded_offset, concat_axis_size)
#if SIMD == 1
    dst += dst_offset0 + READ_OVERLAP * get_global_id(1) * DST_EXT_OFFSET;
    for (int i = 0; i < thr_elems; ++i) {
        struct write_info_t info = WRITE_INFO(i);
        if (info.write) dst[info.idx] = src[i];
    }
#else
    const size_t lane = get_sub_group_local_id() % SIMD;
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

    dst += dst_offset0 + READ_OVERLAP * get_global_id(1) * DST_EXT_OFFSET;
#if (WRITE_BLOCK * DATA_TYPE_SIZE % 16 != 0)
    for (int i = 0; i < thr_elems; ++i) {
        struct write_info_t info = WRITE_INFO(i * SIMD + lane);
        if (info.write) dst[info.idx] = buf.v1[i];
    }
    if (lane < READ_BLOCK % SIMD) {
        struct write_info_t info = WRITE_INFO(thr_elems * SIMD + lane);
        if (info.write) dst[info.idx] = buf.v1[thr_elems];
    }
#else
    // Break up the data that was read into several blocks that may be written
    // sequentially. If the block size is not divisible by the subgroup size,
    // borrow values from the next block(s), if available, to fill a scattered
    // write.
    const size_t elems_per_iteration = MAX(SIMD, WRITE_BLOCK);
    const size_t iterations = DIV_UP(READ_BLOCK, elems_per_iteration);

#pragma unroll
    for (int j = 0; j < iterations; ++j) {
        const size_t buf_off = DIV_UP(j * elems_per_iteration, SIMD);
        const size_t block_off = buf_off * SIMD;
        // Accounting for any values borrowed from the last iteration, this
        // block only has `iter_elems` values to write:
        const size_t iter_elems = WRITE_BLOCK - (block_off % WRITE_BLOCK);
        const size_t thr_iter_elems = iter_elems / SIMD;
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
            const size_t written = block_off + thr_iter_elems * SIMD;
            struct write_info_t info = WRITE_INFO(written + lane);
            if (info.write && lane < MIN(SIMD, READ_BLOCK - written))
                dst[info.idx] = buf.v1[buf_off + thr_iter_elems];
        }
    }
#endif // aligned for block writes
#endif // SIMD == 1
}
