/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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
#ifndef GPU_INTEL_OCL_BNORM_NHWC_REUSABLE_H
#define GPU_INTEL_OCL_BNORM_NHWC_REUSABLE_H

#define MAX_IC_BLOCK_SGROUPS (MAX_IC_BLOCK / SUB_GROUP_SIZE)
#define MAX_IC_TAIL_SGROUPS (VECT_SIZE - 1)
#define MAY_HAVE_IC_TAIL (MAX_IC_TAIL_SGROUPS > 0)

#define VECT_DT_N VECT_SIZE
#include "gpu/intel/ocl/dispatch.h"
#include "gpu/intel/ocl/ocl_types.h"

#define LOAD_FLOAT_1x16(ptr) \
    as_float(intel_sub_group_block_read((const __global uint *)(ptr)))

#define LOAD_UINT_1x16(ptr) \
    as_uint(intel_sub_group_block_read((const __global uint *)(ptr)))

#define LOAD_UINT_8x16(ptr) \
    convert_uint8(as_uint8( \
            intel_sub_group_block_read8((const __global uint *)(ptr))))

#define LOAD_CHAR_1x16(ptr) \
    as_char(intel_sub_group_block_read_uc((const __global uchar *)(ptr)))

#define LOAD_CHAR_8x16(ptr) \
    convert_char8(as_char8( \
            intel_sub_group_block_read_uc8((const __global uchar *)(ptr))))

#define LOAD_DATA_1x16(ptr) \
    CONVERT_FLOAT_T(AS_DATA_T(BLOCK_READ((const __global BLOCK_DATA_T *)(ptr))))

#define LOAD_DATA_8x16(ptr) \
    CONVERT_FLOAT8_T( \
            AS_DATA8_T(BLOCK_READ8((const __global BLOCK_DATA_T *)(ptr))))

#define LOAD_VECT_DATA(ptr) \
    CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T( \
            VECT_BLOCK_READ((const __global BLOCK_DATA_T *)(ptr))))

#define LOAD_VECT_CHAR(ptr) \
    CONVERT_VECT_CHAR_T( \
            AS_VECT_CHAR_T(VECT_UCHAR_READ((const __global uchar *)(ptr))))

#define LOAD_VECT_FLOAT(ptr) \
    AS_VECT_FLOAT_T(VECT_UINT_READ((const __global uint *)(ptr)))

#define STORE_DATA_1x16(ptr, val) \
    BLOCK_WRITE((__global BLOCK_DATA_T *)(ptr), \
            AS_BLOCK_DATA_T(CONVERT_DATA_T(val)))

#define STORE_DATA_8x16(ptr, val) \
    BLOCK_WRITE8((__global BLOCK_DATA_T *)ptr, \
            AS_BLOCK_DATA8_T(CONVERT_DATA8_T(val)))

#define STORE_VECT_DATA(ptr, val) \
    VECT_BLOCK_WRITE((__global BLOCK_DATA_T *)(ptr), \
            AS_VECT_BLOCK_DATA_T(CONVERT_VECTOR_DATA_T(val)))

#define STORE_FLOAT_1x16(ptr, val) \
    intel_sub_group_block_write((__global uint *)(ptr), as_uint(val))

#define STORE_FLOAT_8x16(ptr, val) \
    intel_sub_group_block_write8((__global uint *)(ptr), as_uint8(val))

#define STORE_CHAR_1x16(ptr, val) \
    intel_sub_group_block_write_uc((__global uchar *)(ptr), as_uchar(val))

#define STORE_CHAR_8x16(ptr, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(ptr), as_uchar8(val))

#define STORE_VECT_CHAR(ptr, val) \
    VECT_UCHAR_WRITE((__global uchar *)(ptr), \
            AS_VECT_UCHAR_T(CONVERT_VECT_CHAR_T(val)))

// To align result for scalar and vector type of arguments
#if VECT_DT_N == 1
#define ISGREATER(x, y) ((x) > (y) ? -1 : 0)
#else
#define ISGREATER(x, y) isgreater((x), (y))
#endif

#define ACCUM_DATA_T float
#define ACCUM_DATA8_T float8
#define ACCUM_DATA2_T float2
#define SUM_DATA_T ACCUM_DATA2_T

#define AS_VECT_FLOAT(a) *(VECT_FLOAT_T *)(a)

// Kahan summation algorithm. It's much more precise than simple sum and works
// just as fast, since kernel is still memory-bound.
SUM_DATA_T summation(ACCUM_DATA_T input, SUM_DATA_T state) {
    ACCUM_DATA2_T ret;
    ACCUM_DATA_T y = input - state.s1;
    ACCUM_DATA_T t = state.s0 + y;
    ret.s1 = (t - state.s0) - y;
    ret.s0 = t;
    return ret;
}
#endif // GPU_INTEL_OCL_BNORM_NHWC_REUSABLE_H
