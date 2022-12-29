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
#ifndef GPU_OCL_GEN9_BNORM_H
#define GPU_OCL_GEN9_BNORM_H

#define VECT_DT_N VECT_SIZE

#include "gpu/ocl/ocl_types.h"

#define IS_IC_EQ_8 (IC == 8)
#define HAS_IC_TAIL (IC != IC16)
#define HAS_STAT_SP_BLOCK_TAIL (SP % STAT_SP_BLOCK)

#if NHWC_OPTIMIZED

#if HAS_IC_TAIL
#error IC tail processing not supported
#endif

#else // NHWC_OPTIMIZED

#if HAS_IC_TAIL && !USE_NHWC
#error IC tail processing not supported
#endif
#define HAS_STAT_SP_TAIL (STAT_SP_TAIL != STAT_SP_NBLOCKS)
#define HAS_SP_TAIL (SP != SP_TAIL)

#endif // NHWC_OPTIMIZED

#define IC_BLOCK_SGROUPS (IC_BLOCK / 16)
#define IC_TAIL_SGROUPS (IC_BLOCK_SGROUPS % VECT_SIZE)
#define IC_VECT_SGROUPS (IC_BLOCK_SGROUPS - IC_TAIL_SGROUPS)
#define HAS_IC_VECT_TAIL (IC_TAIL_SGROUPS > 0)

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

#if HAS_IC_TAIL
#define MAYBE_LAST_IC_LOAD_FLOAT_1x16(ptr, idx) \
    (is_last_ic_block ? (simd_id < 8 ? ptr[(idx) + simd_id] : 0.0f) \
                      : as_float(intel_sub_group_block_read( \
                              (const __global uint *)(&ptr[(idx)]))))
#else
#define MAYBE_LAST_IC_LOAD_FLOAT_1x16(ptr, idx) LOAD_FLOAT_1x16(&ptr[(idx)])
#endif

#if USE_NHWC
#define IC_BLOCK_STRIDE IC
#else
#define IC_BLOCK_STRIDE 16
#endif

#if NHWC_OPTIMIZED
#define REDUCE_NUM_SGROUPS IC_BLOCK_SGROUPS
#else
#define REDUCE_NUM_SGROUPS 1
#endif

#define CALC_SLM_LINE_SIZE (REDUCE_NUM_SGROUPS * GWS_LWS0_CALC)
#define CALC_SLM_SIZE (CALC_SLM_LINE_SIZE * GWS_LWS1_CALC * GWS_LWS2_CALC)

#endif // GPU_OCL_GEN9_BNORM_H
