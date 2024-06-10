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

// DATATYPE SPECIALIZATION MECHANISM
// ===
//
// NOTE: This block describes the general function of this header driven by
// host-configured macros prefixed with DT_, SRC_DT, and DST_DT_. See their
// logic, below. The header provides other miscellaneous functions whose
// documentation and generalization is an open task.
// ---
//
// This header enables individual OpenCL kernels to generalize across datatypes
// by defining macros as stand-ins for types, conversions, and constants. Based
// on a given configuration macro (one of DT_F32, DT_F64, etc - see
// kernel_ctx_t::set_data_type), this section defines families of macros that
// each correspond to an abstract type and whose members define their associated
// properties. These abstract types (and base names of macro families) are:
//
//   - DATA: A data type corresponding to the configuration macro (DT_F32,
//       DT_F64, etc). In practice, DATA is made to represent the source data
//       type of the kernel.
//
//   - DEF_ACC: A data type suitable for accumulation over types of DATA.
//
//   - FLT_ACC: A strictly floating point version of DEF_ACC.
//
//   - POST_OP: A data type by convention for performing internal operations on
//       outputs after the main function of kernels.
//
//   - REF: A data type suitable for strictly kernel-internal storage.
//
// Individual macros derive from these base names and define elements of the
// corresponding types per the given configuration macro (DT_F32, etc). The
// macros are loosely standardized, but the following table of common
// definitions illustrates their "spirit":
//
//   MACRO DEFS VS CONFIG MACRO
//   ===
//            OpenCL    DATA_T  DEF_ACC_DATA_T  TO_DATA_T                           DATA_TO_REF
//   DT_F32   float     float   float           (float)(v)                          convert_float
//   DT_F64   double    double  double          (double)(v)                         convert_float
//   DT_F16   half      half    float           convert_half                        convert_half
//   DT_BF16  bfloat16  ushort  float           cvt_f32_to_bf16                     cvt_bf16_to_f32
//   DT_BF8   bf8       uchar   float           cvt_hf_to_f8_e5m2(convert_half(v))  convert_float(cvt_f8_e5m2_to_hf(v))
//   DT_HF8   hf8       uchar   float           cvt_hf_to_f8_e4m3(convert_half(v))  convert_float(cvt_f8_e4m3_to_hf(v))
//   DT_S8    char      char    int             convert_char_sat_rte(v)             convert_float
//   DT_U8    uchar     uchar   int             convert_uchar_sat_rte(v)            convert_float
//   DT_S32   int       int     int             convert_int_sat_rte                 convert_float
//
// Similar mechanisms for source and destination data types are controlled by
// SRC_DT_ and DST_DT_ prefixed macros. Critically, these mechanisms define
// SRC_DATA_T and DST_DATA_T types along with the conversion macros TO_DST and
// TO_SRC. These are crucial macros utilized throughout oneDNN's OpenCL.
//
// Example: See reusable_softmax.cl concisely demonstrate device-side use of
//          these mechanisms.
//
// HOST-SIDE IMPLEMENTATION
//
// oneDNN builds kernels after configuring the above mechanisms with host-side
// functions:
//
//   - kernel_ctx_t::set_data_type() (see src/gpu/compute/kernel_ctx.hpp)
//     - sets one of macro DT_F32, DT_F64, etc
//     - typically set to kernel input type (matching def_data_type(.., "SRC))
//
//   - def_data_type(.., "SRC" or "DST")
//     - sets one of SRC_DT_U8, SRC_DT_S8, etc (same for DST)
//     - header sets SRC_DATA_T and DST_DATA_T
//     - header sets TO_SRC(), TO_DST() conversions
//     - TO_DST() is very often utilized for final output to global memory

#ifndef GPU_INTEL_OCL_OCL_TYPES_H
#define GPU_INTEL_OCL_OCL_TYPES_H

#include "gpu/intel/ocl/ocl_custom_types.h"
#include "gpu/intel/ocl/ocl_math_utils.h"
#include "gpu/intel/ocl/ocl_utils.h"

#define auto __auto_type
#define typeof(x) __typeof__(x)

#define unroll_for __attribute__((opencl_unroll_hint)) for
#define unroll_for_by(factor) __attribute__((opencl_unroll_hint(factor))) for
#define unroll_2_for unroll_for_by(2)
#define unroll_4_for unroll_for_by(4)
#define unroll_8_for unroll_for_by(8)
#define unroll_16_for unroll_for_by(16)

#define for_ for

#if defined(DT_F16) || defined(SRC_DT_F16) || defined(SRC0_DT_F16) \
        || defined(SRC1_DT_F16) || defined(DST_DT_F16) || defined(WEI_DT_F16) \
        || defined(BIA_DT_F16) || defined(ACC_DT_F16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if DT_F64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#if DT_F64 == 1
#define AS_POST_OP_DATA_T(v) v
#else
#define AS_POST_OP_DATA_T(v) (float)(v)
#endif

#if DT_F32 == 1
#define DATA_T float
#define DATA2_T float2
#define DATA4_T float4
#define DATA8_T float8
#define DATA16_T float16
#define DATA_MAX FLT_MAX
#define DATA_MIN -DATA_MAX
#define DATA_ZERO 0.0f
#define DATA_ONE 1.0f
#define DEF_ACC_DATA_T float
#define DEF_ACC_DATA2_T float2
#define DEF_ACC_DATA4_T float4
#define DEF_ACC_DATA8_T float8
#define POST_OP_DATA_T float
#define TO_DATA_T(v) (float)(v)
#define TO_DEF_ACC_DATA_T convert_float
#define TO_DEF_ACC_DATA2_T convert_float2
#define TO_DEF_ACC_DATA4_T convert_float4
#define TO_DEF_ACC_DATA8_T convert_float8
#define DATA_TO_REF convert_float
#define CONVERT_DATA_T convert_float
#define CONVERT_DATA2_T convert_float2
#define CONVERT_DATA4_T convert_float4
#define CONVERT_DATA8_T convert_float8
#define CONVERT_FLOAT_T convert_float
#define CONVERT_FLOAT2_T convert_float2
#define CONVERT_FLOAT4_T convert_float4
#define CONVERT_FLOAT8_T convert_float8

#define BLOCK_READ intel_sub_group_block_read
#define BLOCK_WRITE intel_sub_group_block_write
#define BLOCK_READ2 intel_sub_group_block_read2
#define BLOCK_READ4 intel_sub_group_block_read4
#define BLOCK_READ8 intel_sub_group_block_read8
#define BLOCK_WRITE2 intel_sub_group_block_write2
#define BLOCK_WRITE4 intel_sub_group_block_write4
#define BLOCK_WRITE8 intel_sub_group_block_write8

#define AS_DATA_T as_float
#define AS_DATA2_T as_float2
#define AS_DATA4_T as_float4
#define AS_DATA8_T as_float8

#define AS_UINT_T as_uint
#define AS_UINT2_T as_uint2
#define AS_UINT4_T as_uint4
#define AS_UINT8_T as_uint8

#define BLOCK_DATA_T uint
#define BLOCK_DATA2_T uint2
#define BLOCK_DATA4_T uint4
#define BLOCK_DATA8_T uint8
#define AS_BLOCK_DATA_T as_uint
#define AS_BLOCK_DATA2_T as_uint2
#define AS_BLOCK_DATA4_T as_uint4
#define AS_BLOCK_DATA8_T as_uint8

#define FLT_ACC_DATA_T float
#define TO_FLT_ACC_DATA_T convert_float

#elif DT_F64 == 1
#define DATA_T double
#define DATA2_T double2
#define DATA4_T double4
#define DATA8_T double8
#define DATA16_T double16
#define DATA_MAX DBL_MAX
#define DATA_MIN -DATA_MAX
#define DATA_ZERO 0.0d
#define DATA_ONE 1.0d
#define DEF_ACC_DATA_T double
#define DEF_ACC_DATA2_T double2
#define DEF_ACC_DATA4_T double4
#define DEF_ACC_DATA8_T double8
#define POST_OP_DATA_T double
#define TO_DATA_T(v) (double)(v)
#define TO_DEF_ACC_DATA_T(v) (double)(v)
#define TO_DEF_ACC_DATA2_T convert_double2
#define TO_DEF_ACC_DATA4_T convert_double4
#define TO_DEF_ACC_DATA8_T convert_double8
#define DATA_TO_REF convert_float
#define CONVERT_DATA_T convert_double
#define CONVERT_DATA2_T convert_double2
#define CONVERT_DATA4_T convert_double4
#define CONVERT_DATA8_T convert_double8
#define CONVERT_FLOAT_T convert_float
#define CONVERT_FLOAT2_T convert_float2
#define CONVERT_FLOAT4_T convert_float4
#define CONVERT_FLOAT8_T convert_float8

#define BLOCK_DATA_T ulong
#define BLOCK_DATA2_T ulong2
#define BLOCK_DATA4_T ulong4
#define BLOCK_DATA8_T ulong8
#define AS_BLOCK_DATA_T as_ulong
#define AS_BLOCK_DATA2_T as_ulong2
#define AS_BLOCK_DATA4_T as_ulong4
#define AS_BLOCK_DATA8_T as_ulong8

#define BLOCK_READ intel_sub_group_block_read_ul
#define BLOCK_WRITE intel_sub_group_block_write_ul
#define BLOCK_READ2 intel_sub_group_block_read_ul2
#define BLOCK_READ4 intel_sub_group_block_read_ul4
#define BLOCK_READ8 intel_sub_group_block_read_ul8
#define BLOCK_WRITE2 intel_sub_group_block_write_ul2
#define BLOCK_WRITE4 intel_sub_group_block_write_ul4
#define BLOCK_WRITE8 intel_sub_group_block_write_ul8

#define AS_DATA_T as_double
#define AS_DATA2_T as_double2
#define AS_DATA4_T as_double4
#define AS_DATA8_T as_double8

#define AS_ULONG as_ulong
#define AS_ULONG2 as_ulong2
#define AS_ULONG4 as_ulong4
#define AS_ULONG8 as_ulong8

#define FLT_ACC_DATA_T double
#define TO_FLT_ACC_DATA_T(v) (double)(v)

#elif DT_F16 == 1

#define DATA_T half
#define DATA2_T half2
#define DATA4_T half4
#define DATA8_T half8
#define DATA16_T half16
#define AS_DATA2_T as_half2
#define DATA_MAX HALF_MAX
#define DATA_MIN -DATA_MAX
#define DATA_ZERO 0.0h
#define DATA_ONE 1.0h
#define DEF_ACC_DATA_T float
#define DEF_ACC_DATA2_T float2
#define DEF_ACC_DATA4_T float4
#define DEF_ACC_DATA8_T float8
#define POST_OP_DATA_T float
#define TO_DATA_T convert_half
#define TO_DEF_ACC_DATA_T convert_float
#define TO_DEF_ACC_DATA2_T convert_float2
#define TO_DEF_ACC_DATA4_T convert_float4
#define TO_DEF_ACC_DATA8_T convert_float8
#define DATA_TO_REF convert_half
#define CONVERT_DATA_T convert_half
#define CONVERT_DATA2_T convert_half2
#define CONVERT_DATA4_T convert_half4
#define CONVERT_DATA8_T convert_half8
#define CONVERT_FLOAT_T convert_float
#define CONVERT_FLOAT2_T convert_float2
#define CONVERT_FLOAT4_T convert_float4
#define CONVERT_FLOAT8_T convert_float8

#define BLOCK_READ intel_sub_group_block_read_us
#define BLOCK_WRITE intel_sub_group_block_write_us
#define BLOCK_READ2 intel_sub_group_block_read_us2
#define BLOCK_READ4 intel_sub_group_block_read_us4
#define BLOCK_READ8 intel_sub_group_block_read_us8
#define BLOCK_WRITE2 intel_sub_group_block_write_us2
#define BLOCK_WRITE4 intel_sub_group_block_write_us4
#define BLOCK_WRITE8 intel_sub_group_block_write_us8
#define AS_DATA_T as_half
#define AS_DATA2_T as_half2
#define AS_DATA4_T as_half4
#define AS_DATA8_T as_half8

#define AS_UINT_T as_ushort
#define AS_UINT2_T as_ushort2
#define AS_UINT4_T as_ushort4
#define AS_UINT8_T as_ushort8

#define BLOCK_DATA_T ushort
#define BLOCK_DATA2_T ushort2
#define BLOCK_DATA4_T ushort4
#define BLOCK_DATA8_T ushort8
#define AS_BLOCK_DATA_T as_ushort
#define AS_BLOCK_DATA2_T as_ushort2
#define AS_BLOCK_DATA4_T as_ushort4
#define AS_BLOCK_DATA8_T as_ushort8

#define FLT_ACC_DATA_T float
#define TO_FLT_ACC_DATA_T convert_float

#elif DT_BF16 == 1
#define DATA_T ushort
#define DATA2_T ushort2
#define POST_OP_DATA_T float
#define DATA2_T ushort2
#define DATA4_T ushort4
#define DATA8_T ushort8
#define DATA16_T ushort16
#define DATA_MAX (ushort)0x7F7F
#define DATA_MIN (ushort)0xFF7F
#define DATA_ZERO (ushort)0x0000
#define DATA_ONE (ushort)0x3F80
#define DEF_ACC_DATA_T float
#define DEF_ACC_DATA2_T float2
#define DEF_ACC_DATA4_T float4
#define DEF_ACC_DATA8_T float8
#define TO_DATA_T cvt_f32_to_bf16
#define TO_DEF_ACC_DATA_T cvt_bf16_to_f32
#define TO_DEF_ACC_DATA2_T cvt_bf16_to_f32
#define TO_DEF_ACC_DATA4_T cvt_bf16_to_f32
#define TO_DEF_ACC_DATA8_T cvt_bf16_to_f32
#define DATA_TO_REF cvt_bf16_to_f32
#define CONVERT_DATA_T(v) cvt_f32_to_bf16(convert_float(v))
#define CONVERT_DATA2_T(v) cvt_f32_to_bf16(convert_float2(v))
#define CONVERT_DATA4_T(v) cvt_f32_to_bf16(convert_float4(v))
#define CONVERT_DATA8_T(v) cvt_f32_to_bf16(convert_float8(v))
#define CONVERT_FLOAT_T cvt_bf16_to_f32
#define CONVERT_FLOAT2_T cvt_bf16_to_f32
#define CONVERT_FLOAT4_T cvt_bf16_to_f32
#define CONVERT_FLOAT8_T cvt_bf16_to_f32

#define BLOCK_READ intel_sub_group_block_read_us
#define BLOCK_WRITE intel_sub_group_block_write_us
#define BLOCK_READ2 intel_sub_group_block_read_us2
#define BLOCK_READ4 intel_sub_group_block_read_us4
#define BLOCK_READ8 intel_sub_group_block_read_us8
#define BLOCK_WRITE2 intel_sub_group_block_write_us2
#define BLOCK_WRITE4 intel_sub_group_block_write_us4
#define BLOCK_WRITE8 intel_sub_group_block_write_us8
#define AS_DATA_T as_ushort
#define AS_DATA2_T as_ushort2
#define AS_DATA4_T as_ushort4
#define AS_DATA8_T as_ushort8

#define AS_UINT_T as_ushort
#define AS_UINT2_T as_ushort2
#define AS_UINT4_T as_ushort4
#define AS_UINT8_T as_ushort8

#define BLOCK_DATA_T ushort
#define BLOCK_DATA2_T ushort2
#define BLOCK_DATA4_T ushort4
#define BLOCK_DATA8_T ushort8
#define AS_BLOCK_DATA_T as_ushort
#define AS_BLOCK_DATA2_T as_ushort2
#define AS_BLOCK_DATA4_T as_ushort4
#define AS_BLOCK_DATA8_T as_ushort8

#define FLT_ACC_DATA_T float
#define TO_FLT_ACC_DATA_T cvt_bf16_to_f32

#elif DT_BF8 == 1
#define DATA_T uchar
#define DATA2_T uchar2
#define DATA4_T uchar4
#define DATA8_T uchar8
#define DATA16_T uchar16
#define DATA_MAX (uchar)0x7B
#define DATA_MIN (uchar)0xFB
#define DATA_ZERO (uchar)0x00
#define DATA_ONE (uchar)0x3C
#define DEF_ACC_DATA_T float
#define DEF_ACC_DATA2_T float2
#define DEF_ACC_DATA4_T float4
#define DEF_ACC_DATA8_T float8
#define POST_OP_DATA_T float
#define TO_DATA_T(v) cvt_hf_to_f8_e5m2(convert_half(v)
#define TO_DEF_ACC_DATA_T(v) convert_float(cvt_f8_e5m2_to_hf(v))
#define TO_DEF_ACC_DATA2_T(v) convert_float(cvt_f8_e5m2_to_hf(v))
#define TO_DEF_ACC_DATA4_T(v) convert_float(cvt_f8_e5m2_to_hf(v))
#define TO_DEF_ACC_DATA8_T(v) convert_float(cvt_f8_e5m2_to_hf(v))
#define DATA_TO_REF(v) convert_float(cvt_f8_e5m2_to_hf(v))
#define CONVERT_DATA_T(v) cvt_hf_to_f8_e5m2(convert_half(v))
#define CONVERT_DATA2_T(v) cvt_hf_to_f8_e5m2(convert_half2(v))
#define CONVERT_DATA4_T(v) cvt_hf_to_f8_e5m2(convert_half4(v))
#define CONVERT_DATA8_T(v) cvt_hf_to_f8_e5m2(convert_half8(v))
#define CONVERT_FLOAT_T(v) convert_float(cvt_f8_e5m2_to_hf(v))
#define CONVERT_FLOAT2_T(v) convert_float2(cvt_f8_e5m2_to_hf(v))
#define CONVERT_FLOAT4_T(v) convert_float4(cvt_f8_e5m2_to_hf(v))
#define CONVERT_FLOAT8_T(v) convert_float8(cvt_f8_e5m2_to_hf(v))

#define BLOCK_READ intel_sub_group_block_read_uc
#define BLOCK_WRITE intel_sub_group_block_write_uc
#define BLOCK_READ2 intel_sub_group_block_read_uc2
#define BLOCK_READ4 intel_sub_group_block_read_uc4
#define BLOCK_READ8 intel_sub_group_block_read_uc8
#define BLOCK_WRITE2 intel_sub_group_block_write_uc2
#define BLOCK_WRITE4 intel_sub_group_block_write_uc4
#define BLOCK_WRITE8 intel_sub_group_block_write_uc8
#define AS_DATA_T as_uchar
#define AS_DATA2_T as_uchar2
#define AS_DATA4_T as_uchar4
#define AS_DATA8_T as_uchar8
#define AS_DATA16_T as_uchar16

#define AS_UINT_T as_uchar
#define AS_UINT2_T as_uchar2
#define AS_UINT4_T as_uchar4
#define AS_UINT8_T as_uchar8
#define AS_INT8_T as_uint8

#define BLOCK_DATA_T uchar
#define BLOCK_DATA2_T uchar2
#define BLOCK_DATA4_T uchar4
#define BLOCK_DATA8_T uchar8
#define AS_BLOCK_DATA_T as_uchar
#define AS_BLOCK_DATA2_T as_uchar2
#define AS_BLOCK_DATA4_T as_uchar4
#define AS_BLOCK_DATA8_T as_uchar8

#define MMAD_DATA_T half
#define MMAD_DATA4_T half4
#define MMAD_DATA8_T half8
#define MMAD_ACC_DATA4_T half4
#define MMAD_ACC_DATA8_T half8

#define FLT_ACC_DATA_T float
#define TO_FLT_ACC_DATA_T convert_float(cvt_f8_e5m2_to_hf(v))

#elif DT_HF8 == 1
#define DATA_T uchar
#define DATA2_T uchar2
#define DATA4_T uchar4
#define DATA8_T uchar8
#define DATA16_T uchar16
#define DATA_MAX (uchar)0x7E
#define DATA_MIN (uchar)0xFE
#define DATA_ZERO (uchar)0x00
#define DATA_ONE (uchar)0x38
#define DEF_ACC_DATA_T float
#define DEF_ACC_DATA2_T float2
#define DEF_ACC_DATA4_T float4
#define DEF_ACC_DATA8_T float8
#define POST_OP_DATA_T float
#define TO_DATA_T(v) cvt_hf_to_f8_e4m3(convert_half(v))
#define TO_DEF_ACC_DATA_T(v) convert_float(cvt_f8_e4m3_to_hf(v))
#define TO_DEF_ACC_DATA2_T(v) convert_float(cvt_f8_e4m3_to_hf(v))
#define TO_DEF_ACC_DATA4_T(v) convert_float(cvt_f8_e4m3_to_hf(v))
#define TO_DEF_ACC_DATA8_T(v) convert_float(cvt_f8_e4m3_to_hf(v))
#define DATA_TO_REF(v) convert_float(cvt_f8_e4m3_to_hf(v))
#define CONVERT_DATA_T(v) cvt_hf_to_f8_e4m3(convert_half(v))
#define CONVERT_DATA2_T(v) cvt_hf_to_f8_e4m3(convert_half2(v))
#define CONVERT_DATA4_T(v) cvt_hf_to_f8_e4m3(convert_half4(v))
#define CONVERT_DATA8_T(v) cvt_hf_to_f8_e4m3(convert_half8(v))
#define CONVERT_FLOAT_T(v) convert_float(cvt_f8_e4m3_to_hf(v))
#define CONVERT_FLOAT2_T(v) convert_float2(cvt_f8_e4m3_to_hf(v))
#define CONVERT_FLOAT4_T(v) convert_float4(cvt_f8_e4m3_to_hf(v))
#define CONVERT_FLOAT8_T(v) convert_float8(cvt_f8_e4m3_to_hf(v))

#define BLOCK_READ intel_sub_group_block_read_uc
#define BLOCK_WRITE intel_sub_group_block_write_uc
#define BLOCK_READ2 intel_sub_group_block_read_uc2
#define BLOCK_READ4 intel_sub_group_block_read_uc4
#define BLOCK_READ8 intel_sub_group_block_read_uc8
#define BLOCK_WRITE2 intel_sub_group_block_write_uc2
#define BLOCK_WRITE4 intel_sub_group_block_write_uc4
#define BLOCK_WRITE8 intel_sub_group_block_write_uc8
#define AS_DATA_T as_uchar
#define AS_DATA2_T as_uchar2
#define AS_DATA4_T as_uchar4
#define AS_DATA8_T as_uchar8
#define AS_DATA16_T as_uchar16

#define AS_UINT_T as_uchar
#define AS_UINT2_T as_uchar2
#define AS_UINT4_T as_uchar4
#define AS_UINT8_T as_uchar8
#define AS_INT8_T as_uint8

#define BLOCK_DATA_T uchar
#define BLOCK_DATA2_T uchar2
#define BLOCK_DATA4_T uchar4
#define BLOCK_DATA8_T uchar8
#define AS_BLOCK_DATA_T as_uchar
#define AS_BLOCK_DATA2_T as_uchar2
#define AS_BLOCK_DATA4_T as_uchar4
#define AS_BLOCK_DATA8_T as_uchar8

#define MMAD_DATA_T half
#define MMAD_DATA4_T half4
#define MMAD_DATA8_T half8
#define MMAD_ACC_DATA4_T half4
#define MMAD_ACC_DATA8_T half8

#define FLT_ACC_DATA_T float
#define TO_FLT_ACC_DATA_T(v) convert_float(cvt_f8_e4m3_to_hf(v))

#elif DT_S8 == 1
#define DATA_T char
#define DATA2_T char2
#define DATA4_T char4
#define DATA8_T char8
#define DATA16_T char16
#define DATA_MAX CHAR_MAX
#define DATA_MIN CHAR_MIN
#define DATA_ZERO 0
#define DATA_ONE 1
#define INT8_T int8
#define DEF_ACC_DATA_T int
#define DEF_ACC_DATA2_T int2
#define DEF_ACC_DATA4_T int4
#define DEF_ACC_DATA8_T int8
#define POST_OP_DATA_T float
#define TO_DATA_T(v) convert_char_sat_rte(v)
#define TO_DEF_ACC_DATA_T convert_int_sat_rte
#define TO_DEF_ACC_DATA2_T convert_int2_sat_rte
#define TO_DEF_ACC_DATA4_T convert_int4_sat_rte
#define TO_DEF_ACC_DATA8_T convert_int8_sat_rte
#define DATA_TO_REF convert_float
#define CONVERT_DATA_T convert_char_sat_rte
#define CONVERT_DATA2_T convert_char2_sat_rte
#define CONVERT_DATA4_T convert_char4_sat_rte
#define CONVERT_DATA8_T convert_char8_sat_rte
#define CONVERT_FLOAT_T convert_float
#define CONVERT_FLOAT2_T convert_float2
#define CONVERT_FLOAT4_T convert_float4
#define CONVERT_FLOAT8_T convert_float8

#define BLOCK_READ intel_sub_group_block_read_uc
#define BLOCK_WRITE intel_sub_group_block_write_uc
#define BLOCK_READ2 intel_sub_group_block_read_uc2
#define BLOCK_READ4 intel_sub_group_block_read_uc4
#define BLOCK_READ8 intel_sub_group_block_read_uc8
#define BLOCK_WRITE2 intel_sub_group_block_write_uc2
#define BLOCK_WRITE4 intel_sub_group_block_write_uc4
#define BLOCK_WRITE8 intel_sub_group_block_write_uc8
#define AS_DATA_T as_char
#define AS_DATA2_T as_char2
#define AS_DATA4_T as_char4
#define AS_DATA8_T as_char8
#define AS_DATA16_T as_char16

#define AS_UINT_T as_uchar
#define AS_UINT2_T as_uchar2
#define AS_UINT4_T as_uchar4
#define AS_UINT8_T as_uchar8
#define AS_INT8_T as_int8

#define BLOCK_DATA_T uchar
#define BLOCK_DATA2_T uchar2
#define BLOCK_DATA4_T uchar4
#define BLOCK_DATA8_T uchar8
#define AS_BLOCK_DATA_T as_uchar
#define AS_BLOCK_DATA2_T as_uchar2
#define AS_BLOCK_DATA4_T as_uchar4
#define AS_BLOCK_DATA8_T as_uchar8

#define FLT_ACC_DATA_T float
#define TO_FLT_ACC_DATA_T convert_float

#elif DT_U8 == 1
#define DATA_T uchar
#define DATA2_T uchar2
#define DATA4_T uchar4
#define DATA8_T uchar8
#define DATA16_T uchar16
#define DATA_MAX UCHAR_MAX
#define DATA_MIN 0
#define DATA_ZERO 0
#define DATA_ONE 1
#define INT8_T uint8
#define DEF_ACC_DATA_T int
#define DEF_ACC_DATA2_T int2
#define DEF_ACC_DATA4_T int4
#define DEF_ACC_DATA8_T int8
#define POST_OP_DATA_T float
#define TO_DATA_T(v) convert_uchar_sat_rte(v)
#define TO_DEF_ACC_DATA_T convert_int_sat_rte
#define TO_DEF_ACC_DATA2_T convert_int2_sat_rte
#define TO_DEF_ACC_DATA4_T convert_int4_sat_rte
#define TO_DEF_ACC_DATA8_T convert_int8_sat_rte
#define DATA_TO_REF convert_float
#define CONVERT_DATA_T convert_uchar_sat_rte
#define CONVERT_DATA2_T convert_uchar2_sat_rte
#define CONVERT_DATA4_T convert_uchar4_sat_rte
#define CONVERT_DATA8_T convert_uchar8_sat_rte
#define CONVERT_FLOAT_T convert_float
#define CONVERT_FLOAT2_T convert_float2
#define CONVERT_FLOAT4_T convert_float4
#define CONVERT_FLOAT8_T convert_float8

#define BLOCK_READ intel_sub_group_block_read_uc
#define BLOCK_WRITE intel_sub_group_block_write_uc
#define BLOCK_READ2 intel_sub_group_block_read_uc2
#define BLOCK_READ4 intel_sub_group_block_read_uc4
#define BLOCK_READ8 intel_sub_group_block_read_uc8
#define BLOCK_WRITE2 intel_sub_group_block_write_uc2
#define BLOCK_WRITE4 intel_sub_group_block_write_uc4
#define BLOCK_WRITE8 intel_sub_group_block_write_uc8
#define AS_DATA_T as_uchar
#define AS_DATA2_T as_uchar2
#define AS_DATA4_T as_uchar4
#define AS_DATA8_T as_uchar8
#define AS_DATA16_T as_uchar16

#define AS_UINT_T as_uchar
#define AS_UINT2_T as_uchar2
#define AS_UINT4_T as_uchar4
#define AS_UINT8_T as_uchar8
#define AS_INT8_T as_uint8

#define BLOCK_DATA_T uchar
#define BLOCK_DATA2_T uchar2
#define BLOCK_DATA4_T uchar4
#define BLOCK_DATA8_T uchar8
#define AS_BLOCK_DATA_T as_uchar
#define AS_BLOCK_DATA2_T as_uchar2
#define AS_BLOCK_DATA4_T as_uchar4
#define AS_BLOCK_DATA8_T as_uchar8

#define FLT_ACC_DATA_T float
#define TO_FLT_ACC_DATA_T convert_float

#elif DT_S32 == 1
#define DATA_T int
#define DATA2_T int2
#define DATA4_T int4
#define DATA8_T int8
#define DATA16_T int16
#define DATA_MAX INT_MAX
#define DATA_MIN INT_MIN
#define DATA_ZERO 0
#define DATA_ONE 1
#define DATA_TO_REF convert_float
#define TO_DATA_T(v) convert_int_sat_rte
#define TO_DATA8_T(v) convert_int8_sat_rte
#define CONVERT_DATA_T convert_int_sat_rte
#define CONVERT_DATA2_T convert_int2_sat_rte
#define CONVERT_DATA4_T convert_int4_sat_rte
#define CONVERT_DATA8_T convert_int8_sat_rte
#define CONVERT_FLOAT_T convert_float
#define CONVERT_FLOAT2_T convert_float2
#define CONVERT_FLOAT4_T convert_float4
#define CONVERT_FLOAT8_T convert_float8
#define DEF_ACC_DATA_T int
#define DEF_ACC_DATA2_T int2
#define DEF_ACC_DATA4_T int4
#define DEF_ACC_DATA8_T int8
#define TO_DEF_ACC_DATA_T convert_int_sat_rte
#define TO_DEF_ACC_DATA2_T convert_int2_sat_rte
#define TO_DEF_ACC_DATA4_T convert_int4_sat_rte
#define TO_DEF_ACC_DATA8_T convert_int8_sat_rte
#define POST_OP_DATA_T float
#define DATA_MIN INT_MIN
#define DATA_MAX INT_MAX
#define DATA_ZERO 0

#define BLOCK_READ intel_sub_group_block_read
#define BLOCK_WRITE intel_sub_group_block_write
#define BLOCK_READ2 intel_sub_group_block_read2
#define BLOCK_READ4 intel_sub_group_block_read4
#define BLOCK_READ8 intel_sub_group_block_read8
#define BLOCK_WRITE2 intel_sub_group_block_write2
#define BLOCK_WRITE4 intel_sub_group_block_write4
#define BLOCK_WRITE8 intel_sub_group_block_write8

#define AS_DATA_T as_int
#define AS_DATA2_T as_int2
#define AS_DATA4_T as_int4
#define AS_DATA8_T as_int8

#define AS_UINT_T as_uint
#define AS_UINT2_T as_uint2
#define AS_UINT4_T as_uint4
#define AS_UINT8_T as_uint8

#define BLOCK_DATA_T uint
#define BLOCK_DATA2_T uint2
#define BLOCK_DATA4_T uint4
#define BLOCK_DATA8_T uint8
#define AS_BLOCK_DATA_T as_uint
#define AS_BLOCK_DATA2_T as_uint2
#define AS_BLOCK_DATA4_T as_uint4
#define AS_BLOCK_DATA8_T as_uint8

#define FLT_ACC_DATA_T float
#define TO_FLT_ACC_DATA_T convert_float

#elif !defined(DT_UNDEF)
#error "Unexpected data type"
#endif

#if VECT_DT_N == 1
#define VECT_DATA_T DATA_T
#define VECT_DEF_ACC_DATA_T DEF_ACC_DATA_T
#define AS_VECT_DEF_ACC_DATA_T TO_DEF_ACC_DATA_T
#define AS_VECT_DATA_T AS_DATA_T
#define VECT_BLOCK_READ BLOCK_READ
#define VECT_BLOCK_WRITE BLOCK_WRITE
#define VECT_DST_BLOCK_WRITE BLOCK_WRITE_DST
#define VECT_UINT_READ intel_sub_group_block_read
#define VECT_UINT_WRITE intel_sub_group_block_write
#define VECT_UCHAR_READ intel_sub_group_block_read_uc
#define VECT_UCHAR_WRITE intel_sub_group_block_write_uc
#define VECT_BLOCK_DATA_T BLOCK_DATA_T
#define AS_VECT_BLOCK_DATA_T AS_BLOCK_DATA_T
#define CONVERT_VECT_FLOAT_T CONVERT_FLOAT_T
#define CONVERT_VECTOR_DATA_T CONVERT_DATA_T
#define CONVERT_VECTOR_DST_DATA_T CONVERT_DST_DATA_T
#define CONVERT_VECT_CHAR_T convert_char
#define CONVERT_VECT_INT_T convert_int
#define VECT_INT_T int
#define VECT_UINT_T uint
#define VECT_FLOAT_T float
#define VECT_CHAR_T char
#define AS_VECT_INT_T as_int
#define AS_VECT_UINT_T as_uint
#define AS_VECT_FLOAT_T as_float
#define AS_VECT_CHAR_T as_char
#define AS_VECT_UCHAR_T as_uchar
#elif VECT_DT_N == 2
#define VECT_DATA_T DATA2_T
#define VECT_DEF_ACC_DATA_T DEF_ACC_DATA2_T
#define AS_VECT_DEF_ACC_DATA_T TO_DEF_ACC_DATA2_T
#define AS_VECT_DATA_T AS_DATA2_T
#define VECT_BLOCK_READ BLOCK_READ2
#define VECT_BLOCK_WRITE BLOCK_WRITE2
#define VECT_DST_BLOCK_WRITE BLOCK_WRITE_DST2
#define VECT_UINT_READ intel_sub_group_block_read2
#define VECT_UINT_WRITE intel_sub_group_block_write2
#define VECT_UCHAR_READ intel_sub_group_block_read_uc2
#define VECT_UCHAR_WRITE intel_sub_group_block_write_uc2
#define VECT_BLOCK_DATA_T BLOCK_DATA2_T
#define AS_VECT_BLOCK_DATA_T AS_BLOCK_DATA2_T
#define CONVERT_VECT_FLOAT_T CONVERT_FLOAT2_T
#define CONVERT_VECTOR_DATA_T CONVERT_DATA2_T
#define CONVERT_VECTOR_DST_DATA_T CONVERT_DST_DATA2_T
#define CONVERT_VECT_CHAR_T convert_char2
#define CONVERT_VECT_INT_T convert_int2
#define VECT_INT_T int2
#define VECT_UINT_T uint2
#define VECT_FLOAT_T float2
#define VECT_CHAR_T char2
#define AS_VECT_INT_T as_int2
#define AS_VECT_UINT_T as_uint2
#define AS_VECT_FLOAT_T as_float2
#define AS_VECT_CHAR_T as_char2
#define AS_VECT_UCHAR_T as_uchar2
#elif VECT_DT_N == 4
#define VECT_DATA_T DATA4_T
#define VECT_DEF_ACC_DATA_T DEF_ACC_DATA4_T
#define AS_VECT_DEF_ACC_DATA_T TO_DEF_ACC_DATA4_T
#define AS_VECT_DATA_T AS_DATA4_T
#define VECT_BLOCK_READ BLOCK_READ4
#define VECT_BLOCK_WRITE BLOCK_WRITE4
#define VECT_DST_BLOCK_WRITE BLOCK_WRITE_DST4
#define VECT_UINT_READ intel_sub_group_block_read4
#define VECT_UINT_WRITE intel_sub_group_block_write4
#define VECT_UCHAR_READ intel_sub_group_block_read_uc4
#define VECT_UCHAR_WRITE intel_sub_group_block_write_uc4
#define VECT_BLOCK_DATA_T BLOCK_DATA4_T
#define AS_VECT_BLOCK_DATA_T AS_BLOCK_DATA4_T
#define CONVERT_VECT_FLOAT_T CONVERT_FLOAT4_T
#define CONVERT_VECTOR_DATA_T CONVERT_DATA4_T
#define CONVERT_VECTOR_DST_DATA_T CONVERT_DST_DATA4_T
#define CONVERT_VECT_CHAR_T convert_char4
#define CONVERT_VECT_INT_T convert_int4
#define VECT_INT_T int4
#define VECT_UINT_T uint4
#define VECT_FLOAT_T float4
#define VECT_CHAR_T char4
#define AS_VECT_INT_T as_int4
#define AS_VECT_UINT_T as_uint4
#define AS_VECT_FLOAT_T as_float4
#define AS_VECT_CHAR_T as_char4
#define AS_VECT_UCHAR_T as_uchar4
#elif VECT_DT_N == 8
#define VECT_DATA_T DATA8_T
#define VECT_DEF_ACC_DATA_T DEF_ACC_DATA8_T
#define AS_VECT_DEF_ACC_DATA_T TO_DEF_ACC_DATA8_T
#define AS_VECT_DATA_T AS_DATA8_T
#define VECT_BLOCK_READ BLOCK_READ8
#define VECT_BLOCK_WRITE BLOCK_WRITE8
#define VECT_DST_BLOCK_WRITE BLOCK_WRITE_DST8
#define VECT_UINT_READ intel_sub_group_block_read8
#define VECT_UINT_WRITE intel_sub_group_block_write8
#define VECT_UCHAR_READ intel_sub_group_block_read_uc8
#define VECT_UCHAR_WRITE intel_sub_group_block_write_uc8
#define VECT_BLOCK_DATA_T BLOCK_DATA8_T
#define AS_VECT_BLOCK_DATA_T AS_BLOCK_DATA8_T
#define CONVERT_VECT_FLOAT_T CONVERT_FLOAT8_T
#define CONVERT_VECTOR_DATA_T CONVERT_DATA8_T
#define CONVERT_VECTOR_DST_DATA_T CONVERT_DST_DATA8_T
#define CONVERT_VECT_CHAR_T convert_char8
#define CONVERT_VECT_INT_T convert_int8
#define VECT_INT_T int8
#define VECT_UINT_T uint8
#define VECT_FLOAT_T float8
#define VECT_CHAR_T char8
#define AS_VECT_INT_T as_int8
#define AS_VECT_UINT_T as_uint8
#define AS_VECT_FLOAT_T as_float8
#define AS_VECT_CHAR_T as_char8
#define AS_VECT_UCHAR_T as_uchar8
#endif

#ifdef SRC_DATA_T
#define SRC_DATA2_T CONCAT2(SRC_DATA_T, 2)
#define SRC_DATA4_T CONCAT2(SRC_DATA_T, 4)
#define SRC_DATA8_T CONCAT2(SRC_DATA_T, 8)
#define SRC_DATA16_T CONCAT2(SRC_DATA_T, 16)

#define AS_SRC_DATA_T CONCAT2(as_, SRC_DATA_T)
#define AS_SRC_DATA2_T CONCAT2(as_, SRC_DATA2_T)
#define AS_SRC_DATA4_T CONCAT2(as_, SRC_DATA4_T)
#define AS_SRC_DATA8_T CONCAT2(as_, SRC_DATA8_T)
#define AS_SRC_DATA16_T CONCAT2(as_, SRC_DATA16_T)
#if SRC_DT_U8
#define SRC_TO_REF(x) convert_float(x)
#define SRC_TO_REF8(x) convert_float8(x)
#define REF_TO_SRC(x) convert_uchar(x)
#elif SRC_DT_S8
#define SRC_TO_REF(x) convert_float(x)
#define SRC_TO_REF8(x) convert_float8(x)
#define REF_TO_SRC(x) convert_char(x)
#elif SRC_DT_BF16
#define SRC_TO_REF(x) cvt_bf16_to_f32(x)
#define SRC_TO_REF8(x) cvt_bf16_to_f32(x)
#define REF_TO_SRC(x) cvt_f32_to_bf16(x)
#elif SRC_DT_F16
#define SRC_TO_REF(x) convert_float(x)
#define SRC_TO_REF8(x) convert_float8(x)
#define REF_TO_SRC(x) convert_half(x)
#elif SRC_DT_BF8
#define SRC_TO_REF(x) convert_float(cvt_f8_e5m2_to_hf(x))
#define SRC_TO_REF8(x) convert_float8(cvt_f8_e5m2_to_hf(x))
#define REF_TO_SRC(x) cvt_hf_to_f8_e5m2(convert_half(x))
#elif SRC_DT_HF8
#define SRC_TO_REF(x) convert_float(cvt_f8_e4m3_to_hf(x))
#define SRC_TO_REF8(x) convert_float8(cvt_f8_e4m3_to_hf(x))
#define REF_TO_SRC(x) cvt_hf_to_f8_e4m3(convert_half(x))
#elif SRC_DT_U4
#define GET_HALF_BYTE(x, y) get_half_byte(x, y)
#define SRC_TO_REF(x) convert_float(x)
#elif SRC_DT_S4
#define GET_HALF_BYTE(x, y) get_half_byte(x, y)
#define SRC_TO_REF(x) convert_float(cvt_s4_to_f32(x))
#else
#define SRC_TO_REF(x) (x)
#define SRC_TO_REF8(x) (x)
#define REF_TO_SRC(x) (x)
#endif
#if SRC_DT_BF16
#define TO_SRC(x) cvt_f32_to_bf16(x)
#elif SRC_DT_F16
#define TO_SRC(x) convert_half(x)
#elif SRC_DT_BF8
#define TO_SRC(x) cvt_hf_to_f8_e5m2(convert_half(x))
#elif SRC_DT_HF8
#define TO_SRC(x) cvt_hf_to_f8_e4m3(convert_half(x))
#elif SRC_DT_U8
#define TO_SRC(x) convert_uchar_sat_rte(x)
#elif SRC_DT_S8
#define TO_SRC(x) convert_char_sat_rte(x)
#elif SRC_DT_S32
#define TO_SRC(x) convert_int_sat_rte(x)
#else
#define TO_SRC(x) (x)
#endif
#endif

#ifdef A_DATA_T
#define A_DATA8_T CONCAT2(A_DATA_T, 8)
#if A_DT_BF16
#define A_TO_REF(x) cvt_bf16_to_f32(x)
#define A_TO_REF8(x) cvt_bf16_to_f32(x)
#define REF_TO_A(x) cvt_f32_to_bf16(x)
#elif A_DT_BF8
#define A_TO_REF(x) cvt_f8_e5m2_to_hf(x)
#define A_TO_REF8(x) cvt_f8_e5m2_to_hf(x)
#define REF_TO_A(x) cvt_hf_to_f8_e5m2(convert_half(x))
#elif A_DT_HF8
#define A_TO_REF(x) cvt_f8_e4m3_to_hf(x)
#define A_TO_REF8(x) cvt_f8_e4m3_to_hf(x)
#define REF_TO_A(x) cvt_hf_to_f8_e4m3(convert_half(x))
#else
#define A_TO_REF(x) (x)
#define A_TO_REF8(x) (x)
#define REF_TO_A(x) (x)
#endif
#if A_DT_BF16
#define TO_A(x) cvt_f32_to_bf16(x)
#elif A_DT_BF8
#define TO_A(x) cvt_hf_to_f8_e5m2(convert_half(x))
#elif A_DT_HF8
#define TO_A(x) cvt_hf_to_f8_e4m3(convert_half(x))
#elif A_DT_U8
#define TO_A(x) convert_uchar_sat_rte(x)
#elif A_DT_S8
#define TO_A(x) convert_char_sat_rte(x)
#elif A_DT_S32
#define TO_A(x) convert_int_sat_rte(x)
#else
#define TO_A(x) (x)
#endif
#endif

#ifdef WEI_DATA_T
#if WEI_DT_BF16
#define WEI_TO_REF(x) cvt_bf16_to_f32(x)
#define REF_TO_WEI(x) cvt_f32_to_bf16(x)
#elif WEI_DT_BF8
#define WEI_TO_REF(x) convert_float(cvt_f8_e5m2_to_hf(x))
#define REF_TO_WEI(x) cvt_hf_to_f8_e5m2(convert_half(x))
#elif WEI_DT_HF8
#define WEI_TO_REF(x) convert_float(cvt_f8_e4m3_to_hf(x))
#define REF_TO_WEI(x) cvt_hf_to_f8_e4m3(convert_half(x))
#else
#define WEI_TO_REF(x) (x)
#define REF_TO_WEI(x) (x)
#endif
#if WEI_DT_BF16
#define TO_WEI(x) cvt_f32_to_bf16(x)
#elif WEI_DT_BF8
#define TO_WEI(x) cvt_hf_to_f8_e5m2(convert_half(x))
#elif WEI_DT_HF8
#define TO_WEI(x) cvt_hf_to_f8_e4m3(convert_half(x))
#elif WEI_DT_U8
#define TO_WEI(x) convert_uchar_sat_rte(x)
#elif WEI_DT_S8
#define TO_WEI(x) convert_char_sat_rte(x)
#elif WEI_DT_S32
#define TO_WEI(x) convert_int_sat_rte(x)
#else
#define TO_WEI(x) (x)
#endif
#endif

#ifdef DIFF_WEI_DATA_T
#if DIFF_WEI_DT_BF16
#define DIFF_WEI_TO_REF(x) cvt_bf16_to_f32(x)
#define REF_TO_DIFF_WEI(x) cvt_f32_to_bf16(x)
#else
#define DIFF_WEI_TO_REF(x) (x)
#define REF_TO_DIFF_WEI(x) (x)
#endif
#if DIFF_WEI_DT_BF16
#define TO_DIFF_WEI(x) cvt_f32_to_bf16(x)
#elif DIFF_WEI_DT_U8
#define TO_DIFF_WEI(x) convert_uchar_sat_rte(x)
#elif DIFF_WEI_DT_S8
#define TO_DIFF_WEI(x) convert_char_sat_rte(x)
#elif DIFF_WEI_DT_S32
#define TO_DIFF_WEI(x) convert_int_sat_rte(x)
#else
#define TO_DIFF_WEI(x) (x)
#endif
#endif

// f16/bf16 scale-shift support for normalization primitives
#ifdef WEI_DATA_T
#if WEI_DT_F32 == 1
#define AS_WEI_DATA_T as_float
#define BLOCK_WEI_READ intel_sub_group_block_read
#define BLOCK_WEI_WRITE intel_sub_group_block_write
#define BLOCK_WEI_DATA_T uint
#define AS_BLOCK_WEI_DATA_T as_uint
#define CONVERT_WEI_FLOAT_T convert_float
#define CONVERT_WEI_DATA_T convert_float
#elif WEI_DT_F16 == 1
#define AS_WEI_DATA_T as_half
#define BLOCK_WEI_READ intel_sub_group_block_read_us
#define BLOCK_WEI_WRITE intel_sub_group_block_write_us
#define BLOCK_WEI_DATA_T ushort
#define AS_BLOCK_WEI_DATA_T as_ushort
#define CONVERT_WEI_FLOAT_T convert_float
#define CONVERT_WEI_DATA_T convert_half
#elif WEI_DT_BF16 == 1
#define AS_WEI_DATA_T as_ushort
#define BLOCK_WEI_READ intel_sub_group_block_read_us
#define BLOCK_WEI_WRITE intel_sub_group_block_write_us
#define BLOCK_WEI_DATA_T ushort
#define AS_BLOCK_WEI_DATA_T as_ushort
#define CONVERT_WEI_FLOAT_T cvt_bf16_to_f32
#define CONVERT_WEI_DATA_T cvt_f32_to_bf16
#endif
#if VECT_DT_N == 1
#define VECT_BLOCK_WEI_READ BLOCK_WEI_READ
#define VECT_BLOCK_WEI_WRITE BLOCK_WEI_WRITE
#define AS_VECT_WEI_DATA_T AS_WEI_DATA_T
#define AS_VECT_BLOCK_WEI_DATA_T AS_BLOCK_WEI_DATA_T
#define CONVERT_VECT_WEI_FLOAT_T CONVERT_WEI_FLOAT_T
#define CONVERT_VECT_WEI_DATA_T CONVERT_WEI_DATA_T
#else
#define VECT_BLOCK_WEI_READ CONCAT2(BLOCK_WEI_READ, VECT_DT_N)
#define VECT_BLOCK_WEI_WRITE CONCAT2(BLOCK_WEI_WRITE, VECT_DT_N)
#define AS_VECT_WEI_DATA_T CONCAT2(AS_WEI_DATA_T, VECT_DT_N)
#define AS_VECT_BLOCK_WEI_DATA_T CONCAT2(AS_BLOCK_WEI_DATA_T, VECT_DT_N)
#if WEI_DT_BF16 == 1
#define CONVERT_VECT_WEI_FLOAT_T CONVERT_WEI_FLOAT_T
#define CONVERT_VECT_WEI_DATA_T CONVERT_WEI_DATA_T
#else
#define CONVERT_VECT_WEI_FLOAT_T CONCAT2(CONVERT_WEI_FLOAT_T, VECT_DT_N)
#define CONVERT_VECT_WEI_DATA_T CONCAT2(CONVERT_WEI_DATA_T, VECT_DT_N)
#endif
#endif
#define LOAD_VECT_WEI(ptr) \
    CONVERT_VECT_WEI_FLOAT_T(AS_VECT_WEI_DATA_T( \
            VECT_BLOCK_WEI_READ((const __global BLOCK_WEI_DATA_T *)(ptr))))
#define SAVE_VECT_WEI(ptr, val) \
    VECT_BLOCK_WEI_WRITE((__global BLOCK_WEI_DATA_T *)(ptr), \
            AS_VECT_BLOCK_WEI_DATA_T(CONVERT_VECT_WEI_DATA_T(val)))
#endif

#ifdef B_DATA_T
#if B_DT_BF16
#define B_TO_REF(x) cvt_bf16_to_f32(x)
#define REF_TO_B(x) cvt_f32_to_bf16(x)
#elif B_DT_BF8
#define B_TO_REF(x) cvt_f8_e5m2_to_hf(x)
#define REF_TO_B(x) cvt_hf_to_f8_e5m2(convert_half(x))
#elif B_DT_HF8
#define B_TO_REF(x) cvt_f8_e4m3_to_hf(x)
#define REF_TO_B(x) cvt_hf_to_f8_e4m3(convert_half(x))
#else
#define B_TO_REF(x) (x)
#define REF_TO_B(x) (x)
#endif
#if B_DT_BF16
#define TO_B(x) cvt_f32_to_bf16(x)
#elif B_DT_BF8
#define TO_B(x) cvt_hf_to_f8_e5m2(convert_half(x))
#elif B_DT_HF8
#define TO_B(x) cvt_hf_to_f8_e4m3(convert_half(x))
#elif B_DT_U8
#define TO_B(x) convert_uchar_sat_rte(x)
#elif B_DT_S8
#define TO_B(x) convert_char_sat_rte(x)
#elif B_DT_S32
#define TO_B(x) convert_int_sat_rte(x)
#else
#define TO_B(x) (x)
#endif
#endif

#ifdef BIA_DATA_T
#define BIA_DATA2_T CONCAT2(BIA_DATA_T, 2)
#if BIA_DT_BF16
#define BIA_TO_REF(x) cvt_bf16_to_f32(x)
#define REF_TO_BIA(x) cvt_f32_to_bf16(x)
#elif BIA_DT_BF8
#define BIA_TO_REF(x) convert_float(cvt_f8_e5m2_to_hf(x))
#define REF_TO_BIA(x) cvt_hf_to_f8_e5m2(convert_half(x))
#elif BIA_DT_HF8
#define BIA_TO_REF(x) convert_float(cvt_f8_e4m3_to_hf(x))
#define REF_TO_BIA(x) cvt_hf_to_f8_e4m3(convert_half(x))
#else
#define BIA_TO_REF(x) (x)
#define REF_TO_BIA(x) (x)
#endif
#if BIA_DT_BF16
#define TO_BIA(x) cvt_f32_to_bf16(x)
#elif BIA_DT_BF8
#define TO_BIA(x) cvt_hf_to_f8_e5m2(convert_half(x))
#elif BIA_DT_HF8
#define TO_BIA(x) cvt_hf_to_f8_e4m3(convert_half(x))
#elif BIA_DT_U8
#define TO_BIA(x) convert_uchar_sat_rte(x)
#elif BIA_DT_S8
#define TO_BIA(x) convert_char_sat_rte(x)
#elif BIA_DT_S32
#define TO_BIA(x) convert_int_sat_rte(x)
#else
#define TO_BIA(x) (x)
#endif
#endif

#ifdef DST_DATA_T
#define DST_DATA2_T CONCAT2(DST_DATA_T, 2)
#define DST_DATA4_T CONCAT2(DST_DATA_T, 4)
#define DST_DATA8_T CONCAT2(DST_DATA_T, 8)
#define DST_DATA16_T CONCAT2(DST_DATA_T, 16)

#define AS_DST_DATA_T CONCAT2(as_, DST_DATA_T)
#define AS_DST_DATA2_T CONCAT2(as_, DST_DATA2_T)
#define AS_DST_DATA4_T CONCAT2(as_, DST_DATA4_T)
#define AS_DST_DATA8_T CONCAT2(as_, DST_DATA8_T)
#define AS_DST_DATA16_T CONCAT2(as_, DST_DATA16_T)

#if DST_DT_F32 || DST_DT_F16
#define CONVERT_DST_DATA_T CONCAT2(convert_, DST_DATA_T)
#define CONVERT_DST_DATA2_T CONCAT2(convert_, DST_DATA2_T)
#define CONVERT_DST_DATA4_T CONCAT2(convert_, DST_DATA4_T)
#define CONVERT_DST_DATA8_T CONCAT2(convert_, DST_DATA8_T)
#define CONVERT_DST_DATA16_T CONCAT2(convert_, DST_DATA16_T)
#elif DST_DT_BF16
#define CONVERT_DST_DATA_T TO_DST
#define CONVERT_DST_DATA2_T TO_DST2
#define CONVERT_DST_DATA4_T TO_DST4
#define CONVERT_DST_DATA8_T TO_DST8
#define CONVERT_DST_DATA16_T TO_DST16
#else
#define CONVERT_DST_DATA_T CONCAT3(convert_, DST_DATA_T, _sat_rte)
#define CONVERT_DST_DATA2_T CONCAT3(convert_, DST_DATA2_T, _sat_rte)
#define CONVERT_DST_DATA4_T CONCAT3(convert_, DST_DATA4_T, _sat_rte)
#define CONVERT_DST_DATA8_T CONCAT3(convert_, DST_DATA8_T, _sat_rte)
#define CONVERT_DST_DATA16_T CONCAT3(convert_, DST_DATA16_T, _sat_rte)
#endif

// Block read/write macros for dst.
#if DST_DT_U8 || DST_DT_S8
#define BLOCK_READ_DST2(ptr) \
    AS_DST_DATA2_T(intel_sub_group_block_read_uc2((__global uchar *)ptr))
#define BLOCK_WRITE_DST2(ptr, v) \
    intel_sub_group_block_write_uc2((__global uchar *)ptr, as_uchar2(v))

#define BLOCK_READ_DST(ptr) \
    AS_DST_DATA_T(intel_sub_group_block_read_uc((__global uchar *)ptr))
#define BLOCK_WRITE_DST(ptr, v) \
    intel_sub_group_block_write_uc((__global uchar *)ptr, as_uchar(v))

#define BLOCK_READ_DST2(ptr) \
    AS_DST_DATA2_T(intel_sub_group_block_read_uc2((__global uchar *)ptr))
#define BLOCK_WRITE_DST2(ptr, v) \
    intel_sub_group_block_write_uc2((__global uchar *)ptr, as_uchar2(v))

#define BLOCK_READ_DST4(ptr) \
    AS_DST_DATA4_T(intel_sub_group_block_read_uc4((__global uchar *)ptr))
#define BLOCK_WRITE_DST4(ptr, v) \
    intel_sub_group_block_write_uc4((__global uchar *)ptr, as_uchar4(v))

#define BLOCK_READ_DST8(ptr) \
    AS_DST_DATA8_T(intel_sub_group_block_read_uc8((__global uchar *)ptr))
#define BLOCK_WRITE_DST8(ptr, v) \
    intel_sub_group_block_write_uc8((__global uchar *)ptr, as_uchar8(v))

#define BLOCK_READ_DST16(ptr) \
    AS_DST_DATA16_T(intel_sub_group_block_read_uc16((__global uchar *)ptr))
#define BLOCK_WRITE_DST16(ptr, v) \
    intel_sub_group_block_write_uc16((__global uchar *)ptr, as_uchar16(v))

#elif DST_DT_F16 || DST_DT_BF16
#define BLOCK_READ_DST(ptr) \
    AS_DST_DATA_T(intel_sub_group_block_read_us((__global ushort *)ptr))
#define BLOCK_WRITE_DST(ptr, v) \
    intel_sub_group_block_write_us((__global ushort *)ptr, as_ushort(v))

#define BLOCK_READ_DST2(ptr) \
    AS_DST_DATA2_T(intel_sub_group_block_read_us2((__global ushort *)ptr))
#define BLOCK_WRITE_DST2(ptr, v) \
    intel_sub_group_block_write_us2((__global ushort *)ptr, as_ushort2(v))

#define BLOCK_READ_DST4(ptr) \
    AS_DST_DATA4_T(intel_sub_group_block_read_us4((__global ushort *)ptr))
#define BLOCK_WRITE_DST4(ptr, v) \
    intel_sub_group_block_write_us4((__global ushort *)ptr, as_ushort4(v))

#define BLOCK_READ_DST8(ptr) \
    AS_DST_DATA8_T(intel_sub_group_block_read_us8((__global ushort *)ptr))
#define BLOCK_WRITE_DST8(ptr, v) \
    intel_sub_group_block_write_us8((__global ushort *)ptr, as_ushort8(v))

#define BLOCK_READ_DST16(ptr) \
    (DST_DATA16_T)( \
            BLOCK_READ_DST8(ptr), BLOCK_READ_DST8(ptr + 8 * SUB_GROUP_SIZE))
#define BLOCK_WRITE_DST16(ptr, v) \
    do { \
        BLOCK_WRITE_DST8(ptr, (v).s01234567); \
        BLOCK_WRITE_DST8(ptr + 8 * SUB_GROUP_SIZE, (v).s89abcdef); \
    } while (0)

#elif DST_DT_S32 || DST_DT_F32

#define BLOCK_READ_DST(ptr) \
    AS_DST_DATA_T(intel_sub_group_block_read((__global uint *)ptr))
#define BLOCK_WRITE_DST(ptr, v) \
    intel_sub_group_block_write((__global uint *)ptr, as_uint(v))

#define BLOCK_READ_DST2(ptr) \
    AS_DST_DATA2_T(intel_sub_group_block_read2((__global uint *)ptr))
#define BLOCK_WRITE_DST2(ptr, v) \
    intel_sub_group_block_write2((__global uint *)ptr, as_uint2(v))

#define BLOCK_READ_DST4(ptr) \
    AS_DST_DATA4_T(intel_sub_group_block_read4((__global uint *)ptr))
#define BLOCK_WRITE_DST4(ptr, v) \
    intel_sub_group_block_write4((__global uint *)ptr, as_uint4(v))

#define BLOCK_READ_DST8(ptr) \
    AS_DST_DATA8_T(intel_sub_group_block_read8((__global uint *)ptr))
#define BLOCK_WRITE_DST8(ptr, v) \
    intel_sub_group_block_write8((__global uint *)ptr, as_uint8(v))

#define BLOCK_READ_DST16(ptr) \
    (DST_DATA16_T)( \
            BLOCK_READ_DST8(ptr), BLOCK_READ_DST8(ptr + 8 * SUB_GROUP_SIZE))
#define BLOCK_WRITE_DST16(ptr, v) \
    do { \
        BLOCK_WRITE_DST8(ptr, (v).s01234567); \
        BLOCK_WRITE_DST8(ptr + 8 * SUB_GROUP_SIZE, (v).s89abcdef); \
    } while (0)

#elif DST_DT_F16 || DST_DT_BF16

#define BLOCK_READ_DST(ptr) \
    AS_DST_DATA_T(intel_sub_group_block_read_us((__global ushort *)ptr))
#define BLOCK_WRITE_DST(ptr, v) \
    intel_sub_group_block_write_us((__global ushort *)ptr, as_ushort(v))

#define BLOCK_READ_DST2(ptr) \
    AS_DST_DATA2_T(intel_sub_group_block_read_us2((__global ushort *)ptr))
#define BLOCK_WRITE_DST2(ptr, v) \
    intel_sub_group_block_write_us2((__global ushort *)ptr, as_short2(v))

#define BLOCK_READ_DST4(ptr) \
    AS_DST_DATA4_T(intel_sub_group_block_read_us4((__global ushort *)ptr))
#define BLOCK_WRITE_DST4(ptr, v) \
    intel_sub_group_block_write_us4((__global ushort *)ptr, as_ushort4(v))

#define BLOCK_READ_DST8(ptr) \
    AS_DST_DATA8_T(intel_sub_group_block_read_us8((__global ushort *)ptr))
#define BLOCK_WRITE_DST8(ptr, v) \
    intel_sub_group_block_write_us8((__global ushort *)ptr, as_ushort8(v))

#define BLOCK_READ_DST16(ptr) \
    (DST_DATA16_T)( \
            BLOCK_READ_DST8(ptr), BLOCK_READ_DST8(ptr + 8 * SUB_GROUP_SIZE))
#define BLOCK_WRITE_DST16(ptr, v) \
    do { \
        BLOCK_WRITE_DST8(ptr, (v).s01234567); \
        BLOCK_WRITE_DST8(ptr + 8 * SUB_GROUP_SIZE, (v).s89abcdef); \
    } while (0)

#endif

#if DST_DT_BF16
#define DST_TO_REF(x) cvt_bf16_to_f32(x)
#define DST_TO_REF2(x) cvt_bf16_to_f32(x)
#define DST_TO_REF8(x) cvt_bf16_to_f32(x)
#define REF_TO_DST(x) cvt_f32_to_bf16(x)
#define REF_TO_DST8(x) cvt_f32_to_bf16(convert_float8(x))
#elif DST_DT_BF8
#define DST_TO_REF(x) convert_float(cvt_f8_e5m2_to_hf(x))
#define DST_TO_REF2(x) convert_float(cvt_f8_e5m2_to_hf(x))
#define DST_TO_REF8(x) convert_float(cvt_f8_e5m2_to_hf(x))
#define REF_TO_DST(x) cvt_hf_to_f8_e5m2(convert_half(x))
#define REF_TO_DST8(x) cvt_hf_to_f8_e5m2(convert_half8(x))
#elif DST_DT_HF8
#define DST_TO_REF(x) convert_float(cvt_f8_e4m3_to_hf(x))
#define DST_TO_REF2(x) convert_float(cvt_f8_e4m3_to_hf(x))
#define DST_TO_REF8(x) convert_float(cvt_f8_e4m3_to_hf(x))
#define REF_TO_DST(x) cvt_hf_to_f8_e4m3(convert_half(x))
#define REF_TO_DST8(x) cvt_hf_to_f8_e4m3(convert_half8(x))
#elif DST_DT_F16
#define REF_TO_DST(x) convert_half(x)
#define DST_TO_REF(x) convert_float(x)
#define DST_TO_REF2(x) convert_float2(x)
#define DST_TO_REF8(x) convert_float8(x)
#elif DST_DT_U8
#define DST_TO_REF(x) (x)
#define DST_TO_REF2(x) (x)
#define DST_TO_REF8(x) (x)
#define REF_TO_DST(x) convert_uchar(x)
#define REF_TO_DST8(x) convert_uchar8(x)
#elif DST_DT_S8
#define DST_TO_REF(x) (x)
#define DST_TO_REF2(x) (x)
#define DST_TO_REF8(x) (x)
#define REF_TO_DST(x) convert_char(x)
#define REF_TO_DST8(x) convert_char8(x)
#else
#define DST_TO_REF(x) (x)
#define DST_TO_REF2(x) (x)
#define DST_TO_REF8(x) (x)
#define REF_TO_DST(x) (x)
#define REF_TO_DST8(x) (x)
#endif

#if DST_DT_F64
#define TO_DST(x) convert_double(x)
#define TO_DST2(x) convert_double2(x)
#define TO_DST4(x) convert_double4(x)
#define TO_DST8(x) convert_double8(x)
#elif DST_DT_BF16
#define TO_DST(x) cvt_f32_to_bf16(convert_float(x))
#define TO_DST2(x) cvt_f32_to_bf16(convert_float2(x))
#define TO_DST4(x) cvt_f32_to_bf16(convert_float4(x))
#define TO_DST8(x) cvt_f32_to_bf16(convert_float8(x))
#elif DST_DT_F16
#define TO_DST(x) convert_half(x)
#define TO_DST2(x) convert_half2(x)
#define TO_DST4(x) convert_half4(x)
#define TO_DST8(x) convert_half8(x)
#elif DST_DT_BF8
#define TO_DST(x) cvt_hf_to_f8_e5m2(convert_half(x))
#define TO_DST2(x) cvt_hf_to_f8_e5m2(convert_half2(x))
#define TO_DST4(x) cvt_hf_to_f8_e5m2(convert_half4(x))
#define TO_DST8(x) cvt_hf_to_f8_e5m2(convert_half8(x))
#define TO_DST16(x) cvt_hf_to_f8_e5m2(convert_half16(x))
#elif DST_DT_HF8
#define TO_DST(x) cvt_hf_to_f8_e4m3(convert_half(x))
#define TO_DST2(x) cvt_hf_to_f8_e4m3(convert_half2(x))
#define TO_DST4(x) cvt_hf_to_f8_e4m3(convert_half4(x))
#define TO_DST8(x) cvt_hf_to_f8_e4m3(convert_half8(x))
#define TO_DST16(x) cvt_hf_to_f8_e4m3(convert_half16(x))
#elif DST_DT_U4
#define SET_DOUBLE_HALF_BYTE(x, y, z) set_double_half_byte(x, y, z)
#define TO_DST(x) cvt_f32_to_u4(convert_float(x))
#define TO_DST2(x) cvt_f32_to_u4(convert_float2(x))
#define TO_DST4(x) cvt_f32_to_u4(convert_float4(x))
#define TO_DST8(x) cvt_f32_to_u4(convert_float8(x))
#define TO_DST16(x) cvt_f32_to_u4(convert_float16(x))
#elif DST_DT_S4
#define SET_DOUBLE_HALF_BYTE(x, y, z) set_double_half_byte(x, y, z)
#define TO_DST(x) cvt_f32_to_s4(convert_float(x))
#define TO_DST2(x) cvt_f32_to_s4(convert_float2(x))
#define TO_DST4(x) cvt_f32_to_s4(convert_float4(x))
#define TO_DST8(x) cvt_f32_to_s4(convert_float8(x))
#define TO_DST16(x) cvt_f32_to_s4(convert_float16(x))
#elif DST_DT_U8
#define TO_DST(x) convert_uchar_sat_rte(x)
#define TO_DST2(x) convert_uchar2_sat_rte(x)
#define TO_DST4(x) convert_uchar4_sat_rte(x)
#define TO_DST8(x) convert_uchar8_sat_rte(x)
#define TO_DST16(x) convert_uchar16_sat_rte(x)
#elif DST_DT_S8
#define TO_DST(x) convert_char_sat_rte(x)
#define TO_DST2(x) convert_char2_sat_rte(x)
#define TO_DST4(x) convert_char4_sat_rte(x)
#define TO_DST8(x) convert_char8_sat_rte(x)
#define TO_DST16(x) convert_char16_sat_rte(x)
#elif DST_DT_S32
#define TO_DST(x) convert_int_sat_rte(x)
#define TO_DST2(x) convert_int2_sat_rte(x)
#define TO_DST4(x) convert_int4_sat_rte(x)
#define TO_DST8(x) convert_int8_sat_rte(x)
#elif DST_DT_F32
#define TO_DST(x) convert_float(x)
#define TO_DST2(x) convert_float2(x)
#define TO_DST4(x) convert_float4(x)
#define TO_DST8(x) convert_float8(x)
#elif DST_DT_F64
#define TO_DST(x) convert_double(x)
#define TO_DST2(x) convert_double2(x)
#define TO_DST4(x) convert_double4(x)
#define TO_DST8(x) convert_double8(x)
#else
#error "Not expected"
#endif
#endif

#ifdef C_DATA_T
#define C_DATA8_T CONCAT2(C_DATA_T, 8)
#if C_DT_BF16
#define C_TO_REF(x) cvt_bf16_to_f32(x)
#define C_TO_REF8(x) cvt_bf16_to_f32(x)
#define REF_TO_C(x) cvt_f32_to_bf16(x)
#define REF_TO_C8(x) cvt_f32_to_bf16(convert_float8(x))
#elif C_DT_BF8
#define C_TO_REF(x) cvt_f8_e5m2_to_hf(convert_half(x))
#define C_TO_REF8(x) cvt_f8_e5m2_to_hf(convert_half8(x))
#define REF_TO_C(x) cvt_hf_to_f8_e5m2(convert_half(x))
#define REF_TO_C8(x) cvt_hf_to_f8_e5m2(convert_half8(x))
#elif C_DT_HF8
#define C_TO_REF(x) cvt_f8_e4m3_to_hf(convert_half(x))
#define C_TO_REF8(x) cvt_f8_e4m3_to_hf(convert_half8(x))
#define REF_TO_C(x) cvt_hf_to_f8_e4m3(convert_half(x))
#define REF_TO_C8(x) cvt_hf_to_f8_e4m3(convert_half8(x))
#else
#define C_TO_REF(x) (x)
#define C_TO_REF8(x) (x)
#define REF_TO_C(x) (x)
#define REF_TO_C8(x) (x)
#endif
#if C_DT_BF16
#define TO_C(x) cvt_f32_to_bf16(x)
#define TO_C8(x) cvt_f32_to_bf16(convert_float8(x))
#elif C_DT_BF8
#define TO_C(x) cvt_hf_to_f8_e5m2(convert_half(x))
#define TO_C8(x) cvt_hf_to_f8_e5m2(convert_half8(x))
#elif C_DT_HF8
#define TO_C(x) cvt_hf_to_f8_e4m3(convert_half(x))
#define TO_C8(x) cvt_hf_to_f8_e4m3(convert_half8(x))
#elif C_DT_F16
#define TO_C(x) convert_half(x)
#define TO_C8(x) convert_half8(x)
#elif C_DT_U8
#define TO_C(x) convert_uchar_sat_rte(x)
#define TO_C8(x) convert_uchar8_sat_rte(x)
#elif C_DT_S8
#define TO_C(x) convert_char_sat_rte(x)
#define TO_C8(x) convert_char8_sat_rte(x)
#elif C_DT_S32
#define TO_C(x) convert_int_sat_rte(x)
#define TO_C8(x) convert_int8_sat_rte(x)
#elif C_DT_F32
#define TO_C(x) convert_float(x)
#define TO_C8(x) convert_float8(x)
#elif C_DT_F64
#define TO_C(x) convert_double(x)
#define TO_C8(x) convert_double8(x)
#else
#error "Not expected"
#endif
#endif

#ifdef ACC_DATA_T
#if ACC_DT_F16
#define TO_ACC(x) convert_half(x)
#elif ACC_DT_F32
#define TO_ACC(x) convert_float(x)
#elif ACC_DT_F64
#define TO_ACC(x) convert_double(x)
#elif ACC_DT_S32
#define TO_ACC(x) convert_int(x)
#else
#error "Unexpected accumulation data type"
#endif
#endif

#ifdef SUM_DATA_T
#define SUM_DATA2_T CONCAT2(SUM_DATA_T, 2)
#define SUM_DATA4_T CONCAT2(SUM_DATA_T, 4)
#define SUM_DATA8_T CONCAT2(SUM_DATA_T, 8)
#define SUM_DATA16_T CONCAT2(SUM_DATA_T, 16)
#define AS_SUM_DATA_T CONCAT2(as_, SUM_DATA_T)
#define AS_SUM_DATA2_T CONCAT2(as_, SUM_DATA2_T)
#define AS_SUM_DATA4_T CONCAT2(as_, SUM_DATA4_T)
#define AS_SUM_DATA8_T CONCAT2(as_, SUM_DATA8_T)
#define AS_SUM_DATA16_T CONCAT2(as_, SUM_DATA16_T)
#if SUM_DT_BF16
#define SUM_TO_REF cvt_bf16_to_f32
#elif SUM_DT_BF8
#define SUM_TO_REF(x) convert_float(cvt_f8_e5m2_to_hf(x))
#elif SUM_DT_HF8
#define SUM_TO_REF(x) convert_float(cvt_f8_e4m3_to_hf(x))
#else
#define SUM_TO_REF
#endif
#endif

#define OFF_MD_2(prefix, x0, x1, x2, x3, x4, x5) \
    ((((x0) / CONCAT2(prefix, _B0_2)) / CONCAT2(prefix, _B0_1) \
             * CONCAT2(prefix, _S0_0)) \
            + (((x0) / CONCAT2(prefix, _B0_2)) % CONCAT2(prefix, _B0_1) \
                    * CONCAT2(prefix, _S0_1)) \
            + (((x0) % CONCAT2(prefix, _B0_2)) * CONCAT2(prefix, _S0_2)) \
            + (((x1) / CONCAT2(prefix, _B1_2)) / CONCAT2(prefix, _B1_1) \
                    * CONCAT2(prefix, _S1_0)) \
            + (((x1) / CONCAT2(prefix, _B1_2)) % CONCAT2(prefix, _B1_1) \
                    * CONCAT2(prefix, _S1_1)) \
            + (((x1) % CONCAT2(prefix, _B1_2)) * CONCAT2(prefix, _S1_2)) \
            + (((x2) / CONCAT2(prefix, _B2_2)) / CONCAT2(prefix, _B2_1) \
                    * CONCAT2(prefix, _S2_0)) \
            + (((x2) / CONCAT2(prefix, _B2_2)) % CONCAT2(prefix, _B2_1) \
                    * CONCAT2(prefix, _S2_1)) \
            + (((x2) % CONCAT2(prefix, _B2_2)) * CONCAT2(prefix, _S2_2)) \
            + (((x3) / CONCAT2(prefix, _B3_2)) / CONCAT2(prefix, _B3_1) \
                    * CONCAT2(prefix, _S3_0)) \
            + (((x3) / CONCAT2(prefix, _B3_2)) % CONCAT2(prefix, _B3_1) \
                    * CONCAT2(prefix, _S3_1)) \
            + (((x3) % CONCAT2(prefix, _B3_2)) * CONCAT2(prefix, _S3_2)) \
            + (((x4) / CONCAT2(prefix, _B4_2)) / CONCAT2(prefix, _B4_1) \
                    * CONCAT2(prefix, _S4_0)) \
            + (((x4) / CONCAT2(prefix, _B4_2)) % CONCAT2(prefix, _B4_1) \
                    * CONCAT2(prefix, _S4_1)) \
            + (((x4) % CONCAT2(prefix, _B4_2)) * CONCAT2(prefix, _S4_2)) \
            + (((x5) / CONCAT2(prefix, _B5_2)) / CONCAT2(prefix, _B5_1) \
                    * CONCAT2(prefix, _S5_0)) \
            + (((x5) / CONCAT2(prefix, _B5_2)) % CONCAT2(prefix, _B5_1) \
                    * CONCAT2(prefix, _S5_1)) \
            + (((x5) % CONCAT2(prefix, _B5_2)) * CONCAT2(prefix, _S5_2)))

#define OFF_MD_3(prefix, x0, x1, x2, x3, x4, x5) \
    ((((((x0) / CONCAT2(prefix, _B0_3)) / CONCAT2(prefix, _B0_2)) \
              / CONCAT2(prefix, _B0_1)) \
             * CONCAT2(prefix, _S0_0)) \
            + (((((x0) / CONCAT2(prefix, _B0_3)) / CONCAT2(prefix, _B0_2)) \
                       % CONCAT2(prefix, _B0_1)) \
                    * CONCAT2(prefix, _S0_1)) \
            + ((((x0) / CONCAT2(prefix, _B0_3)) % CONCAT2(prefix, _B0_2)) \
                    * CONCAT2(prefix, _S0_2)) \
            + (((x0) % CONCAT2(prefix, _B0_3)) * CONCAT2(prefix, _S0_3)) \
            + (((((x1) / CONCAT2(prefix, _B1_3)) / CONCAT2(prefix, _B1_2)) \
                       / CONCAT2(prefix, _B1_1)) \
                    * CONCAT2(prefix, _S1_0)) \
            + (((((x1) / CONCAT2(prefix, _B1_3)) / CONCAT2(prefix, _B1_2)) \
                       % CONCAT2(prefix, _B1_1)) \
                    * CONCAT2(prefix, _S1_1)) \
            + ((((x1) / CONCAT2(prefix, _B1_3)) % CONCAT2(prefix, _B1_2)) \
                    * CONCAT2(prefix, _S1_2)) \
            + (((x1) % CONCAT2(prefix, _B1_3)) * CONCAT2(prefix, _S1_3)) \
            + (((((x2) / CONCAT2(prefix, _B2_3)) / CONCAT2(prefix, _B2_2)) \
                       / CONCAT2(prefix, _B2_1)) \
                    * CONCAT2(prefix, _S2_0)) \
            + (((((x2) / CONCAT2(prefix, _B2_3)) / CONCAT2(prefix, _B2_2)) \
                       % CONCAT2(prefix, _B2_1)) \
                    * CONCAT2(prefix, _S2_1)) \
            + ((((x2) / CONCAT2(prefix, _B2_3)) % CONCAT2(prefix, _B2_2)) \
                    * CONCAT2(prefix, _S2_2)) \
            + (((x2) % CONCAT2(prefix, _B2_3)) * CONCAT2(prefix, _S2_3)) \
            + (((((x3) / CONCAT2(prefix, _B3_3)) / CONCAT2(prefix, _B3_2)) \
                       / CONCAT2(prefix, _B3_1)) \
                    * CONCAT2(prefix, _S3_0)) \
            + (((((x3) / CONCAT2(prefix, _B3_3)) / CONCAT2(prefix, _B3_2)) \
                       % CONCAT2(prefix, _B3_1)) \
                    * CONCAT2(prefix, _S3_1)) \
            + ((((x3) / CONCAT2(prefix, _B3_3)) % CONCAT2(prefix, _B3_2)) \
                    * CONCAT2(prefix, _S3_2)) \
            + (((x3) % CONCAT2(prefix, _B3_3)) * CONCAT2(prefix, _S3_3)) \
            + (((((x4) / CONCAT2(prefix, _B4_3)) / CONCAT2(prefix, _B4_2)) \
                       / CONCAT2(prefix, _B4_1)) \
                    * CONCAT2(prefix, _S4_0)) \
            + (((((x4) / CONCAT2(prefix, _B4_3)) / CONCAT2(prefix, _B4_2)) \
                       % CONCAT2(prefix, _B4_1)) \
                    * CONCAT2(prefix, _S4_1)) \
            + ((((x4) / CONCAT2(prefix, _B4_3)) % CONCAT2(prefix, _B4_2)) \
                    * CONCAT2(prefix, _S4_2)) \
            + (((x4) % CONCAT2(prefix, _B4_3)) * CONCAT2(prefix, _S4_3)) \
            + (((((x5) / CONCAT2(prefix, _B5_3)) / CONCAT2(prefix, _B5_2)) \
                       / CONCAT2(prefix, _B5_1)) \
                    * CONCAT2(prefix, _S5_0)) \
            + (((((x5) / CONCAT2(prefix, _B5_3)) / CONCAT2(prefix, _B5_2)) \
                       % CONCAT2(prefix, _B5_1)) \
                    * CONCAT2(prefix, _S5_1)) \
            + ((((x5) / CONCAT2(prefix, _B5_3)) % CONCAT2(prefix, _B5_2)) \
                    * CONCAT2(prefix, _S5_2)) \
            + (((x5) % CONCAT2(prefix, _B5_3)) * CONCAT2(prefix, _S5_3)))

#define OFF_MD(prefix, x0, x1, x2, x3, x4, x5) \
    CONCAT2(OFF_MD_, CONCAT2(prefix, _NLEVELS))(prefix, x0, x1, x2, x3, x4, x5)

#if SRC_NDIMS == 3
#define CONV_SRC_OFF(n, c, d, h, w) OFF_MD(SRC, n, c, w, 0, 0, 0)
#elif SRC_NDIMS == 4
#define CONV_SRC_OFF(n, c, d, h, w) OFF_MD(SRC, n, c, h, w, 0, 0)
#elif SRC_NDIMS == 5
#define CONV_SRC_OFF(n, c, d, h, w) OFF_MD(SRC, n, c, d, h, w, 0)
#endif

#if WEI_NDIMS == 3
#define CONV_WEI_OFF(g, o, i, d, h, w) OFF_MD(WEI, o, i, w, 0, 0, 0)
#elif WEI_NDIMS == 4
#if WITH_GROUPS == 0
#define CONV_WEI_OFF(g, o, i, d, h, w) OFF_MD(WEI, o, i, h, w, 0, 0)
#else
#define CONV_WEI_OFF(g, o, i, d, h, w) OFF_MD(WEI, g, o, i, w, 0, 0)
#endif
#elif WEI_NDIMS == 5
#if WITH_GROUPS == 0
#define CONV_WEI_OFF(g, o, i, d, h, w) OFF_MD(WEI, o, i, d, h, w, 0)
#else
#define CONV_WEI_OFF(g, o, i, d, h, w) OFF_MD(WEI, g, o, i, h, w, 0)
#endif
#elif WEI_NDIMS == 6
#define CONV_WEI_OFF(g, o, i, d, h, w) OFF_MD(WEI, g, o, i, d, h, w)
#endif

#if DST_NDIMS == 3
#define CONV_DST_OFF(n, c, d, h, w) OFF_MD(DST, n, c, w, 0, 0, 0)
#elif DST_NDIMS == 4
#define CONV_DST_OFF(n, c, d, h, w) OFF_MD(DST, n, c, h, w, 0, 0)
#elif DST_NDIMS == 5
#define CONV_DST_OFF(n, c, d, h, w) OFF_MD(DST, n, c, d, h, w, 0)
#endif

#if NDIMS == 2
#define SRC_OFF(x0, x1, d, h, w) \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 \
            + ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1)

#if WITH_GROUPS == 1
#define WEI_OFF(x0, x1, x2, d, h, w) \
    (((x0) % WEI_B0) * WEI_SB0 + ((x0) / WEI_B0) * WEI_S0 \
            + ((x1) % WEI_B1) * WEI_SB1 + ((x1) / WEI_B1) * WEI_S1 \
            + ((x2) % WEI_B2) * WEI_SB2 + ((x2) / WEI_B2) * WEI_S2)
#else
#define WEI_OFF(g, x0, x1, d, h, w) \
    (((x0) % WEI_B0) * WEI_SB0 + ((x0) / WEI_B0) * WEI_S0 \
            + ((x1) % WEI_B1) * WEI_SB1 + ((x1) / WEI_B1) * WEI_S1)
#endif

#define DST_OFF(x0, x1, d, h, w) \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 \
            + ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1)
#elif NDIMS == 3
#define SRC_OFF(x0, x1, d, h, x2) \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 \
            + ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1 \
            + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2)

#if WITH_GROUPS == 1
#define WEI_OFF(x0, x1, x2, d, h, x3) \
    (((x0) % WEI_B0) * WEI_SB0 + ((x0) / WEI_B0) * WEI_S0 \
            + ((x1) % WEI_B1) * WEI_SB1 + ((x1) / WEI_B1) * WEI_S1 \
            + ((x2) % WEI_B2) * WEI_SB2 + ((x2) / WEI_B2) * WEI_S2 \
            + ((x3) % WEI_B3) * WEI_SB3 + ((x3) / WEI_B3) * WEI_S3)
#else
#define WEI_OFF(g, x0, x1, d, h, x2) \
    (((x0) % WEI_B0) * WEI_SB0 + ((x0) / WEI_B0) * WEI_S0 \
            + ((x1) % WEI_B1) * WEI_SB1 + ((x1) / WEI_B1) * WEI_S1 \
            + ((x2) % WEI_B2) * WEI_SB2 + ((x2) / WEI_B2) * WEI_S2)
#endif

#define DST_OFF(x0, x1, d, h, x2) \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 \
            + ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1 \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2)
#elif NDIMS == 4
#define SRC_OFF(x0, x1, d, x2, x3) \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 \
            + ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1 \
            + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2 \
            + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3)

#if WITH_GROUPS == 1
#define WEI_OFF(x0, x1, x2, d, x3, x4) \
    (((x0) % WEI_B0) * WEI_SB0 + ((x0) / WEI_B0) * WEI_S0 \
            + ((x1) % WEI_B1) * WEI_SB1 + ((x1) / WEI_B1) * WEI_S1 \
            + ((x2) % WEI_B2) * WEI_SB2 + ((x2) / WEI_B2) * WEI_S2 \
            + ((x3) % WEI_B3) * WEI_SB3 + ((x3) / WEI_B3) * WEI_S3 \
            + ((x4) % WEI_B4) * WEI_SB4 + ((x4) / WEI_B4) * WEI_S4)
#else
#define WEI_OFF(g, x1, x2, d, x3, x4) \
    (((x1) % WEI_B0) * WEI_SB0 + ((x1) / WEI_B0) * WEI_S0 \
            + ((x2) % WEI_B1) * WEI_SB1 + ((x2) / WEI_B1) * WEI_S1 \
            + ((x3) % WEI_B2) * WEI_SB2 + ((x3) / WEI_B2) * WEI_S2 \
            + ((x4) % WEI_B3) * WEI_SB3 + ((x4) / WEI_B3) * WEI_S3)
#endif

#define DST_OFF(x0, x1, d, x2, x3) \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 \
            + ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1 \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2 \
            + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3)
#elif NDIMS == 5
#define SRC_OFF(x0, x1, x2, x3, x4) \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 \
            + ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1 \
            + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2 \
            + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3 \
            + ((x4) % SRC_B4) * SRC_SB4 + ((x4) / SRC_B4) * SRC_S4)

#if WITH_GROUPS == 1
#define WEI_OFF(x0, x1, x2, x3, x4, x5) \
    (((x0) % WEI_B0) * WEI_SB0 + ((x0) / WEI_B0) * WEI_S0 \
            + ((x1) % WEI_B1) * WEI_SB1 + ((x1) / WEI_B1) * WEI_S1 \
            + ((x2) % WEI_B2) * WEI_SB2 + ((x2) / WEI_B2) * WEI_S2 \
            + ((x3) % WEI_B3) * WEI_SB3 + ((x3) / WEI_B3) * WEI_S3 \
            + ((x4) % WEI_B4) * WEI_SB4 + ((x4) / WEI_B4) * WEI_S4 \
            + ((x5) % WEI_B5) * WEI_SB5 + ((x5) / WEI_B5) * WEI_S5)
#else
#define WEI_OFF(g, x1, x2, x3, x4, x5) \
    (((x1) % WEI_B0) * WEI_SB0 + ((x1) / WEI_B0) * WEI_S0 \
            + ((x2) % WEI_B1) * WEI_SB1 + ((x2) / WEI_B1) * WEI_S1 \
            + ((x3) % WEI_B2) * WEI_SB2 + ((x3) / WEI_B2) * WEI_S2 \
            + ((x4) % WEI_B3) * WEI_SB3 + ((x4) / WEI_B3) * WEI_S3 \
            + ((x5) % WEI_B4) * WEI_SB4 + ((x5) / WEI_B4) * WEI_S4)
#endif

#define DST_OFF(x0, x1, x2, x3, x4) \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 \
            + ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1 \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2 \
            + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3 \
            + ((x4) % DST_B4) * DST_SB4 + ((x4) / DST_B4) * DST_S4)
#endif

#if SRC_DT_U8 == 1
#define SRC_DT_ALIAS UCHAR
#elif SRC_DT_S8 == 1
#define SRC_DT_ALIAS CHAR
#elif SRC_DT_F16 == 1
#define SRC_DT_ALIAS HALF
#elif SRC_DT_BF16 == 1
#define SRC_DT_ALIAS BFLOAT
#elif SRC_DT_F32 == 1
#define SRC_DT_ALIAS FLOAT
#elif SRC_DT_F64 == 1
#define SRC_DT_ALIAS DOUBLE
#endif

#if DST_DT_U8 == 1
#define DST_DT_ALIAS UCHAR
#elif DST_DT_S8 == 1
#define DST_DT_ALIAS CHAR
#elif DST_DT_F16 == 1
#define DST_DT_ALIAS HALF
#elif DST_DT_BF16 == 1
#define DST_DT_ALIAS BFLOAT
#elif DST_DT_F32 == 1
#define DST_DT_ALIAS FLOAT
#elif DST_DT_F64 == 1
#define DST_DT_ALIAS DOUBLE
#endif

#define ALIAS(prefix) CONCAT2(prefix, _DT_ALIAS)

// BLOCK types
#define BLOCK1_T uchar
#define BLOCK2_T ushort
#define BLOCK4_T uint
#define BLOCK8_T ulong
#define BLOCK_T(alias) CONCAT3(BLOCK, SIZEOF(alias), _T)

#define BLOCK1_ALIAS UCHAR
#define BLOCK2_ALIAS USHORT
#define BLOCK4_ALIAS UINT
#define BLOCK8_ALIAS ULONG
#define BLOCK_ALIAS(prefix) CONCAT3(BLOCK, SIZEOF(ALIAS(prefix)), _ALIAS)

#define SIZEOF_UCHAR 1
#define SIZEOF_CHAR 1
#define SIZEOF_BFLOAT 2
#define SIZEOF_HALF 2
#define SIZEOF_FLOAT 4
#define SIZEOF_DOUBLE 8
#define SIZEOF(alias) CONCAT2(SIZEOF_, alias)

#define READ_UCHAR8 intel_sub_group_block_read_uc8
#define READ_USHORT8 intel_sub_group_block_read_us8
#define READ_UINT8 intel_sub_group_block_read8
#define READ_ULONG8 intel_sub_group_block_read_ul8

#define READ_BLOCK8(prefix, ptr) READ_BLOCK_N(prefix, 8)(ptr)
#define READ_BLOCK_N(prefix, n) CONCAT3(READ_, BLOCK_ALIAS(prefix), n)

#define WRITE_UCHAR8 intel_sub_group_block_write_uc8
#define WRITE_USHORT8 intel_sub_group_block_write_us8
#define WRITE_UINT8 intel_sub_group_block_write8
#define WRITE_ULONG8 intel_sub_group_block_write_ul8

#define WRITE_BLOCK8(prefix, ptr, val) WRITE_BLOCK_N(prefix, 8)(ptr, val)
#define WRITE_BLOCK_N(prefix, n) CONCAT3(WRITE_, BLOCK_ALIAS(prefix), n)

#define AS_UCHAR8 as_uchar8
#define AS_CHAR8 as_char8
#define AS_HALF8 as_half8
#define AS_USHORT8 as_ushort8
#define AS_BFLOAT8 as_ushort8
#define AS_FLOAT8 as_float8
#define AS_INT8 as_int8
#define AS_UINT8 as_uint8
#define AS_DOUBLE8 as_double8

#define BLOCK_TO_DATA8(prefix, val) BLOCK_TO_DATA_N(prefix, 8)(val)
#define BLOCK_TO_DATA_N(prefix, n) CONCAT3(AS_, ALIAS(prefix), n)

#define DATA_TO_BLOCK8(prefix, val) DATA_TO_BLOCK_N(prefix, 8)(val)
#define DATA_TO_BLOCK_N(prefix, n) CONCAT3(AS_, BLOCK_ALIAS(prefix), n)

#define UCHAR_TO_FLOAT1 convert_float
#define UCHAR_TO_FLOAT8 convert_float8
#define FLOAT_TO_UCHAR1 convert_uchar_sat_rte
#define FLOAT_TO_UCHAR8 convert_uchar8_sat_rte

#define CHAR_TO_FLOAT1 convert_float
#define CHAR_TO_FLOAT8 convert_float8
#define FLOAT_TO_CHAR1 convert_char_sat_rte
#define FLOAT_TO_CHAR8 convert_char8_sat_rte

#define HALF_TO_FLOAT1 convert_float
#define HALF_TO_FLOAT8 convert_float8

#define HALF_TO_DOUBLE1 convert_double
#define HALF_TO_DOUBLE8 convert_double8

#define FLOAT_TO_HALF1 convert_half
#define FLOAT_TO_HALF8 convert_half8

#define BFLOAT_TO_DOUBLE1 cvt_bf16_to_f64
#define BFLOAT_TO_DOUBLE8 cvt_bf16_to_f64

#define BFLOAT_TO_FLOAT1 cvt_bf16_to_f32
#define BFLOAT_TO_FLOAT8 cvt_bf16_to_f32
#define FLOAT_TO_BFLOAT1 cvt_f32_to_bf16
#define FLOAT_TO_BFLOAT8 cvt_f32_to_bf16

#define FLOAT_TO_FLOAT1 convert_float
#define FLOAT_TO_FLOAT8 convert_float8

#define FLOAT_TO_DOUBLE1 convert_double
#define FLOAT_TO_DOUBLE8 convert_double8

#define DOUBLE_TO_DOUBLE1 convert_double
#define DOUBLE_TO_DOUBLE8 convert_double8

#define DOUBLE_TO_FLOAT1 convert_float
#define DOUBLE_TO_FLOAT8 convert_float8

#define DATA_TO_FLOAT(prefix, val) DATA_TO_FLOAT_N(prefix, 1)(val)
#define DATA_TO_FLOAT8(prefix, val) DATA_TO_FLOAT_N(prefix, 8)(val)
#define DATA_TO_FLOAT_N(prefix, n) CONCAT3(ALIAS(prefix), _TO_FLOAT, n)

#define DATA_TO_DOUBLE(prefix, val) DATA_TO_DOUBLE_N(prefix, 1)(val)
#define DATA_TO_DOUBLE8(prefix, val) DATA_TO_DOUBLE_N(prefix, 8)(val)
#define DATA_TO_DOUBLE_N(prefix, n) CONCAT3(ALIAS(prefix), _TO_DOUBLE, n)

#define FLOAT_TO_DATA(prefix, val) FLOAT_TO_DATA_N(prefix, 1)(val)
#define FLOAT_TO_DATA8(prefix, val) FLOAT_TO_DATA_N(prefix, 8)(val)
#define FLOAT_TO_DATA_N(prefix, n) CONCAT3(FLOAT_TO_, ALIAS(prefix), n)

#define DOUBLE_TO_DATA(prefix, val) DOUBLE_TO_DATA_N(prefix, 1)(val)
#define DOUBLE_TO_DATA8(prefix, val) DOUBLE_TO_DATA_N(prefix, 8)(val)
#define DOUBLE_TO_DATA_N(prefix, n) CONCAT3(DOUBLE_TO_, ALIAS(prefix), n)

#endif
