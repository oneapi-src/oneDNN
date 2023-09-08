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

#ifndef GPU_OCL_TYPES_INTEROP_H
#define GPU_OCL_TYPES_INTEROP_H

// Structures intended for use in both OpenCL C and C++ code. This is especially
// helpful for parameter transfer.

// Due to limitations in ocl_gpu_engine.cpp:preprocess_headers(), <cstdint> must
// be included external to this file when compiling outside of an OpenCL context
#ifdef __OPENCL_VERSION__
typedef uchar uint8_t;
typedef long int64_t;
#endif

typedef struct {
    int64_t array[3];
} int64x3_t;

// Number of terms usable by a single dispatcher, each defined in ocl_types.h
#define MAX_INDEXING_TERMS 10

typedef struct {
    int64_t sizes[MAX_INDEXING_TERMS];
    int64_t strides[MAX_INDEXING_TERMS];
    int64_t blocks[MAX_INDEXING_TERMS];
} dispatch_gws_rt_params_t;

#define ZERO_PAD_MASK_DATA_TYPE uint8_t
#define ZERO_PAD_MASK_DT_BITS (8 * sizeof(ZERO_PAD_MASK_DATA_TYPE))
#define ZERO_PAD_MAX_STEP_SIZE 1536

#define ZERO_PAD_MASK_SIZE (ZERO_PAD_MAX_STEP_SIZE / ZERO_PAD_MASK_DT_BITS)
#define ZERO_PAD_BIT_MODE 0
#define ZERO_PAD_LOOKUP_MODE 1

typedef struct {
    ZERO_PAD_MASK_DATA_TYPE mask[ZERO_PAD_MASK_SIZE];
} zero_pad_mask_t;

#endif
