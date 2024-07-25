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

#ifndef GPU_INTEL_OCL_SIMPLE_SOFTMAX_H
#define GPU_INTEL_OCL_SIMPLE_SOFTMAX_H

#include "gpu/intel/ocl/ocl_post_ops.h"
#include "gpu/intel/ocl/ocl_types.h"

#define CONCAt2(a, b) a##b
#define CONCAT2(a, b) CONCAt2(a, b)

#define IS_HALF_half 1
#define IS_HALF(dt) CONCAT2(IS_HALF_, dt)

#if IS_FWD
#define DD(i) CONCAt2(DST_D, i)
#elif IS_BWD
#define DD(i) CONCAt2(SRC_D, i)
#else
#error unsupported data parameter
#endif

#define OFF(dim, idx) \
    (dim % CONCAT2(DATA_B, idx)) * CONCAT2(DATA_SB, idx) \
            + (dim / CONCAT2(DATA_B, idx)) * CONCAT2(DATA_S, idx)

#if SOFTMAX_AXIS_IDX == 0
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(softmax_dim, 0) + OFF(dim0, 1) + OFF(dim1, 2) + OFF(dim2, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    softmax_dim >= DD(0) || dim0 >= DD(1) || dim1 >= DD(2) || dim2 >= DD(3) \
            || dim3 >= DD(4) || dim4 >= DD(5)
#elif SOFTMAX_AXIS_IDX == 1
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(softmax_dim, 1) + OFF(dim1, 2) + OFF(dim2, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    dim0 >= DD(0) || softmax_dim >= DD(1) || dim1 >= DD(2) || dim2 >= DD(3) \
            || dim3 >= DD(4) || dim4 >= DD(5)
#elif SOFTMAX_AXIS_IDX == 2
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(softmax_dim, 2) + OFF(dim2, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    dim0 >= DD(0) || dim1 >= DD(1) || softmax_dim >= DD(2) || dim2 >= DD(3) \
            || dim3 >= DD(4) || dim4 >= DD(5)
#elif SOFTMAX_AXIS_IDX == 3
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(dim2, 2) + OFF(softmax_dim, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    dim0 >= DD(0) || dim1 >= DD(1) || dim2 >= DD(2) || softmax_dim >= DD(3) \
            || dim3 >= DD(4) || dim4 >= DD(5)
#elif SOFTMAX_AXIS_IDX == 4
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(dim2, 2) + OFF(dim3, 3) \
            + OFF(softmax_dim, 4) + OFF(dim4, 5)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    dim0 >= DD(0) || dim1 >= DD(1) || dim2 >= DD(2) || dim3 >= DD(3) \
            || softmax_dim >= DD(4) || dim4 >= DD(5)
#elif SOFTMAX_AXIS_IDX == 5
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(dim2, 2) + OFF(dim3, 3) + OFF(dim4, 4) \
            + OFF(softmax_dim, 5)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    dim0 >= DD(0) || dim1 >= DD(1) || dim2 >= DD(2) || dim3 >= DD(3) \
            || dim4 >= DD(4) || softmax_dim >= DD(5)
#else
#error unsupported softmax dimension
#endif

#endif // end ifdef
