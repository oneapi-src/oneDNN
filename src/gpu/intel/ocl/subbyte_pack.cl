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

#define DT_UNDEF 1
#include "gpu/intel/ocl/ocl_types.h"
#include "gpu/intel/ocl/types_interop.h"

__kernel void subbyte_pack(__global uchar *restrict src,
        __global uchar *restrict dst, off_t n, int bits, int64x3_t offset) {
    const uchar mask = (1 << bits) - 1;

    const off_t dst_off = get_global_id(0) + offset.array[0];
    const off_t src_off = (8 / bits) * dst_off;

    uchar packed = 0;
    for (int i = 0, j = 0; i < 8; i += bits, ++j) {
        uchar byte = src_off + j < n ? src[src_off + j] : 0;
        packed |= (byte & mask) << i;
    }
    dst[dst_off] = packed;
}
