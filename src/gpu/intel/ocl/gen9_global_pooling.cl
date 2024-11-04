/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#include "gpu/intel/ocl/dispatch.h"
#include "gpu/intel/ocl/ocl_types.h"

#define ALG_AVG (ALG_AVG_NP || ALG_AVG_P)

#if IS_FWD
KERNEL_ATTR
__kernel void gen9_global_pooling_fwd(
        __global DATA_T *src, __global int *ws, __global DST_DATA_T *dst) {

    if (GWS_OVERFLOW) return;

    const off_t mb = get_global_id(0) / C;
    const off_t oc = get_global_id(0) % C;

    const off_t dst_off = DST_OFF(mb, oc, 0, 0, 0);

#if ALG_MAX
#if DT_BF16
    DEF_ACC_DATA_T dst_val = DATA_TO_REF(src[0]);
#else
    float dst_val = src[0];
#endif
#if IS_TRAINING
    off_t max_idx = -1;
#endif
#else
#if DT_BF16
    DEF_ACC_DATA_T dst_val = DATA_TO_REF(0.f);
#else
    float dst_val = 0.f;
#endif
#endif

    for (off_t id = 0; id < ID; id++) {
        for (off_t ih = 0; ih < IH; ih++) {
            for (off_t iw = 0; iw < IW; iw++) {
                off_t src_off = SRC_OFF(mb, oc, id, ih, iw);
#if DT_BF16
                DEF_ACC_DATA_T val = DATA_TO_REF(src[src_off]);
#else
                float val = DATA_TO_REF(src[src_off]);
#endif
#if ALG_MAX
                if (val > dst_val) {
                    dst_val = val;
#if IS_TRAINING
                    max_idx = id * IH * IW + ih * IW + iw;
#endif
                }
#else
                dst_val += val;
#endif
            }
        }
    }

#if ALG_MAX
    dst[dst_off] = TO_DST(dst_val);
#if IS_TRAINING
    ws[dst_off] = max_idx;
#endif
#else
    dst[dst_off] = TO_DST(dst_val / convert_float(ID * IH * IW));
#endif
}
#endif // IS_FWD

#if IS_BWD

#if DT_BF16 || DT_F16
#define DST_BLOCK_WRITE(dst, val) \
    BLOCK_WRITE((__global ushort *)(dst), as_ushort(val))
#endif // DT_BF16
#if DT_F32
#define DST_BLOCK_WRITE(dst, val) \
    BLOCK_WRITE((__global uint *)(dst), as_uint(val))
#endif // DT_F32

KERNEL_ATTR
__kernel void gen9_global_pooling_bwd(__global DATA_T *diff_src,
        __global int *ws, __global DATA_T *diff_dst) {

    if (GWS_OVERFLOW) return;

    const off_t mb = GWS_GET_MB();
#if IS_VECTORIZED
    const off_t c = GWS_GET_C() + get_sub_group_local_id();
#else
    const off_t c = GWS_GET_C();
#endif
    const off_t spatial = GWS_GET_SPATIAL();

    const bool is_in_padded_area = NEED_ZERO_PADDING && (mb >= MB || c >= C);
    const off_t dst_off = DST_OFF(mb, c, 0, 0, 0);
#if ALG_AVG
    // Read dst value only once
    const DATA_T dst_val = diff_dst[dst_off];
#endif // ALG_AVG
    int ws_val = ws[dst_off];
    for (off_t sp_idx = spatial;
            sp_idx < min(spatial + SPATIAL_CHUNK, (off_t)SPATIAL_DIM);
            sp_idx++) {
        const off_t iw = sp_idx % IW;
        const off_t ih = ((sp_idx - iw) % (IH * IW)) / IW;
        const off_t id = (sp_idx - iw - ih * IW) / (IH * IW);
        DATA_T val_to_write;
        if (is_in_padded_area)
            val_to_write = DATA_ZERO;
        else {
#if ALG_MAX
            // Read dst value only in case it's going to be used
            const off_t current_input_idx = id * IH * IW + ih * IW + iw;
            if (current_input_idx == ws_val) {
                val_to_write = diff_dst[dst_off];
            } else {
                val_to_write = DATA_ZERO;
            }
#else // ALG_MAX
            float dst_val_f = DATA_TO_REF(dst_val) / SPATIAL_DIM;
            val_to_write = CONVERT_DATA_T(dst_val_f);
#endif // ALG_MAX
        }
        const off_t src_off = SRC_OFF(mb, GWS_GET_C(), id, ih, iw);
#if IS_VECTORIZED
        DST_BLOCK_WRITE(&diff_src[src_off], val_to_write);
#else
        diff_src[src_off] = val_to_write;
#endif
    }
}
#endif // IS_BWD
