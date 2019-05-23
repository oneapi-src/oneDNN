/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#ifndef PATCH_JIT_PRIMITIVE_CONF_HPP
#define PATCH_JIT_PRIMITIVE_CONF_HPP

#include <stdint.h>

namespace mkldnn {
namespace impl {
namespace cpu {

/* resize_bilinear */
struct jit_bilinear_conf_t {
    int ndims;
    int mb, c;
    int id, ih, iw, od, oh, ow;
    int align_corners;
    bool is_training;
    bool is_backward;
    bool simple_alg;
    data_type_t ind_dt;

    int c_block, c_tail, nb_c;
    int ur_c, ur_c_tail;
    int ur_w;
    int ur_w_tail;
    size_t tail[4];
    data_type_t src_dt;
    data_type_t dst_dt;
};

struct jit_bilinear_call_s {
    const float *src;
    const float *dst;
    const void *indices;
    const float *src_prf;
    const float *dst_prf;
    const void *indices_prf;
    size_t oh;
    const float* init_value;
    float ker_area_h;
};

}
}
}

#endif
