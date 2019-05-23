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

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "utils.hpp"
#include "cpu_resize_bilinear_pd.hpp"

#include "jit_uni_bilinear_kernel_f32.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_bilinear_call_s, field)

template <cpu_isa_t isa>
status_t jit_uni_bilinear_kernel_f32<isa>::init_conf(jit_bilinear_conf_t &jpp,
            const resize_bilinear_pd_t *ppd) {
    const memory_desc_wrapper src_d(
            ppd->is_fwd() ? ppd->src_md() : ppd->diff_src_md());
    const memory_desc_wrapper dst_d(
            ppd->is_fwd() ? ppd->dst_md() : ppd->diff_dst_md());

    const int simd_w = isa == avx512_common ? 16 : 8;
    jpp.c = utils::rnd_up(src_d.dims()[1], simd_w);
    jpp.c_block = simd_w;
    jpp.nb_c = jpp.c / jpp.c_block;
    return status::success;
}


template struct jit_uni_bilinear_kernel_f32<sse41>;
template struct jit_uni_bilinear_kernel_f32<avx2>;
template struct jit_uni_bilinear_kernel_f32<avx512_common>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
