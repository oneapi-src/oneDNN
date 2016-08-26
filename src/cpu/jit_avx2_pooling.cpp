/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "jit_avx2_pooling.hpp"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;

template <impl::precision_t prec>
jit_avx2_pooling<prec>::jit_avx2_pooling(const pooling_primitive_desc_t &ppd,
            const primitive_at_t *inputs, const primitive *outputs[])
            : pooling<jit_avx2_pooling<prec>>(ppd, inputs, outputs)
{
    const memory_desc_wrapper
            src_d(ppd.src_primitive_desc.memory_desc),
            dst_d(ppd.dst_primitive_desc.memory_desc);

    const uint32_t simd_w = 8;
    jpp.mb = src_d.dims()[0];
    jpp.c = dst_d.dims()[1];

    jpp.ih = src_d.dims()[2];
    jpp.iw = src_d.dims()[3];

    jpp.oh = dst_d.dims()[2];
    jpp.ow = dst_d.dims()[3];

    jpp.t_pad = this->_ppd.pooling_desc.padding[0];
    jpp.l_pad = this->_ppd.pooling_desc.padding[1];

    jpp.kh = this->_ppd.pooling_desc.kernel[0];
    jpp.kw = this->_ppd.pooling_desc.kernel[1];

    jpp.stride_h = this->_ppd.pooling_desc.strides[0];
    jpp.stride_w = this->_ppd.pooling_desc.strides[1];

    jpp.c_block = simd_w;
    jpp.nb_c = jpp.c / jpp.c_block;

    jpp.ur_h = 1; /* no code-unrolling by h so far */
    jpp.ur_w = (this->_is_training) ? 3 : 8;
    if (jpp.ow < jpp.ur_w) jpp.ur_w = jpp.ow;
    jpp.ur_w_tail = jpp.ow % jpp.ur_w;

    generator = new jit_avx2_pooling_generator_f32(&jpp, this->_is_training);
//TODO: if(generator == nullptr) return nullptr;
    jit_ker = (void (*)(void*))generator->getCode();
//TODO: if(jit_ker == nullptr) return nullptr;
}

template <impl::precision_t prec>
status_t jit_avx2_pooling<prec>::execute_forward() {
    const data_t *src = reinterpret_cast<const data_t *>
        (this->input()[0].primitive->output()[
            this->input()[0].output_index]->memory_const());
    uint32_t *indices = this->_is_training
        ? reinterpret_cast<uint32_t*>(this->input()[1].primitive->output()[
                this->input()[1].output_index]->memory())
        : nullptr;
    data_t *dst = reinterpret_cast<data_t*>(this->output()[0]->memory());

    const memory_desc_wrapper
        src_d(this->_ppd.src_primitive_desc.memory_desc),
        indices_d(this->_ppd.indices_primitive_desc.memory_desc),
        dst_d(this->_ppd.dst_primitive_desc.memory_desc);

    uint32_t arr_init[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    auto ker = [&](uint32_t n, uint32_t c, uint32_t oh) {
        jit_pooling_kernel_t  par_pool = {};

        const int ij = oh * jpp.stride_h;
        const int i_t_overflow = nstl::max(0, jpp.t_pad-ij);
        const int i_b_overflow = nstl::max(jpp.ih,
                                        ij+jpp.kh-jpp.t_pad)-jpp.ih;
        const uint32_t ih = nstl::max(ij - jpp.t_pad, 0);
        par_pool.src = const_cast<data_t *>
                                 (&src[src_d.blk_off(n, c, ih, 0)]);
        par_pool.dst = &dst[dst_d.blk_off(n, c, oh, 0)];
        par_pool.indices = &indices[indices_d.blk_off(n, c, oh, 0)];
        par_pool.kh_padding = jpp.kh - i_t_overflow - i_b_overflow;
        par_pool.kw_padding = 0;
        par_pool.init_array = arr_init;

        this->jit_ker((void*)&par_pool);
    };

#   pragma omp parallel for collapse(3) schedule(static)
    for (uint32_t n = 0; n < jpp.mb; ++n) {
        for (uint32_t c = 0; c < jpp.nb_c; ++c) {
            for (uint32_t oh = 0; oh < jpp.oh; ++oh) {
                ker (n, c, oh);
            }
        }
    }
    return success;
}

template <impl::precision_t prec>
status_t jit_avx2_pooling<prec>::set_default_parameters(
        pooling_desc_t &pool_d) {
    if (pool_d.src_desc.format == any)
        CHECK(types::set_default_format<prec>(pool_d.src_desc, nChw8c));
    return pooling<jit_avx2_pooling<prec>>::template
        set_default_parameters<void>(pool_d);
}

template <impl::precision_t prec>
status_t jit_avx2_pooling<prec>::constraint(const pooling_desc_t &pool_d) {
    bool args_ok = true
        && one_of(pool_d.prop_kind, prop_kind::forward_training,
                prop_kind::forward_scoring)
        && pool_d.alg_kind == alg_kind::pooling_max
        && pool_d.src_desc.format == nChw8c
        && pool_d.dst_desc.format == pool_d.src_desc.format
        && pool_d.kernel[0] == pool_d.kernel[1];
    return args_ok ? success : unimplemented;
}

template <impl::precision_t prec>
jit_avx2_pooling<prec>::~jit_avx2_pooling()
{
    delete generator;
}

template <impl::precision_t prec>
const primitive_impl jit_avx2_pooling<prec>::implementation = {
    jit_avx2_pooling<prec>::create
};

template class jit_avx2_pooling<precision::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
