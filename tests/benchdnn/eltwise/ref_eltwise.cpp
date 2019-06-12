/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "src/common/mkldnn_thread.hpp"

#include "eltwise/eltwise.hpp"

namespace eltwise {

void compute_ref_fwd(const prb_t *p, const dnn_mem_t &src, dnn_mem_t &dst) {
    const float *src_ptr = (const float *)src;
    float *dst_ptr = (float *)dst;
    const auto nelems = src.nelems();

    mkldnn::impl::parallel_nd(nelems, [&](int64_t i) {
            dst_ptr[i] = compute_eltwise_fwd(p->alg, src_ptr[i], 1.0, p->alpha,
                p->beta);
    });
}

void compute_ref_bwd(const prb_t *p, const dnn_mem_t &src,
        const dnn_mem_t &diff_dst, dnn_mem_t &diff_src) {
    const float *src_ptr = (const float *)src;
    const float *d_dst_ptr = (const float *)diff_dst;
    float *d_src_ptr = (float *)diff_src;
    const auto nelems = src.nelems();

    mkldnn::impl::parallel_nd(nelems, [&](int64_t i) {
        d_src_ptr[i] = compute_eltwise_bwd(
                p->alg, d_dst_ptr[i], src_ptr[i], p->alpha, p->beta);
    });
}

}
