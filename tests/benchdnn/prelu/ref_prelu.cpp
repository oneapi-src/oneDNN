/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include <algorithm>

#include <assert.h>

#include "tests/test_thread.hpp"

#include "prelu/prelu.hpp"

namespace prelu {

void compute_ref_fwd(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &wei = args.find(DNNL_ARG_WEIGHTS);
    const dnn_mem_t &dst = args.find(DNNL_ARG_DST);

    float *dst_ptr = (float *)dst;

    const auto nelems = src.nelems();
    const auto weights_broadcast_mask = prb->get_broadcast_mask();

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        const auto wei_idx = src.get_scale_idx(i, weights_broadcast_mask);
        const float s = src.get_elem(i);
        float res = s * (s > 0 ? 1.f : wei.get_elem(wei_idx));
        maybe_saturate(prb->sdt[0], res);
        dst_ptr[i] = res;
    });
}

void compute_ref_bwd(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &wei = args.find(DNNL_ARG_WEIGHTS);
    const dnn_mem_t &d_dst = args.find(DNNL_ARG_DIFF_DST);
    const dnn_mem_t &d_src = args.find(DNNL_ARG_DIFF_SRC);
    const dnn_mem_t &d_wei = args.find(DNNL_ARG_DIFF_WEIGHTS);

    float *d_src_ptr = (float *)d_src;
    float *d_wei_ptr = (float *)d_wei;
    float *d_wei_buf = d_wei_ptr;

    const auto src_nelems = d_src.nelems();
    const auto wei_nelems = d_wei.nelems();

    const auto ker = [&](int64_t i, int64_t wei_idx, int64_t d_wei_idx) {
        float s = src.get_elem(i);
        float dd = d_dst.get_elem(i);
        float d_src = dd * (s > 0 ? 1.f : wei.get_elem(wei_idx));
        maybe_saturate(prb->sdt[0], d_src);
        d_src_ptr[i] = d_src;
        d_wei_buf[d_wei_idx] += MIN2(0.f, s) * dd;
    };

    dnnl::impl::parallel_nd(wei_nelems, [&](int64_t i) { d_wei_ptr[i] = 0; });

    if (wei_nelems == 1) {
        const auto num_thr = MIN2(
                static_cast<int64_t>(dnnl_get_max_threads()), src_nelems);
        d_wei_buf = new float[num_thr];
        dnnl::impl::parallel(num_thr, [&](const int ithr, const int nthr) {
            int64_t start {0}, end {0};
            dnnl::impl::balance211(src_nelems, nthr, ithr, start, end);
            d_wei_buf[ithr] = 0;

            for (int64_t i = start; i < end; ++i)
                ker(i, 0, ithr);
        });

        for (int64_t i = 0; i < num_thr; i++)
            d_wei_ptr[0] += d_wei_buf[i];
        delete[] d_wei_buf;

    } else if (src_nelems == wei_nelems) {
        dnnl::impl::parallel_nd(src_nelems, [&](int64_t i) { ker(i, i, i); });
    } else {
        const auto weights_broadcast_mask = prb->get_broadcast_mask();

        dnnl::impl::parallel(0, [&](const int ithr, const int nthr) {
            int64_t start {0}, end {0};
            dnnl::impl::balance211(wei_nelems, nthr, ithr, start, end);
            if (start == end) return;

            for (int64_t i = 0; i < src_nelems; ++i) {
                const auto wei_idx
                        = d_src.get_scale_idx(i, weights_broadcast_mask);
                if (wei_idx < start || wei_idx >= end) continue;
                ker(i, wei_idx, wei_idx);
            }
        });
    }
}

void compute_ref(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    if (prb->dir & FLAG_FWD)
        compute_ref_fwd(prb, args);
    else
        compute_ref_bwd(prb, args);
}

} // namespace prelu
