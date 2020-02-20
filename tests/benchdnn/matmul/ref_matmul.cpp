/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "src/common/dnnl_thread.hpp"

#include "matmul/matmul.hpp"

namespace matmul {

void compute_ref(const prb_t *p, dnn_mem_t &src_m, dnn_mem_t &wei_m,
        dnn_mem_t &bia_m, dnn_mem_t &dst_m) {
    const int64_t MB = p->mb;
    const int64_t M = p->m;
    const int64_t N = p->n;
    const int64_t K = p->k;

    const int src_zero_point = p->attr.zero_points[DNNL_ARG_SRC];
    const int wei_zero_point = p->attr.zero_points[DNNL_ARG_WEIGHTS];
    const int dst_zero_point = p->attr.zero_points[DNNL_ARG_DST];

    dnn_mem_t dst_tmp(dst_m.md_, dnnl_f32, dnnl_format_tag_undef, engine_tgt);

    dnnl::impl::parallel_nd(MB, M, N, [&](int64_t mb, int64_t m, int64_t n) {
        auto src = (const float *)src_m;
        auto wei = (const float *)wei_m;

        float dst = 0;
        for (int64_t k = 0; k < K; ++k)
            dst += (src[src_off_f(p, mb, m, k)] - src_zero_point)
                    * (wei[wei_off_f(p, mb, k, n)] - wei_zero_point);

        ((float *)dst_tmp)[dst_off_f(p, mb, m, n)] = dst;
    });

    dnnl::impl::parallel_nd(MB, M, N, [&](int64_t mb, int64_t m, int64_t n) {
        size_t dst_off = dst_off_f(p, mb, m, n);
        float &dst = ((float *)dst_m)[dst_off];

        float tmp = ((float *)dst_tmp)[dst_off];
        if (p->bia_dt != dnnl_data_type_undef) {
            int64_t wei_off = bia_off_f(p, mb, m, n);
            float *bia_ptr = (float *)bia_m;
            tmp += bia_ptr[wei_off];
        }
        maybe_scale(tmp, p->scales, n, p->attr);
        maybe_post_ops(tmp, dst, p->attr);
        tmp += dst_zero_point;
        dst = tmp;
    });
}

} // namespace matmul
