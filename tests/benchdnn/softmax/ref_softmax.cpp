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

#include "softmax/softmax.hpp"

namespace softmax {

void get_sizes(const prb_t *p, int64_t &outer_size, int64_t &inner_size,
        int64_t &axis_size) {
    outer_size = inner_size = axis_size = 1;
    for (int i = 0; i < p->axis; i++)
        outer_size *= p->dims[i];
    for (int i = p->axis + 1; i < (int)p->dims.size(); i++)
        inner_size *= p->dims[i];
    axis_size = p->dims[p->axis];
}

void compute_ref_fwd(const prb_t *p, const dnn_mem_t &src, dnn_mem_t &dst) {
    int64_t outer_size{0}, inner_size{0}, axis_size{0};
    get_sizes(p, outer_size, inner_size, axis_size);

    const float *src_ptr = (const float *)src;
    float *dst_ptr = (float *)dst;

    mkldnn::impl::parallel_nd(outer_size, inner_size,
            [&](int64_t ou, int64_t in) {
        float space_denom = 0.;
        float space_max = -FLT_MAX;

        for (int64_t as = 0; as < axis_size; ++as) {
            int64_t idx = ou * axis_size * inner_size + as * inner_size + in;
            space_max = MAX2(space_max, src_ptr[idx]);
        }

        for (int64_t as = 0; as < axis_size; ++as) {
            int64_t idx = ou * axis_size * inner_size + as * inner_size + in;
            float D = dst_ptr[idx] = expf(src_ptr[idx] - space_max);
            space_denom += D;
        }

        for (int64_t as = 0; as < axis_size; ++as) {
            int64_t idx = ou * axis_size * inner_size + as * inner_size + in;
            dst_ptr[idx] /= space_denom;
        }
    });
}

void compute_ref_bwd(const prb_t *p, const dnn_mem_t &dst,
        const dnn_mem_t &diff_dst, dnn_mem_t &diff_src) {
    int64_t outer_size{0}, inner_size{0}, axis_size{0};
    get_sizes(p, outer_size, inner_size, axis_size);

    const float *dst_ptr = (const float *)dst;
    const float *d_dst_ptr = (const float *)diff_dst;
    float *d_src_ptr = (float *)diff_src;

    mkldnn::impl::parallel_nd(outer_size, inner_size,
            [&](int64_t ou, int64_t in) {
        float part_deriv_sum = 0.;

        for (int64_t as = 0; as < axis_size; ++as) {
            int64_t idx = ou * axis_size * inner_size + as * inner_size + in;
            part_deriv_sum += dst_ptr[idx] * d_dst_ptr[idx];
        }

        for (int64_t as = 0; as < axis_size; ++as) {
            int64_t idx = ou * axis_size * inner_size + as * inner_size + in;
            d_src_ptr[idx] = dst_ptr[idx] * (d_dst_ptr[idx] - part_deriv_sum);
        }
    });
}

}
