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

#include "binary/binary.hpp"

namespace binary {

void perform_op(const prb_t *p, float *dst, float x, float y) {
    float res = 0;
    if (p->alg == ADD) {
        res = x + y;
    } else if (p->alg == MUL) {
        res = x * y;
    } else if (p->alg == MAX) {
        res = MAX2(x, y);
    } else if (p->alg == MIN) {
        res = MIN2(x, y);
    } else {
        assert(!"operation not supported!");
    }
    maybe_post_ops(res, *dst, p->attr);
    *dst = res;
};

int64_t map_idx_B(const prb_t *p, int64_t idx) {
    dims_t dims = off2dims_idx(p->sdims[0], idx);
    for (int d = 0; d < p->ndims[0]; ++d)
        dims[d] *= (!p->broadcast_dims[d]);
    return dims_off(p->sdims[1], dims);
}

void compute_ref(const prb_t *p, const dnn_mem_t &src0, const dnn_mem_t &src1,
        dnn_mem_t &dst) {
    float *dst_ptr = (float *)dst;
    const float *A = (const float *)src0;
    const float *B = (const float *)src1;

    float scales[2] = {p->attr.scales.get(DNNL_ARG_SRC_0).scale,
            p->attr.scales.get(DNNL_ARG_SRC_1).scale};

    const auto nelems_A = src0.nelems();
    const auto nelems_B = src1.nelems();

    dnnl::impl::parallel_nd(nelems_A, [&](int64_t i) {
        int64_t idx_B = nelems_B == nelems_A ? i : map_idx_B(p, i);
        perform_op(p, &dst_ptr[i], scales[0] * A[i], scales[1] * B[idx_B]);
    });
}

} // namespace binary
