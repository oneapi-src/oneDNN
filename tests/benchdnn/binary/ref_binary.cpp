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

#include "tests/test_thread.hpp"

#include "binary/binary.hpp"

namespace binary {

float compute_binary(alg_t alg, float src0, float src1) {
    if (alg == ADD) {
        return src0 + src1;
    } else if (alg == MUL) {
        return src0 * src1;
    } else if (alg == MAX) {
        return MAX2(src0, src1);
    } else if (alg == MIN) {
        return MIN2(src0, src1);
    } else {
        assert(!"operation not supported!");
    }
    return 0;
};

void compute_ref(const prb_t *p, const dnn_mem_t &src0, const dnn_mem_t &src1,
        dnn_mem_t &dst) {
    float *dst_ptr = (float *)dst;
    const float *A = (const float *)src0;
    const float *B = (const float *)src1;

    float scales[2] = {p->attr.scales.get(DNNL_ARG_SRC_0).scale,
            p->attr.scales.get(DNNL_ARG_SRC_1).scale};

    const auto nelems_A = src0.nelems();
    const auto broadcast_mask = p->get_broadcast_mask();

    dnnl::impl::parallel_nd(nelems_A, [&](int64_t i) {
        auto idx_B = src0.get_scale_idx(i, broadcast_mask);
        float res = compute_binary(
                p->alg, scales[0] * A[i], scales[1] * B[idx_B]);
        float &dst = dst_ptr[i];
        maybe_post_ops(p->attr, res, dst);
        dst = res;
    });
}

} // namespace binary
