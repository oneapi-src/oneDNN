/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#include "utils/parallel.hpp"

#include "binary/binary.hpp"

namespace binary {

void compute_ref(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    const dnn_mem_t &src0 = args.find(DNNL_ARG_SRC_0);
    const dnn_mem_t &src1 = args.find(DNNL_ARG_SRC_1);
    const dnn_mem_t &dst = args.find(DNNL_ARG_DST);

    float *dst_ptr = (float *)dst;
    const float *A = (const float *)src0;
    const float *B = (const float *)src1;

    float scales[2] = {prb->attr.scales.get(DNNL_ARG_SRC_0).scale,
            prb->attr.scales.get(DNNL_ARG_SRC_1).scale};

    const auto nelems = dst.nelems();
    const auto broadcast_mask_A = prb->get_broadcast_mask(0);
    const auto broadcast_mask_B = prb->get_broadcast_mask(1);
    auto v_po_masks = prb->attr.post_ops.get_po_masks();

    benchdnn_parallel_nd(nelems, [&](int64_t i) {
        const auto idx_A = dst.get_scale_idx(i, broadcast_mask_A);
        const auto idx_B = dst.get_scale_idx(i, broadcast_mask_B);
        float res = compute_binary(
                prb->alg, scales[0] * A[idx_A], scales[1] * B[idx_B]);
        float &dst_fp = dst_ptr[i];

        const auto v_po_vals = prepare_po_vals(dst, args, v_po_masks, i);

        maybe_post_ops(
                prb->attr, res, maybe_saturate(prb->ddt, dst_fp), v_po_vals);
        maybe_saturate(prb->ddt, res);
        dst_fp = res;
    });
}

} // namespace binary
