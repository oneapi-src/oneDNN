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

#include "eltwise/eltwise.hpp"

namespace eltwise {

void compute_ref_fwd(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &dst = args.find(DNNL_ARG_DST);

    float *dst_ptr = (float *)dst;

    const auto nelems = src.nelems();
    auto v_po_masks = prb->attr.post_ops.get_po_masks();

    benchdnn_parallel_nd(nelems, [&](int64_t i) {
        float res = compute_eltwise_fwd(
                prb->alg, src.get_elem(i), prb->alpha, prb->beta);

        const auto v_po_vals = prepare_po_vals(dst, args, v_po_masks, i);

        maybe_post_ops(prb->attr, res, 0.f, v_po_vals);

        // Backward use_dst case requires data adjustment since lower data type
        // may have less exact values which will be propagated further.
        res = ((prb->dir & FLAG_BWD) && prb->use_dst())
                ? round_to_nearest_representable(prb->dt, res)
                : res;
        dst_ptr[i] = res;
    });
}

void compute_ref_bwd(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &dst = args.find(DNNL_ARG_DST);
    const dnn_mem_t &source = prb->use_dst() ? dst : src;
    const dnn_mem_t &d_dst = args.find(DNNL_ARG_DIFF_DST);
    const dnn_mem_t &d_src = args.find(DNNL_ARG_DIFF_SRC);

    float *d_src_ptr = (float *)d_src;
    const auto nelems = src.nelems();

    benchdnn_parallel_nd(nelems, [&](int64_t i) {
        d_src_ptr[i] = compute_eltwise_bwd(prb->alg, d_dst.get_elem(i),
                source.get_elem(i), prb->alpha, prb->beta);
    });
}

void compute_ref(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    compute_ref_fwd(prb, args);
    if (prb->dir & FLAG_BWD) compute_ref_bwd(prb, args);
}

} // namespace eltwise
