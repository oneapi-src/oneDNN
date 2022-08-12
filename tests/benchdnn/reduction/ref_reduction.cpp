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

#include <limits>
#include <math.h>

#include "utils/parallel.hpp"

#include "common.hpp"
#include "dnnl_memory.hpp"

#include "reduction.hpp"

namespace reduction {

void init_acc(float &acc, alg_t alg) {
    switch (alg) {
        case max: acc = std::numeric_limits<float>::lowest(); break;
        case min: acc = std::numeric_limits<float>::max(); break;
        case sum: acc = 0.0f; break;
        case mul: acc = 1.0f; break;
        case mean:
        case norm_lp_max:
        case norm_lp_sum:
        case norm_lp_power_p_max:
        case norm_lp_power_p_sum: acc = 0.0f; break;
        default: assert(!"unknown algorithm");
    }
}

void accumulate(float &dst, const float src, alg_t alg, float p, float eps) {
    switch (alg) {
        case max: dst = MAX2(dst, src); break;
        case min: dst = MIN2(dst, src); break;
        case mean:
        case sum: dst += src; break;
        case mul: dst *= src; break;
        case norm_lp_max:
        case norm_lp_sum:
        case norm_lp_power_p_max:
        case norm_lp_power_p_sum: dst += pow(fabs(src), p); break;
        default: assert(!"unknown algorithm");
    }
}

void finalize(float &dst, alg_t alg, float p, float eps, dnnl_dim_t n) {
    switch (alg) {
        case mean: dst /= n; break;
        case norm_lp_max:
            dst = MAX2(dst, eps);
            dst = pow(dst, 1.0f / p);
            break;
        case norm_lp_sum:
            dst += eps;
            dst = pow(dst, 1.0f / p);
            break;
        case norm_lp_power_p_max: dst = MAX2(dst, eps); break;
        case norm_lp_power_p_sum: dst += eps; break;
        default: break;
    }
}

void compute_ref(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    const dnn_mem_t &src = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &dst = args.find(DNNL_ARG_DST);

    float *dst_ptr = (float *)dst;

    const auto &ndims = prb->ndims;
    const auto &src_dims = prb->vdims[0];
    const auto &dst_dims = prb->vdims[1];

    const auto alg = prb->alg;
    const auto p = prb->p;
    const auto eps = prb->eps;

    dims_t reduce_dims(ndims, 1);
    int64_t reduce_size {1}, idle_size {1};

    for (int d = 0; d < ndims; ++d) {
        const bool is_reduction_dim = src_dims[d] != dst_dims[d];
        if (is_reduction_dim) {
            reduce_dims[d] = src_dims[d];
            reduce_size *= reduce_dims[d];
        } else {
            idle_size *= dst_dims[d];
        }
    }

    if (reduce_size == 1) return;

    auto v_po_masks = prb->attr.post_ops.get_po_masks();
    benchdnn_parallel_nd(idle_size, [&](int64_t f) {
        dims_t idle_pos = off2dims_idx(dst_dims, f);
        const int64_t dst_off = md_off_v(dst, idle_pos.data());
        const int64_t src_idle_off = md_off_v(src, idle_pos.data());
        float acc {0.0f};
        init_acc(acc, alg);
        for (int64_t r = 0; r < reduce_size; ++r) {
            dims_t reduce_pos = off2dims_idx(reduce_dims, r);
            const int64_t src_reduce_off = md_off_v(src, reduce_pos.data());
            const int64_t src_off = src_idle_off + src_reduce_off;
            accumulate(acc, src.get_elem(src_off), alg, p, eps);
        }
        finalize(acc, alg, p, eps, reduce_size);

        const auto v_po_vals = prepare_po_vals(dst, args, v_po_masks, dst_off);

        maybe_post_ops(prb->attr, acc, dst_ptr[dst_off], v_po_vals);
        dst_ptr[dst_off] = acc;
    });
}

} // namespace reduction
