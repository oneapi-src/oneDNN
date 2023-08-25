/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "reorder/reorder.hpp"

namespace reorder {

void compute_ref(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    const dnn_mem_t &src = args.find(DNNL_ARG_FROM);
    const dnn_mem_t &dst = args.find(DNNL_ARG_TO);
    const dnn_mem_t &s8_comp = args.find(DNNL_ARG_SRC_1);
    const dnn_mem_t &zp_comp = args.find(DNNL_ARG_SRC_2);
    const dnn_mem_t &src_scales
            = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const dnn_mem_t &dst_scales
            = args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    const dnn_mem_t &src_zps
            = args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    const dnn_mem_t &dst_zps
            = args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    const bool has_src_scale = !prb->attr.scales.get(DNNL_ARG_SRC).is_def();
    const bool has_dst_scale = !prb->attr.scales.get(DNNL_ARG_DST).is_def();
    const int src_scale_mask = attr_t::get_default_mask(
            prb->attr.scales.get(DNNL_ARG_SRC).policy);
    const int dst_scale_mask = attr_t::get_default_mask(
            prb->attr.scales.get(DNNL_ARG_DST).policy);

    const auto dst_dt = prb->ddt;
    const auto nelems = src.nelems();
    // This is native to reorder zero point which comes from reorder attributes.
    const bool has_src_zp = !prb->attr.zero_points.get(DNNL_ARG_SRC).is_def();
    const bool has_dst_zp = !prb->attr.zero_points.get(DNNL_ARG_DST).is_def();
    assert(IMPLICATION(has_src_zp, src_zps.nelems() == 1));
    assert(IMPLICATION(has_dst_zp, dst_zps.nelems() == 1));
    const int src_zero_point = has_src_zp ? src_zps.get_elem(0) : 0;
    const int dst_zero_point = has_dst_zp ? dst_zps.get_elem(0) : 0;

    float beta = 0;
    const auto &po = prb->attr.post_ops;
    const int beta_idx = po.find(attr_t::post_ops_t::kind_t::SUM);
    if (beta_idx >= 0) beta = po.entry[beta_idx].sum.scale;

    // These are non-native compensations coming from other primitives with
    // s8s8 or zero-points support to pre-compute compensated part and apply it
    // at the end of computations.
    const bool need_s8_comp = s8_comp.dt() == dnnl_s32;
    const bool need_zp_comp = zp_comp.dt() == dnnl_s32;
    const bool need_comp = need_s8_comp || need_zp_comp;
    // `adjust_scale` participates only with s8s8 compensation.
    const float s8_scale_factor = need_s8_comp ? reorder_rescale_factor() : 1.f;

    benchdnn_parallel_nd(nelems, [&](int64_t idx) {
        float s = src.get_elem(idx) - src_zero_point;
        float d = 0;
        if (beta_idx >= 0) d = dst.get_elem(idx) - dst_zero_point;

        float src_scale = 1.f, dst_scale = 1.f;
        if (has_src_scale) {
            int64_t src_mask_idx = src.get_scale_idx(idx, src_scale_mask);
            src_scale = src_scales.get_elem(src_mask_idx);
        }
        if (has_dst_scale) {
            int64_t dst_mask_idx = dst.get_scale_idx(idx, dst_scale_mask);
            dst_scale = dst_scales.get_elem(dst_mask_idx);
        }
        float value = (s8_scale_factor * src_scale * s + beta * d) / dst_scale
                + dst_zero_point;
        value = maybe_saturate(dst_dt, value);
        if (dst_dt == dnnl_s32 && value >= (float)INT_MAX)
            value = BENCHDNN_S32_TO_F32_SAT_CONST;

        dst.set_elem(idx, round_to_nearest_representable(dst_dt, value));
    });

    if (!need_comp) return;

    // mostly following benchdnn/ref_reduction.cpp/compute_ref
    const auto nelems_s8_comp = s8_comp.nelems();
    const auto nelems_zp_comp = zp_comp.nelems();
    const auto nelems_comp = MAX2(nelems_s8_comp, nelems_zp_comp);
    if (nelems_comp == 0) SAFE_V(FAIL);

    const auto &ndims = src.ndims();
    assert(nelems_comp > 0);
    assert(IMPLICATION(
            need_s8_comp && need_zp_comp, nelems_s8_comp == nelems_zp_comp));

    int comp_mask = 0;
    for (const auto &i_oflag : prb->oflag) {
        if ((i_oflag.first == FLAG_S8S8_COMP || i_oflag.first == FLAG_ZP_COMP)
                && i_oflag.second != FLAG_NONE) {
            comp_mask = i_oflag.second;
            break;
        }
    }

    dims_t comp_dims(ndims, 1); // src_dims with '1' at non-masked dims.
    dims_t reduce_dims(ndims, 1); // complementary to above.
    for (int i = 0; i < ndims; ++i) {
        if (comp_mask & (1 << i)) {
            comp_dims[i] = src.dims()[i];
            reduce_dims[i] = 1;
        } else {
            comp_dims[i] = 1;
            reduce_dims[i] = src.dims()[i];
        }
    }

    const auto nelems_reduce = nelems / nelems_comp;
    benchdnn_parallel_nd(nelems_comp, [&](int64_t f) {
        dims_t idle_pos = off2dims_idx(comp_dims, f);
        const int64_t src_idle_off = md_off_v(src, idle_pos.data());
        int comp_val = 0;
        for (int64_t r = 0; r < nelems_reduce; ++r) {
            dims_t reduce_pos = off2dims_idx(reduce_dims, r);
            const int64_t src_reduce_off = md_off_v(src, reduce_pos.data());
            const int64_t src_off = src_idle_off + src_reduce_off;

            float src_scale = 1.f, dst_scale = 1.f;
            if (has_src_scale) {
                int64_t src_mask_idx
                        = src.get_scale_idx(src_off, src_scale_mask);
                src_scale = src_scales.get_elem(src_mask_idx);
            }
            if (has_dst_scale) {
                int64_t dst_mask_idx
                        = dst.get_scale_idx(src_off, dst_scale_mask);
                dst_scale = dst_scales.get_elem(dst_mask_idx);
            }

            const float alpha = src_scale / dst_scale;
            const float value = src.get_elem(src_off) * alpha * s8_scale_factor;
            comp_val -= maybe_saturate(dst_dt, value);
        }
        if (need_zp_comp) zp_comp.set_elem(f, comp_val);
        comp_val *= 128;
        if (need_s8_comp) s8_comp.set_elem(f, comp_val);
    });
}

} // namespace reorder
