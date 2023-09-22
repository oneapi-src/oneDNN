/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "cpu/ref_io_helper.hpp"
#include "cpu/simple_q10n.hpp"

#include "cpu/ref_inner_product_int8.hpp"
#include "cpu/ref_inner_product_utils.hpp"

#include "cpu/cpu_primitive.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

status_t ref_inner_product_int8_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const auto MB = pd()->MB();
    const auto OC = pd()->OC();
    const auto IC = pd()->IC();
    const auto KD = pd()->KD();
    const auto KH = pd()->KH();
    const auto KW = pd()->KW();

    const auto ndims = pd()->ndims();

    auto ker = [&](dim_t mb, dim_t oc) {
        int d = 0;
        for_(dim_t ic = 0; ic < IC; ++ic)
        for_(dim_t kd = 0; kd < KD; ++kd)
        for_(dim_t kh = 0; kh < KH; ++kh)
        for (dim_t kw = 0; kw < KW; ++kw) {
            const auto src_off = ref_ip_utils::get_data_off(
                    src_d, ndims, mb, ic, kd, kh, kw);
            const auto wei_off = ref_ip_utils::get_weights_off(
                    weights_d, ndims, oc, ic, kd, kh, kw);
            const int s = io::load_int_value(src_d.data_type(), src, src_off);
            const int w = io::load_int_value(
                    weights_d.data_type(), weights, wei_off);
            d += s * w;
        }
        return d;
    };

    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(wei_scales, DNNL_ARG_WEIGHTS);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    const auto &attr_scales = pd()->attr()->scales_;
    const bool with_dst_scales
            = !attr_scales.get(DNNL_ARG_DST).has_default_values();

    auto maybe_oscale = [&](float &d, dim_t oc) {
        // scale_idx_mult = 1 for per_oc scales and 0, otherwise
        const int scale_idx_mult
                = attr_scales.get(DNNL_ARG_WEIGHTS).mask_ == (1 << 0);
        d *= src_scales[0] * wei_scales[oc * scale_idx_mult];
    };

    auto sum_dt = pd()->attr()->post_ops_.get_sum_dt(dst_d.data_type());

    parallel_nd(MB, OC, [&](dim_t mb, dim_t oc) {
        int acc = ker(mb, oc);

        float d = static_cast<float>(acc);
        maybe_oscale(d, oc);

        if (bias) {
            const auto bias_off = bias_d.off(oc);
            const float b
                    = io::load_float_value(bias_d.data_type(), bias, bias_off);
            d += b;
        }

        dim_t dst_off = dst_d.off(mb, oc);
        dim_t dst_l_off = (mb * OC + oc);

        ref_post_ops_t::args_t args;
        args.dst_val = io::load_float_value(sum_dt, dst, dst_off);
        args.ctx = &ctx;
        args.l_offset = dst_l_off;
        args.dst_md = pd()->dst_md();
        ref_post_ops->execute(d, args);

        if (with_dst_scales) d *= dst_scales[0];
        io::store_float_value(dst_d.data_type(), d, dst, dst_off);
    });

    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
