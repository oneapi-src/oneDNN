/******************************************************************************
* Copyright 2023 Intel Corporation
* Copyright 2023 KNS Group LLC (YADRO)
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

#include "rvv_nchw_pooling.hpp"
#include <algorithm>
#include <riscv_vector.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace {
void MaxPooling(const float *src, float *dst, const dim_t batch,
        const dim_t channels, const dim_t outD, const dim_t outH,
        const dim_t outW, const dim_t inD, const dim_t inH, const dim_t inW,
        const dim_t kerD, const dim_t kerH, const dim_t kerW,
        const dim_t strideD, const dim_t strideH, const dim_t strideW,
        const dim_t padFront, const dim_t padTop, const dim_t padLeft) {
    float arr_flt_min
            [riscv_nchw_pooling_fwd_t<data_type::f32>::max_kernel_width];
    for (int i = 0;
            i < riscv_nchw_pooling_fwd_t<data_type::f32>::max_kernel_width; i++)
        arr_flt_min[i] = -__FLT_MAX__;

    for (int mb = 0; mb < batch; mb++)
        for (int c = 0; c < channels; c++)
            for (int od = 0; od < outD; od++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++) {
                        const size_t dst_offset
                                = (size_t)outW * outH * outD * channels * mb
                                + (size_t)outW * outH * outD * c
                                + (size_t)outW * outH * od + (size_t)outW * oh
                                + (size_t)ow;
                        const auto src_offset = ((size_t)inW * inH * inD)
                                * ((size_t)channels * mb + c);
                        const auto local_src = &src[src_offset];
                        const auto IWH = (size_t)inW * inH;

                        int od_offset = od * strideD - padFront;
                        int oh_offset = oh * strideH - padTop;
                        int ow_offset = ow * strideW - padLeft;
                        size_t size = std::min(ow_offset + kerW, inW)
                                - std::max(ow_offset, 0);
                        size_t cycleLength = vsetvl_e32m8(size);
                        vfloat32m8_t vmax
                                = vle32_v_f32m8(&arr_flt_min[0], cycleLength);

                        for (int id = std::max(od_offset, 0);
                                id < std::min(od_offset + kerD, inD); id++)
                            for (int ih = std::max(oh_offset, 0);
                                    ih < std::min(oh_offset + kerH, inH);
                                    ih++) {
                                const auto local_src_offset = IWH * id
                                        + (size_t)inW * ih
                                        + std::max(ow_offset, 0);

                                size_t iw = 0;
                                for (; iw < size - cycleLength;
                                        iw += cycleLength) {
                                    vfloat32m8_t vsrc = vle32_v_f32m8(
                                            &local_src[local_src_offset + iw],
                                            cycleLength);
                                    vmax = vfmax_vv_f32m8(
                                            vsrc, vmax, cycleLength);
                                }

                                size_t tailLength = vsetvl_e32m8(size - iw);
                                {
                                    vfloat32m8_t vsrc = vle32_v_f32m8(
                                            &local_src[local_src_offset + iw],
                                            tailLength);
                                    vmax = vfmax_vv_f32m8(
                                            vsrc, vmax, tailLength);
                                }
                            }

                        vfloat32m1_t min_scalar;
                        float min = -__FLT_MAX__;
                        min_scalar = vle32_v_f32m1(&min, 1);

                        cycleLength = vsetvl_e32m8(size);
                        vfloat32m1_t vred_res;
                        vred_res = vfredmax_vs_f32m8_f32m1(
                                vred_res, vmax, min_scalar, cycleLength);

                        float red_res;
                        vse32_v_f32m1(&red_res, vred_res, 1);
                        dst[dst_offset] = red_res;
                    }
}
} // namespace

template <data_type_t d_type>
riscv_nchw_pooling_fwd_t<d_type>::riscv_nchw_pooling_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <>
status_t riscv_nchw_pooling_fwd_t<data_type::f32>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    src += src_d.off_l(0);
    dst += dst_d.off_l(0);

    const dim_t MB = pd()->MB();
    const dim_t C = pd()->OC();
    const dim_t OD = pd()->OD();
    const dim_t OH = pd()->OH();
    const dim_t OW = pd()->OW();
    const dim_t ID = pd()->ID();
    const dim_t IH = pd()->IH();
    const dim_t IW = pd()->IW();
    const dim_t KD = pd()->KD();
    const dim_t KH = pd()->KH();
    const dim_t KW = pd()->KW();
    const dim_t SD = pd()->KSD();
    const dim_t SH = pd()->KSH();
    const dim_t SW = pd()->KSW();
    const dim_t padF = pd()->padFront();
    const dim_t padT = pd()->padT();
    const dim_t padL = pd()->padL();

    const auto alg = pd()->desc()->alg_kind;
    const bool is_max_pool = alg == alg_kind::pooling_max;

    if (!is_max_pool) { return status::unimplemented; }

    MaxPooling(src, dst, MB, C, OD, OH, OW, ID, IH, IW, KD, KH, KW, SD, SH, SW,
            padF, padT, padL);

    return status::success;
}

template struct riscv_nchw_pooling_fwd_t<data_type::f32>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
