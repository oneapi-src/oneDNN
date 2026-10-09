/*******************************************************************************
* Copyright 2022 IBM Corporation
* Copyright 2026 Intel Corporation
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

#include "cpu/ppc64/ppc64_gemm_reorder.hpp"

#include <altivec.h>
#include <cstdint>
#include <iostream>
#include <unistd.h> // For thread sleep

namespace dnnl {
namespace impl {
namespace cpu {
namespace ppc64 {

using namespace dnnl::impl::cpu::q10n;

namespace {
inline status_t get_scales_mask(
        const primitive_attr_t *attr, int *src_mask, int *dst_mask) {
    const auto &s = attr->scales_;

    if (src_mask == nullptr || dst_mask == nullptr)
        return status::invalid_arguments;

    *src_mask = 0;
    if (!s.has_default_values(DNNL_ARG_SRC))
        *src_mask = s.get_mask(DNNL_ARG_SRC);

    *dst_mask = 0;
    if (!s.has_default_values(DNNL_ARG_DST))
        *dst_mask = s.get_mask(DNNL_ARG_DST);

    // This is used in a check function.
    if (*src_mask > 0 && *dst_mask > 0 && *dst_mask != *src_mask)
        return status::invalid_arguments;
    return status::success;
}
} // namespace

typedef __vector signed long long vec_i64 __attribute__((aligned(8)));
typedef __vector short vec_i16 __attribute__((aligned(2)));
typedef __vector unsigned char vec_ut;
typedef __vector signed char vec_t;
typedef __vector signed short vec_short_t;
typedef __vector signed int vec_int_t;
typedef __vector float vec_float_t;

status_t ppc64_matrixA_reorder_t::pd_t::init(const engine_t *engine,
        const engine_t *src_engine, const engine_t *dst_engine) {
    using namespace status;

    using namespace format_tag;

    status_t status = cpu_reorder_pd_t::init(engine, src_engine, dst_engine);
    if (status != success) return status;

    const memory_desc_wrapper id(src_md_), od(dst_md_);

    const int ndims = id.ndims();

    const auto type_i = id.data_type();
    const auto type_o = od.data_type();

    const auto in_strides = id.strides();
    const auto out_strides = od.strides();

    const bool is_row_major = ((in_strides[0] == out_strides[0])
                                      && (in_strides[1] == out_strides[1])
                                      && (out_strides[1] == 1))
            ? true
            : false;
    const bool dt_ok = true && utils::one_of(type_i, data_type::f32)
            && utils::one_of(type_o, data_type::u8, data_type::s8);
    const bool args_ok = dt_ok && ndims == 2 && is_row_major;

    if (!args_ok) return invalid_arguments;
    init_scratchpad();
    return status::success;
}

typedef __vector unsigned int VecUInt;

template <typename InputType, typename OutputType>
void kernel(InputType *inp, OutputType *out, int N, const float SrcScale,
        const float DstScale, const int SrcZeroPoint, const int DstZeroPoint,
        const float beta) {
    //OutputType ZeroPoint) {

    constexpr int32_t MinimumValue = std::numeric_limits<OutputType>::min();
    constexpr int32_t MaximumValue = std::numeric_limits<OutputType>::max();

    __vector float SrcScaleVector = vec_splats(SrcScale);
    __vector float DstScaleVector = vec_splats(DstScale);

    __vector float MinimumValueVector = vec_splats(float(MinimumValue));
    __vector float MaximumValueVector = vec_splats(float(MaximumValue));
    __vector float SrcZeroPointVector = vec_splats(float(SrcZeroPoint));
    __vector float DstZeroPointVector = vec_splats(float(DstZeroPoint));

    while (N >= 16) {
        auto FloatVector0 = vec_xl(0, inp);
        auto FloatVector1 = vec_xl(0, inp + 4);
        auto FloatVector2 = vec_xl(0, inp + 8);
        auto FloatVector3 = vec_xl(0, inp + 12);

        FloatVector0 = vec_sub(FloatVector0, SrcZeroPointVector);
        FloatVector0 = vec_mul(FloatVector0, SrcScaleVector);
        FloatVector1 = vec_sub(FloatVector1, SrcZeroPointVector);
        FloatVector1 = vec_mul(FloatVector1, SrcScaleVector);
        FloatVector2 = vec_sub(FloatVector2, SrcZeroPointVector);
        FloatVector2 = vec_mul(FloatVector2, SrcScaleVector);
        FloatVector3 = vec_sub(FloatVector3, SrcZeroPointVector);
        FloatVector3 = vec_mul(FloatVector3, SrcScaleVector);

        if (beta) {
            FloatVector0[0] += beta * (float)out[0];
            FloatVector0[1] += beta * (float)out[1];
            FloatVector0[2] += beta * (float)out[2];
            FloatVector0[3] += beta * (float)out[3];

            FloatVector1[0] += beta * (float)out[4];
            FloatVector1[1] += beta * (float)out[5];
            FloatVector1[2] += beta * (float)out[6];
            FloatVector1[3] += beta * (float)out[7];

            FloatVector2[0] += beta * (float)out[8];
            FloatVector2[1] += beta * (float)out[9];
            FloatVector2[2] += beta * (float)out[10];
            FloatVector2[3] += beta * (float)out[11];

            FloatVector3[0] += beta * (float)out[12];
            FloatVector3[1] += beta * (float)out[13];
            FloatVector3[2] += beta * (float)out[14];
            FloatVector3[3] += beta * (float)out[15];
        }
        FloatVector0 = vec_mul(FloatVector0, DstScaleVector);
        FloatVector1 = vec_mul(FloatVector1, DstScaleVector);
        FloatVector2 = vec_mul(FloatVector2, DstScaleVector);
        FloatVector3 = vec_mul(FloatVector3, DstScaleVector);

        FloatVector0 = vec_round(FloatVector0);
        FloatVector1 = vec_round(FloatVector1);
        FloatVector2 = vec_round(FloatVector2);
        FloatVector3 = vec_round(FloatVector3);

        FloatVector0 = vec_add(FloatVector0, DstZeroPointVector);
        FloatVector1 = vec_add(FloatVector1, DstZeroPointVector);
        FloatVector2 = vec_add(FloatVector2, DstZeroPointVector);
        FloatVector3 = vec_add(FloatVector3, DstZeroPointVector);

        FloatVector0 = vec_max(FloatVector0, MinimumValueVector);
        FloatVector1 = vec_max(FloatVector1, MinimumValueVector);
        FloatVector2 = vec_max(FloatVector2, MinimumValueVector);
        FloatVector3 = vec_max(FloatVector3, MinimumValueVector);

        FloatVector0 = vec_min(FloatVector0, MaximumValueVector);
        FloatVector1 = vec_min(FloatVector1, MaximumValueVector);
        FloatVector2 = vec_min(FloatVector2, MaximumValueVector);
        FloatVector3 = vec_min(FloatVector3, MaximumValueVector);

        VecUInt IntegerVector0 = vec_ctu(FloatVector0, 0);
        VecUInt IntegerVector1 = vec_ctu(FloatVector1, 0);
        VecUInt IntegerVector2 = vec_ctu(FloatVector2, 0);
        VecUInt IntegerVector3 = vec_ctu(FloatVector3, 0);

        auto ShortVector0 = vec_pack(IntegerVector0, IntegerVector1);
        auto ShortVector1 = vec_pack(IntegerVector2, IntegerVector3);
        auto CharVector = vec_pack(ShortVector0, ShortVector1);

        vec_xst(CharVector, 0, (uint8_t *)out);
        out += 16;
        inp += 16;
        N -= 16;
    }
#ifdef __MMA__
    while (N >= 4) {
        auto FloatVector = vec_xl(0, inp);
        FloatVector = vec_sub(FloatVector, SrcZeroPointVector);
        FloatVector = vec_mul(FloatVector, SrcScaleVector);

        if (beta) {
            FloatVector[0] += beta * (float)out[0];
            FloatVector[1] += beta * (float)out[1];
            FloatVector[2] += beta * (float)out[2];
            FloatVector[3] += beta * (float)out[3];
        }
        FloatVector = vec_mul(FloatVector, DstScaleVector);
        FloatVector = vec_round(FloatVector);
        FloatVector = vec_add(FloatVector, DstZeroPointVector);

        FloatVector = vec_max(FloatVector, MinimumValueVector);
        FloatVector = vec_min(FloatVector, MaximumValueVector);
        auto IntegerVector = vec_ctu(FloatVector, 0);

        auto ShortVector = vec_pack(IntegerVector, vec_splats((uint32_t)0));
        auto CharVector = vec_pack(ShortVector, vec_splats((uint16_t)0));

        vec_xst_len(CharVector, (uint8_t *)out, N);

        out += 4;
        inp += 4;
        N -= 4;
    }

    if (N > 0) {
        auto FloatVector = vec_xl_len(const_cast<float *>(inp), 4 * N);
        FloatVector = vec_sub(FloatVector, SrcZeroPointVector);
        FloatVector = vec_mul(FloatVector, SrcScaleVector);
        if (beta) {
            if (N == 1) { FloatVector[0] += beta * (float)out[0]; }
            if (N == 2) {
                FloatVector[0] += beta * (float)out[0];
                FloatVector[1] += beta * (float)out[1];
            }
            if (N == 3) {
                FloatVector[0] += beta * (float)out[0];
                FloatVector[1] += beta * (float)out[1];
                FloatVector[2] += beta * (float)out[2];
            }
        }
        FloatVector = vec_mul(FloatVector, DstScaleVector);
        FloatVector = vec_round(FloatVector);
        FloatVector = vec_add(FloatVector, DstZeroPointVector);

        FloatVector = vec_max(FloatVector, MinimumValueVector);
        FloatVector = vec_min(FloatVector, MaximumValueVector);
        auto IntegerVector = vec_ctu(FloatVector, 0);

        auto ShortVector = vec_pack(IntegerVector, vec_splats((uint32_t)0));
        auto CharVector = vec_pack(ShortVector, vec_splats((uint16_t)0));
        vec_xst_len(CharVector, (uint8_t *)out, N);
    }
#else
    // For Other than Power10 below code will run
    // Remaining 1-15 elements
    while (N > 0) {
        float val = (*inp - SrcZeroPoint) * SrcScale;
        if (beta) val += beta * *out;
        val = val * DstScale + DstZeroPoint;
        val = std::fmin(
                std::fmax(val, float(MinimumValue)), float(MaximumValue));
        *out = uint8_t(std::nearbyint(val));
        inp++;
        out++;
        N -= 1;
    }
#endif
}

status_t ppc64_matrixA_reorder_t::execute_body(const exec_ctx_t &ctx) const {
    using namespace utils;

    const auto input = CTX_IN_MEM(const float *, DNNL_ARG_FROM);
    auto output = CTX_OUT_MEM(unsigned char *, DNNL_ARG_TO);
    const auto &scratchpad = ctx.get_scratchpad_grantor();
    MAYBE_UNUSED(scratchpad);
    const auto input_d = ctx.memory_mdw(DNNL_ARG_FROM, pd()->src_md());

    DEFINE_ARG_SCALES_BUFFER_ATTR(pd()->attr(), src_scales, DNNL_ARG_FROM);
    DEFINE_ARG_SCALES_BUFFER_ATTR(pd()->attr(), dst_scales_, DNNL_ARG_TO);

    int src_scales_mask, dst_scales_mask;
    CHECK(get_scales_mask(pd()->attr(), &src_scales_mask, &dst_scales_mask));

    int scales_mask = std::max(src_scales_mask, dst_scales_mask);
    MAYBE_UNUSED(scales_mask);

    dim_t D_start, D_mask, D_rest;
    pd()->get_D_values(input_d, scales_mask, &D_start, &D_mask, &D_rest);

    const float *dst_scales = pd()->precompute_scales(
            scratchpad, pd()->attr(), D_mask, dst_scales_);

    const int32_t *src_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_FROM);
    int src_zp = src_zero_points ? src_zero_points[0] : 0;

    const int32_t *dst_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_TO);
    int dst_zp = dst_zero_points ? dst_zero_points[0] : 0;

    const float alpha = src_scales[0] * dst_scales[0];
    MAYBE_UNUSED(alpha);
    const float beta = pd()->beta();

    const auto &dims = input_d.dims();
    const auto in_strides = input_d.blocking_desc().strides;
    const auto M = dims[0];
    const auto K = dims[1];

    // Calculate block sizes
    dim_t M_b = 16;
    dim_t K_b = 64;
    K_b = std::min(K_b, K);

    const dim_t num_M_blocks = (M + M_b - 1) / M_b;
    const dim_t num_K_blocks = (K + K_b - 1) / K_b;

    parallel_nd(num_M_blocks, num_K_blocks, [&](dim_t mb, dim_t kb) {
        dim_t M_start = mb * M_b;
        dim_t M_end = nstl::min(M_start + M_b, M);
        dim_t K_start = kb * K_b;
        dim_t K_end = nstl::min(K_start + K_b, K);
        // Iterate over the block
        for (dim_t i = M_start; i < M_end; ++i) {
            kernel<const float, unsigned char>(
                    input + i * in_strides[0] + K_start,
                    output + i * in_strides[0] + K_start, K_end - K_start,
                    src_scales[0], dst_scales[0], src_zp, dst_zp, beta);
        }
    });

    return status::success;
}

} // namespace ppc64
} // namespace cpu
} // namespace impl
} // namespace dnnl
