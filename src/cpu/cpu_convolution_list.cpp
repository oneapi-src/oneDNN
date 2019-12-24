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

#include "cpu_engine.hpp"

#include "cpu/gemm_bf16_convolution.hpp"
#include "cpu/gemm_convolution.hpp"
#include "cpu/gemm_x8s8s32x_convolution.hpp"
#include "cpu/jit_avx2_1x1_convolution.hpp"
#include "cpu/jit_avx2_convolution.hpp"
#include "cpu/jit_avx2_x8s8s32x_1x1_convolution.hpp"
#include "cpu/jit_avx2_x8s8s32x_convolution.hpp"
#include "cpu/jit_avx512_common_1x1_convolution.hpp"
#include "cpu/jit_avx512_common_convolution.hpp"
#include "cpu/jit_avx512_common_convolution_winograd.hpp"
#include "cpu/jit_avx512_core_bf16_1x1_convolution.hpp"
#include "cpu/jit_avx512_core_bf16_convolution.hpp"
#include "cpu/jit_avx512_core_f32_wino_conv_2x3.hpp"
#include "cpu/jit_avx512_core_f32_wino_conv_4x3.hpp"
#include "cpu/jit_avx512_core_u8s8s32x_wino_convolution.hpp"
#include "cpu/jit_avx512_core_x8s8s32x_1x1_convolution.hpp"
#include "cpu/jit_avx512_core_x8s8s32x_convolution.hpp"
#include "cpu/jit_sse41_1x1_convolution.hpp"
#include "cpu/jit_sse41_convolution.hpp"
#include "cpu/jit_uni_dw_convolution.hpp"
#include "cpu/ref_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using pd_create_f = engine_t::primitive_desc_create_f;

namespace {
using namespace dnnl::impl::data_type;

#define INSTANCE(...) &primitive_desc_t::create<__VA_ARGS__::pd_t>
static const pd_create_f impl_list[] = {
        /* f32 */
        INSTANCE(jit_avx512_common_dw_convolution_fwd_t),
        INSTANCE(jit_avx512_common_dw_convolution_bwd_data_t),
        INSTANCE(jit_avx512_common_dw_convolution_bwd_weights_t),
        INSTANCE(jit_avx512_common_1x1_convolution_fwd_f32_t),
        INSTANCE(jit_avx512_common_1x1_convolution_bwd_data_f32_t),
        INSTANCE(jit_avx512_common_1x1_convolution_bwd_weights_t),
        INSTANCE(jit_avx512_core_f32_wino_conv_2x3_fwd_t),
        INSTANCE(jit_avx512_core_f32_wino_conv_4x3_fwd_t),
        INSTANCE(jit_avx512_core_f32_wino_conv_4x3_bwd_data_t),
        INSTANCE(jit_avx512_core_f32_wino_conv_4x3_bwd_weights_t),
        INSTANCE(jit_avx512_common_convolution_winograd_fwd_t),
        INSTANCE(jit_avx512_common_convolution_winograd_bwd_data_t),
        INSTANCE(jit_avx512_common_convolution_winograd_bwd_weights_t),
        INSTANCE(jit_avx512_common_convolution_fwd_t<f32>),
        INSTANCE(jit_avx512_common_convolution_bwd_data_t<f32>),
        INSTANCE(jit_avx512_common_convolution_bwd_weights_t<f32>),
        INSTANCE(jit_avx2_dw_convolution_fwd_t),
        INSTANCE(jit_avx2_dw_convolution_bwd_data_t),
        INSTANCE(jit_avx2_dw_convolution_bwd_weights_t),
        INSTANCE(jit_avx2_1x1_convolution_fwd_t),
        INSTANCE(jit_avx2_1x1_convolution_bwd_data_t),
        INSTANCE(jit_avx2_1x1_convolution_bwd_weights_t),
        INSTANCE(jit_sse41_dw_convolution_fwd_t),
        INSTANCE(jit_sse41_dw_convolution_bwd_data_t),
        INSTANCE(jit_sse41_dw_convolution_bwd_weights_t),
        INSTANCE(jit_sse41_1x1_convolution_fwd_t),
        INSTANCE(jit_avx2_convolution_fwd_t),
        INSTANCE(jit_avx2_convolution_bwd_data_t),
        INSTANCE(jit_avx2_convolution_bwd_weights_t),
        INSTANCE(jit_sse41_convolution_fwd_t),
        INSTANCE(gemm_convolution_fwd_t),
        INSTANCE(gemm_convolution_bwd_data_t),
        INSTANCE(gemm_convolution_bwd_weights_t),
        INSTANCE(ref_convolution_fwd_t<f32>),
        INSTANCE(ref_convolution_bwd_data_t<f32, f32, f32, f32>),
        INSTANCE(ref_convolution_bwd_weights_t<f32, f32, f32, f32>),
        /* bfloat16 */
        INSTANCE(jit_uni_dw_convolution_fwd_t<avx512_core, bf16, bf16>),
        INSTANCE(jit_uni_dw_convolution_fwd_t<avx512_core, bf16, f32>),
        INSTANCE(jit_uni_dw_convolution_bwd_data_t<avx512_core, bf16, bf16>),
        INSTANCE(jit_uni_dw_convolution_bwd_data_t<avx512_core, bf16, f32>),
        INSTANCE(jit_uni_dw_convolution_bwd_weights_t<avx512_core, bf16, bf16>),
        INSTANCE(jit_uni_dw_convolution_bwd_weights_t<avx512_core, bf16, f32>),
        INSTANCE(jit_avx512_core_bf16_1x1_convolution_fwd_t<f32>),
        INSTANCE(jit_avx512_core_bf16_1x1_convolution_fwd_t<bf16>),
        INSTANCE(jit_avx512_core_bf16_1x1_convolution_bwd_data_t<f32>),
        INSTANCE(jit_avx512_core_bf16_1x1_convolution_bwd_data_t<bf16>),
        INSTANCE(jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<f32>),
        INSTANCE(jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<bf16>),
        INSTANCE(jit_avx512_core_bf16_convolution_fwd_t),
        INSTANCE(jit_avx512_core_bf16_convolution_bwd_data_t),
        INSTANCE(jit_avx512_core_bf16_convolution_bwd_weights_t),
        INSTANCE(gemm_bf16_convolution_fwd_t<f32>),
        INSTANCE(gemm_bf16_convolution_fwd_t<bf16>),
        INSTANCE(gemm_bf16_convolution_bwd_data_t<f32>),
        INSTANCE(gemm_bf16_convolution_bwd_data_t<bf16>),
        INSTANCE(gemm_bf16_convolution_bwd_weights_t<f32>),
        INSTANCE(gemm_bf16_convolution_bwd_weights_t<bf16>),
        /* int */
        INSTANCE(jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<f32>),
        INSTANCE(jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<s32>),
        INSTANCE(jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<s8>),
        INSTANCE(jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<u8>),
        INSTANCE(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<u8, f32>),
        INSTANCE(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<u8, s32>),
        INSTANCE(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<u8, u8>),
        INSTANCE(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<u8, s8>),
        INSTANCE(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<s8, f32>),
        INSTANCE(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<s8, s32>),
        INSTANCE(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<s8, u8>),
        INSTANCE(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<s8, s8>),
        INSTANCE(jit_avx512_core_x8s8s32x_convolution_fwd_t<u8, f32>),
        INSTANCE(jit_avx512_core_x8s8s32x_convolution_fwd_t<u8, s32>),
        INSTANCE(jit_avx512_core_x8s8s32x_convolution_fwd_t<u8, u8>),
        INSTANCE(jit_avx512_core_x8s8s32x_convolution_fwd_t<u8, s8>),
        INSTANCE(jit_avx512_core_x8s8s32x_convolution_fwd_t<s8, f32>),
        INSTANCE(jit_avx512_core_x8s8s32x_convolution_fwd_t<s8, s32>),
        INSTANCE(jit_avx512_core_x8s8s32x_convolution_fwd_t<s8, u8>),
        INSTANCE(jit_avx512_core_x8s8s32x_convolution_fwd_t<s8, s8>),
        INSTANCE(jit_avx2_x8s8s32x_1x1_convolution_fwd_t<u8, f32>),
        INSTANCE(jit_avx2_x8s8s32x_1x1_convolution_fwd_t<u8, s32>),
        INSTANCE(jit_avx2_x8s8s32x_1x1_convolution_fwd_t<u8, u8>),
        INSTANCE(jit_avx2_x8s8s32x_1x1_convolution_fwd_t<u8, s8>),
        INSTANCE(jit_avx2_x8s8s32x_1x1_convolution_fwd_t<s8, f32>),
        INSTANCE(jit_avx2_x8s8s32x_1x1_convolution_fwd_t<s8, s32>),
        INSTANCE(jit_avx2_x8s8s32x_1x1_convolution_fwd_t<s8, u8>),
        INSTANCE(jit_avx2_x8s8s32x_1x1_convolution_fwd_t<s8, s8>),
        INSTANCE(jit_avx2_x8s8s32x_convolution_fwd_t<u8, f32>),
        INSTANCE(jit_avx2_x8s8s32x_convolution_fwd_t<u8, s32>),
        INSTANCE(jit_avx2_x8s8s32x_convolution_fwd_t<u8, u8>),
        INSTANCE(jit_avx2_x8s8s32x_convolution_fwd_t<u8, s8>),
        INSTANCE(jit_avx2_x8s8s32x_convolution_fwd_t<s8, f32>),
        INSTANCE(jit_avx2_x8s8s32x_convolution_fwd_t<s8, s32>),
        INSTANCE(jit_avx2_x8s8s32x_convolution_fwd_t<s8, u8>),
        INSTANCE(jit_avx2_x8s8s32x_convolution_fwd_t<s8, s8>),
        INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<u8, s32>),
        INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<u8, u8>),
        INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<u8, s8>),
        INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<u8, f32>),
        INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<s8, s32>),
        INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<s8, u8>),
        INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<s8, s8>),
        INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<s8, f32>),
        INSTANCE(_gemm_u8s8s32x_convolution_bwd_data_t<s32>),
        INSTANCE(_gemm_u8s8s32x_convolution_bwd_data_t<u8>),
        INSTANCE(_gemm_u8s8s32x_convolution_bwd_data_t<s8>),
        INSTANCE(_gemm_u8s8s32x_convolution_bwd_data_t<f32>),
        INSTANCE(ref_convolution_fwd_t<u8, s8, f32, s32>),
        INSTANCE(ref_convolution_fwd_t<u8, s8, s32, s32>),
        INSTANCE(ref_convolution_fwd_t<u8, s8, s8, s32>),
        INSTANCE(ref_convolution_fwd_t<u8, s8, u8, s32>),
        INSTANCE(ref_convolution_bwd_data_t<f32, s8, u8, s32>),
        INSTANCE(ref_convolution_bwd_data_t<s32, s8, u8, s32>),
        INSTANCE(ref_convolution_bwd_data_t<s8, s8, u8, s32>),
        INSTANCE(ref_convolution_bwd_data_t<u8, s8, u8, s32>),
        /* eol */
        nullptr,
};
#undef INSTANCE
} // namespace

const pd_create_f *get_convolution_impl_list(const convolution_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
