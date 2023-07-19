/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
* Copyright 2020-2023 Arm Ltd. and affiliates
* Copyright 2020-2021 FUJITSU LIMITED
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

#include <map>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"

#include "cpu/cpu_engine.hpp"

#include "cpu/gemm_convolution.hpp"
#include "cpu/gemm_x8s8s32x_convolution.hpp"
#include "cpu/ref_convolution.hpp"
#include "cpu/ref_convolution_int8.hpp"
#include "cpu/ref_fused_convolution.hpp"

#if DNNL_X64
#include "cpu/x64/gemm_bf16_convolution.hpp"
#include "cpu/x64/ip_convolution.hpp"
#include "cpu/x64/jit_avx2_1x1_convolution.hpp"
#include "cpu/x64/jit_avx2_convolution.hpp"
#include "cpu/x64/jit_avx512_common_1x1_convolution.hpp"
#include "cpu/x64/jit_avx512_common_convolution.hpp"
#include "cpu/x64/jit_avx512_core_amx_1x1_convolution.hpp"
#include "cpu/x64/jit_avx512_core_amx_convolution.hpp"
#include "cpu/x64/jit_avx512_core_bf16_1x1_convolution.hpp"
#include "cpu/x64/jit_avx512_core_bf16_convolution.hpp"
#include "cpu/x64/jit_avx512_core_x8s8s32x_1x1_convolution.hpp"
#include "cpu/x64/jit_avx512_core_x8s8s32x_convolution.hpp"
#include "cpu/x64/jit_brdgmm_dw_conv.hpp"
#include "cpu/x64/jit_brgemm_1x1_conv.hpp"
#include "cpu/x64/jit_brgemm_conv.hpp"
#include "cpu/x64/jit_brgemm_conv_bwd.hpp"
#include "cpu/x64/jit_brgemm_conv_bwd_strided.hpp"
#include "cpu/x64/jit_brgemm_conv_bwd_w.hpp"
#include "cpu/x64/jit_sse41_1x1_convolution.hpp"
#include "cpu/x64/jit_sse41_convolution.hpp"
#include "cpu/x64/jit_uni_dw_convolution.hpp"
#include "cpu/x64/jit_uni_x8s8s32x_1x1_convolution.hpp"
#include "cpu/x64/jit_uni_x8s8s32x_convolution.hpp"
using namespace dnnl::impl::cpu::x64;
#elif DNNL_AARCH64
#include "cpu/aarch64/jit_sve_512_1x1_convolution.hpp"
#include "cpu/aarch64/jit_sve_512_convolution.hpp"
#include "cpu/aarch64/jit_sve_512_x8s8s32x_convolution.hpp"
#include "cpu/aarch64/jit_uni_dw_convolution.hpp"
#if DNNL_AARCH64 && DNNL_AARCH64_USE_ACL
#include "cpu/aarch64/acl_depthwise_convolution.hpp"
#include "cpu/aarch64/acl_gemm_convolution.hpp"
#include "cpu/aarch64/acl_indirect_gemm_convolution.hpp"
#endif
using namespace dnnl::impl::cpu::aarch64;
#endif

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::prop_kind;

// clang-format off
const std::map<pk_dt_impl_key_t, std::vector<impl_list_item_t>> &impl_list_map() {
    static const std::map<pk_dt_impl_key_t, std::vector<impl_list_item_t>> the_map = REG_CONV_P({
        // FWD fp
        {{forward, f32, f32, f32}, {
            CPU_INSTANCE_AVX512(brdgmm_dw_convolution_fwd_t)
            CPU_INSTANCE_X64(ip_convolution_fwd_t)
            CPU_INSTANCE_AMX(brgemm_1x1_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx, true>)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core, true>)
            CPU_INSTANCE_AVX512(jit_avx512_common_dw_convolution_fwd_t)
            CPU_INSTANCE_AVX512(jit_avx512_common_1x1_convolution_fwd_f32_t)
            CPU_INSTANCE_AVX512(jit_avx512_common_convolution_fwd_t<f32>)
            CPU_INSTANCE_AVX2(jit_avx2_dw_convolution_fwd_t)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2, true>)
            CPU_INSTANCE_AVX2(jit_avx2_1x1_convolution_fwd_t)
            CPU_INSTANCE_SSE41(jit_sse41_dw_convolution_fwd_t)
            CPU_INSTANCE_SSE41(jit_sse41_1x1_convolution_fwd_t)
            CPU_INSTANCE_AVX2(jit_avx2_convolution_fwd_t)
            CPU_INSTANCE_SSE41(jit_sse41_convolution_fwd_t)
            CPU_INSTANCE_AARCH64(jit_sve_512_dw_convolution_fwd_t)
            CPU_INSTANCE_AARCH64(jit_sve_512_1x1_convolution_fwd_f32_t)
            CPU_INSTANCE_AARCH64(jit_sve_512_convolution_fwd_t<f32>)
            CPU_INSTANCE_AARCH64_ACL(acl_depthwise_convolution_fwd_t)
            CPU_INSTANCE_AARCH64_ACL(acl_indirect_gemm_convolution_fwd_t)
            CPU_INSTANCE_AARCH64_ACL(acl_gemm_convolution_fwd_t<f32>)
            CPU_INSTANCE(gemm_convolution_fwd_t)
            CPU_INSTANCE(ref_convolution_fwd_t)
            CPU_INSTANCE(ref_fused_convolution_fwd_t)
            nullptr,
        }},
        {{forward, bf16, bf16, f32}, {
            CPU_INSTANCE_AVX512(brdgmm_dw_convolution_fwd_t)
            CPU_INSTANCE_X64(ip_convolution_fwd_t)
            CPU_INSTANCE_AMX(brgemm_1x1_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx, true>)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_1x1_convolution_fwd_t)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_convolution_fwd_t)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core_bf16>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_bf16>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_bf16, true>)
            CPU_INSTANCE_AVX512(jit_uni_dw_convolution_fwd_t<avx512_core, bf16, f32>)
            CPU_INSTANCE_AVX512(jit_avx512_core_bf16_1x1_convolution_fwd_t<f32>)
            CPU_INSTANCE_AVX512(jit_avx512_core_bf16_convolution_fwd_t)
            CPU_INSTANCE_AVX512(gemm_bf16_convolution_fwd_t<f32>)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2, true>)
            CPU_INSTANCE(ref_convolution_fwd_t)
            nullptr,
        }},
        {{forward, bf16, bf16, bf16}, {
            CPU_INSTANCE_AVX512(brdgmm_dw_convolution_fwd_t)
            CPU_INSTANCE_X64(ip_convolution_fwd_t)
            CPU_INSTANCE_AMX(brgemm_1x1_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx, true>)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_1x1_convolution_fwd_t)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_convolution_fwd_t)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core_bf16>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_bf16>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_bf16, true>)
            CPU_INSTANCE_AVX512(jit_uni_dw_convolution_fwd_t<avx512_core, bf16, bf16>)
            CPU_INSTANCE_AVX512(jit_avx512_core_bf16_1x1_convolution_fwd_t<bf16>)
            CPU_INSTANCE_AVX512(jit_avx512_core_bf16_convolution_fwd_t)
            CPU_INSTANCE_AVX512(gemm_bf16_convolution_fwd_t<bf16>)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2, true>)
            CPU_INSTANCE(ref_convolution_fwd_t)
            CPU_INSTANCE(ref_fused_convolution_fwd_t)
            nullptr,
        }},
        {{forward, f16, f16, f32}, {
            CPU_INSTANCE_AVX512(brdgmm_dw_convolution_fwd_t)
            CPU_INSTANCE_X64(ip_convolution_fwd_t)
            CPU_INSTANCE_AMX(brgemm_1x1_convolution_fwd_t<avx512_core_amx_fp16>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx_fp16>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx_fp16, true>)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core_fp16>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_fp16>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_fp16, true>)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2, true>)
            CPU_INSTANCE(ref_convolution_fwd_t)
            nullptr,
        }},
        {{forward, f16, f16, f16}, {
            CPU_INSTANCE_AVX512(brdgmm_dw_convolution_fwd_t)
            CPU_INSTANCE_X64(ip_convolution_fwd_t)
            CPU_INSTANCE_AMX(brgemm_1x1_convolution_fwd_t<avx512_core_amx_fp16>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx_fp16>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx_fp16, true>)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core_fp16>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_fp16>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_fp16, true>)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2, true>)
            CPU_INSTANCE_AARCH64_ACL(acl_indirect_gemm_convolution_fwd_t)
            CPU_INSTANCE_AARCH64_ACL(acl_gemm_convolution_fwd_t<f16>)
            CPU_INSTANCE(ref_convolution_fwd_t)
            CPU_INSTANCE(ref_fused_convolution_fwd_t)
            nullptr,
        }},
        // BWD_D fp
        {{backward_data, f32, f32, f32}, REG_BWD_D_PK({
            CPU_INSTANCE_X64(ip_convolution_bwd_data_t)
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_t<avx512_core_amx>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core, true>)
            CPU_INSTANCE_AVX512(jit_avx512_common_dw_convolution_bwd_data_t)
            CPU_INSTANCE_AVX512(jit_avx512_common_1x1_convolution_bwd_data_f32_t)
            CPU_INSTANCE_AVX512(jit_avx512_common_convolution_bwd_data_t<f32>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_t<avx2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2, true>)
            CPU_INSTANCE_AVX2(jit_avx2_dw_convolution_bwd_data_t)
            CPU_INSTANCE_AVX2(jit_avx2_1x1_convolution_bwd_data_t)
            CPU_INSTANCE_SSE41(jit_sse41_dw_convolution_bwd_data_t)
            CPU_INSTANCE_AVX2(jit_avx2_convolution_bwd_data_t)
            CPU_INSTANCE_AARCH64(jit_sve_512_dw_convolution_bwd_data_t)
            CPU_INSTANCE_AARCH64(jit_sve_512_1x1_convolution_bwd_data_f32_t)
            CPU_INSTANCE_AARCH64(jit_sve_512_convolution_bwd_data_t<f32>)
            CPU_INSTANCE(gemm_convolution_bwd_data_t)
            CPU_INSTANCE(ref_convolution_bwd_data_t)
            nullptr,
        })},
        {{backward_data, f32, bf16, bf16}, REG_BWD_D_PK({
            CPU_INSTANCE_X64(ip_convolution_bwd_data_t)
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_strided_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_strided_t<avx512_core_amx, true>)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_convolution_bwd_data_t)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_t<avx512_core_bf16>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core_bf16>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core_bf16, true>)
            CPU_INSTANCE_AVX512(jit_uni_dw_convolution_bwd_data_t<avx512_core, bf16, f32>)
            CPU_INSTANCE_AVX512(jit_avx512_core_bf16_1x1_convolution_bwd_data_t<f32>)
            CPU_INSTANCE_AVX512(jit_avx512_core_bf16_convolution_bwd_data_t)
            CPU_INSTANCE_AVX512(gemm_bf16_convolution_bwd_data_t<f32>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni_2, true>)
            CPU_INSTANCE(ref_convolution_bwd_data_t)
            nullptr,
        })},
        {{backward_data, bf16, bf16, bf16}, REG_BWD_D_PK({
            CPU_INSTANCE_X64(ip_convolution_bwd_data_t)
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_strided_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_strided_t<avx512_core_amx, true>)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_convolution_bwd_data_t)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_t<avx512_core_bf16>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core_bf16>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core_bf16, true>)
            CPU_INSTANCE_AVX512(jit_uni_dw_convolution_bwd_data_t<avx512_core, bf16, bf16>)
            CPU_INSTANCE_AVX512(jit_avx512_core_bf16_1x1_convolution_bwd_data_t<bf16>)
            CPU_INSTANCE_AVX512(jit_avx512_core_bf16_convolution_bwd_data_t)
            CPU_INSTANCE_AVX512(gemm_bf16_convolution_bwd_data_t<bf16>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni_2, true>)
            CPU_INSTANCE(ref_convolution_bwd_data_t)
            nullptr,
        })},
        {{backward_data, f32, f16, f16}, REG_BWD_D_PK({
            CPU_INSTANCE_X64(ip_convolution_bwd_data_t)
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_t<avx512_core_amx_fp16>)
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_strided_t<avx512_core_amx_fp16>)
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_strided_t<avx512_core_amx_fp16, true>)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_convolution_bwd_data_t)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_t<avx512_core_fp16>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core_fp16>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core_fp16, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni_2, true>)
            CPU_INSTANCE(ref_convolution_bwd_data_t)
            nullptr,
        })},
        {{backward_data, f16, f16, f16}, REG_BWD_D_PK({
            CPU_INSTANCE_X64(ip_convolution_bwd_data_t)
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_t<avx512_core_amx_fp16>)
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_strided_t<avx512_core_amx_fp16>)
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_strided_t<avx512_core_amx_fp16, true>)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_convolution_bwd_data_t)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_t<avx512_core_fp16>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core_fp16>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core_fp16, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni_2, true>)
            CPU_INSTANCE(ref_convolution_bwd_data_t)
            nullptr,
        })},
        // BWD_W fp
        {{backward_weights, f32, f32, f32}, REG_BWD_PK({
            CPU_INSTANCE_X64(ip_convolution_bwd_weights_t)
            CPU_INSTANCE_AVX512(jit_avx512_common_dw_convolution_bwd_weights_t)
            CPU_INSTANCE_AVX512(jit_avx512_common_1x1_convolution_bwd_weights_t)
            CPU_INSTANCE_AVX512(jit_avx512_common_convolution_bwd_weights_t<f32>)
            CPU_INSTANCE_AVX2(jit_avx2_dw_convolution_bwd_weights_t)
            CPU_INSTANCE_AVX2(jit_avx2_1x1_convolution_bwd_weights_t)
            CPU_INSTANCE_SSE41(jit_sse41_dw_convolution_bwd_weights_t)
            CPU_INSTANCE_AVX2(jit_avx2_convolution_bwd_weights_t)
            CPU_INSTANCE_AARCH64(jit_sve_512_dw_convolution_bwd_weights_t)
            CPU_INSTANCE_AARCH64(jit_sve_512_1x1_convolution_bwd_weights_t)
            CPU_INSTANCE_AARCH64(jit_sve_512_convolution_bwd_weights_t<f32>)
            CPU_INSTANCE(gemm_convolution_bwd_weights_t)
            CPU_INSTANCE(ref_convolution_bwd_weights_t)
            nullptr,
        })},
        {{backward_weights, bf16, f32, bf16}, REG_BWD_PK({
            CPU_INSTANCE_X64(ip_convolution_bwd_weights_t)
            CPU_INSTANCE_AVX512(jit_uni_dw_convolution_bwd_weights_t<avx512_core, bf16, f32>)
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_weights_t)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_convolution_bwd_weights_t)
            CPU_INSTANCE_AVX512(jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<f32>)
            CPU_INSTANCE_AVX512(jit_avx512_core_bf16_convolution_bwd_weights_t)
            CPU_INSTANCE_AVX512(gemm_bf16_convolution_bwd_weights_t<f32>)
            CPU_INSTANCE(ref_convolution_bwd_weights_t)
            nullptr,
        })},
        {{backward_weights, bf16, bf16, bf16}, REG_BWD_PK({
            CPU_INSTANCE_X64(ip_convolution_bwd_weights_t)
            CPU_INSTANCE_AVX512(jit_uni_dw_convolution_bwd_weights_t<avx512_core, bf16, bf16>)
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_weights_t)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_convolution_bwd_weights_t)
            CPU_INSTANCE_AVX512(jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<bf16>)
            CPU_INSTANCE_AVX512(jit_avx512_core_bf16_convolution_bwd_weights_t)
            CPU_INSTANCE_AVX512(gemm_bf16_convolution_bwd_weights_t<bf16>)
            CPU_INSTANCE(ref_convolution_bwd_weights_t)
            nullptr,
        })},
        {{backward_weights, f16, f32, f16}, REG_BWD_PK({
            CPU_INSTANCE_X64(ip_convolution_bwd_weights_t)
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_weights_t)
            CPU_INSTANCE(ref_convolution_bwd_weights_t)
            nullptr,
        })},
        {{backward_weights, f16, f16, f16}, REG_BWD_PK({
            CPU_INSTANCE_X64(ip_convolution_bwd_weights_t)
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_weights_t)
            CPU_INSTANCE(ref_convolution_bwd_weights_t)
            nullptr,
        })},
        // FWD int8 (src:s8)
        {{forward, s8, s8, f32}, {
            CPU_INSTANCE_AVX512(brdgmm_dw_convolution_fwd_t)
            CPU_INSTANCE_X64(ip_convolution_fwd_t)
            CPU_INSTANCE_AMX(brgemm_1x1_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx, true>)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_1x1_convolution_fwd_t)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_convolution_fwd_t)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core, true>)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni, true>)
            CPU_INSTANCE_AVX2(jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2>)
            CPU_INSTANCE_AVX2(jit_uni_x8s8s32x_convolution_fwd_t<avx2>)
            CPU_INSTANCE_SSE41(jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41>)
            CPU_INSTANCE_SSE41(jit_uni_x8s8s32x_convolution_fwd_t<sse41>)
            CPU_INSTANCE_AARCH64(jit_sve_512_x8s8s32x_convolution_fwd_t<s8, f32>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE(ref_convolution_int8_fwd_t)
            CPU_INSTANCE(ref_fused_convolution_fwd_t)
            nullptr,
        }},
        {{forward, s8, s8, bf16}, {
            CPU_INSTANCE_AVX512(brdgmm_dw_convolution_fwd_t)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_1x1_convolution_fwd_t)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_convolution_fwd_t)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core, true>)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2, true>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE(ref_convolution_int8_fwd_t)
            nullptr,
        }},
        {{forward, s8, s8, s32}, {
            CPU_INSTANCE_AVX512(brdgmm_dw_convolution_fwd_t)
            CPU_INSTANCE_X64(ip_convolution_fwd_t)
            CPU_INSTANCE_AMX(brgemm_1x1_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx, true>)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_1x1_convolution_fwd_t)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_convolution_fwd_t)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core, true>)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni, true>)
            CPU_INSTANCE_AVX2(jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2>)
            CPU_INSTANCE_AVX2(jit_uni_x8s8s32x_convolution_fwd_t<avx2>)
            CPU_INSTANCE_SSE41(jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41>)
            CPU_INSTANCE_SSE41(jit_uni_x8s8s32x_convolution_fwd_t<sse41>)
            CPU_INSTANCE_AARCH64(jit_sve_512_x8s8s32x_convolution_fwd_t<s8, s32>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE(ref_convolution_int8_fwd_t)
            CPU_INSTANCE(ref_fused_convolution_fwd_t)
            nullptr,
        }},
        {{forward, s8, s8, s8}, {
            CPU_INSTANCE_AVX512(brdgmm_dw_convolution_fwd_t)
            CPU_INSTANCE_X64(ip_convolution_fwd_t)
            CPU_INSTANCE_AMX(brgemm_1x1_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx, true>)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_1x1_convolution_fwd_t)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_convolution_fwd_t)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core, true>)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni, true>)
            CPU_INSTANCE_AVX2(jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2>)
            CPU_INSTANCE_AVX2(jit_uni_x8s8s32x_convolution_fwd_t<avx2>)
            CPU_INSTANCE_SSE41(jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41>)
            CPU_INSTANCE_SSE41(jit_uni_x8s8s32x_convolution_fwd_t<sse41>)
            CPU_INSTANCE_AARCH64(jit_sve_512_x8s8s32x_convolution_fwd_t<s8, s8>)
            CPU_INSTANCE_AARCH64_ACL(acl_gemm_convolution_fwd_t<s8, s8, s8, s32>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE(ref_convolution_int8_fwd_t)
            CPU_INSTANCE(ref_fused_convolution_fwd_t)
            nullptr,
        }},
        {{forward, s8, s8, u8}, {
            CPU_INSTANCE_AVX512(brdgmm_dw_convolution_fwd_t)
            CPU_INSTANCE_X64(ip_convolution_fwd_t)
            CPU_INSTANCE_AMX(brgemm_1x1_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx, true>)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_1x1_convolution_fwd_t)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_convolution_fwd_t)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core, true>)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni, true>)
            CPU_INSTANCE_AVX2(jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2>)
            CPU_INSTANCE_AVX2(jit_uni_x8s8s32x_convolution_fwd_t<avx2>)
            CPU_INSTANCE_SSE41(jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41>)
            CPU_INSTANCE_SSE41(jit_uni_x8s8s32x_convolution_fwd_t<sse41>)
            CPU_INSTANCE_AARCH64(jit_sve_512_x8s8s32x_convolution_fwd_t<s8, u8>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE(ref_convolution_int8_fwd_t)
            CPU_INSTANCE(ref_fused_convolution_fwd_t)
            nullptr,
        }},
        // FWD int8 (src:u8)
        {{forward, u8, s8, f32}, {
            CPU_INSTANCE_AVX512(brdgmm_dw_convolution_fwd_t)
            CPU_INSTANCE_X64(ip_convolution_fwd_t)
            CPU_INSTANCE_AMX(brgemm_1x1_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx, true>)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_1x1_convolution_fwd_t)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_convolution_fwd_t)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core, true>)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni, true>)
            CPU_INSTANCE_AVX2(jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2>)
            CPU_INSTANCE_AVX2(jit_uni_x8s8s32x_convolution_fwd_t<avx2>)
            CPU_INSTANCE_SSE41(jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41>)
            CPU_INSTANCE_SSE41(jit_uni_x8s8s32x_convolution_fwd_t<sse41>)
            CPU_INSTANCE_AARCH64(jit_sve_512_x8s8s32x_convolution_fwd_t<u8, f32>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE(ref_convolution_int8_fwd_t)
            nullptr,
        }},
        {{forward, u8, s8, bf16}, {
            CPU_INSTANCE_AVX512(brdgmm_dw_convolution_fwd_t)
            CPU_INSTANCE_AMX(brgemm_1x1_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx, true>)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_1x1_convolution_fwd_t)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_convolution_fwd_t)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core, true>)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2, true>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE(ref_convolution_int8_fwd_t)
            nullptr,
        }},
        {{forward, u8, s8, s32}, {
            CPU_INSTANCE_AVX512(brdgmm_dw_convolution_fwd_t)
            CPU_INSTANCE_X64(ip_convolution_fwd_t)
            CPU_INSTANCE_AMX(brgemm_1x1_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx, true>)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_1x1_convolution_fwd_t)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_convolution_fwd_t)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core, true>)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni, true>)
            CPU_INSTANCE_AVX2(jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2>)
            CPU_INSTANCE_AVX2(jit_uni_x8s8s32x_convolution_fwd_t<avx2>)
            CPU_INSTANCE_SSE41(jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41>)
            CPU_INSTANCE_SSE41(jit_uni_x8s8s32x_convolution_fwd_t<sse41>)
            CPU_INSTANCE_AARCH64(jit_sve_512_x8s8s32x_convolution_fwd_t<u8, s32>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE(ref_convolution_int8_fwd_t)
            nullptr,
        }},
        {{forward, u8, s8, s8}, {
            CPU_INSTANCE_AVX512(brdgmm_dw_convolution_fwd_t)
            CPU_INSTANCE_X64(ip_convolution_fwd_t)
            CPU_INSTANCE_AMX(brgemm_1x1_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx, true>)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_1x1_convolution_fwd_t)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_convolution_fwd_t)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core, true>)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni, true>)
            CPU_INSTANCE_AVX2(jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2>)
            CPU_INSTANCE_AVX2(jit_uni_x8s8s32x_convolution_fwd_t<avx2>)
            CPU_INSTANCE_SSE41(jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41>)
            CPU_INSTANCE_SSE41(jit_uni_x8s8s32x_convolution_fwd_t<sse41>)
            CPU_INSTANCE_AARCH64(jit_sve_512_x8s8s32x_convolution_fwd_t<u8, s8>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE(ref_convolution_int8_fwd_t)
            CPU_INSTANCE(ref_fused_convolution_fwd_t)
            nullptr,
        }},
        {{forward, u8, s8, u8}, {
            CPU_INSTANCE_AVX512(brdgmm_dw_convolution_fwd_t)
            CPU_INSTANCE_X64(ip_convolution_fwd_t)
            CPU_INSTANCE_AMX(brgemm_1x1_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(brgemm_convolution_fwd_t<avx512_core_amx, true>)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_1x1_convolution_fwd_t)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_convolution_fwd_t)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_1x1_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(brgemm_convolution_fwd_t<avx512_core, true>)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_1x1_convolution_fwd_t<avx2_vnni>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni_2, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni>)
            CPU_INSTANCE_AVX2(brgemm_convolution_fwd_t<avx2_vnni, true>)
            CPU_INSTANCE_AVX2(jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2>)
            CPU_INSTANCE_AVX2(jit_uni_x8s8s32x_convolution_fwd_t<avx2>)
            CPU_INSTANCE_SSE41(jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41>)
            CPU_INSTANCE_SSE41(jit_uni_x8s8s32x_convolution_fwd_t<sse41>)
            CPU_INSTANCE_AARCH64(jit_sve_512_x8s8s32x_convolution_fwd_t<u8, u8>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_fwd_t)
            CPU_INSTANCE(ref_convolution_int8_fwd_t)
            CPU_INSTANCE(ref_fused_convolution_fwd_t)
            nullptr,
        }},
        // BWD int8 (diff_dst:u8)
        {{backward_data, f32, s8, u8}, REG_BWD_D_PK({
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_strided_t<avx512_core_amx, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni_2, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni, true>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_bwd_data_t)
            CPU_INSTANCE(ref_convolution_int8_bwd_data_t)
            nullptr,
        })},
        {{backward_data, bf16, s8, u8}, REG_BWD_D_PK({
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_strided_t<avx512_core_amx, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni_2, true>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_bwd_data_t)
            CPU_INSTANCE(ref_convolution_int8_bwd_data_t)
            nullptr,
        })},
        {{backward_data, s32, s8, u8}, REG_BWD_D_PK({
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_strided_t<avx512_core_amx, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni_2, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni, true>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_bwd_data_t)
            CPU_INSTANCE(ref_convolution_int8_bwd_data_t)
            nullptr,
        })},
        {{backward_data, s8, s8, u8}, REG_BWD_D_PK({
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_strided_t<avx512_core_amx, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni_2, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni, true>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_bwd_data_t)
            CPU_INSTANCE(ref_convolution_int8_bwd_data_t)
            nullptr,
        })},
        {{backward_data, u8, s8, u8}, REG_BWD_D_PK({
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_strided_t<avx512_core_amx, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni_2, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni, true>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_bwd_data_t)
            CPU_INSTANCE(ref_convolution_int8_bwd_data_t)
            nullptr,
        })},
        // BWD int8 (diff_dst:s8)
        {{backward_data, f32, s8, s8}, REG_BWD_D_PK({
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_strided_t<avx512_core_amx, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni_2, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni, true>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_bwd_data_t)
            CPU_INSTANCE(ref_convolution_int8_bwd_data_t)
            nullptr,
        })},
        {{backward_data, bf16, s8, s8}, REG_BWD_D_PK({
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_strided_t<avx512_core_amx, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni_2, true>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_bwd_data_t)
            CPU_INSTANCE(ref_convolution_int8_bwd_data_t)
            nullptr,
        })},
        {{backward_data, s32, s8, s8}, REG_BWD_D_PK({
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_strided_t<avx512_core_amx, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni_2, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni, true>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_bwd_data_t)
            CPU_INSTANCE(ref_convolution_int8_bwd_data_t)
            nullptr,
        })},
        {{backward_data, s8, s8, s8}, REG_BWD_D_PK({
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_strided_t<avx512_core_amx, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni_2, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni, true>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_bwd_data_t)
            CPU_INSTANCE(ref_convolution_int8_bwd_data_t)
            nullptr,
        })},
        {{backward_data, u8, s8, s8}, REG_BWD_D_PK({
            CPU_INSTANCE_AMX(brgemm_convolution_bwd_strided_t<avx512_core_amx, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core_vnni, true>)
            CPU_INSTANCE_AVX512(brgemm_convolution_bwd_strided_t<avx512_core, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni_2, true>)
            CPU_INSTANCE_AVX2(brgemm_convolution_bwd_strided_t<avx2_vnni, true>)
            CPU_INSTANCE(gemm_x8s8s32x_convolution_bwd_data_t)
            CPU_INSTANCE(ref_convolution_int8_bwd_data_t)
            nullptr,
        })},
    });
    return the_map;
}
// clang-format on
} // namespace

const impl_list_item_t *get_convolution_impl_list(
        const convolution_desc_t *desc) {
    static const impl_list_item_t empty_list[] = {nullptr};

    const bool is_fwd = utils::one_of(
            desc->prop_kind, forward_training, forward_inference);
    prop_kind_t prop_kind = is_fwd ? forward : desc->prop_kind;

    pk_dt_impl_key_t key {
            prop_kind,
            conv_prop_invariant_src_d(desc)->data_type,
            conv_prop_invariant_wei_d(desc)->data_type,
            conv_prop_invariant_dst_d(desc)->data_type,
    };

    const auto impl_list_it = impl_list_map().find(key);
    return impl_list_it != impl_list_map().cend() ? impl_list_it->second.data()
                                                  : empty_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
