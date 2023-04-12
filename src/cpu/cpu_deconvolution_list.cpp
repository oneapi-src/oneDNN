/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
* Copyright 2022 FUJITSU LIMITED
* Copyright 2022 Arm Ltd. and affiliates
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

#include "cpu/cpu_engine.hpp"

#include "cpu/ref_deconvolution.hpp"

#if DNNL_X64
#include "cpu/x64/jit_avx512_core_amx_deconvolution.hpp"
#include "cpu/x64/jit_avx512_core_x8s8s32x_1x1_deconvolution.hpp"
#include "cpu/x64/jit_avx512_core_x8s8s32x_deconvolution.hpp"
#include "cpu/x64/jit_brgemm_deconv.hpp"
#include "cpu/x64/jit_uni_x8s8s32x_1x1_deconvolution.hpp"
#include "cpu/x64/jit_uni_x8s8s32x_deconvolution.hpp"
using namespace dnnl::impl::cpu::x64;
#elif DNNL_AARCH64
#include "cpu/aarch64/jit_sve_512_core_x8s8s32x_deconvolution.hpp"
#if DNNL_AARCH64_USE_ACL
#include "cpu/aarch64/acl_deconvolution.hpp"
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
const std::map<pk_impl_key_t, std::vector<impl_list_item_t>> &impl_list_map() {
    static const std::map<pk_impl_key_t, std::vector<impl_list_item_t>> the_map = REG_DECONV_P({
        {{forward}, {
            CPU_INSTANCE_AMX(brgemm_deconvolution_fwd_t<avx512_core_amx_fp16>)
            CPU_INSTANCE_AMX(brgemm_deconvolution_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AMX(jit_avx512_core_amx_deconvolution_fwd_t)
            CPU_INSTANCE_AVX512(brgemm_deconvolution_fwd_t<avx512_core_fp16>)
            CPU_INSTANCE_AVX512(brgemm_deconvolution_fwd_t<avx512_core_bf16>)
            CPU_INSTANCE_AVX512(brgemm_deconvolution_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_deconvolution_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t)
            CPU_INSTANCE_AVX512(jit_avx512_core_x8s8s32x_deconvolution_fwd_t)
            CPU_INSTANCE_AVX2(brgemm_deconvolution_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_deconvolution_fwd_t<avx2_vnni>)
            CPU_INSTANCE_AVX2(brgemm_deconvolution_fwd_t<avx2>)
            CPU_INSTANCE_AVX2(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t<avx2>)
            CPU_INSTANCE_AVX2(jit_uni_x8s8s32x_deconvolution_fwd_t<avx2>)
            CPU_INSTANCE_SSE41(jit_uni_x8s8s32x_1x1_deconvolution_fwd_t<sse41>)
            CPU_INSTANCE_SSE41(jit_uni_x8s8s32x_deconvolution_fwd_t<sse41>)
            CPU_INSTANCE_AARCH64(jit_sve_512_core_x8s8s32x_deconvolution_fwd_t)
            CPU_INSTANCE_AARCH64_ACL(acl_deconvolution_fwd_t)
            CPU_INSTANCE(ref_deconvolution_fwd_t)
            nullptr,
        }},
        {{backward_data}, REG_BWD_PK({
            CPU_INSTANCE(ref_deconvolution_bwd_data_t)
            nullptr,
        })},
        {{backward_weights}, REG_BWD_PK({
            CPU_INSTANCE(ref_deconvolution_bwd_weights_t)
            nullptr,
        })},
    });
    return the_map;
}
// clang-format on
} // namespace

const impl_list_item_t *get_deconvolution_impl_list(
        const deconvolution_desc_t *desc) {
    static const impl_list_item_t empty_list[] = {nullptr};

    const bool is_fwd = utils::one_of(
            desc->prop_kind, forward_training, forward_inference);
    prop_kind_t prop_kind = is_fwd ? forward : desc->prop_kind;

    pk_impl_key_t key {prop_kind};

    const auto impl_list_it = impl_list_map().find(key);
    return impl_list_it != impl_list_map().cend() ? impl_list_it->second.data()
                                                  : empty_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
