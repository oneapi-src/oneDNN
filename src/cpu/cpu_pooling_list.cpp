/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
* Copyright 2020 FUJITSU LIMITED
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

#include "cpu/nchw_pooling.hpp"
#include "cpu/nhwc_pooling.hpp"
#include "cpu/ref_pooling.hpp"

#if DNNL_X64
#include "cpu/x64/jit_uni_i8i8_pooling.hpp"
#include "cpu/x64/jit_uni_pooling.hpp"
using namespace dnnl::impl::cpu::x64;
#elif DNNL_AARCH64
#include "cpu/aarch64/jit_uni_i8i8_pooling.hpp"
#include "cpu/aarch64/jit_uni_pooling.hpp"
using namespace dnnl::impl::cpu::aarch64;
#if DNNL_AARCH64_USE_ACL
#include "cpu/aarch64/acl_pooling.hpp"
#endif // DNNL_AARCH64_USE_ACL
#elif DNNL_RV64
#if DNNL_RISCV_USE_RVV_INTRINSICS
#include "cpu/rv64/rvv_nchw_pooling.hpp"
using namespace dnnl::impl::cpu::rv64;
#endif // DNNL_RISCV_USE_RVV_INTRINSICS
#endif

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::prop_kind;

// clang-format off
const std::map<pk_impl_key_t, std::vector<impl_list_item_t>> &impl_list_map() {
    static const std::map<pk_impl_key_t, std::vector<impl_list_item_t>> the_map = REG_POOLING_P({
        {{forward}, {
            /* fp */
            CPU_INSTANCE_X64(jit_uni_pooling_fwd_t<avx512_core_fp16, f16>)
            CPU_INSTANCE_X64(jit_uni_pooling_fwd_t<avx512_core, bf16>)
            CPU_INSTANCE_X64(jit_uni_pooling_fwd_t<avx512_core, f32>)
            CPU_INSTANCE_X64(jit_uni_pooling_fwd_t<avx2_vnni_2, bf16>)
            CPU_INSTANCE_X64(jit_uni_pooling_fwd_t<avx2_vnni_2, f16>)
            CPU_INSTANCE_X64(jit_uni_pooling_fwd_t<avx2, f32>)
            CPU_INSTANCE_X64(jit_uni_pooling_fwd_t<avx, f32>)
            CPU_INSTANCE_X64(jit_uni_pooling_fwd_t<sse41, f32>)
            CPU_INSTANCE_AARCH64(jit_uni_pooling_fwd_t<sve_512, f32>)
            CPU_INSTANCE_AARCH64_ACL(acl_pooling_fwd_t)
            CPU_INSTANCE_RV64GCV(riscv_nchw_pooling_fwd_t<f32>)
            CPU_INSTANCE(nchw_pooling_fwd_t<bf16>)
            CPU_INSTANCE(nchw_pooling_fwd_t<f32>)
            CPU_INSTANCE(nchw_pooling_fwd_t<f16>)
            CPU_INSTANCE(nhwc_pooling_fwd_t<bf16>)
            CPU_INSTANCE(nhwc_pooling_fwd_t<f32>)
            CPU_INSTANCE(nhwc_pooling_fwd_t<f16>)
            CPU_INSTANCE(ref_pooling_fwd_t<f32>)
            CPU_INSTANCE(ref_pooling_fwd_t<bf16, f32>)
            CPU_INSTANCE(ref_pooling_fwd_t<f16, f32>)
            /* int */
            CPU_INSTANCE_X64(jit_uni_i8i8_pooling_fwd_t<avx512_core>)
            CPU_INSTANCE_X64(jit_uni_i8i8_pooling_fwd_t<avx2>)
            CPU_INSTANCE_X64(jit_uni_i8i8_pooling_fwd_t<sse41>)
            CPU_INSTANCE_AARCH64(jit_uni_i8i8_pooling_fwd_t<sve_512>)
            CPU_INSTANCE(ref_pooling_fwd_t<s32>)
            CPU_INSTANCE(ref_pooling_fwd_t<s8, s32>)
            CPU_INSTANCE(ref_pooling_fwd_t<u8, s32>)
            nullptr,
        }},
        {{backward}, REG_BWD_PK({
            CPU_INSTANCE_X64(jit_uni_pooling_bwd_t<avx512_core_fp16, f16>)
            CPU_INSTANCE_X64(jit_uni_pooling_bwd_t<avx512_core, bf16>)
            CPU_INSTANCE_X64(jit_uni_pooling_bwd_t<avx512_core, f32>)
            CPU_INSTANCE_X64(jit_uni_pooling_bwd_t<avx2, f32>)
            CPU_INSTANCE_X64(jit_uni_pooling_bwd_t<avx, f32>)
            CPU_INSTANCE_X64(jit_uni_pooling_bwd_t<sse41, f32>)
            CPU_INSTANCE_AARCH64(jit_uni_pooling_bwd_t<sve_512, f32>)
            CPU_INSTANCE(nchw_pooling_bwd_t<bf16>)
            CPU_INSTANCE(nchw_pooling_bwd_t<f32>)
            CPU_INSTANCE(nchw_pooling_bwd_t<f16>)
            CPU_INSTANCE(nhwc_pooling_bwd_t<bf16>)
            CPU_INSTANCE(nhwc_pooling_bwd_t<f32>)
            CPU_INSTANCE(nhwc_pooling_bwd_t<f16>)
            CPU_INSTANCE(ref_pooling_bwd_t)
            nullptr,
        })},
    });
    return the_map;
}
// clang-format on
} // namespace

const impl_list_item_t *get_pooling_impl_list(const pooling_desc_t *desc) {
    static const impl_list_item_t empty_list[] = {nullptr};

    const bool is_fwd = utils::one_of(
            desc->prop_kind, forward_training, forward_inference);
    prop_kind_t prop_kind = is_fwd ? forward : backward;

    pk_impl_key_t key {prop_kind};

    const auto impl_list_it = impl_list_map().find(key);
    return impl_list_it != impl_list_map().cend() ? impl_list_it->second.data()
                                                  : empty_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
