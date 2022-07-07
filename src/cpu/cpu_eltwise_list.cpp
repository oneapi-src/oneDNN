/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
* Copyright 2021 FUJITSU LIMITED
* Copyright 2021-2022 Arm Ltd. and affiliates
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

#include "cpu/ref_eltwise.hpp"

#if DNNL_X64
#include "cpu/x64/jit_uni_eltwise.hpp"
#include "cpu/x64/jit_uni_eltwise_int.hpp"
using namespace dnnl::impl::cpu::x64;
#elif DNNL_AARCH64
#include "cpu/aarch64/jit_uni_eltwise.hpp"
#include "cpu/aarch64/jit_uni_eltwise_int.hpp"
#if DNNL_AARCH64_USE_ACL
#include "cpu/aarch64/acl_eltwise.hpp"
#endif // DNNL_AARCH64_USE_ACL
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
    static const std::map<pk_impl_key_t, std::vector<impl_list_item_t>> the_map = REG_ELTWISE_P({
        {{forward}, {
            CPU_INSTANCE_X64(jit_uni_eltwise_fwd_t, avx512_core, f32)
            CPU_INSTANCE_X64(jit_uni_eltwise_fwd_t, avx512_core, bf16)
            CPU_INSTANCE_X64(jit_uni_eltwise_fwd_t, avx2_vnni_2, bf16)
            CPU_INSTANCE_X64(jit_uni_eltwise_fwd_t, avx2, f32)
            CPU_INSTANCE_X64(jit_uni_eltwise_fwd_t, avx, f32)
            CPU_INSTANCE_X64(jit_uni_eltwise_fwd_t, sse41, f32)
            CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t, avx512_core, s32)
            CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t, avx512_core, s8)
            CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t, avx512_core, u8)
            CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t, avx2, s32)
            CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t, avx2, s8)
            CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t, avx2, u8)
            CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t, sse41, s32)
            CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t, sse41, s8)
            CPU_INSTANCE_X64(jit_uni_eltwise_int_fwd_t, sse41, u8)
            CPU_INSTANCE_AARCH64(jit_uni_eltwise_fwd_t, sve_512, f32)
            CPU_INSTANCE_AARCH64(jit_uni_eltwise_fwd_t, sve_256, f32)
            CPU_INSTANCE_AARCH64(jit_uni_eltwise_fwd_t, sve_128, f32)
            CPU_INSTANCE_AARCH64(jit_uni_eltwise_int_fwd_t, sve_512, s32)
            CPU_INSTANCE_AARCH64(jit_uni_eltwise_int_fwd_t, sve_512, s8)
            CPU_INSTANCE_AARCH64(jit_uni_eltwise_int_fwd_t, sve_512, u8)
            CPU_INSTANCE_AARCH64_ACL(acl_eltwise_fwd_t)
            CPU_INSTANCE(ref_eltwise_fwd_t, f32)
            CPU_INSTANCE(ref_eltwise_fwd_t, bf16)
            CPU_INSTANCE(ref_eltwise_fwd_t, s32)
            CPU_INSTANCE(ref_eltwise_fwd_t, s8)
            CPU_INSTANCE(ref_eltwise_fwd_t, u8)
            nullptr,
        }},
        {{backward}, REG_BWD_PK({
            CPU_INSTANCE_X64(jit_uni_eltwise_bwd_t, avx512_core, f32)
            CPU_INSTANCE_X64(jit_uni_eltwise_bwd_t, avx512_core, bf16)
            CPU_INSTANCE_X64(jit_uni_eltwise_bwd_t, avx2, f32)
            CPU_INSTANCE_X64(jit_uni_eltwise_bwd_t, avx, f32)
            CPU_INSTANCE_X64(jit_uni_eltwise_bwd_t, sse41, f32)
            CPU_INSTANCE_AARCH64(jit_uni_eltwise_bwd_t, sve_512, f32)
            CPU_INSTANCE_AARCH64(jit_uni_eltwise_bwd_t, sve_256, f32)
            CPU_INSTANCE_AARCH64(jit_uni_eltwise_bwd_t, sve_128, f32)
            CPU_INSTANCE(ref_eltwise_bwd_t, f32)
            CPU_INSTANCE(ref_eltwise_bwd_t, bf16)
            nullptr,
        })},
    });
    return the_map;
}
// clang-format on
} // namespace

const impl_list_item_t *get_eltwise_impl_list(const eltwise_desc_t *desc) {
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
