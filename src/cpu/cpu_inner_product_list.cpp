/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include "cpu/gemm_inner_product.hpp"
#include "cpu/gemm_x8s8s32x_inner_product.hpp"
#include "cpu/ref_inner_product.hpp"
#include "cpu/ref_inner_product_int8.hpp"

#if DNNL_X64
#include "cpu/x64/gemm_bf16_inner_product.hpp"
#include "cpu/x64/jit_brgemm_inner_product.hpp"
using namespace dnnl::impl::cpu::x64;
#endif

#if DNNL_AARCH64 && DNNL_AARCH64_USE_ACL
#include "cpu/aarch64/acl_inner_product.hpp"
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
    static const std::map<pk_dt_impl_key_t, std::vector<impl_list_item_t>> the_map = REG_IP_P({
        {{forward, f32, f32, f32}, {
            CPU_INSTANCE_AMX(brgemm_inner_product_fwd_t<avx512_core_amx>) // bf32
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2>)
            CPU_INSTANCE_AARCH64_ACL(acl_inner_product_fwd_t)
            CPU_INSTANCE(gemm_inner_product_fwd_t<f32>)
            CPU_INSTANCE(ref_inner_product_fwd_t)
            nullptr,
        }},
        {{forward, bf16, bf16, f32}, {
            CPU_INSTANCE_AMX(brgemm_inner_product_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core_bf16>)
            CPU_INSTANCE_AVX512(gemm_bf16_inner_product_fwd_t<f32>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE(ref_inner_product_fwd_t)
            nullptr,
        }},
        {{forward, bf16, bf16, bf16}, {
            CPU_INSTANCE_AMX(brgemm_inner_product_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core_bf16>)
            CPU_INSTANCE_AVX512(gemm_bf16_inner_product_fwd_t<bf16>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE(ref_inner_product_fwd_t)
            nullptr,
        }},
        {{forward, f16, f16, f32}, {
            CPU_INSTANCE_AMX(brgemm_inner_product_fwd_t<avx512_core_amx_fp16>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core_fp16>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE(ref_inner_product_fwd_t)
            nullptr,
        }},
        {{forward, f16, f16, f16}, {
            CPU_INSTANCE_AMX(brgemm_inner_product_fwd_t<avx512_core_amx_fp16>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core_fp16>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AARCH64_ACL(acl_inner_product_fwd_t)
            CPU_INSTANCE(ref_inner_product_fwd_t)
            nullptr,
        }},
        {{backward_data, f32, f32, f32}, REG_BWD_PK({
            CPU_INSTANCE_AMX(brgemm_inner_product_bwd_data_t<avx512_core_amx>) // bf32
            CPU_INSTANCE_AVX512(brgemm_inner_product_bwd_data_t<avx512_core>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_bwd_data_t<avx2>)
            CPU_INSTANCE(gemm_inner_product_bwd_data_t<f32>)
            CPU_INSTANCE(ref_inner_product_bwd_data_t)
            nullptr,
        })},
        {{backward_data, f32, bf16, bf16}, REG_BWD_PK({
            CPU_INSTANCE_AMX(brgemm_inner_product_bwd_data_t<avx512_core_amx>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_bwd_data_t<avx512_core_bf16>)
            CPU_INSTANCE_AVX512(gemm_bf16_inner_product_bwd_data_t<f32>)
            CPU_INSTANCE(ref_inner_product_bwd_data_t)
            nullptr,
        })},
        {{backward_data, bf16, bf16, bf16}, REG_BWD_PK({
            CPU_INSTANCE_AMX(brgemm_inner_product_bwd_data_t<avx512_core_amx>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_bwd_data_t<avx512_core_bf16>)
            CPU_INSTANCE_AVX512(gemm_bf16_inner_product_bwd_data_t<bf16>)
            CPU_INSTANCE(ref_inner_product_bwd_data_t)
            nullptr,
        })},
        {{backward_data, f32, f16, f16}, REG_BWD_PK({
            CPU_INSTANCE_AMX(brgemm_inner_product_bwd_data_t<avx512_core_amx_fp16>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_bwd_data_t<avx512_core_fp16>)
            CPU_INSTANCE(ref_inner_product_bwd_data_t)
            nullptr,
        })},
        {{backward_data, f16, f16, f16}, REG_BWD_PK({
            CPU_INSTANCE_AMX(brgemm_inner_product_bwd_data_t<avx512_core_amx_fp16>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_bwd_data_t<avx512_core_fp16>)
            CPU_INSTANCE(ref_inner_product_bwd_data_t)
            nullptr,
        })},
        {{backward_weights, f32, f32, f32}, REG_BWD_PK({
            CPU_INSTANCE_AMX(brgemm_inner_product_bwd_weights_t<avx512_core_amx>) // bf32
            CPU_INSTANCE_AVX512(brgemm_inner_product_bwd_weights_t<avx512_core>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_bwd_weights_t<avx2>)
            CPU_INSTANCE(gemm_inner_product_bwd_weights_t<f32>)
            CPU_INSTANCE(ref_inner_product_bwd_weights_t)
            nullptr,
        })},
        {{backward_weights, bf16, f32, bf16}, REG_BWD_PK({
            CPU_INSTANCE_AMX(brgemm_inner_product_bwd_weights_t<avx512_core_amx>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_bwd_weights_t<avx512_core_bf16>)
            CPU_INSTANCE_AVX512(gemm_bf16_inner_product_bwd_weights_t<f32>)
            CPU_INSTANCE(ref_inner_product_bwd_weights_t)
            nullptr,
        })},
        {{backward_weights, bf16, bf16, bf16}, REG_BWD_PK({
            CPU_INSTANCE_AMX(brgemm_inner_product_bwd_weights_t<avx512_core_amx>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_bwd_weights_t<avx512_core_bf16>)
            CPU_INSTANCE_AVX512(gemm_bf16_inner_product_bwd_weights_t<bf16>)
            CPU_INSTANCE(ref_inner_product_bwd_weights_t)
            nullptr,
        })},
        {{backward_weights, f16, f32, f16}, REG_BWD_PK({
            CPU_INSTANCE_AMX(brgemm_inner_product_bwd_weights_t<avx512_core_amx_fp16>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_bwd_weights_t<avx512_core_fp16>)
            CPU_INSTANCE(ref_inner_product_bwd_weights_t)
            nullptr,
        })},
        {{backward_weights, f16, f16, f16}, REG_BWD_PK({
            CPU_INSTANCE_AMX(brgemm_inner_product_bwd_weights_t<avx512_core_amx_fp16>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_bwd_weights_t<avx512_core_fp16>)
            CPU_INSTANCE(ref_inner_product_bwd_weights_t)
            nullptr,
        })},
        {{forward, s8, s8, f32}, {
            CPU_INSTANCE_AMX(brgemm_inner_product_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni>)
            CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t)
            CPU_INSTANCE(ref_inner_product_int8_fwd_t)
            nullptr,
        }},
        {{forward, s8, s8, s32}, {
            CPU_INSTANCE_AMX(brgemm_inner_product_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni>)
            CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t)
            CPU_INSTANCE(ref_inner_product_int8_fwd_t)
            nullptr,
        }},
        {{forward, s8, s8, s8}, {
            CPU_INSTANCE_AMX(brgemm_inner_product_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni>)
            CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t)
            CPU_INSTANCE(ref_inner_product_int8_fwd_t)
            nullptr,
        }},
        {{forward, s8, s8, u8}, {
            CPU_INSTANCE_AMX(brgemm_inner_product_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni>)
            CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t)
            CPU_INSTANCE(ref_inner_product_int8_fwd_t)
            nullptr,
        }},
        {{forward, u8, s8, f32}, {
            CPU_INSTANCE_AMX(brgemm_inner_product_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni>)
            CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t)
            CPU_INSTANCE(ref_inner_product_int8_fwd_t)
            nullptr,
        }},
        {{forward, u8, s8, s32}, {
            CPU_INSTANCE_AMX(brgemm_inner_product_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni>)
            CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t)
            CPU_INSTANCE(ref_inner_product_int8_fwd_t)
            nullptr,
        }},
        {{forward, u8, s8, s8}, {
            CPU_INSTANCE_AMX(brgemm_inner_product_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni>)
            CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t)
            CPU_INSTANCE(ref_inner_product_int8_fwd_t)
            nullptr,
        }},
        {{forward, u8, s8, u8}, {
            CPU_INSTANCE_AMX(brgemm_inner_product_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni>)
            CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t)
            CPU_INSTANCE(ref_inner_product_int8_fwd_t)
            nullptr,
        }},
        {{forward, s8, s8, bf16}, {
            CPU_INSTANCE_AMX(brgemm_inner_product_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE(ref_inner_product_int8_fwd_t)
            nullptr,
        }},
        {{forward, u8, s8, bf16}, {
            CPU_INSTANCE_AMX(brgemm_inner_product_fwd_t<avx512_core_amx>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core_vnni>)
            CPU_INSTANCE_AVX512(brgemm_inner_product_fwd_t<avx512_core>)
            CPU_INSTANCE_AVX2(brgemm_inner_product_fwd_t<avx2_vnni_2>)
            CPU_INSTANCE(ref_inner_product_int8_fwd_t)
            nullptr,
        }},
    });
    return the_map;
}
// clang-format on
} // namespace

const impl_list_item_t *get_inner_product_impl_list(
        const inner_product_desc_t *desc) {
    static const impl_list_item_t empty_list[] = {nullptr};

    const bool is_fwd = utils::one_of(
            desc->prop_kind, forward_training, forward_inference);
    prop_kind_t prop_kind = is_fwd ? forward : desc->prop_kind;

    const memory_desc_t *src_md = desc->prop_kind == backward_data
            ? &desc->diff_src_desc
            : &desc->src_desc;
    const memory_desc_t *wei_md = desc->prop_kind == backward_weights
            ? &desc->diff_weights_desc
            : &desc->weights_desc;
    const memory_desc_t *dst_md
            = is_fwd ? &desc->dst_desc : &desc->diff_dst_desc;
    pk_dt_impl_key_t key {
            prop_kind,
            src_md->data_type,
            wei_md->data_type,
            dst_md->data_type,
    };

    const auto impl_list_it = impl_list_map().find(key);
    return impl_list_it != impl_list_map().cend() ? impl_list_it->second.data()
                                                  : empty_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
