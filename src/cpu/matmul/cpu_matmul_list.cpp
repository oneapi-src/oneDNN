/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "cpu/matmul/gemm_bf16_matmul.hpp"
#include "cpu/matmul/gemm_f32_matmul.hpp"
#include "cpu/matmul/gemm_x8s8s32x_matmul.hpp"
#include "cpu/matmul/ref_matmul.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using pd_create_f = engine_t::primitive_desc_create_f;

namespace {
using namespace dnnl::impl::data_type;

/// @copydoc INSTANCE_CREATOR
#define INSTANCE_CREATOR(...) DEFAULT_INSTANCE_CREATOR(__VA_ARGS__)
// don't really know yet about which ones require jit.
// Can run vanilla tests with extra verbosity and see which ones get skipped? XXX
static const pd_create_f impl_list[] = {
        // clang-format off
        INSTANCE(matmul::gemm_f32_matmul_t)
        INSTANCE(matmul::gemm_bf16_matmul_t<f32>)
        INSTANCE(matmul::gemm_bf16_matmul_t<bf16>)
        INSTANCE(matmul::gemm_x8s8s32x_matmul_t<s8, s8, f32>)
        INSTANCE(matmul::gemm_x8s8s32x_matmul_t<s8, s8, s32>)
        INSTANCE(matmul::gemm_x8s8s32x_matmul_t<s8, s8, s8>)
        INSTANCE(matmul::gemm_x8s8s32x_matmul_t<s8, s8, u8>)
        INSTANCE(matmul::gemm_x8s8s32x_matmul_t<u8, s8, f32>)
        INSTANCE(matmul::gemm_x8s8s32x_matmul_t<u8, s8, s32>)
        INSTANCE(matmul::gemm_x8s8s32x_matmul_t<u8, s8, s8>)
        INSTANCE(matmul::gemm_x8s8s32x_matmul_t<u8, s8, u8>)
        INSTANCE(matmul::ref_matmul_t<f32>)
        // I think following 2 may require sse41 XXX
        INSTANCE(matmul::ref_matmul_t<bf16, bf16, f32, f32>)
        INSTANCE(matmul::ref_matmul_t<bf16, bf16, bf16, f32>)
        INSTANCE(matmul::ref_matmul_t<s8, s8, f32, s32>)
        INSTANCE(matmul::ref_matmul_t<s8, s8, s32, s32>)
        INSTANCE(matmul::ref_matmul_t<s8, s8, s8, s32>)
        INSTANCE(matmul::ref_matmul_t<s8, s8, u8, s32>)
        INSTANCE(matmul::ref_matmul_t<u8, s8, f32, s32>)
        INSTANCE(matmul::ref_matmul_t<u8, s8, s32, s32>)
        INSTANCE(matmul::ref_matmul_t<u8, s8, s8, s32>)
        INSTANCE(matmul::ref_matmul_t<u8, s8, u8, s32>)
        // clang-format off
        /* eol */
        nullptr,
};
} // namespace

const pd_create_f *get_matmul_impl_list(const matmul_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
