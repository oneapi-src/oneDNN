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

#if TARGET_X86_JIT
#include "cpu/jit_uni_i8i8_pooling.hpp"
#include "cpu/jit_uni_pooling.hpp"
#endif // TARGET_X86_JIT
#include "cpu/nchw_pooling.hpp"
#include "cpu/nhwc_pooling.hpp"
#include "cpu/ref_pooling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using pd_create_f = engine_t::primitive_desc_create_f;

namespace {
using namespace dnnl::impl::data_type;

/// @copydoc INSTANCE_CREATOR
#define INSTANCE_CREATOR(...) DEFAULT_INSTANCE_CREATOR(__VA_ARGS__)
static const pd_create_f impl_list[] = {
        // clang-format off
        /* fp */
        INSTANCE_avx512(jit_uni_pooling_fwd_t<avx512_core, bf16>)
        INSTANCE_avx512(jit_uni_pooling_bwd_t<avx512_core, bf16>)
        INSTANCE_avx512(jit_uni_pooling_fwd_t<avx512_common, f32>)
        INSTANCE_avx512(jit_uni_pooling_bwd_t<avx512_common, f32>)
        INSTANCE_avx(jit_uni_pooling_fwd_t<avx, f32>)
        INSTANCE_avx(jit_uni_pooling_bwd_t<avx, f32>)
        INSTANCE_sse41(jit_uni_pooling_fwd_t<sse41, f32>)
        INSTANCE_sse41(jit_uni_pooling_bwd_t<sse41, f32>)
        INSTANCE(nchw_pooling_fwd_t<bf16>)
        INSTANCE(nchw_pooling_bwd_t<bf16>)
        INSTANCE(nchw_pooling_fwd_t<f32>)
        INSTANCE(nchw_pooling_bwd_t<f32>)
        INSTANCE(nhwc_pooling_fwd_t<bf16>)
        INSTANCE(nhwc_pooling_bwd_t<bf16>)
        INSTANCE(nhwc_pooling_fwd_t<f32>)
        INSTANCE(nhwc_pooling_bwd_t<f32>)
        INSTANCE(ref_pooling_fwd_t<f32>)
        INSTANCE(ref_pooling_fwd_t<bf16, f32>)
        INSTANCE(ref_pooling_bwd_t<f32>)
        INSTANCE(ref_pooling_bwd_t<bf16>)
        /* int */
        INSTANCE_avx512(jit_uni_i8i8_pooling_fwd_t<avx512_core>)
        INSTANCE_avx2(jit_uni_i8i8_pooling_fwd_t<avx2>)
        INSTANCE(ref_pooling_fwd_t<s32>)
        INSTANCE(ref_pooling_fwd_t<s8, s32>)
        INSTANCE(ref_pooling_fwd_t<u8, s32>)
        INSTANCE(ref_pooling_bwd_t<s32>)
        // clang-format on
        /* eol */
        nullptr,
};
} // namespace

const pd_create_f *get_pooling_impl_list(const pooling_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
