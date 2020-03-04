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
#include "cpu/jit_uni_batch_normalization.hpp"
#include "cpu/jit_uni_batch_normalization_s8.hpp"
#include "cpu/jit_uni_tbb_batch_normalization.hpp"
#endif // TARGET_X86_JIT
#include "cpu/ncsp_batch_normalization.hpp"
#include "cpu/nspc_batch_normalization.hpp"
#include "cpu/ref_batch_normalization.hpp"

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
        INSTANCE_avx512(jit_uni_batch_normalization_fwd_t<avx512_common>)
        INSTANCE_avx512(jit_uni_batch_normalization_bwd_t<avx512_common>)
        INSTANCE_avx2(jit_uni_batch_normalization_fwd_t<avx2>)
        INSTANCE_avx2(jit_uni_batch_normalization_bwd_t<avx2>)
        INSTANCE_sse41(jit_uni_batch_normalization_fwd_t<sse41>)
        INSTANCE_sse41(jit_uni_batch_normalization_bwd_t<sse41>)
        INSTANCE_avx512(jit_uni_tbb_batch_normalization_fwd_t<avx512_common>)
        INSTANCE_avx512(jit_uni_tbb_batch_normalization_bwd_t<avx512_common>)
        INSTANCE_avx2(jit_uni_tbb_batch_normalization_fwd_t<avx2>)
        INSTANCE_avx2(jit_uni_tbb_batch_normalization_bwd_t<avx2>)
        INSTANCE_sse41(jit_uni_tbb_batch_normalization_fwd_t<sse41>)
        INSTANCE_sse41(jit_uni_tbb_batch_normalization_bwd_t<sse41>)
        INSTANCE(ncsp_batch_normalization_fwd_t<f32>)
        INSTANCE(ncsp_batch_normalization_bwd_t<f32>)
        INSTANCE(nspc_batch_normalization_fwd_t<f32>)
        INSTANCE(nspc_batch_normalization_bwd_t<f32>)
        INSTANCE(ref_batch_normalization_fwd_t<f32>)
        INSTANCE(ref_batch_normalization_bwd_t<f32>)
        INSTANCE(ncsp_batch_normalization_fwd_t<bf16>)
        INSTANCE(ncsp_batch_normalization_bwd_t<bf16>)
        INSTANCE(nspc_batch_normalization_fwd_t<bf16>)
        INSTANCE(nspc_batch_normalization_bwd_t<bf16>)
        INSTANCE(ref_batch_normalization_fwd_t<bf16>)
        INSTANCE(ref_batch_normalization_bwd_t<bf16>)
        /* int */
        INSTANCE_avx512(jit_uni_batch_normalization_s8_fwd_t<avx512_core>)
        INSTANCE_avx2(jit_uni_batch_normalization_s8_fwd_t<avx2>)
        INSTANCE(ref_batch_normalization_fwd_t<s8>)
        // clang-format on
        /* eol */
        nullptr,
};
} // namespace

const pd_create_f *get_batch_normalization_impl_list(
        const batch_normalization_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
// vim: et ts=4 sw=4 cindent cino=+2s,^=l0,\:0,N-s
