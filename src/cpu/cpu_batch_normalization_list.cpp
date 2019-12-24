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

#include "cpu/jit_uni_batch_normalization.hpp"
#include "cpu/jit_uni_batch_normalization_s8.hpp"
#include "cpu/jit_uni_tbb_batch_normalization.hpp"
#include "cpu/ncsp_batch_normalization.hpp"
#include "cpu/nspc_batch_normalization.hpp"
#include "cpu/ref_batch_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using pd_create_f = engine_t::primitive_desc_create_f;

namespace {
using namespace dnnl::impl::data_type;

#define INSTANCE(...) &primitive_desc_t::create<__VA_ARGS__::pd_t>
static const pd_create_f impl_list[] = {
        /* fp */
        INSTANCE(jit_uni_batch_normalization_fwd_t<avx512_common>),
        INSTANCE(jit_uni_batch_normalization_bwd_t<avx512_common>),
        INSTANCE(jit_uni_batch_normalization_fwd_t<avx2>),
        INSTANCE(jit_uni_batch_normalization_bwd_t<avx2>),
        INSTANCE(jit_uni_batch_normalization_fwd_t<sse41>),
        INSTANCE(jit_uni_batch_normalization_bwd_t<sse41>),
        INSTANCE(jit_uni_tbb_batch_normalization_fwd_t<avx512_common>),
        INSTANCE(jit_uni_tbb_batch_normalization_bwd_t<avx512_common>),
        INSTANCE(jit_uni_tbb_batch_normalization_fwd_t<avx2>),
        INSTANCE(jit_uni_tbb_batch_normalization_bwd_t<avx2>),
        INSTANCE(jit_uni_tbb_batch_normalization_fwd_t<sse41>),
        INSTANCE(jit_uni_tbb_batch_normalization_bwd_t<sse41>),
        INSTANCE(ncsp_batch_normalization_fwd_t<f32>),
        INSTANCE(ncsp_batch_normalization_bwd_t<f32>),
        INSTANCE(ncsp_batch_normalization_fwd_t<bf16>),
        INSTANCE(ncsp_batch_normalization_bwd_t<bf16>),
        INSTANCE(nspc_batch_normalization_fwd_t<f32>),
        INSTANCE(nspc_batch_normalization_bwd_t<f32>),
        INSTANCE(nspc_batch_normalization_fwd_t<bf16>),
        INSTANCE(nspc_batch_normalization_bwd_t<bf16>),
        INSTANCE(ref_batch_normalization_fwd_t<f32>),
        INSTANCE(ref_batch_normalization_bwd_t<f32>),
        INSTANCE(ref_batch_normalization_fwd_t<bf16>),
        INSTANCE(ref_batch_normalization_bwd_t<bf16>),
        /* int */
        INSTANCE(jit_uni_batch_normalization_s8_fwd_t<avx512_core>),
        INSTANCE(jit_uni_batch_normalization_s8_fwd_t<avx2>),
        INSTANCE(ref_batch_normalization_fwd_t<s8>),
        /* eol */
        nullptr,
};
#undef INSTANCE
} // namespace

const pd_create_f *get_batch_normalization_impl_list(
        const batch_normalization_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
