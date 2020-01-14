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

#include "cpu/jit_uni_i8i8_pooling.hpp"
#include "cpu/jit_uni_pooling.hpp"
#include "cpu/nchw_pooling.hpp"
#include "cpu/nhwc_pooling.hpp"
#include "cpu/ref_pooling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using pd_create_f = engine_t::primitive_desc_create_f;

namespace {
using namespace dnnl::impl::data_type;

#define INSTANCE(...) &primitive_desc_t::create<__VA_ARGS__::pd_t>
static const pd_create_f impl_list[] = {
        /* fp */
        INSTANCE(jit_uni_pooling_fwd_t<avx512_core, bf16>),
        INSTANCE(jit_uni_pooling_bwd_t<avx512_core, bf16>),
        INSTANCE(jit_uni_pooling_fwd_t<avx512_common, f32>),
        INSTANCE(jit_uni_pooling_bwd_t<avx512_common, f32>),
        INSTANCE(jit_uni_pooling_fwd_t<avx, f32>),
        INSTANCE(jit_uni_pooling_bwd_t<avx, f32>),
        INSTANCE(jit_uni_pooling_fwd_t<sse41, f32>),
        INSTANCE(jit_uni_pooling_bwd_t<sse41, f32>),
        INSTANCE(nchw_pooling_fwd_t<bf16>),
        INSTANCE(nchw_pooling_bwd_t<bf16>),
        INSTANCE(nchw_pooling_fwd_t<f32>),
        INSTANCE(nchw_pooling_bwd_t<f32>),
        INSTANCE(nhwc_pooling_fwd_t<bf16>),
        INSTANCE(nhwc_pooling_bwd_t<bf16>),
        INSTANCE(nhwc_pooling_fwd_t<f32>),
        INSTANCE(nhwc_pooling_bwd_t<f32>),
        INSTANCE(ref_pooling_fwd_t<f32>),
        INSTANCE(ref_pooling_fwd_t<bf16, f32>),
        INSTANCE(ref_pooling_bwd_t<f32>),
        INSTANCE(ref_pooling_bwd_t<bf16>),
        /* int */
        INSTANCE(jit_uni_i8i8_pooling_fwd_t<avx512_core>),
        INSTANCE(jit_uni_i8i8_pooling_fwd_t<avx2>),
        INSTANCE(ref_pooling_fwd_t<s32>),
        INSTANCE(ref_pooling_fwd_t<s8, s32>),
        INSTANCE(ref_pooling_fwd_t<u8, s32>),
        INSTANCE(ref_pooling_bwd_t<s32>),
        /* eol */
        nullptr,
};
#undef INSTANCE
} // namespace

const pd_create_f *get_pooling_impl_list(const pooling_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
