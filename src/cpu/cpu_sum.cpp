/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
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

#include "common/impl_list_item.hpp"

#include "cpu/cpu_engine.hpp"

#include "cpu/ref_sum.hpp"
#include "cpu/simple_sum.hpp"

#if DNNL_X64
#include "cpu/x64/jit_uni_xf16_sum.hpp"
using namespace dnnl::impl::cpu::x64;
#endif

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;
#define INSTANCE(...) \
    impl_list_item_t(impl_list_item_t::sum_type_deduction_helper_t< \
            __VA_ARGS__::pd_t>()),
#define SUM_INSTANCE_AVX512(...) REG_AVX512_ISA(INSTANCE(__VA_ARGS__))
#define SUM_INSTANCE_AVX2(...) REG_AVX2_ISA(INSTANCE(__VA_ARGS__))
// clang-format off
constexpr impl_list_item_t cpu_sum_impl_list[] = REG_SUM_P({
        SUM_INSTANCE_AVX512(jit_xf16_sum_t<bf16, bf16, avx512_core>)
        SUM_INSTANCE_AVX512(jit_xf16_sum_t<bf16, f32, avx512_core>)
        SUM_INSTANCE_AVX2(jit_xf16_sum_t<bf16, bf16, avx2_vnni_2>)
        SUM_INSTANCE_AVX2(jit_xf16_sum_t<bf16, f32, avx2_vnni_2>)
        SUM_INSTANCE_AVX2(jit_xf16_sum_t<f16, f16, avx2_vnni_2>)
        SUM_INSTANCE_AVX2(jit_xf16_sum_t<f16, f32, avx2_vnni_2>)
        INSTANCE(simple_sum_t<f16>)
        INSTANCE(simple_sum_t<f16, f32>)
        INSTANCE(simple_sum_t<bf16>)
        INSTANCE(simple_sum_t<bf16, f32>)
        INSTANCE(simple_sum_t<f32>)
        INSTANCE(ref_sum_t)
        nullptr,
});
// clang-format on
#undef INSTANCE_X64
#undef INSTANCE
} // namespace

const impl_list_item_t *cpu_engine_impl_list_t::get_sum_implementation_list() {
    return cpu_sum_impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
