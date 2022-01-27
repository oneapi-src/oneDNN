/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "cpu/ref_reduction.hpp"

#if DNNL_X64
#include "cpu/x64/jit_uni_reduction.hpp"
using namespace dnnl::impl::cpu::x64;
#endif

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;

// clang-format off
constexpr impl_list_item_t impl_list[] = REG_REDUCTION_P({
    CPU_INSTANCE_X64(jit_uni_reduction_t)

    CPU_INSTANCE(ref_reduction_t<f32, f32, f32>)
    CPU_INSTANCE(ref_reduction_t<bf16, bf16, f32>)
    CPU_INSTANCE(ref_reduction_t<bf16, f32, f32>)
    CPU_INSTANCE(ref_reduction_t<s8, s8, s32>)
    CPU_INSTANCE(ref_reduction_t<s8, s32, s32>)
    CPU_INSTANCE(ref_reduction_t<s8, f32, f32>)
    CPU_INSTANCE(ref_reduction_t<u8, u8, s32>)
    CPU_INSTANCE(ref_reduction_t<u8, s32, s32>)
    CPU_INSTANCE(ref_reduction_t<u8, f32, f32>)
    /* eol */
    nullptr,
});
// clang-format on
} //namespace

const impl_list_item_t *get_reduction_impl_list(const reduction_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl
