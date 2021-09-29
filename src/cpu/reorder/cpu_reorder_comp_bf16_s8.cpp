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

#include "cpu/reorder/cpu_reorder.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// clang-format off

const impl_list_map_t comp_bf16_s8_impl_list_map {
    // bf16 -> s8
    {{bf16, s8, 2}, {
        REG_REORDER_P(DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_uni_reorder_t)))
        REG_REORDER_P(REG_SR(bf16, ab, s8, BA16a16b4a, fmt_order::keep, spec::conv_req_comp))
        REG_REORDER_P(REG_SR(bf16, ab, s8, BA16a32b4a, fmt_order::keep, spec::conv_req_comp))
        REG_REORDER_P(REG_SR(bf16, ab, s8, BA16a48b4a, fmt_order::keep, spec::conv_req_comp))
        REG_REORDER_P(REG_SR(bf16, ab, s8, BA16a64b4a, fmt_order::keep, spec::conv_req_comp))
        REG_REORDER_P(REG_SR(bf16, ba, s8, BA16a16b4a, fmt_order::keep, spec::conv_req_comp))
        REG_REORDER_P(REG_SR(bf16, ba, s8, BA16a32b4a, fmt_order::keep, spec::conv_req_comp))
        REG_REORDER_P(REG_SR(bf16, ba, s8, BA16a48b4a, fmt_order::keep, spec::conv_req_comp))
        REG_REORDER_P(REG_SR(bf16, ba, s8, BA16a64b4a, fmt_order::keep, spec::conv_req_comp))

        nullptr,
    }},
    // bf16 -> s8
    {{bf16, s8, 3}, {
        REG_REORDER_P(DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_uni_reorder_t)))

        nullptr,
    }},
    {{bf16, s8, 4}, {
        REG_REORDER_P(DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_uni_reorder_t)))

        nullptr,
    }},
    {{bf16, s8, 5}, {
        REG_REORDER_P(DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_uni_reorder_t)))

        nullptr,
    }},
    {{bf16, s8, 6}, {
        REG_REORDER_P(DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_uni_reorder_t)))

        nullptr,
    }},
};

// clang-format on

} // namespace cpu
} // namespace impl
} // namespace dnnl
