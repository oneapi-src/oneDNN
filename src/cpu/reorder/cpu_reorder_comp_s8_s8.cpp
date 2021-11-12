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

const impl_list_map_t comp_s8_s8_impl_list_map REG_REORDER_P({
    // s8 -> s8
    {{s8, s8, 2}, {
        DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_uni_reorder_t))
        REG_SR(s8, ab, s8, BA16a16b4a, fmt_order::keep, spec::conv_req_comp)
        REG_SR(s8, ab, s8, BA16a32b4a, fmt_order::keep, spec::conv_req_comp)
        REG_SR(s8, ab, s8, BA16a48b4a, fmt_order::keep, spec::conv_req_comp)
        REG_SR(s8, ab, s8, BA16a64b4a, fmt_order::keep, spec::conv_req_comp)
        REG_SR(s8, ba, s8, BA16a16b4a, fmt_order::keep, spec::conv_req_comp)
        REG_SR(s8, ba, s8, BA16a32b4a, fmt_order::keep, spec::conv_req_comp)
        REG_SR(s8, ba, s8, BA16a48b4a, fmt_order::keep, spec::conv_req_comp)
        REG_SR(s8, ba, s8, BA16a64b4a, fmt_order::keep, spec::conv_req_comp)

        nullptr,
    }},
    // s8 -> s8
    {{s8, s8, 3}, {
        DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_uni_reorder_t))
        REG_REORDER_P(REG_SR(s8, any, s8, wio, fmt_order::keep, spec::conv_req_comp)) // required for non-x64 only

        nullptr,
    }},
    {{s8, s8, 4}, {
        DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_uni_reorder_t))
        REG_REORDER_P(REG_SR(s8, any, s8, hwio, fmt_order::keep, spec::conv_req_comp)) // required for non-x64 only
        REG_REORDER_P(REG_SR(s8, any, s8, wigo, fmt_order::keep, spec::conv_req_comp)) // required for non-x64 only
        REG_REORDER_P(REG_SR(s8, goiw, s8, Goiw16g, fmt_order::keep, spec::conv_req_comp)) // required for non-x64 only
        REG_REORDER_P(REG_SR(s8, goiw, s8, Goiw8g, fmt_order::keep, spec::conv_req_comp)) // required for non-x64 only
        REG_REORDER_P(REG_SR(s8, goiw, s8, Goiw4g, fmt_order::keep, spec::conv_req_comp)) // required for non-x64 only

        nullptr,
    }},
    {{s8, s8, 5}, {
        DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_uni_reorder_t))
        REG_REORDER_P(REG_SR(s8, any, s8, hwigo, fmt_order::keep, spec::conv_req_comp)) // required for non-x64 only
        REG_REORDER_P(REG_SR(s8, any, s8, dhwio, fmt_order::keep, spec::conv_req_comp)) // required for non-x64 only
        REG_REORDER_P(REG_SR(s8, goihw, s8, Goihw16g, fmt_order::keep, spec::conv_req_comp)) // required for non-x64 only
        REG_REORDER_P(REG_SR(s8, goihw, s8, Goihw8g, fmt_order::keep, spec::conv_req_comp)) // required for non-x64 only
        REG_REORDER_P(REG_SR(s8, goihw, s8, Goihw4g, fmt_order::keep, spec::conv_req_comp)) // required for non-x64 only

        nullptr,
    }},
    {{s8, s8, 6}, {
        DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_uni_reorder_t))
        REG_REORDER_P(REG_SR(s8, any, s8, dhwigo, fmt_order::keep, spec::conv_req_comp)) // required for non-x64 only

        nullptr,
    }},
});

// clang-format on

} // namespace cpu
} // namespace impl
} // namespace dnnl
