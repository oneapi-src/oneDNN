/*******************************************************************************
* Copyright 2020 Intel Corporation
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

const impl_list_map_t regular_s8_impl_list_map {
    // s8 ->
    {{s8, data_type::undef, 0}, {
        rnn_weights_reorder_s8_t<s8>::pd_t::create,
        rnn_brgemm_weights_reorder_s8_t<s8, s8>::pd_t::create,

        REG_FAST_DIRECT_COPY_COMMA(s8, f32)
        REG_FAST_DIRECT_COPY_COMMA(s8, s32)
        REG_FAST_DIRECT_COPY_COMMA(s8, bf16)
        REG_FAST_DIRECT_COPY_COMMA(s8, s8)
        REG_FAST_DIRECT_COPY_COMMA(s8, u8)

        DNNL_X64_ONLY(x64::jit_uni_reorder_create,)
        DNNL_AARCH64_ONLY(aarch64::jit_uni_reorder_create,)

        REG_SR_BIDIR(s8, any, f32, nChw16c),
        REG_SR_BIDIR(s8, any, s32, nChw16c),
        REG_SR_BIDIR(s8, any, bf16, nChw16c),
        REG_SR_BIDIR(s8, any, s8, nChw16c),
        REG_SR_BIDIR(s8, any, u8, nChw16c),

        REG_SR_BIDIR(s8, any, f32, OIhw4i16o4i),
        REG_SR_BIDIR(s8, any, bf16, OIhw4i16o4i),
        REG_SR_BIDIR(s8, any, s8, OIhw4i16o4i),
        REG_SR_BIDIR(s8, any, f32, gOIhw4i16o4i),
        REG_SR_BIDIR(s8, any, bf16, gOIhw4i16o4i),
        REG_SR_BIDIR(s8, any, s8, gOIhw4i16o4i),

        REG_SR(s8, any, f32, any, fmt_order::any, spec::reference),
        REG_SR(s8, any, s32, any, fmt_order::any, spec::reference),
        REG_SR(s8, any, bf16, any, fmt_order::any, spec::reference),
        REG_SR(s8, any, s8, any, fmt_order::any, spec::reference),
        REG_SR(s8, any, u8, any, fmt_order::any, spec::reference),

        nullptr,
    }},
};

// clang-format on

} // namespace cpu
} // namespace impl
} // namespace dnnl
