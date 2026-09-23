/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2024 Arm Ltd. and affiliates
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
const impl_list_map_t &regular_bf16_impl_list_map() {
    static const impl_list_map_t the_map = REG_REORDER_P({
        // bf16 ->
        {{bf16, data_type::undef, 0}, {
            CPU_INSTANCE(rnn_weights_reorder_t<bf16, bf16>)
            CPU_INSTANCE_X64_ZEN(x64::zen::reorder::zen_reorder_t)
            CPU_INSTANCE_X64(x64::brgemm_matmul_copy_reorder_t)
            CPU_INSTANCE_X64(x64::jit_uni_reorder_direct_copy_t)
            CPU_INSTANCE_X64(x64::jit_blk_reorder_t)
            CPU_INSTANCE_X64(x64::jit_uni_reorder_t)

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(bf16, any, f32, nChw16c))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(bf16, any, f32, nCdhw16c))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(bf16, any, s8, nChw16c))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(bf16, any, s8, nCdhw16c))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(bf16, any, u8, nChw16c))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(bf16, any, u8, nCdhw16c))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(bf16, any, bf16, nChw16c))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(bf16, any, bf16, nCdhw16c))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(bf16, any, f32, OIdhw16o16i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(bf16, any, f32, OIdhw16i16o))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(bf16, any, s8, OIdhw16o16i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(bf16, any, s8, OIdhw16i16o))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(bf16, any, u8, OIdhw16o16i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(bf16, any, u8, OIdhw16i16o))

            CPU_INSTANCE_AARCH64(aarch64::jit_uni_reorder_t)
            CPU_INSTANCE_RV64(rv64::jit_uni_reorder_t)

            CPU_INSTANCE(ref_reorder_t)

            nullptr,
        }},
    });
    return the_map;
}

// clang-format on

} // namespace cpu
} // namespace impl
} // namespace dnnl
