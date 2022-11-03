/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
* Copyright 2022 FUJITSU LIMITED
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
#include "cpu/reorder/cpu_reorder.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// clang-format off

const impl_list_map_t &regular_s8_impl_list_map() {
    static const impl_list_map_t the_map = REG_REORDER_P({
        // s8 ->
        {{s8, data_type::undef, 0}, {
            // TODO: move it down when checks for sparse md are implemented in other implementations.
            DNNL_X64_ONLY(REG_SPARSE_SR(s8, oi, s8, OI16i64o4i, sparse_inputs_order::keep, sparse_spec::reference))
            DNNL_X64_ONLY(REG_SPARSE_SR(s8, io, s8, OI16i64o4i, sparse_inputs_order::keep, sparse_spec::reference))

            CPU_REORDER_INSTANCE(rnn_weights_reorder_s8_t, s8)
            CPU_REORDER_INSTANCE(rnn_brgemm_weights_reorder_s8_t, s8, s8)
            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::brgemm_matmul_matrix_B_reorder_t))

            REG_FAST_DIRECT_COPY(s8, f32)
            REG_FAST_DIRECT_COPY(s8, s32)
            REG_FAST_DIRECT_COPY(s8, bf16)
            REG_FAST_DIRECT_COPY(s8, f16)
            REG_FAST_DIRECT_COPY(s8, s8)
            REG_FAST_DIRECT_COPY(s8, u8)

            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64_jit_blk_reorder_t))
            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64_jit_uni_reorder_t))

            DNNL_AARCH64_ONLY(CPU_REORDER_INSTANCE(aarch64_jit_blk_reorder_t))
            DNNL_AARCH64_ONLY(CPU_REORDER_INSTANCE(aarch64_jit_uni_reorder_t))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(s8, any, f32, nChw16c))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(s8, any, s32, nChw16c))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(s8, any, bf16, nChw16c))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(s8, any, s8, nChw16c))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(s8, any, u8, nChw16c))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(s8, any, f32, OIhw4i16o4i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(s8, any, bf16, OIhw4i16o4i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(s8, any, s8, OIhw4i16o4i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(s8, any, f32, gOIhw4i16o4i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(s8, any, bf16, gOIhw4i16o4i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(s8, any, s8, gOIhw4i16o4i))

            REG_SR(s8, any, f32, any, fmt_order_any, spec_reference)
            REG_SR(s8, any, s32, any, fmt_order_any, spec_reference)
            REG_SR(s8, any, bf16, any, fmt_order_any, spec_reference)
            REG_SR(s8, any, f16, any, fmt_order_any, spec_reference)
            REG_SR(s8, any, s8, any, fmt_order_any, spec_reference)
            REG_SR(s8, any, u8, any, fmt_order_any, spec_reference)

            nullptr,
        }},
    });
    return the_map;
}

// clang-format on

} // namespace cpu
} // namespace impl
} // namespace dnnl
