/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include "cpu/reorder/cpu_reorder.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// clang-format off

const impl_list_map_t &regular_f32_f32_impl_list_map() {
    static const impl_list_map_t the_map = REG_REORDER_P({
        // f32 -> f32
        {{f32, f32, 0}, {
            REG_FAST_DIRECT_COPY_F32_F32

            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::brgemm_matmul_matrix_B_reorder_t))
            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_blk_reorder_t))
            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_uni_reorder_t))

            DNNL_AARCH64_ONLY(CPU_REORDER_INSTANCE(aarch64::jit_blk_reorder_t))
            DNNL_AARCH64_ONLY(CPU_REORDER_INSTANCE(aarch64::jit_uni_reorder_t))
            REG_SR(f32, any, f32, any, fmt_order::any, spec::reference)

            nullptr,
        }},
        {{f32, f32, 3}, {
            REG_FAST_DIRECT_COPY_F32_F32

            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::brgemm_matmul_matrix_B_reorder_t))
            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_blk_reorder_t))
            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_uni_reorder_t))

            DNNL_AARCH64_ONLY(CPU_REORDER_INSTANCE(aarch64::jit_blk_reorder_t))
            DNNL_AARCH64_ONLY(CPU_REORDER_INSTANCE(aarch64::jit_uni_reorder_t))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, nCw16c))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, nCw8c))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, nCw4c))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, nCw4c, f32, nCw16c))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, nCw8c, f32, nCw16c))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIw4i4o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIw4o4i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIw8i8o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIw8o8i))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIw16o16i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIw16i16o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, IOw16o16i))

            REG_SR(f32, any, f32, any, fmt_order::any, spec::reference)

            nullptr,
        }},
        {{f32, f32, 4}, {
            CPU_REORDER_INSTANCE(rnn_weights_reorder_t<f32, f32>)

            REG_FAST_DIRECT_COPY_F32_F32

            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_blk_reorder_t))
            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_uni_reorder_t))

            DNNL_AARCH64_ONLY(CPU_REORDER_INSTANCE(aarch64::jit_blk_reorder_t))
            DNNL_AARCH64_ONLY(CPU_REORDER_INSTANCE(aarch64::jit_uni_reorder_t))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, nChw16c))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, nChw8c))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, nChw4c))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, nChw4c, f32, nChw16c))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, nChw8c, f32, nChw16c))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIw4i4o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIw4o4i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIw8i8o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIw8o8i))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIw16o16i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIw16i16o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gIOw16o16i))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIhw4i4o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIhw4o4i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, Ohwi8o))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIhw8i8o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIhw8o8i))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, Oihw4o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, Oihw16o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, Ohwi4o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, Ohwi16o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIhw16o16i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIhw16i16o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, IOhw16o16i))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIhw4i16o4i))

            REG_SR(f32, any, f32, any, fmt_order::any, spec::reference)

            nullptr,
        }},
        {{f32, f32, 5}, {
            CPU_REORDER_INSTANCE(rnn_weights_reorder_t<f32, f32>)

            REG_FAST_DIRECT_COPY_F32_F32

            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_blk_reorder_t))
            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_uni_reorder_t))

            DNNL_AARCH64_ONLY(CPU_REORDER_INSTANCE(aarch64::jit_blk_reorder_t))
            DNNL_AARCH64_ONLY(CPU_REORDER_INSTANCE(aarch64::jit_uni_reorder_t))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, nCdhw16c))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, nCdhw8c))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, nCdhw4c))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, nCdhw4c, f32, nCdhw16c))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, nCdhw8c, f32, nCdhw16c))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIhw4i4o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIhw4o4i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOhwi8o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIhw8i8o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIhw8o8i))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOihw4o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOihw16o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOhwi4o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOhwi16o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIhw16o16i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIhw16i16o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gIOhw16o16i))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIdhw4i4o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIdhw4o4i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, Odhwi8o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIdhw8i8o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIdhw8o8i))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, Oidhw4o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, Oidhw16o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, Odhwi16o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIdhw16o16i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, OIdhw16i16o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, IOdhw16o16i))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIhw4i16o4i))

            REG_SR(f32, any, f32, any, fmt_order::any, spec::reference)

            nullptr,
        }},
        {{f32, f32, 6}, {
            REG_FAST_DIRECT_COPY_F32_F32

            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_blk_reorder_t))
            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_uni_reorder_t))

            DNNL_AARCH64_ONLY(CPU_REORDER_INSTANCE(aarch64::jit_blk_reorder_t))
            DNNL_AARCH64_ONLY(CPU_REORDER_INSTANCE(aarch64::jit_uni_reorder_t))


            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIdhw4i4o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIdhw4o4i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOdhwi8o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIdhw8i8o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIdhw8o8i))

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOidhw4o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOidhw16o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOdhwi16o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIdhw16o16i))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gOIdhw16i16o))
            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, f32, gIOdhw16o16i))

            REG_SR(f32, any, f32, any, fmt_order::any, spec::reference)

            nullptr,
        }},
    });
    return the_map;
}

// clang-format on

} // namespace cpu
} // namespace impl
} // namespace dnnl
