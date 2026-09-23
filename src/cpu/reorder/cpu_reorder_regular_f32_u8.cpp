/*******************************************************************************
* Copyright 2020 Intel Corporation
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

const impl_list_map_t &regular_f32_u8_impl_list_map() {
    static const impl_list_map_t the_map = REG_REORDER_P({
        // f32 -> u8
        {{f32, u8, 0}, {
            CPU_INSTANCE(rnn_data_reorder_t<f32, u8>)

            CPU_INSTANCE_X64(x64::jit_uni_reorder_direct_copy_t)
            CPU_INSTANCE_X64(x64::jit_blk_reorder_t)
            CPU_INSTANCE_X64(x64::jit_uni_reorder_t)

            CPU_INSTANCE_AARCH64(aarch64::jit_blk_reorder_t)
            CPU_INSTANCE_AARCH64(aarch64::jit_uni_reorder_t)

            CPU_INSTANCE_PPC64(ppc64::ppc64_matrixA_reorder_t)

            CPU_INSTANCE_RV64(rv64::jit_blk_reorder_t)
            CPU_INSTANCE_RV64(rv64::jit_uni_reorder_t)

            REG_FAST_DIRECT_COPY(f32, u8)

            DNNL_NON_X64_ONLY(REG_SR_BIDIR(f32, any, u8, nChw16c))

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
