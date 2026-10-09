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
            CPU_INSTANCE(rnn_weights_reorder_s8_t<s8>)
            CPU_INSTANCE(rnn_brgemm_weights_reorder_s8_t<s8, s8>)
            CPU_INSTANCE_X64_ZEN(x64::zen::reorder::zen_reorder_t)
            CPU_INSTANCE_X64(x64::brgemm_matmul_copy_reorder_t)

            CPU_INSTANCE_X64(x64::jit_uni_reorder_direct_copy_t)
            CPU_INSTANCE_X64(x64::jit_blk_reorder_t)
            CPU_INSTANCE_X64(x64::jit_uni_reorder_t)

            CPU_INSTANCE_AARCH64(aarch64::jit_blk_reorder_t)
            CPU_INSTANCE_AARCH64(aarch64::jit_uni_reorder_t)

            CPU_INSTANCE_RV64(rv64::jit_blk_reorder_t)
            CPU_INSTANCE_RV64(rv64::jit_uni_reorder_t)

            CPU_INSTANCE(simple_reorder_t)



            CPU_INSTANCE(ref_reorder_t)

            CPU_INSTANCE_X64(simple_sparse_reorder_t<s8, impl::format_tag_t, any, s8, impl::format_tag_t, any>)

            nullptr,
        }},
    });
    return the_map;
}

// clang-format on

} // namespace cpu
} // namespace impl
} // namespace dnnl
