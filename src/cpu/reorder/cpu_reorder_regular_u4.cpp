/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

const impl_list_map_t &regular_u4_impl_list_map() {
    static const impl_list_map_t the_map = REG_REORDER_P({
        {{f32, u4, 0}, {
            REG_SR(f32, any, u4, any, fmt_order::any, spec::reference)
            nullptr,
        }},
        {{u4, data_type::undef, 0}, {
            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::brgemm_matmul_copy_reorder_t))
            REG_SR(u4, any, f32, any, fmt_order::any, spec::reference)
            REG_SR(u4, any, bf16, any, fmt_order::any, spec::reference)
            REG_SR(u4, any, f16, any, fmt_order::any, spec::reference)
            nullptr,
        }},
    });
    return the_map;
}

// clang-format on

} // namespace cpu
} // namespace impl
} // namespace dnnl
