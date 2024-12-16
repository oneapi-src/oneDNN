/*******************************************************************************
* Copyright 2024 Intel Corporation
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

const impl_list_map_t &regular_fp4_impl_list_map() {
    static const impl_list_map_t the_map = REG_REORDER_P({
        {{f32, f4_e2m1, 0}, {
            REG_SR(f32, any, f4_e2m1, any, fmt_order::any, spec::reference)
            nullptr,
        }},
        {{f4_e2m1, data_type::undef, 0}, {
            REG_SR(f4_e2m1, any, f32, any, fmt_order::any, spec::reference)
            nullptr,
        }},
        {{f32, f4_e3m0, 0}, {
            REG_SR(f32, any, f4_e3m0, any, fmt_order::any, spec::reference)
            nullptr,
        }},
        {{f4_e3m0, data_type::undef, 0}, {
            REG_SR(f4_e3m0, any, f32, any, fmt_order::any, spec::reference)
            nullptr,
        }},
    });
    return the_map;
}

// clang-format on

} // namespace cpu
} // namespace impl
} // namespace dnnl
