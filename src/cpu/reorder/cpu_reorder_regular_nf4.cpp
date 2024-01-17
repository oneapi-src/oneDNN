/*******************************************************************************
* Copyright 2021 Intel Corporation
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

const impl_list_map_t &regular_nf4_impl_list_map() {
    static const impl_list_map_t the_map = REG_REORDER_P({
        // nf4 ->
        {{nf4, data_type::undef, 0}, {
            REG_SR(nf4, any, nf4, OI8i8o2i, fmt_order_keep)
            REG_SR(nf4, any, nf4, OI8i16o2i, fmt_order_keep)
            REG_SR(nf4, any, nf4, OI8i24o2i, fmt_order_keep)
            REG_SR(nf4, any, nf4, OI8i32o2i, fmt_order_keep)
            REG_SR(nf4, any, nf4, OI8i64o2i, fmt_order_keep)
            REG_SR(nf4, any, nf4, OI16i16o2i, fmt_order_keep)
            REG_SR(nf4, any, nf4, OI16i32o2i, fmt_order_keep)
            REG_SR(nf4, any, nf4, OI16i48o2i, fmt_order_keep)
            REG_SR(nf4, any, nf4, OI16i64o2i, fmt_order_keep)
            REG_SR(nf4, any, f32, any, fmt_order_keep, spec::reference)
            nullptr,
        }},
    });
    return the_map;
}

// clang-format on

} // namespace cpu
} // namespace impl
} // namespace dnnl
