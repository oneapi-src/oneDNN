/*******************************************************************************
* Copyright 2017-2021 Intel Corporation
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

#include "cpu/cpu_engine.hpp"

#include "common/impl_list_item.hpp"
#include "cpu/ref_concat.hpp"
#include "cpu/simple_concat.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
#define INSTANCE(...) \
    impl_list_item_t(impl_list_item_t::concat_type_deduction_helper_t< \
            __VA_ARGS__::pd_t>()),
// clang-format off
const impl_list_item_t cpu_concat_impl_list[] = {
        INSTANCE(simple_concat_t<data_type::f32>)
        INSTANCE(simple_concat_t<data_type::u8>)
        INSTANCE(simple_concat_t<data_type::s8>)
        INSTANCE(simple_concat_t<data_type::s32>)
        INSTANCE(simple_concat_t<data_type::bf16>)
        INSTANCE(ref_concat_t)
        nullptr,
};
// clang-format on
#undef INSTANCE
} // namespace

const impl_list_item_t *
cpu_engine_impl_list_t::get_concat_implementation_list() {
    return cpu_concat_impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
