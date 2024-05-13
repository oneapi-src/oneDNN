/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include "gpu/gpu_impl_list.hpp"

#include "gpu/intel/ocl/gen9_concat.hpp"
#include "gpu/intel/ocl/multi_concat.hpp"
#include "gpu/intel/ocl/ref_concat.hpp"
#include "gpu/intel/ocl/simple_concat.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

namespace {
#define CONCAT_INSTANCE(...) \
    impl_list_item_t(impl_list_item_t::concat_type_deduction_helper_t< \
            __VA_ARGS__::pd_t>()),

// clang-format off
constexpr impl_list_item_t impl_list[] = REG_CONCAT_P({
        CONCAT_INSTANCE(intel::ocl::simple_concat_t)
        CONCAT_INSTANCE(intel::ocl::gen9_concat_t)
        CONCAT_INSTANCE(intel::ocl::multi_concat_t)
        CONCAT_INSTANCE(intel::ocl::ref_concat_t)
        nullptr,
});
// clang-format on
#undef INSTANCE
} // namespace

const impl_list_item_t *get_concat_impl_list() {
    return impl_list;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
