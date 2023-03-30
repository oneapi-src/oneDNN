/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
* Copyright 2020-2022 Codeplay Software Limited
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
#include "common/engine.hpp"
#include "common/impl_list_item.hpp"
#include "gpu/amd/miopen_reorder.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/ocl/cross_engine_reorder.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

namespace {

#define REORDER_INSTANCE(...) \
    impl_list_item_t( \
            impl_list_item_t::reorder_type_deduction_helper_t<__VA_ARGS__>()),

// clang-format off
constexpr impl_list_item_t hip_reorder_impl_list[] = {
        REORDER_INSTANCE(gpu::ocl::cross_engine_reorder_t::pd_t)
        REORDER_INSTANCE(miopen_reorder_t::pd_t)
        nullptr,
};
// clang-format on

} // namespace

const impl_list_item_t *
hip_gpu_engine_impl_list_t::get_reorder_implementation_list(
        const memory_desc_t *, const memory_desc_t *) {
    return hip_reorder_impl_list;
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl
