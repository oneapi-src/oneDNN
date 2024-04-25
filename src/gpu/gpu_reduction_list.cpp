/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#include "gpu/gpu_impl_list.hpp"

#include "gpu/intel/jit/jit_reduction.hpp"
#include "gpu/intel/ocl/reduction/atomic_reduction.hpp"
#include "gpu/intel/ocl/reduction/combined_reduction.hpp"
#include "gpu/intel/ocl/reduction/ref_reduction.hpp"
#include "gpu/intel/ocl/reduction/reusable_ref_reduction.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

namespace {

#ifdef DNNL_DEV_MODE
#define JIT_REDUCTION_INSTANCE INSTANCE(intel::jit::jit_reduction_t)
#else
#define JIT_REDUCTION_INSTANCE
#endif

// clang-format off
constexpr impl_list_item_t impl_list[] = REG_REDUCTION_P({
        JIT_REDUCTION_INSTANCE
        INSTANCE(intel::ocl::atomic_reduction_t)
        INSTANCE(intel::ocl::combined_reduction_t)
        INSTANCE(intel::ocl::ref_reduction_t)
        INSTANCE(intel::ocl::reusable_ref_reduction_t)
        nullptr,
});
// clang-format on

#undef JIT_REDUCTION_INSTANCE
} // namespace

const impl_list_item_t *get_reduction_impl_list(const reduction_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
