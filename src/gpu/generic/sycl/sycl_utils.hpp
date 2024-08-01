/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef GPU_GENERIC_SYCL_SYCL_UTILS_HPP
#define GPU_GENERIC_SYCL_SYCL_UTILS_HPP

#include "common/memory_desc.hpp"
#include "common/memory_desc_wrapper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

inline bool md_dims_in_range(const dnnl::impl::memory_desc_t *desc) {
    auto wrap = dnnl::impl::memory_desc_wrapper(desc);
    for (int i = 0; i < wrap.ndims(); i++) {
        if (wrap.dims()[i] > INT_MAX) { return false; }
    }

    return true;
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
