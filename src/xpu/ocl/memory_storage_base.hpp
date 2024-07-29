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

#ifndef XPU_OCL_MEMORY_STORAGE_BASE_HPP
#define XPU_OCL_MEMORY_STORAGE_BASE_HPP

#include "common/memory_storage.hpp"

#include "xpu/ocl/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ocl {

class memory_storage_base_t : public impl::memory_storage_t {
public:
    // Explicitly define ctors due to a "circular dependencies" bug in ICC.
    memory_storage_base_t(
            impl::engine_t *engine, const memory_storage_t *root_storage)
        : impl::memory_storage_t(engine, root_storage) {}
    memory_storage_base_t(impl::engine_t *engine)
        : memory_storage_base_t(engine, this) {}

    virtual memory_kind_t memory_kind() const = 0;
};

} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif // XPU_OCL_MEMORY_STORAGE_BASE_HPP
