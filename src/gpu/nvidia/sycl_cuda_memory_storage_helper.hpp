/*******************************************************************************
* Copyright 2022 Intel Corporation
* Copyright 2022 Codeplay Software Limited
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

#ifndef GPU_NVIDIA_SYCL_CUDA_MEMORY_STORAGE_HELPER_HPP
#define GPU_NVIDIA_SYCL_CUDA_MEMORY_STORAGE_HELPER_HPP

#include <optional>
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "sycl/sycl_memory_storage.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

#define CTX_IN_MEMORY(arg) \
    static_cast<sycl::sycl_memory_storage_base_t *>(&CTX_IN_STORAGE(arg))

#define CTX_OUT_MEMORY(arg) \
    static_cast<sycl::sycl_memory_storage_base_t *>(&CTX_OUT_STORAGE(arg))

#define CTX_IN_OPTIONAL_ACCESSOR(arg, memory) \
    get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(arg))>(memory, cgh)

#define CTX_OUT_OPTIONAL_ACCESSOR(arg, memory) \
    get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(arg))>(memory, cgh)

template <typename T_acc>
inline std::optional<T_acc> get_cudnn_accessor(
        sycl::sycl_memory_storage_base_t *mem, ::sycl::handler &cgh) {
    std::optional<T_acc> acc;
    switch (mem->memory_kind()) {
        case sycl::memory_kind::buffer:
            acc.emplace(
                    utils::downcast<sycl::sycl_buffer_memory_storage_t *>(mem)
                            ->buffer(),
                    cgh);
            break;
        case sycl::memory_kind::usm:
            // USM storage does not use accessors, so in this case empty
            // std:optional is returned
            break;
        default: assert(!"unexpected memory kind");
    }
    return acc;
}

template <typename T_acc>
inline void *get_cudnn_ptr(cuda_sycl_scoped_context_handler_t &sc,
        const compat::interop_handle &ih, const std::optional<T_acc> &acc,
        sycl::sycl_memory_storage_base_t *mem) {
    void *ptr;
    switch (mem->memory_kind()) {
        case sycl::memory_kind::buffer:
            ptr = sc.memory<void *>(ih, acc.value());
            break;
        case sycl::memory_kind::usm:
            ptr = utils::downcast<const sycl::sycl_usm_memory_storage_t *>(mem)
                          ->usm_ptr();
            break;
        default: assert(!"unexpected memory kind");
    }
    return ptr;
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
