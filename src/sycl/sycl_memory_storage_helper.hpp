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

#ifndef SYCL_SYCL_MEMORY_STORAGE_HELPER_HPP
#define SYCL_SYCL_MEMORY_STORAGE_HELPER_HPP

#include <optional>
#include "sycl/sycl_memory_storage.hpp"

#ifdef DNNL_SYCL_CUDA
#include "gpu/nvidia/sycl_cuda_compat.hpp"
#endif

#ifdef DNNL_SYCL_HIP
#include "gpu/amd/sycl_hip_compat.hpp"
#endif

namespace dnnl {
namespace impl {
namespace sycl {

#define CTX_IN_SYCL_MEMORY(arg) \
    dnnl::impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read>( \
            &CTX_IN_STORAGE(arg), cgh)

#define CTX_OUT_SYCL_MEMORY(arg) \
    dnnl::impl::sycl::sycl_memory_arg_t<::sycl::access::mode::write>( \
            &CTX_OUT_STORAGE(arg), cgh)

#define CTX_SCRATCH_SYCL_MEMORY(arg) \
    dnnl::impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read_write>( \
            ctx.get_scratchpad_grantor().get_memory_storage(arg).get(), cgh)

template <::sycl::access_mode mode>
class sycl_memory_arg_t {
#if defined(DNNL_SYCL_CUDA)
    static constexpr auto be = ::sycl::backend::ext_oneapi_cuda;
#elif defined(DNNL_SYCL_HIP)
    static constexpr auto be = ::sycl::backend::ext_oneapi_hip;
#else
    static_assert(false,
            "This file is not expected to be used for Intel GPU vendor.");
#endif

public:
    sycl_memory_arg_t() = default;
    sycl_memory_arg_t(memory_storage_t *raw_mem, ::sycl::handler &cgh) {
        if (!raw_mem || raw_mem->is_null()) { return; }
        auto *mem = static_cast<impl::sycl::sycl_memory_storage_base_t *>(
                raw_mem);
        switch (mem->memory_kind()) {
            case sycl::memory_kind::buffer: {
                auto *buffer_storage
                        = utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                                mem);
                acc_.emplace(buffer_storage->buffer(), cgh);
                offset_ = buffer_storage->base_offset();
                break;
            }
            case sycl::memory_kind::usm: {
                raw_ptr_ = utils::downcast<
                        const sycl::sycl_usm_memory_storage_t *>(mem)
                                   ->usm_ptr();
                break;
            }
            default: assert(!"unexpected memory kind");
        }
    }

    sycl_memory_arg_t(::sycl::buffer<uint8_t> buf, ::sycl::handler &cgh,
            size_t offset = 0)
        : offset_ {offset} {
        acc_.emplace(buf, cgh);
    }

    template <typename T = void>
    T *get_native_pointer(
#ifdef DNNL_SYCL_CUDA
            const gpu::nvidia::compat::interop_handle
#endif
#ifdef DNNL_SYCL_HIP
            const gpu::amd::compat::interop_handle
#endif
                    &ih) const {
        void *raw_ptr = nullptr;
        if (acc_.has_value()) {
            raw_ptr = reinterpret_cast<T *>(
                    reinterpret_cast<uint8_t *>(
                            ih.get_native_mem<be>(acc_.value()))
                    + offset_);
        } else {
            raw_ptr = raw_ptr_;
        }
        return reinterpret_cast<T *>(raw_ptr);
    }

    bool empty() const { return !raw_ptr_ && !acc_.has_value(); }

private:
    void *raw_ptr_ = nullptr;
    std::optional<::sycl::accessor<uint8_t, 1, mode>> acc_;
    size_t offset_;
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
