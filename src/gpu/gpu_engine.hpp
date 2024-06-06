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

#ifndef GPU_GPU_ENGINE_HPP
#define GPU_GPU_ENGINE_HPP

#include <mutex>

#include "common/engine.hpp"

#include "gpu/gpu_impl_list.hpp"

#define CTX_GPU_RES_STORAGE(arg) \
    (*(ctx.get_resource_mapper() \
                    ->template get<gpu_resource_t>(this) \
                    ->get_memory_storage(arg)))

namespace dnnl {
namespace impl {
namespace gpu {

class engine_t : public impl::engine_t {
public:
    using dnnl::impl::engine_t::engine_t;

    const impl_list_item_t *get_reorder_implementation_list(
            const memory_desc_t *src_md,
            const memory_desc_t *dst_md) const override {
        return gpu::gpu_impl_list_t::get_reorder_implementation_list(
                src_md, dst_md);
    }

    const impl_list_item_t *get_concat_implementation_list() const override {
        return gpu::gpu_impl_list_t::get_concat_implementation_list();
    }

    const impl_list_item_t *get_sum_implementation_list() const override {
        return gpu::gpu_impl_list_t::get_sum_implementation_list();
    }

    const impl_list_item_t *get_implementation_list(
            const op_desc_t *desc) const override {
        return gpu::gpu_impl_list_t::get_implementation_list(desc);
    }

    int get_buffer_alignment() const { return impl()->get_buffer_alignment(); }

    status_t get_service_stream(impl::stream_t *&stream) override {
        status_t status = status::success;
        if (service_stream_ == nullptr) {
            const std::lock_guard<std::mutex> lock(service_stream_mutex_);
            if (service_stream_ == nullptr) {
                impl::stream_t *service_stream_ptr;
                status = create_stream(
                        &service_stream_ptr, stream_flags::default_flags);
                if (status == status::success)
                    service_stream_.reset(service_stream_ptr);
            }
        }
        stream = service_stream_.get();
        return status;
    }

private:
    std::unique_ptr<impl::stream_t> service_stream_;
    std::mutex service_stream_mutex_;
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
