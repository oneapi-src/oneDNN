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

#ifndef GPU_GENERIC_SYCL_ENGINE_HPP
#define GPU_GENERIC_SYCL_ENGINE_HPP

#include "common/stream.hpp"

#include "xpu/sycl/engine_impl.hpp"

#include "gpu/gpu_engine.hpp"
#include "gpu/gpu_impl_list.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

status_t engine_create(impl::engine_t **engine, engine_kind_t engine_kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index);

class engine_t : public gpu::engine_t {
public:
    engine_t(const ::sycl::device &dev, const ::sycl::context &ctx,
            size_t index);

    status_t init() { return init_impl(); }

    status_t create_stream(
            impl::stream_t **stream, impl::stream_impl_t *stream_impl) override;

    bool mayiuse_system_memory_allocators() const override {
        return impl()->mayiuse_system_memory_allocators();
    }

    DECLARE_COMMON_SYCL_ENGINE_FUNCTIONS();

protected:
    const xpu::sycl::engine_impl_t *impl() const {
        return (const xpu::sycl::engine_impl_t *)impl::engine_t::impl();
    }
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
