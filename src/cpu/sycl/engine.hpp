/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef CPU_SYCL_ENGINE_HPP
#define CPU_SYCL_ENGINE_HPP

#include "common/impl_list_item.hpp"

#include "xpu/sycl/engine_impl.hpp"
#include "xpu/sycl/utils.hpp"

#include "cpu/cpu_engine.hpp"
#include "cpu/sycl/stream.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace sycl {

status_t engine_create(impl::engine_t **engine, const ::sycl::device &dev,
        const ::sycl::context &ctx, size_t index);

class engine_t : public cpu::cpu_engine_t {
public:
    engine_t(
            const ::sycl::device &dev, const ::sycl::context &ctx, size_t index)
        : cpu::cpu_engine_t(new xpu::sycl::engine_impl_t(
                engine_kind::cpu, dev, ctx, index)) {}

    status_t init() { return init_impl(); }

    status_t create_memory_storage(memory_storage_t **storage, unsigned flags,
            size_t size, void *handle) override {
        assert(runtime_kind() == runtime_kind::sycl);
        if (runtime_kind() != runtime_kind::sycl) return status::runtime_error;

        return impl()->create_memory_storage(
                storage, this, flags, size, handle);
    }

    status_t create_stream(impl::stream_t **stream,
            impl::stream_impl_t *stream_impl) override {
        return cpu::sycl::stream_t::create_stream(stream, this, stream_impl);
    }

    bool mayiuse_system_memory_allocators() const override {
        return impl()->mayiuse_system_memory_allocators();
    }

    DECLARE_COMMON_SYCL_ENGINE_FUNCTIONS();

protected:
    const xpu::sycl::engine_impl_t *impl() const {
        return (const xpu::sycl::engine_impl_t *)impl::engine_t::impl();
    }

    ~engine_t() override = default;
};

} // namespace sycl
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
