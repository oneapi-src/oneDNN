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

#ifndef CPU_SYCL_STREAM_HPP
#define CPU_SYCL_STREAM_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/primitive_iface.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_stream.hpp"

#include "xpu/sycl/context.hpp"
#include "xpu/sycl/memory_storage.hpp"
#include "xpu/sycl/stream_impl.hpp"

#include "cpu/sycl/stream_cpu_thunk.hpp"
#include "cpu/sycl/stream_submit_cpu_primitive.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace sycl {

struct stream_t : public cpu::cpu_stream_t {
    static status_t create_stream(impl::stream_t **stream, engine_t *engine,
            impl::stream_impl_t *stream_impl) {
        std::unique_ptr<stream_t> s(new stream_t(engine, stream_impl));
        if (!s) return status::out_of_memory;

        status_t status = s->init();
        if (status != status::success) {
            // Stream owns stream_impl only if it's created successfully (including initialization).
            s->impl_.release();
            return status;
        }
        *stream = s.release();
        return status::success;
    }

    status_t wait() override {
        queue().wait_and_throw();
        return status::success;
    }

    ::sycl::queue &queue() const { return *impl()->queue(); }

    status_t enqueue_primitive(const primitive_iface_t *prim_iface,
            exec_ctx_t &exec_ctx) override {
        assert(engine()->kind() == engine_kind::cpu);
        auto event = queue().submit([&](::sycl::handler &cgh) {
            register_deps(cgh);
            submit_cpu_primitive(this, prim_iface, exec_ctx, cgh);
        });
        sycl_ctx().set_deps({event});
        return status::success;
    }

    const xpu::sycl::context_t &sycl_ctx() const { return impl()->sycl_ctx(); }
    xpu::sycl::context_t &sycl_ctx() { return impl()->sycl_ctx(); }

    ::sycl::event get_output_event() const {
        return impl()->get_output_event();
    }

    void register_deps(::sycl::handler &cgh) const {
        return impl()->register_deps(cgh);
    }

    void after_exec_hook() override;

protected:
    xpu::sycl::stream_impl_t *impl() const {
        return (xpu::sycl::stream_impl_t *)impl::stream_t::impl_.get();
    }

    stream_t(engine_t *engine, impl::stream_impl_t *stream_impl)
        : cpu::cpu_stream_t(engine, stream_impl) {}

private:
    status_t init();
};
} // namespace sycl
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
