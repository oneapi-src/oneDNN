/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#include "common/impl_list_item.hpp"
#include "common/utils.hpp"
#include "hip/hip_runtime.h"

#include "miopen/miopen.h"
#include "sycl/sycl_utils.hpp"

#include "gpu/amd/miopen_binary.hpp"
#include "gpu/amd/miopen_eltwise.hpp"
#include "gpu/amd/sycl_hip_compat.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"
#include "gpu/amd/sycl_hip_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

bool is_amd_gpu(const ::sycl::device &dev) {
    constexpr int amd_vendor_id = 0x1002;
    return dev.is_gpu()
            && dev.get_info<::sycl::info::device::vendor_id>() == amd_vendor_id;
}

status_t hip_engine_create(engine_t **engine, engine_kind_t engine_kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index) {
    CHECK(amd::check_device(engine_kind));
    std::unique_ptr<amd::sycl_hip_engine_t, engine_deleter_t> hip_engine(
            (new amd::sycl_hip_engine_t(dev, ctx, index)));
    if (!hip_engine) return status::out_of_memory;

    CHECK(hip_engine->init());
    *engine = hip_engine.release();

    return status::success;
}

sycl_hip_engine_t::sycl_hip_engine_t(engine_kind_t kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index)
    : base_t(kind, dev, ctx, index) {
    set_miopen_handle();
}

sycl_hip_engine_t::sycl_hip_engine_t(
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index)
    : sycl_hip_engine_t(engine_kind::gpu, dev, ctx, index) {
    assert(is_amd_gpu(dev));
}
status_t sycl_hip_engine_t::set_miopen_handle() {
    // scoped context will make sure the top of the stack context is
    // the engine context while creating the miopen handle.
    hip_sycl_scoped_context_handler_t sc(*this);
    miopenHandle_t handle;
    CHECK(MIOPEN_EXECUTE_FUNC_S(miopenCreate, &handle));
    miopen_handle_.set(
            std::unique_ptr<miopenHandle_t, void (*)(miopenHandle_t *)>(
                    new miopenHandle_t(handle), [](miopenHandle_t *h) {
                        if (h != nullptr)
                            MIOPEN_EXECUTE_FUNC_V(miopenDestroy, *h);
                        delete h;
                    }));
    handle = nullptr;
    return status::success;
}
hipCtx_t sycl_hip_engine_t::get_underlying_context() const {
    return compat::get_native<hipCtx_t>(context());
}

status_t sycl_hip_engine_t::create_stream(stream_t **stream, unsigned flags) {
    return sycl_hip_stream_t::create_stream(stream, this, flags);
}

status_t sycl_hip_engine_t::create_stream(
        stream_t **stream, ::sycl::queue &queue) {
    return sycl_hip_stream_t::create_stream(stream, this, queue);
}

miopenHandle_t *sycl_hip_engine_t::get_miopen_handle() {
    if (!miopen_handle_.is_set()) set_miopen_handle();
    return miopen_handle_.get().get();
}

device_id_t sycl_hip_engine_t::device_id() const {
    return device_id_t(static_cast<int>(impl::sycl::backend_t::amd),
            static_cast<uint64_t>(compat::get_native<hipDevice_t>(device())),
            static_cast<uint64_t>(0));
}

void sycl_hip_engine_t::activate_stream_miopen(stream_t *stream) {
    hip_sycl_scoped_context_handler_t sc(*this);
    auto hip_stream = utils::downcast<sycl_hip_stream_t *>(stream);
    auto streamId = hip_stream->get_underlying_stream();
    assert(context() == hip_stream->queue().get_context());
    hipStream_t current_stream_id = nullptr;
    auto miopen_handle = get_miopen_handle();
    MIOPEN_EXECUTE_FUNC_S(miopenGetStream, *miopen_handle, &current_stream_id);
    if (current_stream_id != streamId) {
        MIOPEN_EXECUTE_FUNC_S(miopenSetStream, *miopen_handle, streamId);
    }
}

namespace {
using namespace dnnl::impl::data_type;

// clang-format off
constexpr dnnl::impl::impl_list_item_t sycl_hip_impl_list[] = {
        // Binary
        INSTANCE(miopen_binary_t)
        // Elementwise
        INSTANCE(miopen_eltwise_fwd_t)
        INSTANCE(miopen_eltwise_bwd_t)
        nullptr,
};
// clang-format on
} // namespace

const dnnl::impl::impl_list_item_t *sycl_hip_engine_t::get_implementation_list(
        const op_desc_t *) const {
    return sycl_hip_impl_list;
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl
