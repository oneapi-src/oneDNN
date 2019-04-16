/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "ocl/ocl_engine.hpp"

#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "ocl/ocl_memory_storage.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

#include <CL/cl.h>

namespace mkldnn {
namespace impl {
namespace ocl {

status_t ocl_engine_t::init() {
    CHECK(cl_engine_t::init());

    cl_int err = CL_SUCCESS;
    if (is_user_context_) {
        err = clRetainContext(context_);
        if (err != CL_SUCCESS)
            context_ = nullptr;
    } else {
        context_
                = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    }

    OCL_CHECK(err);

    status_t status
            = ocl_utils::check_device(engine_kind::gpu, device_, context_);
    if (status != status::success)
        return status;

    stream_t *service_stream_ptr;
    status = create_stream(&service_stream_ptr, stream_flags::default_flags);
    if (status != status::success)
        return status;
    service_stream_.reset(service_stream_ptr);
    return status::success;
}

status_t ocl_engine_t::create_memory_storage(
        memory_storage_t **storage, unsigned flags, size_t size, void *handle) {
    return safe_ptr_assign<memory_storage_t>(
            *storage, new ocl_memory_storage_t(this, flags, size, handle));
}

status_t ocl_engine_t::create_stream(stream_t **stream, unsigned flags) {
    return ocl_stream_t::create_stream(stream, this, flags);
}

status_t ocl_engine_t::create_stream(
        stream_t **stream, cl_command_queue queue) {
    return ocl_stream_t::create_stream(stream, this, queue);
}

using pd_create_f = mkldnn::impl::engine_t::primitive_desc_create_f;

namespace {
using namespace mkldnn::impl::data_type;

#define INSTANCE(...) &primitive_desc_t::create<__VA_ARGS__::pd_t>
static const pd_create_f ocl_impl_list[] = {
    nullptr,
};

#undef INSTANCE
} // namespace

const pd_create_f *ocl_engine_t::get_implementation_list() const {
    return ocl_impl_list;
}

} // namespace ocl
} // namespace impl
} // namespace mkldnn
