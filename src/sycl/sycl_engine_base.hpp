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

#ifndef SYCL_ENGINE_BASE_HPP
#define SYCL_ENGINE_BASE_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/memory_storage.hpp"
#include "common/stream.hpp"
#include "ocl/cl_device_info.hpp"
#include "ocl/cl_engine.hpp"
#include "ocl/ocl_utils.hpp"

#include <CL/sycl.hpp>

namespace mkldnn {
namespace impl {
namespace sycl {

class sycl_engine_base_t : public ocl::cl_engine_t
{
public:
    sycl_engine_base_t(engine_kind_t kind, const cl::sycl::device &dev,
            const cl::sycl::context &ctx)
        : cl_engine_t(kind, backend_kind::sycl,
                  ocl::cl_device_info_t(
                          ocl::ocl_utils::make_ocl_wrapper(dev.get())))
        , device_(dev)
        , context_(ctx) {}

    virtual status_t init() {
        CHECK(ocl::cl_engine_t::init());
        stream_t *service_stream_ptr;
        status_t status = create_stream(&service_stream_ptr, stream_flags::default_flags);
        if (status != status::success)
            return status;
        service_stream_.reset(service_stream_ptr);
        return status::success;
    }

    virtual status_t create_memory_storage(memory_storage_t **storage,
            unsigned flags, size_t size, size_t alignment,
            void *handle) override;

    virtual status_t create_stream(stream_t **stream, unsigned flags) override;
    status_t create_stream(stream_t **stream, cl::sycl::queue &queue);

    const cl::sycl::device &device() const { return device_; }
    const cl::sycl::context &context() const { return context_; }

    stream_t *service_stream() const { return service_stream_.get(); }

    virtual cl_device_id ocl_device() const override {
        return ocl::ocl_utils::make_ocl_wrapper(device().get());
    }
    virtual cl_context ocl_context() const override {
        return ocl::ocl_utils::make_ocl_wrapper(context().get());
    }

private:
    cl::sycl::device device_;
    cl::sycl::context context_;

    std::unique_ptr<stream_t> service_stream_;
};

} // namespace sycl
} // namespace impl
} // namespace mkldnn

#endif
