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

#ifndef OCL_ENGINE_HPP
#define OCL_ENGINE_HPP

#include "mkldnn.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"
#include "ocl/ocl_device_info.hpp"
#include "ocl/ocl_utils.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

class ocl_engine_t : public engine_t
{
public:
    static status_t get_ocl_devices(std::vector<cl_device_id> *devices);

    ocl_engine_t(cl_device_id adevice)
        : engine_t(engine_kind::gpu, backend_kind::ocl)
        , device_(adevice)
        , device_info_(adevice)
        , context_(nullptr)
        , is_user_context_(false) {}
    ocl_engine_t(cl_device_id adevice, cl_context acontext)
        : engine_t(engine_kind::gpu, backend_kind::ocl)
        , device_(adevice)
        , device_info_(adevice)
        , context_(acontext)
        , is_user_context_(true) {}
    virtual ~ocl_engine_t() override {
        if (context_) {
            clReleaseContext(context_);
        }
    }

    status_t init();

    virtual status_t create_memory_storage(
            memory_storage_t **storage, size_t size) override;
    virtual status_t create_memory_storage(
            memory_storage_t **storage, void *handle) override;

    virtual status_t create_stream(stream_t **stream, unsigned flags) override;
    status_t create_stream(stream_t **stream, cl_command_queue queue);

    virtual const concat_primitive_desc_create_f *
    get_concat_implementation_list() const override;
    virtual const reorder_primitive_desc_create_f *
    get_reorder_implementation_list() const override;
    virtual const sum_primitive_desc_create_f *
    get_sum_implementation_list() const override;
    virtual const primitive_desc_create_f *
    get_implementation_list() const override;

    cl_device_id device() const { return device_; }
    cl_context context() const { return context_; }

    stream_t *service_stream() const { return service_stream_.get(); }

    virtual bool mayiuse(device_ext_t ext) const {
        return device_info_.has(ext);
    }

private:
    cl_device_id device_;
    device_info_t device_info_;
    cl_context context_;
    bool is_user_context_;

    std::unique_ptr<stream_t> service_stream_;
};

class ocl_engine_factory_t : public engine_factory_t
{
public:
    virtual size_t count() const override {
        std::vector<cl_device_id> ocl_devices;
        status_t status
                = ocl_utils::get_ocl_devices(&ocl_devices, CL_DEVICE_TYPE_GPU);
        if (status != status::success)
            return status;
        return ocl_devices.size();
    }

    virtual status_t engine_create(
            engine_t **engine, size_t index) const override {
        status_t status;
        std::vector<cl_device_id> ocl_devices;

        status = ocl_utils::get_ocl_devices(&ocl_devices, CL_DEVICE_TYPE_GPU);
        if (status != status::success)
            return status;

        if (index >= ocl_devices.size())
            return status::invalid_arguments;

        auto *ocl_engine = new ocl_engine_t(ocl_devices[index]);
        if (!ocl_engine)
            return status::out_of_memory;

        status = ocl_engine->init();
        if (status != status::success) {
            delete ocl_engine;
            return status;
        }
        *engine = ocl_engine;
        return status::success;
    }

    status_t engine_create(
            engine_t **engine, cl_device_id device, cl_context context) {
        auto *ocl_engine = new ocl_engine_t(device, context);
        if (!ocl_engine)
            return status::out_of_memory;

        status_t status = ocl_engine->init();
        if (status != status::success) {
            delete ocl_engine;
            return status;
        }
        *engine = ocl_engine;
        return status::success;
    }
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif
