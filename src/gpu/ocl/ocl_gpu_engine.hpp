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

#ifndef GPU_OCL_OCL_GPU_ENGINE_HPP
#define GPU_OCL_OCL_GPU_ENGINE_HPP

#include "dnnl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/ocl/ocl_gpu_device_info.hpp"
#include "gpu/ocl/ocl_gpu_kernel.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

class ocl_gpu_engine_impl_list_t {
public:
    static const engine_t::concat_primitive_desc_create_f *
    get_concat_implementation_list();
    static const engine_t::reorder_primitive_desc_create_f *
    get_reorder_implementation_list(
            const memory_desc_t *src_md, const memory_desc_t *dst_md);
    static const engine_t::sum_primitive_desc_create_f *
    get_sum_implementation_list();
    static const engine_t::primitive_desc_create_f *get_implementation_list();
};

class ocl_gpu_engine_t : public compute::compute_engine_t {
public:
    static status_t get_ocl_devices(std::vector<cl_device_id> *devices);

    ocl_gpu_engine_t(cl_device_id adevice)
        : compute::compute_engine_t(engine_kind::gpu, runtime_kind::ocl,
                new ocl_gpu_device_info_t(adevice))
        , device_(adevice)
        , context_(nullptr)
        , is_user_context_(false) {}
    ocl_gpu_engine_t(cl_device_id adevice, cl_context acontext)
        : compute::compute_engine_t(engine_kind::gpu, runtime_kind::ocl,
                new ocl_gpu_device_info_t(adevice))
        , device_(adevice)
        , context_(acontext)
        , is_user_context_(true) {}
    virtual ~ocl_gpu_engine_t() override {
        if (context_) { clReleaseContext(context_); }
    }

    status_t init();

    virtual status_t create_memory_storage(memory_storage_t **storage,
            unsigned flags, size_t size, void *handle) override;

    virtual status_t create_stream(stream_t **stream, unsigned flags) override;
    status_t create_stream(stream_t **stream, cl_command_queue queue);

    virtual status_t create_kernels(std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx) const override;

    virtual const concat_primitive_desc_create_f *
    get_concat_implementation_list() const override {
        return ocl_gpu_engine_impl_list_t::get_concat_implementation_list();
    }

    virtual const reorder_primitive_desc_create_f *
    get_reorder_implementation_list(const memory_desc_t *src_md,
            const memory_desc_t *dst_md) const override {
        return ocl_gpu_engine_impl_list_t::get_reorder_implementation_list(
                src_md, dst_md);
    }

    virtual const sum_primitive_desc_create_f *
    get_sum_implementation_list() const override {
        return ocl_gpu_engine_impl_list_t::get_sum_implementation_list();
    }

    virtual const primitive_desc_create_f *get_implementation_list(
            const op_desc_t *desc) const override {
        UNUSED(desc);
        return ocl_gpu_engine_impl_list_t::get_implementation_list();
    }

    virtual cl_device_id device() const { return device_; }
    virtual cl_context context() const { return context_; }

    stream_t *service_stream() const { return service_stream_.get(); }

private:
    cl_device_id device_;
    cl_context context_;
    bool is_user_context_;

    std::unique_ptr<stream_t> service_stream_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
