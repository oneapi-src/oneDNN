/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_impl_list.hpp"
#include "gpu/ocl/ocl_gpu_engine_id.hpp"
#include "gpu/ocl/ocl_gpu_kernel.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

class ocl_gpu_engine_t : public compute::compute_engine_t {
public:
    ocl_gpu_engine_t(cl_device_id adevice, cl_context acontext, size_t index)
        : compute::compute_engine_t(engine_kind::gpu, runtime_kind::ocl, index)
        , device_(adevice)
        , context_(acontext)
        , is_user_context_(acontext) {}

    status_t init() override;
    status_t init(const std::vector<uint8_t> &cache_blob);

    status_t create_memory_storage(memory_storage_t **storage, unsigned flags,
            size_t size, void *handle) override;

    status_t create_stream(stream_t **stream, unsigned flags) override;
    status_t create_stream(stream_t **stream, cl_command_queue queue);

    status_t create_compiled_bundle(compute::compiled_bundle_t &generator,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx) const override;

    status_t create_compiled_kernel(compute::compiled_kernel_t &generator,
            jit::jit_generator_base &jitter) const override;

    status_t create_binary_from_ocl_source(compute::binary_t &binary,
            const char *code_string,
            const compute::kernel_ctx_t &kernel_ctx) const;

    status_t create_kernels_from_bundle(std::vector<compute::kernel_t> &kernels,
            const std::vector<const char *> &kernel_names,
            const compute::compiled_bundle_t &generator) const override;

    status_t create_kernel_from_binary(compute::kernel_t &kernel,
            const compute::binary_t &binary,
            const char *kernel_name) const override;

    status_t create_kernels_from_cache_blob(const cache_blob_t &cache_blob,
            std::vector<compute::kernel_t> &kernels,
            const std::vector<const char *> &kernel_names) const override;

    status_t create_kernel(compute::kernel_t *kernel,
            jit::jit_generator_base *jitter,
            const cache_blob_t &cache_blob) const override;

    status_t create_kernels(std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx,
            const cache_blob_t &cache_blob) const override;

    status_t create_kernels_from_ocl_source(
            std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const char *source_string,
            const compute::kernel_ctx_t &kernel_ctx) const override;

    const impl_list_item_t *get_concat_implementation_list() const override {
        return gpu_impl_list_t::get_concat_implementation_list();
    }

    const impl_list_item_t *get_reorder_implementation_list(
            const memory_desc_t *src_md,
            const memory_desc_t *dst_md) const override {
        return gpu_impl_list_t::get_reorder_implementation_list(src_md, dst_md);
    }

    const impl_list_item_t *get_sum_implementation_list() const override {
        return gpu_impl_list_t::get_sum_implementation_list();
    }

    const impl_list_item_t *get_implementation_list(
            const op_desc_t *desc) const override {
        return gpu_impl_list_t::get_implementation_list(desc);
    }

    cl_device_id device() const { return device_; }
    cl_context context() const { return context_; }
    cl_platform_id platform() const { return platform_; }

    device_id_t device_id() const override {
        return std::make_tuple(0, reinterpret_cast<uint64_t>(device()), 0);
    }

    status_t serialize_device(serialization_stream_t &sstream) const override;

    status_t get_cache_blob_size(size_t *size) const {
        return device_info_->get_cache_blob_size(size);
    }

    status_t get_cache_blob(size_t size, uint8_t *cache_blob) const {
        return device_info_->get_cache_blob(size, cache_blob);
    }

    engine_id_t engine_id() const override {
        return engine_id_t(new ocl_gpu_engine_id_impl_t(
                device(), context(), kind(), runtime_kind(), index()));
    }

protected:
    status_t build_program_from_source(ocl_wrapper_t<cl_program> &program,
            const char *code_string,
            const compute::kernel_ctx_t &kernel_ctx) const;

    ~ocl_gpu_engine_t() override {
        if (device_) { clReleaseDevice(device_); }
        if (context_) { clReleaseContext(context_); }
    }

    status_t init_device_info() override;
    status_t init_device_info(const std::vector<uint8_t> &cache_blob) override;

private:
    cl_device_id device_;
    cl_context context_;
    cl_platform_id platform_ = nullptr;
    bool is_user_context_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
