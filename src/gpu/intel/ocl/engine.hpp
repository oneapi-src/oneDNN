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

#ifndef GPU_INTEL_OCL_ENGINE_HPP
#define GPU_INTEL_OCL_ENGINE_HPP

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_impl_list.hpp"
#include "gpu/intel/compute/compute_engine.hpp"
#include "gpu/intel/ocl/utils.hpp"
#include "xpu/ocl/engine_impl.hpp"
#include "xpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

status_t engine_create(impl::engine_t **engine, engine_kind_t engine_kind,
        cl_device_id dev, cl_context ctx, size_t index,
        const std::vector<uint8_t> &cache_blob);

class engine_t : public compute::compute_engine_t {
public:
    engine_t(cl_device_id adevice, cl_context acontext, size_t index)
        : compute::compute_engine_t(
                new xpu::ocl::engine_impl_t(adevice, acontext, index)) {}

    status_t init() override;
    status_t init(const std::vector<uint8_t> &cache_blob);

    status_t create_stream(
            impl::stream_t **stream, impl::stream_impl_t *stream_impl) override;

    status_t create_kernel_from_binary(compute::kernel_t &kernel,
            const xpu::binary_t &binary, const char *kernel_name,
            const compute::program_src_t &src) const override;

    status_t create_kernels_from_cache_blob(const cache_blob_t &cache_blob,
            std::vector<compute::kernel_t> &kernels,
            const std::vector<const char *> &kernel_names) const override;

    status_t create_kernel(compute::kernel_t *kernel,
            jit::generator_base_t *jitter) const override;

    status_t create_kernels(std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx) const override;

    static status_t create_kernels_from_program(
            std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names, cl_program program,
            const compute::program_src_t &src);

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

    gpu_utils::device_id_t device_id() const override {
        return std::make_tuple(0, reinterpret_cast<uint64_t>(device()), 0);
    }

    status_t serialize_device(serialization_stream_t &sstream) const override;

    status_t get_cache_blob_size(size_t *size) const override {
        return device_info_->get_cache_blob_size(size);
    }

    status_t get_cache_blob(size_t size, uint8_t *cache_blob) const override {
        return device_info_->get_cache_blob(size, cache_blob);
    }

    status_t create_program(xpu::ocl::wrapper_t<cl_program> &program,
            compute::program_src_t &src,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx) const;

    DECLARE_COMMON_OCL_ENGINE_FUNCTIONS();

protected:
    const xpu::ocl::engine_impl_t *impl() const {
        return (const xpu::ocl::engine_impl_t *)gpu::engine_t::impl();
    }

    status_t build_program_from_source(xpu::ocl::wrapper_t<cl_program> &program,
            compute::program_src_t &src, const char *code_string,
            const compute::kernel_ctx_t &kernel_ctx) const;

    ~engine_t() override = default;

    status_t init_device_info() override;
    status_t init_device_info(const std::vector<uint8_t> &cache_blob) override;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
