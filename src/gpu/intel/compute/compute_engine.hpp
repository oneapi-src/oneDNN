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

#ifndef GPU_INTEL_COMPUTE_COMPUTE_ENGINE_HPP
#define GPU_INTEL_COMPUTE_COMPUTE_ENGINE_HPP

#include <cassert>
#include <memory>
#include <vector>
#include <initializer_list>

#include "common/c_types_map.hpp"
#include "common/engine_impl.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/resource.hpp"
#include "common/stream.hpp"
#include "common/verbose.hpp"

#include "xpu/utils.hpp"

#include "gpu/gpu_engine.hpp"

#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/compute/dispatch.hpp"
#include "gpu/intel/compute/kernel.hpp"
#include "gpu/intel/compute/kernel_ctx.hpp"
#include "gpu/intel/jit/generator_base.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

class compute_engine_t : public gpu::engine_t {
public:
    compute_engine_t(impl::engine_impl_t *impl) : engine_t(impl) {}

    virtual status_t init();
    status_t init(const std::vector<uint8_t> &cache_blob);

    const device_info_t *device_info() const { return device_info_.get(); }

    virtual status_t create_kernel(
            compute::kernel_t *kernel, jit::generator_base_t *jitter) const = 0;

    virtual status_t create_kernels(std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx) const = 0;

    status_t create_kernel_bundle(kernel_bundle_t &bundle,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx) const {
        std::vector<kernel_t> kernels;
        CHECK(create_kernels(&kernels, kernel_names, kernel_ctx));
        bundle = kernel_bundle_t(std::move(kernels), kernel_names);
        return status::success;
    }

    virtual status_t create_kernel_from_binary(compute::kernel_t &kernel,
            const xpu::binary_t &binary, const char *kernel_name,
            const program_src_t &src) const = 0;

    virtual status_t create_kernels_from_cache_blob(
            const cache_blob_t &cache_blob,
            std::vector<compute::kernel_t> &kernels,
            const std::vector<const char *> &kernel_names) const = 0;

    status_t create_kernel_from_cache_blob(const cache_blob_t &cache_blob,
            compute::kernel_t &kernel, const char *kernel_name) const {
        std::vector<compute::kernel_t> kernels;
        CHECK(create_kernels_from_cache_blob(
                cache_blob, kernels, {kernel_name}));
        if (kernels.size() != 1) return status::runtime_error;
        kernel = std::move(kernels[0]);
        return status::success;
    };

    status_t get_zero_pad_primitive(
            impl::primitive_t *&result, const resource_mapper_t *&resources) {
        std::call_once(zero_pad_init_, [&]() -> void {
            zero_pad_desc_t desc;
            desc.primitive_kind = primitive_kind::zero_pad;
            primitive_desc_iterator_t it(
                    this, (op_desc_t *)&desc, nullptr, nullptr);
            std::shared_ptr<primitive_desc_t> zero_pad_pd(*(++it));
            if (zero_pad_pd == nullptr) return;

            status_t status
                    = zero_pad_pd->create_primitive(zero_pad_primitive_, this);
            if (status != status::success) { zero_pad_primitive_.reset(); }
        });

        result = zero_pad_primitive_.get();
        resources = &zero_pad_resources_;
        return result != nullptr ? status::success : status::unimplemented;
    };

    bool mayiuse_f16_accumulator_with_f16() const override {
        // XeHPC+ must use f32 accumulation with f16 operations as documented.
        switch (device_info_->gpu_arch()) {
            case gpu_arch_t::gen9:
            case gpu_arch_t::gen11:
            case gpu_arch_t::xe_lp:
            case gpu_arch_t::xe_hp:
            case gpu_arch_t::xe_hpg: return true;
            default: return false;
        }
    }

    bool mayiuse(device_ext_t ext) const { return device_info_->has(ext); }

    bool is_gen9() const {
        return device_info_->gpu_arch() == gpu_arch_t::gen9;
    }
    bool is_gen11() const {
        return device_info_->gpu_arch() == gpu_arch_t::gen11;
    }
    bool is_xe_lp() const {
        return device_info_->gpu_arch() == gpu_arch_t::xe_lp;
    }
    bool is_xe_hp() const {
        return device_info_->gpu_arch() == gpu_arch_t::xe_hp;
    }
    bool is_xe_hpg() const {
        return device_info_->gpu_arch() == gpu_arch_t::xe_hpg;
    }
    bool is_xe_hpc() const {
        return device_info_->gpu_arch() == gpu_arch_t::xe_hpc;
    }
    bool mayiuse_ngen_kernels() const {
        return device_info_->mayiuse_ngen_kernels();
    }
    bool mayiuse_non_uniform_work_groups() const {
        return device_info_->mayiuse_non_uniform_work_groups();
    }
    /// Returns true if the engine can directly access pointers from system allocators
    bool mayiuse_system_memory_allocators() const override {
        return device_info_->mayiuse_system_memory_allocators();
    }
    bool mayiuse_sub_group(int size) const {
        return device_info_->mayiuse_sub_group(size);
    }
    bool mayiuse_sub_group(std::initializer_list<int> sizes) const {
        for (int size : sizes)
            if (!mayiuse_sub_group(size)) return false;
        return true;
    }
    bool mayiuse_block_reads_writes_with_sub_group(int size) const {
        return size <= 16
                ? true
                : device_info_->gpu_arch() >= compute::gpu_arch_t::xe_hpc;
    }
    bool mayiuse_large_grf_mode() const {
        return device_info()->mayiuse_systolic();
    }

    dispatch_t create_dispatch(const memory_desc_t *md = nullptr) const {
        return dispatch_t(this, md);
    }

    virtual gpu_utils::device_id_t device_id() const = 0;

protected:
    virtual status_t init_device_info() = 0;
    virtual status_t init_device_info(const std::vector<uint8_t> &cache_blob) {
        assert(!"unexpected");
        return status::runtime_error;
    }

    ~compute_engine_t() override = default;

    std::shared_ptr<device_info_t> device_info_;

private:
    // Implement a zero_pad_primitive shared across the engine. The purpose is
    // to prevent extra overhead associated with creating zero_pad_primitives
    // for different inputs as ideally the zero_pad operations fast relative to
    // the time to create the primitive.
    std::shared_ptr<impl::primitive_t> zero_pad_primitive_;
    resource_mapper_t zero_pad_resources_;
    std::once_flag zero_pad_init_;
};

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

// Exported for testing purposes only.
extern "C" bool DNNL_API dnnl_impl_gpu_mayiuse_ngen_kernels(
        dnnl::impl::engine_t *engine);

#endif
