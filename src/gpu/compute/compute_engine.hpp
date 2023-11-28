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

#ifndef GPU_COMPUTE_COMPUTE_ENGINE_HPP
#define GPU_COMPUTE_COMPUTE_ENGINE_HPP

#include <cassert>
#include <memory>
#include <vector>
#include <initializer_list>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/resource.hpp"
#include "common/verbose.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/compute/dispatch.hpp"
#include "gpu/compute/kernel.hpp"
#include "gpu/compute/kernel_ctx.hpp"
#include "gpu/jit/jit_generator_base.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

class compute_engine_t : public engine_t {
public:
    compute_engine_t(
            engine_kind_t kind, runtime_kind_t runtime_kind, size_t index)
        : engine_t(kind, runtime_kind, index) {}

    virtual status_t init();
    status_t init(const std::vector<uint8_t> &cache_blob);

    const device_info_t *device_info() const { return device_info_.get(); }

    virtual status_t create_kernel(compute::kernel_t *kernel,
            jit::jit_generator_base *jitter,
            const cache_blob_t &cache_blob) const = 0;

    virtual status_t create_kernels(std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx,
            const cache_blob_t &cache_blob) const = 0;

    status_t create_kernel_bundle(kernel_bundle_t &bundle,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx,
            const cache_blob_t &cache_blob = cache_blob_t()) const {
        std::vector<kernel_t> kernels;
        CHECK(create_kernels(&kernels, kernel_names, kernel_ctx, cache_blob));
        bundle = kernel_bundle_t(std::move(kernels), kernel_names);
        return status::success;
    }

    virtual status_t create_kernels_from_ocl_source(
            std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const char *source_string,
            const compute::kernel_ctx_t &kernel_ctx) const {
        assert(!"unexpected");
        return status::success;
    };

    virtual status_t create_kernel_from_binary(compute::kernel_t &kernel,
            const compute::binary_t &binary, const char *kernel_name) const = 0;

    virtual status_t create_kernels_from_cache_blob(
            const cache_blob_t &cache_blob,
            std::vector<compute::kernel_t> &kernels,
            const std::vector<const char *> &kernel_names) const = 0;

    status_t get_zero_pad_primitive(
            primitive_t *&result, const resource_mapper_t *&resources) {
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
        // XXX: XeHPG 128EU A0 causes hangs with large GRF mode.
        if (is_xe_hpg() && device_info()->eu_count() == 128
                && device_info()->stepping_id() == 0
                && device_info()->mayiuse_systolic())
            return false;
        return device_info_->gpu_arch() >= compute::gpu_arch_t::xe_hp;
    }

    dispatch_t create_dispatch(const memory_desc_t *md = nullptr) const {
        return dispatch_t(this, md);
    }

    status_t get_service_stream(stream_t *&stream) override {
        status_t status = status::success;
        if (service_stream_ == nullptr) {
            const std::lock_guard<std::mutex> lock(service_stream_mutex_);
            if (service_stream_ == nullptr) {
                stream_t *service_stream_ptr;
                status = create_stream(
                        &service_stream_ptr, stream_flags::default_flags);
                if (status == status::success)
                    service_stream_.reset(service_stream_ptr);
            }
        }
        stream = service_stream_.get();
        return status;
    }

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
    std::shared_ptr<primitive_t> zero_pad_primitive_;
    resource_mapper_t zero_pad_resources_;
    std::once_flag zero_pad_init_;
    std::unique_ptr<stream_t> service_stream_;
    std::mutex service_stream_mutex_;
};

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

// Exported for testing purposes only.
extern "C" bool DNNL_API dnnl_impl_gpu_mayiuse_ngen_kernels(
        dnnl::impl::engine_t *engine);

#endif
