/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include <mutex>
#include <thread>

#include "gpu/compute/device_info.hpp"

#include "common/verbose.hpp"
#include "gpu/jit/binary_format.hpp"

#ifdef DNNL_WITH_SYCL
#include "sycl/sycl_engine_base.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

uint64_t get_future_extensions(compute::gpu_arch_t gpu_arch) {
    using namespace compute;

    uint64_t extensions = 0;
    switch (gpu_arch) {
        case gpu_arch_t::gen9:
        case gpu_arch_t::gen11: break;
        case gpu_arch_t::xe_hp:
        case gpu_arch_t::xe_hpg:
        case gpu_arch_t::xe_hpc:
            extensions |= (uint64_t)device_ext_t::intel_global_float_atomics;
            extensions |= (uint64_t)
                    device_ext_t::intel_subgroup_matrix_multiply_accumulate;
            extensions |= (uint64_t)device_ext_t::
                    intel_subgroup_split_matrix_multiply_accumulate;
            extensions
                    |= (uint64_t)device_ext_t::intel_variable_eu_thread_count;
            extensions |= (uint64_t)device_ext_t::future_bf16_cvt;
        case gpu_arch_t::xe_lp:
            extensions |= (uint64_t)device_ext_t::intel_subgroup_local_block_io;
            extensions |= (uint64_t)device_ext_t::intel_dot_accumulate;
            break;
        case gpu_arch_t::unknown: break;
    }
    return extensions;
}

bool device_info_t::mayiuse_ngen_kernels(engine_t *engine) {
    static std::mutex m;
    std::lock_guard<std::mutex> guard(m);

    if (checked_ngen_kernels_) return mayiuse_ngen_kernels_;

    auto status
            = jit::gpu_supports_binary_format(&mayiuse_ngen_kernels_, engine);
    if (status != status::success) mayiuse_ngen_kernels_ = false;

    if (get_verbose())
        printf("onednn_verbose,info,gpu,binary_kernels:%s\n",
                mayiuse_ngen_kernels_ ? "enabled" : "disabled");

    checked_ngen_kernels_ = true;

    return mayiuse_ngen_kernels_;
}

bool device_info_t::mayiuse_sub_group(int size) const {
    switch (gpu_arch()) {
        case gpu_arch_t::xe_hpc: return utils::one_of(size, 16, 32);
        default: return utils::one_of(size, 8, 16, 32);
    }
}

int device_info_t::max_eus_per_wg(gpu_arch_t gpu_arch) {
    switch (gpu_arch) {
        case gpu::compute::gpu_arch_t::gen9:
        case gpu::compute::gpu_arch_t::gen11:
        case gpu::compute::gpu_arch_t::xe_hpc: return 8;
        case gpu::compute::gpu_arch_t::xe_lp:
        case gpu::compute::gpu_arch_t::xe_hp:
        case gpu::compute::gpu_arch_t::xe_hpg: return 16;
        case gpu::compute::gpu_arch_t::unknown: return 8;
    }
    return 8;
}

int device_info_t::max_subgroup_size(gpu_arch_t gpu_arch) {
    switch (gpu_arch) {
        case gpu::compute::gpu_arch_t::gen9: return 16;
        case gpu::compute::gpu_arch_t::gen11:
        case gpu::compute::gpu_arch_t::xe_hpc: return 32;
        case gpu::compute::gpu_arch_t::xe_lp:
        case gpu::compute::gpu_arch_t::xe_hp:
        case gpu::compute::gpu_arch_t::xe_hpg:
        case gpu::compute::gpu_arch_t::unknown: return 16;
    }
    return 16;
}

int device_info_t::threads_per_eu(gpu_arch_t gpu_arch, bool large_grf_mode) {
    switch (gpu_arch) {
        case gpu::compute::gpu_arch_t::gen9:
        case gpu::compute::gpu_arch_t::gen11:
        case gpu::compute::gpu_arch_t::xe_lp: return 7;
        case gpu::compute::gpu_arch_t::xe_hp:
        case gpu::compute::gpu_arch_t::xe_hpg:
        case gpu::compute::gpu_arch_t::xe_hpc: return large_grf_mode ? 4 : 8;
        case gpu::compute::gpu_arch_t::unknown: return 7;
    }
    return 7;
}

int device_info_t::max_slm_size_per_tg(
        gpu_arch_t gpu_arch, bool large_grf_mode) {
    int slm_size = 0; // SLM size per SS or DSS.
    switch (gpu_arch) {
        case gpu::compute::gpu_arch_t::gen9:
        case gpu::compute::gpu_arch_t::gen11: slm_size = (1 << 16); break;
        case gpu::compute::gpu_arch_t::xe_lp:
        case gpu::compute::gpu_arch_t::xe_hp:
        case gpu::compute::gpu_arch_t::xe_hpc:
        case gpu::compute::gpu_arch_t::xe_hpg: slm_size = (1 << 17); break;
        case gpu::compute::gpu_arch_t::unknown: assert(!"not expected");
    }
    return slm_size / threads_per_eu(gpu_arch, large_grf_mode);
}

int device_info_t::slm_memory_bank_count(gpu_arch_t gpu_arch) {
    switch (gpu_arch) {
        case gpu::compute::gpu_arch_t::gen9:
        case gpu::compute::gpu_arch_t::gen11:
        case gpu::compute::gpu_arch_t::xe_lp: return 16;
        case gpu::compute::gpu_arch_t::xe_hp: return 65;
        case gpu::compute::gpu_arch_t::xe_hpc: return 64;
        case gpu::compute::gpu_arch_t::xe_hpg: return 32;
        case gpu::compute::gpu_arch_t::unknown: assert(!"not expected");
    }
    return 32;
}
// Returns SLM bank granularity in bytes.
int device_info_t::slm_memory_bank_granularity(gpu_arch_t gpu_arch) {
    switch (gpu_arch) {
        case gpu::compute::gpu_arch_t::gen9:
        case gpu::compute::gpu_arch_t::gen11:
        case gpu::compute::gpu_arch_t::xe_lp:
        case gpu::compute::gpu_arch_t::xe_hp: return 4;
        case gpu::compute::gpu_arch_t::xe_hpc:
        case gpu::compute::gpu_arch_t::xe_hpg: return 8;
        case gpu::compute::gpu_arch_t::unknown: assert(!"not expected");
    }
    return 4;
}

status_t device_info_t::init_attributes_common(engine_t *engine) {
    // TODO: Fix for discrete GPUs. The code below is written for
    // integrated GPUs assuming that last-level cache for GPU is shared
    // with CPU.
    // Integrated GPUs share LLC with CPU which is L3 cache on CPU.

    // XXX: this is the only place where GPU runtime functionally depends on
    // CPU runtime. The `llc_cache_size_` is used only in one kernel for gen9.
    // The idea is to use approximate cache size.

    // llc_cache_size_ = cpu::platform::get_per_core_cache_size(3)
    //        * cpu::platform::get_num_cores();
    // Assumption is that HT is likely enabled on client systems.
    llc_cache_size_ = std::thread::hardware_concurrency() * (1 << 20);

    bool ocl_backend = true;

#ifdef DNNL_WITH_SYCL
    using namespace impl::sycl;
    if (engine->runtime_kind() == runtime_kind::sycl) {
        auto *sycl_engine = utils::downcast<const sycl_engine_base_t *>(engine);
        ocl_backend = (sycl_engine->backend() == backend_t::opencl);
    }
#endif

    hw_threads_[0] = eu_count_ * threads_per_eu(gpu_arch_, false);
    hw_threads_[1] = eu_count_ * threads_per_eu(gpu_arch_, true);

    max_eus_per_wg_ = max_eus_per_wg(gpu_arch_);
    max_subgroup_size_ = max_subgroup_size(gpu_arch_);

    mayiuse_non_uniform_work_groups_ = ocl_backend;

    return status::success;
}

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl
