/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "gpu/compute/device_info.hpp"

#include "common/verbose.hpp"
#include "cpu/platform.hpp"
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
        case gpu_arch_t::gen12lp:
            extensions |= (uint64_t)device_ext_t::intel_dot_accumulate;
            break;
        default: break;
    }
    return extensions;
}

inline gpu_arch_t str2gpu_arch(const char *str) {
#define CASE(_case) \
    if (!strcmp(STRINGIFY(_case), str)) return gpu_arch_t::_case

    CASE(gen9);
    CASE(gen12lp);
    return gpu_arch_t::unknown;
#undef CASE
}

bool device_info_t::mayiuse_ngen_kernels(engine_t *engine) {
    static std::mutex m;
    std::lock_guard<std::mutex> guard(m);

    if (checked_ngen_kernels_) return mayiuse_ngen_kernels_;

    auto status
            = jit::gpu_supports_binary_format(&mayiuse_ngen_kernels_, engine);
    if (status != status::success) mayiuse_ngen_kernels_ = false;

    if (get_verbose())
        printf("dnnl_verbose,info,gpu,binary_kernels:%s\n",
                mayiuse_ngen_kernels_ ? "enabled" : "disabled");

    checked_ngen_kernels_ = true;

    return mayiuse_ngen_kernels_;
}

status_t device_info_t::init_attributes_common(engine_t *engine) {
    // TODO: Fix for discrete GPUs. The code below is written for
    // integrated GPUs assuming that last-level cache for GPU is shared
    // with CPU.
    // Integrated GPUs share LLC with CPU which is L3 cache on CPU.
    llc_cache_size_ = cpu::platform::get_per_core_cache_size(3)
            * cpu::platform::get_num_cores();

    // Assume 7 threads by default
    int32_t threads_per_eu = 7;
    switch (gpu_arch_) {
        case gpu::compute::gpu_arch_t::gen9:
        case gpu::compute::gpu_arch_t::gen12lp: threads_per_eu = 7; break;
        default: break;
    }

    hw_threads_ = eu_count_ * threads_per_eu;

    mayiuse_non_uniform_work_groups_ = true;
#ifdef DNNL_WITH_SYCL
    if (engine->runtime_kind() == runtime_kind::sycl) {
        auto *sycl_engine
                = utils::downcast<const sycl::sycl_engine_base_t *>(engine);
        // Level Zero backend does not support non-uniform work-groups.
        mayiuse_non_uniform_work_groups_
                = (sycl_engine->backend() == sycl::backend_t::opencl);
    }
#endif

    return status::success;
}

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl
