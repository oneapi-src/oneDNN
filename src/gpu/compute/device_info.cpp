/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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
#include <type_traits>

#include "gpu/compute/device_info.hpp"

#ifdef DNNL_WITH_SYCL
#include "sycl/sycl_engine_base.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

uint64_t get_future_extensions(
        compute::gpu_arch_t gpu_arch, bool mayiuse_systolic) {
    using namespace compute;

    uint64_t extensions = 0;
    switch (gpu_arch) {
        case gpu_arch_t::gen9:
        case gpu_arch_t::gen11: break;
        case gpu_arch_t::xe_hp:
        case gpu_arch_t::xe_hpg:
        case gpu_arch_t::xe_hpc:
            extensions |= (uint64_t)device_ext_t::intel_global_float_atomics;
            extensions
                    |= (uint64_t)device_ext_t::intel_variable_eu_thread_count;
        case gpu_arch_t::xe_lp:
            extensions |= (uint64_t)device_ext_t::intel_subgroup_local_block_io;
            extensions |= (uint64_t)device_ext_t::intel_dot_accumulate;
            break;
        case gpu_arch_t::unknown: break;
    }
    if (mayiuse_systolic) {
        extensions |= (uint64_t)
                device_ext_t::intel_subgroup_matrix_multiply_accumulate;
        extensions |= (uint64_t)
                device_ext_t::intel_subgroup_split_matrix_multiply_accumulate;
        extensions |= (uint64_t)device_ext_t::future_bf16_cvt;
    }
    return extensions;
}

bool device_info_t::mayiuse_sub_group(int size) const {
    switch (gpu_arch()) {
        case gpu_arch_t::xe_hpc: return utils::one_of(size, 16, 32);
        default: return utils::one_of(size, 8, 16, 32);
    }
}

bool device_info_t::has_native(data_type_t type) const {
    switch (type) {
        case data_type::undef:
        case data_type::u8:
        case data_type::s8:
        case data_type::s32:
        case data_type::f16:
        case data_type::f32:
        case data_type::boolean: return true;
        case data_type::f64: return has(device_ext_t::khr_fp64);
        case data_type::bf16: return has(device_ext_t::future_bf16_cvt);
        default: return false;
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

int device_info_t::max_exec_size(gpu_arch_t gpu_arch) {

    switch (gpu_arch) {
        case gpu::compute::gpu_arch_t::xe_hpc: return 128;
        default: return 64;
    }
    return 64;
}

int device_info_t::max_subgroup_size(data_type_t type) const {

    if (type == data_type::undef) { return max_subgroup_size_; }

    return static_cast<int>(std::min((size_t)max_subgroup_size_,
            ((size_t)max_exec_size()) / types::data_type_size(type)));
}

size_t device_info_t::max_wg_size(bool large_grf_mode) const {
    return large_grf_mode ? max_wg_size_ / 2 : max_wg_size_;
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

int device_info_t::max_slm_size(gpu_arch_t gpu_arch) {
    int slm_size = 0; // SLM size per SS or DSS.
    switch (gpu_arch) {
        case gpu::compute::gpu_arch_t::gen9:
        case gpu::compute::gpu_arch_t::gen11:
        case gpu::compute::gpu_arch_t::xe_lp: slm_size = (1 << 16); break;
        case gpu::compute::gpu_arch_t::xe_hp:
        case gpu::compute::gpu_arch_t::xe_hpc:
        case gpu::compute::gpu_arch_t::xe_hpg: slm_size = (1 << 17); break;
        case gpu::compute::gpu_arch_t::unknown: assert(!"not expected");
    }
    return slm_size;
}

int device_info_t::max_slm_size_per_tg(gpu_arch_t gpu_arch) {
    switch (gpu_arch) {
        case gpu::compute::gpu_arch_t::xe_hp:
        case gpu::compute::gpu_arch_t::xe_hpg: return (1 << 16);
        default: return max_slm_size(gpu_arch);
    }
}

int device_info_t::max_slm_size_per_tg(
        gpu_arch_t gpu_arch, int tg_size, bool large_grf_mode) {
    int eus_per_ss = max_eus_per_wg(gpu_arch);
    int tgs_per_ss
            = eus_per_ss * threads_per_eu(gpu_arch, large_grf_mode) / tg_size;
    int slm_per_tg = max_slm_size(gpu_arch) / tgs_per_ss;
    return std::min(max_slm_size_per_tg(gpu_arch), slm_per_tg);
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
    max_exec_size_ = max_exec_size(gpu_arch_);
    mayiuse_non_uniform_work_groups_ = ocl_backend;

    return status::success;
}

status_t device_info_t::init_serialized_device_info(
        const std::vector<uint8_t> &cache_blob) {
    if (!cache_blob.empty()) {
        serialized_device_info_.write(cache_blob.data(), cache_blob.size());
        return status::success;
    }

    serialized_device_info_.write(&gpu_arch_);
    serialized_device_info_.write(&stepping_id_);
    serialized_device_info_.write(&runtime_version_.major);
    serialized_device_info_.write(&runtime_version_.minor);
    serialized_device_info_.write(&runtime_version_.build);
    serialized_device_info_.write(hw_threads_, 2);
    serialized_device_info_.write(&eu_count_);
    serialized_device_info_.write(&max_eus_per_wg_);
    serialized_device_info_.write(&max_subgroup_size_);
    serialized_device_info_.write(&max_exec_size_);
    serialized_device_info_.write(&max_wg_size_);
    serialized_device_info_.write(&llc_cache_size_);
    serialized_device_info_.write(&extensions_);
    serialized_device_info_.write(&mayiuse_systolic_);
    serialized_device_info_.write(&mayiuse_ngen_kernels_);
    serialized_device_info_.write(&mayiuse_non_uniform_work_groups_);

    const size_t name_size = name_.size();
    serialized_device_info_.write(&name_size);
    serialized_device_info_.write(name_.data(), name_size);

    return status::success;
}

status_t device_info_t::init_from_cache_blob(
        const std::vector<uint8_t> &cache_blob) {
    if (cache_blob.empty()) return status::invalid_arguments;

    size_t pos = 0;
#define DESERIALIZE(val, expected_type) \
    static_assert(std::is_same<std::remove_reference<decltype(val)>::type, \
                          expected_type>::value, \
            #val " has incorrect type"); \
    (val) = *reinterpret_cast<const expected_type *>(cache_blob.data() + pos); \
    pos += sizeof(expected_type);

    DESERIALIZE(gpu_arch_, compute::gpu_arch_t);
    DESERIALIZE(stepping_id_, int);
    DESERIALIZE(runtime_version_.major, int);
    DESERIALIZE(runtime_version_.minor, int);
    DESERIALIZE(runtime_version_.build, int);
    DESERIALIZE(hw_threads_[0], int32_t);
    DESERIALIZE(hw_threads_[1], int32_t);
    DESERIALIZE(eu_count_, int32_t);
    DESERIALIZE(max_eus_per_wg_, int32_t);
    DESERIALIZE(max_subgroup_size_, int32_t);
    DESERIALIZE(max_exec_size_, int);
    DESERIALIZE(max_wg_size_, size_t);
    DESERIALIZE(llc_cache_size_, size_t);
    DESERIALIZE(extensions_, uint64_t);
    DESERIALIZE(mayiuse_systolic_, bool);
    DESERIALIZE(mayiuse_ngen_kernels_, bool);
    DESERIALIZE(mayiuse_non_uniform_work_groups_, bool);
#undef DESERIALIZE

    // name_ is not trivially copyable type
    const size_t name_size
            = *reinterpret_cast<const size_t *>(cache_blob.data() + pos);
    pos += sizeof(size_t);
    name_ = std::string(
            reinterpret_cast<const char *>(cache_blob.data() + pos), name_size);
    pos += name_size;
    assert(name_size == name_.size());
    assert(pos == cache_blob.size());

    return status::success;
}

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl
