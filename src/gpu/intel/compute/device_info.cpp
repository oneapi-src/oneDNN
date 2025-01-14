/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#include <cstdint>
#include <thread>
#include <type_traits>

#include "common/type_helpers.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/jit/utils/ngen_type_bridge.hpp"
#include "gpu/intel/utils.hpp"

#ifdef DNNL_WITH_SYCL
#include "gpu/intel/sycl/engine.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
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
        case gpu_arch_t::xe2:
        case gpu_arch_t::xe_hpc:
        case gpu_arch_t::xe3:
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

bool device_info_t::is_integrated() const {
    auto family = static_cast<ngen::ProductFamily>(gpu_product_family_);
    return ngen::getPlatformType(family) == ngen::PlatformType::Integrated;
}

bool device_info_t::mayiuse_sub_group(int size) const {
    switch (gpu_arch()) {
        case gpu_arch_t::gen9:
        case gpu_arch_t::gen11:
        case gpu_arch_t::xe_lp:
        case gpu_arch_t::xe_hp:
        case gpu_arch_t::xe_hpg: return utils::one_of(size, 8, 16, 32);
        default: return utils::one_of(size, 16, 32);
    }
}

bool device_info_t::mayiuse_float_atomic_add(data_type_t type) const {
    switch (type) {
        case data_type::f16: return has_native(native_ext_t::fp16_atomic_add);
        case data_type::f32: return has_native(native_ext_t::fp32_atomic_add);
        case data_type::f64: return has_native(native_ext_t::fp64_atomic_add);
        default: return false;
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
        case data_type::f8_e5m2: return gpu_arch_ >= gpu_arch_t::xe_hpc;
        case data_type::u4:
        case data_type::s4:
        case data_type::f8_e4m3: return false;
        default: return false;
    }
}

int device_info_t::max_eus_per_wg(gpu_arch_t gpu_arch) {
    switch (gpu_arch) {
        case gpu::intel::compute::gpu_arch_t::gen9:
        case gpu::intel::compute::gpu_arch_t::gen11:
        case gpu::intel::compute::gpu_arch_t::xe_hpc:
        case gpu::intel::compute::gpu_arch_t::xe2:
        case gpu::intel::compute::gpu_arch_t::xe3: return 8;
        case gpu::intel::compute::gpu_arch_t::xe_lp:
        case gpu::intel::compute::gpu_arch_t::xe_hp:
        case gpu::intel::compute::gpu_arch_t::xe_hpg: return 16;
        case gpu::intel::compute::gpu_arch_t::unknown: return 8;
    }
    return 8;
}

int device_info_t::max_subgroup_size(gpu_arch_t gpu_arch) {
    switch (gpu_arch) {
        case gpu::intel::compute::gpu_arch_t::gen9: return 16;
        case gpu::intel::compute::gpu_arch_t::gen11:
        case gpu::intel::compute::gpu_arch_t::xe_hpc:
        case gpu::intel::compute::gpu_arch_t::xe2:
        case gpu::intel::compute::gpu_arch_t::xe3: return 32;
        case gpu::intel::compute::gpu_arch_t::xe_lp:
        case gpu::intel::compute::gpu_arch_t::xe_hp:
        case gpu::intel::compute::gpu_arch_t::xe_hpg:
        case gpu::intel::compute::gpu_arch_t::unknown: return 16;
    }
    return 16;
}

int device_info_t::grf_size(gpu_arch_t gpu_arch) {
    ngen::HW hw = jit::convert_dnnl_arch_to_ngen(gpu_arch);
    return ngen::GRF::bytes(hw);
}

int device_info_t::min_subgroup_size() const {
    switch (gpu_arch()) {
        case gpu_arch_t::gen9:
        case gpu_arch_t::gen11:
        case gpu_arch_t::xe_lp:
        case gpu_arch_t::xe_hp:
        case gpu_arch_t::xe_hpg: return 8;
        case gpu_arch_t::xe_hpc:
        case gpu_arch_t::xe2:
        case gpu_arch_t::xe3: return 16;
        default: return 0;
    }
}

int device_info_t::max_exec_size(gpu_arch_t gpu_arch) {
    switch (gpu_arch) {
        case gpu::intel::compute::gpu_arch_t::xe_hpc:
        case gpu::intel::compute::gpu_arch_t::xe2:
        case gpu::intel::compute::gpu_arch_t::xe3: return 128;
        default: return 64;
    }
    return 64;
}

int device_info_t::max_subgroup_size(data_type_t type) const {

    if (type == data_type::undef) { return max_subgroup_size_; }

    return static_cast<int>(std::min((size_t)max_subgroup_size_,
            ((size_t)max_exec_size()) / types::data_type_size(type)));
}

size_t device_info_t::max_wg_size(
        bool large_grf_mode, size_t subgroup_size) const {
    size_t device_max_wg_size
            = large_grf_mode ? max_wg_size_ / 2 : max_wg_size_;
    if (subgroup_size > 0) {
        size_t sg_max_wg_size = threads_per_eu(gpu_arch_, large_grf_mode)
                * max_eus_per_wg_ * subgroup_size;
        return std::min(device_max_wg_size, sg_max_wg_size);
    }
    return device_max_wg_size;
}

int device_info_t::threads_per_eu(gpu_arch_t gpu_arch, bool large_grf_mode) {
    switch (gpu_arch) {
        case gpu::intel::compute::gpu_arch_t::gen9:
        case gpu::intel::compute::gpu_arch_t::gen11:
        case gpu::intel::compute::gpu_arch_t::xe_lp: return 7;
        case gpu::intel::compute::gpu_arch_t::xe_hp:
        case gpu::intel::compute::gpu_arch_t::xe_hpg:
        case gpu::intel::compute::gpu_arch_t::xe_hpc:
        case gpu::intel::compute::gpu_arch_t::xe2:
        case gpu::intel::compute::gpu_arch_t::xe3:
            return large_grf_mode ? 4 : 8;
        case gpu::intel::compute::gpu_arch_t::unknown: return 7;
    }
    return 7;
}

int device_info_t::max_slm_size(gpu_arch_t gpu_arch) {
    int slm_size = 0; // SLM size per SS or DSS.
    switch (gpu_arch) {
        case gpu::intel::compute::gpu_arch_t::gen9:
        case gpu::intel::compute::gpu_arch_t::gen11:
        case gpu::intel::compute::gpu_arch_t::xe_lp:
            slm_size = (1 << 16);
            break;
        case gpu::intel::compute::gpu_arch_t::xe_hp:
        case gpu::intel::compute::gpu_arch_t::xe_hpg:
        case gpu::intel::compute::gpu_arch_t::xe_hpc:
        case gpu::intel::compute::gpu_arch_t::xe2:
        case gpu::intel::compute::gpu_arch_t::xe3: slm_size = (1 << 17); break;
        case gpu::intel::compute::gpu_arch_t::unknown: assert(!"not expected");
    }
    return slm_size;
}

int device_info_t::max_slm_size_per_tg(gpu_arch_t gpu_arch) {
    switch (gpu_arch) {
        case gpu::intel::compute::gpu_arch_t::xe_hp:
        case gpu::intel::compute::gpu_arch_t::xe_hpg: return (1 << 16);
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

size_t device_info_t::icache_size() const {
    switch (gpu_arch_) {
        case gpu::intel::compute::gpu_arch_t::gen9:
        case gpu::intel::compute::gpu_arch_t::gen11:
        case gpu::intel::compute::gpu_arch_t::xe_lp:
        case gpu::intel::compute::gpu_arch_t::xe_hp: return 48 * 1024;
        case gpu::intel::compute::gpu_arch_t::xe_hpg: return 96 * 1024;
        case gpu::intel::compute::gpu_arch_t::xe_hpc: return 80 * 1024;
        case gpu::intel::compute::gpu_arch_t::xe2: return 96 * 1024;
        case gpu::intel::compute::gpu_arch_t::xe3: return 96 * 1024;
        case gpu::intel::compute::gpu_arch_t::unknown: assert(!"not expected");
    }
    return 0;
}

status_t device_info_t::init_attributes_common(impl::engine_t *engine) {
    bool ocl_backend = true;

#ifdef DNNL_WITH_SYCL
    if (engine->runtime_kind() == runtime_kind::sycl) {
        const auto *sycl_engine_impl
                = utils::downcast<const xpu::sycl::engine_impl_t *>(
                        engine->impl());
        ocl_backend
                = (sycl_engine_impl->backend() == xpu::sycl::backend_t::opencl);
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
    serialized_device_info_.write(&gpu_product_family_);
    serialized_device_info_.write(&stepping_id_);
    serialized_device_info_.write(&ip_version_);
    serialized_device_info_.write(&runtime_version_.major);
    serialized_device_info_.write(&runtime_version_.minor);
    serialized_device_info_.write(&runtime_version_.build);
    serialized_device_info_.write(hw_threads_, 2);
    serialized_device_info_.write(&eu_count_);
    serialized_device_info_.write(&max_eus_per_wg_);
    serialized_device_info_.write(&max_subgroup_size_);
    serialized_device_info_.write(&max_exec_size_);
    serialized_device_info_.write(&max_wg_size_);
    serialized_device_info_.write(&l3_cache_size_);
    serialized_device_info_.write(&extensions_);
    serialized_device_info_.write(&native_extensions_);
    serialized_device_info_.write(&mayiuse_systolic_);
    serialized_device_info_.write(&mayiuse_ngen_kernels_);
    serialized_device_info_.write(&mayiuse_system_memory_allocators_);
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
    DESERIALIZE(gpu_product_family_, int);
    DESERIALIZE(stepping_id_, int);
    DESERIALIZE(ip_version_, uint32_t);
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
    DESERIALIZE(l3_cache_size_, size_t);
    DESERIALIZE(extensions_, uint64_t);
    DESERIALIZE(native_extensions_, uint64_t);
    DESERIALIZE(mayiuse_systolic_, bool);
    DESERIALIZE(mayiuse_ngen_kernels_, bool);
    DESERIALIZE(mayiuse_system_memory_allocators_, bool);
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

void device_info_t::fixup_l3_cache_size() {
    // XXX: OpenCL/DPCPP does not report correct cache size for this
    // configuration.
    if (gpu_arch() == gpu_arch_t::xe2 && eu_count() <= 64) {
        l3_cache_size_ = (1 << 23);
    }
}

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
