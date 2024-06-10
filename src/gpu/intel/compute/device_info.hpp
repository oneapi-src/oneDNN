/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_INTEL_COMPUTE_DEVICE_INFO_HPP
#define GPU_INTEL_COMPUTE_DEVICE_INFO_HPP

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "common/c_types_map.hpp"
#include "common/serialization_stream.hpp"
#include "common/utils.hpp"
#include "common/z_magic.hpp"

#include "xpu/utils.hpp"

#include "oneapi/dnnl/dnnl_config.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

enum class gpu_arch_t {
    unknown,
    gen9,
    gen11,
    xe_lp,
    xe_hp,
    xe_hpg,
    xe_hpc,
    xe2,
};

static inline std::string to_string(gpu_arch_t arch) {
#define CASE(_case) \
    if (arch == gpu_arch_t::_case) return STRINGIFY(_case)
    CASE(gen9);
    CASE(gen11);
    CASE(xe_lp);
    CASE(xe_hp);
    CASE(xe_hpg);
    CASE(xe_hpc);
    CASE(xe2);
    return "unknown";
#undef CASE
}

static inline gpu_arch_t str2gpu_arch(const char *str) {
#define CASE(_case) \
    if (!strcmp(STRINGIFY(_case), str)) return gpu_arch_t::_case

    CASE(gen9);
    CASE(gen11);
    CASE(xe_lp);
    CASE(xe_hp);
    CASE(xe_hpg);
    CASE(xe_hpc);
    CASE(xe2);
    return gpu_arch_t::unknown;
#undef CASE
}

enum class device_ext_t : uint64_t {
    // clang-format off
    // OpenCL data types
    khr_fp16 = 1ull << 0,
    khr_fp64 = 1ull << 1,
    // OpenCL atomics
    khr_global_int32_base_atomics     = 1ull << 2,
    khr_global_int32_extended_atomics = 1ull << 3,
    khr_int64_base_atomics            = 1ull << 4,
    khr_int64_extended_atomics        = 1ull << 5,
    khr_local_int32_base_atomics      = 1ull << 6,
    khr_local_int32_extended_atomics  = 1ull << 7,
    ext_float_atomics                 = 1ull << 8,
    // Intel specific Gen9+
    intel_subgroups              = 1ull << 16,
    intel_required_subgroup_size = 1ull << 17,
    intel_subgroups_char         = 1ull << 18,
    intel_subgroups_short        = 1ull << 19,
    intel_subgroups_long         = 1ull << 20,
    // Intel specific Xe_LP+
    intel_subgroup_local_block_io = 1ull << 21,
    intel_dot_accumulate          = 1ull << 22,
    // Intel specific Xe_HP+
    intel_global_float_atomics                      = 1ull << 23,
    intel_subgroup_matrix_multiply_accumulate       = 1ull << 24,
    intel_subgroup_split_matrix_multiply_accumulate = 1ull << 25,
    intel_variable_eu_thread_count                  = 1ull << 26,
    intel_unified_shared_memory                     = 1ull << 27,
    // Future extensions
    future_bf16_cvt                                 = 1ull << 31,
    last
    // clang-format on
};

static inline const char *ext2cl_str(device_ext_t ext) {
#define CASE(x) \
    case device_ext_t::x: return STRINGIFY(CONCAT2(cl_, x));
    switch (ext) {
        CASE(khr_fp16)
        CASE(khr_fp64)

        CASE(khr_global_int32_base_atomics)
        CASE(khr_global_int32_extended_atomics)
        CASE(khr_int64_base_atomics)
        CASE(khr_int64_extended_atomics)
        CASE(khr_local_int32_base_atomics)
        CASE(khr_local_int32_extended_atomics)
        CASE(ext_float_atomics)

        CASE(intel_subgroups)
        CASE(intel_required_subgroup_size)
        CASE(intel_subgroups_char)
        CASE(intel_subgroups_short)
        CASE(intel_subgroups_long)

        CASE(intel_subgroup_local_block_io)
        CASE(intel_dot_accumulate)

        CASE(intel_global_float_atomics)
        CASE(intel_subgroup_matrix_multiply_accumulate)
        CASE(intel_subgroup_split_matrix_multiply_accumulate)
        CASE(intel_variable_eu_thread_count)
        CASE(intel_unified_shared_memory)
        CASE(future_bf16_cvt)
        default: return nullptr;
    }
#undef CASE
}

enum class native_ext_t : uint64_t {
    // clang-format off
    // OpenCL data types
    fp32_atomic_add = 1ull << 0,                   
    fp32_atomic_min_max = 1ull << 1, 
    fp32_atomic_load_store = 1ull << 2,  
    fp16_atomic_add = 1ull << 3,                   
    fp16_atomic_min_max = 1ull << 4,              
    fp16_atomic_load_store = 1ull << 5,  
    fp64_atomic_add = 1ull << 6,
    fp64_atomic_min_max = 1ull << 7,
    fp64_atomic_load_store = 1ull << 8,  
    last
};

// Needed workaround for future HW extensions
uint64_t get_future_extensions(
        compute::gpu_arch_t gpu_arch, bool mayiuse_systolic);

struct device_info_t {
public:
    virtual ~device_info_t() = default;

    status_t init(
            impl::engine_t *engine, const std::vector<uint8_t> &cache_blob = {}) {
        if (!cache_blob.empty()) {
            CHECK(init_from_cache_blob(cache_blob));
            return init_serialized_device_info(cache_blob);
        }

        CHECK(init_device_name(engine));
        CHECK(init_arch(engine));
        CHECK(init_runtime_version(engine));
        CHECK(init_extensions(engine));
        CHECK(init_attributes(engine));
        fixup_l3_cache_size();

        CHECK(init_attributes_common(engine));

        if (dnnl_version()->gpu_runtime == DNNL_RUNTIME_OCL) {
            CHECK(init_serialized_device_info());
        }

        return status::success;
    }

    bool has(device_ext_t ext) const { return extensions_ & (uint64_t)ext; }
    bool has_native(native_ext_t ext) const { return native_extensions_ & (uint64_t)ext; }
    gpu_arch_t gpu_arch() const { return gpu_arch_; }
    int stepping_id() const { return stepping_id_; }
    uint32_t ip_version() const { return ip_version_; }
    int max_eus_per_wg() const { return max_eus_per_wg_; }
    static int max_eus_per_wg(gpu_arch_t gpu_arch);

    static int max_exec_size(gpu_arch_t gpu_arch);

    int max_exec_size() const { return max_exec_size(gpu_arch()); }
    int max_subgroup_size(data_type_t type = data_type::undef) const;
    static int max_subgroup_size(gpu_arch_t gpu_arch);
    static int grf_size(gpu_arch_t gpu_arch);
    int grf_size() const { return grf_size(gpu_arch_); };
    int min_subgroup_size() const;
    size_t max_wg_size(bool large_grf_mode) const;
    int eu_count() const { return eu_count_; }
    int hw_threads() const { return hw_threads_[0]; }
    int hw_threads(bool large_grf_mode) const {
        return hw_threads_[large_grf_mode ? 1 : 0];
    }
    static int threads_per_eu(gpu_arch_t gpu_arch, bool large_grf_mode = false);
    static int max_slm_size(gpu_arch_t gpu_arch);
    static int max_slm_size_per_tg(gpu_arch_t gpu_arch);
    static int max_slm_size_per_tg(
            gpu_arch_t gpu_arch, int tg_size, bool large_grf_mode = false);
    size_t l3_cache_size() const { return l3_cache_size_; }
    size_t icache_size() const;

    const xpu::runtime_version_t &runtime_version() const {
        return runtime_version_;
    }
    const std::string &name() const { return name_; }

    bool mayiuse_ngen_kernels() const { return mayiuse_ngen_kernels_; }

    bool mayiuse_systolic() const { return mayiuse_systolic_; }

    bool mayiuse_non_uniform_work_groups() const {
        return mayiuse_non_uniform_work_groups_;
    }

    /// Returns true if the engine can directly access pointers from system allocators
    bool mayiuse_system_memory_allocators() const {
        return mayiuse_system_memory_allocators_;
    }

    bool mayiuse_sub_group(int size) const;

    bool mayiuse_float_atomic_add(data_type_t type) const;

    bool has_native(data_type_t type) const;

    const std::vector<uint8_t> &get_cache_blob() const {
        return serialized_device_info_.get_data();
    }

    status_t get_cache_blob_size(size_t *size) const {
        (*size) = serialized_device_info_.get_data().size();
        return status::success;
    }

    status_t get_cache_blob(size_t size, uint8_t *cache_blob) const {
        const auto &cb = serialized_device_info_.get_data();
        if (size != cb.size()) return status::invalid_arguments;
        std::memcpy(cache_blob, cb.data(), size);
        return status::success;
    }

protected:
    virtual status_t init_device_name(impl::engine_t *engine) = 0;
    virtual status_t init_arch(impl::engine_t *engine) = 0;
    virtual status_t init_runtime_version(impl::engine_t *engine) = 0;
    virtual status_t init_extensions(impl::engine_t *engine) = 0;
    virtual status_t init_attributes(impl::engine_t *engine) = 0;

    compute::gpu_arch_t gpu_arch_ = compute::gpu_arch_t::unknown;
    int stepping_id_ = 0;
    uint32_t ip_version_ = 0;
    bool mayiuse_systolic_ = false;
    bool mayiuse_ngen_kernels_ = false;
    bool mayiuse_system_memory_allocators_ = false;

    std::string name_;
    xpu::runtime_version_t runtime_version_;

    // total number of hardware threads:
    // [0] - default mode
    // [1] - large GRF mode
    int32_t hw_threads_[2] = {0, 0};
    int32_t eu_count_ = 0;
    int32_t max_eus_per_wg_ = 0;
    int32_t max_subgroup_size_ = 16;
    int max_exec_size_ = 0;
    size_t max_wg_size_ = 0;
    size_t l3_cache_size_ = 0;

    // extensions_ and gpu_arch_ describe effective extensions and GPU architecture.
    uint64_t extensions_ = 0;
    // native extensions, may differ from support reported by runtime.
    uint64_t native_extensions_ = 0;

private:
    status_t init_attributes_common(impl::engine_t *engine);
    status_t init_serialized_device_info(
            const std::vector<uint8_t> &cache_blob = {});
    status_t init_from_cache_blob(const std::vector<uint8_t> &cache_blob);
    void fixup_l3_cache_size();

    bool mayiuse_non_uniform_work_groups_ = false;

    serialization_stream_t serialized_device_info_;
};

} // namespace compute
}} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
