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

#include <cstdint>
#include <thread>
#include <type_traits>

#include "common/type_helpers.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/jit/utils/type_bridge.hpp"
#include "gpu/intel/utils.hpp"
#include "ngen.hpp"

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
        case gpu_arch_t::xe_hp:
        case gpu_arch_t::xe_hpg:
        case gpu_arch_t::xe2:
        case gpu_arch_t::xe_hpc:
        case gpu_arch_t::xe3:
        case gpu_arch_t::xe3p:
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

device_info_t::~device_info_t() = default;

status_t device_info_t::init(
        impl::engine_t *engine, const std::vector<uint8_t> &cache_blob) {
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

    if (engine->is_cache_blob_supported()) {
        CHECK(init_serialized_device_info());
    }

    return status::success;
}

ngen::HW device_info_t::ngen_hw() const {
    return ngen::getCore(product_->family);
}

int device_info_t::stepping_id() const {
    return product_->stepping;
}

bool device_info_t::is_integrated() const {
    return product_->type == ngen::PlatformType::Integrated;
}

// LNL drivers before NEO 39167 (Windows 101.8970) miss L1 flushes.
bool device_info_t::has_l1_flush_bug() const {
    if (!product_ || product_->family != ngen::ProductFamily::LNL) return false;
    const auto &v = runtime_version_;
    // Windows OpenCL reports 32.0.101.<revision>.
    if (v.minor == 0 && v.build == 101) return v.revision < 8970;
    return v.build < 39167;
}

std::string device_info_t::get_cl_ext_options() const {
    using namespace compute;

    std::string opts;
    for (uint64_t i_ext = 1; i_ext < (uint64_t)device_ext_t::last;
            i_ext <<= 1) {
        auto ext = (device_ext_t)i_ext;

        // Use real GPU extensions
        if (!has(ext)) continue;

        // These extensions are not handled properly by the OpenCL runtime.
        // Pass macros for them manually.
        if (utils::one_of(ext, device_ext_t::intel_global_float_atomics,
                    device_ext_t::intel_subgroup_matrix_multiply_accumulate,
                    device_ext_t::
                            intel_subgroup_split_matrix_multiply_accumulate,
                    device_ext_t::intel_global_float_atomics,
                    device_ext_t::future_bf16_cvt,
                    device_ext_t::intel_dot_accumulate))
            opts += std::string("-D") + ext2cl_str(ext) + " ";
    }
    return opts;
}

bool device_info_t::mayiuse_sub_group(int size) const {
    switch (gpu_arch()) {
        case gpu_arch_t::xe_lp:
        case gpu_arch_t::xe_hp:
        case gpu_arch_t::xe_hpg: return utils::one_of(size, 8, 16, 32);
        case gpu_arch_t::xe_hpc:
        case gpu_arch_t::xe2:
        case gpu_arch_t::xe3:
        case gpu_arch_t::xe3p: return utils::one_of(size, 16, 32);
        default: return utils::one_of(size, 32);
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
        case gpu::intel::compute::gpu_arch_t::xe_hpc:
        case gpu::intel::compute::gpu_arch_t::xe2:
        case gpu::intel::compute::gpu_arch_t::xe3p:
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
        case gpu::intel::compute::gpu_arch_t::xe_hpc:
        case gpu::intel::compute::gpu_arch_t::xe2:
        case gpu::intel::compute::gpu_arch_t::xe3p:
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
        case gpu_arch_t::xe_lp:
        case gpu_arch_t::xe_hp:
        case gpu_arch_t::xe_hpg: return 8;
        case gpu_arch_t::xe_hpc:
        case gpu_arch_t::xe2:
        case gpu_arch_t::xe3p:
        case gpu_arch_t::xe3: return 16;
        default: return 0;
    }
}

int device_info_t::max_exec_size(gpu_arch_t gpu_arch) {
    switch (gpu_arch) {
        case gpu::intel::compute::gpu_arch_t::xe_hpc:
        case gpu::intel::compute::gpu_arch_t::xe2:
        case gpu::intel::compute::gpu_arch_t::xe3p:
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
        int grf_per_thread, size_t subgroup_size) const {
    // max_wg_size_ implicitly assumes 128 GRF/thread - other GRF modes will vary
    size_t thread_per_eu = threads_per_eu(grf_per_thread);
    size_t base_thread_per_eu = threads_per_eu(128);
    size_t device_max_wg_size
            = max_wg_size_ * thread_per_eu / base_thread_per_eu;
    if (subgroup_size > 0) {
        size_t sg_max_wg_size = thread_per_eu * max_eus_per_wg_ * subgroup_size;
        return std::min(device_max_wg_size, sg_max_wg_size);
    }
    return device_max_wg_size;
}

namespace {
int grf_per_eu(const ngen::Product &product) {
    ngen::HW hw = ngen::getCore(product.family);
    switch (hw) {
        case ngen::HW::XeLP: return 896;
        case ngen::HW::XeHP:
        case ngen::HW::XeHPG:
        case ngen::HW::XeHPC:
        case ngen::HW::Xe2:
        case ngen::HW::Xe3: return 1024;
        case ngen::HW::Xe3p:
            return product.family == ngen::ProductFamily::CRI ? 2048 : 1024;
        case ngen::HW::Unknown: return 896;
        default: gpu_error_not_expected();
    }
    return 1024;
}

int max_threads_per_eu(const ngen::Product &product) {
    ngen::HW hw = ngen::getCore(product.family);
    gpu_arch_t arch = jit::convert_ngen_arch_to_dnnl(hw);
    using namespace gpu::intel::compute;
    switch (arch) {
        case gpu_arch_t::xe_lp: return 7;
        case gpu_arch_t::xe_hp:
        case gpu_arch_t::xe_hpg:
        case gpu_arch_t::xe_hpc:
        case gpu_arch_t::xe2:
        case gpu_arch_t::xe3: return 8;
        case gpu_arch_t::xe3p:
            return product.family == ngen::ProductFamily::NVLP ? 10 : 8;
        case gpu_arch_t::unknown: return 8;
    }
    gpu_error_not_expected();
    return 8;
}
} // namespace

int device_info_t::threads_per_eu(
        const ngen::Product &product, int grf_per_thread) {
    gpu_assert(grf_per_thread > 0) << "Invalid GRF per thread";

    // CRI has an exception: 256 GRF/thread is restricted to 4 threads/EU due to lack of accumulators
    if (product.family == ngen::ProductFamily::CRI && grf_per_thread == 256)
        return 4;

    int hw_max = max_threads_per_eu(product);
    return std::min(hw_max, grf_per_eu(product) / grf_per_thread);
}

int device_info_t::max_slm_size(const ngen::Product &product) {
    auto family = product.family;
    auto gpu_arch = jit::convert_ngen_arch_to_dnnl(ngen::getCore(family));
    int slm_size = 0; // SLM size per SS or DSS.
    switch (gpu_arch) {
        case gpu::intel::compute::gpu_arch_t::xe_lp:
            slm_size = (1 << 16);
            break;
        case gpu::intel::compute::gpu_arch_t::xe_hp:
        case gpu::intel::compute::gpu_arch_t::xe_hpg:
        case gpu::intel::compute::gpu_arch_t::xe_hpc:
        case gpu::intel::compute::gpu_arch_t::xe2:
        case gpu::intel::compute::gpu_arch_t::xe3: slm_size = (1 << 17); break;
        case gpu::intel::compute::gpu_arch_t::xe3p:
            slm_size
                    = (family == ngen::ProductFamily::NVLP) ? 3 << 16 : 3 << 17;
            break;
        case gpu::intel::compute::gpu_arch_t::unknown: assert(!"not expected");
    }
    return slm_size;
}

int device_info_t::max_slm_size_per_tg(const ngen::Product &product) {
    auto gpu_arch
            = jit::convert_ngen_arch_to_dnnl(ngen::getCore(product.family));
    switch (gpu_arch) {
        case gpu::intel::compute::gpu_arch_t::xe_hp:
        case gpu::intel::compute::gpu_arch_t::xe_hpg: return (1 << 16);
        default: return max_slm_size(product);
    }
}

int device_info_t::max_slm_size_per_tg(
        int tg_size, int grf_per_thread, const ngen::Product &product) {
    ngen::HW hw = ngen::getCore(product.family);
    auto gpu_arch = jit::convert_ngen_arch_to_dnnl(hw);
    int eus_per_ss = max_eus_per_wg(gpu_arch);
    int tgs_per_ss
            = eus_per_ss * threads_per_eu(product, grf_per_thread) / tg_size;
    int slm_per_tg = max_slm_size(product) / tgs_per_ss;
    return std::min(max_slm_size_per_tg(product), slm_per_tg);
}

size_t device_info_t::icache_size() const {
    switch (gpu_arch_) {
        case gpu::intel::compute::gpu_arch_t::xe_lp:
        case gpu::intel::compute::gpu_arch_t::xe_hp: return 48 * 1024;
        case gpu::intel::compute::gpu_arch_t::xe_hpg: return 96 * 1024;
        case gpu::intel::compute::gpu_arch_t::xe_hpc: return 80 * 1024;
        case gpu::intel::compute::gpu_arch_t::xe2: return 96 * 1024;
        case gpu::intel::compute::gpu_arch_t::xe3: return 96 * 1024;
        case gpu::intel::compute::gpu_arch_t::xe3p: return 80 * 1024;
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

    max_eus_per_wg_ = max_eus_per_wg(gpu_arch_);
    max_subgroup_size_ = max_subgroup_size(gpu_arch_);
    max_exec_size_ = max_exec_size(gpu_arch_);
    mayiuse_non_uniform_work_groups_ = ocl_backend;

    return status::success;
}

status_t device_info_t::init_serialized_device_info(
        const std::vector<uint8_t> &cache_blob) {
    if (!cache_blob.empty()) {
        serialized_device_info_ = serialization_stream_t::from_data(cache_blob);
        return status::success;
    }

    serialized_device_info_.append(gpu_arch_);
    serialized_device_info_.append(product_ != nullptr);
    if (product_) serialized_device_info_.append(*product_);
    serialized_device_info_.append(ip_version_);
    serialized_device_info_.append(runtime_version_.major);
    serialized_device_info_.append(runtime_version_.minor);
    serialized_device_info_.append(runtime_version_.build);
    serialized_device_info_.append(runtime_version_.revision);
    serialized_device_info_.append(eu_count_);
    serialized_device_info_.append(max_clock_mhz_);
    serialized_device_info_.append(max_eus_per_wg_);
    serialized_device_info_.append(max_subgroup_size_);
    serialized_device_info_.append(max_exec_size_);
    serialized_device_info_.append(max_wg_size_);
    serialized_device_info_.append(memory_size_);
    serialized_device_info_.append(l3_cache_size_);
    serialized_device_info_.append(max_kernel_param_size_);
    serialized_device_info_.append(device_address_bits_);
    serialized_device_info_.append(max_allocation_size_);
    serialized_device_info_.append(extensions_);
    serialized_device_info_.append(native_extensions_);
    serialized_device_info_.append(mayiuse_systolic_);
    serialized_device_info_.append(mayiuse_ngen_kernels_);
    serialized_device_info_.append(mayiuse_system_memory_allocators_);
    serialized_device_info_.append(mayiuse_non_uniform_work_groups_);
    serialized_device_info_.append(is_efficient_64bit_);

    serialized_device_info_.append(name_);

    return status::success;
}

status_t device_info_t::init_from_cache_blob(
        const std::vector<uint8_t> &cache_blob) {
    if (cache_blob.empty()) return status::invalid_arguments;

    auto s = serialization_stream_t::from_data(cache_blob);
    deserializer_t d(s);

    d.pop(gpu_arch_);
    auto hadProduct = d.pop<bool>();
    if (hadProduct) {
        product_ = utils::make_unique<ngen::Product>(d.pop<ngen::Product>());
    }
    d.pop(ip_version_);
    d.pop(runtime_version_.major);
    d.pop(runtime_version_.minor);
    d.pop(runtime_version_.build);
    d.pop(runtime_version_.revision);
    d.pop(eu_count_);
    d.pop(max_clock_mhz_);
    d.pop(max_eus_per_wg_);
    d.pop(max_subgroup_size_);
    d.pop(max_exec_size_);
    d.pop(max_wg_size_);
    d.pop(memory_size_);
    d.pop(l3_cache_size_);
    d.pop(max_kernel_param_size_);
    d.pop(device_address_bits_);
    d.pop(max_allocation_size_);
    d.pop(extensions_);
    d.pop(native_extensions_);
    d.pop(mayiuse_systolic_);
    d.pop(mayiuse_ngen_kernels_);
    d.pop(mayiuse_system_memory_allocators_);
    d.pop(mayiuse_non_uniform_work_groups_);
    d.pop(is_efficient_64bit_);

    d.pop(name_);
    assert(d.empty());

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
