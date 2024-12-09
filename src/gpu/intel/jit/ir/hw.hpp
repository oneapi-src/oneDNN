/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_IR_HW_HPP
#define GPU_INTEL_JIT_IR_HW_HPP

#include "gpu/intel/compute/compute_engine.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/jit/ir/core.hpp"
#include "gpu/intel/jit/utils/ngen_type_bridge.hpp"
#include "gpu/intel/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

// Provides access to HW configuration which includes non-configurable
// properties.
class hw_t {
public:
    hw_t() = default;
    explicit hw_t(ngen::HW hw) : hw_(hw) {}
    explicit hw_t(const impl::engine_t *engine) {
        using namespace compute;
        auto compute_engine = utils::downcast<const compute_engine_t *>(engine);

        auto *device_info = compute_engine->device_info();
        gpu_arch_t gpu_arch = device_info->gpu_arch();
        product_family_ = static_cast<ngen::ProductFamily>(
                device_info->gpu_product_family());
        stepping_id_ = device_info->stepping_id();
        eu_count_ = device_info->eu_count();
        max_wg_size_ = static_cast<int>(
                device_info->max_wg_size(/*large_grf_mode=*/false));
        l3_cache_size_ = device_info->l3_cache_size();
        large_grf_support_ = compute_engine->mayiuse_large_grf_mode();
        systolic_support_ = device_info->mayiuse_systolic();
        with_atomic_fp64_
                = device_info->mayiuse_float_atomic_add(data_type::f64);

#ifdef DNNL_DEV_MODE
        gpu_arch_t old_arch = gpu_arch;
        gpu_arch = gpu_utils::dev_getenv(
                "gpu_arch", gpu_arch, &eu_count_, &max_wg_size_);
        if (old_arch != gpu_arch)
            large_grf_support_ = gpu_arch >= compute::gpu_arch_t::xe_hp;
#endif

        hw_ = convert_dnnl_arch_to_ngen(gpu_arch);
    }

    bool is_undef() const { return hw_ == ngen::HW::Unknown; }
    bool has_fp64_atomic_support() const { return with_atomic_fp64_; }
    ngen::HW to_ngen() const { return hw_; }
    ngen::ProductFamily product_family() const { return product_family_; }
    int stepping_id() const { return stepping_id_; }
    int eu_count() const { return eu_count_; }
    int large_grf_support() const { return large_grf_support_; }
    int grf_size() const { return ngen::GRF::bytes(hw_); }
    int systolic_support() const { return systolic_support_; }
    size_t l3_cache_size() const { return l3_cache_size_; }

    int max_tg_size(int regs, int simd) const {
        int wg_size = max_wg_size(regs);
        int eu_based_tg_size = eus_per_ss_or_dss()
                * utils::rnd_down_pow2(threads_per_eu(regs));
        int wg_based_tg_size = wg_size / simd;
        return std::min(eu_based_tg_size, wg_based_tg_size);
    }

    // Number of EUs per subslice or dual subslice.
    int eus_per_ss_or_dss() const {
        auto arch = convert_ngen_arch_to_dnnl(hw_);
        return compute::device_info_t::max_eus_per_wg(arch);
    }

    int threads_per_eu(int regs = 128) const {
        auto arch = convert_ngen_arch_to_dnnl(hw_);
        bool is_large_grf = (regs > 128);
        return compute::device_info_t::threads_per_eu(arch, is_large_grf);
    }

    bool prefer_large_grf(const gpu_primitive_attr_t *gpu_attr) const {
        if (!gpu_attr || !large_grf_support_) return false;
        return gpu_attr->threads_per_eu() * 2 == threads_per_eu();
    }

    int cache_line_size() const;

    std::string str() const {
        std::ostringstream oss;
        oss << to_string(hw_);
        oss << ", stepping: " << stepping_id();
        oss << ", EUs: " << eu_count();
        return oss.str();
    }

    std::string brief_str() const { return ir_utils::to_lower(to_string(hw_)); }

    IR_DEFINE_DUMP()

    bool operator<(ngen::HW rhs) const { return hw_ < rhs; }
    bool operator>(ngen::HW rhs) const { return hw_ > rhs; }
    bool operator<=(ngen::HW rhs) const { return hw_ <= rhs; }
    bool operator>=(ngen::HW rhs) const { return hw_ >= rhs; }
    bool operator==(ngen::HW rhs) const { return hw_ == rhs; }
    bool operator!=(ngen::HW rhs) const { return hw_ != rhs; }
#if __cplusplus >= 202002L
    bool operator==(const hw_t &other) const = default;
#endif

private:
    int max_wg_size(int regs = 128) const {
        bool is_large_grf = (regs > 128);
        return is_large_grf ? max_wg_size_ / 2 : max_wg_size_;
    }

    ngen::HW hw_ = ngen::HW::Unknown;
    ngen::ProductFamily product_family_ = ngen::ProductFamily::Unknown;
    int stepping_id_ = -1;
    int eu_count_ = 0;
    int max_wg_size_ = 0;
    size_t l3_cache_size_ = 0;
    bool large_grf_support_ = false;
    bool systolic_support_ = false;
    bool with_atomic_fp64_ = false;
};

class exec_config_t {
public:
    exec_config_t() = default;
    exec_config_t(const hw_t &hw) : hw_(hw) {}
    exec_config_t(const hw_t &hw, int regs, int simd)
        : hw_(hw), regs_(regs), simd_(simd) {}

    const hw_t &hw() const { return hw_; }
    int regs() const { return regs_; }
    int simd() const { return simd_; }
    int vec_size() const { return vec_size_; }
    int grf_size() const { return hw_.grf_size(); }
    void set_regs(int regs) { regs_ = regs; }
    void set_simd(int simd) { simd_ = simd; }
    void set_vec_size(int vec_size) { vec_size_ = vec_size; }

    std::string str() const {
        std::ostringstream oss;
        oss << hw_.str();
        oss << ", SIMD: " << simd();
        if (vec_size() != simd()) oss << " (" << vec_size() << ")";
        oss << ", regs: " << regs();
        return oss.str();
    }

private:
    hw_t hw_;
    int regs_ = 0;
    int simd_ = 0;
    int vec_size_ = 0;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
