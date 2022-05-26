/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_JIT_CONV_HW_CONFIG_HPP
#define GPU_JIT_CONV_HW_CONFIG_HPP

#include "gpu/compute/compute_engine.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/jit/conv/utils.hpp"
#include "gpu/jit/ngen_type_bridge.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

static std::string to_string(ngen::HW hw) {
#define CASE(name) \
    case ngen::HW::name: return #name;
    switch (hw) {
        CASE(Unknown)
        CASE(Gen9)
        CASE(Gen10)
        CASE(Gen11)
        CASE(XeLP)
        CASE(XeHP)
        CASE(XeHPG)
        CASE(XeHPC)
        default: ir_error_not_expected();
    }
#undef CASE
    return "Unexpected";
}

// Provides access to HW configuration which includes non-configurable
// properties (e.g. EU count, stepping) and configurable properties (e.g. SIMD
// size, number of registers).
class hw_config_t {
public:
    hw_config_t() = default;

    hw_config_t(const engine_t *engine) {
        using namespace compute;
        auto compute_engine = utils::downcast<const compute_engine_t *>(engine);

        auto *device_info = compute_engine->device_info();
        gpu_arch_t gpu_arch = device_info->gpu_arch();
        stepping_id_ = device_info->stepping_id();
        eu_count_ = device_info->eu_count();
        max_wg_size_ = device_info->max_wg_size();
        large_grf_support_ = compute_engine->mayiuse_large_grf_mode();

#ifdef GEN_CONV_DEBUG
        gpu_arch_t old_arch = gpu_arch;
        gpu_arch = ir_utils::getenv_gpu(
                "gpu_arch", gpu_arch, &eu_count_, &max_wg_size_);
        if (old_arch != gpu_arch)
            large_grf_support_ = gpu_arch >= compute::gpu_arch_t::xe_hp;
#endif

        hw_ = convert_dnnl_arch_to_ngen(gpu_arch);
    }

    ngen::HW hw() const { return hw_; }
    int stepping_id() const { return stepping_id_; }
    int eu_count() const { return eu_count_; }
    int large_grf_support() const { return large_grf_support_; }
    int simd_size() const { return simd_size_; }
    int vec_size() const { return vec_size_; }
    int regs() const { return regs_; }
    int grf_size() const { return ngen::GRF::bytes(hw_); }

    void set_simd_size(int value) { simd_size_ = value; }
    void set_vec_size(int value) { vec_size_ = value; }
    void set_regs(int value) { regs_ = value; }
    void set_max_tg_size(int value) { max_tg_size_ = value; }

    int max_tg_size() const {
        if (max_tg_size_ != 0) return max_tg_size_;

        const compute::gpu_arch_t arch = convert_ngen_arch_to_dnnl(hw_);
        const int max_eus_per_wg = compute::device_info_t::max_eus_per_wg(arch);
        const int threads_per_eu
                = compute::device_info_t::threads_per_eu(arch, regs_ > 128);
        const int wg_per_thr = simd_size_
                * compute::device_info_t::threads_per_eu(arch) / threads_per_eu;

        // Optimal thread group size may differ from hardware thread count due
        // to simd_size used in computation.
        return std::min(max_eus_per_wg * utils::rnd_down_pow2(threads_per_eu),
                static_cast<int>(max_wg_size_ / wg_per_thr));
    }

    int max_slm_size() const {
        return compute::device_info_t::max_slm_size_per_tg(
                convert_ngen_arch_to_dnnl(hw_), regs_ > 128);
    }

    std::string str() const {
        using namespace ir_utils;
        std::ostringstream oss;
        oss << to_string(hw_);
        oss << ", stepping: " << stepping_id();
        oss << ", EUs: " << eu_count();
        oss << ", max TG: " << max_tg_size();
        oss << ", SIMD: " << simd_size();
        if (vec_size() != simd_size()) oss << " (" << vec_size() << ")";
        oss << ", regs: " << regs();
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    ngen::HW hw_ = ngen::HW::Unknown;
    int stepping_id_;
    int eu_count_;
    size_t max_wg_size_;
    bool large_grf_support_;

    // SIMD width (used for thread dispatching).
    int simd_size_ = 0;

    // Vector size - used as the execution size for compute instructions
    // (mad/dp4a/dpas).
    int vec_size_ = 0;

    int regs_ = 0; // Number of registers.
    int max_tg_size_ = 0; // Max thread group size.
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
