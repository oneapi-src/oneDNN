/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef GPU_JIT_IR_HW_CONFIG_HPP
#define GPU_JIT_IR_HW_CONFIG_HPP

#include "gpu/compute/compute_engine.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/jit/ir/core.hpp"
#include "gpu/jit/utils/ngen_type_bridge.hpp"
#include "gpu/jit/utils/utils.hpp"

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
// properties.
class hw_config_t {
public:
    hw_config_t() = default;

    hw_config_t(const engine_t *engine, bool large_grf_mode) {
        using namespace compute;
        auto compute_engine = utils::downcast<const compute_engine_t *>(engine);

        auto *device_info = compute_engine->device_info();
        gpu_arch_t gpu_arch = device_info->gpu_arch();
        stepping_id_ = device_info->stepping_id();
        eu_count_ = device_info->eu_count();
        max_wg_size_
                = static_cast<int>(device_info->max_wg_size(large_grf_mode));
        large_grf_support_ = compute_engine->mayiuse_large_grf_mode();
        systolic_support_ = device_info->mayiuse_systolic();

#ifdef DNNL_DEV_MODE
        gpu_arch_t old_arch = gpu_arch;
        gpu_arch = gpu_utils::dev_getenv(
                "gpu_arch", gpu_arch, &eu_count_, &max_wg_size_);
        if (old_arch != gpu_arch)
            large_grf_support_ = gpu_arch >= compute::gpu_arch_t::xe_hp;
#endif

        hw_ = convert_dnnl_arch_to_ngen(gpu_arch);
    }

    ngen::HW hw() const { return hw_; }
    int stepping_id() const { return stepping_id_; }
    int eu_count() const { return eu_count_; }
    int max_wg_size() const { return max_wg_size_; }
    int large_grf_support() const { return large_grf_support_; }
    int grf_size() const { return ngen::GRF::bytes(hw_); }
    int systolic_support() const { return systolic_support_; }

    std::string str() const {
        std::ostringstream oss;
        oss << to_string(hw_);
        oss << ", stepping: " << stepping_id();
        oss << ", EUs: " << eu_count();
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    ngen::HW hw_ = ngen::HW::Unknown;
    int stepping_id_;
    int eu_count_;
    int max_wg_size_;
    bool large_grf_support_;
    bool systolic_support_;
};

class exec_config_t {
public:
    exec_config_t() = default;
    exec_config_t(const hw_config_t &hw_cfg) : hw_cfg_(hw_cfg) {}

    const hw_config_t &hw_cfg() const { return hw_cfg_; }
    int regs() const { return regs_; }
    int simd() const { return simd_; }
    int vec_size() const { return vec_size_; }

    ngen::HW hw() const { return hw_cfg_.hw(); }
    int grf_size() const { return hw_cfg_.grf_size(); }

    void set_regs(int regs) { regs_ = regs; }
    void set_simd(int simd) { simd_ = simd; }
    void set_vec_size(int vec_size) { vec_size_ = vec_size; }

    std::string str() const {
        std::ostringstream oss;
        oss << hw_cfg_.str();
        oss << ", SIMD: " << simd();
        if (vec_size() != simd()) oss << " (" << vec_size() << ")";
        oss << ", regs: " << regs();
        return oss.str();
    }

private:
    hw_config_t hw_cfg_;
    int regs_ = 0;
    int simd_ = 0;
    int vec_size_ = 0;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
