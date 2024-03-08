/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef GPU_JIT_IR_HW_HPP
#define GPU_JIT_IR_HW_HPP

#include "gpu/compute/compute_engine.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/jit/ir/core.hpp"
#include "gpu/jit/utils/ngen_type_bridge.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

inline std::string to_string(ngen::HW hw) {
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
        CASE(Xe2)
        default: ir_error_not_expected();
    }
#undef CASE
    return "Unexpected";
}

inline ngen::HW str_to_ngen_hw(const std::string &s) {
#define CASE(name) \
    do { \
        auto s_cur = to_string(ngen::HW::name); \
        if (utils::one_of(s, s_cur, ir_utils::to_lower(s_cur))) \
            return ngen::HW::name; \
    } while (false)
    CASE(Unknown);
    CASE(Gen9);
    CASE(Gen10);
    CASE(Gen11);
    CASE(XeLP);
    CASE(XeHP);
    CASE(XeHPG);
    CASE(XeHPC);
    CASE(Xe2);
#undef CASE
    return ngen::HW::Unknown;
}

// Provides access to HW configuration which includes non-configurable
// properties.
class hw_t {
public:
    hw_t() = default;
    explicit hw_t(ngen::HW hw) : hw_(hw) {}
    explicit hw_t(const engine_t *engine) {
        using namespace compute;
        auto compute_engine = utils::downcast<const compute_engine_t *>(engine);

        auto *device_info = compute_engine->device_info();
        gpu_arch_t gpu_arch = device_info->gpu_arch();
        stepping_id_ = device_info->stepping_id();
        eu_count_ = device_info->eu_count();
        max_wg_size_ = static_cast<int>(
                device_info->max_wg_size(/*large_grf_mode=*/false));
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
    int stepping_id() const { return stepping_id_; }
    int eu_count() const { return eu_count_; }
    int large_grf_support() const { return large_grf_support_; }
    int grf_size() const { return ngen::GRF::bytes(hw_); }
    int systolic_support() const { return systolic_support_; }

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

    bool operator==(const hw_t &other) const {
        if (hw_ != other.hw_) return false;
        if (stepping_id_ != other.stepping_id_) return false;
        if (eu_count_ != other.eu_count_) return false;
        if (max_wg_size_ != other.max_wg_size_) return false;
        if (large_grf_support_ != other.large_grf_support_) return false;
        if (systolic_support_ != other.systolic_support_) return false;
        return true;
    }

    bool operator!=(const hw_t &other) const { return !operator==(other); }

    size_t get_hash() const {
        return ir_utils::get_hash(hw_, stepping_id_, eu_count_, max_wg_size_,
                large_grf_support_, systolic_support_);
    }

    void serialize(std::ostream &out) const {
        ir_utils::serialize(hw_, out);
        ir_utils::serialize(stepping_id_, out);
        ir_utils::serialize(eu_count_, out);
        ir_utils::serialize(max_wg_size_, out);
        ir_utils::serialize(large_grf_support_, out);
        ir_utils::serialize(systolic_support_, out);
    }

    void deserialize(std::istream &in) {
        ir_utils::deserialize(hw_, in);
        ir_utils::deserialize(stepping_id_, in);
        ir_utils::deserialize(eu_count_, in);
        ir_utils::deserialize(max_wg_size_, in);
        ir_utils::deserialize(large_grf_support_, in);
        ir_utils::deserialize(systolic_support_, in);
    }

private:
    int max_wg_size(int regs = 128) const {
        bool is_large_grf = (regs > 128);
        return is_large_grf ? max_wg_size_ / 2 : max_wg_size_;
    }

    ngen::HW hw_ = ngen::HW::Unknown;
    int stepping_id_ = -1;
    int eu_count_ = 0;
    int max_wg_size_ = 0;
    bool large_grf_support_ = false;
    bool systolic_support_ = false;
    bool with_atomic_fp64_ = false;
};

inline hw_t str_to_hw(const std::string &s) {
    return hw_t(str_to_ngen_hw(s));
}

class exec_config_t {
public:
    exec_config_t() = default;
    exec_config_t(const hw_t &hw) : hw_(hw) {}

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
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
