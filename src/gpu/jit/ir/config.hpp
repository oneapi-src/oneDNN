/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_JIT_IR_CONFIG_HPP
#define GPU_JIT_IR_CONFIG_HPP

#include <iostream>
#include <sstream>
#include <unordered_map>

#include "gpu/config.hpp"
#include "gpu/jit/ir/hw.hpp"
#include "gpu/jit/ir/post_ops.hpp"
#include "gpu/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class layout_param_t : public dnnl::impl::gpu::param_t {
public:
    const layout_t &user() const { return user_; }
    const layout_t &compute() const { return compute_; }
    const layout_t &user_unnormalized() const { return user_unnormalized_; }
    const layout_t &compute_unnormalized() const {
        return compute_unnormalized_;
    }

    const std::string &user_unnormalized_tag() const {
        return user_unnormalized_tag_;
    }
    const std::string &compute_unnormalized_tag() const {
        return compute_unnormalized_tag_;
    }

    void set_from_str(const std::string &s) override {
        auto parts = gpu_utils::split(s, ".");
        switch ((int)parts.size()) {
            case 1:
                compute_unnormalized_tag_ = parts[0];
                user_unnormalized_tag_ = parts[0];
                break;
            case 2:
                compute_unnormalized_tag_ = parts[0];
                user_unnormalized_tag_ = parts[1];
                break;
            default: ir_error_not_expected();
        }
    }

    void set_user(const layout_t &l) { user_ = l; }
    void set_compute(const layout_t &l) { compute_ = l; }
    void set_user_unnormalized(const layout_t &l, const std::string &tag) {
        user_unnormalized_ = l;
        user_unnormalized_tag_ = tag;
    }
    void set_compute_unnormalized(const layout_t &l, const std::string &tag) {
        compute_unnormalized_ = l;
        compute_unnormalized_tag_ = tag;
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << short_name() << "=";
        oss << compute_unnormalized_tag_;
        if (user_unnormalized_tag_ != compute_unnormalized_tag_)
            oss << "." << user_unnormalized_tag_;
        return oss.str();
    }

private:
    layout_t user_;
    layout_t compute_;
    layout_t user_unnormalized_;
    layout_t compute_unnormalized_;
    std::string user_unnormalized_tag_;
    std::string compute_unnormalized_tag_;
};

class src_layout_param_t : public layout_param_t {
public:
    std::string name() const override { return "src"; }
    std::string desc() const override { return "Source layout."; }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }
};

class dst_layout_param_t : public layout_param_t {
    std::string name() const override { return "dst"; }
    std::string desc() const override { return "Destination layout."; }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }
};

class exec_cfg_param_t : public value_param_t<exec_config_t> {
public:
    using value_param_t::is_overridden;
    using value_param_t::value_param_t;

    std::string name() const override { return "exec-cfg"; }
    std::string desc() const override {
        return "Execution config (hardware config, number of registers, SIMD, "
               "etc).";
    }
    bool is_overridable() const override { return true; }
    bool is_default(const std::string &key) const override {
        if (key == "regs") return false;
        if (key == "simd") return false;
        if (key == "vec") return value_.vec_size() == value_.simd();
        ir_error_not_expected() << key;
        return false;
    }

    std::vector<std::string> accepted_keys() const override {
        std::vector<std::string> ret;
        ret.push_back("regs");
        ret.push_back("simd");
        ret.push_back("vec");
        return ret;
    }

    void set_from_str(
            const std::string &key, const std::string &value) override {
        if (key == "regs") {
            value_.set_regs(std::stoi(value));
        } else if (key == "simd") {
            value_.set_simd(std::stoi(value));
        } else if (key == "vec") {
            value_.set_vec_size(std::stoi(value));
        } else {
            ir_error_not_expected() << key;
        }
    }

    std::string str(const std::string &key) const override {
        std::ostringstream oss;
        if (key == "regs") {
            oss << "regs=" << value_.regs();
        } else if (key == "simd") {
            oss << "simd=" << value_.simd();
        } else if (key == "vec") {
            if (!is_default("vec")) oss << "vec=" << value_.vec_size();
        }
        return oss.str();
    }
};

class grid_param_t : public value_param_t<grid_info_t> {
public:
    using value_param_t::value_param_t;

    bool is_overridable() const override { return false; }
};

class kernel_grid_param_t : public grid_param_t {
public:
    std::string name() const override { return "kernel-grid"; }
    std::string desc() const override {
        return "Number of thread groups across dimensions (kernel grid).";
    }
    bool is_overridable() const override { return false; }
};

class thread_group_grid_param_t : public grid_param_t {
public:
    std::string name() const override { return "tg-grid"; }
    std::string desc() const override { return "Thread group grid."; }
    bool is_overridable() const override { return false; }
};

class prim_config_t : public container_config_t {
public:
    ~prim_config_t() override = default;
    std::string str() const override = 0;

    void set_zp_cfg(const zero_points_config_t &zp_cfg) { zp_cfg_ = zp_cfg; }
    const zero_points_config_t &zp_cfg() const { return zp_cfg_; }

    // Return thread utilization as a percentage. If this value is low,
    // parallelism is a fundamental limitation to the current work scheduling.
    static float get_thread_utilization(
            const exec_config_t &exec_cfg, int kg_elems, int tg_elems) {
        auto arch = convert_ngen_arch_to_dnnl(exec_cfg.hw().to_ngen());
        int eus_per_slice = compute::device_info_t::max_eus_per_wg(arch);
        int slice_count = exec_cfg.hw().eu_count() / eus_per_slice;

        int min_wg_per_slice_wave = std::max(eus_per_slice / tg_elems, 1);
        int min_wg_per_wave = slice_count * min_wg_per_slice_wave;
        return (100.f * kg_elems) / utils::rnd_up(kg_elems, min_wg_per_wave);
    }

    // Return wave utilization as a percentage. If this value is low, memory
    // latency may be an issue due to limited use of SMT to hide the latency.
    static float get_wave_utilization(
            const exec_config_t &exec_cfg, int kg_elems, int tg_elems) {
        auto arch = convert_ngen_arch_to_dnnl(exec_cfg.hw().to_ngen());
        int threads_per_eu = compute::device_info_t::threads_per_eu(
                arch, exec_cfg.regs() > 128);
        int eus_per_slice = compute::device_info_t::max_eus_per_wg(arch);
        int slice_count = exec_cfg.hw().eu_count() / eus_per_slice;

        int wgs_per_slice = eus_per_slice * threads_per_eu / tg_elems;
        ir_assert(wgs_per_slice > 0);
        int wgs_per_tile = slice_count * wgs_per_slice;
        return (100.f * kg_elems) / utils::rnd_up(kg_elems, wgs_per_tile);
    }

#define DECL_PARAM(name) \
    const name##_param_t &name##_param() const { \
        (void)name##_init_; \
        ir_assert(!name##_.is_undef()); \
        return name##_; \
    } \
    name##_param_t &name##_param() { return name##_; } \
    const name##_param_t::value_t &name() const { \
        ir_assert(!name##_.is_undef()); \
        return name##_.get(); \
    } \
    void set_##name(const name##_param_t::value_t &value) { \
        name##_.set(value); \
    }
#define DECL_PARAM2(name) \
    const name##_param_t &name() const { \
        (void)name##_init_; \
        ir_assert(!name##_.is_undef()); \
        return name##_; \
    } \
    name##_param_t &name() { return name##_; }
    DECL_PARAM(exec_cfg)
    DECL_PARAM2(src_layout)
    DECL_PARAM2(dst_layout)
#undef DECL_PARAM
#undef DECL_PARAM2

    int sort_key(const param_t *param) const override;

protected:
    zero_points_config_t zp_cfg_;

#define INIT_PARAM(name) \
    name##_param_t name##_; \
    param_init_t name##_init_ \
            = register_param([](const container_config_t *c) { \
                  return &static_cast<const prim_config_t *>(c)->name##_; \
              });
    INIT_PARAM(exec_cfg)
    INIT_PARAM(src_layout)
    INIT_PARAM(dst_layout)
#undef INIT_PARAM
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
