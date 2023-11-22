/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#ifndef GPU_JIT_CONV_CONFIG_HPP
#define GPU_JIT_CONV_CONFIG_HPP

#include <iostream>
#include <sstream>
#include <unordered_map>

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/math_utils.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/compute/compute_engine.hpp"
#include "gpu/jit/conv/key.hpp"
#include "gpu/jit/conv/problem.hpp"
#include "gpu/jit/ir/config.hpp"
#include "gpu/jit/ir/fma.hpp"
#include "gpu/jit/ir/hw.hpp"
#include "gpu/jit/ir/message_patterns.hpp"
#include "gpu/jit/ir/post_ops.hpp"
#include "gpu/jit/ir/tensor.hpp"
#include "gpu/jit/ir/tensor_config.hpp"
#include "gpu/jit/jit_eltwise_injector.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Special optimization techniques for backward by data convolution.
//
// skip_out_of_bound_w enables skip-conditions for kw loop (unit stride only):
//     for (int kw = 0; kw < KW++) {
//         if (iw + PW - kw >= OW) continue;
//         ...
//     }
//
// skip_strided_{dh,dhw} kinds enable skip-conditions to handle strided
// backward by data convolution:
//     for (int kw = 0; kw < KW++) {
//         if ((iw + PW - kw) % SW != 0) continue;
//         ...
//     }
// skip_strided_dh enables skip-conditions for kd and kh loops.
// skip_strided_dhw enables skip-conditions for kd, kh and kw loops.
enum class bwd_d_optimize_kind_t {
    undef,
    none,
    skip_out_of_bound_w,
    skip_strided_dh,
    skip_strided_dhw,
};

inline std::string to_string(bwd_d_optimize_kind_t kind) {
    switch (kind) {
#define CASE(name) \
    case bwd_d_optimize_kind_t::name: return #name
        CASE(undef);
        CASE(none);
        CASE(skip_out_of_bound_w);
        CASE(skip_strided_dh);
        CASE(skip_strided_dhw);
#undef CASE
        default: ir_error_not_expected();
    }
    return "unknown";
}

inline bwd_d_optimize_kind_t to_bwd_d_optimize_kind(const std::string &s) {
#define CASE(name) \
    if (s == #name) return bwd_d_optimize_kind_t::name
    CASE(none);
    CASE(skip_out_of_bound_w);
    CASE(skip_strided_dh);
    CASE(skip_strided_dhw);
#undef CASE
    ir_error_not_expected();
    return bwd_d_optimize_kind_t::undef;
}

class bwd_d_optimize_kind_param_t
    : public value_param_t<bwd_d_optimize_kind_t> {
public:
    bwd_d_optimize_kind_param_t()
        : value_param_t(bwd_d_optimize_kind_t::undef) {}

    std::string name() const override { return "bwd-d-optimize"; }
    std::string desc() const override {
        return "Kind of special optimization for strided backward by data "
               "convolution.";
    }
    bool is_undef() const override {
        return get() == bwd_d_optimize_kind_t::undef;
    }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return get() == default_value; }

    void set_from_str(const std::string &s) override {
        value_ = to_bwd_d_optimize_kind(s);
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << short_name() << "=" << to_string(value_);
        return oss.str();
    }

    static const bwd_d_optimize_kind_t default_value;
};

class fma_kind_param_t : public value_param_t<fma_kind_t> {
public:
    fma_kind_param_t() : value_param_t(fma_kind_t::undef) {}

    std::string name() const override { return "fma"; }
    std::string desc() const override { return "FMA kind."; }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }

    void set_from_str(const std::string &s) override {
        value_ = str_to_fma_kind(s);
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << short_name() << "=" << to_string(value_);
        return oss.str();
    }
};

class pad_slm_param_t : public bool_param_t {
public:
    pad_slm_param_t() : bool_param_t(default_value) {}
    std::string name() const override { return "pad-slm"; }
    std::string desc() const override {
        return "Whether to pad SLM layout to avoid bank conflicts.";
    }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return get() == default_value; }

    static const bool default_value;
};

class pipeline_param_t : public param_t {
public:
    std::string name() const override { return "pipeline"; }
    std::string short_name() const override { return "P"; }
    std::string desc() const override { return "General pipeline parameters."; }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }

    bool do_unroll() const { return do_unroll_; }
    bool reuse_headers() const { return reuse_headers_; }

    void set_from_str(const std::string &s) override {
        do_unroll_ = false;
        reuse_headers_ = false;
        for (auto c : s) {
            switch (c) {
                case 'u': do_unroll_ = true; break;
                case 'r': reuse_headers_ = true; break;
                default: ir_error_not_expected() << s;
            }
        }
    }

    void set(bool do_unroll, bool reuse_headers) {
        do_unroll_ = do_unroll;
        reuse_headers_ = reuse_headers;
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << short_name() << "=";
        if (do_unroll_) oss << "u";
        if (reuse_headers_) oss << "r";
        return oss.str();
    }

private:
    bool do_unroll_ = false;
    bool reuse_headers_ = false;
};

class prb_param_t : public value_param_t<conv_problem_t> {
public:
    using value_param_t::value_param_t;

    std::string name() const override { return "prb"; }
    std::string desc() const override { return "Convolution problem."; }
    bool is_overridable() const override { return false; }

    void set_pd(const convolution_pd_t *pd) {
        value_.conv_pd = pd;
        value_.attr = pd->attr();
    }
};

class prefetch_param_t : public param_t {
public:
    std::string name() const override { return "prefetch"; }
    std::string short_name() const override { return "p"; }
    std::string desc() const override { return "Parameters for prefetching."; }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return bufs_ == 0; }

    int bufs() const { return bufs_; }
    bool a() const { return a_; }
    bool b() const { return b_; }

    operator bool() const { return bufs_ > 0; }

    void set_from_str(const std::string &s) override {
        a_ = false;
        b_ = false;
        bool ab_set = false;
        auto parts = gpu_utils::split(s, ".");
        for (auto &p : parts) {
            if (utils::one_of(p, "a", "b", "ab", "ba")) {
                ab_set = true;
                a_ = p.find("a") != std::string::npos;
                b_ = p.find("b") != std::string::npos;
                continue;
            }
            ir_assert(p.size() >= 2) << p;
            char name = p[0];
            int value = std::stoi(p.substr(1));
            switch (name) {
                case 'x': bufs_ = value; break;
                default: ir_error_not_expected() << p;
            }
        }
        if (!ab_set && bufs_ > 0) {
            a_ = true;
            b_ = true;
        }
    }

    void set(int bufs, bool a, bool b) {
        bufs_ = bufs;
        a_ = a;
        b_ = b;
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << short_name() << "=";
        oss << "x" << bufs_;
        if (a_ != b_) oss << "." << (a_ ? "a" : "b");
        return oss.str();
    }

private:
    int bufs_ = 0;
    // Whether prefetch for A is enabled.
    bool a_ = false;
    // Whether prefetch for B is enabled.
    bool b_ = false;
};

class slm_param_t : public param_t {
public:
    std::string name() const override { return "slm"; }
    std::string short_name() const override { return "s"; }
    std::string desc() const override { return "SLM buffering parameters."; }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return bufs_ == 0; }

    int bufs() const { return bufs_; }
    int gmem_bufs() const { return gmem_bufs_; }
    int sync_version() const { return sync_version_; }
    bool a() const { return a_; }
    bool b() const { return b_; }

    operator bool() const { return bufs() > 0; }

    void set_from_str(const std::string &s) override {
        a_ = false;
        b_ = false;
        bool ab_set = false;
        auto parts = gpu_utils::split(s, ".");
        for (auto &p : parts) {
            if (utils::one_of(p, "a", "b", "ab", "ba")) {
                ab_set = true;
                a_ = p.find("a") != std::string::npos;
                b_ = p.find("b") != std::string::npos;
                continue;
            }
            ir_assert(p.size() >= 2) << p;
            char name = p[0];
            int value = std::stoi(p.substr(1));
            switch (name) {
                case 'x': bufs_ = value; break;
                case 'g': gmem_bufs_ = value; break;
                case 'v': sync_version_ = value; break;
                default: ir_error_not_expected() << p;
            }
        }
        if (!ab_set && bufs_ > 0) {
            a_ = true;
            b_ = true;
        }
    }

    void set(int bufs, int gmem_bufs, bool a, bool b) {
        bufs_ = bufs;
        gmem_bufs_ = gmem_bufs;
        a_ = a;
        b_ = b;
    }

    void set_bufs(int bufs) { bufs_ = bufs; }
    void set_gmem_bufs(int gmem_bufs) { gmem_bufs_ = gmem_bufs; }

    std::string str() const override {
        std::ostringstream oss;
        oss << short_name() << "=";
        oss << "x" << bufs_;
        oss << ".g" << gmem_bufs_;
        if (sync_version_ != -1) oss << ".v" << sync_version_;
        if (a_ != b_) oss << "." << (a_ ? "a" : "b");
        return oss.str();
    }

private:
    // Number of SLM buffers to use (0, 1, 2 or 3).
    int bufs_ = 0;
    // Number of GRF buffers to use for GMEM -> SLM copy (0, 1 or 2).
    int gmem_bufs_ = 0;
    // See slm_sync_manager_t for more details.
    int sync_version_ = -1;
    // Whether SLM buffering for A is enabled.
    bool a_ = false;
    // Whether SLM buffering for B is enabled.
    bool b_ = false;
};

// Subtiles to split into for the inner A x B multiplication:
//
// Case 1. a_subtiles = 1, b_subtiles = 1
//     A = load(...)
//     B = load(...)
//     C += A * B
//
// Case 2. a_subtiles > 1, b_subtiles = 1
//     B = load(...)
//     for i in range(0, a_subtiles):
//         A_i = load(...)
//         C_i += A_i * B
//
// Case 3. a_subtiles = 1, b_subtiles > 1
//     A = load(...)
//     for j in range(0, b_subtiles):
//         B_j = load(...)
//         C_j += A * B_j
//
// Tiling for A and tiling for B are mutually exclusive. Using subtiles helps
// to reduce GRF consumption.
class subtiles_param_t : public param_t {
public:
    std::string name() const override { return "subtiles"; }
    std::string short_name() const override { return "S"; }
    std::string desc() const override { return "Sub-iteration blocking."; }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return (a_ == 1) && (b_ == 1); }

    int a() const { return a_; }
    int b() const { return b_; }

    void set_from_str(const std::string &s) override {
        a_ = 1;
        b_ = 1;
        for (auto &kv : ir_utils::to_string_int_map(s)) {
            if (kv.first == "a") {
                a_ = kv.second;
            } else if (kv.first == "b") {
                b_ = kv.second;
            } else {
                ir_error_not_expected() << kv.first;
            }
        }
    }

    void set(int a, int b) {
        a_ = a;
        b_ = b;
    }

    void set_a(int a) { a_ = a; }
    void set_b(int b) { b_ = b; }

    std::string str() const override {
        std::ostringstream oss;
        oss << short_name() << "=";
        if (a_ != 1) oss << "a" << a_;
        if (b_ != 1) oss << "b" << b_;
        return oss.str();
    }

private:
    int a_ = 1;
    int b_ = 1;
};

class wei_layout_param_t : public layout_param_t {
    std::string name() const override { return "wei"; }
    std::string desc() const override { return "Weights layout."; }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }
};

class bia_layout_param_t : public layout_param_t {
public:
    std::string name() const override { return "bia"; }
    std::string desc() const override { return "Bias layout."; }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }
};

namespace constants {
// Maximum number of SLM buffers.
static const int max_slm_bufs = 3;

// GRF usage for kernel arguments, local work IDs/sizes, signal header,
// temporary expressions, etc.
static const int reserved_regs_default = 16;
} // namespace constants

struct conv_plan_t;
class conv_tiler_t;

class conv_config_t : public prim_config_t {
public:
#define DECL_PARAM(name) \
    const name##_param_t &name##_param() const { \
        ir_assert(!name##_.is_undef()); \
        (void)name##_init_; \
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

    DECL_PARAM(bwd_d_optimize_kind)
    DECL_PARAM(fma_kind)
    DECL_PARAM(pad_slm)
    DECL_PARAM(prb)
    DECL_PARAM2(pipeline)
    DECL_PARAM2(prefetch)
    DECL_PARAM2(slm)
    DECL_PARAM2(subtiles)
    DECL_PARAM2(unroll)
    DECL_PARAM2(wei_layout)
    DECL_PARAM2(bia_layout)

#undef DECL_PARAM
#undef DECL_PARAM2

    std::string str() const override;

    const std::vector<prb_dim_t> &index_dims() const override {
        return conv_index_dims(prb().prop_kind());
    }
    prb_tile_t shape(bool pad) const override;

    std::string blocking_brief_str() const;

    conv_key_t key() const;

    // Blocks for padding. This is to comply with
    // zero-padding requirements. For example if the output
    // layout is nChw32c but there are only 8 channels to
    // compute and store, we still need to pad 8 to 32 and
    // spawn more thread groups to ensure 32c block is
    // properly zero-padded.
    int pad_block(const prb_dim_t &d) const override;

    int unroll(const prb_dim_t &d) const { return unroll()(d); }

    int reserved_regs() const;

    const hw_t &hw() const { return exec_cfg().hw(); }

    bool is_ge_xe_hpc() const { return hw() >= ngen::HW::XeHPC; }

    int grf_size() const { return hw().grf_size(); }

    int regs() const { return exec_cfg().regs(); }

    int simd() const { return exec_cfg().simd(); }

    int vec_size() const { return exec_cfg().vec_size(); }

    bool is_dp_fma() const { return jit::is_dp_fma(fma_kind()); }

    bool is_dpas_or_dpasw_fma() const {
        return utils::one_of(fma_kind(), fma_kind_t::dpas, fma_kind_t::dpasw);
    }

    const layout_param_t &a_layout() const {
        return prb().pick_a<const layout_param_t &>(
                src_layout(), wei_layout(), dst_layout());
    }

    const layout_param_t &b_layout() const {
        return prb().pick_b<const layout_param_t &>(
                src_layout(), wei_layout(), dst_layout());
    }

    compute::nd_range_t nd_range() const {
        size_t gws[3];
        size_t lws[3];
        for (int i = 0; i < 3; i++) {
            lws[i] = thread_group_grid().dim(i) * (i == 0 ? simd() : 1);
            gws[i] = kernel_grid().dim(i) * lws[i];
        }
        return compute::nd_range_t(gws, lws);
    }

    void set_pd(const convolution_pd_t *pd) { prb_.set_pd(pd); }

    void set_regs(int regs) {
        auto tmp = exec_cfg();
        tmp.set_regs(regs);
        set_exec_cfg(tmp);
    }

    void set_simd(int simd) {
        auto tmp = exec_cfg();
        tmp.set_simd(simd);
        set_exec_cfg(tmp);
    }

    void set_vec_size(int vec_size) {
        auto tmp = exec_cfg();
        tmp.set_vec_size(vec_size);
        set_exec_cfg(tmp);
    }

    void set_tiler(const std::shared_ptr<conv_tiler_t> &tiler);
    const conv_tiler_t &tiler() const;
    conv_tiler_t &tiler();

    void set_plan(const std::shared_ptr<conv_plan_t> &plan);
    const conv_plan_t &plan() const;

    bool can_skip_wei_zero_out() const;
    bool can_skip_bia_zero_out() const;

private:
    std::shared_ptr<conv_plan_t> plan_;
    std::shared_ptr<conv_tiler_t> tiler_;

#define INIT_PARAM(name) \
    name##_param_t name##_; \
    param_init_t name##_init_ \
            = register_param([](const container_config_t *c) { \
                  return &((const conv_config_t *)c)->name##_; \
              });

    INIT_PARAM(bwd_d_optimize_kind)
    INIT_PARAM(fma_kind)
    INIT_PARAM(pad_slm)
    INIT_PARAM(prb)
    INIT_PARAM(pipeline)
    INIT_PARAM(prefetch)
    INIT_PARAM(slm)
    INIT_PARAM(subtiles)
    INIT_PARAM(unroll)
    INIT_PARAM(wei_layout)
    INIT_PARAM(bia_layout)

#undef INIT_PARAM
};

class bmnk_dim_helper_t {
public:
    bmnk_dim_helper_t(const conv_config_t &cfg) {
        auto &prb = cfg.prb();
        gemm_iter_ = to_gemm(cfg.iter_dims().get(), prb),
        gemm_thread_group_ = to_gemm(cfg.thread_group_dims().get(), prb);
        gemm_loop_ = to_gemm(cfg.loop_dims().get(), prb);
    }

    int iter_dim(prb_dim_t d) const { return gemm_iter_.get(d, 1); }

    int thread_group_dim(prb_dim_t d) const {
        return gemm_thread_group_.get(d, 1);
    }

    int loop_dim(prb_dim_t d) const { return gemm_loop_.get(d, 1); }

private:
    prb_tile_t gemm_iter_;
    prb_tile_t gemm_thread_group_;
    prb_tile_t gemm_loop_;
};

status_t init_pd_time_cfg(const conv_problem_t &prb, conv_config_t &cfg,
        const engine_t *engine, convolution_pd_t *pd, primitive_attr_t *attr);
status_t init_cfg(conv_config_t &cfg, const primitive_t *prim);
int slm_bufs_hint(const conv_problem_t &prb, int m_tg, int n_tg,
        bool do_src_zp_compensation, bool enable_a, bool enable_b,
        bool do_unroll);
tensor_config_t get_tensor_config(const conv_config_t &cfg);
int estimate_register_count(const conv_config_t &cfg);
const std::array<prb_tile_t, 3> &get_kernel_grid_conv_dims(
        const conv_problem_t &prb);
const std::array<prb_tile_t, 3> &get_thread_group_grid_conv_dims(
        const conv_problem_t &prb);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
