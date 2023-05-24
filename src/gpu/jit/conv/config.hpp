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
#include "gpu/jit/conv/block_helper.hpp"
#include "gpu/jit/ir/config.hpp"
#include "gpu/jit/ir/fma.hpp"
#include "gpu/jit/ir/hw_config.hpp"
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

// Description of the convolution problem.
class conv_problem_t {
public:
    conv_problem_t() = default;

    status_t init(const engine_t *engine, const convolution_pd_t *conv_pd);

    bool is_stride1() const { return sd == 1 && sh == 1 && sw == 1; }

    // Reduces dimensions for 1x1 kernel.
    void try_reduce_to_1d();

    // Helper methods.
    bool is_s32_accumulator() const { return acc_data_type == data_type::s32; }
    bool is_f32_conv() const {
        return utils::everyone_is(src_data_type, wei_data_type, data_type::f32);
    }
    bool is_f64_conv() const {
        return utils::everyone_is(src_data_type, wei_data_type, data_type::f64);
    }
    bool is_int8_dst() const {
        return utils::one_of(dst_data_type, data_type::s8, data_type::u8);
    }
    bool is_mixed_int8() const {
        return utils::one_of(a_data_type, dnnl_f16, dnnl_f32)
                && utils::one_of(c_data_type, dnnl_u8, dnnl_s8);
    }
    bool matches_user_types() const {
        if (is_fwd) {
            return a_data_type == src_data_type && b_data_type == wei_data_type
                    && c_data_type == dst_data_type;
        } else if (is_bwd_d) {
            return a_data_type == dst_data_type && b_data_type == wei_data_type
                    && c_data_type == src_data_type;
        } else if (is_bwd_w) {
            return a_data_type == src_data_type && b_data_type == dst_data_type
                    && c_data_type == wei_data_type;
        } else {
            ir_error_not_expected();
            return false;
        }
    }

    bool reduce_b() const { return is_bwd_w && with_bias; }

    const memory_desc_t &a_md() const {
        return *pick_a(conv_pd->invariant_src_md(), conv_pd->invariant_wei_md(),
                conv_pd->invariant_dst_md());
    }

    const memory_desc_t &b_md() const {
        return *pick_b(conv_pd->invariant_src_md(), conv_pd->invariant_wei_md(),
                conv_pd->invariant_dst_md());
    }

    const memory_desc_t &c_md() const {
        return *pick_c(conv_pd->invariant_src_md(), conv_pd->invariant_wei_md(),
                conv_pd->invariant_dst_md());
    }

    template <typename T>
    T &&pick_a(T &&src, T &&wei, T &&dst) const {
        return (is_fwd || is_bwd_w) ? std::forward<T>(src)
                                    : std::forward<T>(dst);
    }

    template <typename T>
    T &&pick_b(T &&src, T &&wei, T &&dst) const {
        return (is_fwd || is_bwd_d) ? std::forward<T>(wei)
                                    : std::forward<T>(dst);
    }

    template <typename T>
    T &&pick_c(T &&src, T &&wei, T &&dst) const {
        return std::forward<T>(is_fwd ? dst : is_bwd_d ? src : wei);
    }

    template <typename T>
    T &&pick_by_dir(T &&fwd, T &&bwd_d, T &&bwd_w) const {
        return std::forward<T>(is_fwd ? fwd : is_bwd_d ? bwd_d : bwd_w);
    }

    std::string desc_str(bool print_mb = true) const;

    const convolution_pd_t *conv_pd;
    const primitive_attr_t *attr;

    data_type_t src_data_type;
    data_type_t wei_data_type;
    data_type_t dst_data_type;
    data_type_t bia_data_type;
    fpmath_mode_t fpmath_mode;

    bool is_fwd;
    bool is_bwd_d;
    bool is_bwd_w;
    bool with_bias;
    bool with_groups;
    bool with_sum;
    bool is_dw;

    int ndims;
    int mb; // Batch size.
    int g; // Groups.
    int ic, oc; // Input and output channels.
    int id, ih, iw; // Input spatial sizes.
    int od, oh, ow; // Output spatial sizes.
    int kd, kh, kw; // Kernel sizes.
    int sd, sh, sw; // Strides.
    int pd, ph, pw; // Padding in the beginning.
    int dd, dh, dw; // Dilation.
    int reduced_dim; // Indicates which dims were shifted over or reduced.
    int isp, osp, ksp; // Combined input/output/kernel spatial size.

    data_type_t a_data_type;
    data_type_t b_data_type;
    data_type_t c_data_type;
    data_type_t acc_data_type;

    int a_data_type_size;
    int b_data_type_size;
    int c_data_type_size;
    int acc_data_type_size;

private:
    // Initializes A/B/C data types (GEMM notation: C += A * B) according to
    // the following convention:
    // FWD:        src -> A,      wei -> B,      dst -> C
    // BWD_D: diff_dst -> A,      wei -> B, diff_src -> C
    // BWD_W:      src -> A, diff_dst -> B, diff_wei -> C
    status_t init_abc_data_types(ngen::HW hw) {
        a_data_type = pick_a(src_data_type, wei_data_type, dst_data_type);
        b_data_type = pick_b(src_data_type, wei_data_type, dst_data_type);
        // Always use f32 for accumulation/storing in the main kernel.
        c_data_type = is_bwd_w
                ? data_type::f32
                : pick_c(src_data_type, wei_data_type, dst_data_type);

        if (utils::everyone_is(
                    data_type::f32, a_data_type, b_data_type, c_data_type)) {

            // TODO: bf16 and f16 currently perform worse than tf32, this is
            // likely due to an extra reorder required on the b buffer.
            bool use_matching_fpmath = false;
#ifdef GEN_CONV_DEBUG
            use_matching_fpmath = ir_utils::getenv_bool(
                    "use_matching_fpmath", use_matching_fpmath);
#endif
            if (use_matching_fpmath
                    && attr->mayidownconvert(data_type::f32, data_type::bf16)
                    && fma_kind::get_supported_kind(hw, data_type::bf16,
                               data_type::bf16, data_type::f32)
                            != fma_kind_t::unknown) {
                a_data_type = data_type::bf16;
                b_data_type = data_type::bf16;
            } else if (use_matching_fpmath
                    && attr->mayidownconvert(data_type::f32, data_type::f16)
                    && fma_kind::get_supported_kind(hw, data_type::f16,
                               data_type::f16, data_type::f32)
                            != fma_kind_t::unknown) {
                a_data_type = data_type::f16;
                b_data_type = data_type::f16;
            } else if (attr->mayidownconvert(data_type::f32, data_type::tf32)
                    && fma_kind::get_supported_kind(hw, data_type::tf32,
                               data_type::tf32, data_type::f32)
                            != fma_kind_t::unknown) {
                a_data_type = data_type::tf32;
                b_data_type = data_type::tf32;
            }
        }

        a_data_type_size = (int)types::data_type_size(a_data_type);
        b_data_type_size = (int)types::data_type_size(b_data_type);
        c_data_type_size = (int)types::data_type_size(c_data_type);
        return status::success;
    }

    status_t init_acc_data_type() {
        auto a = a_data_type;
        auto b = b_data_type;
        acc_data_type = data_type::undef;
        if (utils::one_of(a, data_type::s8, data_type::u8)
                && utils::one_of(b, data_type::s8, data_type::u8)) {
            acc_data_type = data_type::s32;
        } else if (utils::everyone_is(data_type::f16, a, b)
                || utils::everyone_is(data_type::bf16, a, b)) {
            acc_data_type = data_type::f32;
        } else if (utils::everyone_is(data_type::tf32, a, b)) {
            acc_data_type = data_type::f32;
        } else if (utils::everyone_is(data_type::f32, a, b)) {
            acc_data_type = data_type::f32;
        } else if (utils::everyone_is(data_type::f64, a, b)) {
            acc_data_type = data_type::f64;
        }
        if (acc_data_type == data_type::undef) return status::unimplemented;
        acc_data_type_size = (int)types::data_type_size(acc_data_type);
        return status::success;
    }

    bool with_sum_post_op() {
        auto &post_ops = attr->post_ops_;
        return post_ops.find(primitive_kind::sum) != -1;
    }
};

bool is_small_ic(const conv_problem_t &prb);

class conv_hint_t {
public:
    conv_hint_t() = default;
    conv_hint_t(int def_max_tg_size) : def_max_tg_size_(def_max_tg_size) {}

    int max_tg_size() const {
        if (max_tg_size_ != 0) return max_tg_size_;
        return def_max_tg_size_;
    }

    bool max_tg_overridden() const { return max_tg_overridden_; }

    void set_max_tg_size(int value) {
        max_tg_overridden_ = max_tg_size_ != 0;
        max_tg_size_ = value;
    }

private:
    int max_tg_size_ = 0;
    int def_max_tg_size_ = 0;
    bool max_tg_overridden_ = false;
};

class conv_arg_helper_t {
public:
    conv_arg_helper_t(const conv_problem_t &prb) : prb_(prb) {}

    int src_arg_key() const {
        if (prb_.is_fwd) return DNNL_ARG_SRC;
        if (prb_.is_bwd_d) return DNNL_ARG_DIFF_SRC;
        if (prb_.is_bwd_w) return DNNL_ARG_SRC;
        ir_error_not_expected();
        return -1;
    }

    bool is_src_input() const { return prb_.is_fwd || prb_.is_bwd_w; }
    bool is_src_output() const { return prb_.is_bwd_d; }

    int wei_arg_key() const {
        if (prb_.is_fwd) return DNNL_ARG_WEIGHTS;
        if (prb_.is_bwd_d) return DNNL_ARG_WEIGHTS;
        if (prb_.is_bwd_w) return DNNL_ARG_DIFF_WEIGHTS;
        ir_error_not_expected();
        return -1;
    }

    bool is_wei_input() const { return prb_.is_fwd || prb_.is_bwd_d; }
    bool is_wei_output() const { return prb_.is_bwd_w; }

    int bia_arg_key() const {
        if (prb_.is_fwd) return DNNL_ARG_BIAS;
        if (prb_.is_bwd_d) return DNNL_ARG_BIAS;
        if (prb_.is_bwd_w) return DNNL_ARG_DIFF_BIAS;
        ir_error_not_expected();
        return -1;
    }

    bool is_bia_input() const { return prb_.is_fwd || prb_.is_bwd_d; }
    bool is_bia_output() const { return prb_.is_bwd_w; }

    int dst_arg_key() const {
        if (prb_.is_fwd) return DNNL_ARG_DST;
        if (prb_.is_bwd_d) return DNNL_ARG_DIFF_DST;
        if (prb_.is_bwd_w) return DNNL_ARG_DIFF_DST;
        ir_error_not_expected();
        return -1;
    }

    bool is_dst_input() const { return prb_.is_bwd_d || prb_.is_bwd_w; }
    bool is_dst_output() const { return prb_.is_fwd; }

private:
    const conv_problem_t &prb_;
};

class grid_param_t : public value_param_t<grid_info_t> {
public:
    using value_param_t::value_param_t;

    bool is_overridable() const override { return false; }
};

inline std::unordered_map<std::string, int> to_map(const std::string &s) {
    std::unordered_map<std::string, int> ret;
    int name_beg = -1;
    int value_beg = -1;
    for (int pos = 0; pos < (int)s.size() + 1; pos++) {
        bool prev_digit = pos > 0 && std::isdigit(s[pos - 1]);
        bool cur_digit = pos < (int)s.size() && std::isdigit(s[pos]);
        if ((pos == 0 || prev_digit) && !cur_digit) {
            if (name_beg != -1 && value_beg != -1) {
                auto key = s.substr(name_beg, value_beg - name_beg);
                auto value = std::stoi(s.substr(value_beg, pos - value_beg));
                ret[key] = value;
            }
            name_beg = pos;
            value_beg = -1;
        }
        if (!prev_digit && cur_digit) value_beg = pos;
    }
    return ret;
}

class map_param_t : public param_t {
public:
    using value_t = std::unordered_map<std::string, int>;

    const value_t &get() const { return map_; }

    bool is_empty() const { return map_.empty(); }

    int get(const std::string &name) const {
        auto it = map_.find(name);
        if (it == map_.end()) return 1;
        return it->second;
    }

    int operator()(const std::string &name) const { return get(name); }

    void set_from_str(const std::string &s) override {
        map_.clear();
        map_ = to_map(s);
    }

    void set(const std::string &name, int dim) {
        auto it = map_.find(name);
        if (dim == 1) {
            if (it != map_.end()) map_.erase(it);
            return;
        }
        map_[name] = dim;
    }

    void set(const value_t &value) { map_ = value; }

    std::string str() const override {
        std::ostringstream oss;
        oss << short_name() << "=";
        for (auto &kv : map_) {
            oss << kv.first << kv.second;
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    value_t map_;
};

// Parameters for kernel generation.

class bia_layout_param_t : public layout_param_t {
public:
    std::string name() const override { return "bia"; }
    std::string desc() const override { return "Bias layout."; }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }
};

class bwd_d_optimize_strided_param_t : public bool_param_t {
public:
    bwd_d_optimize_strided_param_t() : bool_param_t(false) {}
    std::string name() const override { return "bwd-d-optimize-strided"; }
    std::string desc() const override {
        return "Apply special optimization for strided BWD_D convolution.";
    }
    bool is_overridable() const override { return false; }
};

class bwd_d_optimize_unstrided_param_t : public bool_param_t {
public:
    bwd_d_optimize_unstrided_param_t() : bool_param_t(false) {}
    std::string name() const override { return "bwd-d-optimize-unstrided"; }
    std::string desc() const override {
        return "Apply special optimization for unstrided BWD_D convolution.";
    }
    bool is_overridable() const override { return false; }
};

class bwd_d_optimize_strided_iw_param_t : public bool_param_t {
public:
    bwd_d_optimize_strided_iw_param_t() : bool_param_t(false) {}
    std::string name() const override { return "bwd-d-optimize-strided-iw"; }
    std::string desc() const override {
        return "Apply special optimization for strided BWD_D convolution (iw "
               "dimension).";
    }
    bool is_overridable() const override { return false; }
};

// TODO: Remove, use heuristics to determine if it's worth to sacrifice EU
// utilization for larger SLM size.
class check_slm_size_param_t : public bool_param_t {
public:
    check_slm_size_param_t() : bool_param_t(default_value) {}
    std::string name() const override { return "check-slm-size"; }
    std::string short_name() const override { return "c"; }
    std::string desc() const override {
        return "Whether to check SLM size to ensure full EU utilization.";
    }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return get() == default_value; }

    static const bool default_value;
};

class dims_param_t : public map_param_t {
public:
    std::string name() const override { return "dims"; }
    std::string desc() const override { return "Problem dimensions."; }
    bool is_overridable() const override { return false; }
};

class fma_kind_param_t : public value_param_t<fma_kind_t> {
public:
    fma_kind_param_t() : value_param_t(fma_kind_t::unknown) {}

    std::string name() const override { return "fma"; }
    std::string desc() const override { return "FMA kind."; }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }

    void set_from_str(const std::string &s) override {
        value_ = fma_kind::from_string(s);
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << short_name() << "=" << fma_kind::to_string(value_);
        return oss.str();
    }
};

class fuse_spatial_param_t : public bool_param_t {
public:
    fuse_spatial_param_t() : bool_param_t(default_value) {}
    std::string name() const override { return "fuse-spatial"; }
    std::string short_name() const override { return "fsp"; }
    std::string desc() const override {
        return "Whether to apply blocking to fused spatial (otherwise only `w` "
               "is blocked).";
    }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return get() == default_value; }

    static const bool default_value;
};

class hint_param_t : public value_param_t<conv_hint_t> {
public:
    using value_param_t::value_param_t;

    std::string name() const override { return "hint"; }
    std::string desc() const override { return "Configuration hint."; }
    bool is_overridable() const override { return false; }
};

// TODO: Remove, use internal logic.
class hoist_masks_from_compute_loop_param_t : public bool_param_t {
public:
    hoist_masks_from_compute_loop_param_t() : bool_param_t(false) {}
    std::string name() const override {
        return "hoist-masks-from-compute-loop";
    }
    std::string desc() const override {
        return "Whether to move send mask initialization out of compute loop.";
    }
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

class iter_dims_param_t : public map_param_t {
public:
    std::string name() const override { return "iter"; }
    std::string short_name() const override { return "i"; }
    std::string desc() const override {
        return "Iteration-level dimension blocks.";
    }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }
};

class loop_dims_param_t : public dims_param_t {
public:
    std::string name() const override { return "loop"; }
    std::string short_name() const override { return "l"; }
    std::string desc() const override { return "Loop-level dimension blocks."; }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }
};

class shrink_tg_dims_param_t : public bool_param_t {
public:
    shrink_tg_dims_param_t() : bool_param_t(default_value) {}
    std::string name() const override { return "shrink-tg-dims"; }
    std::string short_name() const override { return "stg"; }
    std::string desc() const override {
        return "Whether to adjust tile sizes depending on batch size.";
    }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return get() == default_value; }

    static const bool default_value;
};

class ow_kw_grf_cache_param_t : public bool_param_t {
public:
    ow_kw_grf_cache_param_t() : bool_param_t(false) {}
    std::string name() const override { return "ow-kw-grf-cache"; }
    std::string desc() const override {
        return "Whether to use GRF cache to reuse source for ow/kw pairs";
    }
    bool is_overridable() const override { return false; }
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

class padded_dims_param_t : public map_param_t {
public:
    std::string name() const override { return "pad"; }
    std::string desc() const override {
        return "Padded dimensions (rounded-up for blocks and to comply with "
               "required zero padding in output layouts) .";
    }
    bool is_overridable() const override { return false; }
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
        auto parts = ir_utils::split(s, ".");
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

// TODO: Remove this parameter and enable 2D block messages based on the
// generation flow.
class send_2d_nhwc_param_t : public bool_param_t {
public:
    send_2d_nhwc_param_t() : bool_param_t(false) {}
    std::string name() const override { return "2d-send-nhwc"; }
    std::string desc() const override {
        return "Whether to use the optimal NHWC setup relying on 2D block "
               "messages.";
    }
    bool is_overridable() const override { return false; }
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
        auto parts = ir_utils::split(s, ".");
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
        for (auto &kv : to_map(s)) {
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

class thread_group_grid_param_t : public grid_param_t {
public:
    std::string name() const override { return "tg-grid"; }
    std::string desc() const override { return "Thread group grid."; }
    bool is_overridable() const override { return false; }
};

class thread_group_dims_param_t : public map_param_t {
public:
    std::string name() const override { return "tg"; }
    std::string short_name() const override { return "T"; }
    std::string desc() const override {
        return "Thread group-level dimension blocks.";
    }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }
};

class unroll_param_t : public map_param_t {
public:
    std::string name() const override { return "unroll"; }
    std::string short_name() const override { return "u"; }
    std::string desc() const override {
        return "Per-dimension unroll factors.";
    }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return is_empty(); }
};

class wei_layout_param_t : public layout_param_t {
    std::string name() const override { return "wei"; }
    std::string desc() const override { return "Weights layout."; }
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

class conv_config_t : public prim_config_t {
public:
#define DECL_PARAM(name) \
    const name##_param_t &name##_param() const { \
        (void)name##_init_; \
        return name##_; \
    } \
    name##_param_t &name##_param() { return name##_; } \
    const name##_param_t::value_t &name() const { return name##_.get(); } \
    void set_##name(const name##_param_t::value_t &value) { \
        name##_.set(value); \
    }

#define DECL_PARAM2(name) \
    const name##_param_t &name() const { \
        (void)name##_init_; \
        return name##_; \
    } \
    name##_param_t &name() { return name##_; }

    DECL_PARAM(bwd_d_optimize_strided)
    DECL_PARAM(bwd_d_optimize_unstrided)
    DECL_PARAM(bwd_d_optimize_strided_iw)
    DECL_PARAM(check_slm_size)
    DECL_PARAM(fma_kind)
    DECL_PARAM(fuse_spatial)
    DECL_PARAM(hint)
    DECL_PARAM(hoist_masks_from_compute_loop)
    DECL_PARAM(kernel_grid)
    DECL_PARAM(ow_kw_grf_cache)
    DECL_PARAM(pad_slm)
    DECL_PARAM(prb)
    DECL_PARAM(send_2d_nhwc)
    DECL_PARAM(shrink_tg_dims)
    DECL_PARAM(thread_group_grid)
    DECL_PARAM2(bia_layout)
    DECL_PARAM2(dims)
    DECL_PARAM2(iter_dims)
    DECL_PARAM2(loop_dims)
    DECL_PARAM2(padded_dims)
    DECL_PARAM2(pipeline)
    DECL_PARAM2(prefetch)
    DECL_PARAM2(slm)
    DECL_PARAM2(subtiles)
    DECL_PARAM2(thread_group_dims)
    DECL_PARAM2(unroll)
    DECL_PARAM2(wei_layout)

#undef DECL_PARAM
#undef DECL_PARAM2

    send_pattern_t a_load_pattern;
    send_pattern_t b_load_pattern;

    std::string str() const override;

    std::string blocking_brief_str() const;

    // Helper methods.
    int dim(const std::string &name) const { return dims()(name); }

    int iter_dim(const std::string &name) const { return iter_dims()(name); }

    int padded_dim(const std::string &name) const {
        return padded_dims()(name);
    }

    int loop_dim(const std::string &name) const { return loop_dims()(name); }

    int thread_group_dim(const std::string &name) const {
        return thread_group_dims()(name);
    }

    // Blocks for padding. This is to comply with
    // zero-padding requirements. For example if the output
    // layout is nChw32c but there are only 8 channels to
    // compute and store, we still need to pad 8 to 32 and
    // spawn more thread groups to ensure 32c block is
    // properly zero-padded.
    int pad_block(const std::string &name) const {
        auto &src = src_layout().compute();
        auto &wei = wei_layout().compute();
        auto &dst = dst_layout().compute();

#define CASE(_name, layout, idx) \
    if (name == _name) return layout.inner_block(idx)

        if (prb().is_fwd) {
            CASE("mb", dst, 0);
            CASE("g", dst, 1);
            CASE("oc", dst, 2);
        } else if (prb().is_bwd_d) {
            CASE("mb", src, 0);
            CASE("g", src, 1);
            CASE("ic", src, 2);
        } else if (prb().is_bwd_w) {
            CASE("g", wei, 0);
            CASE("oc", wei, 1);
            CASE("ic", wei, 2);
        }
#undef CASE
        return 1;
    }

    int unroll(const std::string &name) const { return unroll()(name); }

    int reserved_regs() const;

    const hw_config_t &hw_cfg() const { return exec_cfg().hw_cfg(); }

    ngen::HW hw() const { return hw_cfg().hw(); }

    bool is_ge_xe_hpc() const { return hw() >= ngen::HW::XeHPC; }

    int grf_size() const { return hw_cfg().grf_size(); }

    int regs() const { return exec_cfg().regs(); }

    int simd() const { return exec_cfg().simd(); }

    int vec_size() const { return exec_cfg().vec_size(); }

    bool is_g_mad() const {
        return fma_kind() == fma_kind_t::mad && prb().g > 1 && prb().ic < 4
                && prb().oc < 4 && prb().mb < 8 && !prb().is_dw;
    }

    bool is_broadcast_oc() const {
        return prb().is_fwd && fma_kind() == fma_kind_t::mad
                && hw() <= ngen::HW::XeLP && !prb().is_s32_accumulator()
                && prb().oc < 4 && prb().g == 1
                && (fuse_spatial() ? prb().osp : prb().ow) % vec_size() == 0;
    }

    bool is_dp_fma() const {
        return utils::one_of(fma_kind(), fma_kind_t::dpas, fma_kind_t::dpasw,
                fma_kind_t::dp4a);
    }

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

    int grid_dim(const std::string &dim) const {
        return ir_utils::safe_divide(padded_dim(dim),
                loop_dim(dim) * thread_group_dim(dim) * iter_dim(dim));
    }

    int iter_dim(std::initializer_list<const char *> dims) const {
        int ret = 1;
        for (auto *dim : dims)
            ret *= iter_dim(dim);
        return ret;
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

    void set_plan(const std::shared_ptr<conv_plan_t> &plan);
    const conv_plan_t &plan() const;

    bool can_skip_wei_zero_out() const;
    bool can_skip_bia_zero_out() const;

private:
    std::shared_ptr<conv_plan_t> plan_;

#define INIT_PARAM(name) \
    name##_param_t name##_; \
    param_init_t name##_init_ = register_param([](const prim_config_t *c) { \
        return &((const conv_config_t *)c)->name##_; \
    });

    INIT_PARAM(bia_layout)
    INIT_PARAM(bwd_d_optimize_strided)
    INIT_PARAM(bwd_d_optimize_unstrided)
    INIT_PARAM(bwd_d_optimize_strided_iw)
    INIT_PARAM(check_slm_size)
    INIT_PARAM(dims)
    INIT_PARAM(fma_kind)
    INIT_PARAM(fuse_spatial)
    INIT_PARAM(hint)
    INIT_PARAM(hoist_masks_from_compute_loop)
    INIT_PARAM(iter_dims)
    INIT_PARAM(kernel_grid)
    INIT_PARAM(loop_dims)
    INIT_PARAM(ow_kw_grf_cache)
    INIT_PARAM(pad_slm)
    INIT_PARAM(padded_dims)
    INIT_PARAM(pipeline)
    INIT_PARAM(prb)
    INIT_PARAM(prefetch)
    INIT_PARAM(send_2d_nhwc)
    INIT_PARAM(shrink_tg_dims)
    INIT_PARAM(slm)
    INIT_PARAM(subtiles)
    INIT_PARAM(thread_group_dims)
    INIT_PARAM(thread_group_grid)
    INIT_PARAM(unroll)
    INIT_PARAM(wei_layout)

#undef INIT_PARAM
};

class bmnk_dim_helper_t {
public:
    bmnk_dim_helper_t(const conv_config_t &cfg) : prb_(cfg.prb()), cfg_(cfg) {}

    int iter_dim(char bmnk) const {
        int ret = 1;
        for (auto &kv : cfg_.iter_dims().get()) {
            if (to_bmnk(kv.first) != bmnk) continue;
            ret *= kv.second;
        }
        return ret;
    }

    int prb_iter_ndims(char bmnk) const {
        int ret = 0;
        for (auto &kv : cfg_.iter_dims().get()) {
            if (to_bmnk(kv.first) != bmnk) continue;
            if (kv.second == 1) continue;
            ret++;
        }
        return ret;
    }

    int thread_group_dim(char bmnk) const {
        int ret = 1;
        for (auto &kv : cfg_.thread_group_dims().get()) {
            if (to_bmnk(kv.first) != bmnk) continue;
            ret *= kv.second;
        }
        return ret;
    }

    int loop_dim(char bmnk) const {
        int ret = 1;
        for (auto &kv : cfg_.loop_dims().get()) {
            if (to_bmnk(kv.first) != bmnk) continue;
            ret *= kv.second;
        }
        return ret;
    }

private:
    static bool contains(const char **array, const std::string &s) {
        for (const char **ptr = array; *ptr; ptr++) {
            if (s == *ptr) return true;
        }
        return false;
    }

    char to_bmnk(const std::string &dim_name) const {
        static const char *fwd_b_dims[] = {"g", nullptr};
        static const char *fwd_m_dims[]
                = {"mb", "osp", "od", "oh", "ow", nullptr};
        static const char *fwd_n_dims[] = {"oc", nullptr};
        static const char *fwd_k_dims[] = {"ic", "kd", "kh", "kw", nullptr};
        static const char *bwd_d_b_dims[] = {"g", nullptr};
        static const char *bwd_d_m_dims[] = {"mb", "id", "ih", "iw", nullptr};
        static const char *bwd_d_n_dims[] = {"ic", nullptr};
        static const char *bwd_d_k_dims[] = {"oc", "kd", "kh", "kw", nullptr};
        static const char *bwd_w_b_dims[] = {"g", nullptr};
        static const char *bwd_w_m_dims[] = {"ic", "kd", "kh", "kw", nullptr};
        static const char *bwd_w_n_dims[] = {"oc", nullptr};
        static const char *bwd_w_k_dims[] = {"mb", "od", "oh", "ow", nullptr};

        // XXX: Do not use pick_by_dir() to work around MSVC compiler bug.
        const char **b_dims = nullptr;
        const char **m_dims = nullptr;
        const char **n_dims = nullptr;
        const char **k_dims = nullptr;
        if (prb_.is_fwd) {
            b_dims = fwd_b_dims;
            m_dims = fwd_m_dims;
            n_dims = fwd_n_dims;
            k_dims = fwd_k_dims;
        } else if (prb_.is_bwd_d) {
            b_dims = bwd_d_b_dims;
            m_dims = bwd_d_m_dims;
            n_dims = bwd_d_n_dims;
            k_dims = bwd_d_k_dims;
        } else if (prb_.is_bwd_w) {
            b_dims = bwd_w_b_dims;
            m_dims = bwd_w_m_dims;
            n_dims = bwd_w_n_dims;
            k_dims = bwd_w_k_dims;
        } else {
            ir_error_not_expected();
        }

        if (contains(b_dims, dim_name)) return 'b';
        if (contains(m_dims, dim_name)) return 'm';
        if (contains(n_dims, dim_name)) return 'n';
        if (contains(k_dims, dim_name)) return 'k';

        ir_error_not_expected() << dim_name;
        return ' ';
    }

    const conv_problem_t &prb_;
    const conv_config_t &cfg_;
};

status_t init_pd_time_cfg(const conv_problem_t &prb, conv_config_t &cfg,
        const engine_t *engine, convolution_pd_t *pd, primitive_attr_t *attr);
status_t init_cfg(conv_config_t &cfg, const convolution_pd_t *pd);
tensor_config_t get_tensor_config(const conv_config_t &cfg);
int estimate_register_count(const conv_config_t &cfg);
bool can_use_a_2d_send(const conv_config_t &cfg);
bool can_use_b_2d_send(const conv_config_t &cfg);
bool matches_tag(const memory_desc_t &md, const std::string &tag);
const char **get_kernel_grid_conv_dims(const conv_problem_t &prb, int idx);
const char **get_thread_group_grid_conv_dims(
        const conv_problem_t &prb, int idx);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
