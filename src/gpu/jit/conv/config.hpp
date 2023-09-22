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
#include "gpu/jit/conv/params.hpp"
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

    // Number of operations (including virtual padding operations).
    double ops() const {
        double ret = 2.0;
        ret *= (double)g * mb * oc * ic;
        ret *= ksp;
        ret *= (is_bwd_d ? isp : osp);
        return ret;
    }
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
    bool reduce_b() const { return is_bwd_w && with_bias; }

    prop_kind_t prop_kind() const {
        if (is_fwd) return prop_kind::forward;
        if (is_bwd_d) return prop_kind::backward_data;
        if (is_bwd_w) return prop_kind::backward_weights;
        ir_error_not_expected();
        return prop_kind::undef;
    }

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
        return std::forward<T>(ab_swap_transpose ? (is_bwd_w ? dst : wei)
                        : (is_fwd || is_bwd_w)   ? src
                                                 : dst);
    }

    template <typename T>
    T &&pick_b(T &&src, T &&wei, T &&dst) const {
        return std::forward<T>(ab_swap_transpose
                        ? ((is_fwd || is_bwd_w) ? src : dst)
                        : (is_fwd || is_bwd_d) ? wei
                                               : dst);
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
    bool ab_swap_transpose = false;

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
    status_t init_abc_data_types(const hw_config_t &hw_cfg) {
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
            bool use_matching_fpmath
                    = gpu_utils::dev_getenv("use_matching_fpmath", false);
            if (use_matching_fpmath
                    && attr->mayidownconvert(data_type::f32, data_type::bf16)
                    && fma_kind::get_supported_kind(hw_cfg, data_type::bf16,
                               data_type::bf16, data_type::f32)
                            != fma_kind_t::unknown) {
                a_data_type = data_type::bf16;
                b_data_type = data_type::bf16;
            } else if (use_matching_fpmath
                    && attr->mayidownconvert(data_type::f32, data_type::f16)
                    && fma_kind::get_supported_kind(hw_cfg, data_type::f16,
                               data_type::f16, data_type::f32)
                            != fma_kind_t::unknown) {
                a_data_type = data_type::f16;
                b_data_type = data_type::f16;
            } else if (attr->mayidownconvert(data_type::f32, data_type::tf32)
                    && fma_kind::get_supported_kind(hw_cfg, data_type::tf32,
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

    void init_transpose(const hw_config_t &hw_cfg) {
        using sm = primitive_attr_t::skip_mask_t;
        auto attr_skip_mask = sm::post_ops | sm::sum_dt | sm::scales_runtime;
        bool allow_ab_transpose
                = gpu_utils::dev_getenv("allow_ab_transpose", true);
        bool any_zp = !attr->has_default_values(attr_skip_mask);
        bool any_f64
                = utils::one_of(data_type::f64, src_data_type, dst_data_type);
        if (!allow_ab_transpose || any_zp || any_f64 || with_groups) {
            ab_swap_transpose = false;
            return;
        }
        int max_sp = (hw_cfg.hw() >= ngen::HW::XeHPC) ? 1240 : 512;
        bool do_ic_swap = ((is_fwd || is_bwd_w) && oc < 6);
        bool do_oc_swap = ((is_bwd_d) && ic < 6);
        bool allow_bwd_w = !is_bwd_w
                || ((src_data_type != data_type::f32
                            || fpmath_mode == dnnl_fpmath_mode_tf32)
                        && osp % 8 == 0);
        bool allow_bwd_d
                = !is_bwd_d || (src_data_type == data_type::f32 && osp == isp);
        bool allow_fwd = !is_fwd
                || (dst_data_type != data_type::f32
                        && dst_data_type != data_type::f64 && mb <= 8
                        && ih != iw && iw <= max_sp);
        ab_swap_transpose = allow_fwd && allow_bwd_d && allow_bwd_w
                && (do_oc_swap || do_ic_swap);
    }
};

bool is_small_ic(const conv_problem_t &prb);

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

inline std::unordered_map<std::string, int> to_string_int_map(
        const std::string &s) {
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

class tile_param_t : public param_t {
public:
    using value_t = conv_tile_t;

    const value_t &get() const { return tile_; }

    bool is_empty() const { return tile_.is_empty(); }

    int get(const conv_dim_t &dim) const { return tile_.at(dim, 1); }

    int operator()(const conv_dim_t &dim) const { return get(dim); }

    void set_from_str(const std::string &s) override {
        tile_ = conv_tile_t();
        for (auto &kv : to_string_int_map(s)) {
            tile_[conv_dim_t::from_name(kv.first)] = kv.second;
        }
    }

    void set(const conv_dim_t &dim, int size) { tile_[dim] = size; }

    void set(const value_t &value) { tile_ = value; }

    template <typename T>
    void set(const tile_generic_t<T> &tile) {
        for (auto d : tile) {
            set(d.str(), tile[d]);
        }
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << short_name() << "=" << tile_.str();
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    value_t tile_;
};

// Parameters for kernel generation.

class bia_layout_param_t : public layout_param_t {
public:
    std::string name() const override { return "bia"; }
    std::string desc() const override { return "Bias layout."; }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }
};

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

class dims_param_t : public tile_param_t {
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

class kernel_grid_param_t : public grid_param_t {
public:
    std::string name() const override { return "kernel-grid"; }
    std::string desc() const override {
        return "Number of thread groups across dimensions (kernel grid).";
    }
    bool is_overridable() const override { return false; }
};

class iter_dims_param_t : public tile_param_t {
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

class padded_dims_param_t : public tile_param_t {
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
        for (auto &kv : to_string_int_map(s)) {
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

class thread_group_dims_param_t : public tile_param_t {
public:
    std::string name() const override { return "tg"; }
    std::string short_name() const override { return "T"; }
    std::string desc() const override {
        return "Thread group-level dimension blocks.";
    }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }
};

class unroll_param_t : public tile_param_t {
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
    DECL_PARAM(kernel_grid)
    DECL_PARAM(pad_slm)
    DECL_PARAM(prb)
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

    std::string str() const override;

    std::string blocking_brief_str() const;

    conv_key_t key() const;

    // Helper methods.
    int dim(const conv_dim_t &d) const { return dims()(d); }

    int iter_dim(const conv_dim_t &d) const { return iter_dims()(d); }

    int padded_dim(const conv_dim_t &d) const { return padded_dims()(d); }

    int loop_dim(const conv_dim_t &d) const { return loop_dims()(d); }

    int thread_group_dim(const conv_dim_t &d) const {
        return thread_group_dims()(d);
    }

    // Blocks for padding. This is to comply with
    // zero-padding requirements. For example if the output
    // layout is nChw32c but there are only 8 channels to
    // compute and store, we still need to pad 8 to 32 and
    // spawn more thread groups to ensure 32c block is
    // properly zero-padded.
    int pad_block(const conv_dim_t &d) const;

    int unroll(const conv_dim_t &d) const { return unroll()(d); }

    int reserved_regs() const;

    const hw_config_t &hw_cfg() const { return exec_cfg().hw_cfg(); }

    ngen::HW hw() const { return hw_cfg().hw(); }

    bool is_ge_xe_hpc() const { return hw() >= ngen::HW::XeHPC; }

    int grf_size() const { return hw_cfg().grf_size(); }

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

    int grid_dim(const conv_dim_t &dim) const {
        return ir_utils::safe_divide(padded_dim(dim),
                loop_dim(dim) * thread_group_dim(dim) * iter_dim(dim));
    }

    int iter_dim(std::initializer_list<conv_dim_t> dims) const {
        int ret = 1;
        for (auto &dim : dims)
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

    void set_params_id(int id);
    conv_params_t params() const;

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
    int params_id_ = -1;

#define INIT_PARAM(name) \
    name##_param_t name##_; \
    param_init_t name##_init_ = register_param([](const prim_config_t *c) { \
        return &((const conv_config_t *)c)->name##_; \
    });

    INIT_PARAM(bia_layout)
    INIT_PARAM(bwd_d_optimize_kind)
    INIT_PARAM(dims)
    INIT_PARAM(fma_kind)
    INIT_PARAM(iter_dims)
    INIT_PARAM(kernel_grid)
    INIT_PARAM(loop_dims)
    INIT_PARAM(pad_slm)
    INIT_PARAM(padded_dims)
    INIT_PARAM(pipeline)
    INIT_PARAM(prb)
    INIT_PARAM(prefetch)
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
    bmnk_dim_helper_t(const conv_config_t &cfg) {
        gemm_iter_ = to_gemm(cfg.iter_dims().get(), cfg.prb().prop_kind(),
                cfg.prb().ab_swap_transpose),
        gemm_thread_group_ = to_gemm(cfg.thread_group_dims().get(),
                cfg.prb().prop_kind(), cfg.prb().ab_swap_transpose);
        gemm_loop_ = to_gemm(cfg.loop_dims().get(), cfg.prb().prop_kind(),
                cfg.prb().ab_swap_transpose);
    }

    int iter_dim(gemm_dim_t d) const { return gemm_iter_.at(d, 1); }

    int thread_group_dim(gemm_dim_t d) const {
        return gemm_thread_group_.at(d, 1);
    }

    int loop_dim(gemm_dim_t d) const { return gemm_loop_.at(d, 1); }

private:
    gemm_tile_t gemm_iter_;
    gemm_tile_t gemm_thread_group_;
    gemm_tile_t gemm_loop_;
};

status_t init_pd_time_cfg(const conv_problem_t &prb, conv_config_t &cfg,
        const engine_t *engine, convolution_pd_t *pd, primitive_attr_t *attr);
status_t init_cfg(conv_config_t &cfg, const primitive_t *prim);
int slm_bufs_hint(const conv_problem_t &prb, int m_tg, int n_tg,
        bool do_src_zp_compensation, bool enable_a, bool enable_b,
        bool do_unroll);
tensor_config_t get_tensor_config(const conv_config_t &cfg);
int estimate_register_count(const conv_config_t &cfg);
const conv_tile_t *get_transpose_kernel_grid_conv_dims(
        const conv_problem_t &prb, int idx);
const conv_tile_t *get_transpose_thread_group_grid_conv_dims(
        const conv_problem_t &prb, int idx);
const conv_tile_t *get_kernel_grid_conv_dims(
        const conv_problem_t &prb, int idx);
const conv_tile_t *get_thread_group_grid_conv_dims(
        const conv_problem_t &prb, int idx);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
