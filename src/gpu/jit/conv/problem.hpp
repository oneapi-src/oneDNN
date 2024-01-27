/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_JIT_CONV_PROBLEM_HPP
#define GPU_JIT_CONV_PROBLEM_HPP

#include <string>
#include <vector>

#include "gpu/jit/ir/problem.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

bool is_conv_index(const prb_dim_t &dim);
bool is_conv_index(const prb_dim_t &dim, prop_kind_t prop);
const std::vector<prb_dim_t> &conv_dims();
const std::vector<prb_dim_t> &conv_index_dims(prop_kind_t prop);

const std::vector<prb_dim_t> &conv_layout_dims(
        tensor_kind_t tensor_kind, bool src_dst_with_group = false);

template <typename T>
T &&pick_a(prop_kind_t prop, T &&src, T &&wei, T &&dst) {
    return utils::one_of(prop, prop_kind::forward, prop_kind::backward_weights)
            ? std::forward<T>(src)
            : std::forward<T>(dst);
}

template <typename T>
T &&pick_b(prop_kind_t prop, T &&src, T &&wei, T &&dst) {
    return utils::one_of(prop, prop_kind::forward, prop_kind::backward_data)
            ? std::forward<T>(wei)
            : std::forward<T>(dst);
}

template <typename T>
T &&pick_c(prop_kind_t prop, T &&src, T &&wei, T &&dst) {
    bool is_fwd = (prop == prop_kind::forward);
    bool is_bwd_d = (prop == prop_kind::backward_data);
    return std::forward<T>(is_fwd ? dst : is_bwd_d ? src : wei);
}

tensor_kind_t to_abc(prop_kind_t prop, tensor_kind_t tensor);
const std::vector<prb_dim_t> &conv_stride_dims();
const std::vector<prb_dim_t> &conv_dilation_dims();
const std::vector<prb_dim_t> &conv_padding_dims();

class hw_t;

// Description of the convolution problem.
class conv_problem_t {
public:
    conv_problem_t() = default;

    status_t init(const engine_t *engine, const convolution_pd_t *conv_pd);

    bool is_stride1() const { return sd == 1 && sh == 1 && sw == 1; }

    // If possible, reduces dimensions for 1x1 kernel and shifts spatial
    // dimensions.
    void normalize_shape();

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

    const memory_desc_t &a_md() const;
    const memory_desc_t &b_md() const;
    const memory_desc_t &c_md() const;

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

    const convolution_pd_t *conv_pd = nullptr;
    const primitive_attr_t *attr = nullptr;

    data_type_t src_data_type = data_type::undef;
    data_type_t wei_data_type = data_type::undef;
    data_type_t dst_data_type = data_type::undef;
    data_type_t bia_data_type = data_type::undef;
    fpmath_mode_t fpmath_mode = fpmath_mode::strict;
    bool deterministic = false;

    bool is_fwd = false;
    bool is_bwd_d = false;
    bool is_bwd_w = false;
    bool with_bias = false;
    bool with_groups = false;
    bool with_sum = false;
    bool is_dw = false;
    bool ab_swap_transpose = false;

    int ndims = 0;
    int mb = 0; // Batch size.
    int g = 0; // Groups.
    int ic = 0, oc = 0; // Input and output channels.
    int id = 0, ih = 0, iw = 0; // Input spatial sizes.
    int od = 0, oh = 0, ow = 0; // Output spatial sizes.
    int kd = 0, kh = 0, kw = 0; // Kernel sizes.
    int sd = 0, sh = 0, sw = 0; // Strides.
    int pd = 0, ph = 0, pw = 0; // Padding in the beginning.
    int dd = 0, dh = 0, dw = 0; // Dilation.
    // Mapping for spatial dimensions (e.g. when 3D convolution is reduced to 1D).
    std::array<int, 3> dhw_map = {-1, -1, -1};
    int isp = 0, osp = 0, ksp = 0; // Combined input/output/kernel spatial size.

    data_type_t a_data_type = data_type::undef;
    data_type_t b_data_type = data_type::undef;
    data_type_t c_data_type = data_type::undef;
    data_type_t acc_data_type = data_type::undef;

    int a_data_type_size = 0;
    int b_data_type_size = 0;
    int c_data_type_size = 0;
    int acc_data_type_size = 0;

private:
    // Initializes A/B/C data types (GEMM notation: C += A * B) according to
    // the following convention:
    // FWD:        src -> A,      wei -> B,      dst -> C
    // BWD_D: diff_dst -> A,      wei -> B, diff_src -> C
    // BWD_W:      src -> A, diff_dst -> B, diff_wei -> C
    status_t init_abc_data_types(const hw_t &hw);

    status_t init_acc_data_type();

    bool with_sum_post_op() const;

    void init_transpose(const hw_t &hw);
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

prb_dim_t to_gemm(
        const prb_dim_t &d, prop_kind_t prop, bool is_transpose = false);
prb_tile_t to_gemm(
        const prb_tile_t &t, prop_kind_t prop, bool is_transpose = false);
inline prb_dim_t to_gemm(const prb_dim_t &d, const conv_problem_t &prb) {
    return to_gemm(d, prb.prop_kind(), prb.ab_swap_transpose);
}
inline prb_tile_t to_gemm(const prb_tile_t &t, const conv_problem_t &prb) {
    return to_gemm(t, prb.prop_kind(), prb.ab_swap_transpose);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
