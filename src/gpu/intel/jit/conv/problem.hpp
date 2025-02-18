/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_CONV_PROBLEM_HPP
#define GPU_INTEL_JIT_CONV_PROBLEM_HPP

#include <string>
#include <vector>

#include "common/c_types_map.hpp"
#include "gpu/intel/jit/ir/problem.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

bool is_conv_index(const pvar_t &dim);
bool is_conv_index(const pvar_t &dim, prop_kind_t prop);
const std::vector<pvar_t> &conv_dims();
const std::vector<pvar_t> &conv_index_dims(prop_kind_t prop);

const std::vector<pvar_t> &conv_layout_dims(
        tensor_kind_t tensor_kind, bool src_dst_with_group = false);

template <typename T>
T &&pick_abc(tensor_kind_t abc, prop_kind_t prop, T &&src, T &&wei, T &&dst) {
    bool is_fwd = (prop == prop_kind::forward);
    bool is_bwd_d = (prop == prop_kind::backward_data);
    bool is_bwd_w = (prop == prop_kind::backward_weights);
    switch (abc) {
        case tensor_kind_t::a:
            if (is_fwd || is_bwd_w) return std::forward<T>(src);
            return std::forward<T>(dst);
        case tensor_kind_t::b:
            if (is_fwd || is_bwd_d) return std::forward<T>(wei);
            return std::forward<T>(dst);
        case tensor_kind_t::c:
            if (is_fwd) return std::forward<T>(dst);
            if (is_bwd_d) return std::forward<T>(src);
            return std::forward<T>(wei);
        default: gpu_error_not_expected();
    }
    return std::forward<T>(src);
}

template <typename T>
T &&pick_a(prop_kind_t prop, T &&src, T &&wei, T &&dst) {
    return std::forward<T>(pick_abc(tensor_kind_t::a, prop,
            std::forward<T>(src), std::forward<T>(wei), std::forward<T>(dst)));
}

template <typename T>
T &&pick_b(prop_kind_t prop, T &&src, T &&wei, T &&dst) {
    return std::forward<T>(pick_abc(tensor_kind_t::b, prop,
            std::forward<T>(src), std::forward<T>(wei), std::forward<T>(dst)));
}

template <typename T>
T &&pick_c(prop_kind_t prop, T &&src, T &&wei, T &&dst) {
    return std::forward<T>(pick_abc(tensor_kind_t::c, prop,
            std::forward<T>(src), std::forward<T>(wei), std::forward<T>(dst)));
}

tensor_kind_t to_abc(prop_kind_t prop, tensor_kind_t tensor);
tensor_kind_t from_abc(prop_kind_t prop, tensor_kind_t abc);
const std::vector<pvar_t> &conv_stride_dims();
const std::vector<pvar_t> &conv_dilation_dims();
const std::vector<pvar_t> &conv_padding_dims();

class hw_t;

// Description of the convolution problem.
class conv_problem_t {
public:
    conv_problem_t() = default;

    status_t init(impl::engine_t *engine, const convolution_pd_t *conv_pd);

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
    bool is_f64_accumulator() const { return acc_data_type == data_type::f64; }
    bool is_fp8_conv() const {
        return utils::one_of(
                       src_data_type, data_type::f8_e4m3, data_type::f8_e5m2)
                || utils::one_of(
                        wei_data_type, data_type::f8_e5m2, data_type::f8_e4m3);
    }
    bool is_f32_conv() const {
        return utils::everyone_is(src_data_type, wei_data_type, data_type::f32);
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
        gpu_error_not_expected();
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

    bool is_fwd = false;
    bool is_bwd_d = false;
    bool is_bwd_w = false;
    bool with_bias = false;
    bool with_groups = false;
    bool with_sum = false;
    bool is_dw = false;
    bool ab_swap_transpose = false;
    bool strided = false;

    int ndims = 0;
    dim_t mb = 0; // Batch size.
    dim_t g = 0; // Groups.
    dim_t ic = 0, oc = 0; // Input and output channels.
    dim_t id = 0, ih = 0, iw = 0; // Input spatial sizes.
    dim_t od = 0, oh = 0, ow = 0; // Output spatial sizes.
    dim_t kd = 0, kh = 0, kw = 0; // Kernel sizes.
    dim_t sd = 0, sh = 0, sw = 0; // Strides.
    dim_t pd = 0, ph = 0, pw = 0; // Padding in the beginning.
    dim_t dd = 0, dh = 0, dw = 0; // Dilation.
    // Mapping for spatial dimensions (e.g. when 3D convolution is reduced to 1D).
    std::array<int, 3> dhw_map = {-1, -1, -1};
    dim_t isp = 0, osp = 0,
          ksp = 0; // Combined input/output/kernel spatial size.

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

void normalize_conv_shape(dim_t &id, dim_t &od, dim_t &kd, dim_t &sd, dim_t &dd,
        dim_t &pd, dim_t &ih, dim_t &oh, dim_t &kh, dim_t &sh, dim_t &dh,
        dim_t &ph, dim_t &iw, dim_t &ow, dim_t &kw, dim_t &sw, dim_t &dw,
        dim_t &pw, bool can_flatten_spatial, std::array<int, 3> &dhw_map);
bool is_small_ic(const conv_problem_t &prb);

class conv_arg_helper_t {
public:
    conv_arg_helper_t(const conv_problem_t &prb) : prb_(prb) {}

    int src_arg_key() const {
        if (prb_.is_fwd) return DNNL_ARG_SRC;
        if (prb_.is_bwd_d) return DNNL_ARG_DIFF_SRC;
        if (prb_.is_bwd_w) return DNNL_ARG_SRC;
        gpu_error_not_expected();
        return DNNL_ARG_UNDEF;
    }

    bool is_src_input() const { return prb_.is_fwd || prb_.is_bwd_w; }
    bool is_src_output() const { return prb_.is_bwd_d; }

    int wei_arg_key() const {
        if (prb_.is_fwd) return DNNL_ARG_WEIGHTS;
        if (prb_.is_bwd_d) return DNNL_ARG_WEIGHTS;
        if (prb_.is_bwd_w) return DNNL_ARG_DIFF_WEIGHTS;
        gpu_error_not_expected();
        return DNNL_ARG_UNDEF;
    }

    bool is_wei_input() const { return prb_.is_fwd || prb_.is_bwd_d; }
    bool is_wei_output() const { return prb_.is_bwd_w; }

    int bia_arg_key() const {
        if (prb_.is_fwd) return DNNL_ARG_BIAS;
        if (prb_.is_bwd_d) return DNNL_ARG_BIAS;
        if (prb_.is_bwd_w) return DNNL_ARG_DIFF_BIAS;
        gpu_error_not_expected();
        return DNNL_ARG_UNDEF;
    }

    bool is_bia_input() const { return prb_.is_fwd || prb_.is_bwd_d; }
    bool is_bia_output() const { return prb_.is_bwd_w; }

    int dst_arg_key() const {
        if (prb_.is_fwd) return DNNL_ARG_DST;
        if (prb_.is_bwd_d) return DNNL_ARG_DIFF_DST;
        if (prb_.is_bwd_w) return DNNL_ARG_DIFF_DST;
        gpu_error_not_expected();
        return DNNL_ARG_UNDEF;
    }

    bool is_dst_input() const { return prb_.is_bwd_d || prb_.is_bwd_w; }
    bool is_dst_output() const { return prb_.is_fwd; }

private:
    const conv_problem_t &prb_;
};

pvar_t to_gemm(const pvar_t &d, prop_kind_t prop, bool is_transpose = false);
pvar_tile_t to_gemm(
        const pvar_tile_t &t, prop_kind_t prop, bool is_transpose = false);
inline pvar_t to_gemm(const pvar_t &d, const conv_problem_t &prb) {
    return to_gemm(d, prb.prop_kind(), prb.ab_swap_transpose);
}
inline pvar_tile_t to_gemm(const pvar_tile_t &t, const conv_problem_t &prb) {
    return to_gemm(t, prb.prop_kind(), prb.ab_swap_transpose);
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
