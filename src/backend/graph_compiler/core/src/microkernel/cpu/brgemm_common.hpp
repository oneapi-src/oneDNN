/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_MICROKERNEL_CPU_BRGEMM_COMMON_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_MICROKERNEL_CPU_BRGEMM_COMMON_HPP

#include <map>
#include <utility>
#include <vector>
#include <compiler/ir/sc_data_type.hpp>
namespace sc {
namespace brgemm {

enum attr_key {
    // if unrollaed kernel is used (use_uker == true)
    // then "max_bs" is the the only batch size that can be used on
    // kernel call else "max_bs" is the maximum batch size that can be
    // used
    max_bs = 0, // int
    max_top_vpad, // int
    max_bottom_vpad, // int
    hint_expected_A_size, // int64_t
    hint_expected_B_size, // int64_t
    hint_expected_C_size, // int64_t
    hint_innermost_loop, // bool
    hint_loop_order, // enum, not use for now
    hint_prefetching, // default true, not use for now
    wary_tail_read, // bool
    generate_skip_accumulation, // bool
    // Value of bd_mask_level specifies how bd_mask is used in brgemm kernel
    // 0 - bd_mask is not used
    // 1 - bd_mask is used on storing stage only
    // 2 - bd_mask used both on reading and storing stages
    bd_mask_level, // int
    // use_uker is a boolean value that determines whether to use the unrolled
    // kernel or not
    use_uker, // bool
    // use_interleave_stores is a value that determines whether to use the
    // interleave stores or not
    // Currently we don't allow this value to be true as it needs more amx
    // buffer
    use_interleave_stores, // bool
    nkeys = use_interleave_stores + 1,
};

// inherit from onednn
enum alg_kind_t {
    alg_kind_undef,
    eltwise_begin = 0x1f,
    /// Eltwise: ReLU
    eltwise_relu = eltwise_begin,
    /// Eltwise: hyperbolic tangent non-linearity (tanh)
    eltwise_tanh = 0x2f,
    /// Eltwise: exponential linear unit (elu)
    eltwise_elu = 0x3f,
    /// Eltwise: square
    eltwise_square = 0x4f,
    /// Eltwise: abs
    eltwise_abs = 0x5f,
    /// Eltwise: square root
    eltwise_sqrt = 0x6f,
    /// Eltwise: linear
    eltwise_linear = 0x7f,
    /// Eltwise: bounded_relu
    eltwise_bounded_relu = 0x8f,
    /// Eltwise: soft_relu
    eltwise_soft_relu = 0x9f,
    /// Eltwise: logistic
    eltwise_logistic = 0xaf,
    /// Eltwise: exponent
    eltwise_exp = 0xbf,
    /// Eltwise: gelu
    ///
    /// @note Tanh approximation formula is used to approximate
    /// the cumulative distribution function of a Gaussian here
    eltwise_gelu_tanh = 0xcf,
    /// Eltwise: tanh-based gelu (alias for eltwise_gelu_tanh)
    eltwise_gelu = eltwise_gelu_tanh,
    /// Eltwise: swish
    eltwise_swish = 0xdf,
    /// Eltwise: natural logarithm
    eltwise_log = 0xef,
    /// Eltwise: clip
    eltwise_clip = 0xff,
    /// Eltwise: clip version 2
    eltwise_clip_v2 = 0x10,
    /// Eltwise: pow
    eltwise_pow = 0x20,
    /// Eltwise: erf-based gelu
    eltwise_gelu_erf = 0x30,
    /// Eltwise: round
    eltwise_round = 0x40,
    /// Eltwise: logsigmoid
    eltwise_logsigmoid = 0x50,
    /// Eltwise: mish
    eltwise_mish = 0x60,
    /// Eltwise: hardswish
    eltwise_hardswish = 0x70,
    /// Eltwise: ReLU (dst for backward)
    eltwise_relu_use_dst_for_bwd = 0x100,
    /// Eltwise: hyperbolic tangent non-linearity (tanh) (dst for backward)
    eltwise_tanh_use_dst_for_bwd = 0x101,
    /// Eltwise: exponential linear unit (elu) (dst for backward)
    eltwise_elu_use_dst_for_bwd = 0x102,
    /// Eltwise: square root (dst for backward)
    eltwise_sqrt_use_dst_for_bwd = 0x103,
    /// Eltwise: logistic (dst for backward)
    eltwise_logistic_use_dst_for_bwd = 0x104,
    /// Eltwise: exp (dst for backward)
    eltwise_exp_use_dst_for_bwd = 0x105,
    /// Eltwise: clip version 2 (dst for backward)
    eltwise_clip_v2_use_dst_for_bwd = 0x106,
    eltwise_end = eltwise_clip_v2_use_dst_for_bwd,
    binary_begin = 0x1fff0,
    /// Binary add
    binary_add = binary_begin,
    /// Binary mul
    binary_mul = 0x1fff1,
    /// Binary max
    binary_max = 0x1fff2,
    /// Binary min
    binary_min = 0x1fff3,
    /// Binary div
    binary_div = 0x1fff4,
    /// Binary sub
    binary_sub = 0x1fff5,
    /// Binary greater or equal
    binary_ge = 0x1fff6,
    /// Binary greater than
    binary_gt = 0x1fff7,
    /// Binary less or equal
    binary_le = 0x1fff8,
    /// Binary less than
    binary_lt = 0x1fff9,
    /// Binary equal
    binary_eq = 0x1fffa,
    /// Binary not equal
    binary_ne = 0x1fffb,
    binary_end = binary_ne,
    /// customized alg kind, because in onednn side, these postops are described
    /// as specific interfaces like `set_output_scales()`
    bias_add,
    out_scales,
    a_zp,
    b_zp,
    c_zp,
    out_dtype,
};

// enumerate of buffer type in post op calculation
enum postop_data_kind : int {
    bias = 0,
    scales,
    binary_post_ops_rhs,
    oc_logical_off,
    dst_row_logical_off,
    data_C_ptr,
    first_mb_matrix_addr_off,
    a_zp_compensations,
    b_zp_compensations,
    c_zp_values,
    skip_accumulation,
};

struct attrs_setting_t {
    static const int max_attrs_num = attr_key::nkeys; // without bd_mask
    typedef std::pair<attr_key, int64_t> attrs_map_t;
    int num_ = 0;
    attrs_map_t map_[];
};

// Todo: currently we don't support sum post op(inplace add)

// elementwise post op define
struct elt_op_t {
    elt_op_t() : elt_op_t(alg_kind_t::alg_kind_undef) {}
    elt_op_t(alg_kind_t alg, float scale = 1.f, float alpha = 1.f,
            float beta = 0.f)
        : alg_(alg), scale_(scale), alpha_(alpha), beta_(beta) {}
    alg_kind_t alg_;
    float scale_;
    float alpha_; // 0.f for general relu.
    float beta_;
};

// binary post op define
struct bin_op_t {
    bin_op_t(alg_kind_t alg, const int *shape, sc_data_etype dtype)
        : alg_(alg) {
        shape_[0] = shape[0];
        shape_[1] = shape[1];
        assert(shape_[0] > 0 && shape_[1] > 0);
        dtype_ = dtype;
    }
    alg_kind_t alg_ = alg_kind_t::alg_kind_undef;
    int shape_[2] = {0};
    sc_data_etype dtype_ = sc_data_etype::F32;
};

// customize bias op, align onednn sematic
// bias add occured before zp/scale calculation in onednn.
struct bias_op_t {
    bias_op_t(sc_data_etype dtype)
        : alg_(alg_kind_t::bias_add), dtype_(dtype) {}
    alg_kind_t alg_ = alg_kind_t::bias_add;
    sc_data_etype dtype_ = sc_data_etype::F32;
};

// Currently we only support single scale, but onednn need a vector of scales,
// even for `per_tensor`.
struct scale_op_t {
    scale_op_t() = default;
    scale_op_t(float scale) : alg_(alg_kind_t::out_scales), scale_(scale) {}
    alg_kind_t alg_ = alg_kind_t::out_scales;
    float scale_ = 1.f;
};

// currently not support zp because of brgemm interface.
// But it is effective.
struct zp_op_t {
    zp_op_t(alg_kind_t alg, int zp) : alg_(alg), zp_(zp) {}
    alg_kind_t alg_ = alg_kind_t::b_zp;
    int zp_ = 0;
};

struct out_op_t {
    out_op_t(sc_data_etype dtype) : dtype_(dtype) {}
    alg_kind_t alg_ = alg_kind_t::out_dtype;
    sc_data_etype dtype_ = sc_data_etype::F32;
};

struct empty_op_t {
    alg_kind_t alg_ = alg_kind_t::alg_kind_undef;
};

union postop_setting_t {
    postop_setting_t() {
        static_assert(sizeof(postop_setting_t) == sizeof(int64_t) * 2,
                "postop setting size is bigger than 16 bytes.");
        pack_info_[0] = 0;
        pack_info_[1] = 0;
        empty_op_ = empty_op_t();
    }
    postop_setting_t(const elt_op_t &op) { elt_op_ = op; }
    postop_setting_t(const bin_op_t &op) { bin_op_ = op; }
    postop_setting_t(const bias_op_t &op) { bias_op_ = op; }
    postop_setting_t(const scale_op_t &op) { scale_op_ = op; }
    postop_setting_t(const zp_op_t &op) { zp_op_ = op; }
    postop_setting_t(const out_op_t &op) { out_op_ = op; }
    bool operator==(const postop_setting_t &other) const {
        return pack_info_[0] == other.pack_info_[0]
                && pack_info_[1] == other.pack_info_[1];
    }
    empty_op_t empty_op_;
    elt_op_t elt_op_;
    bin_op_t bin_op_;
    bias_op_t bias_op_;
    scale_op_t scale_op_;
    zp_op_t zp_op_;
    out_op_t out_op_;
    int64_t pack_info_[2];
};

// allow multiple post ops.
struct postops_setting_t {
    // currently we support maximum 9 postops because of alignment of brgemm
    // cache `brg_arg` in runtime.
    static const int max_postops_num = 9;
    static const int op_size = sizeof(postop_setting_t);
    // number of post ops;
    int num_ = 0;
    postop_setting_t ops_[];
};

// nargs inherited from `brgemm_post_ops_data_t` in onednn backend.
static const int postops_data_init_func_nargs = 11;
static const int postops_data_size = postops_data_init_func_nargs * 8; // bytes
} // namespace brgemm

using sc_brgemm_attrs_t = std::map<brgemm::attr_key, int64_t>;
// to use bd_mask, we need to set brgemm kind to list_addr, use amx, max_bs>=1,
// bd_mask_level>=0 and use_uker=true
using sc_brgemm_bd_mask_t = std::vector<char>;
using sc_brgemm_postops_setting_t = std::vector<brgemm::postop_setting_t>;
} // namespace sc

#endif
