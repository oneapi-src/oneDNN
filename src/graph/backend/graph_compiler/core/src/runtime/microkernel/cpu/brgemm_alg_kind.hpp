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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_MICROKERNEL_CPU_BRGEMM_ALG_KIND_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_MICROKERNEL_CPU_BRGEMM_ALG_KIND_HPP

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace brgemm {

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
} // namespace brgemm
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
