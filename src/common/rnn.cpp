/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
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

#include "dnnl.h"

#include "c_types_map.hpp"
#include "cpu/gemm/os_blas.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {
namespace rnn {

int get_gates_count(dnnl_alg_kind_t cell_kind) {
    switch (cell_kind) {
        case dnnl::impl::alg_kind::vanilla_rnn: return 1;
        case dnnl::impl::alg_kind::vanilla_gru: return 3;
        case dnnl::impl::alg_kind::lbr_gru: return 3;
        case dnnl::impl::alg_kind::vanilla_lstm: return 4;
        default: assert(!"unknown cell kind"); return 0;
    }
    return 0;
}

} // namespace rnn
} // namespace impl
} // namespace dnnl

namespace {
using namespace dnnl::impl;
using namespace dnnl::impl::status;
using namespace dnnl::impl::types;
using namespace dnnl::impl::utils;

memory_desc_t copy_maybe_null(const memory_desc_t *md) {
    return md ? *md : zero_md();
}

rnn_desc_t zero_rnn_desc() {
    auto rd = rnn_desc_t();
    rd.src_layer_desc = zero_md();
    rd.src_iter_desc = zero_md();
    rd.weights_layer_desc = zero_md();
    rd.weights_iter_desc = zero_md();
    rd.bias_desc = zero_md();
    rd.dst_layer_desc = zero_md();
    rd.dst_iter_desc = zero_md();
    rd.diff_src_layer_desc = zero_md();
    rd.diff_src_iter_desc = zero_md();
    rd.diff_weights_layer_desc = zero_md();
    rd.diff_weights_iter_desc = zero_md();
    rd.diff_bias_desc = zero_md();
    rd.diff_dst_layer_desc = zero_md();
    rd.diff_dst_iter_desc = zero_md();
    return rd;
}

status_t check_data_type_consistency_fwd(dnnl_alg_kind_t cell_kind,
        prop_kind_t prop_kind, const memory_desc_t *src_layer_desc,
        const memory_desc_t *src_iter_desc,
        const memory_desc_t *src_iter_c_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc, const memory_desc_t *dst_iter_desc,
        const memory_desc_t *dst_iter_c_desc) {
    using namespace data_type;
    data_type_t src_layer_dt = src_layer_desc->data_type;
    data_type_t dst_layer_dt = dst_layer_desc->data_type;
    data_type_t weights_iter_dt = weights_iter_desc->data_type;
    data_type_t weights_layer_dt = weights_layer_desc->data_type;

    bool is_forward = !(prop_kind == prop_kind::backward);
    bool is_inference = prop_kind == prop_kind::forward_inference;
    bool is_lstm = cell_kind == dnnl_vanilla_lstm;

    bool cell_state_check
            = IMPLICATION(!is_zero_md(src_iter_c_desc),
                      one_of(src_iter_c_desc->data_type, f32, f16))
            && IMPLICATION(!is_zero_md(dst_iter_c_desc),
                    one_of(dst_iter_c_desc->data_type, f32, f16));

    bool is_f32 = everyone_is(f32, src_layer_dt, dst_layer_dt, weights_iter_dt,
                          weights_layer_dt)
            && IMPLICATION(
                    !is_zero_md(src_iter_desc), src_iter_desc->data_type == f32)
            && IMPLICATION(
                    !is_zero_md(dst_iter_desc), dst_iter_desc->data_type == f32)
            && IMPLICATION(!is_zero_md(bias_desc), bias_desc->data_type == f32);

    bool is_bf16 = everyone_is(bf16, src_layer_dt, dst_layer_dt,
                           weights_iter_dt, weights_layer_dt)
            && IMPLICATION(!is_zero_md(src_iter_desc),
                    src_iter_desc->data_type == bf16)
            && IMPLICATION(!is_zero_md(dst_iter_desc),
                    dst_iter_desc->data_type == bf16)
            && IMPLICATION(!is_zero_md(bias_desc), bias_desc->data_type == f32);

    bool is_f16 = is_forward
            && everyone_is(f16, src_layer_dt, dst_layer_dt, weights_iter_dt,
                    weights_layer_dt)
            && IMPLICATION(
                    !is_zero_md(src_iter_desc), src_iter_desc->data_type == f16)
            && IMPLICATION(
                    !is_zero_md(dst_iter_desc), dst_iter_desc->data_type == f16)
            && IMPLICATION(!is_zero_md(bias_desc), bias_desc->data_type == f16);

    bool is_u8u8u8 = is_inference && is_lstm && src_layer_dt == u8
            && IMPLICATION(
                    !is_zero_md(src_iter_desc), src_iter_desc->data_type == u8)
            && IMPLICATION(!is_zero_md(src_iter_c_desc),
                    src_iter_c_desc->data_type == f32)
            && IMPLICATION(
                    !is_zero_md(dst_iter_desc), dst_iter_desc->data_type == u8)
            && IMPLICATION(!is_zero_md(dst_iter_c_desc),
                    dst_iter_c_desc->data_type == f32)
            && one_of(dst_layer_dt, u8, f32)
            && everyone_is(s8, weights_iter_dt, weights_layer_dt)
            && IMPLICATION(!is_zero_md(bias_desc), bias_desc->data_type == f32);

    bool is_f32u8f32 = is_inference && is_lstm && src_layer_dt == u8
            && IMPLICATION(
                    !is_zero_md(src_iter_desc), src_iter_desc->data_type == f32)
            && IMPLICATION(
                    !is_zero_md(dst_iter_desc), dst_iter_desc->data_type == f32)
            && one_of(dst_layer_dt, u8, f32)
            && everyone_is(s8, weights_iter_dt, weights_layer_dt)
            && IMPLICATION(!is_zero_md(bias_desc), bias_desc->data_type == f32);

    return cell_state_check
                    && (is_f32 || is_bf16 || is_f16 || is_u8u8u8 || is_f32u8f32)
            ? success
            : unimplemented;
}

status_t check_data_type_consistency_bwd(dnnl_alg_kind_t cell_kind,
        prop_kind_t prop_kind, const memory_desc_t *diff_src_layer_desc,
        const memory_desc_t *diff_src_iter_desc,
        const memory_desc_t *diff_src_iter_c_desc,
        const memory_desc_t *diff_weights_layer_desc,
        const memory_desc_t *diff_weights_iter_desc,
        const memory_desc_t *diff_bias_desc,
        const memory_desc_t *diff_dst_layer_desc,
        const memory_desc_t *diff_dst_iter_desc,
        const memory_desc_t *diff_dst_iter_c_desc) {
    using namespace data_type;
    data_type_t diff_src_layer_dt = diff_src_layer_desc->data_type;
    data_type_t diff_dst_layer_dt = diff_dst_layer_desc->data_type;
    data_type_t diff_weights_iter_dt = diff_weights_iter_desc->data_type;
    data_type_t diff_weights_layer_dt = diff_weights_layer_desc->data_type;

    /* We require diffs to be f32, even for bf16 */
    bool are_diff_f32 = everyone_is(f32, diff_src_layer_dt, diff_dst_layer_dt,
                                diff_weights_iter_dt, diff_weights_layer_dt)
            && IMPLICATION(!is_zero_md(diff_src_iter_desc),
                    diff_src_iter_desc->data_type == f32)
            && IMPLICATION(!is_zero_md(diff_dst_iter_desc),
                    diff_dst_iter_desc->data_type == f32)
            && IMPLICATION(!is_zero_md(diff_bias_desc),
                    diff_bias_desc->data_type == f32)
            && IMPLICATION(!is_zero_md(diff_src_iter_c_desc),
                    diff_src_iter_c_desc->data_type == f32)
            && IMPLICATION(!is_zero_md(diff_dst_iter_c_desc),
                    diff_dst_iter_c_desc->data_type == f32);

    return are_diff_f32 ? success : unimplemented;
}

status_t check_dim_consistency(dnnl_alg_kind_t cell_kind,
        rnn_direction_t direction, dim_t L, dim_t D, dim_t T, dim_t N, dim_t G,
        dim_t SLC, dim_t SIC, dim_t DLC, dim_t DIC,
        const memory_desc_t *src_layer_desc, const memory_desc_t *src_iter_desc,
        const memory_desc_t *src_iter_c_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc, const memory_desc_t *dst_iter_desc,
        const memory_desc_t *dst_iter_c_desc) {
    bool args_ok;

    // * algorithm specific
    args_ok = true
            && IMPLICATION(utils::one_of(cell_kind, alg_kind::vanilla_gru,
                                   alg_kind::lbr_gru),
                    DIC == SIC);
    if (!args_ok) return invalid_arguments;
    dim_t extra_bias = cell_kind == alg_kind::lbr_gru;

    // * on num layers
    args_ok = true && L == weights_layer_desc->dims[0]
            && L == weights_iter_desc->dims[0]
            && IMPLICATION(!is_zero_md(bias_desc), L == bias_desc->dims[0])
            && IMPLICATION(
                    !is_zero_md(src_iter_desc), L == src_iter_desc->dims[0])
            && IMPLICATION(
                    !is_zero_md(src_iter_c_desc), L == src_iter_c_desc->dims[0])
            && IMPLICATION(
                    !is_zero_md(dst_iter_desc), L == dst_iter_desc->dims[0])
            && IMPLICATION(!is_zero_md(dst_iter_c_desc),
                    L == dst_iter_c_desc->dims[0]);
    if (!args_ok) return invalid_arguments;

    // * on num directions
    args_ok = true && D == weights_layer_desc->dims[1]
            && D == weights_iter_desc->dims[1]
            && IMPLICATION(!is_zero_md(bias_desc), D == bias_desc->dims[1])
            && IMPLICATION(
                    !is_zero_md(src_iter_desc), D == src_iter_desc->dims[1])
            && IMPLICATION(
                    !is_zero_md(src_iter_c_desc), D == src_iter_c_desc->dims[1])
            && IMPLICATION(
                    !is_zero_md(dst_iter_desc), D == dst_iter_desc->dims[1])
            && IMPLICATION(!is_zero_md(dst_iter_c_desc),
                    D == dst_iter_c_desc->dims[1]);
    if (!args_ok) return invalid_arguments;

    // * on num iterations
    args_ok = true && T == src_layer_desc->dims[0]
            && T == dst_layer_desc->dims[0];
    if (!args_ok) return invalid_arguments;

    // * on mb
    args_ok = true && N == src_layer_desc->dims[1]
            && N == dst_layer_desc->dims[1]
            && IMPLICATION(
                    !is_zero_md(src_iter_desc), N == src_iter_desc->dims[2])
            && IMPLICATION(
                    !is_zero_md(src_iter_c_desc), N == src_iter_c_desc->dims[2])
            && IMPLICATION(
                    !is_zero_md(dst_iter_desc), N == dst_iter_desc->dims[2])
            && IMPLICATION(!is_zero_md(dst_iter_c_desc),
                    N == dst_iter_c_desc->dims[2]);
    if (!args_ok) return invalid_arguments;

    // * on num gates
    args_ok = true && G == dnnl::impl::rnn::get_gates_count(cell_kind)
            && G == weights_layer_desc->dims[3]
            && G == weights_iter_desc->dims[3]
            && IMPLICATION(!is_zero_md(bias_desc),
                    G + extra_bias == bias_desc->dims[2]);
    if (!args_ok) return invalid_arguments;

    // * on slc
    args_ok = true && SLC == weights_layer_desc->dims[2]
            && SLC == src_layer_desc->dims[2];
    if (!args_ok) return invalid_arguments;

    // * on sic
    args_ok = true && SIC == weights_iter_desc->dims[2]
            && IMPLICATION(
                    !is_zero_md(src_iter_desc), SIC == src_iter_desc->dims[3]);
    if (!args_ok) return invalid_arguments;

    // * on dlc
    dim_t dlc_multiplier = (direction == dnnl_bidirectional_concat) ? 2 : 1;
    args_ok = true && DLC == dlc_multiplier * DIC
            && DLC == dst_layer_desc->dims[2];
    if (!args_ok) return invalid_arguments;

    // * on dic
    args_ok = true && DIC == weights_layer_desc->dims[4]
            && DIC == weights_iter_desc->dims[4]
            && IMPLICATION(!is_zero_md(bias_desc), DIC == bias_desc->dims[3])
            && IMPLICATION(!is_zero_md(src_iter_c_desc),
                    DIC == src_iter_c_desc->dims[3])
            && IMPLICATION(
                    !is_zero_md(dst_iter_desc), DIC == dst_iter_desc->dims[3])
            && IMPLICATION(!is_zero_md(dst_iter_c_desc),
                    DIC == dst_iter_c_desc->dims[3]);
    if (!args_ok) return invalid_arguments;

    // * unrolling/fusion conditions
    args_ok = true && IMPLICATION(L > 1, (dlc_multiplier * SLC) == DLC)
            && IMPLICATION(T > 1, SIC == DIC);
    if (!args_ok) return invalid_arguments;

    return success;
}

status_t rnn_common_fwd_desc_init(dnnl_rnn_desc_t *rnn_desc,
        prop_kind_t prop_kind, dnnl_alg_kind_t cell_kind,
        const rnn_direction_t direction, const memory_desc_t *src_layer_desc,
        const memory_desc_t *src_iter_desc,
        const memory_desc_t *src_iter_c_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc, const memory_desc_t *dst_iter_desc,
        const memory_desc_t *dst_iter_c_desc, unsigned flags,
        dnnl_alg_kind_t activation = dnnl_alg_kind_undef, float alpha = 0.0f,
        float beta = 0.0f) {

    // check that a supported cell kind has been passed
    bool args_ok = true
            && one_of(cell_kind, dnnl_vanilla_rnn, dnnl_vanilla_lstm,
                    dnnl_vanilla_gru, dnnl_lbr_gru);
    if (!args_ok) return invalid_arguments;

    // check that all mandatory parameters are non-null
    args_ok = args_ok
            && !any_null(src_layer_desc, weights_layer_desc, weights_iter_desc,
                    dst_layer_desc);
    if (!args_ok) return invalid_arguments;

    // check that optional parameters are passed properly, namely:
    // - if lstm and XXX_iter is provided, XXX_iter_c should be provided too
    auto xnor_md = [=](const memory_desc_t *a_md, const memory_desc_t *b_md) {
        return is_zero_md(a_md) == is_zero_md(b_md);
    };

    args_ok = args_ok
            && IMPLICATION(cell_kind == dnnl_vanilla_lstm,
                    xnor_md(src_iter_desc, src_iter_c_desc)
                            && xnor_md(dst_iter_desc, dst_iter_c_desc));
    if (!args_ok) return invalid_arguments;

    //check dimensions consistency
    dim_t L = weights_layer_desc->dims[0];
    dim_t T = src_layer_desc->dims[0];
    dim_t N = src_layer_desc->dims[1];
    const dim_t D = one_of(direction, dnnl_unidirectional_left2right,
                            dnnl_unidirectional_right2left)
            ? 1
            : 2;
    dim_t G = dnnl::impl::rnn::get_gates_count(cell_kind);
    dim_t SLC = src_layer_desc->dims[2];
    dim_t SIC = weights_iter_desc->dims[2];
    dim_t DLC = dst_layer_desc->dims[2];
    dim_t DIC = weights_layer_desc->dims[4];

    CHECK(check_dim_consistency(cell_kind, direction, L, D, T, N, G, SLC, SIC,
            DLC, DIC, src_layer_desc, src_iter_desc, src_iter_c_desc,
            weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc,
            dst_iter_desc, dst_iter_c_desc));

    CHECK(check_data_type_consistency_fwd(cell_kind, prop_kind, src_layer_desc,
            src_iter_desc, src_iter_c_desc, weights_layer_desc,
            weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc,
            dst_iter_c_desc));

    // Create the descriptor
    dnnl_rnn_desc_t rd = zero_rnn_desc();

    rd.primitive_kind = primitive_kind::rnn;
    rd.prop_kind = prop_kind;
    rd.cell_kind = cell_kind;
    rd.direction = direction;
    rd.src_layer_desc = copy_maybe_null(src_layer_desc);
    rd.src_iter_desc = copy_maybe_null(src_iter_desc);
    rd.src_iter_c_desc = copy_maybe_null(src_iter_c_desc);
    rd.weights_layer_desc = copy_maybe_null(weights_layer_desc);
    rd.weights_iter_desc = copy_maybe_null(weights_iter_desc);
    rd.bias_desc = copy_maybe_null(bias_desc);
    rd.dst_layer_desc = copy_maybe_null(dst_layer_desc);
    rd.dst_iter_desc = copy_maybe_null(dst_iter_desc);
    rd.dst_iter_c_desc = copy_maybe_null(dst_iter_c_desc);

    rd.flags = flags;
    rd.activation_kind = activation;
    rd.alpha = alpha;
    rd.beta = beta;

    *rnn_desc = rd;

    return success;
}

status_t rnn_common_bwd_desc_init(dnnl_rnn_desc_t *rnn_desc,
        dnnl_prop_kind_t prop_kind, dnnl_alg_kind_t cell_kind,
        const dnnl_rnn_direction_t direction,
        const dnnl_memory_desc_t *src_layer_desc,
        const dnnl_memory_desc_t *src_iter_desc,
        const dnnl_memory_desc_t *src_iter_c_desc,
        const dnnl_memory_desc_t *weights_layer_desc,
        const dnnl_memory_desc_t *weights_iter_desc,
        const dnnl_memory_desc_t *bias_desc,
        const dnnl_memory_desc_t *dst_layer_desc,
        const dnnl_memory_desc_t *dst_iter_desc,
        const dnnl_memory_desc_t *dst_iter_c_desc,
        const dnnl_memory_desc_t *diff_src_layer_desc,
        const dnnl_memory_desc_t *diff_src_iter_desc,
        const dnnl_memory_desc_t *diff_src_iter_c_desc,
        const dnnl_memory_desc_t *diff_weights_layer_desc,
        const dnnl_memory_desc_t *diff_weights_iter_desc,
        const dnnl_memory_desc_t *diff_bias_desc,
        const dnnl_memory_desc_t *diff_dst_layer_desc,
        const dnnl_memory_desc_t *diff_dst_iter_desc,
        const dnnl_memory_desc_t *diff_dst_iter_c_desc, unsigned flags,
        dnnl_alg_kind_t activation = dnnl_alg_kind_undef, float alpha = 0.0f,
        float beta = 0.0f) {

    // check that a supported cell kind has been passed
    bool args_ok = true
            && one_of(cell_kind, dnnl_vanilla_rnn, dnnl_vanilla_lstm,
                    dnnl_vanilla_gru, dnnl_lbr_gru);
    if (!args_ok) return invalid_arguments;

    // check that all mandatory parameters are non-null
    args_ok = args_ok
            && !any_null(src_layer_desc, weights_layer_desc, weights_iter_desc,
                    dst_layer_desc, diff_src_layer_desc,
                    diff_weights_layer_desc, diff_weights_iter_desc,
                    diff_dst_layer_desc);
    if (!args_ok) return invalid_arguments;

    // check that optional parameters are passed properly, namely:
    // - if lstm and XXX_iter is provided, XXX_iter_c should be provided too
    // - if XXX_iter is provided, diff_XXX_iter should be provided too
    auto xnor_md = [=](const memory_desc_t *a_md, const memory_desc_t *b_md) {
        return is_zero_md(a_md) == is_zero_md(b_md);
    };

    args_ok = args_ok
            && IMPLICATION(cell_kind == dnnl_vanilla_lstm,
                    xnor_md(src_iter_desc, src_iter_c_desc)
                            && xnor_md(dst_iter_desc, dst_iter_c_desc));

    args_ok = args_ok && xnor_md(bias_desc, diff_bias_desc)
            && xnor_md(src_iter_desc, diff_src_iter_desc)
            && xnor_md(src_iter_c_desc, src_iter_c_desc)
            && xnor_md(dst_iter_desc, diff_dst_iter_desc)
            && xnor_md(dst_iter_c_desc, diff_dst_iter_c_desc);
    if (!args_ok) return invalid_arguments;

    //check dimensions consistency
    int L = weights_layer_desc->dims[0];
    int T = src_layer_desc->dims[0];
    int N = src_layer_desc->dims[1];
    const int D = one_of(direction, dnnl_unidirectional_left2right,
                          dnnl_unidirectional_right2left)
            ? 1
            : 2;
    int G = dnnl::impl::rnn::get_gates_count(cell_kind);
    int SLC = src_layer_desc->dims[2];
    int SIC = weights_iter_desc->dims[2];
    int DLC = dst_layer_desc->dims[2];
    int DIC = weights_layer_desc->dims[4];

    status_t st = check_dim_consistency(cell_kind, direction, L, D, T, N, G,
            SLC, SIC, DLC, DIC, src_layer_desc, src_iter_desc, src_iter_c_desc,
            weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc,
            dst_iter_desc, dst_iter_c_desc);
    if (st != success) return st;

    st = check_dim_consistency(cell_kind, direction, L, D, T, N, G, SLC, SIC,
            DLC, DIC, diff_src_layer_desc, diff_src_iter_desc,
            diff_src_iter_c_desc, diff_weights_layer_desc,
            diff_weights_iter_desc, diff_bias_desc, diff_dst_layer_desc,
            diff_dst_iter_desc, diff_dst_iter_c_desc);
    if (st != success) return st;

    CHECK(check_data_type_consistency_fwd(cell_kind, prop_kind, src_layer_desc,
            src_iter_desc, src_iter_c_desc, weights_layer_desc,
            weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc,
            dst_iter_c_desc));

    CHECK(check_data_type_consistency_bwd(cell_kind, prop_kind,
            diff_src_layer_desc, diff_src_iter_desc, diff_src_iter_c_desc,
            diff_weights_layer_desc, diff_weights_iter_desc, diff_bias_desc,
            diff_dst_layer_desc, diff_dst_iter_desc, diff_dst_iter_c_desc));

    dnnl_rnn_desc_t rd = zero_rnn_desc();

    rd.primitive_kind = primitive_kind::rnn;
    rd.prop_kind = prop_kind;
    rd.cell_kind = cell_kind;
    rd.direction = direction;

    rd.src_layer_desc = copy_maybe_null(src_layer_desc);
    rd.src_iter_desc = copy_maybe_null(src_iter_desc);
    rd.src_iter_c_desc = copy_maybe_null(src_iter_c_desc);
    rd.weights_layer_desc = copy_maybe_null(weights_layer_desc);
    rd.weights_iter_desc = copy_maybe_null(weights_iter_desc);
    rd.bias_desc = copy_maybe_null(bias_desc);
    rd.dst_layer_desc = copy_maybe_null(dst_layer_desc);
    rd.dst_iter_desc = copy_maybe_null(dst_iter_desc);
    rd.dst_iter_c_desc = copy_maybe_null(dst_iter_c_desc);
    rd.diff_src_layer_desc = copy_maybe_null(diff_src_layer_desc);
    rd.diff_src_iter_desc = copy_maybe_null(diff_src_iter_desc);
    rd.diff_src_iter_c_desc = copy_maybe_null(diff_src_iter_c_desc);
    rd.diff_weights_layer_desc = copy_maybe_null(diff_weights_layer_desc);
    rd.diff_weights_iter_desc = copy_maybe_null(diff_weights_iter_desc);
    rd.diff_bias_desc = copy_maybe_null(diff_bias_desc);
    rd.diff_dst_layer_desc = copy_maybe_null(diff_dst_layer_desc);
    rd.diff_dst_iter_desc = copy_maybe_null(diff_dst_iter_desc);
    rd.diff_dst_iter_c_desc = copy_maybe_null(diff_dst_iter_c_desc);

    rd.flags = flags;
    rd.activation_kind = activation;
    rd.alpha = alpha;
    rd.beta = beta;

    *rnn_desc = rd;

    return success;
}

} // namespace

/* Public C Api */

status_t dnnl_vanilla_rnn_forward_desc_init(dnnl_rnn_desc_t *rnn_desc,
        dnnl_prop_kind_t prop_kind, const dnnl_alg_kind_t activation,
        const dnnl_rnn_direction_t direction,
        const dnnl_memory_desc_t *src_layer_desc,
        const dnnl_memory_desc_t *src_iter_desc,
        const dnnl_memory_desc_t *weights_layer_desc,
        const dnnl_memory_desc_t *weights_iter_desc,
        const dnnl_memory_desc_t *bias_desc,
        const dnnl_memory_desc_t *dst_layer_desc,
        const dnnl_memory_desc_t *dst_iter_desc, unsigned flags, float alpha,
        float beta) {
    status_t st = rnn_common_fwd_desc_init(rnn_desc, prop_kind,
            dnnl_vanilla_rnn, direction, src_layer_desc, src_iter_desc,
            &glob_zero_md, weights_layer_desc, weights_iter_desc, bias_desc,
            dst_layer_desc, dst_iter_desc, &glob_zero_md, flags, activation,
            alpha, beta);
    return st;
}

status_t dnnl_lstm_forward_desc_init(dnnl_rnn_desc_t *rnn_desc,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const dnnl_memory_desc_t *src_layer_desc,
        const dnnl_memory_desc_t *src_iter_desc,
        const dnnl_memory_desc_t *src_iter_c_desc,
        const dnnl_memory_desc_t *weights_layer_desc,
        const dnnl_memory_desc_t *weights_iter_desc,
        const dnnl_memory_desc_t *bias_desc,
        const dnnl_memory_desc_t *dst_layer_desc,
        const dnnl_memory_desc_t *dst_iter_desc,
        const dnnl_memory_desc_t *dst_iter_c_desc, unsigned flags) {

    status_t st = rnn_common_fwd_desc_init(rnn_desc, prop_kind,
            dnnl_vanilla_lstm, direction, src_layer_desc, src_iter_desc,
            src_iter_c_desc, weights_layer_desc, weights_iter_desc, bias_desc,
            dst_layer_desc, dst_iter_desc, dst_iter_c_desc, flags);
    return st;
}

status_t dnnl_gru_forward_desc_init(dnnl_rnn_desc_t *rnn_desc,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const dnnl_memory_desc_t *src_layer_desc,
        const dnnl_memory_desc_t *src_iter_desc,
        const dnnl_memory_desc_t *weights_layer_desc,
        const dnnl_memory_desc_t *weights_iter_desc,
        const dnnl_memory_desc_t *bias_desc,
        const dnnl_memory_desc_t *dst_layer_desc,
        const dnnl_memory_desc_t *dst_iter_desc, unsigned flags) {
    status_t st = rnn_common_fwd_desc_init(rnn_desc, prop_kind,
            dnnl_vanilla_gru, direction, src_layer_desc, src_iter_desc,
            &glob_zero_md, weights_layer_desc, weights_iter_desc, bias_desc,
            dst_layer_desc, dst_iter_desc, &glob_zero_md, flags);
    return st;
}

status_t dnnl_lbr_gru_forward_desc_init(dnnl_rnn_desc_t *rnn_desc,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const dnnl_memory_desc_t *src_layer_desc,
        const dnnl_memory_desc_t *src_iter_desc,
        const dnnl_memory_desc_t *weights_layer_desc,
        const dnnl_memory_desc_t *weights_iter_desc,
        const dnnl_memory_desc_t *bias_desc,
        const dnnl_memory_desc_t *dst_layer_desc,
        const dnnl_memory_desc_t *dst_iter_desc, unsigned flags) {
    status_t st = rnn_common_fwd_desc_init(rnn_desc, prop_kind, dnnl_lbr_gru,
            direction, src_layer_desc, src_iter_desc, &glob_zero_md,
            weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc,
            dst_iter_desc, &glob_zero_md, flags);
    return st;
}

status_t dnnl_vanilla_rnn_backward_desc_init(dnnl_rnn_desc_t *rnn_desc,
        dnnl_prop_kind_t prop_kind, const dnnl_alg_kind_t activation,
        const dnnl_rnn_direction_t direction,
        const dnnl_memory_desc_t *src_layer_desc,
        const dnnl_memory_desc_t *src_iter_desc,
        const dnnl_memory_desc_t *weights_layer_desc,
        const dnnl_memory_desc_t *weights_iter_desc,
        const dnnl_memory_desc_t *bias_desc,
        const dnnl_memory_desc_t *dst_layer_desc,
        const dnnl_memory_desc_t *dst_iter_desc,
        const dnnl_memory_desc_t *diff_src_layer_desc,
        const dnnl_memory_desc_t *diff_src_iter_desc,
        const dnnl_memory_desc_t *diff_weights_layer_desc,
        const dnnl_memory_desc_t *diff_weights_iter_desc,
        const dnnl_memory_desc_t *diff_bias_desc,
        const dnnl_memory_desc_t *diff_dst_layer_desc,
        const dnnl_memory_desc_t *diff_dst_iter_desc, unsigned flags,
        float alpha, float beta) {
    status_t st = rnn_common_bwd_desc_init(rnn_desc, prop_kind,
            dnnl_vanilla_rnn, direction, src_layer_desc, src_iter_desc,
            &glob_zero_md, weights_layer_desc, weights_iter_desc, bias_desc,
            dst_layer_desc, dst_iter_desc, &glob_zero_md, diff_src_layer_desc,
            diff_src_iter_desc, &glob_zero_md, diff_weights_layer_desc,
            diff_weights_iter_desc, diff_bias_desc, diff_dst_layer_desc,
            diff_dst_iter_desc, &glob_zero_md, flags, activation, alpha, beta);
    return st;
}

status_t dnnl_lstm_backward_desc_init(dnnl_rnn_desc_t *rnn_desc,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const dnnl_memory_desc_t *src_layer_desc,
        const dnnl_memory_desc_t *src_iter_desc,
        const dnnl_memory_desc_t *src_iter_c_desc,
        const dnnl_memory_desc_t *weights_layer_desc,
        const dnnl_memory_desc_t *weights_iter_desc,
        const dnnl_memory_desc_t *bias_desc,
        const dnnl_memory_desc_t *dst_layer_desc,
        const dnnl_memory_desc_t *dst_iter_desc,
        const dnnl_memory_desc_t *dst_iter_c_desc,
        const dnnl_memory_desc_t *diff_src_layer_desc,
        const dnnl_memory_desc_t *diff_src_iter_desc,
        const dnnl_memory_desc_t *diff_src_iter_c_desc,
        const dnnl_memory_desc_t *diff_weights_layer_desc,
        const dnnl_memory_desc_t *diff_weights_iter_desc,
        const dnnl_memory_desc_t *diff_bias_desc,
        const dnnl_memory_desc_t *diff_dst_layer_desc,
        const dnnl_memory_desc_t *diff_dst_iter_desc,
        const dnnl_memory_desc_t *diff_dst_iter_c_desc, unsigned flags) {

    status_t st = rnn_common_bwd_desc_init(rnn_desc, prop_kind,
            dnnl_vanilla_lstm, direction, src_layer_desc, src_iter_desc,
            src_iter_c_desc, weights_layer_desc, weights_iter_desc, bias_desc,
            dst_layer_desc, dst_iter_desc, dst_iter_c_desc, diff_src_layer_desc,
            diff_src_iter_desc, diff_src_iter_c_desc, diff_weights_layer_desc,
            diff_weights_iter_desc, diff_bias_desc, diff_dst_layer_desc,
            diff_dst_iter_desc, diff_dst_iter_c_desc, flags);
    return st;
}

status_t dnnl_gru_backward_desc_init(dnnl_rnn_desc_t *rnn_desc,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const dnnl_memory_desc_t *src_layer_desc,
        const dnnl_memory_desc_t *src_iter_desc,
        const dnnl_memory_desc_t *weights_layer_desc,
        const dnnl_memory_desc_t *weights_iter_desc,
        const dnnl_memory_desc_t *bias_desc,
        const dnnl_memory_desc_t *dst_layer_desc,
        const dnnl_memory_desc_t *dst_iter_desc,
        const dnnl_memory_desc_t *diff_src_layer_desc,
        const dnnl_memory_desc_t *diff_src_iter_desc,
        const dnnl_memory_desc_t *diff_weights_layer_desc,
        const dnnl_memory_desc_t *diff_weights_iter_desc,
        const dnnl_memory_desc_t *diff_bias_desc,
        const dnnl_memory_desc_t *diff_dst_layer_desc,
        const dnnl_memory_desc_t *diff_dst_iter_desc, unsigned flags) {
    status_t st = rnn_common_bwd_desc_init(rnn_desc, prop_kind,
            dnnl_vanilla_gru, direction, src_layer_desc, src_iter_desc,
            &glob_zero_md, weights_layer_desc, weights_iter_desc, bias_desc,
            dst_layer_desc, dst_iter_desc, &glob_zero_md, diff_src_layer_desc,
            diff_src_iter_desc, &glob_zero_md, diff_weights_layer_desc,
            diff_weights_iter_desc, diff_bias_desc, diff_dst_layer_desc,
            diff_dst_iter_desc, &glob_zero_md, flags);
    return st;
}

status_t dnnl_lbr_gru_backward_desc_init(dnnl_rnn_desc_t *rnn_desc,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const dnnl_memory_desc_t *src_layer_desc,
        const dnnl_memory_desc_t *src_iter_desc,
        const dnnl_memory_desc_t *weights_layer_desc,
        const dnnl_memory_desc_t *weights_iter_desc,
        const dnnl_memory_desc_t *bias_desc,
        const dnnl_memory_desc_t *dst_layer_desc,
        const dnnl_memory_desc_t *dst_iter_desc,
        const dnnl_memory_desc_t *diff_src_layer_desc,
        const dnnl_memory_desc_t *diff_src_iter_desc,
        const dnnl_memory_desc_t *diff_weights_layer_desc,
        const dnnl_memory_desc_t *diff_weights_iter_desc,
        const dnnl_memory_desc_t *diff_bias_desc,
        const dnnl_memory_desc_t *diff_dst_layer_desc,
        const dnnl_memory_desc_t *diff_dst_iter_desc, unsigned flags) {
    status_t st = rnn_common_bwd_desc_init(rnn_desc, prop_kind, dnnl_lbr_gru,
            direction, src_layer_desc, src_iter_desc, &glob_zero_md,
            weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc,
            dst_iter_desc, &glob_zero_md, diff_src_layer_desc,
            diff_src_iter_desc, &glob_zero_md, diff_weights_layer_desc,
            diff_weights_iter_desc, diff_bias_desc, diff_dst_layer_desc,
            diff_dst_iter_desc, &glob_zero_md, flags);
    return st;
}
