/*******************************************************************************
* Copyright 2018-2023 Intel Corporation
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

#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"
#include <initializer_list>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#define VCHECK_RNN(f, msg, ...) \
    VCHECK(primitive, create, check, rnn, (f), msg, ##__VA_ARGS__);

#define VCONDCHECK_RNN(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, rnn, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCONDCHECK_RNN_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, rnn, (cond), status::unimplemented, \
            msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace rnn {

int get_gates_count(dnnl_alg_kind_t cell_kind) {
    switch (cell_kind) {
        case dnnl::impl::alg_kind::vanilla_rnn: return 1;
        case dnnl::impl::alg_kind::vanilla_gru:
        case dnnl::impl::alg_kind::vanilla_augru: return 3;
        case dnnl::impl::alg_kind::lbr_gru:
        case dnnl::impl::alg_kind::lbr_augru: return 3;
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

void maybe_init_md(memory_desc_t &md, const memory_desc_t *with_md) {
    if (with_md) md = *with_md;
}

bool xnor_md(const memory_desc_t *a_md, const memory_desc_t *b_md) {
    return is_zero_md(a_md) == is_zero_md(b_md);
}

status_t check_runtime_dims_or_strides(
        std::initializer_list<const memory_desc_t *> l) {
    bool runtime_dims_or_strides = false;
    for (auto md : l)
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(md).has_runtime_dims_or_strides();
    return runtime_dims_or_strides ? unimplemented : success;
}

template <typename... DTs>
bool expect_dt(const memory_desc_t &md, DTs... dts) {
    return IMPLICATION(!is_zero_md(&md), utils::one_of(md.data_type, dts...));
}

status_t expect_dims(const memory_desc_t &md, std::initializer_list<dim_t> dims,
        bool allow_zero = true) {
    if (is_zero_md(&md))
        return (allow_zero || dims.size() == 0) ? success : invalid_arguments;

    if (md.ndims != (int)dims.size()) return invalid_arguments;

    int d_in_md = 0;
    for (auto d : dims)
        if (d != md.dims[d_in_md++]) return invalid_arguments;

    return success;
}

status_t check_data_type_consistency_fwd(const rnn_desc_t &r) {
    using namespace data_type;
    data_type_t src_layer_dt = r.src_layer_desc.data_type;
    data_type_t dst_layer_dt = r.dst_layer_desc.data_type;
    data_type_t weights_iter_dt = r.weights_iter_desc.data_type;
    data_type_t weights_layer_dt = r.weights_layer_desc.data_type;
    data_type_t weights_projection_dt = r.weights_projection_desc.data_type;

    const bool is_forward = !(r.prop_kind == prop_kind::backward);
    const bool is_inference = r.prop_kind == prop_kind::forward_inference;
    const bool is_int8_ok
            = one_of(r.cell_kind, dnnl_vanilla_lstm, dnnl_vanilla_gru);

    const bool cell_state_check = expect_dt(r.src_iter_c_desc, f32, bf16, f16)
            && expect_dt(r.dst_iter_c_desc, f32, bf16, f16);

    const bool is_f32 = everyone_is(f32, src_layer_dt, dst_layer_dt,
                                weights_iter_dt, weights_layer_dt)
            && expect_dt(r.src_iter_desc, f32)
            && expect_dt(r.weights_peephole_desc, f32)
            && expect_dt(r.weights_projection_desc, f32)
            && expect_dt(r.dst_iter_desc, f32) && expect_dt(r.bias_desc, f32);

    const bool is_bf16 = everyone_is(bf16, src_layer_dt, dst_layer_dt,
                                 weights_iter_dt, weights_layer_dt)
            && expect_dt(r.src_iter_desc, bf16)
            && IMPLICATION(r.cell_kind == dnnl_vanilla_lstm,
                    expect_dt(r.weights_peephole_desc, f32))
            /* weights_peephole_desc is reused as attention_desc */
            && IMPLICATION(
                    one_of(r.cell_kind, dnnl_vanilla_augru, dnnl_lbr_augru),
                    expect_dt(r.weights_peephole_desc, bf16))
            && one_of(weights_projection_dt, bf16, data_type::undef)
            && expect_dt(r.dst_iter_desc, bf16)
            && one_of(r.bias_desc.data_type, bf16, f32);

    const bool is_f16 = is_forward
            && everyone_is(f16, src_layer_dt, dst_layer_dt, weights_iter_dt,
                    weights_layer_dt)
            && expect_dt(r.src_iter_desc, f16)
            && expect_dt(r.weights_peephole_desc, f16)
            && r.weights_peephole_desc.data_type == data_type::undef
            && expect_dt(r.dst_iter_desc, f16) && expect_dt(r.bias_desc, f16);

    const bool is_u8u8u8 = is_inference && is_int8_ok && src_layer_dt == u8
            && one_of(dst_layer_dt, u8, f32)
            && everyone_is(s8, weights_iter_dt, weights_layer_dt)
            && expect_dt(r.src_iter_desc, u8)
            && expect_dt(r.src_iter_c_desc, f32)
            && r.weights_peephole_desc.data_type == data_type::undef
            && one_of(weights_projection_dt, s8, data_type::undef)
            && expect_dt(r.dst_iter_desc, u8)
            && expect_dt(r.dst_iter_c_desc, f32) && expect_dt(r.bias_desc, f32);

    const bool is_f32u8f32 = is_inference && is_int8_ok && src_layer_dt == u8
            && everyone_is(s8, weights_iter_dt, weights_layer_dt)
            && r.weights_peephole_desc.data_type == data_type::undef
            && one_of(weights_projection_dt, s8, data_type::undef)
            && one_of(dst_layer_dt, u8, f32) && expect_dt(r.src_iter_desc, f32)
            && expect_dt(r.dst_iter_desc, f32) && expect_dt(r.bias_desc, f32);

    const bool is_s8s8s8 = is_inference && is_int8_ok && src_layer_dt == s8
            && one_of(dst_layer_dt, s8, f32)
            && everyone_is(s8, weights_iter_dt, weights_layer_dt)
            && expect_dt(r.src_iter_desc, s8)
            && expect_dt(r.src_iter_c_desc, f32)
            && r.weights_peephole_desc.data_type == data_type::undef
            && one_of(weights_projection_dt, s8, data_type::undef)
            && expect_dt(r.dst_iter_desc, s8)
            && expect_dt(r.dst_iter_c_desc, f32) && expect_dt(r.bias_desc, f32);

    const bool is_f32s8f32 = is_inference && is_int8_ok && src_layer_dt == s8
            && everyone_is(s8, weights_iter_dt, weights_layer_dt)
            && r.weights_peephole_desc.data_type == data_type::undef
            && one_of(weights_projection_dt, s8, data_type::undef)
            && one_of(dst_layer_dt, s8, f32) && expect_dt(r.src_iter_desc, f32)
            && expect_dt(r.dst_iter_desc, f32) && expect_dt(r.bias_desc, f32);

    return cell_state_check
                    && (is_f32 || is_bf16 || is_f16 || is_u8u8u8 || is_f32u8f32
                            || is_s8s8s8 || is_f32s8f32)
            ? success
            : unimplemented;
}

status_t check_data_type_consistency_bwd(const rnn_desc_t &r) {
    using namespace data_type;

    /* We require diffs to be f32, even for bf16 */
    bool are_diff_f32 = everyone_is(f32, r.diff_src_layer_desc.data_type,
                                r.diff_dst_layer_desc.data_type,
                                r.diff_weights_iter_desc.data_type,
                                r.diff_weights_layer_desc.data_type)
            && expect_dt(r.diff_src_iter_desc, f32)
            && expect_dt(r.diff_dst_iter_desc, f32)
            && expect_dt(r.diff_weights_peephole_desc, f32)
            && expect_dt(r.diff_weights_projection_desc, f32)
            && expect_dt(r.diff_bias_desc, f32)
            && expect_dt(r.diff_src_iter_c_desc, f32)
            && expect_dt(r.diff_dst_iter_c_desc, f32);

    return are_diff_f32 ? success : unimplemented;
}

status_t check_dim_consistency(const rnn_desc_t &r) {
    const bool is_lstm_projection = r.cell_kind == dnnl_vanilla_lstm
            && !is_zero_md(&r.weights_projection_desc);

    const dim_t L = r.weights_layer_desc.dims[0];
    const dim_t T = r.src_layer_desc.dims[0];
    const dim_t N = r.src_layer_desc.dims[1];
    const dim_t D = one_of(r.direction, dnnl_unidirectional_left2right,
                            dnnl_unidirectional_right2left)
            ? 1
            : 2;
    const dim_t G = rnn::get_gates_count(r.cell_kind);
    const dim_t SLC = r.src_layer_desc.dims[2];
    const dim_t SIC = r.weights_iter_desc.dims[2];
    const dim_t DLC = r.dst_layer_desc.dims[2];
    const dim_t DHC = r.weights_layer_desc.dims[4];
    const dim_t DIC
            = is_lstm_projection ? r.weights_projection_desc.dims[3] : DHC;

    const bool extra_bias = utils::one_of(
            r.cell_kind, alg_kind::lbr_gru, alg_kind::lbr_augru);
    const dim_t dlc_multiplier
            = (r.direction == dnnl_bidirectional_concat) ? 2 : 1;

    VCONDCHECK_RNN(
            IMPLICATION(utils::one_of(r.cell_kind, alg_kind::vanilla_gru,
                                alg_kind::lbr_gru, alg_kind::vanilla_augru,
                                alg_kind::lbr_augru),
                    SIC == DHC),
            VERBOSE_INCONSISTENT_DIM, "weights_iter", 2, "weights_layer", 4);
    VCONDCHECK_RNN(dlc_multiplier * DIC == DLC, VERBOSE_INCONSISTENT_DIM,
            is_lstm_projection ? "weights_proj" : "weithts_layer",
            is_lstm_projection ? 3 : 4, "dst_layer", 2);
    VCONDCHECK_RNN(IMPLICATION(L > 1, dlc_multiplier * SLC == DLC),
            VERBOSE_INCONSISTENT_DIM, "src_layer", 2, "dst_layer", 2);
    VCONDCHECK_RNN(IMPLICATION(T > 1, SIC == DIC), VERBOSE_INCONSISTENT_DIM,
            "weights_iter", 2,
            is_lstm_projection ? "weights_proj" : "weithts_layer",
            is_lstm_projection ? 3 : 4);

#define CHECK_DIMS(t, allow_empty, ...) \
    do { \
        std::initializer_list<dim_t> _dims_ = {__VA_ARGS__}; \
        VCHECK_RNN((expect_dims(r.t##_desc, _dims_, allow_empty)), \
                VERBOSE_BAD_DIM, #t, -1); \
    } while (0)

    const bool is_augru = utils::one_of(
            r.cell_kind, alg_kind::vanilla_augru, alg_kind::lbr_augru);
    CHECK_DIMS(src_layer, false, T, N, SLC);
    CHECK_DIMS(src_iter, true, L, D, N, SIC);
    CHECK_DIMS(src_iter_c, true, L, D, N, DHC);
    CHECK_DIMS(weights_layer, false, L, D, SLC, G, DHC);
    CHECK_DIMS(weights_iter, false, L, D, SIC, G, DHC);
    if (is_augru)
        CHECK_DIMS(weights_peephole, true, T, N, 1);
    else
        CHECK_DIMS(weights_peephole, true, L, D, 3, DHC);
    CHECK_DIMS(weights_projection, true, L, D, DHC, DIC);
    CHECK_DIMS(bias, true, L, D, G + extra_bias, DHC);
    CHECK_DIMS(dst_layer, false, T, N, DLC);
    CHECK_DIMS(dst_iter, true, L, D, N, DIC);
    CHECK_DIMS(dst_iter_c, true, L, D, N, DHC);

    if (r.prop_kind == prop_kind::backward) {
        CHECK_DIMS(diff_src_layer, false, T, N, SLC);
        CHECK_DIMS(diff_src_iter, true, L, D, N, SIC);
        CHECK_DIMS(diff_src_iter_c, true, L, D, N, DHC);
        CHECK_DIMS(diff_weights_layer, false, L, D, SLC, G, DHC);
        CHECK_DIMS(diff_weights_iter, false, L, D, SIC, G, DHC);
        if (is_augru)
            CHECK_DIMS(diff_weights_peephole, true, T, N, 1);
        else
            CHECK_DIMS(diff_weights_peephole, true, L, D, 3, DHC);
        CHECK_DIMS(diff_weights_projection, true, L, D, DHC, DIC);
        CHECK_DIMS(diff_bias, true, L, D, G + extra_bias, DHC);
        CHECK_DIMS(diff_dst_layer, false, T, N, DLC);
        CHECK_DIMS(diff_dst_iter, true, L, D, N, DIC);
        CHECK_DIMS(diff_dst_iter_c, true, L, D, N, DHC);
    }
#undef CHECK_DIMS
    return success;
} // namespace

status_t rnn_common_fwd_desc_init(rnn_desc_t *rnn_desc, prop_kind_t prop_kind,
        alg_kind_t cell_kind, const rnn_direction_t direction,
        const memory_desc_t *src_layer_desc, const memory_desc_t *src_iter_desc,
        const memory_desc_t *src_iter_c_desc,
        const memory_desc_t *attention_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc,
        const memory_desc_t *weights_peephole_desc,
        const memory_desc_t *weights_projection_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_layer_desc,
        const memory_desc_t *dst_iter_desc,
        const memory_desc_t *dst_iter_c_desc, unsigned flags,
        alg_kind_t activation = alg_kind::undef, float alpha = 0.0f,
        float beta = 0.0f) {
    using namespace alg_kind;

    // check that a supported cell kind has been passed
    VCONDCHECK_RNN(one_of(cell_kind, vanilla_rnn, vanilla_lstm, vanilla_gru,
                           lbr_gru, vanilla_augru, lbr_augru),
            VERBOSE_BAD_ALGORITHM);

    // check that all mandatory parameters are non-null
    VCONDCHECK_RNN(!any_null(src_layer_desc, weights_layer_desc,
                           weights_iter_desc, dst_layer_desc),
            VERBOSE_NULL_ARG);

    if (cell_kind == dnnl_vanilla_rnn) {
        VCONDCHECK_RNN(one_of(activation, eltwise_relu, eltwise_tanh,
                               eltwise_logistic),
                VERBOSE_BAD_ALGORITHM);
    }

    if (cell_kind == dnnl_vanilla_lstm) {
        // check if optional *_iter is provided then *_iter_c is provided too
        VCONDCHECK_RNN(xnor_md(src_iter_desc, src_iter_c_desc)
                        && xnor_md(dst_iter_desc, dst_iter_c_desc),
                VERBOSE_NULL_ARG);
    }

    // check augru-specific restrictions
    const bool is_augru = one_of(cell_kind, dnnl_vanilla_augru, dnnl_lbr_augru);
    if (is_augru) {
        VCONDCHECK_RNN(direction == dnnl_unidirectional_left2right,
                VERBOSE_BAD_PARAM, "direction != unidirectional_left2right");
        VCONDCHECK_RNN(weights_layer_desc->dims[0] == 1, VERBOSE_BAD_PARAM,
                "num_layers != 1");
    }

    VCHECK_RNN(
            check_runtime_dims_or_strides({src_layer_desc, src_iter_desc,
                    src_iter_c_desc, weights_layer_desc, weights_iter_desc,
                    weights_peephole_desc, weights_projection_desc, bias_desc,
                    dst_layer_desc, dst_iter_desc, dst_iter_c_desc}),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    // Create the descriptor
    auto rd = rnn_desc_t();

    rd.primitive_kind = primitive_kind::rnn;
    rd.prop_kind = prop_kind;
    rd.cell_kind = cell_kind;
    rd.direction = direction;
    maybe_init_md(rd.src_layer_desc, src_layer_desc);
    maybe_init_md(rd.src_iter_desc, src_iter_desc);
    maybe_init_md(rd.src_iter_c_desc, src_iter_c_desc);
    maybe_init_md(rd.weights_layer_desc, weights_layer_desc);
    maybe_init_md(rd.weights_iter_desc, weights_iter_desc);
    maybe_init_md(rd.weights_peephole_desc, weights_peephole_desc);
    if (is_augru) maybe_init_md(rd.weights_peephole_desc, attention_desc);
    maybe_init_md(rd.weights_projection_desc, weights_projection_desc);
    maybe_init_md(rd.bias_desc, bias_desc);
    maybe_init_md(rd.dst_layer_desc, dst_layer_desc);
    maybe_init_md(rd.dst_iter_desc, dst_iter_desc);
    maybe_init_md(rd.dst_iter_c_desc, dst_iter_c_desc);

    rd.flags = flags;
    rd.activation_kind = activation;
    rd.alpha = alpha;
    rd.beta = beta;

    VCHECK_RNN(check_data_type_consistency_fwd(rd), VERBOSE_UNSUPPORTED_DT_CFG);
    CHECK(check_dim_consistency(rd));

    *rnn_desc = rd;

    return success;
}

status_t rnn_common_bwd_desc_init(rnn_desc_t *rnn_desc, prop_kind_t prop_kind,
        alg_kind_t cell_kind, const dnnl_rnn_direction_t direction,
        const memory_desc_t *src_layer_desc, const memory_desc_t *src_iter_desc,
        const memory_desc_t *src_iter_c_desc,
        const memory_desc_t *attention_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc,
        const memory_desc_t *weights_peephole_desc,
        const memory_desc_t *weights_projection_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_layer_desc,
        const memory_desc_t *dst_iter_desc,
        const memory_desc_t *dst_iter_c_desc,
        const memory_desc_t *diff_src_layer_desc,
        const memory_desc_t *diff_src_iter_desc,
        const memory_desc_t *diff_src_iter_c_desc,
        const memory_desc_t *diff_attention_desc,
        const memory_desc_t *diff_weights_layer_desc,
        const memory_desc_t *diff_weights_iter_desc,
        const memory_desc_t *diff_weights_peephole_desc,
        const memory_desc_t *diff_weights_projection_desc,
        const memory_desc_t *diff_bias_desc,
        const memory_desc_t *diff_dst_layer_desc,
        const memory_desc_t *diff_dst_iter_desc,
        const memory_desc_t *diff_dst_iter_c_desc, unsigned flags,
        alg_kind_t activation = alg_kind::undef, float alpha = 0.0f,
        float beta = 0.0f) {
    using namespace alg_kind;

    // check that a supported cell kind has been passed
    VCONDCHECK_RNN(one_of(cell_kind, vanilla_rnn, vanilla_lstm, vanilla_gru,
                           lbr_gru, vanilla_augru, lbr_augru),
            VERBOSE_BAD_ALGORITHM);

    // check that all mandatory parameters are non-null
    VCONDCHECK_RNN(!any_null(src_layer_desc, weights_layer_desc,
                           weights_iter_desc, dst_layer_desc,
                           diff_src_layer_desc, diff_weights_layer_desc,
                           diff_weights_iter_desc, diff_dst_layer_desc),
            VERBOSE_NULL_ARG);

    if (cell_kind == dnnl_vanilla_rnn) {
        VCONDCHECK_RNN(one_of(activation, eltwise_relu, eltwise_tanh,
                               eltwise_logistic),
                VERBOSE_BAD_ALGORITHM);
    }

    if (cell_kind == dnnl_vanilla_lstm) {
        // check if optional *_iter is provided then *_iter_c is provided too
        VCONDCHECK_RNN(xnor_md(src_iter_desc, src_iter_c_desc)
                        && xnor_md(dst_iter_desc, dst_iter_c_desc),
                VERBOSE_NULL_ARG);
    }

    const bool is_augru = one_of(cell_kind, dnnl_vanilla_augru, dnnl_lbr_augru);
    // check augru-specific restrictions
    if (is_augru) {
        VCONDCHECK_RNN(direction == dnnl_unidirectional_left2right,
                VERBOSE_BAD_PARAM, "direction != unidirectional_left2right");
        VCONDCHECK_RNN(weights_layer_desc->dims[0] == 1, VERBOSE_BAD_PARAM,
                "num_layers != 1");
        VCONDCHECK_RNN(!any_null(attention_desc, diff_attention_desc),
                VERBOSE_NULL_ARG);
    }

    // check if optional md is provided then diff_md is provided too
    VCONDCHECK_RNN(xnor_md(bias_desc, diff_bias_desc), VERBOSE_NULL_ARG);
    VCONDCHECK_RNN(xnor_md(weights_peephole_desc, diff_weights_peephole_desc),
            VERBOSE_NULL_ARG);
    VCONDCHECK_RNN(
            xnor_md(weights_projection_desc, diff_weights_projection_desc),
            VERBOSE_NULL_ARG);
    VCONDCHECK_RNN(
            xnor_md(src_iter_desc, diff_src_iter_desc), VERBOSE_NULL_ARG);
    VCONDCHECK_RNN(
            xnor_md(src_iter_c_desc, diff_src_iter_c_desc), VERBOSE_NULL_ARG);
    VCONDCHECK_RNN(
            xnor_md(dst_iter_desc, diff_dst_iter_desc), VERBOSE_NULL_ARG);
    VCONDCHECK_RNN(
            xnor_md(dst_iter_c_desc, diff_dst_iter_c_desc), VERBOSE_NULL_ARG);

    VCHECK_RNN(check_runtime_dims_or_strides({src_layer_desc, src_iter_desc,
                       src_iter_c_desc, attention_desc, weights_layer_desc,
                       weights_iter_desc, weights_peephole_desc,
                       weights_projection_desc, bias_desc, dst_layer_desc,
                       dst_iter_desc, dst_iter_c_desc, diff_src_layer_desc,
                       diff_src_iter_desc, diff_src_iter_c_desc,
                       diff_attention_desc, diff_weights_layer_desc,
                       diff_weights_iter_desc, diff_weights_peephole_desc,
                       diff_weights_projection_desc, diff_bias_desc,
                       diff_dst_layer_desc, diff_dst_iter_desc,
                       diff_dst_iter_c_desc}),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    auto rd = rnn_desc_t();

    rd.primitive_kind = primitive_kind::rnn;
    rd.prop_kind = prop_kind;
    rd.cell_kind = cell_kind;
    rd.direction = direction;

    maybe_init_md(rd.src_layer_desc, src_layer_desc);
    maybe_init_md(rd.src_iter_desc, src_iter_desc);
    maybe_init_md(rd.src_iter_c_desc, src_iter_c_desc);
    maybe_init_md(rd.weights_layer_desc, weights_layer_desc);
    maybe_init_md(rd.weights_iter_desc, weights_iter_desc);
    maybe_init_md(rd.weights_peephole_desc, weights_peephole_desc);
    if (is_augru) maybe_init_md(rd.weights_peephole_desc, attention_desc);
    maybe_init_md(rd.weights_projection_desc, weights_projection_desc);
    maybe_init_md(rd.bias_desc, bias_desc);
    maybe_init_md(rd.dst_layer_desc, dst_layer_desc);
    maybe_init_md(rd.dst_iter_desc, dst_iter_desc);
    maybe_init_md(rd.dst_iter_c_desc, dst_iter_c_desc);
    maybe_init_md(rd.diff_src_layer_desc, diff_src_layer_desc);
    maybe_init_md(rd.diff_src_iter_desc, diff_src_iter_desc);
    maybe_init_md(rd.diff_src_iter_c_desc, diff_src_iter_c_desc);
    maybe_init_md(rd.diff_weights_layer_desc, diff_weights_layer_desc);
    maybe_init_md(rd.diff_weights_iter_desc, diff_weights_iter_desc);
    maybe_init_md(rd.diff_weights_peephole_desc, diff_weights_peephole_desc);
    if (is_augru)
        maybe_init_md(rd.diff_weights_peephole_desc, diff_attention_desc);
    maybe_init_md(
            rd.diff_weights_projection_desc, diff_weights_projection_desc);
    maybe_init_md(rd.diff_bias_desc, diff_bias_desc);
    maybe_init_md(rd.diff_dst_layer_desc, diff_dst_layer_desc);
    maybe_init_md(rd.diff_dst_iter_desc, diff_dst_iter_desc);
    maybe_init_md(rd.diff_dst_iter_c_desc, diff_dst_iter_c_desc);

    rd.flags = flags;
    rd.activation_kind = activation;
    rd.alpha = alpha;
    rd.beta = beta;

    CHECK(check_data_type_consistency_fwd(rd));
    CHECK(check_data_type_consistency_bwd(rd));

    CHECK(check_dim_consistency(rd));

    *rnn_desc = rd;

    return success;
}

status_t rnn_attr_check(const rnn_desc_t &desc, const engine_t *engine,
        const primitive_attr_t *attr) {
    using smask_t = primitive_attr_t::skip_mask_t;

    if (attr == nullptr) return status::success;
    if (attr->has_default_values()) return status::success;

    primitive_attr_t::skip_mask_t attr_mask
            = primitive_attr_t::skip_mask_t::rnn_tparams;
    // Check attributes
    if (utils::one_of(desc.prop_kind, prop_kind::forward_inference,
                prop_kind::forward_training)) {
        const data_type_t wei_layer_dt = desc.weights_layer_desc.data_type;

        const bool is_int8 = wei_layer_dt == data_type::s8;
        if (is_int8
                && one_of(desc.cell_kind, alg_kind::vanilla_lstm,
                        alg_kind::vanilla_gru))
            attr_mask |= smask_t::rnn_data_qparams
                    | smask_t::rnn_weights_qparams
                    | smask_t::rnn_weights_projection_qparams;

        VCONDCHECK_RNN_UNIMPL(
                attr->has_default_values(attr_mask), VERBOSE_UNSUPPORTED_ATTR);

        // Check weights scales
        if (!attr->rnn_weights_qparams_.has_default_values()) {
            const auto &sc = attr->rnn_weights_qparams_;
            const int mask = sc.mask_;

            switch (desc.weights_layer_desc.ndims) {
                case 5:
                    VCONDCHECK_RNN_UNIMPL(utils::one_of(mask, 0, 24),
                            VERBOSE_UNSUPPORTED_SCALES_CFG);
                    break;
                case 4:
                    VCONDCHECK_RNN_UNIMPL(utils::one_of(mask, 0, 8),
                            VERBOSE_UNSUPPORTED_SCALES_CFG);
                    break;
                default:
                    VCONDCHECK_RNN_UNIMPL(
                            mask == 0, VERBOSE_UNSUPPORTED_SCALES_CFG);
            }
        }
    } else {
        VCONDCHECK_RNN_UNIMPL(
                attr->has_default_values(attr_mask), VERBOSE_UNSUPPORTED_ATTR);
    }

    return status::success;
}

} // namespace

/* Public C Api */

status_t dnnl_vanilla_rnn_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        dnnl_prop_kind_t prop_kind, const dnnl_alg_kind_t activation,
        const dnnl_rnn_direction_t direction,
        const memory_desc_t *src_layer_desc, const memory_desc_t *src_iter_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc, const memory_desc_t *dst_iter_desc,
        unsigned flags, float alpha, float beta, const primitive_attr_t *attr) {

    auto rnn_desc = rnn_desc_t();
    CHECK(rnn_common_fwd_desc_init(&rnn_desc, prop_kind, dnnl_vanilla_rnn,
            direction, src_layer_desc, src_iter_desc, nullptr, nullptr,
            weights_layer_desc, weights_iter_desc, nullptr, nullptr, bias_desc,
            dst_layer_desc, dst_iter_desc, nullptr, flags, activation, alpha,
            beta));
    CHECK(rnn_attr_check(rnn_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&rnn_desc, nullptr, attr);
}

status_t dnnl_lstm_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const memory_desc_t *src_layer_desc, const memory_desc_t *src_iter_desc,
        const memory_desc_t *src_iter_c_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc,
        const memory_desc_t *weights_peephole_desc,
        const memory_desc_t *weights_projection_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_layer_desc,
        const memory_desc_t *dst_iter_desc,
        const memory_desc_t *dst_iter_c_desc, unsigned flags,
        const primitive_attr_t *attr) {

    auto rnn_desc = rnn_desc_t();
    CHECK(rnn_common_fwd_desc_init(&rnn_desc, prop_kind, dnnl_vanilla_lstm,
            direction, src_layer_desc, src_iter_desc, src_iter_c_desc, nullptr,
            weights_layer_desc, weights_iter_desc, weights_peephole_desc,
            weights_projection_desc, bias_desc, dst_layer_desc, dst_iter_desc,
            dst_iter_c_desc, flags));
    CHECK(rnn_attr_check(rnn_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&rnn_desc, nullptr, attr);
}

status_t dnnl_gru_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const memory_desc_t *src_layer_desc, const memory_desc_t *src_iter_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc, const memory_desc_t *dst_iter_desc,
        unsigned flags, const primitive_attr_t *attr) {

    auto rnn_desc = rnn_desc_t();
    CHECK(rnn_common_fwd_desc_init(&rnn_desc, prop_kind, dnnl_vanilla_gru,
            direction, src_layer_desc, src_iter_desc, nullptr, nullptr,
            weights_layer_desc, weights_iter_desc, nullptr, nullptr, bias_desc,
            dst_layer_desc, dst_iter_desc, nullptr, flags));
    CHECK(rnn_attr_check(rnn_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&rnn_desc, nullptr, attr);
}

status_t dnnl_lbr_gru_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const memory_desc_t *src_layer_desc, const memory_desc_t *src_iter_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc, const memory_desc_t *dst_iter_desc,
        unsigned flags, const primitive_attr_t *attr) {

    auto rnn_desc = rnn_desc_t();
    CHECK(rnn_common_fwd_desc_init(&rnn_desc, prop_kind, dnnl_lbr_gru,
            direction, src_layer_desc, src_iter_desc, nullptr, nullptr,
            weights_layer_desc, weights_iter_desc, nullptr, nullptr, bias_desc,
            dst_layer_desc, dst_iter_desc, nullptr, flags));
    CHECK(rnn_attr_check(rnn_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&rnn_desc, nullptr, attr);
}

status_t dnnl_augru_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const memory_desc_t *src_layer_desc, const memory_desc_t *src_iter_desc,
        const memory_desc_t *attention_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc, const memory_desc_t *dst_iter_desc,
        unsigned flags, const primitive_attr_t *attr) {

    auto rnn_desc = rnn_desc_t();
    CHECK(rnn_common_fwd_desc_init(&rnn_desc, prop_kind, dnnl_vanilla_augru,
            direction, src_layer_desc, src_iter_desc, nullptr, attention_desc,
            weights_layer_desc, weights_iter_desc, nullptr, nullptr, bias_desc,
            dst_layer_desc, dst_iter_desc, nullptr, flags));
    CHECK(rnn_attr_check(rnn_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&rnn_desc, nullptr, attr);
}

status_t dnnl_lbr_augru_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const memory_desc_t *src_layer_desc, const memory_desc_t *src_iter_desc,
        const memory_desc_t *attention_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc, const memory_desc_t *dst_iter_desc,
        unsigned flags, const primitive_attr_t *attr) {

    auto rnn_desc = rnn_desc_t();
    CHECK(rnn_common_fwd_desc_init(&rnn_desc, prop_kind, dnnl_lbr_augru,
            direction, src_layer_desc, src_iter_desc, nullptr, attention_desc,
            weights_layer_desc, weights_iter_desc, nullptr, nullptr, bias_desc,
            dst_layer_desc, dst_iter_desc, nullptr, flags));
    CHECK(rnn_attr_check(rnn_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&rnn_desc, nullptr, attr);
}

status_t dnnl_vanilla_rnn_backward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        dnnl_prop_kind_t prop_kind, const dnnl_alg_kind_t activation,
        const dnnl_rnn_direction_t direction,
        const memory_desc_t *src_layer_desc, const memory_desc_t *src_iter_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc, const memory_desc_t *dst_iter_desc,
        const memory_desc_t *diff_src_layer_desc,
        const memory_desc_t *diff_src_iter_desc,
        const memory_desc_t *diff_weights_layer_desc,
        const memory_desc_t *diff_weights_iter_desc,
        const memory_desc_t *diff_bias_desc,
        const memory_desc_t *diff_dst_layer_desc,
        const memory_desc_t *diff_dst_iter_desc, unsigned flags, float alpha,
        float beta, const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto rnn_desc = rnn_desc_t();
    CHECK(rnn_common_bwd_desc_init(&rnn_desc, prop_kind, dnnl_vanilla_rnn,
            direction, src_layer_desc, src_iter_desc, nullptr, nullptr,
            weights_layer_desc, weights_iter_desc, nullptr, nullptr, bias_desc,
            dst_layer_desc, dst_iter_desc, nullptr, diff_src_layer_desc,
            diff_src_iter_desc, nullptr, nullptr, diff_weights_layer_desc,
            diff_weights_iter_desc, nullptr, nullptr, diff_bias_desc,
            diff_dst_layer_desc, diff_dst_iter_desc, nullptr, flags, activation,
            alpha, beta));
    CHECK(rnn_attr_check(rnn_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&rnn_desc, hint_fwd_pd, attr);
}

status_t dnnl_lstm_backward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const memory_desc_t *src_layer_desc, const memory_desc_t *src_iter_desc,
        const memory_desc_t *src_iter_c_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc,
        const memory_desc_t *weights_peephole_desc,
        const memory_desc_t *weights_projection_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_layer_desc,
        const memory_desc_t *dst_iter_desc,
        const memory_desc_t *dst_iter_c_desc,
        const memory_desc_t *diff_src_layer_desc,
        const memory_desc_t *diff_src_iter_desc,
        const memory_desc_t *diff_src_iter_c_desc,
        const memory_desc_t *diff_weights_layer_desc,
        const memory_desc_t *diff_weights_iter_desc,
        const memory_desc_t *diff_weights_peephole_desc,
        const memory_desc_t *diff_weights_projection_desc,
        const memory_desc_t *diff_bias_desc,
        const memory_desc_t *diff_dst_layer_desc,
        const memory_desc_t *diff_dst_iter_desc,
        const memory_desc_t *diff_dst_iter_c_desc, unsigned flags,
        const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto rnn_desc = rnn_desc_t();
    CHECK(rnn_common_bwd_desc_init(&rnn_desc, prop_kind, dnnl_vanilla_lstm,
            direction, src_layer_desc, src_iter_desc, src_iter_c_desc, nullptr,
            weights_layer_desc, weights_iter_desc, weights_peephole_desc,
            weights_projection_desc, bias_desc, dst_layer_desc, dst_iter_desc,
            dst_iter_c_desc, diff_src_layer_desc, diff_src_iter_desc,
            diff_src_iter_c_desc, nullptr, diff_weights_layer_desc,
            diff_weights_iter_desc, diff_weights_peephole_desc,
            diff_weights_projection_desc, diff_bias_desc, diff_dst_layer_desc,
            diff_dst_iter_desc, diff_dst_iter_c_desc, flags));
    CHECK(rnn_attr_check(rnn_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&rnn_desc, hint_fwd_pd, attr);
}

status_t dnnl_gru_backward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const memory_desc_t *src_layer_desc, const memory_desc_t *src_iter_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc, const memory_desc_t *dst_iter_desc,
        const memory_desc_t *diff_src_layer_desc,
        const memory_desc_t *diff_src_iter_desc,
        const memory_desc_t *diff_weights_layer_desc,
        const memory_desc_t *diff_weights_iter_desc,
        const memory_desc_t *diff_bias_desc,
        const memory_desc_t *diff_dst_layer_desc,
        const memory_desc_t *diff_dst_iter_desc, unsigned flags,
        const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto rnn_desc = rnn_desc_t();
    CHECK(rnn_common_bwd_desc_init(&rnn_desc, prop_kind, dnnl_vanilla_gru,
            direction, src_layer_desc, src_iter_desc, nullptr, nullptr,
            weights_layer_desc, weights_iter_desc, nullptr, nullptr, bias_desc,
            dst_layer_desc, dst_iter_desc, nullptr, diff_src_layer_desc,
            diff_src_iter_desc, nullptr, nullptr, diff_weights_layer_desc,
            diff_weights_iter_desc, nullptr, nullptr, diff_bias_desc,
            diff_dst_layer_desc, diff_dst_iter_desc, nullptr, flags));
    CHECK(rnn_attr_check(rnn_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&rnn_desc, hint_fwd_pd, attr);
}

status_t dnnl_lbr_gru_backward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const memory_desc_t *src_layer_desc, const memory_desc_t *src_iter_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc, const memory_desc_t *dst_iter_desc,
        const memory_desc_t *diff_src_layer_desc,
        const memory_desc_t *diff_src_iter_desc,
        const memory_desc_t *diff_weights_layer_desc,
        const memory_desc_t *diff_weights_iter_desc,
        const memory_desc_t *diff_bias_desc,
        const memory_desc_t *diff_dst_layer_desc,
        const memory_desc_t *diff_dst_iter_desc, unsigned flags,
        const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto rnn_desc = rnn_desc_t();
    CHECK(rnn_common_bwd_desc_init(&rnn_desc, prop_kind, dnnl_lbr_gru,
            direction, src_layer_desc, src_iter_desc, nullptr, nullptr,
            weights_layer_desc, weights_iter_desc, nullptr, nullptr, bias_desc,
            dst_layer_desc, dst_iter_desc, nullptr, diff_src_layer_desc,
            diff_src_iter_desc, nullptr, nullptr, diff_weights_layer_desc,
            diff_weights_iter_desc, nullptr, nullptr, diff_bias_desc,
            diff_dst_layer_desc, diff_dst_iter_desc, nullptr, flags));
    CHECK(rnn_attr_check(rnn_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&rnn_desc, hint_fwd_pd, attr);
}

status_t dnnl_augru_backward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const memory_desc_t *src_layer_desc, const memory_desc_t *src_iter_desc,
        const memory_desc_t *attention_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc, const memory_desc_t *dst_iter_desc,
        const memory_desc_t *diff_src_layer_desc,
        const memory_desc_t *diff_src_iter_desc,
        const memory_desc_t *diff_attention_desc,
        const memory_desc_t *diff_weights_layer_desc,
        const memory_desc_t *diff_weights_iter_desc,
        const memory_desc_t *diff_bias_desc,
        const memory_desc_t *diff_dst_layer_desc,
        const memory_desc_t *diff_dst_iter_desc, unsigned flags,
        const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto rnn_desc = rnn_desc_t();
    CHECK(rnn_common_bwd_desc_init(&rnn_desc, prop_kind, dnnl_vanilla_augru,
            direction, src_layer_desc, src_iter_desc, nullptr, attention_desc,
            weights_layer_desc, weights_iter_desc, nullptr, nullptr, bias_desc,
            dst_layer_desc, dst_iter_desc, nullptr, diff_src_layer_desc,
            diff_src_iter_desc, nullptr, diff_attention_desc,
            diff_weights_layer_desc, diff_weights_iter_desc, nullptr, nullptr,
            diff_bias_desc, diff_dst_layer_desc, diff_dst_iter_desc, nullptr,
            flags));
    CHECK(rnn_attr_check(rnn_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&rnn_desc, hint_fwd_pd, attr);
}

status_t dnnl_lbr_augru_backward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        dnnl_prop_kind_t prop_kind, dnnl_rnn_direction_t direction,
        const memory_desc_t *src_layer_desc, const memory_desc_t *src_iter_desc,
        const memory_desc_t *attention_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc, const memory_desc_t *dst_iter_desc,
        const memory_desc_t *diff_src_layer_desc,
        const memory_desc_t *diff_src_iter_desc,
        const memory_desc_t *diff_attention_desc,
        const memory_desc_t *diff_weights_layer_desc,
        const memory_desc_t *diff_weights_iter_desc,
        const memory_desc_t *diff_bias_desc,
        const memory_desc_t *diff_dst_layer_desc,
        const memory_desc_t *diff_dst_iter_desc, unsigned flags,
        const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto rnn_desc = rnn_desc_t();
    CHECK(rnn_common_bwd_desc_init(&rnn_desc, prop_kind, dnnl_lbr_augru,
            direction, src_layer_desc, src_iter_desc, nullptr, attention_desc,
            weights_layer_desc, weights_iter_desc, nullptr, nullptr, bias_desc,
            dst_layer_desc, dst_iter_desc, nullptr, diff_src_layer_desc,
            diff_src_iter_desc, nullptr, diff_attention_desc,
            diff_weights_layer_desc, diff_weights_iter_desc, nullptr, nullptr,
            diff_bias_desc, diff_dst_layer_desc, diff_dst_iter_desc, nullptr,
            flags));
    CHECK(rnn_attr_check(rnn_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&rnn_desc, hint_fwd_pd, attr);
}
