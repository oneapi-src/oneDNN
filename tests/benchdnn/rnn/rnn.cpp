/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#include <float.h>
#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "dnnl.h"

#include "src/common/dnnl_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "norm.hpp"

#include "rnn/rnn.hpp"
#include "rnn/rnn_aux.hpp"

#define COMPARE_DAT(rc, a, lay) \
    do { \
        dnn_mem_t CONCAT2(a, _dt_plain)(CONCAT2(a, _dt), fp, lay, engine_tgt); \
        rc |= compare_dat( \
                p, a, CONCAT2(a, _dt_plain), CONCAT2(a, _fp), r, true); \
    } while (0)

// Using hidden attr API for testing RNN
dnnl_status_t dnnl_primitive_attr_set_rnn_tparams(dnnl_primitive_attr_t attr,
        bool mode, dnnl_dim_t ngates, const float *scales, float cscale);

namespace rnn {

void create_dnnl_rnn_attr(const prb_t &p, dnnl_primitive_attr_t *dnnl_attr) {
    DNN_SAFE_V(dnnl_primitive_attr_create(dnnl_attr));

    if (p.skip_nonlinear)
        DNN_SAFE_V(dnnl_primitive_attr_set_rnn_tparams(*dnnl_attr, true,
                p.n_gates(), p.linear_scales, p.linear_cscale));

    if (p.scale_policy == policy_t::PER_OC) {
        DNN_SAFE_V(dnnl_primitive_attr_set_rnn_weights_qparams(
                *dnnl_attr, p.dhc * p.n_gates(), 0x18, p.wei_oc_scales));
    } else if (p.scale_policy == policy_t::COMMON && p.wei_scale != 1.) {
        DNN_SAFE_V(dnnl_primitive_attr_set_rnn_weights_qparams(
                *dnnl_attr, 1, 0, &p.wei_scale));
    }

    if (p.data_scale != 1.0 || p.data_shift != 0.0) {
        DNN_SAFE_V(dnnl_primitive_attr_set_rnn_data_qparams(
                *dnnl_attr, p.data_scale, p.data_shift));
    }

    DNN_SAFE_V(dnnl_primitive_attr_set_scratchpad_mode(
            *dnnl_attr, scratchpad_mode));
}

int check_s8s8_reorder(const prb_t &p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    // TODO: enable for all cpu_kind when supported
    if (engine_tgt_kind != dnnl_cpu) return OK;

    // In the main test, we fill buffers with f32 and reorder to s8
    // with quantization.

    // The endgoal is to check that the reorder
    // f32_plain_nonquantized --reorder--> s8_packed_quantized
    // gives the same output as the sequence
    // f32_plain --quant--> s8_plain_quantized --reorder--> s8_packed_quantized

    // Here,
    // 1. we quantize the f32 plain memory to s8 plain memory,
    // 2. we reorder the s8 plain to s8 packed (queried from rnn primitive desc)
    // 3. we check that the two memory are bitwise identical.

    // Note: the two s8 packed memories need to have the same
    // alignment as packed buffer is aligned internally and the offset
    // is kept in the metadata.
    // Works fine with dnn_mem_t as it is align to 2MB large page boundary
    dnn_mem_t mem_s8_src(mem_fp.md_, dnnl_s8, engine_tgt);
    dnn_mem_t mem_s8_dst(mem_dt.md_, dnnl_s8, engine_tgt);

    /* 1. compute f32_plain --quant--> s8_plain_quantized */
    /* Do fixed partitioning to have same filling for any number of threads */
    auto nelems = mem_fp.nelems();
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);
    dnnl::impl::parallel_nd(n_chunks, [&](int idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            const float current_scale = p.scale_policy == policy_t::PER_OC
                    ? p.wei_oc_scales[idx % (p.dhc * p.n_gates())]
                    : p.wei_scale;
            float val_f32 = mem_fp.get_elem(idx);
            //int8_t val_s8 = saturate<dnnl_s8>(val_f32);
            int8_t val_s8 = saturate<dnnl_s8>(val_f32 * current_scale);
            mem_s8_src.set_elem(idx, val_s8);
        }
    });

    /* 2. compute s8_plain_quantized --reorder--> s8_packed_quantized */
    mem_s8_dst.reorder(mem_s8_src);

    /* 3. we check that the two memory are bitwise identical. */
    auto sz = mem_dt.size();
    uint8_t *s8_dst_handle = (uint8_t *)mem_s8_dst;
    uint8_t *mem_dt_handle = (uint8_t *)mem_dt;

    // check that both have the same size
    assert(mem_dt.size() == mem_s8_dst.size());
    // check that both have the same alignment modulo align_data in gemm_pack_storage.hpp
    assert((uint64_t)s8_dst_handle % 0x1000
            == (uint64_t)mem_dt_handle % 0x1000);
    for (size_t i = 0; i < sz; ++i) {
        if (s8_dst_handle[i] != mem_dt_handle[i]) { return FAIL; }
    }

    return OK;
}

int fill_memory(const prb_t &p, rnn_data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, dnnl_data_type_t dt, float mean, float stddev,
        float min, float max, const_dnnl_primitive_attr_t attr = nullptr,
        bool flip_sign = false) {
    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return OK;
    assert(mem_dt.nelems() == mem_fp.nelems());

    // For non-int8 RNN the data is filled according to cfg directly.
    // However, for int8 RNN we have slightly obscure logic, at least for now:
    // 1. cfg describes the quantized data;
    // 2. We fill first f32 de-quantized data, by inverse-applying the scale
    //    and shift to the data generated by cfg distribution;
    // 3. We reorder the data for the DNNL RNN primitive
    // 4. Q10n of the data for reference benchdnn RNN:
    //    4.a. If the tensor is weights -- q10n it here;
    //    4.b. If the tensor is data -- reference benchdnn RNN will quantize it.

    // default, nothing to be done
    float scale = 1.f, shift = 0.f;
    bool need_recompute_scale = false;
    const_dnnl_primitive_attr_t reorder_attr = nullptr;

    if (p.is_int8()) {
        if (kind == weights_input || kind == weights_states) {
            need_recompute_scale = p.scale_policy == policy_t::PER_OC;
            if (!need_recompute_scale) scale = p.wei_scale;
        } else if (dt == dnnl_u8 && (kind == input || kind == states)) {
            scale = p.data_scale;
            shift = p.data_shift;
        }
        // pass rnn attributes to f32 -> int8 reorders
        if (dt != dnnl_f32) reorder_attr = attr;
    }

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);
    dnnl::impl::parallel_nd(n_chunks, [&](int idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        std::minstd_rand msr;
        msr.seed(idx_start + kind);
        std::normal_distribution<float> gen(mean, stddev);
        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float val = round_to_nearest_representable(dt, gen(msr));
            val = MAX2(MIN2(val, max), min);

            const float current_scale = need_recompute_scale
                    ? p.wei_oc_scales[idx % (p.dhc * p.n_gates())]
                    : scale;
            val = (val - shift) / current_scale; // change only int8-case

            // Vanilla RNN with RELU testing related only: flip the sign of
            // inputs for `mb` == 0 to test RELU part
            if (flip_sign) {
                assert(kind == input || kind == states);
                auto ld = kind == input ? p.slc : p.sic;
                if (idx % (p.mb * ld) < ld) val *= -1;
            }
            mem_fp.set_elem(idx, val);
        }
    });

    mem_dt.reorder(mem_fp, {reorder_attr});
    if ((reorder_attr != nullptr) && (dt == dnnl_s8))
        if (check_s8s8_reorder(p, mem_dt, mem_fp) != OK) return FAIL;

    // Bullet 4.a holds: quantize weights for int8 benchdnn reference RNN
    if (p.is_int8() && (kind == weights_input || kind == weights_states)) {
        dnnl::impl::parallel_nd(n_chunks, [&](int idx_chunk) {
            int64_t idx_start = idx_chunk * chunk_size;
            int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
            for (int64_t idx = idx_start; idx < idx_end; ++idx) {
                const float current_scale = need_recompute_scale
                        ? p.wei_oc_scales[idx % (p.dhc * p.n_gates())]
                        : scale;

                float val = ((float *)mem_fp)[idx];
                val = round(current_scale * val);

                mem_fp.set_elem(idx, MAX2(MIN2(val, max), min));
            }
        });
    }

    return OK;
}

int fill_memory(const prb_t &p, rnn_data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, const_dnnl_primitive_attr_t attr = nullptr,
        bool flip_sign = false) {
    const dt_conf_t::entry_t &c = p.cfg[kind];
    return fill_memory(p, kind, mem_dt, mem_fp, c.dt, c.f_mean, c.f_stddev,
            c.f_min, c.f_max, attr, flip_sign);
}

int fill_activation(const prb_t &p, rnn_data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, const_dnnl_primitive_attr_t attr = nullptr) {
    // In general, we mostly want to use positive values to avoid
    // cancellation from happening during computation.  The only case
    // where we actually want negative values to appear is for 1 layer
    // 1 iteration tests using vanilla_rnn and non-zero alpha. In that
    // case, we want to check that alpha is applied accordingly. Here
    // skip_nonlinear is checked as we want to test relu with non-zero
    // alpha, and not the linear function that would replace it under
    // skip_nonlinear=true.
    bool flip_sign = p.skip_nonlinear == false && p.alg == VANILLA_RNN
            && p.activation == RELU && (kind == input || kind == states);
    return fill_memory(p, kind, mem_dt, mem_fp, attr, flip_sign);
}

int fill_c_states(const prb_t &p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        const_dnnl_primitive_attr_t attr = nullptr) {
    const bool special_case = p.prop == dnnl_backward && p.skip_nonlinear;
    if (!special_case) return fill_memory(p, c_states, mem_dt, mem_fp, attr);

    // The scaling factors in tparams when testing backward are common for
    // for forward and backward passes, and computed as 1 over maximum of
    // the accumulation chain:
    // - ~n_gates on FWD
    // - ~dhc * n_gates on BWD_D
    // - ~mb * n_gates on BWD_W
    //
    // This makes tparam relatively small for the forward pass (compare to
    // the forward pass when we test forward only). This in turn, makes
    // c_states converge relatively fast to the value ~ i_gate * c_gate,
    // which is (typically) way smaller than the original distribution for
    // c_states.
    //
    // TODO: use different tparams for forward and
    //       backward passes when testing BWD_DW.
    //
    // The problem appears on backward pass. Consider diff_f_gate that
    // contributes to backward weights when batch or number of iterations
    // is big:
    //   diff_f_gate[iter] = src_c_state[iter] * diff_dst[iter].
    //   diff_weights += ~diff_f_gate[iter].
    //
    // Assume, that diff_dst[iter] is always about the same for every iter.
    // Since src_c_state[0] >> src_c_state[iter] for iter > 0, this makes the
    // diff_weight be highly dependent on the order of accumulating the
    // diff_f_gate[iter].
    //
    // Originally we had something like:
    // diff_weights = v + v * 10^-5 + ... + v * 10^-5 (n_iter * MB summands).
    // Depending on the order of summation the difference might exceed the
    // typical bound approximation: coefficient * log(number_of_summands).
    //
    // Anyways, the algorithm below tries to put the first src_c_state[iter = 0]
    // in the same ballpark as all the subsequent src_c_state[iter > 0].
    //
    // The estimation is based on the following rough assumptions:
    //   c_state[iter+1] = f_gate * c_state[iter] + i_gate * c_gate
    //                  ~= f_gate * small_value   + i_gate * c_gate
    //                  ~=                          i_gate * c_gate.
    //   i_gate ~= tparams[i_gate] * (
    //              1 / ngates * mean_src_layer +
    //              1 / ngates * mean_src_iter  +
    //              mean_bias);
    //
    // Same for c_gate.
    // The (1 / ngates) factor is taken from fill_weights().

    float expect_gemm_output = (1.f / p.n_gates()) * p.cfg[input].f_mean
            + (1.f / p.n_gates()) * p.cfg[states].f_mean + p.cfg[bias].f_mean;
    float expect_i_gate = (float)p.linear_scales[LSTM_I] * expect_gemm_output;
    float expect_c_gate = (float)p.linear_scales[LSTM_C] * expect_gemm_output;
    float expect_c_state_mean = expect_i_gate * expect_c_gate;

    float adjust_factor = 1;

    const bool need_adjust = expect_c_state_mean < p.cfg[c_states].f_mean
            && p.cfg[c_states].f_mean != 0;
    if (need_adjust)
        adjust_factor = expect_c_state_mean / p.cfg[c_states].f_mean;

    const dt_conf_t::entry_t &c = p.cfg[c_states];
    return fill_memory(p, c_states, mem_dt, mem_fp, c.dt,
            c.f_mean * adjust_factor, c.f_stddev * adjust_factor,
            c.f_min * adjust_factor, c.f_max * adjust_factor, attr);
}

int fill_weights(const prb_t &p, rnn_data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, const_dnnl_primitive_attr_t attr = nullptr) {

    const dt_conf_t::entry_t &c = p.cfg[kind];
    if (c.dt == dnnl_s8) return fill_memory(p, kind, mem_dt, mem_fp, attr);

    auto dims = mem_fp.md_.dims;
    auto L = dims[0];
    auto D = dims[1];
    auto I = dims[2];
    auto G = dims[3];
    auto O = dims[4];

    for (int64_t i = 0; i < mem_dt.nelems(); i++)
        mem_fp.set_elem(i, 0.0f);

    for_(int64_t l = 0; l < L; l++)
    for_(int64_t d = 0; d < D; d++)
    for_(int64_t g = 0; g < G; g++)
    for (int64_t o = 0; o < O; o++) {
        const float val
                = round_to_nearest_representable(c.dt, 1.f / p.n_gates());
        auto i_off = ((o + g * 7 + d * 11 + l * 13) % I);
        mem_fp.set_elem(
                l * D * I * G * O + d * I * G * O + i_off * G * O + g * O + o,
                val);
    }

    mem_dt.reorder(mem_fp);
    return OK;
}

int fill_bias(const prb_t &p, rnn_data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp) {
    // To reduce likelihood of cancellation happening in bwd by bias,
    // (especially for GRU), we want diff_bias to be sparse
    auto dims = mem_fp.md_.dims;
    auto L = dims[0];
    auto D = dims[1];
    auto G = dims[2];
    auto O = dims[3];

    std::minstd_rand msr;
    std::normal_distribution<float> gen(
            p.cfg[kind].f_mean, p.cfg[kind].f_stddev);
    msr.seed(kind);

    for_(int64_t l = 0; l < L; l++)
    for_(int64_t d = 0; d < D; d++)
    for_(int64_t g = 0; g < G; g++)
    for (int64_t o = 0; o < O; o++) {
        auto idx = l * D * G * O + d * G * O + g * O + o;
        auto val = round_to_nearest_representable(
                p.cfg[kind].dt, gen(msr) * flip_coin(idx, 0.05f));
        mem_fp.set_elem(idx, val);
    }
    mem_dt.reorder(mem_fp);
    return OK;
}

inline int init_pd(
        const prb_t &p, dnnl_primitive_desc_t &rpd, res_t *r, bool is_fwd) {
    dnnl_rnn_desc_t rd;
    dnnl_prop_kind_t fwd_prop = dnnl_prop_kind_undef;
    switch (p.prop) {
        case dnnl_forward: fwd_prop = dnnl_forward_inference; break;
        // If we are testing backward, we have to run forward training first
        // in order to generate a valid workspace.
        case dnnl_backward: fwd_prop = dnnl_forward_training; break;
        default: DNN_SAFE(dnnl_invalid_arguments, CRIT);
    }

    const bool is_gru_lbr = p.alg == LBR_GRU;
    // Enable testing non trivial strides in correctness mode and non-int8
    // FIXME: enable the stride testing back when the corresponding missing
    //        reorder added to the library;
    // TODO:  testing with non-trivial stride should be a testing option!
    int the_stride = (bench_mode == CORR && !p.is_int8()) ? 1 : 0;
    /// @todo we need to add stride support for diff_* tensors too
    dnnl_memory_desc_t input_d, states_d, c_states_d, weights_input_d,
            weights_states_d, weights_peephole_d {}, bias_d, dst_last_layer_d,
            dst_last_iteration_d, dst_c_last_iteration_d, diff_input_d,
            diff_states_d, diff_c_states_d, diff_weights_input_d,
            diff_weights_states_d, diff_weights_peephole_d {}, diff_bias_d,
            diff_last_layer_d, diff_last_iteration_d, diff_c_last_iteration_d;

    // dimensions with ref
    dnnl_dims_t input_dims = {p.n_iter, p.mb, p.slc};
    // bidirectional = 2, s for lstm = 2, for all other = 1
    dnnl_dims_t weights_input_dims
            = {p.n_layer, p.n_dir(), p.slc, p.n_gates(), p.dhc};
    dnnl_dims_t weights_states_dims
            = {p.n_layer, p.n_dir(), p.sic, p.n_gates(), p.dhc};
    dnnl_dims_t weights_peephole_dims = {p.n_layer, p.n_dir(), 3, p.dhc};
    dnnl_dims_t bias_dims
            = {p.n_layer, p.n_dir(), p.n_gates() + is_gru_lbr, p.dhc};
    // dnnl_tnc
    int64_t lastlay_dlc
            = (p.direction == dnnl_bidirectional_concat) ? 2 * p.dlc : p.dlc;
    dnnl_dims_t dst_last_layer_dims = {p.n_iter, p.mb, lastlay_dlc};

    DNN_SAFE(dnnl_memory_desc_init_by_tag(
                     &input_d, 3, input_dims, p.cfg[input].dt, dnnl_tnc),
            WARN);
    input_d.format_desc.blocking.strides[0] += the_stride;

    dnnl_dims_t states_dims = {p.n_layer, p.n_dir(), p.mb, p.sic};
    DNN_SAFE(dnnl_memory_desc_init_by_tag(
                     &states_d, 4, states_dims, p.cfg[states].dt, dnnl_ldnc),
            WARN);
    states_d.format_desc.blocking.strides[2] = p.sic + the_stride;
    for (int d = 1; d >= 0; --d)
        states_d.format_desc.blocking.strides[d]
                = states_d.format_desc.blocking.strides[d + 1]
                * states_d.dims[d + 1];

    dnnl_dims_t c_states_dims = {p.n_layer, p.n_dir(), p.mb, p.dhc};
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&c_states_d, 4, c_states_dims,
                     p.cfg[c_states].dt, dnnl_ldnc),
            WARN);
    c_states_d.format_desc.blocking.strides[2] = p.dhc + the_stride;
    for (int d = 1; d >= 0; --d)
        c_states_d.format_desc.blocking.strides[d]
                = c_states_d.format_desc.blocking.strides[d + 1]
                * c_states_d.dims[d + 1];

    DNN_SAFE(dnnl_memory_desc_init_by_tag(&weights_input_d, 5,
                     weights_input_dims, p.cfg[weights_input].dt,
                     dnnl_format_tag_any),
            WARN);

    DNN_SAFE(dnnl_memory_desc_init_by_tag(&weights_states_d, 5,
                     weights_states_dims, p.cfg[weights_states].dt,
                     dnnl_format_tag_any),
            WARN);

    if (p.is_lstm_peephole()) {
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&weights_peephole_d, 4,
                         weights_peephole_dims, p.cfg[weights_peephole].dt,
                         dnnl_ldgo),
                WARN);
    }

    DNN_SAFE(dnnl_memory_desc_init_by_tag(&bias_d, 4, bias_dims, p.cfg[bias].dt,
                     dnnl_format_tag_any),
            WARN);

    DNN_SAFE(dnnl_memory_desc_init_by_tag(&dst_last_layer_d, 3,
                     dst_last_layer_dims, p.cfg[dst_last_layer].dt, dnnl_tnc),
            WARN);
    dst_last_layer_d.format_desc.blocking.strides[0] += the_stride;

    dnnl_dims_t dst_last_iteration_dims = {p.n_layer, p.n_dir(), p.mb, p.dhc};
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&dst_last_iteration_d, 4,
                     dst_last_iteration_dims, p.cfg[dst_last_iteration].dt,
                     dnnl_ldnc),
            WARN);
    dst_last_iteration_d.format_desc.blocking.strides[2] = p.dhc + the_stride;
    for (int d = 1; d >= 0; --d)
        dst_last_iteration_d.format_desc.blocking.strides[d]
                = dst_last_iteration_d.format_desc.blocking.strides[d + 1]
                * dst_last_iteration_d.dims[d + 1];

    dnnl_dims_t dst_c_last_iteration_dims = {p.n_layer, p.n_dir(), p.mb, p.dhc};
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&dst_c_last_iteration_d, 4,
                     dst_c_last_iteration_dims, p.cfg[dst_c_last_iteration].dt,
                     dnnl_ldnc),
            WARN);

    dst_c_last_iteration_d.format_desc.blocking.strides[2] = p.dhc + the_stride;
    for (int d = 1; d >= 0; --d)
        dst_c_last_iteration_d.format_desc.blocking.strides[d]
                = dst_c_last_iteration_d.format_desc.blocking.strides[d + 1]
                * dst_c_last_iteration_d.dims[d + 1];

    // Initializing the forward pass
    // When inference, we use forward_inference
    // When training, we use forward_training
    if (is_fwd) {
        DNN_SAFE(init_rnn_fwd_desc(&rd, p, fwd_prop, &input_d, &states_d,
                         &c_states_d, &weights_input_d, &weights_states_d,
                         &weights_peephole_d, &bias_d, &dst_last_layer_d,
                         &dst_last_iteration_d, &dst_c_last_iteration_d),
                WARN);
    } else {
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_input_d, 3, input_dims,
                         p.cfg[dst_diff_input].dt, dnnl_format_tag_any),
                WARN);
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_states_d, 4, states_dims,
                         p.cfg[dst_diff_states].dt, dnnl_format_tag_any),
                WARN);
        DNN_SAFE(
                dnnl_memory_desc_init_by_tag(&diff_c_states_d, 4, c_states_dims,
                        p.cfg[dst_diff_c_states].dt, dnnl_format_tag_any),
                WARN);
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_weights_input_d, 5,
                         weights_input_dims, p.cfg[dst_diff_weights_input].dt,
                         dnnl_format_tag_any),
                WARN);
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_weights_states_d, 5,
                         weights_states_dims, p.cfg[dst_diff_weights_states].dt,
                         dnnl_format_tag_any),
                WARN);
        if (p.is_lstm_peephole()) {
            DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_weights_peephole_d, 4,
                             weights_peephole_dims,
                             p.cfg[dst_diff_weights_peephole].dt, dnnl_ldgo),
                    WARN);
        }
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_bias_d, 4, bias_dims,
                         p.cfg[dst_diff_bias].dt, dnnl_format_tag_any),
                WARN);
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_last_layer_d, 3,
                         dst_last_layer_dims, p.cfg[diff_last_layer].dt,
                         dnnl_format_tag_any),
                WARN);
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_last_iteration_d, 4,
                         dst_last_iteration_dims, p.cfg[diff_last_iteration].dt,
                         dnnl_format_tag_any),
                WARN);
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_c_last_iteration_d, 4,
                         dst_c_last_iteration_dims,
                         p.cfg[diff_c_last_iteration].dt, dnnl_format_tag_any),
                WARN);
        DNN_SAFE(init_rnn_bwd_desc(&rd, p, p.prop, &input_d, &states_d,
                         &c_states_d, &weights_input_d, &weights_states_d,
                         &weights_peephole_d, &bias_d, &dst_last_layer_d,
                         &dst_last_iteration_d, &dst_c_last_iteration_d,
                         &diff_input_d, &diff_states_d, &diff_c_states_d,
                         &diff_weights_input_d, &diff_weights_states_d,
                         &diff_weights_peephole_d, &diff_bias_d,
                         &diff_last_layer_d, &diff_last_iteration_d,
                         &diff_c_last_iteration_d),
                WARN);
    }

    dnnl_primitive_attr_t dnnl_attr;
    create_dnnl_rnn_attr(p, &dnnl_attr);
    dnnl_status_t init_status = dnnl_primitive_desc_create(
            &rpd, &rd, dnnl_attr, engine_tgt, NULL);
    dnnl_primitive_attr_destroy(dnnl_attr);
    if (init_status == dnnl_unimplemented) return r->state = UNIMPLEMENTED, OK;
    SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(rpd);
    BENCHDNN_PRINT(5, "dnnl implementation: %s\n", impl_str);

    return OK;
}

int doit(const prb_t &p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;

    dnnl_primitive_desc_t rpd;
    SAFE(init_pd(p, rpd, r, true), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

    dnnl_primitive_t c;
    auto cleanup = [&]() {
        DNN_SAFE(dnnl_primitive_destroy(c), CRIT);
        return OK;
    };

    DNN_SAFE(dnnl_primitive_create(&c, rpd), WARN);
    DNN_SAFE(dnnl_primitive_desc_destroy(rpd), CRIT);

    const_dnnl_primitive_desc_t const_fpd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(c, &const_fpd), CRIT);

    if (dnn_mem_t::check_mem_size(const_fpd) != OK) {
        DNN_SAFE(dnnl_primitive_destroy(c), CRIT);
        return r->state = SKIPPED, OK;
    }

    const auto q = [](const_dnnl_primitive_desc_t pd,
                           int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(pd, dnnl_query_exec_arg_md, index);
    };

    const auto &input_md = q(const_fpd, DNNL_ARG_SRC_LAYER);
    const auto &states_md = q(const_fpd, DNNL_ARG_SRC_ITER);
    const auto &c_states_md = q(const_fpd, DNNL_ARG_SRC_ITER_C);
    const auto &weights_input_md = q(const_fpd, DNNL_ARG_WEIGHTS_LAYER);
    const auto &weights_states_md = q(const_fpd, DNNL_ARG_WEIGHTS_ITER);
    const auto &weights_peephole_md = q(const_fpd, DNNL_ARG_WEIGHTS_PEEPHOLE);
    const auto &bias_md = q(const_fpd, DNNL_ARG_BIAS);
    const auto &dst_last_layer_md = q(const_fpd, DNNL_ARG_DST_LAYER);
    const auto &dst_last_iteration_md = q(const_fpd, DNNL_ARG_DST_ITER);
    const auto &dst_c_last_iteration_md = q(const_fpd, DNNL_ARG_DST_ITER_C);
    const auto &workspace_md = q(const_fpd, DNNL_ARG_WORKSPACE);
    const auto &scratchpad_md = q(const_fpd, DNNL_ARG_SCRATCHPAD);

    dnn_mem_t input_dt(input_md, engine_tgt);
    dnn_mem_t states_dt(states_md, engine_tgt);
    dnn_mem_t c_states_dt(c_states_md, engine_tgt);
    dnn_mem_t weights_input_dt(weights_input_md, engine_tgt);
    dnn_mem_t weights_states_dt(weights_states_md, engine_tgt);
    dnn_mem_t weights_peephole_dt(weights_peephole_md, engine_tgt);
    dnn_mem_t bias_dt(bias_md, engine_tgt);
    dnn_mem_t dst_last_layer_dt(dst_last_layer_md, engine_tgt);
    dnn_mem_t dst_last_iteration_dt(dst_last_iteration_md, engine_tgt);
    dnn_mem_t dst_c_last_iteration_dt(dst_c_last_iteration_md, engine_tgt);
    dnn_mem_t workspace_dt(workspace_md, engine_tgt);
    dnn_mem_t scratchpad_dt(scratchpad_md, engine_tgt);

    const auto fp = dnnl_f32;
    dnn_mem_t input_fp(input_md, fp, dnnl_tnc, engine_tgt);
    dnn_mem_t states_fp(states_md, fp, dnnl_ldnc, engine_tgt);
    dnn_mem_t c_states_fp(c_states_md, fp, dnnl_ldnc, engine_tgt);
    dnn_mem_t weights_input_fp(weights_input_md, fp, dnnl_ldigo, engine_tgt);
    dnn_mem_t weights_states_fp(weights_states_md, fp, dnnl_ldigo, engine_tgt);
    dnn_mem_t weights_peephole_fp(
            weights_peephole_md, fp, dnnl_ldgo, engine_tgt);
    dnn_mem_t bias_fp(bias_md, fp, dnnl_ldgo, engine_tgt);
    dnn_mem_t dst_last_layer_fp(dst_last_layer_md, fp, dnnl_tnc, engine_tgt);
    dnn_mem_t dst_last_iteration_fp(
            dst_last_iteration_md, fp, dnnl_ldnc, engine_tgt);
    dnn_mem_t dst_c_last_iteration_fp(
            dst_c_last_iteration_md, fp, dnnl_ldnc, engine_tgt);

    dnn_mem_t bwd_weights_input_dt;
    dnn_mem_t bwd_weights_states_dt;
    dnn_mem_t dst_diff_input_dt;
    dnn_mem_t dst_diff_states_dt;
    dnn_mem_t dst_diff_c_states_dt;
    dnn_mem_t dst_diff_weights_input_dt;
    dnn_mem_t dst_diff_weights_states_dt;
    dnn_mem_t dst_diff_weights_peephole_dt;
    dnn_mem_t dst_diff_bias_dt;
    dnn_mem_t diff_last_layer_dt;
    dnn_mem_t diff_last_iteration_dt;
    dnn_mem_t diff_c_last_iteration_dt;

    // for int8 RNN we need pass attributes for data q10n
    const_dnnl_primitive_attr_t rnn_attr;
    DNN_SAFE(dnnl_primitive_desc_get_attr(const_fpd, &rnn_attr), WARN);

    SAFE(fill_activation(p, input, input_dt, input_fp, rnn_attr), WARN);
    SAFE(fill_activation(p, states, states_dt, states_fp, rnn_attr), WARN);
    if (p.alg == VANILLA_LSTM)
        SAFE(fill_c_states(p, c_states_dt, c_states_fp, rnn_attr), WARN);
    SAFE(fill_weights(p, weights_input, weights_input_dt, weights_input_fp,
                 rnn_attr),
            WARN);
    SAFE(fill_weights(p, weights_states, weights_states_dt, weights_states_fp,
                 rnn_attr),
            WARN);
    SAFE(fill_memory(
                 p, weights_peephole, weights_peephole_dt, weights_peephole_fp),
            WARN);
    SAFE(fill_memory(p, bias, bias_dt, bias_fp), WARN);
    SAFE(fill_activation(
                 p, dst_last_layer, dst_last_layer_dt, dst_last_layer_fp),
            WARN);
    SAFE(fill_activation(p, dst_last_iteration, dst_last_iteration_dt,
                 dst_last_iteration_fp),
            WARN);
    if (p.alg == VANILLA_LSTM)
        SAFE(fill_memory(p, dst_c_last_iteration, dst_c_last_iteration_dt,
                     dst_c_last_iteration_fp),
                WARN);

    args_t args;

    // Running the forward pass
    args.set(DNNL_ARG_SRC_LAYER, input_dt);
    args.set(DNNL_ARG_SRC_ITER, states_dt);
    args.set(DNNL_ARG_SRC_ITER_C, c_states_dt);
    args.set(DNNL_ARG_WEIGHTS_LAYER, weights_input_dt);
    args.set(DNNL_ARG_WEIGHTS_ITER, weights_states_dt);
    args.set(DNNL_ARG_WEIGHTS_PEEPHOLE, weights_peephole_dt);
    args.set(DNNL_ARG_BIAS, bias_dt);
    args.set(DNNL_ARG_DST_LAYER, dst_last_layer_dt);
    args.set(DNNL_ARG_DST_ITER, dst_last_iteration_dt);
    args.set(DNNL_ARG_DST_ITER_C, dst_c_last_iteration_dt);
    args.set(DNNL_ARG_WORKSPACE, workspace_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

    DNN_SAFE_CLEAN(execute_and_wait(c, stream_tgt, args), WARN, cleanup);

    if ((p.prop == dnnl_forward) && (bench_mode & CORR)) {
        compute_ref_fwd(p, input_fp, states_fp, c_states_fp, weights_input_fp,
                weights_states_fp, weights_peephole_fp, bias_fp,
                dst_last_layer_fp, dst_last_iteration_fp,
                dst_c_last_iteration_fp);

        int compare_status = OK;
        COMPARE_DAT(compare_status, dst_last_layer, dnnl_tnc);
        COMPARE_DAT(compare_status, dst_last_iteration, dnnl_ldnc);
        if (p.alg == VANILLA_LSTM)
            COMPARE_DAT(compare_status, dst_c_last_iteration, dnnl_ldnc);
        SAFE_CLEAN(compare_status, WARN, cleanup);
    }

    if (p.prop == dnnl_backward) {
        SAFE(init_pd(p, rpd, r, false), WARN);
        DNN_SAFE(dnnl_primitive_destroy(c), CRIT);
        if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

        DNN_SAFE(dnnl_primitive_create(&c, rpd), WARN);
        DNN_SAFE(dnnl_primitive_desc_destroy(rpd), CRIT);

        const_dnnl_primitive_desc_t const_bpd;
        DNN_SAFE(dnnl_primitive_get_primitive_desc(c, &const_bpd), CRIT);

        if (dnn_mem_t::check_mem_size(const_bpd) != OK) {
            DNN_SAFE(dnnl_primitive_destroy(c), CRIT);
            return r->state = SKIPPED, OK;
        }

        const auto &bwd_weights_input_md = q(const_bpd, DNNL_ARG_WEIGHTS_LAYER);
        const auto &bwd_weights_states_md = q(const_bpd, DNNL_ARG_WEIGHTS_ITER);
        const auto &diff_src_layer_md = q(const_bpd, DNNL_ARG_DIFF_SRC_LAYER);
        const auto &diff_src_iter_md = q(const_bpd, DNNL_ARG_DIFF_SRC_ITER);
        const auto &diff_src_iter_c_md = q(const_bpd, DNNL_ARG_DIFF_SRC_ITER_C);
        const auto &diff_weights_layer_md
                = q(const_bpd, DNNL_ARG_DIFF_WEIGHTS_LAYER);
        const auto &diff_weights_iter_md
                = q(const_bpd, DNNL_ARG_DIFF_WEIGHTS_ITER);
        const auto &diff_weights_peephole_md
                = q(const_bpd, DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE);
        const auto &diff_bias_md = q(const_bpd, DNNL_ARG_DIFF_BIAS);
        const auto &diff_dst_layer_md = q(const_bpd, DNNL_ARG_DIFF_DST_LAYER);
        const auto &diff_dst_iter_md = q(const_bpd, DNNL_ARG_DIFF_DST_ITER);
        const auto &diff_dst_iter_c_md = q(const_bpd, DNNL_ARG_DIFF_DST_ITER_C);
        const auto &bwd_scratchpad_md = q(const_bpd, DNNL_ARG_SCRATCHPAD);

        bwd_weights_input_dt = dnn_mem_t(bwd_weights_input_md, engine_tgt);
        bwd_weights_states_dt = dnn_mem_t(bwd_weights_states_md, engine_tgt);
        dst_diff_input_dt = dnn_mem_t(diff_src_layer_md, engine_tgt);
        dst_diff_states_dt = dnn_mem_t(diff_src_iter_md, engine_tgt);
        dst_diff_c_states_dt = dnn_mem_t(diff_src_iter_c_md, engine_tgt);
        dst_diff_weights_input_dt
                = dnn_mem_t(diff_weights_layer_md, engine_tgt);
        dst_diff_weights_states_dt
                = dnn_mem_t(diff_weights_iter_md, engine_tgt);
        dst_diff_weights_peephole_dt
                = dnn_mem_t(diff_weights_peephole_md, engine_tgt);
        dst_diff_bias_dt = dnn_mem_t(diff_bias_md, engine_tgt);
        diff_last_layer_dt = dnn_mem_t(diff_dst_layer_md, engine_tgt);
        diff_last_iteration_dt = dnn_mem_t(diff_dst_iter_md, engine_tgt);
        diff_c_last_iteration_dt = dnn_mem_t(diff_dst_iter_c_md, engine_tgt);
        scratchpad_dt = dnn_mem_t(bwd_scratchpad_md, engine_tgt);

        dnn_mem_t dst_diff_input_fp(
                diff_src_layer_md, fp, dnnl_tnc, engine_tgt);
        dnn_mem_t dst_diff_states_fp(
                diff_src_iter_md, fp, dnnl_ldnc, engine_tgt);
        dnn_mem_t dst_diff_c_states_fp(
                diff_src_iter_c_md, fp, dnnl_ldnc, engine_tgt);
        dnn_mem_t dst_diff_weights_input_fp(
                diff_weights_layer_md, fp, dnnl_ldigo, engine_tgt);
        dnn_mem_t dst_diff_weights_states_fp(
                diff_weights_iter_md, fp, dnnl_ldigo, engine_tgt);
        dnn_mem_t dst_diff_weights_peephole_fp(
                diff_weights_peephole_md, fp, dnnl_ldgo, engine_tgt);
        dnn_mem_t dst_diff_bias_fp(diff_bias_md, fp, dnnl_ldgo, engine_tgt);
        dnn_mem_t diff_last_layer_fp(
                diff_dst_layer_md, fp, dnnl_tnc, engine_tgt);
        dnn_mem_t diff_last_iteration_fp(
                diff_dst_iter_md, fp, dnnl_ldnc, engine_tgt);
        dnn_mem_t diff_c_last_iteration_fp(
                diff_dst_iter_c_md, fp, dnnl_ldnc, engine_tgt);

        SAFE(bwd_weights_states_dt.reorder(weights_states_dt), WARN);
        SAFE(bwd_weights_input_dt.reorder(weights_input_dt), WARN);
        SAFE(fill_activation(
                     p, dst_diff_input, dst_diff_input_dt, dst_diff_input_fp),
                WARN);
        SAFE(fill_activation(p, dst_diff_states, dst_diff_states_dt,
                     dst_diff_states_fp),
                WARN);
        if (p.alg == VANILLA_LSTM)
            SAFE(fill_memory(p, dst_diff_c_states, dst_diff_c_states_dt,
                         dst_diff_c_states_fp),
                    WARN);
        SAFE(fill_weights(p, dst_diff_weights_input, dst_diff_weights_input_dt,
                     dst_diff_weights_input_fp),
                WARN);
        SAFE(fill_weights(p, dst_diff_weights_states,
                     dst_diff_weights_states_dt, dst_diff_weights_states_fp),
                WARN);
        SAFE(fill_memory(p, dst_diff_weights_peephole,
                     dst_diff_weights_peephole_dt,
                     dst_diff_weights_peephole_fp),
                WARN);
        SAFE(fill_bias(p, dst_diff_bias, dst_diff_bias_dt, dst_diff_bias_fp),
                WARN);
        SAFE(fill_activation(p, diff_last_layer, diff_last_layer_dt,
                     diff_last_layer_fp),
                WARN);
        SAFE(fill_activation(p, diff_last_iteration, diff_last_iteration_dt,
                     diff_last_iteration_fp),
                WARN);
        if (p.alg == VANILLA_LSTM)
            SAFE(fill_memory(p, diff_c_last_iteration, diff_c_last_iteration_dt,
                         diff_c_last_iteration_fp),
                    WARN);

        args.clear();
        args.set(DNNL_ARG_SRC_LAYER, input_dt);
        args.set(DNNL_ARG_SRC_ITER, states_dt);
        args.set(DNNL_ARG_SRC_ITER_C, c_states_dt);
        args.set(DNNL_ARG_WEIGHTS_LAYER, bwd_weights_input_dt);
        args.set(DNNL_ARG_WEIGHTS_ITER, bwd_weights_states_dt);
        args.set(DNNL_ARG_WEIGHTS_PEEPHOLE, weights_peephole_dt);
        args.set(DNNL_ARG_BIAS, bias_dt);
        args.set(DNNL_ARG_DST_LAYER, dst_last_layer_dt);
        args.set(DNNL_ARG_DST_ITER, dst_last_iteration_dt);
        args.set(DNNL_ARG_DST_ITER_C, dst_c_last_iteration_dt);
        args.set(DNNL_ARG_DIFF_DST_LAYER, diff_last_layer_dt);
        args.set(DNNL_ARG_DIFF_DST_ITER, diff_last_iteration_dt);
        args.set(DNNL_ARG_DIFF_DST_ITER_C, diff_c_last_iteration_dt);
        args.set(DNNL_ARG_WORKSPACE, workspace_dt);
        args.set(DNNL_ARG_DIFF_SRC_LAYER, dst_diff_input_dt);
        args.set(DNNL_ARG_DIFF_SRC_ITER, dst_diff_states_dt);
        args.set(DNNL_ARG_DIFF_SRC_ITER_C, dst_diff_c_states_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS_LAYER, dst_diff_weights_input_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS_ITER, dst_diff_weights_states_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE, dst_diff_weights_peephole_dt);
        args.set(DNNL_ARG_DIFF_BIAS, dst_diff_bias_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        DNN_SAFE_CLEAN(execute_and_wait(c, stream_tgt, args), WARN, cleanup);

        if (bench_mode & CORR) {
            compute_ref_bwd(p, input_fp, states_fp, c_states_fp,
                    diff_last_layer_fp, diff_last_iteration_fp,
                    diff_c_last_iteration_fp, weights_input_fp,
                    weights_states_fp, weights_peephole_fp, bias_fp,
                    dst_last_layer_fp, dst_last_iteration_fp,
                    dst_c_last_iteration_fp, dst_diff_input_fp,
                    dst_diff_states_fp, dst_diff_c_states_fp,
                    dst_diff_weights_input_fp, dst_diff_weights_states_fp,
                    dst_diff_weights_peephole_fp, dst_diff_bias_fp);

            int compare_fwd_status = OK;
            COMPARE_DAT(compare_fwd_status, dst_last_layer, dnnl_tnc);
            COMPARE_DAT(compare_fwd_status, dst_last_iteration, dnnl_ldnc);
            if (p.alg == VANILLA_LSTM)
                COMPARE_DAT(
                        compare_fwd_status, dst_c_last_iteration, dnnl_ldnc);
            SAFE_CLEAN(compare_fwd_status, WARN, cleanup);

            int compare_bwd_data_status = OK;
            COMPARE_DAT(compare_bwd_data_status, dst_diff_input, dnnl_tnc);
            COMPARE_DAT(compare_bwd_data_status, dst_diff_states, dnnl_ldnc);
            if (p.alg == VANILLA_LSTM)
                COMPARE_DAT(
                        compare_bwd_data_status, dst_diff_c_states, dnnl_ldnc);
            SAFE_CLEAN(compare_bwd_data_status, WARN, cleanup);

            int compare_bwd_weights_status = OK;
            COMPARE_DAT(compare_bwd_weights_status, dst_diff_weights_input,
                    dnnl_ldigo);
            COMPARE_DAT(compare_bwd_weights_status, dst_diff_weights_states,
                    dnnl_ldigo);
            if (p.is_lstm_peephole())
                COMPARE_DAT(compare_bwd_weights_status,
                        dst_diff_weights_peephole, dnnl_ldgo);
            COMPARE_DAT(compare_bwd_weights_status, dst_diff_bias, dnnl_ldgo);
            SAFE_CLEAN(compare_bwd_weights_status, WARN, cleanup);
        }
    }

    measure_perf(r->timer, c, args);
    cleanup();

    return OK;
}
} // namespace rnn
