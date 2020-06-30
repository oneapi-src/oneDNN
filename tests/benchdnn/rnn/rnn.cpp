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

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "norm.hpp"

#include "rnn/rnn.hpp"
#include "rnn/rnn_aux.hpp"

#define COMPARE_DAT(rc, kind, a, lay) \
    do { \
        dnn_mem_t CONCAT2(a, _dt_plain)( \
                CONCAT2(a, _dt), fp, lay, test_engine); \
        rc |= compare_dat( \
                p, kind, CONCAT2(a, _dt_plain), CONCAT2(a, _fp), r, true); \
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

    DNN_SAFE_V(dnnl_primitive_attr_set_rnn_weights_qparams(
            *dnnl_attr, p.wei_nscales, p.wei_scales_mask, p.wei_scales));

    if (p.data_scale != 1.0 || p.data_shift != 0.0)
        DNN_SAFE_V(dnnl_primitive_attr_set_rnn_data_qparams(
                *dnnl_attr, p.data_scale, p.data_shift));

    DNN_SAFE_V(dnnl_primitive_attr_set_scratchpad_mode(
            *dnnl_attr, scratchpad_mode));
}

int check_s8s8_reorder(const prb_t &p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    // TODO: enable for all cpu_kind when supported
    if (engine_tgt_kind != dnnl_cpu) return OK;

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_DPCPP
    // DPC++ does not provide a simple way to access the underlying
    // buffer alignment.
    return OK;
#endif

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
    dnn_mem_t mem_s8_src(mem_fp.md_, dnnl_s8, get_test_engine());
    dnn_mem_t mem_s8_dst(mem_dt.md_, dnnl_s8, get_test_engine());

    /* 1. compute f32_plain --quant--> s8_plain_quantized */
    /* Do fixed partitioning to have same filling for any number of threads */
    auto nelems = mem_fp.nelems();
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);
    auto quantize = [&](const float *scales, int nscales, int idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            const float current_scale = scales[idx % nscales];
            float val_f32 = mem_fp.get_elem(idx);
            int8_t val_s8 = saturate<dnnl_s8>(val_f32 * current_scale);
            mem_s8_src.set_elem(idx, val_s8);
        }
    };
    dnnl::impl::parallel_nd(n_chunks,
            [&](int idx) { quantize(p.wei_scales, p.wei_nscales, idx); });

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

int fill_memory(const prb_t &p, data_kind_t kind, dnn_mem_t &mem_dt,
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
    // 3. We reorder the data for the oneDNN RNN primitive
    // 4. Q10n of the data for reference benchdnn RNN:
    //    4.a. If the tensor is weights -- q10n it here;
    //    4.b. If the tensor is data -- reference benchdnn RNN will quantize it.

    // pass rnn attributes to f32 -> int8 reorders only
    const_dnnl_primitive_attr_t reorder_attr = nullptr;
    if (p.is_int8() && (dt != dnnl_f32)) reorder_attr = attr;
    float default_scales[1] = {1.0f};
    float default_shift = 0.0f;

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);

    // 2. We fill first f32 de-quantized data, by inverse-applying the scale
    //    and shift to the data generated by cfg distribution;
    auto fill_chunk = [&](const float *scales, int nscales, float shift,
                              int idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        std::minstd_rand msr;
        msr.seed(idx_start + kind);
        std::normal_distribution<float> gen(mean, stddev);
        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float val = round_to_nearest_representable(dt, gen(msr));
            val = MAX2(MIN2(val, max), min);
            val = (val - shift)
                    / scales[idx % nscales]; // change only int8-case

            // Vanilla RNN with RELU testing related only: flip the sign of
            // inputs for `mb` == 0 to test RELU part
            if (flip_sign) {
                assert(kind == SRC_LAYER || kind == SRC_ITER);
                auto ld = kind == SRC_LAYER ? p.slc : p.sic;
                if (idx % (p.mb * ld) < ld) val *= -1;
            }
            mem_fp.set_elem(idx, val);
        }
    };
    switch (kind) {
        case WEIGHTS_LAYER:
        case WEIGHTS_ITER:
            dnnl::impl::parallel_nd(n_chunks, [&](int idx) {
                fill_chunk(p.wei_scales, p.wei_nscales, 0.0f, idx);
            });
            break;
        case SRC_LAYER:
        case SRC_ITER:
            dnnl::impl::parallel_nd(n_chunks, [&](int idx) {
                fill_chunk(&(p.data_scale), 1, p.data_shift, idx);
            });
            break;
        default: // we do no scale/shift
            dnnl::impl::parallel_nd(n_chunks, [&](int idx) {
                fill_chunk(default_scales, 1, default_shift, idx);
            });
    }

    // 3. We reorder the data for the DNNL RNN primitive
    mem_dt.reorder(mem_fp, {reorder_attr});
    if ((reorder_attr != nullptr) && (dt == dnnl_s8))
        if (check_s8s8_reorder(p, mem_dt, mem_fp) != OK) return FAIL;

    // Bullet 4.a holds: quantize weights for int8 benchdnn reference RNN
    if (p.is_int8()) {
        auto quantize_chunk
                = [&](const float *scales, int nscales, int idx_chunk) {
                      int64_t idx_start = idx_chunk * chunk_size;
                      int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
                      for (int64_t idx = idx_start; idx < idx_end; ++idx) {
                          float current_scale = scales[idx % nscales];
                          float val = ((float *)mem_fp)[idx];
                          val = round(current_scale * val);
                          mem_fp.set_elem(idx, MAX2(MIN2(val, max), min));
                      }
                  };
        switch (kind) {
            case WEIGHTS_LAYER:
            case WEIGHTS_ITER:
                dnnl::impl::parallel_nd(n_chunks, [&](int idx) {
                    quantize_chunk(p.wei_scales, p.wei_nscales, idx);
                });
                break;
            default: // Nothing to do
                break;
        }
    }

    return OK;
}

int fill_memory(const prb_t &p, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, const_dnnl_primitive_attr_t attr = nullptr,
        bool flip_sign = false) {
    const dt_conf_t::entry_t &c = p.cfg[kind];
    return fill_memory(p, kind, mem_dt, mem_fp, c.dt, c.f_mean, c.f_stddev,
            c.f_min, c.f_max, attr, flip_sign);
}

int fill_activation(const prb_t &p, data_kind_t kind, dnn_mem_t &mem_dt,
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
            && p.activation == RELU && (kind == SRC_LAYER || kind == SRC_ITER);
    return fill_memory(p, kind, mem_dt, mem_fp, attr, flip_sign);
}

int fill_src_iter_c(const prb_t &p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        const_dnnl_primitive_attr_t attr = nullptr) {
    const bool special_case = p.prop == dnnl_backward && p.skip_nonlinear;
    if (!special_case) return fill_memory(p, SRC_ITER_C, mem_dt, mem_fp, attr);

    // The scaling factors in tparams when testing backward are common for
    // for forward and backward passes, and computed as 1 over maximum of
    // the accumulation chain:
    // - ~n_gates on FWD
    // - ~dhc * n_gates on BWD_D
    // - ~mb * n_gates on BWD_W
    //
    // This makes tparam relatively small for the forward pass (compare to
    // the forward pass when we test forward only). This in turn, makes
    // src_iter_c converge relatively fast to the value ~ i_gate * c_gate,
    // which is (typically) way smaller than the original distribution for
    // src_iter_c.
    //
    // TODO: use different tparams for forward and
    //       backward passes when testing BWD_DW.
    //
    // The problem appears on backward pass. Consider diff_f_gate that
    // contributes to backward weights when batch or number of iterations
    // is big:
    //   diff_f_gate[iter] = src_iter_c[iter] * diff_dst[iter].
    //   diff_weights += ~diff_f_gate[iter].
    //
    // Assume, that diff_dst[iter] is always about the same for every iter.
    // Since src_iter_c[0] >> src_iter_c[iter] for iter > 0, this makes the
    // diff_weight be highly dependent on the order of accumulating the
    // diff_f_gate[iter].
    //
    // Originally we had something like:
    // diff_weights = v + v * 10^-5 + ... + v * 10^-5 (n_iter * MB summands).
    // Depending on the order of summation the difference might exceed the
    // typical bound approximation: coefficient * log(number_of_summands).
    //
    // Anyways, the algorithm below tries to put the first src_iter_c[iter = 0]
    // in the same ballpark as all the subsequent src_iter_c[iter > 0].
    //
    // The estimation is based on the following rough assumptions:
    //   src_iter_c[iter+1] = f_gate * src_iter_c[iter] + i_gate * c_gate
    //                     ~= f_gate * small_value      + i_gate * c_gate
    //                     ~=                             i_gate * c_gate.
    //   i_gate ~= tparams[i_gate] * (
    //              1 / ngates * mean_src_layer +
    //              1 / ngates * mean_src_iter  +
    //              mean_bias);
    //
    // Same for c_gate.
    // The (1 / ngates) factor is taken from fill_weights().

    float expect_gemm_output = (1.f / p.n_gates()) * p.cfg[SRC_LAYER].f_mean
            + (1.f / p.n_gates()) * p.cfg[SRC_ITER].f_mean + p.cfg[BIAS].f_mean;
    float expect_i_gate = (float)p.linear_scales[LSTM_I] * expect_gemm_output;
    float expect_c_gate = (float)p.linear_scales[LSTM_C] * expect_gemm_output;
    float expect_src_iter_c_mean = expect_i_gate * expect_c_gate;

    float adjust_factor = 1;

    const bool need_adjust = expect_src_iter_c_mean < p.cfg[SRC_ITER_C].f_mean
            && p.cfg[SRC_ITER_C].f_mean != 0;
    if (need_adjust)
        adjust_factor = expect_src_iter_c_mean / p.cfg[SRC_ITER_C].f_mean;

    const dt_conf_t::entry_t &c = p.cfg[SRC_ITER_C];
    return fill_memory(p, SRC_ITER_C, mem_dt, mem_fp, c.dt,
            c.f_mean * adjust_factor, c.f_stddev * adjust_factor,
            c.f_min * adjust_factor, c.f_max * adjust_factor, attr);
}

int fill_weights(const prb_t &p, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, const_dnnl_primitive_attr_t attr = nullptr) {
    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return OK;

    const dt_conf_t::entry_t &c = p.cfg[kind];
    if (c.dt == dnnl_s8) return fill_memory(p, kind, mem_dt, mem_fp, attr);

    const auto &dims = mem_fp.md_.dims;
    const int64_t L = dims[0];
    const int64_t D = dims[1];
    const int64_t I = dims[2];
    const int64_t G = (kind == WEIGHTS_PROJECTION) ? 1 : dims[3];
    const int64_t O = (kind == WEIGHTS_PROJECTION) ? dims[3] : dims[4];

    for (int64_t i = 0; i < mem_dt.nelems(); i++)
        mem_fp.set_elem(i, 0.0f);

    float gate_factor = (kind == WEIGHTS_PROJECTION) ? 1.f : 1.f / p.n_gates();

    for_(int64_t l = 0; l < L; l++)
    for_(int64_t d = 0; d < D; d++)
    for_(int64_t g = 0; g < G; g++)
    for (int64_t o = 0; o < O; o++) {
        float val = round_to_nearest_representable(c.dt, gate_factor);
        int64_t i_off = ((19 * o + g * 7 + d * 11 + l * 13) % I);
        mem_fp.set_elem((((l * D + d) * I + i_off) * G + g) * O + o, val);
    }

    mem_dt.reorder(mem_fp);
    return OK;
}

int fill_bias(const prb_t &p, data_kind_t kind, dnn_mem_t &mem_dt,
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

static int init_pd(dnnl_engine_t engine, const prb_t *p_ptr,
        dnnl_primitive_desc_t &rpd, res_t *r, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    const auto &p = *p_ptr;
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
    // Enable testing with non trivial strides
    int the_stride = p.trivial_strides ? 0 : 1;
    /// @todo we need to add stride support for diff_* tensors too
    dnnl_memory_desc_t src_layer_d, src_iter_d, src_iter_c_d, weights_layer_d,
            weights_iter_d, weights_peephole_d {}, weights_projection_d {},
            bias_d, dst_layer_d, dst_iter_d, dst_iter_c_d, diff_src_layer_d,
            diff_src_iter_d, diff_src_iter_c_d, diff_weights_layer_d,
            diff_weights_iter_d, diff_weights_peephole_d {},
            diff_weights_projection_d {}, diff_bias_d, diff_dst_layer_d,
            diff_dst_iter_d, diff_dst_iter_c_d;

    // dimensions with ref
    dnnl_dims_t src_layer_dims = {p.n_iter, p.mb, p.slc};
    // bidirectional = 2, s for lstm = 2, for all other = 1
    dnnl_dims_t weights_layer_dims
            = {p.n_layer, p.n_dir(), p.slc, p.n_gates(), p.dhc};
    dnnl_dims_t weights_iter_dims
            = {p.n_layer, p.n_dir(), p.sic, p.n_gates(), p.dhc};
    dnnl_dims_t weights_peephole_dims = {p.n_layer, p.n_dir(), 3, p.dhc};
    dnnl_dims_t weights_projection_dims = {p.n_layer, p.n_dir(), p.dhc, p.dic};
    dnnl_dims_t bias_dims
            = {p.n_layer, p.n_dir(), p.n_gates() + is_gru_lbr, p.dhc};
    // dnnl_tnc
    dnnl_dims_t dst_layer_dims = {p.n_iter, p.mb, p.dlc(PRIMITIVE)};

    DNN_SAFE(dnnl_memory_desc_init_by_tag(&src_layer_d, 3, src_layer_dims,
                     p.cfg[SRC_LAYER].dt, dnnl_tnc),
            WARN);
    src_layer_d.format_desc.blocking.strides[0] += the_stride;

    dnnl_dims_t src_iter_dims = {p.n_layer, p.n_dir(), p.mb, p.sic};
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&src_iter_d, 4, src_iter_dims,
                     p.cfg[SRC_ITER].dt, dnnl_ldnc),
            WARN);
    src_iter_d.format_desc.blocking.strides[2] = p.sic + the_stride;
    for (int d = 1; d >= 0; --d)
        src_iter_d.format_desc.blocking.strides[d]
                = src_iter_d.format_desc.blocking.strides[d + 1]
                * src_iter_d.dims[d + 1];

    dnnl_dims_t src_iter_c_dims = {p.n_layer, p.n_dir(), p.mb, p.dhc};
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&src_iter_c_d, 4, src_iter_c_dims,
                     p.cfg[SRC_ITER_C].dt, dnnl_ldnc),
            WARN);
    src_iter_c_d.format_desc.blocking.strides[2] = p.dhc + the_stride;
    for (int d = 1; d >= 0; --d)
        src_iter_c_d.format_desc.blocking.strides[d]
                = src_iter_c_d.format_desc.blocking.strides[d + 1]
                * src_iter_c_d.dims[d + 1];

    DNN_SAFE(dnnl_memory_desc_init_by_tag(&weights_layer_d, 5,
                     weights_layer_dims, p.cfg[WEIGHTS_LAYER].dt,
                     dnnl_format_tag_any),
            WARN);

    DNN_SAFE(dnnl_memory_desc_init_by_tag(&weights_iter_d, 5, weights_iter_dims,
                     p.cfg[WEIGHTS_ITER].dt, dnnl_format_tag_any),
            WARN);

    if (p.is_lstm_peephole()) {
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&weights_peephole_d, 4,
                         weights_peephole_dims, p.cfg[WEIGHTS_PEEPHOLE].dt,
                         dnnl_ldgo),
                WARN);
    }

    if (p.is_lstm_projection()) {
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&weights_projection_d, 4,
                         weights_projection_dims, p.cfg[WEIGHTS_PROJECTION].dt,
                         dnnl_format_tag_any),
                WARN);
    }

    DNN_SAFE(dnnl_memory_desc_init_by_tag(&bias_d, 4, bias_dims, p.cfg[BIAS].dt,
                     dnnl_format_tag_any),
            WARN);

    DNN_SAFE(dnnl_memory_desc_init_by_tag(&dst_layer_d, 3, dst_layer_dims,
                     p.cfg[DST_LAYER].dt, dnnl_tnc),
            WARN);
    dst_layer_d.format_desc.blocking.strides[0] += the_stride;

    dnnl_dims_t dst_iter_dims = {p.n_layer, p.n_dir(), p.mb, p.dic};
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&dst_iter_d, 4, dst_iter_dims,
                     p.cfg[DST_ITER].dt, dnnl_ldnc),
            WARN);
    dst_iter_d.format_desc.blocking.strides[2] = p.dic + the_stride;
    for (int d = 1; d >= 0; --d)
        dst_iter_d.format_desc.blocking.strides[d]
                = dst_iter_d.format_desc.blocking.strides[d + 1]
                * dst_iter_d.dims[d + 1];

    dnnl_dims_t dst_iter_c_dims = {p.n_layer, p.n_dir(), p.mb, p.dhc};
    DNN_SAFE(dnnl_memory_desc_init_by_tag(&dst_iter_c_d, 4, dst_iter_c_dims,
                     p.cfg[DST_ITER_C].dt, dnnl_ldnc),
            WARN);

    dst_iter_c_d.format_desc.blocking.strides[2] = p.dhc + the_stride;
    for (int d = 1; d >= 0; --d)
        dst_iter_c_d.format_desc.blocking.strides[d]
                = dst_iter_c_d.format_desc.blocking.strides[d + 1]
                * dst_iter_c_d.dims[d + 1];

    // Initializing the forward pass
    // When inference, we use forward_inference
    // When training, we use forward_training
    if (dir & FLAG_FWD) {
        DNN_SAFE(init_rnn_fwd_desc(&rd, p, fwd_prop, &src_layer_d, &src_iter_d,
                         &src_iter_c_d, &weights_layer_d, &weights_iter_d,
                         &weights_peephole_d, &weights_projection_d, &bias_d,
                         &dst_layer_d, &dst_iter_d, &dst_iter_c_d),
                WARN);
    } else {
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_src_layer_d, 3,
                         src_layer_dims, p.cfg[DIFF_SRC_LAYER].dt,
                         dnnl_format_tag_any),
                WARN);
        DNN_SAFE(
                dnnl_memory_desc_init_by_tag(&diff_src_iter_d, 4, src_iter_dims,
                        p.cfg[DIFF_SRC_ITER].dt, dnnl_format_tag_any),
                WARN);
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_src_iter_c_d, 4,
                         src_iter_c_dims, p.cfg[DIFF_SRC_ITER_C].dt,
                         dnnl_format_tag_any),
                WARN);
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_weights_layer_d, 5,
                         weights_layer_dims, p.cfg[DIFF_WEIGHTS_LAYER].dt,
                         dnnl_format_tag_any),
                WARN);
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_weights_iter_d, 5,
                         weights_iter_dims, p.cfg[DIFF_WEIGHTS_ITER].dt,
                         dnnl_format_tag_any),
                WARN);
        if (p.is_lstm_peephole()) {
            DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_weights_peephole_d, 4,
                             weights_peephole_dims,
                             p.cfg[DIFF_WEIGHTS_PEEPHOLE].dt, dnnl_ldgo),
                    WARN);
        }
        if (p.is_lstm_projection()) {
            DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_weights_projection_d, 4,
                             weights_projection_dims,
                             p.cfg[DIFF_WEIGHTS_PROJECTION].dt,
                             dnnl_format_tag_any),
                    WARN);
        }
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_bias_d, 4, bias_dims,
                         p.cfg[DIFF_BIAS].dt, dnnl_format_tag_any),
                WARN);
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_dst_layer_d, 3,
                         dst_layer_dims, p.cfg[DIFF_DST_LAYER].dt,
                         dnnl_format_tag_any),
                WARN);
        DNN_SAFE(
                dnnl_memory_desc_init_by_tag(&diff_dst_iter_d, 4, dst_iter_dims,
                        p.cfg[DIFF_DST_ITER].dt, dnnl_format_tag_any),
                WARN);
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_dst_iter_c_d, 4,
                         dst_iter_c_dims, p.cfg[DIFF_DST_ITER_C].dt,
                         dnnl_format_tag_any),
                WARN);
        DNN_SAFE(
                init_rnn_bwd_desc(&rd, p, p.prop, &src_layer_d, &src_iter_d,
                        &src_iter_c_d, &weights_layer_d, &weights_iter_d,
                        &weights_peephole_d, &weights_projection_d, &bias_d,
                        &dst_layer_d, &dst_iter_d, &dst_iter_c_d,
                        &diff_src_layer_d, &diff_src_iter_d, &diff_src_iter_c_d,
                        &diff_weights_layer_d, &diff_weights_iter_d,
                        &diff_weights_peephole_d, &diff_weights_projection_d,
                        &diff_bias_d, &diff_dst_layer_d, &diff_dst_iter_d,
                        &diff_dst_iter_c_d),
                WARN);
    }

    dnnl_primitive_attr_t dnnl_attr;
    create_dnnl_rnn_attr(p, &dnnl_attr);
    dnnl_status_t init_status
            = dnnl_primitive_desc_create(&rpd, &rd, dnnl_attr, engine, NULL);
    dnnl_primitive_attr_destroy(dnnl_attr);
    if (init_status == dnnl_unimplemented) return r->state = UNIMPLEMENTED, OK;
    SAFE(init_status, WARN);

    // Return if pd is not the one being tested
    if ((dir & FLAG_FWD) && (p.prop == dnnl_backward)) return OK;

    r->impl_name = query_impl_info(rpd);
    BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", r->impl_name.c_str());

    return OK;
}

void check_known_skipped_case(const prb_t &p, res_t *r) {
    check_known_skipped_case_common({p.cfg[SRC_LAYER].dt}, r);
    if (r->state == SKIPPED) return;

    if (p.maybe_skip()) {
        r->state = SKIPPED, r->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

int doit(const prb_t &p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;

    check_known_skipped_case(p, r);
    if (r->state == SKIPPED) return OK;

    dnnl_primitive_t c {};
    SAFE(init_prim(&c, init_pd, &p, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;
    auto cleanup = [&]() {
        DNN_SAFE(dnnl_primitive_destroy(c), CRIT);
        return OK;
    };

    const_dnnl_primitive_desc_t const_fpd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(c, &const_fpd), CRIT);

    if (dnn_mem_t::check_mem_size(const_fpd) != OK) {
        DNN_SAFE(dnnl_primitive_destroy(c), CRIT);
        return r->state = SKIPPED, r->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto q = [](const_dnnl_primitive_desc_t pd,
                           int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(pd, dnnl_query_exec_arg_md, index);
    };

    const auto &src_layer_md = q(const_fpd, DNNL_ARG_SRC_LAYER);
    const auto &src_iter_md = q(const_fpd, DNNL_ARG_SRC_ITER);
    const auto &src_iter_c_md = q(const_fpd, DNNL_ARG_SRC_ITER_C);
    const auto &weights_layer_md = q(const_fpd, DNNL_ARG_WEIGHTS_LAYER);
    const auto &weights_iter_md = q(const_fpd, DNNL_ARG_WEIGHTS_ITER);
    const auto &weights_peephole_md = q(const_fpd, DNNL_ARG_WEIGHTS_PEEPHOLE);
    const auto &weights_projection_md
            = q(const_fpd, DNNL_ARG_WEIGHTS_PROJECTION);
    const auto &bias_md = q(const_fpd, DNNL_ARG_BIAS);
    const auto &dst_layer_md = q(const_fpd, DNNL_ARG_DST_LAYER);
    const auto &dst_iter_md = q(const_fpd, DNNL_ARG_DST_ITER);
    const auto &dst_iter_c_md = q(const_fpd, DNNL_ARG_DST_ITER_C);
    const auto &workspace_md = q(const_fpd, DNNL_ARG_WORKSPACE);
    const auto &scratchpad_md = q(const_fpd, DNNL_ARG_SCRATCHPAD);

    const auto &test_engine = get_test_engine();

    dnn_mem_t src_layer_dt(src_layer_md, test_engine);
    dnn_mem_t src_iter_dt(src_iter_md, test_engine);
    dnn_mem_t src_iter_c_dt(src_iter_c_md, test_engine);
    dnn_mem_t weights_layer_dt(weights_layer_md, test_engine);
    dnn_mem_t weights_iter_dt(weights_iter_md, test_engine);
    dnn_mem_t weights_peephole_dt(weights_peephole_md, test_engine);
    dnn_mem_t weights_projection_dt(weights_projection_md, test_engine);
    dnn_mem_t bias_dt(bias_md, test_engine);
    dnn_mem_t dst_layer_dt(dst_layer_md, test_engine);
    dnn_mem_t dst_iter_dt(dst_iter_md, test_engine);
    dnn_mem_t dst_iter_c_dt(dst_iter_c_md, test_engine);
    dnn_mem_t workspace_dt(workspace_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    const auto fp = dnnl_f32;
    dnn_mem_t src_layer_fp(src_layer_md, fp, dnnl_tnc, test_engine);
    dnn_mem_t src_iter_fp(src_iter_md, fp, dnnl_ldnc, test_engine);
    dnn_mem_t src_iter_c_fp(src_iter_c_md, fp, dnnl_ldnc, test_engine);
    dnn_mem_t weights_layer_fp(weights_layer_md, fp, dnnl_ldigo, test_engine);
    dnn_mem_t weights_iter_fp(weights_iter_md, fp, dnnl_ldigo, test_engine);
    dnn_mem_t weights_peephole_fp(
            weights_peephole_md, fp, dnnl_ldgo, test_engine);
    dnn_mem_t weights_projection_fp(
            weights_projection_md, fp, dnnl_ldio, test_engine);
    dnn_mem_t bias_fp(bias_md, fp, dnnl_ldgo, test_engine);
    dnn_mem_t dst_layer_fp(dst_layer_md, fp, dnnl_tnc, test_engine);
    dnn_mem_t dst_iter_fp(dst_iter_md, fp, dnnl_ldnc, test_engine);
    dnn_mem_t dst_iter_c_fp(dst_iter_c_md, fp, dnnl_ldnc, test_engine);

    dnn_mem_t bwd_weights_layer_dt;
    dnn_mem_t bwd_weights_iter_dt;
    dnn_mem_t bwd_weights_projection_dt;
    dnn_mem_t diff_src_layer_dt;
    dnn_mem_t diff_src_iter_dt;
    dnn_mem_t diff_src_iter_c_dt;
    dnn_mem_t diff_weights_layer_dt;
    dnn_mem_t diff_weights_iter_dt;
    dnn_mem_t diff_weights_peephole_dt;
    dnn_mem_t diff_weights_projection_dt;
    dnn_mem_t diff_bias_dt;
    dnn_mem_t diff_dst_layer_dt;
    dnn_mem_t diff_dst_iter_dt;
    dnn_mem_t diff_dst_iter_c_dt;

    // for int8 RNN we need pass attributes for data q10n
    const_dnnl_primitive_attr_t rnn_attr;
    DNN_SAFE(dnnl_primitive_desc_get_attr(const_fpd, &rnn_attr), WARN);

    SAFE(fill_activation(p, SRC_LAYER, src_layer_dt, src_layer_fp, rnn_attr),
            WARN);
    SAFE(fill_activation(p, SRC_ITER, src_iter_dt, src_iter_fp, rnn_attr),
            WARN);
    if (p.alg == VANILLA_LSTM)
        SAFE(fill_src_iter_c(p, src_iter_c_dt, src_iter_c_fp, rnn_attr), WARN);
    SAFE(fill_weights(p, WEIGHTS_LAYER, weights_layer_dt, weights_layer_fp,
                 rnn_attr),
            WARN);
    SAFE(fill_weights(
                 p, WEIGHTS_ITER, weights_iter_dt, weights_iter_fp, rnn_attr),
            WARN);
    SAFE(fill_memory(
                 p, WEIGHTS_PEEPHOLE, weights_peephole_dt, weights_peephole_fp),
            WARN);
    SAFE(fill_weights(p, WEIGHTS_PROJECTION, weights_projection_dt,
                 weights_projection_fp),
            WARN);
    SAFE(fill_memory(p, BIAS, bias_dt, bias_fp), WARN);
    SAFE(fill_activation(p, DST_LAYER, dst_layer_dt, dst_layer_fp), WARN);
    SAFE(fill_activation(p, DST_ITER, dst_iter_dt, dst_iter_fp), WARN);
    if (p.alg == VANILLA_LSTM)
        SAFE(fill_memory(p, DST_ITER_C, dst_iter_c_dt, dst_iter_c_fp), WARN);

    args_t args;

    // Running the forward pass
    args.set(DNNL_ARG_SRC_LAYER, src_layer_dt);
    args.set(DNNL_ARG_SRC_ITER, src_iter_dt);
    args.set(DNNL_ARG_SRC_ITER_C, src_iter_c_dt);
    args.set(DNNL_ARG_WEIGHTS_LAYER, weights_layer_dt);
    args.set(DNNL_ARG_WEIGHTS_ITER, weights_iter_dt);
    args.set(DNNL_ARG_WEIGHTS_PEEPHOLE, weights_peephole_dt);
    args.set(DNNL_ARG_WEIGHTS_PROJECTION, weights_projection_dt);
    args.set(DNNL_ARG_BIAS, bias_dt);
    args.set(DNNL_ARG_DST_LAYER, dst_layer_dt);
    args.set(DNNL_ARG_DST_ITER, dst_iter_dt);
    args.set(DNNL_ARG_DST_ITER_C, dst_iter_c_dt);
    args.set(DNNL_ARG_WORKSPACE, workspace_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

    SAFE_CLEAN(execute_and_wait(c, args), WARN, cleanup);

    if ((p.prop == dnnl_forward) && (bench_mode & CORR)) {
        compute_ref_fwd(p, src_layer_fp, src_iter_fp, src_iter_c_fp,
                weights_layer_fp, weights_iter_fp, weights_peephole_fp,
                weights_projection_fp, bias_fp, dst_layer_fp, dst_iter_fp,
                dst_iter_c_fp);

        int compare_status = OK;
        COMPARE_DAT(compare_status, DST_LAYER, dst_layer, dnnl_tnc);
        COMPARE_DAT(compare_status, DST_ITER, dst_iter, dnnl_ldnc);
        if (p.alg == VANILLA_LSTM)
            COMPARE_DAT(compare_status, DST_ITER_C, dst_iter_c, dnnl_ldnc);
        SAFE_CLEAN(compare_status, WARN, cleanup);
    }

    if (p.prop == dnnl_backward) {
        dnnl_primitive_t bwd_p {};
        int status = init_prim(&bwd_p, init_pd, &p, r, FLAG_BWD);
        dnnl_primitive_destroy(c);
        if (status != OK) return status;
        if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;
        c = bwd_p;

        const_dnnl_primitive_desc_t const_bpd;
        DNN_SAFE(dnnl_primitive_get_primitive_desc(c, &const_bpd), CRIT);

        if (dnn_mem_t::check_mem_size(const_bpd) != OK) {
            DNN_SAFE(dnnl_primitive_destroy(c), CRIT);
            return r->state = SKIPPED, r->reason = NOT_ENOUGH_RAM, OK;
        }

        const auto &bwd_weights_layer_md = q(const_bpd, DNNL_ARG_WEIGHTS_LAYER);
        const auto &bwd_weights_iter_md = q(const_bpd, DNNL_ARG_WEIGHTS_ITER);
        const auto &bwd_weights_projection_md
                = q(const_bpd, DNNL_ARG_WEIGHTS_PROJECTION);
        const auto &diff_src_layer_md = q(const_bpd, DNNL_ARG_DIFF_SRC_LAYER);
        const auto &diff_src_iter_md = q(const_bpd, DNNL_ARG_DIFF_SRC_ITER);
        const auto &diff_src_iter_c_md = q(const_bpd, DNNL_ARG_DIFF_SRC_ITER_C);
        const auto &diff_weights_layer_md
                = q(const_bpd, DNNL_ARG_DIFF_WEIGHTS_LAYER);
        const auto &diff_weights_iter_md
                = q(const_bpd, DNNL_ARG_DIFF_WEIGHTS_ITER);
        const auto &diff_weights_peephole_md
                = q(const_bpd, DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE);
        const auto &diff_weights_projection_md
                = q(const_bpd, DNNL_ARG_DIFF_WEIGHTS_PROJECTION);
        const auto &diff_bias_md = q(const_bpd, DNNL_ARG_DIFF_BIAS);
        const auto &diff_dst_layer_md = q(const_bpd, DNNL_ARG_DIFF_DST_LAYER);
        const auto &diff_dst_iter_md = q(const_bpd, DNNL_ARG_DIFF_DST_ITER);
        const auto &diff_dst_iter_c_md = q(const_bpd, DNNL_ARG_DIFF_DST_ITER_C);
        const auto &bwd_scratchpad_md = q(const_bpd, DNNL_ARG_SCRATCHPAD);

        bwd_weights_layer_dt = dnn_mem_t(bwd_weights_layer_md, test_engine);
        bwd_weights_iter_dt = dnn_mem_t(bwd_weights_iter_md, test_engine);
        bwd_weights_projection_dt
                = dnn_mem_t(bwd_weights_projection_md, test_engine);
        diff_src_layer_dt = dnn_mem_t(diff_src_layer_md, test_engine);
        diff_src_iter_dt = dnn_mem_t(diff_src_iter_md, test_engine);
        diff_src_iter_c_dt = dnn_mem_t(diff_src_iter_c_md, test_engine);
        diff_weights_layer_dt = dnn_mem_t(diff_weights_layer_md, test_engine);
        diff_weights_iter_dt = dnn_mem_t(diff_weights_iter_md, test_engine);
        diff_weights_peephole_dt
                = dnn_mem_t(diff_weights_peephole_md, test_engine);
        diff_weights_projection_dt
                = dnn_mem_t(diff_weights_projection_md, test_engine);
        diff_bias_dt = dnn_mem_t(diff_bias_md, test_engine);
        diff_dst_layer_dt = dnn_mem_t(diff_dst_layer_md, test_engine);
        diff_dst_iter_dt = dnn_mem_t(diff_dst_iter_md, test_engine);
        diff_dst_iter_c_dt = dnn_mem_t(diff_dst_iter_c_md, test_engine);
        scratchpad_dt = dnn_mem_t(bwd_scratchpad_md, test_engine);

        dnn_mem_t diff_src_layer_fp(
                diff_src_layer_md, fp, dnnl_tnc, test_engine);
        dnn_mem_t diff_src_iter_fp(
                diff_src_iter_md, fp, dnnl_ldnc, test_engine);
        dnn_mem_t diff_src_iter_c_fp(
                diff_src_iter_c_md, fp, dnnl_ldnc, test_engine);
        dnn_mem_t diff_weights_layer_fp(
                diff_weights_layer_md, fp, dnnl_ldigo, test_engine);
        dnn_mem_t diff_weights_iter_fp(
                diff_weights_iter_md, fp, dnnl_ldigo, test_engine);
        dnn_mem_t diff_weights_peephole_fp(
                diff_weights_peephole_md, fp, dnnl_ldgo, test_engine);
        dnn_mem_t diff_weights_projection_fp(
                diff_weights_projection_md, fp, dnnl_ldio, test_engine);
        dnn_mem_t diff_bias_fp(diff_bias_md, fp, dnnl_ldgo, test_engine);
        dnn_mem_t diff_dst_layer_fp(
                diff_dst_layer_md, fp, dnnl_tnc, test_engine);
        dnn_mem_t diff_dst_iter_fp(
                diff_dst_iter_md, fp, dnnl_ldnc, test_engine);
        dnn_mem_t diff_dst_iter_c_fp(
                diff_dst_iter_c_md, fp, dnnl_ldnc, test_engine);

        SAFE(bwd_weights_iter_dt.reorder(weights_iter_dt), WARN);
        SAFE(bwd_weights_layer_dt.reorder(weights_layer_dt), WARN);
        if (p.is_lstm_projection())
            SAFE(bwd_weights_projection_dt.reorder(weights_projection_dt),
                    WARN);
        SAFE(fill_activation(
                     p, DIFF_SRC_LAYER, diff_src_layer_dt, diff_src_layer_fp),
                WARN);
        SAFE(fill_activation(
                     p, DIFF_SRC_ITER, diff_src_iter_dt, diff_src_iter_fp),
                WARN);
        if (p.alg == VANILLA_LSTM)
            SAFE(fill_memory(p, DIFF_SRC_ITER_C, diff_src_iter_c_dt,
                         diff_src_iter_c_fp),
                    WARN);
        SAFE(fill_weights(p, DIFF_WEIGHTS_LAYER, diff_weights_layer_dt,
                     diff_weights_layer_fp),
                WARN);
        SAFE(fill_weights(p, DIFF_WEIGHTS_ITER, diff_weights_iter_dt,
                     diff_weights_iter_fp),
                WARN);
        SAFE(fill_memory(p, DIFF_WEIGHTS_PEEPHOLE, diff_weights_peephole_dt,
                     diff_weights_peephole_fp),
                WARN);
        SAFE(fill_memory(p, DIFF_WEIGHTS_PROJECTION, diff_weights_projection_dt,
                     diff_weights_projection_fp),
                WARN);
        SAFE(fill_bias(p, DIFF_BIAS, diff_bias_dt, diff_bias_fp), WARN);
        SAFE(fill_activation(
                     p, DIFF_DST_LAYER, diff_dst_layer_dt, diff_dst_layer_fp),
                WARN);
        SAFE(fill_activation(
                     p, DIFF_DST_ITER, diff_dst_iter_dt, diff_dst_iter_fp),
                WARN);
        if (p.alg == VANILLA_LSTM)
            SAFE(fill_memory(p, DIFF_DST_ITER_C, diff_dst_iter_c_dt,
                         diff_dst_iter_c_fp),
                    WARN);

        args.clear();
        args.set(DNNL_ARG_SRC_LAYER, src_layer_dt);
        args.set(DNNL_ARG_SRC_ITER, src_iter_dt);
        args.set(DNNL_ARG_SRC_ITER_C, src_iter_c_dt);
        args.set(DNNL_ARG_WEIGHTS_LAYER, bwd_weights_layer_dt);
        args.set(DNNL_ARG_WEIGHTS_ITER, bwd_weights_iter_dt);
        args.set(DNNL_ARG_WEIGHTS_PEEPHOLE, weights_peephole_dt);
        args.set(DNNL_ARG_WEIGHTS_PROJECTION, bwd_weights_projection_dt);
        args.set(DNNL_ARG_BIAS, bias_dt);
        args.set(DNNL_ARG_DST_LAYER, dst_layer_dt);
        args.set(DNNL_ARG_DST_ITER, dst_iter_dt);
        args.set(DNNL_ARG_DST_ITER_C, dst_iter_c_dt);
        args.set(DNNL_ARG_DIFF_DST_LAYER, diff_dst_layer_dt);
        args.set(DNNL_ARG_DIFF_DST_ITER, diff_dst_iter_dt);
        args.set(DNNL_ARG_DIFF_DST_ITER_C, diff_dst_iter_c_dt);
        args.set(DNNL_ARG_WORKSPACE, workspace_dt);
        args.set(DNNL_ARG_DIFF_SRC_LAYER, diff_src_layer_dt);
        args.set(DNNL_ARG_DIFF_SRC_ITER, diff_src_iter_dt);
        args.set(DNNL_ARG_DIFF_SRC_ITER_C, diff_src_iter_c_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS_LAYER, diff_weights_layer_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS_ITER, diff_weights_iter_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE, diff_weights_peephole_dt);
        args.set(DNNL_ARG_DIFF_WEIGHTS_PROJECTION, diff_weights_projection_dt);
        args.set(DNNL_ARG_DIFF_BIAS, diff_bias_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        SAFE_CLEAN(execute_and_wait(c, args), WARN, cleanup);

        if (bench_mode & CORR) {
            compute_ref_bwd(p, src_layer_fp, src_iter_fp, src_iter_c_fp,
                    diff_dst_layer_fp, diff_dst_iter_fp, diff_dst_iter_c_fp,
                    weights_layer_fp, weights_iter_fp, weights_peephole_fp,
                    weights_projection_fp, bias_fp, dst_layer_fp, dst_iter_fp,
                    dst_iter_c_fp, diff_src_layer_fp, diff_src_iter_fp,
                    diff_src_iter_c_fp, diff_weights_layer_fp,
                    diff_weights_iter_fp, diff_weights_peephole_fp,
                    diff_weights_projection_fp, diff_bias_fp);

            int compare_fwd_status = OK;
            COMPARE_DAT(compare_fwd_status, DST_LAYER, dst_layer, dnnl_tnc);
            COMPARE_DAT(compare_fwd_status, DST_ITER, dst_iter, dnnl_ldnc);
            if (p.alg == VANILLA_LSTM)
                COMPARE_DAT(
                        compare_fwd_status, DST_ITER_C, dst_iter_c, dnnl_ldnc);
            SAFE_CLEAN(compare_fwd_status, WARN, cleanup);

            int compare_bwd_data_status = OK;
            COMPARE_DAT(compare_bwd_data_status, DIFF_SRC_LAYER, diff_src_layer,
                    dnnl_tnc);
            COMPARE_DAT(compare_bwd_data_status, DIFF_SRC_ITER, diff_src_iter,
                    dnnl_ldnc);
            if (p.alg == VANILLA_LSTM)
                COMPARE_DAT(compare_bwd_data_status, DIFF_SRC_ITER_C,
                        diff_src_iter_c, dnnl_ldnc);
            SAFE_CLEAN(compare_bwd_data_status, WARN, cleanup);

            int compare_bwd_weights_status = OK;
            COMPARE_DAT(compare_bwd_weights_status, DIFF_WEIGHTS_LAYER,
                    diff_weights_layer, dnnl_ldigo);
            COMPARE_DAT(compare_bwd_weights_status, DIFF_WEIGHTS_ITER,
                    diff_weights_iter, dnnl_ldigo);
            if (p.is_lstm_peephole())
                COMPARE_DAT(compare_bwd_weights_status, DIFF_WEIGHTS_PEEPHOLE,
                        diff_weights_peephole, dnnl_ldgo);
            if (p.is_lstm_projection())
                COMPARE_DAT(compare_bwd_weights_status, DIFF_WEIGHTS_PROJECTION,
                        diff_weights_projection, dnnl_ldio);
            COMPARE_DAT(compare_bwd_weights_status, DIFF_BIAS, diff_bias,
                    dnnl_ldgo);
            SAFE_CLEAN(compare_bwd_weights_status, WARN, cleanup);
        }
    }

    measure_perf(r->timer, c, args);
    cleanup();

    return OK;
}
} // namespace rnn
