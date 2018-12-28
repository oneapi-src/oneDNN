/*******************************************************************************
* Copyright 2018 Intel Corporation
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

/*
  General architecture

  for diff states, we have n_states + 1 as we have n_states diff
  to propagate to the previous iteration and 1 states to propagate
  to the previous layer
  index 0 is dh for cell(t-1, l) to consume
  index 1 is dc for cell(t-1, l) to consume
  index 2 is dh for cell(t, l-1) to consume
  this indexing enables to have the same indexing for states in elemwise
  function
  only the cell execution function should be impacted

 */

#include "math_utils.hpp"
#include "mkldnn_thread.hpp"

#include "ref_rnn.hpp"
#include "../gemm/gemm.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::memory_tracking::names;
using namespace rnn_utils;
#define AOC array_offset_calculator

template <prop_kind_t aprop>
void _ref_rnn_common_t<aprop>::gates_reduction(const rnn_conf_t &rnn,
        const float *ws_gates_, float *diff_bias_) const {
    auto body = [&](int i, int k) {
        for (int j = 0; j < rnn.mb; j++)
            diff_bias_[i * rnn.dic + k]
                    += ws_gates_[j * rnn.gates_ws_ld + i * rnn.dic + k];
    };

    // @todo block k on simd-width
#if MKLDNN_THR == MKLDNN_THR_OMP && _OPENMP >= 201307 \
    /* icc 17.0 has a problem with simd collapse */ \
    && !((defined __INTEL_COMPILER) && (__INTEL_COMPILER == 1700))
#pragma omp parallel for simd collapse(2)
    for (int i = 0; i < rnn.n_gates; i++)
        for (int k = 0; k < rnn.dic; k++)
            body(i, k);
#else
    parallel_nd(rnn.n_gates, rnn.dic, body);
#endif
}

template <prop_kind_t aprop>
gemm_sig(_ref_rnn_common_t<aprop>::gemm) {
    extended_sgemm(&transA, &transB, &m, &n, &k, &alpha, a_, &ldA, b_, &ldB,
            &beta, c_, &ldC, nullptr, pd()->rnn_.use_jit_gemm);
}

template <prop_kind_t aprop>
gemm_sig(_ref_rnn_common_t<aprop>::packed_gemm) {
#if (USE_MKL_PACKED_GEMM)
    assert(transA == 'N');
    cblas_sgemm_compute(CblasColMajor, CblasPacked,
            (transB == 'T') ? CblasTrans : CblasNoTrans, m, n, k, a_, ldA, b_,
            ldB, beta, c_, ldC);
#else
    UNUSED(transA);
    UNUSED(transB);
    UNUSED(m);
    UNUSED(n);
    UNUSED(k);
    UNUSED(alpha);
    UNUSED(ldA);
    UNUSED(b_);
    UNUSED(ldB);
    UNUSED(beta);
    UNUSED(c_);
    UNUSED(ldC);
    assert(!"packed gemm is disabled");
#endif
}

//*************** Grid computations strategy: linear ***************//
template <prop_kind_t aprop>
grid_execution_sig(_ref_rnn_common_t<aprop>::linear_execution) {
    AOC<float, 5> ws_states(ws_states_, rnn.n_layer + 1, rnn.n_dir,
            rnn.n_states, rnn.n_iter + 1, rnn.states_nld * rnn.states_ws_ld);
    AOC<float, 5> ws_diff_states(ws_diff_states_, rnn.n_layer + 1, rnn.n_dir,
            (rnn.n_states + 1), rnn.n_iter + 1,
            rnn.states_nld * rnn.states_ws_ld);
    AOC<float, 4> ws_gates(ws_gates_, rnn.n_layer, rnn.n_dir, rnn.n_iter,
            rnn.gates_nld * rnn.gates_ws_ld);
    AOC<float *, 3> weights_input(
            weights_layer_, rnn.n_layer, rnn.n_dir, rnn.n_parts_weights_layer);
    AOC<float *, 3> weights_states(
            weights_states_, rnn.n_layer, rnn.n_dir, rnn.n_parts_weights_iter);
    AOC<float*, 3> bias(
        bias_, rnn.n_layer, rnn.n_dir, rnn.n_parts_bias);
    AOC<float, 3> diff_weights_layer(diff_weights_layer_, rnn.n_layer,
            rnn.n_dir,
            rnn.diff_weights_layer_nld * rnn.diff_weights_layer_ws_ld);
    AOC<float, 3> diff_weights_iter(diff_weights_iter_, rnn.n_layer, rnn.n_dir,
            rnn.diff_weights_iter_nld * rnn.diff_weights_iter_ws_ld);
    AOC<float, 3> diff_bias(
            diff_bias_, rnn.n_layer, rnn.n_dir, rnn.n_bias * rnn.dic);
    AOC<float, 4> ws_grid(
            ws_grid_, rnn.n_layer, rnn.n_dir, rnn.n_iter, (int)rnn.ws_per_cell);

    // We run the grid of computation
    for (int dir = 0; dir < rnn.n_dir; dir++) {
        for (int j = 0; j < rnn.n_layer; j++) {
            int lay = (aprop == prop_kind::forward) ? j : rnn.n_layer - j - 1;
            if ((aprop == prop_kind::forward) && rnn.merge_gemm_layer) {
                (this->*gemm_layer_func)('N', 'N', rnn.n_gates * rnn.dic,
                        rnn.mb * rnn.n_iter, rnn.slc, 1.0,
                        weights_input(lay, dir, 0), rnn.weights_iter_ws_ld,
                        &(ws_states(lay, dir, 0, 1, 0)), rnn.states_ws_ld, 0.0,
                        &(ws_gates(lay, dir, 0, 0)), rnn.gates_ws_ld);
            }
            for (int i = 0; i < rnn.n_iter; i++) {
                int iter = (aprop == prop_kind::forward) ? i : rnn.n_iter - i - 1;
                (this->*cell_func)(rnn,
                        &(ws_states(lay + 1, dir, 0, iter + 1, 0)),
                        &(ws_diff_states(lay, dir, 0, iter, 0)),
                        &(weights_input(lay, dir, 0)),
                        &(weights_states(lay, dir, 0)),
                        &(bias(lay, dir, 0)),
                        &(ws_states(lay, dir, 0, iter + 1, 0)),
                        &(ws_states(lay + 1, dir, 0, iter, 0)),
                        &(ws_diff_states(lay + 1, dir, 0, iter, 0)),
                        &(ws_diff_states(lay, dir, 0, iter + 1, 0)),
                        &(diff_weights_layer(lay, dir, 0)),
                        &(diff_weights_iter(lay, dir, 0)),
                        &(diff_bias(lay, dir, 0)),
                        &(ws_gates(lay, dir, iter, 0)),
                        &(ws_grid(lay, dir, iter, 0)),
                        ws_cell_);
            }
            if ((aprop == prop_kind::backward) && rnn.merge_gemm_layer) {
                (this->*gemm_layer_func)('N', 'N', rnn.slc, rnn.mb * rnn.n_iter,
                        rnn.n_gates * rnn.dic, 1.0, weights_input(lay, dir, 0),
                        rnn.weights_layer_ws_ld, &(ws_gates(lay, dir, 0, 0)),
                        rnn.gates_ws_ld, 0.0,
                        &(ws_diff_states(lay, dir, rnn.n_states, 0, 0)),
                        rnn.states_ws_ld);
                gemm('N', 'T', rnn.n_gates * rnn.dic, rnn.slc,
                        rnn.mb * rnn.n_iter, 1.0, &(ws_gates(lay, dir, 0, 0)),
                        rnn.gates_ws_ld, &(ws_states(lay, dir, 0, 1, 0)),
                        rnn.states_ws_ld, 1.0,
                        &(diff_weights_layer(lay, dir, 0)),
                        rnn.diff_weights_layer_ws_ld);
            }
            if ((aprop == prop_kind::backward) && rnn.merge_gemm_iter) {
                gemm('N', 'T', rnn.n_gates * rnn.dic, rnn.sic,
                        rnn.mb * rnn.n_iter, 1.0, &(ws_gates(lay, dir, 0, 0)),
                        rnn.gates_ws_ld, &(ws_states(lay + 1, dir, 0, 0, 0)),
                        rnn.states_ws_ld, 1.0,
                        &(diff_weights_iter(lay, dir, 0)), rnn.diff_weights_iter_ws_ld);
            }
        }
    }
}

//********* GRID computations strategy: utility functions **********//

template <>
void _ref_rnn_common_t<prop_kind::forward>::copy_init_layer(
        const rnn_conf_t &rnn, float * __restrict ws_states_,
        float * __restrict ws_diff_states_, const float * __restrict xt_,
        const float  * __restrict diff_dst_layer_) const {

    AOC<float, 5> ws_states(ws_states_, rnn.n_dir, rnn.n_states, rnn.n_iter + 1,
            rnn.mb, rnn.states_ws_ld);
    auto xt_d = memory_desc_wrapper(pd()->src_pd(0));

    parallel_nd(rnn.n_iter, rnn.mb, [&](int it, int b) {
        auto xxt = xt_ + xt_d.blk_off(it, b);
        float * ws_l2r_ptr = &(ws_states(0, 0, it + 1, b, 0));
        float * ws_r2l_ptr = &(ws_states(rnn.n_dir - 1, 0, rnn.n_iter - it, b, 0));
        if (rnn.exec_dir != r2l)
            for (int c = 0; c < rnn.slc; c++)
                ws_l2r_ptr[c] = xxt[c];
        if (rnn.exec_dir != l2r)
            for (int c = 0; c < rnn.slc; c++)
                ws_r2l_ptr[c] = xxt[c];
    });
}

template <>
void _ref_rnn_common_t<prop_kind::backward>::copy_init_layer(
        const rnn_conf_t &rnn, float *ws_states_, float *ws_diff_states_,
        const float *xt_, const float *diff_dst_layer_) const {
    AOC<float, 6> ws_diff_states(ws_diff_states_, rnn.n_layer + 1, rnn.n_dir,
            (rnn.n_states + 1), rnn.n_iter + 1, rnn.mb, rnn.states_ws_ld);
    auto diff_dst_layer_d = memory_desc_wrapper(pd()->diff_dst_pd(0));

    switch (rnn.exec_dir) {
    case bi_concat:
        parallel_nd(rnn.n_iter, rnn.mb, [&](int it, int b) {
            auto diff_dst_layer_x
                    = diff_dst_layer_ + diff_dst_layer_d.blk_off(it, b);
            for (int s = 0; s < rnn.dic; s++) {
                ws_diff_states(rnn.n_layer, 0, rnn.n_states, it, b, s)
                        = diff_dst_layer_x[s];
                ws_diff_states(
                        rnn.n_layer, 1, rnn.n_states, rnn.n_iter - it - 1, b, s)
                        = diff_dst_layer_x[rnn.dic + s];
            }
        });
        break;
    case bi_sum:
        parallel_nd(rnn.n_iter, rnn.mb, [&](int it, int b) {
            auto diff_dst_layer_x
                    = diff_dst_layer_ + diff_dst_layer_d.blk_off(it, b);
            for (int s = 0; s < rnn.dic; s++) {
                ws_diff_states(rnn.n_layer, 0, rnn.n_states, it, b, s)
                        = diff_dst_layer_x[s];
                ws_diff_states(
                        rnn.n_layer, 1, rnn.n_states, rnn.n_iter - it - 1, b, s)
                        = diff_dst_layer_x[s];
            }
        });
        break;
    case l2r:
        parallel_nd(rnn.n_iter, rnn.mb, [&](int it, int b) {
            auto diff_dst_layer_x
                    = diff_dst_layer_ + diff_dst_layer_d.blk_off(it, b);
            for (int s = 0; s < rnn.dic; s++) {
                ws_diff_states(rnn.n_layer, 0, rnn.n_states, it, b, s)
                        = diff_dst_layer_x[s];
            }
        });
        break;
    case r2l:
        parallel_nd(rnn.n_iter, rnn.mb, [&](int it, int b) {
            auto diff_dst_layer_x = diff_dst_layer_
                    + diff_dst_layer_d.blk_off(rnn.n_iter - it - 1, b);
            for (int s = 0; s < rnn.dic; s++) {
                ws_diff_states(rnn.n_layer, 0, rnn.n_states, it, b, s)
                        = diff_dst_layer_x[s];
            }
        });
        break;
    default: assert(!"Unsupported direction"); break;
    }
}

template <>
void _ref_rnn_common_t<prop_kind::forward>::copy_init_iter(
        const rnn_conf_t &rnn, float * __restrict ws_states_,
        float * __restrict ws_diff_states_,
        const float * __restrict firstit_states_,
        const float * __restrict diff_dst_iter_) const {
    AOC<float, 6> ws_states(ws_states_, rnn.n_layer + 1, rnn.n_dir,
            rnn.n_states, rnn.n_iter + 1, rnn.mb, rnn.states_ws_ld);
    auto firstit_states_d = memory_desc_wrapper(pd()->src_pd(1));
    if (firstit_states_) {
        parallel_nd(rnn.n_layer, rnn.n_dir, rnn.n_states, rnn.mb,
                    [&](int lay, int dir, int state, int b) {
                    array_copy(&(ws_states(lay + 1, dir, state, 0, b, 0)),
                        firstit_states_ + firstit_states_d.blk_off(
                        lay, dir, state, b), rnn.sic);
        });
    } else {
        parallel_nd(rnn.n_layer, rnn.n_dir, rnn.n_states, rnn.mb,
                    [&](int lay, int dir, int state, int b) {
                    for (int j = 0; j < rnn.sic; j++)
                        ws_states(lay + 1, dir, state, 0, b, j) = 0.0f;
        });
    }
}

template <>
void _ref_rnn_common_t<prop_kind::backward>::copy_init_iter(
        const rnn_conf_t &rnn, float *ws_states_, float *ws_diff_states_,
        const float *firstit_states_, const float *diff_dst_iter_) const {
    AOC<float, 6> ws_diff_states(ws_diff_states_, rnn.n_layer + 1, rnn.n_dir,
            rnn.n_states + 1, rnn.n_iter + 1, rnn.mb, rnn.states_ws_ld);
    auto diff_dst_iter_d = memory_desc_wrapper(pd()->diff_dst_pd(1));
    if (diff_dst_iter_) {
        parallel_nd(rnn.n_layer, rnn.n_dir, rnn.n_states, rnn.mb,
                [&](int lay, int dir, int state, int b) {
                    array_copy(&(ws_diff_states(
                                       lay, dir, state, rnn.n_iter, b, 0)),
                            diff_dst_iter_
                                    + diff_dst_iter_d.blk_off(
                                              lay, dir, state, b),
                            rnn.dic);
                });
    } else {
        parallel_nd(rnn.n_layer, rnn.n_dir, rnn.n_states, rnn.mb,
                [&](int lay, int dir, int state, int i) {
                    for (int j = 0; j < rnn.dic; j++)
                        ws_diff_states(lay, dir, state, rnn.n_iter, i, j)
                                = 0.0f;
                });
    }
}

template <>
void _ref_rnn_common_t<prop_kind::forward>::copy_res_layer(
        const rnn_conf_t &rnn, float *dst_layer_, float *diff_src_layer,
        const float *ws_states_, const float *ws_diff_states_) const {
    auto dst_layer_d = memory_desc_wrapper(pd()->dst_pd(0));
    AOC<const float, 6> ws_states(ws_states_, rnn.n_layer + 1, rnn.n_dir,
            rnn.n_states, rnn.n_iter + 1, rnn.mb, rnn.states_ws_ld);

    parallel_nd(rnn.n_iter, rnn.mb, [&](int it, int b) {
        int dir = 0;
        if (rnn.exec_dir != r2l) {
            for (int s = 0; s < rnn.dic; s++)
                dst_layer_[dst_layer_d.blk_off(it, b, dir * rnn.dic + s)]
                        = ws_states(rnn.n_layer, dir, 0, it + 1, b, s);
            dir = 1;
        }
        if (rnn.exec_dir != l2r) {
            for (int s = 0; s < rnn.dic; s++)
                switch (rnn.exec_dir) {
                case bi_sum:
                    dst_layer_[dst_layer_d.blk_off(it, b, s)] += ws_states(
                            rnn.n_layer, dir, 0, rnn.n_iter - it, b, s);
                    break;
                default:
                    dst_layer_[dst_layer_d.blk_off(it, b, dir * rnn.dic + s)]
                            = ws_states(
                                    rnn.n_layer, dir, 0, rnn.n_iter - it, b, s);
                }
        }
    });
}

template <>
void _ref_rnn_common_t<prop_kind::backward>::copy_res_layer(
        const rnn_conf_t &rnn, float *dst_layer_, float *diff_src_layer_,
        const float *ws_states_, const float *ws_diff_states_) const {
    auto diff_src_layer_d = memory_desc_wrapper(pd()->diff_src_pd(0));
    AOC<const float, 6> ws_diff_states(ws_diff_states_, rnn.n_layer + 1,
            rnn.n_dir, rnn.n_states + 1, rnn.n_iter + 1, rnn.mb,
            rnn.states_ws_ld);

    parallel_nd(rnn.n_iter, rnn.mb, [&](int it, int b) {
        int dir = 0;
        for (int s = 0; s < rnn.slc; s++) {
            float *dst_addr = diff_src_layer_
                    + diff_src_layer_d.blk_off(
                              (rnn.exec_dir == r2l) ? rnn.n_iter - 1 - it : it,
                              b, dir * rnn.slc + s);
            float res = ws_diff_states(0, 0, rnn.n_states, it, b, s);
            if (rnn.n_dir - 1)
                res += ws_diff_states(
                        0, 1, rnn.n_states, rnn.n_iter - 1 - it, b, s);
            dst_addr[0] = res;
        }
    });
}

template <>
void _ref_rnn_common_t<prop_kind::forward>::copy_res_iter(
        const rnn_conf_t &rnn, float *dst_iter_, float *diff_src_iter_,
        const float *ws_states_, const float *ws_diff_states_) const {
    auto dst_iter_d = memory_desc_wrapper(pd()->dst_pd(1));
    AOC<const float, 6> ws_states(ws_states_, rnn.n_layer + 1, rnn.n_dir,
            rnn.n_states, rnn.n_iter + 1, rnn.mb, rnn.states_ws_ld);
    if (dst_iter_) {
        parallel_nd(rnn.n_layer, rnn.n_dir, rnn.n_states, rnn.mb,
                [&](int lay, int dir, int state, int b) {
                    for (int s = 0; s < rnn.dic; s++) {
                        dst_iter_[dst_iter_d.blk_off(lay, dir, state, b, s)]
                                = ws_states(
                                        lay + 1, dir, state, rnn.n_iter, b, s);
                    }
                });
    }
}

template <>
void _ref_rnn_common_t<prop_kind::backward>::copy_res_iter(
        const rnn_conf_t &rnn, float *dst_iter_, float *diff_src_iter_,
        const float *ws_states_, const float *ws_diff_states_) const {
    auto diff_src_iter_d = memory_desc_wrapper(pd()->diff_src_pd(1));
    AOC<const float, 6> ws_diff_states(ws_diff_states_, rnn.n_layer + 1,
            rnn.n_dir, rnn.n_states + 1, rnn.n_iter + 1, rnn.mb,
            rnn.states_ws_ld);
    if (diff_src_iter_) {
        parallel_nd(rnn.n_layer, rnn.n_dir, rnn.n_states, rnn.mb,
                [&](int lay, int dir, int state, int b) {
                    for (int s = 0; s < rnn.sic; s++) {
                        diff_src_iter_[diff_src_iter_d.blk_off(
                                lay, dir, state, b, s)]
                                = ws_diff_states(lay, dir, state, 0, b, s);
                    }
                });
    }
}

template <prop_kind_t aprop>
bias_prepare_sig(_ref_rnn_common_t<aprop>::bias_prepare) {
    /* Original set of bias provided by the user */
    AOC<const float, 5> b(
            b_, rnn.n_layer, rnn.n_dir, rnn.n_bias * rnn.dic);
    /* Array of pointers initialized in packing */
    AOC<float *, 3> bias(bias_, rnn.n_layer, rnn.n_dir, rnn.n_parts_bias);
    AOC<float, 3> scratch_bias(scratch_bias_, rnn.n_layer, rnn.n_dir, rnn.n_bias * rnn.dic);
    for (int i = 0; i < rnn.n_layer; i++) {
        for (int d = 0; d < rnn.n_dir; d++) {
            int offset_bias = 0;
            for (int p = 0; p < rnn.n_parts_bias; p++) {
                bias(i, d, p) = (float *) &b(i, d, offset_bias);
                offset_bias += rnn.parts_bias[p] * rnn.dic;
            }
        }
    }

}

template <prop_kind_t aprop>
bias_finalize_sig(_ref_rnn_common_t<aprop>::bias_finalize) {
}

template <prop_kind_t aprop>
packing_sig(_ref_rnn_common_t<aprop>::pack_weights) {
#if (USE_MKL_PACKED_GEMM)
    /* Original set of weights provided by the user */
    AOC<const float, 5> w(
            w_, rnn.n_layer, rnn.n_dir, IC_size, rnn.n_gates, OC_size);
    /* Array of pointers initialized in packing */
    AOC<float *, 3> weights(weights_, rnn.n_layer, rnn.n_dir, n_parts);

    int m = 0, n = 0, k = 0;
    auto transA = CblasNoTrans;
    bool is_igo = fmt == memory_format::ldigo;
    if (is_igo) {
        m = rnn.n_gates * OC_size;
        n = rnn.mb;
        k = IC_size;
        //todo: do a transposition if ldgoi
        transA = CblasNoTrans;
    } else {
        m = IC_size;
        n = rnn.mb;
        k = rnn.n_gates * OC_size;
        //TODO: do a transposition if ldigo
        transA = CblasNoTrans;
    }

    int total_pack_size = 0;
    for (int p = 0; p < n_parts; p++)
        total_pack_size += part_weights_pack_size[p];

    AOC<float, 3> scratch_weights(scratch_weights_, rnn.n_layer, rnn.n_dir, total_pack_size);
    for (int i = 0; i < rnn.n_layer; i++) {
        for (int d = 0; d < rnn.n_dir; d++) {
            int offset_weights = 0;
            for (int p = 0; p < n_parts; p++) {
                int m_p = is_igo ? (gates_per_part[p] * OC_size) : m;
                int k_p = is_igo ? k : (gates_per_part[p] * OC_size);
                int g = (p > 0) ? gates_per_part[p - 1] : 0;
                weights(i, d, p) = &scratch_weights(i, d, offset_weights);
                cblas_sgemm_pack(CblasColMajor, CblasAMatrix, transA, m_p, n,
                        k_p, 1.0f, &(w(i, d, 0, g, 0)), m, weights(i, d, p));
                offset_weights += part_weights_pack_size[p];
            }
        }
    }
#else
    assert(!"packed gemm is disabled");
    UNUSED(rnn);
    UNUSED(OC_size);
    UNUSED(IC_size);
    UNUSED(weights_);
    UNUSED(n_parts);
    UNUSED(gates_per_part);
    UNUSED(w_);
    UNUSED(scratch_weights_);
    UNUSED(scratch_bias_);
    UNUSED(do_copy);
#endif
}

template <prop_kind_t aprop>
packing_sig(_ref_rnn_common_t<aprop>::no_pack_weights) {
    /* Original set of weights provided by the user */
    AOC<const float, 3> w(
            w_, rnn.n_layer, rnn.n_dir, IC_size * rnn.n_gates * OC_size);
    /* Array of pointers initialized in packing */
    AOC<float *, 3> weights(weights_, rnn.n_layer, rnn.n_dir, n_parts);
    int m = 0, n = 0, ldA = 0;

    bool is_igo = fmt == memory_format::ldigo;
    if (is_igo) {
        m = rnn.n_gates * OC_size;
        n = IC_size;
    } else {
        m = IC_size;
        n = rnn.n_gates * OC_size;
    }
    ldA = get_good_ld(m);

    if (!do_copy) {
        for (int i = 0; i < rnn.n_layer; i++)
            for (int d = 0; d < rnn.n_dir; d++) {
                size_t offset_weights = 0;
                for (int p = 0; p < n_parts; p++) {
                    weights(i, d, p) = (float *) &w(i, d, offset_weights);
                    offset_weights += is_igo
                        ? gates_per_part[p] * OC_size
                        : gates_per_part[p] * OC_size * IC_size;
                }
            }
        return;
    }

    /* We always assume
       - column major
       - alpha = 1.0f
    */
    auto copy_matrix = [](char trans, int nrows, int ncols, const float *src,
            const int ld_src, float *dst, const int ld_dst) {
        parallel_nd(ncols, [&](int i) {
                for (int j = 0; j < nrows; j++)
                    dst[i * ld_dst + j] = src[i * ld_src + j];
            });
    };

    AOC<float, 3> tmp(scratch_weights_, rnn.n_layer, rnn.n_dir, ldA * n);
    parallel_nd(rnn.n_layer, rnn.n_dir, [&](int i, int d) {
            auto src_mat = &(w(i, d, 0));
            auto dst_mat = &(tmp(i, d, 0));
            copy_matrix('N', m, n, src_mat, m, dst_mat, ldA);
            weights(i, d, 0) = &tmp(i, d, 0);
            for (int p = 1; p < n_parts; p++) {
                size_t offset = is_igo
                    ? gates_per_part[p - 1] * OC_size
                    : gates_per_part[p - 1] * OC_size * ldA;
                weights(i, d, p) = &tmp(i, d, offset);
            }
        });
}

template <prop_kind_t aprop>
free_packed_sig(_ref_rnn_common_t<aprop>::free_packed_weights) {
    UNUSED(rnn);
    UNUSED(n_parts);
    UNUSED(weights_);
#if !(USE_MKL_PACKED_GEMM)
    assert(!"packed gemm is disabled");
#endif
}

template <prop_kind_t aprop>
free_packed_sig(_ref_rnn_common_t<aprop>::free_no_packed_weights) {
    // IN this case, only scratchpad is used, so no free necessary
}

//********************* Execution function *********************//
template <prop_kind_t aprop>
void _ref_rnn_common_t<aprop>::execute_() const {
    const rnn_conf_t &rnn = this->pd()->rnn_;

    int input_idx = 0;
    int output_idx = 0;
    auto input
            = reinterpret_cast<const float *>(this->input_memory(input_idx++));
    auto states = pd()->with_src_iter() ?
            reinterpret_cast<const float *>(this->input_memory(input_idx++)) :
            nullptr;
    auto w_input
            = reinterpret_cast<const float *>(this->input_memory(input_idx++));
    auto w_state
            = reinterpret_cast<const float *>(this->input_memory(input_idx++));
    auto bias = pd()->with_bias() ?
            reinterpret_cast<const float *>(this->input_memory(input_idx++)) :
            nullptr;

    auto dst_last_layer = rnn.is_fwd ?
            reinterpret_cast<float *>(this->memory(output_idx++)) :
            const_cast<float *>(reinterpret_cast<const float *>(
                    this->input_memory(input_idx++)));
    auto dst_last_iter = pd()->with_dst_iter() ?
            (rnn.is_fwd ? reinterpret_cast<float *>(
                                  this->memory(output_idx++)) :
                          const_cast<float *>(reinterpret_cast<const float *>(
                                  this->input_memory(input_idx++)))) :
            nullptr;

    auto diff_dst_layer = rnn.is_fwd ?
            nullptr :
            reinterpret_cast<const float *>(this->input_memory(input_idx++));
    auto diff_dst_iter = rnn.is_fwd || !pd()->with_dst_iter() ?
            nullptr :
            reinterpret_cast<const float *>(this->input_memory(input_idx++));

    auto scratchpad = this->scratchpad();

    auto ptr_wei_layer =
        scratchpad.template get<float *>(key_rnn_ptrs_wei_layer);
    auto ptr_wei_iter =
        scratchpad.template get<float *>(key_rnn_ptrs_wei_iter);
    auto ptr_bias =
        scratchpad.template get<float *>(key_rnn_ptrs_bia);

    // fetchihg buffers from the workspace
    // if no workspace was provided we use the scratchpad
    float *scratch_ptr = scratchpad.template get<float>(key_rnn_space);
    float *ws_ptr = nullptr;
    if (rnn.use_workspace)
        ws_ptr = rnn.is_fwd ?
            reinterpret_cast<float *>(this->memory(output_idx++)) :
            const_cast<float *>(reinterpret_cast<const float *>(
                    this->input_memory(input_idx++)));
    float *base_ptr = rnn.use_workspace ? ws_ptr : scratch_ptr;
    float *ws_gates = base_ptr + ws_gates_offset_;
    float *ws_states = base_ptr + ws_states_offset_;
    float *ws_diff_states = base_ptr + ws_diff_states_offset_;
    float *ws_grid = base_ptr + ws_grid_comp_offset_;
    float *ws_cell = base_ptr + ws_cell_comp_offset_;

    auto diff_src_layer = rnn.is_fwd ?
            nullptr :
            reinterpret_cast<float *>(this->memory(output_idx++));
    auto diff_src_iter = rnn.is_fwd || !pd()->with_src_iter() ?
            nullptr :
            reinterpret_cast<float *>(this->memory(output_idx++));
    auto diff_weights_layer = rnn.is_fwd ?
            nullptr :
            reinterpret_cast<float *>(this->memory(output_idx++));
    auto diff_weights_iter = rnn.is_fwd ?
            nullptr :
            reinterpret_cast<float *>(this->memory(output_idx++));
    auto diff_bias = rnn.is_fwd || !pd()->with_bias() ?
            nullptr :
            reinterpret_cast<float *>(this->memory(output_idx++));

    // Fetching extra buffers from scratchpad
    float *ws_weights_layer = scratch_ptr + ws_weights_layer_offset_;
    float *ws_weights_iter = scratch_ptr + ws_weights_iter_offset_;
    float *ws_bias = scratch_ptr + ws_bias_offset_;
    float *ws_diff_weights_layer = scratch_ptr + ws_diff_weights_layer_offset_;
    float *ws_diff_weights_iter = scratch_ptr + ws_diff_weights_iter_offset_;

    // initialize diff_states to 0
    if (aprop == prop_kind::backward) {
        array_set(ws_diff_states, 0.0f, rnn.ws_diff_states_size);
        if (rnn.copy_diff_weights_layer) {
            parallel_nd(rnn.ws_diff_weights_layer_size,
                    [&](size_t i) { ws_diff_weights_layer[i] = 0.; });
        } else
            ws_diff_weights_layer = diff_weights_layer;
        if (rnn.copy_diff_weights_iter) {
            parallel_nd(rnn.ws_diff_weights_iter_size,
                    [&](size_t i) { ws_diff_weights_iter[i] = 0.; });
        } else
            ws_diff_weights_iter = diff_weights_iter;
    }

    /* Pack(if using packed gemm API) or copy(if input arrays have bad leading
     * dimension */
    (this->*bias_preparation_func)(rnn, ptr_bias, bias, ws_bias);

    (this->*weights_iter_pack_func)(rnn, rnn.weights_iter_fmt, rnn.dic, rnn.sic,
            rnn.n_parts_weights_iter, rnn.parts_weights_iter,  rnn.part_weights_iter_pack_size,
            ptr_wei_iter, w_state, ws_weights_iter,
            ptr_bias, bias, ws_bias,
            rnn.copy_weights_iter);
    (this->*weights_layer_pack_func)(rnn, rnn.weights_layer_fmt, rnn.dic, rnn.slc,
            rnn.n_parts_weights_layer, rnn.parts_weights_layer, rnn.part_weights_layer_pack_size,
            ptr_wei_layer, w_input, ws_weights_layer,
            ptr_bias, bias, ws_bias,
            rnn.copy_weights_layer);

    (this->*bias_finalization_func)(rnn, ptr_bias, bias, ws_bias);

    // we first need to copy the initial states and input into ws
    copy_init_layer(rnn, ws_states, ws_diff_states, input, diff_dst_layer);
    copy_init_iter(rnn, ws_states, ws_diff_states, states, diff_dst_iter);

    // run the execution on the grid
    (this->*grid_computation)(rnn, ptr_wei_layer, ptr_wei_iter,
            ptr_bias, ws_states, ws_diff_states,
            ws_gates, ws_cell, ws_grid, ws_diff_weights_layer,
            ws_diff_weights_iter, diff_bias);

    // Finally we copy the results to the result buffers
    copy_res_layer(
            rnn, dst_last_layer, diff_src_layer, ws_states, ws_diff_states);
    copy_res_iter(
            rnn, dst_last_iter, diff_src_iter, ws_states, ws_diff_states);

    // copy of the diff weights if bwd
    if (aprop == prop_kind::backward){
        // TODO: write an impl of matcopy in MKL-DNN
        // TODO: support ldgoi using the trans parameters
        AOC<float, 3> diff_weights_layer_aoc_t(diff_weights_layer, rnn.n_layer,
                rnn.n_dir,
                rnn.diff_weights_layer_nld * rnn.diff_weights_layer_ld);
        AOC<float, 3> diff_weights_iter_aoc_t(diff_weights_iter, rnn.n_layer,
                rnn.n_dir,
                rnn.diff_weights_iter_nld * rnn.diff_weights_iter_ld);
        AOC<float, 3> ws_diff_weights_layer_aoc_t(ws_diff_weights_layer,
                rnn.n_layer, rnn.n_dir,
                rnn.diff_weights_layer_nld * rnn.diff_weights_layer_ws_ld);
        AOC<float, 3> ws_diff_weights_iter_aoc_t(ws_diff_weights_iter,
                rnn.n_layer, rnn.n_dir,
                rnn.diff_weights_iter_nld * rnn.diff_weights_iter_ws_ld);

        /*
           - assumes column major and non transposed matrices
           - computes B = A + B
        */
        auto inplace_matadd = [=](const int nrows, const int ncols,
                const float *A, const int ldA, float *B, const int ldB){
            for(int i = 0; i < ncols; i++)
                for(int j = 0; j < nrows; j++)
                    B[i * ldB + j] += A[i * ldA + j];
        };
        parallel_nd(rnn.n_layer, rnn.n_dir, [&](int i, int d) {
            auto wei_lay = &(diff_weights_layer_aoc_t(i, d, 0));
            auto wei_it = &(diff_weights_iter_aoc_t(i, d, 0));
            auto ws_wei_lay = &(ws_diff_weights_layer_aoc_t(i, d, 0));
            auto ws_wei_it = &(ws_diff_weights_iter_aoc_t(i, d, 0));
            if (rnn.copy_diff_weights_layer)
                inplace_matadd(rnn.n_gates * rnn.dic, rnn.slc, ws_wei_lay,
                        rnn.diff_weights_layer_ws_ld, wei_lay,
                        rnn.diff_weights_layer_ld);
            if (rnn.copy_diff_weights_iter)
                inplace_matadd(rnn.n_gates * rnn.dic, rnn.sic, ws_wei_it,
                        rnn.diff_weights_iter_ws_ld, wei_it,
                        rnn.diff_weights_iter_ld);
        });
    }

    // We free the packed weights if they were packed internally
    (this->*weights_iter_free_packed_func)(rnn, rnn.n_parts_weights_iter, ptr_wei_iter);
    (this->*weights_layer_free_packed_func)(rnn, rnn.n_parts_weights_layer, ptr_wei_layer);
};

/* Fix for MSVS warning C4661 */
template<> cell_execution_sig(ref_rnn_fwd_t::cell_execution);
template<> cell_execution_sig(ref_rnn_bwd_t::cell_execution);
template<> cell_execution_sig(ref_rnn_fwd_t::cell_execution_gru);
template<> cell_execution_sig(ref_rnn_bwd_t::cell_execution_gru);
template<> cell_execution_sig(ref_rnn_fwd_t::cell_execution_gru_lbr);
template<> cell_execution_sig(ref_rnn_bwd_t::cell_execution_gru_lbr);
template<> elemwise_sig(ref_rnn_fwd_t::rnn_elemwise);
template<> elemwise_sig(ref_rnn_bwd_t::rnn_elemwise);
template<> elemwise_sig(ref_rnn_fwd_t::lstm_elemwise);
template<> elemwise_sig(ref_rnn_bwd_t::lstm_elemwise);
template<> elemwise_sig(ref_rnn_fwd_t::gru_lbr_elemwise);
template<> elemwise_sig(ref_rnn_bwd_t::gru_lbr_elemwise);

template struct _ref_rnn_common_t<prop_kind::forward>;
template struct _ref_rnn_common_t<prop_kind::backward>;

#undef AOC
}
}
}
