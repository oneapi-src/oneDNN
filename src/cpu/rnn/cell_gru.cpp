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
 * Cell execution GRU
 */

#include "math_utils.hpp"
#include "mkldnn_thread.hpp"

#include "ref_rnn.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::math;

#define AOC array_offset_calculator

inline float one_m_square(float x) {
    return (1.0f - x) * (1.0f + x);
}
inline float x_m_square(float x) {
    return (1.0f - x) * x;
}

template <>
cell_execution_sig(_ref_rnn_common_t<prop_kind::forward>::cell_execution_gru) {
    AOC<float, 3> ws_gates(ws_gates_, batch, conf_.GC());
    AOC<const float, 2> bias(bias_, n_gates, dic);
    AOC<float, 2> states_t_l(states_t_l_, batch, wic);
    AOC<float, 2> states_tm1_l(states_tm1_l_, batch, wic);

    // 1. gemm Wx[0-2],x
    if (!merge_gemm_layer) {
        (this->*gemm_input_func)(n_gates * dic, batch, slc, conf_.WL_GLD(), slc,
                batch, wic, conf_.GC(), batch, w_input_[0], states_t_lm1_,
                ws_gates_, false, 0.0f);
    }

    // 2. gemm Wh[0-1],h
    (this->*gemm_state_func)((n_gates - 1) * dic, batch, sic, conf_.WI_GLD(),
            sic, batch, wic, conf_.GC(), batch, w_state_[0], states_tm1_l_,
            ws_gates_, false, 1.0f);

    // 3. activation zt and rt + elemwise multiplication rt,ht-1
    parallel_nd(batch, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < dic; j++) {
            ws_gates(i, 0 * dic + j)
                    = logistic_fwd(ws_gates(i, 0 * dic + j) + bias(0, j));
            ws_gates(i, 1 * dic + j)
                    = logistic_fwd(ws_gates(i, 1 * dic + j) + bias(1, j));
            states_t_l(i, j) = states_tm1_l(i, j) * ws_gates(i, 1 * dic + j);
        }
    });

    // 4. gemm Wh[2],h~t
    (this->*gemm_state_func)(dic, batch, sic, conf_.WI_GLD(), sic, batch, wic,
            conf_.GC(), batch, w_state_[1], states_t_l_,
            &(ws_gates(0, 2 * dic)), false, 1.0f);

    // 5. activation h~t + calculate ht
    parallel_nd(batch, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < dic; j++) {
            ws_gates(i, 2 * dic + j)
                    = tanh_fwd(ws_gates(i, 2 * dic + j) + bias(2, j));
            states_t_l(i, j) = states_tm1_l(i, j) * ws_gates(i, 0 * dic + j)
                    + (1.0f - ws_gates(i, 0 * dic + j))
                            * ws_gates(i, 2 * dic + j);
        }
    });
}

template <>
cell_execution_sig(_ref_rnn_common_t<prop_kind::backward>::cell_execution_gru) {
    AOC<float, 2> ws_gates(ws_gates_, batch, conf_.GC());
    AOC<const float, 2> states_tm1_l(states_tm1_l_, batch, wic);
    AOC<float, 4> diff_states_t_l(diff_states_t_l_, n_states + 1, iter_stride,
            batch, wic); // dht-1 dxt
    AOC<float, 3> diff_w_state(diff_w_state_, sic, conf_.GC());
    AOC<float, 4> diff_states_tp1_l(
            diff_states_tp1_l_, n_states + 1, iter_stride, batch, wic);
    AOC<float, 4> diff_states_t_lp1(
            diff_states_t_lp1_, n_states + 1, iter_stride, batch, wic);
    // use state memory for intermediate computations
    float *dhG1_ = &(diff_states_t_l(n_states, 0, 0, 0));
    float *hG1_ = dhG1_;
    AOC<float, 2> dhG1(dhG1_, batch, wic);
    AOC<float, 2> hG1(hG1_, batch, wic);

    // 1. calculate dG2, dG1, and part of dht-1
    // dG2^ = dh * (1 - G0) * (1 - G2^2)
    // dG0^ = dh * (ht-1 - G2) * u * (1 - G0)
    // dht-1 (part) = dh * G0
    parallel_nd(batch, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < dic; j++) {
            float h = states_tm1_l(i, j);
            float dHt = diff_states_tp1_l(0, 0, i, j)
                    + diff_states_t_lp1(n_states, 0, i, j);
            float dG2 = (1.0f - ws_gates(i, 0 * dic + j)) * dHt
                    * one_m_square(ws_gates(i, 2 * dic + j));
            float dG0 = (h - ws_gates(i, 2 * dic + j)) * dHt
                    * x_m_square(ws_gates(i, 0 * dic + j));

            diff_states_t_l(0, 0, i, j) = dHt * ws_gates(i, 0 * dic + j);
            ws_gates(i, 0 * dic + j) = dG0;
            ws_gates(i, 2 * dic + j) = dG2;
        }
    });

    // 2. calculate intermediate d(hG1)
    // d(hG1) = dG2 * W2h^t
    (this->*gemm_state_func)(sic, batch, dic, conf_.WI_GLD(), n_gates * dic,
            batch, conf_.GC(), wic, batch, w_state_[1], &(ws_gates(0, 2 * dic)),
            dhG1_, false, 0.0f);

    // 3. calculate dG1^ and part of dht-1
    // dG1^ = d(hG1) * h * G1 * (1 - G1)
    // dht-1 (part) += d(hG1) * G1
    // h * G1 (required for dWh)
    parallel_nd(batch, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < dic; j++) {
            float h = states_tm1_l(i, j);
            float G1 = ws_gates(i, 1 * dic + j);
            diff_states_t_l(0, 0, i, j) += dhG1(i, j) * G1;
            ws_gates(i, 1 * dic + j) = dhG1(i, j) * h * x_m_square(G1);
            hG1(i, j) = G1 * h;
        }
    });

    // 4. calculate diff weights
    // dWh1 += dG1 * h, dWh2 += dG2 * h, dWh3 += dG3 * (G1(*)h)
    gemm((n_gates - 1) * dic, sic, batch, conf_.GC(), batch, wic, batch,
            conf_.DWI_GLD(), sic, ws_gates_, states_tm1_l_, diff_w_state_, true,
            1.0f);
    gemm(dic, sic, batch, conf_.GC(), batch, wic, batch, conf_.DWI_GLD(), sic,
            &(ws_gates(0, 2 * dic)), hG1_, &(diff_w_state(0, 2 * dic)), true,
            1.0f);

    // 5. calculate diff states
    // dht-1 += dG1 * W1h + dG0 * W0h
    (this->*gemm_state_func)(sic, batch, (n_gates - 1) * dic, conf_.WI_GLD(),
            n_gates * dic, batch, conf_.GC(), wic, batch, w_state_[0],
            ws_gates_, diff_states_t_l_, false, 1.0f);

    if (!merge_gemm_layer) {
        // dWx += [dG0 dG1 dG2] * [x]
        gemm(n_gates * dic, slc, batch, conf_.GC(), batch, wic, batch,
                conf_.DWL_GLD(), slc, ws_gates_, states_t_lm1_, diff_w_input_,
                true, 1.0f);
        // dx = dG2 * W2x + dG1 * W1x + dG0 * W0x
        (this->*gemm_input_func)(slc, batch, n_gates * dic, conf_.WL_GLD(),
                n_gates * dic, batch, conf_.GC(), wic, batch, w_input_[0],
                ws_gates_, &(diff_states_t_l(n_states, 0, 0, 0)), false, 0.0f);
    }

    // 6. calculate diff bias
    gates_reduction(n_gates, dic, wic, batch, ws_gates_, diff_bias_);
}
#undef AOC

}
}
}
