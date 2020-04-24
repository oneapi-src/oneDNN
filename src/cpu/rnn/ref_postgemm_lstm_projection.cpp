/*******************************************************************************
* Copyright 2020 Intel Corporation
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
 * Cell execution LSTM projection
 */

#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"

#include "cpu/simple_q10n.hpp"

#include "cpu/rnn/postgemm_dispatcher.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::math;
using namespace rnn_utils;

namespace {
template <typename dst_layer_t, typename dst_iter_t>
void proj_dst_copy(const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, dst_iter_t *dst_iter_,
        const dst_layer_t *dst_layer_) {
    assert(rnn.dic == rnn.dlc);
    static_assert(sizeof(dst_layer_t) == sizeof(dst_iter_t),
            "memcpy requires the same data type size for src and dst");
    auto dst_layer_ld = rnn.dst_layer_ld(cell_position, true);

    // If dst_iter is not nullptr, we need to copy the state to dst_iter
    if (dst_iter_ != nullptr) {
        auto dst_iter_ld = rnn.dst_iter_ld(cell_position);
        for (int i = 0; i < rnn.mb; i++)
            std::memcpy(dst_iter_ + i * dst_iter_ld,
                    dst_layer_ + i * dst_layer_ld,
                    rnn.dlc * sizeof(dst_layer_t));
    }
}
} // namespace

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_f32_t::lstm_projection_postgemm) {
    // nothing to do for f32, except copy to dst_iter if needed
    proj_dst_copy(rnn, cell_position, dst_iter_, dst_layer_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_bf16_t::lstm_projection_postgemm) {
    auto dst_layer_ld = rnn.dst_layer_ld(cell_position, true);

    // Currently, scratch_gates_ contains the output of the projection
    for (int i = 0; i < rnn.mb; i++)
        cvt_float_to_bfloat16((bfloat16_t *)dst_layer_ + i * dst_layer_ld,
                (float *)scratch_gates_ + i * rnn.scratch_gates_ld, rnn.dlc);

    // we copy to dst_iter if necessary
    proj_dst_copy(rnn, cell_position, dst_iter_, dst_layer_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_u8_t::lstm_projection_postgemm) {
    // Here, we use
    // - scratch_gates to pass the s32 output of the projection
    // - src_iter_c to pass the projection compensation

    auto dst_layer_ld = rnn.dst_layer_ld(cell_position, true);
    auto w_proj_comp = src_iter_c_;

    float *weights_projection_scales
            = pd_->attr()->rnn_weights_projection_qparams_.scales_;
    float data_shift = pd_->attr()->rnn_data_qparams_.shift_;
    float data_scale = pd_->attr()->rnn_data_qparams_.scale_;

    auto quantize_f32_u8 = [&](float f) {
        float qf = f * data_scale + data_shift;
        qf = nstl::min(qf, 255.0f);
        qf = nstl::max(qf, 0.0f);
        return qz_a1b0<float, dst_layer_t>()(qf);
    };

    auto dequantize_s32_f32 = [&](gemm_acc_t s, int j) {
        float wscale = pd_->attr()->rnn_weights_projection_qparams_.mask_ == 0
                ? weights_projection_scales[0]
                : weights_projection_scales[j];
        float wcomp = w_proj_comp[j] * data_shift;

        return (saturate<float>(s) - wcomp) / (wscale * data_scale);
    };

    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dlc; j++) {
            int scratch_off = i * rnn.scratch_gates_ld + j;
            int dst_off = i * dst_layer_ld + j;
            float tmp = dequantize_s32_f32(scratch_gates_[scratch_off], j);
            dst_layer_[dst_off] = quantize_f32_u8(tmp);
        }
    });

    proj_dst_copy(rnn, cell_position, dst_iter_, dst_layer_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_f32_t::lstm_projection_postgemm) {
    assert(!"unsupported");
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_bf16_t::lstm_projection_postgemm) {
    assert(!"unsupported");
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
