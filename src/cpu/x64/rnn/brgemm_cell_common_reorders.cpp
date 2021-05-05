/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "cpu/x64/rnn/brgemm_cell_common_reorders.hpp"
#include "cpu/rnn/rnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <typename Dt>
void scratch_gates_blocked_reorder_t::execute(
        const Dt *src, Dt *dst, const bool n_tail) const {

    const auto ld_GO = rnn_.scratch_gates_ld;
    const auto &d_wei = rnn_.diff_wei_brgemm;
    const auto I = rnn_.mb;
    const auto O = n_tail ? d_wei.n_tail : d_wei.n_block;
    const auto o_block = 32;
    const auto i_block = 4 / sizeof(Dt);

    for (int ib = 0; ib < I; ib += i_block) {
        const auto off_plain = ib * ld_GO;
        const auto off_blk = ib * o_block;

        const Dt *const inp = &src[off_plain];
        Dt *const out = &dst[off_blk];

        for (int i = 0; i < i_block; i++) {
            const bool should_fill_ib = ((i + ib) < I);
            const auto inp_i_off = i * ld_GO;
            for (int o = 0; o < o_block; o++) {
                const auto off_inner_blk = o * i_block + i;
                if (should_fill_ib && (o < O))
                    out[off_inner_blk] = inp[inp_i_off + o];
                else
                    out[off_inner_blk] = 0;
            }
        }
    }
}

template <>
void src_layer_iter_transpose_t::execute<bfloat16_t>(
        const bfloat16_t *src, bfloat16_t *dst) const {
    // (mb, slc) -> (rnn.slc, rnn.mb)
    //  I    O       m_block    I

    const int O = m_block_;
    const int I = rnn_.mb;

    if (rnn_.mb == 1) {
        uint32_t *dst_32 = reinterpret_cast<uint32_t *>(dst);
        for (int o = 0; o < O; o++) {
            const uint32_t inp = static_cast<uint32_t>(src[o].raw_bits_);
            uint32_t &out = dst_32[o];
            out = inp;
        }
    } else {
        const auto I_extended = utils::rnd_up(I, 2);
        const bool add_extra_column = (I_extended > I);
        for (int o = 0; o < O; o++) {
            const auto base_off = o * I_extended;
            for (int i = 0; i < I; i++) {
                const auto inp = &src[i * ld_src_ + o];
                auto out = &dst[base_off + i];
                *out = *inp;
            }
            if (add_extra_column) { dst[base_off + I] = 0; }
        }
    }
}

template <>
void src_layer_iter_transpose_t::execute<float>(
        const float *src, float *dst) const {
    // (mb, slc) -> (rnn.slc, rnn.mb)
    //  I    O       m_block    I

    const int O = m_block_;
    const int I = rnn_.mb;

    for (int o = 0; o < O; o++) {
        const auto base_off = o * I;
        for (int i = 0; i < I; i++) {
            const auto inp = &src[i * ld_src_ + o];
            auto out = &dst[base_off + i];
            *out = *inp;
        }
    }
}

template <typename Dt>
void src_layer_iter_transpose_t::execute_in_parallel(
        const Dt *src, Dt *dst) const {
    // (mb, slc) -> (rnn.slc, rnn.mb)
    //  I    O       m_block    I

    const int O = m_block_;
    const int I = rnn_.mb;
    const auto I_extended
            = (sizeof(Dt) != sizeof(float)) ? utils::rnd_up(I, 2) : I;
    const bool add_extra_column = (I_extended > I);

    parallel_nd(O, I, [&](int o, int i) {
        const auto inp = &src[i * ld_src_ + o];
        auto out = &dst[o * I_extended + i];
        *out = *inp;
        if (i == I - 1 && add_extra_column) {
            auto out = &dst[o * I_extended + I];
            *out = 0;
        }
    });
}

template void scratch_gates_blocked_reorder_t::execute<float>(
        const float *, float *, const bool) const;
template void scratch_gates_blocked_reorder_t::execute<bfloat16_t>(
        const bfloat16_t *, bfloat16_t *, const bool) const;

template void src_layer_iter_transpose_t::execute_in_parallel<float>(
        const float *, float *) const;
template void src_layer_iter_transpose_t::execute_in_parallel<bfloat16_t>(
        const bfloat16_t *, bfloat16_t *) const;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
