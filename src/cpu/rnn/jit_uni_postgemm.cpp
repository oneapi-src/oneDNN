/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "c_types_map.hpp"

#include "jit_uni_gru_cell_postgemm_1.hpp"
#include "jit_uni_gru_cell_postgemm_2.hpp"
#include "jit_uni_gru_lbr_cell_postgemm.hpp"
#include "jit_uni_lstm_cell_postgemm.hpp"
#include "jit_uni_rnn_cell_postgemm.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template struct jit_uni_gru_cell_postgemm_part1_fwd<sse41, data_type::f32>;
template struct jit_uni_gru_cell_postgemm_part1_fwd<avx2, data_type::f32>;
template struct jit_uni_gru_cell_postgemm_part1_fwd<avx512_core, data_type::f32>;
template struct jit_uni_gru_cell_postgemm_part2_fwd<sse41, data_type::f32>;
template struct jit_uni_gru_cell_postgemm_part2_fwd<avx2, data_type::f32>;
template struct jit_uni_gru_cell_postgemm_part2_fwd<avx512_core, data_type::f32>;

template struct jit_uni_gru_lbr_cell_postgemm_fwd<sse41, data_type::f32>;
template struct jit_uni_gru_lbr_cell_postgemm_fwd<avx2, data_type::f32>;
template struct jit_uni_gru_lbr_cell_postgemm_fwd<avx512_core, data_type::f32>;

template struct jit_uni_lstm_cell_postgemm_fwd<sse41, data_type::f32>;
template struct jit_uni_lstm_cell_postgemm_fwd<avx2, data_type::f32>;
template struct jit_uni_lstm_cell_postgemm_fwd<avx512_core, data_type::f32>;
template struct jit_uni_lstm_cell_postgemm_fwd<sse41, data_type::u8>;
template struct jit_uni_lstm_cell_postgemm_fwd<avx2, data_type::u8>;
template struct jit_uni_lstm_cell_postgemm_fwd<avx512_core, data_type::u8>;

template struct jit_uni_rnn_cell_postgemm_fwd<sse41, data_type::f32>;
template struct jit_uni_rnn_cell_postgemm_fwd<avx2, data_type::f32>;
template struct jit_uni_rnn_cell_postgemm_fwd<avx512_core, data_type::f32>;
template struct jit_uni_rnn_cell_postgemm_fwd<sse41, data_type::u8>;
template struct jit_uni_rnn_cell_postgemm_fwd<avx2, data_type::u8>;
template struct jit_uni_rnn_cell_postgemm_fwd<avx512_core, data_type::u8>;


}
}
}

