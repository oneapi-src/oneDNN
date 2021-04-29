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

#ifndef CPU_X64_RNN_BRGEMM_CELL_COMMON_REORDERS_HPP
#define CPU_X64_RNN_BRGEMM_CELL_COMMON_REORDERS_HPP

#include "cpu/x64/rnn/jit_brgemm_transpose.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace rnn_utils {
struct rnn_conf_t;
}
namespace x64 {
struct scratch_gates_blocked_reorder_t {
    scratch_gates_blocked_reorder_t(const cpu::rnn_utils::rnn_conf_t &rnn)
        : rnn_(rnn) {};

    template <typename Dt>
    void execute(const Dt *src, Dt *dst, const bool n_tail) const;

private:
    const cpu::rnn_utils::rnn_conf_t &rnn_;
};

struct src_layer_iter_transpose_t {
    src_layer_iter_transpose_t(const cpu::rnn_utils::rnn_conf_t &rnn,
            const int m_block, const int ld_src,
            const jit_brgemm_transpose_t *const kernel_transpose = nullptr)
        : rnn_(rnn)
        , m_block_(m_block)
        , ld_src_(ld_src)
        , kernel_transpose_(kernel_transpose) {};

    template <typename Dt>
    void execute(const Dt *src, Dt *dst) const;

    template <typename Dt>
    void execute_in_parallel(const Dt *src, Dt *dst) const;

private:
    const cpu::rnn_utils::rnn_conf_t &rnn_;
    const int m_block_;
    const int ld_src_;
    const jit_brgemm_transpose_t *const kernel_transpose_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
