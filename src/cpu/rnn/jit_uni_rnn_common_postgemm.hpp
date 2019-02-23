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

#ifndef CPU_JIT_RNN_POSTGEMM
#define CPU_JIT_RNN_POSTGEMM


#include "rnn_utils.hpp"
#include "../jit_generator.hpp"
#include "../jit_uni_eltwise.hpp"
#include "c_types_map.hpp"
#include "utils.hpp"

#include "mkldnn_thread.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_uni_rnn_postgemm : public jit_generator {

    typedef void (*kernel_t)(void *gates_, const void *bias, void *states_t_l_,
                     void *c_states_t_l_, void *c_states_tm1_l_);

    jit_uni_rnn_postgemm(const rnn_utils::rnn_conf_t &rnn, const primitive_attr_t *attr): rnn_(rnn), attr_(attr){}

    virtual void init() = 0;

template <typename src_data_t, typename acc_data_t>
    rnn_elemwise_sig(execute) {
        rnn_utils::ws_gates_aoc<acc_data_t> ws_gates(rnn, ws_gates_);
        rnn_utils::bias_aoc_t bias(rnn, bias_);
        rnn_utils::ws_states_aoc<src_data_t> states_t_l(rnn, states_t_l_);
        rnn_utils::ws_states_aoc_t c_states_t_l(rnn, c_states_t_l_);
        rnn_utils::ws_states_aoc_t c_states_tm1_l(rnn, c_states_tm1_l_);

        // Todo: add parallelization on dic for the batch 1 case
        // Assumption: the kernel runs a loop on dic elements
        parallel_nd(rnn.mb, [&](int i) {
                auto b_ = &bias(0, 0);
                auto g_ = &ws_gates(i, 0, 0);
                auto s_tl_ = &states_t_l(i, 0);
                auto c_tl_ = &c_states_t_l(i, 0);
                auto c_tm1l_ = &c_states_tm1_l(i, 0);
                kernel_(g_, b_, s_tl_, c_tm1l_, c_tl_);
            });
    }

protected:
    kernel_t kernel_;
    const rnn_utils::rnn_conf_t &rnn_;
    const primitive_attr_t *attr_;
};


}
}
}

#endif
