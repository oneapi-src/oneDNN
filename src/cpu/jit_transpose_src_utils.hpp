/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#ifndef CPU_JIT_TRANSPOSE_SRC_HPP
#define CPU_JIT_TRANSPOSE_SRC_HPP

#include "cpu_barrier.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_trans_src_t {
    struct ctx_t {
        const void *src;
        const void *tr_src;
        const void *src_prf;
        const void *tr_src_prf;

        /* 1st conv 4fma: backward by weights */
        int nthr_oc_b; /* number of threads process given src image */
        int tr_src_ih_start, tr_src_ih_end; /* thread's transposition bounds */
        simple_barrier::ctx_t *tr_src_bctx; /* transposition synchronization */
    };

    jit_trans_src_t(const jit_conv_conf_t *conf)
        : conf_(conf), ker_(nullptr) {}
    virtual ~jit_trans_src_t() {}

    void operator()(const ctx_t *ctx)
    { assert(ker_); ker_(ctx); }

    const jit_conv_conf_t *conf_;
    void (*ker_)(const ctx_t *);
};

jit_trans_src_t *create_trans_src(const jit_conv_conf_t *conf);

}
}
}

#endif
