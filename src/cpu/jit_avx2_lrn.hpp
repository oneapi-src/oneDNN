/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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

#ifndef CPU_JIT_AVX2_LRN_FWD_HPP
#define CPU_JIT_AVX2_LRN_FWD_HPP

#include "c_types_map.hpp"
#include "cpu_lrn_pd.hpp"
#include "cpu_engine.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx2_lrn_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_lrn_fwd_pd_t {
        pd_t(engine_t *engine, const lrn_desc_t *adesc,
                const lrn_fwd_pd_t *hint_fwd_pd)
            : cpu_lrn_fwd_pd_t(engine, adesc, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(jit_avx2_lrn_fwd_t);

        virtual status_t init() override;
    };

    jit_avx2_lrn_fwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs);
    ~jit_avx2_lrn_fwd_t();

    typedef typename prec_trait<data_type::f32>::type data_t;

    virtual void execute(event_t *e) {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;

    struct jit_avx2_lrn_kernel_f32;
    jit_avx2_lrn_kernel_f32 *ker_, *ker_first_, *ker_last_;
};

struct jit_avx2_lrn_bwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_lrn_bwd_pd_t {
        pd_t(engine_t *engine, const lrn_desc_t *adesc,
                const lrn_fwd_pd_t *hint_fwd_pd)
            : cpu_lrn_bwd_pd_t(engine, adesc, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(jit_avx2_lrn_bwd_t);

        virtual status_t init() override;
    };

    jit_avx2_lrn_bwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs);
    ~jit_avx2_lrn_bwd_t();

    typedef typename prec_trait<data_type::f32>::type data_t;

    virtual void execute(event_t *e) {
        execute_backward();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward();
    pd_t conf_;

    struct jit_avx2_lrn_kernel_f32;
    jit_avx2_lrn_kernel_f32 *ker_, *ker_first_, *ker_last_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
