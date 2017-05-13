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

#ifndef CPU_JIT_UNI_RELU_HPP
#define CPU_JIT_UNI_RELU_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_relu_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_relu_kernel_f32;

template <cpu_isa_t isa>
struct jit_uni_relu_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_relu_fwd_pd_t {
        pd_t(engine_t *engine, const relu_desc_t *adesc,
             const relu_fwd_pd_t *hint_fwd_pd)
            : cpu_relu_fwd_pd_t(engine, adesc, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(jit_uni_relu_fwd_t<isa>);

        virtual status_t init() override;
    };

    jit_uni_relu_fwd_t(const pd_t *pd, const input_vector &inputs,
                       const output_vector &outputs);
    ~jit_uni_relu_fwd_t();

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e)
    {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
    jit_uni_relu_kernel_f32<isa> *kernel_;
};

template <cpu_isa_t isa>
struct jit_uni_relu_bwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_relu_bwd_pd_t {
        pd_t(engine_t *engine, const relu_desc_t *adesc,
             const relu_fwd_pd_t *hint_fwd_pd)
            : cpu_relu_bwd_pd_t(engine, adesc, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(jit_uni_relu_bwd_t<isa>);

        virtual status_t init() override;
    };

    jit_uni_relu_bwd_t(const pd_t *pd, const input_vector &inputs,
                       const output_vector &outputs);
    ~jit_uni_relu_bwd_t();

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e)
    {
        execute_backward();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward();
    pd_t conf_;
    jit_uni_relu_kernel_f32<isa> *kernel_;
};

}
}
}

#endif
