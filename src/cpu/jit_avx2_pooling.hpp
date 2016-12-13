/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#ifndef CPU_JIT_AVX2_POOLING_HPP
#define CPU_JIT_AVX2_POOLING_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_pooling_pd.hpp"
#include "cpu_engine.hpp"
#include "jit_avx2_pool_kernel_f32.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx2_pooling_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_pooling_fwd_pd_t {
        pd_t(engine_t *engine, const pooling_desc_t *adesc,
                const pooling_fwd_pd_t *hint_fwd_pd)
            : cpu_pooling_fwd_pd_t(engine, adesc, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(jit_avx2_pooling_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace alg_kind;
            using namespace utils;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && set_default_params() == status::success
                && one_of(desc()->prop_kind, forward_training,
                        forward_inference)
                && one_of(desc()->alg_kind, pooling_max, pooling_avg)
                && everyone_is(data_type::f32, src_pd()->desc()->data_type,
                        dst_pd()->desc()->data_type);
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == forward_training;
            if (desc()->alg_kind == pooling_max && is_training) {
                auto indices_desc = *dst_pd()->desc();
                indices_desc.data_type = data_type::s32;
                ws_pd_ = cpu_memory_t::pd_t(engine_, &indices_desc);
            }

            return jit_avx2_pool_kernel_f32::init_conf(jpp_, desc_,
                    src_pd_.desc(), dst_pd_.desc(), is_training);
        }

        jit_pool_conf_t jpp_;
    };

    jit_avx2_pooling_fwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
    { kernel_ = new jit_avx2_pool_kernel_f32(conf_.jpp_); }
    ~jit_avx2_pooling_fwd_t() { delete kernel_; };

    typedef typename prec_trait<data_type::f32>::type data_t;

    virtual void execute(event_t *e) {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
    jit_avx2_pool_kernel_f32 *kernel_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

