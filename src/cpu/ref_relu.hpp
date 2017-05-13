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

#ifndef CPU_REF_RELU_HPP
#define CPU_REF_RELU_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_relu_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct ref_relu_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_relu_fwd_pd_t {
        pd_t(engine_t *engine, const relu_desc_t *adesc,
                const relu_fwd_pd_t *hint_fwd_pd)
            : cpu_relu_fwd_pd_t(engine, adesc, hint_fwd_pd), is_dense(false) {}

        DECLARE_COMMON_PD_T(ref_relu_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(engine()->kind() == engine_kind::cpu);

            is_dense = memory_desc_wrapper(src_pd()).is_dense();
            bool ok = true
                && utils::one_of(desc()->prop_kind, forward_training,
                        forward_inference)
                && utils::everyone_is(data_type, desc()->data_desc.data_type)
                && utils::implication(!is_dense, src_pd()->desc()->ndims == 4);
            if (!ok) return status::unimplemented;

            return status::success;
        }

        bool is_dense;
    };

    ref_relu_fwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) {
        if (conf_.is_dense) execute_forward_dense();
        else execute_forward_generic();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward_dense();
    void execute_forward_generic();
    pd_t conf_;
};

template <impl::data_type_t data_type>
struct ref_relu_bwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_relu_bwd_pd_t {
        pd_t(engine_t *engine, const relu_desc_t *adesc,
                const relu_fwd_pd_t *hint_fwd_pd)
            : cpu_relu_bwd_pd_t(engine, adesc, hint_fwd_pd), is_dense_(false)
        {}

        DECLARE_COMMON_PD_T(ref_relu_bwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && desc()->prop_kind == backward_data
                && utils::everyone_is(data_type, desc()->data_desc.data_type,
                        desc()->diff_data_desc.data_type);
            if (!ok) return status::unimplemented;

            const bool same_fmt = memory_desc_wrapper(diff_dst_pd())
                == memory_desc_wrapper(src_pd());
            is_dense_ = memory_desc_wrapper(src_pd()).is_dense() && same_fmt;

            return status::success;
        }

        bool is_dense_;
    };

    ref_relu_bwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) {
        if (conf_.is_dense_) execute_backward_dense();
        else execute_backward_generic();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_dense();
    void execute_backward_generic();
    pd_t conf_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
