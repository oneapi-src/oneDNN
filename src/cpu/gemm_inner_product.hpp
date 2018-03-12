/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef CPU_GEMM_INNER_PRODUCT_HPP
#define CPU_GEMM_INNER_PRODUCT_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_inner_product_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct gemm_inner_product_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_inner_product_fwd_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : cpu_inner_product_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("gemm:blas", gemm_inner_product_fwd_t);

        virtual status_t init() override {
#ifdef USE_CBLAS
            using namespace prop_kind;
            using namespace memory_format;
            using namespace utils;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && one_of(desc()->prop_kind, forward_training,
                        forward_inference)
                && everyone_is(data_type, desc()->src_desc.data_type,
                        desc()->weights_desc.data_type,
                        desc()->dst_desc.data_type)
                && implication(this->with_bias(),
                        data_type == desc()->bias_desc.data_type)
                && implication(src_pd_.desc()->format == nChw8c,
                        weights_pd_.desc()->format == oIhw8i)
                && implication(src_pd_.desc()->format == nchw,
                        weights_pd_.desc()->format == oihw)
                && implication(src_pd_.desc()->format == ncdhw,
                        weights_pd_.desc()->format == oidhw)
                && implication(src_pd_.desc()->format == nc,
                        weights_pd_.desc()->format == oi)
                && dst_pd_.desc()->format == nc
                && memory_desc_wrapper(src_pd()).is_dense()
                && memory_desc_wrapper(dst_pd()).is_dense()
                && memory_desc_wrapper(weights_pd()).is_dense()
                && attr()->has_default_values();
            return ok ? status::success : status::unimplemented;
#else
            return status::unimplemented;
#endif
        }
    };

    gemm_inner_product_fwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
};

template <impl::data_type_t data_type>
struct gemm_inner_product_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_inner_product_bwd_data_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : cpu_inner_product_bwd_data_pd_t(engine, adesc, attr,
                    hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("gemm:blas", gemm_inner_product_bwd_data_t);

        virtual status_t init() override {
#ifdef USE_CBLAS
            using namespace prop_kind;
            using namespace memory_format;
            using namespace utils;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && desc()->prop_kind == backward_data
                && everyone_is(data_type, desc()->diff_src_desc.data_type,
                        desc()->weights_desc.data_type,
                        desc()->diff_dst_desc.data_type)
                && implication(diff_src_pd_.desc()->format == nChw8c,
                        weights_pd_.desc()->format == oIhw8i)
                && implication(diff_src_pd_.desc()->format == nchw,
                        weights_pd_.desc()->format == oihw)
                && implication(diff_src_pd_.desc()->format == ncdhw,
                        weights_pd_.desc()->format == oidhw)
                && implication(diff_src_pd_.desc()->format == nc,
                        weights_pd_.desc()->format == oi)
                && diff_dst_pd_.desc()->format == nc
                && memory_desc_wrapper(diff_src_pd()).is_dense()
                && memory_desc_wrapper(diff_dst_pd()).is_dense()
                && memory_desc_wrapper(weights_pd()).is_dense()
                && attr()->has_default_values();
            return ok ? status::success : status::unimplemented;
#else
            return status::unimplemented;
#endif
        }
    };

    gemm_inner_product_bwd_data_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) {
        execute_backward_data();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_data();
    pd_t conf_;
};

template <impl::data_type_t data_type>
struct gemm_inner_product_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_inner_product_bwd_weights_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : cpu_inner_product_bwd_weights_pd_t(engine, adesc, attr,
                    hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("gemm:blas", gemm_inner_product_bwd_weights_t);

        virtual status_t init() override {
#ifdef USE_CBLAS
            using namespace prop_kind;
            using namespace memory_format;
            using namespace utils;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && desc()->prop_kind == backward_weights
                && everyone_is(data_type, desc()->src_desc.data_type,
                        desc()->diff_weights_desc.data_type,
                        desc()->diff_dst_desc.data_type)
                && implication(src_pd_.desc()->format == nChw8c,
                        diff_weights_pd_.desc()->format == oIhw8i)
                && implication(src_pd_.desc()->format == nchw,
                        diff_weights_pd_.desc()->format == oihw)
                && implication(src_pd_.desc()->format == ncdhw,
                        diff_weights_pd_.desc()->format == oidhw)
                && implication(src_pd_.desc()->format == nc,
                        diff_weights_pd_.desc()->format == oi)
                && diff_dst_pd_.desc()->format == nc
                && memory_desc_wrapper(src_pd()).is_dense()
                && memory_desc_wrapper(diff_dst_pd()).is_dense()
                && memory_desc_wrapper(diff_weights_pd()).is_dense()
                && attr()->has_default_values();
            return ok ? status::success : status::unimplemented;
#else
            return status::unimplemented;
#endif
        }
    };

    gemm_inner_product_bwd_weights_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) {
        execute_backward_weights();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_weights();
    pd_t conf_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

