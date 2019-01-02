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

#ifndef CPU_REF_LRN_HPP
#define CPU_REF_LRN_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_lrn_pd.hpp"
#include "cpu_primitive.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct ref_lrn_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_lrn_fwd_pd_t {
        using cpu_lrn_fwd_pd_t::cpu_lrn_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_lrn_fwd_t);

        status_t init() {
            bool ok = true
                && is_fwd()
                && src_md()->data_type == data_type
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return status::success;
        }
    };

    ref_lrn_fwd_t(const pd_t *apd): cpu_primitive_t(apd) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        using namespace memory_format;
        switch (pd()->src_md()->format) {
        case nChw16c: execute_forward<nChw16c>(ctx); break;
        case nChw8c: execute_forward<nChw8c>(ctx); break;
        case nchw: execute_forward<nchw>(ctx); break;
        case nhwc: execute_forward<nhwc>(ctx); break;
        // XXX: fix compatibility with 0.14
        // mkldnn_any is used to call ref code for arbitrary format
        default: execute_forward<mkldnn_any>(ctx);
        }
        return status::success;
    }

private:
    template<memory_format_t fmt>
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

template <impl::data_type_t data_type>
struct ref_lrn_bwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_lrn_bwd_pd_t {
        using cpu_lrn_bwd_pd_t::cpu_lrn_bwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_lrn_bwd_t);

        status_t init() {
            using namespace alg_kind;

            bool ok = true
                && !is_fwd()
                && utils::one_of(desc()->alg_kind, lrn_across_channels
                        /*, lrn_within_channel */) // not supported yet
                && utils::everyone_is(data_type,
                        src_md()->data_type,
                        diff_src_md()->data_type)
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return status::success;
        }
    };

    ref_lrn_bwd_t(const pd_t *apd): cpu_primitive_t(apd) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        using namespace memory_format;
        switch (pd()->src_md()->format) {
        case nChw16c: execute_backward<nChw16c>(ctx); break;
        case nChw8c: execute_backward<nChw8c>(ctx); break;
        case nchw: execute_backward<nchw>(ctx); break;
        case nhwc: execute_backward<nhwc>(ctx); break;
        // XXX: fix compatibility with 0.14
        // mkldnn_any is used to call ref code for arbitrary format
        default: execute_backward<mkldnn_any>(ctx);
        }
        return status::success;
    }

private:
    template<memory_format_t fmt>
    void execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
