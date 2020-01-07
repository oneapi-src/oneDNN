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

#ifndef CPU_REF_LAYER_NORMALIZATION_HPP
#define CPU_REF_LAYER_NORMALIZATION_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_isa_traits.hpp"
#include "cpu_layer_normalization_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t d_type>
struct ref_layer_normalization_fwd_t : public primitive_impl_t {
    struct pd_t : public cpu_layer_normalization_fwd_pd_t {
        pd_t(engine_t *engine, const layer_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const layer_normalization_fwd_pd_t *hint_fwd_pd)
            : cpu_layer_normalization_fwd_pd_t(
                    engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("lnorm_ref:any", ref_layer_normalization_fwd_t);

        status_t init() {
            using namespace data_type;
            bool ok = true && is_fwd()
                    && IMPLICATION(d_type == bf16, mayiuse(avx512_core))
                    && src_md()->data_type == d_type
                    && stat_md()->data_type == f32
                    && IMPLICATION(
                            use_scaleshift(), weights_md()->data_type == f32)
                    && attr()->has_default_values()
                    && set_default_formats_common();
            if (!ok) return status::unimplemented;

            return status::success;
        }
    };

    ref_layer_normalization_fwd_t(const pd_t *apd) : primitive_impl_t(apd) {}

    typedef typename prec_traits<d_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
};

template <data_type_t d_type>
struct ref_layer_normalization_bwd_t : public primitive_impl_t {
    struct pd_t : public cpu_layer_normalization_bwd_pd_t {
        pd_t(engine_t *engine, const layer_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const layer_normalization_fwd_pd_t *hint_fwd_pd)
            : cpu_layer_normalization_bwd_pd_t(
                    engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("lnorm_ref:any", ref_layer_normalization_bwd_t);

        status_t init() {
            using namespace data_type;
            bool ok = true && is_bwd()
                    && IMPLICATION(d_type == bf16, mayiuse(avx512_core))
                    && set_default_formats_common()
                    && utils::everyone_is(d_type, src_md()->data_type,
                            diff_src_md()->data_type)
                    && stat_md()->data_type == f32
                    && IMPLICATION(use_scaleshift(),
                            utils::everyone_is(f32, weights_md()->data_type,
                                    diff_weights_md()->data_type))
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return status::success;
        }
    };

    ref_layer_normalization_bwd_t(const pd_t *apd) : primitive_impl_t(apd) {}

    typedef typename prec_traits<d_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward(ctx);
        return status::success;
    }

private:
    void execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
