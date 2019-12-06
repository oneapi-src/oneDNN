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

#ifndef CPU_SIMPLE_RESAMPLING_HPP
#define CPU_SIMPLE_RESAMPLING_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_isa_traits.hpp"

#include "cpu_resampling_pd.hpp"
#include "resampling_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct simple_resampling_fwd_t : public primitive_impl_t {
    struct pd_t : public cpu_resampling_fwd_pd_t {
        using cpu_resampling_fwd_pd_t::cpu_resampling_fwd_pd_t;

        DECLARE_COMMON_PD_T("simple:any", simple_resampling_fwd_t);

        status_t init() {
            using namespace format_tag;
            using namespace data_type;
            bool ok = is_fwd() && !has_zero_dim_memory()
                    && utils::everyone_is(
                            data_type, src_md()->data_type, dst_md()->data_type)
                    && IMPLICATION(data_type == bf16, mayiuse(avx512_core))
                    && set_default_params() == status::success
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            format_tag_t dat_tag = memory_desc_matches_one_of_tag(*src_md(),
                    nChw8c, nCdhw8c, nChw16c, nCdhw16c, ncw, nchw, ncdhw, nwc,
                    nhwc, ndhwc);
            if (!memory_desc_matches_tag(*dst_md(), dat_tag))
                return status::unimplemented;

            return status::success;
        }
    };

    simple_resampling_fwd_t(const pd_t *apd) : primitive_impl_t(apd) {
        if (pd()->desc()->alg_kind == alg_kind::resampling_nearest)
            interpolate = &simple_resampling_fwd_t::nearest;
        else {
            if (pd()->ndims() == 5)
                interpolate = &simple_resampling_fwd_t::trilinear;
            else if (pd()->ndims() == 4)
                interpolate = &simple_resampling_fwd_t::bilinear;
            else
                interpolate = &simple_resampling_fwd_t::linear;
        }
        const memory_desc_wrapper src_d(pd()->src_md());
        // non-spatial innermost physical dimension
        nsp_inner_ = src_d.blocking_desc().strides[pd()->ndims() - 1];
    }

    ~simple_resampling_fwd_t() {}

    typedef typename prec_traits<data_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    dim_t nsp_inner_;
    void nearest(const float *src, float *dst, dim_t stride_d, dim_t stride_h,
            dim_t stride_w, dim_t od, dim_t oh, dim_t ow) const;
    void linear(const float *src, float *dst, dim_t stride_d, dim_t stride_h,
            dim_t stride_w, dim_t od, dim_t oh, dim_t ow) const;
    void bilinear(const float *src, float *dst, dim_t stride_d, dim_t stride_h,
            dim_t stride_w, dim_t od, dim_t oh, dim_t ow) const;
    void trilinear(const float *src, float *dst, dim_t stride_d, dim_t stride_h,
            dim_t stride_w, dim_t od, dim_t oh, dim_t ow) const;
    void (simple_resampling_fwd_t::*interpolate)(const float *src, float *dst,
            dim_t stride_d, dim_t stride_h, dim_t stride_w, dim_t od, dim_t oh,
            dim_t ow) const;

    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    void execute_forward(const exec_ctx_t &ctx) const;
};

template <impl::data_type_t data_type>
struct simple_resampling_bwd_t : public primitive_impl_t {
    struct pd_t : public cpu_resampling_bwd_pd_t {
        using cpu_resampling_bwd_pd_t::cpu_resampling_bwd_pd_t;

        DECLARE_COMMON_PD_T("simple:any", simple_resampling_bwd_t);

        status_t init() {
            using namespace format_tag;
            using namespace data_type;
            bool ok = !is_fwd() && !has_zero_dim_memory()
                    && utils::everyone_is(data_type, diff_src_md()->data_type,
                            diff_dst_md()->data_type)
                    && IMPLICATION(data_type == bf16, mayiuse(avx512_core))
                    && set_default_params() == status::success
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            format_tag_t dat_tag = memory_desc_matches_one_of_tag(
                    *diff_src_md(), nChw8c, nCdhw8c, nChw16c, nCdhw16c, ncw,
                    nchw, ncdhw, nwc, nhwc, ndhwc);
            if (!memory_desc_matches_tag(*diff_dst_md(), dat_tag))
                return status::unimplemented;

            return status::success;
        }
    };

    simple_resampling_bwd_t(const pd_t *apd) : primitive_impl_t(apd) {
        if (pd()->desc()->alg_kind == alg_kind::resampling_nearest)
            interpolate = &simple_resampling_bwd_t::nearest;
        else {
            if (pd()->ndims() == 5)
                interpolate = &simple_resampling_bwd_t::trilinear;
            else if (pd()->ndims() == 4)
                interpolate = &simple_resampling_bwd_t::bilinear;
            else
                interpolate = &simple_resampling_bwd_t::linear;
        }
        const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
        // non-spatial innermost physical dimension
        nsp_inner_ = diff_src_d.blocking_desc().strides[pd()->ndims() - 1];
    }

    ~simple_resampling_bwd_t() {}

    typedef typename prec_traits<data_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward(ctx);
        return status::success;
    }

private:
    dim_t nsp_inner_;
    void nearest(float *diff_src, const float *diff_dst, dim_t stride_d,
            dim_t stride_h, dim_t stride_w, dim_t id, dim_t ih, dim_t iw) const;
    void linear(float *diff_src, const float *diff_dst, dim_t stride_d,
            dim_t stride_h, dim_t stride_w, dim_t id, dim_t ih, dim_t iw) const;
    void bilinear(float *diff_src, const float *diff_dst, dim_t stride_d,
            dim_t stride_h, dim_t stride_w, dim_t id, dim_t ih, dim_t iw) const;
    void trilinear(float *diff_src, const float *diff_dst, dim_t stride_d,
            dim_t stride_h, dim_t stride_w, dim_t id, dim_t ih, dim_t iw) const;
    void (simple_resampling_bwd_t::*interpolate)(float *diff_src,
            const float *diff_dst, dim_t stride_d, dim_t stride_h,
            dim_t stride_w, dim_t id, dim_t ih, dim_t iw) const;

    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    void execute_backward(const exec_ctx_t &ctx) const;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
