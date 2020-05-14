/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/cpu_resampling_pd.hpp"
#include "cpu/resampling_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct simple_resampling_fwd_t : public primitive_t {
    struct pd_t : public cpu_resampling_fwd_pd_t {
        using cpu_resampling_fwd_pd_t::cpu_resampling_fwd_pd_t;

        DECLARE_COMMON_PD_T("simple:any", simple_resampling_fwd_t);

        status_t init(engine_t *engine) {
            using namespace format_tag;
            using namespace data_type;
            bool ok = is_fwd() && !has_zero_dim_memory()
                    && utils::everyone_is(
                            data_type, src_md()->data_type, dst_md()->data_type)
                    && platform::has_data_type_support(data_type)
                    && set_default_params() == status::success
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            format_tag_t dat_tag = memory_desc_matches_one_of_tag(*src_md(),
                    nCw8c, nChw8c, nCdhw8c, nCw16c, nChw16c, nCdhw16c, ncw,
                    nchw, ncdhw, nwc, nhwc, ndhwc);
            if (!memory_desc_matches_tag(*dst_md(), dat_tag))
                return status::unimplemented;

            return status::success;
        }
    };

    simple_resampling_fwd_t(const pd_t *apd) : primitive_t(apd) {
        if (pd()->desc()->alg_kind == alg_kind::resampling_nearest)
            interpolate = &simple_resampling_fwd_t::nearest;
        else {
            if (pd()->ndims() == 5)
                interpolate = &simple_resampling_fwd_t::trilinear;
            else if (pd()->ndims() == 4)
                interpolate = &simple_resampling_fwd_t::bilinear;
            else
                interpolate = &simple_resampling_fwd_t::linear;

            fill_coeffs();
        }
        const memory_desc_wrapper src_d(pd()->src_md());
        // non-spatial innermost physical dimension
        nsp_inner_ = src_d.blocking_desc().strides[pd()->ndims() - 1];
        // input/output layout: (nsp0, d, h, w, nsp1)
        nsp_outer_ = src_d.nelems(true)
                / (pd()->ID() * pd()->IH() * pd()->IW() * nsp_inner_);
        stride_d_ = pd()->IH() * pd()->IW() * nsp_inner_;
        stride_h_ = pd()->IW() * nsp_inner_;
        stride_w_ = nsp_inner_;
    }

    ~simple_resampling_fwd_t() {}

    typedef typename prec_traits<data_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void fill_coeffs() {
        using namespace resampling_utils;
        linear_coeffs_.reserve(pd()->OD() + pd()->OH() + pd()->OW());
        for (dim_t od = 0; od < pd()->OD(); od++)
            linear_coeffs_.push_back(
                    linear_coeffs_t(od, pd()->FD(), pd()->ID()));
        for (dim_t oh = 0; oh < pd()->OH(); oh++)
            linear_coeffs_.push_back(
                    linear_coeffs_t(oh, pd()->FH(), pd()->IH()));
        for (dim_t ow = 0; ow < pd()->OW(); ow++)
            linear_coeffs_.push_back(
                    linear_coeffs_t(ow, pd()->FW(), pd()->IW()));
    }

    void nearest(
            const data_t *src, data_t *dst, dim_t od, dim_t oh, dim_t ow) const;
    void linear(
            const data_t *src, data_t *dst, dim_t od, dim_t oh, dim_t ow) const;
    void bilinear(
            const data_t *src, data_t *dst, dim_t od, dim_t oh, dim_t ow) const;
    void trilinear(
            const data_t *src, data_t *dst, dim_t od, dim_t oh, dim_t ow) const;
    void (simple_resampling_fwd_t::*interpolate)(
            const data_t *src, data_t *dst, dim_t od, dim_t oh, dim_t ow) const;
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    dim_t nsp_outer_;
    dim_t stride_d_;
    dim_t stride_h_;
    dim_t stride_w_;
    dim_t nsp_inner_;
    std::vector<resampling_utils::linear_coeffs_t> linear_coeffs_;
};

template <impl::data_type_t data_type>
struct simple_resampling_bwd_t : public primitive_t {
    struct pd_t : public cpu_resampling_bwd_pd_t {
        using cpu_resampling_bwd_pd_t::cpu_resampling_bwd_pd_t;

        DECLARE_COMMON_PD_T("simple:any", simple_resampling_bwd_t);

        status_t init(engine_t *engine) {
            using namespace format_tag;
            using namespace data_type;
            bool ok = !is_fwd() && !has_zero_dim_memory()
                    && utils::everyone_is(data_type, diff_src_md()->data_type,
                            diff_dst_md()->data_type)
                    && platform::has_data_type_support(data_type)
                    && set_default_params() == status::success
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            format_tag_t dat_tag = memory_desc_matches_one_of_tag(
                    *diff_src_md(), nCw8c, nChw8c, nCdhw8c, nCw16c, nChw16c,
                    nCdhw16c, ncw, nchw, ncdhw, nwc, nhwc, ndhwc);
            if (!memory_desc_matches_tag(*diff_dst_md(), dat_tag))
                return status::unimplemented;

            return status::success;
        }
    };

    simple_resampling_bwd_t(const pd_t *apd) : primitive_t(apd) {
        if (pd()->desc()->alg_kind == alg_kind::resampling_nearest)
            interpolate = &simple_resampling_bwd_t::nearest;
        else {
            if (pd()->ndims() == 5)
                interpolate = &simple_resampling_bwd_t::trilinear;
            else if (pd()->ndims() == 4)
                interpolate = &simple_resampling_bwd_t::bilinear;
            else
                interpolate = &simple_resampling_bwd_t::linear;

            fill_coeffs();
            fill_weights();
        }
        const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
        // non-spatial innermost physical dimension
        nsp_inner_ = diff_src_d.blocking_desc().strides[pd()->ndims() - 1];
        // input/output layout: (nsp0, d, h, w, nsp1)
        nsp_outer_ = diff_src_d.nelems(true)
                / (pd()->ID() * pd()->IH() * pd()->IW() * nsp_inner_);
        stride_d_ = pd()->OH() * pd()->OW() * nsp_inner_;
        stride_h_ = pd()->OW() * nsp_inner_;
        stride_w_ = nsp_inner_;
    }

    ~simple_resampling_bwd_t() {}

    typedef typename prec_traits<data_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward(ctx);
        return status::success;
    }

private:
    void fill_coeffs() {
        using namespace resampling_utils;
        bwd_linear_coeffs_.reserve(pd()->ID() + pd()->IH() + pd()->IW());
        for (dim_t id = 0; id < pd()->ID(); id++)
            bwd_linear_coeffs_.push_back(bwd_linear_coeffs_t(
                    id, pd()->FD(), pd()->ID(), pd()->OD()));
        for (dim_t ih = 0; ih < pd()->IH(); ih++)
            bwd_linear_coeffs_.push_back(bwd_linear_coeffs_t(
                    ih, pd()->FH(), pd()->IH(), pd()->OH()));
        for (dim_t iw = 0; iw < pd()->IW(); iw++)
            bwd_linear_coeffs_.push_back(bwd_linear_coeffs_t(
                    iw, pd()->FW(), pd()->IW(), pd()->OW()));
    }

    void fill_weights() {
        using namespace resampling_utils;
        bwd_linear_weights_.reserve(2 * (pd()->OD() + pd()->OH() + pd()->OW()));
        for (dim_t od = 0; od < pd()->OD(); od++) {
            bwd_linear_weights_.push_back(linear_weight(0, od, pd()->FD()));
            bwd_linear_weights_.push_back(linear_weight(1, od, pd()->FD()));
        }
        for (dim_t oh = 0; oh < pd()->OH(); oh++) {
            bwd_linear_weights_.push_back(linear_weight(0, oh, pd()->FH()));
            bwd_linear_weights_.push_back(linear_weight(1, oh, pd()->FH()));
        }
        for (dim_t ow = 0; ow < pd()->OW(); ow++) {
            bwd_linear_weights_.push_back(linear_weight(0, ow, pd()->FW()));
            bwd_linear_weights_.push_back(linear_weight(1, ow, pd()->FW()));
        }
    }

    void nearest(data_t *diff_src, const data_t *diff_dst, dim_t id, dim_t ih,
            dim_t iw) const;
    void linear(data_t *diff_src, const data_t *diff_dst, dim_t id, dim_t ih,
            dim_t iw) const;
    void bilinear(data_t *diff_src, const data_t *diff_dst, dim_t id, dim_t ih,
            dim_t iw) const;
    void trilinear(data_t *diff_src, const data_t *diff_dst, dim_t id, dim_t ih,
            dim_t iw) const;
    void (simple_resampling_bwd_t::*interpolate)(data_t *diff_src,
            const data_t *diff_dst, dim_t id, dim_t ih, dim_t iw) const;
    void execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    dim_t nsp_outer_;
    dim_t stride_d_;
    dim_t stride_h_;
    dim_t stride_w_;
    dim_t nsp_inner_;
    std::vector<resampling_utils::bwd_linear_coeffs_t> bwd_linear_coeffs_;
    std::vector<float> bwd_linear_weights_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
