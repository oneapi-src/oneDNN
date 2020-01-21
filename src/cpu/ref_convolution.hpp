/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
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

#ifndef CPU_REF_CONVOLUTION_HPP
#define CPU_REF_CONVOLUTION_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_convolution_pd.hpp"
#include "ref_eltwise.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <impl::data_type_t src_type, impl::data_type_t wei_type = src_type,
        impl::data_type_t dst_type = src_type,
        impl::data_type_t acc_type = dst_type>
struct ref_convolution_fwd_t : public primitive_impl_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_convolution_fwd_t);

        status_t init() {
            using namespace data_type;

            bool ok = true && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(src_type, wei_type, data_type::undef,
                            dst_type, acc_type)
                    && IMPLICATION(with_bias(),
                            true
                                    && IMPLICATION(src_type == u8,
                                            utils::one_of(bias_md_.data_type,
                                                    f32, s32, s8, u8))
                                    && IMPLICATION(src_type == f32,
                                            bias_md_.data_type == f32))
                    && set_default_formats()
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops)
                    && post_ops_ok();
            return ok ? status::success : status::unimplemented;
        }

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

        bool post_ops_ok() const {
            // to be consistent with other primitives and documentation
            // the number and sequence of post op is limited
            using namespace dnnl::impl::primitive_kind;
            auto const &po = attr()->post_ops_;
            auto is_eltwise
                    = [&](int idx) { return po.entry_[idx].is_eltwise(); };

            switch (po.len_) {
                case 0: return true;
                case 1: return is_eltwise(0) || po.contain(sum, 0);
                case 2:
                    return (po.contain(sum, 0) && is_eltwise(1))
                            || (po.contain(sum, 1) && is_eltwise(0));
                default: return false;
            }
            return false;
        }
    };

    ref_convolution_fwd_t(const pd_t *apd) : primitive_impl_t(apd) {
        for (int idx = 0; idx < dnnl_post_ops::capacity; ++idx)
            eltwises_[idx] = nullptr;
        auto &post_ops = pd()->attr()->post_ops_;
        for (int idx = 0; idx < post_ops.len_; ++idx) {
            const auto &e = post_ops.entry_[idx];
            if (e.kind != dnnl_sum)
                eltwises_[idx] = new ref_eltwise_scalar_fwd_t(e.eltwise);
        }
    }

    ~ref_convolution_fwd_t() {
        for (int idx = 0; idx < dnnl_post_ops::capacity; ++idx)
            if (eltwises_[idx] != nullptr) delete eltwises_[idx];
    }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    ref_eltwise_scalar_fwd_t *eltwises_[dnnl_post_ops::capacity];
};

template <impl::data_type_t diff_src_type, impl::data_type_t wei_type,
        impl::data_type_t diff_dst_type,
        impl::data_type_t acc_type = diff_src_type>
struct ref_convolution_bwd_data_t : public primitive_impl_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        using cpu_convolution_bwd_data_pd_t::cpu_convolution_bwd_data_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_convolution_bwd_data_t);

        status_t init() {
            bool ok = true && desc()->prop_kind == prop_kind::backward_data
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(diff_src_type, wei_type,
                            data_type::undef, diff_dst_type, acc_type)
                    && set_default_formats() && attr()->has_default_values();

            return ok ? status::success : status::unimplemented;
        }

        virtual bool support_bias() const override { return true; }

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    ref_convolution_bwd_data_t(const pd_t *apd) : primitive_impl_t(apd) {}

    typedef typename prec_traits<diff_src_type>::type diff_src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_data(ctx);
        return status::success;
    }

private:
    void execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
};

template <impl::data_type_t src_type, impl::data_type_t diff_wei_type,
        impl::data_type_t diff_dst_type,
        impl::data_type_t acc_type = diff_wei_type>
struct ref_convolution_bwd_weights_t : public primitive_impl_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        using cpu_convolution_bwd_weights_pd_t::
                cpu_convolution_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_convolution_bwd_weights_t);

        status_t init() {
            bool ok = true && desc()->prop_kind == prop_kind::backward_weights
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(src_type, diff_wei_type, diff_wei_type,
                            diff_dst_type, acc_type)
                    && set_default_formats() && attr()->has_default_values();
            return ok ? status::success : status::unimplemented;
        }

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    ref_convolution_bwd_weights_t(const pd_t *apd) : primitive_impl_t(apd) {}

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<diff_wei_type>::type diff_wei_data_t;
    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_weights(ctx);
        return status::success;
    }

private:
    void execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
