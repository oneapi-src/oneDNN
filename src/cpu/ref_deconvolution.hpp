/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef CPU_REF_DECONVOLUTION_HPP
#define CPU_REF_DECONVOLUTION_HPP

#include <assert.h>
#include <string.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_iterator.hpp"
#include "common/stream.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/cpu_deconvolution_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

static status_t weights_axes_permutation(
        memory_desc_t *o_md, const memory_desc_t *i_md, bool with_groups) {
    int perm[DNNL_MAX_NDIMS] {}; // deconv to conv weight permutation
    for (int d = 0; d < DNNL_MAX_NDIMS; ++d)
        perm[d] = d;
    nstl::swap(perm[0 + with_groups], perm[1 + with_groups]);

    return dnnl_memory_desc_permute_axes(o_md, i_md, perm);
}

static status_t conv_descr_create(
        const deconvolution_desc_t *dd, convolution_desc_t *cd) {
    using namespace prop_kind;
    alg_kind_t alg_kind = dd->alg_kind == alg_kind::deconvolution_direct
            ? alg_kind::convolution_direct
            : alg_kind::convolution_winograd;

    const memory_desc_t *src_md, *dst_md, *d_weights_d;
    prop_kind_t prop_kind;
    if (utils::one_of(dd->prop_kind, forward_training, forward_inference)) {
        prop_kind = backward_data;
        src_md = &dd->dst_desc;
        dst_md = &dd->src_desc;
        d_weights_d = &dd->weights_desc;
    } else if (dd->prop_kind == backward_data) {
        prop_kind = forward_training;
        src_md = &dd->diff_dst_desc;
        dst_md = &dd->diff_src_desc;
        d_weights_d = &dd->weights_desc;
    } else {
        prop_kind = dd->prop_kind;
        src_md = &dd->diff_dst_desc;
        dst_md = &dd->src_desc;
        d_weights_d = &dd->diff_weights_desc;
    }

    /* create weights desc for convolution */
    memory_desc_t c_weights_d;
    const bool with_groups = d_weights_d->ndims == src_md->ndims + 1;
    CHECK(weights_axes_permutation(&c_weights_d, d_weights_d, with_groups));

    return conv_desc_init(cd, prop_kind, alg_kind, src_md, &c_weights_d,
            prop_kind != backward_weights ? &dd->bias_desc : nullptr, dst_md,
            dd->strides, dd->dilates, dd->padding[0], dd->padding[1]);
}

struct ref_deconvolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_deconvolution_fwd_pd_t {
        pd_t(const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_fwd_pd_t(other)
            , conv_pd_(other.conv_pd_->clone())
            , conv_supports_bias_(other.conv_supports_bias_)
            , dst_tag_(other.dst_tag_) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(conv_pd_->name(), ref_deconvolution_fwd_t);

        status_t init_convolution(engine_t *engine) {
            using namespace format_tag;
            using namespace data_type;

            primitive_attr_t conv_attr(*attr());
            if (!conv_attr.is_initialized()) return status::out_of_memory;
            conv_attr.set_scratchpad_mode(scratchpad_mode::user);

            convolution_desc_t cd;
            CHECK(conv_descr_create(desc(), &cd));
            dnnl_primitive_desc_iterator it(
                    engine, (op_desc_t *)&cd, &conv_attr, nullptr);
            if (!it.is_initialized()) return status::out_of_memory;

            while (++it != it.end()) {
                conv_pd_.reset(it.fetch_once());
                conv_supports_bias_
                        = utils::downcast<cpu_convolution_bwd_data_pd_t *>(
                                conv_pd_.get())
                                  ->support_bias();
                bool bf16_tag_supported = memory_desc_matches_one_of_tag(
                        *conv_pd_->diff_src_md(),
                        utils::pick(ndims() - 3, ncw, nchw, ncdhw),
                        utils::pick(ndims() - 3, nwc, nhwc, ndhwc),
                        utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c));
                bool ref_deconv_supports_bias
                        = desc()->accum_data_type == data_type::f32
                        && utils::one_of(desc()->dst_desc.data_type, f32, bf16)
                        && IMPLICATION(desc()->src_desc.data_type == bf16,
                                bf16_tag_supported);
                bool ok = conv_pd_->weights_md()->extra.flags == 0
                        /* deconv reference code can process only f32 bias */
                        && IMPLICATION(with_bias(),
                                conv_supports_bias_
                                        || ref_deconv_supports_bias);
                if (ok) return status::success;
            }
            return status::unimplemented;
        }

        status_t init(engine_t *engine) {
            using namespace format_tag;

            bool ok = is_fwd()
                    && utils::one_of(desc()->alg_kind,
                            alg_kind::deconvolution_direct,
                            alg_kind::deconvolution_winograd)
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            CHECK(init_convolution(engine));

            if (weights_md_.format_kind == format_kind::any)
                CHECK(weights_axes_permutation(
                        &weights_md_, conv_pd_->weights_md(), with_groups()));
            if (src_md_.format_kind == format_kind::any)
                src_md_ = *conv_pd_->diff_dst_md();
            if (dst_md_.format_kind == format_kind::any)
                dst_md_ = *conv_pd_->diff_src_md();
            if (bias_md_.format_kind == format_kind::any)
                CHECK(memory_desc_init_by_tag(bias_md_, x));

            dst_tag_ = memory_desc_matches_one_of_tag(dst_md_,
                    utils::pick(ndims() - 3, ncw, nchw, ncdhw),
                    utils::pick(ndims() - 3, nwc, nhwc, ndhwc),
                    utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c),
                    utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c));

            init_scratchpad();
            return status::success;
        }

        std::unique_ptr<primitive_desc_t> conv_pd_;
        bool conv_supports_bias_;
        format_tag_t dst_tag_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    conv_pd_->scratchpad_registry());
        }
    };

    ref_deconvolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        return pd()->conv_pd_->create_primitive(conv_p_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    void compute_fwd_bias(float *dst, const float *bias) const;
    template <data_type_t dst_type, data_type_t bia_type>
    void compute_fwd_bias_ncdhw(typename prec_traits<dst_type>::type *dst,
            const typename prec_traits<bia_type>::type *bias) const;

    template <data_type_t dst_type, data_type_t bia_type>
    void compute_fwd_bias_ndhwc(typename prec_traits<dst_type>::type *dst,
            const typename prec_traits<bia_type>::type *bias) const;

    template <data_type_t dst_type, data_type_t bia_type, int blksize>
    void compute_fwd_bias_nCdhwXc(typename prec_traits<dst_type>::type *dst,
            const typename prec_traits<bia_type>::type *bias) const;

    template <data_type_t dst_type, data_type_t bia_type>
    void compute_bias(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> conv_p_;
};

struct ref_deconvolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_deconvolution_bwd_data_pd_t {
        pd_t(const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_bwd_data_pd_t(other)
            , conv_pd_(other.conv_pd_->clone()) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(conv_pd_->name(), ref_deconvolution_bwd_data_t);

        status_t init_convolution(engine_t *engine) {
            using namespace types;

            convolution_desc_t cd;
            status_t status = conv_descr_create(desc(), &cd);
            if (status != status::success) return status;
            primitive_attr_t conv_attr(*attr());
            if (!conv_attr.is_initialized()) return status::out_of_memory;
            conv_attr.set_scratchpad_mode(scratchpad_mode::user);

            dnnl_primitive_desc_iterator it(
                    engine, (op_desc_t *)&cd, &conv_attr, nullptr);
            if (!it.is_initialized()) return status::out_of_memory;
            while (++it != it.end()) {
                conv_pd_.reset(it.fetch_once());
                if (conv_pd_->weights_md()->extra.flags == 0)
                    return status::success;
            }

            return status::unimplemented;
        }

        status_t init(engine_t *engine) {
            using namespace data_type;
            auto dsrc_type = desc()->diff_src_desc.data_type;
            auto wei_type = desc()->weights_desc.data_type;
            auto ddst_type = desc()->diff_dst_desc.data_type;
            bool ok = true && desc()->prop_kind == prop_kind::backward_data
                    && (utils::everyone_is(f32, dsrc_type, wei_type, ddst_type)
                            || (utils::one_of(dsrc_type, f32, bf16)
                                    && utils::everyone_is(
                                            bf16, wei_type, ddst_type)))
                    && utils::one_of(desc()->alg_kind,
                            alg_kind::deconvolution_direct,
                            alg_kind::deconvolution_winograd)
                    && attr()->has_default_values();

            if (ok) {
                CHECK(init_convolution(engine));
                if (weights_md_.format_kind == format_kind::any)
                    CHECK(weights_axes_permutation(&weights_md_,
                            conv_pd_->weights_md(), with_groups()));
                if (diff_src_md_.format_kind == format_kind::any)
                    diff_src_md_ = *conv_pd_->dst_md();
                if (diff_dst_md_.format_kind == format_kind::any)
                    diff_dst_md_ = *conv_pd_->src_md();
                init_scratchpad();
                return status::success;
            }

            return status::unimplemented;
        }

        std::unique_ptr<primitive_desc_t> conv_pd_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    conv_pd_->scratchpad_registry());
        }
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    ref_deconvolution_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        return pd()->conv_pd_->create_primitive(conv_p_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> conv_p_;
};

struct ref_deconvolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_deconvolution_bwd_weights_pd_t {
        pd_t(const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_bwd_weights_pd_t(other)
            , conv_pd_(other.conv_pd_->clone())
            , dst_tag_(other.dst_tag_) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(conv_pd_->name(), ref_deconvolution_bwd_weights_t);

        status_t init_convolution(engine_t *engine) {
            using namespace types;
            using namespace format_tag;

            convolution_desc_t cd;
            status_t status = conv_descr_create(desc(), &cd);
            if (status != status::success) return status;
            primitive_attr_t conv_attr(*attr());
            if (!conv_attr.is_initialized()) return status::out_of_memory;
            conv_attr.set_scratchpad_mode(scratchpad_mode::user);

            dnnl_primitive_desc_iterator it(
                    engine, (op_desc_t *)&cd, &conv_attr, nullptr);
            if (!it.is_initialized()) return status::out_of_memory;
            while (++it != it.end()) {
                conv_pd_.reset(it.fetch_once());
                bool bf16_ref_deconv_supports_bias = IMPLICATION(with_bias()
                                && desc()->src_desc.data_type
                                        == data_type::bf16,
                        memory_desc_matches_one_of_tag(*conv_pd_->src_md(),
                                utils::pick(ndims() - 3, ncw, nchw, ncdhw),
                                utils::pick(ndims() - 3, nwc, nhwc, ndhwc),
                                utils::pick(ndims() - 3, nCw16c, nChw16c,
                                        nCdhw16c)));
                if (conv_pd_->diff_weights_md()->extra.flags == 0
                        && bf16_ref_deconv_supports_bias) {
                    return status::success;
                }
            }
            return status::unimplemented;
        }

        status_t init(engine_t *engine) {
            using namespace format_tag;
            using namespace data_type;
            auto src_type = desc()->src_desc.data_type;
            auto dwei_type = desc()->diff_weights_desc.data_type;
            auto ddst_type = desc()->diff_dst_desc.data_type;
            bool ok = true && desc()->prop_kind == prop_kind::backward_weights
                    && (utils::everyone_is(f32, src_type, dwei_type, ddst_type)
                            || (utils::one_of(dwei_type, f32, bf16)
                                    && utils::everyone_is(
                                            bf16, src_type, ddst_type)))
                    && utils::one_of(desc()->alg_kind,
                            alg_kind::deconvolution_direct,
                            alg_kind::deconvolution_winograd)
                    && attr()->has_default_values();

            if (ok) {
                CHECK(init_convolution(engine));
                if (diff_weights_md_.format_kind == format_kind::any)
                    CHECK(weights_axes_permutation(&diff_weights_md_,
                            conv_pd_->diff_weights_md(), with_groups()));
                if (src_md_.format_kind == format_kind::any)
                    src_md_ = *conv_pd_->diff_dst_md();
                if (diff_dst_md_.format_kind == format_kind::any)
                    diff_dst_md_ = *conv_pd_->src_md();
                if (diff_bias_md_.format_kind == format_kind::any)
                    CHECK(memory_desc_init_by_tag(diff_bias_md_, x));

                dst_tag_ = memory_desc_matches_one_of_tag(diff_dst_md_,
                        utils::pick(ndims() - 3, ncw, nchw, ncdhw),
                        utils::pick(ndims() - 3, nwc, nhwc, ndhwc),
                        utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c),
                        utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c));
                init_scratchpad();
                return status::success;
            }

            return status::unimplemented;
        }

        std::unique_ptr<primitive_desc_t> conv_pd_;
        format_tag_t dst_tag_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    conv_pd_->scratchpad_registry());
        }
    };

    ref_deconvolution_bwd_weights_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        return pd()->conv_pd_->create_primitive(conv_p_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    void compute_bwd_bias(float *diff_bias, const float *diff_dst) const;

    template <data_type_t dbia_type, data_type_t ddst_type>
    void compute_bwd_bias_ncdhw(
            typename prec_traits<dbia_type>::type *diff_bias,
            const typename prec_traits<ddst_type>::type *diff_dst) const;

    template <data_type_t dbia_type, data_type_t ddst_type>
    void compute_bwd_bias_ndhwc(
            typename prec_traits<dbia_type>::type *diff_bias,
            const typename prec_traits<ddst_type>::type *diff_dst) const;

    template <data_type_t dbia_type, data_type_t ddst_type, int blksize>
    void compute_bwd_bias_nCdhwXc(
            typename prec_traits<dbia_type>::type *diff_bias,
            const typename prec_traits<ddst_type>::type *diff_dst) const;

    template <data_type_t dbia_type, data_type_t ddst_type>
    void compute_bias(const exec_ctx_t &ctx) const;
    std::shared_ptr<primitive_t> conv_p_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
