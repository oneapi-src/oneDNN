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

#include "c_types_map.hpp"
#include "primitive_iterator.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_deconvolution_pd.hpp"

#if DNNL_VERBOSE_EXTRA
#include "consistency.hpp"
#include "dnnl_debug.h"
#define AND_(...) SCHKV(ok, __VA_ARGS__)
#endif

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

struct ref_deconvolution_fwd_t : public primitive_impl_t {
    struct pd_t : public cpu_deconvolution_fwd_pd_t {
        pd_t(engine_t *engine, const deconvolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr) {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_fwd_pd_t(other)
            , conv_pd_(other.conv_pd_->clone())
            , conv_supports_bias_(other.conv_supports_bias_)
            , dst_tag_(other.dst_tag_) {}

        pd_t &operator=(const pd_t &other) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(other);
            cpu_deconvolution_fwd_pd_t::operator=(other);
            delete conv_pd_;
            conv_pd_ = other.conv_pd_->clone();
            conv_supports_bias_ = other.conv_supports_bias_;
            dst_tag_ = other.dst_tag_;
            return *this;
        }

        ~pd_t() { delete conv_pd_; }

        DECLARE_COMMON_PD_T(conv_pd_->name(), ref_deconvolution_fwd_t);

        status_t init_convolution() {
            using namespace format_tag;
            using namespace data_type;

            convolution_desc_t cd;
            CHECK(conv_descr_create(desc(), &cd));

            dnnl_primitive_desc_iterator it(
                    engine_, (op_desc_t *)&cd, &attr_, nullptr);
            while (++it != it.end()) {
                conv_pd_ = it.fetch_once();
                conv_supports_bias_
                        = static_cast<cpu_convolution_bwd_data_pd_t *>(conv_pd_)
                                  ->support_bias();
                bool ref_deconv_supports_bias = true
                        && desc()->accum_data_type == data_type::f32
                        && utils::one_of(desc()->dst_desc.data_type, f32, bf16)
                        && IMPLICATION(desc()->src_desc.data_type == bf16,
                                memory_desc_matches_one_of_tag(
                                        *conv_pd_->diff_src_md(),
                                        utils::pick(
                                                ndims() - 3, ncw, nchw, ncdhw),
                                        utils::pick(ndims() - 3, nCw16c,
                                                nChw16c, nCdhw16c)));
                bool ok = true
                        && conv_pd_->weights_md()->extra.flags == 0
                        /* deconv reference code can process only f32 bias */
                        && IMPLICATION(with_bias(),
                                conv_supports_bias_
                                        || ref_deconv_supports_bias);
                if (ok) return status::success;

                delete conv_pd_;
            }
            conv_pd_ = nullptr;
            return status::unimplemented;
        }

        status_t init() {
            using namespace format_tag;
#if !DNNL_VERBOSE_EXTRA
            bool ok = true && is_fwd()
                    && utils::one_of(desc()->alg_kind,
                            alg_kind::deconvolution_direct,
                            alg_kind::deconvolution_winograd)
                    && attr()->has_default_values();
#else // track reason we could not use
            Consistency ok("\ndeconvolution_fwd bad init:");
            AND_(is_fwd());
            AND_(utils::one_of(desc()->alg_kind, alg_kind::deconvolution_direct,
                    alg_kind::deconvolution_winograd));
            AND_(attr()->has_default_values());
#endif

#if !DNNL_VERBOSE_EXTRA
            if (ok) {
                CHECK(init_convolution());
                if (weights_md_.format_kind == format_kind::any)
                    CHECK(weights_axes_permutation(&weights_md_,
                            conv_pd_->weights_md(), with_groups()));
                if (src_md_.format_kind == format_kind::any)
                    src_md_ = *conv_pd_->diff_dst_md();
                if (dst_md_.format_kind == format_kind::any)
                    dst_md_ = *conv_pd_->diff_src_md();
                if (bias_md_.format_kind == format_kind::any)
                    CHECK(memory_desc_init_by_tag(bias_md_, x));

                dst_tag_ = memory_desc_matches_one_of_tag(dst_md_,
                        utils::pick(ndims() - 3, ncw, nchw, ncdhw),
                        utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c),
                        utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c));

                return status::success;
            }

            return status::unimplemented;
#else // debug
            if (ok) {
                OK_CHECK(init_convolution());
                if (weights_md_.format_kind == format_kind::any) {
                    OK_CHECK(weights_axes_permutation(&weights_md_,
                            conv_pd_->weights_md(), with_groups()));
                    // weights_md_ = desc_.weights_desc; // is this now done by weights_axes_permutation ? XXX CHECKME
                }
                if (src_md_.format_kind == format_kind::any)
                    src_md_ = *conv_pd_->diff_dst_md();
                if (dst_md_.format_kind == format_kind::any)
                    dst_md_ = *conv_pd_->diff_src_md();
                if (bias_md_.format_kind == format_kind::any)
                    OK_CHECK(memory_desc_init_by_tag(bias_md_, x));

                dst_tag_ = memory_desc_matches_one_of_tag(dst_md_,
                        utils::pick(ndims() - 3, ncw, nchw, ncdhw),
                        utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c),
                        utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c));
            }
            if (ok) return status::success;

            return status::unimplemented;
#endif
        }

        virtual void init_scratchpad_md() override {
            scratchpad_md_ = *conv_pd_->scratchpad_md();
        }

        primitive_desc_t *conv_pd_;
        bool conv_supports_bias_;
        format_tag_t dst_tag_;
    };

    ref_deconvolution_fwd_t(const pd_t *apd) : primitive_impl_t(apd) {
        pd()->conv_pd_->create_primitive((primitive_t **)&conv_p_);
    }
    ~ref_deconvolution_fwd_t() { delete conv_p_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[DNNL_ARG_DIFF_DST] = args.at(DNNL_ARG_SRC);
        conv_args[DNNL_ARG_WEIGHTS] = args.at(DNNL_ARG_WEIGHTS);
        if (pd()->with_bias() && pd()->conv_supports_bias_)
            conv_args[DNNL_ARG_BIAS] = args.at(DNNL_ARG_BIAS);
        conv_args[DNNL_ARG_DIFF_SRC] = args.at(DNNL_ARG_DST);
        if (!types::is_zero_md(pd()->scratchpad_md()))
            conv_args[DNNL_ARG_SCRATCHPAD] = args.at(DNNL_ARG_SCRATCHPAD);
        exec_ctx_t conv_ctx(ctx.stream(), std::move(conv_args));

        conv_p_->execute(conv_ctx);

        if (pd()->with_bias() && !pd()->conv_supports_bias_) {
            using namespace data_type;

            auto dst_type = pd()->dst_md()->data_type;
            auto bia_type = pd()->weights_md(1)->data_type;
            if (utils::everyone_is(f32, dst_type, bia_type))
                compute_bias<f32, f32>(ctx);
            else if (utils::everyone_is(bf16, dst_type, bia_type))
                compute_bias<bf16, bf16>(ctx);
            else if (dst_type == f32 && bia_type == bf16)
                compute_bias<f32, bf16>(ctx);
            else if (dst_type == bf16 && bia_type == f32)
                compute_bias<bf16, f32>(ctx);
        }
        return status::success;
    }

private:
    void compute_fwd_bias(float *dst, const float *bias) const;
    template <data_type_t dst_type, data_type_t bia_type>
    void compute_fwd_bias_ncdhw(typename prec_traits<dst_type>::type *dst,
            const typename prec_traits<bia_type>::type *bias) const;

    template <data_type_t dst_type, data_type_t bia_type, int blksize>
    void compute_fwd_bias_nCdhwXc(typename prec_traits<dst_type>::type *dst,
            const typename prec_traits<bia_type>::type *bias) const;

    template <data_type_t dst_type, data_type_t bia_type>
    void compute_bias(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    primitive_t *conv_p_;
};

struct ref_deconvolution_bwd_data_t : public primitive_impl_t {
    struct pd_t : public cpu_deconvolution_bwd_data_pd_t {
        pd_t(engine_t *engine, const deconvolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr) {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_bwd_data_pd_t(other)
            , conv_pd_(other.conv_pd_->clone()) {}

        pd_t &operator=(const pd_t &other) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(other);
            cpu_deconvolution_bwd_data_pd_t::operator=(other);
            delete conv_pd_;
            conv_pd_ = other.conv_pd_->clone();
            return *this;
        }

        ~pd_t() { delete conv_pd_; }

        DECLARE_COMMON_PD_T(conv_pd_->name(), ref_deconvolution_bwd_data_t);

        status_t init_convolution() {
            using namespace types;

            convolution_desc_t cd;
            status_t status = conv_descr_create(desc(), &cd);
            if (status != status::success) return status;

            dnnl_primitive_desc_iterator it(
                    engine_, (op_desc_t *)&cd, &attr_, nullptr);
            while (++it != it.end()) {
                conv_pd_ = it.fetch_once();
                if (conv_pd_->weights_md()->extra.flags == 0)
                    return status::success;
                delete conv_pd_;
            }
            return status::unimplemented;
        }

        status_t init() {
            using namespace data_type;
            auto dsrc_type = desc()->diff_src_desc.data_type;
            auto wei_type = desc()->weights_desc.data_type;
            auto ddst_type = desc()->diff_dst_desc.data_type;
#if !DNNL_VERBOSE_EXTRA
            bool ok = true && desc()->prop_kind == prop_kind::backward_data
                    && (utils::everyone_is(f32, dsrc_type, wei_type, ddst_type)
                            || (utils::one_of(dsrc_type, f32, bf16)
                                    && utils::everyone_is(
                                            bf16, wei_type, ddst_type)))
                    && utils::one_of(desc()->alg_kind,
                            alg_kind::deconvolution_direct,
                            alg_kind::deconvolution_winograd)
                    && attr()->has_default_values();
#else // debug
            Consistency ok("\ndeconvolution_bwd_data bad init:");
            AND_(desc()->prop_kind == prop_kind::backward_data);
            AND_(utils::everyone_is(f32, dsrc_type, wei_type, ddst_type)
                    || (utils::one_of(dsrc_type, f32, bf16)
                            && utils::everyone_is(bf16, wei_type, ddst_type)));
            AND_(utils::one_of(desc()->alg_kind, alg_kind::deconvolution_direct,
                    alg_kind::deconvolution_winograd));
            AND_(attr()->has_default_values());
#endif

#if !DNNL_VERBOSE_EXTRA
            if (ok) {
                CHECK(init_convolution());
                if (weights_md_.format_kind == format_kind::any)
                    CHECK(weights_axes_permutation(&weights_md_,
                            conv_pd_->weights_md(), with_groups()));
                if (diff_src_md_.format_kind == format_kind::any)
                    diff_src_md_ = *conv_pd_->dst_md();
                if (diff_dst_md_.format_kind == format_kind::any)
                    diff_dst_md_ = *conv_pd_->src_md();

                return status::success;
            }
#else // debug
            if (ok) {
                OK_CHECK(init_convolution());
                if (weights_md_.format_kind == format_kind::any) {
                    OK_CHECK(weights_axes_permutation(&weights_md_,
                            conv_pd_->weights_md(), with_groups()));
                    // removed? XXX weights_md_ = desc_.weights_desc;
                }
                if (diff_src_md_.format_kind == format_kind::any)
                    diff_src_md_ = *conv_pd_->dst_md();
                if (diff_dst_md_.format_kind == format_kind::any)
                    diff_dst_md_ = *conv_pd_->src_md();
            }
            if (ok) return status::success;
#endif
            return status::unimplemented;
        }

        virtual void init_scratchpad_md() override {
            scratchpad_md_ = *conv_pd_->scratchpad_md();
        }

        primitive_desc_t *conv_pd_;
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    ref_deconvolution_bwd_data_t(const pd_t *apd) : primitive_impl_t(apd) {
        pd()->conv_pd_->create_primitive((primitive_t **)&conv_p_);
    }
    ~ref_deconvolution_bwd_data_t() { delete conv_p_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[DNNL_ARG_SRC] = args.at(DNNL_ARG_DIFF_DST);
        conv_args[DNNL_ARG_WEIGHTS] = args.at(DNNL_ARG_WEIGHTS);
        conv_args[DNNL_ARG_DST] = args.at(DNNL_ARG_DIFF_SRC);
        if (!types::is_zero_md(pd()->scratchpad_md()))
            conv_args[DNNL_ARG_SCRATCHPAD] = args.at(DNNL_ARG_SCRATCHPAD);
        exec_ctx_t conv_ctx(ctx.stream(), std::move(conv_args));

        conv_p_->execute(conv_ctx);
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    primitive_t *conv_p_;
};

struct ref_deconvolution_bwd_weights_t : public primitive_impl_t {
    struct pd_t : public cpu_deconvolution_bwd_weights_pd_t {
        pd_t(engine_t *engine, const deconvolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_bwd_weights_pd_t(
                    engine, adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr) {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_bwd_weights_pd_t(other)
            , conv_pd_(other.conv_pd_->clone())
            , dst_tag_(other.dst_tag_) {}

        pd_t &operator=(const pd_t &other) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(other);
            cpu_deconvolution_bwd_weights_pd_t::operator=(other);
            delete conv_pd_;
            conv_pd_ = other.conv_pd_->clone();
            return *this;
        }

        ~pd_t() { delete conv_pd_; }

        DECLARE_COMMON_PD_T(conv_pd_->name(), ref_deconvolution_bwd_weights_t);

        status_t init_convolution() {
            using namespace types;
            using namespace format_tag;

            convolution_desc_t cd;
            status_t status = conv_descr_create(desc(), &cd);
            if (status != status::success) return status;

            dnnl_primitive_desc_iterator it(
                    engine_, (op_desc_t *)&cd, &attr_, nullptr);
            while (++it != it.end()) {
                conv_pd_ = it.fetch_once();
                bool bf16_ref_deconv_supports_bias = IMPLICATION(with_bias()
                                && desc()->src_desc.data_type
                                        == data_type::bf16,
                        memory_desc_matches_one_of_tag(*conv_pd_->src_md(),
                                utils::pick(ndims() - 3, ncw, nchw, ncdhw),
                                utils::pick(ndims() - 3, nCw16c, nChw16c,
                                        nCdhw16c)));
                if (conv_pd_->diff_weights_md()->extra.flags == 0
                        && bf16_ref_deconv_supports_bias)
                    return status::success;
                delete conv_pd_;
            }
            return status::unimplemented;
        }

        status_t init() {
            using namespace format_tag;
            using namespace data_type;
            auto src_type = desc()->src_desc.data_type;
            auto dwei_type = desc()->diff_weights_desc.data_type;
            auto ddst_type = desc()->diff_dst_desc.data_type;
#if !DNNL_VERBOSE_EXTRA
            bool ok = true && desc()->prop_kind == prop_kind::backward_weights
                    && (utils::everyone_is(f32, src_type, dwei_type, ddst_type)
                            || (utils::one_of(dwei_type, f32, bf16)
                                    && utils::everyone_is(
                                            bf16, src_type, ddst_type)))
                    && utils::one_of(desc()->alg_kind,
                            alg_kind::deconvolution_direct,
                            alg_kind::deconvolution_winograd)
                    && attr()->has_default_values();
#else // debug  XXX pare down once all tests pass on all cpus CHECKME
            Consistency ok("\ndeconvolution_bwd_weights bad init:");
            AND_(desc()->prop_kind == prop_kind::backward_weights);
            AND_(utils::everyone_is(f32, src_type, dwei_type, ddst_type)
                    || (utils::one_of(dwei_type, f32, bf16)
                            && utils::everyone_is(bf16, src_type, ddst_type)));
            AND_(utils::one_of(desc()->alg_kind, alg_kind::deconvolution_direct,
                    alg_kind::deconvolution_winograd));
            AND_(attr()->has_default_values());
#endif

#if !DNNL_VERBOSE_EXTRA
            if (ok) {
                CHECK(init_convolution());
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
                        utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c),
                        utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c));

                return status::success;
            }
#else
            if (ok) {
                OK_CHECK(init_convolution());
                if (diff_weights_md_.format_kind == format_kind::any) {
                    CHECK(weights_axes_permutation(&diff_weights_md_,
                            conv_pd_->diff_weights_md(), with_groups()));
                }
                if (src_md_.format_kind == format_kind::any)
                    src_md_ = *conv_pd_->diff_dst_md();
                if (diff_dst_md_.format_kind == format_kind::any)
                    diff_dst_md_ = *conv_pd_->src_md();
                if (diff_bias_md_.format_kind == format_kind::any)
                    CHECK(memory_desc_init_by_tag(diff_bias_md_, x));

                dst_tag_ = memory_desc_matches_one_of_tag(diff_dst_md_,
                        utils::pick(ndims() - 3, ncw, nchw, ncdhw),
                        utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c),
                        utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c));
            }
            if (ok) return status::success;
#endif

            return status::unimplemented;
        }

        virtual void init_scratchpad_md() override {
            scratchpad_md_ = *conv_pd_->scratchpad_md();
        }

        primitive_desc_t *conv_pd_;
        format_tag_t dst_tag_;
    };

    ref_deconvolution_bwd_weights_t(const pd_t *apd) : primitive_impl_t(apd) {
        pd()->conv_pd_->create_primitive((primitive_t **)&conv_p_);
    }
    ~ref_deconvolution_bwd_weights_t() { delete conv_p_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[DNNL_ARG_DIFF_DST] = args.at(DNNL_ARG_SRC);
        conv_args[DNNL_ARG_SRC] = args.at(DNNL_ARG_DIFF_DST);
        conv_args[DNNL_ARG_DIFF_WEIGHTS] = args.at(DNNL_ARG_DIFF_WEIGHTS);
        if (!types::is_zero_md(pd()->scratchpad_md()))
            conv_args[DNNL_ARG_SCRATCHPAD] = args.at(DNNL_ARG_SCRATCHPAD);
        exec_ctx_t conv_ctx(ctx.stream(), std::move(conv_args));

        status_t status = conv_p_->execute(conv_ctx);
        if (status != status::success) return status;

        if (pd()->with_bias()) {
            using namespace data_type;

            auto dbia_type = pd()->diff_weights_md(1)->data_type;
            auto ddst_type = pd()->diff_dst_md()->data_type;
            if (utils::everyone_is(f32, dbia_type, ddst_type))
                compute_bias<f32, f32>(ctx);
            else if (utils::everyone_is(bf16, dbia_type, ddst_type))
                compute_bias<bf16, bf16>(ctx);
            else if (dbia_type == f32 && ddst_type == bf16)
                compute_bias<f32, bf16>(ctx);
        }
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    void compute_bwd_bias(float *diff_bias, const float *diff_dst) const;

    template <data_type_t dbia_type, data_type_t ddst_type>
    void compute_bwd_bias_ncdhw(
            typename prec_traits<dbia_type>::type *diff_bias,
            const typename prec_traits<ddst_type>::type *diff_dst) const;

    template <data_type_t dbia_type, data_type_t ddst_type, int blksize>
    void compute_bwd_bias_nCdhwXc(
            typename prec_traits<dbia_type>::type *diff_bias,
            const typename prec_traits<ddst_type>::type *diff_dst) const;

    template <data_type_t dbia_type, data_type_t ddst_type>
    void compute_bias(const exec_ctx_t &ctx) const;
    primitive_t *conv_p_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino=+2s,^=l0,\:0,N-s
#endif
