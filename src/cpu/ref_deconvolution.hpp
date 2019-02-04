/*******************************************************************************
* Copyright 2018 Intel Corporation
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
#include "type_helpers.hpp"
#include "utils.hpp"
#include "primitive_iterator.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_deconvolution_pd.hpp"
#include "cpu_primitive.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

static status_t compute_blocked_format(bool with_groups,
    const memory_desc_t *oi_md, memory_desc_t *io_md)
{
    /* Computes blocking for *i*o* format from *o*i* format */
    if (oi_md->ndims != io_md->ndims) return status::invalid_arguments;
    blocking_desc_t oi_blk = oi_md->layout_desc.blocking,
        &io_blk = io_md->layout_desc.blocking;
    io_blk = oi_blk;
    nstl::swap(io_blk.strides[0][0+with_groups], io_blk.strides[0][1+with_groups]);
    nstl::swap(io_blk.strides[1][0+with_groups], io_blk.strides[1][1+with_groups]);
    nstl::swap(io_md->padded_dims[0+with_groups], io_md->padded_dims[1+with_groups]);
    nstl::swap(io_md->padded_offsets[0+with_groups],
         io_md->padded_offsets[1+with_groups]);
    nstl::swap(io_blk.block_dims[0+with_groups], io_blk.block_dims[1+with_groups]);
    io_md->format = memory_format::blocked;
    return status::success;
}

static status_t conv_descr_create(const deconvolution_desc_t *dd,
        convolution_desc_t *cd)
{
    using namespace prop_kind;
    using namespace memory_format;
    alg_kind_t alg_kind = ( dd->alg_kind == alg_kind::deconvolution_direct
        ? alg_kind::convolution_direct : alg_kind::convolution_winograd );
    prop_kind_t prop_kind;
    const memory_desc_t *src_md, *dst_md;
    memory_desc_t c_weights_d, d_weights_d;
    bool with_groups;
    if ( utils::one_of(dd->prop_kind, forward_training, forward_inference) ) {
        prop_kind = backward_data;
        src_md = &dd->dst_desc;
        dst_md = &dd->src_desc;
        d_weights_d = dd->weights_desc;
    } else if (dd->prop_kind == backward_data) {
        prop_kind = forward_training;
        src_md = &dd->diff_dst_desc;
        dst_md = &dd->diff_src_desc;
        d_weights_d = dd->weights_desc;
    } else {
        prop_kind = dd->prop_kind;
        src_md = &dd->diff_dst_desc;
        dst_md = &dd->src_desc;
        d_weights_d = dd->diff_weights_desc;
    }
    with_groups = d_weights_d.ndims == src_md->ndims + 1;

    /* create weights desc for convolution */
    c_weights_d = d_weights_d;
    nstl::swap(c_weights_d.dims[with_groups + 0], c_weights_d.dims[with_groups + 1]);
    if (c_weights_d.format != any)
    {
        if (utils::one_of(c_weights_d.format, gOIhw8i16o2i, OIhw8i16o2i,
            gOIhw8o16i2o, OIhw8o16i2o, gOIhw4i16o4i, OIhw4i16o4i))
            return status::unimplemented;
        CHECK( compute_blocked_format(with_groups, &d_weights_d, &c_weights_d));
    }
    return conv_desc_init(cd, prop_kind, alg_kind, src_md, &(c_weights_d),
            (prop_kind != backward_weights ? &(dd->bias_desc) : nullptr),
            dst_md, dd->strides, dd->dilates,
            dd->padding[0], dd->padding[1], dd->padding_kind);
}

struct ref_deconvolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_deconvolution_fwd_pd_t {
        pd_t(engine_t *engine,
                const deconvolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr)
        {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_fwd_pd_t(other)
            , conv_pd_(other.conv_pd_->clone())
            , conv_supports_bias_(other.conv_supports_bias_)
        {}

        ~pd_t() { delete conv_pd_; }

        DECLARE_COMMON_PD_T(conv_pd_->name(), ref_deconvolution_fwd_t);

        status_t init_convolution() {
            using namespace memory_format;
            using namespace types;
            convolution_desc_t cd;
            status_t status;

            status = conv_descr_create(desc(), &cd);
            if (status != status::success) return status;

            mkldnn_primitive_desc_iterator it(engine_, (op_desc_t *)&cd,
                &attr_, nullptr);
            while (++it != it.end()) {
                conv_pd_ = *it;
                conv_supports_bias_ = static_cast<cpu_convolution_bwd_data_pd_t *>
                    (conv_pd_)->support_bias();
                bool output_f32 = utils::everyone_is(data_type::f32,
                        desc()->accum_data_type,
                        desc()->dst_desc.data_type);
                auto wei_fmt =
                    format_normalize(conv_pd_->weights_md()->format);

                bool ok = true
                    /* only weights in non-double-blocked format are supported */
                    && (wei_fmt == blocked && !is_format_double_blocked(wei_fmt))
                    /* deconv reference code can process only f32 bias */
                    && IMPLICATION(with_bias(),
                            conv_supports_bias_ || output_f32);
                if (ok)
                    return status::success;
                delete conv_pd_;
            }
            return status::unimplemented;
        }

        status_t init() {
            bool ok = true
                && is_fwd()
                && utils::one_of(desc()->alg_kind,
                        alg_kind::deconvolution_direct,
                        alg_kind::deconvolution_winograd)
                && attr()->post_ops_.has_default_values();

            if (ok) {
                CHECK(init_convolution());
                if (weights_md_.format == memory_format::any) {
                    CHECK(compute_blocked_format(with_groups(),
                        conv_pd_->weights_md(), &desc_.weights_desc));
                    weights_md_ = desc_.weights_desc;
                }
                if (src_md_.format == memory_format::any)
                    CHECK(types::set_default_format(src_md_,
                                conv_pd_->diff_dst_md()->format));
                if (dst_md_.format == memory_format::any)
                    CHECK(types::set_default_format(dst_md_,
                                conv_pd_->diff_src_md()->format));
                if (bias_md_.format == memory_format::any)
                    CHECK(types::set_default_format(bias_md_, memory_format::x));

                return status::success;
            }

            return status::unimplemented;
        }
        primitive_desc_t *conv_pd_;
        bool conv_supports_bias_;
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    ref_deconvolution_fwd_t(const pd_t *apd): cpu_primitive_t(apd)
    { pd()->conv_pd_->create_primitive((primitive_t **)&conv_p_); }
    ~ref_deconvolution_fwd_t() { delete conv_p_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[MKLDNN_ARG_DIFF_DST] = args.at(MKLDNN_ARG_SRC);
        conv_args[MKLDNN_ARG_WEIGHTS] = args.at(MKLDNN_ARG_WEIGHTS);
        if (pd()->with_bias() && pd()->conv_supports_bias_)
            conv_args[MKLDNN_ARG_BIAS] = args.at(MKLDNN_ARG_BIAS);
        conv_args[MKLDNN_ARG_DIFF_SRC] = args.at(MKLDNN_ARG_DST);
        const exec_ctx_t conv_ctx(ctx.stream(), std::move(conv_args));

        conv_p_->execute(conv_ctx);

        if (pd()->with_bias() && !pd()->conv_supports_bias_) {
            auto bias = CTX_IN_MEM(const data_t *, MKLDNN_ARG_BIAS);
            auto dst = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DST);

            switch (pd()->dst_md()->format) {
            case memory_format::ncw :
            case memory_format::nchw :
            case memory_format::ncdhw :
                compute_fwd_bias_ncdhw(bias, dst);
                break;
            case memory_format::nChw8c :
            case memory_format::nCdhw8c :
                compute_fwd_bias_nCdhwXc<8>(bias, dst);
                break;
            case memory_format::nCw16c :
            case memory_format::nChw16c :
            case memory_format::nCdhw16c :
                compute_fwd_bias_nCdhwXc<16>(bias, dst);
                break;
            default:
                compute_fwd_bias(bias, dst);
                break;
            }
        }
        return status::success;
    }

private:
    void compute_fwd_bias(const data_t *bias, data_t *dst) const;
    void compute_fwd_bias_ncdhw(const data_t *bias, data_t *dst) const;
    template <int blksize> void compute_fwd_bias_nCdhwXc(const data_t *bias,
            data_t *dst) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    primitive_t *conv_p_;
};

struct ref_deconvolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_deconvolution_bwd_data_pd_t {
        pd_t(engine_t *engine, const deconvolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr)
        {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_bwd_data_pd_t(other)
            , conv_pd_(other.conv_pd_->clone()) {}

        ~pd_t() { delete conv_pd_; }

        DECLARE_COMMON_PD_T(conv_pd_->name(), ref_deconvolution_bwd_data_t);

        status_t init_convolution() {
            using namespace memory_format;
            using namespace types;

            convolution_desc_t cd;
            status_t status = conv_descr_create(desc(), &cd);
            if (status != status::success) return status;

             mkldnn_primitive_desc_iterator it(engine_, (op_desc_t *)&cd,
                &attr_, nullptr);
             while (++it != it.end()) {
                conv_pd_ = *it;
                auto wei_fmt =
                    format_normalize(conv_pd_->weights_md()->format);
                /* only weights in non-double-blocked format are supported */
                if (wei_fmt == blocked && !is_format_double_blocked(wei_fmt))
                    return status::success;
                delete conv_pd_;
            }

            return status::unimplemented;
        }

        status_t init() {
            using namespace data_type;
            bool ok = true
                && desc()->prop_kind == prop_kind::backward_data
                && utils::everyone_is(data_type::f32,
                        desc()->diff_src_desc.data_type,
                        desc()->weights_desc.data_type,
                        desc()->diff_dst_desc.data_type)
                && utils::one_of(desc()->alg_kind,
                        alg_kind::deconvolution_direct,
                        alg_kind::deconvolution_winograd);

            if (ok) {
                CHECK(init_convolution());
                if (weights_md_.format == memory_format::any) {
                    CHECK(compute_blocked_format(with_groups(),
                        conv_pd_->weights_md(), &desc_.weights_desc));
                    weights_md_ = desc_.weights_desc;
                }
                if (diff_src_md_.format == memory_format::any)
                    CHECK(types::set_default_format(diff_src_md_,
                                conv_pd_->dst_md()->format));
                if (diff_dst_md_.format == memory_format::any)
                    CHECK(types::set_default_format(diff_dst_md_,
                                conv_pd_->src_md()->format));

                return status::success;
            }

            return status::unimplemented;
        }
        primitive_desc_t *conv_pd_;
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    ref_deconvolution_bwd_data_t(const pd_t *apd): cpu_primitive_t(apd)
    { pd()->conv_pd_->create_primitive((primitive_t **)&conv_p_); }
    ~ref_deconvolution_bwd_data_t() { delete conv_p_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[MKLDNN_ARG_SRC] = args.at(MKLDNN_ARG_DIFF_DST);
        conv_args[MKLDNN_ARG_WEIGHTS] = args.at(MKLDNN_ARG_WEIGHTS);
        conv_args[MKLDNN_ARG_DST] = args.at(MKLDNN_ARG_DIFF_SRC);
        const exec_ctx_t conv_ctx(ctx.stream(), std::move(conv_args));

        conv_p_->execute(conv_ctx);
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    primitive_t *conv_p_;
};

struct ref_deconvolution_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_deconvolution_bwd_weights_pd_t {
        pd_t(engine_t *engine,
                const deconvolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr)
        {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_bwd_weights_pd_t(other)
            , conv_pd_(other.conv_pd_->clone()) {}

        ~pd_t() { delete conv_pd_; }

        DECLARE_COMMON_PD_T(conv_pd_->name(), ref_deconvolution_bwd_weights_t);

        status_t init_convolution() {
            using namespace memory_format;
            using namespace types;

            convolution_desc_t cd;
            status_t status = conv_descr_create(desc(), &cd);
            if (status != status::success) return status;

             mkldnn_primitive_desc_iterator it(engine_, (op_desc_t *)&cd,
                &attr_, nullptr);
             while (++it != it.end()) {
                conv_pd_ = *it;
                auto wei_fmt = format_normalize(
                        conv_pd_->diff_weights_md()->format);
                /* only weights in non-double-blocked format are supported */
                if (wei_fmt == blocked && !is_format_double_blocked(wei_fmt))
                    return status::success;
                delete conv_pd_;
            }
            return status::unimplemented;
        }

        status_t init() {
            bool ok = true
                && desc()->prop_kind == prop_kind::backward_weights
                && utils::everyone_is(data_type::f32,
                        desc()->src_desc.data_type,
                        desc()->diff_weights_desc.data_type,
                        desc()->diff_dst_desc.data_type)
                && utils::one_of(desc()->alg_kind,
                        alg_kind::deconvolution_direct,
                        alg_kind::deconvolution_winograd)
                && attr()->has_default_values();
            if (ok) {
                CHECK(init_convolution());
                if (diff_weights_md_.format == memory_format::any) {
                    CHECK(compute_blocked_format(with_groups(),
                        conv_pd_->diff_weights_md(), &desc_.diff_weights_desc));
                    diff_weights_md_ = desc_.diff_weights_desc;
                }
                if (src_md_.format == memory_format::any)
                    CHECK(types::set_default_format(src_md_,
                                conv_pd_->diff_dst_md()->format));
                if (diff_dst_md_.format == memory_format::any)
                    CHECK(types::set_default_format(diff_dst_md_,
                                conv_pd_->src_md()->format));
                if (diff_bias_md_.format == memory_format::any)
                    CHECK(types::set_default_format(diff_bias_md_,
                                memory_format::x));

                return status::success;
            }

            return status::unimplemented;
        }
        primitive_desc_t *conv_pd_;
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    ref_deconvolution_bwd_weights_t(const pd_t *apd): cpu_primitive_t(apd)
    { pd()->conv_pd_->create_primitive((primitive_t **)&conv_p_); }
    ~ref_deconvolution_bwd_weights_t() { delete conv_p_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[MKLDNN_ARG_DIFF_DST] = args.at(MKLDNN_ARG_SRC);
        conv_args[MKLDNN_ARG_SRC] = args.at(MKLDNN_ARG_DIFF_DST);
        conv_args[MKLDNN_ARG_DIFF_WEIGHTS] = args.at(MKLDNN_ARG_DIFF_WEIGHTS);
        const exec_ctx_t conv_ctx(ctx.stream(), std::move(conv_args));

        status_t status = conv_p_->execute(conv_ctx);
        if (status != status::success) return status;

        if (pd()->with_bias()) {
            auto diff_dst = CTX_IN_MEM(const data_t *, MKLDNN_ARG_DIFF_DST);
            auto diff_bias = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DIFF_BIAS);

            switch (pd()->diff_dst_md()->format) {
            case memory_format::ncw :
            case memory_format::nchw :
            case memory_format::ncdhw :
                compute_bwd_bias_ncdhw(diff_dst, diff_bias);
                break;
            case memory_format::nChw8c :
                compute_bwd_bias_nCdhwXc<8>(diff_dst, diff_bias);
                break;
            case memory_format::nCw16c :
            case memory_format::nChw16c :
            case memory_format::nCdhw16c :
                compute_bwd_bias_nCdhwXc<16>(diff_dst, diff_bias);
                break;
            default:
                compute_bwd_bias(diff_dst, diff_bias);
                break;
            }
        }
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    void compute_bwd_bias(const data_t *diff_dst, data_t *diff_bias) const;
    void compute_bwd_bias_ncdhw(const data_t *diff_dst,
            data_t *diff_bias) const;
    template <int blksize> void compute_bwd_bias_nCdhwXc(
            const data_t *diff_dst, data_t *diff_bias) const;

    primitive_t *conv_p_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
