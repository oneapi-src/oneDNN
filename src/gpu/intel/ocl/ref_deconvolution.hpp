/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_REF_DECONVOLUTION_HPP
#define GPU_INTEL_OCL_REF_DECONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_deconvolution_pd.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

static status_t weights_axes_permutation(
        memory_desc_t *o_md, const memory_desc_t *i_md, bool with_groups) {
    int perm[DNNL_MAX_NDIMS] {}; // deconv to conv weight permutation
    for (int d = 0; d < DNNL_MAX_NDIMS; ++d)
        perm[d] = d;
    nstl::swap(perm[0 + with_groups], perm[1 + with_groups]);

    return memory_desc_permute_axes(*o_md, *i_md, perm);
}

static status_t conv_descr_create(
        const deconvolution_desc_t *dd, convolution_desc_t *cd) {
    using namespace prop_kind;
    alg_kind_t alg_kind = alg_kind::convolution_direct;

    const memory_desc_t *src_md, *dst_md, *d_weights_d;
    prop_kind_t prop_kind;

    switch (dd->prop_kind) {
        case forward:
        case forward_inference:
            prop_kind = backward_data;
            src_md = &dd->dst_desc;
            dst_md = &dd->src_desc;
            d_weights_d = &dd->weights_desc;
            break;
        case backward_data:
            prop_kind = forward_training;
            src_md = &dd->diff_dst_desc;
            dst_md = &dd->diff_src_desc;
            d_weights_d = &dd->weights_desc;
            break;
        case backward_weights:
            prop_kind = dd->prop_kind;
            src_md = &dd->diff_dst_desc;
            dst_md = &dd->src_desc;
            d_weights_d = &dd->diff_weights_desc;
            break;
        default: assert(!"unknown prop kind"); return status::invalid_arguments;
    }

    // Create weights desc for convolution
    memory_desc_t c_weights_d;
    const bool with_groups = d_weights_d->ndims == src_md->ndims + 1;
    CHECK(weights_axes_permutation(&c_weights_d, d_weights_d, with_groups));

    return conv_desc_init(cd, prop_kind, alg_kind, src_md, &c_weights_d,
            prop_kind != backward_weights ? &dd->bias_desc : nullptr, dst_md,
            dd->strides, dd->dilates, dd->padding[0], dd->padding[1]);
}

struct ref_deconvolution_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_deconvolution_fwd_pd_t {
        pd_t(const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : gpu_deconvolution_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other) = default;

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(name_.c_str(), ref_deconvolution_fwd_t);
        status_t init_convolution(impl::engine_t *engine) {
            convolution_desc_t cd;
            CHECK(conv_descr_create(desc(), &cd));
            primitive_attr_t conv_attr(*attr());
            if (!conv_attr.is_initialized()) return status::out_of_memory;
            primitive_desc_iterator_t it(
                    engine, (op_desc_t *)&cd, &conv_attr, nullptr);
            if (!it.is_initialized()) return status::out_of_memory;
            conv_pd_ = *(++it);

            return (conv_pd_) ? status::success : status::unimplemented;
        }

        status_t init(impl::engine_t *engine) {
            using namespace format_tag;
            using sm = primitive_attr_t::skip_mask_t;

            const auto attr_skip_mask = sm::post_ops | sm::zero_points_runtime
                    | sm::scales_runtime;

            VDISPATCH_DECONVOLUTION(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_DECONVOLUTION(
                    desc()->alg_kind == alg_kind::deconvolution_direct,
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_DECONVOLUTION(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_DECONVOLUTION(
                    post_ops_with_binary_ok(
                            attr(), desc()->dst_desc.data_type, ndims()),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_DECONVOLUTION(
                    (utils::everyone_is(data_type::f32,
                             desc()->src_desc.data_type,
                             desc()->weights_desc.data_type,
                             desc()->dst_desc.data_type)
                            || (utils::everyone_is(data_type::f64,
                                    desc()->src_desc.data_type,
                                    desc()->weights_desc.data_type,
                                    desc()->dst_desc.data_type))
                            || ((utils::everyone_is(data_type::f16,
                                         desc()->src_desc.data_type,
                                         desc()->weights_desc.data_type)
                                        || utils::everyone_is(data_type::f32,
                                                desc()->src_desc.data_type,
                                                desc()->weights_desc.data_type)
                                        || utils::everyone_is(data_type::bf16,
                                                desc()->src_desc.data_type,
                                                desc()->weights_desc.data_type))
                                    && utils::one_of(desc()->dst_desc.data_type,
                                            data_type::f16, data_type::u8,
                                            data_type::s8))
                            || (utils::everyone_is(data_type::bf16,
                                        desc()->src_desc.data_type,
                                        desc()->weights_desc.data_type)
                                    && utils::one_of(desc()->dst_desc.data_type,
                                            data_type::f32, data_type::bf16))
                            || (utils::everyone_is(data_type::f16,
                                        desc()->src_desc.data_type,
                                        desc()->weights_desc.data_type)
                                    && utils::one_of(desc()->dst_desc.data_type,
                                            data_type::f32, data_type::f16))
                            || (desc()->weights_desc.data_type == data_type::s8
                                    && utils::one_of(desc()->src_desc.data_type,
                                            data_type::u8, data_type::s8)
                                    && desc()->dst_desc.data_type
                                            != data_type::f64)),
                    VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_DECONVOLUTION_SC(
                    init_convolution(engine), "init_convolution()");
            if (weights_md_.format_kind == format_kind::any) {
                VDISPATCH_DECONVOLUTION_SC(
                        weights_axes_permutation(&weights_md_,
                                conv_pd_->weights_md(), with_groups()),
                        "weights_axes_permutation()");
            }
            if (src_md_.format_kind == format_kind::any)
                src_md_ = *conv_pd_->diff_dst_md();
            if (dst_md_.format_kind == format_kind::any)
                dst_md_ = *conv_pd_->diff_src_md();
            if (bias_md_.format_kind == format_kind::any) {
                VDISPATCH_DECONVOLUTION_SC(memory_desc_init_by_tag(bias_md_, x),
                        VERBOSE_UNSUPPORTED_TAG);
            }
            init_name();
            init_scratchpad();
            VDISPATCH_DECONVOLUTION_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_ATTR);

            return status::success;
        }

        std::shared_ptr<primitive_desc_t> conv_pd_;

    private:
        std::string name_ = "ocl:ref:any";

        void init_name() {
            name_.append("+");
            name_.append(conv_pd_->name());
        }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    conv_pd_->scratchpad_registry());
        }
    };

    status_t init(impl::engine_t *engine) override {
        return create_nested_primitive(conv_p_, pd()->conv_pd_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace memory_tracking::names;
        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[DNNL_ARG_DIFF_DST] = args.at(DNNL_ARG_SRC);
        conv_args[DNNL_ARG_WEIGHTS] = args.at(DNNL_ARG_WEIGHTS);
        conv_args[DNNL_ARG_DIFF_SRC] = args.at(DNNL_ARG_DST);
        if (pd()->with_bias())
            conv_args[DNNL_ARG_BIAS] = args.at(DNNL_ARG_BIAS);

        for (int idx = 0; idx < pd()->attr()->post_ops_.len(); ++idx) {
            if (pd()->attr()->post_ops_.entry_[idx].is_binary()) {
                conv_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1]
                        = args.at(DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx)
                                | DNNL_ARG_SRC_1);
            } else if (pd()->attr()->post_ops_.entry_[idx].is_prelu()) {
                conv_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx)
                        | DNNL_ARG_WEIGHTS]
                        = args.at(DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx)
                                | DNNL_ARG_WEIGHTS);
            }
        }
        const auto z_src = DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC;
        const auto z_dst = DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST;
        if (args.find(z_src) != args.end()) conv_args[z_src] = args.at(z_src);
        if (args.find(z_dst) != args.end()) conv_args[z_dst] = args.at(z_dst);

        for (int arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            int key = DNNL_ARG_ATTR_SCALES | arg;
            if (args.find(key) != args.end()) conv_args[key] = args.at(key);
        }

        exec_ctx_t conv_ctx(ctx, std::move(conv_args));

        nested_scratchpad_t ns(ctx, key_nested, conv_p_);
        conv_ctx.set_scratchpad_grantor(ns.grantor());
        // Executing the convolution kernel
        return conv_p_->execute(conv_ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> conv_p_;
};

struct ref_deconvolution_bwd_data_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_deconvolution_bwd_data_pd_t {
        pd_t(const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : gpu_deconvolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr) {}

        pd_t(const pd_t &other) = default;

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(name_.c_str(), ref_deconvolution_bwd_data_t);

        status_t init_convolution(impl::engine_t *engine) {
            convolution_desc_t cd;
            CHECK(conv_descr_create(desc(), &cd));
            primitive_attr_t conv_attr(*attr());
            if (!conv_attr.is_initialized()) return status::out_of_memory;
            primitive_desc_iterator_t it(
                    engine, (op_desc_t *)&cd, &conv_attr, nullptr);
            if (!it.is_initialized()) return status::out_of_memory;
            conv_pd_ = *(++it);
            return (conv_pd_) ? status::success : status::unimplemented;
        }

        status_t init(impl::engine_t *engine) {
            VDISPATCH_DECONVOLUTION(
                    desc()->prop_kind == prop_kind::backward_data,
                    VERBOSE_BAD_PROPKIND);

            VDISPATCH_DECONVOLUTION(
                    (utils::everyone_is(data_type::f32,
                             desc()->diff_src_desc.data_type,
                             desc()->weights_desc.data_type,
                             desc()->diff_dst_desc.data_type)
                            || (utils::everyone_is(data_type::f64,
                                    desc()->diff_src_desc.data_type,
                                    desc()->weights_desc.data_type,
                                    desc()->diff_dst_desc.data_type))
                            || utils::everyone_is(data_type::f16,
                                    desc()->weights_desc.data_type,
                                    desc()->diff_dst_desc.data_type)
                            || utils::everyone_is(data_type::bf16,
                                    desc()->weights_desc.data_type,
                                    desc()->diff_dst_desc.data_type)),
                    VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_DECONVOLUTION(
                    utils::one_of(desc()->diff_src_desc.data_type,
                            data_type::bf16, data_type::f16, data_type::f32,
                            data_type::f64),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_DECONVOLUTION(
                    desc()->alg_kind == alg_kind::deconvolution_direct,
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_DECONVOLUTION(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_DECONVOLUTION_SC(
                    init_convolution(engine), "init_convolution()");
            if (weights_md_.format_kind == format_kind::any)
                VDISPATCH_DECONVOLUTION_SC(
                        weights_axes_permutation(&weights_md_,
                                conv_pd_->weights_md(), with_groups()),
                        "weights_axes_permutation()");
            if (diff_src_md_.format_kind == format_kind::any)
                diff_src_md_ = *conv_pd_->dst_md();
            if (diff_dst_md_.format_kind == format_kind::any)
                diff_dst_md_ = *conv_pd_->src_md();

            init_name();
            init_scratchpad();

            return status::success;
        }

        std::shared_ptr<primitive_desc_t> conv_pd_;

    private:
        std::string name_ = "ocl:ref:any";

        void init_name() {
            name_.append("+");
            name_.append(conv_pd_->name());
        }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    conv_pd_->scratchpad_registry());
        }
    };

    status_t init(impl::engine_t *engine) override {
        return create_nested_primitive(conv_p_, pd()->conv_pd_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace memory_tracking::names;
        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[DNNL_ARG_SRC] = args.at(DNNL_ARG_DIFF_DST);
        conv_args[DNNL_ARG_WEIGHTS] = args.at(DNNL_ARG_WEIGHTS);
        conv_args[DNNL_ARG_DST] = args.at(DNNL_ARG_DIFF_SRC);
        if (!types::is_zero_md(pd()->scratchpad_md()))
            conv_args[DNNL_ARG_SCRATCHPAD] = args.at(DNNL_ARG_SCRATCHPAD);
        exec_ctx_t conv_ctx(ctx, std::move(conv_args));

        nested_scratchpad_t ns(ctx, key_nested, conv_p_);
        conv_ctx.set_scratchpad_grantor(ns.grantor());
        // Executing the convolution kernel
        return conv_p_->execute(conv_ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> conv_p_;
};

struct ref_deconvolution_bwd_weights_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_deconvolution_bwd_weights_pd_t {
        pd_t(const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : gpu_deconvolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other) = default;

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(name_.c_str(), ref_deconvolution_bwd_weights_t);

        status_t init_convolution(impl::engine_t *engine) {
            convolution_desc_t cd;
            CHECK(conv_descr_create(desc(), &cd));
            primitive_attr_t conv_attr(*attr());
            if (!conv_attr.is_initialized()) return status::out_of_memory;
            primitive_desc_iterator_t it(
                    engine, (op_desc_t *)&cd, &conv_attr, nullptr);
            if (!it.is_initialized()) return status::out_of_memory;
            conv_pd_ = *(++it);
            return (conv_pd_) ? status::success : status::unimplemented;
        }

        status_t init(impl::engine_t *engine) {
            using namespace format_tag;
            VDISPATCH_DECONVOLUTION(
                    desc()->prop_kind == prop_kind::backward_weights,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_DECONVOLUTION(
                    (utils::everyone_is(data_type::f32,
                             desc()->src_desc.data_type,
                             desc()->diff_weights_desc.data_type,
                             desc()->diff_dst_desc.data_type)
                            || utils::everyone_is(data_type::f64,
                                    desc()->src_desc.data_type,
                                    desc()->diff_weights_desc.data_type,
                                    desc()->diff_dst_desc.data_type)
                            || utils::everyone_is(data_type::f16,
                                    desc()->diff_dst_desc.data_type,
                                    desc()->src_desc.data_type)
                            || utils::everyone_is(data_type::bf16,
                                    desc()->diff_dst_desc.data_type,
                                    desc()->src_desc.data_type)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_DECONVOLUTION(utils::one_of(desc()->alg_kind,
                                            alg_kind::deconvolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_DECONVOLUTION(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_DECONVOLUTION(
                    utils::one_of(desc()->diff_weights_desc.data_type,
                            data_type::bf16, data_type::f16, data_type::f32,
                            data_type::f64),
                    VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_DECONVOLUTION_SC(
                    init_convolution(engine), "init_convolution()");
            if (diff_weights_md_.format_kind == format_kind::any) {
                VDISPATCH_DECONVOLUTION_SC(
                        weights_axes_permutation(&diff_weights_md_,
                                conv_pd_->diff_weights_md(), with_groups()),
                        "weights_axes_permutation()");
            }
            if (src_md_.format_kind == format_kind::any)
                src_md_ = *conv_pd_->diff_dst_md();
            if (diff_dst_md_.format_kind == format_kind::any)
                diff_dst_md_ = *conv_pd_->src_md();
            if (diff_bias_md_.format_kind == format_kind::any) {
                VDISPATCH_DECONVOLUTION_SC(
                        memory_desc_init_by_tag(diff_bias_md_, x),
                        VERBOSE_UNSUPPORTED_TAG);
            }

            init_name();
            init_scratchpad();

            return status::success;
        }

        std::shared_ptr<primitive_desc_t> conv_pd_;

    private:
        std::string name_ = "ocl:ref:any";

        void init_name() {
            name_.append("+");
            name_.append(conv_pd_->name());
        }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    conv_pd_->scratchpad_registry());
        }
    };

    status_t init(impl::engine_t *engine) override {
        // Creating convolution primitve
        CHECK(create_nested_primitive(conv_p_, pd()->conv_pd_, engine));

        if (!pd()->with_bias()) return status::success;
        // Initializing values for the deconv bias kernel
        compute::kernel_ctx_t kernel_ctx;

        memory_desc_wrapper diff_dst_mdw(pd()->diff_dst_md());
        kernel_ctx.set_data_type(pd()->diff_dst_md()->data_type);
        offsets_t off;
        set_offsets(diff_dst_mdw, off.dst_off);
        def_offsets(off.dst_off, kernel_ctx, "DST",
                pd()->desc()->diff_dst_desc.ndims);

        kernel_ctx.define_int("MB", pd()->MB());
        kernel_ctx.define_int("OH", pd()->OH());
        kernel_ctx.define_int("OW", pd()->OW());
        kernel_ctx.define_int("OD", pd()->OD());
        kernel_ctx.define_int("OC", pd()->OC() / pd()->G());
        kernel_ctx.define_int("NDIMS", pd()->desc()->src_desc.ndims);

        gws[0] = pd()->OC();

        dst_data_type = pd()->diff_dst_md()->data_type;
        bias_data_type = pd()->diff_weights_md(1)->data_type;
        accum_data_type = pd()->desc()->accum_data_type;

        def_data_type(kernel_ctx, dst_data_type, "DST");
        def_data_type(kernel_ctx, bias_data_type, "BIA");
        def_data_type(kernel_ctx, accum_data_type, "ACC");

        CHECK(create_kernel(
                engine, &bias_kernel_, "ref_deconv_backward_bias", kernel_ctx));
        if (!bias_kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace memory_tracking::names;

        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[DNNL_ARG_DIFF_DST] = args.at(DNNL_ARG_SRC);
        conv_args[DNNL_ARG_SRC] = args.at(DNNL_ARG_DIFF_DST);
        conv_args[DNNL_ARG_DIFF_WEIGHTS] = args.at(DNNL_ARG_DIFF_WEIGHTS);
        if (!types::is_zero_md(pd()->scratchpad_md()))
            conv_args[DNNL_ARG_SCRATCHPAD] = args.at(DNNL_ARG_SCRATCHPAD);
        exec_ctx_t conv_ctx(ctx, std::move(conv_args));

        nested_scratchpad_t ns(ctx, key_nested, conv_p_);
        conv_ctx.set_scratchpad_grantor(ns.grantor());

        status_t status = conv_p_->execute(conv_ctx);
        if (status != status::success) return status;

        if (pd()->with_bias()) {
            // Calling the bias kernel if bias=1
            auto &diff_bias = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);
            auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);

            compute::kernel_arg_list_t arg_list;
            arg_list.set(0, diff_dst);
            arg_list.set(1, diff_bias);

            // Setting up global work-space to {OC*G, 1, 1}
            auto nd_range = compute::nd_range_t(gws);
            status = parallel_for(ctx, nd_range, bias_kernel_, arg_list);
        }
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> conv_p_;
    compute::kernel_t bias_kernel_;
    compute::range_t gws = compute::range_t::empty(1);
    data_type_t dst_data_type = data_type::undef;
    data_type_t bias_data_type = data_type::undef;
    data_type_t accum_data_type = data_type::undef;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
