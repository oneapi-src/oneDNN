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

#ifndef OCL_REF_DECONVOLUTION_HPP
#define OCL_REF_DECONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "compute/compute.hpp"
#include "ocl/jit_primitive_conf.hpp"
#include "ocl/ocl_convolution_pd.hpp"
#include "ocl/ocl_deconvolution_pd.hpp"
#include "ocl/ocl_stream.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

static status_t compute_blocked_format(
        bool with_groups, const memory_desc_t *oi_md, memory_desc_t *io_md) {
    // Computes blocking for *i*o* format from *o*i* format
    bool sanity_check_ok = true && oi_md->ndims == io_md->ndims
            && oi_md->format_kind == format_kind::blocked;
    if (!sanity_check_ok) return status::invalid_arguments;

    const blocking_desc_t &oi_blk = oi_md->format_desc.blocking;
    blocking_desc_t io_blk = io_md->format_desc.blocking;

    io_md->format_kind = format_kind::blocked;
    io_blk = oi_blk;

    const int ID_OC = 0 + with_groups;
    const int ID_IC = 1 + with_groups;

    nstl::swap(io_blk.strides[ID_OC], io_blk.strides[ID_IC]);
    for (int i_blk = 0; i_blk < io_blk.inner_nblks; ++i_blk) {
        if (utils::one_of(io_blk.inner_idxs[i_blk], ID_OC, ID_IC)) {
            io_blk.inner_idxs[i_blk]
                    = (io_blk.inner_idxs[i_blk] == ID_OC ? ID_IC : ID_OC);
        }
    }

    return memory_desc_init_by_blocking_desc(*io_md, io_blk);
}

static status_t conv_descr_create(
        const deconvolution_desc_t *dd, convolution_desc_t *cd) {
    using namespace prop_kind;
    alg_kind_t alg_kind = alg_kind::convolution_direct;

    const memory_desc_t *src_md, *dst_md, *d_weights_d;
    prop_kind_t prop_kind;
    memory_desc_t c_weights_d;

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

    const bool with_groups = d_weights_d->ndims == src_md->ndims + 1;

    // Create weights desc for convolution
    c_weights_d = *d_weights_d;

    const int ID_OC = 0 + with_groups;
    const int ID_IC = 1 + with_groups;
    nstl::swap(c_weights_d.dims[ID_OC], c_weights_d.dims[ID_IC]);
    nstl::swap(c_weights_d.padded_dims[ID_OC], c_weights_d.padded_dims[ID_IC]);
    nstl::swap(c_weights_d.padded_offsets[ID_OC],
            c_weights_d.padded_offsets[ID_IC]);

    if (c_weights_d.format_kind != format_kind::any)
        CHECK(compute_blocked_format(with_groups, d_weights_d, &c_weights_d));

    return conv_desc_init(cd, prop_kind, alg_kind, src_md, &c_weights_d,
            prop_kind != backward_weights ? &dd->bias_desc : nullptr, dst_md,
            dd->strides, dd->dilates, dd->padding[0], dd->padding[1]);
}

struct ref_deconvolution_fwd_t : public primitive_impl_t {
    struct pd_t : public ocl_deconvolution_fwd_pd_t {
        pd_t(engine_t *engine, const deconvolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : ocl_deconvolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr) {}

        pd_t(const pd_t &other)
            : ocl_deconvolution_fwd_pd_t(other)
            , conv_pd_(other.conv_pd_->clone()) {}

        pd_t &operator=(const pd_t &other) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(other);
            ocl_deconvolution_fwd_pd_t::operator=(other);
            delete conv_pd_;
            conv_pd_ = other.conv_pd_->clone();
            return *this;
        }

        ~pd_t() { delete conv_pd_; }

        DECLARE_COMMON_PD_T(conv_pd_->name(), ref_deconvolution_fwd_t);

        status_t init_convolution() {
            convolution_desc_t cd;
            CHECK(conv_descr_create(desc(), &cd));
            status_t status = dnnl_primitive_desc_create(
                    &conv_pd_, (op_desc_t *)&cd, &attr_, engine_, nullptr);
            return status;
        }

        status_t init() {
            using namespace format_tag;

            bool ok = true && is_fwd()
                    && desc()->alg_kind == alg_kind::deconvolution_direct
                    && attr()->has_default_values()
                    && (utils::everyone_is(data_type::f32,
                                desc()->src_desc.data_type,
                                desc()->weights_desc.data_type,
                                desc()->dst_desc.data_type)
                            || utils::everyone_is(data_type::f16,
                                    desc()->src_desc.data_type,
                                    desc()->weights_desc.data_type,
                                    desc()->dst_desc.data_type)
                            || (utils::everyone_is(data_type::bf16,
                                        desc()->src_desc.data_type,
                                        desc()->weights_desc.data_type)
                                    && utils::one_of(desc()->dst_desc.data_type,
                                            data_type::f32, data_type::bf16))
                            || desc()->dst_desc.data_type == data_type::u8);
            if (ok) {
                CHECK(init_convolution());
                if (weights_md_.format_kind == format_kind::any) {
                    CHECK(compute_blocked_format(with_groups(),
                            conv_pd_->weights_md(), &desc_.weights_desc));
                    weights_md_ = desc_.weights_desc;
                }
                if (src_md_.format_kind == format_kind::any)
                    src_md_ = *conv_pd_->diff_dst_md();
                if (dst_md_.format_kind == format_kind::any)
                    dst_md_ = *conv_pd_->diff_src_md();
                if (bias_md_.format_kind == format_kind::any)
                    CHECK(memory_desc_init_by_tag(bias_md_, x));

                return status::success;
            }

            return status::unimplemented;
        }

        virtual void init_scratchpad_md() override {
            scratchpad_md_ = *conv_pd_->scratchpad_md();
        }

        primitive_desc_t *conv_pd_;
    };

    ref_deconvolution_fwd_t(const pd_t *apd) : primitive_impl_t(apd) {}

    ~ref_deconvolution_fwd_t() { delete conv_p_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[DNNL_ARG_DIFF_DST] = args.at(DNNL_ARG_SRC);
        conv_args[DNNL_ARG_WEIGHTS] = args.at(DNNL_ARG_WEIGHTS);
        conv_args[DNNL_ARG_DIFF_SRC] = args.at(DNNL_ARG_DST);
        if (pd()->with_bias())
            conv_args[DNNL_ARG_BIAS] = args.at(DNNL_ARG_BIAS);
        exec_ctx_t conv_ctx(ctx.stream(), std::move(conv_args));

        // Executing the convolution kernel
        status_t status = conv_p_->execute(conv_ctx);
        return status;
    }

    status_t init() override {
        // Creating convolution primitve
        status_t conv_status
                = pd()->conv_pd_->create_primitive((primitive_t **)&conv_p_);
        return conv_status;
    }
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    primitive_t *conv_p_ = nullptr;
};

struct ref_deconvolution_bwd_data_t : public primitive_impl_t {
    struct pd_t : public ocl_deconvolution_bwd_data_pd_t {
        pd_t(engine_t *engine, const deconvolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : ocl_deconvolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr) {}

        pd_t(const pd_t &other)
            : ocl_deconvolution_bwd_data_pd_t(other)
            , conv_pd_(other.conv_pd_->clone()) {}

        pd_t &operator=(const pd_t &other) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(other);
            ocl_deconvolution_bwd_data_pd_t::operator=(other);
            delete conv_pd_;
            conv_pd_ = other.conv_pd_->clone();
            return *this;
        }

        ~pd_t() { delete conv_pd_; }

        DECLARE_COMMON_PD_T(conv_pd_->name(), ref_deconvolution_bwd_data_t);

        status_t init_convolution() {
            convolution_desc_t cd;
            CHECK(conv_descr_create(desc(), &cd));
            status_t status = dnnl_primitive_desc_create(
                    &conv_pd_, (op_desc_t *)&cd, &attr_, engine_, nullptr);
            return status;
        }

        status_t init() {
            bool ok = true && desc()->prop_kind == prop_kind::backward_data
                    && (utils::everyone_is(data_type::f32,
                                desc()->diff_src_desc.data_type,
                                desc()->weights_desc.data_type,
                                desc()->diff_dst_desc.data_type)
                            || utils::everyone_is(data_type::bf16,
                                    desc()->weights_desc.data_type,
                                    desc()->diff_dst_desc.data_type))
                    && utils::one_of(desc()->diff_src_desc.data_type,
                            data_type::bf16, data_type::f32)
                    && desc()->alg_kind == alg_kind::deconvolution_direct
                    && attr()->has_default_values();

            if (ok) {
                CHECK(init_convolution());
                if (weights_md_.format_kind == format_kind::any) {
                    CHECK(compute_blocked_format(with_groups(),
                            conv_pd_->weights_md(), &desc_.weights_desc));
                    weights_md_ = desc_.weights_desc;
                }
                if (diff_src_md_.format_kind == format_kind::any)
                    diff_src_md_ = *conv_pd_->dst_md();
                if (diff_dst_md_.format_kind == format_kind::any)
                    diff_dst_md_ = *conv_pd_->src_md();

                return status::success;
            }

            return status::unimplemented;
        }

        virtual void init_scratchpad_md() override {
            scratchpad_md_ = *conv_pd_->scratchpad_md();
        }

        primitive_desc_t *conv_pd_;
    };

    ref_deconvolution_bwd_data_t(const pd_t *apd) : primitive_impl_t(apd) {}

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

        // Executing the convolution kernel
        status_t status = conv_p_->execute(conv_ctx);
        return status;
    }

    status_t init() override {
        status_t conv_status
                = pd()->conv_pd_->create_primitive((primitive_t **)&conv_p_);
        return conv_status;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    primitive_t *conv_p_ = nullptr;
};

struct ref_deconvolution_bwd_weights_t : public primitive_impl_t {
    struct pd_t : public ocl_deconvolution_bwd_weights_pd_t {
        pd_t(engine_t *engine, const deconvolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : ocl_deconvolution_bwd_weights_pd_t(
                    engine, adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr) {}

        pd_t(const pd_t &other)
            : ocl_deconvolution_bwd_weights_pd_t(other)
            , conv_pd_(other.conv_pd_->clone()) {}

        pd_t &operator=(const pd_t &other) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(other);
            ocl_deconvolution_bwd_weights_pd_t::operator=(other);
            delete conv_pd_;
            conv_pd_ = other.conv_pd_->clone();
            return *this;
        }

        ~pd_t() { delete conv_pd_; }

        DECLARE_COMMON_PD_T(conv_pd_->name(), ref_deconvolution_bwd_weights_t);

        status_t init_convolution() {
            convolution_desc_t cd;
            CHECK(conv_descr_create(desc(), &cd));
            status_t status = dnnl_primitive_desc_create(
                    &conv_pd_, (op_desc_t *)&cd, &attr_, engine_, nullptr);
            return status;
        }

        status_t init() {
            using namespace format_tag;
            bool ok = true && desc()->prop_kind == prop_kind::backward_weights
                    && (utils::everyone_is(data_type::f32,
                                desc()->src_desc.data_type,
                                desc()->diff_weights_desc.data_type,
                                desc()->diff_dst_desc.data_type)
                            || utils::everyone_is(data_type::bf16,
                                    desc()->diff_dst_desc.data_type,
                                    desc()->src_desc.data_type))
                    && utils::one_of(
                            desc()->alg_kind, alg_kind::deconvolution_direct)
                    && attr()->has_default_values()
                    && utils::one_of(desc()->diff_weights_desc.data_type,
                            data_type::bf16, data_type::f32);
            if (ok) {
                CHECK(init_convolution());
                if (diff_weights_md_.format_kind == format_kind::any) {
                    CHECK(compute_blocked_format(with_groups(),
                            conv_pd_->diff_weights_md(),
                            &desc_.diff_weights_desc));
                    diff_weights_md_ = desc_.diff_weights_desc;
                }
                if (src_md_.format_kind == format_kind::any)
                    src_md_ = *conv_pd_->diff_dst_md();
                if (diff_dst_md_.format_kind == format_kind::any)
                    diff_dst_md_ = *conv_pd_->src_md();
                if (diff_bias_md_.format_kind == format_kind::any)
                    CHECK(memory_desc_init_by_tag(diff_bias_md_, x));

                return status::success;
            }

            return status::unimplemented;
        }

        virtual void init_scratchpad_md() override {
            scratchpad_md_ = *conv_pd_->scratchpad_md();
        }

        primitive_desc_t *conv_pd_;
    };

    ref_deconvolution_bwd_weights_t(const pd_t *apd) : primitive_impl_t(apd) {}

    ~ref_deconvolution_bwd_weights_t() { delete conv_p_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        auto *compute_stream
                = utils::downcast<compute::compute_stream_t *>(ctx.stream());

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
            // Calling the bias kernel if bias=1
            auto &diff_bias = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);
            auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);

            compute::kernel_arg_list_t arg_list;
            arg_list.set(0, diff_dst);
            arg_list.set(1, diff_bias);

            // Setting up global work-space to {OC*G, 1, 1}
            auto nd_range = compute::nd_range_t({gws[0], gws[1], gws[2]});
            status = compute_stream->parallel_for(
                    nd_range, bias_kernel, arg_list);
        }
        return status::success;
    }

    status_t init() override {
        // Creating convolution primitve
        status_t conv_status
                = pd()->conv_pd_->create_primitive((primitive_t **)&conv_p_);
        if (conv_status != status::success) return conv_status;

        // Initializing values for the deconv bias kernel
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute::kernel_ctx_t kernel_ctx;

        memory_desc_wrapper diff_dst_mdw(pd()->diff_dst_md());
        kernel_ctx.set_data_type(pd()->diff_dst_md()->data_type);
        jit_offsets jit_off;
        set_offsets(diff_dst_mdw, jit_off.dst_off);
        def_offsets(jit_off.dst_off, kernel_ctx, "DST",
                pd()->desc()->diff_dst_desc.ndims);

        kernel_ctx.define_int("MB", pd()->MB());
        kernel_ctx.define_int("OH", pd()->OH());
        kernel_ctx.define_int("OW", pd()->OW());
        kernel_ctx.define_int("OD", pd()->OD());
        kernel_ctx.define_int("OC", pd()->OC() / pd()->G());
        kernel_ctx.define_int("NDIMS", pd()->desc()->src_desc.ndims);

        gws[0] = pd()->OC() * pd()->G();
        gws[1] = 1;
        gws[2] = 1;

        dst_data_type = pd()->diff_dst_md()->data_type;
        bias_data_type = pd()->with_bias() ? pd()->diff_weights_md(1)->data_type
                                           : data_type::undef;
        accum_data_type = pd()->desc()->accum_data_type;

        def_data_type(kernel_ctx, dst_data_type, "DST");
        def_data_type(kernel_ctx, bias_data_type, "BIA");
        def_data_type(kernel_ctx, accum_data_type, "ACC");

        compute_engine->create_kernel(
                &bias_kernel, "ref_deconv_backward_bias", kernel_ctx);
        if (!bias_kernel) return status::runtime_error;

        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    primitive_t *conv_p_ = nullptr;
    compute::kernel_t bias_kernel;
    size_t gws[3];
    data_type_t dst_data_type = data_type::undef;
    data_type_t bias_data_type = data_type::undef;
    data_type_t accum_data_type = data_type::undef;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif
