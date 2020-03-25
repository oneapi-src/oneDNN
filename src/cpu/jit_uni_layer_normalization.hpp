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

#ifndef CPU_JIT_UNI_LAYER_NORMALIZATION_HPP
#define CPU_JIT_UNI_LAYER_NORMALIZATION_HPP

#include "cpu_layer_normalization_pd.hpp"
#include "dnnl_thread.hpp"
#include "jit_uni_layer_normalization_kernels.hpp"
#include "memory_tracking.hpp"
#include "primitive.hpp"
#include "reorder_pd.hpp"
#include "stream.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
/* Stats and src here are compatible if
 * stat_strides[:] == data_strides[:] / last_data_dimension
 * i.e. abcd & abc, bacd & bac - compatible */
static status_t fill_compatible_stats_md(
        const memory_desc_t &src_md, memory_desc_t &stat_md) {
    stat_md = src_md;
    stat_md.ndims -= 1;
    return memory_desc_init_by_blocking_desc(
            stat_md, src_md.format_desc.blocking);
}

static status_t create_reorder_pd(engine_t *engine,
        const memory_desc_t *from_md, const memory_desc_t *to_md,
        std::unique_ptr<primitive_desc_t> &reorder_pd) {
    auto r_impls = engine->get_reorder_implementation_list(from_md, to_md);
    for (auto r = r_impls; *r; ++r) {
        primitive_attr_t r_attr;
        r_attr.set_scratchpad_mode(scratchpad_mode::user);
        reorder_pd_t *r_pd = nullptr;
        if ((*r)(&r_pd, engine, &r_attr, engine, from_md, engine, to_md)
                == status::success) {
            reorder_pd.reset(r_pd);
            break;
        }
    }
    return status::success;
}

struct jit_uni_layer_normalization_fwd_t : public primitive_t {
    struct pd_t : public cpu_layer_normalization_fwd_pd_t {
        pd_t(const layer_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const layer_normalization_fwd_pd_t *hint_fwd_pd)
            : cpu_layer_normalization_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other) : cpu_layer_normalization_fwd_pd_t(other) {
            copy_from(other);
            reordered_stat_md_ = other.reordered_stat_md_;
        }

        pd_t &operator=(const pd_t &other) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(other);
            cpu_layer_normalization_fwd_pd_t::operator=(other);
            copy_from(other);
            reordered_stat_md_ = other.reordered_stat_md_;
            return *this;
        }
        ~pd_t() = default;

        DECLARE_COMMON_PD_T("jit_uni_layer_normalization:any",
                jit_uni_layer_normalization_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper stat_d(stat_md());

            bool ok = true && is_fwd() && !has_zero_dim_memory()
                    && utils::everyone_is(f32, src_md()->data_type,
                            stat_md()->data_type, dst_md()->data_type)
                    && IMPLICATION(
                            use_scaleshift(), weights_md()->data_type == f32)
                    && src_d.is_blocking_desc()
                    && src_d.blocking_desc().strides[ndims() - 1]
                            == 1 //plain format, last logical dim is last physical
                    && attr()->has_default_values()
                    && set_default_formats_common();
            if (!ok) return status::unimplemented;

            CHECK(fill_compatible_stats_md(*src_md(), reordered_stat_md_));

            if (reordered_stat_md_ != *stat_md() && !stats_are_tmp()) {
                CHECK(create_reorder_pd(engine,
                        stats_are_src() ? stat_md() : &reordered_stat_md_,
                        stats_are_src() ? &reordered_stat_md_ : stat_md(),
                        reorder_pd_));
            }

            init_scratchpad();
            return status::success;
        }

        bool use_tmp_stats() const { return reorder_pd_ || stats_are_tmp(); }

        std::unique_ptr<primitive_desc_t> reorder_pd_;
        memory_desc_t reordered_stat_md_;

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            if (use_tmp_stats()) {
                scratchpad.book<float>(key_lnorm_tmp_mean, across_axis());
                scratchpad.book<float>(key_lnorm_tmp_var, across_axis());
            }
            if (reordered_stat_md_ != *stat_md() && !stats_are_tmp()) {
                scratchpad.book(key_nested, reorder_pd_->scratchpad_registry());
            }
        }

        void copy_from(const pd_t &other) {
            reordered_stat_md_ = other.reordered_stat_md_;
            reorder_pd_.reset(
                    other.reorder_pd_ ? other.reorder_pd_->clone() : nullptr);
        }
    };

    virtual status_t init(engine_t *engine) override {
        if (pd()->reorder_pd_)
            pd()->reorder_pd_->create_primitive(reorder_, engine);
        stat_kernel_ = new statistics_kernel_t(pd());
        data_kernel_ = new data_kernel_t(pd());
        return status::success;
    }

    jit_uni_layer_normalization_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    ~jit_uni_layer_normalization_fwd_t() {
        delete stat_kernel_;
        delete data_kernel_;
    }

    void reorder_stat(const exec_ctx_t &ctx, engine_t *engine,
            const memory_arg_t &in, const memory_arg_t &out) const {
        using namespace memory_tracking::names;
        exec_args_t r_args;
        r_args[DNNL_ARG_SRC] = in;
        r_args[DNNL_ARG_DST] = out;
        exec_ctx_t r_ctx(ctx.stream(), std::move(r_args));

        nested_scratchpad_t ns(ctx, key_nested, reorder_);
        r_ctx.set_scratchpad_grantor(ns.grantor());
        reorder_->execute(r_ctx);
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        /* LN supports arbitrary layout for input/output statistics.
         * For best performance we compute LN with statistics in the same format
         * as data tensor (i.e. data in abcd, stats in abc) and user's
         * input/output statistics are reordered if necessary */
        using namespace memory_tracking::names;
        engine_t *engine = ctx.stream()->engine();
        auto scratchpad = ctx.get_scratchpad_grantor();
        auto mean_handle = scratchpad.template get<void>(key_lnorm_tmp_mean);
        auto variance_handle = scratchpad.template get<void>(key_lnorm_tmp_var);
        memory_t mean(engine, &(pd()->reordered_stat_md_),
                memory_flags_t::use_runtime_ptr, mean_handle);
        memory_t variance(engine, &(pd()->reordered_stat_md_),
                memory_flags_t::use_runtime_ptr, variance_handle);

        // reorder input stats
        if (pd()->stats_are_src() && reorder_) {
            reorder_stat(
                    ctx, engine, ctx.args().at(DNNL_ARG_MEAN), {&mean, false});
            reorder_stat(ctx, engine, ctx.args().at(DNNL_ARG_VARIANCE),
                    {&variance, false});
        }
        execute_forward(ctx);
        // reorder output stats
        if (!pd()->stats_are_src() && reorder_) {
            reorder_stat(
                    ctx, engine, {&mean, true}, ctx.args().at(DNNL_ARG_MEAN));
            reorder_stat(ctx, engine, {&variance, true},
                    ctx.args().at(DNNL_ARG_VARIANCE));
        }

        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    statistics_kernel_t *stat_kernel_;
    data_kernel_t *data_kernel_;
    std::shared_ptr<primitive_t> reorder_;
};

struct jit_uni_layer_normalization_bwd_t : public primitive_t {
    struct pd_t : public cpu_layer_normalization_bwd_pd_t {
        pd_t(const layer_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const layer_normalization_fwd_pd_t *hint_fwd_pd)
            : cpu_layer_normalization_bwd_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other) : cpu_layer_normalization_bwd_pd_t(other) {
            copy_from(other);
        }

        pd_t &operator=(const pd_t &other) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(other);
            cpu_layer_normalization_bwd_pd_t::operator=(other);
            copy_from(other);
            return *this;
        }
        ~pd_t() = default;

        DECLARE_COMMON_PD_T("jit_uni_layer_normalization:any",
                jit_uni_layer_normalization_bwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper stat_d(stat_md());

            bool ok = true && is_bwd() && !has_zero_dim_memory()
                    && set_default_formats_common()
                    && utils::everyone_is(f32, src_md()->data_type,
                            diff_src_md()->data_type, stat_md()->data_type)
                    && IMPLICATION(use_scaleshift(),
                            utils::everyone_is(f32, weights_md()->data_type,
                                    diff_weights_md()->data_type))
                    && src_d.is_blocking_desc()
                    && src_d.blocking_desc().strides[ndims() - 1]
                            == 1 //plain format, last logical dim is last physical
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            CHECK(fill_compatible_stats_md(*src_md(), reordered_stat_md_));

            if (reordered_stat_md_ != *stat_md()) {
                CHECK(create_reorder_pd(
                        engine, stat_md(), &reordered_stat_md_, reorder_pd_));
            }

            init_scratchpad();
            return status::success;
        }

        bool use_tmp_stats() const { return reorder_pd_.get(); }

        std::unique_ptr<primitive_desc_t> reorder_pd_;
        memory_desc_t reordered_stat_md_;

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            if (use_tmp_stats()) {
                scratchpad.book<float>(key_lnorm_tmp_mean, across_axis());
                scratchpad.book<float>(key_lnorm_tmp_var, across_axis());
            }
            scratchpad.book<float>(key_lnorm_reduction,
                    2 * norm_axis() * dnnl_get_max_threads());
            scratchpad.book<float>(key_lnorm_tmp_diff_ss, 2 * norm_axis());
            if (reordered_stat_md_ != *stat_md() && !stats_are_tmp()) {
                scratchpad.book(key_nested, reorder_pd_->scratchpad_registry());
            }
        }

        void copy_from(const pd_t &other) {
            reordered_stat_md_ = other.reordered_stat_md_;
            reorder_pd_.reset(
                    other.reorder_pd_ ? other.reorder_pd_->clone() : nullptr);
        }
    };

    virtual status_t init(engine_t *engine) override {
        if (pd()->reorder_pd_)
            pd()->reorder_pd_->create_primitive(reorder_, engine);
        diff_ss_kernel_ = new diff_ss_kernel_t(pd());
        diff_data_kernel_ = new diff_data_kernel_t(pd());
        return status::success;
    }

    jit_uni_layer_normalization_bwd_t(const pd_t *apd) : primitive_t(apd) {}

    ~jit_uni_layer_normalization_bwd_t() {
        delete diff_ss_kernel_;
        delete diff_data_kernel_;
    }

    void reorder_stat(const exec_ctx_t &ctx, engine_t *engine,
            const memory_arg_t &in, const memory_arg_t &out) const {
        using namespace memory_tracking::names;
        exec_args_t r_args;
        r_args[DNNL_ARG_SRC] = in;
        r_args[DNNL_ARG_DST] = out;
        exec_ctx_t r_ctx(ctx.stream(), std::move(r_args));

        nested_scratchpad_t ns(ctx, key_nested, reorder_);
        r_ctx.set_scratchpad_grantor(ns.grantor());
        reorder_->execute(r_ctx);
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        using namespace memory_tracking::names;
        /* LN supports arbitrary layout for input/output statistics.
         * For best performance we compute LN with statistics in the same format
         * as data tensor (i.e. data in abcd, stats in abc) and user's
         * input/output statistics are reordered if necessary */

        if (reorder_) {
            engine_t *engine = ctx.stream()->engine();
            auto scratchpad = ctx.get_scratchpad_grantor();
            auto mean_handle
                    = scratchpad.template get<void>(key_lnorm_tmp_mean);
            auto variance_handle
                    = scratchpad.template get<void>(key_lnorm_tmp_var);
            memory_t mean(engine, &(pd()->reordered_stat_md_),
                    memory_flags_t::use_runtime_ptr, mean_handle);
            memory_t variance(engine, &(pd()->reordered_stat_md_),
                    memory_flags_t::use_runtime_ptr, variance_handle);
            reorder_stat(
                    ctx, engine, ctx.args().at(DNNL_ARG_MEAN), {&mean, false});
            reorder_stat(ctx, engine, ctx.args().at(DNNL_ARG_VARIANCE),
                    {&variance, false});
        }

        execute_backward(ctx);
        return status::success;
    }

private:
    void execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> reorder_;
    diff_ss_kernel_t *diff_ss_kernel_;
    diff_data_kernel_t *diff_data_kernel_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
