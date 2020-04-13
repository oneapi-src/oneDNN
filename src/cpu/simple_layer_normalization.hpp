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

#ifndef CPU_SIMPLE_LAYER_NORMALIZATION_HPP
#define CPU_SIMPLE_LAYER_NORMALIZATION_HPP

#include <memory>

#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/reorder_pd.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"

#include "cpu/simple_layer_normalization_kernels.hpp"

#include "cpu/cpu_layer_normalization_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct simple_layer_normalization_fwd_t : public primitive_t {
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

        DECLARE_COMMON_PD_T("simple_layer_normalization:any",
                simple_layer_normalization_fwd_t);

        status_t init(engine_t *engine);

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
        stat_kernel_.reset(lnorm_utils::statistics_kernel_t::create(pd()));
        data_kernel_.reset(lnorm_utils::data_kernel_t::create(pd()));
        return status::success;
    }

    simple_layer_normalization_fwd_t(const pd_t *apd) : primitive_t(apd) {}

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

    std::unique_ptr<lnorm_utils::statistics_kernel_t> stat_kernel_;
    std::unique_ptr<lnorm_utils::data_kernel_t> data_kernel_;
    std::shared_ptr<primitive_t> reorder_;
};

struct simple_layer_normalization_bwd_t : public primitive_t {
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

        DECLARE_COMMON_PD_T("simple_layer_normalization:any",
                simple_layer_normalization_bwd_t);

        status_t init(engine_t *engine);

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
        diff_ss_kernel_.reset(lnorm_utils::diff_ss_kernel_t::create(pd()));
        diff_data_kernel_.reset(lnorm_utils::diff_data_kernel_t::create(pd()));
        return status::success;
    }

    simple_layer_normalization_bwd_t(const pd_t *apd) : primitive_t(apd) {}

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

    std::unique_ptr<lnorm_utils::diff_ss_kernel_t> diff_ss_kernel_;
    std::unique_ptr<lnorm_utils::diff_data_kernel_t> diff_data_kernel_;
    std::shared_ptr<primitive_t> reorder_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
