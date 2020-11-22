/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#ifndef CPU_GEMM_CONVOLUTION_HPP
#define CPU_GEMM_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"

#include "cpu/gemm/gemm.hpp"
#include "cpu/gemm_convolution_utils.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/cpu_convolution_pd.hpp"

#include "ref_depthwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct gemm_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(
                GEMM_IMPL_STR, gemm_convolution_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            bool ok = true && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(data_type::f32, data_type::f32,
                            data_type::f32, data_type::f32, data_type::f32)
                    && !has_zero_dim_memory()
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops,
                            data_type::f32)
                    && post_ops_ok();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_md_, weights_md_, dst_md_, bias_md_, *attr(),
                    dnnl_get_max_threads());
        }

        conv_gemm_conf_t jcp_;

    protected:
        bool post_ops_ok() const {
            using namespace dnnl::impl::primitive_kind;
            auto const &po = attr()->post_ops_;

            auto all_post_ops_supported = [&]() {
                bool ok = true;

                for (int i = 0; i < po.len(); i++) {
                    ok = ok && utils::one_of(po.entry_[i].kind, sum, eltwise, depthwise, quantization);
                }
                return ok;
            };
            auto contain = [&](dnnl::impl::primitive_kind_t kind) { return po.find(kind) != -1; };
            auto position = [&](dnnl::impl::primitive_kind_t kind) { return po.find(kind); };
            auto count = [&](dnnl::impl::primitive_kind_t kind) { return po.count(kind); };

            return all_post_ops_supported() &&
                   count(primitive_kind::sum) <= 1 &&
                   IMPLICATION(contain(primitive_kind::sum), position(primitive_kind::sum) == 0);
        }
    };

    gemm_convolution_fwd_t(const pd_t *apd)
        : primitive_t(apd), eltwise_(nullptr) {}

    status_t init(engine_t *engine) override {
        const auto &post_ops = pd()->attr()->post_ops_;
        const data_t one = 1.0, zero = 0.0;
        beta_ = post_ops.find(primitive_kind::sum) >= 0 ? one : zero;

        bool has_bias = pd()->with_bias(),
                has_post_ops = post_ops.len() > 0,
                has_scale = !pd()->attr()->output_scales_.has_default_values();
        postops_in_ip_ = has_bias || has_post_ops || has_scale;

        CHECK(safe_ptr_assign(pp_kernel_, pp_kernel_t::create(pd(), pd()->jcp_)));
        return (pp_kernel_) ? pp_kernel_->create_kernel() : status::success;
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        bool is_nspc = pd()->jcp_.is_nspc;
        return is_nspc ? execute_forward_nspc(ctx) : execute_forward_ncsp(ctx);
    }

private:
    status_t execute_forward_ncsp(const exec_ctx_t &ctx) const;
    status_t execute_forward_nspc(const exec_ctx_t &ctx) const;
    status_t execute_forward_thr_nspc(const int ithr, const int nthr,
            const data_t *src_base, const data_t *wei_base,
            const data_t *bia_base, data_t *dst_base,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    using pp_kernel_t = gemm_convolution_utils::pp_kernel_t;
    std::unique_ptr<pp_kernel_t> pp_kernel_;
    bool postops_in_ip_;
    data_t beta_;

    // todo: antonvor: remove "eltwise_" if we don't use nspc
    std::unique_ptr<ref_eltwise_scalar_fwd_t> eltwise_;
};

struct gemm_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_convolution_bwd_data_t,
                USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_data
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(data_type::f32, data_type::f32,
                            data_type::undef, data_type::f32, data_type::f32)
                    && !has_zero_dim_memory()
                    && is_supported_post_ops();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), diff_src_md_, weights_md_, diff_dst_md_, bias_md_,
                    *attr(), dnnl_get_max_threads());
        }

        conv_gemm_conf_t jcp_;

    protected:
        virtual bool is_supported_post_ops() const {
            const auto &p = this->attr()->post_ops_;
            if (p.len() > 1)
                return false;

            auto all_post_ops_supported = [&]() {
                bool ok = true;

                for (int i = 0; i < p.len(); i++) {
                    ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::depthwise);
                }
                return ok;
            };

            return all_post_ops_supported();
        }
    };


    gemm_convolution_bwd_data_t(const pd_t *apd) : primitive_t(apd) {
        const auto &post_ops = pd()->attr()->post_ops_;
        for (int i = 0; i < post_ops.len(); i++) {
            auto &post_op = post_ops.entry_[i];
            if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(new ref_depthwise_scalar_fwd_t(post_op.depthwise.alg));
            }
        }
    }

    ~gemm_convolution_bwd_data_t() {
        for (auto inj : depthwise_injectors)
            delete inj;
        depthwise_injectors.clear();
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        bool is_nspc = pd()->jcp_.is_nspc;
        return is_nspc ? execute_backward_data_nspc(ctx)
                       : execute_backward_data_ncsp(ctx);
    }

private:
    status_t execute_backward_data_nspc(const exec_ctx_t &ctx) const;
    status_t execute_backward_data_ncsp(const exec_ctx_t &ctx) const;
    status_t execute_backward_data_thr_nspc(const int ithr, const int nthr,
            const data_t *diff_dst_base, const data_t *wei_base,
            const data_t *bia_base, data_t *diff_src_base,
            const memory_tracking::grantor_t &scratchpad) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    nstl::vector<ref_depthwise_scalar_fwd_t*> depthwise_injectors;
};

struct gemm_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_convolution_bwd_weights_t,
                USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_weights
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(data_type::f32, data_type::f32,
                            data_type::f32, data_type::f32, data_type::f32)
                    && !has_zero_dim_memory() && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_md_, diff_weights_md_, diff_dst_md_,
                    diff_bias_md_, *attr(), dnnl_get_max_threads());
        }

        conv_gemm_conf_t jcp_;
    };

    gemm_convolution_bwd_weights_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        const bool is_nspc = pd()->jcp_.is_nspc;
        return is_nspc ? execute_backward_weights_nspc(ctx)
                       : execute_backward_weights_ncsp(ctx);
    }

private:
    status_t execute_backward_weights_ncsp(const exec_ctx_t &ctx) const;
    status_t execute_backward_weights_nspc(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
