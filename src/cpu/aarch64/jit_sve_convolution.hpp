/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
* Copyright 2020-2024 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_SVE_CONVOLUTION_HPP
#define CPU_AARCH64_JIT_SVE_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_barrier.hpp"
#include "cpu/aarch64/cpu_reducer.hpp"
#include "cpu/cpu_convolution_pd.hpp"

#include "cpu/aarch64/jit_sve_conv_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <impl::data_type_t src_type, impl::data_type_t wei_type = src_type,
        impl::data_type_t dst_type = src_type, cpu_isa_t isa = isa_undef>
struct jit_sve_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_sve_convolution_fwd_t);

        status_t init(engine_t *engine) {
            bool ok = true && mayiuse(isa) && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(src_type, wei_type, dst_type, dst_type,
                            data_type::undef)
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops, dst_type)
                    && !has_zero_dim_memory();
            if (!ok) return status::unimplemented;

            status_t status = jit_sve_conv_fwd_kernel<isa>::init_conf(jcp_,
                    *desc(), src_md_, weights_md_, dst_md_, bias_md_, *attr(),
                    dnnl_get_max_threads());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_sve_conv_fwd_kernel<isa>::init_scratchpad(scratchpad, jcp_);

            return status;
        }

        jit_conv_conf_t jcp_ = utils::zero<decltype(jcp_)>();
    };

    jit_sve_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    using src_data_t = typename prec_traits_t<src_type>::type;
    using wei_data_t = typename prec_traits_t<wei_type>::type;
    using dst_data_t = typename prec_traits_t<dst_type>::type;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_sve_conv_fwd_kernel<isa>(pd()->jcp_, *pd()->attr())));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->ndims() == 3)
            execute_forward_1d(ctx);
        else if (pd()->ndims() == 4)
            execute_forward_2d(ctx);
        else if (pd()->ndims() == 5)
            execute_forward_3d(ctx);
        else
            assert(false);

        if (pd()->wants_zero_pad_dst()) ctx.zero_pad_output(DNNL_ARG_DST);

        return status::success;
    }

private:
    void prepare_padded_bias(const dst_data_t *&bias,
            const memory_tracking::grantor_t &scratchpad) const;
    void execute_forward_1d(const exec_ctx_t &ctx) const;
    void execute_forward_2d(const exec_ctx_t &ctx) const;
    void execute_forward_3d(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_sve_conv_fwd_kernel<isa>> kernel_;
};

template <impl::data_type_t diff_dst_type,
        impl::data_type_t wei_type = diff_dst_type,
        impl::data_type_t diff_src_type = diff_dst_type,
        cpu_isa_t isa = isa_undef>
struct jit_sve_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        using cpu_convolution_bwd_data_pd_t::cpu_convolution_bwd_data_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_sve_convolution_bwd_data_t);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_data
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(diff_src_type, wei_type,
                            data_type::undef, diff_dst_type, data_type::undef)
                    && attr()->has_default_values() && !has_zero_dim_memory();
            if (!ok) return status::unimplemented;

            status_t status = jit_sve_conv_bwd_data_kernel_f32<isa>::init_conf(
                    jcp_, *desc(), diff_src_md_, weights_md_, diff_dst_md_,
                    dnnl_get_max_threads());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_sve_conv_bwd_data_kernel_f32<isa>::init_scratchpad(
                    scratchpad, jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_ = utils::zero<decltype(jcp_)>();
    };

    jit_sve_convolution_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    using diff_dst_data_t = typename prec_traits_t<diff_dst_type>::type;
    using wei_data_t = typename prec_traits_t<wei_type>::type;
    using diff_src_data_t = typename prec_traits_t<diff_src_type>::type;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_sve_conv_bwd_data_kernel_f32<isa>(pd()->jcp_)));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->ndims() == 3)
            execute_backward_data_1d(ctx);
        else if (pd()->ndims() == 4)
            execute_backward_data_2d(ctx);
        else if (pd()->ndims() == 5)
            execute_backward_data_3d(ctx);
        else
            assert(false);
        return status::success;
    }

private:
    void execute_backward_data_1d(const exec_ctx_t &ctx) const;
    void execute_backward_data_2d(const exec_ctx_t &ctx) const;
    void execute_backward_data_3d(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_sve_conv_bwd_data_kernel_f32<isa>> kernel_;
};

template <impl::data_type_t src_type,
        impl::data_type_t diff_dst_type = src_type,
        impl::data_type_t diff_weights_type = src_type,
        cpu_isa_t isa = isa_undef>
struct jit_sve_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        using cpu_convolution_bwd_weights_pd_t::
                cpu_convolution_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_sve_convolution_bwd_weights_t);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_weights
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(src_type, diff_weights_type,
                            diff_weights_type, diff_dst_type, data_type::undef)
                    && attr()->has_default_values() && !has_zero_dim_memory();
            if (!ok) return status::unimplemented;

            status_t status
                    = jit_sve_conv_bwd_weights_kernel_f32<isa>::init_conf(jcp_,
                            *desc(), src_md_, diff_weights_md_, diff_bias_md_,
                            diff_dst_md_, dnnl_get_max_threads());
            if (status != status::success) return status;

            init_balancers();

            auto scratchpad = scratchpad_registry().registrar();
            jit_sve_conv_bwd_weights_kernel_f32<isa>::init_scratchpad(
                    scratchpad, jcp_);

            auto reducer_bia_scratchpad = memory_tracking::registrar_t(
                    scratchpad, memory_tracking::names::prefix_reducer_bia);
            reducer_bia_conf_.init_scratchpad(reducer_bia_scratchpad);

            return status;
        }

        jit_conv_conf_t jcp_ = utils::zero<decltype(jcp_)>();
        typename cpu_reducer_t<diff_weights_type, isa>::conf_t
                reducer_bia_conf_;

    private:
        void init_balancers() {
            const size_t max_buffer_size = jcp_.nthr * 3 * 5 * 5 * 16 * 16;
            if (with_bias()) {
                reducer_bia_conf_.init(reduce_balancer_t(jcp_.nthr,
                        jcp_.oc_block, jcp_.ngroups * jcp_.nb_oc, jcp_.mb,
                        max_buffer_size, true));
            }
        }
    };

    jit_sve_convolution_bwd_weights_t(const pd_t *apd) : primitive_t(apd) {}

    using src_data_t = typename prec_traits_t<src_type>::type;
    using diff_dst_data_t = typename prec_traits_t<diff_dst_type>::type;
    using diff_weights_data_t = typename prec_traits_t<diff_weights_type>::type;

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_weights(ctx);
        return status::success;
    }

private:
    void execute_backward_weights(const exec_ctx_t &ctx) const;
    void prepare_scratchpad_data(const exec_ctx_t &ctx) const;
    struct thread_info_t;
    void compute_diff_weights(const thread_info_t *) const;
    void compute_diff_weights_2d(const thread_info_t *) const;
    void compute_diff_weights_3d(const thread_info_t *) const;
    void reduce_diff_weights(const thread_info_t *) const;
    void reduce_diff_weights_3d(const thread_info_t *) const;
    void compute_diff_bias(const thread_info_t *) const;
    void reduce_diff_bias(const thread_info_t *) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    int nthr_, nthr_mb_, nthr_g_, nthr_oc_b_, nthr_ic_b_;

    jit_sve_conv_bwd_weights_kernel_f32<isa> *kernel_;
    cpu_accumulator_1d_t<diff_weights_type, isa> *acc_ker_;
    cpu_reducer_t<diff_weights_type, isa> *reducer_bias_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
