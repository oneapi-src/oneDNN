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

#ifndef CPU_JIT_AVX512_CORE_BF16_1X1_CONVOLUTION_HPP
#define CPU_JIT_AVX512_CORE_BF16_1X1_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_reducer.hpp"
#include "dnnl_thread.hpp"
#include "utils.hpp"

#include "jit_avx512_core_bf16_1x1_conv_kernel.hpp"
#include "jit_transpose_src_utils.hpp"
#include "jit_uni_1x1_conv_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <impl::data_type_t dst_type>
struct jit_avx512_core_bf16_1x1_convolution_fwd_t : public primitive_impl_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
            , rtus_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_bf16_1x1:", jcp_.isa, ""),
                jit_avx512_core_bf16_1x1_convolution_fwd_t);

        status_t init() {
            bool ok = true && mayiuse(avx512_core) && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(data_type::bf16, data_type::bf16,
                            data_type::undef, dst_type, data_type::undef)
                    && IMPLICATION(with_bias(),
                            utils::one_of(weights_md(1)->data_type,
                                    data_type::f32, data_type::bf16))
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops)
                    && !has_zero_dim_memory() && set_default_formats();

            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = desc();
            const memory_desc_t *src_d = src_md();
            rtus_prepare(this, conv_d, src_d, dst_md());

            status_t status = jit_avx512_core_bf16_1x1_conv_kernel::init_conf(
                    jcp_, *conv_d, *src_d, *weights_md(), *dst_md(), *attr(),
                    dnnl_get_max_threads(), rtus_.reduce_src_);
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_core_bf16_1x1_conv_kernel::init_scratchpad(
                    scratchpad, jcp_);

            rtus_prepare_space_info(this, scratchpad);

            return status::success;
        }

        jit_1x1_conv_conf_t jcp_;
        reduce_to_unit_stride_t rtus_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nCw16c, nChw16c);
            auto wei_tag = utils::pick(2 * ndims() - 6 + with_groups(),
                    OIw8i16o2i, gOIw8i16o2i, OIhw8i16o2i, gOIhw8i16o2i);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);
    jit_avx512_core_bf16_1x1_convolution_fwd_t(const pd_t *apd)
        : primitive_impl_t(apd), kernel_(nullptr), rtus_driver_(nullptr) {
        kernel_ = new jit_avx512_core_bf16_1x1_conv_kernel(
                pd()->jcp_, *pd()->attr());
        init_rtus_driver<avx512_common>(this);
    }
    ~jit_avx512_core_bf16_1x1_convolution_fwd_t() {
        delete kernel_;
        delete rtus_driver_;
    }

    typedef typename prec_traits<data_type::bf16>::type src_data_t;
    typedef typename prec_traits<data_type::bf16>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    void execute_forward_thr(const int ithr, const int nthr,
            const src_data_t *src, const wei_data_t *weights, const char *bias,
            dst_data_t *dst,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    jit_avx512_core_bf16_1x1_conv_kernel *kernel_;

    rtus_driver_t<avx512_common> *rtus_driver_;
};

template <impl::data_type_t diff_src_type>
struct jit_avx512_core_bf16_1x1_convolution_bwd_data_t
    : public primitive_impl_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
            , rtus_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_bf16_1x1:", jcp_.isa, ""),
                jit_avx512_core_bf16_1x1_convolution_bwd_data_t);

        status_t init() {
            bool ok = true && mayiuse(avx512_core) && is_bwd_d()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(diff_src_type, data_type::bf16,
                            data_type::undef, data_type::bf16, data_type::undef)
                    && attr()->has_default_values() && !has_zero_dim_memory()
                    && set_default_formats();
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = desc();
            const memory_desc_t *diff_src_d = diff_src_md();
            rtus_prepare(this, conv_d, diff_src_d, diff_dst_md());

            status_t status = jit_avx512_core_bf16_1x1_conv_kernel::init_conf(
                    jcp_, *conv_d, *diff_src_d, *weights_md(), *diff_dst_md(),
                    *attr(), dnnl_get_max_threads(), rtus_.reduce_src_);
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            rtus_prepare_space_info(this, scratchpad);

            return status::success;
        }

        // TODO (Roma): structs conf header cleanup
        jit_1x1_conv_conf_t jcp_;
        reduce_to_unit_stride_t rtus_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            auto dat_tag = utils::pick(ndims() - 3, nCw16c, nChw16c);
            auto wei_tag = utils::pick(2 * ndims() - 6 + with_groups(),
                    IOw8o16i2o, gIOw8o16i2o, IOhw8o16i2o, gIOhw8o16i2o);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);

    jit_avx512_core_bf16_1x1_convolution_bwd_data_t(const pd_t *apd)
        : primitive_impl_t(apd), kernel_(nullptr), rtus_driver_(nullptr) {
        kernel_ = new jit_avx512_core_bf16_1x1_conv_kernel(
                pd()->jcp_, *pd()->attr());
        init_rtus_driver<avx512_common>(this);
    }
    ~jit_avx512_core_bf16_1x1_convolution_bwd_data_t() {
        delete kernel_;
        delete rtus_driver_;
    }

    typedef typename prec_traits<data_type::bf16>::type diff_dst_data_t;
    typedef typename prec_traits<data_type::bf16>::type wei_data_t;
    typedef typename prec_traits<diff_src_type>::type diff_src_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_data(ctx);
        return status::success;
    }

private:
    void execute_backward_data(const exec_ctx_t &ctx) const;
    void execute_backward_data_thr(const int, const int,
            const diff_dst_data_t *, const wei_data_t *, diff_src_data_t *,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    jit_avx512_core_bf16_1x1_conv_kernel *kernel_;
    /* reduction to unit stride */
    rtus_driver_t<avx512_common> *rtus_driver_;
};

template <impl::data_type_t diff_weights_type>
struct jit_avx512_core_bf16_1x1_convolution_bwd_weights_t
    : public primitive_impl_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
            , rtus_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_bf16_1x1:", jcp_.isa, ""),
                jit_avx512_core_bf16_1x1_convolution_bwd_weights_t);

        status_t init() {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true && mayiuse(avx512_core) && is_bwd_w()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(data_type::bf16, diff_weights_type,
                            data_type::undef, data_type::bf16, data_type::undef)
                    && IMPLICATION(with_bias(),
                            utils::one_of(diff_weights_md(1)->data_type,
                                    data_type::f32, data_type::bf16))
                    && attr()->has_default_values() && !has_zero_dim_memory()
                    && set_default_formats();
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = desc();
            const memory_desc_t *src_d = src_md();
            rtus_prepare(this, conv_d, src_d, diff_dst_md());

            status_t status = jit_avx512_core_bf16_1x1_conv_kernel::init_conf(
                    jcp_, *conv_d, *src_d, *diff_weights_md(0), *diff_dst_md(),
                    *attr(), dnnl_get_max_threads(), rtus_.reduce_src_);
            if (status != status::success) return status;

            init_balancers();

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_core_bf16_1x1_conv_kernel::init_scratchpad(
                    scratchpad, jcp_);

            auto reducer_bia_scratchpad = memory_tracking::registrar_t(
                    scratchpad, memory_tracking::names::prefix_reducer_bia);
            reducer_bia_conf_.init_scratchpad(reducer_bia_scratchpad);

            rtus_prepare_space_info(this, scratchpad);

            return status::success;
        }

        // TODO (Roma): structs conf header cleanup
        jit_1x1_conv_conf_t jcp_;
        cpu_reducer_t<data_type::f32>::conf_t reducer_bia_conf_;
        reduce_to_unit_stride_t rtus_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            auto dat_tag = utils::pick(ndims() - 3, nCw16c, nChw16c);
            auto wei_tag = utils::pick(2 * ndims() - 6 + with_groups(),
                    OIw16i16o, gOIw16i16o, OIhw16i16o, gOIhw16i16o);

            bool ok = set_default_formats_common(dat_tag, wei_tag, dat_tag);
            return ok;
        }

    private:
        void init_balancers() {
            const size_t max_buffer_size = jcp_.nthr * 3 * 5 * 5 * 16 * 16;
            if (with_bias()) {
                reducer_bia_conf_.init(reduce_balancer_t(jcp_.nthr,
                        jcp_.oc_block, jcp_.ngroups * jcp_.nb_load, jcp_.mb,
                        max_buffer_size, true));
            }
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);

    jit_avx512_core_bf16_1x1_convolution_bwd_weights_t(const pd_t *apd);

    ~jit_avx512_core_bf16_1x1_convolution_bwd_weights_t() {
        delete acc_ker_;
        delete kernel_;
        delete reducer_bias_;
        delete rtus_driver_;
        delete tr_reorder_;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_weights(ctx);
        return status::success;
    }

    typedef typename prec_traits<data_type::bf16>::type src_data_t;
    typedef typename prec_traits<data_type::bf16>::type diff_dst_data_t;

    typedef typename prec_traits<diff_weights_type>::type diff_wei_data_t;

private:
    void execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    jit_avx512_core_bf16_1x1_conv_kernel *kernel_;
    cpu_accumulator_1d_t<data_type::f32> *acc_ker_;
    cpu_reducer_t<data_type::f32> *reducer_bias_;

    /* reduction to unit stride */
    rtus_driver_t<avx512_common> *rtus_driver_;

    jit_avx512_core_bf16_reorder_s16c_to_S16c2s_t *tr_reorder_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
