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
#include "jit_uni_dw_convolution.hpp"

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
            , rtus_()
            , jcp_dw_()
            , dw_conv_pd_(nullptr) {}

        pd_t(const pd_t &other) : cpu_convolution_fwd_pd_t(other) {
            jcp_ = other.jcp_;
            rtus_ = other.rtus_;
            jcp_dw_ = other.jcp_dw_;
            if (other.dw_conv_pd_)
                dw_conv_pd_ = static_cast<decltype(dw_conv_pd_)>(
                        other.dw_conv_pd_->clone());
        }

        pd_t &operator=(const pd_t &other) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(other);
            cpu_convolution_fwd_pd_t::operator=(other);
            jcp_ = other.jcp_;
            rtus_ = other.rtus_;
            jcp_dw_ = other.jcp_dw_;
            if (dw_conv_pd_) delete dw_conv_pd_;
            if (dw_conv_pd_)
                dw_conv_pd_ = static_cast<decltype(dw_conv_pd_)>(
                        other.dw_conv_pd_->clone());

            return *this;
        }

        ~pd_t() {
            if (dw_conv_pd_) {
                delete dw_conv_pd_;
                dw_conv_pd_ = nullptr;
            };
        }
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

            if (jcp_.with_dw_conv) {
                status = depthwise_po_init();
                if (status != status::success) return status;
            }

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_core_bf16_1x1_conv_kernel::init_scratchpad(
                    scratchpad, jcp_);

            rtus_prepare_space_info(this, scratchpad);

            return status::success;
        }

        virtual const memory_desc_t *dst_md(int index = 0) const override {
            return jcp_.with_dw_conv ? dw_conv_pd_->dst_md(index) : &dst_md_;
        }

        virtual const memory_desc_t *arg_md(int index = 0) const override {
            if (jcp_.with_dw_conv) {
                switch (index) {
                    case DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS:
                        return dw_conv_pd_->weights_md(0);
                    case DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS:
                        return dw_conv_pd_->weights_md(1);
                    default: break;
                }
            }
            return convolution_fwd_pd_t::arg_md(index);
        }

        virtual arg_usage_t arg_usage(int arg) const override {

            if (utils::one_of(arg, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS,
                        DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS))
                return arg_usage_t::input;

            return convolution_fwd_pd_t::arg_usage(arg);
        }

        jit_1x1_conv_conf_t jcp_;
        reduce_to_unit_stride_t rtus_;
        jit_conv_conf_t jcp_dw_;
        cpu_convolution_fwd_pd_t *dw_conv_pd_ = nullptr;

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c);
            auto wei_tag = utils::pick(2 * ndims() - 6 + with_groups(),
                    OIw8i16o2i, gOIw8i16o2i, OIhw8i16o2i, gOIhw8i16o2i,
                    OIdhw8i16o2i, gOIdhw8i16o2i);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

        status_t depthwise_po_init() {
            using namespace memory_tracking;
            auto &jcp_1x1 = jcp_;
            auto &jcp_dw = jcp_dw_;
            auto &attr_1x1 = attr_;
            const auto &src_md = dst_md_;
            const memory_desc_wrapper src_d(src_md);
            const auto nthr = dnnl_get_max_threads();
            auto l2_cache = get_cache_size(2, true) * nthr;

            bool ok = true && is_fwd()
                    && (attr_1x1.post_ops_.find(primitive_kind::sum) == -1)
                    // TODO: Below may be further tuned.
                    && (l2_cache * 2 < src_d.size())
                    // load_grp_count check can be redundant due to l2 check
                    // above. Adding it explicitly as the current driver doesn't
                    // work if this condition fails.
                    && (jcp_1x1.load_grp_count < 2);
            if (!ok) return status::unimplemented;

            int dw_po_index
                    = attr_1x1.post_ops_.find(primitive_kind::convolution);

            auto status = get_depthwise_primitive_desc(
                    (primitive_desc_t **)&dw_conv_pd_, src_md, attr_1x1,
                    engine(), dw_po_index);
            if (status != status::success) return status::unimplemented;

            auto dw_dst_dt = dw_conv_pd_->dst_md()->data_type;
            // Check if the fusable_pd matches with the primitive_desc returned
            // by dnnl_primitive_desc_create(). If it doesn't match, then there
            // probably exists a more optimized version of depthwise convolution
            // than the fusable_pd. In this case, fallback to reference fusion.
#define CASE(sdt, ddt) \
    case ddt: { \
        auto fusable_pd \
                = new typename jit_uni_dw_convolution_fwd_t<avx512_core, sdt, \
                        ddt>::pd_t(engine(), dw_conv_pd_->desc(), \
                        dw_conv_pd_->attr(), nullptr); \
        if (fusable_pd->init() != status::success) { \
            delete fusable_pd; \
            return status::unimplemented; \
        } \
        auto key1 = primitive_hashing::key_t(fusable_pd, 0); \
        auto key2 = primitive_hashing::key_t(dw_conv_pd_, 0); \
        if (!(key1 == key2)) { \
            delete fusable_pd; \
            return status::unimplemented; \
        } \
        jcp_dw = static_cast<decltype(fusable_pd)>(dw_conv_pd_)->jcp_; \
        break; \
    }
            if (jcp_1x1.dst_dt == data_type::bf16) {
                switch (dw_dst_dt) {
                    CASE(data_type::bf16, data_type::bf16);
                    CASE(data_type::bf16, data_type::f32);
                    default: return status::unimplemented;
                }
            } else
                return status::unimplemented;
#undef CASE

            ok = true
                    && (dnnl_memory_desc_equal(&src_md, dw_conv_pd_->src_md(0)))
                    && (jcp_1x1.oc_without_padding % jcp_1x1.oc_block == 0)
                    && IMPLICATION(
                            jcp_dw.ow_block, jcp_dw.ow_block == jcp_dw.ow);
            if (!ok) return status::unimplemented;

            assert(dw_conv_pd_->dst_md(0)->format_kind != format_kind::any);
            assert(dw_conv_pd_->weights_md(0)->format_kind != format_kind::any);
            assert(IMPLICATION(
                    dw_conv_pd_->weights_md(1)->data_type != data_type::undef,
                    dw_conv_pd_->weights_md(1)->format_kind
                            != format_kind::any));

            jcp_dw.is_fused_conv = true;
            // TODO: Support/experiment arbitary oc_work in dw conv.
            // Until then we keep ch_work perfectly divisible.
            while (jcp_1x1.nb_load % jcp_1x1.nb_load_blocking != 0)
                --jcp_1x1.nb_load_blocking;
            jcp_1x1.nb_load_blocking_max = jcp_1x1.nb_load_blocking;

            while (jcp_1x1.nb_load_blocking % jcp_dw.nb_ch_blocking != 0)
                --jcp_dw.nb_ch_blocking;

            jcp_dw.dw_conv_buffer_oc
                    = jcp_1x1.nb_load_blocking * jcp_1x1.oc_block;
            jcp_1x1.bcast_loop_output_step
                    = jcp_1x1.ur * jcp_1x1.load_block * jcp_1x1.typesize_out;

            registrar_t scratchpad(scratchpad_registry_);
            registrar_t dw_scratchpad(scratchpad, names::prefix_fusion);

            size_t dw_conv_buffer_size_ = (size_t)nthr * jcp_dw.kh * jcp_1x1.ow
                    * jcp_dw.dw_conv_buffer_oc
                    * types::data_type_size(jcp_dw.dst_dt);
            assert(dw_conv_buffer_size_);
            dw_scratchpad.book(memory_tracking::names::key_fusion_inout_buffer,
                    dw_conv_buffer_size_);

            dw_conv_kernel_t::init_scratchpad(dw_scratchpad, jcp_dw);

            return status::success;
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);
    jit_avx512_core_bf16_1x1_convolution_fwd_t(const pd_t *apd)
        : primitive_impl_t(apd), kernel_(nullptr), rtus_driver_(nullptr) {
        kernel_ = new jit_avx512_core_bf16_1x1_conv_kernel(
                pd()->jcp_, *pd()->attr());

        if (pd()->jcp_.with_dw_conv) {
            kernel_dw_ = new dw_conv_kernel_t(pd()->jcp_dw_);
        }

        init_rtus_driver<avx512_common>(this);
    }
    ~jit_avx512_core_bf16_1x1_convolution_fwd_t() {
        delete kernel_;
        if (kernel_dw_) { delete kernel_dw_; }
        delete rtus_driver_;
    }

    typedef typename prec_traits<data_type::bf16>::type src_data_t;
    typedef typename prec_traits<data_type::bf16>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    // Note: In case of fused depthwise convolution, the final output datatype
    // may not be dst_data_t.
    typedef typename prec_traits<dst_type>::type dw_wei_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    void execute_forward_thr(const int ithr, const int nthr,
            const src_data_t *src, const wei_data_t *weights, const char *bias,
            const dw_wei_data_t *weights_dw, const float *bias_dw,
            const char *dst,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    jit_avx512_core_bf16_1x1_conv_kernel *kernel_;
    rtus_driver_t<avx512_common> *rtus_driver_;
    using dw_conv_kernel_t
            = jit_uni_dw_conv_fwd_kernel<avx512_core, data_type::bf16>;
    dw_conv_kernel_t *kernel_dw_ = nullptr;
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
            jit_avx512_core_bf16_1x1_conv_kernel::init_scratchpad(
                    scratchpad, jcp_);
            rtus_prepare_space_info(this, scratchpad);

            return status::success;
        }

        // TODO (Roma): structs conf header cleanup
        jit_1x1_conv_conf_t jcp_;
        reduce_to_unit_stride_t rtus_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            auto dat_tag = utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c);
            auto wei_tag = utils::pick(2 * ndims() - 6 + with_groups(),
                    IOw8o16i2o, gIOw8o16i2o, IOhw8o16i2o, gIOhw8o16i2o,
                    IOdhw8o16i2o, gIOdhw8o16i2o);

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

            auto dat_tag = utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c);
            auto wei_tag = utils::pick(2 * ndims() - 6 + with_groups(),
                    OIw16i16o, gOIw16i16o, OIhw16i16o, gOIhw16i16o, OIdhw16i16o,
                    gOIdhw16i16o);

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
