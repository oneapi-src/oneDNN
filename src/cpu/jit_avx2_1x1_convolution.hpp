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

#ifndef CPU_JIT_AVX2_1X1_CONVOLUTION_HPP
#define CPU_JIT_AVX2_1X1_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "memory_tracking.hpp"
#include "utils.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_reducer.hpp"

#include "jit_avx2_1x1_conv_kernel_f32.hpp"
#include "jit_uni_1x1_conv_utils.hpp"
#include "jit_uni_dw_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct jit_avx2_1x1_convolution_fwd_t : public primitive_impl_t {
    // TODO: (Roma) Code duplication duplication! Remove with templates
    //              (maybe...)!
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
            , rtus_()
            , dw_conv_pd_(nullptr) {}

        pd_t(const pd_t &other) : cpu_convolution_fwd_pd_t(other) {
            jcp_ = other.jcp_;
            rtus_ = other.rtus_;
            if (other.dw_conv_pd_)
                dw_conv_pd_ = static_cast<decltype(dw_conv_pd_)>(
                        other.dw_conv_pd_->clone());
        }

        pd_t &operator=(const pd_t &other) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(other);
            cpu_convolution_fwd_pd_t::operator=(other);
            jcp_ = other.jcp_;
            rtus_ = other.rtus_;
            if (dw_conv_pd_) delete dw_conv_pd_;
            if (other.dw_conv_pd_)
                dw_conv_pd_ = static_cast<decltype(dw_conv_pd_)>(
                        other.dw_conv_pd_->clone());

            return *this;
        }

        ~pd_t() {
            if (dw_conv_pd_) delete dw_conv_pd_;
        }

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_1x1:", jcp_.isa, ""),
                jit_avx2_1x1_convolution_fwd_t);

        status_t init() {
            bool ok = true && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(data_type::f32, data_type::f32,
                            data_type::f32, data_type::f32, data_type::f32)
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops)
                    && !has_zero_dim_memory() && set_default_formats();
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = desc();
            const memory_desc_t *src_d = src_md();
            rtus_prepare(this, conv_d, src_d, dst_md());

            status_t status = jit_avx2_1x1_conv_kernel_f32::init_conf(
                    jcp_, *conv_d, *src_d, *weights_md(), *dst_md(), *attr());
            if (status != status::success) return status;

            if (jcp_.with_dw_conv) {
                status = depthwise_po_init();
                if (status != status::success) return status;
            }

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx2_1x1_conv_kernel_f32::init_scratchpad(scratchpad, jcp_);

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
        jit_avx2_dw_convolution_fwd_t::pd_t *dw_conv_pd_ = nullptr;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            auto dat_tag = utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, gOIw8i8o, gOIhw8i8o, gOIdhw8i8o)
                    : utils::pick(ndims() - 3, OIw8i8o, OIhw8i8o, OIdhw8i8o);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

        status_t depthwise_po_init() {

            using namespace memory_tracking;
            auto &jcp_1x1 = jcp_;
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

            auto fusable_pd = new jit_avx2_dw_convolution_fwd_t::pd_t(engine(),
                    dw_conv_pd_->desc(), dw_conv_pd_->attr(), nullptr);
            status = fusable_pd->init();
            if (status != status::success) { return status; }

            // Check if the fusable_pd matches with the primitive_desc returned
            // by dnnl_primitive_desc_create(). If it doesn't match, then there
            // probably exists a more optimized version of depthwise convolution
            // than the fusable_pd. In this case, fallback to reference fusion.
            auto key1 = primitive_hashing::key_t(fusable_pd, 0);
            auto key2 = primitive_hashing::key_t(dw_conv_pd_, 0);
            if (!(key1 == key2)) return status::unimplemented;

            auto &jcp_dw = dw_conv_pd_->jcp_;

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
            // Until then we keep oc_work perfectly divisible.
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

    jit_avx2_1x1_convolution_fwd_t(const pd_t *apd)
        : primitive_impl_t(apd), kernel_(nullptr), rtus_driver_(nullptr) {
        kernel_ = new jit_avx2_1x1_conv_kernel_f32(pd()->jcp_, *pd()->attr());

        if (pd()->jcp_.with_dw_conv) {
            kernel_dw_ = new dw_conv_kernel_t(pd()->dw_conv_pd_->jcp_);
        }

        init_rtus_driver<avx2>(this);
    }

    ~jit_avx2_1x1_convolution_fwd_t() {
        delete kernel_;
        if (kernel_dw_) { delete kernel_dw_; }
        delete rtus_driver_;
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    void execute_forward_thr(const int ithr, const int nthr, const data_t *src,
            const data_t *weights, const data_t *bias, const data_t *weights_dw,
            const data_t *bias_dw, data_t *dst,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    jit_avx2_1x1_conv_kernel_f32 *kernel_;
    rtus_driver_t<avx2> *rtus_driver_;
    using dw_conv_kernel_t = jit_uni_dw_conv_fwd_kernel<avx2, data_type::f32>;
    dw_conv_kernel_t *kernel_dw_ = nullptr;
};

struct jit_avx2_1x1_convolution_bwd_data_t : public primitive_impl_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
            , rtus_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_1x1:", avx2, ""),
                jit_avx2_1x1_convolution_bwd_data_t);

        status_t init() {
            bool ok = true && desc()->prop_kind == prop_kind::backward_data
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(data_type::f32, data_type::f32,
                            data_type::undef, data_type::f32, data_type::f32)
                    && attr()->has_default_values() && !has_zero_dim_memory()
                    && set_default_formats();
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = desc();
            const memory_desc_t *diff_src_d = diff_src_md();
            rtus_prepare(this, conv_d, diff_src_d, diff_dst_md());

            status_t status = jit_avx2_1x1_conv_kernel_f32::init_conf(jcp_,
                    *conv_d, *diff_src_d, *weights_md(), *diff_dst_md(),
                    *attr());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx2_1x1_conv_kernel_f32::init_scratchpad(scratchpad, jcp_);

            rtus_prepare_space_info(this, scratchpad);

            return status::success;
        }

        jit_1x1_conv_conf_t jcp_;
        reduce_to_unit_stride_t rtus_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            auto dat_tag = utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, gOIw8o8i, gOIhw8o8i, gOIdhw8o8i)
                    : utils::pick(ndims() - 3, OIw8o8i, OIhw8o8i, OIdhw8o8i);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);

    jit_avx2_1x1_convolution_bwd_data_t(const pd_t *apd)
        : primitive_impl_t(apd), kernel_(nullptr), rtus_driver_(nullptr) {
        kernel_ = new jit_avx2_1x1_conv_kernel_f32(pd()->jcp_, *pd()->attr());
        init_rtus_driver<avx2>(this);
    }

    ~jit_avx2_1x1_convolution_bwd_data_t() {
        delete kernel_;
        delete rtus_driver_;
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_data(ctx);
        return status::success;
    }

private:
    void execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    jit_avx2_1x1_conv_kernel_f32 *kernel_;
    rtus_driver_t<avx2> *rtus_driver_;
};

struct jit_avx2_1x1_convolution_bwd_weights_t : public primitive_impl_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
            , rtus_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_1x1:", avx2, ""),
                jit_avx2_1x1_convolution_bwd_weights_t);

        status_t init() {
            bool ok = true && desc()->prop_kind == prop_kind::backward_weights
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(data_type::f32, data_type::f32,
                            data_type::f32, data_type::f32, data_type::f32)
                    && attr()->has_default_values() && !has_zero_dim_memory()
                    && set_default_formats();
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = desc();
            const memory_desc_t *src_d = src_md();
            rtus_prepare(this, conv_d, src_d, diff_dst_md());

            status_t status = jit_avx2_1x1_conv_kernel_f32::init_conf(jcp_,
                    *conv_d, *src_d, *diff_weights_md(), *diff_dst_md(),
                    *attr());
            if (status != status::success) return status;

            init_balancers();

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx2_1x1_conv_kernel_f32::init_scratchpad(scratchpad, jcp_);

            rtus_prepare_space_info(this, scratchpad);

            auto reducer_bia_scratchpad = memory_tracking::registrar_t(
                    scratchpad, memory_tracking::names::prefix_reducer_bia);
            reducer_bia_conf_.init_scratchpad(reducer_bia_scratchpad);

            auto reducer_wei_scratchpad = memory_tracking::registrar_t(
                    scratchpad, memory_tracking::names::prefix_reducer_wei);
            reducer_wei_conf_.init_scratchpad(reducer_wei_scratchpad);

            return status::success;
        }

        jit_1x1_conv_conf_t jcp_;
        cpu_reducer_t<data_type::f32>::conf_t reducer_bia_conf_;
        cpu_reducer_2d_t<data_type::f32>::conf_t reducer_wei_conf_;
        reduce_to_unit_stride_t rtus_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            auto dat_tag = utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, gOIw8i8o, gOIhw8i8o, gOIdhw8i8o)
                    : utils::pick(ndims() - 3, OIw8i8o, OIhw8i8o, OIdhw8i8o);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

    private:
        void init_balancers() {
            const int ic_block = jcp_.bcast_block;
            const int nb_ic = jcp_.nb_bcast;
            const int nb_ic_blocking = jcp_.nb_bcast_blocking;
            const int bcast_work = utils::div_up(nb_ic, nb_ic_blocking);

            const int oc_block = jcp_.load_block;
            const int nb_oc = jcp_.nb_load;
            const int nb_oc_blocking = jcp_.nb_load_blocking;
            const int load_work = utils::div_up(nb_oc, nb_oc_blocking);

            const int job_size
                    = nb_oc_blocking * nb_ic_blocking * ic_block * oc_block;
            const int njobs_x = bcast_work;
            const int njobs_y = jcp_.ngroups * load_work;

            const int max_threads = dnnl_get_max_threads();
            const size_t max_buffer_size = max_threads * job_size * 8;

            if (with_bias()) {
                reducer_bia_conf_.init(reduce_balancer_t(max_threads, oc_block,
                        jcp_.ngroups * jcp_.oc / oc_block, jcp_.mb,
                        max_buffer_size, true));
            }

            reducer_wei_conf_.init(
                    reduce_balancer_t(max_threads, job_size, njobs_y * njobs_x,
                            jcp_.mb * jcp_.nb_reduce, max_buffer_size, true),
                    job_size / nb_oc_blocking, nb_oc_blocking, ic_block,
                    nb_ic * ic_block * oc_block, nb_oc);
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);

    jit_avx2_1x1_convolution_bwd_weights_t(const pd_t *apd);

    ~jit_avx2_1x1_convolution_bwd_weights_t() {
        delete kernel_;
        delete rtus_driver_;
        delete reducer_weights_;
        delete reducer_bias_;
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_weights(ctx);
        return status::success;
    }

private:
    void execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    jit_avx2_1x1_conv_kernel_f32 *kernel_;
    cpu_reducer_2d_t<data_type::f32> *reducer_weights_;
    cpu_reducer_t<data_type::f32> *reducer_bias_;
    rtus_driver_t<avx2> *rtus_driver_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
