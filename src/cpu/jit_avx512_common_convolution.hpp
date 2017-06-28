/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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

#ifndef CPU_JIT_AVX512_COMMON_CONVOLUTION_HPP
#define CPU_JIT_AVX512_COMMON_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "jit_avx512_common_conv_kernel.hpp"
#include "jit_transpose_src_utils.hpp"
#include "cpu_reducer.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <bool with_relu, impl::data_type_t src_type,
         impl::data_type_t wei_type = src_type,
         impl::data_type_t dst_type = src_type>
struct _jit_avx512_common_convolution_fwd_t : public cpu_primitive_t {
    struct pd_t : public _cpu_convolution_fwd_pd_t<with_relu> {
        pd_t(engine_t *engine, const typename pd_t::base_desc_t *adesc,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_convolution_fwd_pd_t<with_relu>(engine, adesc, hint_fwd_pd)
            , jcp_({})
        {
        }

        DECLARE_COMMON_PD_T(_jit_avx512_common_convolution_fwd_t);

        virtual status_t init() override
        {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true && this->set_default_params() == status::success
                    && utils::one_of(this->cdesc_().prop_kind, forward_training,
                               forward_inference)
                    && utils::implication(this->base_pkind
                                       == primitive_kind::convolution_relu,
                               this->cdesc_().prop_kind == forward_inference)
                    && this->cdesc_().alg_kind == alg_kind::convolution_direct
                    && this->cdesc_().src_desc.data_type == src_type
                    && this->cdesc_().weights_desc.data_type == wei_type
                    && this->cdesc_().dst_desc.data_type == dst_type
                    && utils::implication(this->with_bias(), dst_type
                                       == this->cdesc_().bias_desc.data_type)
                    && !(with_relu && this->negative_slope()!= 0.
                                   && dst_type == data_type::s32
                                   && src_type == data_type::s16
                                   && wei_type == data_type::s16);
            if (!ok)
                return status::unimplemented;

            return jit_avx512_common_conv_fwd_kernel::init_conf(jcp_,
                    this->cdesc_(), *this->src_pd_.desc(),
                    *this->weights_pd_.desc(), *this->dst_pd_.desc(), with_relu,
                    this->negative_slope());
        }

        jit_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override
        {
            using namespace memory_format;
            const bool flat = this->IC() == 3;
            if (this->src_pd_.desc()->format == any) {
                CHECK(this->src_pd_.set_format(flat ? nchw : nChw16c));
            }
            if (this->dst_pd_.desc()->format == any) {
                CHECK(this->dst_pd_.set_format(nChw16c));
            }
            if (this->weights_pd_.desc()->format == any) {
                if (dst_type == data_type::s32
                 && src_type == data_type::s16
                 && wei_type == data_type::s16) {
                        CHECK(this->weights_pd_.set_format(this->with_groups() ?
                                        gOIhw8i16o2i : OIhw8i16o2i));
                    }
                else if (dst_type == data_type::f32
                      && src_type == data_type::f32
                      && wei_type == data_type::f32) {
                        CHECK(this->weights_pd_.set_format(this->with_groups()
                                    ? (flat ? gOhwi16o : gOIhw16i16o)
                                    : (flat ? Ohwi16o : OIhw16i16o)));
                      }
            }
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            return status::success;
        }
    };

    _jit_avx512_common_convolution_fwd_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
    {
        kernel_ = new jit_avx512_common_conv_fwd_kernel(conf_.jcp_);
    }
    ~_jit_avx512_common_convolution_fwd_t() { delete kernel_; };

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    virtual void execute(event_t *e)
    {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
    jit_avx512_common_conv_fwd_kernel *kernel_;
};

template <impl::data_type_t src_type, impl::data_type_t wei_type = src_type,
         impl::data_type_t dst_type = src_type>
using jit_avx512_common_convolution_fwd_t =
    _jit_avx512_common_convolution_fwd_t<false, src_type, wei_type, dst_type>;

template <impl::data_type_t src_type, impl::data_type_t wei_type = src_type,
         impl::data_type_t dst_type = src_type>
using jit_avx512_common_convolution_relu_t =
    _jit_avx512_common_convolution_fwd_t<true, src_type, wei_type, dst_type>;

template <impl::data_type_t diff_dst_type,
          impl::data_type_t wei_type = diff_dst_type,
          impl::data_type_t diff_src_type = diff_dst_type>
struct jit_avx512_common_convolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, hint_fwd_pd)
            , jcp_({})
        {}

        DECLARE_COMMON_PD_T(jit_avx512_common_convolution_bwd_data_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind, backward_data) // XXX (this->!)
                && this->desc()->alg_kind == alg_kind::convolution_direct
                && this->desc()->diff_dst_desc.data_type == diff_dst_type
                && this->desc()->weights_desc.data_type == wei_type
                && this->desc()->diff_src_desc.data_type == diff_src_type;
            if (!ok) return status::unimplemented;

            return jit_avx512_common_conv_bwd_data_kernel_f32::init_conf(jcp_,
                    *this->desc(), *this->diff_src_pd_.desc(),
                    *this->weights_pd_.desc(), *this->diff_dst_pd_.desc());
        }

        jit_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;

            if (this->diff_src_pd_.desc()->format == any)
                CHECK(this->diff_src_pd_.set_format(nChw16c));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(nChw16c));
            if (this->weights_pd_.desc()->format == any) {
                if (diff_dst_type == data_type::s16
                 && diff_src_type == data_type::s32
                 && wei_type == data_type::s16) {
                        CHECK(this->weights_pd_.set_format(this->with_groups() ?
                                    gOIhw8o16i2o : OIhw8o16i2o));
                 } else if (diff_dst_type == data_type::f32
                         && diff_src_type == data_type::f32
                         && wei_type == data_type::f32) {
                        CHECK(this->weights_pd_.set_format(this->with_groups()
                                    ? gOIhw16o16i : OIhw16o16i));
                      }
            }
            return status::success;
        }
    };

    jit_avx512_common_convolution_bwd_data_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
    { kernel_ = new jit_avx512_common_conv_bwd_data_kernel_f32(conf_.jcp_); }
    ~jit_avx512_common_convolution_bwd_data_t() { delete kernel_; };

    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<diff_src_type>::type diff_src_data_t;

    virtual void execute(event_t *e) {
        switch (conf_.desc()->prop_kind) {
        case prop_kind::backward_data:
            execute_backward_data();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_data();
    pd_t conf_;
    jit_avx512_common_conv_bwd_data_kernel_f32 *kernel_;
};

struct jit_avx512_common_convolution_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public  cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, hint_fwd_pd)
            , jcp_({}) {}

        DECLARE_COMMON_PD_T(jit_avx512_common_convolution_bwd_weights_t);

        virtual status_t init() override {
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && this->desc()->prop_kind == prop_kind::backward_weights
                && this->desc()->alg_kind == alg_kind::convolution_direct
                && utils::everyone_is(data_type::f32,
                        this->desc()->src_desc.data_type,
                        this->desc()->diff_dst_desc.data_type,
                        this->desc()->diff_weights_desc.data_type);
            if (!ok) return status::unimplemented;

            return jit_avx512_common_conv_bwd_weights_kernel_f32::init_conf(
                    jcp_,
                    *this->desc(), *this->src_pd_.desc(),
                    *this->diff_weights_pd_.desc(),
                    *this->diff_dst_pd_.desc());
        }

        jit_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            const bool flat = this->IC() == 3;

            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(flat ? nchw : nChw16c));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(nChw16c));
            if (this->diff_weights_pd_.desc()->format == any)
                CHECK(this->diff_weights_pd_.set_format(this->with_groups()
                            ? (flat ? gOhwi16o : gOIhw16i16o)
                            : (flat ? Ohwi16o : OIhw16i16o)));
            if (this->diff_bias_pd_.desc()->format == any)
                CHECK(this->diff_bias_pd_.set_format(x));
            return status::success;
        }
    };

    jit_avx512_common_convolution_bwd_weights_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
        , kernel_(nullptr), trans_kernel_(nullptr), ws_(nullptr)
        , reducer_weights_(nullptr), reducer_bias_(nullptr)
    {
        kernel_ = new jit_avx512_common_conv_bwd_weights_kernel_f32(conf_.jcp_);
        const int max_threads = omp_get_max_threads();
        const size_t max_buffer_size = max_threads * 3 * 5 * 5 * 16 * 16;
        const auto &j = conf_.jcp_;

        reducer_weights_ = new cpu_reducer_t<data_type::f32>(reduce_balancer_t(
                    max_threads, j.kh * j.kw * j.ic_block * j.oc_block,
                    j.ngroups * j.nb_ic * j.nb_oc, j.mb, max_buffer_size));
        if (conf_.with_bias()) {
            reducer_bias_ = new cpu_reducer_t<data_type::f32>(
                    reduce_balancer_t(max_threads, j.oc_block,
                        j.ngroups * j.nb_oc, j.mb, max_buffer_size));
        }
        if (j.transpose_src) {
            const int ws_size = j.mb * j.ngroups *
                j.nb_ic * j.ic_block * j.tr_iw * j.ih;
            ws_ = (data_t*)malloc(ws_size * sizeof(data_t), 64);
            trans_kernel_ = new jit_transpose_src(&conf_.jcp_);
        }
    }
    ~jit_avx512_common_convolution_bwd_weights_t() {
        delete kernel_;
        delete trans_kernel_;
        free(ws_);
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) {
        execute_backward_weights();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_weights();
    pd_t conf_;
    jit_avx512_common_conv_bwd_weights_kernel_f32 *kernel_;
    jit_transpose_src *trans_kernel_;
    data_t *ws_;
    cpu_reducer_t<data_type::f32> *reducer_weights_, *reducer_bias_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
