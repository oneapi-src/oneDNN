/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#ifndef JIT_AVX512_MIC_S16S16S32_CONVOLUTION_HPP
#define JIT_AVX512_MIC_S16S16S32_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "jit_avx512_common_conv_kernel.hpp"
#include "cpu_reducer.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <bool with_relu>
struct _jit_avx512_mic_s16s16s32_convolution_fwd_t : public cpu_primitive_t {
    struct pd_t : public _cpu_convolution_fwd_pd_t<with_relu> {
        pd_t(engine_t *engine, const typename pd_t::base_desc_t *adesc,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_convolution_fwd_pd_t<with_relu>(engine, adesc, hint_fwd_pd)
            , jcp_({})
        {
        }

        DECLARE_COMMON_PD_T(_jit_avx512_mic_s16s16s32_convolution_fwd_t<
                                                                    with_relu>);

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
                    && utils::everyone_is(data_type::s16,
                               this->cdesc_().src_desc.data_type,
                               this->cdesc_().weights_desc.data_type)
                    && utils::everyone_is(data_type::s32,
                               this->cdesc_().dst_desc.data_type)
                    && utils::implication(this->with_bias(),
                        data_type::s32 == this->cdesc_().bias_desc.data_type)
                    && this->IC() != 3; // TODO add support of first convolution
            if (!ok)
                return status::unimplemented;

            return jit_avx512_common_conv_fwd_kernel::init_conf(jcp_,
                    this->cdesc_(), *this->src_pd_.desc(),
                    *this->weights_pd_.desc(), *this->dst_pd_.desc());
        }

        jit_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override
        {
            using namespace memory_format;
            if (this->src_pd_.desc()->format == any) {
                CHECK(this->src_pd_.set_format(nChw16c));
            }
            if (this->dst_pd_.desc()->format == any) {
                CHECK(this->dst_pd_.set_format(nChw16c));
            }
            if (this->weights_pd_.desc()->format == any) {
                CHECK(this->weights_pd_.set_format(this->with_groups() ?
                                gOIhw8i16o2i : OIhw8i16o2i));
            }
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            return status::success;
        }
    };
    _jit_avx512_mic_s16s16s32_convolution_fwd_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
    {
        kernel_ = new jit_avx512_common_conv_fwd_kernel(conf_.jcp_);
    }
    ~_jit_avx512_mic_s16s16s32_convolution_fwd_t() { delete kernel_; };

    typedef typename prec_traits<data_type::s16>::type data_input_t;
    typedef typename prec_traits<data_type::s32>::type data_output_t;

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

using jit_avx512_mic_s16s16s32_convolution_fwd_t
        = _jit_avx512_mic_s16s16s32_convolution_fwd_t<false>;
}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
