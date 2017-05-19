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

#ifndef CPU_JIT_AVX512_CORE_U8S8U8_CONVOLUTION_HPP
#define CPU_JIT_AVX512_CORE_U8S8U8_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"

#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx512_core_u8s8u8_conv_fwd_ker_t;

template <bool with_relu>
struct _jit_avx512_core_u8s8u8_convolution_fwd_t : public cpu_primitive_t {
    struct pd_t : public _cpu_convolution_fwd_pd_t<with_relu> {
        pd_t(engine_t *engine, const typename pd_t::base_desc_t *adesc,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_convolution_fwd_pd_t<with_relu>(engine, adesc, hint_fwd_pd)
            , jcp_({})
        {}

        DECLARE_COMMON_PD_T(
                _jit_avx512_core_u8s8u8_convolution_fwd_t<with_relu>);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->cdesc_().prop_kind, forward_training,
                        forward_inference)
                && utils::implication(with_relu,
                        this->cdesc_().prop_kind == forward_inference)
                && this->cdesc_().alg_kind == alg_kind::convolution_direct
                && utils::everyone_is(data_type::u8,
                        this->cdesc_().src_desc.data_type,
                        this->cdesc_().dst_desc.data_type)
                && this->cdesc_().weights_desc.data_type == data_type::s8
                && utils::implication(this->with_bias(), utils::one_of(
                            this->cdesc_().bias_desc.data_type,
                            data_type::s32, data_type::s8, data_type::u8))
                && this->cdesc_().accum_data_type == data_type::s32;

            if (!ok) return status::unimplemented;

            return jit_conf();
        }

        jit_conv_conf_t jcp_;

    protected:
        status_t jit_conf();

        virtual status_t set_default_params() override {
            using namespace memory_format;

            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(nhwc));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(nhwc));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(this->with_groups()
                            ? gOhIw16o4i : OhIw16o4i));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));

            return status::success;
        }
    };

    _jit_avx512_core_u8s8u8_convolution_fwd_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs);
    ~_jit_avx512_core_u8s8u8_convolution_fwd_t();

    typedef typename prec_traits<data_type::u8>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<data_type::u8>::type dst_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual void execute(event_t *e) {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;

    jit_avx512_core_u8s8u8_conv_fwd_ker_t *ker_;
    size_t ws_per_thread_;
    acc_data_t *ws_;
};

using jit_avx512_core_u8s8u8_convolution_fwd_t
        = _jit_avx512_core_u8s8u8_convolution_fwd_t<false>;
using jit_avx512_core_u8s8u8_convolution_relu_t
        = _jit_avx512_core_u8s8u8_convolution_fwd_t<true>;

}
}
}

#endif
