/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef CPU_JIT_AVX512_CORE_X8S8S32X_CONVOLUTION_HPP
#define CPU_JIT_AVX512_CORE_X8S8S32X_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "jit_transpose_src_utils.hpp"
#include "cpu_reducer.hpp"
#include "cpu_barrier.hpp"

#include "jit_avx512_core_x8s8s32x_conv_kernel.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t src_type, impl::data_type_t dst_type>
struct jit_avx512_core_x8s8s32x_convolution_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
        {
        }
        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_int8:", avx512_core, ""),
                jit_avx512_core_x8s8s32x_convolution_fwd_t<src_type, dst_type>);

        virtual status_t init() override
        {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                    && utils::one_of(this->desc()->prop_kind, forward_training,
                               forward_inference)
                    && this->desc()->alg_kind == alg_kind::convolution_direct
                    && !this->has_zero_dim_memory()
                    && this->desc()->src_desc.data_type == src_type
                    && this->desc()->dst_desc.data_type == dst_type
                    && IMPLICATION(this->with_bias(), utils::one_of(
                            this->desc()->bias_desc.data_type, data_type::f32,
                            data_type::s32, data_type::s8, data_type::u8))
                    && this->desc()->accum_data_type == data_type::s32;
            if (!ok)
                return status::unimplemented;

            return jit_avx512_core_x8s8s32x_fwd_kernel::init_conf(
                    jcp_, *this->desc(), this->src_pd_, this->weights_pd_,
                    this->dst_pd_,this->bias_pd_, *this->attr(),
                    mkldnn_get_max_threads());
        }

        jit_conv_conf_t jcp_;
    };

    jit_avx512_core_x8s8s32x_convolution_fwd_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
        , local_scales_(nullptr)
    {
        kernel_ = new jit_avx512_core_x8s8s32x_fwd_kernel(conf_.jcp_,
                    *conf_.attr());
        if (conf_.jcp_.signed_input && conf_.jcp_.ver != ver_vnni) {
            size_t scales_size = (conf_.attr()->output_scales_.count_ == 1)
                ? 16
                : conf_.attr()->output_scales_.count_;
            local_scales_ = (float *)malloc(sizeof(float) * scales_size, 64);
            for (size_t i = 0; i < scales_size; i++) {
                local_scales_[i] = conf_.attr()->output_scales_.scales_[i] *
                                        (1.f / conf_.jcp_.wei_adj_scale);
            }
        }
    }

    ~jit_avx512_core_x8s8s32x_convolution_fwd_t() {
        delete kernel_;
        if (local_scales_) free(local_scales_);
    };

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    virtual void execute(event_t *e)
    {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
    jit_avx512_core_x8s8s32x_fwd_kernel *kernel_;
    float *local_scales_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
