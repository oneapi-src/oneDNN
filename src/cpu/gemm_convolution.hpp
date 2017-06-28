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

#ifndef CPU_JIT_GEMM_CONVOLUTION_HPP
#define CPU_JIT_GEMM_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "jit_avx2_gemm_f32.hpp"
#include "jit_avx512_common_gemm_f32.hpp"
#include "jit_primitive_conf.hpp"
#include "gemm_convolution_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <bool with_relu, bool run_jit, cpu_isa_t isa>
struct _gemm_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public _cpu_convolution_fwd_pd_t<with_relu> {
        pd_t(engine_t *engine,
                const typename pd_t::base_desc_t *adesc,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_convolution_fwd_pd_t<with_relu>(engine, adesc, hint_fwd_pd)
            , jcp_({}) {}

        DECLARE_COMMON_PD_T(_gemm_convolution_fwd_t<with_relu, run_jit, isa>);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            if (run_jit) {
                if (!mayiuse(isa)) return status::unimplemented;
            } else {
#ifndef USE_CBLAS
                return status::unimplemented;
#endif
            }

            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->cdesc_().prop_kind, forward_training,
                        forward_inference)
                && utils::implication(
                        this->base_pkind == primitive_kind::convolution_relu,
                        this->cdesc_().prop_kind == forward_inference)
                && this->cdesc_().alg_kind == alg_kind::convolution_direct
                && utils::everyone_is(data_type::f32,
                        this->cdesc_().src_desc.data_type,
                        this->cdesc_().weights_desc.data_type,
                        this->cdesc_().dst_desc.data_type)
                && utils::implication(this->with_bias(),
                        data_type::f32 == this->cdesc_().bias_desc.data_type)
                && this->src_pd_.desc()->format == nchw
                && this->dst_pd_.desc()->format == nchw
                && this->weights_pd_.desc()->format == (this->with_groups()
                        ? goihw : oihw);
            return ok ? status::success : status::unimplemented;
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(nchw));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(nchw));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(this->with_groups()
                            ? goihw : oihw));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            return status::success;
        }
    };

    _gemm_convolution_fwd_t(const pd_t *pd, const input_vector &inputs,
           const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
    {
        using namespace prop_kind;

        if (run_jit)
            sgemm_ = new jit_uni_gemm_f32('N', 'N', 0.0, false);

        jit_gemm_convolution_utils::init_conf(conf_.jcp_,
            *(conf_.cdesc()), conf_.src_pd(), conf_.weights_pd(0),
            conf_.dst_pd(), with_relu, conf_.negative_slope());
        jit_gemm_convolution_utils::prepare_workspace(this->conf_.jcp_,
            &this->ws, false, 0L);
    }
    ~_gemm_convolution_fwd_t() {
        if (run_jit) delete sgemm_;
        if (this->ws) free(this->ws);
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
    using jit_uni_gemm_f32 = typename utils::conditional
          <isa == avx2, jit_avx2_gemm_f32, jit_avx512_common_gemm_f32>::type;
    jit_uni_gemm_f32 *sgemm_;
    data_t *ws;
};

using jit_avx512_common_gemm_convolution_fwd_t =
                         _gemm_convolution_fwd_t<false, true, avx512_common>;
using jit_avx512_common_gemm_convolution_relu_t =
                         _gemm_convolution_fwd_t<true, true, avx512_common>;
using jit_avx2_gemm_convolution_fwd_t =
                         _gemm_convolution_fwd_t<false, true, avx2>;
using jit_avx2_gemm_convolution_relu_t =
                         _gemm_convolution_fwd_t<true, true, avx2>;
using mkl_gemm_convolution_fwd_t =
                         _gemm_convolution_fwd_t<false, false, isa_any>;
using mkl_gemm_convolution_relu_t =
                         _gemm_convolution_fwd_t<true, false, isa_any>;

template <bool run_jit, cpu_isa_t isa>
struct _gemm_convolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, hint_fwd_pd)
            , jcp_({})
        {}

        DECLARE_COMMON_PD_T(_gemm_convolution_bwd_data_t<run_jit, isa>);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            if (run_jit) {
                if (!mayiuse(isa)) return status::unimplemented;
            } else {
#ifndef USE_CBLAS
                return status::unimplemented;
#endif
            }

            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind, backward,
                        backward_data)
                && this->desc()->alg_kind == alg_kind::convolution_direct
                && utils::everyone_is(data_type::f32,
                        this->desc()->diff_src_desc.data_type,
                        this->desc()->weights_desc.data_type,
                        this->desc()->diff_dst_desc.data_type)
                && this->diff_src_pd_.desc()->format == nchw
                && this->diff_dst_pd_.desc()->format == nchw
                && this->weights_pd_.desc()->format == (this->with_groups()
                        ? goihw : oihw);
            return ok ? status::success : status::unimplemented;
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->diff_src_pd_.desc()->format == any)
                CHECK(this->diff_src_pd_.set_format(nchw));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(nchw));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(this->with_groups()
                            ? goihw : oihw));
            return status::success;
        }
    };

    _gemm_convolution_bwd_data_t(const pd_t *pd, const input_vector &inputs,
              const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
    {
        using namespace prop_kind;

        if (run_jit)
            sgemm_ = new jit_uni_gemm_f32('N', 'T', 0.0, false);

        jit_gemm_convolution_utils::init_conf(conf_.jcp_,
            *(conf_.desc()), conf_.diff_src_pd(), conf_.weights_pd(0),
            conf_.diff_dst_pd());
        jit_gemm_convolution_utils::prepare_workspace(this->conf_.jcp_,
            &this->ws, true, 0L);
    }
    ~_gemm_convolution_bwd_data_t() {
        if (run_jit) delete sgemm_;
        if (this->ws) free(this->ws);
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) {
        switch (conf_.desc()->prop_kind) {
        case prop_kind::backward:
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
    using jit_uni_gemm_f32 = typename utils::conditional
          <isa == avx2, jit_avx2_gemm_f32, jit_avx512_common_gemm_f32>::type;
    jit_uni_gemm_f32 *sgemm_;
    data_t *ws;
};

using jit_avx512_common_gemm_convolution_bwd_data_t =
                         _gemm_convolution_bwd_data_t<true, avx512_common>;
using jit_avx2_gemm_convolution_bwd_data_t =
                         _gemm_convolution_bwd_data_t<true, avx2>;
using mkl_gemm_convolution_bwd_data_t =
                         _gemm_convolution_bwd_data_t<false, isa_any>;

template <bool run_jit, cpu_isa_t isa>
struct _gemm_convolution_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, hint_fwd_pd)
            , jcp_({})
        {}

        DECLARE_COMMON_PD_T(_gemm_convolution_bwd_weights_t<run_jit, isa>);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            if (run_jit) {
                if (!mayiuse(isa)) return status::unimplemented;
            } else {
#ifndef USE_CBLAS
                return status::unimplemented;
#endif
            }

            bool ok = true
            && this->set_default_params() == status::success
            && utils::one_of(this->desc()->prop_kind, backward,
                    backward_weights)
            && this->desc()->alg_kind == alg_kind::convolution_direct
            && utils::everyone_is(data_type::f32,
                    this->desc()->src_desc.data_type,
                    this->desc()->diff_weights_desc.data_type,
                    this->desc()->diff_dst_desc.data_type)
            && utils::implication(this->with_bias(),
                    data_type::f32 == this->desc()->diff_bias_desc.data_type)
            && this->src_pd_.desc()->format == nchw
            && this->diff_dst_pd_.desc()->format == nchw
            && this->diff_weights_pd_.desc()->format == (this->with_groups()
                    ? goihw : oihw);
            return ok ? status::success : status::unimplemented;
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(nchw));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(nchw));
            if (this->diff_weights_pd_.desc()->format == any)
                CHECK(this->diff_weights_pd_.set_format(this->with_groups()
                            ? goihw : oihw));
            if (this->diff_bias_pd_.desc()->format == any)
                CHECK(this->diff_bias_pd_.set_format(x));
            return status::success;
        }
    };

    _gemm_convolution_bwd_weights_t(const pd_t *pd, const input_vector &inputs,
              const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
    {
        using namespace prop_kind;
        if (run_jit) {
            sgemm_0 = new jit_uni_gemm_f32('T', 'N', 0.0, false);
            sgemm_1 = new jit_uni_gemm_f32('T', 'N', 1.0, false);
        }

        jit_gemm_convolution_utils::init_conf(conf_.jcp_,
            *(conf_.desc()), conf_.src_pd(), conf_.diff_weights_pd(0),
            conf_.diff_dst_pd());
        const memory_desc_wrapper weights_d(conf_.diff_weights_pd(0));
        jit_gemm_convolution_utils::prepare_workspace(this->conf_.jcp_,
            &this->ws, true, weights_d.size());
    }
    ~_gemm_convolution_bwd_weights_t() {
        if (run_jit) {
            delete sgemm_0;
            delete sgemm_1;
        }
        if (this->ws) free(this->ws);
     };

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) {
        switch (conf_.desc()->prop_kind) {
        case prop_kind::backward:
        case prop_kind::backward_weights:
            execute_backward_weights();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_weights();
    pd_t conf_;
    using jit_uni_gemm_f32 = typename utils::conditional
          <isa == avx2, jit_avx2_gemm_f32, jit_avx512_common_gemm_f32>::type;
    jit_uni_gemm_f32 *sgemm_0, *sgemm_1;
    data_t *ws;
};

using jit_avx512_common_gemm_convolution_bwd_weights_t =
                         _gemm_convolution_bwd_weights_t<true, avx512_common>;
using jit_avx2_gemm_convolution_bwd_weights_t =
                         _gemm_convolution_bwd_weights_t<true, avx2>;
using mkl_gemm_convolution_bwd_weights_t =
                         _gemm_convolution_bwd_weights_t<false, isa_any>;

}
}
}

#endif
