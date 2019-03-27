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
#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "jit_transpose_src_utils.hpp"
#include "jit_uni_1x1_conv_utils.hpp"
#include "jit_avx512_core_bf16_1x1_conv_kernel.hpp"
#include "jit_avx512_core_bf16cvt.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template<impl::data_type_t dst_type>
struct _jit_avx512_core_bf16_1x1_convolution_fwd_t : public cpu_primitive_t {
    struct pd_t: public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr,
                    hint_fwd_pd)
            , jcp_(), rtus_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_bf16_1x1:", avx512_core, ""),
                _jit_avx512_core_bf16_1x1_convolution_fwd_t<dst_type>);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace utils;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && mayiuse(avx512_core)
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind, forward_training,
                        forward_inference)
                && utils::one_of(this->desc()->alg_kind,
                           alg_kind::convolution_auto,
                           alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && this->desc()->src_desc.data_type == data_type::bf16
                && this->desc()->dst_desc.data_type == dst_type
                && this->desc()->weights_desc.data_type == data_type::bf16
                && IMPLICATION(this->with_bias(),
                            data_type::f32 == this->desc()->bias_desc.data_type);

            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = this->desc();
            const memory_desc_t *src_d = this->src_pd_.desc();
            rtus_prepare(this, conv_d, src_d, this->dst_pd_.desc());

            status_t status = jit_avx512_core_bf16_1x1_conv_kernel::init_conf(jcp_,
                    *conv_d, *src_d, *this->weights_pd_.desc(),
                    *this->dst_pd_.desc(), *this->attr(),
                    mkldnn_get_max_threads(), rtus_.reduce_src_);
            if (status != status::success) return status;

            if (status == status::success
                    && this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));

            init_scratchpad();

            return status::success;
        }

        jit_1x1_conv_conf_t jcp_;
        reduce_to_unit_stride_t rtus_;

        private:
            void init_scratchpad() {
                using namespace mkldnn::impl::memory_tracking::names;
                auto scratchpad = scratchpad_registry().registrar();
                if (jcp_.with_bias && jcp_.oc != jcp_.oc_without_padding)
                    scratchpad.book(key_conv_padded_bias, sizeof(float) * jcp_.oc);
                rtus_prepare_space_info(this, scratchpad);
            }

        protected:
            virtual status_t set_default_params() override {
                using namespace memory_format;
                if (this->src_pd_.desc()->format == any)
                    CHECK(this->src_pd_.set_format(nChw16c));
                if (this->dst_pd_.desc()->format == any)
                    CHECK(this->dst_pd_.set_format(nChw16c));
                if (this->weights_pd_.desc()->format == any)
                    CHECK(this->weights_pd_.set_format(this->with_groups()
                        ? gOIhw8i16o2i : OIhw8i16o2i));
                if (this->bias_pd_.desc()->format == any)
                    CHECK(this->bias_pd_.set_format(x));
                return status::success;
            }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);
    _jit_avx512_core_bf16_1x1_convolution_fwd_t(const pd_t *apd,
                                          const input_vector &inputs,
                                          const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs)
        , kernel_(nullptr), rtus_driver_(nullptr)
    {
        kernel_ = new jit_avx512_core_bf16_1x1_conv_kernel(pd()->jcp_,
                    *pd()->attr());
        init_rtus_driver<avx512_common>(this);

    }
    ~_jit_avx512_core_bf16_1x1_convolution_fwd_t() {
        delete kernel_;
        delete rtus_driver_;
    }

    typedef typename prec_traits<data_type::bf16>::type src_data_t;
    typedef typename prec_traits<data_type::bf16>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

  private:
    void execute_forward() const;
    void execute_forward_thr(const int ithr, const int nthr,
            const src_data_t *src, const wei_data_t *weights,
            const float *bias, dst_data_t *dst,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_avx512_core_bf16_1x1_conv_kernel *kernel_;

    rtus_driver_t<avx512_common> *rtus_driver_;
};

template <impl::data_type_t dst_type>
using jit_avx512_core_bf16_1x1_convolution_fwd_t =
    _jit_avx512_core_bf16_1x1_convolution_fwd_t<dst_type>;

template <impl::data_type_t diff_src_type>
struct _jit_avx512_core_bf16_1x1_convolution_bwd_data_t : public cpu_primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_(), rtus_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_bf16_1x1:", avx512_core, ""),
                _jit_avx512_core_bf16_1x1_convolution_bwd_data_t<diff_src_type>);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && mayiuse(avx512_core)
                && this->set_default_params() == status::success
                && this->desc()->prop_kind == backward_data
                && utils::one_of(this->desc()->alg_kind,
                        alg_kind::convolution_auto,
                        alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && this->desc()->diff_src_desc.data_type == diff_src_type
                && this->desc()->weights_desc.data_type == data_type::bf16
                && this->desc()->diff_dst_desc.data_type == data_type::bf16;
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = this->desc();
            const memory_desc_t *diff_src_d = this->diff_src_pd_.desc();
            rtus_prepare(this, conv_d, diff_src_d, this->diff_dst_pd_.desc());

            status_t status = jit_avx512_core_bf16_1x1_conv_kernel::init_conf(jcp_,
                            *conv_d, *diff_src_d,
                            *this->weights_pd_.desc(),
                            *this->diff_dst_pd_.desc(), *this->attr(),
                            mkldnn_get_max_threads(), rtus_.reduce_src_);
            if (status != status::success) return status;

            if (status == status::success
                    && this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));

            init_scratchpad();

            return status::success;
        }

        // TODO (Roma): structs conf header cleanup
        jit_1x1_conv_conf_t jcp_;
        reduce_to_unit_stride_t rtus_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            rtus_prepare_space_info(this, scratchpad);
        }

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;

            if (this->diff_src_pd_.desc()->format == any)
                CHECK(this->diff_src_pd_.set_format(nChw16c));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(nChw16c));
            if (this->weights_pd_.desc()->format == any) {
                CHECK(this->weights_pd_.set_format(this->with_groups()
                    ? gIOhw8o16i2o : IOhw8o16i2o));
            }
            return status::success;
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);
    _jit_avx512_core_bf16_1x1_convolution_bwd_data_t(const pd_t *apd,
                                              const input_vector &inputs,
                                              const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs)
        , kernel_(nullptr), rtus_driver_(nullptr)
    {
        kernel_ = new jit_avx512_core_bf16_1x1_conv_kernel(pd()->jcp_,
                    *pd()->attr());
        init_rtus_driver<avx512_common>(this);
    }
    ~_jit_avx512_core_bf16_1x1_convolution_bwd_data_t()
    {
        delete kernel_;
        delete rtus_driver_;
    }

    typedef typename prec_traits<data_type::bf16>::type diff_dst_data_t;
    typedef typename prec_traits<data_type::bf16>::type wei_data_t;
    typedef typename prec_traits<diff_src_type>::type diff_src_data_t;

    virtual void execute(event_t *e) const {
        switch (pd()->desc()->prop_kind) {
        case prop_kind::backward_data:
            execute_backward_data();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

  private:
    void execute_backward_data() const;
    void execute_backward_data_thr(const int, const int,
            const diff_dst_data_t *, const wei_data_t *, diff_src_data_t *,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_avx512_core_bf16_1x1_conv_kernel *kernel_;
    /* reduction to unit stride */
    rtus_driver_t<avx512_common> *rtus_driver_;
};

template <impl::data_type_t diff_src_type>
using jit_avx512_core_bf16_1x1_convolution_bwd_data_t =
    _jit_avx512_core_bf16_1x1_convolution_bwd_data_t<diff_src_type>;

}
}
}
#endif
