/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#ifndef GEMM_X8S8S32X_CONVOLUTION_HPP
#define GEMM_X8S8S32X_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_primitive.hpp"

#include "jit_primitive_conf.hpp"
#include "jit_generator.hpp"
#include "gemm_convolution_utils.hpp"

#include "gemm/gemm.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t src_type, data_type_t dst_type>
struct _gemm_x8s8s32x_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(IGEMM_S8U8S32_IMPL_STR,
                _gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>);

        status_t init() {
            using namespace data_type;
            using namespace memory_format;

            bool ok = true
                && is_fwd()
                && this->set_default_alg_kind(alg_kind::convolution_direct)
                && this->expect_data_types(src_type, s8, data_type::undef,
                        dst_type, s32)
                && IMPLICATION(this->with_bias(), utils::one_of(
                            this->desc()->bias_desc.data_type, f32, s32, s8,
                            u8))
                && this->set_default_params() == status::success
                && !this->has_zero_dim_memory()
                && utils::everyone_is(nhwc, this->src_md_.format,
                        this->dst_md_.format)
                && this->weights_md_.format == (this->with_groups()
                        ? ((src_type == s8) ? hwigo_s8s8 : hwigo)
                        : ((src_type == s8) ? hwio_s8s8 : hwio))
                && this->is_gemm_conv_format();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *this->desc(), this->src_md(), this->weights_md(0),
                    this->dst_md(), mkldnn_get_max_threads());
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        status_t set_default_params() {
            using namespace memory_format;
            const bool is_sign_input =
                desc()->src_desc.data_type == data_type::s8;

            if (src_md_.format == any)
                CHECK(types::set_default_format(src_md_, nhwc));
            if (dst_md_.format == any)
                CHECK(types::set_default_format(dst_md_, nhwc));
            if (weights_md_.format == any)
                CHECK(types::set_default_format(weights_md_, with_groups()
                            ? (is_sign_input ? hwigo_s8s8 : hwigo)
                            : (is_sign_input ? hwio_s8s8 : hwio)));
            if (bias_md_.format == any)
                CHECK(types::set_default_format(bias_md_, x));
            return status::success;
        }

        bool is_gemm_conv_format() {
            using namespace mkldnn::impl::primitive_kind;
            auto const &po = this->attr()->post_ops_;
            auto is_relu = [&](int idx) {
                return po.entry_[idx].is_relu(true, false); };

            switch (po.len_) {
            case 0: return true;
            case 1: return is_relu(0) || po.contain(sum, 0);
            case 2: return po.contain(sum, 0) && is_relu(1);
            default: return false;
            }
            return false;
        }
    };

    _gemm_x8s8s32x_convolution_fwd_t(const pd_t *apd)
        : cpu_primitive_t(apd, true), pp_ker_(nullptr)
    { pp_ker_ = new pp_ker_t(pd()); }
    ~_gemm_x8s8s32x_convolution_fwd_t() { delete pp_ker_; }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    // XXX: this is throwaway code that will become unnecessary when we have a
    // sufficiently advanced igemm jit generator that supports quantization,
    // relu, and whatnot
    class pp_ker_t : jit_generator {
    public:
        DECLARE_CPU_JIT_AUX_FUNCTIONS(
        _gemm_x8s8s32x_convolution_fwd_t::pp_kernel);
        pp_ker_t(const pd_t *pd);

        void operator()(dst_data_t *dst, const acc_data_t *acc,
            const char *bias, const float *scales,
            float nslope, float sum_scale, float signed_scale,
            int g, size_t start, size_t end);

        size_t dst_os_stride_;

    private:
        void generate();

        struct ker_args {
            dst_data_t *dst;
            const acc_data_t *acc;
            const char *bias;
            const float *scales;
            float nslope;
            float sum_scale;
            float signed_scale;
            size_t len;
            size_t oc_offset;
        };
        void(*ker_)(const ker_args *args);

        const jit_gemm_conv_conf_t &jcp_;
        size_t OC_;
        size_t OS_;
        data_type_t bias_data_type_;
        size_t bias_data_type_size_;
        size_t scale_idx_mult_;
        round_mode_t rmode_;
        bool do_bias_;
        bool do_relu_;
        bool do_sum_;
        bool do_signed_scaling_;
        size_t vlen_;
    };

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    void execute_forward(const exec_ctx_t &ctx) const;
    void execute_forward_thr(const int ithr, const int nthr,
            const src_data_t *src_base, const wei_data_t *wei_base,
            const char *bia_base, dst_data_t *dst_base,
            const memory_tracking::grantor_t &scratchpad) const;

    int nthr_;
    pp_ker_t *pp_ker_;

};

template <data_type_t dst_type>
struct _gemm_u8s8s32x_convolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_data_pd_t{
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(IGEMM_S8U8S32_IMPL_STR,
                _gemm_u8s8s32x_convolution_bwd_data_t<dst_type>);

        status_t init() {
            using namespace data_type;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
                && this->desc()->prop_kind == prop_kind::backward_data
                && this->set_default_alg_kind(alg_kind::convolution_direct)
                && this->expect_data_types(dst_type, s8, data_type::undef, u8,
                        s32)
                && IMPLICATION(this->with_bias(), utils::one_of(
                            this->desc()->bias_desc.data_type, f32, s32, s8, u8))
                && this->set_default_params() == status::success
                && !this->has_zero_dim_memory()
                && utils::everyone_is(nhwc, this->diff_src_md_.format,
                        this->diff_dst_md_.format)
                && this->weights_md_.format == (this->with_groups()
                        ? hwigo : hwio)
                && attr()->post_ops_.has_default_values();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *this->desc(), this->diff_src_md(), this->weights_md(0),
                    this->diff_dst_md(), mkldnn_get_max_threads());
        }

        virtual bool support_bias() const override { return true; }

        jit_gemm_conv_conf_t jcp_;

    protected:
        status_t set_default_params() {
            using namespace memory_format;

            if (diff_src_md_.format == any)
                CHECK(types::set_default_format(diff_src_md_, nhwc));
            if (diff_dst_md_.format == any)
                CHECK(types::set_default_format(diff_dst_md_, nhwc));
            if (weights_md_.format == any)
                CHECK(types::set_default_format(weights_md_,
                            with_groups() ? hwigo : hwio));
            if (bias_md_.format == any)
                CHECK(types::set_default_format(bias_md_, x));
             return status::success;
        }
    };

    _gemm_u8s8s32x_convolution_bwd_data_t(const pd_t *apd)
        : cpu_primitive_t(apd, true) {}

    typedef typename prec_traits<data_type::u8>::type diff_dst_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type diff_src_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_data(ctx);
        return status::success;
    }

private:
    void execute_backward_data(const exec_ctx_t &ctx) const;
    void execute_backward_data_thr(const int ithr, const int nthr,
            const diff_dst_data_t *diff_dst_base, const wei_data_t *wei_base,
            const char *bia_base, diff_src_data_t *diff_src_base,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif
