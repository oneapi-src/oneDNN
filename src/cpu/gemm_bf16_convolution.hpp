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

#ifndef CPU_GEMM_BF16_CONVOLUTION_HPP
#define CPU_GEMM_BF16_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_reducer.hpp"
#include "eltwise/jit_uni_eltwise_injector.hpp"
#include "gemm/gemm.hpp"
#include "gemm_convolution_utils.hpp"
#include "jit_avx512_core_bf16cvt.hpp"
#include "primitive.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t dst_data_type>
struct gemm_bf16_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_bf16_convolution_fwd_t,
                USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            bool ok = true && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(data_type::bf16, data_type::bf16,
                            data_type::undef, dst_data_type, data_type::f32)
                    && IMPLICATION(with_bias(),
                            utils::one_of(desc()->bias_desc.data_type,
                                    data_type::bf16, data_type::f32))
                    && !has_zero_dim_memory()
                    && set_default_formats_common(
                            dat_tag(), wei_tag(), dat_tag())
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops)
                    && post_ops_ok()
                    && memory_desc_matches_tag(*src_md(), dat_tag())
                    && memory_desc_matches_tag(*dst_md(), dat_tag())
                    && memory_desc_matches_tag(*weights_md(), wei_tag());
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_md(), weights_md(0), dst_md(),
                    dnnl_get_max_threads());
        }

        bool is_postprocess_required() const {
            bool post_ops_sum_only_for_dst_f32 = true
                    && dst_data_type == data_type::f32
                    && attr()->post_ops_.len_ == 1
                    && attr()->post_ops_.contain(primitive_kind::sum, 0);
            bool is_pp_for_post_ops_required = true
                    && attr()->post_ops_.len_ > 0
                    && !post_ops_sum_only_for_dst_f32;
            return dst_data_type == data_type::bf16 || with_bias()
                    || is_pp_for_post_ops_required;
        }

        conv_gemm_conf_t jcp_;

    protected:
        format_tag_t dat_tag() const {
            using namespace format_tag;
            return utils::pick(ndims() - 3, ncw, nchw, ncdhw);
        }

        format_tag_t wei_tag() const {
            using namespace format_tag;
            return with_groups() ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                                 : utils::pick(ndims() - 3, oiw, oihw, oidhw);
        }

        bool post_ops_ok() const {
            auto const &po = attr()->post_ops_;
            auto is_eltwise
                    = [&](int idx) { return po.entry_[idx].is_eltwise(); };
            auto is_sum = [&](int idx) { return po.entry_[idx].is_sum(); };

            switch (po.len_) {
                case 0: return true; // no post_ops
                case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
                case 2: return is_sum(0) && is_eltwise(1); // sum -> eltwise
                default: return false;
            }
            return false;
        }
    };

    gemm_bf16_convolution_fwd_t(const pd_t *apd)
        : primitive_t(apd), pp_ker_(nullptr) {
        const auto &post_ops = pd()->attr()->post_ops_;
        const acc_data_t one = 1.0, zero = 0.0;
        beta_ = dst_data_type == data_type::f32
                        && post_ops.find(primitive_kind::sum) >= 0
                ? one
                : zero;

        if (this->pd()->is_postprocess_required())
            pp_ker_ = new pp_ker_t(this->pd());
    }

    ~gemm_bf16_convolution_fwd_t() { delete pp_ker_; }

    typedef typename prec_traits<dst_data_type>::type dst_data_t;
    typedef typename prec_traits<data_type::f32>::type acc_data_t;
    typedef typename prec_traits<data_type::bf16>::type src_data_t;
    typedef typename prec_traits<data_type::bf16>::type wei_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    class pp_ker_t : jit_generator {
    public:
        DECLARE_CPU_JIT_AUX_FUNCTIONS(gemm_bf16_convolution_fwd_t::pp_kernel);
        pp_ker_t(const pd_t *pd);

        ~pp_ker_t() {
            delete bf16_emu_;
            delete eltwise_injector_;
        }

        void operator()(dst_data_t *dst, const acc_data_t *acc,
                const acc_data_t *bias, float sum_scale, size_t dst_str,
                size_t acc_str, size_t len, bool do_parallel);

    private:
        struct ker_args {
            dst_data_t *dst;
            const acc_data_t *acc;
            const acc_data_t *bias;
            float sum_scale;
            size_t dst_stride_in_bytes;
            size_t acc_stride_in_bytes;
            size_t spatial_length;
            size_t oc_work;
        };

        enum { default_unroll_2_pow_ = 2 };

        Xbyak::Reg64 reg_param = abi_param1;
        Xbyak::Reg64 reg_dst_base = rdx;
        Xbyak::Reg64 reg_acc_base = rax;
        Xbyak::Reg64 reg_dst = rsi;
        Xbyak::Reg64 reg_acc = rbp;
        Xbyak::Reg64 reg_bias = rbx;

        Xbyak::Reg64 reg_len = r8;
        Xbyak::Reg64 reg_tmp = rcx; // intentional for shifting purposes
        Xbyak::Reg64 reg_rem_mask = r9;
        Xbyak::Opmask kreg_rem_mask = k1;
        Xbyak::Reg64 reg_oc_iter = r11;
        Xbyak::Reg64 reg_len_iter = r12;
        Xbyak::Reg64 reg_dst_str = r13;
        Xbyak::Reg64 reg_acc_str = r14;

        Xbyak::Reg64 reserved_eltwise_gpr = r10;
        Xbyak::Opmask reserved_eltwise_maskr = k2;

        Xbyak::Zmm vreg_sum_scale, vreg_bias;

        Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(27);
        Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(28);
        Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(29);
        Xbyak::Reg64 bf16_emu_reserv_4 = r15;
        Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(30);
        Xbyak::Zmm bf16_emu_reserv_6 = Xbyak::Zmm(31);

        void (*ker_)(const ker_args *args);
        const conv_gemm_conf_t &jcp_;
        size_t OC_;
        bool do_bias_;
        bool do_eltwise_;
        bool do_sum_;
        int max_data_reg_idx_, max_unroll_, compute_reg_step_;
        int data_reg_base_idx_;
        size_t vlen_;
        cpu_isa_t isa_;
        bf16_emulation_t *bf16_emu_;
        jit_uni_eltwise_injector_f32<avx512_core> *eltwise_injector_;

        void generate();
        int vreg_dst_idx(int iter) {
            int idx = data_reg_base_idx_ + iter * compute_reg_step_ + 0;
            assert(idx <= max_data_reg_idx_);
            return idx;
        }
        int vreg_prev_dst_idx(int iter) {
            int idx = data_reg_base_idx_ + iter * compute_reg_step_ + 1;
            assert(idx <= max_data_reg_idx_);
            return idx;
        }

        Xbyak::Zmm vreg_dst(int iter) {
            return Xbyak::Zmm(vreg_dst_idx(iter));
        };

        Xbyak::Ymm vreg_dst_ymm(int iter) {
            return Xbyak::Ymm(vreg_dst_idx(iter));
        };

        Xbyak::Zmm vreg_prev_dst(int iter) {
            return Xbyak::Zmm(vreg_prev_dst_idx(iter));
        };

        Xbyak::Ymm vreg_prev_dst_ymm(int iter) {
            return Xbyak::Ymm(vreg_prev_dst_idx(iter));
        };
    };

    acc_data_t beta_;
    pp_ker_t *pp_ker_;
};

template <data_type_t diff_src_data_type>
struct gemm_bf16_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_bf16_convolution_bwd_data_t,
                USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_data
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(diff_src_data_type, data_type::bf16,
                            data_type::undef, data_type::bf16, data_type::f32)
                    && !has_zero_dim_memory()
                    && set_default_formats_common(
                            dat_tag(), wei_tag(), dat_tag())
                    && attr()->has_default_values()
                    && memory_desc_matches_tag(*diff_src_md(), dat_tag())
                    && memory_desc_matches_tag(*diff_dst_md(), dat_tag())
                    && memory_desc_matches_tag(*weights_md(), wei_tag());
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), diff_src_md(), weights_md(0), diff_dst_md(),
                    dnnl_get_max_threads());
        }

        conv_gemm_conf_t jcp_;

    protected:
        format_tag_t dat_tag() const {
            using namespace format_tag;
            return utils::pick(ndims() - 3, ncw, nchw, ncdhw);
        }

        format_tag_t wei_tag() const {
            using namespace format_tag;
            return with_groups() ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                                 : utils::pick(ndims() - 3, oiw, oihw, oidhw);
        }
    };

    gemm_bf16_convolution_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<data_type::bf16>::type diff_dst_data_t;
    typedef typename prec_traits<data_type::f32>::type acc_data_t;
    typedef typename prec_traits<diff_src_data_type>::type diff_src_data_t;
    typedef typename prec_traits<data_type::bf16>::type wei_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_data(ctx);
        return status::success;
    }

private:
    void execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

template <data_type_t diff_wei_data_type>
struct gemm_bf16_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_bf16_convolution_bwd_weights_t,
                USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_weights
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(data_type::bf16, diff_wei_data_type,
                            data_type::undef, data_type::bf16, data_type::f32)
                    && IMPLICATION(with_bias(),
                            utils::one_of(desc()->diff_bias_desc.data_type,
                                    data_type::bf16, data_type::f32))
                    && !has_zero_dim_memory() && attr()->has_default_values()
                    && set_default_formats_common(
                            dat_tag(), wei_tag(), dat_tag())
                    && memory_desc_matches_tag(*src_md(), dat_tag())
                    && memory_desc_matches_tag(*diff_dst_md(), dat_tag())
                    && memory_desc_matches_tag(*diff_weights_md(), wei_tag());
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_md(), diff_weights_md(0), diff_dst_md(),
                    dnnl_get_max_threads());
        }

        conv_gemm_conf_t jcp_;

    protected:
        format_tag_t dat_tag() const {
            using namespace format_tag;
            return utils::pick(ndims() - 3, ncw, nchw, ncdhw);
        }

        format_tag_t wei_tag() const {
            using namespace format_tag;
            return with_groups() ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                                 : utils::pick(ndims() - 3, oiw, oihw, oidhw);
        }
    };

    gemm_bf16_convolution_bwd_weights_t(const pd_t *apd)
        : primitive_t(apd), acc_ker_(nullptr) {
        acc_ker_ = new cpu_accumulator_1d_t<data_type::f32>();
    }

    ~gemm_bf16_convolution_bwd_weights_t() { delete acc_ker_; }

    typedef typename prec_traits<data_type::bf16>::type diff_dst_data_t;
    typedef typename prec_traits<data_type::f32>::type acc_data_t;
    typedef typename prec_traits<data_type::bf16>::type src_data_t;
    typedef typename prec_traits<diff_wei_data_type>::type diff_wei_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_weights(ctx);
        return status::success;
    }

private:
    void bf16_bwd_weights_reduction_par(int ithr_mb, int nthr_mb,
            const conv_gemm_conf_t &jcp, const acc_data_t *weights_reduce_base,
            diff_wei_data_t *weights_base) const;

    void execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    cpu_accumulator_1d_t<data_type::f32> *acc_ker_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
