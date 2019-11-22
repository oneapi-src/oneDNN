/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
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

#ifndef CPU_REF_RNN_HPP
#define CPU_REF_RNN_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "../cpu_isa_traits.hpp"
#include "../gemm/gemm.hpp"
#include "../gemm/os_blas.hpp"

#include "cpu_rnn_pd.hpp"
#include "jit_uni_rnn_common_postgemm_dispatcher.hpp"
#include "rnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <typename src_data_t, typename acc_data_t>
void gates_reduction(const rnn_utils::rnn_conf_t &rnn,
        const src_data_t *ws_gates_, acc_data_t *diff_bias_) {
    auto body = [&](int i, int k) {
        for (int j = 0; j < rnn.mb; j++)
            diff_bias_[i * rnn.dic + k]
                    += ws_gates_[j * rnn.gates_ws_ld + i * rnn.dic + k];
    };

    // @todo block k on simd-width
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP \
        && _OPENMP >= 201307 /* icc 17.0 has a problem with simd collapse */ \
        && !((defined __INTEL_COMPILER) && (__INTEL_COMPILER == 1700))
#pragma omp parallel for simd collapse(2)
    for (int i = 0; i < rnn.n_gates; i++)
        for (int k = 0; k < rnn.dic; k++)
            body(i, k);
#else
    parallel_nd(rnn.n_gates, rnn.dic, body);
#endif
}

template <prop_kind_t aprop, impl::data_type_t src_type,
        impl::data_type_t weights_type, impl::data_type_t acc_type>
struct _ref_rnn_common_t : public primitive_impl_t {
    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<weights_type>::type weights_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    typedef typename utils::conditional<aprop == prop_kind::forward, acc_data_t,
            src_data_t>::type scratch_data_t;

    using class_name
            = _ref_rnn_common_t<aprop, src_type, weights_type, acc_type>;

    typedef rnn_cell_execution_sig((class_name::*cell_execution_f));
    typedef rnn_grid_execution_sig((class_name::*grid_execution_f));

    typedef rnn_gemm_sig((class_name::*gemm_t));
    typedef rnn_bias_prepare_sig((class_name::*bias_prepare_t));
    typedef rnn_bias_finalize_sig((class_name::*bias_finalize_t));
    typedef rnn_weights_assign_sig((class_name::*weights_assign_t));

    using base_pd_t =
            typename utils::conditional<false || aprop == prop_kind::forward,
                    cpu_rnn_fwd_pd_t, cpu_rnn_bwd_pd_t>::type;

    struct pd_t : public base_pd_t {
        using base_pd_t::base_pd_t;

        DECLARE_COMMON_PD_T("ref:any", class_name, USE_GLOBAL_SCRATCHPAD);

        status_t init() {
            using namespace prop_kind;
            using namespace utils;
            using namespace format_tag;
            using namespace rnn_utils;
            const alg_kind_t cell_kind = this->desc()->cell_kind;

            data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
            data_type_t weights_iter_dt
                    = this->desc()->weights_iter_desc.data_type;
            data_type_t weights_layer_dt
                    = this->desc()->weights_layer_desc.data_type;

            bool ok = true
                    && one_of(cell_kind, alg_kind::vanilla_rnn,
                            alg_kind::vanilla_lstm, alg_kind::vanilla_gru,
                            alg_kind::lbr_gru)
                    && IMPLICATION(aprop == prop_kind::forward,
                            one_of(this->desc()->prop_kind, forward_training,
                                    forward_inference))
                    && IMPLICATION(aprop == backward,
                            one_of(this->desc()->prop_kind, backward))
                    && src_layer_dt == src_type
                    && everyone_is(
                            weights_type, weights_iter_dt, weights_layer_dt)
                    && this->set_default_params() == status::success
                    && this->with_bias();
            if (!ok) return status::unimplemented;

            init_conf(rnn_, *this->desc(), this->src_md(0), this->src_md(1),
                    this->src_md(2), this->weights_md(0), this->weights_md(1),
                    this->dst_md(0), this->dst_md(1), this->dst_md(2));

            // check that bf16 gemm is available
            ok = ok
                    && IMPLICATION(
                            rnn_.dt_conf == all_bf16, mayiuse(avx512_core));

            /* check that only supported attr have been passed */
            primitive_attr_t::skip_mask_t attr_mask
                    = primitive_attr_t::skip_mask_t::rnn_tparams;
            if (weights_layer_dt == data_type::s8)
                attr_mask = attr_mask
                        | primitive_attr_t::skip_mask_t::rnn_data_qparams
                        | primitive_attr_t::skip_mask_t::rnn_weights_qparams;
            ok = ok && this->attr()->has_default_values(attr_mask);
            if (!ok) return status::unimplemented;

            // Set weights descriptors to desired format
            memory_desc_t new_weights_layer_md = *this->weights_md(0);
            CHECK(set_expected_desc(rnn_, new_weights_layer_md, false));
            if (this->weights_layer_md_.format_kind == format_kind::any) {
                this->weights_layer_md_ = new_weights_layer_md;
            } else if (this->weights_layer_md_.format_kind
                    == format_kind::rnn_packed) {
                if (this->weights_layer_md_ != new_weights_layer_md)
                    return status::unimplemented;
            }

            memory_desc_t new_weights_iter_md = *this->weights_md(1);
            CHECK(set_expected_desc(rnn_, new_weights_iter_md, true));
            if (this->weights_iter_md_.format_kind == format_kind::any) {
                this->weights_iter_md_ = new_weights_iter_md;
            } else if (this->weights_iter_md_.format_kind
                    == format_kind::rnn_packed) {
                if (this->weights_iter_md_ != new_weights_iter_md)
                    return status::unimplemented;
            }

            CHECK(this->check_layout_consistency());

            set_conf(rnn_, *this->desc(), this->weights_md(0),
                    this->weights_md(1), this->diff_weights_md(0),
                    this->diff_weights_md(1));

            size_t scratchpad_sz {0}, ws_sz {0};
            get_scratchpad_and_workspace_sizes(rnn_, scratchpad_sz, ws_sz);

            // initialize the workspace if needed
            if (rnn_.is_training) {
                dims_t ws_dims = {(dim_t)ws_sz};
                dnnl_memory_desc_init_by_tag(&this->ws_md_, 1, ws_dims,
                        data_type::u8, format_tag::x);
            }

            init_scratchpad(scratchpad_sz);

            return status::success;
        }

        rnn_utils::rnn_conf_t rnn_;

    private:
        void init_scratchpad(size_t scratchpad_sz) {
            using namespace memory_tracking::names;
            auto scratchpad = this->scratchpad_registry().registrar();
            scratchpad.book(key_rnn_space, sizeof(float) * scratchpad_sz, 4096);

            int max_nparts = this->cell_kind() == alg_kind::vanilla_gru ? 2 : 1;
            int ptr_wei_sz = rnn_.n_layer * rnn_.n_dir * max_nparts;
            scratchpad.book(
                    key_rnn_ptrs_wei_layer, sizeof(float *) * ptr_wei_sz);
            scratchpad.book(
                    key_rnn_ptrs_wei_iter, sizeof(float *) * ptr_wei_sz);
            scratchpad.book(key_rnn_ptrs_bia, sizeof(float *) * ptr_wei_sz);
            scratchpad.book(key_rnn_gates,
                    sizeof(scratch_data_t) * rnn_.scratch_gates_size);
            scratchpad.book(
                    key_rnn_cell, sizeof(acc_data_t) * rnn_.scratch_cell_size);
        }
    };

    _ref_rnn_common_t(const pd_t *apd)
        : primitive_impl_t(apd), rnn_postgemm_(nullptr) {
        /// @todo set max_feature_size assuming that we limit the number of
        /// iterations and layer to one if slc != dic and sic != dic
        /// respectively

        bias_preparation_func = &class_name::bias_prepare;
        bias_finalization_func = &class_name::bias_finalize;

        auto set_gemm_funcs
                = [](bool packed_gemm, gemm_t &g, weights_assign_t &a) {
                      if (packed_gemm) {
                          g = &class_name::packed_gemm;
                          a = &class_name::assign_packed_weights;
                      } else {
                          g = &class_name::gemm;
                          a = &class_name::assign_weights;
                      }
                  };
        set_gemm_funcs(pd()->rnn_.use_iter_packed_gemm, gemm_iter_func,
                weights_iter_assign_func);

        set_gemm_funcs(pd()->rnn_.use_layer_packed_gemm, gemm_layer_func,
                weights_layer_assign_func);

        rnn_postgemm_ = new rnn_postgemm_dispatcher<aprop, src_type, acc_type>(
                pd()->rnn_, pd());
        assert(rnn_postgemm_ != nullptr);
        switch (pd()->cell_kind()) {
            case alg_kind::vanilla_rnn:
            case alg_kind::vanilla_lstm:
                cell_func = &class_name::cell_execution;
                break;
            case alg_kind::vanilla_gru:
                cell_func = &class_name::cell_execution_gru;
                break;
            case alg_kind::lbr_gru:
                cell_func = &class_name::cell_execution_gru_lbr;
                break;
            default: break;
        }

        grid_computation = &class_name::linear_execution;

        size_t scratchpad_size, workspace_size;
        rnn_utils::set_offsets(pd()->rnn_, ws_gates_offset_, ws_states_offset_,
                ws_c_states_offset_, ws_diff_states_offset_,
                ws_grid_comp_offset_, ws_bias_offset_, scratch_gates_offset_,
                scratch_cell_offset_, scratchpad_size, workspace_size);
    }

    ~_ref_rnn_common_t() { delete rnn_postgemm_; }

    // typedef typename prec_traits::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_(ctx);
        return status::success;
    }

private:
    void execute_(const exec_ctx_t &ctx) const;
    rnn_grid_execution_sig(linear_execution);
    rnn_cell_execution_sig(cell_execution);
    rnn_cell_execution_sig(cell_execution_gru);
    rnn_cell_execution_sig(cell_execution_gru_lbr);
    rnn_gemm_sig(gemm);
    rnn_gemm_sig(packed_gemm);
    rnn_bias_prepare_sig(bias_prepare);
    rnn_bias_finalize_sig(bias_finalize);
    rnn_weights_assign_sig(assign_weights);
    rnn_weights_assign_sig(assign_packed_weights);

    float (*activation_func)(float s, float alpha, float cliping);

    void copy_init_layer(const rnn_utils::rnn_conf_t &rnn,
            src_data_t *ws_states_, acc_data_t *ws_diff_states_,
            const src_data_t *xt_, const acc_data_t *diff_dst_layer) const;

    template <typename input_data_t>
    void copy_init_iter(const rnn_utils::rnn_conf_t &rnn,
            src_data_t *ws_states_, float *ws_c_states_,
            acc_data_t *ws_diff_states_, const input_data_t *firstit_states_,
            const float *firstit_c_states_, const acc_data_t *diff_dst_iter_,
            const float *diff_dst_iter_c_) const;

    template <typename dst_layer_dt, typename dst_iter_dt>
    void copy_res_layer(const rnn_utils::rnn_conf_t &rnn,
            dst_layer_dt *dst_layer_, acc_data_t *diff_src_layer_,
            const dst_iter_dt *dst_iter_, const src_data_t *ws_states_,
            const acc_data_t *ws_diff_states_) const;

    template <typename output_data_t, typename dst_data_t>
    void copy_res_iter(const rnn_utils::rnn_conf_t &rnn,
            output_data_t *dst_iter_, float *dst_iter_c_,
            acc_data_t *diff_src_iter_, float *diff_src_iter_c_,
            const dst_data_t *dst_layer_, const src_data_t *ws_states_,
            const float *ws_c_states, const acc_data_t *ws_diff_states_) const;

    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    size_t ws_gates_offset_;
    size_t ws_states_offset_;
    size_t ws_c_states_offset_;
    size_t ws_bias_offset_;
    size_t ws_diff_states_offset_;
    size_t ws_grid_comp_offset_;
    size_t scratch_gates_offset_;
    size_t scratch_cell_offset_;
    rnn_postgemm_dispatcher<aprop, src_type, acc_type> *rnn_postgemm_;

    grid_execution_f grid_computation;
    cell_execution_f cell_func;

    bias_prepare_t bias_preparation_func;
    bias_finalize_t bias_finalization_func;
    weights_assign_t weights_layer_assign_func;
    weights_assign_t weights_iter_assign_func;

    gemm_t gemm_layer_func;
    gemm_t gemm_iter_func;
};

using ref_rnn_fwd_f32_t = _ref_rnn_common_t<prop_kind::forward, data_type::f32,
        data_type::f32, data_type::f32>;
using ref_rnn_bwd_f32_t = _ref_rnn_common_t<prop_kind::backward, data_type::f32,
        data_type::f32, data_type::f32>;
using ref_rnn_fwd_bf16_t = _ref_rnn_common_t<prop_kind::forward,
        data_type::bf16, data_type::bf16, data_type::f32>;
using ref_rnn_bwd_bf16_t = _ref_rnn_common_t<prop_kind::backward,
        data_type::bf16, data_type::bf16, data_type::f32>;
using ref_rnn_fwd_u8s8_t = _ref_rnn_common_t<prop_kind::forward, data_type::u8,
        data_type::s8, data_type::s32>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
