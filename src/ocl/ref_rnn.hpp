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

#ifndef OCL_REF_RNN_HPP
#define OCL_REF_RNN_HPP

#include <assert.h>
#include <stdio.h>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/primitive_iterator.hpp"

#include "ocl/ocl_engine.hpp"
#include "ocl/ocl_rnn_pd.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_memory_storage.hpp"
#include "ocl/ocl_utils.hpp"
#include "ocl/rnn_utils.hpp"
#include "ocl/jit_ref_rnn_kernel.hpp"
#include "ocl/jit_gen9_gemm.hpp"

// not implemented
#define USE_MKL_PACKED_GEMM 0


// just to debug
#define EMULATED_SCRATCHPAD 1

extern const char *ref_rnn_kernel;

////////////////////////////////////////////////////////////
//#define DEBUGPRINT

#ifdef DEBUGPRINT
#define DPRINT(fmt, ...) printf(fmt, __VA_ARGS__);fflush(0)
#define DTHR 0
#define DPRINT1T(fmt, ...) \
    if (ithr==DTHR) printf(fmt, __VA_ARGS__);fflush(0)
#else
#define DPRINT(fmt, ...)
#define DPRINT1T(ithr,fmt, ...)
#endif
////////////////////////////////////////////////////////////

namespace mkldnn {
namespace impl {
namespace ocl {

enum gemm_kind_t {
    gemm_iter,
    gemm_layer,
    gemm_diff_wei_iter,
    gemm_diff_wei_layer
};

template <prop_kind_t aprop, impl::data_type_t src_type,
        impl::data_type_t weights_type>
struct _ref_rnn_common_t : public primitive_t {
    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<weights_type>::type weights_data_t;

    using class_name = _ref_rnn_common_t<aprop, src_type, weights_type>;

    typedef elemwise_sig((class_name::*elemwise_f));
    typedef cell_execution_sig((class_name::*cell_execution_f));
    typedef grid_execution_sig((class_name::*grid_execution_f));

    typedef gemm_sig((class_name::*gemm_t));
    typedef packing_sig((class_name::*packing_t));
    typedef free_packed_sig((class_name::*free_packed_t));

    using base_pd_t =
            typename utils::conditional<false || aprop == prop_kind::forward,
                    ocl_rnn_fwd_pd_t, ocl_rnn_bwd_pd_t>::type;

    struct pd_t : public base_pd_t {

        using base_pd_t::base_pd_t;

        pd_t(const pd_t &other) : base_pd_t(other) { copy_from(other); }

        pd_t &operator=(const pd_t &other) {
            MKLDNN_SHORT_CIRCUIT_SELF_ASSIGN(other);
            clear();
            copy_from(other);
            return *this;
        }

        ~pd_t() { clear(); }

        DECLARE_COMMON_PD_T("ref:any", class_name);

        status_t init() {
            using namespace prop_kind;
            using namespace utils;
            using namespace format_tag;

            assert(this->engine()->kind() == engine_kind::gpu);
            auto *ocl_engine
                = utils::downcast<const ocl_engine_t *>(this->engine());


            const alg_kind_t cell_kind = this->desc()->cell_kind;

            data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
            data_type_t weights_iter_dt
                    = this->desc()->weights_iter_desc.data_type;
            data_type_t weights_layer_dt
                    = this->desc()->weights_layer_desc.data_type;

            bool ok = true
                    && one_of(cell_kind, alg_kind::vanilla_rnn,
                               alg_kind::vanilla_lstm)
                    && IMPLICATION(aprop == prop_kind::forward,
                               one_of(this->desc()->prop_kind, forward_training,
                                       forward_inference))
                    && IMPLICATION(aprop == backward,
                               one_of(this->desc()->prop_kind, backward))
                    && src_layer_dt == src_type
                    && everyone_is(
                               weights_type, weights_iter_dt, weights_layer_dt)
                    && this->set_default_params() == status::success
                    && IMPLICATION(src_type == data_type::f16,
                        this->desc()->prop_kind == forward_inference)
                    && ocl_engine->mayiuse(cl_device_ext_t::intel_subgroups)
                    && IMPLICATION(src_type == data_type::f16, true
                            && ocl_engine->mayiuse(cl_device_ext_t::khr_fp16)
                            && ocl_engine->mayiuse(
                                cl_device_ext_t::intel_subgroups_short));
            if (!ok) {
                return status::unimplemented;
            }

            ok = ok && utils::one_of(cell_kind, alg_kind::vanilla_rnn,
                               alg_kind::vanilla_lstm, alg_kind::vanilla_gru,
                               alg_kind::lbr_gru);

            ok = ok && this->with_bias();
            switch (aprop) {
            case (prop_kind::forward):
                ok = ok && utils::one_of(this->desc()->prop_kind,
                                   forward_training, forward_inference);
                ok = ok
                    && memory_desc_matches_one_of_tag(
                            this->weights_layer_md_, ldigo)
                    && memory_desc_matches_one_of_tag(
                            this->weights_iter_md_, ldigo);
                break;
            case (prop_kind::backward):
                ok = ok && utils::one_of(this->desc()->prop_kind, backward);
                ok = ok
                    && memory_desc_matches_one_of_tag(
                            this->weights_layer_md_, ldgoi)
                    && memory_desc_matches_one_of_tag(
                            this->weights_iter_md_, ldgoi);
                break;
            default: ok = false;
            }
            if (!ok) {
                return status::unimplemented;
            }

            // Check dimensions consistency
            int ls_multiplier
                    = (this->direction() == mkldnn_bidirectional_concat) ? 2 :
                                                                           1;

            ok = ok && (ls_multiplier * this->DIC() == this->DLC())
                    && ((ls_multiplier * this->SLC()) == this->DLC()
                               || (this->L() == 1))
                    && (this->SIC() == this->DIC() || (this->T() == 1));
            if (!ok) {
                return status::unimplemented;
            }

            // initialize the workspace_pd if needed
            if (this->desc()->prop_kind != forward_inference){
                dims_t ws_dims = { (dim_t)rnn_utils::get_ws_size(*this) };
                mkldnn_memory_desc_init_by_tag(&this->ws_md_, 1, ws_dims,
                        src_type, x);
            }

#if !EMULATED_SCRATCHPAD
            auto scratchpad_sz = rnn_utils::get_scratchpad_size(*this);
            init_scratchpad(scratchpad_sz);
#endif

            status_t status = init_base<aprop>(jrnn_, this, this->jit_off_);
            if (status != status::success) {
                return status;
            }

            auto create_gemm_pd = [&](primitive_desc_t **gemm_pd, int m, int n,
                    int k, int lda, int ldb, int ldc, data_type_t a_dt,
                    data_type_t b_dt, data_type_t c_dt, bool is_B_trans,
                    float beta) -> status_t
            {
                gemm_desc_t gemm_desc;
                gemm_desc.primitive_kind = primitive_kind::gemm;
                gemm_desc.transa = transpose::notrans;
                gemm_desc.transb = is_B_trans
                    ? transpose::trans : transpose::notrans;
                gemm_desc.m = m;
                gemm_desc.n = n;
                gemm_desc.k = k;
                gemm_desc.lda = lda;
                gemm_desc.ldb = ldb;
                gemm_desc.ldc = ldc;
                gemm_desc.alpha = 1.0;
                gemm_desc.beta = beta;
                gemm_desc.a_type = a_dt;
                gemm_desc.b_type = b_dt;
                gemm_desc.c_type = c_dt;

                op_desc_t op_desc(gemm_desc);

                primitive_attr_t dummy_attr;

                return mkldnn_primitive_desc_create(gemm_pd, &op_desc,
                        &dummy_attr, this->engine(), nullptr);
            };

            int batch = this->MB();
            int n_gates = this->G();
            int slc = this->SLC();
            int sic = this->SIC();
            int dic = this->DIC();
            int wic = nstl::max(slc, nstl::max(sic, dic));

            bool gemm_ok = true;

            switch(aprop) {
            case prop_kind::forward:
                gemm_ok = true
                    && utils::everyone_is(status::success,
                            create_gemm_pd(&gemm_layer_pd_, n_gates * dic,
                                batch, slc, n_gates * dic, wic, n_gates * dic,
                                weights_type, src_type, src_type, false, 0.0),
                            create_gemm_pd(&gemm_iter_pd_, n_gates * dic,
                                batch, sic, n_gates * dic, wic, n_gates * dic,
                                weights_type, src_type, src_type, false, 1.0));
                break;
            case prop_kind::backward:
                gemm_ok = true
                    && utils::everyone_is(status::success,
                    create_gemm_pd(&gemm_iter_pd_, sic, batch, n_gates * dic,
                        slc, n_gates * dic, wic, weights_type, src_type,
                        src_type, false, 0.0f),
                    create_gemm_pd(&gemm_layer_pd_, slc, batch, n_gates * dic,
                        sic, n_gates * dic, wic, weights_type, src_type,
                        src_type, false, 0.0f),
                    create_gemm_pd(&gemm_diff_wei_layer_pd_, n_gates * dic, slc,
                        batch, n_gates * dic, wic, n_gates * dic, src_type,
                        src_type, weights_type, true, 1.0f),
                    create_gemm_pd(&gemm_diff_wei_iter_pd_, n_gates * dic, sic,
                        batch, n_gates * dic, wic, n_gates * dic, src_type,
                        src_type, weights_type, true, 1.0f));
                break;
            default:
                assert(!"unknown prop_kind");
                return status::invalid_arguments;
            }

            if (!gemm_ok) {
                return status::unimplemented;
            }

            return status::success;
        }

        jit_rnn_conf_t jrnn_;
        jit_rnn_offsets jit_off_;

        primitive_desc_t *gemm_iter_pd_ = nullptr;
        primitive_desc_t *gemm_layer_pd_ = nullptr;
        primitive_desc_t *gemm_diff_wei_layer_pd_ = nullptr;
        primitive_desc_t *gemm_diff_wei_iter_pd_ = nullptr;

    private:
        void init_scratchpad(size_t scratchpad_sz) {
            using namespace memory_tracking::names;
            auto scratchpad = this->scratchpad_registry().registrar();
            scratchpad.book(key_rnn_space, sizeof(float) * scratchpad_sz, 4096);
        }

        void copy_from(const pd_t &other) {
            jrnn_ = other.jrnn_;
            jit_off_ = other.jit_off_;
            gemm_layer_pd_ = other.gemm_layer_pd_
                ? other.gemm_layer_pd_->clone() : nullptr;
            gemm_iter_pd_ = other.gemm_iter_pd_
                ? other.gemm_iter_pd_->clone() : nullptr;
            gemm_diff_wei_layer_pd_ = other.gemm_diff_wei_layer_pd_
                ? other.gemm_diff_wei_layer_pd_->clone() : nullptr;
            gemm_diff_wei_iter_pd_ = other.gemm_diff_wei_iter_pd_
                ? other.gemm_diff_wei_iter_pd_->clone() : nullptr;
        }

        void clear() {
            delete gemm_layer_pd_;
            delete gemm_iter_pd_;
            delete gemm_diff_wei_layer_pd_;
            delete gemm_diff_wei_iter_pd_;
        }

    };  // struct pd_t : public base_pd_t

    status_t init() override {
        auto jit = ocl_jit_t(ref_rnn_kernel);
        jit_ref_rnn_kernel::init_const_def(jit, pd()->jrnn_, pd()->jit_off_);

        status_t status = jit.build(engine());
        if (status != status::success)
            return status;

        copy_init_layer_kernel_
            = jit.get_kernel("ref_rnn_copy_init_layer_kernel");
        if (!copy_init_layer_kernel_) return status::runtime_error;
        copy_init_iter_kernel_
            = jit.get_kernel("ref_rnn_copy_init_iter_kernel");
        if (!copy_init_iter_kernel_) return status::runtime_error;
        copy_res_layer_kernel_
            = jit.get_kernel("ref_rnn_copy_res_layer_kernel");
        if (!copy_res_layer_kernel_) return status::runtime_error;
        copy_res_iter_kernel_ = jit.get_kernel("ref_rnn_copy_res_iter_kernel");
        if (!copy_res_iter_kernel_) return status::runtime_error;
        ws_set_kernel_ = jit.get_kernel("ref_rnn_ws_set_kernel");
        if (!ws_set_kernel_) return status::runtime_error;
        elemwise_fwd_kernel_ = jit.get_kernel("ref_rnn_elemwise_fwd_kernel");
        if (!elemwise_fwd_kernel_) return status::runtime_error;
        elemwise_bwd_kernel_ = jit.get_kernel("ref_rnn_elemwise_bwd_kernel");
        if (!elemwise_bwd_kernel_) return status::runtime_error;
        gates_reduction_kernel_
            = jit.get_kernel("ref_rnn_gates_reduction_kernel");
        if (!gates_reduction_kernel_) return status::runtime_error;

        bool gemm_ok = true;

        switch(aprop) {
        case prop_kind::forward:
            gemm_ok = true
                && utils::everyone_is(status::success,
                        pd()->gemm_layer_pd_->create_primitive(&gemm_layer_),
                        pd()->gemm_iter_pd_->create_primitive(&gemm_iter_));
            break;
        case prop_kind::backward:
            gemm_ok = true
                && utils::everyone_is(status::success,
                        pd()->gemm_iter_pd_->create_primitive(&gemm_iter_),
                        pd()->gemm_layer_pd_->create_primitive(&gemm_layer_),
                        pd()->gemm_diff_wei_layer_pd_->create_primitive(&gemm_diff_wei_layer_),
                        pd()->gemm_diff_wei_iter_pd_->create_primitive(&gemm_diff_wei_iter_)
                    );
            break;
        default:
            assert(!"unknown prop_kind");
            return status::invalid_arguments;
        }

        if (!gemm_ok)
            return status::runtime_error;

#if EMULATED_SCRATCHPAD
        if (use_scratchpad_) {
            auto scratchpad_sz = rnn_utils::get_scratchpad_size(*this->pd());
            engine()->create_memory_storage(&scratchpad_,
                    scratchpad_sz * sizeof(src_data_t));
        }
#endif
        return status::success;
    } // status_t init() override

    //
    // constructor
    //
    _ref_rnn_common_t(const pd_t *apd)
        : primitive_t(apd) {

        using namespace rnn_utils;
        /// @todo set max_feature_size assuming that we limit the number of
        /// iterations and layer to one if slc != dic and sic != dic
        /// respectively
        ///
        ker_ = new jit_ref_rnn_kernel(pd()->jrnn_);

#if USE_MKL_PACKED_GEMM
// TBD
#else
        use_layer_packed_gemm_ = false;
        use_iter_packed_gemm_ = false;
#endif

        auto set_pack_funcs = [](bool packed_gemm, gemm_t &g, bool pack_w,
                packing_t &p, free_packed_t &f) {
            g = packed_gemm ? &class_name::packed_gemm
                : &class_name::gemm_primitive;
            p = pack_w ? &class_name::pack_weights :
                             &class_name::no_pack_weights;
            f = pack_w ? &class_name::free_packed_weights :
                             &class_name::free_no_packed_weights;
        };

        set_pack_funcs(false, gemm_iter_func, false, weights_state_pack_func,
                weights_state_free_packed_func);

        set_pack_funcs(false, gemm_layer_func, false, weights_input_pack_func,
                weights_input_free_packed_func);

        switch (pd()->cell_kind()) {
        case alg_kind::vanilla_lstm:
            cell_func = &class_name::cell_execution;
            elemwise_func = &class_name::lstm_elemwise;
            break;
        case alg_kind::vanilla_rnn: // @todo switch on cell kind
            cell_func = &class_name::cell_execution;
            elemwise_func = &class_name::rnn_elemwise;
            break;
        case alg_kind::vanilla_gru:
            cell_func = &class_name::cell_execution_gru;
            break;
        case alg_kind::lbr_gru:
            cell_func = &class_name::cell_execution_gru_lbr;
            elemwise_func = &class_name::gru_lbr_elemwise;
            break;
        default: break;
        }

        n_output_features
                = (pd()->direction() == mkldnn_bidirectional_concat) ? 2 : 1;
        switch (pd()->direction()) {
        case mkldnn_unidirectional_left2right: exec_dir = b2t_l2r; break;
        case mkldnn_unidirectional_right2left: exec_dir = b2t_r2l; break;
        case mkldnn_bidirectional_concat: exec_dir = b2t_bi_concat; break;
        case mkldnn_bidirectional_sum: exec_dir = b2t_bi_sum; break;
        default: break;
        }

        /// @todo put a heuristic to choose between linear execution and
        /// wavefront
        grid_computation = &class_name::linear_execution;

        set_offsets(*pd(), ws_gates_offset_, ws_states_offset_,
                ws_diff_states_offset_, ws_grid_comp_offset_,
                ws_cell_comp_offset_);

        // we need to allocate memory for:
        // - the states to compute a pass.
        // - the intermediate results from the gates.
        // - the diff_states to compute the backward pass (training only)
        // These should be allocated on scratchpad if fwd inference
        // or on a workspace provided by the user for training.
        /// @todo shall we require the workspace for training or make it
        /// optional?

        // if no worskpace is provided on forward, we use a scratchpad
        // NOTE: here we use a large worskpace for simplicity:
        // - for states:
        //   - TODO: allocate only n_iter * dic + dic for linear execution
        //   (inference)
        //   - TODO: allocate only n_layer_wav * (2*dic) for wavefront
        //   execution (inference)
        // - for gates:
        //   - TODO: allocate only batch * n_gates * dic for linear execution
        //   (inference)
        //   = TODO: allocate only n_layer_wav * batch * n_gates * dic for
        //   wavefront execution (inference)

        use_scratchpad_
            = (pd()->desc()->prop_kind == prop_kind::forward_inference);
        use_workspace_ = !use_scratchpad_;

        int max_nparts = (pd()->cell_kind() == alg_kind::vanilla_gru) ? 2 : 1;
        int ptr_wei_sz = pd()->L() * pd()->D() * max_nparts; // as unused, tmp

        offset_wei_input_ = (size_t *)malloc(sizeof(size_t) * ptr_wei_sz, 64);
        offset_wei_state_ = (size_t *)malloc(sizeof(size_t) * ptr_wei_sz, 64);
    }

    ~_ref_rnn_common_t() {
        delete ker_;
#if EMULATED_SCRATCHPAD
        if (use_scratchpad_) {
            // SK : good wor emulated scratchpad too
            delete scratchpad_;
        }
#endif
        free(offset_wei_input_);
        free(offset_wei_state_);

        delete gemm_iter_;
        delete gemm_layer_;
        delete gemm_diff_wei_layer_;
        delete gemm_diff_wei_iter_;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_(ctx);
    }

private:
    status_t execute_(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    grid_execution_sig(linear_execution);
    // grid_execution_sig(wavefront_execution);
    cell_execution_sig(cell_execution);
    cell_execution_sig(cell_execution_gru);
    cell_execution_sig(cell_execution_gru_lbr);
    elemwise_sig(rnn_elemwise);
    elemwise_sig(lstm_elemwise);
    elemwise_sig(gru_lbr_elemwise);
    gemm_sig(gemm_primitive);
    gemm_sig(packed_gemm);
    packing_sig(pack_weights);
    packing_sig(no_pack_weights);
    free_packed_sig(free_packed_weights);
    free_packed_sig(free_no_packed_weights);

    float (*activation_func)(float dd, float s, float alpha, float cliping);
    void copy_init_layer(const stream_t *s, bool lr, bool rl, int n_iter,
            int batch, int slc, const memory_storage_t &ws,
            const memory_storage_t &input,
            const memory_storage_t &diff_dst_layer) const;
    void copy_init_iter(const stream_t *s, int n_layer, int n_direction,
            int batch, int sic, int dic,
            const memory_storage_t &ws, const memory_storage_t &firstit_states,
            const memory_storage_t &firstit_c_states,
            const memory_storage_t &diff_dst_iter,
            const memory_storage_t &diff_dst_iter_c) const;
    void copy_res_layer(const stream_t *s, bool lr, bool rl, int n_iter,
            int batch, int slc, int dlc, const memory_storage_t &dst_last_layer,
            const memory_storage_t &diff_src_layer,
            const memory_storage_t &ws) const;
    void copy_res_iter(const stream_t *s, int n_layer, int n_direction,
            int batch, int sic, int dic,
            const memory_storage_t &dst_last_iter,
            const memory_storage_t &dst_last_iter_c,
            const memory_storage_t &diff_src_iter,
            const memory_storage_t &diff_src_iter_c,
            const memory_storage_t &ws) const;
    void gates_reduction(const exec_ctx_t &ctx, int dir, int lay, int iter,
            int n_gates, int dic, int batch, const memory_storage_t &gates,
            const memory_storage_t &diff_bias) const;
    void ws_set(const stream_t *s, const memory_storage_t &workspace,
            const cl_ulong ws_offset, const float val, const size_t size) const;

    jit_ref_rnn_kernel *ker_;
    ocl_kernel_t copy_init_layer_kernel_;
    ocl_kernel_t copy_init_iter_kernel_;
    ocl_kernel_t copy_res_layer_kernel_;
    ocl_kernel_t copy_res_iter_kernel_;

    ocl_kernel_t ws_set_kernel_;
    ocl_kernel_t elemwise_fwd_kernel_;
    ocl_kernel_t elemwise_bwd_kernel_;
    ocl_kernel_t gates_reduction_kernel_;

    /* GEMM primitives */
    primitive_t *gemm_layer_ = nullptr;
    primitive_t *gemm_iter_ = nullptr;
    primitive_t *gemm_diff_wei_layer_ = nullptr;
    primitive_t *gemm_diff_wei_iter_ = nullptr;

    bool use_workspace_;
    bool use_scratchpad_;

    memory_storage_t *scratchpad_;

    bool use_layer_packed_gemm_;
    bool use_iter_packed_gemm_;

    cl_ulong ws_gates_offset_;      // used as params to ocl kernel
    cl_ulong ws_states_offset_;
    cl_ulong ws_diff_states_offset_;
    cl_ulong ws_grid_comp_offset_;
    cl_ulong ws_cell_comp_offset_;

    int n_output_features;

    size_t *offset_wei_input_;
    size_t *offset_wei_state_;

    rnn_utils::execution_direction exec_dir;
    grid_execution_f grid_computation;
    cell_execution_f cell_func;

    packing_t weights_input_pack_func;
    packing_t weights_state_pack_func;

    gemm_t gemm_iter_func;
    gemm_t gemm_layer_func;
    elemwise_f elemwise_func;

    free_packed_t weights_input_free_packed_func;
    free_packed_t weights_state_free_packed_func;
};
using ref_rnn_fwd_f16_t
    = _ref_rnn_common_t<prop_kind::forward, data_type::f16, data_type::f16>;
using ref_rnn_fwd_f32_t
    = _ref_rnn_common_t<prop_kind::forward, data_type::f32, data_type::f32>;
using ref_rnn_bwd_f32_t
    = _ref_rnn_common_t<prop_kind::backward, data_type::f32, data_type::f32>;
}
}
}
#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
