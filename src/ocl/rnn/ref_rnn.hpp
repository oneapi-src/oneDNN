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
#include "common/primitive_iterator.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "compute/compute.hpp"

#include "ocl/gemm/ocl_gemm.hpp"
#include "ocl/ocl_memory_storage.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"
#include "ocl/rnn/jit_ref_rnn_kernel.hpp"
#include "ocl/rnn/ocl_rnn_pd.hpp"
#include "ocl/rnn/rnn_utils.hpp"

// not implemented
#define USE_MKL_PACKED_GEMM 0

// TODO just to debug
#define WS_NAN_FILLING 0

namespace dnnl {
namespace impl {
namespace ocl {

enum gemm_kind_t {
    gemm_iter_fwd,
    gemm_layer_fwd,
    gemm_iter_bwd,
    gemm_layer_bwd,
    gemm_diff_wei_iter,
    gemm_diff_wei_layer
};

template <prop_kind_t aprop>
struct _ref_rnn_common_t : public primitive_impl_t {

    using class_name = _ref_rnn_common_t<aprop>;

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
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(other);
            base_pd_t::operator=(other);
            clear();
            copy_from(other);
            return *this;
        }

        ~pd_t() { clear(); }

        DECLARE_COMMON_PD_T("ref:any", class_name);

        status_t init() {
            using namespace prop_kind;
            using namespace utils;
            using namespace rnn_utils;
            using namespace format_tag;

            assert(this->engine()->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<const compute::compute_engine_t *>(
                            this->engine());

            const alg_kind_t cell_kind = this->desc()->cell_kind;

            data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
            data_type_t weights_iter_dt
                    = this->desc()->weights_iter_desc.data_type;
            data_type_t weights_layer_dt
                    = this->desc()->weights_layer_desc.data_type;
            bool src_is_u8 = src_layer_dt == data_type::u8;
            bool src_is_f16 = src_layer_dt == data_type::f16;
            if (src_is_u8 && !src_is_f16)
                acc_data_t = data_type::s32;
            else if (!src_is_u8 && src_is_f16)
                acc_data_t = data_type::f16;
            else if (!src_is_u8 && !src_is_f16)
                acc_data_t = data_type::f32;
            src_type = src_layer_dt;
            weights_type = weights_layer_dt;

            bool ok = true
                    && one_of(cell_kind, alg_kind::vanilla_rnn,
                            alg_kind::vanilla_lstm)
                    && IMPLICATION(aprop == prop_kind::forward,
                            one_of(this->desc()->prop_kind, forward_training,
                                    forward_inference))
                    && IMPLICATION(aprop == backward,
                            one_of(this->desc()->prop_kind, backward))
                    && src_layer_dt == src_type
                    && ((aprop == prop_kind::forward
                                && src_layer_dt == data_type::u8
                                && weights_layer_dt == data_type::s8)
                            || (aprop == prop_kind::forward
                                    && one_of(src_layer_dt, data_type::f16,
                                            data_type::f32, data_type::bf16)
                                    && weights_layer_dt == src_layer_dt)
                            || (aprop == prop_kind::backward
                                    && one_of(weights_layer_dt, data_type::f32,
                                            data_type::bf16)
                                    && weights_layer_dt == src_layer_dt))
                    && weights_iter_dt == weights_layer_dt
                    && everyone_is(
                            weights_type, weights_iter_dt, weights_layer_dt)
                    && this->set_default_params() == status::success
                    && this->with_bias()
                    && IMPLICATION(src_type == data_type::f16
                                    || src_type == data_type::u8,
                            this->desc()->prop_kind == forward_inference)
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && IMPLICATION(src_type == data_type::f16,
                            true
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::khr_fp16)
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::
                                                    intel_subgroups_short));
            if (!ok) return status::unimplemented;

            init_rnn_conf(rnn_conf_, *this->desc(), this->src_md(0),
                    this->src_md(1), this->weights_md(0), this->weights_md(1),
                    this->dst_md(0));
            init_test_mode(rnn_conf_, *this->attr());

            /* check that only supported attr have been passed */
            primitive_attr_t::skip_mask_t attr_mask
                    = primitive_attr_t::skip_mask_t::rnn_tparams;
            if (weights_layer_dt == data_type::s8)
                attr_mask = attr_mask
                        | primitive_attr_t::skip_mask_t::rnn_data_qparams
                        | primitive_attr_t::skip_mask_t::rnn_weights_qparams;
            ok = ok && this->attr()->has_default_values(attr_mask);

            // TODO: implement something like check layout consistency
            switch (aprop) {
                case (prop_kind::forward): break;
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
            if (!ok) return status::unimplemented;

            // Set weights descriptors to desired format
            memory_desc_t new_weights_layer_md = *this->weights_md(0);
            CHECK(set_expected_desc(rnn_conf_, new_weights_layer_md, false));

            if (this->weights_layer_md_.format_kind == format_kind::any) {
                this->weights_layer_md_ = new_weights_layer_md;
            } else if (this->weights_layer_md_.format_kind
                    == format_kind::rnn_packed) {
                if (dnnl::impl::operator!=(
                            this->weights_layer_md_, new_weights_layer_md))
                    return status::unimplemented;
            }

            memory_desc_t new_weights_iter_md = *this->weights_md(1);
            CHECK(set_expected_desc(rnn_conf_, new_weights_iter_md, true));
            if (this->weights_iter_md_.format_kind == format_kind::any) {
                this->weights_iter_md_ = new_weights_iter_md;
            } else if (this->weights_iter_md_.format_kind
                    == format_kind::rnn_packed) {
                if (dnnl::impl::operator!=(
                            this->weights_iter_md_, new_weights_iter_md))
                    return status::unimplemented;
            }

            // Check dimensions consistency
            int ls_multiplier
                    = (this->direction() == dnnl_bidirectional_concat) ? 2 : 1;

            ok = ok && (ls_multiplier * this->DIC() == this->DLC())
                    && ((ls_multiplier * this->SLC()) == this->DLC()
                            || (this->L() == 1))
                    && (this->SIC() == this->DIC() || (this->T() == 1));
            if (!ok) return status::unimplemented;

            set_rnn_conf(rnn_conf_, *this->desc(), this->weights_md(0),
                    this->weights_md(1), this->diff_weights_md(0),
                    this->diff_weights_md(1));

            size_t scratchpad_sz {0}, ws_sz {0};
            get_scratchpad_and_workspace_sizes(rnn_conf_, scratchpad_sz, ws_sz);

            // initialize the workspace_pd if needed
            if (rnn_conf_.use_workspace) {
                dims_t ws_dims = {(dim_t)ws_sz};
                dnnl_memory_desc_init_by_tag(
                        &this->ws_md_, 1, ws_dims, data_type::u8, x);
            }

            init_scratchpad(scratchpad_sz);

            rnn_conf_.acc_data_type = acc_data_t;
            rnn_conf_.acc_data_type_elsz = types::data_type_size(acc_data_t);
            status_t status
                    = init_jit<aprop>(jrnn_, rnn_conf_, this, this->jit_off_);
            if (status != status::success) { return status; }

            auto create_gemm_pd
                    = [&](primitive_desc_t **gemm_pd, int m, int n, int k,
                              int lda, int ldb, int ldc, data_type_t a_dt,
                              data_type_t b_dt, data_type_t c_dt,
                              bool is_B_trans, float beta) -> status_t {
                gemm_desc_t gemm_desc;
                gemm_desc.primitive_kind = primitive_kind::gemm;
                gemm_desc.transa = transpose::notrans;
                gemm_desc.transb
                        = is_B_trans ? transpose::trans : transpose::notrans;
                gemm_desc.batch = 1;
                gemm_desc.m = m;
                gemm_desc.n = n;
                gemm_desc.k = k;
                gemm_desc.lda = lda;
                gemm_desc.ldb = ldb;
                gemm_desc.ldc = ldc;
                gemm_desc.stride_a = lda;
                gemm_desc.stride_b = ldb;
                gemm_desc.stride_c = ldc;
                gemm_desc.a_type = a_dt;
                gemm_desc.b_type = b_dt;
                gemm_desc.c_type = c_dt;
                gemm_desc.acc_type = c_dt;

                primitive_attr_t dummy_attr;
                dummy_attr.post_ops_.append_sum(beta);

                return dnnl_primitive_desc_create(gemm_pd,
                        (op_desc_t *)&gemm_desc, &dummy_attr, this->engine(),
                        nullptr);
            };

            int batch = rnn_conf_.mb;
            int n_gates = rnn_conf_.n_gates;
            int slc = rnn_conf_.slc;
            int sic = rnn_conf_.sic;
            int dic = rnn_conf_.dic;

            int layer_merged_size = rnn_conf_.merge_gemm_layer
                    ? batch * rnn_conf_.n_iter
                    : batch;
            int iter_merged_size = rnn_conf_.merge_gemm_iter
                    ? batch * rnn_conf_.n_iter
                    : batch;

            bool gemm_ok = true;

            switch (aprop) {
                case prop_kind::forward:
                    gemm_ok = true
                            && utils::everyone_is(status::success,
                                    create_gemm_pd(&gemm_layer_fwd_pd_,
                                            n_gates * dic, layer_merged_size,
                                            slc, rnn_conf_.weights_layer_ld,
                                            rnn_conf_.states_ws_ld,
                                            rnn_conf_.gates_ws_ld, weights_type,
                                            src_type, rnn_conf_.acc_data_type,
                                            false, 0.0),
                                    create_gemm_pd(&gemm_iter_fwd_pd_,
                                            n_gates * dic, batch, sic,
                                            rnn_conf_.weights_iter_ld,
                                            rnn_conf_.states_ws_ld,
                                            rnn_conf_.gates_ws_ld, weights_type,
                                            src_type, rnn_conf_.acc_data_type,
                                            false, 1.0));
                    break;
                case prop_kind::backward:
                    gemm_ok = true
                            && utils::everyone_is(status::success,
                                    create_gemm_pd(&gemm_iter_bwd_pd_, sic,
                                            batch, n_gates * dic,
                                            rnn_conf_.weights_iter_ld,
                                            rnn_conf_.scratch_gates_ld,
                                            rnn_conf_.diff_states_ws_ld,
                                            weights_type, src_type,
                                            rnn_conf_.acc_data_type, false,
                                            0.0f),
                                    create_gemm_pd(&gemm_layer_bwd_pd_, slc,
                                            layer_merged_size, n_gates * dic,
                                            rnn_conf_.weights_layer_ld,
                                            rnn_conf_.scratch_gates_ld,
                                            rnn_conf_.diff_states_ws_ld,
                                            weights_type, src_type,
                                            rnn_conf_.acc_data_type, false,
                                            0.0f),
                                    create_gemm_pd(&gemm_diff_wei_layer_pd_,
                                            n_gates * dic, slc,
                                            layer_merged_size,
                                            rnn_conf_.scratch_gates_ld,
                                            rnn_conf_.states_ws_ld,
                                            rnn_conf_.diff_weights_layer_ld,
                                            weights_type, src_type,
                                            rnn_conf_.acc_data_type, true,
                                            1.0f),
                                    create_gemm_pd(&gemm_diff_wei_iter_pd_,
                                            n_gates * dic, sic,
                                            iter_merged_size,
                                            rnn_conf_.scratch_gates_ld,
                                            rnn_conf_.states_ws_ld,
                                            rnn_conf_.diff_weights_iter_ld,
                                            weights_type, src_type,
                                            rnn_conf_.acc_data_type, true,
                                            1.0f));
                    break;
                default:
                    assert(!"unknown prop_kind");
                    return status::invalid_arguments;
            }

            if (!gemm_ok) return status::unimplemented;
            return status::success;
        }

        jit_rnn_conf_t jrnn_;
        jit_rnn_offsets jit_off_;
        rnn_utils::rnn_conf_t rnn_conf_;
        data_type_t acc_data_t;
        data_type_t src_type;
        data_type_t weights_type;

        primitive_desc_t *gemm_iter_fwd_pd_ = nullptr;
        primitive_desc_t *gemm_layer_fwd_pd_ = nullptr;
        primitive_desc_t *gemm_iter_bwd_pd_ = nullptr;
        primitive_desc_t *gemm_layer_bwd_pd_ = nullptr;
        primitive_desc_t *gemm_diff_wei_layer_pd_ = nullptr;
        primitive_desc_t *gemm_diff_wei_iter_pd_ = nullptr;

    private:
        void init_scratchpad(size_t scratchpad_sz) {
            using namespace memory_tracking::names;
            auto scratchpad = this->scratchpad_registry().registrar();
            scratchpad.book(key_rnn_space, scratchpad_sz, 4096);
            scratchpad.book(key_rnn_gates, rnn_conf_.scratch_gates_size, 4096);
        }

        void copy_from(const pd_t &other) {
            jrnn_ = other.jrnn_;
            jit_off_ = other.jit_off_;
            rnn_conf_ = other.rnn_conf_;
            acc_data_t = other.acc_data_t;
            src_type = other.src_type;
            weights_type = other.weights_type;
            gemm_layer_fwd_pd_ = other.gemm_layer_fwd_pd_
                    ? other.gemm_layer_fwd_pd_->clone()
                    : nullptr;
            gemm_iter_fwd_pd_ = other.gemm_iter_fwd_pd_
                    ? other.gemm_iter_fwd_pd_->clone()
                    : nullptr;
            gemm_layer_bwd_pd_ = other.gemm_layer_bwd_pd_
                    ? other.gemm_layer_bwd_pd_->clone()
                    : nullptr;
            gemm_iter_bwd_pd_ = other.gemm_iter_bwd_pd_
                    ? other.gemm_iter_bwd_pd_->clone()
                    : nullptr;
            gemm_diff_wei_layer_pd_ = other.gemm_diff_wei_layer_pd_
                    ? other.gemm_diff_wei_layer_pd_->clone()
                    : nullptr;
            gemm_diff_wei_iter_pd_ = other.gemm_diff_wei_iter_pd_
                    ? other.gemm_diff_wei_iter_pd_->clone()
                    : nullptr;
        }

        void clear() {
            delete gemm_layer_fwd_pd_;
            delete gemm_iter_fwd_pd_;
            delete gemm_layer_bwd_pd_;
            delete gemm_iter_bwd_pd_;
            delete gemm_diff_wei_layer_pd_;
            delete gemm_diff_wei_iter_pd_;
        }

    }; // struct pd_t : public base_pd_t

    status_t init() override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute::kernel_ctx_t kernel_ctx;

        jit_ref_rnn_kernel::init_const_def(
                kernel_ctx, pd()->jrnn_, pd()->jit_off_);

        std::vector<const char *> kernel_names
                = { "ref_rnn_bias_prepare",
                      "ref_rnn_copy_init_layer",
                      "ref_rnn_copy_init_iter",
                      "ref_rnn_copy_res_layer",
                      "ref_rnn_copy_res_iter",
                      "ref_rnn_ws_set",
                      "ref_rnn_elemwise_fwd",
                      "ref_rnn_elemwise_bwd",
                      "ref_rnn_gates_reduction"
#if DEBUGPRINT
                      ,
                      "ref_rnn_ws_print"
#endif
                  };

        std::vector<compute::kernel_t> kernels;
        auto status = compute_engine->create_kernels(
                &kernels, kernel_names, kernel_ctx);
        CHECK(status);

        bias_prepare_kernel_ = kernels[0];
        copy_init_layer_kernel_ = kernels[1];
        copy_init_iter_kernel_ = kernels[2];
        copy_res_layer_kernel_ = kernels[3];
        copy_res_iter_kernel_ = kernels[4];
        ws_set_kernel_ = kernels[5];
        elemwise_fwd_kernel_ = kernels[6];
        elemwise_bwd_kernel_ = kernels[7];
        gates_reduction_kernel_ = kernels[8];
#if DEBUGPRINT
        ws_print_kernel_ = kernels[9];
#endif

        if (pd()->rnn_conf_.is_int8) {
            size_t size = pd()->rnn_conf_.n_gates * pd()->rnn_conf_.dic
                    * sizeof(float); // G * O * sizeof(float);
            memory_storage_t *temp_buf_ptr;
            engine()->create_memory_storage(&temp_buf_ptr, size);
            scales_buf_.reset(temp_buf_ptr);
            if (!scales_buf_) return status::runtime_error;
        }

        // Prepare testmode scales defined by attributes. Doesn't introduce
        // primitive state, because it is a constant memory -- will not be
        // changed during execution.
        // TODO: add the testmode scales to ws
        if (pd()->rnn_conf_.is_testmode && pd_->attr()->rnn_tparams_.scales_) {
            size_t size = pd()->rnn_conf_.tm_ngates
                    * sizeof(*pd_->attr()->rnn_tparams_.scales_);
            memory_storage_t *temp_buf_ptr;
            engine()->create_memory_storage(&temp_buf_ptr, size);
            tm_scales_buf_.reset(temp_buf_ptr);
            if (!tm_scales_buf_) return status::runtime_error;

            void *tmp_ptr = nullptr;
            status = tm_scales_buf_->map_data(&tmp_ptr);
            if (status != status::success) return status;
            utils::array_copy((float *)tmp_ptr,
                    pd()->attr()->rnn_tparams_.scales_,
                    pd()->attr()->rnn_tparams_.ngates_);
            status = tm_scales_buf_->unmap_data(tmp_ptr);
            if (status != status::success) return status;
        }

        bool gemm_ok = true;

        switch (aprop) {
            case prop_kind::forward:
                gemm_ok = true
                        && utils::everyone_is(status::success,
                                pd()->gemm_layer_fwd_pd_->create_primitive(
                                        &gemm_layer_fwd_),
                                pd()->gemm_iter_fwd_pd_->create_primitive(
                                        &gemm_iter_fwd_));
                break;
            case prop_kind::backward:
                gemm_ok = true
                        && utils::everyone_is(status::success,
                                pd()->gemm_layer_bwd_pd_->create_primitive(
                                        &gemm_layer_bwd_),
                                pd()->gemm_iter_bwd_pd_->create_primitive(
                                        &gemm_iter_bwd_),
                                pd()->gemm_diff_wei_layer_pd_->create_primitive(
                                        &gemm_diff_wei_layer_),
                                pd()->gemm_diff_wei_iter_pd_->create_primitive(
                                        &gemm_diff_wei_iter_));
                break;
            default:
                assert(!"unknown prop_kind");
                return status::invalid_arguments;
        }

        if (!gemm_ok) return status::runtime_error;

        return status::success;
    } // status_t init() override

    _ref_rnn_common_t(const pd_t *apd) : primitive_impl_t(apd) {
        using namespace rnn_utils;
        /// @todo set max_feature_size assuming that we limit the number of
        /// iterations and layer to one if slc != dic and sic != dic
        /// respectively
        ///
        ker_ = new jit_ref_rnn_kernel(pd()->jrnn_);

        auto set_pack_funcs = [](bool packed_gemm, gemm_t &g, bool pack_w,
                                      packing_t &p, free_packed_t &f) {
            g = packed_gemm ? &class_name::packed_gemm
                            : &class_name::gemm_primitive;
            p = pack_w ? &class_name::pack_weights
                       : &class_name::no_pack_weights;
            f = pack_w ? &class_name::free_packed_weights
                       : &class_name::free_no_packed_weights;
        };

        set_pack_funcs(false, gemm_iter_func, false, weights_state_pack_func,
                weights_state_free_packed_func);

        set_pack_funcs(false, gemm_layer_func, false, weights_input_pack_func,
                weights_input_free_packed_func);

        switch (pd()->cell_kind()) {
            case alg_kind::vanilla_lstm:
                cell_func = &class_name::cell_execution;
                elemwise_func = pd()->src_type == data_type::u8
                                && pd()->weights_type == data_type::s8
                        ? &class_name::lstm_elemwise_u8s8
                        : &class_name::lstm_elemwise;
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

        /// @todo put a heuristic to choose between linear execution and
        /// wavefront
        grid_computation = &class_name::linear_execution;

        size_t scratchpad_size, workspace_size;
        rnn_utils::set_offsets(pd()->rnn_conf_, ws_gates_offset_,
                ws_states_offset_, ws_c_states_offset_, ws_diff_states_offset_,
                ws_grid_comp_offset_, ws_cell_comp_offset_, ws_bias_offset_,
                scratch_gates_offset_, scratchpad_size, workspace_size);

        int max_nparts = (pd()->cell_kind() == alg_kind::vanilla_gru) ? 2 : 1;
        int offset_wei_sz = pd()->L() * pd()->D() * max_nparts;

        offset_wei_input_
                = (size_t *)malloc(sizeof(size_t) * offset_wei_sz, 64);
        offset_wei_state_
                = (size_t *)malloc(sizeof(size_t) * offset_wei_sz, 64);
    }

    ~_ref_rnn_common_t() {
        delete ker_;
        free(offset_wei_input_);
        free(offset_wei_state_);

        delete gemm_iter_fwd_;
        delete gemm_layer_fwd_;
        delete gemm_iter_bwd_;
        delete gemm_layer_bwd_;
        delete gemm_diff_wei_layer_;
        delete gemm_diff_wei_iter_;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_(ctx);
    }

private:
    status_t execute_(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    grid_execution_sig(linear_execution);
    // grid_execution_sig(wavefront_execution);
    cell_execution_sig(cell_execution);
    cell_execution_sig(cell_execution_gru);
    cell_execution_sig(cell_execution_gru_lbr);
    elemwise_sig(rnn_elemwise);
    elemwise_sig(lstm_elemwise);
    elemwise_sig(lstm_elemwise_u8s8);
    elemwise_sig(gru_lbr_elemwise);
    gemm_sig(gemm_primitive);
    gemm_sig(packed_gemm);
    packing_sig(pack_weights);
    packing_sig(no_pack_weights);
    free_packed_sig(free_packed_weights);
    free_packed_sig(free_no_packed_weights);

    float (*activation_func)(float dd, float s, float alpha, float cliping);
    void bias_prepare(compute::compute_stream_t *compute_stream, int n_layer,
            int n_dir, int n_bias, int n_gates, int dic,
            const memory_storage_t &ws, const memory_storage_t &scales,
            const memory_storage_t &wei_layer, const memory_storage_t &wei_iter,
            const memory_storage_t &bias) const;
    void copy_init_layer(compute::compute_stream_t *compute_stream, bool lr,
            bool rl, int n_iter, int batch, int slc, const memory_storage_t &ws,
            const memory_storage_t &input,
            const memory_storage_t &diff_dst_layer) const;
    void copy_init_iter(compute::compute_stream_t *compute_stream, int n_layer,
            int n_dir, int batch, int sic, int dic, const memory_storage_t &ws,
            const memory_storage_t &firstit_states,
            const memory_storage_t &firstit_c_states,
            const memory_storage_t &diff_dst_iter,
            const memory_storage_t &diff_dst_iter_c, const float shift,
            const float scale, const bool quantize) const;
    void copy_res_layer(compute::compute_stream_t *compute_stream, bool lr,
            bool rl, int n_iter, int batch, int slc, int dlc,
            const memory_storage_t &dst_last_layer,
            const memory_storage_t &diff_src_layer, const memory_storage_t &ws,
            const float shift, const float scale, const bool dequantize) const;
    void copy_res_iter(compute::compute_stream_t *compute_stream, int n_layer,
            int n_dir, int batch, int sic, int dic,
            const memory_storage_t &dst_last_iter,
            const memory_storage_t &dst_last_iter_c,
            const memory_storage_t &diff_src_iter,
            const memory_storage_t &diff_src_iter_c, const memory_storage_t &ws,
            const float shift, const float scale, const bool dequantize) const;
    void gates_reduction(const exec_ctx_t &ctx, int dir, int lay, int iter,
            int n_gates, int dic, int batch, const memory_storage_t &gates,
            const memory_storage_t &diff_bias) const;
    void ws_set(compute::compute_stream_t *compute_stream,
            const memory_storage_t &workspace, const cl_ulong ws_offset,
            const int ws_part, const float val, const size_t size) const;
#if DEBUGPRINT
    void ws_print(compute::compute_stream_t *s,
            const memory_storage_t &workspace) const;
    compute::kernel_t ws_print_kernel_;
#endif

    jit_ref_rnn_kernel *ker_;
    compute::kernel_t bias_prepare_kernel_;
    compute::kernel_t copy_init_layer_kernel_;
    compute::kernel_t copy_init_iter_kernel_;
    compute::kernel_t copy_res_layer_kernel_;
    compute::kernel_t copy_res_iter_kernel_;

    compute::kernel_t ws_set_kernel_;
    compute::kernel_t elemwise_fwd_kernel_;
    compute::kernel_t elemwise_bwd_kernel_;
    compute::kernel_t gates_reduction_kernel_;

    /* GEMM primitives */
    primitive_t *gemm_layer_fwd_ = nullptr;
    primitive_t *gemm_iter_fwd_ = nullptr;
    primitive_t *gemm_layer_bwd_ = nullptr;
    primitive_t *gemm_iter_bwd_ = nullptr;
    primitive_t *gemm_diff_wei_layer_ = nullptr;
    primitive_t *gemm_diff_wei_iter_ = nullptr;

    std::unique_ptr<memory_storage_t> scales_buf_;
    std::unique_ptr<memory_storage_t> tm_scales_buf_;

    cl_ulong ws_gates_offset_;
    cl_ulong ws_states_offset_;
    cl_ulong ws_c_states_offset_;
    cl_ulong ws_diff_states_offset_;
    cl_ulong ws_grid_comp_offset_;
    cl_ulong ws_cell_comp_offset_;
    cl_ulong ws_bias_offset_;
    cl_ulong scratch_gates_offset_;

    size_t *offset_wei_input_;
    size_t *offset_wei_state_;

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
using ref_rnn_fwd_t = _ref_rnn_common_t<prop_kind::forward>;
using ref_rnn_bwd_t = _ref_rnn_common_t<prop_kind::backward>;
} // namespace ocl
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
