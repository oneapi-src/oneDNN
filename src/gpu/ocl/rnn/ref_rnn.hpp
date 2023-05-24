/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef GPU_OCL_RNN_REF_RNN_HPP
#define GPU_OCL_RNN_REF_RNN_HPP

#include <assert.h>
#include <stdio.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/gpu_rnn_pd.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/ocl/rnn/rnn_utils.hpp"
#include "gpu/primitive_conf.hpp"

// TODO just to debug
#define WS_NAN_FILLING 0

#define DEBUGPRINT 0

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

enum gemm_kind_t {
    gemm_iter_fwd,
    gemm_iter_fwd_2,
    gemm_layer_fwd,
    gemm_iter_bwd,
    gemm_iter_bwd_2,
    gemm_layer_bwd,
    gemm_diff_wei_iter,
    gemm_diff_wei_iter_2,
    gemm_diff_wei_layer
};

struct workspace_t {
    using mst = memory_storage_t;
    workspace_t(
            const mst &ws, const rnn_conf_t &rnn, const rnn_utils::conf_t &conf)
        : ws_(ws)
        , gates_(conf.ws_gates_size > 0 ? ws.get_sub_storage(
                         rnn.ws_gates_offset, conf.ws_gates_size)
                                        : nullptr)
        , states_(conf.ws_states_size > 0 ? ws.get_sub_storage(
                          rnn.ws_states_offset, conf.ws_states_size)
                                          : nullptr)
        , c_states_(conf.ws_c_states_size > 0 ? ws.get_sub_storage(
                            rnn.ws_c_state_offset, conf.ws_c_states_size)
                                              : nullptr)
        , bias_(conf.ws_bias_size > 0 ? ws.get_sub_storage(
                        rnn.ws_bias_offset, conf.ws_bias_size)
                                      : nullptr)
        , grid_comp_(conf.ws_grid_comp_size > 0 ? ws.get_sub_storage(
                             rnn.ws_grid_comp_offset, conf.ws_grid_comp_size)
                                                : nullptr) {}

    const mst &ws() const { return ws_; }
    const mst &gates() const { return gates_ ? *gates_ : mst::empty_storage(); }
    const mst &states() const {
        return states_ ? *states_ : mst::empty_storage();
    }
    const mst &c_states() const {
        return c_states_ ? *c_states_ : mst::empty_storage();
    }
    const mst &bias() const { return bias_ ? *bias_ : mst::empty_storage(); }
    const mst &grid_comp() const {
        return grid_comp_ ? *grid_comp_ : mst::empty_storage();
    }

private:
    const mst &ws_;
    std::unique_ptr<mst> gates_;
    std::unique_ptr<mst> states_;
    std::unique_ptr<mst> c_states_;
    std::unique_ptr<mst> bias_;
    std::unique_ptr<mst> grid_comp_;
};

template <prop_kind_t aprop>
struct _ref_rnn_common_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;

    using class_name = _ref_rnn_common_t<aprop>;

    typedef elemwise_sig((class_name::*elemwise_f));
    typedef elemwise_sig_gru((class_name::*elemwise_gru_f));
    typedef elemwise_sig_gru_lbr((class_name::*elemwise_gru_lbr_f));
    typedef cell_execution_sig((class_name::*cell_execution_f));
    typedef grid_execution_sig((class_name::*grid_execution_f));
    typedef gemm_sig((class_name::*gemm_t));
    typedef weights_assign_sig((class_name::*weights_assign_t));

    using base_pd_t =
            typename utils::conditional<false || aprop == prop_kind::forward,
                    gpu_rnn_fwd_pd_t, gpu_rnn_bwd_pd_t>::type;
    enum {
        key_gemm_iter_fwd = memory_tracking::names::key_nested_multiple,
        key_gemm_iter_fwd_2,
        key_gemm_layer_fwd,
        key_gemm_iter_bwd,
        key_gemm_iter_bwd_2,
        key_gemm_layer_bwd,
        key_gemm_diff_wei_layer,
        key_gemm_diff_wei_iter,
        key_gemm_diff_wei_iter_2,
    };

    struct pd_t : public base_pd_t {

        using base_pd_t::base_pd_t;

        pd_t(const pd_t &other) = default;

        DECLARE_COMMON_PD_T("ref:any", class_name);

        status_t init(engine_t *engine);

        status_t set_default_params();

        rnn_conf_t conf;
        rnn_offsets_t off;
        rnn_utils::conf_t rnn_conf;
        data_type_t acc_data_t = data_type::undef;
        data_type_t src_type = data_type::undef;
        data_type_t weights_type = data_type::undef;
        bool is_xe_hpc = false;
        int subgroup_size = 0;
        int max_eus_per_wg = 0;
        bool use_subgroup_reduction = false;

        std::shared_ptr<primitive_desc_t> gemm_iter_fwd_pd_;
        std::shared_ptr<primitive_desc_t> gemm_iter_fwd_2_pd_;
        std::shared_ptr<primitive_desc_t> gemm_layer_fwd_pd_;
        std::shared_ptr<primitive_desc_t> gemm_iter_bwd_pd_;
        std::shared_ptr<primitive_desc_t> gemm_iter_bwd_2_pd_;
        std::shared_ptr<primitive_desc_t> gemm_layer_bwd_pd_;
        std::shared_ptr<primitive_desc_t> gemm_diff_wei_layer_pd_;
        std::shared_ptr<primitive_desc_t> gemm_diff_wei_iter_pd_;
        std::shared_ptr<primitive_desc_t> gemm_diff_wei_iter_2_pd_;

    private:
        void init_scratchpad(size_t scratchpad_sz) {
            using namespace memory_tracking::names;
            auto scratchpad = this->scratchpad_registry().registrar();
            scratchpad.book(key_rnn_space, scratchpad_sz, 1,
                    OCL_BUFFER_ALIGNMENT, 4096);
            scratchpad.book(key_rnn_gates, rnn_conf.scratch_gates_size, 1,
                    OCL_BUFFER_ALIGNMENT, 4096);
            scratchpad.book(key_rnn_cell, rnn_conf.scratch_cell_size, 1,
                    OCL_BUFFER_ALIGNMENT, 4096);
            scratchpad.book(key_rnn_diff_states,
                    rnn_conf.scratch_diff_states_size, 1, OCL_BUFFER_ALIGNMENT,
                    4096);
            scratchpad.book(key_rnn_diff_ht, rnn_conf.scratch_dhG1_size, 1,
                    OCL_BUFFER_ALIGNMENT, 4096);
            // book scratchpad for nested primitives
            switch (aprop) {
                case prop_kind::forward:
                    scratchpad.book(key_gemm_iter_fwd,
                            gemm_iter_fwd_pd_->scratchpad_registry());
                    scratchpad.book(key_gemm_layer_fwd,
                            gemm_layer_fwd_pd_->scratchpad_registry());
                    if (conf.is_vanilla_gru)
                        scratchpad.book(key_gemm_iter_fwd_2,
                                gemm_iter_fwd_2_pd_->scratchpad_registry());
                    break;
                case prop_kind::backward:
                    scratchpad.book(key_gemm_iter_bwd,
                            gemm_iter_bwd_pd_->scratchpad_registry());
                    scratchpad.book(key_gemm_layer_bwd,
                            gemm_layer_bwd_pd_->scratchpad_registry());
                    scratchpad.book(key_gemm_diff_wei_layer,
                            gemm_diff_wei_layer_pd_->scratchpad_registry());
                    scratchpad.book(key_gemm_diff_wei_iter,
                            gemm_diff_wei_iter_pd_->scratchpad_registry());
                    if (conf.is_vanilla_gru) {
                        scratchpad.book(key_gemm_iter_bwd_2,
                                gemm_iter_bwd_2_pd_->scratchpad_registry());
                        scratchpad.book(key_gemm_diff_wei_iter_2,
                                gemm_diff_wei_iter_2_pd_
                                        ->scratchpad_registry());
                    }
                    break;
                default: assert(!"unknown prop_kind");
            }
        }
    }; // struct pd_t : public base_pd_t

    status_t init(engine_t *engine) override;

    ~_ref_rnn_common_t() {
        free(wei_layer_offset_ptr);
        free(wei_iter_offset_ptr);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_(ctx);
    }

protected:
    status_t init_res_storage(
            engine_t *engine, gpu_resource_t *r) const override;

private:
    status_t execute_(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::nd_range_t get_nd_range(std::vector<int> gws) const {
        // Try to schedule one local thread per eu
        int subgroup_size = pd()->subgroup_size;
        int lws_max = pd()->max_eus_per_wg * subgroup_size;
        std::vector<int> lws;
        lws.reserve(gws.size());
        for (int i = 0; i < (int)gws.size(); i++) {
            int l_dim = 2 * gws[i] < lws_max ? utils::rnd_up_pow2(gws[i])
                                             : lws_max;
            if (i == 0 && l_dim < subgroup_size) l_dim = subgroup_size;
            lws.emplace_back(l_dim);
            gws[i] = utils::rnd_up(gws[i], l_dim);
            lws_max = lws_max / l_dim;
        }

        return compute::nd_range_t(gws, lws);
    }

    // set the class names
    grid_execution_sig(linear_execution);

    cell_execution_sig(cell_execution);
    cell_execution_sig(cell_execution_gru);
    cell_execution_sig(cell_execution_gru_lbr);

    elemwise_sig(rnn_elemwise);
    elemwise_sig(lstm_elemwise);
    elemwise_sig(lstm_elemwise_u8s8);
    elemwise_sig_gru(gru_elemwise);
    elemwise_sig_gru_lbr(gru_lbr_elemwise);

    gemm_sig(gemm_primitive);

    weights_assign_sig(assign_weights);

    float (*activation_func)(float dd, float s, float alpha, float cliping);
    status_t bias_prepare(const exec_ctx_t &ctx,
            compute::compute_stream_t *compute_stream, int n_layer, int n_dir,
            int n_bias, int n_gates, int dhc, const memory_storage_t &ws_bias,
            const memory_storage_t &scales, const memory_storage_t &wei_layer,
            const memory_storage_t &wei_iter,
            const memory_storage_t &bias) const;
    status_t copy_init_layer(const exec_ctx_t &ctx,
            compute::compute_stream_t *compute_stream, bool lr, bool rl,
            int n_iter, int batch, int slc, const memory_storage_t &ws_states,
            const memory_storage_t &scratch_diff_states,
            const memory_storage_t &input,
            const memory_storage_t &diff_dst_layer) const;
    status_t copy_init_iter(const exec_ctx_t &ctx,
            compute::compute_stream_t *compute_stream, int n_layer, int n_dir,
            int batch, int sic, int dhc, const workspace_t &ws,
            const memory_storage_t &scratch_diff_states,
            const memory_storage_t &firstit_states,
            const memory_storage_t &firstit_c_states,
            const memory_storage_t &diff_dst_iter,
            const memory_storage_t &diff_dst_iter_c, const float shift,
            const float scale, const bool quantize) const;
    status_t copy_res_layer(const exec_ctx_t &ctx,
            compute::compute_stream_t *compute_stream, bool lr, bool rl,
            int n_iter, int batch, int slc, int dlc,
            const memory_storage_t &scratch_diff_states,
            const memory_storage_t &dst_last_layer,
            const memory_storage_t &diff_src_layer,
            const memory_storage_t &ws_states, const float shift,
            const float scale, const bool dequantize) const;
    status_t copy_res_iter(const exec_ctx_t &ctx,
            compute::compute_stream_t *compute_stream, int n_layer, int n_dir,
            int batch, int sic, int dhc,
            const memory_storage_t &scratch_diff_states,
            const memory_storage_t &dst_last_iter,
            const memory_storage_t &dst_last_iter_c,
            const memory_storage_t &diff_src_iter,
            const memory_storage_t &diff_src_iter_c, const workspace_t &ws,
            const float shift, const float scale, const bool dequantize) const;
    status_t gates_reduction(const exec_ctx_t &ctx, int dir, int lay, int iter,
            int n_gates, int dhc, int batch, const memory_storage_t &gates,
            const memory_storage_t &cell,
            const memory_storage_t &diff_bias) const;
    status_t ws_set(const exec_ctx_t &ctx,
            compute::compute_stream_t *compute_stream,
            const memory_storage_t &workspace, const cl_ulong ws_offset,
            const int ws_part, const float val, const size_t size) const;
#if DEBUGPRINT
    status_t ws_print(const exec_ctx_t &ctx, compute::compute_stream_t *s,
            const workspace_t &workspace) const;
    compute::kernel_t ws_print_kernel_;
#endif

    compute::kernel_t bias_prepare_kernel_;
    compute::kernel_t copy_init_layer_kernel_;
    compute::kernel_t copy_init_iter_kernel_;
    compute::kernel_t copy_res_layer_kernel_;
    compute::kernel_t copy_res_iter_kernel_;

    compute::kernel_t ws_set_kernel_;
    compute::kernel_t elemwise_fwd_kernel_;
    compute::kernel_t elemwise_bwd_kernel_;
    compute::kernel_t gates_reduction_kernel_;

    // ptrs to GEMM primitives
    std::shared_ptr<primitive_t> gemm_layer_fwd_;
    std::shared_ptr<primitive_t> gemm_iter_fwd_;
    std::shared_ptr<primitive_t> gemm_iter_fwd_2_;
    std::shared_ptr<primitive_t> gemm_layer_bwd_;
    std::shared_ptr<primitive_t> gemm_iter_bwd_;
    std::shared_ptr<primitive_t> gemm_iter_bwd_2_;
    std::shared_ptr<primitive_t> gemm_diff_wei_layer_;
    std::shared_ptr<primitive_t> gemm_diff_wei_iter_;
    std::shared_ptr<primitive_t> gemm_diff_wei_iter_2_;

    // offset variables set in workspace and used in offset calculations for
    // grid & cell execution and fwd & bwd kernel macros
    cl_ulong ws_gates_offset_;
    cl_ulong ws_states_offset_;
    cl_ulong ws_c_states_offset_;
    cl_ulong ws_grid_comp_offset_;
    cl_ulong ws_bias_offset_;
    cl_ulong scratch_dhG1_offset_;
    cl_ulong scratch_cell_offset_;
    cl_ulong scratch_gates_offset_;
    cl_ulong scratch_diff_states_offset_;

    // ptrs for storing weight offsets which are pre-calculated in
    // in grid execution as weights_*_assing_func
    size_t *wei_layer_offset_ptr;
    size_t *wei_iter_offset_ptr;

    grid_execution_f grid_computation;
    cell_execution_f cell_func;

    weights_assign_t weights_layer_assign_func;
    weights_assign_t weights_iter_assign_func;

    gemm_t gemm_iter_func;
    gemm_t gemm_layer_func;
    elemwise_f elemwise_common;
    elemwise_gru_f elemwise_gru;
    elemwise_gru_lbr_f elemwise_gru_lbr;

    enum { SCALES_ = 0, TM_SCALES_ = 1 };
};
using ref_rnn_fwd_t = _ref_rnn_common_t<prop_kind::forward>;
using ref_rnn_bwd_t = _ref_rnn_common_t<prop_kind::backward>;
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
