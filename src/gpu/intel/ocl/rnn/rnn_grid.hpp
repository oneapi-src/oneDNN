/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_INTEL_OCL_RNN_RNN_GRID_HPP
#define GPU_INTEL_OCL_RNN_RNN_GRID_HPP

#include <stdio.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/gpu_rnn_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/ocl/rnn/rnn_utils.hpp"
#include "gpu/intel/ocl/utils.hpp"
#include "gpu/intel/primitive_conf.hpp"

// TODO just to debug
#define WS_NAN_FILLING 0

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

enum gemm_kind_t {
    gemm_iter_fwd,
    gemm_iter_fwd_2,
    gemm_layer_fwd,
    gemm_layer_fwd_src,
    gemm_iter_bwd,
    gemm_iter_bwd_2,
    gemm_layer_bwd,
    gemm_layer_bwd_src,
    gemm_diff_wei_iter,
    gemm_diff_wei_iter_2,
    gemm_diff_wei_layer,
    gemm_diff_wei_layer_src
};

template <prop_kind_t aprop>
struct simple_rnn_common_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;

    using class_name = simple_rnn_common_t<aprop>;

    using elemwise_f = elemwise_sig((class_name::*));
    using elemwise_gru_f = elemwise_sig_gru((class_name::*));
    using elemwise_gru_lbr_f = elemwise_sig_gru_lbr((class_name::*));
    using cell_execution_f = cell_execution_sig((class_name::*));
    using grid_execution_f = grid_execution_sig((class_name::*));
    using gemm_t = gemm_sig((class_name::*));

    using base_pd_t =
            typename utils::conditional<false || aprop == prop_kind::forward,
                    gpu_rnn_fwd_pd_t, gpu_rnn_bwd_pd_t>::type;

    struct pd_t : public base_pd_t {

        using base_pd_t::base_pd_t;

        pd_t(const pd_t &other) = default;

        DECLARE_COMMON_PD_T("ocl:simple:any", class_name);

        status_t init(impl::engine_t *engine);

        status_t set_default_params();

        rnn_utils::ocl_conf_t ocl_conf = {};
        rnn_offsets_t off = {};
        rnn_utils::conf_t rnn_conf = {};
        data_type_t acc_data_t = data_type::undef;
        data_type_t src_type = data_type::undef;
        data_type_t weights_type = data_type::undef;
        bool is_xe_hpc = false;
        int max_eus_per_wg = 0;

        std::shared_ptr<primitive_desc_t> gemm_iter_fwd_pd_;
        std::shared_ptr<primitive_desc_t> gemm_iter_fwd_2_pd_;
        std::shared_ptr<primitive_desc_t> gemm_layer_fwd_pd_;
        std::shared_ptr<primitive_desc_t> gemm_layer_fwd_src_pd_;
        std::shared_ptr<primitive_desc_t> gemm_iter_bwd_pd_;
        std::shared_ptr<primitive_desc_t> gemm_iter_bwd_2_pd_;
        std::shared_ptr<primitive_desc_t> gemm_layer_bwd_pd_;
        std::shared_ptr<primitive_desc_t> gemm_layer_bwd_src_pd_;
        std::shared_ptr<primitive_desc_t> gemm_diff_wei_layer_pd_;
        std::shared_ptr<primitive_desc_t> gemm_diff_wei_layer_src_pd_;
        std::shared_ptr<primitive_desc_t> gemm_diff_wei_iter_pd_;
        std::shared_ptr<primitive_desc_t> gemm_diff_wei_iter_2_pd_;

    private:
        void init_scratchpad(dim_t workspace_size) {
            using namespace memory_tracking::names;
            auto scratchpad = this->scratchpad_registry().registrar();
            scratchpad.book(key_rnn_space, workspace_size, 1,
                    OCL_BUFFER_ALIGNMENT, 4096);
            rnn_utils::scratch_t::book(scratchpad, rnn_conf,
                    {
                            gemm_iter_fwd_pd_.get(),
                            gemm_iter_fwd_2_pd_.get(),
                            gemm_layer_fwd_pd_.get(),
                            gemm_layer_fwd_src_pd_.get(),
                            gemm_iter_bwd_pd_.get(),
                            gemm_iter_bwd_2_pd_.get(),
                            gemm_layer_bwd_pd_.get(),
                            gemm_layer_bwd_src_pd_.get(),
                            gemm_diff_wei_layer_pd_.get(),
                            gemm_diff_wei_layer_src_pd_.get(),
                            gemm_diff_wei_iter_pd_.get(),
                            gemm_diff_wei_iter_2_pd_.get(),
                    });
        }
    }; // struct pd_t : public base_pd_t

    status_t init(impl::engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_(ctx);
    }

protected:
    status_t init_res_storage(
            impl::engine_t *engine, gpu_resource_t *r) const override;

private:
    status_t execute_(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::nd_range_t get_nd_range(std::vector<dim_t> gws) const {
        // Try to schedule one local thread per eu
        int subgroup_size = pd()->ocl_conf.subgroup_size;
        int lws_max = pd()->max_eus_per_wg * subgroup_size;
        std::vector<dim_t> lws;
        lws.reserve(gws.size());
        for (size_t i = 0; i < gws.size(); i++) {
            dim_t l_dim = 2 * gws[i] <= lws_max ? utils::rnd_up_pow2(gws[i])
                                                : lws_max;
            if (i == 0 && l_dim < subgroup_size) l_dim = subgroup_size;
            lws.emplace_back(l_dim);
            gws[i] = utils::rnd_up(gws[i], l_dim);
            lws_max = lws_max / l_dim;
        }

        return compute::nd_range_t(gws, {lws});
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

    float (*activation_func)(float dd, float s, float alpha, float cliping)
            = nullptr;
    status_t bias_prepare(const exec_ctx_t &ctx,
            compute::compute_stream_t *compute_stream, dim_t n_layer,
            dim_t n_dir, dim_t n_bias, dim_t n_gates, dim_t dhc,
            const memory_storage_t &ws_bias, const memory_storage_t &scales,
            const memory_storage_t &wei_layer, const memory_storage_t &wei_iter,
            const memory_storage_t &bias) const;
    status_t copy_init_layer(const exec_ctx_t &ctx,
            compute::compute_stream_t *compute_stream, bool lr, bool rl,
            dim_t n_iter, dim_t batch, dim_t slc, dim_t dhc, dim_t n_layer,
            dim_t n_dir, dim_t n_states, dim_t states_ws_ld,
            dim_t scratch_diff_states_ld, const memory_storage_t &ws_states,
            const memory_storage_t *scratch_diff_states,
            const memory_storage_t &input,
            const memory_storage_t &diff_dst_layer) const;
    status_t copy_init_iter(const exec_ctx_t &ctx,
            compute::compute_stream_t *compute_stream, dim_t n_layer,
            dim_t n_dir, dim_t batch, dim_t sic, dim_t dhc, dim_t n_iter,
            dim_t n_states, dim_t states_ws_ld, dim_t scratch_diff_states_ld,
            const rnn_utils::workspace_t &ws,
            const memory_storage_t *scratch_diff_states,
            const memory_storage_t &firstit_states,
            const memory_storage_t &firstit_c_states,
            const memory_storage_t &diff_dst_iter,
            const memory_storage_t &diff_dst_iter_c, const float shift,
            const float scale, const bool quantize) const;
    status_t copy_res_layer(const exec_ctx_t &ctx,
            compute::compute_stream_t *compute_stream, bool lr, bool rl,
            dim_t n_iter, dim_t batch, dim_t slc, dim_t dhc, dim_t n_layer,
            dim_t n_dir, dim_t n_states, dim_t states_ws_ld,
            dim_t scratch_diff_states_ld,
            const memory_storage_t *scratch_diff_states,
            const memory_storage_t &dst_last_layer,
            const memory_storage_t &diff_src_layer,
            const memory_storage_t &ws_states, const float shift,
            const float scale, const bool dequantize) const;
    status_t copy_res_iter(const exec_ctx_t &ctx,
            compute::compute_stream_t *compute_stream, dim_t n_layer,
            dim_t n_dir, dim_t batch, dim_t sic, dim_t dhc, dim_t n_iter,
            dim_t n_states, dim_t states_ws_ld, dim_t scratch_diff_states_ld,
            const memory_storage_t *scratch_diff_states,
            const memory_storage_t &dst_last_iter,
            const memory_storage_t &dst_last_iter_c,
            const memory_storage_t &diff_src_iter,
            const memory_storage_t &diff_src_iter_c,
            const rnn_utils::workspace_t &ws, const float shift,
            const float scale, const bool dequantize) const;

    std::vector<compute::kernel_t> kernels_;

    // ptrs to GEMM primitives
    std::shared_ptr<impl::primitive_t> gemm_layer_fwd_;
    std::shared_ptr<impl::primitive_t> gemm_layer_fwd_src_;
    std::shared_ptr<impl::primitive_t> gemm_iter_fwd_;
    std::shared_ptr<impl::primitive_t> gemm_iter_fwd_2_;
    std::shared_ptr<impl::primitive_t> gemm_layer_bwd_;
    std::shared_ptr<impl::primitive_t> gemm_layer_bwd_src_;
    std::shared_ptr<impl::primitive_t> gemm_iter_bwd_;
    std::shared_ptr<impl::primitive_t> gemm_iter_bwd_2_;
    std::shared_ptr<impl::primitive_t> gemm_diff_wei_layer_;
    std::shared_ptr<impl::primitive_t> gemm_diff_wei_layer_src_;
    std::shared_ptr<impl::primitive_t> gemm_diff_wei_iter_;
    std::shared_ptr<impl::primitive_t> gemm_diff_wei_iter_2_;

    // offset variables set in workspace and used in offset calculations for
    // grid & cell execution and fwd & bwd kernel macros
    dim_t ws_gates_offset_ = 0;
    dim_t ws_states_offset_ = 0;
    dim_t ws_c_states_offset_ = 0;
    dim_t ws_grid_comp_offset_ = 0;
    dim_t ws_bias_offset_ = 0;

    // ptrs for storing weight offsets which are pre-calculated in
    // in grid execution as weights_*_assing_func
    std::vector<dim_t> wei_layer_offsets;
    std::vector<dim_t> wei_iter_offsets;

    grid_execution_f grid_computation = nullptr;
    cell_execution_f cell_func = nullptr;

    elemwise_f elemwise_common = nullptr;
    elemwise_gru_f elemwise_gru = nullptr;
    elemwise_gru_lbr_f elemwise_gru_lbr = nullptr;

    enum { SCALES_ = 0, TM_SCALES_ = 1 };
};
using simple_rnn_fwd_t = simple_rnn_common_t<prop_kind::forward>;
using simple_rnn_bwd_t = simple_rnn_common_t<prop_kind::backward>;
} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
