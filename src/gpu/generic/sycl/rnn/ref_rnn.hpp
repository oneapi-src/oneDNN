/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_RNN_REF_RNN_HPP
#define GPU_INTEL_OCL_RNN_REF_RNN_HPP

#include <stdio.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/utils.hpp"
#include "gpu/generic/sycl/rnn/rnn_utils.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/gpu_rnn_pd.hpp"

#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"

#include "gpu/generic/sycl/sycl_gpu_kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

enum gemm_kind_t {
    gemm_iter_fwd,
    gemm_layer_fwd,
    gemm_iter_bwd,
    gemm_layer_bwd
};

template <prop_kind_t aprop>
struct _ref_rnn_common_t : public primitive_t {
    using primitive_t::primitive_t;

    using class_name = _ref_rnn_common_t<aprop>;

    typedef cell_execution_sig((class_name::*cell_execution_f));
    typedef grid_execution_sig((class_name::*grid_execution_f));
    typedef gemm_sig((class_name::*gemm_t));

    using base_pd_t =
            typename utils::conditional<false || aprop == prop_kind::forward,
                    gpu_rnn_fwd_pd_t, gpu_rnn_bwd_pd_t>::type;

    struct pd_t : public base_pd_t {

        using base_pd_t::base_pd_t;

        pd_t(const pd_t &other) = default;

        DECLARE_COMMON_PD_T("ref:any", class_name);

        status_t init(impl::engine_t *engine);

        status_t set_default_params();

        rnn_offsets_t off;
        rnn_utils::conf_t rnn_conf = {};
        data_type_t acc_data_t = data_type::undef;
        data_type_t src_type = data_type::undef;
        data_type_t weights_type = data_type::undef;

        std::shared_ptr<primitive_desc_t> vanilla_cell_act_pd_;
        std::shared_ptr<primitive_desc_t> gemm_iter_fwd_pd_;
        std::shared_ptr<primitive_desc_t> gemm_layer_fwd_pd_;
        std::shared_ptr<primitive_desc_t> gemm_iter_bwd_pd_;
        std::shared_ptr<primitive_desc_t> gemm_layer_bwd_pd_;

        sycl_rnn_copy_init_layer_conf_t copy_init_layer_conf_;
        sycl_rnn_copy_init_iter_conf_t copy_init_iter_conf_;
        sycl_rnn_copy_res_layer_conf_t copy_res_layer_conf_;
        sycl_rnn_copy_res_iter_conf_t copy_res_iter_conf_;
        sycl_rnn_bias_conf_t sycl_rnn_bias_conf_t_;

    private:
        void init_scratchpad(dim_t workspace_size) {
            using namespace memory_tracking::names;
            auto scratchpad = this->scratchpad_registry().registrar();
            scratchpad.book(key_rnn_space, workspace_size, 1);
            rnn_utils::scratch_t::book(scratchpad, rnn_conf,
                    {gemm_iter_fwd_pd_.get(), gemm_layer_fwd_pd_.get()});
        }
    };

    status_t init(impl::engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_(ctx);
    }

private:
    status_t execute_(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    // set the class names
    grid_execution_sig(linear_execution);

    cell_execution_sig(cell_execution);

    gemm_sig(gemm_primitive);

    float (*activation_func)(float dd, float s, float alpha, float cliping)
            = nullptr;
    status_t copy_init_layer(const exec_ctx_t &ctx, dim_t n_iter, dim_t batch,
            dim_t slc, dim_t dhc, dim_t n_layer, dim_t n_dir, dim_t n_states,
            dim_t states_ws_ld, const memory_storage_t &ws_states,
            const memory_storage_t &input) const;
    status_t copy_init_iter(const exec_ctx_t &ctx, dim_t n_layer, dim_t n_dir,
            dim_t batch, dim_t sic, dim_t dhc, dim_t n_iter, dim_t n_states,
            dim_t states_ws_ld, const rnn_utils::workspace_t &ws,
            const memory_storage_t &firstit_states) const;
    status_t copy_res_layer(const exec_ctx_t &ctx, dim_t n_iter, dim_t batch,
            dim_t slc, dim_t dhc, dim_t n_layer, dim_t n_dir, dim_t n_states,
            dim_t states_ws_ld, const memory_storage_t &dst_last_layer,
            const rnn_utils::workspace_t &ws) const;
    status_t copy_res_iter(const exec_ctx_t &ctx, dim_t n_layer, dim_t n_dir,
            dim_t batch, dim_t sic, dim_t dhc, dim_t n_iter, dim_t n_states,
            dim_t states_ws_ld, const memory_storage_t &dst_last_iter,
            const rnn_utils::workspace_t &ws) const;
    status_t rnn_bias(const exec_ctx_t &ctx, dim_t batch, dim_t dhc, dim_t iter,
            dim_t lay, dim_t dir, const rnn_utils::workspace_t &ws,
            const rnn_utils::scratch_t &scratch,
            const rnn_utils ::user_data_t &user_data) const;

    // ptrs to GEMM primitives
    std::shared_ptr<impl::primitive_t> gemm_layer_fwd_;
    std::shared_ptr<impl::primitive_t> gemm_iter_fwd_;

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

    kernel_t copy_init_layer_kernel_;
    kernel_t copy_init_iter_kernel_;
    kernel_t copy_res_layer_kernel_;
    kernel_t copy_res_iter_kernel_;
    kernel_t bias_kernel_;

    enum { SCALES_ = 0, TM_SCALES_ = 1 };
};

using ref_rnn_fwd_t = _ref_rnn_common_t<prop_kind::forward>;

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
