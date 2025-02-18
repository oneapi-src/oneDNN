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

// General architecture
//
// for diff states, we have n_states + 1 as we have n_states diff
// to propagate to the previous iteration and 1 states to propagate
// to the previous layer
// index 0 is dh for cell(t-1, l) to consume
// index 1 is dc for cell(t-1, l) to consume
// index 2 is dh for cell(t, l-1) to consume
// this indexing enables to have the same indexing for states in elemwise
// function
// only the cell execution function should be impacted

#include "gpu/generic/sycl/rnn/ref_rnn.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc.hpp"

#include "common/matmul_pd.hpp"
#include "common/stream.hpp"
#include "common/type_helpers.hpp"
#include "gpu/generic/sycl/rnn/rnn_kernels.hpp"

#include <memory>

#define DPRINT(fmt, ...) \
    do { \
        if (get_verbose_dev_mode(verbose_t::debuginfo) >= 2) { \
            printf(fmt, __VA_ARGS__); \
            fflush(nullptr); \
        } \
    } while (0)

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::math;
using namespace prop_kind;
using namespace alg_kind;
using namespace rnn_utils;
using namespace dnnl::impl::memory_tracking::names;

status_t _ref_rnn_common_t::pd_t::set_default_params() {
    using namespace format_tag;
    if (src_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_layer_md_, tnc));
    if (dst_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_layer_md_, tnc));

    // Optional parameters
    if ((!types::is_zero_md(&src_iter_md_))
            && (src_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(src_iter_md_, ldnc));
    if ((!types::is_zero_md(&bias_md_))
            && (bias_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(bias_md_, ldgo));
    if ((!types::is_zero_md(&dst_iter_md_))
            && (dst_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(dst_iter_md_, ldnc));

    return status::success;
}

status_t _ref_rnn_common_t::pd_t::init(impl::engine_t *engine) {
    using namespace prop_kind;
    using namespace utils;
    using namespace rnn_utils;
    using namespace format_tag;

    assert(engine->kind() == engine_kind::gpu);

    const alg_kind_t cell_kind = this->desc()->cell_kind;

    data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
    data_type_t weights_iter_dt = this->desc()->weights_iter_desc.data_type;
    data_type_t weights_layer_dt = this->desc()->weights_layer_desc.data_type;
    data_type_t bias_dt = this->desc()->bias_desc.data_type;

    acc_data_t = data_type::f32;

    src_type = src_layer_dt;
    weights_type = weights_layer_dt;

    VDISPATCH_RNN(
            one_of(cell_kind, alg_kind::vanilla_rnn), VERBOSE_BAD_ALGORITHM);
    VDISPATCH_RNN(weights_iter_dt == weights_layer_dt, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_RNN_SC(this->set_default_params(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_RNN(this->with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_RNN(this->desc()->prop_kind == forward_inference
                    || this->desc()->prop_kind == forward_training,
            VERBOSE_UNSUPPORTED_DT_CFG);

    init_rnn_conf(rnn_conf, this, acc_data_t);

    // Check that only supported attr have been passed.
    primitive_attr_t::skip_mask_t attr_mask
            = primitive_attr_t::skip_mask_t::rnn_tparams;
    if (weights_layer_dt == data_type::s8) {
        attr_mask = attr_mask | primitive_attr_t::skip_mask_t::rnn_data_qparams
                | primitive_attr_t::skip_mask_t::rnn_weights_qparams
                | primitive_attr_t::skip_mask_t::fpmath_mode;
    }
    VDISPATCH_RNN(this->attr()->has_default_values(attr_mask),
            VERBOSE_UNSUPPORTED_ATTR);

    // Set weights descriptors to desired format
    VDISPATCH_RNN_SC(set_weights_desc(this->weights_layer_md_, rnn_conf),
            "unsupported weights layer memory descriptor");
    VDISPATCH_RNN_SC(set_weights_desc(this->weights_iter_md_, rnn_conf),
            "unsupported weights iter memory descriptor");

    // Currently only run L2R
    VDISPATCH_RNN(this->direction() == dnnl_unidirectional_left2right,
            VERBOSE_BAD_ALGORITHM);
    // Check dimensions consistency
    VDISPATCH_RNN((this->SIC() == this->DHC() || (this->T() == 1)),
            VERBOSE_INCONSISTENT_DIM, "SIC", (int)this->SIC(), "DHC",
            (int)this->DHC());

    set_rnn_conf(rnn_conf, *this->desc());

    dim_t workspace_size = get_workspace_size(rnn_conf);

    // initialize the workspace_pd if needed
    if (rnn_conf.use_workspace) {
        dims_t ws_dims = {workspace_size};
        VDISPATCH_RNN_SC(memory_desc_init_by_tag(
                                 this->ws_md_, 1, ws_dims, data_type::u8, x),
                "memory_desc_init_by_tag()");
    }

    memory_desc_t state_md;
    dims_t state_dims = {rnn_conf.n_layer, rnn_conf.n_dir, rnn_conf.n_iter + 1,
            rnn_conf.mb, rnn_conf.states_ws_ld};

    CHECK(memory_desc_init_by_tag(state_md, 5, state_dims,
            rnn_conf.src_data_type, format_tag::abcde));

    copy_init_layer_conf_ = sycl_rnn_copy_conf_t {
            xpu::sycl::md_t(this->src_md(0)), xpu::sycl::md_t(&state_md),
            rnn_conf.slc, rnn_conf.n_dir, rnn_conf.n_layer, rnn_conf.n_iter,
            rnn_conf.mb, rnn_conf.states_ws_ld, true, true};

    xpu::sycl::md_t src_iter_md = this->src_md(1)->data_type == data_type::undef
            ? xpu::sycl::md_t()
            : xpu::sycl::md_t(this->src_md(1));

    copy_init_iter_conf_ = sycl_rnn_copy_conf_t {src_iter_md,
            xpu::sycl::md_t(&state_md), rnn_conf.sic, rnn_conf.n_dir,
            rnn_conf.n_layer, rnn_conf.n_iter, rnn_conf.mb,
            rnn_conf.states_ws_ld, false, true};

    copy_res_layer_conf_ = sycl_rnn_copy_conf_t {xpu::sycl::md_t(&state_md),
            xpu::sycl::md_t(this->dst_md(0)), rnn_conf.dhc, rnn_conf.n_dir,
            rnn_conf.n_layer, rnn_conf.n_iter, rnn_conf.mb,
            rnn_conf.states_ws_ld, true, false};

    xpu::sycl::md_t dst_iter_md = this->dst_md(1)->data_type == data_type::undef
            ? xpu::sycl::md_t()
            : xpu::sycl::md_t(this->dst_md(1));

    copy_res_iter_conf_ = sycl_rnn_copy_conf_t {xpu::sycl::md_t(&state_md),
            dst_iter_md, rnn_conf.dhc, rnn_conf.n_dir, rnn_conf.n_layer,
            rnn_conf.n_iter, rnn_conf.mb, rnn_conf.states_ws_ld, false, false};

    sycl_rnn_bias_conf_t_ = sycl_rnn_bias_conf_t();
    sycl_rnn_bias_conf_t_.dst_md = xpu::sycl::md_t(this->dst_md(0));
    sycl_rnn_bias_conf_t_.bias_type = bias_dt;
    sycl_rnn_bias_conf_t_.batch = rnn_conf.mb;
    sycl_rnn_bias_conf_t_.dhc = rnn_conf.dhc;
    sycl_rnn_bias_conf_t_.gates_ws_ld = rnn_conf.gates_ws_ld;
    sycl_rnn_bias_conf_t_.states_ws_ld = rnn_conf.states_ws_ld;
    sycl_rnn_bias_conf_t_.activation_kind = this->activation_kind();
    sycl_rnn_bias_conf_t_.alpha = this->desc()->alpha;

    auto fpmath_mode = this->attr()->fpmath_.mode_;

    // The inputs of create_gemm_pd describe a gemm in column major.
    // Below, we have to transpose the a and b descriptor to describe
    // the GEMM as a row major problem.
    auto create_gemm_pd =
            [&](std::shared_ptr<primitive_desc_t> &gemm_pd, dim_t m, dim_t n,
                    dim_t k, strides_t<2> a_strides, strides_t<2> b_strides,
                    strides_t<2> c_strides, data_type_t a_dt, data_type_t b_dt,
                    data_type_t c_dt, float beta) -> status_t {
        memory_desc_t a_md, b_md, c_md, bias_md;

        dims_t a_dims = {n, k}, b_dims = {k, m}, c_dims = {n, m};

        dims_t b_strides_md = {b_strides[0], b_strides[1]};
        CHECK(memory_desc_init_by_strides(
                b_md, 2, b_dims, rnn_conf.wei_layer_type, b_strides_md));
        dims_t a_strides_md = {a_strides[0], a_strides[1]};
        CHECK(memory_desc_init_by_strides(
                a_md, 2, a_dims, rnn_conf.src_data_type, a_strides_md));
        dims_t c_strides_md = {c_strides[0], c_strides[1]};
        CHECK(memory_desc_init_by_strides(
                c_md, 2, c_dims, rnn_conf.dst_data_type, c_strides_md));

        primitive_attr_t attr;
        if (beta != 0) { CHECK(attr.post_ops_.append_sum(beta)); }
        CHECK(attr.set_fpmath_mode(fpmath_mode));
        attr.deterministic_ = this->attr()->deterministic_;

        matmul_desc_t matmul_desc;
        dnnl::impl::matmul_desc_init(
                &matmul_desc, &a_md, &b_md, &bias_md, &c_md);

        primitive_desc_iterator_t it(engine,
                reinterpret_cast<op_desc_t *>(&matmul_desc), &attr, nullptr);

        while (++it != it.end()) {
            if (*it) {
                gemm_pd = *it;
                return status::success;
                break;
            }
        }
        return status::unimplemented;
    };

    float gemm_iter_fwd_beta = this->is_lbr() ? 0.0f : 1.0f;

    // Setup gemm PDs

    dim_t batch = rnn_conf.mb;
    dim_t n_gates = rnn_conf.n_gates;
    dim_t slc = rnn_conf.slc;
    dim_t sic = rnn_conf.sic;
    dim_t dhc = rnn_conf.dhc;

    strides_t<5> wei_layer_strides = get_outer_strides(this->weights_md(0));
    strides_t<5> wei_iter_strides = get_outer_strides(this->weights_md(1));

    VDISPATCH_RNN_SC(create_gemm_pd(gemm_layer_fwd_pd_, n_gates * dhc, batch,
                             slc, {rnn_conf.states_ws_ld, 1},
                             {wei_layer_strides[2], wei_layer_strides[4]},
                             {rnn_conf.scratch_gates_ld, 1}, weights_type,
                             src_type, rnn_conf.acc_data_type, 0.0),
            "create_gemm_pd(gemm_layer_fwd_pd_)");

    VDISPATCH_RNN_SC(create_gemm_pd(gemm_iter_fwd_pd_, n_gates * dhc, batch,
                             sic, {rnn_conf.states_ws_ld, 1},
                             {wei_iter_strides[2], wei_iter_strides[4]},
                             {rnn_conf.gates_ws_ld, 1}, weights_type, src_type,
                             rnn_conf.acc_data_type, gemm_iter_fwd_beta),
            "create_gemm_pd(gemm_iter_fwd_pd_)");

    init_scratchpad(rnn_conf.use_workspace ? 0 : workspace_size);
    return status::success;
}

status_t _ref_rnn_common_t::init(impl::engine_t *engine) {
    using namespace rnn_utils;

    switch (pd()->cell_kind()) {
        case dnnl_vanilla_rnn:
            cell_func = [this](const cell_ctx_t &cell_struct) -> status_t {
                return this->cell_execution(cell_struct);
            };
            break;
        default: break;
    }
    grid_func = [this](const grid_ctx_t &grid_struct) -> status_t {
        return this->linear_execution(grid_struct);
    };

    const conf_t &rnn = pd()->rnn_conf;
    rnn_utils::set_workspace_offsets(rnn, ws_gates_offset_, ws_states_offset_);

    // IMPORTANT SYCL STUFF
    const auto copy_kid = ::sycl::get_kernel_id<ref_rnn_copy_t>();
    this->create_kernel(engine, copy_kid, &copy_kernel_);
    const auto bias_kid = ::sycl::get_kernel_id<ref_rnn_bias>();
    this->create_kernel(engine, bias_kid, &bias_kernel_);

    bool gemm_ok = true;
    auto create_nested_gemm =
            [&](const std::shared_ptr<primitive_desc_t> &prim_desc,
                    std::shared_ptr<impl::primitive_t> &prim) {
                std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t>
                        pair;
                bool gemm_ok = prim_desc->create_primitive_nested(pair, engine)
                        == status::success;
                prim = pair.first;
                return gemm_ok;
            };

    gemm_ok = gemm_ok
            && create_nested_gemm(pd()->gemm_layer_fwd_pd_, gemm_layer_fwd_);
    gemm_ok = gemm_ok
            && create_nested_gemm(pd()->gemm_iter_fwd_pd_, gemm_iter_fwd_);

    if (!gemm_ok) return status::runtime_error;

    return status::success;
} // namespace sycl

status_t _ref_rnn_common_t::gemm_primitive(impl::engine_t *engine,
        const exec_ctx_t &ctx, std::unique_ptr<memory_storage_t> &a,
        std::unique_ptr<memory_storage_t> &b,
        std::unique_ptr<memory_storage_t> &c, gemm_kind_t gemm_kind) const {
    std::unique_ptr<memory_t, memory_deleter_t> arg1, arg2, arg3;
    exec_args_t gemm_args;
    std::shared_ptr<impl::primitive_desc_t> gemm_pd;

    switch (gemm_kind) {
        case gemm_iter_fwd: gemm_pd = pd()->gemm_iter_fwd_pd_; break;
        case gemm_layer_fwd: gemm_pd = pd()->gemm_layer_fwd_pd_; break;
    }

    CHECK(safe_ptr_assign(arg2,
            new memory_t(
                    ctx.stream()->engine(), gemm_pd->src_md(0), a->clone())));
    CHECK(safe_ptr_assign(arg1,
            new memory_t(ctx.stream()->engine(), gemm_pd->weights_md(0),
                    b->clone())));
    CHECK(safe_ptr_assign(arg3,
            new memory_t(
                    ctx.stream()->engine(), gemm_pd->dst_md(0), c->clone())));

    gemm_args[DNNL_ARG_SRC] = memory_arg_t {arg1.get(), true};
    gemm_args[DNNL_ARG_WEIGHTS] = memory_arg_t {arg2.get(), true};
    gemm_args[DNNL_ARG_DST] = memory_arg_t {arg3.get(), false};

    exec_ctx_t gemm_ctx(ctx, std::move(gemm_args));

    std::unique_ptr<nested_scratchpad_t> ns;
    const auto init_gemm_nested_scratchpad
            = [&](const std::shared_ptr<impl::primitive_t> &gemm, int key) {
                  ns = utils::make_unique<nested_scratchpad_t>(ctx, key, gemm);
                  gemm_ctx.set_scratchpad_grantor(ns->grantor());
              };

    switch (gemm_kind) {
        case gemm_iter_fwd:
            init_gemm_nested_scratchpad(
                    gemm_iter_fwd_, rnn_utils::scratch_t::key_gemm_iter_fwd);
            CHECK(gemm_iter_fwd_->execute(gemm_ctx));
            break;
        case gemm_layer_fwd:
            init_gemm_nested_scratchpad(
                    gemm_layer_fwd_, rnn_utils::scratch_t::key_gemm_layer_fwd);
            CHECK(gemm_layer_fwd_->execute(gemm_ctx));
            break;

        default: assert(!"unknown gemm_kind"); return status::runtime_error;
    }

    return status::success;
}

//*************** Grid computations strategy: linear ***************//
status_t _ref_rnn_common_t::linear_execution(const grid_ctx_t &grid_struct) {

    dim_t n_layer = grid_struct.rnn.n_layer;
    dim_t n_dir = grid_struct.rnn.n_dir;
    dim_t n_iter = grid_struct.rnn.n_iter;

    for (dim_t dir = 0; dir < n_dir; dir++) {
        for (dim_t j = 0; j < n_layer; j++) {
            dim_t lay = j;
            for (dim_t i = 0; i < n_iter; i += grid_struct.rnn.iter_loop) {
                dim_t iter = i;
                const cell_ctx_t c_struct
                        = {grid_struct.engine, grid_struct.ctx, dir, lay, iter,
                                grid_struct.user_data, grid_struct.workspace,
                                grid_struct.scratch, grid_struct.rnn};
                CHECK(cell_func(c_struct));
            }
        }
    }
    return status::success;
}
//********* GRID computations strategy: utility functions **********//

status_t _ref_rnn_common_t::copy_init_layer(const exec_ctx_t &ctx, dim_t batch,
        dim_t dhc, dim_t slc, dim_t n_iter, dim_t n_layer, dim_t n_dir,
        dim_t n_states, dim_t states_ws_ld, const rnn_utils::workspace_t &ws,
        const memory_storage_t &input) const {

    auto max_wg_size_per_dim = calc_local_range(ctx);

    parallel_for(ctx, copy_kernel_, [&](::sycl::handler &cgh) {
        auto src_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &input)
                          ->get_in_memory_arg(ctx.stream(), cgh);
        auto dst_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &ws.states())
                          ->get_out_memory_arg(ctx.stream(), cgh);

        ref_rnn_copy_t copy_kernel(
                pd()->copy_init_layer_conf_, src_mem_arg, dst_mem_arg);
        size_t local_batch = max_wg_size_per_dim;
        size_t local_iter = max_wg_size_per_dim;
        size_t local_channel = max_wg_size_per_dim;
        size_t global_batch = calc_global_range(
                static_cast<size_t>(local_batch), static_cast<size_t>(batch));
        size_t global_iter = calc_global_range(
                static_cast<size_t>(local_iter), static_cast<size_t>(n_iter));
        size_t global_channels = calc_global_range(
                static_cast<size_t>(local_channel), static_cast<size_t>(slc));
        cgh.parallel_for(
                ::sycl::nd_range<3>(::sycl::range<3>(global_iter, global_batch,
                                            global_channels),
                        ::sycl::range<3>(
                                local_iter, local_batch, local_channel)),
                copy_kernel);
    });

    return status::success;
}

status_t _ref_rnn_common_t::copy_init_iter(const exec_ctx_t &ctx, dim_t batch,
        dim_t dhc, dim_t sic, dim_t n_iter, dim_t n_layer, dim_t n_dir,
        dim_t n_states, dim_t states_ws_ld, const rnn_utils::workspace_t &ws,
        const memory_storage_t &firstit_states) const {

    auto max_wg_size_per_dim = calc_local_range(ctx);

    parallel_for(ctx, copy_kernel_, [&](::sycl::handler &cgh) {
        auto src_iter_mem_arg = firstit_states
                ? utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &firstit_states)
                          ->get_in_memory_arg(ctx.stream(), cgh)
                : xpu::sycl::memory_storage_base_t::empty_in_memory_arg(
                        ctx.stream(), cgh);
        auto ws_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &ws.states())
                          ->get_out_memory_arg(ctx.stream(), cgh);

        ref_rnn_copy_t copy_kernel(
                pd()->copy_init_iter_conf_, src_iter_mem_arg, ws_mem_arg);
        size_t local_batch = max_wg_size_per_dim;
        size_t local_channel = max_wg_size_per_dim;
        size_t local_lay_dir = max_wg_size_per_dim;
        size_t global_batch
                = calc_global_range(static_cast<size_t>(max_wg_size_per_dim),
                        static_cast<size_t>(batch));
        size_t global_channels = calc_global_range(
                static_cast<size_t>(max_wg_size_per_dim),
                std::max(static_cast<size_t>(sic), static_cast<size_t>(dhc)));
        size_t global_lay_dir
                = calc_global_range(static_cast<size_t>(max_wg_size_per_dim),
                        static_cast<size_t>(n_layer * n_dir));
        cgh.parallel_for(
                ::sycl::nd_range<3>(::sycl::range<3>(global_lay_dir,
                                            global_batch, global_channels),
                        ::sycl::range<3>(
                                local_lay_dir, local_batch, local_channel)),
                copy_kernel);
    });
    return status::success;
}

status_t _ref_rnn_common_t::copy_res_layer(const exec_ctx_t &ctx, dim_t batch,
        dim_t dhc, dim_t slc, dim_t n_iter, dim_t n_layer, dim_t n_dir,
        dim_t n_states, dim_t states_ws_ld,
        const memory_storage_t &dst_last_layer,
        const rnn_utils::workspace_t &ws) const {

    auto max_wg_size_per_dim = calc_local_range(ctx);

    parallel_for(ctx, copy_kernel_, [&](::sycl::handler &cgh) {
        auto ws_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &ws.states())
                          ->get_in_memory_arg(ctx.stream(), cgh);
        auto dst_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &dst_last_layer)
                          ->get_out_memory_arg(ctx.stream(), cgh);

        ref_rnn_copy_t copy_kernel(
                pd()->copy_res_layer_conf_, ws_mem_arg, dst_mem_arg);
        size_t local_batch = max_wg_size_per_dim;
        size_t local_iter = max_wg_size_per_dim;
        size_t local_channel = max_wg_size_per_dim;
        size_t global_batch
                = calc_global_range(static_cast<size_t>(max_wg_size_per_dim),
                        static_cast<size_t>(batch));
        size_t global_iter
                = calc_global_range(static_cast<size_t>(max_wg_size_per_dim),
                        static_cast<size_t>(n_iter));
        size_t global_channels
                = calc_global_range(static_cast<size_t>(max_wg_size_per_dim),
                        static_cast<size_t>(n_states * dhc));
        cgh.parallel_for(
                ::sycl::nd_range<3>(::sycl::range<3>(global_iter, global_batch,
                                            global_channels),
                        ::sycl::range<3>(
                                local_iter, local_batch, local_channel)),
                copy_kernel);
    });
    return status::success;
}

status_t _ref_rnn_common_t::copy_res_iter(const exec_ctx_t &ctx, dim_t batch,
        dim_t dhc, dim_t sic, dim_t n_iter, dim_t n_layer, dim_t n_dir,
        dim_t n_states, dim_t states_ws_ld,
        const memory_storage_t &dst_last_iter,
        const rnn_utils::workspace_t &ws) const {

    auto max_wg_size_per_dim = calc_local_range(ctx);

    parallel_for(ctx, copy_kernel_, [&](::sycl::handler &cgh) {
        auto src_iter
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &ws.states())
                          ->get_in_memory_arg(ctx.stream(), cgh);
        auto dst_iter = dst_last_iter
                ? utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &dst_last_iter)
                          ->get_out_memory_arg(ctx.stream(), cgh)
                : xpu::sycl::memory_storage_base_t::empty_out_memory_arg(
                        ctx.stream(), cgh);
        ref_rnn_copy_t copy_kernel(
                pd()->copy_res_iter_conf_, src_iter, dst_iter);

        size_t local_batch = max_wg_size_per_dim;
        size_t local_channel = max_wg_size_per_dim;
        size_t local_lay_dir = max_wg_size_per_dim;
        size_t global_batch
                = calc_global_range(static_cast<size_t>(max_wg_size_per_dim),
                        static_cast<size_t>(batch));
        size_t global_channels
                = calc_global_range(static_cast<size_t>(max_wg_size_per_dim),
                        static_cast<size_t>(dhc));
        size_t global_lay_dir
                = calc_global_range(static_cast<size_t>(max_wg_size_per_dim),
                        static_cast<size_t>(n_layer * n_dir));
        cgh.parallel_for(
                ::sycl::nd_range<3>(::sycl::range<3>(global_lay_dir,
                                            global_batch, global_channels),
                        ::sycl::range<3>(
                                local_lay_dir, local_batch, local_channel)),
                copy_kernel);
    });

    return status::success;
}

status_t _ref_rnn_common_t::rnn_bias(const exec_ctx_t &ctx, dim_t batch,
        dim_t dhc, dim_t iter, dim_t lay, dim_t dir,
        const rnn_utils::workspace_t &ws, const rnn_utils::scratch_t &scratch,
        const rnn_utils ::user_data_t &user_data) const {

    auto max_wg_size_per_dim = calc_local_range(ctx);

    parallel_for(ctx, bias_kernel_, [&](::sycl::handler &cgh) {
        auto src_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        scratch.gates(0).get())
                          ->get_inout_memory_arg(ctx.stream(), cgh);
        auto bias_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        user_data.bias(lay, dir).get())
                          ->get_in_memory_arg(ctx.stream(), cgh);

        auto dst_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        ws.states(lay, dir, iter - 1).get())
                          ->get_out_memory_arg(ctx.stream(), cgh);
        ref_rnn_bias bias_kernel(pd()->sycl_rnn_bias_conf_t_, src_mem_arg,
                bias_mem_arg, dst_mem_arg);

        size_t local_batch = max_wg_size_per_dim;
        size_t local_channel = max_wg_size_per_dim;
        size_t global_batch
                = calc_global_range(static_cast<size_t>(max_wg_size_per_dim),
                        static_cast<size_t>(batch));
        size_t global_channels
                = calc_global_range(static_cast<size_t>(max_wg_size_per_dim),
                        static_cast<size_t>(dhc));
        cgh.parallel_for(
                ::sycl::nd_range<3>(
                        ::sycl::range<3>(global_channels, global_batch, 1),
                        ::sycl::range<3>(local_channel, local_batch, 1)),
                bias_kernel);
    });

    return status::success;
}

// //********************* Execution function *********************//

status_t _ref_rnn_common_t::execute_(const exec_ctx_t &ctx) const {

    impl::engine_t *engine = ctx.stream()->engine();

    auto rnn_pd = this->pd();

    const conf_t &rnn = this->pd()->rnn_conf;

    dim_t n_layer = rnn.n_layer;
    dim_t n_dir = rnn.n_dir;
    dim_t n_states = rnn.n_states;
    dim_t n_iter = rnn.n_iter;
    dim_t n_gates = rnn.n_gates;
    dim_t n_bias = rnn.n_bias;
    dim_t batch = rnn.mb;
    dim_t slc = rnn.slc;
    dim_t sic = rnn.sic;
    dim_t dhc = rnn.dhc;
    dim_t dlc = rnn.dlc;

    auto &src_layer_native_ = CTX_IN_STORAGE(DNNL_ARG_SRC_LAYER);
    auto &src_iter_native_ = CTX_IN_STORAGE(DNNL_ARG_SRC_ITER);
    auto &wei_layer_native_ = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_LAYER);
    auto &wei_iter_native_ = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_ITER);
    auto &bias_native_ = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    auto &dst_last_layer_native_ = CTX_OUT_STORAGE(DNNL_ARG_DST_LAYER);
    auto &dst_last_iter_native_ = CTX_OUT_STORAGE(DNNL_ARG_DST_ITER);

    auto scratch_workspace
            = ctx.get_scratchpad_grantor().get_memory_storage(key_rnn_space);
    auto &workspace_ = rnn.is_training ? CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE)
                                       : *scratch_workspace;
    const auto &workspace = rnn_utils::workspace_t(workspace_, rnn);

    const auto scratch
            = rnn_utils::scratch_t(rnn, ctx.get_scratchpad_grantor());

    const rnn_utils::user_data_t user_data(wei_layer_native_,
            pd()->weights_md(0), wei_iter_native_, pd()->weights_md(1),
            bias_native_, pd()->weights_md(2));

    DPRINT("\n%s\n", "+++++++++++++++");
    DPRINT("%s\n", "+++++++++++++++");
    DPRINT("  n_layer         = %lld\n", static_cast<long long>(n_layer));
    DPRINT("  n_dir           = %lld\n", static_cast<long long>(n_dir));
    DPRINT("  n_iter          = %lld\n", static_cast<long long>(n_iter));
    DPRINT("  n_gates         = %lld\n", static_cast<long long>(n_gates));
    DPRINT("  n_bias          = %lld\n", static_cast<long long>(n_bias));
    DPRINT("  n_states        = %lld\n", static_cast<long long>(n_states));
    DPRINT("  n_weights_layer = %lld\n", static_cast<long long>(rnn_pd->SLC()));
    DPRINT("  n_weights_iter  = %lld\n", static_cast<long long>(rnn_pd->SIC()));
    DPRINT("  batch           = %lld\n", static_cast<long long>(batch));
    DPRINT("  slc             = %lld\n", static_cast<long long>(slc));
    DPRINT("  sic             = %lld\n", static_cast<long long>(sic));
    DPRINT("  dhc             = %lld\n", static_cast<long long>(dhc));
    DPRINT("  dlc             = %lld\n", static_cast<long long>(dlc));
    DPRINT("%s\n", "+++++++++++++++");
    DPRINT("  use_workspace   = %s\n", rnn.use_workspace ? "yes" : "no");
    DPRINT("%s\n", "+++++++++++++++");
    DPRINT("  with_bias       = %s\n", rnn_pd->with_bias() ? "yes" : "no");
    DPRINT("  with_dst_iter   = %s\n", rnn_pd->with_dst_iter() ? "yes" : "no");
    DPRINT("%s\n", "+++++++++++++++");

    CHECK(copy_init_layer(ctx, batch, dhc, slc, n_iter, n_layer, n_dir,
            n_states, rnn.states_ws_ld, workspace, src_layer_native_));

    CHECK(copy_init_iter(ctx, batch, dhc, sic, n_iter, n_layer, n_dir, n_states,
            rnn.states_ws_ld, workspace, src_iter_native_));

    // run the execution on the grid
    const grid_ctx_t &grid_struct {
            engine, ctx, user_data, workspace, scratch, pd()->rnn_conf};
    CHECK(this->grid_func(grid_struct));

    // Finally we copy the results to the result buffers

    CHECK(copy_res_layer(ctx, batch, dhc, slc, n_iter, n_layer, n_dir, n_states,
            rnn.states_ws_ld, dst_last_layer_native_, workspace));

    CHECK(copy_res_iter(ctx, batch, dhc, sic, n_iter, n_layer, n_dir, n_states,
            rnn.states_ws_ld, dst_last_iter_native_, workspace));

    return status::success;
};

struct _ref_rnn_common_t;

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
