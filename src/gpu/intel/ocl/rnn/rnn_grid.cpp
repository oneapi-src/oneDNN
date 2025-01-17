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

#include "gpu/intel/ocl/rnn/rnn_grid.hpp"

#include "common/c_types_map.hpp"
#include "common/gemm_utils.hpp"
#include "common/type_helpers.hpp"
#include "gpu/intel/gemm/gpu_gemm.hpp"
#include "gpu/intel/gpu_primitive_attr.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::gpu::intel::gpu_utils;
using namespace prop_kind;
using namespace alg_kind;
using namespace rnn_utils;
using namespace dnnl::impl::memory_tracking::names;

#define AOC array_offset_calculator

static status_t init_layouts_data(rnn_offsets_t &off,
        ocl_conf_t::inner_layouts_t &inner_layouts, const rnn_pd_t *pd,
        const rnn_utils::conf_t &rnn) {
    const memory_desc_wrapper &src_layer_d = pd->src_md(0);
    const memory_desc_wrapper &src_iter_d = pd->src_md(1);
    const memory_desc_wrapper &src_iter_c_d = pd->src_md(2);
    const memory_desc_wrapper &weights_layer_d = pd->weights_md(0);
    const memory_desc_wrapper &weights_iter_d = pd->weights_md(1);
    const memory_desc_wrapper &bias_d = pd->weights_md(2);
    const memory_desc_wrapper &dst_layer_d = pd->dst_md(0);
    const memory_desc_wrapper &dst_iter_d = pd->dst_md(1);
    const memory_desc_wrapper &dst_iter_c_d = pd->dst_md(2);
    const memory_desc_wrapper &diff_src_layer_d = pd->diff_src_md(0);
    const memory_desc_wrapper &diff_src_iter_d = pd->diff_src_md(1);
    const memory_desc_wrapper &diff_src_iter_c_d = pd->diff_src_md(2);
    const memory_desc_wrapper &diff_weights_layer_d = pd->diff_weights_md(0);
    const memory_desc_wrapper &diff_weights_iter_d = pd->diff_weights_md(1);
    const memory_desc_wrapper &diff_bias_d = pd->diff_weights_md(2);
    const memory_desc_wrapper &diff_dst_layer_d = pd->diff_dst_md(0);
    const memory_desc_wrapper &diff_dst_iter_d = pd->diff_dst_md(1);
    const memory_desc_wrapper &diff_dst_iter_c_d = pd->diff_dst_md(2);

    off.src_layer = gpu::intel::get_outer_strides(src_layer_d);
    inner_layouts.src_layer = gpu::intel::get_inner_layout(src_layer_d);
    off.src_iter = gpu::intel::get_outer_strides(src_iter_d);
    inner_layouts.src_iter = gpu::intel::get_inner_layout(src_iter_d);
    if (pd->with_src_iter_c()) {
        off.src_iter_c = gpu::intel::get_outer_strides(src_iter_c_d);
        inner_layouts.src_iter_c = gpu::intel::get_inner_layout(src_iter_c_d);
    }
    off.weights_layer = gpu::intel::get_outer_strides(weights_layer_d);
    inner_layouts.weights_layer = gpu::intel::get_inner_layout(weights_layer_d);
    off.weights_layer_comp_off
            = weights_layer_d.dims()[0] * weights_layer_d.strides()[0];
    off.weights_iter = gpu::intel::get_outer_strides(weights_iter_d);
    inner_layouts.weights_iter = gpu::intel::get_inner_layout(weights_iter_d);
    off.weights_iter_comp_off
            = weights_iter_d.dims()[0] * weights_iter_d.strides()[0];
    off.bias = gpu::intel::get_outer_strides(bias_d);
    inner_layouts.bias = gpu::intel::get_inner_layout(bias_d);
    off.dst_layer = gpu::intel::get_outer_strides(dst_layer_d);
    inner_layouts.dst_layer = gpu::intel::get_inner_layout(dst_layer_d);
    off.dst_iter = gpu::intel::get_outer_strides(dst_iter_d);
    inner_layouts.dst_iter = gpu::intel::get_inner_layout(dst_iter_d);
    if (pd->with_dst_iter_c()) {
        off.dst_iter_c = gpu::intel::get_outer_strides(dst_iter_c_d);
        inner_layouts.dst_iter_c = gpu::intel::get_inner_layout(dst_iter_c_d);
    }

    if (!pd->is_fwd()) {
        if (!utils::everyone_is(rnn.diff_data_type,
                    diff_src_layer_d.data_type(), diff_dst_layer_d.data_type()))
            return status::unimplemented;
        if (!utils::one_of(diff_src_iter_d.data_type(), rnn.diff_data_type,
                    data_type::undef)
                || !utils::one_of(diff_src_iter_c_d.data_type(),
                        rnn.diff_data_type, data_type::undef)
                || !utils::one_of(diff_dst_iter_d.data_type(),
                        rnn.diff_data_type, data_type::undef)
                || !utils::one_of(diff_dst_iter_c_d.data_type(),
                        rnn.diff_data_type, data_type::undef))
            return status::unimplemented;

        off.diff_src_layer = gpu::intel::get_outer_strides(diff_src_layer_d);
        inner_layouts.diff_src_layer
                = gpu::intel::get_inner_layout(diff_src_layer_d);
        off.diff_src_iter = gpu::intel::get_outer_strides(diff_src_iter_d);
        inner_layouts.diff_src_iter
                = gpu::intel::get_inner_layout(diff_src_iter_d);
        if (pd->with_src_iter_c()) {
            off.diff_src_iter_c
                    = gpu::intel::get_outer_strides(diff_src_iter_c_d);
            inner_layouts.diff_src_iter_c
                    = gpu::intel::get_inner_layout(diff_src_iter_c_d);
        }
        off.diff_weights_layer
                = gpu::intel::get_outer_strides(diff_weights_layer_d);
        inner_layouts.diff_weights_layer
                = gpu::intel::get_inner_layout(diff_weights_layer_d);
        off.diff_weights_iter
                = gpu::intel::get_outer_strides(diff_weights_iter_d);
        inner_layouts.diff_weights_iter
                = gpu::intel::get_inner_layout(diff_weights_iter_d);
        off.diff_bias = gpu::intel::get_outer_strides(diff_bias_d);
        inner_layouts.diff_bias = gpu::intel::get_inner_layout(diff_bias_d);
        off.diff_dst_layer = gpu::intel::get_outer_strides(diff_dst_layer_d);
        inner_layouts.diff_dst_layer
                = gpu::intel::get_inner_layout(diff_dst_layer_d);
        off.diff_dst_iter = gpu::intel::get_outer_strides(diff_dst_iter_d);
        inner_layouts.diff_dst_iter
                = gpu::intel::get_inner_layout(diff_dst_iter_d);
        if (pd->with_dst_iter_c()) {
            off.diff_dst_iter_c
                    = gpu::intel::get_outer_strides(diff_dst_iter_c_d);
            inner_layouts.diff_dst_iter_c
                    = gpu::intel::get_inner_layout(diff_dst_iter_c_d);
        }
    }
    return status::success;
}

static status_t init_ocl_conf(rnn_utils::ocl_conf_t &ocl_conf,
        const rnn_pd_t *rnn_pd, const rnn_utils::conf_t &rnn,
        int threads_per_eu, const compute::device_info_t &device_info,
        rnn_offsets_t &off) {

    using namespace rnn_utils;

    const memory_desc_wrapper &src_iter_c_d = rnn_pd->src_md(2);
    const memory_desc_wrapper &weights_layer_d = rnn_pd->weights_md(0);
    const memory_desc_wrapper &dst_iter_c_d = rnn_pd->dst_md(2);

    ocl_conf.src_dt = rnn.src_data_type;
    ocl_conf.src_c_dt = src_iter_c_d.data_type();
    ocl_conf.wei_dt = weights_layer_d.data_type();
    ocl_conf.bia_dt = rnn.bias_data_type;
    ocl_conf.acc_dt = rnn.acc_data_type;
    ocl_conf.aux_dt = rnn.aux_data_type;
    ocl_conf.ws_state_dt = rnn.src_data_type;
    ocl_conf.diff_dt = rnn.diff_data_type;
    ocl_conf.input_dt = rnn.input_data_type;
    ocl_conf.output_dt = rnn.output_data_type;
    ocl_conf.dst_dt = rnn.dst_data_type;
    ocl_conf.dst_c_dt = dst_iter_c_d.data_type();

    ocl_conf.is_fwd = rnn.is_fwd;

    ocl_conf.with_bias = rnn_pd->with_bias();
    ocl_conf.with_src_iter = rnn_pd->with_src_iter();
    ocl_conf.with_src_iter_c = rnn_pd->with_src_iter_c();
    ocl_conf.with_dst_iter = rnn_pd->with_dst_iter();
    ocl_conf.with_dst_iter_c = rnn_pd->with_dst_iter_c();
    ocl_conf.copy_bias = rnn.copy_bias;
    ocl_conf.is_int8 = rnn.is_int8;
    ocl_conf.is_training = rnn.is_training;
    ocl_conf.recompute_gates = rnn.recompute_gates;
    ocl_conf.copy_src_layer = rnn.copy_src_layer;
    ocl_conf.copy_diff_dst_layer = rnn.copy_diff_dst_layer;
    ocl_conf.copy_diff_src_layer = rnn.copy_diff_src_layer;

    ocl_conf.cell_kind = rnn_pd->cell_kind();
    ocl_conf.activation_kind = rnn_pd->activation_kind();
    ocl_conf.direction_kind = rnn_pd->direction();

    ocl_conf.wei_qparam_mask = rnn_pd->attr()->rnn_weights_qparams_.mask_;
    ocl_conf.is_testmode = rnn.is_testmode;

    ocl_conf.threads_per_eu = threads_per_eu;
    ocl_conf.subgroup_size = dev_getenv(
            "subgroup_size", device_info.max_subgroup_size(ocl_conf.acc_dt));
    auto max_elemwise_threads
            = utils::div_up(rnn.mb * rnn.dhc, ocl_conf.subgroup_size);
    auto max_elemwise_threads_per_eu
            = utils::div_up(max_elemwise_threads, device_info.eu_count());
    auto preferred_threads_per_eu = 4;
    ocl_conf.deterministic = rnn_pd->attr()->deterministic_;
    ocl_conf.elemwise_bwd_batch_block = dev_getenv("bwd_batch_block",
            into<int>(ocl_conf.deterministic
                            ? rnn.mb
                            : std::min(into<dim_t>(8),
                                    utils::rnd_up_pow2(
                                            max_elemwise_threads_per_eu
                                            / preferred_threads_per_eu))));
    ocl_conf.need_bias_atomic_reduce
            = !ocl_conf.is_fwd && ocl_conf.elemwise_bwd_batch_block < rnn.mb;

    ocl_conf.cell_comp.is_enabled
            = rnn.cell_fusion.gemm_layer || rnn.cell_fusion.gemm_iter;
    if (ocl_conf.cell_comp.is_enabled) {
        bool fuse_gemm_layer = rnn.cell_fusion.gemm_layer;
        bool fuse_gemm_iter = rnn.cell_fusion.gemm_iter;

        // Due to poor performing tail handling, exact divisibility on subgroup
        // size is preferred
        for (int subgroup_size = ocl_conf.subgroup_size;
                subgroup_size >= device_info.min_subgroup_size();
                subgroup_size /= 2) {
            if (rnn.dhc % subgroup_size == 0) {
                ocl_conf.subgroup_size = subgroup_size;
                break;
            }
        }

        int dhc_thr = dev_getenv("dhc_thr", 1);
        int mb_thr = dev_getenv("mb_thr", 1);

        std::array<dim_t, 9> dhc_hw_threads = {1, 2, 3, 4, 5, 6, 7, 8, 16};
        std::array<dim_t, 3> mb_hw_threads = {1, 2, 4};
        dim_t dhc_tg_best = 1;
        dim_t mb_tg_best = 1;
        double best_score = 0;
        for (auto b_thread : mb_hw_threads) {
            for (auto d_thread : dhc_hw_threads) {
                dim_t dhc_tg = d_thread * ocl_conf.subgroup_size;
                dim_t dhc_block = dhc_thr * dhc_tg;
                dim_t mb_tg = b_thread;
                dim_t mb_block = mb_thr * mb_tg;
                if (size_t(dhc_tg * mb_tg) > device_info.max_wg_size(
                            threads_per_eu == 4, ocl_conf.subgroup_size))
                    break;

                double score = [&]() {
                    // subslice efficiency
                    dim_t used_b_threads
                            = std::min(utils::div_up(rnn.mb, mb_thr), b_thread);
                    dim_t used_d_threads = std::min(
                            utils::div_up(
                                    rnn.dhc, dhc_thr * ocl_conf.subgroup_size),
                            d_thread);
                    double ss_eff = 1.0 * (used_d_threads * used_b_threads)
                            / device_info.max_eus_per_wg();
                    {
                        // Scale to prefer device efficiency over subslice
                        // saturation
                        std::array<double, 4> c {.7, .13, .10, .07};

                        ss_eff = c[0] * nstl::clamp(ss_eff - 0, 0.0, 1.0)
                                + c[1] * nstl::clamp(ss_eff - 1, 0.0, 1.0)
                                + c[2] * nstl::clamp(ss_eff - 2, 0.0, 1.0)
                                + c[3] * nstl::clamp(ss_eff - 3, 0.0, 1.0);
                    }

                    double work_eff
                            = (1.0 * rnn.dhc
                                      / utils::rnd_up(rnn.dhc, dhc_block))
                            * (1.0 * rnn.mb / utils::rnd_up(rnn.mb, mb_block));

                    dim_t ss_count = device_info.eu_count()
                            / device_info.max_eus_per_wg();
                    dim_t wg_to_fill_ss_eu
                            = utils::div_up(device_info.max_eus_per_wg(),
                                    (b_thread * d_thread));
                    dim_t ss_work
                            = utils::div_up(utils::div_up(rnn.dhc, dhc_block)
                                            * utils::div_up(rnn.mb, mb_block),
                                    wg_to_fill_ss_eu);

                    double device_eff
                            = 1.0 * ss_work / utils::rnd_up(ss_work, ss_count);

                    return ss_eff * work_eff * device_eff;
                }();

                if (score > best_score) {
                    dhc_tg_best = dhc_tg;
                    mb_tg_best = mb_tg;
                    best_score = score;
                }
            }
        }

        dim_t dhc_tg = dev_getenv("dhc_tg", into<int>(dhc_tg_best));
        dim_t mb_tg = dev_getenv("mb_tg", into<int>(mb_tg_best));

        int mb_tail = dev_getenv("mb_tail",
                rnn.mb % (mb_tg * mb_thr) != 0
                        || rnn.mb % ocl_conf.subgroup_size != 0);
        int dhc_tail
                = dev_getenv("dhc_tail", rnn.dhc % (dhc_tg * dhc_thr) != 0);
        int k_block = ocl_conf.subgroup_size;

        gpu_assert(dhc_tg % ocl_conf.subgroup_size == 0);

        ocl_conf.cell_comp.compute_gemm_layer = fuse_gemm_layer;
        ocl_conf.cell_comp.gemm_layer_k_tail
                = fuse_gemm_layer && (rnn.slc % k_block != 0);
        ocl_conf.cell_comp.compute_gemm_iter = fuse_gemm_iter;
        ocl_conf.cell_comp.gemm_iter_k_tail
                = fuse_gemm_iter && (rnn.sic % k_block != 0);
        ocl_conf.cell_comp.dhc_tail = dhc_tail;
        ocl_conf.cell_comp.mb_tail = mb_tail;
        ocl_conf.cell_comp.enable_iter_block = rnn.iter_loop != 1;
        ocl_conf.cell_comp.dhc_thr = dhc_thr;
        ocl_conf.cell_comp.dhc_tg = into<int>(dhc_tg);
        ocl_conf.cell_comp.mb_thr = mb_thr;
        ocl_conf.cell_comp.mb_tg = into<int>(mb_tg);
    }

    return status::success;
}

status_t ocl_conf_t::init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const {

    // Fwd operations are not well optimized for larger grf mode
    primitive_attr_t ocl_attr;
    if (!is_fwd)
        CHECK(ocl_attr.set_gpu_attr(gpu_primitive_attr_t(threads_per_eu)));
    ocl_attr.deterministic_ = deterministic;
    kernel_ctx = compute::kernel_ctx_t(&ocl_attr);

    kernel_ctx.add_option("-cl-std=CL2.0");

    kernel_ctx.define_int("IS_FWD", is_fwd);
    kernel_ctx.define_int("IS_TRAINING", is_training);
    kernel_ctx.define_int("RECOMPUTE_GATES", recompute_gates);
    kernel_ctx.define_int("WITH_BIAS", with_bias);
    kernel_ctx.define_int("WITH_SRC_ITER", with_src_iter);
    kernel_ctx.define_int("WITH_SRC_ITER_C", with_src_iter_c);
    kernel_ctx.define_int("WITH_DST_ITER", with_dst_iter);
    kernel_ctx.define_int("WITH_DST_ITER_C", with_dst_iter_c);
    kernel_ctx.define_int("COPY_SRC_LAYER", copy_src_layer);
    kernel_ctx.define_int("COPY_DIFF_DST_LAYER", copy_diff_dst_layer);
    kernel_ctx.define_int("COPY_DIFF_SRC_LAYER", copy_diff_src_layer);

    kernel_ctx.define_int("ELEMWISE_BWD_BATCH_BLOCK", elemwise_bwd_batch_block);
    kernel_ctx.define_int("NEED_BIAS_ATOMIC_REDUCE", need_bias_atomic_reduce);
    kernel_ctx.define_int("VANILLA_RNN", alg_kind::vanilla_rnn);
    kernel_ctx.define_int("VANILLA_LSTM", alg_kind::vanilla_lstm);
    kernel_ctx.define_int("VANILLA_GRU", alg_kind::vanilla_gru);
    kernel_ctx.define_int("LBR_GRU", alg_kind::lbr_gru);
    kernel_ctx.define_int("CELL_KIND", cell_kind);

    kernel_ctx.define_int("ELTWISE_RELU", alg_kind::eltwise_relu);
    kernel_ctx.define_int("ELTWISE_TANH", alg_kind::eltwise_tanh);
    kernel_ctx.define_int("ELTWISE_LOGISTIC", alg_kind::eltwise_logistic);
    kernel_ctx.define_int("ACTIVATION_KIND", activation_kind);

    kernel_ctx.define_int("WS_GATES", rnn_utils::gates);
    kernel_ctx.define_int("WS_STATES", rnn_utils::states);
    kernel_ctx.define_int("WS_C_STATES", rnn_utils::c_states);
    kernel_ctx.define_int("WS_BIAS", rnn_utils::bias);

    kernel_ctx.define_int("L2R", dnnl_unidirectional_left2right);
    kernel_ctx.define_int("R2L", dnnl_unidirectional_right2left);
    kernel_ctx.define_int("CONCAT", dnnl_bidirectional_concat);
    kernel_ctx.define_int("SUM", dnnl_bidirectional_sum);
    kernel_ctx.define_int("DIRECTION_KIND", direction_kind);

    kernel_ctx.define_int("SUBGROUP_SIZE", subgroup_size);

    def_block_offsets(inner_layouts.src_layer, kernel_ctx, "SRC_L");
    def_block_offsets(inner_layouts.src_iter, kernel_ctx, "SRC_I");
    if (with_src_iter_c) {
        def_block_offsets(inner_layouts.src_iter_c, kernel_ctx, "SRC_I_C");
    }
    def_block_offsets(inner_layouts.weights_layer, kernel_ctx, "WEI_L");
    def_block_offsets(inner_layouts.weights_iter, kernel_ctx, "WEI_I");
    def_block_offsets(inner_layouts.dst_layer, kernel_ctx, "DST_L");
    def_block_offsets(inner_layouts.dst_iter, kernel_ctx, "DST_I");
    if (with_dst_iter_c)
        def_block_offsets(inner_layouts.dst_iter_c, kernel_ctx, "DST_I_C");
    def_block_offsets(inner_layouts.bias, kernel_ctx, "BIAS");

    if (!is_fwd) {
        def_block_offsets(
                inner_layouts.diff_src_layer, kernel_ctx, "DIFF_SRC_L");
        def_block_offsets(
                inner_layouts.diff_src_iter, kernel_ctx, "DIFF_SRC_I");
        if (with_src_iter_c)
            def_block_offsets(
                    inner_layouts.diff_src_iter_c, kernel_ctx, "DIFF_SRC_I_C");
        def_block_offsets(
                inner_layouts.diff_weights_layer, kernel_ctx, "DIFF_WEI_L");
        def_block_offsets(
                inner_layouts.diff_weights_iter, kernel_ctx, "DIFF_WEI_I");
        def_block_offsets(
                inner_layouts.diff_dst_layer, kernel_ctx, "DIFF_DST_L");
        def_block_offsets(
                inner_layouts.diff_dst_iter, kernel_ctx, "DIFF_DST_I");
        if (with_dst_iter_c)
            def_block_offsets(
                    inner_layouts.diff_dst_iter_c, kernel_ctx, "DIFF_DST_I_C");
        def_block_offsets(inner_layouts.diff_bias, kernel_ctx, "DIFF_BIAS");
    }

    if (src_dt == data_type::f16) {
        kernel_ctx.set_data_type(data_type::f16);
    } else
        kernel_ctx.set_data_type(data_type::f32);

    def_data_type(kernel_ctx, ws_state_dt, "WS_STATE");
    def_data_type(kernel_ctx, src_dt, "SRC");
    def_data_type(kernel_ctx, src_c_dt, "SRC_C");
    def_data_type(kernel_ctx, wei_dt, "WEI_LAYER");
    def_data_type(kernel_ctx, wei_dt, "WEI_ITER");
    def_data_type(kernel_ctx, acc_dt, "ACC");
    def_data_type(kernel_ctx, aux_dt, "AUX");
    def_data_type(kernel_ctx, bia_dt, "BIAS");
    def_data_type(kernel_ctx, dst_dt, "DST");
    def_data_type(kernel_ctx, dst_c_dt, "DST_C");
    def_data_type(kernel_ctx, input_dt, "INPUT");
    def_data_type(kernel_ctx, output_dt, "OUTPUT");
    def_data_type(kernel_ctx, diff_dt, "DIFF");

    kernel_ctx.define_int("IS_INT8", is_int8);
    kernel_ctx.define_int("COPY_BIAS", copy_bias);
    kernel_ctx.define_int("WEI_QPARAM_MASK", wei_qparam_mask);
    kernel_ctx.define_int("IS_TESTMODE", is_testmode);

    if (cell_comp.is_enabled) {
        kernel_ctx.define_int("CELL_COMP_ENABLED", cell_comp.is_enabled);
        kernel_ctx.define_int(
                "CELL_COMPUTE_GEMM_LAYER", cell_comp.compute_gemm_layer);
        kernel_ctx.define_int(
                "CELL_GEMM_LAYER_K_TAIL", cell_comp.gemm_layer_k_tail);
        kernel_ctx.define_int(
                "CELL_COMPUTE_GEMM_ITER", cell_comp.compute_gemm_iter);
        kernel_ctx.define_int(
                "CELL_GEMM_ITER_K_TAIL", cell_comp.gemm_iter_k_tail);
        kernel_ctx.define_int("CELL_DHC_TAIL", cell_comp.dhc_tail);
        kernel_ctx.define_int("CELL_MB_TAIL", cell_comp.mb_tail);
        kernel_ctx.define_int(
                "CELL_ENABLE_ITER_BLOCK", cell_comp.enable_iter_block);
        kernel_ctx.define_int("CELL_DHC_THR", cell_comp.dhc_thr);
        kernel_ctx.define_int("CELL_BATCH_THR", cell_comp.mb_thr);
    }

    return status::success;
}

template <>
status_t simple_rnn_common_t<prop_kind::forward>::pd_t::set_default_params() {
    using namespace format_tag;
    if (src_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_layer_md_, tnc));
    if (dst_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_layer_md_, tnc));

    // Optional parameters
    if ((!types::is_zero_md(&src_iter_md_))
            && (src_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(src_iter_md_, ldnc));
    if ((!types::is_zero_md(&src_iter_c_md_))
            && (src_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(src_iter_c_md_, ldnc));
    if ((!types::is_zero_md(&bias_md_))
            && (bias_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(bias_md_, ldgo));
    if ((!types::is_zero_md(&dst_iter_md_))
            && (dst_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(dst_iter_md_, ldnc));
    if ((!types::is_zero_md(&dst_iter_c_md_))
            && (dst_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(dst_iter_c_md_, ldnc));

    return status::success;
}

template <>
status_t simple_rnn_common_t<prop_kind::backward>::pd_t::set_default_params() {
    using namespace format_tag;
    int arch_ld = is_xe_hpc ? 128 : 64;
    if (src_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_layer_md_, tnc));
    if (weights_layer_md_.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(weights_layer_md_, ldgoi));
        if (!rnn_conf.is_int8)
            CHECK(rnn_utils::set_good_strides(
                    arch_ld, weights_layer_md_, ldgoi));
    }
    if (dst_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_layer_md_, tnc));

    if (weights_iter_md_.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(weights_iter_md_, ldgoi));
        if (!rnn_conf.is_int8)
            CHECK(rnn_utils::set_good_strides(
                    arch_ld, weights_iter_md_, ldgoi));
    }

    if (diff_src_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(diff_src_layer_md_, tnc));
    if (diff_weights_layer_md_.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_weights_layer_md_, ldigo));
        if (!rnn_conf.is_int8)
            CHECK(rnn_utils::set_good_strides(
                    arch_ld, diff_weights_layer_md_, ldigo));
    }
    if (diff_weights_iter_md_.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_weights_iter_md_, ldigo));
        if (!rnn_conf.is_int8)
            CHECK(rnn_utils::set_good_strides(
                    arch_ld, diff_weights_iter_md_, ldigo));
    }
    if (diff_dst_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(diff_dst_layer_md_, tnc));

    // Optional parameters
    if ((!types::is_zero_md(&src_iter_md_))
            && (src_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(src_iter_md_, ldnc));
    if ((!types::is_zero_md(&src_iter_c_md_))
            && (src_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(src_iter_c_md_, ldnc));
    if ((!types::is_zero_md(&bias_md_))
            && (bias_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(bias_md_, ldgo));
    if ((!types::is_zero_md(&dst_iter_md_))
            && (dst_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(dst_iter_md_, ldnc));
    if ((!types::is_zero_md(&dst_iter_c_md_))
            && (dst_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(dst_iter_c_md_, ldnc));

    if ((!types::is_zero_md(&diff_src_iter_md_))
            && (diff_src_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_src_iter_md_, ldnc));
    if ((!types::is_zero_md(&diff_src_iter_c_md_))
            && (diff_src_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_src_iter_c_md_, ldnc));
    if ((!types::is_zero_md(&diff_bias_md_))
            && (diff_bias_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_bias_md_, ldgo));
    if ((!types::is_zero_md(&diff_dst_iter_md_))
            && (diff_dst_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_dst_iter_md_, ldnc));
    if ((!types::is_zero_md(&diff_dst_iter_c_md_))
            && (diff_dst_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_dst_iter_c_md_, ldnc));

    return status::success;
}

template <prop_kind_t aprop>
status_t simple_rnn_common_t<aprop>::pd_t::init(impl::engine_t *engine) {
    using namespace prop_kind;
    using namespace utils;
    using namespace rnn_utils;
    using namespace format_tag;

    assert(engine->kind() == engine_kind::gpu);
    auto *compute_engine
            = utils::downcast<const compute::compute_engine_t *>(engine);

    const compute::device_info_t &device_info
            = *(compute_engine->device_info());
    max_eus_per_wg = device_info.max_eus_per_wg();

    const alg_kind_t cell_kind = this->desc()->cell_kind;

    data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
    data_type_t weights_iter_dt = this->desc()->weights_iter_desc.data_type;
    data_type_t weights_layer_dt = this->desc()->weights_layer_desc.data_type;
    data_type_t bias_dt = this->desc()->bias_desc.data_type;

    bool src_is_u8 = src_layer_dt == data_type::u8;
    bool src_is_f16 = src_layer_dt == data_type::f16;
    if (src_is_u8)
        acc_data_t = data_type::s32;
    else if (src_is_f16 && aprop == prop_kind::forward_inference)
        acc_data_t = data_type::f16;
    else
        acc_data_t = data_type::f32;

    src_type = src_layer_dt;
    weights_type = weights_layer_dt;

    VDISPATCH_RNN(
            one_of(cell_kind, alg_kind::vanilla_rnn, alg_kind::vanilla_lstm,
                    alg_kind::lbr_gru, alg_kind::vanilla_gru),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_RNN(!this->is_lstm_peephole(), "is_lstm_peephole");
    VDISPATCH_RNN(!this->is_lstm_projection(), "is_lstm_projection");
    VDISPATCH_RNN(IMPLICATION(aprop == prop_kind::forward,
                          one_of(this->desc()->prop_kind, forward_training,
                                  forward_inference)),
            VERBOSE_BAD_PROPKIND);
    VDISPATCH_RNN(IMPLICATION(aprop == backward,
                          one_of(this->desc()->prop_kind, backward)),
            VERBOSE_BAD_PROPKIND);
    VDISPATCH_RNN(
            IMPLICATION(src_type == data_type::bf16, bias_dt == data_type::f32),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_RNN(((aprop == prop_kind::forward && src_layer_dt == data_type::u8
                           && weights_layer_dt == data_type::s8
                           && cell_kind == alg_kind::vanilla_lstm)
                          || (aprop == prop_kind::forward
                                  && one_of(src_layer_dt, data_type::f16,
                                          data_type::f32, data_type::bf16)
                                  && weights_layer_dt == src_layer_dt)
                          || (aprop == prop_kind::backward
                                  && one_of(weights_layer_dt, data_type::f32,
                                          data_type::f16, data_type::bf16)
                                  && weights_layer_dt == src_layer_dt)),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_RNN(weights_iter_dt == weights_layer_dt, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_RNN_SC(this->set_default_params(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_RNN(this->with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_RNN(IMPLICATION(src_layer_dt == data_type::u8,
                          this->desc()->prop_kind == forward_inference),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_RNN(
            compute_engine->mayiuse(compute::device_ext_t::intel_subgroups),
            VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "subgroups");
    VDISPATCH_RNN(
            IMPLICATION(src_layer_dt == data_type::f16,
                    true
                            && compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16)
                            && compute_engine->mayiuse(compute::device_ext_t::
                                            intel_subgroups_short)),
            VERBOSE_UNSUPPORTED_DT_CFG);

    init_rnn_conf(rnn_conf, *this->desc(), this->src_md(0), this->src_md(1),
            this->weights_md(0), this->weights_md(1), this->dst_md(0),
            this->dst_md(1), this->diff_dst_md(0), this->desc()->bias_desc,
            acc_data_t, device_info);

    if (rnn_conf.is_int8) {
        auto has_trivial_strides = [](const memory_desc_wrapper &md) {
            return md.is_dense(true);
        };
        VCONDCHECK(primitive, create, dispatch, rnn,
                has_trivial_strides(this->src_layer_md_), status::unimplemented,
                VERBOSE_NONTRIVIAL_STRIDE);
        VCONDCHECK(primitive, create, dispatch, rnn,
                has_trivial_strides(this->src_iter_md_), status::unimplemented,
                VERBOSE_NONTRIVIAL_STRIDE);
        VCONDCHECK(primitive, create, dispatch, rnn,
                has_trivial_strides(this->src_iter_c_md_),
                status::unimplemented, VERBOSE_NONTRIVIAL_STRIDE);
        VCONDCHECK(primitive, create, dispatch, rnn,
                has_trivial_strides(this->dst_layer_md_), status::unimplemented,
                VERBOSE_NONTRIVIAL_STRIDE);
        VCONDCHECK(primitive, create, dispatch, rnn,
                has_trivial_strides(this->dst_iter_md_), status::unimplemented,
                VERBOSE_NONTRIVIAL_STRIDE);
        VCONDCHECK(primitive, create, dispatch, rnn,
                has_trivial_strides(this->dst_iter_c_md_),
                status::unimplemented, VERBOSE_NONTRIVIAL_STRIDE);
    }

    init_test_mode(rnn_conf, *this->attr());

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

    // TODO: implement something like check layout consistency
    switch (aprop) {
        case (prop_kind::forward): break;
        case (prop_kind::backward):
            VDISPATCH_RNN(utils::one_of(this->desc()->prop_kind, backward),
                    VERBOSE_BAD_PROPKIND);
            break;
        default: return status::unimplemented;
    }

    // Set weights descriptors to desired format
    VDISPATCH_RNN_SC(set_weights_desc(this->weights_layer_md_, rnn_conf),
            "unsupported weights layer memory descriptor");
    VDISPATCH_RNN_SC(set_weights_desc(this->weights_iter_md_, rnn_conf),
            "unsupported weights iter memory descriptor");

    // Check dimensions consistency
    int ls_multiplier
            = (this->direction() == dnnl_bidirectional_concat) ? 2 : 1;

    VDISPATCH_RNN((ls_multiplier * this->DHC() == this->DLC()),
            VERBOSE_INCONSISTENT_DIM, "DHC", (int)this->DHC(), "DLC",
            (int)this->DLC());
    VDISPATCH_RNN(
            (ls_multiplier * this->SLC()) == this->DLC() || (this->L() == 1),
            VERBOSE_INCONSISTENT_DIM, "SLC", (int)this->SLC(), "DLC",
            (int)this->DLC());
    VDISPATCH_RNN((this->SIC() == this->DHC() || (this->T() == 1)),
            VERBOSE_INCONSISTENT_DIM, "SIC", (int)this->SIC(), "DHC",
            (int)this->DHC());

    set_rnn_conf(rnn_conf, *this->desc(), this->src_md(0), this->diff_src_md(0),
            this->diff_dst_md(0), this->weights_md(0), this->weights_md(1),
            this->diff_weights_md(0), this->diff_weights_md(1));

    dim_t workspace_size = get_workspace_size(rnn_conf);

    // initialize the workspace_pd if needed
    if (rnn_conf.use_workspace) {
        dims_t ws_dims = {workspace_size};
        VDISPATCH_RNN_SC(memory_desc_init_by_tag(
                                 this->ws_md_, 1, ws_dims, data_type::u8, x),
                "memory_desc_init_by_tag()");
    }

    VDISPATCH_RNN_SC(
            init_layouts_data(off, ocl_conf.inner_layouts, this, rnn_conf),
            "init_layouts_data()");

    dim_t batch = rnn_conf.mb;
    dim_t n_gates = rnn_conf.n_gates;
    dim_t slc = rnn_conf.slc;
    dim_t sic = rnn_conf.sic;
    dim_t dhc = rnn_conf.dhc;

    auto fpmath_mode = this->attr()->fpmath_.mode_;
    int threads_per_eu = 0;

    // The inputs of create_gemm_pd describe a gemm in column major.
    // Below, we have to transpose the a and b descriptor to describe
    // the GEMM as a row major problem.
    auto create_gemm_pd =
            [&](std::shared_ptr<primitive_desc_t> &gemm_pd, dim_t m, dim_t n,
                    dim_t k, strides_t<2> a_strides, strides_t<2> b_strides,
                    strides_t<2> c_strides, data_type_t a_dt, data_type_t b_dt,
                    data_type_t c_dt, float beta) -> status_t {
        memory_desc_t a_md, b_md, c_md;
        dims_t a_dims = {n, k}, b_dims = {k, m}, c_dims = {n, m};

        dims_t b_strides_md = {b_strides[0], b_strides[1]};
        CHECK(memory_desc_init_by_strides(b_md, 2, b_dims, b_dt, b_strides_md));
        dims_t a_strides_md = {a_strides[0], a_strides[1]};
        CHECK(memory_desc_init_by_strides(a_md, 2, a_dims, a_dt, a_strides_md));
        dims_t c_strides_md = {c_strides[0], c_strides[1]};
        CHECK(memory_desc_init_by_strides(c_md, 2, c_dims, c_dt, c_strides_md));

        primitive_attr_t attr;
        CHECK(attr.post_ops_.append_sum(beta));
        CHECK(attr.set_fpmath_mode(fpmath_mode));
        attr.deterministic_ = this->attr()->deterministic_;
        CHECK(dnnl::impl::create_gemm_pd(gemm_pd, engine, &a_md, &b_md, &c_md,
                &glob_zero_md, c_dt, &attr));
        if (threads_per_eu == 0)
            CHECK(gemm_pd->query(
                    query::preferred_gpu_threads_per_eu, 0, &threads_per_eu));
        else if (get_verbose_dev_mode(verbose_t::debuginfo) > 1) {
            auto t = 0;
            CHECK(gemm_pd->query(query::preferred_gpu_threads_per_eu, 0, &t));
            if (t != threads_per_eu)
                verbose_printf("[WARNING] GEMM grf modes are inconsistent");
        }
        return status::success;
    };

    dim_t layer_merged_size
            = rnn_conf.merge_gemm_layer ? batch * rnn_conf.n_iter : batch;
    dim_t iter_merged_size
            = rnn_conf.merge_gemm_iter ? batch * rnn_conf.n_iter : batch;

    float gemm_iter_fwd_beta = this->is_lbr() ? 0.0f : 1.0f;
    float gemm_iter_bwd_beta = this->is_lbr() ? 1.0f : 0.0f;
    if (aprop == prop_kind::forward || rnn_conf.recompute_gates) {
        if (!rnn_conf.cell_fusion.gemm_layer) {
            VDISPATCH_RNN_SC(
                    create_gemm_pd(gemm_layer_fwd_pd_, n_gates * dhc,
                            layer_merged_size, slc, {rnn_conf.states_ws_ld, 1},
                            {off.weights_layer[2], off.weights_layer[4]},
                            {rnn_conf.scratch_gates_ld, 1}, weights_type,
                            src_type, rnn_conf.acc_data_type, 0.0),
                    "create_gemm_pd(gemm_layer_fwd_pd_)");
            if (!rnn_conf.copy_src_layer) {
                if (off.src_layer[1] != rnn_conf.states_ws_ld)
                    VDISPATCH_RNN_SC(
                            create_gemm_pd(gemm_layer_fwd_src_pd_,
                                    n_gates * dhc, layer_merged_size, slc,
                                    {off.src_layer[1], off.src_layer[2]},
                                    {off.weights_layer[2],
                                            off.weights_layer[4]},
                                    {rnn_conf.scratch_gates_ld, 1},
                                    weights_type, src_type,
                                    rnn_conf.acc_data_type, 0.0),
                            "create_gemm_pd(gemm_layer_fwd_src_pd_)");
                else
                    gemm_layer_fwd_src_pd_ = gemm_layer_fwd_pd_;
            }
        }
        if (!rnn_conf.cell_fusion.gemm_iter) {
            if (rnn_conf.is_vanilla_gru) {
                VDISPATCH_RNN_SC(
                        create_gemm_pd(gemm_iter_fwd_pd_, (n_gates - 1) * dhc,
                                batch, sic, {rnn_conf.states_ws_ld, 1},
                                {off.weights_iter[2], off.weights_iter[4]},
                                {rnn_conf.scratch_gates_ld, 1}, weights_type,
                                src_type, rnn_conf.acc_data_type,
                                gemm_iter_fwd_beta),
                        "create_gemm_pd(gemm_iter_fwd_pd_)");
                VDISPATCH_RNN_SC(
                        create_gemm_pd(gemm_iter_fwd_2_pd_, dhc, batch, sic,
                                {rnn_conf.states_ws_ld, 1},
                                {off.weights_iter[2], off.weights_iter[4]},
                                {rnn_conf.scratch_gates_ld, 1}, weights_type,
                                src_type, rnn_conf.acc_data_type,
                                gemm_iter_fwd_beta),
                        "create_gemm_pd(gemm_iter_fwd_2_pd_)");
            } else {
                VDISPATCH_RNN_SC(
                        create_gemm_pd(gemm_iter_fwd_pd_, n_gates * dhc, batch,
                                sic, {rnn_conf.states_ws_ld, 1},
                                {off.weights_iter[2], off.weights_iter[4]},
                                {rnn_conf.gates_ws_ld, 1}, weights_type,
                                src_type, rnn_conf.acc_data_type,
                                gemm_iter_fwd_beta),
                        "create_gemm_pd(gemm_iter_fwd_pd_)");
            }
        }
    }

    if (aprop == prop_kind::backward) {
        if (rnn_conf.is_vanilla_gru) {
            VDISPATCH_RNN_SC(
                    create_gemm_pd(gemm_iter_bwd_pd_, sic, batch,
                            (n_gates - 1) * dhc,
                            {rnn_conf.scratch_diff_gates_ld, 1},
                            {off.weights_iter[4], off.weights_iter[2]},
                            {rnn_conf.scratch_diff_states_ld, 1}, weights_type,
                            src_type, rnn_conf.acc_data_type, 1.0f),
                    "create_gemm_pd(gemm_iter_bwd_pd_)");
            VDISPATCH_RNN_SC(
                    create_gemm_pd(gemm_iter_bwd_2_pd_, sic, batch, dhc,
                            {rnn_conf.scratch_diff_gates_ld, 1},
                            {off.weights_iter[4], off.weights_iter[2]},
                            {rnn_conf.scratch_diff_states_ld, 1}, weights_type,
                            src_type, rnn_conf.acc_data_type, 0.0f),
                    "create_gemm_pd(gemm_iter_bwd_2_pd_)");
            VDISPATCH_RNN_SC(
                    create_gemm_pd(gemm_diff_wei_iter_pd_, (n_gates - 1) * dhc,
                            sic, iter_merged_size, {1, rnn_conf.states_ws_ld},
                            {rnn_conf.scratch_diff_gates_ld, 1},
                            {off.diff_weights_iter[2],
                                    off.diff_weights_iter[4]},
                            weights_type, src_type, rnn_conf.acc_data_type,
                            1.0f),
                    "create_gemm_pd(gemm_diff_wei_iter_pd_)");
            VDISPATCH_RNN_SC(
                    create_gemm_pd(gemm_diff_wei_iter_2_pd_, dhc, sic,
                            iter_merged_size, {1, rnn_conf.states_ws_ld},
                            {rnn_conf.scratch_diff_gates_ld, 1},
                            {off.diff_weights_iter[2],
                                    off.diff_weights_iter[4]},
                            weights_type, src_type, rnn_conf.acc_data_type,
                            1.0f),
                    "create_gemm_pd(gemm_diff_wei_iter_2_pd_)");
        } else {
            VDISPATCH_RNN_SC(
                    create_gemm_pd(gemm_iter_bwd_pd_, sic, batch, n_gates * dhc,
                            {rnn_conf.scratch_diff_gates_ld, 1},
                            {off.weights_iter[4], off.weights_iter[2]},
                            {rnn_conf.scratch_diff_states_ld, 1}, weights_type,
                            src_type, rnn_conf.acc_data_type,
                            gemm_iter_bwd_beta),
                    "create_gemm_pd(gemm_iter_bwd_pd_)");
            VDISPATCH_RNN_SC(
                    create_gemm_pd(gemm_diff_wei_iter_pd_, n_gates * dhc, sic,
                            iter_merged_size, {1, rnn_conf.states_ws_ld},
                            {rnn_conf.scratch_diff_gates_ld, 1},
                            {off.diff_weights_iter[2],
                                    off.diff_weights_iter[4]},
                            weights_type, src_type, rnn_conf.acc_data_type,
                            1.0f),
                    "create_gemm_pd(gemm_diff_wei_iter_pd_)");
        }
        VDISPATCH_RNN_SC(
                create_gemm_pd(gemm_layer_bwd_pd_, slc, layer_merged_size,
                        n_gates * dhc, {rnn_conf.scratch_diff_gates_ld, 1},
                        {off.weights_layer[4], off.weights_layer[2]},
                        {rnn_conf.scratch_diff_states_ld, 1}, weights_type,
                        src_type, rnn_conf.acc_data_type, 0.0f),
                "create_gemm_pd(gemm_layer_bwd_pd_)");
        if (!rnn_conf.copy_diff_src_layer) {
            if (rnn_conf.scratch_diff_states_ld != off.diff_src_layer[1])
                VDISPATCH_RNN_SC(
                        create_gemm_pd(gemm_layer_bwd_src_pd_, slc,
                                layer_merged_size, n_gates * dhc,
                                {rnn_conf.scratch_diff_gates_ld, 1},
                                {off.weights_layer[4], off.weights_layer[2]},
                                {off.diff_src_layer[1], 1}, weights_type,
                                src_type, rnn_conf.acc_data_type, 0.0f),
                        "create_gemm_pd(gemm_layer_bwd_src_pd_)");
            else
                gemm_layer_bwd_src_pd_ = gemm_layer_bwd_pd_;
        }
        VDISPATCH_RNN_SC(
                create_gemm_pd(gemm_diff_wei_layer_pd_, n_gates * dhc, slc,
                        layer_merged_size, {1, rnn_conf.states_ws_ld},
                        {rnn_conf.scratch_diff_gates_ld, 1},
                        {off.diff_weights_layer[2], off.diff_weights_layer[4]},
                        weights_type, src_type, rnn_conf.acc_data_type, 1.0f),
                "create_gemm_pd(gemm_diff_wei_layer_pd_)");
        if (!rnn_conf.copy_src_layer) {
            if (off.src_layer[1] != rnn_conf.states_ws_ld)
                VDISPATCH_RNN_SC(create_gemm_pd(gemm_diff_wei_layer_src_pd_,
                                         n_gates * dhc, slc, layer_merged_size,
                                         {off.src_layer[2], off.src_layer[1]},
                                         {rnn_conf.scratch_diff_gates_ld, 1},
                                         {off.diff_weights_layer[2],
                                                 off.diff_weights_layer[4]},
                                         weights_type, src_type,
                                         rnn_conf.acc_data_type, 1.0f),
                        "create_gemm_pd(gemm_diff_wei_layer_src_pd_)");
            else
                gemm_diff_wei_layer_src_pd_ = gemm_diff_wei_layer_pd_;
        }
    }

    VDISPATCH_RNN_SC(init_ocl_conf(ocl_conf, this, rnn_conf, threads_per_eu,
                             device_info, this->off),
            "init_ocl_conf()");

    init_scratchpad(rnn_conf.use_workspace ? 0 : workspace_size);
    return status::success;
}

template <prop_kind_t aprop>
status_t simple_rnn_common_t<aprop>::init(impl::engine_t *engine) {
    using namespace rnn_utils;

    switch (pd()->cell_kind()) {
        case dnnl_vanilla_lstm:
            cell_func = &class_name::cell_execution;
            elemwise_common = pd()->src_type == data_type::u8
                            && pd()->weights_type == data_type::s8
                    ? &class_name::lstm_elemwise_u8s8
                    : &class_name::lstm_elemwise;
            break;
        case dnnl_vanilla_rnn:
            cell_func = &class_name::cell_execution;
            elemwise_common = &class_name::rnn_elemwise;
            break;
        case dnnl_vanilla_gru:
            cell_func = &class_name::cell_execution_gru;
            elemwise_gru = &class_name::gru_elemwise;
            break;
        case dnnl_lbr_gru:
            cell_func = &class_name::cell_execution_gru_lbr;
            elemwise_gru_lbr = &class_name::gru_lbr_elemwise;
            break;
        default: break;
    }

    grid_computation = &class_name::linear_execution;

    const conf_t &rnn = pd()->rnn_conf;
    rnn_utils::set_workspace_offsets(rnn, ws_gates_offset_, ws_states_offset_,
            ws_c_states_offset_, ws_grid_comp_offset_, ws_bias_offset_);

    auto kernel_names = pd()->ocl_conf.get_kernel_names();
    CHECK(create_kernels(engine, kernels_, kernel_names, pd()->ocl_conf));

    bool gemm_ok = utils::everyone_is(status::success,
            pd()->gemm_layer_fwd_pd_ ? create_nested_primitive(
                    gemm_layer_fwd_, pd()->gemm_layer_fwd_pd_, engine)
                                     : status::success,
            pd()->gemm_layer_fwd_src_pd_ ? create_nested_primitive(
                    gemm_layer_fwd_src_, pd()->gemm_layer_fwd_src_pd_, engine)
                                         : status::success,
            pd()->gemm_iter_fwd_pd_ ? create_nested_primitive(
                    gemm_iter_fwd_, pd()->gemm_iter_fwd_pd_, engine)
                                    : status::success);
    switch (aprop) {
        case prop_kind::forward:
            gemm_ok = true
                    && utils::everyone_is(status::success,
                            rnn.is_vanilla_gru
                                    ? create_nested_primitive(gemm_iter_fwd_2_,
                                            pd()->gemm_iter_fwd_2_pd_, engine)
                                    : status::success);
            break;
        case prop_kind::backward:
            gemm_ok = true
                    && utils::everyone_is(status::success,
                            create_nested_primitive(gemm_layer_bwd_,
                                    pd()->gemm_layer_bwd_pd_, engine),
                            (pd()->gemm_layer_bwd_src_pd_
                                            ? create_nested_primitive(
                                                    gemm_layer_bwd_src_,
                                                    pd()->gemm_layer_bwd_src_pd_,
                                                    engine)
                                            : status::success),
                            create_nested_primitive(gemm_iter_bwd_,
                                    pd()->gemm_iter_bwd_pd_, engine),
                            create_nested_primitive(gemm_diff_wei_layer_,
                                    pd()->gemm_diff_wei_layer_pd_, engine),
                            (pd()->gemm_diff_wei_layer_src_pd_
                                            ? create_nested_primitive(
                                                    gemm_diff_wei_layer_src_,
                                                    pd()->gemm_diff_wei_layer_src_pd_,
                                                    engine)
                                            : status::success),
                            create_nested_primitive(gemm_diff_wei_iter_,
                                    pd()->gemm_diff_wei_iter_pd_, engine),
                            rnn.is_vanilla_gru
                                    ? create_nested_primitive(gemm_iter_bwd_2_,
                                            pd()->gemm_iter_bwd_2_pd_, engine)
                                    : status::success,
                            rnn.is_vanilla_gru ? create_nested_primitive(
                                    gemm_diff_wei_iter_2_,
                                    pd()->gemm_diff_wei_iter_2_pd_, engine)
                                               : status::success);
            break;
        default: assert(!"unknown prop_kind"); return status::invalid_arguments;
    }

    if (!gemm_ok) return status::runtime_error;

    return status::success;
}

template <prop_kind_t aprop>
status_t simple_rnn_common_t<aprop>::init_res_storage(
        impl::engine_t *engine, gpu_resource_t *r) const {
    if (pd()->rnn_conf.is_int8 && pd()->rnn_conf.copy_bias) {
        dim_t size = pd()->rnn_conf.n_gates * pd()->rnn_conf.dhc
                * sizeof(float); // G * O * sizeof(float);
        memory_storage_t *tmp_mem_storage_ptr = nullptr;
        CHECK(engine->create_memory_storage(&tmp_mem_storage_ptr, size));
        // copy bias to memory storage
        std::unique_ptr<memory_storage_t> tmp_mem_storage(tmp_mem_storage_ptr);
        void *scales_ptr = nullptr;
        CHECK(tmp_mem_storage->map_data(&scales_ptr, nullptr,
                sizeof(float) * pd()->rnn_conf.n_gates * pd()->rnn_conf.dhc));
        utils::array_copy((float *)scales_ptr,
                pd()->attr()->rnn_weights_qparams_.scales_,
                pd()->rnn_conf.n_gates * pd()->rnn_conf.dhc);
        CHECK(tmp_mem_storage->unmap_data(scales_ptr, nullptr));
        r->add_memory_storage(SCALES_, std::move(tmp_mem_storage));
    }

    // Prepare testmode scales defined by attributes. Doesn't introduce
    // primitive state, because it is a constant memory -- will not be
    // changed during execution.
    // TODO: add the testmode scales to ws
    if (pd()->rnn_conf.is_testmode && pd_->attr()->rnn_tparams_.scales_) {
        dim_t size = pd()->rnn_conf.tm_ngates
                * sizeof(*pd_->attr()->rnn_tparams_.scales_);
        memory_storage_t *tmp_mem_storage_ptr = nullptr;
        CHECK(engine->create_memory_storage(&tmp_mem_storage_ptr, size));

        std::unique_ptr<memory_storage_t> tmp_mem_storage(tmp_mem_storage_ptr);
        void *tm_scales_ptr = nullptr;
        CHECK(tmp_mem_storage->map_data(&tm_scales_ptr, nullptr,
                sizeof(float) * pd()->attr()->rnn_tparams_.ngates_));
        utils::array_copy((float *)tm_scales_ptr,
                pd()->attr()->rnn_tparams_.scales_,
                pd()->attr()->rnn_tparams_.ngates_);
        CHECK(tmp_mem_storage->unmap_data(tm_scales_ptr, nullptr));
        r->add_memory_storage(TM_SCALES_, std::move(tmp_mem_storage));
    }
    return status::success;
}

template <prop_kind_t aprop>
gemm_sig((simple_rnn_common_t<aprop>::gemm_primitive)) {
    // We flip A and B here since the GEMM API is row major but the
    // RNN code describes GEMM in column major fashion
    gemm_exec_args_t gemm_args;
    gemm_args.a = b.get();
    gemm_args.b = a.get();
    gemm_args.c = c.get();

    auto gemm_ctx = gemm_exec_ctx_t(ctx, gemm_args);

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
            CHECK(gpu_gemm(gemm_iter_fwd_)->execute(gemm_ctx));
            break;
        case gemm_iter_fwd_2:
            init_gemm_nested_scratchpad(gemm_iter_fwd_2_,
                    rnn_utils::scratch_t::key_gemm_iter_fwd_2);
            CHECK(gpu_gemm(gemm_iter_fwd_2_)->execute(gemm_ctx));
            break;
        case gemm_layer_fwd:
            init_gemm_nested_scratchpad(
                    gemm_layer_fwd_, rnn_utils::scratch_t::key_gemm_layer_fwd);
            CHECK(gpu_gemm(gemm_layer_fwd_)->execute(gemm_ctx));
            break;
        case gemm_layer_fwd_src:
            init_gemm_nested_scratchpad(gemm_layer_fwd_src_,
                    rnn_utils::scratch_t::key_gemm_layer_fwd_src);
            CHECK(gpu_gemm(gemm_layer_fwd_src_)->execute(gemm_ctx));
            break;
        case gemm_iter_bwd:
            init_gemm_nested_scratchpad(
                    gemm_iter_bwd_, rnn_utils::scratch_t::key_gemm_iter_bwd);
            CHECK(gpu_gemm(gemm_iter_bwd_)->execute(gemm_ctx));
            break;
        case gemm_iter_bwd_2:
            init_gemm_nested_scratchpad(gemm_iter_bwd_2_,
                    rnn_utils::scratch_t::key_gemm_iter_bwd_2);
            CHECK(gpu_gemm(gemm_iter_bwd_2_)->execute(gemm_ctx));
            break;
        case gemm_layer_bwd:
            init_gemm_nested_scratchpad(
                    gemm_layer_bwd_, rnn_utils::scratch_t::key_gemm_layer_bwd);
            CHECK(gpu_gemm(gemm_layer_bwd_)->execute(gemm_ctx));
            break;
        case gemm_layer_bwd_src:
            init_gemm_nested_scratchpad(gemm_layer_bwd_src_,
                    rnn_utils::scratch_t::key_gemm_layer_bwd);
            CHECK(gpu_gemm(gemm_layer_bwd_src_)->execute(gemm_ctx));
            break;
        case gemm_diff_wei_iter:
            init_gemm_nested_scratchpad(gemm_diff_wei_iter_,
                    rnn_utils::scratch_t::key_gemm_diff_wei_iter);
            CHECK(gpu_gemm(gemm_diff_wei_iter_)->execute(gemm_ctx));
            break;
        case gemm_diff_wei_layer:
            init_gemm_nested_scratchpad(gemm_diff_wei_layer_,
                    rnn_utils::scratch_t::key_gemm_diff_wei_layer);
            CHECK(gpu_gemm(gemm_diff_wei_layer_)->execute(gemm_ctx));
            break;
        case gemm_diff_wei_layer_src:
            init_gemm_nested_scratchpad(gemm_diff_wei_layer_src_,
                    rnn_utils::scratch_t::key_gemm_diff_wei_layer_src);
            CHECK(gpu_gemm(gemm_diff_wei_layer_src_)->execute(gemm_ctx));
            break;
        case gemm_diff_wei_iter_2:
            init_gemm_nested_scratchpad(gemm_diff_wei_iter_2_,
                    rnn_utils::scratch_t::key_gemm_diff_wei_iter_2);
            CHECK(gpu_gemm(gemm_diff_wei_iter_2_)->execute(gemm_ctx));
            break;
        default: assert(!"unknown gemm_kind"); return status::runtime_error;
    }
    return status::success;
}

//*************** Grid computations strategy: linear ***************//
template <prop_kind_t aprop>
grid_execution_sig((simple_rnn_common_t<aprop>::linear_execution)) {
    const conf_t &rnn = pd()->rnn_conf;
    dim_t n_layer = rnn.n_layer;
    dim_t n_dir = rnn.n_dir;
    dim_t n_iter = rnn.n_iter;

    if (aprop == prop_kind::backward && pd()->diff_weights_overwrite()) {
        compute::compute_stream_t *compute_stream
                = utils::downcast<compute::compute_stream_t *>(ctx.stream());
        auto zero = [&](const memory_storage_t &data, int arg_id) {
            auto mdw = memory_desc_wrapper(pd()->arg_md(arg_id));
            return compute_stream->fill(data, 0, mdw.size(),
                    compute_stream->ctx().get_deps(),
                    compute_stream->ctx().get_deps());
        };

        CHECK(zero(diff_bias, DNNL_ARG_DIFF_BIAS));
        CHECK(zero(user_data.diff_wei_layer(), DNNL_ARG_DIFF_WEIGHTS_LAYER));
        CHECK(zero(user_data.diff_wei_iter(), DNNL_ARG_DIFF_WEIGHTS_ITER));
    }

    // Grid Computation for RNN with a cell execution call
    for (dim_t dir = 0; dir < n_dir; dir++) {
        for (dim_t j = 0; j < n_layer; j++) {
            dim_t lay = (aprop == prop_kind::forward) ? j : n_layer - j - 1;

            auto grid_iter = rnn.merge_gemm_iter
                    ? workspace.states_range(lay, n_layer, dir, dir, -1, -1)
                    : sub_buffer_t();

            if ((aprop == prop_kind::forward || rnn.recompute_gates)
                    && rnn.merge_gemm_layer && !rnn.cell_fusion.gemm_layer) {
                auto grid_layer = (!rnn.copy_src_layer && lay == 0)
                        ? user_data.src_layer(dir, 0, true)
                        : workspace.states_range(
                                lay - 1, lay - 1, dir, dir, 0, n_iter);

                auto gemm_grid_layer_fwd = (!rnn.copy_src_layer && lay == 0)
                        ? gemm_layer_fwd_src
                        : gemm_layer_fwd;

                CHECK(gemm_primitive(engine, ctx,
                        user_data.wei_layer(lay, dir, true), grid_layer,
                        *scratch.gates(), gemm_grid_layer_fwd));
            }

            for (dim_t i = 0; i < n_iter; i += rnn.iter_loop) {
                dim_t iter = (aprop == prop_kind::forward) ? i : n_iter - i - 1;
                CHECK((this->*cell_func)(engine, ctx, dir, lay, iter, user_data,
                        workspace, scratch, diff_bias, scales, tm_scales));
            }

            if (aprop == prop_kind::backward && rnn.merge_gemm_layer) {
                auto grid_layer = (!rnn.copy_src_layer && lay == 0)
                        ? user_data.src_layer(dir, 0)
                        : workspace.states(lay - 1, dir, 0);

                auto gemm_diff_wei_grid_layer
                        = (!rnn.copy_src_layer && lay == 0)
                        ? gemm_diff_wei_layer_src
                        : gemm_diff_wei_layer;

                // TODO: Fix sub-buffer size
                auto diff_states
                        = scratch.diff_states(lay, dir, rnn.n_states, 0);

                CHECK(gemm_primitive(engine, ctx,
                        user_data.wei_layer(lay, dir, true),
                        *scratch.diff_gates(), diff_states, gemm_layer_bwd));
                CHECK(gemm_primitive(engine, ctx, *scratch.diff_gates(),
                        grid_layer, user_data.diff_wei_layer(lay, dir, true),
                        gemm_diff_wei_grid_layer));
            }

            if (aprop == prop_kind::backward && rnn.merge_gemm_iter) {
                CHECK(gemm_primitive(engine, ctx, *scratch.diff_gates(),
                        grid_iter, user_data.diff_wei_iter(lay, dir, true),
                        gemm_diff_wei_iter));
            }
        }
    }
    return status::success;
}
//********* GRID computations strategy: utility functions **********//

template <prop_kind_t aprop>
status_t simple_rnn_common_t<aprop>::bias_prepare(const exec_ctx_t &ctx,
        compute::compute_stream_t *compute_stream, dim_t n_layer, dim_t n_dir,
        dim_t n_bias, dim_t n_gates, dim_t dhc, const memory_storage_t &ws_bias,
        const memory_storage_t &scales, const memory_storage_t &wei_layer,
        const memory_storage_t &wei_iter, const memory_storage_t &bias) const {

    float data_shift = pd()->attr()->rnn_data_qparams_.shift_;
    float data_scale = pd()->attr()->rnn_data_qparams_.scale_;

    compute::kernel_arg_list_t arg_list;
    arg_list.append(ws_bias);
    arg_list.append(scales);
    arg_list.append(wei_layer);
    arg_list.append(wei_iter);
    arg_list.append(bias);
    arg_list.append(into<int32_t>(dhc));
    arg_list.append(into<int32_t>(n_layer));
    arg_list.append(into<int32_t>(n_dir));
    arg_list.append(data_shift);
    arg_list.append(data_scale);

    arg_list.append(into<int32_t>(pd()->off.weights_layer_comp_off));
    arg_list.append(into<int32_t>(pd()->off.weights_iter_comp_off));
    arg_list.append(pd()->off.bias);

    return parallel_for(ctx,
            compute::nd_range_t({into<size_t>(dhc), into<size_t>(n_bias),
                    into<size_t>(n_layer * n_dir)}),
            kernels_[kernel_id::bias_prepare], arg_list);
}

template <prop_kind_t aprop>
status_t simple_rnn_common_t<aprop>::copy_init_layer(const exec_ctx_t &ctx,
        compute::compute_stream_t *compute_stream, bool lr, bool rl,
        dim_t batch, dim_t dhc, dim_t slc, dim_t n_iter, dim_t n_layer,
        dim_t n_dir, dim_t n_states, dim_t states_ws_ld,
        dim_t scratch_diff_states_ld, const memory_storage_t &ws_states,
        const memory_storage_t *scratch_diff_states,
        const memory_storage_t &input,
        const memory_storage_t &diff_dst_layer) const {

    int32_t unused_ld = 0;

    if (aprop == prop_kind::forward) {
        compute::kernel_arg_list_t arg_list;
        arg_list.append(ws_states);
        arg_list.append(input);
        arg_list.append(memory_storage_t::empty_storage());
        arg_list.append(into<int32_t>(lr));
        arg_list.append(into<int32_t>(rl));

        arg_list.append(into<int32_t>(batch));
        arg_list.append(into<int32_t>(dhc));
        arg_list.append(into<int32_t>(slc));
        arg_list.append(into<int32_t>(n_iter));
        arg_list.append(into<int32_t>(n_layer));
        arg_list.append(into<int32_t>(n_dir));
        arg_list.append(into<int32_t>(n_states));
        arg_list.append(into<int32_t>(states_ws_ld));
        arg_list.append(unused_ld);
        arg_list.append(pd()->off.src_layer);

        return parallel_for(ctx,
                compute::nd_range_t(get_nd_range({slc, batch, n_iter})),
                kernels_[kernel_id::copy_init_layer], arg_list);
    } else {
        compute::kernel_arg_list_t arg_list;
        arg_list.append(memory_storage_t::empty_storage());
        arg_list.append(diff_dst_layer);
        arg_list.append(*scratch_diff_states);
        arg_list.append(0);
        arg_list.append(0);

        arg_list.append(into<int32_t>(batch));
        arg_list.append(into<int32_t>(dhc));
        arg_list.append(into<int32_t>(slc));
        arg_list.append(into<int32_t>(n_iter));
        arg_list.append(into<int32_t>(n_layer));
        arg_list.append(into<int32_t>(n_dir));
        arg_list.append(into<int32_t>(n_states));
        arg_list.append(unused_ld);
        arg_list.append(into<int32_t>(scratch_diff_states_ld));
        arg_list.append(pd()->off.diff_dst_layer);

        return parallel_for(ctx,
                compute::nd_range_t(get_nd_range({dhc, batch, n_iter})),
                kernels_[kernel_id::copy_init_layer], arg_list);
    }
}

template <prop_kind_t aprop>
status_t simple_rnn_common_t<aprop>::copy_init_iter(const exec_ctx_t &ctx,
        compute::compute_stream_t *compute_stream, dim_t batch, dim_t dhc,
        dim_t sic, dim_t n_iter, dim_t n_layer, dim_t n_dir, dim_t n_states,
        dim_t states_ws_ld, dim_t scratch_diff_states_ld,
        const rnn_utils::workspace_t &ws,
        const memory_storage_t *scratch_diff_states,
        const memory_storage_t &firstit_states,
        const memory_storage_t &firstit_c_states,
        const memory_storage_t &diff_dst_iter,
        const memory_storage_t &diff_dst_iter_c, const float shift,
        const float scale, const bool quantize) const {

    int32_t unused_ld = 0;
    if (aprop == prop_kind::forward) {
        dim_t max_d = std::max(dhc, sic);
        compute::kernel_arg_list_t arg_list;
        arg_list.append(ws.states());
        arg_list.append(ws.c_states());
        arg_list.append(firstit_states);
        arg_list.append(firstit_c_states);
        arg_list.append(memory_storage_t::empty_storage());

        arg_list.append(into<int32_t>(batch));
        arg_list.append(into<int32_t>(dhc));
        arg_list.append(into<int32_t>(sic));
        arg_list.append(into<int32_t>(n_iter));
        arg_list.append(into<int32_t>(n_layer));
        arg_list.append(into<int32_t>(n_dir));
        arg_list.append(into<int32_t>(n_states));
        arg_list.append(into<int32_t>(states_ws_ld));

        arg_list.append(pd()->off.src_iter);
        if (pd()->ocl_conf.with_src_iter_c)
            arg_list.append(pd()->off.src_iter_c);

        arg_list.append(shift);
        arg_list.append(scale);
        arg_list.append(into<int32_t>(quantize));
        arg_list.append(unused_ld);
        return parallel_for(ctx,
                compute::nd_range_t({into<size_t>(max_d), into<size_t>(batch),
                        into<size_t>(n_layer * n_dir)}),
                kernels_[kernel_id::copy_init_iter], arg_list);
    } else {
        compute::kernel_arg_list_t arg_list;
        arg_list.append(memory_storage_t::empty_storage());
        arg_list.append(memory_storage_t::empty_storage());
        arg_list.append(diff_dst_iter);
        arg_list.append(diff_dst_iter_c);
        arg_list.append(*scratch_diff_states);

        arg_list.append(into<int32_t>(batch));
        arg_list.append(into<int32_t>(dhc));
        arg_list.append(into<int32_t>(sic));
        arg_list.append(into<int32_t>(n_iter));
        arg_list.append(into<int32_t>(n_layer));
        arg_list.append(into<int32_t>(n_dir));
        arg_list.append(into<int32_t>(n_states));
        arg_list.append(unused_ld);
        arg_list.append(pd()->off.diff_dst_iter);
        if (pd()->ocl_conf.with_dst_iter_c)
            arg_list.append(pd()->off.diff_dst_iter_c);
        arg_list.append(into<int32_t>(scratch_diff_states_ld));

        return parallel_for(ctx,
                compute::nd_range_t({into<size_t>(dhc), into<size_t>(batch),
                        into<size_t>(n_layer * n_dir)}),
                kernels_[kernel_id::copy_init_iter], arg_list);
    }
}

template <prop_kind_t aprop>
status_t simple_rnn_common_t<aprop>::copy_res_layer(const exec_ctx_t &ctx,
        compute::compute_stream_t *compute_stream, bool lr, bool rl,
        dim_t batch, dim_t dhc, dim_t slc, dim_t n_iter, dim_t n_layer,
        dim_t n_dir, dim_t n_states, dim_t states_ws_ld,
        dim_t scratch_diff_states_ld,
        const memory_storage_t *scratch_diff_states,
        const memory_storage_t &dst_last_layer,
        const memory_storage_t &diff_src_layer,
        const memory_storage_t &ws_states, const float shift, const float scale,
        const bool dequantize) const {

    int32_t unused_ld = 0;
    if (aprop == prop_kind::forward) {
        compute::kernel_arg_list_t arg_list;
        arg_list.append(ws_states);
        arg_list.append(dst_last_layer);
        arg_list.append(memory_storage_t::empty_storage());
        arg_list.append(into<int32_t>(lr));
        arg_list.append(into<int32_t>(rl));

        arg_list.append(into<int32_t>(batch));
        arg_list.append(into<int32_t>(dhc));
        arg_list.append(into<int32_t>(slc));
        arg_list.append(into<int32_t>(n_iter));
        arg_list.append(into<int32_t>(n_layer));
        arg_list.append(into<int32_t>(n_dir));
        arg_list.append(into<int32_t>(n_states));
        arg_list.append(into<int32_t>(states_ws_ld));
        arg_list.append(unused_ld);

        arg_list.append(pd()->off.dst_layer);

        arg_list.append(shift);
        arg_list.append(scale);
        arg_list.append(into<int32_t>(dequantize));
        return parallel_for(ctx, get_nd_range({dhc, batch, n_iter}),
                kernels_[kernel_id::copy_res_layer], arg_list);
    } else {
        compute::kernel_arg_list_t arg_list;
        arg_list.append(memory_storage_t::empty_storage());
        arg_list.append(diff_src_layer);
        arg_list.append(*scratch_diff_states);
        arg_list.append(into<int32_t>(lr));
        arg_list.append(into<int32_t>(rl));

        arg_list.append(into<int32_t>(batch));
        arg_list.append(into<int32_t>(dhc));
        arg_list.append(into<int32_t>(slc));
        arg_list.append(into<int32_t>(n_iter));
        arg_list.append(into<int32_t>(n_layer));
        arg_list.append(into<int32_t>(n_dir));
        arg_list.append(into<int32_t>(n_states));
        arg_list.append(unused_ld);
        arg_list.append(into<int32_t>(scratch_diff_states_ld));
        arg_list.append(pd()->off.diff_src_layer);

        return parallel_for(ctx, get_nd_range({slc, batch, n_iter}),
                kernels_[kernel_id::copy_res_layer], arg_list);
    }
}

template <prop_kind_t aprop>
status_t simple_rnn_common_t<aprop>::copy_res_iter(const exec_ctx_t &ctx,
        compute::compute_stream_t *compute_stream, dim_t batch, dim_t dhc,
        dim_t sic, dim_t n_iter, dim_t n_layer, dim_t n_dir, dim_t n_states,
        dim_t states_ws_ld, dim_t scratch_diff_states_ld,
        const memory_storage_t *scratch_diff_states,
        const memory_storage_t &dst_last_iter,
        const memory_storage_t &dst_last_iter_c,
        const memory_storage_t &diff_src_iter,
        const memory_storage_t &diff_src_iter_c,
        const rnn_utils::workspace_t &ws, const float shift, const float scale,
        const bool dequantize) const {

    int32_t unused_ld = 0;
    if (aprop == prop_kind::forward) {
        compute::kernel_arg_list_t arg_list;
        arg_list.append(ws.states());
        arg_list.append(ws.c_states());
        arg_list.append(dst_last_iter);
        arg_list.append(dst_last_iter_c);
        arg_list.append(memory_storage_t::empty_storage());

        arg_list.append(into<int32_t>(batch));
        arg_list.append(into<int32_t>(dhc));
        arg_list.append(into<int32_t>(sic));
        arg_list.append(into<int32_t>(n_iter));
        arg_list.append(into<int32_t>(n_layer));
        arg_list.append(into<int32_t>(n_dir));
        arg_list.append(into<int32_t>(n_states));
        arg_list.append(into<int32_t>(states_ws_ld));
        arg_list.append(unused_ld);

        arg_list.append(pd()->off.dst_iter);
        if (pd()->ocl_conf.with_dst_iter_c)
            arg_list.append(pd()->off.dst_iter_c);

        arg_list.append(shift);
        arg_list.append(scale);
        arg_list.append(into<int32_t>(dequantize));
        return parallel_for(ctx,
                compute::nd_range_t({into<size_t>(dhc), into<size_t>(batch),
                        into<size_t>(n_layer * n_dir)}),
                kernels_[kernel_id::copy_res_iter], arg_list);
    } else {
        dim_t max_d = std::max(dhc, sic);
        compute::kernel_arg_list_t arg_list;
        arg_list.append(memory_storage_t::empty_storage());
        arg_list.append(memory_storage_t::empty_storage());
        arg_list.append(diff_src_iter);
        arg_list.append(diff_src_iter_c);
        arg_list.append(*scratch_diff_states);

        arg_list.append(into<int32_t>(batch));
        arg_list.append(into<int32_t>(dhc));
        arg_list.append(into<int32_t>(sic));
        arg_list.append(into<int32_t>(n_iter));
        arg_list.append(into<int32_t>(n_layer));
        arg_list.append(into<int32_t>(n_dir));
        arg_list.append(into<int32_t>(n_states));
        arg_list.append(unused_ld);
        arg_list.append(into<int32_t>(scratch_diff_states_ld));

        arg_list.append(pd()->off.diff_src_iter);
        if (pd()->ocl_conf.with_src_iter_c)
            arg_list.append(pd()->off.diff_src_iter_c);

        return parallel_for(ctx,
                compute::nd_range_t({into<size_t>(max_d), into<size_t>(batch),
                        into<size_t>(n_layer * n_dir)}),
                kernels_[kernel_id::copy_res_iter], arg_list);
    }
}

//********************* Execution function *********************//

template <prop_kind_t aprop>
status_t simple_rnn_common_t<aprop>::execute_(const exec_ctx_t &ctx) const {

    impl::engine_t *engine = ctx.stream()->engine();
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

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

    bool is_fwd = rnn.is_fwd;

    auto &src_layer_native_ = CTX_IN_STORAGE(DNNL_ARG_SRC_LAYER);
    auto &src_iter_native_ = CTX_IN_STORAGE(DNNL_ARG_SRC_ITER);
    auto &src_c_iter_native_ = CTX_IN_STORAGE(DNNL_ARG_SRC_ITER_C);
    auto &wei_layer_native_ = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_LAYER);
    auto &wei_iter_native_ = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_ITER);
    auto &bias_native_ = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    auto &dst_last_layer_native_ = is_fwd ? CTX_OUT_STORAGE(DNNL_ARG_DST_LAYER)
                                          : CTX_IN_STORAGE(DNNL_ARG_DST_LAYER);
    auto &dst_last_iter_native_ = is_fwd ? CTX_OUT_STORAGE(DNNL_ARG_DST_ITER)
                                         : CTX_IN_STORAGE(DNNL_ARG_DST_ITER);
    auto &dst_last_iter_c_native_ = is_fwd
            ? CTX_OUT_STORAGE(DNNL_ARG_DST_ITER_C)
            : CTX_IN_STORAGE(DNNL_ARG_DST_ITER_C);

    auto &diff_dst_layer_native_ = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST_LAYER);
    auto &diff_dst_iter_native_ = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST_ITER);
    auto &diff_dst_iter_c_native_ = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST_ITER_C);

    auto scratch_workspace
            = ctx.get_scratchpad_grantor().get_memory_storage(key_rnn_space);
    auto &workspace_ = rnn.is_training ? is_fwd
                    ? CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE)
                    : CTX_IN_STORAGE(DNNL_ARG_WORKSPACE)
                                       : *scratch_workspace;
    const auto &workspace = rnn_utils::workspace_t(workspace_, rnn);

    const auto scratch
            = rnn_utils::scratch_t(rnn, ctx.get_scratchpad_grantor());

    auto &diff_src_layer_native_ = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC_LAYER);
    auto &diff_src_iter_native_ = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC_ITER);
    auto &diff_src_iter_c_native_ = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC_ITER_C);

    auto &diff_weights_layer_native_
            = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS_LAYER);
    auto &diff_weights_iter_native_
            = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS_ITER);
    auto &diff_bias_native_ = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);

    const rnn_utils::user_data_t user_data(src_layer_native_, wei_layer_native_,
            wei_iter_native_, bias_native_, diff_src_layer_native_,
            diff_dst_layer_native_, diff_weights_layer_native_,
            diff_weights_iter_native_, rnn, pd()->off);

    // TODO: implement without copies
    bool is_lr = !one_of(rnn.exec_dir, r2l, r2l);
    bool is_rl = !one_of(rnn.exec_dir, l2r, l2r);

    const memory_storage_t *scales_buf = nullptr;
    if (pd()->rnn_conf.is_int8 && pd()->rnn_conf.copy_bias) {
        scales_buf = &CTX_GPU_RES_STORAGE(SCALES_);
    }

    // bias prepare if needed
    if (rnn.copy_bias) {
        CHECK(bias_prepare(ctx, compute_stream, n_layer, n_dir, n_bias, n_gates,
                dhc, workspace.bias(), *scales_buf, wei_layer_native_,
                wei_iter_native_, user_data.bias()));
    }

    float shift = (pd()->attr()->rnn_data_qparams_.shift_);
    float scale = (pd()->attr()->rnn_data_qparams_.scale_);

    if ((rnn.is_fwd && rnn.copy_src_layer)
            || (!rnn.is_fwd && rnn.copy_diff_dst_layer)) {
        CHECK(copy_init_layer(ctx, compute_stream, is_lr, is_rl, batch, dhc,
                slc, n_iter, n_layer, n_dir, n_states, rnn.states_ws_ld,
                rnn.scratch_diff_states_ld, workspace.states(),
                scratch.diff_states(), src_layer_native_,
                diff_dst_layer_native_));
    }
    const bool quantize = pd()->with_src_iter()
            && pd()->src_md(1)->data_type == data_type::f32 && rnn.is_int8;
    CHECK(copy_init_iter(ctx, compute_stream, batch, dhc, sic, n_iter, n_layer,
            n_dir, n_states, rnn.states_ws_ld, rnn.scratch_diff_states_ld,
            workspace, scratch.diff_states(), src_iter_native_,
            src_c_iter_native_, diff_dst_iter_native_, diff_dst_iter_c_native_,
            shift, scale, quantize));

    const memory_storage_t *tm_scales_buf = nullptr;
    if (pd()->rnn_conf.is_testmode && pd_->attr()->rnn_tparams_.scales_) {
        tm_scales_buf = &CTX_GPU_RES_STORAGE(TM_SCALES_);
    }

    // run the execution on the grid
    CHECK((this->*grid_computation)(engine, ctx, user_data, workspace, scratch,
            diff_bias_native_, scales_buf, tm_scales_buf));

    // Finally we copy the results to the result buffers

    if (rnn.is_fwd || rnn.copy_diff_src_layer) {
        const bool dequantize_l
                = pd()->dst_md(0)->data_type == data_type::f32 && rnn.is_int8;
        CHECK(copy_res_layer(ctx, compute_stream, is_lr, is_rl, batch, dhc, slc,
                n_iter, n_layer, n_dir, n_states, rnn.states_ws_ld,
                rnn.scratch_diff_states_ld, scratch.diff_states(),
                dst_last_layer_native_, diff_src_layer_native_,
                workspace.states(), shift, scale, dequantize_l));
    }
    const bool dequantize_i = pd()->with_dst_iter()
            && pd()->dst_md(1)->data_type == data_type::f32 && rnn.is_int8;
    CHECK(copy_res_iter(ctx, compute_stream, batch, dhc, sic, n_iter, n_layer,
            n_dir, n_states, rnn.states_ws_ld, rnn.scratch_diff_states_ld,
            scratch.diff_states(), dst_last_iter_native_,
            dst_last_iter_c_native_, diff_src_iter_native_,
            diff_src_iter_c_native_, workspace, shift, scale, dequantize_i));

    return status::success;
};
// Fix for MSVS warning C4661.
template <>
cell_execution_sig(simple_rnn_fwd_t::cell_execution);
template <>
cell_execution_sig(simple_rnn_bwd_t::cell_execution);
template <>
cell_execution_sig(simple_rnn_fwd_t::cell_execution_gru);
template <>
cell_execution_sig(simple_rnn_bwd_t::cell_execution_gru);
template <>
cell_execution_sig(simple_rnn_fwd_t::cell_execution_gru_lbr);
template <>
cell_execution_sig(simple_rnn_bwd_t::cell_execution_gru_lbr);
template <>
elemwise_sig(simple_rnn_fwd_t::rnn_elemwise);
template <>
elemwise_sig(simple_rnn_bwd_t::rnn_elemwise);
template <>
elemwise_sig(simple_rnn_fwd_t::lstm_elemwise);
template <>
elemwise_sig(simple_rnn_bwd_t::lstm_elemwise);
template <>
elemwise_sig(simple_rnn_fwd_t::lstm_elemwise_u8s8);
template <>
elemwise_sig(simple_rnn_bwd_t::lstm_elemwise_u8s8);
template <>
elemwise_sig_gru_lbr(simple_rnn_fwd_t::gru_lbr_elemwise);
template <>
elemwise_sig_gru_lbr(simple_rnn_bwd_t::gru_lbr_elemwise);
template <>
elemwise_sig_gru(simple_rnn_fwd_t::gru_elemwise);
template <>
elemwise_sig_gru(simple_rnn_bwd_t::gru_elemwise);

template struct simple_rnn_common_t<prop_kind::forward>;
template struct simple_rnn_common_t<prop_kind::backward>;

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
