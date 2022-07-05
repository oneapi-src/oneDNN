/*******************************************************************************
 * * Copyright 2019-2022 Intel Corporation
 * *
 * * Licensed under the Apache License, Version 2.0 (the "License");
 * * you may not use this file except in compliance with the License.
 * * You may obtain a copy of the License at
 * *
 * *     http://www.apache.org/licenses/LICENSE-2.0
 * *
 * * Unless required by applicable law or agreed to in writing, software
 * * distributed under the License is distributed on an "AS IS" BASIS,
 * * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * * See the License for the specific language governing permissions and
 * * limitations under the License.
 * *******************************************************************************/

#include "oneapi/dnnl/dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_graph_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/norm.hpp"

#include "bnorm/bnorm.hpp"
#include "bnorm/graph_bnorm.hpp"

#include <string>
#include <vector>

namespace benchdnnext {
namespace bnorm {

static int check_known_skipped_case_graph(
        const ::bnorm::prb_t *prb, res_t *res) noexcept {

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, ::bnorm::init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    check_known_skipped_case_graph_common(
            {prb->dt}, normalize_tag(prb->tag, prb->ndims), prb->dir, res);
    if (res->state == SKIPPED) return OK;

    for (const auto &po : prb->attr.post_ops.entry) {
        if (po.kind == attr_t::post_ops_t::RELU && prb->dir & FLAG_INF) {
            continue;
        } else {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return OK;
        }
    }

    check_graph_eltwise_post_ops(prb->attr, res);
    return OK;
}

fill_status_t append_graph_with_block(const ::bnorm::prb_t *prb) {
    using graph_dt = dnnl::graph::logical_tensor::data_type;
    using graph_lt = dnnl::graph::logical_tensor::layout_type;
    graph_t &graph = graph_t::get();

    const auto connect_to_previous_block = graph.has_blocks();

    // handle main op
    const auto op_id = graph.generate_id_for(entry_kind::BNORM);
    const auto src_lt_kind
            = prb->dir == FLAG_FWD ? lt_kind::SRC : lt_kind::DIFF_SRC;
    const auto dst_lt_kind
            = prb->dir & FLAG_FWD ? lt_kind::DST : lt_kind::DIFF_DST;
    const auto src_id = connect_to_previous_block
            ? graph.get_last_block_out_id()
            : graph.generate_id_for(op_id, src_lt_kind);
    const auto sc_id = graph.generate_id_for(op_id, lt_kind::SC);
    const auto mean_id = graph.generate_id_for(op_id, lt_kind::MEAN);
    const auto var_id = graph.generate_id_for(op_id, lt_kind::VAR);
    const auto dst_id = graph.generate_id_for(op_id, dst_lt_kind);

    const auto common_dt = convert_dt(prb->dt);
    dims_t base_dims = prb->data_dims();
    dims_t stat_dims = {prb->ic};

    graph.create_lt(src_id, common_dt, base_dims, graph_lt::strided);
    graph.create_lt(dst_id, common_dt, base_dims, graph_lt::strided);
    graph.create_lt(sc_id, graph_dt::f32, stat_dims, graph_lt::strided);
    graph.create_lt(mean_id, graph_dt::f32, stat_dims, graph_lt::strided);
    graph.create_lt(var_id, graph_dt::f32, stat_dims, graph_lt::strided);

    std::vector<size_t> src_ids {};
    std::vector<size_t> dst_ids {};
    dnnl::graph::op::kind bnorm_kind;
    if (prb->dir & FLAG_FWD) {
        const auto sh_id = graph.generate_id_for(op_id, lt_kind::SH);
        graph.create_lt(sh_id, graph_dt::f32, stat_dims, graph_lt::strided);
        src_ids = {src_id, sc_id, sh_id, mean_id, var_id};
        if (prb->dir & FLAG_INF) {
            dst_ids = {dst_id};
            bnorm_kind = dnnl::graph::op::kind::BatchNormInference;
        } else {
            const auto rmean_id
                    = graph.generate_id_for(op_id, lt_kind::RUN_MEAN);
            const auto bmean_id
                    = graph.generate_id_for(op_id, lt_kind::BATCH_MEAN);
            const auto rvar_id = graph.generate_id_for(op_id, lt_kind::RUN_VAR);
            const auto bvar_id
                    = graph.generate_id_for(op_id, lt_kind::BATCH_VAR);
            graph.create_lt(
                    rmean_id, graph_dt::f32, stat_dims, graph_lt::strided);
            graph.create_lt(
                    bmean_id, graph_dt::f32, stat_dims, graph_lt::strided);
            graph.create_lt(
                    rvar_id, graph_dt::f32, stat_dims, graph_lt::strided);
            graph.create_lt(
                    bvar_id, graph_dt::f32, stat_dims, graph_lt::strided);
            dst_ids = {dst_id, rmean_id, rvar_id, bmean_id, bvar_id};
            bnorm_kind = dnnl::graph::op::kind::BatchNormForwardTraining;
        }
    } else {
        const auto fwd_src_id = graph.generate_id_for(op_id, lt_kind::SRC);
        const auto d_sc_id = graph.generate_id_for(op_id, lt_kind::DIFF_SC);
        const auto d_sh_id = graph.generate_id_for(op_id, lt_kind::DIFF_SH);
        graph.create_lt(fwd_src_id, common_dt, base_dims, graph_lt::strided);
        graph.create_lt(d_sc_id, graph_dt::f32, stat_dims, graph_lt::strided);
        graph.create_lt(d_sh_id, graph_dt::f32, stat_dims, graph_lt::strided);
        src_ids = {fwd_src_id, dst_id, sc_id, mean_id, var_id};
        dst_ids = {src_id, d_sc_id, d_sh_id};
        bnorm_kind = dnnl::graph::op::kind::BatchNormTrainingBackprop;
    }

    dnnl::graph::op bnorm_op(op_id, bnorm_kind, graph.stringify_id(op_id));
    bnorm_op.set_attr("epsilon", prb->eps);
    bnorm_op.set_attr("data_format", std::string("NCX"));

    graph.append(op_id, bnorm_op, src_ids, dst_ids);

    fill_status_t status;
    // handle post ops
    for (const auto &entry : prb->attr.post_ops.entry) {
        if (entry.is_eltwise_kind()) {
            status = append_graph_with_eltwise(entry);
            BENCHDNNEXT_VERIFY(status);
        }
    }

    graph.close_block();

    return fill_status::DONE;
}

int doit(const ::bnorm::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const auto status = append_graph_with_block(prb);
    if (status != fill_status::DONE
            && status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        cleanup();
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto &graph = graph_t::get();

    // Filter partitions
    const auto partitions = graph.get_partitions();
    if (partitions.empty() || partitions.size() > 1) {
        cleanup();
        return res->state = FAILED, FAIL;
    }

    const auto par = partitions[0];
    if (!par.is_supported()) {
        cleanup();
        return res->state = UNIMPLEMENTED, FAIL;
    }

    const bool is_fwd = prb->dir & FLAG_FWD;
    const bool use_ss = prb->use_ss();
    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::bnorm::init_pd, prb, res, par, ins, outs);

    static const engine_t cpu_engine(dnnl_cpu);

    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto shift_fp = make_dnn_mem(
            is_fwd ? ins[2] : outs[2], dt::f32, use_sh ? tag::x : tag::axb);
    auto mean_fp = make_dnn_mem(ins[3], dt::f32, tag::abx);
    auto var_fp = make_dnn_mem(ins[4], dt::f32, tag::abx);
    dnn_mem_t &dst_fp = src_fp; // in-place reference
    auto src_hat_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto ws_fp = make_dnn_mem(ins[0], dt::u8, tag::abx);

    const auto placeholder_dst_dt = make_dnn_mem(outs[0], tag::abx);
    auto src_dt = make_dnn_mem(ins[0], tag::abx);
    auto shift_dt = make_dnn_mem(
            is_fwd ? ins[2] : outs[2], use_sh ? tag::x : tag::axb);
    auto mean_dt = make_dnn_mem(ins[3], tag::abx);
    auto var_dt = make_dnn_mem(ins[4], tag::abx);
    const dnn_mem_t &dst_dt = prb->inplace ? src_dt : placeholder_dst_dt;

    dnn_mem_t scale_fp, scale_dt, d_shift_dt, d_scale_dt;
    if (use_sc || use_sh) {
        scale_fp = make_dnn_mem(is_fwd ? ins[1] : ins[2], dt::f32, tag::abx);
        scale_dt = make_dnn_mem(is_fwd ? ins[1] : ins[2], dt::f32, tag::abx);
    } else {
        // scale and shift are combined in a single 2D tensor of shape 2xC
        // this logical_tensor is used to indicate the memory size for scale
        dnnl::graph::logical_tensor lt {0, dt::f32, {2, prb->ic},
                dnnl::graph::logical_tensor::layout_type::strided};
        scale_fp = make_dnn_mem(lt, dt::f32, tag::abx);
        scale_dt = make_dnn_mem(lt, dt::f32, tag::abx);
    }

    dnn_mem_t d_dst_dt, placeholder_d_src_dt;
    dnn_mem_t r_mean_dt, r_var_dt, b_mean_dt, b_var_dt;

    if (::bnorm::prepare_fwd(prb, src_fp, mean_fp, var_fp, scale_fp, shift_fp)
            != OK) {
        cleanup();
        return res->state = MISTRUSTED, OK;
    }
    /*  When dnnl_use_scaleshift is used, benchdnn populates data
        to the same memory for scale and shift and dnnlgraph expects
        the data in scale and shift. Hence this explicit copy. */
    if (!(use_sc || use_sh)) {
        for (int64_t i = 0; i < prb->ic; i++) {
            shift_fp.set_elem(i, scale_fp.get_elem(prb->ic + i));
        }
    }
    SAFE(src_dt.reorder(src_fp), WARN);
    SAFE(scale_dt.reorder(scale_fp), WARN);
    SAFE(shift_dt.reorder(shift_fp), WARN);
    SAFE(mean_dt.reorder(mean_fp), WARN);
    SAFE(var_dt.reorder(var_fp), WARN);

    std::vector<dnnl::graph::tensor> tensors_in, tensors_out;
    const dnnl::graph::engine &eng = get_test_engine();
    args_t args, ref_args;

    if (prb->dir & FLAG_FWD) {
        tensors_in.emplace_back(ins[0], eng, static_cast<void *>(src_dt));
        tensors_in.emplace_back(ins[1], eng, static_cast<void *>(scale_dt));
        tensors_in.emplace_back(ins[2], eng, static_cast<void *>(shift_dt));
        tensors_in.emplace_back(ins[3], eng, static_cast<void *>(mean_dt));
        tensors_in.emplace_back(ins[4], eng, static_cast<void *>(var_dt));
        tensors_out.emplace_back(outs[0], eng, static_cast<void *>(dst_dt));
        if (!(prb->dir & FLAG_INF)) {
            r_mean_dt = make_dnn_mem(outs[1], tag::abx);
            r_var_dt = make_dnn_mem(outs[2], tag::abx);
            b_mean_dt = make_dnn_mem(outs[3], tag::abx);
            b_var_dt = make_dnn_mem(outs[4], tag::abx);
            tensors_out.emplace_back(
                    outs[1], eng, static_cast<void *>(r_mean_dt));
            tensors_out.emplace_back(
                    outs[2], eng, static_cast<void *>(r_var_dt));
            tensors_out.emplace_back(
                    outs[3], eng, static_cast<void *>(b_mean_dt));
            tensors_out.emplace_back(
                    outs[4], eng, static_cast<void *>(b_var_dt));
        }
        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_MEAN, mean_fp);
            ref_args.set(DNNL_ARG_VARIANCE, var_fp);
            ref_args.set(
                    use_sc ? DNNL_ARG_SCALE : DNNL_ARG_SCALE_SHIFT, scale_fp);
            ref_args.set(DNNL_ARG_SHIFT, shift_fp);
            ref_args.set(DNNL_ARG_WORKSPACE, ws_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(DNNL_ARG_DST_1, src_hat_fp); // Reference aux arg.

            SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);
            args.set(DNNL_ARG_SRC, src_dt);
            args.set(DNNL_ARG_MEAN, mean_dt);
            args.set(DNNL_ARG_VARIANCE, var_dt);
            std::vector<data_kind_t> kinds {DST};
            if (!(prb->flags & ::bnorm::GLOB_STATS) && !(prb->dir & FLAG_INF)) {
                kinds.push_back(MEAN);
                kinds.push_back(VAR);
            }

            check_correctness(
                    prb, kinds, args, ref_args, ::bnorm::setup_cmp, res);

            cleanup();

            return OK;
        }
    } else {
        auto d_dst_fp = make_dnn_mem(ins[1], dt::f32, tag::abx);
        dnn_mem_t &d_src_fp = d_dst_fp; // in-place in ref code

        d_dst_dt = make_dnn_mem(ins[1], tag::abx);
        if (!prb->inplace) {
            placeholder_d_src_dt = make_dnn_mem(outs[0], tag::abx);
        }
        dnn_mem_t &d_src_dt = prb->inplace ? d_dst_dt : placeholder_d_src_dt;

        SAFE(::bnorm::prepare_bwd(prb, d_dst_dt, d_dst_fp), WARN);

        dnn_mem_t d_scale_fp;
        if (use_sc || use_sh) {
            d_scale_fp = make_dnn_mem(outs[1], dt::f32, tag::abx);
            d_scale_dt = make_dnn_mem(outs[1], dt::f32, tag::abx);
        } else {
            // scale and shift are combined in a single 2D tensor of shape 2xC
            // this logical_tensor is used to indicate the memory size for scale
            dnnl::graph::logical_tensor lt {0, dt::f32, {2, prb->ic},
                    dnnl::graph::logical_tensor::layout_type::strided};
            d_scale_fp = make_dnn_mem(lt, dt::f32, tag::abx);
            d_scale_dt = make_dnn_mem(lt, dt::f32, tag::abx);
        }
        auto d_shift_fp
                = make_dnn_mem(outs[2], dt::f32, use_sh ? tag::x : tag::axb);
        d_shift_dt = make_dnn_mem(outs[2], use_sh ? tag::x : tag::axb);

        tensors_in.emplace_back(ins[0], eng, static_cast<void *>(src_dt));
        tensors_in.emplace_back(ins[1], eng, static_cast<void *>(d_dst_dt));
        tensors_in.emplace_back(ins[2], eng, static_cast<void *>(scale_dt));
        tensors_in.emplace_back(ins[3], eng, static_cast<void *>(mean_dt));
        tensors_in.emplace_back(ins[4], eng, static_cast<void *>(var_dt));
        tensors_out.emplace_back(outs[0], eng, static_cast<void *>(d_src_dt));
        tensors_out.emplace_back(outs[1], eng, static_cast<void *>(d_scale_dt));
        tensors_out.emplace_back(outs[2], eng, static_cast<void *>(d_shift_dt));

        if (is_bench_mode(CORR)) {
            SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_MEAN, mean_fp);
            ref_args.set(DNNL_ARG_VARIANCE, var_fp);
            ref_args.set(
                    use_sc ? DNNL_ARG_SCALE : DNNL_ARG_SCALE_SHIFT, scale_fp);
            ref_args.set(DNNL_ARG_SHIFT, shift_fp);
            ref_args.set(DNNL_ARG_WORKSPACE, ws_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(DNNL_ARG_DST_1, src_hat_fp); // Reference aux arg.
            ref_args.set(DNNL_ARG_DIFF_DST, d_dst_fp);
            ref_args.set(DNNL_ARG_DIFF_SRC, d_src_fp);
            ref_args.set(
                    use_sc ? DNNL_ARG_DIFF_SCALE : DNNL_ARG_DIFF_SCALE_SHIFT,
                    d_scale_fp);
            ref_args.set(DNNL_ARG_DIFF_SHIFT, d_shift_fp);

            args.set(DNNL_ARG_SRC, src_dt);
            args.set(DNNL_ARG_MEAN, mean_dt);
            args.set(DNNL_ARG_VARIANCE, var_dt);
            args.set(DNNL_ARG_DIFF_DST, d_dst_dt);
            args.set(use_sc ? DNNL_ARG_SCALE : DNNL_ARG_SCALE_SHIFT, scale_dt);
            args.set(DNNL_ARG_SHIFT, shift_dt);
            args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
            args.set(use_sc ? DNNL_ARG_DIFF_SCALE : DNNL_ARG_DIFF_SCALE_SHIFT,
                    d_scale_dt);
            args.set(DNNL_ARG_DIFF_SHIFT, d_shift_dt);

            std::vector<data_kind_t> kinds {SRC};
            if ((use_ss || use_sc) && (prb->dir & FLAG_WEI)) {
                kinds.push_back(use_sc ? SC : SS);
            }
            if (use_sh && (prb->dir & FLAG_WEI)) kinds.push_back(SH);
            check_correctness(
                    prb, kinds, args, ref_args, ::bnorm::setup_cmp, res);

            cleanup();

            return OK;
        }
    }

    SAFE(measure_perf(
                 res->timer_map.perf_timer(), cp, tensors_in, tensors_out, res),
            WARN);

    cleanup();

    return OK;
}
} // namespace bnorm
} // namespace benchdnnext
