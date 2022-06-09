/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "prelu/graph_prelu.hpp"
#include "utils/compare.hpp"

namespace benchdnnext {
namespace prelu {

static void check_known_skipped_case_graph(
        const ::prelu::prb_t *prb, res_t *res) noexcept {
    // TODO: to align with original benchdnn, we should consider moving
    // skip_unimplemented_prb call after compilation step
    skip_invalid_and_unimplemented_prb(prb, res);
}

static std::vector<dnnl::graph::logical_tensor::data_type> collect_data_types(
        const ::prelu::prb_t *prb) {
    return {convert_dt(prb->sdt[0]), convert_dt(prb->sdt[1])};
}

fill_status_t append_graph_with_block(const ::prelu::prb_t *prb) {
    graph_t &graph = graph_t::get();

    const auto orig_dts = collect_data_types(prb);
    const auto connect_to_previous_block = graph.has_blocks();

    // handle main op
    const auto op_id = graph.generate_id_for(entry_kind::PRELU);
    const auto src_lt_kind
            = prb->dir == FLAG_FWD ? lt_kind::SRC : lt_kind::DIFF_SRC;
    const auto wei_lt_kind
            = prb->dir == FLAG_FWD ? lt_kind::WEI : lt_kind::DIFF_WEI;
    const auto dst_lt_kind
            = prb->dir & FLAG_FWD ? lt_kind::DST : lt_kind::DIFF_DST;
    const auto src_id = connect_to_previous_block
            ? graph.get_last_block_out_id()
            : graph.generate_id_for(op_id, src_lt_kind);
    const auto wei_id = graph.generate_id_for(op_id, wei_lt_kind);
    const auto dst_id = graph.generate_id_for(op_id, dst_lt_kind);

    const auto data_dt = orig_dts[0];
    const auto wei_dt = orig_dts[1];

    const auto data_dims = prb->vdims[0];
    const auto wei_dims = prb->vdims[1];
    const auto data_tag = prb->stag[0];
    const auto wei_tag = prb->stag[1];

    graph.create_lt(src_id, data_dt, data_dims, data_tag);
    graph.create_lt(wei_id, wei_dt, wei_dims, wei_tag);
    graph.create_lt(dst_id, data_dt, data_dims, data_tag);

    std::vector<size_t> src_ids {};
    std::vector<size_t> dst_ids {};
    dnnl::graph::op::kind prelu_kind;
    if (prb->dir & FLAG_FWD) {
        src_ids = {src_id, wei_id};
        dst_ids = {dst_id};
        prelu_kind = dnnl::graph::op::kind::PReLU;
    } else { // bwd
        const auto fwd_src_id = graph.generate_id_for(op_id, lt_kind::SRC);
        const auto fwd_wei_id = graph.generate_id_for(op_id, lt_kind::WEI);
        graph.create_lt(fwd_src_id, data_dt, data_dims, data_tag);
        graph.create_lt(fwd_wei_id, wei_dt, wei_dims, wei_tag);
        src_ids = {fwd_src_id, fwd_wei_id, dst_id};
        dst_ids = {src_id, wei_id};
        prelu_kind = dnnl::graph::op::kind::PReLUBackprop;
    }

    dnnl::graph::op prelu_op(op_id, prelu_kind, graph.stringify_id(op_id));
    prelu_op.set_attr("data_format", std::string("NCX"));
    if (prb->dir & FLAG_FWD) prelu_op.set_attr("per_channel_broadcast", false);

    graph.append(op_id, prelu_op, src_ids, dst_ids);

    graph.close_block();

    return fill_status::DONE;
}

int doit(const ::prelu::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

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

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();
    auto cp = compile_partition(::prelu::init_pd, prb, res, par, ins, outs);

    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto src_dt = make_dnn_mem(ins[0], prb->stag[0]);
    auto slope_fp = make_dnn_mem(ins[1], dt::f32, tag::abx);
    auto slope_dt = make_dnn_mem(ins[1], prb->stag[1]);

    SAFE(::prelu::fill_data(SRC, src_dt, src_fp), WARN);
    SAFE(::prelu::fill_data(WEI, slope_dt, slope_fp), WARN);

    std::vector<dnnl::graph::tensor> tensors_in;
    std::vector<dnnl::graph::tensor> tensors_out;
    const dnnl::graph::engine &eng = get_test_engine();

    tensors_in.emplace_back(
            dnnl::graph::tensor(ins[0], eng, static_cast<void *>(src_dt)));
    tensors_in.emplace_back(
            dnnl::graph::tensor(ins[1], eng, static_cast<void *>(slope_dt)));

    if (prb->dir & FLAG_FWD) {
        auto dst_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);
        auto dst_dt = make_dnn_mem(outs[0], prb->stag[0]);

        tensors_out.emplace_back(
                dnnl::graph::tensor(outs[0], eng, static_cast<void *>(dst_dt)));

        SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

        if (is_bench_mode(CORR)) {
            args_t args;
            args.set(DNNL_ARG_DST, dst_dt);

            args_t ref_args;
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_WEIGHTS, slope_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);

            check_correctness(
                    prb, {DST}, args, ref_args, ::prelu::setup_cmp, res);
        }
    } else if (prb->dir & FLAG_BWD) {
        auto d_dst_fp = make_dnn_mem(ins[2], dt::f32, tag::abx);
        auto d_dst_dt = make_dnn_mem(ins[2], prb->stag[0]);
        auto d_src_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);
        auto d_src_dt = make_dnn_mem(outs[0], prb->stag[0]);
        auto d_slope_fp = make_dnn_mem(outs[1], dt::f32, tag::abx);
        auto d_slope_dt = make_dnn_mem(outs[1], prb->stag[1]);

        SAFE(::prelu::fill_data(DST, d_dst_dt, d_dst_fp), WARN);

        tensors_in.emplace_back(dnnl::graph::tensor(
                ins[2], eng, static_cast<void *>(d_dst_dt)));
        tensors_out.emplace_back(dnnl::graph::tensor(
                outs[0], eng, static_cast<void *>(d_src_dt)));
        tensors_out.emplace_back(dnnl::graph::tensor(
                outs[1], eng, static_cast<void *>(d_slope_dt)));

        SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

        if (is_bench_mode(CORR)) {
            args_t args;
            args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
            args.set(DNNL_ARG_DIFF_WEIGHTS, d_slope_dt);

            args_t ref_args;
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_WEIGHTS, slope_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, d_dst_fp);
            ref_args.set(DNNL_ARG_DIFF_SRC, d_src_fp);
            ref_args.set(DNNL_ARG_DIFF_WEIGHTS, d_slope_fp);

            check_correctness(
                    prb, {SRC, WEI}, args, ref_args, ::prelu::setup_cmp, res);
        }
    } else {
        SAFE(FAIL, CRIT);
    }

    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);

    cleanup();

    return OK;
}
} // namespace prelu
} // namespace benchdnnext
