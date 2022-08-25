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

#include "dnnl_graph_common.hpp"
#include "utils/compare.hpp"

#include "softmax/graph_softmax.hpp"
#include "softmax/softmax.hpp"

namespace benchdnnext {
namespace softmax {

static int check_known_skipped_case_graph(
        const ::softmax::prb_t *prb, res_t *res) noexcept {

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, ::softmax::init_pd, prb, res), WARN);

    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    check_known_skipped_case_graph_common(
            {prb->sdt}, normalize_tag(prb->stag, prb->ndims), prb->dir, res);

    return OK;
}

fill_status_t append_graph_with_block(const ::softmax::prb_t *prb) {
    graph_t &graph = graph_t::get();

    const auto connect_to_previous_block = graph.has_blocks();

    // handle main op
    const auto op_id = graph.generate_id_for(entry_kind::SOFTMAX);
    const auto src_lt_kind
            = prb->dir & FLAG_FWD ? lt_kind::SRC : lt_kind::DIFF_SRC;
    const auto dst_lt_kind
            = prb->dir & FLAG_FWD ? lt_kind::DST : lt_kind::DIFF_DST;
    const auto src_id = connect_to_previous_block
            ? graph.get_last_block_out_id()
            : graph.generate_id_for(op_id, src_lt_kind);
    const auto dst_id = graph.generate_id_for(op_id, dst_lt_kind);

    const auto common_dt = convert_dt(prb->sdt);

    graph.create_lt(src_id, common_dt, prb->dims, prb->stag);
    graph.create_lt(dst_id, common_dt, prb->dims, prb->stag);

    std::vector<size_t> src_ids {};
    std::vector<size_t> dst_ids {};
    dnnl::graph::op::kind softmax_kind;
    if (prb->dir & FLAG_FWD) {
        src_ids.push_back(src_id);
        dst_ids.push_back(dst_id);
        softmax_kind = prb->alg == ::softmax::SOFTMAX
                ? dnnl::graph::op::kind::SoftMax
                : dnnl::graph::op::kind::LogSoftmax;
    } else {
        const auto fwd_dst_id = graph.generate_id_for(op_id, lt_kind::DST);
        graph.create_lt(fwd_dst_id, common_dt, prb->dims, prb->stag);
        src_ids = {dst_id, fwd_dst_id};
        dst_ids.push_back(src_id);
        softmax_kind = prb->alg == ::softmax::SOFTMAX
                ? dnnl::graph::op::kind::SoftMaxBackprop
                : dnnl::graph::op::kind::LogSoftmaxBackprop;
    }

    dnnl::graph::op softmax_op(op_id, softmax_kind, graph.stringify_id(op_id));
    softmax_op.set_attr("axis", static_cast<int64_t>(prb->axis));

    graph.append(op_id, softmax_op, src_ids, dst_ids);

    graph.close_block();

    return fill_status::DONE;
}

int doit(const ::softmax::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;

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

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::softmax::init_pd, prb, res, par, ins, outs);

    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    dnn_mem_t &dst_fp = src_fp; // in-place reference

    const auto placeholder_dst_dt = make_dnn_mem(outs[0], (prb->stag).c_str());
    auto src_dt = make_dnn_mem(ins[0], (prb->stag).c_str());
    const dnn_mem_t &dst_dt = prb->inplace ? src_dt : placeholder_dst_dt;

    std::vector<dnnl::graph::tensor> tensors_in, tensors_out;
    const dnnl::graph::engine &eng = get_test_engine();

    if (prb->dir & FLAG_FWD) {
        SAFE(::softmax::fill_data_fwd(prb, src_dt, src_fp), WARN);

        tensors_in.emplace_back(ins[0], eng, static_cast<void *>(src_dt));
        tensors_out.emplace_back(outs[0], eng, static_cast<void *>(dst_dt));

        SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

        if (is_bench_mode(CORR)) {
            args_t args, ref_args;

            args.set(DNNL_ARG_DST, dst_dt);
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);

            check_correctness(
                    prb, {DST}, args, ref_args, ::softmax::setup_cmp, res);
        }
    } else if (prb->dir & FLAG_BWD) {
        auto d_dst_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
        auto d_dst_dt = make_dnn_mem(ins[0], (prb->stag).c_str());

        auto placeholder_d_src_dt = make_dnn_mem(outs[0], (prb->stag).c_str());
        dnn_mem_t &d_src_fp = d_dst_fp; // in-place reference
        dnn_mem_t &d_src_dt = prb->inplace ? d_dst_dt : placeholder_d_src_dt;

        const bool neg_sign = prb->alg == ::softmax::SOFTMAX ? true : false;
        SAFE(::softmax::fill_data_bwd(prb, src_dt, src_fp, neg_sign), WARN);
        SAFE(::softmax::fill_data_bwd(prb, d_dst_dt, d_dst_fp, !neg_sign),
                WARN);

        tensors_in.emplace_back(ins[0], eng, static_cast<void *>(d_dst_dt));
        tensors_in.emplace_back(ins[1], eng, static_cast<void *>(src_dt));
        tensors_out.emplace_back(outs[0], eng, static_cast<void *>(d_src_dt));

        SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

        if (is_bench_mode(CORR)) {
            args_t args, ref_args;

            args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, d_dst_fp);
            ref_args.set(DNNL_ARG_DIFF_SRC, d_src_fp);

            check_correctness(
                    prb, {SRC}, args, ref_args, ::softmax::setup_cmp, res);
        }
    } else {
        SAFE(FAIL, CRIT);
    }

    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);

    cleanup();

    return OK;
}

} // namespace softmax
} // namespace benchdnnext
