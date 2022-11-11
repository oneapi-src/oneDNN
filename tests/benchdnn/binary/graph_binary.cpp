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

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "dnnl_graph_common.hpp"
#include "utils/compare.hpp"

#include "binary/binary.hpp"
#include "binary/graph_binary.hpp"

#include <tuple>

namespace benchdnnext {
namespace binary {

static void check_broadcast_rules(const ::binary::prb_t *prb, res_t *res) {
    const auto src0_dims = prb->vdims[0];
    const auto src1_dims = prb->vdims[1];

    // General broadcast rules:
    // Two dimensions are compatible when
    // 1) they are equal, or
    // 2) one of them is 1
    for (auto d = 0; d < (int)prb->vdims[0].size(); ++d) {
        if (src0_dims[d] == src1_dims[d] || src0_dims[d] == 1
                || src1_dims[d] == 1) {
            continue;
        } else {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }
}

static int check_known_skipped_case_graph(
        const ::binary::prb_t *prb, res_t *res) {

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, ::binary::init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    auto const_pd = query_pd(prim);
    if (check_mem_size(const_pd) != OK) {
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    using p = attr_t::post_ops_t;
    // Binary ops supports relu, sigmoid, sum and binary post-ops.
    // Other cases are being skipped.
    for (const auto &po : prb->attr.post_ops.entry) {
        if (po.kind == p::RELU || po.kind == p::LOGISTIC || po.is_binary_kind()
                || po.is_sum_kind()) {
            continue;
        } else {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return OK;
        }
    }

    check_graph_eltwise_post_ops(prb->attr, res);
    return OK;
}

fill_status_t append_graph_with_block(const ::binary::prb_t *prb) {
    graph_t &graph = graph_t::get();

    const auto connect_to_previous_block = graph.has_blocks();

    // handle main op
    const auto op_id = graph.generate_id_for(entry_kind::BINARY);
    const auto src0_id = connect_to_previous_block
            ? graph.get_last_block_out_id()
            : graph.generate_id_for(op_id, lt_kind::SRC);
    const auto src1_id = graph.generate_id_for(op_id, lt_kind::SRC1);
    const auto dst_id = graph.generate_id_for(op_id, lt_kind::DST);

    graph.create_lt(
            src0_id, convert_dt(prb->sdt[0]), prb->vdims[0], prb->stag[0]);
    graph.create_lt(
            src1_id, convert_dt(prb->sdt[1]), prb->vdims[1], prb->stag[1]);
    graph.create_lt(dst_id, convert_dt(prb->ddt), prb->dst_dims, prb->dtag);

    const auto binary_kind
            = convert_alg_kind(attr_t::post_ops_t::kind2dnnl_kind(prb->alg));
    bool has_post_sum = false;
    for (const auto &po : prb->attr.post_ops.entry)
        if (po.is_sum_kind()) has_post_sum = true;
    const std::string auto_broadcast
            = (prb->vdims[0] == prb->vdims[1] && !has_post_sum) ? "none"
                                                                : "numpy";

    dnnl::graph::op binary_op(op_id, binary_kind, graph.stringify_id(op_id));
    binary_op.set_attr("auto_broadcast", auto_broadcast);

    graph.append(op_id, binary_op, {src0_id, src1_id}, {dst_id});

    // handle post ops
    fill_status_t status;
    for (const auto &entry : prb->attr.post_ops.entry) {
        if (entry.is_binary_kind()) {
            std::tie(status, std::ignore) = append_graph_with_binary(entry);
            BENCHDNNEXT_VERIFY(status);
        } else if (entry.is_eltwise_kind()) {
            status = append_graph_with_eltwise(entry);
            BENCHDNNEXT_VERIFY(status);
        } else if (entry.is_sum_kind()) {
            std::tie(status, std::ignore) = append_graph_with_sum(entry);
            BENCHDNNEXT_VERIFY(status);
        }
    }

    graph.close_block();

    return fill_status::DONE;
}

int doit(const ::binary::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;
    check_broadcast_rules(prb, res);
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

    auto cp = compile_partition(::binary::init_pd, prb, res, par, ins, outs);

    auto src0_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto src1_fp = make_dnn_mem(ins[1], dt::f32, tag::abx);
    auto dst_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);

    auto src0_dt = make_dnn_mem(ins[0], prb->stag[0]);
    auto src1_dt = make_dnn_mem(ins[1], prb->stag[1]);
    auto dst_dt = make_dnn_mem(outs[0], prb->dtag);

    SAFE(::binary::fill_mem(0, src0_dt, src0_fp), WARN);
    SAFE(::binary::fill_mem(1, src1_dt, src1_fp), WARN);
    SAFE(::binary::fill_mem(2, dst_dt, dst_fp), WARN);

    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    size_t idx_ins = 2;
    const auto post_bin_indices
            = get_post_bin_indices(prb->attr.post_ops.entry);
    for (size_t i = 0; i < post_bin_indices.size(); ++i) {
        binary_po_dt.emplace_back(make_dnn_mem(ins[idx_ins], tag::abx));
        binary_po_fp.emplace_back(
                make_dnn_mem(ins[idx_ins++], dt::f32, tag::abx));
        const int arg = static_cast<int>(post_bin_indices[i]);
        const int po_idx = DNNL_ARG_ATTR_MULTIPLE_POST_OP(arg) | DNNL_ARG_SRC_1;
        ::binary::fill_mem(po_idx, binary_po_dt[i], binary_po_fp[i]);
        binary_po_args.push_back(po_idx);
    }

    const dnnl::graph::engine &eng = get_test_engine();

    dnnl::graph::tensor src0_tensor(ins[0], eng, static_cast<void *>(src0_dt));
    dnnl::graph::tensor src1_tensor(ins[1], eng, static_cast<void *>(src1_dt));
    dnnl::graph::tensor dst_tensor(outs[0], eng, static_cast<void *>(dst_dt));

    std::vector<dnnl::graph::tensor> tensors_in {src0_tensor, src1_tensor};
    std::vector<dnnl::graph::tensor> tensors_out {dst_tensor};

    idx_ins = 2;
    size_t bin_dt_idx = 0;
    for (const auto &po_entry : prb->attr.post_ops.entry) {
        if (po_entry.is_sum_kind()) {
            dnnl::graph::tensor sum_src1_tensor(
                    ins[idx_ins], eng, static_cast<void *>(dst_dt));
            tensors_in.emplace_back(sum_src1_tensor);
        } else if (po_entry.is_binary_kind()) {
            dnnl::graph::tensor bin_src1_tensor(ins[idx_ins], eng,
                    static_cast<void *>(binary_po_dt[bin_dt_idx]));
            tensors_in.emplace_back(bin_src1_tensor);
            ++bin_dt_idx;
        }
        ++idx_ins;
    }

    SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

    if (is_bench_mode(CORR)) {
        args_t args, ref_args;

        args.set(DNNL_ARG_DST, dst_dt);
        ref_args.set(DNNL_ARG_SRC_0, src0_fp);
        ref_args.set(DNNL_ARG_SRC_1, src1_fp);
        ref_args.set(DNNL_ARG_DST, dst_fp);
        ref_args.set(binary_po_args, binary_po_fp);

        check_correctness(prb, {DST}, args, ref_args, ::binary::setup_cmp, res);
    }

    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);

    cleanup();

    return OK;
}

} // namespace binary
} // namespace benchdnnext
