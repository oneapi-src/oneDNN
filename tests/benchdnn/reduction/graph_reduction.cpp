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
#include "reduction/graph_reduction.hpp"
#include "reduction/reduction.hpp"

#include <tuple>

namespace benchdnnext {
namespace reduction {

static int check_known_skipped_case_graph(
        const ::reduction::prb_t *prb, res_t *res) noexcept {

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, ::reduction::init_pd, prb, res), WARN);

    auto const_pd = query_pd(prim);
    if (check_mem_size(const_pd) != OK) {
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    return OK;
}

fill_status_t append_graph_with_block(const ::reduction::prb_t *prb) {
    graph_t &graph = graph_t::get();

    const auto connect_to_previous_block = graph.has_blocks();

    // handle main op
    const auto op_id = graph.generate_id_for(entry_kind::REDUCTION);
    const auto src_id = connect_to_previous_block
            ? graph.get_last_block_out_id()
            : graph.generate_id_for(op_id, lt_kind::SRC);
    const auto dst_id = graph.generate_id_for(op_id, lt_kind::DST);

    const auto src_dims = prb->vdims[0];
    const auto dst_dims = prb->vdims[1];

    graph.create_lt(src_id, convert_dt(prb->sdt), src_dims, prb->stag);
    graph.create_lt(dst_id, convert_dt(prb->ddt), dst_dims, prb->dtag);

    const auto reduction_kind
            = convert_alg_kind(::reduction::alg2alg_kind(prb->alg));
    std::vector<int64_t> axes;
    const int reduction_dim_size = 1;
    for (auto d = 0; d < prb->ndims; ++d) {
        const bool is_reduction_dim = dst_dims[d] == reduction_dim_size
                && src_dims[d] != dst_dims[d];
        if (is_reduction_dim) axes.push_back(d);
    }

    dnnl::graph::op reduction_op(
            op_id, reduction_kind, graph.stringify_id(op_id));
    reduction_op.set_attr("keep_dims", true).set_attr("axes", axes);

    graph.append(op_id, reduction_op, {src_id}, {dst_id});

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

bool is_norm_alg(const ::reduction::alg_t alg) {
    return alg == ::reduction::alg_t::norm_lp_max
            || alg == ::reduction::alg_t::norm_lp_sum
            || alg == ::reduction::alg_t::norm_lp_power_p_max
            || alg == ::reduction::alg_t::norm_lp_power_p_sum;
}

int doit(const ::reduction::prb_t *prb, res_t *res) {
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

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::reduction::init_pd, prb, res, par, ins, outs);

    dnn_mem_t src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    dnn_mem_t dst_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);

    dnn_mem_t src_dt = make_dnn_mem(ins[0], prb->stag);
    dnn_mem_t dst_dt = make_dnn_mem(outs[0], prb->dtag);

    SAFE(::reduction::fill_src(prb, src_dt, src_fp), WARN);
    SAFE(::reduction::fill_dst(prb, dst_dt, dst_fp), WARN);

    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    const auto post_bin_indices
            = get_post_bin_indices(prb->attr.post_ops.entry);

    size_t idx_ins = 0;

    // norm relative algorithm should initialize data with positive data.
    const bool binary_po_only_positive_vals = is_norm_alg(prb->alg);
    for (size_t i = 0; i < post_bin_indices.size(); ++i) {
        binary_po_fp.emplace_back(
                make_dnn_mem(ins[++idx_ins], dt::f32, tag::abx));
        binary_po_dt.emplace_back(make_dnn_mem(ins[idx_ins], prb->dtag));
        const int po_idx = DNNL_ARG_ATTR_MULTIPLE_POST_OP(
                                   static_cast<int>(post_bin_indices[i]))
                | DNNL_ARG_SRC_1;
        ::binary::fill_mem(po_idx, binary_po_dt[i], binary_po_fp[i],
                binary_po_only_positive_vals);
        binary_po_args.push_back(po_idx);
    }

    const dnnl::graph::engine &eng = get_test_engine();

    dnnl::graph::tensor src_tensor(ins[0], eng, static_cast<void *>(src_dt));
    dnnl::graph::tensor dst_tensor(outs[0], eng, static_cast<void *>(dst_dt));

    std::vector<dnnl::graph::tensor> tensors_in {src_tensor};
    std::vector<dnnl::graph::tensor> tensors_out {dst_tensor};

    idx_ins = 0;
    size_t bin_dt_idx = 0;
    for (const auto &po_entry : prb->attr.post_ops.entry) {
        if (po_entry.is_binary_kind()) {
            dnnl::graph::tensor bin_tensor(ins[++idx_ins], eng,
                    static_cast<void *>(binary_po_dt[bin_dt_idx++]));
            tensors_in.emplace_back(bin_tensor);
        } else if (po_entry.is_sum_kind()) {
            dnnl::graph::tensor sum_tensor(
                    ins[++idx_ins], eng, static_cast<void *>(dst_dt));
            tensors_in.emplace_back(sum_tensor);
        }
    }

    SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

    if (is_bench_mode(CORR)) {
        args_t args, ref_args;

        args.set(DNNL_ARG_DST, dst_dt);
        ref_args.set(DNNL_ARG_SRC, src_fp);
        ref_args.set(DNNL_ARG_DST, dst_fp);
        ref_args.set(binary_po_args, binary_po_fp);

        check_correctness(
                prb, {DST}, args, ref_args, ::reduction::setup_cmp, res);
    }

    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);

    cleanup();

    return OK;
}

} // namespace reduction
} // namespace benchdnnext
