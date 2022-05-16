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

namespace benchdnnext {
namespace binary {

void check_broadcast_rules(const ::binary::prb_t *prb, res_t *res) {
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

void check_known_skipped_case_graph(const ::binary::prb_t *prb, res_t *res) {
    using p = attr_t::post_ops_t;
    // Binary ops supports relu, sigmoid, sum and binary post-ops.
    // Other cases are being skipped.
    for (const auto &po : prb->attr.post_ops.entry) {
        if (po.kind == p::RELU || po.kind == p::LOGISTIC || po.is_binary_kind()
                || po.is_sum_kind()) {
            continue;
        } else {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }

    check_graph_eltwise_post_ops(prb->attr, res);
    if (res->state == SKIPPED) return;
}

fill_status_t binary_graph_prb_t::handle_main_op_(const ::binary::prb_t *prb) {
    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);
    const std::string SRC0 {TENSOR_ID + "_SRC"};
    const std::string SRC1 {TENSOR_ID + "_SRC1"};
    const std::string DST {TENSOR_ID + "_DST"};

    tensor_descs_.emplace(
            SRC0, convert_dt(prb->sdt[0]), prb->vdims[0], prb->stag[0]);
    tensor_descs_.emplace(
            SRC1, convert_dt(prb->sdt[1]), prb->vdims[1], prb->stag[1]);
    tensor_descs_.emplace(DST, convert_dt(prb->ddt), prb->dst_dims, prb->dtag);

    const auto op_kind
            = convert_alg_kind(attr_t::post_ops_t::kind2dnnl_kind(prb->alg));

    dnnl::graph::op binary(new_op_id, op_kind,
            {tensor_descs_[SRC0], tensor_descs_[SRC1]}, {tensor_descs_[DST]},
            "binary");

    bool has_post_sum = false;
    for (const auto &po : prb->attr.post_ops.entry) {
        if (po.is_sum_kind()) { has_post_sum = true; }
    }
    const std::string auto_broadcast
            = (prb->vdims[0] == prb->vdims[1] && !has_post_sum) ? "none"
                                                                : "numpy";

    binary.set_attr("auto_broadcast", auto_broadcast);

    ops_.emplace_back(binary);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

fill_status_t binary_graph_prb_t::handle_elt_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.binary.eltw_handler(*this, po_entry);
}

fill_status_t binary_graph_prb_t::handle_bin_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.binary.bin_handler(*this, po_entry);
}

fill_status_t binary_graph_prb_t::handle_sum_() {
    return po_handler.binary.sum_handler(*this);
}

int doit(const ::binary::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    // TODO: to align with original benchdnn, we should consider moving
    // skip_unimplemented_prb call after compilation step
    skip_invalid_and_unimplemented_prb(prb, res);
    if (res->state == SKIPPED) return OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;
    check_broadcast_rules(prb, res);
    if (res->state == SKIPPED) return OK;

    binary_graph_prb_t graph_prb(prb);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h = graph_prb.to_graph();

    // Filter partitions
    const auto partitions = graph_h.get_partitions();
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

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

    return OK;
}

} // namespace binary
} // namespace benchdnnext
