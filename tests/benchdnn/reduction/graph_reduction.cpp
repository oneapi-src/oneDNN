/*******************************************************************************
* Copyright 2021 Intel Corporation
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

namespace benchdnnext {
namespace reduction {

reduction_graph_prb_t::spec_t::spec_t(const ::reduction::prb_t *prb) {
    src_dims = prb->vdims[0];
    dst_dims = prb->vdims[1];

    // oneDNN require src dims and dst dims to have
    // the same number of dimensions
    if (src_dims.size() != dst_dims.size()) return;

    const int reduction_dim_size = 1;
    for (auto d = 0; d < prb->ndims; ++d) {
        const bool is_reduction_dim = dst_dims[d] == reduction_dim_size
                && src_dims[d] != dst_dims[d];
        if (is_reduction_dim) axes.push_back(d);
    }

    src_dt = convert_dt(prb->sdt);
    dst_dt = convert_dt(prb->ddt);

    raw_src_tag = prb->stag;
    raw_dst_tag = prb->dtag;

    alg = convert_alg_kind(::reduction::alg2alg_kind(prb->alg));
}

void check_known_skipped_case_graph(const ::reduction::prb_t *prb, res_t *res) {
    ::reduction::check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return;
}

fill_status_t reduction_graph_prb_t::handle_main_op_() {
    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);
    const std::string SRC {TENSOR_ID + "_SRC"};
    const std::string DST {TENSOR_ID + "_DST"};

    tensor_descs_.emplace(SRC, spec_.src_dt, spec_.src_dims, spec_.raw_src_tag);
    tensor_descs_.emplace(DST, spec_.dst_dt, spec_.dst_dims, spec_.raw_dst_tag);

    dnnl::graph::op reduction(new_op_id, spec_.alg, {tensor_descs_[SRC]},
            {tensor_descs_[DST]}, "reduction");

    reduction.set_attr("keep_dims", spec_.keep_dims);
    reduction.set_attr("axes", spec_.axes);

    ops_.emplace_back(reduction);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

fill_status_t reduction_graph_prb_t::handle_bin_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.reduction.bin_handler(*this, spec_.raw_dst_tag, po_entry);
}

fill_status_t reduction_graph_prb_t::handle_elt_(
        const attr_t::post_ops_t::entry_t &po) {
    return po_handler.reduction.eltw_handler(*this, po);
}

fill_status_t reduction_graph_prb_t::handle_sum_() {
    return po_handler.reduction.sum_handler(*this);
}

int doit(const ::reduction::prb_t *prb, res_t *res) {
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

    reduction_graph_prb_t graph_prb(prb);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h = graph_prb.to_graph();

    // Filter partitions
    auto partitions = graph_h.get_partitions();
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

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
    if (graph_prb.has_post_bin()) {
        binary_po_fp.emplace_back(make_dnn_mem(ins.back(), dt::f32, tag::abx));
        binary_po_dt.emplace_back(make_dnn_mem(ins.back(), prb->dtag));
        const int idx = 0;
        ::binary::fill_mem(DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx),
                binary_po_dt.back(), binary_po_fp.back());
    }

    dnnl::graph::engine &eng = get_test_engine();

    dnnl::graph::tensor src_tensor(ins[0], eng, static_cast<void *>(src_dt));
    dnnl::graph::tensor dst_tensor(outs[0], eng, static_cast<void *>(dst_dt));
    dnnl::graph::tensor bin_tensor;
    dnnl::graph::tensor sum_tensor;

    std::vector<dnnl::graph::tensor> tensors_in {src_tensor};
    std::vector<dnnl::graph::tensor> tensors_out {dst_tensor};

    if (graph_prb.has_post_bin()) {
        bin_tensor = dnnl::graph::tensor(
                ins.back(), eng, static_cast<void *>(binary_po_dt.back()));
        tensors_in.emplace_back(bin_tensor);
    } else if (graph_prb.has_post_sum()) {
        sum_tensor = dnnl::graph::tensor(
                ins.back(), eng, static_cast<void *>(dst_dt));
        tensors_in.emplace_back(sum_tensor);
    }

    SAFE(execute_and_wait(cp, tensors_in, tensors_out), WARN);

    if (is_bench_mode(CORR)) {
        ::reduction::compute_ref(prb, src_fp, binary_po_fp, dst_fp);
        compare::compare_t cmp;
        // `5` is a temporary magic const for GPU to pass norm algs.
        // TODO: consider change the filling with power-of-two values for better
        // answer precision.
        cmp.set_threshold(5 * epsilon_dt(prb->ddt));
        SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
    }

    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);

    return OK;
}

} // namespace reduction
} // namespace benchdnnext