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

#include <algorithm>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "dnnl_graph_common.hpp"

#include "compare.hpp"
#include "graph_matmul.hpp"

namespace benchdnnext {
namespace matmul {

matmul_graph_prb_t::spec_t::spec_t(const ::matmul::prb_t *prb) {
    src_dims = get_runtime_dims(prb->src_dims(), prb->src_runtime_dim_mask());
    wei_dims = get_runtime_dims(
            prb->weights_dims(), prb->weights_runtime_dim_mask());
    dst_dims = get_runtime_dims(prb->dst_dims(), prb->dst_runtime_dim_mask());

    src_dt = convert_dt(prb->cfg[SRC].dt);
    wei_dt = convert_dt(prb->cfg[WEI].dt);
    dst_dt = convert_dt(prb->cfg[DST].dt);
    bia_dt = convert_dt(prb->bia_dt);

    src_tag = convert_tag(prb->stag);
    wei_tag = convert_tag(prb->wtag);
    dst_tag = convert_tag(prb->dtag);
}

fill_status_t matmul_graph_prb_t::handle_main_op_() {
    using op = dnnl::graph::op;

    const std::string SRC {"matmul_src"};
    const std::string WEI {"matmul_wei"};
    const std::string DST {"matmul_dst"};

    tensor_descs_.emplace(SRC, spec_.src_dt, spec_.src_dims, lt::strided);
    tensor_descs_.emplace(WEI, spec_.wei_dt, spec_.wei_dims, lt::strided);
    tensor_descs_.emplace(DST, spec_.dst_dt, spec_.dst_dims, lt::strided);

    const size_t new_op_id = ops_.size();
    op matmul(new_op_id, op::kind::MatMul,
            {tensor_descs_[SRC], tensor_descs_[WEI]}, {tensor_descs_[DST]},
            "matmul");

    matmul.set_attr("transpose_a", spec_.transpose_a)
            .set_attr("transpose_b", spec_.transpose_b);

    ops_.emplace_back(matmul);
    curr_out_map_ids_.assign({DST});

    return fill_status::DONE;
}

fill_status_t matmul_graph_prb_t::handle_bia_() {
    return po_handler.matmul.bias_handler(*this, spec_.dst_tag, spec_.bia_dt);
}

fill_status_t matmul_graph_prb_t::handle_elt_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.matmul.eltw_handler(*this, po_entry);
}

fill_status_t matmul_graph_prb_t::handle_sum_() {
    return po_handler.matmul.sum_handler(*this);
}

dims_t get_runtime_dims(const dims_t &dims, const ::matmul::dims_mask_t &mask) {
    if (mask.none() || dims.empty()) return dims;
    dims_t runtime_dims;
    runtime_dims.resize(dims.size());
    const int64_t axis_unknown_flag = -1;
    for (size_t i = 0; i < dims.size(); ++i) {
        runtime_dims[i] = mask[i] ? axis_unknown_flag : dims[i];
    }
    return runtime_dims;
}

int doit(const ::matmul::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;

    res->impl_name = "graph";
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    matmul_graph_prb_t graph_prb(prb);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto g = graph_prb.to_graph();

    // Filter partitions
    const auto partitions = g.get_partitions();
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    const auto &e = benchdnnext::get_test_engine();
    auto cp = par.compile(ins, outs, e);

    const auto apply_bias = convert_dt(prb->bia_dt) != dt::undef;

    dnn_mem_t src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    dnn_mem_t wei_fp = make_dnn_mem(ins[1], dt::f32, tag::abx);
    dnn_mem_t dst_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);
    dnn_mem_t bia_fp;
    if (apply_bias) bia_fp = make_dnn_mem(ins[2], dt::f32, tag::x);
    // matmul operator doesn't support binary post-ops yet
    std::vector<dnn_mem_t> binary_po_fp;

    dnn_mem_t src_dt = make_dnn_mem(ins[0], tag::abx);
    dnn_mem_t wei_dt = make_dnn_mem(ins[1], tag::abx);
    dnn_mem_t dst_dt = make_dnn_mem(outs[0], tag::abx);
    dnn_mem_t bia_dt;
    if (apply_bias) bia_dt = make_dnn_mem(ins[2], tag::x);

    SAFE(fill_data(SRC, prb, src_dt, src_fp, res), WARN);
    SAFE(fill_data(WEI, prb, wei_dt, wei_fp, res), WARN);
    SAFE(fill_data(DST, prb, dst_dt, dst_fp, res), WARN);
    if (apply_bias) SAFE(fill_data(BIA, prb, bia_dt, bia_fp, res), WARN);

    dnnl::graph::tensor src_tensor(ins[0], static_cast<void *>(src_dt));
    dnnl::graph::tensor wei_tensor(ins[1], static_cast<void *>(wei_dt));
    dnnl::graph::tensor dst_tensor(outs[0], static_cast<void *>(dst_dt));
    dnnl::graph::tensor bia_tensor;
    dnnl::graph::tensor sum_src1_tensor;

    std::vector<dnnl::graph::tensor> input_ts {src_tensor, wei_tensor};
    std::vector<dnnl::graph::tensor> output_ts {dst_tensor};

    if (apply_bias) {
        bia_tensor = dnnl::graph::tensor(ins[2], static_cast<void *>(bia_dt));
        input_ts.push_back(bia_tensor);
    }
    if (graph_prb.has_post_sum()) {
        sum_src1_tensor
                = dnnl::graph::tensor(ins.back(), static_cast<void *>(dst_dt));
        input_ts.push_back(sum_src1_tensor);
    }

    SAFE(execute_and_wait(cp, input_ts, output_ts), WARN);

    if (is_bench_mode(CORR)) {
        const auto &dnnl_test_engine = ::get_test_engine();
        ::matmul::compute_ref(dnnl_test_engine, prb, src_fp, wei_fp, bia_fp,
                binary_po_fp, dst_fp);
        compare::compare_t cmp;
        cmp.set_threshold(prb->cfg[DST].eps);
        cmp.set_data_kind(DST);
        cmp.set_zero_trust_percent(90.f); // TODO: why so bad filling?

        SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
    }

    measure_perf(res->timer, cp, input_ts, output_ts);

    return OK;
}

} // namespace matmul
} // namespace benchdnnext
