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

#include "prelu/graph_prelu.hpp"
#include "utils/compare.hpp"

namespace benchdnnext {
namespace prelu {

prelu_graph_prb_t::spec_t::spec_t(const ::prelu::prb_t *prb) noexcept {
    data_dims = prb->vdims[0];
    slope_dims = prb->vdims[1];

    raw_data_tag = prb->stag[0];
    raw_slope_tag = prb->stag[1];

    data_dt = convert_dt(prb->sdt[0]);
    slope_dt = convert_dt(prb->sdt[1]);
}

void check_known_skipped_case_graph(
        const ::prelu::prb_t *prb, res_t *res) noexcept {
    ::prelu::check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return;
}

fill_status_t prelu_graph_prb_t::handle_main_op_() {
    using op = dnnl::graph::op;

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);
    const std::string SRC {TENSOR_ID + "_SRC"};
    const std::string SLOPE {TENSOR_ID + "_SLOPE"};
    const std::string DST {TENSOR_ID + "_DST"};

    tensor_descs_.emplace(
            SRC, spec_.data_dt, spec_.data_dims, spec_.raw_data_tag);
    tensor_descs_.emplace(
            SLOPE, spec_.slope_dt, spec_.slope_dims, spec_.raw_slope_tag);
    tensor_descs_.emplace(
            DST, spec_.data_dt, spec_.data_dims, spec_.raw_data_tag);
    op prelu_op(new_op_id, op::kind::PReLU,
            {tensor_descs_[SRC], tensor_descs_[SLOPE]}, {tensor_descs_[DST]},
            "prelu");
    prelu_op.set_attr<std::string>("data_format", spec_.data_format);
    prelu_op.set_attr<bool>(
            "per_channel_broadcast", spec_.broadcast_to_channel);
    ops_.emplace_back(prelu_op);
    curr_out_map_ids_.assign({TENSOR_ID});
    return fill_status::DONE;
}

int doit(const ::prelu::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;
    check_sum_post_ops(prb->attr, res);
    if (res->state == SKIPPED) return OK;

    prelu_graph_prb_t graph_prb(prb);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h = graph_prb.to_graph();
    const auto spec = graph_prb.spec();

    const auto partitions = graph_h.get_partitions();
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();
    auto cp = compile_partition(::prelu::init_pd, prb, res, par, ins, outs);

    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto slope_fp = make_dnn_mem(ins[1], dt::f32, tag::abx);
    auto dst_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);

    auto src_dt = make_dnn_mem(ins[0], prb->stag[0]);
    auto slope_dt = make_dnn_mem(ins[1], prb->stag[1]);
    auto dst_dt = make_dnn_mem(outs[0], prb->stag[0]);

    SAFE(::prelu::fill_data(SRC, src_dt, src_fp), WARN);
    SAFE(::prelu::fill_data(WEI, slope_dt, slope_fp), WARN);

    std::vector<dnnl::graph::tensor> tensors_in;
    std::vector<dnnl::graph::tensor> tensors_out;
    dnnl::graph::engine &eng = get_test_engine();

    tensors_in.emplace_back(
            dnnl::graph::tensor(ins[0], eng, static_cast<void *>(src_dt)));
    tensors_in.emplace_back(
            dnnl::graph::tensor(ins[1], eng, static_cast<void *>(slope_dt)));
    tensors_out.emplace_back(
            dnnl::graph::tensor(outs[0], eng, static_cast<void *>(dst_dt)));

    SAFE(execute_and_wait(cp, tensors_in, tensors_out), WARN);
    if (is_bench_mode(CORR)) {
        TIME_REF(::prelu::compute_ref_fwd(prb, src_fp, slope_fp, dst_fp));
        compare::compare_t cmp;
        cmp.set_threshold(2 * epsilon_dt(prb->sdt[0]));
        cmp.set_zero_trust_percent(50.f); // Due to filling
        cmp.set_data_kind(DST);
        SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
    }
    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);
    return OK;
}
} // namespace prelu
} // namespace benchdnnext
