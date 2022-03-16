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

softmax_graph_prb_t::spec_t::spec_t(const ::softmax::prb_t *prb) noexcept {
    using graph_op = dnnl::graph::op;
    is_bwd_pass = prb->dir & FLAG_BWD;
    axis = prb->axis;
    dims = prb->dims;
    softmax_dt = convert_dt(prb->sdt);
    switch (prb->alg) {
        case ::softmax::SOFTMAX:
            op_kind = (is_bwd_pass) ? graph_op::kind::SoftMaxBackprop
                                    : graph_op::kind::SoftMax;
            break;
        case ::softmax::LOGSOFTMAX:
            op_kind = (is_bwd_pass) ? graph_op::kind::LogSoftmaxBackprop
                                    : graph_op::kind::LogSoftmax;
            break;
        default: op_kind = graph_op::kind::LastSymbol;
    }
    tag = prb->stag;
}

void check_known_skipped_case_graph(
        const ::softmax::prb_t *prb, res_t *res) noexcept {
    check_known_skipped_case_common({prb->sdt}, prb->dir, res);
    if (res->state == SKIPPED) return;
    check_known_skipped_case_graph_common(
            {prb->sdt}, normalize_tag(prb->stag, prb->ndims), prb->dir, res);
    if (res->state == SKIPPED) return;
}

fill_status_t softmax_graph_prb_t::handle_main_op_() {
    using logical_tensor = dnnl::graph::logical_tensor;
    using op = dnnl::graph::op;

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);

    const std::string DST {TENSOR_ID + "_DST"};
    tensor_descs_.emplace(DST, spec_.softmax_dt, spec_.dims, spec_.tag);

    std::string name;
    std::vector<logical_tensor> inputs;
    std::vector<logical_tensor> outputs;
    if (spec_.is_bwd_pass) {
        name = spec_.op_kind == op::kind::SoftMaxBackprop
                ? "SoftMaxBackprop"
                : "LogSoftmaxBackprop";
        const std::string DIFF_DST {TENSOR_ID + "_DIFF_DST"};
        const std::string DIFF_SRC {TENSOR_ID + "_DIFF_SRC"};

        tensor_descs_.emplace(
                DIFF_SRC, spec_.softmax_dt, spec_.dims, spec_.tag);
        tensor_descs_.emplace(
                DIFF_DST, spec_.softmax_dt, spec_.dims, spec_.tag);
        inputs = {tensor_descs_[DIFF_DST], tensor_descs_[DST]};
        outputs = {tensor_descs_[DIFF_SRC]};
    } else {
        name = spec_.op_kind == op::kind::SoftMax ? "SoftMax" : "LogSoftmax";
        const std::string SRC {TENSOR_ID + "_SRC"};

        tensor_descs_.emplace(SRC, spec_.softmax_dt, spec_.dims, spec_.tag);
        inputs = {tensor_descs_[SRC]};
        outputs = {tensor_descs_[DST]};
    }

    op softmax_op(new_op_id, spec_.op_kind, inputs, outputs, name);
    softmax_op.set_attr<int64_t>("axis", spec_.axis);

    ops_.emplace_back(softmax_op);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

int doit(const ::softmax::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

    softmax_graph_prb_t graph_prb(prb);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h = graph_prb.to_graph();

    const auto partitions = graph_h.get_partitions();
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::softmax::init_pd, prb, res, par, ins, outs);

    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    dnn_mem_t &dst_fp = src_fp; // in-place reference

    const auto placeholder_dst_dt = make_dnn_mem(outs[0], (prb->stag).c_str());
    auto src_dt = make_dnn_mem(ins[0], (prb->stag).c_str());
    const dnn_mem_t &dst_dt = prb->inplace ? src_dt : placeholder_dst_dt;

    std::vector<dnnl::graph::tensor> tensors_in, tensors_out;
    dnnl::graph::engine &eng = get_test_engine();

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

    return OK;
}

} // namespace softmax
} // namespace benchdnnext
