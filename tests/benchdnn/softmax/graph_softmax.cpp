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

#include "compare.hpp"
#include "dnnl_graph_common.hpp"

#include "softmax/graph_softmax.hpp"
#include "softmax/softmax.hpp"

namespace benchdnnext {
namespace softmax {

softmax_graph_prb_t::spec_t::spec_t(const ::softmax::prb_t *prb) noexcept {
    axis = prb->axis;
    dims = prb->dims;
    softmax_dt = convert_dt(prb->dt);
    switch (prb->alg) {
        case ::softmax::SOFTMAX:
            op_kind = dnnl::graph::op::kind::SoftMax;
            break;
        case ::softmax::LOGSOFTMAX:
            op_kind = dnnl::graph::op::kind::LogSoftmax;
            break;
        default: op_kind = dnnl::graph::op::kind::LastSymbol;
    }
}

void check_known_skipped_case_graph(
        const ::softmax::prb_t *prb, res_t *res) noexcept {
    check_known_skipped_case_common({prb->dt}, prb->dir, res);
    if (res->state == SKIPPED) return;
    check_known_skipped_case_graph_common(
            {prb->dt}, normalize_tag(prb->tag, prb->ndims), prb->dir, res);
    if (res->state == SKIPPED) return;
}

void add_additional_softmax_check(compare::compare_t &cmp) noexcept {
    const auto softmax_add_check
            = [&](const compare::compare_t::driver_check_func_args_t &args) {
                  // SSE4.1 and OpenCL rdiff tolerance is too high for
                  // certain scenarios.
                  return args.diff < epsilon_dt(args.dt);
              };
    cmp.set_driver_check_function(softmax_add_check);
}

fill_status_t softmax_graph_prb_t::handle_main_op_() {
    using op = dnnl::graph::op;

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);
    const std::string SRC {TENSOR_ID + "_SRC"};
    const std::string DST {TENSOR_ID + "_DST"};

    tensor_descs_.emplace(SRC, spec_.softmax_dt, spec_.dims, lt::strided);
    tensor_descs_.emplace(DST, spec_.softmax_dt, spec_.dims, lt::strided);

    std::string name
            = spec_.op_kind == op::kind::SoftMax ? "Softmax" : "LogSoftMax";
    op softmax_op(new_op_id, spec_.op_kind, {tensor_descs_[SRC]},
            {tensor_descs_[DST]}, name);
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

    const auto placeholder_dst_dt = make_dnn_mem(outs[0], (prb->tag).c_str());
    auto src_dt = make_dnn_mem(ins[0], (prb->tag).c_str());
    const dnn_mem_t &dst_dt = prb->inplace ? src_dt : placeholder_dst_dt;

    std::vector<dnnl::graph::tensor> tensors_in, tensors_out;
    if (prb->dir & FLAG_FWD) {
        SAFE(::softmax::fill_data_fwd(prb, src_dt, src_fp), WARN);

        tensors_in.emplace_back(ins[0], static_cast<void *>(src_dt));
        tensors_out.emplace_back(outs[0], static_cast<void *>(dst_dt));

        SAFE(execute_and_wait(cp, tensors_in, tensors_out), WARN);

        if (is_bench_mode(CORR)) {
            ::softmax::compute_ref_fwd(prb, src_fp, dst_fp);

            compare::compare_t cmp;
            const float trh_coeff_log
                    = prb->alg == ::softmax::LOGSOFTMAX ? 4 : 1;
            const float trh_coeff_f32 = prb->dt == dnnl_f32 ? 10.f : 1.f;
            const float trh
                    = trh_coeff_log * trh_coeff_f32 * epsilon_dt(prb->dt);
            cmp.set_threshold(trh);

            const int64_t axis_size = prb->dims[prb->axis];
            cmp.set_zero_trust_percent(axis_size < 10 ? 100.f : 60.f);

            add_additional_softmax_check(cmp);

            SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
        }
    }
    SAFE(measure_perf(res->timer, cp, tensors_in, tensors_out), WARN);

    return OK;
}

} // namespace softmax
} // namespace benchdnnext
