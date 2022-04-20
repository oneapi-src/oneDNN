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

void check_known_skipped_case_graph(
        const ::prelu::prb_t *prb, res_t *res) noexcept {
    // TODO: to align with original benchdnn, we should consider moving
    // skip_unimplemented_prb call after compilation step
    skip_invalid_and_unimplemented_prb(prb, res);
}

fill_status_t prelu_graph_prb_t::handle_main_op_(const ::prelu::prb_t *prb) {
    using logical_tensor = dnnl::graph::logical_tensor;
    using op = dnnl::graph::op;

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);
    const std::string SRC {TENSOR_ID + "_SRC"};
    const std::string SLOPE {TENSOR_ID + "_SLOPE"};

    const auto data_dt = convert_dt(prb->sdt[0]);
    const auto slope_dt = convert_dt(prb->sdt[1]);
    const auto data_dims = prb->vdims[0];
    const auto slope_dims = prb->vdims[1];
    const auto data_tag = prb->stag[0];
    const auto slope_tag = prb->stag[1];

    tensor_descs_.emplace(SRC, data_dt, data_dims, data_tag);
    tensor_descs_.emplace(SLOPE, slope_dt, slope_dims, slope_tag);

    std::string name;
    std::vector<logical_tensor> inputs;
    std::vector<logical_tensor> outputs;
    if (prb->dir & FLAG_FWD) {
        name = "prelu";
        const std::string DST {TENSOR_ID + "_DST"};

        tensor_descs_.emplace(DST, data_dt, data_dims, data_tag);
        inputs = {tensor_descs_[SRC], tensor_descs_[SLOPE]};
        outputs = {tensor_descs_[DST]};
    } else { // bwd
        name = "prelu_bwd";
        const std::string DIFF_DST {TENSOR_ID + "_DIFF_DST"};
        const std::string DIFF_SRC {TENSOR_ID + "_DIFF_SRC"};
        const std::string DIFF_SLOPE {TENSOR_ID + "_DIFF_SLOPE"};

        tensor_descs_.emplace(DIFF_DST, data_dt, data_dims, data_tag);
        tensor_descs_.emplace(DIFF_SRC, data_dt, data_dims, data_tag);
        tensor_descs_.emplace(DIFF_SLOPE, slope_dt, slope_dims, slope_tag);
        inputs = {tensor_descs_[SRC], tensor_descs_[SLOPE],
                tensor_descs_[DIFF_DST]};
        outputs = {tensor_descs_[DIFF_SRC], tensor_descs_[DIFF_SLOPE]};
    }

    const std::string data_format {"NCX"};
    const bool per_channel_broadcast {false};

    op prelu_op(new_op_id, op_kind, inputs, outputs, name);
    prelu_op.set_attr<std::string>("data_format", data_format);
    if (prb->dir & FLAG_FWD)
        prelu_op.set_attr<bool>("per_channel_broadcast", per_channel_broadcast);

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

    prelu_graph_prb_t graph_prb(prb);
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

    return OK;
}
} // namespace prelu
} // namespace benchdnnext
