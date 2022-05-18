/*******************************************************************************
 * * Copyright 2019-2022 Intel Corporation
 * *
 * * Licensed under the Apache License, Version 2.0 (the "License");
 * * you may not use this file except in compliance with the License.
 * * You may obtain a copy of the License at
 * *
 * *     http://www.apache.org/licenses/LICENSE-2.0
 * *
 * * Unless required by applicable law or agreed to in writing, software
 * * distributed under the License is distributed on an "AS IS" BASIS,
 * * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * * See the License for the specific language governing permissions and
 * * limitations under the License.
 * *******************************************************************************/

#include "oneapi/dnnl/dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_graph_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/norm.hpp"

#include "bnorm/bnorm.hpp"
#include "bnorm/graph_bnorm.hpp"

#include <string>
#include <vector>

namespace benchdnnext {
namespace bnorm {

void check_known_skipped_case_graph(
        const ::bnorm::prb_t *prb, res_t *res) noexcept {
    // TODO: to align with original benchdnn, we should consider moving
    // skip_unimplemented_prb call after compilation step
    skip_invalid_and_unimplemented_prb(prb, res);
    if (res->state == SKIPPED) return;
    check_known_skipped_case_graph_common(
            {prb->dt}, normalize_tag(prb->tag, prb->ndims), prb->dir, res);
    if (res->state == SKIPPED) return;

    for (const auto &po : prb->attr.post_ops.entry) {
        if (po.kind == attr_t::post_ops_t::RELU && prb->dir & FLAG_INF) {
            continue;
        } else {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }

    check_graph_eltwise_post_ops(prb->attr, res);
    if (res->state == SKIPPED) return;
}

fill_status_t bnorm_graph_prb_t::handle_main_op_(const ::bnorm::prb_t *prb) {
    using op = dnnl::graph::op;
    using graph_dt = dnnl::graph::logical_tensor::data_type;

    std::string op_name {};
    op::kind op_kind {op::kind::LastSymbol};
    std::vector<dnnl::graph::logical_tensor> inputs {};
    std::vector<dnnl::graph::logical_tensor> outputs {};

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);

    const std::string SRC {TENSOR_ID + "_SRC"};
    const std::string SCALE {TENSOR_ID + "_SCALE"};
    const std::string MEAN {TENSOR_ID + "_MEAN"};
    const std::string VAR {TENSOR_ID + "_VAR"};

    graph_dt bnorm_dt = convert_dt(prb->dt);
    dims_t dims = prb->data_dims();
    dims_t s_dims = {prb->ic};

    tensor_descs_.emplace(SRC, bnorm_dt, dims, lt::strided);
    tensor_descs_.emplace(SCALE, graph_dt::f32, s_dims, lt::strided);
    tensor_descs_.emplace(MEAN, graph_dt::f32, s_dims, lt::strided);
    tensor_descs_.emplace(VAR, graph_dt::f32, s_dims, lt::strided);
    if (prb->dir & FLAG_FWD) {
        op_name = "bnorm";
        const std::string SHIFT {TENSOR_ID + "_SHIFT"};
        const std::string DST {TENSOR_ID + "_DST"};
        tensor_descs_.emplace(SHIFT, graph_dt::f32, s_dims, lt::strided);
        tensor_descs_.emplace(DST, bnorm_dt, dims, lt::strided);
        inputs = {tensor_descs_[SRC], tensor_descs_[SCALE],
                tensor_descs_[SHIFT], tensor_descs_[MEAN], tensor_descs_[VAR]};
        if (prb->dir & FLAG_INF) {
            op_kind = op::kind::BatchNormInference;
            outputs = {tensor_descs_[DST]};
        } else {
            op_kind = op::kind::BatchNormForwardTraining;
            const std::string RUN_MEAN {TENSOR_ID + "_RUN_MEAN"};
            const std::string RUN_VAR {TENSOR_ID + "_RUN_VAR"};
            const std::string BATCH_MEAN {TENSOR_ID + "_BATCH_MEAN"};
            const std::string BATCH_VAR {TENSOR_ID + "_BATCH_VAR"};
            tensor_descs_.emplace(RUN_MEAN, graph_dt::f32, s_dims, lt::strided);
            tensor_descs_.emplace(RUN_VAR, graph_dt::f32, s_dims, lt::strided);
            tensor_descs_.emplace(
                    BATCH_MEAN, graph_dt::f32, s_dims, lt::strided);
            tensor_descs_.emplace(
                    BATCH_VAR, graph_dt::f32, s_dims, lt::strided);
            outputs = {tensor_descs_[DST], tensor_descs_[RUN_MEAN],
                    tensor_descs_[RUN_VAR], tensor_descs_[BATCH_MEAN],
                    tensor_descs_[BATCH_VAR]};
        }
    } else {
        op_name = "bnorm_bwd";
        op_kind = op::kind::BatchNormTrainingBackprop;
        const std::string DIFF_DST {TENSOR_ID + "_DIFF_DST"};
        const std::string DIFF_SRC {TENSOR_ID + "_DIFF_SRC"};
        const std::string DIFF_SCALE {TENSOR_ID + "_DIFF_SCALE"};
        const std::string DIFF_SHIFT {TENSOR_ID + "_DIFF_SHIFT"};
        tensor_descs_.emplace(DIFF_DST, bnorm_dt, dims, lt::strided);
        tensor_descs_.emplace(DIFF_SRC, bnorm_dt, dims, lt::strided);
        tensor_descs_.emplace(DIFF_SCALE, graph_dt::f32, s_dims, lt::strided);
        tensor_descs_.emplace(DIFF_SHIFT, graph_dt::f32, s_dims, lt::strided);
        inputs = {tensor_descs_[SRC], tensor_descs_[DIFF_DST],
                tensor_descs_[SCALE], tensor_descs_[MEAN], tensor_descs_[VAR]};
        outputs = {tensor_descs_[DIFF_SRC], tensor_descs_[DIFF_SCALE],
                tensor_descs_[DIFF_SHIFT]};
    }

    op bnorm_op(new_op_id, op_kind, inputs, outputs, op_name);

    bnorm_op.set_attr("epsilon", prb->eps);
    bnorm_op.set_attr("data_format", std::string("NCX"));

    ops_.emplace_back(bnorm_op);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

fill_status_t bnorm_graph_prb_t::handle_elt_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.bnorm.eltw_handler(*this, po_entry);
}

int doit(const ::bnorm::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

    bnorm_graph_prb_t graph_prb(prb);
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

    const bool is_fwd = prb->dir & FLAG_FWD;
    const bool use_ss = prb->use_ss();
    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::bnorm::init_pd, prb, res, par, ins, outs);

    static const engine_t cpu_engine(dnnl_cpu);

    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto shift_fp = make_dnn_mem(
            is_fwd ? ins[2] : outs[2], dt::f32, use_sh ? tag::x : tag::axb);
    auto mean_fp = make_dnn_mem(ins[3], dt::f32, tag::abx);
    auto var_fp = make_dnn_mem(ins[4], dt::f32, tag::abx);
    dnn_mem_t &dst_fp = src_fp; // in-place reference
    auto src_hat_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto ws_fp = make_dnn_mem(ins[0], dt::u8, tag::abx);

    const auto placeholder_dst_dt = make_dnn_mem(outs[0], tag::abx);
    auto src_dt = make_dnn_mem(ins[0], tag::abx);
    auto shift_dt = make_dnn_mem(
            is_fwd ? ins[2] : outs[2], use_sh ? tag::x : tag::axb);
    auto mean_dt = make_dnn_mem(ins[3], tag::abx);
    auto var_dt = make_dnn_mem(ins[4], tag::abx);
    const dnn_mem_t &dst_dt = prb->inplace ? src_dt : placeholder_dst_dt;

    dnn_mem_t scale_fp, scale_dt, d_shift_dt, d_scale_dt;
    if (use_sc || use_sh) {
        scale_fp = make_dnn_mem(is_fwd ? ins[1] : ins[2], dt::f32, tag::abx);
        scale_dt = make_dnn_mem(is_fwd ? ins[1] : ins[2], dt::f32, tag::abx);
    } else {
        // scale and shift are combined in a single 2D tensor of shape 2xC
        // this logical_tensor is used to indicate the memory size for scale
        dnnl::graph::logical_tensor lt {0, dt::f32, {2, prb->ic},
                dnnl::graph::logical_tensor::layout_type::strided};
        scale_fp = make_dnn_mem(lt, dt::f32, tag::abx);
        scale_dt = make_dnn_mem(lt, dt::f32, tag::abx);
    }

    dnn_mem_t d_dst_dt, placeholder_d_src_dt;
    dnn_mem_t r_mean_dt, r_var_dt, b_mean_dt, b_var_dt;

    if (::bnorm::prepare_fwd(prb, src_fp, mean_fp, var_fp, scale_fp, shift_fp)
            != OK) {
        return res->state = MISTRUSTED, OK;
    }
    /*  When dnnl_use_scaleshift is used, benchdnn populates data
        to the same memory for scale and shift and dnnlgraph expects
        the data in scale and shift. Hence this explicit copy. */
    if (!(use_sc || use_sh)) {
        for (int64_t i = 0; i < prb->ic; i++) {
            shift_fp.set_elem(i, scale_fp.get_elem(prb->ic + i));
        }
    }
    SAFE(src_dt.reorder(src_fp), WARN);
    SAFE(scale_dt.reorder(scale_fp), WARN);
    SAFE(shift_dt.reorder(shift_fp), WARN);
    SAFE(mean_dt.reorder(mean_fp), WARN);
    SAFE(var_dt.reorder(var_fp), WARN);

    std::vector<dnnl::graph::tensor> tensors_in, tensors_out;
    const dnnl::graph::engine &eng = get_test_engine();
    args_t args, ref_args;

    if (prb->dir & FLAG_FWD) {
        tensors_in.emplace_back(ins[0], eng, static_cast<void *>(src_dt));
        tensors_in.emplace_back(ins[1], eng, static_cast<void *>(scale_dt));
        tensors_in.emplace_back(ins[2], eng, static_cast<void *>(shift_dt));
        tensors_in.emplace_back(ins[3], eng, static_cast<void *>(mean_dt));
        tensors_in.emplace_back(ins[4], eng, static_cast<void *>(var_dt));
        tensors_out.emplace_back(outs[0], eng, static_cast<void *>(dst_dt));
        if (!(prb->dir & FLAG_INF)) {
            r_mean_dt = make_dnn_mem(outs[1], tag::abx);
            r_var_dt = make_dnn_mem(outs[2], tag::abx);
            b_mean_dt = make_dnn_mem(outs[3], tag::abx);
            b_var_dt = make_dnn_mem(outs[4], tag::abx);
            tensors_out.emplace_back(
                    outs[1], eng, static_cast<void *>(r_mean_dt));
            tensors_out.emplace_back(
                    outs[2], eng, static_cast<void *>(r_var_dt));
            tensors_out.emplace_back(
                    outs[3], eng, static_cast<void *>(b_mean_dt));
            tensors_out.emplace_back(
                    outs[4], eng, static_cast<void *>(b_var_dt));
        }
        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_MEAN, mean_fp);
            ref_args.set(DNNL_ARG_VARIANCE, var_fp);
            ref_args.set(
                    use_sc ? DNNL_ARG_SCALE : DNNL_ARG_SCALE_SHIFT, scale_fp);
            ref_args.set(DNNL_ARG_SHIFT, shift_fp);
            ref_args.set(DNNL_ARG_WORKSPACE, ws_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(DNNL_ARG_DST_1, src_hat_fp); // Reference aux arg.

            SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);
            args.set(DNNL_ARG_SRC, src_dt);
            args.set(DNNL_ARG_MEAN, mean_dt);
            args.set(DNNL_ARG_VARIANCE, var_dt);
            std::vector<data_kind_t> kinds {DST};
            if (!(prb->flags & ::bnorm::GLOB_STATS) && !(prb->dir & FLAG_INF)) {
                kinds.push_back(MEAN);
                kinds.push_back(VAR);
            }

            check_correctness(
                    prb, kinds, args, ref_args, ::bnorm::setup_cmp, res);
            return OK;
        }
    } else {
        auto d_dst_fp = make_dnn_mem(ins[1], dt::f32, tag::abx);
        dnn_mem_t &d_src_fp = d_dst_fp; // in-place in ref code

        d_dst_dt = make_dnn_mem(ins[1], tag::abx);
        if (!prb->inplace) {
            placeholder_d_src_dt = make_dnn_mem(outs[0], tag::abx);
        }
        dnn_mem_t &d_src_dt = prb->inplace ? d_dst_dt : placeholder_d_src_dt;

        SAFE(::bnorm::prepare_bwd(prb, d_dst_dt, d_dst_fp), WARN);

        dnn_mem_t d_scale_fp;
        if (use_sc || use_sh) {
            d_scale_fp = make_dnn_mem(outs[1], dt::f32, tag::abx);
            d_scale_dt = make_dnn_mem(outs[1], dt::f32, tag::abx);
        } else {
            // scale and shift are combined in a single 2D tensor of shape 2xC
            // this logical_tensor is used to indicate the memory size for scale
            dnnl::graph::logical_tensor lt {0, dt::f32, {2, prb->ic},
                    dnnl::graph::logical_tensor::layout_type::strided};
            d_scale_fp = make_dnn_mem(lt, dt::f32, tag::abx);
            d_scale_dt = make_dnn_mem(lt, dt::f32, tag::abx);
        }
        auto d_shift_fp
                = make_dnn_mem(outs[2], dt::f32, use_sh ? tag::x : tag::axb);
        d_shift_dt = make_dnn_mem(outs[2], use_sh ? tag::x : tag::axb);

        tensors_in.emplace_back(ins[0], eng, static_cast<void *>(src_dt));
        tensors_in.emplace_back(ins[1], eng, static_cast<void *>(d_dst_dt));
        tensors_in.emplace_back(ins[2], eng, static_cast<void *>(scale_dt));
        tensors_in.emplace_back(ins[3], eng, static_cast<void *>(mean_dt));
        tensors_in.emplace_back(ins[4], eng, static_cast<void *>(var_dt));
        tensors_out.emplace_back(outs[0], eng, static_cast<void *>(d_src_dt));
        tensors_out.emplace_back(outs[1], eng, static_cast<void *>(d_scale_dt));
        tensors_out.emplace_back(outs[2], eng, static_cast<void *>(d_shift_dt));

        if (is_bench_mode(CORR)) {
            SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_MEAN, mean_fp);
            ref_args.set(DNNL_ARG_VARIANCE, var_fp);
            ref_args.set(
                    use_sc ? DNNL_ARG_SCALE : DNNL_ARG_SCALE_SHIFT, scale_fp);
            ref_args.set(DNNL_ARG_SHIFT, shift_fp);
            ref_args.set(DNNL_ARG_WORKSPACE, ws_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(DNNL_ARG_DST_1, src_hat_fp); // Reference aux arg.
            ref_args.set(DNNL_ARG_DIFF_DST, d_dst_fp);
            ref_args.set(DNNL_ARG_DIFF_SRC, d_src_fp);
            ref_args.set(
                    use_sc ? DNNL_ARG_DIFF_SCALE : DNNL_ARG_DIFF_SCALE_SHIFT,
                    d_scale_fp);
            ref_args.set(DNNL_ARG_DIFF_SHIFT, d_shift_fp);

            args.set(DNNL_ARG_SRC, src_dt);
            args.set(DNNL_ARG_MEAN, mean_dt);
            args.set(DNNL_ARG_VARIANCE, var_dt);
            args.set(DNNL_ARG_DIFF_DST, d_dst_dt);
            args.set(use_sc ? DNNL_ARG_SCALE : DNNL_ARG_SCALE_SHIFT, scale_dt);
            args.set(DNNL_ARG_SHIFT, shift_dt);
            args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
            args.set(use_sc ? DNNL_ARG_DIFF_SCALE : DNNL_ARG_DIFF_SCALE_SHIFT,
                    d_scale_dt);
            args.set(DNNL_ARG_DIFF_SHIFT, d_shift_dt);

            std::vector<data_kind_t> kinds {SRC};
            if ((use_ss || use_sc) && (prb->dir & FLAG_WEI)) {
                kinds.push_back(use_sc ? SC : SS);
            }
            if (use_sh && (prb->dir & FLAG_WEI)) kinds.push_back(SH);
            check_correctness(
                    prb, kinds, args, ref_args, ::bnorm::setup_cmp, res);
            return OK;
        }
    }

    SAFE(measure_perf(
                 res->timer_map.perf_timer(), cp, tensors_in, tensors_out, res),
            WARN);

    return OK;
}
} // namespace bnorm
} // namespace benchdnnext
