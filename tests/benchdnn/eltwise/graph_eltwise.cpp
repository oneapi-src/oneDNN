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

#include "utils/compare.hpp"

#include "binary/binary.hpp"
#include "eltwise/graph_eltwise.hpp"

namespace benchdnnext {
namespace eltwise {

static void check_known_skipped_case_graph(
        const ::eltwise::prb_t *prb, res_t *res) noexcept {
    // TODO: to align with original benchdnn, we should consider moving
    // skip_unimplemented_prb call after compilation step
    skip_invalid_and_unimplemented_prb(prb, res);
    if (res->state == SKIPPED) return;

    check_graph_eltwise_params(res, prb->alg, prb->alpha, prb->beta);
    if (res->state == SKIPPED) return;
}

static quant_data_t get_qdata_for(int arg, const ::eltwise::prb_t *prb) {
    if (arg == SRC || arg == DST)
        return quant_data_t(convert_dt(prb->dt), prb->tag);

    BENCHDNN_PRINT(
            0, "warning: returning default quant_data_t for arg: %d\n", arg);
    return quant_data_t();
}

static quant_data_t get_qdata_for(
        const attr_t::post_ops_t::entry_t &entry, const ::eltwise::prb_t *prb) {
    return bin_po_entry2quant_data(entry, prb->tag, convert_dt(prb->dt));
}

fill_status_t append_graph_with_block(const ::eltwise::prb_t *prb) {
    graph_t &graph = graph_t::get();

    const auto with_dq = is_low_precision({convert_dt(prb->dt)});
    const auto connect_to_previous_block = !with_dq && graph.has_blocks();

    // handle main op
    const auto op_id = graph.generate_id_for(entry_kind::ELTWISE);
    const auto src_id = connect_to_previous_block
            ? graph.get_last_block_out_id()
            : graph.generate_id_for(op_id, lt_kind::SRC);

    auto common_dt = dequantize_dtype(convert_dt(prb->dt));

    graph.create_lt(src_id, common_dt, prb->dims, prb->tag);
    std::vector<size_t> src_ids {src_id};
    std::vector<size_t> dst_ids {};

    if (prb->dir & FLAG_FWD) {
        const auto dst_id = graph.generate_id_for(op_id, lt_kind::DST);
        graph.create_lt(dst_id, common_dt, prb->dims, prb->tag);
        dst_ids.push_back(dst_id);
    } else {
        const auto diff_src_id
                = graph.generate_id_for(op_id, lt_kind::DIFF_SRC);
        const auto diff_dst_id
                = graph.generate_id_for(op_id, lt_kind::DIFF_DST);
        graph.create_lt(diff_src_id, common_dt, prb->dims, prb->tag);
        graph.create_lt(diff_dst_id, common_dt, prb->dims, prb->tag);
        src_ids.push_back(diff_src_id);
        dst_ids.push_back(diff_dst_id);
    }

    const auto dnnl_kind = attr_t::post_ops_t::kind2dnnl_kind(prb->alg);
    const auto op_kind = convert_alg_kind(dnnl_kind, prb->dir & FLAG_FWD);
    int64_t softplus_beta = 0;
    if (dnnl_kind == dnnl_eltwise_soft_relu)
        softplus_beta = 1;
    else if (dnnl_kind == dnnl_eltwise_logsigmoid)
        softplus_beta = -1;

    dnnl::graph::op eltw_op(op_id, op_kind, graph.stringify_id(op_id));

    // set alpha, beta, min and max for relevant ops
    switch (op_kind) {
        case dnnl::graph::op::kind::Elu:
            eltw_op.set_attr("alpha", prb->alpha);
            break;
        case dnnl::graph::op::kind::EluBackprop:
            eltw_op.set_attr("alpha", prb->alpha);
            eltw_op.set_attr("use_dst", prb->use_dst());
            break;
        case dnnl::graph::op::kind::ReLUBackprop:
        case dnnl::graph::op::kind::SigmoidBackprop:
        case dnnl::graph::op::kind::SqrtBackprop:
        case dnnl::graph::op::kind::TanhBackprop:
            eltw_op.set_attr("use_dst", prb->use_dst());
            break;
        case dnnl::graph::op::kind::Clamp:
            eltw_op.set_attr("min", prb->alpha);
            eltw_op.set_attr("max", prb->beta);
            break;
        case dnnl::graph::op::kind::ClampBackprop:
            eltw_op.set_attr("min", prb->alpha);
            eltw_op.set_attr("max", prb->beta);
            // Since backend uses clp_v2 for ClampBackprop
            eltw_op.set_attr("use_dst", prb->use_dst());
            break;
        case dnnl::graph::op::kind::SoftPlus:
        case dnnl::graph::op::kind::SoftPlusBackprop:
            eltw_op.set_attr("beta", softplus_beta);
            break;
        default: break;
    }

    graph.append(op_id, eltw_op, src_ids, dst_ids);

    fill_status_t status;
    // if required - apply dequantize to block inputs
    if (with_dq) {
        status = insert_dequant_before(src_id, get_qdata_for(SRC, prb));
        BENCHDNNEXT_VERIFY(status);
    }

    // handle post ops
    for (const auto &entry : prb->attr.post_ops.entry) {
        const auto with_src1_dq = is_dequantize_required_for(entry);
        size_t po_src1_id;
        if (entry.is_binary_kind()) {
            std::tie(status, po_src1_id) = append_graph_with_binary(entry);
            BENCHDNNEXT_VERIFY(status);
            if (with_src1_dq) {
                status = insert_dequant_before(
                        po_src1_id, get_qdata_for(entry, prb));
                BENCHDNNEXT_VERIFY(status);
            }
        }
    }

    // if required - add quantize op
    if (with_dq) {
        status = insert_quant_after(
                graph.get_cur_block_out_id(), get_qdata_for(DST, prb));
        BENCHDNNEXT_VERIFY(status);
    }

    graph.close_block();

    return fill_status::DONE;
}

static void graph_bwd_check_correctness(const ::eltwise::prb_t *prb,
        const args_t &args, const args_t &ref_args, res_t *res) {
    compare::compare_t cmp;
    cmp.set_data_kind(SRC);
    ::eltwise::setup_cmp(cmp, prb, SRC, ref_args);

    const int arg = DNNL_ARG_DIFF_SRC;
    const auto &mem_dt = args.find(arg);
    const auto &mem_fp = ref_args.find(arg);

    cmp.compare(mem_fp, mem_dt, prb->attr, res);
}

int doit(const ::eltwise::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

    const auto status = append_graph_with_block(prb);
    if (status != fill_status::DONE
            && status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        cleanup();
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto &graph = graph_t::get();

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

    auto cp = compile_partition(::eltwise::init_pd, prb, res, par, ins, outs);

    static const engine_t cpu_engine(dnnl_cpu);

    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    // we need src_fp for proper comparison, => no in-place reference
    auto dst_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);

    dnn_mem_t placeholder_dst_dt = [&outs, &prb] {
        if (!prb->inplace) {
            return make_dnn_mem(outs[0], (prb->tag).c_str());
        } else {
            return dnn_mem_t();
        }
    }();

    auto src_dt = make_dnn_mem(ins[0], (prb->tag).c_str());
    dnn_mem_t &dst_dt = prb->inplace ? src_dt : placeholder_dst_dt;
    // eltwise operator supports only relu-add (single binary post-op)
    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    if (!prb->attr.post_ops.entry.empty()) {
        binary_po_fp.emplace_back(
                make_dnn_mem(ins.back(), dt::f32, (prb->tag).c_str()));
        binary_po_dt.emplace_back(make_dnn_mem(ins.back(), (prb->tag).c_str()));
        const int po_idx = DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1;
        ::binary::fill_mem(po_idx, binary_po_dt.back(), binary_po_fp.back());
        binary_po_args.push_back(po_idx);
    }

    SAFE(::eltwise::fill_data(prb, SRC, src_dt, src_fp), WARN);

    const dnnl::graph::engine &eng = get_test_engine();
    std::vector<dnnl::graph::tensor> tensors_in, tensors_out;
    args_t args, ref_args;

    if (prb->dir & FLAG_FWD) {
        tensors_in.emplace_back(
                dnnl::graph::tensor(ins[0], eng, static_cast<void *>(src_dt)));
        tensors_out.emplace_back(
                dnnl::graph::tensor(outs[0], eng, static_cast<void *>(dst_dt)));
        if (!prb->attr.post_ops.entry.empty()) {
            tensors_in.emplace_back(dnnl::graph::tensor(
                    ins.back(), eng, static_cast<void *>(binary_po_dt.back())));
        }
        SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

        if (is_bench_mode(CORR)) {
            args.set(DNNL_ARG_DST, dst_dt);

            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(binary_po_args, binary_po_fp);

            check_correctness(
                    prb, {DST}, args, ref_args, ::eltwise::setup_cmp, res);
        }
        SAFE(measure_perf(
                     res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
                WARN);
    } else {
        if (prb->use_dst()) {
            tensors_in.emplace_back(dnnl::graph::tensor(
                    ins[0], eng, static_cast<void *>(dst_dt)));
        } else {
            tensors_in.emplace_back(dnnl::graph::tensor(
                    ins[0], eng, static_cast<void *>(src_dt)));
        }

        auto d_dst_fp = make_dnn_mem(ins[1], dt::f32, tag::abx);
        auto d_dst_dt = make_dnn_mem(ins[1], (prb->tag).c_str());
        SAFE(::eltwise::fill_data(prb, DST, d_dst_dt, d_dst_fp), WARN);
        tensors_in.emplace_back(dnnl::graph::tensor(
                ins[1], eng, static_cast<void *>(d_dst_dt)));

        dnn_mem_t d_src_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);
        // dnn_mem_t &d_src_fp = d_dst_fp; // in-place reference
        dnn_mem_t placeholder_d_src_dt = [&outs, &prb] {
            if (!prb->inplace) {
                return make_dnn_mem(outs[0], (prb->tag).c_str());
            } else {
                return dnn_mem_t();
            }
        }();
        dnn_mem_t &d_src_dt = prb->inplace ? d_dst_dt : placeholder_d_src_dt;
        tensors_out.emplace_back(dnnl::graph::tensor(
                outs[0], eng, static_cast<void *>(d_src_dt)));
        if (is_bench_mode(CORR)) {
            args.set(DNNL_ARG_DIFF_SRC, d_src_dt);

            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, d_dst_fp);
            ref_args.set(DNNL_ARG_DIFF_SRC, d_src_fp);

            ::eltwise::compute_ref(prb, ref_args);

            if (prb->use_dst()) SAFE(dst_dt.reorder(dst_fp), WARN);
            SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

            graph_bwd_check_correctness(prb, args, ref_args, res);
        }
        SAFE(measure_perf(
                     res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
                WARN);
    }

    cleanup();

    return OK;
}

} // namespace eltwise
} // namespace benchdnnext
