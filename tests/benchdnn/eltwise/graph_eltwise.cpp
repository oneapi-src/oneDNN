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

void check_known_skipped_case_graph(
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

fill_status_t eltwise_graph_prb_t::handle_main_op_(
        const ::eltwise::prb_t *prb) {
    using op = dnnl::graph::op;

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);

    auto common_dt = benchdnnext::set_main_op_dtype(convert_dt(prb->dt));

    std::vector<dnnl::graph::logical_tensor> ltensors_in;
    std::vector<dnnl::graph::logical_tensor> ltensors_out;

    const std::string SRC {TENSOR_ID + "_SRC"};
    tensor_descs_.emplace(SRC, common_dt, prb->dims, prb->tag);
    ltensors_in.push_back({tensor_descs_[SRC]});

    if (prb->dir & FLAG_FWD) {
        const std::string DST {TENSOR_ID + "_DST"};
        tensor_descs_.emplace(DST, common_dt, prb->dims, prb->tag);
        ltensors_out.push_back({tensor_descs_[DST]});
    } else {
        const std::string DIFF_SRC {TENSOR_ID + "_DIFF_SRC"};
        const std::string DIFF_DST {TENSOR_ID + "_DIFF_DST"};
        tensor_descs_.emplace(DIFF_SRC, common_dt, prb->dims, prb->tag);
        tensor_descs_.emplace(DIFF_DST, common_dt, prb->dims, prb->tag);
        ltensors_in.push_back({tensor_descs_[DIFF_DST]});
        ltensors_out.push_back({tensor_descs_[DIFF_SRC]});
    }

    const auto dnnl_kind = attr_t::post_ops_t::kind2dnnl_kind(prb->alg);
    const auto op_kind = convert_alg_kind(dnnl_kind, prb->dir & FLAG_FWD);
    int64_t softplus_beta = 0;
    if (dnnl_kind == dnnl_eltwise_soft_relu)
        softplus_beta = 1;
    else if (dnnl_kind == dnnl_eltwise_logsigmoid)
        softplus_beta = -1;

    op eltwise_op(new_op_id, op_kind, ltensors_in, ltensors_out, "eltwise");

    //Set alpha, beta, min and max for relevant ops
    switch (op_kind) {
        case dnnl::graph::op::kind::Elu:
            eltwise_op.set_attr("alpha", prb->alpha);
            break;
        case dnnl::graph::op::kind::EluBackprop:
            eltwise_op.set_attr("alpha", prb->alpha);
            eltwise_op.set_attr("use_dst", prb->use_dst());
            break;
        case dnnl::graph::op::kind::ReLUBackprop:
        case dnnl::graph::op::kind::SigmoidBackprop:
        case dnnl::graph::op::kind::SqrtBackprop:
        case dnnl::graph::op::kind::TanhBackprop:
            eltwise_op.set_attr("use_dst", prb->use_dst());
            break;
        case dnnl::graph::op::kind::HardTanh:
            eltwise_op.set_attr("min", prb->alpha);
            eltwise_op.set_attr("max", prb->beta);
            break;
        case dnnl::graph::op::kind::HardTanhBackprop:
            eltwise_op.set_attr("min", prb->alpha);
            eltwise_op.set_attr("max", prb->beta);
            // Since backend uses clp_v2 for HardTanhBackprop
            eltwise_op.set_attr("use_dst", prb->use_dst());
            break;
        case dnnl::graph::op::kind::SoftPlus:
        case dnnl::graph::op::kind::SoftPlusBackprop:
            eltwise_op.set_attr("beta", softplus_beta);
            break;
        default: break;
    }

    ops_.emplace_back(eltwise_op);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

fill_status_t eltwise_graph_prb_t::handle_bin_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.eltwise.bin_handler(*this, po_entry);
}

fill_status_t eltwise_graph_prb_t::handle_low_precision_(
        const ::eltwise::prb_t *prb) {
    const std::string OP_REPR = "main";
    const auto src_lt_id = tensor_id[OP_REPR].back() + "_SRC";
    const auto dst_lt_id = curr_out_map_ids_.back() + "_DST";

    fill_status_t status
            = po_handler.eltwise.low_precision_handler.insert_dequant_before(
                    src_lt_id, get_qdata_for(SRC, prb), *this);
    BENCHDNNEXT_VERIFY(status);

    status = po_handler.eltwise.low_precision_handler.insert_quant_after(
            dst_lt_id, get_qdata_for(DST, prb), *this);
    BENCHDNNEXT_VERIFY(status);

    for (const auto &entry : prb->attr.post_ops.entry) {
        if (entry.is_binary_kind()) {
            const auto bin_src1_lt_id = tensor_id["binary"].back() + "_SRC";
            status = po_handler.eltwise.low_precision_handler
                             .insert_dequant_before(bin_src1_lt_id,
                                     bin_po_entry2quant_data(entry, prb->tag,
                                             convert_dt(prb->dt)),
                                     *this);
            BENCHDNNEXT_VERIFY(status);
            break;
        }
    }

    return status;
}

void graph_bwd_check_correctness(const ::eltwise::prb_t *prb,
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

    eltwise_graph_prb_t graph_prb(prb);
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
    if (graph_prb.has_post_bin()) {
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
        if (graph_prb.has_post_bin()) {
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

        return OK;
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
        return OK;
    }
}
} // namespace eltwise
} // namespace benchdnnext
