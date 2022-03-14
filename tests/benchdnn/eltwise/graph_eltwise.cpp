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

eltwise_graph_prb_t::spec_t::spec_t(const ::eltwise::prb_t *prb) noexcept {
    dims = prb->dims;
    eltwise_dt = convert_dt(prb->dt);
    data_format = convert_tag(prb->tag);
    raw_data_format = prb->tag;
    is_fwd_pass = prb->dir & FLAG_FWD;
    use_dst = prb->use_dst();

    auto dnnl_kind = attr_t::post_ops_t::kind2dnnl_kind(prb->alg);
    op_kind = convert_alg_kind(dnnl_kind, is_fwd_pass);

    alpha = prb->alpha;
    beta = prb->beta;

    if (dnnl_kind == dnnl_eltwise_soft_relu)
        softplus_beta = 1;
    else if (dnnl_kind == dnnl_eltwise_logsigmoid)
        softplus_beta = -1;
}

void check_known_skipped_case_graph(
        const ::eltwise::prb_t *prb, res_t *res) noexcept {
    ::eltwise::check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return;

    check_graph_eltwise_params(res, prb->alg, prb->alpha, prb->beta);
    if (res->state == SKIPPED) return;
}

fill_status_t eltwise_graph_prb_t::handle_main_op_() {
    using op = dnnl::graph::op;

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);

    dt data_type;
    if (benchdnnext::is_low_precision({spec_.eltwise_dt})) {
        data_type = dt::f32;
    } else {
        data_type = spec_.eltwise_dt;
    }

    std::vector<dnnl::graph::logical_tensor> ltensors_in;
    std::vector<dnnl::graph::logical_tensor> ltensors_out;

    const std::string SRC {TENSOR_ID + "_SRC"};
    tensor_descs_.emplace(SRC, data_type, spec_.dims, spec_.raw_data_format);
    ltensors_in.push_back({tensor_descs_[SRC]});

    if (spec_.is_fwd_pass) {
        const std::string DST {TENSOR_ID + "_DST"};
        tensor_descs_.emplace(
                DST, data_type, spec_.dims, spec_.raw_data_format);
        ltensors_out.push_back({tensor_descs_[DST]});
    } else {
        const std::string DIFF_SRC {TENSOR_ID + "_DIFF_SRC"};
        const std::string DIFF_DST {TENSOR_ID + "_DIFF_DST"};
        tensor_descs_.emplace(
                DIFF_SRC, data_type, spec_.dims, spec_.raw_data_format);
        tensor_descs_.emplace(
                DIFF_DST, data_type, spec_.dims, spec_.raw_data_format);
        ltensors_in.push_back({tensor_descs_[DIFF_DST]});
        ltensors_out.push_back({tensor_descs_[DIFF_SRC]});
    }

    op eltwise_op(
            new_op_id, spec_.op_kind, ltensors_in, ltensors_out, "eltwise");

    //Set alpha, beta, min and max for relevant ops
    switch (spec_.op_kind) {
        case dnnl::graph::op::kind::Elu:
            eltwise_op.set_attr("alpha", spec_.alpha);
            break;
        case dnnl::graph::op::kind::EluBackprop:
            eltwise_op.set_attr("alpha", spec_.alpha);
            eltwise_op.set_attr("use_dst", spec_.use_dst);
            break;
        case dnnl::graph::op::kind::ReLUBackprop:
        case dnnl::graph::op::kind::SigmoidBackprop:
        case dnnl::graph::op::kind::SqrtBackprop:
        case dnnl::graph::op::kind::TanhBackprop:
            eltwise_op.set_attr("use_dst", spec_.use_dst);
            break;
        case dnnl::graph::op::kind::HardTanh:
            eltwise_op.set_attr("min", spec_.alpha);
            eltwise_op.set_attr("max", spec_.beta);
            break;
        case dnnl::graph::op::kind::HardTanhBackprop:
            eltwise_op.set_attr("min", spec_.alpha);
            eltwise_op.set_attr("max", spec_.beta);
            // Since backend uses clp_v2 for HardTanhBackprop
            eltwise_op.set_attr("use_dst", spec_.use_dst);
            break;
        case dnnl::graph::op::kind::SoftPlus:
        case dnnl::graph::op::kind::SoftPlusBackprop:
            eltwise_op.set_attr("beta", spec_.softplus_beta);
            break;
        default: break;
    }

    ops_.emplace_back(eltwise_op);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

fill_status_t eltwise_graph_prb_t::handle_bin_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.eltwise.bin_handler(*this, spec_.data_format, po_entry);
}

fill_status_t eltwise_graph_prb_t::handle_low_precision_(
        const ::eltwise::prb_t *prb_) {
    low_precision_attr lp_attr = low_precision_attr::lp_attr(
            spec_.eltwise_dt, spec_.eltwise_dt, spec_.raw_data_format);

    fill_status_t ctor_status;
    ctor_status
            = po_handler.pool.low_precision_handler.handle_low_precision_src(
                    *this, lp_attr);
    if (ctor_status != fill_status::DONE) return ctor_status;

    ctor_status
            = po_handler.pool.low_precision_handler.handle_low_precision_dst(
                    *this, lp_attr);
    if (ctor_status != fill_status::DONE) return ctor_status;

    if (has_post_bin()) {
        ctor_status = po_handler.pool.low_precision_handler
                              .handle_low_precision_post_bin(*this, lp_attr,
                                      prb_->attr.post_ops.entry);
    }

    return ctor_status;
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

    const bool is_fwd = prb->dir & FLAG_FWD;
    const dnn_mem_t &arg_fp = !is_fwd && prb->use_dst() ? dst_fp : src_fp;

    // Shouldn't be defined inside since not available when `eltwise_add_check`
    // is invoked due to removed from stack.
    const float trh
            = ::eltwise::get_eltwise_threshold(prb->dt, prb->alg, is_fwd);
    compare::compare_t cmp;
    if (is_bench_mode(CORR)) {
        cmp.set_threshold(trh);
        cmp.set_zero_trust_percent(
                ::eltwise::get_eltwise_zero_trust_percent(prb));

        const auto eltwise_add_check =
                [&](const compare::compare_t::driver_check_func_args_t &args) {
                    // Some algorithms require absolute value comparison for inputs
                    // where catastrophic cancellation may happen.
                    const float src = arg_fp.get_elem(args.idx);
                    if (::eltwise::check_abs_err(prb, src, trh))
                        return args.diff <= trh;
                    if (prb->attr.post_ops.binary_index() != -1)
                        return args.diff <= trh;
                    return false;
                };
        cmp.set_driver_check_function(eltwise_add_check);
    }

    dnnl::graph::engine &eng = get_test_engine();
    std::vector<dnnl::graph::tensor> tensors_in, tensors_out;
    args_t ref_args;

    if (is_fwd) {
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
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(binary_po_args, binary_po_fp);

            TIME_REF(::eltwise::compute_ref(prb, ref_args));
            SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
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
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, d_dst_fp);
            ref_args.set(DNNL_ARG_DIFF_SRC, d_src_fp);
            ::eltwise::compute_ref(prb, ref_args);
            if (prb->use_dst()) SAFE(dst_dt.reorder(dst_fp), WARN);
            SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);
            SAFE(cmp.compare(d_src_fp, d_src_dt, prb->attr, res), WARN);
        }
        SAFE(measure_perf(
                     res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
                WARN);
        return OK;
    }
}
} // namespace eltwise
} // namespace benchdnnext
