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
#include "oneapi/dnnl/dnnl_types.h"

#include "dnnl_graph_common.hpp"
#include "utils/compare.hpp"

#include "binary/binary.hpp"
#include "matmul/graph_matmul.hpp"

#include <algorithm>

namespace benchdnnext {
namespace matmul {

matmul_graph_prb_t::spec_t::spec_t(const ::matmul::prb_t *prb) noexcept {
    src_dims = get_runtime_dims(prb->src_dims(), prb->src_runtime_dim_mask());
    wei_dims = get_runtime_dims(
            prb->weights_dims(), prb->weights_runtime_dim_mask());
    dst_dims = get_runtime_dims(prb->dst_dims(), prb->dst_runtime_dim_mask());
    if (dt::undef != convert_dt(prb->bia_dt)) {
        bia_dims.resize(dst_dims.size());
        for (int i = 0; i < prb->ndims; ++i)
            bia_dims[i] = (prb->bia_mask & (1 << i)) ? dst_dims[i] : 1;
        bia_dims = get_runtime_dims(bia_dims, prb->dst_runtime_dim_mask());
    }

    src_dt = convert_dt(prb->cfg[SRC].dt);
    wei_dt = convert_dt(prb->cfg[WEI].dt);
    dst_dt = convert_dt(prb->cfg[DST].dt);
    bia_dt = convert_dt(prb->bia_dt);

    data_format = "NCX";

    raw_src_tag = prb->stag;
    raw_wei_tag = prb->stag;
    raw_dst_tag = prb->stag;
}

void check_known_skipped_case_graph(
        const ::matmul::prb_t *prb, res_t *res) noexcept {
    ::matmul::check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return;

    for (const auto &po : prb->attr.post_ops.entry) {
        if (po.is_eltwise_kind()) {
            // for swish, alpha is always set to 1.0 in the oneDNN graph
            // it's because swish is represented as Multiply+Sigmoid
            // and Sigmoid don't have either alpha or beta
            const auto is_swish = po.kind == attr_t::post_ops_t::SWISH;
            const auto alpha_differs_from_one
                    = std::fabs(1.0 - po.eltwise.alpha) > 1.0e-05;
            if (is_swish && alpha_differs_from_one) {
                res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
                return;
            }
        }
    }
}

fill_status_t matmul_graph_prb_t::handle_main_op_() {
    using op = dnnl::graph::op;

    const bool with_bias = has_post_bia();
    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);

    // this is needed to align with po_handlers convention
    // some patterns like `matmul + bias + swish` may want to
    // reuse bias output via `tensor_id["bias"].back() + "_DST"`
    if (with_bias) tensor_id["bias"].push_back(TENSOR_ID);

    const std::string SRC {TENSOR_ID + "_SRC"};
    const std::string WEI {TENSOR_ID + "_WEI"};
    const std::string BIA {TENSOR_ID + "_BIA"};
    const std::string DST {TENSOR_ID + "_DST"};

    const auto is_lprec = is_low_precision(get_dtypes());
    const auto with_tc = with_typecast(get_dtypes());
    const auto change_dt = is_lprec || with_tc;
    const auto default_dt = (with_tc) ? dt::bf16 : dt::f32;
    dt src_dt = (change_dt) ? default_dt : spec_.src_dt;
    dt wei_dt = (change_dt) ? default_dt : spec_.wei_dt;
    dt dst_dt = (change_dt) ? default_dt : spec_.dst_dt;
    tensor_descs_.emplace(SRC, src_dt, spec_.src_dims, spec_.raw_src_tag);
    tensor_descs_.emplace(WEI, wei_dt, spec_.wei_dims, spec_.raw_wei_tag);
    tensor_descs_.emplace(DST, dst_dt, spec_.dst_dims, spec_.raw_dst_tag);
    if (with_bias) {
        tensor_descs_.emplace(BIA, spec_.bia_dt, spec_.bia_dims, lt::strided);
    }

    std::vector<dnnl::graph::logical_tensor> lt_inputs {
            tensor_descs_[SRC], tensor_descs_[WEI]};
    std::vector<dnnl::graph::logical_tensor> lt_outputs {tensor_descs_[DST]};
    if (with_bias) lt_inputs.push_back(tensor_descs_[BIA]);

    op matmul(new_op_id, op::kind::MatMul, lt_inputs, lt_outputs, "matmul");

    matmul.set_attr("transpose_a", spec_.transpose_a)
            .set_attr("transpose_b", spec_.transpose_b);

    ops_.emplace_back(matmul);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

fill_status_t matmul_graph_prb_t::handle_elt_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.matmul.eltw_handler(*this, po_entry);
}

fill_status_t matmul_graph_prb_t::handle_bin_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.matmul.bin_handler(*this, spec_.data_format, po_entry);
}

fill_status_t matmul_graph_prb_t::handle_sum_() {
    return po_handler.matmul.sum_handler(*this);
}

fill_status_t matmul_graph_prb_t::handle_typecast_(
        const ::matmul::prb_t *prb_) {
    using op = dnnl::graph::op;

    const std::string SRC = tensor_id["main"].back() + "_SRC";
    const std::string WEI = tensor_id["main"].back() + "_WEI";

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["typecast"].push_back(TENSOR_ID);
    const std::string TCSRC {TENSOR_ID + "_SRC"};
    const std::string TCWEI {TENSOR_ID + "_WEI"};

    tensor_descs_.emplace(TCSRC, dt::f32, spec_.src_dims, spec_.raw_src_tag);
    tensor_descs_.emplace(TCWEI, dt::f32, spec_.wei_dims, spec_.raw_wei_tag);

    op typecast_src(ops_.size(), op::kind::TypeCast, {tensor_descs_[TCSRC]},
            {tensor_descs_[SRC]}, "typecast_src");
    ops_.emplace_back(typecast_src);
    op typecast_wei(ops_.size(), op::kind::TypeCast, {tensor_descs_[TCWEI]},
            {tensor_descs_[WEI]}, "typecast_wei");
    ops_.emplace_back(typecast_wei);

    return fill_status_t::DONE;
}
fill_status_t matmul_graph_prb_t::handle_low_precision_(
        const ::matmul::prb_t *prb_) {
    using op = dnnl::graph::op;

    const std::string IN_KEY
            = (with_typecast(get_dtypes())) ? "typecast" : "main";
    const std::string SRC = tensor_id[IN_KEY].back() + "_SRC";
    const std::string WEI = tensor_id[IN_KEY].back() + "_WEI";
    const std::string DST = curr_out_map_ids_.back() + "_DST";

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["dequant"].push_back(TENSOR_ID);
    const std::string QSRC {TENSOR_ID + "_SRC"};
    const std::string QWEI {TENSOR_ID + "_WEI"};
    const std::string QDST {TENSOR_ID + "_DST"};

    const std::string qsrc_type = spec_.src_dt == dt::u8 ? "uint8" : "int8";
    const std::string qwei_type = spec_.wei_dt == dt::u8 ? "uint8" : "int8";
    const std::string qdst_type = spec_.dst_dt == dt::u8 ? "uint8" : "int8";

    // `with_qdst == false` means that we are dealing
    // with Quantized lacking pattern, like x8s8f32 or x8s8bf16
    const bool with_qdst = dt::u8 == spec_.dst_dt || dt::s8 == spec_.dst_dt;

    tensor_descs_.emplace(
            QSRC, spec_.src_dt, spec_.src_dims, spec_.raw_src_tag);
    tensor_descs_.emplace(
            QWEI, spec_.wei_dt, spec_.wei_dims, spec_.raw_wei_tag);
    if (with_qdst)
        tensor_descs_.emplace(
                QDST, spec_.dst_dt, spec_.dst_dims, spec_.raw_dst_tag);

    const std::string wei_qtype = prb_->attr.oscale.policy == policy_t::COMMON
            ? "per_tensor"
            : "per_channel";
    const int64_t count
            = prb_->attr.oscale.policy == policy_t::COMMON ? 1 : prb_->n;

    oscales_.resize(count, 1.f);
    // if oscale is not default, copy values and pass it to oneDNN Graph
    if (!prb_->attr.oscale.is_def()) {
        for (int64_t c = 0; c < count; ++c)
            oscales_[c] = prb_->scales[c];
    }

    // currently, only policy_t::COMMON is supported for asymmetric quant
    // for src and dst, other policy is not suppoted by oneDNN Graph.
    // zps for src
    const int64_t common_zp_count = 1;
    const int64_t dflt_zp_val = 0;
    src_zero_points.resize(common_zp_count, dflt_zp_val);
    // if zp is not default, copy values and pass it to oneDNN Graph
    if (!prb_->attr.zero_points.is_def(DNNL_ARG_SRC)) {
        const auto &src_zp_e = prb_->attr.zero_points.get(DNNL_ARG_SRC);
        if (src_zp_e.policy != policy_t::COMMON)
            return fill_status::UNSUPPORTED_CONFIG;
        src_zero_points[0] = prb_->src_zp[0];
    }

    // zps for wei
    wei_zero_points.resize(common_zp_count, dflt_zp_val);
    // if zp is not default, copy values and pass it to oneDNN Graph
    if (!prb_->attr.zero_points.is_def(DNNL_ARG_WEIGHTS)) {
        const auto &wei_zp_e = prb_->attr.zero_points.get(DNNL_ARG_WEIGHTS);
        if (wei_zp_e.policy != policy_t::COMMON)
            return fill_status::UNSUPPORTED_CONFIG;
        wei_zero_points[0] = wei_zp_e.value;
    }

    // zps for dst
    dst_zero_points.resize(common_zp_count, dflt_zp_val);
    // if zp is not default, copy values and pass it to oneDNN Graph
    if (!prb_->attr.zero_points.is_def(DNNL_ARG_DST)) {
        const auto &dst_zp_e = prb_->attr.zero_points.get(DNNL_ARG_DST);
        if (dst_zp_e.policy != policy_t::COMMON)
            return fill_status::UNSUPPORTED_CONFIG;
        dst_zero_points[0] = prb_->dst_zp[0];
    }

    op dequant_src(ops_.size(), op::kind::Dequantize, {tensor_descs_[QSRC]},
            {tensor_descs_[SRC]}, "dequant_src");
    dequant_src.set_attr("scales", std::vector<float> {1.f})
            .set_attr("zps", src_zero_points)
            .set_attr("qtype", std::string("per_tensor"))
            .set_attr("in_type", qsrc_type)
            .set_attr("axis", static_cast<int64_t>(0));
    ops_.emplace_back(dequant_src);

    op dequant_wei(ops_.size(), op::kind::Dequantize, {tensor_descs_[QWEI]},
            {tensor_descs_[WEI]}, "dequant_wei");
    dequant_wei.set_attr("scales", oscales_)
            .set_attr("zps", wei_zero_points)
            .set_attr("qtype", wei_qtype)
            .set_attr("in_type", qwei_type)
            .set_attr("axis", static_cast<int64_t>(0));
    ops_.emplace_back(dequant_wei);

    if (with_qdst) {
        op quant_dst(ops_.size(), op::kind::Quantize, {tensor_descs_[DST]},
                {tensor_descs_[QDST]}, "quant");
        quant_dst.set_attr("scales", std::vector<float> {1.f})
                .set_attr("zps", dst_zero_points)
                .set_attr("qtype", std::string("per_tensor"))
                .set_attr("out_type", qdst_type)
                .set_attr("axis", static_cast<int64_t>(0));
        ops_.emplace_back(quant_dst);
    }

    if (has_post_sum()) {
        const low_precision_attr lp_attr(
                spec_.dst_dt, spec_.dst_dims, prb_->dtag);
        po_handler.conv.low_precision_handler.handle_low_precision_post_sum(
                *this, lp_attr, prb_->attr.post_ops.entry);
    }

    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status_t::DONE;
}

dims_t get_runtime_dims(
        const dims_t &dims, const ::matmul::dims_mask_t &mask) noexcept {
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
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

    matmul_graph_prb_t graph_prb(prb);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h = graph_prb.to_graph();

    // Filter partitions
    const auto partitions = graph_h.get_partitions();
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::matmul::init_pd, prb, res, par, ins, outs);

    const auto apply_bias = convert_dt(prb->bia_dt) != dt::undef;

    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto wei_fp = make_dnn_mem(ins[1], dt::f32, tag::abx);
    auto dst_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);
    dnn_mem_t bia_fp;
    if (apply_bias) bia_fp = make_dnn_mem(ins[2], dt::f32, tag::abx);

    auto src_dt = make_dnn_mem(ins[0], prb->stag);
    auto wei_dt = make_dnn_mem(ins[1], prb->wtag);
    auto dst_dt = make_dnn_mem(outs[0], prb->dtag);
    dnn_mem_t bia_dt;
    if (apply_bias) bia_dt = make_dnn_mem(ins[2], tag::abx);

    SAFE(fill_data(SRC, prb, src_dt, src_fp, res), WARN);
    SAFE(fill_data(WEI, prb, wei_dt, wei_fp, res), WARN);
    SAFE(fill_data(DST, prb, dst_dt, dst_fp, res), WARN);
    if (apply_bias) SAFE(fill_data(BIA, prb, bia_dt, bia_fp, res), WARN);

    // matmul operator supports only binary-add (single binary post-op)
    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    if (graph_prb.has_post_bin()) {
        binary_po_fp.emplace_back(make_dnn_mem(ins.back(), dt::f32, tag::abx));
        binary_po_dt.emplace_back(make_dnn_mem(ins.back(), prb->dtag));
        const int idx = 0;
        binary::fill_mem(DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx),
                binary_po_dt.back(), binary_po_fp.back());
    }

    dnnl::graph::engine &eng = get_test_engine();

    dnnl::graph::tensor src_tensor(ins[0], eng, static_cast<void *>(src_dt));
    dnnl::graph::tensor wei_tensor(ins[1], eng, static_cast<void *>(wei_dt));
    dnnl::graph::tensor dst_tensor(outs[0], eng, static_cast<void *>(dst_dt));
    dnnl::graph::tensor bia_tensor;
    dnnl::graph::tensor bin_tensor;
    dnnl::graph::tensor sum_src1_tensor;

    std::vector<dnnl::graph::tensor> tensors_in {src_tensor, wei_tensor};
    std::vector<dnnl::graph::tensor> tensors_out {dst_tensor};

    if (apply_bias) {
        bia_tensor
                = dnnl::graph::tensor(ins[2], eng, static_cast<void *>(bia_dt));
        tensors_in.emplace_back(bia_tensor);
    }
    // we can't have fuse with both sum and binary-add at the same time
    if (graph_prb.has_post_bin()) {
        bin_tensor = dnnl::graph::tensor(
                ins.back(), eng, static_cast<void *>(binary_po_dt.back()));
        tensors_in.emplace_back(bin_tensor);
    } else if (graph_prb.has_post_sum()) {
        sum_src1_tensor = dnnl::graph::tensor(
                ins.back(), eng, static_cast<void *>(dst_dt));
        tensors_in.emplace_back(sum_src1_tensor);
    }

    SAFE(execute_and_wait(cp, tensors_in, tensors_out), WARN);

    if (is_bench_mode(CORR)) {
        const auto &dnnl_test_engine = ::get_test_engine();

        if (apply_bias && is_low_precision(graph_prb.get_dtypes())) {
            dnn_mem_t bia_fp_scaled;
            bia_fp_scaled = make_dnn_mem(ins[2], dt::f32, tag::abx);
            scale_bia(bia_fp_scaled, bia_fp, graph_prb.get_oscales());
            ::matmul::compute_ref(dnnl_test_engine, prb, src_fp, wei_fp,
                    bia_fp_scaled, binary_po_fp, dst_fp);
        } else {
            ::matmul::compute_ref(dnnl_test_engine, prb, src_fp, wei_fp, bia_fp,
                    binary_po_fp, dst_fp);
        }

        compare::compare_t cmp;
        cmp.set_threshold(prb->cfg[DST].eps);
        cmp.set_data_kind(DST);
        cmp.set_zero_trust_percent(90.f); // TODO: why so bad filling?

        SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
    }

    SAFE(measure_perf(res->timer, cp, tensors_in, tensors_out), WARN);

    return OK;
}

} // namespace matmul
} // namespace benchdnnext
