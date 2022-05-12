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

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_types.h"

#include "dnnl_graph_common.hpp"

#include "binary/binary.hpp"
#include "conv/conv_dw_fusion.hpp"
#include "conv/graph_conv.hpp"
#include "prelu/prelu.hpp"

#include <string>
#include <vector>

namespace benchdnnext {
namespace conv {

namespace graph = dnnl::graph;

void check_known_skipped_case_graph(
        const ::conv::prb_t *prb, res_t *res) noexcept {
    // TODO: to align with original benchdnn, we should consider moving
    // skip_unimplemented_prb call after compilation step
    skip_invalid_and_unimplemented_prb(prb, res);
    if (res->state == SKIPPED) return;

    check_graph_eltwise_post_ops(prb->attr, res);
    if (res->state == SKIPPED) return;

    check_graph_zps_support(prb->attr.zero_points, res);
}

static quant_data_t get_qdata_for(int arg, const ::conv::prb_t *prb) {
    const auto q_dt = convert_dt(prb->cfg[arg].dt);
    if (arg == SRC) {
        const int64_t zp_val = prb->attr.zero_points.is_def(DNNL_ARG_SRC)
                ? 0L
                : prb->src_zp[0];
        return quant_data_t(q_dt, {1.0f}, {zp_val}, prb->stag);
    } else if (arg == WEI) {
        const auto scales = get_scales(prb->attr.oscale, prb->scales, prb->oc);
        const std::vector<int64_t> zps(scales.size(), 0L);
        const std::string q_type = prb->attr.oscale.policy == policy_t::COMMON
                ? "per_tensor"
                : "per_channel";
        return quant_data_t(q_dt, scales, zps, q_type, 0, prb->wtag);
    } else if (arg == DST) {
        const float scale_val = 1.f
                * (1.f / get_post_eltwise_scale(prb->attr.post_ops.entry));
        const int64_t zp_val = prb->attr.zero_points.is_def(DNNL_ARG_DST)
                ? 0L
                : prb->dst_zp[0];
        return quant_data_t(q_dt, {scale_val}, {zp_val}, prb->dtag);
    }

    BENCHDNN_PRINT(
            0, "warning: returning default quant_data_t for arg: %d\n", arg);
    return quant_data_t();
}

fill_status_t conv_graph_prb_t::handle_main_op_(const ::conv::prb_t *prb) {
    using logical_tensor = dnnl::graph::logical_tensor;
    using kind = dnnl::graph::op::kind;

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);

    // this is needed to align with po_handlers convention
    // some patterns like `conv + bias + swish` may want to
    // reuse bias output via `tensor_id["bias"].back() + "_DST"`
    if (has_post_bia_) tensor_id["bias"].push_back(TENSOR_ID);

    dims_t wei_dims = prb->wei_dims();
    if (prb->has_groups) {
        // group convolution convert
        dim_t groups = wei_dims[0];
        wei_dims.erase(wei_dims.begin());
        wei_dims[0] *= groups;
    }

    dims_t dilations = prb->dilations();
    // oneDNN graph dilation = 1 is equivalent of oneDNN
    // dilation = 0
    std::transform(dilations.begin(), dilations.end(), dilations.begin(),
            [](const dim_t d) { return d + 1; });

    auto src_dt = benchdnnext::set_main_op_dtype(convert_dt(prb->cfg[SRC].dt));
    auto wei_dt = benchdnnext::set_main_op_dtype(convert_dt(prb->cfg[WEI].dt));
    auto dst_dt = benchdnnext::set_main_op_dtype(convert_dt(prb->cfg[DST].dt));
    auto bia_dt = convert_dt(prb->cfg[BIA].dt);

    std::string op_name {};
    kind op_kind {kind::LastSymbol};
    std::vector<logical_tensor> inputs {};
    std::vector<logical_tensor> outputs {};

    if (prb->dir & FLAG_FWD) {
        op_name = "Convolution";
        op_kind = kind::Convolution;

        const std::string SRC {TENSOR_ID + "_SRC"};
        const std::string WEI {TENSOR_ID + "_WEI"};
        const std::string BIA {TENSOR_ID + "_BIA"};
        const std::string DST {TENSOR_ID + "_DST"};

        tensor_descs_.emplace(SRC, src_dt, prb->src_dims(), prb->stag);
        tensor_descs_.emplace(WEI, wei_dt, wei_dims, prb->wtag,
                tensor_descs_t::property_type::constant);
        tensor_descs_.emplace(DST, dst_dt, prb->dst_dims(), prb->dtag);
        if (has_post_bia_)
            tensor_descs_.emplace(BIA, bia_dt, prb->bia_dims(), lt::strided,
                    tensor_descs_t::property_type::constant);

        inputs = {tensor_descs_[SRC], tensor_descs_[WEI]};
        outputs = {tensor_descs_[DST]};
        if (has_post_bia_) inputs.push_back(tensor_descs_[BIA]);
    } else if (prb->dir & FLAG_BWD) {
        if (prb->dir == BWD_D) {
            op_name = "ConvolutionBackpropData";
            op_kind = kind::ConvolutionBackpropData;

            const std::string DIFF_SRC {TENSOR_ID + "DIFF_SRC"};
            const std::string WEI {TENSOR_ID + "_WEI"};
            const std::string DIFF_DST {TENSOR_ID + "DIFF_DST"};

            tensor_descs_.emplace(DIFF_SRC, src_dt, prb->src_dims(), prb->stag);
            tensor_descs_.emplace(WEI, wei_dt, wei_dims, prb->wtag,
                    tensor_descs_t::property_type::constant);
            tensor_descs_.emplace(DIFF_DST, dst_dt, prb->dst_dims(), prb->dtag);

            inputs = {tensor_descs_[DIFF_DST], tensor_descs_[WEI]};
            outputs = {tensor_descs_[DIFF_SRC]};
        } else if (prb->dir == BWD_W) {
            op_name = "ConvolutionBackpropFilter";
            op_kind = kind::ConvolutionBackpropFilters;

            const std::string SRC {TENSOR_ID + "_SRC"};
            const std::string DIFF_WEI {TENSOR_ID + "DIFF_WEI"};
            const std::string DIFF_DST {TENSOR_ID + "DIFF_DST"};

            tensor_descs_.emplace(SRC, src_dt, prb->src_dims(), prb->stag);
            tensor_descs_.emplace(DIFF_DST, dst_dt, prb->dst_dims(), prb->dtag);
            tensor_descs_.emplace(DIFF_WEI, wei_dt, wei_dims, prb->wtag);

            inputs = {tensor_descs_[SRC], tensor_descs_[DIFF_DST]};
            outputs = {tensor_descs_[DIFF_WEI]};
        } else {
            return fill_status::UNSUPPORTED_CONFIG;
        }
    } else {
        return fill_status::UNSUPPORTED_CONFIG;
    }

    graph::op conv_op(new_op_id, op_kind, inputs, outputs, op_name);

    conv_op.set_attr("strides", prb->strides())
            .set_attr("pads_begin", prb->padding())
            .set_attr("pads_end", prb->padding_r())
            .set_attr("dilations", dilations)
            .set_attr("auto_pad", std::string("None"))
            .set_attr("groups", prb->g)
            .set_attr("data_format", std::string("NCX"))
            .set_attr("filter_format", std::string("OIX"));

    ops_.emplace_back(conv_op);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

fill_status_t conv_graph_prb_t::handle_dw_(const ::conv::prb_t *prb_) {
    using op = dnnl::graph::op;

    std::unique_ptr<::conv::prb_t> dw_prb
            = ::conv_dw_fusion::get_fused_conv_prb(prb_);
    if (!dw_prb) return fill_status::UNSUPPORTED_OP;

    conv::conv_graph_prb_t dw_graph_prb(dw_prb.get());
    if (dw_graph_prb.ctor_status != fill_status::DONE
            && dw_graph_prb.ctor_status
                    != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return dw_graph_prb.ctor_status;
    }

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["dw"].push_back(TENSOR_ID);
    const std::string DW_WEI {TENSOR_ID + "_WEI"};
    const std::string DW_DST {TENSOR_ID + "_DST"};

    dims_t wei_dims = dw_prb->wei_dims();
    // depthwise convolution must have groups
    if (!dw_prb->has_groups) return fill_status::UNSUPPORTED_OP;
    // group convolution convert
    dim_t groups = wei_dims[0];
    wei_dims.erase(wei_dims.begin());
    wei_dims[0] *= groups;

    auto dw_wei_dt
            = benchdnnext::set_main_op_dtype(convert_dt(dw_prb->cfg[WEI].dt));
    auto dw_dst_dt
            = benchdnnext::set_main_op_dtype(convert_dt(dw_prb->cfg[DST].dt));

    tensor_descs_.emplace(DW_WEI, dw_wei_dt, wei_dims, dw_prb->wtag,
            tensor_descs_t::property_type::constant);
    tensor_descs_.emplace(DW_DST, dw_dst_dt, dw_prb->dst_dims(), dw_prb->dtag);

    op dw(new_op_id, op::kind::Convolution,
            {tensor_descs_[curr_out_map_ids_.back() + "_DST"],
                    tensor_descs_[DW_WEI]},
            {tensor_descs_[DW_DST]}, "dw");

    const std::string auto_pad {"None"};
    const std::string data_format {"NCX"};
    const std::string filter_format {"OIX"};

    dims_t dilations = dw_prb->dilations();
    // oneDNN graph dilation = 1 is equivalent of oneDNN
    // dilation = 0
    std::transform(dilations.begin(), dilations.end(), dilations.begin(),
            [](const dim_t d) { return d + 1; });

    dw.set_attr("strides", dw_prb->strides())
            .set_attr("pads_begin", dw_prb->padding())
            .set_attr("pads_end", dw_prb->padding_r())
            .set_attr("dilations", dilations)
            .set_attr("auto_pad", auto_pad)
            .set_attr("groups", dw_prb->g)
            .set_attr("data_format", data_format)
            .set_attr("filter_format", filter_format);

    ops_.emplace_back(dw);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

fill_status_t conv_graph_prb_t::handle_elt_(
        const attr_t::post_ops_t::entry_t &po) {
    return po_handler.conv.eltw_handler(*this, po, has_post_bia_);
}

fill_status_t conv_graph_prb_t::handle_sum_() {
    return po_handler.conv.sum_handler(*this);
}

fill_status_t conv_graph_prb_t::handle_low_precision_(
        const ::conv::prb_t *prb) {
    // if there will be support for x8x8bf16 case, conditionally change
    // OP_REPR to "typecast"
    const std::string OP_REPR = "main";
    const auto src_lt_id = tensor_id[OP_REPR].back() + "_SRC";
    const auto wei_lt_id = tensor_id[OP_REPR].back() + "_WEI";
    const auto dst_lt_id = curr_out_map_ids_.back() + "_DST";

    fill_status_t status
            = po_handler.conv.low_precision_handler.insert_dequant_before(
                    src_lt_id, get_qdata_for(SRC, prb), *this);
    BENCHDNNEXT_VERIFY(status);

    status = po_handler.conv.low_precision_handler.insert_dequant_before(
            wei_lt_id, get_qdata_for(WEI, prb), *this, true);
    BENCHDNNEXT_VERIFY(status);

    // `with_qdst == false` means that we are dealing
    // with x8s8f32 pattern
    const bool with_qdst = convert_dt(prb->cfg[DST].dt)
            != dnnl::graph::logical_tensor::data_type::f32;
    if (with_qdst) {
        status = po_handler.conv.low_precision_handler.insert_quant_after(
                dst_lt_id, get_qdata_for(DST, prb), *this);
        BENCHDNNEXT_VERIFY(status);
    }

    for (const auto &entry : prb->attr.post_ops.entry) {
        if (entry.is_sum_kind()) {
            const auto sum_src1_lt_id = tensor_id["sum"].back() + "_SRC";
            status = po_handler.conv.low_precision_handler
                             .insert_dequant_before(sum_src1_lt_id,
                                     sum_po_entry2quant_data(entry, prb->dtag,
                                             convert_dt(prb->cfg[DST].dt)),
                                     *this);
            BENCHDNNEXT_VERIFY(status);
            break;
        }
    }

    size_t bin_id {0};
    for (const auto &entry : prb->attr.post_ops.entry) {
        if (entry.is_binary_kind()) {
            const auto bin_src1_lt_id = tensor_id["binary"][bin_id++] + "_SRC";
            status = po_handler.conv.low_precision_handler
                             .insert_dequant_before(bin_src1_lt_id,
                                     bin_po_entry2quant_data(entry, prb->dtag,
                                             convert_dt(prb->cfg[DST].dt)),
                                     *this);
            BENCHDNNEXT_VERIFY(status);
        }
    }

    return status;
}

fill_status_t conv_graph_prb_t::handle_bin_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.conv.bin_handler(*this, po_entry);
}

int doit(const ::conv::prb_t *prb, res_t *res) {
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

    conv_graph_prb_t graph_prb(prb);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h = graph_prb.to_graph();

    // Filer partitions
    const auto partitions
            = graph_h.get_partitions(dnnl::graph::partition::policy::fusion);
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::conv::init_pd, prb, res, par, ins, outs);

    int idx_ins = 0;
    auto src_fp
            = make_dnn_mem(ins[idx_ins], prb->src_dims(), dt::f32, tag::abx);
    auto src_dt = make_dnn_mem(ins[idx_ins], prb->src_dims(), prb->stag);

    dnn_mem_t wei_fp, wei_dt;
    if (prb->dir == BWD_W) {
        wei_fp = make_dnn_mem(outs[0], prb->wei_dims(), dt::f32, tag::abx);
        wei_dt = make_dnn_mem(outs[0], prb->wei_dims(), prb->wtag);
    } else {
        wei_fp = make_dnn_mem(
                ins[++idx_ins], prb->wei_dims(), dt::f32, tag::abx);
        wei_dt = make_dnn_mem(ins[idx_ins], prb->wei_dims(), prb->wtag);
    }

    dnn_mem_t bia_fp, bia_dt;
    if (prb->dir == FWD_B) {
        bia_fp = make_dnn_mem(ins[++idx_ins], dt::f32, tag::x);
        bia_dt = make_dnn_mem(ins[idx_ins], tag::x);
    }

    dnn_mem_t dst_fp, dst_dt;
    if (prb->dir == BWD_W) {
        dst_fp = make_dnn_mem(ins[1], prb->dst_dims(), dt::f32, tag::abx);
        dst_dt = make_dnn_mem(ins[1], prb->dst_dims(), prb->dtag);
    } else {
        dst_fp = make_dnn_mem(outs[0], prb->dst_dims(), dt::f32, tag::abx);
        dst_dt = make_dnn_mem(outs[0], prb->dst_dims(), prb->dtag);
    }

    SAFE(fill_src(prb, src_dt, src_fp, res), WARN);
    SAFE(fill_wei(prb, wei_dt, wei_fp, res), WARN);
    SAFE(fill_bia(prb, bia_dt, bia_fp, res), WARN);
    SAFE(fill_dst(prb, dst_dt, dst_fp, res), WARN);

    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<dnn_mem_t> prelu_po_fp, prelu_po_dt;
    std::vector<int> prelu_po_args;
    //TODO - Please add support for prelu in dnnl-graph
    //SAFE(prelu::setup_prelu_po(
    //             const_pd, dst_md, prelu_po_args, prelu_po_fp, prelu_po_dt),
    //        WARN);

    const std::vector<attr_t::post_ops_t::entry_t> &po_entry
            = prb->attr.post_ops.entry;
    const std::vector<size_t> post_bin_indices = get_post_bin_indices(po_entry);

    for (size_t i = 0; i < post_bin_indices.size(); i++) {
        binary_po_fp.emplace_back(
                make_dnn_mem(ins[++idx_ins], dt::f32, tag::abx));
        binary_po_dt.emplace_back(make_dnn_mem(ins[idx_ins], tag::abx));
        binary::fill_mem(DNNL_ARG_ATTR_MULTIPLE_POST_OP(
                                 static_cast<int>(post_bin_indices[i])),
                binary_po_dt[i], binary_po_fp[i]);
    }

    const dnnl::graph::engine &eng = get_test_engine();

    idx_ins = 0;
    graph::tensor src_tensor(ins[idx_ins], eng,
            static_cast<void *>(prb->dir == BWD_D ? dst_dt : src_dt));
    graph::tensor wei_tensor;
    if (prb->dir == BWD_W)
        wei_tensor = graph::tensor(outs[0], eng, static_cast<void *>(wei_dt));
    else
        wei_tensor = graph::tensor(
                ins[++idx_ins], eng, static_cast<void *>(wei_dt));
    graph::tensor bia_tensor;
    if (prb->dir == FWD_B)
        bia_tensor = graph::tensor(
                ins[++idx_ins], eng, static_cast<void *>(bia_dt));

    graph::tensor dst_tensor;
    if (prb->dir == BWD_W)
        dst_tensor = graph::tensor(ins[1], eng, static_cast<void *>(dst_dt));
    else
        dst_tensor = graph::tensor(outs[0], eng,
                static_cast<void *>(prb->dir == BWD_D ? src_dt : dst_dt));

    std::vector<graph::tensor> tensors_in {src_tensor};
    if (prb->dir == BWD_W)
        tensors_in.emplace_back(dst_tensor);
    else
        tensors_in.emplace_back(wei_tensor);
    if (prb->dir == FWD_B) tensors_in.emplace_back(bia_tensor);

    graph::tensor sum_src1_tensor;
    graph::tensor bin_tensor;

    size_t bin_dt_idx = 0;
    for (size_t i = 0; i < po_entry.size(); i++) {
        if (po_entry[i].is_sum_kind()) { // Always use in-place operation.
            sum_src1_tensor = graph::tensor(
                    ins[++idx_ins], eng, static_cast<void *>(dst_dt));
            tensors_in.emplace_back(sum_src1_tensor);
        } else if (po_entry[i].is_binary_kind()) {
            bin_tensor = graph::tensor(ins[++idx_ins], eng,
                    static_cast<void *>(binary_po_dt[bin_dt_idx]));
            tensors_in.emplace_back(bin_tensor);
            ++bin_dt_idx;
        }
    }

    std::vector<graph::tensor> tensors_out {};
    if (prb->dir == BWD_W)
        tensors_out.emplace_back(wei_tensor);
    else
        tensors_out.emplace_back(dst_tensor);

    SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

    args_t args, ref_args;

    if (is_bench_mode(CORR)) {
        if (prb->dir & FLAG_FWD) {
            args.set(DNNL_ARG_DST, dst_dt);
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_WEIGHTS, wei_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);

            std::vector<int> binary_po_args;
            for (size_t idx_bin : post_bin_indices) {
                binary_po_args.emplace_back((DNNL_ARG_ATTR_MULTIPLE_POST_OP(
                                                     static_cast<int>(idx_bin))
                        | DNNL_ARG_SRC_1));
            }
            ref_args.set(binary_po_args, binary_po_fp);
            ref_args.set(prelu_po_args, prelu_po_fp);

            // re-scale bias
            dnn_mem_t bia_fp_scaled;
            if (prb->dir == FWD_B) {
                bia_fp_scaled = make_dnn_mem(ins[2], dt::f32, tag::x);
                scale_bia(bia_fp_scaled, bia_fp,
                        get_scales(prb->attr.oscale, prb->scales, prb->oc));
            }
            ref_args.set(DNNL_ARG_BIAS, bia_fp_scaled);

            check_correctness(
                    prb, {DST}, args, ref_args, ::conv::setup_cmp, res);
        } else if (prb->dir == BWD_D) {
            args.set(DNNL_ARG_DIFF_SRC, src_dt);
            ref_args.set(DNNL_ARG_DIFF_SRC, src_fp);
            ref_args.set(DNNL_ARG_WEIGHTS, wei_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, dst_fp);

            check_correctness(
                    prb, {SRC}, args, ref_args, ::conv::setup_cmp, res);
        } else if (prb->dir == BWD_W) {
            args.set(DNNL_ARG_DIFF_WEIGHTS, wei_dt);
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, dst_fp);
            ref_args.set(DNNL_ARG_DIFF_WEIGHTS, wei_fp);

            check_correctness(
                    prb, {WEI}, args, ref_args, ::conv::setup_cmp, res);
        } else {
            SAFE(FAIL, CRIT);
        }
    }
    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);

    return OK;
}

} // namespace conv
} // namespace benchdnnext
