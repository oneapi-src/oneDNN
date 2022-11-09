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
#include <tuple>
#include <utility>
#include <vector>

namespace benchdnnext {
namespace conv {

namespace graph = dnnl::graph;

static std::vector<dnnl::graph::logical_tensor::data_type> collect_data_types(
        const ::conv::prb_t *prb) {
    return {convert_dt(prb->cfg[SRC].dt), convert_dt(prb->cfg[WEI].dt),
            convert_dt(prb->cfg[DST].dt)};
}

static int check_known_skipped_case_graph(
        const ::conv::prb_t *prb, res_t *res) noexcept {

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, ::conv::init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const auto orig_dts = collect_data_types(prb);
    check_post_sum_for_bf16in_f32out(prb->attr, res, orig_dts);
    if (res->state == SKIPPED) return OK;

    check_graph_eltwise_post_ops(prb->attr, res);
    if (res->state == SKIPPED) return OK;

    check_graph_scales_and_zps_support(prb->attr, res);
    return OK;
}

static quant_data_t get_qdata_for(int arg, const ::conv::prb_t *prb) {
    const auto q_dt = convert_dt(prb->cfg[arg].dt);
    if (arg == SRC) {
        const int64_t zp_val = prb->attr.zero_points.is_def(DNNL_ARG_SRC)
                ? 0L
                : prb->src_zp[0];
        return quant_data_t(q_dt, {1.0f}, {zp_val}, prb->stag);
    } else if (arg == WEI) {
        const auto scales = get_scales(prb->attr.scales.get(DNNL_ARG_WEIGHTS),
                prb->wei_scales, prb->oc);
        const std::vector<int64_t> zps(scales.size(), 0L);
        const std::string q_type = prb->attr.scales.get(DNNL_ARG_WEIGHTS).policy
                        == policy_t::COMMON
                ? "per_tensor"
                : "per_channel";
        return quant_data_t(q_dt, scales, zps, q_type, 0, prb->wtag);
    } else if (arg == DST) {
        const int64_t zp_val = prb->attr.zero_points.is_def(DNNL_ARG_DST)
                ? 0L
                : prb->dst_zp[0];
        return quant_data_t(q_dt, {1.f}, {zp_val}, prb->dtag);
    }

    BENCHDNN_PRINT(
            0, "warning: returning default quant_data_t for arg: %d\n", arg);
    return quant_data_t();
}

static quant_data_t get_qdata_for(
        const attr_t::post_ops_t::entry_t &entry, const ::conv::prb_t *prb) {
    if (entry.is_binary_kind())
        return bin_po_entry2quant_data(
                entry, prb->dtag, convert_dt(prb->cfg[DST].dt));
    else if (entry.is_sum_kind())
        return sum_po_entry2quant_data(
                entry, prb->dtag, convert_dt(prb->cfg[DST].dt));

    printf("warning: returning default quant_data_t for unsupported post op\n");
    return quant_data_t();
}

static std::pair<fill_status_t, size_t> append_graph_with_depthwise(
        const ::conv::prb_t *prb) {
    std::unique_ptr<::conv::prb_t> dw_prb
            = ::conv_dw_fusion::get_fused_conv_prb(prb);
    if (!dw_prb) return std::make_pair(fill_status::UNSUPPORTED_OP, 0);

    graph_t &graph = graph_t::get();

    const auto op_id = graph.generate_id_for(entry_kind::CONV);
    const auto src_id = graph.generate_id_for(op_id, lt_kind::SRC, true);
    const auto wei_id = graph.generate_id_for(op_id, lt_kind::WEI);
    const auto dst_id = graph.generate_id_for(op_id, lt_kind::DST);

    dims_t wei_dims = dw_prb->wei_dims();
    // depthwise convolution must have groups
    if (!dw_prb->has_groups)
        return std::make_pair(fill_status::UNSUPPORTED_OP, 0);
    // group convolution convert
    dim_t groups = wei_dims[0];
    wei_dims.erase(wei_dims.begin());
    wei_dims[0] *= groups;

    dims_t dilations = dw_prb->dilations();
    // oneDNN graph dilation = 1 is equivalent of oneDNN
    // dilation = 0
    std::transform(dilations.begin(), dilations.end(), dilations.begin(),
            [](const dim_t d) { return d + 1; });

    const auto wei_dt = dequantize_dtype(convert_dt(dw_prb->cfg[WEI].dt));
    const auto dst_dt = dequantize_dtype(convert_dt(dw_prb->cfg[DST].dt));

    graph.create_lt(wei_id, wei_dt, wei_dims, dw_prb->wtag,
            dnnl::graph::logical_tensor::property_type::constant);
    graph.create_lt(dst_id, dst_dt, dw_prb->dst_dims(), dw_prb->dtag);

    dnnl::graph::op dw_op(op_id, dnnl::graph::op::kind::Convolution,
            graph.stringify_id(op_id));
    dw_op.set_attr("strides", dw_prb->strides())
            .set_attr("pads_begin", dw_prb->padding())
            .set_attr("pads_end", dw_prb->padding_r())
            .set_attr("dilations", dilations)
            .set_attr("auto_pad", std::string("None"))
            .set_attr("groups", dw_prb->g)
            .set_attr("data_format", std::string("NCX"))
            .set_attr("filter_format", std::string("OIX"));

    graph.append(op_id, dw_op, {src_id, wei_id}, {dst_id});

    return std::make_pair(fill_status::DONE, wei_id);
}

fill_status_t append_graph_with_block(const ::conv::prb_t *prb) {
    graph_t &graph = graph_t::get();

    const auto orig_dts = collect_data_types(prb);
    const auto with_dq = is_low_precision(orig_dts);
    const auto with_tc = with_typecast(orig_dts);
    const auto with_tc_after = with_typecast_after(orig_dts[0], orig_dts[2]);
    const auto connect_to_previous_block = !with_dq && graph.has_blocks();

    // handle main op
    const auto op_id = graph.generate_id_for(entry_kind::CONV);
    const auto src_lt_kind
            = prb->dir == BWD_D ? lt_kind::DIFF_SRC : lt_kind::SRC;
    const auto wei_lt_kind
            = prb->dir == BWD_W ? lt_kind::DIFF_WEI : lt_kind::WEI;
    const auto dst_lt_kind
            = prb->dir & FLAG_FWD ? lt_kind::DST : lt_kind::DIFF_DST;
    const auto src_id = connect_to_previous_block
            ? graph.get_last_block_out_id()
            : graph.generate_id_for(op_id, src_lt_kind);
    const auto wei_id = graph.generate_id_for(op_id, wei_lt_kind);
    const auto bia_id = graph.generate_id_for(op_id, lt_kind::BIA);
    const auto dst_id = graph.generate_id_for(op_id, dst_lt_kind);

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

    const auto change_dt = with_dq || with_tc_after || with_tc;
    const auto default_dt = (with_tc_after || with_tc) ? dt::bf16 : dt::f32;
    const auto src_dt = change_dt ? default_dt : orig_dts[0];
    const auto wei_dt = change_dt ? default_dt : orig_dts[1];
    const auto dst_dt = change_dt ? default_dt : orig_dts[2];
    const auto bia_dt = with_tc_after ? dt::bf16 : convert_dt(prb->cfg[BIA].dt);

    const auto wei_ptype = prb->dir == BWD_W
            ? dnnl::graph::logical_tensor::property_type::undef
            : dnnl::graph::logical_tensor::property_type::constant;

    graph.create_lt(src_id, src_dt, prb->src_dims(), prb->stag);
    graph.create_lt(wei_id, wei_dt, wei_dims, prb->wtag, wei_ptype);
    graph.create_lt(dst_id, dst_dt, prb->dst_dims(), prb->dtag);
    if (prb->dir & FLAG_BIA)
        graph.create_lt(bia_id, bia_dt, prb->bia_dims(), lt::strided,
                dnnl::graph::logical_tensor::property_type::constant);

    std::vector<size_t> src_ids {};
    std::vector<size_t> dst_ids {};
    dnnl::graph::op::kind conv_kind;
    if (prb->dir & FLAG_FWD) {
        src_ids = {src_id, wei_id};
        dst_ids = {dst_id};
        conv_kind = dnnl::graph::op::kind::Convolution;
    } else if (prb->dir == BWD_D) {
        src_ids = {dst_id, wei_id};
        dst_ids = {src_id};
        conv_kind = dnnl::graph::op::kind::ConvolutionBackpropData;
    } else if (prb->dir == BWD_W) {
        src_ids = {src_id, dst_id};
        dst_ids = {wei_id};
        conv_kind = dnnl::graph::op::kind::ConvolutionBackpropFilters;
    } else {
        return fill_status::UNSUPPORTED_CONFIG;
    }
    if (prb->dir & FLAG_BIA) src_ids.push_back(bia_id);

    dnnl::graph::op conv_op(op_id, conv_kind, graph.stringify_id(op_id));
    conv_op.set_attr("strides", prb->strides())
            .set_attr("pads_begin", prb->padding())
            .set_attr("pads_end", prb->padding_r())
            .set_attr("dilations", dilations)
            .set_attr("auto_pad", std::string("None"))
            .set_attr("groups", prb->g)
            .set_attr("data_format", std::string("NCX"))
            .set_attr("filter_format", std::string("OIX"));

    graph.append(op_id, conv_op, src_ids, dst_ids);

    fill_status_t status;
    // if required - apply dequantize to block inputs
    if (with_dq) {
        status = insert_dequant_before(src_id, get_qdata_for(SRC, prb));
        BENCHDNNEXT_VERIFY(status);
        status = insert_dequant_before(wei_id, get_qdata_for(WEI, prb), true);
        BENCHDNNEXT_VERIFY(status);
    }

    // handle post ops
    for (const auto &entry : prb->attr.post_ops.entry) {
        const auto with_src1_dq
                = entry.is_sum_kind() || is_dequantize_required_for(entry);
        size_t po_src1_id;
        if (entry.is_binary_kind()) {
            std::tie(status, po_src1_id) = append_graph_with_binary(entry);
            BENCHDNNEXT_VERIFY(status);
            if (with_src1_dq) {
                status = insert_dequant_before(
                        po_src1_id, get_qdata_for(entry, prb));
                BENCHDNNEXT_VERIFY(status);
            }
        } else if (entry.is_eltwise_kind()) {
            const bool is_swish
                    = entry.kind == attr_t::post_ops_t::kind_t::SWISH
                    && (prb->dir & FLAG_BIA);
            status = is_swish ? append_graph_with_swish(entry, dst_id)
                              : append_graph_with_eltwise(entry);
            BENCHDNNEXT_VERIFY(status);
        } else if (entry.is_sum_kind()) {
            std::tie(status, po_src1_id) = append_graph_with_sum(entry);
            BENCHDNNEXT_VERIFY(status);
            if (with_dq && with_src1_dq) {
                status = insert_dequant_before(
                        po_src1_id, get_qdata_for(entry, prb));
                BENCHDNNEXT_VERIFY(status);
            }
        } else if (entry.is_convolution_kind()) {
            // with support for int8 conv + dw pattern
            // we should insert dequantize before wei_id
            // (2nd returned value which is now ignored)
            std::tie(status, std::ignore) = append_graph_with_depthwise(prb);
            BENCHDNNEXT_VERIFY(status);
        }
    }

    // add typecast op after conv
    if (with_tc_after) {
        status = insert_typecast_after(
                graph.get_cur_block_out_id(), orig_dts[2]);
        BENCHDNNEXT_VERIFY(status);
    }

    // if required - add quantize op
    if (is_low_precision({orig_dts[2]})) {
        status = insert_quant_after(
                graph.get_cur_block_out_id(), get_qdata_for(DST, prb));
        BENCHDNNEXT_VERIFY(status);
    }

    graph.close_block();

    return fill_status::DONE;
}

int doit(const ::conv::prb_t *prb, res_t *res) {
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const auto status = append_graph_with_block(prb);
    if (status != fill_status::DONE
            && status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        cleanup();
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto &graph = graph_t::get();

    auto mode = convert_fpmath_mode(prb->attr.fpmath_mode);
    // Filter partitions
    const auto partitions = graph.get_partitions(mode);
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

    size_t bin_dt_idx = 0;
    for (size_t i = 0; i < po_entry.size(); i++) {
        if (po_entry[i].is_sum_kind()) { // Always use in-place operation.
            dnnl::graph::tensor sum_src1_tensor(
                    ins[++idx_ins], eng, static_cast<void *>(dst_dt));
            tensors_in.emplace_back(sum_src1_tensor);
        } else if (po_entry[i].is_binary_kind()) {
            dnnl::graph::tensor bin_src1_tensor(ins[++idx_ins], eng,
                    static_cast<void *>(binary_po_dt[bin_dt_idx]));
            tensors_in.emplace_back(bin_src1_tensor);
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
                std::vector<float> scales
                        = get_scales(prb->attr.scales.get(DNNL_ARG_WEIGHTS),
                                prb->wei_scales, prb->oc);
                int bia_mask = scales.size() == 1 ? 0 : 1;
                scale_bia(bia_fp_scaled, bia_fp, scales, bia_mask);
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
    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out,
                 ins, outs),
            WARN);

    cleanup();

    return OK;
}

} // namespace conv
} // namespace benchdnnext
