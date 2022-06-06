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
    ::conv::check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return;

    check_graph_eltwise_post_ops(prb->attr, res);
    if (res->state == SKIPPED) return;
}

fill_status_t conv_graph_prb_t::handle_main_op_() {
    using logical_tensor = dnnl::graph::logical_tensor;
    using kind = dnnl::graph::op::kind;

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);

    dims_t wei_dims = spec_.wei_dims;
    if (spec_.has_groups) {
        // group convolution convert
        dim_t groups = wei_dims[0];
        wei_dims.erase(wei_dims.begin());
        wei_dims[0] *= groups;
    }

    auto src_dt = benchdnnext::set_main_op_dtype(spec_.src_dt);
    auto wei_dt = benchdnnext::set_main_op_dtype(spec_.wei_dt);
    auto dst_dt = benchdnnext::set_main_op_dtype(spec_.dst_dt);

    std::string op_name = "";
    kind op_kind {kind::LastSymbol};
    std::vector<logical_tensor> inputs {};
    std::vector<logical_tensor> outputs {};

    if (spec_.dir & FLAG_FWD) {
        op_name = "Convolution";
        op_kind = kind::Convolution;

        const std::string SRC {TENSOR_ID + "_SRC"};
        const std::string WEI {TENSOR_ID + "_WEI"};
        const std::string DST {TENSOR_ID + "_DST"};

        tensor_descs_.emplace(SRC, src_dt, spec_.src_dims, spec_.raw_src_tag);
        tensor_descs_.emplace(WEI, wei_dt, wei_dims, spec_.raw_wei_tag,
                tensor_descs_t::property_type::constant);
        tensor_descs_.emplace(DST, dst_dt, spec_.dst_dims, spec_.raw_dst_tag);

        inputs = {tensor_descs_[SRC], tensor_descs_[WEI]};
        outputs = {tensor_descs_[DST]};
    } else if (spec_.dir & FLAG_BWD) {
        if (spec_.dir == BWD_D) {
            op_name = "ConvolutionBackpropData";
            op_kind = kind::ConvolutionBackpropData;

            const std::string DIFF_SRC {TENSOR_ID + "DIFF_SRC"};
            const std::string WEI {TENSOR_ID + "_WEI"};
            const std::string DIFF_DST {TENSOR_ID + "DIFF_DST"};

            tensor_descs_.emplace(
                    DIFF_SRC, src_dt, spec_.src_dims, spec_.raw_src_tag);
            tensor_descs_.emplace(WEI, wei_dt, wei_dims, spec_.raw_wei_tag,
                    tensor_descs_t::property_type::constant);
            tensor_descs_.emplace(
                    DIFF_DST, dst_dt, spec_.dst_dims, spec_.raw_dst_tag);

            inputs = {tensor_descs_[DIFF_DST], tensor_descs_[WEI]};
            outputs = {tensor_descs_[DIFF_SRC]};
        } else if (spec_.dir == BWD_W) {
            op_name = "ConvolutionBackpropFilter";
            op_kind = kind::ConvolutionBackpropFilters;

            const std::string SRC {TENSOR_ID + "_SRC"};
            const std::string DIFF_WEI {TENSOR_ID + "DIFF_WEI"};
            const std::string DIFF_DST {TENSOR_ID + "DIFF_DST"};

            tensor_descs_.emplace(
                    SRC, src_dt, spec_.src_dims, spec_.raw_src_tag);
            tensor_descs_.emplace(
                    DIFF_DST, dst_dt, spec_.dst_dims, spec_.raw_dst_tag);
            tensor_descs_.emplace(
                    DIFF_WEI, wei_dt, wei_dims, spec_.raw_wei_tag);

            inputs = {tensor_descs_[SRC], tensor_descs_[DIFF_DST]};
            outputs = {tensor_descs_[DIFF_WEI]};
        } else {
            return fill_status::UNSUPPORTED_CONFIG;
        }
    } else {
        return fill_status::UNSUPPORTED_CONFIG;
    }

    graph::op conv_op(new_op_id, op_kind, inputs, outputs, op_name);

    conv_op.set_attr("strides", spec_.strides)
            .set_attr("pads_begin", spec_.pads_begin)
            .set_attr("pads_end", spec_.pads_end)
            .set_attr("dilations", spec_.dilations)
            .set_attr("auto_pad", spec_.auto_pad)
            .set_attr("groups", spec_.groups)
            .set_attr("data_format", spec_.data_format)
            .set_attr("filter_format", spec_.filter_format);

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
    const auto dw_spec = dw_graph_prb.spec();

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["dw"].push_back(TENSOR_ID);
    const std::string DW_WEI {TENSOR_ID + "_WEI"};
    const std::string DW_DST {TENSOR_ID + "_DST"};

    dims_t wei_dims = dw_spec.wei_dims;
    // depthwise convolution must have groups
    if (!dw_spec.has_groups) return fill_status::UNSUPPORTED_OP;
    // group convolution convert
    dim_t groups = wei_dims[0];
    wei_dims.erase(wei_dims.begin());
    wei_dims[0] *= groups;

    tensor_descs_.emplace(DW_WEI, dw_spec.wei_dt, wei_dims, dw_spec.raw_wei_tag,
            tensor_descs_t::property_type::constant);
    tensor_descs_.emplace(
            DW_DST, dw_spec.dst_dt, dw_spec.dst_dims, dw_spec.raw_dst_tag);

    op dw(new_op_id, op::kind::Convolution,
            {tensor_descs_[curr_out_map_ids_.back() + "_DST"],
                    tensor_descs_[DW_WEI]},
            {tensor_descs_[DW_DST]}, "dw");

    dw.set_attr("strides", dw_spec.strides)
            .set_attr("pads_begin", dw_spec.pads_begin)
            .set_attr("pads_end", dw_spec.pads_end)
            .set_attr("dilations", dw_spec.dilations)
            .set_attr("auto_pad", dw_spec.auto_pad)
            .set_attr("groups", dw_spec.groups)
            .set_attr("data_format", dw_spec.data_format)
            .set_attr("filter_format", dw_spec.filter_format);

    ops_.emplace_back(dw);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

fill_status_t conv_graph_prb_t::handle_bia_() {
    return po_handler.conv.bias_handler(*this, spec_.data_format, spec_.bia_dt);
}

fill_status_t conv_graph_prb_t::handle_elt_(
        const attr_t::post_ops_t::entry_t &po) {
    return po_handler.conv.eltw_handler(*this, po);
}

fill_status_t conv_graph_prb_t::handle_sum_() {
    return po_handler.conv.sum_handler(*this);
}

fill_status_t conv_graph_prb_t::handle_low_precision_(
        const ::conv::prb_t *prb_) {

    const bool def_oscales = prb_->attr.oscale.is_def();

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

    const int64_t oscale_count
            = prb_->attr.oscale.policy == policy_t::COMMON ? 1 : prb_->oc;
    wei_zero_points = std::vector<int64_t>(oscale_count, 0L);

    // zps for dst
    dst_zero_points.resize(common_zp_count, dflt_zp_val);
    // if zp is not default, copy values and pass it to oneDNN Graph
    if (!prb_->attr.zero_points.is_def(DNNL_ARG_DST)) {
        const auto &dst_zp_e = prb_->attr.zero_points.get(DNNL_ARG_DST);
        if (dst_zp_e.policy != policy_t::COMMON)
            return fill_status::UNSUPPORTED_CONFIG;
        dst_zero_points[0] = prb_->dst_zp[0];
    }

    const float common_scale = [&prb_, this]() {
        if (has_post_eltwise()) {
            const float post_eltwise_scale
                    = get_post_eltwise_scale(prb_->attr.post_ops.entry);
            // benchdnn ext. need to convert post relu scale to quant scale to
            // get same result as benchdnn primitive did
            return 1.f * (1 / post_eltwise_scale);
        } else {
            return 1.f;
        }
    }();

    low_precision_attr lp_attr = low_precision_attr::lp_attr(spec_.src_dt,
            spec_.wei_dt, spec_.dst_dt, spec_.raw_src_tag, spec_.raw_wei_tag,
            spec_.raw_dst_tag, prb_->attr.oscale.policy, &oscales, common_scale,
            &src_zero_points, &wei_zero_points, &dst_zero_points, prb_->scales,
            prb_->oc, def_oscales);

    fill_status_t ctor_status
            = po_handler.conv.low_precision_handler.handle_low_precision_src(
                    *this, lp_attr);
    if (ctor_status != fill_status::DONE) return ctor_status;

    ctor_status
            = po_handler.conv.low_precision_handler.handle_low_precision_wei(
                    *this, lp_attr);
    if (ctor_status != fill_status::DONE) return ctor_status;

    // `with_qdst == false` means that we are dealing
    // with x8s8f32 pattern
    const bool with_qdst = dt::f32 != spec_.dst_dt;
    if (with_qdst) {
        ctor_status = po_handler.conv.low_precision_handler
                              .handle_low_precision_dst(*this, lp_attr);
    }
    if (ctor_status != fill_status::DONE) return ctor_status;

    if (has_post_sum()) {
        ctor_status = po_handler.conv.low_precision_handler
                              .handle_low_precision_post_sum(*this, lp_attr,
                                      prb_->attr.post_ops.entry);
    }

    return ctor_status;
}

fill_status_t conv_graph_prb_t::handle_bin_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.conv.bin_handler(*this, spec_.data_format, po_entry);
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
    const auto spec = graph_prb.spec();

    // Filer partitions
    const auto partitions
            = graph_h.get_partitions(dnnl::graph::partition::policy::fusion);
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto init_pd = [&](dnnl_engine_t engine, const ::conv::prb_t *prb,
                           dnnl_primitive_desc_t &cpd, res_t *res, dir_t dir,
                           const_dnnl_primitive_desc_t hint) {
        SAFE(::conv::init_pd(engine, prb, cpd, res, dir, hint), WARN);
        return OK;
    };
    auto cp = compile_partition(init_pd, prb, res, par, ins, outs);

    int idx_ins = 0;
    auto src_fp = make_dnn_mem(ins[idx_ins], spec.src_dims, dt::f32, tag::abx);
    auto src_dt = make_dnn_mem(ins[idx_ins], spec.src_dims, spec.raw_src_tag);

    dnn_mem_t wei_fp, wei_dt;
    if (prb->dir == BWD_W) {
        wei_fp = make_dnn_mem(outs[0], spec.wei_dims, dt::f32, tag::abx);
        wei_dt = make_dnn_mem(outs[0], spec.wei_dims, spec.raw_wei_tag);
    } else {
        wei_fp = make_dnn_mem(ins[++idx_ins], spec.wei_dims, dt::f32, tag::abx);
        wei_dt = make_dnn_mem(ins[idx_ins], spec.wei_dims, spec.raw_wei_tag);
    }

    dnn_mem_t bia_fp, bia_dt;
    if (prb->dir == FWD_B) {
        bia_fp = make_dnn_mem(ins[++idx_ins], dt::f32, tag::x);
        bia_dt = make_dnn_mem(ins[idx_ins], tag::x);
    }

    dnn_mem_t dst_fp, dst_dt;
    if (prb->dir == BWD_W) {
        dst_fp = make_dnn_mem(ins[1], spec.dst_dims, dt::f32, tag::abx);
        dst_dt = make_dnn_mem(ins[1], spec.dst_dims, spec.raw_dst_tag);
    } else {
        dst_fp = make_dnn_mem(outs[0], spec.dst_dims, dt::f32, tag::abx);
        dst_dt = make_dnn_mem(outs[0], spec.dst_dims, spec.raw_dst_tag);
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

    dnnl::graph::engine &eng = get_test_engine();

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

    SAFE(execute_and_wait(cp, tensors_in, tensors_out), WARN);

    args_t ref_args;

    if (is_bench_mode(CORR)) {
        if (prb->dir & FLAG_FWD) {
            dnnl_primitive_t c_ref = nullptr;
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
                scale_bia(bia_fp_scaled, bia_fp, graph_prb.get_oscales());
            }
            ref_args.set(DNNL_ARG_BIAS, bia_fp_scaled);

            ::conv::compute_ref_fwd(prb, c_ref, ref_args);
            SAFE(compare_data(prb, DST, dst_dt, dst_fp, res), WARN);
        } else if (prb->dir == BWD_D) {
            dnnl_primitive_t c_ref = nullptr;
            ref_args.set(DNNL_ARG_DIFF_SRC, src_fp);
            ref_args.set(DNNL_ARG_WEIGHTS, wei_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, dst_fp);
            ::conv::compute_ref_bwd_d(prb, c_ref, ref_args);
            SAFE(compare_data(prb, SRC, src_dt, src_fp, res), WARN);
        } else if (prb->dir == BWD_W) {
            dnnl_primitive_t c_ref = nullptr;
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, dst_fp);
            ref_args.set(DNNL_ARG_DIFF_WEIGHTS, wei_fp);
            ::conv::compute_ref_bwd_w(prb, c_ref, ref_args);
            SAFE(compare_data(prb, WEI, wei_dt, wei_fp, res), WARN);
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
