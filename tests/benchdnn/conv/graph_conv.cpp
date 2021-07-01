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

#include "binary/binary.hpp"
#include "conv/graph_conv.hpp"

#include <algorithm>
#include <string>
#include <vector>

#define CONV_3D_NDIMS 5
#define CONV_2D_NDIMS 4
#define CONV_1D_NDIMS 3
#define CONV_MAX_NDIMS CONV_3D_NDIMS

namespace benchdnnext {
namespace conv {

namespace graph = dnnl::graph;

conv_graph_prb_t::spec_t::spec_t(const ::conv::prb_t *prb) {
    groups = prb->has_groups ? (int64_t)prb->g : 1;

    dim_t src_1d_dims[] = {prb->mb, prb->ic, prb->iw};
    dim_t src_2d_dims[] = {prb->mb, prb->ic, prb->ih, prb->iw};
    dim_t src_3d_dims[] = {prb->mb, prb->ic, prb->id, prb->ih, prb->iw};

    dim_t wei_1d_dims[] = {prb->g, prb->oc / prb->g, prb->ic / prb->g, prb->kw};
    dim_t wei_2d_dims[]
            = {prb->g, prb->oc / prb->g, prb->ic / prb->g, prb->kh, prb->kw};
    dim_t wei_3d_dims[] = {prb->g, prb->oc / prb->g, prb->ic / prb->g, prb->kd,
            prb->kh, prb->kw};

    bia_dims.assign({prb->oc});

    dim_t dst_1d_dims[] = {prb->mb, prb->oc, prb->ow};
    dim_t dst_2d_dims[] = {prb->mb, prb->oc, prb->oh, prb->ow};
    dim_t dst_3d_dims[] = {prb->mb, prb->oc, prb->od, prb->oh, prb->ow};

    switch (prb->ndims) {
        case CONV_3D_NDIMS: {
            src_dims.assign(src_3d_dims, end(src_3d_dims));
            dst_dims.assign(dst_3d_dims, end(dst_3d_dims));

            wei_dims.assign(
                    wei_3d_dims + (prb->has_groups ? 0 : 1), end(wei_3d_dims));
        } break;

        case CONV_2D_NDIMS: {
            src_dims.assign(src_2d_dims, end(src_2d_dims));
            dst_dims.assign(dst_2d_dims, end(dst_2d_dims));

            wei_dims.assign(
                    wei_2d_dims + (prb->has_groups ? 0 : 1), end(wei_2d_dims));
        } break;

        case CONV_1D_NDIMS: {
            src_dims.assign(src_1d_dims, end(src_1d_dims));
            dst_dims.assign(dst_1d_dims, end(dst_1d_dims));

            wei_dims.assign(
                    wei_1d_dims + (prb->has_groups ? 0 : 1), end(wei_1d_dims));
        } break;

        default: break;
    }

    dim_t strides_nd[] = {prb->sd, prb->sh, prb->sw};
    dim_t dilates_nd[] = {prb->dd, prb->dh, prb->dw};
    dim_t padding_nd[] = {prb->pd, prb->ph, prb->pw};
    dim_t padding_r_nd[] = {prb->pd_r, prb->ph_r, prb->pw_r};

    const size_t spatial_offset = CONV_MAX_NDIMS - prb->ndims;
    strides.assign(strides_nd + spatial_offset, end(strides_nd));
    pads_begin.assign(padding_nd + spatial_offset, end(padding_nd));
    pads_end.assign(padding_r_nd + spatial_offset, end(padding_r_nd));
    dilations.assign(dilates_nd + spatial_offset, end(dilates_nd));
    std::transform(dilations.begin(), dilations.end(), dilations.begin(),
            [](const dim_t d) { return d + 1; });

    src_dt = convert_dt(prb->cfg[SRC].dt);
    wei_dt = convert_dt(prb->cfg[WEI].dt);
    bia_dt = convert_dt(prb->cfg[BIA].dt);
    dst_dt = convert_dt(prb->cfg[DST].dt);

    data_format = tag2data_format(prb->stag);
    filter_format = tag2filter_format(prb->wtag);
}

fill_status_t conv_graph_prb_t::handle_main_op_() {
    using kind = graph::op::kind;

    const std::string SRC {"conv_src"};
    const std::string WEI {"conv_wei"};
    const std::string DST {"conv_dst"};

    dims_t wei_dims = spec_.wei_dims;
    if (prb->has_groups) {
        // group convolution convert
        dim_t groups = wei_dims[0];
        wei_dims.erase(wei_dims.begin());
        wei_dims[0] *= groups;
    }

    tensor_descs_.emplace(SRC, dt::f32, spec_.src_dims, prb->stag);
    tensor_descs_.emplace(WEI, dt::f32, wei_dims, prb->wtag);
    tensor_descs_.emplace(DST, dt::f32, spec_.dst_dims, prb->dtag);

    const size_t new_op_id = ops_.size();
    graph::op conv_op(new_op_id, kind::Convolution,
            {tensor_descs_[SRC], tensor_descs_[WEI]}, {tensor_descs_[DST]},
            "conv");

    conv_op.set_attr("strides", spec_.strides)
            .set_attr("pads_begin", spec_.pads_begin)
            .set_attr("pads_end", spec_.pads_end)
            .set_attr("dilations", spec_.dilations)
            .set_attr("auto_pad", spec_.auto_pad)
            .set_attr("groups", spec_.groups)
            .set_attr("data_format", spec_.data_format)
            .set_attr("filter_format", spec_.filter_format);

    ops_.emplace_back(conv_op);
    curr_out_map_ids_.assign({DST});

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

fill_status_t conv_graph_prb_t::handle_low_precision_() {
    if (spec_.src_dt != dt::f32 || spec_.wei_dt != dt::f32
            || spec_.dst_dt != dt::f32) {
        std::string output_name = curr_out_map_ids_.front();
        std::string qoutput_name = "q" + output_name;

        const std::string qsrc_type = spec_.src_dt == dt::u8 ? "uint8" : "int8";
        const std::string qwei_type = spec_.wei_dt == dt::u8 ? "uint8" : "int8";
        const std::string qdst_type = spec_.dst_dt == dt::u8 ? "uint8" : "int8";

        const std::string qtype = prb->attr.oscale.policy == policy_t::COMMON
                ? "per_tensor"
                : "per_channel";
        const int64_t count
                = prb->attr.oscale.policy == policy_t::COMMON ? 1 : prb->oc;
        oscales.resize(count, 1.f);
        for (int64_t c = 0; c < count; ++c)
            oscales[c] = prb->scales[c];

        dims_t wei_dims = spec_.wei_dims;
        if (prb->has_groups) {
            // group convolution convert
            dim_t groups = wei_dims[0];
            wei_dims.erase(wei_dims.begin());
            wei_dims[0] *= groups;
        }

        tensor_descs_.emplace(
                "qconv_src", spec_.src_dt, spec_.src_dims, prb->stag);
        tensor_descs_.emplace("qconv_wei", spec_.wei_dt, wei_dims, prb->wtag);
        tensor_descs_.emplace(
                qoutput_name, spec_.dst_dt, spec_.dst_dims, prb->dtag);

        graph::op dequant_src(ops_.size(), graph::op::kind::Dequantize,
                {tensor_descs_["qconv_src"]}, {tensor_descs_["conv_src"]},
                "dequant_src");
        dequant_src.set_attr("scales", std::vector<float> {1.f})
                .set_attr("zps", std::vector<int64_t> {0})
                .set_attr<std::string>("qtype", "per_tensor")
                .set_attr("in_type", qsrc_type)
                .set_attr("axis", static_cast<int64_t>(0));
        ops_.emplace_back(dequant_src);

        graph::op dequant_wei(ops_.size(), graph::op::kind::Dequantize,
                {tensor_descs_["qconv_wei"]}, {tensor_descs_["conv_wei"]},
                "dequant_wei");
        dequant_wei.set_attr("scales", oscales)
                .set_attr("zps", std::vector<int64_t>(count, 0L))
                .set_attr("qtype", qtype)
                .set_attr("in_type", qwei_type)
                .set_attr("axis", static_cast<int64_t>(0));
        ops_.emplace_back(dequant_wei);

        graph::op quant_dst(ops_.size(), graph::op::kind::Quantize,
                {tensor_descs_[output_name]}, {tensor_descs_[qoutput_name]},
                "quant");
        quant_dst.set_attr("scales", std::vector<float> {1.f})
                .set_attr("zps", std::vector<int64_t> {0L})
                .set_attr<std::string>("qtype", "per_tensor")
                .set_attr("out_type", qdst_type)
                .set_attr("axis", static_cast<int64_t>(0));
        ops_.emplace_back(quant_dst);

        if (has_post_sum_) {
            tensor_descs_.emplace(
                    "qsum_src1", spec_.dst_dt, spec_.dst_dims, prb->stag);
            graph::op dequant_sum(ops_.size(), graph::op::kind::Dequantize,
                    {tensor_descs_["qsum_src1"]}, {tensor_descs_["sum_src1"]},
                    "dequant_sum");
            dequant_sum.set_attr("scales", std::vector<float> {1.f})
                    .set_attr("zps", std::vector<int64_t> {0L});
            ops_.emplace_back(dequant_sum);
        }
        curr_out_map_ids_.assign({qoutput_name});
    }

    return fill_status::DONE;
}

fill_status_t conv_graph_prb_t::handle_bin_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.conv.bin_handler(*this, spec_.data_format, po_entry);
}

int doit(const ::conv::prb_t *prb, res_t *res) {
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    conv_graph_prb_t graph_prb(prb);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h = graph_prb.to_graph();
    auto spec = graph_prb.spec();

    // Filer partitions
    const auto partitions
            = graph_h.get_partitions(dnnl::graph::partition::policy::fusion);
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    const auto &e = benchdnnext::get_test_engine();
    auto cp = par.compile(ins, outs, e);

    auto src_fp = make_dnn_mem(ins[0], spec.src_dims, dt::f32, tag::abx);
    auto wei_fp = make_dnn_mem(ins[1], spec.wei_dims, dt::f32, tag::abx);
    dnn_mem_t bia_fp;
    if (prb->dir == FWD_B) bia_fp = make_dnn_mem(ins[2], dt::f32, tag::x);
    auto dst_fp = make_dnn_mem(outs[0], spec.dst_dims, dt::f32, tag::abx);

    auto src_dt = make_dnn_mem(ins[0], spec.src_dims, prb->stag);
    auto wei_dt = make_dnn_mem(ins[1], spec.wei_dims, prb->wtag);
    dnn_mem_t bia_dt;
    if (prb->dir == FWD_B) bia_dt = make_dnn_mem(ins[2], tag::x);
    auto dst_dt = make_dnn_mem(outs[0], spec.dst_dims, prb->dtag);

    SAFE(fill_src(prb, src_dt, src_fp, res), WARN);
    SAFE(fill_wei(prb, wei_dt, wei_fp, res), WARN);
    SAFE(fill_bia(prb, bia_dt, bia_fp, res), WARN);
    SAFE(fill_dst(prb, dst_dt, dst_fp, res), WARN);

    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    if (graph_prb.has_post_bin()) {
        binary_po_fp.emplace_back(make_dnn_mem(ins.back(), dt::f32, tag::abx));
        binary_po_dt.emplace_back(make_dnn_mem(ins.back(), tag::abx));
        const int idx = 0;
        binary::fill_mem(DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx),
                binary_po_dt.back(), binary_po_fp.back());
    }

    graph::tensor src_tensor(ins[0], static_cast<float *>(src_dt));
    graph::tensor wei_tensor(ins[1], static_cast<float *>(wei_dt));
    graph::tensor bia_tensor;
    if (prb->dir == FWD_B)
        bia_tensor = graph::tensor(ins[2], static_cast<float *>(bia_dt));
    graph::tensor dst_tensor(outs[0], static_cast<float *>(dst_dt));

    std::vector<graph::tensor> tensors_in {src_tensor, wei_tensor};
    if (prb->dir == FWD_B) tensors_in.emplace_back(bia_tensor);

    graph::tensor sum_src1_tensor;
    graph::tensor bin_tensor;
    if (graph_prb.has_post_sum()) { // Always use in-place operation.
        size_t idx = prb->dir == FWD_B ? 3 : 2;
        sum_src1_tensor = graph::tensor(ins[idx], static_cast<float *>(dst_dt));
        tensors_in.emplace_back(sum_src1_tensor);
    } else if (graph_prb.has_post_bin()) {
        bin_tensor = graph::tensor(
                ins.back(), static_cast<void *>(binary_po_dt.back()));
        tensors_in.emplace_back(bin_tensor);
    }
    std::vector<graph::tensor> tensors_out {dst_tensor};

    SAFE(execute_and_wait(cp, tensors_in, tensors_out), WARN);

    if (is_bench_mode(CORR)) {
        const auto fp = dnnl_f32;
        const auto src_tag = tag::abx;
        dnnl_primitive_t c_ref = nullptr;

        // re-scale bias
        dnn_mem_t bia_fp_scaled;
        if (prb->dir == FWD_B) {
            bia_fp_scaled = make_dnn_mem(ins[2], dt::f32, tag::x);
            scale_bia(bia_fp_scaled, bia_fp, graph_prb.get_oscales());
        }

        const auto &dnnl_test_engine = ::get_test_engine();
        ::conv::compute_ref_fwd(prb, c_ref, src_fp, wei_fp, bia_fp_scaled,
                binary_po_fp, dst_fp);
        dnn_mem_t dst(dst_dt, fp, src_tag, dnnl_test_engine);
        SAFE(compare_dst(prb, dst, dst_fp, res, true), WARN);
    }

    measure_perf(res->timer, cp, tensors_in, tensors_out);

    return OK;
}

} // namespace conv
} // namespace benchdnnext
