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

#include <algorithm>
#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_types.h"

#include "dnnl_graph_common.hpp"

#include "binary/binary.hpp"

#include "graph_conv.hpp"

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

    bia_dim.assign({prb->oc});

    dim_t dst_1d_dims[] = {prb->mb, prb->oc, prb->ow};
    dim_t dst_2d_dims[] = {prb->mb, prb->oc, prb->oh, prb->ow};
    dim_t dst_3d_dims[] = {prb->mb, prb->oc, prb->od, prb->oh, prb->ow};

    switch (prb->ndims) {
        case CONV_3D_NDIMS: {
            src_dim.assign(src_3d_dims, end(src_3d_dims));
            dst_dim.assign(dst_3d_dims, end(dst_3d_dims));

            wei_dim.assign(
                    wei_3d_dims + (prb->has_groups ? 0 : 1), end(wei_3d_dims));
        } break;

        case CONV_2D_NDIMS: {
            src_dim.assign(src_2d_dims, end(src_2d_dims));
            dst_dim.assign(dst_2d_dims, end(dst_2d_dims));

            wei_dim.assign(
                    wei_2d_dims + (prb->has_groups ? 0 : 1), end(wei_2d_dims));
        } break;

        case CONV_1D_NDIMS: {
            src_dim.assign(src_1d_dims, end(src_1d_dims));
            dst_dim.assign(dst_1d_dims, end(dst_1d_dims));

            wei_dim.assign(
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

    src_dtype = convert_dt(prb->cfg[SRC].dt);
    wei_dtype = convert_dt(prb->cfg[WEI].dt);
    bia_dtype = convert_dt(prb->cfg[BIA].dt);
    dst_dtype = convert_dt(prb->cfg[DST].dt);

    data_format = tag2data_format(prb->stag);
    filter_format = tag2filter_format(prb->wtag);
}

fill_status_t conv_graph_prb_t::handle_main_op_() {
    using kind = graph::op::kind;

    const std::string SRC {"conv_src"};
    const std::string WEI {"conv_wei"};
    const std::string DST {"conv_dst"};

    dims_t wei_dim = spec_.wei_dim;
    if (prb->has_groups) {
        // group convolution convert
        dim_t groups = wei_dim[0];
        wei_dim.erase(wei_dim.begin());
        wei_dim[0] *= groups;
    }

    tensor_descs_.emplace(SRC, dt::f32, spec_.src_dim, prb->stag);
    tensor_descs_.emplace(WEI, dt::f32, wei_dim, prb->wtag);
    tensor_descs_.emplace(DST, dt::f32, spec_.dst_dim, prb->dtag);

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
    return po_handler.conv.bias_handler(
            *this, spec_.data_format, spec_.bia_dtype);
}

fill_status_t conv_graph_prb_t::handle_bin_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.conv.bin_handler(*this, spec_.data_format, po_entry);
}

fill_status_t conv_graph_prb_t::handle_elt_(
        const attr_t::post_ops_t::entry_t &po) {
    return po_handler.conv.eltw_handler(*this, po);
}

fill_status_t conv_graph_prb_t::handle_sum_() {
    return po_handler.conv.sum_handler(*this);
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

    graph::graph graph = graph_prb.to_graph();

    // Filer partitions
    std::vector<dnnl::graph::partition> partitions
            = graph.get_partitions(dnnl::graph::partition::policy::fusion);
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;
    graph::partition par = partitions[0];

    const auto inputs = par.get_in_ports();
    const auto outputs = par.get_out_ports();

    auto cp = compile_partition(res->create_timer, par, inputs, outputs);

    auto spec = graph_prb.spec();
    dnn_mem_t src_fp = make_dnn_mem(inputs[0], spec.src_dim, dt::f32, tag::abx);
    dnn_mem_t wei_fp = make_dnn_mem(inputs[1], spec.wei_dim, dt::f32, tag::abx);
    dnn_mem_t bia_fp;
    if (prb->dir == FWD_B) bia_fp = make_dnn_mem(inputs[2], dt::f32, tag::x);
    dnn_mem_t dst_fp
            = make_dnn_mem(outputs[0], spec.dst_dim, dt::f32, tag::abx);

    dnn_mem_t src_dt = make_dnn_mem(inputs[0], spec.src_dim, prb->stag);
    dnn_mem_t wei_dt = make_dnn_mem(inputs[1], spec.wei_dim, prb->wtag);
    dnn_mem_t bia_dt;
    if (prb->dir == FWD_B) bia_dt = make_dnn_mem(inputs[2], tag::x);
    dnn_mem_t dst_dt = make_dnn_mem(outputs[0], spec.dst_dim, prb->dtag);

    SAFE(fill_src(prb, src_dt, src_fp, res), WARN);
    SAFE(fill_wei(prb, wei_dt, wei_fp, res), WARN);
    SAFE(fill_bia(prb, bia_dt, bia_fp, res), WARN);
    SAFE(fill_dst(prb, dst_dt, dst_fp, res), WARN);

    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    if (graph_prb.has_post_bin()) {
        binary_po_fp.emplace_back(
                make_dnn_mem(inputs.back(), dt::f32, tag::abx));
        binary_po_dt.emplace_back(make_dnn_mem(inputs.back(), tag::abx));
        const int idx = 0;
        binary::fill_mem(DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx),
                binary_po_dt.back(), binary_po_fp.back());
    }

    graph::tensor src_tensor(inputs[0], static_cast<float *>(src_dt));
    graph::tensor wei_tensor(inputs[1], static_cast<float *>(wei_dt));
    graph::tensor bia_tensor;
    if (prb->dir == FWD_B)
        bia_tensor = graph::tensor(inputs[2], static_cast<float *>(bia_dt));
    graph::tensor dst_tensor(outputs[0], static_cast<float *>(dst_dt));

    std::vector<graph::tensor> input_ts {src_tensor, wei_tensor};
    if (prb->dir == FWD_B) input_ts.push_back(bia_tensor);

    graph::tensor sum_src1_tensor;
    graph::tensor bin_tensor;
    if (graph_prb.has_post_sum()) { // Always use in-place operation.
        size_t idx = prb->dir == FWD_B ? 3 : 2;
        sum_src1_tensor
                = graph::tensor(inputs[idx], static_cast<float *>(dst_dt));
        input_ts.push_back(sum_src1_tensor);
    } else if (graph_prb.has_post_bin()) {
        bin_tensor = graph::tensor(
                inputs.back(), static_cast<void *>(binary_po_dt.back()));
        input_ts.push_back(bin_tensor);
    }
    std::vector<graph::tensor> output_ts {dst_tensor};

    SAFE(execute_and_wait(cp, input_ts, output_ts), WARN);

    if (is_bench_mode(CORR)) {
        const auto fp = dnnl_f32;
        const auto src_tag = tag::abx;
        dnnl_primitive_t c_ref = nullptr;

        // re-scale bias
        dnn_mem_t bia_fp_scaled;
        if (prb->dir == FWD_B) {
            bia_fp_scaled = make_dnn_mem(inputs[2], dt::f32, tag::x);
            scale_bia(bia_fp_scaled, bia_fp, graph_prb.oscales);
        }

        const auto &dnnl_test_engine = ::get_test_engine();
        ::conv::compute_ref_fwd(prb, c_ref, src_fp, wei_fp, bia_fp_scaled,
                binary_po_fp, dst_fp);
        dnn_mem_t dst(dst_dt, fp, src_tag, dnnl_test_engine);
        SAFE(compare_dst(prb, dst, dst_fp, res, true), WARN);
    }
    SAFE(measure_perf(res->timer, cp, input_ts, output_ts), WARN);

    return OK;
}

} // namespace conv
} // namespace benchdnnext
