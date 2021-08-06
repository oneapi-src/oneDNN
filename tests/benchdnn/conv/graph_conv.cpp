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

#include <string>
#include <vector>

namespace benchdnnext {
namespace conv {

namespace graph = dnnl::graph;

fill_status_t conv_graph_prb_t::handle_main_op_() {
    using kind = graph::op::kind;

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);

    const std::string SRC {TENSOR_ID + "_SRC"};
    const std::string WEI {TENSOR_ID + "_WEI"};
    const std::string DST {TENSOR_ID + "_DST"};

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

    tensor_descs_.emplace(SRC, src_dt, spec_.src_dims, spec_.raw_src_tag);
    tensor_descs_.emplace(WEI, wei_dt, wei_dims, spec_.raw_wei_tag);
    tensor_descs_.emplace(DST, dst_dt, spec_.dst_dims, spec_.raw_dst_tag);

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
        const ::conv::prb_t *prb) {
    const std::string SRC = tensor_id["main"].back() + "_SRC";
    const std::string WEI = tensor_id["main"].back() + "_WEI";
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

    tensor_descs_.emplace(QSRC, spec_.src_dt, spec_.src_dims, prb->stag);
    tensor_descs_.emplace(QWEI, spec_.wei_dt, wei_dims, prb->wtag);
    tensor_descs_.emplace(QDST, spec_.dst_dt, spec_.dst_dims, prb->dtag);

    graph::op dequant_src(ops_.size(), graph::op::kind::Dequantize,
            {tensor_descs_[QSRC]}, {tensor_descs_[SRC]}, "dequant_src");
    dequant_src.set_attr("scales", std::vector<float> {1.f})
            .set_attr("zps", std::vector<int64_t> {0})
            .set_attr<std::string>("qtype", "per_tensor")
            .set_attr("in_type", qsrc_type)
            .set_attr("axis", static_cast<int64_t>(0));
    ops_.emplace_back(dequant_src);

    graph::op dequant_wei(ops_.size(), graph::op::kind::Dequantize,
            {tensor_descs_[QWEI]}, {tensor_descs_[WEI]}, "dequant_wei");
    dequant_wei.set_attr("scales", oscales)
            .set_attr("zps", std::vector<int64_t>(count, 0L))
            .set_attr("qtype", qtype)
            .set_attr("in_type", qwei_type)
            .set_attr("axis", static_cast<int64_t>(0));
    ops_.emplace_back(dequant_wei);

    graph::op quant_dst(ops_.size(), graph::op::kind::Quantize,
            {tensor_descs_[DST]}, {tensor_descs_[QDST]}, "quant");
    quant_dst.set_attr("scales", std::vector<float> {1.f})
            .set_attr("zps", std::vector<int64_t> {0L})
            .set_attr<std::string>("qtype", "per_tensor")
            .set_attr("out_type", qdst_type)
            .set_attr("axis", static_cast<int64_t>(0));
    ops_.emplace_back(quant_dst);

    if (has_post_sum()) {
        const std::string QPSUM_SRC {TENSOR_ID + "_SUM_SRC1"};
        const std::string POST_SUM_SRC = tensor_id["sum"].back() + "_SRC";
        tensor_descs_.emplace(
                QPSUM_SRC, spec_.dst_dt, spec_.dst_dims, prb->stag);
        graph::op dequant_sum(ops_.size(), graph::op::kind::Dequantize,
                {tensor_descs_[QPSUM_SRC]}, {tensor_descs_[POST_SUM_SRC]},
                "dequant_sum");
        dequant_sum.set_attr("scales", std::vector<float> {1.f})
                .set_attr("zps", std::vector<int64_t> {0L});
        ops_.emplace_back(dequant_sum);
    }
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

fill_status_t conv_graph_prb_t::handle_bin_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.conv.bin_handler(*this, spec_.data_format, po_entry);
}

int doit(const ::conv::prb_t *prb, res_t *res) {
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    ::conv::check_known_skipped_case(prb, res);
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

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    auto init_pd = [&](dnnl_engine_t engine, const ::conv::prb_t *prb,
                           dnnl_primitive_desc_t &cpd, res_t *res, dir_t dir,
                           const_dnnl_primitive_desc_t hint) {
        SAFE(::conv::init_pd_custom(engine, prb, cpd, res), WARN);
        return OK;
    };
    auto cp = compile_partition(init_pd, prb, res, par, ins, outs);

    auto src_fp = make_dnn_mem(ins[0], spec.src_dims, dt::f32, tag::abx);
    auto wei_fp = make_dnn_mem(ins[1], spec.wei_dims, dt::f32, tag::abx);

    dnn_mem_t bia_fp;
    if (prb->dir == FWD_B) bia_fp = make_dnn_mem(ins[2], dt::f32, tag::x);
    auto dst_fp = make_dnn_mem(outs[0], spec.dst_dims, dt::f32, tag::abx);

    auto src_dt = make_dnn_mem(ins[0], spec.src_dims, spec.raw_src_tag);
    auto wei_dt = make_dnn_mem(ins[1], spec.wei_dims, spec.raw_wei_tag);
    dnn_mem_t bia_dt;
    if (prb->dir == FWD_B) bia_dt = make_dnn_mem(ins[2], tag::x);
    auto dst_dt = make_dnn_mem(outs[0], spec.dst_dims, spec.raw_dst_tag);

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
        const size_t idx = prb->dir == FWD_B ? 3 : 2;
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
    SAFE(measure_perf(res->timer, cp, tensors_in, tensors_out), WARN);

    return OK;
}

} // namespace conv
} // namespace benchdnnext
