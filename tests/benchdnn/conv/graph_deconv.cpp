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
#include "conv/graph_deconv.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace benchdnnext {
namespace deconv {

void check_known_skipped_case_graph(
        const ::conv::prb_t *prb, res_t *res) noexcept {
    ::deconv::check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return;

    const bool with_groups = prb->g > 1;
    if (with_groups) {
        const std::string wei_dnnl_fmt_tag_str
                = get_ou_format(normalize_tag(prb->wtag, prb->ndims));
        const dnnl_format_tag_t wei_dnnl_fmt_tag
                = dnnl_fmt_str2tag(wei_dnnl_fmt_tag_str);
        const std::vector<dnnl_format_tag_t> acceptable_wei_fmt_tags {
                dnnl_acbd, dnnl_acbde, dnnl_acbdef};
        const bool valid_wei_fmt_tag = std::any_of(
                acceptable_wei_fmt_tags.begin(), acceptable_wei_fmt_tags.end(),
                [wei_dnnl_fmt_tag](const dnnl_format_tag_t tag) {
                    return tag == wei_dnnl_fmt_tag;
                });
        // required in order to make strides aligned with oneDNN graph expectations
        if (!valid_wei_fmt_tag)
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
    }
}

dims_t get_acbdx_strides(const dims_t &wei_dims) {
    // permute dims OIX => IOX
    const dims_t wei_dims_permuted = [&wei_dims]() {
        auto d = wei_dims;
        std::swap(d[0], d[1]);
        return d;
    }();
    // calculate the original strides
    dims_t strides(wei_dims_permuted.size());
    strides[strides.size() - 1] = 1;
    for (int i = static_cast<int>(strides.size()) - 2; i >= 0; --i) {
        strides[i] = wei_dims_permuted[i + 1] * strides[i + 1];
    }
    // permute strides IOX => OIX
    const dims_t strides_permuted = [&strides]() {
        auto s = strides;
        std::swap(s[0], s[1]);
        return s;
    }();

    return strides_permuted;
}

fill_status_t deconv_graph_prb_t::handle_main_op_() {
    using op = dnnl::graph::op;

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
        wei_dims[1] *= groups;
    }

    auto src_dt = benchdnnext::set_main_op_dtype(spec_.src_dt);
    auto wei_dt = benchdnnext::set_main_op_dtype(spec_.wei_dt);
    auto dst_dt = benchdnnext::set_main_op_dtype(spec_.dst_dt);

    for (auto actual_dt : {src_dt, wei_dt, dst_dt}) {
        if (std::find(dt_constraints.begin(), dt_constraints.end(), actual_dt)
                == dt_constraints.end()) {
            return fill_status::UNSUPPORTED_CONFIG;
        }
    }

    if (spec_.has_groups && spec_.groups > 1) {
        const auto strides_permuted = get_acbdx_strides(wei_dims);
        tensor_descs_.emplace(WEI, wei_dt, wei_dims, strides_permuted);
    } else {
        tensor_descs_.emplace(WEI, wei_dt, wei_dims, spec_.raw_wei_tag);
    }

    tensor_descs_.emplace(SRC, src_dt, spec_.src_dims, spec_.raw_src_tag);
    tensor_descs_.emplace(DST, dst_dt, spec_.dst_dims, spec_.raw_dst_tag);

    op deconv_op(new_op_id, op::kind::ConvTranspose,
            {tensor_descs_[SRC], tensor_descs_[WEI]}, {tensor_descs_[DST]},
            "deconv");

    deconv_op.set_attr("strides", spec_.strides)
            .set_attr("pads_begin", spec_.pads_begin)
            .set_attr("pads_end", spec_.pads_end)
            .set_attr("dilations", spec_.dilations)
            .set_attr("auto_pad", spec_.auto_pad)
            .set_attr("groups", spec_.groups)
            .set_attr("data_format", spec_.data_format)
            .set_attr("filter_format", spec_.filter_format);

    ops_.emplace_back(deconv_op);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

fill_status_t deconv_graph_prb_t::handle_bia_() {
    return po_handler.deconv.bias_handler(
            *this, spec_.data_format, spec_.bia_dt);
}

fill_status_t deconv_graph_prb_t::handle_sum_() {
    return po_handler.deconv.sum_handler(*this);
}

fill_status_t deconv_graph_prb_t::handle_bin_(
        const attr_t::post_ops_t::entry_t &po) {
    return po_handler.deconv.bin_handler(*this, spec_.data_format, po);
}

fill_status_t deconv_graph_prb_t::handle_elt_(
        const attr_t::post_ops_t::entry_t &po) {
    return po_handler.deconv.eltw_handler(*this, po);
}

fill_status_t deconv_graph_prb_t::handle_low_precision_(
        const ::conv::prb_t *prb) {
    const bool def_oscales = prb->attr.oscale.is_def();

    // currently, only policy_t::COMMON is supported for asymmetric quant
    // for src and dst, other policies are not suppoted by oneDNN Graph.
    const int64_t common_zp_count = 1;
    const int64_t dflt_zp_val = 0;
    src_zero_points.resize(common_zp_count, dflt_zp_val);
    // if zp is not default, copy values and pass it to oneDNN Graph
    if (!prb->attr.zero_points.is_def(DNNL_ARG_SRC)) {
        const auto &src_zp_e = prb->attr.zero_points.get(DNNL_ARG_SRC);
        if (src_zp_e.policy != policy_t::COMMON)
            return fill_status::UNSUPPORTED_CONFIG;
        src_zero_points[0] = prb->src_zp[0];
    }

    const int64_t oscale_count
            = prb->attr.oscale.policy == policy_t::COMMON ? 1 : prb->oc;
    wei_zero_points = std::vector<int64_t>(oscale_count, 0L);

    dst_zero_points.resize(common_zp_count, dflt_zp_val);
    // if zp is not default, copy values and pass it to oneDNN Graph
    if (!prb->attr.zero_points.is_def(DNNL_ARG_DST)) {
        const auto &dst_zp_e = prb->attr.zero_points.get(DNNL_ARG_DST);
        if (dst_zp_e.policy != policy_t::COMMON)
            return fill_status::UNSUPPORTED_CONFIG;
        dst_zero_points[0] = prb->dst_zp[0];
    }

    const float common_scale = [&prb, this]() {
        if (has_post_eltwise()) {
            const float post_eltwise_scale
                    = get_post_eltwise_scale(prb->attr.post_ops.entry);
            // benchdnn ext. need to convert post relu scale to quant scale to
            // get same result as benchdnn primitive did
            return 1.f * (1 / post_eltwise_scale);
        } else {
            return 1.f;
        }
    }();

    low_precision_attr lp_attr = low_precision_attr::lp_attr(spec_.src_dt,
            spec_.wei_dt, spec_.dst_dt, spec_.raw_src_tag, spec_.raw_wei_tag,
            spec_.raw_dst_tag, prb->attr.oscale.policy, &oscales, common_scale,
            &src_zero_points, &wei_zero_points, &dst_zero_points, prb->scales,
            prb->oc, def_oscales);

    dims_t wei_dims = spec_.wei_dims;
    if (spec_.has_groups) {
        // group convolution convert
        dim_t groups = wei_dims[0];
        wei_dims.erase(wei_dims.begin());
        wei_dims[1] *= groups;
    }

    if (spec_.has_groups && spec_.groups > 1) {
        const auto strides_permuted = get_acbdx_strides(wei_dims);
        lp_attr.set_wei_strides(strides_permuted);
    }

    fill_status_t ctor_status;
    ctor_status
            = po_handler.deconv.low_precision_handler.handle_low_precision_src(
                    *this, lp_attr);
    if (ctor_status != fill_status::DONE) return ctor_status;

    ctor_status
            = po_handler.deconv.low_precision_handler.handle_low_precision_wei(
                    *this, lp_attr);
    if (ctor_status != fill_status::DONE) return ctor_status;

    ctor_status
            = po_handler.deconv.low_precision_handler.handle_low_precision_dst(
                    *this, lp_attr);
    if (ctor_status != fill_status::DONE) return ctor_status;

    return ctor_status;
}

int doit(const ::conv::prb_t *prb, res_t *res) {
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

    deconv_graph_prb_t graph_prb(prb);
    if (!check_graph_creation_status(&graph_prb, res)) { return OK; }

    auto graph_h = graph_prb.to_graph();
    const auto spec = graph_prb.spec();

    const auto partitions
            = graph_h.get_partitions(dnnl::graph::partition::policy::fusion);
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, OK;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    const auto &e = benchdnnext::get_test_engine();
    auto cp = par.compile(ins, outs, e);

    auto src_fp = make_dnn_mem(ins[0], spec.src_dims, dt::f32, tag::abx);
    auto wei_fp = make_dnn_mem(ins[1], spec.wei_dims, dt::f32, tag::abx);
    auto dst_fp = make_dnn_mem(outs[0], spec.dst_dims, dt::f32, tag::abx);
    dnn_mem_t bia_fp;
    if (graph_prb.has_post_bia())
        bia_fp = make_dnn_mem(ins[2], dt::f32, tag::x);

    auto wei_tr_dims = spec.wei_dims;
    std::swap(wei_tr_dims[1], wei_tr_dims[2]);
    auto wei_tr_fp = make_dnn_mem(ins[1], wei_tr_dims, dt::f32, tag::abx);

    auto src_dt = make_dnn_mem(ins[0], spec.src_dims, spec.raw_src_tag);
    auto wei_dt = make_dnn_mem(ins[1], spec.wei_dims, spec.raw_wei_tag);
    auto dst_dt = make_dnn_mem(outs[0], spec.dst_dims, spec.raw_dst_tag);
    dnn_mem_t bia_dt;
    if (graph_prb.has_post_bia()) bia_dt = make_dnn_mem(ins[2], tag::x);

    SAFE(fill_src(prb, src_dt, src_fp, res), WARN);
    SAFE(fill_wei(prb, wei_dt, wei_fp, res), WARN);
    if (graph_prb.has_post_bia())
        SAFE(fill_bia(prb, bia_dt, bia_fp, res), WARN);
    // NOTE: currently we support only forward pass.
    // In this case, there is no need to fill dst.

    SAFE(::deconv::transpose_data_wei(prb, wei_fp, wei_tr_fp), WARN);

    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    if (graph_prb.has_post_bin()) {
        binary_po_fp.emplace_back(make_dnn_mem(ins.back(), dt::f32, tag::abx));
        binary_po_dt.emplace_back(make_dnn_mem(ins.back(), tag::abx));
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

    if (graph_prb.has_post_bia()) {
        bia_tensor
                = dnnl::graph::tensor(ins[2], eng, static_cast<void *>(bia_dt));
        tensors_in.emplace_back(bia_tensor);
    }
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
    args_t ref_args;

    if (is_bench_mode(CORR)) {
        dnnl_primitive_t c_ref = nullptr;
        {
            ::conv::prb_t prb_tr((::conv::desc_t)*prb, prb->dir, prb->cfg,
                    prb->stag, prb->wtag, prb->dtag, prb->alg, prb->attr,
                    prb->mb, true);
            std::swap(prb_tr.ic, prb_tr.oc);
            std::swap(prb_tr.ih, prb_tr.oh);
            std::swap(prb_tr.id, prb_tr.od);
            std::swap(prb_tr.iw, prb_tr.ow);

            std::vector<int> binary_po_args;
            for (int idx = 0; idx < binary_po_fp.size(); idx++) {
                binary_po_args.emplace_back(
                        (DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1));
            }
            args_t ref_args;
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_WEIGHTS, wei_fp);
            ref_args.set(DNNL_ARG_BIAS, bia_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(DNNL_ARG_DIFF_WEIGHTS, wei_tr_fp); // Hack. See ref.
            ref_args.set(binary_po_args, binary_po_fp);
            ::deconv::compute_ref_fwd(&prb_tr, c_ref, ref_args);
        }

        SAFE(compare_data(prb, DST, dst_dt, dst_fp, res), WARN);
    }

    measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out);

    return OK;
}

} // namespace deconv
} // namespace benchdnnext
