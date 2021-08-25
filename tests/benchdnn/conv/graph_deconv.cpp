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

#include "conv/graph_deconv.hpp"

#include <string>
#include <utility>
#include <vector>

namespace benchdnnext {
namespace deconv {

void check_known_skipped_case_graph(
        const ::conv::prb_t *prb, res_t *res) noexcept {
    ::deconv::check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return;

    const auto with_groups = prb->g > 1;
    const auto filter_format = tag2filter_format(prb->wtag, with_groups);
    // required in order to make strides aligned with oneDNN graph expectations
    if (with_groups && filter_format != "IOX") {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
    }
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
    if (spec_.has_groups && spec_.groups > 1) {
        // permute dims OIX => IOX
        dims_t wei_dims_permuted = [&wei_dims]() {
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
        dims_t strides_permuted = [&strides]() {
            auto s = strides;
            std::swap(s[0], s[1]);
            return s;
        }();

        tensor_descs_.emplace(WEI, dt::f32, wei_dims, strides_permuted);
    } else {
        tensor_descs_.emplace(WEI, dt::f32, wei_dims, spec_.raw_wei_tag);
    }

    tensor_descs_.emplace(SRC, dt::f32, spec_.src_dims, spec_.raw_src_tag);
    tensor_descs_.emplace(DST, dt::f32, spec_.dst_dims, spec_.raw_dst_tag);

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

int doit(const ::conv::prb_t *prb, res_t *res) {
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

    deconv_graph_prb_t graph_prb(prb);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h = graph_prb.to_graph();
    const auto spec = graph_prb.spec();

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
    auto dst_fp = make_dnn_mem(outs[0], spec.dst_dims, dt::f32, tag::abx);

    auto wei_tr_dims = spec.wei_dims;
    std::swap(wei_tr_dims[1], wei_tr_dims[2]);
    auto wei_tr_fp = make_dnn_mem(ins[1], wei_tr_dims, dt::f32, tag::abx);
    std::vector<dnn_mem_t> binary_po_fp;
    dnn_mem_t bia_fp;

    auto src_dt = make_dnn_mem(ins[0], spec.src_dims, spec.raw_src_tag);
    auto wei_dt = make_dnn_mem(ins[1], spec.wei_dims, spec.raw_wei_tag);
    auto dst_dt = make_dnn_mem(outs[0], spec.dst_dims, spec.raw_dst_tag);

    SAFE(fill_src(prb, src_dt, src_fp, res), WARN);
    SAFE(fill_wei(prb, wei_dt, wei_fp, res), WARN);
    // NOTE: currently we support only forward pass.
    // In this case, there is no need to fill dst.

    SAFE(::deconv::transpose_data_wei(prb, wei_fp, wei_tr_fp), WARN);

    dnnl::graph::tensor src_tensor(ins[0], static_cast<void *>(src_dt));
    dnnl::graph::tensor wei_tensor(ins[1], static_cast<void *>(wei_dt));
    dnnl::graph::tensor dst_tensor(outs[0], static_cast<void *>(dst_dt));

    std::vector<dnnl::graph::tensor> tensors_in {src_tensor, wei_tensor};
    std::vector<dnnl::graph::tensor> tensors_out {dst_tensor};

    SAFE(execute_and_wait(cp, tensors_in, tensors_out), WARN);

    if (is_bench_mode(CORR)) {
        const auto fp = dnnl_f32;
        const auto src_tag = tag::abx;
        dnnl_primitive_t c_ref = nullptr;

        const auto &dnnl_test_engine = ::get_test_engine();
        {
            ::conv::prb_t prb_tr((::conv::desc_t)*prb, prb->dir, prb->cfg,
                    prb->stag, prb->wtag, prb->dtag, prb->alg, prb->attr,
                    prb->mb, true);
            std::swap(prb_tr.ic, prb_tr.oc);
            std::swap(prb_tr.ih, prb_tr.oh);
            std::swap(prb_tr.id, prb_tr.od);
            std::swap(prb_tr.iw, prb_tr.ow);
            ::conv::compute_ref_bwd_d(&prb_tr, c_ref, dst_fp, wei_tr_fp, bia_fp,
                    binary_po_fp, src_fp);
        }
        dnn_mem_t dst(dst_dt, fp, src_tag, dnnl_test_engine);
        SAFE(compare_dst(prb, dst, dst_fp, res, true), WARN);
    }

    measure_perf(res->timer, cp, tensors_in, tensors_out);

    return OK;
}

} // namespace deconv
} // namespace benchdnnext
