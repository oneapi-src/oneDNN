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

#include "compare.hpp"
#include "dnnl_graph_common.hpp"

#include "pool/graph_pool.hpp"

#include <algorithm>

namespace benchdnnext {
namespace pool {

pool_graph_prb_t::spec_t::spec_t(const ::pool::prb_t *prb) {
    using graph_op = dnnl::graph::op::kind;

    dim_t src_1d_dims[] = {prb->mb, prb->ic, prb->iw};
    dim_t src_2d_dims[] = {prb->mb, prb->ic, prb->ih, prb->iw};
    dim_t src_3d_dims[] = {prb->mb, prb->ic, prb->id, prb->ih, prb->iw};

    dim_t dst_1d_dims[] = {prb->mb, prb->ic, prb->ow};
    dim_t dst_2d_dims[] = {prb->mb, prb->ic, prb->oh, prb->ow};
    dim_t dst_3d_dims[] = {prb->mb, prb->ic, prb->od, prb->oh, prb->ow};

    switch (prb->ndims) {
        case 5: {
            src_dims.assign(src_3d_dims, end(src_3d_dims));
            dst_dims.assign(dst_3d_dims, end(src_3d_dims));
        } break;

        case 4: {
            src_dims.assign(src_2d_dims, end(src_2d_dims));
            dst_dims.assign(dst_2d_dims, end(dst_2d_dims));
        } break;

        case 3: {
            src_dims.assign(src_1d_dims, end(src_1d_dims));
            dst_dims.assign(dst_1d_dims, end(dst_1d_dims));
        } break;

        default: break;
    }

    src_dt = convert_dt(prb->cfg[SRC].dt);
    dst_dt = convert_dt(prb->cfg[DST].dt);

    dim_t strides_nd[] = {prb->sd, prb->sh, prb->sw};
    dim_t kernel_nd[] = {prb->kd, prb->kh, prb->kw};
    dim_t padding_l_nd[] = {prb->pd, prb->ph, prb->pw};
    dim_t padding_r_nd[] = {prb->pd_r, prb->ph_r, prb->pw_r};

    const size_t max_ndims = 5;
    const size_t offset = max_ndims - prb->ndims;

    strides.assign(strides_nd + offset, end(strides_nd));
    kernel.assign(kernel_nd + offset, end(kernel_nd));
    pads_begin.assign(padding_l_nd + offset, end(padding_l_nd));
    pads_end.assign(padding_r_nd + offset, end(padding_r_nd));

    rounding_type = "floor";
    data_format = convert_tag(prb->tag);

    op_kind = (prb->alg == ::pool::max) ? graph_op::MaxPool : graph_op::AvgPool;

    // attributes specific to pooling type
    dim_t dilation_nd[] = {prb->dd, prb->dh, prb->dw};
    dilations.assign(dilation_nd + offset, end(dilation_nd));
    // "0" dilations in oneDNN is "1" in oneDNN graph
    std::for_each(dilations.begin(), dilations.end(), [](dim_t &v) { v += 1; });

    exclude_pad = prb->alg == ::pool::avg_np;
}

fill_status_t pool_graph_prb_t::handle_main_op_() {
    using op = dnnl::graph::op;

    const std::string SRC {"pool_src"};
    const std::string DST {"pool_dst"};

    tensor_descs_.emplace(SRC, spec_.src_dt, spec_.src_dims, lt::strided);
    tensor_descs_.emplace(DST, spec_.dst_dt, spec_.dst_dims, lt::strided);

    const size_t new_op_id = ops_.size();
    const std::string op_name
            = (spec_.op_kind == op::kind::MaxPool) ? "max_pool" : "avg_pool";
    op pool(new_op_id, spec_.op_kind, {tensor_descs_[SRC]},
            {tensor_descs_[DST]}, op_name);

    pool.set_attr("strides", spec_.strides)
            .set_attr("pads_begin", spec_.pads_begin)
            .set_attr("pads_end", spec_.pads_end)
            .set_attr("kernel", spec_.kernel)
            .set_attr("rounding_type", spec_.rounding_type)
            .set_attr("data_format", spec_.data_format)
            .set_attr("auto_pad", spec_.auto_pad);

    if (spec_.op_kind == op::kind::MaxPool) {
        pool.set_attr("dilations", spec_.dilations);
    } else { // AvgPool
        pool.set_attr("exclude_pad", spec_.exclude_pad);
    }

    ops_.emplace_back(pool);
    curr_out_map_ids_.assign({DST});

    return fill_status::DONE;
}

int doit(const ::pool::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    pool_graph_prb_t graph_prb(prb);
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

    const auto &e = benchdnnext::get_test_engine();
    auto cp = par.compile(ins, outs, e);

    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto dst_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);
    dnn_mem_t ws_fp;
    std::vector<dnn_mem_t> binary_po_fp;

    auto src_dt = make_dnn_mem(ins[0], tag::abx);
    auto dst_dt = make_dnn_mem(outs[0], tag::abx);

    SAFE(fill_src(prb, src_dt, src_fp, res), WARN);
    SAFE(fill_dst(prb, dst_dt, dst_fp, res), WARN);

    dnnl::graph::tensor src_tensor(ins[0], static_cast<float *>(src_dt));
    dnnl::graph::tensor dst_tensor(outs[0], static_cast<float *>(dst_dt));

    std::vector<dnnl::graph::tensor> tensors_in {src_tensor};
    std::vector<dnnl::graph::tensor> tensors_out {dst_tensor};

    SAFE(execute_and_wait(cp, tensors_in, tensors_out), WARN);

    if (is_bench_mode(CORR)) {
        // currently we run benchmark only with dir equal FWD_I
        ::pool::compute_ref_fwd(prb, src_fp, binary_po_fp, dst_fp, ws_fp);
        compare::compare_t cmp;
        cmp.set_threshold(prb->cfg[DST].eps);
        cmp.set_data_kind(DST);
        cmp.set_zero_trust_percent(100.f); // TODO: consider enabling

        SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
    }

    measure_perf(res->timer, cp, tensors_in, tensors_out);

    return OK;
}

} // namespace pool
} // namespace benchdnnext
