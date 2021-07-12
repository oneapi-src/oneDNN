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

#include "compare.hpp"
#include "dnnl_graph_common.hpp"

#include "binary/binary.hpp"
#include "binary/graph_binary.hpp"

namespace benchdnnext {
namespace binary {

binary_graph_prb_t::spec_t::spec_t(const ::binary::prb_t *prb) {
    src0_dims = prb->sdims[0];
    src1_dims = prb->sdims[1];

    const auto ndims = prb->ndims[0];
    dst_dims.reserve(ndims);
    for (auto d = 0; d < ndims; ++d)
        dst_dims.push_back(std::max(src0_dims[d], src1_dims[d]));

    src0_dt = convert_dt(prb->sdt[0]);
    src1_dt = convert_dt(prb->sdt[1]);
    dst_dt = convert_dt(prb->ddt);

    src0_tag = convert_tag(prb->stag[0]);
    src1_tag = convert_tag(prb->stag[1]);
    dst_tag = convert_tag(prb->dtag);

    op_kind = convert_alg_kind(attr_t::post_ops_t::kind2dnnl_kind(prb->alg));
}

void check_broadcast_rules(const ::binary::prb_t *prb, res_t *res) {
    const auto src0_dims = prb->sdims[0];
    const auto src1_dims = prb->sdims[1];

    // General broadcast rules:
    // Two dimensions are compatible when
    // 1) they are equal, or
    // 2) one of them is 1
    for (auto d = 0; d < prb->ndims[0]; ++d) {
        if (src0_dims[d] == src1_dims[d] || src0_dims[d] == 1
                || src1_dims[d] == 1) {
            continue;
        } else {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }
}

void check_known_skipped_case_graph(const ::binary::prb_t *prb, res_t *res) {
    using p = attr_t::post_ops_t;
    // TODO (kgajdamo):
    // Divide op is not supported at the moment.
    // Remove below when divide will be enabled.
    if (prb->alg == p::kind_t::DIV) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
    // MAX, MUL, MIN supports relu, sigmoid, sum post-ops.
    // ADD supports relu and sigmoid post-ops.
    // Other cases are being skipped.
    for (const auto &po : prb->attr.post_ops.entry) {
        if (po.kind == p::RELU || po.kind == p::LOGISTIC
                || (po.is_sum_kind() && prb->alg != p::kind_t::ADD)) {
            continue;
        } else {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }
}

fill_status_t binary_graph_prb_t::handle_main_op_() {
    const std::string SRC0 {"bin_src0"};
    const std::string SRC1 {"bin_src1"};
    const std::string DST {"bin_dst"};

    tensor_descs_.emplace(SRC0, spec_.src0_dt, spec_.src0_dims, lt::strided);
    tensor_descs_.emplace(SRC1, spec_.src1_dt, spec_.src1_dims, lt::strided);
    tensor_descs_.emplace(DST, spec_.dst_dt, spec_.dst_dims, lt::strided);

    const size_t new_op_id = ops_.size();

    dnnl::graph::op binary(new_op_id, spec_.op_kind,
            {tensor_descs_[SRC0], tensor_descs_[SRC1]}, {tensor_descs_[DST]},
            "binary");

    binary.set_attr("auto_broadcast", spec_.auto_broadcast);
    binary.set_attr("backend", spec_.backend);

    ops_.emplace_back(binary);
    curr_out_map_ids_.assign({DST});

    return fill_status::DONE;
}

fill_status_t binary_graph_prb_t::handle_elt_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.binary.eltw_handler(*this, po_entry);
}

fill_status_t binary_graph_prb_t::handle_sum_() {
    return po_handler.binary.sum_handler(*this);
}

int doit(const ::binary::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    ::binary::check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;
    check_broadcast_rules(prb, res);
    if (res->state == SKIPPED) return OK;

    binary_graph_prb_t graph_prb(prb);
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

    auto src0_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto src1_fp = make_dnn_mem(ins[1], dt::f32, tag::abx);
    auto dst_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);
    // binary operator doesn't support binary post-ops yet
    std::vector<dnn_mem_t> binary_po_fp;

    auto src0_dt = make_dnn_mem(ins[0], prb->stag[0]);
    auto src1_dt = make_dnn_mem(ins[1], prb->stag[1]);
    auto dst_dt = make_dnn_mem(outs[0], prb->dtag);

    SAFE(::binary::fill_mem(0, src0_dt, src0_fp), WARN);
    SAFE(::binary::fill_mem(1, src1_dt, src1_fp), WARN);
    SAFE(::binary::fill_mem(2, dst_dt, dst_fp), WARN);

    dnnl::graph::tensor src0_tensor(ins[0], static_cast<void *>(src0_dt));
    dnnl::graph::tensor src1_tensor(ins[1], static_cast<void *>(src1_dt));
    dnnl::graph::tensor dst_tensor(outs[0], static_cast<void *>(dst_dt));
    dnnl::graph::tensor sum_src1_tensor;

    std::vector<dnnl::graph::tensor> tensors_in {src0_tensor, src1_tensor};
    std::vector<dnnl::graph::tensor> tensors_out {dst_tensor};

    if (graph_prb.has_post_sum()) {
        sum_src1_tensor
                = dnnl::graph::tensor(ins.back(), static_cast<void *>(dst_dt));
        tensors_in.emplace_back(sum_src1_tensor);
    }

    SAFE(execute_and_wait(cp, tensors_in, tensors_out), WARN);

    if (is_bench_mode(CORR)) {
        ::binary::compute_ref(prb, src0_fp, src1_fp, binary_po_fp, dst_fp);
        compare::compare_t cmp;

        cmp.set_threshold(epsilon_dt(dst_dt.dt()));
        const auto binary_add_check =
                [&](const compare::compare_t::driver_check_func_args_t &args) {
                    // fp16 result can slightly mismatch for division due to difference
                    // in backends implementations.
                    return prb->alg == attr_t::post_ops_t::kind_t::DIV
                            ? args.diff < epsilon_dt(args.dt)
                            : false;
                };
        cmp.set_driver_check_function(binary_add_check);

        SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
    }

    measure_perf(res->timer, cp, tensors_in, tensors_out);

    return OK;
}

} // namespace binary
} // namespace benchdnnext
