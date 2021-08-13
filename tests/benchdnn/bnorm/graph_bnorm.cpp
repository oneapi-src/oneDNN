/*******************************************************************************
 * * Copyright 2019-2021 Intel Corporation
 * *
 * * Licensed under the Apache License, Version 2.0 (the "License");
 * * you may not use this file except in compliance with the License.
 * * You may obtain a copy of the License at
 * *
 * *     http://www.apache.org/licenses/LICENSE-2.0
 * *
 * * Unless required by applicable law or agreed to in writing, software
 * * distributed under the License is distributed on an "AS IS" BASIS,
 * * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * * See the License for the specific language governing permissions and
 * * limitations under the License.
 * *******************************************************************************/

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_graph_buildin_ops.h"

#include "dnnl_common.hpp"
#include "dnnl_graph_common.hpp"
#include "dnnl_memory.hpp"

#include "bnorm/bnorm.hpp"
#include "bnorm/graph_bnorm.hpp"
#include "norm.hpp"

#include <string>
#include <vector>

namespace benchdnnext {
namespace bnorm {

bnorm_graph_prb_t::spec_t::spec_t(const ::bnorm::prb_t *prb) noexcept {
    const dims_t dims_0d = {prb->mb, prb->ic};
    const dims_t dims_1d = {prb->mb, prb->ic, prb->iw};
    const dims_t dims_2d = {prb->mb, prb->ic, prb->ih, prb->iw};
    const dims_t dims_3d = {prb->mb, prb->ic, prb->id, prb->ih, prb->iw};
    dims = [&](int n) {
        switch (n) {
            case 5: return dims_3d;
            case 4: return dims_2d;
            case 3: return dims_1d;
            default: return dims_0d;
        }
    }(prb->ndims);

    s_dims = {prb->ic};
    bnorm_dt = convert_dt(prb->dt);
    epsilon = prb->eps;
    tag = prb->tag;
}

void check_known_skipped_case_graph(
        const ::bnorm::prb_t *prb, res_t *res) noexcept {
    check_known_skipped_case_common({prb->dt}, prb->dir, res);
    if (res->state == SKIPPED) return;

    for (const auto &po : prb->attr.post_ops.entry) {
        if (po.kind == attr_t::post_ops_t::RELU) {
            continue;
        } else {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }
}

fill_status_t bnorm_graph_prb_t::handle_main_op_() {
    using op = dnnl::graph::op;

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);
    const std::string SRC {TENSOR_ID + "_SRC"};
    const std::string SCALE {TENSOR_ID + "_SCALE"};
    const std::string SHIFT {TENSOR_ID + "_SHIFT"};
    const std::string MEAN {TENSOR_ID + "_MEAN"};
    const std::string VAR {TENSOR_ID + "_VAR"};
    const std::string DST {TENSOR_ID + "_DST"};

    tensor_descs_.emplace(SRC, spec_.bnorm_dt, spec_.dims, lt::strided);
    tensor_descs_.emplace(SCALE, spec_.bnorm_dt, spec_.s_dims, lt::strided);
    tensor_descs_.emplace(SHIFT, spec_.bnorm_dt, spec_.s_dims, lt::strided);
    tensor_descs_.emplace(MEAN, spec_.bnorm_dt, spec_.s_dims, lt::strided);
    tensor_descs_.emplace(VAR, spec_.bnorm_dt, spec_.s_dims, lt::strided);
    tensor_descs_.emplace(DST, spec_.bnorm_dt, spec_.dims, lt::strided);

    op bnorm_op(new_op_id, dnnl::graph::op::kind::BatchNormInference,
            {tensor_descs_[SRC], tensor_descs_[SCALE], tensor_descs_[SHIFT],
                    tensor_descs_[MEAN], tensor_descs_[VAR]},
            {tensor_descs_[DST]}, "bnorm");

    bnorm_op.set_attr("epsilon", spec_.epsilon);
    bnorm_op.set_attr<std::string>("data_format", convert_tag(spec_.tag));

    ops_.emplace_back(bnorm_op);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

fill_status_t bnorm_graph_prb_t::handle_elt_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.bnorm.eltw_handler(*this, po_entry);
}

int doit(const ::bnorm::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

    bnorm_graph_prb_t graph_prb(prb);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h = graph_prb.to_graph();

    const auto partitions = graph_h.get_partitions();
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::bnorm::init_pd, prb, res, par, ins, outs);

    dnnl_dim_t data_dims[] = {prb->mb, prb->ic, prb->ih, prb->iw};

    static const engine_t cpu_engine(dnnl_cpu);

    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto shift_fp
            = make_dnn_mem(ins[2], dt::f32, prb->use_sh() ? tag::x : tag::axb);
    auto mean_fp = make_dnn_mem(ins[3], dt::f32, tag::abx);
    auto var_fp = make_dnn_mem(ins[4], dt::f32, tag::abx);
    dnn_mem_t &dst_fp = src_fp; // in-place reference
    dnn_mem_t src_hat_fp(4, data_dims, dnnl_f32, tag::abx, cpu_engine);
    dnn_mem_t ws_fp(4, data_dims, dnnl_u8, tag::abx, cpu_engine);

    const auto placeholder_dst_dt = make_dnn_mem(outs[0], tag::abx);
    auto src_dt = make_dnn_mem(ins[0], tag::abx);
    auto shift_dt = make_dnn_mem(ins[2], prb->use_sh() ? tag::x : tag::axb);
    auto mean_dt = make_dnn_mem(ins[3], tag::abx);
    auto var_dt = make_dnn_mem(ins[4], tag::abx);
    const dnn_mem_t &dst_dt = prb->inplace ? src_dt : placeholder_dst_dt;

    dnn_mem_t scale_fp, scale_dt;
    if (prb->use_sc() || prb->use_sh()) {
        scale_fp = make_dnn_mem(ins[1], dt::f32, tag::abx);
        scale_dt = make_dnn_mem(ins[1], dt::f32, tag::abx);
    } else {
        dnnl_dim_t dims_ss[2];
        dims_ss[0] = prb->ic;
        dims_ss[1] = prb->ic;
        scale_fp = dnn_mem_t(2, dims_ss, dnnl_f32, tag::abx, cpu_engine);
        scale_dt = dnn_mem_t(2, dims_ss, dnnl_f32, tag::abx, cpu_engine);
    }

    if (::bnorm::prepare_fwd(prb, src_fp, mean_fp, var_fp, scale_fp, shift_fp)
            != OK) {
        return res->state = MISTRUSTED, OK;
    }
    /*  When dnnl_use_scaleshift is used, benchdnn populates data
        to the same memory for scale and shift and dnnlgraph expects
        the data in scale and shift. Hence this explicit copy. */
    if (!(prb->use_sc() || prb->use_sh())) {
        for (int64_t i = 0; i < prb->ic; i++) {
            ((float *)shift_fp)[i] = ((float *)scale_fp)[prb->ic + i];
        }
    }
    SAFE(src_dt.reorder(src_fp), WARN);
    SAFE(scale_dt.reorder(scale_fp), WARN);
    SAFE(shift_dt.reorder(shift_fp), WARN);
    SAFE(mean_dt.reorder(mean_fp), WARN);
    SAFE(var_dt.reorder(var_fp), WARN);

    std::vector<dnnl::graph::tensor> tensors_in, tensors_out;
    tensors_in.emplace_back(ins[0], static_cast<void *>(src_dt));
    tensors_in.emplace_back(ins[1], static_cast<void *>(scale_dt));
    tensors_in.emplace_back(ins[2], static_cast<void *>(shift_dt));
    tensors_in.emplace_back(ins[3], static_cast<void *>(mean_dt));
    tensors_in.emplace_back(ins[4], static_cast<void *>(var_dt));
    tensors_out.emplace_back(outs[0], static_cast<void *>(dst_dt));
    SAFE(execute_and_wait(cp, tensors_in, tensors_out), WARN);

    if (is_bench_mode(CORR)) {
        static const engine_t cpu_engine(dnnl_cpu);
        ::bnorm::compute_ref_fwd(prb, src_fp, mean_fp, var_fp, scale_fp,
                shift_fp, ws_fp, dst_fp, src_hat_fp);
        if (prb->dir & FLAG_FWD) {
            if (!(prb->flags & ::bnorm::GLOB_STATS) && !(prb->dir & FLAG_INF)) {
                SAFE(::bnorm::compare(prb, MEAN, mean_fp, mean_dt, res), WARN);
                SAFE(::bnorm::compare(prb, VAR, var_fp, var_dt, res), WARN);
            }
            dnn_mem_t dst(dst_dt, dnnl_f32, tag::abx, cpu_engine);
            SAFE(::bnorm::compare(
                         prb, DATA, dst_fp, dst, res, &scale_fp, &shift_fp),
                    WARN);
        }
    }

    SAFE(measure_perf(res->timer, cp, tensors_in, tensors_out), WARN);

    return OK;
}

} // namespace bnorm
} // namespace benchdnnext
