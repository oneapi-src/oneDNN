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

#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl_graph_buildin_ops.h"

#include "dnnl_graph_common.hpp"
#include "lnorm/graph_lnorm.hpp"

namespace benchdnnext {
namespace lnorm {

lnorm_graph_prb_t::spec_t::spec_t(const ::lnorm::prb_t *prb) {
    dims = prb->dims;
    ss_dims = dims_t {1, prb->c};
    lnorm_dt = convert_dt(prb->dt);

    //TOOD - not supported for now. NOT USED only for training.
    keep_stats = false;
    if (!(prb->flags & ::lnorm::GLOB_STATS)) { keep_stats = false; }
    if (!(prb->flags & ::lnorm::USE_SCALESHIFT)) { use_affine = false; }
    epsilon = 1.f / 16;
}

fill_status_t lnorm_graph_prb_t::handle_main_op_() {
    using op = dnnl::graph::op;

    const std::string SRC {"lnorm_src"};
    const std::string GAMMA {"lnorm_gamma"};
    const std::string BETA {"lnorm_beta"};
    const std::string DST {"lnorm_dst"};
    const std::string MEAN {"lnorm_mean"};
    const std::string VAR {"lnorm_mean"};

    tensor_descs_.emplace(SRC, spec_.lnorm_dt, spec_.dims, lt::strided);
    tensor_descs_.emplace(GAMMA, spec_.lnorm_dt, spec_.ss_dims, lt::strided);
    tensor_descs_.emplace(BETA, spec_.lnorm_dt, spec_.ss_dims, lt::strided);
    tensor_descs_.emplace(DST, spec_.lnorm_dt, spec_.dims, lt::strided);
    tensor_descs_.emplace(MEAN, spec_.lnorm_dt, spec_.dims, lt::strided);
    tensor_descs_.emplace(VAR, spec_.lnorm_dt, spec_.dims, lt::strided);

    std::vector<dnnl::graph::logical_tensor> ltensors_in;
    std::vector<dnnl::graph::logical_tensor> ltensors_out;

    ltensors_in.push_back({tensor_descs_[SRC]});
    ltensors_out.push_back({tensor_descs_[DST]});
    if (spec_.use_affine) {
        ltensors_in.push_back(tensor_descs_[GAMMA]);
        ltensors_in.push_back(tensor_descs_[BETA]);
    }
#if 0 //TODO:  - not supported for now.
    if (spec_.keep_stats) {
        ltensors_out.push_back(tensor_descs_[MEAN]);
        ltensors_out.push_back(tensor_descs_[VAR]);
    }
#endif
    op lnorm_op(1, dnnl::graph::op::kind::LayerNorm, ltensors_in, ltensors_out,
            "LayerNorm");

    lnorm_op.set_attr("keep_stats", spec_.keep_stats);
    lnorm_op.set_attr("use_affine", spec_.use_affine);
    lnorm_op.set_attr("epsilon", spec_.epsilon);

    ops_.emplace_back(lnorm_op);
    curr_out_map_ids_.assign({"lnorm_dst"});

    return fill_status::DONE;
}

void check_known_skipped_case(const ::lnorm::prb_t *prb, res_t *res) {
    check_known_skipped_case_common({prb->dt}, prb->dir, res);
    if (res->state == SKIPPED) return;
}

int doit(const ::lnorm::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    lnorm_graph_prb_t graph_prb(prb);
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

    const auto &e = benchdnnext::get_test_engine();
    auto cp = par.compile(ins, outs, e);

    dnnl_dim_t dims_ss[2];
    dims_ss[0] = prb->c;
    dims_ss[1] = prb->c;
    static const engine_t cpu_engine(dnnl_cpu);

    dnn_mem_t src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    dnn_mem_t ss_fp(2, dims_ss, dnnl_f32, tag::abx, cpu_engine);
    dnn_mem_t &dst_fp = src_fp; // in-place reference
    dnn_mem_t mean_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);
    dnn_mem_t var_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);

    dnn_mem_t placeholder_dst_dt;
    if (!prb->inplace) { placeholder_dst_dt = make_dnn_mem(outs[0], tag::abx); }

    if (::lnorm::prepare_fwd(prb, src_fp, mean_fp, var_fp, ss_fp) != OK) {
        return res->state = MISTRUSTED, OK;
    }

    //TODO: stat_tag from prb not passed.. used tag::abx for status.
    dnn_mem_t src_dt = make_dnn_mem(ins[0], tag::abx);
    dnn_mem_t mean_dt = make_dnn_mem(outs[0], tag::abx);
    dnn_mem_t var_dt = make_dnn_mem(outs[0], tag::abx);
    dnn_mem_t ss_dt(2, dims_ss, prb->dt, tag::abx, cpu_engine);
    dnn_mem_t &dst_dt = prb->inplace ? src_dt : placeholder_dst_dt;

    SAFE(src_dt.reorder(src_fp), WARN);
#if 0 //TODO - not supported for now.
    //TODO: not tested - as global stats not there in DNNL Graph
    if (prb->flags & ::lnorm::GLOB_STATS) {
            /* prepare mean & var if they are inputs */
            SAFE(mean_dt.reorder(mean_fp), WARN);
            SAFE(var_dt.reorder(var_fp), WARN);
    }
#endif
    if (graph_prb.spec_.use_affine) { SAFE(ss_dt.reorder(ss_fp), WARN); }

    std::vector<dnnl::graph::tensor> tensors_in;
    std::vector<dnnl::graph::tensor> tensors_out;
    tensors_in.push_back(
            dnnl::graph::tensor(ins[0], static_cast<void *>(src_dt)));
    tensors_out.push_back(
            dnnl::graph::tensor(outs[0], static_cast<void *>(dst_dt)));
    std::vector<float> gamma_v(prb->c, 0.f), beta_v(prb->c, 0.f);
    if (graph_prb.spec_.use_affine) {
        for (int64_t i = 0; i < prb->c; i++) {
            gamma_v[i] = ss_dt.get_elem(i);
            beta_v[i] = ss_dt.get_elem(prb->c + i);
        }
        tensors_in.push_back(dnnl::graph::tensor(ins[1], gamma_v.data()));
        tensors_in.push_back(dnnl::graph::tensor(ins[2], beta_v.data()));
    }
#if 0 //TOOD - not supported for now.
    if (graph_prb.spec_.keep_stats) {
        tensors_out.push_back(dnnl::graph::tensor(outs[1], static_cast<void *>(mean_dt)));
        tensors_out.push_back(dnnl::graph::tensor(outs[2], static_cast<void *>(var_dt)));
    }
#endif
    SAFE(execute_and_wait(cp, tensors_in, tensors_out), WARN);
    if (bench_mode & CORR) {
        ::lnorm::compute_ref_fwd(prb, src_fp, mean_fp, var_fp, ss_fp, dst_fp);
        //TODO: this will be used only when training test cases are enabled.
        if (!(prb->flags & ::lnorm::GLOB_STATS) && !(prb->dir & FLAG_INF)) {
            dnn_mem_t mean(mean_dt, prb->dt, tag::abx, cpu_engine);
            dnn_mem_t var(var_dt, prb->dt, tag::abx, cpu_engine);
            SAFE(::lnorm::compare(prb, MEAN, mean_fp, mean, res), WARN);
            SAFE(::lnorm::compare(prb, VAR, var_fp, var, res), WARN);
        }

        dnn_mem_t dst(dst_dt, prb->dt, tag::abx, cpu_engine);
        SAFE(::lnorm::compare(prb, DATA, dst_fp, dst, res, &ss_fp), WARN);
    }
    measure_perf(res->timer, cp, tensors_in, tensors_out);
    return OK;
}

} // namespace lnorm
} // namespace benchdnnext
