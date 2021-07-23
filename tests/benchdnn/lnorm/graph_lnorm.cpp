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

#include "compare.hpp"
#include "dnnl_graph_common.hpp"
#include "lnorm/graph_lnorm.hpp"

namespace benchdnnext {
namespace lnorm {

lnorm_graph_prb_t::spec_t::spec_t(const ::lnorm::prb_t *prb) {
    dims = prb->dims;
    ss_dims = dims_t {1, prb->c};
    lnorm_dt = convert_dt(prb->dt);

    for (int i = 0; i < prb->ndims - 1; i++) {
        stat_dims.emplace_back(prb->dims[i]);
    }

    if (prb->dir != FWD_D) { keep_stats = false; }
    if (!(prb->flags & ::lnorm::USE_SCALESHIFT)) { use_affine = false; }
    epsilon = 1.f / 16;
}

fill_status_t lnorm_graph_prb_t::handle_main_op_() {
    using op = dnnl::graph::op;
    using graph_dt = dnnl::graph::logical_tensor::data_type;

    const std::string SRC {"lnorm_src"};
    const std::string GAMMA {"lnorm_gamma"};
    const std::string BETA {"lnorm_beta"};
    const std::string DST {"lnorm_dst"};
    const std::string MEAN {"lnorm_mean"};
    const std::string VAR {"lnorm_var"};

    tensor_descs_.emplace(SRC, spec_.lnorm_dt, spec_.dims, lt::strided);
    tensor_descs_.emplace(GAMMA, spec_.lnorm_dt, spec_.ss_dims, lt::strided);
    tensor_descs_.emplace(BETA, spec_.lnorm_dt, spec_.ss_dims, lt::strided);
    tensor_descs_.emplace(DST, spec_.lnorm_dt, spec_.dims, lt::strided);
    tensor_descs_.emplace(MEAN, graph_dt::f32, spec_.stat_dims, lt::strided);
    tensor_descs_.emplace(VAR, graph_dt::f32, spec_.stat_dims, lt::strided);

    std::vector<dnnl::graph::logical_tensor> ltensors_in;
    std::vector<dnnl::graph::logical_tensor> ltensors_out;

    ltensors_in.push_back({tensor_descs_[SRC]});
    ltensors_out.push_back({tensor_descs_[DST]});
    if (spec_.use_affine) {
        ltensors_in.push_back(tensor_descs_[GAMMA]);
        ltensors_in.push_back(tensor_descs_[BETA]);
    }
    if (spec_.keep_stats) {
        ltensors_out.push_back(tensor_descs_[MEAN]);
        ltensors_out.push_back(tensor_descs_[VAR]);
    }

    op lnorm_op(1, dnnl::graph::op::kind::LayerNorm, ltensors_in, ltensors_out,
            "LayerNorm");

    lnorm_op.set_attr("begin_norm_axis", spec_.begin_norm_axis);
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
    check_known_skipped_case_graph_common(
            {prb->dt}, normalize_tag(prb->tag, prb->ndims), prb->dir, res);
    if (res->state == SKIPPED) return;
    /* GLOBAL STATS cannot be passed as DNNL Graph doesnt support this */
    if (prb->flags & ::lnorm::GLOB_STATS) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
    }
    /* dnnl_use_scale and dnnl_use_shift are new features yet to be added */
    if (prb->use_sc() || prb->use_sh()) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
    }
}

/* When the error is larger than eps, It could be
 * due to catastrophic cancellation in final result
 * which is computed as `Y = a * X + b`.
 * When `a * X`  is close to `b` and `sign(a * X) = - sign(b)`.
 * Then large error in `a * X` could result in a final
 * result (which has a cancellation i.e. `|Y| = |a*X - (-b)|`)
 * which has no meaningful digits left in mantissa.*/
void add_additional_fwd_lnorm_check(const ::lnorm::prb_t *&prb,
        const dnn_mem_t &ss_fp, const dnn_mem_t &sh_fp, const dnn_mem_t &dst_fp,
        const float &eps, compare::compare_t &cmp) {
    using cmp_args_t = compare::compare_t::driver_check_func_args_t;
    const auto lnorm_add_check = [&](const cmp_args_t &args) {
        bool scale_or_shift = prb->use_ss() || prb->use_sc() || prb->use_sh();
        if (!scale_or_shift) return false;

        ::dims_t l_dims(dst_fp.md_);
        dims_t dims_idx = off2dims_idx(l_dims, args.idx);
        int64_t c = dims_idx[prb->ndims - 1];
        const float beta = prb->use_sh() ? ((const float *)sh_fp)[c]
                                         : ((const float *)ss_fp)[prb->c + c];
        /* Using an empirically derived threshold,
         * check if cancellation error
         * in `|Y| = |a*X - (-b)|` is huge.*/
        bool maybe_cancellation_error
                = (fabsf(args.got - beta)
                          / (fabsf(args.got) > FLT_MIN ? fabsf(args.got) : 1))
                > 1.0f;
        if (maybe_cancellation_error) {
            /* Check for error in `a * X` */
            float diff_aX
                    = fabsf((args.got - beta) - (args.got + args.diff - beta));
            return diff_aX <= eps;
        }
        return false;
    };
    cmp.set_driver_check_function(lnorm_add_check);
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

    const bool use_ss = prb->use_ss();
    const bool use_sc = prb->use_sc();
    const bool use_sh = prb->use_sh();

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
    dnn_mem_t src_dt = make_dnn_mem(ins[0], (prb->tag).c_str());
    dnn_mem_t &dst_fp = src_fp; // in-place reference

    dnn_mem_t placeholder_dst_dt;
    if (!prb->inplace) {
        placeholder_dst_dt = make_dnn_mem(outs[0], (prb->tag).c_str());
    }
    dnn_mem_t &dst_dt = prb->inplace ? src_dt : placeholder_dst_dt;

    dnn_mem_t mean_fp, var_fp, mean_dt, var_dt;
    if (graph_prb.spec_.keep_stats) {
        mean_dt = make_dnn_mem(outs[1], (prb->stat_tag).c_str());
        mean_fp = make_dnn_mem(outs[1], dt::f32, tag::abx);
        var_dt = make_dnn_mem(outs[2], (prb->stat_tag).c_str());
        var_fp = make_dnn_mem(outs[2], dt::f32, tag::abx);
    } else {
        /* prepare_fwd needs these memories for inference and training cases.
         * Below memories are used when keep_stats=false */
        dnnl::graph::logical_tensor mean_lt(1,
                dnnl::graph::logical_tensor::data_type::f32,
                graph_prb.spec_.stat_dims, lt::strided);
        dnnl::graph::logical_tensor var_lt(2,
                dnnl::graph::logical_tensor::data_type::f32,
                graph_prb.spec_.stat_dims, lt::strided);
        mean_dt = make_dnn_mem(mean_lt, (prb->stat_tag).c_str());
        mean_fp = make_dnn_mem(mean_lt, dt::f32, tag::abx);
        var_dt = make_dnn_mem(var_lt, (prb->stat_tag).c_str());
        var_fp = make_dnn_mem(var_lt, dt::f32, tag::abx);
    }
    dnn_mem_t ss_fp(2, dims_ss, dnnl_f32, tag::abx, cpu_engine);
    dnn_mem_t ss_dt(2, dims_ss, dnnl_f32, tag::abx, cpu_engine);
    dnnl_dim_t dims_sh[] = {prb->c};
    dnn_mem_t sh_fp(
            1, dims_sh, dnnl_f32, use_sh ? tag::x : tag::abx, cpu_engine);
    dnn_mem_t sh_dt(
            1, dims_sh, dnnl_f32, use_sh ? tag::x : tag::abx, cpu_engine);
    if (::lnorm::prepare_fwd(prb, src_fp, mean_fp, var_fp, ss_fp, sh_fp)
            != OK) {
        return res->state = MISTRUSTED, OK;
    }

    SAFE(src_dt.reorder(src_fp), WARN);
    //TODO: not tested - as global stats not there in DNNL Graph
    if (prb->flags & ::lnorm::GLOB_STATS) {
        /* prepare mean & var if they are inputs */
        SAFE(mean_dt.reorder(mean_fp), WARN);
        SAFE(var_dt.reorder(var_fp), WARN);
    }

    //TODO: when new attributes are available for SCALE or SHIFT update below.
    if (use_sh) { SAFE(sh_dt.reorder(sh_fp), WARN); }
    if (graph_prb.spec_.use_affine || use_sc) {
        SAFE(ss_dt.reorder(ss_fp), WARN);
    }

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
    if (use_sc) {
        for (int64_t i = 0; i < prb->c; i++) {
            gamma_v[i] = ss_dt.get_elem(i);
        }
        tensors_in.push_back(dnnl::graph::tensor(ins.back(), gamma_v.data()));
    }
    if (use_sh) {
        for (int64_t i = 0; i < prb->c; i++) {
            beta_v[i] = ss_dt.get_elem(prb->c + i);
        }
        tensors_in.push_back(dnnl::graph::tensor(ins.back(), beta_v.data()));
    }
    //TODO - not supported for now.
    if (graph_prb.spec_.keep_stats) {
        tensors_out.push_back(
                dnnl::graph::tensor(outs[1], static_cast<void *>(mean_dt)));
        tensors_out.push_back(
                dnnl::graph::tensor(outs[2], static_cast<void *>(var_dt)));
    }

    SAFE(execute_and_wait(cp, tensors_in, tensors_out), WARN);
    if (is_bench_mode(CORR)) {
        ::lnorm::compute_ref_fwd(
                prb, src_fp, mean_fp, var_fp, ss_fp, sh_fp, dst_fp);

        compare::compare_t cmp_data;
        const int digits_f32 = 24;
        const float eps = (1 << (digits_f32 - digits_dt(prb->dt))) * 5e-7;
        cmp_data.set_threshold(eps);
        cmp_data.set_data_kind(DATA);
        // TODO: improve bf16 filling
        if (prb->dt == dnnl_bf16) cmp_data.set_zero_trust_percent(100.f);

        add_additional_fwd_lnorm_check(
                prb, ss_fp, sh_fp, dst_fp, eps, cmp_data);
        SAFE(cmp_data.compare(dst_fp, dst_dt, prb->attr, res), WARN);

        //TODO: this will be used only when training test cases are enabled.
        if (!(prb->flags & ::lnorm::GLOB_STATS) && !(prb->dir & FLAG_INF)) {
            compare::compare_t cmp_mean;
            cmp_mean.set_data_kind(MEAN);
            if (prb->dt == dnnl_bf16 || prb->dt == dnnl_f16)
                cmp_mean.set_zero_trust_percent(100.f);
            SAFE(cmp_mean.compare(mean_fp, mean_dt, prb->attr, res), WARN);

            compare::compare_t cmp_var;
            cmp_var.set_data_kind(VAR);
            if (prb->dt == dnnl_bf16 || prb->dt == dnnl_f16)
                cmp_var.set_zero_trust_percent(100.f);
            SAFE(cmp_var.compare(var_fp, var_dt, prb->attr, res), WARN);
        }
    }
    return measure_perf(res->timer, cp, tensors_in, tensors_out);
}

} // namespace lnorm
} // namespace benchdnnext
