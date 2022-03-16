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

#include "oneapi/dnnl/dnnl_graph_buildin_ops.h"

#include "dnnl_graph_common.hpp"
#include "utils/compare.hpp"

#include "lnorm/graph_lnorm.hpp"

#include <string>
#include <vector>

namespace benchdnnext {
namespace lnorm {

lnorm_graph_prb_t::spec_t::spec_t(const ::lnorm::prb_t *prb) noexcept {
    using graph_op = dnnl::graph::op;
    is_fwd_pass = prb->dir & FLAG_FWD;

    op_kind = is_fwd_pass ? graph_op::kind::LayerNorm
                          : graph_op::kind::LayerNormBackprop;

    dims = prb->dims;
    ss_dims = dims_t {prb->c};
    lnorm_dt = convert_dt(prb->dt);

    for (int i = 0; i < prb->ndims - 1; i++) {
        stat_dims.emplace_back(prb->dims[i]);
    }

    if (prb->dir != FWD_D) { keep_stats = false; }

    use_affine = prb->use_ss();
    epsilon = 1.f / 16;
}

void check_known_skipped_case_graph(
        const ::lnorm::prb_t *prb, res_t *res) noexcept {
    check_known_skipped_case_common({prb->dt}, prb->dir, res);
    if (res->state == SKIPPED) return;
    check_known_skipped_case_graph_common(
            {prb->dt}, normalize_tag(prb->tag, prb->ndims), prb->dir, res);
    if (res->state == SKIPPED) return;
    /* GLOBAL STATS cannot be passed as DNNL Graph doesnt support this */
    if (prb->flags & ::lnorm::GLOB_STATS) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
    }
    /* STAT_TAG=tag::undef not supported */
    if (prb->stat_tag == tag::undef) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
    }
    /* oneDNN Graph supports either both scale and shift or neither of them.
     In order to run the test with the use_affine=true attribute,
     use dnnl_use_scaleshift flag (--flags=S). */
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
        const dnn_mem_t &ss_fp, const dnn_mem_t &dst_fp, const float &eps,
        compare::compare_t &cmp) {
    using cmp_args_t = compare::compare_t::driver_check_func_args_t;
    const auto lnorm_add_check = [&](const cmp_args_t &args) {
        if (!prb->use_ss()) return false;

        ::dims_t l_dims = md2dims(dst_fp.md_);
        const dims_t dims_idx = off2dims_idx(l_dims, args.idx);
        const int64_t c = dims_idx[prb->ndims - 1];
        const float beta = static_cast<const float *>(ss_fp)[prb->c + c];
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

fill_status_t lnorm_graph_prb_t::handle_main_op_() {
    using op = dnnl::graph::op;
    using graph_dt = dnnl::graph::logical_tensor::data_type;

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);

    // common for forward and backward pass
    const std::string SRC {TENSOR_ID + "_SRC"};
    const std::string GAMMA {TENSOR_ID + "_GAMMA"};
    const std::string BETA {TENSOR_ID + "_BETA"};
    const std::string MEAN {TENSOR_ID + "_MEAN"};
    const std::string VAR {TENSOR_ID + "_VAR"};

    // specific for forward pass
    const std::string DST {TENSOR_ID + "_DST"};

    // specific for backward pass
    const std::string DIFF_DST {TENSOR_ID + "_DIFF_DST"};
    const std::string DIFF_SRC {TENSOR_ID + "_DIFF_SRC"};
    const std::string DIFF_GAMMA {TENSOR_ID + "_DIFF_GAMMA"};
    const std::string DIFF_BETA {TENSOR_ID + "_DIFF_BETA"};

    //NOTE: beta, gamma, mean and variance supports only f32
    tensor_descs_.emplace(SRC, spec_.lnorm_dt, spec_.dims, lt::strided);
    tensor_descs_.emplace(GAMMA, graph_dt::f32, spec_.ss_dims, lt::strided);
    tensor_descs_.emplace(BETA, graph_dt::f32, spec_.ss_dims, lt::strided);
    tensor_descs_.emplace(MEAN, graph_dt::f32, spec_.stat_dims, lt::strided);
    tensor_descs_.emplace(VAR, graph_dt::f32, spec_.stat_dims, lt::strided);

    if (spec_.is_fwd_pass) {
        tensor_descs_.emplace(DST, spec_.lnorm_dt, spec_.dims, lt::strided);
    } else {
        tensor_descs_.emplace(
                DIFF_DST, spec_.lnorm_dt, spec_.dims, lt::strided);
        tensor_descs_.emplace(
                DIFF_SRC, spec_.lnorm_dt, spec_.dims, lt::strided);
        tensor_descs_.emplace(
                DIFF_GAMMA, graph_dt::f32, spec_.ss_dims, lt::strided);
        tensor_descs_.emplace(
                DIFF_BETA, graph_dt::f32, spec_.ss_dims, lt::strided);
    }

    std::vector<dnnl::graph::logical_tensor> inputs;
    std::vector<dnnl::graph::logical_tensor> outputs;

    inputs.push_back(tensor_descs_[SRC]);
    if (spec_.is_fwd_pass) {
        outputs.push_back(tensor_descs_[DST]);

        if (spec_.keep_stats) {
            outputs.push_back(tensor_descs_[MEAN]);
            outputs.push_back(tensor_descs_[VAR]);
        }
    } else {
        inputs.push_back(tensor_descs_[DIFF_DST]);
        inputs.push_back(tensor_descs_[MEAN]);
        inputs.push_back(tensor_descs_[VAR]);

        outputs.push_back(tensor_descs_[DIFF_SRC]);

        if (spec_.use_affine) {
            outputs.push_back(tensor_descs_[DIFF_GAMMA]);
            outputs.push_back(tensor_descs_[DIFF_BETA]);
        }
    }

    if (spec_.use_affine) {
        inputs.push_back(tensor_descs_[GAMMA]);
        inputs.push_back(tensor_descs_[BETA]);
    }

    const std::string op_name
            = spec_.is_fwd_pass ? "LayerNorm" : "LayerNormBackprop";
    op lnorm_op(new_op_id, get_main_op_kind(), inputs, outputs, op_name);

    if (spec_.is_fwd_pass) {
        lnorm_op.set_attr("keep_stats", spec_.keep_stats);
    }
    lnorm_op.set_attr("begin_norm_axis", spec_.begin_norm_axis);
    lnorm_op.set_attr("use_affine", spec_.use_affine);
    lnorm_op.set_attr("epsilon", spec_.epsilon);

    ops_.emplace_back(lnorm_op);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

int doit(const ::lnorm::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

    lnorm_graph_prb_t graph_prb(prb);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h = graph_prb.to_graph();
    const auto spec = graph_prb.spec();

    const auto partitions = graph_h.get_partitions();
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::lnorm::init_pd, prb, res, par, ins, outs);

    static const engine_t cpu_engine(dnnl_cpu);

    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto src_dt = make_dnn_mem(ins[0], (prb->tag).c_str());
    dnn_mem_t &dst_fp = src_fp; // in-place reference
    dnn_mem_t placeholder_dst_dt;
    if (!prb->inplace) {
        placeholder_dst_dt = make_dnn_mem(outs[0], (prb->tag).c_str());
    }
    dnn_mem_t &dst_dt = prb->inplace ? src_dt : placeholder_dst_dt;

    dnn_mem_t mean_fp, var_fp, mean_dt, var_dt;
    if (spec.keep_stats || prb->dir & FLAG_BWD) {
        mean_dt = make_dnn_mem(
                spec.is_fwd_pass ? outs[1] : ins[2], (prb->stat_tag).c_str());
        mean_fp = make_dnn_mem(
                spec.is_fwd_pass ? outs[1] : ins[2], dt::f32, tag::abx);
        var_dt = make_dnn_mem(
                spec.is_fwd_pass ? outs[2] : ins[3], (prb->stat_tag).c_str());
        var_fp = make_dnn_mem(
                spec.is_fwd_pass ? outs[2] : ins[3], dt::f32, tag::abx);
    } else {
        /* prepare_fwd needs these memories for inference and training cases.
         * Below memories are used when keep_stats=false */
        dnnl::graph::logical_tensor mean_lt(1,
                dnnl::graph::logical_tensor::data_type::f32, spec.stat_dims,
                lt::strided);
        dnnl::graph::logical_tensor var_lt(2,
                dnnl::graph::logical_tensor::data_type::f32, spec.stat_dims,
                lt::strided);
        mean_dt = make_dnn_mem(mean_lt, (prb->stat_tag).c_str());
        mean_fp = make_dnn_mem(mean_lt, dt::f32, tag::abx);
        var_dt = make_dnn_mem(var_lt, (prb->stat_tag).c_str());
        var_fp = make_dnn_mem(var_lt, dt::f32, tag::abx);
    }

    const size_t ndims_ss = 2;
    dnnl_dim_t dims_ss[ndims_ss] = {prb->c, prb->c};

    dnn_mem_t ss_fp(ndims_ss, dims_ss, dnnl_f32, tag::abx, cpu_engine);
    dnn_mem_t ss_dt(ndims_ss, dims_ss, dnnl_f32, tag::abx, cpu_engine);
    dnnl_dim_t dims_sh[] = {prb->c};
    // We use ss mem descriptor for both gamma and beta, so below sh is not used
    // and its declaration is needed only to pass it to native benchdnn
    // function, which fills the data.
    dnn_mem_t sh_fp(1, dims_sh, dnnl_f32, tag::abx, cpu_engine);

    std::vector<dnnl::graph::tensor> tensors_in;
    std::vector<dnnl::graph::tensor> tensors_out;

    if (prb->dir & FLAG_FWD) {
        if (::lnorm::prepare_fwd(prb, src_fp, mean_fp, var_fp, ss_fp, sh_fp)
                != OK) {
            return res->state = MISTRUSTED, OK;
        }

        SAFE(src_dt.reorder(src_fp), WARN);
        //TODO - not supported for now.
        //TODO: not tested - as global stats not there in DNNL Graph
        if (prb->flags & ::lnorm::GLOB_STATS) {
            /* prepare mean & var if they are inputs */
            SAFE(mean_dt.reorder(mean_fp), WARN);
            SAFE(var_dt.reorder(var_fp), WARN);
        }

        // oneDNN Graph supports either both scale and shift or neither of them
        if (spec.use_affine) SAFE(ss_dt.reorder(ss_fp), WARN);

        dnnl::graph::engine &eng = get_test_engine();

        tensors_in.emplace_back(
                dnnl::graph::tensor(ins[0], eng, static_cast<void *>(src_dt)));
        tensors_out.emplace_back(
                dnnl::graph::tensor(outs[0], eng, static_cast<void *>(dst_dt)));
        std::vector<float> gamma_v(prb->c, 0.f), beta_v(prb->c, 0.f);
        if (spec.use_affine) {
            for (int64_t i = 0; i < prb->c; i++) {
                gamma_v[i] = ss_dt.get_elem(i);
                beta_v[i] = ss_dt.get_elem(prb->c + i);
            }
            tensors_in.emplace_back(
                    dnnl::graph::tensor(ins[1], eng, gamma_v.data()));
            tensors_in.emplace_back(
                    dnnl::graph::tensor(ins[2], eng, beta_v.data()));
        }

        if (spec.keep_stats) {
            tensors_out.emplace_back(dnnl::graph::tensor(
                    outs[1], eng, static_cast<void *>(mean_dt)));
            tensors_out.emplace_back(dnnl::graph::tensor(
                    outs[2], eng, static_cast<void *>(var_dt)));
        }

        SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

        if (is_bench_mode(CORR)) {
            args_t ref_args;
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_MEAN, mean_fp);
            ref_args.set(DNNL_ARG_VARIANCE, var_fp);
            ref_args.set(DNNL_ARG_SCALE_SHIFT, ss_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);

            TIME_REF(::lnorm::compute_ref(prb, ref_args));

            compare::compare_t cmp;
            const int digits_f32 = 24;
            const float eps = (1 << (digits_f32 - digits_dt(prb->dt))) * 5e-7;
            cmp.set_threshold(eps);
            cmp.set_data_kind(DST);
            // TODO: improve bf16 filling
            if (prb->dt == dnnl_bf16) cmp.set_zero_trust_percent(100.f);

            add_additional_fwd_lnorm_check(prb, ss_fp, dst_fp, eps, cmp);
            SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);

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
    } else {
        dnn_mem_t d_ss_fp(ndims_ss, dims_ss, dnnl_f32, tag::abx, cpu_engine);
        dnn_mem_t d_ss_dt(ndims_ss, dims_ss, dnnl_f32, tag::abx, cpu_engine);

        dnn_mem_t placeholder_d_src_dt;
        // backward pass
        auto d_dst_fp = make_dnn_mem(ins[1], dt::f32, tag::abx);
        auto d_dst_dt = make_dnn_mem(ins[1], (prb->tag).c_str());

        dnn_mem_t &d_src_fp = d_dst_fp; // in-place in ref code
        if (!prb->inplace) {
            placeholder_d_src_dt = make_dnn_mem(outs[0], (prb->tag).c_str());
        }
        dnn_mem_t &d_src_dt = prb->inplace ? d_dst_dt : placeholder_d_src_dt;

        if (prepare_bwd(prb, src_fp, d_dst_fp, mean_fp, var_fp, ss_fp, sh_fp)
                != OK) {
            return res->state = MISTRUSTED, OK;
        }

        SAFE(src_dt.reorder(src_fp), WARN);
        SAFE(d_dst_dt.reorder(d_dst_fp), WARN);
        SAFE(mean_dt.reorder(mean_fp), WARN);
        SAFE(var_dt.reorder(var_fp), WARN);
        if (spec.use_affine) SAFE(ss_dt.reorder(ss_fp), WARN);

        dnnl::graph::engine &eng = get_test_engine();

        dnnl::graph::tensor src_tensor(
                ins[0], eng, static_cast<void *>(src_dt));
        dnnl::graph::tensor d_dst_tensor(
                ins[1], eng, static_cast<void *>(d_dst_dt));
        dnnl::graph::tensor mean_tensor(
                ins[2], eng, static_cast<void *>(mean_dt));
        dnnl::graph::tensor var_tensor(
                ins[3], eng, static_cast<void *>(var_dt));

        dnnl::graph::tensor d_src_tensor(
                outs[0], eng, static_cast<void *>(d_src_dt));

        dnnl::graph::tensor gamma_tensor, beta_tensor, d_gamma_tensor,
                d_beta_tensor;

        std::vector<float> gamma_v(prb->c, 0.f), beta_v(prb->c, 0.f),
                d_gamma_v(prb->c, 0.f), d_beta_v(prb->c, 0.f);
        if (spec.use_affine) {
            for (int64_t i = 0; i < prb->c; i++) {
                gamma_v[i] = ss_dt.get_elem(i);
                beta_v[i] = ss_dt.get_elem(prb->c + i);
            }
            gamma_tensor = dnnl::graph::tensor(
                    ins[4], eng, static_cast<void *>(gamma_v.data()));
            beta_tensor = dnnl::graph::tensor(
                    ins[5], eng, static_cast<void *>(beta_v.data()));
            d_gamma_tensor = dnnl::graph::tensor(
                    outs[1], eng, static_cast<void *>(d_gamma_v.data()));
            d_beta_tensor = dnnl::graph::tensor(
                    outs[2], eng, static_cast<void *>(d_beta_v.data()));
        }

        tensors_in.push_back(src_tensor);
        tensors_in.push_back(d_dst_tensor);
        tensors_in.push_back(mean_tensor);
        tensors_in.push_back(var_tensor);

        tensors_out.push_back(d_src_tensor);

        if (spec.use_affine) {
            tensors_in.push_back(gamma_tensor);
            tensors_in.push_back(beta_tensor);

            tensors_out.push_back(d_gamma_tensor);
            tensors_out.push_back(d_beta_tensor);
        }

        SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

        if (spec.use_affine) {
            for (int64_t i = 0; i < prb->c; i++) {
                d_ss_dt.set_elem(i, d_gamma_v[i]);
                d_ss_dt.set_elem(prb->c + i, d_beta_v[i]);
            }
        }

        if (is_bench_mode(CORR)) {
            args_t ref_args;
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_MEAN, mean_fp);
            ref_args.set(DNNL_ARG_VARIANCE, var_fp);
            ref_args.set(DNNL_ARG_SCALE_SHIFT, ss_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, d_dst_fp);
            ref_args.set(DNNL_ARG_DIFF_SRC, d_src_fp);
            ref_args.set(DNNL_ARG_DIFF_SCALE_SHIFT, d_ss_fp);

            TIME_REF(::lnorm::compute_ref(prb, ref_args));

            compare::compare_t cmp_data;
            const int digits_f32 = 24;
            const float eps = (1 << (digits_f32 - digits_dt(prb->dt))) * 2e-7;
            cmp_data.set_threshold(eps);
            cmp_data.set_data_kind(SRC);
            cmp_data.set_zero_trust_percent(70.f);
            SAFE(cmp_data.compare(d_src_fp, d_src_dt, prb->attr, res), WARN);

            if (spec.use_affine && (prb->dir & FLAG_WEI)) {
                compare::compare_t cmp_ss;
                cmp_ss.set_threshold(eps);
                cmp_ss.set_data_kind(SC);
                SAFE(cmp_ss.compare(d_ss_fp, d_ss_dt, prb->attr, res), WARN);
            }
        }
    }

    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);

    return OK;
}

} // namespace lnorm
} // namespace benchdnnext
