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

#include "dnnl_graph_common.hpp"
#include "utils/compare.hpp"

#include "lnorm/graph_lnorm.hpp"

#include <string>
#include <vector>

namespace benchdnnext {
namespace lnorm {

static std::vector<int64_t> get_stat_dims(
        const std::vector<int64_t> &lnorm_dims) {
    return std::vector<int64_t>(
            lnorm_dims.begin(), lnorm_dims.begin() + (lnorm_dims.size() - 1));
}

void check_known_skipped_case_graph(
        const ::lnorm::prb_t *prb, res_t *res) noexcept {
    // TODO: to align with original benchdnn, we should consider moving
    // skip_unimplemented_prb call after compilation step
    skip_invalid_and_unimplemented_prb(prb, res);
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

fill_status_t lnorm_graph_prb_t::handle_main_op_(const ::lnorm::prb_t *prb) {
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

    const dims_t ss_dims {prb->c};
    const dims_t stat_dims = get_stat_dims(prb->dims);
    const graph_dt lnorm_dt {convert_dt(prb->dt)};

    //NOTE: beta, gamma, mean and variance supports only f32
    tensor_descs_.emplace(SRC, lnorm_dt, prb->dims, lt::strided);
    tensor_descs_.emplace(GAMMA, graph_dt::f32, ss_dims, lt::strided);
    tensor_descs_.emplace(BETA, graph_dt::f32, ss_dims, lt::strided);
    tensor_descs_.emplace(MEAN, graph_dt::f32, stat_dims, lt::strided);
    tensor_descs_.emplace(VAR, graph_dt::f32, stat_dims, lt::strided);

    if (prb->dir & FLAG_FWD) {
        tensor_descs_.emplace(DST, lnorm_dt, prb->dims, lt::strided);
    } else {
        tensor_descs_.emplace(DIFF_DST, lnorm_dt, prb->dims, lt::strided);
        tensor_descs_.emplace(DIFF_SRC, lnorm_dt, prb->dims, lt::strided);
        tensor_descs_.emplace(DIFF_GAMMA, graph_dt::f32, ss_dims, lt::strided);
        tensor_descs_.emplace(DIFF_BETA, graph_dt::f32, ss_dims, lt::strided);
    }

    std::vector<dnnl::graph::logical_tensor> inputs;
    std::vector<dnnl::graph::logical_tensor> outputs;

    inputs.push_back(tensor_descs_[SRC]);
    if (prb->dir & FLAG_FWD) {
        outputs.push_back(tensor_descs_[DST]);

        if (prb->dir == FWD_D) {
            outputs.push_back(tensor_descs_[MEAN]);
            outputs.push_back(tensor_descs_[VAR]);
        }
    } else {
        inputs.push_back(tensor_descs_[DIFF_DST]);
        inputs.push_back(tensor_descs_[MEAN]);
        inputs.push_back(tensor_descs_[VAR]);

        outputs.push_back(tensor_descs_[DIFF_SRC]);

        if (prb->use_ss()) {
            outputs.push_back(tensor_descs_[DIFF_GAMMA]);
            outputs.push_back(tensor_descs_[DIFF_BETA]);
        }
    }

    if (prb->use_ss()) {
        inputs.push_back(tensor_descs_[GAMMA]);
        inputs.push_back(tensor_descs_[BETA]);
    }

    const auto op_kind = prb->dir & FLAG_FWD
            ? dnnl::graph::op::kind::LayerNorm
            : dnnl::graph::op::kind::LayerNormBackprop;

    op lnorm_op(new_op_id, op_kind, inputs, outputs, "layernorm");

    lnorm_op.set_attr("begin_norm_axis", int64_t(-1))
            .set_attr("use_affine", prb->use_ss())
            .set_attr("epsilon", float(1.f / 16));
    if (prb->dir & FLAG_FWD) lnorm_op.set_attr("keep_stats", prb->dir == FWD_D);

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

    const auto partitions = graph_h.get_partitions();
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::lnorm::init_pd, prb, res, par, ins, outs);

    static const engine_t cpu_engine(dnnl_cpu);

    const bool keep_stats {prb->dir == FWD_D};
    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto src_dt = make_dnn_mem(ins[0], (prb->tag).c_str());
    dnn_mem_t &dst_fp = src_fp; // in-place reference
    dnn_mem_t placeholder_dst_dt;
    if (!prb->inplace) {
        placeholder_dst_dt = make_dnn_mem(outs[0], (prb->tag).c_str());
    }
    dnn_mem_t &dst_dt = prb->inplace ? src_dt : placeholder_dst_dt;

    dnn_mem_t mean_fp, var_fp, mean_dt, var_dt;
    if (keep_stats || prb->dir & FLAG_BWD) {
        mean_dt = make_dnn_mem(prb->dir & FLAG_FWD ? outs[1] : ins[2],
                (prb->stat_tag).c_str());
        mean_fp = make_dnn_mem(
                prb->dir & FLAG_FWD ? outs[1] : ins[2], dt::f32, tag::abx);
        var_dt = make_dnn_mem(prb->dir & FLAG_FWD ? outs[2] : ins[3],
                (prb->stat_tag).c_str());
        var_fp = make_dnn_mem(
                prb->dir & FLAG_FWD ? outs[2] : ins[3], dt::f32, tag::abx);
    } else {
        /* prepare_fwd needs these memories for inference and training cases.
         * Below memories are used when keep_stats=false */
        dnnl::graph::logical_tensor mean_lt(1,
                dnnl::graph::logical_tensor::data_type::f32,
                get_stat_dims(prb->dims), lt::strided);
        dnnl::graph::logical_tensor var_lt(2,
                dnnl::graph::logical_tensor::data_type::f32,
                get_stat_dims(prb->dims), lt::strided);
        mean_dt = make_dnn_mem(mean_lt, (prb->stat_tag).c_str());
        mean_fp = make_dnn_mem(mean_lt, dt::f32, tag::abx);
        var_dt = make_dnn_mem(var_lt, (prb->stat_tag).c_str());
        var_fp = make_dnn_mem(var_lt, dt::f32, tag::abx);
    }

    // scale and shift are combined in a single 2D tensor of shape 2xC
    // this logical_tensor is used to indicate the memory size for scale
    dnnl::graph::logical_tensor lt_ss {3, dt::f32, {2, prb->c},
            dnnl::graph::logical_tensor::layout_type::strided};
    auto ss_fp = make_dnn_mem(lt_ss, dt::f32, tag::abx);
    auto ss_dt = make_dnn_mem(lt_ss, dt::f32, tag::abx);
    // We use ss mem descriptor for both gamma and beta, so below sh is not used
    // and its declaration is needed only to pass it to native benchdnn
    // function, which fills the data.
    dnnl::graph::logical_tensor lt_sh {4, dt::f32, dims_t {prb->c},
            dnnl::graph::logical_tensor::layout_type::strided};
    auto sh_fp = make_dnn_mem(lt_sh, dt::f32, tag::x);

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
        if (prb->use_ss()) SAFE(ss_dt.reorder(ss_fp), WARN);

        const dnnl::graph::engine &eng = get_test_engine();

        tensors_in.emplace_back(
                dnnl::graph::tensor(ins[0], eng, static_cast<void *>(src_dt)));
        tensors_out.emplace_back(
                dnnl::graph::tensor(outs[0], eng, static_cast<void *>(dst_dt)));
        auto gamma_v = make_dnn_mem(lt_sh, dt::f32, tag::x);
        auto beta_v = make_dnn_mem(lt_sh, dt::f32, tag::x);
        if (prb->use_ss()) {
            for (int64_t i = 0; i < prb->c; i++) {
                gamma_v.set_elem(i, ss_dt.get_elem(i));
                beta_v.set_elem(i, ss_dt.get_elem(prb->c + i));
            }
            tensors_in.emplace_back(dnnl::graph::tensor(
                    ins[1], eng, static_cast<void *>(gamma_v)));
            tensors_in.emplace_back(dnnl::graph::tensor(
                    ins[2], eng, static_cast<void *>(beta_v)));
        }

        if (keep_stats) {
            tensors_out.emplace_back(dnnl::graph::tensor(
                    outs[1], eng, static_cast<void *>(mean_dt)));
            tensors_out.emplace_back(dnnl::graph::tensor(
                    outs[2], eng, static_cast<void *>(var_dt)));
        }

        SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

        if (is_bench_mode(CORR)) {
            args_t args;
            args.set(DNNL_ARG_DST, dst_dt);

            args_t ref_args;
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_MEAN, mean_fp);
            ref_args.set(DNNL_ARG_VARIANCE, var_fp);
            ref_args.set(DNNL_ARG_SCALE_SHIFT, ss_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);

            std::vector<data_kind_t> kinds {DST};
            if (!(prb->flags & ::lnorm::GLOB_STATS) && !(prb->dir & FLAG_INF)) {
                args.set(DNNL_ARG_MEAN, mean_dt);
                args.set(DNNL_ARG_VARIANCE, var_dt);
                kinds.push_back(MEAN);
                kinds.push_back(VAR);
            }

            check_correctness(
                    prb, kinds, args, ref_args, ::lnorm::setup_cmp, res);
        }
    } else {
        auto d_ss_fp = make_dnn_mem(lt_ss, dt::f32, tag::abx);
        auto d_ss_dt = make_dnn_mem(lt_ss, dt::f32, tag::abx);

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
        if (prb->use_ss()) SAFE(ss_dt.reorder(ss_fp), WARN);

        const dnnl::graph::engine &eng = get_test_engine();

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

        auto gamma_v = make_dnn_mem(lt_sh, dt::f32, tag::x);
        auto beta_v = make_dnn_mem(lt_sh, dt::f32, tag::x);
        auto d_gamma_v = make_dnn_mem(lt_sh, dt::f32, tag::x);
        auto d_beta_v = make_dnn_mem(lt_sh, dt::f32, tag::x);
        if (prb->use_ss()) {
            for (int64_t i = 0; i < prb->c; i++) {
                gamma_v.set_elem(i, ss_dt.get_elem(i));
                beta_v.set_elem(i, ss_dt.get_elem(prb->c + i));
            }
            gamma_tensor = dnnl::graph::tensor(
                    ins[4], eng, static_cast<void *>(gamma_v));
            beta_tensor = dnnl::graph::tensor(
                    ins[5], eng, static_cast<void *>(beta_v));
            d_gamma_tensor = dnnl::graph::tensor(
                    outs[1], eng, static_cast<void *>(d_gamma_v));
            d_beta_tensor = dnnl::graph::tensor(
                    outs[2], eng, static_cast<void *>(d_beta_v));
        }

        tensors_in.push_back(src_tensor);
        tensors_in.push_back(d_dst_tensor);
        tensors_in.push_back(mean_tensor);
        tensors_in.push_back(var_tensor);

        tensors_out.push_back(d_src_tensor);

        if (prb->use_ss()) {
            tensors_in.push_back(gamma_tensor);
            tensors_in.push_back(beta_tensor);

            tensors_out.push_back(d_gamma_tensor);
            tensors_out.push_back(d_beta_tensor);
        }

        SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

        if (prb->use_ss()) {
            for (int64_t i = 0; i < prb->c; i++) {
                d_ss_dt.set_elem(i, d_gamma_v.get_elem(i));
                d_ss_dt.set_elem(prb->c + i, d_beta_v.get_elem(i));
            }
        }

        if (is_bench_mode(CORR)) {
            args_t args, ref_args;

            args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_MEAN, mean_fp);
            ref_args.set(DNNL_ARG_VARIANCE, var_fp);
            ref_args.set(DNNL_ARG_SCALE_SHIFT, ss_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, d_dst_fp);
            ref_args.set(DNNL_ARG_DIFF_SRC, d_src_fp);
            ref_args.set(DNNL_ARG_DIFF_SCALE_SHIFT, d_ss_fp);

            std::vector<data_kind_t> kinds {SRC};
            if (prb->use_ss() && (prb->dir & FLAG_WEI)) {
                args.set(DNNL_ARG_DIFF_SCALE_SHIFT, d_ss_dt);
                kinds.push_back(SS);
            }

            check_correctness(
                    prb, kinds, args, ref_args, ::lnorm::setup_cmp, res);
        }
    }

    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);

    return OK;
}

} // namespace lnorm
} // namespace benchdnnext
