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

#include <vector>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "dnnl_graph_common.hpp"
#include "utils/compare.hpp"

#include "concat/concat.hpp"
#include "concat/graph_concat.hpp"

namespace benchdnnext {
namespace concat {

static void check_ranks_shapes_and_dtypes(
        const ::concat::prb_t *prb, res_t *res) {
    auto rank = prb->vdims[0].size();
    auto axis = prb->axis;

    // In oneDNN graph axis can take negative values. But in oneDNN
    // it has to have values greater or equal to 0. Therefore for consistency,
    // we create a condition, that the axis has a range of values
    // the same as the native benchdnn code.
    // Axis should also be smaller than the rank.
    const auto axis_in_range = (axis >= 0 || axis < static_cast<int64_t>(rank));
    if (!axis_in_range) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }

    // Rank of all tensors should match
    const auto same_rank = std::all_of(prb->vdims.cbegin(), prb->vdims.cend(),
            [rank](const dims_t &sdim) { return rank == sdim.size(); });
    if (!same_rank) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }

    // Shapes for all inputs should match at every position except axis position
    for (auto i = 0; i < rank; ++i) {
        if (axis == static_cast<int64_t>(i)) continue;
        const auto d = prb->dst_dims[i];
        const auto same_dim
                = std::all_of(prb->vdims.cbegin(), prb->vdims.cend(),
                        [d, i](const dims_t &sdim) { return d == sdim[i]; });
        if (!same_dim) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }

    const auto dst_axis_dim = prb->dst_dims[axis];
    const auto src_axis_dim_sum = std::accumulate(prb->vdims.cbegin(),
            prb->vdims.cend(), 0L, [axis](int64_t acc, const dims_t &sdim) {
                return acc + sdim[axis];
            });
    if (dst_axis_dim != src_axis_dim_sum) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }

    // Types of all tensors should match
    const auto same_dtypes = prb->sdt == prb->ddt;
    if (!same_dtypes) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

static void check_known_skipped_case_graph(
        const ::concat::prb_t *prb, res_t *res) {
    // srcs and dst should have same tags
    const auto tag = prb->dtag;
    const auto same_tags = std::all_of(prb->stag.cbegin(), prb->stag.cend(),
            [tag](const std::string &stag) { return !tag.compare(stag); });
    if (!same_tags) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

static quant_data_t get_qdata_for(
        int arg, const ::concat::prb_t *prb, size_t occurrence = 0) {
    if (arg == SRC) {
        return quant_data_t(convert_dt(prb->sdt), prb->stag[occurrence]);
    } else if (arg == DST) {
        return quant_data_t(convert_dt(prb->ddt), prb->dtag);
    }

    BENCHDNN_PRINT(
            0, "warning: returning default quant_data_t for arg: %d\n", arg);
    return quant_data_t();
}

static std::vector<dnnl::graph::logical_tensor::data_type> collect_data_types(
        const ::concat::prb_t *prb) {
    return {convert_dt(prb->sdt), convert_dt(prb->ddt)};
}

fill_status_t append_graph_with_block(const ::concat::prb_t *prb) {
    graph_t &graph = graph_t::get();

    const auto orig_dts = collect_data_types(prb);
    const auto with_dq = is_low_precision(orig_dts);
    const auto connect_to_previous_block = !with_dq && graph.has_blocks();

    // handle main op
    const auto op_id = graph.generate_id_for(entry_kind::CONCAT);

    auto src_dt = dequantize_dtype(orig_dts[0]);
    std::vector<size_t> src_ids;
    for (size_t i = 0; i < prb->n_inputs(); ++i) {
        const auto src_i_id = (i == 0 && connect_to_previous_block)
                ? graph.get_last_block_out_id()
                : graph.generate_id_for(op_id, lt_kind::SRC_I, i);
        graph.create_lt(src_i_id, src_dt, prb->vdims[i], prb->stag[i]);
        src_ids.push_back(src_i_id);
    }

    const auto dst_id = graph.generate_id_for(op_id, lt_kind::DST);
    auto dst_dt = dequantize_dtype(orig_dts[1]);
    graph.create_lt(dst_id, dst_dt, prb->dst_dims, prb->dtag);
    std::vector<size_t> dst_ids {dst_id};

    dnnl::graph::op concat_op(
            op_id, dnnl::graph::op::kind::Concat, graph.stringify_id(op_id));
    concat_op.set_attr<int64_t>("axis", prb->axis);

    graph.append(op_id, concat_op, src_ids, dst_ids);

    fill_status_t status;
    // if required - apply dequantize to block inputs
    if (with_dq) {
        for (size_t i = 0; i < src_ids.size(); ++i) {
            status = insert_dequant_before(
                    src_ids[i], get_qdata_for(SRC, prb, i));
            BENCHDNNEXT_VERIFY(status);
        }
    }

    // if required - add quantize op
    if (is_low_precision({orig_dts[1]})) {
        status = insert_quant_after(
                graph.get_cur_block_out_id(), get_qdata_for(DST, prb));
        BENCHDNNEXT_VERIFY(status);
    }

    graph.close_block();

    return fill_status::DONE;
}

int doit(const ::concat::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    // TODO: to align with original benchdnn, we should consider moving
    // skip_unimplemented_prb call after compilation step
    skip_invalid_and_unimplemented_prb(prb, res);
    if (res->state == SKIPPED) return OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;
    check_ranks_shapes_and_dtypes(prb, res);
    if (res->state == SKIPPED) return OK;

    const auto status = append_graph_with_block(prb);
    if (status != fill_status::DONE
            && status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        cleanup();
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto &graph = graph_t::get();

    // // Filter partitions
    const auto partitions = graph.get_partitions();
    if (partitions.empty() || partitions.size() > 1) {
        cleanup();
        return res->state = FAILED, FAIL;
    }

    const auto par = partitions[0];
    if (!par.is_supported()) {
        cleanup();
        return res->state = UNIMPLEMENTED, FAIL;
    }

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::concat::init_pd, prb, res, par, ins, outs);

    auto dst_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);
    auto dst_dt = make_dnn_mem(outs[0], prb->dtag);

    std::vector<dnn_mem_t> src_fp, src_dt;
    src_fp.reserve(prb->n_inputs());
    src_dt.reserve(prb->n_inputs());

    std::vector<dnnl::graph::tensor> tensors_in {};
    tensors_in.reserve(prb->n_inputs());

    const dnnl::graph::engine &eng = get_test_engine();

    args_t args, ref_args;
    for (auto i = 0; i < prb->n_inputs(); ++i) {
        src_fp.emplace_back(make_dnn_mem(ins[i], dt::f32, tag::abx));
        src_dt.emplace_back(make_dnn_mem(ins[i], prb->stag[i]));

        SAFE(::concat::fill_src(i, prb->ddt, src_dt[i], src_fp[i]), WARN);

        tensors_in.emplace_back(dnnl::graph::tensor(
                ins[i], eng, static_cast<void *>(src_dt[i])));

        if (is_bench_mode(CORR)) {
            ref_args.set(DNNL_ARG_MULTIPLE_SRC + i, src_fp[i]);
        }
    }

    dnnl::graph::tensor dst_tensor(outs[0], eng, static_cast<void *>(dst_dt));
    std::vector<dnnl::graph::tensor> tensors_out = {dst_tensor};

    SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

    if (is_bench_mode(CORR)) {
        args.set(DNNL_ARG_DST, dst_dt);
        ref_args.set(DNNL_ARG_DST, dst_fp);

        check_correctness(prb, {DST}, args, ref_args, ::concat::setup_cmp, res);
    }

    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);

    cleanup();

    return OK;
}

} // namespace concat
} // namespace benchdnnext
