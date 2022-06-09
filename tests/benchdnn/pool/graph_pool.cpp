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

#include "dnn_graph_types.hpp"
#include "dnnl_graph_common.hpp"
#include "utils/compare.hpp"

#include "binary/binary.hpp"
#include "pool/graph_pool.hpp"

#include <algorithm>

namespace benchdnnext {
namespace pool {

static void check_known_skipped_case_graph(
        const ::pool::prb_t *prb, res_t *res) noexcept {
    // TODO: to align with original benchdnn, we should consider moving
    // skip_unimplemented_prb call after compilation step
    skip_invalid_and_unimplemented_prb(prb, res);
    if (res->state == SKIPPED) return;

    for (const auto &po : prb->attr.post_ops.entry) {
        if (po.is_binary_kind()) {
            // currently, in the backend there are supported
            // only two policies for binary post op:
            // COMMON and PER_OC
            const auto policy = po.binary.policy;
            if (!(policy == attr_t::COMMON || policy == attr_t::PER_OC)) {
                res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
                return;
            }

            // currently, for int8 cases, in the backend we
            // support only int8 data type for 2nd binary input
            const dt src_dt = convert_dt(prb->cfg[SRC].dt);
            const dt dst_dt = convert_dt(prb->cfg[DST].dt);
            const dt bin_src_dt = convert_dt(po.binary.src1_dt);
            if (is_low_precision({src_dt, dst_dt})
                    && !is_low_precision({bin_src_dt})) {
                res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
                return;
            }
        }
    }
}

static quant_data_t get_qdata_for(int arg, const ::pool::prb_t *prb) {
    const auto q_dt = convert_dt(prb->cfg[arg].dt);
    if (arg == SRC || arg == DST) return quant_data_t(q_dt, prb->tag);

    BENCHDNN_PRINT(
            0, "warning: returning default quant_data_t for arg: %d\n", arg);
    return quant_data_t();
}

static quant_data_t get_qdata_for(
        const attr_t::post_ops_t::entry_t &entry, const ::pool::prb_t *prb) {
    if (entry.is_binary_kind())
        return bin_po_entry2quant_data(
                entry, prb->tag, convert_dt(prb->cfg[DST].dt));

    printf("warning: returning default quant_data_t for unsupported post op\n");
    return quant_data_t();
}

static std::vector<dnnl::graph::logical_tensor::data_type> collect_data_types(
        const ::pool::prb_t *prb) {
    return {convert_dt(prb->cfg[SRC].dt), convert_dt(prb->cfg[DST].dt)};
}

fill_status_t append_graph_with_block(const ::pool::prb_t *prb) {
    graph_t &graph = graph_t::get();

    const auto orig_dts = collect_data_types(prb);
    const auto with_dq = is_low_precision(orig_dts);
    const auto connect_to_previous_block = !with_dq && graph.has_blocks();

    // handle main op
    const auto op_id = graph.generate_id_for(entry_kind::POOL);
    const auto src_lt_kind
            = prb->dir == FLAG_FWD ? lt_kind::SRC : lt_kind::DIFF_SRC;
    const auto dst_lt_kind
            = prb->dir & FLAG_FWD ? lt_kind::DST : lt_kind::DIFF_DST;
    const auto src_id = connect_to_previous_block
            ? graph.get_last_block_out_id()
            : graph.generate_id_for(op_id, src_lt_kind);
    const auto dst_id = graph.generate_id_for(op_id, dst_lt_kind);

    dims_t dilations = prb->dilations();
    // oneDNN graph dilation = 1 is equivalent of oneDNN
    // dilation = 0
    std::transform(dilations.begin(), dilations.end(), dilations.begin(),
            [](const dim_t d) { return d + 1; });

    const auto src_dt = dequantize_dtype(orig_dts[0]);
    const auto dst_dt = dequantize_dtype(orig_dts[1]);

    graph.create_lt(src_id, src_dt, prb->src_dims(), prb->tag);
    graph.create_lt(dst_id, dst_dt, prb->dst_dims(), prb->tag);

    std::vector<size_t> src_ids {};
    std::vector<size_t> dst_ids {};
    dnnl::graph::op::kind pool_kind;
    if (prb->dir & FLAG_FWD) {
        src_ids = {src_id};
        dst_ids = {dst_id};
        pool_kind = prb->alg == ::pool::max ? dnnl::graph::op::kind::MaxPool
                                            : dnnl::graph::op::kind::AvgPool;
    } else {
        src_ids = {dst_id};
        dst_ids = {src_id};
        pool_kind = prb->alg == ::pool::max
                ? dnnl::graph::op::kind::MaxPoolBackprop
                : dnnl::graph::op::kind::AvgPoolBackprop;
        if (prb->alg == ::pool::max) {
            const auto fwd_src_id = graph.generate_id_for(op_id, lt_kind::SRC);
            const auto fwd_dst_indices_id
                    = graph.generate_id_for(op_id, lt_kind::DST);
            graph.create_lt(fwd_src_id, src_dt, prb->src_dims(), prb->tag);
            if (is_bench_mode(PERF)) {
                graph.create_lt(fwd_dst_indices_id,
                        dnnl::graph::logical_tensor::data_type::s32,
                        prb->dst_dims(), prb->tag);
                src_ids = {fwd_src_id, dst_id, fwd_dst_indices_id};
            } else {
                src_ids = {fwd_src_id, dst_id};
            }
        }
    }

    dnnl::graph::op pool_op(op_id, pool_kind, graph.stringify_id(op_id));
    pool_op.set_attr("strides", prb->strides())
            .set_attr("pads_begin", prb->padding())
            .set_attr("pads_end", prb->padding_r())
            .set_attr("kernel", prb->kernel())
            .set_attr("data_format", std::string("NCX"))
            .set_attr("auto_pad", std::string("None"));

    if (prb->alg == ::pool::max)
        pool_op.set_attr("dilations", dilations);
    else // AvgPool
        pool_op.set_attr("exclude_pad", prb->alg == ::pool::avg_np);
    if (prb->dir & FLAG_FWD)
        pool_op.set_attr("rounding_type", std::string("floor"));
    if (prb->alg != ::pool::max && prb->dir & FLAG_BWD)
        pool_op.set_attr("input_shape", prb->src_dims());

    graph.append(op_id, pool_op, src_ids, dst_ids);

    fill_status_t status;
    // if required - apply dequantize to block inputs
    if (with_dq) {
        status = insert_dequant_before(src_id, get_qdata_for(SRC, prb));
        BENCHDNNEXT_VERIFY(status);
    }

    // handle post ops
    for (const auto &entry : prb->attr.post_ops.entry) {
        const auto with_src1_dq = is_dequantize_required_for(entry);
        size_t po_src1_id;
        if (entry.is_binary_kind()) {
            std::tie(status, po_src1_id) = append_graph_with_binary(entry);
            BENCHDNNEXT_VERIFY(status);
            if (with_src1_dq) {
                status = insert_dequant_before(
                        po_src1_id, get_qdata_for(entry, prb));
                BENCHDNNEXT_VERIFY(status);
            }
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

static void graph_bwd_check_correctness(const ::pool::prb_t *prb,
        const args_t &args, const args_t &ref_args, res_t *res) {
    compare::compare_t cmp;
    cmp.set_data_kind(SRC);
    ::pool::setup_cmp(cmp, prb, SRC, ref_args);

    const int arg = DNNL_ARG_DIFF_SRC;
    const auto &mem_dt = args.find(arg);
    const auto &mem_fp = ref_args.find(arg);

    cmp.compare(mem_fp, mem_dt, prb->attr, res);
}

int doit(const ::pool::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

    const auto status = append_graph_with_block(prb);
    if (status != fill_status::DONE
            && status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        cleanup();
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto &graph = graph_t::get();

    // Filter partitions
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

    bool is_fwd = prb->dir & FLAG_FWD;
    bool is_max_pool = prb->alg == ::pool::max;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::pool::init_pd, prb, res, par, ins, outs);

    auto src_fp = make_dnn_mem(is_fwd ? ins[0] : outs[0], dt::f32, tag::abx);
    auto dst_fp = make_dnn_mem(is_fwd ? outs[0] : is_max_pool ? ins[1] : ins[0],
            dt::f32, tag::abx);
    dnn_mem_t ws_fp
            = make_dnn_mem(is_fwd ? outs[0] : is_max_pool ? ins[1] : ins[0],
                    dt::s32, tag::abx);

    auto src_dt = make_dnn_mem(is_fwd ? ins[0] : outs[0], prb->tag);
    auto dst_dt = make_dnn_mem(
            is_fwd ? outs[0] : is_max_pool ? ins[1] : ins[0], prb->tag);

    SAFE(fill_src(prb, src_dt, src_fp, res), WARN);

    std::vector<dnnl::graph::tensor> tensors_in, tensors_out;

    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    if (!prb->attr.post_ops.entry.empty()) {
        binary_po_fp.emplace_back(make_dnn_mem(ins.back(), dt::f32, tag::abx));
        binary_po_dt.emplace_back(make_dnn_mem(ins.back(), prb->tag));
        const int po_idx = DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1;
        ::binary::fill_mem(po_idx, binary_po_dt.back(), binary_po_fp.back());
        binary_po_args.push_back(po_idx);
    }
    const dnnl::graph::engine &eng = get_test_engine();

    dnn_mem_t d_dst_dt, d_src_dt, ws_dt;
    args_t args, ref_args;

    if (is_fwd) {
        tensors_in.emplace_back(ins[0], eng, static_cast<void *>(src_dt));
        tensors_out.emplace_back(outs[0], eng, static_cast<void *>(dst_dt));
        if (!prb->attr.post_ops.entry.empty()) {
            tensors_in.emplace_back(dnnl::graph::tensor(
                    ins.back(), eng, static_cast<void *>(binary_po_dt.back())));
        }
        if (is_bench_mode(CORR)) {
            args.set(DNNL_ARG_DST, dst_dt);
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(DNNL_ARG_WORKSPACE, ws_fp);
            ref_args.set(binary_po_args, binary_po_fp);

            SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);
            check_correctness(
                    prb, {DST}, args, ref_args, ::pool::setup_cmp, res);

            cleanup();
            return OK;
        }
    } else {
        auto d_dst_fp = make_dnn_mem(
                is_max_pool ? ins[1] : ins[0], dt::f32, tag::abx);
        auto d_src_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);
        d_dst_dt = make_dnn_mem(is_max_pool ? ins[1] : ins[0], prb->tag);
        d_src_dt = make_dnn_mem(outs[0], prb->tag);
        SAFE(fill_dst(prb, d_dst_dt, d_dst_fp, res), WARN);
        tensors_out.emplace_back(outs[0], eng, static_cast<void *>(d_src_dt));

        args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
        ref_args.set(DNNL_ARG_SRC, src_fp);
        ref_args.set(DNNL_ARG_DST, dst_fp);
        ref_args.set(DNNL_ARG_WORKSPACE, ws_fp);
        ref_args.set(binary_po_args, binary_po_fp);
        ref_args.set(DNNL_ARG_DIFF_DST, d_dst_fp);
        ref_args.set(DNNL_ARG_DIFF_SRC, d_src_fp);

        TIME_REF(::pool::compute_ref(prb, ref_args));

        if (is_max_pool) {
            tensors_in.emplace_back(ins[0], eng, static_cast<void *>(src_dt));
            tensors_in.emplace_back(ins[1], eng, static_cast<void *>(d_dst_dt));
            if (is_bench_mode(PERF)) {
                ws_dt = make_dnn_mem(ins[2], prb->tag);
                SAFE(ws_dt.reorder(ws_fp), WARN);
                tensors_in.emplace_back(
                        ins[2], eng, static_cast<void *>(ws_dt));
            }
        } else {
            tensors_in.emplace_back(ins[0], eng, static_cast<void *>(d_dst_dt));
        }

        if (is_bench_mode(CORR)) {
            SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);
            graph_bwd_check_correctness(prb, args, ref_args, res);

            cleanup();
            return OK;
        }
    }

    SAFE(measure_perf(
                 res->timer_map.perf_timer(), cp, tensors_in, tensors_out, res),
            WARN);

    cleanup();

    return OK;
}

} // namespace pool
} // namespace benchdnnext
