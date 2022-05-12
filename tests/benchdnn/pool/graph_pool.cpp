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

void check_known_skipped_case_graph(
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

fill_status_t pool_graph_prb_t::handle_main_op_(const ::pool::prb_t *prb) {
    using op = dnnl::graph::op;

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);
    std::vector<dnnl::graph::logical_tensor> inputs {};
    std::vector<dnnl::graph::logical_tensor> outputs {};

    auto src_dt = benchdnnext::set_main_op_dtype(convert_dt(prb->cfg[SRC].dt));
    auto dst_dt = benchdnnext::set_main_op_dtype(convert_dt(prb->cfg[DST].dt));

    dnnl::graph::op::kind op_kind;
    if (prb->dir & FLAG_FWD) {
        op_kind = (prb->alg == ::pool::max) ? dnnl::graph::op::kind::MaxPool
                                            : dnnl::graph::op::kind::AvgPool;
    } else {
        op_kind = (prb->alg == ::pool::max)
                ? dnnl::graph::op::kind::MaxPoolBackprop
                : dnnl::graph::op::kind::AvgPoolBackprop;
    }

    const std::string SRC {TENSOR_ID + "_SRC"};
    tensor_descs_.emplace(SRC, src_dt, prb->src_dims(), prb->tag);
    if (prb->dir & FLAG_FWD) {
        const std::string DST {TENSOR_ID + "_DST"};
        tensor_descs_.emplace(DST, dst_dt, prb->dst_dims(), prb->tag);
        inputs = {tensor_descs_[SRC]};
        outputs = {tensor_descs_[DST]};
    } else {
        const std::string DIFF_DST {TENSOR_ID + "_DIFF_DST"};
        const std::string DIFF_SRC {TENSOR_ID + "_DIFF_SRC"};
        tensor_descs_.emplace(DIFF_DST, dst_dt, prb->dst_dims(), prb->tag);
        tensor_descs_.emplace(DIFF_SRC, src_dt, prb->src_dims(), prb->tag);
        outputs = {tensor_descs_[DIFF_SRC]};
        if (op_kind == op::kind::AvgPoolBackprop) {
            inputs = {tensor_descs_[DIFF_DST]};
        } else {
            if (is_bench_mode(PERF)) {
                const std::string DST_FWD_I {TENSOR_ID + "_DST_FWD_I"};
                tensor_descs_.emplace(
                        DST_FWD_I, dt::s32, prb->dst_dims(), prb->tag);
                inputs = {tensor_descs_[SRC], tensor_descs_[DIFF_DST],
                        tensor_descs_[DST_FWD_I]};
            } else { //workaround for correctness test
                inputs = {tensor_descs_[SRC], tensor_descs_[DIFF_DST]};
            }
        }
    }

    op pool(new_op_id, op_kind, inputs, outputs, "pool");

    dims_t dilations = prb->dilations();
    // oneDNN graph dilation = 1 is equivalent of oneDNN
    // dilation = 0
    std::transform(dilations.begin(), dilations.end(), dilations.begin(),
            [](const dim_t d) { return d + 1; });

    pool.set_attr("strides", prb->strides())
            .set_attr("pads_begin", prb->padding())
            .set_attr("pads_end", prb->padding_r())
            .set_attr("kernel", prb->kernel())
            .set_attr("data_format", std::string("NCX"))
            .set_attr("auto_pad", std::string("None"));

    if (op_kind == op::kind::MaxPool || op_kind == op::kind::MaxPoolBackprop) {
        pool.set_attr("dilations", dilations);
    } else { // AvgPool
        pool.set_attr("exclude_pad", prb->alg == ::pool::avg_np);
    }
    if (prb->dir & FLAG_FWD)
        pool.set_attr("rounding_type", std::string("floor"));
    if (op_kind == op::kind::AvgPoolBackprop)
        pool.set_attr("input_shape", prb->src_dims());

    ops_.emplace_back(pool);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

fill_status_t pool_graph_prb_t::handle_bin_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.pool.bin_handler(*this, po_entry);
}

fill_status_t pool_graph_prb_t::handle_low_precision_(
        const ::pool::prb_t *prb) {
    const std::string OP_REPR = "main";
    const auto src_lt_id = tensor_id[OP_REPR].back() + "_SRC";
    const auto dst_lt_id = curr_out_map_ids_.back() + "_DST";

    fill_status_t status
            = po_handler.pool.low_precision_handler.insert_dequant_before(
                    src_lt_id, get_qdata_for(SRC, prb), *this);
    BENCHDNNEXT_VERIFY(status);

    status = po_handler.pool.low_precision_handler.insert_quant_after(
            dst_lt_id, get_qdata_for(DST, prb), *this);
    BENCHDNNEXT_VERIFY(status);

    for (const auto &entry : prb->attr.post_ops.entry) {
        if (entry.is_binary_kind()) {
            const auto bin_src1_lt_id = tensor_id["binary"].back() + "_SRC";
            status = po_handler.pool.low_precision_handler
                             .insert_dequant_before(bin_src1_lt_id,
                                     bin_po_entry2quant_data(entry, prb->tag,
                                             convert_dt(prb->cfg[DST].dt)),
                                     *this);
            BENCHDNNEXT_VERIFY(status);
            break;
        }
    }

    return status;
}

void graph_bwd_check_correctness(const ::pool::prb_t *prb, const args_t &args,
        const args_t &ref_args, res_t *res) {
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
    if (graph_prb.has_post_bin()) {
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
        if (graph_prb.has_post_bin()) {
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

            return OK;
        }
    }

    SAFE(measure_perf(
                 res->timer_map.perf_timer(), cp, tensors_in, tensors_out, res),
            WARN);

    return OK;
}

} // namespace pool
} // namespace benchdnnext
