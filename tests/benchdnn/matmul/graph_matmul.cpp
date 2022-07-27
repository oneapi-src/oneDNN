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

#include "dnnl_graph_common.hpp"
#include "utils/compare.hpp"

#include "binary/binary.hpp"
#include "matmul/graph_matmul.hpp"

#include <algorithm>
#include <tuple>

namespace benchdnnext {
namespace matmul {

static int check_known_skipped_case_graph(
        const ::matmul::prb_t *prb, res_t *res) noexcept {

    // Not support u8s8u8 with 3 or higher matric dimensions.
    bool skipped_dt = prb->src_dt() == dnnl_u8 && prb->wei_dt() == dnnl_s8
            && prb->dst_dt() == dnnl_u8;
    if (is_gpu() && skipped_dt && prb->src_dims().size() > 2) {
        res->state = SKIPPED;
        res->reason = CASE_NOT_SUPPORTED;
        return OK;
    }

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, ::matmul::init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    auto const_pd = query_pd(prim);
    if (check_mem_size(const_pd) != OK) {
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    check_graph_eltwise_post_ops(prb->attr, res);
    if (res->state == SKIPPED) return OK;

    // Not support post-sum with zero_points. TODO(xiang): remove after onednn fix this.
    if (is_gpu()) {
        const auto &po = prb->attr.post_ops;
        for (int idx = 0; idx < po.len(); ++idx) {
            const auto &e = po.entry[idx];
            if (e.is_sum_kind() && e.sum.zero_point != 0) {
                res->state = SKIPPED;
                res->reason = CASE_NOT_SUPPORTED;
                return OK;
            }
        }
    }

    check_graph_scales_and_zps_support(prb->attr, res);
    return OK;
}

static quant_data_t get_qdata_for(int arg, const ::matmul::prb_t *prb) {
    if (arg == SRC) {
        const auto q_dt = convert_dt(prb->src_dt());
        const int64_t zp_val = prb->attr.zero_points.is_def(DNNL_ARG_SRC)
                ? 0L
                : prb->src_zp[0];
        return quant_data_t(q_dt, {1.0f}, {zp_val}, prb->stag);
    } else if (arg == WEI) {
        const auto q_dt = convert_dt(prb->wei_dt());
        const auto scales = get_scales(prb->attr.oscale, prb->scales, prb->n);
        const std::vector<int64_t> zps(scales.size(), 0L);
        const std::string q_type = prb->attr.oscale.policy == policy_t::COMMON
                ? "per_tensor"
                : "per_channel";
        return quant_data_t(q_dt, scales, zps, q_type, 0, prb->wtag);
    } else if (arg == DST) {
        const auto q_dt = convert_dt(prb->dst_dt());
        const float scale_val = 1.f
                * (1.f / get_post_eltwise_scale(prb->attr.post_ops.entry));
        const int64_t zp_val = prb->attr.zero_points.is_def(DNNL_ARG_DST)
                ? 0L
                : prb->dst_zp[0];
        return quant_data_t(q_dt, {scale_val}, {zp_val}, prb->dtag);
    }

    BENCHDNN_PRINT(
            0, "warning: returning default quant_data_t for arg: %d\n", arg);
    return quant_data_t();
}

static quant_data_t get_qdata_for(
        const attr_t::post_ops_t::entry_t &entry, const ::matmul::prb_t *prb) {
    if (entry.is_binary_kind())
        return bin_po_entry2quant_data(
                entry, prb->dtag, convert_dt(prb->dst_dt()));
    else if (entry.is_sum_kind())
        return sum_po_entry2quant_data(
                entry, prb->dtag, convert_dt(prb->dst_dt()));

    BENCHDNN_PRINT(0,
            "warning: returning default quant_data_t for %s post op\n",
            entry.is_binary_kind() ? "binary" : "sum");
    return quant_data_t();
}

static std::vector<dnnl::graph::logical_tensor::data_type> collect_data_types(
        const ::matmul::prb_t *prb) {
    return {convert_dt(prb->src_dt()), convert_dt(prb->wei_dt()),
            convert_dt(prb->dst_dt())};
}

static dims_t get_runtime_dims(
        const dims_t &dims, const ::matmul::dims_mask_t &mask) noexcept {
    if (mask.none() || dims.empty()) return dims;
    dims_t runtime_dims;
    runtime_dims.resize(dims.size());
    const int64_t axis_unknown_flag = -1;
    for (size_t i = 0; i < dims.size(); ++i) {
        runtime_dims[i] = mask[i] ? axis_unknown_flag : dims[i];
    }
    return runtime_dims;
}

fill_status_t append_graph_with_block(const ::matmul::prb_t *prb) {
    graph_t &graph = graph_t::get();

    const auto orig_dts = collect_data_types(prb);
    const auto with_dq = is_low_precision(orig_dts);
    const auto with_tc = with_typecast(orig_dts);
    const auto connect_to_previous_block = !with_dq && graph.has_blocks();

    // handle main op
    const auto op_id = graph.generate_id_for(entry_kind::MATMUL);
    auto src_id = connect_to_previous_block
            ? graph.get_last_block_out_id()
            : graph.generate_id_for(op_id, lt_kind::SRC);
    auto wei_id = graph.generate_id_for(op_id, lt_kind::WEI);
    const auto bia_id = graph.generate_id_for(op_id, lt_kind::BIA);
    const auto dst_id = graph.generate_id_for(op_id, lt_kind::DST);

    const auto change_dt = with_dq || with_tc;
    const auto default_dt = with_tc ? dt::bf16 : dt::f32;
    const auto src_dt = change_dt ? default_dt : orig_dts[0];
    const auto wei_dt = change_dt ? default_dt : orig_dts[1];
    const auto dst_dt = change_dt ? default_dt : orig_dts[2];
    const auto bia_dt = convert_dt(prb->bia_dt);

    const auto src_dims
            = get_runtime_dims(prb->src_dims(), prb->src_runtime_dim_mask());
    const auto wei_dims = get_runtime_dims(
            prb->weights_dims(), prb->weights_runtime_dim_mask());
    const auto dst_dims
            = get_runtime_dims(prb->dst_dims, prb->dst_runtime_dim_mask());

    graph.create_lt(src_id, src_dt, src_dims, prb->stag);
    graph.create_lt(wei_id, wei_dt, wei_dims, prb->wtag,
            dnnl::graph::logical_tensor::property_type::constant);
    graph.create_lt(dst_id, dst_dt, dst_dims, prb->dtag);
    const bool with_bia
            = bia_dt != dnnl::graph::logical_tensor::data_type::undef;
    if (with_bia) {
        dims_t bia_dims(dst_dims.size());
        for (int i = 0; i < prb->ndims; ++i)
            bia_dims[i] = (prb->bia_mask & (1 << i)) ? dst_dims[i] : 1;
        bia_dims = get_runtime_dims(bia_dims, prb->dst_runtime_dim_mask());
        graph.create_lt(bia_id, bia_dt, bia_dims,
                dnnl::graph::logical_tensor::layout_type::strided,
                dnnl::graph::logical_tensor::property_type::constant);
    }

    std::vector<size_t> src_ids {src_id, wei_id};
    std::vector<size_t> dst_ids {dst_id};
    if (with_bia) src_ids.push_back(bia_id);

    dnnl::graph::op mm_op(
            op_id, dnnl::graph::op::kind::MatMul, graph.stringify_id(op_id));
    mm_op.set_attr("transpose_a", false).set_attr("transpose_b", false);

    graph.append(op_id, mm_op, src_ids, dst_ids);

    fill_status_t status;
    // if required - apply typecast to block inputs
    if (with_tc) {
        std::tie(status, src_id) = insert_typecast_before(src_id);
        BENCHDNNEXT_VERIFY(status);
        std::tie(status, wei_id) = insert_typecast_before(wei_id, true);
        BENCHDNNEXT_VERIFY(status);
    }

    // if required - apply dequantize to block inputs
    if (with_dq) {
        status = insert_dequant_before(src_id, get_qdata_for(SRC, prb));
        BENCHDNNEXT_VERIFY(status);
        status = insert_dequant_before(wei_id, get_qdata_for(WEI, prb), true);
        BENCHDNNEXT_VERIFY(status);
    }

    // handle post ops
    for (const auto &entry : prb->attr.post_ops.entry) {
        const auto with_src1_dq
                = entry.is_sum_kind() || is_dequantize_required_for(entry);
        size_t po_src1_id;
        if (entry.is_binary_kind()) {
            std::tie(status, po_src1_id) = append_graph_with_binary(entry);
            BENCHDNNEXT_VERIFY(status);
            if (with_src1_dq) {
                status = insert_dequant_before(
                        po_src1_id, get_qdata_for(entry, prb));
                BENCHDNNEXT_VERIFY(status);
            }
        } else if (entry.is_eltwise_kind()) {
            const bool is_swish
                    = entry.kind == attr_t::post_ops_t::kind_t::SWISH
                    && with_bia;
            status = is_swish ? append_graph_with_swish(entry, dst_id)
                              : append_graph_with_eltwise(entry);
            BENCHDNNEXT_VERIFY(status);
        } else if (entry.is_sum_kind()) {
            std::tie(status, po_src1_id) = append_graph_with_sum(entry);
            BENCHDNNEXT_VERIFY(status);
            if (with_dq && with_src1_dq) {
                status = insert_dequant_before(
                        po_src1_id, get_qdata_for(entry, prb));
                BENCHDNNEXT_VERIFY(status);
            }
        }
    }

    // if required - add quantize op
    if (is_low_precision({orig_dts[2]})) {
        status = insert_quant_after(
                graph.get_cur_block_out_id(), get_qdata_for(DST, prb));
        BENCHDNNEXT_VERIFY(status);
    }

    graph.close_block();

    return fill_status::DONE;
}

int doit(const ::matmul::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const auto status = append_graph_with_block(prb);
    if (status != fill_status::DONE
            && status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        cleanup();
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto &graph = graph_t::get();

    auto mode = convert_fpmath_mode(prb->attr.fpmath_mode);
    // Filter partitions
    const auto partitions = graph.get_partitions(mode);
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

    auto cp = compile_partition(::matmul::init_pd, prb, res, par, ins, outs);

    const auto apply_bias = convert_dt(prb->bia_dt) != dt::undef;

    size_t idx_ins = 0;
    auto src_fp = make_dnn_mem(ins[idx_ins], dt::f32, tag::abx);
    auto src_dt = make_dnn_mem(ins[idx_ins], prb->stag);
    auto wei_fp = make_dnn_mem(ins[++idx_ins], dt::f32, tag::abx);
    auto wei_dt = make_dnn_mem(ins[idx_ins], prb->wtag);
    auto dst_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);
    auto dst_dt = make_dnn_mem(outs[0], prb->dtag);
    dnn_mem_t bia_fp, bia_dt;
    if (apply_bias) {
        bia_fp = make_dnn_mem(ins[++idx_ins], dt::f32, tag::abx);
        bia_dt = make_dnn_mem(ins[idx_ins], tag::abx);
    }

    SAFE(fill_data(SRC, prb, src_dt, src_fp, res), WARN);
    SAFE(fill_data(WEI, prb, wei_dt, wei_fp, res), WARN);
    SAFE(fill_data(DST, prb, dst_dt, dst_fp, res), WARN);
    if (apply_bias) SAFE(fill_data(BIA, prb, bia_dt, bia_fp, res), WARN);

    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    const std::vector<attr_t::post_ops_t::entry_t> &po_entry
            = prb->attr.post_ops.entry;
    const std::vector<size_t> post_bin_indices = get_post_bin_indices(po_entry);
    for (size_t i = 0; i < post_bin_indices.size(); i++) {
        binary_po_fp.emplace_back(
                make_dnn_mem(ins[++idx_ins], dt::f32, tag::abx));
        binary_po_dt.emplace_back(make_dnn_mem(ins[idx_ins], tag::abx));
        binary::fill_mem(DNNL_ARG_ATTR_MULTIPLE_POST_OP(
                                 static_cast<int>(post_bin_indices[i])),
                binary_po_dt[i], binary_po_fp[i]);
    }

    const dnnl::graph::engine &eng = get_test_engine();

    idx_ins = 0;
    dnnl::graph::tensor src_tensor(
            ins[idx_ins], eng, static_cast<void *>(src_dt));
    dnnl::graph::tensor wei_tensor(
            ins[++idx_ins], eng, static_cast<void *>(wei_dt));
    dnnl::graph::tensor dst_tensor(outs[0], eng, static_cast<void *>(dst_dt));
    dnnl::graph::tensor bia_tensor;
    dnnl::graph::tensor bin_tensor;
    dnnl::graph::tensor sum_src1_tensor;

    std::vector<dnnl::graph::tensor> tensors_in {src_tensor, wei_tensor};
    std::vector<dnnl::graph::tensor> tensors_out {dst_tensor};

    if (apply_bias) {
        bia_tensor = dnnl::graph::tensor(
                ins[++idx_ins], eng, static_cast<void *>(bia_dt));
        tensors_in.emplace_back(bia_tensor);
    }

    size_t bin_dt_idx = 0;
    for (size_t i = 0; i < po_entry.size(); i++) {
        // we can't have fuse with both sum and binary-add at the same time
        if (po_entry[i].is_sum_kind()) { // Always use in-place operation.
            sum_src1_tensor = dnnl::graph::tensor(
                    ins[++idx_ins], eng, static_cast<void *>(dst_dt));
            tensors_in.emplace_back(sum_src1_tensor);
        } else if (po_entry[i].is_binary_kind()) {
            bin_tensor = dnnl::graph::tensor(ins[++idx_ins], eng,
                    static_cast<void *>(binary_po_dt[bin_dt_idx]));
            tensors_in.emplace_back(bin_tensor);
            ++bin_dt_idx;
        }
    }

    SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

    dnn_mem_t bia_fp_scaled;
    args_t args, ref_args;

    if (is_bench_mode(CORR)) {
        args.set(DNNL_ARG_DST, dst_dt);
        ref_args.set(DNNL_ARG_SRC, src_fp);
        ref_args.set(DNNL_ARG_WEIGHTS, wei_fp);
        ref_args.set(DNNL_ARG_DST, dst_fp);

        std::vector<int> binary_po_args;
        for (size_t idx_bin : post_bin_indices) {
            binary_po_args.emplace_back(
                    (DNNL_ARG_ATTR_MULTIPLE_POST_OP(static_cast<int>(idx_bin))
                            | DNNL_ARG_SRC_1));
        }
        ref_args.set(binary_po_args, binary_po_fp);

        if (apply_bias
                && is_low_precision(
                        {convert_dt(prb->src_dt()), convert_dt(prb->wei_dt()),
                                convert_dt(prb->dst_dt())})) {
            bia_fp_scaled = make_dnn_mem(ins[2], dt::f32, tag::abx);
            scale_bia(bia_fp_scaled, bia_fp,
                    get_scales(prb->attr.oscale, prb->scales, prb->n));
            ref_args.set(DNNL_ARG_BIAS, bia_fp_scaled);
        } else {
            ref_args.set(DNNL_ARG_BIAS, bia_fp);
        }

        check_correctness(prb, {DST}, args, ref_args, ::matmul::setup_cmp, res);
    }

    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out,
                 ins, outs),
            WARN);

    cleanup();

    return OK;
}

} // namespace matmul
} // namespace benchdnnext
