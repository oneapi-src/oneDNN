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

#include <tuple>
#include <utility>
#include "utils/compare.hpp"

// TODO: refactor the driver to avoid using extra flags of a memory descriptor.
#include "common/memory_desc.hpp"

#include "reorder/graph_reorder.hpp"

namespace benchdnnext {
namespace reorder {

static int check_known_skipped_case_graph(
        const ::reorder::prb_t *prb, res_t *res) noexcept {

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, ::reorder::init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    /* reorder op requires source and destination data types to be same.
       four possible cases:
       (1) input datatype == output datatype and different layout
            (a) input = int8 and output = int8
            invoke fusion operation: dequantize operator followed by reorder operater and quantize operator
            (b) other situations
            reorder does layout conversion as memory operator.
       (2) input datatype != output datatype
            (a) input = fp32 and output = bf16 invoke typecast operator
            (b) input = bf16 and output = fp32 invoke typecast operator
       (3) input datatype != output datatype
            (a) input = fp32 and output = int8 and same layout
             invoke quantize operator
            (b) input = fp32 and output = int8 and different layout
             invoke quantize operator followed by reorder
            (c) input = int8 and output = fp32 and same layout
             invoke dequantize operator
            (d) input = int8 and output = fp32 and different layout
             invoke dequantize operator followed by reorder
       (4)  Complex cases like reorder from f32 abx to bf16 axb, may need
            combine Reorder operator and TypeCast operator
    */
    /* TODO: involves multiple operators. Eg: reorder+typecast */
    if (prb->sdt != prb->ddt && prb->stag != prb->dtag) {
        if (!((prb->sdt == dnnl_s8 && prb->ddt == dnnl_u8)
                    || (prb->sdt == dnnl_u8 && prb->ddt == dnnl_s8))) {
            res->state = SKIPPED;
        }
    }
    if (prb->stag == prb->dtag) {
        if ((prb->sdt == dnnl_s8 || prb->ddt == dnnl_u8)
                && (prb->ddt == dnnl_bf16)) {
            res->state = SKIPPED;
        }
    }

    //TODO: Currently  Skip test cases for zps with per-channel quantization.
    if (prb->attr.zero_points.points.size() > 0
            && prb->attr.scales.get(DNNL_ARG_SRC).policy
                    != attr_t::policy_t::COMMON) {
        res->state = SKIPPED;
    }

    check_graph_scales_and_zps_support(prb->attr, res);
    return OK;
}

static bool is_quantize(graph_dt src_dt, graph_dt dst_dt) {
    return src_dt == graph_dt::f32 && is_low_precision({dst_dt});
}

static bool is_dequantize(graph_dt src_dt, graph_dt dst_dt) {
    return dst_dt == graph_dt::f32 && is_low_precision({src_dt});
}

static bool is_typecast(graph_dt src_dt, graph_dt dst_dt) {
    return (src_dt == graph_dt::f32
                   && (dst_dt == graph_dt::bf16 || dst_dt == graph_dt::f16))
            || ((src_dt == graph_dt::bf16 || src_dt == graph_dt::f16)
                    && dst_dt == graph_dt::f32);
}

static bool is_typecast_quantize(graph_dt src_dt, graph_dt dst_dt) {
    return (src_dt == graph_dt::bf16 && is_low_precision({dst_dt}));
}

static bool is_dequantize_reorder_quantize(graph_dt src_dt, graph_dt dst_dt) {
    return is_low_precision({src_dt}) && is_low_precision({dst_dt});
}

// Library doesn't support fusions with dynamic q/dq, so we parse runtime scales
// and zps to static attibutes so that the dynamic q/dq can be translated to q/dq.
static bool is_runtime_to_static(graph_dt src_dt, graph_dt dst_dt) {
    return is_typecast_quantize(src_dt, dst_dt)
            || is_dequantize_reorder_quantize(src_dt, dst_dt);
}

static void set_quant_op_attr(dnnl::graph::op &op, const std::string &qtype,
        const std::vector<float> &scales, const std::vector<int64_t> &zps,
        const int64_t &axis) {
    op.set_attr("qtype", qtype);
    //TODO: use zps - revisit
    op.set_attr("zps", zps);
    op.set_attr("scales", scales);
    //TODO: axis doesnt support PER_DIM_01
    if (qtype == "per_channel") op.set_attr("axis", axis);
}

static int fill_zps(const ::reorder::prb_t *prb, const int64_t axis,
        std::vector<int64_t> &src_zps, std::vector<int64_t> &dst_zps) {
    if (prb->attr.scales.get(DNNL_ARG_SRC).policy == attr_t::policy_t::COMMON) {
        if (prb->attr.zero_points.is_def()) {
            src_zps.emplace_back(0);
            dst_zps.emplace_back(0);
        } else if (prb->ddt == dnnl_s8 || prb->ddt == dnnl_u8) {
            //Quantize Op
            src_zps.emplace_back(0);
            dst_zps.emplace_back(prb->dst_zp[0]);
        } else if ((prb->sdt == dnnl_s8 || prb->sdt == dnnl_u8)
                && prb->ddt == dnnl_f32) {
            //Dequantize Op
            src_zps.emplace_back(prb->src_zp[0]);
            dst_zps.emplace_back(0);
        }
    } else {
        //TODO: needs update for PER_DIM_01
        for (int i = 0; i < prb->dims[axis]; i++) {
            //TODO: src_zps and dst_zps could be different.
            //Need to modify depending on spec - NOT SUPPORTED
            src_zps.emplace_back(0);
            dst_zps.emplace_back(0);
        }
    }

    return OK;
}

static int fill_scales(const ::reorder::prb_t *prb, const int64_t axis,
        std::vector<float> &scales) {
    if (prb->attr.scales.get(DNNL_ARG_SRC).policy == attr_t::policy_t::COMMON) {
        if (prb->attr.scales.is_def()) {
            scales.emplace_back(1.f);
        } else {
            scales.emplace_back(prb->src_scales[0]);
        }
    } else {
        //TODO: needs update for PER_DIM_01
        for (int i = 0; i < prb->dims[axis]; i++) {
            scales.emplace_back(prb->src_scales[i]);
        }
    }
    //Need to inverse scale
    if (prb->ddt == dnnl_s8 || prb->ddt == dnnl_u8) {
        for (size_t i = 0; i < scales.size(); i++) {
            scales[i] = 1.f / scales[i];
        }
    }
    return OK;
}

static void prepare_runtime_scales(const ::reorder::prb_t *prb,
        dnn_mem_t &scales_dt, const dnnl::graph::logical_tensor &in,
        std::vector<float> scales, int64_t axis) {
    // scales is required input for dynamic q/deq
    scales_dt = make_dnn_mem(in, dt::f32, tag::x);
    fill_scales(prb, axis, scales);
    for (size_t i = 0; i < scales.size(); i++) {
        scales_dt.set_elem(i, scales[i]);
    }
}

static void maybe_prepare_runtime_zero_points(const ::reorder::prb_t *prb,
        dnn_mem_t &zps_dt, const dnnl::graph::logical_tensor &in,
        std::vector<int64_t> src_zps, std::vector<int64_t> dst_zps,
        int64_t axis) {
    // zps is optional input for dynamic q/deq
    if (prb->attr.zero_points.is_def()) return;

    zps_dt = make_dnn_mem(in, dt::s32, tag::x);
    fill_zps(prb, axis, src_zps, dst_zps);

    if (is_quantize(convert_dt(prb->sdt), convert_dt(prb->ddt))) {
        for (size_t i = 0; i < dst_zps.size(); i++) {
            zps_dt.set_elem(i, static_cast<int32_t>(dst_zps[i]));
        }
    } else {
        for (size_t i = 0; i < src_zps.size(); i++) {
            zps_dt.set_elem(i, static_cast<int32_t>(src_zps[i]));
        }
    }
}

fill_status_t append_graph_with_block(const ::reorder::prb_t *prb) {
    using graph_dt = dnnl::graph::logical_tensor::data_type;

    graph_t &graph = graph_t::get();

    const auto connect_to_previous_block = graph.has_blocks();

    const auto src_dt = convert_dt(prb->sdt);
    const auto dst_dt = convert_dt(prb->ddt);
    const auto qtype
            = convert_attr_policy(prb->attr.scales.get(DNNL_ARG_SRC).policy);
    bool runtime = prb->attr.scales.get(DNNL_ARG_SRC).runtime;
    // axis is used only for PER_DIM_0 and PER_DIM_1 policies
    int64_t axis = prb->attr.scales.get(DNNL_ARG_SRC).policy
                    == attr_t::policy_t::PER_DIM_1
            ? 1
            : 0;
    std::vector<float> default_scales({1.0});
    const auto sz_dims
            = qtype == "per_channel" ? dims_t {prb->dims[axis]} : dims_t {1};

    const auto is_reorder = src_dt == dst_dt;
    const auto is_qdq
            = is_quantize(src_dt, dst_dt) || is_dequantize(src_dt, dst_dt);
    const auto is_tc = is_typecast(src_dt, dst_dt);
    const auto is_tc_q = is_typecast_quantize(src_dt, dst_dt);
    const auto is_dq_r_q = is_dequantize_reorder_quantize(src_dt, dst_dt);

    std::vector<float> scales;
    std::vector<int64_t> src_zps, dst_zps;

    if (!runtime || is_runtime_to_static(src_dt, dst_dt)) {
        fill_scales(prb, axis, scales);
        fill_zps(prb, axis, src_zps, dst_zps);
    }

    if (is_dq_r_q) {
        // SRC->SRC_F32->DST_F32->DST
        const auto op0_id = graph.generate_id_for(entry_kind::DEQUANTIZE);
        const auto op0_src_id = connect_to_previous_block
                ? graph.get_last_block_out_id()
                : graph.generate_id_for(op0_id, lt_kind::SRC);
        const auto op0_dst_id = graph.generate_id_for(op0_id, lt_kind::DST);

        graph.create_lt(op0_src_id, src_dt, prb->dims, prb->stag);
        graph.create_lt(op0_dst_id, graph_dt::f32, prb->dims, prb->stag);

        dnnl::graph::op dequantize_op(op0_id, dnnl::graph::op::kind::Dequantize,
                graph.stringify_id(op0_id));
        set_quant_op_attr(dequantize_op, qtype, default_scales, src_zps, axis);

        graph.append(op0_id, dequantize_op, {op0_src_id}, {op0_dst_id});

        const auto op1_id = graph.generate_id_for(entry_kind::REORDER);
        const auto op1_dst_id = graph.generate_id_for(op1_id, lt_kind::DST);

        graph.create_lt(op1_dst_id, graph_dt::f32, prb->dims, prb->dtag);

        dnnl::graph::op reorder_op(op1_id, dnnl::graph::op::kind::Reorder,
                graph.stringify_id(op1_id));

        graph.append(op1_id, reorder_op, {op0_dst_id}, {op1_dst_id});

        const auto op2_id = graph.generate_id_for(entry_kind::QUANTIZE);
        const auto op2_dst_id = graph.generate_id_for(op2_id, lt_kind::DST);

        graph.create_lt(op2_dst_id, dst_dt, prb->dims, prb->dtag);

        dnnl::graph::op quantize_op(op2_id, dnnl::graph::op::kind::Quantize,
                graph.stringify_id(op2_id));
        quantize_op.set_attr("qtype", qtype);
        set_quant_op_attr(quantize_op, qtype, scales, dst_zps, axis);

        graph.append(op2_dst_id, quantize_op, {op1_dst_id}, {op2_dst_id});
    } else if (is_reorder || is_tc || (!runtime && is_qdq)) {
        entry_kind_t e_kind;
        dnnl::graph::op::kind op_kind;
        if (is_reorder) {
            e_kind = entry_kind::REORDER;
            op_kind = dnnl::graph::op::kind::Reorder;
        } else if (is_tc) {
            e_kind = entry_kind::TYPECAST;
            op_kind = dnnl::graph::op::kind::TypeCast;
        } else {
            e_kind = is_quantize(src_dt, dst_dt) ? entry_kind::QUANTIZE
                                                 : entry_kind::DEQUANTIZE;
            op_kind = is_quantize(src_dt, dst_dt)
                    ? dnnl::graph::op::kind::Quantize
                    : dnnl::graph::op::kind::Dequantize;
        }

        const auto op_id = graph.generate_id_for(e_kind);
        const auto src_id = connect_to_previous_block
                ? graph.get_last_block_out_id()
                : graph.generate_id_for(op_id, lt_kind::SRC);
        const auto dst_id = graph.generate_id_for(op_id, lt_kind::DST);

        graph.create_lt(src_id, src_dt, prb->dims, prb->stag);
        graph.create_lt(dst_id, dst_dt, prb->dims, prb->dtag);

        dnnl::graph::op aop(op_id, op_kind, graph.stringify_id(op_id));
        if (is_qdq) {
            const auto zps = is_quantize(src_dt, dst_dt) ? dst_zps : src_zps;
            set_quant_op_attr(aop, qtype, scales, zps, axis);
        }

        graph.append(op_id, aop, {src_id}, {dst_id});
    } else if (runtime && is_qdq) {
        const auto e_kind = is_quantize(src_dt, dst_dt)
                ? entry_kind::QUANTIZE
                : entry_kind::DEQUANTIZE;
        const auto op_kind = is_quantize(src_dt, dst_dt)
                ? dnnl::graph::op::kind::DynamicQuantize
                : dnnl::graph::op::kind::DynamicDequantize;

        const auto op_id = graph.generate_id_for(e_kind);
        const auto src0_id = connect_to_previous_block
                ? graph.get_last_block_out_id()
                : graph.generate_id_for(op_id, lt_kind::SRC_I, size_t(0));
        const auto src1_id
                = graph.generate_id_for(op_id, lt_kind::SRC_I, size_t(1));
        const auto dst_id = graph.generate_id_for(op_id, lt_kind::DST);

        graph.create_lt(src0_id, src_dt, prb->dims, prb->stag);
        graph.create_lt(src1_id, graph_dt::f32, sz_dims,
                dnnl::graph::logical_tensor::layout_type::strided);
        graph.create_lt(dst_id, dst_dt, prb->dims, prb->dtag);

        std::vector<size_t> src_ids = {src0_id, src1_id};
        std::vector<size_t> dst_ids = {dst_id};
        if (!prb->attr.zero_points.is_def()) {
            const auto src2_id
                    = graph.generate_id_for(op_id, lt_kind::SRC_I, size_t(2));
            graph.create_lt(src2_id, graph_dt::s32, sz_dims,
                    dnnl::graph::logical_tensor::layout_type::strided);
            src_ids.push_back(src2_id);
        }

        dnnl::graph::op dynamic_qdq_op(
                op_id, op_kind, graph.stringify_id(op_id));
        dynamic_qdq_op.set_attr<std::string>("qtype", qtype);
        if (qtype == "per_channel")
            dynamic_qdq_op.set_attr<int64_t>("axis", axis);

        graph.append(op_id, dynamic_qdq_op, src_ids, dst_ids);
    } else if (is_tc_q) {
        const auto op0_id = graph.generate_id_for(entry_kind::TYPECAST);
        const auto op0_src_id = connect_to_previous_block
                ? graph.get_last_block_out_id()
                : graph.generate_id_for(op0_id, lt_kind::SRC);
        const auto op0_dst_id = graph.generate_id_for(op0_id, lt_kind::DST);

        graph.create_lt(op0_src_id, src_dt, prb->dims, prb->stag);
        graph.create_lt(op0_dst_id, graph_dt::f32, prb->dims, prb->stag);

        dnnl::graph::op typecast_op(op0_id, dnnl::graph::op::kind::TypeCast,
                graph.stringify_id(op0_id));

        graph.append(op0_id, typecast_op, {op0_src_id}, {op0_dst_id});

        const auto op1_id = graph.generate_id_for(entry_kind::QUANTIZE);
        const auto op1_dst_id = graph.generate_id_for(op1_id, lt_kind::DST);

        graph.create_lt(op1_dst_id, dst_dt, prb->dims, prb->dtag);

        dnnl::graph::op quantize_op(op1_id, dnnl::graph::op::kind::Quantize,
                graph.stringify_id(op1_id));
        set_quant_op_attr(quantize_op, qtype, scales, dst_zps, axis);

        graph.append(op1_id, quantize_op, {op0_dst_id}, {op1_dst_id});
    } else {
        return fill_status::UNHANDLED_CONFIG_OPTIONS;
    }

    // handle post ops
    fill_status_t status;
    for (const auto &entry : prb->attr.post_ops.entry) {
        if (entry.is_sum_kind()) {
            std::tie(status, std::ignore) = append_graph_with_sum(entry);
            BENCHDNNEXT_VERIFY(status);
        }
    }

    graph.close_block();

    return fill_status::DONE;
}

int doit(const ::reorder::prb_t *prb, res_t *res) {
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

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::reorder::init_pd, prb, res, par, ins, outs);

    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    // we need src_fp for proper comparison, => no in-place reference
    auto dst_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);

    auto src_dt = make_dnn_mem(ins[0], (prb->stag).c_str());
    auto dst_dt = make_dnn_mem(outs[0], (prb->dtag).c_str());

    //TODO: need to extend for post ops
    SAFE(fill_mem(prb, SRC, src_dt, src_fp), WARN);
    SAFE(src_dt.reorder(src_fp), WARN);

    dnn_mem_t scales_dt, zps_dt;
    std::vector<float> scales;
    std::vector<int64_t> src_zps, dst_zps;
    if (prb->attr.scales.get(DNNL_ARG_SRC).runtime
            && !is_runtime_to_static(
                    convert_dt(prb->sdt), convert_dt(prb->ddt))) {
        // axis is used only for PER_DIM_0 and PER_DIM_1 policies
        int64_t axis = prb->attr.scales.get(DNNL_ARG_SRC).policy
                        == attr_t::policy_t::PER_DIM_1
                ? 1
                : 0;

        prepare_runtime_scales(prb, scales_dt, ins[1], scales, axis);
        maybe_prepare_runtime_zero_points(
                prb, zps_dt, ins[2], src_zps, dst_zps, axis);
    }

    //TODO: fill for sum / zeropoints
    std::vector<dnnl::graph::tensor> tensors_in;
    std::vector<dnnl::graph::tensor> tensors_out;
    const dnnl::graph::engine &eng = get_test_engine();

    tensors_in.emplace_back(
            dnnl::graph::tensor(ins[0], eng, static_cast<void *>(src_dt)));
    if (prb->attr.scales.get(DNNL_ARG_SRC).runtime
            && !is_runtime_to_static(
                    convert_dt(prb->sdt), convert_dt(prb->ddt))) {
        tensors_in.emplace_back(dnnl::graph::tensor(
                ins[1], eng, static_cast<void *>(scales_dt)));
        if (!prb->attr.zero_points.is_def())
            tensors_in.emplace_back(dnnl::graph::tensor(
                    ins[2], eng, static_cast<void *>(zps_dt)));
    }
    tensors_out.emplace_back(
            dnnl::graph::tensor(outs[0], eng, static_cast<void *>(dst_dt)));

    if (!prb->attr.post_ops.entry.empty()) {
        dnnl::graph::tensor sum_src1_tensor = dnnl::graph::tensor(
                ins.back(), eng, static_cast<void *>(dst_dt));
        tensors_in.emplace_back(sum_src1_tensor);
    }

    SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

    if (is_bench_mode(CORR)) {
        const auto assign_comp_mem
                = [&](dnn_mem_t &m, ::reorder::flag_bit_t flag) {
                      if (prb->is_reorder_with_compensation(flag)) {
                          dims_t dims = prb->get_compensation_dims(flag);
                          dnnl::graph::logical_tensor var {(size_t)1000 + flag,
                                  dt::s32, dims, lt::undef};
                          m = make_dnn_mem(var, dt::s32, "abx");
                      }
                      return OK;
                  };

        dnn_mem_t dst_s8_comp_ref, dst_zp_comp_ref;
        assign_comp_mem(dst_s8_comp_ref, ::reorder::FLAG_S8S8_COMP);
        assign_comp_mem(dst_zp_comp_ref, ::reorder::FLAG_ZP_COMP);

        args_t args, ref_args;

        args.set(DNNL_ARG_TO, dst_dt);
        ref_args.set(DNNL_ARG_FROM, src_fp);
        ref_args.set(DNNL_ARG_TO, dst_fp);
        ref_args.set(DNNL_ARG_SRC_1, dst_s8_comp_ref); // Additional input
        ref_args.set(DNNL_ARG_SRC_2, dst_zp_comp_ref); // Additional input
        // ref_args.set(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, scales_dt);

        // Validate main reorder part.
        // Remove extra desc so that reorders with compensation could have
        // proper reorder from blocked layout to plain for comparison.
        dnnl::impl::memory_extra_desc_t empty_extra {};
        const auto orig_dst_extra = dst_dt.md_->extra;
        dst_dt.md_->extra = empty_extra;

        check_correctness(
                prb, {DST}, args, ref_args, ::reorder::setup_cmp, res);

        // Restore extra for compensation comparison and performance mode.
        dst_dt.md_->extra = orig_dst_extra;

        // Validate compensated reorder part.
        if (prb->is_reorder_with_compensation(::reorder::FLAG_ANY)) {
            compare_compensation(
                    prb, dst_s8_comp_ref, dst_zp_comp_ref, dst_dt, res);
        }
    }
    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);

    cleanup();

    return OK;
}

} // namespace reorder
} // namespace benchdnnext
