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

#include <utility>
#include "utils/compare.hpp"

#include "reorder/graph_reorder.hpp"

namespace benchdnnext {
namespace reorder {

void check_known_skipped_case_graph(
        const ::reorder::prb_t *prb, res_t *res) noexcept {
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
            && prb->attr.oscale.policy != attr_t::policy_t::COMMON) {
        res->state = SKIPPED;
    }

    check_graph_scales_and_zps_support(prb->attr, res);
}

bool is_quantize(graph_dt src_dt, graph_dt dst_dt) {
    return src_dt == graph_dt::f32 && is_low_precision({dst_dt});
}

bool is_dequantize(graph_dt src_dt, graph_dt dst_dt) {
    return dst_dt == graph_dt::f32 && is_low_precision({src_dt});
}

void set_quant_op_attr(dnnl::graph::op &op_, const std::string &qtype,
        const std::vector<float> &scales, const std::vector<int64_t> &zps,
        const int64_t &axis) {
    op_.set_attr("qtype", qtype);
    //TODO: use zps - revisit
    op_.set_attr("zps", zps);
    op_.set_attr("scales", scales);
    //TODO: axis doesnt support PER_DIM_01
    if (qtype == "per_channel") op_.set_attr("axis", axis);
}

int fill_zps(const ::reorder::prb_t *prb, const int64_t axis,
        std::vector<int64_t> &src_zps, std::vector<int64_t> &dst_zps) {
    if (prb->attr.oscale.policy == attr_t::policy_t::COMMON) {
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

int fill_scales(const ::reorder::prb_t *prb, const int64_t axis,
        std::vector<float> &scales) {
    if (prb->attr.oscale.policy == attr_t::policy_t::COMMON) {
        scales.emplace_back(prb->scales[0]);
    } else {
        //TODO: needs update for PER_DIM_01
        for (int i = 0; i < prb->dims[axis]; i++) {
            scales.emplace_back(prb->scales[i]);
        }
    }
    //Need to inverse scale
    if (prb->ddt == dnnl_s8 || prb->ddt == dnnl_u8) {
        for (int i = 0; i < scales.size(); i++) {
            scales[i] = 1.f / scales[i];
        }
    }
    return OK;
}

void prepare_runtime_scales(const ::reorder::prb_t *prb, dnn_mem_t &scales_dt,
        const dnnl::graph::logical_tensor &in, std::vector<float> scales,
        int64_t axis) {
    // scales is required input for dynamic q/deq
    scales_dt = make_dnn_mem(in, dt::f32, tag::x);
    fill_scales(prb, axis, scales);
    for (int i = 0; i < scales.size(); i++) {
        scales_dt.set_elem(i, scales[i]);
    }
}

void maybe_prepare_runtime_zero_points(const ::reorder::prb_t *prb,
        dnn_mem_t &zps_dt, const dnnl::graph::logical_tensor &in,
        std::vector<int64_t> src_zps, std::vector<int64_t> dst_zps,
        int64_t axis) {
    // zps is optional input for dynamic q/deq
    if (prb->attr.zero_points.is_def()) return;

    zps_dt = make_dnn_mem(in, dt::s32, tag::x);
    fill_zps(prb, axis, src_zps, dst_zps);

    if (is_quantize(convert_dt(prb->sdt), convert_dt(prb->ddt))) {
        for (int i = 0; i < dst_zps.size(); i++) {
            zps_dt.set_elem(i, static_cast<int32_t>(dst_zps[i]));
        }
    } else {
        for (int i = 0; i < src_zps.size(); i++) {
            zps_dt.set_elem(i, static_cast<int32_t>(src_zps[i]));
        }
    }
}

fill_status_t reorder_graph_prb_t::handle_main_op_(
        const ::reorder::prb_t *prb) {
    using op = dnnl::graph::op;

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);
    const std::string SRC {TENSOR_ID + "_SRC"};
    const std::string DST {TENSOR_ID + "_DST"};
    // specific for dynamic q/deq
    const std::string SCALES {TENSOR_ID + "_SCALES"};
    const std::string ZPS {TENSOR_ID + "_ZPS"};

    const auto src_dt = convert_dt(prb->sdt);
    const auto dst_dt = convert_dt(prb->ddt);
    const auto qtype = convert_attr_policy(prb->attr.oscale.policy);
    bool runtime = prb->attr.oscale.runtime;
    // axis is used only for PER_DIM_0 and PER_DIM_1 policies
    int64_t axis
            = prb->attr.oscale.policy == attr_t::policy_t::PER_DIM_1 ? 1 : 0;

    std::vector<float> scales;
    std::vector<float> default_scales({1.0});
    std::vector<int64_t> src_zps, dst_zps;

    if (!runtime) {
        fill_scales(prb, axis, scales);
        fill_zps(prb, axis, src_zps, dst_zps);
    }

    const auto sz_dims
            = qtype == "per_channel" ? dims_t {prb->dims[axis]} : dims_t {1};

    tensor_descs_.emplace(SRC, src_dt, prb->dims, prb->stag);
    tensor_descs_.emplace(DST, dst_dt, prb->dims, prb->dtag);

    if (is_low_precision({src_dt}) && is_low_precision({dst_dt})) {
        //SRC->SRC_F32->DST_F32->DST
        const std::string SRC_F32 {TENSOR_ID + "_SRC_F32"};
        const std::string DST_F32 {TENSOR_ID + "_DST_F32"};
        tensor_descs_.emplace(SRC_F32, graph_dt::f32, prb->dims, prb->stag);
        tensor_descs_.emplace(DST_F32, graph_dt::f32, prb->dims, prb->dtag);

        op dequantize_op(new_op_id, op::kind::Dequantize, {tensor_descs_[SRC]},
                {tensor_descs_[SRC_F32]}, "dequantize");
        set_quant_op_attr(dequantize_op, qtype, default_scales, src_zps, axis);
        ops_.emplace_back(dequantize_op);

        op reorder_op(ops_.size(), op::kind::Reorder, {tensor_descs_[SRC_F32]},
                {tensor_descs_[DST_F32]}, "reorder");
        ops_.emplace_back(reorder_op);

        op quantize_op(ops_.size(), op::kind::Quantize,
                {tensor_descs_[DST_F32]}, {tensor_descs_[DST]}, "quantize");
        quantize_op.set_attr("qtype", qtype);
        set_quant_op_attr(quantize_op, qtype, scales, dst_zps, axis);
        ops_.emplace_back(quantize_op);
    } else if (src_dt == dst_dt) {
        op reorder_op(new_op_id, op::kind::Reorder, {tensor_descs_[SRC]},
                {tensor_descs_[DST]}, "reorder");
        ops_.emplace_back(reorder_op);
    } else if ((src_dt == graph_dt::f32 && dst_dt == graph_dt::bf16)
            || (src_dt == graph_dt::bf16 && dst_dt == graph_dt::f32)) {
        op typecast_op(new_op_id, op::kind::TypeCast, {tensor_descs_[SRC]},
                {tensor_descs_[DST]}, "typecast");
        ops_.emplace_back(typecast_op);
    } else if (is_quantize(src_dt, dst_dt)) {
        if (!runtime) {
            op quantize_op(new_op_id, op::kind::Quantize, {tensor_descs_[SRC]},
                    {tensor_descs_[DST]}, "quantize");
            set_quant_op_attr(quantize_op, qtype, scales, dst_zps, axis);
            ops_.emplace_back(quantize_op);
        } else {
            tensor_descs_.emplace(SCALES, graph_dt::f32, sz_dims, lt::strided);
            std::vector<dnnl::graph::logical_tensor> inputs
                    = {tensor_descs_[SRC], tensor_descs_[SCALES]};
            if (!prb->attr.zero_points.is_def()) {
                tensor_descs_.emplace(ZPS, graph_dt::s32, sz_dims, lt::strided);
                inputs.push_back(tensor_descs_[ZPS]);
            }
            op dync_quantize(new_op_id, op::kind::DynamicQuantize, inputs,
                    {tensor_descs_[DST]}, "dynamic_quantize");
            dync_quantize.set_attr<std::string>("qtype", qtype);
            if (qtype == "per_channel")
                dync_quantize.set_attr<int64_t>("axis", axis);
            ops_.emplace_back(dync_quantize);
        }
    } else if (is_dequantize(src_dt, dst_dt)) {
        if (!runtime) {
            op dequantize_op(new_op_id, op::kind::Dequantize,
                    {tensor_descs_[SRC]}, {tensor_descs_[DST]}, "dequantize");
            set_quant_op_attr(dequantize_op, qtype, scales, src_zps, axis);
            ops_.emplace_back(dequantize_op);
        } else {
            tensor_descs_.emplace(SCALES, graph_dt::f32, sz_dims, lt::strided);
            std::vector<dnnl::graph::logical_tensor> inputs
                    = {tensor_descs_[SRC], tensor_descs_[SCALES]};

            if (!prb->attr.zero_points.is_def()) {
                tensor_descs_.emplace(ZPS, graph_dt::s32, sz_dims, lt::strided);
                inputs.push_back(tensor_descs_[ZPS]);
            }
            op dync_dequantize(new_op_id, op::kind::DynamicDequantize, inputs,
                    {tensor_descs_[DST]}, "dynamic_dequantize");
            dync_dequantize.set_attr<std::string>("qtype", qtype);
            if (qtype == "per_channel")
                dync_dequantize.set_attr<int64_t>("axis", axis);
            ops_.emplace_back(dync_dequantize);
        }
    } else if (src_dt == graph_dt::bf16 && is_low_precision({dst_dt})) {
        const std::string SRC_F32 {TENSOR_ID + "_SRC_F32"};
        tensor_descs_.emplace(SRC_F32, graph_dt::f32, prb->dims, prb->stag);
        op typecast_op(new_op_id, op::kind::TypeCast, {tensor_descs_[SRC]},
                {tensor_descs_[SRC_F32]}, "typecast");
        ops_.emplace_back(typecast_op);
        op quantize_op(ops_.size(), op::kind::Quantize,
                {tensor_descs_[SRC_F32]}, {tensor_descs_[DST]}, "quantize");
        set_quant_op_attr(quantize_op, qtype, scales, dst_zps, axis);
        ops_.emplace_back(quantize_op);
    } else {
        return fill_status::UNHANDLED_CONFIG_OPTIONS;
    }

    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

fill_status_t reorder_graph_prb_t::handle_sum_() {
    return po_handler.reorder.sum_handler(*this);
}

int doit(const ::reorder::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    // TODO: to align with original benchdnn, we should consider moving
    // skip_unimplemented_prb call after compilation step
    skip_invalid_and_unimplemented_prb(prb, res);
    if (res->state == SKIPPED) return OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

    reorder_graph_prb_t graph_prb(prb);
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

    auto cp = compile_partition(::reorder::init_pd, prb, res, par, ins, outs);

    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    // we need src_fp for proper comparison, => no in-place reference
    auto dst_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);

    auto src_dt = make_dnn_mem(ins[0], (prb->stag).c_str());
    auto dst_dt = make_dnn_mem(outs[0], (prb->dtag).c_str());

    //TODO: need to extend for post ops
    SAFE(fill_memory(prb, SRC, src_fp), WARN);
    SAFE(src_dt.reorder(src_fp), WARN);

    dnn_mem_t scales_dt, zps_dt;
    std::vector<float> scales;
    std::vector<int64_t> src_zps, dst_zps;
    if (prb->attr.oscale.runtime) {
        // axis is used only for PER_DIM_0 and PER_DIM_1 policies
        int64_t axis = prb->attr.oscale.policy == attr_t::policy_t::PER_DIM_1
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
    if (prb->attr.oscale.runtime) {
        tensors_in.emplace_back(dnnl::graph::tensor(
                ins[1], eng, static_cast<void *>(scales_dt)));
        if (!prb->attr.zero_points.is_def())
            tensors_in.emplace_back(dnnl::graph::tensor(
                    ins[2], eng, static_cast<void *>(zps_dt)));
    }
    tensors_out.emplace_back(
            dnnl::graph::tensor(outs[0], eng, static_cast<void *>(dst_dt)));

    if (graph_prb.has_post_sum()) {
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

        // Validate main reorder part.
        // Remove extra desc so that reorders with compensation could have
        // proper reorder from blocked layout to plain for comparison.
        dnnl_memory_extra_desc_t empty_extra {};
        const auto orig_dst_extra = dst_dt.md_.extra;
        dst_dt.md_.extra = empty_extra;

        check_correctness(
                prb, {DST}, args, ref_args, ::reorder::setup_cmp, res);

        // Restore extra for compensation comparison and performance mode.
        dst_dt.md_.extra = orig_dst_extra;

        // Validate compensated reorder part.
        if (prb->is_reorder_with_compensation(::reorder::FLAG_ANY)) {
            compare_compensation(
                    prb, dst_s8_comp_ref, dst_zp_comp_ref, dst_dt, res);
        }
    }
    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);

    return OK;
}

} // namespace reorder
} // namespace benchdnnext
