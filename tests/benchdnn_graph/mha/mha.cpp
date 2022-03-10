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

#include "mha/mha.hpp"

namespace mha {

int dt2cfg(graph_dt dt) {
    switch (dt) {
        case graph_dt::f32: return CFG_F32;
        case graph_dt::s8: return CFG_S8;
        case graph_dt::u8: return CFG_U8;
        case graph_dt::bf16: return CFG_BF16;
        default: assert(!"unsupported datatype"); return CFG_DT_UNSUPPORTED;
    }
}

std::ostream &operator<<(std::ostream &s, const mha_graph_spec_t &spec) {
    dump_global_params(s);
    settings_t def;

    s << "--pattern=" << spec.mha_pattern << " ";
    s << "--head=" << spec.head << " ";
    s << "--dt=" << convert_dt(spec.mha_inout_dt) << " ";
    s << "--tag=" << spec.tag << " ";
    if (!spec.quan_attr.oscale.is_def())
        s << "--attr-quan-oscale=" << spec.quan_attr.oscale << " ";
    if (!spec.quan_attr.zero_points.is_def())
        s << "--attr-quan-zero-points=" << spec.quan_attr.zero_points << " ";
    if (!spec.dequan_attr.oscale.is_def())
        s << "--attr-dequan-oscale=" << spec.dequan_attr.oscale << " ";
    if (!spec.dequan_attr.zero_points.is_def())
        s << "--attr-dequan-zero-points=" << spec.dequan_attr.zero_points
          << " ";
    s << spec.dims[0] << "x" << spec.dims[1] << "x" << spec.dims[2];

    return s;
}

void mha_graph_spec_t::generate_scales() {
    const attr_t::scale_t &oscale = quan_attr.oscale;
    int mask = attr_t::get_default_mask(oscale.policy);

    int64_t uniq_scales = 1;
    for (int d = 0; d < this->ndims; ++d)
        if (mask & (1 << d)) uniq_scales *= this->dims[d];

    for (int d = 0; d < uniq_scales; ++d)
        quan_scales.emplace_back(oscale.scale);
    if (uniq_scales > 1) quan_scales[uniq_scales - 1] /= 2.f;

    const attr_t::scale_t &dq_oscale = dequan_attr.oscale;
    mask = attr_t::get_default_mask(dq_oscale.policy);

    uniq_scales = 1;
    for (int d = 0; d < this->ndims; ++d)
        if (mask & (1 << d)) uniq_scales *= this->dims[d];

    for (int d = 0; d < uniq_scales; ++d)
        dequan_scales.emplace_back(dq_oscale.scale);
    if (uniq_scales > 1) dequan_scales[uniq_scales - 1] /= 2.f;
}

//TODO: zero_points policy per_tensor only supported.
//No per_channel support.
void mha_graph_spec_t::generate_zero_points() {
    const attr_t::zero_points_t &zero_points = quan_attr.zero_points;
    const auto &q_e = zero_points.get(DNNL_ARG_DST);
    assert(q_e.policy == policy_t::COMMON);
    quan_zps.emplace_back(q_e.value);
    const attr_t::zero_points_t &dq_zero_points = dequan_attr.zero_points;
    const auto &dq_e = dq_zero_points.get(DNNL_ARG_SRC);
    assert(dq_e.policy == policy_t::COMMON);
    dequan_zps.emplace_back(dq_e.value);
}

inline int fill_data(
        graph_dt dt, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {

    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return OK;

    assert(mem_dt.nelems() == mem_fp.nelems());

    const auto &c = cfg[dt2cfg(dt)];
    float c_f_min = c.f_min, c_f_max = c.f_max, c_f_scale = c.f_scale;

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);

    dnnl::impl::parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand msr(nelems + idx_start + 1);
        msr.discard(1);

        std::uniform_int_distribution<> gen(c_f_min, c_f_max);

        // make sure the first element is not zero
        if (idx_start == 0) {
            float val = 0;
            while (val == 0)
                val = static_cast<float>(gen(msr));
            mem_fp.set_elem(0, val * c_f_scale);
            idx_start += 1;
        }

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            auto val = static_cast<float>(gen(msr)) * c_f_scale;
            mem_fp.set_elem(idx, val);
        }
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);
    return OK;
}

fill_status_t mha_graph_prb_t::build_mha_subgraph(
        const mha_graph_spec_t &spec) {
    using op = dnnl::graph::op;

    dims_t qkv_shape = {
            spec.dims[0], spec.dims[1], spec.head, (spec.dims[2] / spec.head)};

    //attributes
    dims_t qkv_transpose_axis = {0, 2, 1, 3};
    dims_t k2_transpose_axis = {0, 1, 3, 2};

    size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);

    build_tensor_desc(spec);

    if (spec.mha_pattern == "v1") {
        addDequanOp(spec, STRINGIFY(QINT8_SRC), STRINGIFY(QSRC));
        addDequanOp(spec, STRINGIFY(KINT8_SRC), STRINGIFY(KSRC));
        addDequanOp(spec, STRINGIFY(VINT8_SRC), STRINGIFY(VSRC));
        addStaticReshapeOp(spec, STRINGIFY(QSRC), STRINGIFY(QTSRC), qkv_shape);
        addStaticReshapeOp(spec, STRINGIFY(KSRC), STRINGIFY(KTSRC), qkv_shape);
        addStaticReshapeOp(spec, STRINGIFY(VSRC), STRINGIFY(VTSRC), qkv_shape);
        addStaticTransposeOp(
                spec, STRINGIFY(QTSRC), STRINGIFY(QDST), qkv_transpose_axis);
        addStaticTransposeOp(
                spec, STRINGIFY(KTSRC), STRINGIFY(KT2SRC), qkv_transpose_axis);
        addStaticTransposeOp(
                spec, STRINGIFY(KT2SRC), STRINGIFY(KDST), k2_transpose_axis);
        addStaticTransposeOp(
                spec, STRINGIFY(VTSRC), STRINGIFY(VDST), qkv_transpose_axis);
    } else if (spec.mha_pattern == "v2") {
        addDequanOp(spec, STRINGIFY(QINT8_SRC), STRINGIFY(QDST));
        addDequanOp(spec, STRINGIFY(KINT8_SRC), STRINGIFY(KDST));
        addDequanOp(spec, STRINGIFY(VINT8_SRC), STRINGIFY(VDST));
        //v3 pattern only s8/u8 so output will be from dequan op
    } else if (spec.mha_pattern == "v3") {
        addDequanOp(spec, STRINGIFY(QINT8_SRC), STRINGIFY(QSRC));
        addDequanOp(spec, STRINGIFY(KINT8_SRC), STRINGIFY(KSRC));
        addDequanOp(spec, STRINGIFY(VINT8_SRC), STRINGIFY(VSRC));
        addTypecastOp(STRINGIFY(QSRC), STRINGIFY(QDST));
        addTypecastOp(STRINGIFY(KSRC), STRINGIFY(KDST));
        addTypecastOp(STRINGIFY(VSRC), STRINGIFY(VDST));
    }

    //matmul Q.KT
    ops_.emplace_back(op(ops_.size(), op::kind::MatMul,
            {tensor_descs_[STRINGIFY(QDST)], tensor_descs_[STRINGIFY(KDST)]},
            {tensor_descs_[STRINGIFY(QKMATMULDST)]}, "QK_matmul"));
    //score
    new_op_id = ops_.size();
    ops_.emplace_back(op(new_op_id, op::kind::Divide,
            {tensor_descs_[STRINGIFY(QKMATMULDST)],
                    tensor_descs_[STRINGIFY(QKDIVSCALE)]},
            {tensor_descs_[STRINGIFY(QKDIVDST)]}, "SCORE_divide"));
    ops_[new_op_id].set_attr("auto_broadcast", std::string("numpy"));
    //add
    new_op_id = ops_.size();
    ops_.emplace_back(op(new_op_id, op::kind::Add,
            {tensor_descs_[STRINGIFY(QKDIVDST)],
                    tensor_descs_[STRINGIFY(QKATTN)]},
            {tensor_descs_[STRINGIFY(QKADD)]}, "SCORE_add"));
    ops_[new_op_id].set_attr("auto_broadcast", std::string("numpy"));
    //softmax
    new_op_id = ops_.size();
    ops_.emplace_back(
            op(new_op_id, op::kind::SoftMax, {tensor_descs_[STRINGIFY(QKADD)]},
                    {tensor_descs_[STRINGIFY(QKSOFTMAX)]}, "SCORE_softmax"));
    ops_[new_op_id].set_attr("axis", static_cast<int64_t>(3));

    //For INT8 quantize and dequantize the softmax output.
    //For int8-bf16 there are typecast before and after quantize ops
    std::string quan_tensor = STRINGIFY(QKSOFTMAX);
    if (spec.mha_pattern == "v3") {
        addTypecastOp(STRINGIFY(QKSOFTMAX), STRINGIFY(QKSOFTMAXQUANCAST));
        quan_tensor = STRINGIFY(QKSOFTMAXQUANCAST);
    }
    addQuanOp(spec, quan_tensor, STRINGIFY(QKSOFTMAXQUAN));
    addDequanOp(spec, STRINGIFY(QKSOFTMAXQUAN), STRINGIFY(QKSOFTMAXDEQUAN));
    std::string matmul_qkvtensor = (spec.MHA_int8) ? STRINGIFY(QKSOFTMAXDEQUAN)
                                                   : STRINGIFY(QKSOFTMAX);
    if (spec.mha_pattern == "v3") {
        addTypecastOp(matmul_qkvtensor, STRINGIFY(QKSOFTMAXDEQUANCAST));
        matmul_qkvtensor = STRINGIFY(QKSOFTMAXDEQUANCAST);
    }

    //QKV matmul
    ops_.emplace_back(op(ops_.size(), op::kind::MatMul,
            {tensor_descs_[matmul_qkvtensor], tensor_descs_[STRINGIFY(VDST)]},
            {tensor_descs_[STRINGIFY(QKVMATMUL)]}, "QKV_matmul"));
    //QKV transpose
    addStaticTransposeOp(
            spec, STRINGIFY(QKVMATMUL), STRINGIFY(QKVTDST), qkv_transpose_axis);

    if (spec.mha_pattern == "v1") {
        addStaticReshapeOp(
                spec, STRINGIFY(QKVTDST), STRINGIFY(QKVDST), spec.dims);
    } else if (spec.mha_pattern == "v2") {
        addReorderOp(STRINGIFY(QKVTDST), STRINGIFY(QKVDST));
    } else if (spec.mha_pattern == "v3") {
        addReorderOp(STRINGIFY(QKVTDST), STRINGIFY(QKVCASTDST));
        addTypecastOp(STRINGIFY(QKVCASTDST), STRINGIFY(QKVDST));
    } else {
        return fill_status::UNHANDLED_CONFIG_OPTIONS;
    }

    addQuanOp(spec, STRINGIFY(QKVDST), STRINGIFY(QKVINT8DST));
    curr_out_map_ids_.assign({TENSOR_ID});
    return fill_status::DONE;
}

void check_known_skipped_case(const mha_graph_spec_t *spec, res_t *res) {
    if ((spec->mha_pattern == "")
            || (spec->mha_pattern != "v1" && spec->mha_pattern != "v2"
                    && spec->mha_pattern != "v3")) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
    }
    if ((spec->mha_pattern == "v3")
            && (spec->mha_inout_dt == graph_dt::f32
                    || spec->mha_inout_dt == graph_dt::bf16)) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
    }

    if (spec->head > spec->dims[2]) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
    }
    if ((((int)(spec->dims[2] / spec->head)) * spec->head) != spec->dims[2]) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
    }
    if (spec->quan_attr.oscale.policy != attr_t::policy_t::COMMON
            || spec->dequan_attr.oscale.policy != attr_t::policy_t::COMMON) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
    }
}
int doit(const mha_graph_spec_t *spec, res_t *res) {

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case(spec, res);
    if (res->state == SKIPPED) return OK;

    mha_graph_prb_t graph_prb(*spec);
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

    auto ins = par.get_in_ports();
    auto outs = par.get_out_ports();
    dnnl::graph::engine &engine = benchdnnext::get_test_engine();
    //sort the in_port based on id
    std::sort(ins.begin(), ins.end(),
            [](const dnnl::graph::logical_tensor &a,
                    const dnnl::graph::logical_tensor &b) {
                return a.get_id() < b.get_id();
            });
    //Compile Partitions.
    auto cp = par.compile(ins, outs, engine);

    auto q_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto k_fp = make_dnn_mem(ins[1], dt::f32, tag::abx);
    auto v_fp = make_dnn_mem(ins[2], dt::f32, tag::abx);

    auto q_dt = make_dnn_mem(ins[0], spec->tag);
    auto k_dt = make_dnn_mem(ins[1], spec->tag);
    auto v_dt = make_dnn_mem(ins[2], spec->tag);

    // TODO: use for correctness check
    //auto mha_out_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);
    auto mha_out_dt = make_dnn_mem(outs[0], spec->tag);
    fill_data(spec->mha_dt, q_dt, q_fp, res);
    fill_data(spec->mha_dt, k_dt, k_fp, res);
    fill_data(spec->mha_dt, v_dt, v_fp, res);

    float scale = sqrt(spec->dims[2] / spec->head);
    //number of elements = bs * seq_len
    std::vector<int> attention_mask_filter(spec->dims[0] * spec->dims[1], 1);

    std::vector<dnnl::graph::tensor> tensors_in;
    std::vector<dnnl::graph::tensor> tensors_out;
    //Execute Partitions.
    tensors_in.emplace_back(
            dnnl::graph::tensor(ins[0], engine, static_cast<void *>(q_dt)));
    tensors_in.emplace_back(
            dnnl::graph::tensor(ins[1], engine, static_cast<void *>(k_dt)));
    tensors_in.emplace_back(
            dnnl::graph::tensor(ins[2], engine, static_cast<void *>(v_dt)));
    tensors_in.emplace_back(
            dnnl::graph::tensor(ins[3], engine, static_cast<void *>(&scale)));
    tensors_in.emplace_back(dnnl::graph::tensor(
            ins[4], engine, static_cast<void *>(attention_mask_filter.data())));
    tensors_out.emplace_back(dnnl::graph::tensor(
            outs[0], engine, static_cast<void *>(mha_out_dt)));
    SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);
    //TODO: when correctnes check enabled.
    res->state = PASSED;
    return OK;
}
} // namespace mha
