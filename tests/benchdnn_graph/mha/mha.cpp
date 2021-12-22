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
    int QKV_IN_SIZE = 12;
    int bs = spec.dims[0];
    int seq_len = spec.dims[1];
    int embed_sz = spec.dims[2];
    int query_sz = embed_sz / spec.head;

    //shapes
    dims_t input_shape = spec.dims;
    dims_t qkv_shape = {bs, seq_len, spec.head, query_sz};
    dims_t qkv_transpose_shape = {bs, spec.head, seq_len, query_sz};
    dims_t qkv_transpose_axis = {0, 2, 1, 3};
    dims_t k2_transpose_shape = {bs, spec.head, query_sz, seq_len};
    dims_t k2_transpose_axis = {0, 1, 3, 2};
    dims_t qk_matmul_shape = {bs, spec.head, seq_len, seq_len};
    dims_t attention_mask_shape = {bs, 1, 1, seq_len};
    //attributes
    bool special_zero = false;

    size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);

    std::string str[QKV_IN_SIZE];
    std::vector<dnnl::graph::op> ops;
    ops.reserve(QKV_IN_SIZE);
    //Q(i=0), K(i=1), V(i=2)
    for (int i = 0; i < 3; i++) {
        str[i * 4] = TENSOR_ID + "_INT8_" + std::to_string(i * 4);
        str[i * 4 + 1] = TENSOR_ID + "_SRC_" + std::to_string(i * 4 + 1);
        str[i * 4 + 2] = TENSOR_ID + "_DST_" + std::to_string(i * 4 + 2);
        str[i * 4 + 3] = TENSOR_ID + "_TDST_" + std::to_string(i * 4 + 3);
        if (spec.MHA_int8) {
            tensor_descs_.emplace(
                    str[i * 4], spec.mha_inout_dt, input_shape, spec.tag);
        }
        tensor_descs_.emplace(
                str[i * 4 + 1], spec.mha_dt, input_shape, spec.tag);
        tensor_descs_.emplace(str[i * 4 + 2], spec.mha_dt, qkv_shape, spec.tag);
        tensor_descs_.emplace(
                str[i * 4 + 3], spec.mha_dt, qkv_transpose_shape, spec.tag);
        //Dequantize
        if (spec.MHA_int8) {
            new_op_id = ops_.size();
            ops.emplace_back(op(new_op_id, op::kind::Dequantize,
                    {tensor_descs_[str[i * 4]]},
                    {tensor_descs_[str[i * 4 + 1]]},
                    str[i * 4] + "_dequantize"));

            ops[new_op_id].set_attr<std::string>("qtype", spec.dequan_qtype);
            ops[new_op_id].set_attr<std::vector<int64_t>>(
                    "zps", spec.dequan_zps);
            ops[new_op_id].set_attr<std::vector<float>>(
                    "scales", spec.dequan_scales);
            ops_.emplace_back(ops[new_op_id]);
        }
        //Reshape
        new_op_id = ops_.size();
        ops.emplace_back(op(new_op_id, op::kind::StaticReshape,
                {tensor_descs_[str[i * 4 + 1]]},
                {tensor_descs_[str[i * 4 + 2]]}, str[i * 4 + 1] + "_reshape"));
        ops[new_op_id].set_attr("shape", qkv_shape);
        ops[new_op_id].set_attr("special_zero", special_zero);
        ops_.emplace_back(ops[new_op_id]);

        //Transpose
        new_op_id = ops_.size();
        ops.emplace_back(op(new_op_id, op::kind::StaticTranspose,
                {tensor_descs_[str[i * 4 + 2]]},
                {tensor_descs_[str[i * 4 + 3]]},
                str[i * 4 + 3] + "_transpose"));
        ops[new_op_id].set_attr("order", qkv_transpose_axis);
        ops_.emplace_back(ops[new_op_id]);
    }
    //K transpose 2
    new_op_id = ops_.size();
    const std::string KT2DST {TENSOR_ID + "_KT2DST"};
    tensor_descs_.emplace(KT2DST, spec.mha_dt, k2_transpose_shape, spec.tag);
    op k2_transpose_op(new_op_id, op::kind::StaticTranspose,
            {tensor_descs_[str[7]]}, {tensor_descs_[KT2DST]}, "KT2_transpose");
    k2_transpose_op.set_attr("order", k2_transpose_axis);
    ops_.emplace_back(k2_transpose_op);
    //matmul Q.KT
    new_op_id = ops_.size();
    const std::string QKMATMULDST {TENSOR_ID + "_QKMATMULDST"};
    tensor_descs_.emplace(QKMATMULDST, spec.mha_dt, qk_matmul_shape, spec.tag);
    op qk_matmul_op(new_op_id, op::kind::MatMul,
            {tensor_descs_[str[3]], tensor_descs_[KT2DST]},
            {tensor_descs_[QKMATMULDST]}, "QK_matmul");
    ops_.emplace_back(qk_matmul_op);
    //score
    new_op_id = ops_.size();
    const std::string QKDIVSCALE {TENSOR_ID + "_QKDIVSCALE"};
    const std::string QKDIV {TENSOR_ID + "_QKDIV"};
    //Divscale is f32 or bf16 depending on input.
    tensor_descs_.emplace(QKDIVSCALE, spec.mha_dt, {1}, spec.tag);
    tensor_descs_.emplace(QKDIV, spec.mha_dt, qk_matmul_shape, spec.tag);
    op qk_divide_op(new_op_id, op::kind::Divide,
            {tensor_descs_[QKMATMULDST], tensor_descs_[QKDIVSCALE]},
            {tensor_descs_[QKDIV]}, "SCORE_divide");
    qk_divide_op.set_attr("auto_broadcast", std::string("numpy"));
    ops_.emplace_back(qk_divide_op);
    //add
    new_op_id = ops_.size();
    const std::string QKATTN {TENSOR_ID + "_QKATTN"};
    const std::string QKADD {TENSOR_ID + "_QKADD"};
    tensor_descs_.emplace(QKATTN, spec.mha_dt, attention_mask_shape, spec.tag);
    tensor_descs_.emplace(QKADD, spec.mha_dt, qk_matmul_shape, spec.tag);
    op qk_add_op(new_op_id, op::kind::Add,
            {tensor_descs_[QKDIV], tensor_descs_[QKATTN]},
            {tensor_descs_[QKADD]}, "SCORE_add");
    qk_add_op.set_attr("auto_broadcast", std::string("numpy"));
    ops_.emplace_back(qk_add_op);
    //softmax
    new_op_id = ops_.size();
    const std::string QKSOFTMAX {TENSOR_ID + "_QKSOFTMAX"};
    tensor_descs_.emplace(QKSOFTMAX, spec.mha_dt, qk_matmul_shape, spec.tag);
    op qk_softmax_op(new_op_id, op::kind::SoftMax, {tensor_descs_[QKADD]},
            {tensor_descs_[QKSOFTMAX]}, "SCORE_softmax");
    qk_softmax_op.set_attr("axis", static_cast<int64_t>(3));
    ops_.emplace_back(qk_softmax_op);
    std::string intensor = QKSOFTMAX;
    //For INT8 quantize and dequantize the softmax output.
    if (spec.MHA_int8) {
        new_op_id = ops_.size();
        const std::string QKSOFTMAXQUAN {TENSOR_ID + "_QKSOFTMAXQUAN"};
        tensor_descs_.emplace(
                QKSOFTMAXQUAN, spec.mha_inout_dt, qk_matmul_shape, spec.tag);
        op qk_softmax_quan_op(new_op_id, op::kind::Quantize,
                {tensor_descs_[QKSOFTMAX]}, {tensor_descs_[QKSOFTMAXQUAN]},
                "softmax_quantize");
        qk_softmax_quan_op.set_attr<std::string>("qtype", spec.quan_qtype);
        qk_softmax_quan_op.set_attr<std::vector<int64_t>>("zps", spec.quan_zps);
        qk_softmax_quan_op.set_attr<std::vector<float>>(
                "scales", spec.quan_scales);
        ops_.emplace_back(qk_softmax_quan_op);

        new_op_id = ops_.size();
        const std::string QKSOFTMAXDEQUAN {TENSOR_ID + "_QKSOFTMAXDEQUAN"};
        tensor_descs_.emplace(
                QKSOFTMAXDEQUAN, spec.mha_dt, qk_matmul_shape, spec.tag);
        op qk_softmax_dequan_op(new_op_id, op::kind::Dequantize,
                {tensor_descs_[QKSOFTMAXQUAN]},
                {tensor_descs_[QKSOFTMAXDEQUAN]}, "softmax_dequantize");
        qk_softmax_dequan_op.set_attr<std::string>("qtype", spec.dequan_qtype);
        qk_softmax_dequan_op.set_attr<std::vector<int64_t>>(
                "zps", spec.dequan_zps);
        qk_softmax_dequan_op.set_attr<std::vector<float>>(
                "scales", spec.dequan_scales);
        ops_.emplace_back(qk_softmax_dequan_op);
        intensor = QKSOFTMAXDEQUAN;
    }

    //QKV matmul
    new_op_id = ops_.size();
    const std::string QKVMATMUL {TENSOR_ID + "_QKVMATMUL"};
    tensor_descs_.emplace(
            QKVMATMUL, spec.mha_dt, qkv_transpose_shape, spec.tag);
    op qkv_matmul_op(new_op_id, op::kind::MatMul,
            {tensor_descs_[intensor], tensor_descs_[str[11]]},
            {tensor_descs_[QKVMATMUL]}, "QKV_matmul");
    ops_.emplace_back(qkv_matmul_op);
    //QKV transpose
    new_op_id = ops_.size();
    const std::string QKVTDST {TENSOR_ID + "_QKVTDST"};
    tensor_descs_.emplace(QKVTDST, spec.mha_dt, qkv_shape, spec.tag);
    op qkv_transpose_op(new_op_id, op::kind::StaticTranspose,
            {tensor_descs_[QKVMATMUL]}, {tensor_descs_[QKVTDST]},
            "QKV_transpose");
    qkv_transpose_op.set_attr("order", qkv_transpose_axis);
    ops_.emplace_back(qkv_transpose_op);
    //QKV reshape
    new_op_id = ops_.size();
    const std::string QKVDST {TENSOR_ID + "_QKVDST"};
    tensor_descs_.emplace(QKVDST, spec.mha_dt, input_shape, spec.tag);
    op qkv_reshape_op(new_op_id, op::kind::StaticReshape,
            {tensor_descs_[QKVTDST]}, {tensor_descs_[QKVDST]}, "QKV_reshape");
    qkv_reshape_op.set_attr("shape", input_shape);
    qkv_reshape_op.set_attr("special_zero", special_zero);
    ops_.emplace_back(qkv_reshape_op);

    if (spec.MHA_int8) {
        new_op_id = ops_.size();
        const std::string QKVINT8DST {TENSOR_ID + "_QKVINT8DST"};
        tensor_descs_.emplace(
                QKVINT8DST, spec.mha_inout_dt, input_shape, spec.tag);
        op qkv_quan_op(new_op_id, op::kind::Quantize, {tensor_descs_[QKVDST]},
                {tensor_descs_[QKVINT8DST]}, "QKV_out_quan");
        qkv_quan_op.set_attr<std::string>("qtype", spec.quan_qtype);
        qkv_quan_op.set_attr<std::vector<int64_t>>("zps", spec.quan_zps);
        qkv_quan_op.set_attr<std::vector<float>>("scales", spec.quan_scales);
        ops_.emplace_back(qkv_quan_op);
    }
    curr_out_map_ids_.assign({TENSOR_ID});
    return fill_status::DONE;
}

void check_known_skipped_case(const mha_graph_spec_t *spec, res_t *res) {
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
    SAFE(execute_and_wait(cp, tensors_in, tensors_out), WARN);

    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);
    //TODO: when correctnes check enabled.
    res->state = PASSED;
    return OK;
}
} // namespace mha
