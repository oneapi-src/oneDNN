/*******************************************************************************
* Copyright 2022 Intel Corporation
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
#include <map>
#include "mha.hpp"

namespace mha {

void mha_graph_prb_t::add_dequan_op(const mha_graph_spec_t &spec,
        const std::string src, const std::string dst) {
    if (!spec.MHA_int8) { return; }
    using op = dnnl::graph::op;
    size_t op_id = ops_.size();
    ops_.emplace_back(op(op_id, op::kind::Dequantize, {tensor_descs_[src]},
            {tensor_descs_[dst]}, dst + "_dequantize"));
    ops_[op_id].set_attr<std::string>("qtype", spec.dequan_qtype);
    ops_[op_id].set_attr<std::vector<int64_t>>("zps", spec.dequan_zps);
    ops_[op_id].set_attr<std::vector<float>>("scales", spec.dequan_scales);
}

void mha_graph_prb_t::add_staticreshape_op(const mha_graph_spec_t &spec,
        const std::string src, const std::string dst, const dims_t shape) {
    using op = dnnl::graph::op;
    size_t op_id = ops_.size();
    ops_.emplace_back(op(op_id, op::kind::StaticReshape, {tensor_descs_[src]},
            {tensor_descs_[dst]}, dst + "_reshape"));
    ops_[op_id].set_attr("shape", shape);
    ops_[op_id].set_attr("special_zero", false);
}

void mha_graph_prb_t::add_statictranspose_op(const mha_graph_spec_t &spec,
        const std::string src, const std::string dst, dims_t axis) {
    using op = dnnl::graph::op;
    size_t op_id = ops_.size();
    ops_.emplace_back(op(op_id, op::kind::StaticTranspose, {tensor_descs_[src]},
            {tensor_descs_[dst]}, dst + "_transpose"));
    ops_[op_id].set_attr("order", axis);
}

void mha_graph_prb_t::add_quan_op(const mha_graph_spec_t &spec,
        const std::string src, const std::string dst) {
    if (!spec.MHA_int8) { return; }
    using op = dnnl::graph::op;
    size_t op_id = ops_.size();
    ops_.emplace_back(op(op_id, op::kind::Quantize, {tensor_descs_[src]},
            {tensor_descs_[dst]}, dst + "_quantize"));
    ops_[op_id].set_attr<std::string>("qtype", spec.quan_qtype);
    ops_[op_id].set_attr<std::vector<int64_t>>("zps", spec.quan_zps);
    ops_[op_id].set_attr<std::vector<float>>("scales", spec.quan_scales);
}
void mha_graph_prb_t::add_typecast_op(
        const std::string src, const std::string dst) {
    using op = dnnl::graph::op;
    ops_.emplace_back(op(ops_.size(), op::kind::TypeCast, {tensor_descs_[src]},
            {tensor_descs_[dst]}, dst + "_typecast"));
}

void mha_graph_prb_t::add_reorder_op(
        const std::string src, const std::string dst) {
    using op = dnnl::graph::op;
    ops_.emplace_back(op(ops_.size(), op::kind::Reorder, {tensor_descs_[src]},
            {tensor_descs_[dst]}, dst + "_reorder"));
}

void mha_graph_prb_t::add_matmul_op(bool is_transpose_a, bool is_transpose_b,
        const std::string src1, const std::string src2, const std::string dst) {
    using op = dnnl::graph::op;
    auto new_op_id = ops_.size();
    ops_.emplace_back(op(new_op_id, op::kind::MatMul,
            {tensor_descs_[src1], tensor_descs_[src2]}, {tensor_descs_[dst]},
            dst));
    ops_[new_op_id].set_attr("transpose_a", is_transpose_a);
    ops_[new_op_id].set_attr("transpose_b", is_transpose_b);
}

void mha_graph_prb_t::add_arith_op(bool add_op, dnnl::graph::op::kind op_kind,
        bool set_broadcast, const std::string src1, const std::string src2,
        const std::string dst) {
    using op = dnnl::graph::op;
    if (!add_op) { return; }
    auto new_op_id = ops_.size();
    ops_.emplace_back(
            op(new_op_id, op_kind, {tensor_descs_[src1], tensor_descs_[src2]},
                    {tensor_descs_[dst]}, dst));
    if (set_broadcast) {
        ops_[new_op_id].set_attr("auto_broadcast", std::string("numpy"));
    }
}
void mha_graph_prb_t::add_reducesum_op(const mha_graph_spec_t &spec,
        const std::string src, const std::string dst) {
    if (spec.dims[0] == 1) return;
    using op = dnnl::graph::op;
    auto op_id = ops_.size();
    ops_.emplace_back(op(op_id, op::kind::ReduceSum, {tensor_descs_[src]},
            {tensor_descs_[dst]}, dst));
    ops_[op_id].set_attr("axes", std::vector<int64_t> {-1});
    ops_[op_id].set_attr("keep_dims", true);
}

void mha_graph_prb_t::add_end_op(const std::string src) {
    using op = dnnl::graph::op;
    ops_.emplace_back(op(ops_.size(), op::kind::End, {tensor_descs_[src]}, {},
            "end_" + src));
}

void mha_graph_prb_t::build_tensor_desc_fwd(const mha_graph_spec_t &spec) {
    int64_t bs = spec.dims[0];
    int64_t seq_len = spec.dims[1];
    int64_t embed_sz = spec.dims[2];
    int64_t query_sz = embed_sz / spec.head;
    dims_t qkv_shape = {bs, seq_len, spec.head, query_sz};
    dims_t qkv_transpose_shape = {bs, spec.head, seq_len, query_sz};
    dims_t k2_transpose_shape = {bs, spec.head, query_sz, seq_len};
    dims_t qk_matmul_shape = {bs, spec.head, seq_len, seq_len};
    dims_t attention_mask_shape = {bs, 1, 1, seq_len};
    _tensor_desc_info tensor_desc_cfg;

    for (int i = 0; i < TENSOR_DESC_BWD_TYPES_TOTAL; i++) {
        tensor_desc_cfg[i].isavailable = false;
    }

    if (spec.mha_pattern == "v1") {
        //int8 input feeding into dequantize op - only for int8
        tensor_desc_cfg[QINT8_SRC] = {spec.MHA_int8, STRINGIFY(QINT8_SRC),
                spec.mha_inout_dt, spec.dims};
        tensor_desc_cfg[KINT8_SRC] = {spec.MHA_int8, STRINGIFY(KINT8_SRC),
                spec.mha_inout_dt, spec.dims};
        tensor_desc_cfg[VINT8_SRC] = {spec.MHA_int8, STRINGIFY(VINT8_SRC),
                spec.mha_inout_dt, spec.dims};
        //fp32/b1f16 input feeding into reshape op
        tensor_desc_cfg[QSRC] = {true, STRINGIFY(QSRC), spec.mha_dt, spec.dims};
        tensor_desc_cfg[KSRC] = {true, STRINGIFY(KSRC), spec.mha_dt, spec.dims};
        tensor_desc_cfg[VSRC] = {true, STRINGIFY(VSRC), spec.mha_dt, spec.dims};
        //fp32/b1f16 input feeding into transpose op output of reshape
        tensor_desc_cfg[QTSRC]
                = {true, STRINGIFY(QTSRC), spec.mha_dt, qkv_shape};
        tensor_desc_cfg[KTSRC]
                = {true, STRINGIFY(KTSRC), spec.mha_dt, qkv_shape};
        tensor_desc_cfg[VTSRC]
                = {true, STRINGIFY(VTSRC), spec.mha_dt, qkv_shape};
        //fp32/b1f16 input feeding into transpose 2 op output of transpose - only for K
        tensor_desc_cfg[KT2SRC]
                = {true, STRINGIFY(KT2SRC), spec.mha_dt, qkv_transpose_shape};
        //fp32/bf16 output ready for matmul
        tensor_desc_cfg[QDST]
                = {true, STRINGIFY(QDST), spec.mha_dt, qkv_transpose_shape};
        tensor_desc_cfg[KDST]
                = {true, STRINGIFY(KDST), spec.mha_dt, k2_transpose_shape};
        tensor_desc_cfg[VDST]
                = {true, STRINGIFY(VDST), spec.mha_dt, qkv_transpose_shape};
        tensor_desc_cfg[QKVDST]
                = {true, STRINGIFY(QKVDST), spec.mha_dt, spec.dims};
        tensor_desc_cfg[QKVINT8DST] = {spec.MHA_int8, STRINGIFY(QKVINT8DST),
                spec.mha_inout_dt, spec.dims};
    } else if (spec.mha_pattern == "v2") {
        //int8 input feeding into dequantize op - only for int8
        tensor_desc_cfg[QINT8_SRC] = {spec.MHA_int8, STRINGIFY(QINT8_SRC),
                spec.mha_inout_dt, qkv_transpose_shape};
        tensor_desc_cfg[KINT8_SRC] = {spec.MHA_int8, STRINGIFY(KINT8_SRC),
                spec.mha_inout_dt, k2_transpose_shape};
        tensor_desc_cfg[VINT8_SRC] = {spec.MHA_int8, STRINGIFY(VINT8_SRC),
                spec.mha_inout_dt, qkv_transpose_shape};
        //fp32/bf16 output ready for matmul
        tensor_desc_cfg[QDST]
                = {true, STRINGIFY(QDST), spec.mha_dt, qkv_transpose_shape};
        tensor_desc_cfg[KDST]
                = {true, STRINGIFY(KDST), spec.mha_dt, k2_transpose_shape};
        tensor_desc_cfg[VDST]
                = {true, STRINGIFY(VDST), spec.mha_dt, qkv_transpose_shape};
        tensor_desc_cfg[QKVDST]
                = {true, STRINGIFY(QKVDST), spec.mha_dt, qkv_shape};
        tensor_desc_cfg[QKVINT8DST] = {spec.MHA_int8, STRINGIFY(QKVINT8DST),
                spec.mha_inout_dt, qkv_shape};
    } else if (spec.mha_pattern == "v3") {
        //int8 input feeding into dequantize op - only for int8
        tensor_desc_cfg[QINT8_SRC] = {spec.MHA_int8, STRINGIFY(QINT8_SRC),
                spec.mha_inout_dt, qkv_transpose_shape};
        tensor_desc_cfg[KINT8_SRC] = {spec.MHA_int8, STRINGIFY(KINT8_SRC),
                spec.mha_inout_dt, k2_transpose_shape};
        tensor_desc_cfg[VINT8_SRC] = {spec.MHA_int8, STRINGIFY(VINT8_SRC),
                spec.mha_inout_dt, qkv_transpose_shape};
        tensor_desc_cfg[QSRC] = {
                true, STRINGIFY(QSRC), spec.mha_quan_dt, qkv_transpose_shape};
        tensor_desc_cfg[KSRC]
                = {true, STRINGIFY(KSRC), spec.mha_quan_dt, k2_transpose_shape};
        tensor_desc_cfg[VSRC] = {
                true, STRINGIFY(VSRC), spec.mha_quan_dt, qkv_transpose_shape};
        //fp32/bf16 output ready for matmul
        tensor_desc_cfg[QDST]
                = {true, STRINGIFY(QDST), spec.mha_dt, qkv_transpose_shape};
        tensor_desc_cfg[KDST]
                = {true, STRINGIFY(KDST), spec.mha_dt, k2_transpose_shape};
        tensor_desc_cfg[VDST]
                = {true, STRINGIFY(VDST), spec.mha_dt, qkv_transpose_shape};
        tensor_desc_cfg[QKSOFTMAXQUANCAST]
                = {true, STRINGIFY(QKSOFTMAXQUANCAST), spec.mha_quan_dt,
                        qk_matmul_shape};
        tensor_desc_cfg[QKSOFTMAXDEQUANCAST] = {true,
                STRINGIFY(QKSOFTMAXDEQUANCAST), spec.mha_dt, qk_matmul_shape};
        tensor_desc_cfg[QKVCASTDST]
                = {true, STRINGIFY(QKVCASTDST), spec.mha_dt, qkv_shape};
        tensor_desc_cfg[QKVDST]
                = {true, STRINGIFY(QKVDST), spec.mha_quan_dt, qkv_shape};
        tensor_desc_cfg[QKVINT8DST] = {spec.MHA_int8, STRINGIFY(QKVINT8DST),
                spec.mha_inout_dt, qkv_shape};
    } else if (spec.is_fwd_training) {
        tensor_desc_cfg[QDST]
                = {true, STRINGIFY(QDST), spec.mha_dt, qkv_transpose_shape};
        tensor_desc_cfg[KDST]
                = {true, STRINGIFY(KDST), spec.mha_dt, qkv_transpose_shape};
        tensor_desc_cfg[VDST]
                = {true, STRINGIFY(VDST), spec.mha_dt, qkv_transpose_shape};
        tensor_desc_cfg[QKDROPOUT]
                = {true, STRINGIFY(QKDROPOUT), spec.mha_dt, qk_matmul_shape};
        tensor_desc_cfg[QKVDST]
                = {true, STRINGIFY(QKVDST), spec.mha_dt, spec.dims};
    }

    tensor_desc_cfg[QKMATMULDST]
            = {true, STRINGIFY(QKMATMULDST), spec.mha_dt, qk_matmul_shape};
    tensor_desc_cfg[QKDIVSCALE]
            = {true, STRINGIFY(QKDIVSCALE), graph_dt::f32, {1}};
    tensor_desc_cfg[QKDIVDST]
            = {true, STRINGIFY(QKDIVDST), spec.mha_dt, qk_matmul_shape};
    tensor_desc_cfg[QKATTN]
            = {true, STRINGIFY(QKATTN), spec.mha_dt, attention_mask_shape};
    tensor_desc_cfg[QKADD]
            = {true, STRINGIFY(QKADD), spec.mha_dt, qk_matmul_shape};
    tensor_desc_cfg[QKSOFTMAX]
            = {true, STRINGIFY(QKSOFTMAX), spec.mha_dt, qk_matmul_shape};
    tensor_desc_cfg[QKSOFTMAXQUAN] = {spec.MHA_int8, STRINGIFY(QKSOFTMAXQUAN),
            spec.mha_inout_dt, qk_matmul_shape};
    auto dequandt = spec.MHA_int8 ? spec.mha_quan_dt : spec.mha_dt;
    tensor_desc_cfg[QKSOFTMAXDEQUAN] = {spec.MHA_int8 || spec.is_fwd_training,
            STRINGIFY(QKSOFTMAXDEQUAN), dequandt, qk_matmul_shape};
    tensor_desc_cfg[QKVMATMUL]
            = {true, STRINGIFY(QKVMATMUL), spec.mha_dt, qkv_transpose_shape};
    tensor_desc_cfg[QKVTDST]
            = {true, STRINGIFY(QKVTDST), spec.mha_dt, qkv_shape};

    for (int i = 0; i < TENSOR_DESC_FWD_TYPES_TOTAL; i++) {
        if (tensor_desc_cfg[i].isavailable) {
            tensor_descs_.emplace(tensor_desc_cfg[i].tensor_desc_name,
                    tensor_desc_cfg[i].dt, tensor_desc_cfg[i].dims, spec.tag);
            std::string tensor_name = tensor_desc_cfg[i].tensor_desc_name;
            ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
                    = {tensor_name, tensor_descs_[tensor_name], -1, -1};
        }
    }
}

void mha_graph_prb_t::build_tensor_desc_bwd(const mha_graph_spec_t &spec) {
    int64_t bs = spec.dims[0];
    int64_t seq_len = spec.dims[1];
    int64_t embed_sz = spec.dims[2];
    int64_t query_sz = embed_sz / spec.head;
    dims_t qkv_shape = {bs, seq_len, spec.head, query_sz};
    dims_t qkv_transpose_shape = {bs, spec.head, seq_len, query_sz};
    dims_t softmax_sum_shape = {bs, spec.head, seq_len, 1};
    dims_t k2_transpose_shape = {bs, spec.head, query_sz, seq_len};
    dims_t qk_matmul_shape = {bs, spec.head, seq_len, seq_len};
    dims_t attention_mask_shape = {bs, 1, 1, seq_len};
    dims_t grad_wei_shape = {bs, spec.head, seq_len, query_sz};
    _tensor_desc_info tensor_desc_cfg;

    for (int i = 0; i < TENSOR_DESC_BWD_TYPES_TOTAL; i++) {
        tensor_desc_cfg[i].isavailable = false;
    }
    tensor_desc_cfg[QKSOFTMAXDEQUAN]
            = {true, STRINGIFY(QKSOFTMAXDEQUAN), spec.mha_dt, qk_matmul_shape};
    tensor_desc_cfg[QDST]
            = {true, STRINGIFY(QDST), spec.mha_dt, qkv_transpose_shape};
    tensor_desc_cfg[KDST]
            = {true, STRINGIFY(KDST), spec.mha_dt, qkv_transpose_shape};
    tensor_desc_cfg[VDST]
            = {true, STRINGIFY(VDST), spec.mha_dt, qkv_transpose_shape};
    tensor_desc_cfg[QKDROPOUT]
            = {true, STRINGIFY(QKDROPOUT), spec.mha_dt, qk_matmul_shape};
    tensor_desc_cfg[QKSOFTMAX]
            = {true, STRINGIFY(QKSOFTMAX), spec.mha_dt, qk_matmul_shape};
    tensor_desc_cfg[QKDIVSCALE]
            = {true, STRINGIFY(QKDIVSCALE), graph_dt::f32, {1}};
    tensor_desc_cfg[GRADIN] = {true, STRINGIFY(GRADIN), spec.mha_dt, spec.dims};
    tensor_desc_cfg[GRADTIN]
            = {true, STRINGIFY(GRADTIN), spec.mha_dt, qkv_shape};
    tensor_desc_cfg[GRADTOUT]
            = {true, STRINGIFY(GRADTOUT), spec.mha_dt, qkv_transpose_shape};
    tensor_desc_cfg[GRADVWEI]
            = {true, STRINGIFY(GRADVWEI), spec.mha_dt, qkv_transpose_shape};
    tensor_desc_cfg[GRADVDATA]
            = {true, STRINGIFY(GRADVDATA), spec.mha_dt, qk_matmul_shape};
    tensor_desc_cfg[GRADMUL1DATA]
            = {true, STRINGIFY(GRADMUL1DATA), spec.mha_dt, qk_matmul_shape};
    tensor_desc_cfg[GRADMUL2DATA]
            = {true, STRINGIFY(GRADMUL2DATA), spec.mha_dt, qk_matmul_shape};
    tensor_desc_cfg[GRADSOFTMAXSUM]
            = {true, STRINGIFY(GRADSOFTMAXSUM), spec.mha_dt, softmax_sum_shape};
    tensor_desc_cfg[GRADSOFTMAXSUB]
            = {true, STRINGIFY(GRADSOFTMAXSUB), spec.mha_dt, qk_matmul_shape};
    tensor_desc_cfg[GRADMUL3DATA]
            = {true, STRINGIFY(GRADMUL3DATA), spec.mha_dt, qk_matmul_shape};
    tensor_desc_cfg[GRADDIVDATA]
            = {true, STRINGIFY(GRADDIVDATA), spec.mha_dt, qk_matmul_shape};
    tensor_desc_cfg[GRADKWEI]
            = {true, STRINGIFY(GRADKWEI), spec.mha_dt, grad_wei_shape};
    tensor_desc_cfg[GRADQWEI]
            = {true, STRINGIFY(GRADQWEI), spec.mha_dt, grad_wei_shape};

    for (int i = 0; i < TENSOR_DESC_BWD_TYPES_TOTAL; i++) {
        if (tensor_desc_cfg[i].isavailable) {
            tensor_descs_.emplace(tensor_desc_cfg[i].tensor_desc_name,
                    tensor_desc_cfg[i].dt, tensor_desc_cfg[i].dims, spec.tag);
            std::string tensor_name = tensor_desc_cfg[i].tensor_desc_name;
            ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
                    = {tensor_name, tensor_descs_[tensor_name], -1, -1};
        }
    }
}
} // namespace mha
