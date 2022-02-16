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
#include "mha.hpp"

namespace mha {

void mha_graph_prb_t::addDequanOp(const mha_graph_spec_t &spec,
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

void mha_graph_prb_t::addStaticReshapeOp(const mha_graph_spec_t &spec,
        const std::string src, const std::string dst, const dims_t shape) {
    using op = dnnl::graph::op;
    size_t op_id = ops_.size();
    ops_.emplace_back(op(op_id, op::kind::StaticReshape, {tensor_descs_[src]},
            {tensor_descs_[dst]}, dst + "_reshape"));
    ops_[op_id].set_attr("shape", shape);
    ops_[op_id].set_attr("special_zero", false);
}

void mha_graph_prb_t::addStaticTransposeOp(const mha_graph_spec_t &spec,
        const std::string src, const std::string dst, dims_t axis) {
    using op = dnnl::graph::op;
    size_t op_id = ops_.size();
    ops_.emplace_back(op(op_id, op::kind::StaticTranspose, {tensor_descs_[src]},
            {tensor_descs_[dst]}, dst + "_transpose"));
    ops_[op_id].set_attr("order", axis);
}

void mha_graph_prb_t::addQuanOp(const mha_graph_spec_t &spec,
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
void mha_graph_prb_t::addTypecastOp(
        const std::string src, const std::string dst) {
    using op = dnnl::graph::op;
    ops_.emplace_back(op(ops_.size(), op::kind::TypeCast, {tensor_descs_[src]},
            {tensor_descs_[dst]}, dst + "_typecast"));
}

void mha_graph_prb_t::addReorderOp(
        const std::string src, const std::string dst) {
    using op = dnnl::graph::op;
    ops_.emplace_back(op(ops_.size(), op::kind::Reorder, {tensor_descs_[src]},
            {tensor_descs_[dst]}, dst + "_reorder"));
}

void mha_graph_prb_t::build_tensor_desc(const mha_graph_spec_t &spec) {
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
        tensor_desc_cfg[QKSOFTMAXQUANCAST].isavailable = false;
        tensor_desc_cfg[QKSOFTMAXDEQUANCAST].isavailable = false;
        tensor_desc_cfg[QKVDST]
                = {true, STRINGIFY(QKVDST), spec.mha_dt, spec.dims};
        tensor_desc_cfg[QKVCASTDST].isavailable = false;
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
        tensor_desc_cfg[QSRC].isavailable = false;
        tensor_desc_cfg[KSRC].isavailable = false;
        tensor_desc_cfg[VSRC].isavailable = false;
        tensor_desc_cfg[QTSRC].isavailable = false;
        tensor_desc_cfg[KTSRC].isavailable = false;
        tensor_desc_cfg[KT2SRC].isavailable = false;
        tensor_desc_cfg[VTSRC].isavailable = false;
        //fp32/bf16 output ready for matmul
        tensor_desc_cfg[QDST]
                = {true, STRINGIFY(QDST), spec.mha_dt, qkv_transpose_shape};
        tensor_desc_cfg[KDST]
                = {true, STRINGIFY(KDST), spec.mha_dt, k2_transpose_shape};
        tensor_desc_cfg[VDST]
                = {true, STRINGIFY(VDST), spec.mha_dt, qkv_transpose_shape};
        tensor_desc_cfg[QKSOFTMAXQUANCAST].isavailable = false;
        tensor_desc_cfg[QKSOFTMAXDEQUANCAST].isavailable = false;
        tensor_desc_cfg[QKVDST]
                = {true, STRINGIFY(QKVDST), spec.mha_dt, qkv_shape};
        tensor_desc_cfg[QKVCASTDST].isavailable = false;
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
        tensor_desc_cfg[QTSRC].isavailable = false;
        tensor_desc_cfg[KTSRC].isavailable = false;
        tensor_desc_cfg[KT2SRC].isavailable = false;
        tensor_desc_cfg[VTSRC].isavailable = false;
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
    tensor_desc_cfg[QKSOFTMAXDEQUAN] = {spec.MHA_int8,
            STRINGIFY(QKSOFTMAXDEQUAN), spec.mha_quan_dt, qk_matmul_shape};
    tensor_desc_cfg[QKVMATMUL]
            = {true, STRINGIFY(QKVMATMUL), spec.mha_dt, qkv_transpose_shape};
    tensor_desc_cfg[QKVTDST]
            = {true, STRINGIFY(QKVTDST), spec.mha_dt, qkv_shape};

    for (int i = 0; i < TENSOR_DESC_TYPES_TOTAL; i++) {
        if (tensor_desc_cfg[i].isavailable) {
            tensor_descs_.emplace(tensor_desc_cfg[i].tensor_desc_name,
                    tensor_desc_cfg[i].dt, tensor_desc_cfg[i].dims, spec.tag);
        }
    }
}

} // namespace mha
