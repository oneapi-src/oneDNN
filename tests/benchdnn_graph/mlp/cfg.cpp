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
#include "mlp.hpp"
namespace mlp {

void mlp_graph_prb_t::addQuanDequanOp(const mlp_graph_spec_t &spec,
        const std::string src, const std::string dst, std::vector<float> scales,
        std::vector<int64_t> zps, bool isQuanOp) {
    if (!spec.is_mlp_int8) { return; }
    using op = dnnl::graph::op;
    size_t op_id = ops_.size();
    auto op_kind = (isQuanOp) ? op::kind::Quantize : op::kind::Dequantize;
    std::string op_str = (isQuanOp) ? dst + "_quantize" : dst + "_dequantize";
    ops_.emplace_back(op(op_id, op_kind, {tensor_descs_[src]},
            {tensor_descs_[dst]}, op_str));
    ops_[op_id].set_attr<std::string>(
            "qtype", convert_attr_policy(spec.attr.oscale.policy));
    ops_[op_id].set_attr<std::vector<int64_t>>("zps", zps);
    ops_[op_id].set_attr<std::vector<float>>("scales", scales);
    ops_[op_id].set_attr("axis", static_cast<int64_t>(0));
}

void mlp_graph_prb_t::addMatmulActFuncOp(
        const mlp_graph_spec_t &spec, int layer_num) {
    using op = dnnl::graph::op;
    std::string layer_str = std::to_string(layer_num);
    std::vector<dnnl::graph::logical_tensor> lt_inputs {
            tensor_descs_[STRINGIFY(DATA_) + layer_str],
            tensor_descs_[STRINGIFY(WEI_) + layer_str]};
    std::vector<dnnl::graph::logical_tensor> lt_outputs {
            tensor_descs_[STRINGIFY(MATMUL_) + layer_str]};
    if (spec.has_bias)
        lt_inputs.push_back(tensor_descs_[STRINGIFY(BIA_) + layer_str]);
    ops_.emplace_back(op(ops_.size(), op::kind::MatMul, lt_inputs, lt_outputs,
            "matmul_" + layer_str));

    std::string out_tensor = (spec.is_mlp_int8)
            ? STRINGIFY(DATA_OUT_) + layer_str
            : STRINGIFY(DATA_) + std::to_string(layer_num + 1);
    ops_.emplace_back(op(ops_.size(), spec.activation_func[layer_num],
            {tensor_descs_[STRINGIFY(MATMUL_) + layer_str]},
            {tensor_descs_[out_tensor]}, "activation_func_" + layer_str));
}

void mlp_graph_prb_t::build_tensor_desc(const mlp_graph_spec_t &spec) {

    std::string tensor_name;
    for (int i = 0; i < spec.num_hidden_layers; i++) {
        if (spec.is_mlp_int8) {
            tensor_name = STRINGIFY(DATA_INT8_) + std::to_string(i);
            tensor_descs_.emplace(tensor_name, spec.mlp_src_dt,
                    spec.layer_dims[i], spec.raw_data_tag);
            tensor_name = STRINGIFY(WEI_INT8_) + std::to_string(i);
            tensor_descs_.emplace(tensor_name, spec.mlp_wei_dt,
                    spec.weight_dims[i], lt::strided,
                    tensor_descs_t::property_type::constant);
            tensor_name = STRINGIFY(DATA_) + std::to_string(i);
            tensor_descs_.emplace(tensor_name, spec.mlp_layer_dt,
                    spec.layer_dims[i], spec.raw_data_tag);
            tensor_name = STRINGIFY(WEI_) + std::to_string(i);
            tensor_descs_.emplace(tensor_name, spec.mlp_layer_dt,
                    spec.weight_dims[i], spec.raw_wei_tag);
        } else {
            tensor_name = STRINGIFY(DATA_) + std::to_string(i);
            tensor_descs_.emplace(tensor_name, spec.mlp_layer_dt,
                    spec.layer_dims[i], spec.raw_data_tag);
            tensor_name = STRINGIFY(WEI_) + std::to_string(i);
            tensor_descs_.emplace(tensor_name, spec.mlp_layer_dt,
                    spec.weight_dims[i], lt::strided,
                    tensor_descs_t::property_type::constant);
        }

        if (spec.has_bias) {
            tensor_name = STRINGIFY(BIA_) + std::to_string(i);
            tensor_descs_.emplace(tensor_name, spec.mlp_bias_dt,
                    spec.bias_dims[i], lt::strided,
                    tensor_descs_t::property_type::constant);
        }

        tensor_name = STRINGIFY(MATMUL_) + std::to_string(i);
        tensor_descs_.emplace(tensor_name, spec.mlp_layer_dt,
                spec.layer_dims[i + 1], spec.raw_data_tag);

        if (spec.is_mlp_int8) {
            tensor_name = STRINGIFY(DATA_OUT_) + std::to_string(i);
            tensor_descs_.emplace(tensor_name, spec.mlp_layer_dt,
                    spec.layer_dims[i + 1], spec.raw_data_tag);
        }
    }

    std::string out_tensor
            = (spec.is_mlp_int8) ? STRINGIFY(DATA_INT8_) : STRINGIFY(DATA_);
    tensor_descs_.emplace(out_tensor + std::to_string(spec.num_hidden_layers),
            spec.mlp_src_dt, spec.layer_dims.back(), spec.raw_data_tag);
}
} // namespace mlp