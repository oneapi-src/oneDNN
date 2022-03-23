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
#include "mlp.hpp"
namespace mlp {

void mlp_graph_prb_t::add_quan_dequan_op(const mlp_graph_spec_t &spec,
        const std::string src, const std::string dst, std::vector<float> scales,
        std::vector<int64_t> zps, bool isquanop) {
    if (!spec.is_mlp_int8) { return; }
    using op = dnnl::graph::op;
    size_t op_id = ops_.size();
    auto op_kind = (isquanop) ? op::kind::Quantize : op::kind::Dequantize;
    std::string op_str = (isquanop) ? dst + "_quantize" : dst + "_dequantize";
    ops_.emplace_back(op(op_id, op_kind, {tensor_descs_[src]},
            {tensor_descs_[dst]}, op_str));
    ops_[op_id].set_attr<std::string>(
            "qtype", convert_attr_policy(spec.attr.oscale.policy));
    ops_[op_id].set_attr<std::vector<int64_t>>("zps", zps);
    ops_[op_id].set_attr<std::vector<float>>("scales", scales);
    ops_[op_id].set_attr("axis", static_cast<int64_t>(0));
}

void mlp_graph_prb_t::add_matmul_op(
        const mlp_graph_spec_t &spec, int layer_num, bool is_fwd_pass) {
    using op = dnnl::graph::op;
    std::string layer_str = std::to_string(layer_num);
    if (is_fwd_pass) {
        std::vector<dnnl::graph::logical_tensor> lt_inputs {
                tensor_descs_[STRINGIFY(DATA_) + layer_str],
                tensor_descs_[STRINGIFY(WEI_) + layer_str]};
        std::vector<dnnl::graph::logical_tensor> lt_outputs {
                tensor_descs_[STRINGIFY(MATMUL_) + layer_str]};
        if (spec.has_bias)
            lt_inputs.push_back(tensor_descs_[STRINGIFY(BIA_) + layer_str]);
        ops_.emplace_back(op(ops_.size(), op::kind::MatMul, lt_inputs,
                lt_outputs, "matmul_" + layer_str));
    } else {
        auto op_id = ops_.size();
        std::vector<dnnl::graph::logical_tensor> lt_data_inputs {
                tensor_descs_[STRINGIFY(ACTFUNC_GRAD_) + layer_str],
                tensor_descs_[STRINGIFY(WEI_) + layer_str]};
        std::vector<dnnl::graph::logical_tensor> lt_data_outputs {
                tensor_descs_[STRINGIFY(DATA_GRAD_) + layer_str]};
        ops_.emplace_back(op(op_id, op::kind::MatMul, lt_data_inputs,
                lt_data_outputs, "matmul_bp_data_" + layer_str));
        ops_[op_id].set_attr("transpose_b", true);
        op_id = ops_.size();
        std::vector<dnnl::graph::logical_tensor> lt_wei_inputs {
                tensor_descs_[STRINGIFY(DATA_) + layer_str],
                tensor_descs_[STRINGIFY(ACTFUNC_GRAD_) + layer_str]};
        std::vector<dnnl::graph::logical_tensor> lt_wei_outputs {
                tensor_descs_[STRINGIFY(WEI_GRAD_) + layer_str]};
        ops_.emplace_back(op(op_id, op::kind::MatMul, lt_wei_inputs,
                lt_wei_outputs, "matmul_bp_wei_" + layer_str));
        ops_[op_id].set_attr("transpose_a", true);
    }
}

void mlp_graph_prb_t::add_actfunc_op(
        const mlp_graph_spec_t &spec, int layer_num, bool is_fwd_pass) {
    using op = dnnl::graph::op;
    std::string layer_str = std::to_string(layer_num);
    if (is_fwd_pass) {
        std::string out_tensor = (spec.is_mlp_int8)
                ? STRINGIFY(DATA_OUT_) + layer_str
                : STRINGIFY(DATA_) + std::to_string(layer_num + 1);
        ops_.emplace_back(op(ops_.size(), spec.activation_func[layer_num],
                {tensor_descs_[STRINGIFY(MATMUL_) + layer_str]},
                {tensor_descs_[out_tensor]}, "activation_func_" + layer_str));
    } else {
        auto get_backward_op = [](dnnl::graph::op::kind opkind) {
            dnnl::graph::op::kind backprop_opkind;
            if (opkind == dnnl::graph::op::kind::ReLU) {
                backprop_opkind = dnnl::graph::op::kind::ReLUBackprop;
            } else if (opkind == dnnl::graph::op::kind::Sigmoid) {
                backprop_opkind = dnnl::graph::op::kind::SigmoidBackprop;
            } else {
                //do not reach here
                assert(1);
                backprop_opkind = dnnl::graph::op::kind::LastSymbol;
            }
            return backprop_opkind;
        };

        std::string in_grad_tensor = (spec.use_dst)
                ? STRINGIFY(DATA_) + std::to_string(layer_num + 1)
                : STRINGIFY(MATMUL_) + layer_str;
        auto op_id = ops_.size();
        ops_.emplace_back(
                op(op_id, get_backward_op(spec.activation_func[layer_num]),
                        {tensor_descs_[in_grad_tensor],
                                tensor_descs_[STRINGIFY(DATA_GRAD_)
                                        + std::to_string(layer_num + 1)]},
                        {tensor_descs_[STRINGIFY(ACTFUNC_GRAD_) + layer_str]},
                        "activation_bp_func_" + layer_str));
        ops_[op_id].set_attr("use_dst", spec.use_dst);
    }
}

void mlp_graph_prb_t::add_statictranspose_op(
        const mlp_graph_spec_t &spec, int layer_num) {
    using op = dnnl::graph::op;
    std::string layer_str = std::to_string(layer_num);
    size_t op_id = ops_.size();
    ops_.emplace_back(op(op_id, op::kind::StaticTranspose,
            {tensor_descs_[STRINGIFY(DATA_) + layer_str]},
            {tensor_descs_[STRINGIFY(DATA_TGRAD_) + layer_str]},
            "statictranspose_data_" + layer_str));
    ops_[op_id].set_attr("order", std::vector<int64_t> {1, 0});
    op_id = ops_.size();
    ops_.emplace_back(op(op_id, op::kind::StaticTranspose,
            {tensor_descs_[STRINGIFY(WEI_) + layer_str]},
            {tensor_descs_[STRINGIFY(WEI_TGRAD_) + layer_str]},
            "statictranspose_wei_" + layer_str));
    ops_[op_id].set_attr("order", std::vector<int64_t> {1, 0});
}

void mlp_graph_prb_t::add_reducesum_op(
        const mlp_graph_spec_t &spec, int layer_num) {
    if (!spec.has_bias || spec.batch_sz == 1) return;
    using op = dnnl::graph::op;
    std::string layer_str = std::to_string(layer_num);
    auto op_id = ops_.size();
    ops_.emplace_back(op(op_id, op::kind::ReduceSum,
            {tensor_descs_[STRINGIFY(ACTFUNC_GRAD_) + layer_str]},
            {tensor_descs_[STRINGIFY(BIA_GRAD_) + layer_str]},
            "ReduceSum" + layer_str));
    ops_[op_id].set_attr("axes", std::vector<int64_t> {0});
}

void mlp_graph_prb_t::add_end_op(const mlp_graph_spec_t &spec, int layer_num) {
    std::string tensor_name = (spec.use_dst)
            ? STRINGIFY(DATA_) + std::to_string(layer_num + 1)
            : STRINGIFY(MATMUL_) + std::to_string(layer_num);

    using op = dnnl::graph::op;
    ops_.emplace_back(op(ops_.size(), op::kind::End,
            {tensor_descs_[tensor_name]}, {}, "end_" + tensor_name));
}

void mlp_graph_prb_t::build_tensor_desc_fwd(const mlp_graph_spec_t &spec) {
    std::string tensor_name;
    for (int i = 0; i < spec.num_hidden_layers; i++) {
        if (spec.is_mlp_int8) {
            tensor_name = STRINGIFY(DATA_INT8_) + std::to_string(i);
            tensor_descs_.emplace(tensor_name, spec.mlp_src_dt,
                    spec.layer_dims[i], spec.raw_data_tag);
            ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
                    = {tensor_descs_[tensor_name], SRC, -1, -1};
            desc_ltid_lut[tensor_name] = tensor_descs_[tensor_name].get_id();
            tensor_name = STRINGIFY(WEI_INT8_) + std::to_string(i);
            tensor_descs_.emplace(tensor_name, spec.mlp_wei_dt,
                    spec.weight_dims[i], lt::strided,
                    tensor_descs_t::property_type::constant);
            ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
                    = {tensor_descs_[tensor_name], WEI, -1, -1};
            desc_ltid_lut[tensor_name] = tensor_descs_[tensor_name].get_id();
            tensor_name = STRINGIFY(DATA_) + std::to_string(i);
            tensor_descs_.emplace(tensor_name, spec.mlp_layer_dt,
                    spec.layer_dims[i], spec.raw_data_tag);
            ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
                    = {tensor_descs_[tensor_name], -1, -1, -1};
            desc_ltid_lut[tensor_name] = tensor_descs_[tensor_name].get_id();
            tensor_name = STRINGIFY(WEI_) + std::to_string(i);
            tensor_descs_.emplace(tensor_name, spec.mlp_layer_dt,
                    spec.weight_dims[i], spec.raw_wei_tag);
            ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
                    = {tensor_descs_[tensor_name], -1, -1, -1};
            desc_ltid_lut[tensor_name] = tensor_descs_[tensor_name].get_id();
        } else {
            tensor_name = STRINGIFY(DATA_) + std::to_string(i);
            tensor_descs_.emplace(tensor_name, spec.mlp_layer_dt,
                    spec.layer_dims[i], spec.raw_data_tag);
            ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
                    = {tensor_descs_[tensor_name], SRC, -1, -1};
            desc_ltid_lut[tensor_name] = tensor_descs_[tensor_name].get_id();

            tensor_name = STRINGIFY(WEI_) + std::to_string(i);
            if (spec.is_fwd_inference) {
                tensor_descs_.emplace(tensor_name, spec.mlp_layer_dt,
                        spec.weight_dims[i], lt::strided,
                        tensor_descs_t::property_type::constant);
            } else {
                tensor_descs_.emplace(tensor_name, spec.mlp_layer_dt,
                        spec.weight_dims[i], spec.raw_wei_tag);
            }
            ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
                    = {tensor_descs_[tensor_name], WEI, -1, -1};
            desc_ltid_lut[tensor_name] = tensor_descs_[tensor_name].get_id();
        }

        if (spec.has_bias) {
            tensor_name = STRINGIFY(BIA_) + std::to_string(i);
            if (spec.is_fwd_inference) {
                tensor_descs_.emplace(tensor_name, spec.mlp_bias_dt,
                        spec.bias_dims[i], lt::strided,
                        tensor_descs_t::property_type::constant);
            } else {
                tensor_descs_.emplace(tensor_name, spec.mlp_bias_dt,
                        spec.bias_dims[i], std::string(tag::x));
            }
            ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
                    = {tensor_descs_[tensor_name], BIA, -1, -1};
            desc_ltid_lut[tensor_name] = tensor_descs_[tensor_name].get_id();
        }

        tensor_name = STRINGIFY(MATMUL_) + std::to_string(i);
        tensor_descs_.emplace(tensor_name, spec.mlp_layer_dt,
                spec.layer_dims[i + 1], spec.raw_data_tag);
        ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
                = {tensor_descs_[tensor_name], DST, -1, -1};
        desc_ltid_lut[tensor_name] = tensor_descs_[tensor_name].get_id();

        if (spec.is_mlp_int8) {
            tensor_name = STRINGIFY(DATA_OUT_) + std::to_string(i);
            tensor_descs_.emplace(tensor_name, spec.mlp_layer_dt,
                    spec.layer_dims[i + 1], spec.raw_data_tag);
            ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
                    = {tensor_descs_[tensor_name], -1, -1, -1};
            desc_ltid_lut[tensor_name] = tensor_descs_[tensor_name].get_id();
        }
    }

    std::string out_tensor
            = (spec.is_mlp_int8) ? STRINGIFY(DATA_INT8_) : STRINGIFY(DATA_);
    tensor_name = out_tensor + std::to_string(spec.num_hidden_layers);
    tensor_descs_.emplace(tensor_name, spec.mlp_src_dt, spec.layer_dims.back(),
            spec.raw_data_tag);
    ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
            = {tensor_descs_[tensor_name], DST, -1, -1};
    desc_ltid_lut[tensor_name] = tensor_descs_[tensor_name].get_id();
}

void mlp_graph_prb_t::build_tensor_desc_bwd(const mlp_graph_spec_t &spec) {
    std::string tensor_name;
    tensor_name
            = STRINGIFY(DATA_GRAD_) + std::to_string(spec.num_hidden_layers);
    tensor_descs_.emplace(tensor_name, spec.mlp_src_dt, spec.layer_dims.back(),
            spec.raw_data_tag);
    ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
            = {tensor_descs_[tensor_name], SRC, -1, -1};
    desc_ltid_lut[tensor_name] = tensor_descs_[tensor_name].get_id();

    for (int i = spec.num_hidden_layers - 1; i >= 0; i--) {
        tensor_name = STRINGIFY(ACTFUNC_GRAD_) + std::to_string(i);
        tensor_descs_.emplace(tensor_name, spec.mlp_src_dt,
                spec.layer_dims[i + 1], spec.raw_data_tag);
        ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
                = {tensor_descs_[tensor_name], DST, -1, -1};
        desc_ltid_lut[tensor_name] = tensor_descs_[tensor_name].get_id();
        if (spec.use_static_transpose) {
            tensor_name = STRINGIFY(DATA_TGRAD_) + std::to_string(i);
            tensor_descs_.emplace(tensor_name, spec.mlp_src_dt,
                    {spec.layer_dims[i][1], spec.layer_dims[i][0]},
                    spec.raw_data_tag);
            ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
                    = {tensor_descs_[tensor_name], -1, -1, -1};
            desc_ltid_lut[tensor_name] = tensor_descs_[tensor_name].get_id();
            tensor_name = STRINGIFY(WEI_TGRAD_) + std::to_string(i);
            tensor_descs_.emplace(tensor_name, spec.mlp_wei_dt,
                    {spec.weight_dims[i][1], spec.weight_dims[i][0]},
                    spec.raw_data_tag);
            ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
                    = {tensor_descs_[tensor_name], -1, -1, -1};
            desc_ltid_lut[tensor_name] = tensor_descs_[tensor_name].get_id();
        }
        tensor_name = STRINGIFY(DATA_GRAD_) + std::to_string(i);
        tensor_descs_.emplace(tensor_name, spec.mlp_layer_dt,
                spec.layer_dims[i], spec.raw_data_tag);
        ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
                = {tensor_descs_[tensor_name], SRC, -1, -1};
        desc_ltid_lut[tensor_name] = tensor_descs_[tensor_name].get_id();
        tensor_name = STRINGIFY(WEI_GRAD_) + std::to_string(i);
        tensor_descs_.emplace(tensor_name, spec.mlp_layer_dt,
                spec.weight_dims[i], spec.raw_wei_tag);
        ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
                = {tensor_descs_[tensor_name], WEI, -1, -1};
        desc_ltid_lut[tensor_name] = tensor_descs_[tensor_name].get_id();
        if (spec.has_bias) {
            tensor_name = STRINGIFY(BIA_GRAD_) + std::to_string(i);
            tensor_descs_.emplace(tensor_name, spec.mlp_bias_dt,
                    spec.bias_dims[i], std::string(tag::x));
            ltid_desc_lut[tensor_descs_[tensor_name].get_id()]
                    = {tensor_descs_[tensor_name], BIA, -1, -1};
            desc_ltid_lut[tensor_name] = tensor_descs_[tensor_name].get_id();
        }
    }
}
} // namespace mlp
