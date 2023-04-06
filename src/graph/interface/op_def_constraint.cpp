/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "graph/interface/op_def_constraint.hpp"

namespace dnnl {
namespace impl {
namespace graph {

// check function for data_type of BatchNorm.
// only when data is bf16, gamma/beta/mean/var can be bf16.
// If data is bf16, gamma/beta/mean/var can be f32 or bf16.
bool check_bn_data_type(const op_t *n) {
    const logical_tensor_t &src_lt
            = n->get_input_value(0)->get_logical_tensor();
    const logical_tensor_t &aux_lt
            = n->get_input_value(2)->get_logical_tensor();

    if (src_lt.data_type != data_type::bf16
            && aux_lt.data_type == data_type::bf16)
        return false;
    else
        return true;
}

// check function for data_type of LayerNorm.
// only when data is bf16, gamma/beta/mean/var can be bf16.
// If data is bf16, gamma/beta/mean/var can be f32 or bf16.
bool check_ln_data_type(const op_t *n) {
    auto input_values = n->get_input_values();
    auto output_values = n->get_output_values();

    const logical_tensor_t &src_lt = input_values[0]->get_logical_tensor();
    logical_tensor_t aux_lt;
    // check if optional input /output exists
    if (input_values.size() == 1 && output_values.size() == 1) {
        return true;
    } else {
        if (input_values.size() > 2) {
            aux_lt = input_values[2]->get_logical_tensor();
        } else {
            aux_lt = output_values[1]->get_logical_tensor();
        }
    }
    if (src_lt.data_type != data_type::bf16
            && aux_lt.data_type == data_type::bf16)
        return false;
    else
        return true;
}

// check function for data_type of Typecast.
// for TypeCast, input & output should not have the same dtype
bool check_typecast_data_type(const op_t *n) {
    const logical_tensor_t &src_lt
            = n->get_input_value(0)->get_logical_tensor();
    const logical_tensor_t &aux_lt
            = n->get_output_value(0)->get_logical_tensor();

    if (src_lt.data_type == aux_lt.data_type) return false;
    if (src_lt.data_type == data_type::f16
            && aux_lt.data_type == data_type::bf16)
        return false;
    if (src_lt.data_type == data_type::bf16
            && aux_lt.data_type == data_type::f16)
        return false;
    return true;
}

// check function for input_shape of Avgpool backward.
// if input_shape is not specified in inputs,
// it should be specified in attributes.
bool check_avgpool_bwd_input_shape(const op_t *n) {
    const size_t inputs_num = n->num_inputs();
    if (inputs_num == 1) { return n->has_attr(op_attr::src_shape); }
    return true;
}

// check function for output_shape of Convolution backward data.
// if output_shape is not specified in inputs,
// it should be specified in attributes.
bool check_conv_bwd_data_output_shape(const op_t *n) {
    auto inputs_num = n->num_inputs();
    if (inputs_num == 2) { return n->has_attr(op_attr::dst_shape); }
    return true;
}

// check function for weights_shape of Convolution[Transpose] backward weights.
// if weights_shape is not specified in inputs,
// it should be specified in attributes.
bool check_conv_bwd_weights_weights_shape(const op_t *n) {
    auto inputs_num = n->num_inputs();
    if (inputs_num == 2) { return n->has_attr(op_attr::weights_shape); }
    return true;
}

// check function for sizes and scales of Interpolate[Backward].
// for this op, sizes and scales can not be compatible.
bool check_interpolate_sizes_scales(const op_t *n) {
    const size_t sz_sizes = n->has_attr(op_attr::sizes)
            ? n->get_attr<std::vector<int64_t>>(op_attr::sizes).size()
            : 0;
    const size_t sz_scales = n->has_attr(op_attr::scales)
            ? n->get_attr<std::vector<float>>(op_attr::scales).size()
            : 0;
    if ((sz_sizes && sz_scales) || (!sz_sizes && !sz_scales)) { return false; }
    return true;
}

// check function for output number of LayerNorm forward.
// if keep_stats == true, outputs should include mean and variance.
bool check_ln_fwd_outputs_num(const op_t *n) {
    const size_t actual_num = n->num_outputs();
    const bool keep_stats = n->has_attr(op_attr::keep_stats)
            ? n->get_attr<bool>(op_attr::keep_stats)
            : true;
    if (keep_stats) { return actual_num == 3; }
    return true;
}

// check function for output number of LayerNorm backward.
// if use_affine == true, outputs should include mean and variance.
bool check_ln_bwd_use_affine(const op_t *n) {
    const size_t actual_num = n->num_outputs();
    const bool use_affine = n->has_attr(op_attr::use_affine)
            ? n->get_attr<bool>(op_attr::use_affine)
            : true;
    if (use_affine) { return actual_num == 3; }
    return true;
}

// check function foraxes of Reduce.
// including Reduce: L1/L2/Max/Mean/Min/Prod/Sum.
// attribute_axes and input_axes is incompatible.
bool check_reduce_axes(const op_t *n) {
    const bool axes = n->has_attr(op_attr::axes);
    const size_t inputs_num = n->num_inputs();
    const bool input_axes = (inputs_num == 2);
    if ((axes && input_axes) || (!axes && !input_axes)) { return false; }
    return true;
}

// check function for scales and zps of Quantize/Dequantize.
// the number of scales and zps should keep same.
// especially, when qtype == "per-tensor", sz_scales/zps should be 1.
bool check_quant_dequant_scales_zps(const op_t *n) {
    const int64_t sz_scales
            = n->get_attr<std::vector<float>>(op_attr::scales).size();
    const int64_t sz_zps
            = n->get_attr<std::vector<int64_t>>(op_attr::zps).size();
    if (sz_scales != sz_zps) { return false; }

    // qtype is not a required attribute.
    const auto qtype = n->has_attr(op_attr::qtype)
            ? n->get_attr<std::string>(op_attr::qtype)
            : "per_tensor";

    if (qtype == "per_tensor") { return sz_scales == 1; }
    return true;
}

// check function for scales and zps of DynamicQuantize/DynamicDequantize.
// the number of scales and zps should keep same.
// especially, when qtype == "per-tensor", sz_scales/zps should be 1.
// unlike Quantize/Dequantize, scales and zps are inputs here.
bool check_dyn_quant_dequant_scales_zps(const op_t *n) {
    const int64_t inputs_num = n->num_inputs();
    const int64_t sz_scales
            = n->get_input_value(1)->get_logical_tensor().dims[0];

    // in case of not setting value for scales
    if (sz_scales == DNNL_GRAPH_UNKNOWN_DIM) { return true; }

    // qtype is not a required attribute.
    const auto qtype = n->has_attr(op_attr::qtype)
            ? n->get_attr<std::string>(op_attr::qtype)
            : "per_tensor";
    // zps is not a required input.
    if (inputs_num == 2) {
        return qtype == "per_tensor" ? sz_scales == 1 : true;
    } else {
        const int64_t sz_zps
                = n->get_input_value(2)->get_logical_tensor().dims[0];

        // in case of not setting value for zps
        if (sz_zps == DNNL_GRAPH_UNKNOWN_DIM) { return true; }

        if (sz_scales != sz_zps) { return false; }
        return qtype == "per_tensor" ? sz_scales == 1 : true;
    }
    return true;
}

} // namespace graph
} // namespace impl
} // namespace dnnl
