/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include "common/verbose.hpp"

#include "graph/interface/op_def_constraint.hpp"

#define VCHECK_SHAPE_INFER(cond, msg, ...) \
    VCONDCHECK(graph, create, check, add_op, (cond), false, msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace graph {

// check function for padding value of Conv, Convtranspose, Pooling, etc. Both
// pads_begin and pads_end should be a s64 list containing non-negative values.
bool check_pads(const op_t *n) {
    auto hasNegative = [](const dims &pads) {
        return std::any_of(pads.begin(), pads.end(),
                [](int element) { return element < 0; });
    };
    const dims pads_begin = n->get_attr<dims>(op_attr::pads_begin);
    VCHECK_SHAPE_INFER(!hasNegative(pads_begin),
            "%s, pads_begin should be a s64 list containing non-negative "
            "values",
            op_t::kind2str(n->get_kind()).c_str());
    const dims pads_end = n->get_attr<dims>(op_attr::pads_end);
    VCHECK_SHAPE_INFER(!hasNegative(pads_end),
            "%s, pads_end should be a s64 list containing non-negative "
            "values",
            op_t::kind2str(n->get_kind()).c_str());

    return true;
}

// check function for pool dilations.
// dilations size should be same as kernel size.
bool check_maxpool_dilations(const op_t *n) {
    const dims dilations = n->get_attr<dims>(op_attr::dilations);
    const dims kernel = n->get_attr<dims>(op_attr::kernel);
    const size_t dilations_size = dilations.size();
    const size_t kernel_size = kernel.size();

    // default dilations is vector(12,1) if user not set
    if ((dilations_size == DNNL_MAX_NDIMS) && (dilations_size != kernel_size)) {
        bool allOnes = std::all_of(dilations.begin(), dilations.end(),
                [](dim_t element) { return element == 1; });
        if (allOnes) return true;
    }

    VCHECK_SHAPE_INFER(dilations_size == kernel_size,
            "%s, dilations size should be same as kernel_size",
            op_t::kind2str(n->get_kind()).c_str());

    return true;
}

// check function for data_type of BatchNorm.
// only when data is bf16, gamma/beta/mean/var can be bf16.
// If data is bf16, gamma/beta/mean/var can be f32 or bf16.
bool check_bn_data_type(const op_t *n) {
    const logical_tensor_t &src_lt
            = n->get_input_value(0)->get_logical_tensor();
    const logical_tensor_t &aux_lt
            = n->get_input_value(2)->get_logical_tensor();

    VCHECK_SHAPE_INFER(!(src_lt.data_type != data_type::bf16
                               && aux_lt.data_type == data_type::bf16),
            "%s, given data type %s v.s. expected data type bf16",
            op_t::kind2str(n->get_kind()).c_str(),
            dnnl_dt2str(src_lt.data_type));
    return true;
}

// For MatMul, it's required that src and wei have the same data type. When
// src/wei is xf16, dst can be f32 or xf16 (the same type as src/wei). We can
// disable this check to allow f32f32xf16 when there is a request.
bool check_matmul_dtype(const op_t *mm) {
    const auto inputs = mm->get_input_values();
    const auto outputs = mm->get_output_values();

    const logical_tensor_t &src = inputs[0]->get_logical_tensor();
    const logical_tensor_t &dst = outputs[0]->get_logical_tensor();
    if (src.data_type != dst.data_type) {
        if (dst.data_type != data_type::f32) {
            VCHECK_SHAPE_INFER(false, "%s, %s src + %s dst is not supported",
                    op_t::kind2str(mm->get_kind()).c_str(),
                    dnnl_dt2str(src.data_type), dnnl_dt2str(dst.data_type));
        }
    }

    return true;
}

// For SoftMax, if the src is f32, dst can be xf16. Otherwise, src and dst
// should have the same data type.
bool check_softmax_dtype(const op_t *n) {
    const auto inputs = n->get_input_values();
    const auto outputs = n->get_output_values();

    const logical_tensor_t &src = inputs[0]->get_logical_tensor();
    const logical_tensor_t &dst = outputs[0]->get_logical_tensor();
    if (src.data_type != dst.data_type) {
        if (src.data_type != data_type::f32) {
            VCHECK_SHAPE_INFER(false, "%s, %s src + %s dst is not supported",
                    op_t::kind2str(n->get_kind()).c_str(),
                    dnnl_dt2str(src.data_type), dnnl_dt2str(dst.data_type));
        }
    }

    return true;
}

// For binary operations (Add/Subtract/Multiply/Divide):
// - if src_0 and src_1 have different data types, one of them should be f32.
// - if src_0 and src_1 have different data types, dst should be f32.
bool check_binary_dtype(const op_t *n) {
    const auto inputs = n->get_input_values();
    const auto outputs = n->get_output_values();

    const auto &dt_0 = inputs[0]->get_logical_tensor().data_type;
    const auto &dt_1 = inputs[1]->get_logical_tensor().data_type;
    const auto &dt_2 = outputs[0]->get_logical_tensor().data_type;
    if (dt_0 != dt_1) {
        if ((dt_0 != data_type::f32 && dt_1 != data_type::f32)
                || dt_2 != data_type::f32) {
            VCHECK_SHAPE_INFER(false,
                    "%s, %s src_0 %s src_1 %s dst is not supported",
                    op_t::kind2str(n->get_kind()).c_str(), dnnl_dt2str(dt_0),
                    dnnl_dt2str(dt_1), dnnl_dt2str(dt_2));
        }
    }

    if (dt_2 != data_type::f32) {
        if (dt_0 != dt_2 || dt_1 != dt_2) {
            VCHECK_SHAPE_INFER(false,
                    "%s, %s src_0 %s src_1 %s dst is not supported",
                    op_t::kind2str(n->get_kind()).c_str(), dnnl_dt2str(dt_0),
                    dnnl_dt2str(dt_1), dnnl_dt2str(dt_2));
        }
    }

    return true;
}

// check function for data_type of LayerNorm and GroupNorm.
// only when data is bf16, gamma/beta/mean/var can be bf16.
// If data is bf16, gamma/beta/mean/var can be f32 or bf16.
bool check_ln_gn_data_type(const op_t *n) {
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

    VCHECK_SHAPE_INFER(!(src_lt.data_type != data_type::bf16
                               && aux_lt.data_type == data_type::bf16),
            "%s, given data type %s v.s. expected data type bf16.",
            op_t::kind2str(n->get_kind()).c_str(),
            dnnl_dt2str(src_lt.data_type));
    return true;
}

// check function for data_type of Typecast.
// for TypeCast, input & output should not have the same dtype
bool check_typecast_data_type(const op_t *n) {
    const logical_tensor_t &src_lt
            = n->get_input_value(0)->get_logical_tensor();
    const logical_tensor_t &aux_lt
            = n->get_output_value(0)->get_logical_tensor();

    const auto is_f16_and_bf16_tc
            = (src_lt.data_type == data_type::bf16
                      && aux_lt.data_type == data_type::f16)
            || (src_lt.data_type == data_type::f16
                    && aux_lt.data_type == data_type::bf16);

    VCHECK_SHAPE_INFER(src_lt.data_type != aux_lt.data_type,
            "%s, input and output should not have the same data type.",
            op_t::kind2str(n->get_kind()).c_str());
    VCHECK_SHAPE_INFER((!is_f16_and_bf16_tc),
            "%s, typecast does not support conversion between bf16 and f16.",
            op_t::kind2str(n->get_kind()).c_str());
    return true;
}

// check function for src_shape of Avgpool backward.
// if src_shape is not specified in inputs,
// it should be specified in attributes.
bool check_avgpool_bwd_input_shape(const op_t *n) {
    const size_t inputs_num = n->num_inputs();
    if (inputs_num == 1) {
        VCHECK_SHAPE_INFER((n->has_attr(op_attr::src_shape)),
                "%s, src_shape should be specified in attributes if it's not "
                "given in inputs.",
                op_t::kind2str(n->get_kind()).c_str());
    }

    return true;
}

// check function for dst_shape of Convolution backward data.
// if dst_shape is not specified in inputs,
// it should be specified in attributes.
bool check_conv_bwd_data_output_shape(const op_t *n) {
    auto inputs_num = n->num_inputs();
    if (inputs_num == 2) {
        VCHECK_SHAPE_INFER((n->has_attr(op_attr::dst_shape)),
                "%s, dst_shape should be specified in attributes if it's not "
                "given in inputs.",
                op_t::kind2str(n->get_kind()).c_str());
    }
    return true;
}

// check function for weights_shape of Convolution[Transpose] backward weights.
// if weights_shape is not specified in inputs,
// it should be specified in attributes.
bool check_conv_bwd_weights_weights_shape(const op_t *n) {
    auto inputs_num = n->num_inputs();
    if (inputs_num == 2) {
        VCHECK_SHAPE_INFER((n->has_attr(op_attr::weights_shape)),
                "%s, weights_shape should be specified in attributes if it's "
                "not given in inputs.",
                op_t::kind2str(n->get_kind()).c_str());
    }

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
    const auto sizes_or_scales
            = ((!sz_sizes && sz_scales) || (sz_sizes && !sz_scales));
    VCHECK_SHAPE_INFER(sizes_or_scales,
            "%s, exactly one of the sizes and scales should be provided.",
            op_t::kind2str(n->get_kind()).c_str());
    return true;
}

// check function for output number of LayerNorm and GroupNorm forward.
// if keep_stats == true, outputs should include mean and variance.
bool check_ln_gn_fwd_outputs_num(const op_t *n) {
    const size_t actual_num = n->num_outputs();
    const bool keep_stats = n->has_attr(op_attr::keep_stats)
            ? n->get_attr<bool>(op_attr::keep_stats)
            : true;
    if (keep_stats) {
        VCHECK_SHAPE_INFER((actual_num == 3),
                "%s, outputs should include mean and variance if keep_stats is "
                "true, given output num: %zu.",
                op_t::kind2str(n->get_kind()).c_str(), actual_num);
    }

    return true;
}

// check function for output number of LayerNorm backward.
// if use_affine == true, outputs should include mean and variance.
bool check_ln_bwd_use_affine(const op_t *n) {
    const size_t actual_num = n->num_outputs();
    const bool use_affine = n->has_attr(op_attr::use_affine)
            ? n->get_attr<bool>(op_attr::use_affine)
            : true;
    if (use_affine) {
        VCHECK_SHAPE_INFER((actual_num == 3),
                "%s, outputs should include mean and variance if use_affine is "
                "true, given output num: %zu.",
                op_t::kind2str(n->get_kind()).c_str(), actual_num);
    }
    return true;
}

// check function foraxes of Reduce.
// including Reduce: L1/L2/Max/Mean/Min/Prod/Sum.
// attribute_axes and input_axes is incompatible.
bool check_reduce_axes(const op_t *n) {
    const bool axes = n->has_attr(op_attr::axes);
    const size_t inputs_num = n->num_inputs();
    const bool input_axes = (inputs_num == 2);
    const auto axes_attr_or_input_axes
            = ((axes && !input_axes) || (!axes && input_axes));
    VCHECK_SHAPE_INFER(axes_attr_or_input_axes,
            "%s, exactly one of attribute axes and the second input tensor "
            "axes should be available.",
            op_t::kind2str(n->get_kind()).c_str());
    return true;
}

// Check function for scales and zps of Quantize/Dequantize. The sizes of scales
// and zps (if presented) should be same. Especially when qtype == "per-tensor",
// size of scales/zps should be 1. For f8 quantization, zps is not required.
bool check_quant_dequant_scales_zps(const op_t *n) {
    const logical_tensor_t &src_lt
            = n->get_input_value(0)->get_logical_tensor();
    const logical_tensor_t &dst_lt
            = n->get_input_value(0)->get_logical_tensor();
    const int64_t sz_scales
            = n->get_attr<std::vector<float>>(op_attr::scales).size();

    // qtype is not a required attribute.
    const auto qtype = n->has_attr(op_attr::qtype)
            ? n->get_attr<std::string>(op_attr::qtype)
            : "per_tensor";
    if (qtype == "per_tensor") {
        VCHECK_SHAPE_INFER((sz_scales == 1),
                "%s, the number of scales and zps should be 1 for per-tensor "
                "policy. given scale size: %d.",
                op_t::kind2str(n->get_kind()).c_str(),
                static_cast<int>(sz_scales));
    }

    if (n->has_attr(op_attr::zps)) {
        // f8 quantization or dequantization does not support zps.
        const bool f8_src = utils::one_of(
                src_lt.data_type, data_type::f8_e5m2, data_type::f8_e4m3);
        const bool f8_dst = utils::one_of(
                dst_lt.data_type, data_type::f8_e5m2, data_type::f8_e4m3);
        if (f8_src || f8_dst) {
            VCHECK_SHAPE_INFER(false,
                    "%s, f8 quantization or dequantization does not support "
                    "zps.",
                    op_t::kind2str(n->get_kind()).c_str());
        }

        const int64_t sz_zps
                = n->get_attr<std::vector<int64_t>>(op_attr::zps).size();

        VCHECK_SHAPE_INFER((sz_zps == sz_scales),
                "%s, the number of scales and zps should keep same. given "
                "scale size: %d, given zp size: %d.",
                op_t::kind2str(n->get_kind()).c_str(),
                static_cast<int>(sz_scales), static_cast<int>(sz_zps));
    }

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
        if (qtype == "per_tensor") {
            VCHECK_SHAPE_INFER((sz_scales == 1),
                    "%s, scales should be 1 for per_tensor policy. "
                    "given scale size: %d.",
                    op_t::kind2str(n->get_kind()).c_str(),
                    static_cast<int>(sz_scales));
        }

        return true;
    } else {
        const int64_t sz_zps
                = n->get_input_value(2)->get_logical_tensor().dims[0];

        // in case of not setting value for zps
        if (sz_zps == DNNL_GRAPH_UNKNOWN_DIM) { return true; }

        if (qtype == "per_group") {
            const auto &ndims
                    = n->get_input_value(1)->get_logical_tensor().ndims;
            const auto &scale_ndims
                    = n->get_input_value(1)->get_logical_tensor().ndims;
            const auto &scale_dims
                    = n->get_input_value(1)->get_logical_tensor().dims;
            const auto &zp_ndims
                    = n->get_input_value(2)->get_logical_tensor().ndims;
            const auto &zp_dims
                    = n->get_input_value(2)->get_logical_tensor().dims;
            VCHECK_SHAPE_INFER((ndims >= 2),
                    "group quantization requires at least two dimensions");
            VCHECK_SHAPE_INFER(((ndims == scale_ndims) && (ndims == zp_ndims)),
                    "%s, input, scales and zps should keep the number of "
                    "dimensions for group quantization",
                    op_t::kind2str(n->get_kind()).c_str());
            VCHECK_SHAPE_INFER(
                    (std::equal(scale_dims, scale_dims + ndims, zp_dims)),
                    "%s, scales and zps should keep the same shape for group "
                    "quantization",
                    op_t::kind2str(n->get_kind()).c_str());
        }

        if (qtype == "per_channel") {
            VCHECK_SHAPE_INFER((sz_zps == 1 || sz_scales == sz_zps),
                    "%s, zps should be 1 or equals to scales size for "
                    "per_channel policy, given zps size: %d and scales size: "
                    "%d",
                    op_t::kind2str(n->get_kind()).c_str(),
                    static_cast<int>(sz_zps), static_cast<int>(sz_scales));
        }

        if (qtype == "per_tensor") {
            VCHECK_SHAPE_INFER((sz_zps == 1),
                    "%s, zps should be 1 for per_tensor policy. "
                    "given zps size: %d.",
                    op_t::kind2str(n->get_kind()).c_str(),
                    static_cast<int>(sz_zps));
        }

        return true;
    }
    return true;
}

} // namespace graph
} // namespace impl
} // namespace dnnl
