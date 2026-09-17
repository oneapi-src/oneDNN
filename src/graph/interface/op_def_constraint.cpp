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

#include "common/verbose.hpp"

#include "graph/interface/op_def_constraint.hpp"

#define VCHECK_SHAPE_INFER(cond, msg, ...) \
    VCONDCHECK(graph, create, check, shape_infer, (cond), false, msg, \
            ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace graph {

// check function for padding value of Conv, Convtranspose, Pooling, etc. Both
// pads_begin and pads_end should be a s64 list containing non-negative values.
bool check_pads(const op_t *n) {
    auto hasNegative = [](const dims &pads) {
        return std::any_of(pads.begin(), pads.end(),
                [](dim_t element) { return element < 0; });
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
    const logical_tensor_t &src_lt = n->get_input_logical_tensor(0);
    const logical_tensor_t &aux_lt = n->get_input_logical_tensor(2);

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
    const auto &inputs = mm->get_input_values();
    const auto &outputs = mm->get_output_values();

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

// softmax:
//   1. f32 -> f32/bf16/f16
//   2. bf16 -> f32/bf16
//   3. f16 -> f32/f16
bool check_softmax_dtype(const op_t *n) {
    const auto &inputs = n->get_input_values();
    const auto &outputs = n->get_output_values();

    const logical_tensor_t &src = inputs[0]->get_logical_tensor();
    const logical_tensor_t &dst = outputs[0]->get_logical_tensor();
    if (src.data_type != dst.data_type) {
        if (src.data_type != data_type::f32
                && dst.data_type != data_type::f32) {
            VCHECK_SHAPE_INFER(false, "%s, %s src + %s dst is not supported",
                    op_t::kind2str(n->get_kind()).c_str(),
                    dnnl_dt2str(src.data_type), dnnl_dt2str(dst.data_type));
        }
    }

    return true;
}

// For SoftMaxBackward, diff_src should be f32,  or the same data type as dst
// and diff_dst.
bool check_softmax_bwd_output_dtype(const op_t *n) {
    const auto &inputs = n->get_input_values();
    const auto &outputs = n->get_output_values();

    const logical_tensor_t &diff_dst = inputs[0]->get_logical_tensor();
    const logical_tensor_t &diff_src = outputs[0]->get_logical_tensor();
    if (diff_src.data_type != diff_dst.data_type
            && diff_src.data_type != data_type::f32) {
        VCHECK_SHAPE_INFER(false,
                "%s, %s diff_dst + %s diff_src is not supported",
                op_t::kind2str(n->get_kind()).c_str(),
                dnnl_dt2str(diff_dst.data_type),
                dnnl_dt2str(diff_src.data_type));
    }

    return true;
}

// check function for data_type of LayerNorm, GroupNorm and RMSNorm.
// only when data is bf16, gamma/beta/mean/var can be bf16.
// If data is bf16, gamma/beta/mean/var can be f32 or bf16.
bool check_norm_data_type(const op_t *n) {
    const auto &input_values = n->get_input_values();
    const auto &output_values = n->get_output_values();

    const logical_tensor_t &src_lt = input_values[0]->get_logical_tensor();
    logical_tensor_t aux_lt;
    // check if optional input /output exists
    if (input_values.size() == 1 && output_values.size() == 1) {
        return true;
    } else {
        // RMSNorm uses only one aux tensor
        if (input_values.size() > 1) {
            aux_lt = input_values[1]->get_logical_tensor();
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
    const logical_tensor_t &src_lt = n->get_input_logical_tensor(0);
    const logical_tensor_t &aux_lt = n->get_output_logical_tensor(0);

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
    const logical_tensor_t &src_lt = n->get_input_logical_tensor(0);
    const logical_tensor_t &dst_lt = n->get_input_logical_tensor(0);
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

// Validates the shape/attribute constraints of DynamicQuantize and
// DynamicDequantize. These checks run at op-schema (graph build) time so that
// invalid graphs are rejected early instead of failing later at compilation.
bool check_dyn_quant_dequant_scales_zps(const op_t *n) {
    const std::string op_name = op_t::kind2str(n->get_kind());
    const bool has_zps = n->num_inputs() == 3;
    const auto &src_lt = n->get_input_logical_tensor(0);
    const auto &scales_lt = n->get_input_logical_tensor(1);

    // f8 quantization/dequantization does not support zps.
    if (has_zps) {
        const auto &dst_lt = n->get_output_logical_tensor(0);
        const bool f8 = utils::one_of(src_lt.data_type, data_type::f8_e5m2,
                                data_type::f8_e4m3)
                || utils::one_of(dst_lt.data_type, data_type::f8_e5m2,
                        data_type::f8_e4m3);
        VCHECK_SHAPE_INFER(!f8,
                "%s, f8 quantization or dequantization does not support zps.",
                op_name.c_str());
    }

    const bool has_mask = n->has_attr(op_attr::mask);
    const bool has_group = n->has_attr(op_attr::group_shape)
            && !n->get_attr<dims>(op_attr::group_shape).empty();
    const std::string qtype = n->has_attr(op_attr::qtype)
            ? n->get_attr<std::string>(op_attr::qtype)
            : "per_tensor";

    // There is a usage of graph API that the input/output logical tensors of
    // the op are not fully specialized (containing unknown ndim and dims) when
    // adding the op into a graph. The concrete shapes and layouts will be
    // deferred to the compile API. For this case, skip the checks and allow the
    // op to be added.
    const int64_t r = src_lt.ndims;
    if (r < 0) return true; // unknown source rank

    // Attribute/structure checks that do not depend on dimension sizes.
    int64_t norm_axis = 1;
    if (has_group) {
        const auto &group_shape = n->get_attr<dims>(op_attr::group_shape);
        VCHECK_SHAPE_INFER(r >= 2,
                "%s, per_group quantization requires source rank >= 2.",
                op_name.c_str());
        VCHECK_SHAPE_INFER(static_cast<int64_t>(group_shape.size()) == r,
                "%s, group_shape length must equal source rank.",
                op_name.c_str());
        VCHECK_SHAPE_INFER(has_mask || qtype == "per_group",
                "%s, group_shape requires mask or per_group qtype.",
                op_name.c_str());
        for (int64_t d = 0; d < r; ++d)
            VCHECK_SHAPE_INFER(group_shape[d] >= 1,
                    "%s, group_shape values must be positive.",
                    op_name.c_str());
    }
    if (has_mask) {
        const int64_t mask = n->get_attr<int64_t>(op_attr::mask);
        VCHECK_SHAPE_INFER(mask >= 0,
                "%s, mask must be non-negative, given mask: %d.",
                op_name.c_str(), static_cast<int>(mask));
        VCHECK_SHAPE_INFER((mask >> r) == 0,
                "%s, mask must not select dimensions outside the source rank. "
                "given mask: %d, source rank: %d.",
                op_name.c_str(), static_cast<int>(mask), static_cast<int>(r));
        VCHECK_SHAPE_INFER(qtype == "per_tensor",
                "%s, qtype must be per_tensor (default) when mask is set.",
                op_name.c_str());
        if (n->has_attr(op_attr::axis))
            VCHECK_SHAPE_INFER(n->get_attr<int64_t>(op_attr::axis) == 1,
                    "%s, axis must be the default (1) when mask is set.",
                    op_name.c_str());
        if (has_group && r >= 2) {
            // Primitive API requires the last two mask bits set for per_group.
            const int64_t last_two = (1LL << (r - 1)) | (1LL << (r - 2));
            VCHECK_SHAPE_INFER((mask & last_two) == last_two,
                    "%s, per_group requires the last two mask bits to be set.",
                    op_name.c_str());
        }
    } else if (!has_group) {
        if (qtype == "per_channel") {
            norm_axis = n->has_attr(op_attr::axis)
                    ? n->get_attr<int64_t>(op_attr::axis)
                    : 1;
            if (norm_axis < 0) norm_axis += r;
            VCHECK_SHAPE_INFER(norm_axis >= 0 && norm_axis < r,
                    "%s, axis is out of range.", op_name.c_str());
        } else if (qtype == "per_group") {
            VCHECK_SHAPE_INFER(false,
                    "%s, per_group quantization requires group_shape.",
                    op_name.c_str());
        } else {
            VCHECK_SHAPE_INFER(qtype == "per_tensor",
                    "%s, unsupported qtype: %s.", op_name.c_str(),
                    qtype.c_str());
        }
    }

    // Size validation runs only when every involved shape is fully known.
    using ltw = logical_tensor_wrapper_t;
    if (ltw(src_lt).is_shape_unknown() || ltw(scales_lt).is_shape_unknown())
        return true;
    if (has_zps && ltw(n->get_input_logical_tensor(2)).is_shape_unknown())
        return true;

    // Canonical (full-rank) expected scales size per dim; 1 == shared dim.
    std::vector<int64_t> exp(r, 1);
    if (has_group) {
        const auto &group_shape = n->get_attr<dims>(op_attr::group_shape);
        for (int64_t d = 0; d < r; ++d) {
            const int64_t sd = src_lt.dims[d], gd = group_shape[d];
            VCHECK_SHAPE_INFER(gd <= sd && sd % gd == 0,
                    "%s, group_shape[%d] must divide src[%d].", op_name.c_str(),
                    static_cast<int>(d), static_cast<int>(d));
            exp[d] = sd / gd;
        }
    } else if (has_mask) {
        const int64_t mask = n->get_attr<int64_t>(op_attr::mask);
        for (int64_t d = 0; d < r; ++d)
            if (mask & (1LL << d)) exp[d] = src_lt.dims[d];
    } else if (qtype == "per_channel") {
        exp[norm_axis] = src_lt.dims[norm_axis];
    }

    // Packed dims
    std::vector<int64_t> packed_exp;
    for (int64_t d = 0; d < r; ++d)
        if (exp[d] != 1) packed_exp.push_back(exp[d]);

    // scales must match the full-rank or the packed dims.
    const int64_t snd = scales_lt.ndims;
    if (packed_exp.empty()) {
        for (int64_t d = 0; d < snd; ++d)
            VCHECK_SHAPE_INFER(scales_lt.dims[d] == 1,
                    "%s, scales must be scalar for per_tensor quantization.",
                    op_name.c_str());
    } else if (snd == r) {
        for (int64_t d = 0; d < r; ++d)
            VCHECK_SHAPE_INFER(scales_lt.dims[d] == exp[d],
                    "%s, scales dim %d does not match expected size %d.",
                    op_name.c_str(), static_cast<int>(d),
                    static_cast<int>(exp[d]));
    } else if (static_cast<size_t>(snd) == packed_exp.size()) {
        for (size_t k = 0; k < packed_exp.size(); ++k)
            VCHECK_SHAPE_INFER(scales_lt.dims[k] == packed_exp[k],
                    "%s, packed scales dim %d does not match expected size %d.",
                    op_name.c_str(), static_cast<int>(k),
                    static_cast<int>(packed_exp[k]));
    } else {
        VCHECK_SHAPE_INFER(false,
                "%s, scales rank %d matches neither the full-rank (%d) nor the "
                "packed (%d) layout.",
                op_name.c_str(), static_cast<int>(snd), static_cast<int>(r),
                static_cast<int>(packed_exp.size()));
    }

    // zps: scalar (broadcast) or exactly the scales shape. The grouped
    // lowering path does not support scalar broadcast and indexes zps against
    // the full scales shape, so require a shape match when groups are set.
    if (has_zps) {
        const auto &zps_lt = n->get_input_logical_tensor(2);
        const int64_t znd = zps_lt.ndims;
        bool zps_scalar = !has_group;
        for (int64_t d = 0; d < znd; ++d)
            if (zps_lt.dims[d] != 1) zps_scalar = false;
        if (!zps_scalar) {
            VCHECK_SHAPE_INFER(znd == scales_lt.ndims,
                    "%s, zps must be scalar or match the scales shape.",
                    op_name.c_str());
            for (int64_t d = 0; d < znd; ++d)
                VCHECK_SHAPE_INFER(zps_lt.dims[d] == scales_lt.dims[d],
                        "%s, zps shape does not match scales shape.",
                        op_name.c_str());
        }
    }

    return true;
}
} // namespace graph
} // namespace impl
} // namespace dnnl
