/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef LLGA_INTERFACE_SHAPE_INFER_HPP
#define LLGA_INTERFACE_SHAPE_INFER_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "ir.hpp"
#include "logical_tensor.hpp"

namespace llga {
namespace impl {

static std::vector<int64_t> get_weight_spatial_dims(
        const std::string &format, const std::vector<int64_t> &weight_dims) {
    std::vector<int64_t> weight_spatial_dims(weight_dims);
    if (format == "OIX") {
        weight_spatial_dims.erase(
                weight_spatial_dims.begin(), weight_spatial_dims.begin() + 2);
    } else if (format == "XIO") {
        weight_spatial_dims.erase(
                weight_spatial_dims.end() - 2, weight_spatial_dims.end());
    }
    return weight_spatial_dims;
}

static int64_t get_weight_i(
        const std::string &format, const std::vector<int64_t> &weight_dims) {
    if (format == "OIX") {
        return weight_dims[1];
    } else { // if (format == "XIO")
        return weight_dims[weight_dims.size() - 2];
    }
}

static int64_t get_weight_o(
        const std::string &format, const std::vector<int64_t> &weight_dims) {
    if (format == "OIX") {
        return weight_dims[0];
    } else { // if (format == "XIO")
        return weight_dims[weight_dims.size() - 1];
    }
}

static int64_t get_src_n(const std::vector<int64_t> &src_dims) {
    //if (format == "NCX" || format == "NXC")
    return src_dims[0];
}

static int64_t get_src_c(
        const std::string &format, const std::vector<int64_t> &src_dims) {
    if (format == "NCX") {
        return src_dims[1];
    } else { // if (format == "NXC") {
        return src_dims[src_dims.size() - 1];
    }
}

static std::vector<int64_t> get_src_spatial_dims(
        const std::string &format, const std::vector<int64_t> &src_dims) {
    std::vector<int64_t> src_spatial_dims(src_dims);
    if (format == "NCX") {
        src_spatial_dims.erase(
                src_spatial_dims.begin(), src_spatial_dims.begin() + 2);
    } else if (format == "NXC") {
        src_spatial_dims.erase(
                src_spatial_dims.begin(), src_spatial_dims.begin() + 1);
        src_spatial_dims.erase(
                src_spatial_dims.end() - 1, src_spatial_dims.end());
    }
    return src_spatial_dims;
}

static std::vector<int64_t> make_output_dims(const std::string &format,
        int64_t n, int64_t c, const std::vector<int64_t> &spatial_dims) {
    std::vector<int64_t> output_dims;
    if (format == "NCX") {
        output_dims.push_back(n);
        output_dims.push_back(c);
        for (auto dim : spatial_dims) {
            output_dims.push_back(dim);
        }
    } else if (format == "NXC") {
        output_dims.push_back(n);
        for (auto dim : spatial_dims) {
            output_dims.push_back(dim);
        }
        output_dims.push_back(c);
    }
    return output_dims;
}

static std::vector<int64_t> make_output_weight_dims(
        const std::string &filter_format, int64_t i, int64_t o,
        std::vector<int64_t> &weight_spatial_dims) {
    std::vector<int64_t> output_weight_dims;
    if (filter_format == "XIO") {
        for (auto dim : weight_spatial_dims) {
            output_weight_dims.push_back(dim);
        }
        output_weight_dims.push_back(i);
        output_weight_dims.push_back(o);
    } else if (filter_format == "OIX") {
        output_weight_dims.push_back(o);
        output_weight_dims.push_back(i);
        for (auto dim : weight_spatial_dims) {
            output_weight_dims.push_back(dim);
        }
    }
    return output_weight_dims;
}

static status_t check_partial_shape_correctness(
        const std::vector<int64_t> &infered_shape,
        const std::vector<int64_t> &expected_shape) {
    if (infered_shape.size() != expected_shape.size()) {
        // infered logical tensor shape doesn't have expected ndims
        return status::invalid_shape;
    }

    for (size_t i = 0; i < infered_shape.size(); ++i) {
        if (expected_shape[i] != -1 && infered_shape[i] != expected_shape[i]) {
            // infered logical tensor shape doesn't align with expectation
            return status::invalid_shape;
        }
    }
    return status::success;
}

static std::vector<int64_t> compute_dense_strides(
        const std::vector<int64_t> &output_dims) {
    std::vector<int64_t> output_strides(output_dims.size());
    for (auto it = output_dims.begin(); it < output_dims.end(); ++it) {
        const auto val = std::accumulate(std::next(it), output_dims.end(), 1,
                std::multiplies<int64_t>());
        const auto dist = std::distance(output_dims.begin(), it);
        output_strides[static_cast<size_t>(dist)] = val;
    }
    return output_strides;
}

static int64_t get_n_channels(
        const node_t *const n, const std::vector<int64_t> &input_dims) {
    const std::string data_f_attr_name = "data_format";
    const std::string default_data_f = "NXC";
    const std::string data_f = [n, &data_f_attr_name, &default_data_f]() {
        if (n->has_attr(data_f_attr_name)) {
            return n->get_attr<std::string>(data_f_attr_name);
        }
        return default_data_f;
    }();
    const auto channels
            = (data_f == default_data_f) ? input_dims.back() : input_dims[1];
    return channels;
}

static bool are_all_shapes_known(
        const std::vector<logical_tensor_t *> &outputs) {
    if (std::all_of(outputs.cbegin(), outputs.cend(),
                [](const logical_tensor_t *const lt) {
                    const auto outx = logical_tensor_wrapper(lt);
                    return !outx.is_shape_unknown();
                })) {
        return true;
    }
    return false;
}

static bool verify_shapes_in_range(const std::vector<logical_tensor_t *> &lts,
        const size_t beg_idx, const size_t end_idx,
        const std::function<bool(const std::vector<int64_t>)> validator) {
    for (auto idx = beg_idx; idx < end_idx; ++idx) {
        const auto ltx = logical_tensor_wrapper(lts[idx]);
        const std::vector<int64_t> ltx_dims = ltx.vdims();
        if (!validator(ltx_dims)) return false;
    }
    return true;
}

static void set_shapes_in_range(const std::vector<logical_tensor_t *> &lts,
        const size_t beg_idx, const size_t end_idx,
        const std::vector<int64_t> &new_dims) {
    for (auto idx = beg_idx; idx < end_idx; ++idx) {
        utils::array_copy(lts[idx]->dims, new_dims.data(), new_dims.size());
        lts[idx]->ndims = static_cast<int32_t>(new_dims.size());
        auto ltx = logical_tensor_wrapper(lts[idx]);
        if (ltx.is_strided()) {
            const std::vector<dim_t> ltx_strides
                    = compute_dense_strides(new_dims);
            utils::array_copy(lts[idx]->layout.strides, ltx_strides.data(),
                    ltx_strides.size());
        }
    }
}

void infer_auto_pad(const std::vector<int64_t> &weight_spatial_dims,
        const std::string &auto_pad, const std::vector<int64_t> strides,
        std::vector<int64_t> &new_pads_begin,
        std::vector<int64_t> &new_pads_end,
        const std::vector<int64_t> &dilations) {
    if (auto_pad == "VALID") {
        size_t rank = weight_spatial_dims.size();
        new_pads_begin.assign(rank, 0);
        new_pads_end.assign(rank, 0);
    } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        for (size_t i = 0; i < weight_spatial_dims.size(); ++i) {
            int64_t padding_needed
                    = (weight_spatial_dims[i] - strides[i]) * dilations[i];
            int64_t padding_lhs = padding_needed / 2;
            int64_t padding_rhs = padding_needed - padding_lhs;
            new_pads_begin[i]
                    = (auto_pad == "SAME_UPPER" ? padding_lhs : padding_rhs);
            new_pads_end[i]
                    = (auto_pad == "SAME_UPPER" ? padding_rhs : padding_lhs);
        }
    }
}

status_t calculate_broadcast_shape(std::vector<int64_t> &lhs_shape,
        std::vector<int64_t> &rhs_shape,
        std::vector<int64_t> &broadcast_shape) {
    auto lhs_rank = lhs_shape.size();
    auto rhs_rank = rhs_shape.size();
    auto max_rank = std::max(lhs_rank, rhs_rank);

    // left-pad the lhs_shape with ones
    lhs_shape.insert(lhs_shape.begin(), max_rank - lhs_rank, 1);
    // left-pad the rhs_shape with ones
    rhs_shape.insert(rhs_shape.begin(), max_rank - rhs_rank, 1);

    for (size_t index = 0; index < max_rank; ++index) {
        int64_t lhs_dim = lhs_shape.at(index);
        int64_t rhs_dim = rhs_shape.at(index);

        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            // broadcast_shape: incompatible arg shapes
            return status::invalid_shape;
        }

        broadcast_shape.push_back(std::max(lhs_dim, rhs_dim));
    }

    return status::success;
}

static void set_infered_shape_and_strides(
        logical_tensor_t *output, const std::vector<dim_t> &infered_shape) {
    utils::array_copy(output->dims, infered_shape.data(), infered_shape.size());
    output->ndims = static_cast<int32_t>(infered_shape.size());
    // We compute default dense strides for strided output. FWK can
    // overwrite these strides if it has specific strides
    if (logical_tensor_wrapper(output).is_strided()) {
        const std::vector<dim_t> strides = compute_dense_strides(infered_shape);
        utils::array_copy(
                output->layout.strides, strides.data(), strides.size());
    }
}

// check if output shape is already known
// if shape is unknown, infer output shape (change output lt)
// otherwise infer pad (change node attrs)
status_t infer_conv_output_shape(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper(inputs[0]);
    auto in1 = logical_tensor_wrapper(inputs[1]);
    auto out0 = logical_tensor_wrapper(outputs[0]);

    // get attr value
    const std::vector<int64_t> &strides
            = n->get_attr<std::vector<int64_t>>("strides");
    const std::vector<int64_t> &dilations
            = n->get_attr<std::vector<int64_t>>("dilations");
    const std::vector<int64_t> &pads_begin
            = n->get_attr<std::vector<int64_t>>("pads_begin");
    const std::vector<int64_t> &pads_end
            = n->get_attr<std::vector<int64_t>>("pads_end");
    const std::vector<int64_t> src_dims = in0.vdims();
    const std::vector<int64_t> weight_dims = in1.vdims();
    std::string filter_format = n->get_attr<std::string>("filter_format");
    std::string src_format = n->get_attr<std::string>("data_format");
    // check if src channel == weight input channel
    if (get_src_c(src_format, src_dims)
            != get_weight_i(filter_format, weight_dims)) {
        // src channel is not equal to weights input channel
        return status::invalid_shape;
    }
    std::vector<int64_t> src_spatial_dims
            = get_src_spatial_dims(src_format, src_dims);
    std::vector<int64_t> weight_spatial_dims
            = get_weight_spatial_dims(filter_format, weight_dims);

    std::vector<int64_t> infered_pads_begin(pads_begin);
    std::vector<int64_t> infered_pads_end(pads_end);

    if (n->has_attr("auto_pad")
            && n->get_attr<std::string>("auto_pad") != "None") {
        std::string auto_pad = n->get_attr<std::string>("auto_pad");
        // infer auto_pad
        infer_auto_pad(weight_spatial_dims, auto_pad, strides,
                infered_pads_begin, infered_pads_end, dilations);
        if (!out0.is_shape_unknown()) {
            // if shape is known, infer pad (change node attrs)
            n->set_attr<std::vector<int64_t>>("pads_begin", infered_pads_begin);
            n->set_attr<std::vector<int64_t>>("pads_end", infered_pads_end);
            return status::success;
        }
    }

    std::vector<int64_t> output_spatial_dims;
    for (size_t i = 0; i < src_spatial_dims.size(); ++i) {
        int64_t src_padded_dim = src_spatial_dims[i] + infered_pads_begin[i]
                + infered_pads_end[i];
        int64_t weight_dilated_dim
                = dilations[i] * (weight_spatial_dims[i] - 1) + 1;
        output_spatial_dims.push_back(
                ((src_padded_dim - weight_dilated_dim) / strides[i]) + 1);
    }

    const std::vector<dim_t> out0_shape = make_output_dims(src_format,
            get_src_n(src_dims), get_weight_o(filter_format, weight_dims),
            output_spatial_dims);
    if (out0.ndims() != -1) {
        status_t ret
                = check_partial_shape_correctness(out0_shape, out0.vdims());
        if (ret != status::success) { return ret; }
    }

    set_infered_shape_and_strides(outputs[0], out0_shape);
    return status::success;
}

status_t infer_conv_bprop_data_output_shape(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper(inputs[0]);
    auto in1 = logical_tensor_wrapper(inputs[1]);
    auto out0 = logical_tensor_wrapper(outputs[0]);

    // get attr value
    const std::vector<int64_t> &strides
            = n->get_attr<std::vector<int64_t>>("strides");
    const std::vector<int64_t> &pads_begin
            = n->get_attr<std::vector<int64_t>>("pads_begin");
    const std::vector<int64_t> &pads_end
            = n->get_attr<std::vector<int64_t>>("pads_end");
    const std::vector<int64_t> &dilations
            = n->get_attr<std::vector<int64_t>>("dilations");
    const std::vector<int64_t> src_dims = in0.vdims();
    const std::vector<int64_t> weight_dims = in1.vdims();
    std::string filter_format = n->get_attr<std::string>("filter_format");
    std::string src_format = n->get_attr<std::string>("data_format");
    // check if src channel == weight input channel
    if (get_src_c(src_format, src_dims)
            != get_weight_i(filter_format, weight_dims)) {
        // src channel is not equal to weights input channel
        return status::invalid_shape;
    }
    std::vector<int64_t> src_spatial_dims
            = get_src_spatial_dims(src_format, src_dims);
    std::vector<int64_t> weight_spatial_dims
            = get_weight_spatial_dims(filter_format, weight_dims);

    std::vector<int64_t> infered_pads_begin(pads_begin);
    std::vector<int64_t> infered_pads_end(pads_end);

    std::vector<int64_t> output_padding(src_spatial_dims.size(), 0);
    if (n->has_attr("output_padding")) {
        output_padding = n->get_attr<std::vector<int64_t>>("output_padding");
    }

    if (n->has_attr("auto_pad")
            && n->get_attr<std::string>("auto_pad") != "None") {
        std::string auto_pad = n->get_attr<std::string>("auto_pad");
        // infer auto_pad
        infer_auto_pad(weight_spatial_dims, auto_pad, strides,
                infered_pads_begin, infered_pads_end, dilations);
        if (!out0.is_shape_unknown()) {
            // if shape is known, infer pad (change node attrs)
            n->set_attr<std::vector<int64_t>>("pads_begin", infered_pads_begin);
            n->set_attr<std::vector<int64_t>>("pads_end", infered_pads_end);
            return status::success;
        }
    }
    std::vector<int64_t> output_spatial_dims;
    // third input - output_shape is optional.
    // When output_shape is specified pads_pegin and pads_end are ignored,
    // and auto_pad defines how to distribute padding amount around the tensor.
    if (inputs.size() == 3 && logical_tensor_wrapper(inputs[2]).ndims() != -1) {
        // Since we have no access to the data of the third input
        // (output_shape), we cannot set output spatial shape.
        return status::unsupported;
    } else {
        for (size_t i = 0; i < src_spatial_dims.size(); ++i) {
            int64_t padded_dim = output_padding[i] - infered_pads_begin[i]
                    - infered_pads_end[i];
            int64_t weight_dilated_dim
                    = dilations[i] * (weight_spatial_dims[i] - 1) + 1;
            output_spatial_dims.push_back(strides[i] * (src_spatial_dims[i] - 1)
                    + weight_dilated_dim + padded_dim);
        }
    }

    const std::vector<dim_t> out0_shape = make_output_dims(src_format,
            get_src_n(src_dims), get_weight_o(filter_format, weight_dims),
            output_spatial_dims);
    utils::array_copy(outputs[0]->dims, out0_shape.data(), out0_shape.size());
    outputs[0]->ndims = static_cast<int32_t>(out0_shape.size());

    return status::success;
}

status_t infer_conv_bprop_filters_output_shape(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper(inputs[0]);
    auto in2 = logical_tensor_wrapper(inputs[2]);
    auto out0 = logical_tensor_wrapper(outputs[0]);

    // get attr value
    const std::vector<int64_t> &strides
            = n->get_attr<std::vector<int64_t>>("strides");
    const std::vector<int64_t> &pads_begin
            = n->get_attr<std::vector<int64_t>>("pads_begin");
    const std::vector<int64_t> &pads_end
            = n->get_attr<std::vector<int64_t>>("pads_end");
    const std::vector<int64_t> &dilations
            = n->get_attr<std::vector<int64_t>>("dilations");
    std::string filter_format = n->get_attr<std::string>("filter_format");
    std::string src_format = n->get_attr<std::string>("data_format");

    const std::vector<int64_t> src_dims = in0.vdims();
    const std::vector<int64_t> output_delta_dims = in2.vdims();

    std::vector<int64_t> src_spatial_dims
            = get_src_spatial_dims(src_format, src_dims);
    std::vector<int64_t> output_delta_spatial_dims
            = get_src_spatial_dims(src_format, output_delta_dims);

    std::vector<int64_t> infered_pads_begin(pads_begin);
    std::vector<int64_t> infered_pads_end(pads_end);

    if (n->has_attr("auto_pad")
            && n->get_attr<std::string>("auto_pad") != "None") {
        std::string auto_pad = n->get_attr<std::string>("auto_pad");

        if (auto_pad == "VALID") {
            size_t rank = src_spatial_dims.size();
            infered_pads_begin.assign(rank, 0);
            infered_pads_end.assign(rank, 0);
        } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
            // Since we have no access to the data of the second input
            // (weights_shape), we cannot calculate auto pads.
            return status::unsupported;
        }
        if (!out0.is_shape_unknown()) {
            // if shape is known, infer pad (change node attrs)
            n->set_attr<std::vector<int64_t>>("pads_begin", infered_pads_begin);
            n->set_attr<std::vector<int64_t>>("pads_end", infered_pads_end);
            return status::success;
        }
    }

    // Since we have no access to the data of the second input (weights_shape),
    // we have to get weights spatial shape using another way.
    // To do that we use transformed convolution output size formula.
    std::vector<int64_t> weight_spatial_dims;
    for (size_t i = 0; i < src_spatial_dims.size(); ++i) {
        int64_t src_padded_dim = src_spatial_dims[i] + infered_pads_begin[i]
                + infered_pads_end[i];
        int64_t strided_output_delta
                = strides[i] * (output_delta_spatial_dims[i] - 1);
        weight_spatial_dims.push_back(
                ((src_padded_dim - strided_output_delta - 1) / dilations[i])
                + 1);
    }

    const std::vector<dim_t> out0_shape = make_output_weight_dims(filter_format,
            get_src_c(src_format, src_dims),
            get_src_c(src_format, output_delta_dims), weight_spatial_dims);

    if (out0.ndims() != -1) {
        status_t ret
                = check_partial_shape_correctness(out0_shape, out0.vdims());
        if (ret != status::success) { return ret; }
    }
    utils::array_copy(outputs[0]->dims, out0_shape.data(), out0_shape.size());
    outputs[0]->ndims = static_cast<int32_t>(out0_shape.size());

    return status::success;
}

// check if output shape is already known
// if shape is unknown, infer output shape (change output lt)
// otherwise infer pad (change node attrs)
status_t infer_pool_output_shape(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper(inputs[0]);
    auto out0 = logical_tensor_wrapper(outputs[0]);

    // get attr value
    const std::vector<int64_t> &strides
            = n->get_attr<std::vector<int64_t>>("strides");
    const std::vector<int64_t> &kernel
            = n->get_attr<std::vector<int64_t>>("kernel");
    std::vector<int64_t> dilations(kernel.size(), 1);
    if (n->has_attr("dilations"))
        dilations = n->get_attr<std::vector<int64_t>>("dilations");
    const std::vector<int64_t> &pads_begin
            = n->get_attr<std::vector<int64_t>>("pads_begin");
    const std::vector<int64_t> &pads_end
            = n->get_attr<std::vector<int64_t>>("pads_end");
    const std::vector<int64_t> src_dims = in0.vdims();

    std::string rounding_type = "floor";
    if (n->has_attr("rounding_type")) {
        rounding_type = n->get_attr<std::string>("rounding_type");
    }
    std::string src_format = n->get_attr<std::string>("data_format");
    std::vector<int64_t> src_spatial_dims
            = get_src_spatial_dims(src_format, src_dims);

    std::vector<int64_t> infered_pads_begin(pads_begin);
    std::vector<int64_t> infered_pads_end(pads_end);
    if (n->has_attr("auto_pad")
            && n->get_attr<std::string>("auto_pad") != "None") {
        std::string auto_pad = n->get_attr<std::string>("auto_pad");
        // infer auto_pad
        infer_auto_pad(kernel, auto_pad, strides, infered_pads_begin,
                infered_pads_end, dilations);
    }

    std::vector<int64_t> output_spatial_dims;
    for (size_t i = 0; i < src_spatial_dims.size(); ++i) {
        int64_t src_padded_dim = src_spatial_dims[i] + infered_pads_begin[i]
                + infered_pads_end[i];
        int64_t kernel_dilated_dim = dilations[i] * (kernel[i] - 1) + 1;
        int64_t out_value;
        if (rounding_type == "ceil") {
            out_value = (src_padded_dim - kernel_dilated_dim - 1) / strides[i]
                    + 2;
        } else {
            out_value = (src_padded_dim - kernel_dilated_dim) / strides[i] + 1;
        }
        output_spatial_dims.push_back(out_value);
    }

    // if shape is known, infer pad (change node attrs)
    if (!out0.is_shape_unknown()) {
        if (rounding_type == "ceil") {
            for (size_t i = 0; i < src_spatial_dims.size(); ++i) {
                int64_t kernel_dilated_dim = dilations[i] * (kernel[i] - 1) + 1;
                int64_t cur_pads_end = (output_spatial_dims[i] - 1) * strides[i]
                        + kernel_dilated_dim - src_spatial_dims[i]
                        - pads_begin[i];
                infered_pads_end[i] = cur_pads_end;
            }
        }
        n->set_attr<std::vector<int64_t>>("pads_begin", infered_pads_begin);
        n->set_attr<std::vector<int64_t>>("pads_end", infered_pads_end);
        return status::success;
    }

    std::vector<dim_t> out_shape
            = make_output_dims(src_format, get_src_n(src_dims),
                    get_src_c(src_format, src_dims), output_spatial_dims);
    if (out0.ndims() != -1) {
        status_t ret = check_partial_shape_correctness(out_shape, out0.vdims());
        if (ret != status::success) { return ret; }
    }

    set_infered_shape_and_strides(outputs[0], out_shape);
    return status::success;
}

status_t infer_matmul_output_shape(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper(inputs[0]);
    auto in1 = logical_tensor_wrapper(inputs[1]);
    auto out0 = logical_tensor_wrapper(outputs[0]);

    // check if output shape is already known
    if (!out0.is_shape_unknown()) return status::success;
    // get attr value
    bool transpose_a = 0;
    if (n->has_attr("transpose_a")) {
        transpose_a = n->get_attr<bool>("transpose_a");
    }
    bool transpose_b = 0;
    if (n->has_attr("transpose_b")) {
        transpose_b = n->get_attr<bool>("transpose_b");
    }
    const std::vector<int64_t> input0_dims = in0.vdims();
    const std::vector<int64_t> input1_dims = in1.vdims();
    size_t input0_rank = input0_dims.size();
    size_t input1_rank = input1_dims.size();
    std::vector<int64_t> updated_input0(input0_dims);
    std::vector<int64_t> updated_input1(input1_dims);
    if (transpose_a && input0_rank > 1) {
        std::swap(updated_input0[input0_rank - 2],
                updated_input0[input0_rank - 1]);
    }
    if (transpose_b && input1_rank > 1) {
        std::swap(updated_input1[input1_rank - 2],
                updated_input1[input1_rank - 1]);
    }

    std::vector<int64_t> infered_out_shape;
    if (input0_rank == 1 && input1_rank == 1) {
        if (updated_input0 != updated_input1) {
            // matmul: incompatible arg shapes
            return status::invalid_shape;
        }
        // example: input0={1,1}, input1={1,1}, output={2}
        infered_out_shape = {1};
    } else if (input0_rank == 1) {
        if (updated_input0[0] != updated_input1[input1_rank - 2]) {
            // matmul: incompatible arg shapes
            return status::invalid_shape;
        }
        // example: input0 shape {3}, input1 shape {2,3,4},
        // output shape {2,4}
        updated_input1.erase(
                updated_input1.begin() + static_cast<int64_t>(input1_rank) - 2);
        infered_out_shape = updated_input1;
    } else if (input1_rank == 1) {
        if (updated_input1[0] != updated_input0[input0_rank - 1]) {
            // matmul: incompatible arg shapes
            return status::invalid_shape;
        }
        // example: input0 shape {2,3,4}, input1 shape {4},
        // output shape {2,3}
        updated_input0.erase(
                updated_input0.begin() + static_cast<int64_t>(input0_rank) - 1);
        infered_out_shape = updated_input0;
    } else if (input0_rank == 2 && input1_rank == 2) {
        if (updated_input0[1] != updated_input1[0]) {
            // matmul: incompatible arg shapes
            return status::invalid_shape;
        }
        // example: input0 shape {1, 3}, input1 shape {3, 2},
        // output shape {1,2}
        infered_out_shape = {updated_input0[0], updated_input1[1]};
    } else {
        if (updated_input0[input0_rank - 1]
                != updated_input1[input1_rank - 2]) {
            // matmul: incompatible arg shapes
            return status::invalid_shape;
        }
        std::vector<int64_t> input0_batch_dims {
                updated_input0.begin(), updated_input0.end() - 2};
        std::vector<int64_t> input1_batch_dims {
                updated_input1.begin(), updated_input1.end() - 2};
        status_t ret = calculate_broadcast_shape(
                input0_batch_dims, input1_batch_dims, infered_out_shape);
        if (ret != status::success) { return ret; }
        infered_out_shape.push_back(updated_input0[input0_rank - 2]);
        infered_out_shape.push_back(updated_input1[input1_rank - 1]);
    }

    if (out0.ndims() != -1) {
        status_t ret = check_partial_shape_correctness(
                infered_out_shape, out0.vdims());
        if (ret != status::success) { return ret; }
    }

    set_infered_shape_and_strides(outputs[0], infered_out_shape);
    return status::success;
}

status_t infer_identity_output_shape(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out0 = logical_tensor_wrapper(outputs[0]);
    auto in0 = logical_tensor_wrapper(inputs[0]);
    if (!out0.is_shape_unknown()) return status::success;

    // check if partial set shape aligns with infered shape
    if (out0.ndims() != -1) {
        status_t ret
                = check_partial_shape_correctness(in0.vdims(), out0.vdims());
        if (ret != status::success) { return ret; }
    }

    // We should compute output dense strides instead of
    // directly copying input strides to it
    set_infered_shape_and_strides(outputs[0], in0.vdims());
    UNUSED(n);
    return status::success;
}

status_t identity_output_shape_on_pos(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs,
        std::vector<uint32_t> &positions) {
    for (auto &pos : positions) {
        std::vector<logical_tensor_t *> ins = {inputs[pos]};
        std::vector<logical_tensor_t *> outs = {outputs[pos]};
        auto status = infer_identity_output_shape(n, ins, outs);
        if (status != status::success) return status;
    }
    return status::success;
}

status_t infer_bias_backprop_output_shape(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out = logical_tensor_wrapper(outputs[0]);
    if (!out.is_shape_unknown()) return status::success;

    auto in = logical_tensor_wrapper(inputs[0]);
    std::vector<int64_t> input_dims = in.vdims();
    if (input_dims.size() < 4) {
        // bias add backprop: input should have at least 4 dims.
        return status::invalid_shape;
    }

    const auto channels = get_n_channels(n, input_dims);
    std::vector<int64_t> new_out_dims = {channels};

    set_infered_shape_and_strides(outputs[0], new_out_dims);
    return status::success;
}

status_t infer_bias_add_output_shape(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out = logical_tensor_wrapper(outputs[0]);
    if (!out.is_shape_unknown()) return status::success;

    auto in = logical_tensor_wrapper(inputs[0]);
    std::vector<int64_t> input_dims = in.vdims();
    if (input_dims.size() < 4) {
        // bias add: input should have at least 4 dims.
        return status::invalid_shape;
    }
    auto bias = logical_tensor_wrapper(inputs[1]);
    std::vector<int64_t> bias_dims = bias.vdims();
    if (bias_dims.size() != 1) {
        // bias add: bias should have exactly 1 dimension.
        return status::invalid_shape;
    }

    const auto channels = get_n_channels(n, input_dims);
    if (bias_dims[0] != channels) {
        // bias add: bias size should match input channels size.
        return status::invalid_shape;
    }

    return infer_identity_output_shape(n, inputs, outputs);
}

status_t infer_norm_output_shape(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto status = infer_identity_output_shape(n, inputs, outputs);
    if (status != status::success) return status;

    const bool keep_stats = n->has_attr("keep_stats")
            ? n->get_attr<bool>("keep_stats")
            : false;
    if (!keep_stats) return status::success;

    auto in0 = logical_tensor_wrapper(inputs[0]);
    const std::vector<int64_t> input0_dims = in0.vdims();

    const int64_t begin_norm_axis = n->has_attr("begin_norm_axis")
            ? n->get_attr<int64_t>("begin_norm_axis")
            : -1;

    auto out1 = logical_tensor_wrapper(outputs[1]);
    auto out2 = logical_tensor_wrapper(outputs[2]);
    std::vector<int64_t> output_dims(input0_dims);

    auto norm_starting_position
            = begin_norm_axis >= 0 ? output_dims.begin() : output_dims.end();

    output_dims.erase(
            norm_starting_position + begin_norm_axis, output_dims.end());

    // check if output shape is already known
    if (out1.is_shape_unknown()) {
        set_infered_shape_and_strides(outputs[1], output_dims);
    }

    // check if output shape is already known
    if (out2.is_shape_unknown()) {
        set_infered_shape_and_strides(outputs[2], output_dims);
    }
    return status::success;
}

status_t infer_norm_bprop_output_shape(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    std::vector<uint32_t> identity_shapes_pos = {0};
    if (n->has_attr("use_affine") && n->get_attr<bool>("use_affine") == true) {
        // when use_affine parameter is set,
        // there will be two additional outputs
        identity_shapes_pos.insert(identity_shapes_pos.end(), {1, 2});
    }
    return identity_output_shape_on_pos(
            n, inputs, outputs, identity_shapes_pos);
}

status_t infer_elemwise_arithmetic_output_shape(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper(inputs[0]);
    auto in1 = logical_tensor_wrapper(inputs[1]);
    // check if output shape is already known
    auto out0 = logical_tensor_wrapper(outputs[0]);
    if (!out0.is_shape_unknown()) return status::success;

    const std::string broadcast_attr_name = "auto_broadcast";
    const bool shapes_should_match = [n, &broadcast_attr_name]() {
        if (n->has_attr(broadcast_attr_name)) {
            const auto &auto_broadcast
                    = n->get_attr<std::string>(broadcast_attr_name);
            return auto_broadcast == "none";
        }
        return false;
    }();

    std::vector<int64_t> input0_dims = in0.vdims();
    std::vector<int64_t> input1_dims = in1.vdims();
    std::vector<int64_t> infered_out_shape;
    if (shapes_should_match) {
        if (input0_dims != input1_dims) {
            // add: incompatible input shapes (auto_broadcast=none)
            return status::invalid_shape;
        }
        infered_out_shape = input0_dims;
    } else {
        status_t ret = calculate_broadcast_shape(
                input0_dims, input1_dims, infered_out_shape);
        if (ret != status::success) { return ret; }
    }
    // check if partial set shape aligns with infered shape
    if (out0.ndims() != -1) {
        status_t ret = check_partial_shape_correctness(
                infered_out_shape, out0.vdims());
        if (ret != status::success) { return ret; }
    }

    set_infered_shape_and_strides(outputs[0], infered_out_shape);
    return status::success;
}

status_t infer_bn_fwd_train_output_shape(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using cvec_int64 = const std::vector<int64_t>;

    if (are_all_shapes_known(outputs)) return status::success;

    const auto in = logical_tensor_wrapper(inputs[0]);
    cvec_int64 input_dims = in.vdims();
    if (input_dims.size() < 4) return status::invalid_shape;

    const auto channels = get_n_channels(n, input_dims);
    const auto validator = [&channels](cvec_int64 &vec) {
        return vec.size() == 1 && vec[0] == channels;
    };
    if (!verify_shapes_in_range(inputs, 1, inputs.size(), validator))
        return status::invalid_shape;

    infer_identity_output_shape(n, inputs, outputs);
    cvec_int64 new_out_dims = {channels};
    set_shapes_in_range(outputs, 1, outputs.size(), new_out_dims);
    return status::success;
}

status_t infer_bn_bwd_output_shape(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using cvec_int64 = const std::vector<int64_t>;

    if (are_all_shapes_known(outputs)) return status::success;

    const auto in = logical_tensor_wrapper(inputs[0]);
    cvec_int64 input_dims = in.vdims();
    const auto out_delta = logical_tensor_wrapper(inputs[1]);
    cvec_int64 out_delta_dims = out_delta.vdims();
    if (input_dims.size() < 4 || out_delta_dims.size() < 4)
        return status::invalid_shape;

    const auto channels = get_n_channels(n, input_dims);
    const auto validator = [&channels](cvec_int64 &vec) {
        return vec.size() == 1 && vec[0] == channels;
    };
    if (!verify_shapes_in_range(inputs, 2, inputs.size(), validator))
        return status::invalid_shape;

    infer_identity_output_shape(n, inputs, outputs);
    cvec_int64 new_out_dims = {channels};
    set_shapes_in_range(outputs, 1, outputs.size(), new_out_dims);
    return status::success;
}

status_t infer_concat_output_shape(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out0 = logical_tensor_wrapper(outputs[0]);
    if (!out0.is_shape_unknown()) return status::success;

    // if only one tensor to concat, out_shape is same as input_shape
    if (inputs.size() == 1) {
        infer_identity_output_shape(n, inputs, outputs);
        return status::success;
    }
    auto in0 = logical_tensor_wrapper(inputs[0]);
    auto data_type = in0.data_type();
    int64_t axis = n->get_attr<int64_t>("axis");
    auto ndims = in0.ndims();
    auto dims = in0.dims();
    if (axis < -ndims || axis >= ndims) {
        return status::invalid_argument;
    } else if (axis < 0) {
        axis += ndims;
    }

    int64_t sum = 0;
    for (auto iter = inputs.cbegin(); iter != inputs.cend(); iter++) {
        auto lt_inN = logical_tensor_wrapper(*iter);
        const auto &lt_inN_dims = lt_inN.vdims();
        if (lt_inN.ndims() != ndims) { return status::invalid_shape; }
        if (lt_inN.data_type() != data_type) { return status::unsupported; }
        for (int32_t i = 0; i < ndims; i++) {
            if (i != axis) {
                // input dims should be same except axis dim.
                if (dims[i] != lt_inN_dims[static_cast<size_t>(i)]) {
                    return status::invalid_shape;
                }
            } else {
                sum += lt_inN_dims[static_cast<size_t>(axis)];
            }
        }
    };

    utils::array_copy(outputs[0]->dims, dims, static_cast<size_t>(ndims));
    outputs[0]->dims[axis] = sum;
    return status::success;
}

status_t infer_unsupported_output_shape(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    UNUSED(n);
    UNUSED(inputs);
    auto out0 = logical_tensor_wrapper(outputs[0]);
    if (out0.is_shape_unknown()) return status::unsupported;
    return status::success;
}

status_t infer_exponent_output_shape(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    UNUSED(n);
    auto out0 = logical_tensor_wrapper(outputs[0]);
    if (!out0.is_shape_unknown()) return status::success;
    auto in = logical_tensor_wrapper(inputs[3]);
    auto dims = in.vdims();
    set_shapes_in_range(outputs, 0, 1, dims);
    return status::success;
}

status_t infer_reduce_sum_output_shape(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    UNUSED(n);
    UNUSED(inputs);
    auto out0 = logical_tensor_wrapper(outputs[0]);
    // check if output shape is already known
    if (!out0.is_shape_unknown()) return status::success;

    // When the second input is an empty list,
    // then this operation does nothing, it is an identity.
    // For dims.size() == 0, we set ndims to -1 and can't get
    // vdims. Therefore, this path is not supported.

    // since we don't have an access to the second input data,
    // which contain axis indices, we cannot calculate output shape
    return status::unsupported;
}

} // namespace impl
} // namespace llga

#endif
