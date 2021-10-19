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

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "interface/shape_infer.hpp"

namespace dnnl {
namespace graph {
namespace impl {

/// convert shape to ncx or oix
dims canonicalize(const dims &shape, const std::string &format) {
    dims ret(shape);
    const size_t ndims = shape.size();

    if (ndims <= 2 || "NCX" == format || "OIX" == format) return ret;

    if ("NXC" == format) {
        // NXC -> NCX
        ret[1] = shape[ndims - 1]; // c
        for (size_t i = 2; i < ndims; ++i) {
            ret[i] = shape[i - 1];
        }
    } else if ("XIO" == format) {
        // XIO -> OIX
        ret[0] = shape[ndims - 1]; // oc
        ret[1] = shape[ndims - 2]; // ic
        for (size_t i = 2; i < ndims; ++i) {
            ret[i] = shape[i - 2];
        }
    } else {
        assert(!"invalid format");
    }
    return ret;
}

inline dims ncx2nxc(const dims &shape) {
    const size_t ndims = shape.size();
    if (ndims <= 2) return shape;

    dims ret(shape);
    // x
    for (size_t i = 2; i < ndims; ++i) {
        ret[i - 1] = shape[i];
    }
    // c
    ret[ndims - 1] = shape[1];

    return ret;
}

/// make a dims according to the format. Only for data format ncx or nxc.
inline dims make_data_dims(const std::string &format, const dim_t n,
        const dim_t c, const dims &x) {
    dims ret;
    if (format == "NCX") {
        ret.push_back(n);
        ret.push_back(c);
        ret.insert(ret.end(), x.begin(), x.end());
    } else if (format == "NXC") {
        ret.push_back(n);
        ret.insert(ret.end(), x.begin(), x.end());
        ret.push_back(c);
    } else {
        assert(!"invalid format");
    }

    return ret;
}

/// make a dims according to the format. Only for filter format xio or oix.
inline dims make_filter_dims(const std::string &format, const dim_t i,
        const dim_t o, const dims &x) {
    dims ret;
    if (format == "XIO") {
        ret.insert(ret.begin(), x.begin(), x.end());
        ret.push_back(i);
        ret.push_back(o);
    } else if (format == "OIX") {
        ret.push_back(o);
        ret.push_back(i);
        ret.insert(ret.end(), x.begin(), x.end());
    } else {
        assert(!"invalid format");
    }

    return ret;
}

/// validate the inferred shape with the expected one.
bool validate(const dims &inferred, const dims &expected) {
    if (inferred.size() != expected.size()) { return false; }

    for (size_t i = 0; i < inferred.size(); ++i) {
        if (expected[i] != -1 && inferred[i] != expected[i]) { return false; }
    }
    return true;
}

/// get the dense strides of a given shape
/// eg. (3, 4, 5) -> (20, 5, 1)
inline dims get_dense_strides(const dims &shape) {
    dims strides(shape.size());
    for (auto it = shape.begin(); it < shape.end(); ++it) {
        const auto val = std::accumulate(
                std::next(it), shape.end(), 1, std::multiplies<dim_t>());
        const auto dist = std::distance(shape.begin(), it);
        strides[static_cast<size_t>(dist)] = val;
    }
    return strides;
}

/// shapes of the logical tensors in the vector are known
inline bool every_shape_is_known(const std::vector<logical_tensor_t *> &lts) {
    bool ret = std::all_of(
            lts.cbegin(), lts.cend(), [](const logical_tensor_t *const lt) {
                return !logical_tensor_wrapper(lt).is_shape_unknown();
            });
    return ret;
}

inline bool verify_shapes_in_range(const std::vector<logical_tensor_t *> &lts,
        const size_t begin, const size_t end,
        const std::function<bool(const dims)> &validator) {
    for (size_t idx = begin; idx < end; ++idx) {
        const dims ltx_dims = logical_tensor_wrapper(lts[idx]).vdims();
        if (!validator(ltx_dims)) return false;
    }

    return true;
}

void set_shape_and_strides(logical_tensor_t &lt, const dims &shape) {
    utils::array_copy(lt.dims, shape.data(), shape.size());
    lt.ndims = static_cast<int32_t>(shape.size());

    auto ltw = logical_tensor_wrapper(lt);
    // don't overwrite strides provided by users
    if (ltw.is_strided() && ltw.is_stride_unknown()) {
        const dims strides = get_dense_strides(shape);
        utils::array_copy(lt.layout.strides, strides.data(), strides.size());
    }
}

inline void set_shapes_in_range(const std::vector<logical_tensor_t *> &lts,
        const size_t begin, const size_t end, const dims &shape) {
    for (auto idx = begin; idx < end; ++idx) {
        set_shape_and_strides(*lts[idx], shape);
    }
}

/// infer the padding sizes according auto_pad type
status_t infer_auto_pad(const dim_t in_dim, const dim_t stride,
        const dim_t kernel, const dim_t dilation, const std::string &auto_pad,
        dim_t &pad_begin, dim_t &pad_end, bool is_deconv) {
    if (auto_pad == "VALID") {
        pad_begin = 0;
        pad_end = 0;
    } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        // TODO(xxx): need to improve?
        if (1 != dilation) return status::unsupported;

        dim_t legacy = (in_dim + stride - 1) / stride;
        dim_t pad_needed = is_deconv ? kernel - stride
                                     : (legacy - 1) * stride + kernel - in_dim;
        pad_begin = auto_pad == "SAME_LOWER" ? ((pad_needed + 1) / 2)
                                             : (pad_needed / 2);
        pad_end = pad_needed - pad_begin;
    } else {
        return status::invalid_argument;
    }

    return status::success;
}

/// numpy broadcasting
/// TODO(xxx): 0-D broadcasting?
status_t broadcast(const dims &lhs, const dims &rhs, dims &broadcasted) {
    const size_t lhs_rank = lhs.size();
    const size_t rhs_rank = rhs.size();
    const size_t max_rank = std::max(lhs_rank, rhs_rank);

    broadcasted.resize(max_rank);
    const size_t bl = max_rank - lhs_rank;
    const size_t br = max_rank - rhs_rank;

    for (size_t index = 0; index < max_rank; ++index) {
        dim_t l = 1, r = 1;
        if (index >= bl) l = lhs[index - bl];
        if (index >= br) r = rhs[index - br];
        if (l != r) {
            if (l != 1 && r != 1) return status::invalid_shape;
            broadcasted[index] = (l == 1 ? r : l);
        } else {
            broadcasted[index] = l;
        }
    }

    return status::success;
}

/// This function assumes the size of all vectors are correct. Eg. size of
/// strides/dilations/pads should be the same as spatial size of src_dims and
/// fil_dims. Size of output_dims should be the same as size of src_dims.
inline void infer_conv_ncx_oix(const dims &src_dims, const dims &fil_dims,
        const dims &strides, const dims &dilations, const dims &pads_begin,
        const dims &pads_end, dims &output_dims) {
    output_dims[0] = src_dims[0]; // n
    output_dims[1] = fil_dims[0]; // c
    for (size_t i = 2; i < src_dims.size(); ++i) {
        dim_t padded = src_dims[i] + pads_begin[i - 2] + pads_end[i - 2];
        dim_t dilated = dilations[i - 2] * (fil_dims[i] - 1) + 1;
        output_dims[i] = ((padded - dilated) / strides[i - 2]) + 1;
    }
}

status_t infer_conv_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper(inputs[0]); // src
    auto in1 = logical_tensor_wrapper(inputs[1]); // filter
    auto out0 = logical_tensor_wrapper(outputs[0]); // output

    // get attr value
    const dim_t g = n->get_attr<dim_t>("groups");
    const auto &strides = n->get_attr<dims>("strides");
    const auto &dilations = n->get_attr<dims>("dilations");
    const auto &pads_begin = n->get_attr<dims>("pads_begin");
    const auto &pads_end = n->get_attr<dims>("pads_end");
    std::string fil_fmt = n->get_attr<std::string>("filter_format");
    std::string src_fmt = n->get_attr<std::string>("data_format");

    // check if src channel / groups == weight input channel
    if (in0.get_src_c(src_fmt) / g != in1.get_weight_i(fil_fmt)) {
        return status::invalid_shape;
    }

    // spatial dims
    dims src_sp = in0.get_src_spatial_dims(src_fmt);
    dims fil_sp = in1.get_weight_spatial_dims(fil_fmt);

    // if paddings are empty vectors?
    dims new_pads_begin(pads_begin);
    if (new_pads_begin.empty()) { new_pads_begin.assign(src_sp.size(), 0); }
    dims new_pads_end(pads_end);
    if (new_pads_end.empty()) { new_pads_end.assign(src_sp.size(), 0); }

    // strides and dilations are required and should be correctly provided.
    if (strides.size() != src_sp.size() || dilations.size() != fil_sp.size()
            || new_pads_begin.size() != src_sp.size()
            || new_pads_end.size() != src_sp.size()) {
        return status::invalid_shape;
    }

    if (n->has_attr("auto_pad")
            && n->get_attr<std::string>("auto_pad") != "None") {
        std::string auto_pad = n->get_attr<std::string>("auto_pad");
        // infer auto padding sizes
        for (size_t i = 0; i < src_sp.size(); ++i) {
            infer_auto_pad(src_sp[i], strides[i], fil_sp[i], dilations[i],
                    auto_pad, new_pads_begin[i], new_pads_end[i]);
        }

        n->set_attr("pads_begin", new_pads_begin);
        n->set_attr("pads_end", new_pads_end);
    }

    // infer output shape
    dims output_dims(in0.vdims());
    infer_conv_ncx_oix(canonicalize(in0.vdims(), src_fmt),
            canonicalize(in1.vdims(), fil_fmt), strides, dilations,
            new_pads_begin, new_pads_end, output_dims);
    // output shape should have the same format as input data
    if ("NXC" == src_fmt) { output_dims = ncx2nxc(output_dims); }

    if (out0.ndims() != -1) {
        if (!validate(output_dims, out0.vdims())) {
            return status::invalid_shape;
        }
    }

    set_shape_and_strides(*outputs[0], output_dims);
    return status::success;
}

status_t infer_conv_bprop_data_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper(inputs[0]); // data
    auto in1 = logical_tensor_wrapper(inputs[1]); // filter
    auto out0 = logical_tensor_wrapper(outputs[0]); // output

    // get attr value
    const dim_t g = n->get_attr<dim_t>("groups");
    const auto &strides = n->get_attr<dims>("strides");
    const auto &dilations = n->get_attr<dims>("dilations");
    const auto &pads_begin = n->get_attr<dims>("pads_begin");
    const auto &pads_end = n->get_attr<dims>("pads_end");
    std::string fil_fmt = n->get_attr<std::string>("filter_format");
    std::string src_fmt = n->get_attr<std::string>("data_format");

    // check if diff_dst channel == weight output channel.
    // Since the input of conv_bwd_data op is diff_dst, which has the same shape
    // with conv fwd op's dst, so it's channel should be equal to weight's o
    // channel.
    if (in0.get_src_c(src_fmt) != in1.get_weight_o(fil_fmt)) {
        return status::invalid_shape;
    }

    dims src_sp = in0.get_src_spatial_dims(src_fmt);
    dims fil_sp = in1.get_weight_spatial_dims(fil_fmt);

    // if paddings are empty vectors?
    dims new_pads_begin(pads_begin);
    if (new_pads_begin.empty()) { new_pads_begin.assign(src_sp.size(), 0); }
    dims new_pads_end(pads_end);
    if (new_pads_end.empty()) { new_pads_end.assign(src_sp.size(), 0); }

    // strides and dilations are required and should be correctly provided.
    if (strides.size() != src_sp.size() || dilations.size() != fil_sp.size()
            || new_pads_begin.size() != src_sp.size()
            || new_pads_end.size() != src_sp.size()) {
        return status::invalid_shape;
    }

    dims output_padding(src_sp.size(), 0);
    if (n->has_attr("output_padding")) {
        output_padding = n->get_attr<dims>("output_padding");
    }

    if (n->has_attr("auto_pad")
            && n->get_attr<std::string>("auto_pad") != "None") {
        std::string auto_pad = n->get_attr<std::string>("auto_pad");

        // infer auto_pad
        for (size_t i = 0; i < src_sp.size(); ++i) {
            infer_auto_pad(src_sp[i], strides[i], fil_sp[i], dilations[i],
                    auto_pad, new_pads_begin[i], new_pads_end[i]);
        }

        n->set_attr("pads_begin", new_pads_begin);
        n->set_attr("pads_end", new_pads_end);
    }

    dims output_sp;
    // third input - output_shape is optional.
    // When output_shape is specified pads_begin and pads_end are ignored,
    // and auto_pad defines how to distribute padding amount around the tensor.
    if (inputs.size() == 3 && logical_tensor_wrapper(inputs[2]).ndims() != -1) {
        // Since we have no access to the data of the third input
        // (output_shape), we cannot set output spatial shape.
        return status::unsupported;
    } else {
        for (size_t i = 0; i < src_sp.size(); ++i) {
            dim_t padded
                    = output_padding[i] - new_pads_begin[i] - new_pads_end[i];
            dim_t dilated = dilations[i] * (fil_sp[i] - 1) + 1;
            output_sp.push_back(
                    strides[i] * (src_sp[i] - 1) + dilated + padded);
        }
    }

    const dims out0_shape = make_data_dims(
            src_fmt, in0.get_src_n(), in1.get_weight_i(fil_fmt) * g, output_sp);

    set_shape_and_strides(*outputs[0], out0_shape);

    return status::success;
}

status_t infer_conv_bprop_filters_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper(inputs[0]);
    auto in2 = logical_tensor_wrapper(inputs[2]);
    auto out0 = logical_tensor_wrapper(outputs[0]);

    // get attr value
    const auto &strides = n->get_attr<dims>("strides");
    const auto &pads_begin = n->get_attr<dims>("pads_begin");
    const auto &pads_end = n->get_attr<dims>("pads_end");
    const auto &dilations = n->get_attr<dims>("dilations");
    std::string fil_fmt = n->get_attr<std::string>("filter_format");
    std::string src_fmt = n->get_attr<std::string>("data_format");

    const dims src_dims = in0.vdims();
    const dims output_delta_dims = in2.vdims();

    dims src_sp = in0.get_src_spatial_dims(src_fmt);
    dims output_delta_sp = in2.get_src_spatial_dims(src_fmt);

    // if paddings are empty vectors?
    dims new_pads_begin(pads_begin);
    if (new_pads_begin.empty()) { new_pads_begin.assign(src_sp.size(), 0); }
    dims new_pads_end(pads_end);
    if (new_pads_end.empty()) { new_pads_end.assign(src_sp.size(), 0); }

    // strides and dilations are required and should be correctly provided.
    if (strides.size() != src_sp.size() || dilations.size() != src_sp.size()
            || new_pads_begin.size() != src_sp.size()
            || new_pads_end.size() != src_sp.size()) {
        return status::invalid_shape;
    }

    if (n->has_attr("auto_pad")
            && n->get_attr<std::string>("auto_pad") != "None") {
        std::string auto_pad = n->get_attr<std::string>("auto_pad");

        if (auto_pad == "VALID") {
            size_t rank = src_sp.size();
            new_pads_begin.assign(rank, 0);
            new_pads_end.assign(rank, 0);
        } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
            // Since we have no access to the data of the second input
            // (weights_shape), we cannot calculate auto pads.
            return status::unsupported;
        }

        n->set_attr("pads_begin", new_pads_begin);
        n->set_attr("pads_end", new_pads_end);
    }

    // Since we have no access to the data of the second input (weights_shape),
    // we have to get weights spatial shape using another way.
    // To do that we use transformed convolution output size formula.
    dims fil_sp;
    for (size_t i = 0; i < src_sp.size(); ++i) {
        dim_t padded = src_sp[i] + new_pads_begin[i] + new_pads_end[i];
        dim_t strided = strides[i] * (output_delta_sp[i] - 1);
        fil_sp.push_back(((padded - strided - 1) / dilations[i]) + 1);
    }

    const dims out0_shape = make_filter_dims(
            fil_fmt, in0.get_src_c(src_fmt), in2.get_src_c(src_fmt), fil_sp);

    if (out0.ndims() != -1) {
        if (!validate(out0_shape, out0.vdims())) {
            return status::invalid_shape;
        }
    }

    set_shape_and_strides(*outputs[0], out0_shape);

    return status::success;
}

status_t infer_convtranspose_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper(inputs[0]);
    auto in1 = logical_tensor_wrapper(inputs[1]);
    auto out0 = logical_tensor_wrapper(outputs[0]);

    // get attr value
    const dim_t g = n->get_attr<dim_t>("groups");
    const auto &strides = n->get_attr<dims>("strides");
    const auto &dilations = n->get_attr<dims>("dilations");
    const auto &pads_begin = n->get_attr<dims>("pads_begin");
    const auto &pads_end = n->get_attr<dims>("pads_end");
    std::string fil_fmt = n->get_attr<std::string>("filter_format");
    std::string src_fmt = n->get_attr<std::string>("data_format");

    if (!out0.is_shape_unknown()) {
        // check if dst channel / groups == weight output channel
        if (out0.get_src_c(src_fmt) / g != in1.get_weight_o(fil_fmt)) {
            return status::invalid_shape;
        }
    }

    dims src_sp = in0.get_src_spatial_dims(src_fmt);
    dims fil_sp = in1.get_weight_spatial_dims(fil_fmt);

    // if paddings are empty vectors?
    dims new_pads_begin(pads_begin);
    if (new_pads_begin.empty()) { new_pads_begin.assign(src_sp.size(), 0); }
    dims new_pads_end(pads_end);
    if (new_pads_end.empty()) { new_pads_end.assign(src_sp.size(), 0); }

    // strides and dilations are required and should be correctly provided.
    if (strides.size() != src_sp.size() || dilations.size() != fil_sp.size()
            || new_pads_begin.size() != src_sp.size()
            || new_pads_end.size() != src_sp.size()) {
        return status::invalid_shape;
    }

    dims output_padding(src_sp.size(), 0);
    if (n->has_attr("output_padding")) {
        output_padding = n->get_attr<dims>("output_padding");
    }

    if (n->has_attr("auto_pad")
            && n->get_attr<std::string>("auto_pad") != "None") {
        std::string auto_pad = n->get_attr<std::string>("auto_pad");

        // infer auto_pad
        for (size_t i = 0; i < src_sp.size(); ++i) {
            infer_auto_pad(src_sp[i], strides[i], fil_sp[i], dilations[i],
                    auto_pad, new_pads_begin[i], new_pads_end[i], true);
        }

        n->set_attr("pads_begin", new_pads_begin);
        n->set_attr("pads_end", new_pads_end);
    }

    dims output_sp;
    for (size_t i = 0; i < src_sp.size(); ++i) {
        dim_t padded = output_padding[i] - new_pads_begin[i] - new_pads_end[i];
        dim_t dilated = dilations[i] * (fil_sp[i] - 1) + 1;
        output_sp.push_back(strides[i] * (src_sp[i] - 1) + dilated + padded);
    }

    const dims out0_shape = make_data_dims(
            src_fmt, in0.get_src_n(), in1.get_weight_o(fil_fmt) * g, output_sp);

    if (out0.ndims() != -1) {
        if (!validate(out0_shape, out0.vdims())) {
            return status::invalid_shape;
        }
    }

    set_shape_and_strides(*outputs[0], out0_shape);

    return status::success;
}

status_t infer_pool_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper(inputs[0]);
    auto out0 = logical_tensor_wrapper(outputs[0]);

    // get attr value
    const dims &strides = n->get_attr<dims>("strides");
    const dims &kernel = n->get_attr<dims>("kernel");
    const dims &pads_begin = n->get_attr<dims>("pads_begin");
    const dims &pads_end = n->get_attr<dims>("pads_end");
    std::string rounding_type = "floor";
    if (n->has_attr("rounding_type")) {
        rounding_type = n->get_attr<std::string>("rounding_type");
    }
    std::string src_format = n->get_attr<std::string>("data_format");

    dims dilations(kernel.size(), 1);
    if (n->has_attr("dilations")) {
        auto dilations_tmp = n->get_attr<dims>("dilations");
        if (dilations_tmp.size() != dilations.size()) {
            return status::invalid_argument;
        } else {
            dilations = dilations_tmp;
        }
    }

    const dims src_dims = in0.vdims();

    dims src_sp = in0.get_src_spatial_dims(src_format);

    // if paddings are empty vectors?
    dims new_pads_begin(pads_begin);
    if (new_pads_begin.empty()) { new_pads_begin.assign(src_sp.size(), 0); }
    dims new_pads_end(pads_end);
    if (new_pads_end.empty()) { new_pads_end.assign(src_sp.size(), 0); }
    if (n->has_attr("auto_pad")
            && n->get_attr<std::string>("auto_pad") != "None") {
        std::string auto_pad = n->get_attr<std::string>("auto_pad");
        // infer auto_pad
        for (size_t i = 0; i < src_sp.size(); ++i) {
            infer_auto_pad(src_sp[i], strides[i], kernel[i], dilations[i],
                    auto_pad, new_pads_begin[i], new_pads_end[i]);
        }
        n->set_attr("pads_begin", new_pads_begin);
        n->set_attr("pads_end", new_pads_end);
    }

    dims output_sp;
    for (size_t i = 0; i < src_sp.size(); ++i) {
        dim_t padded = src_sp[i] + new_pads_begin[i] + new_pads_end[i];
        dim_t dilated = dilations[i] * (kernel[i] - 1) + 1;
        dim_t out_value;
        if (rounding_type == "ceil") {
            out_value = (padded - dilated - 1) / strides[i] + 2;
        } else {
            out_value = (padded - dilated) / strides[i] + 1;
        }
        output_sp.push_back(out_value);
    }

    dims out_shape = make_data_dims(
            src_format, in0.get_src_n(), in0.get_src_c(src_format), output_sp);
    if (out0.ndims() != -1) {
        if (!validate(out_shape, out0.vdims())) {
            return status::invalid_shape;
        }
    }

    set_shape_and_strides(*outputs[0], out_shape);
    return status::success;
}

status_t infer_matmul_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper(inputs[0]);
    auto in1 = logical_tensor_wrapper(inputs[1]);
    auto out0 = logical_tensor_wrapper(outputs[0]);

    // check if output shape is already known
    if (!out0.is_shape_unknown()) return status::success;
    // get attr value
    bool transpose_a = false;
    if (n->has_attr("transpose_a")) {
        transpose_a = n->get_attr<bool>("transpose_a");
    }
    bool transpose_b = false;
    if (n->has_attr("transpose_b")) {
        transpose_b = n->get_attr<bool>("transpose_b");
    }
    const dims input0_dims = in0.vdims();
    const dims input1_dims = in1.vdims();
    size_t input0_rank = input0_dims.size();
    size_t input1_rank = input1_dims.size();
    dims updated_input0(input0_dims);
    dims updated_input1(input1_dims);
    if (transpose_a && input0_rank > 1) {
        std::swap(updated_input0[input0_rank - 2],
                updated_input0[input0_rank - 1]);
    }
    if (transpose_b && input1_rank > 1) {
        std::swap(updated_input1[input1_rank - 2],
                updated_input1[input1_rank - 1]);
    }

    dims inferred_out_shape;
    if (input0_rank == 1 && input1_rank == 1) {
        if (updated_input0 != updated_input1) {
            // matmul: incompatible arg shapes
            return status::invalid_shape;
        }
        // example: input0={1,1}, input1={1,1}, output={2}
        inferred_out_shape = {1};
    } else if (input0_rank == 1) {
        if (updated_input0[0] != updated_input1[input1_rank - 2]) {
            // matmul: incompatible arg shapes
            return status::invalid_shape;
        }
        // example: input0 shape {3}, input1 shape {2,3,4},
        // output shape {2,4}
        updated_input1.erase(
                updated_input1.begin() + static_cast<dim_t>(input1_rank) - 2);
        inferred_out_shape = updated_input1;
    } else if (input1_rank == 1) {
        if (updated_input1[0] != updated_input0[input0_rank - 1]) {
            // matmul: incompatible arg shapes
            return status::invalid_shape;
        }
        // example: input0 shape {2,3,4}, input1 shape {4},
        // output shape {2,3}
        updated_input0.erase(
                updated_input0.begin() + static_cast<dim_t>(input0_rank) - 1);
        inferred_out_shape = updated_input0;
    } else if (input0_rank == 2 && input1_rank == 2) {
        if (updated_input0[1] != updated_input1[0]) {
            // matmul: incompatible arg shapes
            return status::invalid_shape;
        }
        // example: input0 shape {1, 3}, input1 shape {3, 2},
        // output shape {1,2}
        inferred_out_shape = {updated_input0[0], updated_input1[1]};
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
        status_t ret = broadcast(
                input0_batch_dims, input1_batch_dims, inferred_out_shape);
        if (ret != status::success) { return ret; }
        inferred_out_shape.push_back(updated_input0[input0_rank - 2]);
        inferred_out_shape.push_back(updated_input1[input1_rank - 1]);
    }

    if (out0.ndims() != -1) {
        if (!validate(inferred_out_shape, out0.vdims())) {
            return status::invalid_shape;
        }
    }

    set_shape_and_strides(*outputs[0], inferred_out_shape);
    return status::success;
}

status_t infer_identity_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out0 = logical_tensor_wrapper(outputs[0]);
    auto in0 = logical_tensor_wrapper(inputs[0]);
    if (!out0.is_shape_unknown()) return status::success;

    // check if partial set shape aligns with inferred shape
    if (out0.ndims() != -1) {
        if (!validate(in0.vdims(), out0.vdims())) {
            return status::invalid_shape;
        }
    }

    // We should compute output dense strides instead of
    // directly copying input strides to it
    set_shape_and_strides(*outputs[0], in0.vdims());
    UNUSED(n);
    return status::success;
}

status_t identity_output_shape_on_pos(op_t *n,
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

status_t infer_bias_backprop_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out = logical_tensor_wrapper(outputs[0]);
    if (!out.is_shape_unknown()) return status::success;

    auto in = logical_tensor_wrapper(inputs[0]);
    dims input_dims = in.vdims();
    if (input_dims.size() < 4) {
        // bias add backprop: input should have at least 4 dims.
        return status::invalid_shape;
    }

    std::string fmt = n->has_attr("data_format")
            ? n->get_attr<std::string>("data_format")
            : "NXC";

    const auto channels = in.get_src_c(fmt);
    dims new_out_dims = {channels};

    set_shape_and_strides(*outputs[0], new_out_dims);
    return status::success;
}

status_t infer_bias_add_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out = logical_tensor_wrapper(outputs[0]);
    if (!out.is_shape_unknown()) return status::success;

    auto in = logical_tensor_wrapper(inputs[0]);
    dims input_dims = in.vdims();
    if (input_dims.size() < 4) {
        // bias add: input should have at least 4 dims.
        return status::invalid_shape;
    }
    auto bias = logical_tensor_wrapper(inputs[1]);
    dims bias_dims = bias.vdims();
    if (bias_dims.size() != 1) {
        // bias add: bias should have exactly 1 dimension.
        return status::invalid_shape;
    }

    // following the spec of convolution, nxc as default format
    std::string fmt = n->has_attr("data_format")
            ? n->get_attr<std::string>("data_format")
            : "NXC";

    const auto channels = in.get_src_c(fmt);
    if (bias_dims[0] != channels) {
        // bias add: bias size should match input channels size.
        return status::invalid_shape;
    }

    return infer_identity_output_shape(n, inputs, outputs);
}

status_t infer_norm_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto status = infer_identity_output_shape(n, inputs, outputs);
    if (status != status::success) return status;

    const bool keep_stats = n->has_attr("keep_stats")
            ? n->get_attr<bool>("keep_stats")
            : false;
    if (!keep_stats) return status::success;

    auto in0 = logical_tensor_wrapper(inputs[0]);
    const dims input0_dims = in0.vdims();

    const dim_t begin_norm_axis = n->has_attr("begin_norm_axis")
            ? n->get_attr<dim_t>("begin_norm_axis")
            : -1;

    auto out1 = logical_tensor_wrapper(outputs[1]);
    auto out2 = logical_tensor_wrapper(outputs[2]);
    dims output_dims(input0_dims);

    auto norm_starting_position
            = begin_norm_axis >= 0 ? output_dims.begin() : output_dims.end();

    output_dims.erase(
            norm_starting_position + begin_norm_axis, output_dims.end());

    // check if output shape is already known
    if (out1.is_shape_unknown()) {
        set_shape_and_strides(*outputs[1], output_dims);
    }

    // check if output shape is already known
    if (out2.is_shape_unknown()) {
        set_shape_and_strides(*outputs[2], output_dims);
    }
    return status::success;
}

status_t infer_norm_bprop_output_shape(op_t *n,
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

status_t infer_elemwise_arithmetic_output_shape(op_t *n,
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

    dims input0_dims = in0.vdims();
    dims input1_dims = in1.vdims();
    dims inferred_out_shape;
    if (shapes_should_match) {
        if (input0_dims != input1_dims) {
            // add: incompatible input shapes (auto_broadcast=none)
            return status::invalid_shape;
        }
        inferred_out_shape = input0_dims;
    } else {
        status_t ret = broadcast(input0_dims, input1_dims, inferred_out_shape);
        if (ret != status::success) { return ret; }
    }
    // check if partial set shape aligns with inferred shape
    if (out0.ndims() != -1) {
        if (!validate(inferred_out_shape, out0.vdims())) {
            return status::invalid_shape;
        }
    }

    set_shape_and_strides(*outputs[0], inferred_out_shape);
    return status::success;
}

status_t infer_bn_fwd_train_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using cvec_int64 = const std::vector<int64_t>;

    if (every_shape_is_known(outputs)) return status::success;

    const auto in = logical_tensor_wrapper(inputs[0]);
    cvec_int64 input_dims = in.vdims();
    if (input_dims.size() < 4) return status::invalid_shape;

    std::string fmt = n->has_attr("data_format")
            ? n->get_attr<std::string>("data_format")
            : "NXC";

    const auto channels = in.get_src_c(fmt);
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

status_t infer_bn_bwd_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using cvec_int64 = const std::vector<int64_t>;

    if (every_shape_is_known(outputs)) return status::success;

    const auto in = logical_tensor_wrapper(inputs[0]);
    cvec_int64 input_dims = in.vdims();
    const auto out_delta = logical_tensor_wrapper(inputs[1]);
    cvec_int64 out_delta_dims = out_delta.vdims();
    if (input_dims.size() < 4 || out_delta_dims.size() < 4)
        return status::invalid_shape;

    std::string fmt = n->has_attr("data_format")
            ? n->get_attr<std::string>("data_format")
            : "NXC";

    const auto channels = in.get_src_c(fmt);
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

status_t infer_concat_output_shape(op_t *n,
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
    if (data_type != out0.data_type()) return status::unsupported;

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

    std::vector<int64_t> infered_out_shape(dims, dims + ndims);
    infered_out_shape[axis] = sum;
    set_shape_and_strides(*outputs[0], infered_out_shape);
    return status::success;
}

status_t infer_unsupported_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    UNUSED(n);
    UNUSED(inputs);
    auto out0 = logical_tensor_wrapper(outputs[0]);
    if (out0.is_shape_unknown()) return status::unsupported;
    return status::success;
}

/// Shape inference function for PowBackpropExponent
status_t infer_exponent_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    UNUSED(n);
    auto out0 = logical_tensor_wrapper(outputs[0]); // exponent_delta
    if (!out0.is_shape_unknown()) return status::success;

    auto in = logical_tensor_wrapper(inputs[3]); // exponent
    auto dims = in.vdims();
    set_shape_and_strides(*outputs[0], dims);
    return status::success;
}

status_t infer_reduce_sum_output_shape(op_t *n,
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
} // namespace graph
} // namespace dnnl
