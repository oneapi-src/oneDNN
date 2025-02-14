/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#include <iterator>
#include <sstream>

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "utils/fill.hpp"
#include "utils/parser.hpp" // get_substr()

#include "flex_rewrite.hpp"
#include "parser.hpp"

namespace graph {

void flex_rewrite::rewrite_linked_shape_and_attr(deserialized_graph &dgraph) {
    for (auto &aop : dgraph.ops_) {
        if (aop.kind_ == "DynamicDequantize") {
            auto &attr = aop.attrs_;
            if (attr.find("qtype") == attr.end()
                    || attr["qtype"].str_value_ != "per_group")
                continue;
            if (attr.find("group_shape") == attr.end()) {
                BENCHDNN_PRINT(0,
                        "Error: missed `group-shape` attribute for "
                        "per-group quantization for op with id=\'%zu\'\n",
                        aop.id_);
                SAFE_V(FAIL);
            }

            bool input_shape_rewrite = std::any_of(aop.in_lts_.begin(),
                    aop.in_lts_.end(), [&](const deserialized_lt &in_lt) {
                        return in_shapes_.count(in_lt.id_)
                                && in_shapes_[in_lt.id_] != "default";
                    });
            bool group_shape_rewrite = op_attrs_.count(aop.id_)
                    && parse_attrs(op_attrs_.at(aop.id_)).count("group_shape");

            auto &group_shape = attr["group_shape"].s64_vector_;
            const auto &src_lt = aop.in_lts_[0];

            if (!group_shape_rewrite && !input_shape_rewrite) continue;
            if (input_shape_rewrite && group_shape_rewrite) {
                // if both input shapes and group_shape are provided, check if
                // the new shape are valid.
                auto &scale_lt = aop.in_lts_[1];
                if (src_lt.shape_.size() != scale_lt.shape_.size()) {
                    BENCHDNN_PRINT(0,
                            "Error: the ndims of scale tensor should align "
                            "with the ndims of input tensor for op with "
                            "id=\'%zu\'\n",
                            aop.id_);
                    SAFE_V(FAIL);
                }

                if (src_lt.shape_.size() != group_shape.size()) {
                    BENCHDNN_PRINT(0,
                            "Error: the ndims of `group-shape` attribute "
                            "should align with the input ndims for op with "
                            "id=\'%zu\'\n",
                            aop.id_);
                    SAFE_V(FAIL);
                }

                for (size_t idx = 0; idx < src_lt.shape_.size(); ++idx) {
                    if (src_lt.shape_[idx]
                            != scale_lt.shape_[idx] * group_shape[idx]) {
                        BENCHDNN_PRINT(0,
                                "Error: the input shape should equal with the "
                                "product of corresponding dimension of scale "
                                "shape and group shape, input shape: %lld, "
                                "scale shape: %lld, group shape: %lld\n",
                                (long long)src_lt.shape_[idx],
                                (long long)scale_lt.shape_[idx],
                                (long long)group_shape[idx]);
                        SAFE_V(FAIL);
                    }
                }

                if (aop.in_lts_.size() > 2) {
                    auto &zp_lt = aop.in_lts_[2];
                    if (scale_lt.shape_.size() != zp_lt.shape_.size()) {
                        BENCHDNN_PRINT(0,
                                "Error: the ndims of scale tensor should align "
                                "with the ndims of zero-point tensor for op "
                                "with id=\'%zu\'\n",
                                aop.id_);
                        SAFE_V(FAIL);
                    }

                    for (size_t idx = 0; idx < scale_lt.shape_.size(); ++idx) {
                        if (scale_lt.shape_[idx] != zp_lt.shape_[idx]) {
                            BENCHDNN_PRINT(0,
                                    "Error: the shape of zero-point tensor "
                                    "should be the same as the shape of scale "
                                    "tensor for op with id=\'%zu\'\n",
                                    aop.id_);
                            SAFE_V(FAIL);
                        }
                    }
                }
                continue;
            }

            if (group_shape_rewrite && !input_shape_rewrite) {
                // if user only rewrite group shape attribute, update the scale
                // shape and zps shape (if available) accordingly.
                dims_t new_group_quant_scale_zps_dims(
                        src_lt.shape_.size(), DNNL_GRAPH_UNKNOWN_DIM);
                if (src_lt.shape_.size() != group_shape.size()) {
                    BENCHDNN_PRINT(0,
                            "Error: the ndims of `group-shape` attribute "
                            "should align with the input ndims for op with "
                            "id=\'%zu\'\n",
                            aop.id_);
                    SAFE_V(FAIL);
                }

                for (size_t idx = 0; idx < src_lt.shape_.size(); ++idx) {
                    if (src_lt.shape_[idx] % group_shape[idx] != 0) {
                        BENCHDNN_PRINT(0,
                                "Error: the dimension of `group-shape` "
                                "attribute should be divisible by the "
                                "corresponding dimensions of the input shape, "
                                "group shape: %lld, input shape: %lld\n",
                                (long long)group_shape[idx],
                                (long long)src_lt.shape_[idx]);
                        SAFE_V(FAIL);
                    }
                    new_group_quant_scale_zps_dims[idx]
                            = src_lt.shape_[idx] / group_shape[idx];
                }

                auto &scale_lt = aop.in_lts_[1];
                scale_lt.shape_ = new_group_quant_scale_zps_dims;
                scale_lt.stride_ = memory_tag2strides(
                        scale_lt.shape_, dgraph.lt_2_mtag_[scale_lt.id_]);
                if (aop.in_lts_.size() > 2) {
                    auto &zp_lt = aop.in_lts_[2];
                    zp_lt.shape_ = new_group_quant_scale_zps_dims;
                    zp_lt.stride_ = memory_tag2strides(
                            zp_lt.shape_, dgraph.lt_2_mtag_[zp_lt.id_]);
                }
            } else if (input_shape_rewrite && !group_shape_rewrite) {
                // if user only rewrites input shapes, update the group-shape
                // attribute accordingly.
                auto &scale_lt = aop.in_lts_[1];
                if (src_lt.shape_.size() != scale_lt.shape_.size()) {
                    BENCHDNN_PRINT(0,
                            "Error: the ndims of scale tensor should align "
                            "with the ndims of input tensor for op with "
                            "id=\'%zu\'\n",
                            aop.id_);
                    SAFE_V(FAIL);
                }

                std::vector<int64_t> new_group_shape(src_lt.shape_.size(), 1);
                for (size_t idx = 0; idx < src_lt.shape_.size(); ++idx) {
                    if (src_lt.shape_[idx] % scale_lt.shape_[idx] != 0) {
                        BENCHDNN_PRINT(0,
                                "Error: the dimension of scale  shape should "
                                "be divisible by the corresponding dimensions "
                                "of the input  shape, scale shape: %lld, input "
                                "shape: %lld\n",
                                (long long)scale_lt.shape_[idx],
                                (long long)src_lt.shape_[idx]);
                        SAFE_V(FAIL);
                    }
                    new_group_shape[idx]
                            = src_lt.shape_[idx] / scale_lt.shape_[idx];
                }

                group_shape = new_group_shape;
            }
        }
    }
}

void flex_rewrite::rewrite(deserialized_graph &dgraph) {
    bool change_stride = false;
    inports_shape_rewrite(dgraph, change_stride);
    if (!(op_attrs_.size() == 1 && op_attrs_.count(0)
                && op_attrs_.at(0) == "default")) {
        op_attrs_rewrite(dgraph);
    }
    infer_output_shape(dgraph, change_stride);
    quantized_graph_rewrite(dgraph);
    graph_attrs_rewrite(dgraph);
    rewrite_linked_shape_and_attr(dgraph);
    dt_rewrite(dgraph);
    dt_map_rewrite(dgraph);
}

void flex_rewrite::split_ncx(const std::string &data_format, dims_t &in,
        int64_t &n, int64_t &c, dims_t &x) const {
    x.clear();
    n = in[0];
    if (data_format == "NCX") {
        c = in[1];
        for (size_t i = 2; i < in.size(); i++) {
            x.emplace_back(in[i]);
        }
    } else { // NXC
        for (size_t i = 1; i < in.size() - 1; i++) {
            x.emplace_back(in[i]);
        }
        c = in[in.size() - 1];
    }
}

void flex_rewrite::merge_ncx(const std::string &data_format, dims_t &out,
        int64_t n, int64_t c, const dims_t &x) const {
    out.clear();
    out.push_back(n);
    if (data_format == "NCX") {
        out.push_back(c);
        for (size_t i = 0; i < x.size(); i++) {
            out.push_back(x[i]);
        }
    } else { // NXC
        for (size_t i = 0; i < x.size(); i++) {
            out.push_back(x[i]);
        }
        out.push_back(c);
    }
}

void flex_rewrite::split_oix(const std::string &data_format, dims_t &in,
        dims_t &oi, dims_t &x) const {
    x.clear();
    if (data_format == "OIX" || data_format == "IOX") {
        for (size_t i = 2; i < in.size(); i++) {
            x.push_back(in[i]);
        }
        if (data_format == "OIX") {
            oi = {in[0], in[1]};
        } else { //IOX
            oi = {in[1], in[0]};
        }

    } else if (data_format == "XIO" || data_format == "XOI") {
        for (size_t i = 0; i < in.size() - 2; i++) {
            x.push_back(in[i]);
        }
        if (data_format == "XIO") {
            oi = {in[in.size() - 1], in[in.size() - 2]};
        } else { // XOI
            oi = {in[in.size() - 2], in[in.size() - 1]};
        }
    }
}

template <typename T>
std::string stdvec2string(const std::vector<T> &v) {
    std::string s;
    if (v.empty()) return s;

    s.append("[");
    const size_t sz = v.size() - 1;
    for (size_t i = 0; i < sz; i++) {
        s.append(std::to_string(v[i])).append(", ");
    }
    s.append(std::to_string(v[sz])).append("]");
    return s;
}

void flex_rewrite::broadcast(const dims_t &x, const dims_t &y, dims_t &z,
        const std::string &x_str, const std::string &y_str) const {
    const size_t x_rank = x.size();
    const size_t y_rank = y.size();
    const size_t max_rank = std::max(x_rank, y_rank);
    z.resize(max_rank);
    const size_t bx = max_rank - x_rank;
    const size_t by = max_rank - y_rank;
    for (size_t i = 0; i < max_rank; ++i) {
        int64_t l = 1, r = 1;
        if (i >= bx) l = x[i - bx];
        if (i >= by) r = y[i - by];
        if (l != r) {
            if (l != 1 && r != 1) {
                BENCHDNN_PRINT(0,
                        "Error: batched dimensions \'%lld\' from \'%s\' and "
                        "\'%lld\' from \'%s\' are not consistent. They should "
                        "be equal to each other or one of them should be equal "
                        "to 1.\n",
                        (long long)l, x_str.c_str(), (long long)r,
                        y_str.c_str());
                SAFE_V(FAIL);
            }
            z[i] = (l == 1 ? r : l);
        } else {
            // Batch sizes are equal, use it as a final value.
            z[i] = l;
        }
    }
}

void flex_rewrite::cal_pads(dims_t &pads_begin, dims_t &pads_end,
        const deserialized_op &aop, const dims_t &spatial_dims,
        const dims_t &strides, const dims_t &kernel, bool deconv) const {
    pads_begin.clear();
    pads_end.clear();

    pads_begin = aop.attrs_.at("pads_begin").s64_vector_;
    pads_end = aop.attrs_.at("pads_end").s64_vector_;
    if (pads_begin.empty()) { pads_begin.assign(spatial_dims.size(), 0); }
    if (pads_end.empty()) { pads_end.assign(spatial_dims.size(), 0); }
    if (aop.attrs_.find("auto_pad") != aop.attrs_.end()
            && aop.attrs_.at("auto_pad").str_value_ != "None") {
        const std::string &auto_pad = aop.attrs_.at("auto_pad").str_value_;
        if (auto_pad == "VALID") {
            for (size_t i = 0; i < spatial_dims.size(); i++) {
                pads_begin[i] = 0;
                pads_end[i] = 0;
            }
        } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
            // total padding is the same for both conditions
            for (size_t i = 0; i < spatial_dims.size(); i++) {
                auto legacy = (spatial_dims[i] + strides[i] - 1) / strides[i];
                auto pad_needed = deconv ? kernel[i] - strides[i]
                                         : (legacy - 1) * strides[i] + kernel[i]
                                - spatial_dims[i];
                if (pad_needed >= 0) {
                    pads_begin[i] = auto_pad == "SAME_LOWER"
                            ? ((pad_needed + 1) / 2)
                            : (pad_needed / 2);
                    pads_end[i] = pad_needed - pads_begin[i];
                }
            }
        } else {
            fprintf(stderr, "graph: invalid arguments for `auto_pad`!\n");
            SAFE_V(FAIL);
        }
    }
}

void flex_rewrite::infer_output_shape(
        deserialized_graph &dgraph, bool change_stride) {
    auto &gi = dgraph.graph_tensors_;
    for (auto &aop : dgraph.ops_) {
        auto kind = opstr2kind(aop.kind_);
        size_t in0, in1, out0;
        int64_t n, c, axis, sum, groups, in_size, out_size, use_oi = 0;
        dims_t strides, kernel, pads_begin, pads_end, dilations, spatial_dims,
                output_padding;
        dims_t adims, x, y, oi;
        std::string data_format, weights_format, auto_broadcast;
        bool floor, special_zero;
        // set default value for some attributes
        if (aop.attrs_.find("data_format") != aop.attrs_.end()
                && aop.attrs_["data_format"].str_value_ == "NCX") {
            data_format = "NCX";
        } else {
            data_format = "NXC";
        }
        if (aop.attrs_.find("auto_broadcast") != aop.attrs_.end()
                && aop.attrs_["auto_broadcast"].str_value_ == "none") {
            auto_broadcast = "none";
        } else {
            auto_broadcast = "numpy";
        }
        if (aop.attrs_.find("weights_format") != aop.attrs_.end()) {
            weights_format = aop.attrs_["weights_format"].str_value_;
        } else if (aop.kind_.find("ConvTranspose") != std::string::npos) {
            // ConvTanspose fwd/bwd ops
            weights_format = "XOI";
        } else {
            // Conv fwd/bwd ops
            weights_format = "XIO";
        }
        if (aop.attrs_.find("groups") != aop.attrs_.end()) {
            groups = aop.attrs_["groups"].s64_value_;
        } else {
            groups = 1;
        }
        switch (kind) {
            // infer_identity_output_shape
            case dnnl::graph::op::kind::Abs:
            case dnnl::graph::op::kind::AbsBackward:
            case dnnl::graph::op::kind::BatchNormInference:
            case dnnl::graph::op::kind::Clamp:
            case dnnl::graph::op::kind::ClampBackward:
            case dnnl::graph::op::kind::Dequantize:
            case dnnl::graph::op::kind::DynamicDequantize:
            case dnnl::graph::op::kind::DynamicQuantize:
            case dnnl::graph::op::kind::Elu:
            case dnnl::graph::op::kind::EluBackward:
            case dnnl::graph::op::kind::Exp:
            case dnnl::graph::op::kind::GELU:
            case dnnl::graph::op::kind::GELUBackward:
            case dnnl::graph::op::kind::GenIndex:
            case dnnl::graph::op::kind::GreaterEqual:
            case dnnl::graph::op::kind::HardSigmoid:
            case dnnl::graph::op::kind::HardSigmoidBackward:
            case dnnl::graph::op::kind::HardSwish:
            case dnnl::graph::op::kind::HardSwishBackward:
            case dnnl::graph::op::kind::InterpolateBackward:
            case dnnl::graph::op::kind::LeakyReLU:
            case dnnl::graph::op::kind::Log:
            case dnnl::graph::op::kind::LogSoftmax:
            case dnnl::graph::op::kind::LogSoftmaxBackward:
            case dnnl::graph::op::kind::Mish:
            case dnnl::graph::op::kind::MishBackward:
            case dnnl::graph::op::kind::Pow:
            case dnnl::graph::op::kind::PReLU:
            case dnnl::graph::op::kind::Quantize:
            case dnnl::graph::op::kind::Reciprocal:
            case dnnl::graph::op::kind::ReLU:
            case dnnl::graph::op::kind::ReLUBackward:
            case dnnl::graph::op::kind::Reorder:
            case dnnl::graph::op::kind::Round:
            case dnnl::graph::op::kind::Sigmoid:
            case dnnl::graph::op::kind::SigmoidBackward:
            case dnnl::graph::op::kind::SoftMax:
            case dnnl::graph::op::kind::SoftMaxBackward:
            case dnnl::graph::op::kind::SoftPlus:
            case dnnl::graph::op::kind::SoftPlusBackward:
            case dnnl::graph::op::kind::Sqrt:
            case dnnl::graph::op::kind::SqrtBackward:
            case dnnl::graph::op::kind::Square:
            case dnnl::graph::op::kind::Tanh:
            case dnnl::graph::op::kind::TanhBackward:
            case dnnl::graph::op::kind::TypeCast:
            // infer_bias_add_output_shape
            case dnnl::graph::op::kind::BiasAdd:
                in0 = aop.in_lts_[0].id_;
                out0 = aop.out_lts_[0].id_;
                gi[out0] = gi[in0];
                break;
            // infer_pool_bwd_output_shape
            case dnnl::graph::op::kind::MaxPoolBackward:
                in0 = aop.in_lts_[0].id_;
                out0 = aop.out_lts_[0].id_;
                gi[out0] = gi[in0];
                split_ncx(data_format, gi[in0], n, c, spatial_dims);
                strides = aop.attrs_["strides"].s64_vector_;
                kernel = aop.attrs_["kernel"].s64_vector_;
                cal_pads(pads_begin, pads_end, aop, spatial_dims, strides,
                        kernel, false);
                aop.attrs_["pads_begin"].s64_vector_ = pads_begin;
                aop.attrs_["pads_end"].s64_vector_ = pads_end;
                break;
            // infer_elemwise_arithmetic_output_shape
            // need to handle auto_broadcast
            case dnnl::graph::op::kind::Add:
            case dnnl::graph::op::kind::Divide:
            case dnnl::graph::op::kind::Maximum:
            case dnnl::graph::op::kind::Minimum:
            case dnnl::graph::op::kind::Multiply:
            case dnnl::graph::op::kind::SquaredDifference:
            case dnnl::graph::op::kind::Subtract:
                in0 = aop.in_lts_[0].id_;
                in1 = aop.in_lts_[1].id_;
                out0 = aop.out_lts_[0].id_;
                if (auto_broadcast == "none") {
                    gi[out0] = gi[in0];
                } else {
                    broadcast(gi[in0], gi[in1], gi[out0]);
                }
                break;
            // infer_pool_output_shape
            case dnnl::graph::op::kind::AvgPool:
            case dnnl::graph::op::kind::MaxPool:
                in0 = aop.in_lts_[0].id_;
                out0 = aop.out_lts_[0].id_;
                gi[out0].clear();
                split_ncx(data_format, gi[in0], n, c, spatial_dims);

                floor = true;
                if (aop.attrs_.find("rounding_type") != aop.attrs_.end()
                        && aop.attrs_["rounding_type"].str_value_ == "ceil") {
                    floor = false;
                }
                strides = aop.attrs_["strides"].s64_vector_;
                kernel = aop.attrs_["kernel"].s64_vector_;
                if (aop.attrs_.find("dilations") != aop.attrs_.end()) {
                    dilations = aop.attrs_["dilations"].s64_vector_;
                } else {
                    dilations.resize(kernel.size());
                    for (size_t i = 0; i < dilations.size(); i++) {
                        dilations[i] = 1;
                    }
                }
                cal_pads(pads_begin, pads_end, aop, spatial_dims, strides,
                        kernel, false);
                aop.attrs_["pads_begin"].s64_vector_ = pads_begin;
                aop.attrs_["pads_end"].s64_vector_ = pads_end;
                x.clear();
                for (size_t i = 0; i < spatial_dims.size(); i++) {
                    auto padded = spatial_dims[i] + pads_begin[i] + pads_end[i];
                    auto dilated = dilations[i] * (kernel[i] - 1) + 1;
                    if (floor) {
                        x.push_back((padded - dilated) / strides[i] + 1);
                    } else {
                        x.push_back((padded - dilated - 1) / strides[i] + 2);
                    }
                }
                merge_ncx(data_format, gi[out0], n, c, x);
                break;
            // infer_pool_bwd_output_shape
            case dnnl::graph::op::kind::AvgPoolBackward:
                out0 = aop.out_lts_[0].id_;
                gi[out0] = aop.attrs_["src_shape"].s64_vector_;
                split_ncx(data_format, gi[out0], n, c, spatial_dims);
                strides = aop.attrs_["strides"].s64_vector_;
                kernel = aop.attrs_["kernel"].s64_vector_;
                cal_pads(pads_begin, pads_end, aop, spatial_dims, strides,
                        kernel, false);
                aop.attrs_["pads_begin"].s64_vector_ = pads_begin;
                aop.attrs_["pads_end"].s64_vector_ = pads_end;
                break;
            // infer_bn_fwd_train_output_shape
            case dnnl::graph::op::kind::BatchNormForwardTraining:
            // infer_bn_bwd_output_shape
            case dnnl::graph::op::kind::BatchNormTrainingBackward:
                in0 = aop.in_lts_[0].id_;
                out0 = aop.out_lts_[0].id_;
                gi[out0] = gi[in0];
                split_ncx(data_format, gi[in0], n, c, x);
                for (size_t i = 1; i < aop.out_lts_.size(); i++) {
                    gi[aop.out_lts_[i].id_] = {c};
                }
                break;
            // infer_bias_bwd_output_shape
            case dnnl::graph::op::kind::BiasAddBackward:
                in0 = aop.in_lts_[0].id_;
                out0 = aop.out_lts_[0].id_;
                split_ncx(data_format, gi[in0], n, c, x);
                gi[out0] = {c};
                break;
            // infer_concat_output_shape
            case dnnl::graph::op::kind::Concat:
                in0 = aop.in_lts_[0].id_;
                out0 = aop.out_lts_[0].id_;
                sum = 0;
                axis = aop.attrs_["axis"].s64_value_;
                if (axis < 0) { axis += gi[in0].size(); }
                for (size_t i = 0; i < aop.in_lts_.size(); i++) {
                    sum += gi[aop.in_lts_[i].id_][axis];
                }

                gi[out0].clear();
                for (size_t i = 0; i < gi[in0].size(); i++) {
                    gi[out0].push_back(gi[in0][i]);
                }
                gi[out0][axis] = sum;
                break;
            // infer_convtranspose_bwd_data_output_shape
            case dnnl::graph::op::kind::ConvTransposeBackwardData: use_oi = 1;
            // infer_conv_output_shape
            case dnnl::graph::op::kind::Convolution:
                split_ncx(data_format, gi[aop.in_lts_[0].id_], n, c,
                        spatial_dims);
                split_oix(weights_format, gi[aop.in_lts_[1].id_], oi, kernel);
                strides = aop.attrs_["strides"].s64_vector_;
                dilations = aop.attrs_["dilations"].s64_vector_;
                cal_pads(pads_begin, pads_end, aop, spatial_dims, strides,
                        kernel, false);
                aop.attrs_["pads_begin"].s64_vector_ = pads_begin;
                aop.attrs_["pads_end"].s64_vector_ = pads_end;

                x.clear();
                for (size_t i = 0; i < spatial_dims.size(); i++) {
                    auto padded = spatial_dims[i] + pads_begin[i] + pads_end[i];
                    auto dialated = dilations[i] * (kernel[i] - 1) + 1;
                    x.push_back((padded - dialated) / strides[i] + 1);
                }
                merge_ncx(
                        data_format, gi[aop.out_lts_[0].id_], n, oi[use_oi], x);
                break;
            // infer_conv_bwd_data_output_shape
            case dnnl::graph::op::kind::ConvolutionBackwardData:
                if (aop.attrs_.find("dst_shape") != aop.attrs_.end()) {
                    gi[aop.out_lts_[0].id_]
                            = aop.attrs_["dst_shape"].s64_vector_;
                }
                split_ncx(data_format, gi[aop.in_lts_[0].id_], n, c,
                        spatial_dims);
                split_oix(weights_format, gi[aop.in_lts_[1].id_], oi, kernel);
                strides = aop.attrs_["strides"].s64_vector_;
                dilations = aop.attrs_["dilations"].s64_vector_;
                cal_pads(pads_begin, pads_end, aop, spatial_dims, strides,
                        kernel, false);
                aop.attrs_["pads_begin"].s64_vector_ = pads_begin;
                aop.attrs_["pads_end"].s64_vector_ = pads_end;
                break;
            // infer_convtranspose_output_shape
            case dnnl::graph::op::kind::ConvTranspose:
                split_ncx(data_format, gi[aop.in_lts_[0].id_], n, c,
                        spatial_dims);
                split_oix(weights_format, gi[aop.in_lts_[1].id_], oi, kernel);
                strides = aop.attrs_["strides"].s64_vector_;
                dilations = aop.attrs_["dilations"].s64_vector_;
                cal_pads(pads_begin, pads_end, aop, spatial_dims, strides,
                        kernel, true);
                aop.attrs_["pads_begin"].s64_vector_ = pads_begin;
                aop.attrs_["pads_end"].s64_vector_ = pads_end;

                if (aop.attrs_.find("output_padding") != aop.attrs_.end()) {
                    output_padding = aop.attrs_["output_padding"].s64_vector_;
                } else {
                    output_padding.clear();
                    for (size_t i = 0; i < spatial_dims.size(); i++) {
                        output_padding.push_back(0);
                    }
                }

                x.clear();
                for (size_t i = 0; i < spatial_dims.size(); i++) {
                    auto padded
                            = output_padding[i] - pads_begin[i] - pads_end[i];
                    auto dialated = dilations[i] * (kernel[i] - 1) + 1;
                    x.push_back(strides[i] * (spatial_dims[i] - 1) + dialated
                            + padded);
                }
                merge_ncx(data_format, gi[aop.out_lts_[0].id_], n,
                        oi[use_oi] * groups, x);
                break;
            // infer_conv_bwd_weights_output_shape
            case dnnl::graph::op::kind::ConvolutionBackwardWeights:
            // infer_convtranspose_bwd_weights_output_shape
            case dnnl::graph::op::kind::ConvTransposeBackwardWeights:
                if (aop.attrs_.find("weights_shape") != aop.attrs_.end()) {
                    gi[aop.out_lts_[0].id_]
                            = aop.attrs_["weights_shape"].s64_vector_;
                }
                split_ncx(data_format, gi[aop.in_lts_[0].id_], n, c,
                        spatial_dims);
                split_oix(weights_format, gi[aop.out_lts_[0].id_], oi, kernel);
                strides = aop.attrs_["strides"].s64_vector_;
                dilations = aop.attrs_["dilations"].s64_vector_;
                cal_pads(pads_begin, pads_end, aop, spatial_dims, strides,
                        kernel, false);
                aop.attrs_["pads_begin"].s64_vector_ = pads_begin;
                aop.attrs_["pads_end"].s64_vector_ = pads_end;
                break;
            // infer_interpolate_output_shape
            case dnnl::graph::op::kind::Interpolate:
                in0 = aop.in_lts_[0].id_;
                out0 = aop.out_lts_[0].id_;
                if (aop.attrs_.find("data_format") != aop.attrs_.end()
                        && aop.attrs_["data_format"].str_value_ == "NCX") {
                    data_format = "NCX";
                } else {
                    data_format = "NXC";
                }
                split_ncx(data_format, gi[in0], n, c, x);
                if (aop.attrs_.find("scales") != aop.attrs_.end()) {
                    for (size_t i = 0; i < x.size(); i++) {
                        x[i] = x[i] * aop.attrs_["scales"].f32_vector_[i];
                    }
                } else if (aop.attrs_.find("sizes") != aop.attrs_.end()) {
                    for (size_t i = 0; i < x.size(); i++) {
                        x[i] = aop.attrs_["sizes"].s64_vector_[i];
                    }
                }
                merge_ncx(data_format, gi[out0], n, c, x);
                break;
            // infer_norm_output_shape
            case dnnl::graph::op::kind::GroupNorm:
                // infer shape for dst.
                in0 = aop.in_lts_[0].id_;
                out0 = aop.out_lts_[0].id_;
                gi[out0] = gi[in0];
                // attr `keep_stats` is optional, default is `true`
                if (aop.attrs_.find("keep_stats") == aop.attrs_.end()
                        || aop.attrs_["keep_stats"].bool_value_) {
                    // `true` means it has 3 output: dst, mean and var.
                    //  need to infer shape for mean and var
                    if (aop.out_lts_.size() == 3) {
                        int64_t groups = 0;
                        if (aop.attrs_.find("groups") == aop.attrs_.end()) {
                            fprintf(stderr,
                                    "graph: groups is required for "
                                    "GroupNorm!\n");
                            SAFE_V(FAIL);
                        } else {
                            groups = aop.attrs_["groups"].s64_value_;
                        }
                        size_t out1 = aop.out_lts_[1].id_;
                        size_t out2 = aop.out_lts_[2].id_;
                        gi[out1].clear();
                        gi[out2].clear();
                        // mean/var shape is N,C
                        std::vector<int64_t> mv_shape = {gi[in0][0], groups};
                        gi[out1] = mv_shape;
                        gi[out2] = mv_shape;
                    } else {
                        fprintf(stderr,
                                "graph: GroupNorm output number "
                                "mismatch!\n");
                        SAFE_V(FAIL);
                    }
                }
                break;
            // infer_norm_output_shape
            case dnnl::graph::op::kind::LayerNorm:
                in0 = aop.in_lts_[0].id_;
                out0 = aop.out_lts_[0].id_;
                gi[out0] = gi[in0];
                if (aop.attrs_.find("keep_stats") == aop.attrs_.end()
                        || aop.attrs_["keep_stats"].bool_value_) {
                    int64_t axis = -1;
                    if (aop.attrs_.find("begin_norm_axis")
                            != aop.attrs_.end()) {
                        axis = aop.attrs_["begin_norm_axis"].s64_value_;
                    }
                    axis = axis >= 0 ? axis : gi[in0].size() + axis;
                    if (aop.out_lts_.size() == 3) {
                        size_t out1 = aop.out_lts_[1].id_,
                               out2 = aop.out_lts_[2].id_;
                        gi[out1].clear();
                        gi[out2].clear();
                        for (int64_t i = 0; i < axis; i++) {
                            gi[out1].push_back(gi[in0][i]);
                            gi[out2].push_back(gi[in0][i]);
                        }
                    } else {
                        fprintf(stderr,
                                "graph: LayerNorm output number "
                                "mismatch!\n");
                        SAFE_V(FAIL);
                    }
                }
                break;
            // infer_norm_bwd_out_shape
            case dnnl::graph::op::kind::LayerNormBackward:
                in0 = aop.in_lts_[0].id_;
                out0 = aop.out_lts_[0].id_;
                gi[out0] = gi[in0];
                if (aop.attrs_.find("use_affine") == aop.attrs_.end()
                        || aop.attrs_["use_affine"].bool_value_) {
                    gi[aop.out_lts_[1].id_] = gi[aop.in_lts_[4].id_];
                    gi[aop.out_lts_[2].id_] = gi[aop.in_lts_[5].id_];
                }
                break;
            // infer_matmul_output_shape
            case dnnl::graph::op::kind::MatMul:
                x = gi[aop.in_lts_[0].id_];
                y = gi[aop.in_lts_[1].id_];
                out0 = aop.out_lts_[0].id_;
                if (x.size() > 1
                        && aop.attrs_.find("transpose_a") != aop.attrs_.end()
                        && aop.attrs_["transpose_a"].bool_value_) {
                    auto tmp = x[x.size() - 1];
                    x[x.size() - 1] = x[x.size() - 2];
                    x[x.size() - 2] = tmp;
                }
                if (y.size() > 1
                        && aop.attrs_.find("transpose_b") != aop.attrs_.end()
                        && aop.attrs_["transpose_b"].bool_value_) {
                    auto tmp = y[y.size() - 1];
                    y[y.size() - 1] = y[y.size() - 2];
                    y[y.size() - 2] = tmp;
                }
                if (x.size() == 1 && y.size() == 1) {
                    gi[out0] = {};
                } else if (x.size() == 1) {
                    assert(x[0] == y[y.size() - 2]);
                    n = y[y.size() - 1];
                    y.pop_back();
                    y[y.size() - 1] = n;
                    gi[out0] = y;
                } else if (y.size() == 1) {
                    assert(x[x.size() - 1] == y[0]);
                    n = x[x.size() - 1];
                    x.pop_back();
                    x[x.size() - 1] = n;
                    gi[out0] = x;
                } else {
                    // Check that K is consistent in updated inputs.
                    if (x[x.size() - 1] != y[y.size() - 2]) {
                        BENCHDNN_PRINT(0,
                                "Error: updated shapes are not consistent. "
                                "Expected element \'%lld\' from \'%s\' to be "
                                "equal to element \'%lld\' from \'%s\'.\n",
                                (long long)(x[x.size() - 1]),
                                stdvec2string(x).c_str(),
                                (long long)(y[y.size() - 2]),
                                stdvec2string(y).c_str());
                        SAFE_V(FAIL);
                    }
                    size_t M = x[x.size() - 2];
                    size_t N = y[y.size() - 1];
                    dims_t x_batch(x.begin(), x.end() - 2);
                    dims_t y_batch(y.begin(), y.end() - 2);
                    broadcast(x_batch, y_batch, gi[out0], stdvec2string(x),
                            stdvec2string(y));
                    gi[out0].push_back(M);
                    gi[out0].push_back(N);
                }
                break;
            // infer_prelu_bwd_output_shape
            case dnnl::graph::op::kind::PReLUBackward:
                gi[aop.out_lts_[0].id_] = gi[aop.in_lts_[0].id_];
                gi[aop.out_lts_[1].id_] = gi[aop.in_lts_[1].id_];
                break;
            // infer_reduce_output_shape
            case dnnl::graph::op::kind::ReduceL1:
            case dnnl::graph::op::kind::ReduceL2:
            case dnnl::graph::op::kind::ReduceMax:
            case dnnl::graph::op::kind::ReduceMean:
            case dnnl::graph::op::kind::ReduceMin:
            case dnnl::graph::op::kind::ReduceProd:
            case dnnl::graph::op::kind::ReduceSum:
                in0 = aop.in_lts_[0].id_;
                out0 = aop.out_lts_[0].id_;

                if (aop.attrs_.find("axes") != aop.attrs_.end()) {
                    gi[out0] = gi[in0];
                    auto axes = aop.attrs_["axes"].s64_vector_;
                    auto keep_dims = false;
                    if (aop.attrs_.find("keep_dims") != aop.attrs_.end()) {
                        keep_dims = aop.attrs_["keep_dims"].bool_value_;
                    }
                    for (size_t i = 0; i < axes.size(); i++) {
                        gi[out0][(gi[in0].size() + axes[i]) % gi[in0].size()]
                                = keep_dims ? 1 : 0;
                    }
                    if (!keep_dims) {
                        gi[out0].erase(std::remove(gi[out0].begin(),
                                               gi[out0].end(), 0),
                                gi[out0].end());
                    }
                }
                break;
            // infer_select_output_shape
            case dnnl::graph::op::kind::Select:
                in0 = aop.in_lts_[0].id_;
                out0 = aop.out_lts_[0].id_;
                if (auto_broadcast == "none") {
                    gi[out0] = gi[in0];
                } else {
                    broadcast(gi[aop.in_lts_[1].id_], gi[aop.in_lts_[2].id_],
                            gi[out0]);
                    // one way broadcast only check whether cond can broadcast to the output
                    // no need to do one way broadcast
                }
                break;
            // infer_static_reshape_output_shape
            case dnnl::graph::op::kind::StaticReshape:
                in0 = aop.in_lts_[0].id_;
                out0 = aop.out_lts_[0].id_;
                adims = aop.attrs_["shape"].s64_vector_;
                special_zero = aop.attrs_["special_zero"].bool_value_;
                axis = -1;
                for (size_t i = 0; i < adims.size(); i++) {
                    if (adims[i] == 0 && special_zero) {
                        adims[i] = gi[in0][i];
                    } else if (adims[i] == -1) {
                        axis = i;
                    }
                }
                in_size = out_size = 1;
                for (size_t i = 0; i < gi[in0].size(); i++) {
                    in_size = gi[in0][i] >= 0 ? in_size * gi[in0][i] : in_size;
                }
                for (size_t i = 0; i < adims.size(); i++) {
                    out_size = adims[i] >= 0 ? out_size * adims[i] : out_size;
                }
                if (axis != -1) { adims[axis] = in_size / out_size; }
                gi[out0] = adims;
                break;
            // infer_static_transpose_output_shape
            case dnnl::graph::op::kind::StaticTranspose:
                in0 = aop.in_lts_[0].id_;
                out0 = aop.out_lts_[0].id_;
                gi[out0].clear();
                adims = aop.attrs_["order"].s64_vector_;
                if (adims.empty()) {
                    for (size_t i = 0; i < gi[in0].size(); i++) {
                        gi[out0].push_back(gi[in0][gi[in0].size() - i - 1]);
                    }
                } else {
                    for (size_t i = 0; i < gi[in0].size(); i++) {
                        gi[out0].push_back(gi[in0][(gi[in0].size() + adims[i])
                                % gi[in0].size()]);
                    }
                }
                break;
            // infer_unsupported_output_shape
            case dnnl::graph::op::kind::Wildcard:
            // no output, do nothing
            case dnnl::graph::op::kind::End:
            case dnnl::graph::op::kind::LastSymbol: break;
        }

        for (auto &lt : aop.in_lts_) {
            lt.shape_ = gi[lt.id_];
            lt.stride_
                    = memory_tag2strides(gi[lt.id_], dgraph.lt_2_mtag_[lt.id_]);
        }
        // update outputs for the current op
        update_output_info(aop, dgraph, change_stride);
    }
}

/// @brief Get a new shape and new strides from CML. Re-written shapes and strides
/// must pass compatibility check.
/// @param in_shape String input from CML.
/// @param shape Parsed updated shape.
/// @param stride Parsed updated strides.
/// @param msg Error message info when function returns `false` value.
/// @return `true` if an inport info is valid and `false` otherwise. A message `msg`
/// describes an error occurred.
bool flex_rewrite::get_inport_shape_stride(const std::string &in_shape,
        std::string &shape, std::string &stride, std::string &msg) {
    assert(shape.empty() && stride.empty());
    if (in_shape == "0" || in_shape == "-") {
        shape = in_shape;
        return true;
    }
    // valid stride: "acdb"
    const auto all_letters = [](const std::string &ss) -> bool {
        return std::all_of(ss.cbegin(), ss.cend(), [](int c) {
            assert(c < UINT8_MAX);
            return std::isalpha(c);
        });
    };
    // valid shape: "2x32x4x4"
    const auto all_digit_cross = [](const std::string &ss) -> bool {
        return std::all_of(ss.cbegin(), ss.cend(), [](int c) {
            assert(c < UINT8_MAX);
            return std::isdigit(c) || c == 'x';
        });
    };

    size_t in_length = in_shape.size();
    size_t star_pos = in_shape.find('*');
    const static std::string err_msg
            = "A shape is expected in the form of `NUMxNUMxNUM...`. A tag must "
              "be composed of letters only.";
    // shape and stride are provided
    if (star_pos != std::string::npos) {
        // invalid CML info, e.g. "1x2x3*", "*abc"
        if (star_pos == 0 || star_pos == in_length - 1) {
            msg = "A shape must be provided before the `*`, a tag - after the "
                  "`*`. Alternatively, remove the `*` to apply the default "
                  "shape or stride.";
            return false;
        }
        shape = in_shape.substr(0, star_pos);
        stride = in_shape.substr(star_pos + 1, in_length);

        if (all_letters(stride) && all_digit_cross(shape)) {
            return true;
        } else {
            msg = err_msg;
            return false;
        }
    } else if (all_letters(in_shape)) { // user only provide a new stride
        stride = in_shape;
        return true;
    } else if (all_digit_cross(in_shape)) { // user only provide a new shape
        shape = in_shape;
        return true;
    } else { // a valid input info is rather strict, return false if the input info is none of the above
        msg = err_msg;
        return false;
    }
}

void flex_rewrite::inports_shape_rewrite(
        deserialized_graph &dgraph, bool &change_stride) {
    // reminder mb rewrite status
    if (mb_ != 0 && dgraph.graph_inputs_with_mb_.empty()) {
        BENCHDNN_PRINT(0,
                "Error: flex_rewrite: can't rewrite mb value with \'%ld\'.\n",
                (long)mb_);
    }

    const auto set_default_deserialized_lt = [](deserialized_lt &lt) {
        auto ndims = lt.shape_.size();
        logical_tensor::dims infer_dim(ndims, -1);
        lt.shape_ = infer_dim;
        lt.stride_ = infer_dim;
    };

    // check stride provided from cml, return false, if one is not a valid tag
    // for example: when stride is "dcba", return true;
    // when stride size is 4, "zcba" & "aabc" are all invalid stride
    const auto is_valid_stride = [](const std::string &stride) -> bool {
        assert(!stride.empty());
        for (size_t i = 0; i < stride.size(); ++i) {
            if (stride.find(char('a' + i)) == std::string::npos) {
                return false;
            }
        }
        return true;
    };

    for_(auto &aop : dgraph.ops_)
    for (auto &lt : aop.in_lts_) {
        // if 'lt' is not a inport, set default logical tensor info
        if (dgraph.graph_tensors_.find(lt.id_) == dgraph.graph_tensors_.end()) {
            // At the same time check if in_shapes contain non-inport tensors.
            if (in_shapes_.find(lt.id_) != in_shapes_.end()) {
                BENCHDNN_PRINT(0,
                        "Error: \'in-shapes\' option contains a tensor with "
                        "id=\'%zu\' which is not an input for a given graph.\n",
                        lt.id_);
                SAFE_V(FAIL);
            }
            set_default_deserialized_lt(lt);
            continue;
        }

        const bool has_mb_rewrite = mb_ != 0
                && std::find(dgraph.graph_inputs_with_mb_.begin(),
                           dgraph.graph_inputs_with_mb_.end(), lt.id_)
                        != dgraph.graph_inputs_with_mb_.end();
        if (in_shapes_.find(lt.id_) != in_shapes_.end()
                && in_shapes_[lt.id_] != "default") {

            std::string new_shape, new_stride, message;
            bool result = get_inport_shape_stride(
                    in_shapes_[lt.id_], new_shape, new_stride, message);
            if (!result) {
                BENCHDNN_PRINT(0,
                        "Error: `--in-shapes` is not valid. Reason: %s\n",
                        message.c_str());
                SAFE_V(FAIL);
            }

            // Rewrite logic covers the following scenarios:
            // shape, stride: ["-", ""]
            // shape, stride: ["1x32x4", ""] including ["0", ""]
            // shape, stride: ["1x32x4", "abc"]
            // shape, stride: ["", "abc"]
            // and checks has been done accordingliy
            size_t ndims = lt.shape_.size(); // the original rank from JSON
            dims_t new_shape_dims;
            // shape "-" means this logical tensor is 0 rank with shape: []
            const bool zero_rank = new_shape == "-";
            // if the current logical tensor is rewritten to rank-0
            if (zero_rank) {
                lt.shape_ = dims_t {};
                lt.stride_ = dims_t {};
                dgraph.graph_tensors_[lt.id_] = dims_t {};
                dgraph.lt_2_mtag_[lt.id_] = "";
                continue;
            }
            if (!new_shape.empty()) {
                new_shape_dims = string_to_shape(new_shape);
                if (!new_stride.empty()) {
                    if (new_shape_dims.size() != new_stride.size()) {
                        BENCHDNN_PRINT(0,
                                "Error: the shape size (`%zu`) must coinside "
                                "with the stride size (`%zu`).\n",
                                new_shape_dims.size(), new_stride.size());
                        SAFE_V(FAIL);
                    }
                    if (!is_valid_stride(new_stride)) {
                        BENCHDNN_PRINT(0,
                                "Error: the tag provided is not valid: `%s`: "
                                "unexpected letters encountered.\n",
                                new_stride.c_str());
                        SAFE_V(FAIL);
                    }
                }
                if (has_mb_rewrite) { new_shape_dims[0] = mb_; }
            }

            // if rank change and no stride provided
            if (new_shape_dims.size() != ndims && new_stride.empty()) {
                change_stride = true;
                // use default stride
                dgraph.lt_2_mtag_[lt.id_]
                        = get_default_tag(new_shape_dims.size());
            }

            // update new value to dgraph
            if (!new_shape.empty()) {
                lt.shape_ = new_shape_dims;
                dgraph.graph_tensors_[lt.id_] = new_shape_dims;
            }
            if (!new_stride.empty()) {
                // original 4d, no new shape and stride.size() != 4,
                // this is not valid stride
                if (new_shape.empty() && new_stride.size() != ndims) {
                    BENCHDNN_PRINT(0,
                            "Error: the tag provided is not valid: `%s`: the "
                            "tag size must be `%d`.\n",
                            new_stride.c_str(), static_cast<int>(ndims));
                    SAFE_V(FAIL);
                }
                if (!is_valid_stride(new_stride)) {
                    BENCHDNN_PRINT(0,
                            "Error: the tag provided is not valid: `%s`: "
                            "unexpected letters encountered.\n",
                            new_stride.c_str());
                    SAFE_V(FAIL);
                }
                if (dgraph.lt_2_mtag_[lt.id_] != new_stride) {
                    change_stride = true;
                    dgraph.lt_2_mtag_[lt.id_] = new_stride;
                }
            }
            lt.stride_
                    = memory_tag2strides(lt.shape_, dgraph.lt_2_mtag_[lt.id_]);

        } else if (has_mb_rewrite) {
            lt.shape_[0] = mb_;
            dgraph.graph_tensors_[lt.id_] = lt.shape_;
        }
    }

    for_(auto &aop : dgraph.ops_)
    for (auto &lt : aop.out_lts_) {
        set_default_deserialized_lt(lt);
    }
}

void flex_rewrite::op_attrs_rewrite(deserialized_graph &dgraph) {
    std::vector<size_t> op_ids_;
    op_ids_.reserve(dgraph.ops_.size());
    for (const auto &aop : dgraph.ops_) {
        op_ids_.emplace_back(aop.id_);
    }

    for (const auto &temp_attrs : op_attrs_) {
        auto iter = std::find(op_ids_.begin(), op_ids_.end(), temp_attrs.first);
        if (iter == op_ids_.end()) {
            BENCHDNN_PRINT(0, "graph: rewrite: no op id %zd in the graph.\n",
                    temp_attrs.first);
            SAFE_V(FAIL);
        }
        auto &temp_op = dgraph.ops_[std::distance(op_ids_.begin(), iter)];
        const auto attrs = parse_attrs(temp_attrs.second);
        for (const auto &new_attr : attrs) {
            auto attr_name = new_attr.first;
            if (!temp_op.attrs_.count(attr_name)) {
                BENCHDNN_PRINT(0,
                        "graph: rewrite: no attr name `%s` in op %zd.\n",
                        attr_name.c_str(), temp_attrs.first);
                SAFE_V(FAIL);
            }
            auto new_val = new_attr.second;
            auto attr_type = temp_op.attrs_[attr_name].type_;
            if (attr_type == "string") {
                temp_op.attrs_[attr_name].str_value_ = new_val;
            } else if (attr_type == "bool") {
                temp_op.attrs_[attr_name].bool_value_
                        = str2bool(new_val.c_str());
            } else if (attr_type == "s64") {
                temp_op.attrs_[attr_name].s64_value_ = stoll(new_val);
            } else if (attr_type == "s64[]") {
                temp_op.attrs_[attr_name].s64_vector_
                        = string_to_shape(new_val);
            } else if (attr_type == "f32") {
                temp_op.attrs_[attr_name].f32_value_ = atof(new_val.c_str());
            } else if (attr_type == "f32[]") {
                temp_op.attrs_[attr_name].f32_vector_
                        = string_to_f32_vec(new_val);
            }
        }
    }
}

inline bool is_int8_quantization(const deserialized_op &aop) {
    if (aop.kind_ == "Dequantize") {
        const auto dt = aop.in_lts_.front().get_data_type();
        return (dt == logical_tensor::data_type::u8
                || dt == logical_tensor::data_type::s8);
    } else if (aop.kind_ == "Quantize") {
        const auto dt = aop.out_lts_.front().get_data_type();
        return (dt == logical_tensor::data_type::u8
                || dt == logical_tensor::data_type::s8);
    } else {
        // should not reach here
        return false;
    }
}

void flex_rewrite::quantized_graph_rewrite(deserialized_graph &dgraph) {
    for (auto &aop : dgraph.ops_) {
        if (aop.kind_ != "Dequantize" && aop.kind_ != "Quantize") continue;

        auto &attr = aop.attrs_;
        const auto is_int8 = is_int8_quantization(aop);
        if (attr.find("scales") == attr.end()
                || attr.find("qtype") == attr.end()
                || attr["qtype"].str_value_ != "per_channel")
            continue;
        if (is_int8 && attr.find("zps") == attr.end()) continue;

        const auto &pre_scales = attr["scales"].f32_vector_;
        const auto pre_scales_size = pre_scales.size();
        int64_t axis = 1;
        auto ndims = aop.in_lts_.front().shape_.size();
        if (attr.find("axis") != attr.end()) {
            axis = (attr["axis"].s64_value_ + ndims) % ndims;
        }
        const int64_t scales_zp_dim = aop.in_lts_.front().shape_[axis];
        std::vector<float> scales(scales_zp_dim);

        for (int64_t i = 0; i < scales_zp_dim; i++)
            scales[i] = pre_scales[i % pre_scales_size];
        aop.attrs_["scales"].f32_vector_ = scales;

        if (is_int8) {
            auto pre_zps = attr["zps"].s64_vector_;
            std::vector<int64_t> zps;
            for (int64_t i = 0; i < scales_zp_dim; i++) {
                if (static_cast<size_t>(i) < pre_zps.size()) {
                    zps.push_back(pre_zps[i]);
                } else {
                    zps.push_back(0);
                }
            }
            aop.attrs_["zps"].s64_vector_ = zps;
        }
    }
}

// Select: only rewrite src_1/src_2/dst as `cond` is always `bool`.
void dt_rewrite_select(deserialized_op &select, const std::string &dt) {
    select.in_lts_[1].data_type_ = dt;
    select.in_lts_[2].data_type_ = dt;
    select.out_lts_[0].data_type_ = dt;
}

// Normalization ops: only rewrite src/dst/diff_src/diff_dst as f16
// normalization still requires f32 for gamma/beta/etc. This is good for most of
// the cases. But there is a potential issue if gamma/beta/etc is connected to
// another op which will rewrite the data type at other places.
void dt_rewrite_norm(deserialized_op &norm, const std::string &dt) {
    if (norm.kind_ == "BatchNormTrainingBackward"
            || norm.kind_ == "LayerNormBackward") {
        // rewrite for src/diff_dst/diff_src.
        norm.in_lts_[0].data_type_ = dt;
        norm.in_lts_[1].data_type_ = dt;
        norm.out_lts_[0].data_type_ = dt;
    } else {
        // only rewrite for src/dst.
        norm.in_lts_[0].data_type_ = dt;
        norm.out_lts_[0].data_type_ = dt;
    }
}

void flex_rewrite::dt_rewrite(deserialized_graph &dgraph) {
    if (dt_ == dnnl_data_type_undef) return;

    // We can only do data type rewriting for pure floating-point graph.
    static const std::vector<dnnl_data_type_t> fp_dts {
            dnnl_f32, dnnl_bf16, dnnl_f16};
    if (!std::any_of(fp_dts.begin(), fp_dts.end(),
                [this](const dnnl_data_type_t &dt) { return dt_ == dt; })) {
        BENCHDNN_PRINT(0, "graph: rewrite: `%s` data type is not supported\n",
                dt2str(dt_));
        SAFE_V(FAIL);
    }

    static const std::vector<std::string> lowp_ops {
            "TypeCast",
            "Quantize",
            "Dequantize",
            "DynamicQuantize",
            "DynamicDequantize",
    };
    // If the graph contains mix-precision ops, we cannot rewrite the data type
    // trivially.
    for (auto &aop : dgraph.ops_) {
        if (std::any_of(lowp_ops.begin(), lowp_ops.end(),
                    [&aop](const std::string &k) { return aop.kind_ == k; })) {
            BENCHDNN_PRINT(0,
                    "graph: rewrite: the graph contains operation `%s`\n",
                    aop.kind_.c_str());
            SAFE_V(FAIL);
        }
    }

    // Normalization ops need additional handling. See the comments of function
    // `dt_rewrite_norm`.
    static const std::vector<std::string> norm_ops {
            "BatchNormForwardTraining",
            "BatchNormInference",
            "BatchNormTrainingBackward",
            "GroupNorm",
            "LayerNorm",
            "LayerNormBackward",
    };

    // rewrite
    std::string str_dt(dt2str(dt_));
    for (auto &aop : dgraph.ops_) {
        if (aop.kind_ == "Select") {
            dt_rewrite_select(aop, str_dt);
        } else if (std::any_of(norm_ops.begin(), norm_ops.end(),
                           [&aop](const std::string &k) {
                               return aop.kind_ == k;
                           })) {
            dt_rewrite_norm(aop, str_dt);
        } else if (aop.kind_ == "GenIndex") {
            // GenIndex: only rewrite src dtype
            aop.in_lts_[0].data_type_ = str_dt;
        } else if (aop.kind_ == "GreaterEqual") {
            // GreaterEqual: only rewrite src dtype when it's floating-point
            if (std::any_of(fp_dts.begin(), fp_dts.end(),
                        [&aop](const dnnl_data_type_t &fp_dt) {
                            return aop.in_lts_[0].data_type_ == dt2str(fp_dt);
                        })) {
                aop.in_lts_[0].data_type_ = str_dt;
                aop.in_lts_[1].data_type_ = str_dt;
            }
        } else {
            for (auto &lt : aop.in_lts_) {
                lt.data_type_ = str_dt;
            }

            for (auto &lt : aop.out_lts_) {
                lt.data_type_ = str_dt;
            }
        }
    }
}

void flex_rewrite::dt_map_rewrite(deserialized_graph &dgraph) {
    // check the IDs and data types in dt_map.
    for (const auto &v : dt_map_) {
        if (v.second == dnnl_data_type_undef) return;

        bool found_id = false;
        for (auto &aop : dgraph.ops_) {
            for (auto &lt : aop.in_lts_) {
                if (lt.id_ == v.first) found_id = true;
            }

            for (auto &lt : aop.out_lts_) {
                if (lt.id_ == v.first) found_id = true;
            }
        }

        if (!found_id) {
            BENCHDNN_PRINT(0,
                    "graph: rewrite: ID `%zd` is not found in the graph\n",
                    v.first);
            SAFE_V(FAIL);
        }
    }

    // rewrite
    for (const auto &v : dt_map_) {
        const std::string str_dt(dt2str(v.second));
        for (auto &aop : dgraph.ops_) {
            for (auto &lt : aop.in_lts_) {
                if (lt.id_ == v.first) lt.data_type_ = str_dt;
            }

            for (auto &lt : aop.out_lts_) {
                if (lt.id_ == v.first) lt.data_type_ = str_dt;
            }
        }
    }
}

void flex_rewrite::graph_attrs_rewrite(deserialized_graph &dgraph) {

    // if the fpmath mode is specified by users through cml, replace the fpmath
    // mode from JSON file with the value from cml.
    if (fpmath_mode_.override_json_value_) dgraph.set_fpmath_mode(fpmath_mode_);

    for (auto &aop : dgraph.ops_) {
        // save the graph-level config for ops
        const auto &mode = dgraph.get_fpmath_mode();
        aop.fpmath_mode_ = mode.first;
        aop.fpmath_mode_apply_to_int_ = mode.second;
    }
}

/// @brief Update the output shape & stride & members after infer output shape
/// @param aop The current op of the graph
/// @param dgraph A deserialized graph
/// @param change_stride A boolean value indicating whether the graph input strides
/// have been changed.
void flex_rewrite::update_output_info(
        deserialized_op &aop, deserialized_graph &dgraph, bool change_stride) {
    auto kind = opstr2kind(aop.kind_);
    auto &gi = dgraph.graph_tensors_;
    // if a input stride is not changed, the output stride should not be changed as well
    if (!change_stride) {
        for (auto &lt : aop.out_lts_) {
            lt.shape_ = gi[lt.id_];
            lt.stride_
                    = memory_tag2strides(gi[lt.id_], dgraph.lt_2_mtag_[lt.id_]);
        }
        return;
    }
    // step1: get dominate stride info
    // get the src stride, normally index 0 input tensor
    std::string dominate_stride = dgraph.lt_2_mtag_[aop.in_lts_.front().id_];
    // step2: determine out stride info
    switch (kind) {
        // category 1: all output lts have the same dim size
        case dnnl::graph::op::kind::Abs:
        case dnnl::graph::op::kind::AbsBackward:
        case dnnl::graph::op::kind::Add:
        case dnnl::graph::op::kind::AvgPool:
        case dnnl::graph::op::kind::AvgPoolBackward:
        case dnnl::graph::op::kind::BatchNormInference:
        case dnnl::graph::op::kind::BiasAdd:
        case dnnl::graph::op::kind::Clamp:
        case dnnl::graph::op::kind::ClampBackward:
        case dnnl::graph::op::kind::Concat:
        case dnnl::graph::op::kind::Convolution:
        case dnnl::graph::op::kind::ConvolutionBackwardData:
        case dnnl::graph::op::kind::ConvolutionBackwardWeights:
        case dnnl::graph::op::kind::ConvTranspose:
        case dnnl::graph::op::kind::ConvTransposeBackwardData:
        case dnnl::graph::op::kind::ConvTransposeBackwardWeights:
        case dnnl::graph::op::kind::Dequantize:
        case dnnl::graph::op::kind::Divide:
        case dnnl::graph::op::kind::DynamicDequantize:
        case dnnl::graph::op::kind::DynamicQuantize:
        case dnnl::graph::op::kind::Elu:
        case dnnl::graph::op::kind::EluBackward:
        case dnnl::graph::op::kind::Exp:
        case dnnl::graph::op::kind::GELU:
        case dnnl::graph::op::kind::GELUBackward:
        case dnnl::graph::op::kind::GenIndex:
        case dnnl::graph::op::kind::GreaterEqual:
        case dnnl::graph::op::kind::GroupNorm:
        case dnnl::graph::op::kind::HardSigmoid:
        case dnnl::graph::op::kind::HardSigmoidBackward:
        case dnnl::graph::op::kind::HardSwish:
        case dnnl::graph::op::kind::HardSwishBackward:
        case dnnl::graph::op::kind::Interpolate:
        case dnnl::graph::op::kind::InterpolateBackward:
        case dnnl::graph::op::kind::LayerNorm:
        case dnnl::graph::op::kind::LeakyReLU:
        case dnnl::graph::op::kind::Log:
        case dnnl::graph::op::kind::LogSoftmax:
        case dnnl::graph::op::kind::LogSoftmaxBackward:
        case dnnl::graph::op::kind::Maximum:
        case dnnl::graph::op::kind::MatMul:
        case dnnl::graph::op::kind::MaxPool:
        case dnnl::graph::op::kind::MaxPoolBackward:
        case dnnl::graph::op::kind::Minimum:
        case dnnl::graph::op::kind::Mish:
        case dnnl::graph::op::kind::MishBackward:
        case dnnl::graph::op::kind::Multiply:
        case dnnl::graph::op::kind::Pow:
        case dnnl::graph::op::kind::PReLU:
        case dnnl::graph::op::kind::PReLUBackward:
        case dnnl::graph::op::kind::Quantize:
        case dnnl::graph::op::kind::Reciprocal:
        case dnnl::graph::op::kind::ReduceL1:
        case dnnl::graph::op::kind::ReduceL2:
        case dnnl::graph::op::kind::ReduceMax:
        case dnnl::graph::op::kind::ReduceMean:
        case dnnl::graph::op::kind::ReduceMin:
        case dnnl::graph::op::kind::ReduceProd:
        case dnnl::graph::op::kind::ReduceSum:
        case dnnl::graph::op::kind::ReLU:
        case dnnl::graph::op::kind::ReLUBackward:
        case dnnl::graph::op::kind::Round:
        case dnnl::graph::op::kind::Select:
        case dnnl::graph::op::kind::Sigmoid:
        case dnnl::graph::op::kind::SigmoidBackward:
        case dnnl::graph::op::kind::SoftMax:
        case dnnl::graph::op::kind::SoftMaxBackward:
        case dnnl::graph::op::kind::SoftPlus:
        case dnnl::graph::op::kind::SoftPlusBackward:
        case dnnl::graph::op::kind::Sqrt:
        case dnnl::graph::op::kind::Square:
        case dnnl::graph::op::kind::SqrtBackward:
        case dnnl::graph::op::kind::Subtract:
        case dnnl::graph::op::kind::Tanh:
        case dnnl::graph::op::kind::TanhBackward:
        case dnnl::graph::op::kind::TypeCast: {
            for (auto &lt : aop.out_lts_) {
                // shape has been determined in 'gi' by infer_out_shape()
                // so update shape in lt here
                lt.shape_ = gi[lt.id_];
                dgraph.lt_2_mtag_[lt.id_] = dominate_stride;
                lt.stride_ = memory_tag2strides(gi[lt.id_], dominate_stride);
            }
            break;
        }
        // category 2: there exists logical tensors with output dimention size of 1
        case dnnl::graph::op::kind::BatchNormForwardTraining:
        case dnnl::graph::op::kind::BatchNormTrainingBackward:
        case dnnl::graph::op::kind::LayerNormBackward: {
            for (auto &lt : aop.out_lts_) {
                lt.shape_ = gi[lt.id_];
                if (lt.shape_.size() != 1) {
                    dgraph.lt_2_mtag_[lt.id_] = dominate_stride;
                    lt.stride_
                            = memory_tag2strides(gi[lt.id_], dominate_stride);
                } else {
                    dgraph.lt_2_mtag_[lt.id_] = "a";
                    lt.stride_ = memory_tag2strides(gi[lt.id_], "a");
                }
            }
            break;
        }
        // category 3. special ops, set dst stride as "abcd...", as the purppose
        // of those ops are modifing the input stride, and the output stride can
        // not be specified via flex rewrite currently, therefore a default stride
        // represented by "abcd..." is set to the output
        case dnnl::graph::op::kind::Reorder:
        case dnnl::graph::op::kind::StaticReshape:
        case dnnl::graph::op::kind::StaticTranspose: {
            assert(aop.out_lts_.size() == 1);
            auto &lt = aop.out_lts_.front();
            lt.shape_ = gi[lt.id_];
            dgraph.lt_2_mtag_[lt.id_] = get_default_tag(lt.shape_.size());
            lt.stride_
                    = memory_tag2strides(gi[lt.id_], dgraph.lt_2_mtag_[lt.id_]);
            break;
        }
        // catecory 4. no real cases for those ops or they are not supported, so
        // just skip them
        case dnnl::graph::op::kind::BiasAddBackward:
        case dnnl::graph::op::kind::End:
        case dnnl::graph::op::kind::SquaredDifference:
        case dnnl::graph::op::kind::Wildcard: break;
        default:
            BENCHDNN_PRINT(0, "%s is not supported\n", aop.kind_.c_str());
            SAFE_V(FAIL);
            break;
    }
}

} // namespace graph
