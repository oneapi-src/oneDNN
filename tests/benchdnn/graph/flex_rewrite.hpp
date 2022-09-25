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

#ifndef BENCHDNN_GRAPH_FLEX_REWRITE_HPP
#define BENCHDNN_GRAPH_FLEX_REWRITE_HPP

#include <iostream>
#include <iterator>
#include <map>
#include <sstream>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "deserialize.hpp"
#include "utils.hpp"

namespace graph {

struct flex_rewrite {
    std::map<size_t, std::string> in_shapes;
    std::map<size_t, std::string> op_attrs;
    int64_t mb;

    flex_rewrite(const std::map<size_t, std::string> in_shapes,
            const std::map<size_t, std::string> op_attrs, const int64_t mb)
        : in_shapes(in_shapes), op_attrs(op_attrs), mb(mb) {}

    void rewrite(deserialized_graph &dgraph) {
        input_shape_rewrite(dgraph);
        if (!(op_attrs.size() == 1 && op_attrs.count(0)
                    && op_attrs.at(0) == "default")) {
            op_attrs_rewrite(dgraph);
        }
        infer_output_shape(dgraph);
        quantized_graph_rewrite(dgraph);
    }

    void split_ncx(std::string data_format, dims_t &in, int64_t &n, int64_t &c,
            dims_t &x) {
        x.clear();
        n = in[0];
        if (data_format == "NCX") {
            c = in[1];
            for (size_t i = 2; i < in.size(); i++) {
                x.push_back(in[i]);
            }
        } else { // NXC
            for (size_t i = 1; i < in.size() - 1; i++) {
                x.push_back(in[i]);
            }
            c = in[in.size() - 1];
        }
    }
    void merge_ncx(std::string data_format, dims_t &out, int64_t n, int64_t c,
            dims_t &x) {
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
    void split_oix(std::string data_format, dims_t &in, dims_t &oi, dims_t &x) {
        x.clear();
        if (data_format == "OIX") {
            oi = {in[0], in[1]};
            for (size_t i = 2; i < in.size(); i++) {
                x.push_back(in[i]);
            }
        } else { // XIO
            for (size_t i = 0; i < in.size() - 2; i++) {
                x.push_back(in[i]);
            }
            oi = {in[in.size() - 1], in[in.size() - 2]};
        }
    }

    void broadcast(const dims_t &x, const dims_t &y, dims_t &z) {
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
                    fprintf(stderr, "graph: invalid shape!\n");
                    exit(2);
                }
                z[i] = (l == 1 ? r : l);
            } else {
                z[i] = l;
            }
        }
    }

    // return the pad_begin + pad_end for each dimension
    void cal_pads(dims_t &pads, deserialized_op &aop, dims_t &spatial_dims,
            dims_t &strides, dims_t &kernel, bool deconv) {
        dims_t v;
        pads.clear();
        std::string auto_pad = "NONE";
        if (aop.attrs_.find("auto_pad") != aop.attrs_.end()) {
            auto_pad = aop.attrs_["auto_pad"].get_string();
            transform(auto_pad.begin(), auto_pad.end(), auto_pad.begin(),
                    toupper);
        }
        if (aop.attrs_.find("auto_pad") == aop.attrs_.end()
                || auto_pad == "NONE") {
            v = aop.attrs_["pads_begin"].get_s64_vector();
            for (size_t i = 0; i < v.size(); i++) {
                pads.push_back(v[i]);
            }
            v = aop.attrs_["pads_end"].get_s64_vector();
            for (size_t i = 0; i < v.size(); i++) {
                pads[i] += v[i];
            }
        } else {
            if (auto_pad == "VALID") {
                for (size_t i = 0; i < spatial_dims.size(); i++) {
                    pads.push_back(0);
                }
            } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
                // SAME_UPPER or SAME_LOWER
                // total padding is the same for both conditions
                for (size_t i = 0; i < spatial_dims.size(); i++) {
                    auto legacy
                            = (spatial_dims[i] + strides[i] - 1) / strides[i];
                    auto pad_needed = deconv ? kernel[i] - strides[i]
                                             : (legacy - 1) * strides[i]
                                    + kernel[i] - spatial_dims[i];
                    if (pad_needed < 0) {
                        fprintf(stderr, "graph: auto_pad failed!\n");
                        exit(2);
                    }
                    if (auto_pad == "SAME_UPPER") {
                        pads.push_back((pad_needed - 1) / 2);
                    } else {
                        pads.push_back(pad_needed / 2);
                    }
                }
            } else {
                fprintf(stderr, "graph: invalid arguments!\n");
                exit(2);
            }
        }
    }

    void infer_output_shape(deserialized_graph &dgraph) {
        auto &gi = dgraph.graph_inputs;
        for (auto &aop : dgraph.ops_) {
            auto kind = opstr2kind(aop.kind_);
            size_t in0, in1, out0;
            int64_t n, c, axis, sum, groups, in_size, out_size, use_oi = 0;
            dims_t strides, kernel, pads, dilations, spatial_dims,
                    output_padding;
            dims_t dims, x, y, oi;
            std::string data_format, filter_format, auto_broadcast;
            bool floor, special_zero;
            // set default value for some attributes
            if (aop.attrs_.find("data_format") != aop.attrs_.end()
                    && aop.attrs_["data_format"].get_string() == "NCX") {
                data_format = "NCX";
            } else {
                data_format = "NXC";
            }
            if (aop.attrs_.find("auto_broadcast") != aop.attrs_.end()
                    && aop.attrs_["auto_broadcast"].get_string() == "none") {
                auto_broadcast = "none";
            } else {
                auto_broadcast = "numpy";
            }
            if (aop.attrs_.find("filter_format") != aop.attrs_.end()
                    && aop.attrs_["filter_format"].get_string() == "OIX") {
                filter_format = "OIX";
            } else {
                filter_format = "XIO";
            }
            if (aop.attrs_.find("groups") != aop.attrs_.end()) {
                groups = aop.attrs_["groups"].get_s64();
            } else {
                groups = 1;
            }
            switch (kind) {
                // infer_identity_output_shape
                case dnnl::graph::op::kind::Abs:
                case dnnl::graph::op::kind::AbsBackprop:
                case dnnl::graph::op::kind::BatchNormInference:
                case dnnl::graph::op::kind::Clamp:
                case dnnl::graph::op::kind::ClampBackprop:
                case dnnl::graph::op::kind::Dequantize:
                case dnnl::graph::op::kind::DynamicDequantize:
                case dnnl::graph::op::kind::DynamicQuantize:
                case dnnl::graph::op::kind::Elu:
                case dnnl::graph::op::kind::EluBackprop:
                case dnnl::graph::op::kind::Erf:
                case dnnl::graph::op::kind::Exp:
                case dnnl::graph::op::kind::GELU:
                case dnnl::graph::op::kind::GELUBackprop:
                case dnnl::graph::op::kind::HardSwish:
                case dnnl::graph::op::kind::HardSwishBackprop:
                case dnnl::graph::op::kind::InterpolateBackprop:
                case dnnl::graph::op::kind::LeakyReLU:
                case dnnl::graph::op::kind::Log:
                case dnnl::graph::op::kind::LogSoftmax:
                case dnnl::graph::op::kind::LogSoftmaxBackprop:
                case dnnl::graph::op::kind::Mish:
                case dnnl::graph::op::kind::MishBackprop:
                case dnnl::graph::op::kind::Negative:
                case dnnl::graph::op::kind::PowBackprop:
                case dnnl::graph::op::kind::PReLU:
                case dnnl::graph::op::kind::Quantize:
                case dnnl::graph::op::kind::Reciprocal:
                case dnnl::graph::op::kind::ReLU:
                case dnnl::graph::op::kind::ReLUBackprop:
                case dnnl::graph::op::kind::Reorder:
                case dnnl::graph::op::kind::Round:
                case dnnl::graph::op::kind::Sigmoid:
                case dnnl::graph::op::kind::SigmoidBackprop:
                case dnnl::graph::op::kind::Sign:
                case dnnl::graph::op::kind::SoftMax:
                case dnnl::graph::op::kind::SoftMaxBackprop:
                case dnnl::graph::op::kind::SoftPlus:
                case dnnl::graph::op::kind::SoftPlusBackprop:
                case dnnl::graph::op::kind::Sqrt:
                case dnnl::graph::op::kind::SqrtBackprop:
                case dnnl::graph::op::kind::Square:
                case dnnl::graph::op::kind::Tanh:
                case dnnl::graph::op::kind::TanhBackprop:
                case dnnl::graph::op::kind::TypeCast:
                // infer_pool_bwd_output_shape
                case dnnl::graph::op::kind::MaxPoolBackprop:
                // infer_bias_add_output_shape
                case dnnl::graph::op::kind::BiasAdd:
                    in0 = aop.in_lts_[0].id_;
                    out0 = aop.out_lts_[0].id_;
                    gi[out0] = gi[in0];
                    break;
                // infer_elemwise_arithmetic_output_shape
                // need to handle auto_broadcast
                case dnnl::graph::op::kind::Add:
                case dnnl::graph::op::kind::Divide:
                case dnnl::graph::op::kind::Equal:
                case dnnl::graph::op::kind::Greater:
                case dnnl::graph::op::kind::GreaterEqual:
                case dnnl::graph::op::kind::Less:
                case dnnl::graph::op::kind::LessEqual:
                case dnnl::graph::op::kind::LogicalAnd:
                case dnnl::graph::op::kind::LogicalNot:
                case dnnl::graph::op::kind::LogicalOr:
                case dnnl::graph::op::kind::LogicalXor:
                case dnnl::graph::op::kind::Maximum:
                case dnnl::graph::op::kind::Minimum:
                case dnnl::graph::op::kind::Multiply:
                case dnnl::graph::op::kind::NotEqual:
                case dnnl::graph::op::kind::Pow:
                case dnnl::graph::op::kind::Rsqrt:
                case dnnl::graph::op::kind::SquaredDifference:
                case dnnl::graph::op::kind::Subtract:
                    in0 = aop.in_lts_[0].id_;
                    in1 = aop.in_lts_[1].id_;
                    out0 = aop.out_lts_[0].id_;
                    if (auto_broadcast == "none") {
                        // not checking dim(in0) == dim(in1)
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
                            && aop.attrs_["rounding_type"].get_string()
                                    == "ceil") {
                        floor = false;
                    }
                    strides = aop.attrs_["strides"].get_s64_vector();
                    kernel = aop.attrs_["kernel"].get_s64_vector();
                    if (aop.attrs_.find("dilations") != aop.attrs_.end()) {
                        dilations = aop.attrs_["dilations"].get_s64_vector();
                    } else {
                        dilations.resize(kernel.size());
                        for (size_t i = 0; i < dilations.size(); i++) {
                            dilations[i] = 1;
                        }
                    }
                    cal_pads(pads, aop, spatial_dims, strides, kernel, false);
                    x.clear();
                    for (size_t i = 0; i < spatial_dims.size(); i++) {
                        auto padded = spatial_dims[i] + pads[i];
                        auto dilated = dilations[i] * (kernel[i] - 1) + 1;
                        if (floor) {
                            x.push_back((padded - dilated) / strides[i] + 1);
                        } else {
                            x.push_back(
                                    (padded - dilated - 1) / strides[i] + 2);
                        }
                    }
                    merge_ncx(data_format, gi[out0], n, c, x);
                    break;
                // infer_pool_bwd_output_shape
                case dnnl::graph::op::kind::AvgPoolBackprop:
                    out0 = aop.out_lts_[0].id_;
                    gi[out0] = aop.attrs_["input_shape"].get_s64_vector();
                    break;
                // infer_bn_fwd_train_output_shape
                case dnnl::graph::op::kind::BatchNormForwardTraining:
                // infer_bn_bwd_output_shape
                case dnnl::graph::op::kind::BatchNormTrainingBackprop:
                    in0 = aop.in_lts_[0].id_;
                    out0 = aop.out_lts_[0].id_;
                    gi[out0] = gi[in0];
                    split_ncx(data_format, gi[in0], n, c, x);
                    for (size_t i = 1; i < aop.out_lts_.size(); i++) {
                        gi[aop.out_lts_[i].id_] = {c};
                    }
                    break;
                // infer_bias_backprop_output_shape
                case dnnl::graph::op::kind::BiasAddBackprop:
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
                    axis = aop.attrs_["axis"].get_s64();
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
                // infer_convtranspose_bprop_data_output_shape
                case dnnl::graph::op::kind::ConvTransposeBackpropData:
                    use_oi = 1;
                // infer_conv_output_shape
                case dnnl::graph::op::kind::Convolution:
                    split_ncx(data_format, gi[aop.in_lts_[0].id_], n, c,
                            spatial_dims);
                    split_oix(
                            filter_format, gi[aop.in_lts_[1].id_], oi, kernel);
                    strides = aop.attrs_["strides"].get_s64_vector();
                    dilations = aop.attrs_["dilations"].get_s64_vector();
                    cal_pads(pads, aop, spatial_dims, strides, kernel, false);

                    x.clear();
                    for (size_t i = 0; i < spatial_dims.size(); i++) {
                        auto padded = spatial_dims[i] + pads[i];
                        auto dialated = dilations[i] * (kernel[i] - 1) + 1;
                        x.push_back((padded - dialated) / strides[i] + 1);
                    }
                    merge_ncx(data_format, gi[aop.out_lts_[0].id_], n,
                            oi[use_oi], x);
                    break;
                // infer_conv_bprop_data_output_shape
                case dnnl::graph::op::kind::ConvolutionBackpropData:
                    if (aop.attrs_.find("output_shape") != aop.attrs_.end()) {
                        gi[aop.out_lts_[0].id_]
                                = aop.attrs_["output_shape"].get_s64_vector();
                    }
                    break;
                // infer_convtranspose_output_shape
                case dnnl::graph::op::kind::ConvTranspose:
                    split_ncx(data_format, gi[aop.in_lts_[0].id_], n, c,
                            spatial_dims);
                    split_oix(
                            filter_format, gi[aop.in_lts_[1].id_], oi, kernel);
                    strides = aop.attrs_["strides"].get_s64_vector();
                    dilations = aop.attrs_["dilations"].get_s64_vector();
                    cal_pads(pads, aop, spatial_dims, strides, kernel, true);

                    if (aop.attrs_.find("output_padding") != aop.attrs_.end()) {
                        output_padding
                                = aop.attrs_["output_padding"].get_s64_vector();
                    } else {
                        output_padding.clear();
                        for (size_t i = 0; i < spatial_dims.size(); i++) {
                            output_padding.push_back(0);
                        }
                    }

                    x.clear();
                    for (size_t i = 0; i < spatial_dims.size(); i++) {
                        auto padded = output_padding[i] - pads[i];
                        auto dialated = dilations[i] * (kernel[i] - 1) + 1;
                        x.push_back(strides[i] * (spatial_dims[i] - 1)
                                + dialated + padded);
                    }
                    merge_ncx(data_format, gi[aop.out_lts_[0].id_], n,
                            oi[use_oi] * groups, x);
                    break;
                // infer_conv_bprop_filters_output_shape
                case dnnl::graph::op::kind::ConvolutionBackpropFilters:
                // infer_convtranspose_bprop_filters_output_shape
                case dnnl::graph::op::kind::ConvTransposeBackpropFilters:
                    if (aop.attrs_.find("filter_shape") != aop.attrs_.end()) {
                        gi[aop.out_lts_[0].id_]
                                = aop.attrs_["filter_shape"].get_s64_vector();
                    }
                    break;
                // infer_unsupported_output_shape
                case dnnl::graph::op::kind::DynamicReshape:
                case dnnl::graph::op::kind::DynamicTranspose:
                case dnnl::graph::op::kind::Index:
                case dnnl::graph::op::kind::Wildcard: break;
                // no output, do nothing
                case dnnl::graph::op::kind::End: break;
                // infer_interpolate_output_shape
                case dnnl::graph::op::kind::Interpolate:
                    in0 = aop.in_lts_[0].id_;
                    out0 = aop.out_lts_[0].id_;
                    if (aop.attrs_.find("data_format") != aop.attrs_.end()
                            && aop.attrs_["data_format"].get_string()
                                    == "NCX") {
                        data_format = "NCX";
                    } else {
                        data_format = "NXC";
                    }
                    split_ncx(data_format, gi[in0], n, c, x);
                    if (aop.attrs_.find("scales") != aop.attrs_.end()) {
                        for (size_t i = 0; i < x.size(); i++) {
                            x[i] = x[i]
                                    * aop.attrs_["scales"].get_f32_vector()[i];
                        }
                    } else if (aop.attrs_.find("sizes") != aop.attrs_.end()) {
                        for (size_t i = 0; i < x.size(); i++) {
                            x[i] = aop.attrs_["sizes"].get_s64_vector()[i];
                        }
                    }
                    merge_ncx(data_format, gi[out0], n, c, x);
                    break;
                // infer_norm_output_shape
                case dnnl::graph::op::kind::LayerNorm:
                    in0 = aop.in_lts_[0].id_;
                    out0 = aop.out_lts_[0].id_;
                    gi[out0] = gi[in0];
                    if (aop.attrs_.find("keep_stats") == aop.attrs_.end()
                            || aop.attrs_["keep_stats"].get_bool()) {
                        size_t out1 = aop.out_lts_[1].id_,
                               out2 = aop.out_lts_[2].id_;
                        gi[out1].clear();
                        gi[out2].clear();

                        int64_t axis = -1;
                        if (aop.attrs_.find("begin_norm_axis")
                                != aop.attrs_.end()) {
                            axis = aop.attrs_["begin_norm_axis"].get_s64();
                        }
                        axis = axis >= 0 ? axis : gi[in0].size() + axis;
                        for (int64_t i = 0; i < axis; i++) {
                            gi[out1].push_back(gi[in0][i]);
                            gi[out2].push_back(gi[in0][i]);
                        }
                    }
                    break;
                // infer_norm_bprop_output_shape
                case dnnl::graph::op::kind::LayerNormBackprop:
                    in0 = aop.in_lts_[0].id_;
                    out0 = aop.out_lts_[0].id_;
                    gi[out0] = gi[in0];
                    if (aop.attrs_.find("use_affine") == aop.attrs_.end()
                            || aop.attrs_["use_affine"].get_bool()) {
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
                            && aop.attrs_.find("transpose_a")
                                    != aop.attrs_.end()
                            && aop.attrs_["transpose_a"].get_bool()) {
                        auto tmp = x[x.size() - 1];
                        x[x.size() - 1] = x[x.size() - 2];
                        x[x.size() - 2] = tmp;
                    }
                    if (y.size() > 1
                            && aop.attrs_.find("transpose_b")
                                    != aop.attrs_.end()
                            && aop.attrs_["transpose_b"].get_bool()) {
                        auto tmp = y[y.size() - 1];
                        y[y.size() - 1] = y[y.size() - 2];
                        y[y.size() - 2] = tmp;
                    }
                    if (x.size() == 1 && y.size() == 1) {
                        gi[out0] = {};
                    } else if (x.size() == 1) {
                        n = y[y.size() - 1];
                        y.pop_back();
                        y[y.size() - 1] = n;
                        gi[out0] = y;
                    } else if (y.size() == 1) {
                        n = x[x.size() - 1];
                        x.pop_back();
                        x[x.size() - 1] = n;
                        gi[out0] = x;
                    } else {
                        // crash if x[x.size() - 1] != y[y.size() - 2]
                        size_t a = x[x.size() - 2], b = y[y.size() - 1];
                        x.pop_back();
                        x.pop_back();
                        y.pop_back();
                        y.pop_back();
                        broadcast(x, y, gi[out0]);
                        gi[out0].push_back(a);
                        gi[out0].push_back(b);
                    }
                    break;
                // infer_exponent_output_shape
                case dnnl::graph::op::kind::PowBackpropExponent:
                    gi[aop.out_lts_[0].id_] = gi[aop.in_lts_[3].id_];
                    break;
                // infer_prelu_bwd_output_shape
                case dnnl::graph::op::kind::PReLUBackprop:
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
                        auto axes = aop.attrs_["axes"].get_s64_vector();
                        auto keep_dims = false;
                        if (aop.attrs_.find("keep_dims") != aop.attrs_.end()) {
                            keep_dims = aop.attrs_["keep_dims"].get_bool();
                        }
                        for (size_t i = 0; i < axes.size(); i++) {
                            gi[out0]
                              [(gi[in0].size() + axes[i]) % gi[in0].size()]
                                    = keep_dims ? 1 : 0;
                        }
                        if (!keep_dims) {
                            gi[out0].erase(std::remove_if(gi[out0].begin(),
                                    gi[out0].end(),
                                    [](int64_t d) { return d == 0; }));
                        }

                    } else {
                        // not support
                    }
                    break;
                // infer_select_output_shape
                case dnnl::graph::op::kind::Select:
                    in0 = aop.in_lts_[0].id_;
                    out0 = aop.out_lts_[0].id_;
                    if (auto_broadcast == "none") {
                        gi[out0] = gi[in0];
                    } else {
                        broadcast(gi[aop.in_lts_[1].id_],
                                gi[aop.in_lts_[2].id_], gi[out0]);
                        // one way broadcast only check whether cond can broadcast to the output
                        // no need to do one way broadcast
                    }
                    break;
                // infer_static_reshape_output_shape
                case dnnl::graph::op::kind::StaticReshape:
                    in0 = aop.in_lts_[0].id_;
                    out0 = aop.out_lts_[0].id_;
                    dims = aop.attrs_["shape"].get_s64_vector();
                    special_zero = aop.attrs_["special_zero"].get_bool();
                    axis = -1;
                    for (size_t i = 0; i < dims.size(); i++) {
                        if (dims[i] == 0 && special_zero) {
                            dims[i] = gi[in0][i];
                        } else if (dims[i] == -1) {
                            axis = i;
                        }
                    }
                    in_size = out_size = 1;
                    for (size_t i = 0; i < gi[in0].size(); i++) {
                        in_size = gi[in0][i] >= 0 ? in_size * gi[in0][i]
                                                  : in_size;
                    }
                    for (size_t i = 0; i < dims.size(); i++) {
                        out_size = dims[i] >= 0 ? out_size * dims[i] : out_size;
                    }
                    if (axis != -1) { dims[axis] = in_size / out_size; }
                    gi[out0] = dims;
                    break;
                // infer_static_transpose_output_shape
                case dnnl::graph::op::kind::StaticTranspose:
                    in0 = aop.in_lts_[0].id_;
                    out0 = aop.out_lts_[0].id_;
                    gi[out0].clear();
                    dims = aop.attrs_["order"].get_s64_vector();
                    if (dims.empty()) {
                        for (size_t i = 0; i < gi[in0].size(); i++) {
                            gi[out0].push_back(gi[in0][gi[in0].size() - i - 1]);
                        }
                    } else {
                        for (size_t i = 0; i < gi[in0].size(); i++) {
                            gi[out0].push_back(
                                    gi[in0][(gi[in0].size() + dims[i])
                                            % gi[in0].size()]);
                        }
                    }
                    break;
                case dnnl::graph::op::kind::LastSymbol: break;
            }

            for (auto &lt : aop.in_lts_) {
                lt.shape_ = gi[lt.id_];
                bool is_nxc = std::find(dgraph.nxc_lt.begin(),
                                      dgraph.nxc_lt.end(), lt.id_)
                        != dgraph.nxc_lt.end();
                lt.stride_ = shape_to_stride(gi[lt.id_], is_nxc);
            }
            for (auto &lt : aop.out_lts_) {
                lt.shape_ = gi[lt.id_];
                bool is_nxc = std::find(dgraph.nxc_lt.begin(),
                                      dgraph.nxc_lt.end(), lt.id_)
                        != dgraph.nxc_lt.end();
                lt.stride_ = shape_to_stride(gi[lt.id_], is_nxc);
            }
        }
    }

    void input_shape_rewrite(deserialized_graph &dgraph) {
        for (auto &aop : dgraph.ops_) {
            for (auto &lt : aop.in_lts_) {
                size_t ndims = lt.shape_.size();
                if (dgraph.graph_inputs.find(lt.id_)
                        != dgraph.graph_inputs.end()) {
                    if (in_shapes.find(lt.id_) != in_shapes.end()
                            && in_shapes[lt.id_] != "default") {
                        logical_tensor::dims_t temp_shape
                                = string_to_shape(in_shapes[lt.id_]);
                        // mb rewrite included in shape rewrite
                        if (std::find(dgraph.graph_inputs_with_mb.begin(),
                                    dgraph.graph_inputs_with_mb.end(), lt.id_)
                                        != dgraph.graph_inputs_with_mb.end()
                                && mb != 0) {
                            temp_shape[0] = mb;
                        }
                        if (temp_shape.size() == ndims) {
                            lt.shape_ = temp_shape;
                            dgraph.graph_inputs[lt.id_] = temp_shape;
                            bool is_nxc = std::find(dgraph.nxc_lt.begin(),
                                                  dgraph.nxc_lt.end(), lt.id_)
                                    != dgraph.nxc_lt.end();
                            lt.stride_ = shape_to_stride(lt.shape_, is_nxc);
                        } else {
                            BENCHDNN_PRINT(0,
                                    "Wrong shape dims for tensor: %zd!\n",
                                    lt.id_);
                        }
                    } else if (std::find(dgraph.graph_inputs_with_mb.begin(),
                                       dgraph.graph_inputs_with_mb.end(),
                                       lt.id_)
                                    != dgraph.graph_inputs_with_mb.end()
                            && mb != 0) {
                        lt.shape_[0] = mb;
                        dgraph.graph_inputs[lt.id_] = lt.shape_;
                    }
                } else {
                    logical_tensor::dims_t infer_dim(ndims, -1);
                    lt.shape_ = infer_dim;
                    lt.stride_ = infer_dim;
                }
            }
            for (auto &lt : aop.out_lts_) {
                size_t ndims = lt.shape_.size();
                logical_tensor::dims_t infer_dim(ndims, -1);
                lt.shape_ = infer_dim;
                lt.stride_ = infer_dim;
            }
        }

        std::string shapes_str = "";
        for (auto graph_input : dgraph.graph_inputs) {
            std::string shape_str = std::to_string(graph_input.first) + ":"
                    + shape_to_string(graph_input.second) + " ";
            shapes_str += shape_str;
        }
        BENCHDNN_PRINT(1, "Graph input tensor ids and shapes: %s\n",
                shapes_str.c_str());
    }

    void op_attrs_rewrite(deserialized_graph &dgraph) {
        for (auto &aop : dgraph.ops_) {
            if (op_attrs.find(aop.id_) != op_attrs.end()) {
                std::map<std::string, std::string> temp_attrs
                        = parse_attrs(op_attrs[aop.id_]);
                for (auto &attr_pair : aop.attrs_) {
                    std::string attr_name = attr_pair.first;
                    if (temp_attrs.count(attr_name)) {
                        std::string new_val = temp_attrs[attr_name];
                        auto attr_type = attr_pair.second.get_type();
                        if (attr_type == "string") {
                            attr_pair.second.str_value_ = new_val;
                        } else if (attr_type == "bool") {
                            attr_pair.second.bool_value_
                                    = str2bool(new_val.c_str());
                        } else if (attr_type == "s64") {
                            attr_pair.second.s64_value_ = stoll(new_val);
                        } else if (attr_type == "s64[]") {
                            attr_pair.second.s64_vector_
                                    = string_to_shape(new_val);
                        } else if (attr_type == "f32") {
                            attr_pair.second.f32_value_ = atof(new_val.c_str());
                        } else if (attr_type == "f32[]") {
                            attr_pair.second.f32_vector_
                                    = string_to_f32_vec(new_val);
                        }
                    }
                }
            }
        }
    }

    void quantized_graph_rewrite(deserialized_graph &dgraph) {
        for (auto &aop : dgraph.ops_) {
            if (aop.kind_ != "Dequantize" && aop.kind_ != "Quantize") {
                continue;
            }
            auto &attr = aop.attrs_;
            if (attr.find("scales") != attr.end()
                    && attr.find("zps") != attr.end()
                    && attr.find("qtype") != attr.end()
                    && attr["qtype"].get_string() == "per_channel") {
                auto pre_scales = attr["scales"].get_f32_vector();
                auto pre_zps = attr["zps"].get_s64_vector();
                int64_t axis = 1;
                auto ndims = aop.in_lts_.front().shape_.size();
                if (attr.find("axis") != attr.end()) {
                    axis = (attr["axis"].get_s64() + ndims) % ndims;
                }
                size_t scales_num = aop.in_lts_.front().shape_[axis];
                std::vector<float> scales;
                std::vector<int64_t> zps;
                for (size_t i = 0; i < scales_num; i++) {
                    if (i < pre_scales.size()) {
                        scales.push_back(pre_scales[i]);
                    } else {
                        scales.push_back(pre_scales[0]);
                    }
                    if (i < pre_zps.size()) {
                        zps.push_back(pre_zps[i]);
                    } else {
                        zps.push_back(0);
                    }
                }
                aop.attrs_["scales"].f32_vector_ = scales;
                aop.attrs_["zps"].s64_vector_ = zps;
            }
        }
    }

    std::map<std::string, std::string> parse_attrs(std::string attrs_str) {
        std::map<std::string, std::string> attrs_map;
        attrs_map.clear();
        std::string::size_type key_pos = 0;
        std::string::size_type key_end, val_pos, val_end;
        std::map<size_t, std::string> key_val_case;
        key_val_case.clear();
        while ((key_end = attrs_str.find(':', key_pos)) != std::string::npos) {
            if ((val_pos = attrs_str.find_first_not_of(':', key_end))
                    == std::string::npos)
                break;
            val_end = attrs_str.find('*', val_pos);
            std::string key_str = attrs_str.substr(key_pos, key_end - key_pos);
            std::string val_str = attrs_str.substr(val_pos, val_end - val_pos);
            if (attrs_map.count(key_str)) {
                BENCHDNN_PRINT(0,
                        "Repeat attr: %s, will use first value for it.\n",
                        key_str.c_str());
            } else {
                attrs_map.emplace(key_str, val_str);
            }
            key_pos = val_end;
            if (key_pos != std::string::npos) ++key_pos;
        }
        return attrs_map;
    }

    std::vector<float> string_to_f32_vec(std::string val_str) {
        std::vector<float> f32_vec;
        std::string::size_type pos = 0;
        while ((pos = val_str.find('x')) != std::string::npos) {
            f32_vec.push_back(atof(val_str.substr(0, pos).c_str()));
            val_str.erase(0, pos + 1);
        }
        f32_vec.push_back(atof(val_str.c_str()));
        return f32_vec;
    }

    logical_tensor::dims_t string_to_shape(std::string shape_str) {
        logical_tensor::dims_t shape;
        std::string::size_type pos = 0;
        while ((pos = shape_str.find('x')) != std::string::npos) {
            shape.push_back(stoll(shape_str.substr(0, pos)));
            shape_str.erase(0, pos + 1);
        }
        shape.push_back(stoll(shape_str));
        return shape;
    }

    std::string shape_to_string(logical_tensor::dims_t shape) {
        if (shape.size() > 0) {
            std::stringstream ss;
            std::copy(shape.begin(), shape.end(),
                    std::ostream_iterator<int64_t>(ss, "x"));
            std::string res = ss.str();
            return res.substr(0, res.length() - 1);
        } else {
            return "";
        }
    }

    logical_tensor::dims_t shape_to_stride(
            logical_tensor::dims_t shape, bool is_nxc) {
        const size_t ndims = shape.size();
        logical_tensor::dims_t strides(ndims);
        if (is_nxc) {
            dnnl_dim_t c_dim = shape[1];
            for (size_t i = 2; i < ndims; ++i) {
                shape[i - 1] = shape[i];
            }
            shape[ndims - 1] = c_dim;
        }
        for (auto it = shape.begin(); it < shape.end(); ++it) {
            const auto val = std::accumulate(std::next(it), shape.end(), 1,
                    std::multiplies<dnnl_dim_t>());
            const auto dist = std::distance(shape.begin(), it);
            strides[static_cast<size_t>(dist)] = val;
        }
        if (is_nxc) {
            for (size_t i = ndims - 1; i > 1; --i) {
                strides[i] = strides[i - 1];
            }
            strides[1] = 1;
        }
        return strides;
    }
};

} // namespace graph

#endif
