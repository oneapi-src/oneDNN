/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "flex_rewrite.hpp"
#include "parser.hpp"

namespace {
std::string shape_to_string(const dnnl::graph::logical_tensor::dims &shape) {
    if (shape.empty()) return std::string();

    std::stringstream ss;
    std::copy(shape.begin(), shape.end(),
            std::ostream_iterator<int64_t>(ss, "x"));
    auto res = ss.str();
    return res.substr(0, res.length() - 1);
}
} // namespace

namespace graph {

void flex_rewrite::rewrite(deserialized_graph &dgraph) {
    input_shape_rewrite(dgraph);
    if (!(op_attrs_.size() == 1 && op_attrs_.count(0)
                && op_attrs_.at(0) == "default")) {
        op_attrs_rewrite(dgraph);
    }
    infer_output_shape(dgraph);
    quantized_graph_rewrite(dgraph);
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

void flex_rewrite::broadcast(
        const dims_t &x, const dims_t &y, dims_t &z) const {
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
                fprintf(stderr, "graph: failed to broadcast in infer shape!\n");
                exit(2);
            }
            z[i] = (l == 1 ? r : l);
        } else {
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
            exit(2);
        }
    }
}

void flex_rewrite::infer_output_shape(deserialized_graph &dgraph) {
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
                        exit(2);
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
                    assert(x[x.size() - 1] == y[y.size() - 2]);
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
        for (auto &lt : aop.out_lts_) {
            lt.shape_ = gi[lt.id_];
            lt.stride_
                    = memory_tag2strides(gi[lt.id_], dgraph.lt_2_mtag_[lt.id_]);
        }
    }
}

void flex_rewrite::input_shape_rewrite(deserialized_graph &dgraph) {
    // reminder mb rewrite status
    if (mb_ != 0 && dgraph.graph_inputs_with_mb_.empty()) {
        BENCHDNN_PRINT(1,
                "graph: rewrite: Cannot rewrite mb as "
                "%ld!\n",
                (long)mb_);
    }

    const auto set_default_deserialized_lt = [](deserialized_lt &lt) {
        auto ndims = lt.shape_.size();
        logical_tensor::dims infer_dim(ndims, -1);
        lt.shape_ = infer_dim;
        lt.stride_ = infer_dim;
    };

    for_(auto &aop : dgraph.ops_)
    for (auto &lt : aop.in_lts_) {
        if (dgraph.graph_tensors_.find(lt.id_) == dgraph.graph_tensors_.end()) {
            set_default_deserialized_lt(lt);
            continue;
        }

        const bool has_mb_rewrite = mb_ != 0
                && std::find(dgraph.graph_inputs_with_mb_.begin(),
                           dgraph.graph_inputs_with_mb_.end(), lt.id_)
                        != dgraph.graph_inputs_with_mb_.end();
        if (in_shapes_.find(lt.id_) != in_shapes_.end()
                && in_shapes_[lt.id_] != "default") {
            auto temp_shape = string_to_shape(in_shapes_[lt.id_]);
            // mb rewrite included in shape rewrite
            if (has_mb_rewrite) { temp_shape[0] = mb_; }
            size_t ndims = lt.shape_.size();
            if (temp_shape.size() != ndims) {
                BENCHDNN_PRINT(0,
                        "graph: rewrite: driver does not support changing "
                        "shape rank currently, please keep same with origin "
                        "json input for tensor: "
                        "%zd!\n",
                        lt.id_);
                exit(2);
            }
            lt.shape_ = temp_shape;
            dgraph.graph_tensors_[lt.id_] = temp_shape;
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

    std::string shapes_str;
    for (const auto &graph_input : dgraph.graph_tensors_) {
        std::string shape_str = std::to_string(graph_input.first) + ":"
                + shape_to_string(graph_input.second) + " ";
        shapes_str += shape_str;
    }
    BENCHDNN_PRINT(
            1, "Graph input tensor ids and shapes: %s\n", shapes_str.c_str());
}

void flex_rewrite::op_attrs_rewrite(deserialized_graph &dgraph) {
    std::vector<size_t> op_ids_;
    for (const auto &aop : dgraph.ops_) {
        op_ids_.emplace_back(aop.id_);
    }

    for (const auto &temp_attrs : op_attrs_) {
        auto iter = std::find(op_ids_.begin(), op_ids_.end(), temp_attrs.first);
        if (iter == op_ids_.end()) {
            BENCHDNN_PRINT(0, "graph: rewrite: no op id %zd in the graph.\n",
                    temp_attrs.first);
            exit(2);
        }
        auto &temp_op = dgraph.ops_[std::distance(op_ids_.begin(), iter)];
        const auto attrs = parse_attrs(temp_attrs.second);
        for (const auto &new_attr : attrs) {
            auto attr_name = new_attr.first;
            if (!temp_op.attrs_.count(attr_name)) {
                BENCHDNN_PRINT(0,
                        "graph: rewrite: no attr name `%s` in op %zd.\n",
                        attr_name.c_str(), temp_attrs.first);
                exit(2);
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

void flex_rewrite::quantized_graph_rewrite(deserialized_graph &dgraph) {
    for (auto &aop : dgraph.ops_) {
        if (aop.kind_ != "Dequantize" && aop.kind_ != "Quantize") continue;

        auto &attr = aop.attrs_;
        if (attr.find("scales") == attr.end() || attr.find("zps") == attr.end()
                || attr.find("qtype") == attr.end()
                || attr["qtype"].str_value_ != "per_channel")
            continue;

        auto pre_scales = attr["scales"].f32_vector_;
        auto pre_zps = attr["zps"].s64_vector_;
        int64_t axis = 1;
        auto ndims = aop.in_lts_.front().shape_.size();
        if (attr.find("axis") != attr.end()) {
            axis = (attr["axis"].s64_value_ + ndims) % ndims;
        }
        const int64_t scales_zp_dim = aop.in_lts_.front().shape_[axis];
        std::vector<float> scales(scales_zp_dim, pre_scales[0]);

        attr_t::arg_scales_t::entry_t e(policy_t::PER_OC);
        const dnnl_dims_t scales_dims {scales_zp_dim};
        const auto scales_md
                = dnn_mem_t::init_md(1, scales_dims, dnnl_f32, tag::abx);
        dnn_mem_t scales_fp(scales_md, ::get_test_engine());
        dnn_mem_t dummy;
        fill_scales(e, dummy, scales_fp);
        for (int i = 0; i < scales_fp.nelems(); i++)
            scales[i] = scales_fp.get_elem(i);

        std::vector<int64_t> zps;
        for (int64_t i = 0; i < scales_zp_dim; i++) {
            if (static_cast<size_t>(i) < pre_zps.size()) {
                zps.push_back(pre_zps[i]);
            } else {
                zps.push_back(0);
            }
        }
        aop.attrs_["scales"].f32_vector_ = scales;
        aop.attrs_["zps"].s64_vector_ = zps;
    }
}

} // namespace graph
