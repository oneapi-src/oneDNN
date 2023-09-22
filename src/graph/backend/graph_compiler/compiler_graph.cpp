/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "compiler_graph.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace compiler_impl {

static const std::unordered_map<op_kind_t, std::string, utils::enum_hash_t>
        compiler_backend_op {{op_kind::Add, "add"}, {op_kind::Subtract, "sub"},
                {op_kind::Multiply, "mul"}, {op_kind::Divide, "div"},
                {op_kind::Pow, "pow"}, {op_kind::MatMul, "matmul"},
                {op_kind::Quantize, "quantize"},
                {op_kind::Dequantize, "dequantize"},
                {op_kind::DynamicDequantize, "dynamic_dequantize"},
                {op_kind::DynamicQuantize, "dynamic_quantize"},
                {op_kind::StaticReshape, "static_reshape"},
                {op_kind::StaticTranspose, "transpose"},
                {op_kind::SoftMax, "softmax"}, {op_kind::Reorder, "reorder"},
                {op_kind::TypeCast, "cast"}, {op_kind::ReLU, "relu"},
                {op_kind::Sigmoid, "sigmoid"}, {op_kind::GELU, "gelu"},
                {op_kind::ReLUBackward, "relu_backprop"},
                {op_kind::SigmoidBackward, "sigmoid_backprop"},
                {op_kind::GELUBackward, "gelu_backprop"},
                {op_kind::ReduceSum, "reduce_sum"}, {op_kind::BiasAdd, "add"},
                {op_kind::Convolution, "conv_fwd"},
                {op_kind::ConvolutionBackwardData, "conv_bwd_data"},
                {op_kind::ConvolutionBackwardWeights, "conv_bwd_weight"},
                {op_kind::BatchNormForwardTraining,
                        "batchnorm_forward_training"},
                {op_kind::BatchNormTrainingBackward,
                        "batchnorm_training_backprop"},
                {op_kind::Maximum, "max"}, {op_kind::LayerNorm, "layernorm"},
                {op_kind::Select, "select"}, {op_kind::Tanh, "tanh"},
                {op_kind::ReduceMean, "reduce_mean"},
                {op_kind::Concat, "concat"}};

// we convert all int64[] to int32[] except for allowlist
static std::unordered_set<graph::op_attr_t> type_conversion_allowlist {
        graph::op_attr::strides, graph::op_attr::pads_begin,
        graph::op_attr::pads_end, graph::op_attr::weights_shape,
        graph::op_attr::dst_shape, graph::op_attr::dilations};

gc::any_map_t compiler_graph_impl_t::convert_op_attrs(
        const std::unordered_map<graph::op_attr_t,
                graph::utils::attribute_value_t> &attrs) {
    gc::any_map_t backend_attrs;
    for (const auto &attr : attrs) {
        auto kind = attr.second.get_kind();
        auto name = graph::op_t::attr2str(attr.first);
        switch (kind) {
            case attribute_kind::i: {
                auto val = attr.second.get<int64_t>();
                backend_attrs.set(name, (int)val);
            } break;
            case attribute_kind::is: {
                auto val = attr.second.get<std::vector<int64_t>>();
                // TODO(yifei): generalize the logic here
                if (type_conversion_allowlist.find(attr.first)
                        != type_conversion_allowlist.end()) {
                    backend_attrs.set(name, val);
                } else {
                    std::vector<int> val_int32(val.begin(), val.end());
                    backend_attrs.set(name, val_int32);
                }
            } break;
            case attribute_kind::f: {
                auto val = attr.second.get<float>();
                backend_attrs.set(name, val);
            } break;
            case attribute_kind::fs: {
                auto val = attr.second.get<std::vector<float>>();
                backend_attrs.set(name, val);
            } break;
            case attribute_kind::s: {
                auto val = attr.second.get<std::string>();
                backend_attrs.set(name, val);
            } break;
            case attribute_kind::b: {
                auto val = attr.second.get<bool>();
                backend_attrs.set(name, val);
            } break;
            default: assert(0 && "attr value type can't support"); break;
        }
    }
    return backend_attrs;
}

static int convert_axis(int64_t axis, int64_t dim) {
    assert(axis < dim && axis >= -dim);
    return axis < 0 ? dim + axis : axis;
}

gc::sc_op_ptr compiler_graph_impl_t::make_backend_op(const op_t *aop,
        const std::vector<gc::graph_tensor_ptr> &producer_lt,
        const std::vector<gc::graph_tensor_ptr> &consumer_lt) {
    if (!is_supported_op(aop->get_kind())) { return nullptr; }
    gc::any_map_t backend_attrs;
    std::unordered_map<graph::op_attr_t, graph::utils::attribute_value_t> attrs
            = aop->get_attributes();
    auto input_dim = producer_lt[0]->details_.get_plain_dims().size();
    if (aop->get_kind() == op_kind::Quantize
            || aop->get_kind() == op_kind::Dequantize
            || aop->get_kind() == op_kind::DynamicQuantize
            || aop->get_kind() == op_kind::DynamicDequantize) {
        if (attrs.find(graph::op_attr::qtype) != attrs.end()) {
            backend_attrs.set("per_channel",
                    (attrs[graph::op_attr::qtype].get<std::string>()
                            == "per_channel"));
        }
        int64_t axis = attrs.find(graph::op_attr::axis) != attrs.end()
                ? attrs[graph::op_attr::axis].get<int64_t>()
                : 1;
        backend_attrs.set("channel_axis", convert_axis(axis, input_dim));
        if (aop->get_kind() == op_kind::Quantize
                || aop->get_kind() == op_kind::Dequantize) {
            std::vector<float> scales
                    = attrs[graph::op_attr::scales].get<std::vector<float>>();
            std::vector<int64_t> zps_int64
                    = attrs[graph::op_attr::zps].get<std::vector<int64_t>>();
            std::vector<int> zps(zps_int64.begin(), zps_int64.end());
            backend_attrs.set("scales", scales);
            backend_attrs.set("zero_points", zps);
        }
        backend_attrs.set("dtype",
                convert_data_type(aop->get_output_value(0)
                                          ->get_logical_tensor()
                                          .data_type));
    } else if (aop->get_kind() == op_kind::SoftMax) {
        backend_attrs.set("axis",
                std::vector<int>(1,
                        convert_axis(attrs[graph::op_attr::axis].get<int64_t>(),
                                input_dim)));
    } else if (aop->get_kind() == op_kind::StaticReshape) {
        backend_attrs.set("shape",
                attrs[graph::op_attr::shape].get<std::vector<int64_t>>());
        backend_attrs.set("special_zero",
                attrs[graph::op_attr::special_zero].get<bool>());
    } else if (aop->get_kind() == op_kind::StaticTranspose) {
        std::vector<int64_t> order_int64
                = attrs[graph::op_attr::order].get<std::vector<int64_t>>();
        std::vector<int> order(order_int64.size());
        std::transform(order_int64.begin(), order_int64.end(), order.begin(),
                [input_dim](int64_t axis) -> int {
                    return convert_axis(axis, input_dim);
                });
        backend_attrs.set("order", order);
    } else if (aop->get_kind() == op_kind::TypeCast) {
        backend_attrs.set("dtype",
                convert_data_type(aop->get_output_value(0)
                                          ->get_logical_tensor()
                                          .data_type));
    } else if (aop->get_kind() == op_kind::ReduceSum
            || aop->get_kind() == op_kind::ReduceMean) {
        assert(attrs.find(graph::op_attr::axes) != attrs.end());
        std::vector<int64_t> axes
                = attrs[graph::op_attr::axes].get<std::vector<int64_t>>();
        std::vector<int> rd_axis(axes.size());
        std::transform(axes.begin(), axes.end(), rd_axis.begin(),
                [input_dim](int64_t axis) -> int {
                    return convert_axis(axis, input_dim);
                });
        backend_attrs.set("rd_axis", rd_axis);
        if (attrs.find(graph::op_attr::keep_dims) != attrs.end()) {
            backend_attrs.set(
                    "keep_dims", attrs[graph::op_attr::keep_dims].get<bool>());
        } else {
            backend_attrs.set("keep_dims", false);
        }
    } else if (aop->get_kind() == op_kind::BiasAdd) {
        std::string data_format = "NXC";
        if (attrs.find(graph::op_attr::data_format) != attrs.end()) {
            data_format = attrs[graph::op_attr::data_format].get<std::string>();
        }
        if (data_format == "NXC") {
            backend_attrs.set("bc_axis",
                    std::vector<int> {static_cast<int>(input_dim) - 1});
        } else {
            backend_attrs.set("bc_axis", std::vector<int> {1});
        }
    } else {
        backend_attrs = convert_op_attrs(aop->get_attributes());
    }
    return make(compiler_backend_op.find(aop->get_kind())->second, producer_lt,
            consumer_lt, backend_attrs);
}

gc::graph_tensor_ptr compiler_graph_impl_t::convert_logical_tensor(
        const graph::logical_tensor_t &lt) {
    gc::sc_data_format_t lt_format;
    if (lt.ndims == 0) {
        return std::make_shared<gc::graph_tensor>(nullptr, lt_format,
                std::vector<int64_t> {1}, convert_data_type(lt.data_type));
    }
    std::vector<bool> visited(lt.ndims);
    if (lt.layout_type == layout_type::strided) {
        std::vector<int64_t> ordered_strides {
                lt.layout.strides, lt.layout.strides + lt.ndims};
        std::sort(ordered_strides.begin(), ordered_strides.end(),
                std::greater<int64_t>());
        std::vector<int> storage_args(lt.ndims);
        for (int s = 0; s < lt.ndims; ++s) {
            for (int j = 0; j < lt.ndims; ++j) {
                if (!visited[j] && ordered_strides[s] == lt.layout.strides[j]) {
                    storage_args[s] = j;
                    visited[j] = true;
                    break;
                }
            }
        }
        lt_format = gc::sc_data_format_t(storage_args);
        return std::make_shared<gc::graph_tensor>(nullptr, lt_format,
                std::vector<int64_t> {lt.dims, lt.dims + lt.ndims},
                convert_data_type(lt.data_type), ordered_strides);
    }
    return std::make_shared<gc::graph_tensor>(nullptr, lt_format,
            std::vector<int64_t> {lt.dims, lt.dims + lt.ndims},
            convert_data_type(lt.data_type));
}

inline gc::sc_data_type_t compiler_graph_impl_t::convert_data_type(
        data_type_t dtype) {
    if (dtype == data_type::f16)
        return gc::sc_data_type_t(gc::sc_data_etype::F16, 1);
    if (dtype == data_type::bf16)
        return gc::sc_data_type_t(gc::sc_data_etype::BF16, 1);
    if (dtype == data_type::f32)
        return gc::sc_data_type_t(gc::sc_data_etype::F32, 1);
    if (dtype == data_type::s32)
        return gc::sc_data_type_t(gc::sc_data_etype::S32, 1);
    if (dtype == data_type::s8)
        return gc::sc_data_type_t(gc::sc_data_etype::S8, 1);
    if (dtype == data_type::u8 || dtype == data_type::boolean)
        return gc::sc_data_type_t(gc::sc_data_etype::U8, 1);
    assert(0 && "undefined or unknown data_type");
    return gc::sc_data_type_t();
}

gc::sc_op_ptr compiler_graph_impl_t::make_compiler_backend_input(
        const graph::logical_tensor_t &in_lt, const size_t &partition_id) {
    auto lrt = compiler_graph_impl_t::convert_logical_tensor(in_lt);
    auto in_ret = this->make_input({lrt});
    in_ret->attrs_["temp.tensor_id"] = (partition_id << 32) + in_lt.id;
    if (in_lt.property == property_type::constant) {
        in_ret->attrs_.set("constant", 1); // set as local_const
    }
    return in_ret;
}

bool compiler_graph_impl_t::is_supported_op(op_kind_t name) {
    return compiler_backend_op.find(name) != compiler_backend_op.end();
}

} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
