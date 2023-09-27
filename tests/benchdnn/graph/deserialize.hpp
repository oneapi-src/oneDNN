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

#ifndef BENCHDNN_GRAPH_DESERIALIZE_HPP
#define BENCHDNN_GRAPH_DESERIALIZE_HPP

#include "oneapi/dnnl/dnnl_graph.hpp"

// This file provides basic functionality to parse JSON file. It doesn't tie
// library internals to the benchmark.
// It requires user to define an object to parse and `load` routines.
#include "src/graph/utils/json.hpp"

#include "utils.hpp"

namespace graph {

using namespace dnnl::graph;
using namespace dnnl::impl::graph;

struct deserialized_attr {
    std::string type_;
    std::string str_value_;
    bool bool_value_;
    int64_t s64_value_;
    std::vector<int64_t> s64_vector_;
    float f32_value_;
    std::vector<float> f32_vector_;

    void load(utils::json::json_reader_t *reader);
};

struct deserialized_lt {
    size_t id_;
    std::string data_type_;
    logical_tensor::dims shape_;
    logical_tensor::dims stride_;
    std::string layout_type_;
    std::string property_type_;

    logical_tensor::data_type get_data_type() const;

    logical_tensor::property_type get_property_type() const;

    logical_tensor create() const;

    void load(utils::json::json_reader_t *reader);
};

struct deserialized_op {
    size_t id_;
    std::string name_;
    std::string kind_;
    std::unordered_map<std::string, deserialized_attr> attrs_;
    std::vector<deserialized_lt> in_lts_;
    std::vector<deserialized_lt> out_lts_;

    op create() const;

    void load(utils::json::json_reader_t *reader);

    bool get_attr_string(std::string &attr, const std::string &attr_name) const;

    bool get_attr_bool(bool &attr, const std::string &attr_name) const;

    bool get_attr_f32(float &attr, const std::string &attr_name) const;

    bool get_attr_s64(int64_t &attr, const std::string &attr_name) const;

    bool get_attr_f32_vector(
            std::vector<float> &attr, const std::string &attr_name) const;

    bool get_attr_s64_vector(
            std::vector<int64_t> &attr, const std::string &attr_name) const;

    bool has_NXC_format() const;

    logical_tensor::dims get_NCX_shape(size_t idx, bool input) const;
};

using op_ref_list_t = std::list<std::reference_wrapper<const deserialized_op>>;

struct deserialized_graph {
    void load(const std::string &pass_config_json);

    dnnl::graph::graph to_graph(dnnl::fpmath_mode fpmath_mode) const;
    const std::vector<size_t> &get_input_ports() const { return input_ports_; };

    std::vector<deserialized_op> ops_;
    // record all tensors id and its dims
    std::map<size_t, logical_tensor::dims> graph_tensors_;
    // reorder logical tensor id to memory tag.
    // memory tag can be abx, axb, or other special tag
    // need to maintain for further use
    std::map<size_t, std::string> lt_2_mtag_;
    std::vector<size_t> graph_inputs_with_mb_;

private:
    std::string engine_kind_;
    std::string version_;
    std::string fpmath_mode_;
    std::vector<size_t> input_ports_;
    std::vector<size_t> output_ports_;

    std::map<size_t, std::vector<deserialized_op>> in_lt_2_ops_;
    std::map<size_t, deserialized_op> out_lt_2_op_;
    std::vector<std::string> binary_ops_ {"Add", "BiasAdd", "Divide", "Maximum",
            "Minimum", "Multiply", "Substract"};
    // need change dst_shape or weight_shape attribute value
    std::vector<std::string> unsupport_mb_rewrite_ops_ {
            "ConvolutionBackwardData", "ConvolutionBackwardWeights",
            "ConvTransposeBackwardWeights"};
    // bwd ops have multiple inputs
    std::vector<std::string> bwd_ops_ {"AbsBackward", "AvgPoolBackward",
            "BatchNormTrainingBackward", "BiasAddBackward", "ClampBackward",
            "ConvolutionBackwardData", "ConvolutionBackwardWeights",
            "ConvTransposeBackwardData", "ConvTransposeBackwardWeights",
            "EluBackward", "GELUBackward", "HardSwishBackward",
            "InterpolateBackward", "LayerNormBackward", "LogSoftmaxBackward",
            "MaxPoolBackward", "MishBackward", "ReLUBackward",
            "SigmoidBackward", "SoftMaxBackward", "SoftPlusBackward",
            "SqrtBackward", "TanhBackward"};

    bool check_tensor_with_mb(size_t tensor_id) const;
};

} // namespace graph

#endif
