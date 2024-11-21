/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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
    // Outputs the information about lt from operator<< into a string.
    std::string get_string() const;
};
std::ostream &operator<<(std::ostream &s, const deserialized_lt &dlt);

struct deserialized_op {
    size_t id_;
    std::string name_;
    std::string kind_;
    std::string fpmath_mode_;
    std::string fpmath_mode_apply_to_int_;

    std::unordered_map<std::string, deserialized_attr> attrs_;
    std::vector<deserialized_lt> in_lts_;
    std::vector<deserialized_lt> out_lts_;

    op create() const;

    void load(utils::json::json_reader_t *reader);
    // Outputs the information about op from operator<< into a string.
    std::string get_string() const;

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

    // Returns `true` if `deserialized_op` wasn't created.
    bool empty() const { return kind_.empty(); }
};
std::ostream &operator<<(std::ostream &s, const deserialized_op &dop);

using op_ref_t = std::reference_wrapper<const deserialized_op>;
using op_ref_list_t = std::list<op_ref_t>;

struct deserialized_graph {
    void load(const std::string &pass_config_json);

    dnnl::graph::graph to_graph(const graph_fpmath_mode_t &fpmath_mode) const;
    const std::vector<size_t> &get_input_ports() const { return input_ports_; };

    std::vector<deserialized_op> ops_;
    // record all tensors id and its dims
    std::map<size_t, logical_tensor::dims> graph_tensors_;
    // reorder logical tensor id to memory tag.
    // memory tag can be abx, axb, or other special tag
    // need to maintain for further use
    std::map<size_t, std::string> lt_2_mtag_;
    std::vector<size_t> graph_inputs_with_mb_;

    // Returns an op based on its ID.
    const deserialized_op &get_op(size_t id) const;
    // Returns an op based on its output logical tensor ID.
    const deserialized_op &get_op_by_out_lt(size_t out_lt_id) const;
    // Returns an op based on its input logical tensor ID.
    const deserialized_op &get_op_by_in_lt(size_t in_lt_id) const;

    // Outputs the information about graph from operator<< into a string.
    std::string get_string() const;

    // Return the fpmath mode attribute
    const std::pair<std::string, std::string> get_fpmath_mode() const {
        return std::make_pair(fpmath_mode_, fpmath_mode_apply_to_int_);
    }

    void set_fpmath_mode(const graph_fpmath_mode_t &fpmath_mode) {
        fpmath_mode_ = fpmath_mode.mode_;
        fpmath_mode_apply_to_int_ = bool2str(fpmath_mode.apply_to_int_);
    }

private:
    std::string engine_kind_;
    std::string version_;
    std::string fpmath_mode_;
    std::string fpmath_mode_apply_to_int_;
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
std::ostream &operator<<(std::ostream &s, const deserialized_graph &dg);

} // namespace graph

#endif
