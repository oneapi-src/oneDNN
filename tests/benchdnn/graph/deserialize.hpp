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

#ifndef BENCHDNN_GRAPH_DESERIALIZE_HPP
#define BENCHDNN_GRAPH_DESERIALIZE_HPP

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <stdexcept>

#include "dnnl_graph_common.hpp"
#include "json.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "utils.hpp"

namespace graph {

using namespace dnnl::graph;
using namespace dnnl::graph::impl;

struct deserialized_attr {
    std::string type_;
    std::string str_value_;
    bool bool_value_;
    int64_t s64_value_;
    std::vector<int64_t> s64_vector_;
    float f32_value_;
    std::vector<float> f32_vector_;

    std::string get_type() { return type_; }
    std::string get_string() { return str_value_; }
    bool get_bool() { return bool_value_; }
    int64_t get_s64() { return s64_value_; }
    std::vector<int64_t> get_s64_vector() { return s64_vector_; }
    float get_f32() { return f32_value_; }
    std::vector<float> get_f32_vector() { return f32_vector_; }

    void load(utils::json::json_reader_t *reader) {
        reader->begin_object();
        std::string key_entry;
        std::string value_entry;
        reader->next_object_item(&key_entry);
        if (key_entry == "type") {
            reader->read<std::string>(&type_);
            if (type_ == "string") {
                reader->next_object_item(&value_entry);
                if (value_entry == "value") {
                    reader->read<std::string>(&str_value_);
                }
            } else if (type_ == "bool") {
                reader->next_object_item(&value_entry);
                if (value_entry == "value") {
                    reader->read<bool>(&bool_value_);
                }
            } else if (type_ == "s64") {
                reader->next_object_item(&value_entry);
                if (value_entry == "value") {
                    reader->read<int64_t>(&s64_value_);
                }
            } else if (type_ == "s64[]") {
                reader->next_object_item(&value_entry);
                if (value_entry == "value") {
                    reader->read<std::vector<int64_t>>(&s64_vector_);
                }
            } else if (type_ == "f32") {
                reader->next_object_item(&value_entry);
                if (value_entry == "value") {
                    reader->read<float>(&f32_value_);
                }
            } else if (type_ == "f32[]") {
                reader->next_object_item(&value_entry);
                if (value_entry == "value") {
                    reader->read<std::vector<float>>(&f32_vector_);
                }
            }
            reader->next_object_item(&value_entry);
        }
    }
};

struct deserialized_lt {
    size_t id_;
    std::string data_type_;
    logical_tensor::dims_t shape_;
    logical_tensor::dims_t stride_;
    std::string layout_type_;
    std::string property_type_;

    logical_tensor::data_type get_data_type() {
        if (data_type_ == "f32") {
            return logical_tensor::data_type::f32;
        } else if (data_type_ == "f16") {
            return logical_tensor::data_type::f16;
        } else if (data_type_ == "s8") {
            return logical_tensor::data_type::s8;
        } else if (data_type_ == "u8") {
            return logical_tensor::data_type::u8;
        } else if (data_type_ == "bf16") {
            return logical_tensor::data_type::bf16;
        } else if (data_type_ == "s32") {
            return logical_tensor::data_type::s32;
        } else {
            return logical_tensor::data_type::undef;
        }
    }

    logical_tensor::property_type get_property_type() {
        if (property_type_ == "constant") {
            return logical_tensor::property_type::constant;
        } else if (property_type_ == "variable") {
            return logical_tensor::property_type::variable;
        } else {
            return logical_tensor::property_type::undef;
        }
    }

    logical_tensor create() {
        if (layout_type_ == "any") {
            return logical_tensor(id_, get_data_type(), shape_,
                    logical_tensor::layout_type::any, get_property_type());
        } else {
            return logical_tensor(
                    id_, get_data_type(), shape_, stride_, get_property_type());
        }
    }

    void load(utils::json::json_reader_t *reader) {
        utils::json::read_helper_t helper;

        helper.declare_field("id", &id_);
        helper.declare_field("dtype", &data_type_);
        helper.declare_field("shape", &shape_);
        helper.declare_field("stride", &stride_);
        helper.declare_field("layout_type", &layout_type_);
        helper.declare_field("property_type", &property_type_);
        helper.read_fields(reader);
    }
};

struct deserialized_op {
    size_t id_;
    std::string name_;
    std::string kind_;
    std::unordered_map<std::string, deserialized_attr> attrs_;
    std::vector<deserialized_lt> in_lts_;
    std::vector<deserialized_lt> out_lts_;

    void load(utils::json::json_reader_t *reader) {
        utils::json::read_helper_t helper;

        helper.declare_field("id", &id_);
        helper.declare_field("name", &name_);
        helper.declare_field("kind", &kind_);
        helper.declare_field("attrs", &attrs_);
        helper.declare_field("inputs", &in_lts_);
        helper.declare_field("outputs", &out_lts_);
        helper.read_fields(reader);
    }

    op create() {
        op aop(id_, opstr2kind(kind_), name_);
        for (typename std::unordered_map<std::string,
                     deserialized_attr>::const_iterator it
                = attrs_.begin();
                it != attrs_.end(); ++it) {
            const auto attr_name = it->first;
            auto attr_value = it->second;
            const std::string type = attr_value.get_type();
            if (type == "string") {
                const auto value = attr_value.get_string();
                aop.set_attr(attr_name, value);
            } else if (type == "bool") {
                const auto value = attr_value.get_bool();
                aop.set_attr(attr_name, value);
            } else if (type == "s64") {
                const auto value = attr_value.get_s64();
                aop.set_attr(attr_name, value);
            } else if (type == "s64[]") {
                const auto value = attr_value.get_s64_vector();
                aop.set_attr(attr_name, value);
            } else if (type == "f32") {
                const auto value = attr_value.get_f32();
                aop.set_attr(attr_name, value);
            } else if (type == "f32[]") {
                const auto value = attr_value.get_f32_vector();
                aop.set_attr(attr_name, value);
            } else {
            }
        }

        for (auto lt : in_lts_) {
            aop.add_input(lt.create());
        }
        for (auto lt : out_lts_) {
            aop.add_output(lt.create());
        }

        return aop;
    }
};

struct deserialized_graph {
    std::vector<deserialized_op> ops_;
    std::string engine_kind;
    std::string version;
    std::string fpmath_mode_str;

    std::map<size_t, std::vector<deserialized_op>> in_lt_2_ops;
    std::map<size_t, deserialized_op> out_lt_2_op;
    std::map<size_t, logical_tensor::dims_t> graph_inputs;
    std::vector<size_t> nxc_lt;
    std::vector<size_t> graph_inputs_with_mb;
    std::vector<std::string> op_kind_with_mb {"Abs", "AbsBackprop", "AvgPool",
            "AvgPoolBackprop", "BatchNormForwardTraining", "BatchNormInference",
            "BatchNormTrainingBackprop", "Clamp", "ClampBackprop",
            "Convolution", "ConvolutionBackpropData",
            "ConvolutionBackpropFilters", "ConvTranspose",
            "ConvTransposeBackpropData", "ConvTransposeBackpropFilters", "Elu",
            "EluBackprop", "Exp", "GELU", "GELUBackprop", "HardSwish",
            "HardSwishBackprop", "Interpolate", "InterpolateBackprop",
            "LeakyReLU", "Log", "LogSoftmax", "LogSoftmaxBackprop", "MaxPool",
            "MaxPoolBackprop", "Mish", "MishBackprop", "Reciprocal", "ReLU",
            "ReLUBackprop", "Round", "Sigmoid", "SigmoidBackprop", "SoftMax",
            "SoftMaxBackprop", "SoftPlus", "SoftPlusBackprop", "Sqrt",
            "SqrtBackprop", "Square", "Tanh", "TanhBackprop"};
    std::vector<std::string> op_kind_without_mb {"MatMul"};

    void load(const std::string &pass_config_json) {
        std::ifstream fs(pass_config_json.c_str());
        BENCHDNN_PRINT(
                1, "Deserializing graph from %s\n", pass_config_json.c_str());
        utils::json::json_reader_t read(&fs);
        utils::json::read_helper_t helper;
        helper.declare_field("graph", &ops_);
        helper.declare_field("version", &version);
        helper.declare_field("engine_kind", &engine_kind);
        helper.declare_field("fpmath_mode", &fpmath_mode_str);
        helper.read_fields(&read);
        if (ops_.size()) {
            BENCHDNN_PRINT(1,
                    "The graph was serialized with oneDNN Graph v%s on %s "
                    "eigine "
                    "with %s math mode.\n",
                    version.c_str(), engine_kind.c_str(),
                    fpmath_mode_str.c_str());
        } else {
            BENCHDNN_PRINT(0, "There is no op in the graph %s\n",
                    pass_config_json.c_str());
            exit(2);
        }

        std::map<size_t, size_t> deg; // record indegree for each op
        std::map<size_t, deserialized_op> ops; // op_id -> op
        nxc_lt.clear();
        for (auto aop : ops_) {
            ops[aop.id_] = aop;
            deg[aop.id_] = 0;
            for (auto lt : aop.in_lts_) {
                in_lt_2_ops[lt.id_].push_back(aop);
            }
            for (auto lt : aop.out_lts_) {
                out_lt_2_op[lt.id_] = aop;
                // collect graph internal and output tensors memory layout
                if (lt.stride_.size() > 2 && lt.stride_[1] == 1
                        && lt.stride_[lt.stride_.size() - 1] != 1) {
                    nxc_lt.emplace_back(lt.id_);
                }
            }
        }

        for (auto item : in_lt_2_ops) {
            // count indegree for each op
            // do not count if input is a external input (does not contain a output)
            if (out_lt_2_op.find(item.first) != out_lt_2_op.end()) {
                for (auto aop : item.second) {
                    deg[aop.id_]++;
                }
            }
        }

        ops_.clear();

        for (auto item : deg) {
            if (item.second == 0) { ops_.push_back(ops[item.first]); }
        }
        for (size_t idx = 0; idx < ops_.size(); idx++) {
            auto op = ops_[idx];
            // for each output id of the op, find the ops with the same input id
            // check the input
            for (auto out : op.out_lts_) {
                for (auto aop : in_lt_2_ops[out.id_]) {
                    deg[aop.id_]--;
                    if (deg[aop.id_] == 0) { ops_.push_back(ops[aop.id_]); }
                }
            }
        }
        if (ops.size() != ops_.size()) {
            BENCHDNN_PRINT(0, "FAIL: the graph %s is not a DAG.\n",
                    pass_config_json.c_str());
            exit(2);
        }

        for (auto in_lt : in_lt_2_ops) {
            if (out_lt_2_op.find(in_lt.first) == out_lt_2_op.end()) {
                auto aop = in_lt_2_ops[in_lt.first][0];
                for (auto lt : aop.in_lts_) {
                    if (lt.id_ == in_lt.first) {
                        // collect graph input tensors memory layout
                        if (lt.stride_.size() > 2 && lt.stride_[1] == 1
                                && lt.stride_[lt.stride_.size() - 1] != 1) {
                            nxc_lt.emplace_back(lt.id_);
                        }
                        graph_inputs.emplace(in_lt.first, lt.shape_);
                    }
                }
            }
        }

        for (auto graph_in : graph_inputs) {
            if (check_input_with_mb(graph_in.first)) {
                graph_inputs_with_mb.push_back(graph_in.first);
            }
        }
    }

    dnnl::graph::graph to_graph(dnnl::graph::graph::fpmath_mode fpmath_mode) {
        const dnnl::graph::engine &engine = benchdnnext::get_test_engine();
        dnnl::graph::graph g(engine.get_kind(), fpmath_mode);
        for (auto aop : ops_) {
            g.add_op(aop.create());
        }
        return g;
    }

    bool check_input_with_mb(size_t input_lt_id) {
        if (in_lt_2_ops.find(input_lt_id) == in_lt_2_ops.end()) return false;
        for (auto aop : in_lt_2_ops[input_lt_id]) {
            if (std::find(op_kind_without_mb.begin(), op_kind_without_mb.end(),
                        aop.kind_)
                    != op_kind_without_mb.end()) {
                return false;
            } else if (std::find(op_kind_with_mb.begin(), op_kind_with_mb.end(),
                               aop.kind_)
                            != op_kind_with_mb.end()
                    && (input_lt_id == aop.in_lts_[0].id_)) {
                return true;
            } else {
                return check_input_with_mb(aop.out_lts_[0].id_);
            }
        }
        return false;
    }
};

} // namespace graph

#endif
