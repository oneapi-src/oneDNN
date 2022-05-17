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

#ifndef COMMON_DESERIALIZE_HPP
#define COMMON_DESERIALIZE_HPP

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "json.hpp"
#include "utils.hpp"

using namespace dnnl::graph;

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

    void load(impl::utils::json::json_reader_t *reader) {
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

    void load(impl::utils::json::json_reader_t *reader) {
        impl::utils::json::read_helper_t helper;

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

    void load(impl::utils::json::json_reader_t *reader) {
        impl::utils::json::read_helper_t helper;

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

    dnnl::graph::graph load(const std::string &pass_config_json) {
        std::ifstream fs(pass_config_json.c_str());
        std::cout << "deserializing graph from " << pass_config_json << "\n";

        impl::utils::json::json_reader_t read(&fs);
        impl::utils::json::read_helper_t helper;
        std::string version;
        std::string engine_kind;
        std::string fpmath_mode;
        helper.declare_field("graph", &ops_);
        helper.declare_field("version", &version);
        helper.declare_field("engine_kind", &engine_kind);
        helper.declare_field("fpmath_mode", &fpmath_mode);
        helper.read_fields(&read);
        graph g(engine_kind_str2kind(engine_kind),
                fpmath_mode_kind_str2kind(fpmath_mode));
        std::cout << "the graph was serialized with oneDNN Graph v" << version
                  << "\n";

        for (auto aop : ops_) {
            g.add_op(aop.create());
        }
        return g;
    }
};

#endif
