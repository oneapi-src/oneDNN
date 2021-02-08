/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <limits>
#include <memory>

#include "op_schema.hpp"
#include "opset.hpp"
#include "utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {

op_schema::op_schema() : name_("unknown"), version_(0) {}
op_schema::op_schema(std::string op_name, opset_version version)
    : name_(std::move(op_name)), version_(version) {}

op_schema &op_schema::set_name(const std::string &name) {
    name_ = name;
    return *this;
}

const std::string &op_schema::get_name() const {
    return name_;
}

op_schema &op_schema::set_doc(const std::string &doc) {
    doc_ = doc;
    return *this;
}

const std::string &op_schema::get_doc() const {
    return doc_;
}

op_schema &op_schema::since_version(opset_version n) {
    version_ = n;
    return *this;
}

opset_version op_schema::get_since_version() const {
    return version_;
}

op_schema &op_schema::set_num_inputs(std::set<size_t> input_num) {
    num_inputs_ = std::move(input_num);
    return *this;
}

op_schema &op_schema::set_num_inputs(size_t input_num) {
    num_inputs_.insert(input_num);
    return *this;
}

std::set<size_t> op_schema::get_num_inputs() const {
    return num_inputs_;
}

op_schema &op_schema::set_num_outputs(std::set<size_t> output_num) {
    num_outputs_ = std::move(output_num);
    return *this;
}

op_schema &op_schema::set_num_outputs(size_t output_num) {
    num_outputs_.insert(output_num);
    return *this;
}

std::set<size_t> op_schema::get_num_outputs() const {
    return num_outputs_;
}

op_schema &op_schema::set_input(size_t in_offset, std::string in_name,
        const std::string &in_description) {
    validate_input_(in_offset);
    inputs_.emplace_back(formal_parameter(std::move(in_name), in_description));

    return *this;
}

const std::vector<op_schema::formal_parameter> &op_schema::get_inputs() const {
    return inputs_;
}

op_schema &op_schema::set_output(size_t out_offset, std::string out_name,
        const std::string &out_description) {
    validate_output_(out_offset);
    outputs_.emplace_back(
            formal_parameter(std::move(out_name), out_description));

    return *this;
}

const std::vector<op_schema::formal_parameter> &op_schema::get_outputs() const {
    return outputs_;
}

op_schema &op_schema::set_attr(const std::string &name,
        const std::string &description, bool required) {
    assertm(attributes_.count(name) == 0,
            "provided attribute has already been set");
    attributes_[name] = attribute(name, description, required);
    return *this;
}

op_schema &op_schema::set_attr(const std::string &name,
        const std::string &description, bool required, const char *value) {
    assertm(attributes_.count(name) == 0,
            "provided attribute has already been set");
    attributes_[name]
            = attribute(name, description, required, {std::string(value)});
    return *this;
}

const std::unordered_map<std::string, op_schema::attribute> &
op_schema::get_attrs() const {
    return attributes_;
}

op_schema &op_schema::set_shape_inference_function(shape_infer_fn fn) {
    tensor_inference_function_ = std::move(fn);
    return *this;
}

shape_infer_fn op_schema::get_shape_inference_function() const {
    return tensor_inference_function_;
}

bool op_schema::verify_param_num(size_t actual_num,
        std::set<size_t> expected_num, param_num_option option) const {
    switch (option) {
        case param_num_option::fixed: {
            // fixed option only has one valid number
            if (expected_num.size() != 1
                    || expected_num.find(actual_num) == expected_num.end()) {
                return false;
            }
        } break;
        case param_num_option::optional: {
            if (expected_num.find(actual_num) == expected_num.end())
                return false;
        } break;
        case param_num_option::variadic: {
            auto lt = expected_num.begin();
            auto gt = expected_num.upper_bound(actual_num);
            auto end = expected_num.end();
            if ((lt != end && *lt > actual_num) || lt == end || gt == end)
                return false;
        } break;
        default: return false;
    }
    return true;
}

void op_schema::set_default_attribute(op_t *l_op) const {
    auto actual_attrs = l_op->get_attributes();
    auto expected_attrs = this->get_attrs();
    for (auto iter = expected_attrs.begin(); iter != expected_attrs.end();
            ++iter) {
        // if default attribute not set in op, set it to default value
        if (!iter->second.required_ && actual_attrs.count(iter->first) == 0) {
            auto value = iter->second.attr_;
            const auto &name = iter->first;
            l_op->set_attr(name, value);
        }
    }
}

bool op_schema::verify(op_t *l_op) const {
    size_t actual_num_inputs = l_op->inputs().size();
    auto expected_num_inputs = get_num_inputs();
    bool param_num_verify_result = verify_param_num(
            actual_num_inputs, expected_num_inputs, inputs_option);
    if (!param_num_verify_result) { return false; }

    size_t actual_num_outputs = l_op->outputs().size();
    auto expected_num_outputs = get_num_outputs();
    param_num_verify_result = verify_param_num(
            actual_num_outputs, expected_num_outputs, outputs_option);
    if (!param_num_verify_result) { return false; }

    const auto &actual_attrs = l_op->get_attributes();
    auto expected_attrs = get_attrs();
    for (auto iter = expected_attrs.begin(); iter != expected_attrs.end();
            ++iter) {
        if (iter->second.required_ && actual_attrs.count(iter->first) == 0) {
            return false;
        }
    }
    return true;
}

status_t op_schema::shape_infer(node_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) const {
    shape_infer_fn fn = get_shape_inference_function();
    return fn(n, inputs, outputs);
}

size_t op_schema::get_max_valid_param_num(
        const std::set<size_t> &param_num, param_num_option option) const {
    size_t max_valid_num = 0;
    if (option == param_num_option::fixed
            || option == param_num_option::optional) {
        if (!param_num.empty()) { max_valid_num = *param_num.rbegin(); }
    } else {
        max_valid_num = std::numeric_limits<size_t>::max();
    }

    return max_valid_num;
}

void op_schema::validate_input_(size_t in_offset) {
    assertm(inputs_offset.find(in_offset) == inputs_offset.end(),
            "provided `in_offset` has already been set");

    inputs_offset.insert(in_offset);
    size_t max_valid_num = get_max_valid_param_num(num_inputs_, inputs_option);
    assertm(max_valid_num > 0, "input set before setting num_inputs_");
    assertm(in_offset < max_valid_num,
            "input offset exceeds declared num of inputs");
    UNUSED(max_valid_num);
}

void op_schema::validate_output_(size_t out_offset) {
    assertm(outputs_offset.find(out_offset) == outputs_offset.end(),
            "provided `out_offset` has already been set");

    outputs_offset.insert(out_offset);
    size_t max_valid_num
            = get_max_valid_param_num(num_outputs_, outputs_option);
    assertm(max_valid_num > 0, "output set before setting num_outputs_");
    assertm(out_offset < max_valid_num,
            "output offset exceeds declared num of outputs");
    UNUSED(max_valid_num);
}

op_schema &op_schema::set_inputs_option(param_num_option option) {
    inputs_option = option;
    return *this;
}

op_schema::param_num_option op_schema::get_inputs_option() const {
    return inputs_option;
}

op_schema &op_schema::set_outputs_option(param_num_option option) {
    outputs_option = option;
    return *this;
}

op_schema::param_num_option op_schema::get_outputs_option() const {
    return outputs_option;
}

op_schema_registry::op_schema_registry_once::op_schema_registry_once(
        op_schema &schema) {
    auto &op_map = get_map_without_ensuring_registration();

    auto &op_name = schema.get_name();
    auto op_version = schema.get_since_version();

    op_map[op_name].insert(std::pair<opset_version, op_schema &&>(
            op_version, std::move(schema)));
}

op_name_version_schema_map &
op_schema_registry::get_map_without_ensuring_registration() {
    static op_name_version_schema_map op_map;
    return op_map;
}

op_name_version_schema_map &op_schema_registry::get_map() {
    auto &op_map = get_map_without_ensuring_registration();
    class register_opset_t {
    public:
        register_opset_t() { register_opset_schema(); }
    };
    static register_opset_t ro;

    return op_map;
}

const op_schema *op_schema_registry::get_op_schema(op_kind_t kind) {
    const std::string opname = op_t::kind2str(kind);
    auto &op_map = get_map();
    if (op_map.count(opname)) {
        return &op_map[opname].rbegin()->second;
    } else {
        return nullptr;
    }
}

void register_schema(op_schema &&schema) {
    op_schema_registry::op_schema_registry_once DNNL_GRAPH_UNUSED registration(
            schema);
}

} // namespace impl
} // namespace graph
} // namespace dnnl
