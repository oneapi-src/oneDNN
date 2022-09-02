/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "interface/op_schema.hpp"
#include "interface/opset.hpp"

#include "utils/utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {

op_schema_t::op_schema_t() : op_kind_(op_kind::LastSymbol), version_(0) {}

op_schema_t::op_schema_t(op_kind_t kind, opset_version version)
    : op_kind_(kind), version_(version) {}

// the rvalue reference design is based on the fact that these
// functions are only called internally with rvalue intputs.
op_schema_t &op_schema_t::set_op_kind(op_kind_t kind) {
    op_kind_ = kind;
    return *this;
}

op_kind_t op_schema_t::get_op_kind() const {
    return op_kind_;
}

op_schema_t &op_schema_t::set_doc(std::string &&doc) {
    doc_ = std::move(doc);
    return *this;
}

const std::string &op_schema_t::get_doc() const {
    return doc_;
}

op_schema_t &op_schema_t::since_version(opset_version n) {
    version_ = n;
    return *this;
}

opset_version op_schema_t::get_since_version() const {
    return version_;
}

op_schema_t &op_schema_t::set_num_inputs(std::set<size_t> &&input_num) {
    num_inputs_ = std::move(input_num);
    return *this;
}

op_schema_t &op_schema_t::set_num_inputs(size_t input_num) {
    num_inputs_.insert(input_num);
    return *this;
}

std::set<size_t> op_schema_t::get_num_inputs() const {
    return num_inputs_;
}

op_schema_t &op_schema_t::set_num_outputs(std::set<size_t> &&output_num) {
    num_outputs_ = std::move(output_num);
    return *this;
}

op_schema_t &op_schema_t::set_num_outputs(size_t output_num) {
    num_outputs_.insert(output_num);
    return *this;
}

std::set<size_t> op_schema_t::get_num_outputs() const {
    return num_outputs_;
}

op_schema_t &op_schema_t::set_input(size_t in_offset, std::string &&in_name,
        std::string &&in_description, std::string &&dtype_string) {
    verify_input_(in_offset);
    inputs_.emplace_back(op_parameter_t(std::move(in_name),
            std::move(in_description), std::move(dtype_string)));
    return *this;
}

const std::vector<op_schema_t::op_parameter_t> &
op_schema_t::get_inputs() const {
    return inputs_;
}

op_schema_t &op_schema_t::set_output(size_t out_offset, std::string &&out_name,
        std::string &&out_description, std::string &&dtype_string) {
    verify_output_(out_offset);
    outputs_.emplace_back(op_parameter_t(std::move(out_name),
            std::move(out_description), std::move(dtype_string)));
    return *this;
}

op_schema_t &op_schema_t::set_commutative_inputs() {
    assertm(num_inputs_.size() == 1 && *(num_inputs_.begin()) == 2,
            "commutative inputs can only be enabled for ops with two inputs");
    commutative_inputs_enabled_ = true;
    return *this;
}

bool op_schema_t::get_commutative_inputs() const {
    return commutative_inputs_enabled_;
}

op_schema_t &op_schema_t::set_type_constraints(
        std::string &&dtype_string, std::set<data_type_t> &&dtypes) {
    op_parameter_dtype_map_[std::move(dtype_string)] = std::move(dtypes);
    return *this;
}

const std::vector<op_schema_t::op_parameter_t> &
op_schema_t::get_outputs() const {
    return outputs_;
}

op_schema_t &op_schema_t::set_attr(op_attr_t name, std::string &&description,
        bool required, attribute_kind_t attr_kind) {
    assertm(attributes_.count(name) == 0,
            "provided attribute has already been set");
    attributes_[name]
            = attribute_t(name, std::move(description), required, attr_kind);
    return *this;
}

op_schema_t &op_schema_t::set_attr(op_attr_t name, std::string &&description,
        bool required, attribute_kind_t attr_kind, const char *value) {
    assertm(attributes_.count(name) == 0,
            "provided attribute has already been set");
    attributes_[name] = attribute_t(name, std::move(description), required,
            attr_kind, {std::string(value)});
    return *this;
}

const std::unordered_map<op_attr_t, op_schema_t::attribute_t> &
op_schema_t::get_attrs() const {
    return attributes_;
}

op_schema_t &op_schema_t::set_shape_inference_function(shape_infer_fn fn) {
    tensor_inference_function_ = std::move(fn);
    return *this;
}

shape_infer_fn op_schema_t::get_shape_inference_function() const {
    return tensor_inference_function_;
}

op_schema_t &op_schema_t::set_type_constraint_function(type_constraint_fn fn) {
    op_type_constraint_function_ = std::move(fn);
    return *this;
}

type_constraint_fn op_schema_t::get_type_constraint_function() const {
    return op_type_constraint_function_;
}

op_schema_t &op_schema_t::set_additional_item(
        const std::string &key, const utils::any_t &value) {
    additional_items_map_.insert({key, value});
    return *this;
}

const utils::any_t &op_schema_t::get_additional_item(
        const std::string &key) const {
    auto it = additional_items_map_.find(key);
    assertm(it != additional_items_map_.end(), "don't have such item");
    return it->second;
}

bool op_schema_t::has_additional_item(const std::string &key) const {
    return additional_items_map_.count(key);
}

bool op_schema_t::verify_param_num(size_t actual_num,
        const std::set<size_t> &expected_num, param_num_option option) const {
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
            if (expected_num.size() != 2) return false;
            auto lower = expected_num.begin();
            auto upper = lower++;
            if (*lower > *upper) {
                upper = expected_num.begin();
                lower = upper++;
            }
            if (*lower > actual_num || *upper < actual_num) return false;
        } break;
        default: return false;
    }
    return true;
}

bool op_schema_t::verify_param_dtype(
        const std::vector<std::shared_ptr<value_t>> &actual_values,
        const std::vector<op_schema_t::op_parameter_t> &expected_params,
        param_num_option option,
        std::unordered_map<std::string, std::set<data_type_t>>
                &dtype_constraints) const {
    size_t offset = 0;
    for (auto &v : actual_values) {
        const logical_tensor_t &lt = v->get_logical_tensor();
        const std::string &dtype_string = expected_params[offset].dtype_string_;
        if (dtype_string == "any") continue;
        std::set<data_type_t> &expected_dtypes
                = dtype_constraints[dtype_string];
        if (expected_dtypes.find(lt.data_type) == expected_dtypes.end())
            return false;

        if (expected_dtypes.size() != 1) {
            // dtype for current dtype_string has not been fixed
            // fix the dtype for current dtype_string
            dtype_constraints[dtype_string] = {lt.data_type};
        }

        if (option != param_num_option::variadic) { offset += 1; }
    }

    return true;
}

bool op_schema_t::verify_attributes(
        const std::unordered_map<op_attr_t, utils::attribute_value_t>
                &actual_attrs,
        const std::unordered_map<op_attr_t, attribute_t> &expected_attrs,
        bool check_undefined_attrs) const {
    // check if required attributes are not provided
    for (const auto &elem : expected_attrs) {
        if (elem.second.required_ && actual_attrs.count(elem.first) == 0) {
            DEBUG_PRINT_ERROR("attribute \"" + op_t::attr2str(elem.first)
                    + "\" is required but not set");
            return false;
        }
    }
    // check if the data types of actual attributes meet requirements
    for (const auto &elem : actual_attrs) {
        const op_attr_t attr_name = elem.first;
        if (expected_attrs.count(attr_name) != 0
                && elem.second.get_kind()
                        != expected_attrs.at(attr_name).attr_kind_) {
            DEBUG_PRINT_ERROR("attribute \"" + op_t::attr2str(elem.first)
                    + "\" has invalid type");
            return false;
        }
    }

    // check if user set undefined attributes
    if (check_undefined_attrs) {
        for (const auto &elem : actual_attrs) {
            if (expected_attrs.count(elem.first) == 0) {
                DEBUG_PRINT_ERROR("attribute \"" + op_t::attr2str(elem.first)
                        + "\" is not defined in spec");
                return false;
            }
        }
    }

    return true;
}

void op_schema_t::set_default_attribute(op_t *l_op) const {
    const std::unordered_map<op_attr_t, utils::attribute_value_t> &actual_attrs
            = l_op->get_attributes();
    const std::unordered_map<op_attr_t, op_schema_t::attribute_t>
            &expected_attrs = this->get_attrs();
    for (auto iter = expected_attrs.begin(); iter != expected_attrs.end();
            ++iter) {
        // if default attribute not set in op, set it to default value
        if (iter->second.has_default_value_
                && actual_attrs.count(iter->first) == 0) {
            utils::attribute_value_t value = iter->second.attr_;
            op_attr_t name = iter->first;
            l_op->set_attr(name, value);
        }
    }
}

bool op_schema_t::verify(const op_t *l_op, bool check_undefined_attrs) const {
    size_t actual_num_inputs = l_op->num_inputs();
    std::set<size_t> expected_num_inputs = get_num_inputs();
    bool param_num_verify_result = verify_param_num(
            actual_num_inputs, expected_num_inputs, inputs_option);
    if (!param_num_verify_result) { return false; }

    // this is used to pass input dtype constraints to output
    bool param_dtype_verify_result = true;
    std::unordered_map<std::string, std::set<data_type_t>> dtype_constraints
            = op_parameter_dtype_map_;
    param_dtype_verify_result = verify_param_dtype(l_op->get_input_values(),
            inputs_, inputs_option, dtype_constraints);
    if (!param_dtype_verify_result) { return false; }

    size_t actual_num_outputs = l_op->num_outputs();
    std::set<size_t> expected_num_outputs = get_num_outputs();
    param_num_verify_result = verify_param_num(
            actual_num_outputs, expected_num_outputs, outputs_option);
    if (!param_num_verify_result) { return false; }

    param_dtype_verify_result = verify_param_dtype(l_op->get_output_values(),
            outputs_, outputs_option, dtype_constraints);
    if (!param_dtype_verify_result) { return false; }

    type_constraint_fn tc_fn = get_type_constraint_function();
    if (tc_fn) { param_dtype_verify_result = tc_fn(l_op); }
    if (!param_dtype_verify_result) { return false; }

    auto attrs = l_op->get_attributes();
    bool attr_verify_result
            = verify_attributes(attrs, attributes_, check_undefined_attrs);

    if (!attr_verify_result) { return false; };

    // additional check for Quantize and Dequantize
    // TODO(team): better to move it op schema as a constraint.
    const op_kind_t opk = l_op->get_kind();
    if (opk == op_kind::Quantize || opk == op_kind::Dequantize) {
        // attrs should always have the attributes, verified above.
        const size_t sz_scales
                = attrs[op_attr::scales].get<std::vector<float>>().size();
        const size_t sz_zps
                = attrs[op_attr::zps].get<std::vector<int64_t>>().size();
        if (sz_scales != sz_zps) return false;

        // qtype is not a required attribute.
        auto it = attrs.find(op_attr::qtype);
        if (it == attrs.end()
                || it->second.get<std::string>() == "per_tensor") {
            if (sz_scales != 1 || sz_zps != 1) { return false; }
        }
    }
    return true;
}

status_t op_schema_t::shape_infer(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) const {
    shape_infer_fn fn = get_shape_inference_function();
    return fn(n, inputs, outputs);
}

size_t op_schema_t::get_max_valid_param_num(
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

void op_schema_t::verify_input_(size_t in_offset) {
    assertm(inputs_offset.find(in_offset) == inputs_offset.end(),
            "provided `in_offset` has already been set");

    inputs_offset.insert(in_offset);
    size_t max_valid_num = get_max_valid_param_num(num_inputs_, inputs_option);
    assertm(max_valid_num > 0, "input set before setting num_inputs_");
    assertm(in_offset < max_valid_num,
            "input offset exceeds declared num of inputs");
    UNUSED(max_valid_num);
}

void op_schema_t::verify_output_(size_t out_offset) {
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

op_schema_t &op_schema_t::set_inputs_option(param_num_option option) {
    inputs_option = option;
    return *this;
}

op_schema_t::param_num_option op_schema_t::get_inputs_option() const {
    return inputs_option;
}

op_schema_t &op_schema_t::set_outputs_option(param_num_option option) {
    outputs_option = option;
    return *this;
}

op_schema_t::param_num_option op_schema_t::get_outputs_option() const {
    return outputs_option;
}

op_schema_registry_t::op_schema_registry_once_t::op_schema_registry_once_t(
        op_schema_t &&schema) {
    op_kind_version_schema_map &op_map
            = get_map_without_ensuring_registration();

    const op_kind_t kind = schema.get_op_kind();
    opset_version op_version = schema.get_since_version();

    // The schema registry may be being written by one thread to register
    // internal ops for a backend. At the same time, the schema registry may
    // also be being read by another thread to add a spec op into a graph. So,
    // we use the read/write lock here to avoid data race.
    get_rw_mutex().lock_write();
    op_map[kind].insert(std::pair<opset_version, op_schema_t &&>(
            op_version, std::move(schema)));
    get_rw_mutex().unlock_write();
}

op_kind_version_schema_map &
op_schema_registry_t::get_map_without_ensuring_registration() {
    static op_kind_version_schema_map op_map;
    return op_map;
}

utils::rw_mutex_t &op_schema_registry_t::get_rw_mutex() {
    static utils::rw_mutex_t mutex;
    return mutex;
}

op_kind_version_schema_map &op_schema_registry_t::get_map() {
    op_kind_version_schema_map &op_map
            = get_map_without_ensuring_registration();
    class register_opset_t {
    public:
        register_opset_t() { register_opset_schema(); }
    };
    static register_opset_t ro;

    return op_map;
}

const op_schema_t *op_schema_registry_t::get_op_schema(op_kind_t kind) {
    auto &op_map = get_map();
    op_schema_t *schema = nullptr;
    get_rw_mutex().lock_read();
    if (op_map.count(kind)) { schema = &op_map[kind].rbegin()->second; }
    get_rw_mutex().unlock_read();
    return schema;
}

void register_schema(op_schema_t &&schema) {
    op_schema_registry_t::op_schema_registry_once_t DNNL_GRAPH_UNUSED
            registration(std::move(schema));
}

} // namespace impl
} // namespace graph
} // namespace dnnl
