/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include "oneapi/dnnl/dnnl_graph.h"

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/op.hpp"
#include "graph/interface/op_schema.hpp"

using namespace dnnl::impl::graph;

namespace {
// Check if an attribute should have single float value.
bool is_f(const op_attr_t &attr) {
    return attr >= op_attr::alpha && attr < op_attr::scales;
}

// Check if an attribute should have single integer value.
bool is_i(const op_attr_t &attr) {
    return attr >= op_attr::axis && attr < op_attr::axes;
}
} // namespace

/// constructor
dnnl_graph_op::dnnl_graph_op(
        size_t id, op_kind_t kind, std::string name, bool internal)
    : id_ {id}, kind_ {kind}, name_ {std::move(name)}, internal_ {internal} {
    if (name_.empty()) { name_ = kind2str(kind_) + "_" + std::to_string(id_); }
}

status_t DNNL_API dnnl_graph_op_create(op_t **op, size_t id,
        dnnl_graph_op_kind_t kind, const char *verbose_name) {
    if (utils::any_null(op, verbose_name)) return status::invalid_arguments;

    *op = new op_t {id, kind, verbose_name};
    return status::success;
}

status_t DNNL_API dnnl_graph_op_destroy(op_t *op) {
    delete op;
    return status::success;
}

status_t DNNL_API dnnl_graph_op_add_input(
        op_t *op, const logical_tensor_t *input) {
    if (utils::any_null(op, input)) return status::invalid_arguments;

    op->add_input(*input);
    return status::success;
}

status_t DNNL_API dnnl_graph_op_add_output(
        op_t *op, const logical_tensor_t *output) {
    if (utils::any_null(op, output)) return status::invalid_arguments;

    op->add_output(*output);
    return status::success;
}

status_t DNNL_API dnnl_graph_op_set_attr_f32(op_t *op,
        dnnl_graph_op_attr_t name, const float *value, size_t value_len) {
    if (utils::any_null(op, value)) return status::invalid_arguments;

    if (is_f(name)) {
        op->set_attr<float>(name, *value);
    } else {
        std::vector<float> val(value, value + value_len);
        op->set_attr<std::vector<float>>(name, val);
    }

    return status::success;
}

status_t DNNL_API dnnl_graph_op_set_attr_bool(op_t *op,
        dnnl_graph_op_attr_t name, const uint8_t *value, size_t value_len) {
    if (utils::any_null(op, value)) return status::invalid_arguments;
    // we don't support bool vector.
    if (value_len != 1) return status::invalid_arguments;

    const bool val = static_cast<bool>(*value);
    op->set_attr<bool>(name, val);

    return status::success;
}

status_t DNNL_API dnnl_graph_op_set_attr_s64(op_t *op,
        dnnl_graph_op_attr_t name, const int64_t *value, size_t value_len) {
    if (utils::any_null(op, value)) return status::invalid_arguments;

    if (is_i(name)) {
        op->set_attr<int64_t>(name, *value);
    } else {
        std::vector<int64_t> val(value, value + value_len);
        op->set_attr<std::vector<int64_t>>(name, val);
    }

    return status::success;
}

status_t DNNL_API dnnl_graph_op_set_attr_str(op_t *op,
        dnnl_graph_op_attr_t name, const char *value, size_t value_len) {
    if (utils::any_null(op, value)) return status::invalid_arguments;
    if (value_len == 0) return status::invalid_arguments;

    op->set_attr<std::string>(name, std::string(value));

    return status::success;
}

status_t DNNL_API dnnl_graph_op_get_id(const op_t *op, size_t *id) {
    if (utils::any_null(op, id)) return status::invalid_arguments;

    *id = op->get_id();
    return status::success;
}

status_t DNNL_API dnnl_graph_op_get_kind(
        const op_t *op, dnnl_graph_op_kind_t *kind) {
    if (utils::any_null(op, kind)) return status::invalid_arguments;

    *kind = static_cast<dnnl_graph_op_kind_t>(op->get_kind());
    return status::success;
}
