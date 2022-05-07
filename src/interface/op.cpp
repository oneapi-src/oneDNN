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

#include "oneapi/dnnl/dnnl_graph.h"

#include "c_types_map.hpp"
#include "op.hpp"
#include "op_schema.hpp"

using namespace dnnl::graph::impl;

/// constructor
dnnl_graph_op::dnnl_graph_op(
        size_t id, op_kind_t kind, std::string name, bool internal)
    : id_ {id}, kind_ {kind}, name_ {std::move(name)}, internal_ {internal} {
    if (name_.empty()) { name_ = kind2str(kind_) + "_" + std::to_string(id_); }
}

status_t DNNL_GRAPH_API dnnl_graph_op_create(op_t **op, uint64_t id,
        op_kind_t kind, const char *const verbose_name) {
    *op = new op_t {id, kind, verbose_name};
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_op_destroy(op_t *op) {
    delete op;
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_op_add_input(
        op_t *op, const logical_tensor_t *input) {
    op->add_input(*input);
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_op_add_output(
        op_t *op, const logical_tensor_t *output) {
    op->add_output(*output);
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_op_add_attr(op_t *op, const char *name,
        attribute_kind_t kind, const void *value, size_t value_len) {
    switch (kind) {
        case attribute_kind::i:
            op->set_attr(name, *static_cast<const int64_t *>(value));
            break;
        case attribute_kind::is: {
            const auto beg = static_cast<const int64_t *>(value);
            op->set_attr(name,
                    std::vector<int64_t> {beg, std::next(beg, value_len)});
        } break;
        case attribute_kind::f:
            op->set_attr(name, *static_cast<const float *>(value));
            break;
        case attribute_kind::fs: {
            const auto beg = static_cast<const float *>(value);
            op->set_attr(
                    name, std::vector<float> {beg, std::next(beg, value_len)});
        } break;
        case attribute_kind::s: {
            const auto beg = static_cast<const char *>(value);
            op->set_attr(name, std::string(beg));
        } break;
        case attribute_kind::b:
            op->set_attr(name, *static_cast<const bool *>(value));
            break;
        default: return status::unimplemented;
    }
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_op_get_id(const op_t *op, size_t *id) {
    *id = op->get_id();
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_op_get_kind(
        const op_t *op, op_kind_t *kind) {
    *kind = op->get_kind();
    return status::success;
}
