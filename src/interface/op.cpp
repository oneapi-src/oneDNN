/*******************************************************************************
* Copyright 2020 Intel Corporation
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

using namespace dnnl::graph::impl;

status_t DNNL_GRAPH_API dnnl_graph_op_create(op_t **created_op, uint64_t id,
        op_kind_t kind, const char *const debug_string) {
    *created_op = new op_t {id, kind, debug_string};
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_op_destroy(op_t *op) {
    delete op;
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_op_add_input(
        op_t *op, const logical_tensor_t *input) {
    op->add_input(input);
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_op_add_output(
        op_t *op, const logical_tensor_t *output) {
    op->add_output(output);
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_op_get_attr_kind(
        const op_t *op, const char *name, attribute_kind_t *kind) {
    return op->kind_of(name, *kind);
}

status_t DNNL_GRAPH_API dnnl_graph_op_add_attr(op_t *op, const char *name,
        attribute_kind_t kind, const void *attr, int64_t attr_no) {
    switch (kind) {
        case attribute_kind::i:
            op->set_attr(name, *static_cast<const int64_t *>(attr));
            break;
        case attribute_kind::is: {
            const auto beg = static_cast<const int64_t *>(attr);
            op->set_attr(
                    name, std::vector<int64_t> {beg, std::next(beg, attr_no)});
        } break;
        case attribute_kind::f:
            op->set_attr(name, *static_cast<const float *>(attr));
            break;
        case attribute_kind::fs: {
            const auto beg = static_cast<const float *>(attr);
            op->set_attr(
                    name, std::vector<float> {beg, std::next(beg, attr_no)});
        } break;
        case attribute_kind::s: {
            const auto beg = static_cast<const char *>(attr);
            op->set_attr(name, std::string(beg));
        } break;
        case attribute_kind::b:
            op->set_attr(name, *static_cast<const bool *>(attr));
            break;
        default: return status::unsupported;
    }
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_op_get_attr(const op_t *op, const char *name,
        attribute_kind_t kind, const void **attr, int64_t *attr_no) {
    status_t status = status::success;
    switch (kind) {
        case attribute_kind::i: {
            const int64_t *attr_addr {nullptr};
            status = op->attr(name, &attr_addr);
            *attr = attr_addr;
        } break;
        case attribute_kind::is: {
            const std::vector<int64_t> *attr_addr {nullptr};
            status = op->attr(name, &attr_addr);
            assert(attr_addr != nullptr);
            *attr = attr_addr->data();
            *attr_no = static_cast<int64_t>(attr_addr->size());
        } break;
        case attribute_kind::f: {
            const float *attr_addr = static_cast<const float *>(*attr);
            status = op->attr(name, &attr_addr);
            *attr = attr_addr;
        } break;
        case attribute_kind::fs: {
            const std::vector<float> *attr_addr {nullptr};
            status = op->attr(name, &attr_addr);
            *attr = attr_addr->data();
            *attr_no = static_cast<int64_t>(attr_addr->size());
        } break;
        case attribute_kind::s: {
            const std::string *attr_addr {nullptr};
            status = op->attr(name, &attr_addr);
            *attr = attr_addr->c_str();
        } break;
        case attribute_kind::b: {
            const bool *attr_addr = static_cast<const bool *>(*attr);
            status = op->attr(name, &attr_addr);
            *attr = attr_addr;
        } break;
        default: return status::unsupported;
    }
    return status;
}

status_t DNNL_GRAPH_API dnnl_graph_op_get_id(const op_t *op, size_t *id) {
    *id = op->id();
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_op_get_kind(
        const op_t *op, dnnl_graph_op_kind_t *kind) {
    *kind = op->kind();
    return status::success;
}
