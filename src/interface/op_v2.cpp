/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include "op_v2.hpp"

#include "op_schema.hpp"

using namespace dnnl::graph::impl;

/// constructor
dnnl_graph_op_v2::dnnl_graph_op_v2(
        size_t id, op_kind_t kind, std::string name, bool internal)
    : id_ {id}
    , kind_ {kind}
    , name_ {std::move(name)}
    , schema_ {op_schema_registry::get_op_schema(kind)}
    , internal_ {internal} {
    if (name_.empty()) { name_ = kind2str(kind_) + "_" + std::to_string(id_); }
}

bool dnnl_graph_op_v2::verify() const {
    // always return true if there is no corresponding op schema
    // return nullptr == schema_ || schema_->verify(this);
    return nullptr == schema_;
}
