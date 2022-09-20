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

#include "graph/interface/tensor.hpp"

#include "graph/utils/utils.hpp"

using namespace dnnl::impl::graph;

status_t DNNL_API dnnl_graph_tensor_create(tensor_t **tensor,
        const logical_tensor_t *logical_tensor, engine_t *eng, void *handle) {
    if (utils::any_null(tensor, logical_tensor, eng))
        return status::invalid_arguments;

    *tensor = new tensor_t {*logical_tensor, eng, handle};
    return status::success;
}

status_t DNNL_API dnnl_graph_tensor_destroy(tensor_t *tensor) {
    delete tensor;
    return status::success;
}

status_t DNNL_API dnnl_graph_tensor_get_data_handle(
        const tensor_t *tensor, void **handle) {
    if (utils::any_null(tensor, handle)) return status::invalid_arguments;

    *handle = tensor->get_data_handle();
    return status::success;
}

status_t DNNL_API dnnl_graph_tensor_set_data_handle(
        tensor_t *tensor, void *handle) {
    if (tensor == nullptr) return status::invalid_arguments;

    tensor->set_data_handle(handle);
    return status::success;
}

status_t DNNL_API dnnl_graph_tensor_get_engine(
        const tensor_t *tensor, engine_t **engine) {
    if (utils::any_null(tensor, engine)) return status::invalid_arguments;

    *engine = const_cast<engine_t *>(tensor->get_engine());

    return status::success;
}
