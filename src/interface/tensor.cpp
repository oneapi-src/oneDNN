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

#include "oneapi/dnnl/dnnl_graph.h"

#include "c_types_map.hpp"
#include "logical_tensor.hpp"
#include "tensor.hpp"

using namespace dnnl::graph::impl;

status_t DNNL_GRAPH_API dnnl_graph_tensor_create_with_logical_tensor(
        tensor_t **created_tensor, const logical_tensor_t *logical_tensor,
        void *data_handle) {
    *created_tensor = new tensor_t {*logical_tensor, data_handle};
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_tensor_destroy(tensor_t *tensor) {
    delete tensor;
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_tensor_get_if_type(
        const tensor_t *tensor, data_type_t type, void **data_handle) {
    *data_handle = tensor->get_void_data_handle_if_is(type);
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_tensor_set_data_handle(
        tensor_t *tensor, void *data_handle) {
    tensor->set_data_handle(data_handle);
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_tensor_get_element_num(
        const tensor_t *tensor, int64_t *num) {
    auto lt = tensor->get_logical_tensor();
    *num = logical_tensor_wrapper(lt).nelems();
    return status::success;
}
