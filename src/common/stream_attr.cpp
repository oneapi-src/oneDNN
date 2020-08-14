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

#include "dnnl.h"

#include "stream_attr.hpp"
#include "utils.hpp"

using namespace dnnl::impl;

dnnl_status_t dnnl_stream_attr_create(
        dnnl_stream_attr_t *attr, dnnl_engine_kind_t kind) {
    if (utils::any_null(attr)) return status::invalid_arguments;
    dnnl_stream_attr *result = new dnnl_stream_attr(kind);
    if (result == nullptr) return status::out_of_memory;
    *attr = result;
    return status::success;
}

dnnl_status_t dnnl_stream_attr_destroy(dnnl_stream_attr_t attr) {
    delete attr;
    return status::success;
}

dnnl_status_t dnnl_stream_attr_set_threadpool(
        dnnl_stream_attr_t attr, void *threadpool) {
    if (utils::any_null(attr)) return status::invalid_arguments;
    return attr->set_threadpool(
            reinterpret_cast<dnnl::threadpool_iface *>(threadpool));
}

dnnl_status_t dnnl_stream_attr_get_threadpool(
        dnnl_stream_attr_t attr, void **threadpool) {
    if (utils::any_null(attr, threadpool)) return status::invalid_arguments;
    dnnl::threadpool_iface *tp;
    auto status = attr->get_threadpool(&tp);
    if (status == status::success) *threadpool = static_cast<void *>(tp);
    return status;
}
