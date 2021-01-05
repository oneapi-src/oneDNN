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
#include "oneapi/dnnl/dnnl_graph_sycl.h"

#include "allocator.hpp"
#include "c_types_map.hpp"
#include "utils.hpp"

using namespace dnnl::graph::impl;

status_t DNNL_GRAPH_API dnnl_graph_allocator_create(
        allocator_t **created_allocator, cpu_allocate_f cpu_malloc,
        cpu_deallocate_f cpu_free) {
    if (utils::any_null(cpu_malloc, cpu_free)) {
        *created_allocator = new allocator_t {};
    } else {
        *created_allocator = new allocator_t {cpu_malloc, cpu_free};
    }
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_sycl_interop_allocator_create(
        dnnl_graph_allocator_t **created_allocator, sycl_allocate_f sycl_malloc,
        sycl_deallocate_f sycl_free) {
#if DNNL_GRAPH_WITH_SYCL
    if (utils::any_null(sycl_malloc, sycl_free)) {
        *created_allocator = new allocator_t {};
    } else {
        *created_allocator = new allocator_t {sycl_malloc, sycl_free};
    }
    return status::success;
#else
    UNUSED(created_allocator);
    UNUSED(sycl_malloc);
    UNUSED(sycl_free);
    return status::unsupported;
#endif
}

status_t DNNL_GRAPH_API dnnl_graph_allocator_destroy(allocator_t *allocator) {
    delete allocator;
    return status::success;
}
