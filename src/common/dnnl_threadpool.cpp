/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL

#include "oneapi/dnnl/dnnl_threadpool.h"

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "utils.hpp"

dnnl_status_t dnnl_threadpool_interop_set_max_concurrency(int max_concurrency) {
    using namespace dnnl::impl;
    threadpool_utils::get_threadlocal_max_concurrency() = max_concurrency;
    return status::success;
}

dnnl_status_t dnnl_threadpool_interop_get_max_concurrency(
        int *max_concurrency) {
    using namespace dnnl::impl;
    if (max_concurrency == nullptr) return status::invalid_arguments;

    *max_concurrency = threadpool_utils::get_threadlocal_max_concurrency();
    return status::success;
}

#endif
