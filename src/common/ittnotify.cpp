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

#include "ittnotify.hpp"
#include "utils.hpp"

#if defined(DNNL_ENABLE_ITT_TASKS)
#include "common/ittnotify/ittnotify.h"
#include "dnnl_debug.h"
#endif

namespace dnnl {
namespace impl {
namespace itt {

static setting_t<int> itt_task_level {__itt_task_level_high};

bool get_itt(__itt_task_level level) {
    if (!itt_task_level.initialized()) {
        // Assumes that all threads see the same environment
        static int val
                = getenv_int_user("ITT_TASK_LEVEL", itt_task_level.get());
        itt_task_level.set(val);
    }
    return level <= itt_task_level.get();
}

#if defined(DNNL_ENABLE_ITT_TASKS)

namespace {

thread_local primitive_kind_t thread_primitive_kind;
thread_local const char *thread_primitive_name;

__itt_domain *itt_domain(primitive_kind_t kind) {
#define CASE(x) __itt_domain_create("dnnl.execute." STRINGIFY(x))
    static __itt_domain *prim_kind_itt_domain[] = {
            CASE(undefined),
            CASE(reorder),
            CASE(shuffle),
            CASE(concat),
            CASE(sum),
            CASE(convolution),
            CASE(deconvolution),
            CASE(eltwise),
            CASE(lrn),
            CASE(batch_normalization),
            CASE(inner_product),
            CASE(rnn),
            CASE(gemm),
            CASE(binary),
            CASE(matmul),
            CASE(resampling),
            CASE(pooling),
            CASE(reduction),
            CASE(prelu),
            CASE(softmax),
            CASE(layer_normalization),
    };
#undef CASE

    int kind_idx = (int)kind;
    assert(kind_idx >= 0);
    assert((size_t)kind_idx
            < sizeof(prim_kind_itt_domain) / sizeof(prim_kind_itt_domain[0]));

    return prim_kind_itt_domain[kind_idx];
}

} // namespace

void primitive_task_start(primitive_kind_t kind, const char *task_name) {
    if (kind == primitive_kind::undefined) return;

    __itt_string_handle *str_handle = __itt_string_handle_create(task_name);
    __itt_task_begin(itt_domain(kind), __itt_null, __itt_null, str_handle);
    thread_primitive_kind = kind;
    thread_primitive_name = task_name;
}

primitive_kind_t primitive_task_get_current_kind() {
    return thread_primitive_kind;
}

const char *primitive_task_get_current_name() {
    return thread_primitive_name;
}

void primitive_task_end() {
    if (thread_primitive_kind != primitive_kind::undefined) {
        __itt_task_end(itt_domain(thread_primitive_kind));
        thread_primitive_name = nullptr;
        thread_primitive_kind = primitive_kind::undefined;
    }
}
#else
void primitive_task_start(primitive_kind_t kind) {
    UNUSED(kind);
}
primitive_kind_t primitive_task_get_current_kind() {
    return primitive_kind::undefined;
}
void primitive_task_end() {}
#endif

} // namespace itt
} // namespace impl
} // namespace dnnl
