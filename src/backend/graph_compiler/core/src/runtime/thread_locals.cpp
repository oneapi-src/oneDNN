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

#include "thread_locals.hpp"
#include "config.hpp"
#include "parallel.hpp"

namespace sc {

SC_API void release_runtime_memory(runtime::engine *engine) {
    sc_parallel_call_cpu_with_env_impl(
            [](void *v1, void *v2, int64_t i, generic_val *args) {
                runtime::tls_buffer.main_memory_pool.release();
                runtime::tls_buffer.thread_memory_pool.release();
                runtime::tls_buffer.amx_buffer.release();
            },
            nullptr, nullptr, 0, runtime_config_t::get().threads_per_instance_,
            1, nullptr);
}

namespace runtime {
thread_local thread_local_buffer tls_buffer;

} // namespace runtime
} // namespace sc
