/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_CONFIG_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_CONFIG_HPP
#include <stdint.h>
#include <string>
#include <runtime/generic_val.hpp>
#include <runtime/threadpool_mode.hpp>
#include <util/def.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

struct thread_pool_table {
    // submits job in thread pool
    void (*parallel_call)(void (*pfunc)(void *, void *, int64_t, generic_val *),
            uint64_t flags, void *rtl_ctx, void *module_env, int64_t begin,
            int64_t end, int64_t step, generic_val *args);
    // submits job in GC-managed thread pool
    void (*parallel_call_managed)(
            void (*pfunc)(void *, void *, int64_t, generic_val *),
            uint64_t flags, void *rtl_ctx, void *module_env, int64_t begin,
            int64_t end, int64_t step, generic_val *args);
    // gets the max number of threads in pool
    int (*get_num_threads)();
    // sets the max number of threads in pool
    void (*set_num_threads)(int v);
    // get the current thread id in pool. Should be 0~N
    int (*get_thread_id)();
    // returns non-zero if is in parallel section
    int (*is_in_parallel)();
};

struct SC_API runtime_config_t {
    enum trace_mode_t { OFF = 0, FAST, KERNEL, MULTI_THREAD };
    thread_pool_table *thread_pool_table_;
    // if in muti-instance simulation, the number of threads per instance.
    int get_num_threads() { return thread_pool_table_->get_num_threads(); }
    bool set_num_threads(int num) const;
    std::string trace_out_path_;
    int trace_initial_cap_ = 4096;
    trace_mode_t trace_mode_ = OFF;
    bool execution_verbose_ = false;
    thread_pool_mode_t managed_thread_pool_ = thread_pool_mode_t::DIRECT;
    int verbose_level_ = 0;
    static runtime_config_t &get() noexcept;

private:
    runtime_config_t();
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
