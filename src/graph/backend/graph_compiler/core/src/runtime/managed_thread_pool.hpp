/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_MANAGED_THREAD_POOL_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_MANAGED_THREAD_POOL_HPP
#include <atomic>
#include <runtime/context.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace runtime {
struct thread_manager {
    using idle_func_t = uint64_t (*)(std::atomic<int32_t> *remaining,
            int32_t expected_remain, int32_t tid, void *args);
    struct thread_pool_state {
        struct task_type {
            void (*pfunc)(void *, void *, int64_t, generic_val *);
            void *stream;
            void *module_env;
            int64_t begin;
            int64_t end;
            int64_t step;
            generic_val *args;
        } task;
        int num_threads;

        std::atomic<int> trigger;
        idle_func_t idle_func = nullptr;
        void *idle_args = nullptr;
        uint64_t execution_flags = 0;

        alignas(64) std::atomic<int> remaining;

        void wait_all();
        void reset_scoreboard();
    } state;
#ifdef SC_KERNEL_PROFILE
    int instance_id_;
#endif
    thread_manager();
    using main_func_t = void (*)(runtime::stream_t *, void *, generic_val *);
    void run_main_function(main_func_t f, runtime::stream_t *stream,
            void *mod_data, generic_val *args);
    static thread_local thread_manager cur_mgr;
};
} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
