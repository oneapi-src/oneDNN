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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_MICROKERNEL_CPU_KERNEL_TIMER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_MICROKERNEL_CPU_KERNEL_TIMER_HPP
#ifdef SC_KERNEL_PROFILE
#include <runtime/config.hpp>
#include <runtime/thread_locals.hpp>

inline bool sc_is_trace_enabled() {
    namespace gc = dnnl::impl::graph::gc;
    auto mode = gc::runtime_config_t::get().trace_mode_;
    return (mode == gc::runtime_config_t::trace_mode_t::KERNEL
                   && gc::runtime::thread_local_buffer_t::tls_buffer()
                                   .additional_->linear_thread_id_
                           == 0)
            || mode == gc::runtime_config_t::trace_mode_t::MULTI_THREAD;
}

inline void sc_make_timer_id(int flops, int num) {
    namespace gc = dnnl::impl::graph::gc;
    if (sc_is_trace_enabled()) {
        auto &log = gc::runtime::thread_local_buffer_t::tls_buffer()
                            .additional_->trace_.trace_logs_.back();
        log.arg_ = flops;
    }
}

#define sc_make_timer(desc, num) sc_make_timer_id(desc->flops_, num);
#else
#define sc_make_timer(id, num)
#define sc_make_timer_id(id, num)
#endif
#endif
