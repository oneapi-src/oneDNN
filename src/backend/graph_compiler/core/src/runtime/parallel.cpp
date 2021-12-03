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

#include "common/generic_val.hpp"
#include "config.hpp"
#include "context.hpp"
#include <runtime/parallel.hpp>
#include <util/utils.hpp>

// todo: handle signed integers
extern "C" void sc_parallel_call_cpu(void (*pfunc)(uint64_t, sc::generic_val *),
        uint64_t begin, uint64_t end, uint64_t step, sc::generic_val *args) {
    int run_threads = sc::runtime_config_t::get().threads_per_instance_;
#ifdef SC_OMP_ENABLED
#pragma omp parallel for num_threads(run_threads)
#endif
    for (uint64_t i = begin; i < end; i += step) {
        pfunc(i, args);
    }
}

extern "C" void sc_parallel_call_cpu_with_env_impl(
        void (*pfunc)(void *, void *, uint64_t, sc::generic_val *),
        void *rtl_ctx, void *module_env, uint64_t begin, uint64_t end,
        uint64_t step, sc::generic_val *args) {
    int run_threads = sc::runtime_config_t::get().threads_per_instance_;
#ifdef SC_OMP_ENABLED
#pragma omp parallel for num_threads(run_threads)
#endif
    for (uint64_t i = begin; i < end; i += step) {
        pfunc(rtl_ctx, module_env, i, args);
    }
}

extern "C" void sc_parallel_call_cpu_with_env(
        void (*pfunc)(void *, void *, uint64_t, sc::generic_val *),
        void *rtl_ctx, void *module_env, uint64_t begin, uint64_t end,
        uint64_t step, sc::generic_val *args) {
    sc::runtime::stream_t *stream
            = reinterpret_cast<sc::runtime::stream_t *>(rtl_ctx);
    stream->vtable()->parallel_call(
            pfunc, rtl_ctx, module_env, begin, end, step, args);
}
