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

#include <cmath>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include "config.hpp"
#include <runtime/data_type.hpp>
#include <runtime/env_var.hpp>
#include <runtime/env_vars.hpp>
#include <runtime/logging.hpp>
#include <runtime/managed_thread_pool_exports.hpp>
#include <runtime/os.hpp>
#include <runtime/parallel.hpp>
#include <runtime/runtime.hpp>
#include <util/def.hpp>
#include <util/os.hpp>
#include <util/string_utils.hpp>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

SC_MODULE(runtime.support)

using namespace dnnl::impl::graph::gc;
extern "C" void print_float(float f) {
    printf("%f\n", f);
}

extern "C" void print_index(uint64_t f) {
    printf("%llu\n", static_cast<unsigned long long>(f)); // NOLINT
}

extern "C" void print_int(int f) {
    printf("%d\n", f);
}

extern "C" void print_str(char *f) {
    fputs(f, stdout);
}

extern "C" void *sc_global_aligned_alloc(size_t sz, size_t align) {
    return aligned_alloc(align, (sz / align + 1) * align);
}

extern "C" void sc_global_aligned_free(void *ptr, size_t align) {
    aligned_free(ptr);
}

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace runtime {
size_t get_os_page_size() {
#ifdef _WIN32
    // fix-me: (win32) impl
    return 4096;
#else
    static size_t v = getpagesize();
    return v;
#endif
}
} // namespace runtime

runtime_config_t &runtime_config_t::get() noexcept {
    static runtime_config_t cfg {};
    return cfg;
}

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
extern int get_max_threadpool_concurrency();
#define SET_NUM_THREADS_SUCCESS(NUM) ((NUM) <= get_max_threadpool_concurrency())
#else
// We are free to set num threads in TBB and OMP
#define SET_NUM_THREADS_SUCCESS(NUM) true
#endif

bool runtime_config_t::set_num_threads(int num) const {
    thread_pool_table_->set_num_threads(num);
    return SET_NUM_THREADS_SUCCESS(num);
}

using namespace env_key;
runtime_config_t::runtime_config_t() {
    thread_pool_table_ = &sc_pool_table;
    constexpr int default_MTP =
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
            1;
#else
            0;
#endif
    managed_thread_pool_
            = (utils::getenv_int(env_names[SC_MANAGED_THREAD_POOL], default_MTP)
                    != 0);

    if (managed_thread_pool_) {
        thread_pool_table_->parallel_call_managed = &sc_parallel_call_managed;
    }
    trace_initial_cap_ = utils::getenv_int(env_names[SC_TRACE_INIT_CAP], 4096);
    trace_out_path_ = utils::getenv_string(env_names[SC_TRACE]);
    char mode = 0;

    if (trace_out_path_.size() == 1) {
        // for SC_TRACE=1
        mode = trace_out_path_[0] - '0';
        trace_out_path_ = "sctrace.json";
    } else if (trace_out_path_.size() > 2 && trace_out_path_[1] == ',') {
        // for SC_TRACE=2,abc.json
        mode = trace_out_path_[0] - '0';
        trace_out_path_ = trace_out_path_.substr(2);
    } else if (trace_out_path_.empty()) {
        mode = 0;
    } else {
        // for SC_TRACE=abc.json
        mode = 1;
    }

    switch (mode) {
        case 0: trace_mode_ = trace_mode_t::OFF; break;
        case 1: trace_mode_ = trace_mode_t::FAST; break;
        case 2: trace_mode_ = trace_mode_t::KERNEL; break;
        case 3: trace_mode_ = trace_mode_t::MULTI_THREAD; break;
        default:
            trace_mode_ = trace_mode_t::OFF;
            trace_out_path_ = "";
            break;
    }

    constexpr int default_verbose = 0;
    int tmp_get_verbose_level
            = utils::getenv_int(env_names[SC_VERBOSE], default_verbose);
    if (tmp_get_verbose_level < 0 || tmp_get_verbose_level > 2) {
        tmp_get_verbose_level = 0;
    }
    verbose_level_ = tmp_get_verbose_level;
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
