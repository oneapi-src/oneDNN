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

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "config.hpp"
#include <runtime/logging.hpp>
#include <util/assert.hpp>
#ifdef _MSC_VER
#else
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#endif
#include <runtime/env_vars.hpp>
#include <runtime/microkernel/cpu/kernel_timer.hpp>
#include <runtime/runtime.hpp>
#include <runtime/thread_locals.hpp>
#include <util/string_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_MODULE(runtime.trace);

static struct trace_env_t {
    std::mutex name_lock_;
    std::vector<std::string> names_ {
            "brgemm", "list_brgemm", "barrier", "barrier_internal", "prefetch"};
} env;

namespace runtime {

static void write_json_traces(FILE *outf,
        const std::list<thread_local_buffer_t *> &tls_buffers, int64_t min_val,
        size_t trace_size, bool main_thread_found) {
    fputs(R"({
"traceEvents": [
)",
            outf);
    size_t i = 0;
    for (auto *tlb : tls_buffers) {
        if (runtime_config_t::get().trace_mode_
                < runtime_config_t::trace_mode_t::MULTI_THREAD) {
            if (!tlb->additional_->is_main_thread_ && main_thread_found) {
                tlb->additional_->trace_.trace_logs_.clear();
                continue;
            }
        }
        for (auto &v : tlb->additional_->trace_.trace_logs_) {
            fprintf(outf,
                    R"({"pid":1, "tid":%zu, "ts":%lf, "ph":"%c", "name":"%s@%d", "args":{"flop":%d}, "cat":"call" }%c
)",
                    (size_t)tlb->additional_->linear_thread_id_,
                    (v.tick_ - min_val) / 1000.0, v.in_or_out_ ? 'E' : 'B',
                    env.names_[v.func_id_].c_str(), v.func_id_, v.arg_,
                    i == trace_size - 1 ? ' ' : ',');
            i++;
        }
        tlb->additional_->trace_.trace_logs_.clear();
    }
    fputs(R"(],
"sc_version": "0.0.0"
}
)",
            outf);
}

static void write_compact_traces(FILE *outf,
        const std::list<thread_local_buffer_t *> &tls_buffers, int64_t min_val,
        size_t trace_size, bool main_thread_found) {
    fprintf(outf, "symbols:");
    for (size_t i = 0; i < env.names_.size(); i++) {
        fprintf(outf, "%zu-%s,", i, env.names_[i].c_str());
    }
    fprintf(outf, "\n");
    for (auto *tlb : tls_buffers) {
        fprintf(outf, "trace:%d,%d:", tlb->additional_->linear_thread_id_,
                tlb->additional_->instance_id_);
        for (auto &v : tlb->additional_->trace_.trace_logs_) {
            fprintf(outf, "%ld-%d-%d-%d,", (long)(v.tick_ - min_val), // NOLINT
                    v.in_or_out_, (int)v.func_id_, (int)v.arg_); // NOLINT
        }
        fprintf(outf, "\n");
        tlb->additional_->trace_.trace_logs_.clear();
    }
}

void write_traces(const std::list<thread_local_buffer_t *> &tls_buffers) {
    std::string &tracep = runtime_config_t::get().trace_out_path_;
    size_t trace_cap = runtime_config_t::get().trace_initial_cap_;
    if (tracep.empty()) { return; }
    size_t trace_size = 0;
    int64_t min_val = std::numeric_limits<uint64_t>::max();
    bool main_thread_found = false;
    bool already_warned = false;
    for (auto v : tls_buffers) {
        auto cur_trace_size = v->additional_->trace_.trace_logs_.size();
        trace_size += cur_trace_size;
        if (!already_warned && cur_trace_size > trace_cap) {
            already_warned = true;
            SC_MODULE_WARN << "Received too many traces: " << cur_trace_size
                           << ". The initial capacity is " << trace_cap
                           << ". This will cause inaccurate result. Please "
                              "consider enlarge "
                           << env_names[env_key::SC_TRACE_INIT_CAP];
        }
        if (v->additional_->is_main_thread_) { main_thread_found = true; }
        for (auto &log : v->additional_->trace_.trace_logs_) {
            min_val = std::min(log.tick_, min_val);
        }
    }
    if (trace_size == 0UL) { return; }
    FILE *outf;
    const char *filename;
    bool compact = false;
    if (tracep == "stderr") {
        outf = stderr;
        filename = "*stderr*";
    } else {
        outf = fopen(tracep.c_str(), "w");
        filename = tracep.c_str();
        compact = !utils::string_endswith(tracep, ".json");
    }
    SC_WARN << "Generating traces to " << filename << " ...";
    if (compact) {
        write_compact_traces(
                outf, tls_buffers, min_val, trace_size, main_thread_found);
    } else {
        write_json_traces(
                outf, tls_buffers, min_val, trace_size, main_thread_found);
    }
    if (outf != stderr) { fclose(outf); }
}
} // namespace runtime

SC_INTERNAL_API void generate_trace_file() {
    dnnl::impl::graph::gc::release_runtime_memory(nullptr);
}

int register_traced_func(const std::string &name) {
    std::lock_guard<std::mutex> guard(env.name_lock_);
    env.names_.emplace_back(name);
    return env.names_.size() - 1;
}

int get_last_trace_func_id() {
    std::lock_guard<std::mutex> guard(env.name_lock_);
    return env.names_.size() - 1;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

using namespace dnnl::impl::graph::gc;
extern "C" void sc_make_trace(int id, int in_or_out, int arg) {
    auto &trace_mgr
            = runtime::thread_local_buffer_t::tls_buffer().additional_->trace_;
    if (trace_mgr.trace_logs_.empty()) {
        trace_mgr.trace_logs_.reserve(
                runtime_config_t::get().trace_initial_cap_);
    }
    auto t = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::time_point_cast<std::chrono::nanoseconds>(t)
                       .time_since_epoch()
                       .count();
    trace_mgr.trace_logs_.emplace_back(runtime::trace_manager_t::trace_log_t {
            static_cast<uint16_t>(id), static_cast<char>(in_or_out), arg, now});
}

extern "C" void sc_make_trace_kernel(int id, int in_or_out, int arg) {
#ifdef SC_KERNEL_PROFILE
    if (sc_is_trace_enabled()) { sc_make_trace(id, in_or_out, arg); }
#endif
}
