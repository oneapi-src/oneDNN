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

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include "config.hpp"
#ifdef _MSC_VER
#else
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#endif
#include <util/utils.hpp>

namespace sc {

struct trace_log_t {
    int tid_;
    int func_id_;
    int in_or_out_;
    int64_t tick_;
};

static struct trace_env_t {
    std::mutex name_lock_;
    std::vector<std::string> names_;
    std::vector<trace_log_t> traces_;
    // fields for multi-instance mode
    std::atomic<int64_t> num_traces_ = {-1};
    bool multi_instance_;

    trace_env_t() { multi_instance_ = true; }

    void gen_trace_file_and_clear() {
        if (traces_.empty()) { return; }
        std::string &tracep = sc::runtime_config_t::get().trace_out_path_;
        if (tracep == "0") {
            SC_WARN << "Got traces but SC_TRACE=0, exiting...";
            return;
        }
        FILE *outf;
        const char *filename;
        if (tracep == "1" || tracep.empty()) {
            outf = fopen("sctrace.json", "w");
            filename = "sctrace.json";
        } else if (tracep == "stderr") {
            outf = stderr;
            filename = "*stderr*";
        } else {
            outf = fopen(tracep.c_str(), "w");
            filename = tracep.c_str();
        }
        SC_WARN << "Generating traces to " << filename << " ...";
        fputs(R"({
"traceEvents": [
)",
                outf);
        size_t trace_size;
        if (multi_instance_) {
            if (num_traces_ >= 0) {
                trace_size = num_traces_;
            } else {
                trace_size = 0;
            }
        } else {
            trace_size = traces_.size();
        }
        for (size_t i = 0; i < trace_size; i++) {
            auto &v = traces_[i];
            fprintf(outf,
                    R"({"pid":1, "tid":%d, "ts":%lf, "ph":"%c", "name":"%s@%d", "args":{}, "cat":"call" }%c
)",
                    v.tid_, (v.tick_ - traces_.front().tick_) / 1000.0,
                    v.in_or_out_ ? 'E' : 'B', names_[v.func_id_].c_str(),
                    v.func_id_, i == trace_size - 1 ? ' ' : ',');
        }
        fputs(R"(],
"sc_version": "0.0.0"
}
)",
                outf);
        if (outf != stderr) { fclose(outf); }
        traces_.clear();
    }

    ~trace_env_t() { gen_trace_file_and_clear(); }
} env;

SC_INTERNAL_API void generate_trace_file() {
    env.gen_trace_file_and_clear();
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

} // namespace sc

using namespace sc;
extern "C" void sc_make_trace(int id, int in_or_out) {
    if (env.multi_instance_) {
        if (env.num_traces_ == -1) {
            std::lock_guard<std::mutex> guard(env.name_lock_);
            // double check locking
            if (env.num_traces_ == -1) {
                env.traces_.resize(runtime_config_t::get().trace_initial_cap_);
                env.num_traces_ = 0;
            }
        }
        auto idx = env.num_traces_++;
        COMPILE_ASSERT(static_cast<size_t>(idx) < env.traces_.size(),
                "Too many traces generated. Please try to enlarge "
                "SC_TRACE_INIT_CAP.");
        auto t = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::time_point_cast<std::chrono::nanoseconds>(t)
                           .time_since_epoch()
                           .count();
#if defined(_MSC_VER) || defined(__APPLE__)
        // fix-me: (win32)
        throw std::runtime_error(
                "syscall(__NR_gettid) is not support on windows.");
#else
        pid_t tid = syscall(__NR_gettid);
        static_assert(std::is_same<pid_t, int>::value, "Expecting pid_t=int");
        env.traces_[idx] = trace_log_t {tid, id, in_or_out, now};
        return;
#endif
    }
    // single instace mode
    // we use different part of code for single/multi instance mode because
    // muti-instance mode has larger overhead (atomic, system call for tid, etc)
    if (env.traces_.empty()) {
        env.traces_.reserve(runtime_config_t::get().trace_initial_cap_);
    }
    auto t = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::time_point_cast<std::chrono::nanoseconds>(t)
                       .time_since_epoch()
                       .count();
    env.traces_.emplace_back(trace_log_t {/*tid*/ 1, id, in_or_out, now});
}
