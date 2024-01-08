/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef UTILS_TASK_EXECUTOR_HPP
#define UTILS_TASK_EXECUTOR_HPP

#include "rnn/rnn_task.hpp"
#include "utils/parallel.hpp"

extern int repeats_per_prb;

template <typename prb_t, typename perf_report_t, typename create_func_t,
        typename check_cache_func_t, typename do_func_t>
struct rnn_task_executor_t {
    virtual ~rnn_task_executor_t() { assert(tasks_.empty()); }

    void submit(std::shared_ptr<const prb_t> prb,
            const std::string &perf_template, const create_func_t &create_func,
            const check_cache_func_t &check_cache_func,
            const do_func_t &do_func) {
        static const int nthreads = benchdnn_get_max_threads();
        for (int r = 0; r < repeats_per_prb; r++) {
            tasks_.emplace_back(prb, perf_template, create_func,
                    check_cache_func, do_func, get_idx());
            if (has_bench_mode_modifier(mode_modifier_t::par_create)
                    && static_cast<int>(tasks_.size()) < nthreads)
                continue;
            flush();
        }
    }

    void flush() {
        // Special case is needed for THREADPOOL RUNTIME. Both `Parallel_nd` and
        // `createit` calls activate threadpool which causes undesired behavior.
        if (tasks_.size() == 1)
            tasks_[0].create();
        else
            benchdnn_parallel_nd(
                    tasks_.size(), [&](int i) { tasks_[i].create(); });

        // Check caches first to avoid filling cache with service reorders.
        for (auto &t : tasks_) {
            t.check_cache();
        }

        for (auto &t : tasks_) {
            t.exec();
        }

        tasks_.clear();
    }

    std::vector<rnn_task_t<prb_t, perf_report_t, create_func_t,
            check_cache_func_t, do_func_t>>
            tasks_;

    int get_idx() {
        static int idx = 0;
        return idx++;
    }
};

#endif
