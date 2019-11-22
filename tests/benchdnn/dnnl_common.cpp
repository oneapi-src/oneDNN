/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#include <assert.h>
#include "dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

// Engine kind used to run DNNL primitives for testing
dnnl_engine_kind_t engine_tgt_kind = dnnl_cpu;

// Engine used to run DNNL primitives for testing
dnnl_engine_t engine_tgt;

// Stream for target engine
dnnl_stream_t stream_tgt;

// Engine used to run CPU implementations (use-fast-gpu option)
dnnl_engine_t engine_cpu;

// Stream for CPU engine
dnnl_stream_t stream_cpu;

args_t &args_t::set(int arg, const dnn_mem_t &mem) {
    args_.push_back(std::make_pair(arg, &mem));
    return *this;
}

// Unmap before passing the memory to execute
void execute_unmap_args(
        const args_t &args, std::vector<dnnl_exec_arg_t> &dnnl_args) {
    dnnl_args.resize(args.size());
    for (int i = 0; i < args.size(); ++i) {
        if (args.dnn_mem(i).is_mapped()) args.dnn_mem(i).unmap();

        dnnl_args[i].arg = args.arg(i);
        dnnl_args[i].memory = args.dnn_mem(i).m_;
    }
}

// Map the memory back after execute
void execute_map_args(const args_t &args) {
    for (int i = 0; i < args.size(); ++i)
        if (!args.dnn_mem(i).is_mapped()) args.dnn_mem(i).map();
}

dnnl_status_t execute_and_wait(
        dnnl_primitive_t prim, dnnl_stream_t stream, const args_t &args) {

    std::vector<dnnl_exec_arg_t> dnnl_args;
    execute_unmap_args(args, dnnl_args);

    dnnl_status_t status = dnnl_primitive_execute(
            prim, stream, (int)dnnl_args.size(), dnnl_args.data());
    if (status != dnnl_success) return status;
    status = dnnl_stream_wait(stream);
    if (status != dnnl_success) return status;

    execute_map_args(args);
    return dnnl_success;
}

inline bool should_stop(const benchdnn_timer_t &t) {
    const bool stop = false
            || (fix_times_per_prb && t.times() >= fix_times_per_prb)
            || (!fix_times_per_prb && t.total_ms() >= max_ms_per_prb
                    && t.times() >= min_times_per_prb);
    return stop;
}

inline int measure_perf_individual(benchdnn_timer_t &t, dnnl_primitive_t prim,
        std::vector<dnnl_exec_arg_t> &dnnl_args) {
    t.reset();
    while (true) {
        DNN_SAFE(dnnl_primitive_execute(prim, stream_tgt, (int)dnnl_args.size(),
                         dnnl_args.data()),
                WARN);
        t.stamp();
        if (should_stop(t)) break;
    }
    return OK;
}

inline int measure_perf_aggregate(benchdnn_timer_t &t, dnnl_primitive_t prim,
        std::vector<dnnl_exec_arg_t> &dnnl_args) {
    const int max_batch_times = 10000;

    // Warm-up run
    t.reset();
    DNN_SAFE(dnnl_primitive_execute(
                     prim, stream_tgt, (int)dnnl_args.size(), dnnl_args.data()),
            WARN);
    DNN_SAFE(dnnl_stream_wait(stream_tgt), WARN);
    t.stamp();

    int cur_batch_times
            = fix_times_per_prb ? fix_times_per_prb : min_times_per_prb;
    --cur_batch_times;

    while (true) {
        for (int i = 0; i < cur_batch_times; i++) {
            DNN_SAFE(dnnl_primitive_execute(prim, stream_tgt,
                             (int)dnnl_args.size(), dnnl_args.data()),
                    WARN);
        }
        DNN_SAFE(dnnl_stream_wait(stream_tgt), WARN);
        t.stamp(cur_batch_times);

        if (should_stop(t)) break;

        // Adjust cur_batch_times after the first batch run
        if (t.times() == cur_batch_times + 1) {
            double ms_min = t.ms(benchdnn_timer_t::min);
            // Heuristic: try to use ~5 batch runs for the whole benchmark
            int batch_times_heuristic = (ms_min == 0.0)
                    ? INT_MAX
                    : MAX2(1,
                            (int)((max_ms_per_prb - t.total_ms()) / ms_min
                                    / 5));
            cur_batch_times = MIN2(max_batch_times, batch_times_heuristic);
        }
    }
    return OK;
}

int measure_perf(benchdnn_timer_t &t, dnnl_primitive_t prim, args_t &args) {
    int ret = OK;
    if (bench_mode & PERF) {
        std::vector<dnnl_exec_arg_t> dnnl_args;
        execute_unmap_args(args, dnnl_args);

        // For CPU: measure indiividual iterations
        // For GPU: measure iterations in batches to hide driver overhead
        if (engine_tgt_kind == dnnl_cpu)
            ret = measure_perf_individual(t, prim, dnnl_args);
        else
            ret = measure_perf_aggregate(t, prim, dnnl_args);

        if (ret == OK) execute_map_args(args);
    }
    return ret;
}

void maybe_prepare_runtime_scales(dnn_mem_t &scales_m, const attr_t &attr,
        int64_t scale_cnt, const float *scales, dnnl_engine_t engine) {
    if (!attr.oscale.runtime) return;

    using P = attr_t::scale_t::policy_t;
    const int64_t count = attr.oscale.policy == P::COMMON ? 1 : scale_cnt;

    scales_m = dnn_mem_t(1, &count, dnnl_f32, dnnl_a, engine);
    for (int64_t c = 0; c < count; ++c)
        ((float *)scales_m)[c] = scales[c];
}

void maybe_prepare_runtime_scales(dnn_mem_t &scales_m,
        const attr_bundle_t &attr_bundle, dnnl_engine_t engine) {
    maybe_prepare_runtime_scales(scales_m, attr_bundle.attr,
            (int64_t)attr_bundle.oscale.size(), attr_bundle.oscale.data(),
            engine);
}

void maybe_prepare_runtime_zero_points(dnn_mem_t &zero_points_m,
        const attr_t &attr, int arg, dnnl_engine_t engine) {
    if (!attr.zero_points.runtime(arg)) return;

    int64_t count = 1;
    zero_points_m = dnn_mem_t(1, &count, dnnl_s32, dnnl_a, engine);
    ((int *)zero_points_m)[0] = attr.zero_points[arg];
}
