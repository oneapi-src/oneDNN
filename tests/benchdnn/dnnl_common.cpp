/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

int measure_perf(benchdnn_timer_t &t, dnnl_primitive_t prim, args_t &args) {
    if (bench_mode & PERF) {
        std::vector<dnnl_exec_arg_t> dnnl_args;
        execute_unmap_args(args, dnnl_args);

        t.reset();
        while (true) {
            DNN_SAFE(dnnl_primitive_execute(prim, stream_tgt,
                             (int)dnnl_args.size(), dnnl_args.data()),
                    WARN);
            DNN_SAFE(dnnl_stream_wait(stream_tgt), WARN);
            t.stamp();
            const bool stop = false
                    || (fix_times_per_prb && t.times() >= fix_times_per_prb)
                    || (!fix_times_per_prb && t.total_ms() >= max_ms_per_prb
                            && t.times() >= min_times_per_prb);
            if (stop) {
                execute_map_args(args);
                break;
            }
        }
    }
    return OK;
}
