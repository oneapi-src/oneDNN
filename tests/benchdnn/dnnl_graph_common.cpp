/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "dnnl_graph_common.hpp"

namespace graph = dnnl::graph;

namespace benchgraph {

inline bool should_stop(const benchdnn_timer_t &t) {
    const bool stop = false
            || (fix_times_per_prb && t.times() >= fix_times_per_prb)
            || (!fix_times_per_prb && t.total_ms() >= max_ms_per_prb
                    && t.times() >= min_times_per_prb);
    return stop;
}

void compiled_partition_executor(graph::compiled_partition &cp,
        graph::stream &stream, const std::vector<graph::tensor> &inputs,
        const std::vector<graph::tensor> &outputs) {
    cp.execute(stream, inputs, outputs);
}

inline int measure_perf_individual(benchdnn_timer_t &t, graph::stream &stream,
        perf_function_t &perf_func, const std::vector<graph::tensor> &inputs,
        const std::vector<graph::tensor> &outputs) {
    t.reset();
    while (true) {
        BENCHGRAPH_SAFE(perf_func(stream, inputs, outputs), WARN);
        t.stamp();
        if (should_stop(t)) break;
    }
    return OK;
}

int measure_perf(benchdnn_timer_t &t, perf_function_t &perf_func,
        const std::vector<graph::tensor> &inputs,
        const std::vector<graph::tensor> &outputs) {
    int ret = OK;
    if (bench_mode & PERF) {
        graph::stream stream(::benchgraph::get_test_engine());
        ret = measure_perf_individual(t, stream, perf_func, inputs, outputs);
    }
    return ret;
}

int measure_perf(benchdnn_timer_t &t, graph::compiled_partition &cp,
        const std::vector<graph::tensor> &inputs,
        const std::vector<graph::tensor> &outputs) {
    perf_function_t perf_func
            = std::bind(&compiled_partition_executor, cp, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3);

    return measure_perf(t, perf_func, inputs, outputs);
}

} // namespace benchgraph
