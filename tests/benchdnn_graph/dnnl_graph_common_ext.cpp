/*******************************************************************************
* Copyright 2022 Intel Corporation
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
#include "dnnl_graph_common_ext.hpp"

namespace benchdnnext {

inline int measure_perf_individual(timer::timer_t &t,
        dnnl::graph::stream &stream, std::vector<perf_function_t> &perf_func_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v) {
    t.reset();
    while (true) {
        auto sz = perf_func_v.size();
        for (int i = 0; i < sz; i++) {
            BENCHDNNEXT_SAFE(
                    perf_func_v[i](stream, inputs_v[i], outputs_v[i]), WARN);
        }
        t.stamp();
        if (should_stop(t)) break;
    }
    return OK;
}

int measure_perf(timer::timer_t &t, std::vector<perf_function_t> &perf_func_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v) {
    if (is_bench_mode(PERF)) {
        //TODO: update stream to use threadpool based when supported.
        dnnl::graph::stream stream(::benchdnnext::get_test_engine());
        return measure_perf_individual(
                t, stream, perf_func_v, inputs_v, outputs_v);
    } else {
        return OK;
    }
}

int measure_perf(timer::timer_t &t,
        std::vector<dnnl::graph::compiled_partition> &cp_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v,
        res_t *res) {
    std::vector<perf_function_t> perf_func_v;
    for (int i = 0; i < cp_v.size(); i++) {
        perf_func_v.push_back(std::bind(&compiled_partition_executor, cp_v[i],
                std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3));
    }
    int status = measure_perf(t, perf_func_v, inputs_v, outputs_v);
    if (res) res->state = PASSED;

    return status;
}

} // namespace benchdnnext
