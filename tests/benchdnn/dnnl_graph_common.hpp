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

#ifndef DNNL_GRAPH_COMMON_HPP
#define DNNL_GRAPH_COMMON_HPP

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "dnnl_common.hpp"

namespace benchgraph {

using dims_t = dnnl::graph::logical_tensor::dims_t;
using dim_t = dims_t::value_type;

using input_list_t = std::vector<dnnl::graph::logical_tensor>;
using output_list_t = std::vector<dnnl::graph::logical_tensor>;

#define BENCHGRAPH_SAFE(f, s) \
    do { \
        try { \
            f; \
        } catch (const dnnl::graph::error &e) { \
            if (s == CRIT || s == WARN) { \
                BENCHDNN_PRINT(0, "error [%s:%d]: '%s' -> %s\n", \
                        __PRETTY_FUNCTION__, __LINE__, #f, e.what()); \
                fflush(0); \
                if (s == CRIT) exit(2); \
            } \
            return FAIL; \
        } \
    } while (0)

struct id_manager {
    id_manager() : frozen_(false) {};

    size_t operator[](const std::string &arg) {
        const auto &it = knots_.find(arg);
        if (it != knots_.end()) return it->second;
        if (frozen_) {
            std::cout << "Unrecognized argument [" << arg << "]!\n";
            std::abort();
        }
        const auto &new_it = knots_.emplace(arg, knots_.size());
        if (new_it.second) {
            return new_it.first->second;
        } else {
            std::cout << "New argument [" << arg
                      << "] is failed to be added to knots.\n";
            std::abort();
        }
    }

    void freeze() { frozen_ = true; }

private:
    std::map<std::string, size_t> knots_;
    // indicates that the graph is frozen
    bool frozen_;
};

typedef std::function<void(dnnl::graph::stream &,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs)>
        perf_function_t;

// Engine used to run oneDNN fusion patterns for testing.
inline dnnl::graph::engine &get_test_engine() {
    dnnl::graph::engine::kind graph_engine_kind = engine_tgt_kind == dnnl_cpu
            ? dnnl::graph::engine::kind::cpu
            : dnnl::graph::engine::kind::gpu;
    static dnnl::graph::engine instance(
            graph_engine_kind, static_cast<int>(engine_index));
    return instance;
}

int measure_perf(benchdnn_timer_t &t, perf_function_t &perf_func,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs);

int measure_perf(benchdnn_timer_t &t, dnnl::graph::compiled_partition &cp,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs);

} // namespace benchgraph

#endif