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

/// @example cpu_multithreading_get_partitions.cpp
/// @copybrief cpu_multithreading_get_partitions_cpp
/// Annotated version: @ref cpu_multithreading_get_partitions_cpp

/// @page cpu_multithreading_get_partitions_cpp CPU example for multithread matmul+relu pattern
///
/// > Example code: @ref cpu_multithreading_get_partitions.cpp

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "common/execution_context.hpp"
#include "common/helpers_any_layout.hpp"
#include "common/utils.hpp"

#define assertm(exp, msg) assert(((void)msg, exp))

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;

// digraph G {
// Wildcard -> MatMul;
// MatMul -> ReLU;
// }

// Test matmul relu different shape compile and execute
int main(int argc, char **argv) {
    std::cout << "========Example: MatMul+ReLU========\n";

    engine::kind engine_kind = parse_engine_kind(argc, argv);
    if (engine_kind == engine::kind::gpu) {
        std::cout << "Don't support gpu now\n";
        return -1;
    }

    // Step 2: Construct several graphs
    size_t num_graphs = 20; // this is also the thread num
    auto thread_func = [&](size_t tid) {
        std::cout << "Start thread " << tid << std::endl;
        graph g(engine_kind);
        std::vector<int64_t> input0_dims {1, 64};
        std::vector<int64_t> input1_dims {64, 1};
        std::vector<int64_t> dst_dims {1, 1};

        logical_tensor matmul_input0_desc {
                7 * tid, data_type::f32, input0_dims, layout_type::strided};
        logical_tensor matmul_input1_desc {
                7 * tid + 1, data_type::f32, input1_dims, layout_type::strided};
        logical_tensor matmul_dst_desc {
                7 * tid + 2, data_type::f32, dst_dims, layout_type::strided};

        op wildcard {7 * tid + 3, op::kind::Wildcard, {},
                {matmul_input0_desc, matmul_input1_desc}, "wildcard"};

        op matmul {7 * tid + 4, op::kind::MatMul,
                {matmul_input0_desc, matmul_input1_desc}, {matmul_dst_desc},
                "matmul"};

        logical_tensor relu_dst_desc {
                7 * tid + 5, data_type::f32, dst_dims, layout_type::strided};

        op relu {7 * tid + 6, op::kind::ReLU, {matmul_dst_desc},
                {relu_dst_desc}, "relu"};

        /// Add OP
        std::cout << "thread " << tid
                  << ": Add op to graph--------------------------------";
        g.add_op(wildcard);
        g.add_op(matmul);
        g.add_op(relu);

        // Step 3: Filter partitions
        /// Graph will be filtered into 1 partitions: `matmul+relu`
        /// `export DNNL_GRAPH_DUMP=1` can save internal graphs before/after graph fusion into dot files
        std::cout << "thread " << tid
                  << ": Filter partitions------------------------------";
        auto partitions = g.get_partitions();

        std::cout << "thread " << tid
                  << ": Number of returned partitions: " << partitions.size()
                  << "\n";
        for (size_t i = 0; i < partitions.size(); ++i) {
            std::cout << "Partition[" << partitions[i].get_id()
                      << "]'s supporting status: "
                      << (partitions[i].is_supported() ? "true" : "false")
                      << "\n";
        }
    };

    std::vector<std::thread> workers;
    for (size_t t_num = 0; t_num < num_graphs; t_num++) {
        workers.emplace_back(thread_func, t_num);
    }

    for (size_t t_num = 0; t_num < num_graphs; t_num++) {
        workers[t_num].join();
    }

    std::cout << "============Run Example Successfully===========\n";

    return 0;
}
