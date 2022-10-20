/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

/// @example cpu_multithreading_int8_matmul_relu_pattern.cpp
/// @copybrief cpu_multithreading_int8_matmul_relu_pattern_cpp
/// Annotated version: @ref cpu_multithreading_int8_matmul_relu_pattern_cpp

/// @page cpu_multithreading_int8_matmul_relu_pattern_cpp CPU example for
/// multithreads int8 matmul+relu pattern
///
/// > Example code: @ref cpu_multithreading_int8_matmul_relu_pattern.cpp

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
// input0 -> dequant_in0;
// input1 -> dequant_in1;
// dequant_in0 -> matmul;
// dequant_in1 -> matmul;
// matmul -> relu;
// relu -> quant_out0;
// }

// Test matmul relu different shape compile and execute
// clang-format off
int main(int argc, char **argv) {
    std::cout << "========Example: INT8 MatMul+ReLU========\n";

    engine::kind engine_kind = parse_engine_kind(argc, argv);
    if (engine_kind == engine::kind::gpu) {
        std::cout << "Don't support gpu now\n";
        return -1;
    }

    // Step 2: Construct a graph
    graph g(engine_kind);

    auto &id_mgr = logical_id_manager::get();

    /// Create logical tensor
    std::cout << "Create logical tensor--------------------------";

    std::vector<int64_t> input0_dims {4, 3, 64};
    std::vector<int64_t> input1_dims {3, 64}; // need transpose
    std::vector<int64_t> input2_dims {3}; // bias
    std::vector<int64_t> dst_dims {4, 3, 3};

    logical_tensor dequant_in0_desc {id_mgr["dequant_in0_desc"], data_type::u8, input0_dims, layout_type::strided};
    op input0 {id_mgr["input0"], op::kind::Wildcard, {}, {dequant_in0_desc}, "input0"};

    logical_tensor matmul_input0_desc {id_mgr["matmul_input0_desc"], data_type::f32, layout_type::strided};
    op dequant_in0 {id_mgr["dequant_in0"], op::kind::Dequantize, {dequant_in0_desc}, {matmul_input0_desc}, "dequant_in0"};
    dequant_in0.set_attr<std::vector<float>>("scales", {0.1f});
    dequant_in0.set_attr<std::vector<int64_t>>("zps", {10});
    dequant_in0.set_attr<std::string>("qtype", "per_tensor");

    logical_tensor dequant_in1_desc {id_mgr["dequant_in1_desc"], data_type::s8, input1_dims, layout_type::strided};
    op input1 {id_mgr["input1"], op::kind::Wildcard, {}, {dequant_in1_desc}, "input1"};

    logical_tensor matmul_input1_desc {id_mgr["matmul_input1_desc"], data_type::f32, layout_type::strided};
    op dequant_in1 {id_mgr["dequant_in1"], op::kind::Dequantize, {dequant_in1_desc}, {matmul_input1_desc}, "dequant_in1"};
    dequant_in1.set_attr<std::vector<float>>("scales", {0.1f, 0.1f, 0.1f});
    dequant_in1.set_attr<std::vector<int64_t>>("zps", {0, 0, 0});
    dequant_in1.set_attr<std::string>("qtype", "per_channel");
    dequant_in1.set_attr<int64_t>("axis", 0);

    logical_tensor matmul_input2_desc {id_mgr["matmul_input2_desc"], data_type::f32, layout_type::strided};
    logical_tensor matmul_dst_desc {id_mgr["matmul_dst_desc"], data_type::f32, layout_type::strided};
    op matmul {id_mgr["matmul"], op::kind::MatMul, {matmul_input0_desc, matmul_input1_desc, matmul_input2_desc}, {matmul_dst_desc}, "matmul"};
    matmul.set_attr<bool>("transpose_a", false);
    matmul.set_attr<bool>("transpose_b", true);

    logical_tensor relu_dst_desc {id_mgr["relu_dst_desc"], data_type::f32, dst_dims, layout_type::strided};
    op relu {id_mgr["relu"], op::kind::ReLU, {matmul_dst_desc}, {relu_dst_desc}, "relu"};

    logical_tensor quant_dst_desc {id_mgr["quant_dst_desc"], data_type::u8, dst_dims, layout_type::strided};
    op quant_out0 {id_mgr["quant_out0"], op::kind::Quantize, {relu_dst_desc}, {quant_dst_desc}, "quant_out0"};
    quant_out0.set_attr<std::vector<float>>("scales", {0.1f});
    quant_out0.set_attr<std::vector<int64_t>>("zps", {10});
    quant_out0.set_attr<std::string>("qtype", "per_tensor");
    std::cout << "Success!\n";

    std::unordered_map<size_t, op::kind> op_id_kind_map {{id_mgr["input0"], op::kind::Wildcard},
        {id_mgr["dequant_in0"], op::kind::Dequantize}, {id_mgr["input1"], op::kind::Wildcard},
        {id_mgr["dequant_in1"], op::kind::Dequantize}, {id_mgr["matmul"], op::kind::MatMul},
        {id_mgr["relu"], op::kind::ReLU}, {id_mgr["quant_out0"], op::kind::Quantize}};

    /// Add OP
    std::cout << "Add op to graph--------------------------------";
    g.add_op(input0);
    g.add_op(input1);
    g.add_op(dequant_in0);
    g.add_op(dequant_in1);
    g.add_op(matmul);
    g.add_op(relu);
    g.add_op(quant_out0);
    id_mgr.freeze(); // graph is built up, and the arguments set could be frozen
    std::cout << "Success!\n";

    // Step 3: Filter partitions
    /// Graph will be filtered into 1 partitions: `matmul+relu`
    /// `export DNNL_GRAPH_DUMP=1` can save internal graphs before/after graph fusion into dot files
    std::cout << "Filter partitions------------------------------";
    auto partitions = g.get_partitions();
    std::cout << "Success!\n";

    std::cout << "Number of returned partitions: " << partitions.size() << "\n";
    for (size_t i = 0; i < partitions.size(); ++i) {
        std::cout << "Partition[" << partitions[i].get_id()
                  << "]'s supporting status: "
                  << (partitions[i].is_supported() ? "true" : "false") << "\n";
    }

    /// FIXME(wuxun): this is to simulate the current issue: PyTorch cannot get
    /// dims when adding op to graph
    matmul_input2_desc = logical_tensor {id_mgr["matmul_input2_desc"], data_type::f32, input2_dims, layout_type::strided};

    /// mark the output logical tensors of partition as ANY layout enabled
    std::unordered_set<size_t> id_to_set_any_layout;
    set_any_layout(partitions, id_to_set_any_layout);

    /// construct a new engine
    engine e {engine_kind, 0};

    /// construct a new stream
    stream s {e};

    std::vector<compiled_partition> c_partitions(partitions.size());

    size_t thread_num = 8; 

    // mapping from id to tensors
    std::vector<tensor_map> tms(thread_num);

    // mapping from id to queried logical tensor from compiled partition
    // used to record the logical tensors that are previously enabled with ANY layout
    std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;

    for (size_t i = 0; i < partitions.size(); ++i) {
        if (partitions[i].is_supported()) {
            std::cout << "\nPartition[" << partitions[i].get_id() << "] is being processed.\n";
            // FIXME(xx) There is an issue that pytorch still cannot get ndims info when adding ops 
            // to graph. So, in this example, we simulate this scenario to test our implementation.
            // Before adding op to graph, the ndims of matmul's inputs are unknown (see Line#79, 
            // Line#88 and Line#95).
            // std::vector<logical_tensor> inputs = partitions[i].get_inputs();
            // std::vector<logical_tensor> outputs = partitions[i].get_outputs();
            std::vector<logical_tensor> inputs {dequant_in0_desc, dequant_in1_desc, matmul_input2_desc};
            std::vector<logical_tensor> outputs {quant_dst_desc};

            /// replace input logical tensor with the queried one
            replace_with_queried_logical_tensors(inputs, id_to_queried_logical_tensors);

            /// update output logical tensors with ANY layout
            update_tensors_with_any_layout(outputs, id_to_set_any_layout);

            std::cout << "Compiling--------------------------------------";
            /// compile to generate compiled partition
            c_partitions[i] = partitions[i].compile(inputs, outputs, e);
            std::cout << "Success!\n";

            record_queried_logical_tensors(partitions[i].get_out_ports(), c_partitions[i],
                id_to_queried_logical_tensors);

            auto thread_func = [&](size_t tid) {
                std::cout << "Start thread " << tid << std::endl;
                std::vector<tensor> input_ts = tms[tid].construct_and_initialize_tensors(inputs, c_partitions[i], e, 1);
                std::vector<tensor> output_ts = tms[tid].construct_and_initialize_tensors(outputs, c_partitions[i], e, 0);
                /// construct a new stream
                stream s {e};

                /// execute the compiled partition
                c_partitions[i].execute(s, input_ts, output_ts);
                std::cout << "End thread " << tid << std::endl;

            };

            std::vector<std::thread> workers;
            for (size_t t_num=0; t_num < thread_num; t_num++) {
                workers.emplace_back(thread_func, t_num);
            }

            for (size_t t_num=0; t_num < thread_num; t_num++) {
                workers[t_num].join();
            }
        } else {
            std::vector<size_t> unsupported_op_ids = partitions[i].get_ops();
            assertm(unsupported_op_ids.size() == 1, "Unsupported partition only "
                "contains single op.");
            if (op_id_kind_map[unsupported_op_ids[0]] == op::kind::Wildcard) {
                std::cout << "\nWarning (actually an error): partition " << partitions[i].get_id() <<
                        " contains only a Wildcard op which cannot be computed.\n";
            } else {
                /// Users need to write implementation code by themselves.
                continue;
            }
        }
    }
    
    std::cout << "Check correctness------------------------------";
    std::cout << "Skipped!\n";

    std::cout << "============Run Example Successfully===========\n";

    return 0;
}
// clang-format on
