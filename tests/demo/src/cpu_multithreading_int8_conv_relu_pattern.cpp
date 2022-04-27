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

/// @example cpu_multithreading_int8_conv_relu_pattern.cpp
/// @copybrief cpu_multithreading_int8_conv_relu_pattern_cpp
/// Annotated version: @ref cpu_multithreading_int8_conv_relu_pattern_cpp

/// @page cpu_multithreading_int8_conv_relu_pattern_cpp CPU example for
/// multithreads int8 conv+relu pattern
///
/// > Example code: @ref cpu_multithreading_int8_conv_relu_pattern.cpp

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unordered_set>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "common/execution_context.hpp"
#include "common/helpers_any_layout.hpp"
#include "common/utils.hpp"

#define assertm(exp, msg) assert(((void)msg, exp))

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;

// digraph G {
// input0 -> dequant0;
// dequant0 -> conv0;
// dequant1 -> conv0;
// conv0 -> relu0;
// relu0 -> quant0;
// }

// Test conv relu different shape compile and execute
// clang-format off
int main(int argc, char **argv) {
    std::cout << "========Example: INT8 Conv+ReLU========\n";

    engine::kind engine_kind = parse_engine_kind(argc, argv);
    if (engine_kind == engine::kind::gpu) {
        std::cout << "Don't support gpu now\n";
        return -1;
    }

    // Construct a graph
    graph g(engine_kind);

    auto &id_mgr = logical_id_manager::get();

    /// Create logical tensor
    std::cout << "Create logical tensor--------------------------";

    std::vector<int64_t> input_dims {8, 56, 56, 256}; // NXC
    std::vector<int64_t> conv0_weight_dims {1, 1 , 256, 64}; // XIO
    std::vector<int64_t> conv0_bias_dims {64};
    std::vector<int64_t> conv0_dst_dims {8, 56, 56, 64};

    logical_tensor dequant0_src_desc {id_mgr["dequant0_src"], data_type::u8, input_dims, layout_type::strided};
    op input0(id_mgr["input0"], op::kind::Wildcard, {}, {dequant0_src_desc}, "input0");

    logical_tensor conv0_src_desc {id_mgr["conv0_src"], data_type::f32, input_dims, layout_type::strided};
    op dequant0(id_mgr["dequant0"], op::kind::Dequantize, {dequant0_src_desc}, {conv0_src_desc}, "dequant0");
    dequant0.set_attr<std::string>("qtype", "per_tensor");
    dequant0.set_attr<std::vector<float>>("scales", {0.1f});
    dequant0.set_attr<std::vector<int64_t>>("zps", {10});

    logical_tensor dequant1_src_desc {id_mgr["dequant1_src"], data_type::s8, conv0_weight_dims, layout_type::strided};
    logical_tensor conv0_weight_desc {id_mgr["conv0_weight"], data_type::f32, conv0_weight_dims, layout_type::strided};
    op dequant1(id_mgr["dequant1"], op::kind::Dequantize, {dequant1_src_desc}, {conv0_weight_desc}, "dequant1");
    dequant1.set_attr<std::string>("qtype", "per_channel");
    std::vector<float> wei_scales(64, 0.1f);
    std::vector<int64_t> wei_zps(64, 0);
    dequant1.set_attr<std::vector<float>>("scales", wei_scales);
    dequant1.set_attr<std::vector<int64_t>>("zps", wei_zps);
    dequant1.set_attr<int64_t>("axis", 1);

    logical_tensor conv0_bias_desc {id_mgr["conv0_bias"], data_type::f32, conv0_bias_dims, layout_type::strided};
    logical_tensor conv0_dst_desc {id_mgr["conv0_dst"], data_type::f32, conv0_dst_dims, layout_type::strided};
    op conv0(id_mgr["conv0"], op::kind::Convolution, {conv0_src_desc, conv0_weight_desc, conv0_bias_desc}, {conv0_dst_desc}, "conv0");
    conv0.set_attr<std::vector<int64_t>>("strides",  {1, 1});
    conv0.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv0.set_attr<std::vector<int64_t>>("pads_end",  {0, 0});
    conv0.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv0.set_attr<std::string>("data_format", "NXC");
    conv0.set_attr<std::string>("filter_format", "XIO");
    conv0.set_attr<int64_t>("groups", 1);

    logical_tensor relu0_dst_desc {id_mgr["relu0_dst"], data_type::f32, conv0_dst_dims, layout_type::strided};
    op relu0(id_mgr["relu0"], op::kind::ReLU, {conv0_dst_desc}, {relu0_dst_desc}, "relu0");

    logical_tensor quant0_dst_desc {id_mgr["quant0_dst"], data_type::u8, conv0_dst_dims, layout_type::strided};
    op quant0(id_mgr["quant0"], op::kind::Quantize, {relu0_dst_desc}, {quant0_dst_desc}, "quant0");
    quant0.set_attr<std::string>("qtype", "per_tensor");
    quant0.set_attr<std::vector<float>>("scales", {0.1f});
    quant0.set_attr<std::vector<int64_t>>("zps", {10});
    std::cout << "Success!\n";

    std::unordered_map<size_t, op::kind> op_id_kind_map {{id_mgr["input0"], op::kind::Wildcard}, {id_mgr["dequant0"], op::kind::Dequantize},
            {id_mgr["dequant1"], op::kind::Dequantize}, {id_mgr["conv0"], op::kind::Convolution}, {id_mgr["relu0"], op::kind::ReLU},
            {id_mgr["quant0"], op::kind::Quantize}};

    /// Select OP
    std::cout << "Select op to graph-----------------------------";
    g.add_op(input0);
    g.add_op(dequant0);
    g.add_op(dequant1);
    g.add_op(conv0);
    g.add_op(relu0);
    g.add_op(quant0);
    id_mgr.freeze(); // graph is built up, and the arguments set could be frozen
    std::cout << "Success!\n";

    // Step 3: Filter partitions
    /// Graph will be filtered into 1 partitions: `conv0+relu0`
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
            std::cout << "\nPartition[" << partitions[i].get_id()
                      << "] is being processed.\n";
            std::vector<logical_tensor> inputs = partitions[i].get_in_ports();
            std::vector<logical_tensor> outputs = partitions[i].get_out_ports();

            /// replace input logical tensor with the queried one
            replace_with_queried_logical_tensors(inputs, id_to_queried_logical_tensors);

            /// update output logical tensors with ANY layout
            update_tensors_with_any_layout(outputs, id_to_set_any_layout);

            std::cout << "Compiling--------------------------------------";
            /// compile to generate compiled partition
            c_partitions[i] = partitions[i].compile(inputs, outputs, e);
            std::cout << "Success!\n";

            record_queried_logical_tensors(partitions[i].get_out_ports(),
                    c_partitions[i], id_to_queried_logical_tensors);

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

    // Step 6 : Check correctness of the output results
    std::cout << "Check correctness------------------------------";
    std::cout << "Skipped!\n";

    std::cout << "============Run Example Successfully===========\n";

    return 0;
}
// clang-format on
