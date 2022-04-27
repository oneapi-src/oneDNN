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

/// @example cpu_int8_maxpool_pattern.cpp
/// @copybrief cpu_int8_maxpool_pattern_cpp
/// Annotated version: @ref cpu_int8_maxpool_pattern_cpp

/// @page cpu_int8_maxpool_pattern_cpp CPU example for int8 maxpooling pattern
///
/// > Example code: @ref cpu_int8_maxpool_pattern.cpp

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
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
// dequant_in0 -> maxpool;
// matmul -> quant_out;
// }

// clang-format off
int main(int argc, char **argv) {
    std::cout << "========Example: INT8 MaxPool========\n";

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

    std::vector<int64_t> src_dims {3, 27, 27, 3}; // NXC
    // OH = (IH - ((KH - 1) * DH + KH) + PH_L + PH_R) / SH + 1
    // OW = (IW - ((KW - 1) * DW + KW) + PW_L + PW_R) / SW + 1
    std::vector<int64_t> dst_dims {3, 5, 5, 3}; // NXC

    logical_tensor dequant_src_desc {id_mgr["dequant_src_desc"], data_type::u8, src_dims, layout_type::strided};
    logical_tensor maxpool_src_desc {id_mgr["maxpool_src_desc"], data_type::f32, src_dims, layout_type::strided};
    op dequant_in {id_mgr["dequant_in0"], op::kind::Dequantize, {dequant_src_desc}, {maxpool_src_desc}, "dequant_in"};
    dequant_in.set_attr<std::vector<float>>("scales", {0.1f});
    dequant_in.set_attr<std::vector<int64_t>>("zps", {10});
    dequant_in.set_attr<std::string>("qtype", "per_tensor");

    logical_tensor maxpool_dst_desc {id_mgr["maxpool_dst_desc"], data_type::f32, dst_dims, layout_type::strided};
    op maxpool {id_mgr["maxpool"], op::kind::MaxPool, {maxpool_src_desc}, {maxpool_dst_desc}, "maxpool"};
    maxpool.set_attr<std::vector<int64_t>>("strides", {4, 4});
    maxpool.set_attr<std::vector<int64_t>>("kernel", {11, 11});
    maxpool.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    maxpool.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    maxpool.set_attr<std::string>("data_format", "NXC");
    maxpool.set_attr<std::vector<int64_t>>("dilations", {1, 1});

    logical_tensor quant_dst_desc {id_mgr["quant_dst_desc"], data_type::u8, dst_dims, layout_type::strided};
    op quant_out {id_mgr["quant_out0"], op::kind::Quantize, {maxpool_dst_desc}, {quant_dst_desc}, "quant_out"};
    quant_out.set_attr<std::vector<float>>("scales", {0.1f});
    quant_out.set_attr<std::vector<int64_t>>("zps", {10});
    quant_out.set_attr<std::string>("qtype", "per_tensor");
    std::cout << "Success!\n";

    std::unordered_map<size_t, op::kind> op_id_kind_map {{id_mgr["dequant_in0"], op::kind::Dequantize},
        {id_mgr["maxpool"], op::kind::MaxPool}, {id_mgr["quant_out0"], op::kind::Quantize}};

    /// Add OP
    std::cout << "Add op to graph--------------------------------";
    g.add_op(dequant_in);
    g.add_op(maxpool);
    g.add_op(quant_out);
    id_mgr.freeze(); // graph is built up, and the arguments set could be frozen
    std::cout << "Success!\n";

    // Step 3: Filter partitions
    /// Graph will be filtered into 1 partitions: `maxpool`
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

    // mapping from id to tensors
    tensor_map tm;

    // mapping from id to queried logical tensor from compiled partition
    // used to record the logical tensors that are previously enabled with ANY layout
    std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;

    for (size_t i = 0; i < partitions.size(); ++i) {
        if (partitions[i].is_supported()) {
            std::cout << "\nPartition[" << partitions[i].get_id() << "] is being processed.\n";
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

            record_queried_logical_tensors(partitions[i].get_out_ports(), c_partitions[i],
                id_to_queried_logical_tensors);

            std::cout << "Creating tensors and allocating memory buffer--";
            std::vector<tensor> input_ts = tm.construct_and_initialize_tensors(inputs, c_partitions[i], e, 1);
            std::vector<tensor> output_ts = tm.construct_and_initialize_tensors(outputs, c_partitions[i], e, 0);
            std::cout << "Success!\n";

            std::cout << "Executing compiled partition-------------------";
            /// execute the compiled partition
            c_partitions[i].execute(s, input_ts, output_ts);
            std::cout << "Success!\n";
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
