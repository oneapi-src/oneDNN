/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

/// @example cpu_inplace_options.cpp
/// @copybrief cpu_inplace_options_cpp
/// Annotated version: @ref cpu_inplace_options_cpp

/// @page cpu_inplace_options_cpp CPU example support inplace options
///
/// > Example code: @ref cpu_inplace_options.cpp

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

// ======== Unoptimized graph ========
// digraph G {
//   Wildcard_100002 -> Convolution_100003;
//   Convolution_100003 -> BiasAdd_100004;
//   Wildcard_100005 -> Convolution_100006;
//   Convolution_100006 -> BiasAdd_100007;
//   BiasAdd_100004 -> Add_100008;
//   BiasAdd_100007 -> Add_100008;
// }
// ======== Optimized graph ==========
// digraph G {
//   Wildcard_100005 -> Conv_bias_100215;
//   Wildcard_100002 -> Conv_bias_add_100125;
//   Conv_bias_100215 -> Conv_bias_add_100125;
// }
// clang-format off
int main(int argc, char **argv) {
    std::cout << "========Example: Add(Conv+BiasAdd, Conv+BiasAdd)========\n";

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

    std::vector<int64_t> conv0_input_dims {8, 256, 56, 56};
    std::vector<int64_t> conv0_weight_dims {64, 256, 1, 1};
    std::vector<int64_t> conv0_bias_dims {64};
    std::vector<int64_t> conv0_dst_dims {8, 64, 56, 56};

    std::vector<int64_t> conv1_input_dims {8, 256, 56, 56};
    std::vector<int64_t> conv1_weight_dims {64, 256, 1, 1};
    std::vector<int64_t> conv1_bias_dims {64};
    std::vector<int64_t> conv1_dst_dims {8, 64, 56, 56};

    logical_tensor conv0_src_desc {id_mgr["conv0_src"], data_type::f32, conv0_input_dims, layout_type::strided};
    logical_tensor conv0_weight_desc {id_mgr["conv0_weight"], data_type::f32, conv0_weight_dims, layout_type::strided};
    logical_tensor conv0_bias_desc {id_mgr["conv0_bias"], data_type::f32, conv0_bias_dims, layout_type::strided};
    logical_tensor conv0_dst_desc {id_mgr["conv0_dst"], data_type::f32, conv0_dst_dims, layout_type::strided};
    
    op conv_input0(id_mgr["conv_input0"], op::kind::Wildcard, {}, {conv0_src_desc}, "conv_input0");

    op conv0(id_mgr["conv0"], op::kind::Convolution, {conv0_src_desc, conv0_weight_desc}, {conv0_dst_desc}, "conv0");
    conv0.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv0.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv0.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv0.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv0.set_attr<std::string>("data_format", "NCX");
    conv0.set_attr<std::string>("filter_format", "OIX");
    conv0.set_attr<int64_t>("groups", 1);

    logical_tensor conv0_bias_dst_desc {id_mgr["conv0_bias_dst"], data_type::f32, conv0_dst_dims, layout_type::strided};
    
    op bias0(id_mgr["bias0"], op::kind::BiasAdd, {conv0_dst_desc, conv0_bias_desc}, {conv0_bias_dst_desc}, "bias0");

    logical_tensor conv1_src_desc {id_mgr["conv1_src"], data_type::f32, conv1_input_dims, layout_type::strided};
    logical_tensor conv1_weight_desc {id_mgr["conv1_weight"], data_type::f32, conv1_weight_dims, layout_type::strided};
    logical_tensor conv1_bias_desc {id_mgr["conv1_bias"], data_type::f32, conv1_bias_dims, layout_type::strided};
    logical_tensor conv1_dst_desc {id_mgr["conv1_dst"], data_type::f32, conv1_dst_dims, layout_type::strided};
    
    op conv_input1(id_mgr["conv_input1"], op::kind::Wildcard, {}, {conv1_src_desc}, "conv_input1");

    op conv1(id_mgr["conv1"], op::kind::Convolution, {conv1_src_desc, conv1_weight_desc}, {conv1_dst_desc}, "conv1");
    conv1.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv1.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv1.set_attr<std::string>("data_format", "NCX");
    conv1.set_attr<std::string>("filter_format", "OIX");
    conv1.set_attr<int64_t>("groups", 1);
    
    logical_tensor conv1_bias_dst_desc {id_mgr["conv1_bias_dst"], data_type::f32, conv1_dst_dims, layout_type::strided};
    
    op bias1(id_mgr["bias1"], op::kind::BiasAdd, {conv1_dst_desc, conv1_bias_desc}, {conv1_bias_dst_desc}, "bias1");

    logical_tensor add_dst_desc {id_mgr["add_dst"], data_type::f32, conv0_dst_dims, layout_type::strided};
    
    op add(id_mgr["add"], op::kind::Add, {conv0_bias_dst_desc, conv1_bias_dst_desc}, {add_dst_desc}, "add");
    
    logical_tensor abs_dst_desc {id_mgr["abs_dst"], data_type::f32, conv0_dst_dims, layout_type::strided};

    op abs(id_mgr["abs"], op::kind::Abs, {add_dst_desc}, {abs_dst_desc}, "abs0");
    std::cout << "Success!\n";

    std::unordered_map<size_t, op::kind> op_id_kind_map {{id_mgr["conv_input0"], op::kind::Wildcard},
        {id_mgr["conv0"], op::kind::Convolution}, {id_mgr["conv_input1"], op::kind::Wildcard},
        {id_mgr["bias0"], op::kind::BiasAdd}, {id_mgr["conv1"], op::kind::Convolution}, {id_mgr["bias1"], op::kind::BiasAdd},
        {id_mgr["add"], op::kind::Add}, {id_mgr["abs"], op::kind::Abs}};

    /// Add OP
    std::cout << "Add OP to graph--------------------------------";
    g.add_op(conv_input0);
    g.add_op(conv0);
    g.add_op(bias0);
    g.add_op(conv_input1);
    g.add_op(conv1);
    g.add_op(bias1);
    g.add_op(add);
    g.add_op(abs);
    id_mgr.freeze(); // graph is built up, and the arguments set could be frozen
    std::cout << "Success!\n";

    // Step 3: Filter and get partitions
    /// Graph will be filtered into 2 partitions: `conv0+bias0+sum and `conv1+bias1`
    /// `export DNNL_GRAPH_DUMP=1` can save internal graphs before/after graph fusion into dot files
    std::cout << "Filter and get partition-----------------------";
    auto partitions = g.get_partitions(partition::policy::fusion);
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

    // Initialize engine and stream
    const int32_t device_id = 0;
    engine eng {engine_kind, device_id};
    allocator alloc {&allocate, &deallocate};
    eng.set_allocator(alloc);
    stream strm {eng};

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
            c_partitions[i] = partitions[i].compile(inputs, outputs, eng);
            std::cout << "Success!\n";

            record_queried_logical_tensors(partitions[i].get_out_ports(), c_partitions[i],
                id_to_queried_logical_tensors);

            std::cout << "Creating tensors and allocating memory buffer--";
            std::vector<tensor> input_ts = tm.construct_and_initialize_tensors(inputs, c_partitions[i], 1);
            std::vector<tensor> output_ts = tm.construct_and_initialize_tensors(outputs, c_partitions[i], 0);
            std::cout << "Success!\n";
            
            // get inplace pairs and set tensor's handle accordingly
            auto inplace_option = c_partitions[i].get_inplace_ports();
            std::cout << "This partition has " << inplace_option.size() << " in-place option(s)\n";
            tm.update_tensor_handle_by_inplace_options(input_ts, output_ts, inputs, outputs, inplace_option);

            std::cout << "Executing compiled partition-------------------";
            /// execute the compiled partition
            c_partitions[i].execute(strm, input_ts, output_ts);
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

    // Step 6: Check correctness of the output results
    std::cout << "Check correctness------------------------------";
    float expected_result = (1 * 1 * 1 * 256 + /* conv0 bias */ 1.f) * 2.f;

    float *actual_output_ptr = tm.get(add_dst_desc.get_id()).get_data_handle<float>();
    auto output_dims = add_dst_desc.get_dims();
    auto num_elem = product(output_dims);
    std::vector<float> expected_output(num_elem, expected_result);
    compare_data(expected_output.data(), actual_output_ptr, num_elem);
    std::cout << "Success!\n";

    std::cout << "============Run Example Successfully===========\n";
    return 0;
}
// clang-format on
