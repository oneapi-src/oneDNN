/*******************************************************************************
* Copyright 2023 Intel Corporation
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

/// @example cpu_getting_started.cpp
/// @copybrief graph_cpu_getting_started_cpp
/// > Annotated version: @ref graph_cpu_getting_started_cpp

/// @page graph_cpu_getting_started_cpp Getting started on CPU with Graph API
/// This is an example to demonstrate how to build a simple graph and run it on
/// CPU.
///
/// > Example code: @ref cpu_getting_started.cpp
///
/// Some key take-aways included in this example:
///
/// * how to build a graph and get partitions from it
/// * how to create an engine, allocator and stream
/// * how to compile a partition
/// * how to execute a compiled partition
///
/// Some assumptions in this example:
///
/// * Only workflow is demonstrated without checking correctness
/// * Unsupported partitions should be handled by users themselves
///

/// @page graph_cpu_getting_started_cpp
/// @section graph_cpu_getting_started_cpp_headers Public headers
///
/// To start using oneDNN Graph, we must include the @ref dnnl_graph.hpp header
/// file in the application. All the C++ APIs reside in namespace `dnnl::graph`.
///
/// @page graph_cpu_getting_started_cpp
/// @snippet cpu_getting_started.cpp Headers and namespace
//[Headers and namespace]
#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <assert.h>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "example_utils.hpp"
#include "graph_example_utils.hpp"

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;
//[Headers and namespace]

/// @page graph_cpu_getting_started_cpp
/// @section graph_cpu_getting_started_cpp_tutorial cpu_getting_started_tutorial() function
///
void cpu_getting_started_tutorial() {

    dim N = 8, IC = 3, OC1 = 96, OC2 = 96;
    dim IH = 225, IW = 225, KH1 = 11, KW1 = 11, KH2 = 1, KW2 = 1;

    dims conv0_input_dims {N, IC, IH, IW};
    dims conv0_weight_dims {OC1, IC, KH1, KW1};
    dims conv0_bias_dims {OC1};
    dims conv1_weight_dims {OC1, OC2, KH2, KW2};
    dims conv1_bias_dims {OC2};

    /// @page graph_cpu_getting_started_cpp
    /// @subsection graph_cpu_getting_started_cpp_get_partition Build Graph and Get Partitions
    ///
    /// In this section, we are trying to build a graph containing the pattern
    /// `conv0->relu0->conv1->relu1`. After that, we can get all of
    /// partitions which are determined by backend.
    ///

    /// To build a graph, the connection relationship of different ops must be
    /// known. In oneDNN Graph, #dnnl::graph::logical_tensor is used to express
    /// such relationship. So, next step is to create logical tensors for these
    /// ops including inputs and outputs.
    ///
    /// @note It's not necessary to provide concrete shape/layout information
    /// at graph partitioning stage. Users can provide these information till
    /// compilation stage.
    ///
    /// Create input/output #dnnl::graph::logical_tensor for first `Convolution` op.
    /// @snippet cpu_getting_started.cpp Create conv's logical tensor
    //[Create conv's logical tensor]
    logical_tensor conv0_src_desc {0, data_type::f32};
    logical_tensor conv0_weight_desc {1, data_type::f32};
    logical_tensor conv0_dst_desc {2, data_type::f32};
    //[Create conv's logical tensor]

    /// Create first `Convolution` op (#dnnl::graph::op) and attaches attributes
    /// to it, such as `strides`, `pads_begin`, `pads_end`, `data_format`, etc.
    /// @snippet cpu_getting_started.cpp Create first conv
    //[Create first conv]
    op conv0(0, op::kind::Convolution, {conv0_src_desc, conv0_weight_desc},
            {conv0_dst_desc}, "conv0");
    conv0.set_attr<dims>(op::attr::strides, {4, 4});
    conv0.set_attr<dims>(op::attr::pads_begin, {0, 0});
    conv0.set_attr<dims>(op::attr::pads_end, {0, 0});
    conv0.set_attr<dims>(op::attr::dilations, {1, 1});
    conv0.set_attr<int64_t>(op::attr::groups, 1);
    conv0.set_attr<std::string>(op::attr::data_format, "NCX");
    conv0.set_attr<std::string>(op::attr::weights_format, "OIX");
    //[Create first conv]

    /// Create input/output logical tensors for first `BiasAdd` op and create the first `BiasAdd` op
    /// @snippet cpu_getting_started.cpp Create first bias_add
    //[Create first bias_add]
    logical_tensor conv0_bias_desc {3, data_type::f32};
    logical_tensor conv0_bias_add_dst_desc {
            4, data_type::f32, layout_type::undef};
    op conv0_bias_add(1, op::kind::BiasAdd, {conv0_dst_desc, conv0_bias_desc},
            {conv0_bias_add_dst_desc}, "conv0_bias_add");
    conv0_bias_add.set_attr<std::string>(op::attr::data_format, "NCX");
    //[Create first bias_add]

    /// Create output logical tensors for first `Relu` op and create the op.
    /// @snippet cpu_getting_started.cpp Create first relu
    //[Create first relu]
    logical_tensor relu0_dst_desc {5, data_type::f32};
    op relu0(2, op::kind::ReLU, {conv0_bias_add_dst_desc}, {relu0_dst_desc},
            "relu0");
    //[Create first relu]

    /// Create input/output logical tensors for second `Convolution` op and create the second `Convolution` op.
    /// @snippet cpu_getting_started.cpp Create second conv
    //[Create second conv]
    logical_tensor conv1_weight_desc {6, data_type::f32};
    logical_tensor conv1_dst_desc {7, data_type::f32};
    op conv1(3, op::kind::Convolution, {relu0_dst_desc, conv1_weight_desc},
            {conv1_dst_desc}, "conv1");
    conv1.set_attr<dims>(op::attr::strides, {1, 1});
    conv1.set_attr<dims>(op::attr::pads_begin, {0, 0});
    conv1.set_attr<dims>(op::attr::pads_end, {0, 0});
    conv1.set_attr<dims>(op::attr::dilations, {1, 1});
    conv1.set_attr<int64_t>(op::attr::groups, 1);
    conv1.set_attr<std::string>(op::attr::data_format, "NCX");
    conv1.set_attr<std::string>(op::attr::weights_format, "OIX");
    //[Create second conv]

    /// Create input/output logical tensors for second `BiasAdd` op and create the op.
    /// @snippet cpu_getting_started.cpp Create second bias_add
    //[Create second bias_add]
    logical_tensor conv1_bias_desc {8, data_type::f32};
    logical_tensor conv1_bias_add_dst_desc {9, data_type::f32};
    op conv1_bias_add(4, op::kind::BiasAdd, {conv1_dst_desc, conv1_bias_desc},
            {conv1_bias_add_dst_desc}, "conv1_bias_add");
    conv1_bias_add.set_attr<std::string>(op::attr::data_format, "NCX");
    //[Create second bias_add]

    /// Create output logical tensors for second `Relu` op and create the op
    /// @snippet cpu_getting_started.cpp Create second relu
    //[Create second relu]
    logical_tensor relu1_dst_desc {10, data_type::f32};
    op relu1(5, op::kind::ReLU, {conv1_bias_add_dst_desc}, {relu1_dst_desc},
            "relu1");
    //[Create second relu]

    /// Finally, those created ops will be added into the graph. The graph
    /// inside will maintain a list to store all these ops. To create a graph,
    /// #dnnl::engine::kind is needed because the returned partitions
    /// maybe vary on different devices. For this example, we use CPU engine.
    ///
    /// @note The order of adding op doesn't matter. The connection will
    /// be obtained through logical tensors.
    ///
    /// Create graph and add ops to the graph
    /// @snippet cpu_getting_started.cpp Create graph and add ops
    //[Create graph and add ops]
    graph g(dnnl::engine::kind::cpu);

    g.add_op(conv0);
    g.add_op(conv0_bias_add);
    g.add_op(relu0);

    g.add_op(conv1);
    g.add_op(conv1_bias_add);
    g.add_op(relu1);
    //[Create graph and add ops]

    /// After adding all ops into the graph, call
    /// #dnnl::graph::graph::get_partitions() to indicate that the
    /// graph building is over and is ready for partitioning. Adding new
    /// ops into a finalized graph or partitioning a unfinalized graph
    /// will both lead to a failure.
    ///
    /// @snippet cpu_getting_started.cpp Finialize graph
    //[Finialize graph]
    g.finalize();
    //[Finialize graph]

    /// After finished above operations, we can get partitions by calling
    /// #dnnl::graph::graph::get_partitions().
    ///
    /// In this example, the graph will be partitioned into two partitions:
    /// 1. conv0 + conv0_bias_add + relu0
    /// 2. conv1 + conv1_bias_add + relu1
    ///
    /// @snippet cpu_getting_started.cpp Get partition
    //[Get partition]
    auto partitions = g.get_partitions();
    //[Get partition]

    // Check partitioning results to ensure the examples works. Users do
    // not need to follow this step.
    assert(partitions.size() == 2);

    /// @page graph_cpu_getting_started_cpp
    /// @subsection graph_cpu_getting_started_cpp_compile Compile and Execute Partition
    ///
    /// In the real case, users like framework should provide device information
    /// at this stage. But in this example, we just use a self-defined device to
    /// simulate the real behavior.
    ///
    /// Create a #dnnl::engine. Also, set a user-defined
    /// #dnnl::graph::allocator to this engine.
    ///
    /// @snippet cpu_getting_started.cpp Create engine
    //[Create engine]
    allocator alloc {};
    dnnl::engine eng
            = make_engine_with_allocator(dnnl::engine::kind::cpu, 0, alloc);
    //[Create engine]

    /// Create a #dnnl::stream on a given engine
    ///
    /// @snippet cpu_getting_started.cpp Create stream
    //[Create stream]
    dnnl::stream strm {eng};
    //[Create stream]

    // Mapping from logical tensor id to output tensors
    // used to the connection relationship between partitions (e.g partition 0's
    // output tensor is fed into partition 1)
    std::unordered_map<size_t, tensor> global_outputs_ts_map;

    // Memory buffers binded to the partition input/output tensors
    // that helpe manage the lifetime of these tensors
    std::vector<std::shared_ptr<void>> data_buffer;

    // Mapping from id to queried logical tensor from compiled partition
    // used to record the logical tensors that are previously enabled with
    // ANY layout
    std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;

    // This is a helper function which helps decide which logical tensor is
    // needed to be set with `dnnl::graph::logical_tensor::layout_type::any`
    // layout.
    // This function is not a part to Graph API, but similar logic is
    // essential for Graph API integration to achieve best performance.
    // Typically, users need implement the similar logic in their code.
    std::unordered_set<size_t> ids_with_any_layout;
    set_any_layout(partitions, ids_with_any_layout);

    // Mapping from logical tensor id to the concrete shapes.
    // In practical usage, concrete shapes and layouts are not given
    // until compilation stage, hence need this mapping to mock the step.
    std::unordered_map<size_t, dims> concrete_shapes {{0, conv0_input_dims},
            {1, conv0_weight_dims}, {3, conv0_bias_dims},
            {6, conv1_weight_dims}, {8, conv1_bias_dims}};

    // Compile and execute the partitions, including the following steps:
    //
    // 1. Update the input/output logical tensors with concrete shape and layout
    // 2. Compile the partition
    // 3. Update the output logical tensors with queried ones after compilation
    // 4. Allocate memory and bind the data buffer for the partition
    // 5. Execute the partition
    //
    // Although they are not part of the APIs, these steps are esstential for
    // the integration of Graph API., hence users need to implement similar
    // logic.
    for (const auto &partition : partitions) {
        if (!partition.is_supported()) {
            std::cout
                    << "cpu_get_started: Got unsupported partition, users need "
                       "handle the operators by themselves."
                    << std::endl;
            continue;
        }

        std::vector<logical_tensor> inputs = partition.get_input_ports();
        std::vector<logical_tensor> outputs = partition.get_output_ports();

        // Update input logical tensors with concrete shape and layout
        for (auto &input : inputs) {
            const auto id = input.get_id();
            // If the tensor is an output of another partition,
            // use the cached logical tensor
            if (id_to_queried_logical_tensors.find(id)
                    != id_to_queried_logical_tensors.end())
                input = id_to_queried_logical_tensors[id];
            else
                // Create logical tensor with strided layout
                input = logical_tensor {id, input.get_data_type(),
                        concrete_shapes[id], layout_type::strided};
        }

        // Update output logical tensors with concrete shape and layout
        for (auto &output : outputs) {
            const auto id = output.get_id();
            output = logical_tensor {id, output.get_data_type(),
                    DNNL_GRAPH_UNKNOWN_NDIMS, // set output dims to unknown
                    ids_with_any_layout.count(id) ? layout_type::any
                                                  : layout_type::strided};
        }

        /// Compile the partition to generate compiled partition with the
        /// input and output logical tensors.
        ///
        /// @snippet cpu_getting_started.cpp Compile partition
        //[Compile partition]
        compiled_partition cp = partition.compile(inputs, outputs, eng);
        //[Compile partition]

        // Update output logical tensors with queried one
        for (auto &output : outputs) {
            const auto id = output.get_id();
            output = cp.query_logical_tensor(id);
            id_to_queried_logical_tensors[id] = output;
        }

        // Allocate memory for the partition, and bind the data buffers with
        // input and output logical tensors
        std::vector<tensor> inputs_ts, outputs_ts;
        allocate_graph_mem(inputs_ts, inputs, data_buffer,
                global_outputs_ts_map, eng, /*is partition input=*/true);
        allocate_graph_mem(outputs_ts, outputs, data_buffer,
                global_outputs_ts_map, eng, /*is partition input=*/false);

        /// Execute the compiled partition on the specified stream.
        ///
        /// @snippet cpu_getting_started.cpp Execute compiled partition
        //[Execute compiled partition]
        cp.execute(strm, inputs_ts, outputs_ts);
        //[Execute compiled partition]
    }

    // Wait for all compiled partition's execution finished
    strm.wait();

    /// @page graph_cpu_getting_started_cpp
    ///
    std::cout << "Graph:" << std::endl
              << " [conv0_src] [conv0_wei]" << std::endl
              << "       \\      /" << std::endl
              << "         conv0" << std::endl
              << "          \\    [conv0_bias_src1]" << std::endl
              << "           \\      /" << std::endl
              << "         conv0_bias_add" << std::endl
              << "                |" << std::endl
              << "              relu0" << std::endl
              << "                \\   [conv1_wei]" << std::endl
              << "                 \\    /" << std::endl
              << "                  conv1" << std::endl
              << "                    \\  [conv1_bias_src1]" << std::endl
              << "                     \\      /" << std::endl
              << "                  conv1_bias_add" << std::endl
              << "                          |" << std::endl
              << "                        relu1" << std::endl
              << "                          |" << std::endl
              << "                      [relu_dst]" << std::endl
              << "Note:" << std::endl
              << " '[]' represents a logical tensor, which refers to "
                 "inputs/outputs of the graph. "
              << std::endl;
}

int main(int argc, char **argv) {
    return handle_example_errors(
            {engine::kind::cpu}, cpu_getting_started_tutorial);
}
