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

/// @example cpu_inference_int8.cpp
/// @copybrief graph_cpu_inference_int8_cpp
/// Annotated version: @ref graph_cpu_inference_int8_cpp

/// @page graph_cpu_inference_int8_cpp Convolution int8 inference example with Graph API
/// This is an example to demonstrate how to build an int8 graph with Graph
/// API and run it on CPU.
///
/// > Example code: @ref cpu_inference_int8.cpp
///
/// Some assumptions in this example:
///
/// * Only workflow is demonstrated without checking correctness
/// * Unsupported partitions should be handled by users themselves
///

/// @page graph_cpu_inference_int8_cpp
/// @section graph_cpu_inference_int8_cpp_headers Public headers
///
/// To start using oneDNN Graph, we must include the @ref dnnl_graph.hpp header
/// file in the application. All the C++ APIs reside in namespace `dnnl::graph`.
///
/// @page graph_cpu_inference_int8_cpp
/// @snippet cpu_inference_int8.cpp Headers and namespace
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
using property_type = logical_tensor::property_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;
//[Headers and namespace]

/// @page graph_cpu_inference_int8_cpp
/// @section graph_cpu_inference_int8_cpp_tutorial simple_pattern_int8() function
///
void simple_pattern_int8() {

    dim N = 8, IC = 256, IH = 56, IW = 56, KH = 1, KW = 1, OC = 64;

    dims conv_input_dims {N, IH, IW, IC};
    dims conv_weight_dims {KH, KW, IC, OC};
    dims conv_bias_dims {OC};

    /// @page graph_cpu_inference_int8_cpp
    /// @subsection graph_cpu_inference_int8_cpp_get_partition Build Graph and Get Partitions
    ///
    /// In this section, we are trying to build a graph indicating an int8
    /// convolution with relu post-op. After that, we can get all of
    /// partitions which are determined by backend.
    ///
    /// Create input/output #dnnl::graph::logical_tensor and op for the first `Dequantize`.
    /// @snippet cpu_inference_int8.cpp Create dequant's logical tensor and the op
    //[Create dequant's logical tensor and the op]
    logical_tensor dequant0_src_desc {0, data_type::u8};
    logical_tensor conv_src_desc {1, data_type::f32};
    op dequant0(2, op::kind::Dequantize, {dequant0_src_desc}, {conv_src_desc},
            "dequant0");
    dequant0.set_attr<std::string>(op::attr::qtype, "per_tensor");
    dequant0.set_attr<std::vector<float>>(op::attr::scales, {0.1f});
    dequant0.set_attr<std::vector<int64_t>>(op::attr::zps, {10});
    //[Create dequant's logical tensor and the op]

    /// Create input/output #dnnl::graph::logical_tensor and op for the second `Dequantize`.
    ///
    /// @note It's necessary to provide scale and weight information
    /// on the `Dequantize` on weight.
    ///
    /// @note Users can set weight property type to `constant` to enable
    /// dnnl weight cache for better performance
    ///
    /// @snippet cpu_inference_int8.cpp Create dequant's logical tensor and the op.
    //[Create dequant's logical tensor and the op.]
    logical_tensor dequant1_src_desc {3, data_type::s8};
    logical_tensor conv_weight_desc {
            4, data_type::f32, 4, layout_type::undef, property_type::constant};
    op dequant1(5, op::kind::Dequantize, {dequant1_src_desc},
            {conv_weight_desc}, "dequant1");
    dequant1.set_attr<std::string>(op::attr::qtype, "per_channel");
    // the memory format of weight is XIO, which indicates channel equals
    // to 64 for the convolution.
    std::vector<float> wei_scales(64, 0.1f);
    dims wei_zps(64, 0);
    dequant1.set_attr<std::vector<float>>(op::attr::scales, wei_scales);
    dequant1.set_attr<std::vector<int64_t>>(op::attr::zps, wei_zps);
    dequant1.set_attr<int64_t>(op::attr::axis, 1);
    //[Create dequant's logical tensor and the op.]

    /// Create input/output #dnnl::graph::logical_tensor the op for `Convolution`.
    /// @snippet cpu_inference_int8.cpp Create conv's logical tensor and the op
    //[Create conv's logical tensor and the op]
    logical_tensor conv_bias_desc {
            6, data_type::f32, 1, layout_type::undef, property_type::constant};
    logical_tensor conv_dst_desc {7, data_type::f32, layout_type::undef};

    // create the convolution op
    op conv(8, op::kind::Convolution,
            {conv_src_desc, conv_weight_desc, conv_bias_desc}, {conv_dst_desc},
            "conv");
    conv.set_attr<dims>(op::attr::strides, {1, 1});
    conv.set_attr<dims>(op::attr::pads_begin, {0, 0});
    conv.set_attr<dims>(op::attr::pads_end, {0, 0});
    conv.set_attr<dims>(op::attr::dilations, {1, 1});
    conv.set_attr<std::string>(op::attr::data_format, "NXC");
    conv.set_attr<std::string>(op::attr::weights_format, "XIO");
    conv.set_attr<int64_t>(op::attr::groups, 1);
    //[Create conv's logical tensor and the op]

    /// Create input/output #dnnl::graph::logical_tensor the op for `ReLu`.
    /// @snippet cpu_inference_int8.cpp Create ReLu's logical tensor and the op
    //[Create ReLu's logical tensor and the op]
    logical_tensor relu_dst_desc {9, data_type::f32, layout_type::undef};
    op relu(10, op::kind::ReLU, {conv_dst_desc}, {relu_dst_desc}, "relu");
    //[Create ReLu's logical tensor and the op]

    /// Create input/output #dnnl::graph::logical_tensor the op for `Quantize`.
    /// @snippet cpu_inference_int8.cpp Create Quantize's logical tensor and the op
    //[Create Quantize's logical tensor and the op]
    logical_tensor quant_dst_desc {11, data_type::u8, layout_type::undef};
    op quant(
            12, op::kind::Quantize, {relu_dst_desc}, {quant_dst_desc}, "quant");
    quant.set_attr<std::string>(op::attr::qtype, "per_tensor");
    quant.set_attr<std::vector<float>>(op::attr::scales, {0.1f});
    quant.set_attr<std::vector<int64_t>>(op::attr::zps, {10});
    //[Create Quantize's logical tensor and the op]

    /// Finally, those created ops will be added into the graph. The graph
    /// inside will maintain a list to store all these ops. To create a graph,
    /// #dnnl::engine::kind is needed because the returned partitions
    /// maybe vary on different devices. For this example, we use CPU engine.
    ///
    /// @note The order of adding op doesn't matter. The connection will
    /// be obtained through logical tensors.
    ///
    /// Create graph and add ops to the graph
    /// @snippet cpu_inference_int8.cpp Create graph and add ops
    //[Create graph and add ops]
    graph g(dnnl::engine::kind::cpu);

    g.add_op(dequant0);
    g.add_op(dequant1);
    g.add_op(conv);
    g.add_op(relu);
    g.add_op(quant);
    //[Create graph and add ops]

    g.finalize();

    /// After finished above operations, we can get partitions by calling
    /// #dnnl::graph::graph::get_partitions().
    ///
    /// In this example, the graph will be partitioned into one partition.
    ///
    /// @snippet cpu_inference_int8.cpp Get partition
    //[Get partition]
    auto partitions = g.get_partitions();
    //[Get partition]

    // Check partitioning results to ensure the examples works. Users do
    // not need to follow this step.
    assert(partitions.size() == 1);

    /// @page graph_cpu_inference_int8_cpp
    /// @subsection graph_cpu_inference_int8_cpp_compile Compile and Execute Partition
    ///
    /// In the real case, users like framework should provide device information
    /// at this stage. But in this example, we just use a self-defined device to
    /// simulate the real behavior.
    ///
    /// Create a #dnnl::engine. Also, set a user-defined
    /// #dnnl::graph::allocator to this engine.
    ///
    /// @snippet cpu_inference_int8.cpp Create engine
    //[Create engine]
    allocator alloc {};
    dnnl::engine eng
            = make_engine_with_allocator(dnnl::engine::kind::cpu, 0, alloc);
    dnnl::stream strm {eng};
    //[Create engine]

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
    std::unordered_map<size_t, dims> concrete_shapes {
            {0, conv_input_dims}, {3, conv_weight_dims}, {6, conv_bias_dims}};

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
            std::cout << "cpu_inference_int8: Got unsupported partition, users "
                         "need handle the operators by themselves."
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

    // wait for all compiled partition's execution finished
    strm.wait();

    /// @page graph_cpu_inference_int8_cpp
    std::cout << "Graph:" << std::endl
              << " [dq0_src]   [dq1_src]" << std::endl
              << "    |            |" << std::endl
              << " dequant0    dequant1" << std::endl
              << "       \\      /" << std::endl
              << "         conv" << std::endl
              << "          |" << std::endl
              << "         relu" << std::endl
              << "          |" << std::endl
              << "        quant" << std::endl
              << "          |" << std::endl
              << "     [quant_dst]" << std::endl
              << "Note:" << std::endl
              << " '[]' represents a logical tensor, which refers to "
                 "inputs/outputs of the graph. "
              << std::endl;
}

int main(int argc, char **argv) {
    return handle_example_errors({engine::kind::cpu}, simple_pattern_int8);
}
