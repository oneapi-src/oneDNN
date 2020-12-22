/*******************************************************************************
* Copyright 2020 Intel Corporation
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

/// @example cpu_conversion_simple_pattern.cpp
/// @copybrief cpu_conversion_simple_pattern_cpp
/// Annotated version: @ref cpu_conversion_simple_pattern_cpp

/// @page cpu_conversion_simple_pattern_cpp CPU example for data conversion.
///
/// > Example code: @ref cpu_conversion_simple_pattern.cpp
///
/// This example aims to demonstrate the usage of data conversion API.
/// If a tensor has a different layout with backend preferred layout, users can
/// pre-allocate memory buffer and use this API to convert layout. If possible,
/// users can also cache this memory buffer for futher reuse and hence improve
/// performance. Besides that, this API also supports data conversion between
/// different data types, like `fp32<->bf16/fp16`.

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/utils.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

using namespace dnnl::graph;

// digraph G {
// Wildcard_100002 -> Convolution_100003;
// Convolution_100003 -> ReLU_100005;
// }

// Test conv relu different shape compile and execute
int cpu_conversion_simple_pattern_tutorial(engine::kind engine_kind) {
    std::cout << "========Example: Conv+ReLU========\n";

    if (engine_kind == engine::kind::gpu) {
        std::cout << "Don't support gpu now\n";
        return -1;
    }

    // Step 1: Initialize engine and stream
    // (todo)xinyu: improve this part when gpu pass is ready
    std::cout << "Initialize CPU engine and stream---------------";
    /// Create an `engine` according to `engine_kind` and `device_id`. Then,
    /// create a `stream` based on this engine
    /// @snippet cpu_conversion_simple_pattern.cpp Create engine and stream
    //[Create engine and stream]
    const int32_t device_id = 0;
    engine eng {engine_kind, device_id};
    stream strm {eng};
    //[Create engine and stream]
    std::cout << "Success!\n";

    // Step 2: Construct a graph
    /// Constructs a graph for further analysis
    /// @snippet cpu_conversion_simple_pattern.cpp Create graph
    //[Create graph]
    graph g(engine_kind);
    //[Create graph]

    auto &id_mgr = logical_id_manager::get();

    // Create OP and set attributes
    std::cout << "Create op--------------------------------------";
    // inuput node
    op input0(id_mgr["input0"], op::kind::Wildcard, "input0");

    // conv+relu
    /// Create Ops and set attributes to them. In this example, we need to
    /// create `conv` Op and `relu` Op.
    /// @snippet cpu_conversion_simple_pattern.cpp Create op
    //[Create op]
    op conv0(id_mgr["conv0"], op::kind::Convolution, "conv0");
    conv0.set_attr("strides", std::vector<int64_t> {1, 1});
    conv0.set_attr("pads_begin", std::vector<int64_t> {0, 0});
    conv0.set_attr("pads_end", std::vector<int64_t> {0, 0});
    conv0.set_attr("dilations", std::vector<int64_t> {1, 1});
    conv0.set_attr("data_format", std::string("NXC"));
    conv0.set_attr("filter_format", std::string("OIX"));
    conv0.set_attr("groups", int64_t {1});
    op relu0(id_mgr["relu0"], op::kind::ReLU, "relu0");
    //[Create op]
    std::cout << "Success!\n";

    // Create logical tensor
    std::cout << "Create logical tensor--------------------------";

    std::vector<int64_t> input_dims {8, 56, 56, 256}; // NXC
    std::vector<int64_t> conv0_weight_dims {64, 256, 1, 1}; // OIX
    std::vector<int64_t> conv0_dst_dims {-1, -1, -1, -1};

    /// Creates logical tensors for those inputs and outputs of the OPs, which
    /// will be used to define the connection relationship between OPs.
    /// @snippet cpu_conversion_simple_pattern.cpp Create logical tensor
    //[Create logical tensor]
    logical_tensor conv0_src_desc {id_mgr["conv0_src"],
            logical_tensor::data_type::f32, input_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv0_weight_desc {id_mgr["conv0_weight"],
            logical_tensor::data_type::f32, conv0_weight_dims,
            logical_tensor::layout_type::undef};
    logical_tensor conv0_dst_desc {id_mgr["conv0_dst"],
            logical_tensor::data_type::f32, conv0_dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor relu0_dst_desc {id_mgr["relu0_dst"],
            logical_tensor::data_type::f32, conv0_dst_dims,
            logical_tensor::layout_type::undef};
    //[Create logical tensor]
    std::cout << "Success!\n";

    // Add Input/Output
    std::cout << "Add logical tensor to op-----------------------";
    input0.add_output(conv0_src_desc);

    /// Adds input and output logical tensors to the corresponding OPs.
    /// @snippet cpu_conversion_simple_pattern.cpp Add input and output
    //[Add input and output]
    conv0.add_inputs({conv0_src_desc, conv0_weight_desc});
    conv0.add_output(conv0_dst_desc);
    relu0.add_input(conv0_dst_desc);
    relu0.add_output(relu0_dst_desc);
    //[Add input and output]
    std::cout << "Success!\n";

    /// Select OP
    std::cout << "Select op to graph-----------------------------";
    /// Adds all of OPs into the created graph. Graph inside will maintain a
    /// list to store all these OPs.
    /// @snippet cpu_conversion_simple_pattern.cpp Add op
    //[Add op]
    g.add_op(input0);
    g.add_op(conv0);
    g.add_op(relu0);
    //[Add op]
    id_mgr.freeze(); // graph is built up, and the arguments set could be frozen
    std::cout << "Success!\n";

    std::cout << "Filter partitions------------------------------";
    /// Builds graph based on the selected Ops and gets partitions according to
    /// the patition policy and pre-defined pattern list.
    /// In this example, the graph will be filtered into a single partition:
    /// `conv+relu`.
    ///
    /// @note
    ///     Setting `DNNL_GRAPH_DUMP=1` can save internal graphs into dot files
    ///     before/after graph fusion.
    ///
    /// @snippet cpu_conversion_simple_pattern.cpp Get partition
    //[Get partition]
    auto partitions = g.get_partitions(partition::policy::fusion);
    //[Get partition]

    if (partitions.size() != 1) {
        throw std::runtime_error("wrong partition number");
    }
    std::cout << "Success!\n";

    // Step 4: Prepare logical tensors with proper format and compile partitions
    std::cout << "Prepare logical tensors with proper format-----";
    // layout_id::abcd, but format is NXC
    /// Sets proper format to the logical tensors for inputs/outputs of
    /// this partition.
    ///
    /// @note
    ///    In this partition, input data has plain layout while weights and
    ///    output have `any` layout. For `any` layout, backend will determine
    ///    the best opaque layout.
    ///
    /// @snippet cpu_conversion_simple_pattern.cpp Set format for logical tensors
    //[Set format for logical tensors]
    logical_tensor conv0_src_desc_plain {id_mgr["conv0_src"],
            logical_tensor::data_type::f32, input_dims,
            logical_tensor::layout_type::strided};
    logical_tensor conv0_weight_desc_any {id_mgr["conv0_weight"],
            logical_tensor::data_type::f32, conv0_weight_dims,
            logical_tensor::layout_type::any};
    logical_tensor relu0_dst_desc_any {id_mgr["relu0_dst"],
            logical_tensor::data_type::f32, conv0_dst_dims,
            logical_tensor::layout_type::any};
    //[Set format for logical tensors]
    std::cout << "Success!\n";

    std::cout << "Infer shape------------------------------------";
    std::vector<logical_tensor> in0 {
            conv0_src_desc_plain, conv0_weight_desc_any};
    std::vector<logical_tensor> out0 {relu0_dst_desc_any};
    /// Infers the shape of output from the partition.
    /// @snippet cpu_conversion_simple_pattern.cpp Infer shape
    //[Infer shape]
    partitions[0].infer_shape(in0, out0);
    //[Infer shape]
    std::cout << "Success!\n";
    std::vector<int64_t> infered_relu0_dst_dims = out0[0].get_dims();
    std::cout << "Infered_shape: " << infered_relu0_dst_dims[0] << ","
              << infered_relu0_dst_dims[1] << "," << infered_relu0_dst_dims[2]
              << "," << infered_relu0_dst_dims[3] << "\n";

    std::cout << "Compile partition 0----------------------------";
    /// Compile the partition to generate compiled partition based on the
    /// input and output logical tensors.
    /// @snippet cpu_conversion_simple_pattern.cpp Compile partition
    //[Compile partition]
    auto cp0 = partitions[0].compile(in0, out0, eng);
    //[Compile partition]
    std::cout << "Success!\n";

    std::cout << "Query layout id from compiled partition 0------";
    relu0_dst_desc_any = cp0.query_logical_tensor(id_mgr["relu0_dst"]);
    std::cout << "Success!\n";

    // Step 5: Prepare tensor
    std::cout << "Prepare tensors (data conversion) -------------";
    std::vector<float> conv0_src_data(
            static_cast<size_t>(product(input_dims)), 1.0f);
    std::vector<float> conv0_weight_data(
            static_cast<size_t>(product(conv0_weight_dims)), 1.0f);
    std::vector<float> relu0_dst_data(
            relu0_dst_desc_any.get_mem_size() / sizeof(float), 0.0);

    tensor conv0_src(conv0_src_desc_plain, conv0_src_data.data());
    tensor relu0_dst(relu0_dst_desc_any, relu0_dst_data.data());

    /// Prepare input weight tensor. Firstly, we may create a logical tensor
    /// with `strided` layout. And then, create a tensor associated with this
    /// logical tensor and real data buffer.
    /// @snippet cpu_conversion_simple_pattern.cpp Create tensor
    //[Create tensor]
    logical_tensor conv0_weight_desc_plain {id_mgr["conv0_weight"],
            logical_tensor::data_type::f32, conv0_weight_dims,
            logical_tensor::layout_type::strided};
    tensor conv0_weight(conv0_weight_desc_plain, conv0_weight_data.data());
    //[Create tensor]

    /// Convert data to the opaque layout which is determined by the backend.
    /// Here, we can query the logical tensor of `weight` tensor. And then,
    /// compare the logical tensor created by users with the queried from
    /// compiled partition. If the layouts are different, users can pre-allocate
    /// memory buffer and use converison API to convert `weight` tensor to the
    /// best layout.
    /// @snippet cpu_conversion_simple_pattern.cpp Convert data
    //[Convert data]
    logical_tensor conv0_weight_desc_queried
            = cp0.query_logical_tensor(id_mgr["conv0_weight"]);
    void *buffer = nullptr;
    if (!conv0_weight_desc_queried.has_same_layout_and_dtype(
                conv0_weight_desc_plain)) {
        buffer = allocate(conv0_weight_desc_queried.get_mem_size(),
                allocator::attribute());
        // create a conversion partition
        dnnl::graph::conversion convert {};
        // compile to compiled partition
        compiled_partition convert_executable = convert.compile(
                conv0_weight_desc_plain, conv0_weight_desc_queried, eng);
        // real tensor with queried layout
        tensor conv0_weight_r {conv0_weight_desc_queried, buffer};
        // execute the conversion
        convert_executable.execute(strm, {conv0_weight}, {conv0_weight_r});
        // release conv0_weight's buffer, and update with new tensor
        conv0_weight = conv0_weight_r;
    }
    //[Convert data]

    std::cout << "Success!\n";

    // Step 6: Submit compiled partitions
    std::cout << "Submit compiled partitions --------------------";
    std::vector<tensor> in_list_0 {conv0_src, conv0_weight};
    std::vector<tensor> out_list_0 {relu0_dst};
    /// Executes the compiled partition on the specified stream.
    /// @snippet cpu_conversion_simple_pattern.cpp Execute compiled partition
    //[Execute compiled partition]
    cp0.execute(strm, in_list_0, out_list_0);
    //[Execute compiled partition]

    std::cout << "Success!\n";

    //     Step 6 : Check correctness of the output results
    std::cout << "Check correctness------------------------------";
    for (auto v : relu0_dst_data) {
        if (std::abs(v - 256.0) > 1e-6f) {
            if (buffer) deallocate(buffer);
            throw std::runtime_error(
                    "output result is not equal to excepted "
                    "results");
        }
    }
    std::cout << "Success!\n";

    std::cout << "============Run Example Successfully===========\n";

    if (buffer) deallocate(buffer);

    /// @page cpu_conversion_simple_pattern_cpp
    return 0;
}

int main(int argc, char **argv) {
    engine::kind engine_kind = parse_engine_kind(argc, argv);
    return cpu_conversion_simple_pattern_tutorial(engine_kind);
}
