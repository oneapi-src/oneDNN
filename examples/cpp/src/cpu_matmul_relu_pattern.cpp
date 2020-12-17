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

/// @example cpu_matmul_relu_pattern.cpp
/// @copybrief cpu_matmul_relu_pattern_cpp
/// Annotated version: @ref cpu_matmul_relu_pattern_cpp

/// @page cpu_matmul_relu_pattern_cpp CPU example for matmul+relu pattern
///
/// > Example code: @ref cpu_matmul_relu_pattern.cpp

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "common/utils.hpp"

using namespace dnnl::graph;

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

    // Step 1: Initialize engine and stream
    /// (todo)xinyu: improve this part when gpu pass is ready
    std::cout << "Initialize CPU engine and stream---------------";
    const int32_t device_id = 0;
    engine eng {engine_kind, device_id};
    stream strm {eng};
    std::cout << "Success!\n";

    // Step 2: Construct a graph
    graph g(engine_kind);

    auto &id_mgr = logical_id_manager::get();

    /// Create OP and set attributes
    std::cout << "Create op---------------------------------";
    /// inuput node
    op wildcard(id_mgr["wildcard"], op::kind::Wildcard, "wildcard");

    /// matmul+relu
    op matmul(id_mgr["matmul"], op::kind::MatMul, "matmul");
    op relu(id_mgr["relu"], op::kind::ReLU, "relu");
    std::cout << "Success!\n";

    /// Create logical tensor
    std::cout << "Create logical tensor--------------------------";

    std::vector<int64_t> input0_dims {1, 64};
    std::vector<int64_t> input1_dims {64, 1};
    std::vector<int64_t> dst_dims {-1, -1};

    logical_tensor matmul_input0_desc {id_mgr["matmul_input0"],
            logical_tensor::data_type::f32, input0_dims,
            logical_tensor::layout_type::undef};
    logical_tensor matmul_input1_desc {id_mgr["matmul_input1"],
            logical_tensor::data_type::f32, input1_dims,
            logical_tensor::layout_type::undef};
    logical_tensor matmul_dst_desc {id_mgr["matmul_dst"],
            logical_tensor::data_type::f32, dst_dims,
            logical_tensor::layout_type::undef};
    logical_tensor relu_dst_desc {id_mgr["relu_dst"],
            logical_tensor::data_type::f32, dst_dims,
            logical_tensor::layout_type::undef};

    std::cout << "Success!\n";

    /// Add Input/Output
    std::cout << "Add logical tensor to op------------------";
    wildcard.add_output(matmul_input0_desc);
    wildcard.add_output(matmul_input1_desc);

    matmul.add_inputs({matmul_input0_desc, matmul_input1_desc});
    matmul.add_output(matmul_dst_desc);
    relu.add_input(matmul_dst_desc);
    relu.add_output(relu_dst_desc);
    std::cout << "Success!\n";

    /// Select OP
    std::cout << "Select op to graph------------------------";
    g.add_op(wildcard);
    g.add_op(matmul);
    g.add_op(relu);
    id_mgr.freeze(); // graph is built up, and the arguments set could be frozen
    std::cout << "Success!\n";

    // Step 3: Filter partitions
    /// Graph will be filtered into 1 partitions: `matmul+relu`
    /// `export DNNL_GRAPH_DUMP=1` can save internal graphs before/after graph fusion into dot files
    std::cout << "Filter partitions------------------------------";
    auto partitions = g.get_partitions(partition::policy::fusion);

    if (partitions.size() != 1) {
        throw std::runtime_error("wrong partition number");
    }
    std::cout << "Success!\n";

    // Step 4: Prepare logical tensors with proper format and compile partitions
    std::cout << "Prepare logical tensors with proper format-----";
    // layout_id::abcd, but format is NXC
    logical_tensor matmul_input0_desc_plain {id_mgr["matmul_input0"],
            logical_tensor::data_type::f32, input0_dims,
            logical_tensor::layout_type::strided};
    logical_tensor matmul_input1_desc_plain {id_mgr["matmul_input1"],
            logical_tensor::data_type::f32, input1_dims,
            logical_tensor::layout_type::strided};
    logical_tensor relu_dst_desc_plain {id_mgr["relu_dst"],
            logical_tensor::data_type::f32, dst_dims,
            logical_tensor::layout_type::strided};
    std::cout << "Success!\n";

    std::cout << "Infer shape----------------------------------";
    std::vector<logical_tensor> in0 {
            matmul_input0_desc_plain, matmul_input1_desc_plain};
    std::vector<logical_tensor> out0 {relu_dst_desc_plain};
    partitions[0].infer_shape(in0, out0);
    std::cout << "Success!\n";
    std::vector<int64_t> infered_relu_dst_dims = out0[0].get_dims();
    std::cout << "Infered_shape: " << infered_relu_dst_dims[0] << ","
              << infered_relu_dst_dims[1] << "\n";

    std::cout << "Compile partition 0----------------------------";
    auto cp0 = partitions[0].compile(in0, out0, eng);
    std::cout << "Success!\n";

    std::cout << "Query layout id from compiled partition 0------";
    relu_dst_desc_plain = cp0.query_logical_tensor(id_mgr["relu_dst"]);
    std::cout << "Success!\n";

    // Step 5: Prepare tensor and submit compiled partitions
    std::cout << "Prepare tensor and submit compiled partitions--";
    std::vector<float> matmul_input0_data(
            static_cast<size_t>(product(input0_dims)), 1.0f);
    std::vector<float> matmul_input1_data(
            static_cast<size_t>(product(input1_dims)), 1.0f);
    std::vector<float> relu_dst_data(
            cp0.query_logical_tensor(id_mgr["relu_dst"]).get_mem_size()
                    / sizeof(float),
            0.0);

    tensor matmul_input0(matmul_input0_desc_plain, matmul_input0_data.data());
    tensor matmul_input1(matmul_input1_desc_plain, matmul_input1_data.data());
    tensor relu_dst(relu_dst_desc_plain, relu_dst_data.data());

    std::vector<tensor> in_list_0 {matmul_input0, matmul_input1};
    std::vector<tensor> out_list_0 {relu_dst};
    cp0.execute(strm, in_list_0, out_list_0);

    std::cout << "Success!\n";

    // Step 6 : Check correctness of the output results
    std::cout << "Check correctness------------------------------";
    if (std::abs(relu_dst_data[0] - 64.0) > 1e-6f) {
        throw std::runtime_error(
                "output result is not equal to excepted "
                "results");
    }
    std::cout << "Success!\n";

    std::cout << "============Run Example Successfully===========\n";

    return 0;
}
