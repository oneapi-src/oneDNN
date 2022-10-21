/*******************************************************************************
* Copyright 2022 Intel Corporation
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

/// @example cpu_mlp_pattern_dynamic_int8.cpp
/// @copybrief cpu_mlp_pattern_dynamic_int8_cpp
/// > Annotated version: @ref cpu_mlp_pattern_dynamic_int8_cpp

/// @page cpu_mlp_pattern_dynamic_int8_cpp CPU example for a simple int8 dynamic shape pattern

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

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using property_type = logical_tensor::property_type;

// Test mlp dynamic shape pattern different shape compile and execute
// clang-format off
int main(int argc, char **argv) {
    std::cout << "========Example: MLP dynamic shape========\n";

    engine::kind engine_kind = parse_engine_kind(argc, argv);
    if (engine_kind == engine::kind::gpu) {
        std::cout << "Don't support gpu now\n";
        return -1;
    }
    const engine::kind ekind = engine::kind::cpu;

    // Step 2: Construct a graph
    graph g(ekind);

    // dynamic batch size

    /// create logical tensor
    std::cout << "Create logical tensor--------------------------";
    
    const dnnl_graph_dim_t batch_size = -1;
    std::vector<int64_t> input_dims {batch_size, 1024};
    std::vector<int64_t> layer_1_weight_dims {1024, 4096};
    std::vector<int64_t> layer_2_weight_dims {4096, 1024};
    std::vector<int64_t> layer_1_bias_dims {4096};
    std::vector<int64_t> layer_2_bias_dims {1024};

    /// @note It's not necessary to provide concrete shape/layout information
    /// at graph partitioning stage. Users can provide these information till
    /// compilation stage. Dynamic shape needs dense strides.
    ///
    /// create layer_1
    /// per-tensor asymmetric quantized input activation with op dequant_inp
    logical_tensor dequant_input_desc {
            0, data_type::u8, input_dims, layout_type::undef};
    logical_tensor input_desc {
            1, data_type::f32, input_dims, layout_type::undef};
    op dequant_inp(2, op::kind::Dequantize, {dequant_input_desc}, {input_desc},
            "dequant_inp");
    dequant_inp.set_attr<std::string>(op::attr::qtype, "per_tensor");
    dequant_inp.set_attr<std::vector<float>>(op::attr::scales, {0.1f});
    dequant_inp.set_attr<std::vector<int64_t>>(op::attr::zps, {10});

    /// layer_1, per-channel symmetric quantized weight with op dequant_weight_1
    /// weight and bias of layer_1 are constant.
    logical_tensor dequant_layer_1_weight_desc {3, data_type::s8,
            layer_1_weight_dims, layout_type::undef, property_type::constant};
    logical_tensor layer_1_weight_desc {
            4, data_type::f32, layer_1_weight_dims, layout_type::undef};
    op dequant_weight_1(5, op::kind::Dequantize, {dequant_layer_1_weight_desc},
            {layer_1_weight_desc}, "dequant_weight_1");
    dequant_weight_1.set_attr<std::string>(op::attr::qtype, "per_channel");
    std::vector<float> wei_scales_1(4096, 0.1f);
    std::vector<int64_t> wei_zps_1(4096, 0);
    dequant_weight_1.set_attr<std::vector<float>>(
            op::attr::scales, wei_scales_1);
    dequant_weight_1.set_attr<std::vector<int64_t>>(op::attr::zps, wei_zps_1);
    dequant_weight_1.set_attr<int64_t>(op::attr::axis, 1);
    /// layer_1 bias
    logical_tensor layer_1_bias_desc {6, data_type::f32, layer_1_bias_dims,
            layout_type::undef, property_type::constant};
    /// layer_1 output tensor, we even don't know its ndim.
    logical_tensor layer_1_matmul_out_desc {
            7, data_type::f32, -1, layout_type::undef};
    /// layer_1 create op matmul
    op matmul_1(8, op::kind::MatMul,
            {input_desc, layer_1_weight_desc, layer_1_bias_desc},
            {layer_1_matmul_out_desc}, "matmul_1");
    /// layer_1 create op relu
    logical_tensor layer_1_relu_desc {
            9, data_type::f32, -1, layout_type::undef};
    op relu_1(10, op::kind::ReLU, {layer_1_matmul_out_desc},
            {layer_1_relu_desc}, "relu_1");
    /// layer_1 create op quantize
    logical_tensor layer_1_quant_out_desc {
            11, data_type::u8, -1, layout_type::undef};
    op quant_out_1(12, op::kind::Quantize, {layer_1_relu_desc},
            {layer_1_quant_out_desc}, "quant_out_1");
    quant_out_1.set_attr<std::string>(op::attr::qtype, "per_tensor");
    quant_out_1.set_attr<std::vector<float>>(op::attr::scales, {0.1f});
    quant_out_1.set_attr<std::vector<int64_t>>(op::attr::zps, {10});

    /// create layer_2
    logical_tensor layer_2_inp_desc {
            13, data_type::f32, -1, layout_type::undef};
    /// layer_2 create op dequantize, has the same attributes with quant_out_1.
    op dequant_inp_2(14, op::kind::Dequantize, {layer_1_quant_out_desc},
            {layer_2_inp_desc}, "dequant_inp_2");
    dequant_inp_2.set_attr<std::string>(op::attr::qtype, "per_tensor");
    dequant_inp_2.set_attr<std::vector<float>>(op::attr::scales, {0.1f});
    dequant_inp_2.set_attr<std::vector<int64_t>>(op::attr::zps, {10});

    /// layer_2, per-channel symmetric quantized weight with op dequant_weight_2
    /// weight and bias of layer_2 are constant.
    logical_tensor dequant_layer_2_weight_desc {15, data_type::s8,
            layer_2_weight_dims, layout_type::undef, property_type::constant};
    logical_tensor layer_2_weight_desc {
            16, data_type::f32, layer_2_weight_dims, layout_type::undef};

    op dequant_weight_2(17, op::kind::Dequantize, {dequant_layer_2_weight_desc},
            {layer_2_weight_desc}, "dequant_weight_2");
    std::vector<float> wei_scales_2(1024, 0.1f);
    std::vector<int64_t> wei_zps_2(1024, 0);
    dequant_weight_2.set_attr<std::string>(op::attr::qtype, "per_channel");
    dequant_weight_2.set_attr<std::vector<float>>(
            op::attr::scales, wei_scales_2);
    dequant_weight_2.set_attr<std::vector<int64_t>>(op::attr::zps, wei_zps_2);
    dequant_weight_2.set_attr<int64_t>(op::attr::axis, 1);

    logical_tensor layer_2_bias_desc {18, data_type::f32, layer_2_bias_dims,
            layout_type::undef, property_type::constant};
    /// layer_2 matmul output tensor, we even don't know its ndim.
    logical_tensor layer_2_matmul_out_desc {
            19, data_type::f32, -1, layout_type::undef};
    /// layer_2 create op matmul
    op matmul_2(20, op::kind::MatMul,
            {layer_2_inp_desc, layer_2_weight_desc, layer_2_bias_desc},
            {layer_2_matmul_out_desc}, "matmul_2");
    /// layer_2 create op relu
    logical_tensor layer_2_relu_desc {
            21, data_type::f32, -1, layout_type::undef};
    op relu_2(22, op::kind::ReLU, {layer_2_matmul_out_desc},
            {layer_2_relu_desc}, "relu_2");
    /// layer_2 create op quantize
    logical_tensor out_desc {23, data_type::u8, -1, layout_type::undef};
    op quant_out_2(24, op::kind::Quantize, {layer_2_relu_desc}, {out_desc},
            "quant_out_2");
    quant_out_2.set_attr<std::string>(op::attr::qtype, "per_tensor");
    quant_out_2.set_attr<std::vector<float>>(op::attr::scales, {0.1f});
    quant_out_2.set_attr<std::vector<int64_t>>(op::attr::zps, {10});
    std::cout << "Success!\n";

    /// add the operators to the graph
    ///
    /// @note The order of adding op doesn't matter.
    ///
    std::cout << "Add op to graph--------------------------------";
    g.add_op(dequant_inp);
    g.add_op(dequant_weight_1);
    g.add_op(matmul_1);
    g.add_op(relu_1);
    g.add_op(quant_out_1);
    g.add_op(dequant_inp_2);
    g.add_op(dequant_weight_2);
    g.add_op(matmul_2);
    g.add_op(relu_2);
    g.add_op(quant_out_2);
    std::cout << "Success!\n";

    /// dequant0    dequant1
    ///       \      /
    ///        matmul
    ///          |
    ///         relu
    ///          |
    ///        quant
    ///          |
    ///       dequant2   dequant3
    ///          |      /
    ///          matmul
    ///            |
    ///           relu
    ///            |
    ///          quant

    /// The graph will be partitioned into 1 partitions.
    auto partitions = g.get_partitions();

    /// Contains the ids of logical tensors which will be set with any layout
    std::unordered_set<size_t> ids_with_any_layout;
    /// This is a helper function which helps decide which logical tensor is
    /// needed to be set with `dnnl::graph::logical_tensor::layout_type::any`
    /// layout. Typically, users need implement the similar logic in their code
    /// for best performance.
    set_any_layout(partitions, ids_with_any_layout);

    /// construct a new engine and stream
    engine eng {ekind, 0};
    stream strm {eng};

    // mapping from logical tensor id to output tensors
    // used to the connection relationship between partitions (e.g partition 0's
    // output tensor is fed into partition 1)
    std::unordered_map<size_t, tensor> global_outputs_ts_map;
    // manage the lifetime of memory buffers binded to those input/output tensors
    std::vector<std::shared_ptr<void>> data_buffers;

    // mapping from id to queried logical tensor from compiled partition
    // used to record the logical tensors that are previously enabled with ANY layout
    std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;

    if (partitions.size() == 1) { std::cout << "The MLP dynamic shape pattern hits the graph compiler backend.\n"; }
    for (const auto &partition : partitions) {
        if (partition.is_supported()) {
            /// partition compilation begin
            std::vector<logical_tensor> inputs = partition.get_in_ports();
            std::vector<logical_tensor> outputs = partition.get_out_ports();

            // update input logical tensors with concrete layout
            for (size_t idx = 0; idx < inputs.size(); ++idx) {
                size_t id = inputs[idx].get_id();
                // the tensor is an output of another partition
                if (id_to_queried_logical_tensors.find(id)
                        != id_to_queried_logical_tensors.end())
                    inputs[idx] = id_to_queried_logical_tensors[id];
                else {
                    auto ori_lt = inputs[idx];
                    // create logical tensor with strided layout
                    inputs[idx] = logical_tensor {ori_lt.get_id(),
                            ori_lt.get_data_type(), ori_lt.get_dims(),
                            layout_type::strided, ori_lt.get_property_type()};
                }
            }

            // update output logical tensors with concrete layout
            for (size_t idx = 0; idx < outputs.size(); ++idx) {
                size_t id = outputs[idx].get_id();
                layout_type ltype = layout_type::strided;
                if (ids_with_any_layout.count(id)) ltype = layout_type::any;
                auto ori_lt = outputs[idx];
                // create logical tensor with strided/any layout
                outputs[idx] = logical_tensor {ori_lt.get_id(),
                        ori_lt.get_data_type(), -1, ltype,
                        ori_lt.get_property_type()};
            }

            /// Compile the partition to generate compiled partition with the
            /// input and output logical tensors.
            /// In dynamic mode, need only one compilation for multiple execution.
            /// @snippet cpu_get_started.cpp Compile partition
            //[Compile partition]
            std::cout << "Compiling--------------------------------------";
            compiled_partition cp = partition.compile(inputs, outputs, eng);
            std::cout << "Success!\n";
            //[Compile partition]

            // update output logical tensors with queried one
            for (size_t idx = 0; idx < outputs.size(); ++idx) {
                size_t id = outputs[idx].get_id();
                outputs[idx] = cp.query_logical_tensor(id);
                id_to_queried_logical_tensors[id] = outputs[idx];
            }
            /// partition compilation end

            /// partition execution begin
            /// Different real batch size candidates.
            /// Each candidate is executed with the same compiled_parition.
            std::vector<int64_t> real_batch_size_candidates
                    = {1, 3, 7, 16, 32, 63, 64, 128, 129, 512};
            for (auto &cur_bs : real_batch_size_candidates) {
                // Binding data buffers with input and output logical tensors
                // Input and output tensors should have the exact shape during execution(not -1 in compilation).
                std::vector<tensor> inputs_ts, outputs_ts;
                inputs_ts.reserve(inputs.size());
                outputs_ts.reserve(outputs.size());
                std::cout << "Creating tensors and allocating memory buffer--";
                for (const auto &in : inputs) {
                    size_t id = in.get_id();
                    /// fill real shapes during execution.
                    auto exec_dims = in.get_dims();
                    if (exec_dims[0] == -1) { exec_dims[0] = cur_bs; }
                    logical_tensor exec_in {id, in.get_data_type(), exec_dims,
                            in.get_layout_type(), in.get_property_type()};
                    size_t mem_size = exec_in.get_mem_size();
                    // check if the input is an output of another partition
                    auto pos = global_outputs_ts_map.find(id);
                    if (pos != global_outputs_ts_map.end()) {
                        inputs_ts.push_back(pos->second);
                        continue;
                    }
                    // memory allocation
                    data_buffers.push_back({});
                    data_buffers.back().reset(malloc(mem_size), cpu_deletor {});
                    inputs_ts.push_back(
                            tensor {exec_in, eng, data_buffers.back().get()});
                }

                for (const auto &out : outputs) {
                    /// fill real shapes during execution.
                    auto exec_dims = out.get_dims();
                    if (exec_dims[0] == -1) { exec_dims[0] = cur_bs; }
                    logical_tensor exec_out {out.get_id(), out.get_data_type(),
                            exec_dims, out.get_layout_type(),
                            out.get_property_type()};
                    size_t mem_size = exec_out.get_mem_size();
                    // memory allocation
                    data_buffers.push_back({});
                    data_buffers.back().reset(malloc(mem_size), cpu_deletor {});
                    outputs_ts.push_back(
                            tensor {exec_out, eng, data_buffers.back().get()});
                    global_outputs_ts_map[exec_out.get_id()]
                            = outputs_ts.back();
                }
                std::cout << "Success!\n";

                /// Execute the compiled partition 1 on the specified stream.
                /// @snippet cpu_get_started.cpp Execute compiled partition 1
                //[Execute compiled partition]
                std::cout << "Executing compiled partition-------------------";
                cp.execute(strm, inputs_ts, outputs_ts);
                std::cout << "Success!\n";
                //[Execute compiled partition]
            }
        } else {
            std::cout << "cpu_int8_mlp_dynamic_pattern.cpp: got unsupported "
                         "partition, users need handle the operators by "
                         "themselves."
                      << std::endl;
        }
    }
    // wait for all compiled partition's execution finished
    strm.wait();
    std::cout << "Check correctness------------------------------";
    std::cout << "Skipped!\n";

    std::cout << "============Run Example Successfully===========\n";
    return 0;
}
// clang-format on
