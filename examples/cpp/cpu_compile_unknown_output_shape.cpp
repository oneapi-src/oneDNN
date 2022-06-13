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

/// @example cpu_compile_unknown_output_shape.cpp
/// @copybrief cpu_compile_unknown_output_shape_cpp
/// Annotated version: @ref cpu_compile_unknown_output_shape_cpp

/// @page cpu_compile_unknown_output_shape_cpp CPU example for a simple f32 pattern.
/// This example demonstrates to build a graph without shape information and
/// compile the partitions with only input shapes. During the compilation stage,
/// output shapes will be inferred according to the input shapes and
/// can be queried after the compilation.
/// > Example code: @ref cpu_compile_unknown_output_shape.cpp

#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "common/example_utils.hpp"
#include "common/helpers_any_layout.hpp"

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;

int main(int argc, char **argv) {
    // clang-format off
    const engine::kind ekind = engine::kind::cpu;

    /// create a graph
    graph g(ekind);

    /// @note It's not necessary to provide concrete shape/layout information
    /// at graph partitioning stage. Users can provide these information till
    /// compilation stage.
    /// create input logical tensor with unknown shape
    logical_tensor conv_src_desc {0, data_type::f32, layout_type::undef};
    logical_tensor conv_weight_desc {1, data_type::f32, layout_type::undef};

    /// let the library to calculate the output shape.
    logical_tensor conv_dst_desc {2, data_type::f32, layout_type::undef};

    /// create op conv
    op conv {2, op::kind::Convolution, {conv_src_desc, conv_weight_desc},
            {conv_dst_desc}, "conv"};
    conv.set_attr<std::vector<int64_t>>(op::attr::strides, {1, 1});
    conv.set_attr<std::vector<int64_t>>(op::attr::pads_begin, {0, 0});
    conv.set_attr<std::vector<int64_t>>(op::attr::pads_end, {0, 0});
    conv.set_attr<std::vector<int64_t>>(op::attr::dilations, {1, 1});
    conv.set_attr<std::string>(op::attr::data_format, "NXC");
    conv.set_attr<std::string>(op::attr::filter_format, "OIX");
    conv.set_attr<int64_t>(op::attr::groups, 1);

    /// let the library to calculate the output shape.
    logical_tensor relu_dst_desc {3, data_type::f32, layout_type::undef};

    /// create op relu
    op relu {4, op::kind::ReLU, {conv_dst_desc}, {relu_dst_desc}, "relu"};

    /// add the ops to the graph
    /// @note The order of adding op doesn't matter.
    g.add_op(conv);
    g.add_op(relu);

    /// The graph will be partitioned into 1 partitions: `conv + relu`
    auto partitions = g.get_partitions();

    /// create a new engine and stream
    engine eng {ekind, 0};
    stream strm {eng};

    // mapping from logical tensor id to output tensors
    // used to the connection relationship between partitions (e.g partition 0's
    // output tensor is fed into partition 1)
    std::unordered_map<size_t, tensor> global_outputs_ts_map;
    // manage the lifetime of memory buffers binded to those input/output tensors
    std::vector<std::shared_ptr<void>> data_buffers;

    /// The map from logical tensor to concrete shape, which will be used
    /// re-fill logical tensor during compilation stage.
    std::unordered_map<size_t, std::vector<int64_t> > real_input_shapes_map;
    std::vector<int64_t> conv_input_dims {8, 56, 56, 256}; // NXC
    std::vector<int64_t> conv_weight_dims {64, 256, 1, 1}; // OIX
    real_input_shapes_map[0] = conv_input_dims;
    real_input_shapes_map[1] = conv_weight_dims;

    for (const auto &partition : partitions) {
        if (partition.is_supported()) {
            std::vector<logical_tensor> inputs = partition.get_in_ports();
            std::vector<logical_tensor> outputs = partition.get_out_ports();

            // update input logical tensors with concrete shape and plain layout
            for (size_t idx = 0; idx < inputs.size(); ++idx) {
                auto ori_lt = inputs[idx];
                inputs[idx] = logical_tensor {ori_lt.get_id(),
                        ori_lt.get_data_type(), real_input_shapes_map[ori_lt.get_id()],
                        layout_type::strided};
            }

            // update output logical tensors with unknown shape and plain layout
            for (size_t idx = 0; idx < outputs.size(); ++idx) {
                auto ori_lt = outputs[idx];
                outputs[idx] = logical_tensor {ori_lt.get_id(),
                        ori_lt.get_data_type(), layout_type::strided};
            }

            /// Compile the partition to generate compiled partition with the
            /// input and output logical tensors.
            //[Compile partition]
            compiled_partition cp = partition.compile(inputs, outputs, eng);
            //[Compile partition]

            // update output logical tensors with inferred shape
            for (size_t idx = 0; idx < outputs.size(); ++idx) {
                size_t id = outputs[idx].get_id();
                outputs[idx] = cp.query_logical_tensor(id);
            }

            // Binding data buffers with input and output logical tensors
            std::vector<tensor> inputs_ts, outputs_ts;
            inputs_ts.reserve(inputs.size());
            outputs_ts.reserve(outputs.size());
            for (const auto &in : inputs) {
                size_t id = in.get_id();
                size_t mem_size = in.get_mem_size();
                // check if the input is an output of another partition
                auto pos = global_outputs_ts_map.find(id);
                if (pos != global_outputs_ts_map.end()) {
                    inputs_ts.push_back(pos->second);
                    continue;
                }
                // memory allocation
                data_buffers.push_back({});
                data_buffers.back().reset(malloc(mem_size), cpu_deletor {});
                inputs_ts.emplace_back(
                        tensor {in, eng, data_buffers.back().get()});
            }

            for (const auto &out : outputs) {
                size_t mem_size = out.get_mem_size();
                // memory allocation
                data_buffers.push_back({});
                data_buffers.back().reset(malloc(mem_size), cpu_deletor {});
                outputs_ts.emplace_back(
                        tensor {out, eng, data_buffers.back().get()});
                global_outputs_ts_map[out.get_id()] = outputs_ts.back();
            }

            /// Execute the compiled partition on the specified stream.
            //[Execute compiled partition]
            cp.execute(strm, inputs_ts, outputs_ts);
            //[Execute compiled partition]
        } else {
            std::cout << "cpu_compile_unknown_output_shape: got unsupported partition, users need "
                "handle the operators by themselves." << std::endl;
        }
    }
    // wait for all compiled partition's execution finished
    strm.wait();
    // clang-format on

    return 0;
}
