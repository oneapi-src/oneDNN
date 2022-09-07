/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

/// @example cpu_inplace_ports.cpp
/// @copybrief cpu_inplace_ports_cpp
/// Annotated version: @ref cpu_inplace_ports_cpp

/// @page cpu_inplace_ports_cpp CPU example supports inplace computation
///
/// > Example code: @ref cpu_inplace_ports.cpp

#include <algorithm>
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

    std::vector<int64_t> conv_input_dims = {8, 56, 56, 256};
    std::vector<int64_t> conv_weight_dims = {1, 1, 256, 64};
    std::vector<int64_t> conv_bias_dims = {64};
    std::vector<int64_t> conv_dst_dims = {8, 56, 56, 64};

    /// create logical tensors for op conv
    /// @note It's not necessary to provide concrete shape/layout information
    /// at graph partitioning stage. Users can provide these information till
    /// compilation stage.
    ///
    logical_tensor conv_src_desc {
            0, data_type::f32, conv_input_dims, layout_type::undef};
    logical_tensor conv_weight_desc {
            1, data_type::f32, conv_weight_dims, layout_type::undef};
    logical_tensor conv_bias_desc {
            2, data_type::f32, conv_bias_dims, layout_type::undef};
    logical_tensor conv_dst_desc {
            3, data_type::f32, conv_dst_dims, layout_type::undef};

    /// create op conv
    op conv(0, op::kind::Convolution, {conv_src_desc, conv_weight_desc},
            {conv_dst_desc}, "conv");
    conv.set_attr<std::vector<int64_t>>(op::attr::strides, {1, 1});
    conv.set_attr<std::vector<int64_t>>(op::attr::pads_begin, {0, 0});
    conv.set_attr<std::vector<int64_t>>(op::attr::pads_end, {0, 0});
    conv.set_attr<std::vector<int64_t>>(op::attr::dilations, {1, 1});
    conv.set_attr<std::string>(op::attr::data_format, "NXC");
    conv.set_attr<std::string>(op::attr::filter_format, "XIO");
    conv.set_attr<int64_t>(op::attr::groups, 1);

    /// create output logical tensor for op bias_add
    logical_tensor conv_bias_dst_desc {
            4, data_type::f32, conv_dst_dims, layout_type::undef};

    /// create op bias_add
    op bias(1, op::kind::BiasAdd, {conv_dst_desc, conv_bias_desc},
            {conv_bias_dst_desc}, "bias");
    bias.set_attr<std::string>(op::attr::data_format, "NXC");

    /// create another input logical tensor for op add
    logical_tensor add_src1_desc {
            5, data_type::f32, conv_dst_dims, layout_type::undef};

    /// create output logical tensor for op add
    logical_tensor add_dst_desc {
            6, data_type::f32, conv_dst_dims, layout_type::undef};

    /// create op add
    op add(2, op::kind::Add, {conv_bias_dst_desc, add_src1_desc},
            {add_dst_desc}, "add");

    /// create output logical tensor for op abs
    logical_tensor abs_dst_desc {
            7, data_type::f32, conv_dst_dims, layout_type::undef};

    /// create op abs
    op abs(3, op::kind::Abs, {add_dst_desc}, {abs_dst_desc}, "abs0");

    /// add above operators to the graph
    ///
    /// @note The order of adding op doesn't matter.
    ///
    g.add_op(conv);
    g.add_op(bias);
    g.add_op(add);
    g.add_op(abs);

    /// conv
    ///   |
    /// bias       src1
    ///     \       /
    ///        add
    ///         |
    ///        abs
    ///         |

    /// the graph will be partitioned into 1 partitions:
    ///   - conv + bias + add + abs
    auto partitions = g.get_partitions();

    /// Contains the ids of logical tensors which will be set with any layout
    std::unordered_set<size_t> ids_with_any_layout;
    /// This is a helper function which helps decide which logical tensor is
    /// needed to be set with `dnnl::graph::logical_tensor::layout_type::any`
    /// layout. Typically, users need implement the similar logic in their code
    /// for best performance.
    set_any_layout(partitions, ids_with_any_layout);

    /// create engine, allocator, and stream
    engine eng {ekind, 0};
    stream strm {eng};

    // mapping from logical tensor id to input/output tensors
    // used to the connection relationship between partitions (e.g partition 0's
    // output tensor is fed into partition 1)
    std::unordered_map<size_t, tensor> global_outputs_ts_map,
            global_inputs_ts_map;
    // manage the lifetime of memory buffers binded to those input/output tensors
    std::vector<std::shared_ptr<void>> data_buffers;

    // mapping from id to queried logical tensor from compiled partition
    // used to record the logical tensors that are previously enabled with ANY layout
    std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;

    // initial input data
    std::vector<float> conv_src_desc_data(product(conv_input_dims), 1.f);
    std::vector<float> conv_weight_desc_data(product(conv_weight_dims), 1.f);
    std::vector<float> conv_bias_desc_data(product(conv_bias_dims), 1.f);
    std::vector<float> add_src1_desc_data(product(conv_dst_dims), 1.f);

    // the first partition input can be think as the dummy partition output. so
    // we can use global_outputs_ts_map to initialize the input data 
    global_outputs_ts_map[0] = tensor {conv_src_desc, eng, conv_src_desc_data.data()};
    global_outputs_ts_map[1] = tensor {conv_weight_desc, eng, conv_weight_desc_data.data()};
    global_outputs_ts_map[2] = tensor {conv_bias_desc, eng, conv_bias_desc_data.data()};
    global_outputs_ts_map[5] = tensor {add_src1_desc, eng, add_src1_desc_data.data()};

    for (const auto &partition : partitions) {
        if (partition.is_supported()) {
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
                            layout_type::strided};
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
                        ori_lt.get_data_type(), ori_lt.get_dims(), ltype};
            }

            /// Compile the partition to generate compiled partition with the
            /// input and output logical tensors.
            /// @snippet cpu_get_started.cpp Compile partition
            //[Compile partition]
            compiled_partition cp = partition.compile(inputs, outputs, eng);
            //[Compile partition]

            // update output logical tensors with queried one
            for (size_t idx = 0; idx < outputs.size(); ++idx) {
                size_t id = outputs[idx].get_id();
                outputs[idx] = cp.query_logical_tensor(id);
                id_to_queried_logical_tensors[id] = outputs[idx];
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
                }
                else{
                    // memory allocation
                    data_buffers.push_back({});
                    data_buffers.back().reset(malloc(mem_size), cpu_deletor {});
                    inputs_ts.push_back(tensor {in, eng, data_buffers.back().get()});
                }
                global_inputs_ts_map[id] = inputs_ts.back();
            }
            // In this example,the in-place pairs are {{5,7}}
            // It indicates that add_src1_desc and abs_dst_desc 
            // can share the same memory buffer for computation
            auto inplace_ports = cp.get_inplace_ports();
            for (const auto &out : outputs) {
                size_t id = out.get_id();
                size_t mem_size = out.get_mem_size();
                // check inplace pairs
                auto pos = std::find_if(inplace_ports.begin(),
                        inplace_ports.end(),
                        [&id](std::pair<size_t, size_t> &p) {
                            return id == p.second;
                        });
                if (pos != inplace_ports.end()) {
                    // found the inplace port, directly set input buffer to this
                    // output tensor
                    auto in_buffer = global_inputs_ts_map[pos->first]
                                             .get_data_handle();
                    outputs_ts.push_back(tensor {out, eng, in_buffer});
                }
                else{
                    // memory allocation
                    data_buffers.push_back({});
                    data_buffers.back().reset(malloc(mem_size), cpu_deletor {});
                    outputs_ts.push_back(tensor {out, eng, data_buffers.back().get()});
                }
                global_outputs_ts_map[out.get_id()] = outputs_ts.back();
            }

            /// Execute the compiled partition 1 on the specified stream.
            /// @snippet cpu_get_started.cpp Execute compiled partition 1
            //[Execute compiled partition]
            cp.execute(strm, inputs_ts, outputs_ts);
            //[Execute compiled partition]
        } else {
            std::cout << "cpu_inplace_ports: got unsupported partition, users need "
                "handle the operators by themselves." << std::endl;
        }
    }
    // wait for all compiled partition's execution finished
    strm.wait();
    float expected_result = 258.f;
    float *actual_output_ptr = (float*)global_outputs_ts_map[abs_dst_desc.get_id()]
                                       .get_data_handle();
    auto output_dims = abs_dst_desc.get_dims();
    auto num_elem = product(output_dims);
    std::vector<float> expected_output(num_elem, expected_result);
    std::vector<float> actual_output {
            actual_output_ptr, actual_output_ptr + num_elem};
    if (expected_output != actual_output)
        throw std::runtime_error("failed to check accuracy");
    // clang-format on
    return 0;
}
