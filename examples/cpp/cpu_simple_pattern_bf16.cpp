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

/// @example cpu_simple_pattern_bf16.cpp
/// @copybrief cpu_simple_pattern_bf16_cpp
/// Annotated version: @ref cpu_simple_pattern_bf16_cpp

/// @page cpu_simple_pattern_bf16_cpp CPU example for a simple bf16 pattern
///
/// > Example code: @ref cpu_simple_pattern_bf16.cpp

#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "dnnl.hpp"

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "common/example_utils.hpp"
#include "common/helpers_any_layout.hpp"

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;

int main(int argc, char **argv) {
    // clang-format off
    const auto isa = dnnl_get_effective_cpu_isa();
    if (isa < dnnl_cpu_isa_avx512_core) {
        std::cout << "cpu_simple_pattern_bf16: skip bf16 examples for systems"
                     "that do not support avx512_core"
                  << std::endl;
        return 0;
    }

    /// create a graph
    const engine::kind ekind = engine::kind::cpu;
    graph g(ekind);

    /// create logical tensor with bf16 data type
    std::vector<int64_t> conv0_src_dims {8, 3, 227, 227};
    std::vector<int64_t> conv0_weight_dims {96, 3, 11, 11};
    std::vector<int64_t> conv0_bias_dims {96};

    /// @note It's not necessary to provide concrete shape/layout information
    /// at graph partitioning stage. Users can provide these information till
    /// compilation stage.
    ///
    logical_tensor conv0_src_desc {
            0, data_type::bf16, conv0_src_dims, layout_type::undef};
    logical_tensor conv0_weight_desc {
            1, data_type::bf16, conv0_weight_dims, layout_type::undef};
    logical_tensor conv0_bias_desc {
            2, data_type::bf16, conv0_bias_dims, layout_type::undef};

    /// don't know the output shape of conv1, let the library to infer
    logical_tensor conv0_dst_desc {3, data_type::bf16, 4, layout_type::undef};

    /// create op conv0
    op conv0(4, op::kind::Convolution, {conv0_src_desc, conv0_weight_desc},
            {conv0_dst_desc}, "conv0");
    conv0.set_attr<std::vector<int64_t>>("strides", {4, 4});
    conv0.set_attr<std::vector<int64_t>>("pads_begin", {111, 111});
    conv0.set_attr<std::vector<int64_t>>("pads_end", {111, 111});
    conv0.set_attr<std::string>("auto_pad", "VALID");
    conv0.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv0.set_attr<std::string>("data_format", "NCX");
    conv0.set_attr<std::string>("filter_format", "OIX");
    conv0.set_attr<int64_t>("groups", 1);

    logical_tensor conv0_bias_add_dst_desc {
            5, data_type::bf16, 4, layout_type::undef};

    op conv0_bias_add(6, op::kind::BiasAdd, {conv0_dst_desc, conv0_bias_desc},
            {conv0_bias_add_dst_desc}, "conv0_bias_add");
    conv0_bias_add.set_attr<std::string>("data_format", "NCX");

    logical_tensor relu0_dst_desc {7, data_type::bf16, 4, layout_type::undef};

    op relu0(8, op::kind::ReLU, {conv0_bias_add_dst_desc}, {relu0_dst_desc},
            "relu0");

    /// create logical tensors for conv1 with bf16 data type
    std::vector<int64_t> conv1_weight_dims {96, 96, 1, 1};
    std::vector<int64_t> conv1_bias_dims {96};

    logical_tensor conv1_weight_desc {
            9, data_type::bf16, conv1_weight_dims, layout_type::undef};
    logical_tensor conv1_bias_desc {
            10, data_type::bf16, conv1_bias_dims, layout_type::undef};
    logical_tensor conv1_dst_desc {11, data_type::bf16, 4, layout_type::undef};

    /// create op conv1
    op conv1(12, op::kind::Convolution, {relu0_dst_desc, conv1_weight_desc},
            {conv1_dst_desc}, "conv1");
    conv1.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv1.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv1.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv1.set_attr<std::string>("data_format", "NCX");
    conv1.set_attr<std::string>("filter_format", "OIX");
    conv1.set_attr<int64_t>("groups", 1);

    logical_tensor conv1_bias_add_dst_desc {
            13, data_type::bf16, 4, layout_type::undef};

    op conv1_bias_add(14, op::kind::BiasAdd, {conv1_dst_desc, conv1_bias_desc},
            {conv1_bias_add_dst_desc}, "conv1_bias_add");
    conv1_bias_add.set_attr<std::string>("data_format", "NCX");

    logical_tensor relu1_dst_desc {15, data_type::bf16, 4, layout_type::undef};

    op relu1(16, op::kind::ReLU, {conv1_bias_add_dst_desc}, {relu1_dst_desc},
            "relu1");

    /// add the ops into the graph
    ///
    /// @note The order of adding op doesn't matter.
    ///
    g.add_op(conv0);
    g.add_op(conv0_bias_add);
    g.add_op(relu0);

    g.add_op(conv1);
    g.add_op(conv1_bias_add);
    g.add_op(relu1);

    /// The graph will be partitioned into two partitions:
    /// - conv0 + conv0_bias_add + relu0
    /// - conv1 + conv1_bias_add + relu1
    auto partitions = g.get_partitions(partition::policy::fusion);

    /// Contains the ids of logical tensors which will be set with any layout
    std::unordered_set<size_t> ids_with_any_layout;
    /// This is a helper function which helps decide which logical tensor is
    /// needed to be set with `dnnl::graph::logical_tensor::layout_type::any`
    /// layout. Typically, users need implement the similar logic in their code
    /// for best performance.
    set_any_layout(partitions, ids_with_any_layout);

    /// create a new engine and stream
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
                    continue;
                }
                // memory allocation
                data_buffers.push_back({});
                data_buffers.back().reset(malloc(mem_size), cpu_deletor {});
                inputs_ts.push_back(
                        tensor {in, eng, data_buffers.back().get()});
            }

            for (const auto &out : outputs) {
                size_t mem_size = out.get_mem_size();
                // memory allocation
                data_buffers.push_back({});
                data_buffers.back().reset(malloc(mem_size), cpu_deletor {});
                outputs_ts.push_back(
                        tensor {out, eng, data_buffers.back().get()});
                global_outputs_ts_map[out.get_id()] = outputs_ts.back();
            }

            /// Execute the compiled partition 1 on the specified stream.
            /// @snippet cpu_get_started.cpp Execute compiled partition 1
            //[Execute compiled partition]
            cp.execute(strm, inputs_ts, outputs_ts);
            //[Execute compiled partition]
        } else {
            std::cout << "cpu_simple_pattern_bf16: got unsupported partition, users need "
                "handle the operators by themselves." << std::endl;
        }
    }
    // wait for all compiled partition's execution finished
    strm.wait();
    // clang-format on

    return 0;
}
