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

/// @example cpu_simple_pattern_int8.cpp
/// @copybrief cpu_simple_pattern_int8_cpp
/// Annotated version: @ref cpu_simple_pattern_int8_cpp

/// @page cpu_simple_pattern_int8_cpp CPU example for int8 conv + relu pattern
///
/// > Example code: @ref cpu_simple_pattern_int8.cpp

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

    /// create logical tensor
    std::vector<int64_t> conv_input_dims {8, 56, 56, 256}; // NXC
    std::vector<int64_t> conv_weight_dims {1, 1, 256, 64}; // XIO
    std::vector<int64_t> conv_bias_dims {64};

    /// @note It's not necessary to provide concrete shape/layout information
    /// at graph partitioning stage. Users can provide these information till
    /// compilation stage.
    ///
    /// per-tensor asymmetric quantized src activation with op dequant0
    logical_tensor dequant0_src_desc {
            0, data_type::u8, conv_input_dims, layout_type::undef};
    logical_tensor conv_src_desc {
            1, data_type::f32, conv_input_dims, layout_type::undef};
    op dequant0(2, op::kind::Dequantize, {dequant0_src_desc}, {conv_src_desc},
            "dequant0");
    dequant0.set_attr<std::string>(op::attr::qtype, "per_tensor");
    dequant0.set_attr<std::vector<float>>(op::attr::scales, {0.1f});
    dequant0.set_attr<std::vector<int64_t>>(op::attr::zps, {10});

    /// per-channel symmetric quantized weight with op dequant1
    logical_tensor dequant1_src_desc {
            3, data_type::s8, conv_weight_dims, layout_type::undef};
    logical_tensor conv_weight_desc {
            4, data_type::f32, conv_weight_dims, layout_type::undef};
    op dequant1(5, op::kind::Dequantize, {dequant1_src_desc},
            {conv_weight_desc}, "dequant1");
    dequant1.set_attr<std::string>(op::attr::qtype, "per_channel");
    std::vector<float> wei_scales(64, 0.1f);
    std::vector<int64_t> wei_zps(64, 0);
    dequant1.set_attr<std::vector<float>>(op::attr::scales, wei_scales);
    dequant1.set_attr<std::vector<int64_t>>(op::attr::zps, wei_zps);
    dequant1.set_attr<int64_t>(op::attr::axis, 1);

    logical_tensor conv_bias_desc {
            6, data_type::f32, conv_bias_dims, layout_type::undef};
    /// output tensor, we even don't know its ndim.
    logical_tensor conv_dst_desc {7, data_type::f32, -1, layout_type::undef};

    /// create the convolution op
    op conv(8, op::kind::Convolution,
            {conv_src_desc, conv_weight_desc, conv_bias_desc}, {conv_dst_desc},
            "conv");
    conv.set_attr<std::vector<int64_t>>(op::attr::strides, {1, 1});
    conv.set_attr<std::vector<int64_t>>(op::attr::pads_begin, {0, 0});
    conv.set_attr<std::vector<int64_t>>(op::attr::pads_end, {0, 0});
    conv.set_attr<std::vector<int64_t>>(op::attr::dilations, {1, 1});
    conv.set_attr<std::string>(op::attr::data_format, "NXC");
    conv.set_attr<std::string>(op::attr::filter_format, "XIO");
    conv.set_attr<int64_t>(op::attr::groups, 1);

    /// create op relu
    logical_tensor relu_dst_desc {9, data_type::f32, -1, layout_type::undef};
    op relu(10, op::kind::ReLU, {conv_dst_desc}, {relu_dst_desc}, "relu");

    /// create op quant
    logical_tensor quant_dst_desc {11, data_type::u8, -1, layout_type::undef};
    op quant(
            12, op::kind::Quantize, {relu_dst_desc}, {quant_dst_desc}, "quant");
    quant.set_attr<std::string>(op::attr::qtype, "per_tensor");
    quant.set_attr<std::vector<float>>(op::attr::scales, {0.1f});
    quant.set_attr<std::vector<int64_t>>(op::attr::zps, {10});

    /// add the operators to the graph
    ///
    /// @note The order of adding op doesn't matter.
    ///
    g.add_op(dequant0);
    g.add_op(dequant1);
    g.add_op(conv);
    g.add_op(relu);
    g.add_op(quant);

    /// dequant0    dequant1
    ///       \      /
    ///         conv
    ///          |
    ///         relu
    ///          |
    ///         quant

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
                outputs[idx] = logical_tensor {
                        ori_lt.get_id(), ori_lt.get_data_type(), -1, ltype};
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
            std::cout << "cpu_simple_pattern_int8: got unsupported partition, users need "
                "handle the operators by themselves." << std::endl;
        }
    }
    // wait for all compiled partition's execution finished
    strm.wait();
    // clang-format on

    return 0;
}
