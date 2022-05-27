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

/// @example cpu_cnn_training_f32.cpp
/// @copybrief cpu_cnn_training_f32_cpp
/// Annotated version: @ref cpu_cnn_training_f32_cpp

/// @page cpu_cnn_training_f32_cpp CPU example for a simple f32 pattern
///
/// > Example code: @ref cpu_cnn_training_f32.cpp

#include <math.h>
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
    const engine::kind ekind = engine::kind::cpu;

    /// create a graph
    graph g(ekind);

    const int batch = 32;

    /// create logical tensors
    logical_tensor::dims_t net_src_dims {batch, 3, 224, 224};
    logical_tensor::dims_t net_params_dims {96, 3, 5, 5};
    logical_tensor::dims_t net_dst_dims {batch, 96, 37, 37};

    /// @note It's not necessary to provide concrete shape/layout information
    /// at graph partitioning stage. Users can provide these information till
    /// compilation stage.
    ///
    logical_tensor conv_src_lt {
            0, data_type::f32, net_src_dims, layout_type::undef};
    logical_tensor conv_filters_lt {
            1, data_type::f32, net_params_dims, layout_type::undef};
    logical_tensor conv_dst_lt {2, data_type::f32, 4, layout_type::undef};

    // create op Convolution
    // {batch, 3, 224, 224} (x) {96, 3, 5, 5} -> {batch, 96, 74, 74}
    // strides: {3, 3}
    op conv {0, op::kind::Convolution, {conv_src_lt, conv_filters_lt},
            {conv_dst_lt}, "conv"};
    conv.set_attr<std::vector<int64_t>>(op::attr::strides, {3, 3});
    conv.set_attr<std::vector<int64_t>>(op::attr::pads_begin, {0, 0});
    conv.set_attr<std::vector<int64_t>>(op::attr::pads_end, {0, 0});
    conv.set_attr<std::vector<int64_t>>(op::attr::dilations, {1, 1});
    conv.set_attr<std::string>(op::attr::data_format, "NCX");
    conv.set_attr<std::string>(op::attr::filter_format, "OIX");
    conv.set_attr<int64_t>(op::attr::groups, 1);

    /// create op ReLU
    // {batch, 96, 74, 74} -> {batch, 96, 74, 74}
    logical_tensor relu_dst_lt {3, data_type::f32, 4, layout_type::undef};
    op relu {1, op::kind::ReLU, {conv_dst_lt}, {relu_dst_lt}, "relu"};

    /// create op MaxPool
    // {batch, 96, 74, 74} -> {batch, 96, 37, 37}
    logical_tensor pool_dst_lt {4, data_type::f32, 4, layout_type::undef};
    op pool {2, op::kind::MaxPool, {relu_dst_lt}, {pool_dst_lt}, "pool"};
    pool.set_attr<std::vector<int64_t>>(op::attr::kernel, {2, 2});
    pool.set_attr<std::vector<int64_t>>(op::attr::strides, {2, 2});
    pool.set_attr<std::vector<int64_t>>(op::attr::dilations, {1, 1});
    pool.set_attr<std::vector<int64_t>>(op::attr::pads_begin, {0, 0});
    pool.set_attr<std::vector<int64_t>>(op::attr::pads_end, {0, 0});
    pool.set_attr<std::string>(op::attr::data_format, "NCX");

    /// create op MaxPoolBackprop
    // {batch, 96, 37, 37} -> {batch, 96, 74, 74}
    logical_tensor pool_diff_dst_lt {
            5, data_type::f32, net_dst_dims, layout_type::undef};
    logical_tensor pool_diff_src_lt {6, data_type::f32, 4, layout_type::undef};

    op pool_bwd {3, op::kind::MaxPoolBackprop, {relu_dst_lt, pool_diff_dst_lt},
            {pool_diff_src_lt}, "pool_bwd"};
    pool_bwd.set_attr<std::vector<int64_t>>(op::attr::kernel, {2, 2});
    pool_bwd.set_attr<std::vector<int64_t>>(op::attr::strides, {2, 2});
    pool_bwd.set_attr<std::vector<int64_t>>(op::attr::dilations, {1, 1});
    pool_bwd.set_attr<std::vector<int64_t>>(op::attr::pads_begin, {0, 0});
    pool_bwd.set_attr<std::vector<int64_t>>(op::attr::pads_end, {0, 0});
    pool_bwd.set_attr<std::string>(op::attr::data_format, "NCX");

    /// create op ReLUBackprop
    // {batch, 96, 74, 74} -> {batch, 96, 74, 74}
    logical_tensor relu_diff_src_lt {7, data_type::f32, 4, layout_type::undef};
    op relu_bwd {4, op::kind::ReLUBackprop, {relu_dst_lt, pool_diff_src_lt},
            {relu_diff_src_lt}, "relu_bwd"};
    relu_bwd.set_attr<bool>(op::attr::use_dst, true);

    // create op ConvolutionBackpropData
    // {batch, 96, 74, 74} (x) {96, 3, 5, 5} -> {batch, 3, 224, 224}
    logical_tensor conv_diff_src_lt {8, data_type::f32, 4, layout_type::undef};
    op conv_bwd_data {5, op::kind::ConvolutionBackpropData,
            {relu_diff_src_lt, conv_filters_lt}, {conv_diff_src_lt},
            "conv_bwd_data"};
    conv_bwd_data.set_attr<std::vector<int64_t>>(op::attr::strides, {3, 3});
    conv_bwd_data.set_attr<std::vector<int64_t>>(op::attr::dilations, {1, 1});
    conv_bwd_data.set_attr<std::vector<int64_t>>(op::attr::pads_begin, {0, 0});
    conv_bwd_data.set_attr<std::vector<int64_t>>(op::attr::pads_end, {0, 0});
    conv_bwd_data.set_attr<std::string>(op::attr::data_format, "NCX");
    conv_bwd_data.set_attr<std::string>(op::attr::filter_format, "OIX");
    conv_bwd_data.set_attr<std::vector<int64_t>>(
            op::attr::output_shape, net_src_dims);

    // create op ConvolutionBackpropFilters
    // {batch, 96, 74, 74} (x) {batch, 3, 224, 224} -> {96, 3, 5, 5}
    logical_tensor conv_diff_filters_lt {
            9, data_type::f32, 4, layout_type::undef};
    op conv_bwd_filters {6, op::kind::ConvolutionBackpropFilters,
            {conv_src_lt, relu_diff_src_lt}, {conv_diff_filters_lt},
            "conv_bwd_filters"};
    conv_bwd_filters.set_attr<std::vector<int64_t>>(op::attr::strides, {3, 3});
    conv_bwd_filters.set_attr<std::vector<int64_t>>(
            op::attr::dilations, {1, 1});
    conv_bwd_filters.set_attr<std::vector<int64_t>>(
            op::attr::pads_begin, {0, 0});
    conv_bwd_filters.set_attr<std::vector<int64_t>>(op::attr::pads_end, {0, 0});
    conv_bwd_filters.set_attr<std::string>(op::attr::data_format, "NCX");
    conv_bwd_filters.set_attr<std::string>(op::attr::filter_format, "OIX");
    conv_bwd_filters.set_attr<std::vector<int64_t>>(
            op::attr::filter_shape, net_params_dims);

    /// add the ops to the graph
    ///
    /// @note The order of adding op doesn't matter.
    ///
    g.add_op(conv);
    g.add_op(relu);
    g.add_op(pool);
    g.add_op(pool_bwd);
    g.add_op(relu_bwd);
    g.add_op(conv_bwd_data);
    g.add_op(conv_bwd_filters);

    /// The graph will be partitioned into 6 partitions
    auto partitions = g.get_partitions();

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
            /// @snippet cpu_cnn_training_f32.cpp Compile partition
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
            /// @snippet cpu_cnn_training_f32.cpp Execute compiled partition 1
            //[Execute compiled partition]
            cp.execute(strm, inputs_ts, outputs_ts);
            //[Execute compiled partition]
        } else {
            std::cout << "cpu_cnn_training_f32: got unsupported partition, "
                         "users need "
                         "handle the operators by themselves."
                      << std::endl;
        }
    }
    // wait for all compiled partition's execution finished
    strm.wait();

    return 0;
}
