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

/// @example gpu_simple_pattern_fp16.cpp
/// @copybrief gpu_simple_pattern_fp16_cpp
/// Annotated version: @ref gpu_simple_pattern_fp16_cpp

/// @page gpu_simple_pattern_fp16_cpp SYCL GPU example for conv+relu+conv+relu pattern
///
/// > Example code: @ref gpu_simple_pattern_fp16.cpp

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"

#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "common/example_utils.hpp"
#include "common/helpers_any_layout.hpp"

using namespace dnnl::graph;
using namespace cl::sycl;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;

int main(int argc, char **argv) {
    // clang-format off
    /// create a graph with a gpu device kind
    const engine::kind ekind = engine::kind::gpu;
    graph g(ekind);

    std::vector<int64_t> conv0_input_dims {8, 3, 227, 227};
    std::vector<int64_t> conv0_weight_dims {96, 3, 11, 11};
    std::vector<int64_t> conv0_bias_dims {96};
    std::vector<int64_t> conv1_weight_dims {96, 96, 1, 1};
    std::vector<int64_t> conv1_bias_dims {96};

    /// create logical tensors with f16 data type for conv0
    ///
    /// @note It's not necessary to provide concrete shape/layout information
    /// at graph partitioning stage. Users can provide these information till
    /// compilation stage.
    ///
    logical_tensor conv0_src_desc {
            0, data_type::f16, conv0_input_dims, layout_type::undef};
    logical_tensor conv0_weight_desc {
            1, data_type::f16, conv0_weight_dims, layout_type::undef};

    logical_tensor conv0_dst_desc {2, data_type::f16, 4, layout_type::undef};

    /// create conv0 operator
    op conv0 {3, op::kind::Convolution, {conv0_src_desc, conv0_weight_desc},
            {conv0_dst_desc}, "conv0"};
    conv0.set_attr<std::vector<int64_t>>(op::attr::strides, {4, 4});
    conv0.set_attr<std::vector<int64_t>>(op::attr::pads_begin, {0, 0});
    conv0.set_attr<std::vector<int64_t>>(op::attr::pads_end, {0, 0});
    conv0.set_attr<std::vector<int64_t>>(op::attr::dilations, {1, 1});
    conv0.set_attr<int64_t>(op::attr::groups, 1);
    conv0.set_attr<std::string>(op::attr::data_format, "NCX");
    conv0.set_attr<std::string>(op::attr::filter_format, "OIX");

    logical_tensor conv0_bias_desc {
            4, data_type::f16, conv0_bias_dims, layout_type::undef};

    logical_tensor conv0_bias_add_dst_desc {
            5, data_type::f16, 4, layout_type::undef};

    /// create conv0_bias_add
    op conv0_bias_add {6, op::kind::BiasAdd, {conv0_dst_desc, conv0_bias_desc},
            {conv0_bias_add_dst_desc}, "conv0_bias_add"};
    conv0_bias_add.set_attr<std::string>(op::attr::data_format, "NCX");

    logical_tensor relu0_dst_desc {7, data_type::f16, 4, layout_type::undef};

    /// create relu0 operator
    op relu0 {8, op::kind::ReLU, {conv0_bias_add_dst_desc}, {relu0_dst_desc},
            "relu0"};

    /// create logical tensors with f16 data type for conv1
    logical_tensor conv1_weight_desc {
            9, data_type::f16, conv1_weight_dims, layout_type::undef};
    logical_tensor conv1_bias_desc {
            10, data_type::f16, conv1_bias_dims, layout_type::undef};
    logical_tensor conv1_dst_desc {11, data_type::f16, 4, layout_type::undef};

    /// create conv1 operator
    op conv1 {12, op::kind::Convolution, {relu0_dst_desc, conv1_weight_desc},
            {conv1_dst_desc}, "conv1"};
    conv1.set_attr<std::vector<int64_t>>(op::attr::strides, {1, 1});
    conv1.set_attr<std::vector<int64_t>>(op::attr::pads_begin, {0, 0});
    conv1.set_attr<std::vector<int64_t>>(op::attr::pads_end, {0, 0});
    conv1.set_attr<std::vector<int64_t>>(op::attr::dilations, {1, 1});
    conv1.set_attr<int64_t>(op::attr::groups, 1);
    conv1.set_attr<std::string>(op::attr::data_format, "NCX");
    conv1.set_attr<std::string>(op::attr::filter_format, "OIX");

    logical_tensor conv1_bias_add_dst_desc {
            13, data_type::f16, 4, layout_type::any};

    op conv1_bias_add {14, op::kind::BiasAdd, {conv1_dst_desc, conv1_bias_desc},
            {conv1_bias_add_dst_desc}, "conv1_bias_add"};
    conv1_bias_add.set_attr<std::string>(op::attr::data_format, "NCX");

    logical_tensor relu1_dst_desc {15, data_type::f16, 4, layout_type::undef};
    op relu1 {16, op::kind::ReLU, {conv1_bias_add_dst_desc}, {relu1_dst_desc},
            "relu1"};

    /// add operators to the graph
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

    /// Below codes are to create runtime objects like allocator, engine and stream.
    /// Unlike CPU example, users need provide sycl device, sycl context, and sycl queue.
    /// oneDNN Graph provides different interoperability APIs which are defined
    /// at `dnnl_graph_sycl.hpp`.

    /// construct an allocator
    allocator alloc = sycl_interop::make_allocator(sycl_malloc_wrapper, sycl_free_wrapper);

    /// construct an engine
    sycl::queue q(gpu_selector {}, sycl::property::queue::in_order {});
    engine eng = sycl_interop::make_engine(q.get_device(), q.get_context(), alloc);

    /// construct a stream
    dnnl::graph::stream strm = sycl_interop::make_stream(eng, q);

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
                data_buffers.back().reset(
                        ::sycl::malloc_shared(mem_size, q.get_device(), q.get_context()), sycl_deletor {q.get_context()});
                inputs_ts.push_back(tensor {in, eng, data_buffers.back().get()});
            }

            for (const auto &out : outputs) {
                size_t mem_size = out.get_mem_size();
                // memory allocation
                data_buffers.push_back({});
                data_buffers.back().reset(
                        ::sycl::malloc_shared(
                                mem_size, q.get_device(), q.get_context()), sycl_deletor {q.get_context()});
                outputs_ts.push_back(tensor {out, eng, data_buffers.back().get()});
                global_outputs_ts_map[out.get_id()] = outputs_ts.back();
            }

            /// Execute the compiled partition 1 on the specified stream.
            /// @snippet cpu_get_started.cpp Execute compiled partition 1
            //[Execute compiled partition]
            sycl_interop::execute(cp, strm, inputs_ts, outputs_ts);
            //[Execute compiled partition]
        } else {
            std::cout << "gpu_simple_pattern_fp16: got unsupported partition, users need "
                "handle the operators by themselves." << std::endl;
        }
    }
    // wait for all compiled partition's execution finished
    strm.wait();
    // clang-format on

    return 0;
}
