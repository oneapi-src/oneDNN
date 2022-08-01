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

#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "graph_example_utils.hpp"
#include "helpers_any_layout.hpp"

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using engine = dnnl::engine;
using stream = dnnl::stream;

void cpu_get_started_tutorial() {
    const int MB = 8, IC = 3, IH = 227, IW = 227;
    const int OC0 = 96, KH0 = 11, KW0 = 11;
    const int OC1 = 96, KH1 = 1, KW1 = 1;

    std::vector<int64_t> conv0_src_dims = {MB, IC, IH, IW};
    std::vector<int64_t> conv0_wei_dims = {OC0, IC, KH0, KW0};
    std::vector<int64_t> conv0_bia_dims = {OC0};
    std::vector<int64_t> conv1_wei_dims = {OC1, OC0, KH1, KW1};
    std::vector<int64_t> conv1_bia_dims = {OC1};
    std::vector<int64_t> dst_dims = {-1, -1, -1, -1};

    // graph
    graph g(engine::kind::cpu);

    // conv0 inputs and outputs
    auto conv0_src_lt = logical_tensor(
            0, data_type::f32, conv0_src_dims, layout_type::undef);
    auto conv0_wei_lt = logical_tensor(
            1, data_type::f32, conv0_wei_dims, layout_type::undef);
    auto conv0_dst_lt
            = logical_tensor(2, data_type::f32, dst_dims, layout_type::undef);

    // conv0
    op conv0(0, op::kind::Convolution, "conv0");
    conv0.add_inputs({conv0_src_lt, conv0_wei_lt});
    conv0.add_outputs({conv0_dst_lt});
    conv0.set_attr<std::vector<int64_t>>(op::attr::strides, {4, 4});
    conv0.set_attr<std::vector<int64_t>>(op::attr::pads_begin, {0, 0});
    conv0.set_attr<std::vector<int64_t>>(op::attr::pads_end, {0, 0});
    conv0.set_attr<std::vector<int64_t>>(op::attr::dilations, {1, 1});
    conv0.set_attr<int64_t>(op::attr::groups, 1);
    conv0.set_attr<std::string>(op::attr::data_format, "NCX");
    conv0.set_attr<std::string>(op::attr::filter_format, "OIX");

    // conv0_bias_add inputs and outputs
    auto conv0_bia_lt = logical_tensor(
            3, data_type::f32, conv0_bia_dims, layout_type::undef);
    auto conv0_bias_add_dst_lt
            = logical_tensor(4, data_type::f32, dst_dims, layout_type::undef);

    // conv0_bias_add
    op conv0_bias_add(1, op::kind::BiasAdd, "conv0_bias_add");
    conv0_bias_add.add_inputs({conv0_dst_lt, conv0_bia_lt});
    conv0_bias_add.add_outputs({conv0_bias_add_dst_lt});
    conv0_bias_add.set_attr<std::string>(op::attr::data_format, "NCX");

    // relu0 output
    auto relu0_dst_lt
            = logical_tensor(5, data_type::f32, dst_dims, layout_type::undef);

    // relu0
    op relu0(2, op::kind::ReLU, "relu0");
    relu0.add_inputs({conv0_bias_add_dst_lt});
    relu0.add_outputs({relu0_dst_lt});

    // conv1 inputs and outputs
    auto conv1_wei_lt = logical_tensor(
            6, data_type::f32, conv1_wei_dims, layout_type::undef);
    auto conv1_dst_lt
            = logical_tensor(7, data_type::f32, dst_dims, layout_type::undef);

    // conv1
    op conv1(3, op::kind::Convolution, "conv1");
    conv1.add_inputs({relu0_dst_lt, conv1_wei_lt});
    conv1.add_outputs({conv1_dst_lt});
    conv1.set_attr<std::vector<int64_t>>(op::attr::strides, {1, 1});
    conv1.set_attr<std::vector<int64_t>>(op::attr::pads_begin, {0, 0});
    conv1.set_attr<std::vector<int64_t>>(op::attr::pads_end, {0, 0});
    conv1.set_attr<std::vector<int64_t>>(op::attr::dilations, {1, 1});
    conv1.set_attr<int64_t>(op::attr::groups, 1);
    conv1.set_attr<std::string>(op::attr::data_format, "NCX");
    conv1.set_attr<std::string>(op::attr::filter_format, "OIX");

    // conv1_bias_add inputs and outputs
    auto conv1_bias_lt = logical_tensor(
            8, data_type::f32, conv1_bia_dims, layout_type::undef);
    auto conv1_bias_add_dst_lt
            = logical_tensor(9, data_type::f32, dst_dims, layout_type::undef);

    // conv1_bias_add
    op conv1_bias_add(4, op::kind::BiasAdd, "conv1_bias_add");
    conv1_bias_add.add_inputs({conv1_dst_lt, conv1_bias_lt});
    conv1_bias_add.add_outputs({conv1_bias_add_dst_lt});
    conv1_bias_add.set_attr<std::string>(op::attr::data_format, "NCX");

    // relu1 output
    auto relu1_dst_lt
            = logical_tensor(10, data_type::f32, dst_dims, layout_type::undef);

    // relu1
    op relu1(5, op::kind::ReLU, "relu1");
    relu1.add_inputs({conv1_bias_add_dst_lt});
    relu1.add_outputs({relu1_dst_lt});

    // construct graph
    g.add_op(conv0);
    g.add_op(conv0_bias_add);
    g.add_op(relu0);

    g.add_op(conv1);
    g.add_op(conv1_bias_add);
    g.add_op(relu1);

    // partition
    auto partitions = g.get_partitions();

    std::unordered_set<size_t> ids_with_any_layout;
    set_any_layout(partitions, ids_with_any_layout);

    allocator alloc {};
    dnnl::engine eng
            = make_engine_with_allocator(dnnl::engine::kind::cpu, 0, alloc);

    dnnl::stream strm {eng};

    std::unordered_map<size_t, tensor> global_outputs_ts_map;
    std::vector<std::shared_ptr<void>> data_buffers;
    std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;

    // loop over the returned partitions
    for (const auto &partition : partitions) {
        if (partition.is_supported()) {
            std::vector<logical_tensor> inputs = partition.get_in_ports();
            std::vector<logical_tensor> outputs = partition.get_out_ports();

            for (size_t idx = 0; idx < inputs.size(); ++idx) {
                size_t id = inputs[idx].get_id();
                if (id_to_queried_logical_tensors.find(id)
                        != id_to_queried_logical_tensors.end())
                    inputs[idx] = id_to_queried_logical_tensors[id];
                else {
                    auto ori_lt = inputs[idx];
                    inputs[idx] = logical_tensor {ori_lt.get_id(),
                            ori_lt.get_data_type(), ori_lt.get_dims(),
                            layout_type::strided};
                }
            }

            for (size_t idx = 0; idx < outputs.size(); ++idx) {
                size_t id = outputs[idx].get_id();
                layout_type ltype = layout_type::strided;
                if (ids_with_any_layout.count(id)) ltype = layout_type::any;
                auto ori_lt = outputs[idx];
                outputs[idx] = logical_tensor {ori_lt.get_id(),
                        ori_lt.get_data_type(), ori_lt.get_dims(), ltype};
            }

            // compile a partition
            compiled_partition cp = partition.compile(inputs, outputs, eng);

            for (size_t idx = 0; idx < outputs.size(); ++idx) {
                size_t id = outputs[idx].get_id();
                outputs[idx] = cp.query_logical_tensor(id);
                id_to_queried_logical_tensors[id] = outputs[idx];
            }

            std::vector<tensor> inputs_ts, outputs_ts;
            inputs_ts.reserve(inputs.size());
            outputs_ts.reserve(outputs.size());
            for (const auto &in : inputs) {
                size_t id = in.get_id();
                size_t mem_size = in.get_mem_size();
                auto pos = global_outputs_ts_map.find(id);
                if (pos != global_outputs_ts_map.end()) {
                    inputs_ts.push_back(pos->second);
                    continue;
                }
                data_buffers.push_back({});
                data_buffers.back().reset(malloc(mem_size), cpu_deletor {});
                inputs_ts.push_back(
                        tensor {in, eng, data_buffers.back().get()});
            }

            for (const auto &out : outputs) {
                size_t mem_size = out.get_mem_size();
                data_buffers.push_back({});
                data_buffers.back().reset(malloc(mem_size), cpu_deletor {});
                outputs_ts.push_back(
                        tensor {out, eng, data_buffers.back().get()});
                global_outputs_ts_map[out.get_id()] = outputs_ts.back();
            }

            // execute a compiled partition
            cp.execute(strm, inputs_ts, outputs_ts);
        } else {
            std::cout << "cpu_graph_getting_started: unsupported partition. "
                         "Users should handle the operators by themselves."
                      << std::endl;
        }
    }
    strm.wait();
}

int main(int argc, char **argv) {
    cpu_get_started_tutorial();
    return 0;
}