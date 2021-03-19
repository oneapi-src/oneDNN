/*******************************************************************************
* Copyright 2021 Intel Corporation
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

/// @example cpu_programming.cpp
/// @copybrief cpu_programming_cpp
/// > Annotated version: @ref cpu_programming_cpp

/// @page cpu_programming_cpp Example for demonstrating programming model
///
/// > Example code: @ref cpu_programming.cpp
///
/// This example will construct the below graph. The graph has two outputs which
/// are connected to End op. Now, Conv and Add ops should not be fused due to
/// tensor1 is also used as an output of the graph.
///
///         Conv     Wildcard
///           |         |
///        tensor1   tensor2
///       /      \     /
///     End        Add
///                 |
///              tensor3
///                 |
///                ReLU
///                 |
///              tensor4
///                 |
///                End
///

#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "common/utils.hpp"

#define assertm(exp, msg) assert(((void)msg, exp))

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;

// global mapping from id to tensor to manage the life cycle of created tensors
// need to hold until all compiled partitions' execution finished
class tensorMap {
private:
    std::unordered_map<size_t, tensor> data_;

public:
    tensorMap() = default;

    ~tensorMap() {
        // free the memory buffer
        for (auto iter = data_.begin(); iter != data_.end(); ++iter) {
            float *mem_ptr = iter->second.get_data_handle<float>();
            if (mem_ptr != nullptr) deallocate(mem_ptr);
        }
    }

    // insert new tensor into the map
    void insert(size_t id, const tensor &ts) { data_[id] = ts; }
    // return if this tensor map contains a tensor according to the given id
    bool has(size_t id) const { return data_.find(id) != data_.end(); }
    // retrieve the id from this tensor map according to the given id
    const tensor get(size_t id) const { return data_.at(id); }
};

/// Below are some helper functions

// fill the memory according to the given value
//  src -> target memory buffer
//  total_size -> total number of bytes of this buffer
//  val -> fixed value for initialization
template <typename T>
void fill_buffer(void *src, size_t total_size, int val) {
    int num_elem = static_cast<int>(total_size / sizeof(T));
    T *src_casted = static_cast<T *>(src);
    // can be implemented through OpenMP
    for (size_t i = 0; i < num_elem; ++i)
        *(src_casted + i) = static_cast<T>(val);
}

// create input/output tensors for the compiled partition and initialize value
//  ids -> the given list of ids of tensors to be constructed and initialized
//  c_partition -> target compiled partition
//  tm -> a context to manage tensors' life cycle.
//  value -> fixed value for initialization of the buffer
std::vector<tensor> construct_and_initialize_tensors(
        const std::vector<logical_tensor> &lts, compiled_partition &c_partition,
        tensorMap &tm, int value) {
    std::vector<tensor> ret;
    ret.reserve(lts.size());
    for (auto &lt : lts) {
        size_t id = lt.get_id();
        if (tm.has(id)) {
            // tensorMap has already contained this tensor, just get it out
            ret.emplace_back(tm.get(id));
        } else {
            // tensorMap doesn't contain this tensor, need create from logical tensor
            // query logical tensor from compiled partition
            logical_tensor new_lt = c_partition.query_logical_tensor(id);
            // allocate memory buffer
            void *mem_ptr = allocate(new_lt.get_mem_size(), {});
            // initialize value
            if (new_lt.get_data_type() == data_type::f32)
                fill_buffer<float>(mem_ptr, new_lt.get_mem_size(), value);
            else if (new_lt.get_data_type() == data_type::s8)
                fill_buffer<int8_t>(mem_ptr, new_lt.get_mem_size(), value);
            else if (new_lt.get_data_type() == data_type::s32)
                fill_buffer<int32_t>(mem_ptr, new_lt.get_mem_size(), value);
            else if (new_lt.get_data_type() == data_type::u8)
                fill_buffer<uint8_t>(mem_ptr, new_lt.get_mem_size(), value);
            else
                throw std::runtime_error(
                        "Currently, bfloat16/float16 are not supported in this "
                        "example.");

            ret.emplace_back(tensor {new_lt, mem_ptr});
            tm.insert(new_lt.get_id(), ret.back());
        }
    }
    return ret;
}

// clang-format off
int cpu_programming_tutorial(engine::kind engine_kind) {
    /// construct a graph based on the given engine kind
    graph g(engine_kind);

    std::vector<int64_t> input_dims {8, 3, 227, 227};
    std::vector<int64_t> weight_dims {96, 3, 11, 11};
    std::vector<int64_t> bias_dims {96};
    std::vector<int64_t> dst_dims {8, 96, 217, 217};
    std::vector<int64_t> add_input_2_dims {8, 96, 217, 217};

    logical_id_manager &id_mgr = logical_id_manager::get();

    /// each logical tensor should be given with unique name
    /// the name is 1:1 mapping with the given id
    logical_tensor conv_data_lt {id_mgr["conv_data"], data_type::f32, input_dims, layout_type::strided};
    logical_tensor conv_weight_lt {id_mgr["conv_weight"], data_type::f32, weight_dims, layout_type::strided};
    logical_tensor conv_bias_lt {id_mgr["conv_bias"], data_type::f32, bias_dims, layout_type::strided};
    logical_tensor conv_dst_lt {id_mgr["dst_dims"], data_type::f32, dst_dims, layout_type::strided};

    op conv {0, op::kind::Convolution, {conv_data_lt, conv_weight_lt, conv_bias_lt}, {conv_dst_lt}, "conv_0"};
    conv.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
    conv.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
    conv.set_attr<std::vector<int64_t>>("dilations", {1, 1});
    conv.set_attr<int64_t>("groups", 1);
    conv.set_attr<std::string>("data_format", "NCX");
    conv.set_attr<std::string>("filter_format", "OIX");

    logical_tensor add_input_2_lt {id_mgr["add_input_2"], data_type::f32, add_input_2_dims, layout_type::strided};
    logical_tensor add_dst_lt {id_mgr["add_dst"], data_type::f32, dst_dims, layout_type::strided};

    op add {1, op::kind::Add, {conv_dst_lt, add_input_2_lt}, {add_dst_lt}, "add_0"};

    logical_tensor relu_dst_lt {id_mgr["relu_dst"], data_type::f32, dst_dims, layout_type::strided};

    op relu {2, op::kind::ReLU, {add_dst_lt}, {relu_dst_lt}, "relu_0"};

    op wildcard {3, op::kind::Wildcard, {}, {add_input_2_lt}, "wildcard_0"};

    op end_0 {4, op::kind::End, {conv_dst_lt}, {}, "end_0"};
    op end_1 {5, op::kind::End, {relu_dst_lt}, {}, "end_1"};

    /// mapping from op id to op kind
    std::unordered_map<size_t, op::kind> op_id_kind_map {{0, op::kind::Convolution},
        {1, op::kind::Add}, {2, op::kind::ReLU}, {3, op::kind::Wildcard}, {4, op::kind::End}, {5, op::kind::End}};

    /// add op to graph
    g.add_op(conv);
    g.add_op(add);
    g.add_op(relu);
    g.add_op(wildcard);
    g.add_op(end_0);
    g.add_op(end_1);

    /// get partitions from the graph
    std::vector<partition> partitions = g.get_partitions();

    std::cout << "Number of returned partitions: " << partitions.size() << "\n";
    for (size_t i = 0; i < partitions.size(); ++i) {
        std::cout << "Partition[" << partitions[i].get_id()
                  << "]'s supporting status: "
                  << (partitions[i].is_supported() ? "true" : "false") << "\n";
    }

    /// construct a new engine
    int device_id = 0;
    engine e {engine_kind, device_id};

    /// construct a new stream
    stream s {e};

    std::vector<compiled_partition> c_partitions(partitions.size());

    // mapping from id to tensors
    tensorMap tm;

    /// infer shape, compile, execute
    for (size_t i = 0; i < partitions.size(); ++i) {
        if (partitions[i].is_supported()) {
            std::cout << "\nPartition[" << partitions[i].get_id() << "] is being processed.\n";
            std::vector<logical_tensor> inputs = partitions[i].get_inputs();
            std::vector<logical_tensor> outputs = partitions[i].get_outputs();

            std::cout << "Compiling-------------------------------------------";
            /// compile to generate compiled partition
            c_partitions[i] = partitions[i].compile(inputs, outputs, e);
            std::cout << "Successfully.\n";

            std::cout << "Creating tensors and allocating memory buffer-------";
            std::vector<tensor> input_ts = construct_and_initialize_tensors(inputs, c_partitions[i], tm, 1);
            std::vector<tensor> output_ts = construct_and_initialize_tensors(outputs, c_partitions[i], tm, 0);
            std::cout << "Successfully.\n";

            std::cout << "Executing compiled partition------------------------";
            /// execute the compiled partition
            c_partitions[i].execute(s, input_ts, output_ts);
            std::cout << "Successfully.\n";
        } else {
            std::vector<size_t> unsupported_op_ids = partitions[i].get_ops();
            assertm(unsupported_op_ids.size() == 1, "Unsupported partition only "
                "contains single op.");
            if (op_id_kind_map[unsupported_op_ids[0]] == op::kind::Wildcard) {
                std::cout << "\nWarning (actually an error): partition " << partitions[i].get_id() <<
                        " contains only a Wildcard op which cannot be computed.\n";
            } else {
                /// Users need to write implementation code by themselves.
                continue;
            }
        }
    }

    /// check correctness
    float expected_output_1 = /*weight = */11 * 11 * /*channel = */3 + /*bias = */ + /*bias = */1.0f;
    float expected_output_2 = (/*weight = */11 * 11 * /*channel = */3 + /*bias = */ + /*bias = */1.0f) + /*add*/1.0f;

    if (partitions.size() == 6) {
        float *actual_output_ptr1 = tm.get(conv_dst_lt.get_id()).get_data_handle<float>();
        auto output_dims = conv_dst_lt.get_dims();
        auto num_elem = std::accumulate(output_dims.begin(), output_dims.end(), 0);
        for (int i = 0; i < num_elem; ++i) {
            if (std::abs(expected_output_1 - actual_output_ptr1[i]) > 1e-6f) {
                printf("expected = %.2f, actual = %.2f\n", expected_output_1, actual_output_ptr1[i]);
                throw std::runtime_error(
                        "output result is not equal to excepted "
                        "results");
            }
        }
    }

    float *actual_output_ptr2 = tm.get(relu_dst_lt.get_id()).get_data_handle<float>();
    auto output_dims = relu_dst_lt.get_dims();
    auto num_elem = std::accumulate(output_dims.begin(), output_dims.end(), 0);
    for (int i = 0; i < num_elem; ++i) {
        if (std::abs(expected_output_2 - actual_output_ptr2[i]) > 1e-6f) {
            printf("expected = %.2f, actual = %.2f\n", expected_output_2, actual_output_ptr2[i]);
            throw std::runtime_error(
                    "output result is not equal to excepted "
                    "results");
        }
    }
    std::cout << "Example passed successfully!\n";
    return 0;
}
// clang-format on
int main(int argc, char **argv) {
    engine::kind engine_kind = parse_engine_kind(argc, argv);
    return cpu_programming_tutorial(engine_kind);
}
