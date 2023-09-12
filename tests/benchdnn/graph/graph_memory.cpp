/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "graph_memory.hpp"

namespace graph {

dnn_graph_mem_t::dnn_graph_mem_t(const dnn_mem_t &mem,
        const deserialized_lt &lt, const bool is_op_input,
        const bool is_fake_output) {

    // Init memory for all inputs and outputs that needs comparison
    const auto &prim_dt = mem.dt();
    const auto &graph_dt = static_cast<dnnl_data_type_t>(lt.get_data_type());
    const bool is_boolean
            = lt.get_data_type() == logical_tensor::data_type::boolean;

    // Use data type from graph path to represent boolean
    const auto &c_data_type = is_boolean ? prim_dt : graph_dt;

    // Get memory tag of primitive memory
    int ndims = mem.ndims();
    dims_t strides(mem.strides(), mem.strides() + ndims);
    std::string mtag = strides2memory_tag(ndims, strides);

    graph_dims_ = lt.shape_;
    graph_strides_ = lt.stride_;

    // We create memory for graph path in two steps:
    // 1. Create memory objects.
    // 2. Do memory copy if needed.
    //
    // For inputs, graph path needs data from reference path,
    // and the data movement requires both memories have the same
    // shape, so the tag of graph path is used to create the memory.
    //
    // For outputs, use shape & tag from graph path for fake outputs,
    // otherwise use shape & tag from ref path side

    // Create memory for graph path
    const auto data_type = static_cast<dnnl::memory::data_type>(c_data_type);
    if (is_op_input) {
        if (graph_dims_.empty()) graph_dims_.push_back(1);
        if (graph_strides_.empty()) graph_strides_.push_back(1);

        // create graph memory
        dnnl::memory::desc md(graph_dims_, data_type, graph_strides_);
        mem_ = dnn_mem_t(md.get(), ::get_test_engine());

        const auto prim_to_graph_memcpy = [](dnn_mem_t &graph_mem,
                                                  const dnn_mem_t &prim_mem) {
            const void *prim_data_handle = static_cast<const void *>(prim_mem);
            void *graph_data_handle = graph_mem.get_mapped_pointer<void>();
            std::memcpy(graph_data_handle, prim_data_handle, graph_mem.size());
        };

        // Not do reorder for boolean data tensor
        if (!is_boolean && prim_dt != c_data_type) {
            dnn_mem_t c_mem(
                    ndims, mem.dims(), c_data_type, mtag, ::get_test_engine());
            SAFE_V(c_mem.reorder(mem));
            prim_to_graph_memcpy(mem_, c_mem);
        } else {
            prim_to_graph_memcpy(mem_, mem);
        }
    } else {
        if (is_fake_output) {
            dnnl::memory::desc md(graph_dims_, data_type, graph_strides_);
            mem_ = dnn_mem_t(md.get(), ::get_test_engine());
        } else {
            mem_ = dnn_mem_t(mem.md_, c_data_type, mtag, ::get_test_engine());
        }
    }
}

dnnl::graph::tensor dnn_graph_mem_t::make_graph_tensor(
        const deserialized_lt &lt) const {
    void *data_handle;
    dnnl_memory_get_data_handle(mem_.m_, &data_handle);
    dnnl::graph::logical_tensor graph_lt(lt.id_, lt.get_data_type(), lt.shape_,
            str2layout(lt.layout_type_), lt.get_property_type());
    dnnl::graph::tensor ret(graph_lt, get_graph_engine(), data_handle);

    return ret;
}

} // namespace graph
