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
        const deserialized_lt &lt, const bool should_use_graph_shape,
        const bool is_op_input, const bool is_fake_output) {

    // Init memory for all inputs and outputs that needs comparison
    const auto &prim_dt = mem.dt();
    const auto &graph_dt = static_cast<dnnl_data_type_t>(lt.get_data_type());

    // For int8 cases, as graph driver will modify the data type of leading
    // ops to u8/s8 in the reference path and use corresponding drivers to
    // generate data, special handling is needed. If it's found that data
    // type in ref path is u8/s8, it will be used.
    //
    // The reason why not always using primitive data type is that the driver
    // rewrites data type in graph path for bf16 case handling. So we prefer
    // data type in graph, and for int8 cases, that from ref path will be used.
    //
    dnnl_data_type_t c_data_type
            = prim_dt == dnnl_s8 || prim_dt == dnnl_u8 ? prim_dt : graph_dt;

    int ndims = mem.ndims();
    dims_t strides(mem.strides(), mem.strides() + ndims);
    std::string mtag = strides2memory_tag(ndims, strides);

    use_graph_shape_ = should_use_graph_shape;
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

    // create memory for graph path
    const auto &graph_mtag = strides2memory_tag(ndims, graph_strides_);
    const auto data_type = static_cast<dnnl::memory::data_type>(c_data_type);
    if (is_op_input) {
        if (should_use_graph_shape) {
            dnnl::memory::desc md(graph_dims_, data_type, graph_strides_);
            mem_ = dnn_mem_t(md.get(), ::get_test_engine());

            const void *prim_data_handle = static_cast<void *>(mem);
            void *graph_data_handle = mem_.get_mapped_pointer<void>();
            const auto &mem_size = mem.size();
            std::memcpy(graph_data_handle, prim_data_handle, mem_size);
        } else {
            mem_ = dnn_mem_t(
                    mem.md_, c_data_type, graph_mtag, ::get_test_engine());
            mem_.reorder(mem);
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

const dnn_mem_t &dnn_graph_mem_t::reorder_back_mem() {
    if (use_graph_shape_) { reshape_md(mem_, graph_dims_, graph_strides_); }
    return mem_;
}

} // namespace graph
