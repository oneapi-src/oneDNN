/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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
#include "allocator.hpp"

#include "oneapi/dnnl/dnnl_graph.hpp"

// 0.75f is taken randomly and is subject to change in future.
static constexpr float capacity_factor = 0.75f;

namespace graph {

size_t get_benchdnn_cpu_limit() {
    static size_t cpu_device_capacity = get_cpu_ram_size();
    const double benchdnn_cpu_limit = capacity_factor * cpu_device_capacity;
    assert(benchdnn_cpu_limit > 0);
    return benchdnn_cpu_limit;
}

size_t get_benchdnn_device_limit() {
    if (is_cpu()) return 0;
    static size_t gpu_device_capacity = 0;
    static size_t gpu_max_alloc_capacity = 0;
    SAFE(get_gpu_ram_sizes(gpu_device_capacity, gpu_max_alloc_capacity), WARN);

    const double benchdnn_device_limit = capacity_factor * gpu_device_capacity;
    assert(benchdnn_device_limit > 0);
    return benchdnn_device_limit;
}

// Constructs memories for all inputs and outputs needed for comparison.
dnn_graph_mem_t::dnn_graph_mem_t(const dnn_mem_t &mem,
        const deserialized_lt &lt, const bool is_op_input,
        const bool is_fake_output)
    : graph_dims_(lt.shape_), graph_strides_(lt.stride_) {
    const auto &prim_dt = mem.dt();
    // Conversion from graph types to dnnl types + boolean to u8.
    const auto &graph_dt = convert_dt(lt.get_data_type());

    // Get memory tag of primitive memory
    int ndims = mem.ndims();
    dims_t strides(mem.strides(), mem.strides() + ndims);
    std::string mtag = strides2memory_tag(ndims, strides);

    const auto &g_eng = get_graph_engine().operator const dnnl::engine &();

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
    const auto data_type = static_cast<dnnl::memory::data_type>(graph_dt);
    if (is_op_input) {
        if (graph_dims_.empty()) graph_dims_.push_back(1);
        if (graph_strides_.empty()) graph_strides_.push_back(1);

        // create graph memory
        dnnl::memory::desc md(graph_dims_, data_type, graph_strides_);
        mem_ = dnn_mem_t(md.get(), g_eng.get());

        const auto prim_to_graph_memcpy = [](dnn_mem_t &graph_mem,
                                                  const dnn_mem_t &prim_mem) {
            const void *prim_data_handle = static_cast<const void *>(prim_mem);
            void *graph_data_handle = graph_mem.get_mapped_pointer<void>();
            std::memcpy(graph_data_handle, prim_data_handle, graph_mem.size());
        };

        if (prim_dt != graph_dt) {
            // Call a reorder (for data conversion) when reference memory
            // doesn't coincide with the graph memory...
            dnn_mem_t c_mem(ndims, mem.dims(), graph_dt, mtag, g_eng.get());
            SAFE_V(c_mem.reorder(mem));
            prim_to_graph_memcpy(mem_, c_mem);
        } else {
            // ... otherwise, perform a plain memcpy.
            prim_to_graph_memcpy(mem_, mem);
        }
    } else {
        if (is_fake_output) {
            dnnl::memory::desc md(graph_dims_, data_type, graph_strides_);
            mem_ = dnn_mem_t(md.get(), g_eng.get());
        } else {
            mem_ = dnn_mem_t(mem.md_, graph_dt, mtag, g_eng.get());
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

void flush_temp_memory() {
    using namespace dnnl::graph;
    // flush the constant tensor cache.
    const auto kind = engine_tgt_kind == dnnl_cpu ? engine::kind::cpu
                                                  : engine::kind::gpu;
    static size_t ct_capacity = get_constant_tensor_cache_capacity(kind);
    if (ct_capacity > 0) set_constant_tensor_cache_capacity(kind, ct_capacity);

        // flush the compiled partition cache.
#ifndef DNNL_GRAPH_DISABLE_COMPILED_PARTITION_CACHE
    static int cp_capacity = get_compiled_partition_cache_capacity();
    set_compiled_partition_cache_capacity(0); // clear the cache
    set_compiled_partition_cache_capacity(
            cp_capacity); // reset the cache capacity.
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (!has_bench_mode_bit(mode_bit_t::corr) && is_gpu()) {
        auto &graph_mem_mgr = graph_mem_manager_t::get_instance();
        graph_mem_mgr.clear_memory_pool();
    }
#endif
}

} // namespace graph
