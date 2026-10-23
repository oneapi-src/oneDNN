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
#include "allocator.hpp"

#include "utils/system.hpp"

#include "oneapi/dnnl/dnnl_graph.hpp"

#include <cstring>

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
        const deserialized_lt_t &lt, const bool is_op_input, res_t *res,
        const bool use_graph_layout)
    : graph_dims_(lt.shape_), graph_strides_(lt.stride_) {
    const auto &g_eng = lt.is_host_scalar()
            ? get_graph_host_engine().operator const dnnl::engine &()
            : get_graph_engine().operator const dnnl::engine &();

    // For inputs, graph path needs data from reference path,
    // and the data movement requires both memories have the same
    // shape, so the tag of graph path is used to create the memory.
    //
    // For outputs, use shape & tag from graph path for fake outputs,
    // otherwise use shape & tag from ref path side

    // Create memory for graph path
    const auto &graph_dt = convert_dt(lt.get_data_type());
    const auto data_type = static_cast<dnnl::memory::data_type>(graph_dt);

    if (graph_dims_.empty()) {
        // As graph strides are deduced from graph dims, they should be in
        // compliance with each other.
        assert(graph_strides_.empty());

        graph_dims_.push_back(1);
        graph_strides_.push_back(1);
    }

    if (is_op_input) {
        // Create graph memory with memory description from graph path.
        dnnl::memory::desc md(graph_dims_, data_type, graph_strides_);
        mem_ = dnn_mem_t(md.get(), g_eng.get(), /* prefill = */ true);

        if (!has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) {
            // Fill data from reference memories.
            fill_mem_with_data(mem, g_eng, res);
        }

    } else {
        if (use_graph_layout) {
            // For some cases such as fake outputs and no reference memory
            // mode, which means the output does not have correctponding
            // argument in primitives, we need to create them with memory
            // description from graph path.
            dnnl::memory::desc md(graph_dims_, data_type, graph_strides_);
            mem_ = dnn_mem_t(md.get(), g_eng.get(), /* prefill = */ true);

        } else {
            // Use information from the graph logical tensor to create output
            // memories. The memory strides must match the compiled partition's
            // output strides so that the comparison reads data correctly.
            // When the graph LT has valid (explicit) strides and the graph
            // dims match the reference dims (i.e. no NXC->NCX reordering
            // occurred), use the graph strides; otherwise fall back to the
            // reference memory descriptor's tag
            bool has_valid_graph_strides
                    = graph_strides_.size() == graph_dims_.size()
                    && std::all_of(graph_strides_.begin(), graph_strides_.end(),
                            [](int64_t s) { return s > 0; });

            bool dims_match = has_valid_graph_strides
                    && (graph_dims_.size() == static_cast<size_t>(mem.ndims()));
            if (dims_match) {
                for (int i = 0; i < mem.ndims(); i++) {
                    if (graph_dims_[i] != mem.dims()[i]) {
                        dims_match = false;
                        break;
                    }
                }
            }

            if (has_valid_graph_strides && dims_match) {
                dnnl::memory::desc md(graph_dims_, data_type, graph_strides_);
                mem_ = dnn_mem_t(md.get(), g_eng.get(), /* prefill = */ true);
            } else {
                // Get memory tag of primitive memory
                int ndims = mem.ndims();
                dims_t strides(mem.strides(), mem.strides() + ndims);
                dims_t shape(mem.dims(), mem.dims() + ndims);
                std::string mtag = strides2memory_tag(shape, strides, false);

                mem_ = dnn_mem_t(mem.md_, graph_dt, mtag, g_eng.get(),
                        /* prefill = */ true);
            }
        }
    }
}

int dnn_graph_mem_t::fill_mem_with_data(
        const dnn_mem_t &mem, const dnnl::engine &eng, res_t *res) {
    // If `mem` comes empty, don't update the underlying `mem_` then.
    if (!mem) return OK;

    const auto &src_eng = mem.engine();
    const auto &dst_eng = eng.get();

    const auto &src_dt = mem.dt();
    const auto &dst_dt = mem_.dt();
    if (src_dt == dst_dt && mem.size() != mem_.size()) {
        BENCHDNN_PRINT(0, "%s\n",
                "Error: failed to fill graph memory with given memory\n");
        SAFE(FAIL, WARN);
    }

    int ndims = mem.ndims();
    const auto prim_to_graph_memcpy
            = [](dnn_mem_t &graph_mem, const dnn_mem_t &prim_mem) {
        const void *prim_data_handle = static_cast<const void *>(prim_mem);
        void *graph_data_handle = graph_mem.get_mapped_pointer<void>();
        std::memcpy(graph_data_handle, prim_data_handle, graph_mem.size());
    };

    if (src_dt != dst_dt || src_eng != dst_eng) {
        // If dt or eng is different, need to transfer data under same dt or
        // engine to perform a data copy.
        dnn_mem_t c_mem(ndims, mem.dims(), dst_dt, mem.strides(), dst_eng,
                /* prefill = */ true);
        SAFE_V(c_mem.reorder(mem, res));
        prim_to_graph_memcpy(mem_, c_mem);
    } else {
        prim_to_graph_memcpy(mem_, mem);
    }

    return OK;
}

dnnl::graph::tensor dnn_graph_mem_t::make_graph_tensor(
        const deserialized_lt_t &lt, res_t *) const {
    void *data_handle;
    dnnl_memory_get_data_handle(mem_.m_, &data_handle);
    dnnl::graph::logical_tensor graph_lt = lt.create();
    const auto &g_eng = get_graph_engine().operator const dnnl::engine &();

    if (lt.is_host_scalar()) {
        return dnnl::graph::tensor::make_scalar_tensor(graph_lt, data_handle);
    } else {
        return dnnl::graph::tensor(graph_lt, g_eng, data_handle);
    }
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

    if (!has_bench_mode_bit(mode_bit_t::corr)) {
        auto &graph_mem_mgr = graph_mem_manager_t::get_instance();
        graph_mem_mgr.clear_memory_pool();
    }
}

} // namespace graph
