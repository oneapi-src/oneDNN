/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef BENCHDNN_GRAPH_MEMORY_HPP
#define BENCHDNN_GRAPH_MEMORY_HPP

#include <mutex>
#include <tuple>
#include "common.hpp"
#include "deserialize.hpp"
#include "dnnl_common.hpp"
#include <type_traits>
#include <unordered_map>

#include "setting_handler.hpp"
#include "utils/compare.hpp"
#include "utils/settings.hpp"

#ifdef DNNL_WITH_SYCL
#include "dnnl_sycl.hpp"
#endif

enum mem_platform_t {
    CPU_REQ = 0,
    GPU_REQ = 1,
};

enum mem_path_t {
    GRAPH_INTERNAL
    = 0, // Memory allocated during graph runtime, such as scratchpad and constant tensor cache.
    GRAPH_USER = 1, // Given inputs and outputs memories from the user
    REF = 2, // Memory allocated for the reference path
};

namespace graph {

// The memory footprint for a whole graph. The memories mainly come from:
// 1. Reference primitives, such as the test objects and reference memories
// used for data filling;
// 2. Graph path, such as the input/output tensors, constance cache and
// scratchpads inside the compiled partitions;
// 3. Correctness check, extra memories are needed for reordering the results
// to abx+f32.
// 4. Other optimizations, such as the mapping machanism.
//
//
// The args will be checked and increased at the following stages:
// 1. Initilaization of reference path
// 2. Runtime of graph path
//
// And it will be reset at the following stages:
// 1. After the testing for each partition finishs, the memory size for the
//    refrence path will be reset. In addition, for correctness mode, the size
//    of graph input/output memories will be reset as well.
// 2. After each case ends, the record will be reset.
struct graph_memory_req_args_t {
public:
    static graph_memory_req_args_t &get_instance() {
        static graph_memory_req_args_t _instance;
        return _instance;
    }

    // Increase the memory size for given device and path.
    void increase_mem_req(
            mem_platform_t device, mem_path_t path, size_t mem_req) {
        if (device < 0 || path < 0) return;
        if (device >= req_.size() || path >= req_.front().size()) return;

        std::lock_guard<std::mutex> guard(mutex_);
        req_[device][path] += mem_req;
    }

    // Get the memory size on a specific device
    size_t get_mem_req(mem_platform_t device) {
        if (device < 0 || device >= req_.size()) return INT_MAX;

        std::lock_guard<std::mutex> guard(mutex_);
        size_t total_req = 0;
        for (const size_t &req : req_[device])
            total_req += req;
        return total_req;
    }

    // Reset the memory request for the specific path.
    void reset_path(mem_path_t path) {
        if (path < 0 || path > req_.front().size()) return;

        std::lock_guard<std::mutex> guard(mutex_);
        req_[CPU_REQ][path] = 0;
        req_[GPU_REQ][path] = 0;
    }

    // Reset the memory size args for both paths on all devices
    void reset_all() {
        req_ = std::vector<std::vector<size_t>>(2, std::vector<size_t>(3, 0));
    }

private:
    graph_memory_req_args_t() {
        req_ = std::vector<std::vector<size_t>>(2, std::vector<size_t>(3, 0));
    }
    ~graph_memory_req_args_t() {};
    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(graph_memory_req_args_t);

    // The detailed memory size requrest for specific path and devices.
    std::vector<std::vector<size_t>> req_;
    std::mutex mutex_;
};

inline void reset_graph_mem_req() {
    auto &graph_mem_req = graph_memory_req_args_t::get_instance();
    // reset the memory request for the graph.
    graph_mem_req.reset_all();
}

size_t get_benchdnn_cpu_limit();
size_t get_benchdnn_device_limit();

struct dnn_graph_mem_t {

public:
    // Construct graph memory on ref path based on primitive mem
    // Apart from oneDNN memory tag, oneDNN Graph has op attributes `data_format`
    // (NXC/NCX) and `weights_format`(OIX/IOX/XOI/XIO) to indicate the order of
    // dimensions.
    //
    // For example, tensor with shape [1,4,4,3] and NXC data_format means
    // batch_size=1, spacial_dims=4x4, channel_dim=3. This semantic info is
    // necessary for some scenarios, i.e. per_channel quantization.
    //
    // As the order of dimensions in oneDNN memory descriptor is always
    // NC[D[H[W]]], to align with oneDNN, the shape and strides should be changed
    // properly in setting conversion stage to get the memory tag correctly.
    //
    // Meanwhile, in graph path, the shape and strides of the memory for graph
    // path should remain the same as in the deserialized graph. In addition,
    // it should has the same data as that for reference path.
    //
    // Therefore, it needs to be checked that if the shape and strides of the
    // logical tensor have been modified. If so, the driver should use the shape
    // and strides from deserialized graph instead.
    //
    //
    // The constructor accepts three boolean parameters:
    // 1. is_op_input: whether the logical tensor is an input of an op
    // 2. is_fake_output: for fake outputs, the driver cannot create memory
    // objects based on primitive memory for them, but construct memory
    // from graph shape. The default value is false.
    //
    dnn_graph_mem_t(const dnn_mem_t &mem, const deserialized_lt &lt,
            const bool is_op_input, const bool is_fake_output = false);

    dnnl::graph::tensor make_graph_tensor(const deserialized_lt &lt) const;

    const dnn_mem_t &get_mem() const { return mem_; }

    void map_mem() { mem_.map(); }
    void unmap_mem() { mem_.unmap(); }

private:
    dnn_mem_t mem_;
    std::shared_ptr<void> buffer_;
    dnnl::memory::dims graph_dims_;
    dnnl::memory::dims graph_strides_;
};

using partition_mem_map_t = std::unordered_map<size_t, dnn_graph_mem_t>;

void flush_temp_memory();

} // namespace graph

#endif
