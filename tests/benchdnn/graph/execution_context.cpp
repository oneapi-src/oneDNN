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

#include "execution_context.hpp"

namespace graph {

void tensor_map::insert_or_replace(size_t id, const dnnl::graph::tensor &ts) {
    if (has(id)) {
        fprintf(stderr,
                "graph: Partition: repeat tensor id `%zd`, exiting...\n", id);
        exit(2);
    }
    data_[id] = ts;
}

std::vector<dnnl::graph::tensor> tensor_map::construct_and_initialize_tensors(
        const std::vector<dnnl::graph::logical_tensor> &lts,
        const dnnl::graph::compiled_partition &c_partition,
        const dnnl::engine &eng, int value) {
    std::vector<dnnl::graph::tensor> ret;
    ret.reserve(lts.size());
    const auto &inplace_ports = c_partition.get_inplace_ports();
    for (const auto &lt : lts) {
        auto id = lt.get_id();
        auto pos = std::find_if(inplace_ports.begin(), inplace_ports.end(),
                [id](const std::pair<size_t, size_t> &p) {
                    return id == p.second;
                });
        // tensor map already has requested tensor, take it out
        if (has(id)) {
            ret.emplace_back(data_.at(id));
            continue;
        }
        // tensor map doesn't contain requested tensor, need to create from
        // logical tensor queried from compiled partition
        const auto &new_lt = c_partition.query_logical_tensor(id);

        if (pos != inplace_ports.end()) {
            auto in_buffer = data_.at(pos->first).get_data_handle();
            ret.emplace_back(dnnl::graph::tensor {new_lt, eng, in_buffer});
            insert_or_replace(new_lt.get_id(), ret.back());
            continue;
        }
        // memory buffer allocation and initialization
        void *mem_ptr = nullptr;
        if (eng.get_kind() == dnnl::engine::kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
            mem_ptr = ::sycl::malloc_shared(
                    new_lt.get_mem_size(), q_.get_device(), q_.get_context());
            buffer_map_[new_lt.get_id()].reset(
                    mem_ptr, sycl_deletor {q_.get_context()});
            GRAPH_SWITCH_TYPE(new_lt.get_data_type(), dtype, {
                fill_buffer<dtype>(q_, mem_ptr,
                        static_cast<size_t>(
                                new_lt.get_mem_size() / sizeof(dtype)),
                        value);
            });
#else
            mem_ptr = malloc(new_lt.get_mem_size());
            buffer_map_[new_lt.get_id()].reset(mem_ptr, cpu_deletor {});
            GRAPH_SWITCH_TYPE(new_lt.get_data_type(), dtype, {
                fill_buffer<dtype>(mem_ptr, new_lt.get_mem_size(), value);
            });
#endif
        } else {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
            mem_ptr = ::sycl::malloc_shared(
                    new_lt.get_mem_size(), q_.get_device(), q_.get_context());
            buffer_map_[new_lt.get_id()].reset(
                    mem_ptr, sycl_deletor {q_.get_context()});
            GRAPH_SWITCH_TYPE(new_lt.get_data_type(), dtype, {
                fill_buffer<dtype>(q_, mem_ptr,
                        static_cast<size_t>(
                                new_lt.get_mem_size() / sizeof(dtype)),
                        value);
            });
#endif
        }
        ret.emplace_back(dnnl::graph::tensor {new_lt, eng, mem_ptr});
        insert_or_replace(new_lt.get_id(), ret.back());
    }
    return ret;
}

} // namespace graph
