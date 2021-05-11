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
#ifndef BACKEND_DNNL_SUBGRAPH_MEMORY_BINDING_HPP
#define BACKEND_DNNL_SUBGRAPH_MEMORY_BINDING_HPP

#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>

#include "interface/value.hpp"

#include "dnnl.hpp"

#include "backend/dnnl/subgraph/passes.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

class execution_args_mgr {
public:
    execution_args_mgr() = default;

    // Disable assignment and copy
    execution_args_mgr(const execution_args_mgr &) = delete;
    execution_args_mgr(execution_args_mgr &&) = delete;
    execution_args_mgr &operator=(const execution_args_mgr &) = delete;
    execution_args_mgr &operator=(execution_args_mgr &&) = delete;

    int64_t init_args() {
        auto ret = data_.insert({counter++, std::unordered_map<int, memory>()});
        return ret.first->first;
    }

    std::unordered_map<int, memory> &get_args(int64_t key) {
        return data_[key];
    }

    // for external memory
    const memory &get_external_input_mem(size_t index) {
        return external_input_mems_[index];
    }

    const memory &get_external_output_mem(size_t index) {
        return external_output_mems_[index];
    }

    void add_external_input_mem(const memory &mem) {
        external_input_mems_.emplace_back(mem);
    }

    void add_external_output_mem(const memory &mem) {
        external_output_mems_.emplace_back(mem);
    }

    // for internal memory
    const std::vector<memory> &get_internal_mems() { return internal_mems_; }

    void add_internal_mem(const memory &mem) {
        internal_mems_.emplace_back(mem);
    }

    void add_value_mem_map(const std::pair<value_t *, memory> &map) {
        value_mem_map_.insert(map);
    }

    bool find_value_mem_map(value_t *key, memory &mem) {
        auto pos = value_mem_map_.find(key);
        if (pos != value_mem_map_.end()) {
            mem = pos->second;
            return true;
        }
        return false;
    }

    size_t get_internal_mem_size() {
        size_t total_size = 0;
        for (auto &mem : internal_mems_) {
            total_size += mem.get_desc().get_size();
        }
        return total_size;
    }

private:
    std::unordered_map<int64_t, std::unordered_map<int, memory>> data_;
    int64_t counter {0};

    std::vector<memory> external_input_mems_;
    std::vector<memory> external_output_mems_;
    std::vector<memory> internal_mems_;
    std::unordered_map<value_t *, memory> value_mem_map_; // pointer -> mem
};

impl::status_t memory_binding(
        std::vector<std::shared_ptr<impl::op_t>> &subgraph,
        const std::vector<impl::logical_tensor_t> &inputs,
        const std::vector<impl::logical_tensor_t> &outputs,
        const dnnl::engine &p_engine, execution_args_mgr &exec_arg_mgr,
        primitive_attr_mgr &prm_attr_mgr);

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
