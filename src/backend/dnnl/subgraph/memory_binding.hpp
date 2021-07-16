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

#include "utils/utils.hpp"

#include "backend/dnnl/subgraph/passes.hpp"
#include "backend/dnnl/utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

class execution_args_mgr {
public:
    execution_args_mgr() = default;

    // Disable movement ctor
    execution_args_mgr(execution_args_mgr &&) = delete;
    execution_args_mgr &operator=(const execution_args_mgr &) = delete;
    execution_args_mgr &operator=(execution_args_mgr &&) = delete;

    // Deep copy constructor
    execution_args_mgr(const execution_args_mgr &other)
        : counter(other.counter), topo_ordered_keys_(other.topo_ordered_keys_) {
        // clone
        for (auto &val_mem : other.value_mem_map_) {
            memory cloned_mem(val_mem.second.get_desc(),
                    val_mem.second.get_engine(), nullptr);
            value_mem_map_.insert({val_mem.first, cloned_mem});
        }

        auto find_val = [&](const memory &mem) -> value_t * {
            auto pos = std::find_if(other.value_mem_map_.begin(),
                    other.value_mem_map_.end(),
                    [&](const std::pair<value_t *, memory> &val_mem) {
                        return val_mem.second.get() == mem.get();
                    });
            assertm(pos != other.value_mem_map_.end(), "can't find such mem");
            if (pos != other.value_mem_map_.end())
                return pos->first;
            else
                return nullptr;
        };

        // copy alias
        for (auto &mem : other.external_input_mems_) {
            external_input_mems_.emplace_back(value_mem_map_.at(find_val(mem)));
        }

        for (auto &mem : other.external_output_mems_) {
            external_output_mems_.emplace_back(
                    value_mem_map_.at(find_val(mem)));
        }

        for (auto &mem : other.internal_variable_mems_) {
            internal_variable_mems_.emplace_back(
                    value_mem_map_.at(find_val(mem)));
        }

        for (auto &mem : other.internal_constant_mems_) {
            internal_constant_mems_.emplace_back(
                    value_mem_map_.at(find_val(mem)));
        }

        for (auto &key_args : other.data_) {
            int64_t key = key_args.first;
            const std::unordered_map<int, memory> &args = key_args.second;

            std::unordered_map<int, memory> new_args;
            for (auto &arg : args) {
                int idx = arg.first;
                const memory &mem = arg.second;
                new_args.insert({idx, value_mem_map_.at(find_val(mem))});
            }
            data_.insert({key, new_args});
        }
    }

    int64_t init_args() {
        auto ret = data_.insert({counter++, std::unordered_map<int, memory>()});
        return ret.first->first;
    }

    std::unordered_map<int, memory> &get_args(int64_t key) {
        return data_.at(key);
    }

    const std::unordered_map<int, memory> &get_args(int64_t key) const {
        return data_.at(key);
    }

    const std::unordered_map<int64_t, std::unordered_map<int, memory>> &
    get_args() const {
        return data_;
    }

    // for external memory
    const memory &get_external_input_mem(size_t index) {
        return external_input_mems_[index];
    }

    const memory &get_external_output_mem(size_t index) {
        return external_output_mems_[index];
    }

    const std::vector<memory> &get_external_input_mems() const {
        return external_input_mems_;
    }

    const std::vector<memory> &get_external_output_mems() const {
        return external_output_mems_;
    }

    void add_external_input_mem(const memory &mem) {
        external_input_mems_.emplace_back(mem);
    }

    void add_external_output_mem(const memory &mem) {
        external_output_mems_.emplace_back(mem);
    }

    const std::vector<memory> &get_internal_variable_mems() const {
        return internal_variable_mems_;
    }

    void add_internal_variable_mem(const memory &mem) {
        internal_variable_mems_.emplace_back(mem);
    }

    void add_value_mem_map(const std::pair<value_t *, memory> &map) {
        value_mem_map_.insert(map);
    }

    bool find_value_mem_map(value_t *key, memory &mem) const {
        auto pos = value_mem_map_.find(key);
        if (pos != value_mem_map_.end()) {
            mem = pos->second;
            return true;
        }
        return false;
    }

    void add_topo_ordered_key(int64_t key) {
        topo_ordered_keys_.emplace_back(key);
    }

    std::vector<int64_t> get_topo_ordered_keys() const {
        return topo_ordered_keys_;
    }

    void add_internal_constant_mem(const memory &mem) {
        internal_constant_mems_.emplace_back(mem);
    }

    const std::vector<memory> &get_internal_constant_mems() const {
        return internal_constant_mems_;
    }

private:
    std::unordered_map<int64_t, std::unordered_map<int, memory>> data_;
    int64_t counter {0};

    std::vector<memory> external_input_mems_;
    std::vector<memory> external_output_mems_;
    std::vector<memory> internal_variable_mems_;
    std::vector<memory> internal_constant_mems_;
    std::unordered_map<value_t *, memory> value_mem_map_; // pointer -> mem
    std::vector<int64_t> topo_ordered_keys_;
};

impl::status_t memory_binding(
        std::vector<std::shared_ptr<impl::op_t>> &subgraph,
        const std::vector<impl::logical_tensor_t> &inputs,
        const std::vector<impl::logical_tensor_t> &outputs,
        const dnnl::engine &p_engine, execution_args_mgr &exec_arg_mgr,
        primitive_attr_mgr &prm_attr_mgr);

class subgraph_resource_t {
public:
    subgraph_resource_t(const execution_args_mgr &exec_args_mgr)
        : exec_args_mgr_(exec_args_mgr) {
        for (int64_t key : exec_args_mgr_.get_topo_ordered_keys()) {
            exec_args_.emplace_back(exec_args_mgr_.get_args(key));
        }
    }

    const execution_args_mgr &get_exec_args_mgr() const {
        return exec_args_mgr_;
    }

    const std::vector<exec_args> &get_exec_args() const { return exec_args_; }

    execution_args_mgr &get_exec_args_mgr() { return exec_args_mgr_; }

    std::vector<exec_args> &get_exec_args() { return exec_args_; }

private:
    execution_args_mgr exec_args_mgr_;
    std::vector<exec_args> exec_args_;
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
