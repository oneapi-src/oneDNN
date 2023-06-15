/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_OP_DISPATCH_TABLES_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_OP_DISPATCH_TABLES_HPP
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <util/reflection.hpp>

#include "dispatch_table.hpp"
#include "hash_dispatch_table.hpp"
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
struct jit_module;

namespace runtime {
struct op_dispatch_tables_t {
    // format table, fmt0, unknown, unknown => fmt0, fmt1, fmt2. Currently we
    // use hash table.
    std::unique_ptr<hash_dispatch_table_t> format_table_;
    // impl kind table, configs => impl kind
    std::unique_ptr<dispatch_table_t> impl_kind_table_;
    // kernel table, fmt0, fmt1, fmt2 => kernel.
    std::unique_ptr<dispatch_table_t> kernel_table_;
    // pointer to kernel dispatch function.
    dispatch_table_t::dispatch_func_t kernel_dispatch_func_ = nullptr;
    reflection::shared_general_object_t op_info_;
    std::vector<std::shared_ptr<jit_module>> compiled_modules_;
    op_dispatch_tables_t() = default;
    virtual ~op_dispatch_tables_t();
    void set_format_table_keys(uint64_t *keys, uint64_t num_keys,
            uint64_t *values, uint64_t num_values);
    void set_impl_kind_table_keys(uint64_t *keys, uint64_t num_keys, int value);
    void set_kernel_table(
            std::unique_ptr<dispatch_table_t> &&kernel_table_ptr) {
        kernel_table_ = std::move(kernel_table_ptr);
    }
    void set_kernel_dispatch_func(dispatch_table_t::dispatch_func_t p) {
        kernel_dispatch_func_ = p;
    }

private:
    std::vector<std::unique_ptr<uint64_t[]>> format_values_;
    std::vector<int *> impl_kind_values_;
};
using op_dispatch_tables_ptr = std::shared_ptr<op_dispatch_tables_t>;
// the map for dispatch tables: global table var => op dispatch table
using dispatch_table_map_t
        = std::unordered_map<std::string, op_dispatch_tables_ptr>;

} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
