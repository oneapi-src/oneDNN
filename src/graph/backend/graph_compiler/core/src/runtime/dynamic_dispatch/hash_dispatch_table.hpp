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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_HASH_DISPATCH_TABLE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_HASH_DISPATCH_TABLE_HPP

#include <memory>
#include "dispatch_table.hpp"
#include <runtime/dispatch_key.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {

/**
 * The dispatch table implemented by Open Addressing hash map.
 * @param num_args the number of input args to hash as the key
 * @param capacity the capacity of the hash map, must be power of 2
 * */
struct hash_dispatch_table_t : public dispatch_table_t {
    const uint32_t num_args_;
    const uint32_t size_per_entry_;
    // [entry 1: [key1, key2, ... keyN], value], [entry 2: [key1, key2, ...
    // keyN], value]
    std::unique_ptr<char[]> buffer_;
    // the capacity of the hash bucket. Must be power of 2
    size_t capacity_;
    size_t num_elems_ = 0;

    hash_dispatch_table_t(uint32_t num_args, size_t capacity);

    uint64_t *get_keys_by_idx(uint64_t idx) {
        return reinterpret_cast<uint64_t *>(
                buffer_.get() + size_per_entry_ * idx);
    }

    void *&get_value_by_idx(uint64_t idx) {
        return *reinterpret_cast<void **>(buffer_.get() + size_per_entry_ * idx
                + num_args_ * sizeof(uint64_t));
    }
    // returns v % capacity_
    size_t bound_by_capacity(size_t v) { return v & (~(-(int64_t)capacity_)); }

    void *get(uint64_t *keys, uint64_t num_keys) final override;
    void set(uint64_t *keys, uint64_t num_keys, void *value) final override;

    dispatch_func_t get_dispatch_func() final override;
};

} // namespace runtime

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
