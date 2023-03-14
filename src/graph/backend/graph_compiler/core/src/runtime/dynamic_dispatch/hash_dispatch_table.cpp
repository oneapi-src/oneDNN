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

#include <assert.h>
#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <vector>
#include "hash_dispatch_table.hpp"
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {
hash_dispatch_table_t::hash_dispatch_table_t(uint32_t num_args, size_t capacity)
    : num_args_(num_args)
    , size_per_entry_(num_args * sizeof(uint64_t) + sizeof(void *))
    , buffer_ {new char[size_per_entry_ * capacity]}
    , capacity_(capacity) {
    assert((capacity_ & (capacity_ - 1)) == 0);
    memset(buffer_.get(), 0, size_per_entry_ * capacity);
}

static uint64_t hash(uint64_t x) {
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    x = x ^ (x >> 31);
    x = (x == 0) ? 1 : x;
    return x;
}

static uint64_t simple_hash_combine(uint64_t *keys, uint64_t num_keys) {
    uint64_t simple_hash = keys[0];
    for (uint64_t i = 1; i < num_keys; i++) {
        simple_hash = simple_hash * 0x9e3779b9 + keys[i];
    }
    return simple_hash;
}

static bool keys_equals(uint64_t *keys1, uint64_t *keys2, uint64_t num_args) {
    for (uint32_t i = 0; i < num_args; i++) {
        if (keys1[i] != keys2[i]) { return false; }
    }
    return true;
}

static bool keys_empty(uint64_t *keys1, uint64_t num_args) {
    for (uint32_t i = 0; i < num_args; i++) {
        if (keys1[i] != 0) { return false; }
    }
    return true;
}

void *hash_dispatch_table_t::get(uint64_t *keys, uint64_t num_keys) {
    assert(num_keys == num_args_);
    uint64_t simple_hash = simple_hash_combine(keys, num_args_);
    uint64_t hash_v = bound_by_capacity(hash(simple_hash));
    for (uint64_t i = 0; i < capacity_; i++) {
        auto idx = bound_by_capacity((i + hash_v));
        if (keys_equals(get_keys_by_idx(idx), keys, num_args_)) {
            return get_value_by_idx(idx);
        }
        if (keys_empty(get_keys_by_idx(idx), num_args_)) { return nullptr; }
    }

    return nullptr;
}

void hash_dispatch_table_t::set(
        uint64_t *keys, uint64_t num_keys, void *value) {
    assert(num_keys == num_args_);
    if (num_elems_ > capacity_ / 2) {
        // todo: rehash
        throw std::runtime_error("Rehash not implemented");
    }
    uint64_t simple_hash = simple_hash_combine(keys, num_args_);
    uint64_t hash_v = hash(simple_hash) & (~(-capacity_));

    for (uint64_t i = 0; i < capacity_; i++) {
        auto idx = bound_by_capacity(i + hash_v);
        auto cur_key = get_keys_by_idx(idx);
        if (keys_empty(cur_key, num_args_)) {
            for (size_t num = 0; num < num_args_; num++) {
                cur_key[num] = keys[num];
            }
            get_value_by_idx(idx) = value;
            num_elems_++;
            return;
        }
        if (keys_equals(cur_key, keys, num_args_)) {
            get_value_by_idx(idx) = value;
            return;
        }
    }
}

hash_dispatch_table_t::dispatch_func_t
hash_dispatch_table_t::get_dispatch_func() {
    return [](dispatch_table_t *ths, uint64_t *keys, uint64_t num_keys) {
        return static_cast<hash_dispatch_table_t *>(ths)->get(keys, num_keys);
    };
}

} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
