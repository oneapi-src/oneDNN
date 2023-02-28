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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_STATIC_DISPATCH_TABLE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_STATIC_DISPATCH_TABLE_HPP

#include <array>
#include <assert.h>
#include "dispatch_table.hpp"
#include <runtime/dispatch_key.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {

template <uint64_t... values>
struct static_dispatch_keys {
    using linearizer = dispatch_key::linear_converter<values...>;
    static constexpr int num_of_values = linearizer::idx + 1;
};

template <int index, typename First, typename... Keys>
struct static_dispatch_trait {
    using next_trait = static_dispatch_trait<index + 1, Keys...>;
    static constexpr int value = First::num_of_values * next_trait::value;
    static constexpr int num_keys = next_trait::num_keys + 1;
    static uint64_t compute(uint64_t *keys) {
        return First::linearizer::call(dispatch_key(keys[index]))
                * next_trait::value
                + next_trait::compute(keys);
    }
};

template <int index, typename First>
struct static_dispatch_trait<index, First> {
    static constexpr int value = First::num_of_values;
    static constexpr int num_keys = 1;
    static uint64_t compute(uint64_t *keys) {
        return First::linearizer::call(dispatch_key(keys[index]));
    }
};

template <typename block_index_compute, uint64_t blocks, typename... Keys>
struct static_dispatch_table_t : public dispatch_table_t {
    using trait = static_dispatch_trait<0, Keys...>;
    static constexpr int size = trait::value * blocks;
    std::array<void *, size> table_;
    static size_t compute_linear_index(uint64_t *keys, uint64_t num_keys) {
        assert(num_keys == trait::num_keys);
        return trait::compute(keys) * blocks
                + block_index_compute::call(keys, num_keys);
    }

    static void *dispatch(
            dispatch_table_t *ths, uint64_t *keys, uint64_t num_keys) {
        auto index = compute_linear_index(keys, num_keys);
        return static_cast<static_dispatch_table_t *>(ths)->table_[index];
    }

    void *get(uint64_t *keys, uint64_t num_keys) override {
        return table_[compute_linear_index(keys, num_keys)];
    }
    void set(uint64_t *keys, uint64_t num_keys, void *value) override {
        table_[compute_linear_index(keys, num_keys)] = value;
    }
    dispatch_func_t get_dispatch_func() override { return &dispatch; }
};
} // namespace runtime

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
