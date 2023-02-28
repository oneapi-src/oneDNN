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
#include <stdio.h>
#include <vector>
#include "dyn_dispatch_table.hpp"
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {
size_t dyn_dispatch_table_t::compute_linear_index(
        uint64_t *keys, uint64_t num_keys) const {
    uint64_t idx = 0;
    assert(number_of_args_ == num_keys);
    uint64_t cumulative_size = 1;
    uint64_t start = 0;
    for (uint64_t i = 0; i < number_of_args_; i++) {
        dispatch_key fmt = keys[i];
        uint32_t key = fmt.format_kind_;
        bool found = false;
        auto next_start = start + number_of_candidates_[i];
        for (int j = 0; j < number_of_candidates_[i]; j++) {
            auto &candidate = format_look_up_table_[j + start];
            if (candidate == key) {
                idx += j * cumulative_size;
                found = true;
                break;
            }
        }
        start = next_start;
        cumulative_size *= number_of_candidates_[i];

        assert(found);
        (void)found;
    }

    return idx * number_of_blocks_ + block_to_idx_(keys, num_keys);
}

void *dyn_dispatch_table_t::dispatch(
        dispatch_table_t *ths, uint64_t *keys, uint64_t num_keys) {
    auto thetable = static_cast<dyn_dispatch_table_t *>(ths);
    return thetable->table_[thetable->compute_linear_index(keys, num_keys)];
}

dyn_dispatch_table_t::dyn_dispatch_table_t(
        std::vector<format_arg_t> &&format_args,
        block_extract_func_t block_to_idx, size_t number_of_blocks)
    : block_to_idx_(block_to_idx)
    , number_of_blocks_(number_of_blocks)
    , number_of_args_(format_args.size()) {
    uint64_t cur_size = 1;
    int arg_idx = 0;
    int idx = 0;
    for (auto &args : format_args) {
        cur_size *= args.info_.size();
        for (auto &info : args.info_) {
            format_look_up_table_[idx] = info.key_;
            idx++;
        }
        number_of_candidates_[arg_idx] = args.info_.size();
        arg_idx++;
    }
    // calculate the total table size and reserve buffer
    table_.resize(cur_size * number_of_blocks);
}

// the unrolled loop to find v in
template <uint64_t idx, uint64_t total_size>
struct unrolled_find_t {
    static uint64_t call(uint32_t *cur, uint32_t v) {
        if (cur[idx] == v) { return idx; }
        return unrolled_find_t<idx + 1, total_size - 1>::call(cur, v);
    }
};

template <uint64_t idx>
struct unrolled_find_t<idx, 1> {
    static uint64_t call(uint32_t *cur, uint32_t v) {
        assert(cur[idx] == v);
        return idx;
    }
};

template <int idx, uint64_t cum_sum, uint64_t cum_product,
        int first_num_candidates, int... num_candidates>
struct semi_dyn_dispatch_t {
    static size_t call(dyn_dispatch_table_t *ths, uint64_t *keys) {
        auto ptr = unrolled_find_t<0, first_num_candidates>::call(
                &ths->format_look_up_table_[cum_sum],
                dispatch_key(keys[idx]).format_kind_);
        return ptr * cum_product
                + semi_dyn_dispatch_t<idx + 1, cum_sum + first_num_candidates,
                        cum_product * first_num_candidates,
                        num_candidates...>::call(ths, keys);
    }
};

template <int idx, uint64_t cum_sum, uint64_t cum_product,
        int first_num_candidates>
struct semi_dyn_dispatch_t<idx, cum_sum, cum_product, first_num_candidates> {
    static size_t call(dyn_dispatch_table_t *ths, uint64_t *keys) {
        auto ptr = unrolled_find_t<0, first_num_candidates>::call(
                &ths->format_look_up_table_[cum_sum],
                dispatch_key(keys[idx]).format_kind_);
        return ptr * cum_product;
    }
};

template <int... num_candidates>
static void *semi_dispatch(
        dispatch_table_t *ths, uint64_t *keys, uint64_t num_keys) {
    dyn_dispatch_table_t *table = static_cast<dyn_dispatch_table_t *>(ths);
    auto result_idx = semi_dyn_dispatch_t<0, 0, 1, num_candidates...>::call(
            table, keys);
    // assert(result_idx * table->number_of_blocks_
    //                 + table->block_to_idx_(keys, num_keys)
    //         == table->compute_linear_index(keys, num_keys));
    return table->table_[result_idx * table->number_of_blocks_
            + table->block_to_idx_(keys, num_keys)];
}

dyn_dispatch_table_t::dispatch_func_t
dyn_dispatch_table_t::get_dispatch_func() {
    if (number_of_args_ == 3
            && number_of_candidates_ == std::array<int, 4> {2, 2, 2, 0}) {
        return &semi_dispatch<2, 2, 2>;
    }
    return &dispatch;
};

} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
