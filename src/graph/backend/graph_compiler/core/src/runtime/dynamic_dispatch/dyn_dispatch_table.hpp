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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_DYN_DISPATCH_TABLE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_DYN_DISPATCH_TABLE_HPP

#include <array>
#include <vector>
#include "dispatch_table.hpp"
#include <runtime/dispatch_key.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {

/**
 * The convertion table to map a list of formats to a single linear index. The
 * blocking of the formats must be in [16,32,48,64]. The table has a fixed
 * number of format_args, and the user needs to input the same number of formats
 * at the run time to dispatch the list of formats. The "fixed" here means
 * "fixed" since construction of this table.
 *
 * The candidates of a format arg should be sorted to "most likely used" first.
 * So that it will speedup finding the format.
 *
 * This dispatch table also provides specialized dispatch function for some
 * combinations of candidate numbers: 2x2x2
 *
 * If the candidate numbers are these, it will use the fast dispatch function.
 *
 * @arg format_args the vector of format_arg_t. Each element for an input
 * format. A format_arg_t contains the format kind candidates for the format
 * argument.
 *
 * */
struct dyn_dispatch_table_t : public dispatch_table_t {
    struct format_candidate_t {
        uint32_t key_;
        format_candidate_t() = default;
        format_candidate_t(uint64_t format_kind)
            : key_(static_cast<uint32_t>(format_kind)) {}
    };
    struct format_arg_t {
        std::vector<format_candidate_t> info_;
    };
    using block_extract_func_t
            = uint64_t (*)(uint64_t *keys, uint64_t num_keys);
    std::array<uint32_t, 8> format_look_up_table_;
    // the function to convert block numbers in format keys into an integer in
    // [0, number_of_blocks-1]
    block_extract_func_t block_to_idx_;
    size_t number_of_blocks_;
    std::vector<void *> table_;
    size_t number_of_args_;
    std::array<int, 4> number_of_candidates_;

    size_t compute_linear_index(uint64_t *keys, uint64_t num_keys) const;
    void *get(uint64_t *keys, uint64_t num_keys) final override {
        return table_[compute_linear_index(keys, num_keys)];
    }

    void set(uint64_t *keys, uint64_t num_keys, void *value) final override {
        table_[compute_linear_index(keys, num_keys)] = value;
    }

    dyn_dispatch_table_t(std::vector<format_arg_t> &&format_args,
            block_extract_func_t block_to_idx, size_t number_of_blocks);
    static void *dispatch(
            dispatch_table_t *ths, uint64_t *keys, uint64_t num_keys);

    dispatch_func_t get_dispatch_func() final override;
};

} // namespace runtime

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
