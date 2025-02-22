/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef UTILS_COLD_CACHE_HPP
#define UTILS_COLD_CACHE_HPP

#include <ostream>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_types.h"

#include "dnnl_memory.hpp"

enum class cold_cache_mode_t : unsigned {
    // Cold cache is disabled.
    none = 0x0,
    // Cold cache is enabled for weights execution argument.
    wei = 0x1,
    // Cold cache is enabled for all execution arguments.
    all = 0x2,
    // Cold cache is enabled for custom execution arguments, which must be
    // specified directly in code.
    custom = 0x4,
};

// User's choices for enabling cold-cache.
struct cold_cache_input_t {
    // Requested mode.
    cold_cache_mode_t cold_cache_mode_ = cold_cache_mode_t::none;
    // Optional cold TLB (Translation Lookaside Buffer) enabling.
    bool cold_tlb_ = false;
    // If TLB is enabled, the size of extra memory to touch.
    // The less memory used, the faster the execution should be, but the effect
    // of cold TLB might not be observed until a certain amount of memory
    // touched to thrash TLB. This amount is system dependent.
    //
    // Keep string to return it to the user in the repro line.
    std::string cold_tlb_size_str_ = "1.0G";
    // Countable value of the string stored to use inside the implementation.
    size_t cold_tlb_size_ = 1024 * 1024 * 1024;

    bool operator==(const cold_cache_input_t &other) const {
        // Don't compare `cold_tlb_size_` as it's the product of
        // `cold_tlb_size_str_`.
        return cold_cache_mode_ == other.cold_cache_mode_
                && cold_tlb_ == other.cold_tlb_
                && cold_tlb_size_str_ == other.cold_tlb_size_str_;
    }
    bool operator!=(const cold_cache_input_t &other) const {
        return !operator==(other);
    }
};

extern cold_cache_input_t cold_cache_input;

const cold_cache_input_t &default_cold_cache_input();

std::ostream &operator<<(std::ostream &s, cold_cache_mode_t cold_cache_mode);
std::ostream &operator<<(
        std::ostream &s, const cold_cache_input_t &cold_cache_input);

struct cold_cache_t {
    // Default constructor to have an ability create cold_cache in std::vector.
    // Such cold_cache is always disabled.
    cold_cache_t() = default;

    // Initializes a cold_cache object with extra memories to iterate over.
    // It identifies how many buffers must be created to avoid cache hits.
    // A memory heuristic relies on target total cache size: it is divided
    // evenly across arguments requested to be `cold`.
    //
    // In worst case scenario when hot arguments fully occupy memory pool limit
    // devoted for cold cache, an extra memory for cold arguments will still be
    // allocated.
    cold_cache_t(const std::vector<dnnl_exec_arg_t> &dnnl_args,
            dnnl_stream_t stream);

    ~cold_cache_t();

    // Move-assignment operator goes in pair with a default constructor.
    cold_cache_t &operator=(cold_cache_t &&rhs);

    // Takes arguments passed to execute function in a hot-loop and updates
    // memory pointers to ones from cold cache.
    // Returns `true` when:
    // * Cold cache is disabled.
    // * Cold cache is enabled and update was successful.
    // Otherwise, return `false`.
    bool update_dnnl_args(std::vector<dnnl_exec_arg_t> &dnnl_args);

    // Informs if cold cache spent all its resources when they were limited.
    bool should_stop() const;

private:
    cold_cache_input_t cold_cache_input_;
    bool enabled_ = false;
    size_t n_buffers_top_limit_ = 0;
    size_t n_buffers_bottom_limit_ = 0;
    // `n_buffers` is responsible for the number of allocated buffers per arg.
    size_t n_buffers_ = 0;
    bool override_n_buffers_ = false;
    std::unordered_map<int, std::vector<dnn_mem_t>> cache_;

    // Memory allocations are time consuming on GPU, thus, introducing the
    // upper bound for the number of buffers in cold-cache.
    // Since `no_ref_memory` allocations use `memset` call to initialize the
    // data, the assumption is it makes newly created memory objects with newly
    // allocated buffer underneath get into the GPU cache. Using these memory
    // objects in cold-cache run won't be "cold" any longer.
    // Thus, introducing an extra reorder with brand new memory objects which
    // sole purpose is to reset the state of the cache by entirely thrashing it.
    static constexpr size_t gpu_n_buffers_top_limit_ = 100;

    size_t cc_counter_ = 0;

    // Returns `true`, if cold-cache was requested and eligible.
    bool use_cold_cache(const std::vector<dnnl_exec_arg_t> &dnnl_args) const;

    int thrash_reorder(size_t mem_size, size_t granularity) const;

    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(cold_cache_t);
};

#endif
