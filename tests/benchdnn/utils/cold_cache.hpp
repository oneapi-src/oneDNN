/*******************************************************************************
* Copyright 2023 Intel Corporation
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

extern cold_cache_mode_t default_cold_cache_mode; // default cold cache mode
extern cold_cache_mode_t cold_cache_mode; // user cold cache mode

std::ostream &operator<<(std::ostream &s, cold_cache_mode_t cold_cache_mode);

struct cold_cache_t {
    // Default constructor to have an ability create cold_cache in std::vector.
    // Such cold_cache is always disabled.
    cold_cache_t();

    // Initializes a cold_cache object with extra memories to iterate over.
    // It identifies how many buffers must be created to avoid cache hits.
    // A memory heuristic relies on target total cache size: it is divided
    // evenly across arguments requested to be `cold`.
    //
    // In worst case scenario when hot arguments fully occupy memory pool limit
    // devoted for cold cache, an extra memory for cold arguments will still be
    // allocated.
    cold_cache_t(const std::vector<dnnl_exec_arg_t> &dnnl_args);

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
    bool enabled_;
    size_t n_buffers_top_limit_;
    size_t n_buffers_bottom_limit_;
    // `n_buffers` is responsible for the number of allocated buffers per arg.
    size_t n_buffers_;
    bool override_n_buffers_;
    std::unordered_map<int, std::vector<dnn_mem_t>> cache_;

    // Memory allocations are time consuming on GPU, thus, limiting number of
    // buffers from above.
    // Since `no_host_memory` allocations use `memset` call to initialize the
    // data, the assumption is it makes newly created memory objects get into
    // GPU cache. Using these memory objects in cold cache run will not be
    // "cold" any more.
    // Thus, an extra reorder is added with brand new memory objects to reset
    // the state of the cache.
    static constexpr size_t gpu_n_buffers_top_limit_ = 100;

    size_t cc_counter_ = 0;

    // Returns `true`, if "cold cache" was requested and eligible.
    bool use_cold_cache(const std::vector<dnnl_exec_arg_t> &dnnl_args);

    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(cold_cache_t);
};

#endif
