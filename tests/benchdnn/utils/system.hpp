/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef UTILS_SYSTEM_HPP
#define UTILS_SYSTEM_HPP

#include <cstddef>

struct cpu_cache_args_t {
    size_t L2_size = 0;
    size_t L3_size = 0; // = L3_per_core
    size_t num_cores = 0;
    size_t total_socket_size = 0; // (L2 + L3_per_core) * num_cores
};

size_t get_cpu_ram_size();
int get_gpu_ram_sizes(size_t &ram_size, size_t &max_alloc_size);
int get_cpu_cache_size(cpu_cache_args_t &cache_args);
int get_gpu_cache_size(size_t &cache_size);

// Reports performance-related system settings when they are not set to values
// providing stable performance measurements. Dispatches to engine-specific
// checks: CPU (min/max frequency, Turbo mode, and C-states) or GPU (per-tile
// min/max frequency). Controlled by the `--summary=[no-]perf-hints` knob.
// Prints the hints at most once per run.
void report_perf_hints();

#endif
