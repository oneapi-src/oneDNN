/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef UTILS_SUMMARY_HPP
#define UTILS_SUMMARY_HPP

// Responsible for printing service information.
struct summary_t {
    // Prints up to 10 failed cases reproducers at the end of the run.
    bool failed_cases = true;
    // Prints statistics about implementations used over the run.
    // CSV may be added on top of a table for mass search from distributed
    // automated systems.
    bool impl_names = true;
    bool impl_names_csv = false;
    // Reports performance-related system settings when they are not set to
    // values providing stable performance measurements. Inspects CPU settings
    // (frequency, Turbo mode, and C-states) or GPU settings (per-tile
    // frequency) depending on the engine.
    bool perf_hints = true;
};

extern summary_t summary;

void print_impl_names_summary();
void print_impl_names_csv_summary();

#endif
