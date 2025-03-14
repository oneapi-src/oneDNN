/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef UTILS_IMPL_FILTER_HPP
#define UTILS_IMPL_FILTER_HPP

#include <string>
#include <vector>

struct impl_filter_t {
    impl_filter_t() = default;
    impl_filter_t(const std::vector<std::string> &impl_names, bool use_impl,
            bool respect_global_filter)
        : impl_names_(impl_names)
        , use_impl_(use_impl)
        , respect_global_filter_(respect_global_filter) {}

    bool is_def() const { return impl_names_.empty(); }

    const std::vector<std::string> &get_names() const { return impl_names_; }
    bool use_impl() const { return use_impl_; }
    bool respect_global_filter() const { return respect_global_filter_; }

private:
    std::vector<std::string> impl_names_;
    bool use_impl_ = false; // `true` to `--impl`, `false` to `--skip-impl`.
    // Test objects should respect the global filter. CPU prim_ref objects
    // shouldn't as it affects correctness validation speed.
    // Default is set to `true` for the cases when global is initialized. In
    // such cases local is not initialized but always passed to `fetch_impl`,
    // thus, to pick up values from global it should indicate the global is
    // respected.
    bool respect_global_filter_ = true;
};

extern impl_filter_t global_impl_filter;

// Fixed filter to remove running reference impls for prim_ref support.
const impl_filter_t &get_prim_ref_impl_filter();

std::ostream &operator<<(std::ostream &s, const impl_filter_t &impl_filter);

// Returns `false`, (or use currently fetched implementation) when:
// * `impl_filter` is empty;
// * None of `use_impl_=false` hits happened;
// * One of `use_impl_=true` hits happened;
// Otherwise, returns `true`, meaning the next implementation is desired.
bool need_next_impl(
        const std::string &impl_name, const impl_filter_t &impl_filter);

#endif
