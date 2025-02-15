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

#include "common.hpp"

#include "utils/impl_filter.hpp"

impl_filter_t global_impl_filter {};

const impl_filter_t &get_prim_ref_impl_filter() {
    static const impl_filter_t prim_ref_impl_filter({"ref:any", "ref_int8:any"},
            /* use_impl = */ false,
            /* respect_global_filter = */ false);
    return prim_ref_impl_filter;
}

std::string get_impl_filter_name(const impl_filter_t &impl_filter) {
    const bool use_impl = impl_filter.use_impl();
    std::string prefix = !use_impl ? "skip-" : "";
    std::string s("--");
    s.append(prefix).append("impl");
    return s;
}

// This operator takes the global filter into account as well. No need to dump
// it additionally in a common spot.
std::ostream &operator<<(std::ostream &s, const impl_filter_t &impl_filter) {
    const bool is_global_def = IMPLICATION(
            impl_filter.respect_global_filter(), global_impl_filter.is_def());
    const bool is_def = is_global_def && impl_filter.is_def();
    if (is_def) return s;

    const auto &option_name = !is_global_def
            ? get_impl_filter_name(global_impl_filter)
            : get_impl_filter_name(impl_filter);
    s << option_name << "=";

    const auto &names = !is_global_def ? global_impl_filter.get_names()
                                       : impl_filter.get_names();
    const size_t sz = names.size();
    for (size_t i = 0; i < sz - 1; i++) {
        s << names[i] << ",";
    }
    s << names[sz - 1] << " ";

    return s;
}

bool need_next_impl(
        const std::string &impl_name, const impl_filter_t &impl_filter) {
    const bool is_global_def = IMPLICATION(
            impl_filter.respect_global_filter(), global_impl_filter.is_def());
    const bool is_def = is_global_def && impl_filter.is_def();
    if (is_def) return false;

    const bool use_impl = !is_global_def ? global_impl_filter.use_impl()
                                         : impl_filter.use_impl();
    const auto &names = !is_global_def ? global_impl_filter.get_names()
                                       : impl_filter.get_names();

    // If the name hits the list and `use_impl_=true`, no need the next impl.
    // If the name hits the list and `use_impl_=false`, needs the next impl.
    // If the name doesn't hit the list and `use_impl_=true`, needs the next impl.
    // If the name doesn't hit the list and `use_impl_=false, no need the next impl.
    for (const auto &e : names) {
        if (e.empty()) continue; // Just in case though not expected.
        if (impl_name.find(e) != std::string::npos) return !use_impl;
    }
    return use_impl;
}
