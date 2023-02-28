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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_SCOPED_TIMER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_SCOPED_TIMER_HPP
#include <chrono>
#include <utility>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace utils {

using time_point = std::chrono::high_resolution_clock::time_point;
using time_duration = std::chrono::high_resolution_clock::duration;

// the util timer which will print the time elapsed of the lifetime of the
// object.
template <typename T>
struct scoped_timer : T {
    time_point start_time_;
    scoped_timer(bool enabled, T &&v) : T(std::forward<T>(v)) {
        if (enabled) {
            start_time_ = std::chrono::high_resolution_clock::now();
        }
    }
    scoped_timer(const scoped_timer &) = delete;
    scoped_timer(scoped_timer &&other)
        : T(std::move(other)), start_time_(other.start_time_) {
        other.start_time_ = time_point {};
    }

    ~scoped_timer() {
        if (start_time_ != time_point {}) {
            auto duration
                    = std::chrono::high_resolution_clock::now() - start_time_;
            T::operator()(duration);
        }
    }
};

template <typename T>
inline scoped_timer<T> create_scoped_timer(bool enabled, T &&func) {
    return scoped_timer<T>(enabled, std::move(func));
}

#define SC_SCOPED_TIMER_INFO(name, postfix) \
    ::dnnl::impl::graph::gc::utils::create_scoped_timer( \
            ::dnnl::impl::graph::gc::utils::compiler_configs_t::get() \
                    .print_pass_time_, \
            [](::dnnl::impl::graph::gc::utils::time_duration dur) { \
                SC_MODULE_INFO2(name) \
                        << "took " \
                        << std::chrono::duration_cast< \
                                   std::chrono::microseconds>(dur) \
                                   .count() \
                        << "us" postfix; \
            });

} // namespace utils
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
