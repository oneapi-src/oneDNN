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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_LOGGING_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_LOGGING_HPP
#include <sstream>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {
struct SC_INTERNAL_API logging_stream_t {
    std::ostream *stream_;
    const char *append_;
    logging_stream_t() : stream_(nullptr), append_(nullptr) {}
    logging_stream_t(const char *append);
    logging_stream_t(logging_stream_t &&other) {
        stream_ = other.stream_;
        other.stream_ = nullptr;
        append_ = other.append_;
    }
    ~logging_stream_t();
    operator bool() const { return stream_; };
};

SC_INTERNAL_API logging_stream_t get_info_logging_stream(
        const char *module_name = nullptr);
SC_INTERNAL_API logging_stream_t get_warning_logging_stream(
        const char *module_name = nullptr);
SC_INTERNAL_API logging_stream_t get_fatal_logging_stream(
        const char *module_name = nullptr);

enum verbose_level { FATAL = 0, WARNING, INFO };

void set_logging_stream(std::ostream *s);
std::ostream *get_logging_stream();
} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#define SC_INFO \
    if (auto __sc_stream_temp__ \
            = ::dnnl::impl::graph::gc::runtime::get_info_logging_stream()) \
    (*__sc_stream_temp__.stream_)

#define SC_MODULE_INFO2(NAME) \
    if (auto __sc_stream_temp__ \
            = ::dnnl::impl::graph::gc::runtime::get_info_logging_stream(NAME)) \
    (*__sc_stream_temp__.stream_)

#define SC_MODULE_INFO SC_MODULE_INFO2(__sc_module_name)

#define SC_WARN \
    if (auto __sc_stream_temp__ \
            = ::dnnl::impl::graph::gc::runtime::get_warning_logging_stream()) \
    (*__sc_stream_temp__.stream_)
#define SC_MODULE_WARN \
    if (auto __sc_stream_temp__ \
            = ::dnnl::impl::graph::gc::runtime::get_warning_logging_stream( \
                    __sc_module_name)) \
    (*__sc_stream_temp__.stream_)

#define SC_FATAL \
    if (auto __sc_stream_temp__ \
            = ::dnnl::impl::graph::gc::runtime::get_fatal_logging_stream()) \
    (*__sc_stream_temp__.stream_)
#define SC_MODULE_FATAL \
    if (auto __sc_stream_temp__ \
            = ::dnnl::impl::graph::gc::runtime::get_fatal_logging_stream( \
                    __sc_module_name)) \
    (*__sc_stream_temp__.stream_)

#define SC_MODULE(NAME) static constexpr const char *__sc_module_name = #NAME;

#endif
