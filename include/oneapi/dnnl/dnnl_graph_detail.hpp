/*******************************************************************************
* Copyright 2020 Intel Corporation
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

/// @file
/// Utilities for C++ APIs

#ifndef ONEAPI_DNNL_DNNL_GRAPH_DETAIL_HPP
#define ONEAPI_DNNL_DNNL_GRAPH_DETAIL_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "oneapi/dnnl/dnnl_graph.h"

/// @addtogroup dnnl_graph_api
/// @{

namespace dnnl {
namespace graph {

/// @addtogroup dnnl_graph_api_utils
/// @{

namespace detail {

template <typename T, dnnl_graph_result_t (*del)(T *)>
class handle {
public:
    static constexpr auto default_del = del;

    /// Creates an empty wrapper for underlying C API handle
    handle() = default;
    virtual ~handle() = default;

    /// Custom constrcutor
    ///
    /// @param t Raw pointer to the C API handle
    /// @param weak A flag which indicates whether this wrapper
    ///     is a weak pointer
    handle(T *t, bool weak = false) { reset(t, weak); }

    /// Copy constructor
    handle(const handle &) = default;
    /// Copy assig constructor
    handle &operator=(const handle &) = default;
    /// Move constructor
    handle(handle &&) = default;
    /// Move assign constructor
    handle &operator=(handle &&) = default;

    /// Resets the handle wrapper ojects to wrap a new C API handle
    ///
    /// @param t The raw pointer of C API handle
    /// @param weak A flag which indicates whether this wrapper is a
    ///     weak pointer
    void reset(T *t, bool weak = false) {
        data_.reset(t, weak ? dummy_del : default_del);
    }

    /// Returns the underlying C API handle
    ///
    /// @returns The underlying C API handle
    T *get() const { return data_.get(); }

private:
    std::shared_ptr<T> data_ {0};
    /// Dummy destrcutor
    static dnnl_graph_result_t dummy_del(T *) {
        return dnnl_graph_result_success;
    }
};

#define DNNL_GRAPH_HANDLE_ALIAS(type) \
    using type##_handle = detail::handle<dnnl_graph_##type##_t, \
            dnnl_graph_##type##_destroy>

DNNL_GRAPH_HANDLE_ALIAS(allocator);
DNNL_GRAPH_HANDLE_ALIAS(engine);
DNNL_GRAPH_HANDLE_ALIAS(graph);
DNNL_GRAPH_HANDLE_ALIAS(op);
DNNL_GRAPH_HANDLE_ALIAS(stream_attr);
DNNL_GRAPH_HANDLE_ALIAS(stream);
DNNL_GRAPH_HANDLE_ALIAS(tensor);
DNNL_GRAPH_HANDLE_ALIAS(thread_pool);
DNNL_GRAPH_HANDLE_ALIAS(compiled_partition);
DNNL_GRAPH_HANDLE_ALIAS(partition);

#undef DNNL_GRAPH_HANDLE_ALIAS

#define REGISTER_SYMBOL(s) #s,
const static std::vector<std::string> op_kind_strings
        = {DNNL_GRAPH_FORALL_BUILDIN_OPS(REGISTER_SYMBOL) "LastSymbol"};

#undef REGISTER_SYMBOL

} // namespace detail

/// @} dnnl_graph_api_utils

} // namespace graph
} // namespace dnnl

/// @} dnnl_graph_api

#endif
