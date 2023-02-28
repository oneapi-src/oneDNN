/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_CONTENT_HASH_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_CONTENT_HASH_HPP

#include <vector>
#include "ir_comparer.hpp"
#include "sc_expr.hpp"
#include <unordered_map>
#include <util/compiler_macros.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
template <typename T>
struct content_hash_t {};

template <>
struct content_hash_t<constant_c> {
    std::size_t operator()(const constant_c &k) const;
};

template <>
struct content_hash_t<expr> {
    std::size_t operator()(const expr &k) const;
};

template <>
struct content_hash_t<expr_c> {
    std::size_t operator()(const expr_c &k) const;
};

template <typename T>
struct content_hash_t<std::vector<T>> {
    std::size_t operator()(const std::vector<T> &k) const {
        auto h = content_hash_t<T>();
        std::size_t ret = 0;
        for (auto &v : k) {
            ret = ret * 23 + h(v);
        }
        return ret;
    }
};

template <typename T>
struct content_equals_t {};

template <>
struct content_equals_t<expr_c> {
#if !SC_GNUC_VERSION_LT(7) && !defined(_MSC_VER)
    // Old version of gcc will produce error upon content_hash_map construction
    ir_comparer cmper_;
#endif
    bool operator()(const expr_c &a, const expr_c &b) const;
};

template <typename T, typename TBase>
struct content_equals_t<node_ptr<T, TBase>>
    : content_equals_t<node_ptr<const TBase, TBase>> {};

template <typename T>
struct content_equals_t<std::vector<T>> {
    bool operator()(const std::vector<T> &a, const std::vector<T> &b) const {
        content_equals_t<T> eq;
        if (a.size() != b.size()) { return false; }
        for (unsigned i = 0; i < a.size(); i++) {
            if (!eq(a[i], b[i])) return false;
        }
        return true;
    }
};

template <typename T, typename V>
using content_hash_map
        = std::unordered_map<T, V, content_hash_t<T>, content_equals_t<T>>;
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
