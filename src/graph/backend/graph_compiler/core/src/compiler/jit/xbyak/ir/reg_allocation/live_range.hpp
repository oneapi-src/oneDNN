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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_REG_ALLOCATION_LIVE_RANGE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_REG_ALLOCATION_LIVE_RANGE_HPP

#include <algorithm>
#include <iostream>
#include <vector>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

using stmt_index_t = int64_t;

namespace stmt_index_const {
constexpr stmt_index_t increment = 4;
}

/* *
 * Live range representation for expr, records start and end point for
 * virtual_reg based on its def-use at stmt_index
 * */
struct live_range_t {
    bool defined_ = false;

    stmt_index_t start_ = -1;
    stmt_index_t end_ = -1;

    live_range_t() = default;

    live_range_t(stmt_index_t start)
        : defined_(true), start_(start), end_(start) {}
    live_range_t(stmt_index_t start, stmt_index_t end)
        : defined_(true), start_(start), end_(end) {}

    void update(stmt_index_t index) { end_ = index; }
    void update(stmt_index_t init_index, stmt_index_t index) {
        start_ = (init_index < start_) ? init_index : start_;
        end_ = index;
    }

    bool empty() const { return start_ == end_; }

    bool intersects(const live_range_t &b) const {
        return std::max(start_, b.start_) < std::min(end_, b.end_);
    }

    bool encompasses(const live_range_t &b) const {
        return (start_ < b.start_) && (end_ > b.end_);
    }

    bool enclose(const stmt_index_t &i) const {
        return (start_ < i) && (end_ > i);
    }

    friend std::ostream &operator<<(std::ostream &os, const live_range_t &m) {
        return os << "[" << m.start_ << ", " << m.end_ << "]";
    }
};

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
