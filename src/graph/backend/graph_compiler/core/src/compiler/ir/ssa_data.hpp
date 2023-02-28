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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SSA_DATA_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SSA_DATA_HPP

#include <assert.h>
#include <memory>
#include <utility>
#include "sc_stmt.hpp"
#include <util/weakptr_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

// the additional data for SSA exprs
struct ssa_data_t {
    // the SSA value owner node. Can be define_node or for_node for var
    std::weak_ptr<stmt_base_t> owner_;
    // if it is a var node and is a global variable, it can be assigned multiple
    // times
    bool is_global_ = false;
    bool is_param_ = false;

    // if this expr is referenced by a top-level stmt like evaluate/for/if
    bool referenced_ = false;

    ssa_data_t() = default;

    // gets the value of a var node
    expr get_value_of_var() const;

    // gets the value of a var node
    expr get_value_of_var_nothrow() const;

    // check if this var is locally defined
    bool is_local() { return !is_global_ && !is_param_; }
    // check if this expr is no longer used
    bool is_garbage() { return !is_global_ && !referenced_; }

    bool has_owner() const { return !utils::is_uninitialized_weakptr(owner_); }

    stmt get_owner() const {
        if (utils::is_uninitialized_weakptr(owner_)) { return stmt(); }
        auto ret = owner_.lock();
        assert(ret);
        return stmt(std::move(ret));
    }
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
