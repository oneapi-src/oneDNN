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
#include "node_uniquify.hpp"
#include <utility>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/pass/ir_copy_internal.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/any_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

class node_uniquifier_impl_t : public ir_visitor_t {
    // the map used in ir_copier_impl_t
    std::unordered_map<expr_c, expr> replace_map_;
    ir_copier_impl_t cpyer;
    // all currently met stmts
    std::unordered_set<stmt_c> met_stmts_;
    // all currently met exprs
    std::unordered_set<expr_c> met_exprs_;

public:
    using ir_visitor_t::dispatch;
    node_uniquifier_impl_t()
        : cpyer(replace_map_, /*create_var_tensor*/ false) {}
    expr_c dispatch(expr_c v) override {
        if (met_exprs_.find(v) != met_exprs_.end()) {
            // if it is met, copy it
            // if it is var/tensor, ir_copier_t will not copy it
            return cpyer.copy(v);
        }
        met_exprs_.insert(v);
        // dispatch down to check sub-nodes
        return ir_visitor_t::dispatch(std::move(v));
    }

    stmt_c dispatch(stmt_c v) override {
        if (met_stmts_.find(v) != met_stmts_.end()) {
            // it is is met, copy it
            return cpyer.copy(v);
        }
        met_stmts_.insert(v);
        // dispatch down to check sub-nodes
        return ir_visitor_t::dispatch(std::move(v));
    }
};

func_c node_uniquifier_t::operator()(func_c f) {
    node_uniquifier_impl_t impl;
    return impl.dispatch(std::move(f));
}
expr_c node_uniquifier_t::operator()(expr_c f) {
    node_uniquifier_impl_t impl;
    return impl.dispatch(std::move(f));
}
stmt_c node_uniquifier_t::operator()(stmt_c f) {
    node_uniquifier_impl_t impl;
    return impl.dispatch(std::move(f));
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
