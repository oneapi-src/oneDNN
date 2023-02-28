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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SSA_VISITOR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SSA_VISITOR_HPP
#include <vector>
#include "viewer.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

// The base class of all visitors for SSA IR. Developers who extend this visitor
// may override visit() as usual. The returned values of visit() for expr are
// expected to be simple indexing, constants, vars or tensors. To insert a SSA
// def before the current position, please use `add_def`. To insert after,
// please use `add_def_after_current_stmt`. Nested exprs should not be returned.
// Its difference between ir_visitor_t:
// 1. It enables default dispatching on phi node
// 2. It does not dispatch into the dims of tensor_node
// 3. It keeps expr_base::ssa_data_ consistent after dispatch for expr/stmt
// 4. dispatch() is hidden and users of the SSA visitor should use
// top_level_dispatch() instead, to initiate walking into the IR from top-level
// func/stmt. The developers of SSA visitor can still call old dispatch()
// internally
class ssa_visitor_t : public ir_visitor_t {
protected:
    std::vector<stmt_c> *current_scope_ = nullptr;
    // the stmts to be inserted after visit() of current stmt
    std::vector<stmt_c> stmts_to_insert_after;
    uint64_t var_def_idx_ = 0;
    expr_c dispatch(expr_c e) override;
    stmt_c dispatch(stmt_c e) override;

    define make_def(const expr_c &v);
    define make_def_and_process(const expr_c &v);

    // reference root of the expr nodes that has been visited during one call
    // of top_level_dispatch. will be cleared after top_level_dispatch returns
    std::vector<expr> gc_roots_;

    // starts to traverse from gc_roots_ and recursively mark the `referenced_`
    // bit in ssa_data_t. SSA var definitions that are not referenced will be
    // skipped in the next visitor's visit()
    void mark_garbage();

public:
    // Dispatch and walk down into IR tree from a top-level node. It will also
    // do garbage marking on expr nodes that are no longer referenced after
    // visiting the IR is over
    func_c top_level_dispatch(func_c e);
    stmt_c top_level_dispatch(stmt_c e);

    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    std::vector<stmt_c> *get_current_scope() const noexcept {
        return current_scope_;
    }

    // acccepts an simple expr like "a+b", and puts the SSA var def "c=a+b" into
    // current scope
    expr add_def(const expr_c &v);
    // acccepts an simple expr like "a+b", and puts the SSA var def "c=a+b" into
    // current scope after current visit() of stmt returns
    expr add_def_after_current_stmt(const expr_c &v);
    stmt_c visit(stmts_c v) override;
    expr_c visit(ssa_phi_c v) override;
    expr_c visit(tensor_c v) override;
};

// The base class of all viewers for SSA IR. Its difference between ir_viewer_t:
// 1. It enables default dispatching on phi node
// 2. It will NOT dispatch into the dimensions of tensor_nodes
class ssa_viewer_t : public ir_viewer_t {
public:
    using ir_viewer_t::view;
    void view(ssa_phi_c v) override;
    void view(tensor_c v) override;
    void view(stmts_c v) override;
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
