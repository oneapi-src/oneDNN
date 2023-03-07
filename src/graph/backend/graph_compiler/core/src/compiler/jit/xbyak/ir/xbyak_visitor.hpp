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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_XBYAK_VISITOR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_XBYAK_VISITOR_HPP

#include <compiler/ir/visitor.hpp>

#include "xbyak_expr.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

class xbyak_visitor_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::dispatch_expr_vector;
    using ir_visitor_t::visit;

    stmt_c visit(stmts_c v) override;
    stmt_c visit(for_loop_c v) override;

    expr_c visit(tensor_c v) override;
    expr_c visit(low_level_intrin_c v) override;

    virtual expr_c visit(xbyak_intrin_c v);

protected:
    uint64_t loop_depth() { return loop_depth_; }
    const stmt_base_t *current_scope() { return current_scope_; }

private:
    uint64_t loop_depth_ = 0;
    const stmt_base_t *current_scope_ = nullptr;
};

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
