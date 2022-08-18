/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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
#include "dead_write_eliminate.hpp"
#include <vector>
#include "../visitor.hpp"
#include <compiler/ir/pass/dependency_analyzer.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/any_map.hpp>
#include <util/utils.hpp>

SC_MODULE(pass.dead_write_elim)

namespace sc {

// dead write elimination implementation
class dwe_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;
    std::unordered_set<expr_c> defined;

    // not interested in any exprs
    expr_c dispatch(expr_c v) override { return v; }

    stmt_c visit(define_c v) override {
        defined.insert(v->var_);
        return ir_visitor_t::visit(v);
    }

    stmt_c visit(assign_c v) override {
        if (dependency_analysis::get_dep_info(v.get()).depended_by_.size() == 0
                && v->var_.isa<indexing>()) {
            auto tsr = v->var_.static_as<indexing>()->ptr_.checked_as<tensor>();
            bool no_dead_write = tsr->attr_
                    && (tsr->attr_->get_or_else(attr_keys::no_dead_write, false)
                            || tsr->attr_->get_or_else(
                                    dependency_analysis::attr_directly_accessed,
                                    false));
            bool is_pointer_buffer = tsr->elem_dtype_.is_pointer();
            bool is_read_buffer = tsr->attr().get_or_else("read_buffer", false);
            bool is_tensor_local
                    = (is_read_buffer || defined.find(tsr) != defined.end())
                    && !is_pointer_buffer && !no_dead_write;

            // if it is a store to indexing, and the target tensor is local, and
            // no other stmt depends on it, we can delete it
            if (is_tensor_local) {
                SC_MODULE_INFO << "Remove " << v;
                if (is_read_buffer) {
                    SC_MODULE_WARN << "Writing on read-only buffer: " << v;
                }
                return make_stmt<stmts_node_t>(std::vector<stmt>());
            }
        }
        return v;
    }
};

func_c dead_write_eliminator_t::operator()(func_c f) {
    dependency_analyzer_t ana;
    f = ana(f);
    dwe_impl_t v;
    return v.dispatch(f);
}
} // namespace sc
