/*******************************************************************************
 * Copyright 2023 Intel Corporation
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
#include "loop_split.hpp"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/pass_id.hpp>
#include <compiler/ir/visitor.hpp>
#include <unordered_set>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(loop_splitter, SC_PASS_DEPENDS_ON(),
        SC_PASS_REQUIRE_STATE(CONST_FOLDED, IR_SIMPLIFIED),
        SC_PASS_REQUIRE_NOT_STATE(), SC_PASS_SET_STATE(),
        SC_PASS_UNSET_STATE());

class loop_spiltter_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    inline int64_t divide_and_ceil(int64_t x, int64_t y) {
        return (x + y - 1) / y;
    }

    // not interested in any exprs
    expr_c dispatch(expr_c v) override { return v; }

    stmt_c visit(for_loop_c v) override {
        auto vv = ir_visitor_t::visit(std::move(v)).static_as<for_loop_c>();
        // New range for this for loop
        stmt new_body;
        auto for_range_new_end = [&](int64_t new_end) -> stmt_c {
            const auto end = vv->iter_end_.dyn_as<constant>();
            if (end.defined() && (get_const_as_int(end) > new_end)) {
                auto iter_end = builder::make_constant(
                        {new_end}, vv->iter_end_->dtype_);
                return copy_attr(*vv,
                        make_stmt<for_loop_node_t>(vv->var_, vv->iter_begin_,
                                iter_end, vv->step_, std::move(new_body),
                                vv->incremental_, vv->kind_, vv->num_threads_));
            }
            return vv;
        };
        auto for_range_new_beg = [&](int64_t new_beg) -> stmt_c {
            const auto begn = vv->iter_begin_.dyn_as<constant>();
            const auto step = vv->step_.dyn_as<constant>();
            if (begn.defined() && step.defined()
                    && (get_const_as_int(begn) < new_beg)) {
                auto old_beg = get_const_as_int(begn);
                auto old_stp = get_const_as_int(step);
                auto n = divide_and_ceil(new_beg - old_beg, old_stp);
                auto iter_beg = builder::make_constant(
                        {n * old_stp + old_beg}, vv->iter_begin_->dtype_);
                return copy_attr(*vv,
                        make_stmt<for_loop_node_t>(vv->var_, iter_beg,
                                vv->iter_end_, vv->step_, std::move(new_body),
                                vv->incremental_, vv->kind_, vv->num_threads_));
            }
            return vv;
        };
        // For now, the only case we experienced is one for and one if with one
        // case, thus only match for loop with one if (loop var cmp) stmt
        auto cond = vv->body_.cast<stmts>()
                            .filter([vv](const stmts &v) {
                                return vv->kind_ == for_type::NORMAL
                                        && v->seq_.size() == 1;
                            })
                            .map([](const stmts &v) {
                                return v->seq_.back().as<if_else>();
                            })
                            .filter([&new_body](const if_else &v) {
                                new_body = v->then_case_;
                                return !v->else_case_.defined();
                            })
                            .map([](const if_else &v) {
                                return v->condition_.dyn_as<cmp>();
                            })
                            .filter([vv](const cmp &v) {
                                return v->l_.ptr_same(vv->var_)
                                        && v->r_.isa<constant>();
                            });
        if (cond.has_value()) {
            const auto c = cond.get();
            const auto n = get_const_as_int(c->r_.static_as<constant>());
            switch (c->node_type_) {
                case sc_expr_type::cmp_lt: return for_range_new_end(n);
                case sc_expr_type::cmp_le: return for_range_new_end(n + 1);
                case sc_expr_type::cmp_gt: return for_range_new_beg(n + 1);
                case sc_expr_type::cmp_ge: return for_range_new_beg(n);
                default: break;
            }
        }
        // No match
        return vv;
    }
};

func_c loop_splitter_t::operator()(func_c f) {
    loop_spiltter_impl_t v;
    return v.dispatch(f);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
