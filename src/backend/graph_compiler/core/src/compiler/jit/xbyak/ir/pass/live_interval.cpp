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

#include <map>
#include <string>
#include <utility>

#include <compiler/ir/ir_module.hpp>
#include <compiler/jit/xbyak/ir/xbyak_visitor.hpp>
#include <util/array_ref.hpp>

#include "live_interval.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

class live_interval_impl_t : public xbyak_visitor_t {
public:
    using xbyak_visitor_t::dispatch;
    using xbyak_visitor_t::visit;

    std::map<uint64_t, stmt_index_t> loop_beg_map_;
    std::map<uint64_t, stmt_index_t> loop_end_map_;

    expr_c dispatch(expr_c v) override { return v; }

    func_c dispatch(func_c f) override {
        for (auto &v : f->params_) {
            create_liveness(v, 0);
        }
        return xbyak_visitor_t::dispatch(std::move(f));
    }

    stmt_c visit(define_c v) override {
        auto index = GET_STMT_INDEX(v);
        if (v->init_.defined()) {
            create_liveness(v->var_, index);
            update_dependency_liveness(v->init_, index);
        } else if (v->var_.isa<tensor>()) {
            create_liveness(v->var_, index);
        }
        return std::move(v);
    }

    stmt_c visit(assign_c v) override {
        auto index = GET_STMT_INDEX(v);
        update_dependency_liveness(v->var_, index);
        update_dependency_liveness(v->value_, index);

        return std::move(v);
    }

    stmt_c visit(evaluate_c v) override {
        if (v->value_.defined()) {
            update_dependency_liveness(v->value_, GET_STMT_INDEX(v));
        }
        return std::move(v);
    }

    stmt_c visit(returns_c v) override {
        if (v->value_.defined()) {
            update_dependency_liveness(v->value_, GET_STMT_INDEX(v));
        }
        return std::move(v);
    }

    stmt_c visit(if_else_c v) override {
        update_dependency_liveness(v->condition_, GET_STMT_INIT_INDEX(v));

        if (v->then_case_.defined()) {
            xbyak_visitor_t::dispatch(v->then_case_);
        }
        if (v->else_case_.defined()) {
            xbyak_visitor_t::dispatch(v->else_case_);
        }

        return std::move(v);
    }

    stmt_c visit(for_loop_c v) override {
        auto loop_si = GET_STMT_INIT_INDEX(v);
        auto loop_ei = GET_STMT_INDEX(v);

        update_dependency_liveness(v->var_, loop_si);
        update_dependency_liveness(v->iter_begin_, loop_si);

        auto &stmt_data = GET_STMT_DATA(v->body_);
        loop_end_map_[stmt_data.loop_depth_] = stmt_data.index_;
        loop_beg_map_[stmt_data.loop_depth_] = stmt_data.init_index_;

        xbyak_visitor_t::dispatch(v->body_);

        update_dependency_liveness(v->var_, loop_ei);
        update_dependency_liveness(v->step_, loop_ei);
        update_dependency_liveness(v->iter_end_, loop_ei);

        return std::move(v);
    }

private:
    void create_liveness(const expr &cur, stmt_index_t index) {
        if (!GET_LIVE_RANGE(cur).defined_) {
            GET_LIVE_RANGE(cur) = live_range_t(index);
        }
    }

    void update_liveness(array_ref<expr> ref, stmt_index_t index) {
        for (auto &v : ref) {
            create_liveness(v, index);
            auto &expr_data = GET_EXPR_DATA(v);
            auto &live_range = GET_LIVE_RANGE(v);
            auto def_loop_depth_ = expr_data.def_scope_
                    ? GET_STMT_DATA(expr_data.def_scope_).loop_depth_
                    : 0;
            auto cur_loop_depth_ = current_scope()
                    ? GET_STMT_DATA(current_scope()).loop_depth_
                    : 0;
            if (def_loop_depth_ < cur_loop_depth_) {
                auto scope_beg = loop_beg_map_[def_loop_depth_ + 1];
                auto scope_end = loop_end_map_[def_loop_depth_ + 1];
                live_range.update(scope_beg, scope_end);
            } else {
                live_range.update(index);
            }
            if (v->node_type_ == sc_expr_type::indexing) {
                update_dependency_liveness(v, index);
            }
        }
    }

    void update_dependency_liveness(const expr &v, stmt_index_t index) {
        switch (v->node_type_) {
            case sc_expr_type::undef:
            case sc_expr_type::ssa_phi: assert(0 && "Unreachable"); break;
            case sc_expr_type::constant:
            case sc_expr_type::func_addr: break;
            case sc_expr_type::tensor:
            case sc_expr_type::var: {
                update_liveness(&v, index);
            } break;
            case sc_expr_type::cast: {
                update_liveness(&v.static_as<cast>()->in_, index);
            } break;
            case sc_expr_type::add:
            case sc_expr_type::sub:
            case sc_expr_type::mul:
            case sc_expr_type::div:
            case sc_expr_type::mod: {
                auto val = v.static_as<binary>();
                update_liveness({val->l_, val->r_}, index);
            } break;
            case sc_expr_type::cmp_eq:
            case sc_expr_type::cmp_ne:
            case sc_expr_type::cmp_lt:
            case sc_expr_type::cmp_le:
            case sc_expr_type::cmp_gt:
            case sc_expr_type::cmp_ge: {
                auto val = v.static_as<cmp>();
                update_liveness({val->l_, val->r_}, index);
            } break;
            case sc_expr_type::logic_and:
            case sc_expr_type::logic_or: {
                auto val = v.static_as<logic>();
                update_liveness({val->l_, val->r_}, index);
            } break;
            case sc_expr_type::logic_not: {
                update_liveness(&v.static_as<logic_not>()->in_, index);
            } break;
            case sc_expr_type::select: {
                auto val = v.static_as<select>();
                update_liveness({val->cond_, val->l_, val->r_}, index);
            } break;
            case sc_expr_type::indexing: {
                auto val = v.static_as<indexing>();
                update_liveness(&val->ptr_, index);
                update_liveness(val->idx_, index);
                if (val->mask_.defined()) {
                    update_liveness(&val->mask_, index);
                }
            } break;
            case sc_expr_type::call: {
                auto val = v.static_as<call>();
                auto func = std::dynamic_pointer_cast<expr_base>(val->func_);
                if (func) { update_liveness({expr(func)}, index); }
                update_liveness(val->args_, index);
            } break;
            case sc_expr_type::tensorptr: {
                auto val = v.static_as<tensorptr>();
                update_liveness(&val->base_->ptr_, index);
                update_liveness(val->base_->idx_, index);
            } break;
            case sc_expr_type::intrin_call: {
                update_liveness(v.static_as<intrin_call>()->args_, index);
            } break;
            case sc_expr_type::low_level_intrin: {
                auto val = v.checked_as<xbyak_intrin>();
                update_liveness(val->args_, index);
                if (val->modifier_.cond_mask_.defined()) {
                    update_liveness({val->modifier_.cond_mask_}, index);
                }
            } break;
        }
    }
};

func_c live_interval_t::operator()(func_c v) {
    if (v->name_.find("_should_inline_") != std::string::npos) { return v; }
    live_interval_impl_t live_interval;
    return live_interval.dispatch(std::move(v));
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
