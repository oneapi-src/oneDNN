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
#include <algorithm>
#include <atomic>
#include <iostream>
#include "builder.hpp"
#include "sc_expr.hpp"
#include "visitable.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
expr ir_visitor_base_t::dispatch_impl(expr e) { // NOLINT
    return e->visited_by(this);
}

stmt ir_visitor_base_t::dispatch_impl(stmt s) { // NOLINT
    return s->visited_by(this);
}

template <bool is_inplace, typename T>
expr visit_base_binary(ir_visitor_base_impl_t<is_inplace> *vis, T &&v) {
    bool &changed_ = vis->changed_;
    auto l = vis->dispatch_impl(v->l_);
    auto r = vis->dispatch_impl(v->r_);

    changed_ = !l.ptr_same(v->l_) || !r.ptr_same(v->r_);
    if (is_inplace) {
        v->l_ = std::move(l);
        v->r_ = std::move(r);
        return std::forward<T>(v);
    } else {
        if (changed_) {
            return builder::remake_binary(
                    std::move(l), std::move(r), std::forward<T>(v));
        } else {
            return std::forward<T>(v);
        }
    }
}

template <bool is_inplace>
expr ir_visitor_base_impl_t<is_inplace>::visit_impl(binary v) {
    return visit_base_binary<is_inplace, binary>(this, std::move(v));
}

template <bool is_inplace>
expr ir_visitor_base_impl_t<is_inplace>::visit_impl(logic v) {
    return visit_base_binary<is_inplace, logic>(this, std::move(v));
}

template <bool is_inplace>
expr ir_visitor_base_impl_t<is_inplace>::visit_impl(cmp v) {
    return visit_base_binary<is_inplace, cmp>(this, std::move(v));
}

template <>
bool ir_visitor_base_impl_t<true>::dispatch_expr_vector(
        std::vector<expr> &arr, std::vector<expr> &newval) {
    bool changed = false;
    for (auto &v : arr) {
        auto n = dispatch_impl(v);
        if (!n.ptr_same(v)) {
            changed = true;
            v = std::move(n);
        }
    }
    return changed;
}

template <>
bool ir_visitor_base_impl_t<false>::dispatch_expr_vector(
        std::vector<expr> &arr, std::vector<expr> &newval) {
    bool changed = false;
    newval = arr;
    for (auto itrv = arr.begin(); itrv != arr.end(); ++itrv) {
        auto &v = *itrv;
        auto n = dispatch_impl(v);
        if (!n.ptr_same(v)) {
            newval.at(itrv - arr.begin()) = n;
            changed = true;
        }
    }
    return changed;
}

template <bool is_inplace>
func_t ir_visitor_base_impl_t<is_inplace>::dispatch_impl(func_t s) {
    std::vector<expr> newparam;
    changed_ = dispatch_expr_vector(s->params_, newparam);
    auto body = s->body_.defined() ? dispatch_impl(s->body_) : stmt();
    changed_ |= !body.ptr_same(s->body_);
    if (is_inplace) {
        s->body_ = body;
        return s;
    } else {
        if (changed_) {
            return copy_attr(*s,
                    builder::make_func(s->name_, newparam, body, s->ret_type_));
        }
    }
    return s;
}

template <bool is_inplace>
expr do_visit(ir_visitor_base_impl_t<is_inplace> *vis, binary v) {
    return vis->visit_impl(std::move(v));
}

template <bool is_inplace>
expr do_visit(ir_visitor_base_impl_t<is_inplace> *vis, cmp v) {
    return vis->visit_impl(std::move(v));
}

template <bool is_inplace>
expr do_visit(ir_visitor_base_impl_t<is_inplace> *vis, logic v) {
    return vis->visit_impl(std::move(v));
}

template <bool is_inplace>
expr ir_visitor_base_impl_t<is_inplace>::visit_impl(constant v) {
    return std::move(v);
}
template <bool is_inplace>
expr ir_visitor_base_impl_t<is_inplace>::visit_impl(var v) {
    return std::move(v);
}
template <bool is_inplace>
expr ir_visitor_base_impl_t<is_inplace>::visit_impl(tensor v) {
    std::vector<expr> newdims;
    std::vector<expr> newplaindims;
    std::vector<expr> newstrides;
    changed_ = dispatch_expr_vector(v->dims_, newdims);
    changed_ |= dispatch_expr_vector(v->strides_, newstrides);
    if (is_inplace) {
        return std::move(v);
    } else {
        if (changed_) {
            return copy_attr(*v,
                    make_expr<tensor_node>(v->elem_dtype_, v->name_, newdims,
                            v->address_space_, v->init_value_, newstrides));
        } else {
            return std::move(v);
        }
    }
}
template <bool is_inplace>
expr ir_visitor_base_impl_t<is_inplace>::visit_impl(cast v) {
    auto l = dispatch_impl(v->in_);
    changed_ = !l.ptr_same(v->in_);
    if (is_inplace) {
        v->in_ = l;
        return std::move(v);
    } else {
        if (changed_) {
            return copy_attr(*v, make_expr<cast_node>(v->dtype_, l));
        }
        return std::move(v);
    }
}

#define GEN_VISIT(TYPE) \
    template <bool is_inplace> \
    expr ir_visitor_base_impl_t<is_inplace>::visit_impl(TYPE v) { \
        return do_visit(this, v); \
    }
GEN_VISIT(add)
GEN_VISIT(sub)
GEN_VISIT(mul)
GEN_VISIT(div)
GEN_VISIT(mod)
GEN_VISIT(cmp_eq)
GEN_VISIT(cmp_lt)
GEN_VISIT(cmp_le)
GEN_VISIT(cmp_gt)
GEN_VISIT(cmp_ge)
GEN_VISIT(cmp_ne)
GEN_VISIT(logic_and)
GEN_VISIT(logic_or)

template <bool is_inplace>
expr ir_visitor_base_impl_t<is_inplace>::visit_impl(select v) {
    auto cond = dispatch_impl(v->cond_);
    auto l = dispatch_impl(v->l_);
    auto r = dispatch_impl(v->r_);
    changed_ = !cond.ptr_same(v->cond_) || !l.ptr_same(v->l_)
            || !r.ptr_same(v->r_);
    if (is_inplace) {
        if (changed_) {
            v->cond_ = cond;
            v->l_ = l;
            v->r_ = r;
        }
        return std::move(v);
    } else {
        if (changed_) {
            return copy_attr(*v,
                    builder::make_select(
                            std::move(cond), std::move(l), std::move(r)));
        } else {
            return std::move(v);
        }
    }
}

template <bool is_inplace>
expr ir_visitor_base_impl_t<is_inplace>::visit_impl(indexing v) {
    auto ptr = dispatch_impl(v->ptr_);
    bool changed = !ptr.ptr_same(v->ptr_);
    std::vector<expr> new_arr;
    changed |= dispatch_expr_vector(v->idx_, new_arr);
    expr mask;
    if (v->mask_.defined()) {
        mask = dispatch_impl(v->mask_);
        changed |= (!mask.ptr_same(v->mask_));
    }
    if (is_inplace) {
        if (changed) {
            v->ptr_ = ptr;
            v->mask_ = mask;
        }
        changed_ = changed;
        return std::move(v);
    } else {
        if (changed) {
            return copy_attr(*v,
                    builder::make_indexing(std::move(ptr), std::move(new_arr),
                            v->dtype_.lanes_, std::move(mask)));
        } else {
            return std::move(v);
        }
    }
}

template <bool is_inplace>
expr ir_visitor_base_impl_t<is_inplace>::visit_impl(tensorptr v) {
    auto ptr = visit_impl(v->base_);
    changed_ = !ptr.ptr_same(v->base_);
    if (is_inplace) {
        if (changed_) { v->base_ = ptr.template checked_as<indexing>(); }
        return std::move(v);
    } else {
        if (changed_) {
            return copy_attr(*v,
                    make_expr<tensorptr_node>(
                            ptr.template checked_as<indexing>(), v->shape_,
                            v->is_slice_));
        } else {
            return std::move(v);
        }
    }
}

template <bool is_inplace>
expr ir_visitor_base_impl_t<is_inplace>::visit_impl(intrin_call v) {
    std::vector<expr> new_arr;
    changed_ = dispatch_expr_vector(v->args_, new_arr);
    if (is_inplace) {
        return std::move(v);
    } else {
        if (changed_) {
            return copy_attr(*v, builder::remake_intrin_call(v, new_arr));
        } else {
            return std::move(v);
        }
    }
}

template <bool is_inplace>
expr ir_visitor_base_impl_t<is_inplace>::visit_impl(func_addr v) {
    return v;
}

template <bool is_inplace>
expr ir_visitor_base_impl_t<is_inplace>::visit_impl(ssa_phi v) {
    throw std::runtime_error(
            "ssa_phi should not occur in this visitor. You need to either "
            "remove this phi node in IR or use SSA visitor/viewer instead");
}

template <bool is_inplace>
expr ir_visitor_base_impl_t<is_inplace>::visit_impl(low_level_intrin v) {
    std::vector<expr> new_arr;
    changed_ = dispatch_expr_vector(v->args_, new_arr);
    if (is_inplace) {
        return std::move(v);
    } else {
        if (changed_) {
            return copy_attr(*v, builder::remake_low_level_intrin(v, new_arr));
        } else {
            return std::move(v);
        }
    }
}

template <bool is_inplace>
expr ir_visitor_base_impl_t<is_inplace>::visit_impl(call v) {
    std::vector<expr> new_arr;
    bool changed = dispatch_expr_vector(v->args_, new_arr);
    auto new_callee = v->func_;
    if (auto ex = std::dynamic_pointer_cast<expr_base>(v->func_)) {
        new_callee = dispatch_impl(expr(ex)).impl;
    }
    changed |= (new_callee != v->func_);
    changed_ = changed;
    if (is_inplace) {
        return std::move(v);
    } else {
        if (changed) {
            auto ret = v->remake().static_as<call>();
            ret->args_ = std::move(new_arr);
            ret->func_ = new_callee;
            return copy_attr(*v, std::move(ret));
        } else {
            return std::move(v);
        }
    }
}

template <bool is_inplace>
expr ir_visitor_base_impl_t<is_inplace>::visit_impl(logic_not v) {
    auto l = dispatch_impl(v->in_);
    changed_ = !l.ptr_same(v->in_);
    if (is_inplace) {
        v->in_ = l;
        return std::move(v);
    } else {
        if (changed_) { return copy_attr(*v, make_expr<logic_not_node>(l)); }
        return std::move(v);
    }
}

template <bool is_inplace>
stmt ir_visitor_base_impl_t<is_inplace>::visit_impl(assign v) {
    auto l = dispatch_impl(v->var_);
    auto r = dispatch_impl(v->value_);
    changed_ = (!l.ptr_same(v->var_) || !r.ptr_same(v->value_));
    if (is_inplace) {
        if (changed_) {
            v->var_ = l;
            v->value_ = r;
        }
        return std::move(v);
    } else {
        if (changed_) {
            return copy_attr(
                    *v, make_stmt<assign_node_t>(std::move(l), std::move(r)));
        }
        return std::move(v);
    }
}

template <bool is_inplace>
stmt ir_visitor_base_impl_t<is_inplace>::visit_impl(stmts s) {
    if (is_inplace) {
        changed_ = false;
        for (auto &v : s->seq_) {
            auto n = dispatch_impl(v);
            if (!n.ptr_same(v)) {
                changed_ = true;
                v = std::move(n);
            }
        }
        return s;
    } else {
        std::vector<stmt> seq = s->seq_;
        bool changed = false;
        for (auto &v : seq) {
            auto n = dispatch_impl(v);
            if (!n.ptr_same(v)) {
                changed = true;
                v = std::move(n);
            }
        }
        if (changed) {
            return copy_attr(*s, make_stmt<stmts_node_t>(std::move(seq)));
        }
        return s;
    }
}

template <bool is_inplace>
stmt ir_visitor_base_impl_t<is_inplace>::visit_impl(if_else v) {
    auto cond = dispatch_impl(v->condition_);
    auto thencase = dispatch_impl(v->then_case_);

    stmt elsecase;
    if (v->else_case_.defined()) elsecase = dispatch_impl(v->else_case_);
    changed_ = !cond.ptr_same(v->condition_)
            || !elsecase.ptr_same(v->else_case_)
            || !thencase.ptr_same(v->then_case_);
    if (is_inplace) {
        v->condition_ = cond;
        v->then_case_ = thencase;
        v->else_case_ = elsecase;
        return std::move(v);
    } else {
        if (changed_) {
            return copy_attr(*v,
                    make_stmt<if_else_node_t>(std::move(cond),
                            std::move(thencase), std::move(elsecase)));
        }
        return std::move(v);
    }
}
template <bool is_inplace>
stmt ir_visitor_base_impl_t<is_inplace>::visit_impl(for_loop v) {
    auto var = dispatch_impl(v->var_);
    auto begin = dispatch_impl(v->iter_begin_);
    auto end = dispatch_impl(v->iter_end_);
    auto step = dispatch_impl(v->step_);
    auto body = dispatch_impl(v->body_);

    changed_ = !(var.ptr_same(v->var_) && begin.ptr_same(v->iter_begin_)
            && end.ptr_same(v->iter_end_) && step.ptr_same(v->step_)
            && body.ptr_same(v->body_));
    if (is_inplace) {
        v->var_ = var;
        v->iter_begin_ = begin;
        v->iter_end_ = end;
        v->step_ = step;
        v->body_ = body;
        return std::move(v);
    } else {
        if (changed_) {
            return copy_attr(*v,
                    make_stmt<for_loop_node_t>(std::move(var), std::move(begin),
                            std::move(end), std::move(step), std::move(body),
                            v->incremental_, v->kind_, v->num_threads_));
        }
        return std::move(v);
    }
}

template <bool is_inplace>
stmt ir_visitor_base_impl_t<is_inplace>::visit_impl(evaluate v) {
    auto val = dispatch_impl(v->value_);
    changed_ = !val.ptr_same(v->value_);
    if (is_inplace) {
        v->value_ = val;
        return std::move(v);
    } else {
        if (changed_) { return copy_attr(*v, make_stmt<evaluate_node_t>(val)); }
        return std::move(v);
    }
}

template <bool is_inplace>
stmt ir_visitor_base_impl_t<is_inplace>::visit_impl(returns v) {
    if (v->value_.defined()) {
        auto val = dispatch_impl(v->value_);
        changed_ = !val.ptr_same(v->value_);
        if (is_inplace) {
            v->value_ = val;
            return std::move(v);
        } else {
            if (changed_) {
                return copy_attr(*v, make_stmt<returns_node_t>(val));
            }
            return std::move(v);
        }
    }
    return std::move(v);
}

template <bool is_inplace>
stmt ir_visitor_base_impl_t<is_inplace>::visit_impl(define v) {
    expr init;
    bool changed = false;
    if (v->init_.defined()) {
        init = dispatch_impl(v->init_);
        changed = !init.ptr_same(v->init_);
    }
    auto var = dispatch_impl(v->var_);
    changed |= !var.ptr_same(v->var_);
    changed_ = changed;
    if (is_inplace) {
        v->var_ = var;
        v->init_ = init;
        return std::move(v);
    } else {
        if (changed_) {
            return copy_attr(*v,
                    make_stmt<define_node_t>(
                            std::move(var), v->linkage_, std::move(init)));
        }
        return std::move(v);
    }
}

#define DEFINE_VISITOR_PROXY_IMPL(TYPE, ...) \
    __VA_ARGS__; \
    expr_c ir_visitor_t::visit(TYPE##_c v) { \
        return ir_visitor_base_impl_t<false>::visit_impl(v.remove_const()); \
    } \
    expr ir_visitor_t::visit_impl(TYPE v) { \
        return visit(v.to_const()).remove_const(); \
    }
#define DEFINE_VISITOR_PROXY_BASE(TYPE) DEFINE_VISITOR_PROXY_IMPL(TYPE, )
#define DEFINE_VISITOR_PROXY(TYPE) \
    DEFINE_VISITOR_PROXY_IMPL( \
            TYPE, template struct visitable_t<TYPE##_node, expr_base>)

DEFINE_VISITOR_PROXY(constant) // NOLINT
DEFINE_VISITOR_PROXY(var) // NOLINT
DEFINE_VISITOR_PROXY(cast) // NOLINT

DEFINE_VISITOR_PROXY_BASE(binary) // NOLINT
DEFINE_VISITOR_PROXY(add) // NOLINT
DEFINE_VISITOR_PROXY(sub) // NOLINT
DEFINE_VISITOR_PROXY(mul) // NOLINT
DEFINE_VISITOR_PROXY(div) // NOLINT
DEFINE_VISITOR_PROXY(mod) // NOLINT

DEFINE_VISITOR_PROXY_BASE(cmp) // NOLINT
DEFINE_VISITOR_PROXY(cmp_eq) // NOLINT
DEFINE_VISITOR_PROXY(cmp_lt) // NOLINT
DEFINE_VISITOR_PROXY(cmp_le) // NOLINT
DEFINE_VISITOR_PROXY(cmp_gt) // NOLINT
DEFINE_VISITOR_PROXY(cmp_ge) // NOLINT
DEFINE_VISITOR_PROXY(cmp_ne) // NOLINT

DEFINE_VISITOR_PROXY_BASE(logic) // NOLINT
DEFINE_VISITOR_PROXY(logic_and) // NOLINT
DEFINE_VISITOR_PROXY(logic_or) // NOLINT

DEFINE_VISITOR_PROXY(logic_not) // NOLINT
DEFINE_VISITOR_PROXY(select) // NOLINT
DEFINE_VISITOR_PROXY(indexing) // NOLINT
DEFINE_VISITOR_PROXY(call) // NOLINT
DEFINE_VISITOR_PROXY(tensor) // NOLINT
DEFINE_VISITOR_PROXY(tensorptr) // NOLINT
DEFINE_VISITOR_PROXY(intrin_call) // NOLINT
DEFINE_VISITOR_PROXY(func_addr) // NOLINT
DEFINE_VISITOR_PROXY(ssa_phi) // NOLINT
DEFINE_VISITOR_PROXY(low_level_intrin) // NOLINT

#define DEFINE_VISITOR_PROXY_STMT(TYPE) \
    stmt_c ir_visitor_t::visit(TYPE##_c v) { \
        return ir_visitor_base_impl_t<false>::visit_impl(v.remove_const()); \
    } \
    stmt ir_visitor_t::visit_impl(TYPE v) { \
        return visit(v.to_const()).remove_const(); \
    }
DEFINE_VISITOR_PROXY_STMT(assign) // NOLINT
DEFINE_VISITOR_PROXY_STMT(stmts) // NOLINT
DEFINE_VISITOR_PROXY_STMT(if_else) // NOLINT
DEFINE_VISITOR_PROXY_STMT(evaluate) // NOLINT
DEFINE_VISITOR_PROXY_STMT(define) // NOLINT
DEFINE_VISITOR_PROXY_STMT(returns) // NOLINT
DEFINE_VISITOR_PROXY_STMT(for_loop) // NOLINT

expr_c ir_visitor_t::dispatch(expr_c e) { // NOLINT
    return ir_visitor_base_impl_t<false>::dispatch_impl(e.remove_const());
}

stmt_c ir_visitor_t::dispatch(stmt_c s) { // NOLINT
    return ir_visitor_base_impl_t<false>::dispatch_impl(s.remove_const());
}

func_c ir_visitor_t::dispatch(func_c f) { // NOLINT
    return ir_visitor_base_impl_t<false>::dispatch_impl(
            std::const_pointer_cast<func_base>(f));
}

expr ir_visitor_t::dispatch_impl(expr e) {
    return dispatch(std::move(e)).remove_const();
}
stmt ir_visitor_t::dispatch_impl(stmt s) {
    return dispatch(std::move(s)).remove_const();
}
func_t ir_visitor_t::dispatch_impl(func_t v) {
    return std::const_pointer_cast<func_base>(dispatch(std::move(v)));
}

bool ir_visitor_t::dispatch_expr_vector(
        const std::vector<expr> &arr, std::vector<expr> &newval) {
    return ir_visitor_base_impl_t<false>::dispatch_expr_vector(
            const_cast<std::vector<expr> &>(arr), newval);
}

bool ir_visitor_t::dispatch_expr_vector(
        const std::vector<expr> &arr, std::vector<expr_c> &newval) {
    std::vector<expr> newval2;
    bool ret = ir_visitor_base_impl_t<false>::dispatch_expr_vector(
            const_cast<std::vector<expr> &>(arr), newval2);
    newval.insert(newval.end(), newval2.begin(), newval2.end());
    return ret;
}

uint64_t ir_visitor_t::get_run_id() {
    static std::atomic<uint64_t> id = {0};
    return id++;
}

template class ir_visitor_base_impl_t<true>; // instantiation of the template
template class ir_visitor_base_impl_t<false>; // instantiation of the template

expr_c ir_consistent_visitor_t::dispatch(expr_c e) {
    bool is_var_or_tensor = e.isa<tensor>() || e.isa<var>();
    if (is_var_or_tensor) {
        auto itr = replace_map_.find(e);
        if (itr != replace_map_.end()) { e = itr->second; }
    }
    expr_c newe = ir_visitor_t::dispatch(e);
    if (is_var_or_tensor && !newe.ptr_same(e)) {
        assert(replace_map_.count(e) == 0);
        replace_map_[e] = newe;
    }
    return newe;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
