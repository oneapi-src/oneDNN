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
#include <unordered_map>

#include <utility>
#include <vector>
#include "../builder.hpp"
#include "../intrinsics.hpp"
#include "../util_module_passes.hpp"
#include "../visitor.hpp"
#include "auto_cast.hpp"
#include <compiler/ir/pass_dep_util.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(auto_caster, SC_PASS_DEPENDS_ON(index_flattener),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE(CONST_FOLDED));

static std::unordered_map<sc_data_type_t, int> promotion_priority = {
        {datatypes::f32, 100},
        {datatypes::index, 90},
        {datatypes::u32, 81},
        {datatypes::s32, 80},
        {datatypes::u16, 70},
        {datatypes::u8, 61},
        {datatypes::s8, 60},
};

int get_casting_priority(sc_data_type_t dty) {
    auto itr = promotion_priority.find(dty);
    if (itr != promotion_priority.end()) { return itr->second; }
    return -1;
}

static bool try_cast_to_fp32(expr_c &v1) {
    if (v1->dtype_.type_code_ == sc_data_etype::BF16
            || v1->dtype_.type_code_ == sc_data_etype::F16) {
        v1 = builder::make_cast(
                sc_data_type_t(sc_data_etype::F32, v1->dtype_.lanes_), v1);
        return true;
    }
    return false;
}

static void do_promote(expr_c &v1, expr_c &v2) {
    int pr1 = get_casting_priority(v1->dtype_);
    int pr2 = get_casting_priority(v2->dtype_);
    if (pr1 != -1 && pr2 != -1) {
        // if both types are in the priority map
        assert(pr1 != pr2);
        if (pr1 > pr2) {
            v2 = builder::make_cast(v1->dtype_, v2);
        } else {
            v1 = builder::make_cast(v2->dtype_, v1);
        }
        return;
    }

    if (v1->dtype_.type_code_ == sc_data_etype::F32) {
        if (try_cast_to_fp32(v2)) return;
    } else if (v2->dtype_.type_code_ == sc_data_etype::F32) {
        if (try_cast_to_fp32(v1)) return;
    }
    COMPILE_ASSERT(false,
            "Cannot promote types: " << v1->dtype_ << " and " << v2->dtype_
                                     << ". The expr are: " << v1 << " and "
                                     << v2);
}

static bool cast_to(
        expr_c &v, sc_data_type_t ty, const std::function<void()> &raiseit) {
    if (v->dtype_ == ty) { return false; }
    if (v->dtype_.is_pointer() && ty == datatypes::pointer) {
        // if it is T* => void* cast
        v = builder::make_cast(ty, v);
    } else if (v->dtype_.lanes_ == 1 && ty == datatypes::generic) {
        // if it is T* => void* cast
        v = builder::make_cast(ty, v);
    } else {
        int pri = get_casting_priority(v->dtype_);
        if (pri >= 0) {
            int target_pri = get_casting_priority(ty);
            if (pri >= target_pri) raiseit();
            v = builder::make_cast(ty, v);
        } else {
            if (ty != datatypes::f32) raiseit();
            if (!try_cast_to_fp32(v)) raiseit();
        }
    }
    return true;
}

bool cast_to(expr_c &v, sc_data_type_t ty, expr_c containing) {
    auto raiseit = [&v, ty, &containing]() {
        COMPILE_ASSERT(false,
                "Cannot cast from " << v->dtype_ << " to " << ty
                                    << ". expr = " << containing);
    };
    return cast_to(v, ty, raiseit);
}

static bool cast_to(expr_c &v, sc_data_type_t ty, stmt_c containing) {
    auto raiseit = [&v, ty, &containing]() {
        COMPILE_ASSERT(false,
                "Cannot cast from " << v->dtype_ << " to " << ty
                                    << ". expr = " << containing);
    };
    return cast_to(v, ty, raiseit);
}

static expr_c cast_intrin_binary(
        intrin_call_c oldnode, std::vector<expr> &newargs, bool changed) {
    auto &l = newargs[0];
    auto &r = newargs[1];
    if (l->dtype_ != r->dtype_) {
        expr_c vl = l, vr = r;
        do_promote(vl, vr);
        // write back to newargs
        l = vl.remove_const();
        r = vr.remove_const();
        changed = true;
    }
    sc_data_type_t dtype = l->dtype_;
    if (changed) {
        auto ret = copy_attr(
                *oldnode, builder::remake_intrin_call(oldnode, newargs));
        ret->dtype_ = dtype;
        return std::move(ret);
    } else {
        return std::move(oldnode);
    }
}

class auto_cast_t : public ir_consistent_visitor_t {
public:
    using ir_consistent_visitor_t::dispatch;
    using ir_consistent_visitor_t::visit;
    expr_c visit(binary_c v) override {
        auto l = dispatch(v->l_);
        auto r = dispatch(v->r_);
        bool changed = !l.ptr_same(v->l_) || !r.ptr_same(v->r_);
        if (l.isa<tensor>()) {
            changed = true;
        } else if (l->dtype_ != r->dtype_) {
            do_promote(l, r);
            changed = true;
        } else if (v->dtype_ == datatypes::undef) {
            changed = true;
        }
        sc_data_type_t dtype = l->dtype_;
        if (changed) {
            auto ret = builder::remake_binary(l, r, v);
            ret->dtype_ = dtype;
            return std::move(ret);
        } else {
            return std::move(v);
        }
    }

    expr_c visit(cmp_c v) override {
        auto l = dispatch(v->l_);
        auto r = dispatch(v->r_);
        bool changed = !l.ptr_same(v->l_) || !r.ptr_same(v->r_);
        if (l->dtype_ != r->dtype_) {
            do_promote(l, r);
            changed = true;
        }
        if (changed) {
            return builder::remake_binary(l, r, std::move(v));
        } else {
            return std::move(v);
        }
    }

    // makes all indices the same type with the largest dtype
    expr_c visit(indexing_c v) override {
        auto ptr = dispatch(v->ptr_);
        bool changed = !ptr.ptr_same(v->ptr_);
        std::vector<expr> newidx;
        changed |= dispatch_expr_vector(v->idx_, newidx);
        int max_priori = -100;
        sc_data_type_t target_type;
        // first find max priority
        for (auto &idx : newidx) {
            int priori = get_casting_priority(idx->dtype_);
            if (priori > max_priori) {
                max_priori = priori;
                target_type = idx->dtype_;
            }
        }

        std::vector<expr_c> newidx_c;
        newidx_c.reserve(newidx.size());
        // second pass: cast to max priority dtype
        for (auto &idx : newidx) {
            expr_c idx_c = idx;
            changed |= cast_to(idx_c, target_type, v);
            newidx_c.emplace_back(std::move(idx_c));
        }

        auto new_mask = v->mask_.defined() ? dispatch(v->mask_) : expr_c();
        changed |= !new_mask.ptr_same(v->mask_);

        if (changed) {
            return copy_attr(*v,
                    builder::make_indexing(
                            ptr, newidx_c, v->dtype_.lanes_, new_mask));
        } else {
            return std::move(v);
        }
    }

    stmt_c visit(assign_c v) override {
        auto var = dispatch(v->var_);
        auto val = dispatch(v->value_);
        bool changed = !var.ptr_same(v->var_) || !val.ptr_same(v->value_);
        if (v->value_->dtype_ != v->var_->dtype_) {
            changed |= cast_to(val, v->var_->dtype_, v);
        }
        if (changed) {
            return copy_attr(*v, builder::make_assign_unattached(var, val));
        } else {
            return std::move(v);
        }
    }

    expr_c visit(call_c v) override {
        std::vector<expr> newargs;
        bool changed = dispatch_expr_vector(v->args_, newargs);
        func_t the_func = std::dynamic_pointer_cast<func_base>(v->func_);
        func_t proto_func = v->get_prototype();
        assert(v->args_.size() == proto_func->params_.size());
        for (size_t i = 0; i < newargs.size(); i++) {
            expr_c idx_c = newargs[i];
            changed |= cast_to(idx_c, proto_func->params_[i]->dtype_, v);
            newargs[i] = idx_c.remove_const();
        }
        if (changed) {
            if (the_func) {
                return builder::remake_call(the_func, newargs, v);
            } else {
                return copy_attr(*v, make_expr<call_node>(v->func_, newargs));
            }
        } else {
            return std::move(v);
        }
    }

    expr_c visit(intrin_call_c v) override {
        std::vector<expr> newargs;
        bool changed = dispatch_expr_vector(v->args_, newargs);
        switch (v->type_) {
            case intrin_type::min:
            case intrin_type::max:
            case intrin_type::int_and:
            case intrin_type::int_or:
            case intrin_type::int_xor:
                return cast_intrin_binary(std::move(v), newargs, changed);
            case intrin_type::abs:
            case intrin_type::reinterpret:
            case intrin_type::shl:
            case intrin_type::shr:
            case intrin_type::round:
            case intrin_type::floor:
            case intrin_type::ceil:
            case intrin_type::exp:
            case intrin_type::log:
            case intrin_type::erf:
            case intrin_type::gather:
            case intrin_type::sqrt:
            case intrin_type::rsqrt:
            case intrin_type::reduce_add:
            case intrin_type::reduce_mul:
            case intrin_type::reduce_max:
            case intrin_type::reduce_min:
            case intrin_type::fmadd:
            case intrin_type::unpack_low:
            case intrin_type::unpack_high:
            case intrin_type::shuffle:
            case intrin_type::permute:
            case intrin_type::broadcast:
            case intrin_type::permutex2var:
            case intrin_type::permutexvar:
            case intrin_type::insert:
            case intrin_type::extract:
                // do nothing
                break;
            case intrin_type::brgemm:
                COMPILE_ASSERT(v->check_brgemm_arg_size(
                                       brgemm_args::NUM_FULL_ARGS_STRIDE),
                        "Wrong number of arguments for brgemm, expected equal "
                        "or larger than "
                                << brgemm_args::NUM_FULL_ARGS_STRIDE
                                << ", but got " << newargs.size() << ".");
                // the A, B, C are overloaded arguments
                for (unsigned i = brgemm_args::C + 1;
                        i < brgemm_args::NUM_FULL_ARGS_STRIDE; i++) {
                    // cast newargs[i] to the expected types
                    expr_c v = newargs[i];
                    // specific process for brgemm, as its runtime kernel is too
                    // hard to change.
                    if (v->dtype_ != brgemm_args::arg_types[i]) {
                        v = builder::make_cast(brgemm_args::arg_types[i], v);
                        changed = true;
                    }
                    newargs[i] = v.remove_const();
                }
                break;
            case intrin_type::list_brgemm:
                COMPILE_ASSERT(v->check_brgemm_arg_size(
                                       brgemm_args::NUM_FULL_ARGS_LIST),
                        "Wrong number of arguments for list brgemm, expected "
                        "equal or larger then "
                                << brgemm_args::NUM_FULL_ARGS_LIST
                                << ", but got " << newargs.size() << ".");
                for (unsigned i = 0; i < brgemm_args::NUM_FULL_ARGS_LIST; i++) {
                    // cast newargs[i] to the expected types
                    expr_c v = newargs[i];
                    changed |= cast_to(v, brgemm_args::list_arg_types[i], v);
                    newargs[i] = v.remove_const();
                }
                break;
            default: break;
        }
        if (changed) {
            return copy_attr(*v, builder::remake_intrin_call(v, newargs));
        }
        return std::move(v);
    }

    stmt_c visit(for_loop_c v) override {
        auto var = dispatch(v->var_);
        auto begin = dispatch(v->iter_begin_);
        auto end = dispatch(v->iter_end_);
        auto step = dispatch(v->step_);
        auto body = dispatch(v->body_);

        bool changed = !(var.ptr_same(v->var_) && begin.ptr_same(v->iter_begin_)
                && end.ptr_same(v->iter_end_) && step.ptr_same(v->step_)
                && body.ptr_same(v->body_));
        changed |= cast_to(begin, v->var_->dtype_, v);
        changed |= cast_to(end, v->var_->dtype_, v);
        changed |= cast_to(step, v->var_->dtype_, v);
        if (changed) {
            return copy_attr(*v,
                    builder::make_for_loop_unattached(var, begin, end, step,
                            body, v->incremental_, v->kind_, v->num_threads_));
        } else {
            return std::move(v);
        }
    }
};

func_c auto_caster_t::operator()(func_c f) {
    auto_cast_t pass;
    auto ret = pass.dispatch(std::move(f));
    return ret;
}

stmt_c auto_caster_t::operator()(stmt_c f) {
    auto_cast_t pass;
    return pass.dispatch(std::move(f));
}

expr_c auto_caster_t::operator()(expr_c f) {
    auto_cast_t pass;
    return pass.dispatch(std::move(f));
}

const_ir_module_ptr auto_caster_t::operator()(const_ir_module_ptr f) {
    auto_cast_t pass;
    return dispatch_module_on_visitor(&pass, f);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
