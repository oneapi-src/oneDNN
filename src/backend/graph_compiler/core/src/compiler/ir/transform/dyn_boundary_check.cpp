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
#include "dyn_boundary_check.hpp"
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "auto_cast.hpp"
#include "buffer_schedule.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/visitor.hpp>
#include <util/any_map.hpp>
#include <util/utils.hpp>

namespace sc {

SC_DECL_PASS_INFO(dyn_boundary_check, SC_PASS_DEPENDS_ON(validator),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

// boundary_check_maker is a callback which has args: 1. the value of an arg
// in `args` 2. the max boundary that this kernel may access
// boundary_check_maker returns the value after boundary_check
static void boundary_check_brgemm_update(std::vector<expr> &args,
        const std::function<expr(expr, expr)> &boundary_check_maker) {
    // arg[0], arg[1], arg[2] for A, B, C
    // [3] => num, [4] => M, [5] => N, [6] => K, [7] => LDA, [8] => LDB
    // [9] => LDC, [10] => stride_a, [11] => stride_b
    args[0] = boundary_check_maker(
            args[0], args[10] * (args[3] - 1) + args[7] * args[4]);
    args[1] = boundary_check_maker(
            args[1], args[11] * (args[3] - 1) + args[8] * args[6]);
    args[2] = boundary_check_maker(args[2], args[4] * args[9]);
}

static expr full_mask = builder::make_constant(
        {std::numeric_limits<uint64_t>::max()}, datatypes::index);
class dyn_boundary_check_adder_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    // use ordered map so that the unit test can have stable results
    std::map<std::string, expr> name_cache_;
    // this pass usually runs after auto_caster_t, so we need to manually run
    // casting on newly created nodes
    auto_caster_t caster_;
    expr get_or_create_str(const std::string &v) {
        auto itr = name_cache_.find(v);
        if (itr != name_cache_.end()) { return itr->second; }
        auto ret = builder::make_tensor(
                v + "_0check_name", {v.size() + 1}, datatypes::u8);
        name_cache_[v] = ret;
        return ret;
    }

    std::function<expr(expr, expr)> get_check_maker() {
        auto ths = this;
        return [ths](const expr &old_arg, const expr &bound) {
            if (old_arg.isa<tensor>()) {
                auto tsr = old_arg.static_as<tensor>();
                return builder::tensor_ptr(tsr,
                        {ths->caster_(builtin::boundary_check(
                                ths->get_or_create_str(tsr->name_), (uint64_t)0,
                                bound, full_mask, tsr->dims_[0]))});
            } else {
                auto tsptr = old_arg.checked_as<tensorptr>();
                auto tsr = tsptr->base_->ptr_.checked_as<tensor>();
                return builder::tensor_ptr(tsr,
                        {ths->caster_(builtin::boundary_check(
                                ths->get_or_create_str(tsr->name_),
                                tsptr->base_->idx_[0], bound, full_mask,
                                tsr->dims_[0]))});
            }
        };
    }

    expr_c visit(intrin_call_c v) override {
        auto ret = ir_visitor_t::visit(std::move(v)).static_as<intrin_call_c>();
        if (ret->type_ == intrin_type::brgemm) {
            auto newargs = ret->args_;
            boundary_check_brgemm_update(newargs, get_check_maker());
            return copy_attr(*ret,
                    make_expr<intrin_call_node>(
                            intrin_type::brgemm, newargs, *ret->intrin_attrs_));
        }
        return ret;
    }

    expr_c visit(tensor_c v) override {
        // fix-me(jingze): do not check the indexing in shapes and strides of
        // tensor.
        return v;
    }

    expr_c visit(tensorptr_c v) override {
        auto attr = v->base_->ptr_->attr_.get();
        if (attr && attr->get_or_else(attr_keys::can_be_scheduled, false)) {
            return v;
        }
        return ir_visitor_t::visit(v);
    }

    expr_c visit(indexing_c v) override {
        assert(v->idx_.size() == 1);
        auto tsr = v->ptr_.checked_as<tensor>();

        auto mask = v->mask_.defined() ? v->mask_ : full_mask;
        auto newidx
                = caster_(builtin::boundary_check(get_or_create_str(tsr->name_),
                                  v->idx_[0], (uint64_t)v->dtype_.lanes_, mask,
                                  tsr->dims_[0]))
                          .remove_const();
        auto ret = v->remake().static_as<indexing>();
        ret->idx_ = {std::move(newidx)};
        return ret;
    }

    func_c dispatch(func_c f) override {
        if (!f->body_.defined()) { return f; }
        auto body = ir_visitor_t::dispatch(f->body_);
        // insert the tensor name initialization code
        if (!name_cache_.empty()) {
            stmts newbody
                    = builder::make_stmts_unattached({}).static_as<stmts>();
            for (auto &itr : name_cache_) {
                // tensor def
                newbody->seq_.emplace_back(
                        builder::make_var_tensor_def_unattached(
                                itr.second, linkage::local));
                // write the string
                for (size_t i = 0; i < itr.first.size(); i++) {
                    newbody->seq_.emplace_back(builder::make_assign_unattached(
                            builder::make_indexing(itr.second, expr(i)),
                            make_expr<constant_node>(
                                    uint64_t(itr.first[i]), datatypes::u8)));
                }
                // write the tailing '\0'
                newbody->seq_.emplace_back(builder::make_assign_unattached(
                        builder::make_indexing(
                                itr.second, expr(itr.first.size())),
                        make_expr<constant_node>(uint64_t(0), datatypes::u8)));
            }
            newbody->seq_.emplace_back(body.remove_const());
            return copy_attr(*f,
                    builder::make_func(f->name_, f->params_, std::move(newbody),
                            f->ret_type_));
        } else {
            return f;
        }
    }
};

func_c dyn_boundary_check_t::operator()(func_c f) {
    dyn_boundary_check_adder_t pass;
    return pass.dispatch(f);
}

} // namespace sc
