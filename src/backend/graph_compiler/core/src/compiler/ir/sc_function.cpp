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

#include "sc_function.hpp"
#include <utility>
#include "builder.hpp"
#include "ir_comparer.hpp"
#include <util/any_map.hpp>

namespace sc {

ostream &operator<<(ostream &os, const func_c &e) {
    return os << e.get();
}

ostream &operator<<(ostream &os, const func_base *e) {
    e->to_string(os);
    return os;
}

static ostream &print_single_arg(ostream &os, const expr &arg) {
    if (arg.isa<tensor>()) {
        auto a = arg.static_as<tensor>();
        a->to_string_full(os);
    } else {
        os << arg << ": " << arg->dtype_;
    }
    return os;
}

void func_base::to_string(ostream &os) const {
    os << "func " << name_ << '(';
    if (!params_.empty()) {
        for (unsigned i = 0; i < params_.size() - 1; i++) {
            print_single_arg(os, params_.at(i)) << ", ";
        }
        print_single_arg(os, params_.back());
    }
    os << "): " << ret_type_ << ' ';
    if (body_.defined()) { body_->to_string(os, 0); }
}

func_t::func_t(func_base *ptr) : std::shared_ptr<func_base>(ptr) {}
func_t::func_t(std::shared_ptr<func_base> &&other)
    : std::shared_ptr<func_base>(std::move(other)) {}

func_base::~func_base() = default;

func_base::func_base(const std::string &name, const std::vector<expr> &params,
        stmt body, sc_data_type_t ret_type)
    : name_(name)
    , params_(params)
    , body_(std::move(body))
    , ret_type_(ret_type) {
    if (body_.defined()) {
        decl_ = builder::make_func(name, params, stmt(), ret_type);
    }
}

func_t func_base::remake() const {
    auto ret = builder::make_func(name_, params_, body_, ret_type_);
    if (attr_) { ret->attr_ = utils::make_unique<any_map_t>(*attr_); }
    return ret;
}

bool func_base::equals(const func_c &f) const {
    ir_comparer cmper;
    return this->equals(f, cmper);
}

bool func_base::equals(const func_c &f, ir_comparer &ctx) const {
    func_t shared = std::const_pointer_cast<func_base>(shared_from_this());
    bool name_checking_passed = !ctx.cmp_names_ || (name_ == f->name_);
    return ctx.set_result(shared, f,
                   ret_type_ == f->ret_type_ && name_checking_passed
                           && ctx.expr_arr_equals(params_, f->params_))
            && ctx.check_equals_may_null(body_, f->body_);
}

} // namespace sc
