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

#include "builder.hpp"

#include "ir_comparer.hpp"

#include <utility>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
void ir_comparer::reset() {
    same_ = true;
    if (diff) diff = utils::make_unique<ir_comparer_diff_t>();
    var_mapping_.clear();
}
ir_comparer::ir_comparer(bool needs_diff, bool cmp_names, bool cmp_var_ref,
        bool cmp_callee, bool cmp_commutative)
    : cmp_names_(cmp_names)
    , cmp_callee_(cmp_callee)
    , cmp_var_ref_(cmp_var_ref)
    , cmp_commutative_(cmp_commutative)
    , same_(true) {
    if (needs_diff) diff = utils::make_unique<ir_comparer_diff_t>();
}

bool ir_comparer::set_result(func_c l, func_c r, bool cond) {
    if (same_ && !cond) {
        same_ = false;
        if (diff) {
            diff->first_diff_func_.first = std::move(l);
            diff->first_diff_func_.second = std::move(r);
        }
    }
    return cond;
}

bool ir_comparer::set_result(expr_c l, expr_c r, bool cond) {
    if (same_ && !cond) {
        same_ = false;
        if (diff) {
            diff->first_diff_expr_.first = std::move(l);
            diff->first_diff_expr_.second = std::move(r);
        }
    }
    return cond;
}

bool ir_comparer::set_result(stmt_c l, stmt_c r, bool cond) {
    if (same_ && !cond) {
        same_ = false;
        if (diff) {
            diff->first_diff_stmt_.first = std::move(l);
            diff->first_diff_stmt_.second = std::move(r);
        }
    }
    return cond;
}

bool ir_comparer::expr_arr_equals(
        const std::vector<expr> &a, const std::vector<expr> &b) {
    if (a.size() != b.size()) { return false; }
    for (size_t i = 0; i < a.size(); i++) {
        if (!a.at(i)->equals(b.at(i), *this)) { return false; }
    }
    return true;
}

bool ir_comparer::compare(const func_c &l, const func_c &r, bool auto_reset) {
    bool ret = l->equals(r, *this);
    if (auto_reset) reset();
    return ret;
}

bool ir_comparer::compare(const expr_c &l, expr_c r, bool auto_reset) {
    bool ret = l->equals(std::move(r), *this);
    if (auto_reset) reset();
    return ret;
}

bool ir_comparer::compare(const stmt_c &l, stmt_c r, bool auto_reset) {
    bool ret = l->equals(std::move(r), *this);
    if (auto_reset) reset();
    return ret;
}

bool ir_comparer::check_or_set_expr_mapping(const expr_c &l, const expr_c &r) {
    auto f = var_mapping_.find(l.get());
    if (f != var_mapping_.end()) { return f->second == r.get(); }
    var_mapping_.insert(std::make_pair(l.get(), r.get()));
    return true;
}

void ir_comparer::set_expr_mapping(const expr_c &l, const expr_c &r) {
    var_mapping_.insert(std::make_pair(l.get(), r.get()));
}

bool ir_comparer::get_expr_mapping(const expr_c &l, const expr_c &r) {
    auto f = var_mapping_.find(l.get());
    if (f != var_mapping_.end()) { return f->second == r.get(); }
    return false;
}

std::ostream &operator<<(std::ostream &os, ir_comparer &cmper) {
    if (cmper.same_) {
        os << "same";
    } else {
        os << "not same: ";
        if (cmper.diff) {
            if (cmper.diff->first_diff_expr_.first.defined()
                    || cmper.diff->first_diff_expr_.second.defined()) {
                os << "diff expr = " << cmper.diff->first_diff_expr_.first
                   << " v.s. " << cmper.diff->first_diff_expr_.second;
            } else if (cmper.diff->first_diff_func_.first
                    || cmper.diff->first_diff_func_.second) {
                os << "diff func = " << cmper.diff->first_diff_func_.first
                   << " v.s. " << cmper.diff->first_diff_func_.second;
            } else if (cmper.diff->first_diff_stmt_.first.defined()
                    || cmper.diff->first_diff_stmt_.second.defined()) {
                os << "diff stmt = " << cmper.diff->first_diff_stmt_.first
                   << " v.s. " << cmper.diff->first_diff_stmt_.second;
            }
        }
    }
    os << '\n';
    return os;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
