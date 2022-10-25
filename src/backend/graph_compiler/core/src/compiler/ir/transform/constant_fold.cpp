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
#include <unordered_map>

#include <utility>
#include <vector>
#include "../builder.hpp"
#include "../util_module_passes.hpp"
#include "../visitor.hpp"
#include "auto_cast.hpp"
#include "constant_fold.hpp"
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/viewer.hpp>
#include <unordered_map>
#include <util/any_map.hpp>
#include <util/utils.hpp>
#include <util/variant.hpp>

namespace sc {
namespace constant_folding {

// the range from start to end (including end)
struct const_range_t {
    type_category cate;
    union_val start;
    union_val end;

    enum class infer_result { YES, NO, UNKNOWN };

    bool is_single_value() const { return start == end; }

    bool union_val_less(union_val a, union_val b) const {
        switch (cate) {
            case CATE_FLOAT: return a.f32 < b.f32; break;
            case CATE_INT: return a.s64 < b.s64; break;
            case CATE_UINT: return a.u64 < b.u64; break;
            default:
                throw std::runtime_error("Bad type for type_category");
                return false;
                break;
        }
    }

    bool is_good_range() const {
        return union_val_less(start, end) || is_single_value();
    }

#define def_operator(name, OP) \
    union_val union_val_##name(union_val a, union_val b) const { \
        switch (cate) { \
            case CATE_FLOAT: return a.f32 OP b.f32; break; \
            case CATE_INT: return a.s64 OP b.s64; break; \
            case CATE_UINT: return a.u64 OP b.u64; break; \
            default: \
                throw std::runtime_error("Bad type for type_category"); \
                return 0UL; \
                break; \
        } \
    }

    def_operator(add, +);
    def_operator(sub, -);
    def_operator(mul, *);
    def_operator(div, /);

    void assert_same_category(const const_range_t &other) const {
        COMPILE_ASSERT(
                cate == other.cate, "const_ranges should have same categories");
    }

    infer_result query_equal(const const_range_t &other) const {
        assert_same_category(other);
        if (!is_good_range() || !other.is_good_range()) {
            // there is overflow
            return infer_result::UNKNOWN;
        }
        //[s,e] [s2,e2]
        if (union_val_less(end, other.start)) { return infer_result::NO; }
        // [s2, e2] [s, e]
        if (union_val_less(other.end, start)) { return infer_result::NO; }
        return infer_result::UNKNOWN;
    }

    infer_result less_than(const const_range_t &other, bool allow_equal) const {
        assert_same_category(other);
        if (!is_good_range() || !other.is_good_range()) {
            // there is overflow
            return infer_result::UNKNOWN;
        }
        //[s,e] [s2,e2]
        if (union_val_less(end, other.start)) { return infer_result::YES; }
        // [s,e = s2, e2]
        if (end == other.start) {
            if (allow_equal) {
                return infer_result::YES;
            } else {
                return infer_result::UNKNOWN;
            }
        }
        // [s2, e2==s, e]
        if (start == other.end) {
            if (allow_equal)
                return infer_result::UNKNOWN;
            else
                return infer_result::NO;
        }

        // [s2, e2] [s, e]
        if (union_val_less(other.end, start)) { return infer_result::NO; }

        // otherwise, overlapping
        return infer_result::UNKNOWN;
    }

    bool operator==(const const_range_t &other) const {
        assert_same_category(other);
        return start == other.start && end == other.end;
    }

    const_range_t operator+(const const_range_t &other) const {
        assert_same_category(other);
        return const_range_t {cate, union_val_add(start, other.start),
                union_val_add(end, other.end)};
    }

    const_range_t operator-(const const_range_t &other) const {
        assert_same_category(other);
        return const_range_t {cate, union_val_sub(start, other.end),
                union_val_sub(end, other.start)};
    }

    const_range_t operator*(const const_range_t &other) const {
        assert_same_category(other);
        return const_range_t {cate, union_val_mul(start, other.start),
                union_val_mul(end, other.end)};
    }

    const_range_t operator/(const const_range_t &other) const {
        assert_same_category(other);
        return const_range_t {cate, union_val_div(start, other.end),
                union_val_div(end, other.start)};
    }

    const_range_t get_mod_range() const {
        COMPILE_ASSERT(cate != CATE_FLOAT, "'%' cannot be applied on floats");
        return const_range_t {cate, union_val {0UL}, start.u64 - 1};
    }
};

struct constant_fold_analysis_result_t {
private:
    // use raw pointers to avoid cycles of dependency
    variant<const_range_t, expr_base *> range_or_assignment;

public:
    constant_fold_analysis_result_t() = default;
    constant_fold_analysis_result_t(expr_base *v) : range_or_assignment {v} {}
    constant_fold_analysis_result_t(const const_range_t &v)
        : range_or_assignment {v} {}
    const const_range_t *get_range() const {
        if (range_or_assignment.isa<const_range_t>()) {
            return &range_or_assignment.get<const_range_t>();
        }
        if (range_or_assignment.isa<expr_base *>()) {
            auto result
                    = range_or_assignment.get<expr_base *>()
                              ->get_temp_data()
                              .get_or_null<constant_fold_analysis_result_t>();
            return result ? result->get_range() : nullptr;
        }
        return nullptr;
    }
    expr_base *get_assigned_expr() const {
        return range_or_assignment.isa<expr_base *>()
                ? range_or_assignment.get<expr_base *>()
                : nullptr;
    }
    void set_range(const const_range_t &v) { range_or_assignment = v; }
};

static bool parse_bool_infer_result(
        const_range_t::infer_result r, const_range_t &out) {
    if (r == const_range_t::infer_result::YES) {
        out = {type_category::CATE_UINT, 1UL, 1UL};
        return true;
    } else if (r == const_range_t::infer_result::NO) {
        out = {type_category::CATE_UINT, 0UL, 0UL};
        return true;
    }
    return false;
}

static const_range_t::infer_result flip_infer_result(
        const_range_t::infer_result r) {
    if (r == const_range_t::infer_result::YES) {
        return const_range_t::infer_result::NO;
    }
    if (r == const_range_t::infer_result::NO) {
        return const_range_t::infer_result::YES;
    }
    return r;
}

static bool compute_range(const expr_c &parent, const const_range_t *l,
        const const_range_t *r, const_range_t &out) {
    if (parent->node_type_ == sc_expr_type::mod) {
        out = r->get_mod_range();
        return true;
    }
    if (!l) return false;
    switch (parent->node_type_) {
        case sc_expr_type::add:
            out = *l + *r;
            return true;
            break;
        case sc_expr_type::sub:
            out = *l - *r;
            return true;
            break;
        case sc_expr_type::mul:
            out = *l * *r;
            return true;
            break;
        case sc_expr_type::div:
            out = *l / *r;
            return true;
            break;

        case sc_expr_type::cmp_eq:
            return parse_bool_infer_result(l->query_equal(*r), out);
            break;
        case sc_expr_type::cmp_ne:
            return parse_bool_infer_result(
                    flip_infer_result(l->query_equal(*r)), out);
            break;
        case sc_expr_type::cmp_le:
            return parse_bool_infer_result(l->less_than(*r, true), out);
            break;
        case sc_expr_type::cmp_lt:
            return parse_bool_infer_result(l->less_than(*r, false), out);
            break;
        case sc_expr_type::cmp_ge:
            // l>=r ====> r<l
            return parse_bool_infer_result(r->less_than(*l, false), out);
            break;
        case sc_expr_type::cmp_gt:
            // l>r ====> r<=l
            return parse_bool_infer_result(r->less_than(*l, true), out);
            break;
        default: break;
    }
    return false;
}

static void mark_range_for_const(const expr_c &v, bool fast) {
    if (!fast && v.isa<constant>() && v->dtype_.lanes_ == 1
            && !v->get_temp_data().isa<constant_fold_analysis_result_t>()) {
        auto cate = get_type_category_nothrow(v->dtype_);
        if (cate != CATE_OTHER) {
            auto constv = v.static_as<constant>()->value_.front();
            v->temp_data() = constant_fold_analysis_result_t {
                    const_range_t {cate, constv, constv}};
        }
    }
}

static const const_range_t *get_range_of_expr(const expr_c &v, bool fast) {
    if (fast) { return nullptr; }
    auto ana
            = v->get_temp_data().get_or_null<constant_fold_analysis_result_t>();
    if (!ana) return nullptr;
    return ana->get_range();
}

static const expr_base *get_assigned_expr(const expr_base *v) {
    auto ana
            = v->get_temp_data().get_or_null<constant_fold_analysis_result_t>();
    if (!ana) { return v; }
    if (auto single_assign = ana->get_assigned_expr()) {
        return get_assigned_expr(single_assign);
    }
    return v;
}

// mark the vars that is actually constants. mark the ranges of for-loop-iter
// vars
class constant_fold_analysis_t : public ir_viewer_t {
public:
    // the map of var to single assign value. if key=value, it means that this
    // var is not single assign
    std::unordered_map<expr_c, expr> single_assign_;

    expr_c dispatch(expr_c v) override { return v; }

    func_c dispatch(func_c v) override {
        ir_viewer_t::dispatch(v);
        for (auto &kv : single_assign_) {
            // if the var is assigned once
            if (kv.second.defined() && !kv.first.ptr_same(kv.second)) {
                auto cate = get_type_category_nothrow(kv.first->dtype_);
                if (cate != CATE_OTHER) {
                    mark_range_for_const(kv.second, false);
                    kv.first->temp_data()
                            = constant_fold_analysis_result_t {kv.second.get()};
                }
            }
        }
        return v;
    }

    void view(define_c v) override {
        if (v->var_.isa<var>() && v->var_->dtype_.lanes_ == 1) {
            if (v->init_.defined()) {
                assert(single_assign_.find(v->var_) == single_assign_.end());
                single_assign_[v->var_] = v->init_;
            } else {
                single_assign_[v->var_] = expr();
            }
        }
    }

    void view(assign_c v) override {
        if (v->var_.isa<var>() && v->var_->dtype_.lanes_ == 1) {
            auto itr = single_assign_.find(v->var_);
            if (itr != single_assign_.end()) {
                if (itr->second.defined()) {
                    // the var is already assigned elsewhere, it is not a
                    // single-assign var
                    itr->second = v->var_; // mark it
                } else {
                    itr->second = v->value_;
                }
            }
        }
    }
};

template <typename T>
union_val make_val(T) = delete;

static union_val make_val(float t) {
    union_val a;
    a.f32 = t;
    return a;
}

static union_val make_val(uint64_t t) {
    union_val a;
    a.u64 = t;
    return a;
}

static union_val make_val(int64_t t) {
    union_val a;
    a.s64 = t;
    return a;
}

static union_val make_val(bool t) {
    union_val a;
    a.u64 = t ? 1 : 0;
    return a;
}

template <typename T>
struct extract_val_t {
    static T doit(union_val) = delete;
};

template <>
struct extract_val_t<float> {
    static float doit(union_val v) { return v.f32; }
};

template <>
struct extract_val_t<uint64_t> {
    static uint64_t doit(union_val v) { return v.u64; }
};
template <>
struct extract_val_t<int64_t> {
    static int64_t doit(union_val v) { return v.s64; }
};

template <>
struct extract_val_t<bool> {
    static bool doit(union_val v) { return v.u64; }
};

template <typename SrcT, typename DestT>
static union_val cast_dispatched(union_val val) {
    return make_val(static_cast<DestT>(extract_val_t<SrcT>::doit(val)));
}

template <typename T>
expr create_cast(sc_data_type_t to_dtype, type_category to_cate,
        const std::vector<union_val> &v) {
    std::vector<union_val> ret;
    ret.reserve(v.size());
    union_val (*dispatch)(union_val val);
    switch (to_cate) {
        case CATE_FLOAT: {
            dispatch = cast_dispatched<T, float>;
            break;
        }
        case CATE_INT: {
            dispatch = cast_dispatched<T, int64_t>;
            break;
        }
        case CATE_UINT: {
            dispatch = cast_dispatched<T, uint64_t>;
            break;
        }
        default: COMPILE_ASSERT(0, "Bad cast to " << to_dtype); return expr();
    }
    for (auto val : v) {
        ret.push_back(dispatch(val));
    }
    return make_expr<constant_node>(ret, to_dtype);
}

bool is_const_equal_to(const constant_c &v, int64_t V) {
    auto cate = get_type_category(v->dtype_);
    switch (cate) {
        case CATE_FLOAT: {
            float outv = v->value_[0].f32;
            return outv == V;
        }
        case CATE_INT: {
            int64_t outv = v->value_[0].s64;
            return outv == V;
        }
        case CATE_UINT: {
            uint64_t outv = v->value_[0].u64;
            return outv == static_cast<uint64_t>(V);
        }
        default: assert(0 && "Bad category"); return false;
    }
}

template <typename T, typename... Args>
static size_t check_size_equals(const T &v0) {
    return v0.size();
}

template <typename T, typename... Args>
static size_t check_size_equals(const T &v0, const Args &... args) {
    auto ret = check_size_equals(args...);
    if (v0.size() == 1UL) { return ret; }
    if (ret == 1UL) { return v0.size(); }
    COMPILE_ASSERT(
            v0.size() == ret, "number of constant value elements mismatch");
    return ret;
}

static union_val extract_const_value(
        const std::vector<union_val> &v, size_t idx) {
    if (idx < v.size()) { return v[idx]; }
    return v[0];
}

template <typename T>
static T extract_typed_value(const std::vector<union_val> &v, size_t idx) {
    return extract_val_t<T>::doit(extract_const_value(v, idx));
}

template <typename R, typename... A>
R ret_helper(R (*)(A...));

// decay an lambda to function pointer
template <typename R, typename FirstArg, typename... A>
FirstArg first_arg_helper(R (*)(FirstArg, A...));

template <typename FuncT, typename... Args>
static std::vector<union_val> execute_on_values_impl(
        FuncT func, const Args &... args) {
    using FirstArg = decltype(first_arg_helper(func));
    size_t sz = check_size_equals(args...);
    std::vector<union_val> ret;
    ret.reserve(sz);
    auto first_val = func(extract_typed_value<FirstArg>(args, 0)...);
    ret.push_back(make_val(first_val));
    bool is_same = true;
    for (size_t i = 1; i < sz; i++) {
        auto cur_val = func(extract_typed_value<FirstArg>(args, i)...);
        ret.push_back(make_val(cur_val));
        is_same &= (cur_val == first_val);
    }
    if (is_same) { ret.resize(1); }
    return ret;
}

template <typename FuncT, typename... Args>
static std::vector<union_val> execute_on_values(
        FuncT func, const Args &... args) {
    // the FuncT can be a lambda.
    // the +func trick converts the func to a function pointer
    return execute_on_values_impl(+func, args...);
}

std::vector<union_val> execute_logic_binary(sc_expr_type op,
        const std::vector<union_val> &a, const std::vector<union_val> &b) {
    switch (op) {
        case sc_expr_type::logic_and:
            return execute_on_values(
                    [](bool a, bool b) { return a && b; }, a, b);
        case sc_expr_type::logic_or:
            return execute_on_values(
                    [](bool a, bool b) { return a || b; }, a, b);
        default: assert(0 && "Unknown logic OP"); return {};
    };
}

template <typename T>
T execute_mod(T a, T b) {
    return a % b;
}

template <>
float execute_mod(float a, float b) {
    COMPILE_ASSERT(0, "%% cannot be applied on float type");
    return 0;
}

template <typename T>
T execute_and(T a, T b) {
    return a & b;
}

template <typename T>
T execute_or(T a, T b) {
    return a | b;
}

template <>
float execute_and(float a, float b) {
    COMPILE_ASSERT(0, "& cannot be applied on float type");
    return 0;
}

template <>
float execute_or(float a, float b) {
    COMPILE_ASSERT(0, "| cannot be applied on float type");
    return 0;
}

#define DEF_COMPUTE(expr_) \
    execute_on_values([](T a, T b) { return (expr_); }, x, y);

template <typename T>
std::vector<union_val> execute_binary(sc_expr_type op, intrin_type intrin_op,
        const std::vector<union_val> &x, const std::vector<union_val> &y) {
    switch (op) {
        case sc_expr_type::add: return DEF_COMPUTE(a + b);
        case sc_expr_type::sub: return DEF_COMPUTE(a - b);
        case sc_expr_type::mul: return DEF_COMPUTE(a * b);
        case sc_expr_type::div: return DEF_COMPUTE(a / b);
        case sc_expr_type::mod: return execute_on_values(&execute_mod<T>, x, y);
        case sc_expr_type::intrin_call: {
            switch (intrin_op) {
                case intrin_type::min: return DEF_COMPUTE(a < b ? a : b);
                case intrin_type::max: return DEF_COMPUTE(a > b ? a : b);
                case intrin_type::int_and:
                    return execute_on_values(&execute_and<T>, x, y);
                case intrin_type::int_or:
                    return execute_on_values(&execute_or<T>, x, y);
                default: assert(0 && "Unknown OP");
            }
        }
        case sc_expr_type::cmp_eq: return DEF_COMPUTE(a == b);
        case sc_expr_type::cmp_ne: return DEF_COMPUTE(a != b);
        case sc_expr_type::cmp_lt: return DEF_COMPUTE(a < b);
        case sc_expr_type::cmp_le: return DEF_COMPUTE(a <= b);
        case sc_expr_type::cmp_gt: return DEF_COMPUTE(a > b);
        case sc_expr_type::cmp_ge: return DEF_COMPUTE(a >= b);
        default: assert(0 && "Unknown OP"); return {};
    }
}

expr compute_constexpr(
        const constant_c &cl, const constant_c &cr, const expr_c &parent) {
    COMPILE_ASSERT(cl->dtype_ == cr->dtype_,
            "LHS and RHS should have the same type: " << parent);
    if (parent.instanceof <logic>()) {
        COMPILE_ASSERT(cl->dtype_.type_code_ == sc_data_etype::BOOLEAN,
                "logic op should have boolean operands: " << parent);
        auto res = execute_logic_binary(
                parent->node_type_, cl->value_, cr->value_);
        return make_expr<constant_node>(res, cl->dtype_);
    }
    type_category ty = get_etype_category_nothrow(cl->dtype_.type_code_);
    auto op = parent->node_type_;
    intrin_type intrin_op = intrin_type::NUM_INTRINSICS;
    if (op == intrin_call_node::type_code_)
        intrin_op = parent.static_as<intrin_call_c>()->type_;
    std::vector<union_val> val;
    switch (ty) {
        case CATE_FLOAT:
            val = execute_binary<float>(op, intrin_op, cl->value_, cr->value_);
            break;
        case CATE_UINT:
            val = execute_binary<uint64_t>(
                    op, intrin_op, cl->value_, cr->value_);
            break;
        case CATE_INT:
            val = execute_binary<int64_t>(
                    op, intrin_op, cl->value_, cr->value_);
            break;
        default:
            COMPILE_ASSERT(0, "Type of binary op: " << parent);
            return expr();
    }
    return make_expr<constant_node>(val, parent->dtype_);
}

bool is_op_commutative_and_associative(const expr_c &v) {
    if (v->node_type_ == sc_expr_type::intrin_call) {
        switch (v.static_as<intrin_call_c>()->type_) {
            case intrin_type::max:
            case intrin_type::min:
            case intrin_type::int_and:
            case intrin_type::int_or: return true;
            default: return false;
        }
    }
    switch (v->node_type_) {
        case sc_expr_type::add:
        case sc_expr_type::mul:
        case sc_expr_type::logic_and:
        case sc_expr_type::logic_or: return true;
        default: return false;
    }
}

std::pair<expr_c, expr_c> get_operand_from_binary(const expr_c &a) {
    if (a.instanceof <intrin_call_c>()) {
        auto v = a.static_as<intrin_call_c>();
        return std::make_pair(v->args_[0], v->args_[1]);
    }
    if (a.instanceof <binary_c>()) {
        auto v = a.static_as<binary_c>();
        return std::make_pair(v->l_, v->r_);
    }
    if (a.instanceof <cmp_c>()) {
        auto v = a.static_as<cmp_c>();
        return std::make_pair(v->l_, v->r_);
    }
    assert(a.instanceof <logic_c>());
    auto v = a.static_as<logic_c>();
    return std::make_pair(v->l_, v->r_);
}

bool fold_special_consts(expr_c &orig, expr_c l, const constant_c &r) {
    // todo: handle vector types
    if (r->is_vector()) return false;
    sc_expr_type op = orig->node_type_;
    if (r->dtype_ == datatypes::boolean) {
        bool val = r->value_[0].u64;
        switch (op) {
            case sc_expr_type::logic_and:
                if (val) {
                    // x && 1 = x
                    orig = std::move(l);
                    return true;
                } else {
                    // X && 0 = 0
                    orig = make_expr<constant_node>(
                            uint64_t(0), datatypes::boolean);
                    return true;
                }
                break;
            case sc_expr_type::logic_or:
                if (val) {
                    // x || 1 = 1
                    orig = make_expr<constant_node>(
                            uint64_t(1), datatypes::boolean);
                    return true;
                } else {
                    // X || 0 = X
                    orig = std::move(l);
                    return true;
                }
                break;
            default: {
            };
        }
        return false;
    }

    if (is_const_equal_to(r, 0)) {
        switch (op) {
            case sc_expr_type::add:
            case sc_expr_type::sub: orig = std::move(l); return true;
            case sc_expr_type::mul:
                orig = make_expr<constant_node>(uint64_t(0), orig->dtype_);
                return true;
            default: {
            };
        }
    }
    if (is_const_equal_to(r, 1)) {
        switch (op) {
            case sc_expr_type::mul:
            case sc_expr_type::div: orig = std::move(l); return true;
            case sc_expr_type::mod:
                orig = make_expr<constant_node>(uint64_t(0), orig->dtype_);
                return true;
            default: {
            };
        }
    }
    return false;
}
} // namespace constant_folding

using namespace constant_folding;

/**
 * It will do the following (c as constant, "+" as an example):
 * c1 + c2 => c3
 * c + x => x + c
 * (x + c1) + c2 => x + (c1 + c2)
 * (x + c) + y => (x + y) + c
 * x + (y + c) => (x + y) + c
 * (x + c1) + (y + c2) => (x + y) + (c1 + c2)
 *
 * Also fold special expr:
 * a (+ - * && ||) 0/false
 * a (* / % && ||) 1/true
 * a (- / % && || max min > >= < <= == !=) a
 * */
class constant_fold_t : public ir_consistent_visitor_t {
public:
    using ir_consistent_visitor_t::dispatch;
    using ir_consistent_visitor_t::visit;
    // a comparer with strict var/tensor comparison
    ir_comparer cmper;
    bool fast_;
    constant_fold_t(bool fast) : cmper(false, true, true, false), fast_(fast) {}

    bool is_same_op(expr_c &v1, expr_c &v2) {
        if (v1->node_type_ != v2->node_type_) return false;
        if (v1->node_type_ == sc_expr_type::intrin_call)
            return v1.static_as<intrin_call_c>()->type_
                    == v2.static_as<intrin_call_c>()->type_;
        return true;
    }

    // try to rotate by the rotation rule.
    // returns true if rotation succeed
    bool try_rotate_const(expr_c &parent, expr_c &l, expr_c &r) {
        if (!is_op_commutative_and_associative(parent)) return false;
        if (l.isa<constant>() && !r.isa<constant>()) {
            // c + x => x + c
            std::swap(l, r);
            return true;
        }
        if (is_same_op(parent, l) && !l.isa<constant>() && r.isa<constant>()) {
            // (x + c1) + c2 => x + (c1 + c2)
            auto v = get_operand_from_binary(l);
            if (v.second.isa<constant>()) {
                auto c1 = v.second.static_as<constant_c>();
                r = compute_constexpr(c1, r.static_as<constant_c>(), parent);
                l = v.first;
                return true;
            }
        }
        if (!l.isa<constant>() && !r.isa<constant>()) {
            if (is_same_op(parent, l) && !is_same_op(parent, r)) {
                auto v = get_operand_from_binary(l);
                if (v.second.isa<constant>()) {
                    // (x + c) + y => (x + y) + c
                    l = builder::remake_binary(v.first, r, parent);
                    r = v.second;
                    return true;
                }
            }
            if (!is_same_op(parent, l) && is_same_op(parent, r)) {
                auto v = get_operand_from_binary(r);
                if (v.second.isa<constant>()) {
                    // x + (y + c) => (x + y) + c
                    l = builder::remake_binary(l, v.first, parent);
                    r = v.second;
                    return true;
                }
            }
            if (is_same_op(parent, l) && is_same_op(parent, r)) {
                auto vl = get_operand_from_binary(l);
                auto vr = get_operand_from_binary(r);
                if (vl.second.isa<constant>() && vr.second.isa<constant>()) {
                    // (x + c1) + (y + c2) => (x + y) + (c1 + c2)
                    l = builder::remake_binary(vl.first, vr.first, parent);
                    r = compute_constexpr(vl.second.checked_as<constant>(),
                            vr.second.checked_as<constant>(), parent);
                    return true;
                }
            }
        }
        return false;
    }

    // fold expr like a-a a/a a%a a&&a a||a min(a,a) max(a,a)
    // a!=a a>a ...
    bool fold_special_exprs(expr_c &parent, expr_c lhs, const expr_c &rhs) {
        switch (parent->node_type_) {
            case sc_expr_type::sub:
                if (cmper.compare(lhs, rhs)) {
                    parent = make_expr<constant_node>(0UL, parent->dtype_);
                    return true;
                }
                // fold a + b - c ==> b
                if (lhs.isa<add>()) {
                    if (cmper.compare(lhs.static_as<add>()->l_, rhs)) {
                        parent = lhs.static_as<add>()->r_;
                        return true;
                    }
                    if (cmper.compare(lhs.static_as<add>()->r_, rhs)) {
                        parent = lhs.static_as<add>()->l_;
                        return true;
                    }
                }
                break;
            case sc_expr_type::mod:
                if (cmper.compare(lhs, rhs)) {
                    parent = make_expr<constant_node>(0UL, parent->dtype_);
                    return true;
                }
                if (parent->dtype_.lanes_ > 1) {
                    // todo: handle vector types
                    return false;
                }
                if (rhs.isa<constant_c>()) {
                    int64_t rv1
                            = get_const_as_int(rhs.checked_as<constant_c>());
                    // fold i % C ==> i, if 0 <= i < C
                    if (auto rng = get_range_of_expr(lhs, fast_)) {
                        int64_t end_r = -1;
                        int64_t start_r = -1;
                        if (rng->cate == CATE_INT) {
                            end_r = rng->end.s64;
                            start_r = rng->start.s64;
                        } else if (rng->cate == CATE_UINT) {
                            end_r = rng->end.u64;
                            start_r = rng->start.u64;
                        }
                        if (rng->is_good_range() && end_r >= 0 && start_r >= 0
                                && rv1 > 0 && end_r < rv1) {
                            parent = lhs;
                            return true;
                        }
                    }
                    // fold (x * nC) % C = 0
                    if (lhs->node_type_ == sc_expr_type::mul) {
                        auto rhs_of_lhs = get_operand_from_binary(lhs).second;
                        if (rhs_of_lhs.isa<constant_c>()) {
                            int64_t rv2 = get_const_as_int(
                                    rhs_of_lhs.checked_as<constant_c>());
                            if (rv2 % rv1 == 0) {
                                parent = make_expr<constant_node>(
                                        0UL, parent->dtype_);
                                return true;
                            }
                        }
                    }
                    // fold (x %C) % C = x % C
                    else if (lhs->node_type_ == sc_expr_type::mod) {
                        auto r_l = get_operand_from_binary(lhs);
                        auto rhs_of_lhs = r_l.second;
                        if (rhs_of_lhs.isa<constant_c>()) {
                            int64_t rv2 = get_const_as_int(
                                    rhs_of_lhs.checked_as<constant_c>());
                            if (rv2 == rv1) {
                                parent = builder::make_mod(r_l.first, rhs);
                                return true;
                            }
                        }
                    }
                }
                break;
            case sc_expr_type::div:
                if (cmper.compare(lhs, rhs)) {
                    switch (get_type_category(parent->dtype_)) {
                        case CATE_INT:
                        case CATE_UINT:
                            parent = make_expr<constant_node>(
                                    1UL, parent->dtype_);
                            return true;
                        case CATE_FLOAT:
                            parent = make_expr<constant_node>(
                                    1.0f, parent->dtype_);
                            return true;
                        default: assert(0 && "Bad type"); return false;
                    }
                }
                break;
            case sc_expr_type::intrin_call:
                // todo(xxx): fold &0 |1
                bool can_fold;
                switch (parent.static_as<intrin_call>()->type_) {
                    case intrin_type::max:
                    case intrin_type::min:
                    case intrin_type::int_and:
                    case intrin_type::int_or: can_fold = true; break;
                    default: can_fold = false;
                }
                if (!can_fold) break;
                // if can_fold, fall through
            case sc_expr_type::logic_and:
            case sc_expr_type::logic_or:
                if (cmper.compare(lhs, rhs)) {
                    parent = std::move(lhs);
                    return true;
                }
                break;
            case sc_expr_type::cmp_eq:
            case sc_expr_type::cmp_le:
            case sc_expr_type::cmp_ge:
                if (cmper.compare(lhs, rhs)) {
                    parent = make_expr<constant_node>(1UL, parent->dtype_);
                    return true;
                }
                break;
            case sc_expr_type::cmp_ne:
            case sc_expr_type::cmp_lt:
            case sc_expr_type::cmp_gt:
                if (cmper.compare(lhs, rhs)) {
                    parent = make_expr<constant_node>(0UL, parent->dtype_);
                    return true;
                }
                break;
            default: break;
        }
        return false;
    }

    // fold x/c1/c2 => x /(c1*c2)
    bool fold_successive_div(expr_c &orig, expr_c &l, const expr_c &r) {
        sc_expr_type op = orig->node_type_;
        if (op != sc_expr_type::div) { return false; }
        if (is_same_op(orig, l) && !l.isa<constant>() && r.isa<constant>()) {
            auto v = get_operand_from_binary(l);
            if (v.second.isa<constant>()) {
                auto c1 = v.second.static_as<constant_c>();
                orig = builder::make_div(v.first,
                        compute_constexpr(c1, r.static_as<constant_c>(),
                                builder::make_mul(c1, r)));
                return true;
            }
        }
        return false;
    }

    /** expand Polynomial function
     *  e.g. ((a+b)*c+d)*e = a*c*e+b*c*e+d*e
     *                  *
     *                 / \
     *                +   e
     *               / \
     *              *   d
     *             / \
     *            +   c
     *           / \
     *          a   b
     * */
    expr_c expand_polynomial(expr_c parent) {
        if (parent->dtype_.lanes_ > 1) {
            // todo: handle vector types
            return false;
        }
        switch (parent->node_type_) {
            case sc_expr_type::mul:
            case sc_expr_type::div:
            case sc_expr_type::mod: {
                // TODO(xxx): support (a+b)*(c+d)
                auto l_r = get_operand_from_binary(parent);
                if (!l_r.second.isa<constant_c>()) {
                    break;
                } else {
                    constant_c rv = l_r.second.checked_as<constant_c>();
                    switch (l_r.first->node_type_) {
                        case sc_expr_type::add:
                        case sc_expr_type::sub: {
                            // TODO(xxx): special case for distribution law of
                            // Integer division
                            if (parent->node_type_ == sc_expr_type::div) {
                                auto l_r = get_operand_from_binary(parent);
                                return builder::remake_binary(
                                        expand_polynomial(l_r.first),
                                        expand_polynomial(l_r.second), parent);
                            } else {
                                auto next_lr
                                        = get_operand_from_binary(l_r.first);
                                auto new_parent = builder::remake_binary(
                                        expand_polynomial(
                                                builder::remake_binary(
                                                        next_lr.first, rv,
                                                        parent)),
                                        expand_polynomial(
                                                builder::remake_binary(
                                                        next_lr.second, rv,
                                                        parent)),
                                        l_r.first);
                                if (parent->node_type_ == sc_expr_type::mod)
                                    return builder::remake_binary(
                                            new_parent, l_r.second, parent);
                                else {
                                    return new_parent;
                                }
                            }
                        }
                        case sc_expr_type::mul:
                        case sc_expr_type::div:
                        case sc_expr_type::mod: {
                            if (parent->node_type_ == sc_expr_type::mod) {
                                if (fold_special_exprs(
                                            parent, l_r.first, l_r.second)) {
                                    return expand_polynomial(parent);
                                }
                            }
                            auto next_lr = get_operand_from_binary(l_r.first);
                            // folding
                            if (next_lr.second.isa<constant_c>()) {
                                auto new_parent = expand_polynomial(
                                        builder::remake_binary(
                                                expand_polynomial(
                                                        next_lr.first),
                                                next_lr.second, l_r.first));
                                return builder::remake_binary(
                                        new_parent, l_r.second, parent);
                            } else {
                                break;
                            }
                        }
                        default: return parent;
                    }
                }
                break;
            }
            case sc_expr_type::add:
            case sc_expr_type::sub: {
                auto l_r = get_operand_from_binary(parent);
                return builder::remake_binary(expand_polynomial(l_r.first),
                        expand_polynomial(l_r.second), parent);
            }
            default: return parent;
        }
        return parent;
    }

    expr_c fold_binary_impl(
            expr_c parent, const expr_c &lhs, const expr_c &rhs) {
        auto l = fold_range_dispatch(lhs);
        auto r = fold_range_dispatch(rhs);

        if (l.isa<constant>() && r.isa<constant>()) {
            auto cl = l.static_as<constant_c>();
            auto cr = r.static_as<constant_c>();
            return compute_constexpr(cl, cr, parent);
        }
        try_rotate_const(parent, l, r);
        if (r.isa<constant>()) {
            if (fold_special_consts(parent, l, r.static_as<constant>())) {
                return parent;
            }
        }
        if (fold_special_exprs(parent, l, r)) { return parent; }
        if (fold_successive_div(parent, l, r)) { return parent; }

        mark_range_for_const(l, fast_);
        mark_range_for_const(r, fast_);
        auto l_range = get_range_of_expr(l, fast_);
        auto r_range = get_range_of_expr(r, fast_);
        constant_fold_analysis_result_t new_range;
        bool successful_infer = false;
        if (r_range) {
            const_range_t rg;
            successful_infer = compute_range(parent, l_range, r_range, rg);
            if (successful_infer) {
                if (rg.is_single_value()) {
                    return make_expr<constant_node>(rg.start, parent->dtype_);
                }
            }
            new_range.set_range(rg);
        }

        expr_c ret;
        if (!l.ptr_same(lhs) || !r.ptr_same(rhs)) {
            ret = builder::remake_binary(l, r, parent);
        } else {
            ret = parent;
        }
        if (successful_infer && !get_range_of_expr(ret, fast_)) {
            ret->temp_data() = new_range;
        }
        return ret;
    }

    // run fold_binary_impl repeatedly on the expr until no changes happen
    expr_c fold_binary(expr_c parent) {
        expr_c old = parent;
        auto parent_type = parent->node_type_;
        constexpr int max_iter = 5000;
        int loop_cnt = 0;
        for (;;) {
            auto l_r = get_operand_from_binary(parent);
            expr_c ret = fold_binary_impl(parent, l_r.first, l_r.second);
            bool isT = ret->node_type_ == parent_type;
            if (ret.ptr_same(old) || !isT) { return ret; }
            parent = ret;
            old = std::move(ret);
            loop_cnt++;
            COMPILE_ASSERT(loop_cnt < max_iter,
                    "Constant folder reaches max iteration time. Either the "
                    "expression is too complicated or it is a bug of the "
                    "constant folder.")
        }
    }

    expr_c fold_range_dispatch(const expr_c &in) {
        auto v = dispatch(in);
        if (v.isa<constant>()) { return v; }
        if (auto data = get_range_of_expr(v, fast_)) {
            if (data->is_single_value()) {
                return make_expr<constant_node>(data->start, v->dtype_);
            }
        }
        return v;
    }

    // expr_c visit(cast_c v) override {
    // }
    expr_c visit(cast_c v) override {
        auto in = fold_range_dispatch(v->in_);
        bool changed = !in.ptr_same(v->in_);
        if (in.isa<constant>()) {
            auto inconst = in.as<constant_c>();
            type_category fromty
                    = get_etype_category_nothrow(inconst->dtype_.type_code_);
            type_category toty
                    = get_etype_category_nothrow(v->dtype_.type_code_);
            if (fromty != CATE_OTHER && toty != CATE_OTHER) {
                switch (fromty) {
                    case CATE_FLOAT:
                        return create_cast<float>(
                                v->dtype_, toty, inconst->value_);
                        break;
                    case CATE_UINT:
                        return create_cast<uint64_t>(
                                v->dtype_, toty, inconst->value_);
                        break;
                    case CATE_INT:
                        return create_cast<int64_t>(
                                v->dtype_, toty, inconst->value_);
                        break;
                    default:
                        COMPILE_ASSERT(0, "Bad cast from " << inconst->dtype_);
                        return expr();
                }
            }
        }
        expr_c ret;
        if (changed) {
            ret = copy_attr(*v, builder::make_cast(v->dtype_, in));
        } else {
            ret = v;
        }
        if (auto ana = get_range_of_expr(in, fast_)) {
            auto cur_cate = get_type_category(v->dtype_);
            if (ana->cate != CATE_FLOAT && cur_cate != CATE_FLOAT
                    && !get_range_of_expr(ret, fast_)) {
                ret->temp_data() = constant_fold_analysis_result_t {
                        const_range_t {cur_cate, ana->start, ana->end}};
            }
        }
        return ret;
    }

    expr_c visit(binary_c v) override { return fold_binary(v); }
    expr_c visit(cmp_c v) override { return fold_binary(v); }
    expr_c visit(logic_c v) override { return fold_binary(v); }
    expr_c visit(intrin_call_c v) override {
        auto ret = ir_consistent_visitor_t::visit(std::move(v));
        if (ret.isa<intrin_call>()) {
            auto node = ret.static_as<intrin_call_c>();
            switch (node->type_) {
                case intrin_type::max:
                case intrin_type::min:
                case intrin_type::int_and:
                case intrin_type::int_or: return fold_binary(node);
                default: break;
            }
        }
        return ret;
    }
    expr_c visit(logic_not_c v) override {
        auto in = fold_range_dispatch(v->in_);
        bool changed = !in.ptr_same(v->in_);
        if (in.isa<constant>()) {
            auto inconst = in.as<constant>();
            if (inconst->is_vector()) return v;
            COMPILE_ASSERT(inconst->dtype_ == datatypes::boolean,
                    "logic_not should have a boolean operand: " << v);
            uint64_t v = inconst->value_[0].u64 ? 0 : 1;
            return make_expr<constant_node>(v, datatypes::boolean);
        }
        if (changed) {
            return copy_attr(*v, builder::make_logic_not(in));
        } else {
            return v;
        }
    }
    expr_c visit(select_c v) override {
        auto ret = ir_consistent_visitor_t::visit(std::move(v));
        if (ret.isa<select>()) {
            auto node = ret.static_as<select_c>();
            auto &cond = node->cond_;
            // Currently only eliminate scalar constant select
            if (cond->dtype_ == datatypes::boolean && cond.isa<constant>()) {
                auto c = cond.static_as<constant_c>();
                bool is_false = is_const_equal_to(c, 0);
                return is_false ? node->r_ : node->l_;
            }
        }
        return ret;
    }

    stmt_c visit(for_loop_c v) override {
        auto old_fast = fast_;
        fast_ |= (v->attr_
                && v->attr_->get_or_else("bypass_complex_const_fold", false));
        // don't fold range for the var
        auto var = dispatch(v->var_);
        auto begin = fold_range_dispatch(v->iter_begin_);
        auto end = fold_range_dispatch(v->iter_end_);
        auto step = fold_range_dispatch(v->step_);

        mark_range_for_const(begin, fast_);
        mark_range_for_const(end, fast_);
        if (step.isa<constant>()) {
            auto stepc = get_const_as_int(step.static_as<constant_c>());
            auto begin_r = get_range_of_expr(begin, fast_);
            auto end_r = get_range_of_expr(end, fast_);
            if (stepc > 0 && begin_r && end_r) {
                int64_t max_loop_len = end_r->end.s64 - begin_r->start.s64;
                int64_t real_loop_len = max_loop_len / stepc * stepc;
                if (max_loop_len > 0 && real_loop_len > 0) {
                    var->temp_data() = constant_fold_analysis_result_t {
                            const_range_t {get_type_category(var->dtype_),
                                    begin_r->start,
                                    begin_r->start.s64
                                            + (real_loop_len - 1) * stepc}};
                }
            }
        }

        int64_t loop_len_hint = -1;
        if (!fast_ && v->attr_) {
            loop_len_hint = v->attr_->get_or_else("loop_len_hint", INT64_C(-1));
        }
        // try to fold the for range like for(i = A to A+1 step 1) {}
        if (!fast_ && loop_len_hint == -1) {
            if (!begin.isa<constant>() && !end.isa<constant>()) {
                expr_c real_begin
                        = get_assigned_expr(begin.get())->node_ptr_from_this();
                expr_c real_end
                        = get_assigned_expr(end.get())->node_ptr_from_this();
                auto ths = this;
                auto try_fold = [ths, &loop_len_hint](const expr_c &beg_v,
                                        const expr_c &end_v,
                                        const expr_c &step_v) -> bool {
                    auto loop_len = ths->fold_range_dispatch(
                            (end_v - beg_v) / step_v);
                    if (loop_len.isa<constant>()) {
                        loop_len_hint = get_expr_as_int(loop_len);
                        return true;
                    }
                    return false;
                };
                if (try_fold(real_begin, real_end, step)) {
                    // fall through
                } else if (!real_end.ptr_same(end)
                        && try_fold(real_begin, end, step)) {
                    // fall through
                } else if (!real_begin.ptr_same(begin)
                        && try_fold(begin, real_end, step)) {
                    // fall through
                }
            }
        }

        auto body = dispatch(v->body_);

        bool changed = !(var.ptr_same(v->var_) && begin.ptr_same(v->iter_begin_)
                && end.ptr_same(v->iter_end_) && step.ptr_same(v->step_)
                && body.ptr_same(v->body_));
        stmt ret;
        if (changed) {
            ret = copy_attr(*v,
                    builder::make_for_loop_unattached(var, begin, end, step,
                            body, v->incremental_, v->kind_, v->num_threads_));
        } else {
            ret = std::move(v).remove_const();
        }
        if (loop_len_hint >= 0) {
            ret->attr()["loop_len_hint"] = loop_len_hint;
        }
        fast_ = old_fast;
        return ret;
    }

    stmt_c visit(if_else_c v) override {
        auto cond = fold_range_dispatch(v->condition_);
        auto thencase = dispatch(v->then_case_);

        stmt_c elsecase;
        if (v->else_case_.defined()) elsecase = dispatch(v->else_case_);
        bool changed = !cond.ptr_same(v->condition_)
                || !elsecase.ptr_same(v->else_case_)
                || !thencase.ptr_same(v->then_case_);
        if (v->condition_.isa<constant>()) {
            assert(!v->condition_.as<constant>()->is_vector());
            COMPILE_ASSERT(v->condition_->dtype_ == datatypes::boolean,
                    "IfElse node expects an boolean expr as the condition, got "
                            << v->condition_->dtype_ << " expr = " << v);
            bool val = v->condition_.as<constant>()->value_[0].u64;
            if (val) {
                return v->then_case_;
            } else {
                if (v->else_case_.defined()) { return v->else_case_; }
                return make_stmt<stmts_node_t>(std::vector<stmt>());
            }
        }
        if (changed) {
            return copy_attr(*v,
                    builder::make_if_else_unattached(cond, thencase, elsecase));
        }
        return v;
    }

    func_c dispatch(func_c v) override {
        if (!fast_) {
            func_c cur_f = v;
            func_c ret;
            for (;;) {
                constant_fold_analysis_t ana;
                ana.dispatch(cur_f);
                // keep the exprs alive to make sure the raw pointers in
                // analysis result is valid
                auto keep_alive = std::move(ana.single_assign_);
                ret = ir_visitor_t::dispatch(cur_f);
                if (ret == cur_f) {
                    for (auto &kv : keep_alive) {
                        if (kv.first->temp_data_) {
                            kv.first->temp_data_->clear();
                        }
                    }
                    return ret;
                }
                cur_f = ret;
            }
        }
        return ir_visitor_t::dispatch(v);
    }
};

func_c constant_folder_t::operator()(func_c f) const {
    constant_fold_t pass {fast_};
    return pass.dispatch(std::move(f));
}

stmt_c constant_folder_t::operator()(stmt_c f) const {
    constant_fold_t pass {fast_};
    return pass.dispatch(std::move(f));
}

expr_c constant_folder_t::operator()(expr_c f) const {
    constant_fold_t pass {fast_};
    return pass.dispatch(std::move(f));
}

/**
 *  this feature is currently used to fold the index of reshape/reorder output,
 * so additional folding is added before and after expand_polynomial. TODO: move
 *  this feature to constant folding pass
 *  @param f: original polynomial expr.
 *  @param max_iter: maximum iteration time, default is one.
 * */
expr_c constant_folder_t::expand_polynomial(expr_c f, int max_iter) {
    constant_fold_t pass {true};
    auto ret = pass.dispatch(std::move(f));
    for (int i = 0; i < max_iter; i++) {
        auto old = ret;
        ret = pass.expand_polynomial(old);
        if (ret.ptr_same(old)) { break; }
    }
    return pass.dispatch(ret);
}

const_ir_module_ptr constant_folder_t::operator()(const_ir_module_ptr f) {
    constant_fold_t pass {fast_};
    return dispatch_module_on_visitor(&pass, f);
}

expr do_cast_and_fold(const expr &in) {
    static auto_caster_t caster;
    static constant_folder_t folder {false};
    return folder(caster(in)).remove_const();
}

expr_c do_cast_and_fold(const expr_c &in) {
    return do_cast_and_fold(in.remove_const());
}

} // namespace sc
