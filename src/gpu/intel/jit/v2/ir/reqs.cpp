/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include "gpu/intel/jit/v2/ir/reqs.hpp"

#include "gpu/intel/jit/ir/linear_expr.hpp"
#include "gpu/intel/jit/pass/simplify.hpp"

#include <iostream>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {

bool is_a_mod_b_eq_0(const expr_t &e, expr_t &a, expr_t &b) {
    auto *eq_op = e.as_ptr<binary_op_t>();
    if (!eq_op || eq_op->op_kind != op_kind_t::_eq) return false;
    if (!is_zero(eq_op->b)) return false;
    auto *mod_op = eq_op->a.as_ptr<binary_op_t>();
    if (!mod_op || mod_op->op_kind != op_kind_t::_mod) return false;
    a = mod_op->a;
    b = mod_op->b;
    return true;
}

bool is_a_eq_b_or_c_eq_d(
        const expr_t &e, expr_t &a, expr_t &b, expr_t &c, expr_t &d) {
    auto *or_op = e.as_ptr<binary_op_t>();
    if (!or_op || or_op->op_kind != op_kind_t::_or) return false;
    auto &op0 = or_op->a;
    auto &op1 = or_op->b;
    auto *eq_op0 = op0.as_ptr<binary_op_t>();
    auto *eq_op1 = op1.as_ptr<binary_op_t>();
    if (!eq_op0 || eq_op0->op_kind != op_kind_t::_eq) return false;
    if (!eq_op1 || eq_op1->op_kind != op_kind_t::_eq) return false;
    a = eq_op0->a;
    b = eq_op0->b;
    c = eq_op1->a;
    d = eq_op1->b;
    return true;
}

class linear_cmp_simplifier_t : public ir_mutator_t {
public:
    object_t _mutate(const binary_op_t &obj) override {
        if (!is_cmp_op(obj.op_kind)) return ir_mutator_t::_mutate(obj);

        if (!is_const(obj.b)) return obj;
        dim_t a_div = linear_max_pow2_divisor(obj.a);
        dim_t b_div = to_cpp<dim_t>(obj.b);
        dim_t factor = math::gcd(a_div, b_div);
        if (factor == 1) return obj;

        auto a = linear_div(obj.a, factor);
        auto b = b_div / factor;
        return binary_op_t::make(obj.op_kind, a, b);
    }
};

expr_t simplify_expr(const expr_t &_e) {
    expr_t a, b;
    if (is_a_mod_b_eq_0(_e, a, b)) {
        a = simplify_expr(a);
        if (is_const(b)) return simplify_linear_mod(a, to_cpp<int>(b)) == 0;
        return a % b == 0;
    }
    auto e = _e;
    for (int i = 0; i < 2; i++) {
        e = simplify_rewrite(e);
        e = linear_cmp_simplifier_t().mutate(e);
        e = const_fold(e);
    }
    return e;
}

// Represents a product of dimensions.
class req_lhs_t {
public:
    req_lhs_t() = default;
    explicit req_lhs_t(const pvar_t &pvar) : pvars_ {pvar} {}
    req_lhs_t(const pvar_t &pvar0, const pvar_t &pvar1)
        : pvars_ {pvar0, pvar1} {}
    explicit req_lhs_t(const std::vector<pvar_t> &pvars) : pvars_(pvars) {
        std::sort(pvars_.begin(), pvars_.end());
    }
    explicit req_lhs_t(const expr_t &e) : pvars_(split(e)) {
        std::sort(pvars_.begin(), pvars_.end());
        // Duplicates are not expected.
        for (int i = 1; i < size(); i++) {
            gpu_assert(pvars_[i] != pvars_[i - 1]);
        }
    }

    int size() const { return (int)pvars_.size(); }
    const std::vector<pvar_t> &pvars() const { return pvars_; }
    const pvar_t &operator[](int idx) const { return pvars_[idx]; }
    bool operator==(const req_lhs_t &other) const {
        return pvars_ == other.pvars_;
    }
    bool operator==(const pvar_t &pvar) const {
        return size() == 1 && (*this)[0] == pvar;
    }
    bool operator!=(const req_lhs_t &other) const { return !operator==(other); }

    template <typename T>
    T to_int(const pvar_map_t<T> &values) const {
        T value = 1;
        for (auto &pvar : pvars_) {
            value *= values.at(pvar);
        }
        return value;
    }

    bool is_ge_1() const {
        for (auto &pvar : pvars_) {
            // All dimensions take positive values except padding and dilation.
            if (is_dilation(pvar)) return false;
            if (is_padding(pvar)) return false;
        }
        return true;
    }

    bool has(const pvar_t &pvar) const {
        for (auto &p : pvars_)
            if (p == pvar) return true;
        return false;
    }

    bool has(const req_lhs_t &other) const {
        for (int i = 0; i < other.size(); i++) {
            if (!has(other[i])) return false;
        }
        return true;
    }

    int substitute(const pvar_map_t<dim_t> &values) {
        int factor = 1;
        for (auto &v : values) {
            for (int i = 0; i < size(); i++) {
                if (pvars_[i] == v) {
                    pvars_.erase(pvars_.begin() + i);
                    factor *= values[v];
                    break;
                }
            }
        }
        return factor;
    }

    void stringify(std::ostream &out) const { stringify_impl(out); }

    void stringify_impl(std::ostream &out, const std::string &sep = "*") const {
        gpu_assert(size() > 0);
        bool is_first = true;
        for (auto &p : pvars_) {
            if (!is_first) out << sep;
            out << p.str();
            is_first = false;
        }
    }

    void parse(std::istream &in) {
        auto s = jit::parse<std::string>(in);
        auto parts = gpu_utils::split(s, "*");
        std::vector<expr_t> args;
        for (auto &p : parts) {
            pvars_.push_back(pvar_t(p));
        }
    }

    std::string str() const {
        if (pvars_.empty()) return "(empty)";
        std::ostringstream oss;
        stringify_impl(oss, " * ");
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    static std::vector<pvar_t> split(const expr_t &e) {
        if (auto *var = e.as_ptr<const_var_t>())
            return {pvar_t::from_var(*var)};
        if (auto *op = e.as_ptr<binary_op_t>()) {
            gpu_assert(op->op_kind == op_kind_t::_mul);
            auto a_params = split(op->a);
            auto b_params = split(op->b);
            a_params.insert(a_params.end(), b_params.begin(), b_params.end());
            return a_params;
        }
        gpu_error_not_expected() << "Unknown expression: " << e;
        return {};
    }

    std::vector<pvar_t> pvars_;
};

class req_rhs_entry_t {
public:
    req_rhs_entry_t() = default;
    explicit req_rhs_entry_t(dim_t value) : value_(value) { is_undef_ = false; }
    explicit req_rhs_entry_t(const pvar_t &pvar) : pvar_(pvar) {
        is_undef_ = false;
    }
    explicit req_rhs_entry_t(const expr_t &e) {
        if (is_const(e)) {
            value_ = to_cpp<int>(e);
        } else {
            pvar_ = pvar_t::from_var(e);
            gpu_assert(!pvar_.is_undef()) << e;
        }
        is_undef_ = false;
    }
    explicit req_rhs_entry_t(const std::string &s) {
        if (!s.empty() && std::isdigit(s[0])) {
            value_ = std::stoi(s);
        } else {
            pvar_ = pvar_t(s);
            gpu_assert(!pvar_.is_undef()) << s;
        }
        is_undef_ = false;
    }
    bool is_undef() const { return is_undef_; }
    bool is_pvar() const { return !is_undef_ && !pvar_.is_undef(); }
    bool is_value() const { return !is_undef_ && pvar_.is_undef(); }
    const pvar_t &pvar() const {
        gpu_assert(is_pvar());
        return pvar_;
    }
    dim_t value() const {
        gpu_assert(is_value());
        return value_;
    }
    template <typename T>
    T to_int(const pvar_map_t<T> &values) const {
        gpu_assert(!is_undef());
        if (is_value()) return value_;
        return values.at(pvar_);
    }

    bool operator==(const req_rhs_entry_t &other) const {
        return (is_undef_ == other.is_undef_) && (pvar_ == other.pvar_)
                && (value_ == other.value_);
    }

    bool operator!=(const req_rhs_entry_t &other) const {
        return !operator==(other);
    }

    std::string str() const {
        std::ostringstream oss;
        if (is_pvar()) {
            oss << pvar_.str();
        } else {
            oss << value_;
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    bool is_undef_ = true;
    pvar_t pvar_;
    dim_t value_ = 0;
};

class req_rhs_t {
public:
    req_rhs_t() = default;
    explicit req_rhs_t(dim_t value) { entries_[0] = req_rhs_entry_t(value); }
    explicit req_rhs_t(const expr_t &e) { entries_[0] = req_rhs_entry_t(e); }
    explicit req_rhs_t(
            const req_rhs_entry_t &e0, const req_rhs_entry_t &e1 = {}) {
        entries_[0] = e0;
        entries_[1] = e1;
        gpu_assert(!entries_[0].is_undef());
    }
    req_rhs_t(const expr_t &e0, const expr_t &e1)
        : req_rhs_t(req_rhs_entry_t(e0), req_rhs_entry_t(e1)) {}
    bool is_undef() const { return size() == 0; }
    bool is_pvar() const { return size() == 1 && entries_[0].is_pvar(); }
    bool is_value() const { return size() == 1 && entries_[0].is_value(); }
    const pvar_t &pvar() const {
        gpu_assert(is_pvar());
        return entries_[0].pvar();
    }
    dim_t value() const {
        gpu_assert(is_value());
        return entries_[0].value();
    }
    int size() const {
        return !entries_[0].is_undef() + !entries_[1].is_undef();
    }
    const req_rhs_entry_t &operator[](int idx) const {
        gpu_assert(idx >= 0 && idx < size());
        return entries_[idx];
    }
    bool operator==(const req_rhs_t &other) const {
        for (int i = 0; i < max_entries; i++) {
            if (entries_[i] != other.entries_[i]) return false;
        }
        return true;
    }
    bool operator!=(const req_rhs_t &other) const { return !operator==(other); }

    template <typename T>
    T to_int(const pvar_map_t<T> &values) const {
        T ret = 1;
        for (int i = 0; i < size(); i++) {
            ret *= entries_[i].to_int(values);
        }
        return ret;
    }

    void substitute(const pvar_map_t<dim_t> &values) {
        dim_t value = 1;
        pvar_t pvar;
        for (int i = 0; i < size(); i++) {
            if (entries_[i].is_value()) {
                value *= entries_[i].value();
            } else {
                if (values.has(entries_[i].pvar())) {
                    value *= values.at(entries_[i].pvar());
                } else {
                    gpu_assert(pvar.is_undef());
                    pvar = entries_[i].pvar();
                }
            }
        }
        operator=(req_rhs_t());
        int idx = 0;
        if (value != 1) entries_[idx++] = req_rhs_entry_t(value);
        if (!pvar.is_undef()) entries_[idx++] = req_rhs_entry_t(pvar);
        if (size() == 0) entries_[0] = req_rhs_entry_t(1);
    }

    void stringify_impl(
            std::ostream &out, const std::string &delim = {}) const {
        bool is_first = true;
        for (int i = 0; i < size(); i++) {
            if (!is_first) out << delim << "*" << delim;
            out << entries_[i].str();
            is_first = false;
        }
    }

    void stringify(std::ostream &out) const { stringify_impl(out); }

    void parse(std::istream &in) {
        auto s = jit::parse<std::string>(in);
        auto parts = gpu_utils::split(s, "*");
        gpu_assert(!parts.empty() && (int)parts.size() <= max_entries);
        for (int i = 0; i < (int)parts.size(); i++) {
            entries_[i] = req_rhs_entry_t(parts[i]);
        }
    }

    std::string str() const {
        if (is_undef()) return "(empty)";
        std::ostringstream oss;
        stringify_impl(oss, /*delim=*/" ");
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    static const int max_entries = 2;
    req_rhs_entry_t entries_[max_entries];
};

enum class req_kind_t : uint32_t {
    undef = 0,
    eq = 1,
    ge = 2,
    le = 4,
    mod_eq_0 = 8,
    _or_eq = 16,
};

static auto req_kind_names = nstl::to_array({
        make_enum_name(req_kind_t::undef, "undef"),
        make_enum_name(req_kind_t::eq, "=="),
        make_enum_name(req_kind_t::ge, ">="),
        make_enum_name(req_kind_t::le, "<="),
        make_enum_name(req_kind_t::mod_eq_0, "mod_eq_0"),
        make_enum_name(req_kind_t::_or_eq, "|"),
});
GPU_DEFINE_PARSE_ENUM(req_kind_t, req_kind_names)

bool has_req_op(const std::string &s) {
    for (const char *op : {"=", "%", "|"})
        if (s.find(op) != std::string::npos) return true;
    return false;
}

bool is_pvar_product(const expr_t &e) {
    if (auto *var = e.as_ptr<const_var_t>())
        return !pvar_t::from_var(*var).is_undef();
    if (auto *op = e.as_ptr<binary_op_t>()) {
        if (op->op_kind != op_kind_t::_mul) return false;
        return is_pvar_product(op->a) && is_pvar_product(op->b);
    }
    return false;
}

class req_impl_t {
public:
    req_impl_t() = default;
    req_impl_t(const expr_t &e) {
        if (try_init_mod_eq_0(e)) return;
        if (try_init_or_eq(e)) return;
        if (try_init(e)) return;
        gpu_error_not_expected() << "Cannot handle expression: " << e;
    }
    req_impl_t(req_kind_t kind, const req_lhs_t &lhs, const req_rhs_t &rhs)
        : kind_(kind), lhs_(lhs), rhs_(rhs) {}
    req_impl_t(req_kind_t kind, const req_lhs_t &lhs, int rhs)
        : kind_(kind), lhs_(lhs), rhs_(rhs) {}

    req_kind_t kind() const { return kind_; }
    bool is_undef() const { return kind_ == req_kind_t::undef; }
    const req_lhs_t &lhs() const { return lhs_; }
    const req_rhs_t &rhs() const { return rhs_; }

    void substitute(const pvar_map_t<dim_t> &values) {
        if (kind_ == req_kind_t::_or_eq) {
            auto a = req_impl_t(
                    req_kind_t::eq, req_lhs_t(lhs_[0]), req_rhs_t(rhs_[0]));
            auto b = req_impl_t(
                    req_kind_t::eq, req_lhs_t(lhs_[1]), req_rhs_t(rhs_[1]));
            a.substitute(values);
            b.substitute(values);
            if (a.is_undef() || b.is_undef()) *this = req_impl_t();
            return;
        }
        rhs_.substitute(values);
        if (rhs_.size() != 1) return;
        int factor = lhs_.substitute(values);
        if (factor != 1) {
            gpu_assert(rhs().value() % factor == 0);
            rhs_ = req_rhs_t(rhs().value() / factor);
        }
        if (lhs_.size() == 0) {
            // Fully reduced, check that the requirement evaluates to true and
            // reset it to skip later.
            gpu_assert(fits(pvar_map_t<dim_t>()));
            *this = req_impl_t();
        }
    }

    bool operator==(const req_impl_t &other) const {
        return (kind_ == other.kind_) && (lhs_ == other.lhs_)
                && (rhs_ == other.rhs_);
    }
    bool fits(const pvar_map_t<dim_t> &values) const {
        if (kind_ == req_kind_t::_or_eq) {
            int64_t lhs0 = values.at(lhs_[0]);
            int64_t lhs1 = values.at(lhs_[1]);
            int64_t rhs0 = rhs_[0].to_int(values);
            int64_t rhs1 = rhs_[1].to_int(values);
            return (lhs0 == rhs0) || (lhs1 == rhs1);
        }
        int64_t lhs_val = lhs_.to_int(values);
        int64_t rhs_val = rhs_.to_int(values);
        bool ret = false;
        switch (kind_) {
            case req_kind_t::eq: ret = lhs_val == rhs_val; break;
            case req_kind_t::ge: ret = lhs_val >= rhs_val; break;
            case req_kind_t::le: ret = lhs_val <= rhs_val; break;
            case req_kind_t::mod_eq_0: ret = (lhs_val % rhs_val) == 0; break;
            default: gpu_error_not_expected();
        }
        gpu_check(ret) << "Requirement is not satisfied: " << str()
                       << ". LHS evaluates to " << lhs_val
                       << ", RHS evaluates to " << rhs_val;
        return ret;
    }

    // Checks if the condition is an implication of the current
    // requirement.
    bool can_prove(const req_impl_t &other) const {
        if (*this == other) return true;
        if (can_prove_le_ge(other)) return true;
        if (can_prove_mod(other)) return true;
        return false;
    }

    void stringify(std::ostream &out) const { stringify_impl(out); }

    void stringify_impl(
            std::ostream &out, const std::string &delim = {}) const {
        switch (kind_) {
            case req_kind_t::eq:
            case req_kind_t::ge:
            case req_kind_t::le:
                lhs_.stringify(out);
                out << delim << to_string(kind_) << delim;
                rhs_.stringify(out);
                break;
            case req_kind_t::mod_eq_0:
                lhs_.stringify(out);
                out << delim << "%" << delim;
                rhs_.stringify(out);
                out << delim << "==" << delim << "0";
                break;
            case req_kind_t::_or_eq:
                out << lhs_[0] << delim << "==" << delim << rhs_[0];
                out << delim << to_string(kind_) << delim;
                out << lhs_[1] << delim << "==" << delim << rhs_[1];
                break;
            default: gpu_error_not_expected() << "kind: " << to_string(kind_);
        }
    }

    void parse(std::istream &in) {
        auto s = jit::parse<std::string>(in);
        for (req_kind_t op : {req_kind_t::_or_eq, req_kind_t::eq,
                     req_kind_t::ge, req_kind_t::le}) {
            auto s_op = to_string(op);
            auto pos = s.find(s_op);
            if (pos == std::string::npos) continue;
            auto s_lhs = s.substr(0, pos);
            auto s_rhs = s.substr(pos + s_op.length());
            auto mod_pos = s_lhs.find("%");
            expr_t lhs;
            if (mod_pos != std::string::npos) {
                gpu_assert(op == req_kind_t::eq);
                kind_ = req_kind_t::mod_eq_0;
                auto s_mod_lhs = s_lhs.substr(0, mod_pos);
                auto s_mod_rhs = s_lhs.substr(mod_pos + 1);
                lhs_ = jit::parse<req_lhs_t>(s_mod_lhs);
                rhs_ = jit::parse<req_rhs_t>(s_mod_rhs);
            } else {
                kind_ = op;
                if (op == req_kind_t::_or_eq) {
                    auto a = jit::parse<req_impl_t>(s_lhs);
                    auto b = jit::parse<req_impl_t>(s_rhs);
                    lhs_ = req_lhs_t(a.lhs()[0], b.lhs()[0]);
                    rhs_ = req_rhs_t(a.rhs()[0], b.rhs()[0]);
                } else {
                    lhs_ = jit::parse<req_lhs_t>(s_lhs);
                    rhs_ = jit::parse<req_rhs_t>(s_rhs);
                }
            }
            return;
        }
        gpu_error_not_expected() << s;
    }

    std::string str() const {
        if (kind_ == req_kind_t::undef) return "(empty)";
        std::ostringstream oss;
        stringify_impl(oss, /*delim=*/" ");
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    bool try_init_mod_eq_0(const expr_t &e) {
        expr_t a, b;
        if (!is_a_mod_b_eq_0(e, a, b)) return false;
        if (!is_pvar_product(a)) return false;
        kind_ = req_kind_t::mod_eq_0;
        lhs_ = req_lhs_t(a);
        rhs_ = req_rhs_t(b);
        return true;
    }

    bool try_init_or_eq(const expr_t &e) {
        expr_t lhs0, rhs0;
        expr_t lhs1, rhs1;
        if (!is_a_eq_b_or_c_eq_d(e, lhs0, rhs0, lhs1, rhs1)) return false;
        kind_ = req_kind_t::_or_eq;
        lhs_ = req_lhs_t(pvar_t::from_var(lhs0), pvar_t::from_var(lhs1));
        rhs_ = req_rhs_t(rhs0, rhs1);
        return true;
    }

    bool try_init(const expr_t &e) {
        auto *op = e.as_ptr<binary_op_t>();
        if (!op) return false;
        if (!is_const(op->b)) return false;
        switch (op->op_kind) {
            case op_kind_t::_eq: kind_ = req_kind_t::eq; break;
            case op_kind_t::_ge: kind_ = req_kind_t::ge; break;
            case op_kind_t::_le: kind_ = req_kind_t::le; break;
            default: return false;
        }
        gpu_assert(is_const(op->b)) << "Unexpected non-const RHS: " << op->b;
        auto *div_a_op = op->a.as_ptr<binary_op_t>();
        if (div_a_op && div_a_op->op_kind == op_kind_t::_div) {
            if (!is_pvar_product(div_a_op->a)) return false;
            lhs_ = req_lhs_t(div_a_op->a);
            rhs_ = req_rhs_t(
                    req_rhs_entry_t(op->b), req_rhs_entry_t(div_a_op->b));
            return true;
        }
        if (!is_pvar_product(op->a)) return false;
        lhs_ = req_lhs_t(op->a);
        rhs_ = req_rhs_t(op->b);
        return true;
    }

    bool can_prove_le_ge(const req_impl_t &other) const {
        if (kind() != other.kind()) return false;
        bool is_le = (kind() == req_kind_t::le);
        bool is_ge = (kind() == req_kind_t::ge);
        if (!is_le && !is_ge) return false;
        if (rhs().size() != 1 || other.rhs().size() != 1) return false;
        if (lhs() == other.lhs()) {
            if (is_ge) return rhs().value() >= other.rhs().value();
            if (is_le) return rhs().value() <= other.rhs().value();
            return false;
        }
        switch (kind()) {
            case req_kind_t::le:
                // (a * b <= C) => (a <= C + x) if a >= 1 and b >= 1.
                if (!lhs().is_ge_1()) return false;
                if (other.rhs().value() < rhs().value()) return false;
                for (int i = 0; i < other.lhs().size(); i++) {
                    if (!lhs().has(other.lhs()[i])) return false;
                }
                return true;
            case req_kind_t::ge:
                // (a >= C + x) => (a * b >= C) if a >= 1 and b >= 1.
                if (!other.lhs().is_ge_1()) return false;
                if (other.rhs().value() > rhs().value()) return false;
                for (int i = 0; i < lhs().size(); i++) {
                    if (!other.lhs().has(lhs()[i])) return false;
                }
                return true;
            default: return false;
        }
    }

    bool can_prove_mod(const req_impl_t &other) const {
        // (a % (C * D) == 0) => ((a * b) % C == 0)
        if (kind_ != req_kind_t::mod_eq_0) return false;
        if (!rhs().is_value() || !other.rhs().is_value()) return false;
        if (!other.lhs_.has(lhs_)) return false;
        if (other.kind() == req_kind_t::mod_eq_0)
            return rhs().value() % other.rhs().value() == 0;
        // (a % (C + x) == 0) => ((a * b >= C) if (a * b) > 0
        if (other.kind() == req_kind_t::ge && other.lhs().is_ge_1())
            return rhs().value() >= other.rhs().value();
        return false;
    }

    req_kind_t kind_ = req_kind_t::undef;
    req_lhs_t lhs_;
    req_rhs_t rhs_;
};

void prb_reqs_t::add(const expr_t &_e) {
    auto e = simplify_expr(_e);
    if (auto *imm = e.as_ptr<bool_imm_t>()) {
        if (imm->value) return;
        gpu_error_not_expected() << _e;
    }
    add_if_not_found(req_impl_t(e));
}

void prb_reqs_t::add(const prb_reqs_t &other) {
    for (auto &r : other.reqs_)
        add_if_not_found(r.impl());
}

void prb_reqs_t::add(const pvar_map_t<dim_t> &values) {
    for (auto &v : values) {
        set(v, values[v]);
    }
}

void prb_reqs_t::add_no_simplify(const expr_t &e) {
    add_if_not_found(req_impl_t(e));
}

void prb_reqs_t::set(const pvar_t &pvar, dim_t value) {
    add(pvar.var() == value);
}

void prb_reqs_t::add_if_not_found(const req_impl_t &new_req) {
    for (auto &r : reqs_) {
        if (r.impl() == new_req) return;
    }
    reqs_.emplace_back(new_req);
}

prover_t prb_reqs_t::prover(const prb_reqs_t &parent, bool can_update) {
    return prover_t(&parent, this, can_update);
}

bool prb_reqs_t::fits(const pvar_map_t<dim_t> &values) const {
    for (auto &r : reqs_) {
        gpu_check(r.impl().fits(values));
    }
    return true;
}

void prb_reqs_t::stringify_impl(std::ostream &out, const std::string &req_delim,
        const std::string &delim) const {
    if (reqs_.empty()) {
        out << "x";
        return;
    }
    pvar_map_t<dim_t> var_eq_map;
    bool is_first = true;
    for (auto &r : reqs_) {
        if (r.impl().kind() == req_kind_t::eq) {
            var_eq_map[r.impl().lhs()[0]] = r.impl().rhs().value();
            continue;
        }
        if (!is_first) out << req_delim;
        r.impl().stringify_impl(out, delim);
        is_first = false;
    }
    if (!var_eq_map.is_empty()) {
        if (!is_first) out << req_delim;
        out << var_eq_map.str();
    }
}

void prb_reqs_t::stringify(std::ostream &out) const {
    stringify_impl(out, ":", "");
}

void prb_reqs_t::parse(std::istream &in) {
    reqs_.clear();
    auto s = stream_parse<std::string>(in);
    if (s == "x") return;
    auto parts = gpu_utils::split(s, ":");
    pvar_map_t<int> var_eq_map;
    for (auto &p : parts) {
        if (!has_req_op(p)) {
            var_eq_map = pvar_map_t<int>(p);
            continue;
        }
        auto ri = jit::parse<req_impl_t>(p);
        reqs_.emplace_back(ri);
    }
    for (auto &v : var_eq_map) {
        reqs_.emplace_back(req_impl_t(
                req_kind_t::eq, req_lhs_t(v), req_rhs_t(var_eq_map.at(v))));
    }
}

std::string prb_reqs_t::str() const {
    std::ostringstream oss;
    stringify_impl(oss, "\n", " ");
    return oss.str();
}

prb_reqs_t::req_t::req_t() : impl_(utils::make_unique<req_impl_t>()) {}
prb_reqs_t::req_t::req_t(const req_t &other)
    : impl_(utils::make_unique<req_impl_t>(other.impl())) {}
prb_reqs_t::req_t::req_t(const req_impl_t &impl)
    : impl_(utils::make_unique<req_impl_t>(impl)) {}
prb_reqs_t::req_t::~req_t() = default;
prb_reqs_t::req_t &prb_reqs_t::req_t::operator=(const req_t &other) {
    impl_ = utils::make_unique<req_impl_t>(other.impl());
    return *this;
}
std::string prb_reqs_t::req_t::str() const {
    return impl_->str();
}

void prb_reqs_t::simplify() {
    auto new_reqs = reqs_;
    // Drop redundant requirements.
    for (size_t i = 0; i < new_reqs.size(); i++) {
        auto &ri = new_reqs[i].impl();
        if (ri.is_undef()) continue;
        pvar_map_t<dim_t> sub;
        if (ri.kind() == req_kind_t::eq && ri.lhs().size() == 1) {
            sub[ri.lhs()[0]] = ri.rhs().value();
        }
        for (size_t j = 0; j < new_reqs.size(); j++) {
            auto &rj = new_reqs[j].impl();
            if (rj.is_undef() || i == j) continue;
            if (ri.can_prove(rj)) {
                rj = req_impl_t();
                continue;
            }
            // Propagate scalar values.
            if (!sub.is_empty() && rj.lhs().has(ri.lhs())) {
                rj.substitute(sub);
            }
        }
    }
    reqs_.clear();
    for (auto &r : new_reqs) {
        if (r.impl().is_undef()) continue;
        reqs_.push_back(r);
    }
    // Sort for deterministic representation.
    std::sort(reqs_.begin(), reqs_.end(), [](const req_t &a, const req_t &b) {
        return a.impl().str() < b.impl().str();
    });
}

void prb_reqs_t::substitute(const pvar_map_t<dim_t> &values) {
    for (auto &r : reqs_) {
        r.impl().substitute(values);
    }
    simplify();
}

bool prb_reqs_t::can_prove(const expr_t &to_prove) const {
    auto e = simplify_expr(to_prove);
    if (auto *imm = e.as_ptr<bool_imm_t>()) { return imm->value; }
    return can_prove(req_impl_t(e));
}

bool prb_reqs_t::can_prove(const req_impl_t &to_prove) const {
    for (auto &r : reqs_) {
        if (r.impl().can_prove(to_prove)) return true;
    }
    if (to_prove.kind() == req_kind_t::mod_eq_0 && to_prove.rhs().is_value()) {
        int mod = 1;
        for (int i = 0; i < to_prove.lhs().size(); i++) {
            auto &lhs_pvar = to_prove.lhs()[i];
            mod *= max_factor(lhs_pvar);
        }
        if (mod % to_prove.rhs().value() == 0) return true;
    }
    return false;
}

bool prb_reqs_t::get_value(const pvar_t &pvar, dim_t &value) const {
    for (auto &r : reqs_) {
        auto &ri = r.impl();
        if (ri.kind() == req_kind_t::eq && ri.lhs() == pvar) {
            value = ri.rhs().value();
            return true;
        }
    }
    return false;
}

dim_t prb_reqs_t::max_factor(const pvar_t &pvar) const {
    dim_t ret = 1;
    for (auto &r : reqs_) {
        auto &ri = r.impl();
        if (ri.kind() == req_kind_t::eq && ri.lhs() == pvar)
            return ri.rhs().value();
        if (ri.kind() == req_kind_t::mod_eq_0 && ri.lhs() == pvar
                && ri.rhs().is_value()) {
            ret = std::max(ret, ri.rhs().value());
        }
    }
    return ret;
}

bool prb_reqs_t::is_equal(const pvar_t &pvar, dim_t value) const {
    dim_t pvar_value;
    return get_value(pvar, pvar_value) && pvar_value == value;
}

bool prb_reqs_t::implies(const prb_reqs_t &other) const {
    for (auto &req : other.reqs_) {
        gpu_check(can_prove(req.impl())) << "Cannot prove: " << req.impl();
    }
    return true;
}

expr_t prb_reqs_t::to_expr(const pvar_t &pvar) const {
    dim_t pvar_value;
    if (get_value(pvar, pvar_value)) return pvar_value;
    return pvar.var();
}

const prover_t &prover_t::instance() {
    static prover_t _instance;
    return _instance;
}

bool prover_t::require(const expr_t &_e) const {
    auto e = simplify_expr(_e);
    if (auto *imm = e.as_ptr<bool_imm_t>()) return imm->value;

    req_impl_t ri(e);
    bool is_true = (parent_ && parent_->can_prove(ri));
    if (!is_true && !can_update_) return false;
    reqs_->add_if_not_found(ri);
    return true;
}

} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
