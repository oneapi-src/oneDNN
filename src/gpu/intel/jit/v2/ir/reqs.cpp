/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

bool is_a_mod_b_eq_0(const expr_t &e, expr_t &a, int &b) {
    auto *eq_op = e.as_ptr<binary_op_t>();
    if (!eq_op || eq_op->op_kind != op_kind_t::_eq) return false;
    if (!is_zero(eq_op->b)) return false;
    auto *mod_op = eq_op->a.as_ptr<binary_op_t>();
    if (!mod_op || mod_op->op_kind != op_kind_t::_mod) return false;
    if (!is_const(mod_op->b)) return false;
    a = mod_op->a;
    b = to_cpp<int>(mod_op->b);
    return true;
}

class linear_cmp_simplifier_t : public ir_mutator_t {
public:
    object_t _mutate(const binary_op_t &obj) override {
        if (!is_cmp_op(obj.op_kind)) return ir_mutator_t::_mutate(obj);

        if (!is_const(obj.b)) return obj;
        int a_div = linear_max_pow2_divisor(obj.a);
        int b_div = to_cpp<int>(obj.b);
        int factor = math::gcd(a_div, b_div);
        if (factor == 1) return obj;

        auto a = linear_div(obj.a, factor);
        auto b = b_div / factor;
        return binary_op_t::make(obj.op_kind, a, b);
    }
};

expr_t simplify_expr(const expr_t &_e) {
    expr_t a;
    int b;
    if (is_a_mod_b_eq_0(_e, a, b)) {
        return simplify_linear_mod(simplify_expr(a), b) == 0;
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
class pvar_product_t {
public:
    pvar_product_t() = default;
    explicit pvar_product_t(const expr_t &e) : params_(split(e)) {
        std::sort(params_.begin(), params_.end());
        // Duplicates are not expected.
        for (int i = 1; i < size(); i++) {
            ir_assert(params_[i] != params_[i - 1]);
        }
    }

    int size() const { return (int)params_.size(); }
    const pvar_t &operator[](int idx) const { return params_[idx]; }
    bool operator==(const pvar_product_t &other) const {
        return params_ == other.params_;
    }
    bool operator==(const pvar_t &param) const {
        return size() == 1 && (*this)[0] == param;
    }
    bool operator!=(const pvar_product_t &other) const {
        return !operator==(other);
    }

    int64_t to_int(const pvar_tile_t &sizes) const {
        int64_t value = 1;
        for (auto &param : params_) {
            value *= sizes.at(param);
        }
        return value;
    }

    bool is_ge_1() const {
        for (auto &param : params_) {
            // All dimensions take positive values except padding and dilation.
            if (is_dilation(param)) return false;
            if (is_padding(param)) return false;
        }
        return true;
    }

    int substitute(const pvar_tile_t &tile) {
        int factor = 1;
        for (auto &p : tile) {
            for (int i = 0; i < size(); i++) {
                if (params_[i] == p) {
                    params_.erase(params_.begin() + i);
                    factor *= tile[p];
                    break;
                }
            }
        }
        return factor;
    }

    void stringify(std::ostream &out) const { stringify_impl(out); }

    void stringify_impl(std::ostream &out, const std::string &sep = "*") const {
        ir_assert(size() > 0);
        bool is_first = true;
        for (auto &p : params_) {
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
            params_.push_back(pvar_t(p));
        }
    }

    std::string str() const {
        if (params_.empty()) return "(empty)";
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
            ir_assert(op->op_kind == op_kind_t::_mul);
            auto a_params = split(op->a);
            auto b_params = split(op->b);
            a_params.insert(a_params.end(), b_params.begin(), b_params.end());
            return a_params;
        }
        ir_error_not_expected() << "Unknown expression: " << e;
        return {};
    }

    std::vector<pvar_t> params_;
};

enum class req_kind_t : uint32_t {
    undef = 0,
    eq = 1,
    ge = 2,
    le = 4,
    mod_eq_0 = 8,
};

static auto req_kind_names = nstl::to_array({
        make_enum_name(req_kind_t::undef, "undef"),
        make_enum_name(req_kind_t::eq, "=="),
        make_enum_name(req_kind_t::ge, ">="),
        make_enum_name(req_kind_t::le, "<="),
        make_enum_name(req_kind_t::mod_eq_0, "mod_eq_0"),
});
GPU_DEFINE_PARSE_ENUM(req_kind_t, req_kind_names)

class req_impl_t {
public:
    req_impl_t() = default;
    req_impl_t(const expr_t &e) {
        if (try_init_mod_eq_0(e)) return;
        if (try_init(e)) return;
        ir_error_not_expected() << "Cannot handle expression: " << e;
    }

    req_kind_t kind() const { return kind_; }
    const pvar_product_t &lhs() const { return lhs_; }
    int rhs() const { return rhs_; }

    void substitute(const pvar_tile_t &tile) {
        int factor = lhs_.substitute(tile);
        ir_assert(rhs_ % factor == 0);
        rhs_ /= factor;
        if (lhs_.size() == 0) {
            // Fully reduced, check that the requirement evaluates to true and
            // reset it to skip later.
            ir_assert(fits(pvar_tile_t()));
            lhs_ = pvar_product_t();
            rhs_ = 0;
            kind_ = req_kind_t::undef;
        }
    }

    bool operator==(const req_impl_t &other) const {
        return (kind_ == other.kind_) && (lhs_ == other.lhs_)
                && (rhs_ == other.rhs_);
    }
    bool fits(const pvar_tile_t &sizes) const {
        int64_t lhs = lhs_.to_int(sizes);
        bool ret = false;
        switch (kind_) {
            case req_kind_t::eq: ret = lhs == rhs_; break;
            case req_kind_t::ge: ret = lhs >= rhs_; break;
            case req_kind_t::le: ret = lhs <= rhs_; break;
            case req_kind_t::mod_eq_0: ret = (lhs % rhs_) == 0; break;
            default: ir_error_not_expected();
        }
        ir_check(ret) << "Requirement is not satisfied: " << str()
                      << ". LHS evaluates to " << lhs;
        return ret;
    }

    // Checks if the condition is an implication of the current
    // requirement.
    bool can_prove(const req_impl_t &to_prove) const {
        if (*this == to_prove) return true;
        if (kind_ != to_prove.kind_) return false;
        if (lhs_ != to_prove.lhs_) return false;
        switch (kind_) {
            case req_kind_t::ge: return rhs_ >= to_prove.rhs_;
            case req_kind_t::le: return rhs_ <= to_prove.rhs_;
            case req_kind_t::mod_eq_0: return rhs_ % to_prove.rhs_ == 0;
            default: return false;
        }
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
                out << rhs_;
                break;
            case req_kind_t::mod_eq_0:
                lhs_.stringify(out);
                out << delim << "%" << delim << rhs_ << delim << "==" << delim
                    << "0";
                break;
            default: ir_error_not_expected() << "kind: " << to_string(kind_);
        }
    }

    void parse(std::istream &in) {
        auto s = jit::parse<std::string>(in);
        for (req_kind_t op : {req_kind_t::eq, req_kind_t::ge, req_kind_t::le}) {
            auto s_op = to_string(op);
            auto pos = s.find(s_op);
            if (pos == std::string::npos) continue;
            auto s_lhs = s.substr(0, pos);
            auto s_rhs = s.substr(pos + s_op.length());
            auto mod_pos = s_lhs.find("%");
            expr_t lhs;
            if (mod_pos != std::string::npos) {
                ir_assert(op == req_kind_t::eq);
                kind_ = req_kind_t::mod_eq_0;
                auto s_mod_lhs = s_lhs.substr(0, mod_pos);
                auto s_mod_rhs = s_lhs.substr(mod_pos + 1);
                lhs_ = jit::parse<pvar_product_t>(s_mod_lhs);
                rhs_ = std::stoi(s_mod_rhs);
            } else {
                kind_ = op;
                lhs_ = jit::parse<pvar_product_t>(s_lhs);
                rhs_ = std::stoi(s_rhs);
            }
            return;
        }
        ir_error_not_expected() << s;
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
        expr_t a;
        int b;
        if (!is_a_mod_b_eq_0(e, a, b)) return false;
        kind_ = req_kind_t::mod_eq_0;
        lhs_ = pvar_product_t(a);
        rhs_ = b;
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
        lhs_ = pvar_product_t(op->a);
        rhs_ = to_cpp<int>(op->b);
        return true;
    }

    req_kind_t kind_ = req_kind_t::undef;
    pvar_product_t lhs_;
    int rhs_ = 0;
};

void prb_reqs_t::add(const expr_t &_e) {
    auto e = simplify_expr(_e);
    if (auto *imm = e.as_ptr<bool_imm_t>()) {
        if (imm->value) return;
        ir_error_not_expected() << _e;
    }
    add_if_not_found(req_impl_t(e));
}

void prb_reqs_t::add(const prb_reqs_t &other) {
    for (auto &r : other.reqs_)
        add_if_not_found(r.impl());
}

void prb_reqs_t::add(const pvar_tile_t &tile) {
    for (auto &d : tile) {
        set(d, tile[d]);
    }
}

void prb_reqs_t::set(const pvar_t &pvar, int value) {
    add(pvar.var() == value);
}

void prb_reqs_t::set_any_mod(const pvar_t &pvar) {
    any_mods_.push_back(pvar);
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

bool prb_reqs_t::fits(const pvar_tile_t &sizes) const {
    for (auto &r : reqs_) {
        ir_check(r.impl().fits(sizes));
    }
    return true;
}

void prb_reqs_t::stringify(std::ostream &out) const {
    if (reqs_.empty()) {
        out << "x";
        return;
    }
    bool is_first = true;
    for (auto &r : reqs_) {
        if (!is_first) out << ":";
        r.impl().stringify(out);
        is_first = false;
    }
}

void prb_reqs_t::parse(std::istream &in) {
    reqs_.clear();
    auto s = stream_parse<std::string>(in);
    if (s == "x") return;
    auto parts = gpu_utils::split(s, ":");
    for (auto &p : parts) {
        auto ri = jit::parse<req_impl_t>(p);
        reqs_.emplace_back(ri);
    }
}

std::string prb_reqs_t::str() const {
    std::ostringstream oss;
    bool is_first = true;
    for (auto &r : reqs_) {
        if (!is_first) oss << "\n";
        oss << r.impl().str();
        is_first = false;
    }
    return oss.str();
}

prb_reqs_t::req_t::req_t() : impl_(std::make_shared<req_impl_t>()) {}
prb_reqs_t::req_t::req_t(const req_impl_t &impl)
    : impl_(std::make_shared<req_impl_t>(impl)) {}

void prb_reqs_t::simplify() {
    int default_mod = 1;
    int default_low = 0;
    int default_high = std::numeric_limits<int>::max();
    std::unordered_set<pvar_t> seen;
    for (auto &r : reqs_) {
        auto &lhs = r.impl().lhs();
        for (int i = 0; i < lhs.size(); i++) {
            seen.insert(lhs[i]);
        }
    }
    pvar_map_t<int> low_bound;
    pvar_map_t<int> high_bound;
    pvar_map_t<int> mod;
    pvar_map_t<uint32_t> mask;
    for (auto &pvar : seen) {
        low_bound[pvar] = default_low;
        high_bound[pvar] = default_high;
        mod[pvar] = default_mod;
        mask[pvar] = 0;
    }
    // Collect low/high bounds and modulus information for individual
    // dimensions.
    for (auto &r : reqs_) {
        auto &ri = r.impl();
        if (ri.lhs().size() != 1) continue;
        auto pvar = ri.lhs()[0];
        switch (ri.kind()) {
            case req_kind_t::mod_eq_0: {
                int &f = mod[pvar];
                f = std::max(f, ri.rhs());
                break;
            }
            case req_kind_t::eq:
                low_bound[pvar] = high_bound[pvar] = ri.rhs();
                break;
            case req_kind_t::le:
                high_bound[pvar] = std::min(high_bound[pvar], ri.rhs());
                break;
            case req_kind_t::ge:
                low_bound[pvar] = std::max(low_bound[pvar], ri.rhs());
                break;
            default: break;
        }
    }
    // Set masks based on known modulus information and bounds.
    for (auto &pvar : mask) {
        if (mod[pvar] != default_mod) {
            mask[pvar] |= static_cast<uint32_t>(req_kind_t::mod_eq_0);
        }
        if (low_bound[pvar] == high_bound[pvar]) {
            mask[pvar] |= static_cast<uint32_t>(req_kind_t::eq);
        } else if (low_bound[pvar] != default_low) {
            mask[pvar] |= static_cast<uint32_t>(req_kind_t::ge);
        } else if (high_bound[pvar] != default_high) {
            mask[pvar] |= static_cast<uint32_t>(req_kind_t::le);
        }
    }
    // Drop redundant requirements based on the collected restrictions.
    std::vector<req_t> new_reqs;
    for (auto &r : reqs_) {
        auto &ri = r.impl();
        switch (ri.kind()) {
            case req_kind_t::mod_eq_0: {
                int f = 1;
                for (int i = 0; i < ri.lhs().size(); i++) {
                    f *= mod.at(ri.lhs()[i]);
                }
                if (f % ri.rhs() == 0) continue;
                break;
            }
            case req_kind_t::eq:
                // Individual restrictions are to be added later.
                if (ri.lhs().size() == 1) continue;
                break;
            case req_kind_t::le:
                if (ri.lhs().size() == 1) continue;
                if (ri.lhs().is_ge_1()) {
                    // (a * b <= C + x) => (a <= C) if a >= 1 and b >= 1.
                    for (int i = 0; i < ri.lhs().size(); i++) {
                        auto pvar = ri.lhs()[i];
                        if (mask[pvar]
                                & static_cast<uint32_t>(req_kind_t::le)) {
                            if (high_bound[pvar] >= ri.rhs()) {
                                mask[pvar] &= ~static_cast<uint32_t>(
                                        req_kind_t::le);
                            }
                        }
                    }
                }
                break;
            case req_kind_t::ge: {
                if (ri.lhs().size() == 1) continue;
                bool skip = false;
                if (ri.lhs().is_ge_1()) {
                    // (a >= C + x) => (a * b >= C) if a >= 1 and b >= 1.
                    for (int i = 0; i < ri.lhs().size(); i++) {
                        auto pvar = ri.lhs()[i];
                        if (low_bound[pvar] >= ri.rhs()) {
                            skip = true;
                            break;
                        }
                    }
                }
                if (skip) continue;
            }
            default: break;
        }
        new_reqs.emplace_back(ri);
    }
    pvar_tile_t fixed_pvars;
    // Add requirements for individual dimensions based on the modulus data and
    // bounds.
    for (auto &d : mask) {
        if (mask[d] & static_cast<uint32_t>(req_kind_t::mod_eq_0)) {
            new_reqs.emplace_back(req_impl_t((d.var() % mod[d]) == 0));
        }
        if (mask[d] & static_cast<uint32_t>(req_kind_t::eq)) {
            fixed_pvars[d] = low_bound[d];
            new_reqs.emplace_back(req_impl_t(d.var() == low_bound[d]));
        }
        if (mask[d] & static_cast<uint32_t>(req_kind_t::ge)) {
            new_reqs.emplace_back(req_impl_t(d.var() >= low_bound[d]));
        }
        if (mask[d] & static_cast<uint32_t>(req_kind_t::le)) {
            new_reqs.emplace_back(req_impl_t(d.var() <= high_bound[d]));
        }
    }
    reqs_.clear();
    // Substitute exact values and add overwrite requirements.
    for (auto &r : new_reqs) {
        if (r.impl().lhs().size() != 1) r.impl().substitute(fixed_pvars);
        if (r.impl().kind() == req_kind_t::undef) continue;
        reqs_.push_back(r);
    }
    // Sort for deterministic representation.
    std::sort(reqs_.begin(), reqs_.end(), [](const req_t &a, const req_t &b) {
        return a.impl().str() < b.impl().str();
    });
}

bool prb_reqs_t::can_prove(const expr_t &to_prove) const {
    auto e = simplify_expr(to_prove);
    if (auto *imm = e.as_ptr<bool_imm_t>()) { return imm->value; }
    return can_prove(req_impl_t(e));
}

bool prb_reqs_t::can_prove(const req_impl_t &to_prove, bool use_any_mod) const {
    for (auto &r : reqs_) {
        if (r.impl().can_prove(to_prove)) return true;
    }
    if (to_prove.kind() == req_kind_t::mod_eq_0) {
        int mod = 1;
        for (int i = 0; i < to_prove.lhs().size(); i++) {
            auto &lhs_pvar = to_prove.lhs()[i];
            if (use_any_mod) {
                for (auto &pvar : any_mods_) {
                    if (pvar == lhs_pvar) return true;
                }
            }
            mod *= max_factor(lhs_pvar);
        }
        if (mod % to_prove.rhs() == 0) return true;
    }
    return false;
}

bool prb_reqs_t::get_value(const pvar_t &pvar, int &value) const {
    for (auto &r : reqs_) {
        auto &ri = r.impl();
        if (ri.kind() == req_kind_t::eq && ri.lhs() == pvar) {
            value = ri.rhs();
            return true;
        }
    }
    return false;
}

int prb_reqs_t::max_factor(const pvar_t &pvar) const {
    int ret = 1;
    for (auto &r : reqs_) {
        auto &ri = r.impl();
        if (ri.kind() == req_kind_t::mod_eq_0 && ri.lhs() == pvar) {
            ret = std::max(ret, ri.rhs());
        }
    }
    return ret;
}

bool prb_reqs_t::is_equal(const pvar_t &pvar, int value) const {
    int pvar_value;
    return get_value(pvar, pvar_value) && pvar_value == value;
}

bool prb_reqs_t::implies(const prb_reqs_t &other) const {
    for (auto &req : other.reqs_) {
        ir_check(can_prove(req.impl())) << "Cannot prove: " << req.impl();
    }
    return true;
}

expr_t prb_reqs_t::to_expr(const pvar_t &pvar) const {
    int pvar_value;
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
    bool is_true = (parent_ && parent_->can_prove(ri, /*use_any_mod=*/true));
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
