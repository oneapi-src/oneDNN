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

#ifndef GPU_INTEL_JIT_V2_IR_REQS_HPP
#define GPU_INTEL_JIT_V2_IR_REQS_HPP

#include "gpu/intel/jit/ir/ir.hpp"
#include "gpu/intel/jit/ir/problem.hpp"

#include <iostream>
#include <memory>
#include <vector>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {

class prb_reqs_t;

class prover_t {
public:
    static const prover_t &instance();
    prover_t() = default;
    prover_t(const prb_reqs_t *parent, prb_reqs_t *reqs, bool can_update)
        : parent_(parent), reqs_(reqs), can_update_(can_update) {}
    prover_t(prover_t &other, bool can_update)
        : parent_(other.parent_), reqs_(other.reqs_), can_update_(can_update) {}
    // TODO: Change to non-const.
    bool require(const expr_t &e) const;
    const prb_reqs_t &reqs() const {
        ir_assert(reqs_);
        return *reqs_;
    }
    explicit operator bool() const { return reqs_; }
    bool can_update() const { return can_update_; }

private:
    const prb_reqs_t *parent_ = nullptr;
    prb_reqs_t *reqs_ = nullptr;
    bool can_update_ = false;
};

class req_impl_t;

// Problem requirements: a list of expressions expressing a set of requirements
// to the problem sizes.
class prb_reqs_t {
public:
    friend class prover_t;

    void add(const expr_t &e);
    void add(const prb_reqs_t &other);
    void add(const pvar_map_t<dim_t> &sizes);
    void set(const pvar_t &dim, dim_t value);
    prover_t prover(const prb_reqs_t &parent, bool can_update = true);

    explicit operator bool() const { return !reqs_.empty(); }
    // Checks if the requirements are satisfied for the given problem sizes .
    bool fits(const pvar_map_t<dim_t> &sizes) const;
    // Simplifies and eliminates redundant requirements.
    void simplify();
    void substitute(const pvar_map_t<dim_t> &values);
    // Checks if an expression/condition is an implication of the requirements.
    // For example: prb_reqs_t(oc % 64 == 0) implies (oc % 16) == 0 so the
    // latter can be proven from the original requirements.
    bool can_prove(const expr_t &to_prove) const;
    bool can_prove(const req_impl_t &to_prove) const;
    bool get_value(const pvar_t &dim, dim_t &value) const;
    dim_t max_factor(const pvar_t &dim) const;
    bool is_equal(const pvar_t &dim, dim_t value) const;
    // Checks if other prb_reqs_t object is fully implied from the requirements
    // of this object.
    bool implies(const prb_reqs_t &other) const;
    expr_t to_expr(const pvar_t &dim) const;
    void stringify_impl(std::ostream &out, const std::string &req_delim,
            const std::string &delim) const;
    void stringify(std::ostream &out) const;
    void parse(std::istream &in);
    std::string str() const;

    IR_DEFINE_DUMP()

    static void merge(std::vector<prb_reqs_t> reqs_vec,
            const std::vector<int> &factor_vec, const pvar_t &factor_dim,
            prb_reqs_t &out_reqs);

private:
    // Single requirement, represented as an expression.
    class req_t {
    public:
        req_t();
        req_t(const req_t &other);
        req_t(const req_impl_t &impl);
        ~req_t();
        req_t &operator=(const req_t &other);
        const req_impl_t &impl() const { return *impl_; }
        req_impl_t &impl() { return *impl_; }
        std::string str() const;
        IR_DEFINE_DUMP()

    private:
        std::unique_ptr<req_impl_t> impl_;
    };

    void add_if_not_found(const req_impl_t &new_req);

    std::vector<req_t> reqs_;
};

} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
