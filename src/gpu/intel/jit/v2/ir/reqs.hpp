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
    prover_t(prb_reqs_t *parent) : parent_(parent) {}
    bool require(const expr_t &e) const;
    explicit operator bool() const { return parent_; }

private:
    prb_reqs_t *parent_ = nullptr;
};

class req_impl_t;

// Problem requirements: a list of expressions expressing a set of requirements
// to the problem sizes.
class prb_reqs_t {
public:
    friend class prover_t;

    void add(const expr_t &e);
    void add(const prb_reqs_t &other);
    void add(const prb_tile_t &tile);
    void set(const prb_dim_t &dim, int value);
    prover_t prover(bool enable = true);

    explicit operator bool() const { return !reqs_.empty(); }
    // Checks if the requirements are satisfied for the given problem sizes .
    bool fits(const prb_tile_t &sizes) const;
    // Simplifies and eliminates redundant requirements.
    void simplify();
    // Checks if an expression/condition is an implication of the requirements.
    // For example: prb_reqs_t(oc % 64 == 0) implies (oc % 16) == 0 so the
    // latter can be proven from the original requirements.
    bool can_prove(const expr_t &to_prove) const;
    bool can_prove(const req_impl_t &to_prove) const;
    bool get_value(const prb_dim_t &dim, int &value) const;
    bool is_equal(const prb_dim_t &dim, int value) const;
    // Checks if other prb_reqs_t object is fully implied from the requirements
    // of this object.
    bool implies(const prb_reqs_t &other) const;
    expr_t to_expr(const prb_dim_t &dim) const;
    void stringify(std::ostream &out) const;
    void parse(std::istream &in);
    std::string str() const;

    IR_DEFINE_DUMP()

private:
    // Single requirement, represented as an expression.
    class req_t {
    public:
        req_t();
        req_t(const req_impl_t &impl);
        const req_impl_t &impl() const { return *impl_; }
        req_impl_t &impl() { return *impl_; }

    private:
        std::shared_ptr<req_impl_t> impl_;
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
