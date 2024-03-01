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

#ifndef GPU_JIT_V2_IR_PRB_REQS_HPP
#define GPU_JIT_V2_IR_PRB_REQS_HPP

#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/problem.hpp"

#include <iostream>
#include <memory>
#include <vector>

namespace dnnl {
namespace impl {
namespace gpu {
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

class req_expr_impl_t {
public:
    virtual ~req_expr_impl_t() = default;
    virtual ir_type_id_t expr_kind() const = 0;
    virtual int64_t to_int(const prb_tile_t &sizes) const = 0;
    virtual expr_t to_ir() const = 0;
    virtual void serialize(std::ostream &out) const = 0;
};

// Requirement expression, supports a subset of IR expressions.
// req_expr_t::to_ir() and to_req_expr() can be used to convert req_expr_t to
// expr_t and back. req_expr_t is mainly introduced to expose
// serialization/deserialization functionality.
class req_expr_t {
public:
    req_expr_t(req_expr_impl_t *impl = nullptr) : impl_(impl) {}
    explicit operator bool() const { return (bool)impl_; }

    template <typename T>
    T *as_ptr() {
        return static_cast<T *>(impl_.get());
    }

    template <typename T>
    const T *as_ptr() const {
        return static_cast<const T *>(impl_.get());
    }

    int64_t to_int(const prb_tile_t &sizes) const {
        return impl_->to_int(sizes);
    }
    expr_t to_ir() const { return impl_->to_ir(); }
    void serialize(std::ostream &out) const { return impl_->serialize(out); }
    void deserialize(std::istream &in);

private:
    std::shared_ptr<req_expr_impl_t> impl_;
};

// Problem requirements: a list of expressions expressing a set of requirements
// to the problem sizes.
class prb_reqs_t {
public:
    friend class prover_t;

    void add(const expr_t &e);
    void add(const prb_reqs_t &other);
    prover_t prover(bool enable = true);

    explicit operator bool() const { return !reqs_.empty(); }
    // Checks if the requirements are satisfied for the given problem sizes .
    bool fits(const prb_tile_t &sizes) const;
    // Simplifies and eliminates redundant requirements.
    void simplify();
    // Checks if an expression/condition is an implication of the requirements.
    // For example: prb_reqs_t(oc % 64 == 0) implies (oc % 16) == 0 so the
    // latter can be proven from the original requirements.
    bool can_prove(const expr_t &e) const;
    // Checks if other prb_reqs_t object is fully implied from the requirements
    // of this object.
    bool implies(const prb_reqs_t &other) const;
    void serialize(std::ostream &out) const;
    void deserialize(std::istream &in);
    std::string str() const;

    IR_DEFINE_DUMP()

private:
    // Single requirement, represented as an expression.
    struct req_t {
        req_expr_t expr;

        req_t() = default;
        req_t(const req_expr_t &expr) : expr(expr) {}
        bool fits(const prb_tile_t &sizes) const;
        // Checks if the condition is an implication of the current
        // requirement.
        bool can_prove(const expr_t &expr_to_prove) const;
        void serialize(std::ostream &out) const;
        void deserialize(std::istream &in);
        std::string str() const;
        IR_DEFINE_DUMP()
    };

    void add_if_not_found(const req_expr_t &e);

    std::vector<req_t> reqs_;
};

} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
