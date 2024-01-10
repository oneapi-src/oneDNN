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
    bool prove(const expr_t &e) const;
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

class prb_reqs_t {
public:
    friend class prover_t;

    void add(const expr_t &e);
    void add(const prb_reqs_t &other);
    prover_t prover(bool enable = true);

    explicit operator bool() const { return !reqs_.empty(); }
    bool fits(const prb_tile_t &sizes) const;
    void simplify();
    void serialize(std::ostream &out) const;
    void deserialize(std::istream &in);
    std::string str() const;

    IR_DEFINE_DUMP()

private:
    struct req_t {
        req_expr_t expr;

        req_t() = default;
        req_t(const req_expr_t &expr) : expr(expr) {}
        bool fits(const prb_tile_t &sizes) const;
        void serialize(std::ostream &out) const;
        void deserialize(std::istream &in);
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
