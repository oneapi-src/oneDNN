/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_BACKEND_EXPR_LOCATION_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_BACKEND_EXPR_LOCATION_HPP

#include <sstream>
#include <utility>
#include <compiler/ir/sc_expr.hpp>
#include <compiler/jit/xbyak/backend/operand.hpp>
#include <compiler/jit/xbyak/configured_xbyak.hpp>
#include <compiler/jit/xbyak/ir/xbyak_expr.hpp>
#include <compiler/jit/xbyak/x86_64/native_types.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

class expr_location {
public:
    enum class type {
        none,
        imm,
        reg,
        stack_var,
        stack_tensor,
        simd_constant,
    };

    expr_location()
        : type_(expr_location::type::none)
        , data_type_(x86_64::cpu_data_type::void_t) {};

    type get_type() const;
    x86_64::cpu_data_type get_data_type() const;
    op_ptr_t get_op_ptr() const;

    int64_t get_imm() const;
    int64_t get_stack_var() const;
    int64_t get_stack_tensor() const;
    const Xbyak::Reg &get_reg() const;
    const Xbyak::Address &get_simd_constant() const;

    // Factory methods, for convenience.
    template <typename RegT>
    static expr_location make_reg( //
            RegT reg, x86_64::cpu_data_type cpu_dtype);
    static expr_location make_imm( //
            int64_t imm, x86_64::cpu_data_type cpu_dtype);
    static expr_location make_stack_var( //
            int64_t offset, x86_64::cpu_data_type cpu_dtype);
    static expr_location make_stack_tensor( //
            int64_t offset);
    static expr_location make_simd_constant( //
            Xbyak::Address addr, x86_64::cpu_data_type cpu_dtype);

    friend std::ostream &operator<<(std::ostream &os, const expr_location &v);

private:
    // only allow factory methods
    template <typename T>
    expr_location(type t, x86_64::cpu_data_type dtype, T op)
        : type_(t)
        , data_type_(dtype)
        , content_(wrap_op_ptr<T>(std::move(op))) {}

    type type_;
    x86_64::cpu_data_type data_type_;
    op_ptr_t content_;
};

std::ostream &operator<<(std::ostream &os, const expr_location &v);

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
