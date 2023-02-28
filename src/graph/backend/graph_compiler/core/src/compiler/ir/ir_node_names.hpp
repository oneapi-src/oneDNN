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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_IR_NODE_NAMES_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_IR_NODE_NAMES_HPP

#define SC_EXPAND(x) x

#define FOR_EACH_EXPR_IR_TYPE(F, ...) \
    SC_EXPAND(F(constant, __VA_ARGS__)) \
    SC_EXPAND(F(var, __VA_ARGS__)) \
    SC_EXPAND(F(cast, __VA_ARGS__)) \
    SC_EXPAND(F(add, __VA_ARGS__)) \
    SC_EXPAND(F(sub, __VA_ARGS__)) \
    SC_EXPAND(F(mul, __VA_ARGS__)) \
    SC_EXPAND(F(div, __VA_ARGS__)) \
    SC_EXPAND(F(mod, __VA_ARGS__)) \
    SC_EXPAND(F(cmp_eq, __VA_ARGS__)) \
    SC_EXPAND(F(cmp_ne, __VA_ARGS__)) \
    SC_EXPAND(F(cmp_lt, __VA_ARGS__)) \
    SC_EXPAND(F(cmp_le, __VA_ARGS__)) \
    SC_EXPAND(F(cmp_gt, __VA_ARGS__)) \
    SC_EXPAND(F(cmp_ge, __VA_ARGS__)) \
    SC_EXPAND(F(logic_and, __VA_ARGS__)) \
    SC_EXPAND(F(logic_or, __VA_ARGS__)) \
    SC_EXPAND(F(logic_not, __VA_ARGS__)) \
    SC_EXPAND(F(select, __VA_ARGS__)) \
    SC_EXPAND(F(indexing, __VA_ARGS__)) \
    SC_EXPAND(F(call, __VA_ARGS__)) \
    SC_EXPAND(F(tensor, __VA_ARGS__)) \
    SC_EXPAND(F(tensorptr, __VA_ARGS__)) \
    SC_EXPAND(F(intrin_call, __VA_ARGS__)) \
    SC_EXPAND(F(ssa_phi, __VA_ARGS__)) \
    SC_EXPAND(F(func_addr, __VA_ARGS__)) \
    SC_EXPAND(F(low_level_intrin, __VA_ARGS__))

#define FOR_EACH_BASE_EXPR_IR_TYPE(F, ...) \
    SC_EXPAND(F(binary, __VA_ARGS__)) \
    SC_EXPAND(F(logic, __VA_ARGS__)) \
    SC_EXPAND(F(cmp, __VA_ARGS__))

#define FOR_EACH_STMT_IR_TYPE(F, ...) \
    SC_EXPAND(F(assign, __VA_ARGS__)) \
    SC_EXPAND(F(stmts, __VA_ARGS__)) \
    SC_EXPAND(F(if_else, __VA_ARGS__)) \
    SC_EXPAND(F(evaluate, __VA_ARGS__)) \
    SC_EXPAND(F(for_loop, __VA_ARGS__)) \
    SC_EXPAND(F(returns, __VA_ARGS__)) \
    SC_EXPAND(F(define, __VA_ARGS__))

#endif
