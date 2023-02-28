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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASSLET_SSA_VALUE_HASH_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASSLET_SSA_VALUE_HASH_HPP

#include "passlet.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace passlet {

/**
 * The passlet to assign a hash value to each ssa var definition. This passlet
 * is used for value-numbering. It will get the same hash value when two SSA
 * "trees" are the same, even if some of the intermediate SSA var nodes has
 * different name but with same value. For commutative operators like mul, add,
 * and, or, it uses unordered hashing to ensure hash("a+b") == hash("b+a")
 * @note please make sure to initialize hash value by 0 in default
 * constructor of the result
 * @param stmt_result_func the addresser for stmt->size_t
 * */
struct ssa_value_hash_t : public typed_passlet<size_t> {
    using parent = typed_passlet<size_t>;
    using parent::typed_addresser_t;
    ssa_value_hash_t(const typed_addresser_t &stmt_result_func)
        : parent {nullptr, stmt_result_func} {}
    void view(const define_c &v, pass_phase phase) override;
};
} // namespace passlet
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
