/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_INDEX2VAR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_INDEX2VAR_HPP

#include "../function_pass.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace attr_keys {
constexpr const char *no_index2var = "pass.no_index2var";
}
/**
 * Replace indexing on tensors with vars. "Cache" the values from/to tensors to
 * local variables. At the end of the scope (stmts node), the cached vars will
 * be written back to the tensors
 * @note This pass should run after function inlining pass
 * Currently this pass is very conservative. A tensor has only one "cache slot".
 * It will "flush" the var back to tensor if
 * 1. In the current/parent scopes, there are more than one different index on
 * the same tensor. e.g. if there is A[i] and then A[j] in use, we don't know if
 * A[i] and A[j] points to the same address, so we cannot cache both of them,
 * and need to "flush" the var for A[i] back to A[i]. However, if A[i] is the
 * only outstanding use of tensor A, we can use A[i] mutiple times and cache it
 * in a var. If `i` is changed between to uses of A[i], we also need to flush it
 * 2. There is a tensorptr on the tensor, the "cached" value of the tensor be
 * "flushed"
 * 3. The tensor is passed to a function call
 * 4. The lifetime of the cached var is over at the end of the scope
 * 5. There is a write to a tensor which is cached in parent scope
 * */
class index2var_t : public function_pass_t {
public:
    func_c operator()(func_c f) override;
    stmt_c operator()(const stmts_c &f);
    SC_DECL_PASS_INFO_FUNC();
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
