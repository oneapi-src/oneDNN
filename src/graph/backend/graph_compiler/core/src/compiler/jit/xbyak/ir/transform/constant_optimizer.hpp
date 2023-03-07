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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_TRANSFORM_CONSTANT_OPTIMIZER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_TRANSFORM_CONSTANT_OPTIMIZER_HPP

#include <compiler/ir/function_pass.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

namespace attr_keys {
// attr for constant value, mark if constant value need to be encoded
// data type: bool
constexpr const char *force_simd_encode = "force_simd_encode";
// attr for constant value, mark if simd will be directly load from memory
// data type: bool
constexpr const char *load_simd_value = "load_simd_value";
} // namespace attr_keys

#define FORCE_SIMD_ENCODE(EXPR) \
    (EXPR->attr_ \
            && EXPR->attr_->get_or_else(attr_keys::force_simd_encode, false))

/* *
 * Constant optimizer, mark simd constant and insert broadcast when needed.
 * Add simpile strength reduction for constant div/mod/mul.
 * */
class constant_optimizer_t : public function_pass_t {
public:
    constant_optimizer_t() = default;
    func_c operator()(func_c v) override;

private:
};

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
