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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_TRANSFORM_X86_INTRINSICS_LOWERING_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_TRANSFORM_X86_INTRINSICS_LOWERING_HPP

#include <compiler/config/context.hpp>
#include <compiler/ir/function_pass.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

/* *
 * A prototype of x86 intric lowering, converting high level ir to proper low
 * level ir with accurate operands.
 * */
class x86_intrinsics_lowering_t : public function_pass_t {
public:
    x86_intrinsics_lowering_t(const runtime::target_machine_t &target_machine)
        : target_machine_(target_machine) {}
    func_c operator()(func_c v) override;

private:
    const runtime::target_machine_t &target_machine_;
};

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
