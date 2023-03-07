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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_TRANSFORM_CALL_TRANSFORM_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_TRANSFORM_CALL_TRANSFORM_HPP

#include <compiler/ir/function_pass.hpp>
#include <compiler/jit/xbyak/x86_64/abi_function_interface.hpp>
#include <compiler/jit/xbyak/x86_64/target_profile.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

namespace attr_keys {
// attr for func and call scope, contains x86 calling convention info
// data type: x86_64::abi_function_interface::ptr
constexpr const char *abi_interface = "abi_interface";
} // namespace attr_keys

#define TRANSFORMED_CALL(STMT) \
    (STMT->attr_ && STMT->attr_->has_key(attr_keys::abi_interface))

// Cached ABI infomation inside func and call node
x86_64::abi_function_interface::ptr cached_func_abi_interface(const func_t &v);
x86_64::abi_function_interface::ptr cached_call_abi_interface(const call_c &v);

/* *
 * Extract call node as root expr node and make dedicated stmts for each call
 * node, mark the stmts as function call scope, add reverse order args inside
 * scpoe. The process is aimed to transform all function call temp calculations
 * into a local scope, so that the register utilization can be much higher.
 * */
class call_transform_t : public function_pass_t {
public:
    call_transform_t(const x86_64::target_profile_t &profile);
    func_c operator()(func_c v) override;

private:
    const x86_64::target_profile_t &profile_;
};

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
