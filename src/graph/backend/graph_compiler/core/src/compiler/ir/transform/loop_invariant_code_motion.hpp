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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_LOOP_INVARIANT_CODE_MOTION_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_LOOP_INVARIANT_CODE_MOTION_HPP

#include <compiler/ir/function_pass.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * Do loop invariant code motion(LICM) in SSA  (Single-Static-Assignment)
 * form. It includes two steps:
 * 1. Analysis and mark loop-independent variables. Notice that we do LIP only
 * for local var (mainly process indexing of tensor), not for global var/tensor,
 * local tensor/indexing/tensorptr node or call node. Currently we treat an
 * if-else node as a whole, if an if-else node can not be hoisted, neither can
 * its inner vars.
 * 2. Hoist as many definitions of var out of loops as possible. The definitions
 * should be hoisted out of its inner loop as far as possible.
 * */
class loop_invariant_code_motion_t : public function_pass_t {
public:
    func_c operator()(func_c f) override;
    SC_DECL_PASS_INFO_FUNC();
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
