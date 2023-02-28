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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_SIMPLE_LICM_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_SIMPLE_LICM_HPP

#include "../function_pass.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace attr_key {
constexpr const char *const_attr = "pass.const";
}
/**
 * Simple LICM of non-SSA version, hoist tensor defined inside loop. If the
 * tensor has init values or dimensions related to loop vars, do not hoist.
 * */
class simple_loop_invariant_code_motion_t : public function_pass_t {
public:
    func_c operator()(func_c f) override;
    stmt_c operator()(stmt_c s);
    SC_DECL_PASS_INFO_FUNC();
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
