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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_FUNC_INLINE_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_FUNC_INLINE_HPP

#include <vector>
#include "../function_pass.hpp"
#include "../sc_function.hpp"
#include <compiler/ir/pass_dep_util.hpp>

namespace sc {

/**
 * Inlines function calls with attr["inline_level"] = 2
 * Or manually inline a call_node
 * */
class func_inliner_t : public function_pass_t {
public:
    func_c operator()(func_c f) override;
    /**
     * Inlines the function call and inserts the body to an existing stmt array
     *
     * @param c the call_node to inline
     * @param seq the stmt array to insert into
     * @param insert_idx the index of the insertion point within `seq`
     * @return if the function returns a value, the return value of the call.
     *      Otherwise, null
     * */
    expr_c inline_at(call_c c, std::vector<stmt> &seq, size_t insert_idx,
            const std::vector<define> &module_vars = std::vector<define>());
    SC_DECL_PASS_INFO_FUNC();
};

} // namespace sc

#endif
