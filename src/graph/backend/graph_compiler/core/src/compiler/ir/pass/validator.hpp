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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_VALIDATOR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_VALIDATOR_HPP

#include "../module_pass.hpp"
#include "../sc_function.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * The validator who checks if the IR is legal
 * */
class validator_t : public module_pass_t {
public:
    const_ir_module_ptr operator()(const_ir_module_ptr f) override;
    func_c operator()(func_c f);
    expr_c operator()(expr_c f);
    stmt_c operator()(stmt_c f);
    SC_DECL_PASS_INFO_FUNC();
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
