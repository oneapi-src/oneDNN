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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_AUTO_CAST_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_AUTO_CAST_HPP

#include "../module_pass.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * Adds cast_nodes to the IR to legalize it. It promotes the type if is
 * necessary and is allowed.
 * @see get_casting_priority
 * */
class auto_caster_t : public module_pass_t {
public:
    func_c operator()(func_c f);
    stmt_c operator()(stmt_c s);
    expr_c operator()(expr_c s);
    const_ir_module_ptr operator()(const_ir_module_ptr f) override;
    SC_DECL_PASS_INFO_FUNC();
};

/**
 * Gets the casting auto-promotion priority of a type. A type of lower priority
 * can be auto-cast to a type of larger priority
 *
 * @param dty the type
 * @return if dty is castable, the auto-promotion priority, should > 0.
 *      Otherwise, -1
 * */
int get_casting_priority(sc_data_type_t dty);

/**
 * Auto promote an expr to a type. If it may narrow down the type of the expr,
 * throw an error. If the expr is the same type of the target type, does nothing
 * @param v in and out expr
 * @param ty the target type
 * @param containing the expr containing the type, for showing a user-friendly
 * error message
 * @return true if the expr is changed for casting
 * */
bool cast_to(expr_c &v, sc_data_type_t ty, expr_c containing);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
