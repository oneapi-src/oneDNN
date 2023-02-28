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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_VIEWER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_VIEWER_HPP

#include <vector>
#include "visitor.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
/**
 * The base class of read-only passes.
 * Override the overloaded view() function that you are
 * interested. In view(), call dispatch() on each of the sub-node
 * of the IR node, if you need to "view" the sub-nodes.
 * @note Dispatch rule for super-classes and sub-classes: If you
 * override the view() function of a sub-class (e.g. add), and you
 * call dispatch() on a sub-class object (e.g. a+b), the view()
 * function of the sub-class will be called. And the super-class
 * view() function (e.g. binary) will not be called. If you did not
 * override view() of the sub-class, the super-class view() function
 * will be used by default.
 * */
class ir_viewer_t : public ir_visitor_t {
public:
    /**
     * Visit an array of expr.
     */
    void dispatch_expr_arr(const std::vector<expr> &);

    /**
     * Override the view() functions below to visit the
     * IR that you are interested
     * */

#define SC_VIEWER_METHODS_IMPL(node_type, ...) \
    virtual void view(node_type##_c v);

#define SC_VIEWER_METHODS() \
    FOR_EACH_EXPR_IR_TYPE(SC_VIEWER_METHODS_IMPL) \
    FOR_EACH_STMT_IR_TYPE(SC_VIEWER_METHODS_IMPL) \
    FOR_EACH_BASE_EXPR_IR_TYPE(SC_VIEWER_METHODS_IMPL)

    SC_VIEWER_METHODS()

private:
    SC_VISITOR_METHODS(final)
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
