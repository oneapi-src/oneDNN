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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_VISITOR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_VISITOR_HPP

#include <vector>
#include "sc_function.hpp"
#include <unordered_map>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

// this macro declares visit_impl() on all IR node classes
// The POSTFIX can be "=0", "final", etc.
// use NOLINT since POSTFIX should not be enclosed in parentheses
#define SC_BASE_VISITOR_METHODS_IMPL(node_type, ret_type, POSTFIX) \
    virtual ret_type visit_impl(node_type v) POSTFIX; /* NOLINT*/

#define SC_BASE_VISITOR_METHODS(POSTFIX) \
    FOR_EACH_EXPR_IR_TYPE(SC_BASE_VISITOR_METHODS_IMPL, expr, POSTFIX) \
    FOR_EACH_STMT_IR_TYPE(SC_BASE_VISITOR_METHODS_IMPL, stmt, POSTFIX) \
    FOR_EACH_BASE_EXPR_IR_TYPE(SC_BASE_VISITOR_METHODS_IMPL, expr, POSTFIX)

// The base interface class for all visitors
class ir_visitor_base_t {
public:
    /**
     * Downcasts the pointer and call visit() with the subclass pointer
     * See the dispatch rule above
     * @param e the pointer to dispatch
     * @return the IR node replaces the input node
     */
    virtual expr dispatch_impl(expr e);

    /**
     * Downcasts the pointer and call visit() with the subclass pointer
     * See the dispatch rule above
     * @param e the pointer to dispatch
     * @return the IR node replaces the input node
     */
    virtual stmt dispatch_impl(stmt s);

    // the visitor_impl() methods to be implemented by subclasses
    SC_BASE_VISITOR_METHODS(= 0)

    virtual ~ir_visitor_base_t() = default;
};

/**
 * This class defines the default visit_impl() on all IR nodes. Can be shared by
 * in-places & copy-on-write visitors
 * */
template <bool inplace_>
class ir_visitor_base_impl_t : ir_visitor_base_t {
public:
    using ir_visitor_base_t::dispatch_impl;
    /**
     * Visits a function IR node. The default implementation just calls
     * dispatch() on each of the fields
     * @param e the pointer to dispatch
     * @return the IR node replaces the input node
     */
    virtual func_t dispatch_impl(func_t v);

    /**
     * Visits an array of expr.
     * Returns true if any expr in the array is changed. Otherwise, returns
     * false.
     * @param arr the array of expr to dispatch
     * @return true if any of the expr is changed
     */
    bool dispatch_expr_vector(
            std::vector<expr> &arr, std::vector<expr> &newval);
    SC_BASE_VISITOR_METHODS()
    bool changed_ = false;
};

#define SC_VISITOR_METHODS_IMPL(node_type, ret_type, POSTFIX) \
    virtual ret_type visit(node_type##_c v) POSTFIX; /* NOLINT*/

#define SC_VISITOR_METHODS(POSTFIX) \
    FOR_EACH_EXPR_IR_TYPE(SC_VISITOR_METHODS_IMPL, expr_c, POSTFIX) \
    FOR_EACH_STMT_IR_TYPE(SC_VISITOR_METHODS_IMPL, stmt_c, POSTFIX) \
    FOR_EACH_BASE_EXPR_IR_TYPE(SC_VISITOR_METHODS_IMPL, expr_c, POSTFIX)

/**
 * The base class of copy-on-write IR passes.
 * Override the overloaded visit() function that you are
 * interested. In visit(), call dispatch() on each of the sub-node
 * of the IR node, if you need to "visit" the sub-nodes. visit()
 * function should return the expr/stmt that should replace the input
 * one
 *
 * @note Dispatch rule for super-classes and sub-classes: If you
 * override the visit() function of a sub-class (e.g. add), and you
 * call dispatch() on a sub-class object (e.g. a+b), the visit()
 * function of the sub-class will be called. And the super-class
 * visit() function (e.g. binary) will not be called. If you did not
 * override visit() of the sub-class, the super-class visit() function
 * will be used by default.
 *
 * @note The visitor does not ensure that the IR DAG after the mutation is
 * "consistent" - that is, if a var/tensor is re-made (with a new address), all
 * other IR nodes that use the var/tensor still point to the old var/tensor
 * node. To automatically keep the IR DAG consistent, use
 * ir_consistent_visitor_t.
 * @see ir_consistent_visitor_t
 * */
class ir_visitor_t : private ir_visitor_base_impl_t<false> {
    // overrides ir_visitor_base_impl_t<false>::dispatch_impl to use dispatch()
    expr dispatch_impl(expr e) final;
    stmt dispatch_impl(stmt s) final;
    func_t dispatch_impl(func_t v) final;

public:
    /**
     * Downcasts the pointer and call visit() with the subclass pointer
     * See the dispatch rule above
     * @param e the pointer to dispatch
     * @return the IR node replaces the input node
     */
    virtual expr_c dispatch(expr_c e);

    /**
     * Downcasts the pointer and call visit() with the subclass pointer
     * See the dispatch rule above
     * @param e the pointer to dispatch
     * @return the IR node replaces the input node
     */
    virtual stmt_c dispatch(stmt_c s);

    /**
     * Visits an array of expr.
     * Returns true if any expr in the array is changed. Otherwise, returns
     * false.
     * @param arr the array of expr to dispatch
     * @return true if any of the expr is changed
     */
    bool dispatch_expr_vector(
            const std::vector<expr> &arr, std::vector<expr> &newval);
    bool dispatch_expr_vector(
            const std::vector<expr> &arr, std::vector<expr_c> &newval);

    /**
     * Visits a function IR node. The default implementation just calls
     * dispatch() on each of the fields
     * @param e the pointer to dispatch
     * @return the IR node replaces the input node
     */
    virtual func_c dispatch(func_c v);

    SC_VISITOR_METHODS()

private:
    // we want to hide dispatch_impl and prevent it from being overriden
    SC_BASE_VISITOR_METHODS(final)

protected:
    using ir_visitor_base_impl_t<false>::dispatch_impl;
    // gets a unique run id. Multiple calls of get_run_id() will return
    // different IDs. It is used to mark the age of temp_data on TIR.
    // Passes/Visitors can call this function once in the constructor, and use
    // the run_id to compare with the run_id in the temp_data to avoid reading
    // stale data
    static uint64_t get_run_id();
};

/**
 * Do in-place changes on the IR DAG. Will not
 * call remake if the node is changed. Instead, it will set "changed_"
 * field if any of the sub-node is changed
 * When implementing your own ir_inplace_visitor_t by extending this class,
 * make sure you set changed_ to true when you change any node of the IR.
 * @see ir_visitor_t
 * */
class ir_inplace_visitor_t : public ir_visitor_base_impl_t<true> {};

/**
 * Mutating the IR DAG and keeping the consistency of the DAG. If a tensor/var
 * is changed, it will remember the old->new mapping. Then before a tensor/var
 * is visited, this visitor will check in the mapping to find if the tensor/var
 * is replaced. If so, it will pass the replaced new tensor/var to visit(...)
 *
 * Performance Warning: Overhead in visiting tensors and vars
 * Memory leak Warning: Old replaced vars and tensors are alive in replace_map_
 * @see ir_visitor_t
 * */
class ir_consistent_visitor_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::dispatch_expr_vector;
    // the old -> new mapping for var/tensor
    std::unordered_map<expr_c, expr_c> replace_map_;
    expr_c dispatch(expr_c e) override;
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
