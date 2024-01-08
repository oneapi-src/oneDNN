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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SC_STMT_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SC_STMT_HPP

#include <assert.h>

#include <functional>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>
#include <unordered_map>

#include "sc_expr.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * The IDs for each statememt node
 * */
enum class sc_stmt_type {
    undef = 0,
#define _SC_DEFINE_STMT(t, ...) t,
    FOR_EACH_STMT_IR_TYPE(_SC_DEFINE_STMT)

#undef _SC_DEFINE_STMT
    // clang-format off
    MAX_TYPE = define
    // clang-format on
};

namespace stmt_attr_key {
// Boolean. If true, for_loop_node_t will be merged as possible
constexpr const char *merge_loop = "merge_loop";

constexpr const char *skip_axis_binding_check = "skip_axis_binding_check";

// for_loop_node_t. It represents the root loop invloving reduce computation
constexpr const char *reduce_root_loop = "reduce_root_loop";

// int. If is set, for_loop_node_t will be unrolled in loop_unroller pass.
// The value is the unroll factor, see unroll() of for_loop_node_t. The value 0
// is a special unroll factor, and it unrolls all iterations of the loop
constexpr const char *unroll_loop = "unroll_loop";

// Boolean. Wether parallel for_loop_node_t is balanced
constexpr const char *parallel_loop_balanced = "parallel_loop_balanced";

// int. Indicate the partition granularity (default =1)
constexpr const char *parallel_merge_loop_granularity
        = "parallel_merge_loop_granularity";

// Boolean. If true, there is no need to insert one pre barrier, please ensure
// data dependency first when enable this flag.
constexpr const char *no_post_barrier = "no_post_barrier";

// Boolean. If true, the certain loop could not be fused.
constexpr const char *no_loop_fuse = "no_loop_fuse";
}; // namespace stmt_attr_key

std::ostream &operator<<(std::ostream &os, sc_stmt_type val);

/**
 * The base class of statement IR nodes
 * */
class stmt_base_t : public node_base,
                    virtual public visitable_base_t<stmt_base_t>,
                    public enable_node_ptr_from_this_t<stmt_base_t>
                    SC_LEAK_CHECK(stmt_base_t) {
public:
    // the statement type id of the IR node
    sc_stmt_type node_type_ = sc_stmt_type::undef;

    stmt_base_t(sc_stmt_type type);
    virtual ~stmt_base_t();
    /**
     * Dump the IR node as string to the ostream
     * @param os the output stream
     * */
    virtual void to_string(ostream &os, int indent) const;

    /**
     * Does shallow copying copy on this IR node.
     * Makes a new IR node with the same type and the same values of fields.
     * */
    virtual node_ptr<stmt_base_t, stmt_base_t> remake() const = 0;

    /**
     * Check if `this` is same as another IR node. May change the internal
     * states of `ctx`
     * @param other the other IR node to compare
     * @param ctx the context of the comparison: how "same" is defined,
     *  the internal states, etc.
     * @return true if the nodes are the same
     * */
    virtual bool equals(node_ptr<const stmt_base_t, stmt_base_t> other,
            ir_comparer &ctx) const = 0;

    /**
     * Check if `this` is same as another IR node. It will create a new
     * default ir_comparer context to do comparison.
     * @param other the other IR node to compare
     * @return true if the nodes are the same
     * */
    virtual bool equals(node_ptr<const stmt_base_t, stmt_base_t> other) const;
};

// the alias of statement node_ptr
using stmt = node_ptr<stmt_base_t, stmt_base_t>;
// the alias of statement constant node_ptr
using stmt_c = node_ptr<const stmt_base_t, stmt_base_t>;

// Operator << overrider for std::ostream on statements
extern ostream &operator<<(ostream &os, const stmt_c &);
extern ostream &operator<<(ostream &os, const stmt_base_t *);

/**
 * Assignment node.
 * @param var_ the destination expr. Can be var/indexing
 * @param value_ the value to be assigned. Should have the same dtype of var_
 * */
class assign_node_t : public stmt_base_t,
                      public visitable_t<assign_node_t, stmt_base_t> {
public:
    static constexpr sc_stmt_type type_code_ = sc_stmt_type::assign;

    expr var_;
    expr value_;
    assign_node_t(expr var, expr value)
        : stmt_base_t(sc_stmt_type::assign)
        , var_(std::move(var))
        , value_(std::move(value)) {};
    stmt remake() const override;
    bool equals(stmt_c other, ir_comparer &ctx) const override;
};
using assign = node_ptr<assign_node_t, stmt_base_t>;
using assign_c = node_ptr<const assign_node_t, stmt_base_t>;

/**
 * The node for a sequence of statements. One or more statements
 * can be contained into a stmts_node_t to be further used in other
 * statements like if_else and for_loop.
 * @param seq_ the sequence of statements
 * */
class stmts_node_t : public stmt_base_t,
                     public visitable_t<stmts_node_t, stmt_base_t> {
public:
    static constexpr sc_stmt_type type_code_ = sc_stmt_type::stmts;
    std::vector<stmt> seq_;

    /**
     * Gets the size of the sequence that is contained
     * @return size
     * */
    size_t size() const { return seq_.size(); };
    stmt remake() const override;
    bool equals(stmt_c other, ir_comparer &ctx) const override;

    /**
     * Gets the index-th statements. Will abort if out of index
     * @return the index-th statement
     * */
    stmt operator[](size_t index) const {
        assert(index < size());
        return seq_[index];
    }

    stmts_node_t(std::vector<stmt> &&seq_)
        : stmt_base_t(sc_stmt_type::stmts), seq_(std::move(seq_)) {}
};

using stmts = node_ptr<stmts_node_t, stmt_base_t>;
using stmts_c = node_ptr<const stmts_node_t, stmt_base_t>;

/**
 * The if-else node. `Else` can be empty. If the `condition_` is
 * true, will go to `then_case_`. Else, if `else_case_` is defined,
 * go to `else_case_` case.
 * @param condition_ the condition of the `if`. Should be of boolean type
 * @param then_case_ the `then` block
 * @param else_case_ the `else` block. Nullable (`stmt()`)
 * */
class if_else_node_t : public stmt_base_t,
                       public visitable_t<if_else_node_t, stmt_base_t> {
public:
    static constexpr sc_stmt_type type_code_ = sc_stmt_type::if_else;
    expr condition_;
    stmt then_case_;
    stmt else_case_;
    if_else_node_t(expr condition, stmt then_case, stmt else_case)
        : stmt_base_t(sc_stmt_type::if_else)
        , condition_(std::move(condition))
        , then_case_(std::move(then_case))
        , else_case_(std::move(else_case)) {};
    stmt remake() const override;
    bool equals(stmt_c other, ir_comparer &ctx) const override;
};
using if_else = node_ptr<if_else_node_t, stmt_base_t>;
using if_else_c = node_ptr<const if_else_node_t, stmt_base_t>;

/**
 * Runs an expression.
 * @param value_ the expression
 * @note Any expression that is not directly or indirectly attached to
 *  an statement will not be visible to codegen. This statement node will
 *  reserve an expression in the statement tree.
 * */
class evaluate_node_t : public stmt_base_t,
                        public visitable_t<evaluate_node_t, stmt_base_t> {
public:
    static constexpr sc_stmt_type type_code_ = sc_stmt_type::evaluate;
    expr value_;
    stmt remake() const override;
    bool equals(stmt_c other, ir_comparer &ctx) const override;
    evaluate_node_t(expr value)
        : stmt_base_t(sc_stmt_type::evaluate), value_(std::move(value)) {}
};
using evaluate = node_ptr<evaluate_node_t, stmt_base_t>;
using evaluate_c = node_ptr<const evaluate_node_t, stmt_base_t>;

/**
 * Returns an expression.
 * @param value_ the expression, nullable if the current function return void_t
 * */
class returns_node_t : public stmt_base_t,
                       public visitable_t<returns_node_t, stmt_base_t> {
public:
    static constexpr sc_stmt_type type_code_ = sc_stmt_type::returns;
    expr value_;
    stmt remake() const override;
    bool equals(stmt_c other, ir_comparer &ctx) const override;
    returns_node_t(expr value)
        : stmt_base_t(sc_stmt_type::returns), value_(std::move(value)) {}
};
using returns = node_ptr<returns_node_t, stmt_base_t>;
using returns_c = node_ptr<const returns_node_t, stmt_base_t>;

/**
 * Variable or tensor definition
 * @param var the var or tensor
 * @param linkage
 * @param init the initial value. When \p var is a \c tensor_node , and init_ is
 * not null, it means that the tensor is a "view" over another tensor or it is
 * zero initialized. @see tensor_node
 * */
class define_node_t : public stmt_base_t,
                      public visitable_t<define_node_t, stmt_base_t> {
public:
    static constexpr sc_stmt_type type_code_ = sc_stmt_type::define;
    expr var_;
    expr init_;
    linkage linkage_;
    stmt remake() const override;
    bool equals(stmt_c other, ir_comparer &ctx) const override;
    define_node_t(expr var, linkage linkage, expr init)
        : stmt_base_t(type_code_)
        , var_(std::move(var))
        , init_(std::move(init))
        , linkage_(linkage) {}
};
using define = node_ptr<define_node_t, stmt_base_t>;
using define_c = node_ptr<const define_node_t, stmt_base_t>;

// the mapping of node pointer: either stmt or expr node
using node_ptr_map = std::unordered_map<std::shared_ptr<node_base>,
        std::shared_ptr<node_base>>;

/**
 * The types of for-loops
 * */
enum class for_type {
    NORMAL = 1, // normal sequential for
    PARALLEL = 2, // run the for-loop in parallel (like omp-parallel-for)
};

std::ostream &operator<<(std::ostream &os, for_type val);

/**
 * The node of for-loop.
 * i.e. for(TYPE var_ = iter_begin_; var_ < iter_end_; var_ += step_){
 *  ...
 * }
 *
 * @param var_ the iterate variable. The loop-var is expected to
 *  be used only within the scope of the loop. Should be an integer var
 * @param iter_begin_ the initial value of var_
 * @param iter_end_ the max bound of the loop-var var_. Can never be reached
 * @param step_ the step of var_ in each iteration.
 * @param body_ the body of the loop
 * @param incremental_ if the loop-var var_ is incremental. Not currently used.
 * @param kind_ the kind of the loop. @see for_type
 * @param num_threads_ the number of threads to use when it is a parallel-for. 0
 * for using all avaiable threads in thread group. If the for-loop is not
 * parallelled, it should be 0
 * */
class for_loop_node_t : public stmt_base_t,
                        public visitable_t<for_loop_node_t, stmt_base_t> {
    using ptr_type = node_ptr<for_loop_node_t, stmt_base_t>;

public:
    static constexpr sc_stmt_type type_code_ = sc_stmt_type::for_loop;
    expr var_;
    expr iter_begin_;
    expr iter_end_;
    expr step_;
    stmt body_;
    bool incremental_;
    for_type kind_;
    int num_threads_;

    /**
     * A for-loop node will become
     * invalid after it is merged or fused by merge() and fuse()
     * @return true if the loop is valid.
     * */
    bool isvalid() const;

    /**
     * Do in-place split on this for-loop. The current loop will
     * become the outer loop and the inner loop will be returned.
     * The loop start/end/step must be constants and step should
     * be positive
     * e.g.:
     * for(i, 0, 100) {
     *  ...
     * }
     *
     * after loop_i.split(20);
     * for(i_outer, 0, 5) {
     *  for(i_inner, 0, 20) {
     *      ...
     *  }
     * }
     * The loop_i points to the outer loop
     * @param block the length of the inner loop
     * @return the for_loop node_ptr of the inner loop
     * */
    ptr_type split(int64_t block, node_ptr_map *node_remap = nullptr);

    /**
     * Change the original loop's end and num_threads
     * for i in (0, end, 1) parallel (num_threads)
     *   op1(i)
     * where num_iters == end == num_threads
     * ===>
     * for i in (0, num_threads / num_groups, 1)
     *     parallel (num_threads / num_groups)
     *   for j in (0, num_groups, 1) parallel (num_groups)
     *     iter = i * num_groups + j
     *       op1(iter)
     * Example:
     * for i in (0, 16, 1) parallel (16) {
     *   op(i)
     * }
     * with num_groups = 2:
     * for i in (0, 8, 1) parallel (8) {
     *   for j in (0, 2, 1) parallel (2) {
     *     op(i * 2 + j)
     *   }
     * }
     *
     * @param num_groups the num_threads_ of inner loop
     * @return the for_loop node_ptr of the inner loop
     * */
    ptr_type split_on_num_threads(
            int64_t num_groups, node_ptr_map *node_remap = nullptr);

    /**
     * Do in-place fusion of two nested loop. This is a reverse operaion of
     * split(). The for-loop to fuse should be the next nested loop of this.
     * @note The inner loop will be fused into `this` and be invalidated.
     * @note `this` should have only one child-node that is `ax`. If `this` have
     *  multiple children statements, this function will abort.
     * for(i, 0, 5) {
     *  for(j, 0, 20) {
     *      ...
     *  }
     * }
     *
     * After loop_i.fuse(loop_j);
     * for(fused_i_j, 0, 100) {
     *  i = fused_i_j / 20;
     *  j = fused_i_j % 20;
     *  ...
     * }
     *
     * @param ax the for-loop node_ptr to fuse. Should be the next nested-loop
     * @return The fused for-loop. Should have the same ptr of `this`
     * */
    ptr_type fuse(const ptr_type &ax, node_ptr_map *node_remap = nullptr);

    /**
     * Do in-place reordering on some nested loops. Except the most inner loop,
     * each of the specified for-loops should have only one child statement
     * which is the next inner loop. `this` should be the most outer loop in the
     * specified loops
     *
     * for(i, 0, 5) {
     *  for(j, 0, 20) {
     *      ...
     *  }
     * }
     *
     * After loop_i.reorder({loop_j, loop_i});
     * for(j, 0, 20) {
     *  for(i, 0, 5) {
     *      ...
     *  }
     * }
     *
     * @param parent the parent statement of `this`
     * @param ax the new order of the nested for-loops
     * */
    void reorder(stmt parent, std::vector<ptr_type> &&ax);

    /**
     * Inplace merge a sibling for loop "other" with this. e.g.:
     * for i in (0, 100, 1) {
     *  A[i] = A[i] + 1
     * }
     * for j in (0, 100, 1) {
     *  B[i] = B[i] + 1
     * }
     *
     * After merging i.merge(parent, j):
     * for i in (0, 100, 1) {
     *  A[i] = A[i] + 1
     *  B[i] = B[i] + 1
     * }
     * "other" must be a for loop in the same stmts of this. This function will
     * merge these two loops and put the body of "other" after the body of
     * "this". The ranges of merged for-loops must be the same, but can be
     * non-constants.
     *
     * @param parent the parent stmts_node_t of this and `other`. Nullable. If
     * not null, will erase `other` from parent
     * @param other the other loop to be merged
     * @return the merged for-loop. Should be the same ptr of this
     * */
    ptr_type merge(const stmt &parent, const ptr_type &other);

    /**
     * Merge "num_nested" inner loops of this with
     * a sibling for loop "other". e.g.:
     * for i in (0, 100, 1) {
     *  for j in (0, 100, 1) {
     *      for k in (0, 100, 1) {
     *          A[i,j,k] = A[i,j,k] + 1
     *      }
     *  }
     * }
     * for a in (0, 100, 1) {
     *  for b in (0, 100, 1) {
     *      for c in (0, 100, 1) {
     *          B[a,b,c] = B[a,b,c] + 1
     *      }
     *  }
     * }
     *
     * After merging i.merge(parent, a, 2), it will merge 2 inner
     * loops from i and a:
     * for i in (0, 100, 1) {
     *  for j in (0, 100, 1) {
     *      for k in (0, 100, 1) {
     *          A[i,j,k] = A[i,j,k] + 1
     *      }
     *      for c in (0, 100, 1) {
     *          B[a,b,c] = B[a,b,c] + 1
     *      }
     *  }
     * }
     *
     * @see merge above for the requirements of this function
     * @param parent the parent stmts_node_t of this and `other`.  Nullable. If
     *  not null, will erase `other` from parent
     * @param other the other loop to be merged
     * @param num_nested the number of nested loops to be merged.
     * @return the merged for-loop. Should be the same ptr of this
     * */
    ptr_type merge(stmt parent, ptr_type other, unsigned num_nested);

    /**
     * Merges all nested loops within this and `other`
     * @see merge
     * @return the number of inner loops that were merged
     * */
    int merge_all(stmt parent, ptr_type other);

    /**
     * Unrolls the loop
     * Original loop
     * for(i=A; i<B; i+=c) {
     *
     * }
     *
     * ============>
     * //remainder version
     * for(i_u=0; i_u < (B-A)/(c*factor)+1; i_u+=1) {
     *     if (i_u < (B-A)/(c*factor)) {
     *         int i = i_u * c * factor + A;
     *         unroll(i + c * 1);
     *         unroll(i + c * 2);
     *         ...
     *         unroll(i + c * factor);
     *     } else {
     *         for(i = (B-A)/(c*factor)*(c*factor)+A; i<B; i+=c) {
     *             original body of i
     *         }
     *     }
     * }
     *
     * // or, no remainder:
     * for(i_u=0; i_u < (B-A)/(c*factor); i_u+=1) {
     *     int i = i_u * c * factor + A;
     *     unroll(i + c * 1);
     *     unroll(i + c * 2);
     *     ...
     *     unroll(i + c * factor);
     * }
     * @param factor the unroll factor: how many times the body should be. If is
     * 0, will try to unroll all
     * @param parent the stmts node contains `this` axis
     * */
    void unroll(uint64_t factor = 0, const stmt &parent = stmt());

    /**
     * Merges two sibling loops. The execution order is unchanged: will still do
     * the work in loop 1 then loop 2. This transform is useful when loop1 and
     * loop2 have no dependency and are all parallel-fors. It will eliminate the
     * barrier after loop1. Requires that the steps of the merged loops to be 1
     *
     * Original:
     * for(i, A, B)
     * {
     *  body1
     * }
     * for(j, C, D) {
     *  body1
     * }
     * After =================
     * for(i, A, B + D - C) {
     *  if (i < B) {
     *      body1
     *  } else {
     *      j = i - B + C
     *      body2
     *  }
     * }
     * @param parent the stmts node that contains `this` and `ax`
     * @param ax the sibling loop to be merged after this loop
     * */
    void parallel_merge(const stmt &parent, const ptr_type &ax);

    /**
     * Gets the split factor of this loop for balance211 workload-dispatch.
     * @see builtin::generate_balance211
     * This function is only avaliable when the loop is a nested parallel-for
     * (with num_threads!=0). It returns the maximal number of groups that this
     * parallel-for can be split into. Each sub-thread group will be assign with
     * a workload of the same size after spliting. It may return 0, indicating
     * that this for-loop has non-const boundary.
     */
    uint64_t get_balance211_split_factor() const;

    stmt remake() const override;
    bool equals(stmt_c other, ir_comparer &ctx) const override;

    for_loop_node_t(expr var, expr iter_begin, expr iter_end, expr step,
            stmt body, bool incremental, for_type kind, int num_threads = 0)
        : stmt_base_t(sc_stmt_type::for_loop)
        , var_(std::move(var))
        , iter_begin_(std::move(iter_begin))
        , iter_end_(std::move(iter_end))
        , step_(std::move(step))
        , body_(std::move(body))
        , incremental_(incremental)
        , kind_(kind)
        , num_threads_(num_threads) {}
};
using for_loop = node_ptr<for_loop_node_t, stmt_base_t>;
using for_loop_c = node_ptr<const for_loop_node_t, stmt_base_t>;

/**
 * Makes a statement node_ptr with given arguments.
 * @tparam T the type of the statement to make, should be *_node
 * @param args the arguments to the constructor of T
 * @return a node_ptr of T
 * */
template <typename T, typename... Args>
node_ptr<T, stmt_base_t> make_stmt(Args &&...args) {
    std::shared_ptr<T> ptr = std::make_shared<T>(std::forward<Args>(args)...);
    return node_ptr<T, stmt_base_t>(std::move(ptr));
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

namespace std {
template <>
struct hash<dnnl::impl::graph::gc::stmt> {
    std::size_t operator()(const dnnl::impl::graph::gc::stmt &k) const {
        return hash<dnnl::impl::graph::gc::stmt::impl_ptr>()(k.impl);
    }
};

template <>
struct equal_to<dnnl::impl::graph::gc::stmt> {
    bool operator()(const dnnl::impl::graph::gc::stmt &k,
            const dnnl::impl::graph::gc::stmt &k2) const {
        return k.ptr_same(k2);
    }
};

template <>
struct hash<dnnl::impl::graph::gc::stmt_c> {
    std::size_t operator()(const dnnl::impl::graph::gc::stmt_c &k) const {
        return hash<dnnl::impl::graph::gc::stmt_c::impl_ptr>()(k.impl);
    }
};

template <>
struct equal_to<dnnl::impl::graph::gc::stmt_c> {
    bool operator()(const dnnl::impl::graph::gc::stmt_c &k,
            const dnnl::impl::graph::gc::stmt_c &k2) const {
        return k.ptr_same(k2);
    }
};

} // namespace std

#endif
