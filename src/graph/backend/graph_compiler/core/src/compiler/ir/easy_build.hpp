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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_EASY_BUILD_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_EASY_BUILD_HPP

#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "builder.hpp"
#include <runtime/logging.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

struct context_t;

// the assignment overload for utils::bind_vector_to_args
namespace utils {
template <>
struct SC_INTERNAL_API bind_assigner_t<expr::lvalue_proxy_t, expr> {
    static void assign(expr::lvalue_proxy_t &dst, const expr &src) {
        dst.data_ = src;
        dst.require_remake_ = false;
    }
};
} // namespace utils

namespace builder {
struct SC_INTERNAL_API scope_mgr_t {
    builder::builder_impl_t *ctx_;
    using callback_type = std::function<void(builder::builder_impl_t *, stmt)>;
    callback_type on_pop_;
    scope_mgr_t(builder::builder_impl_t *ctx, callback_type on_pop)
        : ctx_(ctx), on_pop_(std::move(on_pop)) {
        ctx->push_scope();
    }
    scope_mgr_t(scope_mgr_t &&other)
        : ctx_(other.ctx_), on_pop_(std::move(other.on_pop_)) {
        other.ctx_ = nullptr;
    }
    ~scope_mgr_t() {
        if (ctx_) { on_pop_(ctx_, ctx_->pop_scope()); }
    }
};

/**
 * This class builds a for-loop node with RAII of C++.
 * It provides an iterator which can only iterate once.
 * Users should use range-based-for to iterate on this iterator
 * The iterator returns the "var expr" of the for-loop to generate
 * e.g.
 * for (auto i: range(0, 100, "i")) {
 *  buf[i] = buf[i] + 1;
 * }
 *
 * At the end of the scope, this class will push a for-loop node in current
 * build
 * */
struct SC_INTERNAL_API for_range_simulator_t {
    expr var_;
    builder::builder_impl_t *ctx_;
    expr min_;
    expr extent_;
    expr step_;
    for_type type_;
    for_loop *out_;
    int num_threads_;
    struct for_range_iterator_t {
        expr var_;
        bool consumed_;
        expr::lvalue_proxy_t operator*() const {
            return expr::lvalue_proxy_t(var_, false);
        }

        for_range_iterator_t &operator++() {
            consumed_ = true;
            return *this;
        }

        bool operator!=(for_range_iterator_t &other) const {
            return consumed_ != other.consumed_;
        }

        for_range_iterator_t(expr var)
            : var_(std::move(var)), consumed_(false) {}
        for_range_iterator_t() : consumed_(true) {}
    };

    for_range_iterator_t begin() const { return for_range_iterator_t(var_); }

    for_range_iterator_t end() { return for_range_iterator_t(); }

    for_range_simulator_t(builder::builder_impl_t *ctx, for_loop *out,
            const std::string &name, expr min, expr extent, expr step,
            for_type type, int num_threads)
        : var_(builder::make_var(datatypes::index, name))
        , ctx_(ctx)
        , min_(std::move(min))
        , extent_(std::move(extent))
        , step_(std::move(step))
        , type_(type)
        , out_(out)
        , num_threads_(num_threads) {
        ctx->push_scope();
    }

    for_range_simulator_t(const for_range_simulator_t &other) = delete;
    for_range_simulator_t(for_range_simulator_t &&other)
        : var_(std::move(other.var_))
        , ctx_(other.ctx_)
        , min_(std::move(other.min_))
        , extent_(std::move(other.extent_))
        , step_(std::move(other.step_))
        , type_(other.type_)
        , out_(other.out_)
        , num_threads_(other.num_threads_) {}

    ~for_range_simulator_t() {
        if (!var_.defined()) return;
        auto bb = ctx_->pop_scope();
        auto st = ctx_->push_for_loop(
                var_, min_, extent_, step_, bb, true, type_, num_threads_);
        if (out_) { *out_ = st.checked_as<for_loop>(); }
    }
};

SC_INTERNAL_API for_range_simulator_t range(const std::string &name,
        for_loop &out, expr min, expr extent, expr step = expr(1),
        for_type type = for_type::NORMAL, int num_threads = 0);
SC_INTERNAL_API for_range_simulator_t range_nobind(const std::string &name,
        expr min, expr extent, expr step = expr(1),
        for_type type = for_type::NORMAL, int num_threads = 0);
SC_INTERNAL_API for_range_simulator_t range(for_loop &out, expr min,
        expr extent, expr step = expr(1), for_type type = for_type::NORMAL,
        int num_threads = 0);
SC_INTERNAL_API for_range_simulator_t range(expr min, expr extent,
        expr step = expr(1), for_type type = for_type::NORMAL,
        int num_threads = 0);

/**
 * Builds a for-loop. Takes arguments:
 *  the loop var (will create a C++ variable and a var node of the same name)
 *  iter_begin
 *  iter_end
 *  step (optional)
 *  loop type (optional)
 * See range_nobind(). e.g.
 * _for_(i, 0, 100) {...}
 * */
#define _for_(IDX, ...) \
    for (auto IDX : \
            ::dnnl::impl::graph::gc::builder::range_nobind(#IDX, __VA_ARGS__))

/**
 * Builds a for-loop and returns the for-loop node to an output variable.
 * Takes arguments:
 *  output variable: the output for_loop C++ variable. The variable will be
 *      set after the scope of the for loop
 *  the loop var (will create a C++ variable and a var node of the same name)
 *  iter_begin
 *  iter_end
 *  step (optional)
 *  loop type (optional)
 * See range(). e.g.
 * for_loop li;
 * _named_for_(li, i, 0, 100) {...}
 * */
#define _named_for_(OUT, IDX, ...) \
    for (auto IDX : \
            ::dnnl::impl::graph::gc::builder::range(#IDX, OUT, __VA_ARGS__))

struct SC_INTERNAL_API nested_for_ranges_t {
    std::vector<for_range_simulator_t> loops_;
    unsigned cur_var_ = 0;
    expr get_var() {
        assert(cur_var_ < loops_.size());
        return loops_[cur_var_++].var_;
    }
    bool consumed_;
    nested_for_ranges_t(nested_for_ranges_t &other) = delete;
    nested_for_ranges_t(std::vector<for_range_simulator_t> &&loops)
        : loops_(std::move(loops)), consumed_(false) {}
    nested_for_ranges_t(nested_for_ranges_t &&other)
        : loops_(std::move(other.loops_)), consumed_(other.consumed_) {}

    template <typename... Args>
    nested_for_ranges_t(Args... args) : consumed_(false) {
        utils::args_to_vector(loops_, std::move(args)...);
    }

    void step() { consumed_ = true; }
    ~nested_for_ranges_t() {
        // destruct inner loop first
        while (!loops_.empty())
            loops_.pop_back();
    }
    std::vector<expr> get_vars() {
        std::vector<expr> ret;
        for (auto &l : loops_) {
            ret.emplace_back(l.var_);
        }
        return ret;
    }
};

/**
 * Builds nested for loops
 * params: the for_range_simulator_t(s) of the loops. Create
 * for_range_simulator_t by range()/range_nobind() functions
 * */
#define _nested_for_(...) \
    for (auto _0_nested_for \
            = ::dnnl::impl::graph::gc::builder::nested_for_ranges_t( \
                    __VA_ARGS__); \
            !_0_nested_for.consumed_; _0_nested_for.step())

/**
 * Binds a loop-variable in the nested for to a expr variable in C++
 * Calling this macro several times will bind the loop variable from
 * the outer to the inner. Can only be used in a nested-for
 * Also sets the name of the loop-variable.
 * */
#define _iter_var_(v) \
    expr v = _0_nested_for.get_var(); \
    (v).checked_as<var>()->name_ = #v

/**
 *  This class builds a if-then-else node with RAII of C++.
 *  if_simulator_t will generate an iterator with "block_num" inside.
 *  Calling "++" on if_iterator_t will increase "block_num" by one.
 *  Using "*" on if_iterator_t will generate a std::pair<scope_mgr_t,int>, where
 *  the "scope_mgr_t" is the scope guard. When the "then/else" scope is done,
 *  the "scope_mgr_t" will register the generated basic_block_t to
 *  if_simulator_t.{true_block, false_block}. The second element of
 "*if_iterator_t"
 *  is the current "block_num", which can help decide whether we are in
 *  "then block" or "else block". When range-based-for scope is done,
 *  if_simulator_t will be destructed and make and if-else node with registerd
 *  true/false blocks. The "_if_" macro wraps the underlying range-based-for.
 *  It expands like:
    {
    if_simulator_t _simu = if_simulator_t(builder::get_current(), cond);
    if_iterator_t itr = _simu.begin()
    for (;itr != _simu.end(); itr++) {
        std::pair<scope_mgr_t,int> _scope = *itr;
        if (scope.second == 0) {
        // true block
        } else {
        // false block
        }
        // _scope is destoryed here. Will register "true block" or "false block"
        // in _simu
    }
    // _simu is destoryed here. The if-then-else is generated
    }
*/
struct SC_INTERNAL_API if_simulator_t {
    stmt true_block_;
    stmt false_block_;
    builder::builder_impl_t *ctx_;
    expr cond_;

    struct if_iterator_t {
        int block_num_;
        if_simulator_t *if_scope_;
        std::pair<scope_mgr_t, int> operator*() {
            return std::make_pair(scope_mgr_t(if_scope_->ctx_,
                                          [this](builder::builder_impl_t *ctx,
                                                  const stmt &s) {
                                              if (block_num_ == 0) {
                                                  if_scope_->true_block_ = s;
                                              } else {
                                                  if_scope_->false_block_ = s;
                                              }
                                          }),
                    block_num_);
        }

        if_iterator_t &operator++() {
            block_num_++;
            return *this;
        }

        bool operator!=(if_iterator_t &other) const {
            return block_num_ != other.block_num_;
        }

        if_iterator_t(if_simulator_t *if_scope)
            : block_num_(0), if_scope_(if_scope) {}
        if_iterator_t() : block_num_(2), if_scope_(nullptr) {}
    };

    if_iterator_t begin() { return if_iterator_t(this); }

    if_iterator_t end() { return if_iterator_t(); }

    if_simulator_t(builder::builder_impl_t *ctx, expr cond)
        : ctx_(ctx), cond_(std::move(cond)) {}

    if_simulator_t(if_iterator_t &other) = delete;
    ~if_simulator_t() {
        if (!true_block_.defined() || !false_block_.defined()) {
            SC_WARN << "Cannot generate if statements due to undefined "
                       "true_block/false_block for if_simulator, could be "
                       "caused by early destruction from assertion failure";
            return;
        }
        stmt false_block = false_block_.checked_as<stmts>()->seq_.empty()
                ? stmt()
                : false_block_;
        builder::get_current_builder()->push_if_else(
                cond_, true_block_, false_block);
    }
};

/**
 * Builds an if_else node. Takes the parameter of the condition
 * as an expr
 * */
#define _if_(...) \
    for (auto &&__if_scope__ : \
            ::dnnl::impl::graph::gc::builder::if_simulator_t( \
                    ::dnnl::impl::graph::gc::builder::get_current_builder(), \
                    (__VA_ARGS__))) \
        if (__if_scope__.second == 0)
#define _else_ else

struct SC_INTERNAL_API func_simulator_t {
    operator bool() const { return true; }
    std::vector<expr> vargs_;
    func_t *outfunc_;
    std::string name_;
    sc_data_type_t dtype_;
    func_simulator_t(func_simulator_t &&other)
        : vargs_(std::move(other.vargs_))
        , outfunc_(other.outfunc_)
        , name_(std::move(other.name_))
        , dtype_(other.dtype_) {
        other.outfunc_ = nullptr;
    }

    func_simulator_t(const std::string &name, func_t *outfunc,
            sc_data_type_t dtype, std::vector<expr> &&vargs)
        : vargs_(std::move(vargs))
        , outfunc_(outfunc)
        , name_(name)
        , dtype_(dtype) {
        get_current_builder()->push_scope();
    }
    ~func_simulator_t() {
        if (!outfunc_) return;
        auto bb = get_current_builder()->pop_scope();
        *outfunc_ = make_func(name_, vargs_, bb, dtype_);
    }
};

SC_INTERNAL_API func_simulator_t _make_func_simulator(const std::string &name,
        func_t *outfunc, sc_data_type_t dtype,
        std::vector<std::vector<expr>> &&args);

SC_INTERNAL_API std::vector<expr> _make_arg(
        const char *name, sc_data_type_t dtype, const std::vector<int> &args);
SC_INTERNAL_API std::vector<expr> _make_arg(const char *name,
        sc_data_type_t dtype,
        std::initializer_list<unsigned long> args); // NOLINT,
// We must use unsigned long here to let g++ and MSVC to correctly let UL number
// literals find correct overload version of function.
SC_INTERNAL_API std::vector<expr> _make_arg(
        const char *name, sc_data_type_t dtype, const std::vector<expr> &args);

SC_INTERNAL_API std::vector<expr> _make_arg(const char *name,
        sc_data_type_t dtype, std::initializer_list<int> args);

SC_INTERNAL_API std::vector<expr> _make_arg(
        const char *name, sc_data_type_t dtype);

SC_INTERNAL_API func_t _decl_func(const std::string &name, sc_data_type_t dtype,
        std::vector<std::vector<expr>> &&args);

/**
 * Defines a function node named "NAME" and create a C++ variable of func with
 * the same name. arguments:
 *  DTYPE: the return type
 *  NAME: the function name (and the func variable name in C++)
 *  the argument definitions: see _arg_ below
 * e.g. _function_(datatypes::s32, funcA,
 *          _arg_("buffer", datatypes::f32, {100,200})) {
 *  ...
 * }
 * */
#define _function_(DTYPE, NAME, ...) \
    ::dnnl::impl::graph::gc::func_t NAME; \
    if (auto _0_func__ \
            = ::dnnl::impl::graph::gc::builder::_make_func_simulator(#NAME, \
                    &NAME, DTYPE, \
                    std::vector<std::vector<::dnnl::impl::graph::gc::expr>> { \
                            __VA_ARGS__}))
/**
 * Declares a function node named "NAME" and create a C++ variable of func with
 * the same name. This macro will not define a function. It declares an extern
 * function.
 *  arguments:
 *  DTYPE: the return type
 *  NAME: the function name (and the func variable name in C++)
 *  the argument definitions: see _arg_ below
 * */
#define _decl_func_(DTYPE, NAME, ...) \
    ::dnnl::impl::graph::gc::func_t NAME \
            = ::dnnl::impl::graph::gc::builder::_decl_func( \
                    #NAME, DTYPE, {__VA_ARGS__})

/**
 * Declares an argument of function node.
 * arguments:
 *  name: the name of the argument
 *  dtype: the sc_data_type_t of the argument
 *  dimemsions: the integer dimensions of the argument. Wrapped by {}.
 *      Can be empty {} or omitted, meaning this argument is a scalar.
 *  e.g.
 *  _arg_("len", datatypes::s32) // a scalar arg of s32
 *  _arg_("buffer", datatypes::f32, {100,200}) // a tensor arg of f32
 * */
#define _arg_(...) ::dnnl::impl::graph::gc::builder::_make_arg(__VA_ARGS__)

/**
 *  An std::vector of arguments to the function node.
 *  e.g.
 *  std::vector<expr> args = {make_var(...)};
 *  _varg_(args)
 * */
#define _varg_(...) (__VA_ARGS__)

/**
 * Binds all arguments in the function definition to C++ expr variables
 * e.g. _bind_(a,b,c);
 * */
#define _bind_(...) \
    ::dnnl::impl::graph::gc::expr::lvalue_proxy_t __VA_ARGS__; \
    ::dnnl::impl::graph::gc::utils::bind_vector_to_args<0>( \
            _0_func__.vargs_, __VA_ARGS__);

/**
 * Defines a variable within the current scope
 * arguments:
 *  NAME: the name of the variable, should not be quoted
 *  DTYPE: sc_data_type_t of the variable
 * */
#define _var_(NAME, DTYPE) \
    ::dnnl::impl::graph::gc::expr::lvalue_proxy_t NAME( \
            ::dnnl::impl::graph::gc::builder::make_var(DTYPE, #NAME), false); \
    ::dnnl::impl::graph::gc::builder::get_current_builder() \
            ->push_var_tensor_def(NAME, linkage::local);

/**
 * Defines a variable within the current scope with specified name
 * arguments:
 *  NAME: the name of the variable, should not be quoted
 *  VAR_NAME: the name of the tir variable, should be quoted
 *  DTYPE: sc_data_type_t of the variable
 * */
#define _named_var_(NAME, VAR_NAME, DTYPE) \
    ::dnnl::impl::graph::gc::expr::lvalue_proxy_t NAME( \
            ::dnnl::impl::graph::gc::builder::make_var(DTYPE, VAR_NAME), \
            false); \
    ::dnnl::impl::graph::gc::builder::get_current_builder() \
            ->push_var_tensor_def(NAME, linkage::local);

/**
 * Defines a variable within the current scope
 * arguments:
 *  NAME: the name of the variable, should not be quoted
 *  DTYPE: sc_data_type_t of the variable
 *  LINKAGE: the linkage
 *  INIT: the initial value
 * */
#define _var_ex_(NAME, DTYPE, ...) \
    ::dnnl::impl::graph::gc::expr::lvalue_proxy_t NAME( \
            ::dnnl::impl::graph::gc::builder::make_var(DTYPE, #NAME), false); \
    ::dnnl::impl::graph::gc::builder::get_current_builder() \
            ->push_var_tensor_def(NAME, __VA_ARGS__);

/**
 * Defines a variable within the current scope with init value
 * arguments:
 *  NAME: the name of the variable, should not be quoted
 *  DTYPE: sc_data_type_t of the variable
 *  INIT: the initial value
 * */
#define _var_init_(NAME, DTYPE, INIT) \
    _var_ex_(NAME, DTYPE, ::dnnl::impl::graph::gc::linkage::local, INIT)

#define _var_init_copy_(NAME, DTYPE, INIT) \
    _var_init_(NAME, DTYPE, INIT); \
    NAME##_ = NAME;

/**
 * Defines a private linkage global variable within the current scope
 * arguments:
 *  MODULE: the ir_module
 *  NAME: the name of the variable, should not be quoted
 *  DTYPE: sc_data_type_t of the variable
 *  INIT: the initial value
 * */
#define _module_var_(MODULE, NAME, DTYPE, INIT) \
    ::dnnl::impl::graph::gc::expr::lvalue_proxy_t NAME( \
            (MODULE)->make_global_var( \
                    DTYPE, #NAME, linkage::private_global, INIT), \
            false);

/**
 * Defines a public linkage global variable within the current scope
 * arguments:
 *  MODULE: the ir_module
 *  NAME: the name of the variable, should not be quoted
 *  DTYPE: sc_data_type_t of the variable
 *  INIT: the initial value
 * */
#define _global_var_(MODULE, NAME, DTYPE, INIT) \
    ::dnnl::impl::graph::gc::expr::lvalue_proxy_t NAME( \
            (MODULE)->make_global_var( \
                    DTYPE, #NAME, linkage::public_global, INIT), \
            false);

/**
 * Defines a private linkage global tensor within the current scope
 * arguments:
 *  MODULE: the ir_module
 *  NAME: the name of the tensor, should not be quoted
 *  DTYPE: sc_data_type_t of the tensor
 *  INIT: the initial value
 * */
#define _module_tensor_(MODULE, NAME, DTYPE, ...) \
    ::dnnl::impl::graph::gc::expr::lvalue_proxy_t NAME( \
            MODULE->make_global_tensor( \
                    DTYPE, #NAME, {__VA_ARGS__}, linkage::private_global), \
            false);

/**
 * Defines a public linkage global tensor within the current scope
 * arguments:
 *  MODULE: the ir_module
 *  NAME: the name of the tensor, should not be quoted
 *  DTYPE: sc_data_type_t of the tensor
 *  INIT: the initial value
 * */
#define _global_tensor_(MODULE, NAME, DTYPE, ...) \
    ::dnnl::impl::graph::gc::expr::lvalue_proxy_t NAME( \
            MODULE->make_global_tensor( \
                    DTYPE, #NAME, {__VA_ARGS__}, linkage::public_global), \
            false);

/**
 * Defines a tensor within the current scope
 * arguments:
 *  NAME: the name of the tensor, should not be quoted
 *  DTYPE: sc_data_type_t of the tensor
 *  dimemsions: the dimemsions
 * e.g. _tensor_(buffer, datatypes::f32, 100, 200)
 * */
#define _tensor_(NAME, DTYPE, ...) \
    ::dnnl::impl::graph::gc::expr::lvalue_proxy_t NAME( \
            ::dnnl::impl::graph::gc::builder::make_tensor( \
                    #NAME, std::vector<expr> {__VA_ARGS__}, DTYPE), \
            false); \
    ::dnnl::impl::graph::gc::builder::get_current_builder() \
            ->push_var_tensor_def(NAME, linkage::local);

/**
 * Creates and reserve a function call
 * */
#define _evaluate_call_(NAME, ...) \
    ::dnnl::impl::graph::gc::builder::get_current_builder()->push_evaluate( \
            ::dnnl::impl::graph::gc::builder::make_call(NAME, \
                    std::vector<::dnnl::impl::graph::gc::expr> { \
                            __VA_ARGS__}));

/**
 * Creates a returns statement
 * */
#define _return_(...) \
    ::dnnl::impl::graph::gc::builder::get_current_builder()->push_returns( \
            __VA_ARGS__)

} // namespace builder
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
