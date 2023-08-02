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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SC_EXPR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SC_EXPR_HPP

#include <assert.h>

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sc_data_type.hpp"
#include "statics_table.hpp"
#if SC_MEMORY_LEAK_CHECK > 0
#include <util/leak_detector.hpp>
#endif
#include "ir_node_names.hpp"
#include <util/optional.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

struct any_map_t;
struct ssa_data_t;
struct any_t;

using std::ostream;
/**
 * Utility funtion to print indents for formatting
 * @param os the output stream
 * @param indent the number of indents
 * */
extern void print_indents(ostream &os, int indent);

/**
 * The IDs for each expression node
 * */
enum class sc_expr_type {
    undef = 0,
#define _SC_DEFINE_EXPR(t, ...) t,
    FOR_EACH_EXPR_IR_TYPE(_SC_DEFINE_EXPR)

#undef _SC_DEFINE_EXPR
    // clang-format off
    MAX_TYPE = low_level_intrin
    // clang-format on
};

std::ostream &operator<<(std::ostream &os, sc_expr_type val);

// a wrapper for std::is_base_of
// old versions of g++ have std::is_base_of<Cls, Cls> = false
// which is not expected
template <typename Base, typename Derived>
struct is_base_of_t {
    using RMConstDerived = typename std::remove_const<Derived>::type;
    static constexpr bool value = std::is_base_of<Base, RMConstDerived>::value
            || std::is_same<Base, RMConstDerived>::value;
};

template <typename T, typename Base>
class node_ptr;

namespace optional_impl {
template <typename T, typename Base>
struct optional_base<node_ptr<T, Base>> {
    using cur_type_t = node_ptr<T, Base>;
    void init_as_empty(void *v) { new (v) cur_type_t(); }
    void set_has_value(void *v) {}
    bool has_value_impl(const void *v) const {
        return reinterpret_cast<const cur_type_t *>(v)->defined();
    }
};

} // namespace optional_impl

/**
 * The smart pointer for statement nodes, expression node and etc.
 * Uses std::shared_ptr as internal implementation. It reinforces
 * shared_ptr by adding upcast/downcast functionalities.
 *
 * @tparam T the type of the node that this node_ptr points to
 * @tparam Base the root base class of T. Examples: stmt_base_t, expr_base
 * */
template <typename T, typename Base>
class node_ptr_impl_t {
public:
    template <typename _Arg>
    using _assignable =
            typename std::enable_if<std::is_convertible<_Arg *, T *>::value,
                    Base>::type;
    static_assert(
            is_base_of_t<Base, T>::value, "T should be a subclass of Base");

    // the implementation is based on shared_ptr<Base>
    using impl_ptr = std::shared_ptr<Base>;
    using type = T;

    impl_ptr impl;

    // constructible from a sub-class node_ptr<T2, Base>, where T2 < T
    template <typename T2>
    node_ptr_impl_t(const node_ptr<T2, _assignable<T2>> &other) noexcept
        : impl(other.impl) {}

    // Move-constructible from a sub-class node_ptr<T2, Base>, where T2 < T
    template <typename T2>
    node_ptr_impl_t(node_ptr<T2, _assignable<T2>> &&other) noexcept
        : impl(std::move(other.impl)) {}

    // Constructs a node_ptr_impl_t from raw shared_ptr<Base>
    explicit node_ptr_impl_t(const impl_ptr &impl) noexcept : impl(impl) {}
    // Move-constructs a node_ptr_impl_t from raw shared_ptr<Base>
    explicit node_ptr_impl_t(impl_ptr &&impl) noexcept
        : impl(std::move(impl)) {}

    // Constructs an empty node_ptr_impl_t
    node_ptr_impl_t() noexcept = default;

    template <typename T2>
    node_ptr_impl_t &operator=(
            const node_ptr<T2, _assignable<T2>> &other) noexcept {
        impl = other.impl;
        return *this;
    }

    // move-assignable from node_ptr
    template <typename T2>
    node_ptr_impl_t &operator=(node_ptr<T2, _assignable<T2>> &&other) noexcept {
        impl = std::move(other.impl);
        return *this;
    }

    /**
     * Checks if the contained pointer is `exactly` of type T2::type.
     * `defined()` of this must be `true`. See notes.
     *
     * @tparam T2 the type to check. T2 should be node_ptr<X, Base> or its alias
     * like `for_loop`
     * @return `true` if the contained pointer is of type T2.
     * @note isa() will not check if the pointer is a sub-class of T2::type.
     *  That is, for example:
     *      add a = ...;
     *      a.isa<binary>() == false
     * */
    template <typename T2>
    bool isa() const noexcept {
        static_assert(is_base_of_t<T, typename T2::type>::value,
                "Cannot cast T to T2");
        return impl->node_type_ == T2::type::type_code_;
    }

    /**
     * Checks if the contained pointer is an instance of type T2::type or its
     * subclass. Will be slower than `isa()` because it uses dynamic_cast
     * `defined()` of this must be `true`.
     *
     * @tparam T2 the type to check. T2 should be node_ptr<X, Base> or its alias
     * like `for_loop`
     * @return `true` if the contained pointer is of type T2. Will check if the
     *  pointer is a sub-class of T2.
     * */
    template <typename T2>
    bool instanceof () const noexcept { // NOLINT
        return dynamic_cast<typename T2::type *>(impl.get()) != nullptr;
    }

    /**
     * Converts the pointer to T2. Like static_cast, it will not check if the
     * cast is really valid and it is up to the user of this function to ensure
     * the pointer can be casted to T2.
     *
     * @tparam T2 the target type to cast. It should be node_ptr<X, Base>
     * or its alias like `for_loop`. Also the contained type (T2::type) must be
     * a base class or a subclass of T.
     * @return The casted node_ptr of T2.
     * */
    template <typename T2>
    node_ptr<typename T2::type, Base> static_as() const noexcept {
        static_assert(is_base_of_t<T, typename T2::type>::value
                        || is_base_of_t<typename T2::type, T>::value,
                "Cannot cast T to T2");
        return node_ptr<typename T2::type, Base>(impl);
    }
    /**
     * Converts the pointer to T2. It will return an empty node_ptr if the
     * contained pointer is not `exactly` T2::type. It uses `isa()` internally.
     * `defined()` of this must be `true`.
     * @see isa
     *
     * @tparam T2 the target type to cast. It should be node_ptr<X, Base> or its
     * alias like `for_loop`.
     * @return The casted node_ptr of T2. If the type of the contained
     * pointer is not T2::type, returns empty
     * */
    template <typename T2>
    node_ptr<typename T2::type, Base> as() const noexcept {
        if (isa<T2>()) {
            return static_as<T2>();
        } else {
            return node_ptr<typename T2::type, Base>();
        }
    }

    /**
     * Converts the pointer to T2. It will abort if the contained
     * pointer is not `exactly` T2::type. It uses `isa()` internally.
     * `defined()` of this must be `true`.
     * @see isa
     *
     * @tparam T2 the target type to cast. It should be node_ptr<X, Base> or its
     * alias like `for_loop`.
     * @return The casted node_ptr of T2.
     * */
    template <typename T2>
    node_ptr<typename T2::type, Base> checked_as() const {
        assert(isa<T2>() && "checked_as failed");
        return static_as<T2>();
    }

    /**
     * Converts the pointer to T2. It will return an empty node_ptr if the
     * contained pointer is not a subclass of T2::type. It uses `instanceof()`
     * internally. `defined()` of this must be `true`.
     * @see instanceof
     *
     * @tparam T2 the target type to cast. It should be node_ptr<X, Base> or its
     * alias like `for_loop`.
     * @return The casted node_ptr of T2. If the type of the contained
     * pointer is not a subclass of T2::type, returns empty
     * */
    template <typename T2>
    node_ptr<typename T2::type, Base> dyn_as() const noexcept {
        if (instanceof <T2>()) {
            return static_as<T2>();
        } else {
            return node_ptr<typename T2::type, Base>();
        }
    }

    /**
     * @brief Try to downcast to a subclass pointer. If not successful, return
     * an empty optional
     *
     * @tparam T2 a subclass pointer type
     * @return optional<node_ptr<typename T2::type, Base>>
     */
    template <typename T2>
    optional<node_ptr<typename T2::type, Base>> cast() const {
        if (!defined()) return {};
        return as<T2>();
    }

    /**
     * Adds a const-qualifier to the type T
     * @return The casted node_ptr
     * */
    node_ptr<const T, Base> to_const() const noexcept {
        return node_ptr<const T, Base>(impl);
    }

    /**
     * Removes the const-qualifier of T, like const_cast
     * @return The casted node_ptr
     * */
    node_ptr<typename std::remove_const<T>::type, Base>
    remove_const() const noexcept {
        return node_ptr<typename std::remove_const<T>::type, Base>(impl);
    }

    // operator *
    T &operator*() const noexcept { return *get(); }
    // operator ->
    T *operator->() const noexcept { return get(); }
    // gets the contained pointer
    T *get() const noexcept { return static_cast<T *>(impl.get()); }

    /**
     * Checks if the node_ptr contains any pointer
     * @return false if the node_ptr is empty
     * */
    bool defined() const noexcept { return impl.operator bool(); }

    /**
     * Checks if the node_ptr contains the same pointer of another
     * @param v the other node_ptr to compare with
     * @return true if the node_ptrs are the same
     * */
    bool ptr_same(const node_ptr_impl_t &v) const noexcept {
        return v.impl == impl;
    }

    /**
     * Gets the weak_ptr of the pointer
     * */
    std::weak_ptr<Base> weak() const noexcept { return impl; }
};

template <typename T, typename Base>
class node_ptr : public node_ptr_impl_t<T, Base> {
public:
    using parent = node_ptr_impl_t<T, Base>;
    using impl_ptr = typename parent::impl_ptr;
    using parent::parent;
    using type = typename parent::type;
    using parent::operator=;
    using parent::operator*;
    using parent::operator->;
    node_ptr() = default;
};

// this macro defines the alias of a node_ptr and its const version
// clang-format off
#define SC_DEFINE_EXPR_NODE_PTR(TYPE) \
    using TYPE = node_ptr<TYPE##_node, expr_base>; using TYPE##_c = node_ptr<const TYPE##_node, expr_base>; // NOLINT
// clang-format on

/**
 * The base class of the data-nodes which uses node_ptr as their pointer
 * container. It enables users to get the node_ptr from `this`
 * */
template <typename Base>
class enable_node_ptr_from_this_t : public std::enable_shared_from_this<Base> {
public:
    /**
     * Get the node_ptr from `this`
     * @return the node_ptr, of which the contained pointer is this
     * */
    node_ptr<Base, Base> node_ptr_from_this() {
        return node_ptr<Base, Base>(
                std::enable_shared_from_this<Base>::shared_from_this());
    }

    /**
     * Get the node_ptr from `this`. Const version
     * @return the node_ptr, of which the contained pointer is this
     * */
    node_ptr<const Base, Base> node_ptr_from_this() const {
        return node_ptr<const Base, Base>(std::const_pointer_cast<Base>(
                std::enable_shared_from_this<Base>::shared_from_this()));
    }
};

/**
 * Stores an arbitrary single-lane integer or floating-point value.
 *
 * If the length of valid data is less than 32bit (e.g. f32), The unused bits in
 * the union should be filled with 0. By convention, usage of this union assumes
 * the following mapping between C++-type and sc_data_etype:
 *
 *   datatypes::f16     --> float (WIP)
 *   datatypes::bf16    --> float (WIP)
 *   datatypes::f32     --> float
 *   datatypes::s32     --> int64_t
 *   datatypes::s8      --> int64_t
 *   datatypes::u8      --> uint64_t
 *   datatypes::index   --> uint64_t
 *   datatypes::boolean --> uint64_t
 *   datatypes::pointer --> uint64_t
 */
union union_val {
    uint64_t u64;
    int64_t s64;
    struct {
        float f32;
        int32_t unused;
    };
    union_val() = default;
    union_val(uint64_t val_u64) { u64 = val_u64; }
#if defined(_MSC_VER) || defined(__APPLE__)
    union_val(unsigned long val_u64) { u64 = val_u64; } // NOLINT
#endif
    union_val(int64_t val_s64) { s64 = val_s64; }
    union_val(float val_f32) {
        f32 = val_f32;
        unused = 0;
    }
    bool operator==(const union_val &v) const { return u64 == v.u64; }
    bool operator!=(const union_val &v) const { return u64 != v.u64; }
};

class expr_base;
struct span_t;

// the const version of node_ptr of expr_base
template <>
class node_ptr<const expr_base, expr_base>
    : public node_ptr_impl_t<const expr_base, expr_base> {
public:
    using parent = node_ptr_impl_t<const expr_base, expr_base>;
    using impl_ptr = typename parent::impl_ptr;
    using parent::parent;
    using type = typename parent::type;
    using parent::operator=;
    using parent::operator*;
    using parent::operator->;
    // converter from c++ float to f32 `constant` IR
    SC_INTERNAL_API node_ptr(float v);
    // converter from c++ int32_t to s32 `constant` IR
    SC_INTERNAL_API node_ptr(int32_t v);
    // converter from c++ uint64_t to index `constant` IR
    SC_INTERNAL_API node_ptr(uint64_t v);
    // converter from c++ bool to boolean `constant` IR
    SC_INTERNAL_API node_ptr(bool v);
    node_ptr() = default;
};
using expr_c = node_ptr<const expr_base, expr_base>;

/**
 * The enhanced node_ptr class for expr_base. It enables operator []
 * for easier building indexing IR by simply writing `a[]`. It also
 * enables conversion from C++ numbers to `constant` IR nodes.
 **/
template <>
class node_ptr<expr_base, expr_base>
    : public node_ptr_impl_t<expr_base, expr_base> {
public:
    using expr = node_ptr<expr_base, expr_base>;
    using parent = node_ptr_impl_t<expr_base, expr_base>;
    using impl_ptr = typename parent::impl_ptr;
    using parent::parent;
    using type = typename parent::type;
    using parent::operator=;
    using parent::operator*;
    using parent::operator->;

    // converter from c++ float to f32 `constant` IR
    SC_INTERNAL_API node_ptr(float v);
    // converter from c++ int32_t to s32 `constant` IR
    SC_INTERNAL_API node_ptr(int32_t v);
    // converter from c++ uint64_t to index `constant` IR
    SC_INTERNAL_API node_ptr(uint64_t v);
#if defined(_MSC_VER) || defined(__APPLE__)
    // converter from c++ uint64_t to index `constant` IR
    SC_INTERNAL_API node_ptr(unsigned long v) // NOLINT
        : node_ptr(static_cast<uint64_t>(v)) {} // NOLINT
#endif
    // converter from c++ bool to boolean `constant` IR
    SC_INTERNAL_API node_ptr(bool v);

    node_ptr() = default;
    node_ptr(const parent &v) : parent(v) {}
    /**
     * A helper class to wrap LValues of expr. Assignment on LValues should
     * generate a assign_node_t. lvalue_proxy_t can be auto-converted to expr.
     *
     * @param data_ the wrapped expr LValue
     * @param require_remake_ whether to call remake() every time get() is
     *      called. If true, will generate a new expr when getting the expr
     * */
    struct SC_INTERNAL_API lvalue_proxy_t {
        expr::parent data_;
        bool require_remake_;
        lvalue_proxy_t();
        lvalue_proxy_t(expr data, bool require_remake);
        // generates the indexing_node based on the indexed expr and the indices
        expr get() const;
        operator expr() const;
        operator expr_c() const;
        /**
         * Generates an `assign_node_t` and push to the current scope. The
         * `var_` of the `assign_node_t` will be this indexing_node, the
         * `value_` is `other`
         * @param other the value to "assign" to this indexing_node in
         * generating `assign_node_t`
         * */
        void operator=(const expr &other) const;
        /**
         * Generates an `assign_node_t` and push to the current scope. The
         * `var_` of the `assign_node_t` will be this indexing_node, the
         * `value_` is `other.get()`
         * @param other the value to "assign" to this indexing_node in
         * generating `assign_node_t`
         * */
        void operator=(lvalue_proxy_t &other) const;
        lvalue_proxy_t(lvalue_proxy_t &&);
        lvalue_proxy_t(const lvalue_proxy_t &);

        /**
         * Generates a lvalue_proxy_t with a single index. The lvalue_proxy_t
         * can be further "assigned" with expr or be used as expr
         * */
        lvalue_proxy_t operator[](expr index) const;

        /**
         * Generates a lvalue_proxy_t with multi indices. The lvalue_proxy_t
         * can be further "assigned" with expr or be used as expr
         * */
        lvalue_proxy_t operator[](const std::vector<expr> &index) const;

        /**
         * Generates a lvalue_proxy_t with multi indices as a vector. The
         * lvalue_proxy_t can be further "assigned" with expr or be used as expr
         * */
        lvalue_proxy_t operator[](const span_t &index) const;
        expr_base *operator->() const { return get().get(); }
        /**
         * Converts the pointer to T2. Like static_cast, it will not check if
         * the cast is really valid and it is up to the user of this function to
         * ensure the pointer can be casted to T2.
         * */
        template <typename T2>
        node_ptr<typename T2::type, expr_base> static_as() {
            return get().static_as<T2>();
        }
    };

    /**
     * Generates a lvalue_proxy_t with a single index. The lvalue_proxy_t
     * can be further "assigned" with expr or be used as expr
     * */
    lvalue_proxy_t operator[](expr index);

    /**
     * Generates a lvalue_proxy_t with multi indices. The lvalue_proxy_t
     * can be further "assigned" with expr or be used as expr
     * */
    lvalue_proxy_t operator[](const std::vector<expr> &index) const;

    /**
     * Generates a lvalue_proxy_t with multi indices as a vector. The
     * lvalue_proxy_t can be further "assigned" with expr or be used as expr
     * */
    lvalue_proxy_t operator[](const span_t &index) const;
};
using expr = node_ptr<expr_base, expr_base>;
/**
 * The helper struct to easily build a vector indexing node.
 * Specifies a range (start_index, length)
 * @param index_ the start index in the tensor
 * @param length_ the length of the vector to load
 * @param mask_ the mask of vector to load, should be lanes == length_ or
 * bits == length_, e.g. length_==16, vec_f32x16(trans2d) or uint16_t(most
 * cases).
 * */
struct span_t {
    std::vector<expr> index_;
    uint16_t length_;
    expr mask_;

    span_t(std::vector<expr> index, uint16_t length, expr mask = expr())
        : index_(std::move(index)), length_(length), mask_(std::move(mask)) {}

    span_t(span_t &&other)
        : index_(std::move(other.index_))
        , length_(other.length_)
        , mask_(std::move(other.mask_)) {}
};

class ir_comparer;

// forward decl of ir_visitor_t base class
class ir_visitor_base_t;

/**
 * The stmt and expr nodes implements visitable_base_t, so that they can be
 * visited by ir_visitor_base_t. In visited_by(), it will downcast `this` to a
 * sub-class pointer and call the overloaded `vis->visit_impl(...)` of a
 * specific IR sub-class.
 * */
template <typename Base>
struct visitable_base_t {
    virtual node_ptr<Base, Base> visited_by(ir_visitor_base_t *vis) = 0;
};

// Implementation hidden in visitable.hpp. No need to include it everywhere.
template <typename T, typename Base>
struct visitable_t : public virtual visitable_base_t<Base> {
    node_ptr<Base, Base> visited_by(ir_visitor_base_t *vis) final;
};

struct node_base {
    // optional attributes, nullable. In most cases, use attr() to get the
    // attributes
    std::unique_ptr<any_map_t> attr_;
    // temp data after analysis passes
    std::unique_ptr<any_t> temp_data_;

    // temp data after analysis passes, will create if not exists
    any_t &temp_data() const;

    // temp data after analysis passes, will return empty if not exists
    const any_t &get_temp_data() const;

    // returns attr_ if is defined or creates and sets a new any_map_t if not
    // defined
    any_map_t &attr();

    virtual ~node_base();
};

/**
 * The base class of expression IR nodes
 * */
class expr_base : public node_base,
                  public virtual visitable_base_t<expr_base>,
                  public enable_node_ptr_from_this_t<expr_base>
                  SC_LEAK_CHECK(expr_base) {
public:
    expr_base();
    expr_base(sc_data_type_t type);
    expr_base(sc_expr_type exp_type);
    expr_base(sc_data_type_t type, sc_expr_type exp_type);
    // the data type of the expression
    sc_data_type_t dtype_ = datatypes::undef;
    // the expression type id of the IR node
    sc_expr_type node_type_ = sc_expr_type::undef;
    // the additional info after SSA transformation pass. Before SSA
    // transformation, this field should be null
    std::unique_ptr<ssa_data_t> ssa_data_;

    virtual ~expr_base();
    /**
     * Dumps the IR node as string to the ostream
     * @param os the output stream_t
     * */
    virtual void to_string(ostream &os) const;
    /**
     * Does shallow copying copy on this IR node.
     * Makes a new IR node with the same type and the same values of fields.
     * */
    virtual expr remake() const = 0;
    /**
     * Checks if `this` is same as another IR node. May change the internal
     * states of `ctx`
     * @param other the other IR node to compare
     * @param ctx the context of the comparison: how "same" is defined,
     *  the internal states, etc.
     * @return true if the nodes are the same
     * */
    virtual bool equals(expr_c other, ir_comparer &ctx) const = 0;
    /**
     * Checks if `this` is same as another IR node. It will create a new
     * default ir_comparer context to do comparison.
     * @param other the other IR node to compare
     * @return true if the nodes are the same
     * */
    virtual bool equals(expr_c other) const; // NOLINT
};

/**
 * Makes a expression node_ptr with given arguments.
 * @tparam T the type of the expression to make, should be *_node
 * @param args the arguments to the constructor of T
 * @return a node_ptr of T
 * */
template <typename T, typename... Args>
node_ptr<T, expr_base> make_expr(Args &&...args) {
    std::shared_ptr<T> ptr = std::make_shared<T>(std::forward<Args>(args)...);
    return node_ptr<T, expr_base>(std::move(ptr));
}

// Operator << overrider for expr on std::ostream
extern ostream &operator<<(ostream &os, const expr_c &e);
// Operator << overrider for expr_base* on std::ostream
extern ostream &operator<<(ostream &os, const expr_base *e);

/**
 * The expression node for constants
 *
 * NOTE: To avoid confusion within the Graphcompiler code base, it's recommended
 * that all users of this class adhere to the type-mapping convention indicated
 * in the `union_val` documentation.
 */
class constant_node : public expr_base,
                      public visitable_t<constant_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::constant;
    constant_node(int64_t val, sc_data_type_t dtype = datatypes::s32)
        : expr_base(dtype, sc_expr_type::constant)
        , value_(std::vector<union_val>(1, val)) {};
#if defined(_MSC_VER) || defined(__APPLE__)
    constant_node(unsigned long val, // NOLINT
            sc_data_type_t dtype = datatypes::index) // NOLINT
        : expr_base(dtype, sc_expr_type::constant)
        , value_(std::vector<union_val>(1, val)) {};
#endif

    constant_node(uint64_t val, sc_data_type_t dtype = datatypes::index)
        : expr_base(dtype, sc_expr_type::constant)
        , value_(std::vector<union_val>(1, val)) {};

    constant_node(float val, sc_data_type_t dtype = datatypes::f32)
        : expr_base(dtype, sc_expr_type::constant)
        , value_(std::vector<union_val>(1, val)) {};

    constant_node(union_val val, sc_data_type_t dtype = datatypes::f32)
        : expr_base(dtype, sc_expr_type::constant)
        , value_(std::vector<union_val>(1, val)) {};

    constant_node(const std::vector<union_val> &val, sc_data_type_t dtype)
        : expr_base(dtype, sc_expr_type::constant), value_(val) {};

    expr remake() const override;
    bool equals(expr_c other, ir_comparer &ctx) const override;
    bool is_vector() const { return dtype_.lanes_ > 1; }

    float get_f32() const {
        assert(dtype_ == datatypes::f32);
        return value_[0].f32;
    }

    int64_t get_s32() const {
        assert(dtype_ == datatypes::s32);
        return value_[0].s64;
    }

    uint64_t get_index() const {
        assert(dtype_ == datatypes::index);
        return value_[0].u64;
    }

    bool get_boolean() const {
        assert(dtype_ == datatypes::boolean);
        return value_[0].u64;
    }

    // the contained value
    std::vector<union_val> value_;
};

SC_DEFINE_EXPR_NODE_PTR(constant)

enum class linkage {
    public_global, // global variable, externally visible
    private_global, // global variable, visible in current module
    static_local, // C++ "local static", visible in current func, but have
    // static lifetime
    local, // local variable on stack
};

std::ostream &operator<<(std::ostream &os, linkage val);

/**
 * The variable node.
 * @param type the type of the variable
 * @param var_name the name of the variable
 * */
class var_node : public expr_base, public visitable_t<var_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::var;
    var_node(sc_data_type_t type, const std::string &var_name)
        : expr_base(type, sc_expr_type::var), name_(var_name) {}
    // the variable name
    std::string name_;

    bool equals(expr_c other, ir_comparer &ctx) const override;
    expr remake() const override;
};

SC_DEFINE_EXPR_NODE_PTR(var)

/**
 * The cast node, which converts an expression from one type
 * to another
 * @param type the destination type of the casting
 * @param in_expr the expression to convert
 * */
class cast_node : public expr_base, public visitable_t<cast_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::cast;

    cast_node(sc_data_type_t type, expr in_expr)
        : expr_base(type, sc_expr_type::cast), in_(std::move(in_expr)) {}
    // the expression to convert
    expr in_;

    bool equals(expr_c other, ir_comparer &ctx) const override;
    expr remake() const override;
};

SC_DEFINE_EXPR_NODE_PTR(cast)

/**
 * The base class for binary arithmetic Ops:
 *  + - * / % min max
 * @param expr_type the type
 * @param l the left hand side (LHS) of the binary node
 * @param r the right hand side (LHS) of the binary node
 * */
class binary_node : public expr_base {
public:
    binary_node(sc_expr_type expr_type, const expr &l, const expr &r)
        : expr_base(l->dtype_ == r->dtype_ ? l->dtype_ : datatypes::undef,
                expr_type)
        , l_(l)
        , r_(r) {}
    // the left hand side expr
    expr l_;
    // the right hand side expr
    expr r_;
    bool equals(expr_c other, ir_comparer &ctx) const override;
};
SC_DEFINE_EXPR_NODE_PTR(binary)

/**
 * The node for addition (+)
 * */
class add_node : public binary_node, public visitable_t<add_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::add;
    add_node(const expr &l, const expr &r)
        : binary_node(sc_expr_type::add, l, r) {};

    expr remake() const override;
};
SC_DEFINE_EXPR_NODE_PTR(add)

/**
 * The node for subtraction (-)
 * */
class sub_node : public binary_node, public visitable_t<sub_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::sub;
    sub_node(const expr &l, const expr &r)
        : binary_node(sc_expr_type::sub, l, r) {};

    expr remake() const override;
};
SC_DEFINE_EXPR_NODE_PTR(sub)

/**
 * The node for multiplication (*)
 * */
class mul_node : public binary_node, public visitable_t<mul_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::mul;
    mul_node(const expr &l, const expr &r)
        : binary_node(sc_expr_type::mul, l, r) {};

    expr remake() const override;
};
SC_DEFINE_EXPR_NODE_PTR(mul)

/**
 * The node for division (/)
 * */
class div_node : public binary_node, public visitable_t<div_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::div;
    div_node(const expr &l, const expr &r)
        : binary_node(sc_expr_type::div, l, r) {};

    expr remake() const override;
};
SC_DEFINE_EXPR_NODE_PTR(div)

/**
 * The node for modulo (%)
 * */
class mod_node : public binary_node, public visitable_t<mod_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::mod;
    mod_node(const expr &l, const expr &r)
        : binary_node(sc_expr_type::mod, l, r) {};

    expr remake() const override;
};
SC_DEFINE_EXPR_NODE_PTR(mod)

/**
 * The base class for binary comparison Ops:
 *  == != >= > <= <
 * @param expr_type the type
 * @param l the left hand side (LHS) of the cmp node
 * @param r the right hand side (LHS) of the cmp node
 * */
class cmp_node : public expr_base {
public:
    cmp_node(sc_expr_type expr_type, expr l, expr r)
        : expr_base(sc_data_type_t::boolean(l->dtype_.lanes_), expr_type)
        , l_(std::move(l))
        , r_(std::move(r)) {};
    // the left hand side expr
    expr l_;
    // the right hand side expr
    expr r_;
    bool equals(expr_c other, ir_comparer &ctx) const override;
};
SC_DEFINE_EXPR_NODE_PTR(cmp)

/**
 * Compare equals node (==)
 * */
class cmp_eq_node : public cmp_node,
                    public visitable_t<cmp_eq_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::cmp_eq;
    cmp_eq_node(expr l, expr r)
        : cmp_node(sc_expr_type::cmp_eq, std::move(l), std::move(r)) {};

    expr remake() const override;
};
SC_DEFINE_EXPR_NODE_PTR(cmp_eq)

/**
 * Compare not equals node (!=)
 * */
class cmp_ne_node : public cmp_node,
                    public visitable_t<cmp_ne_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::cmp_ne;
    cmp_ne_node(expr l, expr r)
        : cmp_node(sc_expr_type::cmp_ne, std::move(l), std::move(r)) {};

    expr remake() const override;
};
SC_DEFINE_EXPR_NODE_PTR(cmp_ne)

/**
 * Compare less than node (<)
 * */
class cmp_lt_node : public cmp_node,
                    public visitable_t<cmp_lt_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::cmp_lt;
    cmp_lt_node(expr l, expr r)
        : cmp_node(sc_expr_type::cmp_lt, std::move(l), std::move(r)) {};

    expr remake() const override;
};
SC_DEFINE_EXPR_NODE_PTR(cmp_lt)

/**
 * Compare less equals node (<=)
 * */
class cmp_le_node : public cmp_node,
                    public visitable_t<cmp_le_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::cmp_le;
    cmp_le_node(expr l, expr r)
        : cmp_node(sc_expr_type::cmp_le, std::move(l), std::move(r)) {};

    expr remake() const override;
};
SC_DEFINE_EXPR_NODE_PTR(cmp_le)

/**
 * Compare greater than node (>)
 * */
class cmp_gt_node : public cmp_node,
                    public visitable_t<cmp_gt_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::cmp_gt;
    cmp_gt_node(expr l, expr r)
        : cmp_node(sc_expr_type::cmp_gt, std::move(l), std::move(r)) {};

    expr remake() const override;
};
SC_DEFINE_EXPR_NODE_PTR(cmp_gt)

/**
 * Compare greater equals node (>=)
 * */
class cmp_ge_node : public cmp_node,
                    public visitable_t<cmp_ge_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::cmp_ge;
    cmp_ge_node(expr l, expr r)
        : cmp_node(sc_expr_type::cmp_ge, std::move(l), std::move(r)) {};

    expr remake() const override;
};
SC_DEFINE_EXPR_NODE_PTR(cmp_ge)

/**
 * The base class for binary logic Ops:
 *  && ||
 * @param expr_type the type
 * @param l the left hand side (LHS) of the logic node, should be an boolean
 *  expr
 * @param r the right hand side (LHS) of the logic node, should be an
 * boolean expr
 * */
class logic_node : public expr_base {
public:
    logic_node(sc_expr_type expr_type, expr l, expr r)
        : expr_base(datatypes::boolean, expr_type)
        , l_(std::move(l))
        , r_(std::move(r)) {};
    // the left hand side expr
    expr l_;
    // the right hand side expr
    expr r_;
    bool equals(expr_c other, ir_comparer &ctx) const override;
};
SC_DEFINE_EXPR_NODE_PTR(logic)

/**
 * Logic and node (&&)
 * */
class logic_and_node : public logic_node,
                       public visitable_t<logic_and_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::logic_and;
    logic_and_node(expr l, expr r)
        : logic_node(sc_expr_type::logic_and, std::move(l), std::move(r)) {};

    expr remake() const override;
};
SC_DEFINE_EXPR_NODE_PTR(logic_and)

/**
 * Logic or node (||)
 * */
class logic_or_node : public logic_node,
                      public visitable_t<logic_or_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::logic_or;
    logic_or_node(expr l, expr r)
        : logic_node(sc_expr_type::logic_or, std::move(l), std::move(r)) {};

    expr remake() const override;
};
SC_DEFINE_EXPR_NODE_PTR(logic_or)

/**
 * The logic not node (!)
 * @param in the input expr, should be a boolean expr.
 * */
class logic_not_node : public expr_base,
                       public visitable_t<logic_not_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::logic_not;
    logic_not_node(const expr &in)
        : expr_base(in->dtype_, sc_expr_type::logic_not), in_(in) {};
    expr in_;

    expr remake() const override;
    bool equals(expr_c other, ir_comparer &ctx) const override;
};
SC_DEFINE_EXPR_NODE_PTR(logic_not)

/**
 * Conditional operation: ? :
 * @param expr_type the type
 * @param cond the conditional judgment in select node, should be an boolean
 * expr
 * @param l obtained value when previous condition returns true, should have
 * same type with r
 * @param r obtained value when previous condition returns false, should have
 * same type with l
 * */
class select_node : public expr_base,
                    public visitable_t<select_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::select;
    select_node(expr cond, const expr &l, const expr &r)
        : expr_base(l->dtype_, sc_expr_type::select)
        , cond_(std::move(cond))
        , l_(l)
        , r_(r) {
        assert(l->dtype_ == r->dtype_);
    }
    // the condition expr
    expr cond_;
    // obtained expr when condition is true
    expr l_;
    // obtained expr when condition is false
    expr r_;

    expr remake() const override;
    bool equals(expr_c other, ir_comparer &ctx) const override;
};
SC_DEFINE_EXPR_NODE_PTR(select)

class func_base;
class call_node;

/**
 * The enhanced shared_ptr for function node
 * */
class func_t : public std::shared_ptr<func_base> {
public:
    /**
     * Makes a call node, whose callee is this and the parameters
     * are args
     *
     * @param args the arguments of the call. Should be exprs
     * @return the call node node_ptr
     * */
    template <class... Types>
    expr operator()(Types... args) {
        return expr(std::make_shared<call_node>(
                *this, std::vector<expr>({expr(args)...})));
    }
    /**
     * Constructs an empty pointer
     * */
    func_t() = default;

    /**
     * Creates a smart pointer from a raw pointer. Takes the ownership
     * of the raw pointer.
     *
     * @note You should only create a `func_t` by this constructor by passing
     *  a func_base pointer that is just `newed`. You should never pass a
     * raw pointer from another func to this constructor. To get the `func_t`
     * from `this` in the class func_base, use func_t(shared_from_this())
     * instead.
     * */
    func_t(func_base *ptr);

    /**
     * Creates a smart pointer from a shared_ptr
     * */
    func_t(std::shared_ptr<func_base> &&other);

    /**
     * Checks if the func_t contains the same pointer of another
     * @param v the other func_t to compare with
     * @return true if the pointers of the funcs are the same
     * */
    bool ptr_same(const func_t &v) const { return v.get() == get(); }
};

// constant version of func_t
using func_c = std::shared_ptr<const func_base>;

/**
 * The IR node for function calls
 * @param func the callee
 * @param args the arguments
 * */
class call_node : public expr_base, public visitable_t<call_node, expr_base> {
public:
    /**
     * The parallel_call attr. If a call node has a parallel_attr_t, in the
     * run time, the program will submit jobs to run the callee with index
     * (the first argument) from begin_ to end_, where step = step_. The
     * jobs will run the callee in parallel
     * */
    struct parallel_attr_t {
        expr begin_;
        expr end_;
        expr step_;
        parallel_attr_t(expr begin_, expr end_, expr step_);
    };
    static constexpr sc_expr_type type_code_ = sc_expr_type::call;
    call_node(const std::shared_ptr<node_base> &func,
            const std::vector<expr> &args,
            std::vector<parallel_attr_t> &&para_attr = {});
    call_node(const expr &func, const std::vector<expr> &args);
    call_node(const func_t &func, const std::vector<expr> &args,
            std::vector<parallel_attr_t> &&para_attr = {});
    std::shared_ptr<node_base> func_;
    std::vector<expr> args_;
    std::vector<parallel_attr_t> para_attr_;
    func_t get_prototype() const;

    expr remake() const override;
    bool equals(expr_c other, ir_comparer &ctx) const override;
};
SC_DEFINE_EXPR_NODE_PTR(call)

// The address space of a tensor
enum class address_space {
    // decided by the context. In device mode, it will be "device"
    automatic,
    // Device
    device,
    // CPU
    host,
};

class ir_printer_t;
/**
 * The tensor node. A tensor is a single/multidimemsion array.
 * @param dtype the type of the elements of the tensor
 * @param name the name of the tensor
 * @param dims the dimemsions of the tensor, should be integer exprs
 * @param strides the stride information for each dimension
 *
 * @note at the run time, a function-local tensor's memory buffer will be
 * allocated in three ways:
 * 1. if the tensor's `define_node_t` has an init value, the tensor is a "view"
 * of another tensor (possibly with some offsets). This feature is used when
 * rescheduling multiple tensors into a large buffer if their lifetime do not
 * overlap. This usage is *NOT* allowed in the front end and is for internal use
 * only. The "view" of other tensor should not break the strict alias rule.
 * 2. if the tensor's `define_node_t` has no init value and tensor size is
 * small, the tensor will be allocated on stack
 * 3. if the tensor's `define_node_t` has no init value and tensor size is
 * large, the tensor will be allocated on heap via memory pool
 * @note The dtype_ of tensor node should be a pointer, which can be mapped to
 * `T*` in C++. The type of the elements is stored in the field elem_dtype_
 * @note if the tensor's \p init_value_ has special init values. They are either
 * from \p get_zero_tensor_initializer , (it means that the tensor should be
 * initialzied with 0_ or from \p make_tensor_initializer (it means that the
 * tensor will be filled with a single value repeatedly)
 * */
class tensor_node : public expr_base,
                    public visitable_t<tensor_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::tensor;
    tensor_node(sc_data_type_t dtype, const std::string &name,
            const std::vector<expr> &dims,
            address_space address_space = address_space::automatic,
            const std::shared_ptr<static_data_t> &init_value = nullptr,
            const std::vector<expr> &strides = {});
    // The type of the elements
    sc_data_type_t elem_dtype_;
    std::vector<expr> dims_;
    std::string name_;
    address_space address_space_;
    // the initial raw value of the tensor. Currently this field is valid only
    // when the tensor is defined in a global scope, or is zero-initialized
    std::shared_ptr<static_data_t> init_value_;
    std::vector<expr> strides_;

    expr remake() const override;
    bool equals(expr_c other, ir_comparer &ctx) const override;
    /**
     * Prints the tensor with the name and its dimensions, like:
     * name[dim1, dim2, ...]
     * @param os the output stream
     * @param printer the IR printer
     * */
    void to_string_full(ir_printer_t &printer);
    static const std::shared_ptr<static_data_t> &get_zero_tensor_initializer();
    static std::shared_ptr<static_data_t> make_tensor_initializer(
            union_val val);
};
SC_DEFINE_EXPR_NODE_PTR(tensor)

/**
 * The subscript expression node.
 * If dtype_.lanes_ > 1, it is a vector value.
 * If dtype_.lanes_ == 1, it is a scalar value.
 * @param ptr the tensor node to be accessed. Should be a tensor_node
 * @param idx the indices, should be integer exprs
 * @param mask the mask for loading data. Unused. Nullable
 * */
class indexing_node : public expr_base,
                      public visitable_t<indexing_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::indexing;
    indexing_node(const expr &ptr, const std::vector<expr> &idx, expr mask)
        : expr_base(
                sc_data_type_t(
                        etypes::get_pointer_element(ptr->dtype_.type_code_), 1),
                sc_expr_type::indexing)
        , ptr_(ptr)
        , idx_(idx)
        , mask_(std::move(mask)) {}
    indexing_node(const expr &ptr, const std::vector<expr> &idx, uint16_t lanes,
            expr mask)
        : expr_base(sc_data_type_t(
                            etypes::get_pointer_element(ptr->dtype_.type_code_),
                            lanes),
                sc_expr_type::indexing)
        , ptr_(ptr)
        , idx_(idx)
        , mask_(std::move(mask)) {}
    expr ptr_;
    std::vector<expr> idx_;
    expr mask_;

    expr remake() const override;
    bool equals(expr_c other, ir_comparer &ctx) const override;
};
SC_DEFINE_EXPR_NODE_PTR(indexing)

/**
 * The pointer within a tensor, based on the offsets. The result will be a view
 * of the tensor, having its own shape. It works like `&base[i, j, k]` in C++.
 * But it may also reshape the resulting view. The result can be further used in
 * indexing, tensorptr, etc., as if it is a tensor.
 * @note the index_flatten pass will replace all indexing on tensor_ptr with the
 * indexing on its base tensor. The indices are converted to the index within
 * the base tensor respetively
 * @param base The base tensor and the indices
 * @param shape The new shape for the resulting view. Can be empty if there will
 * be no indexing on this pointer
 * @param is_slice Only useful when there is indexing on this pointer. If true,
 * `(&base[i, j, k])[a, b, c]` will be lowered to base[i+a, j+b, k+c].
 * Otherwise, `(&base[i, j, k])[a, b, c]` will be lowered in 3 steps. First
 * calculate the base pointer of `&base[i, j, k]`. Then, lower the offsets `[a,
 * b, c]` with the `shape` field of this pointer. Finally, add the offset to the
 * base pointer
 * */
class tensorptr_node : public expr_base,
                       public visitable_t<tensorptr_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::tensorptr;
    tensorptr_node(indexing base, const std::vector<expr> &shape, bool is_slice)
        : expr_base(sc_data_type_t::pointerof(base->dtype_.type_code_),
                sc_expr_type::tensorptr)
        , base_(std::move(base))
        , shape_(shape)
        , is_slice_(is_slice) {}
    indexing base_;
    std::vector<expr> shape_;
    bool is_slice_;

    expr remake() const override;
    bool equals(expr_c other, ir_comparer &ctx) const override;
};
SC_DEFINE_EXPR_NODE_PTR(tensorptr)

enum class intrin_type {
    min = 0,
    max,
    abs,
    round,
    floor,
    ceil,
    exp,
    log,
    erf,
    sqrt,
    rsqrt,
    reduce_add,
    reduce_mul,
    reduce_max,
    reduce_min,
    fmadd,
    unpack_low,
    unpack_high,
    shuffle,
    permute,
    int_and,
    int_or,
    int_xor,
    reinterpret,
    broadcast,
    isnan,
    // saturated cast, casting f32xN or s32xN to u8xN or s8xN with saturation.
    // After target specific lowering of CPU, if
    // 1) Target machine has no AVX512f, or the SIMD lanes is not 16: It will be
    // lowered to min/max etc. No special instructions will be generated
    // 2) Target machine has AVX512f and SIMD lanes is 16: backend should only
    // take care of saturated_cast's input dtype being s32x16. It should
    // be simply lowered to mask_pmovs_db_512 (for s8x16) or mask_pmovus_db_512
    // (for u8x16)
    saturated_cast,
    // round FP inputs to integers and cast to integer types
    round_and_cast,
    shl, // shift left
    shr, // shift right
    permutex2var,
    permutexvar,
    insert, // insert the value into dst at the location specified
    extract, // extract the value from simd value
    gather, // gather elements from memory.
    read_struct, // read field from a struct
    write_struct, // write a field to a struct
    // tell the thread pool that the next barrier should run a function when the
    // thread is waiting for others
    set_thread_idle_func,
    // _mm_prefetch(X, _MM_HINT_T{N}}). The locality should be set in the
    // intrin_attrs_["locality"]. It should be an int from 0 to 3, ranging from
    // very local to cache (0, or _MM_HINT_T0) to not local (3)
    prefetch,
    // explicitly load from const memory location
    load_const_mem,
    // gets the group id in nested-parallel-for. It should be in 0 to (max
    // number of groups)-1. Takes one parameter of u32 for the level of group.
    get_group_id,
    // gets the thread id in nested-parallel-for. It should be in 0 to (max
    // number of threads in group)-1. Takes one parameter of s32 for the
    // level of group. A special parameter of (-1) gets the global thread id
    get_group_thread_id,
    // Below are micro-kernels, which should be lower to function call before
    // codegen
    brgemm,
    list_brgemm,
    NUM_INTRINSICS,
};

extern ostream &operator<<(ostream &os, intrin_type t);

namespace intrin_attr {
constexpr const char *out_dtype = "out_dtype";
constexpr const char *brgemm_extras = "intrin.brgemm_extras";
// default true, may turn off when reduce axis has block tile.
constexpr const char *allow_brgemm_fusion = "intrin.allow_brgemm_fusion";
// the attr is the name of struct, string type.
constexpr const char *struct_name = "intrin.struct_name";
// the attr is used in read/write struct field, value is enum int.
constexpr const char *struct_field = "intrin.struct_field";
// datatype of field, sc_data_type_t
constexpr const char *field_dtype = "intrin.field_dtype";
} // namespace intrin_attr

/**
 * The intrinsic-call node
 * @param intrin the intrinsic
 * @param args the arguments
 * */
class intrin_call_node : public expr_base,
                         public visitable_t<intrin_call_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::intrin_call;
    intrin_type type_;
    std::vector<expr> args_;
    std::unique_ptr<any_map_t> intrin_attrs_;
    // the attrs will be copied into intrin_attrs_ before calling
    // intrinsic_handler_t::on_initialize
    intrin_call_node(intrin_type intrin, const std::vector<expr> &args,
            const any_map_t &attrs);

    expr remake() const override;
    bool equals(expr_c other, ir_comparer &ctx) const override;
    // check if size of brgemm args is valid
    // return `true` if intrinsic not brgemm
    bool check_brgemm_arg_size(size_t expected_size) const;
};
SC_DEFINE_EXPR_NODE_PTR(intrin_call)

/**
 * The Phi node. It should be only used when the IR is in SSA form. It merges
 * two SSA values defined in two incoming basic blocks of the current basic
 * block and it selects one of them as the Phi node value based of the actual
 * incoming branch taken
 *
 * @param values the possible incoming values
 * */
class ssa_phi_node : public expr_base,
                     public visitable_t<ssa_phi_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::ssa_phi;
    std::vector<expr> values_;
    // if the phi-node depends on a previous value in the last iteration. In
    // traditional SSA, this means that this PHI depends on a value on critical
    // path
    bool is_loop_phi_;
    ssa_phi_node(const std::vector<expr> &values, bool is_loop_phi);

    expr remake() const override;
    bool equals(expr_c other, ir_comparer &ctx) const override;
};
SC_DEFINE_EXPR_NODE_PTR(ssa_phi)

/**
 * The function address node.
 * @param func the function
 * @note the passes may make new function nodes to replace old ones in the
 * ir_module_t. However the function node referenced by this node will not be
 * changed after passes. You can use the name of the function to find the
 * updated function node in the IR module.
 * */
class func_addr_node : public expr_base,
                       public visitable_t<func_addr_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::func_addr;
    func_t func_;
    func_addr_node(func_t f)
        : expr_base(datatypes::pointer, type_code_), func_(std::move(f)) {};

    expr remake() const override;
    bool equals(expr_c other, ir_comparer &ctx) const override;
};
SC_DEFINE_EXPR_NODE_PTR(func_addr)

// intrin_type for low_level_intrin_kind::x86_general
namespace x86_intrin_type {
enum x86_intrin_type_t {
    avx_broadcast_idx = 0,
    avx_mask_cast,
    avx_compare,
    NUM_INTRINSICS,
};
} // namespace x86_intrin_type

extern ostream &operator<<(ostream &os, x86_intrin_type::x86_intrin_type_t t);

/**
 * The backend specific intrinsic kinds
 **/
enum class low_level_intrin_kind {
    x86_general = 0,
    x86_xbyak,
    NUM_INTRIN_KINDS,
};

/**
 * The low-level-intrinsic node
 * @param kind the backend specific intrinsic kind
 * @param type the intrinsic type, defined by each backend
 * @param args the arguments
 * @note low-level-intrinsics will be used on low level ir for different
 * backends to express target-specific intrinsic, e.g. operations and
 * instructions. This node will be visible for normal ir passes but will only be
 * used by low level passes for CPUs and GPUs.
 **/
class low_level_intrin_node
    : public expr_base,
      public visitable_t<low_level_intrin_node, expr_base> {
public:
    static constexpr sc_expr_type type_code_ = sc_expr_type::low_level_intrin;
    low_level_intrin_kind kind_;
    int64_t type_;
    std::vector<expr> args_;
    std::unique_ptr<any_map_t> intrin_attrs_;
    low_level_intrin_node(low_level_intrin_kind kind, int64_t type,
            const std::vector<expr> &args, const any_map_t &attrs);
    expr remake() const override;
    bool equals(expr_c other, ir_comparer &ctx) const override;
};
SC_DEFINE_EXPR_NODE_PTR(low_level_intrin)

/**
 * Gets the integer from the constant node. Will abort if the dtype
 * of the node is not an integer
 * @param c the constant node
 * @return the constant value
 * */
SC_INTERNAL_API extern int64_t get_const_as_int(const constant_c &c);

/**
 * Gets the integer from the expr node. Will abort if the dtype
 * of the node is not an integer
 * @param e the expr node
 * @return the constant value
 * */
extern int64_t get_expr_as_int(const expr_c &e);
/**
 * When \p e isa instance of an expression node that has a \c name_ member,
 * return that name; otherwise raise an exception.
 */
const std::string &get_node_name(const expr &e);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

namespace std {
template <>
struct hash<dnnl::impl::graph::gc::expr> {
    std::size_t operator()(const dnnl::impl::graph::gc::expr &k) const {
        return hash<dnnl::impl::graph::gc::expr::impl_ptr>()(k.impl);
    }
};

template <>
struct equal_to<dnnl::impl::graph::gc::expr> {
    bool operator()(const dnnl::impl::graph::gc::expr &k,
            const dnnl::impl::graph::gc::expr &k2) const {
        return k.ptr_same(k2);
    }
};

template <>
struct hash<dnnl::impl::graph::gc::expr_c> {
    std::size_t operator()(const dnnl::impl::graph::gc::expr_c &k) const {
        return hash<dnnl::impl::graph::gc::expr::impl_ptr>()(k.impl);
    }
};

template <>
struct equal_to<dnnl::impl::graph::gc::expr_c> {
    bool operator()(const dnnl::impl::graph::gc::expr_c &k,
            const dnnl::impl::graph::gc::expr_c &k2) const {
        return k.ptr_same(k2);
    }
};

} // namespace std

#endif
