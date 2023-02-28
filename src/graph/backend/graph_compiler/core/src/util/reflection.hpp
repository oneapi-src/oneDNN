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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_REFLECTION_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_REFLECTION_HPP

#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "any_map.hpp"
#include "general_object.hpp"
#include "utils.hpp"
#include <type_traits>
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace json {
class json_writer;
class json_reader;
} // namespace json

namespace reflection {

template <typename T, typename Dummy>
struct type_registry {
public:
    static SC_INTERNAL_API class_metadata *metadata();
    static_assert(std::is_class<T>::value || std::is_union<T>::value,
            "Pointer is not supported in reflection, only classes are "
            "supported");
    static constexpr unsigned depth = 0;
    static constexpr basic_type base = basic_type::t_class;
    static type type_;
};

#define SC_REFLECT_REG_TYPE(ty) \
    template <> \
    struct type_registry<ty, int> { \
        static constexpr unsigned depth = 0; \
        static constexpr basic_type base = basic_type::t_##ty; \
        static constexpr class_metadata *metadata() { return nullptr; } \
        static constexpr type type_ {base, depth, nullptr}; \
    };

SC_REFLECT_REG_TYPE(uint32_t)
SC_REFLECT_REG_TYPE(uint64_t)
SC_REFLECT_REG_TYPE(int32_t)
SC_REFLECT_REG_TYPE(int64_t)
SC_REFLECT_REG_TYPE(float)
SC_REFLECT_REG_TYPE(double)
SC_REFLECT_REG_TYPE(bool)
using string = std::string;
SC_REFLECT_REG_TYPE(string)

// enum types, redirect to int
template <typename T>
struct type_registry<T,
        typename std::enable_if<std::is_enum<T>::value, int>::type>
    : public type_registry<typename std::underlying_type<T>::type, int> {};

// typename trait for vector/array element name
template <typename T, bool is_enum = std::is_enum<T>::value>
struct type_name_trait_t {};

#define SC_ARRAY_LIKE_FIELDS(ElemT) \
    static constexpr unsigned depth = type_registry<ElemT>::depth + 1; \
    static constexpr basic_type base = type_registry<ElemT>::base; \
    static SC_INTERNAL_API vector_metadata *metadata(); \
    static type type_;

template <typename T>
struct type_registry<std::vector<T>> {
    static_assert(!std::is_same<T, bool>::value,
            "Reflection do not support std::vector<bool>");
    SC_ARRAY_LIKE_FIELDS(T)
};

template <typename T, std::size_t N>
struct type_registry<std::array<T, N>> {
    SC_ARRAY_LIKE_FIELDS(T)
};

template <typename T, std::size_t N>
struct type_registry<T[N]> {
    SC_ARRAY_LIKE_FIELDS(T)
};
#undef SC_ARRAY_LIKE_FIELDS

struct field_address_converter_t {
    virtual void *get(void *obj) = 0;
    virtual std::unique_ptr<field_address_converter_t> copy() = 0;
    virtual ~field_address_converter_t() = default;
};

struct offset_field_converter_t : public field_address_converter_t {
    size_t offset_;
    void *get(void *obj) override { return (char *)obj + offset_; }
    std::unique_ptr<field_address_converter_t> copy() override {
        return utils::make_unique<offset_field_converter_t>(offset_);
    };
    offset_field_converter_t(size_t offset) : offset_(offset) {}
};

struct field_base_t {
    std::string name_;
    type type_;
    std::unique_ptr<field_address_converter_t> addresser_;
    virtual std::unique_ptr<field_base_t> copy() = 0;
    virtual void read(void *obj, void *out) = 0;
    virtual void write(void *obj, void *value) = 0;
    virtual void read(void *obj, any_t &out) = 0;
    virtual void write(void *obj, const any_t &value) = 0;
    virtual ~field_base_t() = default;
};

template <typename T>
struct field : public field_base_t {
    field(const std::string &name,
            std::unique_ptr<field_address_converter_t> &&addresser) {
        name_ = name;
        addresser_ = std::move(addresser);
        type_ = {type_registry<T>::base, type_registry<T>::depth,
                type_registry<T>::metadata()};
    }
    std::unique_ptr<field_base_t> copy() override {
        return utils::make_unique<field<T>>(name_, addresser_->copy());
    }
    using copier = typename any_detail::copy_assign_impl_t<
            std::is_copy_assignable<T>::value, T>;
    void read(void *obj, void *out) override {
        void *dst = out;
        void *src = addresser_->get(obj);
        copier::call(dst, src);
    }
    void write(void *obj, void *value) override {
        void *src = value;
        void *dst = addresser_->get(obj);
        copier::call(dst, src);
    }
    void read(void *obj, any_t &out) override {
        T &src = *reinterpret_cast<T *>(addresser_->get(obj));
        out = src;
    }
    void write(void *obj, const any_t &value) override {
        T &dst = *reinterpret_cast<T *>(addresser_->get(obj));
        copier::call(&dst, &value.get<T>());
    }
};

struct class_vtable_t {
    void (*json_serailize_)(void *, json::json_writer *) = nullptr;
    void (*json_deserailize_)(void *, json::json_reader *) = nullptr;
    // three-way compares with another type, this<other => -1, this==other => 0,
    // this>other => 1.
    int (*compare_)(void *, void *) = nullptr;
};

enum class vector_kind {
    not_vector,
    std_vector,
    std_array,
};

/**
 * The class to hold the type info of a reflection-enabled class.
 * Note that the `class_metadata` object can be created from two sources. One is
 * to use `SC_CLASS*` macro to declare the type info the a class. In this case,
 * the `class_metadata` object is a staic object in the binary and is alive
 * during the whole lifetime of the program/shared library. The second case is
 * that the `class_metadata` object can be created dynamically by the tuner's
 * config space. The metadata object's lifetime will be controlled by a
 * shared_ptr and every living `general_object_t` created by the metadata object
 * will have a shared_ptr to it to make sure the metadata is alive.
 * To adapt the first case (static metadata) for the second case, static
 * metadata are also "managed" by a dummy shared_ptr which has empty deleter.
 * The objects' lifetime is still controlled by the program's whole lifetime.
 * */
struct SC_INTERNAL_API class_metadata {
    std::string name_;
    std::vector<std::unique_ptr<field_base_t>> fields_;
    size_t size_;
    std::unordered_map<std::string, field_base_t *> field_map_;
    // the functor calling the constructor i.e. T()
    void (*constructor_)(void *);
    // the additional initalizer (called after calling constructor)
    void (*initalizer_)(void *) = nullptr;
    // the functor calling the destructor i.e. ~T()
    void (*destructor_)(void *);
    // if this class is vector/array. If kind_!=not_vector, the metadata can
    // static cast to vector_metadata
    vector_kind vector_kind_ = vector_kind::not_vector;
    // a weak ptr to convert this to share_ptr
    std::weak_ptr<class_metadata> shared_this_;
    // creates a type-erased object
    general_object_t make_instance();
    std::shared_ptr<class_metadata> shared_from_this();

    std::unique_ptr<class_vtable_t> vtable_;
};

// the extra data & functions for vector<T>
struct vector_metadata : public class_metadata {
    type element_type_;
    unsigned array_depth_;
    size_t (*size)(void *obj);
    void *(*base)(void *obj);
    void (*move_push)(void *vec, void *to_move);
    void (*push_empty)(void *vec);

    void *ptr_of_element(void *obj, size_t idx) const {
        auto b = base(obj);
        return ((char *)b) + idx * element_type_.size();
    }
};

template <typename T>
struct type_name_trait_t<T, false> {
    static std::string name(const vector_metadata &v) {
        return v.element_type_.to_string();
    }
};

// a dummy metadata deleter (does nothing). It is used in static classes to
// convert a static class_metadata to a shared_ptr. The static class metadata
// object is still managed by the lifetime of the whole program, not the
// shared_ptr. For class_metadata dynamically created in tuning spaces, the
// lifetime is managed by shared_ptr
void dummy_class_metadata_deleter(class_metadata *);

// sets the rtti <=> reflection::type mapping. If alternative_name name is not
// null, using it as the name of the type. This parameter is useful when
// registering a class metadata in the builder, because the class metadata is
// not yet ready.
void set_rtti_map_to_type(const std::type_info *rtti_data, const type &ty,
        const std::string *alternative_name = nullptr);

/**
 * The util class to traverse one or two general_ref_t and can be used to
 * recursively visit their sub-fields and array elements
 *
 * The function dispatch() will parse the input ref, and call the respective
 * visit*() function:
 *      - if v1 is a class object, call visit_class(). By default, it will call
 * dispatch() on all fields
 *      - if v1 is an array, call visit_array(). By default, it will call
 * dispatch() on all elements of `v1` and ignore v2
 *      - if v1 is an basic value, call visit() with its type
 *
 * All `v2` in the functions are optional. The visitor can be used to traverse
 * one general_ref_t by setting `v2 = nullptr` in dispatch(). Also, users can
 * visit 2 general_refs at the same time by setting `v2`. This is useful when
 * comparing two general_refs.
 *
 * The return values of visit*() functions means that if we need to continue to
 * call disaptch() on other fields/elements. If a visit*() call returns false,
 * we can skip the remaining values of the object.
 * */
struct visitor_t {
    virtual bool dispatch(general_ref_t *v1, general_ref_t *v2);
    virtual bool visit_class(general_ref_t *v1, general_ref_t *v2);
    virtual bool visit_array(general_ref_t *v, general_ref_t *v2,
            vector_metadata *vec_meta, size_t len, size_t len2, size_t objsize,
            char *base1, char *base2);
    virtual bool visit(std::string *v, std::string *v2) = 0;
    virtual bool visit(int32_t *v, int32_t *v2) = 0;
    virtual bool visit(uint32_t *v, uint32_t *v2) = 0;
    virtual bool visit(int64_t *v, int64_t *v2) = 0;
    virtual bool visit(uint64_t *v, uint64_t *v2) = 0;
    virtual bool visit(float *v, float *v2) = 0;
    virtual bool visit(double *v, double *v2) = 0;
    virtual bool visit(bool *v, bool *v2) = 0;
};

template <typename T>
struct container_impl;

template <typename T>
struct container_impl<std::vector<T>> {
    using the_type = std::vector<T>;
    static void constructor(void *p) { new (p) the_type(); };
    static void destructor(void *p) {
        auto *ths = reinterpret_cast<the_type *>(p);
        ths->~the_type();
    };

    static void move_push(void *p, void *to_move) {
        std::vector<T> &vec = *reinterpret_cast<std::vector<T> *>(p);
        T &data = *reinterpret_cast<T *>(to_move);
        vec.emplace_back(std::move(data));
    };

    static void push_empty(void *p) {
        std::vector<T> &vec = *reinterpret_cast<std::vector<T> *>(p);
        vec.emplace_back(T());
    };

    static size_t size(void *p) {
        return reinterpret_cast<std::vector<T> *>(p)->size();
    };
    static void *base(void *p) {
        return reinterpret_cast<std::vector<T> *>(p)->data();
    };
    static constexpr vector_kind kind_ = vector_kind::std_vector;
    static void name(std::stringstream &ss) { ss << "v["; }
};

template <typename T, std::size_t N>
struct container_impl<std::array<T, N>> {
    using the_type = std::array<T, N>;
    static void constructor(void *p) { new (p) the_type(); };
    static void destructor(void *p) {
        auto *ths = reinterpret_cast<the_type *>(p);
        ths->~the_type();
    };
    static constexpr void (*move_push)(void *vec, void *to_move) = nullptr;
    static constexpr void (*push_empty)(void *vec) = nullptr;

    static size_t size(void *p) { return N; };
    static void *base(void *p) {
        return reinterpret_cast<std::array<T, N> *>(p)->data();
    };
    static constexpr vector_kind kind_ = vector_kind::std_array;
    static void name(std::stringstream &ss) { ss << "a[" << N; }
};

template <typename T, std::size_t N>
struct container_impl<T[N]> : container_impl<std::array<T, N>> {
    using impl_t = container_impl<std::array<T, N>>;
    using impl_t::base;
    using impl_t::constructor;
    using impl_t::destructor;
    using impl_t::kind_;
    using impl_t::move_push;
    using impl_t::push_empty;
    using impl_t::size;
    static void name(std::stringstream &ss) {
        ss << "na[" << N;
    } // native array
};

// create vector_metadata for container<T>
template <typename VecT>
vector_metadata create_vector_meta_impl(
        const std::type_info *rtti, vector_metadata *ptr) {
    vector_metadata v;
    using T = typename std::remove_reference<decltype(
            std::declval<VecT>()[0])>::type;
    using ImplT = container_impl<VecT>;
    v.size_ = sizeof(VecT);
    v.constructor_ = &ImplT::constructor;
    v.destructor_ = &ImplT::destructor;
    v.vector_kind_ = ImplT::kind_;
    v.element_type_ = {type_registry<T>::base, type_registry<T>::depth,
            type_registry<T>::metadata()};
    v.array_depth_ = type_registry<T>::depth + 1;
    v.size = &ImplT::size;
    v.base = &ImplT::base;
    v.move_push = ImplT::move_push;
    v.push_empty = ImplT::push_empty;
    std::stringstream ss;
    ImplT::name(ss);
    ss << type_name_trait_t<T>::name(v);
    ss << ']';
    v.name_ = ss.str();
    set_rtti_map_to_type(rtti,
            reflection::type {type_registry<T>::base, v.array_depth_, ptr},
            &v.name_);
    return v;
}

template <typename VecT>
vector_metadata create_vector_meta(vector_metadata *ptr) {
    static auto shared = std::shared_ptr<vector_metadata>(
            ptr, dummy_class_metadata_deleter);
    auto v = create_vector_meta_impl<VecT>(&typeid(VecT), ptr);
    v.shared_this_ = shared;
    return v;
}

#define SC_VECTOR_TYPE_DEF(...) \
    SC_INTERNAL_API vector_metadata *type_registry<__VA_ARGS__>::metadata() { \
        static vector_metadata meta = create_vector_meta<__VA_ARGS__>(&meta); \
        return &meta; \
    }
#define SC_VECTOR_TYPE_DEF2(...) \
    type type_registry<__VA_ARGS__>::type_ \
            = {type_registry<__VA_ARGS__>::base, \
                    type_registry<__VA_ARGS__>::depth, \
                    type_registry<__VA_ARGS__>::metadata()}

template <typename T>
SC_VECTOR_TYPE_DEF(std::vector<T>);
template <typename T>
SC_VECTOR_TYPE_DEF2(std::vector<T>);

template <typename T, std::size_t N>
SC_VECTOR_TYPE_DEF(std::array<T, N>);
template <typename T, std::size_t N>
SC_VECTOR_TYPE_DEF2(std::array<T, N>);

template <typename T, std::size_t N>
SC_VECTOR_TYPE_DEF(T[N]);
template <typename T, std::size_t N>
SC_VECTOR_TYPE_DEF2(T[N]);

#undef SC_VECTOR_TYPE_DEF
#undef SC_VECTOR_TYPE_DEF2

// explicit instantiation of commonly used types, implemented in reflection.cpp
extern template struct reflection::type_registry<std::vector<int64_t>>;
extern template struct reflection::type_registry<
        std::vector<std::vector<int64_t>>>;
extern template struct reflection::type_registry<std::vector<int>>;
extern template struct reflection::type_registry<std::vector<float>>;

using class_metadata_ptr = std::shared_ptr<class_metadata>;

template <typename T, typename Dummy>
type type_registry<T, Dummy>::type_ = {type_registry<T, Dummy>::base,
        type_registry<T, Dummy>::depth, type_registry<T, Dummy>::metadata()};

// gets the metadata by its registered name. Returns null if name not found
SC_INTERNAL_API class_metadata *get_metadata(const std::string &name);

// gets the reflection type by the type name (generated by type::to_string())
const type *get_type_by_name(const std::string &name);

// sets the metadata by its registered name. Also sets the rtti =>
// reflection::type map
void set_metadata(const std::string &name, class_metadata *meta,
        const std::type_info *rtti_data);

// gets the reflection::type by the rtti. returns nullptr if not found
const type *get_type_by_rtti(const std::type_info *rtti_data);
// gets the rtti by the reflection::type.
SC_INTERNAL_API const std::type_info *get_rtti_by_type(const type *rtti_data);

template <typename T>
struct class_builder_t {
    // the created config space
    class_metadata metadata_;
    /**
     * @param name the name of the config space. Usually should be the name
     * of the user-defined config class
     * */
    class_builder_t(class_metadata *ptr, const char *name) {
        metadata_.name_ = name;
        metadata_.size_ = sizeof(T);
        metadata_.destructor_ = [](void *p) {
            T *ths = reinterpret_cast<T *>(p);
            ths->~T();
        };
        metadata_.constructor_ = [](void *p) { new (p) T; };
        metadata_.vector_kind_ = vector_kind::not_vector;
        set_metadata(metadata_.name_, ptr, &typeid(T));

        // use a static shared_ptr to make sure the shared_ptr itself is alive
        static auto meta_shared_ptr = std::shared_ptr<class_metadata>(
                ptr, dummy_class_metadata_deleter);
        // the lifetime of metadata is not controlled by shared_ptr, see
        // comments in dummy_class_metadata_deleter
        metadata_.shared_this_ = meta_shared_ptr;
    }

    /**
     * Get the created config space
     * */
    class_metadata get() { return std::move(metadata_); }

    /**
     * Registers the initializer function of the class T
     * @param ptr the member function pointer
     * @note the initializer function will be called on the created
     *      user-defined config object after its constructor is called
     * */
    template <void (T::*method)()>
    class_builder_t &init() {
        metadata_.initalizer_ = [](void *p) {
            T *ths = reinterpret_cast<T *>(p);
            (ths->*method)();
        };
        return *this;
    }

    /**
     * Registers and binds a field of the user-defined config class
     * @param name the field name
     * @param ptr the member pointer
     * */
    template <typename FT>
    class_builder_t &field(const char *name, FT T::*ptr) {
        size_t offset = (size_t)(&(((T *)nullptr)->*ptr)); // NOLINT
        auto field_m = utils::make_unique<offset_field_converter_t>(offset);
        metadata_.fields_.emplace_back(
                utils::make_unique<reflection::field<FT>>(
                        name, std::move(field_m)));
        metadata_.field_map_.insert(
                std::make_pair(name, metadata_.fields_.back().get()));
        return *this;
    }
};

} // namespace reflection
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#define SC_CLASS_WITH_NAME(IDENTIFIER, ...) \
    struct __namespace_checker_##IDENTIFIER; \
    static_assert(std::is_same<__namespace_checker_##IDENTIFIER, \
                          ::dnnl::impl::graph::gc:: \
                                  __namespace_checker_##IDENTIFIER>::value, \
            "SC_CLASS macro should be used in sc namespace!"); \
    namespace reflection { \
    template <> \
    SC_INTERNAL_API class_metadata * \
    type_registry<__VA_ARGS__, int>::metadata(); \
    void *__reflection_init_##IDENTIFIER \
            = &type_registry<__VA_ARGS__, int>::type_; \
    template <> \
    class_metadata *type_registry<__VA_ARGS__, int>::metadata() { \
        using TClass = __VA_ARGS__; \
        static class_metadata meta \
                = class_builder_t<TClass>(&meta, #__VA_ARGS__)

/**
 * The macros to define a class for reflection. Example:
 * struct A {
 *  int a;
 *  std::vector<int> b;
 * };
 *
 * Use SC_CLASS in namespace sc:
 * namespace dnnl{
namespace impl{
namespace graph{
namespace gc{
 *  SC_CLASS(A)
 *      SC_FIELD(a)
 *      SC_FIELD(b)
 *  SC_CLASS_END();
 * }
 *
 * Or
 * namespace dnnl{
namespace impl{
namespace graph{
namespace gc{
 *  SC_CLASS_WITH_NAME(AClassName, ::A)
 *      SC_FIELD(a)
 *      SC_FIELD(b)
 *  SC_CLASS_END();
 * }
 *
 * This macro defines the metadata object for the class. The object is a static
 * member of type_registry<T>. This ensures that the metadata object will be
 * initialized on program initialization.
 * */
#define SC_CLASS(NAME) SC_CLASS_WITH_NAME(NAME, NAME)

#define SC_FIELD(F) .field(#F, &TClass::F)
#define SC_INITIALIZER(F) .init<&TClass::F>() // NOLINT

#define SC_CLASS_END() \
    .get(); \
    return &meta; \
    } \
    }

#define SC_REG_ENUM(NAME) \
    namespace reflection { \
    template <> \
    struct type_name_trait_t<NAME, true> { \
        static constexpr const char *name(const vector_metadata &v) { \
            static_assert( \
                    std::is_enum<NAME>::value, #NAME " should be an enum"); \
            return #NAME; \
        } \
    }; \
    }

#endif
