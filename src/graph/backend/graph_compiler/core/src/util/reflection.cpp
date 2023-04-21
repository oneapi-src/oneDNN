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
#include "reflection.hpp"
#include <common/compiler_workarounds.hpp>
#include <compiler/ir/sc_data_type.hpp>
#include <util/compiler_macros.hpp>
#include <util/string_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

// clang-format off
SC_CLASS(sc_data_type_t)
    SC_FIELD(type_code_)
    SC_FIELD(ldata_)
SC_CLASS_END();

SC_CLASS_WITH_NAME(std_any_map, std::unordered_map<std::string, any_t>)
SC_CLASS_END();

SC_CLASS_WITH_NAME(reflection_general_obj, reflection::general_object_t)
SC_CLASS_END();

SC_CLASS_WITH_NAME(reflection_shared_general_obj, reflection::shared_general_object_t) // NOLINT
SC_CLASS_END();

// clang-format on

// explicit explicit instantiation of commonly used types
template struct reflection::type_registry<std::vector<int64_t>>;
template struct reflection::type_registry<std::vector<std::vector<int64_t>>>;
template struct reflection::type_registry<std::vector<int>>;
template struct reflection::type_registry<std::vector<float>>;

namespace reflection {

#define MACRO_ON_BASIC_TYPES_NO_STRING(M) \
    M(int32_t) \
    M(int64_t) \
    M(uint32_t) \
    M(uint64_t) \
    M(float) \
    M(double) \
    M(bool)
#define MACRO_ON_BASIC_TYPES(M) \
    MACRO_ON_BASIC_TYPES_NO_STRING(M) \
    M(string)

void general_object_t::release() {
    if (data_) {
        vtable_->destructor_(data_.get());
        vtable_ = nullptr;
        data_ = nullptr;
    };
}

int type::cmp(const type &other) const {
    if (base_ < other.base_) {
        return -1;
    } else if (base_ > other.base_) {
        return 1;
    }
    // base==other.base_

    if (array_depth_ < other.array_depth_) {
        return -1;
    } else if (array_depth_ > other.array_depth_) {
        return 1;
    }
    // array_depth_==other.array_depth_

    if ((uintptr_t)meta_ < (uintptr_t)other.meta_) {
        return -1;
    } else if ((uintptr_t)meta_ > (uintptr_t)other.meta_) {
        return 1;
    }
    return 0;
}

general_object_t::general_object_t(general_object_t &&other)
    : data_(std::move(other.data_)), vtable_(other.vtable_) {}
general_object_t::general_object_t(
        std::unique_ptr<char[]> &&data, const class_metadata_ptr &vtable)
    : data_(std::move(data)), vtable_(vtable) {}

std::unique_ptr<void, void (*)(void *)> general_object_t::move_to_unique_ptr() {
    auto ptr = data_.release();
    return {ptr, vtable_->destructor_};
}

general_object_t class_metadata::make_instance() {
    auto data = std::unique_ptr<char[]>(new char[size_]);
    constructor_(data.get());
    return general_object_t(std::move(data), shared_from_this());
}

std::shared_ptr<class_metadata> class_metadata::shared_from_this() {
    constexpr int sz = sizeof(shared_this_);
    auto ret = shared_this_.lock();
    assert(ret && ret.get() == this);
    return ret;
}

void general_object_t::copy_from(
        const std::unordered_map<std::string, any_t> &m) {
    copy_from_any_map(m, data_.get(), vtable_.get());
}

void general_object_t::copy_to(std::unordered_map<std::string, any_t> &m) {
    copy_to_any_map(m, data_.get(), vtable_.get());
}

void general_object_t::copy_from_any_map(
        const std::unordered_map<std::string, any_t> &m, void *object,
        class_metadata *vtable) {
    COMPILE_ASSERT(m.size() == vtable->fields_.size(),
            "The size of any map does not match the number of fields");
    for (auto &v : m) {
        auto itr = vtable->field_map_.find(v.first);
        COMPILE_ASSERT(itr != vtable->field_map_.end(),
                "Cannot find field " << v.first << " in class "
                                     << vtable->name_);
        itr->second->write(object, v.second);
    }
}

size_t type::size() const {
    if (array_depth_ > 0 || base_ == basic_type::t_class) {
        return meta_->size_;
    }
    switch (base_) {
        case basic_type::t_int32_t: return sizeof(int32_t);
        case basic_type::t_int64_t: return sizeof(int64_t);
        case basic_type::t_uint32_t: return sizeof(uint32_t);
        case basic_type::t_uint64_t: return sizeof(uint64_t);
        case basic_type::t_float: return sizeof(float);
        case basic_type::t_double: return sizeof(double);
        case basic_type::t_bool: return sizeof(bool);
        case basic_type::t_string: return sizeof(std::string);
        default: assert(0 && "Base basic type"); break;
    }
    // make compiler happy
    return 0;
}

/**
 * Gets the unique string representation of a type.
 * int32 => i
 * int64 => i
 * uint32 => u
 * uint64 => U
 * float => f
 * double => d
 * bool => b
 * string => s
 * class => class name in class_metadata
 * */
std::string type::to_string() const {
    if (meta_) {
        assert(base_ == basic_type::t_class || array_depth_ > 0);
        return meta_->name_;
    }
    assert(array_depth_ == 0 && base_ != basic_type::t_class);
    switch (base_) {
#define PUT_VALUE(PREFIX, TYPE) \
    case basic_type::t_##TYPE: return PREFIX;
        PUT_VALUE("i", int32_t)
        PUT_VALUE("I", int64_t)
        PUT_VALUE("u", uint32_t)
        PUT_VALUE("U", uint64_t)
        PUT_VALUE("f", float)
        PUT_VALUE("d", double)
        PUT_VALUE("b", bool)
        PUT_VALUE("s", string)
        default: break;
    }
#undef PUT_VALUE
    return std::string();
}

void general_object_t::copy_to_any_map(
        std::unordered_map<std::string, any_t> &m, void *object,
        class_metadata *vtable) {
    for (auto &v : vtable->fields_) {
        any_t outv;
        v->read(object, outv);
        m[v->name_] = outv;
    }
}

// this function wraps the class map to ensure it is initialized first
// if we use class map as a global variable, it may not be initialized when
// initializing other globals

static std::unordered_map<const std::type_info *, type> &get_rtti_type_map() {
#define REGISTER_BASIC_TYPE(T) \
    {&typeid(T), type {basic_type::t_##T, 0, nullptr}},
    static std::unordered_map<const std::type_info *, type> class_map {
            MACRO_ON_BASIC_TYPES(REGISTER_BASIC_TYPE)};
    return class_map;
#undef REGISTER_BASIC_TYPE
}

static std::unordered_map<std::string, type> &get_class_map() {
#define REGISTER_BASIC_TYPE2(NAME, T) \
    { \
        NAME, type { basic_type::t_##T, 0, nullptr } \
    }
    static std::unordered_map<std::string, type> class_map {
            REGISTER_BASIC_TYPE2("i", int32_t),
            REGISTER_BASIC_TYPE2("I", int64_t),
            REGISTER_BASIC_TYPE2("u", uint32_t),
            REGISTER_BASIC_TYPE2("U", uint64_t),
            REGISTER_BASIC_TYPE2("f", float), REGISTER_BASIC_TYPE2("d", double),
            REGISTER_BASIC_TYPE2("b", bool), REGISTER_BASIC_TYPE2("s", string)};
    return class_map;
}

struct type_hash_t {
    size_t operator()(const type *v) const noexcept {
        size_t ret = v->array_depth_;
        ret = (ret << 16) ^ (size_t)v->base_;
        ret ^= (size_t)v->meta_;
        return ret;
    }
};

struct type_compare_eq_t {
    bool operator()(const type *v, const type *v2) const noexcept {
        return *v == *v2;
    }
};

using type_rtti_map = std::unordered_map<const type *, const std::type_info *,
        type_hash_t, type_compare_eq_t>;
static type_rtti_map &get_type_rtti_map() {
    static type_rtti_map class_map = []() {
        // the initial values of rtti_type_map is constructed by
        // reversing get_rtti_type_map()
        auto &map = get_rtti_type_map();
        type_rtti_map ret;
        for (auto &kv : map) {
            ret.insert(std::make_pair(&kv.second, kv.first));
        }
        return ret;
    }();
    return class_map;
}

// gets the reflection::type by the rtti. returns nullptr if not found
const type *get_type_by_rtti(const std::type_info *rtti_data) {
    auto itr = get_rtti_type_map().find(rtti_data);
    if (itr != get_rtti_type_map().end()) { return &itr->second; }
    return nullptr;
}

const std::type_info *get_rtti_by_type(const type *ty) {
    auto itr = get_type_rtti_map().find(ty);
    assert(itr != get_type_rtti_map().end());
    return itr->second;
}

void set_rtti_map_to_type(const std::type_info *rtti_data, const type &ty,
        const std::string *alternative_name) {
    assert(get_rtti_type_map().find(rtti_data) == get_rtti_type_map().end());
    auto &map_ty = get_rtti_type_map()[rtti_data];
    map_ty = ty;
    assert(get_type_rtti_map().find(&map_ty) == get_type_rtti_map().end()
            || get_type_rtti_map()[&map_ty] == rtti_data);
    get_type_rtti_map()[&map_ty] = rtti_data;
    const std::string &name
            = alternative_name ? *alternative_name : ty.to_string();
    assert(get_class_map().find(name) == get_class_map().end());
    get_class_map()[name] = ty;
}

class_metadata *get_metadata(const std::string &name) {
    auto ret = get_type_by_name(name);
    if (ret) { return ret->meta_; }
    return nullptr;
}

const type *get_type_by_name(const std::string &name) {
    auto itr = get_class_map().find(name);
    if (itr != get_class_map().end()) { return &itr->second; }
    return nullptr;
}

void set_metadata(const std::string &name, class_metadata *meta,
        const std::type_info *rtti_data) {
    assert(get_class_map().find(name) == get_class_map().end());
    unsigned array_depth;
    basic_type btype;
    if (meta->vector_kind_ == vector_kind::not_vector) {
        array_depth = 0;
        btype = basic_type::t_class;
    } else {
#if SC_GNUC_VERSION_GE(12)
// Disable gcc's warning. We already checked the vector_kind_. meta must be
// vector_metadata
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
        auto vec_meta = static_cast<vector_metadata *>(meta);
        array_depth = vec_meta->array_depth_;
        btype = vec_meta->element_type_.base_;
#if SC_GNUC_VERSION_GE(12)
#pragma GCC diagnostic pop
#endif
    }
    set_rtti_map_to_type(rtti_data, type {btype, array_depth, meta}, &name);
}
void dummy_class_metadata_deleter(class_metadata *) {}

general_ref_t general_ref_t::from(general_object_t &obj) {
    if (obj.vtable_->vector_kind_ != vector_kind::not_vector) {
        auto metadata = static_cast<vector_metadata *>(obj.vtable_.get());
        return {obj.data_.get(),
                {basic_type::t_class, metadata->array_depth_,
                        obj.vtable_.get()}};
    } else {
        return {obj.data_.get(), {basic_type::t_class, 0, obj.vtable_.get()}};
    }
}

general_ref_t general_ref_t::from(const shared_general_object_t &obj) {
    if (obj.vtable_->vector_kind_ != vector_kind::not_vector) {
        auto metadata = static_cast<vector_metadata *>(obj.vtable_.get());
        return {obj.data_.get(),
                {basic_type::t_class, metadata->array_depth_,
                        obj.vtable_.get()}};
    } else {
        return {obj.data_.get(), {basic_type::t_class, 0, obj.vtable_.get()}};
    }
}

general_ref_t general_ref_t::from(shared_general_object_t &obj) {
    return from(static_cast<const shared_general_object_t &>(obj));
}

general_ref_t general_ref_t::from(const general_object_t &obj) {
    if (obj.vtable_->vector_kind_ != vector_kind::not_vector) {
        auto metadata = static_cast<vector_metadata *>(obj.vtable_.get());
        return {obj.data_.get(),
                {basic_type::t_class, metadata->array_depth_,
                        obj.vtable_.get()}};
    } else {
        return {obj.data_.get(), {basic_type::t_class, 0, obj.vtable_.get()}};
    }
}

using stdanymap = std::unordered_map<std::string, any_t>;
static const reflection::type &general_obj_type() {
    return reflection::type_registry<reflection::general_object_t>::type_;
}
static const reflection::type &std_anymap_type() {
    return reflection::type_registry<stdanymap>::type_;
}

bool visitor_t::dispatch(general_ref_t *v, general_ref_t *v2) {
    if (v2 && !(v->type_ == v2->type_)) { return false; }
    if (v->type_ == general_obj_type()) {
        auto obj = reinterpret_cast<general_object_t *>(v->data_);
        general_object_t *obj2 = v2
                ? reinterpret_cast<general_object_t *>(v2->data_)
                : nullptr;
        auto ref1 = general_ref_t::from(*obj);
        if (v2) {
            auto ref2 = general_ref_t::from(*obj2);
            return dispatch(&ref1, &ref2);
        } else {
            return dispatch(&ref1, nullptr);
        }
    } else if (v->type_.array_depth_ >= 1) {
        auto vec_meta
                = static_cast<reflection::vector_metadata *>(v->type_.meta_);
        size_t len = vec_meta->size(v->data_);
        size_t len2 = v2 ? vec_meta->size(v2->data_) : 0;
        size_t objsize = vec_meta->element_type_.size();
        char *ptrelem = (char *)vec_meta->base(v->data_);
        char *ptrelem2 = v2 ? (char *)vec_meta->base(v2->data_) : nullptr;
        return visit_array(
                v, v2, vec_meta, len, len2, objsize, ptrelem, ptrelem2);
    } else if (v->type_.base_ == reflection::basic_type::t_class) {
        return visit_class(v, v2);
    } else {
// clang-format off
#define PUT_VALUE(TYPE) \
    case basic_type::t_##TYPE: return visit(reinterpret_cast<TYPE *>(v->data_),  v2 ? reinterpret_cast<TYPE *>(v2->data_) : nullptr);  break; // NOLINT
        // clang-format on
        using reflection::basic_type;
        using std::string;
        switch (v->type_.base_) {
            MACRO_ON_BASIC_TYPES(PUT_VALUE)
            default:
                assert(0 && "bad basic type");
                return false;
                break;
        }
#undef PUT_VALUE
    }
}

bool visitor_t::visit_class(general_ref_t *v, general_ref_t *v2) {
    auto metadata = v->type_.meta_;
    for (size_t i = 0; i < metadata->fields_.size(); ++i) {
        auto &fld = metadata->fields_[i];
        auto ptr1 = fld->addresser_->get(v->data_);
        auto ptr2 = v2 ? fld->addresser_->get(v2->data_) : nullptr;
        general_ref_t obj1 = {ptr1, fld->type_};
        general_ref_t obj2 = {ptr2, fld->type_};
        if (!dispatch(&obj1, v2 ? &obj2 : nullptr)) { return false; }
    }
    return true;
}
bool visitor_t::visit_array(general_ref_t *v, general_ref_t *v2,
        vector_metadata *vec_meta, size_t len, size_t len2, size_t objsize,
        char *base1, char *base2) {
    assert(!v2);
    for (unsigned i = 0; i < len; i++) {
        reflection::general_ref_t elem {base1, vec_meta->element_type_};
        if (!dispatch(&elem, nullptr)) { return false; }
        base1 += objsize;
        base2 += objsize;
    }
    return true;
}

// clang-format off
#define IMPL_VISIT(TYPE) \
    virtual bool visit(TYPE *v, TYPE *v2) override { return do_visit(v, v2); } // NOLINT
// clang-format on
struct cmp_visitor_t : public visitor_t {
    int result = 0;
    bool dispatch(general_ref_t *v1, general_ref_t *v2) override {
        int cmpresult = v1->type_.cmp(v2->type_);
        if (cmpresult != 0) {
            result = cmpresult;
            return false;
        }
        return visitor_t::dispatch(v1, v2);
    }
    bool visit_array(general_ref_t *v, general_ref_t *v2,
            vector_metadata *vec_meta, size_t len, size_t len2, size_t objsize,
            char *base1, char *base2) override {
        if (len != len2) {
            if (len < len2) {
                result = -1;
            } else {
                result = 1;
            }
            return false;
        }
        for (unsigned i = 0; i < len; i++) {
            reflection::general_ref_t elem {base1, vec_meta->element_type_};
            reflection::general_ref_t elem2 {base2, vec_meta->element_type_};
            if (!dispatch(&elem, &elem2)) { return false; }
            base1 += objsize;
            base2 += objsize;
        }
        return true;
    }
    bool visit(std::string *v, std::string *v2) override {
        int cmpresult = v->compare(*v2);
        if (cmpresult != 0) {
            result = cmpresult;
            return false;
        }
        return true;
    }
    template <typename T>
    bool do_visit(T *v, T *v2) {
        if (*v == *v2) { return true; }
        if (*v > *v2) {
            result = 1;
        } else {
            result = -1;
        }
        return false;
    }
    MACRO_ON_BASIC_TYPES_NO_STRING(IMPL_VISIT)
};

int general_ref_t::cmp(general_ref_t other) const {
    COMPILE_ASSERT(
            type_ == other.type_, "Cannot compare objects of different types");
    if (type_.meta_ && type_.meta_->vtable_ && type_.meta_->vtable_->compare_) {
        return type_.meta_->vtable_->compare_(data_, other.data_);
    }
    cmp_visitor_t v;
    v.dispatch(const_cast<general_ref_t *>(this), &other);
    return v.result;
}

bool general_ref_t::cmp_equal(general_ref_t other) const {
    return cmp(other) == 0;
}

struct hash_visitor_t : public visitor_t {
    size_t result = 0;
    template <typename T>
    bool do_visit(T *v, T *v2) {
        result = result * 23 + std::hash<T>()(*v);
        return true;
    }
    MACRO_ON_BASIC_TYPES(IMPL_VISIT)
};

size_t general_ref_t::hash() const {
    hash_visitor_t hv;
    hv.dispatch(const_cast<general_ref_t *>(this), nullptr);
    return hv.result;
}

} // namespace reflection
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
