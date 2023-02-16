/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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

#include <util/any_map.hpp>
#include <util/any_reflection_cvt.hpp>
#include <util/reflection.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
static_assert(sizeof(any_t) == 64, "Expecting sizeof(any_t)==64");

namespace utils {
reflection::general_ref_t any_to_general_ref(const any_t &v) {
    COMPILE_ASSERT(!v.empty(), "any_to_general_ref meets empty any value");
    auto ty = reflection::get_type_by_rtti(v.type_code());
    COMPILE_ASSERT(ty,
            "Cannot find the type in reflection for any_t: "
                    << v.type_code()->name());
    return reflection::general_ref_t {(void *)v.get_raw(), *ty};
}
} // namespace utils

namespace any_detail {
static std::unordered_map<const std::type_info *, any_vtable_t *> &
get_rtti_vtable_map() {
    static std::unordered_map<const std::type_info *, any_vtable_t *> map;
    return map;
}

void any_vtable_t::set_rtti_to_vtable_map(
        const std::type_info *typeinfo, any_vtable_t *table) {
    auto &map = get_rtti_vtable_map();
    assert(map.find(typeinfo) == map.end());
    map.insert(std::make_pair(typeinfo, table));
}

any_vtable_t *any_vtable_t::get_vtable_by_rtti(const std::type_info *typeinfo) {
    auto &map = get_rtti_vtable_map();
    auto itr = map.find(typeinfo);
    COMPILE_ASSERT(itr != map.end(),
            "Cannot find any_vtable_t for type: " << typeinfo->name());
    return itr->second;
}

} // namespace any_detail

void any_t::create_buffer(const any_detail::any_vtable_t *vt) {
    if (vt) {
        if (vt->size_ > INLINE_BUFFER_SIZE) data_.ptr_ = new char[vt->size_];
    }
    vtable_ = vt;
}

any_t any_t::make_by_type(const reflection::type *ty) {
    any_t ret;
    if (ty->base_ == reflection::basic_type::t_string
            && ty->array_depth_ == 0) {
        ret = std::string();
    } else {
        auto rtti = reflection::get_rtti_by_type(ty);
        ret.switch_buffer_to_type(
                any_detail::any_vtable_t::get_vtable_by_rtti(rtti));
        if (ty->meta_) { ty->meta_->constructor_(ret.get_raw()); }
    }
    return ret;
}

// switches the buffer to a type. If we are already holding this type, do
// nothing and return true. Else, we release the held object, create new
// buffer, change the vtable and return false
bool any_t::switch_buffer_to_type(const any_detail::any_vtable_t *vt) {
    if (vtable_ != vt) {
        clear();
        create_buffer(vt);
        return false;
    }
    // if same type, no need to clear()
    return true;
}

void any_t::copy_from(const void *data, const any_detail::any_vtable_t *vt) {
    if (vtable_ != vt) {
        if (vt && !vt->copy_ctor_) {
            throw std::runtime_error("The type is not copy-constructible");
        }
    } else {
        if (vt && !vt->copy_assigner_) {
            throw std::runtime_error("The type is not copy-assignable");
        }
    }
    if (switch_buffer_to_type(vt)) {
        if (vtable_) vtable_->copy_assigner_(get_raw(), (void *)data);
    } else {
        if (vtable_) vtable_->copy_ctor_(get_raw(), (void *)data);
    }
}

void any_t::move_from(void *data, const any_detail::any_vtable_t *vt) {
    if (vtable_ != vt) {
        if (vt && !vt->move_ctor_) {
            throw std::runtime_error("The type is not move-constructible");
        }
    } else {
        if (vt && !vt->move_assigner_) {
            throw std::runtime_error("The type is not move-assignable");
        }
    }
    if (switch_buffer_to_type(vt)) {
        if (vtable_) vtable_->move_assigner_(get_raw(), data);
    } else {
        if (vtable_) vtable_->move_ctor_(get_raw(), data);
    }
}

void any_t::clear() {
    if (vtable_) {
        if (vtable_->size_ > INLINE_BUFFER_SIZE) {
            vtable_->destructor_(data_.ptr_);
            delete[] data_.ptr_;
        } else {
            vtable_->destructor_(&data_.inlined_buffer_);
        }
        vtable_ = nullptr;
        data_.ptr_ = nullptr;
    }
}

// if we move from any_t, we can simply "steal" the pointer and vtable
void any_t::move_from_any(any_t &&v) {
    clear();
    if (v.vtable_) {
        vtable_ = v.vtable_;
        if (vtable_->size_ > INLINE_BUFFER_SIZE) {
            // large object, it is a pointer. Move the pointer
            data_.ptr_ = v.data_.ptr_;
            v.data_.ptr_ = nullptr;
            v.vtable_ = nullptr;
        } else {
            // small object, we cannot steal the inlined buffer. Call move ctor
            // instead
            vtable_->move_ctor_(
                    &data_.inlined_buffer_, &v.data_.inlined_buffer_);
        }
    }
}

int any_t::cmp(const any_t &other) const {
    COMPILE_ASSERT(!empty() && !other.empty(), "Comparing an empty any_t");
    reflection::general_ref_t ths = utils::any_to_general_ref(*this);
    reflection::general_ref_t ohr = utils::any_to_general_ref(other);
    return ths.cmp(ohr);
}

size_t any_t::hash() const {
    COMPILE_ASSERT(!empty(), "Hashing an empty any_t");
    return utils::any_to_general_ref(*this).hash();
}

any_t &any_map_t::get_any(const std::string &v) {
    auto itr = impl_.find(v);
    COMPILE_ASSERT(
            itr != impl_.end(), "Cannot find the key " << v << " in the map");
    return itr->second;
}

const any_t &any_map_t::get_any(const std::string &v) const {
    auto itr = impl_.find(v);
    COMPILE_ASSERT(
            itr != impl_.end(), "Cannot find the key " << v << " in the map");
    return itr->second;
}

bool any_map_t::operator==(const any_map_t &other) const {
    if (impl_.size() != other.impl_.size()) { return false; }
    for (auto &kv : impl_) {
        auto otherkv = other.impl_.find(kv.first);
        if (otherkv == other.impl_.end()) { return false; }
        if (kv.second.cmp(otherkv->second) != 0) { return false; }
    }
    return true;
}

size_t any_map_t::hash() const {
    size_t result = 0;
    // using xor to combine the hash results.
    for (auto &kv : impl_) {
        result ^= std::hash<std::string>()(kv.first);
        result ^= kv.second.hash();
    }
    return result;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
