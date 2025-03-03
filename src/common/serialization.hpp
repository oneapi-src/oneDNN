/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#ifndef COMMON_SERIALIZATION_HPP
#define COMMON_SERIALIZATION_HPP

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <type_traits>

#include "common/utils.hpp"

namespace dnnl {
namespace impl {

#define DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(cls) \
    static_assert(serialization_stream_t::is_trivially_serialized<cls>::value, \
            #cls " must be trivially serializable.")

struct serialization_stream_t {
    serialization_stream_t() = default;

    template <typename Arg1, typename... Args>
    serialization_stream_t(const Arg1 &a1, const Args &...args) {
        append(a1, args...);
    }

    static serialization_stream_t from_data(std::vector<uint8_t> data) {
        serialization_stream_t s;
        s.data_ = std::move(data);
        return s;
    }

    bool operator==(const serialization_stream_t &other) const {
        return data_ == other.data_;
    }

#if defined(__cpp_lib_has_unique_object_representations) \
        && __cpp_lib_has_unique_object_representations >= 201606L
    template <typename T>
    struct is_trivially_serialized {
        static const bool value
                = (std::has_unique_object_representations<T>::value
                          || std::is_floating_point<T>::value)
                && !(std::is_pointer<T>::value);
    };

#else
    // Fallback for backward compatibility. As the structure layout should not
    // change between c++ versions, compiling with c++17 will already verify the
    // structures are valid for this use case.
    template <typename T>
    struct is_trivially_serialized {
        static const bool value = std::is_trivially_copyable<T>::value
                && !(std::is_pointer<T>::value);
    };
#endif

    template <typename T>
    struct has_serialize_t {
        using yes_t = uint8_t;
        using no_t = uint16_t;

        template <typename U>
        static yes_t test(utils::enable_if_t<
                std::is_same<decltype(&U::serialize),
                        void (U::*)(serialization_stream_t &) const>::value,
                bool>);
        template <typename U>
        static no_t test(...);

        static const bool value = (sizeof(test<T>(0)) == sizeof(yes_t));
    };

    // Append helper function for structures with the member function
    // void serialize(serialization_stream_t &) const
    template <typename T,
            utils::enable_if_t<has_serialize_t<T>::value, bool> = true>
    void append(const T &t) {
        t.serialize(*this);
    }

    // Append helper function for trivially serialized objects
    template <typename T,
            utils::enable_if_t<is_trivially_serialized<T>::value
                            && !has_serialize_t<T>::value,
                    bool> = true>
    void append(const T &t) {
        std::array<uint8_t, sizeof(T)> type_data;
        std::memcpy(type_data.data(), &t, sizeof(T));
        data_.insert(data_.end(), type_data.begin(), type_data.end());
    }

    template <typename T,
            utils::enable_if_t<utils::is_vector<T>::value, bool> = true>
    void append(const T &v) {
        append(v.size());
        for (const typename T::value_type &d : v)
            append<typename T::value_type>(d);
    }

    template <typename Arg1, typename Arg2, typename... Args>
    void append(const Arg1 &a1, const Arg2 &a2, const Args &...args) {
        append(a1);
        append(a2, args...);
    }

    template <typename T,
            utils::enable_if_t<is_trivially_serialized<T>::value, bool> = true>
    void append_array(size_t size, const T *ptr) {
        append(size);
        const auto *p = reinterpret_cast<const uint8_t *>(ptr);
        data_.insert(data_.end(), p, p + sizeof(T) * size);
    }

    template <typename T,
            utils::enable_if_t<is_trivially_serialized<T>::value, bool> = true>
    T get(size_t idx) const {
        T t {};
        if (data_.size() < idx + sizeof(T)) {
            assert(!"unexpected");
            return t;
        }
        std::memcpy(&t, &data_[idx], sizeof(T));
        return t;
    }

    void get(size_t idx, size_t size, uint8_t *ptr) const {
        if (data_.size() < idx + size) {
            assert(!"unexpected");
            return;
        }
        std::memcpy(ptr, &data_[idx], size);
    }

    size_t get_hash() const { return hash_range(data_.data(), data_.size()); }

    template <typename T>
    static size_t get_hash(const T &t) {
        return serialization_stream_t(t).get_hash();
    }

    std::string str() {
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (auto c : data_) {
            oss << std::setw(2) << static_cast<uint32_t>(c);
        }
        return oss.str();
    }

    bool empty() const { return data_.empty(); }

    const std::vector<uint8_t> &get_data() const { return data_; }

private:
    static size_t hash_range(const uint8_t *v, size_t size) {
        size_t seed = 0;
        const uint8_t *end = v + size;
        for (; v < end; v += sizeof(seed)) {
            size_t value = 0;
            std::memcpy(&value, v,
                    std::min(static_cast<size_t>(end - v), sizeof(seed)));
            seed = hash_combine(seed, value);
        }

        return seed;
    }

    std::vector<uint8_t> data_;
};

struct deserializer_t {
    deserializer_t(const serialization_stream_t &sstream)
        : idx_(0), sstream_(sstream) {}

    template <typename T>
    struct has_deserialize_t {
        using yes_t = uint8_t;
        using no_t = uint16_t;

        template <typename U>
        static yes_t test(
                utils::enable_if_t<std::is_same<decltype(&U::deserialize),
                                           U (*)(deserializer_t &)>::value,
                        bool>);
        template <typename U>
        static no_t test(...);

        static const bool value = (sizeof(test<T>(0)) == sizeof(yes_t));
    };

    // Helper function for structures with the static member function
    // void deserialize(deserializer_t&)
    template <typename T,
            utils::enable_if_t<has_deserialize_t<T>::value, bool> = true>
    void pop(T &t) {
        t = T::deserialize(*this);
    }
    template <typename T,
            utils::enable_if_t<has_deserialize_t<T>::value, bool> = true>
    T pop() {
        return T::deserialize(*this);
    }

    template <typename T,
            utils::enable_if_t<
                    serialization_stream_t::is_trivially_serialized<T>::value
                            && !has_deserialize_t<T>::value,
                    bool> = true>
    void pop(T &t) {
        t = sstream_.get<T>(idx_);
        idx_ += sizeof(T);
    }

    template <typename T,
            utils::enable_if_t<
                    serialization_stream_t::is_trivially_serialized<T>::value
                            && !has_deserialize_t<T>::value,
                    bool> = true>
    T pop() {
        auto idx_start = idx_;
        idx_ += sizeof(T);
        return sstream_.get<T>(idx_start);
    }

    // Helper for vector types
    template <typename T,
            utils::enable_if_t<utils::is_vector<T>::value, bool> = true>
    void pop(T &v) {
        size_t size;
        pop(size);
        v.clear();
        v.reserve(size);
        for (size_t i = 0; i < size; i++) {
            typename T::value_type t = {};
            pop(t);
            v.emplace_back(t);
        }
    }

    template <typename T,
            utils::enable_if_t<
                    serialization_stream_t::is_trivially_serialized<T>::value,
                    bool> = true>
    void pop_array(size_t &size, T *ptr) {
        pop(size);
        sstream_.get(idx_, sizeof(T) * size, reinterpret_cast<uint8_t *>(ptr));
        idx_ += sizeof(T) * size;
    }

    bool empty() const { return idx_ >= sstream_.get_data().size(); }

private:
    size_t idx_ = 0;
    const serialization_stream_t &sstream_;
};

template <typename T>
struct trivially_serializable_t {
    static constexpr bool is_trivially_validatable = true;

    serialization_stream_t serialize() const {
        DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(T);
        return serialization_stream_t(*static_cast<const T *>(this));
    }

    static T deserialize(const serialization_stream_t &s) {
        return deserializer_t(s).pop<T>();
    }
};

} // namespace impl
} // namespace dnnl

#endif
