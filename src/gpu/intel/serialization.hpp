/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_INTEL_SERIALIZATION_HPP
#define GPU_INTEL_SERIALIZATION_HPP

#include <iomanip>
#include <sstream>

#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

#define assert_trivially_serializable(cls) \
    static_assert(serialized_data_t::is_trivially_serialized<cls>::value, \
            #cls " must be trivially serializable.")

struct serialized_data_t {
    serialized_data_t() = default;

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

    const std::vector<uint8_t> &get_data() const { return data; }
    void set_data(std::vector<uint8_t> d) { this->data = std::move(d); }

    template <typename T>
    struct has_serialize {
        using yes_t = uint8_t;
        using no_t = uint16_t;

        template <typename U>
        static yes_t test(gpu_utils::enable_if_t<
                std::is_same<decltype(&U::serialize),
                        void (U::*)(serialized_data_t &) const>::value,
                bool>);
        template <typename U>
        static no_t test(...);

        static const bool value = (sizeof(test<T>(0)) == sizeof(yes_t));
    };

    // Append helper function for structures with the member function
    // void serialize(serialized_data_t &) const
    template <typename T,
            gpu_utils::enable_if_t<has_serialize<T>::value, bool> = true>
    void append(const T &t) {
        t.serialize(*this);
    }

    // Append helper function for trivially serialized objects
    template <typename T,
            gpu_utils::enable_if_t<is_trivially_serialized<T>::value
                            && !has_serialize<T>::value,
                    bool> = true>
    void append(const T &t) {
        std::array<uint8_t, sizeof(T)> type_data;
        std::memcpy(type_data.data(), &t, sizeof(T));
        data.insert(data.end(), type_data.begin(), type_data.end());
    }

    template <typename T,
            gpu_utils::enable_if_t<gpu_utils::is_vector<T>::value, bool> = true>
    void append(const T &v) {
        append(v.size());
        for (const typename T::value_type &d : v)
            append<typename T::value_type>(d);
    };

    template <typename Arg1, typename Arg2, typename... Args>
    void append(const Arg1 &a1, const Arg2 &a2, const Args &...args) {
        append(a1);
        append(a2, args...);
    }

    template <typename T,
            gpu_utils::enable_if_t<is_trivially_serialized<T>::value,
                    bool> = true>
    T get(size_t idx) const {
        T t {};
        if (data.size() < idx + sizeof(T)) {
            assert(!"unexpected");
            return t;
        }
        std::memcpy(&t, &data[idx], sizeof(T));
        return t;
    }

    size_t hash() const { return hash_range(data.data(), data.size()); };
    std::string str() {
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (auto c : data) {
            oss << std::setw(2) << static_cast<uint32_t>(c);
        }
        return oss.str();
    }

protected:
    std::vector<uint8_t> data;
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
    };
};

struct serialized_t : public serialized_data_t {
    template <typename Arg1, typename... Args>
    serialized_t(const Arg1 &a1, const Args &...args) {
        append(a1, args...);
    }

    static serialized_t from_data(std::vector<uint8_t> data) {
        serialized_t s;
        s.set_data(std::move(data));
        return s;
    };

    bool operator==(const serialized_t &other) const {
        return data == other.data;
    }

    size_t get_hash() const { return hash(); }
    template <typename T>
    static size_t get_hash(const T &t) {
        return serialized_t(t).get_hash();
    }

private:
    serialized_t() = default;
};

struct deserializer_t {
    deserializer_t(const serialized_data_t &s) : idx(0), s(s) {}

    template <typename T>
    struct has_deserialize {
        using yes_t = uint8_t;
        using no_t = uint16_t;

        template <typename U>
        static yes_t test(
                gpu_utils::enable_if_t<std::is_same<decltype(&U::deserialize),
                                               U (*)(deserializer_t &)>::value,
                        bool>);
        template <typename U>
        static no_t test(...);

        static const bool value = (sizeof(test<T>(0)) == sizeof(yes_t));
    };

    // Helper function for structures with the static member function
    // void deserialize(deserializer_t&)
    template <typename T,
            gpu_utils::enable_if_t<has_deserialize<T>::value, bool> = true>
    void pop(T &t) {
        t = T::deserialize(*this);
    }
    template <typename T,
            gpu_utils::enable_if_t<has_deserialize<T>::value, bool> = true>
    T pop() {
        return T::deserialize(*this);
    }

    template <typename T,
            gpu_utils::enable_if_t<
                    serialized_data_t::is_trivially_serialized<T>::value
                            && !has_deserialize<T>::value,
                    bool> = true>
    void pop(T &t) {
        t = s.get<T>(idx);
        idx += sizeof(T);
    };
    template <typename T,
            gpu_utils::enable_if_t<
                    serialized_data_t::is_trivially_serialized<T>::value
                            && !has_deserialize<T>::value,
                    bool> = true>
    T pop() {
        auto idx_start = idx;
        idx += sizeof(T);
        return s.get<T>(idx_start);
    };

    // Helper for vector types
    template <typename T,
            gpu_utils::enable_if_t<gpu_utils::is_vector<T>::value, bool> = true>
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

    size_t idx;
    const serialized_data_t &s;
};

template <typename T>
struct trivially_serializable_t {
    static constexpr bool is_trivially_validatable = true;

    serialized_t serialize() const {
        assert_trivially_serializable(T);
        return serialized_t(*static_cast<const T *>(this));
    }

    static T deserialize(const serialized_t &s) {
        return deserializer_t(s).pop<T>();
    }
};

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
