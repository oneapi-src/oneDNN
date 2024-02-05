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

#ifndef GPU_SERIALIZATION_HPP
#define GPU_SERIALIZATION_HPP

#include <iomanip>
#include <sstream>

#include "common/serialization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

#define assert_trivially_serializable(cls) \
    static_assert(serialized_data_t::is_trivially_serialized<cls>::value, \
            #cls " must be trivially serializable.")

struct serialized_data_t {
#if defined(__cpp_lib_has_unique_object_representations) \
        && __cpp_lib_has_unique_object_representations >= 201606L
    template <typename T>
    struct is_trivially_serialized {
        static const bool value
                = std::has_unique_object_representations<T>::value
                || std::is_floating_point<T>::value;
    };

#else
    // Fallback for backward compatibility. As the structure layout should not
    // change between c++ versions, compiling with c++17 will already verify the
    // structures are valid for this use case.
    template <typename T>
    struct is_trivially_serialized {
        static const bool value = std::is_trivially_copyable<T>::value;
    };
#endif

    const std::vector<uint8_t> &get_data() const { return data; }
    void set_data(const std::vector<uint8_t> &data) { this->data = data; }

    template <typename T,
            typename
            = typename std::enable_if<is_trivially_serialized<T>::value>::type>
    void append(const T &t) {
        std::array<uint8_t, sizeof(T)> type_data;
        std::memcpy(type_data.data(), &t, sizeof(T));
        data.insert(data.end(), type_data.begin(), type_data.end());
    }
    void append(const post_ops_t &post_ops) {
        append(post_ops.len());
        serialization_stream_t sstream {};
        serialization::serialize_post_ops(sstream, post_ops);
        auto post_op_data = sstream.get_data();
        data.insert(data.end(), post_op_data.begin(), post_op_data.end());
    }

    template <typename Arg1, typename... Args>
    void append(const Arg1 &a1, const Args &...args) {
        append(a1);
        append(args...);
    }

    template <typename T>
    void append(const std::vector<T> &v) {
        append(v.size());
        for (const T &d : v)
            append(d);
    }

    template <typename T>
    void append_complex(const T &t) {
        t.serialize(*this);
    }

    template <typename Arg1, typename... Args>
    void append_complex(const Arg1 &a1, const Args &...args) {
        append_complex(a1);
        append_complex(args...);
    }

    template <typename T,
            typename
            = typename std::enable_if<is_trivially_serialized<T>::value>::type>
    T get(size_t idx) const {
        T t {};
        if (data.size() < idx + sizeof(T)) {
            assert(!"unexpected");
            return t;
        }
        std::memcpy(&t, &data[idx], sizeof(T));
        return t;
    }

    size_t hash() { return hash_range(data.data(), data.size()); };
    std::string str() {
        std::ostringstream oss;
        oss << std::hex << std::setfill('0') << std::setw(2);
        for (auto c : data) {
            oss << static_cast<uint32_t>(c);
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
    bool operator==(const serialized_t &other) const {
        return data == other.data;
    }
};

struct deserializer_t {
    deserializer_t(const serialized_data_t &s) : idx(0), s(s) {}
    template <typename T,
            typename = typename std::enable_if<
                    serialized_data_t::is_trivially_serialized<T>::value>::type>
    void pop(T &t) {
        t = s.get<T>(idx);
        idx += sizeof(T);
    };

    size_t idx;
    const serialized_data_t &s;
};

template <typename T,
        typename = typename std::enable_if<
                serialized_data_t::is_trivially_serialized<T>::value>::type>
size_t get_hash(const T *t) {
    serialized_t s {};
    s.append(*t);
    return s.hash();
}

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
