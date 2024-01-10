/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#ifndef GPU_JIT_UTILS_UTILS_HPP
#define GPU_JIT_UTILS_UTILS_HPP

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <functional>
#include <iomanip>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "common/math_utils.hpp"
#include "common/utils.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/utils.hpp"

#ifdef DNNL_DEV_MODE
#include "common/profiler.hpp"
#include "common/verbose.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

template <typename T, typename = decltype(std::declval<T>().str(), void())>
inline std::ostream &operator<<(std::ostream &out, const T &obj) {
    out << obj.str();
    return out;
}

namespace ir_utils {

const int LOG_OFF = 0;
const int LOG_WARNING = 100;
const int LOG_SUGGESTION = 120;
const int LOG_INFO = 150;
const int LOG_PERF = 170;
const int LOG_TRACE = 200;

#ifdef DNNL_DEV_MODE
const int LOG_LEVEL = LOG_WARNING;
#else
const int LOG_LEVEL = LOG_OFF;
#endif

template <typename T>
size_t get_hash(const T &t);

template <typename T, size_t N>
size_t get_hash(const std::array<T, N> &a);

template <typename T>
size_t get_hash(const std::vector<T> &v);

template <typename T>
void get_hash_impl(size_t &h, const T &t) {
    h = hash_combine(h, get_hash(t));
}

template <typename ArgHeadT, typename... ArgsT>
void get_hash_impl(size_t &h, const ArgHeadT &head, const ArgsT &...args) {
    size_t h_head = get_hash(head);
    h = hash_combine(h, h_head);
    get_hash_impl(h, args...);
}

template <typename E>
struct enum_hash_t {
    size_t operator()(const E &e) const noexcept {
        return std::hash<size_t>()((size_t)e);
    }
};

template <typename T, typename = void>
struct get_std_hash_helper_t {
    static size_t call(const T &t) { return std::hash<T>()(t); }
};

template <typename T>
struct get_std_hash_helper_t<T,
        typename std::enable_if<std::is_enum<T>::value>::type> {
    static size_t call(const T &t) { return enum_hash_t<T>()(t); }
};

template <typename T, typename = void>
struct get_hash_helper_t {
    static size_t call(const T &t) { return get_std_hash_helper_t<T>::call(t); }
};

template <typename T>
struct get_hash_helper_t<T, decltype(std::declval<T>().get_hash(), void())> {
    static size_t call(const T &t) { return t.get_hash(); }
};

template <typename T>
size_t get_hash(const T &t) {
    return get_hash_helper_t<T>::call(t);
}

template <typename T, size_t N>
size_t get_hash(const std::array<T, N> &a) {
    size_t h = 0;
    for (auto &e : a)
        h = hash_combine(h, get_hash(e));
    return h;
}

template <typename T>
size_t get_hash(const std::vector<T> &v) {
    size_t h = 0;
    for (auto &e : v)
        h = hash_combine(h, get_hash(e));
    return h;
}

template <typename... ArgsT>
size_t get_hash(const ArgsT &...args) {
    size_t h = 0;
    get_hash_impl(h, args...);
    return h;
}

template <typename T>
struct hasher_t {
    size_t operator()(const T &t) const { return t.get_hash(); }
};

template <typename T, typename U, typename = void>
struct is_equal_helper_t {
    static bool call(const T &t, const U &u) { return t == u; }
};

template <typename T, typename U>
struct is_equal_helper_t<T, U,
        decltype(std::declval<T>().is_equal(std::declval<U>()), void())> {
    static bool call(const T &t, const U &u) { return t.is_equal(u); }
};

// Checks equality of objects:
// 1. Uses t.is_equal(u) if is_equal() is available
// 2. Uses (t == u) otherwise
template <typename T, typename U>
bool is_equal(const T &t, const U &u) {
    return is_equal_helper_t<T, U>::call(t, u);
}

// Checks equality of vector elements.
template <typename T, typename U>
bool is_equal(const std::vector<T> &a, const std::vector<U> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (!ir_utils::is_equal(a[i], b[i])) return false;
    return true;
}

// Checks equality of vector elements between each other.
template <typename T>
bool are_all_equal(const std::vector<T> &a) {
    if (a.empty()) return true;
    for (size_t i = 1; i < a.size(); i++)
        if (!ir_utils::is_equal(a[i], a[0])) return false;
    return true;
}

// Checks identity of vector elements.
template <typename T, typename U>
bool is_same(const std::vector<T> &a, const std::vector<U> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (!a[i].is_same(b[i])) return false;
    return true;
}

template <typename T, typename U>
bool contains(const std::vector<T> &vec, const U &u) {
    for (auto &v : vec)
        if (v == u) return true;
    return false;
}

// Checks assertion and, in case of error, evaluates output operators to print
// related messages. Usage:
//     ir_assert(condition) << "Error message" << ...;

#if !defined(NDEBUG) || defined(DNNL_DEV_MODE)
#define ir_assert(cond) \
    !(cond) \
            && dnnl::impl::gpu::gpu_utils::error_stream_t( \
                    __FILE__, __LINE__, #cond)
#else
#define ir_assert(cond) \
    (false) && !(cond) \
            && dnnl::impl::gpu::gpu_utils::error_stream_t( \
                    __FILE__, __LINE__, #cond)
#endif

#define ir_error_not_expected() ir_assert(false) << "Not expected. "
#define ir_except_not_implemented(msg) throw std::runtime_error(msg)

template <int level, bool value = true, bool add_new_line = false>
class base_logger_t {
public:
    base_logger_t(std::ostream &out = std::cout) : out_(out) {}
    ~base_logger_t() {
        if (add_new_line) out_ << std::endl;
    }

    static bool is_enabled() {
#if defined(DNNL_DEV_MODE)
        return get_verbose(verbose_t::debuginfo) >= level;
#else
        return false;
#endif
    }

    operator bool() const { return value; }

    template <typename T>
    base_logger_t &operator<<(const T &obj) {
        using dnnl::impl::gpu::jit::operator<<;
        maybe_print_header();
        out_ << obj;
        return *this;
    }

    base_logger_t &operator<<(std::ostream &(*os)(std::ostream &)) {
        maybe_print_header();
        out_ << os;
        return *this;
    }

private:
    void maybe_print_header() {
        if (!is_first_print_) return;

        switch (level) {
            case LOG_WARNING: out_ << "[WARNING] "; break;
            case LOG_SUGGESTION: out_ << "[SUGGESTION] "; break;
            default: break;
        }
        is_first_print_ = false;
    }

    std::ostream &out_;
    bool is_first_print_ = true;
};

template <int level>
class logger_t : public base_logger_t<level> {
public:
    logger_t() : base_logger_t<level>() {}
};

template <int level>
base_logger_t<level, false, true> make_check_logger() {
    return base_logger_t<level, false, true>();
}

#define ir_perf() \
    ir_utils::logger_t<ir_utils::LOG_PERF>::is_enabled() \
            && ir_utils::logger_t<ir_utils::LOG_PERF>()

// Trace can result in overhead making measurement meaningless
#define ir_perf_no_trace() \
    ir_utils::logger_t<ir_utils::LOG_PERF>::is_enabled() \
            && !ir_utils::logger_t<ir_utils::LOG_TRACE>::is_enabled() \
            && ir_utils::logger_t<ir_utils::LOG_PERF>()

#define ir_info() \
    ir_utils::logger_t<ir_utils::LOG_INFO>::is_enabled() \
            && ir_utils::logger_t<ir_utils::LOG_INFO>()

#define ir_warning() \
    ir_utils::logger_t<ir_utils::LOG_WARNING>::is_enabled() \
            && ir_utils::logger_t<ir_utils::LOG_WARNING>()

#define ir_suggestion() \
    ir_utils::logger_t<ir_utils::LOG_SUGGESTION>::is_enabled() \
            && ir_utils::logger_t<ir_utils::LOG_SUGGESTION>()

#define ir_trace() \
    ir_utils::logger_t<ir_utils::LOG_TRACE>::is_enabled() \
            && ir_utils::logger_t<ir_utils::LOG_TRACE>()

#define ir_check(cond) \
    if (!(cond)) \
    return ir_utils::logger_t<ir_utils::LOG_TRACE>::is_enabled() \
            && ir_utils::make_check_logger<ir_utils::LOG_TRACE>()

// Pretty printers for STL objects.
template <typename KeyT, typename HashT, typename EqualT>
inline std::ostream &operator<<(
        std::ostream &out, const std::unordered_set<KeyT, HashT, EqualT> &s) {
    out << "{";
    for (auto it = s.begin(); it != s.end(); it++) {
        out << (it != s.begin() ? ", " : "") << *it;
    }
    out << "}";
    return out;
}

template <typename KeyT, typename ValueT, typename HashT, typename EqualT>
inline std::ostream &operator<<(std::ostream &out,
        const std::unordered_map<KeyT, ValueT, HashT, EqualT> &m) {
    out << "{";
    for (auto it = m.begin(); it != m.end(); it++) {
        out << (it != m.begin() ? ", " : "") << it->first << ": " << it->second;
    }
    out << "}";
    return out;
}

template <typename ContainerT>
struct seq_print_helper_t {
    seq_print_helper_t(const ContainerT &v, const std::string &sep, int width)
        : v(v), sep(sep), width(width) {}

    const ContainerT &v;
    const std::string sep;
    int width;
};

template <typename T>
seq_print_helper_t<T> make_seq_print_helper(
        const T &v, const std::string &sep = ", ", int width = 0) {
    return seq_print_helper_t<T>(v, sep, width);
}

template <typename T>
inline std::ostream &operator<<(
        std::ostream &out, const seq_print_helper_t<T> &seq) {
    for (auto it = seq.v.begin(); it != seq.v.end(); it++) {
        out << (it != seq.v.begin() ? seq.sep : "") << std::setw(seq.width)
            << *it;
    }
    return out;
}

template <typename T>
inline std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
    out << "[";
    out << make_seq_print_helper(v);
    out << "]";
    return out;
}

template <typename T, typename = void>
struct str_ostream_helper_t {
    static std::string call(const T &t) {
        ir_error_not_expected();
        return {};
    }
};

template <typename T>
struct str_ostream_helper_t<T,
        decltype(std::declval<std::ostream>() << std::declval<T>(), void())> {
    static std::string call(const T &t) {
        std::ostringstream oss;
        oss << t;
        return oss.str();
    }
};

template <typename T, typename = void>
struct str_helper_t {
    static std::string call(const T &t) {
        return str_ostream_helper_t<T>::call(t);
    }
};

template <typename T>
struct str_helper_t<T, decltype(std::declval<T>().str(), void())> {
    static std::string call(const T &t) { return t.str(); }
};

template <typename T>
struct str_helper_t<std::vector<T>, void> {
    static std::string call(const std::vector<T> &v) {
        std::ostringstream oss;
        oss << v;
        return oss.str();
    }
};

// Helper class to pretty-print tables.
// Each operator<<() call corresponds to one cell/header. std::endl or '/n'
// moves to the next row.
class table_t {
public:
    table_t(const std::string &title, const std::vector<std::string> &header)
        : title_(title), header_(header) {}

    template <typename T>
    table_t &operator<<(const T &value) {
        std::ostringstream oss;
        oss << value;
        auto str_value = oss.str();
        size_t pos = 0;
        for (size_t i = 0; i < str_value.length(); i++) {
            if (str_value[i] != '\n') continue;
            cur_row_.push_back(str_value.substr(pos, i - pos));
            new_row();
            pos = i + 1;
        }
        if (str_value.empty() || pos != str_value.length()) {
            cur_row_.push_back(str_value.substr(pos, str_value.length() - pos));
        }
        return *this;
    }

    table_t &operator<<(std::ostream &(*f)(std::ostream &)) {
        auto _endl
                = (std::basic_ostream<char> & (*)(std::basic_ostream<char> &))
                        std::endl;
        if (f == _endl) new_row();
        return *this;
    }

    std::string str() const {
        std::ostringstream oss;
        size_t n = header_.size();
        std::vector<size_t> widths(n);
        for (size_t i = 0; i < n; i++)
            widths[i] = header_[i].length();
        for (auto &r : rows_) {
            for (size_t i = 0; i < n; i++) {
                widths[i] = std::max(widths[i], r[i].length());
            }
        }
        auto print = [&](std::ostream &out, size_t idx, const std::string &s,
                             char pad = ' ') {
            int w = (int)widths[idx];
            if (idx == 0) out << "|" << pad;
            out << std::setw(w);
            out << std::left;
            out << s;
            out << pad << "|";
            if (idx != n - 1) out << pad;
        };
        oss << "=== " << title_ << std::endl;
        for (size_t i = 0; i < n; i++) {
            print(oss, i, header_[i]);
        }
        oss << std::endl;
        for (size_t i = 0; i < n; i++) {
            print(oss, i, std::string(widths[i], '-'), '-');
        }
        oss << std::endl;
        for (auto &r : rows_) {
            for (size_t i = 0; i < n; i++) {
                print(oss, i, r[i]);
            }
            if (&r != &rows_.back()) oss << std::endl;
        }
        return oss.str();
    }

private:
    void new_row() {
        ir_assert(cur_row_.size() == header_.size());
        rows_.emplace_back();
        rows_.back().swap(cur_row_);
    }

    std::string title_;
    std::vector<std::string> header_;
    std::vector<std::vector<std::string>> rows_;

    std::vector<std::string> cur_row_;
};

inline std::string to_string(bool b) {
    return b ? "True" : "False";
}

inline std::string to_lower(const std::string &s) {
    auto ret = s;
    std::transform(ret.begin(), ret.end(), ret.begin(),
            [](char c) { return std::tolower(c); });
    return ret;
}

inline std::string add_indent(const std::string &s, const std::string &indent,
        bool skip_first = false) {
    auto lines = gpu_utils::split(s, "\n");
    std::ostringstream oss;
    for (int i = 0; i < (int)lines.size(); i++) {
        if (i > 0) oss << std::endl;
        if (i == 0 && skip_first) {
            oss << " ";
        } else {
            oss << indent;
        }
        oss << lines[i];
    }
    return oss.str();
}

inline std::string add_tag(
        const std::string &tag, const std::string &s, bool eol = true) {
    std::ostringstream oss;
    oss << tag << ":";
    if (eol) oss << std::endl;
    oss << add_indent(s, "  ", /*skip_first=*/!eol);
    return oss.str();
}

template <typename T>
inline T max_divisor(T n, std::initializer_list<T> divisors) {
    T ret = -1;
    for (auto d : divisors) {
        if (n % d == 0) ret = std::max(ret, d);
    }
    ir_assert(ret != -1);
    return ret;
}

// Equivalent of BLSI instruction (extract lowest set isolated bit).
template <typename T>
inline T max_pow2_divisor(T n) {
    return n & ~(n - 1);
}

template <typename T, typename U>
inline T safe_divide(T a, U b) {
    ir_assert(b != 0 && a % b == 0) << "Can't divide: " << a << " / " << b;
    return a / b;
}

template <typename T, typename U>
inline T safe_div(T a, U b) {
    return safe_divide(a, b);
}

template <typename ContainerT, typename T>
inline int find_index(const ContainerT &c, const T &value) {
    for (int i = 0; i < int(c.size()); i++) {
        if (c[i] == value) return i;
    }
    return -1;
}

template <typename T, typename F>
void for_each_impl(size_t pos, std::vector<T> &idx,
        const std::vector<T> &bounds, const F &f) {
    if (pos == bounds.size()) {
        f(idx);
        return;
    }

    for (T i = 0; i < bounds[pos]; i++) {
        idx[pos] = i;
        for_each_impl(pos + 1, idx, bounds, f);
    }
}

template <typename T, typename F>
void for_each(const std::vector<T> &bounds, const F &f) {
    std::vector<T> idx(bounds.size());
    for_each_impl(0, idx, bounds, f);
}

template <typename MapContainerT, typename KeyT,
        typename ValueT = typename MapContainerT::mapped_type>
ValueT get_or_default(const MapContainerT &map, const KeyT &key,
        const ValueT &default_value) {
    auto it = map.find(key);
    if (it == map.end()) return default_value;
    return it->second;
}

struct debug_profiler_t {
#ifdef DNNL_DEV_MODE
    debug_profiler_t(std::string profile_name) : profile(profile_name) {};
    void start() { profile.start(); };
    void stamp(const char *name) { profile.stamp(name); };
    void stop(const char *name) { profile.stop(name); };
    void stop() { profile.stop(); };
    void reset() { profile.reset(); };
    std::string str() const { return profile.str(); };

private:
    profiler_t profile;
#else
    debug_profiler_t(std::string) {};
    void start() {};
    void stamp(const char *name) {};
    void stop(const char *name) {};
    void stop() {};
    void reset() {};
    std::string str() const { return ""; };
#endif
};

template <typename T>
T quantize(float v, float v_min = 0, float v_max = 1) {
    static_assert(std::is_integral<T>::value, "T must be integer.");
    float f = (v - v_min) / v_max;
    float t_min = std::numeric_limits<T>::min();
    float t_max = std::numeric_limits<T>::max();
    return (T)(t_min + (t_max - t_min) * f + 0.5);
}

template <typename T>
float dequantize(T t, float v_min = 0, float v_max = 1) {
    float t_min = std::numeric_limits<T>::min();
    float t_max = std::numeric_limits<T>::max();
    float f = (t - t_min) / (t_max - t_min);
    return v_min + f * (v_max - v_min);
}

template <class T>
struct is_array_helper_t : std::false_type {};

template <class T, size_t N>
struct is_array_helper_t<std::array<T, N>> : std::true_type {};

template <typename T>
void serialize(const T &t, std::ostream &out);

template <typename T, typename = void>
struct serialize_helper_t {
    static void call(const T &t, std::ostream &out) { t.serialize(out); }
};

template <typename T>
struct serialize_helper_t<T,
        typename std::enable_if<std::is_trivial<T>::value
                && !is_array_helper_t<T>::value>::type> {
    static void call(const T &t, std::ostream &out) {
        out.write((const char *)&t, sizeof(t));
    }
};

template <typename T, size_t N>
struct serialize_helper_t<std::array<T, N>, void> {
    static void call(const std::array<T, N> &a, std::ostream &out) {
        out.write((const char *)&a, sizeof(a));
    }
};

template <typename T>
struct serialize_helper_t<std::vector<T>, void> {
    static void call(const std::vector<T> &v, std::ostream &out) {
        serialize((int)v.size(), out);
        for (auto &t : v)
            serialize(t, out);
    }
};

template <typename KeyT, typename ValueT, typename HashT>
struct serialize_helper_t<std::unordered_map<KeyT, ValueT, HashT>, void> {
    static void call(const std::unordered_map<KeyT, ValueT, HashT> &m,
            std::ostream &out) {
        serialize((int)m.size(), out);
        for (auto &kv : m) {
            serialize(kv.first, out);
            serialize(kv.second, out);
        }
    }
};

template <>
struct serialize_helper_t<std::string, void> {
    static void call(const std::string &s, std::ostream &out) {
        serialize((int)s.size(), out);
        out.write((const char *)s.data(), s.size());
    }
};

template <typename T>
void serialize(const T &t, std::ostream &out) {
    serialize_helper_t<T>::call(t, out);
}

template <typename T>
void serialize(const T &t, const std::string &path) {
    std::ofstream out(path, std::ios::binary);
    serialize(t, out);
}

template <typename T>
T deserialize(std::istream &in);

template <typename T, typename = void>
struct deserialize_helper_t {
    static T call(std::istream &in) {
        T ret;
        ret.deserialize(in);
        return ret;
    }
};

template <typename T>
struct deserialize_helper_t<T,
        typename std::enable_if<std::is_trivial<T>::value
                && !is_array_helper_t<T>::value>::type> {
    static T call(std::istream &in) {
        T ret;
        in.read((char *)&ret, sizeof(ret));
        return ret;
    }
};

template <typename T, size_t N>
struct deserialize_helper_t<std::array<T, N>, void> {
    static std::array<T, N> call(std::istream &in) {
        std::array<T, N> ret;
        in.read((char *)&ret, sizeof(ret));
        return ret;
    }
};

template <typename T>
struct deserialize_helper_t<std::vector<T>, void> {
    static std::vector<T> call(std::istream &in) {
        int size = deserialize<int>(in);
        std::vector<T> ret;
        for (int i = 0; i < size; i++)
            ret.push_back(deserialize<T>(in));
        return ret;
    }
};

template <typename KeyT, typename ValueT, typename HashT>
struct deserialize_helper_t<std::unordered_map<KeyT, ValueT, HashT>, void> {
    static std::unordered_map<KeyT, ValueT, HashT> call(std::istream &in) {
        int size = deserialize<int>(in);
        std::unordered_map<KeyT, ValueT, HashT> ret;
        for (int i = 0; i < size; i++) {
            auto k = deserialize<KeyT>(in);
            auto v = deserialize<ValueT>(in);
            ret.emplace(k, v);
        }
        return ret;
    }
};

template <>
struct deserialize_helper_t<std::string, void> {
    static std::string call(std::istream &in) {
        int size = deserialize<int>(in);
        std::string ret(size, 0);
        in.read((char *)ret.data(), size);
        return ret;
    }
};

template <typename T>
T deserialize(std::istream &in) {
    return deserialize_helper_t<T>::call(in);
}

template <typename T>
void deserialize(T &t, std::istream &in) {
    t = deserialize<T>(in);
}

template <typename T>
void deserialize(T &t, const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        t = T();
        return;
    }
    t = deserialize<T>(in);
}

template <typename T>
void serialize_to_file(const T &t, const std::string &file_name,
        const std::string &var_name,
        const std::vector<std::string> &namespaces = {}) {
    std::ostringstream t_oss;
    serialize(t, t_oss);
    auto str = t_oss.str();
    auto data = std::vector<uint8_t>(str.begin(), str.end());
    std::ostringstream oss;
    oss << "#include <cstdint>\n";
    oss << "#include <vector>\n\n";
    for (auto &ns : namespaces)
        oss << "namespace " << ns << " {\n";
    oss << "\n// clang-format off\n";
    oss << "const std::vector<uint64_t> &get_" << var_name << "() {\n";
    oss << "    static std::vector<uint64_t> data = {\n";
    oss << "        ";
    size_t bytes = data.size();
    size_t u64_bytes = sizeof(uint64_t);
    for (size_t i = 0; i < bytes; i += u64_bytes) {
        uint64_t v = 0;
        size_t len = std::min(bytes - i, u64_bytes);
        std::memcpy(&v, &data[i], len);
        oss << "0x" << std::setfill('0') << std::setw(16) << std::hex << v;
        if (i + u64_bytes < bytes) {
            bool eol = ((i / u64_bytes + 1) % 8 == 0);
            oss << ",";
            oss << (eol ? "\n        " : " ");
        }
    }
    oss << "\n    };";
    oss << "\n    return data;";
    oss << "\n};";
    oss << "\n// clang-format on\n\n";
    for (auto it = namespaces.rbegin(); it != namespaces.rend(); it++)
        oss << "} // namespace " << *it << "\n";
    std::ofstream out(file_name);
    out << oss.str();
}

inline bool str_to_bool(const std::string &s) {
    if (utils::one_of(s, "1", "true", "True")) return true;
    return false;
}

inline int str_to_int(const std::string &s) {
    return std::stoi(s);
}

inline std::string to_string(prop_kind_t prop) {
    switch (prop) {
#define CASE(value, name) \
    case prop_kind::value: return #name
        CASE(undef, undef);
        CASE(forward, fwd);
        CASE(backward_data, bwd_d);
        CASE(backward_weights, bwd_w);
#undef CASE
        default: ir_error_not_expected();
    }
    return {};
}

inline prop_kind_t str_to_prop_kind(const std::string &s) {
#define CASE(value) \
    if (to_string(prop_kind::value) == s) return prop_kind::value
    CASE(forward);
    CASE(backward_data);
    CASE(backward_weights);
#undef CASE
    ir_error_not_expected();
    return prop_kind::undef;
}

inline bool is_big_endian() {
    uint32_t u = 0x01020304;
    uint8_t a[4] = {};
    std::memcpy(a, &u, sizeof(u));
    return a[0] == 0x01;
}

template <typename T, typename U>
T deserialize_from_data(const std::vector<U> &data) {
    if (data.empty()) return T();
    auto *data_str = (const char *)data.data();
    size_t size = data.size() * sizeof(data[0]);
    auto str = std::string(data_str, data_str + size);
    if (is_big_endian()) {
        size_t u_len = sizeof(U);
        for (size_t i = 0; i < str.length(); i += u_len) {
            for (size_t j = 0; j < u_len / 2; j++) {
                std::swap(str[i + j], str[i + u_len - j]);
            }
        }
    }
    std::istringstream iss(str);
    return deserialize<T>(iss);
}

class fast_random_t {
public:
    fast_random_t(int32_t seed = 0) : seed_(seed) {}

    int32_t operator()() {
        seed_ = (1103515245U * seed_ + 12345U) & 0x7fffffff;
        return seed_;
    }

    template <typename T>
    int32_t rand_index(const std::vector<T> &v) {
        return operator()() % (int)v.size();
    }

    template <typename IteratorT>
    void shuffle(IteratorT beg, IteratorT end) {
        int n = (int)(end - beg);
        for (int i = n - 1; i >= 1; i--) {
            int j = operator()() % (i + 1);
            std::swap(*(beg + i), *(beg + j));
        }
    }

private:
    int32_t seed_;
};

template <typename T>
class cli_iface_t {
public:
    using get_func_t = std::string (&)(const T *);
    using set_func_t = void (&)(T *, const std::string &);

    void add_arg(const std::string &key, const std::string &help,
            get_func_t getter, set_func_t setter) {
        args_.emplace_back((int)args_.size(), key, help, getter, setter);
    }

    void parse(const std::string &s, T *obj) {
        bool is_help = (s.find("--help") != std::string::npos);
        if (is_help) {
            for (auto &a : args_) {
                std::cout << "  ";
                std::cout << std::left << std::setw(22) << a.key;
                std::cout << a.help << std::endl;
            }
            return;
        }
        std::vector<bool> seen(args_.size());
        auto parse_part = [&](const std::string &key,
                                  const std::string &value) {
            for (auto &a : args_) {
                if (a.key == key) {
                    if (seen[a.idx]) {
                        std::cout << "Error: argument set twice: " << key
                                  << std::endl;
                        ir_error_not_expected();
                    }
                    a.setter(obj, value);
                    seen[a.idx] = true;
                    return;
                }
            }
            std::cout << "Error: unknown argument: " << key << std::endl;
            ir_error_not_expected();
        };
        std::vector<std::string> parts;
        for (auto &p : gpu_utils::split(s, " ")) {
            if (p.empty()) continue;
            parts.push_back(p);
        }
        int nparts = (int)parts.size();
        for (int i = 0; i < nparts; i += 2) {
            if (i + 1 >= nparts) {
                std::cout << "Error: value is missing for argument: "
                          << parts[i] << std::endl;
                ir_error_not_expected();
            }
            parse_part(parts[i], parts[i + 1]);
        }
    }

    std::string cmd_str(const T *obj) const {
        std::vector<std::string> parts;
        auto add = [&](const std::string &key, const std::string &value) {
            if (value.empty()) return;
            parts.push_back(key);
            parts.push_back(value);
        };
        for (auto &a : args_) {
            add(a.key, a.getter(obj));
        }
        std::ostringstream oss;
        bool is_first = true;
        for (auto &p : parts) {
            if (!is_first) oss << " ";
            oss << p;
            is_first = false;
        }
        return oss.str();
    }

private:
    struct arg_t {
        arg_t(int idx, const std::string &key, const std::string &help,
                get_func_t getter, set_func_t setter)
            : idx(idx), key(key), help(help), getter(getter), setter(setter) {}

        int idx = -1;
        std::string key;
        std::string help;
        get_func_t getter;
        set_func_t setter;
        bool is_parsed = false;
    };

    std::vector<arg_t> args_;
};

inline std::unordered_map<std::string, int> to_string_int_map(
        const std::string &s) {
    std::unordered_map<std::string, int> ret;
    int name_beg = -1;
    int value_beg = -1;
    for (int pos = 0; pos < (int)s.size() + 1; pos++) {
        bool prev_digit = pos > 0 && std::isdigit(s[pos - 1]);
        bool cur_digit = pos < (int)s.size() && std::isdigit(s[pos]);
        if ((pos == 0 || prev_digit) && !cur_digit) {
            if (name_beg != -1 && value_beg != -1) {
                auto key = s.substr(name_beg, value_beg - name_beg);
                auto value = std::stoi(s.substr(value_beg, pos - value_beg));
                ret[key] = value;
            }
            name_beg = pos;
            value_beg = -1;
        }
        if (!prev_digit && cur_digit) value_beg = pos;
    }
    return ret;
}

// Adapted version of magicgu function from Hacker's Delight 10-15.
inline void idiv_magicgu(uint32_t d, uint32_t &m, uint32_t &p) {
    uint32_t s32_max = std::numeric_limits<int32_t>::max();
    ir_assert(d != 0 && d <= s32_max);
    uint64_t nc = (s32_max / d) * d - 1;
    for (p = 32; p < 64; p++) {
        uint64_t _2p = 1LL << p;
        if (_2p > nc * (d - 1 - (_2p - 1) % d)) {
            m = (_2p + d - 1 - (_2p - 1) % d) / d;
            return;
        }
    }
    ir_error_not_expected();
}

inline uint64_t idiv_magicgu_packed(uint32_t d) {
    uint32_t m = 0, p = 0;
    if (math::is_pow2(d)) {
        p = math::ilog2q(d);
    } else {
        ir_utils::idiv_magicgu(d, m, p);
    }
    return m + (static_cast<uint64_t>(p) << 32);
}

} // namespace ir_utils
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
