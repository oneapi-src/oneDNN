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

#ifndef GPU_INTEL_JIT_UTILS_UTILS_HPP
#define GPU_INTEL_JIT_UTILS_UTILS_HPP

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
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/jit/ngen/ngen.hpp"
#include "gpu/intel/serialization.hpp"
#include "gpu/intel/utils.hpp"

#ifdef DNNL_DEV_MODE
#include "common/profiler.hpp"
#include "common/verbose.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

template <typename T, typename = decltype(std::declval<T>().str(), void())>
inline std::ostream &operator<<(std::ostream &out, const T &obj) {
    out << obj.str();
    return out;
}

namespace ir_utils {

const int LOG_DYNAMIC = -1;
const int LOG_OFF = 0;
const int LOG_FATAL = 50;
const int LOG_WARNING = 100;
const int LOG_SUGGESTION = 120;
const int LOG_INFO = 150;
const int LOG_PERF = 170;
const int LOG_TRACE = 200;
const int LOG_CHECK_DEFAULT = LOG_TRACE;

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

template <typename Key, typename T, typename Compare, typename Allocator>
size_t get_hash(const std::map<Key, T, Compare, Allocator> &m) {
    size_t h = 0;
    for (auto &kv : m) {
        h = hash_combine(h, get_hash(kv.first));
        h = hash_combine(h, get_hash(kv.second));
    }
    return h;
}

template <typename... ArgsT>
size_t get_hash(const ArgsT &...args) {
    size_t h = 0;
    get_hash_impl(h, args...);
    return h;
}

template <size_t idx, typename... ArgsT>
size_t get_tuple_hash(const std::tuple<ArgsT...> &tup) {
    constexpr size_t end = std::tuple_size<std::tuple<ArgsT...>>::value - 1;
    size_t h = get_hash(std::get<idx>(tup));
    if (idx == end) return h;
    return hash_combine(h, get_tuple_hash < idx == end ? idx : idx + 1 > (tup));
}

template <typename... ArgsT>
size_t get_hash(const std::tuple<ArgsT...> &tup) {
    return get_tuple_hash<0>(tup);
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
            && dnnl::impl::gpu::intel::gpu_utils::error_stream_t( \
                    __FILE__, __LINE__, #cond)
#else
#define ir_assert(cond) \
    (false) && !(cond) \
            && dnnl::impl::gpu::intel::gpu_utils::error_stream_t( \
                    __FILE__, __LINE__, #cond)
#endif

#define ir_error_not_expected() ir_assert(false) << "Not expected. "
#define ir_except_not_implemented(msg) throw std::runtime_error(msg)

template <int level, bool value = true, bool add_new_line = false>
class base_logger_t {
public:
    template <int L = level>
    base_logger_t(
            typename std::enable_if<L == LOG_DYNAMIC, int>::type dynamic_level,
            std::ostream &out = std::cout)
        : dynamic_level_(dynamic_level), out_(out) {}
    template <int L = level>
    base_logger_t(
            typename std::enable_if<L != LOG_DYNAMIC, std::ostream &>::type out
            = std::cout)
        : out_(out) {}
    ~base_logger_t() {
        if (add_new_line && !is_first_print_) out_ << std::endl;
#if defined(DNNL_DEV_MODE)
        if (get_level() <= LOG_FATAL) {
            out_ << "Aborting after fatal error..." << std::endl;
            abort();
        }
#endif
    }

    template <int L = level>
    static typename std::enable_if<L != LOG_DYNAMIC, bool>::type is_enabled() {
#if defined(DNNL_DEV_MODE)
        return level <= LOG_FATAL || get_verbose(verbose_t::debuginfo) >= level;
#else
        return false;
#endif
    }

    template <int L = level>
    typename std::enable_if<L != LOG_DYNAMIC, int>::type get_level() const {
        return level;
    }

    template <int L = level>
    typename std::enable_if<L == LOG_DYNAMIC, int>::type get_level() const {
        return dynamic_level_;
    }

    operator bool() const { return value; }

    template <typename T>
    base_logger_t &operator<<(const T &obj) {
        using dnnl::impl::gpu::intel::jit::operator<<;
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

        switch (get_level()) {
            case LOG_FATAL: out_ << "[FATAL] "; break;
            case LOG_WARNING: out_ << "[WARNING] "; break;
            case LOG_SUGGESTION: out_ << "[SUGGESTION] "; break;
            default: break;
        }
        is_first_print_ = false;
    }

    int dynamic_level_ = level;
    std::ostream &out_;
    bool is_first_print_ = true;
};

template <int level>
class logger_t : public base_logger_t<level> {
public:
    logger_t() : base_logger_t<level>() {}
};

class ir_check_log_level_t {
public:
    static int level() { return level_; }
    static bool is_enabled() {
#if defined(DNNL_DEV_MODE)
        switch (level_) {
            case LOG_FATAL: return logger_t<LOG_FATAL>::is_enabled();
            case LOG_TRACE: return logger_t<LOG_TRACE>::is_enabled();
            default: abort();
        }
#else
        return false;
#endif
    }
    ir_check_log_level_t(int new_level) {
        old_level_ = level_;
        level_ = new_level;
    }
    ~ir_check_log_level_t() { level_ = old_level_; }
    ir_check_log_level_t(const ir_check_log_level_t &) = delete;

private:
    static thread_local int level_;
    int old_level_ = LOG_CHECK_DEFAULT;
};

template <bool value = true, bool add_new_line = false>
base_logger_t<LOG_DYNAMIC, value, add_new_line> make_logger(int level) {
    return base_logger_t<LOG_DYNAMIC, value, add_new_line>(level);
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
    return ir_utils::ir_check_log_level_t::is_enabled() \
            && ir_utils::make_logger<false, true>( \
                    ir_utils::ir_check_log_level_t::level())

// This macro enables logging in all nested ir_check() calls. This is useful
// when a check function can be used in scenarios when a failed check is
// expected (regular check) or unexpected (e.g. debug assertion).
// Example 1 (regular check):
//     for (auto &cfg: generate_configs()) {
//         // No logging here.
//         if (!cfg.is_ok()) continue;
//         ...
//     }
//
// Example 2 (debug assertion):
//     auto config = read_from_environment(...);
//     // Detailed logging will show the cause of the failed assertion.
//     ir_assert(ir_check_fatal(config.is_ok()));
#define ir_check_fatal(call) \
    (ir_utils::ir_check_log_level_t(ir_utils::LOG_FATAL), (call))

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
    if (s.empty()) {
        oss << " (empty)";
    } else {
        if (eol) oss << std::endl;
        oss << add_indent(s, "  ", /*skip_first=*/!eol);
    }
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
    debug_profiler_t(const std::string &profile_name)
        : profile(profile_name) {};
    void start() { profile.start(); };
    void stamp(const char *name) { profile.stamp(name); };
    void stop(const char *name) { profile.stop(name); };
    void stop() { profile.stop(); };
    void reset() { profile.reset(); };
    std::string str() const { return profile.str(); };

private:
    profiler_t profile;
#else
    debug_profiler_t(const std::string &) {};
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

inline bool str_to_bool(const std::string &s) {
    if (utils::one_of(s, "1", "true", "True")) return true;
    return false;
}

inline int str_to_int(const std::string &s) {
    return std::stoi(s);
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

inline std::vector<std::pair<std::string, int>> to_string_int_pairs(
        const std::string &s) {
    std::vector<std::pair<std::string, int>> ret;
    int name_beg = -1;
    int value_beg = -1;
    for (int pos = 0; pos < (int)s.size() + 1; pos++) {
        bool prev_digit = pos > 0 && std::isdigit(s[pos - 1]);
        bool cur_digit = pos < (int)s.size() && std::isdigit(s[pos]);
        if ((pos == 0 || prev_digit) && !cur_digit) {
            if (name_beg != -1 && value_beg != -1) {
                auto key = s.substr(name_beg, value_beg - name_beg);
                auto value = std::stoi(s.substr(value_beg, pos - value_beg));
                ret.emplace_back(key, value);
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
            m = into<uint32_t>((_2p + d - 1 - (_2p - 1) % d) / d);
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

// Calculate how many unique filter padding states a conv dimension can produce
// (see conv_post_op_view_mapper_t for more context)
inline dim_t max_unique_pad_states(
        dim_t O, dim_t I, dim_t KD, dim_t P, dim_t S, bool lim) {
    dim_t retn = 1;
    if (I > KD) {
        retn += std::min((O - 1) * S - P, dim_t(0))
                + std::max((O - 1) * S + (KD - P), I) + (P - I);
    } else { // I <= KD, no two states are the same
        retn += (O - 1) * S;
    }
    return (lim) ? std::min(retn, P + std::min(I, KD)) : retn;
}

} // namespace ir_utils

template <typename T>
T stream_parse(std::istream &in) {
    T t;
    in >> t;
    ir_assert(!in.fail());
    return t;
}

template <typename T>
bool stream_try_parse(std::istream &in, T &t) {
    in >> t;
    bool ret = !in.fail();
    in.clear();
    return ret;
}

inline void stream_match(std::istream &in, const std::string &s) {
    in >> std::ws;
    for (auto &c : s) {
        auto next = in.get();
        if (next != c || in.fail())
            ir_error_not_expected() << "Cannot match " << s;
    }
}

inline bool stream_try_match(std::istream &in, const std::string &s) {
    in >> std::ws;
    auto pos = in.tellg();
    bool ok = true;
    for (auto &c : s) {
        if (in.get() != c || in.fail()) {
            ok = false;
            break;
        }
    }
    if (!ok) {
        in.clear();
        in.seekg(pos);
    }
    return ok;
}

template <typename T>
using enum_name_t = std::pair<T, const char *>;

template <typename T>
std::pair<T, const char *> make_enum_name(const T &value, const char *name) {
    return std::make_pair(value, name);
}

template <typename E, size_t N>
std::string to_string_impl(
        E e, const std::array<enum_name_t<E>, N> &enum_names);

template <typename E, size_t N>
void to_enum_templ_impl(const std::string &s, E &e,
        const std::array<enum_name_t<E>, N> &enum_names);

template <typename E, size_t N>
bool is_enum_name_templ_impl(
        const std::string &s, const std::array<enum_name_t<E>, N> &enum_names);

#define GPU_DEFINE_PARSE_ENUM(enum_type, enum_names) \
    inline std::string to_string(enum_type e) { \
        return to_string_impl(e, enum_names); \
    } \
    inline void to_enum_impl(const std::string &s, enum_type &e) { \
        to_enum_templ_impl(s, e, enum_names); \
    } \
    inline bool is_enum_name_impl(const std::string &s, const enum_type *) { \
        return is_enum_name_templ_impl(s, enum_names); \
    }

static auto hw_names = nstl::to_array({
        make_enum_name(ngen::Core::Unknown, "unknown"),
        make_enum_name(ngen::Core::Gen9, "gen9"),
        make_enum_name(ngen::Core::Gen10, "gen10"),
        make_enum_name(ngen::Core::Gen11, "gen11"),
        make_enum_name(ngen::Core::XeLP, "xelp"),
        make_enum_name(ngen::Core::XeHP, "xehp"),
        make_enum_name(ngen::Core::XeHPG, "xehpg"),
        make_enum_name(ngen::Core::XeHPC, "xehpc"),
        make_enum_name(ngen::Core::Xe2, "xe2"),
        make_enum_name(ngen::Core::Xe3, "xe3"),
});
GPU_DEFINE_PARSE_ENUM(ngen::HW, hw_names)

static auto product_family_names = nstl::to_array({
        make_enum_name(ngen::ProductFamily::Unknown, "unknown"),
        make_enum_name(ngen::ProductFamily::GenericGen9, "gen9"),
        make_enum_name(ngen::ProductFamily::GenericGen10, "gen10"),
        make_enum_name(ngen::ProductFamily::GenericGen11, "gen11"),
        make_enum_name(ngen::ProductFamily::GenericXeLP, "xelp"),
        make_enum_name(ngen::ProductFamily::GenericXeHP, "xehp"),
        make_enum_name(ngen::ProductFamily::GenericXeHPG, "xehpg"),
        make_enum_name(ngen::ProductFamily::DG2, "dg2"),
        make_enum_name(ngen::ProductFamily::MTL, "mtl"),
        make_enum_name(ngen::ProductFamily::ARL, "arl"),
        make_enum_name(ngen::ProductFamily::GenericXeHPC, "xehpc"),
        make_enum_name(ngen::ProductFamily::PVC, "pvc"),
        make_enum_name(ngen::ProductFamily::GenericXe2, "xe2"),
        make_enum_name(ngen::ProductFamily::GenericXe3, "xe3"),
});
GPU_DEFINE_PARSE_ENUM(ngen::ProductFamily, product_family_names)

static auto prop_kind_names = nstl::to_array({
        make_enum_name(prop_kind::undef, "undef"),
        make_enum_name(prop_kind::forward, "fwd"),
        make_enum_name(prop_kind::backward_data, "bwd_d"),
        make_enum_name(prop_kind::backward_weights, "bwd_w"),
});
GPU_DEFINE_PARSE_ENUM(prop_kind_t, prop_kind_names)

template <typename T>
class parse_iface_t;

template <typename E>
void parse_enum(std::istream &in, E &e);

template <typename T, typename = void>
struct has_parse_iface_t {
    static const bool value = false;
};

template <typename T>
struct has_parse_iface_t<T, decltype(T::init_parse_iface(nullptr), void())> {
    static const bool value = true;
};

template <typename T, typename = void>
struct has_parse_t {
    static const bool value = false;
};

template <typename T>
struct has_parse_t<T,
        decltype(std::declval<T>().parse(std::declval<std::istream &>()),
                void())> {
    static const bool value = true;
};

template <typename T, typename = void>
struct has_stringify_t {
    static const bool value = false;
};

template <typename T>
struct has_stringify_t<T,
        decltype(std::declval<T>().stringify(std::declval<std::ostream &>()),
                void())> {
    static const bool value = true;
};

template <typename T>
struct parse_iface_helper_t {
    static const parse_iface_t<T> &get() {
        static parse_iface_t<T> _iface = []() {
            parse_iface_t<T> iface;
            T::init_parse_iface(&iface);
            return iface;
        }();
        return _iface;
    }
};

template <typename T>
const parse_iface_t<T> &get_parse_iface() {
    return parse_iface_helper_t<T>::get();
}

template <typename T, typename = void>
struct stringify_impl_t {
    static void call(std::ostream &out, const T &t) { out << t; }
};

template <typename T>
struct stringify_impl_t<T,
        typename std::enable_if<has_parse_iface_t<T>::value>::type> {
    static void call(std::ostream &out, const T &t) {
        get_parse_iface<T>().stringify(out, t);
    }
};

template <typename T>
struct stringify_impl_t<T,
        typename std::enable_if<!has_parse_iface_t<T>::value
                && has_stringify_t<T>::value>::type> {
    static void call(std::ostream &out, const T &t) { t.stringify(out); }
};

template <typename T>
struct stringify_impl_t<T,
        typename std::enable_if<std::is_enum<T>::value>::type> {
    static void call(std::ostream &out, const T &t) { out << to_string(t); }
};

template <typename T, typename = void>
struct parse_impl_t {
    static void call(std::istream &in, T &t) { t = stream_parse<T>(in); }
};

template <typename T>
struct parse_impl_t<T,
        typename std::enable_if<has_parse_iface_t<T>::value>::type> {
    static void call(std::istream &in, T &t) {
        get_parse_iface<T>().parse(in, t);
    }
};

template <typename T>
struct parse_impl_t<T,
        typename std::enable_if<!has_parse_iface_t<T>::value
                && has_parse_t<T>::value>::type> {
    static void call(std::istream &in, T &t) { t.parse(in); }
};

template <typename T>
struct parse_impl_t<T, typename std::enable_if<std::is_enum<T>::value>::type> {
    static void call(std::istream &in, T &t) { parse_enum(in, t); }
};

template <typename T>
void stringify(std::ostream &out, const T &t) {
    stringify_impl_t<T>::call(out, t);
}

template <typename T>
std::string stringify(const T &t) {
    std::ostringstream oss;
    stringify_impl_t<T>::call(oss, t);
    return oss.str();
}

template <typename T>
void parse(std::istream &in, T &t) {
    parse_impl_t<T>::call(in, t);
}

template <typename T>
void parse(const std::string &s, T &t) {
    std::istringstream iss(s);
    parse(iss, t);
}

template <typename T>
T parse(std::istream &in) {
    T t;
    parse(in, t);
    return t;
}

template <typename T>
T parse(const std::string &s) {
    T t;
    parse(s, t);
    return t;
}

template <typename T>
class parse_iface_t {
public:
    using base_type = T;

    struct entry_t {
        std::string name;
        std::string help;
        std::string _default;
        bool required = false;
        std::function<void(std::ostream &, const T &)> stringify;
        std::function<void(std::istream &, T &)> parse;

        bool matches_relaxed(const std::string &_s) const {
            auto s = (_s.find("--") == 0 ? _s.substr(2) : _s);
            if (s.length() != name.length()) return false;
            for (size_t i = 0; i < s.length(); i++) {
                if (s[i] == name[i]) continue;
                if (s[i] == '-' && name[i] == '_') continue;
                return false;
            }
            return true;
        }
    };

    template <typename U, U T::*ptr>
    void add(const std::string &name = {}, const std::string &help = {},
            bool required = false) {
        entry_t e;
        e.name = name;
        e.help = help;
        e._default = jit::stringify(U());
        e.required = required;
        e.stringify = [](std::ostream &out, const T &parent) {
            jit::stringify(out, parent.*ptr);
        };
        e.parse = [](std::istream &in, T &parent) {
            jit::parse(in, parent.*ptr);
        };
        add(e);
    }

    void add(const entry_t &e) {
        if (relaxed_) {
            ir_assert(!e.name.empty())
                    << "Relaxed support requires non-empty name.";
            ir_assert(!e.help.empty())
                    << "Relaxed support requires non-empty help.";
        }
        entries_.push_back(e);
    }

    void set_relaxed(bool value) { relaxed_ = value; }

    template <typename Func>
    void set_pre_stringify_func(const Func &func) {
        pre_stringify_func_ = static_cast<void (*)(const T &)>(func);
    }

    template <typename Func>
    void set_post_parse_func(const Func &func) {
        post_parse_func_ = static_cast<void (*)(T &)>(func);
    }

    void stringify(std::ostream &out, const T &parent, bool cli = false) const {
        if (pre_stringify_func_) pre_stringify_func_(parent);
        bool is_first = true;
        for (auto &e : entries_) {
            std::ostringstream e_oss;
            e.stringify(e_oss, parent);
            if (!e.required && e_oss.str() == e._default) continue;
            if (!is_first) out << " ";
            if (!e.name.empty()) {
                if (cli) {
                    out << "--" << e.name << " ";
                } else {
                    out << e.name << "=";
                }
            }
            out << e_oss.str();
            is_first = false;
        }
    }

    void parse(std::istream &in, T &parent) const {
        parent = T();
        if (relaxed_) {
            parse_relaxed(in, parent);
        } else {
            for (auto &e : entries_) {
                if (!e.name.empty()) {
                    stream_match(in, e.name);
                    stream_match(in, "=");
                }
                e.parse(in, parent);
            }
        }
        if (post_parse_func_) post_parse_func_(parent);
    }

    void parse(const std::string &s, T &parent) const {
        std::istringstream iss(s);
        parse(iss, parent);
    }

    int size() const { return static_cast<int>(entries_.size()); }

    std::string cmd_str(const T &parent) const {
        std::ostringstream oss;
        stringify(oss, parent, /*cli=*/true);
        return oss.str();
    }

    void print_help() const {
        std::ios_base::fmtflags f(std::cout.flags());
        for (auto &e : entries_) {
            std::cout << "  ";
            std::cout << std::left << std::setw(22) << e.name;
            std::cout << e.help << std::endl;
        }
        std::cout.flags(f);
    }

private:
    int find_entry_index(const std::string name) const {
        for (int i = 0; i < (int)entries_.size(); i++) {
            if (entries_[i].matches_relaxed(name)) return i;
        }
        return -1;
    }

    void parse_relaxed(std::istream &in, T &parent) const {
        std::vector<bool> seen(entries_.size());
        while (true) {
            std::string name;
            std::string value;
            if (!try_parse_key_value(in, name, value)) break;
            auto idx = find_entry_index(name);
            ir_assert(idx != -1);
            if (seen[idx]) {
                std::cout << "Error: argument set twice: " << name << std::endl;
                ir_error_not_expected();
                exit(1);
            }
            std::istringstream iss(value);
            entries_[idx].parse(iss, parent);
            seen[idx] = true;
        }
        for (size_t i = 0; i < entries_.size(); i++) {
            if (entries_[i].required && !seen[i]) {
                std::cout << "Error: missing required argument: "
                          << entries_[i].name << std::endl;
                ir_error_not_expected();
                exit(1);
            }
        }
    }

    bool try_parse_key_value(
            std::istream &in, std::string &key, std::string &value) const {
        auto pos0 = in.tellg();
        auto restore = [&]() {
            in.clear();
            in.seekg(pos0);
        };
        std::string s;
        if (!(in >> s)) {
            restore();
            return false;
        }
        if (s == "--help") {
            print_help();
            exit(0);
        }
        auto eq_pos = s.find("=");
        key = (eq_pos != std::string::npos) ? s.substr(0, eq_pos) : s;
        if (find_entry_index(key) == -1) {
            restore();
            return false;
        }
        if (eq_pos != std::string::npos) {
            value = s.substr(eq_pos + 1);
        } else {
            if (!(in >> value)) {
                restore();
                return false;
            }
        }
        return true;
    }

    // Whether to handle relaxed (command-line interface) style parse/stringify
    // when the parameter order is not fixed.
    // Default:  param=value and parameter order is fixed
    // Relaxed: --param value and parameter order is flexible
    bool relaxed_ = false;

    std::vector<entry_t> entries_;
    void (*pre_stringify_func_)(const T &) = nullptr;
    void (*post_parse_func_)(T &) = nullptr;
};

template <typename E, size_t N>
std::string to_string_impl(
        E e, const std::array<enum_name_t<E>, N> &enum_names) {
    for (auto &p : enum_names)
        if (p.first == e) return p.second;
    ir_error_not_expected();
    return {};
}

template <typename E, size_t N>
void to_enum_templ_impl(const std::string &s, E &e,
        const std::array<enum_name_t<E>, N> &enum_names) {
    for (auto &p : enum_names) {
        if (p.second == s) {
            e = p.first;
            return;
        }
    }
    ir_error_not_expected();
}

template <typename E>
E to_enum(const std::string &s) {
    E e;
    to_enum_impl(s, e);
    return e;
}

template <typename E, size_t N>
bool is_enum_name_templ_impl(
        const std::string &s, const std::array<enum_name_t<E>, N> &enum_names) {
    for (auto &p : enum_names) {
        if (p.second == s) return true;
    }
    return false;
}

template <typename E>
bool is_enum_name(const std::string &s) {
    E dummy;
    return is_enum_name_impl(s, &dummy);
}

template <typename E>
void parse_enum(std::istream &in, E &e) {
    std::string name;
    in >> name;
    e = to_enum<E>(name);
}

void stringify_to_cpp_file(const std::string &file_name,
        const std::string &var_name, const std::vector<std::string> &namespaces,
        const std::vector<std::string> &lines);

template <typename T>
std::string serialize_to_hex(const T &t) {
    std::ostringstream oss;
    serialized_data_t s;
    s.append(t);
    for (uint8_t d : s.get_data()) {
        oss << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
            << (int)d;
    }
    return oss.str();
}

template <typename T>
void deserialize_from_hex(T &t, const std::string &s_hex) {
    std::vector<uint8_t> data;
    for (size_t i = 0; i < s_hex.size(); i += 2) {
        data.push_back(static_cast<uint8_t>(
                std::stoi(s_hex.substr(i, 2), nullptr, 16)));
    }
    auto s = serialized_t::from_data(std::move(data));
    deserializer_t d(s);
    d.pop(t);
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
