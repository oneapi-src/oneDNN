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

#ifndef GPU_JIT_CONV_UTILS_HPP
#define GPU_JIT_CONV_UTILS_HPP

#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "common/utils.hpp"
#include "gpu/compute/device_info.hpp"

// Uncomment this when jit::ir debugging is required:
//#define GEN_CONV_DEBUG

// Uncomment this when jit::ir profiling is required:
//#define GEN_CONV_PROFILE

#ifdef GEN_CONV_PROFILE
#include "common/profiler.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace ir_utils {

const int LOG_OFF = 0;
const int LOG_WARNING = 100;
const int LOG_INFO = 150;
const int LOG_PERF = 170;
const int LOG_TRACE = 200;

#ifdef GEN_CONV_DEBUG
const int LOG_LEVEL = LOG_WARNING;
#else
const int LOG_LEVEL = LOG_OFF;
#endif

template <typename T>
size_t get_hash(const T &t);

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

class error_stream_t {
public:
    error_stream_t(const char *file, int line, const char *assert_msg) {
        data_ = new data_t(file, line, assert_msg);
    }

    // This is to be able use a steam object in short-circuit evaluation with
    // booleans, see below.
    operator bool() const { return true; }

    template <typename T>
    error_stream_t &operator<<(const T &t) {
        data_->out << t;
        return *this;
    }

    ~error_stream_t() noexcept(false) {
        if (data_ == nullptr) return;

        auto err = std::runtime_error(data_->out.str());
        delete data_;
        data_ = nullptr;

        // This is techincally unsafe. Since error_stream_t is only used in
        // debug builds and since it is only used by ir_assert() which signals
        // an ill-defined program state, nested throws is not a concern.
        throw err; // NOLINT
    }

private:
    struct data_t {
        data_t(const char *file, int line, const char *assert_msg)
            : file(file), line(line) {
            out << "Assertion " << assert_msg << " failed at " << file << ":"
                << line << std::endl;
        }

        const char *file;
        int line;
        std::ostringstream out;
    };

    data_t *data_;
};

// Checks assertion and, in case of error, evaluates output operators to print
// related messages. Usage:
//     ir_assert(condition) << "Error message" << ...;

#if !defined(NDEBUG) || defined(GEN_CONV_DEBUG)
#define ir_assert(cond) \
    !(cond) \
            && dnnl::impl::gpu::jit::ir_utils::error_stream_t( \
                    __FILE__, __LINE__, #cond)
#else
#define ir_assert(cond) \
    (false) && !(cond) \
            && dnnl::impl::gpu::jit::ir_utils::error_stream_t( \
                    __FILE__, __LINE__, #cond)
#endif

#define ir_error_not_expected() ir_assert(false) << "Not expected. "
#define ir_error_not_implemented() ir_assert(false) << "Not implemented. "

template <int level>
class logger_t {
public:
    logger_t(std::ostream &out = std::cout) : out_(out) {}

    operator bool() const { return true; }

    static bool is_enabled() {
#if defined(GEN_CONV_DEBUG) || defined(GEN_CONV_PROFILE)
        static const int log_level(getenv_int("log_level", LOG_LEVEL));
        return log_level >= level;
#else
        return LOG_LEVEL >= level;
#endif
    }

    template <typename T>
    logger_t &operator<<(const T &obj) {
        maybe_print_header();
        out_ << obj;
        return *this;
    }

    logger_t &operator<<(std::ostream &(*os)(std::ostream &)) {
        maybe_print_header();
        out_ << os;
        return *this;
    }

private:
    void maybe_print_header() {
        if (!is_first_print_) return;

        switch (level) {
            case LOG_WARNING: out_ << "[WARNING] "; break;
            default: break;
        }
        is_first_print_ = false;
    }

    std::ostream &out_;
    bool is_first_print_ = true;
};

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

#define ir_trace() \
    ir_utils::logger_t<ir_utils::LOG_TRACE>::is_enabled() \
            && ir_utils::logger_t<ir_utils::LOG_TRACE>()

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
        auto print = [&](std::ostream &out, size_t idx, const std::string &s) {
            int w = (int)widths[idx] + 2;
            out << std::setw(w);
            out << (idx > 0 ? std::right : std::left);
            out << s;
        };
        oss << title_ << std::endl;
        for (size_t i = 0; i < n; i++) {
            print(oss, i, header_[i]);
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

inline std::ostream &operator<<(std::ostream &out, const table_t &table) {
    out << table.str();
    return out;
}

inline bool getenv_bool(const char *s, bool def) {
    return getenv_int(s, def ? 1 : 0) == 1;
}

inline std::string getenv_str(const char *s, const std::string &def) {
    char buf[1024];
    int ret = getenv(s, buf, sizeof(buf));
    if (ret > 0) return buf;
    return def;
}

// Input is a comma separate list containing gpu_arch and optionally eu_count.
inline compute::gpu_arch_t getenv_gpu(const char *s, compute::gpu_arch_t arch,
        int *eu_count = nullptr, size_t *max_wg_size = nullptr) {
    char buf[1024];
    int ret = getenv(s, buf, sizeof(buf));
    if (ret > 0) {
        char *arch_str = buf, *eu_str = nullptr;
        for (int i = 0; i < ret; i++) {
            if (buf[i] == ',') {
                buf[i] = 0;
                if (i < ret - 1) { eu_str = &buf[i + 1]; }
                break;
            }
        }
        arch = compute::str2gpu_arch(arch_str);
        if (eu_count && eu_str) { *eu_count = atoi(eu_str); }
        if (max_wg_size) {
            // Assume maximum wg size is basically the number of threads
            // available in a subslice with simd_size 16
            const size_t max_eus_per_wg
                    = compute::device_info_t::max_eus_per_wg(arch);
            const size_t simd_size = 16;
            const size_t thr_per_eu = utils::rnd_down_pow2(
                    compute::device_info_t::threads_per_eu(arch));
            *max_wg_size = simd_size * max_eus_per_wg * thr_per_eu;
        }
    }
    return arch;
}

inline std::string to_string(bool b) {
    return b ? "True" : "False";
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

struct debug_profiler_t {
#ifdef GEN_CONV_PROFILE
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

inline std::ostream &operator<<(
        std::ostream &out, const debug_profiler_t &profile) {
    out << profile.str();
    return out;
}

} // namespace ir_utils
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
