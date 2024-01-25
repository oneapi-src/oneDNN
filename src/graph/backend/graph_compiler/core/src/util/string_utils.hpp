/*******************************************************************************
 * Copyright 2020-2024 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_STRING_UTILS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_STRING_UTILS_HPP

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace utils {

// Split the string into an array of substrings by the delimiter. The delimiters
// will not occur in the resulting substrings
std::vector<std::string> string_split(
        const std::string &str, const std::string &delimiter);

// returns true if the str starts with prefix
bool string_startswith(const std::string &str, const std::string &prefix);
// returns true if the str ends with prefix
bool string_endswith(const std::string &str, const std::string &prefix);

/// Return a new string in which every occurrence of '\n' in @p in_str
/// has been replaced with \p subst.
std::string replace_newlines(
        const std::string &in_str, const std::string &subst);

/// Assume that \p text is the string provided by an invocation of
/// __PRETTY_FUNCTION__. Attempt to strip away any namespace and/or class names
/// at the beginning, and the parameter list at the end. If successful, return
/// the resulting string. Otherwise just return the original string.
std::string brief_pretty_function(std::string text);

/// Assume that \p filename is provided by an invocation of __FILE__, and
/// \p line_num is provided by an invocation of __LINE__.
/// Return the string "(filename-stripped-of-leading-dirnames):linenum".
std::string brief_lineloc(std::string filename, int64_t line_num);

/**
 * Class to simplify the work of incrementally increasing or decreasing
 * the indentation of text lines during logging.
 *
 * Uses RAII to decrement indentation levels automatically.
 *
 * Example of suggested usage:
 *
 *         class my_visitor : public ir_visitor_t {
 *             private:
 *                 indentation_t ind_;
 *
 *                 virtual void view(cast_c v) {
 *                     cout << ind_ << "HANDLING A CAST NODE" << endl;
 *                     {
 *                         // Increase the indentation used when visiting my
 *                         // descdendent nodes.
 *                         ind_.level_holder_t h;
 *
 *                         dispatch(v->in_);
 *                     }
 *                 }
 *         };
 *
 * To make using this class even more convenient, consider defining a
 * preprocessor macro similar to this:
 *
 *         #define INDENT ind_.level_holder_t h;
 */
class indentation_t {
public:
    indentation_t(size_t chars_per_level = 2);

    class level_holder_t {
    public:
        level_holder_t(indentation_t &owner);
        ~level_holder_t();

    private:
        indentation_t &owner_;
    };

    level_holder_t indent();

private:
    friend std::ostream &operator<<(std::ostream &os, const indentation_t &i);
    size_t chars_per_level_;
    size_t level_ = 0;
};

std::ostream &operator<<(std::ostream &os, const indentation_t &i);

/**
 * Wraps a C++ pointer value so tht it's rendered as a 64-bit
 * hexadecimal value (with leading "0x") OR as the specified
 * alternative string.
 *
 * \param ptr The address to be rendered.
 *
 * \param null_alt_string Governs what text is produced if \p ptr
 *   is null. When \p ptr is null:
 *   - If <i>this</i> parameter is also null, produce "0x00000000".
 *   - Otherwise produce the string pointed to by this parameter.
 *
 * Example:
 *     void* p = malloc(10);
 *     cout << "p = " << as_hex_t(p) << endl;
 *     cout << "p = " << as_hex_t(p, "(null)") endl;
 */
class as_hex_t {
public:
    as_hex_t(const void *ptr, const char *null_alt_string = nullptr);

private:
    friend std::ostream &operator<<(std::ostream &os, as_hex_t a);
    const void *ptr_;
    const char *null_alt_string_;
};

std::ostream &operator<<(std::ostream &os, as_hex_t a);

template <typename T>
void general_print(std::ostream &os, const std::vector<T> &container);
template <typename T1, typename T2>
void general_print(std::ostream &os, const std::pair<T1, T2> &value);

template <typename T>
void general_print(std::ostream &os, const T &value) {
    os << value;
}
template <typename T1, typename T2>
void general_print(std::ostream &os, const std::pair<T1, T2> &value) {
    os << '{';
    general_print(os, value.first);
    os << ": ";
    general_print(os, value.second);
    os << '}';
}

template <typename T>
void general_print(std::ostream &os, const std::vector<T> &container) {
    os << '[';
    bool first = true;
    for (const auto &element : container) {
        if (!first) {
            os << ", ";
        } else {
            first = false;
        }
        general_print(os, element);
    }
    os << ']';
}

template <typename T>
std::string general_print(const T &value) {
    std::stringstream ss;
    general_print(ss, value);
    return ss.str();
}

template <typename T>
std::string print_vector(const T &vec) {
    return general_print(vec);
}

template <typename T>
std::string print_pair_vector(const std::vector<std::pair<T, T>> &pvec) {
    return general_print(pvec);
}

template <typename T>
std::string print_nested_vector(const std::vector<std::vector<T>> &nested_vec) {
    return general_print(nested_vec);
}

} // namespace utils
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
