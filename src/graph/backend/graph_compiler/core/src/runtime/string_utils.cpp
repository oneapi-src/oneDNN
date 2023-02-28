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

#include <util/string_utils.hpp>

#include <iomanip>
#include <sstream>

using std::ostringstream;
using std::string;

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace utils {

std::vector<std::string> string_split(
        const std::string &str, const std::string &delimiter) {
    size_t prev = 0;
    std::vector<std::string> ret;
    for (size_t cur = 0; cur < str.length();) {
        cur = str.find_first_of(delimiter, cur);
        if (cur != std::string::npos) {
            ret.push_back(str.substr(prev, cur - prev));
            cur++;
            prev = cur;
        } else {
            ret.push_back(str.substr(prev));
            break;
        }
    }
    return ret;
}

bool string_startswith(const std::string &str, const std::string &prefix) {
    return !str.compare(0, prefix.size(), prefix);
}

bool string_endswith(const std::string &str, const std::string &prefix) {
    return str.size() >= prefix.size()
            && !str.compare(str.size() - prefix.size(), prefix.size(), prefix);
}

std::string replace_newlines(
        const std::string &in_str, const std::string &subst) {
    std::string out_str;
    out_str.reserve(in_str.size() * 2);
    for (const auto &c : in_str) {
        if (c == '\n') {
            out_str.append(subst);
        } else {
            out_str.push_back(c);
        }
    }
    return out_str;
}

std::string brief_pretty_function(std::string text) {
    const size_t paren_idx = text.find('(');
    if (paren_idx == string::npos) {
        // Confused. Just bail out.
        return text;
    }

    const size_t last_space_before_params_idx = text.rfind(' ', paren_idx);

    const size_t long_func_name_start_idx
            = (last_space_before_params_idx == string::npos)
            ? 0
            : (last_space_before_params_idx + 1);

    const size_t last_colon_before_params_idx = text.rfind(':', paren_idx);

    const size_t short_func_name_start_idx
            = (last_colon_before_params_idx == string::npos)
            ? long_func_name_start_idx
            : last_colon_before_params_idx + 1;

    return text.substr(short_func_name_start_idx);
}

std::string brief_lineloc(std::string filename, int64_t line_num) {
    size_t last_slash_idx = filename.rfind('/');
    if (last_slash_idx != string::npos) {
        filename = string(filename, last_slash_idx + 1);
    }

    ostringstream os;
    os << filename << ":" << line_num;
    return os.str();
}

indentation_t::indentation_t(size_t chars_per_level)
    : chars_per_level_(chars_per_level) {}

indentation_t::level_holder_t::level_holder_t(indentation_t &owner)
    : owner_(owner) {
    ++owner_.level_;
}

indentation_t::level_holder_t::~level_holder_t() {
    --owner_.level_;
}

indentation_t::level_holder_t indentation_t::indent() {
    return level_holder_t(*this);
}

std::ostream &operator<<(std::ostream &os, const indentation_t &i) {
    os << std::string(i.level_ * i.chars_per_level_, ' ');
    return os;
};

as_hex_t::as_hex_t(const void *ptr, const char *null_alt_string)
    : ptr_(ptr), null_alt_string_(null_alt_string) {}

std::ostream &operator<<(std::ostream &os, as_hex_t a) {
    if ((!a.ptr_) && (a.null_alt_string_)) {
        os << a.null_alt_string_;
    } else {
        constexpr size_t chars_per_byte = 2;
        constexpr size_t base_chars = 2;
        constexpr size_t total_width
                = base_chars + chars_per_byte * sizeof(void *);
        std::ostringstream ss;
        ss << std::hex << std::internal << std::showbase << std::setfill('0')
           << std::setw(total_width);
        ss << a.ptr_;
        os << ss.str();
    }
    return os;
}

} // namespace utils
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
