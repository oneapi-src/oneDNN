/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef _PARSER_HPP
#define _PARSER_HPP

#include <stdlib.h>
#include <stdio.h>
#include <string>

#include "mkldnn.h"
#include "mkldnn_memory.hpp"

namespace parser {

extern bool last_parsed_is_problem;
static const auto eol = std::string::npos;

static inline std::string get_pattern(const std::string &option_name) {
    return std::string("--") + option_name + std::string("=");
}

template <typename T, typename F>
static bool parse_vector_str(T &vec, F process_func, const char *str,
        char delimeter = ',') {
    const std::string s = str;
    vec.clear();
    for (size_t start = 0, delim = 0; delim != eol; start = delim + 1) {
        delim = s.find_first_of(delimeter, start);
        size_t val_len = (delim == eol ? s.size() : delim) - start;
        std::string sub_s(s, start, val_len);
        vec.push_back(process_func(sub_s.c_str()));
    }
    return true;
}

template <typename T, typename F>
static bool parse_multivector_str(std::vector<T> &vec, F process_func,
        const char *str, char vector_delim = ',', char element_delim = ':') {
    auto process_subword = [&](const char *word) {
        T v;
        // parse vector elements separated by @p element_delim
        parse_vector_str(v, process_func, word, element_delim);
        return v;
    };

    // parse full vector separated by @p vector_delim
    return parse_vector_str(vec, process_subword, str, vector_delim);
}

template <typename T, typename F>
static bool parse_vector_option(T &vec, F process_func, const char *str,
        const std::string &option_name) {
    const std::string pattern = get_pattern(option_name);
    if (pattern.find(str, 0, pattern.size()) != eol)
        return parse_vector_str(vec, process_func, str + pattern.size());
    return false;
}

template <typename T, typename F>
static bool parse_multivector_option(std::vector<T> &vec, F process_func,
        const char *str, const std::string &option_name) {
    const std::string pattern = get_pattern(option_name);
    if (pattern.find(str, 0, pattern.size()) != eol)
        return parse_multivector_str(vec, process_func, str + pattern.size());
    return false;
}

template <typename T, typename F>
static bool parse_single_value_option(T &val, F process_func, const char *str,
        const std::string &option_name) {
    const std::string pattern = get_pattern(option_name);
    if (pattern.find(str, 0, pattern.size()) != eol)
        return val = process_func(str + pattern.size()), true;
    return false;
}

bool parse_dir(std::vector<dir_t> &dir, const char *str,
        const std::string &option_name = "dir");

bool parse_dt(std::vector<mkldnn_data_type_t> &dt, const char *str,
        const std::string &option_name = "dt");

bool parse_multi_dt(std::vector<std::vector<mkldnn_data_type_t>> &dt,
        const char *str, const std::string &option_name = "sdt");

bool parse_tag(std::vector<mkldnn_format_tag_t> &tag, const char *str,
        const std::string &option_name = "tag");

bool parse_multi_tag(std::vector<std::vector<mkldnn_format_tag_t>> &tag,
        const char *str, const std::string &option_name = "stag");

bool parse_mb(std::vector<int64_t> &mb, const char *str,
        const std::string &option_name = "mb");

bool parse_attr(attr_t &attr, const char *str,
        const std::string &option_name = "attr");

bool parse_axis(std::vector<int> &axis, const char *str,
        const std::string &option_name = "axis");

bool parse_test_pattern_match(const char *&match, const char *str,
        const std::string &option_name = "match");

bool parse_inplace(std::vector<bool> &inplace, const char *str,
        const std::string &option_name = "inplace");

bool parse_skip_impl(const char *&skip_impl, const char *str,
        const std::string &option_name = "skip-impl");

bool parse_allow_unimpl(bool &allow_unimpl, const char *str,
        const std::string &option_name = "allow-unimpl");

bool parse_perf_template(const char *&pt, const char *pt_def,
        const char *pt_csv, const char *str,
        const std::string &option_name = "perf-template");

bool parse_reset(void (*reset_func)(), const char *str,
        const std::string &option_name = "reset");

bool parse_batch(const bench_f bench, const char *str,
        const std::string &option_name = "batch");

bool parse_bench_settings(const char *str);

void parse_dims(dims_t &dims, const char *str);

void parse_multi_dims(std::vector<dims_t> &dims, const char *str);

void catch_unknown_options(const char *str, const char *driver_name);

int parse_last_argument();
}

#endif
