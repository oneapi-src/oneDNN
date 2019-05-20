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

static const auto eol = std::string::npos;

template <typename T, typename F>
static bool parse_vector_str(T &vec, F process_func, const char *str) {
    const std::string s = str;
    vec.clear();
    for (size_t start = 0, comma = 0; comma != eol; start = comma + 1) {
        comma = s.find_first_of(',', start);
        size_t val_len = (comma == eol ? s.size() : comma) - start;
        assert(val_len < 32);
        char value[32] = "";
        s.copy(value, val_len, start);
        vec.push_back(process_func(value));
    }
    return true;
}

template <typename T, typename F>
static bool parse_vector_option(T &vec, F process_func, const char *str,
        const std::string &option_name) {
    const std::string pattern = "--" + option_name + "=";
    if (pattern.find(str, 0, pattern.size()) != eol)
        return parse_vector_str(vec, process_func, str + pattern.size());
    return false;
}

template <typename T, typename F>
static bool parse_single_value_option(T &val, F process_func, const char *str,
        const std::string &option_name) {
    const std::string pattern = "--" + option_name + "=";
    if (pattern.find(str, 0, pattern.size()) != eol)
        return val = process_func(str + pattern.size()), true;
    return false;
}

bool parse_dir(std::vector<dir_t> &dir, const char *str,
        const std::string &option_name = "dir");

bool parse_dt(std::vector<mkldnn_data_type_t> &dt, const char *str,
        const std::string &option_name = "dt");

bool parse_tag(std::vector<mkldnn_format_tag_t> &tag, const char *str,
        const std::string &option_name = "tag");

bool parse_mb(std::vector<int64_t> &mb, const char *str,
        const std::string &option_name = "mb");

bool parse_attr(attr_t &attr, const char *str,
        const std::string &option_name = "attr");

bool parse_axis(std::vector<int> &axis, const char *str,
        const std::string &option_name = "axis");

bool parse_test_pattern_match(const char *&match, const char *str,
        const std::string &option_name = "match");

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

void catch_unknown_options(const char *str, const char *driver_name);

}

#endif
