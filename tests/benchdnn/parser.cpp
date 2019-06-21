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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string>

#include "mkldnn.h"
#include "mkldnn_memory.hpp"
#include "parser.hpp"

namespace parser {

bool last_parsed_is_problem = false;

bool parse_dir(std::vector<dir_t> &dir, const char *str,
        const std::string &option_name/* = "dir"*/) {
    return parse_vector_option(dir, str2dir, str, option_name);
}

bool parse_dt(std::vector<mkldnn_data_type_t> &dt, const char *str,
        const std::string &option_name/* = "dt"*/) {
    return parse_vector_option(dt, str2dt, str, option_name);
}

bool parse_multi_dt(std::vector<std::vector<mkldnn_data_type_t>> &dt,
        const char *str, const std::string &option_name/* = "sdt"*/) {
    return parse_multivector_option(dt, str2dt, str, option_name);
}

bool parse_tag(std::vector<mkldnn_format_tag_t> &tag, const char *str,
        const std::string &option_name/* = "tag"*/) {
    return parse_vector_option(tag, str2fmt_tag, str, option_name);
}

bool parse_multi_tag(std::vector<std::vector<mkldnn_format_tag_t>> &tag,
        const char *str, const std::string &option_name/* = "stag"*/) {
    return parse_multivector_option(tag, str2fmt_tag, str, option_name);
}

bool parse_mb(std::vector<int64_t> &mb, const char *str,
        const std::string &option_name/* = "mb"*/) {
    return parse_vector_option(mb, atoi, str, option_name);
}

bool parse_attr(attr_t &attr, const char *str,
        const std::string &option_name/* = "attr"*/) {
    const std::string pattern = get_pattern(option_name);
    if (pattern.find(str, 0, pattern.size()) != eol) {
        SAFE_V(str2attr(&attr, str + pattern.size()));
        return true;
    }
    return false;
}

bool parse_axis(std::vector<int> &axis, const char *str,
        const std::string &option_name/* = "axis"*/) {
    return parse_vector_option(axis, atoi, str, option_name);
}

bool parse_test_pattern_match(const char *&match, const char *str,
        const std::string &option_name/* = "match"*/) {
    const std::string pattern = get_pattern(option_name);
    if (pattern.find(str, 0, pattern.size()) != eol) {
        match = str + pattern.size();
        return true;
    }
    return false;
}

bool parse_inplace(std::vector<bool> &inplace, const char *str,
        const std::string &option_name/* = "inplace"*/) {
    return parse_vector_option(inplace, str2bool, str, option_name);
}

bool parse_skip_impl(const char *&skip_impl, const char *str,
        const std::string &option_name/* = "skip-impl"*/) {
    const std::string pattern = get_pattern(option_name);
    if (pattern.find(str, 0, pattern.size()) != eol) {
        skip_impl = str + pattern.size();
        return true;
    }
    return false;
}

bool parse_allow_unimpl(bool &allow_unimpl, const char *str,
        const std::string &option_name/* = "allow-unimpl"*/) {
    return parse_single_value_option(allow_unimpl, str2bool, str,
            option_name);
}

bool parse_perf_template(const char *&pt, const char *pt_def,
        const char *pt_csv, const char *str,
        const std::string &option_name/* = "perf-template"*/) {
    const std::string pattern = get_pattern(option_name);
    if (pattern.find(str, 0, pattern.size()) != eol) {
        const std::string csv_pattern = "csv";
        const std::string def_pattern = "def";
        str += pattern.size();
        if (csv_pattern.find(str, 0, csv_pattern.size()) != eol)
            pt = pt_csv;
        else if (def_pattern.find(str, 0, def_pattern.size()) != eol)
            pt = pt_def;
        else
            pt = str;
        return true;
    }
    return false;
}

bool parse_reset(void (*reset_func)(), const char *str,
        const std::string &option_name/* = "reset"*/) {
    const std::string pattern = get_pattern(option_name);
    if (pattern.find(str, 0, pattern.size() - 1) != eol) {
        reset_func();
        return true;
    }
    return false;
}

bool parse_batch(const bench_f bench, const char *str,
        const std::string &option_name/* = "batch"*/) {
    const std::string pattern = get_pattern(option_name);
    if (pattern.find(str, 0, pattern.size()) != eol) {
        SAFE_V(batch(str + pattern.size(), bench));
        return true;
    }
    return false;
}

/* benchdnn common settings */
static bool parse_bench_mode(const char *str,
        const std::string &option_name = "mode") {
    return parse_single_value_option(bench_mode, str2bench_mode, str,
            option_name);
}

static bool parse_max_ms_per_prb(const char *str,
        const std::string &option_name = "max-ms-per-prb") {
    if (parse_single_value_option(max_ms_per_prb, atof, str, option_name)) {
        if (max_ms_per_prb < 100)
            max_ms_per_prb = 100;
        else if (max_ms_per_prb > 60e3)
            max_ms_per_prb = 60e3;
        return true;
    }
    return false;
}

static bool parse_fix_times_per_prb(
        const char *str, const std::string &option_name = "fix-times-per-prb") {
    if (parse_single_value_option(fix_times_per_prb, atoi, str, option_name)) {
        if (fix_times_per_prb < 0)
            fix_times_per_prb = 0;
        return true;
    }
    return false;
}

static bool parse_verbose(const char *str,
        const std::string &option_name = "verbose") {
    const std::string pattern = "-v"; // check short option first
    if (pattern.find(str, 0, pattern.size()) != eol) {
        verbose = atoi(str + pattern.size());
        return true;
    }
    return parse_single_value_option(verbose, atoi, str, option_name);
}

static bool parse_engine_kind(const char *str,
        const std::string &option_name = "engine") {
    if (parse_single_value_option(
                engine_tgt_kind, str2engine_kind, str, option_name)) {

        DNN_SAFE(mkldnn_stream_destroy(stream_tgt), CRIT);
        DNN_SAFE(mkldnn_engine_destroy(engine_tgt), CRIT);

        DNN_SAFE(mkldnn_engine_create(&engine_tgt, engine_tgt_kind, 0), CRIT);
        DNN_SAFE(mkldnn_stream_create(
                         &stream_tgt, engine_tgt, mkldnn_stream_default_flags),
                CRIT);
        return true;
    }
    return false;
}

bool parse_bench_settings(const char *str) {
    last_parsed_is_problem = false; // if start parsing, expect an option

    if (parse_bench_mode(str));
    else if (parse_max_ms_per_prb(str));
    else if (parse_fix_times_per_prb(str));
    else if (parse_verbose(str));
    else if (parse_engine_kind(str));
    else
        return false;
    return true;
}

void parse_dims(dims_t &dims, const char *str) {
    parse_vector_str(dims, atoi, str, 'x');
}

void parse_multi_dims(std::vector<dims_t> &dims, const char *str) {
    parse_multivector_str(dims, atoi, str, ':', 'x');
}

/* utilities */
void catch_unknown_options(const char *str, const char *driver_name) {
    last_parsed_is_problem = true; // if reached, means problem parsing

    const std::string pattern = "--";
    if (pattern.find(str, 0, pattern.size()) != eol) {
        fprintf(stderr, "%s driver: unknown option: `%s`, exiting...\n",
                driver_name, str);
        exit(2);
    }
}

int parse_last_argument() {
    if (!last_parsed_is_problem)
        fprintf(stderr, "WARNING: No problem found for a given option!\n");
    return OK;
}
}
