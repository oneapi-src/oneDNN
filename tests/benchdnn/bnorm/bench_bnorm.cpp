/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sstream>

#include "dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "dnnl_memory.hpp"
#include "parser.hpp"

#include "bnorm/bnorm.hpp"

namespace bnorm {

std::vector<dir_t> dir {FWD_D};
std::vector<dnnl_data_type_t> dt {dnnl_f32};
std::vector<dnnl_format_tag_t> tag {dnnl_nchw};
std::vector<flags_t> flags {0};
std::vector<int64_t> mb {0};
std::vector<bool> inplace {true};

check_alg_t check_alg = ALG_AUTO;
attr_t attr;
const char *pattern = NULL;
const char *skip_impl = "";
bool allow_unimpl = false;
const char *perf_template_csv
        = "perf,%engine%,%name%,%dir%,%dt%,%tag%,%attr%,%flags%,%DESC%,"
          "%-time%,%0time%";
const char *perf_template_def = "perf,%engine%,%name%,%prb%,%-time%,%0time%";
const char *perf_template = perf_template_def;

void reset_parameters() {
    dir = {FWD_D};
    dt = {dnnl_f32};
    tag = {dnnl_nchw};
    flags = {0};
    mb = {0};
    inplace = {true};
    check_alg = ALG_AUTO;
    attr = attr_t();
    pattern = NULL;
    skip_impl = "";
    allow_unimpl = false;
}

void check_correctness(const desc_t *c) {
    for_(const auto &i_dir : dir)
    for_(const auto &i_dt : dt)
    for_(const auto &i_tag : tag)
    for_(const auto &i_flags : flags)
    for_(const auto &i_inplace : inplace)
    for (const auto &i_mb : mb) {
        const prb_t p(*c, i_mb, i_dir, i_dt, i_tag, i_flags, i_inplace, attr,
                check_alg);
        std::stringstream ss;
        ss << p;
        const std::string cpp_pstr = ss.str();
        const char *pstr = cpp_pstr.c_str();

        if (pattern && !match_regex(pstr, pattern)) return;
        print(1, "run: %s\n", pstr);

        res_t res {};
        const int status = doit(&p, &res);

        bool want_perf_report = false;
        parse_result(res, want_perf_report, allow_unimpl, status, pstr);

        if (want_perf_report && bench_mode & PERF) {
            perf_report_t pr(perf_template);
            pr.report(&p, &res, pstr);
        }

        benchdnn_stat.tests++;
    }
}

int bench(int argc, char **argv) {
    driver_name = "bnorm";
    using namespace parser;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = false || parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0]) || parse_dir(dir, argv[0])
                || parse_dt(dt, argv[0]) || parse_tag(tag, argv[0])
                || parse_vector_option(flags, str2flags, argv[0], "flags")
                || parse_single_value_option(
                        check_alg, str2check_alg, argv[0], "check-alg")
                || parse_inplace(inplace, argv[0]) || parse_mb(mb, argv[0])
                || parse_attr(attr, argv[0])
                || parse_test_pattern_match(pattern, argv[0])
                || parse_skip_impl(skip_impl, argv[0])
                || parse_allow_unimpl(allow_unimpl, argv[0])
                || parse_perf_template(perf_template, perf_template_def,
                        perf_template_csv, argv[0])
                || parse_reset(reset_parameters, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            desc_t c;
            SAFE_V(str2desc(&c, argv[0]));
            check_correctness(&c);
        }
    }

    return parse_last_argument();
}

} // namespace bnorm
