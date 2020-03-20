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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sstream>

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "parser.hpp"

#include "bnorm/bnorm.hpp"

namespace bnorm {

void check_correctness(const settings_t &s) {
    for_(const auto &i_dir : s.dir)
    for_(const auto &i_dt : s.dt)
    for_(const auto &i_tag : s.tag)
    for_(const auto &i_flags : s.flags)
    for_(auto i_inplace : s.inplace)
    for (const auto &i_mb : s.mb) {
        const prb_t p(s.desc, i_mb, i_dir, i_dt, i_tag, i_flags, i_inplace,
                s.attr, s.check_alg);
        std::stringstream ss;
        ss << p;
        const std::string cpp_pstr = ss.str();
        const char *pstr = cpp_pstr.c_str();

        if (s.pattern && !match_regex(pstr, s.pattern)) return;
        BENCHDNN_PRINT(1, "run: %s\n", pstr);

        res_t res {};
        const int status = doit(&p, &res);

        bool want_perf_report = false;
        parse_result(res, want_perf_report, s.allow_unimpl, status, pstr);

        if (want_perf_report && bench_mode & PERF) {
            perf_report_t pr(s.perf_template);
            pr.report(&p, &res, pstr);
        }

        benchdnn_stat.tests++;
    }
}

int bench(int argc, char **argv) {
    driver_name = "bnorm";
    using namespace parser;
    static settings_t s;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0]) || parse_dir(s.dir, argv[0])
                || parse_dt(s.dt, argv[0]) || parse_tag(s.tag, argv[0])
                || parse_vector_option(s.flags, str2flags, argv[0], "flags")
                || parse_single_value_option(
                        s.check_alg, str2check_alg, argv[0], "check-alg")
                || parse_inplace(s.inplace, argv[0]) || parse_mb(s.mb, argv[0])
                || parse_attr(s.attr, argv[0])
                || parse_test_pattern_match(s.pattern, argv[0])
                || parse_allow_unimpl(s.allow_unimpl, argv[0])
                || parse_perf_template(s.perf_template, s.perf_template_def,
                        s.perf_template_csv, argv[0])
                || parse_reset(s, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            SAFE_V(str2desc(&s.desc, argv[0]));
            check_correctness(s);
        }
    }

    return parse_last_argument();
}

} // namespace bnorm
