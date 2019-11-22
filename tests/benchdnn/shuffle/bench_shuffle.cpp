/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
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

#include "dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "parser.hpp"

#include "shuffle/shuffle.hpp"

namespace shuffle {

std::vector<dir_t> dir {FWD_D};
std::vector<dnnl_data_type_t> dt {dnnl_f32};
std::vector<dnnl_format_tag_t> tag {dnnl_nchw};
std::vector<int64_t> group {1};
std::vector<int> axis {1};

dims_t dims;
bool allow_unimpl = false;
const char *perf_template_csv
        = "perf,%engine%,%dir%,%dt%,%tag%,%group%,%axis%,%DESC%,%-time%,%"
          "0time%";
const char *perf_template_def = "perf,%engine%,%desc%,%-time%,%0time%";
const char *perf_template = perf_template_def;

void reset_parameters() {
    dir = {FWD_D};
    dt = {dnnl_f32};
    tag = {dnnl_nchw};
    group = {1};
    axis = {1};
    allow_unimpl = false;
}

void check_correctness() {
    for_(const auto &i_dir : dir)
    for_(const auto &i_dt : dt)
    for_(const auto &i_tag : tag)
    for_(const auto &i_group : group)
    for (const auto &i_axis : axis) {
        const prb_t p(dims, i_dir, i_dt, i_tag, i_axis, i_group);
        std::stringstream ss;
        ss << p;
        const std::string cpp_pstr = ss.str();
        const char *pstr = cpp_pstr.c_str();
        BENCHDNN_PRINT(1, "run: %s\n", pstr);

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
    driver_name = "shuffle";
    using namespace parser;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = false || parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0]) || parse_dir(dir, argv[0])
                || parse_dt(dt, argv[0]) || parse_tag(tag, argv[0])
                || parse_vector_option(group, atoi, argv[0], "group")
                || parse_axis(axis, argv[0])
                || parse_allow_unimpl(allow_unimpl, argv[0])
                || parse_perf_template(perf_template, perf_template_def,
                        perf_template_csv, argv[0])
                || parse_reset(reset_parameters, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            parse_dims(dims, argv[0]);
            check_correctness();
        }
    }

    return parse_last_argument();
}

} // namespace shuffle
