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
#include <stdio.h>

#include <sstream>

#include "mkldnn.h"

#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"
#include "parser.hpp"

#include "sum/sum.hpp"

namespace sum {

std::vector<std::vector<mkldnn_data_type_t>> sdt {{mkldnn_f32, mkldnn_f32}};
std::vector<mkldnn_data_type_t> ddt {mkldnn_f32};
std::vector<std::vector<mkldnn_format_tag_t>> stag {{mkldnn_nchw, mkldnn_nchw}};
std::vector<mkldnn_format_tag_t> dtag {mkldnn_format_tag_undef};
std::vector<std::vector<float>> scales {{0.25}, {1}, {4}};

dims_t dims;
bool allow_unimpl = false;
const char *perf_template_csv =
    "perf,%engine%,%sdt%,%ddt%,%stag%,%dtag%,%DESC%,%-time%,%0time%";
const char *perf_template_def = "perf,%engine%,%desc%,%-time%,%0time%";
const char *perf_template = perf_template_def;

void reset_parameters() {
    sdt = {{mkldnn_f32, mkldnn_f32}};
    ddt = {mkldnn_f32};
    stag = {{mkldnn_nchw, mkldnn_nchw}};
    dtag = {mkldnn_nchw};
    scales = {{0.25}, {1}, {4}};
    allow_unimpl = false;
}

void check_correctness() {
    for (const auto &i_sdt: sdt)
    for (const auto &i_ddt: ddt)
    for (const auto &i_stag: stag)
    for (const auto &i_dtag: dtag) {
        if (i_sdt.size() != i_stag.size()) // expect 1:1 match of dt and tag
            SAFE_V(FAIL);

        for (const auto &i_scales: scales) {
            // expect either single scale value, or 1:1 match of dt and scale
            if (i_scales.size() != 1 && i_scales.size() != i_sdt.size())
                SAFE_V(FAIL);

            const prb_t p(dims, i_sdt, i_ddt, i_stag, i_dtag, i_scales);
            std::stringstream ss;
            ss << p;
            const std::string cpp_pstr = ss.str();
            const char *pstr = cpp_pstr.c_str();
            print(1, "run: %s\n", pstr);

            res_t res{};
            int status = doit(&p, &res);

            bool want_perf_report = false;
            parse_result(res, want_perf_report, allow_unimpl, status, pstr);

            if (want_perf_report && bench_mode & PERF) {
                perf_report_t pr(perf_template);
                pr.report(&p, &res, pstr);
            }

            benchdnn_stat.tests++;
        }
    }
}

int bench(int argc, char **argv) {
    using namespace parser;
    for (; argc > 0; --argc, ++argv) {
        if (parse_bench_settings(argv[0]));
        else if (parse_batch(bench, argv[0]));
        else if (parse_multi_dt(sdt, argv[0]));
        else if (parse_dt(ddt, argv[0], "ddt"));
        else if (parse_multi_tag(stag, argv[0]));
        else if (parse_tag(dtag, argv[0], "dtag"));
        else if (parse_multivector_option(scales, atof, argv[0], "scales"));
        else if (parse_allow_unimpl(allow_unimpl, argv[0]));
        else if (parse_perf_template(perf_template, perf_template_def,
                    perf_template_csv, argv[0]));
        else if (parse_reset(reset_parameters, argv[0]));
        else {
            catch_unknown_options(argv[0], "sum");

            parse_dims(dims, argv[0]);
            check_correctness();
        }
    }

    return parse_last_argument();
}

}
