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

#include "dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "parser.hpp"

#include "conv/conv.hpp"

namespace conv {

std::vector<dir_t> dir {FWD_B};
std::vector<const dt_conf_t *> cfg {conf_f32};
std::vector<dnnl_format_tag_t> stag {dnnl_format_tag_any};
std::vector<dnnl_format_tag_t> wtag {dnnl_format_tag_any};
std::vector<dnnl_format_tag_t> dtag {dnnl_format_tag_any};
std::vector<int64_t> mb {0};

alg_t alg = DIRECT;
attr_t attr;
const char *pattern = NULL;
const char *skip_impl = "";
bool allow_unimpl = false;
const char *perf_template_csv
        = "perf,%engine%,%name%,%dir%,%cfg%,%alg%,%attr%,%DESC%,"
          "%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%";
const char *perf_template_def
        = "perf,%engine%,%name%,%prb%,"
          "%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%";
const char *perf_template = perf_template_def;

void reset_parameters() {
    dir = {FWD_B};
    cfg = {conf_f32};
    stag = {dnnl_format_tag_any};
    wtag = {dnnl_format_tag_any};
    dtag = {dnnl_format_tag_any};
    mb = {0};
    alg = DIRECT;
    attr = attr_t();
    pattern = NULL;
    skip_impl = "";
    allow_unimpl = false;
}

void check_correctness(const desc_t *c) {
    for_(const auto &i_dir : dir)
    for_(const auto &i_cfg : cfg)
    for_(const auto &i_stag : stag)
    for_(const auto &i_wtag : wtag)
    for_(const auto &i_dtag : dtag)
    for (const auto &i_mb : mb) {
        const prb_t p(
                *c, i_dir, i_cfg, i_stag, i_wtag, i_dtag, alg, attr, i_mb);
        std::stringstream ss;
        ss << p;
        const std::string cpp_pstr = ss.str();
        const char *pstr = cpp_pstr.c_str();

        if (pattern && !match_regex(pstr, pattern)) return;
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
    driver_name = "conv";
    using namespace parser;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = false || parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0]) || parse_dir(dir, argv[0])
                || parse_cfg(cfg, str2cfg, argv[0])
                || parse_tag(stag, argv[0], "stag")
                || parse_tag(wtag, argv[0], "wtag")
                || parse_tag(dtag, argv[0], "dtag")
                || parse_single_value_option(alg, str2alg, argv[0], "alg")
                || parse_mb(mb, argv[0]) || parse_attr(attr, argv[0])
                || parse_test_pattern_match(pattern, argv[0])
                || parse_skip_impl(skip_impl, argv[0])
                || parse_allow_unimpl(allow_unimpl, argv[0])
                || parse_perf_template(perf_template, perf_template_def,
                        perf_template_csv, argv[0])
                || parse_reset(reset_parameters, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            desc_t c;
            bool is_deconv = 0;
            SAFE_V(str2desc(&c, argv[0], is_deconv));
            check_correctness(&c);
        }
    }

    return parse_last_argument();
}

} // namespace conv
