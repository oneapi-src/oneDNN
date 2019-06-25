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
#include <float.h>
#include <math.h>

#include <sstream>

#include "mkldnn.h"

#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"
#include "parser.hpp"

#include "pool/pool.hpp"

namespace pool {

std::vector<dir_t> dir {FWD_D};
std::vector<const dt_conf_t *> cfg {conf_f32};
std::vector<mkldnn_format_tag_t> tag {mkldnn_nchw};
std::vector<alg_t> alg {MAX};
std::vector<int64_t> mb {0};

const char *skip_impl = "";
bool allow_unimpl = false;
const char *perf_template_csv =
    "perf,%engine%,%name%,%dir%,%cfg%,%tag%,%alg%,%DESC%,%-time%,%0time%";
const char *perf_template_def = "perf,%engine%,%name%,%desc%,%-time%,%0time%";
const char *perf_template = perf_template_def;

void reset_parameters() {
    dir = {FWD_D};
    cfg = {conf_f32};
    mb = {0};
    tag = {mkldnn_nchw};
    alg = {MAX};
    skip_impl = "";
    allow_unimpl = false;
}

void check_correctness(const desc_t *c) {
    for (const auto &i_dir: dir)
    for (const auto &i_cfg: cfg)
    for (const auto &i_tag: tag)
    for (const auto &i_alg: alg)
    for (const auto &i_mb: mb) {
        const prb_t p(*c, i_dir, i_cfg, i_tag, i_alg, i_mb);
        std::stringstream ss;
        ss << p;
        const std::string cpp_pstr = ss.str();
        const char *pstr = cpp_pstr.c_str();
        print(1, "run: %s\n", pstr);

        res_t res{};
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
    using namespace parser;
    for (; argc > 0; --argc, ++argv) {
        if (parse_bench_settings(argv[0]));
        else if (parse_batch(bench, argv[0]));
        else if (parse_tag(tag, argv[0]));
        else if (parse_mb(mb, argv[0]));
        else if (parse_dir(dir, argv[0]));
        else if (parse_vector_option(cfg, str2cfg, argv[0], "cfg"));
        else if (parse_vector_option(alg, str2alg, argv[0], "alg"));
        else if (parse_skip_impl(skip_impl, argv[0]));
        else if (parse_allow_unimpl(allow_unimpl, argv[0]));
        else if (parse_perf_template(perf_template, perf_template_def,
                    perf_template_csv, argv[0]));
        else if (parse_reset(reset_parameters, argv[0]));
        else {
            catch_unknown_options(argv[0], "pool");

            desc_t c;
            SAFE_V(str2desc(&c, argv[0]));
            check_correctness(&c);
        }
    }

    return parse_last_argument();
}

}
