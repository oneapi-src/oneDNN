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

#include "eltwise/eltwise.hpp"

namespace eltwise {

std::vector<dir_t> dir {FWD_D};
std::vector<mkldnn_data_type_t> dt {mkldnn_f32};
std::vector<mkldnn_format_tag_t> tag {mkldnn_nchw};
std::vector<alg_t> alg {attr_t::post_ops_t::RELU};
std::vector<float> scales {0, 0.25, 2};
std::vector<float> alpha {scales};
std::vector<float> beta {scales};
std::vector<int64_t> mb {0};
std::vector<bool> inplace {true};

dims_t dims;
const char *skip_impl = "";
bool allow_unimpl = false;
const char *perf_template_csv =
    "perf,%engine%,%dir%,%dt%,%tag%,%alg%,%DESC%,%-time%,%0time%";
const char *perf_template_def = "perf,%engine%,%desc%,%-time%,%0time%";
const char *perf_template = perf_template_def;

void reset_parameters() {
    dir = {FWD_D};
    dt = {mkldnn_f32};
    tag = {mkldnn_nchw};
    alg = {attr_t::post_ops_t::RELU};
    alpha = scales;
    beta = scales;
    mb = {0};
    inplace = {true};
    skip_impl = "";
    allow_unimpl = false;
}

void check_correctness() {
    for (const auto &i_dir: dir)
    for (const auto &i_dt: dt)
    for (const auto &i_tag: tag)
    for (const auto &i_alg: alg)
    for (const auto &i_alpha: alpha)
    for (const auto &i_beta: beta)
    for (const auto &i_inplace: inplace)
    for (const auto &i_mb: mb) {
        using pk = attr_t::post_ops_t::kind_t;

        // iterator over alpha and beta
        switch (i_alg) {
        case pk::TANH:
        case pk::SQUARE:
        case pk::ABS:
        case pk::SQRT:
        case pk::SRELU:
        case pk::LOGISTIC:
        case pk::EXP:
        case pk::GELU:
            // Skip everything except single alpha and beta value
            if (i_alpha != 0 || i_beta != 0) continue;
        case pk::RELU:
        case pk::ELU:
        case pk::BRELU:
            // Test several alpha values but single beta
            if (i_beta != 0) continue;

        default: ; // Test both alpha and beta
        };

        const prb_t p(dims, i_dir, i_dt, i_tag, i_alg, i_alpha, i_beta,
                i_inplace, i_mb);
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
        else if (parse_dir(dir, argv[0]));
        else if (parse_dt(dt, argv[0]));
        else if (parse_tag(tag, argv[0]));
        else if (parse_mb(mb, argv[0]));
        else if (parse_vector_option(alpha, atof, argv[0], "alpha"));
        else if (parse_vector_option(beta, atof, argv[0], "beta"));
        else if (parse_vector_option(alg, attr_t::post_ops_t::str2kind, argv[0],
                    "alg"));
        else if (parse_inplace(inplace, argv[0]));
        else if (parse_skip_impl(skip_impl, argv[0]));
        else if (parse_allow_unimpl(allow_unimpl, argv[0]));
        else if (parse_perf_template(perf_template, perf_template_def,
                    perf_template_csv, argv[0]));
        else if (parse_reset(reset_parameters, argv[0]));
        else {
            catch_unknown_options(argv[0], "eltwise");

            parse_dims(dims, argv[0]);
            check_correctness();
        }
    }

    return parse_last_argument();
}
}
