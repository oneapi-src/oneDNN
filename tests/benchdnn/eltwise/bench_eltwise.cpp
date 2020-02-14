/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include <sstream>

#include "dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "parser.hpp"

#include "eltwise/eltwise.hpp"

namespace eltwise {

std::vector<dir_t> dir {FWD_D};
std::vector<dnnl_data_type_t> dt {dnnl_f32};
std::vector<dnnl_format_tag_t> tag {dnnl_nchw};
std::vector<alg_t> alg {attr_t::post_ops_t::RELU};
std::vector<float> scales {0, 0.25, -0.25};
std::vector<float> alpha {scales};
std::vector<float> beta {scales};
std::vector<int64_t> mb {0};
std::vector<bool> inplace {true};

dims_t dims;
const char *skip_impl = "";
bool allow_unimpl = false;
const char *perf_template_csv
        = "perf,%engine%,%dir%,%dt%,%tag%,%alg%,%DESC%,%-time%,%0time%";
const char *perf_template_def = "perf,%engine%,%prb%,%-time%,%0time%";
const char *perf_template = perf_template_def;

void reset_parameters() {
    dir = {FWD_D};
    dt = {dnnl_f32};
    tag = {dnnl_nchw};
    alg = {attr_t::post_ops_t::RELU};
    alpha = scales;
    beta = scales;
    mb = {0};
    inplace = {true};
    skip_impl = "";
    allow_unimpl = false;
}

void check_correctness() {
    for_(const auto &i_dir : dir)
    for_(const auto &i_dt : dt)
    for_(const auto &i_tag : tag)
    for_(const auto &i_alg : alg)
    for_(const auto &i_alpha : alpha)
    for_(const auto &i_beta : beta)
    for_(auto i_inplace : inplace)
    for (const auto &i_mb : mb) {
        // iterator over alpha and beta (alphabetic order!)
        switch (i_alg) {
            case alg_t::ABS:
            case alg_t::EXP:
            case alg_t::EXP_DST:
            case alg_t::GELU_TANH:
            case alg_t::LOG:
            case alg_t::LOGISTIC:
            case alg_t::LOGISTIC_DST:
            case alg_t::SQRT:
            case alg_t::SQRT_DST:
            case alg_t::SQUARE:
            case alg_t::SRELU:
            case alg_t::TANH:
            case alg_t::TANH_DST:
            case alg_t::GELU_ERF:
                // Skip everything but alpha = 0 and beta = 0
                if (i_alpha != 0 || i_beta != 0) continue;
            case alg_t::ELU:
            case alg_t::RELU:
            case alg_t::SWISH:
                // Test any alpha value but beta = 0
                if (i_beta != 0) continue;
            case alg_t::ELU_DST:
            case alg_t::RELU_DST:
            case alg_t::BRELU:
                // Test non-negative alpha value but beta = 0
                if (i_alpha < 0 || i_beta != 0) continue;
            case alg_t::CLIP:
                // Test beta >= alpha values
                if (i_beta < i_alpha) continue;

            default:; // Test any alpha and beta values
        };

        const prb_t p(dims, i_dir, i_dt, i_tag, i_alg, i_alpha, i_beta,
                i_inplace, i_mb);
        std::stringstream ss;
        ss << p;
        const std::string cpp_pstr = ss.str();
        const char *pstr = cpp_pstr.c_str();
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
    driver_name = "eltwise";
    using namespace parser;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = false || parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0]) || parse_dir(dir, argv[0])
                || parse_dt(dt, argv[0]) || parse_tag(tag, argv[0])
                || parse_vector_option(alpha, atof, argv[0], "alpha")
                || parse_vector_option(beta, atof, argv[0], "beta")
                || parse_vector_option(
                        alg, attr_t::post_ops_t::str2kind, argv[0], "alg")
                || parse_inplace(inplace, argv[0]) || parse_mb(mb, argv[0])
                || parse_skip_impl(skip_impl, argv[0])
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
} // namespace eltwise
