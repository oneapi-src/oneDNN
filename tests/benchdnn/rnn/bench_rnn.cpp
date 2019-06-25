/*******************************************************************************
 * Copyright 2018 Intel Corporation
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

#include "mkldnn.h"

#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"
#include "parser.hpp"

#include "rnn/rnn.hpp"

namespace rnn {

std::vector<dir_t> prop {FWD_D};
std::vector<const dt_conf_t *> cfg {conf_f32};
std::vector<alg_t> alg {VANILLA_RNN};
std::vector<mkldnn_rnn_direction_t> direction {mkldnn_unidirectional_left2right};
std::vector<activation_t> activation {RELU};
std::vector<int64_t> mb {0};

attr_t attr;
policy_t scale_policy = NONE;
bool allow_unimpl = false;
unsigned int flags = 0x0;
float alpha = 0.0f;
float beta = 0.0f;
const char *perf_template_csv =
    "perf,%engine%,%name%,%prop%,%DESC%,"
    "%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%";
const char *perf_template_def = perf_template_csv;
const char *perf_template = perf_template_def;

void reset_parameters() {
    prop = {FWD_D};
    cfg = {conf_f32};
    alg = {VANILLA_RNN};
    direction = {mkldnn_unidirectional_left2right};
    activation = {RELU};
    mb = {0};
    attr = attr_t();
    scale_policy = NONE;
    allow_unimpl = false;
}

void check_correctness(const desc_t *c) {
    for (const auto &i_prop: prop)
    for (const auto &i_cfg: cfg)
    for (const auto &i_alg: alg)
    for (const auto &i_direction: direction)
    for (const auto &i_activation: activation)
    for (const auto &i_mb: mb) {
        check_case_validity(i_cfg, scale_policy);
        mkldnn_prop_kind_t prop_kind = prop2prop_kind(i_prop);

        const prb_t p(*c, i_cfg, prop_kind, i_alg, i_direction, attr, scale_policy,
                flags, i_activation, alpha, beta, i_mb);
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
        else if (parse_dir(prop, argv[0], "prop"));
        else if (parse_vector_option(cfg, str2cfg, argv[0], "cfg"));
        else if (parse_vector_option(alg, str2alg, argv[0], "alg"));
        else if (parse_vector_option(direction, str2direction, argv[0],
                    "direction"));
        else if (parse_vector_option(activation, str2activation, argv[0],
                    "activation"));
        else if (parse_mb(mb, argv[0]));
        else if (parse_attr(attr, argv[0]));
        else if (parse_single_value_option(scale_policy, str2policy, argv[0],
                    "scaling"));
        else if (parse_allow_unimpl(allow_unimpl, argv[0]));
        else if (parse_perf_template(perf_template, perf_template_def,
                    perf_template_csv, argv[0]));
        else if (parse_reset(reset_parameters, argv[0]));
        else {
            catch_unknown_options(argv[0], "rnn");

            desc_t c;
            SAFE_V(str2desc(&c, argv[0]));
            check_correctness(&c);
        }
    }

    return parse_last_argument();
}

} // namespace rnn
