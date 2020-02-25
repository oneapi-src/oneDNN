/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#include "rnn/rnn.hpp"

namespace rnn {

std::vector<dir_t> prop {FWD_D};
std::vector<const dt_conf_t *> cfg {conf_f32};
std::vector<alg_t> alg {VANILLA_RNN};
std::vector<dnnl_rnn_direction_t> direction {dnnl_unidirectional_left2right};
std::vector<activation_t> activation {UNDEF};
std::vector<bool> skip_nonlinear {false};
std::vector<bool> with_peephole {false};
std::vector<int64_t> mb {0};
std::vector<policy_t> scale_policy {policy_t::NONE};

attr_t attr;
bool allow_unimpl = false;
unsigned int flags = 0x0;
float alpha = .9f;
float beta = 0.0f;
const char *perf_template_csv
        = "perf,%engine%,%name%,%prop%,%cfg%,%alg%,%activation%,%direction%,"
          "%DESC%,%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%";
const char *perf_template_def
        = "perf,%engine%,%name%,%prb%,%Gops%,%Gfreq%,%-time%,%-Gflops%,"
          "%0time%,%0Gflops%";
const char *perf_template = perf_template_def;

void reset_parameters() {
    prop = {FWD_D};
    cfg = {conf_f32};
    alg = {VANILLA_RNN};
    direction = {dnnl_unidirectional_left2right};
    activation = {UNDEF};
    skip_nonlinear = {false};
    with_peephole = {false};
    mb = {0};
    attr = attr_t();
    scale_policy = {policy_t::NONE};
    allow_unimpl = false;
    alpha = .9f;
    beta = 0.0f;
}

void check_correctness(const desc_t *c) {
    for_(const auto &i_prop : prop)
    for_(const auto &i_cfg : cfg)
    for_(const auto &i_alg : alg)
    for_(auto i_with_peephole : with_peephole)
    for_(const auto &i_scale_policy : scale_policy)
    for_(const auto &i_direction : direction)
    for_(const auto &i_activation : activation)
    for_(auto i_skip_nonlinear : skip_nonlinear)
    for (const auto &i_mb : mb) {
        if (i_with_peephole && i_alg != VANILLA_LSTM) continue;

        check_case_validity(i_cfg, i_scale_policy);
        dnnl_prop_kind_t prop_kind = prop2prop_kind(i_prop);

        const prb_t p(*c, i_cfg, prop_kind, i_alg, i_with_peephole, i_direction,
                attr, i_scale_policy, flags, i_activation, alpha, beta,
                i_skip_nonlinear, i_mb);
        std::stringstream ss;
        ss << p;
        const std::string cpp_pstr = ss.str();
        const char *pstr = cpp_pstr.c_str();
        print(1, "run: %s\n", pstr);

        res_t res {};
        const int status = doit(p, &res);

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
    driver_name = "rnn";
    using namespace parser;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = false || parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dir(prop, argv[0], "prop")
                || parse_cfg(cfg, str2cfg, argv[0])
                || parse_vector_option(alg, str2alg, argv[0], "alg")
                || parse_vector_option(
                        direction, str2direction, argv[0], "direction")
                || parse_vector_option(
                        activation, str2activation, argv[0], "activation")
                || parse_scale_policy(scale_policy, argv[0])
                || parse_mb(mb, argv[0])
                || parse_skip_nonlinear(skip_nonlinear, argv[0])
                || parse_vector_option(
                        with_peephole, str2bool, argv[0], "with-peephole")
                || parse_attr(attr, argv[0])
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

} // namespace rnn
