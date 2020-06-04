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

void check_correctness(const settings_t &s) {
    for_(const auto &i_prop : s.prop)
    for_(const auto &i_cfg : s.cfg)
    for_(const auto &i_alg : s.alg)
    for_(auto i_with_peephole : s.with_peephole)
    for_(auto i_with_projection : s.with_projection)
    for_(const auto &i_scale_policy : s.scale_policy)
    for_(const auto &i_direction : s.direction)
    for_(const auto &i_activation : s.activation)
    for_(auto i_skip_nonlinear : s.skip_nonlinear)
    for_(auto i_trivial_strides : s.trivial_strides)
    for (const auto &i_mb : s.mb) {
        if (i_with_peephole && i_alg != VANILLA_LSTM) continue;

        if (!(i_scale_policy == policy_t::COMMON
                    || i_scale_policy == policy_t::PER_OC)) {
            std::stringstream ss;
            ss << i_scale_policy;
            const std::string cpp_pstr = ss.str();
            const char *policy_s = cpp_pstr.c_str();
            fprintf(stderr,
                    "ERROR: rnn driver: --scaling=%s is invalid, supported "
                    "values are `common` and `per_oc`.\n",
                    policy_s),
                    fflush(stderr);
            SAFE_V(FAIL);
        }

        const prb_t p(s.desc, dt_conf_t::create(i_cfg), i_prop, i_alg,
                i_with_peephole, i_with_projection, i_direction, s.attr,
                i_scale_policy, s.flags, i_activation, s.alpha, s.beta,
                i_skip_nonlinear, i_trivial_strides, i_mb);
        std::stringstream ss;
        ss << p;
        const std::string cpp_pstr = ss.str();
        const char *pstr = cpp_pstr.c_str();
        BENCHDNN_PRINT(1, "run: %s\n", pstr);

        res_t res {};
        const int status = doit(p, &res);

        bool want_perf_report = false;
        parse_result(res, want_perf_report, status, pstr);

        if (want_perf_report && bench_mode & PERF) {
            perf_report_t pr(s.perf_template);
            pr.report(&p, &res, pstr);
        }

        benchdnn_stat.tests++;
    }
}

int bench(int argc, char **argv) {
    driver_name = "rnn";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    for (; argc > 0; --argc, ++argv) {
        auto cstr2str = [](const char *str) { return std::string(str); };
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dir(s.prop, def.prop, argv[0], "prop")
                || parse_cfg(s.cfg, def.cfg, cstr2str, argv[0])
                || parse_alg(s.alg, def.alg, str2alg, argv[0])
                || parse_vector_option(s.direction, def.direction,
                        str2direction, argv[0], "direction")
                || parse_vector_option(s.activation, def.activation,
                        str2activation, argv[0], "activation")
                || parse_scale_policy(s.scale_policy, def.scale_policy, argv[0])
                || parse_mb(s.mb, def.mb, argv[0])
                || parse_skip_nonlinear(
                        s.skip_nonlinear, def.skip_nonlinear, argv[0])
                || parse_trivial_strides(
                        s.trivial_strides, def.trivial_strides, argv[0])
                || parse_vector_option(s.with_peephole, def.with_peephole,
                        str2bool, argv[0], "with-peephole")
                || parse_vector_option(s.with_projection, def.with_projection,
                        str2bool, argv[0], "with-projection")
                || parse_attr(s.attr, argv[0])
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

} // namespace rnn
