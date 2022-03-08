/*******************************************************************************
* Copyright 2022 Intel Corporation
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
#include <string>

#include "mlp/mlp.hpp"
#include "utils/parser.hpp"

namespace mlp {

void check_correctness(const settings_t &s) {
    settings_t def;

    for_(const auto &i_wtag : s.wtag)
    for_(const auto &i_dtag : s.dtag)
    for_(const auto &i_biadt : s.bia_dt)
    for_(const auto &i_scale : s.scales)
    for_(const auto &i_zps : s.zero_points)
    for_(const auto &i_training : s.is_training)
    for (const auto &i_cfg : s.cfg) {
        const mlp_graph_spec_t spec(s.prb_dims, i_wtag, i_dtag, i_biadt, i_cfg,
                s.actfunc, i_scale, i_zps, i_training);
        std::stringstream ss;
        ss << spec;
        const std::string cpp_pstr = ss.str();
        const char *pstr = cpp_pstr.c_str();
        BENCHDNN_PRINT(1, "run: %s\n", pstr);
        res_t res {};
        const int status = doit(&spec, &res);
        bool want_perf_report = false;
        parse_result(res, want_perf_report, status, pstr);
        //Due to multiple partition below perf_report needs update
        want_perf_report = false;
        if (want_perf_report && is_bench_mode(PERF)) {
            perf_report_t pr(&spec, s.perf_template);
            pr.report(&res, pstr);
        }
        benchdnn_stat.tests++;
    }
}

int bench(int argc, char **argv) {
    using namespace parser;

    static settings_t s;
    static const settings_t def {};

    driver_name = "mlp";

    for (; argc > 0; --argc, ++argv) {
        std::vector<attr_t::post_ops_t> post_ops {};
        const bool parsed_options = parse_batch(bench, argv[0])
                || parse_cfg(s.cfg, def.cfg, str2cfg, argv[0])
                || parse_inplace(
                        s.is_training, def.is_training, argv[0], "training")
                || parse_dt(s.bia_dt, def.bia_dt, argv[0], "bia_dt")
                || parse_tag(s.dtag, def.dtag, argv[0], "datatag")
                || parse_tag(s.wtag, def.wtag, argv[0], "wtag")
                || parse_attr_post_ops(s.actfunc, argv[0], "actfunc")
                || parse_attr_oscale(s.scales, argv[0], "attr-oscale")
                || parse_attr_zero_points(
                        s.zero_points, argv[0], "attr-zero-points")
                || parse_reset(s, argv[0])
                || parse_perf_template(s.perf_template, s.perf_template_def,
                        s.perf_template_csv, argv[0])
                || parse_help(argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);
            parse_prb_dims(s.prb_dims, argv[0]);
            check_correctness(s);
        }
    }
    return parse_last_argument();
}
} // namespace mlp
