/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "common.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/parser.hpp"

#include "mha/mha.hpp"

namespace mha {

void check_correctness(const settings_t &s) {
    settings_t def;
    for_(const auto &i_dt : s.dt)
    for_(const auto &i_qoscale : s.quan_oscale)
    for_(const auto &i_dqoscale : s.dequan_oscale)
    for_(const auto &i_qzero_points : s.quan_zero_points)
    for_(const auto &i_dqzero_points : s.dequan_zero_points)
    for (const auto &i_head : s.heads) {
        attr_t quan_attr, dequan_attr;
        quan_attr.insert(i_qoscale);
        quan_attr.insert(i_qzero_points);
        dequan_attr.insert(i_dqoscale);
        dequan_attr.insert(i_dqzero_points);

        std::vector<float> qattr_scale = {quan_attr.oscale.scale};
        auto &qscale = quan_attr.oscale.scale == 0 ? s.def_scale : qattr_scale;
        std::vector<float> dqattr_scale = {dequan_attr.oscale.scale};
        auto &dqscale
                = dequan_attr.oscale.scale == 0 ? s.def_scale : dqattr_scale;
        for_(const auto &i_qscale : qscale)
        for (const auto &i_dqscale : dqscale) {
            std::string pattern = s.pattern == nullptr ? std::string()
                                                       : std::string(s.pattern);
            const mha_graph_spec_t spec(pattern, s.prb_dims.dims,
                    s.prb_dims.ndims, i_head, i_dt, quan_attr, dequan_attr,
                    i_qscale, i_dqscale);
            std::stringstream ss;
            ss << spec;
            const std::string cpp_pstr = ss.str();
            const char *pstr = cpp_pstr.c_str();
            BENCHDNN_PRINT(1, "run: %s\n", pstr);
            res_t res {};
            const int status = doit(&spec, &res);
            bool want_perf_report = false;
            parse_result(res, want_perf_report, status, pstr);

            if (want_perf_report && is_bench_mode(PERF)) {
                perf_report_t pr(&spec, s.perf_template);
                pr.report(&res, pstr);
            }

            benchdnn_stat.tests++;
        }
    }
}

int bench(int argc, char **argv) {
    using namespace parser;

    static settings_t s;
    static const settings_t def {};

    driver_name = "mha";

    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_batch(bench, argv[0])
                || parse_test_pattern_match(s.pattern, argv[0], "pattern")
                || parse_axis(s.heads, def.heads, argv[0], "head")
                || parse_dt(s.dt, def.dt, argv[0]) || parse_reset(s, argv[0])
                || parse_attr_oscale(s.quan_oscale, argv[0], "attr-quan-oscale")
                || parse_attr_oscale(
                        s.dequan_oscale, argv[0], "attr-dequan-oscale")
                || parse_attr_zero_points(
                        s.quan_zero_points, argv[0], "attr-quan-zero-points")
                || parse_attr_zero_points(s.dequan_zero_points, argv[0],
                        "attr-dequan-zero-points")
                || parse_perf_template(s.perf_template, s.perf_template_def,
                        s.perf_template_csv, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);
            parse_prb_dims(s.prb_dims, argv[0]);
            check_correctness(s);
        }
    }
    return parse_last_argument();
}

} // namespace mha
