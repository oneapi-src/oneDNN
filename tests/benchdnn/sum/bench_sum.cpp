/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#include "dnnl_common.hpp"
#include "utils/parser.hpp"

#include "sum/sum.hpp"

namespace sum {

void check_correctness(const settings_t &s) {
    for_(const auto &i_sdt : s.sdt)
    for_(const auto &i_ddt : s.ddt)
    for_(const auto &i_stag : s.stag)
    for_(const auto &i_dtag : s.dtag)
    for_(const auto &i_input_scales : s.input_scales)
    for_(const auto &i_scratchpad_mode : s.scratchpad_mode)
    for_(const auto &i_ctx_init : s.ctx_init)
    for_(const auto &i_ctx_exe : s.ctx_exe)
    for (auto i_inplace : s.inplace) {
        const int n_inputs = static_cast<int>(i_sdt.size());
        const int n_stags = static_cast<int>(i_stag.size());
        if (n_stags != n_inputs && n_stags != 1) {
            BENCHDNN_PRINT(0,
                    "ERROR: Expected number of stag arguments is `1` or `%d`, "
                    "provided `%d`.\n",
                    n_inputs, n_stags);
            SAFE_V(FAIL);
        }

        const int n_input_scales = static_cast<int>(i_input_scales.size());
        if (n_input_scales != n_inputs && n_input_scales != 1) {
            BENCHDNN_PRINT(0,
                    "ERROR: Expected number of scales arguments is `1` or "
                    "`%d`, provided `%d`.\n",
                    n_inputs, n_input_scales);
            SAFE_V(FAIL);
        }

        auto attr = settings_t::get_attr(i_scratchpad_mode);

        const prb_t prb(s.prb_dims, i_sdt, i_ddt, i_stag, i_dtag,
                i_input_scales, i_inplace, attr, i_ctx_init, i_ctx_exe);
        std::stringstream ss;
        ss << prb;
        const std::string cpp_pstr = ss.str();
        const char *pstr = cpp_pstr.c_str();
        BENCHDNN_PRINT(1, "run: %s\n", pstr);

        res_t res {};
        doit(&prb, &res);

        parse_result(res, pstr);

        if (is_bench_mode(PERF)) {
            perf_report_t pr(&prb, s.perf_template);
            pr.report(&res, pstr);
        }
    }
}

static const std::string help_scales
        = "FLOAT[:FLOAT...]    (Default: `1.f`)\n    Input scales for source "
          "values.\n    If a single value is specified, will be broadcasted "
          "for all sources, otherwise number of scales should match number of "
          "inputs.\n";

int bench(int argc, char **argv) {
    driver_name = "sum";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_multi_dt(s.sdt, def.sdt, argv[0])
                || parse_dt(s.ddt, def.ddt, argv[0], "ddt")
                || parse_multi_tag(s.stag, def.stag, argv[0])
                || parse_tag(s.dtag, def.dtag, argv[0], "dtag")
                || parse_multivector_option(s.input_scales, def.input_scales,
                        atof, argv[0], "scales", help_scales)
                || parse_inplace(s.inplace, def.inplace, argv[0])
                || parse_attr_scratchpad_mode(
                        s.scratchpad_mode, def.scratchpad_mode, argv[0])
                || parse_ctx_init(s.ctx_init, def.ctx_init, argv[0])
                || parse_ctx_exe(s.ctx_exe, def.ctx_exe, argv[0])
                || parse_perf_template(s.perf_template, s.perf_template_def,
                        s.perf_template_csv(), argv[0])
                || parse_reset(s, argv[0]) || parse_help(argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            parse_prb_dims(s.prb_dims, argv[0]);
            check_correctness(s);
        }
    }

    return parse_last_argument();
}

} // namespace sum
