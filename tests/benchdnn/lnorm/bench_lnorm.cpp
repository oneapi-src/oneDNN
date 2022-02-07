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
#include <string.h>

#include <sstream>

#include "dnnl_common.hpp"
#include "utils/parser.hpp"

#include "bnorm/bnorm.hpp"
#include "lnorm/lnorm.hpp"

using namespace bnorm;

namespace lnorm {

void check_correctness(const settings_t &s) {
    for_(const auto &i_dir : s.dir)
    for_(const auto &i_dt : s.dt)
    for_(const auto &i_tag : s.tag)
    for_(const auto &i_stat_tag : s.stat_tag)
    for_(const auto &i_flags : s.flags)
    for_(const auto &i_oscale : s.oscale)
    for_(const auto &i_scratchpad_mode : s.scratchpad_mode)
    for_(const auto &i_ctx_init : s.ctx_init)
    for_(const auto &i_ctx_exe : s.ctx_exe)
    for (auto i_inplace : s.inplace) {
        auto attr = settings_t::get_attr(i_oscale, i_scratchpad_mode);

        static constexpr int n_inputs = 2;
        if (i_dt.size() != 1 && i_dt.size() != n_inputs) {
            fprintf(stderr,
                    "ERROR: lnorm driver: `dt` option expects either a single "
                    "input or `%d` inputs in SRC, DST order. Current size is: "
                    "\"%ld\"\n",
                    n_inputs, (long)i_dt.size()),
                    fflush(stderr);
            SAFE_V(FAIL);
        }

        const prb_t prb(s.prb_dims, i_tag, i_stat_tag, i_dir, i_dt, i_flags,
                attr, i_ctx_init, i_ctx_exe, i_inplace, s.check_alg);
        std::stringstream ss;
        ss << prb;
        const std::string cpp_pstr = ss.str();
        const char *pstr = cpp_pstr.c_str();

        if (s.pattern && !match_regex(pstr, s.pattern)) return;
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

static const std::string help_flags
        = "FLAGS    (Default: not specified)\n    Specifies normalization "
          "flags. `FLAGS` values are:\n    * `G` for global_stats.\n    * `S` "
          "for scaleshift.\n    * `C` for scale.\n    * `H` for shift.\n";

int bench(int argc, char **argv) {
    driver_name = "lnorm";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dir(s.dir, def.dir, argv[0])
                || parse_multi_dt(s.dt, def.dt, argv[0], "dt")
                || parse_multi_tag(s.tag, def.tag, argv[0], "tag")
                || parse_tag(s.stat_tag, def.stat_tag, argv[0], "stat_tag")
                || parse_vector_option(s.flags, def.flags, str2flags, argv[0],
                        "flags", help_flags)
                || parse_inplace(s.inplace, def.inplace, argv[0])
                || parse_attr_oscale(s.oscale, argv[0])
                || parse_attr_scratchpad_mode(
                        s.scratchpad_mode, def.scratchpad_mode, argv[0])
                || parse_ctx_init(s.ctx_init, def.ctx_init, argv[0])
                || parse_ctx_exe(s.ctx_exe, def.ctx_exe, argv[0])
                || parse_test_pattern_match(s.pattern, argv[0])
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

} // namespace lnorm
