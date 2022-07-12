/*******************************************************************************
* Copyright 2017-2022 Intel Corporation
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

namespace bnorm {

void check_correctness(const settings_t &s) {
    for_(const auto &i_dir : s.dir)
    for_(const auto &i_dt : s.dt)
    for_(const auto &i_tag : s.tag)
    for_(const auto &i_flags : s.flags)
    for_(const auto &i_mb : s.mb)
    for_(const auto &i_post_ops : s.post_ops)
    for_(const auto &i_scratchpad_mode : s.scratchpad_mode)
    for (auto i_inplace : s.inplace) {
        auto attr = settings_t::get_attr(i_post_ops, i_scratchpad_mode);

        const prb_t prb(s.desc, i_mb, i_dir, i_dt, i_tag, i_flags, i_inplace,
                attr, s.check_alg, s.debug_check_ws);
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
          "for scaleshift.\n    * `C` for scale.\n    * `H` for shift.\n    * "
          "`R` for fuse_norm_relu.\n    * `A` for fuse_norm_add_relu.\n";

static const std::string help_check_alg
        = "CHECK_ALG\n    Dev debug setting to validate output for different "
          "inputs. Overrides driver's automatic choice.\n    `CHECK_ALG` "
          "values are `alg_0` or `alg_1`.\n";

static const std::string help_debug_check_ws
        = "BOOL    (Default: `false`)\n    Instructs the driver to validates "
          "workspace correctness on forward prop kind when set to `true`.\n";

int bench(int argc, char **argv) {
    driver_name = "bnorm";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dir(s.dir, def.dir, argv[0])
                || parse_dt(s.dt, def.dt, argv[0])
                || parse_tag(s.tag, def.tag, argv[0])
                || parse_vector_option(s.flags, def.flags, str2flags, argv[0],
                        "flags", help_flags)
                || parse_single_value_option(s.check_alg, def.check_alg,
                        str2check_alg, argv[0], "check-alg", help_check_alg)
                || parse_inplace(s.inplace, def.inplace, argv[0])
                || parse_mb(s.mb, def.mb, argv[0])
                || parse_single_value_option(s.debug_check_ws,
                        def.debug_check_ws, str2bool, argv[0], "debug-check-ws",
                        help_debug_check_ws)
                || parse_attr_post_ops(s.post_ops, argv[0])
                || parse_attr_scratchpad_mode(
                        s.scratchpad_mode, def.scratchpad_mode, argv[0])
                || parse_test_pattern_match(s.pattern, argv[0])
                || parse_perf_template(s.perf_template, s.perf_template_def,
                        s.perf_template_csv(), argv[0])
                || parse_reset(s, argv[0]) || parse_help(argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            SAFE(str2desc(&s.desc, argv[0]), CRIT);
            check_correctness(s);
        }
    }

    return parse_last_argument();
}

} // namespace bnorm
