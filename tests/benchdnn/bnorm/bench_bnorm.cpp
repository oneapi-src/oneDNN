/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
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

#include "dnnl_common.hpp"
#include "utils/parser.hpp"
#include "utils/task_executor.hpp"

#include "bnorm/bnorm.hpp"

namespace bnorm {

using create_func_t = std::function<int(
        std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &, const prb_t *,
        res_t *)>;
using check_cache_func_t = std::function<int(
        std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &, const prb_t *,
        res_t *)>;
using do_func_t = std::function<int(
        const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &,
        const prb_t *, res_t *)>;
using driver_task_executor_t = task_executor_t<prb_t, perf_report_t,
        create_func_t, check_cache_func_t, do_func_t>;

void check_correctness(
        const settings_t &s, driver_task_executor_t &task_executor) {
    for_(const auto &i_dir : s.dir)
    for_(const auto &i_dt : s.dt)
    for_(const auto &i_tag : s.tag)
    for_(const auto &i_flags : s.flags)
    for_(const auto &i_mb : s.mb)
    for_(const auto &i_post_ops : s.post_ops)
    for_(const auto &i_scratchpad_mode : s.scratchpad_mode)
    for_(const auto &i_ctx_init : s.ctx_init)
    for_(const auto &i_ctx_exe : s.ctx_exe)
    for (auto i_inplace : s.inplace) {
        auto attr = settings_t::get_attr(i_post_ops, i_scratchpad_mode);

        const prb_t prb(s.desc, i_mb, i_dir, i_dt, i_tag, i_flags, i_inplace,
                attr, i_ctx_init, i_ctx_exe, s.check_alg, s.debug_check_ws);
        if (s.pattern && !match_regex(prb.str(), s.pattern)) return;

        task_executor.submit(
                prb, s.perf_template, createit, check_cacheit, doit);
    }
}

static const std::string help_flags
        = "FLAGS    (Default: not specified)\n    Specifies normalization "
          "flags. `FLAGS` values are:\n    * `G` for global_stats.\n    * `C` "
          "for scale.\n    * `H` for shift.\n    * `R` for fuse_norm_relu.\n   "
          " * `A` for fuse_norm_add_relu.\n";

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
    driver_task_executor_t task_executor;
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
                || parse_ctx_init(s.ctx_init, def.ctx_init, argv[0])
                || parse_ctx_exe(s.ctx_exe, def.ctx_exe, argv[0])
                || parse_test_pattern_match(s.pattern, argv[0])
                || parse_perf_template(s.perf_template, s.perf_template_def,
                        s.perf_template_csv(), argv[0])
                || parse_reset(s, argv[0]) || parse_help(argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            SAFE(str2desc(&s.desc, argv[0]), CRIT);
            check_correctness(s, task_executor);
        }
    }

    task_executor.flush();

    return parse_last_argument();
}

} // namespace bnorm
