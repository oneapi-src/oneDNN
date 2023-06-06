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

#include "conv/conv.hpp"
#include "conv/conv_dw_fusion.hpp"

namespace conv {

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
    for_(const auto &i_dt_ : s.dt)
    for_(const auto &i_cfg : s.cfg)
    for_(const auto &i_stag : s.stag)
    for_(const auto &i_wtag : s.wtag)
    for_(const auto &i_dtag : s.dtag)
    for_(const auto &i_alg : s.alg)
    for_(const auto &i_scales : s.scales)
    for_(const auto &i_zero_points : s.zero_points)
    for_(const auto &i_post_ops : s.post_ops)
    for_(const auto &i_scratchpad_mode : s.scratchpad_mode)
    for_(const auto &i_fpmath_mode : s.fpmath_mode)
    for_(const auto &i_ctx_init : s.ctx_init)
    for_(const auto &i_ctx_exe : s.ctx_exe)
    for (const auto &i_mb : s.mb) {
        auto attr = settings_t::get_attr(i_scales, i_zero_points, i_post_ops,
                i_scratchpad_mode, i_fpmath_mode);

        auto i_dt = i_dt_;
        if (!i_cfg.empty() && i_dt.size() == 1 && i_dt[0] == dnnl_f32) {
            handle_legacy_cfg(i_dt, i_cfg);
        }

        const prb_t prb(s.desc, i_dir, i_dt, i_stag, i_wtag, i_dtag, i_alg,
                attr, i_ctx_init, i_ctx_exe, i_mb);
        if (s.pattern && !match_regex(prb.str(), s.pattern)) return;

        bool has_dw_po = attr.post_ops.convolution_index() >= 0;
        auto &conv_createit
                = has_dw_po ? conv_dw_fusion::createit : conv::createit;
        auto &conv_check_cacheit = has_dw_po ? conv_dw_fusion::check_cacheit
                                             : conv::check_cacheit;
        auto &conv_doit = has_dw_po ? conv_dw_fusion::doit : conv::doit;
        task_executor.submit(prb, s.perf_template, conv_createit,
                conv_check_cacheit, conv_doit);
    }
}

int verify_input(const settings_t &s) {
    for_(const auto &i_dt : s.dt)
    for (const auto &i_cfg : s.cfg) {
        if (i_cfg.empty()) continue;

        if (i_dt.size() != 1 || i_dt[0] != dnnl_f32) {
            BENCHDNN_PRINT(0, "%s\n",
                    "ERROR: `dt` and `cfg` knobs are incompatible with each "
                    "other. Specify only one of them at a time.");
            return FAIL;
        }
    }

    static constexpr int n_inputs = 3;
    for (const auto &i_dt : s.dt) {
        if (i_dt.size() != 1 && i_dt.size() != n_inputs) {
            BENCHDNN_PRINT(0, "%s%d.\n",
                    "ERROR: `dt` option expects either a single input or three "
                    "inputs in SRC, WEI, DST order. Current size is: ",
                    static_cast<int>(i_dt.size()));
            return FAIL;
        }
    }

    return OK;
}

int bench(int argc, char **argv) {
    driver_name = "conv";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    driver_task_executor_t task_executor;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dir(s.dir, def.dir, argv[0])
                || parse_multi_dt(s.dt, def.dt, argv[0], "dt")
                || parse_cfg(s.cfg, def.cfg, str2cfg, argv[0])
                || parse_tag(s.stag, def.stag, argv[0], "stag")
                || parse_tag(s.wtag, def.wtag, argv[0], "wtag")
                || parse_tag(s.dtag, def.dtag, argv[0], "dtag")
                || parse_alg(s.alg, def.alg, str2alg, argv[0])
                || parse_mb(s.mb, def.mb, argv[0])
                || parse_attr_scales(s.scales, argv[0])
                || parse_attr_zero_points(s.zero_points, argv[0])
                || parse_attr_post_ops(s.post_ops, argv[0])
                || parse_attr_scratchpad_mode(
                        s.scratchpad_mode, def.scratchpad_mode, argv[0])
                || parse_attr_fpmath_mode(
                        s.fpmath_mode, def.fpmath_mode, argv[0])
                || parse_ctx_init(s.ctx_init, def.ctx_init, argv[0])
                || parse_ctx_exe(s.ctx_exe, def.ctx_exe, argv[0])
                || parse_test_pattern_match(s.pattern, argv[0])
                || parse_perf_template(s.perf_template, s.perf_template_def,
                        s.perf_template_csv(), argv[0])
                || parse_reset(s, argv[0]) || parse_help(argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            SAFE(str2desc(&s.desc, argv[0]), CRIT);

            SAFE(verify_input(s), WARN);
            check_correctness(s, task_executor);
        }
    }

    task_executor.flush();

    return parse_last_argument();
}

} // namespace conv
