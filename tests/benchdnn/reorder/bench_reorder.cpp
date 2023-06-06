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

#include <string.h>

#include "dnnl_common.hpp"
#include "utils/parser.hpp"
#include "utils/task_executor.hpp"

#include "reorder.hpp"

namespace reorder {

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
    for_(const auto &i_sdt : s.sdt)
    for_(const auto &i_ddt : s.ddt)
    for_(const auto &i_stag : s.stag)
    for_(const auto &i_dtag : s.dtag)
    for_(const auto &i_oflag : s.oflag)
    for_(const auto &i_cross_engine : s.cross_engine)
    for_(const auto &i_scales : s.scales)
    for_(const auto &i_zero_points : s.zero_points)
    for_(const auto &i_post_ops : s.post_ops)
    for_(const auto &i_scratchpad_mode : s.scratchpad_mode)
    for_(const auto &i_ctx_init : s.ctx_init)
    for_(const auto &i_ctx_exe : s.ctx_exe)
    for (auto i_runtime_dim_mask : s.runtime_dim_mask) {
        const auto &src_scale = i_scales.get(DNNL_ARG_SRC);
        const std::vector<float> src_test_scales = src_scale.scale == 0
                ? s.def_scale
                : std::vector<float>(1, src_scale.scale);
        const auto &dst_scale = i_scales.get(DNNL_ARG_DST);
        const std::vector<float> dst_test_scales = dst_scale.scale == 0
                ? s.def_scale
                : std::vector<float>(1, dst_scale.scale);

        for_(const auto &i_src_test_scale : src_test_scales)
        for (const auto &i_dst_test_scale : dst_test_scales) {
            attr_t::arg_scales_t test_arg_scales;
            test_arg_scales.set(
                    DNNL_ARG_SRC, {src_scale.policy, i_src_test_scale});
            test_arg_scales.set(
                    DNNL_ARG_DST, {dst_scale.policy, i_dst_test_scale});
            auto attr = settings_t::get_attr(test_arg_scales, i_zero_points,
                    i_post_ops, i_scratchpad_mode);

            const prb_t prb(s.prb_dims, i_sdt, i_ddt, i_stag, i_dtag, attr,
                    i_ctx_init, i_ctx_exe, i_oflag, i_cross_engine,
                    i_runtime_dim_mask);
            if (s.pattern && !match_regex(prb.str(), s.pattern)) return;

            task_executor.submit(
                    prb, s.perf_template, createit, check_cacheit, doit);
        }
    }
}

int verify_input(const settings_t &s) {
    for_(const auto &i_scales : s.scales)
    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        if (i_scales.get(arg).policy == policy_t::PER_OC) {
            BENCHDNN_PRINT(0, "%s\n",
                    "ERROR: `per_oc` policy is not supported due to "
                    "potential ambiguity. Please use one of `per_dim_0` or "
                    "`per_dim_1` policies.");
            return FAIL;
        }
    }

    for (const auto &i_cross_engine : s.cross_engine) {
        if (i_cross_engine != NONE && is_cpu()) {
            BENCHDNN_PRINT(0, "%s\n",
                    "ERROR: `cpu` engine does not support anything but "
                    "`--cross-engine=none`.");
            return FAIL;
        }
    }

    return OK;
}

static const std::string help_oflag
        = "FLAG:MASK[+...]    (Default: not specified)\n    Specifies `extra` "
          "field of destination memory descriptor.\n    `FLAG` values are "
          "`s8s8_comp` and `zp_comp`.\n    `MASK` is an non-negative integer "
          "specifying dimension to apply compensation.\n";

static const std::string help_runtime_dim_mask
        = "UINT    (Default: `0`)\n    Specifies a bit-mask that indicates "
          "whether a dimension is `DNNL_RUNTIME_DIM_VAL` if `1` on a "
          "correspondent dimension.\n";

static const std::string help_def_scales
        = "FLOAT\n    Scales, used to improve testing coverage.\n    If "
          "`--attr-scales` is specified, does not have an effect.\n";

static const std::string help_cross_engine
        = "KIND    (Default: `none`)\n    Specifies `KIND` of cross-engine "
          "used for benchmarking.\n    `KIND` values are `none`, `cpu2gpu` or "
          "`gpu2cpu`.\n";

int bench(int argc, char **argv) {
    driver_name = "reorder";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    driver_task_executor_t task_executor;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dt(s.sdt, def.sdt, argv[0], "sdt")
                || parse_dt(s.ddt, def.ddt, argv[0], "ddt")
                || parse_tag(s.stag, def.stag, argv[0], "stag")
                || parse_tag(s.dtag, def.dtag, argv[0], "dtag")
                || parse_multivector_option(s.oflag, def.oflag, str2flag,
                        argv[0], "oflag", help_oflag, ',', '+')
                || parse_vector_option(s.runtime_dim_mask, def.runtime_dim_mask,
                        atoi, argv[0], "runtime-dim-mask",
                        help_runtime_dim_mask)
                || parse_vector_option(s.def_scale, def.def_scale, atof,
                        argv[0], "def-scales", help_def_scales)
                || parse_vector_option(s.cross_engine, def.cross_engine,
                        str2cross_engine, argv[0], "cross-engine",
                        help_cross_engine)
                || parse_attr_scales(s.scales, argv[0])
                || parse_attr_zero_points(s.zero_points, argv[0])
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

            parse_prb_dims(s.prb_dims, argv[0]);

            SAFE(verify_input(s), WARN);
            check_correctness(s, task_executor);
        }
    }

    task_executor.flush();

    return parse_last_argument();
}

} // namespace reorder
