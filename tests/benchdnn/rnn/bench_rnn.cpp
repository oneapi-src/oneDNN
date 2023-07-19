/*******************************************************************************
* Copyright 2018-2023 Intel Corporation
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

#include "oneapi/dnnl/dnnl.h"

#include "dnnl_common.hpp"
#include "utils/parser.hpp"

#include "rnn/rnn.hpp"

// TODO: replace with common abstractions when RNN gets rid of static scales
#include "rnn/rnn_task.hpp"
#include "rnn/rnn_task_executor.hpp"

namespace rnn {

using create_func_t = std::function<int(
        std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &, const prb_t &,
        res_t *)>;
using check_cache_func_t = std::function<int(
        std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &, const prb_t *,
        res_t *)>;
using do_func_t = std::function<int(
        const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &,
        const prb_t &, res_t *)>;
using driver_task_executor_t = rnn_task_executor_t<prb_t, perf_report_t,
        create_func_t, check_cache_func_t, do_func_t>;

void check_correctness(
        const settings_t &s, driver_task_executor_t &task_executor) {
    for_(const auto &i_prop : s.prop)
    for_(const auto &i_cfg : s.cfg)
    for_(const auto &i_alg : s.alg)
    for_(auto i_with_peephole : s.with_peephole)
    for_(auto i_with_projection : s.with_projection)
    for_(const auto &i_scale_policy : s.scale_policy)
    for_(const auto &i_scale_proj_policy : s.scale_proj_policy)
    for_(const auto &i_direction : s.direction)
    for_(const auto &i_activation : s.activation)
    for_(auto i_skip_nonlinear : s.skip_nonlinear)
    for_(auto i_trivial_strides : s.trivial_strides)
    for_(const auto &i_flags : s.flags)
    for_(const auto &i_n_layer : s.n_layer)
    for_(const auto &i_n_iter : s.n_iter)
    for_(const auto &i_mb : s.mb)
    for_(const auto &i_scratchpad_mode : s.scratchpad_mode)
    for_(const auto &i_ctx_init : s.ctx_init)
    for_(const auto &i_ctx_exe : s.ctx_exe)
    for (const auto &i_fpmath_mode : s.fpmath_mode) {
        auto attr = settings_t::get_attr(i_scratchpad_mode, i_fpmath_mode);

        auto prb = std::make_shared<prb_t>(s.desc,
                dt_conf_t::create(i_cfg, attr), i_prop, i_alg, i_with_peephole,
                i_with_projection, i_direction, i_scale_policy,
                i_scale_proj_policy, i_flags, i_activation, attr, i_ctx_init,
                i_ctx_exe, s.alpha, s.beta, i_skip_nonlinear, i_trivial_strides,
                i_n_layer, i_n_iter, i_mb);

        task_executor.submit(
                std::move(prb), s.perf_template, createit, check_cacheit, doit);
    }
}

int verify_input(const settings_t &s) {

    for (const auto &i_scale_policy : s.scale_policy) {
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
    }

    return OK;
}

static const std::string help_direction
        = "DIRECTION    (Default: `left2right`)\n    Specifies evaluation "
          "direction.\n    `DIRECTION` values are: `left2right`, `right2left`, "
          "`concat` and `sum`.\n";

static const std::string help_activation
        = "ACTIVATION    (Default: `relu`)\n    Specifies Vanilla RNN "
          "activation function.\n    `ACTIVATION` values are: `relu`, "
          "`logistic`, and `tanh`.\n";

static const std::string help_l
        = "UINT    (Default: `0`)\n    Overrides number of layers value "
          "specified in a problem descriptor with `UINT` value.\n    When set "
          "to `0`, takes no effect.\n";

static const std::string help_t
        = "UINT    (Default: `0`)\n    Overrides number of timestamps value "
          "specified in a problem descriptor with `UINT` value.\n    When set "
          "to `0`, takes no effect.\n";

static const std::string help_with_peephole
        = "BOOL    (Default: `false`)\n    When set to `true`, enables LSTM "
          "peephole extension.\n";

static const std::string help_with_projection
        = "BOOL    (Default: `false`)\n    When set to `true`, enables LSTM "
          "projection extension.\n";

static const std::string help_flags
        = "FLAGS    (Default: not specified)\n    Specifies rnn flags. `FLAGS` "
          "values are:\n    * `O` for diff_weights_overwrite.\n";

int bench(int argc, char **argv) {
    driver_name = "rnn";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    driver_task_executor_t task_executor;
    for (; argc > 0; --argc, ++argv) {
        auto cstr2str = [](const char *str) { return std::string(str); };
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dir(s.prop, def.prop, argv[0], "prop")
                || parse_cfg(s.cfg, def.cfg, cstr2str, argv[0])
                || parse_alg(s.alg, def.alg, str2alg, argv[0])
                || parse_vector_option(s.direction, def.direction,
                        str2direction, argv[0], "direction", help_direction)
                || parse_vector_option(s.activation, def.activation,
                        str2activation, argv[0], "activation", help_activation)
                || parse_scale_policy(s.scale_policy, def.scale_policy, argv[0])
                || parse_scale_policy(s.scale_proj_policy,
                        def.scale_proj_policy, argv[0], "scaling-proj")
                || parse_mb(s.mb, def.mb, argv[0])
                || parse_vector_option(
                        s.n_layer, def.n_layer, atoi, argv[0], "l", help_l)
                || parse_vector_option(
                        s.n_iter, def.n_iter, atoi, argv[0], "t", help_t)
                || parse_skip_nonlinear(
                        s.skip_nonlinear, def.skip_nonlinear, argv[0])
                || parse_trivial_strides(
                        s.trivial_strides, def.trivial_strides, argv[0])
                || parse_vector_option(s.flags, def.flags, str2flags, argv[0],
                        "flags", help_flags)
                || parse_vector_option(s.with_peephole, def.with_peephole,
                        str2bool, argv[0], "with-peephole", help_with_peephole)
                || parse_vector_option(s.with_projection, def.with_projection,
                        str2bool, argv[0], "with-projection",
                        help_with_projection)
                || parse_attr_scratchpad_mode(
                        s.scratchpad_mode, def.scratchpad_mode, argv[0])
                || parse_attr_fpmath_mode(
                        s.fpmath_mode, def.fpmath_mode, argv[0])
                || parse_ctx_init(s.ctx_init, def.ctx_init, argv[0])
                || parse_ctx_exe(s.ctx_exe, def.ctx_exe, argv[0])
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

} // namespace rnn
