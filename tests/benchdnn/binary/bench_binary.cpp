/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include "dnnl_common.hpp"
#include "utils/parser.hpp"
#include "utils/task_executor.hpp"

#include "binary/binary.hpp"

namespace binary {

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
    for_(const auto &i_alg : s.alg)
    for_(const auto &i_scales : s.scales)
    for_(const auto &i_post_ops : s.post_ops)
    for_(const auto &i_scratchpad_mode : s.scratchpad_mode)
    for_(const auto &i_ctx_init : s.ctx_init)
    for_(const auto &i_ctx_exe : s.ctx_exe)
    for (auto i_inplace : s.inplace) {
        auto attr
                = settings_t::get_attr(i_scales, i_post_ops, i_scratchpad_mode);

        const prb_t prb(s.prb_vdims, i_sdt, i_ddt, i_stag, i_dtag, i_alg,
                i_inplace, attr, i_ctx_init, i_ctx_exe);
        if (s.pattern && !match_regex(prb.str(), s.pattern)) return;

        task_executor.submit(
                prb, s.perf_template, createit, check_cacheit, doit);
    }
}

int verify_input(const settings_t &s) {
    // Expect exactly two inputs for problem dimensions.
    static constexpr int n_inputs = 2;

    if (s.prb_vdims.n_inputs() != n_inputs) {
        fprintf(stderr,
                "ERROR: binary driver: expect problem dimensions in format "
                "`A0xA1x...:B0xB1x...`.\n");
        SAFE_V(FAIL);
    }

    for (const auto &i_sdt : s.sdt) {
        if (i_sdt.size() != n_inputs) {
            fprintf(stderr,
                    "ERROR: binary driver: expect data types in format "
                    "`DT:DT`.\n");
            SAFE_V(FAIL);
        }
    }

    for (const auto &i_stag : s.stag) {
        if (i_stag.size() != n_inputs) {
            fprintf(stderr,
                    "ERROR: binary driver: expect format tags in format "
                    "`TAG:TAG`.\n");
            SAFE_V(FAIL);
        }
    }

    for (const auto &i_alg : s.alg) {
        if (!(i_alg > alg_t::BINARY_START && i_alg < alg_t::BINARY_END)) {
            fprintf(stderr,
                    "ERROR: binary driver: algorithm `%s` does not belong to "
                    "binary algorithm type.\n",
                    attr_t::post_ops_t::kind2str(i_alg));
            SAFE_V(FAIL);
        }
    }

    return OK;
}

int bench(int argc, char **argv) {
    driver_name = "binary";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    driver_task_executor_t task_executor;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_multi_dt(s.sdt, def.sdt, argv[0])
                || parse_dt(s.ddt, def.ddt, argv[0], "ddt")
                || parse_multi_tag(s.stag, def.stag, argv[0])
                || parse_tag(s.dtag, def.dtag, argv[0], "dtag")
                || parse_alg(
                        s.alg, def.alg, attr_t::post_ops_t::str2kind, argv[0])
                || parse_inplace(s.inplace, def.inplace, argv[0])
                || parse_attr_scales(s.scales, argv[0])
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

            parse_prb_vdims(s.prb_vdims, argv[0]);

            SAFE(verify_input(s), WARN);
            check_correctness(s, task_executor);
        }
    }

    task_executor.flush();

    return parse_last_argument();
}

} // namespace binary
