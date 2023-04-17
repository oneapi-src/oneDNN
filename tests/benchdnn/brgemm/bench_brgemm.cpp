/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "brgemm/brgemm.hpp"

namespace brgemm {

#if defined(DNNL_X64) && DNNL_X64 == 1 && DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE

void check_correctness(const settings_t &s, const settings_t &def) {
    for_(const auto &i_dt : s.dt)
    for_(const auto &i_bia_dt : s.bia_dt)
    for_(const auto &i_stag : s.stag)
    for_(const auto &i_wtag : s.wtag)
    for_(const auto &i_dtag : s.dtag)
    for_(const auto &i_ld : s.ld)
    for_(const auto &i_alpha : s.alpha)
    for_(const auto &i_beta : s.beta)
    for_(const auto &i_batch_size : s.batch_size)
    for_(const auto &i_brgemm_attr : s.brgemm_attr)
    for_(const auto &i_scales : s.scales)
    for_(const auto &i_zero_points : s.zero_points)
    for_(const auto &i_post_ops : s.post_ops)
    for_(const auto &i_scratchpad_mode : s.scratchpad_mode)
    for_(const auto &i_fpmath_mode : s.fpmath_mode)
    for_(const auto &i_ctx_init : s.ctx_init)
    for (const auto &i_ctx_exe : s.ctx_exe) {
        auto attr = settings_t::get_attr(i_scales, i_zero_points, i_post_ops,
                i_scratchpad_mode, i_fpmath_mode);

        const prb_t prb(s.prb_vdims, i_dt, i_stag, i_wtag, i_dtag, i_ld,
                i_bia_dt, i_alpha, i_beta, i_batch_size, i_brgemm_attr, attr,
                i_ctx_init, i_ctx_exe);
        if (s.pattern && !match_regex(prb.str(), s.pattern)) return;
        BENCHDNN_PRINT(1, "run: %s\n", prb.str());

        res_t res {};
        doit(&prb, &res);

        parse_result(res, prb.str());

        if (has_bench_mode_bit(mode_bit_t::perf)) {
            perf_report_t pr(&prb, s.perf_template);
            pr.report(&res, prb.str());
        }
    }
}

int verify_input(const settings_t &s) {
    static constexpr int n_inputs = 3;

    if (s.prb_vdims.ndims > 2) {
        fprintf(stderr,
                "ERROR: brgemm driver: problem descriptor supports only "
                "MxK:KxN notion.\n"),
                fflush(stderr);
        SAFE_V(FAIL);
    }

    for (const auto &i_dt : s.dt) {
        if (i_dt.size() != 1 && i_dt.size() != n_inputs) {
            fprintf(stderr,
                    "ERROR: brgemm driver: `dt` option expects either a single "
                    "input or three inputs in SRC, WEI, and DST order. Current "
                    "size is: \"%ld\"\n",
                    (long)i_dt.size()),
                    fflush(stderr);
            SAFE_V(FAIL);
        }
    }

    return OK;
}

static const std::string help_alpha
        = "FLOAT    (Default: 1.f)\n    Specifies real value corresponding to "
          "scaling of accumulator result: `C = alpha * A * B`.\n";

static const std::string help_beta
        = "FLOAT    (Default: 0.f)\n    Specifies real value corresponding to "
          "adding a part of accumulator result: `C = A * B + beta * C`.\n";

static const std::string help_batch_size
        = "UINT    (Default: `1`)\n    Specifies a batch size that indicates "
          "how many batches per kernel call will be used.\n";

static const std::string help_ld
        = "UINT:UINT:UINT    (Default: not specified)\n    Specifies "
          "LDA:LDB:LDD values. If some values are skipped, the default one (K, "
          "N, or N) will be used. If there are no post-ops, LDC will reuse "
          "LDD, otherwise expect LDC always dense.\n";

static const std::string help_brgemm_attr
        = "STRING    (Default: empty)\n    Specifies BRGeMM kernel attributes. "
          "If some values are skipped, the default one will be used.\n";

int bench(int argc, char **argv) {
    // BRGeMM kernel support is available on x86 Intel CPU only.
    if (is_gpu()) return OK;
    driver_name = "brgemm";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    for (; argc > 0; --argc, ++argv) {
        auto cstr2str = [](const char *str) { return std::string(str); };
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_multi_dt(s.dt, def.dt, argv[0], "dt")
                || parse_dt(s.bia_dt, def.bia_dt, argv[0], "bia_dt")
                || parse_multivector_option(
                        s.ld, def.ld, atoi, argv[0], "ld", help_ld)
                || parse_vector_option(s.batch_size, def.batch_size, atoi,
                        argv[0], "bs", help_batch_size)
                || parse_vector_option(
                        s.alpha, def.alpha, atof, argv[0], "alpha", help_alpha)
                || parse_vector_option(
                        s.beta, def.beta, atof, argv[0], "beta", help_beta)
                || parse_vector_option(s.brgemm_attr, def.brgemm_attr, cstr2str,
                        argv[0], "brgemm-attr", help_brgemm_attr)
                || parse_attr_scales(s.scales, argv[0])
                || parse_attr_zero_points(s.zero_points, argv[0])
                || parse_attr_post_ops(s.post_ops, argv[0])
                || parse_attr_scratchpad_mode(
                        s.scratchpad_mode, def.scratchpad_mode, argv[0])
                || parse_attr_fpmath_mode(
                        s.fpmath_mode, def.fpmath_mode, argv[0])
                || parse_test_pattern_match(s.pattern, argv[0])
                || parse_perf_template(s.perf_template, s.perf_template_def,
                        s.perf_template_csv(), argv[0])
                || parse_reset(s, argv[0]) || parse_help(argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            parse_prb_vdims(s.prb_vdims, argv[0]);

            SAFE(verify_input(s), WARN);
            check_correctness(s, def);
        }
    }
    return parse_last_argument();
}

#else

int bench(int argc, char **argv) {
    BENCHDNN_PRINT(
            0, "%s\n", "INFO: brgemm driver: only x64 backend is supported.");
    return OK;
}

#endif

} // namespace brgemm
