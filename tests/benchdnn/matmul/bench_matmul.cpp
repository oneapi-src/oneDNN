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

#include "matmul/matmul.hpp"

namespace matmul {

void check_correctness(const settings_t &s, const settings_t &def) {
    std::vector<std::pair<dnnl_data_type_t, int>> bia_cfg;
    for (const auto &i_bia_dt : s.bia_dt) {
        if (i_bia_dt == dnnl_data_type_undef) {
            bia_cfg.emplace_back(i_bia_dt, 0);
            continue;
        }
        for (const auto &i_bia_mask : s.bia_mask)
            bia_cfg.emplace_back(i_bia_dt, i_bia_mask);
    }

    for_(const auto &i_cfg : s.cfg)
    for_(const auto &i_stag : s.stag)
    for_(const auto &i_wtag : s.wtag)
    for_(const auto &i_dtag : s.dtag)
    for_(const auto &i_strides : s.strides)
    for_(const auto &i_rt_dims_masks : s.rt_dims_masks)
    for_(const auto &i_oscale : s.oscale)
    for_(const auto &i_zero_points : s.zero_points)
    for_(const auto &i_post_ops : s.post_ops)
    for_(const auto &i_scratchpad_mode : s.scratchpad_mode)
    for_(const auto &i_fpmath_mode : s.fpmath_mode)
    for (const auto &i_bia_cfg : bia_cfg) {
        attr_t attr;
        attr.insert(i_oscale);
        attr.insert(i_zero_points);
        attr.insert(i_post_ops);
        attr.insert(i_scratchpad_mode);
        attr.insert(i_fpmath_mode);

        const bool strided_input = !i_strides[STRIDES_SRC].empty()
                || !i_strides[STRIDES_WEI].empty()
                || !i_strides[STRIDES_DST].empty();
        if (strided_input) {
            const bool no_stride_with_tag
                    = IMPLICATION(i_stag != def.stag[0],
                              i_strides[STRIDES_SRC].empty())
                    && IMPLICATION(i_wtag != def.wtag[0],
                            i_strides[STRIDES_WEI].empty())
                    && IMPLICATION(i_dtag != def.dtag[0],
                            i_strides[STRIDES_DST].empty());

            if (!no_stride_with_tag) {
                fprintf(stderr,
                        "ERROR: matmul driver: both `strides` and `tag` knobs "
                        "can not be used with either of `src`, `wei`, and `dst`"
                        " tensors.\n"),
                        fflush(stderr);
                SAFE_V(FAIL);
            }
        }

        const prb_t prb(s.prb_vdims, i_cfg, i_stag, i_wtag, i_dtag, i_strides,
                i_bia_cfg.first, i_bia_cfg.second, i_rt_dims_masks, attr);
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

static const std::string help_bia_mask
        = "UINT    (Default: `2`)\n    Specifies a bit-mask that indicates "
          "which bias dimensions coincide with C matrix dimensions, when `1` "
          "is on a correspondent dimension.\n";

static const std::string help_runtime_dims_masks
        = "UINT:UINT    (Default: `0:0`)\n    Specifies a bit-mask for "
          "matrices A and B that indicates whether a dimension is "
          "`DNNL_RUNTIME_DIM_VAL` if `1` on a correspondent dimension.\n";

int bench(int argc, char **argv) {
    driver_name = "matmul";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_cfg(s.cfg, def.cfg, str2cfg, argv[0])
                || parse_tag(s.stag, def.stag, argv[0], "stag")
                || parse_tag(s.wtag, def.wtag, argv[0], "wtag")
                || parse_tag(s.dtag, def.dtag, argv[0], "dtag")
                || parse_strides(s.strides, def.strides, argv[0], "strides")
                || parse_dt(s.bia_dt, def.bia_dt, argv[0], "bia_dt")
                || parse_vector_option(s.bia_mask, def.bia_mask, atoi, argv[0],
                        "bia_mask", help_bia_mask)
                || parse_multivector_option(s.rt_dims_masks, def.rt_dims_masks,
                        atoi, argv[0], "runtime_dims_masks",
                        help_runtime_dims_masks)
                || parse_attr_oscale(s.oscale, argv[0])
                || parse_attr_zero_points(s.zero_points, argv[0])
                || parse_attr_post_ops(s.post_ops, argv[0])
                || parse_attr_scratchpad_mode(
                        s.scratchpad_mode, def.scratchpad_mode, argv[0])
                || parse_attr_fpmath_mode(
                        s.fpmath_mode, def.fpmath_mode, argv[0])
                || parse_perf_template(s.perf_template, s.perf_template_def,
                        s.perf_template_csv(), argv[0])
                || parse_reset(s, argv[0]) || parse_help(argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            parse_prb_vdims(s.prb_vdims, argv[0]);
            check_correctness(s, def);
        }
    }

    return parse_last_argument();
}

} // namespace matmul
