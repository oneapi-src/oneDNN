/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include "matmul/matmul.hpp"

namespace matmul {

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
    std::vector<std::pair<dnnl_data_type_t, int>> bia_cfg;
    for (const auto &i_bia_dt : s.bia_dt) {
        if (i_bia_dt == dnnl_data_type_undef) {
            bia_cfg.emplace_back(i_bia_dt, 0);
            continue;
        }
        for (const auto &i_bia_mask : s.bia_mask)
            bia_cfg.emplace_back(i_bia_dt, i_bia_mask);
    }

    for_(const auto &i_dt : s.dt)
    for_(const auto &i_stag : s.stag)
    for_(const auto &i_wtag : s.wtag)
    for_(const auto &i_dtag : s.dtag)
    for_(const auto &i_sparse_options : s.sparse_options)
    for_(const auto &i_strides : s.strides)
    for_(const auto &i_rt_dims_masks : s.rt_dims_masks)
    for_(const auto &i_attr : s.attributes)
    for_(const auto &i_ctx_init : s.ctx_init)
    for_(const auto &i_ctx_exe : s.ctx_exe)
    for (const auto &i_bia_cfg : bia_cfg) {
        const prb_t prb(s.prb_vdims, i_dt, i_stag, i_wtag, i_dtag, i_strides,
                i_bia_cfg.first, i_bia_cfg.second, i_rt_dims_masks,
                i_sparse_options,
                i_attr, i_ctx_init, i_ctx_exe, s.impl_filter);
        if (s.pattern && !match_regex(prb.str(), s.pattern)) return;

        task_executor.submit(
                prb, s.perf_template, createit, check_cacheit, doit);
    }
}

int verify_input(const settings_t &s, const settings_t &def) {
    static constexpr int n_inputs = 3;

    for (const auto &i_dt : s.dt) {
        if (i_dt.size() != 1 && i_dt.size() != n_inputs) {
            fprintf(stderr,
                    "ERROR: matmul driver: `dt` option expects either a single "
                    "input or three inputs in SRC, WEI, DST order. Current "
                    "size is: \"%ld\"\n",
                    (long)i_dt.size()),
                    fflush(stderr);
            SAFE_V(FAIL);
        }
    }

    for (const auto &i_strides : s.strides) {
        if (i_strides.size() != n_inputs) {
            std::stringstream ss;
            ss << i_strides;
            fprintf(stderr,
                    "ERROR: matmul driver: `strides` option expects three "
                    "inputs in format `[SRC]:[WEI]:[DST]` (two colons must "
                    "present). Current input is: \"%s\"\n",
                    ss.str().c_str()),
                    fflush(stderr);
            SAFE_V(FAIL);
        }
    }

    for_(const auto &i_strides : s.strides)
    for_(const auto &i_stag : s.stag)
    for_(const auto &i_wtag : s.wtag)
    for (const auto &i_dtag : s.dtag) {
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
    }

    static constexpr int n_vdims_inputs = 2;
    if (s.prb_vdims.n_inputs() != n_vdims_inputs) {
        BENCHDNN_PRINT(0,
                "ERROR: Expected number of dims arguments is `%d`, provided "
                "`%d`.\n",
                n_vdims_inputs, s.prb_vdims.n_inputs());
        SAFE_V(FAIL);
    }
    return OK;
}

static const std::string help_bia_mask
        = "UINT    (Default: `2`)\n    Specifies a bit-mask that indicates "
          "which bias dimensions coincide with C matrix dimensions, when `1` "
          "is on a correspondent dimension.\n";

static const std::string help_runtime_dims_masks
        = "UINT:UINT    (Default: `0:0`)\n    Specifies a bit-mask for "
          "matrices A and B that indicates whether a dimension is "
          "`DNNL_RUNTIME_DIM_VAL` if `1` on a correspondent dimension.\n    "
          "For tensors with runtime dimensions specified a correspondent "
          "memory format must be specified, too.\n";

int bench(int argc, char **argv) {
    driver_name = "matmul";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    static driver_task_executor_t task_executor;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_multi_dt(s.dt, def.dt, argv[0], "dt")
                || parse_tag(s.stag, def.stag, argv[0], "stag")
                || parse_tag(s.wtag, def.wtag, argv[0], "wtag")
                || parse_tag(s.dtag, def.dtag, argv[0], "dtag")
                || parse_encoding(s.sparse_options, argv[0], "encoding")
                || parse_strides(s.strides, def.strides, argv[0], "strides")
                || parse_dt(s.bia_dt, def.bia_dt, argv[0], "bia_dt")
                || parse_vector_option(s.bia_mask, def.bia_mask, atoi, argv[0],
                        "bia_mask", help_bia_mask)
                || parse_multivector_option(s.rt_dims_masks, def.rt_dims_masks,
                        atoi, argv[0], "runtime_dims_masks",
                        help_runtime_dims_masks)
                || parse_driver_shared_settings(s, def, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            parse_prb_vdims(s.prb_vdims, argv[0]);

            SAFE(verify_input(s, def), WARN);
            s.finalize();
            check_correctness(s, task_executor);
        }
    }

    task_executor.flush();

    return parse_last_argument();
}

} // namespace matmul
