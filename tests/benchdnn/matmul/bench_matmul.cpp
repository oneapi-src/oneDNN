/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <sstream>

#include "dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "parser.hpp"

#include "matmul/matmul.hpp"

namespace matmul {

std::vector<const dt_conf_t *> cfg;
std::vector<dnnl_format_tag_t> stag, wtag, dtag;
std::vector<int64_t> ld_src, ld_wei, ld_dst;
std::vector<bool> runtime_mb, runtime_m, runtime_n, runtime_k;
std::vector<dnnl_data_type_t> bia_dt;
std::vector<int> bia_mask;

attr_t attr;
bool allow_unimpl;
const char *skip_impl = "";
const char *perf_template_csv
        = "perf,%engine%,%name%,%cfg%,%attr%,%DESC%,"
          "%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%";
const char *perf_template_def
        = "perf,%engine%,%name%,%prb%,"
          "%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%";
const char *perf_template = perf_template_def;

void reset_parameters() {
    cfg = {defaults::cfg};
    stag = {defaults::tag};
    wtag = {defaults::tag};
    dtag = {defaults::tag};
    ld_src = {defaults::ld};
    ld_wei = {defaults::ld};
    ld_dst = {defaults::ld};
    runtime_mb = {defaults::runtime_val};
    runtime_m = {defaults::runtime_val};
    runtime_n = {defaults::runtime_val};
    runtime_k = {defaults::runtime_val};
    bia_dt = {defaults::bia_dt};
    bia_mask = {defaults::bia_mask};
    attr = attr_t();
    allow_unimpl = false;
    skip_impl = "";
}

void check_correctness(const desc_t *c) {
    std::vector<std::pair<dnnl_data_type_t, int>> bia_cfg;
    for (const auto &i_bia_dt : bia_dt) {
        if (i_bia_dt == dnnl_data_type_undef) {
            bia_cfg.emplace_back(i_bia_dt, 0);
            continue;
        }
        for (const auto &i_bia_mask : bia_mask)
            bia_cfg.emplace_back(i_bia_dt, i_bia_mask);
    }

    for_(const auto &i_cfg : cfg)
    for_(const auto &i_stag : stag)
    for_(const auto &i_wtag : wtag)
    for_(const auto &i_dtag : dtag)
    for_(const auto &i_ld_src : ld_src)
    for_(const auto &i_ld_wei : ld_wei)
    for_(const auto &i_ld_dst : ld_dst)
    for_(const auto &i_runtime_mb : runtime_mb)
    for_(const auto &i_runtime_m : runtime_m)
    for_(const auto &i_runtime_n : runtime_n)
    for_(const auto &i_runtime_k : runtime_k)
    for_(const auto &i_bia_cfg : bia_cfg)
    {
        const prb_t p(*c, i_cfg, i_stag, i_wtag, i_dtag, i_ld_src, i_ld_wei,
                i_ld_dst, i_runtime_mb, i_runtime_m, i_runtime_n, i_runtime_k,
                i_bia_cfg.first, i_bia_cfg.second, attr);
        std::stringstream ss;
        ss << p;
        const std::string cpp_pstr = ss.str();
        const char *pstr = cpp_pstr.c_str();
        BENCHDNN_PRINT(1, "run: %s\n", pstr);

        res_t res {};
        const int status = doit(&p, &res);

        bool want_perf_report = false;
        parse_result(res, want_perf_report, allow_unimpl, status, pstr);

        if (want_perf_report && bench_mode & PERF) {
            perf_report_t pr(perf_template);
            pr.report(&p, &res, pstr);
        }

        benchdnn_stat.tests++;
    }
}

int bench(int argc, char **argv) {
    driver_name = "matmul";
    reset_parameters();

    using namespace parser;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = false || parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_cfg(cfg, str2cfg, argv[0])
                || parse_tag(stag, argv[0], "stag")
                || parse_tag(wtag, argv[0], "wtag")
                || parse_tag(dtag, argv[0], "dtag")
                || parse_vector_option(ld_src, atoi, argv[0], "ld_src")
                || parse_vector_option(ld_wei, atoi, argv[0], "ld_wei")
                || parse_vector_option(ld_dst, atoi, argv[0], "ld_dst")
                || parse_vector_option(
                        runtime_mb, str2bool, argv[0], "runtime_mb")
                || parse_vector_option(
                        runtime_m, str2bool, argv[0], "runtime_m")
                || parse_vector_option(
                        runtime_n, str2bool, argv[0], "runtime_n")
                || parse_vector_option(
                        runtime_k, str2bool, argv[0], "runtime_k")
                || parse_dt(bia_dt, argv[0], "bia_dt")
                || parse_vector_option(bia_mask, atoi, argv[0], "bia_mask")
                || parse_attr(attr, argv[0])
                || parse_allow_unimpl(allow_unimpl, argv[0])
                || parse_skip_impl(skip_impl, argv[0])
                || parse_perf_template(perf_template, perf_template_def,
                        perf_template_csv, argv[0])
                || parse_reset(reset_parameters, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            desc_t c;
            SAFE_V(str2desc(&c, argv[0]));
            check_correctness(&c);
        }
    }

    return parse_last_argument();
}

} // namespace matmul
