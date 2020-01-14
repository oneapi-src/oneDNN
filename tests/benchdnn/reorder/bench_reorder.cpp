/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#include <sstream>

#include "dnnl.h"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "parser.hpp"

#include "reorder.hpp"

namespace reorder {

std::vector<dnnl_data_type_t> sdt {dnnl_f32}, ddt {dnnl_f32};
std::vector<dnnl_format_tag_t> stag {dnnl_nchw}, dtag {dnnl_nchw};
std::vector<float> def_scale {0.125, 0.25, 0.5, 1, 2, 4, 8};
std::vector<flag_t> oflag {FLAG_NONE};
std::vector<unsigned> runtime_dim_mask {0};

dims_t dims;
alg_t alg = ALG_REF;
attr_t attr;
bool allow_unimpl = false;
const char *perf_template_csv
        = "perf,%engine%,%sdt%,%ddt%,%stag%,%dtag%,%flags%,%attr%,%DESC%,"
          "%Gops%,%-time%,%-Gbw%,%0time%,%0Gbw%";
const char *perf_template_def
        = "perf,%engine%,%prb%,%Gops%,%-time%,%-Gbw%,%0time%,%0Gbw%";
const char *perf_template = perf_template_def;

void reset_parameters() {
    sdt = {dnnl_f32};
    ddt = {dnnl_f32};
    stag = {dnnl_nchw};
    dtag = {dnnl_nchw};
    def_scale = {0.125, 0.25, 0.5, 1, 2, 4, 8};
    oflag = {FLAG_NONE};
    runtime_dim_mask = {0};
    alg = ALG_REF;
    attr = attr_t();
    allow_unimpl = false;
}

void check_correctness() {
    for_(const auto &i_sdt : sdt)
    for_(const auto &i_ddt : ddt)
    for_(const auto &i_stag : stag)
    for_(const auto &i_dtag : dtag)
    for_(const auto &i_oflag : oflag)
    for (const auto &i_runtime_dim_mask : runtime_dim_mask) {
        reorder_conf_t reorder_conf {dims, i_stag, i_dtag};
        dt_conf_t iconf = dt2cfg(i_sdt);
        dt_conf_t oconf = dt2cfg(i_ddt);

        std::vector<float> attr_scale = {attr.oscale.scale};
        auto &scale = attr.oscale.scale == 0 ? def_scale : attr_scale;

        for (const auto &i_scale : scale) {
            const prb_t p(reorder_conf, iconf, oconf, attr, alg, i_oflag,
                    i_runtime_dim_mask, i_scale);
            std::stringstream ss;
            ss << p;
            const std::string cpp_pstr = ss.str();
            const char *pstr = cpp_pstr.c_str();
            BENCHDNN_PRINT(1, "run: %s\n", pstr);

            res_t res {};
            int status = doit(&p, &res);

            bool want_perf_report = false;
            parse_result(res, want_perf_report, allow_unimpl, status, pstr);

            if (want_perf_report && bench_mode & PERF) {
                perf_report_t pr(perf_template);
                pr.report(&p, &res, pstr);
            }

            benchdnn_stat.tests++;
        }
    }
}

int bench(int argc, char **argv) {
    driver_name = "reorder";
    using namespace parser;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = false || parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0]) || parse_dt(sdt, argv[0], "sdt")
                || parse_dt(ddt, argv[0], "ddt")
                || parse_tag(stag, argv[0], "stag")
                || parse_tag(dtag, argv[0], "dtag")
                || parse_vector_option(oflag, str2flag, argv[0], "oflag")
                || parse_vector_option(
                        runtime_dim_mask, atoi, argv[0], "runtime-dim-mask")
                || parse_single_value_option(alg, str2alg, argv[0], "alg")
                || parse_vector_option(def_scale, atof, argv[0], "def-scales")
                || parse_attr(attr, argv[0])
                || parse_allow_unimpl(allow_unimpl, argv[0])
                || parse_perf_template(perf_template, perf_template_def,
                        perf_template_csv, argv[0])
                || parse_reset(reset_parameters, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            parse_dims(dims, argv[0]);
            check_correctness();
        }
    }

    return parse_last_argument();
}

} // namespace reorder
