/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "mkldnn.h"
#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"
#include "parser.hpp"

#include "reorder.hpp"

namespace reorder {

std::vector<mkldnn_data_type_t> idt {mkldnn_f32}, odt {mkldnn_f32};
std::vector<mkldnn_format_tag_t> itag {mkldnn_nchw}, otag {mkldnn_nchw};
std::vector<float> def_scale {0.125, 0.25, 0.5, 1, 2, 4, 8};
std::vector<flag_t> oflag {FLAG_NONE};

dims_t dims;
alg_t alg = ALG_REF;
attr_t attr;
bool allow_unimpl = false;
const char *perf_template_csv =
    "perf,%engine%,%idt%,%odt%,%itag%,%otag%,%flags%,%attr%,%DESC%,"
    "%Gops%,%-time%,%-Gbw%,%0time%,%0Gbw%";
const char *perf_template_def =
    "perf,%engine%,%desc%,%Gops%,%-time%,%-Gbw%,%0time%,%0Gbw%";
const char *perf_template = perf_template_def;

void reset_parameters() {
    idt = {mkldnn_f32};
    odt = {mkldnn_f32};
    itag = {mkldnn_nchw};
    otag = {mkldnn_nchw};
    def_scale = {0.125, 0.25, 0.5, 1, 2, 4, 8};
    oflag = {FLAG_NONE};
    alg = ALG_REF;
    attr = attr_t();
    allow_unimpl = false;
}

void check_correctness() {
    for (const auto &i_idt: idt)
    for (const auto &i_odt: odt)
    for (const auto &i_itag: itag)
    for (const auto &i_otag: otag)
    for (const auto &i_oflag: oflag) {
        reorder_conf_t reorder_conf{dims, i_itag, i_otag};
        dt_conf_t iconf = dt2cfg(i_idt);
        dt_conf_t oconf = dt2cfg(i_odt);

        std::vector<float> attr_scale = {attr.oscale.scale};
        auto &scale = attr.oscale.scale == 0 ? def_scale : attr_scale;

        for (const auto &i_scale: scale) {
            const prb_t p(reorder_conf, iconf, oconf, attr, alg, i_oflag,
                    i_scale);
            std::stringstream ss;
            ss << p;
            const std::string cpp_pstr = ss.str();
            const char *pstr = cpp_pstr.c_str();

            res_t res{};
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
    using namespace parser;
    for (; argc > 0; --argc, ++argv) {
        if (parse_bench_settings(argv[0]));
        else if (parse_batch(bench, argv[0]));
        else if (parse_dt(idt, argv[0], "idt"));
        else if (parse_dt(odt, argv[0], "odt"));
        else if (parse_tag(itag, argv[0], "itag"));
        else if (parse_tag(otag, argv[0], "otag"));
        else if (parse_attr(attr, argv[0]));
        else if (parse_vector_option(oflag, str2flag, argv[0], "oflag"));
        else if (parse_vector_option(def_scale, atof, argv[0], "def-scales"));
        else if (parse_single_value_option(alg, str2alg, argv[0], "alg"));
        else if (parse_allow_unimpl(allow_unimpl, argv[0]));
        else if (parse_perf_template(perf_template, perf_template_def,
                    perf_template_csv, argv[0]));
        else if (parse_reset(reset_parameters, argv[0]));
        else {
            catch_unknown_options(argv[0], "reorder");

            dims = str2dims(argv[0]);
            check_correctness();
        }
    }


    return OK;
}

}
