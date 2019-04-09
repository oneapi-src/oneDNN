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

#include <stdlib.h>
#include <stdio.h>

#include "mkldnn.h"

#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"

#include "softmax/softmax.hpp"

namespace softmax {

mkldnn_data_type_t dt = mkldnn_f32;
mkldnn_format_tag_t tag = mkldnn_nchw;
dir_t dir = FWD_D;
int64_t mb = 0;
dims_t dims;
int axis = 1;
bool allow_unimpl = false;
const char *perf_template = "perf,%z,%q,%f,%a,%D,%-t,%0t";

void reset_parameters() {
    dt = mkldnn_f32;
    tag = mkldnn_nchw;
    dir = FWD_D;
    mb = 0;
    axis = 1;
    allow_unimpl = false;
}

void check_correctness() {
    const prb_t p(dims, dir, dt, tag, axis, mb);
    char pstr[max_prb_len];
    prb2str(&p, pstr);

    res_t res{};
    const int status = softmax::doit(&p, &res);

    bool want_perf_report = false;

    parse_result(res, want_perf_report, allow_unimpl, status, pstr);

    if (want_perf_report && bench_mode & PERF)
        perf_report(&p, &res, pstr);

    benchdnn_stat.tests++;
}

int bench(int argc, char **argv, bool main_bench) {
    for (int arg = 0; arg < argc; ++arg) {
        if (!strncmp("--batch=", argv[arg], 8))
            SAFE(batch(argv[arg] + 8, bench), CRIT);
        else if (!strncmp("--dt=", argv[arg], 5))
            dt = str2dt(argv[arg] + 5);
        else if (!strncmp("--tag=", argv[arg], 6))
            tag = str2tag(argv[arg] + 6);
        else if (!strncmp("--mb=", argv[arg], 5))
            mb = atoi(argv[arg] + 5);
        else if (!strncmp("--dir=", argv[arg], 6))
            dir = str2dir(argv[arg] + 6);
        else if (!strncmp("--allow-unimpl=", argv[arg], 15))
            allow_unimpl = str2bool(argv[arg] + 15);
        else if (!strncmp("--perf-template=", argv[arg], 16))
            perf_template = argv[arg] + 16;
        else if (!strcmp("--reset", argv[arg]))
            reset_parameters();
        else if (!strncmp("--mode=", argv[arg], 7))
            bench_mode = str2bench_mode(argv[arg] + 7);
        else if (!strncmp("-v", argv[arg], 2))
            verbose = atoi(argv[arg] + 2);
        else if (!strncmp("--verbose=", argv[arg], 10))
            verbose = atoi(argv[arg] + 10);
        else if (!strncmp("--axis=", argv[arg], 7))
            axis = atoi(argv[arg] + 7);
        else {
            if (!strncmp("--", argv[arg], 2)) {
                fprintf(stderr, "driver: unknown option: `%s`, exiting...\n",
                        argv[arg]);
                exit(2);
            }
            dims = str2dims(argv[arg]);
            check_correctness();
        }
    }

    return OK;
}
}
