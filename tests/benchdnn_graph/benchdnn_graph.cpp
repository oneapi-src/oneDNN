/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include <cstdio>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "utils/parser.hpp"

#include "mha/mha.hpp"

int verbose {0};
bool canonical {false};
bool mem_check {true};
std::string skip_impl;
bench_mode_t bench_mode {CORR};
api_mode_t api_mode {GRAPH};
stat_t benchdnn_stat {0};
const char *driver_name = "";

double max_ms_per_prb {3e3};
int min_times_per_prb {5};
int fix_times_per_prb {0};

int ctimes_per_prb {1000};

bool fast_ref_gpu {DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE};

bool allow_enum_tags_only {true};
int test_start {0};
bool attr_same_pd_check {false};

int main(int argc, char **argv) {
    using namespace parser;

    if (argc < 2) {
        fprintf(stderr, "err: no arguments passed\n");
        return 1;
    }
    --argc;
    ++argv;

    init_fp_mode();
    for (; argc > 0; --argc, ++argv)
        if (!parse_bench_settings(argv[0])) break;

    if (!strcmp("--mha", argv[0])) {
        mha::bench(--argc, ++argv);
    } else {
        fprintf(stderr, "err: unknown driver\n");
    }
}
