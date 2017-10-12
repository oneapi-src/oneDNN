/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include "perf.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "mkldnn.h"

#include "common.hpp"
#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"

#include "conv/conv.hpp"
#include "ip/ip.hpp"

#if defined(_OPENMP)
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_num_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline int omp_in_parallel() { return 0; }
#endif

int verbose {0};
bench_mode_t bench_mode {CORR};
stat_t benchdnn_stat {0};

double max_ms_per_prb {3e3};
int min_times_per_prb {5};
int fix_times_per_prb {0};

int main(int argc, char **argv) {
    prim_t prim = DEF;
    --argc; ++argv;

    while (argc > 0) {
        if (!strcmp("--conv", argv[0])) prim = CONV;
        else if (!strcmp("--ip", argv[0])) prim = IP;
        else if (!strncmp("--mode=", argv[0], 7))
            bench_mode = str2bench_mode(argv[0] + 7);
        else if (!strncmp("-v", argv[0], 2))
            verbose = atoi(argv[0] + 2);
        else if (!strncmp("--verbose=", argv[0], 10))
            verbose = atoi(argv[0] + 10);
        else break;

        --argc;
        ++argv;
    }

    int omp_max_thr = omp_get_max_threads();
    printf("benchdnn init ... omp_max_thr=%d",omp_max_thr); fflush(stdout);
    init();
    printf(" OK\n"); fflush(stdout);
    perf_t const * perf_data = perf_begin();
    if(perf_data == nullptr) { // [ejk] may need some timing system init
        printf("ERROR: perf_begin failed!\n");
        exit(-1);
    }

    switch (prim) {
    case CONV: conv::bench(argc, argv); break;
    case IP: ip::bench(argc, argv); break;
    default: fprintf(stderr, "err: unknown driver\n");
    }

    perf_end(perf_data);
    finalize();

    printf("tests:%d impls:%d %s:%d "
            "skipped:%d mistrusted:%d unimplemented:%d "
            "failed:%d\n",
            benchdnn_stat.tests, benchdnn_stat.impls,
            (bench_mode&CORR? "correct": "passed"), benchdnn_stat.passed,
            benchdnn_stat.skipped, benchdnn_stat.mistrusted,
            benchdnn_stat.unimplemented, benchdnn_stat.failed);

    return !!benchdnn_stat.failed;
}
