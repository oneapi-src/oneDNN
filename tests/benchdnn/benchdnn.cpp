/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
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
#include <string.h>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/parser.hpp"

#include "binary/binary.hpp"
#include "bnorm/bnorm.hpp"
#include "brgemm/brgemm.hpp"
#include "concat/concat.hpp"
#include "conv/conv.hpp"
#include "deconv/deconv.hpp"
#include "eltwise/eltwise.hpp"
#include "ip/ip.hpp"
#include "lnorm/lnorm.hpp"
#include "lrn/lrn.hpp"
#include "matmul/matmul.hpp"
#include "pool/pool.hpp"
#include "prelu/prelu.hpp"
#include "reduction/reduction.hpp"
#include "reorder/reorder.hpp"
#include "resampling/resampling.hpp"
#include "rnn/rnn.hpp"
#include "self/self.hpp"
#include "shuffle/shuffle.hpp"
#include "softmax/softmax.hpp"
#include "sum/sum.hpp"
#include "zeropad/zeropad.hpp"

#ifdef BUILD_GRAPH
#include "graph/graph.hpp"
#endif

int verbose {0};
bool canonical {false};
bool mem_check {true};
std::string skip_impl;
stat_t benchdnn_stat {0};
std::string driver_name;

double max_ms_per_prb {default_max_ms_per_prb};
double default_max_ms_per_prb {3e3};
int min_times_per_prb {5};
int fix_times_per_prb {default_fix_times_per_prb};
int default_fix_times_per_prb {0};
int repeats_per_prb {default_repeats_per_prb};
int default_repeats_per_prb {1};

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

    if (parse_main_help(argv[0])) return 0;

    init_fp_mode();

    for (; argc > 0; --argc, ++argv)
        if (!parse_bench_settings(argv[0])) break;

    if (!strcmp("--self", argv[0])) {
        self::bench(--argc, ++argv);
    } else if (!strcmp("--conv", argv[0])) {
        conv::bench(--argc, ++argv);
    } else if (!strcmp("--deconv", argv[0])) {
        deconv::bench(--argc, ++argv);
    } else if (!strcmp("--ip", argv[0])) {
        ip::bench(--argc, ++argv);
    } else if (!strcmp("--shuffle", argv[0])) {
        shuffle::bench(--argc, ++argv);
    } else if (!strcmp("--reorder", argv[0])) {
        reorder::bench(--argc, ++argv);
    } else if (!strcmp("--bnorm", argv[0])) {
        bnorm::bench(--argc, ++argv);
    } else if (!strcmp("--lnorm", argv[0])) {
        lnorm::bench(--argc, ++argv);
    } else if (!strcmp("--rnn", argv[0])) {
        rnn::bench(--argc, ++argv);
    } else if (!strcmp("--softmax", argv[0])) {
        softmax::bench(--argc, ++argv);
    } else if (!strcmp("--pool", argv[0])) {
        pool::bench(--argc, ++argv);
    } else if (!strcmp("--prelu", argv[0])) {
        prelu::bench(--argc, ++argv);
    } else if (!strcmp("--sum", argv[0])) {
        sum::bench(--argc, ++argv);
    } else if (!strcmp("--eltwise", argv[0])) {
        eltwise::bench(--argc, ++argv);
    } else if (!strcmp("--concat", argv[0])) {
        concat::bench(--argc, ++argv);
    } else if (!strcmp("--lrn", argv[0])) {
        lrn::bench(--argc, ++argv);
    } else if (!strcmp("--binary", argv[0])) {
        binary::bench(--argc, ++argv);
    } else if (!strcmp("--matmul", argv[0])) {
        matmul::bench(--argc, ++argv);
    } else if (!strcmp("--resampling", argv[0])) {
        resampling::bench(--argc, ++argv);
    } else if (!strcmp("--reduction", argv[0])) {
        reduction::bench(--argc, ++argv);
    } else if (!strcmp("--zeropad", argv[0])) {
        zeropad::bench(--argc, ++argv);
    } else if (!strcmp("--brgemm", argv[0])) {
        brgemm::bench(--argc, ++argv);
#ifdef BUILD_GRAPH
    } else if (!strcmp("--graph", argv[0])) {
        graph::bench(--argc, ++argv);
#endif
    } else {
        fprintf(stderr, "err: unknown driver\n");
    }

    printf("tests:%d passed:%d skipped:%d mistrusted:%d unimplemented:%d "
           "invalid_arguments:%d failed:%d listed:%d\n",
            benchdnn_stat.tests, benchdnn_stat.passed, benchdnn_stat.skipped,
            benchdnn_stat.mistrusted, benchdnn_stat.unimplemented,
            benchdnn_stat.invalid_arguments, benchdnn_stat.failed,
            benchdnn_stat.listed);
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        const auto &perf_timer
                = benchdnn_stat.ms.find(timer::names::perf_timer);
        if (perf_timer != benchdnn_stat.ms.end()) {
            const auto &perf_timer_stats = perf_timer->second;
            printf("total perf: min(ms):%g avg(ms):%g\n",
                    perf_timer_stats[timer::timer_t::min],
                    perf_timer_stats[timer::timer_t::avg]);
        }
    }
    if (has_bench_mode_bit(mode_bit_t::corr)) {
        const auto &compute_ref_timer
                = benchdnn_stat.ms.find(timer::names::ref_timer);
        if (compute_ref_timer != benchdnn_stat.ms.end()) {
            const auto &compute_ref_timer_stats = compute_ref_timer->second;
            printf("total compute_ref: sum(s):%.2f\n",
                    compute_ref_timer_stats[timer::timer_t::sum]);
        }
    }

    finalize();

    return !!benchdnn_stat.failed;
}
