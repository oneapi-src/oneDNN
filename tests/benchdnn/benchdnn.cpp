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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dnnl.h"

#include "common.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "parser.hpp"

#include "binary/binary.hpp"
#include "bnorm/bnorm.hpp"
#include "concat/concat.hpp"
#include "conv/conv.hpp"
#include "conv/deconv.hpp"
#include "eltwise/eltwise.hpp"
#include "ip/ip.hpp"
#include "lnorm/lnorm.hpp"
#include "lrn/lrn.hpp"
#include "matmul/matmul.hpp"
#include "pool/pool.hpp"
#include "reorder/reorder.hpp"
#include "resampling/resampling.hpp"
#include "rnn/rnn.hpp"
#include "self/self.hpp"
#include "shuffle/shuffle.hpp"
#include "softmax/softmax.hpp"
#include "sum/sum.hpp"

int verbose {0};
bool canonical {false};
bench_mode_t bench_mode {CORR};
stat_t benchdnn_stat {0};
const char *driver_name = "";

double max_ms_per_prb {3e3};
int min_times_per_prb {5};
int fix_times_per_prb {0};

bool fast_ref_gpu {true};

int main(int argc, char **argv) {
    using namespace parser;

    --argc;
    ++argv;

    prim_t prim = DEF;
    for (; argc > 0; --argc, ++argv) {
        if (parse_bench_settings(argv[0]))
            ;
        else if (!strcmp("--self", argv[0]))
            prim = SELF;
        else if (!strcmp("--conv", argv[0]))
            prim = CONV;
        else if (!strcmp("--deconv", argv[0]))
            prim = DECONV;
        else if (!strcmp("--ip", argv[0]))
            prim = IP;
        else if (!strcmp("--shuffle", argv[0]))
            prim = SHUFFLE;
        else if (!strcmp("--reorder", argv[0]))
            prim = REORDER;
        else if (!strcmp("--bnorm", argv[0]))
            prim = BNORM;
        else if (!strcmp("--lnorm", argv[0]))
            prim = LNORM;
        else if (!strcmp("--rnn", argv[0]))
            prim = RNN;
        else if (!strcmp("--softmax", argv[0]))
            prim = SOFTMAX;
        else if (!strcmp("--pool", argv[0]))
            prim = POOL;
        else if (!strcmp("--sum", argv[0]))
            prim = SUM;
        else if (!strcmp("--eltwise", argv[0]))
            prim = ELTWISE;
        else if (!strcmp("--concat", argv[0]))
            prim = CONCAT;
        else if (!strcmp("--lrn", argv[0]))
            prim = LRN;
        else if (!strcmp("--binary", argv[0]))
            prim = BINARY;
        else if (!strcmp("--matmul", argv[0]))
            prim = MATMUL;
        else if (!strcmp("--resampling", argv[0]))
            prim = RESAMPLING;
        else
            break;
    }

    init_fp_mode();
    init();

    switch (prim) {
        case SELF: self::bench(argc, argv); break;
        case CONV: conv::bench(argc, argv); break;
        case DECONV: deconv::bench(argc, argv); break;
        case IP: ip::bench(argc, argv); break;
        case SHUFFLE: shuffle::bench(argc, argv); break;
        case REORDER: reorder::bench(argc, argv); break;
        case BNORM: bnorm::bench(argc, argv); break;
        case LNORM: lnorm::bench(argc, argv); break;
        case RNN: rnn::bench(argc, argv); break;
        case SOFTMAX: softmax::bench(argc, argv); break;
        case POOL: pool::bench(argc, argv); break;
        case SUM: sum::bench(argc, argv); break;
        case ELTWISE: eltwise::bench(argc, argv); break;
        case CONCAT: concat::bench(argc, argv); break;
        case LRN: lrn::bench(argc, argv); break;
        case BINARY: binary::bench(argc, argv); break;
        case MATMUL: matmul::bench(argc, argv); break;
        case RESAMPLING: resampling::bench(argc, argv); break;
        default: fprintf(stderr, "err: unknown driver\n");
    }

    finalize();

    printf("tests:%d passed:%d "
           "skipped:%d mistrusted:%d unimplemented:%d "
           "failed:%d listed:%d\n",
            benchdnn_stat.tests, benchdnn_stat.passed, benchdnn_stat.skipped,
            benchdnn_stat.mistrusted, benchdnn_stat.unimplemented,
            benchdnn_stat.failed, benchdnn_stat.listed);
    if (bench_mode & PERF) {
        printf("total perf: min(ms):%g avg(ms):%g\n",
                benchdnn_stat.ms[benchdnn_timer_t::min],
                benchdnn_stat.ms[benchdnn_timer_t::avg]);
    }

    return !!benchdnn_stat.failed;
}
