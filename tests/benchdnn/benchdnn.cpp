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

stat_t benchdnn_stat {0};
int verbose {0};

int main(int argc, char **argv) {
    prim_t prim = DEF;
    --argc; ++argv;

    while (true) {
        if (!strcmp("--conv", argv[0])) prim = CONV;
        else if (!strcmp("--ip", argv[0])) prim = IP;
        else if (!strncmp("-v", argv[0], 2)) verbose = atoi(argv[0] + 2);
        else if (!strncmp("--verbose=", argv[0], 10))
            verbose = atoi(argv[0] + 10);
        else break;

        --argc;
        ++argv;
    }

    init();

    switch (prim) {
    case CONV: conv::bench(argc, argv); break;
    case IP: ip::bench(argc, argv); break;
    default: fprintf(stderr, "err: unknown driver\n");
    }

    finalize();

    printf("tests:%d passed:%d "
            "skipped:%d mistrusted:%d unimplemented:%d "
            "failed:%d\n",
            benchdnn_stat.tests, benchdnn_stat.passed,
            benchdnn_stat.skipped, benchdnn_stat.mistrusted,
            benchdnn_stat.unimplemented, benchdnn_stat.failed);

    assert(benchdnn_stat.tests <= benchdnn_stat.passed + benchdnn_stat.skipped
            + benchdnn_stat.mistrusted + benchdnn_stat.unimplemented
            + benchdnn_stat.failed);

    return !!benchdnn_stat.failed;
}
