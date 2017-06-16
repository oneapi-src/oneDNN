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

#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"

#include "conv/conv.hpp"

#include "conv/input_conv.hpp"

namespace conv {

/* cfgs definition
 * arrays: SRC, WEI, BIA, DST, ACC
 * params: {data_type, min, max, f_min, f_max, f_base, f_step, f_sparsity, eps}
 */

const int int_max_exact = 1<<24;
const _dt_conf_t conf_f32 = {
    {mkldnn_f32, -int_max_exact, int_max_exact,  -64,  64, 0, 1, .25, 0.},
    {mkldnn_f32, -int_max_exact, int_max_exact,  -32,  32, 0, 1, 1.0, 0.},
    {mkldnn_f32, -int_max_exact, int_max_exact, -512, 512, 0, 1, 1.0, 0.},
    {mkldnn_f32, -int_max_exact, int_max_exact,  -64,  64, 0, 1, .25, 0.},
    {mkldnn_f32,},
};

const _dt_conf_t conf_f32_full = {
    {mkldnn_f32, -int_max_exact, int_max_exact,  -64,  64, 0, 1, 1.0, 0.},
    {mkldnn_f32, -int_max_exact, int_max_exact,  -32,  32, 0, 1, 1.0, 0.},
    {mkldnn_f32, -int_max_exact, int_max_exact, -512, 512, 0, 1, 1.0, 0.},
    {mkldnn_f32, -int_max_exact, int_max_exact,  -64,  64, 0, 1, 1.0, 0.},
    {mkldnn_f32,},
};

const _dt_conf_t conf_f32_wino = {
    {mkldnn_f32, -int_max_exact, int_max_exact,   0,  16, 3, 1, .25, 1e-5},
    {mkldnn_f32, -int_max_exact, int_max_exact,   2,  64, 2, 1, .75, 5e-5},
    {mkldnn_f32, -int_max_exact, int_max_exact,   1, 128, 1, 1, .25,   0.},
    {mkldnn_f32, -int_max_exact, int_max_exact,   0,  16, 3, 1, .25, 1e-5},
    {mkldnn_f32,},
};

const _dt_conf_t conf_s16s32 = {
    {mkldnn_s16, INT16_MIN, INT16_MAX, -4,  4, 0, 1, .25, 0.},
    {mkldnn_s16, INT16_MIN, INT16_MAX, -5,  5, 0, 1, .25, 0.},
    {mkldnn_s32, INT32_MIN, INT32_MAX, -8, 32, 0, 1, .25, 0.},
    {mkldnn_s32, INT32_MIN, INT32_MAX, -4,  4, 0, 1, .25, 0.},
    {mkldnn_s32,},
};

const _dt_conf_t conf_s8s32 = {
    {mkldnn_u8,          0, UINT8_MAX,  0,   4, 0, 1, .25, 0.},
    {mkldnn_s8,   INT8_MIN,  INT8_MAX, -5,   5, 0, 1, .25, 0.},
    {mkldnn_s32, INT32_MIN, INT32_MAX, -8,  32, 0, 1, .25, 0.},
    {mkldnn_u8,          0, UINT8_MAX,  0, 255, 0, 1, .25, 0.},
    {mkldnn_s32,},
};

/* global driver parameters */
const dt_conf_t *cfg = conf_f32;
const char *pattern = NULL;
dir_t dir = FWD_B;
int mb = 0;
alg_t alg = DIRECT;
merge_t merge = NONE;
const char *skip_impl = "";
bool allow_unimpl = false;

void reset_parameters() {
    cfg = conf_f32;
    pattern = NULL;
    dir = FWD_B;
    mb = 0;
    alg = DIRECT;
    merge = NONE;
    skip_impl = "";
    allow_unimpl = false;
}

void check_correctness(const desc_t *c) {
    const prb_t p(*c, dir, cfg, alg, merge, mb);
    char pstr[max_prb_len];
    prb2str(&p, pstr);

    if (pattern && !match_regex(pstr, pattern))
        return;
    print(1, "run: %s\n", pstr);

    res_t res{};
    const int status = conv::doit(&p, &res);
    (void)status;

    auto &bs = benchdnn_stat;
    const char *state = state2str(res.state);

    switch (res.state) {
    case UNTESTED:
    case FAILED:
        assert(status == FAIL);
        bs.failed++;
        print(0, "%d:%s (errors:%d total:%d) __REPRO: %s\n", bs.tests, state,
                res.errors, res.total, pstr);
        break;
    case SKIPPED:
        assert(status == OK);
        print(0, "%d:%s __REPRO: %s\n", bs.tests, state, pstr);
        bs.skipped++;
        break;
    case UNIMPLEMENTED:
        assert(status == OK);
        print(0, "%d:%s __REPRO: %s\n", bs.tests, state, pstr);
        bs.unimplemented++;
        bs.failed += !allow_unimpl;
        break;
    case MISTRUSTED:
        assert(status == OK);
        bs.mistrusted++;
        print(0, "%d:%s __REPRO: %s\n", bs.tests, state, pstr);
        // bs.failed++; /* temporal workaround for some tests */
        break;
    case PASSED:
        assert(status == OK);
        print(0, "%d:%s __REPRO: %s\n", bs.tests, state, pstr);
        bs.passed++;
        break;
    default:
        assert(!"unknowtn state");
        { []() { SAFE(FAIL, CRIT); return 0; }(); }
    }

    bs.tests++;
}

int batch(const char *fname);

int bench(int argc, char **argv) {
    for (int arg = 0; arg < argc; ++arg) {
        if (!strncmp("--batch=", argv[arg], 8))
            SAFE(batch(argv[arg] + 8), CRIT);
        else if (!strncmp("--cfg=", argv[arg], 6))
            cfg = str2cfg(argv[arg] + 6);
        else if (!strncmp("--match=", argv[arg], 8))
            pattern = argv[arg] + 8;
        else if (!strncmp("--mb=", argv[arg], 5))
            mb = atoi(argv[arg] + 5);
        else if (!strncmp("--dir=", argv[arg], 6))
            dir = str2dir(argv[arg] + 6);
        else if (!strncmp("--alg=", argv[arg], 6))
            alg = str2alg(argv[arg] + 6);
        else if (!strncmp("--merge=", argv[arg], 8))
            merge = str2merge(argv[arg] + 8);
        else if (!strncmp("--skip-impl=", argv[arg], 12))
            skip_impl = argv[arg] + 12;
        else if (!strncmp("--allow-unimpl=", argv[arg], 15))
            allow_unimpl = str2bool(argv[arg] + 15);
        else if (!strcmp("--reset", argv[arg]))
            reset_parameters();
        else if (!strncmp("-v", argv[arg], 2))
            verbose = atoi(argv[arg] + 2);
        else if (!strncmp("--verbose=", argv[arg], 10))
            verbose = atoi(argv[arg] + 10);
        else {
            desc_t c;
            if (str2desc(&c, argv[arg]) == FAIL) {
                fprintf(stderr, "driver: unknown option: `%s`, exiting...\n",
                        argv[arg]);
                exit(2);
            }
            check_correctness(&c);
        }
    }

    /* deprecated? */
    if (benchdnn_stat.tests == 0) {
        /* use default list of problems */
        int N = sizeof(default_list) / sizeof(default_list[0]);
        for (int n = 0; n < N; ++n)
            check_correctness(&default_list[n]);
    }

    return OK;
}

#include <libgen.h>

FILE *open_batch_file(const char *fname) {
    const int max_paths = 4;

    static int n_paths = 0;
    static char search_paths[max_paths][PATH_MAX] = {{0}};

    char *fdir = NULL;
    {
        char fname_copy[PATH_MAX];
        strncpy(fname_copy, fname, PATH_MAX);
        fdir = dirname(fname_copy);
    }

    bool dir_found = false;
    for (int n = 0; n_paths < max_paths && n < n_paths; ++n)
        if (!strcmp(fdir, search_paths[n])) {
            dir_found = true;
            break;
        }
    if (!dir_found)
        strcpy(search_paths[n_paths++], fdir);

    FILE *fp = fopen(fname, "r");
    if (fp) return fp;

    for (int n = 0; n < n_paths; ++n) {
        char fullname[PATH_MAX];
        snprintf(fullname, PATH_MAX, "%s/%s", search_paths[n], fname);
        fp = fopen(fullname, "r");
        print(50, "batch file used: %s\n", fullname);
        if (fp) break;
    }

    return fp;
}

int batch(const char *fname) {
    FILE *fp = open_batch_file(fname);
    SAFE(fp ? OK : FAIL, CRIT);

    const size_t maxlen = 1024;
    char *opts[8*1024] = {0}, buf[maxlen + 1];
    char line[1024];
    int n_opts = 0;
    while (fgets(line, sizeof(line), fp)) {
        int offset = 0;
        const char *l = line;
        while (sscanf(l, "%s%n", buf, &offset) == 1) {
            if (buf[0] == '#')
                break; /* stop reading till eol */

            const int len = strnlen(buf, maxlen) + 1;
            opts[n_opts] = (char *)malloc(len);
            SAFE(opts[n_opts] ? OK : FAIL, CRIT);
            strncpy(opts[n_opts], buf, len);
            ++n_opts;

            l += offset;
        }
    }
    bench(n_opts, opts);

    for (int n = 0; n < n_opts; ++n)
        free(opts[n]);

    fclose(fp);

    return OK;
}

}
