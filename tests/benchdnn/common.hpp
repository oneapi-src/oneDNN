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

#ifndef _COMMON_HPP
#define _COMMON_HPP

#include <assert.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#define OK 0
#define FAIL 1

enum { CRIT = 1, WARN = 2 };

#define SAFE(f, s) do { \
    int status = f; \
    if (status != OK) { \
        if (s == CRIT || s == WARN) { \
            fflush(0), fprintf(stderr, "@@@ error [%s:%d]: '%s' -> %d\n", \
                    __PRETTY_FUNCTION__, __LINE__, \
                    #f, status), fflush(0); \
            if (s == CRIT) exit(1); \
        } \
        return status; \
    } \
} while(0)

#define MIN2(a,b) ((a)<(b)?(a):(b))
#define MAX2(a,b) ((a)>(b)?(a):(b))

#define MIN3(a,b,c) MIN2(a,MIN2(b,c))
#define MAX3(a,b,c) MAX2(a,MAX2(b,c))

#define STRINGIFy(s) #s
#define STRINGIFY(s) STRINGIFy(s)

#define CONCAt2(a,b) a ## b
#define CONCAT2(a,b) CONCAt2(a,b)

inline void *zmalloc(size_t size, int align) {
    void *p;
    int rc = ::posix_memalign(&p, align, size);
    return rc == 0 ? p : 0;
}
inline void zfree(void *ptr) { return ::free(ptr); }

extern int verbose;

#define print(v, fmt, ...) do { \
    if (verbose >= v) { \
        printf(fmt, __VA_ARGS__); \
        /* printf("[%d][%s:%d]" fmt, v, __func__, __LINE__, __VA_ARGS__); */ \
        fflush(0); \
    } \
} while (0)

struct stat_t {
    int tests;
    int passed;
    int failed;
    int skipped;
    int mistrusted;
    int unimplemented;
};
extern stat_t benchdnn_stat;

enum prim_t {
    CONV, IP, DEF = CONV,
};

enum dir_t {
    DIR_UNDEF = 0,
    FLAG_DAT = 1, FLAG_WEI = 2, FLAG_BIA = 4,
    FLAG_FWD = 32, FLAG_BWD = 64,
    FWD_D = FLAG_FWD + FLAG_DAT,
    FWD_B = FLAG_FWD + FLAG_DAT + FLAG_BIA,
    BWD_D = FLAG_BWD + FLAG_DAT,
    BWD_W = FLAG_BWD + FLAG_WEI,
    BWD_WB = FLAG_BWD + FLAG_WEI + FLAG_BIA,
};
dir_t str2dir(const char *str);
const char *dir2str(dir_t dir);

enum res_state_t { UNTESTED = 0, PASSED, SKIPPED, MISTRUSTED, UNIMPLEMENTED,
    FAILED };
const char *state2str(res_state_t state);

struct res_t {
    res_state_t state;
    int errors, total;
};

bool str2bool(const char *str);
const char *bool2str(bool value);

bool match_regex(const char *str, const char *pattern);

#endif
