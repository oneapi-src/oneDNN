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

#ifdef _WIN32
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

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

#define ABS(a) ((a)>0?(a):(-(a)))

#define MIN2(a,b) ((a)<(b)?(a):(b))
#define MAX2(a,b) ((a)>(b)?(a):(b))

#define MIN3(a,b,c) MIN2(a,MIN2(b,c))
#define MAX3(a,b,c) MAX2(a,MAX2(b,c))

#define STRINGIFy(s) #s
#define STRINGIFY(s) STRINGIFy(s)

#define CONCAt2(a,b) a ## b
#define CONCAT2(a,b) CONCAt2(a,b)

inline void *zmalloc(size_t size, int align) {
    void *ptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, align);
    int rc = ((ptr) ? 0 : errno);
#else
    int rc = ::posix_memalign(&ptr, align, size);
#endif /* _WIN32 */
    return rc == 0 ? ptr : 0;
}
inline void zfree(void *ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    return ::free(ptr);
#endif /* _WIN32 */
}

enum bench_mode_t { MODE_UNDEF = 0x0, CORR = 0x1, PERF = 0x2, };
const char *bench_mode2str(bench_mode_t mode);
bench_mode_t str2bench_mode(const char *str);

extern int verbose;
extern bench_mode_t bench_mode;

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

bool str2bool(const char *str);
const char *bool2str(bool value);

bool match_regex(const char *str, const char *pattern);

/* perf */
extern double max_ms_per_prb; /** maximum time spends per prb in ms */
extern int min_times_per_prb; /** minimal amount of runs per prb */
extern int fix_times_per_prb; /** if non-zero run prb that many times */

struct benchdnn_timer_t {
    enum mode_t { min = 0, avg = 1, max = 2, n_modes };

    benchdnn_timer_t() { reset(); }

    void reset(); /** fully reset the measurements */

    void start(); /** restart timer */
    void stop(); /** stop timer & update statistics */

    void stamp() { stop(); }

    int times() const { return times_; }

    double total_ms() const { return ms_[avg]; }

    double ms(mode_t mode = benchdnn_timer_t::min) const
    { return ms_[mode] / (mode == avg ? times_ : 1); }

    benchdnn_timer_t &operator=(const benchdnn_timer_t &rhs);

    int times_;
    double ms_[n_modes], ms_start_;
};

/* result structure */

struct res_t {
    res_state_t state;
    int errors, total;
    benchdnn_timer_t timer;
};

#endif
