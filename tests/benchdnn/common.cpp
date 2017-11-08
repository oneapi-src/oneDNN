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

#include <stdint.h>
#include <limits.h>
#include <assert.h>

#include "common.hpp"

const char *bench_mode2str(bench_mode_t mode) {
    const char *modes[] = {
        "MODE_UNDEF", "CORR", "PERF", "CORR+PERF"
    };
    assert((int)mode < 4);
    return modes[(int)mode];
}

bench_mode_t str2bench_mode(const char *str) {
    bench_mode_t mode = MODE_UNDEF;
    if (strchr(str, 'c') || strchr(str, 'C'))
        mode = (bench_mode_t)((int)mode | (int)CORR);
    if (strchr(str, 'p') || strchr(str, 'P'))
        mode = (bench_mode_t)((int)mode | (int)PERF);
    if (mode == MODE_UNDEF)
        []() { SAFE(FAIL, CRIT); return 0; }();
    return mode;
}

/* perf */
#include <chrono>

static inline double ms_now() {
    auto timePointTmp
        = std::chrono::high_resolution_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(timePointTmp).count();
}

#if !defined(BENCHDNN_USE_RDPMC) || defined(_WIN32)
unsigned long long ticks_now() {
    return (unsigned long long)0;
}
#else
unsigned long long ticks_now() {
    unsigned eax, edx, ecx;

    ecx = (1 << 30) + 1;
    __asm__ volatile("rdpmc" : "=a" (eax), "=d" (edx) : "c" (ecx));

    return (unsigned long long)eax | (unsigned long long)edx << 32;
}
#endif

void benchdnn_timer_t::reset() {
    times_ = 0;
    for (int i = 0; i < n_modes; ++i) ticks_[i] = 0;
    ticks_start_ = 0;
    for (int i = 0; i < n_modes; ++i) ms_[i] = 0;
    ms_start_ = 0;

    start();
}

void benchdnn_timer_t::start() {
    ticks_start_ = ticks_now();
    ms_start_ = ms_now();
}

void benchdnn_timer_t::stop() {
    long long d_ticks = ticks_now() - ticks_start_; /* FIXME: overflow? */
    double d_ms = ms_now() - ms_start_;

    ticks_start_ += d_ticks;
    ms_start_ += d_ms;

    ms_[benchdnn_timer_t::min] = times_
        ? MIN2(ms_[benchdnn_timer_t::min], d_ms) : d_ms;
    ms_[benchdnn_timer_t::avg] += d_ms;
    ms_[benchdnn_timer_t::max] = times_
        ? MAX2(ms_[benchdnn_timer_t::min], d_ms) : d_ms;

    ticks_[benchdnn_timer_t::min] = times_
        ? MIN2(ticks_[benchdnn_timer_t::min], d_ticks) : d_ticks;
    ticks_[benchdnn_timer_t::avg] += d_ticks;
    ticks_[benchdnn_timer_t::max] = times_
        ? MAX2(ticks_[benchdnn_timer_t::min], d_ticks) : d_ticks;

    times_++;
}

benchdnn_timer_t &benchdnn_timer_t::operator=(const benchdnn_timer_t &rhs) {
    if (this == &rhs) return *this;
    times_ = rhs.times_;
    for (int i = 0; i < n_modes; ++i) ticks_[i] = rhs.ticks_[i];
    ticks_start_ = rhs.ticks_start_;
    for (int i = 0; i < n_modes; ++i) ms_[i] = rhs.ms_[i];
    ms_start_ = rhs.ms_start_;
    return *this;
}

/* result structure */
const char *state2str(res_state_t state) {
#define CASE(x) if (state == x) return STRINGIFY(x)
    CASE(UNTESTED);
    CASE(PASSED);
    CASE(SKIPPED);
    CASE(MISTRUSTED);
    CASE(UNIMPLEMENTED);
    CASE(FAILED);
#undef CASE
    assert(!"unknown res state");
    return "STATE_UNDEF";
}

void parse_result(res_t &res, bool &want_perf_report, bool allow_unimpl,
        int status, char *pstr) {
    auto &bs = benchdnn_stat;
    const char *state = state2str(res.state);

    switch (res.state) {
    case UNTESTED:
        if (!(bench_mode & CORR)) {
            want_perf_report = true;
            break;
        }
    case FAILED:
        assert(status == FAIL);
        bs.failed++;
        print(0, "%d:%s (errors:%lu total:%lu) __REPRO: %s\n", bs.tests, state,
                (unsigned long)res.errors,
                (unsigned long)res.total, pstr);
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
        want_perf_report = true;
        bs.passed++;
        break;
    default:
        assert(!"unknown state");
        { []() { SAFE(FAIL, CRIT); return 0; }(); }
    }
}

/* misc */
void *zmalloc(size_t size, size_t align) {
    void *ptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, align);
    int rc = ((ptr) ? 0 : errno);
#else
    // TODO. Heuristics: Increasing the size to alignment increases
    // the stability of performance results.
    if (size < align)
        size = align;
    int rc = ::posix_memalign(&ptr, align, size);
#endif /* _WIN32 */
    return rc == 0 ? ptr : 0;
}

void zfree(void *ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    return ::free(ptr);
#endif /* _WIN32 */
}

bool str2bool(const char *str) {
    return !strcasecmp("true", str) || !strcasecmp("1", str);
}

const char *bool2str(bool value) {
    return value ? "true" : "false";
}

#ifdef _WIN32
/* NOTE: this should be supported on linux as well, but currently
 * having issues for ICC170 and Clang*/
#include <regex>

bool match_regex(const char *str, const char *pattern) {
    std::regex re(pattern);
    return std::regex_search(str, re);
}
#else
#include <sys/types.h>
#include <regex.h>

bool match_regex(const char *str, const char *pattern) {
    static regex_t regex;
    static const char *prev_pattern = NULL;
    if (pattern != prev_pattern) {
        if (prev_pattern)
            regfree(&regex);

        if (regcomp(&regex, pattern, 0)) {
            fprintf(stderr, "could not create regex\n");
            return true;
        }

        prev_pattern = pattern;
    }

    return !regexec(&regex, str, 0, NULL, 0);
}
#endif /* _WIN32 */


