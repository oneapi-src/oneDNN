/*******************************************************************************
* Copyright 2018 Intel Corporation
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
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#include "verbose.hpp"

namespace mkldnn {
namespace impl {

const char *mkldnn_getenv(const char *name) {
#ifdef _WIN32
#   define ENV_BUFLEN 256
    static char value[ENV_BUFLEN];
    int rl = GetEnvironmentVariable(name, value, ENV_BUFLEN);
    if (rl >= ENV_BUFLEN || rl <= 0) value[0] = '\0';
    return value;
#else
    return getenv(name);
#endif
}

const verbose_t *mkldnn_verbose() {
    static verbose_t verbose;
#if !defined(DISABLE_VERBOSE)
    static int initialized = 0;
    if (!initialized) {
        const char *val = mkldnn_getenv("MKLDNN_VERBOSE");
        if (val) verbose.level = atoi(val);
        initialized = 1;
    }
#endif
    return &verbose;
}

double get_msec() {
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    if (frequency.QuadPart == 0)
        QueryPerformanceFrequency(&frequency);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return 1e+3 * now.QuadPart / frequency.QuadPart;
#else
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+3 * time.tv_sec + 1e-3 * time.tv_usec;
#endif
}

}
}
