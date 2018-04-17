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

#include <string.h>
#ifdef _WIN32
#include <malloc.h>
#include <windows.h>
#endif

#include "utils.hpp"

namespace mkldnn {
namespace impl {

int mkldnn_getenv(char *value, const char *name, int length) {
    int result = 0;
    int last_idx = 0;
    if (length > 1) {
        int value_length = 0;
#ifdef _WIN32
        value_length = GetEnvironmentVariable(name, value, length);
        if (value_length >= length) {
            result = -value_length;
        } else {
            last_idx = value_length;
            result = value_length;
        }
#else
        char *buffer = getenv(name);
        if (buffer != NULL) {
            value_length = strlen(buffer);
            if (value_length >= length) {
                result = -value_length;
            } else {
                strncpy(value, buffer, value_length);
                last_idx = value_length;
                result = value_length;
            }
        }
#endif
    }
    value[last_idx] = '\0';
    return result;
}

static bool dump_jit_code;

bool mkldnn_jit_dump() {
    static bool initialized = false;
    if (!initialized) {
        const int len = 2;
        char env_dump[len] = {0};
        dump_jit_code =
            mkldnn_getenv(env_dump, "MKLDNN_JIT_DUMP", len) == 1
            && atoi(env_dump) == 1;
        initialized = true;
    }
    return dump_jit_code;
}

FILE *mkldnn_fopen(const char *filename, const char *mode) {
#ifdef _WIN32
    FILE *fp = NULL;
    return fopen_s(&fp, filename, mode) ? NULL : fp;
#else
    return fopen(filename, mode);
#endif
}

void *malloc(size_t size, int alignment) {
    void *ptr;

#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
    int rc = ptr ? 0 : -1;
#else
    int rc = ::posix_memalign(&ptr, alignment, size);
#endif

    return (rc == 0) ? ptr : 0;
}

void free(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    ::free(p);
#endif
}

}
}
