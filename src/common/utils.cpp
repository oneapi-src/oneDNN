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

#ifdef WIN32
#include <windows.h>
#endif

#include "utils.hpp"

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

FILE *mkldnn_fopen(const char *filename, const char *mode) {
#ifdef _WIN32
    FILE *fp = NULL;
    return fopen_s(&fp, filename, mode) ? NULL : fp;
#else
    return fopen(filename, mode);
#endif
}

}
}
