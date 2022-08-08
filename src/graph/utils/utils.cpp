/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#ifdef _WIN32
#include <windows.h>
#endif

#include <climits>
#include <cstdlib>
#include <cstring>

#include "graph/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {

int getenv_int_internal(const char *name, int default_value) {
    int value = default_value;
    // # of digits in the longest 32-bit signed int + sign + terminating null
    const int len = 12;
    char value_str[len]; // NOLINT
    for (const auto &prefix : {"_ONEDNN_", "_DNNL_"}) {
        std::string name_str = std::string(prefix) + std::string(name);
        if (getenv(name_str.c_str(), value_str, len) > 0) {
            value = std::atoi(value_str);
            break;
        }
    }
    return value;
}

bool check_verbose_string_user(const char *name, const char *expected) {
    // Random number to fit possible string input.
    std::string value;
    const int len = 64;
    char value_str[len]; // NOLINT
    for (const auto &prefix : {"ONEDNN_", "DNNL_"}) {
        std::string name_str = std::string(prefix) + std::string(name);
        if (getenv(name_str.c_str(), value_str, len) > 0) {
            value = value_str;
            break;
        }
    }
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    std::vector<std::string> splits;
    std::string split;
    std::istringstream ss(value);
    while (std::getline(ss, split, ',')) {
        splits.push_back(split);
    }
    return std::find(splits.begin(), splits.end(), std::string(expected))
            != splits.end();
}

} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl
