/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#pragma once

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace utils {

inline FILE * openFile(const char * file_name, const char * mode) {
    FILE * f = fopen(file_name, mode);
    if(f == nullptr)
        throw std::runtime_error(std::string("Failed to open file ") +
                                 file_name + ": " + strerror(errno));

    return f;
}

template <typename Container>
inline void readFile(Container &out, FILE *file)
{
    const int chunk = 65536;
    size_t read = 0;

    out.clear();
    do {
        auto osize = out.size();
        out.resize(osize + chunk);
        read = fread(&out[osize], 1, chunk, file);
        auto status = ferror(file);
        if (status)
            throw std::runtime_error(std::string("Failed to read file: ") + strerror(status));
        out.resize(osize + read);
    } while (read > 0);
}

template <typename T>
inline T readValueFromFile(const std::string &fileName, T defaultValue)
{
    std::ifstream s(fileName);
    if (!s) return defaultValue;

    T value = defaultValue;
    s >> value;

    if (!s) return defaultValue;
    return value;
}

inline int setenv(const char* name, const char* value, int overwrite) {
#ifdef _WIN32
    // Check if variable exists and we're not supposed to overwrite
    if (!overwrite) {
        const char* existing = std::getenv(name);
        if (existing != nullptr) {
            return 0; // Already set, and overwrite not requested
        }
    }
    return _putenv_s(name, value);
#else
    return ::setenv(name, value, overwrite);
#endif
}

inline int unsetenv(const char* name) {
#ifdef _WIN32
    return _putenv_s(name, "");
#else
    return ::unsetenv(name);
#endif
}

inline int strcasecmp(const char* a, const char* b) {
#ifdef _WIN32
    return _stricmp(a, b);
#else
    return ::strcasecmp(a, b);
#endif
}

    
} // namespace utils
