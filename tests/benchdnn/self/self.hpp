/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
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

#ifndef SELF_HPP
#define SELF_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sstream>

#include "common.hpp"
#include "dnnl_common.hpp"

namespace self {

#define SELF_CHECK(c, ...) \
    do { \
        if (!(c)) { \
            printf("[%s:%d] '%s' FAILED ==> ", __PRETTY_FUNCTION__, __LINE__, \
                    STRINGIFY(c)); \
            printf(" " __VA_ARGS__); \
            printf("\n"); \
            return FAIL; \
        } \
    } while (0)

#define SELF_CHECK_EQ(a, b) \
    SELF_CHECK((a) == (b), "%d != %d", (int)(a), (int)(b))
#define SELF_CHECK_NE(a, b) \
    SELF_CHECK((a) != (b), "%d == %d", (int)(a), (int)(b))
#define SELF_CHECK_CASE_STR_EQ(a, b) \
    SELF_CHECK(!strcasecmp(a, b), "'%s' != '%s'", a, b)
#define SELF_CHECK_CASE_STR_NE(a, b) \
    SELF_CHECK(strcasecmp(a, b), "'%s' == '%s'", a, b)
#define SELF_CHECK_CASE_CPP_STR_EQ(a, b) \
    SELF_CHECK(!strcasecmp(a.c_str(), b), "'%s' != '%s'", a.c_str(), b)
#define SELF_CHECK_CASE_CPP_STR_NE(a, b) \
    SELF_CHECK(strcasecmp(a.c_str(), b), "'%s' == '%s'", a.c_str(), b)
#define SELF_CHECK_PRINT_EQ2(obj, expect_str1, expect_str2) \
    do { \
        std::stringstream ss; \
        ss << obj; \
        std::string obj_str = ss.str(); \
        if (std::string(expect_str1) == std::string(expect_str2) \
                && strcasecmp(obj_str.c_str(), expect_str1)) \
            SELF_CHECK(false, "Expected '%s', got '%s'", expect_str1, \
                    obj_str.c_str()); \
        else if (strcasecmp(obj_str.c_str(), expect_str1) \
                && strcasecmp(obj_str.c_str(), expect_str2)) \
            SELF_CHECK(false, "Expected one of ('%s', '%s'), got '%s'", \
                    expect_str1, expect_str2, obj_str.c_str()); \
    } while (0)
#define SELF_CHECK_PRINT_EQ(obj, expect_str) \
    SELF_CHECK_PRINT_EQ2(obj, expect_str, expect_str)

#define RUN(f) \
    do { \
        BENCHDNN_PRINT(1, "%s ...\n", STRINGIFY(f)); \
        int rc = f; \
        benchdnn_stat.tests++; \
        if (rc == OK) \
            benchdnn_stat.passed++; \
        else \
            benchdnn_stat.failed++; \
    } while (0)

void common();
void res();
void conv();
void bnorm();
void memory();
void graph();

int bench(int argc, char **argv);

} // namespace self

#endif
