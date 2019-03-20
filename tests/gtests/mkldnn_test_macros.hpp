/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef MKLDNN_TEST_MACROS_HPP
#define MKLDNN_TEST_MACROS_HPP

#include <iostream>

#include "gtest/gtest.h"

#define TEST_CONCAT_(a, b) a##b
#define TEST_CONCAT(a, b) TEST_CONCAT_(a, b)

#define SKIP_IF(cond, msg)                                      \
    do {                                                        \
        if (cond) {                                             \
            std::cout << "[  SKIPPED ] " << (msg) << std::endl; \
            return;                                             \
        }                                                       \
    } while (0)

#define TEST_F_(test_fixture, test_name) TEST_F(test_fixture, test_name)

#define CPU_TEST_F(test_fixture, test_name) \
    TEST_F_(test_fixture, TEST_CONCAT(test_name, _CPU))

#define GPU_TEST_F(test_fixture, test_name) \
    TEST_F_(test_fixture, TEST_CONCAT(test_name, _GPU))

#define TEST_P_(test_fixture, test_name) TEST_P(test_fixture, test_name)

#define CPU_TEST_P(test_fixture, test_name) \
    TEST_P_(test_fixture, TEST_CONCAT(test_name, _CPU))

#define GPU_TEST_P(test_fixture, test_name) \
    TEST_P_(test_fixture, TEST_CONCAT(test_name, _GPU))

#define INSTANTIATE_TEST_SUITE_P_(prefix, test_case_name, generator) \
    INSTANTIATE_TEST_SUITE_P(prefix, test_case_name, generator)

#define CPU_INSTANTIATE_TEST_SUITE_P(prefix, test_case_name, generator) \
    INSTANTIATE_TEST_SUITE_P_(                                          \
            TEST_CONCAT(prefix, _CPU), test_case_name, generator)

#define GPU_INSTANTIATE_TEST_SUITE_P(prefix, test_case_name, generator) \
    INSTANTIATE_TEST_SUITE_P_(                                          \
            TEST_CONCAT(prefix, _GPU), test_case_name, generator)

#define GPU_INSTANTIATE_TEST_SUITE_P_(prefix, test_case_name, generator) \
    GPU_INSTANTIATE_TEST_SUITE_P(prefix, test_case_name, generator)

#endif
