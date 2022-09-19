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

#include "stdlib.h"

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

// Note: use one non-default value to validate functionality. Rest values, if
// check in loop will not take effect.

namespace {

void custom_setenv(const char *name, const char *value, int overwrite) {
#ifdef _WIN32
    auto status = SetEnvironmentVariable(name, value);
    EXPECT_NE(status, 0);
#else
    auto status = ::setenv(name, value, overwrite);
    EXPECT_EQ(status, 0);
#endif
}

} // namespace

namespace dnnl {

#if DNNL_X64
TEST(onednn_max_cpu_isa_env_var_test, TestEnvVars) {
    custom_setenv("ONEDNN_MAX_CPU_ISA", "SSE41", 1);
    auto got = dnnl_get_effective_cpu_isa();

#if (DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE) && defined(DNNL_ENABLE_MAX_CPU_ISA)
    // Expect set values only for X64...
    EXPECT_EQ(got, dnnl_cpu_isa_sse41);

    auto st = dnnl_set_max_cpu_isa(dnnl_cpu_isa_default);
    EXPECT_EQ(st, dnnl_invalid_arguments);
    auto func_got = dnnl_get_effective_cpu_isa();
    EXPECT_EQ(func_got, dnnl_cpu_isa_sse41);
#else
    // ... while rest should return isa_all
    EXPECT_EQ(got, dnnl_cpu_isa_default);
#endif
}
#endif // DNNL_X64

#if DNNL_X64
TEST(onednn_cpu_isa_hints_var_test, TestEnvVars) {
    custom_setenv("ONEDNN_CPU_ISA_HINTS", "PREFER_YMM", 1);
    auto got = dnnl_get_cpu_isa_hints();

#if (DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE) \
        && defined(DNNL_ENABLE_CPU_ISA_HINTS)
    // Expect set values only for X64...
    EXPECT_EQ(got, dnnl_cpu_isa_prefer_ymm);

    auto st = dnnl_set_cpu_isa_hints(dnnl_cpu_isa_no_hints);
    EXPECT_EQ(st, dnnl_runtime_error);
    auto func_got = dnnl_get_cpu_isa_hints();
    EXPECT_EQ(func_got, dnnl_cpu_isa_prefer_ymm);
#else
    // ... while rest should return isa_all
    EXPECT_EQ(got, dnnl_cpu_isa_no_hints);
#endif
}
#endif // DNNL_X64

TEST(onednn_primitive_cache_capacity_env_var_test, TestEnvVars) {
    custom_setenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "11", 1);
    auto got = get_primitive_cache_capacity();
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    EXPECT_EQ(got, 11);

    set_primitive_cache_capacity(8);
    auto func_got = get_primitive_cache_capacity();
    EXPECT_EQ(func_got, 8);
#else
    EXPECT_EQ(got, 0);
#endif
}

TEST(onednn_default_fpmath_mode_env_var_test, TestEnvVars) {
    custom_setenv("ONEDNN_DEFAULT_FPMATH_MODE", "BF16", 1);
    dnnl_fpmath_mode_t got_val;
    auto st = dnnl_get_default_fpmath_mode(&got_val);
    EXPECT_EQ(st, dnnl_success);
    EXPECT_EQ(got_val, dnnl_fpmath_mode_bf16);

    st = dnnl_set_default_fpmath_mode(dnnl_fpmath_mode_strict);
    EXPECT_EQ(st, dnnl_success);
    dnnl_fpmath_mode_t func_got_val;
    st = dnnl_get_default_fpmath_mode(&func_got_val);
    EXPECT_EQ(st, dnnl_success);
    EXPECT_EQ(func_got_val, dnnl_fpmath_mode_strict);
}

} // namespace dnnl
