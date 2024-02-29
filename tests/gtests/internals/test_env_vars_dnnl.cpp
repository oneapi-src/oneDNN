/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#include "tests/test_isa_common.hpp"

// Note: use one non-default value to validate functionality.

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

void custom_unsetenv(const char *name) {
#ifdef _WIN32
    _putenv((std::string(name) + "=").c_str());
#else
    ::unsetenv(name);
#endif
}

} // namespace

namespace dnnl {

#if DNNL_X64
TEST(dnnl_max_cpu_isa_env_var_test, TestEnvVars) {
    const bool has_cpu = DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE;

    custom_setenv("DNNL_MAX_CPU_ISA", "SSE41", 1);
    auto got = dnnl_get_effective_cpu_isa();
    (void)got;

#if defined(DNNL_ENABLE_MAX_CPU_ISA)
    // Expect env var value to be set when env variable feature is enabled.
    EXPECT_EQ(got, has_cpu ? dnnl_cpu_isa_sse41 : dnnl_cpu_isa_default);
#elif DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    // Native SSE41 will issue an error. Don't check for it.
    if (mayiuse(impl::cpu::x64::avx)) {
        // Otherwise, don't expect it to be set.
        EXPECT_NE(got, dnnl_cpu_isa_sse41);
    }
#endif

    if (has_cpu) {
        // `dnnl_get_effective_cpu_isa` freezes the isa value, any call to set
        // it again results in invalid_arguments.
        auto st = dnnl_set_max_cpu_isa(dnnl_cpu_isa_sse41);
        EXPECT_EQ(st, dnnl_invalid_arguments);
    }
    // Check that second pass of env var doesn't take any effect.
    custom_setenv("DNNL_MAX_CPU_ISA", "AVX", 1);
    got = dnnl_get_effective_cpu_isa();
#if defined(DNNL_ENABLE_MAX_CPU_ISA)
    EXPECT_EQ(got, has_cpu ? dnnl_cpu_isa_sse41 : dnnl_cpu_isa_default);
#elif DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    if (mayiuse(impl::cpu::x64::avx2)) {
        EXPECT_NE(got, dnnl_cpu_isa_sse41);
        EXPECT_NE(got, dnnl_cpu_isa_avx);
    }
#endif
}
#endif // DNNL_X64

#if DNNL_X64
TEST(dnnl_cpu_isa_hints_var_test, TestEnvVars) {
    const bool has_cpu = DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE;
    (void)has_cpu;

    custom_setenv("DNNL_CPU_ISA_HINTS", "PREFER_YMM", 1);
    auto got = dnnl_get_cpu_isa_hints();

#if defined(DNNL_ENABLE_CPU_ISA_HINTS)
    // Expect env var value to be set when env variable feature is enabled.
    EXPECT_EQ(got, has_cpu ? dnnl_cpu_isa_prefer_ymm : dnnl_cpu_isa_no_hints);
#else
    // Otherwise, don't expect it to be set.
    EXPECT_NE(got, dnnl_cpu_isa_prefer_ymm);
#endif

#if (DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE)
    // `dnnl_get_cpu_isa_hints` freezes the hints value, any call to set it
    // again results in runtime_error.
    auto st = dnnl_set_cpu_isa_hints(dnnl_cpu_isa_no_hints);
    EXPECT_EQ(st, dnnl_runtime_error);
#endif
}
#endif // DNNL_X64

TEST(dnnl_primitive_cache_capacity_env_var_test, TestEnvVars) {
    // Since variables with "ONEDNN_" prefix have higher precedence
    // we have to unset "ONEDNN_PRIMITIVE_CACHE_CAPACITY" that could
    // be potentially set.
    custom_unsetenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY");
    custom_setenv("DNNL_PRIMITIVE_CACHE_CAPACITY", "10", 1);
    auto got = get_primitive_cache_capacity();
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    EXPECT_EQ(got, 10);

    set_primitive_cache_capacity(9);
    auto func_got = get_primitive_cache_capacity();
    EXPECT_EQ(func_got, 9);
#else
    EXPECT_EQ(got, 0);
#endif
}

TEST(dnnl_default_fpmath_mode_env_var_test, TestEnvVars) {
    custom_setenv("DNNL_DEFAULT_FPMATH_MODE", "ANY", 1);
    dnnl_fpmath_mode_t got_val;
    auto st = dnnl_get_default_fpmath_mode(&got_val);
    EXPECT_EQ(st, dnnl_success);
    EXPECT_EQ(got_val, dnnl_fpmath_mode_any);

    st = dnnl_set_default_fpmath_mode(dnnl_fpmath_mode_strict);
    EXPECT_EQ(st, dnnl_success);
    dnnl_fpmath_mode_t func_got_val;
    st = dnnl_get_default_fpmath_mode(&func_got_val);
    EXPECT_EQ(st, dnnl_success);
    EXPECT_EQ(func_got_val, dnnl_fpmath_mode_strict);
}

// There's no a separate test for VERBOSE variable as there's no programmable
// public API to identify if it was set through env var or not.
// Same situation with the rest of variables.

} // namespace dnnl
