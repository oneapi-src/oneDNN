#===============================================================================
# Copyright 2021-2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Provides compatibility with oneDNN build options
#===============================================================================

# Sets if DNNL_* var is unset, copy the value from corresponding onednn_* var
macro(dnnl_compat_var dnnl_var onednn_var props)
    if (DEFINED ${onednn_var} AND NOT DEFINED ${dnnl_var})
        if ("${props}" STREQUAL "CACHE STRING")
            set(${dnnl_var} "${${onednn_var}}" CACHE STRING "" FORCE)
        elseif ("${props}" STREQUAL "CACHE BOOL")
            set(${dnnl_var} "${${onednn_var}}" CACHE BOOL "" FORCE)
        else()
            set(${dnnl_var} "${${onednn_var}}")
        endif()
        message(STATUS "DNNL compat: "
            "set ${dnnl_var} to ${onednn_var} with value `${${dnnl_var}}`")
    endif()
endmacro()

set(COMPAT_CACHE_BOOL_VARS
    "EXPERIMENTAL"
    "EXPERIMENTAL_SPARSE"
    "VERBOSE"
    "ENABLE_CONCURRENT_EXEC"
    "ENABLE_PRIMITIVE_CACHE"
    "USE_RT_OBJECTS_IN_PRIMITIVE_CACHE"
    "ENABLE_MAX_CPU_ISA"
    "ENABLE_CPU_ISA_HINTS"
    "BUILD_EXAMPLES"
    "BUILD_TESTS"
    "BUILD_FOR_CI"
    "DEV_MODE"
    "WERROR"
    "ENABLE_JIT_PROFILING"
    "ENABLE_ITT_TASKS"
    "ENABLE_MEM_DEBUG"
    "ENABLE_STACK_CHECKER"
    "AARCH64_USE_ACL"
    )

set(COMPAT_CACHE_STRING_VARS
    "LIBRARY_TYPE"
    "TEST_SET"
    "INSTALL_MODE"
    "CODE_COVERAGE"
    "DPCPP_HOST_COMPILER"
    "LIBRARY_NAME"
    "ENABLE_WORKLOAD"
    "ENABLE_PRIMITIVE"
    "ENABLE_PRIMITIVE_CPU_ISA"
    "ENABLE_PRIMITIVE_GPU_ISA"
    "ARCH_OPT_FLAGS"
    "CPU_RUNTIME"
    "GPU_RUNTIME"
    "GPU_VENDOR"
    "USE_CLANG_SANITIZER"
    "USE_CLANG_TIDY"
    "BLAS_VENDOR"
    )

# Map DNNL_ to ONEDNN_ options

foreach (var ${COMPAT_CACHE_BOOL_VARS})
    dnnl_compat_var("DNNL_${var}" "ONEDNN_${var}" "CACHE BOOL")
endforeach()
dnnl_compat_var(_DNNL_USE_MKL _ONEDNN_USE_MKL "CACHE BOOL")

foreach (var ${COMPAT_CACHE_STRING_VARS})
    dnnl_compat_var("DNNL_${var}" "ONEDNN_${var}" "CACHE STRING")
endforeach()
dnnl_compat_var(_DNNL_TEST_THREADPOOL_IMPL _ONEDNN_TEST_THREADPOOL_IMPL "CACHE STRING")
