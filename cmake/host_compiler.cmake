#===============================================================================
# Copyright 2021-2024 Intel Corporation
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

if(host_compiler_cmake_included)
    return()
endif()
set(host_compiler_cmake_included true)

# There is nothing to do for the default host compiler.
if(DPCPP_HOST_COMPILER_KIND STREQUAL "DEFAULT")
    return()
endif()

if(DPCPP_HOST_COMPILER_KIND STREQUAL "GNU")
    set(DPCPP_HOST_COMPILER_OPTS)

    if(DNNL_TARGET_ARCH STREQUAL "X64")
        if(DNNL_ARCH_OPT_FLAGS STREQUAL "HostOpts")
            platform_gnu_x64_arch_ccxx_flags(DPCPP_HOST_COMPILER_OPTS)
        else()
            # Assumption is that the passed flags are compatible with GNU compiler
            append(DPCPP_HOST_COMPILER_OPTS "${DNNL_ARCH_OPT_FLAGS}")
        endif()
    else()
        message(FATAL_ERROR "The DNNL_DPCPP_HOST_COMPILER option is only supported for DNNL_TARGET_ARCH=X64")
    endif()

    platform_unix_and_mingw_common_ccxx_flags(DPCPP_HOST_COMPILER_OPTS)
    platform_unix_and_mingw_common_cxx_flags(DPCPP_HOST_COMPILER_OPTS)

    sdl_unix_common_ccxx_flags(DPCPP_HOST_COMPILER_OPTS)
    sdl_gnu_common_ccxx_flags(DPCPP_HOST_COMPILER_OPTS)
    sdl_gnu_src_ccxx_flags(DPCPP_SRC_CXX_FLAGS)
    sdl_gnu_example_ccxx_flags(DPCPP_EXAMPLE_CXX_FLAGS)

    # SYCL uses C++17 features in headers hence C++17 support should be enabled
    # for host compiler.
    # The main compiler driver doesn't automatically specify C++ standard for
    # custom host compilers.
    append(DPCPP_HOST_COMPILER_OPTS "-std=c++17")

    # Unconditionally enable OpenMP during compilation to use `#pragma omp simd`
    append(DPCPP_HOST_COMPILER_OPTS "-fopenmp")

    if(UPPERCASE_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
        append(DPCPP_HOST_COMPILER_OPTS "${CMAKE_CXX_FLAGS_RELEASE}")
    else()
        append(DPCPP_HOST_COMPILER_OPTS "${CMAKE_CXX_FLAGS_DEBUG}")
    endif()

    # When a custom host compiler is used some deprecation warnings come
    # from sycl.hpp header. Suppress the warnings for now.
    append(DPCPP_HOST_COMPILER_OPTS "-Wno-deprecated-declarations")

    # SYCL headers contain some comments that trigger warning with GNU compiler
    append(DPCPP_HOST_COMPILER_OPTS "-Wno-comment")

    # Using single_task, cgh.copy, cgh.fill may cause the following warning:
    # "warning: ‘clang::sycl_kernel’ scoped attribute directive ignored [-Wattributes]"
    # We don't have control over it so just suppress it for the time being.
    append(DPCPP_HOST_COMPILER_OPTS "-Wno-attributes")

    # Host compiler operates on preprocessed files and headers, and it
    # mistakenly assumes that anonymous namespace types are used from a header
    # which is not always the case.
    append(DPCPP_HOST_COMPILER_OPTS "-Wno-subobject-linkage")

    platform_gnu_nowarn_ccxx_flags(DPCPP_CXX_NOWARN_FLAGS ${DPCPP_HOST_COMPILER_MAJOR_VER}.${DPCPP_HOST_COMPILER_MINOR_VER})

    append(CMAKE_CXX_FLAGS "-fsycl-host-compiler=${DPCPP_HOST_COMPILER}")
    append_host_compiler_options(CMAKE_CXX_FLAGS "${DPCPP_HOST_COMPILER_OPTS}")

    # When using a non-default host compiler the main compiler doesn't
    # handle some arguments properly and issues the warning.
    # Suppress the warning until the bug is fixed.
    append(CMAKE_CXX_FLAGS "-Wno-unused-command-line-argument")
elseif(NOT DNNL_DPCPP_HOST_COMPILER STREQUAL "DEFAULT")
    message(FATAL_ERROR "The valid values for DNNL_DPCPP_HOST_COMPILER: DEFAULT or an executable of the GNU C++ compiler or an absolute path to it")
endif()
