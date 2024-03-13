#===============================================================================
# Copyright 2016-2019 Intel Corporation
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

cmake_minimum_required(VERSION 2.8.12)

if(POLICY CMP0022)
    cmake_policy(SET CMP0022 NEW)
endif()

# Foo::Bar always refers to an IMPORTED target
if(POLICY CMP0028)
    cmake_policy(SET CMP0028 NEW)
endif()

if(POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

# Enable RPATH on MacOS/OSX
if(POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()

# Do not export symbols from executables
if(POLICY CMP0065)
    cmake_policy(SET CMP0065 NEW)
endif()

# Pass linker flags to try_compile
if(POLICY CMP0056)
    cmake_policy(SET CMP0056 NEW)
endif()

# Always link with full path
if(POLICY CMP0060)
    cmake_policy(SET CMP0060 NEW)
endif()

# Pass compiler flags to try_compile
if(POLICY CMP0066)
    cmake_policy(SET CMP0066 NEW)
endif()

# Use <PackageName>_ROOT env. variable as a prefix
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

# Install rules order
if(POLICY CMP0082)
    cmake_policy(SET CMP0082 NEW)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel RelWithAssert RelWithMDd...")
endif()

set(PROJECT_NAME "oneDNN")
set(PROJECT_FULL_NAME "oneAPI Deep Neural Network Library (oneDNN)")
set(PROJECT_VERSION "3.3.6")

if (CMAKE_VERSION VERSION_LESS 3.0)
    project(${PROJECT_NAME} C CXX)
else()
    cmake_policy(SET CMP0048 NEW)
    project(${PROJECT_NAME} VERSION "${PROJECT_VERSION}" LANGUAGES C CXX)
endif()

if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "oneDNN supports 64 bit platforms only")
endif()

# Set the target architecture.
if(NOT DNNL_TARGET_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
        set(DNNL_TARGET_ARCH "AARCH64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ppc64.*|PPC64.*|powerpc64.*)")
        set(DNNL_TARGET_ARCH "PPC64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(s390x.*|S390X.*)")
        set(DNNL_TARGET_ARCH "S390X")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(rv.*|RV.*|riscv.*|RISCV.*)")
        set(DNNL_TARGET_ARCH "RV64")
    else()
        set(DNNL_TARGET_ARCH "X64")
    endif()
endif()
message(STATUS "DNNL_TARGET_ARCH: ${DNNL_TARGET_ARCH}")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(CMAKE_SRC_CCXX_FLAGS)       # SRC specifics
set(CMAKE_EXAMPLE_CCXX_FLAGS)   # EXAMPLE specifics
set(CMAKE_TEST_CCXX_FLAGS)      # TESTS specifics

string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPERCASE_CMAKE_BUILD_TYPE)

set(LIB_PACKAGE_NAME "dnnl")

include("cmake/dnnl_compat.cmake")

include("cmake/utils.cmake")
include("cmake/options.cmake")
include("cmake/SYCL.cmake")
include("cmake/OpenMP.cmake")
include("cmake/TBB.cmake")
include("cmake/Threadpool.cmake")
include("cmake/OpenCL.cmake")
include("cmake/platform.cmake")
include("cmake/SDL.cmake")
include("cmake/ACL.cmake")
include("cmake/blas.cmake")
include("cmake/doc.cmake")
include("cmake/version.cmake")
include("cmake/coverage.cmake")
include("cmake/build_types.cmake")
include("cmake/testing.cmake")
include("cmake/host_compiler.cmake")
include("cmake/configuring_primitive_list.cmake")

if(UNIX OR MINGW)
    if(CMAKE_VERSION VERSION_LESS "3.1.0")
        # No CMAKE_<lang>_STANDARD, so add directly to CMAKE_<lang>_FLAGS
        # (prepended so the user can override)
        set(CMAKE_C_FLAGS "-std=c99 ${CMAKE_C_FLAGS}")
        # Let SYCL to choose the C++ standard it needs.
        if(NOT DNNL_WITH_SYCL)
            set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
        endif()
    else()
        # CMAKE_<lang>_STANDARD support, so set it to our defaults, unless
        # overridden by the user
        if(NOT DEFINED CMAKE_C_STANDARD)
            set(CMAKE_C_STANDARD 99)
        endif()
        if(NOT DEFINED CMAKE_CXX_STANDARD AND NOT DNNL_WITH_SYCL)
            set(CMAKE_CXX_STANDARD 11)
        endif()

        # Disable -std=gnuXX and -std=gnu++XX
        set(CMAKE_C_EXTENSIONS OFF)
        set(CMAKE_CXX_EXTENSIONS OFF)
    endif()
endif()

# Handle cases when OpenMP runtime is requested but not found: override CPU
# runtime to be sequential
if(DNNL_CPU_RUNTIME STREQUAL "OMP" AND
   DNNL_CPU_THREADING_RUNTIME STREQUAL "SEQ")
    set(DNNL_CPU_RUNTIME "SEQ" CACHE STRING "" FORCE)
endif()

enable_testing()

include_directories_with_host_compiler_before(${PROJECT_SOURCE_DIR}/include)

configure_file(
    "${PROJECT_SOURCE_DIR}/include/oneapi/dnnl/dnnl_config.h.in"
    "${PROJECT_BINARY_DIR}/include/oneapi/dnnl/dnnl_config.h"
)
include_directories_with_host_compiler_before(${PROJECT_BINARY_DIR}/include)

configure_file(
    "${PROJECT_SOURCE_DIR}/README.binary.in"
    "${PROJECT_BINARY_DIR}/README"
)

if(DNNL_INSTALL_MODE MATCHES "^(BUNDLE|BUNDLE_V2)$" AND NOT DEFINED CMAKE_INSTALL_LIBDIR)
    # define CMAKE_INSTALL_LIBDIR as "lib" in the case of bundle
    set(CMAKE_INSTALL_LIBDIR "lib")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)

if(DNNL_INSTALL_MODE STREQUAL "BUNDLE")
    install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_PREFIX})
    install(FILES THIRD-PARTY-PROGRAMS DESTINATION ${CMAKE_INSTALL_PREFIX})
    install(FILES ${PROJECT_BINARY_DIR}/README DESTINATION ${CMAKE_INSTALL_PREFIX})
elseif(DNNL_INSTALL_MODE STREQUAL "BUNDLE_V2")
    install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${LIB_PACKAGE_NAME})
    install(FILES THIRD-PARTY-PROGRAMS DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${LIB_PACKAGE_NAME})
    install(FILES ${PROJECT_BINARY_DIR}/README DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${LIB_PACKAGE_NAME})
else()
    # Cannot use CMAKE_INSTALL_DOCDIR since it uses PROJECT_NAME and not DNNL_LIBRARY_NAME
    install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${LIB_PACKAGE_NAME})
    install(FILES THIRD-PARTY-PROGRAMS DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${LIB_PACKAGE_NAME})
    install(FILES ${PROJECT_BINARY_DIR}/README DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${LIB_PACKAGE_NAME})
endif()
