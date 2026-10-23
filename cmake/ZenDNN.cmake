#*******************************************************************************
# Copyright 2026 Advanced Micro Devices, Inc.
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
#*******************************************************************************

if(zendnn_cmake_included)
    return()
endif()
set(zendnn_cmake_included true)
include("cmake/options.cmake")

if(NOT DNNL_X64_USE_ZEN)
    add_definitions(-DDNNL_X64_USE_ZEN=0)
    return()
endif()

# ZenDNN does not support Windows builds.
if(WIN32)
    message(FATAL_ERROR
        "ONEDNN_X64_USE_ZEN=ON is not supported on Windows. "
        "Build on Linux, or configure with -DONEDNN_X64_USE_ZEN=OFF.")
endif()

# ZenDNN requires CMake >= 3.26.
if(CMAKE_VERSION VERSION_LESS "3.26")
    message(FATAL_ERROR
        "ONEDNN_X64_USE_ZEN=ON requires CMake >= 3.26. "
        "Current CMake: ${CMAKE_VERSION}. "
        "Upgrade CMake, or configure with -DONEDNN_X64_USE_ZEN=OFF.")
endif()

# ZenDNN uses the OpenMP threading runtime exclusively; any other oneDNN CPU
# runtime (TBB, SEQ, THREADPOOL, ...) is incompatible.
if(NOT DNNL_CPU_RUNTIME STREQUAL "OMP")
    message(FATAL_ERROR
        "ONEDNN_X64_USE_ZEN=ON requires ONEDNN_CPU_RUNTIME=OMP; ZenDNN only "
        "supports the OpenMP threading runtime. Current ONEDNN_CPU_RUNTIME: "
        "${DNNL_CPU_RUNTIME}. Configure with -DONEDNN_CPU_RUNTIME=OMP, or "
        "configure with -DONEDNN_X64_USE_ZEN=OFF.")
endif()

# Require GCC >= 11.2 or Clang >= 14; ZenDNN builds only with GCC/Clang.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.2")
        message(FATAL_ERROR
            "ONEDNN_X64_USE_ZEN=ON requires GCC >= 11.2. "
            "Current C++ compiler: ${CMAKE_CXX_COMPILER_ID} "
            "${CMAKE_CXX_COMPILER_VERSION}. "
            "Upgrade GCC, or configure with -DONEDNN_X64_USE_ZEN=OFF.")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14")
        message(FATAL_ERROR
            "ONEDNN_X64_USE_ZEN=ON requires Clang >= 14. "
            "Current C++ compiler: ${CMAKE_CXX_COMPILER_ID} "
            "${CMAKE_CXX_COMPILER_VERSION}. "
            "Upgrade Clang, or configure with -DONEDNN_X64_USE_ZEN=OFF.")
    endif()
else()
    message(FATAL_ERROR
        "ONEDNN_X64_USE_ZEN=ON requires GCC >= 11.2 or Clang >= 14; ZenDNN does "
        "not support other compilers. Current C++ compiler: "
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}. "
        "Build with GCC or Clang, or configure with -DONEDNN_X64_USE_ZEN=OFF.")
endif()

if(NOT ZENDNNROOT AND DEFINED ENV{ZENDNNROOT})
    set(ZENDNNROOT "$ENV{ZENDNNROOT}")
endif()
if(ZENDNNROOT)
    # Replace an empty, NOTFOUND, or previous-ZENDNNROOT zendnnl_DIR so a
    # changed root takes effect on reconfigure; keep an explicit zendnnl_DIR.
    if(NOT zendnnl_DIR OR "${zendnnl_DIR}" STREQUAL "${_ZENDNNROOT_CMAKE_DIR}")
        set(zendnnl_DIR "${ZENDNNROOT}/lib/cmake" CACHE PATH
            "Path to zendnnl CMake config files" FORCE)
    endif()
    set(_ZENDNNROOT_CMAKE_DIR "${ZENDNNROOT}/lib/cmake" CACHE INTERNAL "")
endif()

# Minimum supported ZenDNN version. A missing or too-old ZenDNN fails configure.
set(ZENDNN_MIN_VERSION "6.1.1")
# find_package replaces a zendnnl_DIR without a package by
# zendnnl_DIR-NOTFOUND, so keep the requested path for the error below.
set(_zendnn_requested_dir "${zendnnl_DIR}")
find_package(zendnnl CONFIG)

if(zendnnl_FOUND)
    if(NOT DEFINED zendnnl_VERSION OR zendnnl_VERSION STREQUAL "")
        message(FATAL_ERROR
            "ONEDNN_X64_USE_ZEN=ON requires ZenDNN >= ${ZENDNN_MIN_VERSION}, but the "
            "ZenDNN package at ${zendnnl_DIR} did not report a version. Use a "
            "ZenDNN package that provides version information, or configure with "
            "-DONEDNN_X64_USE_ZEN=OFF.")
    elseif("${zendnnl_VERSION}" VERSION_LESS "${ZENDNN_MIN_VERSION}")
        message(FATAL_ERROR
            "ONEDNN_X64_USE_ZEN=ON requires ZenDNN >= ${ZENDNN_MIN_VERSION}. "
            "Found ZenDNN ${zendnnl_VERSION} at ${zendnnl_DIR}. "
            "Update ZenDNN, or configure with -DONEDNN_X64_USE_ZEN=OFF.")
    endif()
    message(STATUS "Found ZenDNN ${zendnnl_VERSION}: ${zendnnl_DIR}")
    set(_zendnn_package_dir "${zendnnl_DIR}")
    if(NOT ONEDNN_ZENDNN_SOURCE STREQUAL "${ZENDNN_RELEASE_URL}@${ZENDNN_RELEASE_TAG}")
        message(STATUS "ONEDNN_ZENDNN_SOURCE is ignored because a ZenDNN package "
            "was found. Set CMAKE_DISABLE_FIND_PACKAGE_zendnnl=ON to build from it.")
    endif()
else()
    # ZENDNNROOT or zendnnl_DIR names a package; do not fall back to a source
    # build unless find_package is disabled.
    if(_zendnn_requested_dir AND NOT CMAKE_DISABLE_FIND_PACKAGE_zendnnl)
        message(FATAL_ERROR
            "ONEDNN_X64_USE_ZEN=ON but no ZenDNN package was found in "
            "${_zendnn_requested_dir}. Set ZENDNNROOT or zendnnl_DIR to a "
            "ZenDNN install, or configure with "
            "-DCMAKE_DISABLE_FIND_PACKAGE_zendnnl=ON to build from "
            "ONEDNN_ZENDNN_SOURCE anyway.")
    endif()
    # ONEDNN_ZENDNN_SOURCE is a local source tree or <url>@<tag>, split at the
    # last '@'. Git refs cannot contain ':', which rejects a bare SSH URL.
    if(IS_ABSOLUTE "${ONEDNN_ZENDNN_SOURCE}" AND IS_DIRECTORY "${ONEDNN_ZENDNN_SOURCE}")
        set(_zendnn_source_dir "${ONEDNN_ZENDNN_SOURCE}")
    elseif(ONEDNN_ZENDNN_SOURCE MATCHES "^(.+)@([^@:]+)$")
        set(_zendnn_fetch_url "${CMAKE_MATCH_1}")
        set(_zendnn_fetch_tag "${CMAKE_MATCH_2}")
        # Always go through FetchContent so a changed tag updates an existing
        # checkout. SOURCE_SUBDIR has no CMakeLists.txt, so MakeAvailable only
        # populates; the build is the fwk ExternalProject.
        message(STATUS "Fetching ZenDNN from ${_zendnn_fetch_url} (${_zendnn_fetch_tag})")
        include(FetchContent)
        FetchContent_Declare(ZenDNN
            GIT_REPOSITORY ${_zendnn_fetch_url}
            GIT_TAG        ${_zendnn_fetch_tag}
            SOURCE_DIR     "${PROJECT_BINARY_DIR}/ZenDNN-src"
            SOURCE_SUBDIR  "not-available"
        )
        FetchContent_MakeAvailable(ZenDNN)
        FetchContent_GetProperties(ZenDNN SOURCE_DIR _zendnn_source_dir)
    else()
        message(FATAL_ERROR
            "ONEDNN_ZENDNN_SOURCE=${ONEDNN_ZENDNN_SOURCE} is neither an absolute "
            "path to an existing directory nor <git url>@<tag or commit>.")
    endif()

    if(NOT EXISTS "${_zendnn_source_dir}/fwk/ZenDnnlFwkIntegrate.cmake")
        message(FATAL_ERROR
            "ZenDNN framework integration script not found at "
            "${_zendnn_source_dir}/fwk/ZenDnnlFwkIntegrate.cmake. "
            "Ensure ZenDNN source is complete and up-to-date.")
    endif()

    file(READ "${_zendnn_source_dir}/CMakeLists.txt" ZENDNN_CMAKE_CONTENT)
    string(TOUPPER "${ZENDNN_CMAKE_CONTENT}" ZENDNN_CMAKE_CONTENT_UPPER)
    string(REGEX MATCH "PROJECT\\([^)]*VERSION[ \t]+([0-9]+\\.[0-9]+\\.[0-9]+)"
           ZENDNN_VERSION_MATCH "${ZENDNN_CMAKE_CONTENT_UPPER}")
    if(CMAKE_MATCH_1)
        set(zendnnl_VERSION "${CMAKE_MATCH_1}")
    else()
        set(zendnnl_VERSION "")
    endif()

    if(NOT DEFINED zendnnl_VERSION OR zendnnl_VERSION STREQUAL "")
        message(FATAL_ERROR
            "ONEDNN_X64_USE_ZEN=ON requires ZenDNN >= ${ZENDNN_MIN_VERSION}, but the "
            "ZenDNN source at ${_zendnn_source_dir} did not report a version. Use a "
            "ZenDNN source that provides version information, or configure with "
            "-DONEDNN_X64_USE_ZEN=OFF.")
    elseif("${zendnnl_VERSION}" VERSION_LESS "${ZENDNN_MIN_VERSION}")
        message(FATAL_ERROR
            "ONEDNN_X64_USE_ZEN=ON requires ZenDNN >= ${ZENDNN_MIN_VERSION}. "
            "Found ZenDNN ${zendnnl_VERSION} at ${_zendnn_source_dir}. "
            "Update ZenDNN, or configure with -DONEDNN_X64_USE_ZEN=OFF.")
    endif()

    message(STATUS "Using ZenDNN source at: ${_zendnn_source_dir}")
    message(STATUS "Found ZenDNN version: ${zendnnl_VERSION}")

    set(ZENDNNL_SOURCE_DIR "${_zendnn_source_dir}" CACHE PATH "ZenDNN source directory" FORCE)
    set(ZENDNNL_BINARY_DIR "${PROJECT_BINARY_DIR}/ZenDNN-build" CACHE PATH "ZenDNN build directory" FORCE)
    set(ZENDNNL_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/ZenDNN-install" CACHE PATH "ZenDNN install prefix" FORCE)

    # The fwk reuses an existing install in ZENDNNL_INSTALL_PREFIX without
    # rebuilding, so drop it and the build state when the source changes.
    if(DEFINED _ZENDNN_SOURCE_ID AND NOT "${_ZENDNN_SOURCE_ID}" STREQUAL "${ONEDNN_ZENDNN_SOURCE}")
        message(STATUS "ZenDNN source changed, rebuilding ZenDNN")
        file(REMOVE_RECURSE "${ZENDNNL_BINARY_DIR}" "${ZENDNNL_INSTALL_PREFIX}"
            "${CMAKE_CURRENT_BINARY_DIR}/fwk_zendnnl-prefix")
    endif()
    set(_ZENDNN_SOURCE_ID "${ONEDNN_ZENDNN_SOURCE}" CACHE INTERNAL "")

    set(ZENDNNL_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "ZenDNN build type" FORCE)
    set(ZENDNNL_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "C compiler flags for ZenDNN" FORCE)
    set(ZENDNNL_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "C++ compiler flags for ZenDNN" FORCE)
    set(ZENDNNL_VERBOSE_MAKEFILE "${CMAKE_VERBOSE_MAKEFILE}" CACHE BOOL "Verbose ZenDNN build output" FORCE)

    set(ZENDNNL_DEPENDS_AOCLDLP ON CACHE BOOL "ZenDNN depends on AOCL-DLP" FORCE)
    set(ZENDNNL_DEPENDS_ONEDNN OFF CACHE BOOL "ZenDNN depends on oneDNN" FORCE)
    set(ZENDNNL_DEPENDS_LIBXSMM OFF CACHE BOOL "ZenDNN depends on LibXSMM" FORCE)
    set(ZENDNNL_DEPENDS_PARLOOPER OFF CACHE BOOL "ZenDNN depends on parlooper" FORCE)
    set(ZENDNNL_DEPENDS_FBGEMM OFF CACHE BOOL "ZenDNN depends on FBGEMM" FORCE)

    include("${_zendnn_source_dir}/fwk/ZenDnnlFwkIntegrate.cmake")
    message(STATUS "Found ZenDNN framework integration at ${_zendnn_source_dir}")
    message(STATUS "ZenDNN will be built via ExternalProject at ${ZENDNNL_BINARY_DIR}")
    message(STATUS "ZenDNN dependencies: AOCL-DLP=ON, oneDNN=OFF (avoid circular dep), LibXSMM=OFF, parlooper=OFF, FBGEMM=OFF")
endif()

add_definitions(-DDNNL_X64_USE_ZEN=1)

# Helper function to resolve ZenDNN target name
function(zendnn_resolve_target out_var)
    if(TARGET zendnnl::zendnnl_archive)
        set(${out_var} zendnnl::zendnnl_archive PARENT_SCOPE)
    elseif(TARGET zendnnl::zendnnl)
        set(${out_var} zendnnl::zendnnl PARENT_SCOPE)
    else()
        message(FATAL_ERROR
            "ONEDNN_X64_USE_ZEN=ON but neither zendnnl::zendnnl_archive nor "
            "zendnnl::zendnnl target is available after ZenDNN framework integration.")
    endif()
endfunction()

# Link ZenDNN target into ${target} with given visibility (PRIVATE/PUBLIC/INTERFACE)
function(target_link_zendnnl target visibility)
    zendnn_resolve_target(_zendnnl_target)
    # The source-built target is an alias of the fwk's zendnnl_library, and an
    # export resolves aliases to the real name. $<1:...> keeps the package name.
    target_link_libraries(${target} ${visibility}
        "$<BUILD_INTERFACE:${_zendnnl_target}>"
        "$<INSTALL_INTERFACE:$<1:${_zendnnl_target}>>")

    # Keep ZenDNN and AOCL symbols out of the dynamic symbol table. A static
    # oneDNN passes the option on to the library that links it.
    set(_zendnn_exclude_libs
        "LINKER:--exclude-libs,libzendnnl_archive.a:libaocl-dlp.a:libaoclutils.a:libau_cpuid.a")
    get_target_property(_target_type ${target} TYPE)
    if(_target_type STREQUAL "SHARED_LIBRARY")
        target_link_options(${target} PRIVATE ${_zendnn_exclude_libs})
    elseif(_target_type STREQUAL "STATIC_LIBRARY")
        target_link_options(${target} INTERFACE ${_zendnn_exclude_libs})
    endif()

    # Ensure target waits for ZenDNN ExternalProject to complete
    if(TARGET fwk_zendnnl)
        add_dependencies(${target} fwk_zendnnl)
    endif()
endfunction()

# Sets ZENDNN_CONFIG_HINT for the installed dnnl-config.cmake. A static oneDNN
# exports its ZenDNN link, so a source-built ZenDNN is installed with it;
# skipped when oneDNN is a subproject (PyTorch, ideep), whose install would
# otherwise ship the ZenDNN tree. Call after GNUInstallDirs.
function(zendnn_install_package)
    if(_zendnn_package_dir)
        set(ZENDNN_CONFIG_HINT "${_zendnn_package_dir}" PARENT_SCOPE)
    elseif(DNNL_LIBRARY_TYPE STREQUAL "STATIC" AND PROJECT_IS_TOP_LEVEL)
        set(_dir "${CMAKE_INSTALL_LIBDIR}/zendnn")
        install(DIRECTORY "${ZENDNNL_INSTALL_PREFIX}/zendnnl" "${ZENDNNL_INSTALL_PREFIX}/deps"
            DESTINATION "${_dir}" USE_SOURCE_PERMISSIONS)
        set(_deps "${ZENDNNL_SOURCE_DIR}/dependencies")
        set(_doc "${CMAKE_INSTALL_DATAROOTDIR}/doc/${LIB_PACKAGE_NAME}/zendnn")
        install(FILES "${ZENDNNL_SOURCE_DIR}/LICENSE"
            "${ZENDNNL_SOURCE_DIR}/THIRD-PARTY-NOTICES"
            DESTINATION "${_doc}/zendnn")
        install(FILES "${_deps}/aocldlp/LICENSE" "${_deps}/aocldlp/NOTICES"
            DESTINATION "${_doc}/aocl-dlp")
        install(FILES "${_deps}/aoclutils/License.txt" "${_deps}/aoclutils/NOTICES.txt"
            DESTINATION "${_doc}/aocl-utils")
        install(FILES "${_deps}/json/LICENSE.MIT"
            DESTINATION "${_doc}/nlohmann-json")
        set(ZENDNN_CONFIG_HINT "\${PACKAGE_PREFIX_DIR}/${_dir}/zendnnl/lib/cmake" PARENT_SCOPE)
    endif()
endfunction()
