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

file(GLOB_RECURSE HEADERS
    ${PROJECT_BINARY_DIR}/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.hpp
    )
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    )

# propagate SRC specific flags
append(CMAKE_C_FLAGS "${CMAKE_SRC_CCXX_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_SRC_CCXX_FLAGS}")

# propagate no warning flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")

# propagate sanitizer flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")

if(NOT DNNL_VERBOSE)
    add_definitions(-DDISABLE_VERBOSE)
endif()

if(DNNL_ENABLE_CONCURRENT_EXEC)
    add_definitions(-DDNNL_ENABLE_CONCURRENT_EXEC)
endif()

if(DNNL_ENABLE_PRIMITIVE_CACHE)
    message(STATUS "Primitive cache is enabled")
    add_definitions(-DDNNL_ENABLE_PRIMITIVE_CACHE)
endif()

if(NOT DNNL_ENABLE_JIT_PROFILING)
    # XXX: the profiling interface will still be built and present in the
    # library
    message(STATUS "Intel(R) VTune(TM) Amplifier JIT profiling disabled")
    add_definitions(-DDNNL_ENABLE_JIT_PROFILING=0)
elseif(UNIX AND NOT APPLE)
    # Not every compiler adds -ldl automatically
    list(APPEND EXTRA_SHARED_LIBS "${CMAKE_DL_LIBS}")
endif()

if(DNNL_ENABLE_MAX_CPU_ISA)
    add_definitions(-DDNNL_ENABLE_MAX_CPU_ISA)
endif()

if(WIN32)
    add_definitions(-D_WIN)
    add_definitions(-DNOMINMAX)
    # Correct 'jnl' macro/jit issue
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
        append(CMAKE_CXX_FLAGS "/Qlong-double")
    endif()
endif()

add_subdirectory(common)
add_subdirectory(compute)
add_subdirectory(cpu)

if(DNNL_GPU_RUNTIME STREQUAL "OCL")
    add_subdirectory(ocl)
endif()

if(DNNL_WITH_SYCL)
    add_subdirectory(ocl)
    add_subdirectory(sycl)
endif()

add_library(${LIB_NAME} ${DNNL_LIBRARY_TYPE}
    ${VERSION_RESOURCE_FILE} ${HEADERS} ${${LIB_NAME}_SUB_OBJS})

set_property(TARGET ${LIB_NAME} PROPERTY VERSION "${DNNL_VERSION_MAJOR}.${DNNL_VERSION_MINOR}")
set_property(TARGET ${LIB_NAME} PROPERTY SOVERSION "${DNNL_VERSION_MAJOR}")
set_property(TARGET ${LIB_NAME} PROPERTY PUBLIC_HEADER ${HEADERS})

target_include_directories(${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>
    # $<INSTALL_PREFIX> is required for compatibility with cmake 2.8
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>
    )

target_link_libraries_build(${LIB_NAME}
    "${EXTRA_SHARED_LIBS};${EXTRA_STATIC_LIBS}")
target_link_libraries_install(${LIB_NAME} "${EXTRA_SHARED_LIBS}")
if(DNNL_LIBRARY_TYPE STREQUAL "STATIC")
    target_link_libraries_install(${LIB_NAME} "${EXTRA_STATIC_LIBS}")
endif()

set(LIB_EXPORT_NAME "${LIB_NAME}-targets")
install(TARGETS ${LIB_NAME}
    EXPORT "${LIB_EXPORT_NAME}"
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# Write version and package config files
set(LIB_CONFIG_GENERATE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(LIB_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${LIB_NAME}")
set(LIB_VERSION_FILE
    "${LIB_CONFIG_GENERATE_DIR}/${LIB_NAME}-config-version.cmake")
set(LIB_CONFIG_FILE
    "${LIB_CONFIG_GENERATE_DIR}/${LIB_NAME}-config.cmake")
write_basic_package_version_file(
    "${LIB_VERSION_FILE}"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
configure_package_config_file(
    "../cmake/config.cmake.in"
    "${LIB_CONFIG_FILE}"
    INSTALL_DESTINATION ${LIB_CONFIG_INSTALL_DIR})
install(FILES ${LIB_CONFIG_FILE} ${LIB_VERSION_FILE}
    DESTINATION ${LIB_CONFIG_INSTALL_DIR})
string(TOUPPER "${LIB_NAME}::" LIB_NAMESPACE)
install(EXPORT ${LIB_EXPORT_NAME}
    NAMESPACE ${LIB_NAMESPACE}
    DESTINATION ${LIB_CONFIG_INSTALL_DIR})

# MKL-DNN compat cmake files
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND}
    -DDIR=${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake
    -P ${PROJECT_SOURCE_DIR}/cmake/gen_mkldnn_compat_cmakes.cmake)")

# MKL-DNN compat libraries
if (WIN32)
    if (NOT MINGW)
        set(prefix "")
        set(ext "lib")
    else()
        set(prefix "lib")
        if (DNNL_LIBRARY_TYPE STREQUAL "SHARED")
            set(ext "dll.a")
        else()
            set(ext "a")
        endif()
    endif()
    add_custom_target(compat_libs ALL
        ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE_DIR:${LIB_NAME}>/${prefix}dnnl.${ext}
        $<TARGET_FILE_DIR:${LIB_NAME}>/${prefix}mkldnn.${ext}
        DEPENDS ${LIB_NAME})
    install(FILES $<TARGET_FILE_DIR:${LIB_NAME}>/${prefix}mkldnn.${ext}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
else()
    if(DNNL_LIBRARY_TYPE STREQUAL "SHARED")
        set_ternary(ext APPLE ".dylib" ".so")
        set(vers ".${DNNL_VERSION_MAJOR};.${DNNL_VERSION_MAJOR}.${DNNL_VERSION_MINOR}")
    else()
        set(ext ".a")
        set(vers "")
    endif()
    foreach(ver "" ${vers})
        set_ternary(ext_and_ver APPLE "${ver}${ext}" "${ext}${ver}")
        add_custom_command(OUTPUT libmkldnn${ext_and_ver}
            # to make the next command work fine
            COMMAND ${CMAKE_COMMAND} -E remove -f libmkldnn${ext_and_ver}
            COMMAND ${CMAKE_COMMAND} -E create_symlink libdnnl${ext_and_ver} libmkldnn${ext_and_ver}
            DEPENDS ${LIB_NAME})
        add_custom_target(compat_libs${ver} ALL
                DEPENDS libmkldnn${ext_and_ver})
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libmkldnn${ext_and_ver}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    endforeach()
endif()

# Install custom find modules for transitive dependencies

if(DNNL_CPU_THREADING_RUNTIME STREQUAL "TBB")
    if(WIN32)
        install(FILES "../cmake/win/TBBConfig.cmake" RENAME "FindTBB.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR})
    elseif(APPLE)
        install(FILES "../cmake/mac/TBBConfig.cmake" RENAME "FindTBB.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR})
    elseif(UNIX)
        install(FILES "../cmake/lnx/TBBConfig.cmake" RENAME "FindTBB.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR})
    endif()
    set(LIB_CONFIG_INSTALL_DIR_COMPAT "${CMAKE_INSTALL_LIBDIR}/cmake/mkldnn")
    install(FILES "${CMAKE_INSTALL_PREFIX}/${LIB_CONFIG_INSTALL_DIR}/FindTBB.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR_COMPAT})
endif()

if(DNNL_GPU_RUNTIME STREQUAL "OCL")
    install(FILES
        "../cmake/FindOpenCL.cmake"
        DESTINATION ${LIB_CONFIG_INSTALL_DIR})
endif()

if(DNNL_WITH_SYCL)
    install(FILES
        "../cmake/FindOpenCL.cmake"
        "../cmake/FindSYCL.cmake"
        "../cmake/FindDPCPP.cmake"
        "../cmake/FindComputeCpp.cmake"
        DESTINATION ${LIB_CONFIG_INSTALL_DIR})
endif()

# On Windows we need to add dnnl.dll path to CTESTCONFIG_PATH which is later
# passed to ctest and Visual Studio solutions
if(WIN32)
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        foreach(BUILD_TYPE Release Debug RelWithDebInfo MinSizeRel)
            append_to_windows_path_list(CTESTCONFIG_PATH
                "${CMAKE_CURRENT_BINARY_DIR}/${BUILD_TYPE}")
        endforeach()
    else()
        append_to_windows_path_list(CTESTCONFIG_PATH
            "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    set(CTESTCONFIG_PATH "${CTESTCONFIG_PATH}" PARENT_SCOPE)
endif()
