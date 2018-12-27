#===============================================================================
# Copyright 2016-2018 Intel Corporation
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

set(TARGET_NAME ${LIB_NAME})

file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.hpp
    )
file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    )
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/xbyak
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

if(NOT MKLDNN_VERBOSE)
    add_definitions(-DDISABLE_VERBOSE)
endif()

if(MKLDNN_ENABLE_CONCURRENT_EXEC)
    add_definitions(-DMKLDNN_ENABLE_CONCURRENT_EXEC)
endif()

if(VTUNEROOT)
    include_directories(${VTUNEROOT}/include)
    add_definitions(-DJIT_PROFILING_VTUNE)
endif()

if(WIN32)
    add_definitions(-D_WIN)
    add_definitions(-DNOMINMAX)
    # Correct 'jnl' macro/jit issue
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
        append(CMAKE_CXX_FLAGS "/Qlong-double")
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # to make computations more stable and to align the jitted code
    # with the reference one use precise division and square root
    # by default
    file(GLOB FILES_REQUIRED_PREC_SQRT
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu/*normalization*.cpp)
    file(GLOB FILES_REQUIRED_PREC_DIV
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu/*normalization*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu/ref_rnn.cpp)
    if(WIN32)
        set_source_files_properties(${FILES_REQUIRED_PREC_SQRT}
            PROPERTIES COMPILE_FLAGS "/Qprec-sqrt")
        set_source_files_properties(${FILES_REQUIRED_PREC_DIV}
            PROPERTIES COMPILE_FLAGS "/Qprec-div")
    else()
        set_source_files_properties(${FILES_REQUIRED_PREC_SQRT}
            PROPERTIES COMPILE_FLAGS "-prec-sqrt")
        set_source_files_properties(${FILES_REQUIRED_PREC_DIV}
            PROPERTIES COMPILE_FLAGS "-prec-div")
    endif()
endif()

# to make build time reasonable, don't use optimizations for s8u8s32 Xbyak
# kernels
file(GLOB FILES_WITHNO_OPT
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/gemm/s8x8s32/jit_avx512_core_kernel_b0_gemm_s8u8s32_kern.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/gemm/s8x8s32/jit_avx512_core_kernel_gemm_s8u8s32_kern.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/gemm/s8x8s32/jit_avx512_core_u8_copy_an_kern.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/gemm/s8x8s32/jit_avx512_core_u8_copy_at_kern.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/gemm/s8x8s32/jit_avx512_core_u8_copy_bn_kern.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/gemm/s8x8s32/jit_avx512_core_u8_copy_bt_kern.cpp)
if(WIN32 AND NOT MINGW)
    set_source_files_properties(${FILES_WITHNO_OPT}
        PROPERTIES COMPILE_FLAGS "/Od")
else()
    set_source_files_properties(${FILES_WITHNO_OPT}
        PROPERTIES COMPILE_FLAGS "-O0 -U_FORTIFY_SOURCE")
endif()

add_library(${TARGET_NAME} ${MKLDNN_LIBRARY_TYPE} ${HEADERS} ${SOURCES})

if(WIN32)
    # Add mkldnn.dll to execution PATH
    if(NOT(MINGW))
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

target_link_libraries(${TARGET_NAME} ${${TARGET_NAME}_LINKER_LIBS} ${EXTRA_LIBS})
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${TARGET_NAME} PROPERTY VERSION "${PROJECT_VERSION}.0")
set_property(TARGET ${TARGET_NAME} PROPERTY SOVERSION "0")

install(TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib${LIB_SUFFIX}
    ARCHIVE DESTINATION lib${LIB_SUFFIX}
)
install(FILES ${HEADERS} DESTINATION include)
