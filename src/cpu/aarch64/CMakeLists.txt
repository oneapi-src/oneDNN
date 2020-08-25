#===============================================================================
# Copyright 2020 Intel Corporation
# Copyright 2020 FUJITSU LIMITED
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
option(DNNL_INDIRECT_JIT_AARCH64 "enables indirect JIT for AArch64." ON)
    # enabled by default
option(DNNL_NATIVE_JIT_AARCH64 "enables native JIT for AArch64." ON)
    # enabled by default


file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp
    )

if(NOT DNNL_ENABLE_JIT_PROFILING)
    # XXX: the profiling interface will still be built and present
    add_definitions(-DDNNL_ENABLE_JIT_PROFILING=0)
    # Don't enable support for linux_perf and VTune Amplifier
    list(REMOVE_ITEM SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/jit_utils/jitprofiling/jitprofiling.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/jit_utils/linux_perf/linux_perf.cpp"
        )
endif()

if(DNNL_INDIRECT_JIT_AARCH64)
    append(CMAKE_C_FLAGS "-DDNNL_INDIRECT_JIT_AARCH64")
    append(CMAKE_CXX_FLAGS "-DDNNL_INDIRECT_JIT_AARCH64")

    append(CMAKE_C_FLAGS "-DXBYAK_TRANSLATE_AARCH64")
    append(CMAKE_CXX_FLAGS "-DXBYAK_TRANSLATE_AARCH64")

    append(CMAKE_C_FLAGS "-DXBYAK_AARCH64_FOR_DNNL")
    append(CMAKE_CXX_FLAGS "-DXBYAK_AARCH64_FOR_DNNL")
endif()

set(OBJ_LIB ${LIB_NAME}_cpu_aarch64)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
