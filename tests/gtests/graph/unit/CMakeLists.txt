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

include_directories_with_host_compiler(
    ${PROJECT_SOURCE_DIR}/tests
    ${PROJECT_SOURCE_DIR}/src/graph
    ${PROJECT_SOURCE_DIR}/include
)

if(DNNL_ENABLE_JIT_PROFILING OR DNNL_ENABLE_ITT_TASKS)
    if (UNIX AND NOT APPLE)
        # Not every compiler adds -ldl and -lrt automatically
        list(APPEND EXTRA_SHARED_LIBS "${CMAKE_DL_LIBS}")
        list(APPEND EXTRA_SHARED_LIBS rt)
    endif()
endif()

get_property(STATIC_LIB_DEPS GLOBAL PROPERTY DNNL_SUBDIR_EXTRA_STATIC_LIBS)
get_property(SHARED_LIB_DEPS GLOBAL PROPERTY DNNL_SUBDIR_EXTRA_SHARED_LIBS)
get_property(LIB_DEPS GLOBAL PROPERTY DNNL_LIB_DEPS)

file(GLOB COMMON_UNIT_TESTS
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_test_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_test_main.cpp
    ${TEST_THREAD}
    ${TEST_ALLOCATOR}
)

function(register_graph_unit_exe exe_name srcs category)
    string(REPLACE "test_" "test_graph_${category}" exe_new_name ${exe_name})
    set(DEPS
        ${LIB_PACKAGE_NAME}
        ${EXTRA_SHARED_LIBS}
        ${EXTRA_STATIC_LIBS}
        ${STATIC_LIB_DEPS}
        ${SHARED_LIB_DEPS}
        dnnl_gtest
    )
    
    list(APPEND srcs ${COMMON_UNIT_TESTS} ${LIB_DEPS})
    register_graph_exe(${exe_new_name} "${srcs}" "${DEPS}")
endfunction()

add_subdirectory(interface)
add_subdirectory(backend)
add_subdirectory(utils)
