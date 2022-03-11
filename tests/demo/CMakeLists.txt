#===============================================================================
# Copyright 2020-2022 Intel Corporation
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

if(NOT DNNL_GRAPH_BUILD_TESTS)
    return()
endif()

enable_testing()

# propagate common compilation flags
append(CMAKE_C_FLAGS "${DNNL_GRAPH_COMMON_COMPILATION_FLAGS}")
append(CMAKE_CXX_FLAGS "${DNNL_GRAPH_COMMON_COMPILATION_FLAGS}")

# propagate tests specific flags
append(CMAKE_C_FLAGS "${CMAKE_TESTS_CCXX_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_TESTS_CCXX_FLAGS}")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    append_if(WIN32 CMAKE_CXX_FLAGS "/fp:precise")
    append_if(UNIX  CMAKE_CXX_FLAGS "-fp-model precise")
endif()

# Register demo
function(register_demo name srcs)
    if(TARGET ${name})
        message(STATUS "target with the same name already exists")
        return()
    endif()

    add_executable(${name} ${srcs})
    target_include_directories(${name}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/examples/cpp/include
        ${PROJECT_SOURCE_DIR}/tests/demo/include
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/third_party/oneDNN/include
        ${PROJECT_SOURCE_DIR}/third_party/oneDNN/include
    )

    target_link_libraries(${name} 
        LINK_PRIVATE
        dnnl_graph
        ${DNNL_GRAPH_COMMON_LINK_FLAGS}
        ${DNNL_GRAPH_SYCL_LINK_FLAGS}
    )
endfunction()

file(GLOB sources src/*.cpp)

foreach(src ${sources})
    file(RELATIVE_PATH src_rel_path ${CMAKE_CURRENT_SOURCE_DIR}/src/ ${src})
    string(REGEX REPLACE "[/_\\.]" "-" demo_name ${src_rel_path})

    set(cpu_rt_pattern "(SEQ|OMP)")
    set(sycl_rt_pattern "(DPCPP)")
    if(DNNL_GRAPH_CPU_RUNTIME MATCHES ${cpu_rt_pattern})
	# Adding demos for OMP CPU
        if(${demo_name} MATCHES "cpu-*")
            register_demo(${demo_name} ${src})
            add_test("${demo_name}" "${demo_name}" cpu)
            maybe_configure_windows_test("${demo_name}" TEST)
        endif()
    endif()
    if(DNNL_GRAPH_CPU_RUNTIME MATCHES ${sycl_rt_pattern})
	# Adding examples for SYCL CPU
        if(${demo_name} MATCHES "sycl-*")
            register_demo(${demo_name} ${src})
            add_test("${demo_name}-cpu" "${demo_name}" cpu)
            maybe_configure_windows_test("${demo_name}-cpu" TEST)
        endif()
    endif()
    if(DNNL_GRAPH_GPU_RUNTIME MATCHES ${sycl_rt_pattern})
	# Adding demos for SYCL GPU
        if(${demo_name} MATCHES "gpu-*")
            register_demo(${demo_name} ${src})
            add_test("${demo_name}" "${demo_name}" gpu)
            maybe_configure_windows_test("${demo_name}" TEST)
        endif()
	# Adding demos for SYCL GPU
        if(${demo_name} MATCHES "sycl-*")
            register_demo(${demo_name} ${src})
            add_test("${demo_name}-gpu" "${demo_name}" gpu)
            maybe_configure_windows_test("${demo_name}-gpu" TEST)
        endif()
    endif()
endforeach()
