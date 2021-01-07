#===============================================================================
# Copyright 2020 Intel Corporation
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

if(NOT DNNL_GRAPH_BUILD_EXAMPLES)
    return()
endif()

enable_testing()

append(CMAKE_CXX_FLAGS ${DNNL_GRAPH_COMMON_COMPILATION_FLAGS})
append(CMAKE_C_FLAGS ${DNNL_GRAPH_COMMON_COMPILATION_FLAGS})

# Register example
function(register_example name srcs)
    add_executable(${name} ${srcs})
    target_include_directories(${name}  
        PRIVATE
        ${PROJECT_SOURCE_DIR}/examples/cpp/include
        ${PROJECT_SOURCE_DIR}/include/llga
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
    string(REGEX REPLACE "[/_\\.]" "-" example_name ${src_rel_path})

    set(cpu_rt_pattern "(SEQ|OMP)")
    set(gpu_rt_pattern "(DPCPP)")
    if(DNNL_GRAPH_CPU_RUNTIME MATCHES ${cpu_rt_pattern})
	# Adding examples for OMP CPU
        if(${example_name} MATCHES "cpu-*")
            register_example(${example_name} ${src})
            add_test("${example_name}" "${example_name}" cpu)
        endif()
    endif()
    if(DNNL_GRAPH_GPU_RUNTIME MATCHES ${gpu_rt_pattern})
	# Adding examples for SYCL GPU
        if(${example_name} MATCHES "gpu-*")
            register_example(${example_name} ${src})
            add_test("${example_name}" "${example_name}" gpu)
        endif()
	# Adding examples for SYCL CPU and GPU
        if(${example_name} MATCHES "sycl-*")
            register_example(${example_name} ${src})
            add_test("${example_name}-cpu" "${example_name}" cpu)
            add_test("${example_name}-gpu" "${example_name}" gpu)
        endif()
    endif()
endforeach()
