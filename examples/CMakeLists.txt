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

if (NOT DNNL_BUILD_EXAMPLES)
    return()
endif()

# propagate EXAMPLE specific flags
append(CMAKE_C_FLAGS "${CMAKE_EXAMPLE_CCXX_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_EXAMPLE_CCXX_FLAGS}")

# propagate sanitizer flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")

# propagate nowarn flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")

include_directories(${PROJECT_SOURCE_DIR}/include)

if(UNIX)
    find_library(LIBM m)
endif()

file(GLOB sources *.cpp *.c)
file(GLOB headers *.hpp *.h)

# Skip SYCL, GPU and cross-engine examples
foreach(f ${sources})
    get_filename_component(f_name ${f} NAME)
    if(DNNL_GPU_RUNTIME STREQUAL "NONE" AND ${f_name} MATCHES "^(cross_engine|gpu)")
        list(REMOVE_ITEM sources "${f}")
    endif()
    if(NOT DNNL_CPU_RUNTIME STREQUAL "SYCL" AND
       NOT DNNL_GPU_RUNTIME STREQUAL "SYCL" AND
       ${f_name} MATCHES "^sycl")
        list(REMOVE_ITEM sources "${f}")
    endif()
endforeach()

# Skip CPU examples that do not work with SYCL
if(DNNL_CPU_RUNTIME STREQUAL "SYCL")
    foreach(f
            cpu_cnn_training_f32.c
            cpu_rnn_inference_int8.cpp
            cpu_rnn_inference_f32.cpp)
        get_filename_component(f_path
                ${CMAKE_CURRENT_LIST_DIR}/${f} ABSOLUTE)
        list(REMOVE_ITEM sources "${f_path}")
    endforeach()
endif()

# Skip USM example with ComputeCpp
if(DNNL_SYCL_COMPUTECPP)
    list(REMOVE_ITEM sources "${CMAKE_CURRENT_LIST_DIR}/sycl_interop_usm.cpp")
endif()

foreach(src ${sources})
    get_filename_component(src_name ${src} NAME)
    string(REGEX REPLACE "[_\\.]" "-" example_name ${src_name})
    if(${example_name} MATCHES "^(cross-engine|gpu)")
        # Example name contains cross-engine or gpu-
        if(NOT ${example_name} MATCHES ".*opencl" OR DNNL_GPU_RUNTIME STREQUAL "OCL")
            register_exe(${example_name} ${src} "test" ${LIBM})
        endif()
    else()
        set(cpu_rt_pattern "(SEQ|OMP|TBB|SYCL)")
        set(gpu_rt_pattern "(OCL|SYCL)")
        if(${example_name} MATCHES "sycl.*")
            set(cpu_rt_pattern "SYCL")
            set(gpu_rt_pattern "SYCL")
        endif()
        if(DNNL_CPU_RUNTIME MATCHES ${cpu_rt_pattern})
            # Adding test for CPU
            add_test("cpu-${example_name}" "${example_name}" cpu)
            maybe_configure_windows_test("cpu-${example_name}" TEST)
        endif()
        if(DNNL_GPU_RUNTIME MATCHES ${gpu_rt_pattern})
            # Adding test for GPU
            add_test("gpu-${example_name}" "${example_name}" gpu)
            maybe_configure_windows_test("gpu-${example_name}" TEST)
        endif()
        register_exe(${example_name} ${src} "" ${LIBM})
        if(DNNL_SYCL_COMPUTECPP)
            add_sycl_to_target(TARGET ${example_name} SOURCES ${src})
        endif()
    endif()
endforeach()

if (DNNL_INSTALL_MODE STREQUAL "BUNDLE")
    configure_file(CMakeLists.txt.in CMakeLists.txt @ONLY)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeLists.txt
        ${sources} ${headers}
        DESTINATION examples)
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        install(FILES
            ${PROJECT_SOURCE_DIR}/cmake/template.vcxproj.user
            DESTINATION examples)
    endif()
endif()
