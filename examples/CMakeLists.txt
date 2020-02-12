#===============================================================================
# Copyright 2016-2020 Intel Corporation
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
include_directories(${PROJECT_SOURCE_DIR}/examples)

if(UNIX)
    find_library(LIBM m)
endif()

file(GLOB_RECURSE sources *.cpp *.c)
file(GLOB_RECURSE headers *.hpp *.h)

# Skip SYCL, GPU and cross-engine examples
foreach(f ${sources})
    get_filename_component(f_name ${f} NAME)
    if(DNNL_GPU_RUNTIME STREQUAL "NONE" AND ${f_name} MATCHES "^(cross_engine|gpu)")
        list(REMOVE_ITEM sources "${f}")
    endif()
    if(NOT DNNL_WITH_SYCL AND ${f_name} MATCHES "^sycl")
        list(REMOVE_ITEM sources "${f}")
    endif()
endforeach()

# Skip CPU examples that do not work with SYCL
if(DNNL_CPU_SYCL)
    foreach(f
            cpu_cnn_training_f32.c
            cpu_rnn_inference_int8.cpp
            cpu_rnn_inference_f32.cpp)
        get_filename_component(f_path
                ${CMAKE_CURRENT_LIST_DIR}/${f} ABSOLUTE)
        list(REMOVE_ITEM sources "${f_path}")
    endforeach()
endif()

# Skip examples that do not work with Nvidia backend
if(DNNL_SYCL_CUDA)
    foreach(f
            cnn_inference_f32.c
            cnn_inference_f32.cpp
            cnn_training_f32.cpp
            rnn_training_f32.cpp
            sycl_interop.cpp
            sycl_interop_usm.cpp
            primitives/layer_normalization.cpp
            primitives/lstm.cpp
            primitives/shuffle.cpp
            primitives/binary.cpp
            primitives/reorder.cpp
            )
        get_filename_component(f_path
                ${CMAKE_CURRENT_LIST_DIR}/${f} ABSOLUTE)
        list(REMOVE_ITEM sources "${f_path}")
    endforeach()
endif()

# Skip USM example with ComputeCpp
if(DNNL_SYCL_COMPUTECPP)
    list(REMOVE_ITEM sources "${CMAKE_CURRENT_LIST_DIR}/sycl_interop_usm.cpp")
endif()

# Skip matmul examples with SYCL
foreach(f ${sources})
    if(DNNL_WITH_SYCL AND ${f} MATCHES "matmul")
        list(REMOVE_ITEM sources "${f}")
    endif()
endforeach()

foreach(src ${sources})
    file(RELATIVE_PATH src_rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${src})
    string(REGEX REPLACE "[/_\\.]" "-" example_name ${src_rel_path})

    # Put hw-specific part of the name in front.
    # It is important for examples in subdirectories.
    foreach(pat "cpu-" "gpu-" "cross-engine-")
        string(REGEX REPLACE "^(.*)${pat}" "${pat}\\1"
            example_name ${example_name})
    endforeach()

    if(${example_name} MATCHES "(cross-engine|cpu|gpu)-")
        # Example name contains cross-engine, cpu or gpu
        if(NOT ${example_name} MATCHES ".*opencl" OR DNNL_GPU_RUNTIME STREQUAL "OCL")
            register_exe(${example_name} ${src} "test" ${LIBM})
        endif()
    else()
        set(cpu_rt_pattern "(SEQ|OMP|TBB|SYCL|DPCPP)")
        set(gpu_rt_pattern "(OCL|SYCL|DPCPP)")
        if(${example_name} MATCHES "sycl.*")
            set(cpu_rt_pattern "(SYCL|DPCPP)")
            set(gpu_rt_pattern "(SYCL|DPCPP)")
        endif()
        set(should_add_usm_target false)
        if(DNNL_SYCL_DPCPP AND NOT ${example_name} MATCHES "sycl-interop")
            set(should_add_usm_target true)
        endif()
        if(DNNL_CPU_RUNTIME MATCHES ${cpu_rt_pattern})
            # Adding test for CPU
            add_test("cpu-${example_name}" "${example_name}" cpu)
            maybe_configure_windows_test("cpu-${example_name}" TEST)
            if(should_add_usm_target)
                add_test("cpu-usm-${example_name}" "usm-${example_name}" cpu)
                maybe_configure_windows_test("cpu-usm-${example_name}" TEST)
            endif()
        endif()
        if(DNNL_GPU_RUNTIME MATCHES ${gpu_rt_pattern})
            # Adding test for GPU
            add_test("gpu-${example_name}" "${example_name}" gpu)
            maybe_configure_windows_test("gpu-${example_name}" TEST)
            if(should_add_usm_target AND NOT DNNL_SYCL_CUDA)
                add_test("gpu-usm-${example_name}" "usm-${example_name}" gpu)
                maybe_configure_windows_test("gpu-usm-${example_name}" TEST)
            endif()
        endif()
        register_exe(${example_name} ${src} "" ${LIBM})
        if(should_add_usm_target)
            register_exe("usm-${example_name}" ${src} "" ${LIBM})
            target_compile_definitions("usm-${example_name}" PUBLIC -DDNNL_USE_DPCPP_USM)
        endif()
        if(DNNL_SYCL_COMPUTECPP AND NOT ${example_name} MATCHES ".*-c$")
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
    # Skip matmul examples with SYCL
    if(NOT DNNL_WITH_SYCL)
        install(DIRECTORY
            tutorials
            DESTINATION examples)
    endif()
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        install(FILES
            ${PROJECT_SOURCE_DIR}/cmake/template.vcxproj.user
            DESTINATION examples)
    endif()
endif()
