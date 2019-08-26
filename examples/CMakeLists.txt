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

foreach(src ${sources})
    get_filename_component(src_name ${src} NAME)
    string(REGEX REPLACE "[_\\.]" "-" example_name ${src_name})
    if(${example_name} MATCHES "sycl*")
        if(DNNL_GPU_RUNTIME STREQUAL "SYCL")
            if(NOT ${example_name} MATCHES ".*usm" OR DNNL_SYCL_INTEL)
                register_exe(${example_name} ${src} "test" ${LIBM})
                if(DNNL_SYCL_COMPUTECPP AND NOT ${example_name} MATCHES ".*usm")
                    add_sycl_to_target(TARGET ${example_name} SOURCES ${src})
                endif()
            endif()
        endif()
    elseif(${example_name} MATCHES ".*opencl")
         if(DNNL_GPU_RUNTIME STREQUAL "OCL")
            register_exe(${example_name} ${src} "test" ${LIBM})
        endif()
    elseif(${example_name} MATCHES "^(cpu)")
            if(NOT ${example_name} MATCHES "^(cpu-rnn|cpu-cnn)" AND NOT DNNL_WITH_SYCL)
                # Example name contains cpu- but not rnn
                register_exe(${example_name} ${src} "test" ${LIBM})
            endif()
    elseif(${example_name} MATCHES "^(cross-engine|gpu)")
        # Example name contains cross-engine or gpu-
        if(NOT DNNL_GPU_RUNTIME STREQUAL "NONE")
            register_exe(${example_name} ${src} "test" ${LIBM})
        endif()
    else()
        register_exe(${example_name} ${src} "" ${LIBM})
        # Adding test for CPU
        add_test("cpu-${example_name}" "${example_name}" cpu)
        maybe_configure_windows_test("cpu-${example_name}" TEST)
        # Adding test for GPU
        if(NOT DNNL_GPU_RUNTIME STREQUAL "NONE")
           add_test("gpu-${example_name}" "${example_name}" gpu)
           maybe_configure_windows_test("gpu-${example_name}" TEST)
        endif()
    endif()
endforeach()

if (DNNL_INSTALL_MODE STREQUAL "BUNDLE")
    file(GLOB example_sources
         "cpu_*.cpp" "c_*.c" "cnn_*.c" "cnn_*.cpp" "rnn_*.cpp" "*.h*")
    if(DNNL_CPU_RUNTIME STREQUAL "SYCL")
        foreach(ex_skip
                getting_started.cpp
                rnn_inference_f32.cpp
                cpu_rnn_inference_int8.cpp)
            get_filename_component(ex_skip_abs_path
                    ${CMAKE_CURRENT_LIST_DIR}/${ex_skip} ABSOLUTE)
            list(REMOVE_ITEM example_sources "${ex_skip_abs_path}")
        endforeach()
    endif()
    configure_file(CMakeLists.txt.in CMakeLists.txt @ONLY)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeLists.txt
        ${example_sources}
        DESTINATION examples)
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        install(FILES
            ${PROJECT_SOURCE_DIR}/cmake/template.vcxproj.user
            DESTINATION examples)
    endif()
endif()
