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

set_if(UNIX LIBM m)

file(GLOB_RECURSE sources *.cpp *.c)

foreach(src ${sources})
    file(RELATIVE_PATH src_rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${src})
    string(REGEX REPLACE "[/_\\.]" "-" example_name ${src_rel_path})

    # Put hw-specific part of the name in front.
    # It is important for examples in subdirectories.
    foreach(pat "cpu-" "gpu-" "cross-engine-")
        string(REGEX REPLACE "^(.*)${pat}" "${pat}\\1"
            example_name ${example_name})
    endforeach()

    if(${example_name} MATCHES "(cpu-)")
        # Example name contains cpu-
        register_exe(${example_name} ${src} "test" ${LIBM})
    elseif(${example_name} MATCHES "^(cross-engine|gpu)")
        # Example name contains cross-engine or gpu-
        if (NOT DNNL_GPU_RUNTIME STREQUAL "NONE")
            register_exe(${example_name} ${src} "test" ${LIBM})
        endif()
    else()
        register_exe(${example_name} ${src} "" ${LIBM})

        # Adding test for CPU
        add_test("cpu-${example_name}" "${example_name}" cpu)
        maybe_configure_windows_test("cpu-${example_name}" TEST)

        # Adding test for GPU
        if (NOT DNNL_GPU_RUNTIME STREQUAL "NONE")
           add_test("gpu-${example_name}" "${example_name}" gpu)
           maybe_configure_windows_test("gpu-${example_name}" TEST)
        endif()
    endif()
endforeach()

if (DNNL_INSTALL_MODE STREQUAL "BUNDLE")
    file(GLOB example_sources
         "cpu_*.cpp" "c_*.c" "cnn_*.c" "cnn_*.cpp" "rnn_*.cpp" "*.h*")
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
