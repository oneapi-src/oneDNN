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

if (NOT MKLDNN_BUILD_EXAMPLES)
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

register_exe(cpu-getting-started cpu_getting_started.cpp "test")
register_exe(cpu-memory-format-propagation cpu_memory_format_propagation.cpp "test")
register_exe(cpu-performance-profiling-cpp cpu_performance_profiling.cpp "test")

register_exe(cpu-cnn-inference-f32-c cpu_cnn_inference_f32.c "test")
register_exe(cpu-cnn-inference-f32-cpp cpu_cnn_inference_f32.cpp "test")
register_exe(cpu-cnn-training-f32-c cpu_cnn_training_f32.c "test" ${LIBM})
register_exe(cpu-cnn-training-f32-cpp cpu_cnn_training_f32.cpp "test" ${LIBM})
register_exe(cpu-cnn-training-bf16-cpp cpu_cnn_training_bf16.cpp "test" ${LIBM})
register_exe(cpu-cnn-inference-int8-cpp cpu_cnn_inference_int8.cpp "test")
register_exe(cpu-rnn-inference-f32-cpp cpu_rnn_inference_f32.cpp "test")
register_exe(cpu-rnn-inference-int8 cpu_rnn_inference_int8.cpp "test")
register_exe(cpu-rnn-training-f32 cpu_rnn_training_f32.cpp "test")

if(NOT MKLDNN_GPU_RUNTIME STREQUAL "NONE")
    register_exe(gpu-getting-started-c gpu_getting_started.c "test")
    register_exe(gpu-getting-started-cpp gpu_getting_started.cpp "test")
    register_exe(gpu-opencl-interop-cpp gpu_opencl_interop.cpp "test")
endif()

if (MKLDNN_INSTALL_MODE STREQUAL "BUNDLE")
    file(GLOB example_sources "cpu_*.cpp" "cpu_*.c")
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
