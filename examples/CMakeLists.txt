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

include_directories(${PROJECT_SOURCE_DIR}/include)

set_if(UNIX LIBM m)

register_exe(getting-started getting_started.cpp "test")
register_exe(memory-format-propagation memory_format_propagation.cpp "test")

register_exe(cnn-inference-fp32-c cnn_inference_fp32.c "test")
register_exe(cnn-inference-fp32-cpp cnn_inference_fp32.cpp "test")
register_exe(cnn-training-fp32-c cnn_training_fp32.c "test" ${LIBM})
register_exe(cnn-training-fp32-cpp cnn_training_fp32.cpp "test" ${LIBM})
register_exe(cnn-inference-int8-cpp cnn_inference_int8.cpp "test")
register_exe(rnn-inference-fp32-cpp rnn_inference_fp32.cpp "test")
register_exe(rnn-inference-int8 rnn_inference_int8.cpp "test")
register_exe(rnn-training-fp32 rnn_training_fp32.cpp "test")

if(NOT MKLDNN_GPU_BACKEND STREQUAL "NONE")
    register_exe(gpu-getting-started-c gpu_getting_started.c "test")
    register_exe(gpu-getting-started-cpp gpu_getting_started.cpp "test")
endif()

if (MKLDNN_INSTALL_MODE STREQUAL "BUNDLE")
    file(GLOB example_sources "*.cpp" "*.c")
    configure_file(CMakeLists.txt.in CMakeLists.txt @ONLY)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeLists.txt
        ${PROJECT_SOURCE_DIR}/cmake/template.vcxproj.user
        ${example_sources}
        DESTINATION examples)
endif()
