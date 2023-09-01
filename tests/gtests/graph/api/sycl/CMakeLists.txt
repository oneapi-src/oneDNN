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

file(GLOB API_SYCL_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_compiled_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_tensor.cpp
)

foreach(TEST_FILE ${API_SYCL_TEST_SOURCES})
    get_filename_component(exe ${TEST_FILE} NAME_WE)
    # add suffix _usm to test both CPU and GPU
    register_graph_api_exe(${exe}_sycl_usm ${TEST_FILE})
endforeach()
