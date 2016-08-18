#===============================================================================
# Copyright 2016 Intel Corporation
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

if(POLICY CMP0065)
    cmake_policy(SET CMP0065 NEW)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

#add_executable(api-check-c api_check.c)
#target_link_libraries(api-check-c ${LIB_NAME})
#add_test(api-check-c api-check-c)

# add_executable(simple_convolution-c simple_convolution.c)
# set_property(TARGET simple_convolution-c PROPERTY C_STANDARD 99)
# target_link_libraries(simple_convolution-c ${LIB_NAME})
# add_test(simple_convolution-c simple_convolution-c)

add_executable(only_convolution-c only_convolution.c)
set_property(TARGET only_convolution-c PROPERTY C_STANDARD 99)
target_link_libraries(only_convolution-c ${LIB_NAME})
add_test(only_convolution-c only_convolution-c)

add_executable(only_convolution-cpp only_convolution.cpp)
set_property(TARGET only_convolution-cpp PROPERTY CXX_STANDARD 11)
set_property(TARGET only_convolution-cpp PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_libraries(only_convolution-cpp ${LIB_NAME})
add_test(only_convolution-cpp only_convolution-cpp)

#add_executable(api_check-cpp api_check.cpp)
#set_property(TARGET api_check-cpp PROPERTY CXX_STANDARD 11)
#set_property(TARGET api_check-cpp PROPERTY CXX_STANDARD_REQUIRED ON)
#target_link_libraries(api_check-cpp ${LIB_NAME})
#add_test(api_check-cpp api_check-cpp)

add_executable(only_pooling-c only_pooling.c)
set_property(TARGET only_pooling-c PROPERTY C_STANDARD 99)
target_link_libraries(only_pooling-c ${LIB_NAME})
add_test(only_pooling-c only_pooling-c)

add_executable(only_pooling-cpp only_pooling.cpp)
set_property(TARGET only_pooling-cpp PROPERTY CXX_STANDARD 11)
set_property(TARGET only_pooling-cpp PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_libraries(only_pooling-cpp ${LIB_NAME})
add_test(only_pooling-cpp only_pooling-cpp)

set(test_c_symbols "${CMAKE_CURRENT_BINARY_DIR}/test_c_symbols.c")
add_executable(test_c_symbols-c ${test_c_symbols})
add_custom_command(
    OUTPUT ${test_c_symbols}
    COMMAND /bin/sh ${CMAKE_CURRENT_SOURCE_DIR}/generate_c_symbols_refs.sh
    ${CMAKE_SOURCE_DIR} ${test_c_symbols}
)
target_link_libraries(test_c_symbols-c ${LIB_NAME})
add_test(test_c_symbols-c test_c_symbols-c)

add_subdirectory(gtests)
