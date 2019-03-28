#===============================================================================
# Copyright 2017-2018 Intel Corporation
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

file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.hpp
    )
file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/conv
    ${CMAKE_CURRENT_SOURCE_DIR}/ip
    ${CMAKE_CURRENT_SOURCE_DIR}/shuffle
    ${CMAKE_CURRENT_SOURCE_DIR}/reorder
    )

if(BENCHDNN_USE_RDPMC)
    add_definitions(-DBENCHDNN_USE_RDPMC)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    append_if(WIN32 CMAKE_CXX_FLAGS "-Qprec-div -Qprec-sqrt")
    append_if(UNIX  CMAKE_CXX_FLAGS "-prec-div -prec-sqrt -fp-model precise")
endif()

if(UNIX AND NOT APPLE)
    find_library(LIBRT rt)
endif()
register_exe(benchdnn "${SOURCES}" "" "${LIBRT}")

file(COPY inputs DESTINATION .)

if(WIN32 OR MINGW)
    string(REPLACE  ";" "\;" PATH "${CTESTCONFIG_PATH};$ENV{PATH}")
    configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/run_with_env.bat.in"
        "${PROJECT_BINARY_DIR}/run_with_env.bat"
    )
endif()

# register benchdnn test case
function(register_benchdnn_test name cmd)
    string(REPLACE " " ";" cmd "$<TARGET_FILE:benchdnn> ${cmd}")
    string(REPLACE " " ";" ARGV2 "$<TARGET_FILE:benchdnn> ${ARGV2}")
    if(WIN32 OR MINGW)
        set(cmd "cmd;/c;${PROJECT_BINARY_DIR}/run_with_env.bat;${cmd}")
        set(ARGV2 "cmd;/c;${PROJECT_BINARY_DIR}/run_with_env.bat;${ARGV2}")
    endif()
    add_custom_target(${name}
        COMMAND ${cmd}
        COMMAND ${ARGV2} # if any
        DEPENDS benchdnn
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    maybe_configure_windows_test(${name} TARGET)
endfunction()

register_benchdnn_test(test_conv "-v1 --conv --batch=inputs/test_conv_all")
register_benchdnn_test(test_benchdnn_conv "-v1 --conv --batch=inputs/test_conv_all")
register_benchdnn_test(test_benchdnn_deconv "-v1 --deconv --batch=inputs/deconv/test_deconv_all")
register_benchdnn_test(test_benchdnn_rnn "-v1 --rnn --batch=inputs/rnn/test_rnn_small")
register_benchdnn_test(test_benchdnn_reorder "--reorder --batch=inputs/reorder/test_default")
register_benchdnn_test(test_benchdnn_bnorm "--bnorm  --batch=inputs/bnorm/test_bnorm_all")
register_benchdnn_test(test_benchdnn_ip "--ip --batch=inputs/ip/test_ip_all")
register_benchdnn_test(test_benchdnn_regression
    "--conv --batch=inputs/test_conv_regression"
    "--bnorm --batch=inputs/bnorm/test_bnorm_regressions"
    )
register_benchdnn_test(test_benchdnn_regression_large
    "--bnorm --batch=inputs/bnorm/test_bnorm_regressions_large"
    )

if(MKLDNN_INSTALL_MODE STREQUAL "BUNDLE")
    install(TARGETS benchdnn RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/benchdnn)
    install(DIRECTORY inputs DESTINATION ${CMAKE_INSTALL_DATADIR}/benchdnn)
endif()
