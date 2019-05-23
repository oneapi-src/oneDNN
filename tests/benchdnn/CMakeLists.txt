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

if((WIN32 OR MINGW) AND (NOT MKLDNN_BUILD_FOR_CI))
    string(REPLACE  ";" "\;" PATH "${CTESTCONFIG_PATH};$ENV{PATH}")
    configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/run_with_env.bat.in"
        "${PROJECT_BINARY_DIR}/run_with_env.bat"
    )
endif()

# register benchdnn test case
function(register_benchdnn_test name engine cmd)
    set(benchdnn_target ${name}_${engine})

    if(MKLDNN_BUILD_FOR_CI)
       string(REPLACE " " ";" cmd
            "benchdnn --engine=${engine} ${cmd}")
        if (ARGC GREATER 3)
            string(REPLACE " " ";" cmd2
                "benchdnn --engine=${engine} ${ARGV3}")
        endif()

        add_test(${benchdnn_target} ${cmd})
        if (DEFINED cmd2)
            set(second_target_name "${benchdnn_target}2")
            add_test(${second_target_name} ${cmd2})
        endif()
    else()
        string(REPLACE " " ";" cmd
            "$<TARGET_FILE:benchdnn> --engine=${engine} ${cmd}")
        if (ARGC GREATER 3)
            string(REPLACE " " ";" cmd2
                "$<TARGET_FILE:benchdnn> --engine=${engine} ${ARGV3}")
        endif()

        if(WIN32 OR MINGW)
            set(cmd "cmd;/c;${PROJECT_BINARY_DIR}/run_with_env.bat;${cmd}")
            set(ARGV2 "cmd;/c;${PROJECT_BINARY_DIR}/run_with_env.bat;${ARGV2}")
        endif()

        add_custom_target(${benchdnn_target}
            COMMAND ${cmd}
            COMMAND ${cmd2} # if any
            DEPENDS benchdnn
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )

        set_target_properties(${benchdnn_target} PROPERTIES
            EXCLUDE_FROM_DEFAULT_BUILD TRUE)
        maybe_configure_windows_test(${benchdnn_target} TARGET)

        # Create non-suffixed target for compatibility
        if (engine STREQUAL "cpu")
            add_custom_target(${name} DEPENDS ${benchdnn_target})
        endif()
    endif()
endfunction()

register_benchdnn_test(test_benchdnn_conv
    cpu "-v1 --conv --batch=inputs/test_conv_all")
register_benchdnn_test(test_benchdnn_conv_bf16
    cpu "-v1 --conv --batch=inputs/test_conv_bfloat16")
register_benchdnn_test(test_benchdnn_deconv
    cpu "-v1 --deconv --batch=inputs/deconv/test_deconv_all")
register_benchdnn_test(test_benchdnn_rnn
    cpu "-v1 --rnn --batch=inputs/rnn/test_rnn_small")
register_benchdnn_test(test_benchdnn_reorder
    cpu "--reorder --batch=inputs/reorder/test_default")
register_benchdnn_test(test_benchdnn_bnorm
    cpu "--bnorm  --batch=inputs/bnorm/test_bnorm_all")
register_benchdnn_test(test_benchdnn_bnorm_bf16
    cpu "--bnorm  --batch=inputs/bnorm/test_bnorm_bfloat16")
register_benchdnn_test(test_benchdnn_ip
    cpu "--ip --batch=inputs/ip/test_ip_all")
register_benchdnn_test(test_benchdnn_ip_bf16
    cpu "--ip --batch=inputs/ip/test_ip_bfloat16")
register_benchdnn_test(test_benchdnn_shuffle
    cpu "--shuffle --batch=inputs/shuffle/test_shuffle_all")
register_benchdnn_test(test_benchdnn_softmax
    cpu "--softmax --batch=inputs/softmax/test_softmax_all")
register_benchdnn_test(test_benchdnn_pool
    cpu "--pool --batch=inputs/pool/test_pool_all")
register_benchdnn_test(test_benchdnn_regression
    cpu
    "--conv --batch=inputs/test_conv_regression"
    "--bnorm --batch=inputs/bnorm/test_bnorm_regressions"
    )
register_benchdnn_test(test_benchdnn_regression_large
    cpu
    "--bnorm --batch=inputs/bnorm/test_bnorm_regressions_large"
    )

if(NOT MKLDNN_GPU_BACKEND STREQUAL "NONE")
    register_benchdnn_test(test_benchdnn_conv
        gpu "-v1 --conv --batch=inputs/test_conv_gpu")
    register_benchdnn_test(test_benchdnn_bnorm
        gpu "--bnorm --batch=inputs/bnorm/test_bnorm_gpu")
    register_benchdnn_test(test_benchdnn_ip
        gpu "--ip --batch=inputs/ip/test_ip_gpu")
    register_benchdnn_test(test_benchdnn_rnn
        gpu "--rnn --batch=inputs/rnn/test_rnn_gpu")
    register_benchdnn_test(test_benchdnn_pool
        gpu "--pool --batch=inputs/pool/test_pool_gpu")
endif()

if(MKLDNN_INSTALL_MODE STREQUAL "BUNDLE")
    install(TARGETS benchdnn RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
