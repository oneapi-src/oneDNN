#===============================================================================
# Copyright 2017-2020 Intel Corporation
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
list(APPEND SOURCES ${TEST_THREAD})
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

if(WIN32 AND (NOT DNNL_BUILD_FOR_CI))
    string(REPLACE  ";" "\;" PATH "${CTESTCONFIG_PATH};$ENV{PATH}")
    configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/run_with_env.bat.in"
        "${PROJECT_BINARY_DIR}/run_with_env.bat"
    )
endif()

# register benchdnn test case
function(register_benchdnn_test name engine cmd)
    set(benchdnn_target ${name}_${engine})

    if(DNNL_BUILD_FOR_CI)
       string(REPLACE " " ";" cmd
            "benchdnn -v1 --engine=${engine} ${cmd}")
        if (ARGC GREATER 3)
            string(REPLACE " " ";" cmd2
                "benchdnn -v1 --engine=${engine} ${ARGV3}")
        endif()

        add_test(${benchdnn_target} ${cmd})
        if (DEFINED cmd2)
            set(second_target_name "${benchdnn_target}2")
            add_test(${second_target_name} ${cmd2})
        endif()
    else()
        string(REPLACE " " ";" cmd
            "$<TARGET_FILE:benchdnn> -v1 --engine=${engine} ${cmd}")
        if (ARGC GREATER 3)
            string(REPLACE " " ";" cmd2
                "$<TARGET_FILE:benchdnn> -v1 --engine=${engine} ${ARGV3}")
        endif()

        if(WIN32)
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
            maybe_configure_windows_test(${name} TARGET)
        endif()
    endif()
endfunction()

register_benchdnn_test(test_benchdnn_conv_topologies
    cpu "--conv --batch=inputs/conv/test_conv_all_topologies")
register_benchdnn_test(test_benchdnn_conv_topologies_f32_nxc
    cpu "--conv --batch=inputs/conv/test_conv_all_topologies_f32_nxc")
register_benchdnn_test(test_benchdnn_conv_bfloat16 # temporary proxy
    cpu "--conv --batch=inputs/conv/test_conv_bfloat16")
register_benchdnn_test(test_benchdnn_conv_bf16
    cpu "--conv --batch=inputs/conv/test_conv_bfloat16")
register_benchdnn_test(test_benchdnn_conv_f32
    cpu "--conv --batch=inputs/conv/test_conv_f32")
register_benchdnn_test(test_benchdnn_conv_f32_nxc
    cpu "--conv --batch=inputs/conv/test_conv_f32_nxc")
register_benchdnn_test(test_benchdnn_conv_3d
    cpu "--conv --batch=inputs/conv/test_conv_3d")
register_benchdnn_test(test_benchdnn_conv_3d_f32_nxc
    cpu "--conv --batch=inputs/conv/test_conv_3d_f32_nxc")
register_benchdnn_test(test_benchdnn_conv_wino
    cpu "--conv --batch=inputs/conv/test_conv_f32_wino")
register_benchdnn_test(test_benchdnn_conv_int8_wino
    cpu "--conv --batch=inputs/conv/test_conv_int8_wino")
register_benchdnn_test(test_benchdnn_conv_depthwise
    cpu "--conv --batch=inputs/conv/test_conv_depthwise")
register_benchdnn_test(test_benchdnn_conv_dilated
    cpu "--conv --batch=inputs/conv/test_conv_dilated")
register_benchdnn_test(test_benchdnn_conv_dilated_f32_nxc
    cpu "--conv --batch=inputs/conv/test_conv_dilated_f32_nxc")
register_benchdnn_test(test_benchdnn_conv_attributes
    cpu "--conv --batch=inputs/conv/test_conv_attrs")
register_benchdnn_test(test_benchdnn_conv_attributes_f32_nxc
    cpu "--conv --batch=inputs/conv/test_conv_attrs_f32_nxc")
register_benchdnn_test(test_benchdnn_conv_int8
    cpu "--conv --batch=inputs/conv/test_conv_int8")
register_benchdnn_test(test_benchdnn_conv_function
    cpu "--conv --batch=inputs/conv/test_conv_function")
register_benchdnn_test(test_benchdnn_conv_gemm_f32
    cpu "--conv --batch=inputs/conv/test_conv_gemm_f32")
register_benchdnn_test(test_benchdnn_conv_gemm_nxc_f32
    cpu "--conv --batch=inputs/conv/test_conv_gemm_nxc_f32")
register_benchdnn_test(test_benchdnn_conv_gemm_int8
    cpu "--conv --batch=inputs/conv/test_conv_gemm_int8")
register_benchdnn_test(test_benchdnn_conv_gemm_bf16
    cpu "--conv --batch=inputs/conv/test_conv_gemm_bfloat16")
register_benchdnn_test(test_benchdnn_deconv
    cpu "--deconv --batch=inputs/deconv/test_deconv_all")
register_benchdnn_test(test_benchdnn_deconv_nxc_f32
    cpu "--deconv --batch=inputs/deconv/test_deconv_all_nxc_f32")
register_benchdnn_test(test_benchdnn_deconv_bf16
    cpu "--deconv --batch=inputs/deconv/test_deconv_bfloat16")
register_benchdnn_test(test_benchdnn_matmul
    cpu "--matmul --batch=inputs/matmul/test_matmul_all")
register_benchdnn_test(test_benchdnn_resampling
    cpu "--resampling --batch=inputs/resampling/test_resampling_all")
register_benchdnn_test(test_benchdnn_rnn
    cpu "--rnn --batch=inputs/rnn/test_rnn_small")
register_benchdnn_test(test_benchdnn_reorder
    cpu "--reorder --batch=inputs/reorder/test_reorder_all")
register_benchdnn_test(test_benchdnn_reorder_bf16
    cpu "--reorder --batch=inputs/reorder/test_reorder_bfloat16")
register_benchdnn_test(test_benchdnn_bnorm
    cpu "--bnorm  --batch=inputs/bnorm/test_bnorm_all")
register_benchdnn_test(test_benchdnn_lnorm
    cpu "--lnorm  --batch=inputs/lnorm/test_lnorm_all")
register_benchdnn_test(test_benchdnn_bnorm_bf16
    cpu "--bnorm  --batch=inputs/bnorm/test_bnorm_bfloat16")
register_benchdnn_test(test_benchdnn_ip
    cpu "--ip --batch=inputs/ip/test_ip_all")
register_benchdnn_test(test_benchdnn_ip_bf16
    cpu "--ip --batch=inputs/ip/test_ip_bfloat16")
register_benchdnn_test(test_benchdnn_shuffle
    cpu "--shuffle --batch=inputs/shuffle/test_shuffle_all")
register_benchdnn_test(test_benchdnn_shuffle_bf16
    cpu "--shuffle --batch=inputs/shuffle/test_shuffle_bfloat16")
register_benchdnn_test(test_benchdnn_softmax
    cpu "--softmax --batch=inputs/softmax/test_softmax_all")
register_benchdnn_test(test_benchdnn_softmax_bf16
    cpu "--softmax --batch=inputs/softmax/test_softmax_bfloat16")
register_benchdnn_test(test_benchdnn_pool
    cpu "--pool --batch=inputs/pool/test_pool_all")
register_benchdnn_test(test_benchdnn_pool_bf16
    cpu "--pool --batch=inputs/pool/test_pool_bfloat16")
register_benchdnn_test(test_benchdnn_sum
    cpu "--sum --batch=inputs/sum/test_sum_all")
register_benchdnn_test(test_benchdnn_sum_bf16
    cpu "--sum --batch=inputs/sum/test_sum_bfloat16")
register_benchdnn_test(test_benchdnn_eltwise
    cpu "--eltwise --batch=inputs/eltwise/test_eltwise_all")
register_benchdnn_test(test_benchdnn_eltwise_bf16
    cpu "--eltwise --batch=inputs/eltwise/test_eltwise_bfloat16")
register_benchdnn_test(test_benchdnn_concat
    cpu "--concat --batch=inputs/concat/test_concat_all")
register_benchdnn_test(test_benchdnn_concat_bf16
    cpu "--concat --batch=inputs/concat/test_concat_bfloat16")
register_benchdnn_test(test_benchdnn_lrn
    cpu "--lrn --batch=inputs/lrn/test_lrn_all")
register_benchdnn_test(test_benchdnn_lrn_bf16
    cpu "--lrn --batch=inputs/lrn/test_lrn_bfloat16")
register_benchdnn_test(test_benchdnn_binary
    cpu "--binary --batch=inputs/binary/test_binary_all")
register_benchdnn_test(test_benchdnn_binary_bf16
    cpu "--binary --batch=inputs/binary/test_binary_bf16")
register_benchdnn_test(test_benchdnn_regression
    cpu
    "--conv --batch=inputs/conv/test_conv_regression"
    "--bnorm --batch=inputs/bnorm/test_bnorm_regressions"
    )
if(NOT DNNL_USE_CLANG_SANITIZER)
    register_benchdnn_test(test_benchdnn_regression_large
        cpu
        "--bnorm --batch=inputs/bnorm/test_bnorm_regressions_large"
        )
endif()

if(NOT DNNL_GPU_RUNTIME STREQUAL "NONE")
    register_benchdnn_test(test_benchdnn_conv
        gpu "--conv --batch=inputs/conv/test_conv_gpu")
    register_benchdnn_test(test_benchdnn_deconv
        gpu "--deconv --batch=inputs/deconv/test_deconv_gpu")
    register_benchdnn_test(test_benchdnn_bnorm
        gpu "--bnorm --batch=inputs/bnorm/test_bnorm_gpu")
    register_benchdnn_test(test_benchdnn_lnorm
        gpu "--lnorm --batch=inputs/lnorm/test_lnorm_gpu")
    register_benchdnn_test(test_benchdnn_ip
        gpu "--ip --batch=inputs/ip/test_ip_gpu")
    register_benchdnn_test(test_benchdnn_rnn
        gpu "--rnn --batch=inputs/rnn/test_rnn_small_gpu")
    register_benchdnn_test(test_benchdnn_reorder
        gpu "--reorder --batch=inputs/reorder/test_reorder_gpu")
    register_benchdnn_test(test_benchdnn_pool
        gpu "--pool --batch=inputs/pool/test_pool_gpu")
    register_benchdnn_test(test_benchdnn_eltwise
        gpu "--eltwise --batch=inputs/eltwise/test_eltwise_gpu")
    register_benchdnn_test(test_benchdnn_shuffle
        gpu "--shuffle --batch=inputs/shuffle/test_shuffle_gpu")
    register_benchdnn_test(test_benchdnn_softmax
        gpu "--softmax --batch=inputs/softmax/test_softmax_gpu")
    register_benchdnn_test(test_benchdnn_sum
        gpu "--sum --batch=inputs/sum/test_sum_gpu")
    register_benchdnn_test(test_benchdnn_concat
        gpu "--concat --batch=inputs/concat/test_concat_gpu")
    register_benchdnn_test(test_benchdnn_lrn
        gpu "--lrn --batch=inputs/lrn/test_lrn_gpu")
    register_benchdnn_test(test_benchdnn_binary
        gpu "--binary --batch=inputs/binary/test_binary_gpu")
    register_benchdnn_test(test_benchdnn_matmul
        gpu "--matmul --batch=inputs/matmul/test_matmul_gpu")
    register_benchdnn_test(test_benchdnn_resampling
        gpu "--resampling --batch=inputs/resampling/test_resampling_gpu")
endif()
