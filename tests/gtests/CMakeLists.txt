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

add_subdirectory (gtest)

set(APP_NAME "gtest")
set(MAIN_SRC_GTEST ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/gtest
                    ${CMAKE_CURRENT_SOURCE_DIR}/in
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cpu
                    )
if(WIN32)
    # Correct 'jnl' macro/jit issue
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
        append(CMAKE_CXX_FLAGS "/Qlong-double")
    endif()
endif()

# TODO: enable me!
file(GLOB PRIM_TEST_CASES_SRC
                              test_iface_pd.cpp
                              test_iface_pd_iter.cpp
                              test_iface_attr.cpp
                              test_iface_handle.cpp
                              test_iface_stream_attr.cpp
                              test_iface_runtime_dims.cpp
                              test_iface_runtime_attr.cpp
                              test_dnnl_threading.cpp
                              test_memory.cpp
                              test_sum.cpp
                              test_reorder.cpp
                              test_cross_engine_reorder.cpp
                              test_concat.cpp
                              test_softmax.cpp
                              test_eltwise.cpp
                              test_lrn_forward.cpp
                              test_lrn_backward.cpp
                              test_pooling_forward.cpp
                              test_pooling_backward.cpp
                              test_batch_normalization_f32.cpp
                              test_batch_normalization_s8.cpp
                              test_inner_product_forward.cpp
                              test_inner_product_backward_data.cpp
                              test_inner_product_backward_weights.cpp
                              test_shuffle.cpp
                              test_convolution_format_any.cpp
                              test_convolution_forward_f32.cpp
                              test_convolution_forward_u8s8s32.cpp
                              test_convolution_forward_u8s8fp.cpp
                              test_convolution_eltwise_forward_f32.cpp
                              test_convolution_eltwise_forward_x8s8f32s32.cpp
                              test_convolution_backward_data_f32.cpp
                              test_convolution_backward_weights_f32.cpp
                              test_deconvolution.cpp
                              test_gemm_f16.cpp
                              test_gemm_f32.cpp
                              test_gemm_f16f16f32.cpp
                              test_gemm_bf16bf16f32.cpp
                              test_gemm_bf16bf16bf16.cpp
                              test_gemm_u8s8s32.cpp
                              test_gemm_s8s8s32.cpp
                              test_gemm_s8u8s32.cpp
                              test_gemm_u8u8s32.cpp
                              test_rnn_forward.cpp
                              test_layer_normalization.cpp
                              test_binary.cpp
                              test_logsoftmax.cpp
                              test_matmul.cpp
                              test_resampling.cpp
                              test_isa_mask.cpp
                              test_isa_iface.cpp
                              test_global_scratchpad.cpp
                              )

# Some tests will fail on AArch64 and need to be skipped.
# For example, any tests wich check for ISAs not present on AArch64.
# AARCH64_SKIP_TESTS is a list of tests to be removed from PRIM_TEST_CASES_SRC.
if(TARGET_ARCH STREQUAL "AARCH64")
    file(GLOB AARCH64_SKIP_TESTS
                                  test_isa_iface.cpp
                                  )

# Remove AARCH64_SKIP_TESTS entries from PRIM_TEST_CASES_SRC.
    foreach(TEST_FILE ${AARCH64_SKIP_TESTS})
        list(REMOVE_ITEM PRIM_TEST_CASES_SRC "${TEST_FILE}")
    endforeach()
endif()

# Workaround for an Intel compiler bug: stack unwinding does not restore
# some of the nonvolatile registers with /O2 optimization
if(WIN32 AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set_source_files_properties(
            test_convolution_eltwise_forward_x8s8f32s32.cpp
            test_convolution_backward_data_f32.cpp
            test_convolution_backward_weights_f32.cpp
            PROPERTIES COMPILE_FLAGS "/O1")
endif()

# Tests that don't support '--engine' parameter
set_source_files_properties(test_cross_engine_reorder.cpp
        PROPERTIES NO_ENGINE_PARAM true)

foreach(TEST_FILE ${PRIM_TEST_CASES_SRC})
    get_filename_component(exe ${TEST_FILE} NAME_WE)

    add_executable(${exe} ${MAIN_SRC_GTEST} ${TEST_FILE})
    maybe_configure_windows_test(${exe} TARGET)
    add_definitions(-DNOMINMAX) # to allow std::max on Windows with parentheses
    target_link_libraries(${exe} ${LIB_NAME} dnnl_gtest ${EXTRA_SHARED_LIBS})

    get_source_file_property(no_engine_param ${TEST_FILE} NO_ENGINE_PARAM)

    if (NOT no_engine_param)
        target_compile_definitions(${exe} PUBLIC
                -DDNNL_TEST_WITH_ENGINE_PARAM)
    endif()

    if(NOT DNNL_GPU_RUNTIME STREQUAL "NONE" AND NOT no_engine_param)
        add_test(${exe}_cpu ${exe} COMMAND ${exe} --engine=cpu)
        maybe_configure_windows_test(${exe}_cpu TEST)

        add_test(${exe}_gpu ${exe} COMMAND ${exe} --engine=gpu)
        maybe_configure_windows_test(${exe}_gpu TEST)
    else()
        add_test(${exe} ${exe})
        maybe_configure_windows_test(${exe} TEST)
    endif()
endforeach()

add_subdirectory(api)

if(DNNL_GPU_RUNTIME STREQUAL "OCL")
    add_subdirectory(ocl)
endif()

# vim: et ts=4 sw=4
