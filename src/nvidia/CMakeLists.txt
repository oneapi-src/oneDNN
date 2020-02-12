#===============================================================================
#  Copyright 2020 Codeplay Software Limited
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  For your convenience, a copy of the License has been included in this
#  repository.
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#===============================================================================

set(TARGET_NAME ${LIB_NAME}_sycl_nvidia)
find_package(OpenCL REQUIRED)
add_library(
    ${TARGET_NAME} OBJECT
    sycl_cuda_engine.cpp
    sycl_cuda_engine.hpp
    sycl_cuda_scoped_context.cpp
    sycl_cuda_scoped_context.hpp
    sycl_cuda_stream.cpp
    sycl_cuda_stream.hpp
    sycl_cuda_utils.hpp
    cudnn_binary.hpp
    cudnn_binary.cpp
    cudnn_binary_impl.hpp
    cudnn_concat.cpp
    cudnn_reorder_impl.hpp
    cudnn_reorder_impl.cpp
    cudnn_reorder.hpp
    cudnn_reorder.cpp
    cudnn_softmax.hpp
    cudnn_softmax.cpp
    cudnn_softmax_impl.hpp
    cudnn_batch_normalization.cpp
    cudnn_inner_product.cpp
    cudnn_inner_product.hpp
    cudnn_inner_product_impl.hpp
    cudnn_eltwise.hpp
    cudnn_eltwise.cpp
    cudnn_eltwise_impl.hpp
    cudnn_convolution.hpp
    cudnn_convolution.cpp
    cudnn_sum.cpp
    cudnn_pooling.cpp
    cudnn_lrn.hpp
    cudnn_lrn.cpp
    cudnn_lrn_impl.hpp
    cudnn_resampling.cpp
    cudnn_matmul.cpp
    cudnn_deconvolution.cpp
)
set_target_properties(
    ${TARGET_NAME}
    PROPERTIES
        COMPILE_DEFINITIONS
        "$<TARGET_PROPERTY:cuBLAS::cuBLAS,INTERFACE_COMPILE_DEFINITIONS>;$<TARGET_PROPERTY:cuDNN::cuDNN,INTERFACE_COMPILE_DEFINITIONS>"
        COMPILE_OPTIONS
        "$<TARGET_PROPERTY:cuBLAS::cuBLAS,INTERFACE_COMPILE_OPTIONS>;$<TARGET_PROPERTY:cuDNN::cuDNN,INTERFACE_COMPILE_OPTIONS>;$<TARGET_PROPERTY:OpenCL::OpenCL,INTERFACE_COMPILE_OPTIONS>"
)
target_include_directories(
    ${TARGET_NAME}
    PRIVATE $<TARGET_PROPERTY:OpenCL::OpenCL,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:cuDNN::cuDNN,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:cuBLAS::cuBLAS,INTERFACE_INCLUDE_DIRECTORIES>
            ${CMAKE_CURRENT_SOURCE_DIR}/.. ${PROJECT_SOURCE_DIR}/include/)

add_library(${TARGET_NAME}_interface INTERFACE)
target_link_libraries(${TARGET_NAME}_interface INTERFACE cuBLAS::cuBLAS
                                                         cuDNN::cuDNN
                                                         OpenCL::OpenCL)
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${TARGET_NAME}>)
                                                         
set(${LIB_NAME}_INTERFACE
    ${${LIB_NAME}_INTERFACE} ${TARGET_NAME}_interface
    PARENT_SCOPE)
