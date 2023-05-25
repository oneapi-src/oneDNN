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

# Controls enabling primitive options and dispatcher to internal logic
#===============================================================================

if(enable_primitives_cmake_included)
    return()
endif()
set(enable_primitives_cmake_included true)
include("cmake/options.cmake")

set(BUILD_${DNNL_ENABLE_WORKLOAD} TRUE)
message(STATUS "Enabled workload: ${DNNL_ENABLE_WORKLOAD}")

if(DNNL_ENABLE_PRIMITIVE STREQUAL "ALL")
    set(BUILD_PRIMITIVE_ALL TRUE)
else()
    foreach(impl ${DNNL_ENABLE_PRIMITIVE})
        string(TOUPPER ${impl} uimpl)
        if(NOT "${uimpl}" MATCHES
                "^(BATCH_NORMALIZATION|BINARY|CONCAT|CONVOLUTION|DECONVOLUTION|ELTWISE|INNER_PRODUCT|LAYER_NORMALIZATION|LRN|MATMUL|POOLING|PRELU|REDUCTION|REORDER|RESAMPLING|RNN|SHUFFLE|SOFTMAX|SUM)$")
            message(FATAL_ERROR "Unsupported primitive: ${uimpl}")
        endif()
        set(BUILD_${uimpl} TRUE)
    endforeach()
endif()
message(STATUS "Enabled primitives: ${DNNL_ENABLE_PRIMITIVE}")

if (DNNL_ENABLE_PRIMITIVE_CPU_ISA STREQUAL "ALL")
    set(BUILD_PRIMITIVE_CPU_ISA_ALL TRUE)
else()
    foreach(isa ${DNNL_ENABLE_PRIMITIVE_CPU_ISA})
        string(TOUPPER ${isa} uisa)
        if(NOT "${uisa}" MATCHES "^(SSE41|AVX2|AVX512|AMX)$")
            message(FATAL_ERROR "Unsupported primitive CPU ISA: ${uisa}")
        endif()
        set(BUILD_${uisa} TRUE)
    endforeach()
endif()
message(STATUS "Enabled primitive CPU ISA: ${DNNL_ENABLE_PRIMITIVE_CPU_ISA}")

if (DNNL_ENABLE_PRIMITIVE_GPU_ISA STREQUAL "ALL")
    set(BUILD_PRIMITIVE_GPU_ISA_ALL TRUE)
else()
    foreach(isa ${DNNL_ENABLE_PRIMITIVE_GPU_ISA})
        string(TOUPPER ${isa} uisa)
        if(NOT "${uisa}" MATCHES "^(GEN9|GEN11|XELP|XEHP|XEHPG|XEHPC)$")
            message(FATAL_ERROR "Unsupported primitive GPU ISA: ${uisa}")
        endif()
        set(BUILD_${uisa} TRUE)
    endforeach()
endif()
message(STATUS "Enabled primitive GPU ISA: ${DNNL_ENABLE_PRIMITIVE_GPU_ISA}")

if (ONEDNN_ENABLE_GEMM_KERNELS_ISA STREQUAL "ALL")
    set(BUILD_GEMM_KERNELS_ALL TRUE)
elseif (ONEDNN_ENABLE_GEMM_KERNELS_ISA STREQUAL "NONE")
    set(BUILD_GEMM_KERNELS_NONE TRUE)
else()
    foreach(isa ${ONEDNN_ENABLE_GEMM_KERNELS_ISA})
        string(TOUPPER ${isa} uisa)
        if(NOT "${uisa}" MATCHES "^(SSE41|AVX2|AVX512)$")
            message(FATAL_ERROR "Unsupported primitive CPU ISA: ${uisa}")
        endif()
        set(BUILD_GEMM_${uisa} TRUE)
    endforeach()
endif()
message(STATUS "Enabled GeMM kernels ISA: ${ONEDNN_ENABLE_GEMM_KERNELS_ISA}")

# When certain primitives or primitive ISA are switched off, some functions may
# become unused which is expected. Switch off warning for unused functions in
# such cases.
if (NOT DNNL_ENABLE_PRIMITIVE STREQUAL "ALL" OR
        NOT DNNL_ENABLE_PRIMITIVE_CPU_ISA STREQUAL "ALL" OR
        NOT DNNL_ENABLE_PRIMITIVE_GPU_ISA STREQUAL "ALL")
    append(CMAKE_CCXX_FLAGS "-Wno-error=unused-function -Wno-unused-function")
endif()
