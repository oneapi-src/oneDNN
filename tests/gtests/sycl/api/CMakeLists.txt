#===============================================================================
# Copyright 2019-2020 Intel Corporation
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

set(TEST_EXE test_api_sycl)

file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
list(APPEND TEST_SOURCES ${MAIN_SRC_GTEST})

register_exe(${TEST_EXE} "${TEST_SOURCES}" "test" "dnnl_gtest")
# InteropReorderAndUserKernel & EltwiseWithUserKernel tests run SYCL kernels
# so they need to be compiled with the correct device triple
if(DNNL_WITH_SYCL)
    if(DNNL_SYCL_GENERIC)
        CHECK_CXX_COMPILER_FLAG("-fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64" NVIDIA_TARGET_SUPPORTED)
    endif()

    # Enable linking SYCL kernels.
    if(DNNL_SYCL_CUDA OR (DNNL_SYCL_GENERIC AND NVIDIA_TARGET_SUPPORTED))
        append(CMAKE_CXX_FLAGS "-fsycl-targets=nvptx64-nvidia-cuda,spir64")
        append(CMAKE_CXX_FLAGS "-Wno-linker-warnings")
    endif()

    if(DNNL_AMD_ENABLE_SYCL_KERNELS)
        append(CMAKE_CXX_FLAGS "-fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${DNNL_AMD_SYCL_KERNELS_TARGET_ARCH}")
    endif()
endif()
