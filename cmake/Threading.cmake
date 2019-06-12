#===============================================================================
# Copyright 2018 Intel Corporation
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

# Utils for managing threading-related configuration
#===============================================================================

if(Threading_cmake_included)
    return()
endif()
set(Threading_cmake_included true)

# Always require pthreads even for sequential threading (required for e.g.
# std::call_once that relies on mutexes)
find_package(Threads REQUIRED)
list(APPEND EXTRA_SHARED_LIBS "${CMAKE_THREAD_LIBS_INIT}")

# While MKL-DNN defaults to OpenMP (if _OPENMP is defined) without CMake, here
# we default to sequential threading and let OpenMP.cmake and TBB.cmake to
# figure things out. This is especially important because OpenMP is used both
# for threading and vectorization via #pragma omp simd
set(MKLDNN_CPU_RUNTIME_CURRENT "SEQ")
