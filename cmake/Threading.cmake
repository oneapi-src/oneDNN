#===============================================================================
# Copyright 2018-2020 Intel Corporation
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

# CPU threading runtime specifies the threading used by the library:
# sequential, OpenMP or TBB. In future it may be different from CPU runtime.
set(DNNL_CPU_THREADING_RUNTIME "${DNNL_CPU_RUNTIME}")

if(NECVE) # possibly OK to use always?
    set(CMAKE_THREAD_PREFER_PTHREAD 1)
    set(THREADS_PREFER_PTHREAD_FLAG 1)
    set(Threads_FIND_QUIETLY FALSE)
endif()    
# Always require pthreads even for sequential threading (required for e.g.
# std::call_once that relies on mutexes)
find_package(Threads REQUIRED)
list(APPEND EXTRA_SHARED_LIBS "${CMAKE_THREAD_LIBS_INIT}")
