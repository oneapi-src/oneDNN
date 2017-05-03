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

cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME "Intel(R) MKL-DNN")
set(PROJECT_FULL_NAME "Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)")
set(PROJECT_VERSION, "0.7")

project(${PROJECT_NAME} C CXX)

set(LIB_NAME mkldnn)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
    message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to ${CMAKE_BUILD_TYPE}")
endif()

include("cmake/MKL.cmake")
include("cmake/Doxygen.cmake")

# sdl options
if(UNIX OR APPLE)
    set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} -fPIC -Wformat -Wformat-security")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_FORTIFY_SOURCE=2")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -D_FORTIFY_SOURCE=2")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie")
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -z noexecstack -z relro -z now")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -z noexecstack -z relro -z now")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
            set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} -fstack-protector-all")
        else()
            set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} -fstack-protector-strong")
        endif()
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} -fstack-protector-all")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -z noexecstack -z relro -z now")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -z noexecstack -z relro -z now")
    endif()
    set(CCXX_WARN_FLAGS "-Wall -Werror -Wno-unknown-pragmas")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CCXX_FLAGS} -std=c99")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CCXX_FLAGS} -std=c++11 -fvisibility-inlines-hidden")
    set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} ${OPENMP_FLAGS} ${CCXX_WARN_FLAGS} -DMKLDNN_DLL -DMKLDNN_DLL_EXPORTS -fvisibility=internal")
    set(OPENMP_FLAGS "-fopenmp")
    if(HAVE_MKL)
        set(CMAKE_C_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS ${OPENMP_FLAGS})
        set(CMAKE_CXX_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS ${OPENMP_FLAGS})
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -liomp5 -Wl,--as-needed")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -liomp5 -Wl,--as-needed")
    endif()
endif()

if (WIN32)
    # For windows builds we use the following defines in the code to implement windows specfic code:
    # _WIN32: for everything that is windows platform specific (for example posix_memalign)
    # _MSC_VER: for everything that is MSVC specific (if MSVC is used to compile)
    # _WIN: to enable winabi64 calling in jit*
    #
    # We tested with vs2015 and Intel Compilter 17.0.
    # While vs2015 works, it supports only openmp.v2 which does not support 'omp parallel collapse' 
    # so we can't use openmp and need to operate sequential.
    #
    # To build with MSVC:
    #   cmake ..  -A x64 -G "Visual Studio 14 2015"
    # To build with Intel Compiler:
    #   cmake .. -A x64 -G "Visual Studio 14 2015" -T "Intel C++ Compiler 17.0"
    #

    # defines for all windows builds
	add_definitions(-DNOMINMAX -DWIN32 -DOS_WIN -D_MBCS -DWIN64 -DWIN32_LEAN_AND_MEAN -DNOGDI -D_WIN)
    add_definitions(-DMKLDNN_DLL -DMKLDNN_DLL_EXPORTS)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        # defines for Intel compiler
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 /Qopenmp")
    endif()

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        # defines for msvc
        add_definitions(/bigobj /nologo /EHsc /GF /FC /MP /Gm-)
        add_definitions(/wd4267 /wd4244 /wd4800 /wd4805 /wd4002 /wd4003)
    endif()

    set (EXTERNAL_LIBS "${MKLROOT}/../compiler/lib/intel64_win/libiomp5md.lib")

    # TODO: mkl-dnn jit allocates larger chunks of memory on stack (ie. 32k). On windows the
    # default commited stack size is 4K needs to increase to 32k. For now do this with via
    # linker ... need to find a better way of doing this.
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:1048576,32768")

    # collect the final exe's and dll's in a single directory
    set(dir ${CMAKE_CURRENT_BINARY_DIR})
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${dir})
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xHOST")
endif()

enable_testing()

include_directories(include)

add_subdirectory(src)

add_subdirectory(examples)
add_subdirectory(tests)

install(FILES LICENSE DESTINATION share/doc/${LIB_NAME})
