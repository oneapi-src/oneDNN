/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_DEF_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_DEF_HPP

#define SC_UNUSED(x) ((void)(x))
// the macro marks that a function is the top-level API of graph-compiler
#ifdef _MSC_VER
#ifdef SC_DLL
#ifdef SC_DLL_EXPORTS
#define SC_API __declspec(dllexport)
#else
#define SC_API __declspec(dllimport)
#endif
#else
#define SC_API
#endif
#else
#define SC_API __attribute__((visibility("default")))
#endif

#define SC_INTERNAL_API

#ifndef SC_MEMORY_LEAK_CHECK
#define SC_MEMORY_LEAK_CHECK 0
#endif

#if SC_MEMORY_LEAK_CHECK > 0
#define SC_EXTENDS_LEAK_CHECK(T) \
    , public ::dnnl::impl::graph::gc::utils::leak_detect_base<T>
#define SC_LEAK_CHECK(T) \
    , public ::dnnl::impl::graph::gc::utils::leak_detect_base<T>
#else
#define SC_EXTENDS_LEAK_CHECK(T)
#define SC_LEAK_CHECK(T)
#endif

#ifndef SC_CFAKE_JIT_ENABLED
#if defined(_WIN32) || defined(__APPLE__)
#define SC_CFAKE_JIT_ENABLED 0
#else
#define SC_CFAKE_JIT_ENABLED 1
#endif
#endif

#ifndef SC_BUILTIN_JIT_ENABLED
#define SC_BUILTIN_JIT_ENABLED 1
#endif

#define SC_THREAD_POOL_SEQ 0
#define SC_THREAD_POOL_OMP 1
#define SC_THREAD_POOL_TBB 2
#define SC_THREAD_POOL_CUSTOM 3

#endif
