/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_COMPILER_MACROS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_COMPILER_MACROS_HPP

#if defined(__GNUC__)
#define SC_GNUC_VERSION_GE(x) (__GNUC__ >= (x))
#else
#define SC_GNUC_VERSION_GE(x) 0
#endif

#if defined(__GNUC__)
#define SC_GNUC_VERSION_LT(x) (__GNUC__ < (x))
#else
#define SC_GNUC_VERSION_LT(x) 0
#endif

#endif
