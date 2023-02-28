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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_OS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_OS_HPP

#include <stdlib.h>
#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(a, b) _aligned_malloc((b), (a))
#define __PRETTY_FUNCTION__ __FUNCSIG__
#define aligned_free(a) _aligned_free((a))
#else
#define aligned_free(a) free((a))
#endif

#endif
