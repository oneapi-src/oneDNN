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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_ENV_VAR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_ENV_VAR_HPP

#include <string>
#include <util/def.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace utils {
// Reads an environment variable 'name' and stores its string value in the
// 'buffer' of 'buffer_size' bytes (including the terminating zero) on
// success.
//
// - Returns the length of the environment variable string value (excluding
// the terminating 0) if it is set and its contents (including the terminating
// 0) can be stored in the 'buffer' without truncation.
//
// - Returns negated length of environment variable string value and writes
// "\0" to the buffer (if it is not NULL) if the 'buffer_size' is to small to
// store the value (including the terminating 0) without truncation.
//
// - Returns 0 and writes "\0" to the buffer (if not NULL) if the environment
// variable is not set.
//
// - Returns INT_MIN if the 'name' is NULL.
//
// - Returns INT_MIN if the 'buffer_size' is negative.
//
// - Returns INT_MIN if the 'buffer' is NULL and 'buffer_size' is greater than
// zero. Passing NULL 'buffer' with 'buffer_size' set to 0 can be used to
// retrieve the length of the environment variable value string.
//
SC_INTERNAL_API int getenv(const char *name, char *buffer, int buffer_size);

// Reads an integer from the environment
SC_INTERNAL_API int getenv_int(const char *name, int default_value = 0);

// A convenience wrapper for the 'getenv' function defined above.
//
// Note: Due to an apparent limitation in the wrapped 'getenv'
// function, this function makes no distinction between:
// (a) the environment variable not being defined at all, vs.
// (b) the environment variable defined, with a value of empty-string.
//
// This function's behavior is undefined if 'name' is null or the empty-string.
SC_INTERNAL_API std::string getenv_string(const char *name);
} // namespace utils
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
