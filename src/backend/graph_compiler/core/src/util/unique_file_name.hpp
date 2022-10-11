/*******************************************************************************
 * Copyright 2022 Intel Corporation
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
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_UNIQUE_FILE_NAME_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_UNIQUE_FILE_NAME_HPP

#include <string>
#include <util/def.hpp>

namespace sc {
namespace utils {
/**
 * Generates a unique name for file name
 * */
SC_INTERNAL_API std::string get_unique_name_for_file();
} // namespace utils

} // namespace sc

#endif
