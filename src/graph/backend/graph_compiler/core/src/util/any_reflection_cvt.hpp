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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_ANY_REFLECTION_CVT_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_ANY_REFLECTION_CVT_HPP

#include "any_map.hpp"
#include "general_object.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace utils {
// converts an any_t to general ref. The RTTI of the any_t must be registered in
// reflection's type registery. Otherwise, this function will throw an exception
reflection::general_ref_t any_to_general_ref(const any_t &);
} // namespace utils
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
