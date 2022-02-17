/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

// clang-format off

#ifndef UTILS_DEBUG_HPP
#define UTILS_DEBUG_HPP

/// @file
/// Debug capabilities

#include "interface/c_types_map.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace utils {

const char *data_type2str(data_type_t v);
const char *engine_kind2str(engine_kind_t v);
const char *layout_type2str(layout_type_t v);
const char *property_type2str(property_type_t v);

} // namespace utils
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
