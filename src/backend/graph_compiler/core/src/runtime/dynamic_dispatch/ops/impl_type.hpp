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
#include <stdint.h>

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_OPS_IMPL_TYPE_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_OPS_IMPL_TYPE_HPP

namespace sc {
// Predefine all ops' impl algorithm type here.
// impl algorithm type, include normal(padding)/no padding select.
enum impl_kind_t : int {
    normal = 0, // default generate rule
    no_padding = 1, // generate without padding
};

} // namespace sc

#endif
