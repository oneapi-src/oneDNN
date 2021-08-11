/*******************************************************************************
* Copyright 2021 Intel Corporation
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

// We define those internal used operators in this file. For those operators
// defined on API can be found at src/interface/c_types_map.hpp.

#ifndef INTERFACE_INTERNAL_OPS_HPP
#define INTERFACE_INTERNAL_OPS_HPP

#include <string>
#include <vector>

#include "interface/c_types_map.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace op_kind {

enum {
    kAny = 0x1234,
};

const op_kind_t any = static_cast<op_kind_t>(kAny);

} // namespace op_kind
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
