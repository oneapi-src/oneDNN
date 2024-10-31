/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GRAPH_INTERFACE_GRAPH_ATTR_HPP
#define GRAPH_INTERFACE_GRAPH_ATTR_HPP

#include "graph/interface/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace graph {

struct fpmath_t {

    fpmath_t(dnnl_fpmath_mode_t mode = fpmath_mode::strict,
            bool apply_to_int = false)
        : mode_(mode), apply_to_int_(apply_to_int) {}

    bool operator==(const fpmath_t &rhs) const {
        return mode_ == rhs.mode_ && apply_to_int_ == rhs.apply_to_int_;
    }

    graph::fpmath_mode_t mode_;
    bool apply_to_int_ = false;
};

} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
