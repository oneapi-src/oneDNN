/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef BACKEND_DNNL_COMMON_HPP
#define BACKEND_DNNL_COMMON_HPP

#include "oneapi/dnnl/dnnl_graph_types.h"

#include "backend/dnnl/tensor.hpp"
#include "interface/common.hpp"
#include "utils/compatible.hpp"

#include "dnnl.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

using format_tag = dnnl::memory::format_tag;

inline static format_tag get_default_format(size_t ndim) {
    switch (ndim) {
        case 1: return format_tag::a;
        case 2: return format_tag::ab;
        case 3: return format_tag::abc;
        case 4: return format_tag::abcd;
        case 5: return format_tag::abcde;
        case 6: return format_tag::abcdef;
        default: return format_tag::undef;
    }
}

inline static format_tag get_default_format(const dims adims) {
    const auto size = adims.size();
    return get_default_format(size);
}

void fill_layout_info(impl::logical_tensor_t *lt, const tensor::desc &td);

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
