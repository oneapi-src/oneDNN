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

#ifndef GPU_JIT_CONV_NORMALIZATION_HPP
#define GPU_JIT_CONV_NORMALIZATION_HPP

#include <vector>

#include "gpu/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

std::vector<dim_t> normalize_conv_dims(std::vector<dim_t> &dims,
        bool with_groups, int groups, bool is_dw, int reduced_dim,
        bool fuse_spatial, bool add_groups, bool is_wei);

layout_t normalize_conv_layout(const layout_t &_layout, bool with_groups,
        int groups, bool is_dw, int reduced_dim, bool fuse_spatial,
        bool add_groups, bool is_wei);

void normalize_conv_layouts(layout_t &src_layout, layout_t &wei_layout,
        layout_t &dst_layout, layout_t &bia_layout, bool with_groups, int g,
        int ic, int oc, bool is_dw, int reduced_dim, bool fuse_spatial,
        bool add_groups);

inline void normalize_conv_layouts(layout_t &src_layout, layout_t &wei_layout,
        layout_t &dst_layout, bool with_groups, int g, int ic, int oc,
        bool is_dw, int reduced_dim, bool fuse_spatial, bool add_groups) {
    layout_t bia_layout;
    normalize_conv_layouts(src_layout, wei_layout, dst_layout, bia_layout,
            with_groups, g, ic, oc, is_dw, reduced_dim, fuse_spatial,
            add_groups);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
