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

#include "gpu/jit/ir/ir_builder.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

void ir_builder_t::init_kernel_grid(const std::array<int, 3> &kernel_grid_dims,
        const std::array<int, 3> &tg_grid_dims, int simd_size,
        constraint_set_t &cset, std::vector<stmt_t> &init_stmts) {
    int grid_ndims = 3;
    kernel_grid_ = grid_info_t(grid_ndims);
    tg_grid_ = grid_info_t(grid_ndims);
    for (int i = 0; i < grid_ndims; i++) {
        local_id_[i]
                = var_t::make(type_t::u16(), "local_id" + std::to_string(i));
        kernel_grid_.dim(i) = kernel_grid_dims[i];
        kernel_grid_.idx(i)
                = var_t::make(type_t::s32(), "grid_idx" + std::to_string(i));
        tg_grid_.dim(i) = tg_grid_dims[i];
        tg_grid_.idx(i)
                = var_t::make(type_t::s32(), "tg_idx" + std::to_string(i));

        int local_id_bound = tg_grid_dims[i];
        if (i == 0) local_id_bound *= simd_size;
        cset.add_constraint(local_id_[i] >= 0);
        cset.add_constraint(local_id_[i] < local_id_bound);

        cset.add_constraint(kernel_grid_.idx(i) >= 0);
        cset.add_constraint(kernel_grid_.idx(i) < kernel_grid_dims[i]);
        cset.add_constraint(tg_grid_.idx(i) >= 0);
        cset.add_constraint(tg_grid_.idx(i) < tg_grid_dims[i]);
    }

    for (int i = 0; i < grid_ndims; i++) {
        auto value = local_id_[i];
        if (i == 0) value /= simd_size;
        auto &type = tg_grid_.idx(i).type();
        init_stmts.push_back(let_t::make(tg_grid_.idx(i), cast(value, type)));
    }
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
