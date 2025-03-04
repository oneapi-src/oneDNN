/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#include "gpu/intel/jit/ir/ir_builder.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

void ir_builder_t::init_kernel_grid(const grid_info_t &kernel_grid,
        const grid_info_t &tg_grid, int simd_size, constraint_set_t &cset,
        std::vector<stmt_t> &init_stmts) {
    for (dim_idx_t i = 0; i < kernel_grid.ndims(); i++) {
        auto local_id = var_t::make(type_t::u16(), ir_builder_t::local_id(i));
        int local_id_bound = into<int>(tg_grid.dim(i));
        if (i == dim_idx_t(0)) local_id_bound *= simd_size;
        cset.add_constraint(local_id >= 0);
        cset.add_constraint(local_id < local_id_bound);
        cset.add_constraint(kernel_grid.idx(i) >= 0);
        cset.add_constraint(kernel_grid.idx(i) < kernel_grid.dim(i));
        cset.add_constraint(tg_grid.idx(i) >= 0);
        cset.add_constraint(tg_grid.idx(i) < tg_grid.dim(i));
        if (i == 0) local_id /= simd_size;
        auto &type = tg_grid.idx(i).type();
        init_stmts.push_back(let_t::make(tg_grid.idx(i), cast(local_id, type)));
    }
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
