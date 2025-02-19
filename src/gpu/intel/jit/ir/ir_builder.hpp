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

#ifndef GPU_INTEL_JIT_IR_IR_BUILDER_HPP
#define GPU_INTEL_JIT_IR_IR_BUILDER_HPP

#include <array>

#include "gpu/intel/jit/ir/ir.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

class idx_dispatcher_t {
public:
    using vars_t = std::array<expr_t, 3>;

    idx_dispatcher_t(const std::string &tg_idx, const std::string &local_id) {
        auto new_var = [](type_t type, const std::string &prefix, size_t idx) {
            return var_t::make(type, prefix + std::to_string(idx));
        };
        for (size_t i = 0; i < tg_idxs_.size(); i++)
            tg_idxs_[i] = new_var(type_t::s32(), tg_idx, i);
        for (size_t i = 0; i < local_ids_.size(); i++)
            local_ids_[i] = new_var(type_t::u16(), local_id, i);
    }
    idx_dispatcher_t() : idx_dispatcher_t("tg_idx", "local_id") {}

    const vars_t &tg_idxs() const { return tg_idxs_; }
    const vars_t &local_ids() const { return local_ids_; }

private:
    vars_t tg_idxs_;
    vars_t local_ids_;
};

class ir_builder_t {
public:
    const stmt_t &stmt() const { return stmt_; }

protected:
    void init_kernel_grid(const grid_info_t &kernel_grid,
            const grid_info_t &tg_grid, const idx_dispatcher_t &idx_disp,
            int simd_size, constraint_set_t &cset,
            std::vector<stmt_t> &init_stmts);

    virtual void build() = 0;

    stmt_t stmt_;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
