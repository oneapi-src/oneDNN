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

class ir_builder_t {
public:
    const stmt_t &stmt() const { return stmt_; }

#define GENNAME(prefix) \
    static std::string prefix(int idx) { return #prefix + std::to_string(idx); }
    GENNAME(tg_idx)
    GENNAME(thr_idx)
    GENNAME(local_id)
#undef GENNAME

protected:
    void init_kernel_grid(const grid_info_t &kernel_grid,
            const grid_info_t &tg_grid, int simd_size, constraint_set_t &cset,
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
