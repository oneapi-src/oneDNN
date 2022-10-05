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

#ifndef GPU_JIT_IR_IR_BUILDER_HPP
#define GPU_JIT_IR_IR_BUILDER_HPP

#include <array>

#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/kernel_info.hpp"
#include "gpu/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class ir_builder_t {
public:
    const stmt_t &stmt() const { return stmt_; }

    const std::array<expr_t, 3> &local_id() const { return local_id_; }

protected:
    ir_builder_t(const kernel_info_t &kernel_info)
        : kernel_info_(kernel_info) {}

    void init_kernel_grid(const grid_info_t &kernel_grid,
            const grid_info_t &tg_grid, int simd_size, constraint_set_t &cset,
            std::vector<stmt_t> &init_stmts);

    virtual void build() = 0;

    const kernel_info_t &kernel_info_;
    std::array<expr_t, 3> local_id_; // Local IDs (OpenCL) for the 0-th lane.

    stmt_t stmt_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
