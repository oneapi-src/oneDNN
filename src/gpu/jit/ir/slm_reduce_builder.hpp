/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef GPU_JIT_IR_SLM_REDUCE_BUILDER_HPP
#define GPU_JIT_IR_SLM_REDUCE_BUILDER_HPP

#include "gpu/jit/ir/hw.hpp"
#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/tensor.hpp"
#include "gpu/jit/ngen/ngen.hpp"

#include <vector>

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class slm_reduce_builder_t {
public:
    slm_reduce_builder_t() = default;

    slm_reduce_builder_t(ir_context_t &ir_ctx, const grid_info_t &tg_grid,
            const expr_t &reg_buf, const layout_t &reg_layout,
            const tensor_t &thr_tile, int dim = 2);

    bool is_empty() const { return reg_buf_.is_empty(); }

    const layout_t &reg_layout() const { return reg_layout_; }

    const tensor_t &thr_tile() const { return thr_tile_; }

    const stmt_t &store_stmt() const { return store_stmt_; }

    const stmt_t &load_stmt() const { return load_stmt_; }

    const std::vector<stmt_t> &allocs() const { return allocs_; }

    const expr_t &reduce_cond() const { return reduce_cond_; }

    stmt_t stmt() const {
        stmt_t ret;
        ret = ret.append(funcs::barrier());
        ret = ret.append(store_stmt_);
        ret = ret.append(funcs::barrier());
        ret = ret.append(load_stmt_);
        ret = inject_alloc_stmts(ret, allocs_);
        return ret;
    }

private:
    void build();

    uint32_t reduction_mask() const {
        uint32_t mask = 0xFFFFFFFF;
        for (int i = 0; i < tg_ndims_; i++) {
            int k_dim_idx = reg_layout_.ndims() + i;
            mask &= ~(1 << k_dim_idx);
        }
        return mask;
    }

    ir_context_t *ir_ctx_ = nullptr;
    grid_info_t tg_grid_;

    expr_t reg_buf_;
    layout_t reg_layout_;
    tensor_t thr_tile_;

    int dim_ = -1;

    expr_t tmp_reg_buf_;
    int tmp_reg_buf_size_ = 0;

    expr_t slm_buf_;
    int slm_buf_size_ = 0;

    int tg_ndims_ = 0;

    stmt_t store_stmt_;
    stmt_t load_stmt_;
    expr_t reduce_cond_;

    std::vector<stmt_t> allocs_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
