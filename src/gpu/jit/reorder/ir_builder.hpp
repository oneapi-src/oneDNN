/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#ifndef GPU_JIT_REORDER_IR_BUILDER_HPP
#define GPU_JIT_REORDER_IR_BUILDER_HPP

#include <array>

#include "common/convolution_pd.hpp"
#include "gpu/jit/ir/gemm_schedule.hpp"
#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/ir_builder.hpp"
#include "gpu/jit/ir/kernel_info.hpp"
#include "gpu/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class reorder_ir_builder_t : public ir_builder_t {
public:
    reorder_ir_builder_t(const exec_config_t &exec_cfg,
            const kernel_info_t &kernel_info, const layout_t &src_layout,
            const layout_t &dst_layout)
        : ir_builder_t(kernel_info)
        , exec_cfg_(exec_cfg)
        , src_layout_(src_layout)
        , dst_layout_(dst_layout) {
        build();
    }

    const grid_info_t &kernel_grid() const { return kernel_grid_; }

    static void compute_blocks(const exec_config_t &exec_cfg,
            const layout_t &src, const layout_t &dst,
            std::vector<int> &iter_blocks, std::vector<int> &loop_blocks,
            std::vector<int> &tg_blocks, int max_iter_tile_bytes = 0,
            int max_thr_tile_bytes = 0);

    static void compute_blocks(const exec_config_t &exec_cfg,
            const layout_t &src, const layout_t &dst,
            std::vector<int> &tile_blocks, std::vector<int> &tg_blocks);

    static void compute_grid(const layout_t &src, const layout_t &dst,
            const std::vector<int> &iter_blocks,
            const std::vector<int> &loop_blocks,
            const std::vector<int> &tg_blocks, grid_info_t &kernel_grid,
            grid_info_t &tg_grid, std::vector<int> *dim2grid = nullptr);

    static compute::nd_range_t nd_range(const exec_config_t &exec_cfg,
            const layout_t &src, const layout_t &dst);

private:
    void build() override;
    bool try_build(const std::vector<int> &iter_blocks,
            const std::vector<int> &loop_blocks,
            const std::vector<int> &tg_blocks);

    static int max_tile_size(const hw_config_t &hw_cfg, const layout_t &dst,
            const layout_t &src) {
        // XeHPC is fine with 2048 bytes, XeHPG and below can fit 2048 bytes if
        // reorder is a simple copy.
        return (hw_cfg.hw() <= ngen::HW::XeHPG && dst != src) ? 1024 : 2048;
    }

    exec_config_t exec_cfg_;
    grid_info_t kernel_grid_;
    grid_info_t tg_grid_;
    layout_t src_layout_;
    layout_t dst_layout_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
