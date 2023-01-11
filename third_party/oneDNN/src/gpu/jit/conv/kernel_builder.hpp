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

#ifndef GPU_JIT_CONV_KERNEL_BUILDER_HPP
#define GPU_JIT_CONV_KERNEL_BUILDER_HPP

#include <array>

#include "common/convolution_pd.hpp"
#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/gemm_schedule.hpp"
#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/post_op_support.hpp"
#include "gpu/jit/conv/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class kernel_builder_t {
public:
    kernel_builder_t(const conv_config_t &cfg, const convolution_pd_t *pd,
            const kernel_info_t &kernel_info)
        : cfg_(cfg), pd_(pd), kernel_info_(kernel_info) {
        build();
    }

    const stmt_t &stmt() const { return stmt_; }

    const grid_info_t &kernel_grid() const { return kernel_grid_; }

    const std::array<expr_t, 3> &local_id() const { return local_id_; }

private:
    void build();
    void init_fwd(gemm_schedule_t &gemm_schedule, view_t &src_view,
            view_t &wei_view, view_t &dst_view, expr_t &src_buf,
            expr_t &wei_buf, expr_t &dst_buf);
    void init_bwd_d(gemm_schedule_t &gemm_schedule, view_t &dst_view,
            view_t &wei_view, view_t &src_view, expr_t &dst_buf,
            expr_t &wei_buf, expr_t &src_buf);
    void init_bwd_w(gemm_schedule_t &gemm_schedule, view_t &src_view,
            view_t &dst_view, view_t &wei_view, view_t &bia_view,
            expr_t &src_buf, expr_t &dst_buf, expr_t &wei_buf, expr_t &bia_buf,
            expr_t &bia_reduction_condition);

    const conv_config_t &cfg_;
    const convolution_pd_t *pd_;
    const kernel_info_t &kernel_info_;

    std::array<expr_t, 3> local_id_; // Local IDs (OpenCL) for the 0-th lane.
    grid_info_t kernel_grid_; // Kernel grid (consisting of thread groups).
    grid_info_t tg_grid_; // Thread group grid (consisting of threads).

    stmt_t stmt_;
};

class reorder_kernel_builder_t {
public:
    reorder_kernel_builder_t(const hw_config_t &hw_cfg,
            const kernel_info_t &kernel_info, const layout_t &src_layout,
            const layout_t &dst_layout)
        : hw_cfg_(hw_cfg)
        , kernel_info_(kernel_info)
        , src_layout_(src_layout)
        , dst_layout_(dst_layout) {
        build();
    }

    const stmt_t &stmt() const { return stmt_; }

    const grid_info_t &kernel_grid() const { return kernel_grid_; }

    const std::array<expr_t, 3> &local_id() const { return local_id_; }

    static void compute_blocks(const layout_t &src, const layout_t &dst,
            std::vector<int> &iter_blocks, std::vector<int> &loop_blocks,
            std::vector<int> &tg_blocks, int max_iter_tile_bytes = 0,
            int max_thr_tile_bytes = 0);

    static void compute_blocks(const layout_t &src, const layout_t &dst,
            std::vector<int> &tile_blocks, std::vector<int> &tg_blocks);

    static void compute_grid(const layout_t &src, const layout_t &dst,
            const std::vector<int> &iter_blocks,
            const std::vector<int> &loop_blocks,
            const std::vector<int> &tg_blocks, std::array<int, 3> &kernel_grid,
            std::array<int, 3> &tg_grid, std::vector<int> *dim2grid = nullptr);

    static compute::nd_range_t nd_range(
            int simd, const layout_t &src, const layout_t &dst);

private:
    void build();
    bool try_build(const std::vector<int> &iter_blocks,
            const std::vector<int> &loop_blocks,
            const std::vector<int> &tg_blocks);

    static const int default_max_iter_tile_bytes = 2048;
    static const int default_max_thr_tile_bytes = 2048;

    hw_config_t hw_cfg_;
    const kernel_info_t &kernel_info_;
    layout_t src_layout_;
    layout_t dst_layout_;

    std::array<expr_t, 3> local_id_; // Local IDs (OpenCL) for the 0-th lane.
    grid_info_t kernel_grid_; // Kernel grid (consisting of thread groups).
    grid_info_t tg_grid_; // Thread group grid (consisting of threads).

    stmt_t stmt_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
