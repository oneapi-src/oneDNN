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

#include "gpu/jit/reorder/ir_builder.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>
#include <unordered_map>

#include "gpu/jit/ir/gemm_schedule.hpp"
#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/ir/reorder.hpp"
#include "gpu/jit/ir/tensor.hpp"
#include "gpu/jit/pass/pass.hpp"
#include "gpu/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class tile_helper_t {
public:
    tile_helper_t(const layout_t &l)
        : l_(l)
        , running_blocks_(l.blocks().size(), 1)
        , blocks_(l.blocks().size(), 1) {
        const auto &l_blocks = l.blocks();
        const auto size = l_blocks.size();
        while (block_idx_ < size && l_blocks[block_idx_].block == 1)
            block_idx_++;
    }

    bool has_next() const { return block_idx_ < running_blocks_.size(); }

    dim_t size() const {
        dim_t ret = l_.size();
        for (auto b : blocks_)
            ret *= b;
        return ret;
    }

    bool is_dense() const {
        bool is_end = false;
        for (size_t i = 0; i < blocks_.size(); i++) {
            if (blocks_[i] == l_.blocks()[i].block) continue;
            if (blocks_[i] != 1 && is_end) return false;
            is_end = true;
        }
        return true;
    }

    tensor_t tile() const {
        std::vector<dim_t> dims(l_.ndims(), 1);
        for (size_t i = 0; i < blocks_.size(); i++) {
            int dim_idx = l_.blocks()[i].dim_idx;
            dims[dim_idx] *= blocks_[i];
        }
        return tensor_t(dims);
    }

    tensor_t next() {
        dim_t l_block = l_.blocks()[block_idx_].block;
        for (dim_t b = running_blocks_[block_idx_] + 1; b <= l_block; b++) {
            if (l_block % b == 0) {
                running_blocks_[block_idx_] = b;
                return running_tile();
            }
        }
        block_idx_++;
        if (has_next()) return next();
        return tensor_t();
    }

    void accept() { blocks_[block_idx_] = running_blocks_[block_idx_]; }

    static bool can_be_mapped(const layout_t &l, const tensor_t &t) {
        std::vector<dim_t> rem_dims = t.dims();
        for (auto &b : l.blocks()) {
            auto &rem_dim = rem_dims[b.dim_idx];
            if (rem_dim >= b.block) {
                if (rem_dim % b.block != 0) return false;
                rem_dim /= b.block;
                continue;
            }
            if (b.block % rem_dim != 0) return false;
            rem_dim = 1;
        }
        for (auto d : rem_dims)
            ir_assert(d == 1);
        return true;
    }

    static tensor_t merge(const tensor_t &a, const tensor_t &b) {
        std::vector<dim_t> dims(a.ndims());
        for (int i = 0; i < a.ndims(); i++) {
            dims[i] = std::max(a(i), b(i));
        }
        return tensor_t(dims);
    }

private:
    tensor_t running_tile() const {
        std::vector<dim_t> dims(l_.ndims(), 1);
        for (size_t i = 0; i < block_idx_; i++) {
            int dim_idx = l_.blocks()[i].dim_idx;
            dims[dim_idx] *= blocks_[i];
        }
        int dim_idx = l_.blocks()[block_idx_].dim_idx;
        dims[dim_idx] *= running_blocks_[block_idx_];
        return tensor_t(dims);
    }

    const layout_t &l_;
    std::vector<dim_t> running_blocks_;
    std::vector<dim_t> blocks_;
    size_t block_idx_ = 0;
};

void reorder_ir_builder_t::compute_blocks(const exec_config_t &exec_cfg,
        const layout_t &src, const layout_t &dst, std::vector<int> &iter_blocks,
        std::vector<int> &loop_blocks, std::vector<int> &tg_blocks,
        int max_iter_tile_bytes, int max_thr_tile_bytes) {
    if (max_iter_tile_bytes <= 0)
        max_iter_tile_bytes = max_tile_size(exec_cfg.hw_cfg(), dst, src);
    if (max_thr_tile_bytes <= 0)
        max_thr_tile_bytes = max_tile_size(exec_cfg.hw_cfg(), dst, src);

    ir_assert(src.ndims() == dst.ndims());
    int ndims = src.ndims();
    std::vector<dim_t> dims(ndims);
    for (int i = 0; i < ndims; i++) {
        dims[i] = std::max(src.dim(i), dst.dim(i));
    }

    // Pad src/dst layouts to match each other.
    auto pad_layout = [&](const layout_t &l) {
        std::vector<block_t> padded_blocks;
        for (auto &eb : l.enumerated_blocks()) {
            auto b = eb.second;
            if (l.is_outermost(eb)) {
                dim_t inner = l.dim(b.dim_idx) / b.block;
                b.block = ir_utils::safe_divide(dims[b.dim_idx], inner);
            }
            padded_blocks.push_back(b);
        }
        return layout_t(
                l.type(), ndims, 0, padded_blocks, /*do_normalize=*/false);
    };
    layout_t padded_src = pad_layout(src);
    layout_t padded_dst = pad_layout(dst);
    ir_assert(ir_utils::is_equal(padded_src.dims(), padded_dst.dims()));

    int elems = padded_src.elems();
    int max_type_size = std::max(src.type().size(), dst.type().size());
    dim_t max_iter_tile_elems
            = std::min(max_iter_tile_bytes / max_type_size, elems);
    dim_t max_thr_tile_elems
            = std::min(max_thr_tile_bytes / max_type_size, elems);

    tile_helper_t src_th(padded_src);
    tile_helper_t dst_th(padded_dst);

    // Incrementally increase subtiles in src and dst. The goal is to find the
    // maximum src/dst tiles so that the final combined tile covers dense
    // regions as big as possible in src/dst layouts.
    std::vector<tensor_t> candidate_tiles;
    // To ensure there is at least one candidate.
    candidate_tiles.emplace_back(std::vector<dim_t>(ndims, 1));
    for (;;) {
        if (!src_th.has_next() || !dst_th.has_next()) break;
        tile_helper_t *th = &src_th;
        bool src_dense = src_th.is_dense();
        bool dst_dense = dst_th.is_dense();
        // When both sublayouts are dense, try to increase the smallest tile.
        // Otherwise, if there is a dense sublayout try to increase it.
        if (src_dense && dst_dense && dst_th.size() < src_th.size()) {
            th = &dst_th;
        } else if (dst_dense && !src_dense) {
            th = &dst_th;
        }

        auto tile = th->next();
        auto &other_th = (th == &src_th ? dst_th : src_th);
        tile = tile_helper_t::merge(tile, other_th.tile());
        if (tile_helper_t::can_be_mapped(padded_src, tile)
                && tile_helper_t::can_be_mapped(padded_dst, tile)) {
            th->accept();
            candidate_tiles.push_back(tile);
        }
        if (tile.elems() >= max_thr_tile_elems) break;
    }

    std::sort(candidate_tiles.begin(), candidate_tiles.end(),
            [](const tensor_t &a, const tensor_t &b) {
                return a.elems() > b.elems();
            });

    const tensor_t *thr_tile = nullptr;
    const tensor_t *iter_tile = nullptr;
    for (size_t i = 0; i < candidate_tiles.size(); i++) {
        auto &t = candidate_tiles[i];
        if (!thr_tile && t.elems() <= max_thr_tile_elems) thr_tile = &t;
        if (thr_tile && !iter_tile && t.elems() <= max_iter_tile_elems
                && thr_tile->is_divisible(t)) {
            iter_tile = &t;
        }
        if (thr_tile && iter_tile) break;
    }

    ir_assert(thr_tile);
    ir_assert(iter_tile);
    std::vector<int> thr_blocks(
            thr_tile->dims().begin(), thr_tile->dims().end());
    iter_blocks.assign(iter_tile->dims().begin(), iter_tile->dims().end());

    ir_assert(utils::array_product(iter_blocks) <= max_iter_tile_elems);
    ir_assert(utils::array_product(thr_blocks) <= max_thr_tile_elems);

    // Initialize loop blocks.
    loop_blocks.resize(ndims, 1);
    for (int i = 0; i < ndims; i++) {
        loop_blocks[i] = ir_utils::safe_divide(thr_blocks[i], iter_blocks[i]);
    }

    // Initialize thread group blocks.
    // Heuristic: try to split outer dimension and assign its
    // inner part to the thread group. This may give better
    // bandwidth utilization on XeHP/XeHPG.
    tg_blocks.resize(ndims, 1);
    const int tg_factor = 2;
    for (int i = 0; i < ndims; i++) {
        int outer = utils::div_up(dims[i], thr_blocks[i]);
        if (outer % tg_factor == 0) {
            tg_blocks[i] = tg_factor;
            break;
        }
    }
}

void reorder_ir_builder_t::compute_blocks(const exec_config_t &exec_cfg,
        const layout_t &src, const layout_t &dst, std::vector<int> &tile_blocks,
        std::vector<int> &tg_blocks) {
    std::vector<int> iter_blocks;
    std::vector<int> loop_blocks;
    compute_blocks(exec_cfg, src, dst, iter_blocks, loop_blocks, tg_blocks);
    size_t n = iter_blocks.size();
    tile_blocks.resize(n);
    for (size_t i = 0; i < n; i++) {
        tile_blocks[i] = iter_blocks[i] * loop_blocks[i];
    }
}

void reorder_ir_builder_t::compute_grid(const layout_t &src,
        const layout_t &dst, const std::vector<int> &iter_blocks,
        const std::vector<int> &loop_blocks, const std::vector<int> &tg_blocks,
        grid_info_t &kernel_grid, grid_info_t &tg_grid,
        std::vector<int> *dim2grid) {
    int ndims = src.ndims();
    std::vector<dim_t> dims(ndims);
    for (int i = 0; i < ndims; i++) {
        dims[i] = std::max(src.dim(i), dst.dim(i));
    }

    if (dim2grid) dim2grid->resize(ndims, -1);

    const int grid_ndims = 3;
    std::vector<int> kernel_grid_dims(grid_ndims, 1);
    std::vector<int> tg_grid_dims(grid_ndims, 1);
    int grid_idx = 0;
    int max_grid_idx = grid_ndims - 1;
    for (int i = 0; i < ndims; i++) {
        if (dim2grid) (*dim2grid)[i] = grid_idx;
        int outer = utils::div_up(
                dims[i], iter_blocks[i] * loop_blocks[i] * tg_blocks[i]);
        tg_grid_dims[grid_idx] *= tg_blocks[i];
        kernel_grid_dims[grid_idx] *= outer;
        if (outer != 1 && grid_idx != max_grid_idx) grid_idx++;
    }
    kernel_grid = grid_info_t(kernel_grid_dims, "grid_idx");
    tg_grid = grid_info_t(tg_grid_dims, "grid_idx");
}

compute::nd_range_t reorder_ir_builder_t::nd_range(
        const exec_config_t &exec_cfg, const layout_t &src,
        const layout_t &dst) {
    const int simd = exec_cfg.simd();
    std::vector<int> iter_blocks;
    std::vector<int> loop_blocks;
    std::vector<int> tg_blocks;
    compute_blocks(exec_cfg, src, dst, iter_blocks, loop_blocks, tg_blocks);
    grid_info_t kernel_grid;
    grid_info_t tg_grid;
    compute_grid(src, dst, iter_blocks, loop_blocks, tg_blocks, kernel_grid,
            tg_grid);
    std::array<size_t, 3> global;
    std::array<size_t, 3> local;
    for (int i = 0; i < kernel_grid.ndims(); i++) {
        global[i] = kernel_grid[i] * tg_grid[i];
        local[i] = tg_grid[i];
        if (i == 0) {
            global[i] *= simd;
            local[i] *= simd;
        }
    }
    return compute::nd_range_t(global.data(), local.data());
}

void reorder_ir_builder_t::build() {
    std::vector<int> iter_blocks;
    std::vector<int> loop_blocks;
    std::vector<int> tg_blocks;
    compute_blocks(exec_cfg_, src_layout_, dst_layout_, iter_blocks,
            loop_blocks, tg_blocks);

    int max_iters = 10;
    int cur_iter_bytes
            = max_tile_size(exec_cfg_.hw_cfg(), dst_layout_, src_layout_);
    for (int i = 0; i < max_iters; i++) {
        if (try_build(iter_blocks, loop_blocks, tg_blocks)) {
            ir_info() << "Reorder configuration:" << std::endl;
            ir_info() << "  Source layout:              " << src_layout_
                      << std::endl;
            ir_info() << "  Destination layout:         " << dst_layout_
                      << std::endl;
            ir_info() << "  Iteration blocks:           "
                      << ir_utils::make_seq_print_helper(iter_blocks, " x ")
                      << std::endl;
            ir_info() << "  Loop blocks:                "
                      << ir_utils::make_seq_print_helper(loop_blocks, " x ")
                      << std::endl;
            ir_info() << "  Thread group blocks:        "
                      << ir_utils::make_seq_print_helper(tg_blocks, " x ")
                      << std::endl;
            return;
        }

        cur_iter_bytes /= 2;
        while (cur_iter_bytes >= 1) {
            std::vector<int> new_iter_blocks;
            compute_blocks(exec_cfg_, src_layout_, dst_layout_, new_iter_blocks,
                    loop_blocks, tg_blocks, cur_iter_bytes);
            if (!ir_utils::is_equal(new_iter_blocks, iter_blocks)) {
                iter_blocks = new_iter_blocks;
                break;
            }
            cur_iter_bytes /= 2;
        }
    }
    ir_error_not_expected();
}

bool reorder_ir_builder_t::try_build(const std::vector<int> &iter_blocks,
        const std::vector<int> &loop_blocks,
        const std::vector<int> &tg_blocks) {
    constraint_set_t init_cset;

    int ndims = src_layout_.ndims();
    std::vector<expr_t> vars;
    for (int i = 0; i < ndims; i++) {
        char letter = 'a' + i;
        vars.push_back(var_t::make(type_t::s32(), std::string(1, letter)));
    }

    std::vector<int> dim2grid;
    compute_grid(src_layout_, dst_layout_, iter_blocks, loop_blocks, tg_blocks,
            kernel_grid_, tg_grid_, &dim2grid);

    std::vector<stmt_t> init_stmts;
    init_kernel_grid(
            kernel_grid_, tg_grid_, exec_cfg_.simd(), init_cset, init_stmts);

    auto &x = view_t::placeholder_var();

    std::vector<dim_t> vdims(ndims);
    for (int i = 0; i < ndims; i++) {
        vdims[i] = std::max(src_layout_.dim(i), dst_layout_.dim(i));
    }

    view_t src_view(vars, ndims);
    for (int i = 0; i < ndims; i++) {
        int dim = src_layout_.dim(i);
        src_view.set_vdim(vars[i], vdims[i]);
        expr_t mask(true);
        if (dim != vdims[i]) mask = x < dim;
        src_view.set_tdim(i, vars[i], mask);
    }
    src_view.set_tlayout(src_layout_);

    view_t dst_view(vars, ndims);
    for (int i = 0; i < ndims; i++) {
        int dim = dst_layout_.dim(i);
        dst_view.set_vdim(vars[i], vdims[i]);
        expr_t mask(true);
        if (dim != vdims[i]) mask = x < dim;
        dst_view.set_tdim(i, vars[i], mask);
    }
    dst_view.set_tlayout(dst_layout_);

    gemm_schedule_t schedule(init_cset, kernel_grid_, tg_grid_);

    schedule.set_view(src_view);
    schedule.set_view(dst_view);

    std::array<std::vector<expr_t>, 3> fused_idxs;
    for (int i = 0; i < ndims; i++) {
        std::vector<expr_t> ordered;
        auto v = vars[i];
        if (iter_blocks[i] != 1) {
            expr_t outer, inner;
            schedule.split(v, iter_blocks[i], outer, inner);
            schedule.tensorize(inner);
            v = outer;
            ordered.insert(ordered.begin(), outer);
        }
        if (loop_blocks[i] != 1) {
            if (!ordered.empty()) ordered.erase(ordered.begin());
            expr_t outer, inner;
            schedule.split(v, loop_blocks[i], outer, inner);
            v = outer;
            ordered.insert(ordered.begin(), inner);
            ordered.insert(ordered.begin(), outer);
        }
        if (tg_blocks[i] != 1) {
            if (!ordered.empty()) ordered.erase(ordered.begin());
            expr_t outer, inner;
            schedule.split(v, tg_blocks[i], outer, inner);
            schedule.bind(inner, tg_grid_.idx(dim2grid[i]));
            v = outer;
            ordered.insert(ordered.begin(), inner);
            ordered.insert(ordered.begin(), outer);
        }
        fused_idxs[dim2grid[i]].push_back(v);
        schedule.reorder(ordered);
    }

    for (int i = 0; i < (int)fused_idxs.size(); i++) {
        auto &vec = fused_idxs[i];
        if (vec.empty()) continue;
        auto var = (vec.size() == 1 ? vec[0] : schedule.fuse(vec));
        schedule.bind(var, kernel_grid_.idx(i));
    }

    schedule.finalize();

    auto thr_tile = schedule.thr_view_tile(src_view, /*is_relative=*/false);

    auto src_thr_view = src_view.create_sub_view(thr_tile);
    auto dst_thr_view = dst_view.create_sub_view(thr_tile);

    auto src_buf = kernel_info_.arg_var(0);
    auto dst_buf = kernel_info_.arg_var(1);

    ir_context_t ir_ctx(exec_cfg_, init_cset);
    auto reg_buf = ir_ctx.create_tmp_var(type_t::byte_ptr(), "reg");

    std::vector<stmt_t> allocs;
    for (int i = 0; i < kernel_info_.nargs(); i++) {
        auto &var = kernel_info_.arg_var(i);
        if (!var.type().is_ptr()) continue;
        allocs.push_back(alloc_t::make(var, 0, alloc_kind_t::global));
    }

    auto read = make_access_builder(ir_ctx, src_thr_view, src_buf, reg_buf,
            send_op_t::load, send_address_t::a64);
    auto read_stmt = read.stmt();

    auto write = make_access_builder(ir_ctx, dst_thr_view, dst_buf, reg_buf,
            send_op_t::store, send_address_t::a64);
    auto write_stmt = write.stmt();

    auto read_layout = read.reg_layout();
    auto write_layout = write.reg_layout();
    allocs.push_back(
            alloc_t::make(reg_buf, read.reg_buf_size(), alloc_kind_t::grf));

    if (read_layout != write_layout) {
        auto tmp_buf = ir_ctx.create_tmp_var(type_t::byte_ptr(), "tmp");
        allocs.push_back(alloc_t::make(
                tmp_buf, write.reg_buf_size(), alloc_kind_t::grf));

        auto reorder_stmt = create_reorder_stmt(
                read_layout, write_layout, reg_buf, tmp_buf);
        write_stmt = substitute(write_stmt, reg_buf, tmp_buf);
        write_stmt = reorder_stmt.append(write_stmt);
    }

    stmt_ = stmt_t();
    stmt_ = stmt_.append(read_stmt);
    stmt_ = stmt_.append(write_stmt);

    stmt_ = schedule.create_loop_nest(stmt_);
    stmt_ = schedule.create_bind_stmt(stmt_);
    stmt_ = inject_let_stmts(stmt_, init_stmts);
    stmt_ = inject_alloc_stmts(stmt_, allocs);
    stmt_ = inject_external_var_let(stmt_, ir_ctx);

    stmt_ = simplify(stmt_, ir_ctx);
    stmt_ = lift_buffer_offsets_in_send(stmt_, ir_ctx);
    stmt_ = inject_send(stmt_, ir_ctx);
    stmt_ = split_wide_stores(stmt_, ir_ctx);
    stmt_ = fix_int32_overflow(stmt_, ir_ctx);
    stmt_ = eliminate_common_subexprs(
            stmt_, ir_ctx, exec_cfg_.regs() * exec_cfg_.grf_size());
    stmt_ = simplify(stmt_, ir_ctx);
    stmt_ = optimize_alloc_let(stmt_, ir_ctx);
    stmt_ = stmt_group_t::make(stmt_label_t::kernel(), stmt_);

    int ir_usage = get_peak_grf_usage(stmt_, exec_cfg_.grf_size());
    int reserved_usage = 16;
    int grf_usage = ir_usage + reserved_usage;
    if (grf_usage > exec_cfg_.regs()) {
        ir_warning()
                << "Estimated GRF usage is " << grf_usage
                << " which exceeds available space, retry with a smaller tile."
                << std::endl;

        return false;
    }

    ir_trace() << "Reorder kernel body:\n" << stmt_ << std::endl;
    return true;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
