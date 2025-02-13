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

#include "gpu/intel/jit/reorder/ir_builder.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>
#include <unordered_map>

#include "common/c_types_map.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/jit/ir/epilogue.hpp"
#include "gpu/intel/jit/ir/gemm_schedule.hpp"
#include "gpu/intel/jit/ir/ir.hpp"
#include "gpu/intel/jit/ir/message.hpp"
#include "gpu/intel/jit/ir/post_ops.hpp"
#include "gpu/intel/jit/ir/reorder.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/jit/pass/pass.hpp"
#include "gpu/intel/jit/reorder/tiler.hpp"
#include "gpu/intel/jit/utils/range.hpp"
#include "gpu/intel/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

void split_tile(const tensor_t &wg_tile, const tensor_t &iter_tile,
        pvar_tile_t &iter_dims, pvar_tile_t &loop_dims) {

    const auto &wg_dims = wg_tile.dims();
    const auto &it_dims = iter_tile.dims();
    for (dim_idx_t i = 0; i < wg_tile.ndims(); ++i) {
        pvar_t &d = reorder::pvars[i];
        iter_dims[d] = it_dims[i];
        loop_dims[d] = wg_dims[i] / it_dims[i];
    }
}

void reorder_ir_builder_t::build() {
    const auto &wg_block = cfg_.tiles().front();
    auto it = cfg_.tiles().begin();
    const auto end = cfg_.tiles().end();

    int max_iters = 10;
    pvar_tile_t iter_tile, loop_tile;
    for (int i = 0; i < max_iters && it != end; i++, it++) {
        if (!wg_block.is_divisible(*it)) continue;
        split_tile(wg_block, *it, iter_tile, loop_tile);
        if (try_build(iter_tile, loop_tile)) {
            gpu_info() << "Reorder configuration:";
            gpu_info() << "  Source layout:       " << cfg_.src_layout().user();
            gpu_info() << "  Destination layout:  " << cfg_.dst_layout().user();
            gpu_info() << "  Iteration blocks:    " << iter_tile;
            gpu_info() << "  Loop blocks:         " << loop_tile;
            gpu_info() << "  Thread group blocks: " << cfg_.thread_group_dims();
            return;
        }
    }
    gpu_error_not_expected();
}

bool reorder_ir_builder_t::try_build(
        const pvar_tile_t &iter_tile, const pvar_tile_t &loop_tile) {
    constraint_set_t init_cset;

    const auto &padded_dims = cfg_.padded_dims().get();
    auto string_map = padded_dims.to_string_map();
    const auto ndims = cfg_.src_layout().user().ndims();

    std::vector<expr_t> vars;
    vars.reserve(ndims);
    for (auto &d : padded_dims) {
        auto var = var_t::make(type_t::s32(), d.name());
        vars.emplace_back(var);
    }

    std::vector<stmt_t> init_stmts;
    init_kernel_grid(cfg_.kernel_grid(), cfg_.thread_group_grid(),
            cfg_.exec_cfg().simd(), init_cset, init_stmts);

    view_t src_view(vars, ndims);
    view_t dst_view(vars, ndims);
    for (dim_idx_t i = 0; i < ndims; ++i) {
        const auto &d = reorder::pvars[i];
        expr_t &var = vars[i];
        dim_t vdim = padded_dims[d];
        src_view.set_vdim(var, vdim);
        dst_view.set_vdim(var, vdim);
        src_view.set_tdim(i, var);
        dst_view.set_tdim(i, var);
    }
    src_view.set_tlayout(cfg_.src_layout().user());
    dst_view.set_tlayout(cfg_.dst_layout().user());
    src_view.set_tmasks(string_map);
    dst_view.set_tmasks(string_map);

    gemm_schedule_t schedule(
            init_cset, cfg_.kernel_grid(), cfg_.thread_group_grid());

    schedule.set_view(src_view);
    schedule.set_view(dst_view);

    std::array<std::vector<expr_t>, 3> fused_idxs;
    auto find_grid_idx = [&](const pvar_t &d) {
        for (dim_idx_t grid_idx = 0; grid_idx < 3; ++grid_idx) {
            const auto &grid = cfg_.grid()[grid_idx];
            for (auto &grid_dim : grid)
                if (grid_dim == d) return grid_idx;
        }
        gpu_error_not_expected();
        return dim_idx::invalid;
    };
    for (dim_idx_t i = 0; i < ndims; i++) {
        std::vector<expr_t> ordered;
        auto v = vars[i];
        const auto &d = reorder::pvars[i];
        auto grid_idx = find_grid_idx(d);
        const auto &iter_dim = iter_tile[d];
        const auto &loop_dim = loop_tile[d];
        const auto &tg_dim = cfg_.thread_group_dims().get(d);
        if (iter_dim != 1) {
            expr_t outer, inner;
            schedule.split(v, iter_dim, outer, inner);
            schedule.tensorize(inner);
            v = outer;
            ordered.insert(ordered.begin(), outer);
        }
        if (loop_dim != 1) {
            if (!ordered.empty()) ordered.erase(ordered.begin());
            expr_t outer, inner;
            schedule.split(v, loop_dim, outer, inner);
            v = outer;
            ordered.insert(ordered.begin(), inner);
            ordered.insert(ordered.begin(), outer);
        }
        if (tg_dim != 1) {
            if (!ordered.empty()) ordered.erase(ordered.begin());
            expr_t outer, inner;
            schedule.split(v, tg_dim, outer, inner);
            schedule.bind(inner, cfg_.thread_group_grid().idx(grid_idx));
            v = outer;
            ordered.insert(ordered.begin(), inner);
            ordered.insert(ordered.begin(), outer);
        }
        schedule.reorder(ordered);
        fused_idxs[grid_idx].push_back(v);
    }

    for (dim_idx_t i = 0; i < into<dim_idx_t>(fused_idxs.size()); i++) {
        auto &vec = fused_idxs[i];
        if (vec.empty()) continue;
        auto var = (vec.size() == 1 ? vec[0] : schedule.fuse(vec));
        schedule.bind(var, cfg_.kernel_grid().idx(i));
    }

    schedule.finalize();

    auto thr_tile = schedule.thr_view_tile(src_view, /*is_relative=*/false);
    auto src_thr_view = src_view.create_sub_view(thr_tile);
    auto dst_thr_view = dst_view.create_sub_view(thr_tile);

    auto src_buf = kernel_info_.arg_var(0);
    auto dst_buf = kernel_info_.arg_var(1);

    ir_context_t ir_ctx(cfg_.exec_cfg(), init_cset);
    auto reg_buf = ir_ctx.create_tmp_var(type_t::byte_ptr(), "reg");

    std::vector<stmt_t> allocs;
    for (int i = 0; i < kernel_info_.nargs(); i++) {
        auto &var = kernel_info_.arg_var(i);
        if (!var.type().is_ptr()) continue;
        allocs.push_back(alloc_t::make(var, 0, alloc_kind_t::global));
    }

    auto read_params = get_send_params(cfg_.exec_cfg(), send_op_t::load,
            send_address_t::a64, src_thr_view, true);
    read_params.try_legacy = false;
    auto read = make_access_builder(
            ir_ctx, src_thr_view, src_buf, reg_buf, read_params);
    auto &read_stmt = read.stmt();

    auto write_params = get_send_params(cfg_.exec_cfg(), send_op_t::store,
            send_address_t::a64, dst_thr_view, true);
    write_params.try_legacy = false;
    auto write = make_access_builder(
            ir_ctx, dst_thr_view, dst_buf, reg_buf, write_params);
    auto write_stmt = write.stmt();

    auto &read_layout = read.reg_layout();
    auto &write_layout = write.reg_layout();
    int read_buf_size = read.reg_buf_size();
    int write_buf_size = write.reg_buf_size();

    bool has_post_ops = dst_md_ && attr_
            && (!attr_->post_ops_.has_default_values()
                    || !attr_->zero_points_.has_default_values()
                    || !attr_->scales_.has_default_values()
                    || !attr_->rounding_mode_.has_default_values());

    if (has_post_ops) {
        post_op_view_mapper_t view_mapper(dst_view);
        post_op_context_t post_op_ctx(*attr_, cfg_.zp_cfg(), schedule,
                kernel_info_, *dst_md_, *dst_md_, view_mapper);
        write_stmt = create_epilogue_stmt(cfg_.exec_cfg(), ir_ctx, schedule,
                /*force_c_reorder=*/true, post_op_ctx, thr_tile, read_layout,
                dst_buf, reg_buf, write_buf_size);
    } else if (read_layout != write_layout) {
        auto tmp_buf = ir_ctx.create_tmp_var(type_t::byte_ptr(), "tmp");
        allocs.push_back(
                alloc_t::make(tmp_buf, write_buf_size, alloc_kind_t::grf));
        auto reorder_stmt = create_reorder_stmt(
                read_layout, write_layout, reg_buf, tmp_buf);
        write_stmt = substitute(write_stmt, reg_buf, tmp_buf);
        write_stmt = reorder_stmt.append(write_stmt);
    } else {
        read_buf_size = std::max(read_buf_size, write_buf_size);
    }

    allocs.push_back(alloc_t::make(reg_buf, read_buf_size, alloc_kind_t::grf));

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
            stmt_, ir_ctx, cfg_.exec_cfg().regs() * cfg_.exec_cfg().grf_size());
    stmt_ = simplify(stmt_, ir_ctx);
    stmt_ = optimize_alloc_let(stmt_, ir_ctx);
    stmt_ = stmt_group_t::make(stmt_label_t::kernel(), stmt_);

    int ir_regs = get_peak_regs(stmt_, cfg_.exec_cfg().grf_size());
    int reserved_regs = 16;
    int regs = ir_regs + reserved_regs;
    if (regs > cfg_.exec_cfg().regs()) {
        gpu_warning() << "Estimated GRF usage is " << regs
                      << " registers which exceeds available space, retry with "
                         "a smaller tile.";

        return false;
    }

    gpu_trace() << "Reorder kernel body:\n" << stmt_;
    return true;
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
