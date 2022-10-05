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

#include "gpu/jit/pass/slm.hpp"

#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/ir/reorder.hpp"
#include "gpu/jit/ir/tensor.hpp"
#include "gpu/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class slm_buffer_merger_t : public ir_mutator_t {
public:
    slm_buffer_merger_t() {
        slm_base_ = make_buffer("slm");
        slm_off_.push_back(0);
    }

    const expr_t &slm_base() const { return slm_base_; }

    int slm_size() const { return slm_size_; }

    object_t _mutate(const alloc_t &obj) override {
        if (obj.kind != alloc_kind_t::slm) return ir_mutator_t::_mutate(obj);

        auto new_buf = push(obj);
        auto new_obj = ir_mutator_t::_mutate(obj);
        pop();

        auto &alloc = new_obj.as<alloc_t>();
        new_obj = substitute(alloc.body, alloc.buf, new_buf);

        return new_obj;
    }

private:
    expr_t push(const alloc_t &obj) {
        int cur_off = slm_off_.back();
        expr_t new_buf = slm_base_ + cur_off;
        slm_off_.push_back(cur_off + obj.size);
        slm_size_ = std::max(slm_size_, cur_off + obj.size);
        return new_buf;
    }

    void pop() { slm_off_.pop_back(); }

    expr_t slm_base_;
    std::vector<int> slm_off_;
    int slm_size_ = 0;
};

stmt_t merge_slm_buffers(const stmt_t &_stmt, ir_context_t &ir_ctx) {
    trace_start();
    stmt_t stmt = _stmt;
    slm_buffer_merger_t merger;
    stmt = merger.mutate(stmt);
    stmt = alloc_t::make(
            merger.slm_base(), merger.slm_size(), alloc_kind_t::slm, stmt);
    trace_pass("merge_slm_buffers", stmt, ir_ctx);
    return stmt;
}

class slm_reorder_injector_t : public ir_mutator_t {
public:
    slm_reorder_injector_t(
            const stmt_t &root, ngen::HW hw, const grid_info_t &tg_grid)
        : hw_(hw), tg_grid_(tg_grid) {
        alloc_manager_t alloc_mgr(root);
        auto slm_buffers = alloc_mgr.find_buffers(alloc_kind_t::slm);
        ir_assert(slm_buffers.size() == 1);
        slm_base_ = slm_buffers[0];
        slm_size_ = alloc_mgr.total_size(alloc_kind_t::slm);
    }

    const expr_t &slm_base() const { return slm_base_; }

    int slm_size() const { return slm_size_; }

    object_t _mutate(const func_call_t &obj) override {
        if (!is_func_call<reorder_t>(obj)) return obj;

        auto &call = obj.as<func_call_t>();

        auto stmt = create_slm_reorder(call.func.as<reorder_t>(),
                reorder_t::arg_src_buf(call), reorder_t::arg_dst_buf(call));
        if (stmt.is_empty()) return obj;
        return std::move(stmt);
    }

private:
    stmt_t create_slm_reorder(const reorder_t &reorder, const expr_t &src_buf,
            const expr_t &dst_buf) {
        auto src = reorder.src_layout;
        auto dst = reorder.dst_layout;
        if (!src.is_dense() || !dst.is_dense()) return stmt_t();

        layout_t::try_reinterpret_to_wider_type(src, dst);
        if (src.type() != dst.type()) return stmt_t();
        if (src.type().size() != 4) return stmt_t();

        layout_iterator_t src_it(src);
        layout_iterator_t dst_it(dst);

        tensor_t max_tile;
        for (;;) {
            auto src_tile = src_it.tile();
            auto dst_tile = dst_it.tile();
            if (src_tile.is_equal(dst_tile)) {
                auto s = src.map(src_it.tile());
                auto d = dst.map(dst_it.tile());
                if (s.is_dense() && d.is_dense()
                        && src_it.outer_layout() == dst_it.outer_layout()) {
                    if (is_slm_reorder_ok(s, d)) { max_tile = src_tile; }
                }
                if (!src_it.has_next() || !dst_it.has_next()) break;
                ++src_it;
                ++dst_it;
            } else {
                if (src_tile.elems() <= dst_tile.elems()) {
                    if (!src_it.has_next()) break;
                    ++src_it;
                } else {
                    if (!dst_it.has_next()) break;
                    ++dst_it;
                }
            }
        }

        if (max_tile.is_empty()) return stmt_t();

        return create_slm_reorder(max_tile, src, dst, src_buf, dst_buf);
    }

    stmt_t create_slm_reorder(const tensor_t &tile, const layout_t &src,
            const layout_t &dst, const expr_t &src_buf, const expr_t &dst_buf) {
        auto src_tile = src.map(tile);
        auto &src_tile_blocks = src_tile.blocks();
        int simd = src_tile_blocks[0].block;
        int vect_size = src_tile_blocks[1].block;
        int tile_size = simd * vect_size * src.type().size();
        int slm_thr_size = (int)src.size();
        int dword_size = type_t::dword().size();
        int hword_size = type_t::hword().size();
        int hwords = tile_size / hword_size;

        ir_assert(tile_size % hword_size == 0);

        slm_size_ = std::max(slm_size_, slm_thr_size * tg_grid_.elems());

        auto store_send = send_t::make(hw_, send_op_t::store,
                send_address_t::slm, type_t::dword(vect_size), simd);
        auto load_send = send_t::make(hw_, send_op_t::load, send_address_t::slm,
                type_t::hword(hwords), 1);

        std::vector<expr_t> vec(simd);
        for (int i = 0; i < simd; i++)
            vec[i] = expr_t(i * vect_size * dword_size);
        auto vec_off = shuffle_t::make(vec);
        auto tid = tg_grid_.idx(1) * tg_grid_.dim(0) + tg_grid_.idx(0);
        expr_t off0 = tid * slm_thr_size;

        stmt_t store_stmt;
        stmt_t load_stmt;
        src.for_each_tile(tile, [&](const std::vector<dim_t> &start) {
            expr_t off = (int)src.offset_in_bytes(start);
            auto store = store_send.call({slm_base_,
                    shuffle_t::make_broadcast(off0 + off, simd) + vec_off,
                    src_buf + off, expr_t()});
            auto load = load_send.call(
                    {slm_base_, off0 + off, dst_buf + off, expr_t()});
            store_stmt = store_stmt.append(store);
            load_stmt = load_stmt.append(load);
        });

        auto ret = store_stmt.append(load_stmt);
        return ret;
    }

    bool is_slm_reorder_ok(const layout_t &src, const layout_t &dst) const {
        auto &src_blocks = src.blocks();
        auto &dst_blocks = dst.blocks();
        if (src_blocks.size() != 2 || dst_blocks.size() != 2) return false;
        auto &s0 = src_blocks[0];
        auto &s1 = src_blocks[1];
        auto &d0 = dst_blocks[0];
        auto &d1 = dst_blocks[1];

        if (s0.dim_idx != d1.dim_idx || s1.dim_idx != d0.dim_idx) return false;
        ir_assert(s0.block == d1.block);
        ir_assert(s1.block == d0.block);

        int simd = s0.block;
        int vec_size = s1.block;
        if (!utils::one_of(simd, 16)) return false;
        if (!utils::one_of(vec_size, 8)) return false;

        return true;
    }

    ngen::HW hw_;
    grid_info_t tg_grid_;

    expr_t slm_base_;
    int slm_size_ = 0;
};

stmt_t inject_slm_reorder(const stmt_t &s, ir_context_t &ir_ctx,
        const grid_info_t &tg_grid, bool has_slm_usage) {
    trace_start();
    if (has_slm_usage) return s;
    if (ir_ctx.hw() < ngen::HW::XeHPC) return s;
    slm_reorder_injector_t injector(s, ir_ctx.hw(), tg_grid);
    stmt_t ret = injector.mutate(s);

    auto &slm_buf = injector.slm_base();
    int slm_size = injector.slm_size();
    alloc_updater_t alloc_updater;
    alloc_updater.resize(slm_buf, slm_size);
    ret = alloc_updater.update(ret);

    trace_pass("inject_slm_reorder", ret, ir_ctx);
    return ret;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
