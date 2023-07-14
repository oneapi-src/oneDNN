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

#include "gpu/jit/ir/epilogue.hpp"

#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/ir/reduce.hpp"
#include "gpu/jit/ir/reorder.hpp"
#include "gpu/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Zero pads a register buffer of f32 type.
class zero_pad_builder_t {
public:
    zero_pad_builder_t() = default;

    zero_pad_builder_t(ir_context_t &ir_ctx, const view_t &full_mem_view,
            const view_t &mem_view)
        : ir_ctx_(&ir_ctx)
        , full_mem_view_(full_mem_view)
        , mem_view_(mem_view) {}

    bool is_empty() const { return mem_view_.is_empty(); }

    expr_t create_mask(const layout_t &reg_layout, const tensor_t &tile) const {
        ir_assert(!is_empty());
        auto layout = reg_layout.map(tile);
        auto view = mem_view_.create_sub_view(tile);
        mask_tensor_t mask_tensor(layout);
        std::vector<dim_t> args(layout.ndims());
        fill_mask_impl(mask_tensor, 0, args, view, layout);
        mask_tensor.simplify(ir_ctx_->cset());
        return mask_tensor.to_expr(tile.elems());
    }

    stmt_t build_stmt(const layout_t &reg_layout, const expr_t &reg_buf) const {
        ir_assert(mem_view_.nvdims() == reg_layout.ndims())
                << "Incompatible view/layout.";
        int max_step = std::min(
                16, 2 * ir_ctx_->grf_size() / reg_layout.type().size());
        auto base_tile = reg_layout.split_into_max_tile(
                max_step, /*is_dense_tile=*/true);
        stmt_t stmt;
        reg_layout.for_each_tile(
                base_tile, [&](const std::vector<dim_t> &start) {
                    tensor_t tile(base_tile.dims(), start);
                    int off = reg_layout(start) * reg_layout.type().size();
                    auto mask = create_mask(reg_layout, tile);
                    auto zero = to_expr(0, reg_layout.type());
                    auto store = store_t::make(reg_buf, off,
                            shuffle_t::make_broadcast(zero, tile.elems()),
                            store_t::default_stride, -mask);
                    stmt = stmt.append(store);
                });
        return stmt;
    }

private:
    void fill_mask_impl(mask_tensor_t &mask_tensor, int idx,
            std::vector<dim_t> &args, const view_t &view,
            const layout_t &layout) const {
        if (idx == layout.ndims()) {
            std::vector<expr_t> vargs;
            for (int i = 0; i < layout.ndims(); i++)
                vargs.push_back(view.vstart(i) + args[i]);
            expr_t mask = full_mem_view_.vmask(vargs);
            auto off = layout.offset(args, /*ignore_offset=*/true);
            mask_tensor.set_mask(off, mask);
            return;
        }

        for (int i = 0; i < int(layout.dims()[idx]); i++) {
            args[idx] = i;
            fill_mask_impl(mask_tensor, idx + 1, args, view, layout);
        }
    }

    ir_context_t *ir_ctx_;

    view_t full_mem_view_;
    view_t mem_view_;

    stmt_t stmt_;
};

// Represents the state of a post-op tensor.
//
// There are three kinds of tensors:
// - C tensor converted to f32
//   - Never loaded or stored to global memory
// - Input tensor
//   - No store, only load
// - Output tensor
//   - No load, only store
//
// Post-op tensors that are both input/output are not expected/supported as
// they doesn't occur in convolution. Post-op tensors with global reduction
// (like lhs += rhs) are treated as output-only and handled via atomic stores.
//
// A post-op tensor optionally requires:
// - Conversion to f32 (post-ops are done in f32)
// - Reduction
//   - For output tensors with broadcast dimensions
// - Masking during post-ops
//   - When a post-op is not zero preserving
class post_op_tensor_t {
public:
    post_op_tensor_t(ir_context_t &ir_ctx, const post_op_tensor_info_t &info)
        : ir_ctx_(&ir_ctx), info_(info) {
        if (!mem_buf().is_empty()) {
            auto &type = mem_buf().type();
            if (!type.is_ptr()) {
                ir_assert(type.is_f32()) << "Expected f32: " << mem_buf();
                reg_buf_ = mem_buf();
                reg_layout_ = layout_t(
                        type, 0, std::vector<dim_t>(mem_view().nvdims(), 1));
            }
        }
    }

    const view_t &mem_view() const { return info_.view(); }

    const expr_t &mem_buf() const { return info_.buf(); }

    // Bitmask with broadcast information for the tensor:
    // - (mask() & (1 << idx)) == 0 -> idx is a brodcast dimension (equal to 1)
    // - (mask() & (1 << idx)) != 0 -> idx dimension matches the C dimension
    uint32_t mask() const { return info_.mask(); }

    // Placeholder variable to represent the tensor in post-op expressions.
    const expr_t &op_var() const { return info_.op_var(); }

    const layout_t &reg_layout() const { return reg_layout_; }

    const expr_t &reg_buf() const { return reg_buf_; }

    post_op_tensor_t create_sub_tensor(const tensor_t &_tile) const {
        auto ret = *this;
        auto tile = apply_mask(_tile);
        ret.info_ = ret.info_.create_sub_tensor(tile);
        if (!reg_layout_.is_empty()) {
            if (needs_reduction()) {
                tensor_t reduce_tile(_tile.dims(), tile.start());
                ret.reg_layout_ = ret.reg_layout_.map(reduce_tile);
            } else {
                ret.reg_layout_ = ret.reg_layout_.map(tile);
            }
        }
        ret.allocs_.clear();
        return ret;
    }

    bool is_f32_scalar() const {
        return mem_view().type().is_f32() && mem_view().velems() == 1;
    }

    bool needs_load() const {
        if (!info_.is_input()) return false;
        if (!mem_buf().type().is_ptr()) return false;
        return true;
    }

    bool needs_store() const { return info_.is_output(); }

    bool needs_masked_update() const { return info_.needs_masked_update(); }

    bool needs_f32_convert() const {
        return !mem_view().type().is_f32() && !mem_view().type().is_f64();
    }

    bool needs_reduction() const {
        if (!info_.is_output()) return false;

        for (int i = 0; i < mem_view().nvdims(); i++) {
            if (is_broadcast_dim(i)) {
                if (reg_layout_.dims()[i] != 1) return true;
            }
        }
        return false;
    }

    bool needs_compute() const { return info_.needs_compute(); }

    const expr_t &compute_expr() const { return info_.compute_expr(); }

    bool is_broadcast_dim(int dim_idx) const {
        ir_assert(dim_idx >= 0 && dim_idx < mem_view().nvdims());
        return (mask() & (1 << dim_idx)) == 0;
    }

    int estimate_grf_consumption() const {
        int elems = int(mem_view().create_dense_vlayout().elems());

        int ret = 0;
        ret += elems * mem_view().type().size();
        if (needs_f32_convert()) ret += elems * type_t::f32().size();
        return ret;
    }

    void set_reg_layout(const layout_t &layout) { reg_layout_ = layout; }

    void set_reg_buf(const expr_t &buf) { reg_buf_ = buf; }

    void set_preload(bool value = true) { do_preload_ = value; }

    bool do_preload() const { return do_preload_; }

    tensor_t apply_mask(const tensor_t &tile) const {
        ir_assert(mem_view().nvdims() == tile.ndims());

        auto start = tile.start();
        auto dims = tile.dims();

        for (int i = 0; i < tile.ndims(); i++) {
            if (!is_broadcast_dim(i)) continue;
            start[i] = expr_t(0);
            dims[i] = 1;
        }
        return tensor_t(dims, start);
    }

    void init_output_buffer(const tensor_t &tile) {
        ir_assert(needs_store());

        ir_assert(reg_layout_.is_empty());
        ir_assert(reg_buf_.is_empty());

        reg_buf_ = make_tmp_reg_buffer();

        reg_layout_ = mem_view().create_dense_vlayout();
        reg_layout_ = reg_layout_.retype(type_t::f32());

        // If this is output and there are masked dimensions then this buffer
        // is computed via reduction. Extend layout to cover full masked_tile
        // and apply the final reduction after all tiles.
        auto masked_tile = apply_mask(tile);
        for (int i = 0; i < masked_tile.ndims(); i++) {
            if (masked_tile(i) >= tile(i)) continue;
            ir_assert(masked_tile(i) == 1) << "Unexpected output tensor shape.";
            reg_layout_ = reg_layout_.add_outer_block(i, tile(i));
        }
        register_buffer(reg_buf_, reg_layout_.size());
    }

    stmt_t build_load_stmt(const view_t &c_view) {
        ir_assert(needs_load());
        ir_assert(reg_buf_.is_empty());

        reg_buf_ = make_tmp_reg_buffer();
        auto read = make_access_builder(*ir_ctx_, mem_view(), mem_buf(),
                reg_buf_, send_op_t::load, send_address_t::a64,
                get_cache_hint(c_view));
        reg_layout_ = read.reg_layout();
        register_buffer(reg_buf_, read.reg_buf_size());
        return read.stmt();
    }

    stmt_t build_prefetch_stmt(const view_t &c_view) const {
        ir_assert(needs_load());

        auto prefetch = make_access_builder(*ir_ctx_, mem_view(), mem_buf(),
                expr_t(), send_op_t::prefetch, send_address_t::a64,
                get_cache_hint(c_view));
        return prefetch.stmt();
    }

    stmt_t build_convert_stmt() {
        if (!needs_load() || !needs_f32_convert()) return stmt_t();

        auto f32_buf = make_tmp_reg_buffer();
        auto f32_layout = reg_layout_.retype(type_t::f32()).make_dense();

        register_buffer(f32_buf, f32_layout.size());

        // Reorder to f32.
        auto ret = create_reorder_stmt(
                reg_layout_, f32_layout, reg_buf_, f32_buf);

        // Assign new f32 layout and buffer.
        reg_layout_ = f32_layout;
        reg_buf_ = f32_buf;

        return ret;
    }

    stmt_t build_compute_stmt(const std::vector<post_op_tensor_t> &tensors) {
        ir_assert(needs_compute());
        ir_assert(is_f32_scalar()) << "Only f32 scalars are supported.";
        reg_layout_ = mem_view().create_pseudo_vlayout();
        auto e = compute_expr();
        tensor_t tile(std::vector<dim_t>(reg_layout_.ndims(), 1));
        for (auto &t : tensors) {
            if (contains_object(e, t.op_var())) {
                ir_assert(t.is_f32_scalar())
                        << "All tensors in the compute expression must be f32 "
                           "scalars.";
                ir_assert(t.do_preload()) << "All tensors in the compute "
                                             "expression must be preloaded.";
                e = substitute(e, t.op_var(), t.load_expr(tile, 0));
            }
        }
        reg_buf_ = make_tmp_reg_buffer();
        register_buffer(reg_buf_, reg_layout_.size());
        return store_t::make(reg_buf_, 0, e);
    }

    stmt_t build_zero_out_stmt() const {
        ir_assert(needs_store());
        return funcs::zero_out(reg_buf_, reg_layout_.size());
    }

    stmt_t build_reduce_stmt() {
        ir_assert(needs_store());

        stmt_t stmt;

        if (needs_reduction()) {
            auto reduced_layout = mem_view().create_dense_vlayout();
            ir_assert(reduced_layout.size() <= reg_layout_.size());

            stmt = stmt.append(
                    create_reduce_stmt(reg_layout_, reduced_layout, reg_buf_,
                            reg_buf_, tensor_t(), mask(), /*drop_dims=*/false));
            reg_layout_ = reduced_layout;
        }

        return stmt;
    }

    stmt_t build_slm_store_stmt(const grid_info_t &tg_grid) {
        ir_assert(needs_store());
        tensor_t tile(mem_view().vdims());
        slm_reduce_builder_ = slm_reduce_builder_t(
                *ir_ctx_, tg_grid, reg_buf_, reg_layout_, tile, 1);
        return slm_reduce_builder_.store_stmt();
    }

    stmt_t build_slm_load_stmt() {
        ir_assert(needs_store());
        ir_assert(!slm_reduce_builder_.is_empty());

        reg_layout_ = slm_reduce_builder_.reg_layout();

        auto new_tile = slm_reduce_builder_.thr_tile();
        info_ = info_.create_sub_tensor(new_tile);

        auto &slm_allocs = slm_reduce_builder_.allocs();
        allocs_.insert(allocs_.end(), slm_allocs.begin(), slm_allocs.end());

        return slm_reduce_builder_.load_stmt();
    }

    stmt_t build_store_stmt() const {
        ir_assert(needs_store());

        auto write = make_access_builder(*ir_ctx_, mem_view(), mem_buf(),
                reg_buf(), send_op_t::atomic_fadd, send_address_t::a64);
        ir_assert(write.reg_layout() == reg_layout());

        return write.stmt();
    }

    expr_t load_expr(const tensor_t &tile, int dim_idx) const {
        auto &type = reg_layout_.type();
        int elems = is_broadcast_dim(dim_idx) ? 1 : tile.elems();
        int off = reg_layout_.offset_in_bytes(expr_cast<dim_t>(tile.start()));
        auto ret = (reg_buf_.type().is_ptr()
                        ? load_t::make(type.with_elems(elems), reg_buf_, off)
                        : reg_buf_);
        if (elems != tile.elems())
            ret = shuffle_t::make_broadcast(ret, tile.elems());
        return ret;
    }

    stmt_t store_stmt(const tensor_t &tile, int dim_idx, const expr_t &_value,
            const expr_t &mask = expr_t()) const {
        auto value = _value;
        ir_assert(!is_broadcast_dim(dim_idx));
        ir_assert(value.type().elems() == tile.elems());
        // Add cast for booleans for comparison ops.
        if (value.type().is_bool()) {
            value = cast(value, reg_layout_.type().with_elems(tile.elems()));
        }
        int off = reg_layout_.offset_in_bytes(expr_cast<dim_t>(tile.start()));
        auto ret = store_t::make(
                reg_buf_, off, value, store_t::default_stride, mask);
        return ret;
    }

    const std::vector<stmt_t> &allocs() const { return allocs_; }

private:
    expr_t make_tmp_reg_buffer() {
        auto *var = mem_buf().as_ptr<var_t>();
        if (!var) {
            auto *ptr = mem_buf().as_ptr<ptr_t>();
            if (ptr) var = ptr->base.as_ptr<var_t>();
        }
        if (!var && needs_compute()) var = op_var().as_ptr<var_t>();
        ir_assert(var) << "Can't extract variable from buffer: " << mem_buf();
        auto &name = var->name;
        return ir_ctx_->create_tmp_var(type_t::byte_ptr(), "tmp_" + name);
    }

    void register_buffer(const expr_t &buf, int size) {
        for (auto &_a : allocs_) {
            auto &a = _a.as<alloc_t>();
            if (a.buf.is_same(buf)) {
                if (size > a.size) {
                    _a = alloc_t::make(a.buf, a.size, a.kind, a.attrs);
                }
                return;
            }
        }
        allocs_.push_back(alloc_t::make(buf, size, alloc_kind_t::grf));
    }

    send_cache_hint_t get_cache_hint(const view_t &c_view) const {
        ir_assert(mem_view().nvdims() == c_view.nvdims());
        bool per_tensor = true;
        for (int i = 0; i < mem_view().nvdims(); i++) {
            if ((mask() & (1 << i)) != 0) continue;
            if (c_view.vdims()[i] == 1) continue;
            per_tensor = false;
            break;
        }
        if (per_tensor) return send_cache_hint_t::load_once;
        return send_cache_hint_t::undef;
    }

    ir_context_t *ir_ctx_ = nullptr;

    post_op_tensor_info_t info_;

    layout_t reg_layout_;
    expr_t reg_buf_;

    bool do_preload_ = false;

    std::vector<stmt_t> allocs_;

    slm_reduce_builder_t slm_reduce_builder_;
};

// Applies substitutions and broadcasts to generate the final post-op
// expression.
class post_op_bcast_mutator_t : public ir_mutator_t {
public:
    post_op_bcast_mutator_t(
            int elems, const object_map_t<object_t, object_t> &from2to)
        : elems_(elems), from2to_(from2to) {}

    object_t _mutate(const float_imm_t &obj) override {
        return make_bcast(obj);
    }

    object_t _mutate(const int_imm_t &obj) override {
        return make_bcast(float_imm_t::make(obj.value));
    }

    object_t _mutate(const var_t &obj) override {
        auto it = from2to_.find(obj);
        if (it != from2to_.end()) return make_bcast(it->second);

        ir_error_not_expected() << "Unknown variable.";
        return obj;
    }

private:
    object_t make_bcast(const expr_t &e) const {
        if (e.type().elems() == elems_) return e;
        ir_assert(e.type().elems() == 1);
        return shuffle_t::make_broadcast(e, elems_);
    }

    int elems_;
    object_map_t<object_t, object_t> from2to_;
};

// Builds statements to apply a post-op for a given tile.
class post_op_builder_t {
public:
    post_op_builder_t(ngen::HW hw, const post_op_t &post_op)
        : hw_(hw), post_op_(post_op) {}

    const post_op_t &post_op() const { return post_op_; }

    // Applies post-op for a single tile.
    stmt_t build_tile_stmt(const tensor_t &tile,
            const object_map_t<expr_t, post_op_tensor_t *> &args,
            const zero_pad_builder_t &zero_pad_builder) const {
        auto &lhs_tensor = *args.at(post_op_.lhs());
        if (!post_op_.eltwise().is_empty()) {
            // Apply eltwise post-op.
            ir_assert(post_op_.lhs().is_equal(post_op_.rhs()))
                    << "Only supported form is lhs = eltwise(lhs).";
            int lhs_size = lhs_tensor.reg_layout().size();
            int lhs_elems = lhs_size / int(sizeof(float));
            return post_op_.eltwise().call(
                    {expr_t(lhs_elems), lhs_tensor.reg_buf()});
        }

        int inner_dim_idx = -1;
        auto base_inner_tile = find_1d_tile(
                lhs_tensor.reg_layout().type(), args, inner_dim_idx);
        auto inner_layout = lhs_tensor.reg_layout().map(base_inner_tile);
        ir_assert(inner_dim_idx != -1);

        // All post-ops are performed in f32 except f64 bias.
        for (auto &kv : args) {
            ir_assert(kv.second->reg_layout().type().is_f32()
                    || kv.second->reg_layout().type().is_f64());
        }

        // Handle one inner tile at a time. Inner tile covers a single block
        // within a single dimension.
        stmt_t stmt;
        lhs_tensor.reg_layout().for_each_tile(
                base_inner_tile, [&](const std::vector<dim_t> &lhs_start) {
                    tensor_t inner_tile(base_inner_tile.dims(), lhs_start);
                    auto rhs_value = compute_post_op_expr(
                            post_op_.rhs(), inner_tile, inner_dim_idx, args);
                    auto &t = *args.at(post_op_.lhs());
                    expr_t store_mask;
                    if (lhs_tensor.needs_masked_update()) {
                        store_mask = zero_pad_builder.create_mask(
                                inner_layout, inner_tile);
                    }
                    auto inner_stmt = t.store_stmt(
                            inner_tile, inner_dim_idx, rhs_value, store_mask);
                    stmt = stmt.append(inner_stmt);
                });

        return stmt;
    }

private:
    // Returns a 1D tile corresponding to an instruction to partially apply the
    // post-op.
    tensor_t find_1d_tile(const type_t &lhs_type,
            const object_map_t<expr_t, post_op_tensor_t *> &args,
            int &inner_dim_idx) const {
        auto &lhs_tensor = *args.at(post_op_.lhs());

        ir_assert(!lhs_tensor.reg_layout().is_empty());
        std::vector<dim_t> dims(lhs_tensor.mem_view().nvdims(), 1);

        if (lhs_tensor.reg_layout().blocks().empty()) {
            for (dim_t d : lhs_tensor.mem_view().vdims())
                ir_assert(d == 1);
            inner_dim_idx = 0;
        } else {
            auto &b0 = lhs_tensor.reg_layout().blocks()[0];
            ir_assert(dim_t(b0.stride) == 1);
            inner_dim_idx = b0.dim_idx;

            int inner_block = b0.block;
            int max_step = 2 * ngen::GRF::bytes(hw_) / lhs_type.size();
            inner_block = std::max(8, math::gcd(inner_block, max_step));

            for (auto &kv : args) {
                auto &t = *kv.second;
                if (t.is_broadcast_dim(b0.dim_idx)) continue;

                auto &l = t.reg_layout();
                ir_assert(!l.is_empty());
                ir_assert(!l.blocks().empty());
                auto &lb0 = l.blocks()[0];
                ir_assert(lb0.dim_idx == b0.dim_idx);
                ir_assert(dim_t(lb0.stride) == 1);
                inner_block = math::gcd(int(lb0.block), inner_block);
            }
            dims[b0.dim_idx] = inner_block;
        }
        return tensor_t(dims);
    }

    expr_t compute_post_op_expr(const expr_t &expr, const tensor_t &tile,
            int dim_idx,
            const object_map_t<expr_t, post_op_tensor_t *> &args) const {
        object_map_t<object_t, object_t> sub_map;
        for (auto &kv : args) {
            auto &t = *kv.second;
            auto te = t.load_expr(tile, dim_idx);
            sub_map.insert({t.op_var(), te});
        }
        post_op_bcast_mutator_t bcast_mutator(tile.elems(), sub_map);
        return bcast_mutator.mutate(expr);
    }

    ngen::HW hw_;
    post_op_t post_op_;
};

int get_post_op_mem_usage(const post_op_tensor_info_t &info, int c_elems,
        const view_t &c_mem_view, int max_elems_per_dim = 64) {
    int po_elems = 1;
    for (int i = 0; i < info.view().nvdims(); i++) {
        if ((info.mask() & (1 << i)) == 0) continue;
        po_elems *= std::min(max_elems_per_dim, (int)c_mem_view.vdims()[i]);
    }
    po_elems = std::min(po_elems, c_elems);
    int type_size = info.view().type().size();
    int load_size = po_elems * type_size;
    int cvt_size = info.view().type().is_f32() ? 0 : po_elems * sizeof(float);
    return load_size + cvt_size;
}

int find_tile_size(const exec_config_t &exec_cfg,
        const post_op_context_t &post_op_ctx, const view_t &c_mem_view,
        const layout_t &c_reg_layout, int preload_max_size, int post_op_blk) {
    bool with_post_ops = !post_op_ctx.post_ops().empty();
    for (int tile_size = 1024; tile_size >= 1; tile_size /= 2) {
        int c_type_size = c_mem_view.type().size();
        int elems = tile_size / (with_post_ops ? sizeof(float) : c_type_size);
        int c_mul_size = elems * c_type_size;
        int c_f32_size = with_post_ops && !c_mem_view.type().is_f32()
                ? elems * sizeof(float)
                : 0;
        int c_size = c_mul_size + c_f32_size;
        int po_size = 0;

        auto &infos = post_op_ctx.post_op_tensor_infos();
        int npost_ops = int(infos.size());
        for (int i = 0; i < npost_ops; i += post_op_blk) {
            int po_batch_size = 0;
            for (int j = i; j < std::min(npost_ops, i + post_op_blk); j++) {
                auto &t = infos[j];
                if (!t.is_input() || !t.buf().type().is_ptr()) continue;
                po_batch_size += get_post_op_mem_usage(t, elems, c_mem_view);
            }
            po_size = std::max(po_size, po_batch_size);
        }

        int total_size = c_size + preload_max_size + po_size;
        int available_size = exec_cfg.regs() * exec_cfg.grf_size()
                - (int)c_reg_layout.size();
        if (total_size <= available_size * 0.8) return tile_size;
    }
    ir_error_not_expected();
    return -1;
}

// Epilogue consists of the following steps after the main computation (C += A * B):
// - C GRF reorder to match the memory layout for global memory store
// - C conversion to f32 (if there are post-ops)
// - Applying post-ops (if there are any)
// - C conversion to the memory layout data type
// - C store to global memory
// - Reduction and storing output post-op tensors
//
// In general C tensor is updated/transformed following the C stages described
// below. Each C stage is associated with GRF buffer and its layout.
//   Multiplication ->
//     M_x -> [R_f32] -> [P0_f32] -> ... -> [Pn_f32] -> [Z_f32] -> S_y ->
//   GMEM
//
// Where:
// - x      is data type after multiplication
// - y      is destination data type
// - M_x    is the stage after multiplication
// - R_f32  is the stage after reordering from M_x to f32 (optional)
// - Pi_f32 is the stage after applying Pi post-op (optional)
// - Z_f32  is the stage after restoring zero padding (optional)
// - S_y    is the stage before storing C to global memory
class epilogue_builder_t {
public:
    epilogue_builder_t(ir_context_t &ir_ctx, const exec_config_t &exec_cfg,
            const gemm_schedule_t &gemm_schedule, bool force_c_reorder,
            const post_op_context_t &post_op_ctx, const tensor_t &thr_tile,
            const view_t &c_mem_view, const layout_t &c_reg_layout,
            const expr_t &c_mem_buf, const expr_t &c_reg_buf,
            int preload_max_size, int post_op_blk)
        : ir_ctx_(ir_ctx)
        , gemm_schedule_(gemm_schedule)
        , post_op_ctx_(post_op_ctx)
        , c_mem_view_(c_mem_view)
        , c_mem_buf_(c_mem_buf)
        , force_c_reorder_(force_c_reorder)
        , restore_zero_padding_(post_op_ctx.need_to_restore_zero_padding())
        , preload_max_size_(preload_max_size)
        , post_op_blk_(post_op_blk) {

        int tensor_idx = 0;

        // Tile size in bytes. All post-ops are applied to a single tile, then
        // to the next tile, etc.
        tile_size_ = find_tile_size(exec_cfg, post_op_ctx_, c_mem_view_,
                c_reg_layout, preload_max_size_, post_op_blk_);

        ir_trace() << "Creating epilogue with parameters"
                   << ": tile_size = " << tile_size_
                   << ", preload_max_size = " << preload_max_size
                   << ", post_op_blk = " << post_op_blk << std::endl;

        for (auto &po_tensor_info : post_op_ctx_.post_op_tensor_infos()) {
            post_op_tensor_t po_tensor(ir_ctx_, po_tensor_info);
            po_tensor = po_tensor.create_sub_tensor(thr_tile);
            if (po_tensor_info.buf().is_empty()
                    && !po_tensor_info.needs_compute()) {
                // C tensor.
                ir_assert(c_po_idx_ == -1);
                c_po_idx_ = tensor_idx;
            }
            post_op_tensors_.push_back(po_tensor);
            tensor_idx++;
        }

        for (auto &po : post_op_ctx_.post_ops()) {
            post_op_builders_.emplace_back(ir_ctx_.hw(), po);
        }

        // Estimate buffer sizes required to load the full tensor, do not do
        // preload if it requires too much GRF memory.
        int available_size = preload_max_size_;
        for (auto &t : post_op_tensors_) {
            if (!t.needs_load()) continue;
            int required_size = t.estimate_grf_consumption();
            if (required_size > available_size) continue;
            available_size -= required_size;
            t.set_preload();
        }

        build(c_reg_layout, c_reg_buf);
    }

    const stmt_t &stmt() const { return stmt_; }

    int c_reg_buf_size() const { return c_reg_buf_size_; }

private:
    expr_t make_c_tmp_buffer() const {
        return ir_ctx_.create_tmp_var(type_t::byte_ptr(), "c_tmp");
    }

    // Represents a GRF buffer and layout to store C tensor.
    struct c_stage_t {
        c_stage_t(const layout_t &layout, int buf_size, const expr_t &buf,
                const stmt_t &stmt = stmt_t())
            : layout(layout), buf_size(buf_size), buf(buf), stmt(stmt) {}

        void set_next(
                ir_context_t &ir_ctx, c_stage_t *next, bool force_reorder) {
            if (!next) return;
            bool do_reorder
                    = !layout.is_equal(next->layout, /*compare_offset=*/false);
            if (force_reorder) do_reorder = true;
            if (do_reorder) {
                ir_assert(stmt.is_empty());
                // Generate reorder between stages.
                stmt = create_reorder_stmt(
                        layout, next->layout, buf, next->buf);
            } else {
                // Reuse the same GRF buffer for the next stage.
                int this_off = to_cpp<int>(layout.offset_in_bytes());
                int next_off = to_cpp<int>(next->layout.offset_in_bytes());
                ir_assert(next_off == 0);
                next->set_buf(buf[this_off]);
            }
        }

        void set_buf(const expr_t &buf) {
            // Replace old buffer if there is an assigned statement.
            if (!stmt.is_empty()) { stmt = substitute(stmt, this->buf, buf); }
            this->buf = buf;
        }

        const expr_t &buf_base() const {
            if (buf.is<var_t>()) return buf;
            return buf.as<ptr_t>().base;
        }

        int get_buf_size(bool check_base = true) const {
            if (check_base)
                ir_assert(buf.is_same(buf_base()))
                        << "Size must be queried from another stage.";
            return (buf_size == 0) ? int(layout.size()) : buf_size;
        }

        int max_off_bytes() const {
            int l_off_bytes = (int)layout.max_off_bytes(/*ignore_offset=*/true);
            return std::max(buf_size, l_off_bytes);
        }

        void prepend_stmt(const stmt_t &stmt) {
            this->stmt = stmt.append(this->stmt);
        }

        layout_t layout;
        int buf_size;
        expr_t buf;
        stmt_t stmt; // Statement to emit after the stage.
    };

    void build(const layout_t &c_reg_layout, const expr_t &c_reg_buf) {
        c_reg_buf_size_ = c_reg_layout.size();
        auto tmp_type = (post_op_builders_.empty() ? c_mem_view_.type()
                                                   : type_t::f32());
        int tmp_buf_elems = tile_size_ / tmp_type.size();
        auto base_tile = c_mem_view_.split_into_max_tile(
                tmp_buf_elems, /*is_dense=*/false);

        // Generate preload statements.
        for (auto &t : post_op_tensors_) {
            if (!t.do_preload()) continue;
            stmt_ = stmt_.append(t.build_load_stmt(c_mem_view_));
        }

        // Generate prefetch statements.
        if (ir_ctx_.hw() >= ngen::HW::XeHPC) {
            for (auto &t : post_op_tensors_) {
                if (!t.needs_load()) continue;
                if (t.do_preload()) continue;
                stmt_ = stmt_.append(t.build_prefetch_stmt(c_mem_view_));
            }
        }

        // Generate f32 convert statements.
        for (auto &t : post_op_tensors_) {
            if (!t.do_preload()) continue;
            if (!t.needs_f32_convert()) continue;
            stmt_ = stmt_.append(t.build_convert_stmt());
        }

        // Generate compute statements for virtual tensors.
        for (auto &t : post_op_tensors_) {
            if (!t.needs_compute()) continue;
            stmt_ = stmt_.append(t.build_compute_stmt(post_op_tensors_));
        }

        // Initialize buffers for output post-op tensors.
        for (auto &t : post_op_tensors_) {
            if (!t.needs_store()) continue;
            t.init_output_buffer(base_tile);
        }

        // Generate zero-out statements for output post-op tensors.
        for (auto &t : post_op_tensors_) {
            if (!t.needs_store()) continue;
            stmt_ = stmt_.append(t.build_zero_out_stmt());
        }

        // Iterate by tiles and apply post-ops.
        c_mem_view_.for_each_tile(
                base_tile, [&](const std::vector<dim_t> &start) {
                    tensor_t tile(base_tile.dims(), start);
                    auto c_tile_layout = c_reg_layout.map(tile);
                    build_tile(tile, c_tile_layout, c_reg_buf);
                });

        // TODO: Generalize the condition. Iterate through output tensor masks
        // and ensure C is distributed accordingly in thread group.
        bool use_slm_reduction = (gemm_schedule_.tg_grid().dim(1) > 1);

        // Generate reduce and store statements for output post-op tensors.
        stmt_t thr_reduce_stmt;
        stmt_t slm_store_stmt;
        stmt_t slm_load_stmt;
        stmt_t mem_store_stmt;
        for (auto &t : post_op_tensors_) {
            if (!t.needs_store()) continue;

            thr_reduce_stmt = thr_reduce_stmt.append(t.build_reduce_stmt());
            if (use_slm_reduction) {
                auto store_stmt
                        = t.build_slm_store_stmt(gemm_schedule_.tg_grid());
                auto load_stmt = t.build_slm_load_stmt();
                slm_store_stmt = slm_store_stmt.append(store_stmt);
                slm_load_stmt = slm_load_stmt.append(load_stmt);
            }
            mem_store_stmt = mem_store_stmt.append(t.build_store_stmt());
        }

        stmt_ = stmt_.append(thr_reduce_stmt);
        if (!slm_store_stmt.is_empty()) {
            stmt_ = stmt_.append(funcs::barrier());
            stmt_ = stmt_.append(slm_store_stmt);
            stmt_ = stmt_.append(funcs::barrier());
            stmt_ = stmt_.append(slm_load_stmt);
        }

        stmt_ = stmt_.append(mem_store_stmt);

        // Generate alloc statements for post-op tensors.
        std::vector<stmt_t> allocs;
        for (auto &t : post_op_tensors_) {
            auto t_allocs = t.allocs();
            allocs.insert(allocs.end(), t_allocs.begin(), t_allocs.end());
        }
        stmt_ = jit::inject_alloc_stmts(stmt_, allocs, /*put_innermost=*/true);
    }

    // Builds statements for a tile iterating through all post-ops.
    void build_tile(const tensor_t &tile, const layout_t &c_tile_layout,
            const expr_t &c_reg_buf) {
        auto c_mem_tile_view = c_mem_view_.create_sub_view(tile);
        auto tmp_reg_buf = make_c_tmp_buffer();

        type_t post_op_type
                = c_tile_layout.type().is_f64() ? type_t::f64() : type_t::f32();
        bool create_zero_pad_builder = restore_zero_padding_;
        for (auto &t : post_op_tensors_) {
            if (t.needs_masked_update()) {
                create_zero_pad_builder = true;
                break;
            }
        }
        if (create_zero_pad_builder) {
            zero_pad_builder_ = zero_pad_builder_t(
                    ir_ctx_, post_op_ctx_.cp_view(), c_mem_tile_view);
        }

        // S_y -> GMEM.
        auto send_op = gemm_schedule_.with_kernel_grid_k_slicing()
                ? send_op_t::atomic_fadd
                : send_op_t::store;
        auto offset = c_mem_tile_view.tlayout().offset_in_bytes();
        const int cache_line_size = 64;
        const bool allow_2d = !offset.is<int_imm_t>()
                || (offset.as<int_imm_t>().value % cache_line_size == 0);
        auto send_params = get_send_params(ir_ctx_.exec_cfg(), send_op,
                send_address_t::a64, fma_kind_t::unknown, abc_kind_t::c,
                c_mem_tile_view, gemm_schedule_, allow_2d);
        auto r2g = make_access_builder(
                ir_ctx_, c_mem_tile_view, c_mem_buf_, tmp_reg_buf, send_params);

        // Initialize C stages.
        std::vector<c_stage_t> c_stages;

        auto c_fx_layout = r2g.reg_layout().retype(post_op_type).make_dense();
        bool with_post_ops = !post_op_builders_.empty();
        int npost_ops = int(post_op_builders_.size());

        int c_f32_stage_idx = -1;
        int c_zero_pad_stage_idx = -1;

        c_stages.emplace_back(c_tile_layout, 0, c_reg_buf); // M_x
        if (with_post_ops) {
            c_f32_stage_idx = int(c_stages.size());
            c_stages.emplace_back(c_fx_layout, 0, make_c_tmp_buffer()); // R_f32
        }
        if (restore_zero_padding_) {
            c_zero_pad_stage_idx = int(c_stages.size());
            c_stages.emplace_back(c_fx_layout, 0, make_c_tmp_buffer()); // Z_f32
        }
        c_stages.emplace_back(r2g.reg_layout(), r2g.reg_buf_size(), tmp_reg_buf,
                r2g.stmt()); // S_y

        int nstages = int(c_stages.size());

        // Update buffer sizes.
        std::vector<int> buf_sizes(nstages);
        for (int i = 1; i < nstages; i++) {
            auto &s = c_stages[i];
            buf_sizes[i] = s.max_off_bytes();
        }

        // Generate reorders between C stages if needed.
        for (int i = 0; i < nstages; i++) {
            auto *next_stage = (i + 1 < nstages ? &c_stages[i + 1] : nullptr);
            c_stages[i].set_next(ir_ctx_, next_stage,
                    /*force_reorder=*/i == 0 && force_c_reorder_);
        }

        // Update buffer sizes.
        for (int i = nstages - 2; i >= 0; i--) {
            auto &s_cur = c_stages[i];
            auto &s_next = c_stages[i + 1];
            if (s_cur.buf_base().is_same(s_next.buf_base())) {
                buf_sizes[i] = std::max(buf_sizes[i], buf_sizes[i + 1]);
            }
        }

        // Restore zero padding if needed.
        if (c_zero_pad_stage_idx != -1) {
            auto &s = c_stages[c_zero_pad_stage_idx];
            s.prepend_stmt(zero_pad_builder_.build_stmt(s.layout, s.buf));
        }

        // Create sub-tensors for post-ops.
        std::vector<post_op_tensor_t> sub_po_tensors;
        for (auto &t : post_op_tensors_)
            sub_po_tensors.push_back(t.create_sub_tensor(tile));

        // Set C tensor layout and buffer to use for post-ops.
        if (c_f32_stage_idx != -1) {
            auto &s = c_stages[c_f32_stage_idx];
            sub_po_tensors[c_po_idx_].set_reg_layout(s.layout);
            sub_po_tensors[c_po_idx_].set_reg_buf(s.buf);
        }

        stmt_t tile_stmt;

        // Add C stage statements and post-op statements.
        for (int i = 0; i < nstages; i++) {
            if (with_post_ops && i == c_f32_stage_idx) {
                // Emit post-ops in blocks to reduce GRF consumption.
                for (int j = 0; j < npost_ops; j += post_op_blk_) {
                    int k_beg = j;
                    int k_end = std::min(npost_ops, j + post_op_blk_);
                    auto blk_stmt = build_post_op_block_stmt(
                            tile, sub_po_tensors, k_beg, k_end);
                    tile_stmt = tile_stmt.append(blk_stmt);
                }
            }
            tile_stmt = tile_stmt.append(c_stages[i].stmt);
        }

        // Generate alloc statements for C stage buffers.
        object_set_t<expr_t> seen;
        for (int i = 0; i < nstages; i++) {
            auto &s = c_stages[i];
            auto &buf = s.buf_base();
            auto ret = seen.insert(buf);
            if (i == 0 || !ret.second) continue;
            int size = utils::rnd_up(buf_sizes[i], ir_ctx_.grf_size());
            tile_stmt = alloc_t::make(buf, size, alloc_kind_t::grf, tile_stmt);
        }

        stmt_ = stmt_.append(tile_stmt);
        int c_off_bytes = to_cpp<int>(c_tile_layout.offset_in_bytes());
        c_reg_buf_size_ = std::max(c_reg_buf_size_, c_off_bytes + buf_sizes[0]);
    }

    stmt_t build_post_op_block_stmt(const tensor_t &tile,
            std::vector<post_op_tensor_t> &sub_po_tensors, int po_beg,
            int po_end) const {
        // Collect post-op inputs/outputs.
        object_map_t<expr_t, post_op_tensor_t *> args;
        for (int i = po_beg; i < po_end; i++) {
            auto &po_builder = post_op_builders_[i];
            for (auto &t : sub_po_tensors) {
                if (po_builder.post_op().uses(t.op_var())) {
                    args.insert({t.op_var(), &t});
                }
            }
        }

        // Generate load and convert statements for the post-op.
        stmt_t load_stmt;
        stmt_t convert_stmt;
        for (auto &kv : args) {
            auto &t = *kv.second;
            if (!t.needs_load()) continue;
            if (t.do_preload()) continue;
            load_stmt = load_stmt.append(t.build_load_stmt(c_mem_view_));
            if (t.needs_f32_convert()) {
                convert_stmt = convert_stmt.append(t.build_convert_stmt());
            }
        }

        stmt_t stmt;
        stmt = stmt.append(load_stmt);
        stmt = stmt.append(convert_stmt);

        for (int i = po_beg; i < po_end; i++) {
            auto &po_builder = post_op_builders_[i];
            auto po_stmt
                    = po_builder.build_tile_stmt(tile, args, zero_pad_builder_);
            stmt = stmt.append(po_stmt);
        }

        // Generate alloc statements for post-op tensors.
        std::vector<stmt_t> allocs;
        for (auto &kv : args) {
            auto &t = *kv.second;
            if (!t.needs_load()) continue;
            if (t.do_preload()) continue;
            auto t_allocs = t.allocs();
            allocs.insert(allocs.end(), t_allocs.begin(), t_allocs.end());
        }
        stmt = jit::inject_alloc_stmts(stmt, allocs);

        return stmt;
    }

    ir_context_t &ir_ctx_;
    const gemm_schedule_t &gemm_schedule_;
    const post_op_context_t &post_op_ctx_;

    // C view in global memory.
    view_t c_mem_view_;
    expr_t c_mem_buf_;

    // C layout after the main loop.
    layout_t c_reg_layout_;
    expr_t c_reg_buf_;

    const bool force_c_reorder_;
    const bool restore_zero_padding_;

    zero_pad_builder_t zero_pad_builder_;

    // Tile size in bytes. The tile data type is:
    // - the destination data type without post-ops
    // - f32 with post-ops
    int tile_size_;
    int preload_max_size_;
    int post_op_blk_;

    std::vector<post_op_builder_t> post_op_builders_;
    std::vector<post_op_tensor_t> post_op_tensors_;
    int c_po_idx_ = -1;

    stmt_t stmt_;
    int c_reg_buf_size_ = 0;
};

stmt_t create_epilogue_stmt(const exec_config_t &exec_cfg, ir_context_t &ir_ctx,
        const gemm_schedule_t &gemm_schedule, bool force_c_reorder,
        const post_op_context_t &post_op_ctx, const tensor_t &thr_tile,
        const layout_t &c_reg_layout, const expr_t &c_mem_buf,
        const expr_t &c_reg_buf, int &c_reg_buf_size) {
    // Max size of post-op tensor buffers to preload and reuse for all tiles.
    int preload_max_size = 512;
    // Block size to apply post-ops within tile. A post-op may have associated
    // loads/conversions, larger block size helps to have more latency hiding
    // across multiple post-ops.
    int post_op_blk = 8;

    const auto c_mem_view = post_op_ctx.cp_view().create_sub_view(thr_tile);
    epilogue_builder_t builder(ir_ctx, exec_cfg, gemm_schedule, force_c_reorder,
            post_op_ctx, thr_tile, c_mem_view, c_reg_layout, c_mem_buf,
            c_reg_buf, preload_max_size, post_op_blk);
    c_reg_buf_size = builder.c_reg_buf_size();
    return builder.stmt();
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
