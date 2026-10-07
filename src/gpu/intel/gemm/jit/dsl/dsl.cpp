/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include <stack>

#include "dsl/ir/fma.hpp"
#include "dsl/ir/ir.hpp"
#include "dsl/ir/pass/dpas.hpp"
#include "dsl/ir/send.hpp"
#include "dsl/utils/block_2d_utils.hpp"
#include "dsl/utils/logging.hpp"
#include "gemmstone/dsl/dsl.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {

using alloc_t = ir::alloc_t;
using alloc_kind_t = ir::alloc_kind_t;
using assign_t = ir::assign_t;
using binary_op_t = ir::binary_op_t;
using builtin_t = ir::builtin_t;
using ir_context_t = ir::ir_context_t;
using dpas_t = ir::dpas_t;
using for_t = ir::for_t;
using iif_t = ir::iif_t;
using if_t = ir::if_t;
using let_t = ir::let_t;
using load_t = ir::load_t;
using mad_t = ir::mad_t;
using ref_t = ir::ref_t;
using send_address_t = ir::send_address_t;
using send_op_t = ir::send_op_t;
using send_t = ir::send_t;
using shuffle_t = ir::shuffle_t;
using stmt_seq_t = ir::stmt_seq_t;
using store_t = ir::store_t;
using var_t = ir::var_t;
using while_t = ir::while_t;

struct ctx_t {
    bool new_ir_api() const { return new_ir_api_; }

    void declare_kernel(const kernel::iface_t &interface, ir_context_t &ctx,
            bool new_ir_api = false) {
        slm_byte_offset_ = 0;
        new_ir_api_ = new_ir_api;
        dsl_assert(stmts_stack_.empty())
                << "Invalid generation of a kernel within a kernel";
        interface_ = interface;
        ctx_ = &ctx;

        begin_scope();

        if (new_ir_api_) {
            for (int i = 0; i < 3; i++) {
                local_sizes_[i] = var_t::make(
                        local_size_type(), ir::local_size_name(i));
                local_ids_[i]
                        = var_t::make(local_id_type(), ir::local_id_name(i));
                group_ids_[i]
                        = var_t::make(group_id_type(), ir::group_id_name(i));
                subgroup_ids_[i] = def("subgroup_id" + std::to_string(i),
                        extract((local_id(i) / (i == 0 ? simd() : 1)), 0));
            }
            subgroup_local_id_
                    = def("subgroup_local_id", local_ids_[0] & (simd() - 1));
        } else {
            for (size_t i = 0; i < interface.nargs(); i++) {
                const auto &var = interface[i];
                if (var.type().is_ptr()) {
                    if (var.type().is_slm()) {
                        append(alloc_t::make(
                                var, 0, alloc_kind_t::slm, stmt_t {}));
                    } else {
                        append(alloc_t::make(
                                var, 0, alloc_kind_t::global, stmt_t {}));
                    }
                } else {
                    if (!new_ir_api_) append(let_t::make(var, {}, {}));
                }
            }

            for (int i = 0; i < 3; i++) {
                group_ids_[i] = let(ir::tg_idx_name(i), group_id_type(), {});
                local_ids_[i] = let(ir::local_id_name(i), local_id_type(), {});
                local_sizes_[i]
                        = let(ir::local_size_name(i), local_size_type(), {});
            }
        }
    }

    kernel_t end_kernel() {
        dsl_assert(stmts_stack_.size() == 1)
                << "Invalid end of kernel, imbalanced scopes detected";
        auto body = pop_scope();
        if (slm_byte_offset() > 0) {
            auto slm_buf = var_t::make(u8[slm_byte_offset()].with_slm(), "slm");
            auto slm_alloc = builtin_t::make("alloc")(slm_buf);
            body = slm_alloc.append(body);
        }
        kernel_t ret {std::move(interface_), body, ctx_->options()};
        ctx_ = nullptr;
        interface_ = {"undefined_dsl_kernel"};
        return ret;
    }

    int simd() const { return ctx_->options().simd(); }

    const std::array<expr_t, 3> &group_ids() const { return group_ids_; }
    const expr_t &group_id(int idx) const { return group_ids_[idx]; }
    const std::array<expr_t, 3> &local_ids() const { return local_ids_; }
    const expr_t &local_id(int idx) const { return local_ids_[idx]; }
    const std::array<expr_t, 3> &local_sizes() const { return local_sizes_; }
    const expr_t &local_size(int idx) const { return local_sizes_[idx]; }
    const expr_t &subgroup_id(int idx) const { return subgroup_ids_[idx]; }
    const expr_t &subgroup_local_id() const { return subgroup_local_id_; }

    expr_t arg(const std::string &name, bool allow_empty = false) {
        auto a = interface_.find_arg(name, allow_empty);
        expr_t value;
        if (a && ctx_->cset().is_single_value(a, value)) { return value; }
        return a;
    }

    // TODO: Remove IR restriction which requires force_alloc
    lval_t def(const std::string &name, type_t _type, const expr_t &value = {},
            bool force_alloc = false) {
        auto type = _type.with_attr(_type.attr() | type::attr_t::mut);
        auto alloc_var = var(type, name);
        if (new_ir_api_) {
            if (!value.is_empty()) append(assign_t::make(alloc_var, value));
        } else {
            if (force_alloc || type.is_ptr()) {
                append(alloc_t::make(alloc_var, {}));

                if (!value.is_empty()) {
                    dsl_assert(ir::to_cpp<int>(value) == 0);
                    append(ir::funcs::zero_out(alloc_var, type.size()));
                }
            } else {
                append(let_t::make(alloc_var, value, {}));
            }
        }
        return lval_t(alloc_var.as<var_t>());
    }

    lval_t def(const std::string &name, const expr_t &value) {
        return def(name, value.type(), value);
    }

    tensor_t def(const std::string &name, const layout_t &layout,
            type::attr_t attr, const expr_t &value = {}) {
        auto &back = layout.blocks().back();
        auto size = into<int>(back.size * int64_t(back.stride));

        // Padding allocations due to strides with overlapping dimensions have
        // unclear semantics, disallow their use.
        dsl_assert([&]() {
            int64_t max_off = 0;
            for (auto &b : layout.blocks()) {
                max_off += (b.size - 1) * int64_t(b.stride);
            }
            return max_off < size;
        }());

        dsl_assert(layout.offset().is(0));
        auto t = layout.type().with_attr(attr);
        if (any(attr & type::attr_t::slm)) {
            dsl_assert(value.is_empty());
            auto buf = def(name, t[size]);
            auto size_bytes = size * layout.type().size();
            auto off = div_up(slm_byte_offset(), layout.type().size());
            auto off_bytes = off * layout.type().size();
            reserve_slm((off_bytes - slm_byte_offset()) + size_bytes);
            return {buf.ptr(), layout.with_offset(off)};
        }

        // Tensors need to be grf-aligned for loading/storing
        // TODO: IR should be modified to enable loading small tensors (such as
        // scalar values) without GRF alignment.
        size = std::max(size, grf_size() / layout.type().size());
        auto buf = def(name, t[size], value, /*force_alloc=*/!new_ir_api_);
        return {buf.ptr(), layout};
    }

    expr_t let(
            const std::string &name, const type_t &type, const expr_t &value) {
        auto alloc_var = var(type, name);
        append(let_t::make(alloc_var, value, {}));
        return alloc_var;
    }
    expr_t let(const std::string &name, const expr_t &value) {
        return let(name, value.type(), value);
    }

    int slm_byte_offset() const { return slm_byte_offset_; }

    void reserve_slm(int bytes) { slm_byte_offset_ += bytes; }

    void assume(const expr_t &e) { ctx_->add_constraint(e); }

    void begin_scope() { stmts_stack_.emplace(); }

    void end_scope() {
        auto stmt = pop_scope();
        dsl_assert(!stmts_stack_.empty());
        append(stmt);
    }

    stmt_t pop_scope() {
        auto stmt = to_stmt();
        stmts_stack_.pop();
        return stmt;
    }

    void append(stmt_t stmt) {
        dsl_assert(!stmts_stack_.empty())
                << "Cannot instantiate " << stmt << " outside of a kernel";
        stmts().emplace_back(std::move(stmt));
    }

    const ir_context_t *ir_ctx() const { return ctx_; }

private:
    type_t local_id_type() const { return u16; }
    type_t group_id_type() const { return u32; }
    type_t local_size_type() const { return u16; }

    expr_t var(type_t type, const std::string &name) {
        return var_t::make(type, ctx_->create_tmp_name(name));
    }

    stmt_t to_stmt() {
        stmt_t stmt;
        dsl_assert(!stmts_stack_.empty());
        size_t size = stmts().size();
        size_t end = size;
        size_t begin = size - 1;
        while (begin < end) {
            auto &s = stmts()[begin];
            if (s.is<alloc_t>() || s.is<let_t>()) {
                stmt_t body = [&]() {
                    if (begin + 1 >= end) return stmt;
                    auto seq = std::vector<stmt_t>(
                            stmts().begin() + begin + 1, stmts().begin() + end);
                    seq.push_back(stmt);
                    return stmt_seq_t::make(seq);
                }();
                end = begin;

                if (s.is<alloc_t>() && s.as<alloc_t>().body.is_empty()) {
                    auto &a = s.as<alloc_t>();
                    if (a.buf.type().is_ptr())
                        stmt = alloc_t::make(
                                a.buf, a.size, a.kind, a.attrs, body);
                    else
                        stmt = alloc_t::make(a.buf, body);
                } else if (s.is<let_t>() && s.as<let_t>().body.is_empty()) {
                    auto &l = s.as<let_t>();
                    stmt = let_t::make(l.var, l.value, body);
                }
            }
            begin--;
        }

        if (end > 0) {
            std::vector<stmt_t> seq(stmts().begin(), stmts().begin() + end);
            seq.push_back(stmt);
            stmt = stmt_seq_t::make(seq);
        }
        return stmt;
    }

    std::vector<stmt_t> &stmts() { return stmts_stack_.top(); }
    std::stack<std::vector<stmt_t>> stmts_stack_;
    kernel::iface_t interface_ = {"undefined_dsl_kernel"};
    ir_context_t *ctx_ = nullptr;
    std::array<expr_t, 3> group_ids_;
    std::array<expr_t, 3> local_ids_;
    std::array<expr_t, 3> local_sizes_;
    std::array<expr_t, 3> subgroup_ids_;
    expr_t subgroup_local_id_;
    bool new_ir_api_ = false;
    int slm_byte_offset_ = 0;
};

ctx_t &default_ctx() {
    static thread_local ctx_t ctx;
    return ctx;
}

int grf_size() {
    return default_ctx().ir_ctx()->hw().grf_size();
}
int min_align_2d() {
    return block_2d_base_alignment(default_ctx().ir_ctx()->hw());
}
int min_pitch_2d() {
    return block_2d_pitch_alignment(default_ctx().ir_ctx()->hw());
}

void declare_kernel(
        const kernel::iface_t &interface, ir_context_t &ctx, bool new_ir_api) {
    default_ctx().declare_kernel(interface, ctx, new_ir_api);
}

kernel_t end_kernel() {
    return default_ctx().end_kernel();
}

void begin_scope() {
    default_ctx().begin_scope();
}

void end_scope() {
    default_ctx().end_scope();
}

stmt_t pop_scope() {
    return default_ctx().pop_scope();
}

void append(stmt_t stmt) {
    default_ctx().append(std::move(stmt));
}

void assume(const expr_t &e) {
    default_ctx().assume(e);
}

const std::array<expr_t, 3> &group_ids() {
    return default_ctx().group_ids();
}

const expr_t &group_id(int idx) {
    return default_ctx().group_id(idx);
}

const std::array<expr_t, 3> &local_ids() {
    return default_ctx().local_ids();
}

const expr_t &local_id(int idx) {
    return default_ctx().local_id(idx);
}

const std::array<expr_t, 3> &local_sizes() {
    return default_ctx().local_sizes();
}

const expr_t &local_size(int idx) {
    return default_ctx().local_size(idx);
}

expr_t subgroup_id(int idx) {
    return default_ctx().subgroup_id(idx);
}

expr_t subgroup_local_id() {
    return default_ctx().subgroup_local_id();
}

expr_t arg(const std::string &name, bool allow_empty) {
    return default_ctx().arg(name, allow_empty);
}

lval_t def(const std::string &name, const type_t &type, const expr_t &value,
        bool force_alloc) {
    return default_ctx().def(name, type, value, force_alloc);
}

lval_t def(const std::string &name, const expr_t &value) {
    return def(name, value.type(), value);
}

tensor_t def(const std::string &name, const layout_t &layout,
        const expr_t &value, type::attr_t attr) {
    return default_ctx().def(name, layout, attr, value);
}

tensor_t def(
        const std::string &name, const layout_t &layout, type::attr_t attr) {
    return def(name, layout, {}, attr);
}

expr_t iif(
        const expr_t &cond, const expr_t &true_expr, const expr_t &false_expr) {
    return iif_t::make(cond, true_expr, false_expr);
}

expr_t extract(const expr_t &expr, int lane) {
    return shuffle_t::make(expr, {lane});
}

lval_t::lval_t(const type_t &type, const std::string &name)
    : var(var_t::make(type, name)) {}

lval_t &lval_t::operator=(const expr_t &obj) {
    assign(this->var, obj);
    return *this;
}

lval_t lval_t::sub(int off, int elems) const {
    return lval_t(ref_t::make(var, off, elems));
}

expr_t let(const std::string &name, const type_t &type, const expr_t &value) {
    return default_ctx().let(name, type, value);
}

expr_t let(const std::string &name, const expr_t &value) {
    return default_ctx().let(name, value);
}

void assign(const expr_t &var, const expr_t &value) {
    if (default_ctx().new_ir_api()) {
        append(assign_t::make(var, value));
    } else {
        append(store_t::make(var, 0, value));
    }
}

enum class send_kind_t { load, prefetch, store };

void scatter_send(const tensor_t &t, const global_tensor_t &g,
        send_kind_t &op_kind, const icoord_t &base, const send_hint_t &hint) {
    dsl_warning() << "Scatter messages are not yet implemented";
}

layout_t prefetch_layout(const global_tensor_t &g, const idx_t &w_idx) {
    std::vector<layout::block_t> blocks;
    blocks.reserve(g.tile.size());
    blocks.emplace_back(w_idx, g.tile[w_idx]);
    for (auto &idx : g.tile) {
        if (idx != w_idx) blocks.emplace_back(idx, g.tile[idx]);
    }
    return layout_t(g.type, blocks);
}

void block_send(const tensor_t &t, const global_tensor_t &g,
        send_kind_t &op_kind, const icoord_t &base, const send_hint_t &hint) {
    bool is_prefetch = t.buf.is_empty();
    auto &operation_tile = is_prefetch ? g.tile : t.layout.tile();

    idx_t w_idx;
    tile_t tile;
    for (auto &var : operation_tile) {
        if (is_const(g.strides[var]) && ir::to_cpp<int64_t>(g.strides[var]) == 1
                && t.layout.elems() != 1) {
            tile[var] = t.layout[0].size;
            dsl_assert(t.layout[0].idx == var);
            w_idx = var;
        } else {
            tile[var] = 1;
        }
    }
    auto type = g.type;

    auto operation_layout = is_prefetch ? prefetch_layout(g, w_idx) : t.layout;
    for (auto &coord : operation_layout.iter(tile)) {
        auto buffer = is_prefetch ? expr_t() : t.subbuf(coord);
        auto width = !w_idx.is_undef()
                ? std::min(tile[w_idx], operation_tile[w_idx] - coord[w_idx])
                : 1;

        int width_bytes = into<int>(width * type.size());
        auto coord_local = coord;
        while (width_bytes > 0) {
            auto send_type = [&]() {
                if (width_bytes <= 16) { return type_t::byte(width_bytes); }
                auto load_width = rounddown_pow2(std::min(width_bytes, 512));
                return type_t::oword(load_width / 16);
            }();
            auto send_kind = [&]() {
                switch (op_kind) {
                    case send_kind_t::prefetch: return send_op_t::prefetch;
                    case send_kind_t::load: return send_op_t::load;
                    case send_kind_t::store: return send_op_t::store;
                    default: stub(); return send_op_t::undef;
                }
            }();

            auto send_func = send_t::make({}, send_kind, send_address_t::a64,
                    send_type, 1, true, true, hint.cache);
            append(send_func.as<send_t>()(
                    g.buf, g.offset(base + coord_local), buffer, {}));
            width_bytes -= send_type.size();
            coord_local[w_idx] += send_type.size() / type.size();
        }
    }
}

struct conf_2d_t {
    type_t type;
    idx_t w_idx;
    int pack_size;
    bool is_vnni;
    bool is_transpose_vnni;
    bool is_store;

    int unit_size() const {
        return is_transpose_vnni || is_vnni ? std::max(type.size(), 4)
                                            : type.size();
    }

    // Tile used for 2d Messages
    tile_t get_tile(std::array<idx_t, 2> dims) const {
        auto width = pack_size ? pack_size : grf_size() / unit_size();
        auto height = is_store ? 8 : 32;

        if (is_transpose_vnni) return {{dims[1], width}, {dims[0], height}};
        return {{dims[0], width}, {dims[1], height}};
    }
};

void block_2d_send(const conf_2d_t &conf, const tensor_t &t,
        const global_tensor_t &g, send_kind_t op_kind, const icoord_t &base,
        const send_hint_t &hint) {

    bool is_prefetch = t.buf.is_empty();
    auto &operation_tile = is_prefetch ? g.tile : t.layout.tile();

    idx_t w_idx = conf.w_idx;
    idx_t h_idx;
    for (auto &var : operation_tile) {
        if (var != w_idx) {
            dsl_assert(h_idx.is_undef())
                    << "n-dimensional support unimplemented";
            h_idx = var;
        }
    }

    auto tensor_width = g.sizes[w_idx];
    auto tensor_height = g.sizes[h_idx];
    auto tensor_pitch = g.strides[h_idx];
    auto type = g.type;
    auto tile = conf.get_tile({w_idx, h_idx});

    auto operation_layout = is_prefetch ? prefetch_layout(g, w_idx) : t.layout;
    for (auto &coord : operation_layout.iter(tile)) {
        auto buffer = is_prefetch ? expr_t() : t.subbuf(coord);
        int width = into<int>(
                std::min(tile[w_idx], operation_tile[w_idx] - coord[w_idx]));
        int height = into<int>(
                std::min(tile[h_idx], operation_tile[h_idx] - coord[h_idx]));
        int count = std::max(1, into<int>(tile[w_idx] / width));
        auto width_idx
                = g.coord[w_idx] + static_cast<uint32_t>((base + coord)[w_idx]);
        auto height_idx
                = g.coord[h_idx] + static_cast<uint32_t>((base + coord)[h_idx]);
        auto send_kind = [&]() {
            switch (op_kind) {
                case send_kind_t::prefetch: return send_op_t::prefetch_2d;
                case send_kind_t::load: return send_op_t::load_2d;
                case send_kind_t::store: return send_op_t::store_2d;
                default: stub(); return send_op_t::undef;
            }
        }();

        auto send_func = send_t::make_2d({}, send_kind, type, tensor_width,
                tensor_height, tensor_pitch, width, height, count, conf.is_vnni,
                conf.is_transpose_vnni,
                /*zero_out=*/true, hint.cache);

        append(send_func.as<send_t>()(g.buf, g.base_offset * type.size(),
                buffer, {}, width_idx, height_idx));
    }
}

void send(const tensor_t &t, const global_tensor_t &g, send_kind_t op_kind,
        const icoord_t &base, const send_hint_t &hint) {
    bool is_prefetch = t.buf.is_empty();
    auto &operation_tile = is_prefetch ? g.tile : t.layout.tile();
    idx_t w_idx;
    for (auto &var : operation_tile) {
        if (is_const(g.strides[var])
                && ir::to_cpp<int64_t>(g.strides[var]) == 1) {
            dsl_assert(w_idx.is_undef())
                    << "Could not determine inner dimension";
            w_idx = var;
        }
    }

    auto type = g.type;

    dsl_assert(is_prefetch || type == t.layout.type());
    if (operation_tile.size() >= 2 && !w_idx.is_undef()) {
        auto conf = [&]() -> conf_2d_t {
            if (is_prefetch) { return {g.type, w_idx, 0, false, false, false}; }
            auto &l = t.layout;
            int pack_idx = l[0].size * l.type().size() == 4;
            int pack_size = into<int>(l[pack_idx].size);
            bool is_transpose_vnni = l[pack_idx].idx != w_idx;
            bool is_vnni = pack_idx == 1 && !is_transpose_vnni;
            bool is_store = op_kind == send_kind_t::store;
            return {g.type, w_idx, pack_size, is_vnni, is_transpose_vnni,
                    is_store};
        }();

        if (conf.pack_size <= grf_size() / conf.unit_size()) {
            block_2d_send(conf, t, g, op_kind, base, hint);
            return;
        }
    }

    if (is_prefetch || t.layout.elems() == 1 || t.layout[0].idx == w_idx) {
        block_send(t, g, op_kind, base, hint);
    } else {
        scatter_send(t, g, op_kind, base, hint);
    }
}

void prefetch(const global_tensor_t &g, const icoord_t &base,
        const send_hint_t &hint) {
    send({}, g, send_kind_t::prefetch, base, hint);
}

void load(const tensor_t &t, const global_tensor_t &g, const icoord_t &base,
        const send_hint_t &hint) {
    send(t, g, send_kind_t::load, base, hint);
}

void store(const global_tensor_t &g, const tensor_t &t, const icoord_t &base,
        const send_hint_t &hint) {
    send(t, g, send_kind_t::store, base, hint);
}

void mma(const tensor_t &C, const tensor_t &A, const tensor_t &B,
        const tile_t &tile, const icoord_t &base, bool is_systolic) {
    if (is_systolic) {
        int64_t simd = 16;
        int64_t sdepth = 8;
        int64_t max_rcount = 8;

        auto simd_idx = C.layout[0].idx;
        auto sdepth_idx = A.layout[0].idx == C.layout[0].idx ? A.layout[1].idx
                                                             : A.layout[0].idx;
        auto rcount_idx = C.layout[1].idx;
        auto sdepth_pack = A.layout.type().bytes_to_elems(4);

        tile_t inst_tile {{simd_idx, simd}, {sdepth_idx, sdepth * sdepth_pack},
                {rcount_idx, max_rcount}};

        dsl_assert(tile[simd_idx] % simd == 0);
        dsl_assert(tile[sdepth_idx] % (sdepth_pack * sdepth) == 0);
        dsl_assert(C.layout[0].size == simd);

        std::vector<layout::block_t> dpas_blocks;
        dpas_blocks.emplace_back(sdepth_idx, tile[sdepth_idx]);
        for (auto &b : C.layout.blocks()) {
            dpas_blocks.emplace_back(b.idx, b.size);
        }
        auto dpas_layout = C.layout.with(dpas_blocks);

        std::vector<stmt_t> dpas_stmts;
        for (auto &coord : dpas_layout.iter(inst_tile)) {
            int simd = (int)inst_tile[simd_idx];
            auto sdepth = inst_tile[sdepth_idx] / sdepth_pack;
            auto rcount = std::min(inst_tile[rcount_idx],
                    tile[rcount_idx] - coord[rcount_idx]);

            auto dpas = dpas_t::make(false, simd, into<uint8_t>(sdepth),
                    into<uint8_t>(rcount), C.layout.type(), B.layout.type(),
                    A.layout.type());
            // FIXME: This code can access out-of-bounds coordinates, adding
            // modulus to keep the old behavior with v2 layout.
            auto get_coord = [](const tensor_t &t, icoord_t coord) {
                for (auto &d : coord) {
                    coord[d] = coord[d] % t.layout.tile().get(d, coord[d] + 1);
                }
                return coord;
            };
            auto dst = C.subbuf(base + coord);
            auto src1 = A.subbuf(get_coord(A, base + coord));
            auto src2 = B.subbuf(get_coord(B, base + coord));
            dpas_stmts.emplace_back(dpas.as<dpas_t>()(dst, dst, src1, src2));
        }
        append(inject_dpas_atomic(stmt_seq_t::make(dpas_stmts),
                /*filter_by_label=*/false));
    } else {
        auto max_simd = 32;

        const auto &simd_idx = C.layout[0].idx;
        const auto &rcount_idx = C.layout[1].idx;
        const auto &m_idx = simd_idx;
        const auto &n_idx = rcount_idx;
        const auto &k_idx = one_of(A.layout[1].idx, {simd_idx, rcount_idx})
                ? A.layout[0].idx
                : A.layout[1].idx;

        tile_t inst_tile {{{simd_idx, max_simd}, {rcount_idx, 1}, {k_idx, 1}}};

        int M = (int)inst_tile.get(m_idx, 1);
        int N = (int)inst_tile.get(n_idx, 1);
        int K = (int)inst_tile.get(k_idx, 1);
        bool is_a_bcast = (M * K == 1);
        bool is_b_bcast = (K * N == 1);
        int a_stride = is_a_bcast ? 0 : int(A.layout.stride(m_idx));
        int b_stride = is_b_bcast ? 0 : int(B.layout.stride(n_idx));

        std::vector<layout::block_t> mad_blocks;
        mad_blocks.emplace_back(k_idx, tile[k_idx]);
        for (auto &b : C.layout.blocks()) {
            mad_blocks.emplace_back(b.idx, b.size);
        }
        auto mad_layout = C.layout.with(mad_blocks);

        dsl_assert(tile[simd_idx] * C.layout.type().size() % grf_size() == 0);
        for (auto &coord : mad_layout.iter(inst_tile)) {
            int simd = (int)std::min(
                    inst_tile[simd_idx], tile[simd_idx] - coord[simd_idx]);

            auto mad = mad_t::make(default_ctx().ir_ctx()->hw(),
                    C.layout.type(), simd, A.layout.type(), a_stride,
                    B.layout.type(), b_stride);

            auto dst = C.subbuf(base + coord);
            auto src1 = A.subbuf(base + coord);
            auto src2 = B.subbuf(base + coord);

            append(mad.as<mad_t>()(dst, dst, src1, src2));
        }
    }
}

void _if_impl(const expr_t &cond, const stmt_t &if_body) {
    append(if_t::make(cond, if_body));
}
void _if_impl(
        const expr_t &cond, const stmt_t &if_body, const stmt_t &else_body) {
    append(if_t::make(cond, if_body, else_body));
}
void _for_impl(const expr_t &var, const expr_t &bound, const expr_t &step,
        const stmt_t &body) {
    append(for_t::make(var, 0, bound, body, step));
}
void _while_impl(const expr_t &cond, const stmt_t &body) {
    append(while_t::make(cond, body));
}

void binary(op_kind_t op, const tensor_t &dst, const tensor_t &src0,
        const tensor_t &src1) {
    tile_t tile = dst.layout.tile();
    tile_t matching_subtile = [&] {
        tile_t ret;
        for (auto &var : tile) {
            ret[var] = 1;
        }

        auto &bd = dst.layout.blocks();
        auto &b0 = src0.layout.blocks();
        auto &b1 = src1.layout.blocks();
        for (size_t i = 0; i < bd.size(); i++) {
            if (b0.size() <= i) break;
            if (b1.size() <= i) break;
            if (bd[i] == b0[i] && bd[i] == b1[i])
                ret[bd[i].idx] *= bd[i].size;
            else
                break;
        }

        return ret;
    }();

    auto subtile_elems = into<int>(matching_subtile.elems());

    for (auto &coord : dst.layout.iter(matching_subtile)) {
        auto dst_buf = dst.subbuf(coord);
        auto src0_buf = src0.subbuf(coord);
        auto src1_buf = src1.subbuf(coord);

        int simd = default_ctx().simd();
        for (int idx = 0; idx < subtile_elems; idx += simd) {
            int elems = into<int>(std::min(subtile_elems - idx, simd));
            auto s0 = load_t::make(src0.layout.type().with_elems(elems),
                    src0.buf, idx * src0.layout.type().size());
            auto s1 = load_t::make(src1.layout.type().with_elems(elems),
                    src1.buf, idx * src1.layout.type().size());
            assign(dst_buf[dst.layout.type().size() * idx],
                    binary_op_t::make(op, s0, s1));
        }
    }
}

void barrier() {
    append(builtin_t::make("barrier")());
}

} // namespace dsl
GEMMSTONE_NAMESPACE_END
