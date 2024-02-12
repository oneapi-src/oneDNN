/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include "gpu/jit/v2/conv/ir_builder.hpp"

#include <sstream>

#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/kernel_info.hpp"
#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/ir/reorder.hpp"
#include "gpu/jit/pass/dpas_atomic.hpp"
#include "gpu/jit/pass/pass.hpp"
#include "gpu/jit/v2/conv/bridge.hpp"
#include "gpu/jit/v2/conv/plan.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace v2 {
namespace conv {

class let_ctx_t {
public:
    friend class let_ctx_mutator_t;

    let_ctx_t(const kernel_info_t &kernel_info, const grid_context_t &grid_ctx,
            const grid_t &tg_grid, const grid_t &thr_grid,
            const virt_grid_t &virt_grid, int simd, ir_context_t &ir_ctx)
        : kernel_info_(kernel_info), ir_ctx_(ir_ctx) {
        // Handle thread indices.
        for (int i = 0; i < grid_ctx.ndims(); i++) {
            auto value = grid_ctx.local_id(i);
            if (i == 0) value /= simd;
            auto thr_idx = thr_grid.index_var(i);
            let_stmts_.push_back(
                    let_t::make(thr_idx, cast(value, thr_idx.type())));
        }
        // Handle thread group indices.
        for (int i = 0; i < grid_ctx.ndims(); i++) {
            auto tg_idx = tg_grid.index_var(i);
            expr_map_.emplace(tg_idx, grid_ctx.tg_idx(i));
        }
        // Handle virtual grid indices.
        for (auto &kv : virt_grid.idxs()) {
            expr_map_.emplace(kv.first, get(kv.second));
        }
    }

    std::vector<stmt_t> let_stmts() const { return let_stmts_; }

    expr_t get(const expr_t &expr);

    std::vector<expr_t> get(const std::vector<expr_t> &exprs) {
        std::vector<expr_t> ret;
        for (auto &e : exprs)
            ret.push_back(get(e));
        return ret;
    }

    expr_t get_shuffle(const std::vector<expr_t> &exprs) {
        std::vector<expr_t> vec = get(exprs);
        return shuffle_t::make(vec);
    }

private:
    const expr_t *find(const expr_t &expr) const {
        auto it = expr_map_.find(expr);
        if (it != expr_map_.end()) return &it->second;
        return nullptr;
    }

    expr_t register_expr(const expr_t &key, const expr_t &value) {
        if (is_const(key) || key.is<const_var_t>() || key.is<var_t>())
            return value;
        if (auto *cached = find(key)) {
            ir_assert(value.is_same(*cached));
            return value;
        }
        auto tmp_var = ir_ctx_.create_tmp_var(type_t::s32());
        let_stmts_.push_back(let_t::make(tmp_var, value));
        expr_map_.emplace(key, tmp_var);
        return tmp_var;
    }

    expr_t get_var(const var_t &var) {
        if (auto *cached = find(var)) return *cached;
        return var;
    }

    expr_t get_const_var(const const_var_t &var) {
        if (auto *cached = find(var)) return *cached;

        auto value = kernel_info_.find_arg(var.name);
        return register_expr(var, value);
    }

    const kernel_info_t &kernel_info_;
    ir_context_t &ir_ctx_;
    object_eq_map_t<expr_t, expr_t> expr_map_;
    std::vector<stmt_t> let_stmts_;
};

class let_ctx_mutator_t : public ir_mutator_t {
public:
    let_ctx_mutator_t(let_ctx_t &ctx) : ctx_(ctx) {}

    object_t _mutate(const binary_op_t &obj) override {
        switch (obj.op_kind) {
            case op_kind_t::_div_up: return mutate((obj.a + obj.b - 1) / obj.b);
            default: return ir_mutator_t::_mutate(obj);
        }
    }

    object_t _mutate(const const_var_t &obj) override {
        return ctx_.get_const_var(obj);
    }

    object_t _mutate(const linear_t &obj) override {
        return mutate(obj.to_expr());
    }

    object_t _mutate(const var_t &obj) override { return ctx_.get_var(obj); }

private:
    let_ctx_t &ctx_;
};

expr_t let_ctx_t::get(const expr_t &expr) {
    if (auto *cached = find(expr)) return *cached;
    let_ctx_mutator_t mutator(*this);
    auto ret = mutator.mutate(expr);
    return register_expr(expr, ret);
}

struct offset_params_t {
    // Type of the offset.
    type_t type;
    // Execution size:
    // - esize = 1: used as a scalar
    // - esize > 1: used as a vector
    // Note that esize > 1 may be used with scalar offsets, in this case the
    // offset is broadcasted when used.
    int esize = 0;
    // Offset buffer size alignment (e.g. used for header allocations, aligned
    // at a GRF boundary).
    int buf_align = 0;
    // Whether the offset can be used with broadcasting (e.g. scalar mask with
    // multiple slots).
    bool allow_bcast = false;
    // Whether the offset can be used directly from the base (if the offset is
    // equal to the base).
    bool allow_reuse = false;
    // Optional pre-allocated buffer for the offset.
    expr_t buf;
    // Prefix for the buffer name.
    std::string buf_prefix;

    offset_params_t(
            const type_t &type, int esize = 1, const char *buf_prefix = nullptr)
        : type(type), esize(esize) {
        if (buf_prefix) this->buf_prefix = buf_prefix;
    }

    expr_t get_buffer(buffer_manager_t &buf_mgr, int size) const {
        if (buf_prefix.empty()) return buf;
        auto buf_name = buf_mgr.ir_ctx().create_tmp_name(buf_prefix);
        return buf_mgr.get(buf_name, size);
    }
};

// Offset is represented as the sum of three terms:
//     base + shift + shift_vec
// where:
// - (base + shift) is a scalar portion
// - shift_vec is a vector portion
//
// base/shift split is relative which is determined during load/store planning
// to group instructions performing access to a shifted tiles of the same
// sub-layout. In general "shift" portion consists of simpler expressions
// comparing with "base".
// shift_vec is a vector of offsets (e.g. for per slot in a message or per lane
// in a mask comparison).
struct offset_t {
    // Unique ID for the offset.
    int id = -1;
    // GRF buffer for the offset. If empty, the base storage is used for the
    // offset.
    expr_t buf;
    // Offset type (scalar or vector).
    type_t type;
    // Scalar base.
    expr_t base;
    // Scalar shift.
    expr_t shift;
    // Vector shift.
    expr_t shift_vec;
    // Loop increments, used to implement strength reduction.
    std::vector<expr_t> loop_incs;
    // Execution size.
    int esize;

    bool is_equal(const offset_t &other, bool compare_shift = true) const {
        if (type != other.type) return false;
        if (!base.is_equal(other.base)) return false;
        if (compare_shift && !shift.is_equal(other.shift)) return false;
        if (!shift_vec.is_equal(other.shift_vec)) return false;
        if (!ir_utils::is_equal(loop_incs, other.loop_incs)) return false;
        if (esize != other.esize) return false;
        return true;
    }

    bool operator==(const offset_t &other) const { return is_equal(other); }

    expr_t load() const {
        if (buf.is_empty()) return make_broadcast(base);
        return make_broadcast(load_t::make(type, buf, 0));
    }

    stmt_t store(const expr_t &_value) const {
        auto value = _value;
        if (value.type() != type) value = cast(value, type);
        return store_t::make(buf, 0, value);
    }

    stmt_t init_stmt() const {
        if (buf.is_empty()) return stmt_t();
        auto base_bcast = shuffle_t::make_broadcast(base + shift, type.elems());
        return store(base_bcast + shift_vec);
    }

    stmt_t inc_stmt(int loop_idx) const {
        if (loop_incs.empty()) return stmt_t();
        auto inc = loop_incs[loop_idx];
        if (is_zero(inc)) return stmt_t();
        inc = shuffle_t::make_broadcast(inc, type.elems());
        auto value = load_t::make(type, buf, 0) + inc;
        return store(value);
    }

    expr_t make_broadcast(const expr_t &e) const {
        if (e.type().elems() == esize) return e;
        return shuffle_t::make_broadcast(e, esize);
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "buf:       " << buf << std::endl;
        oss << "base:      " << base << std::endl;
        oss << "shift:     " << shift << std::endl;
        oss << "shift_vec: " << shift_vec << std::endl;
        oss << "loop_incs:";
        for (int i = 0; i < (int)loop_incs.size(); i++) {
            oss << std::endl;
            oss << "  " << loop_incs[i];
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

    static bool can_reuse_base(const type_t &type, const expr_t &base,
            const expr_t &shift, const expr_t &shift_vec,
            const std::vector<expr_t> &loop_incs) {
        if (!type.is_scalar()) return false;
        if (!is_zero(shift)) return false;
        if (!is_var(base) && !is_const(base)) return false;
        if (!all_of(shift_vec, 0)) return false;
        for (auto &e : loop_incs)
            if (!is_zero(e)) return false;
        return true;
    }
};

class send_header_t {
public:
    send_header_t() = default;
    send_header_t(const offset_t &off, const stmt_t &local_init = stmt_t())
        : off_(off), local_init_(local_init) {}

    const offset_t &off() const { return off_; }
    const expr_t &to_expr() const { return off_.buf; }
    const stmt_t &local_init() const { return local_init_; }

private:
    offset_t off_;
    stmt_t local_init_;
};

class send_mask_t {
public:
    send_mask_t() = default;

    void add_mask(const offset_t &off, const expr_t &bound, bool do_zero_cmp) {
        entries_.emplace_back(off, bound, do_zero_cmp);
    }

    expr_t to_expr() const {
        if (entries_.empty()) return expr_t();
        expr_t ret;
        for (auto &e : entries_) {
            auto cmp = (e.off.load() < e.off.make_broadcast(e.bound));
            ret = (ret.is_empty() ? cmp : (ret & cmp));
            if (e.do_zero_cmp) ret &= (e.off.load() >= e.off.make_broadcast(0));
        }
        return ret;
    }

private:
    struct entry_t {
        entry_t() = default;
        entry_t(const offset_t &off, const expr_t &bound, bool do_zero_cmp)
            : off(off), bound(bound), do_zero_cmp(do_zero_cmp) {}
        offset_t off;
        expr_t bound;
        bool do_zero_cmp = false;
    };

    std::vector<entry_t> entries_;
};

class offset_ctx_t {
public:
    offset_ctx_t(let_ctx_t &let_ctx, buffer_manager_t &buf_mgr,
            const loop_nest_t &loop_nest, const coord_info_t &coord_info)
        : let_ctx_(let_ctx)
        , buf_mgr_(buf_mgr)
        , loop_nest_(loop_nest)
        , coord_info_(coord_info) {}

    send_header_t add_header(const send_1d_desc_t &desc, const expr_t &mem_buf,
            const addr_t &addr, const expr_t &addr_inc) {
        auto base0 = cast(mem_buf, type_t::u64());
        auto params = offset_params_t(type_t::u64(), desc.slots, "h");
        params.buf_align = buf_mgr_.ir_ctx().grf_size();
        auto off = get_offset(
                base0, addr.base, addr.slot_incs, addr_inc, params);
        stmt_t local_init;
        if (!is_zero(off.shift) && off.type.is_scalar()) {
            for (auto &o : offsets_) {
                if (o.is_equal(off, /*compare_shift=*/false)) {
                    ir_assert(o.type.is_scalar());
                    local_init = off.store(o.load() + off.shift);
                    off.loop_incs.clear();
                    set_offset(off);
                    break;
                }
            }
        }
        return send_header_t(off, local_init);
    }

    send_header_t add_header(const send_2d_desc_t &desc, const expr_t &mem_buf,
            const expr_t &base, const expr_t &x_base, const expr_t &y_base,
            const expr_t &x_inc, const expr_t &y_inc) {
        auto base0 = cast(mem_buf, type_t::u64());
        auto params = offset_params_t(type_t::u64(), /*esize=*/1, "h");
        params.buf_align = desc.hw.grf_size();
        auto off = get_offset(base0, base, expr_t(0), params);
        auto x_params = offset_params_t(type_t::s32());
        auto y_params = offset_params_t(type_t::s32());
        x_params.buf = off.buf + send_t::header_2d_off_x();
        y_params.buf = off.buf + send_t::header_2d_off_y();
        auto x = get_offset(expr_t(0), x_base, x_inc, x_params);
        auto y = get_offset(expr_t(0), y_base, y_inc, y_params);

        int type_size = desc.type.size();
        auto W_enc = let_ctx_.get(desc.W) * type_size - 1;
        auto H_enc = let_ctx_.get(desc.H) - 1;
        auto P_enc = let_ctx_.get(desc.P) * type_size - 1;
        (void)get_offset(
                W_enc, off.buf + send_t::header_2d_off_surface_width());
        (void)get_offset(
                H_enc, off.buf + send_t::header_2d_off_surface_height());
        (void)get_offset(
                P_enc, off.buf + send_t::header_2d_off_surface_pitch());

        uint32_t w_enc = desc.w - 1;
        uint32_t h_enc = desc.h - 1;
        uint32_t count_enc = desc.c - 1;
        uint32_t whc_value = (count_enc << 16) + (h_enc << 8) + w_enc;
        (void)get_offset(whc_value, off.buf + send_t::header_2d_off_whc());

        return send_header_t(off);
    }

    send_mask_t add_mask(const mask_t &mask,
            const std::vector<expr_t> &mask_incs = std::vector<expr_t>()) {
        send_mask_t ret;
        for (int i = 0; i < mask.nmasks(); i++) {
            auto &dm = mask.dim_masks[i];
            if (dm.is_empty()) continue;
            auto shift = mask_incs.empty() ? expr_t(0) : mask_incs[i];
            auto params = offset_params_t(type_t::s32(), dm.slots(), "m");
            params.allow_bcast = true;
            params.allow_reuse = true;
            auto off = get_offset(
                    expr_t(0), dm.base, dm.slot_incs, shift, params);
            ret.add_mask(off, let_ctx_.get(dm.bound), dm.do_zero_cmp);
        }
        return ret;
    }

    stmt_t init_stmt() const {
        stmt_t ret;
        for (auto &o : offsets_) {
            ret = ret.append(o.init_stmt());
        }
        return ret;
    }

    stmt_t inc_loop_stmt(const loop_nest_entry_t &e) const {
        stmt_t ret;
        for (auto &o : offsets_) {
            auto inc = o.inc_stmt(e.idx);
            ret = ret.append(inc);
        }
        return ret;
    }

private:
    // base0 - memory buffer base address
    // base, shift_vec, shift - offset parts (see offset_t description)
    offset_t get_offset(const expr_t &base0, const expr_t &base,
            const std::vector<expr_t> &_shift_vec, const expr_t &_shift,
            const offset_params_t &_params) {
        auto params = _params;
        std::vector<expr_t> loop_idxs;
        for (auto &e : loop_nest_) {
            loop_idxs.push_back(coord_info_.loop_index(e.dim));
        }
        expr_t _base_init;
        std::vector<expr_t> _loop_incs;
        split_to_linear(base, loop_idxs, _base_init, _loop_incs);

        auto type = params.type.with_elems(params.esize);
        auto shift_vec = _shift_vec.empty() ? expr_t(0)
                                            : let_ctx_.get_shuffle(_shift_vec);
        if (params.allow_bcast) {
            if (auto *shuffle = shift_vec.as_ptr<shuffle_t>()) {
                if (shuffle->is_broadcast()) {
                    shift_vec = shuffle->vec[0];
                    type = type.scalar();
                }
            }
        }
        offset_t ret;
        ret.type = type;
        ret.base = base0 + let_ctx_.get(_base_init);
        ret.shift = let_ctx_.get(_shift);
        ret.shift_vec = shift_vec;
        ret.esize = params.esize;

        auto loop_incs = let_ctx_.get(_loop_incs);
        expr_t comp_value = 0;
        for (auto &e : loop_nest_) {
            auto loop_size = coord_info_.loop_size(e.dim);
            auto inc_value = simplify(loop_incs[e.idx] - comp_value);
            auto inc = let_ctx_.get(inc_value);
            ret.loop_incs.push_back(inc);
            comp_value = (loop_incs[e.idx] * loop_size);
        }

        if (params.allow_reuse) {
            for (auto &o : offsets_) {
                if (o == ret) return o;
            }
        }

        bool can_reuse_base = offset_t::can_reuse_base(
                ret.type, ret.base, ret.shift, ret.shift_vec, ret.loop_incs);
        if (!params.allow_reuse || !can_reuse_base) {
            int size = type.size();
            if (params.buf_align != 0)
                size = utils::rnd_up(size, params.buf_align);
            ret.buf = params.get_buffer(buf_mgr_, size);
        }

        return add_offset(ret);
    }

    offset_t get_offset(const expr_t &base0, const expr_t &base,
            const expr_t &shift, const offset_params_t &_params) {
        return get_offset(base0, base, std::vector<expr_t>(), shift, _params);
    }

    offset_t get_offset(const expr_t &base, const expr_t &buf) {
        offset_t ret;
        ret.buf = buf;
        ret.type = base.type();
        ret.base = base;
        ret.shift = expr_t(0);
        ret.shift_vec = expr_t(0);
        ret.esize = 1;
        return add_offset(ret);
    }

    offset_t add_offset(const offset_t &off) {
        offsets_.push_back(off);
        auto &ret = offsets_.back();
        ret.id = offset_id_++;
        return ret;
    }

    void set_offset(const offset_t &off) {
        for (auto &o : offsets_) {
            if (o.id == off.id) {
                o = off;
                return;
            }
        }
        ir_error_not_expected();
    }
    let_ctx_t &let_ctx_;
    buffer_manager_t &buf_mgr_;
    loop_nest_t loop_nest_;
    coord_info_t coord_info_;

    int offset_id_ = 0;
    std::vector<offset_t> offsets_;
};

class iterator_t {
public:
    iterator_t() = default;

    iterator_t(buffer_manager_t &buf_mgr) : buf_mgr_(&buf_mgr) {
        linear_loop_ = loop_t(loop_nest_entry_t(), 0, buf_mgr);
    }

    int nloops() const { return (int)loops_.size(); }

    void add_loop(const loop_nest_entry_t &e, const expr_t &bound) {
        if (is_one(bound)) return;
        loops_.emplace_back(e, bound, *buf_mgr_);
    }

    stmt_t init_stmt() const {
        stmt_t ret;
        for (auto &l : loops_) {
            ret = ret.append(l.store_stmt(0));
        }
        ret = linear_loop_.store_stmt(linear_bound() - 1).append(ret);
        return ret;
    }

    expr_t linear_loop_var() const { return linear_loop_.var(); }

    stmt_t check_bounds_stmt(const stmt_t &body) const {
        return if_t::make(linear_loop_.var() >= 0, body);
    }

    stmt_t inc_stmt(const offset_ctx_t &off_ctx) const {
        stmt_t body;
        for (int i = nloops() - 1; i >= 0; i--) {
            auto &l = loops_[i];
            auto *l_prev = (i - 1 >= 0) ? &loops_[i - 1] : nullptr;
            auto *l_next = (i + 1 < nloops()) ? &loops_[i + 1] : nullptr;
            stmt_t stmt;
            if (l_prev) stmt = stmt.append(l_prev->store_stmt(0));
            stmt = stmt.append(l.inc_stmt());
            stmt = stmt.append(off_ctx.inc_loop_stmt(l.entry));
            if (l_next)
                stmt = stmt.append(if_t::make(l.var() >= l.bound, body));
            body = stmt;
        }
        body = linear_loop_.inc_stmt(-1).append(body);
        return body;
    }

private:
    struct loop_t {
        loop_nest_entry_t entry;
        expr_t bound;
        expr_t var_buf;

        loop_t() = default;
        loop_t(const loop_nest_entry_t &entry, const expr_t &bound,
                buffer_manager_t &buf_mgr)
            : entry(entry), bound(bound) {
            auto buf_name = buf_mgr.ir_ctx().create_tmp_name("i");
            var_buf = buf_mgr.get(buf_name, sizeof(int32_t));
        }

        stmt_t store_stmt(const expr_t &value) const {
            return store_t::make(var_buf, 0, value);
        }

        stmt_t inc_stmt(int inc = 1) const { return store_stmt(var() + inc); }

        expr_t var() const { return load_t::make(type_t::s32(), var_buf, 0); }
    };

    expr_t linear_bound() const {
        expr_t ret;
        for (auto &l : loops_) {
            if (ret.is_empty()) {
                ret = l.bound;
            } else {
                ret *= l.bound;
            }
        }
        return ret;
    }

    buffer_manager_t *buf_mgr_ = nullptr;
    std::vector<loop_t> loops_;
    loop_t linear_loop_;
};

type_t to_send_type(const send_1d_desc_t &desc) {
    if (desc.type_size <= 8) return type_t::u(desc.type_size * 8);
    return type_t::oword(desc.type_size / 16);
}

int get_reg_off(const send_1d_plan_t &plan, const prb_coord_t<int> &coord) {
    return plan.reg_layout.offset_in_bytes(coord);
}

stmt_t create_stmt(const reorder_plan_t &plan, const expr_t &src_buf,
        const expr_t &dst_buf) {
    if (!plan) return stmt_t();
    return create_reorder_stmt(
            to_ir(plan.src), to_ir(plan.dst), src_buf, dst_buf);
}

stmt_t create_stmt(const send_1d_plan_t &plan, const expr_t &mem_buf,
        const expr_t &reg_buf, offset_ctx_t &off_ctx,
        const prb_coord_t<int> &coord, const prb_tile_t &tile,
        const layout_t &payload_layout, const prb_coord_t<int> &payload_coord) {
    for (auto &d : plan.entry_tile) {
        ir_assert(tile.at(d) % plan.entry_tile.at(d) == 0);
    }
    auto op = to_ir(plan.desc.op);
    auto address = send_address_t::a64;
    auto type = to_send_type(plan.desc);
    auto slots = plan.desc.slots;
    auto send_func = jit::send_t::make(
            plan.hw, op, address, type, slots, /*zero_out=*/true);
    auto &send = send_func.as<send_t>();
    stmt_t ret;
    for_each(tile, plan.entry_tile, [&](const prb_coord_t<int> &sub_coord) {
        int entry_idx = plan.reg_layout.to_linear_index(
                plan.entry_tile, coord + sub_coord);
        auto &e = plan.entries[entry_idx];
        ir_assert(e.coord == coord + sub_coord);
        auto header
                = off_ctx.add_header(plan.desc, mem_buf, plan.addr, e.addr_inc);
        auto mask = off_ctx.add_mask(plan.mask, e.mask_incs);
        auto call_reg_buf = reg_buf;
        if (!reg_buf.is_empty())
            call_reg_buf += payload_layout.offset_in_bytes(
                    payload_coord + sub_coord);
        auto call
                = send(mem_buf, header.to_expr(), call_reg_buf, mask.to_expr());
        ret = ret.append(header.local_init());
        ret = ret.append(call);
    });
    return ret;
}

stmt_t create_stmt(const send_2d_plan_t &plan, const expr_t &mem_buf,
        const expr_t &reg_buf, offset_ctx_t &off_ctx,
        const prb_coord_t<int> &coord, const prb_tile_t &tile,
        const layout_t &payload_layout, const prb_coord_t<int> &payload_coord) {
    auto op = to_ir(plan.desc.op, /*is_2d=*/true);
    auto &type = plan.desc.type;
    auto &desc = plan.desc;
    auto send_func = jit::send_t::make_2d(plan.hw, op, type, desc.w, desc.h,
            desc.c, desc.vnni, desc.transpose, /*zero_out=*/true);
    auto &send = send_func.as<send_t>();
    stmt_t ret;
    for_each(tile, plan.entry_tile, [&](const prb_coord_t<int> &sub_coord) {
        int entry_idx = plan.reg_layout.to_linear_index(
                plan.entry_tile, coord + sub_coord);
        auto &e = plan.entries[entry_idx];
        ir_assert(e.coord == coord + sub_coord);
        auto header = off_ctx.add_header(plan.desc, mem_buf, plan.base,
                plan.x_base, plan.y_base, e.x_inc, e.y_inc);
        auto mask = off_ctx.add_mask(plan.mask);
        auto call_reg_buf = reg_buf;
        if (!reg_buf.is_empty())
            call_reg_buf += payload_layout.offset_in_bytes(
                    payload_coord + sub_coord);
        auto call
                = send(mem_buf, header.to_expr(), call_reg_buf, mask.to_expr());
        ret = ret.append(header.local_init());
        ret = ret.append(call);
    });
    return ret;
}

stmt_t create_stmt(const send_plan_t &plan, const expr_t &mem_buf,
        const expr_t &reg_buf, offset_ctx_t &off_ctx,
        const prb_coord_t<int> &coord, const prb_tile_t &tile,
        const layout_t &payload_layout, const prb_coord_t<int> &payload_coord) {
    if (plan.is_1d())
        return create_stmt(plan._1d, mem_buf, reg_buf, off_ctx, coord, tile,
                payload_layout, payload_coord);
    if (plan.is_2d())
        return create_stmt(plan._2d, mem_buf, reg_buf, off_ctx, coord, tile,
                payload_layout, payload_coord);
    ir_error_not_expected();
    return stmt_t();
}

stmt_t create_stmt(const send_plan_t &plan, const expr_t &mem_buf,
        const expr_t &reg_buf, offset_ctx_t &off_ctx) {
    return create_stmt(plan, mem_buf, reg_buf, off_ctx, prb_coord_t<int>(),
            plan.reg_layout().int_dim_sizes(), plan.reg_layout(),
            prb_coord_t<int>());
}

class ir_builder_t {
public:
    ir_builder_t(const kernel_desc_t &desc, const kernel_info_t &kernel_info,
            const grid_context_t &grid_ctx, const plan_t &plan)
        : desc_(desc)
        , kernel_info_(kernel_info)
        , grid_ctx_(grid_ctx)
        , plan_(plan)
        , ir_ctx_(desc.exec_cfg(), cset_)
        , buf_mgr_(ir_ctx_)
        , let_ctx_(kernel_info, grid_ctx, plan.tg_grid, plan.thr_grid,
                  plan.virt_grid, desc.simd, ir_ctx_)
        , off_ctx_(let_ctx_, buf_mgr_, desc_.loop_nest, plan_.coord_info)
        , prefetch_off_ctx_(
                  let_ctx_, buf_mgr_, desc_.loop_nest, plan_.coord_info) {}

    stmt_t build() {
        build_prefetch();
        build_x2r_mul();
        build_c_store();

        stmt_t stmt;
        stmt = loop();
        stmt = inject_compute_alloc(stmt);
        stmt = init_stmt().append(stmt);
        stmt = zero_out_stmt().append(stmt);
        stmt = stmt.append(c_store_stmt_);
        stmt = inject_alloc_and_let(stmt);
        stmt = simplify(stmt, ir_ctx_);
        stmt = optimize_alloc_let(stmt, ir_ctx_);
        stmt = split_wide_stores(stmt, ir_ctx_);
        stmt = fixup_if_conditions(stmt, ir_ctx_);
        stmt = eliminate_common_subexprs(stmt, ir_ctx_, 16, 0);
        stmt = inject_bank_conflict_attribute(stmt, ir_ctx_);
        return stmt;
    }

private:
    stmt_t loop() const {
        auto &loop_nest = desc_.loop_nest;
        auto &coord_info = plan_.coord_info;
        int prefetch_dist = desc_.prefetch.dist;
        stmt_t init_stmt;
        iterator_t prefetch_it;
        if (prefetch_dist > 0) {
            prefetch_it = iterator_t(buf_mgr_);
            for (auto &e : loop_nest) {
                auto bound = let_ctx_.get(coord_info.loop_size(e.dim));
                prefetch_it.add_loop(e, bound);
            }
            init_stmt = init_stmt.append(prefetch_it.init_stmt());
            for (int i = 0; i < prefetch_dist; i++) {
                auto i_prefetch_stmt = prefetch_stmt_;
                if (i > 0)
                    i_prefetch_stmt
                            = prefetch_it.check_bounds_stmt(i_prefetch_stmt);
                init_stmt = init_stmt.append(i_prefetch_stmt);
                init_stmt = init_stmt.append(
                        prefetch_it.inc_stmt(prefetch_off_ctx_));
            }
        }
        stmt_t ret;
        if (prefetch_dist > 0) {
            ret = ret.append(prefetch_it.check_bounds_stmt(prefetch_stmt_));
        }
        ret = ret.append(x2r_mul_stmt_);
        if (prefetch_dist > 0) {
            ret = ret.append(prefetch_it.inc_stmt(prefetch_off_ctx_));
        }
        for (auto &e : loop_nest) {
            auto var = let_ctx_.get(coord_info.loop_index(e.dim));
            auto bound = let_ctx_.get(coord_info.loop_size(e.dim));
            ret = ret.append(off_ctx_.inc_loop_stmt(e));
            ret = for_t::make(var, 0, bound, ret);
        }
        ret = init_stmt.append(ret);
        return ret;
    }

    stmt_t zero_out_stmt() const {
        auto &c_entry = buf_mgr_.find_ref("c");
        auto ret = stmt_group_t::make(stmt_label_t::c_zero_out(),
                funcs::zero_out(c_entry.buf, c_entry.size));
        return ret;
    }

    stmt_t init_stmt() const {
        stmt_t ret;
        ret = ret.append(off_ctx_.init_stmt());
        ret = ret.append(prefetch_off_ctx_.init_stmt());
        return ret;
    }

    stmt_t inject_alloc_and_let(const stmt_t &stmt) const {
        stmt_t ret = stmt;
        ret = inject_out_alloc(ret);
        ret = inject_header_alloc(ret);
        ret = inject_let_stmts(ret, let_ctx_.let_stmts());
        ret = inject_global_alloc(ret);
        ret = inject_index_let(ret);
        ret = inject_external_var_let(ret, ir_ctx_);
        return ret;
    }

    static bool is_compute_alloc_buf(const expr_t &buf) {
        return !is_out_alloc_buf(buf) && !is_offset_buf(buf);
    }

    static bool is_out_alloc_buf(const expr_t &buf) {
        auto &buf_name = buf.as<var_t>().name;
        return utils::one_of(buf_name, "c", "c_tmp");
    }

    static bool is_offset_buf(const expr_t &buf) {
        auto &buf_name = buf.as<var_t>().name;
        if (buf_name.find("h_") == 0) return true;
        if (buf_name.find("m_") == 0) return true;
        return false;
    }

    stmt_t inject_compute_alloc(const stmt_t &stmt) const {
        return buf_mgr_.inject_allocs(stmt, is_compute_alloc_buf);
    }

    stmt_t inject_out_alloc(const stmt_t &stmt) const {
        return buf_mgr_.inject_allocs(stmt, is_out_alloc_buf);
    }

    stmt_t inject_header_alloc(const stmt_t &stmt) const {
        return buf_mgr_.inject_allocs(stmt, is_offset_buf);
    }

    stmt_t inject_global_alloc(const stmt_t &stmt) const {
        std::vector<stmt_t> allocs;
        for (int i = 0; i < kernel_info_.nargs(); i++) {
            auto &var = kernel_info_.arg_var(i);
            if (!var.type().is_ptr()) continue;
            allocs.push_back(alloc_t::make(var, 0, alloc_kind_t::global));
        }
        return inject_alloc_stmts(stmt, allocs);
    }

    stmt_t inject_index_let(const stmt_t &stmt) const {
        auto &tg_grid = plan_.tg_grid;
        auto &coord_info = plan_.coord_info;
        stmt_t ret = stmt;
        for (auto &d : conv_index_dims(plan_.desc.prop)) {
            auto tg_idx = let_ctx_.get(coord_info.tg_index(d));
            if (is_const(tg_idx)) continue;
            auto base_tg_idx = let_ctx_.get(tg_grid.index_var(d));
            if (base_tg_idx.is_empty()) continue;
            auto value = unpack_tg_index(d);
            ret = let_t::make(tg_idx, value, ret);
        }
        return ret;
    }

    expr_t unpack_tg_index(const prb_dim_t &dim) const {
        auto &tg_grid = plan_.tg_grid;
        auto base_idx = let_ctx_.get(tg_grid.index_var(dim));
        if (base_idx.is_empty()) return expr_t();

        expr_t value = base_idx;
        auto &dims = tg_grid.dims(tg_grid.index(dim));
        int ndims = (int)dims.size();
        for (int i = 0; i < ndims; i++) {
            if (dims[i] == dim) break;
            auto i_dim_size
                    = kernel_info_.find_arg(dims[i].str() + "_grid_size");
            auto i_magic = kernel_info_.find_arg(dims[i].str() + "_magic");
            value = ternary_op_t::make(
                    op_kind_t::_idiv, value, i_dim_size, i_magic);
        }
        auto dim_size = kernel_info_.find_arg(dim.str() + "_grid_size");
        auto magic = kernel_info_.find_arg(dim.str() + "_magic");
        value = ternary_op_t::make(op_kind_t::_imod, value, dim_size, magic);
        return value;
    }

    void build_prefetch() {
        auto &prefetch = plan_.prefetch;
        if (prefetch.a_prefetch) {
            auto a_prefetch = create_stmt(prefetch.a_prefetch, a_mem_buf(),
                    expr_t(), prefetch_off_ctx_);
            prefetch_stmt_ = prefetch_stmt_.append(a_prefetch);
        }
        if (prefetch.b_prefetch) {
            auto b_prefetch = create_stmt(prefetch.b_prefetch, b_mem_buf(),
                    expr_t(), prefetch_off_ctx_);
            prefetch_stmt_ = prefetch_stmt_.append(b_prefetch);
        }
    }

    void build_x2r_x_load(const std::string &prefix, const send_plan_t &load,
            const reorder_plan_t &reorder, const expr_t &mem_buf) {
        expr_t load_buf;
        expr_t mul_buf;
        if (reorder) {
            load_buf = buf_mgr_.get(prefix + "_tmp", load.reg_layout().size());
            mul_buf = buf_mgr_.get(prefix, reorder.dst.size());
        } else {
            load_buf = buf_mgr_.get(prefix, load.reg_layout().size());
            mul_buf = load_buf;
        }
        auto load_stmt = create_stmt(load, mem_buf, load_buf, off_ctx_);
        auto reorder_stmt = create_stmt(reorder, load_buf, mul_buf);
        x2r_mul_stmt_ = x2r_mul_stmt_.append(load_stmt);
        x2r_mul_stmt_ = x2r_mul_stmt_.append(reorder_stmt);
    }

    void build_x2r() {
        auto &x2r = plan_.x2r;
        build_x2r_x_load("a", x2r.a_load, x2r.a_reorder, a_mem_buf());
        build_x2r_x_load("b", x2r.b_load, x2r.b_reorder, b_mem_buf());
    }

    void build_mul() {
        auto &fma = plan_.fma;
        auto &a_layout = fma.a_layout;
        auto &b_layout = fma.b_layout;
        auto &c_layout = fma.c_layout;
        auto a_buf = buf_mgr_.get("a");
        auto b_buf = buf_mgr_.get("b");
        auto c_buf = buf_mgr_.get("c", c_layout.size());

        for (auto &d : a_layout.dims())
            ir_assert(fma.inst_tile.has(d)) << d;
        for (auto &d : b_layout.dims())
            ir_assert(fma.inst_tile.has(d)) << d;

        // BMNK order.
        prb_dim_t dims[4];
        int blocks[4] = {1, 1, 1, 1};
        int sizes[4] = {1, 1, 1, 1};
        dim_map_t<prb_dim_t, int> bmnk_map;
        bmnk_map[prb_dims::b] = 0;
        bmnk_map[prb_dims::m] = 1;
        bmnk_map[prb_dims::n] = 2;
        bmnk_map[prb_dims::k] = 3;
        for (auto &d : fma.inst_tile) {
            int idx = bmnk_map.at(to_gemm(d, desc_.prop));
            dims[idx] = d;
            blocks[idx] = fma.inst_tile[d];
            sizes[idx] = (idx != 2 ? a_layout : b_layout).int_dim_size(d);
        }

        // BKNM order.
        int i0 = 0;
        int i1 = 3;
        int i2 = 2;
        int i3 = 1;
        stmt_t stmt;
        prb_coord_t<int> off(0);
        bool is_a_bcast = (blocks[0] * blocks[1] * blocks[3] == 1);
        bool is_b_bcast = (blocks[0] * blocks[2] * blocks[3] == 1);
        func_t fma_func;
        switch (fma.fma) {
            case fma_kind_t::mad: {
                int a_stride = is_a_bcast ? 0 : a_layout.inner_stride();
                int b_stride = is_b_bcast ? 0 : b_layout.inner_stride();
                fma_func = mad_t::make(plan_.hw, c_layout.type(), fma.simd,
                        a_layout.type(), a_stride, b_layout.type(), b_stride);
                break;
            }
            case fma_kind_t::dpas: {
                fma_func = dpas_t::make(/*is_dpasw=*/false, fma.simd, 8, 8,
                        c_layout.type(), b_layout.type(), a_layout.type());
                break;
            }
            default: ir_error_not_expected();
        }
        for (int b = 0; b < sizes[i0]; b += blocks[i0]) {
            off[dims[i0]] = b;
            for (int k = 0; k < sizes[i1]; k += blocks[i1]) {
                off[dims[i1]] = k;
                for (int n = 0; n < sizes[i2]; n += blocks[i2]) {
                    off[dims[i2]] = n;
                    for (int m = 0; m < sizes[i3]; m += blocks[i3]) {
                        off[dims[i3]] = m;
                        int a_off = a_layout.offset_in_bytes(off);
                        int b_off = b_layout.offset_in_bytes(off);
                        int c_off = c_layout.offset_in_bytes(off);
                        auto dst = c_buf[c_off];
                        auto src1 = a_buf[a_off];
                        auto src2 = b_buf[b_off];
                        if (fma.fma == fma_kind_t::dpas) std::swap(src1, src2);
                        stmt = stmt.append(
                                fma_func.call({dst, dst, src1, src2}));
                    }
                }
            }
        }
        stmt = inject_atomic(stmt);
        x2r_mul_stmt_ = x2r_mul_stmt_.append(stmt);
    }

    void build_x2r_mul() {
        build_x2r();
        build_mul();
    }

    void build_c_store() {
        auto &fma = plan_.fma;
        auto &epilogue = plan_.epilogue;
        auto &store = epilogue.c_store;
        auto c_tile = store.reg_layout().int_dim_sizes();
        auto &c_buf = buf_mgr_.find_buf("c");
        for_each(c_tile, epilogue.tile, [&](const prb_coord_t<int> &coord) {
            auto payload_buf = c_buf;
            auto payload_layout = store.reg_layout();
            auto payload_coord = coord;
            if (epilogue.reorder) {
                auto c_tmp_buf
                        = buf_mgr_.get("c_tmp", epilogue.reorder.dst.size());
                int src_off = fma.c_layout.offset_in_bytes(coord);
                auto stmt = create_stmt(
                        epilogue.reorder, c_buf + src_off, c_tmp_buf);
                c_store_stmt_ = c_store_stmt_.append(stmt);
                payload_buf = c_tmp_buf;
                payload_layout = epilogue.reorder.dst;
                payload_coord = prb_coord_t<int>();
            }
            auto stmt = create_stmt(store, c_mem_buf(), payload_buf, off_ctx_,
                    coord, epilogue.tile, payload_layout, payload_coord);
            c_store_stmt_ = c_store_stmt_.append(stmt);
        });
    }

    expr_t mem_buf(tensor_kind_t abc) const {
        std::string name;
        std::string src("src");
        std::string wei("wei");
        std::string dst("dst");
        switch (abc) {
            case tensor_kind_t::a:
                name = pick_a(desc_.prop, src, wei, dst);
                break;
            case tensor_kind_t::b:
                name = pick_b(desc_.prop, src, wei, dst);
                break;
            case tensor_kind_t::c:
                name = pick_c(desc_.prop, src, wei, dst);
                break;
            default: ir_error_not_expected();
        }
        return kernel_info_.find_arg(name.c_str());
    }

    expr_t a_mem_buf() const { return mem_buf(tensor_kind_t::a); }
    expr_t b_mem_buf() const { return mem_buf(tensor_kind_t::b); }
    expr_t c_mem_buf() const { return mem_buf(tensor_kind_t::c); }

    kernel_desc_t desc_;
    kernel_info_t kernel_info_;
    grid_context_t grid_ctx_;
    plan_t plan_;

    mutable constraint_set_t cset_;
    mutable ir_context_t ir_ctx_;
    mutable buffer_manager_t buf_mgr_;
    mutable let_ctx_t let_ctx_;
    mutable offset_ctx_t off_ctx_;
    mutable offset_ctx_t prefetch_off_ctx_;

    stmt_t prefetch_stmt_;
    stmt_t x2r_mul_stmt_;
    stmt_t c_store_stmt_;
};

stmt_t build_ir(const kernel_desc_t &desc, const kernel_info_t &kernel_info,
        const grid_context_t &grid_ctx) {
    auto plan = create_conv_plan(desc);
    if (!plan) ir_except_not_implemented("Cannot create plan.");

    ir_info() << desc << std::endl;
    ir_trace() << plan << std::endl;

    ir_builder_t builder(desc, kernel_info, grid_ctx, plan);
    auto stmt = builder.build();
    ir_trace() << "Convolution kernel body:\n" << stmt << std::endl;
    return stmt;
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
