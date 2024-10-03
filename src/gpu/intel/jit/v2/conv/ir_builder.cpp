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

#include "gpu/intel/jit/v2/conv/ir_builder.hpp"

#include <sstream>

#include "gpu/intel/jit/ir/ir.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/jit/ir/message.hpp"
#include "gpu/intel/jit/ir/reduce.hpp"
#include "gpu/intel/jit/ir/reorder.hpp"
#include "gpu/intel/jit/pass/dpas.hpp"
#include "gpu/intel/jit/pass/pass.hpp"
#include "gpu/intel/jit/utils/trace.hpp"
#include "gpu/intel/jit/v2/conv/bridge.hpp"
#include "gpu/intel/jit/v2/conv/plan.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

struct loop_t {
    size_t idx = 0;
    pvar_t dim;
    expr_t var;
    expr_t size;

    loop_t() = default;
    loop_t(size_t idx, const pvar_t &dim, const expr_t &var, const expr_t &size)
        : idx(idx), dim(dim), var(var), size(size) {}
};

class loop_nest_t {
public:
    loop_nest_t() = default;

    void add_loop(const pvar_t &dim, const expr_t &idx, const expr_t &size) {
        loops_.push_back(loop_t(loops_.size(), dim, idx, size));
    }

    size_t nloops() const { return loops_.size(); }
    const loop_t &operator[](size_t idx) const { return loops_[idx]; }
    std::vector<expr_t> indices() const {
        std::vector<expr_t> ret;
        ret.reserve(nloops());
        for (size_t i = 0; i < nloops(); i++) {
            ret.push_back(loops_[i].var);
        }
        return ret;
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "nloops: " << nloops();
        for (size_t i = 0; i < nloops(); i++) {
            oss << std::endl;
            oss << "  var: " << loops_[i].var << " size: " << loops_[i].size;
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    std::vector<loop_t> loops_;
};

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
    // Whether inline initialization can be used (see offset_t for details).
    bool allow_inline_init = false;
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
    // Offset version. This is relevant for offsets that are used in multiple
    // versions of the same loop, e.g. load and prefetch.
    int version = -1;
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
    // Inline initialization. When set, the offset is initialized right before
    // use. This implies no loop increments and no pre-initialization. This is
    // used as an optimization when offset A is a shifted version of another
    // offset B: in this case we can do A = B + shift and avoid any other
    // operations.
    stmt_t inline_init;

    bool is_equal(const offset_t &other, bool compare_shift = true) const {
        if (version != other.version) return false;
        if (type != other.type) return false;
        if (!base.is_equal(other.base)) return false;
        if (compare_shift && !shift.is_equal(other.shift)) return false;
        if (!shift_vec.is_equal(other.shift_vec)) return false;
        if (!ir_utils::is_equal(loop_incs, other.loop_incs)) return false;
        if (esize != other.esize) return false;
        if (!ir_utils::is_equal(inline_init, other.inline_init)) return false;
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
        if (buf.is_empty() || !inline_init.is_empty()) return stmt_t();
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
    send_header_t(const offset_t &off) : off_(off) {}
    const offset_t &off() const { return off_; }
    const expr_t &to_expr() const { return off_.buf; }

private:
    offset_t off_;
};

class send_mask_t {
public:
    send_mask_t() = default;

    void add_mask(
            const offset_t &off, const expr_t &bound, bool has_underflow) {
        entries_.emplace_back(off, bound, has_underflow);
    }

    expr_t to_expr() const {
        if (entries_.empty()) return expr_t();
        expr_t ret;
        for (auto &e : entries_) {
            auto cmp = (e.off.load() < e.off.make_broadcast(e.bound));
            ret = (ret.is_empty() ? std::move(cmp) : (ret & cmp));
            if (e.has_underflow)
                ret &= (e.off.load() >= e.off.make_broadcast(0));
        }
        return ret;
    }

private:
    struct entry_t {
        entry_t() = default;
        entry_t(const offset_t &off, const expr_t &bound, bool has_underflow)
            : off(off), bound(bound), has_underflow(has_underflow) {}
        offset_t off;
        expr_t bound;
        bool has_underflow = false;
    };

    std::vector<entry_t> entries_;
};

class offset_scope_t {
public:
    offset_scope_t(buffer_manager_t &buf_mgr, ir_context_t &ir_ctx,
            const loop_nest_t &loop_nest)
        : buf_mgr_(buf_mgr), ir_ctx_(ir_ctx), loop_nest_(loop_nest) {}

    send_header_t add_header(int version, const send_1d_desc_t &desc,
            const expr_t &mem_buf, const addr_t &addr, const expr_t &addr_inc) {
        auto base0 = cast(mem_buf, type_t::u64());
        auto params = offset_params_t(type_t::u64(), desc.slots, "h");
        params.buf_align = buf_mgr_.ir_ctx().grf_size();
        params.allow_inline_init = true;
        auto off = get_offset(
                version, base0, addr.base, addr.slot_incs, addr_inc, params);
        return send_header_t(off);
    }

    send_header_t add_header(int version, const send_2d_desc_t &desc,
            const expr_t &mem_buf, const expr_t &base, const expr_t &x_base,
            const expr_t &y_base, const expr_t &x_inc, const expr_t &y_inc) {
        auto base0 = cast(mem_buf, type_t::u64());
        auto params = offset_params_t(type_t::u64(), /*esize=*/1, "h");
        params.buf_align = desc.hw.grf_size();
        auto off = get_offset(version, base0, base, expr_t(0), params);
        auto x_params = offset_params_t(type_t::s32());
        auto y_params = offset_params_t(type_t::s32());
        x_params.buf = off.buf + send_t::header_2d_off_x();
        y_params.buf = off.buf + send_t::header_2d_off_y();
        auto x = get_offset(version, expr_t(0), x_base, x_inc, x_params);
        auto y = get_offset(version, expr_t(0), y_base, y_inc, y_params);

        int type_size = desc.type.size();
        auto W_enc = to_simple_expr(desc.W) * type_size - 1;
        auto H_enc = to_simple_expr(desc.H) - 1;
        auto P_enc = to_simple_expr(desc.P) * type_size - 1;
        (void)get_offset(version, W_enc,
                off.buf + send_t::header_2d_off_surface_width());
        (void)get_offset(version, H_enc,
                off.buf + send_t::header_2d_off_surface_height());
        (void)get_offset(version, P_enc,
                off.buf + send_t::header_2d_off_surface_pitch());

        uint32_t w_enc = desc.w - 1;
        uint32_t h_enc = desc.h - 1;
        uint32_t count_enc = desc.c - 1;
        uint32_t whc_value = (count_enc << 16) + (h_enc << 8) + w_enc;
        (void)get_offset(
                version, whc_value, off.buf + send_t::header_2d_off_whc());

        return send_header_t(off);
    }

    send_mask_t add_mask(int version, const mask_t &mask,
            const std::vector<expr_t> &mask_incs) {
        send_mask_t ret;
        for (int i = 0; i < mask.nmasks(); i++) {
            auto &dm = mask.dim_masks[i];
            if (dm.is_empty()) continue;
            auto shift = mask_incs.empty() ? expr_t(0) : mask_incs[i];
            auto params = offset_params_t(type_t::s32(), dm.slots(), "m");
            params.allow_bcast = true;
            params.allow_reuse = true;
            auto off = get_offset(
                    version, expr_t(0), dm.base, dm.slot_incs, shift, params);
            ret.add_mask(off, to_simple_expr(dm.bound), dm.has_underflow);
        }
        return ret;
    }

    stmt_t init_stmt(int version) const {
        stmt_t ret;
        for (auto &o : offsets_) {
            if (o.version != version) continue;
            ret = ret.append(o.init_stmt());
        }
        return ret;
    }

    stmt_t inc_loop_stmt(int loop_idx, int version) const {
        stmt_t ret;
        for (auto &o : offsets_) {
            if (o.version != version) continue;
            auto inc = o.inc_stmt(loop_idx);
            ret = ret.append(inc);
        }
        return ret;
    }

    stmt_t inject_let_stmts(const stmt_t &stmt) const {
        return jit::inject_let_stmts(stmt, let_stmts_);
    }

private:
    // base0 - memory buffer base address
    // base, shift_vec, shift - offset parts (see offset_t description)
    offset_t get_offset(int version, const expr_t &base0, const expr_t &base,
            const std::vector<expr_t> &_shift_vec, const expr_t &_shift,
            const offset_params_t &_params) {
        auto params = _params;
        expr_t _base_init;
        std::vector<expr_t> _loop_incs;
        split_to_linear(base, loop_nest_.indices(), _base_init, _loop_incs);

        auto type = params.type.with_elems(params.esize);
        auto shift_vec
                = _shift_vec.empty() ? expr_t(0) : shuffle_t::make(_shift_vec);
        if (params.allow_bcast) {
            if (auto *shuffle = shift_vec.as_ptr<shuffle_t>()) {
                if (shuffle->is_broadcast()) {
                    shift_vec = shuffle->vec[0];
                    type = type.scalar();
                }
            }
        }
        offset_t ret;
        ret.version = version;
        ret.type = type;
        ret.base = base0 + _base_init;
        ret.shift = _shift;
        ret.shift_vec = std::move(shift_vec);
        ret.esize = params.esize;

        expr_t comp_value = 0;
        for (size_t i = 0; i < loop_nest_.nloops(); i++) {
            auto loop_size = loop_nest_[i].size;
            auto inc_value = simplify(_loop_incs[i] - comp_value);
            auto inc = to_simple_expr(inc_value);
            ret.loop_incs.push_back(inc);
            comp_value = to_simple_expr(_loop_incs[i] * loop_size);
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
            buf_versions_.emplace(ret.buf, version);
        }

        // Try to use inline initialization.
        if (params.allow_inline_init && !is_zero(ret.shift)
                && ret.type.is_scalar()) {
            for (auto &o : offsets_) {
                if (o.is_equal(ret, /*compare_shift=*/false)) {
                    ir_assert(o.type.is_scalar());
                    ret.inline_init = ret.store(o.load() + ret.shift);
                    ret.loop_incs.clear();
                    break;
                }
            }
        }

        return add_offset(ret);
    }

    offset_t get_offset(int version, const expr_t &base0, const expr_t &base,
            const expr_t &shift, const offset_params_t &_params) {
        return get_offset(
                version, base0, base, std::vector<expr_t>(), shift, _params);
    }

    offset_t get_offset(int version, const expr_t &base, const expr_t &buf) {
        offset_t ret;
        ret.version = version;
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
        return off;
    }

    expr_t to_simple_expr(const expr_t &e) {
        if (is_const(e) || e.is<const_var_t>() || e.is<var_t>()) return e;
        auto it = expr_to_let_var_.find(e);
        if (it != expr_to_let_var_.end()) return it->second;
        auto tmp_var = ir_ctx_.create_tmp_var(type_t::s32());
        let_stmts_.push_back(let_t::make(tmp_var, e));
        expr_to_let_var_.emplace(e, tmp_var);
        return tmp_var;
    }

    buffer_manager_t &buf_mgr_;
    ir_context_t &ir_ctx_;
    loop_nest_t loop_nest_;
    object_eq_map_t<expr_t, expr_t> expr_to_let_var_;
    std::vector<stmt_t> let_stmts_;
    std::vector<offset_t> offsets_;
    object_map_t<expr_t, int> buf_versions_;
};

class offset_ctx_t {
public:
    offset_ctx_t() = default;
    offset_ctx_t(buffer_manager_t &buf_mgr, ir_context_t &ir_ctx,
            const loop_nest_t &loop_nest = loop_nest_t())
        : version_(0)
        , scope_(std::make_shared<offset_scope_t>(buf_mgr, ir_ctx, loop_nest)) {
    }

    offset_ctx_t bump_version() const {
        ir_assert(version_ != -1);
        offset_ctx_t ret = *this;
        ret.version_++;
        return ret;
    }

    send_header_t add_header(const send_1d_desc_t &desc, const expr_t &mem_buf,
            const addr_t &addr, const expr_t &addr_inc) {
        return scope_->add_header(version_, desc, mem_buf, addr, addr_inc);
    }

    send_header_t add_header(const send_2d_desc_t &desc, const expr_t &mem_buf,
            const expr_t &base, const expr_t &x_base, const expr_t &y_base,
            const expr_t &x_inc, const expr_t &y_inc) {
        return scope_->add_header(
                version_, desc, mem_buf, base, x_base, y_base, x_inc, y_inc);
    }

    send_mask_t add_mask(const mask_t &mask,
            const std::vector<expr_t> &mask_incs = std::vector<expr_t>()) {
        return scope_->add_mask(version_, mask, mask_incs);
    }

    stmt_t init_stmt() const { return scope_->init_stmt(version_); }
    stmt_t inject_let_stmts(const stmt_t &stmt) const {
        return scope_->inject_let_stmts(stmt);
    }
    stmt_t inc_loop_stmt(int loop_idx) const {
        return scope_->inc_loop_stmt(loop_idx, version_);
    }

private:
    int version_ = -1;
    std::shared_ptr<offset_scope_t> scope_;
};

class iterator_t {
public:
    iterator_t() = default;

    iterator_t(buffer_manager_t &buf_mgr, const loop_nest_t &loop_nest)
        : loop_nest_(loop_nest) {
        linear_idx_ = loop_index_t(buf_mgr);
        for (size_t i = 0; i < loop_nest.nloops(); i++) {
            loop_idxs_.emplace_back(buf_mgr);
        }
    }

    int nloops() const { return (int)loop_nest_.nloops(); }

    stmt_t init_stmt() const {
        stmt_t ret;
        for (int i = 0; i < nloops(); i++) {
            ret = ret.append(loop_idxs_[i].store(0));
        }
        ret = linear_idx_.store(linear_bound() - 1).append(ret);
        return ret;
    }

    stmt_t check_bounds_stmt(const stmt_t &body) const {
        return if_t::make(linear_idx_.var() >= 0, body);
    }

    stmt_t inc_stmt(const offset_ctx_t &off_ctx) const {
        stmt_t body;
        for (int i = nloops() - 1; i >= 0; i--) {
            stmt_t stmt;
            if (i - 1 >= 0) stmt = stmt.append(loop_idxs_[i - 1].store(0));
            stmt = stmt.append(loop_idxs_[i].inc_stmt());
            stmt = stmt.append(off_ctx.inc_loop_stmt(i));
            if (i + 1 < nloops())
                stmt = stmt.append(if_t::make(
                        loop_idxs_[i].var() >= loop_nest_[i].size, body));
            body = std::move(stmt);
        }
        body = linear_idx_.inc_stmt(-1).append(body);
        return body;
    }

private:
    struct loop_index_t {
        expr_t buf;

        loop_index_t() = default;
        loop_index_t(buffer_manager_t &buf_mgr) {
            auto buf_name = buf_mgr.ir_ctx().create_tmp_name("i");
            buf = buf_mgr.get(buf_name, sizeof(int32_t));
        }

        stmt_t store(const expr_t &value) const {
            return store_t::make(buf, 0, value);
        }

        stmt_t inc_stmt(int inc = 1) const { return store(var() + inc); }

        expr_t var() const { return load_t::make(type_t::s32(), buf, 0); }
    };

    expr_t linear_bound() const {
        expr_t ret;
        for (int i = 0; i < nloops(); i++) {
            if (ret.is_empty()) {
                ret = loop_nest_[i].size;
            } else {
                ret *= loop_nest_[i].size;
            }
        }
        return ret;
    }

    loop_nest_t loop_nest_;
    std::vector<loop_index_t> loop_idxs_;
    loop_index_t linear_idx_;
};

type_t to_send_type(const send_1d_desc_t &desc) {
    if (desc.type_size <= 8) return type_t::u(desc.type_size * 8);
    return type_t::oword(desc.type_size / 16);
}

int get_reg_off(const send_1d_plan_t &plan, const pvar_coord_t<dim_t> &coord) {
    return into<int>(plan.reg_layout.offset_in_bytes(coord));
}

stmt_t create_stmt(const reduce_plan_t &plan, const expr_t &src_buf,
        const expr_t &dst_buf) {
    if (!plan) return stmt_t();
    return create_reduce_stmt(
            to_ir(plan.src), to_ir(plan.dst), src_buf, dst_buf);
}

stmt_t create_stmt(const reorder_plan_t &plan, const expr_t &src_buf,
        const expr_t &dst_buf) {
    if (!plan) return stmt_t();
    return create_reorder_stmt(
            to_ir(plan.src), to_ir(plan.dst), src_buf, dst_buf);
}

stmt_t create_stmt(const send_1d_plan_t &plan, const expr_t &mem_buf,
        const expr_t &reg_buf, offset_ctx_t &off_ctx,
        const pvar_coord_t<dim_t> &coord, const pvar_tile_t &tile,
        const layout_t &payload_layout,
        const pvar_coord_t<dim_t> &payload_coord) {
    for (auto &d : plan.entry_tile) {
        ir_assert(tile.at(d) % plan.entry_tile.at(d) == 0);
    }
    auto op = to_ir(plan.desc.op);
    auto address = to_ir(plan.desc.address);
    auto type = to_send_type(plan.desc);
    auto slots = plan.desc.slots;
    auto send_func = jit::send_t::make(
            plan.hw, op, address, type, slots, /*zero_out=*/true);
    auto &send = send_func.as<send_t>();
    stmt_t ret;
    for_each(tile, plan.entry_tile, [&](const pvar_coord_t<dim_t> &sub_coord) {
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
        ret = ret.append(header.off().inline_init);
        ret = ret.append(call);
    });
    return ret;
}

stmt_t create_stmt(const send_2d_plan_t &plan, const expr_t &mem_buf,
        const expr_t &reg_buf, offset_ctx_t &off_ctx,
        const pvar_coord_t<dim_t> &coord, const pvar_tile_t &tile,
        const layout_t &payload_layout,
        const pvar_coord_t<dim_t> &payload_coord) {
    auto op = to_ir(plan.desc.op, /*is_2d=*/true);
    auto &type = plan.desc.type;
    auto &desc = plan.desc;
    auto send_func = jit::send_t::make_2d(plan.hw, op, type, desc.w, desc.h,
            desc.c, desc.vnni, desc.transpose, /*zero_out=*/true);
    auto &send = send_func.as<send_t>();
    stmt_t ret;
    for_each(tile, plan.entry_tile, [&](const pvar_coord_t<dim_t> &sub_coord) {
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
        ret = ret.append(header.off().inline_init);
        ret = ret.append(call);
    });
    return ret;
}

stmt_t create_stmt(const send_plan_t &plan, const expr_t &mem_buf,
        const expr_t &reg_buf, offset_ctx_t &off_ctx,
        const pvar_coord_t<dim_t> &coord, const pvar_tile_t &tile,
        const layout_t &payload_layout,
        const pvar_coord_t<dim_t> &payload_coord) {
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
    return create_stmt(plan, mem_buf, reg_buf, off_ctx, pvar_coord_t<dim_t>(),
            plan.reg_layout().int_dim_sizes(), plan.reg_layout(),
            pvar_coord_t<dim_t>());
}

class idiv_fixup_mutator_t : public ir_mutator_t {
public:
    idiv_fixup_mutator_t(const kernel_info_t &kernel_info)
        : kernel_info_(kernel_info) {}

    object_t _mutate(const binary_op_t &obj) {
        bool is_var_idiv = (obj.op_kind == op_kind_t::_div) && obj.type.is_int()
                && !is_const(obj.b);
        if (!is_var_idiv) return ir_mutator_t::_mutate(obj);
        ir_assert(obj.b.is<const_var_t>())
                << "Cannot handle integer division, expected const var to "
                   "access magic value: "
                << obj;
        auto magic = kernel_info_.find_arg(obj.b.str() + "_magic");
        return ternary_op_t::make(
                op_kind_t::_idiv, obj.a, cast(obj.b, type_t::u32()), magic);
    }

private:
    const kernel_info_t &kernel_info_;
};

stmt_t fixup_idiv(const stmt_t &s, const kernel_info_t &kernel_info,
        ir_context_t &ir_ctx) {
    trace_start();
    auto ret = idiv_fixup_mutator_t(kernel_info).mutate(s);
    trace_pass("fixup_idiv", ret, ir_ctx);
    return ret;
}

class var_replacer_t : public ir_mutator_t {
public:
    var_replacer_t(const kernel_info_t &kernel_info,
            const grid_context_t &grid_ctx, const grid_t &tg_grid)
        : kernel_info_(kernel_info) {
        for (int i = 0; i < grid_ctx.ndims(); i++) {
            auto tg_idx = tg_grid.index_var(i);
            var_map_.emplace(tg_idx, grid_ctx.tg_idx(i));
        }
    }
    object_t _mutate(const var_t &obj) override {
        return map_var(obj.name, obj, /*is_const=*/false);
    }

    object_t _mutate(const const_var_t &obj) override {
        return map_var(obj.name, obj, /*is_const_var=*/true);
    }

    object_t _mutate(const binary_op_t &obj) override {
        switch (obj.op_kind) {
            case op_kind_t::_div_up: return mutate((obj.a + obj.b - 1) / obj.b);
            default: return ir_mutator_t::_mutate(obj);
        }
    }

private:
    expr_t map_var(
            const std::string &name, const expr_t &var, bool is_const_var) {
        auto it = var_map_.find(var);
        if (it != var_map_.end()) return it->second;
        auto arg = kernel_info_.find_arg(name, /*allow_empty=*/true);
        if (arg.is_empty() && is_const_var) {
            ir_error_not_expected() << "Cannot map const var: " << var;
        }
        auto value = (arg.is_empty() ? var : arg);
        var_map_.emplace(var, value);
        return value;
    }

    const kernel_info_t &kernel_info_;
    object_map_t<expr_t, expr_t> var_map_;
};

stmt_t finalize_vars(const stmt_t &stmt, const kernel_info_t &kernel_info,
        const grid_context_t &grid_ctx, const grid_t &tg_grid,
        ir_context_t &ir_ctx) {
    auto ret = var_replacer_t(kernel_info, grid_ctx, tg_grid).mutate(stmt);
    ret = inject_external_var_let(ret, ir_ctx);
    return ret;
}

loop_nest_t make_loop_nest(
        const loop_desc_t &loop_desc, const coord_info_t &coord_info) {
    loop_nest_t ret;
    for (auto &e : loop_desc) {
        const auto &var = coord_info.loop_index(e.dim);
        const auto &size = coord_info.loop_size(e.dim);
        if (is_one(size)) continue;
        ret.add_loop(e.dim, var, size);
    }
    return ret;
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
        , loop_nest_(make_loop_nest(desc_.loop_desc, plan_.coord_info))
        , off_ctx_(buf_mgr_, ir_ctx_, loop_nest_)
        , prefetch_off_ctx_(off_ctx_.bump_version())
        , epilogue_off_ctx_(prefetch_off_ctx_.bump_version()) {}

    stmt_t build() {
        build_prefetch();
        build_x2r_mul();
        build_epilogue();

        stmt_t compute_stmt;
        compute_stmt = compute_stmt.append(zero_out_stmt());
        compute_stmt = compute_stmt.append(off_ctx_.init_stmt());
        compute_stmt = compute_stmt.append(prefetch_off_ctx_.init_stmt());
        compute_stmt = compute_stmt.append(loop());

        stmt_t epilogue_stmt;
        epilogue_stmt = epilogue_stmt.append(epilogue_off_ctx_.init_stmt());
        epilogue_stmt = epilogue_stmt.append(epilogue_stmt_);

        stmt_t stmt;
        stmt = stmt.append(compute_stmt);
        stmt = stmt.append(epilogue_stmt);

        stmt = inject_alloc_stmts(stmt, buf_mgr_);
        stmt = off_ctx_.inject_let_stmts(stmt);
        stmt = inject_global_alloc(stmt);
        stmt = inject_index_let(stmt);
        stmt = fixup_idiv(stmt, kernel_info_, ir_ctx_);
        stmt = finalize_vars(
                stmt, kernel_info_, grid_ctx_, plan_.tg_grid, ir_ctx_);

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
        auto &coord_info = plan_.coord_info;
        int prefetch_dist = desc_.prefetch.dist;
        stmt_t init_stmt;
        iterator_t prefetch_it;
        if (prefetch_dist > 0) {
            prefetch_it = iterator_t(buf_mgr_, loop_nest_);
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
        for (size_t i = 0; i < loop_nest_.nloops(); i++) {
            auto &loop = loop_nest_[i];
            const auto &var = coord_info.loop_index(loop.dim);
            const auto &bound = loop.size;
            ret = ret.append(off_ctx_.inc_loop_stmt((int)loop.idx));
            ret = for_t::make(var, 0, bound, ret);
        }
        ret = init_stmt.append(ret);
        return ret;
    }

    stmt_t zero_out_stmt() const {
        auto &c_entry = buf_mgr_.find_ref("c");
        auto stmt = funcs::zero_out(c_entry.buf, c_entry.size);
        if (desc_.prop == prop_kind::backward_weights && desc_.with_bias) {
            auto &bia_entry = buf_mgr_.find_ref("bia_buf");
            stmt = stmt.append(funcs::zero_out(bia_entry.buf, bia_entry.size));
        }
        auto ret = stmt_group_t::make(stmt_label_t::c_zero_out(), stmt);
        return ret;
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
            const auto &tg_idx = coord_info.tg_index(d);
            if (is_const(tg_idx)) continue;
            auto base_tg_idx = tg_grid.index_var(d);
            if (base_tg_idx.is_empty()) continue;
            auto value = unpack_tg_index(d);
            ret = let_t::make(tg_idx, value, ret);
        }
        for (auto &kv : plan_.virt_grid.idxs()) {
            ret = let_t::make(kv.first, kv.second, ret);
        }
        for (int i = 0; i < grid_ctx_.ndims(); i++) {
            auto value = grid_ctx_.local_id(i);
            if (i == 0) value /= plan_.desc.simd;
            auto thr_idx = plan_.thr_grid.index_var(i);
            ret = let_t::make(thr_idx, cast(value, thr_idx.type()), ret);
        }
        return ret;
    }

    expr_t unpack_tg_index(const pvar_t &dim) const {
        auto &tg_grid = plan_.tg_grid;
        auto base_idx = tg_grid.index_var(dim);
        if (base_idx.is_empty()) return expr_t();

        expr_t value = std::move(base_idx);
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

    void build_x2r(const x2r_plan_t &plan) {
        auto prefix = to_string(plan.tensor_kind);
        expr_t load_buf;
        expr_t mul_buf;
        if (plan.reorder) {
            load_buf = buf_mgr_.get(
                    prefix + "_tmp", plan.load.reg_layout().size());
            mul_buf = buf_mgr_.get(prefix, plan.reorder.dst.size());
        } else {
            load_buf = buf_mgr_.get(prefix, plan.load.reg_layout().size());
            mul_buf = load_buf;
        }
        auto load_stmt = create_stmt(
                plan.load, mem_buf(plan.tensor_kind), load_buf, off_ctx_);
        auto reorder_stmt = create_stmt(plan.reorder, load_buf, mul_buf);
        x2r_mul_stmt_ = x2r_mul_stmt_.append(load_stmt);
        x2r_mul_stmt_ = x2r_mul_stmt_.append(reorder_stmt);
    }

    void build_mul(const fma_plan_t &fma, const expr_t &c_buf) {
        auto &a_layout = fma.a_layout;
        auto &b_layout = fma.b_layout;
        auto &c_layout = fma.c_layout;
        auto a_buf = buf_mgr_.get("a");
        auto b_buf = buf_mgr_.get("b");

        for (auto &d : a_layout.dims())
            ir_assert(fma.inst_tile.has(d)) << d;
        for (auto &d : b_layout.dims())
            ir_assert(fma.inst_tile.has(d)) << d;

        // BMNK order.
        pvar_t dims[4];
        dim_t blocks[4] = {1, 1, 1, 1};
        int sizes[4] = {1, 1, 1, 1};
        pvar_map_t<int> bmnk_map;
        bmnk_map[pvars::b] = 0;
        bmnk_map[pvars::m] = 1;
        bmnk_map[pvars::n] = 2;
        bmnk_map[pvars::k] = 3;
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
        pvar_coord_t<dim_t> off;
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
                        dim_t a_off = a_layout.offset_in_bytes(off);
                        dim_t b_off = b_layout.offset_in_bytes(off);
                        dim_t c_off = c_layout.offset_in_bytes(off);
                        auto dst = c_buf[c_off];
                        auto src1 = a_buf[a_off];
                        auto src2 = b_buf[b_off];
                        if (fma.fma == fma_kind_t::dpas) std::swap(src1, src2);
                        stmt = stmt.append(fma_func.call(
                                {dst, dst, std::move(src1), std::move(src2)}));
                    }
                }
            }
        }
        stmt = inject_dpas_atomic(stmt, /*filter_by_label=*/false);
        x2r_mul_stmt_ = x2r_mul_stmt_.append(stmt);
    }

    void build_bia_reduce(x2r_plan_t &x2r, const expr_t &bia_buf) {
        if (desc_.prop != prop_kind::backward_weights || !desc_.with_bias
                || x2r.tensor_kind != tensor_kind_t::b)
            return;
        auto b_buf = buf_mgr_.find_buf("b");
        auto &b_layout = x2r.layout;
        auto stmt = create_reduce_stmt(to_ir(b_layout), to_ir(x2r.bia_layout),
                b_buf, bia_buf, tensor_t(), (1 << 1) | (1 << 2));
        x2r_mul_stmt_ = x2r_mul_stmt_.append(stmt);
    }

    void build_x2r_mul() {
        auto &x2r_fma = plan_.x2r_fma;
        auto c_buf = buf_mgr_.get("c", x2r_fma.c_layout.size());
        for (auto &s : x2r_fma.stages) {
            if (s.is_fma()) {
                build_mul(s.fma, c_buf);
            } else if (s.is_x2r()) {
                build_x2r(s.x2r);
                build_bia_reduce(s.x2r,
                        buf_mgr_.get("bia_buf", x2r_fma.bia_layout.size()));
            }
        }
    }

    void build_bias_store() {
        if (desc_.prop != prop_kind::backward_weights || !desc_.with_bias)
            return;
        auto &fma = plan_.x2r_fma;
        auto &epilogue = plan_.epilogue;
        auto &bia_buf = buf_mgr_.find_buf("bia_buf");
        auto bia_tile = epilogue.bia_store.reg_layout().int_dim_sizes();
        auto epilogue_tile = bia_tile;
        for (auto &d : bia_tile)
            epilogue_tile[d] = epilogue.tile[d];
        for_each(
                bia_tile, epilogue_tile, [&](const pvar_coord_t<dim_t> &coord) {
                    auto bia_payload_buf = bia_buf;
                    auto bia_payload_layout = epilogue.bia_store.reg_layout();
                    auto payload_coord = coord;
                    if (epilogue.bia_reorder) {
                        auto bia_tmp_buf = buf_mgr_.get(
                                "bia_tmp", epilogue.bia_reorder.dst.size());
                        dim_t src_off = fma.bia_layout.offset_in_bytes(coord);
                        auto stmt = create_stmt(epilogue.bia_reorder,
                                bia_buf + src_off, bia_tmp_buf);
                        epilogue_stmt_ = epilogue_stmt_.append(stmt);
                        bia_payload_buf = std::move(bia_tmp_buf);
                        bia_payload_layout = epilogue.bia_reorder.dst;
                        payload_coord = pvar_coord_t<dim_t>();
                    }
                    auto bia_stmt = create_stmt(epilogue.bia_store,
                            bia_mem_buf(), bia_payload_buf, epilogue_off_ctx_,
                            coord, epilogue_tile, bia_payload_layout,
                            payload_coord);
                    bia_stmt = if_t::make(epilogue.reduce_cond, bia_stmt);
                    epilogue_stmt_ = epilogue_stmt_.append(bia_stmt);
                });
    }

    void build_slm_reduce() {
        auto &slm_reduce = plan_.epilogue.slm_reduce;
        if (!slm_reduce) return;

        auto &c_buf = buf_mgr_.find_buf("c");
        auto c_tmp_buf
                = buf_mgr_.get("c_reduce", slm_reduce.load.reg_layout().size());
        auto c_slm_buf = buf_mgr_.get("slm", slm_reduce.slm_usage_bytes());
        auto store_stmt = create_stmt(
                slm_reduce.store, c_slm_buf, c_buf, epilogue_off_ctx_);
        auto load_stmt = create_stmt(
                slm_reduce.load, c_slm_buf, c_tmp_buf, epilogue_off_ctx_);
        auto reduce_stmt = create_stmt(slm_reduce.reduce, c_tmp_buf, c_buf);
        epilogue_stmt_ = epilogue_stmt_.append(store_stmt);
        epilogue_stmt_ = epilogue_stmt_.append(funcs::barrier());
        epilogue_stmt_ = epilogue_stmt_.append(load_stmt);
        epilogue_stmt_ = epilogue_stmt_.append(
                funcs::zero_out(c_buf, slm_reduce.c_layout.size()));
        epilogue_stmt_ = epilogue_stmt_.append(reduce_stmt);
    }

    void build_c_store() {
        auto &c_layout = plan_.x2r_fma.c_layout;
        auto &epilogue = plan_.epilogue;
        auto &store = epilogue.c_store;
        auto c_tile = store.reg_layout().int_dim_sizes();
        auto &c_buf = buf_mgr_.find_buf("c");
        for_each(c_tile, epilogue.tile, [&](const pvar_coord_t<dim_t> &coord) {
            auto payload_buf = c_buf;
            auto payload_layout = c_layout;
            auto payload_coord = coord;
            if (epilogue.reorder) {
                auto c_tmp_buf
                        = buf_mgr_.get("c_tmp", epilogue.reorder.dst.size());
                dim_t src_off = c_layout.offset_in_bytes(coord);
                auto stmt = create_stmt(
                        epilogue.reorder, c_buf + src_off, c_tmp_buf);
                epilogue_stmt_ = epilogue_stmt_.append(stmt);
                payload_buf = std::move(c_tmp_buf);
                payload_layout = epilogue.reorder.dst;
                payload_coord = pvar_coord_t<dim_t>();
            }
            auto stmt = create_stmt(store, c_mem_buf(), payload_buf,
                    epilogue_off_ctx_, coord, epilogue.tile, payload_layout,
                    payload_coord);
            epilogue_stmt_ = epilogue_stmt_.append(stmt);
        });
    }

    void build_epilogue() {
        build_slm_reduce();
        build_c_store();
        build_bias_store();
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
        return kernel_info_.find_arg(name);
    }

    expr_t a_mem_buf() const { return mem_buf(tensor_kind_t::a); }
    expr_t b_mem_buf() const { return mem_buf(tensor_kind_t::b); }
    expr_t c_mem_buf() const { return mem_buf(tensor_kind_t::c); }
    expr_t bia_mem_buf() const { return kernel_info_.find_arg("bia"); }

    kernel_desc_t desc_;
    kernel_info_t kernel_info_;
    grid_context_t grid_ctx_;
    plan_t plan_;

    mutable constraint_set_t cset_;
    mutable ir_context_t ir_ctx_;
    mutable buffer_manager_t buf_mgr_;
    loop_nest_t loop_nest_;
    mutable offset_ctx_t off_ctx_;
    mutable offset_ctx_t prefetch_off_ctx_;
    mutable offset_ctx_t epilogue_off_ctx_;

    stmt_t prefetch_stmt_;
    stmt_t x2r_mul_stmt_;
    stmt_t epilogue_stmt_;
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
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
