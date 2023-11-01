/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#include "gpu/jit/pass/pass.hpp"
#include "gpu/jit/v2/conv/bridge.hpp"
#include "gpu/jit/v2/conv/plan.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace v2 {
namespace conv {

class planner_ctx_t {
public:
    planner_ctx_t(const kernel_info_t &kernel_info,
            const grid_context_t &grid_ctx, const linear_expr_ctx_t &expr_ctx,
            const grid_t &tg_grid, const grid_t &thr_grid, int simd,
            ir_context_t &ir_ctx)
        : kernel_info_(kernel_info), ir_ctx_(ir_ctx), expr_ctx_(expr_ctx) {
        for (int i = 0; i < grid_ctx.ndims(); i++) {
            auto value = grid_ctx.local_id(i);
            if (i == 0) value /= simd;
            auto thr_idx = to_ir(thr_grid.index_var(i));
            init_let_stmts_.push_back(
                    let_t::make(thr_idx, cast(value, thr_idx.type())));
        }
        init_expr_map(grid_ctx, tg_grid);
    }

    const hw_t &hw() const { return ir_ctx_.hw(); }
    const kernel_info_t &kernel_info() const { return kernel_info_; }
    ir_context_t &ir_ctx() const { return ir_ctx_; }

    std::vector<stmt_t> init_let_stmts() const { return init_let_stmts_; }

    expr_t to_ir(const expr_t &expr) const { return to_ir_impl(expr); }

    std::vector<expr_t> to_ir(const std::vector<expr_t> &exprs) {
        std::vector<expr_t> ret;
        for (auto &e : exprs)
            ret.push_back(to_ir(e));
        return ret;
    }

    expr_t to_ir_shuffle(const std::vector<expr_t> &exprs) {
        std::vector<expr_t> vec = to_ir(exprs);
        return shuffle_t::make(vec);
    }

private:
    void init_expr_map(const grid_context_t &grid_ctx, const grid_t &tg_grid) {
        for (int i = 0; i < grid_ctx.ndims(); i++) {
            auto tg_idx = tg_grid.index_var(i);
            expr_map_.emplace(tg_idx, grid_ctx.tg_idx(i));
        }
        for (auto &kv : expr_ctx_.const_vars()) {
            to_ir_impl(kv.first, this);
        }
    }

    expr_t to_ir_impl(
            const expr_t &expr, planner_ctx_t *this_mutable = nullptr) const {
        if (expr.is_empty()) return expr_t();
        if (expr.is<int_imm_t>()) return expr;

        auto it = expr_map_.find(expr);
        if (it != expr_map_.end()) return it->second;

        if (expr.is<var_t>()) return expr;

        if (auto *linear = expr.as_ptr<linear_t>()) {
            auto ret = to_ir_impl(linear->c, this_mutable);
            for (int i = 0; i < linear->nargs(); i++) {
                auto u = to_ir_impl(linear->u_vec[i], this_mutable);
                auto v = to_ir_impl(linear->v_vec[i], this_mutable);
                ret = ret + u * v;
            }
            return ret;
        }

        // This is an unknown const_var_t, need to update the object.
        ir_assert(this_mutable) << "Need to handle const var: " << expr;
        if (auto *var = expr.as_ptr<const_var_t>()) {
            auto ir_var
                    = kernel_info_.find_arg(var->name, /*allow_empty=*/true);
            if (ir_var.is_empty())
                ir_var = this_mutable->register_const_var(expr);
            this_mutable->expr_map_.emplace(expr, ir_var);
            return ir_var;
        }

        ir_error_not_expected() << expr;
        return expr_t();
    }

    expr_t register_const_var(const expr_t &_var) {
        auto &var = _var.as<const_var_t>();
        auto &var_value = expr_ctx_.const_value(_var);
        auto &op = var_value.as<binary_op_t>();
        auto a = to_ir_impl(op.a, this);
        auto b = to_ir_impl(op.b, this);
        expr_t ir_value;
        switch (op.op_kind) {
            case op_kind_t::_add: ir_value = a + b; break;
            case op_kind_t::_mul: ir_value = a * b; break;
            case op_kind_t::_minus: ir_value = -a; break;
            case op_kind_t::_div_up: ir_value = (a + b - 1) / b; break;
            default: ir_error_not_expected() << _var;
        }
        auto ir_var = ir_ctx_.create_tmp_var(type_t::s32(), var.name);
        auto let = let_t::make(ir_var, ir_value);
        init_let_stmts_.push_back(let);
        return ir_var;
    }

    const kernel_info_t &kernel_info_;
    ir_context_t &ir_ctx_;
    const linear_expr_ctx_t &expr_ctx_;

    object_map_t<expr_t, expr_t> expr_map_;
    std::vector<stmt_t> init_let_stmts_;
};

class build_ctx_t : public planner_ctx_t {
public:
    build_ctx_t(const plan_t &plan, const kernel_info_t &kernel_info,
            const grid_context_t &grid_ctx, ir_context_t &ir_ctx)
        : planner_ctx_t(kernel_info, grid_ctx, plan.expr_ctx, plan.tg_grid,
                plan.thr_grid, plan.desc.simd, ir_ctx)
        , plan_(plan) {
        a_mem_buf_ = kernel_info.find_arg(a_name());
        b_mem_buf_ = kernel_info.find_arg(b_name());
        c_mem_buf_ = kernel_info.find_arg(c_name());
    }

    const plan_t &plan() const { return plan_; }
    const expr_t &a_mem_buf() const { return a_mem_buf_; }
    const expr_t &b_mem_buf() const { return b_mem_buf_; }
    const expr_t &c_mem_buf() const { return c_mem_buf_; }

    int to_bmnk_idx(const prb_dim_t &dim) const {
        auto bmnk = to_gemm(dim, plan_.desc.prop);
        if (bmnk == prb_dims::b) return 0;
        if (bmnk == prb_dims::m) return 1;
        if (bmnk == prb_dims::n) return 2;
        if (bmnk == prb_dims::k) return 3;
        ir_error_not_expected();
        return 0;
    }

private:
    bool is_fwd() const { return plan_.desc.prop == prop_kind::forward; }
    bool is_bwd_d() const {
        return plan_.desc.prop == prop_kind::backward_data;
    }
    bool is_bwd_w() const {
        return plan_.desc.prop == prop_kind::backward_weights;
    }
    const char *a_name() const { return is_bwd_d() ? "dst" : "src"; }
    const char *b_name() const { return is_bwd_w() ? "dst" : "wei"; }
    const char *c_name() const {
        return is_fwd() ? "dst" : (is_bwd_d() ? "src" : "wei");
    }

    const plan_t &plan_;

    expr_t a_mem_buf_;
    expr_t b_mem_buf_;
    expr_t c_mem_buf_;
};

struct offset_params_t {
    type_t type;
    int esize = 0;
    int buf_align = 0;
    bool allow_bcast = true;
    bool allow_reuse = false;
    expr_t buf;
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

struct offset_t {
    expr_t buf;
    type_t type;
    expr_t base;
    expr_t base_vec;
    std::vector<expr_t> loop_incs;
    int esize;

    bool operator==(const offset_t &other) const {
        if (type != other.type) return false;
        if (!base.is_equal(other.base)) return false;
        if (!base_vec.is_equal(other.base_vec)) return false;
        if (!ir_utils::is_equal(loop_incs, other.loop_incs)) return false;
        if (esize != other.esize) return false;
        return true;
    }

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
        auto base_bcast = shuffle_t::make_broadcast(base, type.elems());
        return store(base_bcast + base_vec);
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
        oss << "buf:      " << buf << std::endl;
        oss << "base:     " << base << std::endl;
        oss << "base_vec: " << base_vec << std::endl;
        oss << "loop_incs:";
        for (int i = 0; i < (int)loop_incs.size(); i++) {
            oss << std::endl;
            oss << "  " << loop_incs[i];
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

    static bool can_reuse_base(const type_t &type, const expr_t &base,
            const expr_t &base_vec, const std::vector<expr_t> &loop_incs) {
        if (!type.is_scalar()) return false;
        if (!is_var(base) && !is_const(base)) return false;
        if (!all_of(base_vec, 0)) return false;
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
    offset_ctx_t(planner_ctx_t &planner_ctx, buffer_manager_t &buf_mgr,
            const loop_nest_t &loop_nest, const coord_info_t &coord_info)
        : planner_ctx_(planner_ctx)
        , buf_mgr_(buf_mgr)
        , loop_nest_(loop_nest)
        , coord_info_(coord_info) {}

    send_header_t add_header(const send_1d_desc_t &desc, const expr_t &mem_buf,
            const addr_t &addr, const expr_t &inc) {
        auto base0 = cast(mem_buf, type_t::u64());
        auto params = offset_params_t(type_t::u64(), desc.slots, "h");
        params.buf_align = planner_ctx_.ir_ctx().grf_size();
        params.allow_bcast = false;
        auto off = get_offset(base0, addr.base, addr.slot_incs, inc, params);
        return send_header_t(off);
    }

    send_header_t add_header(const send_2d_desc_t &desc, const expr_t &mem_buf,
            const expr_t &base, const expr_t &x_base, const expr_t &y_base,
            const expr_t &x_inc, const expr_t &y_inc) {
        auto base0 = cast(mem_buf, type_t::u64());
        auto params = offset_params_t(type_t::u64(), /*esize=*/1, "h");
        params.buf_align = planner_ctx_.ir_ctx().grf_size();
        params.allow_bcast = false;
        auto off = get_offset(base0, base, expr_t(0), params);
        auto x_params = offset_params_t(type_t::s32());
        auto y_params = offset_params_t(type_t::s32());
        x_params.buf = off.buf + send_t::header_2d_off_x();
        y_params.buf = off.buf + send_t::header_2d_off_y();
        auto x = get_offset(expr_t(0), x_base, x_inc, x_params);
        auto y = get_offset(expr_t(0), y_base, y_inc, y_params);

        int type_size = desc.type.size();
        auto W_enc = planner_ctx_.to_ir(desc.W) * type_size - 1;
        auto H_enc = planner_ctx_.to_ir(desc.H) - 1;
        auto P_enc = planner_ctx_.to_ir(desc.P) * type_size - 1;
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
            auto inc = mask_incs.empty() ? expr_t(0) : mask_incs[i];
            auto params = offset_params_t(type_t::s32(), dm.slots(), "m");
            params.allow_reuse = true;
            auto off
                    = get_offset(expr_t(0), dm.base, dm.slot_incs, inc, params);
            ret.add_mask(off, planner_ctx_.to_ir(dm.bound), dm.do_zero_cmp);
        }
        return ret;
    }

    stmt_t init_stmt(const stmt_t &body) const {
        stmt_t ret;
        for (auto &o : offsets_) {
            ret = ret.append(o.init_stmt());
        }
        ret = ret.append(body);
        ret = inject_let_stmts(ret, let_stmts_);
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
    offset_t get_offset(const expr_t &base0, const expr_t &base,
            const std::vector<expr_t> &_base_vec, const expr_t &inc,
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
        auto base_vec = _base_vec.empty()
                ? expr_t(0)
                : planner_ctx_.to_ir_shuffle(_base_vec);
        if (params.allow_bcast) {
            if (auto *shuffle = base_vec.as_ptr<shuffle_t>()) {
                if (shuffle->is_broadcast()) {
                    base_vec = shuffle->vec[0];
                    type = type.scalar();
                }
            }
        }
        offset_t ret;
        ret.type = type;
        ret.base = simplify(base0 + planner_ctx_.to_ir(_base_init)
                + planner_ctx_.to_ir(inc));
        ret.base_vec = base_vec;
        ret.esize = params.esize;

        auto loop_incs = planner_ctx_.to_ir(_loop_incs);
        expr_t comp_value = 0;
        for (auto &e : loop_nest_) {
            auto loop_size = planner_ctx_.to_ir(coord_info_.loop_size(e.dim));
            auto inc_value = simplify(loop_incs[e.idx] - comp_value);
            auto inc = inc_value;
            if (!is_const(inc_value)) {
                auto inc_var
                        = planner_ctx_.ir_ctx().create_tmp_var(type_t::s32());
                let_stmts_.push_back(let_t::make(inc_var, inc_value));
                inc = inc_var;
            }
            ret.loop_incs.push_back(inc);
            comp_value = (loop_incs[e.idx] * loop_size);
        }

        if (params.allow_reuse) {
            for (auto &o : offsets_) {
                if (o == ret) return o;
            }
        }

        bool can_reuse_base = offset_t::can_reuse_base(
                ret.type, ret.base, ret.base_vec, ret.loop_incs);
        if (!params.allow_reuse || !can_reuse_base) {
            int size = type.size();
            if (params.buf_align != 0)
                size = utils::rnd_up(size, params.buf_align);
            ret.buf = params.get_buffer(buf_mgr_, size);
        }

        offsets_.push_back(ret);
        return ret;
    }

    offset_t get_offset(const expr_t &base0, const expr_t &base,
            const expr_t &inc, const offset_params_t &_params) {
        return get_offset(base0, base, std::vector<expr_t>(), inc, _params);
    }

    offset_t get_offset(const expr_t &base, const expr_t &buf) {
        offset_t ret;
        ret.buf = buf;
        ret.type = base.type();
        ret.base = base;
        ret.base_vec = expr_t(0);
        ret.esize = 1;
        offsets_.push_back(ret);
        return ret;
    }

    planner_ctx_t &planner_ctx_;
    buffer_manager_t &buf_mgr_;
    loop_nest_t loop_nest_;
    coord_info_t coord_info_;

    std::vector<offset_t> offsets_;
    std::vector<stmt_t> let_stmts_;
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
        auto call_reg_buf = reg_buf
                + payload_layout.offset_in_bytes(payload_coord + sub_coord);
        auto call
                = send(mem_buf, header.to_expr(), call_reg_buf, mask.to_expr());
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
        auto call_reg_buf = reg_buf
                + payload_layout.offset_in_bytes(payload_coord + sub_coord);
        auto call
                = send(mem_buf, header.to_expr(), call_reg_buf, mask.to_expr());
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

class compute_builder_t {
public:
    compute_builder_t(build_ctx_t &ctx)
        : ctx_(ctx)
        , buf_mgr_(ctx.ir_ctx())
        , off_ctx_(ctx, buf_mgr_, ctx.plan().desc.loop_nest,
                  ctx.plan().coord_info) {}

    void build() {
        build_x2r_mul();
        build_c_store();
    }

    stmt_t iter_stmt() const {
        stmt_t ret;
        ret = ret.append(x2r_mul_stmt_);
        return ret;
    }

    stmt_t loop_nest() const {
        auto &loop_nest = ctx_.plan().desc.loop_nest;
        auto &coord_info = ctx_.plan().coord_info;
        stmt_t ret = iter_stmt();
        for (auto &e : loop_nest) {
            auto var = ctx_.to_ir(coord_info.loop_index(e.dim));
            auto bound = ctx_.to_ir(coord_info.loop_size(e.dim));
            ret = ret.append(off_ctx_.inc_loop_stmt(e));
            ret = for_t::make(var, 0, bound, ret);
        }
        return ret;
    }

    const stmt_t &c_store_stmt() const { return c_store_stmt_; }

    stmt_t zero_out_stmt() const {
        auto &c_entry = buf_mgr_.find_ref("c");
        auto ret = stmt_group_t::make(stmt_label_t::c_zero_out(),
                funcs::zero_out(c_entry.buf, c_entry.size));
        return ret;
    }

    stmt_t init_stmt(const stmt_t &stmt) const {
        return off_ctx_.init_stmt(stmt);
    }

    stmt_t inject_compute_alloc_stmts(const stmt_t &stmt) const {
        return buf_mgr_.inject_allocs(stmt, is_compute_alloc_buf);
    }

    stmt_t inject_out_alloc_stmts(const stmt_t &stmt) const {
        return buf_mgr_.inject_allocs(stmt, is_out_alloc_buf);
    }

    stmt_t inject_header_alloc_stmts(const stmt_t &stmt) const {
        return buf_mgr_.inject_allocs(stmt, is_offset_buf);
    }

private:
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

    void build_x2r() {
        auto &x2r = ctx_.plan().x2r;
        auto a_buf = buf_mgr_.get("a", x2r.a_load.reg_layout().size());
        auto b_buf = buf_mgr_.get("b", x2r.b_load.reg_layout().size());
        auto a_load
                = create_stmt(x2r.a_load, ctx_.a_mem_buf(), a_buf, off_ctx_);
        auto b_load
                = create_stmt(x2r.b_load, ctx_.b_mem_buf(), b_buf, off_ctx_);
        x2r_mul_stmt_ = x2r_mul_stmt_.append(a_load);
        x2r_mul_stmt_ = x2r_mul_stmt_.append(b_load);
    }

    void build_mul() {
        auto &fma = ctx_.plan().fma;
        auto &a_layout = fma.a_layout;
        auto &b_layout = fma.b_layout;
        auto &c_layout = fma.c_layout;
        auto a_buf = buf_mgr_.get("a");
        auto b_buf = buf_mgr_.get("b");
        auto c_buf = buf_mgr_.get("c", c_layout.size());

        // BMNK order.
        prb_dim_t dims[4];
        int blocks[4] = {1, 1, 1, 1};
        int sizes[4] = {1, 1, 1, 1};
        for (auto &d : fma.inst_tile) {
            int idx = ctx_.to_bmnk_idx(d);
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
        int a_stride = is_a_bcast ? 0 : a_layout.inner_stride();
        int b_stride = is_b_bcast ? 0 : b_layout.inner_stride();
        auto mad = mad_t::make(ctx_.hw(), c_layout.type(), fma.simd,
                a_layout.type(), a_stride, b_layout.type(), b_stride);
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
                        stmt = stmt.append(mad.call({c_buf[c_off], c_buf[c_off],
                                a_buf[a_off], b_buf[b_off]}));
                    }
                }
            }
        }
        x2r_mul_stmt_ = x2r_mul_stmt_.append(stmt);
    }

    void build_x2r_mul() {
        build_x2r();
        build_mul();
    }

    void build_c_store() {
        auto &fma = ctx_.plan().fma;
        auto &epilogue = ctx_.plan().epilogue;
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
            auto stmt = create_stmt(store, ctx_.c_mem_buf(), payload_buf,
                    off_ctx_, coord, epilogue.tile, payload_layout,
                    payload_coord);
            c_store_stmt_ = c_store_stmt_.append(stmt);
        });
    }

    build_ctx_t &ctx_;
    buffer_manager_t buf_mgr_;
    offset_ctx_t off_ctx_;

    stmt_t x2r_mul_stmt_;
    stmt_t c_store_stmt_;
};

expr_t unpack(const grid_t &grid, const prb_dim_t &dim, build_ctx_t &ctx) {
    auto base_idx = ctx.to_ir(grid.index_var(dim));
    if (base_idx.is_empty()) return expr_t();

    expr_t value = base_idx;
    auto &dims = grid.dims(grid.index(dim));
    int ndims = (int)dims.size();
    for (int i = 0; i < ndims; i++) {
        if (dims[i] == dim) break;
        auto i_dim_size
                = ctx.kernel_info().find_arg(dims[i].str() + "_grid_size");
        auto i_magic = ctx.kernel_info().find_arg(dims[i].str() + "_magic");
        value = ternary_op_t::make(
                op_kind_t::_idiv, value, i_dim_size, i_magic);
    }
    auto dim_size = ctx.kernel_info().find_arg(dim.str() + "_grid_size");
    auto magic = ctx.kernel_info().find_arg(dim.str() + "_magic");
    value = ternary_op_t::make(op_kind_t::_imod, value, dim_size, magic);
    return value;
}

stmt_t inject_index_let_stmts(const stmt_t &body, build_ctx_t &ctx) {
    auto &tg_grid = ctx.plan().tg_grid;
    auto &coord_info = ctx.plan().coord_info;
    stmt_t ret = body;
    for (auto &d : conv_index_dims(ctx.plan().desc.prop)) {
        auto tg_idx = ctx.to_ir(coord_info.tg_index(d));
        if (is_const(tg_idx)) continue;
        auto base_tg_idx = ctx.to_ir(tg_grid.index_var(d));
        if (base_tg_idx.is_empty()) continue;
        auto value = unpack(tg_grid, d, ctx);
        ret = let_t::make(tg_idx, value, ret);
    }
    return ret;
}

stmt_t inject_global_alloc_stmts(const stmt_t &stmt, build_ctx_t &ctx) {
    std::vector<stmt_t> allocs;
    for (int i = 0; i < ctx.kernel_info().nargs(); i++) {
        auto &var = ctx.kernel_info().arg_var(i);
        if (!var.type().is_ptr()) continue;
        allocs.push_back(alloc_t::make(var, 0, alloc_kind_t::global));
    }
    auto ret = inject_alloc_stmts(stmt, allocs);
    return ret;
}

void ir_builder_t::build() {
    plan_t plan = create_conv_plan(desc_);
    if (!plan) ir_except_not_implemented("Cannot create plan.");

    ir_info() << desc_ << std::endl;
    ir_trace() << plan << std::endl;

    constraint_set_t cset;
    ir_context_t ir_ctx(desc_.exec_cfg(), cset);

    build_ctx_t ctx(plan, kernel_info_, grid_ctx_, ir_ctx);

    compute_builder_t cb(ctx);
    cb.build();

    stmt_t loop_stmt;
    loop_stmt = cb.loop_nest();
    loop_stmt = cb.inject_compute_alloc_stmts(loop_stmt);

    stmt_ = loop_stmt;
    stmt_ = cb.init_stmt(stmt_);
    stmt_ = cb.zero_out_stmt().append(stmt_);
    stmt_ = stmt_.append(cb.c_store_stmt());
    stmt_ = cb.inject_out_alloc_stmts(stmt_);
    stmt_ = cb.inject_header_alloc_stmts(stmt_);
    stmt_ = inject_let_stmts(stmt_, ctx.init_let_stmts());
    stmt_ = inject_global_alloc_stmts(stmt_, ctx);
    stmt_ = inject_index_let_stmts(stmt_, ctx);

    stmt_ = inject_external_var_let(stmt_, ir_ctx);
    stmt_ = simplify(stmt_, ir_ctx);
    stmt_ = optimize_alloc_let(stmt_, ir_ctx);
    stmt_ = split_wide_stores(stmt_, ir_ctx);
    stmt_ = eliminate_common_subexprs(stmt_, ir_ctx, 16, 0);
    stmt_ = inject_bank_conflict_attribute(stmt_, ir_ctx);

    ir_trace() << "Convolution kernel body:\n" << stmt_ << std::endl;
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
