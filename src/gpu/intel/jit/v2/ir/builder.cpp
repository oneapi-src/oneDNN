/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "gpu/intel/jit/v2/ir/builder.hpp"
#include "gpu/intel/jit/ir/message.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {

send_header_t offset_scope_t::add_header(int version,
        const send_2d_desc_t &desc, const expr_t &mem_buf, const expr_t &base,
        const expr_t &x_base, const expr_t &y_base, const expr_t &x_inc,
        const expr_t &y_inc, const loop_nest_t &loop_nest) {
    auto base0 = cast(mem_buf, type_t::u64());
    auto params = offset_params_t(type_t::u64(), /*esize=*/1, "h");
    params.buf_align = desc.hw.grf_size();
    auto off = get_offset(version, base0, base, expr_t(0), params, loop_nest);
    auto x_params = offset_params_t(type_t::s32());
    auto y_params = offset_params_t(type_t::s32());
    x_params.buf = off.buf + send_t::header_2d_off_x();
    y_params.buf = off.buf + send_t::header_2d_off_y();
    auto x = get_offset(version, expr_t(0), x_base, x_inc, x_params, loop_nest);
    auto y = get_offset(version, expr_t(0), y_base, y_inc, y_params, loop_nest);

    int type_size = desc.type.size();
    auto W_enc = to_simple_expr(desc.W) * type_size - 1;
    auto H_enc = to_simple_expr(desc.H) - 1;
    auto P_enc = to_simple_expr(desc.P) * type_size - 1;
    (void)get_offset(version, W_enc,
            off.buf + send_t::header_2d_off_surface_width(), loop_nest);
    (void)get_offset(version, H_enc,
            off.buf + send_t::header_2d_off_surface_height(), loop_nest);
    (void)get_offset(version, P_enc,
            off.buf + send_t::header_2d_off_surface_pitch(), loop_nest);

    uint32_t w_enc = desc.w - 1;
    uint32_t h_enc = desc.h - 1;
    uint32_t count_enc = desc.c - 1;
    uint32_t whc_value = (count_enc << 16) + (h_enc << 8) + w_enc;
    (void)get_offset(version, whc_value, off.buf + send_t::header_2d_off_whc(),
            loop_nest);

    return send_header_t(off);
}

send_mask_t offset_scope_t::add_mask(int version, const mask_t &mask,
        const std::vector<expr_t> &mask_incs, const loop_nest_t &loop_nest) {
    send_mask_t ret;
    for (int i = 0; i < mask.nmasks(); i++) {
        auto &dm = mask.dim_masks[i];
        if (dm.is_empty()) continue;
        auto shift = mask_incs.empty() ? expr_t(0) : mask_incs[i];
        auto params = offset_params_t(type_t::s32(), dm.slots(), "m");
        params.allow_bcast = true;
        params.allow_reuse = true;
        auto off = get_offset(version, expr_t(0), dm.base, dm.slot_incs, shift,
                params, loop_nest);
        ret.add_mask(off, to_simple_expr(dm.bound), dm.has_underflow);
    }
    return ret;
}

offset_t offset_scope_t::get_offset(int version, const expr_t &base0,
        const expr_t &base, const std::vector<expr_t> &_shift_vec,
        const expr_t &_shift, const offset_params_t &_params,
        const loop_nest_t &loop_nest) {
    auto params = _params;
    expr_t _base_init;
    std::vector<expr_t> _loop_incs;
    split_to_linear(base, loop_nest.indices(), loop_nest.init_exprs(),
            _base_init, _loop_incs);

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
    for (size_t i = 0; i < loop_nest.nloops(); i++) {
        auto inc_value = simplify(_loop_incs[i] - comp_value);
        auto inc = to_simple_expr(inc_value);
        ret.loop_incs.push_back(inc);
        if (i == loop_nest.nloops() - 1) break;
        comp_value = to_simple_expr(_loop_incs[i] * loop_nest[i].bound);
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
        if (params.buf_align != 0) size = utils::rnd_up(size, params.buf_align);
        ret.buf = params.get_buffer(buf_mgr_, size);
        buf_versions_.emplace(ret.buf, version);
    }

    // Try to use inline initialization.
    if (params.allow_inline_init && !is_zero(ret.shift)
            && ret.type.is_scalar()) {
        for (auto &o : offsets_) {
            if (o.is_equal(ret, /*compare_shift=*/false)) {
                gpu_assert(o.type.is_scalar());
                ret.inline_init = ret.store(o.load() + ret.shift);
                ret.loop_incs.clear();
                break;
            }
        }
    }

    return add_offset(ret);
}

type_t to_send_type(const send_1d_desc_t &desc) {
    if (desc.type_size <= 8) return type_t::u(desc.type_size * 8);
    return type_t::oword(desc.type_size / 16);
}

stmt_t create_stmt(const send_1d_plan_t &plan, const expr_t &mem_buf,
        const expr_t &reg_buf, offset_ctx_t &off_ctx,
        const pvar_coord_t<dim_t> &coord, const pvar_tile_t &tile) {
    for (auto &d : plan.entry_tile) {
        gpu_assert(tile.at(d) % plan.entry_tile.at(d) == 0);
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
        gpu_assert(e.coord == coord + sub_coord);
        auto header
                = off_ctx.add_header(plan.desc, mem_buf, plan.addr, e.addr_inc);
        auto mask = off_ctx.add_mask(plan.mask, e.mask_incs);
        auto call_reg_buf = reg_buf;
        if (!reg_buf.is_empty())
            call_reg_buf += plan.reg_layout.offset_in_bytes(sub_coord);
        auto call
                = send(mem_buf, header.to_expr(), call_reg_buf, mask.to_expr());
        ret = ret.append(header.off().inline_init);
        ret = ret.append(call);
    });
    return ret;
}

stmt_t create_stmt(const send_2d_plan_t &plan, const expr_t &mem_buf,
        const expr_t &reg_buf, offset_ctx_t &off_ctx,
        const pvar_coord_t<dim_t> &coord, const pvar_tile_t &tile) {
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
        gpu_assert(e.coord == coord + sub_coord);
        auto header = off_ctx.add_header(plan.desc, mem_buf, plan.base,
                plan.x_base, plan.y_base, e.x_inc, e.y_inc);
        auto mask = off_ctx.add_mask(plan.mask);
        auto call_reg_buf = reg_buf;
        if (!reg_buf.is_empty())
            call_reg_buf += plan.reg_layout.offset_in_bytes(sub_coord);
        auto call
                = send(mem_buf, header.to_expr(), call_reg_buf, mask.to_expr());
        ret = ret.append(header.off().inline_init);
        ret = ret.append(call);
    });
    return ret;
}

stmt_t create_stmt(const send_plan_t &plan, const expr_t &mem_buf,
        const expr_t &reg_buf, offset_ctx_t &off_ctx,
        const pvar_coord_t<dim_t> &coord, const pvar_tile_t &tile) {
    if (plan.is_1d())
        return create_stmt(plan._1d, mem_buf, reg_buf, off_ctx, coord, tile);
    if (plan.is_2d())
        return create_stmt(plan._2d, mem_buf, reg_buf, off_ctx, coord, tile);
    gpu_error_not_expected();
    return stmt_t();
}

} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
