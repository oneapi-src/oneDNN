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

#include "gpu/intel/jit/v2/conv/builder.hpp"

#include <sstream>

#include "gpu/intel/jit/ir/ir.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/jit/ir/message.hpp"
#include "gpu/intel/jit/pass/dpas.hpp"
#include "gpu/intel/jit/pass/pass.hpp"
#include "gpu/intel/jit/utils/trace.hpp"
#include "gpu/intel/jit/v2/conv/bridge.hpp"
#include "gpu/intel/jit/v2/conv/plan.hpp"
#include "gpu/intel/jit/v2/ir/builder.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

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
        auto arg = kernel_info_.find_arg(name, /*allow_empty=*/!is_const_var);
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

class buffer_info_t {
public:
    buffer_info_t(buffer_manager_t &buf_mgr, const kernel_desc_t &desc,
            const kernel_info_t &kernel_info, const x2r_fma_plan_t &plan) {
        for (auto &s : plan.stages) {
            if (!s.is_x2r()) continue;
            auto kind = s.x2r.tensor_kind;
            auto name = pick_abc(kind, desc.prop, "src", "wei", "dst");
            if (entries_.count(name) > 0) continue;
            auto &e = entries_[name];
            e.mem_buf = kernel_info.find_arg(name);
            if (s.x2r.reorder) {
                e.reg_buf = buf_mgr.get(
                        to_string(kind), s.x2r.reorder.dst.size());
            } else {
                e.reg_buf = buf_mgr.get(
                        to_string(kind), s.x2r.load.reg_layout().size());
            }
        }
        auto c_name = pick_c(desc.prop, "src", "wei", "dst");
        auto &c_e = entries_[c_name];
        c_e.mem_buf = kernel_info.find_arg(c_name);
        c_e.reg_buf = buf_mgr.get("c", plan.c_layout.size());
        for (auto &abc :
                {tensor_kind_t::a, tensor_kind_t::b, tensor_kind_t::c}) {
            auto name = pick_abc(abc, desc.prop, "src", "wei", "dst");
            entries_[to_string(abc)] = entries_.at(name);
        }

        if (desc.with_bias_fwd() || desc.with_bias_bwd_w()) {
            auto &e = entries_["bia"];
            e.mem_buf = kernel_info.find_arg("bia");
            if (!plan.bia_layout.is_empty())
                e.reg_buf = buf_mgr.get("bia_reduced", plan.bia_layout.size());
        }

        for (size_t i = 0; i < desc.post_ops.len(); i++) {
            auto &po = desc.post_ops[i];
            if (po.is_binary()) {
                std::string name = "binary_" + std::to_string(i);
                auto &e = entries_[name];
                e.mem_buf = kernel_info.find_arg(name);
            }
        }
    }

    const expr_t mem_buf(const std::string &name) const {
        if (entries_.count(name) == 0) return expr_t();
        auto &e = entries_.at(name);
        return e.mem_buf;
    }

    const expr_t reg_buf(const std::string &name) const {
        if (entries_.count(name) == 0) return expr_t();
        auto &e = entries_.at(name);
        return e.reg_buf;
    }

private:
    struct buf_entry_t {
        expr_t mem_buf;
        expr_t reg_buf;
    };

    std::unordered_map<std::string, buf_entry_t> entries_;
};

class x2r_mul_builder_t : public ir_builder_t {
public:
    x2r_mul_builder_t(ir_builder_t &parent, const loop_nest_t &loop_nest,
            const buffer_info_t &buf_info, const kernel_desc_t &desc,
            const x2r_fma_plan_t &plan)
        : ir_builder_t(parent, loop_nest)
        , buf_info_(buf_info)
        , desc_(desc)
        , plan_(plan) {
        for (auto &s : plan_.stages) {
            if (s.is_fma()) {
                build_mul(s.fma);
            } else if (s.is_x2r()) {
                build_x2r(s.x2r);
                if (s.x2r.tensor_kind == tensor_kind_t::b) {
                    uint32_t mask = (1 << 1) | (1 << 2);
                    auto &b_buf = buf_info_.reg_buf("b");
                    if (!s.x2r.bia_layout.is_empty()) {
                        reduce(s.x2r.layout, s.x2r.bia_layout, b_buf,
                                buf_info_.reg_buf("bia"), mask);
                    }
                }
            }
        }
    }

private:
    void build_x2r(const x2r_plan_t &plan) {
        auto &mem_buf = buf_info_.mem_buf(to_string(plan.tensor_kind));
        auto &reg_buf = buf_info_.reg_buf(to_string(plan.tensor_kind));
        auto load_buf
                = load(plan.load, mem_buf, (plan.reorder ? expr_t() : reg_buf));
        if (plan.reorder) reorder(plan.reorder, load_buf, reg_buf);
    }

    void build_mul(const fma_plan_t &fma) {
        auto &a_layout = fma.a_layout;
        auto &b_layout = fma.b_layout;
        auto &c_layout = fma.c_layout;
        auto &a_buf = buf_info_.reg_buf("a");
        auto &b_buf = buf_info_.reg_buf("b");
        auto &c_buf = buf_info_.reg_buf("c");

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
        stmt_t call_stmt;
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
                        call_stmt = call_stmt.append(fma_func.call(
                                {dst, dst, std::move(src1), std::move(src2)}));
                    }
                }
            }
        }
        if (fma.fma == fma_kind_t::dpas) {
            call_stmt
                    = inject_dpas_atomic(call_stmt, /*filter_by_label=*/false);
        }
        emit(call_stmt);
    }

    const buffer_info_t &buf_info_;
    const kernel_desc_t &desc_;
    const x2r_fma_plan_t &plan_;
};

class prefetch_builder_t : public ir_builder_t {
public:
    prefetch_builder_t(ir_builder_t &parent, const loop_nest_t &loop_nest,
            const buffer_info_t &buf_info, const prefetch_plan_t &plan)
        : ir_builder_t(parent, loop_nest), buf_info_(buf_info), plan_(plan) {
        if (plan_.a_prefetch) {
            load(plan_.a_prefetch, buf_info_.mem_buf("a"), expr_t());
        }
        if (plan_.b_prefetch) {
            load(plan_.b_prefetch, buf_info_.mem_buf("b"), expr_t());
        }
    }

private:
    const buffer_info_t &buf_info_;
    const prefetch_plan_t &plan_;
};

class post_op_builder_t : public ir_builder_t {
public:
    post_op_builder_t(ir_builder_t &parent, const kernel_desc_t &desc,
            const pvar_coord_t<expr_t> &coord, const pvar_tile_t &tile,
            alg_kind_t binary_alg, const gpu_post_ops_t::entry_t *post_op_entry,
            const layout_t &lhs_reg_layout, const expr_t &lhs_reg_buf,
            const expr_t &rhs_mem_buf, const type_t &_rhs_type,
            uint16_t rhs_mask, float rhs_scale, int rhs_zero_point)
        : ir_builder_t(parent, loop_nest_t()), desc_(desc) {
        // Binary post-op.
        if (!rhs_mem_buf.is_empty()) {
            auto &c_tag = pick_c(
                    desc_.prop, desc_.src_tag, desc_.wei_tag, desc_.dst_tag);
            auto rhs_type = _rhs_type.is_undef() ? c_tag.type() : _rhs_type;
            auto rhs_view = rhs_mem_view(coord, tile, rhs_type, rhs_mask);
            layout_t rhs_reg_layout;
            auto rhs_reg_buf
                    = load(rhs_view, rhs_mem_buf, expr_t(), &rhs_reg_layout);
            build_binary_post_op(binary_alg, lhs_reg_layout, rhs_reg_layout,
                    lhs_reg_buf, rhs_reg_buf, rhs_scale, rhs_zero_point);
            return;
        }
        // Eltwise post-op.
        auto &e = post_op_entry->as_eltwise();
        auto func = eltwise_t::make(e.alg, e.scale, e.alpha, e.beta);
        emit(func.call({expr_t(lhs_reg_layout.elems()), lhs_reg_buf}));
    }

private:
    view_t rhs_mem_view(const pvar_coord_t<expr_t> &_coord,
            const pvar_tile_t &_tile, const type_t &type, uint16_t mask) {
        dim_mapper_manager_t mger(desc_.prop, desc_.reqs);
        auto &c_mapper = mger.mapper(tensor_kind_t::c);
        auto kind = pick_c(desc_.prop, tensor_kind_t::src, tensor_kind_t::wei,
                tensor_kind_t::dst);
        auto &c_tag = pick_c(
                desc_.prop, desc_.src_tag, desc_.wei_tag, desc_.dst_tag);
        auto rhs_layout
                = make_conv_layout(kind, c_tag, desc_.is_dw, desc_.reqs, mask);
        rhs_layout = rhs_layout.retype(type);
        auto is_bcast = [&](const pvar_t &dim) {
            for (auto &b : rhs_layout.blocks()) {
                if (b.dim == dim) return false;
            }
            return true;
        };
        auto coord = _coord;
        auto tile = _tile;
        for (auto &d : rhs_layout.desc().letter_map()) {
            if (is_bcast(d)) {
                if (tile.has(d)) tile.unset(d);
                if (coord.has(d)) coord.unset(d);
            }
        }
        return view_t(c_mapper, rhs_layout, coord, tile);
    }

    void build_binary_post_op(alg_kind_t alg, const layout_t &lhs,
            const layout_t &_rhs, const expr_t &lhs_buf, const expr_t &_rhs_buf,
            float scale = 1, int zero_point = 0) {
        ir_assert(lhs.type() == type_t::f32());
        auto rhs = _rhs;
        auto rhs_buf = _rhs_buf;
        if (rhs.type() != type_t::f32()) {
            auto rhs_f32 = _rhs.retype(type_t::f32(), /*dense=*/true);
            rhs_buf = reorder(_rhs, rhs_f32, _rhs_buf);
            rhs = rhs_f32;
        }
        if (scale != 1 || zero_point != 0) {
            auto func = eltwise_t::make(
                    alg_kind::eltwise_linear, 1, scale, -zero_point);
            emit(func.call({expr_t(rhs.elems()), rhs_buf}));
        }
        ir_assert(lhs.nblocks() > 0);
        int max_simd = (2 * desc_.hw.grf_size()) / sizeof(float);
        auto &lhs0 = lhs.blocks()[0];
        int elems = math::gcd(max_simd, lhs0.int_size());
        bool is_bcast = !rhs.dim_sizes().has(lhs0.dim);
        if (!is_bcast) {
            auto &rhs0 = rhs.blocks()[0];
            if (rhs0.dim == lhs0.dim) {
                elems = math::gcd(elems, rhs0.int_size());
            } else {
                elems = 1;
            }
        }
        elems = (elems < 8 ? 1 : elems);
        pvar_tile_t tile;
        tile[lhs0.dim] = elems;
        for_each(lhs.int_dim_sizes(), tile,
                [&](const pvar_coord_t<dim_t> &coord) {
                    auto lhs_off = lhs.offset_in_bytes(coord);
                    auto rhs_off = rhs.offset_in_bytes(coord);
                    auto e_l = load_t::make(
                            type_t::f32().with_elems(elems), lhs_buf, lhs_off);
                    auto e_r = load_t::make(
                            type_t::f32().with_elems(is_bcast ? 1 : elems),
                            rhs_buf, rhs_off);
                    if (is_bcast) e_r = shuffle_t::make_broadcast(e_r, elems);
                    auto e_op = binary_op_t::make(
                            alg_kind_to_op_kind(alg), e_l, e_r);
                    emit(store_t::make(lhs_buf, lhs_off, e_op));
                });
    }

    const kernel_desc_t &desc_;
};

class epilogue_builder_t : public ir_builder_t {
public:
    epilogue_builder_t(ir_builder_t &parent, const buffer_info_t &buf_info,
            const kernel_desc_t &desc, const epilogue_plan_t &plan)
        : ir_builder_t(parent, loop_nest_t())
        , buf_info_(buf_info)
        , desc_(desc)
        , plan_(plan) {
        build_slm_reduce();
        build_c_store();
        build_bias_reduce_store();
    }

private:
    void build_slm_reduce() {
        auto &slm_reduce = plan_.slm_reduce;
        if (!slm_reduce) return;

        auto &c_buf = buf_info_.reg_buf("c");
        auto c_tmp_buf = alloc("c_reduce", slm_reduce.load.reg_layout().size());
        auto c_slm_buf = alloc("slm", slm_reduce.slm_usage_bytes());
        store(slm_reduce.store, c_slm_buf, c_buf);
        barrier();
        load(slm_reduce.load, c_slm_buf, c_tmp_buf);
        zero_out(c_buf);
        reduce(slm_reduce.reduce, c_tmp_buf, c_buf);
    }

    void build_c_store() {
        auto &c_layout = plan_.c_reg_layout;
        auto &c_mem_buf = buf_info_.mem_buf("c");
        auto &c_reg_buf = buf_info_.reg_buf("c");
        for_each(c_layout.int_dim_sizes(), plan_.tile,
                [&](const pvar_coord_t<dim_t> &coord) {
                    dim_t off = c_layout.offset_in_bytes(coord);
                    auto store_layout
                            = plan_.c_store.reg_layout().map(plan_.tile);
                    layout_t payload_layout = store_layout;
                    auto payload_buf = build_post_ops(c_layout.map(plan_.tile),
                            plan_.c_coord + coord, c_reg_buf + off,
                            payload_layout);
                    payload_buf = reorder(
                            payload_layout, store_layout, payload_buf);
                    store(plan_.c_store, c_mem_buf, payload_buf, coord,
                            plan_.tile);
                });
    }

    static uint16_t reverse_post_op_mask(uint16_t mask, int ndims) {
        uint16_t ret = 0;
        for (int i = 0; i < ndims; i++) {
            uint16_t bit = (mask >> (ndims - i - 1)) & 0x1;
            ret |= bit << i;
        }
        return ret;
    }

    void build_post_op(const pvar_coord_t<expr_t> &coord,
            const pvar_tile_t &tile, alg_kind_t binary_alg,
            const gpu_post_ops_t::entry_t *post_op_entry,
            const layout_t &lhs_reg_layout, const expr_t &lhs_reg_buf,
            const expr_t &rhs_mem_buf = expr_t(),
            const type_t &rhs_type = type_t::undef(), uint16_t rhs_mask = 0,
            float rhs_scale = 1.0f, int rhs_zero_point = 0) {
        post_op_builder_t builder(*this, desc_, coord, tile, binary_alg,
                post_op_entry, lhs_reg_layout, lhs_reg_buf, rhs_mem_buf,
                rhs_type, rhs_mask, rhs_scale, rhs_zero_point);
        emit(builder.get_init_stmt());
        emit(builder.get_stmt());
    }

    expr_t build_post_ops(const layout_t &layout,
            const pvar_coord_t<expr_t> &coord, const expr_t &_buf,
            layout_t &out_layout) {
        if (desc_.post_ops.len() == 0 && !desc_.with_bias_fwd()) {
            out_layout = layout;
            return _buf;
        }
        auto f32_layout = out_layout.retype(type_t::f32(), /*dense=*/true);
        auto tile = f32_layout.int_dim_sizes();
        int elems = f32_layout.elems();
        ir_assert(elems * type_t::f32().size() == f32_layout.size());
        auto buf = reorder(layout, f32_layout, _buf);
        arg_helper_t arg_helper(desc_);
        auto &c_tag = pick_c(
                desc_.prop, desc_.src_tag, desc_.wei_tag, desc_.dst_tag);
        if (desc_.with_bias_fwd()) {
            build_post_op(coord, tile, alg_kind::binary_add, nullptr,
                    f32_layout, buf, buf_info_.mem_buf("bia"), desc_.bias_type,
                    /*rhs_mask=*/0x2);
        }
        for (size_t i = 0; i < desc_.post_ops.len(); i++) {
            auto &po = desc_.post_ops[i];
            if (po.is_eltwise()) {
                build_post_op(
                        coord, tile, alg_kind::undef, &po, f32_layout, buf);
            } else if (po.is_sum()) {
                auto &s = po.as_sum();
                build_post_op(coord, tile, alg_kind::binary_add, &po,
                        f32_layout, buf, buf_info_.mem_buf("c"), type_t(s.dt),
                        0xFFFF, s.scale, s.zero_point);
            } else if (po.is_binary()) {
                auto &b = po.as_binary();
                auto rhs_buf_name = arg_helper.post_op_name(i);
                auto mask = reverse_post_op_mask(
                        b.src1_desc.broadcast_mask ^ 0xFFFF,
                        c_tag.raw_tag().ndims());
                build_post_op(coord, tile, b.alg, &po, f32_layout, buf,
                        buf_info_.mem_buf(rhs_buf_name), type_t(b.src1_desc.dt),
                        mask);
            } else {
                ir_error_not_expected();
            }
        }
        out_layout = f32_layout;
        return buf;
    }

    void build_bias_reduce_store() {
        if (plan_.bia_reduced_reg_layout.is_empty()) return;
        auto &bia_red_mem_buf = buf_info_.mem_buf("bia");
        auto &bia_red_reg_buf = buf_info_.reg_buf("bia");
        expr_t tmp_buf;
        if (plan_.bia_reorder)
            tmp_buf = alloc("bia_tmp", plan_.bia_reorder.dst.size());
        auto bia_tile = plan_.bia_store.reg_layout().int_dim_sizes();
        auto epilogue_tile = bia_tile;
        for (auto &d : bia_tile)
            epilogue_tile[d] = plan_.tile[d];
        for_each(
                bia_tile, epilogue_tile, [&](const pvar_coord_t<dim_t> &coord) {
                    auto payload_buf = bia_red_reg_buf;
                    if (plan_.bia_reorder) {
                        dim_t src_off
                                = plan_.bia_reduced_reg_layout.offset_in_bytes(
                                        coord);
                        reorder(plan_.bia_reorder, bia_red_reg_buf + src_off,
                                tmp_buf);
                        payload_buf = std::move(tmp_buf);
                    }
                    _if(plan_.reduce_cond, [&]() {
                        store(plan_.bia_store, bia_red_mem_buf, payload_buf,
                                coord, epilogue_tile);
                    });
                });
    }

    const buffer_info_t &buf_info_;
    const kernel_desc_t &desc_;
    const epilogue_plan_t &plan_;
};

class conv_builder_t : public ir_builder_t {
public:
    conv_builder_t(ir_context_t &ir_ctx, const kernel_desc_t &desc,
            const kernel_info_t &kernel_info, const grid_context_t &grid_ctx,
            const plan_t &plan)
        : ir_builder_t(kernel_info, ir_ctx)
        , desc_(desc)
        , grid_ctx_(grid_ctx)
        , plan_(plan)
        , buf_info_(buf_mgr(), desc, kernel_info, plan.x2r_fma)
        , loop_nest_(make_loop_nest(desc_.loop_desc, plan_.coord_info)) {
        loop();
        epilogue();

        auto _stmt = get_stmt();
        _stmt = inject_alloc_stmts(_stmt, buf_mgr());
        _stmt = off_scope().inject_let_stmts(_stmt);
        _stmt = inject_global_alloc(_stmt);
        _stmt = inject_index_let(_stmt);
        _stmt = fixup_idiv(_stmt, kernel_info, ir_ctx);
        _stmt = finalize_vars(
                _stmt, kernel_info, grid_ctx_, plan_.tg_grid, ir_ctx);

        _stmt = simplify(_stmt, ir_ctx);
        _stmt = optimize_alloc_let(_stmt, ir_ctx);
        _stmt = split_wide_stores(_stmt, ir_ctx);
        _stmt = fixup_if_conditions(_stmt, ir_ctx);
        _stmt = eliminate_common_subexprs(_stmt, ir_ctx, 16, 0);
        _stmt = inject_bank_conflict_attribute(_stmt, ir_ctx);
        set_stmt(_stmt);
    }

private:
    void loop() {
        prefetch_builder_t prefetch_builder(
                *this, loop_nest_, buf_info_, plan_.prefetch);
        x2r_mul_builder_t x2r_mul_builder(
                *this, loop_nest_, buf_info_, desc_, plan_.x2r_fma);

        zero_out(buf_info_.reg_buf("c"));
        if (!buf_info_.reg_buf("bia").is_empty())
            zero_out(buf_info_.reg_buf("bia"));

        emit(x2r_mul_builder.get_init_stmt());
        emit(prefetch_builder.get_init_stmt());
        auto &coord_info = plan_.coord_info;
        int prefetch_dist = desc_.prefetch.dist;
        auto x2r_mul_stmt = x2r_mul_builder.get_stmt();
        auto prefetch_stmt = prefetch_builder.get_stmt();
        iterator_t prefetch_it;
        if (prefetch_dist > 0) {
            prefetch_it = iterator_t(buf_mgr(), loop_nest_);
            emit(prefetch_it.init_stmt());
            for (int i = 0; i < prefetch_dist; i++) {
                auto i_prefetch_stmt = prefetch_stmt;
                if (i > 0) {
                    i_prefetch_stmt
                            = prefetch_it.check_bounds_stmt(i_prefetch_stmt);
                }
                emit(i_prefetch_stmt);
                emit(prefetch_it.inc_stmt(prefetch_builder.off_ctx()));
            }
        }
        std::function<void(size_t)> emit_loop;
        emit_loop = [&](size_t i) {
            if (i == 0) {
                // Innermost loop body.
                if (prefetch_dist > 0) {
                    emit(prefetch_it.check_bounds_stmt(prefetch_stmt));
                }
                emit(x2r_mul_stmt);
                if (prefetch_dist > 0) {
                    emit(prefetch_it.inc_stmt(prefetch_builder.off_ctx()));
                }
                return;
            };
            auto &loop = loop_nest_[i - 1];
            const auto &var = coord_info.loop_index(loop.dim);
            const auto &bound = loop.size;
            _for(var, 0, bound, [&]() {
                emit_loop(i - 1);
                emit(x2r_mul_builder.off_ctx().inc_loop_stmt((int)loop.idx));
            });
        };
        emit_loop(loop_nest_.nloops());
    }

    void epilogue() {
        epilogue_builder_t builder(*this, buf_info_, desc_, plan_.epilogue);
        emit(builder.get_init_stmt());
        emit(builder.get_stmt());
    }

    stmt_t inject_global_alloc(const stmt_t &stmt) const {
        std::vector<stmt_t> allocs;
        for (int i = 0; i < kernel_info().nargs(); i++) {
            auto &var = kernel_info().arg_var(i);
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
                    = kernel_info().find_arg(dims[i].str() + "_grid_size");
            auto i_magic = kernel_info().find_arg(dims[i].str() + "_magic");
            value = ternary_op_t::make(
                    op_kind_t::_idiv, value, i_dim_size, i_magic);
        }
        auto dim_size = kernel_info().find_arg(dim.str() + "_grid_size");
        auto magic = kernel_info().find_arg(dim.str() + "_magic");
        value = ternary_op_t::make(op_kind_t::_imod, value, dim_size, magic);
        return value;
    }

    kernel_desc_t desc_;
    grid_context_t grid_ctx_;
    plan_t plan_;
    buffer_info_t buf_info_;
    loop_nest_t loop_nest_;
};

stmt_t build_ir(const kernel_desc_t &desc, const kernel_info_t &kernel_info,
        const grid_context_t &grid_ctx) {
    auto plan = create_conv_plan(desc);
    if (!plan) ir_except_not_implemented("Cannot create plan.");

    ir_info() << desc << std::endl;
    ir_trace() << plan << std::endl;

    constraint_set_t cset;
    ir_context_t ir_ctx(desc.exec_cfg(), cset);
    conv_builder_t builder(ir_ctx, desc, kernel_info, grid_ctx, plan);
    auto stmt = builder.get_stmt();
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
