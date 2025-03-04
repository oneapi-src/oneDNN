/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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
#include "gpu/intel/jit/ir/ir_builder.hpp"
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
            ret = ret.append(loop_idxs_[i].store(loop_nest_[i].init));
        }
        ret = linear_idx_.store(loop_nest_.linear_bound() - 1).append(ret);
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
                        loop_idxs_[i].var() >= loop_nest_[i].bound, body));
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
    idiv_fixup_mutator_t(var_manager_t &var_mgr) : var_mgr_(var_mgr) {}

    object_t _mutate(const binary_op_t &_obj) override {
        auto new_obj = ir_mutator_t::_mutate(_obj);
        auto &obj = new_obj.as<binary_op_t>();
        bool is_var_idivmod
                = utils::one_of(obj.op_kind, op_kind_t::_div, op_kind_t::_mod)
                && obj.type.is_int() && !is_const(obj.b);
        if (!is_var_idivmod) return new_obj;
        auto magic = var_mgr_.get_idiv_magic(obj.b);
        auto i_op = (obj.op_kind == op_kind_t::_div ? op_kind_t::_idiv
                                                    : op_kind_t::_imod);
        return ternary_op_t::make(
                i_op, obj.a, cast(obj.b, type_t::u32()), magic);
    }

private:
    var_manager_t &var_mgr_;
};

stmt_t fixup_idiv(
        const stmt_t &s, var_manager_t &var_mgr, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = idiv_fixup_mutator_t(var_mgr).mutate(s);
    trace_pass("fixup_idiv", ret, ir_ctx);
    return ret;
}

class var_replacer_t : public ir_mutator_t {
public:
    var_replacer_t(var_manager_t &var_mgr) : var_mgr_(var_mgr) {}
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
        expr_t arg;
        if (!is_const_var) {
            arg = var_mgr_.get_arg(name, /*allow_empty=*/true);
        } else {
            arg = var_mgr_.get_arg(var.type(), name);
        }
        auto value = (arg.is_empty() ? var : arg);
        var_map_.emplace(var, value);
        return value;
    }

    var_manager_t &var_mgr_;
    object_map_t<expr_t, expr_t> var_map_;
};

stmt_t finalize_vars(
        const stmt_t &stmt, var_manager_t &var_mgr, ir_context_t &ir_ctx) {
    auto ret = var_replacer_t(var_mgr).mutate(stmt);
    ret = inject_external_var_let(ret, ir_ctx);
    return ret;
}

// Stream-K parameters.
// For more details refer to https://arxiv.org/pdf/2301.03598.
struct stream_k_params_t {
    bool enable = false;

    // The following values are fixed for all threads.
    // Total number of threadgroup-level reductions. One reduction is one
    // thread-group wide iteration.
    expr_t total_iters;
    // Iterations per threadgroup (rounded up).
    expr_t iters_per_tg;
    // Iterations per one output tile, div_up(k, k_blk).
    expr_t iters_per_tile;

    // The following values are thread-specific.
    // Index of this threadgroup.
    expr_t tg_idx;
    // Index of the first threadgroup for the current output tile.
    expr_t tg_beg;
    // Index of the last threadgroup for the current output tile.
    expr_t tg_end;
    // Linear index of the current output tile.
    expr_t tile_idx;

    // Linear index of the first reduction iteration.
    expr_t local_beg;
    // Linear index of the last reduction iteration.
    expr_t local_end;
    // Variables holding initial loop indices.
    std::vector<expr_t> loop_inits;

    stream_k_params_t() = default;

    stream_k_params_t(bool enable, const loop_desc_t &loop_desc)
        : enable(enable) {
        if (!enable) return;
        local_beg = var_t::make(type_t::s32(), "local_beg");
        local_end = var_t::make(type_t::s32(), "local_end");
        for (auto &e : loop_desc) {
            loop_inits.push_back(
                    var_t::make(type_t::s32(), e.dim.str() + "_init"));
        }
    }

    operator bool() const { return enable; }
};

loop_nest_t make_loop_nest(const loop_desc_t &loop_desc,
        const coord_info_t &coord_info, const stream_k_params_t &sk_params) {
    loop_nest_t ret;
    expr_t linear_bound = 1;
    for (auto &e : loop_desc) {
        const auto &var = coord_info.loop_index(e.dim);
        const auto &size = coord_info.loop_size(e.dim);
        if (is_one(size)) continue;
        expr_t init = 0;
        if (sk_params) {
            init = sk_params.loop_inits[e.idx];
        } else {
            linear_bound *= size;
        }
        ret.add_loop(e.dim, var, init, size);
    }
    if (sk_params) linear_bound = sk_params.local_end - sk_params.local_beg;
    ret.set_linear_bound(linear_bound);
    return ret;
}

class buffer_info_t {
public:
    buffer_info_t(buffer_manager_t &buf_mgr, const kernel_desc_t &desc,
            const var_manager_t &var_mgr, const x2r_fma_plan_t &plan) {
        for (auto &s : plan.stages) {
            if (!s.is_x2r()) continue;
            auto kind = s.x2r.tensor_kind;
            auto name = pick_abc(kind, desc.prop, "src", "wei", "dst");
            if (entries_.count(name) > 0) continue;
            auto &e = entries_[name];
            e.mem_buf = var_mgr.get_arg(name);
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
        c_e.mem_buf = var_mgr.get_arg(c_name);
        c_e.reg_buf = buf_mgr.get("c", plan.c_layout.size());
        for (auto &abc :
                {tensor_kind_t::a, tensor_kind_t::b, tensor_kind_t::c}) {
            auto name = pick_abc(abc, desc.prop, "src", "wei", "dst");
            entries_[to_string(abc)] = entries_.at(name);
        }

        if (desc.with_bias_fwd() || desc.with_bias_bwd_w()) {
            auto &e = entries_["bias"];
            e.mem_buf = var_mgr.get_arg("bias");
            if (!plan.bias_layout.is_empty())
                e.reg_buf
                        = buf_mgr.get("bias_reduced", plan.bias_layout.size());
        }

        for (size_t i = 0; i < desc.post_ops.len(); i++) {
            auto &po = desc.post_ops[i];
            if (po.is_binary()) {
                std::string name = "binary_" + std::to_string(i);
                auto &e = entries_[name];
                e.mem_buf = var_mgr.get_arg(name);
            }
        }
    }

    expr_t mem_buf(const std::string &name) const {
        if (entries_.count(name) == 0) return expr_t();
        auto &e = entries_.at(name);
        return e.mem_buf;
    }

    expr_t reg_buf(const std::string &name) const {
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
                    const auto &b_buf = buf_info_.reg_buf("b");
                    if (!s.x2r.bias_layout.is_empty()) {
                        reduce(s.x2r.layout, s.x2r.bias_layout, b_buf,
                                buf_info_.reg_buf("bias"), mask);
                    }
                }
            }
        }
    }

private:
    void build_x2r(const x2r_plan_t &plan) {
        const auto &mem_buf = buf_info_.mem_buf(to_string(plan.tensor_kind));
        const auto &reg_buf = buf_info_.reg_buf(to_string(plan.tensor_kind));
        auto load_buf
                = load(plan.load, mem_buf, (plan.reorder ? expr_t() : reg_buf));
        if (plan.reorder) reorder(plan.reorder, load_buf, reg_buf);
    }

    void build_mul(const fma_plan_t &fma) {
        auto &a_layout = fma.a_layout;
        auto &b_layout = fma.b_layout;
        auto &c_layout = fma.c_layout;
        const auto &a_buf = buf_info_.reg_buf("a");
        const auto &b_buf = buf_info_.reg_buf("b");
        const auto &c_buf = buf_info_.reg_buf("c");

        for (auto &d : a_layout.dims())
            gpu_assert(fma.inst_tile.has(d)) << d;
        for (auto &d : b_layout.dims())
            gpu_assert(fma.inst_tile.has(d)) << d;

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
            default: gpu_error_not_expected();
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
        dim_mapper_manager_t mger(desc_.prop, desc_.spec.reqs());
        auto &c_mapper = mger.mapper(tensor_kind_t::c);
        auto kind = pick_c(desc_.prop, tensor_kind_t::src, tensor_kind_t::wei,
                tensor_kind_t::dst);
        auto &c_tag = pick_c(
                desc_.prop, desc_.src_tag, desc_.wei_tag, desc_.dst_tag);
        auto rhs_layout = make_conv_layout(
                kind, c_tag, desc_.is_dw, desc_.spec.reqs(), mask);
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
        gpu_assert(lhs.type() == type_t::f32());
        auto rhs = _rhs;
        auto rhs_buf = _rhs_buf;
        if (rhs.type() != type_t::f32()) {
            auto rhs_f32 = _rhs.retype(type_t::f32(), /*dense=*/true);
            rhs_buf = reorder(_rhs, rhs_f32, _rhs_buf);
            rhs = rhs_f32;
        }
        if (zero_point != 0) {
            auto func = eltwise_t::make(
                    alg_kind::eltwise_linear, 1, 1.0f, -zero_point);
            emit(func.call({expr_t(rhs.elems()), rhs_buf}));
        }
        if (scale != 1) {
            auto func
                    = eltwise_t::make(alg_kind::eltwise_linear, 1, scale, 0.0f);
            emit(func.call({expr_t(rhs.elems()), rhs_buf}));
        }
        gpu_assert(lhs.nblocks() > 0);
        int max_simd = (2 * desc_.hw_desc.grf_size()) / sizeof(float);
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
                    if (e_op.type().is_bool()) {
                        e_op = cast(e_op, lhs.type().with_elems(elems));
                    }
                    emit(store_t::make(lhs_buf, lhs_off, e_op));
                });
    }

    const kernel_desc_t &desc_;
};

class epilogue_tile_builder_t : public ir_builder_t {
public:
    epilogue_tile_builder_t(ir_builder_t &parent, const buffer_info_t &buf_info,
            const kernel_desc_t &desc, const layout_t &c_layout,
            const expr_t &c_mem_buf, const expr_t &c_reg_buf,
            const pvar_coord_t<expr_t> &c_coord,
            const pvar_coord_t<dim_t> &coord,
            const epilogue_store_plan_t &store_plan)
        : ir_builder_t(parent, loop_nest_t())
        , buf_info_(buf_info)
        , desc_(desc) {
        dim_t off = c_layout.offset_in_bytes(coord);
        auto store_layout
                = store_plan.c_store.reg_layout().map(store_plan.tile);
        layout_t payload_layout = store_layout;
        auto payload_buf = build_post_ops(c_layout.map(store_plan.tile),
                c_coord + coord, c_reg_buf + off, payload_layout);
        payload_buf = reorder(payload_layout, store_layout, payload_buf);
        store(store_plan.c_store, c_mem_buf, payload_buf, coord,
                store_plan.tile);
    }

private:
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
        gpu_assert(elems * type_t::f32().size() == f32_layout.size());
        auto buf = reorder(layout, f32_layout, _buf);
        arg_helper_t arg_helper(desc_);
        auto &c_tag = pick_c(
                desc_.prop, desc_.src_tag, desc_.wei_tag, desc_.dst_tag);
        if (desc_.with_bias_fwd()) {
            build_post_op(coord, tile, alg_kind::binary_add, nullptr,
                    f32_layout, buf, buf_info_.mem_buf("bias"), desc_.bias_type,
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
                gpu_error_not_expected();
            }
        }
        out_layout = f32_layout;
        return buf;
    }

    const buffer_info_t &buf_info_;
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

        const auto &c_buf = buf_info_.reg_buf("c");
        auto c_tmp_buf = alloc("c_reduce", slm_reduce.load.reg_layout().size());
        auto c_slm_buf = alloc("slm", slm_reduce.slm_usage_bytes());
        store(slm_reduce.store, c_slm_buf, c_buf);
        barrier();
        load(slm_reduce.load, c_slm_buf, c_tmp_buf);
        zero_out(c_buf);
        reduce(slm_reduce.reduce, c_tmp_buf, c_buf);
    }

    void build_c_store() {
        auto &store_plan = plan_.store;
        auto &c_layout = plan_.c_reg_layout;
        const auto &c_mem_buf = buf_info_.mem_buf("c");
        const auto &c_reg_buf = buf_info_.reg_buf("c");
        for_each(c_layout.int_dim_sizes(), store_plan.tile,
                [&](const pvar_coord_t<dim_t> &coord) {
                    epilogue_tile_builder_t builder(*this, buf_info_, desc_,
                            c_layout, c_mem_buf, c_reg_buf, plan_.c_coord,
                            coord, store_plan);
                    emit(builder.get_init_stmt());
                    emit(builder.get_stmt());
                });
    }

    void build_bias_reduce_store() {
        if (plan_.bias_layout.is_empty()) return;
        auto &store_plan = plan_.store;
        const auto &bias_red_mem_buf = buf_info_.mem_buf("bias");
        const auto &bias_red_reg_buf = buf_info_.reg_buf("bias");
        expr_t tmp_buf;
        if (store_plan.bias_reorder)
            tmp_buf = alloc("bias_tmp", store_plan.bias_reorder.dst.size());
        auto payload_buf = bias_red_reg_buf;
        if (store_plan.bias_reorder) {
            reorder(store_plan.bias_reorder, bias_red_reg_buf, tmp_buf);
            payload_buf = std::move(tmp_buf);
        }
        _if(plan_.bias_reduce_cond, [&]() {
            store(store_plan.bias_store, bias_red_mem_buf, payload_buf);
        });
    }

    const buffer_info_t &buf_info_;
    const kernel_desc_t &desc_;
    const epilogue_plan_t &plan_;
};

class conv_builder_t : public ir_builder_t {
public:
    conv_builder_t(ir_context_t &ir_ctx, const kernel_desc_t &desc,
            var_manager_t &var_mgr, const plan_t &plan)
        : ir_builder_t(ir_ctx)
        , desc_(desc)
        , var_mgr_(var_mgr)
        , plan_(plan)
        , buf_info_(buf_mgr(), desc, var_mgr, plan.x2r_fma) {

        stream_k_params_t sk_params(desc.use_stream_k, desc_.loop_desc);
        emit_thread_index_let();
        if (desc.use_stream_k) {
            sk_params.total_iters
                    = const_var_t::make(type_t::s32(), "sk_total_iters");
            sk_params.iters_per_tg
                    = const_var_t::make(type_t::s32(), "sk_iters_per_tg");
            sk_params.iters_per_tile
                    = const_var_t::make(type_t::s32(), "sk_iters_per_tile");
            sk_params.tg_idx = plan_.tg_grid.index_var(0);

            auto iter = alloc_var(type_t::s32(), "sk_iter");
            iter = sk_params.tg_idx * sk_params.iters_per_tg;
            auto iter_end = let("sk_iter_end",
                    min(sk_params.total_iters, iter + sk_params.iters_per_tg));

            _while(iter < iter_end, [&]() {
                sk_params.tile_idx
                        = let("sk_tile_idx", iter / sk_params.iters_per_tile);
                auto global_beg = let("sk_global_beg",
                        sk_params.tile_idx * sk_params.iters_per_tile);
                auto global_end = let(
                        "sk_global_end", global_beg + sk_params.iters_per_tile);
                let(sk_params.local_beg, iter - global_beg);
                let(sk_params.local_end,
                        min(iter_end, global_end) - global_beg);
                sk_params.tg_beg
                        = let("sk_tg_beg", global_beg / sk_params.iters_per_tg);
                sk_params.tg_end = let("sk_tg_beg",
                        (global_beg - 1) / sk_params.iters_per_tg + 1);
                emit_thread_group_index_let(sk_params.tile_idx);
                pipeline(sk_params);
                epilogue();
                iter = global_end;
            });
        } else {
            emit_thread_group_index_let();
            pipeline();
            epilogue();
        }

        auto _stmt = get_stmt();
        _stmt = inject_alloc_stmts(_stmt, buf_mgr());
        _stmt = inject_dangling_let_stmts(_stmt);
        _stmt = off_scope().inject_let_stmts(_stmt);
        _stmt = inject_global_alloc(_stmt);
        _stmt = fixup_idiv(_stmt, var_mgr, ir_ctx);
        _stmt = finalize_vars(_stmt, var_mgr, ir_ctx);
        _stmt = merge_slm_buffers(_stmt, ir_ctx);
        _stmt = inject_slm_reorder(_stmt, ir_ctx,
                to_grid_info(plan_.thr_grid, desc_.thread_group_tile),
                /*has_slm_usage=*/(bool)plan_.epilogue.slm_reduce);
        _stmt = inject_send(_stmt, ir_ctx);
        _stmt = simplify(_stmt, ir_ctx);
        _stmt = optimize_alloc_let(_stmt, ir_ctx);
        _stmt = split_wide_stores(_stmt, ir_ctx);
        _stmt = fixup_if_conditions(_stmt, ir_ctx);
        _stmt = eliminate_common_subexprs(_stmt, ir_ctx, 16, 0);
        _stmt = inject_bank_conflict_attribute(_stmt, ir_ctx);
        set_stmt(_stmt);
    }

private:
    void pipeline(const stream_k_params_t &sk_params = {}) {
        auto &loop_desc = desc_.loop_desc;
        auto &coord_info = plan_.coord_info;
        auto value = sk_params.local_beg;
        if (sk_params) {
            // Unpack the initial loop offsets from a linear index.
            for (auto &e : loop_desc) {
                auto &size = coord_info.loop_size(e.dim);
                let(sk_params.loop_inits[e.idx], value % size);
                value /= size;
            }
        }
        auto loop_nest = make_loop_nest(loop_desc, coord_info, sk_params);
        prefetch_builder_t prefetch_builder(
                *this, loop_nest, buf_info_, plan_.prefetch);
        x2r_mul_builder_t x2r_mul_builder(
                *this, loop_nest, buf_info_, desc_, plan_.x2r_fma);

        zero_out(buf_info_.reg_buf("c"));
        if (!buf_info_.reg_buf("bias").is_empty())
            zero_out(buf_info_.reg_buf("bias"));

        emit(x2r_mul_builder.get_init_stmt());
        emit(prefetch_builder.get_init_stmt());
        int prefetch_dist = desc_.prefetch.dist;
        auto x2r_mul_stmt = x2r_mul_builder.get_stmt();
        auto prefetch_stmt = prefetch_builder.get_stmt();
        iterator_t prefetch_it;
        if (prefetch_dist > 0) {
            prefetch_it = iterator_t(buf_mgr(), loop_nest);
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
        if (desc_.use_stream_k) {
            // Use iterator-based loop with Stream-K.
            iterator_t mul_it(buf_mgr(), loop_nest);
            emit(mul_it.init_stmt());
            auto it_var = var_t::make(type_t::s32(), "sk_local_iter");
            _for(it_var, 0, loop_nest.linear_bound(), [&]() {
                if (prefetch_dist > 0) {
                    emit(prefetch_it.check_bounds_stmt(prefetch_stmt));
                }
                emit(x2r_mul_stmt);
                emit(mul_it.inc_stmt(x2r_mul_builder.off_ctx()));
                if (prefetch_dist > 0) {
                    emit(prefetch_it.inc_stmt(prefetch_builder.off_ctx()));
                }
            });
        } else {
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
                }
                auto &loop = loop_nest[i - 1];
                const auto &var = coord_info.loop_index(loop.dim);
                _for(var, 0, loop.bound, [&]() {
                    emit_loop(i - 1);
                    emit(x2r_mul_builder.off_ctx().inc_loop_stmt(
                            (int)loop.idx));
                });
            };
            emit_loop(loop_nest.nloops());
        }
    }

    void epilogue() {
        epilogue_builder_t builder(*this, buf_info_, desc_, plan_.epilogue);
        emit(builder.get_init_stmt());
        emit(builder.get_stmt());
    }

    stmt_t inject_global_alloc(const stmt_t &stmt) const {
        std::vector<stmt_t> allocs;
        for (auto &var : var_mgr_.ptr_args()) {
            allocs.push_back(alloc_t::make(var, 0, alloc_kind_t::global));
        }
        return inject_alloc_stmts(stmt, allocs);
    }

    void emit_thread_index_let() {
        for (int i = 0; i < 3; i++) {
            auto value = var_t::make(
                    type_t::u16(), jit::ir_builder_t::local_id(i));
            if (i == 0) value /= plan_.desc.simd;
            auto thr_idx = plan_.thr_grid.index_var(i);
            let(thr_idx, cast(value, thr_idx.type()));
        }
        for (auto &kv : plan_.virt_grid.idxs()) {
            let(kv.first, kv.second);
        }
    }

    expr_t unpack_tg_index(const pvar_t &dim, const expr_t &base_idx) const {
        auto &tg_grid = plan_.tg_grid;
        expr_t value = base_idx;
        auto &dims = tg_grid.dims(tg_grid.index(dim));
        int ndims = (int)dims.size();
        for (int i = 0; i < ndims; i++) {
            if (dims[i] == dim) {
                if (i == ndims - 1) return value;
                break;
            }
            auto grid_size = var_mgr_.get_grid_size(dims[i].str());
            value = value / grid_size;
        }
        auto grid_size = var_mgr_.get_grid_size(dim.str());
        value = value % grid_size;
        return value;
    }

    void emit_thread_group_index_let(const expr_t &_base_tg_idx = {}) {
        auto &tg_grid = plan_.tg_grid;
        auto &coord_info = plan_.coord_info;
        for (auto &d : conv_index_dims(plan_.desc.prop)) {
            const auto &tg_idx = coord_info.tg_index(d);
            if (is_const(tg_idx)) continue;
            auto base_tg_idx
                    = (!_base_tg_idx.is_empty() ? _base_tg_idx
                                                : tg_grid.index_var(d));
            if (base_tg_idx.is_empty()) continue;
            auto value = unpack_tg_index(d, base_tg_idx);
            let(tg_idx, value);
        }
    }

    kernel_desc_t desc_;
    var_manager_t &var_mgr_;
    plan_t plan_;
    buffer_info_t buf_info_;
};

stmt_t build_ir(const exec_config_t &exec_cfg, const kernel_desc_t &desc,
        var_manager_t &var_mgr) {
    auto plan = create_conv_plan(desc, exec_cfg.hw());
    if (!plan) gpu_except_not_implemented("Cannot create plan.");

    gpu_info() << desc;
    gpu_trace() << plan;

    constraint_set_t cset;
    ir_context_t ir_ctx(exec_cfg, cset);
    conv_builder_t builder(ir_ctx, desc, var_mgr, plan);
    auto stmt = builder.get_stmt();
    gpu_trace() << "Convolution kernel body:\n" << stmt;
    return stmt;
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
