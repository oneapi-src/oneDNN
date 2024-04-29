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

#include "gpu/intel/jit/pooling/ir_builder.hpp"

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
#include "common/utils.hpp"
#include "gpu/intel/jit/codegen/ngen_helpers.hpp"
#include "gpu/intel/jit/ir/epilogue.hpp"
#include "gpu/intel/jit/ir/gemm_schedule.hpp"
#include "gpu/intel/jit/ir/ir.hpp"
#include "gpu/intel/jit/ir/message.hpp"
#include "gpu/intel/jit/ir/post_ops.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/jit/pass/pass.hpp"
#include "gpu/intel/jit/utils/iterator.hpp"
#include "gpu/intel/jit/utils/range.hpp"
#include "gpu/intel/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

class pooling_post_op_view_mapper_t : public post_op_view_mapper_t {
public:
    pooling_post_op_view_mapper_t(const view_t &cp_view, const int ndims)
        : post_op_view_mapper_t(cp_view), ndims_(ndims) {}

    view_t create_view(const type_t &type, uint32_t mask) const override {
        return post_op_view_mapper_t::create_view(type, normalize_mask(mask));
    }

    view_t create_view(const memory_desc_t &md) const override {
        int cp_ndims = cp_view().nvdims();
        ir_assert(cp_ndims >= 3);
        layout_t layout(md, /*do_normalize=*/false);
        std::vector<dim_t> dims(md.dims, md.dims + md.ndims);
        std::vector<dim_t> pad_dims(md.padded_dims, md.padded_dims + md.ndims);
        maybe_reshape_dims(ndims_, layout, dims, pad_dims);
        layout = spatials_to_3d(layout, false, {0, 1, 2});
        dims = dims_to_3d(dims);
        pad_dims = dims_to_3d(pad_dims);
        ir_assert(layout.ndims() == cp_ndims) << "Incompatible dimensions.";
        uint32_t bound_check_mask = 0;
        for (int i = 0; i < cp_ndims; i++) {
            if (dims[i] == 1) continue; // Broadcast, no bound check needed.
            if (pad_dims[i] != cp_view().tlayout().dim(i)) {
                bound_check_mask |= (1 << i);
            } else if (cp_view().has_tmask(i)) {
                bound_check_mask |= (1 << i);
            }
        }
        return view_t(layout, cp_view().vvars(), dims, bound_check_mask);
    }

    bool need_to_restore_zero_padding() const override { return true; }

private:
    static void maybe_reshape_dims(int ndims, layout_t &layout,
            std::vector<dim_t> &dims, std::vector<dim_t> &padded_dims) {
        ir_assert(layout.ndims() == int(dims.size()));
        if (layout.ndims() < ndims) {
            layout = layout_t(layout.type(), ndims, layout.offset(),
                    layout.blocks(), /*do_normalize=*/false);
            dims.resize(ndims, 1);
            padded_dims.resize(ndims, 1);
        }
    }

    static std::vector<dim_t> dims_to_3d(const std::vector<dim_t> &dims) {
        layout_t dummy_layout(type_t::u8(), 0, dims);
        return spatials_to_3d(dummy_layout, false, {0, 1, 2}).dims();
    }

    uint32_t normalize_mask(uint32_t orig_mask) const {
        int cp_ndims = cp_view().nvdims();
        ir_assert(cp_ndims >= 3);
        // Number of dimensions before normalization.
        int orig_ndims = 2 + ndims_;
        std::vector<dim_t> dummy_dims(orig_ndims, 1);
        dim_t mask_set_value = 2;
        for (int i = 0; i < orig_ndims; i++) {
            if ((orig_mask & (1 << i)) != 0) dummy_dims[i] = mask_set_value;
        }
        auto cvt_dims = dims_to_3d(dummy_dims);
        ir_assert(int(cvt_dims.size()) == cp_ndims);

        uint32_t mask = 0;
        for (int i = 0; i < cp_ndims; i++) {
            if (cvt_dims[i] == mask_set_value) mask = mask | (1 << i);
        }
        return mask;
    }

    const int ndims_;
};

class loop_bound_counter_t : public ir_mutator_t {
public:
    int count(const expr_t &e) {
        const auto retn = simplify(mutate(e));
        ir_assert(retn.is<int_imm_t>());
        return to_cpp<int>(retn);
    }
    loop_bound_counter_t(const gemm_schedule_t &s) : schedule_(s) {}

private:
    object_t _mutate(const var_t &v) override {
        return expr_t(schedule_.var_bound(v) - 1);
    }
    const gemm_schedule_t &schedule_;
};

stmt_t pooling_ir_builder_t::try_build(pooling_ir_builder_t &pb,
        const kernel_info_t &ki, const pooling_config_t &cfg,
        const primitive_desc_t &pd) {
    const auto &exec = cfg.exec_cfg();
    const auto &prb = cfg.pooling_problem();
    const auto &src_layout = cfg.src_layout().user();
    const auto &dst_layout = cfg.dst_layout().user();

    const bool is_xe2_small_kdhw = !(prb.kd * prb.kh * prb.kw == 1)
            && (prb.kh * prb.kw <= 9) && (exec.hw().to_ngen() == ngen::HW::Xe2);

    ir_assert(src_layout.ndims() == dst_layout.ndims());

    // Create loop variables.
    auto mb = var_t::make(type_t::s32(), "mb");
    auto oc = var_t::make(type_t::s32(), "oc");

    auto od = var_t::make(type_t::s32(), "od");
    auto oh = var_t::make(type_t::s32(), "oh");
    auto ow = var_t::make(type_t::s32(), "ow");

    auto kd = var_t::make(type_t::s32(), "kd");
    auto kh = var_t::make(type_t::s32(), "kh");
    auto kw = var_t::make(type_t::s32(), "kw");

    // Initialize masks.
    const bool check_iw = utils::need_src_or_dst_check(!prb.is_backward, prb.ow,
            prb.iw, prb.kw, prb.l_pad, prb.stride_w, prb.dw);
    const bool check_ih = utils::need_src_or_dst_check(!prb.is_backward, prb.oh,
            prb.ih, prb.kh, prb.t_pad, prb.stride_h, prb.dh);
    const bool check_id = utils::need_src_or_dst_check(!prb.is_backward, prb.od,
            prb.id, prb.kd, prb.f_pad, prb.stride_d, prb.dd);
    const bool check_idhw = check_id || check_ih || check_iw;

    auto &x = view_t::placeholder_var();

    expr_t id_mask, ih_mask, iw_mask;
    if (check_id) id_mask = (x >= 0) & (x < prb.id);
    if (check_ih) ih_mask = (x >= 0) & (x < prb.ih);
    if (check_iw) iw_mask = (x >= 0) & (x < prb.iw);

    const int simd = exec.simd();
    const auto &lg = cfg.loop_grid();
    const auto &kg = cfg.kernel_grid();
    const auto &tg = cfg.thread_group_grid();
    const auto &dims_grid = cfg.dims_padded();
    std::vector<int> padded_dims(dims_grid.ndims());
    for (int i = 0; i < int(padded_dims.size()); i++)
        padded_dims[i] = dims_grid[i];
    ir_assert(padded_dims.size() == 5);
    std::vector<int> dims {padded_dims[0], int(src_layout.dim(1)),
            padded_dims[2], padded_dims[3], padded_dims[4]};

    // Source.
    auto src_view = view_t({mb, oc, od, oh, ow, kd, kh, kw}, 5);
    src_view.set_vdim(mb, (!is_xe2_small_kdhw) ? dims[0] : prb.mb);
    src_view.set_vdim(oc, dims[1]);
    src_view.set_vdim(od, dims[2]);
    src_view.set_vdim(oh, dims[3]);
    src_view.set_vdim(ow, dims[4]);
    src_view.set_vdim(kd, prb.kd);
    src_view.set_vdim(kh, prb.kh);
    src_view.set_vdim(kw, prb.kw);
    src_view.set_tdim(0, mb);
    src_view.set_tdim(1, oc);
    src_view.set_tdim(
            2, od * prb.stride_d - prb.f_pad + kd * (1 + prb.dd), id_mask);
    src_view.set_tdim(
            3, oh * prb.stride_h - prb.t_pad + kh * (1 + prb.dh), ih_mask);
    src_view.set_tdim(
            4, ow * prb.stride_w - prb.l_pad + kw * (1 + prb.dw), iw_mask);
    src_view.set_tlayout(src_layout);
    src_view.set_tmasks(padded_dims);

    // Destination.
    auto dst_view = view_t({mb, oc, od, oh, ow}, 5);
    dst_view.set_vdim(mb, (!is_xe2_small_kdhw) ? dims[0] : prb.mb);
    dst_view.set_vdim(oc, dims[1]);
    dst_view.set_vdim(od, dims[2]);
    dst_view.set_vdim(oh, dims[3]);
    dst_view.set_vdim(ow, dims[4]);
    dst_view.set_tdim(0, mb);
    dst_view.set_tdim(1, oc);
    dst_view.set_tdim(2, od);
    dst_view.set_tdim(3, oh);
    dst_view.set_tdim(4, ow);
    dst_view.set_tlayout(dst_layout);
    dst_view.set_tmasks(padded_dims);

    constraint_set_t init_cset;
    std::vector<stmt_t> init_stmts;
    pb.init_kernel_grid(kg, tg, simd, init_cset, init_stmts);

    gemm_schedule_t schedule(init_cset, kg, tg);
    schedule.set_view(src_view);
    schedule.set_view(dst_view);
    schedule.set_var_bound(mb, dims[0]);

    auto kg_bind = [&](const std::vector<expr_t> &fuse, int idx) {
        if (fuse.size() > 1)
            schedule.bind(schedule.fuse(fuse), kg.idx(idx));
        else if (fuse.size() == 1)
            schedule.bind(fuse[0], kg.idx(idx));
    };
    auto odhw_to_schedule = [&](expr_t s1, expr_t ns, expr_t s0) {
        int s0_idx = (s0.is_empty()) ? -1 : src_view.vvar_index(s0);
        int s1_idx = src_view.vvar_index(s1);
        int ns_idx = src_view.vvar_index(ns);
        ir_assert((s0_idx <= 4) && (s1_idx <= 4) && (ns_idx <= 4));

        // s1 and ns may swap sides, which affects their fusing order: it has
        // to strictly replicate that of the arguments passed to this lambda!
        const bool need_swap = (s1_idx >= 0) && (s1_idx <= 1);
        // 2 spatials and 2 non-spatials disallowed; only 1 of each or bust
        ir_assert(need_swap != ((ns_idx >= 0) && (ns_idx <= 1)));
        if (need_swap) {
            std::swap(s1_idx, ns_idx);
            std::swap(s1, ns);
        }

        const int s1_tlg_unroll = lg[s1_idx];
        const int s1_unroll = s1_tlg_unroll * tg[s1_idx - 2];
        const auto ps1 = s1.str();

        std::vector<expr_t> s0_fuse, s1_fuse;

        expr_t s1_kg, s1_tlg, s1_tg, s1_lg;
        schedule.split(s1, s1_unroll, s1_kg, s1_tlg, ps1 + "_kg", ps1 + "_tlg");
        schedule.split(
                s1_tlg, s1_tlg_unroll, s1_tg, s1_lg, ps1 + "_tg", ps1 + "_lg");

        schedule.tensorize(s1_lg);
        schedule.bind(s1_tg, tg.idx(s1_idx - 2));
        s1_fuse.emplace_back(s1_kg);

        if (s0_idx >= 0) {
            ir_assert(s0_idx == s1_idx + 1);
            const int s0_tlg_unroll = lg[s0_idx];
            const int s0_unroll = s0_tlg_unroll * tg[s0_idx - 2];
            const int s0_full = s0_unroll * kg[s0_idx - 2];
            const auto ps0 = s0.str();

            if (dims[s0_idx] > s0_full) {
                expr_t s0_split, s0_ktlg; // part of kg[s0] is in kg[s1]
                schedule.split(s0, s0_full, s0_split, s0_ktlg, ps0 + "_split",
                        ps0 + "_ktlg");
                s1_fuse.emplace_back(s0_split);
                s0 = s0_ktlg;
            } else if (dims[s0_idx] <= utils::div_up(s0_full, 2)) {
                expr_t s1_split, s1_ktlg; // part of kg[s1] is in kg[s0]
                const int s1_ext = utils::div_up(s0_full, dims[s0_idx]);
                schedule.split(s1_fuse[0], s1_ext, s1_ktlg, s1_split,
                        ps1 + "_ktlg", ps1 + "_split");
                s1_fuse[0] = s1_ktlg;
                s0_fuse.emplace_back(s1_split);
            }

            expr_t s0_kg, s0_tlg, s0_tg, s0_lg;
            schedule.split(
                    s0, s0_unroll, s0_kg, s0_tlg, ps0 + "_kg", ps0 + "_tlg");
            schedule.split(s0_tlg, s0_tlg_unroll, s0_tg, s0_lg, ps0 + "_tg",
                    ps0 + "_lg");

            schedule.tensorize(s0_lg);
            schedule.bind(s0_tg, tg.idx(s0_idx - 2));
            s0_fuse.emplace_back(s0_kg);
        }

        const int ns_unroll = lg[ns_idx];
        const auto pns = ns.str();

        expr_t ns_kg, ns_lg;
        schedule.split(ns, ns_unroll, ns_kg, ns_lg, pns + "_kg", pns + "_lg");
        if (need_swap)
            s1_fuse.emplace(s1_fuse.begin(), ns_kg);
        else
            s1_fuse.emplace_back(ns_kg);
        schedule.tensorize(ns_lg);

        kg_bind(s0_fuse, s0_idx - 2);
        kg_bind(s1_fuse, s1_idx - 2);
    };
    odhw_to_schedule(oc, od, expr_t());
    if (cfg.is_blocked_by_mb())
        odhw_to_schedule(oh, mb, ow);
    else
        odhw_to_schedule(mb, oh, ow);

    auto kdhw_to_schedule = [&](const expr_t &k) {
        const int k_idx = src_view.vvar_index(k);
        ir_assert((k_idx >= 5) && (k_idx <= 7));
        const int k_dim = lg[k_idx];
        if (k_dim == schedule.var_bound(k)) {
            schedule.tensorize(k);
        } else if (k_dim < schedule.var_bound(k)) {
            if (k_dim > 1) { // otherwise it'll just waste a variable
                expr_t k_lg, k_tnz;
                schedule.split(k, k_dim, k_lg, k_tnz, k.str() + "_lg",
                        k.str() + "_tnz");
                schedule.tensorize(k_tnz);
            }
        } else {
            ir_error_not_expected() << "k_dim > var_bound; this is wrong";
        }
    };
    kdhw_to_schedule(kd);
    kdhw_to_schedule(kh);
    kdhw_to_schedule(kw);

    schedule.finalize();

    const auto expand_loop_kinds = loop_kind_t::serial
            | loop_kind_t::kernel_grid | loop_kind_t::tg_grid;
    mb = schedule.expand(mb, true, expand_loop_kinds);
    oc = schedule.expand(oc, true, expand_loop_kinds);
    od = schedule.expand(od, true, expand_loop_kinds);
    oh = schedule.expand(oh, true, expand_loop_kinds);
    ow = schedule.expand(ow, true, expand_loop_kinds);

    auto src_thr_tile = schedule.thr_view_tile(src_view, /*is_relative=*/false);
    auto src_thr_view = src_view.create_sub_view(src_thr_tile);

    auto dst_thr_tile = schedule.thr_view_tile(dst_view, /*is_relative=*/false);
    auto dst_thr_view = dst_view.create_sub_view(dst_thr_tile);

    const auto &src_buf = ki.arg_var(0);
    const auto &dst_buf = ki.arg_var(1);

    std::vector<stmt_t> allocs;
    for (int i = 0; i < ki.nargs(); i++) {
        auto &var = ki.arg_var(i);
        if (!var.type().is_ptr()) continue;
        allocs.push_back(alloc_t::make(var, 0, alloc_kind_t::global));
    }

    ir_context_t ir_ctx(exec, init_cset);

    auto acc_type = cfg.acc_type(simd);
    auto acc_buf = ir_ctx.create_tmp_var(type_t::byte_ptr(), "acc");
    const auto acc_sc_size = acc_type.scalar().size();
    const auto acc_size = acc_sc_size * lg[4] * lg[3] * lg[2] * lg[1] * lg[0];

    auto read_buf = ir_ctx.create_tmp_var(type_t::byte_ptr(), "read");
    auto read_params = get_send_params(
            exec, send_op_t::load, send_address_t::a64, src_thr_view);
    read_params.try_legacy = false;
    auto read = make_access_builder(ir_ctx, src_thr_view, src_buf, read_buf,
            read_params, /*zero_out=*/false);
    std::vector<stmt_t> read_alloc
            = {alloc_t::make(read_buf, read.reg_buf_size(), alloc_kind_t::grf)};
    const auto &read_layout = read.reg_layout();

    // shall only get used on empty mb's; for all else there's epilogue builder
    auto write_params = get_send_params(
            exec, send_op_t::store, send_address_t::a64, dst_thr_view);
    write_params.try_legacy = false;
    auto write = make_access_builder(
            ir_ctx, dst_thr_view, dst_buf, acc_buf, write_params);
    const auto &write_layout = write.reg_layout();
    auto write_stmt = write.stmt();

    tensor_t src_tile(read_layout.split_into_max_tile(simd, true));
    tensor_t dst_tile(write_layout.split_into_max_tile(simd, true));
    ir_assert(src_tile.elems() == simd);
    ir_assert(dst_tile.elems() == simd);

    const bool is_identity(prb.kd * prb.kh * prb.kw <= 1);

    const type_t read_type(read_layout.type().kind(), simd);
    const type_t write_type(write_layout.type().kind(), simd);

    stmt_t stmt;

    auto gen_fill_values = [](int simd, bool isneg, type_t type) {
        ir_assert(type.scalar().size() <= 4);
        const int mult = 4 / type.scalar().size();
        expr_t v = 0;
        if (isneg) {
            switch (to_ngen(type.scalar())) {
                case ngen::DataType::f: v = 0xFF7FFFFF; break;
                case ngen::DataType::bf: v = 0xFF7FFF7F; break;
                case ngen::DataType::hf: v = 0xFBFFFBFF; break;
                default:
                    v = (mult > 1) ? (mult > 2) ? 0x80808080 : 0x80008000
                                   : 0x80000000;
            }
        }
        v = cast_t::make(type_t::s32(), v);
        auto v_long = shuffle_t::make_broadcast(v, simd);
        auto v_short = shuffle_t::make_broadcast(v, simd / mult);
        return std::make_pair(v_short, v_long);
    };
    auto gen_zero_out = [&](int simd, bool isneg, const expr_t &buf,
                                const tensor_t &tile, const layout_t &layout) {
        stmt_t retn;
        const auto values = gen_fill_values(simd, isneg, layout.type());
        layout.for_each_tile(tile, [&](const std::vector<dim_t> &s) {
            const int off = layout(s) * layout.type().size();
            if (off >= utils::rnd_dn(layout.size(), simd * 4))
                retn = retn.append(store_t::make(buf, off, values.first));
            else if (off % (simd * 4) == 0)
                retn = retn.append(store_t::make(buf, off, values.second));
        });
        return retn;
    };
    const bool is_neg
            = cfg.is_max() && (!read_type.is_int() || read_type.is_signed());
    if (is_identity) {
        allocs.emplace_back(read_alloc[0]);
        write_stmt = substitute(write_stmt, acc_buf, read_buf);
        acc_buf = read_buf;
        acc_type = read_type;
        stmt = (check_idhw)
                ? gen_zero_out(simd, is_neg, acc_buf, dst_tile, write_layout)
                : stmt_t();
        stmt = stmt.append(read.stmt());
    } else {
        ir_assert(acc_size % simd == 0);
        allocs.push_back(alloc_t::make(acc_buf, acc_size, alloc_kind_t::grf));

        stmt_t fill_stmt, compute_stmt = read.stmt();
        stmt = stmt_t();

        const auto a_fv = gen_fill_values(simd, is_neg, acc_type);
        for (int i = 0; i < acc_size; i += simd * 4)
            stmt = stmt.append(store_t::make(acc_buf, i,
                    (acc_size - i < simd * 4) ? a_fv.first : a_fv.second));
        fill_stmt = gen_zero_out(simd, is_neg, read_buf, src_tile, read_layout);

        read_layout.for_each_tile(src_tile, [&](const std::vector<dim_t> &s) {
            const int off_l = read_layout(s) * read_layout.type().size();
            const int off_a = (s[0] * lg[1] + s[1]) * acc_sc_size;

            auto load = cast_t::make(
                    acc_type, load_t::make(read_type, read_buf, off_l));
            auto acc = load_t::make(acc_type, acc_buf, off_a);
            auto op_kind = (cfg.is_max()) ? op_kind_t::_max : op_kind_t::_add;
            auto op = binary_op_t::make(op_kind, acc, load);
            compute_stmt
                    = compute_stmt.append(store_t::make(acc_buf, off_a, op));
        });

        stmt = stmt.append(schedule.create_loop_nest(
                (check_idhw) ? fill_stmt.append(compute_stmt) : compute_stmt));

        if (!cfg.is_max()) {
            expr_t filter(prb.kd * prb.kh * prb.kw);
            if (!cfg.is_padded() && check_idhw) {
                auto dim = [](const expr_t &o, int s, int p, int k, int i) {
                    if (k <= 1) return expr_t(1);
                    return binary_op_t::make(op_kind_t::_min, o * s - p + k, i)
                            - binary_op_t::make(op_kind_t::_max, o * s - p, 0);
                };
                auto dhw = dim(od, prb.stride_d, prb.f_pad, prb.kd, prb.id)
                        * dim(oh, prb.stride_h, prb.t_pad, prb.kh, prb.ih)
                        * dim(ow, prb.stride_w, prb.l_pad, prb.kw, prb.iw);
                filter = cast_t::make(type_t::f32(), dhw);
            }
            filter = shuffle_t::make_broadcast(filter, simd);
            for (int i = 0; i < acc_size; i += simd * acc_sc_size) {
                auto acc = cast_t::make(
                        type_t::f32(simd), load_t::make(acc_type, acc_buf, i));
                stmt = stmt.append(store_t::make(acc_buf, i, acc / filter));
            }
            acc_type = type_t::f32(simd);
        }
        stmt = inject_alloc_stmts(stmt, read_alloc);
    }

    int buf_size = 0;
    pooling_post_op_view_mapper_t view_mapper(dst_view, prb.ndims);
    post_op_context_t post_op_ctx(*pd.attr(), cfg.zp_cfg(), schedule, ki,
            *pd.invariant_dst_md(), *pd.invariant_dst_md(), view_mapper);
    stmt = stmt.append(create_epilogue_stmt(exec, ir_ctx, schedule,
            /*force_c_reorder=*/false, post_op_ctx, dst_thr_tile,
            write_layout.retype(acc_type.scalar()), dst_buf, acc_buf,
            buf_size));

    loop_bound_counter_t lbc(schedule);
    auto exit_cond = (lbc.count(ow) >= prb.ow) ? (ow < prb.ow) : expr_t();
    if (lbc.count(oh) >= prb.oh)
        exit_cond = (!exit_cond.is_empty()) ? (oh < prb.oh) & exit_cond
                                            : (oh < prb.oh);
    if (lbc.count(od) >= prb.od)
        exit_cond = (!exit_cond.is_empty()) ? (od < prb.od) & exit_cond
                                            : (od < prb.od);
    if (!exit_cond.is_empty())
        stmt = if_t::make(shuffle_t::make_broadcast(exit_cond, simd), stmt);

    if (!is_xe2_small_kdhw && ((dims[0] - prb.mb) / lg[0] >= 1)) {
        auto stop = gen_zero_out(simd, false, acc_buf, dst_tile, write_layout);
        stmt = if_t::make(shuffle_t::make_broadcast(mb >= prb.mb, simd),
                stop.append(write_stmt), stmt);
    }

    stmt = schedule.create_bind_stmt(stmt);
    stmt = inject_let_stmts(stmt, init_stmts);
    stmt = inject_alloc_stmts(stmt, allocs);
    stmt = inject_external_var_let(stmt, ir_ctx);

    stmt = simplify(stmt, ir_ctx);
    stmt = lift_buffer_offsets_in_send(stmt, ir_ctx);
    stmt = inject_send(stmt, ir_ctx);
    stmt = split_wide_stores(stmt, ir_ctx);
    stmt = fix_int32_overflow(stmt, ir_ctx);
    stmt = eliminate_common_subexprs(
            stmt, ir_ctx, exec.regs() * exec.grf_size());
    stmt = simplify(stmt, ir_ctx);
    stmt = optimize_alloc_let(stmt, ir_ctx);
    stmt = stmt_group_t::make(stmt_label_t::kernel(), stmt);

    const int regs = get_peak_regs(stmt, exec.grf_size());

    ir_trace() << "Pooling kernel body:\n" << stmt << std::endl;
    ir_trace() << "Pooling cfg (~" << regs << " regs):\n" << cfg << std::endl;

    return (regs > exec.regs()) ? stmt_t() : stmt;
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
