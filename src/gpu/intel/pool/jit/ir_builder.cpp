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

#include "gpu/intel/pool/jit/ir_builder.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gemmstone/../../dsl/ir/ir.hpp"
#include "gemmstone/../../dsl/ir/pass/trace.hpp"
#include "gpu/intel/jit/ir/epilogue.hpp"
#include "gpu/intel/jit/ir/gemm_schedule.hpp"
#include "gpu/intel/jit/ir/post_ops.hpp"
#include "gpu/intel/jit/ir/send_builder.hpp"
#include "gpu/intel/jit/ir/tensor_config.hpp"
#include "gpu/intel/jit/pass/pass.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace pool {
namespace jit {

class post_op_view_mapper_t : public intel::jit::post_op_view_mapper_t {
    using base_t = intel::jit::post_op_view_mapper_t;

public:
    post_op_view_mapper_t(const view_t &cp_view, const int ndims)
        : base_t(cp_view), ndims_(ndims) {}

    view_t create_view(const dsl::type_t &type, uint32_t mask) const override {
        return base_t::create_view(type, normalize_mask(mask));
    }

    view_t create_view(const memory_desc_t &md) const override {
        dim_idx_t cp_ndims = cp_view().nvdims();
        gpu_assert(cp_ndims >= 3);
        layout_t layout = make_layout(md);
        std::vector<dim_t> dims(md.dims, md.dims + md.ndims);
        std::vector<dim_t> pad_dims(md.padded_dims, md.padded_dims + md.ndims);
        maybe_reshape_dims(ndims_, layout, dims, pad_dims);
        layout = spatials_to_3d(layout, false, {0, 1, 2});
        auto tile = dims_to_3d(dims);
        auto pad_tile = dims_to_3d(pad_dims);
        gpu_assert(layout.ndims() == cp_ndims) << "Incompatible dimensions.";
        uint32_t bound_check_mask = 0;
        for (dim_idx_t i = 0; i < cp_ndims; i++) {
            if (tile[i] == 1) continue; // Broadcast, no bound check needed.
            if (pad_tile[i] != cp_view().tlayout().elems(i)) {
                bound_check_mask |= (1 << i);
            } else if (cp_view().has_tmask(i)) {
                bound_check_mask |= (1 << i);
            }
        }
        return view_t(layout, cp_view().vvars(), tile, bound_check_mask);
    }

    bool need_to_restore_zero_padding() const override { return true; }

private:
    static void maybe_reshape_dims(dim_idx_t ndims, layout_t &layout,
            std::vector<dim_t> &dims, std::vector<dim_t> &padded_dims) {
        gpu_assert(layout.ndims() == dims.size());
        if (layout.ndims() < ndims) {
            layout = layout_t(layout.type(), layout.blocks(), layout.offset(),
                    ndims, /*do_normalize=*/false);
            dims.resize(ndims, 1);
            padded_dims.resize(ndims, 1);
        }
    }

    static tile_t dims_to_3d(const std::vector<dim_t> &dims) {
        layout_t dummy_layout(dsl::type_t::u8(), dims);
        return spatials_to_3d(dummy_layout, false, {0, 1, 2}).tile();
    }

    uint32_t normalize_mask(uint32_t orig_mask) const {
        dim_idx_t cp_ndims = cp_view().nvdims();
        gpu_assert(cp_ndims >= 3);
        // Number of dimensions before normalization.
        dim_idx_t orig_ndims = 2 + ndims_;
        std::vector<dim_t> dummy_dims(orig_ndims, 1);
        dim_t mask_set_value = 2;
        for (dim_idx_t i = 0; i < orig_ndims; i++) {
            if ((orig_mask & (1 << i)) != 0) dummy_dims[i] = mask_set_value;
        }
        auto cvt_dims = dims_to_3d(dummy_dims);
        gpu_assert(cvt_dims.size() == cp_ndims);

        uint32_t mask = 0;
        for (dim_idx_t i = 0; i < cp_ndims; i++) {
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
        gpu_assert(retn.is<int_imm_t>());
        return to_cpp<int>(retn);
    }
    loop_bound_counter_t(const gemm_schedule_t &s) : schedule_(s) {}

private:
    object_t _mutate(const var_t &v) override {
        return expr_t(schedule_.var_bound(v) - 1);
    }
    const gemm_schedule_t &schedule_;
};

stmt_t builder_t::try_build(builder_t &pb, const kernel_info_t &ki,
        const config_t &cfg, const primitive_desc_t &pd) {
    const auto &exec = cfg.options();
    const auto &prb = cfg.problem();
    const auto &src_layout = cfg.src_layout().user();
    const auto &dst_layout = cfg.dst_layout().user();

    const bool is_xe2_or_xe3_small_kdhw = !(prb.kd * prb.kh * prb.kw == 1)
            && (prb.kh * prb.kw <= 9)
            && (exec.hw().ngen_hw() == ngen::HW::Xe2
                    || exec.hw().ngen_hw() == ngen::HW::Xe3);

    gpu_assert(src_layout.ndims() == dst_layout.ndims());

    // Create loop variables.
    auto mb = var_t::make(dsl::type_t::s32(), "mb");
    auto oc = var_t::make(dsl::type_t::s32(), "oc");

    auto od = var_t::make(dsl::type_t::s32(), "od");
    auto oh = var_t::make(dsl::type_t::s32(), "oh");
    auto ow = var_t::make(dsl::type_t::s32(), "ow");

    auto kd = var_t::make(dsl::type_t::s32(), "kd");
    auto kh = var_t::make(dsl::type_t::s32(), "kh");
    auto kw = var_t::make(dsl::type_t::s32(), "kw");

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
    const auto &padded_dims = cfg.dims_padded();
    gpu_assert(padded_dims.size() == 5);
    std::vector<dim_t> dims {padded_dims[0], src_layout.elems(1),
            padded_dims[2], padded_dims[3], padded_dims[4]};

    // Source.
    auto src_view = view_t({mb, oc, od, oh, ow, kd, kh, kw}, 5);
    src_view.set_vdim(mb, (!is_xe2_or_xe3_small_kdhw) ? dims[0] : prb.mb);
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
    dst_view.set_vdim(mb, (!is_xe2_or_xe3_small_kdhw) ? dims[0] : prb.mb);
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
        dim_idx_t s0_idx
                = (s0.is_empty()) ? dim_idx::invalid : src_view.vvar_index(s0);
        dim_idx_t s1_idx = src_view.vvar_index(s1);
        dim_idx_t ns_idx = src_view.vvar_index(ns);
        gpu_assert((s0_idx <= 4 || s0_idx == dim_idx::invalid) && (s1_idx <= 4)
                && (ns_idx <= 4));

        // s1 and ns may swap sides, which affects their fusing order: it has
        // to strictly replicate that of the arguments passed to this lambda!
        const bool need_swap = (s1_idx <= 1);
        // 2 spatials and 2 non-spatials disallowed; only 1 of each or bust
        gpu_assert(need_swap != (ns_idx <= 1));
        if (need_swap) {
            std::swap(s1_idx, ns_idx);
            std::swap(s1, ns);
        }

        const dim_t s1_tlg_unroll = lg[s1_idx];
        const dim_t s1_unroll = s1_tlg_unroll * tg[s1_idx - 2];
        const auto ps1 = s1.str();

        std::vector<expr_t> s0_fuse, s1_fuse;

        expr_t s1_kg, s1_tlg, s1_tg, s1_lg;
        schedule.split(s1, s1_unroll, s1_kg, s1_tlg, ps1 + "_kg", ps1 + "_tlg");
        schedule.split(
                s1_tlg, s1_tlg_unroll, s1_tg, s1_lg, ps1 + "_tg", ps1 + "_lg");

        schedule.tensorize(s1_lg);
        schedule.bind(s1_tg, tg.idx(s1_idx - 2));
        s1_fuse.emplace_back(s1_kg);

        if (s0_idx != dim_idx::invalid) {
            gpu_assert(s0_idx == s1_idx + 1);
            const dim_t s0_tlg_unroll = lg[s0_idx];
            const dim_t s0_unroll = s0_tlg_unroll * tg[s0_idx - 2];
            const dim_t s0_full = s0_unroll * kg[s0_idx - 2];
            const auto ps0 = s0.str();

            if (dims[s0_idx] > s0_full) {
                expr_t s0_split, s0_ktlg; // part of kg[s0] is in kg[s1]
                schedule.split(s0, s0_full, s0_split, s0_ktlg, ps0 + "_split",
                        ps0 + "_ktlg");
                s1_fuse.emplace_back(s0_split);
                s0 = std::move(s0_ktlg);
            } else if (dims[s0_idx] <= utils::div_up(s0_full, 2)) {
                expr_t s1_split, s1_ktlg; // part of kg[s1] is in kg[s0]
                const dim_t s1_ext = utils::div_up(s0_full, dims[s0_idx]);
                schedule.split(s1_fuse[0], s1_ext, s1_ktlg, s1_split,
                        ps1 + "_ktlg", ps1 + "_split");
                s1_fuse[0] = std::move(s1_ktlg);
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

        const dim_t ns_unroll = lg[ns_idx];
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
        gpu_assert((k_idx >= 5) && (k_idx <= 7));
        const dim_t k_dim = lg[k_idx];
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
            gpu_error_not_expected() << "k_dim > var_bound; this is wrong";
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

    auto src_thr_tile_coord
            = schedule.thr_view_tile_coord(src_view, /*is_relative=*/false);
    auto src_thr_view = src_view.create_sub_view(src_thr_tile_coord);

    auto dst_thr_tile_coord
            = schedule.thr_view_tile_coord(dst_view, /*is_relative=*/false);
    auto dst_thr_view = dst_view.create_sub_view(dst_thr_tile_coord);

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
    auto acc_buf = ir_ctx.create_tmp_var(
            dsl::type_t::byte(dsl::type::attr_t::ptr), "acc");
    const auto acc_sc_size = acc_type.base().size();
    const auto acc_size = acc_sc_size * lg[4] * lg[3] * lg[2] * lg[1] * lg[0];

    auto read_buf = ir_ctx.create_tmp_var(
            dsl::type_t::byte(dsl::type::attr_t::ptr), "read");
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

    tile_t src_tile(read_layout.max_subtile(simd));
    tile_t dst_tile(write_layout.max_subtile(simd));
    gpu_assert(src_tile.elems() == simd);
    gpu_assert(dst_tile.elems() == simd);

    const bool is_identity(prb.kd * prb.kh * prb.kw <= 1);

    const dsl::type_t read_type(read_layout.type()[simd]);

    stmt_t stmt;

    auto gen_fill_values = [](int simd, bool isneg, dsl::type_t type) {
        gpu_assert(type.base().size() <= 4);
        const int mult = 4 / type.base().size();
        expr_t v = 0;
        if (isneg) {
            v = type.is_f32()        ? 0xFF7FFFFF
                    : type.is_bf16() ? 0xFF7FFF7F
                    : type.is_f16()  ? 0xFBFFFBFF
                    : (mult > 1)     ? (mult > 2) ? 0x80808080 : 0x80008000
                                     : 0x80000000;
        }
        v = cast_t::make(dsl::type_t::s32(), v);
        auto v_long = shuffle_t::make_broadcast(v, simd);
        auto v_short = shuffle_t::make_broadcast(v, simd / mult);
        return std::make_pair(v_short, v_long);
    };
    auto gen_zero_out = [&](int simd, bool isneg, const expr_t &buf,
                                const tile_t &tile, const layout_t &layout) {
        stmt_t retn;
        const auto values = gen_fill_values(simd, isneg, layout.type());
        for (auto &s : layout.iter(tile)) {
            const dim_t off = layout.offset<dim_t>(s) * layout.type().size();
            if (off >= utils::rnd_dn(size_bytes(layout), simd * 4))
                retn = retn.append(store_t::make(buf, off, values.first));
            else if (off % (simd * 4) == 0)
                retn = retn.append(store_t::make(buf, off, values.second));
        }
        return retn;
    };
    const bool is_neg
            = cfg.is_max() && (!read_type.is_int() || read_type.is_signed());
    if (is_identity) {
        allocs.emplace_back(read_alloc[0]);
        write_stmt = substitute(write_stmt, acc_buf, read_buf);
        acc_buf = std::move(read_buf);
        acc_type = read_type;
        stmt = (check_idhw)
                ? gen_zero_out(simd, is_neg, acc_buf, dst_tile, write_layout)
                : stmt_t();
        stmt = stmt.append(read.stmt());
    } else {
        gpu_assert(acc_size % simd == 0);
        auto acc_buf_size = std::max(into<int>(acc_size), write.reg_buf_size());
        allocs.push_back(
                alloc_t::make(acc_buf, acc_buf_size, alloc_kind_t::grf));

        stmt_t fill_stmt, compute_stmt = read.stmt();
        stmt = stmt_t();

        const auto a_fv = gen_fill_values(simd, is_neg, acc_type);
        for (dim_t i = 0; i < acc_size; i += simd * 4)
            stmt = stmt.append(store_t::make(acc_buf, i,
                    (acc_size - i < simd * 4) ? a_fv.first : a_fv.second));
        fill_stmt = gen_zero_out(simd, is_neg, read_buf, src_tile, read_layout);

        for (auto &s : read_layout.iter(src_tile)) {
            const dim_t off_l
                    = read_layout.offset<dim_t>(s) * read_layout.type().size();
            const dim_t off_a = (s.get(0) * lg[1] + s.get(1)) * acc_sc_size;

            auto load = cast_t::make(
                    acc_type, load_t::make(read_type, read_buf, off_l));
            auto acc = load_t::make(acc_type, acc_buf, off_a);
            auto op_kind = (cfg.is_max()) ? op_kind_t::_max : op_kind_t::_add;
            auto op = binary_op_t::make(op_kind, acc, load);
            compute_stmt
                    = compute_stmt.append(store_t::make(acc_buf, off_a, op));
        }

        stmt = stmt.append(schedule.create_loop_nest(
                (check_idhw) ? fill_stmt.append(compute_stmt) : compute_stmt));

        if (!cfg.is_max()) {
            expr_t filter(prb.kd * prb.kh * prb.kw);
            if (!cfg.is_padded() && check_idhw) {
                auto dim = [](const expr_t &o, dim_t s, dim_t p, dim_t k,
                                   dim_t i) {
                    if (k <= 1) return expr_t(1);
                    return binary_op_t::make(op_kind_t::_min, o * s - p + k, i)
                            - binary_op_t::make(op_kind_t::_max, o * s - p, 0);
                };
                auto dhw = dim(od, prb.stride_d, prb.f_pad, prb.kd, prb.id)
                        * dim(oh, prb.stride_h, prb.t_pad, prb.kh, prb.ih)
                        * dim(ow, prb.stride_w, prb.l_pad, prb.kw, prb.iw);
                filter = cast_t::make(dsl::type_t::f32(), dhw);
            }
            filter = shuffle_t::make_broadcast(filter, simd);
            for (dim_t i = 0; i < acc_size; i += simd * acc_sc_size) {
                auto acc = cast_t::make(dsl::type_t::f32(simd),
                        load_t::make(acc_type, acc_buf, i));
                stmt = stmt.append(store_t::make(acc_buf, i, acc / filter));
            }
            acc_type = dsl::type_t::f32(simd);
        }
        stmt = inject_alloc_stmts(stmt, read_alloc);
    }

    if (is_identity && pd.attr()->post_ops_.has_default_values()) {
        stmt = stmt.append(write_stmt);
    } else {
        int buf_size = 0;
        post_op_view_mapper_t view_mapper(dst_view, prb.ndims);
        post_op_context_t post_op_ctx(*pd.attr(), cfg.zp_cfg(), schedule, ki,
                *pd.invariant_dst_md(), *pd.invariant_dst_md(), view_mapper);
        stmt = stmt.append(create_epilogue_stmt(exec, ir_ctx, schedule,
                /*force_c_reorder=*/false, post_op_ctx, dst_thr_tile_coord,
                write_layout.with(acc_type.base()), dst_buf, acc_buf,
                buf_size));
        for (auto &alloc : allocs) {
            auto &a = alloc.as<alloc_t>();
            if (a.buf.is_same(acc_buf))
                alloc = alloc_t::make(a.buf,
                        std::max(a.size, uint32_t(buf_size)), a.kind, a.attrs,
                        a.body);
        }
    }

    loop_bound_counter_t lbc(schedule);
    auto exit_cond = (lbc.count(ow) >= prb.ow) ? (ow < prb.ow) : expr_t();
    if (lbc.count(oh) >= prb.oh)
        exit_cond = exit_cond ? (oh < prb.oh) & exit_cond : (oh < prb.oh);
    if (lbc.count(od) >= prb.od)
        exit_cond = exit_cond ? (od < prb.od) & exit_cond : (od < prb.od);
    if (exit_cond)
        stmt = if_t::make(shuffle_t::make_broadcast(exit_cond, simd), stmt);

    if (!is_xe2_or_xe3_small_kdhw && ((dims[0] - prb.mb) / lg[0] >= 1)) {
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
    stmt = inject_access_map_attribute(stmt, ir_ctx);
    stmt = stmt_group_t::make(stmt_label_t::kernel(), stmt);

    const int regs = get_peak_regs(stmt, exec.grf_size());

    gpu_debug() << "Pooling kernel body:\n" << stmt;
    gpu_debug() << "Pooling cfg (~" << regs << " regs):\n" << cfg;

    return (regs > exec.regs()) ? stmt_t() : std::move(stmt);
}

} // namespace jit
} // namespace pool
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
