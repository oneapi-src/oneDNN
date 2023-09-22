/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include "gpu/jit/conv/ir_builder.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>
#include <unordered_map>

#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/normalization.hpp"
#include "gpu/jit/conv/pipeline.hpp"
#include "gpu/jit/conv/plan.hpp"
#include "gpu/jit/ir/epilogue.hpp"
#include "gpu/jit/ir/fma.hpp"
#include "gpu/jit/ir/gemm_schedule.hpp"
#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/ir/post_ops.hpp"
#include "gpu/jit/ir/reduce.hpp"
#include "gpu/jit/ir/reorder.hpp"
#include "gpu/jit/ir/slm_reduce_builder.hpp"
#include "gpu/jit/ir/tensor.hpp"
#include "gpu/jit/pass/pass.hpp"
#include "gpu/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class buffer_access_verifier_t : public ir_visitor_t {
public:
    void _visit(const alloc_t &obj) override {
        buf_sizes_.emplace(obj.buf, obj.size);
        ir_visitor_t::_visit(obj);
        buf_sizes_.erase(obj.buf);
    }

    void _visit(const func_call_t &obj) override {
        auto &func = obj.func;
        if (auto *dpas = func.as_ptr<dpas_t>()) {
            auto &dst = dpas_t::arg_dst(obj);
            auto &src0 = dpas_t::arg_src0(obj);
            auto &src1 = dpas_t::arg_src1(obj);
            auto &src2 = dpas_t::arg_src2(obj);
            check_access(dst, dpas->dst_size(), obj);
            if (!is_zero(src0)) check_access(src0, dpas->src0_size(), obj);
            check_access(src1, dpas->src1_size(), obj);
            check_access(src2, dpas->src2_size(), obj);
        } else if (func.is<eltwise_t>()) {
            auto &elems = eltwise_t::arg_elems(obj);
            auto &data = eltwise_t::arg_data(obj);
            int size = to_cpp<int>(elems) * sizeof(float);
            check_access(data, size, obj);
        } else if (auto *mad = func.as_ptr<mad_t>()) {
            auto &dst = mad_t::arg_dst(obj);
            auto &src0 = mad_t::arg_src0(obj);
            auto &src1 = mad_t::arg_src1(obj);
            auto &src2 = mad_t::arg_src2(obj);
            check_access(dst, mad->dst_size(), obj);
            if (!is_zero(src0)) check_access(src0, mad->src0_size(), obj);
            check_access(src1, mad->src1_size(), obj);
            check_access(src2, mad->src2_size(), obj);
        } else if (auto *reduce = func.as_ptr<reduce_t>()) {
            auto &dst_buf = reduce_t::arg_dst_buf(obj);
            auto &src_buf = reduce_t::arg_src_buf(obj);
            check_access(dst_buf, reduce->dst_layout.size(), obj);
            check_access(src_buf, reduce->src_layout.size(), obj);
        } else if (auto *reorder = func.as_ptr<reorder_t>()) {
            auto &dst_buf = reorder_t::arg_dst_buf(obj);
            auto &src_buf = reorder_t::arg_src_buf(obj);
            check_access(dst_buf, reorder->dst_layout.size(), obj);
            check_access(src_buf, reorder->src_layout.size(), obj);
            return;
        } else if (auto *send = func.as_ptr<send_t>()) {
            if (!send->is_prefetch() && !send->is_prefetch_2d()) {
                auto &reg_buf = send_t::arg_reg_buf(obj);
                int size = send->payload_size();
                check_access(reg_buf, size, obj);
            }
            return;
        } else if (func.is<builtin_t>()) {
            // No buffers to check.
        } else {
            ir_error_not_expected() << "Unhandled function: " << obj;
        }

        ir_visitor_t::_visit(obj);
    }

    void _visit(const load_t &obj) override {
        auto elem_type = obj.type.scalar();
        int stride_bytes
                = (obj.has_default_stride() ? elem_type.size() : obj.stride);
        int off = to_cpp<int>(obj.off);
        auto stride = (obj.type.elems() - 1) * stride_bytes;
        check_access(obj.buf + off, stride + elem_type.size(), obj);
        ir_visitor_t::_visit(obj);
    }

    void _visit(const store_t &obj) override {
        auto elem_type = obj.value.type().scalar();
        int stride_bytes
                = (obj.has_default_stride() ? elem_type.size() : obj.stride);
        int off = to_cpp<int>(obj.off);
        auto stride = (obj.value.type().elems() - 1) * stride_bytes;
        check_access(obj.buf + off, stride + elem_type.size(), obj);
        ir_visitor_t::_visit(obj);
    }

private:
    void check_access(const expr_t &buf, int size, const object_t &obj) {
        auto &base = (is_var(buf) ? buf : buf.as<ptr_t>().base);
        int off = (is_var(buf) ? 0 : to_cpp<int>(buf.as<ptr_t>().off));
        auto it = buf_sizes_.find(base);
        ir_assert(it != buf_sizes_.end())
                << "Can't find allocation for buffer: " << buf;
        int buf_size = it->second;
        ir_assert(off + size <= buf_size)
                << "Invalid access:\n    " << obj << "\n    Buffer " << base
                << " has size: " << buf_size;
    }

    object_map_t<expr_t, int> buf_sizes_;
};

void verify_buffer_access(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    buffer_access_verifier_t verifier;
    verifier.visit(s);
    trace_pass("verify_buffer_access", s, ir_ctx);
}

expr_t add_grid_guard(
        const expr_t &_cond, const grid_info_t &tg, const grid_info_t &load) {
    auto cond = _cond;
    for (int i = 0; i < tg.ndims(); i++) {
        if (tg[i] == load[i]) continue;
        auto i_cond = (tg.idx(i) < load[i]);
        if (cond.is_empty()) {
            cond = i_cond;
        } else {
            cond = cond & i_cond;
        }
    }
    return cond;
}

stmt_t add_grid_guard(
        const stmt_t &stmt, const grid_info_t &tg, const grid_info_t &load) {
    expr_t cond = add_grid_guard(expr_t(), tg, load);
    if (cond.is_empty()) return stmt;
    return if_t::make(cond, stmt);
}

class compute_builder_t {
public:
    compute_builder_t(const conv_config_t &cfg, ir_context_t &ir_ctx,
            const kernel_info_t &kernel_info)
        : cfg_(cfg)
        , plan_(cfg_.plan())
        , ir_ctx_(ir_ctx)
        , kernel_info_(kernel_info)
        , buf_mgr_(ir_ctx) {
        if (plan_.slm.has_a())
            (void)buf_mgr_.get("a_slm", plan_.slm.a_layout.size());
        if (plan_.slm.has_b())
            (void)buf_mgr_.get("b_slm", plan_.slm.b_layout.size());
    }

    // Setters for original AP/BP/CP buffers (P - problem notation).
    void set_ap_buf(const expr_t &buf) { ap_buf_ = buf; }
    void set_bp_buf(const expr_t &buf) { bp_buf_ = buf; }
    void set_cp_buf(const expr_t &buf) { cp_buf_ = buf; }
    void set_x_reduce_buf(const expr_t &buf) { x_reduce_buf_ = buf; }

    int ab_slm_size() const { return plan_.slm.slm_size(); }

    stmt_t zero_out_stmt() const {
        auto c_entry = buf_mgr_.find("c");
        auto ret = stmt_group_t::make(stmt_label_t::c_zero_out(),
                funcs::zero_out(c_entry.buf, c_entry.size));
        auto x_reduce_entry = buf_mgr_.find("x_reduce", /*allow_empty=*/true);
        if (!x_reduce_entry.is_empty()) {
            ret = ret.append(
                    funcs::zero_out(x_reduce_entry.buf, x_reduce_entry.size));
        }
        return ret;
    }

    stmt_t iter_stmt() const {
        stmt_t stmt;
        bool use_prefetch = !prefetch_stmt_.is_empty();
        bool use_slm = !g2s_load_stmt_.is_empty();
        if (use_prefetch) {
            stmt = stmt.append(stmt_group_t::make(
                    stmt_label_t::prefetch(), prefetch_stmt_));
        } else if (use_slm) {
            stmt = stmt.append(stmt_group_t::make(
                    stmt_label_t::g2s_load(), g2s_load_stmt_));
            stmt = stmt.append(funcs::barrier());
            stmt = stmt.append(stmt_group_t::make(
                    stmt_label_t::g2s_store(), g2s_store_stmt_));
            stmt = stmt.append(funcs::barrier());
        }
        stmt = stmt.append(x2r_mul_stmt_);
        return stmt;
    }

    stmt_t inject_compute_alloc_stmts(const stmt_t &stmt) const {
        return buf_mgr_.inject_allocs(stmt, is_compute_alloc_buf);
    }

    stmt_t inject_out_alloc_stmts(const stmt_t &stmt) const {
        return buf_mgr_.inject_allocs(stmt, is_out_alloc_buf);
    }

    const stmt_t &c_store_stmt() const { return c_store_stmt_; }

    const stmt_t &x_reduced_store_stmt() const { return x_reduce_store_stmt_; }

    void build() {
        build_g2s();
        build_prefetch();
        build_x2r_mul();
        build_c_store();
        build_x_reduce_store();

        // Replace dpas by dpasw when applicable.
        if (cfg_.fma_kind() == fma_kind_t::dpasw) {
            alloc_updater_t alloc_updater;
            inject_dpasw(ir_ctx_.hw(), x2r_mul_stmt_, buf_mgr_.get("c"),
                    c_store_stmt_, alloc_updater,
                    plan_.gemm_schedule.tg_grid().idx(0));
            alloc_updater.update(buf_mgr_);
        }

        // Assign {Atomic} for dpas(w) when applicable.
        x2r_mul_stmt_ = inject_atomic(x2r_mul_stmt_);
    }

private:
    static const expr_t &get_buffer(const stmt_t &s) {
        auto &alloc = s.as<alloc_t>();
        return alloc.buf;
    }

    static bool is_compute_alloc_buf(const expr_t &buf) {
        return !is_out_alloc_buf(buf);
    }

    static bool is_out_alloc_buf(const expr_t &buf) {
        auto &buf_name = buf.as<var_t>().name;
        return utils::one_of(buf_name, "x_reduce", "c");
    }

    void build_g2s() {
        auto &slm = plan_.slm;
        if (slm.has_a()) {
            build_g2s_x("a", ap_buf_, buf_mgr_.get("a_slm"), slm.a_g2s_load,
                    slm.x_reduce_layout, slm.x_reduce, slm.a_reorder,
                    slm.a_g2s_store, slm.a_grid);
        }
        if (slm.has_b()) {
            build_g2s_x("b", bp_buf_, buf_mgr_.get("b_slm"), slm.b_g2s_load,
                    slm.x_reduce_layout, slm.x_reduce, slm.b_reorder,
                    slm.b_g2s_store, slm.b_grid);
        }
    }

    void build_g2s_x(const std::string &prefix, const expr_t &mem_buf,
            const expr_t &slm_buf, const send_plan_t &g2s_load,
            const layout_t &reduce_layout, const reduce_plan_t &g2s_reduce,
            const reorder_plan_t &g2s_reorder, const send_plan_t &g2s_store,
            const grid_info_t &grid) {
        auto g2s_buf = buf_mgr_.get(prefix + "_g2s", g2s_load.reg_buf_size());
        auto load = g2s_load.create_stmt(mem_buf, g2s_buf);
        auto reduce_buf = g2s_reduce
                ? buf_mgr_.get("x_reduce", g2s_reduce.dst_buf_size())
                : expr_t();
        auto store_buf = g2s_reorder
                ? buf_mgr_.get("g2s_tmp", g2s_store.reg_buf_size())
                : g2s_buf;
        if (store_buf.is_same(g2s_buf)) {
            g2s_buf = buf_mgr_.get(prefix + "_g2s", g2s_store.reg_buf_size());
        }
        if (g2s_reorder) {
            g2s_buf = buf_mgr_.get(prefix + "_g2s", g2s_reorder.src.size());
        }
        bool do_reduce = ((cfg_.prb().ab_swap_transpose && prefix == "a")
                || (!cfg_.prb().ab_swap_transpose && prefix == "b"));
        auto reduce = do_reduce
                ? g2s_reduce.create_stmt(g2s_buf, reduce_buf)
                : reduce_plan_t().create_stmt(g2s_buf, expr_t());
        auto reorder = g2s_reorder.create_stmt(g2s_buf, store_buf);
        auto store = g2s_store.create_stmt(slm_buf, store_buf);
        store = reduce.append(reorder).append(store);
        load = add_grid_guard(load, cfg_.thread_group_grid(), grid);
        store = add_grid_guard(store, cfg_.thread_group_grid(), grid);
        g2s_load_stmt_ = g2s_load_stmt_.append(load);
        g2s_store_stmt_ = g2s_store_stmt_.append(store);
    }

    void build_prefetch() {
        auto &prefetch = plan_.prefetch;
        if (prefetch.has_a()) {
            build_prefetch_x(ap_buf_, prefetch.a_prefetch);
        }
        if (prefetch.has_b()) {
            build_prefetch_x(bp_buf_, prefetch.b_prefetch);
        }
    }

    void build_prefetch_x(const expr_t &mem_buf, const send_plan_t &prefetch) {
        prefetch_stmt_ = prefetch_stmt_.append(
                prefetch.create_stmt(mem_buf, expr_t()));
    }

    void build_x2r_mul() {
        auto &x2r = plan_.x2r;
        auto &fma = plan_.fma;
        ir_assert(x2r.split_abc == fma.split_abc);
        ir_assert(x2r.split_factor == fma.split_factor);
        for (int i = 0; i < x2r.split_factor; i++) {
            build_x2r(i);
            stmt_t mul_stmt;
            build_zp_init(i, mul_stmt);
            build_mul(i, mul_stmt);
            build_zp_apply(i, mul_stmt);
            mul_stmt = stmt_group_t::make(stmt_label_t::mul(i), mul_stmt);
            x2r_mul_stmt_ = x2r_mul_stmt_.append(mul_stmt);
        }
    }

    void build_x2r(int subtile_idx) {
        auto &x2r = plan_.x2r;
        auto ax_buf = plan_.slm.has_a() ? buf_mgr_.get("a_slm") : ap_buf_;
        auto bx_buf = plan_.slm.has_b() ? buf_mgr_.get("b_slm") : bp_buf_;
        stmt_t g2r_load_stmt;
        stmt_t s2r_load_stmt;
        build_x2r_x("a", ax_buf, subtile_idx, x2r.a_load, x2r.x_reduce,
                x2r.a_reorder, x2r.a_buf_size(), g2r_load_stmt, s2r_load_stmt);
        build_x2r_x("b", bx_buf, subtile_idx, x2r.b_load, x2r.x_reduce,
                x2r.b_reorder, x2r.b_buf_size(), g2r_load_stmt, s2r_load_stmt);
        g2r_load_stmt = stmt_group_t::make(
                stmt_label_t::g2r_load(subtile_idx), g2r_load_stmt);
        s2r_load_stmt = stmt_group_t::make(
                stmt_label_t::s2r_load(subtile_idx), s2r_load_stmt);
        x2r_mul_stmt_ = x2r_mul_stmt_.append(g2r_load_stmt);
        x2r_mul_stmt_ = x2r_mul_stmt_.append(s2r_load_stmt);
    }

    void build_zp_init(int subtile_idx, stmt_t &mul_stmt) {
        auto &zp = plan_.zp;
        if (!zp) return;

        auto zp_mem_buf = kernel_info_.find_arg("src_zero_points");
        auto zp_buf = buf_mgr_.get("src_zp", zp.load_reg_buf_size());
        auto wei_buf = buf_mgr_.get("b");
        auto zp_mask_buf = buf_mgr_.get("zp_mask", zp.mask_reg_buf_size());
        auto zp_comp_buf = buf_mgr_.get("zp_comp", zp.comp_reg_buf_size());
        auto load = zp.load_create_stmt(zp_mem_buf, zp_buf, subtile_idx);
        auto zp_mask_init = zp.mask_init_create_stmt(zp_mask_buf, subtile_idx);
        auto zp_comp_init = zp.comp_init_create_stmt(
                buf_mgr_, zp_buf, wei_buf, zp_comp_buf, subtile_idx);
        mul_stmt = mul_stmt.append(load);
        mul_stmt = mul_stmt.append(zp_mask_init);
        mul_stmt = mul_stmt.append(zp_comp_init);
    }

    void build_zp_apply(int subtile_idx, stmt_t &mul_stmt) {
        auto &zp = plan_.zp;
        if (!zp) return;

        auto c_buf = buf_mgr_.get("c");
        auto zp_comp_buf = buf_mgr_.get("zp_comp");
        auto zp_mask_buf = buf_mgr_.get("zp_mask");
        auto zp_comp_apply = zp.comp_apply_create_stmt(
                zp_comp_buf, zp_mask_buf, c_buf, subtile_idx);
        mul_stmt = mul_stmt.append(zp_comp_apply);
    }

    void build_x2r_x(const std::string &prefix, const expr_t &x_buf,
            int subtile_idx, const send_plan_t &x2r_load,
            const reduce_plan_t &x2r_reduce, const reorder_plan_t &x2r_reorder,
            int buf_size, stmt_t &g2r_load_stmt, stmt_t &s2r_load_stmt) {
        if (subtile_idx > 0 && x2r_load.split_factor() == 1) return;
        auto reg_buf = buf_mgr_.get(prefix, buf_size);
        auto load_buf = x2r_reorder ? buf_mgr_.get("x2r_tmp",
                                std::max(x2r_load.reg_buf_size(),
                                        x2r_reorder.src_buf_size()))
                                    : reg_buf;
        if (load_buf.is_same(reg_buf)) {
            reg_buf = buf_mgr_.get(prefix, x2r_load.reg_buf_size());
        }
        auto reduce_buf = x2r_reduce
                ? buf_mgr_.get("x_reduce", x2r_reduce.dst_buf_size())
                : expr_t();
        auto load = x2r_load.create_stmt(x_buf, load_buf, subtile_idx);
        bool do_reduce = ((cfg_.prb().ab_swap_transpose && prefix == "a")
                || (!cfg_.prb().ab_swap_transpose && prefix == "b"));
        auto reduce = do_reduce
                ? x2r_reduce.create_stmt(load_buf, reduce_buf)
                : reduce_plan_t().create_stmt(reg_buf, expr_t());
        auto reorder = x2r_reorder.create_stmt(load_buf, reg_buf);
        auto &load_stmt = x2r_load.send_params().is_slm() ? s2r_load_stmt
                                                          : g2r_load_stmt;
        load_stmt = load_stmt.append(load);
        load_stmt = load_stmt.append(reduce);
        load_stmt = load_stmt.append(reorder);
    }

    void build_mul(int subtile_idx, stmt_t &mul_stmt) {
        auto &fma = plan_.fma;
        auto &a_layout = fma.a_layout;
        auto &b_layout = fma.b_layout;
        auto &c_layout = fma.c_layout;
        auto a_buf = buf_mgr_.get("a");
        auto b_buf = buf_mgr_.get("b");
        auto c_buf = buf_mgr_.get("c", c_layout.size());
        int b0 = fma.bmnk_start_idx(bmnk_kind_t::b, subtile_idx);
        int b1 = fma.bmnk_stop_idx(bmnk_kind_t::b, subtile_idx);
        int m0 = fma.bmnk_start_idx(bmnk_kind_t::m, subtile_idx);
        int m1 = fma.bmnk_stop_idx(bmnk_kind_t::m, subtile_idx);
        int n0 = fma.bmnk_start_idx(bmnk_kind_t::n, subtile_idx);
        int n1 = fma.bmnk_stop_idx(bmnk_kind_t::n, subtile_idx);
        int k0 = fma.bmnk_start_idx(bmnk_kind_t::k, subtile_idx);
        int k1 = fma.bmnk_stop_idx(bmnk_kind_t::k, subtile_idx);

        std::vector<int> a_idx(3);
        std::vector<int> b_idx(3);
        std::vector<int> c_idx(3);

        auto fma_funcs = create_fma_funcs(fma);

        stmt_t stmt;
        for (int b = b0; b < b1; b += fma.b_blk) {
            a_idx[0] = b_idx[0] = c_idx[0] = b;
            for (int k = k0; k < k1; k += fma.k_blk) {
                a_idx[2] = b_idx[1] = k;
                for (int n = n0; n < n1; n += fma.n_blk) {
                    b_idx[2] = c_idx[2] = n;
                    for (int m = m0; m < m1; m += fma.m_blk) {
                        a_idx[1] = c_idx[1] = m;
                        int a_off = a_layout.offset_in_bytes(a_idx);
                        int b_off = b_layout.offset_in_bytes(b_idx);
                        int c_off = c_layout.offset_in_bytes(c_idx);
                        a_off = a_off % fma.a_buf_size();
                        b_off = b_off % fma.b_buf_size();
                        stmt = stmt.append(create_fma_block(fma_funcs,
                                a_buf[a_off], b_buf[b_off], c_buf[c_off]));
                    }
                }
            }
        }
        mul_stmt = mul_stmt.append(stmt);
    }

    void build_c_store() {
        auto &gemm_schedule = plan_.gemm_schedule;
        conv_post_op_view_mapper_t view_mapper(gemm_schedule, cfg_.prb());
        post_op_context_t post_op_ctx(*cfg_.prb().attr, cfg_.zp_cfg(),
                gemm_schedule, kernel_info_, *cfg_.prb().conv_pd->dst_md(),
                cfg_.prb().c_md(), view_mapper);
        auto c_buf = buf_mgr_.find("c").buf;
        auto c_thr_reg_layout = plan_.fma.c_prb_layout;
        auto thr_tile = gemm_schedule.c_thr_tile(/*is_relative=*/false);
        expr_t reduce_cond;
        if (gemm_schedule.with_thread_group_k_slicing()) {
            slm_reduce_builder_t slm_reduce_builder(ir_ctx_,
                    gemm_schedule.tg_grid(), c_buf, c_thr_reg_layout, thr_tile);
            c_store_stmt_ = c_store_stmt_.append(slm_reduce_builder.stmt());
            c_thr_reg_layout = slm_reduce_builder.reg_layout();
            thr_tile = slm_reduce_builder.thr_tile();
            reduce_cond = slm_reduce_builder.reduce_cond();
        }

        // Always perform reorder when dpasw is used. This is to ensure
        // that C is properly restored and permuted after dpasw.
        bool force_c_reorder = cfg_.fma_kind() == fma_kind_t::dpasw;

        int c_buf_size = 0;
        auto stmt = create_epilogue_stmt(cfg_.exec_cfg(), ir_ctx_,
                gemm_schedule, force_c_reorder, post_op_ctx, thr_tile,
                c_thr_reg_layout, cp_buf_, c_buf, c_buf_size);
        (void)buf_mgr_.get("c", c_buf_size);
        if (!reduce_cond.is_empty()) stmt = if_t::make(reduce_cond, stmt);
        c_store_stmt_ = c_store_stmt_.append(stmt);
    }

    expr_t get_x_reduce_store_condition() {
        auto &gemm_schedule = plan_.gemm_schedule;
        auto &c_view = gemm_schedule.c_view();
        auto &kd = c_view.vvar("kd");
        auto &kh = c_view.vvar("kh");
        auto &kw = c_view.vvar("kw");
        auto &ic = c_view.vvar("ic");
        expr_t cond(true);
        cond &= (kd == 0);
        cond &= (kh == 0);
        cond &= (kw == 0);
        cond &= (ic == 0);
        loop_kind_t filter = loop_kind_t::kernel_grid;
        if (!plan_.slm.has_b()) filter = filter | loop_kind_t::tg_grid;
        cond = gemm_schedule.expand(cond, /*expand_trivial_vars=*/true, filter);
        if (plan_.slm.has_b()) {
            cond = add_grid_guard(
                    cond, cfg_.thread_group_grid(), plan_.slm.b_grid);
        }
        return cond;
    }

    void build_x_reduce_store() {
        auto &gemm_schedule = plan_.gemm_schedule;
        bool use_atomic
                = gemm_schedule.with_kernel_grid_k_slicing() || cfg_.slm().b();
        auto x_reduce_buf = buf_mgr_.find("x_reduce", /*allow_empty=*/true).buf;
        if (x_reduce_buf.is_empty()) return;
        auto x_reduce_view
                = plan_.bia_view.create_sub_view(plan_.x_reduce_tile());
        auto r2g = make_access_builder(ir_ctx_, x_reduce_view, x_reduce_buf_,
                x_reduce_buf,
                use_atomic ? send_op_t::atomic_fadd : send_op_t::store,
                send_address_t::a64);
        auto cond = get_x_reduce_store_condition();
        x_reduce_store_stmt_ = if_t::make(cond, r2g.stmt());
    }

    std::vector<func_t> create_fma_funcs(const fma_plan_t &fma) const {
        auto &a = fma.a_layout;
        auto &b = fma.b_layout;
        auto &c = fma.c_layout;
        std::vector<func_t> ret;
        switch (fma.fma_kind) {
            case fma_kind_t::mad: {
                int simd = fma.max_bmn_blk();
                int a_stride = fma.is_a_broadcast() ? 0 : (int)a.inner_stride();
                int b_stride = fma.is_b_broadcast() ? 0 : (int)b.inner_stride();
                auto mad = mad_t::make(ir_ctx_.hw(), c.type(), simd, a.type(),
                        a_stride, b.type(), b_stride);
                ret.push_back(mad);
                break;
            }
            case fma_kind_t::dp4a:
            case fma_kind_t::dpas:
            case fma_kind_t::dpasw: {
                const int max_rcount = 8;
                int block_rcount = fma.m_blk;
                int simd = fma.n_blk;
                int sdepth
                        = ir_utils::safe_divide(fma.k_blk * a.type().size(), 4);
                for (int r = 0; r < block_rcount;) {
                    int rcount = std::min(max_rcount, block_rcount - r);
                    auto dpas = dpas_t::make(/*is_dpasw=*/false, simd, sdepth,
                            rcount, c.type(), b.type(), a.type());
                    ret.push_back(dpas);
                    r += rcount;
                }
                break;
            }
            default: ir_error_not_expected();
        }
        return ret;
    }

    stmt_t create_fma_block(const std::vector<func_t> &fmas, const expr_t &a,
            const expr_t &b, const expr_t &c) const {
        bool is_dpas = fmas[0].is<dpas_t>();
        auto src1 = a;
        auto src2 = b;
        auto dst = c;
        if (is_dpas) std::swap(src1, src2);
        if (!is_dpas) ir_assert(fmas.size() == 1);
        stmt_t ret;
        for (auto &f : fmas) {
            ret = ret.append(f.call({dst, dst, src1, src2}));
            auto *dpas = f.as_ptr<dpas_t>();
            if (is_dpas) {
                src2 += dpas->src2_size();
                dst += dpas->dst_size();
            }
        }
        return ret;
    }

    const conv_config_t &cfg_;
    const conv_plan_t &plan_;
    ir_context_t &ir_ctx_;
    const kernel_info_t &kernel_info_;

    stmt_t g2s_load_stmt_;
    stmt_t g2s_store_stmt_;
    stmt_t prefetch_stmt_;
    stmt_t x2r_mul_stmt_;
    stmt_t c_store_stmt_;
    stmt_t x_reduce_store_stmt_;

    buffer_manager_t buf_mgr_;

    expr_t ap_buf_;
    expr_t bp_buf_;
    expr_t cp_buf_;
    expr_t x_reduce_buf_;
};

class compute_loop_label_injector_t : public ir_mutator_t {
public:
    object_t _mutate(const for_t &obj) override {
        if (injected_) return obj;

        bool found_dynamic_init = false;
        auto calls = find_objects<func_call_t>(obj);
        auto loops = find_objects<for_t>(obj);
        for (auto &_f : loops) {
            auto &f = _f.as<for_t>();
            if (!is_const(f.init)) found_dynamic_init = true;
        }
        if (!found_dynamic_init) {
            injected_ = true;
            return stmt_group_t::make(stmt_label_t::compute_loop(), obj);
        }
        return ir_mutator_t::_mutate(obj);
    }

private:
    bool injected_ = false;
};

// Injects compute_loop statement label to the outermost loop that can be
// pipelined. If a loop contains a "continue" function call it can't be
// pipelined because of conditional flow.
stmt_t inject_compute_loop_label(const stmt_t &s) {
    return compute_loop_label_injector_t().mutate(s);
}

void conv_ir_builder_t::build() {
    auto &prb = cfg_.prb();

    trace_reset();

    std::vector<stmt_t> init_stmts;
    auto &plan = cfg_.plan();
    auto gemm_schedule = plan.gemm_schedule;
    auto init_cset = plan.init_cset;
    init_kernel_grid(cfg_.kernel_grid(), cfg_.thread_group_grid(), cfg_.simd(),
            init_cset, init_stmts);

    // Initialize memory buffers.
    std::vector<stmt_t> inner_lets;

    view_t a_view;
    view_t b_view;
    view_t c_view;
    view_t bp_reduced_view;

    expr_t ap_buf = kernel_info_.find_arg(prb.ab_swap_transpose
                    ? (prb.is_bwd_w ? "dst" : "wei")
                    : (prb.is_bwd_d ? "dst" : "src"));
    expr_t bp_buf = kernel_info_.find_arg(
            (prb.ab_swap_transpose ? (prb.is_bwd_d ? "dst" : "src")
                                   : (prb.is_bwd_w ? "dst" : "wei")));
    expr_t cp_buf = kernel_info_.find_arg(
            prb.is_fwd ? "dst" : (prb.is_bwd_d ? "src" : "wei"));
    expr_t x_reduced_mem_buf
            = kernel_info_.find_arg("bia", /*allow_empty=*/true);
    expr_t b_reduction_condition;

    trace_stamp("GEMM Schedule");

    ir_context_t ir_ctx(cfg_.exec_cfg(), init_cset);
    compute_builder_t cb(cfg_, ir_ctx, kernel_info_);
    cb.set_ap_buf(ap_buf);
    cb.set_bp_buf(bp_buf);
    cb.set_cp_buf(cp_buf);
    cb.set_x_reduce_buf(x_reduced_mem_buf);
    cb.build();

    trace_stamp("Compute Builder");

    std::vector<stmt_t> allocs;
    for (int i = 0; i < kernel_info_.nargs(); i++) {
        auto &var = kernel_info_.arg_var(i);
        if (!var.type().is_ptr()) continue;
        allocs.push_back(alloc_t::make(var, 0, alloc_kind_t::global));
    }

    // Create IR statements.
    stmt_t loop_stmt = cb.iter_stmt();
    loop_stmt = gemm_schedule.create_loop_nest(loop_stmt);
    loop_stmt = inject_compute_loop_label(loop_stmt);
    loop_stmt = cb.inject_compute_alloc_stmts(loop_stmt);

    stmt_t c_store_stmt;
    c_store_stmt = c_store_stmt.append(cb.x_reduced_store_stmt());
    c_store_stmt = c_store_stmt.append(cb.c_store_stmt());
    c_store_stmt = stmt_group_t::make(stmt_label_t::c_store(), c_store_stmt);

    stmt_ = loop_stmt;
    stmt_ = stmt_seq_t::make(cb.zero_out_stmt(), stmt_);
    stmt_ = stmt_seq_t::make(stmt_, c_store_stmt);

    stmt_ = cb.inject_out_alloc_stmts(stmt_);

    stmt_ = gemm_schedule.create_bind_stmt(stmt_);
    stmt_ = inject_let_stmts(stmt_, init_stmts);
    stmt_ = inject_alloc_stmts(stmt_, allocs);
    trace_stop("Create Inital IR");

    stmt_ = inject_external_var_let(stmt_, ir_ctx);
    stmt_ = merge_slm_buffers(stmt_, ir_ctx);
    if (!cfg_.pipeline().do_unroll() && cfg_.slm()) {
        stmt_ = inject_simple_slm_buffering(
                stmt_, ir_ctx, cfg_, cb.ab_slm_size());
    } else if (!cfg_.pipeline().do_unroll() && cfg_.prefetch()) {
        // Simplify to remove loops with only 1 iteration
        stmt_ = simplify(stmt_, ir_ctx);
        stmt_ = inject_prefetch_pipeline(stmt_, ir_ctx, cfg_);
    }
    stmt_ = inject_slm_reorder(stmt_, ir_ctx, cfg_.thread_group_grid(),
            cfg_.slm() || gemm_schedule.with_thread_group_k_slicing());
    stmt_ = lift_buffer_offsets_in_send(stmt_, ir_ctx);
    stmt_ = simplify(stmt_, ir_ctx);
    stmt_ = inject_send(stmt_, ir_ctx);
    stmt_ = split_wide_stores(stmt_, ir_ctx);
    stmt_ = lift_alloc(stmt_, ir_ctx, cfg_.pipeline().reuse_headers());
    stmt_ = lift_send_2d_header_store(stmt_, ir_ctx);
    stmt_ = hoist_send_masks(stmt_, ir_ctx, stmt_label_t::c_store(), false,
            cfg_.reserved_regs());
    stmt_ = split_shuffle(stmt_, ir_ctx);
    stmt_ = fixup_if_conditions(stmt_, ir_ctx);
    stmt_ = optimize_int64_exprs(stmt_, ir_ctx);
    stmt_ = fix_int32_overflow(stmt_, ir_ctx);
    stmt_ = eliminate_common_subexprs(
            stmt_, ir_ctx, cfg_.reserved_regs(), cfg_.slm().gmem_bufs());
    stmt_ = hoist_exprs(stmt_, ir_ctx, cfg_.reserved_regs());
    if (cfg_.pipeline().do_unroll())
        stmt_ = loop_strength_reduce(stmt_, ir_ctx);
    stmt_ = optimize_alloc_let(stmt_, ir_ctx);
    if (cfg_.pipeline().do_unroll()) {
        stmt_ = update_loops_for_unrolling(stmt_, ir_ctx);
        stmt_ = inject_unrolling(stmt_, ir_ctx, cfg_, cb.ab_slm_size());
    }
    stmt_ = hoist_send_masks(stmt_, ir_ctx, stmt_label_t::compute_loop(), true,
            cfg_.reserved_regs());
    stmt_ = unroll_loops(stmt_, ir_ctx);
    stmt_ = simplify(stmt_, ir_ctx);
    stmt_ = optimize_alloc_let(stmt_, ir_ctx);

    if (cfg_.hw() > ngen::HW::XeLP) stmt_ = optimize_peephole(stmt_, ir_ctx);
    stmt_ = optimize_barrier(stmt_, ir_ctx);
    if (cfg_.fma_kind() == fma_kind_t::dp4a) stmt_ = inject_dp4a(stmt_, ir_ctx);
    stmt_ = inject_bank_conflict_attribute(stmt_, ir_ctx);
    stmt_ = stmt_group_t::make(stmt_label_t::kernel(), stmt_);

#if !defined(NDEBUG) || defined(DNNL_DEV_MODE)
    verify_buffer_access(stmt_, ir_ctx);
#endif

    ir_trace() << "Convolution kernel body:\n" << stmt_ << std::endl;
    trace_perf();
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
