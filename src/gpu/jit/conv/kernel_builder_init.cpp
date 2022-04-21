/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "gpu/jit/conv/kernel_builder.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

namespace {

bool need_src_or_dst_check(
        bool is_fwd, int o, int i, int k, int p, int s, int d) {
    if (is_fwd) {
        int i_min = -p;
        int i_max = (o - 1) * s - p + (k - 1) * (1 + d);
        return (i_min < 0) || (i_max >= i);
    }
    // Backward.
    int os_min = p - (k - 1) * (1 + d);
    int os_max = (i - 1) + p;
    return (os_min < 0) || (os_max >= o * s);
}

} // namespace

void kernel_builder_t::init_fwd(gemm_schedule_t &gemm_schedule,
        view_t &src_view, view_t &wei_view, view_t &dst_view, expr_t &src_buf,
        expr_t &wei_buf, expr_t &dst_buf) {
    auto &src_layout = cfg_.src_layout;
    auto &wei_layout = cfg_.wei_layout;
    auto &dst_layout = cfg_.dst_layout;

    // Initialize views.
    auto mb = var_t::make(type_t::s32(), "mb");
    auto ic = var_t::make(type_t::s32(), "ic");
    auto oc = var_t::make(type_t::s32(), "oc");
    auto kd = var_t::make(type_t::s32(), "kd");
    auto kh = var_t::make(type_t::s32(), "kh");
    auto kw = var_t::make(type_t::s32(), "kw");
    auto g = var_t::make(type_t::s32(), "g");

    expr_t ow, oh, od, osp;
    bool check_od = false;
    bool check_oh = false;
    bool check_ow = false;
    if (cfg_.fuse_spatial) {
        osp = var_t::make(type_t::s32(), "osp");
        ow = osp;
        oh = osp / cfg_.ow;
        od = osp / (cfg_.oh * cfg_.ow);

        bool is_1d = (cfg_.oh == 1 && cfg_.od == 1);
        bool is_2d = (cfg_.oh != 1 && cfg_.od == 1);
        bool is_3d = !is_1d && !is_2d;

        bool check_osp = (cfg_.osp % cfg_.osp_tg_blk != 0);
        check_ow = is_1d && check_osp;
        check_oh = is_2d && check_osp;
        check_od = is_3d && check_osp;

        if (!is_1d) ow %= cfg_.ow;
        if (!is_2d) oh %= cfg_.oh;
    } else {
        od = var_t::make(type_t::s32(), "od");
        oh = var_t::make(type_t::s32(), "oh");
        ow = var_t::make(type_t::s32(), "ow");
        check_ow = (cfg_.ow < cfg_.padded_dim("ow"));
    }

    // Initialize masks.
    expr_t id_mask, ih_mask, iw_mask;
    expr_t od_mask, oh_mask, ow_mask;

    bool check_kw = (cfg_.kw < cfg_.padded_dim("kw"));
    bool check_iw = check_kw || check_ow
            || need_src_or_dst_check(cfg_.is_fwd, cfg_.ow, cfg_.iw, cfg_.kw,
                    cfg_.pw, cfg_.sw, cfg_.dw);
    bool check_ih = check_oh
            || need_src_or_dst_check(cfg_.is_fwd, cfg_.oh, cfg_.ih, cfg_.kh,
                    cfg_.ph, cfg_.sh, cfg_.dh);
    bool check_id = check_od
            || need_src_or_dst_check(cfg_.is_fwd, cfg_.od, cfg_.id, cfg_.kd,
                    cfg_.pd, cfg_.sd, cfg_.dd);

    auto &x = view_t::placeholder_var();
    if (check_id) id_mask = (x >= 0) & (x < cfg_.id);
    if (check_ih) ih_mask = (x >= 0) & (x < cfg_.ih);
    if (check_iw) iw_mask = (x >= 0) & (x < cfg_.iw);
    if (check_od) od_mask = (x >= 0) & (x < cfg_.od);
    if (check_oh) oh_mask = (x >= 0) & (x < cfg_.oh);
    if (check_ow) ow_mask = (x >= 0) & (x < cfg_.ow);

    // Source.
    if (cfg_.fuse_spatial) {
        src_view = view_t({mb, g, ic, osp, kd, kh, kw}, 6);
    } else {
        src_view = view_t({mb, g, ic, od, oh, ow, kd, kh, kw}, 6);
    }
    src_view.set_vdim(mb, cfg_.mb);
    src_view.set_vdim(g, cfg_.g);
    src_view.set_vdim(ic, cfg_.ic);
    if (cfg_.fuse_spatial) {
        src_view.set_vdim(osp, cfg_.osp);
    } else {
        src_view.set_vdim(od, cfg_.od);
        src_view.set_vdim(oh, cfg_.oh);
        src_view.set_vdim(ow, cfg_.ow);
    }
    src_view.set_vdim(kd, cfg_.kd);
    src_view.set_vdim(kh, cfg_.kh);
    src_view.set_vdim(kw, cfg_.kw);
    src_view.set_tdim(0, mb);
    src_view.set_tdim(1, g);
    src_view.set_tdim(2, ic);
    src_view.set_tdim(3, od * cfg_.sd - cfg_.pd + kd * (1 + cfg_.dd), id_mask);
    src_view.set_tdim(4, oh * cfg_.sh - cfg_.ph + kh * (1 + cfg_.dh), ih_mask);
    src_view.set_tdim(5, ow * cfg_.sw - cfg_.pw + kw * (1 + cfg_.dw), iw_mask);
    src_view.set_tlayout(src_layout);
    src_view.set_tmasks(cfg_.padded_dims(), cfg_.dim_blocks());

    // Weights.
    wei_view = view_t({g, oc, ic, kd, kh, kw}, 6);
    wei_view.set_vdim(g, cfg_.g);
    wei_view.set_vdim(oc, cfg_.oc);
    wei_view.set_vdim(ic, cfg_.ic);
    wei_view.set_vdim(kd, cfg_.kd);
    wei_view.set_vdim(kh, cfg_.kh);
    wei_view.set_vdim(kw, cfg_.kw);
    wei_view.set_tdim(0, g);
    wei_view.set_tdim(1, oc);
    wei_view.set_tdim(2, ic);
    wei_view.set_tdim(3, kd);
    wei_view.set_tdim(4, kh);
    wei_view.set_tdim(5, kw);
    wei_view.set_tlayout(wei_layout);
    wei_view.set_tmasks(cfg_.padded_dims(), cfg_.dim_blocks());

    // Destination.
    if (cfg_.fuse_spatial) {
        dst_view = view_t({mb, g, oc, osp}, 6);
    } else {
        dst_view = view_t({mb, g, oc, od, oh, ow}, 6);
    }
    dst_view.set_vdim(mb, cfg_.mb);
    dst_view.set_vdim(g, cfg_.g);
    dst_view.set_vdim(oc, cfg_.oc);
    if (cfg_.fuse_spatial) {
        dst_view.set_vdim(osp, cfg_.osp);
    } else {
        dst_view.set_vdim(od, cfg_.od);
        dst_view.set_vdim(oh, cfg_.oh);
        dst_view.set_vdim(ow, cfg_.ow);
    }
    dst_view.set_tdim(0, mb);
    dst_view.set_tdim(1, g);
    dst_view.set_tdim(2, oc);
    dst_view.set_tdim(3, od, od_mask);
    dst_view.set_tdim(4, oh, oh_mask);
    dst_view.set_tdim(5, ow, ow_mask);
    dst_view.set_tlayout(dst_layout);
    dst_view.set_tmasks(cfg_.padded_dims(), cfg_.dim_blocks());

    // Initialize GEMM schedule.
    gemm_schedule.set_a_view(src_view);
    gemm_schedule.set_b_view(wei_view);
    gemm_schedule.set_c_view(dst_view);
    gemm_schedule.set_b_vars({g});
    if (cfg_.fuse_spatial) {
        gemm_schedule.set_m_vars({mb, osp});
    } else {
        gemm_schedule.set_m_vars({mb, od, oh, ow});
    }
    gemm_schedule.set_n_vars({oc});
    gemm_schedule.set_k_vars({ic, kd, kh, kw});

    gemm_schedule.for_each_var([&](const expr_t &var) {
        int bound = cfg_.padded_dim(var.as<var_t>().name);
        gemm_schedule.set_var_bound(var, bound);
    });

    expr_t g_tg_blk_idx, g_inner;
    expr_t oc_tg_blk_idx, oc_thr_blk_idx, oc_inner;
    expr_t mb_tg_blk_idx, mb_thr_blk_idx, mb_inner;
    expr_t osp_tg_blk_idx, osp_thr_blk_idx, osp_inner;
    expr_t kw_outer, kw_inner;
    expr_t ic_thr_blk_idx, ic_outer, ic_inner;

    gemm_schedule.split(g, cfg_.g_tg_blk, g_tg_blk_idx, g_inner);
    gemm_schedule.split(oc, cfg_.oc_tg_blk, cfg_.oc_thr_blk, oc_tg_blk_idx,
            oc_thr_blk_idx, oc_inner);
    gemm_schedule.split(mb, cfg_.mb_tg_blk, cfg_.mb_thr_blk, mb_tg_blk_idx,
            mb_thr_blk_idx, mb_inner);
    gemm_schedule.split(osp.is_empty() ? ow : osp, cfg_.osp_tg_blk,
            cfg_.osp_thr_blk, osp_tg_blk_idx, osp_thr_blk_idx, osp_inner);
    gemm_schedule.split(ic, cfg_.ic_blk * cfg_.ic_thr_dim, cfg_.ic_blk,
            ic_outer, ic_thr_blk_idx, ic_inner);
    gemm_schedule.split(kw, cfg_.kw_blk, kw_outer, kw_inner);

    auto g_osp_idx = cfg_.fuse_spatial
            ? gemm_schedule.fuse({g_tg_blk_idx, osp_tg_blk_idx})
            : gemm_schedule.fuse({g_tg_blk_idx, od, oh, osp_tg_blk_idx});
    auto mb_osp_thr_blk_idx
            = gemm_schedule.fuse(mb_thr_blk_idx, osp_thr_blk_idx);

    gemm_schedule.bind(oc_tg_blk_idx, kernel_grid_.idx(0));
    gemm_schedule.bind(g_osp_idx, kernel_grid_.idx(1));
    gemm_schedule.bind(mb_tg_blk_idx, kernel_grid_.idx(2));
    gemm_schedule.bind(oc_thr_blk_idx, tg_grid_.idx(0));
    gemm_schedule.bind(mb_osp_thr_blk_idx, tg_grid_.idx(1));
    gemm_schedule.bind(ic_thr_blk_idx, tg_grid_.idx(2));

    gemm_schedule.tensorize(g_inner);
    gemm_schedule.tensorize(oc_inner);
    gemm_schedule.tensorize(mb_inner);
    gemm_schedule.tensorize(osp_inner);
    gemm_schedule.tensorize(kw_inner);
    gemm_schedule.tensorize(ic_inner);

    gemm_schedule.reorder({ic_outer, kd, kh, kw_outer, oc_thr_blk_idx,
            mb_osp_thr_blk_idx, ic_thr_blk_idx});

    src_buf = kernel_info_.find_arg("src");
    wei_buf = kernel_info_.find_arg("wei");
    dst_buf = kernel_info_.find_arg("dst");
}

void kernel_builder_t::init_bwd_d(gemm_schedule_t &gemm_schedule,
        view_t &dst_view, view_t &wei_view, view_t &src_view, expr_t &dst_buf,
        expr_t &wei_buf, expr_t &src_buf) {
    auto &src_layout = cfg_.src_layout;
    auto &wei_layout = cfg_.wei_layout;
    auto &dst_layout = cfg_.dst_layout;

    // Initialize views.
    auto g = var_t::make(type_t::s32(), "g");
    auto mb = var_t::make(type_t::s32(), "mb");
    auto ic = var_t::make(type_t::s32(), "ic");
    auto oc = var_t::make(type_t::s32(), "oc");
    auto id = var_t::make(type_t::s32(), "id");
    auto ih = var_t::make(type_t::s32(), "ih");
    auto iw = var_t::make(type_t::s32(), "iw");
    auto kd = var_t::make(type_t::s32(), "kd");
    auto kh = var_t::make(type_t::s32(), "kh");
    auto kw = var_t::make(type_t::s32(), "kw");

    // Initialize masks.
    expr_t od_mask(true), oh_mask(true), ow_mask(true);

    bool check_iw = (cfg_.iw < cfg_.padded_dim("iw"));
    bool check_ow = check_iw
            || need_src_or_dst_check(cfg_.is_fwd, cfg_.ow, cfg_.iw, cfg_.kw,
                    cfg_.pw, cfg_.sw, cfg_.dw);
    bool check_oh = need_src_or_dst_check(
            cfg_.is_fwd, cfg_.oh, cfg_.ih, cfg_.kh, cfg_.ph, cfg_.sh, cfg_.dh);
    bool check_od = need_src_or_dst_check(
            cfg_.is_fwd, cfg_.od, cfg_.id, cfg_.kd, cfg_.pd, cfg_.sd, cfg_.dd);

    auto &x = view_t::placeholder_var();
    if (check_od) od_mask = (x >= 0) & (x < cfg_.od);
    if (check_oh) oh_mask = (x >= 0) & (x < cfg_.oh);
    if (check_ow) ow_mask = (x >= 0) & (x < cfg_.ow);

    std::function<expr_t(const expr_t &)> iw_mapping;
    if (cfg_.bwd_d_optimize_strided_iw) {
        // Apply mapping to iw to ensure each thread group has the same
        // stride condition when evaluating skip conditions.
        iw_mapping = [&](const expr_t &e) {
            int iw_bound = utils::rnd_up(cfg_.iw, cfg_.iw_tg_blk);
            ir_assert(iw_bound % cfg_.iw_thr_blk == 0);
            int iw_same_mod_blk = ir_utils::safe_divide(iw_bound, cfg_.sw);
            return (e % iw_same_mod_blk) * cfg_.sw + (e / iw_same_mod_blk);
        };
    } else {
        iw_mapping = [](const expr_t &e) { return e; };
    }

    // Destination.
    dst_view = view_t({mb, g, oc, id, ih, iw, kd, kh, kw}, 6);
    dst_view.set_vdim(mb, cfg_.mb);
    dst_view.set_vdim(g, cfg_.g);
    dst_view.set_vdim(oc, cfg_.oc);
    dst_view.set_vdim(id, cfg_.id);
    dst_view.set_vdim(ih, cfg_.ih);
    dst_view.set_vdim(iw, cfg_.iw);
    dst_view.set_vdim(kd, cfg_.kd);
    dst_view.set_vdim(kh, cfg_.kh);
    dst_view.set_vdim(kw, cfg_.kw);
    dst_view.set_tdim(0, mb);
    dst_view.set_tdim(1, g);
    dst_view.set_tdim(2, oc);

    auto od = id - kd * (1 + cfg_.dd) + cfg_.pd;
    auto oh = ih - kh * (1 + cfg_.dh) + cfg_.ph;
    auto ow = iw_mapping(iw) - kw * (1 + cfg_.dw) + cfg_.pw;

    // When stride optimization is enabled, stride conditions are handled by
    // continue calls in the outer loops.
    if (!cfg_.bwd_d_optimize_strided_iw) {
        od_mask &= (od % cfg_.sd == 0);
        oh_mask &= (oh % cfg_.sh == 0);
        ow_mask &= (ow % cfg_.sw == 0);
    }
    dst_view.set_tdim(3, od / cfg_.sd, od_mask);
    dst_view.set_tdim(4, oh / cfg_.sh, oh_mask);
    dst_view.set_tdim(5, ow / cfg_.sw, ow_mask);

    dst_view.set_tlayout(dst_layout);
    dst_view.set_tmasks(cfg_.padded_dims(), cfg_.dim_blocks());

    // Weights.
    wei_view = view_t({g, oc, ic, kd, kh, kw}, 6);
    wei_view.set_vdim(g, cfg_.g);
    wei_view.set_vdim(ic, cfg_.ic);
    wei_view.set_vdim(oc, cfg_.oc);
    wei_view.set_vdim(kd, cfg_.kd);
    wei_view.set_vdim(kh, cfg_.kh);
    wei_view.set_vdim(kw, cfg_.kw);
    wei_view.set_tdim(0, g);
    wei_view.set_tdim(1, oc);
    wei_view.set_tdim(2, ic);
    wei_view.set_tdim(3, kd);
    wei_view.set_tdim(4, kh);
    wei_view.set_tdim(5, kw);
    wei_view.set_tlayout(wei_layout);
    wei_view.set_tmasks(cfg_.padded_dims(), cfg_.dim_blocks());

    // Source.
    src_view = view_t({mb, g, ic, id, ih, iw}, 6);
    src_view.set_vdim(mb, cfg_.mb);
    src_view.set_vdim(g, cfg_.g);
    src_view.set_vdim(ic, cfg_.ic);
    src_view.set_vdim(id, cfg_.id);
    src_view.set_vdim(ih, cfg_.ih);
    src_view.set_vdim(iw, cfg_.iw);
    src_view.set_tdim(0, mb);
    src_view.set_tdim(1, g);
    src_view.set_tdim(2, ic);
    src_view.set_tdim(3, id);
    src_view.set_tdim(4, ih);
    src_view.set_tdim(5, iw_mapping(iw));
    src_view.set_tlayout(src_layout);
    src_view.set_tmasks(cfg_.padded_dims(), cfg_.dim_blocks());

    // Initialize GEMM schedule.
    gemm_schedule.set_a_view(dst_view);
    gemm_schedule.set_b_view(wei_view);
    gemm_schedule.set_c_view(src_view);
    gemm_schedule.set_b_vars({g});
    gemm_schedule.set_m_vars({mb, id, ih, iw});
    gemm_schedule.set_n_vars({ic});
    gemm_schedule.set_k_vars({oc, kd, kh, kw});

    gemm_schedule.for_each_var([&](const expr_t &var) {
        int bound = cfg_.padded_dim(var.as<var_t>().name);
        gemm_schedule.set_var_bound(var, bound);
    });

    expr_t g_tg_blk_idx, g_inner;
    expr_t ic_tg_blk_idx, ic_thr_blk_idx, ic_inner;
    expr_t mb_tg_blk_idx, mb_thr_blk_idx, mb_inner;
    expr_t iw_tg_blk_idx, iw_thr_blk_idx, iw_inner;
    expr_t oc_blk_idx, oc_inner;

    gemm_schedule.split(g, cfg_.g_tg_blk, g_tg_blk_idx, g_inner);
    gemm_schedule.split(ic, cfg_.ic_tg_blk, cfg_.ic_thr_blk, ic_tg_blk_idx,
            ic_thr_blk_idx, ic_inner);
    gemm_schedule.split(mb, cfg_.mb_tg_blk, cfg_.mb_thr_blk, mb_tg_blk_idx,
            mb_thr_blk_idx, mb_inner);
    gemm_schedule.split(iw, cfg_.iw_tg_blk, cfg_.iw_thr_blk, iw_tg_blk_idx,
            iw_thr_blk_idx, iw_inner);
    gemm_schedule.split(oc, cfg_.oc_blk, oc_blk_idx, oc_inner);

    auto g_idhw_idx = gemm_schedule.fuse({g_tg_blk_idx, id, ih, iw_tg_blk_idx});
    auto mb_iw_thr_blk_idx = gemm_schedule.fuse(mb_thr_blk_idx, iw_thr_blk_idx);

    gemm_schedule.bind(ic_tg_blk_idx, kernel_grid_.idx(0));
    gemm_schedule.bind(g_idhw_idx, kernel_grid_.idx(1));
    gemm_schedule.bind(mb_tg_blk_idx, kernel_grid_.idx(2));
    gemm_schedule.bind(ic_thr_blk_idx, tg_grid_.idx(0));
    gemm_schedule.bind(mb_iw_thr_blk_idx, tg_grid_.idx(1));

    gemm_schedule.tensorize(g_inner);
    gemm_schedule.tensorize(ic_inner);
    gemm_schedule.tensorize(mb_inner);
    gemm_schedule.tensorize(iw_inner);
    gemm_schedule.tensorize(oc_inner);

    if (cfg_.bwd_d_optimize_strided) {
        gemm_schedule.set_skip_condition(kd, od % cfg_.sd != 0);
        gemm_schedule.set_skip_condition(kh, oh % cfg_.sh != 0);
        if (cfg_.bwd_d_optimize_strided_iw)
            gemm_schedule.set_skip_condition(kw, ow % cfg_.sw != 0);
        // Put kd/kh/kw outermost to allow pipelining in oc loop.
        gemm_schedule.reorder({kd, kh, kw, oc_blk_idx});
    } else {
        gemm_schedule.reorder({oc_blk_idx, kd, kh, kw});
    }

    src_buf = kernel_info_.find_arg("src");
    wei_buf = kernel_info_.find_arg("wei");
    dst_buf = kernel_info_.find_arg("dst");
}

void kernel_builder_t::init_bwd_w(gemm_schedule_t &gemm_schedule,
        view_t &src_view, view_t &dst_view, view_t &wei_view, view_t &bia_view,
        expr_t &src_buf, expr_t &dst_buf, expr_t &wei_buf, expr_t &bia_buf,
        expr_t &bia_reduction_condition) {
    auto &src_layout = cfg_.src_layout;
    auto &wei_layout = cfg_.wei_layout;
    auto &dst_layout = cfg_.dst_layout;
    auto &bia_layout = cfg_.bia_layout;

    // Initialize thread group views.
    auto g = var_t::make(type_t::s32(), "g");
    auto mb = var_t::make(type_t::s32(), "mb");
    auto ic = var_t::make(type_t::s32(), "ic");
    auto oc = var_t::make(type_t::s32(), "oc");
    auto od = var_t::make(type_t::s32(), "od");
    auto oh = var_t::make(type_t::s32(), "oh");
    auto ow = var_t::make(type_t::s32(), "ow");
    auto kd = var_t::make(type_t::s32(), "kd");
    auto kh = var_t::make(type_t::s32(), "kh");
    auto kw = var_t::make(type_t::s32(), "kw");

    // Initialize masks.
    expr_t id_mask(true), ih_mask(true), iw_mask(true);

    bool check_ow = (cfg_.ow < cfg_.padded_dim("ow"));
    bool check_oh = (cfg_.oh < cfg_.padded_dim("oh"));
    bool check_od = (cfg_.od < cfg_.padded_dim("od"));
    bool check_kw = (cfg_.kw < cfg_.padded_dim("kw"));
    bool check_iw = check_kw
            || need_src_or_dst_check(/*is_fwd=*/true, cfg_.ow, cfg_.iw, cfg_.kw,
                    cfg_.pw, cfg_.sw, cfg_.dw);
    bool check_ih = need_src_or_dst_check(/*is_fwd=*/true, cfg_.oh, cfg_.ih,
            cfg_.kh, cfg_.ph, cfg_.sh, cfg_.dh);
    bool check_id = need_src_or_dst_check(/*is_fwd=*/true, cfg_.od, cfg_.id,
            cfg_.kd, cfg_.pd, cfg_.sd, cfg_.dd);
    bool check_iw_min = check_iw;
    bool check_ih_min = check_ih;
    bool check_id_min = check_id;
    bool check_iw_max = (check_iw || check_ow);
    bool check_ih_max = (check_ih || check_oh);
    bool check_id_max = (check_id || check_od);

    auto &x = view_t::placeholder_var();
    if (check_id_min) id_mask &= (x >= 0);
    if (check_ih_min) ih_mask &= (x >= 0);
    if (check_iw_min) iw_mask &= (x >= 0);
    if (check_id_max) id_mask &= (x < cfg_.id);
    if (check_ih_max) ih_mask &= (x < cfg_.ih);
    if (check_iw_max) iw_mask &= (x < cfg_.iw);

    // Source.
    src_view = view_t({mb, g, ic, od, oh, ow, kw}, 6);
    src_view.set_vdim(mb, cfg_.mb);
    src_view.set_vdim(g, cfg_.g);
    src_view.set_vdim(ic, cfg_.ic);
    src_view.set_vdim(od, cfg_.od);
    src_view.set_vdim(oh, cfg_.oh);
    src_view.set_vdim(ow, cfg_.ow);
    src_view.set_vdim(kw, cfg_.kw);
    src_view.set_tdim(0, mb);
    src_view.set_tdim(1, g);
    src_view.set_tdim(2, ic);
    src_view.set_tdim(3, od * cfg_.sd - cfg_.pd + kd * (1 + cfg_.dd), id_mask);
    src_view.set_tdim(4, oh * cfg_.sh - cfg_.ph + kh * (1 + cfg_.dh), ih_mask);
    src_view.set_tdim(5, ow * cfg_.sw - cfg_.pw + kw * (1 + cfg_.dw), iw_mask);
    src_view.set_tlayout(src_layout);
    src_view.set_tmasks(cfg_.padded_dims(), cfg_.dim_blocks());

    // Weights.
    wei_view = view_t({g, oc, ic, kd, kh, kw}, 6);
    wei_view.set_vdim(g, cfg_.g);
    wei_view.set_vdim(oc, cfg_.oc);
    wei_view.set_vdim(ic, cfg_.ic);
    wei_view.set_vdim(kd, cfg_.kd);
    wei_view.set_vdim(kh, cfg_.kh);
    wei_view.set_vdim(kw, cfg_.kw);
    wei_view.set_tdim(0, g);
    wei_view.set_tdim(1, oc);
    wei_view.set_tdim(2, ic);
    wei_view.set_tdim(3, kd);
    wei_view.set_tdim(4, kh);
    wei_view.set_tdim(5, kw);
    wei_view.set_tlayout(wei_layout);
    wei_view.set_tmasks(cfg_.padded_dims(), cfg_.dim_blocks());

    // Destination.
    dst_view = view_t({mb, g, oc, od, oh, ow}, 6);
    dst_view.set_vdim(mb, cfg_.mb);
    dst_view.set_vdim(g, cfg_.g);
    dst_view.set_vdim(oc, cfg_.oc);
    dst_view.set_vdim(od, cfg_.od);
    dst_view.set_vdim(oh, cfg_.oh);
    dst_view.set_vdim(ow, cfg_.ow);
    dst_view.set_tdim(0, mb);
    dst_view.set_tdim(1, g);
    dst_view.set_tdim(2, oc);
    dst_view.set_tdim(3, od);
    dst_view.set_tdim(4, oh);
    dst_view.set_tdim(5, ow);
    dst_view.set_tlayout(dst_layout);
    dst_view.set_tmasks(cfg_.padded_dims(), cfg_.dim_blocks());

    // Bias.
    if (cfg_.with_bias) {
        bia_view = view_t({g, oc}, 2);
        bia_view.set_vdim(g, cfg_.g);
        bia_view.set_vdim(oc, cfg_.oc);
        bia_view.set_tdim(0, g);
        bia_view.set_tdim(1, oc);
        bia_view.set_tlayout(bia_layout);
        bia_view.set_tmasks(cfg_.padded_dims(), cfg_.dim_blocks());
    }

    // Initialize GEMM schedule.
    gemm_schedule.set_a_view(src_view);
    gemm_schedule.set_b_view(dst_view);
    gemm_schedule.set_c_view(wei_view);
    gemm_schedule.set_b_vars({g});
    gemm_schedule.set_m_vars({ic, kw});
    gemm_schedule.set_n_vars({oc});
    gemm_schedule.set_k_vars({mb, od, oh, ow});

    gemm_schedule.for_each_var([&](const expr_t &var) {
        int bound = cfg_.padded_dim(var.as<var_t>().name);
        gemm_schedule.set_var_bound(var, bound);
    });

    expr_t g_thr_blk_idx, g_inner;
    expr_t mb_thr_blk_idx, mb_blk_idx, mb_inner;
    expr_t oc_tg_blk_idx, oc_thr_blk_idx, oc_inner;
    expr_t ic_tg_blk_idx, ic_thr_blk_idx, ic_inner;
    expr_t od_thr_blk_idx, od_inner;
    expr_t oh_thr_blk_idx, oh_inner;
    expr_t ow_thr_blk_idx, ow_blk_idx, ow_inner;
    expr_t kw_tg_blk_idx, kw_inner;

    gemm_schedule.split(g, cfg_.g_thr_blk, g_thr_blk_idx, g_inner);
    gemm_schedule.split(mb, cfg_.mb_thr_blk, cfg_.mb_blk, mb_thr_blk_idx,
            mb_blk_idx, mb_inner);
    gemm_schedule.split(ic, cfg_.ic_tg_blk, cfg_.ic_thr_blk, ic_tg_blk_idx,
            ic_thr_blk_idx, ic_inner);
    gemm_schedule.split(oc, cfg_.oc_tg_blk, cfg_.oc_thr_blk, oc_tg_blk_idx,
            oc_thr_blk_idx, oc_inner);
    gemm_schedule.split(od, cfg_.od_thr_blk, od_thr_blk_idx, od_inner);
    gemm_schedule.split(oh, cfg_.oh_thr_blk, oh_thr_blk_idx, oh_inner);
    gemm_schedule.split(ow, cfg_.ow_thr_blk, cfg_.ow_blk, ow_thr_blk_idx,
            ow_blk_idx, ow_inner);
    gemm_schedule.split(kw, cfg_.kw_tg_blk, kw_tg_blk_idx, kw_inner);

    auto odhw_thr_blk_kdhw_ic_tg_blk_idx
            = gemm_schedule.fuse({od_thr_blk_idx, oh_thr_blk_idx,
                    ow_thr_blk_idx, kd, kh, kw_tg_blk_idx, ic_tg_blk_idx});

    auto g_mb_thr_blk_idx = gemm_schedule.fuse({g_thr_blk_idx, mb_thr_blk_idx});

    gemm_schedule.bind(oc_tg_blk_idx, kernel_grid_.idx(0));
    gemm_schedule.bind(odhw_thr_blk_kdhw_ic_tg_blk_idx, kernel_grid_.idx(1));
    gemm_schedule.bind(g_mb_thr_blk_idx, kernel_grid_.idx(2));

    gemm_schedule.bind(oc_thr_blk_idx, tg_grid_.idx(0));
    gemm_schedule.bind(ic_thr_blk_idx, tg_grid_.idx(1));

    gemm_schedule.reorder({od_inner, oh_inner, ow_inner, mb_blk_idx});

    gemm_schedule.unroll(mb_blk_idx, cfg_.mb_unroll);
    gemm_schedule.unroll(ow_blk_idx, cfg_.ow_unroll);

    gemm_schedule.tensorize(g_inner);
    gemm_schedule.tensorize(oc_inner);
    gemm_schedule.tensorize(ic_inner);
    gemm_schedule.tensorize(mb_inner);
    gemm_schedule.tensorize(ow_inner);
    gemm_schedule.tensorize(kw_inner);

    src_buf = kernel_info_.find_arg("src");
    wei_buf = kernel_info_.find_arg("wei");
    dst_buf = kernel_info_.find_arg("dst");

    if (cfg_.with_bias) {
        bia_buf = kernel_info_.find_arg("bia");
        bia_reduction_condition = expr_t(true);
        if (cfg_.kd > 1) bia_reduction_condition &= (kd == 0);
        if (cfg_.kh > 1) bia_reduction_condition &= (kh == 0);
        if (cfg_.kw > 1) bia_reduction_condition &= (kw_tg_blk_idx == 0);
        if (cfg_.grid_dim("ic") > 1)
            bia_reduction_condition &= (ic_tg_blk_idx == 0);
        if (!cfg_.use_b_slm && tg_grid_.dim(1) > 1) {
            bia_reduction_condition &= (tg_grid_.idx(1) == 0);
        }
    }
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
