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

#include "gpu/jit/conv/ir_builder.hpp"

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

// Represents hierarchy of tile levels and corresponding loop/grid indices.
//
// | Tile level | Nesting level | Maps to                |
// |------------|---------------|------------------------|
// | grid_dim   | 0             | Thread group           |
// | loop_dim   | 1             | Loop in thread         |
// | tg_dim     | 2             | Thread in thread group |
// | iter_dim   | 3             | Iteration in loop      |
class dim_tile_t {
public:
    const expr_t &grid_idx() const { return not_empty(grid_idx_); }
    const expr_t &tg_idx() const { return not_empty(tg_idx_); }
    const expr_t &loop_idx() const { return not_empty(loop_idx_); }
    const expr_t &iter_idx() const { return not_empty(iter_idx_); }

    void set_grid_idx(const expr_t &idx) { grid_idx_ = idx; }
    void set_tg_idx(const expr_t &idx) { tg_idx_ = idx; }
    void set_loop_idx(const expr_t &idx) { loop_idx_ = idx; }
    void set_iter_idx(const expr_t &idx) { iter_idx_ = idx; }

private:
    static const expr_t &not_empty(const expr_t &v) {
        ir_assert(!v.is_empty()) << "Queried empty index.";
        return v;
    }

    expr_t grid_idx_;
    expr_t tg_idx_;
    expr_t loop_idx_;
    expr_t iter_idx_;
};

dim_tile_t create_tile(gemm_schedule_t &gemm_schedule, const conv_config_t &cfg,
        const expr_t &dim) {
    dim_tile_t tile;
    auto &name = dim.as<var_t>().name;
    int loop_dim = cfg.loop_dim(name);
    int tg_dim = cfg.thread_group_dim(name);
    int iter_dim = cfg.iter_dim(name);

    std::vector<int> dims = {1, loop_dim, tg_dim, iter_dim};
    int ndims = (int)dims.size();
    std::vector<expr_t> idxs(ndims);

    static const char *suffixes[]
            = {"_grid_idx", "_loop_idx", "_tg_idx", "_iter_idx"};
    auto &dim_name = dim.as<var_t>().name;

    auto has_block = [&](int dim_idx) {
        bool is_thr = (dim_idx == 1);
        bool is_tg = (dim_idx == 2);
        bool is_iter = (dim_idx == 3);
        if (is_thr || is_iter) return true;
        for (int i = 0; i < 3; i++) {
            auto **dd = is_tg ? get_thread_group_grid_conv_dims(cfg.prb(), i)
                              : get_kernel_grid_conv_dims(cfg.prb(), i);
            for (auto **d = dd; *d; d++)
                if (dim_name == *d) return true;
        }
        return false;
    };

    expr_t idx = dim;
    for (int i = ndims - 1; i >= 1; i--) {
        expr_t outer;
        expr_t inner;
        auto outer_name = (i == 1) ? dim_name + suffixes[i] : std::string();
        auto inner_name = dim_name + suffixes[i];
        gemm_schedule.split(idx, dims[i], outer, inner, outer_name, inner_name);
        if (has_block(i)) idxs[i] = inner;
        idx = outer;
    }
    idxs[0] = idx;

    tile.set_grid_idx(idxs[0]);
    tile.set_loop_idx(idxs[1]);
    tile.set_tg_idx(idxs[2]);
    tile.set_iter_idx(idxs[3]);

    return tile;
}

void conv_ir_builder_t::init_fwd(gemm_schedule_t &gemm_schedule,
        view_t &src_view, view_t &wei_view, view_t &dst_view, expr_t &src_buf,
        expr_t &wei_buf, expr_t &dst_buf) {
    auto &src_layout = cfg_.src_layout().compute();
    auto &wei_layout = cfg_.wei_layout().compute();
    auto &dst_layout = cfg_.dst_layout().compute();

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
    if (cfg_.fuse_spatial()) {
        osp = var_t::make(type_t::s32(), "osp");
        ow = osp;
        oh = osp / prb_.ow;
        od = osp / (prb_.oh * prb_.ow);

        bool is_1d = (prb_.oh == 1 && prb_.od == 1);
        bool is_2d = (prb_.oh != 1 && prb_.od == 1);
        bool is_3d = !is_1d && !is_2d;

        bool check_osp = (prb_.osp < cfg_.padded_dim("osp"));
        check_ow = is_1d && check_osp;
        check_oh = is_2d && check_osp;
        check_od = is_3d && check_osp;

        if (!is_1d) ow %= prb_.ow;
        if (!is_2d) oh %= prb_.oh;
    } else {
        od = var_t::make(type_t::s32(), "od");
        oh = var_t::make(type_t::s32(), "oh");
        ow = var_t::make(type_t::s32(), "ow");
        check_ow = (prb_.ow < cfg_.padded_dim("ow"));
    }

    // Initialize masks.
    expr_t id_mask, ih_mask, iw_mask;
    expr_t od_mask, oh_mask, ow_mask;

    bool check_kw = (prb_.kw < cfg_.padded_dim("kw"));
    bool check_iw = check_kw || check_ow
            || need_src_or_dst_check(prb_.is_fwd, prb_.ow, prb_.iw, prb_.kw,
                    prb_.pw, prb_.sw, prb_.dw);
    bool check_ih = check_oh
            || need_src_or_dst_check(prb_.is_fwd, prb_.oh, prb_.ih, prb_.kh,
                    prb_.ph, prb_.sh, prb_.dh);
    bool check_id = check_od
            || need_src_or_dst_check(prb_.is_fwd, prb_.od, prb_.id, prb_.kd,
                    prb_.pd, prb_.sd, prb_.dd);

    auto &x = view_t::placeholder_var();
    if (check_id) id_mask = (x >= 0) & (x < prb_.id);
    if (check_ih) ih_mask = (x >= 0) & (x < prb_.ih);
    if (check_iw) iw_mask = (x >= 0) & (x < prb_.iw);
    if (check_od) od_mask = (x >= 0) & (x < prb_.od);
    if (check_oh) oh_mask = (x >= 0) & (x < prb_.oh);
    if (check_ow) ow_mask = (x >= 0) & (x < prb_.ow);

    // Source.
    if (cfg_.fuse_spatial()) {
        src_view = view_t({mb, g, ic, osp, kd, kh, kw}, 6);
    } else {
        src_view = view_t({mb, g, ic, od, oh, ow, kd, kh, kw}, 6);
    }
    src_view.set_vdim(mb, prb_.mb);
    src_view.set_vdim(g, prb_.g);
    src_view.set_vdim(ic, prb_.ic);
    if (cfg_.fuse_spatial()) {
        src_view.set_vdim(osp, prb_.osp);
    } else {
        src_view.set_vdim(od, prb_.od);
        src_view.set_vdim(oh, prb_.oh);
        src_view.set_vdim(ow, prb_.ow);
    }
    src_view.set_vdim(kd, prb_.kd);
    src_view.set_vdim(kh, prb_.kh);
    src_view.set_vdim(kw, prb_.kw);
    src_view.set_tdim(0, mb);
    src_view.set_tdim(1, g);
    src_view.set_tdim(2, ic);
    src_view.set_tdim(3, od * prb_.sd - prb_.pd + kd * (1 + prb_.dd), id_mask);
    src_view.set_tdim(4, oh * prb_.sh - prb_.ph + kh * (1 + prb_.dh), ih_mask);
    src_view.set_tdim(5, ow * prb_.sw - prb_.pw + kw * (1 + prb_.dw), iw_mask);
    src_view.set_tlayout(src_layout);
    src_view.set_tmasks(cfg_.padded_dims().get(), cfg_.iter_dims().get());

    // Weights.
    wei_view = view_t({g, oc, ic, kd, kh, kw}, 6);
    wei_view.set_vdim(g, prb_.g);
    wei_view.set_vdim(oc, prb_.oc);
    wei_view.set_vdim(ic, prb_.ic);
    wei_view.set_vdim(kd, prb_.kd);
    wei_view.set_vdim(kh, prb_.kh);
    wei_view.set_vdim(kw, prb_.kw);
    wei_view.set_tdim(0, g);
    wei_view.set_tdim(1, oc);
    wei_view.set_tdim(2, ic);
    wei_view.set_tdim(3, kd);
    wei_view.set_tdim(4, kh);
    wei_view.set_tdim(5, kw);
    wei_view.set_tlayout(wei_layout);
    wei_view.set_tmasks(cfg_.padded_dims().get(), cfg_.iter_dims().get());

    // Destination.
    if (cfg_.fuse_spatial()) {
        dst_view = view_t({mb, g, oc, osp}, 6);
    } else {
        dst_view = view_t({mb, g, oc, od, oh, ow}, 6);
    }
    dst_view.set_vdim(mb, prb_.mb);
    dst_view.set_vdim(g, prb_.g);
    dst_view.set_vdim(oc, prb_.oc);
    if (cfg_.fuse_spatial()) {
        dst_view.set_vdim(osp, prb_.osp);
    } else {
        dst_view.set_vdim(od, prb_.od);
        dst_view.set_vdim(oh, prb_.oh);
        dst_view.set_vdim(ow, prb_.ow);
    }
    dst_view.set_tdim(0, mb);
    dst_view.set_tdim(1, g);
    dst_view.set_tdim(2, oc);
    dst_view.set_tdim(3, od, od_mask);
    dst_view.set_tdim(4, oh, oh_mask);
    dst_view.set_tdim(5, ow, ow_mask);
    dst_view.set_tlayout(dst_layout);
    dst_view.set_tmasks(cfg_.padded_dims().get(), cfg_.iter_dims().get());

    // Initialize GEMM schedule.
    gemm_schedule.set_a_view(src_view);
    gemm_schedule.set_b_view(wei_view);
    gemm_schedule.set_c_view(dst_view);
    gemm_schedule.set_b_vars({g});
    if (cfg_.fuse_spatial()) {
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

    auto g_tile = create_tile(gemm_schedule, cfg_, g);
    auto oc_tile = create_tile(gemm_schedule, cfg_, oc);
    auto mb_tile = create_tile(gemm_schedule, cfg_, mb);
    auto osp_tile = create_tile(gemm_schedule, cfg_, osp.is_empty() ? ow : osp);
    auto ic_tile = create_tile(gemm_schedule, cfg_, ic);
    auto kw_tile = create_tile(gemm_schedule, cfg_, kw);

    auto g_osp_grid_idx = cfg_.fuse_spatial()
            ? gemm_schedule.fuse({g_tile.grid_idx(), osp_tile.grid_idx()})
            : gemm_schedule.fuse(
                    {g_tile.grid_idx(), od, oh, osp_tile.grid_idx()});
    auto mb_osp_tg_idx
            = gemm_schedule.fuse(mb_tile.tg_idx(), osp_tile.tg_idx());

    gemm_schedule.bind(oc_tile.grid_idx(), cfg_.kernel_grid().idx(0));
    gemm_schedule.bind(g_osp_grid_idx, cfg_.kernel_grid().idx(1));
    gemm_schedule.bind(mb_tile.grid_idx(), cfg_.kernel_grid().idx(2));
    gemm_schedule.bind(oc_tile.tg_idx(), cfg_.thread_group_grid().idx(0));
    gemm_schedule.bind(mb_osp_tg_idx, cfg_.thread_group_grid().idx(1));
    gemm_schedule.bind(ic_tile.tg_idx(), cfg_.thread_group_grid().idx(2));

    gemm_schedule.tensorize(g_tile.iter_idx());
    gemm_schedule.tensorize(oc_tile.iter_idx());
    gemm_schedule.tensorize(mb_tile.iter_idx());
    gemm_schedule.tensorize(osp_tile.iter_idx());
    gemm_schedule.tensorize(kw_tile.iter_idx());
    gemm_schedule.tensorize(ic_tile.iter_idx());

    gemm_schedule.reorder({ic_tile.loop_idx(), kd, kh, kw_tile.loop_idx(),
            oc_tile.tg_idx(), mb_osp_tg_idx, ic_tile.tg_idx()});

    src_buf = kernel_info_.find_arg("src");
    wei_buf = kernel_info_.find_arg("wei");
    dst_buf = kernel_info_.find_arg("dst");
}

void conv_ir_builder_t::init_bwd_d(gemm_schedule_t &gemm_schedule,
        view_t &dst_view, view_t &wei_view, view_t &src_view, expr_t &dst_buf,
        expr_t &wei_buf, expr_t &src_buf) {
    auto &src_layout = cfg_.src_layout().compute();
    auto &wei_layout = cfg_.wei_layout().compute();
    auto &dst_layout = cfg_.dst_layout().compute();

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

    bool check_iw = (prb_.iw < cfg_.padded_dim("iw"));
    bool check_ow = check_iw
            || need_src_or_dst_check(prb_.is_fwd, prb_.ow, prb_.iw, prb_.kw,
                    prb_.pw, prb_.sw, prb_.dw);
    bool check_oh = need_src_or_dst_check(
            prb_.is_fwd, prb_.oh, prb_.ih, prb_.kh, prb_.ph, prb_.sh, prb_.dh);
    bool check_od = need_src_or_dst_check(
            prb_.is_fwd, prb_.od, prb_.id, prb_.kd, prb_.pd, prb_.sd, prb_.dd);

    auto &x = view_t::placeholder_var();
    if (check_od) od_mask = (x >= 0) & (x < prb_.od);
    if (check_oh) oh_mask = (x >= 0) & (x < prb_.oh);
    if (check_ow) ow_mask = (x >= 0) & (x < prb_.ow);

    std::function<expr_t(const expr_t &)> iw_mapping;
    if (cfg_.bwd_d_optimize_strided_iw()) {
        // Apply mapping to iw to ensure each thread group has the same
        // stride condition when evaluating skip conditions.
        iw_mapping = [&](const expr_t &e) {
            int iw_tg_blk = cfg_.thread_group_dim("iw") * cfg_.iter_dim("iw");
            int iw_bound = utils::rnd_up(prb_.iw, iw_tg_blk);
            int iw_same_mod_blk = ir_utils::safe_divide(iw_bound, prb_.sw);
            return (e % iw_same_mod_blk) * prb_.sw + (e / iw_same_mod_blk);
        };
    } else {
        iw_mapping = [](const expr_t &e) { return e; };
    }

    // Destination.
    dst_view = view_t({mb, g, oc, id, ih, iw, kd, kh, kw}, 6);
    dst_view.set_vdim(mb, prb_.mb);
    dst_view.set_vdim(g, prb_.g);
    dst_view.set_vdim(oc, prb_.oc);
    dst_view.set_vdim(id, prb_.id);
    dst_view.set_vdim(ih, prb_.ih);
    dst_view.set_vdim(iw, prb_.iw);
    dst_view.set_vdim(kd, prb_.kd);
    dst_view.set_vdim(kh, prb_.kh);
    dst_view.set_vdim(kw, prb_.kw);
    dst_view.set_tdim(0, mb);
    dst_view.set_tdim(1, g);
    dst_view.set_tdim(2, oc);

    auto od = id - kd * (1 + prb_.dd) + prb_.pd;
    auto oh = ih - kh * (1 + prb_.dh) + prb_.ph;
    auto ow = iw_mapping(iw) - kw * (1 + prb_.dw) + prb_.pw;

    // When stride optimization is enabled, stride conditions are handled by
    // continue calls in the outer loops.
    if (!cfg_.bwd_d_optimize_strided_iw()) {
        od_mask &= (od % prb_.sd == 0);
        oh_mask &= (oh % prb_.sh == 0);
        ow_mask &= (ow % prb_.sw == 0);
    }
    dst_view.set_tdim(3, od / prb_.sd, od_mask);
    dst_view.set_tdim(4, oh / prb_.sh, oh_mask);
    dst_view.set_tdim(5, ow / prb_.sw, ow_mask);

    dst_view.set_tlayout(dst_layout);
    dst_view.set_tmasks(cfg_.padded_dims().get(), cfg_.iter_dims().get());

    // Weights.
    wei_view = view_t({g, oc, ic, kd, kh, kw}, 6);
    wei_view.set_vdim(g, prb_.g);
    wei_view.set_vdim(ic, prb_.ic);
    wei_view.set_vdim(oc, prb_.oc);
    wei_view.set_vdim(kd, prb_.kd);
    wei_view.set_vdim(kh, prb_.kh);
    wei_view.set_vdim(kw, prb_.kw);
    wei_view.set_tdim(0, g);
    wei_view.set_tdim(1, oc);
    wei_view.set_tdim(2, ic);
    wei_view.set_tdim(3, kd);
    wei_view.set_tdim(4, kh);
    wei_view.set_tdim(5, kw);
    wei_view.set_tlayout(wei_layout);
    wei_view.set_tmasks(cfg_.padded_dims().get(), cfg_.iter_dims().get());

    // Source.
    src_view = view_t({mb, g, ic, id, ih, iw}, 6);
    src_view.set_vdim(mb, prb_.mb);
    src_view.set_vdim(g, prb_.g);
    src_view.set_vdim(ic, prb_.ic);
    src_view.set_vdim(id, prb_.id);
    src_view.set_vdim(ih, prb_.ih);
    src_view.set_vdim(iw, prb_.iw);
    src_view.set_tdim(0, mb);
    src_view.set_tdim(1, g);
    src_view.set_tdim(2, ic);
    src_view.set_tdim(3, id);
    src_view.set_tdim(4, ih);
    src_view.set_tdim(5, iw_mapping(iw));
    src_view.set_tlayout(src_layout);
    src_view.set_tmasks(cfg_.padded_dims().get(), cfg_.iter_dims().get());

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

    auto g_tile = create_tile(gemm_schedule, cfg_, g);
    auto ic_tile = create_tile(gemm_schedule, cfg_, ic);
    auto mb_tile = create_tile(gemm_schedule, cfg_, mb);
    auto iw_tile = create_tile(gemm_schedule, cfg_, iw);
    auto oc_tile = create_tile(gemm_schedule, cfg_, oc);

    auto g_isp_grid_idx = gemm_schedule.fuse(
            {g_tile.grid_idx(), id, ih, iw_tile.grid_idx()});
    auto mb_iw_tg_idx = gemm_schedule.fuse(mb_tile.tg_idx(), iw_tile.tg_idx());

    gemm_schedule.bind(ic_tile.grid_idx(), cfg_.kernel_grid().idx(0));
    gemm_schedule.bind(g_isp_grid_idx, cfg_.kernel_grid().idx(1));
    gemm_schedule.bind(mb_tile.grid_idx(), cfg_.kernel_grid().idx(2));
    gemm_schedule.bind(ic_tile.tg_idx(), cfg_.thread_group_grid().idx(0));
    gemm_schedule.bind(mb_iw_tg_idx, cfg_.thread_group_grid().idx(1));
    gemm_schedule.bind(oc_tile.tg_idx(), cfg_.thread_group_grid().idx(2));

    gemm_schedule.tensorize(g_tile.iter_idx());
    gemm_schedule.tensorize(ic_tile.iter_idx());
    gemm_schedule.tensorize(mb_tile.iter_idx());
    gemm_schedule.tensorize(iw_tile.iter_idx());
    gemm_schedule.tensorize(oc_tile.iter_idx());

    if (cfg_.bwd_d_optimize_strided()) {
        gemm_schedule.set_skip_condition(kd, od % prb_.sd != 0);
        gemm_schedule.set_skip_condition(kh, oh % prb_.sh != 0);
        if (cfg_.bwd_d_optimize_strided_iw())
            gemm_schedule.set_skip_condition(kw, ow % prb_.sw != 0);
        // Put kd/kh/kw outermost to allow pipelining in oc loop.
        gemm_schedule.reorder({kd, kh, kw, oc_tile.loop_idx()});
    } else {
        gemm_schedule.reorder({oc_tile.loop_idx(), kd, kh, kw});
    }

    src_buf = kernel_info_.find_arg("src");
    wei_buf = kernel_info_.find_arg("wei");
    dst_buf = kernel_info_.find_arg("dst");
}

void conv_ir_builder_t::init_bwd_w(gemm_schedule_t &gemm_schedule,
        view_t &src_view, view_t &dst_view, view_t &wei_view, view_t &bia_view,
        expr_t &src_buf, expr_t &dst_buf, expr_t &wei_buf, expr_t &bia_buf,
        expr_t &bia_reduction_condition) {
    auto &src_layout = cfg_.src_layout().compute();
    auto &wei_layout = cfg_.wei_layout().compute();
    auto &dst_layout = cfg_.dst_layout().compute();
    auto &bia_layout = cfg_.bia_layout().compute();

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

    bool check_ow = (prb_.ow < cfg_.padded_dim("ow"));
    bool check_oh = (prb_.oh < cfg_.padded_dim("oh"));
    bool check_od = (prb_.od < cfg_.padded_dim("od"));
    bool check_kw = (prb_.kw < cfg_.padded_dim("kw"));
    bool check_iw = check_kw
            || need_src_or_dst_check(/*is_fwd=*/true, prb_.ow, prb_.iw, prb_.kw,
                    prb_.pw, prb_.sw, prb_.dw);
    bool check_ih = need_src_or_dst_check(/*is_fwd=*/true, prb_.oh, prb_.ih,
            prb_.kh, prb_.ph, prb_.sh, prb_.dh);
    bool check_id = need_src_or_dst_check(/*is_fwd=*/true, prb_.od, prb_.id,
            prb_.kd, prb_.pd, prb_.sd, prb_.dd);
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
    if (check_id_max) id_mask &= (x < prb_.id);
    if (check_ih_max) ih_mask &= (x < prb_.ih);
    if (check_iw_max) iw_mask &= (x < prb_.iw);

    // Source.
    src_view = view_t({mb, g, ic, od, oh, ow, kw}, 6);
    src_view.set_vdim(mb, prb_.mb);
    src_view.set_vdim(g, prb_.g);
    src_view.set_vdim(ic, prb_.ic);
    src_view.set_vdim(od, prb_.od);
    src_view.set_vdim(oh, prb_.oh);
    src_view.set_vdim(ow, prb_.ow);
    src_view.set_vdim(kw, prb_.kw);
    src_view.set_tdim(0, mb);
    src_view.set_tdim(1, g);
    src_view.set_tdim(2, ic);
    src_view.set_tdim(3, od * prb_.sd - prb_.pd + kd * (1 + prb_.dd), id_mask);
    src_view.set_tdim(4, oh * prb_.sh - prb_.ph + kh * (1 + prb_.dh), ih_mask);
    src_view.set_tdim(5, ow * prb_.sw - prb_.pw + kw * (1 + prb_.dw), iw_mask);
    src_view.set_tlayout(src_layout);
    src_view.set_tmasks(cfg_.padded_dims().get(), cfg_.iter_dims().get());

    // Weights.
    wei_view = view_t({g, oc, ic, kd, kh, kw}, 6);
    wei_view.set_vdim(g, prb_.g);
    wei_view.set_vdim(oc, prb_.oc);
    wei_view.set_vdim(ic, prb_.ic);
    wei_view.set_vdim(kd, prb_.kd);
    wei_view.set_vdim(kh, prb_.kh);
    wei_view.set_vdim(kw, prb_.kw);
    wei_view.set_tdim(0, g);
    wei_view.set_tdim(1, oc);
    wei_view.set_tdim(2, ic);
    wei_view.set_tdim(3, kd);
    wei_view.set_tdim(4, kh);
    wei_view.set_tdim(5, kw);
    wei_view.set_tlayout(wei_layout);
    wei_view.set_tmasks(cfg_.padded_dims().get(), cfg_.iter_dims().get());

    // Destination.
    dst_view = view_t({mb, g, oc, od, oh, ow}, 6);
    dst_view.set_vdim(mb, prb_.mb);
    dst_view.set_vdim(g, prb_.g);
    dst_view.set_vdim(oc, prb_.oc);
    dst_view.set_vdim(od, prb_.od);
    dst_view.set_vdim(oh, prb_.oh);
    dst_view.set_vdim(ow, prb_.ow);
    dst_view.set_tdim(0, mb);
    dst_view.set_tdim(1, g);
    dst_view.set_tdim(2, oc);
    dst_view.set_tdim(3, od);
    dst_view.set_tdim(4, oh);
    dst_view.set_tdim(5, ow);
    dst_view.set_tlayout(dst_layout);
    dst_view.set_tmasks(cfg_.padded_dims().get(), cfg_.iter_dims().get());

    // Bias.
    if (prb_.with_bias) {
        bia_view = view_t({g, oc}, 2);
        bia_view.set_vdim(g, prb_.g);
        bia_view.set_vdim(oc, prb_.oc);
        bia_view.set_tdim(0, g);
        bia_view.set_tdim(1, oc);
        bia_view.set_tlayout(bia_layout);
        bia_view.set_tmasks(cfg_.padded_dims().get(), cfg_.iter_dims().get());
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

    auto g_tile = create_tile(gemm_schedule, cfg_, g);
    auto mb_tile = create_tile(gemm_schedule, cfg_, mb);
    auto ic_tile = create_tile(gemm_schedule, cfg_, ic);
    auto oc_tile = create_tile(gemm_schedule, cfg_, oc);
    auto od_tile = create_tile(gemm_schedule, cfg_, od);
    auto oh_tile = create_tile(gemm_schedule, cfg_, oh);
    auto ow_tile = create_tile(gemm_schedule, cfg_, ow);
    auto kw_tile = create_tile(gemm_schedule, cfg_, kw);

    auto osp_ksp_ic_grid_idx = gemm_schedule.fuse(
            {od_tile.grid_idx(), oh_tile.grid_idx(), ow_tile.grid_idx(), kd, kh,
                    kw_tile.grid_idx(), ic_tile.grid_idx()});

    auto g_mb_grid_idx
            = gemm_schedule.fuse({g_tile.grid_idx(), mb_tile.grid_idx()});

    gemm_schedule.bind(oc_tile.grid_idx(), cfg_.kernel_grid().idx(0));
    gemm_schedule.bind(osp_ksp_ic_grid_idx, cfg_.kernel_grid().idx(1));
    gemm_schedule.bind(g_mb_grid_idx, cfg_.kernel_grid().idx(2));

    gemm_schedule.bind(oc_tile.tg_idx(), cfg_.thread_group_grid().idx(0));
    gemm_schedule.bind(ic_tile.tg_idx(), cfg_.thread_group_grid().idx(1));

    gemm_schedule.reorder({od_tile.loop_idx(), oh_tile.loop_idx(),
            ow_tile.loop_idx(), mb_tile.loop_idx()});

    gemm_schedule.unroll(mb_tile.loop_idx(), cfg_.unroll("mb"));
    gemm_schedule.unroll(ow_tile.loop_idx(), cfg_.unroll("ow"));

    gemm_schedule.tensorize(g_tile.iter_idx());
    gemm_schedule.tensorize(oc_tile.iter_idx());
    gemm_schedule.tensorize(ic_tile.iter_idx());
    gemm_schedule.tensorize(mb_tile.iter_idx());
    gemm_schedule.tensorize(ow_tile.iter_idx());
    gemm_schedule.tensorize(kw_tile.iter_idx());

    src_buf = kernel_info_.find_arg("src");
    wei_buf = kernel_info_.find_arg("wei");
    dst_buf = kernel_info_.find_arg("dst");

    if (prb_.with_bias) {
        bia_buf = kernel_info_.find_arg("bia");
        bia_reduction_condition = expr_t(true);
        if (prb_.kd > 1) bia_reduction_condition &= (kd == 0);
        if (prb_.kh > 1) bia_reduction_condition &= (kh == 0);
        if (prb_.kw > 1) bia_reduction_condition &= (kw_tile.grid_idx() == 0);
        if (cfg_.grid_dim("ic") > 1)
            bia_reduction_condition &= (ic_tile.grid_idx() == 0);
        if (!cfg_.slm().b() && cfg_.thread_group_grid().dim(1) > 1) {
            bia_reduction_condition &= (cfg_.thread_group_grid().idx(1) == 0);
        }
    }
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
