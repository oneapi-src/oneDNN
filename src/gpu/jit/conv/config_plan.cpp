/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "gpu/jit/conv/config_plan.hpp"

#include <sstream>

#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/grf_usage.hpp"
#include "gpu/jit/ir/gemm_schedule.hpp"
#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/ir/reduce.hpp"
#include "gpu/jit/ir/reorder.hpp"
#include "gpu/jit/ir/send_plan.hpp"
#include "gpu/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

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

static dim_tile_t create_tile(gemm_schedule_t &gemm_schedule,
        const conv_config_t &cfg, const expr_t &dim) {
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

void init_fwd(const conv_config_t &cfg_, gemm_schedule_t &gemm_schedule,
        view_t &src_view, view_t &wei_view, view_t &dst_view) {
    auto &prb_ = cfg_.prb();

    constraint_set_t init_cset;

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
    src_view.set_tmasks(cfg_.padded_dims().get());

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
    wei_view.set_tmasks(cfg_.padded_dims().get());

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
    dst_view.set_tmasks(cfg_.padded_dims().get());

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
}

void init_bwd_d(const conv_config_t &cfg_, gemm_schedule_t &gemm_schedule,
        view_t &dst_view, view_t &wei_view, view_t &src_view) {
    auto &prb_ = cfg_.prb();
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
        if (prb_.sd != 1) od_mask &= (od % prb_.sd == 0);
        if (prb_.sh != 1) oh_mask &= (oh % prb_.sh == 0);
        if (prb_.sw != 1) ow_mask &= (ow % prb_.sw == 0);
    }
    dst_view.set_tdim(3, od / prb_.sd, od_mask);
    dst_view.set_tdim(4, oh / prb_.sh, oh_mask);
    dst_view.set_tdim(5, ow / prb_.sw, ow_mask);

    dst_view.set_tlayout(dst_layout);
    dst_view.set_tmasks(cfg_.padded_dims().get());

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
    wei_view.set_tmasks(cfg_.padded_dims().get());

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
    src_view.set_tmasks(cfg_.padded_dims().get());

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
}

void init_bwd_w(const conv_config_t &cfg_, gemm_schedule_t &gemm_schedule,
        view_t &src_view, view_t &dst_view, view_t &wei_view,
        view_t &bia_view) {
    auto &prb_ = cfg_.prb();
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
    src_view.set_tmasks(cfg_.padded_dims().get());

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
    wei_view.set_tmasks(cfg_.padded_dims().get());

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
    dst_view.set_tmasks(cfg_.padded_dims().get());

    // Bias.
    if (prb_.with_bias) {
        bia_view = view_t({g, oc}, 2);
        bia_view.set_vdim(g, prb_.g);
        bia_view.set_vdim(oc, prb_.oc);
        bia_view.set_tdim(0, g);
        bia_view.set_tdim(1, oc);
        bia_view.set_tlayout(bia_layout);
        bia_view.set_tmasks(cfg_.padded_dims().get());
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
}

reorder_plan_t create_reorder_plan(
        ngen::HW hw, const layout_t &src, const layout_t &dst) {
    if (src == dst) return reorder_plan_t();
    if (src.type().is_bitwise_compatible(dst.type())
            && src.retype(dst.type()) == dst)
        return reorder_plan_t();
    reorder_plan_t ret(hw);
    ret.src = src;
    ret.dst = dst;
    return ret;
}

bool reorder_plan_t::can_split(int factor) const {
    auto split_src = split(src, factor);
    auto split_dst = split(dst, factor);
    auto split_src_dims = split_src.dims();
    auto split_dst_dims = split_dst.dims();
    return ir_utils::is_equal(split_src_dims, split_dst_dims);
}

void reorder_plan_t::set_split(int factor) {
    if (!*this) return;
    ir_assert(can_split(factor));
    split_factor = factor;
}

stmt_t reorder_plan_t::create_stmt(
        const expr_t &src_buf, const expr_t &dst_buf) const {
    if (!*this) return stmt_t();
    auto split_src = split(src, split_factor);
    auto split_dst = split(dst, split_factor);
    auto stmt = create_reorder_stmt(split_src, split_dst, src_buf, dst_buf);
    return stmt;
}

int reorder_plan_t::estimate_regs() const {
    if (!*this) return 0;

    int dst_size = utils::div_up(dst.size(), split_factor);
    int ret = 0;
    ret += utils::rnd_up(dst_size, grf_size());
    return utils::div_up(ret, grf_size());
}

reduce_plan_t create_reduce_plan(
        ngen::HW hw, const layout_t &src, const layout_t &dst, uint32_t mask) {
    reduce_plan_t ret(hw);
    ret.src = src;
    ret.dst = dst;
    ret.mask = mask;
    return ret;
}

int reduce_plan_t::dst_buf_size() const {
    int dst_size = utils::div_up(dst.size(), split_factor);
    return utils::rnd_up(dst_size, grf_size());
}

bool reduce_plan_t::can_split(int factor) const {
    if (!*this) return true;
    auto split_src = split(src, factor);
    if (split_src.is_empty()) return false;

    // Do not split by reduction dims.
    for (int i = 0; i < src.ndims(); i++) {
        if ((mask & (1 << i)) != 0 && split_src.dim(i) != src.dim(i))
            return false;
    }

    return true;
}

void reduce_plan_t::set_split(int factor) {
    if (!*this) return;
    ir_assert(can_split(factor));
    split_factor = factor;
}

stmt_t reduce_plan_t::create_stmt(
        const expr_t &src_buf, const expr_t &dst_buf) const {
    if (!*this) return stmt_t();
    auto stmt
            = create_reduce_stmt(src, dst, src_buf, dst_buf, tensor_t(), mask);
    return stmt;
}

int reduce_plan_t::estimate_regs() const {
    if (!*this) return 0;

    int ret = 0;
    ret += dst_buf_size();
    return utils::div_up(ret, grf_size());
}

std::string slm_plan_t::str() const {
    std::ostringstream oss;
    if (has_a()) {
        oss << "a_layout: " << a_layout << std::endl;
        oss << a_g2s_load.str("a_g2s_load") << std::endl;
        if (a_reorder) oss << a_reorder.str("a_reorder") << std::endl;
        oss << a_g2s_store.str("a_g2s_store") << std::endl;
    }
    if (has_b()) {
        oss << "b_layout: " << b_layout << std::endl;
        oss << b_g2s_load.str("b_g2s_load") << std::endl;
        if (b_reorder) oss << b_reorder.str("b_reorder") << std::endl;
        oss << b_g2s_store.str("b_g2s_store") << std::endl;
    }
    if (b_reduce) { oss << b_reduce.str("b_reduce") << std::endl; }
    return add_indent("slm_plan", oss.str());
}

std::string prefetch_plan_t::str() const {
    std::ostringstream oss;
    if (a_prefetch) oss << a_prefetch.str("a") << std::endl;
    if (b_prefetch) oss << b_prefetch.str("b") << std::endl;
    return add_indent("prefetch", oss.str());
}

bool x2r_plan_t::can_split(abc_kind_t abc, int factor) const {
    ir_assert(factor > 1);
    bool is_a = (abc == abc_kind_t::a);
    auto &load = (is_a ? a_load : b_load);
    auto &reorder = (is_a ? a_reorder : b_reorder);
    auto &layout = (is_a ? a_layout : b_layout);
    if (!layout.has_outer_block(factor)) return false;
    int dim_idx = layout.blocks().back().dim_idx;
    if (reorder && !reorder.src.has_outer_block(factor, dim_idx)) return false;
    if (!load.can_split(factor)) return false;
    if (abc == abc_kind_t::b && !b_reduce.can_split(factor)) return false;
    return true;
}

void x2r_plan_t::set_split(abc_kind_t abc, int factor) {
    // Reset split factors.
    a_load.set_split(1);
    a_reorder.set_split(1);
    b_load.set_split(1);
    b_reorder.set_split(1);
    b_reduce.set_split(1);
    split_abc = abc;
    split_factor = factor;
    switch (abc) {
        case abc_kind_t::a:
            a_load.set_split(factor);
            a_reorder.set_split(factor);
            break;
        case abc_kind_t::b:
            b_load.set_split(factor);
            b_reorder.set_split(factor);
            b_reduce.set_split(factor);
            break;
        default: break;
    }
}

int x2r_plan_t::estimate_regs(bool reuse_headers) const {
    int a_size = a_load.reg_buf_size();
    int b_size = b_load.reg_buf_size();
    if (a_reorder) a_size += a_load.reg_buf_size();
    if (b_reorder) b_size += b_load.reg_buf_size();
    int ret = 0;
    ret += utils::div_up(a_size, grf_size());
    ret += utils::div_up(b_size, grf_size());
    ret += a_load.estimate_regs(/*with_buffer=*/false, reuse_headers);
    ret += b_load.estimate_regs(/*with_buffer=*/false, reuse_headers);
    ret += b_reduce.estimate_regs();
    ret += a_reorder.estimate_regs();
    ret += b_reorder.estimate_regs();
    return ret;
}

std::string x2r_plan_t::str() const {
    std::ostringstream oss;
    oss << a_load.str("a_load") << std::endl;
    oss << b_load.str("b_load") << std::endl;
    if (b_reduce) oss << b_reduce.str("b_reduce") << std::endl;
    if (a_reorder) oss << a_reorder.str("a_reorder") << std::endl;
    if (b_reorder) oss << b_reorder.str("b_reorder") << std::endl;
    oss << "a_layout: " << a_layout << std::endl;
    oss << "b_layout: " << b_layout << std::endl;
    return add_indent("x2r_plan", oss.str());
}

bool fma_plan_t::can_split(abc_kind_t abc, int factor) const {
    ir_assert(factor > 1);
    bool is_a = (abc == abc_kind_t::a);
    bool is_m = is_a;
    auto &layout = is_a ? a_layout : b_layout;
    int mn_idx = is_a ? 1 : 2;
    int dim = (int)layout.dim(mn_idx);
    if (dim % factor != 0) return false;
    int blk = is_m ? m_blk : n_blk;
    if (blk > dim / factor) return false;
    auto &blocks = layout.blocks();
    if (blocks.empty()) return false;
    auto &b = blocks.back();
    if (b.dim_idx != mn_idx) return false;
    if ((int)b.block % factor != 0) return false;
    return true;
}

void fma_plan_t::set_split(abc_kind_t abc, int factor) {
    split_abc = abc;
    split_factor = factor;
}

int fma_plan_t::a_buf_size() const {
    int a_size = a_layout.size();
    if (split_abc == abc_kind_t::a)
        a_size = utils::div_up(a_size, split_factor);
    return utils::rnd_up(a_size, grf_size());
}

int fma_plan_t::b_buf_size() const {
    int b_size = b_layout.size();
    if (split_abc == abc_kind_t::b)
        b_size = utils::div_up(b_size, split_factor);
    return utils::rnd_up(b_size, grf_size());
}

int fma_plan_t::bmnk_split_idx(
        bmnk_kind_t bmnk, int split_off, bool is_start) const {
    int B = (int)a_layout.dim(0);
    int M = (int)a_layout.dim(1);
    int N = (int)b_layout.dim(2);
    int K = (int)a_layout.dim(2);
    int start[4] = {0, 0, 0, 0};
    int stop[4] = {B, M, N, K};
    bool split_a = (split_abc == abc_kind_t::a);
    bool split_b = (split_abc == abc_kind_t::b);
    bool is_m = (bmnk == bmnk_kind_t::m);
    bool is_n = (bmnk == bmnk_kind_t::n);
    int factor = 1;
    int off = 0;
    if ((split_a && is_m) || (split_b && is_n)) {
        factor = split_factor;
        off = split_off;
    }
    int i0 = start[(int)bmnk];
    int i1 = stop[(int)bmnk];
    ir_assert((i1 - i0) % factor == 0);
    int step = (i1 - i0) / factor;
    int idx = i0 + off * step;
    return is_start ? idx : idx + step;
}

int fma_plan_t::bmnk_start_idx(bmnk_kind_t bmnk, int subtile_idx) const {
    return bmnk_split_idx(bmnk, subtile_idx, true);
}

int fma_plan_t::bmnk_stop_idx(bmnk_kind_t bmnk, int subtile_idx) const {
    return bmnk_split_idx(bmnk, subtile_idx, false);
}

int fma_plan_t::estimate_regs() const {
    return utils::div_up(c_layout.size(), grf_size());
}

std::string fma_plan_t::str() const {
    std::ostringstream oss;
    oss << "a:     " << a_layout << std::endl;
    oss << "b:     " << b_layout << std::endl;
    oss << "c:     " << c_layout << std::endl;
    oss << "c_prb: " << c_prb_layout << std::endl;
    oss << "block: ";
    int blocks[4] = {b_blk, m_blk, n_blk, k_blk};
    for (int i = 0; i < 4; i++) {
        if (blocks[i] == 1) continue;
        oss << "bmnk"[i] << blocks[i];
    }
    return add_indent("fma", oss.str());
}

bool conv_plan_t::can_split(abc_kind_t abc, int factor) const {
    if (!fma.can_split(abc, factor)) return false;
    if (!x2r.can_split(abc, factor)) return false;
    if (zp && !zp.can_split(abc, factor)) return false;
    return true;
}

void conv_plan_t::set_split(abc_kind_t abc, int factor) {
    split_abc = abc;
    split_factor = factor;
    x2r.set_split(abc, factor);
    fma.set_split(abc, factor);
    if (zp) zp.set_split(abc, factor);
}

grf_usage_t conv_plan_t::grf_usage() const {
    bool with_headers = !reuse_headers;

    int out_buf_regs = 0;
    out_buf_regs += utils::div_up(fma.c_layout.size(), grf_size());
    out_buf_regs += utils::div_up(slm.b_reduce.dst_buf_size(), grf_size());
    out_buf_regs += utils::div_up(x2r.b_reduce.dst_buf_size(), grf_size());

    int gmem_load_buf_regs = 0;
    gmem_load_buf_regs += slm.a_g2s_load.estimate_regs(
            /*with_buffer=*/true, with_headers, reuse_headers);
    gmem_load_buf_regs += slm.b_g2s_load.estimate_regs(
            /*with_buffer=*/true, with_headers, reuse_headers);

    bool use_a_slm = x2r.a_load.send_params().is_slm();
    bool use_b_slm = x2r.b_load.send_params().is_slm();
    int a_g2r_buf_regs = use_a_slm
            ? 0
            : x2r.a_load.estimate_regs(/*with_buffer=*/true,
                    /*with_headers=*/false,
                    /*reuse_headers=*/false);
    int b_g2r_buf_regs = use_b_slm
            ? 0
            : x2r.b_load.estimate_regs(/*with_buffer=*/true,
                    /*with_headers=*/false,
                    /*reuse_headers=*/false);
    if (x2r.a_reorder && x2r.b_reorder) {
        ir_assert(!use_a_slm && !use_b_slm);
        // Reuse load buffer when both reorders are enabled.
        gmem_load_buf_regs += std::max(a_g2r_buf_regs, b_g2r_buf_regs);
    } else {
        if (!x2r.a_load.send_params().is_slm())
            gmem_load_buf_regs += a_g2r_buf_regs;
        if (!x2r.b_load.send_params().is_slm())
            gmem_load_buf_regs += b_g2r_buf_regs;
    }
    int gmem_load_regs = 0;
    gmem_load_regs += prefetch.a_prefetch.estimate_regs(
            /*with_buffer=*/false, with_headers, reuse_headers);
    gmem_load_regs += prefetch.b_prefetch.estimate_regs(
            /*with_buffer=*/false, with_headers, reuse_headers);
    if (!use_a_slm)
        gmem_load_regs += x2r.a_load.estimate_regs(
                /*with_buffer=*/false, with_headers, reuse_headers);
    if (!use_b_slm)
        gmem_load_regs += x2r.b_load.estimate_regs(
                /*with_buffer=*/false, with_headers, reuse_headers);
    gmem_load_regs += gmem_load_buf_regs;

    int slm_store_regs = 0;
    slm_store_regs += slm.a_g2s_store.estimate_regs(
            /*with_buffer=*/false, with_headers, reuse_headers);
    slm_store_regs += slm.b_g2s_store.estimate_regs(
            /*with_buffer=*/false, with_headers, reuse_headers);

    int slm_load_regs = 0;
    if (use_a_slm)
        slm_load_regs += x2r.a_load.estimate_regs(
                /*with_buffer=*/true, with_headers, reuse_headers);
    if (use_b_slm)
        slm_load_regs += x2r.b_load.estimate_regs(
                /*with_buffer=*/true, with_headers, reuse_headers);

    int reorder_regs = 0;
    reorder_regs += slm.a_reorder.estimate_regs();
    reorder_regs += slm.b_reorder.estimate_regs();
    reorder_regs += x2r.a_reorder.estimate_regs();
    reorder_regs += x2r.b_reorder.estimate_regs();

    int reused_header_regs = 0;
    if (reuse_headers) {
        for (auto *sp : {&prefetch.a_prefetch, &prefetch.b_prefetch,
                     &x2r.a_load, &x2r.b_load}) {
            reused_header_regs = std::max(
                    reused_header_regs, sp->estimate_regs(false, true, true));
        }
    }

    int zp_regs = zp.estimate_regs();

    grf_usage_t info(grf_size());
    info.add(grf_usage_label_t::out_buf, out_buf_regs);
    info.add(grf_usage_label_t::gmem_load, gmem_load_regs);
    info.add(grf_usage_label_t::slm_store, slm_store_regs);
    info.add(grf_usage_label_t::slm_load, slm_load_regs);
    info.add(grf_usage_label_t::reorder, reorder_regs);
    info.add(grf_usage_label_t::reused_headers, reused_header_regs);
    info.add(grf_usage_label_t::reserved, constants::reserved_regs);
    info.add(grf_usage_label_t::zero_points, zp_regs);
    return info;
}

void conv_plan_t::reset() {
    slm = slm_plan_t(hw);
    prefetch = prefetch_plan_t(hw);
    x2r = x2r_plan_t(hw);
    fma = fma_plan_t(hw);
    split_abc = abc_kind_t::undef;
    split_factor = 1;
    reuse_headers = false;
    max_gmem_bufs = 1;
}

std::string conv_plan_t::str() const {
    using namespace ir_utils;
    std::ostringstream oss;
    if (slm) oss << slm << std::endl;
    if (prefetch) oss << prefetch << std::endl;
    oss << x2r << std::endl;
    oss << fma << std::endl;
    if (zp) oss << zp << std::endl;
    oss << "a_can_split (2): " << to_string(can_split(abc_kind_t::a, 2))
        << std::endl;
    oss << "a_can_split (4): " << to_string(can_split(abc_kind_t::a, 4))
        << std::endl;
    oss << "b_can_split (2): " << to_string(can_split(abc_kind_t::b, 2))
        << std::endl;
    oss << "b_can_split (4): " << to_string(can_split(abc_kind_t::b, 4))
        << std::endl;
    if (split_abc != abc_kind_t::undef)
        oss << "split: " << split_abc << split_factor << std::endl;
    oss << "reuse_headers: " << to_string(reuse_headers) << std::endl;
    oss << grf_usage() << std::endl;
    return jit::add_indent("conv_plan", oss.str());
}

struct fma_config_t {
    fma_config_t(const conv_config_t &cfg) {
        simd = cfg.simd();
        vec_size = cfg.vec_size();
        fma = cfg.fma_kind();
        a_type = type_t(cfg.prb().a_data_type);
        b_type = type_t(cfg.prb().b_data_type);
        is_src1_broadcast = !cfg.prb().is_dw;
    }

    int simd;
    int vec_size;
    fma_kind_t fma;
    type_t a_type;
    type_t b_type;
    bool is_src1_broadcast;
};

bool check_k_blocks_order(layout_t &a, layout_t &b,
        const bmnk_mapper_t &bmnk_mapper, bool overwrite_a_b = false) {
    object_map_t<expr_t, int> k_vars;
    auto k_sub_layout = [&](abc_kind_t abc_kind, const layout_t &l) {
        layout_t k_layout = layout_t(
                type_t::u8(), 0, std::vector<dim_t>(layout_t::max_ndims, 1));
        for (auto &b : l.blocks()) {
            auto bmnk_kind = bmnk_mapper.bmnk_kind(abc_kind, b.dim_idx);
            if (bmnk_kind != bmnk_kind_t::k) continue;
            auto &var = bmnk_mapper.var(abc_kind, b.dim_idx);
            auto ret = k_vars.emplace(var, (int)k_vars.size());
            k_layout = k_layout.add_outer_block(ret.first->second, b.block);
        }
        return k_layout;
    };
    auto a_k = k_sub_layout(abc_kind_t::a, a);
    auto b_k = k_sub_layout(abc_kind_t::b, b);
    if (overwrite_a_b) {
        a = a_k;
        b = b_k;
    }
    return (a_k == b_k);
}

layout_t get_fma_friendly_layout(const fma_config_t &fma_cfg, abc_kind_t abc,
        const bmnk_mapper_t &mapper, const layout_t &layout,
        layout_t other_layout = layout_t()) {
    auto simd = fma_cfg.simd;
    auto fma = fma_cfg.fma;
    bool is_mad = (fma == fma_kind_t::mad);
    bool is_dpas = utils::one_of(fma, fma_kind_t::dpas, fma_kind_t::dpasw);
    bool is_a = (abc == abc_kind_t::a);
    bool is_b = !is_a;
    auto type = (is_a ? fma_cfg.a_type : fma_cfg.b_type);
    int type_size = type.size();

    if (is_dpas) {
        int sdepth = 8;
        int dword_size = 4;
        std::vector<std::pair<int, dim_t>> blocks;
        std::vector<bmnk_kind_t> bmnks;
        if (is_a) {
            // A -> src2
            int k_blk = sdepth * dword_size / type_size;
            blocks.emplace_back(1, k_blk);
            bmnks.push_back(bmnk_kind_t::m);
            bmnks.push_back(bmnk_kind_t::k);
        } else {
            // B -> src1
            int k_blk0 = dword_size / type_size;
            int n_blk = simd;
            int k_blk1 = sdepth;
            blocks.emplace_back(0, k_blk1);
            blocks.emplace_back(1, n_blk);
            blocks.emplace_back(0, k_blk0);
            bmnks.push_back(bmnk_kind_t::k);
            bmnks.push_back(bmnk_kind_t::n);
        }
        auto bmnk_layout = mapper.map_to_bmnk(abc, bmnks, layout).retype(type);
        auto fma_layout
                = bmnk_layout.make_with_block(layout_t(type, 0, blocks));
        auto bmnk_mapper = mapper;
        auto abc_layout = mapper.map_from_bmnk(abc, bmnks, fma_layout, layout);
        if (!other_layout.is_empty()) {
            auto other_kind
                    = abc == abc_kind_t::a ? abc_kind_t::b : abc_kind_t::a;
            bmnk_block_mapper_t from_bmnk_mapper(bmnk_mapper);
            from_bmnk_mapper.push_blocks(abc, layout.blocks());
            from_bmnk_mapper.push_blocks(other_kind, other_layout.blocks());
            bool k_order;
            if (abc == abc_kind_t::a)
                k_order = check_k_blocks_order(
                        fma_layout, other_layout, bmnk_mapper);
            else
                k_order = check_k_blocks_order(
                        other_layout, fma_layout, bmnk_mapper);
            if (k_order && fma_layout == bmnk_layout) return abc_layout;
            bmnk_block_mapper_t to_bmnk_mapper(bmnk_mapper);

            for (auto &b : layout.blocks()) {
                if (bmnk_mapper.bmnk_kind(abc, b.dim_idx) == bmnk_kind_t::k)
                    continue;
                to_bmnk_mapper.push_block(abc, b);
            }
            for (auto a : other_layout.blocks()) {
                if (bmnk_mapper.bmnk_kind(other_kind, a.dim_idx)
                        == bmnk_kind_t::k) {
                    //map a to b dim
                    a.dim_idx = bmnk_mapper.dim_idx(
                            abc, bmnk_mapper.var(other_kind, a.dim_idx));
                    to_bmnk_mapper.push_block(abc, a);
                }
            }
            fma_layout = to_bmnk_mapper.map_from_bmnk(abc, bmnks, fma_layout);
            fma_layout = fma_layout.make_dense();
            return fma_layout;
        }
        return abc_layout;
    }

    if (is_mad) {
        // mad with s8/u8 is not supported, promote to strided s16.
        if (type.is_x8()) return layout.retype(type_t::s16()).make_strided(2);

        // mad with f16 requires aligned regioning for src1/src2.
        if (fma_cfg.a_type.is_f16()) return layout.make_dense();

        if (fma_cfg.a_type.is_bf16()) {
            // bf16 mixed mode requires src1 to be converted to f32 when it's
            // broadcasted.
            if (is_a && fma_cfg.is_src1_broadcast)
                return layout.retype(type_t::f32()).make_dense();
            // bf16 mixed mode mad requires src1 to be packed
            if (is_a) return layout.make_dense();
            // bf16 mixed mode mad requires src2 to be f32.
            if (is_b) return layout.retype(type_t::f32()).make_dense();
        }

        return layout;
    }

    ir_error_not_expected();
    return layout;
}

layout_t get_fma_friendly_layout(const fma_config_t &fma_cfg, abc_kind_t abc,
        const bmnk_mapper_t &mapper, const view_t &view) {
    return get_fma_friendly_layout(
            fma_cfg, abc, mapper, view.create_dense_vlayout());
}

dim_t find_min_stride_without_conflicts(
        ngen::HW hw, dim_t inner_bytes, dim_t dense_stride_bytes) {
    int write_step = 64;
    int stride_step = 16;
    dim_t stride_beg = dense_stride_bytes;
    dim_t stride_end = 2 * dense_stride_bytes;
    auto arch = convert_ngen_arch_to_dnnl(hw);
    const int slm_banks = compute::device_info_t::slm_memory_bank_count(arch);
    const int bank_granularity
            = compute::device_info_t::slm_memory_bank_granularity(arch);
    for (dim_t s = stride_beg; s < stride_end; s += stride_step) {
        bool ok = true;
        for (dim_t off0 = 0; off0 < inner_bytes; off0 += write_step) {
            // Check banks for a single SLM write.
            std::vector<bool> found(slm_banks, false);
            for (dim_t off = off0; off < off0 + write_step;
                    off += bank_granularity) {
                int bank0 = (off / bank_granularity) % slm_banks;
                int bank1 = ((off + s) / bank_granularity) % slm_banks;
                if (found[bank0]) {
                    ok = false;
                    break;
                }
                found[bank0] = true;
                if (found[bank1]) {
                    ok = false;
                    break;
                }
                found[bank1] = true;
            }
            if (ok) return s;
        }
    }

    ir_warning() << "Couldn't find stride without conflicts for SLM padding."
                 << std::endl;

    return dense_stride_bytes;
}

layout_t pad_slm_layout(
        ngen::HW hw, const layout_t &layout, const grid_info_t &grid) {
    // EUs are not fused in XeHPC+ so no need to pad SLM.
    if (hw >= ngen::HW::XeHPC) return layout;
    auto tg_dim0 = grid.dim(0);
    auto tg_dim1 = grid.dim(1);
    int type_size = layout.type().size();

    ir_assert(layout.elems() % tg_dim0 == 0) << layout;
    dim_t inner_block = layout.elems() / tg_dim0;

    ir_assert((inner_block * type_size) % tg_dim1 == 0) << layout;
    dim_t per_thr_bytes = (inner_block * type_size) / tg_dim1;

    std::vector<dim_t> multi_blocks = {inner_block, tg_dim0};
    auto l = layout.split_into_multi_blocks(multi_blocks);

    if (l.is_empty()) {
        ir_warning() << "Couldn't split layout for SLM padding." << std::endl;
        return layout;
    }
    auto padded_blocks = l.blocks();
    dim_t stride = -1;
    dim_t remaining_elems = inner_block;
    bool past_inner_block = remaining_elems == 1;
    for (auto &b : padded_blocks) {
        if (past_inner_block) {
            if (stride == -1) {
                dim_t stride_bytes = find_min_stride_without_conflicts(
                        hw, per_thr_bytes, dim_t(b.stride) * type_size);
                ir_assert(stride_bytes % type_size == 0);
                stride = stride_bytes / type_size;
            }
            b.stride = stride;
            stride = b.stride * b.block;
            continue;
        }
        ir_assert(remaining_elems % b.block == 0);
        remaining_elems /= b.block;
        if (remaining_elems == 1) past_inner_block = true;
    }
    return layout_t(
            layout.type(), layout.ndims(), layout.offset(), padded_blocks);
}

layout_t get_slm_layout(const conv_config_t &cfg, abc_kind_t abc,
        const bmnk_mapper_t &mapper, const view_t &tg_view,
        const grid_info_t &grid) {
    fma_config_t fma_cfg(cfg);
    auto layout = get_fma_friendly_layout(fma_cfg, abc, mapper, tg_view);
    layout = pad_slm_layout(cfg.hw(), layout, grid);
    return layout;
}

type_t get_default_accumulation_type(const type_t &a, const type_t &b) {
    UNUSED(b);
    if (a.is_int()) return type_t::s32();
    if (a.is_f64()) return type_t::f64();
    return type_t::f32();
}

struct reduce_mask_t {
    reduce_mask_t() = default;
    reduce_mask_t(uint32_t mask) : enable(true), mask(mask) {}

    explicit operator bool() const { return enable; }
    bool has(int idx) const { return (mask & (1 << idx)) != 0; }

    bool enable = false;
    uint32_t mask;
};

bool do_reduce(const conv_config_t &cfg, abc_kind_t abc) {
    auto &prb = cfg.prb();
    return (abc == abc_kind_t::b) && prb.is_bwd_w && prb.with_bias;
}

reduce_mask_t reduce_mask(const conv_config_t &cfg, abc_kind_t abc) {
    if (!do_reduce(cfg, abc)) return reduce_mask_t();
    // Reduce by g and oc.
    return reduce_mask_t((1 << 1) | (1 << 2));
}

std::vector<int> get_reduce_dim_map(uint32_t mask, int &ndims) {
    std::vector<int> ret(ndims, -1);
    int dst_idx = 0;
    for (int i = 0; i < ndims; i++) {
        if ((mask & (1 << i)) == 0) continue;
        ret[i] = dst_idx++;
    }
    ndims = dst_idx;
    return ret;
}

tensor_t to_reduce_tensor(const tensor_t &tile, uint32_t mask) {
    int reduce_ndims = tile.ndims();
    auto map = get_reduce_dim_map(mask, reduce_ndims);
    std::vector<dim_t> reduce_dims(reduce_ndims);
    std::vector<expr_t> reduce_start(reduce_ndims);
    for (int i = 0; i < tile.ndims(); i++) {
        if (map[i] == -1) continue;
        reduce_dims[map[i]] = tile(i);
        reduce_start[map[i]] = tile.start(i);
    }
    return tensor_t(reduce_dims, reduce_start);
}

layout_t to_reduce_layout(const layout_t &layout, uint32_t mask) {
    int reduce_ndims = layout.ndims();
    auto map = get_reduce_dim_map(mask, reduce_ndims);
    std::vector<block_t> reduce_blocks;
    for (auto &b : layout.blocks()) {
        if (map[b.dim_idx] == -1) continue;
        auto bb = b;
        bb.dim_idx = map[b.dim_idx];
        reduce_blocks.push_back(bb);
    }
    auto type = get_default_accumulation_type(layout.type(), layout.type());
    return layout_t(type, reduce_ndims, 0, reduce_blocks).make_dense();
}

class direct_view_t {
public:
    direct_view_t() = default;

    direct_view_t(const view_t &view, bool active)
        : view_(view), active_(active) {
        if (!check_tdims(fused_tidx_)) return;
        if (!check_masks()) return;
        init_direct_view();
    }

    explicit operator bool() const {
        return active_ && !direct_view_.is_empty();
    }

    const view_t &get() const { return direct_view_; }

    layout_t transform(const layout_t &layout) const {
        ir_assert((bool)*this);
        ir_assert(fused_tidx_ != -1);
        std::vector<block_t> blocks;
        bool seen = false;
        for (auto &b : layout.blocks()) {
            if (b.block == 1) continue;
            auto &tdim = view_.tdim(b.dim_idx);
            if (b.dim_idx != fused_tidx_) {
                auto vb = b;
                vb.dim_idx = tdim.vidx(0);
                blocks.push_back(vb);
                continue;
            }
            ir_assert(!seen);
            seen = true;
            for (int i = 0; i < tdim.nvargs(); i++) {
                int vidx = tdim.vidx(i);
                dim_t vstride = (dim_t)tdim.vstride(i);
                auto vb = b;
                vb.dim_idx = vidx;
                vb.block = view_.vdims()[vidx];
                vb.stride = b.stride * vstride;
                blocks.push_back(vb);
            }
        }

        return layout_t(layout.type(), view_.nvdims(), 0, blocks);
    }

private:
    bool check_tdims(int &fused_tidx) const {
        int nfused = 0;
        for (int tidx = 0; tidx < view_.ntdims(); tidx++) {
            auto &tdim = view_.tdim(tidx);
            if (tdim.is_identity()) continue;
            int nvars = 0;
            for (int i = 0; i < tdim.nvargs(); i++) {
                auto vdim = view_.vdims()[tdim.vidx(i)];
                if (vdim == 1) continue;
                if (tdim.vstride(i).is_unknown()) return false;
                nvars++;
            }
            if (nvars >= 2) {
                nfused++;
                fused_tidx = tidx;
            }
        }
        if (nfused != 1) return false;
        return true;
    }

    bool check_masks() const {
        for (int tidx = 0; tidx < view_.ntdims(); tidx++) {
            auto &tdim = view_.tdim(tidx);
            for (auto &v : view_.vvars())
                if (contains_object(tdim.mask(), v)) return false;
        }
        return true;
    }

    static std::vector<expr_t> create_vvars(int nvdims) {
        static const int max_nvdims = 128;
        static thread_local std::vector<expr_t> _vvars([] {
            std::vector<expr_t> ret;
            ret.reserve(max_nvdims);
            for (int i = 0; i < max_nvdims; i++)
                ret.push_back(
                        var_t::make(type_t::s32(), "_" + std::to_string(i)));
            return ret;
        }());

        ir_assert(nvdims <= max_nvdims) << "Too many dimensions: " << nvdims;
        return std::vector<expr_t>(_vvars.begin(), _vvars.begin() + nvdims);
    }

    void init_direct_view() {
        std::vector<dim_t> tdim_extents(view_.ntdims());
        std::vector<expr_t> tdim_starts(view_.ntdims());
        for (int tidx = 0; tidx < view_.ntdims(); tidx++) {
            auto &tdim = view_.tdim(tidx);
            auto &textent = tdim_extents[tidx];
            auto &tstart = tdim_starts[tidx];
            if (tdim.is_identity()) {
                textent = view_.vdims()[tdim.vidx(0)];
                tstart = view_.vstart(tdim.vidx(0));
                continue;
            }
            textent = 1;
            tstart = tdim.expr();
            for (int i = 0; i < tdim.nvargs(); i++) {
                auto vidx = tdim.vidx(i);
                auto vdim = view_.vdims()[vidx];
                textent += (dim_t)tdim.vstride(i) * (vdim - 1);
                tstart = substitute(
                        tstart, view_.vvar(vidx), view_.vstart(vidx));
            }
        }

        direct_view_ = view_t(create_vvars(view_.ntdims()), view_.ntdims());
        direct_view_.set_tlayout(view_.tlayout());
        for (int tidx = 0; tidx < view_.ntdims(); tidx++) {
            auto &tdim = view_.tdim(tidx);
            direct_view_.set_vdim(direct_view_.vvars()[tidx],
                    tdim_extents[tidx], tdim_starts[tidx]);
            direct_view_.set_tdim(
                    tidx, direct_view_.vvars()[tidx], tdim.mask());
        }
        // TODO: Simplify tstart.
    }

    view_t view_;
    int fused_tidx_ = -1;
    view_t direct_view_;
    bool active_ = false;
};

layout_t add_batch(const layout_t &layout) {
    auto blocks = layout.blocks();
    for (auto &b : blocks) {
        b.dim_idx++;
    }
    return layout_t(layout.type(), layout.ndims(), layout.offset(), blocks);
}

bool is_dpas_src1_compatible(int simd, bool transpose, const layout_t &layout) {
    const int sdepth = 8;
    auto &type = layout.type();
    auto c_type = (type.is_int() ? type_t::s32() : type_t::f32());
    auto func = dpas_t::make(
            /*is_dpasw=*/false, simd, sdepth, /*rcount=*/1, c_type, type, type);
    auto &dpas = func.as<dpas_t>();
    auto src1_layout = dpas.a_layout();
    if (transpose) src1_layout = src1_layout.transpose();
    src1_layout = add_batch(src1_layout);
    return src1_layout <= layout;
}

bool is_dpas_src2_compatible(int simd, bool transpose, const layout_t &layout) {
    const int sdepth = 8;
    auto &type = layout.type();
    auto c_type = (type.is_int() ? type_t::s32() : type_t::f32());
    auto func = dpas_t::make(
            /*is_dpasw=*/false, simd, sdepth, /*rcount=*/1, c_type, type, type);
    auto &dpas = func.as<dpas_t>();
    auto src2_layout = dpas.b_layout();
    if (transpose) src2_layout = src2_layout.transpose();
    src2_layout = add_batch(src2_layout);
    return src2_layout <= layout;
}

int get_dpas_block_rcount(const layout_t &layout, int dim_idx) {
    if (layout.nblocks() < 2) return 1;

    auto &b0 = layout.blocks()[0];
    if (b0.block * layout.type().size() > 32) return 1;

    auto &b1 = layout.blocks()[1];
    if (b1.dim_idx != dim_idx) return 1;

    int block_rcount = (int)b1.block;
    int max_rcount = 8;

    if (block_rcount % max_rcount == 0) return max_rcount;

    return block_rcount;
}

bool is_mad_compatible(int vec_size, const layout_t &a, const layout_t &b,
        int a_vec_idx, int b_vec_idx) {
    if (a_vec_idx != -1 && !a.is_blocked_by(a_vec_idx, vec_size)) return false;
    if (b_vec_idx != -1 && !b.is_blocked_by(b_vec_idx, vec_size)) return false;
    return true;
}

layout_t get_c_layout(const layout_t &a_layout, const layout_t &b_layout,
        const layout_t &c_blk_layout) {
    std::vector<block_t> blocks;
    const bmnk_kind_t a_bmnks[3]
            = {bmnk_kind_t::b, bmnk_kind_t::m, bmnk_kind_t::k};
    const bmnk_kind_t b_bmnks[3]
            = {bmnk_kind_t::b, bmnk_kind_t::k, bmnk_kind_t::n};
    for (auto &b : a_layout.blocks()) {
        if (a_bmnks[b.dim_idx] == bmnk_kind_t::k) continue;
        blocks.push_back(b);
    }
    for (auto &b : b_layout.blocks()) {
        if (utils::one_of(b_bmnks[b.dim_idx], bmnk_kind_t::b, bmnk_kind_t::k))
            continue;
        blocks.push_back(b);
    }

    layout_t c_layout(c_blk_layout.type(), c_blk_layout.ndims(), 0, blocks);
    c_layout = c_layout.make_with_block(c_blk_layout);
    return c_layout;
}

enum class plan_status_t {
    success,
    error,
    ab_layout_mismatch,
    invalid_layout,
    invalid_send,
    out_of_registers,
};

bool is_layout_error(plan_status_t status) {
    return utils::one_of(status, plan_status_t::invalid_layout,
            plan_status_t::ab_layout_mismatch);
}

#define PLAN_CHECK(status) \
    do { \
        if ((status) != plan_status_t::success) return (status); \
    } while (false)

class plan_builder_t {
public:
    plan_builder_t(conv_config_t &cfg)
        : cfg_(cfg)
        , prb_(cfg.prb())
        , plan_ptr_(get_plan())
        , plan_(*plan_ptr_)
        , gemm_schedule_(plan_.gemm_schedule) {}

    status_t init_plan() {
        plan_status_t status;
        status = try_init_plan();
        if (status == plan_status_t::success) return status::success;

        if (is_layout_error(status)) {
            ir_trace() << "Retry plan initialization without 2D messages"
                       << std::endl;
            for (auto ab : {0b01, 0b10, 0b0}) {
                bool a = ((ab & 0b01) != 0);
                bool b = ((ab & 0b10) != 0);
                enable_send_2d(abc_kind_t::a, a);
                enable_send_2d(abc_kind_t::b, b);
                status = try_init_plan();
                if (status == plan_status_t::success) return status::success;
                if (!is_layout_error(status)) break;
            }
        }

        if (a_direct_view_ || b_direct_view_) {
            ir_trace() << "Retry plan initialization without direct view"
                       << std::endl;
            enable_direct_view(false);
            status = try_init_plan();
            if (status == plan_status_t::success) return status::success;
        }

        if (plan_.slm && !cfg_.slm().is_overridden()) {
            ir_trace() << "Retry plan initialization without SLM" << std::endl;
            enable_slm(false);
            status = try_init_plan();
            if (status == plan_status_t::success) return status::success;
        }

        // Can't create convolution plan.
        return status::runtime_error;
    }

private:
    void set_plan() {
        auto &slm = plan_.slm;
        if (slm) {
            int gmem_buf_size = 0;
            gmem_buf_size += slm.a_g2s_load.estimate_regs(
                    /*with_buffer=*/true, /*with_headers*/ false,
                    plan_.reuse_headers);
            gmem_buf_size += slm.b_g2s_load.estimate_regs(
                    /*with_buffer=*/true, /*with_headers=*/false,
                    plan_.reuse_headers);
            int bound = cfg_.regs() - 5;
            int free = std::max(0, bound - plan_.grf_usage().total());
            plan_.max_gmem_bufs
                    = gmem_buf_size == 0 ? 0 : 1 + free / gmem_buf_size;
        }

        ir_trace() << plan_ << std::endl;
        cfg_.set_plan(plan_ptr_);

        // Propagate plan parameters to config.
        bool do_unroll = cfg_.pipeline().do_unroll();
        if (plan_.reuse_headers) do_unroll = false;
        cfg_.pipeline().set(do_unroll, plan_.reuse_headers);
        cfg_.set_ow_kw_grf_cache((bool)a_direct_view_);
    }

    plan_status_t try_init_plan() {
        plan_.reset();
        PLAN_CHECK(init_x_g2r_direct_view(gemm_schedule_.a_tg_view(),
                gemm_schedule_.a_thr_tile(), a_direct_view_));
        PLAN_CHECK(init_x_g2r_direct_view(gemm_schedule_.b_tg_view(),
                gemm_schedule_.b_thr_tile(), b_direct_view_));
        PLAN_CHECK(init_slm_plan(plan_.slm));
        PLAN_CHECK(init_prefetch_plan(plan_.prefetch));
        PLAN_CHECK(init_x2r_plan(plan_.slm, plan_.x2r));
        PLAN_CHECK(init_fma_plan(plan_.x2r, plan_.fma));
        PLAN_CHECK(init_zp_plan(plan_.x2r, plan_.fma, plan_.zp));
        if (cfg_.subtiles().is_env_overridden()) {
            int a = cfg_.subtiles().a();
            int b = cfg_.subtiles().b();
            if (a > 1) plan_.set_split(abc_kind_t::a, a);
            if (b > 1) plan_.set_split(abc_kind_t::b, b);
        }
        PLAN_CHECK(fixup_grf_usage(plan_));
        set_plan();
        return plan_status_t::success;
    }

    plan_status_t fixup_grf_usage(conv_plan_t &plan) const {
        // XXX: This is an estimation, CSE pass does more accurate counting to
        // not exceed available GRF space.
        int tmp_regs = 5;
        int bound = cfg_.regs() - tmp_regs;
        if (plan.grf_usage().total() < bound) return plan_status_t::success;

        plan_status_t status;

        status = try_drop_prefetch(plan, bound);
        if (status == plan_status_t::success) return plan_status_t::success;

        status = try_apply_ab_split(plan, bound);
        if (status == plan_status_t::success) return plan_status_t::success;

        status = try_reuse_headers(plan, bound);
        if (status == plan_status_t::success) return plan_status_t::success;

        return plan_status_t::out_of_registers;
    }

    plan_status_t try_apply_ab_split(conv_plan_t &plan, int bound) const {
        if (cfg_.subtiles().is_env_overridden())
            return plan_status_t::out_of_registers;
        int min_regs = plan.grf_usage().total();
        auto min_split = std::make_pair(abc_kind_t::undef, 1);
        for (int factor : {2, 4}) {
            for (abc_kind_t abc : {abc_kind_t::a, abc_kind_t::b}) {
                if (plan.can_split(abc, factor)) {
                    plan.set_split(abc, factor);
                    int regs = plan.grf_usage().total();
                    if (regs < bound) return plan_status_t::success;
                    if (regs < min_regs) {
                        min_split = std::make_pair(abc, factor);
                        min_regs = regs;
                    }
                }
            }
        }
        auto abc = min_split.first;
        auto factor = min_split.second;
        plan.set_split(abc, factor);
        return plan_status_t::out_of_registers;
    }

    plan_status_t try_reuse_headers(conv_plan_t &plan, int bound) const {
        if (cfg_.pipeline().is_env_overridden())
            return plan_status_t::out_of_registers;
        plan.reuse_headers = true;
        if (plan.grf_usage().total() < bound) return plan_status_t::success;
        return plan_status_t::out_of_registers;
    }

    plan_status_t try_drop_prefetch(conv_plan_t &plan, int bound) const {
        if (cfg_.prefetch().is_env_overridden())
            return plan_status_t::out_of_registers;
        auto &prefetch = plan.prefetch;
        if (prefetch.a_prefetch) {
            prefetch.a_prefetch = send_plan_t();
            if (plan.grf_usage().total() < bound) return plan_status_t::success;
        }
        if (prefetch.b_prefetch) {
            prefetch.b_prefetch = send_plan_t();
            if (plan.grf_usage().total() < bound) return plan_status_t::success;
        }
        return plan_status_t::out_of_registers;
    }

    plan_status_t init_x_g2r_direct_view(const view_t &tg_view,
            const tensor_t &thr_tile, direct_view_t &direct_view) const {
        auto gmem_view = tg_view.create_sub_view(thr_tile);
        direct_view = direct_view_t(gmem_view, allow_direct_view_);
        return plan_status_t::success;
    }

    bool use_prefetch(abc_kind_t abc) const {
        auto &prb = cfg_.prb();
        bool is_a = (abc == abc_kind_t::a);
        if (cfg_.prefetch().is_overridden()) {
            return is_a ? cfg_.prefetch().a() : cfg_.prefetch().b();
        }
        if (cfg_.hw() < ngen::HW::XeHPC) return false;
        if (is_a && !prb.is_bwd_d && is_small_ic(prb) && cfg_.is_dp_fma())
            return false;
        auto &tg = cfg_.thread_group_grid();
        if (prb.is_bwd_w
                && (!cfg_.pipeline().do_unroll()
                        && (tg[1] == 1 || tg[0] >= tg[1])))
            return false;
        return true;
    }

    bool use_slm(abc_kind_t abc) const {
        bool is_a = (abc == abc_kind_t::a);
        if (cfg_.slm().is_overridden()) {
            return is_a ? cfg_.slm().a() : cfg_.slm().b();
        }

        if (!allow_slm_) return false;
        if (cfg_.hw() >= ngen::HW::XeHPC) return false;

        auto &tg = cfg_.thread_group_grid();
        int tg_idx = (is_a ? 0 : 1);
        if (tg[tg_idx] == 1) return false;

        auto &direct_view = (is_a ? a_direct_view_ : b_direct_view_);
        if ((bool)direct_view) return false;

        return true;
    }

    plan_status_t init_x_slm_plan(abc_kind_t abc, const view_t &tg_view,
            layout_t &slm_layout, grid_info_t &grid, send_plan_t &g2s_load,
            send_plan_t &g2s_store, reorder_plan_t &reorder,
            reduce_mask_t reduce_mask = reduce_mask_t(),
            reduce_plan_t *reduce = nullptr,
            tensor_t *reduce_tile = nullptr) const {
        if (!use_slm(abc)) return plan_status_t::success;
        auto &tg = cfg_.thread_group_grid();
        slm_layout = get_slm_layout(
                cfg_, abc, gemm_schedule_.bmnk_mapper(), tg_view, tg);
        auto thr_tile = slm_layout.split(tg, &grid);
        auto abs_thr_tile = tg_view.vtile().create_sub_tensor(thr_tile);
        auto slm_thr_layout = slm_layout.map(thr_tile);
        auto slm_thr_view = view_t(slm_thr_layout);
        auto thr_view = tg_view.create_sub_view(thr_tile);
        auto load_params = get_send_params(cfg_.exec_cfg(), send_op_t::load,
                send_address_t::a64, abc, thr_view);
        auto store_params = get_send_params(cfg_.exec_cfg(), send_op_t::store,
                send_address_t::slm, abc, slm_thr_view);
        g2s_load = create_send_plan(cfg_.exec_cfg(), thr_view, load_params);
        g2s_store
                = create_send_plan(cfg_.exec_cfg(), slm_thr_view, store_params);
        auto &src = g2s_load.reg_layout();
        auto &dst = g2s_store.reg_layout();
        reorder = create_reorder_plan(cfg_.hw(), src, dst);
        if (reduce_mask) {
            *reduce_tile = to_reduce_tensor(abs_thr_tile, reduce_mask.mask);
            auto reduce_layout = to_reduce_layout(src, reduce_mask.mask);
            *reduce = create_reduce_plan(
                    cfg_.hw(), src, reduce_layout, reduce_mask.mask);
        }
        if (g2s_store.is_scattered()) {
            // Do not use SLM with scattered SLM stores.
            slm_layout = layout_t();
            grid = grid_info_t();
            g2s_load = send_plan_t();
            g2s_store = send_plan_t();
            reorder = reorder_plan_t();
            if (reduce_mask) {
                *reduce_tile = tensor_t();
                *reduce = reduce_plan_t();
            }
        }
        return plan_status_t::success;
    }

    plan_status_t init_slm_plan(slm_plan_t &plan) const {
        PLAN_CHECK(init_x_slm_plan(abc_kind_t::a, gemm_schedule_.a_tg_view(),
                plan.a_layout, plan.a_grid, plan.a_g2s_load, plan.a_g2s_store,
                plan.a_reorder));
        PLAN_CHECK(init_x_slm_plan(abc_kind_t::b, gemm_schedule_.b_tg_view(),
                plan.b_layout, plan.b_grid, plan.b_g2s_load, plan.b_g2s_store,
                plan.b_reorder, reduce_mask(cfg_, abc_kind_t::b),
                &plan.b_reduce, &plan.b_reduce_tile));
        return plan_status_t::success;
    }

    plan_status_t init_x_prefetch_plan(abc_kind_t abc, const view_t &tg_view,
            grid_info_t &grid, send_plan_t &prefetch) const {
        if (!use_prefetch(abc)) return plan_status_t::success;
        auto &tg = cfg_.thread_group_grid();
        auto thr_view = tg_view.split(tg, &grid);
        auto params = get_send_params(cfg_.exec_cfg(), send_op_t::prefetch,
                send_address_t::a64, fma_kind_t::unknown, abc, thr_view,
                gemm_schedule_);
        prefetch = create_send_plan(cfg_.exec_cfg(), thr_view, params);
        return plan_status_t::success;
    }

    plan_status_t init_prefetch_plan(prefetch_plan_t &plan) const {
        PLAN_CHECK(init_x_prefetch_plan(abc_kind_t::a,
                gemm_schedule_.a_tg_view(), plan.a_grid, plan.a_prefetch));
        PLAN_CHECK(init_x_prefetch_plan(abc_kind_t::b,
                gemm_schedule_.b_tg_view(), plan.b_grid, plan.b_prefetch));
        return plan_status_t::success;
    }

    plan_status_t init_x_s2r_plan(abc_kind_t abc, bool has_x_slm,
            const layout_t &slm_layout, const tensor_t &thr_tile,
            send_plan_t &load, layout_t &layout) const {
        if (!has_x_slm) return plan_status_t::success;
        auto thr_view = view_t(slm_layout).create_sub_view(thr_tile);
        auto params = get_send_params(cfg_.exec_cfg(), send_op_t::load,
                send_address_t::slm, abc, thr_view);
        load = create_send_plan(cfg_.exec_cfg(), thr_view, params);
        layout = load.reg_layout();
        if (load.is_scattered()) {
            // Do not use SLM with scattered SLM load.
            return plan_status_t::invalid_send;
        }
        return plan_status_t::success;
    }

    plan_status_t init_x_g2r_plan(abc_kind_t abc, bool has_x_slm,
            const view_t &tg_view, const tensor_t &thr_tile,
            const tensor_t &abs_thr_tile, send_plan_t &load,
            reorder_plan_t &reorder, layout_t &layout,
            reduce_mask_t reduce_mask = reduce_mask_t(),
            reduce_plan_t *reduce = nullptr, tensor_t *reduce_tile = nullptr,
            const layout_t &other_layout = layout_t()) const {
        if (has_x_slm) return plan_status_t::success;
        auto gmem_view = tg_view.create_sub_view(thr_tile);

        auto &direct_view
                = (abc == abc_kind_t::a ? a_direct_view_ : b_direct_view_);
        auto load_view = direct_view ? direct_view.get() : gmem_view;

        auto params = get_send_params(cfg_.exec_cfg(), send_op_t::load,
                send_address_t::a64, cfg_.fma_kind(), abc, load_view,
                gemm_schedule_,
                /*allow_2d_load=*/true);
        load = create_send_plan(cfg_.exec_cfg(), load_view, params);

        auto reg_layout = load.reg_layout();
        if (direct_view) reg_layout = direct_view.transform(reg_layout);

        if (reduce_mask) {
            ir_assert(!direct_view);
            *reduce_tile = to_reduce_tensor(abs_thr_tile, reduce_mask.mask);
            auto reduce_layout = to_reduce_layout(reg_layout, reduce_mask.mask);
            *reduce = create_reduce_plan(
                    cfg_.hw(), reg_layout, reduce_layout, reduce_mask.mask);
        }

        fma_config_t fma_cfg(cfg_);
        layout = get_fma_friendly_layout(fma_cfg, abc,
                gemm_schedule_.bmnk_mapper(), reg_layout, other_layout);
        auto &src = reg_layout;
        auto &dst = layout;
        reorder = create_reorder_plan(cfg_.hw(), src, dst);
        return plan_status_t::success;
    }

    plan_status_t verify_2d() const {
        auto &a = plan_.x2r.a_load.send_params().hint_2d;
        auto &b = plan_.x2r.b_load.send_params().hint_2d;
        int a_vnni_factor = a.enable ? a.vnni_permute_factor : 0;
        int b_vnni_factor = b.enable ? b.vnni_permute_factor : 0;
        if (a_vnni_factor != b_vnni_factor)
            return plan_status_t::ab_layout_mismatch;
        return plan_status_t::success;
    }

    plan_status_t fixup_k_blocks_order(layout_t &a, layout_t &b) const {
        auto bmnk_mapper = gemm_schedule_.bmnk_mapper();
        auto a_k = a;
        auto b_k = b;
        if (check_k_blocks_order(a_k, b_k, bmnk_mapper, true))
            return plan_status_t::success;
        if (cfg_.fma_kind() != fma_kind_t::mad)
            return plan_status_t::ab_layout_mismatch;

        if (a_k.nblocks() == 2 && b_k.nblocks() == 2) {
            auto &a0 = a_k.blocks()[0];
            auto &a1 = a_k.blocks()[1];
            auto &b0 = b_k.blocks()[0];
            auto &b1 = b_k.blocks()[1];
            bool dims_ok
                    = (a0.dim_idx == b1.dim_idx) && (a1.dim_idx == b0.dim_idx);
            bool blocks_ok = (a0.block == b1.block) && (a1.block == b0.block);
            if (dims_ok && blocks_ok) {
                auto a_blocks = a.blocks();
                int i0 = -1;
                int i1 = -1;
                for (int i = 0; i < a.nblocks(); i++) {
                    if (bmnk_mapper.bmnk_kind(
                                abc_kind_t::a, a_blocks[i].dim_idx)
                            == bmnk_kind_t::k) {
                        if (i0 == -1) {
                            i0 = i;
                            continue;
                        }
                        if (i1 == -1) {
                            i1 = i;
                            continue;
                        }
                    }
                }
                std::swap(a_blocks[i0], a_blocks[i1]);
                a = layout_t(a.type(), a.ndims(), a.offset(), a_blocks);
                return plan_status_t::success;
            }
        }

        return plan_status_t::ab_layout_mismatch;
    }

    plan_status_t init_x2r_plan(const slm_plan_t &slm, x2r_plan_t &plan) const {
        PLAN_CHECK(init_x_s2r_plan(abc_kind_t::a, slm.has_a(), slm.a_layout,
                gemm_schedule_.a_thr_tile(), plan.a_load, plan.a_layout));
        PLAN_CHECK(init_x_s2r_plan(abc_kind_t::b, slm.has_b(), slm.b_layout,
                gemm_schedule_.b_thr_tile(), plan.b_load, plan.b_layout));
        PLAN_CHECK(init_x_g2r_plan(abc_kind_t::a, slm.has_a(),
                gemm_schedule_.a_tg_view(), gemm_schedule_.a_thr_tile(),
                gemm_schedule_.a_thr_tile(/*is_relative=*/false), plan.a_load,
                plan.a_reorder, plan.a_layout, reduce_mask_t(), nullptr,
                nullptr, slm.has_b() ? plan.b_layout : layout_t()));
        PLAN_CHECK(init_x_g2r_plan(abc_kind_t::b, slm.has_b(),
                gemm_schedule_.b_tg_view(), gemm_schedule_.b_thr_tile(),
                gemm_schedule_.b_thr_tile(/*is_relative=*/false), plan.b_load,
                plan.b_reorder, plan.b_layout, reduce_mask(cfg_, abc_kind_t::b),
                &plan.b_reduce, &plan.b_reduce_tile,
                (slm.has_a() && !slm.has_b()) || !(slm.has_a() && slm.has_b())
                        ? plan.a_layout
                        : layout_t()));
        PLAN_CHECK(verify_2d());
        PLAN_CHECK(fixup_k_blocks_order(plan.a_layout, plan.b_layout));
        return plan_status_t::success;
    }

    plan_status_t init_fma_plan(const x2r_plan_t &x2r, fma_plan_t &plan) const {
        auto &mapper = gemm_schedule_.bmnk_mapper();
        auto a_layout = mapper.map_to_bmnk(abc_kind_t::a,
                {bmnk_kind_t::b, bmnk_kind_t::m, bmnk_kind_t::k}, x2r.a_layout);
        auto b_layout = mapper.map_to_bmnk(abc_kind_t::b,
                {bmnk_kind_t::b, bmnk_kind_t::k, bmnk_kind_t::n}, x2r.b_layout);
        auto fma_kind = cfg_.fma_kind();
        int simd = cfg_.simd();
        int vec_size = cfg_.vec_size();
        int b_blk = 1;
        int m_blk = 1;
        int n_blk = 1;
        int k_blk = 1;
        auto &a_type = a_layout.type();
        auto &b_type = b_layout.type();
        auto c_type = get_default_accumulation_type(a_type, b_type);
        if (fma_kind == fma_kind_t::mad && a_type.is_f16() && b_type.is_f16()) {
            // FIXME: f16 must use f32 accumulator according to documentation.
            c_type = type_t::f16();
        }
        layout_t c_blk_layout(c_type, 0, std::vector<dim_t>(3, 1));
        switch (fma_kind) {
            case fma_kind_t::dp4a:
            case fma_kind_t::dpas:
            case fma_kind_t::dpasw: {
                const int sdepth = 8;
                const int dword_size = 4;
                ir_assert(is_dpas_src1_compatible(
                        simd, /*transpose=*/true, b_layout));
                ir_assert(is_dpas_src2_compatible(
                        simd, /*transpose=*/true, a_layout));
                ir_assert(a_layout.type().size() == b_layout.type().size());
                int ab_type_size = a_layout.type().size();
                m_blk = get_dpas_block_rcount(a_layout, 1);
                n_blk = simd;
                k_blk = sdepth * dword_size / ab_type_size;
                c_blk_layout = c_blk_layout.add_outer_block(2, n_blk);
                c_blk_layout = c_blk_layout.add_outer_block(1, m_blk);
                break;
            }
            case fma_kind_t::mad:
                if (is_mad_compatible(vec_size, a_layout, b_layout, 0, 0)) {
                    b_blk = vec_size;
                    c_blk_layout = c_blk_layout.add_outer_block(0, vec_size);
                } else if (is_mad_compatible(
                                   vec_size, a_layout, b_layout, -1, 2)) {
                    n_blk = vec_size;
                    c_blk_layout = c_blk_layout.add_outer_block(2, vec_size);
                } else if (is_mad_compatible(
                                   vec_size, a_layout, b_layout, 1, -1)) {
                    m_blk = vec_size;
                    c_blk_layout = c_blk_layout.add_outer_block(1, vec_size);
                } else if (cfg_.prb().g > 1
                        && (a_layout.blocks()[0].block
                                   * a_layout.blocks()[1].block)
                                        % vec_size
                                == 0) {
                } else {
                    return plan_status_t::invalid_layout;
                }
                break;
            default: ir_error_not_expected();
        }

        auto c_layout = get_c_layout(a_layout, b_layout, c_blk_layout);
        bmnk_block_mapper_t c_mapper(mapper);
        c_mapper.push_blocks(abc_kind_t::a, x2r.a_layout.blocks());
        c_mapper.push_blocks(abc_kind_t::b, x2r.b_layout.blocks());
        auto c_prb_layout = c_mapper.map_from_bmnk(abc_kind_t::c,
                {bmnk_kind_t::b, bmnk_kind_t::m, bmnk_kind_t::n}, c_layout);

        plan.a_layout = a_layout;
        plan.b_layout = b_layout;
        plan.c_layout = c_layout;
        plan.c_prb_layout = c_prb_layout;
        plan.fma_kind = fma_kind;
        plan.b_blk = b_blk;
        plan.m_blk = m_blk;
        plan.n_blk = n_blk;
        plan.k_blk = k_blk;
        return plan_status_t::success;
    }

    plan_status_t init_zp_plan(const x2r_plan_t &x2r, const fma_plan_t &fma,
            zp_plan_t &plan) const {
        auto &prb = cfg_.prb();
        auto &zp_cfg = prb.zp_cfg;
        if (!zp_cfg.do_src_compensation) return plan_status_t::success;

        auto b_tile = gemm_schedule_.b_thr_tile(/*is_relative=*/false);

        int g_idx = 0;
        int ic_idx = 2;
        expr_t zp_off;
        dim_t zp_g_dim, zp_ic_dim;
        if (zp_cfg.is_common_src_zero_point) {
            zp_off = expr_t(0);
            zp_g_dim = 1;
            zp_ic_dim = 1;
        } else {
            auto &w_g = b_tile.start()[g_idx];
            auto &w_ic = b_tile.start()[ic_idx];
            zp_off = w_g * prb.ic + w_ic;
            zp_g_dim = b_tile(g_idx);
            zp_ic_dim = b_tile(ic_idx);
        }

        layout_t zp_layout(type_t::s32(), zp_off,
                std::vector<dim_t> {zp_g_dim, zp_ic_dim});
        view_t zp_view(zp_layout);

        plan.init(cfg_, gemm_schedule_, zp_view, x2r.a_layout, x2r.b_layout,
                fma.c_prb_layout);
        return plan_status_t::success;
    }

    std::shared_ptr<conv_plan_t> get_plan() const {
        auto plan_ptr = std::make_shared<conv_plan_t>(cfg_.hw());
        auto &plan = *plan_ptr;

        auto &init_cset = plan.init_cset;
        auto &gemm_schedule = plan.gemm_schedule;
        gemm_schedule = gemm_schedule_t(
                init_cset, cfg_.kernel_grid(), cfg_.thread_group_grid());
        view_t a_view;
        view_t b_view;
        view_t c_view;
        if (prb_.is_fwd) {
            init_fwd(cfg_, gemm_schedule, a_view, b_view, c_view);
        } else if (prb_.is_bwd_d) {
            init_bwd_d(cfg_, gemm_schedule, a_view, b_view, c_view);
        } else if (prb_.is_bwd_w) {
            init_bwd_w(
                    cfg_, gemm_schedule, a_view, b_view, c_view, plan.bia_view);
        } else {
            ir_error_not_expected();
        }
        gemm_schedule.finalize();

        if (cfg_.pipeline().is_overridden()) {
            plan.reuse_headers = cfg_.pipeline().reuse_headers();
        }

        return plan_ptr;
    }

    send_params_t get_send_params(const exec_config_t &exec_cfg, send_op_t op,
            send_address_t address, abc_kind_t abc, const view_t &view) const {
        auto params = jit::get_send_params(exec_cfg, op, address, view);
        bool allow_send_2d
                = (abc == abc_kind_t::a ? allow_a_send_2d_ : allow_b_send_2d_);
        if (!allow_send_2d) params.hint_2d.enable = false;
        return params;
    }

    send_params_t get_send_params(const exec_config_t &exec_cfg, send_op_t op,
            send_address_t address, fma_kind_t fma, abc_kind_t abc,
            const view_t &view, const gemm_schedule_t &gemm_schedule,
            bool allow_2d_load = true) const {
        auto params = jit::get_send_params(exec_cfg, op, address, fma, abc,
                view, gemm_schedule, allow_2d_load);
        bool allow_send_2d
                = (abc == abc_kind_t::a ? allow_a_send_2d_ : allow_b_send_2d_);
        if (!allow_send_2d) params.hint_2d.enable = false;
        return params;
    }

    void enable_direct_view(bool value) { allow_direct_view_ = value; }
    void enable_send_2d(abc_kind_t abc, bool value) {
        auto &allow_send_2d
                = (abc == abc_kind_t::a ? allow_a_send_2d_ : allow_b_send_2d_);
        allow_send_2d = value;
    }
    void enable_slm(bool value) { allow_slm_ = value; }

    conv_config_t &cfg_;
    const conv_problem_t &prb_;
    std::shared_ptr<conv_plan_t> plan_ptr_;
    conv_plan_t &plan_;
    gemm_schedule_t &gemm_schedule_;
    direct_view_t a_direct_view_;
    direct_view_t b_direct_view_;
    bool allow_direct_view_ = true;
    bool allow_a_send_2d_ = true;
    bool allow_b_send_2d_ = true;
    bool allow_slm_ = true;
};

status_t init_plan(conv_config_t &cfg) {
    plan_builder_t plan_builder(cfg);
    return plan_builder.init_plan();
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
