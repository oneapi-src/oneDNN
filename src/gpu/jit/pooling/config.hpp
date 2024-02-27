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

#ifndef GPU_JIT_POOLING_CONFIG_HPP
#define GPU_JIT_POOLING_CONFIG_HPP

#include <iostream>
#include <sstream>

#include "common/utils.hpp"
#include "gpu/jit/ir/config.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class pooling_problem_param_t : public value_param_t<pool_conf_t> {
public:
    using value_param_t::value_param_t;

    std::string name() const override { return "prb"; }
    std::string desc() const override { return "Pooling problem."; }
    bool is_overridable() const override { return false; }
};

class loop_grid_param_t : public grid_param_t {
public:
    std::string name() const override { return "loop_grid"; }
    std::string desc() const override { return "Loop grid."; }
    bool is_overridable() const override { return false; }
};

// padded_dims_param_t vomits pointer errors (!!) for no apparent reason,
// so dims_padded_param_t it shall be.
class dims_padded_param_t : public grid_param_t {
public:
    std::string name() const override { return "pad"; }
    std::string desc() const override {
        return "Padded dimensions (rounded-up for blocks and to comply with "
               "required zero padding in output layouts).";
    }
    bool is_overridable() const override { return false; }
};

// Parameters for kernel generation.
class pooling_config_t : public prim_config_t {
public:
    static bool check_compatibility(const pool_conf_t &prb,
            const exec_config_t &exec, const layout_t &src,
            const post_ops_t &po, type_t dst_dt) {
        const int max_tg = exec.hw().max_tg_size(exec.regs(), exec.simd());
        if (max_tg % 8 != 0) return false;

        // only allow SIMD-aligned channel-first layouts
        const auto &oc_blk = src.blocks()[0];
        if ((oc_blk.dim_idx != 1) || (oc_blk.block % exec.simd())) return false;

        // for some reason 3D pooling works poorly on PVC at the moment
        // TODO: bring PVC 3D pooling back
        if ((prb.kd > 1) && (exec.hw() >= ngen::HW::XeHPC)) return false;

        // for now, prohibit Global
        // TODO: enable asap
        if ((prb.kd * prb.kh * prb.kw > 1) && ((prb.kd == 1) || (prb.od == 1))
                && ((prb.kh == 1) || (prb.oh == 1))
                && ((prb.kw == 1) || (prb.ow == 1)))
            return false;

        // this one is trickier; as there are no masks on OC for performance
        // reasons, padded OC may end up containing garbage, so there's some
        // protection against that
        bool has_additive_po = false;
        float total_added = 0;
        for (int i = 0; i < po.len(); i++) {
            if ((po.entry_[i].is_binary()
                        && ((total_added != 0)
                                || (po.entry_[i].binary.alg == dnnl_binary_add)
                                || (po.entry_[i].binary.alg
                                        == dnnl_binary_sub)))
                    || po.entry_[i].is_sum(false, false)) {
                has_additive_po = true;
                break;
            } else if (po.entry_[i].is_eltwise(false)
                    && (po.entry_[i].eltwise.alg == dnnl_eltwise_linear)) {
                if (total_added != 0) {
                    has_additive_po = true;
                    break;
                }
                total_added += po.entry_[i].eltwise.beta;
            }
        }
        has_additive_po |= (!dst_dt.is_int() && (total_added != 0))
                || (dst_dt.is_int() && (fabsf(total_added) >= 1));
        if ((prb.c % oc_blk.block)
                && (has_additive_po || (prb.id / prb.stride_d < prb.od)
                        || (prb.ih / prb.stride_h < prb.oh)
                        || (prb.iw / prb.stride_w < prb.ow)))
            return false;

        return true; // no more restrictions, the configuration is compatible
    }

    pooling_config_t() = default;
    pooling_config_t(const exec_config_t &ec, const pool_conf_t &prb,
            const layout_t &src, const layout_t &dst) {
        set_pooling_problem(prb);
        src_layout().set_user(spatials_to_3d(src, false, {0, 1, 2}));
        dst_layout().set_user(spatials_to_3d(dst, false, {0, 1, 2}));
        set_exec_cfg(ec);
    }

    prb_tile_t shape(bool pad) const override {
#define SET(g_name, l_name) \
    ret[prb_dims::g_name] = (pad) \
            ? utils::rnd_up(prb.l_name, pad_block(prb_dims::g_name)) \
            : prb.l_name

        const auto &prb = pooling_problem();
        prb_tile_t ret;
        SET(mb, mb);
        SET(oc, c);
        if (is_fwd()) {
            SET(od, od);
            SET(oh, oh);
            SET(ow, ow);
        } else {
            SET(id, id);
            SET(ih, ih);
            SET(iw, iw);
        }
        SET(kd, kd);
        SET(kh, kh);
        SET(kw, kw);
        return ret;

#undef SET
    }

    const std::vector<prb_dim_t> &index_dims() const override {
        auto get_dims = [&](bool is_fwd) {
            std::vector<prb_dim_t> ret;
            ret.push_back(prb_dims::mb);
            ret.push_back(prb_dims::oc);
            if (is_fwd) {
                ret.push_back(prb_dims::od);
                ret.push_back(prb_dims::oh);
                ret.push_back(prb_dims::ow);
            } else {
                ret.push_back(prb_dims::id);
                ret.push_back(prb_dims::ih);
                ret.push_back(prb_dims::iw);
            }
            ret.push_back(prb_dims::kd);
            ret.push_back(prb_dims::kh);
            ret.push_back(prb_dims::kw);
            return ret;
        };
        static std::vector<prb_dim_t> fwd_dims = get_dims(true);
        static std::vector<prb_dim_t> bwd_dims = get_dims(false);
        return (is_fwd()) ? fwd_dims : bwd_dims;
    }

    int pad_block(const prb_dim_t &d) const override {
        switch (d.kind()) {
            default: return 1;
            case prb_dim_kind_t::mb:
                return src_layout().user().inner_block(0, true, false);
            case prb_dim_kind_t::oc:
                return src_layout().user().inner_block(1, true, false);
        }
    }

    bool is_fwd() const { return !pooling_problem().is_backward; }
    bool is_max() const {
        return pooling_problem().alg == alg_kind_t::dnnl_pooling_max;
    }
    bool is_padded() const {
        return pooling_problem().alg
                == alg_kind_t::dnnl_pooling_avg_include_padding;
    }

    bool is_blocked_by_mb() const {
        const auto &blk = src_layout().user().blocks();
        return (blk.size() > 1) && (blk[1].dim_idx == 0);
    }

    type_t acc_type(int len) const {
        const auto read_type = src_layout().user().type();
        switch (0x10 * read_type.is_int() + is_max()) {
            default:
            case 0x00: return type_t::f32(len); break;
            case 0x01: return type_t(read_type.kind(), len); break;
            case 0x10: return type_t::s32(len); break;
            case 0x11:
                return ((read_type.is_signed()) ? type_t::s : type_t::u)(
                        8 * std::max(2, read_type.size()), len);
        }
    }

    void compute_grid() {
        const auto &prb = pooling_problem();
        const auto &src = src_layout().user();
        const auto &exec = exec_cfg();
        const int simd = exec.simd();
        const int eu_count = exec.hw().eu_count();

        //                  mb oc od oh ow kd kh kw
        //                  [0  1][2  3  4][5  6  7]
        std::vector<int> lg {1, 1, 1, 1, 1, 1, 1, 1};
        std::vector<int> tg {1, 1, 1}, kg {1, 1, 1};

        std::vector<int> padded {
                int(src.dim(0)), int(src.dim(1)), prb.od, prb.oh, prb.ow};
        auto &mb = padded[0], &oc = padded[1];
        auto &od = padded[2], &oh = padded[3], &ow = padded[4];

        const bool is_scalar = (prb.kd * prb.kh * prb.kw == 1);

        const int src_type_size = src.type().size();
        const int acc_type_size = acc_type(1).size();
        const int oc_blk = src.blocks()[0].block;
        const int mb_blk = (is_blocked_by_mb()) ? src.blocks()[1].block : mb;
        // the constant being subtracted is heuristic
        const int regs_per_tile
                = exec.regs() - (!is_scalar ? is_blocked_by_mb() ? 8 : 28 : 0);

        auto optimize_load = [](int &dim, int mult) {
            const int optimal_load_size = 256;
            int null = 0;
            while ((dim * mult > optimal_load_size) && (dim > 1))
                cut_dim(dim, null, 1);
        };

        if (!is_scalar && (prb.kh * prb.kw <= 9)) {
            // SMALL FILTERS

            const int max_tg = exec.hw().max_tg_size(exec.regs(), exec.simd());
            ir_assert(max_tg == utils::rnd_up_pow2(max_tg));

            const bool ow_pow2
                    = (ow > 1) && (utils::rnd_up_pow2(oh) * ow > max_tg);
            if (ow_pow2)
                ow = (ow > max_tg) ? utils::rnd_up(ow, max_tg)
                                   : utils::rnd_up_pow2(ow);
            else
                oh = (oh > max_tg) ? utils::rnd_up(oh, max_tg)
                                   : utils::rnd_up_pow2(oh);

            tg[2] = std::min(max_tg, ow);
            tg[1] = (ow_pow2) ? 1 : utils::max_div(oh, max_tg / tg[2]);

            // lg[2], lg[3], lg[4] are to be set here

            od = utils::rnd_up(od, tg[0] * lg[2]);
            oh = utils::rnd_up(oh, tg[1] * lg[3]);
            ow = utils::rnd_up(ow, tg[2] * lg[4]);

            kg[0] = od / (tg[0] * lg[2]);
            kg[1] = (oh / (tg[1] * lg[3])) * (ow / (tg[2] * lg[4]));
            kg[2] = 1;

            if (ow_pow2 && (mb >= 512)) { // lower TGs preferable at higher MBs
                const int low_tg
                        = std::max(1, max_tg / (2 * utils::div_up(mb, 512)));
                if (tg[2] / low_tg > 1) {
                    kg[is_blocked_by_mb() ? 2 : 1] *= tg[2] / low_tg;
                    tg[2] = low_tg;
                }
            }
            const int optimal_oc = std::max(2, 4 / src_type_size); // heuristic
            const int max_grf
                    = exec.grf_size() * exec.regs() * 3 / 4; // heuristic
            // (src + acc) can be 1+2, 1+4, 2+2, 2+4, 4+4 bytes
            const int simds_per_line
                    = max_grf / (simd * (src_type_size + acc_type_size));

            auto calc_non_sp = [](int scale, int simds, int opt, int per_line) {
                int pow2 = 1;
                for (int i = simds; i % 2 == 0; i /= 2)
                    pow2 *= 2;
                return scale
                        * utils::max_div(
                                (pow2 < opt) ? simds : pow2, per_line / scale);
            };
            if (is_blocked_by_mb()) {
                lg[1] = oc_blk / simd;
                lg[0] = mb_blk;
                int null = 0;
                while (lg[1] * lg[0] > simds_per_line) {
                    if (lg[0] > 1)
                        cut_dim(lg[0], null, 1);
                    else
                        cut_dim(lg[1], null, 1);
                }
            } else {
                if (oc == oc_blk) {
                    lg[1] = calc_non_sp(
                            1, oc / simd, optimal_oc, simds_per_line);
                } else if (oc_blk / simd <= simds_per_line) {
                    lg[1] = calc_non_sp(oc_blk / simd, oc / oc_blk, optimal_oc,
                            simds_per_line);
                    if (lg[1] > 32) // HBM reader can break on very large OC
                        lg[1] = utils::rnd_up(lg[1], 4); // heuristic
                } else {
                    lg[1] = utils::max_div(oc_blk / simd, simds_per_line);
                }
                if ((lg[1] < optimal_oc)
                        && (lg[1] == utils::rnd_up_pow2(lg[1]))) {
                    const int oc_simds_per_line = simds_per_line / lg[1];
                    lg[0] = (mb <= oc_simds_per_line)
                            ? mb
                            : utils::max_div(mb, oc_simds_per_line);
                }
            }
            lg[0] = calc_non_sp(1, prb.mb, 1, lg[0]);

            const dim_t total_simds = dim_t(mb) * (oc / simd) * od * oh * ow;
            const dim_t safe_thr_count = eu_count * 4;

            if (total_simds < safe_thr_count * lg[1] * lg[0]) {
                auto find_div = [](int num, int total_simds, int thr_count) {
                    if (total_simds <= thr_count) return 1;
                    const int orig = num;
                    num = 0;
                    for (int div = sqrtf(orig); div >= 1; div--)
                        if (orig % div == 0) {
                            if (total_simds >= thr_count * (orig / div))
                                num = std::max(num, orig / div);
                            if (total_simds >= thr_count * div)
                                num = std::max(num, div);
                        }
                    return (num == 0) ? orig : num;
                };
                if (total_simds < safe_thr_count * lg[1]) { // cut [0] and [1]
                    // NOT heuristic; odd x8 SIMDs on HBM reads break the reader
                    const int mult = std::max(1, 2 / src_type_size);
                    if (lg[1] % mult == 0) {
                        const auto old_lg1 = lg[1];
                        lg[0] = 1;
                        lg[1] = find_div(lg[1] / mult, total_simds / mult,
                                        safe_thr_count)
                                * mult;
                        if ((lg[1] > 1) && (lg[1] % 2)) lg[1] = old_lg1 / lg[1];
                    }
                } else { // only cut [0]
                    lg[0] = find_div(lg[0], total_simds, safe_thr_count);
                }
            }
            const int loop_space = simds_per_line / (lg[0] * lg[1])
                    * (src_type_size + acc_type_size) / src_type_size;
            lg[7] = prb.kw;
            lg[6] = std::max(utils::max_div(prb.kh, loop_space / lg[7]), 1);
            lg[5] = std::max(
                    utils::max_div(prb.kd, loop_space / (lg[7] * lg[6])), 1);
        } else {
            // REGULAR FILTERS

            const int max_tg = utils::max_div(
                    exec.hw().max_tg_size(exec.regs(), exec.simd()), 16);

            if (ow >= utils::rnd_up(ow, max_tg) * 7.f / 8.f)
                ow = utils::rnd_up(ow, max_tg);

            const auto ohw = ow * oh;
            if ((max_tg <= ohw * od) || (ohw == 3 * 3) || (ohw == 3 * 5)) {
                auto loss = [&](int tgw) {
                    return utils::rnd_up(ow, tgw)
                            * utils::rnd_up(oh, max_tg / tgw);
                };
                int ok_tgw = sqrt(max_tg);
                ir_assert(ok_tgw == utils::rnd_up_pow2(ok_tgw));
                for (int tgw = sqrt(max_tg); tgw > 0; tgw >>= 1) {
                    if (loss(tgw) < loss(ok_tgw)) ok_tgw = tgw;
                    if (loss(max_tg / tgw) <= loss(ok_tgw))
                        ok_tgw = max_tg / tgw;
                }
                tg[2] = utils::div_up(ow, utils::div_up(ow, ok_tgw));
                tg[1] = utils::div_up(oh, utils::div_up(oh, max_tg / ok_tgw));
            } else {
                tg[2] = ow;
                tg[1] = oh;
                tg[0] = od;
            }

            if ((tg[1] > 1) && (tg[2] > 1) && (tg[1] * tg[2] % 2))
                tg[1] += (tg[1] * tg[2] > 3 * 3) ? -1 : 1;

            // lg[2], lg[3], lg[4] are to be set here

            od = utils::rnd_up(od, tg[0] * lg[2]);
            oh = utils::rnd_up(oh, tg[1] * lg[3]);
            ow = utils::rnd_up(ow, tg[2] * lg[4]);

            kg[0] = od / (tg[0] * lg[2]);
            kg[1] = oh / (tg[1] * lg[3]);
            kg[2] = ow / (tg[2] * lg[4]);

            if (prb.ow % (tg[2] * lg[4]) == 0) {
                kg[2] *= kg[1];
                kg[1] = 1;
            }

            const int safe_thr_count = eu_count * 7;
            const int max_threads
                    = utils::div_up(dim_t(utils::div_up(oc, simd)) * mb * tg[0]
                                    * tg[1] * tg[2] * kg[0] * kg[1] * kg[2],
                            safe_thr_count);

            if (is_blocked_by_mb()) {
                lg[1] = utils::max_div(oc_blk / simd, max_threads);
                lg[0] = utils::max_div(mb_blk, max_threads / lg[1]);
                if (!is_scalar) {
                    optimize_load(lg[0], lg[1] * simd * src_type_size);
                    optimize_load(lg[1], simd * src_type_size);
                }
            }

            const int simds_per_tile = (regs_per_tile * 32 / simd
                                               - lg[0] * lg[1] * acc_type_size)
                    / src_type_size;

            if (simds_per_tile / (lg[0] * lg[1]) <= prb.kw) {
                lg[7] = utils::max_div(
                        prb.kw, simds_per_tile / (lg[0] * lg[1]));
            } else if (simds_per_tile / (lg[0] * lg[1]) <= prb.kw * prb.kh) {
                lg[7] = prb.kw;
                lg[6] = utils::max_div(
                        prb.kh, simds_per_tile / (lg[0] * lg[1] * prb.kw));
            } else if (simds_per_tile / (lg[0] * lg[1])
                    <= prb.kw * prb.kh * prb.kd) {
                lg[7] = prb.kw;
                lg[6] = prb.kh;
                lg[5] = utils::max_div(prb.kd,
                        simds_per_tile / (lg[0] * lg[1] * prb.kw * prb.kh));
            } else {
                lg[7] = prb.kw;
                lg[6] = prb.kh;
                lg[5] = prb.kd;
            }

            if (!is_blocked_by_mb()) {
                const int oc_outer = oc / simd;
                const int layers_per_thr
                        = simds_per_tile / (lg[7] * lg[6] * lg[5]);
                if (max_threads > 1) {
                    lg[1] = std::min(max_threads,
                            utils::max_div(oc_blk / simd, layers_per_thr));
                    lg[1] = utils::max_div(oc_outer, std::max(lg[1], 1));
                }
                if ((oc == lg[1] * simd) && (max_threads / lg[1] > 1)) {
                    const int mb_reg = layers_per_thr / lg[1] / src_type_size;
                    lg[0] = std::min(max_threads / lg[1],
                            (mb_reg > mb_blk) ? utils::rnd_dn(mb_reg, mb_blk)
                                              : utils::max_div(mb_blk, mb_reg));
                    lg[0] = utils::max_div(mb, std::max(lg[0], 1));
                }
                if ((lg[0] == 1) && (max_threads / lg[1] > 1)) {
                    const int oc_reg = layers_per_thr / lg[1] / src_type_size;
                    const int lg1 = std::min(max_threads / lg[1],
                            utils::max_div(oc_outer / lg[1], oc_reg));
                    lg[1] *= utils::max_div(oc_outer / lg[1], std::max(lg1, 1));
                }
            }
        }
        lg[1] *= simd;
        oc = utils::rnd_up(oc, lg[1]);
        kg[0] *= utils::div_up(oc, lg[1]);
        kg[1] *= utils::div_up(mb, lg[0]);

        set_dims_padded(grid_info_t(padded, ""));
        set_loop_grid(grid_info_t(lg, "lg_idx"));
        set_kernel_grid(grid_info_t(kg, "kg_idx"));
        set_thread_group_grid(grid_info_t(tg, "tg_idx"));
    }

    compute::nd_range_t nd_range() const {
        const auto &kg = kernel_grid();
        const auto &tg = thread_group_grid();
        std::array<size_t, 3> local {size_t(tg[0] * exec_cfg().simd()),
                size_t(tg[1]), size_t(tg[2])};
        std::array<size_t, 3> global {size_t(kg[0]) * local[0],
                size_t(kg[1]) * local[1], size_t(kg[2]) * local[2]};

        return compute::nd_range_t(global.data(), local.data());
    }

    std::string str() const override {
        std::ostringstream oss;
        // clang-format off
        oss << "  Exec config:          " << exec_cfg() << std::endl;
        oss << "  Problem:              " << desc_str() << std::endl;
        const char *names[] = {"Source", "Destination"};
        const layout_param_t *layouts[] = {&src_layout(), &dst_layout()};
        for (int i = 0; i < 2; i++) {
            std::string desc = std::string(names[i]) + " layout:";
            desc.insert(desc.size(), 22 - desc.size(), ' ');
            oss << "  " << desc << layouts[i]->user() << std::endl;
        }
        const int kg_elems = kernel_grid().elems();
        const int tg_elems = thread_group_grid().elems();
        //oss << blocking_brief_str();
        oss << "  Padded dimensions:    " << dims_padded() << std::endl;
        oss << "  Internal loop:        " << loop_grid() << std::endl;
        oss << "  Thread group:         " << thread_group_grid() << std::endl;
        oss << "  Kernel grid:          " << kernel_grid() << std::endl;
        oss << "  Threads:              " << kg_elems * tg_elems
            << " (utilization: "
            << get_thread_utilization(exec_cfg(), kg_elems, tg_elems)
            << "% thread, "
            << get_wave_utilization(exec_cfg(), kg_elems, tg_elems)
            << "% wave)" << std::endl;
        oss << "  Configuration line:   " << get_config_line() << std::endl;
        // clang-format on
        return oss.str();
    }

    int n_cuts() const { return n_cuts_; }
    bool cut() {
        const auto simd = exec_cfg().simd();
        auto kg(kernel_grid());
        auto lg(loop_grid());
        int null = 0;

        if (lg[5] > 1)
            cut_dim(lg[5], null, 1); // kd
        else if (lg[6] > 1)
            cut_dim(lg[6], null, 1); // kh
        else if (lg[0] > 1)
            cut_dim(lg[0], kg[1], 1); // mb
        else if (lg[1] / simd > 1)
            cut_dim(lg[1], kg[0], 2 * simd); // oc
        else if (lg[7] > 1)
            cut_dim(lg[7], null, 1); // kw
        else
            return false;

        set_kernel_grid(kg);
        set_loop_grid(lg);
        n_cuts_++;
        return true;
    }

#define DECL_PARAM(name) \
    const name##_param_t &name##_param() const { \
        ir_assert(!name##_.is_undef()); \
        (void)name##_init_; \
        return name##_; \
    } \
    name##_param_t &name##_param() { return name##_; } \
    const name##_param_t::value_t &name() const { \
        ir_assert(!name##_.is_undef()); \
        return name##_.get(); \
    } \
    void set_##name(const name##_param_t::value_t &value) { \
        name##_.set(value); \
    }

    DECL_PARAM(pooling_problem);
    DECL_PARAM(loop_grid);
    DECL_PARAM(dims_padded);
#undef DECL_PARAM

#define INIT_PARAM(name) \
    name##_param_t name##_; \
    param_init_t name##_init_ \
            = register_param([](const container_config_t *c) { \
                  return &((const pooling_config_t *)c)->name##_; \
              });
    INIT_PARAM(pooling_problem);
    INIT_PARAM(loop_grid);
    INIT_PARAM(dims_padded);
#undef INIT_PARAM

private:
    int n_cuts_ = 0;

    static void cut_dim(int &dn, int &up, int scale) {
        // clang-format off
        static const std::array<unsigned char, 54> primes_up_to_256 = {
              2,   3,   5,   7,  11,  13,  17,  19,  23,  29,  31,
             37,  41,  43,  47,  53,  59,  61,  67,  71,  73,  79,
             83,  89,  97, 101, 103, 107, 109, 113, 127, 131, 137,
            139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
            197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
        };
        // clang-format on
        ir_assert(dn % scale == 0);
        for (int p : primes_up_to_256)
            if (dn % (p * scale) == 0) {
                up *= p;
                dn /= p;
                return;
            }
        up *= dn / scale;
        dn = scale;
    }

    std::string desc_str() const {
        const auto &prb = pooling_problem();
        const std::array<int, 6> xd
                = {prb.id, prb.od, prb.kd, prb.stride_d, prb.dd, prb.f_pad};
        const std::array<int, 6> xh
                = {prb.ih, prb.oh, prb.kh, prb.stride_h, prb.dh, prb.t_pad};
        const std::array<int, 6> xw
                = {prb.iw, prb.ow, prb.kw, prb.stride_w, prb.dw, prb.l_pad};
        const std::array<int, 6> xdef = {1, 1, 1, 1, 0, 0};
        const std::array<char, 6> name = {'i', 'o', 'k', 's', 'd', 'p'};

        const bool has_d = !ir_utils::is_equal(xd, xdef);
        const bool has_h = !ir_utils::is_equal(xh, xdef);
        const bool is_square = !has_d && ir_utils::is_equal(xh, xw);
        const bool is_cube
                = ir_utils::is_equal(xd, xh) && ir_utils::is_equal(xd, xw);
        const bool print_d = has_d;
        const bool print_h = has_h && !is_cube;
        const bool print_w = !is_cube && !is_square;

        std::ostringstream oss;
        oss << "mb" << prb.mb << "ic" << prb.c;
        for (int i = 0; i < int(name.size()); i++) {
            if (print_d && IMPLICATION(i == 3 || i == 4, xd[i] != xdef[i]))
                oss << name[i] << 'd' << xd[i];
            if (print_h && IMPLICATION(i == 3 || i == 4, xh[i] != xdef[i]))
                oss << name[i] << 'h' << xh[i];
            if (print_w && IMPLICATION(i == 3 || i == 4, xw[i] != xdef[i]))
                oss << name[i] << 'w' << xw[i];
        }
        return oss.str();
    }
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
