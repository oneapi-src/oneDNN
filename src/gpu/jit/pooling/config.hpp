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
            const exec_config_t &exec, const layout_t &src) {
        // only 8 or 16 threads in a threadgroup are supported
        const int max_tg = get_max_tg(exec);
        if ((max_tg != 8) && (max_tg != 16)) return false;

        // only allow SIMD-aligned channel-first layouts
        const auto &oc_blk = src.blocks()[0];
        if ((oc_blk.dim_idx != 1) || (oc_blk.block % exec.simd())) return false;

        // for now, prohibit Global ({o|k}dhw = 1) and Dense (odhw = idhw)
        // TODO: enable both
        if (((prb.od == 1) || (prb.kd == 1)) && ((prb.oh == 1) || (prb.kh == 1))
                && ((prb.ow == 1) || (prb.kw == 1)))
            return false;
        if ((prb.od == prb.id) && (prb.oh == prb.ih) && (prb.ow == prb.iw))
            return false;

        return true; // no more restrictions, the configuration is compatible
    }

    pooling_config_t() = default;
    pooling_config_t(const exec_config_t &ec, const pool_conf_t &prb,
            const layout_t &src, const layout_t &dst) {
        set_pooling_problem(prb);
        src_layout().set_user(spatials_to_3d(src, false, 0));
        dst_layout().set_user(spatials_to_3d(dst, false, 0));
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

    bool is_blocked_by_mb() const {
        const auto &blk = src_layout().user().blocks();
        return (blk.size() > 1) && (blk[1].dim_idx == 0);
    }

    void compute_grid() {
        const auto &prb = pooling_problem();
        const auto &src = src_layout().user();
        const auto &exec = exec_cfg();
        const int simd = exec.simd();
        const int eu_count = exec.hw().eu_count();
        const int max_tg = get_max_tg(exec);

        std::vector<int> padded {
                int(src.dim(0)), int(src.dim(1)), prb.od, prb.oh, prb.ow};
        auto &mb = padded[0], &oc = padded[1];
        auto &od = padded[2], &oh = padded[3], &ow = padded[4];

        //                  mb oc od oh ow kd kh kw
        //                  [0  1][2  3  4][5  6  7]
        std::vector<int> lg {1, 1, 1, 1, 1, 1, 1, 1};
        std::vector<int> tg {1, 1, 1}, kg {1, 1, 1};

        const auto ohw = ow * oh;
        if ((max_tg <= ohw * od) || (ohw == 3 * 3) || (ohw == 3 * 5)) {
            auto loss = [&](int tgw) {
                return utils::rnd_up(ow, tgw) * utils::rnd_up(oh, max_tg / tgw);
            };
            int ok_tgw = sqrt(max_tg);
            ir_assert(ok_tgw == utils::rnd_up_pow2(ok_tgw));
            for (int tgw = sqrt(max_tg); tgw > 0; tgw >>= 1) {
                if (loss(tgw) < loss(ok_tgw)) ok_tgw = tgw;
                if (loss(max_tg / tgw) <= loss(ok_tgw)) ok_tgw = max_tg / tgw;
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

        const dim_t oc_outer = oc / simd;
        const dim_t oc_blk = src.blocks()[0].block;
        const dim_t mb_blk = (is_blocked_by_mb()) ? src.blocks()[1].block : mb;

        // lg[2], lg[3], lg[4] are to be set here

        kg[0] = utils::div_up(od, tg[0] * lg[2]);
        kg[1] = utils::div_up(oh, tg[1] * lg[3]);
        kg[2] = utils::div_up(ow, tg[2] * lg[4]);

        if (ow % (tg[2] * lg[4]) == 0) {
            kg[2] *= kg[1];
            kg[1] = 1;
        }

        const dim_t safe_thr_count = eu_count * 7;
        const dim_t max_thr_work = utils::div_up(utils::div_up(oc, simd) * mb
                        * tg[0] * tg[1] * tg[2] * kg[0] * kg[1] * kg[2],
                safe_thr_count);

        // the constant being subtracted is a heuristic
        const int regs_per_tile
                = exec.regs() - ((prb.kd * prb.kh * prb.kw > 1) ? 32 : 0);

        if (is_blocked_by_mb()) {
            lg[1] = utils::max_div(oc_blk / simd, max_thr_work);
            lg[0] = utils::max_div(mb_blk, max_thr_work / lg[1]);

            const float min_used_mb_share = 0.875f; // heuristic!
            if (prb.mb < src.dim(0) * min_used_mb_share)
                lg[0] = math::gcd(lg[0], prb.mb);
        }

        if (regs_per_tile / (lg[0] * lg[1]) <= prb.kw) {
            lg[7] = utils::max_div(prb.kw, regs_per_tile / (lg[0] * lg[1]));
        } else if (regs_per_tile / (lg[0] * lg[1]) <= prb.kw * prb.kh) {
            lg[7] = prb.kw;
            lg[6] = utils::max_div(
                    prb.kh, regs_per_tile / (lg[0] * lg[1] * prb.kw));
        } else if (regs_per_tile / (lg[0] * lg[1])
                <= prb.kw * prb.kh * prb.kd) {
            lg[7] = prb.kw;
            lg[6] = prb.kh;
            lg[5] = utils::max_div(
                    prb.kd, regs_per_tile / (lg[0] * lg[1] * prb.kw * prb.kh));
        } else {
            lg[7] = prb.kw;
            lg[6] = prb.kh;
            lg[5] = prb.kd;
        }

        if (!is_blocked_by_mb()) {
            const int layers_per_thr = regs_per_tile / (lg[7] * lg[6] * lg[5]);
            if (max_thr_work > 1) {
                lg[1] = std::min(max_thr_work,
                        utils::max_div(oc_blk / simd, layers_per_thr));
                lg[1] = utils::max_div(oc_outer, std::max(lg[1], 1));
            }
            if ((oc == lg[1] * simd) && (max_thr_work / lg[1] > 1)) {
                const int mb_reg = layers_per_thr / lg[1] / src.type().size();
                lg[0] = std::min(max_thr_work / lg[1],
                        (mb_reg > mb_blk) ? utils::rnd_dn(mb_reg, mb_blk)
                                          : utils::max_div(mb_blk, mb_reg));
                lg[0] = utils::max_div(mb, std::max(lg[0], 1));
            }
            if ((lg[0] == 1) && (max_thr_work / lg[1] > 1)) {
                const int oc_reg = layers_per_thr / lg[1] / src.type().size();
                const int lg1 = std::min(max_thr_work / lg[1],
                        utils::max_div(oc_outer / lg[1], oc_reg));
                lg[1] *= utils::max_div(oc_outer / lg[1], std::max(lg1, 1));
            }
        }

        lg[1] *= simd;
        kg[0] *= utils::div_up(oc, lg[1]);
        kg[1] *= utils::div_up(mb, lg[0]);

        set_dims_padded(grid_info_t(padded, ""));
        set_loop_grid(grid_info_t(lg, "lg_idx"));
        set_kernel_grid(grid_info_t(kg, "kg_idx"));
        set_thread_group_grid(grid_info_t(tg, "tg_idx"));
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
    DECL_PARAM(kernel_grid);
    DECL_PARAM(thread_group_grid);
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
    INIT_PARAM(kernel_grid);
    INIT_PARAM(thread_group_grid);
    INIT_PARAM(loop_grid);
    INIT_PARAM(dims_padded);
#undef INIT_PARAM

private:
    static int get_max_tg(const exec_config_t &exec) {
        return compute::device_info_t::max_eus_per_wg(
                convert_ngen_arch_to_dnnl(exec.hw().to_ngen()));
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
