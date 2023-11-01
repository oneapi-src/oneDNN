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

class pool_conf_param_t : public value_param_t<pool_conf_t> {
public:
    using value_param_t::value_param_t;

    std::string name() const override { return "pool_conf"; }
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
    static bool check_compatibility(const pool_conf_t &conf,
            const exec_config_t &exec, const layout_t &src) {
        // only 8 or 16 threads in a threadgroup are supported
        const int max_tg = get_max_tg(exec);
        if ((max_tg != 8) && (max_tg != 16)) return false;

        // only allow SIMD-aligned channel-first layouts
        const auto &oc_blk = src.blocks()[0];
        if ((oc_blk.dim_idx != 1) || (oc_blk.block % exec.simd())) return false;

        // for now, prohibit Global (odhw = 1) and Dense (odhw = idhw)
        // TODO: enable both
        if (conf.od == 1 && conf.oh == 1 && conf.ow == 1) return false;
        if (conf.od == conf.id && conf.oh == conf.ih && conf.ow == conf.iw)
            return false;

        return true; // no more restrictions, the configuration is compatible
    }

    pooling_config_t() = default;
    pooling_config_t(const exec_config_t &ec, const pool_conf_t &pool_conf,
            const layout_t &src, const layout_t &dst) {
        set_pool_conf(pool_conf);
        src_layout().set_user(spatials_to_3d(src, false, 0));
        dst_layout().set_user(spatials_to_3d(dst, false, 0));
        set_exec_cfg(ec);
    }

    void compute_grid() {
        const auto &conf = pool_conf();
        const auto &src = src_layout().user();
        const auto &exec = exec_cfg();
        const int simd = exec.simd();
        const int eu_count = exec.hw().eu_count();
        const int max_tg = get_max_tg(exec);

        std::vector<int> padded {
                int(src.dim(0)), int(src.dim(1)), conf.od, conf.oh, conf.ow};
        auto &mb = padded[0], &oc = padded[1];
        auto &od = padded[2], &oh = padded[3], &ow = padded[4];

        //                  mb oc od oh ow kd kh kw
        //                  [0  1][2  3  4][5  6  7]
        std::vector<int> lg {1, 1, 1, 1, 1, 1, 1, 1};
        std::vector<int> tg {1, 1, 1}, kg {1, 1, 1};

        const float filter_aliasing = 0.875f; // heuristic coef (>1 is none)
        const float eu_mismatch = 1.150f; // heuristic coef (<1 is none)

        const bool is_tg_useful = filter_aliasing * conf.kd * conf.kh * conf.kw
                >= conf.stride_d * conf.stride_h * conf.stride_w;
        bool is_plain = (eu_count % max_tg != 0)
                || (utils::rnd_up(ow, eu_count / max_tg) >= eu_mismatch * ow);

        if (!is_plain) {
            ow = utils::rnd_up(conf.ow, eu_count / max_tg);
            tg[2] = utils::max_div(ow / (eu_count / max_tg), max_tg);
            tg[1] = max_tg / tg[2];
            if ((eu_count / (ow * oh) > 1)
                    || ((tg[2] == 1) && ((ow >= max_tg) || !is_tg_useful))) {
                tg[2] = tg[1] = 1;
                is_plain = true;
                ow = conf.ow;
            } else {
                oh = utils::rnd_up(conf.oh, tg[1]);
            }
        }
        if (is_plain && is_tg_useful) {
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
                    if (loss(max_tg / tgw) < loss(ok_tgw))
                        ok_tgw = max_tg / tgw;
                }
                tg[2] = utils::div_up(ow, utils::div_up(ow, ok_tgw));
                tg[1] = utils::div_up(oh, utils::div_up(oh, max_tg / ok_tgw));
            } else {
                tg[2] = ow;
                tg[1] = oh;
                tg[0] = od;
            }
        }
        if ((tg[1] > 1) && (tg[2] > 1) && (tg[1] * tg[2] % 2)) {
            tg[1] += (tg[1] * tg[2] > 3 * 3) ? -1 : 1;
            is_plain = true;
            ow = conf.ow;
            oh = conf.oh;
        }

        // the constant being subtracted is a heuristic
        const int regs_per_tile
                = exec.regs() - ((conf.kd * conf.kh * conf.kw > 1) ? 32 : 0);

        if (regs_per_tile <= conf.kw) {
            lg[7] = utils::max_div(conf.kw, regs_per_tile);
        } else if (regs_per_tile <= conf.kw * conf.kh) {
            lg[7] = conf.kw;
            lg[6] = utils::max_div(conf.kh, regs_per_tile / conf.kw);
        } else if (regs_per_tile <= conf.kw * conf.kh * conf.kd) {
            lg[7] = conf.kw;
            lg[6] = conf.kh;
            lg[5] = utils::max_div(
                    conf.kd, regs_per_tile / (conf.kw * conf.kh));
        } else {
            lg[7] = conf.kw;
            lg[6] = conf.kh;
            lg[5] = conf.kd;
        }

        kg[0] = utils::div_up(od, tg[0] * lg[2]);
        kg[1] = utils::div_up(oh, tg[1] * lg[3]);
        kg[2] = utils::div_up(ow, tg[2] * lg[4]);

        // 2 threads can be active on a single EU at any time, x2 to be sure
        const dim_t safe_thr_count = eu_count * 4;
        const dim_t max_thr_work = tg[0] * tg[1] * tg[2] * kg[0] * kg[1] * kg[2]
                * utils::div_up(oc, simd) * mb / safe_thr_count;
        const dim_t oc_outer = oc / simd;
        const int layers_per_thr = regs_per_tile / (lg[7] * lg[6] * lg[5]);
        const auto &oc_blk = src.blocks()[0];
        if (max_thr_work > 1) {
            lg[1] = std::min(max_thr_work,
                    utils::max_div(oc_blk.block / simd, layers_per_thr));
            lg[1] = utils::max_div(oc_outer, std::max(lg[1], 1));
        }
        const auto &mb_maybe = src.blocks()[1];
        if (((mb_maybe.dim_idx == 0) || (oc == lg[1] * simd))
                && (max_thr_work / lg[1] > 1)) {
            const int mb_blk = (mb_maybe.dim_idx == 0) ? mb_maybe.block : mb;
            const int mb_reg = layers_per_thr / lg[1] / src.type().size();
            lg[0] = std::min(max_thr_work / lg[1],
                    (mb_reg > mb_blk) ? utils::rnd_dn(mb_reg, dim_t(mb_blk))
                                      : utils::max_div(dim_t(mb_blk), mb_reg));
            lg[0] = utils::max_div(mb, std::max(lg[0], 1));
        }
        if ((lg[0] == 1) && (max_thr_work / lg[1] > 1)) {
            const int oc_reg = layers_per_thr / lg[1] / src.type().size();
            const int lg1 = std::min(max_thr_work / lg[1],
                    utils::max_div(oc_outer / lg[1], oc_reg));
            lg[1] *= utils::max_div(oc_outer / lg[1], std::max(lg1, 1));
        }
        lg[1] *= simd;
        kg[0] *= utils::div_up(oc, lg[1]);
        kg[1] *= utils::div_up(mb, lg[0]);

        if (!is_plain) {
            const int eu_sets_per_ow = kg[2] / (eu_count / max_tg);
            if (eu_sets_per_ow > 1) {
                kg[2] /= eu_sets_per_ow;
                kg[1] *= eu_sets_per_ow;
            } else if ((tg[2] > tg[1])
                    && (utils::rnd_up_pow2(tg[2]) * kg[2] == conf.ow)) {
                tg[2] *= tg[1];
                tg[1] = 1;
            }
        }

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

    DECL_PARAM(pool_conf);
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
    INIT_PARAM(pool_conf);
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
        const auto &conf = pool_conf();
        const std::array<int, 6> xd = {
                conf.id, conf.od, conf.kd, conf.stride_d, conf.dd, conf.f_pad};
        const std::array<int, 6> xh = {
                conf.ih, conf.oh, conf.kh, conf.stride_h, conf.dh, conf.t_pad};
        const std::array<int, 6> xw = {
                conf.iw, conf.ow, conf.kw, conf.stride_w, conf.dw, conf.l_pad};
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
        oss << "mb" << conf.mb << "ic" << conf.c;
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
