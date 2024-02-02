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

#include <random>

#include "oneapi/dnnl/dnnl.h"

#include "utils/fill.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "bnorm/bnorm.hpp"
#include "gnorm/gnorm.hpp"

using namespace bnorm;

namespace gnorm {

int fill_mean(const prb_t *prb, const cfg_t &cfg, dnn_mem_t &mem_fp,
        dnn_mem_t &mem_dt) {
    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        // Mean must be computed unless it is passed by user directly.
        if (!(prb->flags & GLOB_STATS) && !(prb->dir & FLAG_BWD)) return OK;

        return fill_random_real(mem_dt, mem_fp, nullptr);
    }

    benchdnn_parallel_nd(prb->mb, prb->g, [&](int64_t mb, int64_t g) {
        const int64_t idx = mb * prb->g + g;
        const float val_coeff = is_integral_dt(prb->dt[0]) ? 1.f : 0.25f;
        float val = 0.f;
        if (cfg.check_alg_ != ALG_0 || (prb->flags & GLOB_STATS)) {
            int64_t mean_val_shift = idx % 7;
            // Bump mean for u8 to keep src values in non-negative range
            if (prb->dt[0] == dnnl_u8 && mean_val_shift < 2) mean_val_shift = 2;
            val = val_coeff * (1LL << mean_val_shift);
        }
        mem_fp.set_elem(idx, val);
    });

    if (mem_dt && IMPLICATION(prb->dir & FLAG_FWD, prb->use_stats()))
        SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_src(const prb_t *prb, const cfg_t &cfg, dnn_mem_t &mem_fp,
        dnn_mem_t &mem_dt, const dnn_mem_t &ref_mean, res_t *res) {
    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        return fill_random_real(mem_dt, mem_fp, res);
    }

    const auto spatial = prb->id * prb->ih * prb->iw;
    const float val_coeff = is_integral_dt(prb->dt[0]) ? 1.f : 0.25f;

    benchdnn_parallel_nd(prb->mb, prb->g, [&](int64_t mb, int64_t g) {
        const int64_t idx = mb * prb->g + g;
        const float m = ref_mean.get_elem(idx);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand b_seed(idx + 1);
        b_seed.discard(2);
        std::bernoulli_distribution b_dist(0.5f);

        for (int64_t c = prb->get_c_start(g); c < prb->get_c_start(g + 1);
                ++c) {
            // l[0] must be even
            int64_t l_base = spatial * (idx * prb->ic / prb->g + c * 239 * 2);
            int64_t off = data_off(prb, mb, c, 0, 0, 0);
            bool bigger_val = false; // Out of spatial loop.

            for_(int64_t d = 0; d < prb->id; ++d)
            for_(int64_t h = 0; h < prb->ih; ++h)
            for (int64_t w = 0; w < prb->iw; ++w) {
                const int64_t sp = d * prb->ih * prb->iw + h * prb->iw + w;
                float val = 0.f;
                if (cfg.check_alg_ == ALG_2) {
                    const bool is_even = sp % 2 == 0;
                    const float sign = is_even ? 1.f : -1.f;
                    // Update the value for even cases.
                    bigger_val = is_even ? b_dist(b_seed) : bigger_val;
                    const float val_shift = sign * val_coeff;
                    // Shift left and right from mean val, shift bigger with
                    // probability.
                    val = m + val_shift + 3.f * bigger_val * val_shift;
                } else {
                    const int64_t l = l_base + sp;

                    // Shortcut for zero values.
                    if (cfg.check_alg_ == ALG_0
                            && !flip_coin(l / 2 * 257ULL, cfg.density_)) {
                        mem_fp.set_elem(off + sp, 0);
                        continue;
                    }

                    const int64_t gen = (l / 2 * 1637) & cfg.flex_mask_;
                    // s_{i} + s_{i+1} = 2 * m
                    const float sign = l % 2 == 0 ? 1.f : -1.f;
                    const float f = sign * gen / (1 << cfg.flex_bits_);

                    val = cfg.check_alg_ == ALG_0 ? f : m * (1.f + f);
                    if (prb->flags & GLOB_STATS) {
                        val = (l % 65) - 32;
                    } else if (cfg.L_ % 2 && (c * spatial + sp == cfg.L_ - 1)) {
                        val = m;
                    }
                }
                mem_fp.set_elem(off + sp,
                        round_to_nearest_representable(prb->dt[0], val));
            }
        }
    });

    if (mem_dt) SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_variance_fwd(const prb_t *prb, const cfg_t &cfg, dnn_mem_t &mem_fp,
        dnn_mem_t &mem_dt, const dnn_mem_t &ref_src,
        const dnn_mem_t &ref_mean) {
    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        // Variance must be computed unless it is passed by user directly.
        if (!(prb->flags & GLOB_STATS)) return OK;

        // Variance must be always positive by definition.
        fill_cfg_t fill_cfg(dnnl_f32, 0.f, 16.f, /* int = */ false,
                attr_t::post_ops_t::kind_t::ADD, "variance");
        return fill_random_real(mem_dt, mem_fp, nullptr, fill_cfg);
    }

    benchdnn_parallel_nd(prb->mb, prb->g, [&](int64_t mb, int64_t g) {
        const int64_t idx = mb * prb->g + g;

        float val = 0.f;
        if (prb->flags & GLOB_STATS) {
            val = ((idx % 7) << 1);
        } else {
            const float m = ref_mean.get_elem(idx);
            for (int64_t c = prb->get_c_start(g); c < prb->get_c_start(g + 1);
                    ++c) {
                int64_t off = data_off(prb, mb, c, 0, 0, 0);

                for_(int64_t d = 0; d < prb->id; ++d)
                for_(int64_t h = 0; h < prb->ih; ++h)
                for (int64_t w = 0; w < prb->iw; ++w) {
                    const int64_t sp = d * prb->ih * prb->iw + h * prb->iw + w;
                    const float s = ref_src.get_elem(sp + off);
                    val += (s - m) * (s - m);
                }
            }
            val /= cfg.L_;
        }
        mem_fp.set_elem(idx, val);
    });

    if (mem_dt && prb->use_stats()) SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_scale(const prb_t *prb, dnn_mem_t &mem_fp, dnn_mem_t &mem_dt) {
    const bool use_sc = prb->use_sc();
    if (!use_sc) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        return fill_random_real(mem_dt, mem_fp, nullptr);
    }

    benchdnn_parallel_nd(prb->ic, [&](int64_t c) {
        float val = (1.f / 8) * (1 << (c % 7));
        if (prb->flags & GLOB_STATS) val *= 8.f;
        mem_fp.set_elem(c, val);
    });

    if (mem_dt) SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_shift(const prb_t *prb, dnn_mem_t &mem_fp, dnn_mem_t &mem_dt) {
    const bool use_sh = prb->use_sh();
    if (!use_sh) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        return fill_random_real(mem_dt, mem_fp, nullptr);
    }

    benchdnn_parallel_nd(prb->ic, [&](int64_t c) {
        float val = ((c % 3) - 1) * (1.f / 512 * (1 << (c % 7)));
        if (prb->flags & GLOB_STATS) val *= 512.f;
        mem_fp.set_elem(c, val);
    });

    if (mem_dt) SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int prepare_fwd(const prb_t *prb, dnn_mem_map_t &mem_map,
        dnn_mem_map_t &ref_mem_map, res_t *res) {
    cfg_t cfg(prb);

    auto &mean = mem_map.at(DNNL_ARG_MEAN);
    auto &ref_mean = ref_mem_map.at(DNNL_ARG_MEAN);
    SAFE(fill_mean(prb, cfg, ref_mean, mean), WARN);

    auto &src = mem_map.at(DNNL_ARG_SRC);
    auto &ref_src = ref_mem_map.at(DNNL_ARG_SRC);
    SAFE(fill_src(prb, cfg, ref_src, src, ref_mean, res), WARN);

    // Need a copy of source data for inplace mode for bitwise testing.
    if (has_bench_mode_bit(mode_bit_t::bitwise) && prb->inplace) {
        auto &src_copy = mem_map.at(-DNNL_ARG_SRC);
        SAFE(bool(src_copy) ? OK : FAIL, WARN);
        SAFE(src_copy.reorder(src), WARN);
    }

    auto &var = mem_map.at(DNNL_ARG_VARIANCE);
    auto &ref_var = ref_mem_map.at(DNNL_ARG_VARIANCE);
    SAFE(fill_variance_fwd(prb, cfg, ref_var, var, ref_src, ref_mean), WARN);

    auto &scale = mem_map.at(DNNL_ARG_SCALE);
    auto &ref_scale = ref_mem_map.at(DNNL_ARG_SCALE);
    SAFE(fill_scale(prb, ref_scale, scale), WARN);

    auto &shift = mem_map.at(DNNL_ARG_SHIFT);
    auto &ref_shift = ref_mem_map.at(DNNL_ARG_SHIFT);
    SAFE(fill_shift(prb, ref_shift, shift), WARN);

    return OK;
}

int fill_variance_bwd(const prb_t *prb, dnn_mem_t &mem_fp, dnn_mem_t &mem_dt) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        // Variance must be always positive by definition.
        fill_cfg_t fill_cfg(dnnl_f32, 0.f, 16.f, /* int = */ false,
                attr_t::post_ops_t::kind_t::ADD, "variance");
        return fill_random_real(mem_dt, mem_fp, nullptr, fill_cfg);
    }

    benchdnn_parallel_nd(prb->mb, prb->g, [&](int64_t mb, int64_t g) {
        const int64_t idx = mb * prb->g + g;
        // final variance = {0.25f, 1.f, 4.f}
        const float val = 0.25f * (1 << ((idx % 3) * 2));
        mem_fp.set_elem(idx, val - prb->eps);
    });

    if (mem_dt) SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_src_bwd(const prb_t *prb, dnn_mem_t &mem_fp, dnn_mem_t &mem_dt,
        const dnn_mem_t &ref_mean, res_t *res) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        return fill_random_real(mem_dt, mem_fp, res);
    }

    const auto SP = prb->id * prb->ih * prb->iw;

    benchdnn_parallel_nd(prb->mb, prb->g, [&](int64_t mb, int64_t g) {
        // Idea behind the filling is to reduce a possibility of cancellation
        // when subtracting a part accumulated over N. For that, we simplify
        // src data to (m+1) and (m-1) points, d_dst data is more or less
        // random but we keep all values as pow2 values to have almost exact
        // summation result.
        int64_t idx = mb * prb->g + g;
        const float m = ref_mean.get_elem(idx);

        for_(int64_t c = prb->get_c_start(g); c < prb->get_c_start(g + 1); ++c)
        for (int64_t sp = 0; sp < SP; ++sp) {
            const int64_t off = data_off(prb, mb, c, 0, 0, sp);
            const float val = sp % 2 == 0 ? (m - 1.f) : (m + 1.f);
            mem_fp.set_elem(
                    off, round_to_nearest_representable(prb->dt[0], val));
        }
    });

    if (mem_dt) SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_diff_dst_bwd(
        const prb_t *prb, dnn_mem_t &mem_fp, dnn_mem_t &mem_dt, res_t *res) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        return fill_random_real(mem_dt, mem_fp, res);
    }

    const auto SP = prb->id * prb->ih * prb->iw;

    benchdnn_parallel_nd(prb->mb, prb->g, [&](int64_t mb, int64_t g) {
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand int_seed(mb + 1);
        int_seed.discard(1);
        std::minstd_rand b_seed(mb + 1);
        b_seed.discard(2);

        // See `fill_src_bwd` comment.
        std::uniform_int_distribution<> data_dist(0, 6);
        std::bernoulli_distribution half_dist(0.5f);

        for_(int64_t c = prb->get_c_start(g); c < prb->get_c_start(g + 1); ++c)
        for (int64_t sp = 0; sp < SP; ++sp) {
            const int64_t off = data_off(prb, mb, c, 0, 0, sp);
            const float sign = half_dist(b_seed) ? 1.f : -1.f;
            // d_dst = powf(2, {-4, ... , 2})
            const float val = sign * 0.0625f * (1LL << data_dist(int_seed));
            mem_fp.set_elem(
                    off, round_to_nearest_representable(prb->dt[0], val));
        }
    });

    if (mem_dt) SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int prepare_bwd(const prb_t *prb, dnn_mem_map_t &mem_map,
        dnn_mem_map_t &ref_mem_map, res_t *res) {
    cfg_t cfg(prb);

    auto &mean = mem_map.at(DNNL_ARG_MEAN);
    auto &ref_mean = ref_mem_map.at(DNNL_ARG_MEAN);
    SAFE(fill_mean(prb, cfg, ref_mean, mean), WARN);

    auto &var = mem_map.at(DNNL_ARG_VARIANCE);
    auto &ref_var = ref_mem_map.at(DNNL_ARG_VARIANCE);
    SAFE(fill_variance_bwd(prb, ref_var, var), WARN);

    auto &src = mem_map.at(DNNL_ARG_SRC);
    auto &ref_src = ref_mem_map.at(DNNL_ARG_SRC);
    SAFE(fill_src_bwd(prb, ref_src, src, ref_mean, res), WARN);

    auto &d_dst = mem_map.at(DNNL_ARG_DIFF_DST);
    auto &ref_d_dst = ref_mem_map.at(DNNL_ARG_DIFF_DST);
    SAFE(fill_diff_dst_bwd(prb, ref_d_dst, d_dst, res), WARN);

    // Need a copy of source data for inplace mode for bitwise testing.
    if (has_bench_mode_bit(mode_bit_t::bitwise) && prb->inplace) {
        auto &d_dst_copy = mem_map.at(-DNNL_ARG_DIFF_DST);
        SAFE(bool(d_dst_copy) ? OK : FAIL, WARN);
        SAFE(d_dst_copy.reorder(d_dst), WARN);
    }

    auto &scale = mem_map.at(DNNL_ARG_SCALE);
    auto &ref_scale = ref_mem_map.at(DNNL_ARG_SCALE);
    SAFE(fill_scale(prb, ref_scale, scale), WARN);

    return OK;
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;
    res_t *res = init_pd_args.res;

    auto src_d = dnn_mem_t::init_md(
            prb->ndims, prb->data_dims().data(), prb->dt[0], prb->tag[0]);

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(
            prb->attr, prb->ndims, prb->data_dims().data());
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    auto flags = (dnnl_normalization_flags_t)prb->flags;
    if (prb->dir & FLAG_FWD) {
        auto dst_d = dnn_mem_t::init_md(
                prb->ndims, prb->data_dims().data(), prb->dt[1], prb->tag[1]);
        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;
        TIME_C_PD(DNN_SAFE_STATUS(
                dnnl_group_normalization_forward_primitive_desc_create(
                        &init_pd_args.pd, init_pd_args.engine, prop,
                        init_pd_args.src_md ? init_pd_args.src_md : src_d,
                        dst_d, prb->g, prb->eps, flags, dnnl_attr)));
    } else {
        auto diff_src_d = dnn_mem_t::init_md(
                prb->ndims, prb->data_dims().data(), prb->dt[0], prb->tag[0]);
        auto diff_dst_d = dnn_mem_t::init_md(
                prb->ndims, prb->data_dims().data(), prb->dt[1], prb->tag[1]);
        auto prop = prb->dir & FLAG_WEI ? dnnl_backward : dnnl_backward_data;
        TIME_C_PD(DNN_SAFE_STATUS(
                dnnl_group_normalization_backward_primitive_desc_create(
                        &init_pd_args.pd, init_pd_args.engine, prop, diff_src_d,
                        diff_dst_d, src_d, prb->g, prb->eps, flags,
                        init_pd_args.hint, dnnl_attr)));
    }

    return dnnl_success;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type({prb->dt[0], prb->dt[1]}, prb->dir, res);
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    // See `skip_invalid_inplace` for details.
    if (prb->inplace) {
        skip_invalid_inplace(
                res, prb->dt[0], prb->dt[1], prb->tag[0], prb->tag[1]);
        if (res->state == SKIPPED) return;
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    const bool compare_with_norm = (prb->dir & FLAG_BWD);
    cmp.set_norm_validation_mode(compare_with_norm);

    const auto dt = prb->dir & FLAG_FWD ? prb->dt[1] : prb->dt[0];
    // Digits must be non-negative for safe left-shifting when `digits_dt`
    // exceeds `digits_f32`.
    const int safe_digits = MAX2(0, digits_dt(dnnl_f32) - digits_dt(dt));
    const float trh_coeff = (1 << safe_digits);
    float trh = trh_coeff * ((kind == SRC || kind == DST) ? 6e-7 : 0);
    if ((kind == SC || kind == SH) && prb->dir & FLAG_BWD)
        trh = trh_coeff * 5e-6;
    cmp.set_threshold(trh);

    // u8 turns half of output into zeros.
    if (prb->dt[1] == dnnl_u8) cmp.set_zero_trust_percent(60.f);

    // When the error is larger than `trh`, it could be due to a catastrophic
    // cancellation in final result which is computed as `Y = a * X + b`.
    // When `a * X` is close to `b` and their signs are opposite, then large
    // error in `a * X` could result in a final result (which has a cancellation
    // i.e. `|Y| = |a*X - (-b)|`), which has no meaningful digits left in
    // mantissa.
    //
    // Since lambda is called when stack is unavailable, need to capture `prb`
    // and `kind` by value to avoid using dangling references.
    const auto gnorm_add_check =
            [&, kind, prb](
                    const compare::compare_t::driver_check_func_args_t &args) {
                if (!((prb->dir & FLAG_FWD) && kind == DST && prb->use_sh()))
                    return false;

                const auto &sh = ref_args.find(DNNL_ARG_SHIFT);
                const auto &dst = ref_args.find(DNNL_ARG_DST);
                const int64_t c = dst.get_scale_idx(
                        args.idx, 1 << 1 /* last_dim_mask */);
                const float beta = sh.get_elem(c);
                // Using an empirically derived threshold, check if
                // cancellation error in `|Y| = |a*X - (-b)|` is huge.
                const float abs_exp = fabsf(args.exp);
                const float norm_denom = abs_exp > FLT_MIN ? abs_exp : 1.f;
                const float abs_exp_delta = fabsf(args.exp - beta);
                bool maybe_cancel_error = abs_exp_delta / norm_denom > 1.f;
                if (!maybe_cancel_error) return false;

                // Check for error in `a * X`
                float diff_aX = fabsf((args.exp - beta) - (args.got - beta));
                float rel_diff_aX = diff_aX
                        / (abs_exp_delta > FLT_MIN ? abs_exp_delta : 1.f);
                return rel_diff_aX <= args.trh;
            };
    cmp.set_driver_check_function(gnorm_add_check);
}

std::vector<int> supported_exec_args(dir_t dir) {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_MEAN,
            DNNL_ARG_VARIANCE,
            DNNL_ARG_SCALE,
            DNNL_ARG_SHIFT,
            DNNL_ARG_DST,
    };
    static const std::vector<int> exec_bwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_MEAN,
            DNNL_ARG_VARIANCE,
            DNNL_ARG_SCALE,
            DNNL_ARG_DIFF_DST,
            DNNL_ARG_DIFF_SCALE,
            DNNL_ARG_DIFF_SHIFT,
            DNNL_ARG_DIFF_SRC,
    };
    return (dir & FLAG_FWD) ? exec_fwd_args : exec_bwd_args;
};

fill_cfg_t binary_po_fill_cfg(
        int exec_arg, const dnn_mem_t &mem, const attr_t &attr) {
    fill_cfg_t cfg;
    const int post_ops_range = DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)
            - DNNL_ARG_ATTR_MULTIPLE_POST_OP(0);
    const bool is_post_ops_arg = (exec_arg & post_ops_range);
    if (is_post_ops_arg) {
        // GNorm output values are in [-1.f; 1.f] range. Using small values
        // leads to the cancellation effect. Config secures only positive and
        // big enough values are used.
        const int bin_po_idx
                = exec_arg / DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE - 1;
        assert(bin_po_idx < attr.post_ops.len());
        const auto alg = attr.post_ops.entry[bin_po_idx].kind;
        cfg = fill_cfg_t(mem.dt(), 4.f, 16.f, /* int = */ true, alg,
                "gnorm_binary_post_op");
    }
    return cfg;
}

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const prb_t *prb, res_t *res, dir_t dir,
        dnnl_primitive_t prim_ref) {
    update_inplace_memory_args(mem_map, prb, dir);
    if (has_bench_mode_modifier(mode_modifier_t::no_host_memory)) return OK;

    // TODO: this function still allocates the full memory print needed to fill
    // the data and each argument can't be destroyed right away since filling
    // requires all of them at a time.
    const auto &ref_engine = get_cpu_engine();

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        // The function targets regular exec_args that are positive.
        // Negative args are used by bitwise and are broken in the `default`
        // branch due to `&` always returns `true`.
        if (exec_arg <= 0) continue;

        auto &mem = entry.second;

        // Scratchpad memory relates to a primitive. If reference needs it,
        // use switch below to define a memory desc for it.
        if (exec_arg != DNNL_ARG_SCRATCHPAD) {
            ref_mem_map.emplace(exec_arg,
                    dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine));
        }
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_DST: break; // Skip on backward.
            case DNNL_ARG_DIFF_SRC: break; // Skip on backward.
            case DNNL_ARG_MEAN:
            case DNNL_ARG_VARIANCE:
                if (prb->dir & FLAG_INF) {
                    const dnnl_dims_t dims2d = {prb->mb, prb->g};
                    ref_mem_map[exec_arg] = dnn_mem_t(
                            2, dims2d, dnnl_f32, tag::abx, ref_engine);
                }
                break;
            default: {
                const auto &binary_fill_cfg
                        = binary_po_fill_cfg(exec_arg, mem, prb->attr);
                std::unordered_map<int, fill_cfg_t> fill_cfg_map {
                        {DNNL_ARG_SRC_1, binary_fill_cfg}};
                SAFE(init_ref_memory_args_default_case(exec_arg, mem, ref_mem,
                             prb->attr, res, fill_cfg_map),
                        WARN);
            } break;
        }
    }

    if (dir & FLAG_FWD) {
        SAFE(prepare_fwd(prb, mem_map, ref_mem_map, res), WARN);
    } else {
        SAFE(prepare_bwd(prb, mem_map, ref_mem_map, res), WARN);
    }

    // Don't keep reference memory if it is not used further.
    if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();

    return OK;
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb) {
    std::vector<data_kind_t> check_kinds;
    if (prb->dir & FLAG_FWD) {
        if (!(prb->flags & GLOB_STATS) && !(prb->dir & FLAG_INF)) {
            check_kinds.push_back(MEAN);
            check_kinds.push_back(VAR);
        }
        check_kinds.push_back(DST);
    } else {
        if (prb->dir & FLAG_WEI) {
            if (prb->use_sc()) check_kinds.push_back(SC);
            if (prb->use_sh()) check_kinds.push_back(SH);
        }
        check_kinds.push_back(SRC);
    }
    assert(!check_kinds.empty());
    return check_kinds;
}

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    v_prim.resize(1);
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res), WARN);
    return OK;
}

int check_cacheit(
        std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    SAFE(check_caches(v_prim[0], prb, res), WARN);
    return OK;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    const auto &prim = v_prim[0];

    dnn_mem_map_t mem_map, ref_mem_map;
    init_memory_args<prb_t>(mem_map, prb, prim, supported_exec_args(prb->dir));
    TIME_FILL(SAFE(init_ref_memory_args(
                           ref_mem_map, mem_map, prim, prb, res, prb->dir),
            WARN));

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(execute_and_wait(prim, args, res), WARN);

    check_correctness(
            prb, get_kinds_to_check(prb), args, ref_args, setup_cmp, res);
    SAFE(check_bitwise(prim, get_kinds_to_check(prb), args, prb->inplace, res),
            WARN);

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace gnorm
