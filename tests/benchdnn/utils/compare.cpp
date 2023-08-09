/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include <algorithm>
#include <atomic>
#include <cmath>
#include <sstream>
#include <string>

#include "utils/parallel.hpp"

#include "common.hpp"
#include "utils/compare.hpp"
#include "utils/norm.hpp"

#include "eltwise/eltwise.hpp"

namespace compare {

namespace {
void dump_point_values(const_dnnl_memory_desc_t md, const std::string &kind_str,
        int64_t l_offset, float exp_f32, float exp, float got, float diff,
        float rel_diff) {
    std::stringstream ss;
    dims_t l_dims = md2dims(md);
    dims_t dims_idx = off2dims_idx(l_dims, l_offset);
    ss << dims_idx;
    std::string ind_str = ss.str();

    BENCHDNN_PRINT(0,
            "[%4ld]%s[%s] exp_f32:%12g exp:%12g got:%12g diff:%8g rdiff:%8g\n",
            (long)l_offset, kind_str.c_str(), ind_str.c_str(), exp_f32, exp,
            got, diff, rel_diff);
}

void dump_norm_values(
        const diff_norm_t &diff_norm, const std::string &kind_str) {
    BENCHDNN_PRINT(0,
            "%s[L0] = %g\n"
            "%s[L1] exp:%8g got:%8g diff:%8g rel_diff:%8g\n"
            "%s[L2] exp:%8g got:%8g diff:%8g rel_diff:%8g\n"
            "%s[L8] exp:%8g got:%8g diff:%8g rel_diff:%8g\n",
            kind_str.c_str(), diff_norm.rel_diff(norm_t::L0), kind_str.c_str(),
            diff_norm.a_[norm_t::L1], diff_norm.b_[norm_t::L1],
            diff_norm.diff_[norm_t::L1], diff_norm.rel_diff(norm_t::L1),
            kind_str.c_str(), diff_norm.a_[norm_t::L2],
            diff_norm.b_[norm_t::L2], diff_norm.diff_[norm_t::L2],
            diff_norm.rel_diff(norm_t::L2), kind_str.c_str(),
            diff_norm.a_[norm_t::L8], diff_norm.b_[norm_t::L8],
            diff_norm.diff_[norm_t::L8], diff_norm.rel_diff(norm_t::L8));
}

bool has_binary_comparison_po(const attr_t &attr) {
    const auto &po = attr.post_ops;
    if (po.is_def()) return false;

    using alg_t = attr_t::post_ops_t::kind_t;
    static const std::vector<alg_t> cmp_alg = {alg_t::MAX, alg_t::MIN,
            alg_t::GE, alg_t::GT, alg_t::LE, alg_t::LT, alg_t::EQ, alg_t::NE};

    for (int idx = 0; idx < po.len(); ++idx) {
        const auto &e = po.entry[idx];
        if (!e.is_binary_kind()) continue;

        if (std::any_of(cmp_alg.cbegin(), cmp_alg.cend(),
                    [&](const alg_t alg) { return e.kind == alg; }))
            return true;
    }
    return false;
}

bool negative_converts_to_zero(const attr_t &attr, dnnl_data_type_t target_dt) {
    using po_kind_t = attr_t::post_ops_t::kind_t;
    const auto &po = attr.post_ops;

    // Check for all post-ops that convert negative to zero
    std::vector<po_kind_t> non_neg_po {po_kind_t::ABS};
    std::vector<po_kind_t> non_neg_alpha_0_po {po_kind_t::CLIP,
            po_kind_t::CLIP_V2, po_kind_t::ELU, po_kind_t::RELU};
    for (int i = 0; i < po.len(); ++i) {
        const auto &e = po.entry[i];
        if (!e.is_eltwise_kind()) continue;

        auto k = e.kind;
        auto alpha = e.eltwise.alpha;

        if (std::any_of(non_neg_po.cbegin(), non_neg_po.cend(),
                    [k](const po_kind_t alg) { return alg == k; }))
            return true;

        if (std::any_of(non_neg_alpha_0_po.cbegin(), non_neg_alpha_0_po.cend(),
                    [k, alpha](const po_kind_t alg) {
                        return alg == k && alpha == 0;
                    }))
            return true;
    }
    // Check for u8 dst
    if (target_dt == dnnl_u8) return true;

    return false;
}
} // namespace

bool compare_extreme_values(float a, float b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    if (std::isinf(a) && std::isinf(b) && std::signbit(a) == std::signbit(b))
        return true;
    return false;
}

compare_t::driver_check_func_args_t::driver_check_func_args_t(
        const dnn_mem_t &exp_mem, const dnn_mem_t &got_f32, const int64_t i,
        const dnnl_data_type_t data_type, const float trh)
    : dt(data_type)
    , idx(i)
    , exp_f32(exp_mem.get_elem(idx))
    , exp(round_to_nearest_representable(dt, exp_f32))
    , got(got_f32.get_elem(idx))
    , diff(fabsf(exp - got))
    , rel_diff(diff / (fabsf(exp) > FLT_MIN ? fabsf(exp) : 1))
    , trh(trh) {}

int compare_t::compare_norm(const dnn_mem_t &exp_mem, const dnn_mem_t &got_mem,
        const attr_t &attr, res_t *res) const {
    const auto nelems = got_mem.nelems();
    if (nelems == 0) {
        if (res->state == EXECUTED) res->state = PASSED;
        return OK;
    }

    res->total = nelems;

    dnn_mem_t got_f32(got_mem, dnnl_f32, tag::abx, get_cpu_engine());
    const auto dt = got_mem.dt();

    diff_norm_t diff_norm;
    const bool need_dump = verbose >= 99;
    for (int64_t i = 0; i < nelems; ++i) {
        driver_check_func_args_t args(exp_mem, got_f32, i, dt, trh_);

        if ((std::isnan(args.exp_f32) && is_integral_dt(dt))
                || std::isinf(args.exp_f32)) {
            // Don't include integer max values or inf values into norm as they
            // make it irrelevant for validation.
            ;
        } else if (is_cpu() && dt == dnnl_s32 && args.exp == max_dt(dnnl_s32)
                && args.got >= BENCHDNN_S32_TO_F32_SAT_CONST
                && args.got < max_dt(dnnl_s32)) {
            // Don't include f32->s32 saturation values into norm as they make
            // it irrelevant for validation.
            ;
        } else {
            diff_norm.update(args.exp, args.got);
        }

        if (need_dump)
            dump_point_values(got_mem.md_, get_kind_str(), i, args.exp_f32,
                    args.exp, args.got, args.diff, args.rel_diff);
    }
    diff_norm.done();

    bool ok = diff_norm.rel_diff(norm_t::L2) <= trh_;
    if (!ok) res->errors = 1;

    const bool dump = need_dump || !ok;
    if (dump) dump_norm_values(diff_norm, get_kind_str());

    if (res->errors) res->state = FAILED;
    if (res->state == EXECUTED) res->state = PASSED;

    return res->state == FAILED ? FAIL : OK;
}

int compare_t::compare_p2p(const dnn_mem_t &exp_mem, const dnn_mem_t &got_mem,
        const attr_t &attr, res_t *res) const {
    const auto nelems = got_mem.nelems();
    if (nelems == 0) {
        if (res->state == EXECUTED) res->state = PASSED;
        return OK;
    }

    res->total = nelems;

    dnn_mem_t got_f32(got_mem, dnnl_f32, tag::abx, get_cpu_engine());
    const auto dt = got_mem.dt();
    const bool has_eltwise
            = attr.post_ops.eltwise_index() != -1 || has_eltwise_post_op_;
    const bool output_has_nans = op_output_has_nans_
            || eltwise::eltwise_alg_returns_nan_or_inf(attr)
            || got_mem.dt() == dnnl_f16;
    const bool has_exp_eltwise
            = attr.post_ops.find(attr_t::post_ops_t::kind_t::EXP) >= 0;
    const bool has_dst_scale = !attr.scales.get(DNNL_ARG_DST).is_def();

    // Atomics to be updated in parallel section, non-atomics - in sequential.
    std::atomic<bool> all_ok(true);
    std::atomic<int64_t> zeros(0);
    std::atomic<float> max_rdiff(0), max_diff(0);
    int64_t n_errors = 0;
    volatile bool from_parallel = true;
    const bool need_dump = verbose >= 99;

    const auto compare_point_values = [&](int64_t i) {
        driver_check_func_args_t args(exp_mem, got_f32, i, dt, trh_);

        bool ok = args.diff == 0.f;
        if (std::isnan(args.exp_f32) && is_integral_dt(dt)) {
            // Relax output requirements for this case, since different backends
            // may implement NaN fp32 -> int32 conversion in a different manner.
            ok = true;
        }
        // If fast check failed, go through all of them.
        if (!ok) {
            // Standard check for relative diff is under set threshold...
            ok = (fabsf(args.exp) > 1e-5f ? args.rel_diff : args.diff) <= trh_;
            // If not, when NaNs or infinity are allowed for the driver, check
            // that both exp and got are NaNs or infinity with same sign...
            if (!ok && output_has_nans)
                ok = compare::compare_extreme_values(args.exp, args.got);
            // If not, use hack to check not fully correct s32 saturation on
            // cpu...
            if (!ok && is_cpu() && dt == dnnl_s32
                    && args.exp == max_dt(dnnl_s32))
                ok = args.got >= BENCHDNN_S32_TO_F32_SAT_CONST
                        && args.got < max_dt(dnnl_s32);
            // If not, check driver additional checks if set...
            if (!ok && driver_check_func_) ok = driver_check_func_(args);
            // If not, check if there are eltwise post-ops, use very relaxed
            // comparison since we can't control inputs for each driver finely
            // or validate if the output value from operation satisfies the
            // check for catastrophic cancellation (see eltwise additional check
            // function). We rely on validation of pure eltwise and let some
            // big rdiff errors slip away hoping that absolute error is good
            // enough.
            if (!ok && has_eltwise) {
                const float experimental_tolerated_trh
                        = std::max(epsilon_dt(dt), 2e-5f);
                ok = args.diff <= experimental_tolerated_trh;
            }
            // For eltwise it also may happen that threshold is really small,
            // but absolute difference is really big. Also exponent is a special
            // transcendental post-op that has accuracy issues with older isa.
            if (!ok && has_eltwise
                    && (fabsf(args.exp) > 1e+5f || has_exp_eltwise)) {
                ok = args.rel_diff <= std::max(epsilon_dt(dt), 5e-6f);
            }
            // Attr dst scale is used as a divisor to quantize data to dt.
            // Implementation might decide to pre-compute inverse value and
            // multiply on it in kernel. This difference might result in a
            // slight error comparing to a division operation.
            if (!ok && has_dst_scale) {
                const float experimental_tolerated_trh
                        = std::max(epsilon_dt(dt), 1e-5f);
                ok = args.rel_diff <= experimental_tolerated_trh;
            }
            // Binary MAX, MIN and comparison operations post-ops may return
            // different results for different backends when NaN is one of
            // inputs. Depending on its position and implementation, either
            // first or second operand may be returned.
            if (!ok && has_binary_comparison_po(attr) && output_has_nans)
                ok = true;
            // Some drivers (like pooling or resampling) on integer data types
            // may result in sporadic order of operations. This may cause a
            // difference around `x.5f` value, and can be rounded either way to
            // `x` or `x + 1` which can't be fixed by filling.
            if (!ok && is_integral_dt(args.dt)) {
                // Check that original value is close to x.5f.
                static constexpr float small_eps = 9e-6f;
                const float floor_val = floorf(args.exp_f32);
                const float ceil_val = ceilf(args.exp_f32);
                if (fabsf((floor_val + 0.5f) - args.exp_f32) < small_eps) {
                    // If it is, check exp and got values are on opposite sides.
                    if (args.exp == floor_val) {
                        ok = args.got == ceil_val;
                    } else if (args.exp == ceil_val) {
                        ok = args.got == floor_val;
                    }
                }
            }
            // Nvidia backend with fpmath mode enabled returns not exact output
            // values (presumably on conversion to fp32), thus, make sure they
            // fit single ulp for a reduced data type.
            if (!ok && is_nvidia_gpu()
                    && attr.fpmath_mode != dnnl_fpmath_mode_strict) {
                const auto deduced_src_dt = deduce_cfg_data_type(dt, attr, SRC);
                ok = args.diff <= epsilon_dt(deduced_src_dt);
            }
        }
        // Update compare stats.
        if (from_parallel && fabsf(args.got) == 0) zeros++;
        if (from_parallel && verbose >= 6) {
            max_rdiff = MAX2(max_rdiff.load(), args.rel_diff);
            max_diff = MAX2(max_diff.load(), args.diff);
        }

        if (!ok && all_ok) all_ok = false;
        if (!ok && !from_parallel) n_errors++;

        const bool dump
                = need_dump || (!ok && (n_errors < 10 || verbose >= 10));
        if (!from_parallel && dump)
            dump_point_values(got_mem.md_, get_kind_str(), i, args.exp_f32,
                    args.exp, args.got, args.diff, args.rel_diff);
    };

    // parallel comparison to speed up the process
    benchdnn_parallel_nd(nelems, compare_point_values);

    // serial comparison with enabled dumping when needed for nicer output.
    if (!all_ok || need_dump) {
        from_parallel = false;
        for (int64_t i = 0; i < nelems; ++i)
            compare_point_values(i);
    }

    // Set state to FAILED in case of any errors.
    if (n_errors) res->errors = n_errors, res->state = FAILED;
    // State could be already FAILED, check zero trust for non-FAILED only.
    const float zeros_percent = 100.f * zeros / nelems;
    float zero_trust_percent = zero_trust_percent_;
    // Adjust default zero trust for cases when negative are converted into 0.
    if (zero_trust_percent_ == default_zero_trust_percent_
            && negative_converts_to_zero(attr, dt)) {
        // (100% - X%) / 2 + X%. X% is default. Each half represents positive
        // and negative in the output equally.
        zero_trust_percent = (100.f + zero_trust_percent_) / 2.f;
    }
    if (res->state != FAILED && zeros_percent > zero_trust_percent
            && nelems >= 10)
        res->state = MISTRUSTED;

    BENCHDNN_PRINT(6, "[COMPARE_STATS]%s: max_diff:%8g max_rdiff:%8g\n",
            get_kind_str().c_str(), max_diff.load(), max_rdiff.load());

    BENCHDNN_PRINT((res->state == MISTRUSTED ? 2 : 6),
            "[COMPARE_TRUST]%s: z:%2.0f%% (>%2.0f%%) (z: %ld, total: %ld)\n",
            get_kind_str().c_str(), zeros_percent, zero_trust_percent,
            (long)zeros.load(), (long)nelems);

    // Set PASSED if no failure in current or previous checks happened and test
    // can be trusted.
    if (res->state == EXECUTED) res->state = PASSED;

    return res->state == FAILED ? FAIL : OK;
}

int compare_t::compare(const dnn_mem_t &exp_mem, const dnn_mem_t &got_mem,
        const attr_t &attr, res_t *res) const {
    std::string add_args = std::string(use_norm_ ? "use_norm:true" : "")
            + std::string(op_output_has_nans_ ? "has_nans:true" : "");
    BENCHDNN_PRINT(6, "[COMPARE]%s: trh=%g zero_trust%%=%.2f%% extra=%s\n",
            get_kind_str().c_str(), trh_, zero_trust_percent_,
            add_args.c_str());
    if (use_norm_) return compare_norm(exp_mem, got_mem, attr, res);
    return compare_p2p(exp_mem, got_mem, attr, res);
}

} // namespace compare
