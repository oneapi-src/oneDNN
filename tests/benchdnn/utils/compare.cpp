/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

static void dump_point_values(const dnnl_memory_desc_t &md, data_kind_t kind,
        int64_t l_offset, float exp_f32, float exp, float got, float diff,
        float rel_diff) {
    std::stringstream ss;
    dims_t l_dims = md2dims(md);
    dims_t dims_idx = off2dims_idx(l_dims, l_offset);
    ss << dims_idx;
    std::string ind_str = ss.str();

    std::string skind;
    if (kind != DAT_TOTAL) skind = "[" + std::string(data_kind2str(kind)) + "]";

    BENCHDNN_PRINT(0,
            "[%4ld]%s[%s] exp_f32:%8g exp:%8g got:%8g diff:%8g rdiff:%8g\n",
            (long)l_offset, skind.c_str(), ind_str.c_str(), exp_f32, exp, got,
            diff, rel_diff);
}

static void dump_norm_values(const diff_norm_t &diff_norm, data_kind_t kind) {
    std::string skind;
    if (kind != DAT_TOTAL) skind = "[" + std::string(data_kind2str(kind)) + "]";

    BENCHDNN_PRINT(0,
            "%s[L0] = %g\n"
            "%s[L1] exp:%8g got:%8g diff:%8g rel_diff:%8g\n"
            "%s[L2] exp:%8g got:%8g diff:%8g rel_diff:%8g\n"
            "%s[L8] exp:%8g got:%8g diff:%8g rel_diff:%8g\n",
            skind.c_str(), diff_norm.rel_diff(norm_t::L0), skind.c_str(),
            diff_norm.a_[norm_t::L1], diff_norm.b_[norm_t::L1],
            diff_norm.diff_[norm_t::L1], diff_norm.rel_diff(norm_t::L1),
            skind.c_str(), diff_norm.a_[norm_t::L2], diff_norm.b_[norm_t::L2],
            diff_norm.diff_[norm_t::L2], diff_norm.rel_diff(norm_t::L2),
            skind.c_str(), diff_norm.a_[norm_t::L8], diff_norm.b_[norm_t::L8],
            diff_norm.diff_[norm_t::L8], diff_norm.rel_diff(norm_t::L8));
}

static bool has_binary_comparison_po(const attr_t &attr) {
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

        if (std::isnan(args.exp_f32) && is_integral_dt(dt)) {
            // Don't include integer max values into norm as they make it
            // irrelevant for validation.
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
            dump_point_values(got_mem.md_, kind_, i, args.exp_f32, args.exp,
                    args.got, args.diff, args.rel_diff);
    }
    diff_norm.done();

    bool ok = diff_norm.rel_diff(norm_t::L2) <= trh_;
    if (!ok) res->errors = 1;

    const bool dump = need_dump || !ok;
    if (dump) dump_norm_values(diff_norm, kind_);

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
    const bool has_eltwise = attr.post_ops.eltwise_index() != -1;

    // Atomics to be updated in parallel section, non-atomics - in sequential.
    std::atomic<bool> all_ok(true);
    std::atomic<int64_t> zeros(0);
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
            // If not, check that both are NaNs or infinity with same sign...
            if (!ok) ok = compare::compare_extreme_values(args.exp, args.got);
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
            // but absolute difference is really big.
            if (!ok && has_eltwise && (fabsf(args.exp) > 1e+6f)) {
                ok = args.rel_diff <= std::max(epsilon_dt(dt), 5e-7f);
            }
            // Binary MAX, MIN and comparison operations post-ops may return
            // different results for different backends when NaN is one of
            // inputs. Depending on its position and implementation, either
            // first or second operand may be returned.
            if (!ok && has_binary_comparison_po(attr) && op_output_has_nans_)
                ok = true;
        }
        // Update zero stats for mistrust testing.
        if (from_parallel && fabsf(args.got) == 0) zeros++;

        if (!ok && all_ok) all_ok = false;
        if (!ok && !from_parallel) n_errors++;

        const bool dump
                = need_dump || (!ok && (n_errors < 10 || verbose >= 10));
        if (!from_parallel && dump)
            dump_point_values(got_mem.md_, kind_, i, args.exp_f32, args.exp,
                    args.got, args.diff, args.rel_diff);
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
    if (res->state != FAILED) {
        const auto zeros_percent = 100.f * zeros / nelems;
        if (nelems >= 10 && zeros_percent > zero_trust_percent_) {
            res->state = MISTRUSTED;
            std::string skind;
            if (kind_ != DAT_TOTAL)
                skind = "[" + std::string(data_kind2str(kind_)) + "]";
            BENCHDNN_PRINT(2,
                    "No trust stats [%s]: z:%2.0f%% (>%2.0f%%) (z: %ld, "
                    "total: %ld)\n",
                    skind.c_str(), zeros_percent, zero_trust_percent_,
                    (long)zeros.load(), (long)nelems);
        }
    }
    // Set PASSED if no failure in current or previous checks happened and test
    // can be trusted.
    if (res->state == EXECUTED) res->state = PASSED;

    return res->state == FAILED ? FAIL : OK;
}

int compare_t::compare(const dnn_mem_t &exp_mem, const dnn_mem_t &got_mem,
        const attr_t &attr, res_t *res) const {
    if (use_norm_) return compare_norm(exp_mem, got_mem, attr, res);
    return compare_p2p(exp_mem, got_mem, attr, res);
}

} // namespace compare
