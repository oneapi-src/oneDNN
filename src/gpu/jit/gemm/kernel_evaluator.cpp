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

#include "kernel_evaluator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

template <typename T1, typename T2>
static inline T1 divUp(T1 x, T2 y) {
    return (x + y - 1) / y;
}

template <typename T1, typename T2>
static inline T1 alignUp(T1 x, T2 y) {
    return divUp(x, y) * y;
}

double evaluateW(const kcatalog::Entry &e, const DerivedEvaluateParams &dp,
        EvaluateAuxOutput &aux) {
    static constexpr double maxPriority = 10000.;
    double priority = e.model.params[kcatalog::ParamWPriority];

    if (priority > maxPriority)
        return priority;
    else if (e.driverInfo.kParallel) {
        int wgCountK = std::max(1, int(dp.hwThreadCapacity / dp.threadCount));
        aux.k0 = alignUp(
                divUp(dp.sizes.k, wgCountK), e.driverInfo.unroll[LoopK]);
        if (aux.k0 < dp.sizes.k)
            return -priority;
        else
            return 2 * maxPriority + priority;
    } else if (dp.threadCount > dp.hwThreadCapacity)
        return 2 * maxPriority - priority;
    else
        return priority;
}

double evaluateSCore(const kcatalog::Entry &e, const DerivedEvaluateParams &dp,
        EvaluateAuxOutput &aux) {
#define PARAM(p) e.model.params[kcatalog::ParamS_##p]

    auto threads = dp.threadCount;
    auto batch = dp.sizes.batch;
    auto m = dp.sizes.m;
    auto n = dp.sizes.n;
    auto k = dp.sizes.k;
    auto kpad = k;
    auto capacity = dp.hwThreadCapacity;
    auto capacity1 = dp.hwMinThreadsToFill;

    if (e.driverInfo.kParallel) {
        if (k > aux.k0) kpad = alignUp(k, aux.k0);
    } else if (e.driverInfo.kParallelLocal) {
        auto k0 = alignUp(
                divUp(k, e.driverInfo.wg[LoopK]), e.driverInfo.unroll[LoopK]);
        k0 = std::max<decltype(k0)>(k0, 2 * e.driverInfo.unroll[LoopK]);
        kpad = k0 * e.driverInfo.wg[LoopK];
    }

    double threadsFull = std::floor(threads / capacity) * capacity;
    double threadsPartial = threads - threadsFull;
    double partialWaves = std::ceil(threadsPartial / capacity1);
    double npartial = std::ceil(threads / capacity1);

    double C0 = (dp.beta == 0.) ? PARAM(C00) : PARAM(C01);
    double C1 = (dp.beta == 0.) ? PARAM(C10) : PARAM(C11);
    double Cm = (dp.beta == 0.) ? PARAM(Cm0) : PARAM(Cm1);
    double ctime = std::max(Cm, C0 + npartial * C1);

    double mtime = PARAM(Ma) * m + PARAM(Mb) * n;

    double Ef = PARAM(Ef);
    double etimeFull = Ef * threadsFull;
    double Ep = std::max(Ef,
            PARAM(Ep0)
                    + (PARAM(Ep1) * dp.partialWaveCount)
                            / std::max(partialWaves, 1.));
    double etimePartial = Ep * partialWaves * capacity1;
    double etimeLB = (etimeFull + etimePartial) * e.driverInfo.unroll[LoopM]
            * e.driverInfo.unroll[LoopN];
    double etimeNoLB = Ef * dp.mPad * dp.nPad * batch;

    double Em = PARAM(Em);
    double etime = (1 - Em) * etimeNoLB + Em * etimeLB;
    if (!dp.effective) {
        double F = PARAM(Fr0) + double(m) * double(n) * double(k) * PARAM(Fr1);
        F = std::max(1.0, std::min(PARAM(Fp), F));
        etime *= F;
    }

    double time = ctime + std::max(mtime * batch, etime) * kpad;

    return time;
#undef PARAM
}

double evaluateS(const kcatalog::Entry &e, const DerivedEvaluateParams &dp,
        EvaluateAuxOutput &aux) {
    if (!e.driverInfo.kParallel)
        return evaluateSCore(e, dp, aux);
    else {
        // Consider choosing k0 to get as close as possible to 1 or 2 full waves.
        int wgCountK1 = std::max(1, int(dp.hwThreadCapacity / dp.threadCount));
        int wgCountK2
                = std::max(1, int(2 * dp.hwThreadCapacity / dp.threadCount));

        int k0_1 = alignUp(divUp(dp.sizes.k, wgCountK1),
                e.driverInfo.unroll[LoopK] * e.driverInfo.wg[LoopK]);
        int k0_2 = alignUp(divUp(dp.sizes.k, wgCountK2),
                e.driverInfo.unroll[LoopK] * e.driverInfo.wg[LoopK]);

        wgCountK1 = std::max<int>(1, divUp(dp.sizes.k, k0_1));
        wgCountK2 = std::max<int>(1, divUp(dp.sizes.k, k0_2));

        auto dp1 = dp;
        dp1.wgCountK = wgCountK1;
        dp1.threadCount *= wgCountK1;
        aux.k0 = k0_1;

        double score = evaluateSCore(e, dp1, aux);

        if (k0_2 != k0_1) {
            auto dp2 = dp;
            dp2.wgCountK = wgCountK2;
            dp2.threadCount *= wgCountK2;
            aux.k0 = k0_2;

            double score2 = evaluateSCore(e, dp2, aux);
            if (score2 < score)
                score = score2;
            else
                aux.k0 = k0_1;
        }

        // Add cost of initial beta scaling if not 1.
        if (dp.beta != 1.) {
            auto dp0 = dp;
            dp0.sizes.k = 0;
            score += evaluateSCore(e, dp0, aux);
        }

        return score;
    }
}

bool alwaysAccept(const kcatalog::Entry &e, const EvaluateParams &p) {
    int64_t mnk[3] = {p.sizes.m, p.sizes.n, p.sizes.k};
    bool accept = true, hasAccepts = false;

    for (int i = 0; i < 3; i++) {
        if (e.restrictions.acceptSizesMin[i] >= 0) {
            hasAccepts = true;
            accept &= (mnk[i] >= e.restrictions.acceptSizesMin[i]);
        }
        if (e.restrictions.acceptSizesMax[i] >= 0) {
            hasAccepts = true;
            accept &= (mnk[i] <= e.restrictions.acceptSizesMax[i]);
        }
    }

    return hasAccepts && accept;
}

DerivedEvaluateParams getDerivedParams(
        const kcatalog::Entry &e, const EvaluateParams &p) {
    DerivedEvaluateParams dp;
    static_cast<EvaluateParams &>(dp) = p;

    auto wgTileM = e.driverInfo.wgTile(LoopM);
    auto wgTileN = e.driverInfo.wgTile(LoopN);
    dp.wgCountM = divUp(p.sizes.m, wgTileM);
    dp.wgCountN = divUp(p.sizes.n, wgTileN);
    dp.wgCountK = 1; /* may be adjusted later */
    dp.mPad = dp.wgCountM * wgTileM;
    dp.nPad = dp.wgCountN * wgTileN;
    dp.threadCount = double(dp.wgCountM * e.driverInfo.wg[LoopM])
            * double(dp.wgCountN * e.driverInfo.wg[LoopN])
            * double(dp.wgCountK * e.driverInfo.wg[LoopK]) * p.sizes.batch;

    switch (e.selector.hw) {
        case kcatalog::HWTagGen9:
        case kcatalog::HWTagGen11:
        case kcatalog::HWTagGen12LP: dp.threadsPerEU = 7; break;
        default: dp.threadsPerEU = (e.driverInfo.grfCount > 128) ? 4 : 8; break;
    }

    int ssCount;
    switch (e.selector.hw) {
        case kcatalog::HWTagGen12LP:
        case kcatalog::HWTagXeHP:
        case kcatalog::HWTagXeHPG: ssCount = p.euCount >> 4; break;
        default: ssCount = p.euCount >> 3; break;
    }

    dp.hwThreadCapacity = dp.threadsPerEU * p.euCount;
    dp.hwMinThreadsToFill = e.driverInfo.wg[LoopM] * e.driverInfo.wg[LoopN]
            * e.driverInfo.wg[LoopK] * ssCount;
    dp.partialWaveCount = divUp(dp.hwThreadCapacity, dp.hwMinThreadsToFill);

    return dp;
}

double evaluate(const kcatalog::Entry &e, const EvaluateParams &p,
        EvaluateAuxOutput &aux) {
    return evaluate(e, getDerivedParams(e, p), aux);
}

double evaluate(const kcatalog::Entry &e, const DerivedEvaluateParams &dp,
        EvaluateAuxOutput &aux) {
    if (alwaysAccept(e, dp)) return -std::numeric_limits<double>::infinity();

    switch (e.model.id) {
        case 'S': return evaluateS(e, dp, aux);
        case 'W': return evaluateW(e, dp, aux);
        default: return std::numeric_limits<double>::quiet_NaN();
    }
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
