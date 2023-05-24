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

    if (e.driverInfo.kParallelLocal())
        aux.k0 = alignUp(divUp(dp.sizes.k, e.driverInfo.wg[LoopK]),
                e.driverInfo.unroll[LoopK]);

    if (priority > maxPriority) /* no op */
        ; // Don't adjust very high values -- these are last resort kernels (lowest priority)
    else if (e.driverInfo.kParallel()) {
        int wgCountK = std::max(1, int(dp.hwThreadCapacity / dp.threadCount));
        aux.k0 = alignUp(
                divUp(dp.sizes.k, wgCountK), e.driverInfo.unroll[LoopK]);
        if (aux.k0 < dp.sizes.k)
            priority = -priority;
        else
            priority = 2 * maxPriority + priority;
    } else if (dp.threadCount > dp.hwThreadCapacity)
        priority = 2 * maxPriority - priority;

    // Prefer pure f16 over mixed f16-f32 kernels.
    if (e.selector.precisions[2][0] != e.selector.precisions[0][0])
        priority += 4 * maxPriority;

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
    auto kthread = k;
    auto capacity = dp.hwThreadCapacity;
    auto capacity1 = dp.hwMinThreadsToFill;

    if (e.driverInfo.kParallel())
        kthread = aux.k0;
    else if (e.driverInfo.kParallelLocal()) {
        kthread = alignUp(
                divUp(k, e.driverInfo.wg[LoopK]), e.driverInfo.unroll[LoopK]);
        kthread = std::max<decltype(kthread)>(
                kthread, 2 * e.driverInfo.unroll[LoopK]);
        aux.k0 = kthread;
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
    mtime *= k;
    mtime *= batch;

    double Ef = PARAM(Ef);
    double etimeFull = Ef * threadsFull;
    double Ep = std::max(Ef,
            PARAM(Ep0)
                    + (PARAM(Ep1) * dp.partialWaveCount)
                            / std::max(partialWaves, 1.));
    double etimePartial = Ep * partialWaves * capacity1;
    double etimeLB = (etimeFull + etimePartial);
    double etimeNoLB = Ef * threads;

    double Em = PARAM(Em);
    if (threads < capacity) Em = 1.;
    double etime = (1 - Em) * etimeNoLB + Em * etimeLB;
    etime *= (e.driverInfo.unroll[LoopM] * e.driverInfo.unroll[LoopN]);
    etime *= kthread;

    if (!dp.effective) {
        double F = PARAM(Fr0) + double(m) * double(n) * double(k) * PARAM(Fr1);
        F = std::max(1.0, std::min(PARAM(Fp), F));
        etime *= F;
    }

    double time = ctime + std::max(mtime, etime);

    return time;
#undef PARAM
}

double evaluateS(const kcatalog::Entry &e, const DerivedEvaluateParams &dp,
        EvaluateAuxOutput &aux) {
    if (!e.driverInfo.kParallel())
        return evaluateSCore(e, dp, aux);
    else {
        // Consider choosing k0 to get as close as possible to 1 or 2 full waves.
        int wgCountK1 = std::max(1, int(dp.hwThreadCapacity / dp.threadCount));
        int wgCountK2
                = std::max(1, int(2 * dp.hwThreadCapacity / dp.threadCount));

        int k0_1
                = alignUp(divUp(dp.sizes.k, wgCountK1 * e.driverInfo.wg[LoopK]),
                        e.driverInfo.unroll[LoopK]);
        int k0_2
                = alignUp(divUp(dp.sizes.k, wgCountK2 * e.driverInfo.wg[LoopK]),
                        e.driverInfo.unroll[LoopK]);

        k0_1 = std::max(k0_1, 1);
        k0_2 = std::max(k0_2, 1);

        wgCountK1 = std::max<int>(
                1, divUp(dp.sizes.k, k0_1 * e.driverInfo.wg[LoopK]));
        wgCountK2 = std::max<int>(
                1, divUp(dp.sizes.k, k0_2 * e.driverInfo.wg[LoopK]));

        auto dp1 = dp;
        dp1.wgCountK = wgCountK1;
        dp1.threadCount *= wgCountK1;
        aux.k0 = k0_1;

        double score = evaluateSCore(e, dp1, aux);

        if (k0_2 != k0_1) {
            auto dp2 = dp;
            dp2.wgCountK = wgCountK2;
            dp2.threadCount *= wgCountK2;

            EvaluateAuxOutput aux2;
            aux2.k0 = k0_2;

            double score2 = evaluateSCore(e, dp2, aux2);
            if (score2 < score) {
                score = score2;
                aux = aux2;
            }
        }

        // Add cost of initial beta scaling if not 1.
        if (dp.beta != 1.) {
            auto dp0 = dp;
            dp0.sizes.k = 0;

            EvaluateAuxOutput aux0;
            aux0.k0 = 1;

            score += evaluateSCore(e, dp0, aux0);
        }

        return score;
    }
}

double evaluateECore(const kcatalog::Entry &e, const DerivedEvaluateParams &dp,
        EvaluateAuxOutput &aux) {
#define PARAM(p) e.model.params[kcatalog::ParamE_##p]

    auto threads = dp.threadCount;
    auto batch = dp.sizes.batch;
    auto m = dp.sizes.m;
    auto n = dp.sizes.n;
    auto k = dp.sizes.k;
    auto kthread = k;
    auto capacity = dp.hwThreadCapacity;
    auto capacity1 = dp.hwMinThreadsToFill;

    if (e.driverInfo.kParallel())
        kthread = aux.k0;
    else if (e.driverInfo.kParallelLocal()) {
        kthread = alignUp(
                divUp(k, e.driverInfo.wg[LoopK]), e.driverInfo.unroll[LoopK]);
        kthread = std::max<decltype(kthread)>(
                kthread, 2 * e.driverInfo.unroll[LoopK]);
        aux.k0 = kthread;
    }

    double threadsFull = std::floor(threads / capacity) * capacity;
    double threadsPartial = threads - threadsFull;
    double partialWaves = std::ceil(threadsPartial / capacity1);
    double npartial = std::ceil(threads / capacity1);

    double Mc = (dp.beta == 0. || dp.autoatomic) ? PARAM(Mc) : PARAM(Mcu);
    double C0 = PARAM(C0), C1 = PARAM(C1);
    if (e.driverInfo.kParallelLocal()) {
        int wgK = std::max(
                1, std::min(e.driverInfo.wg[LoopK], int(divUp(k, aux.k0))));
        C0 += wgK * PARAM(Ck0);
        C1 += wgK * PARAM(Ck1);
    }
    if (e.driverInfo.fusedBeta() && (dp.beta != 1.) && aux.k0 > 0) {
        if (!e.driverInfo.altFusedBeta()) {
            C0 = PARAM(Cb0);
            C1 = PARAM(Cb1);
        }
        Mc += PARAM(Mc);
    }
    if (e.driverInfo.fusedPostOps() && (dp.postOps || dp.cConvert)
            && aux.k0 > 0)
        Mc += PARAM(Mcu);
    double ctime = C0 + npartial * C1 + Mc * m * n * batch;

    double mtime = PARAM(Ma) * m + PARAM(Mb) * n;
    mtime *= k;
    mtime *= batch;

    double Ef = PARAM(Ef);
    double etimeFull = Ef * threadsFull;
    double Ep = std::max(Ef,
            PARAM(Ep0)
                    + (PARAM(Ep1) * dp.partialWaveCount)
                            / std::max(partialWaves, 1.));
    double etimePartial = Ep * partialWaves * capacity1;
    double etimeLB = (etimeFull + etimePartial);
    double etimeNoLB = Ef * threads;

    double Em = PARAM(Em);
    if (threads < capacity) Em = 1.;
    if (e.driverInfo.kParallelVariable() && aux.k0 > 0) Em = 0.;
    double etime = (1 - Em) * etimeNoLB + Em * etimeLB;
    etime *= (e.driverInfo.unroll[LoopM] * e.driverInfo.unroll[LoopN]);
    etime *= kthread;

    if (!dp.effective) {
        double F = PARAM(Fr0) + double(m) * double(n) * double(k) * PARAM(Fr1);
        F = std::max(1.0, std::min(PARAM(Fp), F));
        etime *= F;
    }

    double time = ctime + std::max(mtime, etime);

    return time;
#undef PARAM
}

double evaluateE(const kcatalog::Entry &e, const DerivedEvaluateParams &dp,
        EvaluateAuxOutput &aux) {
    if (e.driverInfo.kParallel()) {
        // Consider choosing k0 to get as close as possible to 1 or 2 full waves.
        int wgCountK1 = std::max(1, int(dp.hwThreadCapacity / dp.threadCount));
        int wgCountK2
                = std::max(1, int(2 * dp.hwThreadCapacity / dp.threadCount));

        int k0_1
                = alignUp(divUp(dp.sizes.k, wgCountK1 * e.driverInfo.wg[LoopK]),
                        e.driverInfo.unroll[LoopK]);
        int k0_2
                = alignUp(divUp(dp.sizes.k, wgCountK2 * e.driverInfo.wg[LoopK]),
                        e.driverInfo.unroll[LoopK]);

        k0_1 = std::max(k0_1, 1);
        k0_2 = std::max(k0_2, 1);

        wgCountK1 = std::max<int>(
                1, divUp(dp.sizes.k, k0_1 * e.driverInfo.wg[LoopK]));
        wgCountK2 = std::max<int>(
                1, divUp(dp.sizes.k, k0_2 * e.driverInfo.wg[LoopK]));

        auto dp1 = dp;
        dp1.wgCountK = wgCountK1;
        dp1.threadCount *= wgCountK1;
        aux.k0 = k0_1;

        double score = evaluateECore(e, dp1, aux);

        if (k0_2 != k0_1) {
            auto dp2 = dp;
            dp2.wgCountK = wgCountK2;
            dp2.threadCount *= wgCountK2;

            EvaluateAuxOutput aux2;
            aux2.k0 = k0_2;

            double score2 = evaluateECore(e, dp2, aux2);
            if (score2 < score) {
                score = score2;
                aux = aux2;
            }
        }

        // Add cost of initial beta scaling if not 1.
        if (!e.driverInfo.fusedBeta() && dp.beta != 1.) {
            auto dp0 = dp;
            dp0.sizes.k = 0;

            EvaluateAuxOutput aux0;
            aux0.k0 = 1;

            score += evaluateECore(e, dp0, aux0);
        }

        return score;
    } else if (e.driverInfo.kParallelVariable()) {
        aux.k0 = 0;
        double score = evaluateECore(e, dp, aux);

        if (dp.sizes.batch > 1) return score;

        int64_t tcount = dp.threadCount;
        if ((tcount != dp.threadCount) || (tcount % dp.hwThreadCapacity != 0)) {
            aux.k0 = 1;
            double scoreKV = evaluateECore(e, dp, aux);
            if (scoreKV < score)
                score = scoreKV;
            else
                aux.k0 = 0;
        }

        return score;
    } else {
        // No global k-parallelization.
        return evaluateECore(e, dp, aux);
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

    auto unrollM = e.driverInfo.unroll[LoopM];
    auto unrollN = e.driverInfo.unroll[LoopN];

    auto wgM = e.driverInfo.wg[LoopM];
    auto wgN = e.driverInfo.wg[LoopN];

    auto wgTileM = wgM * unrollM;
    auto wgTileN = wgN * unrollN;

    dp.wgCountM = divUp(p.sizes.m, wgTileM); /* may be adjusted later */
    dp.wgCountN = divUp(p.sizes.n, wgTileN);
    dp.wgCountK = 1;

    if (!e.driverInfo.fixedWG()) {
        if (p.sizes.m < wgTileM) {
            wgM = std::max<int>(divUp(p.sizes.m, unrollM), 1);
            wgTileM = wgM * unrollM;
            dp.wgCountM = 1;
        }
        if (p.sizes.n < wgTileN) {
            wgN = std::max<int>(divUp(p.sizes.n, unrollN), 1);
            wgTileN = wgN * unrollN;
            dp.wgCountN = 1;
        }
    }

    auto threadsPerWG = wgM * wgN * e.driverInfo.wg[LoopK];

    dp.mPad = dp.wgCountM * wgTileM;
    dp.nPad = dp.wgCountN * wgTileN;
    dp.threadCount = double(dp.wgCountM) * double(dp.wgCountN);

    dp.threadCount *= threadsPerWG;
    dp.threadCount *= (dp.wgCountK * p.sizes.batch);

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
    dp.hwMinThreadsToFill = threadsPerWG * ssCount;
    dp.partialWaveCount = divUp(dp.hwThreadCapacity, dp.hwMinThreadsToFill);

    dp.autoatomic = (dp.beta == 1) && !p.cConvert && !p.postOps
            && e.driverInfo.mayUseAutoAtomic();

    return dp;
}

double evaluate(const kcatalog::Entry &e, const EvaluateParams &p,
        EvaluateAuxOutput &aux) {
    return evaluate(e, getDerivedParams(e, p), aux);
}

double evaluate(const kcatalog::Entry &e, const DerivedEvaluateParams &dp,
        EvaluateAuxOutput &aux) {
    double score = 0.;

    switch (e.model.id) {
        case 'E': score = evaluateE(e, dp, aux); break;
        case 'S': score = evaluateS(e, dp, aux); break;
        case 'W': score = evaluateW(e, dp, aux); break;
        default: score = std::numeric_limits<double>::quiet_NaN(); break;
    }

    if (alwaysAccept(e, dp)) score = -std::numeric_limits<double>::infinity();

    return score;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
