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

#include "kernel_selector.hpp"
#include "common/verbose.hpp"
#include "kernel_evaluator.hpp"

#include <cassert>
#include <cctype>
#include <cstring>

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

inline bool layoutMatch(const char *lref, const char *lpattern) {
    return (lref[0] == lpattern[0]); // This is a sufficient check for now.
}

inline bool precisionMatch(char pref, char ppattern) {
    // Fast case-insensitive compare
    return (pref & ~0x20) == (ppattern & ~0x20);
}

inline bool precisionMatch(const char *pref, const char *ppattern) {
    bool ok = false;
    ok = ok || (ppattern[0] == '?');
    ok = ok || precisionMatch(pref[0], ppattern[0]);
    ok = ok || (ppattern[0] == '[' && precisionMatch(pref[0], ppattern[1]));
    if (ok && pref[0] == '[') {
        ok = ok && precisionMatch(pref[1], ppattern[1])
                && precisionMatch(pref[2], ppattern[2]);
        for (int i = 3; pref[i] != '\0'; i++) {
            if (pref[i] != ppattern[i]) {
                ok = false;
                break;
            }
        }
    }
    return ok;
}

inline bool precisionMinimumMatch(const char *pref, char pmin) {
    uint8_t sizeTable[0x20]
            = {//         A  B  C  D  E  F  G  H  I  J  K  L  M  N  O
                    0, 0, 2, 8, 8, 0, 0, 0, 2, 4, 4, 0, 0, 0, 0, 1,
                    //      P  Q  R  S  T  U  V  W  X  Y  Z
                    0, 0, 0, 4, 4, 0, 0, 2, 0, 0, 16, 0, 0, 0, 0, 0};

    return (sizeTable[pref[0] & 0x1F] >= sizeTable[pmin & 0x1F]);
}

inline bool alignmentMatch(int aref, int apattern) {
    if (aref == 0) aref = 1;
    return (apattern % aref == 0);
}

inline bool tagMatch(const char *tref, const char *tpattern) {
    for (auto c = tref; *c; c++) {
        // Lowercase tags -> must not match pattern
        // Uppercase tags -> must match pattern
        int cu = *c & ~0x20; // tolower(c)
        bool match = (std::strchr(tpattern, cu) != nullptr);
        bool wantMatch = (*c & 0x20) == 0;
        if (match != wantMatch) return false;
    }
    return true;
}

bool matches(const kcatalog::Entry &e, const MatchParams &pattern) {
    bool ok = true;

    if (e.restrictions.steppingMin >= 0)
        ok = ok && (pattern.stepping >= e.restrictions.steppingMin);
    if (e.restrictions.steppingMax >= 0)
        ok = ok && (pattern.stepping < e.restrictions.steppingMax);
    ok = ok && layoutMatch(e.selector.layouts[0], pattern.selector.layouts[0]);
    ok = ok && layoutMatch(e.selector.layouts[1], pattern.selector.layouts[1]);
    ok = ok && layoutMatch(e.selector.layouts[2], pattern.selector.layouts[2]);
    ok = ok
            && precisionMatch(
                    e.selector.precisions[2], pattern.selector.precisions[2]);
    if (pattern.precisionCExt)
        ok = ok
                && precisionMinimumMatch(
                        e.selector.precisions[2], pattern.precisionCExt);
    for (int i = 0; i < 3; i++)
        ok = ok
                && alignmentMatch(
                        e.restrictions.alignment[i], pattern.alignment[i]);
    ok = ok && tagMatch(e.restrictions.tags, pattern.tags);

    for (int i = 0; i < 2; i++)
        if (pattern.unroll[i] > 0)
            ok = ok && (pattern.unroll[i] == e.driverInfo.unroll[i]);

    if (!pattern.ignoreSizes) {
        int64_t mnk[3] = {pattern.sizes.m, pattern.sizes.n, pattern.sizes.k};
        for (int i = 0; i < 3; i++) {
            if (e.restrictions.allowedSizesMin[i] >= 0)
                ok = ok && (mnk[i] >= e.restrictions.allowedSizesMin[i]);
            if (e.restrictions.allowedSizesMax[i] >= 0)
                ok = ok && (mnk[i] <= e.restrictions.allowedSizesMax[i]);
        }
    }

    // Should already be matched.
    ok = ok && (e.selector.hw == pattern.selector.hw);
    ok = ok
            && precisionMatch(
                    e.selector.precisions[0], pattern.selector.precisions[0]);
    ok = ok
            && precisionMatch(
                    e.selector.precisions[1], pattern.selector.precisions[1]);

    return ok;
}

bool lessAligned(int alignA1, int alignB1, int alignA2, int alignB2) {
    alignA1 = std::max(alignA1, 4);
    alignA2 = std::max(alignA2, 4);
    alignB1 = std::max(alignB1, 4);
    alignB2 = std::max(alignB2, 4);
    return (alignA1 <= alignA2) && (alignB1 <= alignB2)
            && (alignA1 + alignB1 < alignB1 + alignB2);
}

const kcatalog::Entry *select(const kcatalog::Catalog &catalog,
        const MatchParams &pattern, const EvaluateParams &eparams,
        EvaluateAuxOutput &aux) {
    return select(catalog, 1, &pattern, eparams, aux);
}

const kcatalog::Entry *select(const kcatalog::Catalog &catalog, int npatterns,
        const MatchParams *patterns, const EvaluateParams &eparams,
        EvaluateAuxOutput &aux) {
    double bestScore = std::numeric_limits<double>::infinity();
    const kcatalog::Entry *bestEntry = nullptr;
    int bestIPattern = -1;
    bool bestIsFallback = false;
    int bestAlignA = 0, bestAlignB = 0;

    bool verbose = (get_verbose(verbose_t::debuginfo) >= 5);

    // TODO: omit evaluation if only one match, if aux output not needed.
    for (int ipattern = 0; ipattern < npatterns; ipattern++) {
        for (auto it = match(catalog, patterns[ipattern]); it; it++) {
            EvaluateAuxOutput thisAux;

            bool fallback
                    = (it->restrictions.tags[0] == kcatalog::ReqAlignFallback);
            int alignA = std::max(it->restrictions.alignment[0], 4);
            int alignB = std::max(it->restrictions.alignment[1], 4);

            if (fallback && lessAligned(alignA, alignB, bestAlignA, bestAlignB))
                continue;

            double score = evaluate(*it, eparams, thisAux);

            bool better = (score < bestScore)
                    | (bestIsFallback
                            && lessAligned(
                                    bestAlignA, bestAlignB, alignA, alignB));

            if (better) {
                bestEntry = &*it;
                bestScore = score;
                bestIPattern = ipattern;
                bestAlignA = alignA;
                bestAlignB = alignB;
                bestIsFallback = fallback;
                aux = thisAux;
            }
            if (verbose) {
                const auto &info = it->driverInfo;
                printf("onednn_verbose,info,gpu,gemm,consider:%dx%d,%dx%dx%d,"
                       "score:%f\n",
                        info.unroll[LoopM], info.unroll[LoopN], info.wg[LoopM],
                        info.wg[LoopN], info.wg[LoopK], score);
            }
        }
    }

    // Late tag checking. If late tags do not match, we abandon the kernel and
    //  force the calling code to take another path.
    if (bestEntry
            && !tagMatch(bestEntry->restrictions.tags,
                    patterns[bestIPattern].lateTags))
        return nullptr;

    return bestEntry;
}

template <bool upper>
const kcatalog::Entry *upper_lower_bound(
        const kcatalog::Catalog &catalog, const kcatalog::Selector &selector) {
    int n = catalog.entryCount;
    const kcatalog::Entry *cur = catalog.entries;

    while (n > 0) {
        auto half = n >> 1;
        auto mid = cur + half;
        if (upper ? (*mid <= selector) : (*mid < selector)) {
            cur = mid + 1;
            n = n - half - 1;
        } else
            n = half;
    }

    return cur;
}

const kcatalog::Entry *lower_bound(
        const kcatalog::Catalog &catalog, const kcatalog::Selector &selector) {
    return upper_lower_bound<false>(catalog, selector);
}

const kcatalog::Entry *upper_bound(
        const kcatalog::Catalog &catalog, const kcatalog::Selector &selector) {
    return upper_lower_bound<true>(catalog, selector);
}

MatchParamsBase::MatchParamsBase(ngen::HW hw, const GEMMProblem &problem) {
    using namespace kcatalog;

    switch (hw) {
        default: assert(!"Unknown architecture");
        case ngen::HW::Gen9: selector.hw = kcatalog::HWTagGen9; break;
        case ngen::HW::Gen11: selector.hw = kcatalog::HWTagGen11; break;
        case ngen::HW::Gen12LP: selector.hw = kcatalog::HWTagGen12LP; break;
        case ngen::HW::XeHP: selector.hw = kcatalog::HWTagXeHP; break;
        case ngen::HW::XeHPG: selector.hw = kcatalog::HWTagXeHPG; break;
        case ngen::HW::XeHPC: selector.hw = kcatalog::HWTagXeHPC; break;
    }

    auto &C = problem.C;
    auto equivCLayout = C.layout;
    if (isPacked(equivCLayout)) {
        bool colMajor = (C.layout == MatrixLayout::Pc)
                ^ (C.crosspack * problem.Tc > 4);
        equivCLayout = (colMajor ? MatrixLayout::N : MatrixLayout::T);
    }

    selector.kernelType = "gemm";

    std::fill(temp.begin(), temp.end(), '\0');
    temp[0] = precisionChar(problem.Ta);
    temp[2] = precisionChar(problem.Tb);
    temp[4] = precisionChar(problem.Tc);
    temp[6] = layoutChar(problem.A.layout);
    temp[8] = layoutChar(problem.B.layout);
    temp[10] = layoutChar(equivCLayout);
    selector.precisions[0] = &temp[0];
    selector.precisions[1] = &temp[2];
    selector.precisions[2] = &temp[4];
    selector.layouts[0] = &temp[6];
    selector.layouts[1] = &temp[8];
    selector.layouts[2] = &temp[10];

    precisionCExt = precisionChar(problem.Tc_ext);

    alignment[0] = problem.A.alignment;
    alignment[1] = problem.B.alignment;
    alignment[2] = problem.C.alignment;

    char *tagPtr = &temp[12];
    lateTags = tagPtr;

    // Late-only tags. Don't choose lower-performing kernels
    //  just to fuse reductions. Instead do reductions in a separate kernel.
    if (problem.sumA) *tagPtr++ = ReqSumA;
    if (problem.sumB) *tagPtr++ = ReqSumB;

    tags = tagPtr;

    if (problem.batch != BatchMode::None) {
        *tagPtr++ = ReqBatch;
        if (problem.batchDims > 1) *tagPtr++ = ReqBatchMultiDim;
    }

    if (problem.abOffset != ABOffset::None) *tagPtr++ = ReqABOffset;

    sizes.batch = sizes.m = sizes.n = sizes.k = 0;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
