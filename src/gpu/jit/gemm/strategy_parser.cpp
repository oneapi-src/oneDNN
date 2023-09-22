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

#include "strategy_parser.hpp"
#include "utils.hpp"

#include <cctype>

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using namespace ngen;

bool native64Bit(ngen::HW hw) {
    EmulationStrategy emulate(hw);
    return !emulate.emulate64;
}

AccessType getAccessType(char c) {
    switch (std::tolower(c)) {
        case 'b': return AccessType::Block;
        case 'p': return AccessType::PseudoBlock;
        case 's': return AccessType::Scattered;
        case 'u': return AccessType::ChannelScattered;
        case 'm': return AccessType::Block2D;
        case 't': return AccessType::Block2DTranspose;
        case 'v': return AccessType::Block2DVNNI;
        default: throw std::runtime_error("Unknown access type.");
    }
}

char downgradeBlock2D(char c) {
    switch (std::tolower(c)) {
        case 'm':
        case 'v': return 'b';
        case 't': return 's';
        default: return c;
    }
}

AddressBase getAddressBase(char c) {
    switch (c) {
        case 'a': return AddressBase::createA64(true);
        case 'c': return AddressBase::createCC(0);
        case 'm': return AddressBase::createSC(0);
        case 's': return AddressBase::createBTS(0);
        default: throw std::runtime_error("Unknown address space.");
    }
}

CacheSettingsLSC getCaching(char l1, char l3) {
    if (l1 == 'd' && l3 == 'd') return CacheSettingsLSC::Default;

    bool l3cached = (l3 == 'c') || (l3 == 'b');
    switch (l1) {
        case 'u':
            return l3cached ? CacheSettingsLSC::L1UC_L3C
                            : CacheSettingsLSC::L1UC_L3UC;
        case 't':
        case 'c':
            return l3cached ? CacheSettingsLSC::L1C_L3C
                            : CacheSettingsLSC::L1C_L3UC;
        case 's':
            return l3cached ? CacheSettingsLSC::L1S_L3C
                            : CacheSettingsLSC::L1S_L3UC;
        case 'b':
        case 'i': return CacheSettingsLSC::L1IAR_L3C; break;
        default: throw std::runtime_error("Unknown cache setting");
    }
}

void getCaching(
        std::stringstream &s, HW hw, MatrixAddressingStrategy &astrategy) {
    auto &cachingR = astrategy.cachingR;
    auto &cachingW = astrategy.cachingW;

    cachingR = CacheSettingsLSC::L1C_L3C;
    cachingW = (hw >= HW::XeHPC) ? CacheSettingsLSC::L1UC_L3WB
                                 : CacheSettingsLSC::L1WB_L3WB;

    if (s.peek() == '{') {
        char eat, l1, l3;
        s >> eat >> l1 >> l3 >> eat;
        if (eat != '}' && eat != '/')
            throw std::runtime_error("Invalid caching syntax");
        cachingR = getCaching(l1, l3);
        if (eat == '/') {
            s >> l1 >> l3 >> eat;
            if (eat != '}') throw std::runtime_error("Invalid caching syntax");
            cachingW = getCaching(l1, l3);
        }
    }
}

static void getTiling(std::stringstream &s, bool doR, bool doC,
        MatrixAddressingStrategy &astrategy) {
    if (s.peek() == '#') {
        char eat = 0;
        int in = 0;
        if (doR) s >> eat >> in, astrategy.tileR = in;
        if (doC) s >> eat >> in, astrategy.tileC = in;
    }
}

void parseStrategy(const char *str, HW hw, const GEMMProblem &problem,
        GEMMStrategy &strategy) {
    std::stringstream s(str);
    bool overrideFusedLoop = false;
    bool gotSR = false;

    char eat, asA, asB, asC, accessA, accessB, accessC;
    char accessAUnaligned = '\0', accessBUnaligned = '\0';
    char accessAPrefetch = 's', accessBPrefetch = 's', accessCPrefetch = 's';

    s >> std::ws >> asA >> accessA;
    if (s.peek() == '/') s >> eat >> accessAUnaligned;
    s >> strategy.ka_load;
    if (s.peek() == '/') s >> eat >> strategy.ka_load_masked;
    getTiling(s, true, false, strategy.A);
    if (s.peek() == 'x') s >> eat >> strategy.A_copies;
    getCaching(s, hw, strategy.A);
    if (s.peek() == '+') {
        strategy.prefetchA = 1;
        s >> eat >> accessAPrefetch >> strategy.ka_prefetch;
        if (s.peek() == ',') s >> eat >> strategy.ka_pfStride;
        if (s.peek() == '@') s >> eat >> strategy.prefetchA;
        if (s.peek() == '/')
            s >> eat >> strategy.prefetchAMasked;
        else
            strategy.prefetchAMasked = strategy.prefetchA;
        getCaching(s, hw, strategy.A_prefetch);
    }
    s >> std::ws >> asB >> accessB;
    if (s.peek() == '/') s >> eat >> accessBUnaligned;
    s >> strategy.kb_load;
    if (s.peek() == '/') s >> eat >> strategy.kb_load_masked;
    getTiling(s, false, true, strategy.B);
    if (s.peek() == 'x') s >> eat >> strategy.B_copies;
    getCaching(s, hw, strategy.B);
    if (s.peek() == '+') {
        strategy.prefetchB = 1;
        s >> eat >> accessBPrefetch >> strategy.kb_prefetch;
        if (s.peek() == ',') s >> eat >> strategy.kb_pfStride;
        if (s.peek() == '@') s >> eat >> strategy.prefetchB;
        if (s.peek() == '/')
            s >> eat >> strategy.prefetchBMasked;
        else
            strategy.prefetchBMasked = strategy.prefetchB;
        getCaching(s, hw, strategy.B_prefetch);
    }
    s >> std::ws >> asC >> accessC;
    getTiling(s, true, true, strategy.C);
    getCaching(s, hw, strategy.C);
    if (s.peek() == '+') {
        strategy.prefetchC = 1;
        s >> eat >> accessCPrefetch;
        if (s.peek() == '@') s >> eat >> strategy.prefetchC;
        getCaching(s, hw, strategy.C_prefetch);
    }

    if (!accessAUnaligned) accessAUnaligned = downgradeBlock2D(accessA);
    if (!accessBUnaligned) accessBUnaligned = downgradeBlock2D(accessB);

    strategy.A.base = strategy.A_prefetch.base = getAddressBase(asA);
    strategy.B.base = strategy.B_prefetch.base = getAddressBase(asB);
    strategy.C.base = strategy.C_prefetch.base = getAddressBase(asC);
    strategy.CO.base = (hw >= HW::XeHPC) ? AddressBase::createA64(true)
                                         : AddressBase::createBTS(0);
    strategy.A.newDP = bool(std::isupper(accessA));
    strategy.B.newDP = bool(std::isupper(accessB));
    strategy.C.newDP = bool(std::isupper(accessC));
    strategy.CO.newDP = strategy.C.newDP;
    strategy.A.accessType = getAccessType(accessA);
    strategy.B.accessType = getAccessType(accessB);
    strategy.C.accessType = getAccessType(accessC);
    strategy.unalignedAccA = getAccessType(accessAUnaligned);
    strategy.unalignedAccB = getAccessType(accessBUnaligned);
    strategy.A.cachingW = CacheSettingsLSC::Default;
    strategy.B.cachingW = CacheSettingsLSC::Default;
    strategy.A_prefetch.prefetch = true;
    strategy.B_prefetch.prefetch = true;
    strategy.C_prefetch.prefetch = true;
    strategy.A_prefetch.newDP = bool(std::isupper(accessAPrefetch));
    strategy.B_prefetch.newDP = bool(std::isupper(accessBPrefetch));
    strategy.C_prefetch.newDP = bool(std::isupper(accessCPrefetch));
    strategy.A_prefetch.accessType = getAccessType(accessAPrefetch);
    strategy.B_prefetch.accessType = getAccessType(accessBPrefetch);
    strategy.C_prefetch.accessType = getAccessType(accessCPrefetch);
    strategy.A_prefetch.cachingW = CacheSettingsLSC::Default;
    strategy.B_prefetch.cachingW = CacheSettingsLSC::Default;
    strategy.C_prefetch.cachingW = CacheSettingsLSC::Default;

    strategy.A.padded |= isPacked(problem.A.layout);
    strategy.B.padded |= isPacked(problem.B.layout);
    strategy.A_prefetch.padded |= isPacked(problem.A.layout);
    strategy.B_prefetch.padded |= isPacked(problem.B.layout);

    strategy.unroll[LoopK] = 1;
    strategy.checkAdd32 = !native64Bit(hw) || (hw >= HW::XeHPC);
    strategy.altCRemainder |= (strategy.C.accessType == AccessType::Block)
            || strategy.kParallel;

    while (!s.eof()) {
        std::string mod;
        s >> mod;
        if (mod == "cs")
            strategy.registerScheme = GEMMStrategy::CSeparate;
        else if (mod == "acb")
            strategy.registerScheme = GEMMStrategy::ACB;
        else if (mod == "bca")
            strategy.registerScheme = GEMMStrategy::BCA;
        else if (mod == "vnc")
            strategy.registerScheme = GEMMStrategy::VNC;
        else if (mod == "int")
            strategy.registerScheme = GEMMStrategy::ABInterleave;
        else if (mod == "nse")
            strategy.registerScheme = GEMMStrategy::NSeparate;
        else if (mod == "vav")
            strategy.registerScheme = GEMMStrategy::VAvoid;
        else if (mod.substr(0, 3) == "grf") {
            mod.erase(0, 3);
            strategy.GRFs = std::stoi(mod);
        } else if (mod == "sys")
            strategy.systolic = true;
        else if (mod == "dw")
            strategy.dpasw = true;
        else if (mod == "fs")
            strategy.fixedSystolic = strategy.systolic = true;
        else if (mod == "ar")
            strategy.altCRemainder = true;
        else if (mod == "sr") {
            strategy.altCRemainder = false;
            gotSR = true;
        } else if (mod == "br")
            strategy.block2DCRemainder = true;
        else if (mod == "bf")
            strategy.block2DCFull = strategy.block2DCRemainder = true;
        else if (mod == "ac")
            strategy.cAccumulators = true;
        else if (mod == "el")
            strategy.cLoadAhead = true;
        else if (mod == "di")
            strategy.delayABInc = true;
        else if (mod == "ba")
            strategy.loadBFirst = true;
        else if (mod == "dm")
            strategy.doubleMasking = true;
        else if (mod == "kd")
            strategy.kDescRem = true;
        else if (mod == "sc")
            strategy.splitCopy = true;
        else if (mod == "sm")
            strategy.coopA = CoopSplit::MN;
        else if (mod == "sn")
            strategy.coopB = CoopSplit::MN;
        else if (mod == "ni")
            strategy.slmUseIncrCopy = false;
        else if (mod == "ek")
            strategy.slmEarlyKMask = true;
        else if (mod == "sf")
            strategy.strictFence = true;
        else if (mod == "ta")
            strategy.slmATrans = true;
        else if (mod == "tb")
            strategy.slmBTrans = true;
        else if (mod == "af")
            strategy.atomicFMA = true;
        else if (mod == "xaf")
            strategy.atomicFMA = strategy.extendedAtomicFMA = true;
        else if (mod == "st")
            strategy.stallAfterLoad = true;
        else if (mod == "ch")
            strategy.checkAdd32 = true;
        else if (mod == "ws")
            strategy.wgInSS = true;
        else if (mod == "wc")
            strategy.C.smode = ScatterSIMD::Wide;
        else if (mod == "cc")
            strategy.forceCopyC = true;
        else if (mod == "njs")
            strategy.jointSplit = false;
        else if (mod == "np") {
            strategy.A.padded = strategy.A_prefetch.padded = false;
            strategy.B.padded = strategy.B_prefetch.padded = false;
        } else if (mod == "pab") {
            strategy.A.padded = strategy.A_prefetch.padded = true;
            strategy.B.padded = strategy.B_prefetch.padded = true;
        } else if (mod == "pc")
            strategy.C.padded = strategy.C_prefetch.padded = true;
        else if (mod == "up")
            strategy.panelCheck = true;
        else if (mod == "mnk") {
            strategy.loopOrder[0] = LoopM;
            strategy.loopOrder[1] = LoopN;
            strategy.loopOrder[2] = LoopK;
        } else if (mod == "nmk") {
            strategy.loopOrder[0] = LoopN;
            strategy.loopOrder[1] = LoopM;
            strategy.loopOrder[2] = LoopK;
        } else if (mod == "fm") {
            strategy.fusedLoop = LoopM;
            overrideFusedLoop = true;
        } else if (mod == "fn") {
            strategy.fusedLoop = LoopN;
            overrideFusedLoop = true;
        } else if (mod == "rm")
            strategy.reverse[LoopM] = true;
        else if (mod == "rn")
            strategy.reverse[LoopN] = true;
        else if (mod == "kb" || mod == "kv") {
            if (mod == "kb") strategy.kParallel = true;
            if (mod == "kv") {
                strategy.kParallelVariable = true;
                strategy.fuseBeta = true;
                strategy.fusePostOps = true;
            }
            strategy.C.atomic = true;
            strategy.CO.atomic = problem.sumA || problem.sumB;
            if (strategy.CO.atomic)
                strategy.CO.base = AddressBase::createA64(true);
        } else if (mod == "kr")
            strategy.kParallelLocal = true;
        else if (mod == "akr")
            strategy.kParallelLocal = strategy.shrinkWGK = true;
        else if (mod == "fb")
            strategy.fuseBeta = true;
        else if (mod == "fp")
            strategy.fusePostOps = true;
        else if (mod == "afb")
            strategy.fuseBeta = strategy.altFusedBeta = true;
        else if (mod == "au")
            strategy.C.atomic = strategy.CO.atomic = true;
        else if (mod == "nau")
            strategy.C.atomic = strategy.CO.atomic = strategy.autoatomic
                    = false;
        else if (mod == "ff")
            strategy.forceWGUpdate = WGFixed;
        else if (mod == "wg") {
            char x;
            s >> strategy.wg[LoopM];
            s >> x;
            s >> strategy.wg[LoopN];
            strategy.wg[LoopK] = 0;
            if (s.peek() == 'x') s >> x >> strategy.wg[LoopK];
        } else if (mod == "nb") {
            char x;
            s >> strategy.namedBarriers[LoopM];
            s >> std::ws >> x;
            s >> strategy.namedBarriers[LoopN];
        } else if (mod == "bo")
            strategy.cWalkOrder = WalkOrder::Boustrophedon;
        else if (mod == "hi")
            strategy.cWalkOrder = WalkOrder::Hilbertlike;
        else if (mod == "li")
            strategy.cWalkOrder = WalkOrder::SimpleLinear;
        else if (mod == "pt")
            strategy.persistent = true;
        else if (mod == "of")
            strategy.arbitrationMode = ngen::ThreadArbitrationMode::OldestFirst;
        else if (mod == "rr")
            strategy.arbitrationMode = ngen::ThreadArbitrationMode::RoundRobin;
        else if (mod == "rrs")
            strategy.arbitrationMode
                    = ngen::ThreadArbitrationMode::RoundRobinOnStall;
        else if (mod == "nq") {
            strategy.A.noExtraPad = strategy.A_prefetch.noExtraPad = true;
            strategy.B.noExtraPad = strategy.B_prefetch.noExtraPad = true;
            strategy.C.noExtraPad = strategy.C_prefetch.noExtraPad = true;
            strategy.CO.noExtraPad = true;
        } else if (mod.length() >= 2) {
            if (mod.substr(0, 2) == "ms")
                strategy.mSplitThresh = stoi(mod.substr(2));
            else if (mod.substr(0, 2) == "ns")
                strategy.nSplitThresh = stoi(mod.substr(2));
            else if (mod.substr(0, 2) == "kc")
                strategy.kChain = stoi(mod.substr(2));
            else if (mod.substr(0, 2) == "ks") {
                char eat;
                std::stringstream ms(mod);
                ms >> eat >> eat >> strategy.unrollKSLM;
                if (!ms.eof() && (ms.peek() == '/'))
                    ms >> eat >> strategy.unrollKSLMMasked;
            } else if (mod.substr(0, 2) == "sb") {
                strategy.barrierFreq = stoi(mod.substr(2));
                strategy.splitBarrier = true;
            } else if (mod.substr(0, 2) == "pk")
                strategy.kPadding = stoi(mod.substr(2));
            else if (mod.substr(0, 2) == "ql") {
                strategy.skewLocalIDs = true;
            } else
                switch (mod[0]) {
                    case 'b':
                        if (isdigit(mod[1]))
                            strategy.barrierFreq = stoi(mod.substr(1));
                        else {
                            LoopType loop;
                            switch (mod[1]) {
                                case 'm': loop = LoopM; break;
                                case 'n': loop = LoopN; break;
                                case 'k': loop = LoopK; break;
                                default:
                                    throw std::runtime_error(
                                            "Unknown strategy modifier.");
                            }
                            size_t alt;
                            strategy.blocking[loop] = stoi(mod.substr(2), &alt);
                            if (strategy.blocking[loop] == 0)
                                strategy.blocking[loop] = 16777216;
                            alt += 3;
                            if (mod.length() > alt)
                                strategy.blockingAlt[loop]
                                        = stoi(mod.substr(alt));
                        }
                        break;
                    case 'c': {
                        mod.erase(0, 1);
                        if (mod[0] == 'a') {
                            mod.erase(0, 1);
                            strategy.slmA = true;
                        }
                        if (mod[0] == 'b') {
                            mod.erase(0, 1);
                            strategy.slmB = true;
                        }
                        std::stringstream ms(mod);
                        ms >> strategy.slmBuffers;
                        ms >> eat;
                        if (!ms.eof()) ms >> strategy.slmCopies;
                        break;
                    }
                    case 'k': {
                        char eat;
                        std::stringstream ms(mod);
                        ms >> eat >> strategy.unroll[LoopK];
                        if (!ms.eof() && (ms.peek() == '/'))
                            ms >> eat >> strategy.unrollK_masked;
                        strategy.extraKAlign = strategy.unroll[LoopK];
                        break;
                    }
                    case 'l': strategy.optAlignAB = stoi(mod.substr(1)); break;
                    default:
                        throw std::runtime_error("Unknown strategy modifier.");
                }
        } else if (!mod.empty())
            throw std::runtime_error("Unknown strategy modifier.");
    }

    if (!overrideFusedLoop) {
        if (strategy.fused) {
            if (strategy.wg[LoopM] == 1)
                strategy.fusedLoop = LoopN;
            else if (strategy.wg[LoopN] == 1)
                strategy.fusedLoop = LoopM;
            else
                strategy.fusedLoop = strategy.loopOrder[0];
        } else
            strategy.fusedLoop = strategy.loopOrder[0];
    }

    if (strategy.ka_pfStride == 0) strategy.ka_pfStride = strategy.ka_prefetch;
    if (strategy.kb_pfStride == 0) strategy.kb_pfStride = strategy.kb_prefetch;

    if (strategy.block2DCRemainder && !gotSR) strategy.altCRemainder = true;

    int poCount = problem.postOps.len();
    strategy.binary.resize(poCount);
    for (auto &astrategy : strategy.binary) {
        astrategy.base = (hw >= HW::XeHPC) ? AddressBase::createA64(true)
                                           : AddressBase::createBTS(0);
        astrategy.newDP = strategy.C.newDP;
    }
}

void adjustStrategy(HW hw, const GEMMProblem &problem, GEMMStrategy &strategy) {
    auto *gemmAStrategy = &strategy.A, *gemmBStrategy = &strategy.B;

    // 2D block accesses use 2D addressing where supported.
    strategy.A.address2D
            |= isBlock2D(strategy.A.accessType) && !isPacked(problem.A.layout);
    strategy.B.address2D
            |= isBlock2D(strategy.B.accessType) && !isPacked(problem.B.layout);
    strategy.C.address2D
            |= isBlock2D(strategy.C.accessType) && !isPacked(problem.C.layout);
    strategy.A_prefetch.address2D |= isBlock2D(strategy.A_prefetch.accessType)
            && !isPacked(problem.A.layout);
    strategy.B_prefetch.address2D |= isBlock2D(strategy.B_prefetch.accessType)
            && !isPacked(problem.B.layout);
    strategy.C_prefetch.address2D |= isBlock2D(strategy.C_prefetch.accessType)
            && !isPacked(problem.C.layout);

    // No need to use split remainder handling for 2D block accesses as there's no penalty for masking.
    if (isBlock2D(strategy.A.accessType)
            && (!strategy.prefetchA
                    || isBlock2D(strategy.A_prefetch.accessType)))
        strategy.remHandling[LoopM] = RemainderHandling::General;
    if (isBlock2D(strategy.B.accessType)
            && (!strategy.prefetchB
                    || isBlock2D(strategy.B_prefetch.accessType)))
        strategy.remHandling[LoopN] = RemainderHandling::General;

    // Also don't split remainder handling if padded.
    if (gemmAStrategy->padded)
        strategy.remHandling[LoopM] = RemainderHandling::General;
    if (gemmBStrategy->padded)
        strategy.remHandling[LoopN] = RemainderHandling::General;

    // But always use split remainder handling when prefetching C if it _isn't_ block 2D
    //  ... in that case there are no C prefetches on the remainder path.
    if (strategy.prefetchC && !isBlock2D(strategy.C_prefetch.accessType))
        strategy.remHandling[LoopM] = strategy.remHandling[LoopN]
                = RemainderHandling::Split;

    // ... and always use split remainder handling on later GPUs when panel checks are active.
    if (strategy.panelCheck && strategy.lateExit() && hw >= HW::XeHP
            && !strategy.fixedSystolic) {
        if (isPacked(problem.A.layout))
            strategy.remHandling[LoopM] = RemainderHandling::Split;
        if (isPacked(problem.B.layout))
            strategy.remHandling[LoopN] = RemainderHandling::Split;
    }

    // Finally, no remainder handling needed for GEMV-type kernels.
    if (strategy.unroll[LoopM] * strategy.wg[LoopM] == 1)
        strategy.remHandling[LoopM] = RemainderHandling::Ignore;
    if (strategy.unroll[LoopN] * strategy.wg[LoopN] == 1)
        strategy.remHandling[LoopN] = RemainderHandling::Ignore;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
