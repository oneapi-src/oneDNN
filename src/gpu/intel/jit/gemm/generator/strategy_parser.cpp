/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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
#include "kernel_catalog.hpp"
#include "internal/utils.hpp"

#include <cctype>
#include <sstream>

#include "internal/namespace_start.hxx"

using namespace ngen;

static void unparseAccessType(std::ostream &s, const MatrixAddressingStrategy &astrategy);
static void unparseAddressBase(std::ostream &s, ngen::AddressBase base);
static void unparseCaching(HW hw, std::ostream &s, const MatrixAddressingStrategy &astrategy);
static void unparseTiling(std::ostream &s, const MatrixAddressingStrategy &astrategy);

bool native64Bit(HW hw)
{
    EmulationStrategy emulate(hw);
    return !emulate.emulate64;
}

AccessType getAccessType(char c)
{
    switch (std::tolower(c)) {
        case 'b': return AccessType::Block;
        case 'p': return AccessType::PseudoBlock;
        case 's': return AccessType::Scattered;
        case 'u': return AccessType::ChannelScattered;
        case 'm': return AccessType::Block2D;
        case 't': return AccessType::Block2DTranspose;
        case 'v': return AccessType::Block2DVNNI;
        case 'c': return AccessType::CacheLine;
        default: stub("Unknown access type.");
    }
}

char downgradeBlock2D(char c)
{
    switch (std::tolower(c)) {
        case 'm':
        case 'v': return 'b';
        case 't': return 's';
        default:  return c;
    }
}

AddressBase getAddressBase(char c)
{
    switch (c) {
        case 'a': return AddressBase::createA64(true);
        case 'c': return AddressBase::createCC(0);
        case 'l': return AddressBase::createSLM();
        case 'm': return AddressBase::createSC(0);
        case 'r': return AddressBase{};
        case 's': return AddressBase::createBTS(0);
        default: stub("Unknown address space.");
    }
}

CacheSettingsLSC getCaching(char l1, char l3)
{
    if (l1 == 'd' && l3 == 'd')
        return CacheSettingsLSC::Default;

    bool l3cached = (l3 == 'c') || (l3 == 'b');
    switch (l1) {
        case 'u': return l3cached ? CacheSettingsLSC::L1UC_L3C : CacheSettingsLSC::L1UC_L3UC;
        case 't':
        case 'c': return l3cached ? CacheSettingsLSC::L1C_L3C  : CacheSettingsLSC::L1C_L3UC;
        case 's': return l3cached ? CacheSettingsLSC::L1S_L3C  : CacheSettingsLSC::L1S_L3UC;
        case 'b':
        case 'i': return CacheSettingsLSC::L1IAR_L3C; break;
        default: stub("Unknown cache setting");
    }
}

CacheSettingsLSC getCachingEntry(std::stringstream &s, HW hw)
{
    {
        char l1, l3;
        s >> l1 >> l3;
        return getCaching(l1, l3);
    }
}

void getCaching(std::stringstream &s, HW hw, MatrixAddressingStrategy &astrategy, bool leaveDefault = false)
{
    auto &cachingR = astrategy.cachingR;
    auto &cachingW = astrategy.cachingW;

    if (!leaveDefault) {
        cachingR = CacheSettingsLSC::L1C_L3C;
        cachingW = CacheSettingsLSC::L1WB_L3WB;
        if (hw >= HW::XeHPC)
            cachingW = CacheSettingsLSC::L1UC_L3WB;
    }

    if (s.peek() == '{') {
        char eat;
        s >> eat;
        cachingR = getCachingEntry(s, hw);
        s >> eat;
        if (eat == '/') {
            cachingW = getCachingEntry(s, hw);
            s >> eat;
        }
        if (eat != '}')
            stub("Invalid caching syntax");
    }
}

static void getTiling(std::stringstream &s, MatrixAddressingStrategy &astrategy)
{
    if (s.peek() == '#') {
        char eat = 0;
        int in = 0;
        s >> eat >> in, astrategy.tileR = in;
        s >> eat >> in, astrategy.tileC = in;
    }
}

void parseStrategy(const char *str, HW hw, const GEMMProblem &problem, GEMMStrategy &strategy)
{
    std::stringstream s(str);
    bool overrideFusedLoop = false;
    bool gotSR = false;

    char eat, asA, asB, asC, accessA, accessB, accessC;
    char accessAUnaligned = '\0', accessBUnaligned = '\0';
    char accessAPrefetch = 's', accessBPrefetch = 's', accessCPrefetch = 's';
    char accessABPrefetchL3 = 'b';

    auto A64 = AddressBase::createA64(true);
    auto BTS = AddressBase::createBTS(0);

    s >> std::ws >> asA >> accessA;
        if (s.peek() == '/') s >> eat >> accessAUnaligned;
        s >> strategy.ka_load;
        if (s.peek() == '/') s >> eat >> strategy.ka_load_masked;
        getTiling(s, strategy.A);
        if (s.peek() == 'x') s >> eat >> strategy.A_copies;
        getCaching(s, hw, strategy.A);
    if (s.peek() == '+') {
        s >> eat;
        if (s.peek() != '+') {
            strategy.prefetchA = 1;
            s >> accessAPrefetch >> strategy.ka_prefetch;
            if (s.peek() == ',') s >> eat >> strategy.ka_pfStride;
            if (s.peek() == '@') s >> eat >> strategy.prefetchA;
            if (s.peek() == '/') s >> eat >> strategy.prefetchAMasked;
            else strategy.prefetchAMasked = strategy.prefetchA;
            getCaching(s, hw, strategy.A_prefetch);
        }
    }
    if (s.peek() == '+') {
        strategy.l3PrefetchA = true;
        s >> eat >> accessABPrefetchL3 >> strategy.ka_prefetchL3;
        if (s.peek() == '@') s >> eat >> strategy.prefetchABL3;
        getCaching(s, hw, strategy.AB_prefetchL3, true);
    }
    s >> std::ws >> asB >> accessB;
        if (s.peek() == '/') s >> eat >> accessBUnaligned;
        s >> strategy.kb_load;
        if (s.peek() == '/') s >> eat >> strategy.kb_load_masked;
        getTiling(s, strategy.B);
        if (s.peek() == 'x') s >> eat >> strategy.B_copies;
        getCaching(s, hw, strategy.B);
    if (s.peek() == '+') {
        s >> eat;
        if (s.peek() != '+') {
            strategy.prefetchB = 1;
            s >> accessBPrefetch >> strategy.kb_prefetch;
            if (s.peek() == ',') s >> eat >> strategy.kb_pfStride;
            if (s.peek() == '@') s >> eat >> strategy.prefetchB;
            if (s.peek() == '/') s >> eat >> strategy.prefetchBMasked;
            else strategy.prefetchBMasked = strategy.prefetchB;
            getCaching(s, hw, strategy.B_prefetch);
        }
    }
    if (s.peek() == '+') {
        strategy.l3PrefetchB = true;
        s >> eat >> accessABPrefetchL3 >> strategy.kb_prefetchL3;
        if (s.peek() == '@') s >> eat >> strategy.prefetchABL3;
        getCaching(s, hw, strategy.AB_prefetchL3, true);
    }
    s >> std::ws >> asC >> accessC;
        getTiling(s, strategy.C);
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
    strategy.CO.base = (hw >= HW::XeHPG) ? A64 : BTS;
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
    strategy.CO.cachingR = CacheSettingsLSC::L1C_L3C;
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

    strategy.AB_prefetchL3.prefetch = true;
    strategy.AB_prefetchL3.newDP = true;
    strategy.AB_prefetchL3.padded = true;
    strategy.AB_prefetchL3.accessType = getAccessType(accessABPrefetchL3);
    strategy.AB_prefetchL3.base = getAddressBase(strategy.l3PrefetchA ? asA : asB);
    if (strategy.AB_prefetchL3.cachingR == CacheSettingsLSC::Default) {
        strategy.AB_prefetchL3.cachingR = CacheSettingsLSC::L1UC_L3C;
    }

    strategy.A.padded |= isPacked(problem.A.layout);
    strategy.B.padded |= isPacked(problem.B.layout);
    strategy.A_prefetch.padded |= isPacked(problem.A.layout);
    strategy.B_prefetch.padded |= isPacked(problem.B.layout);

    strategy.unroll[LoopK] = 1;
    strategy.checkAdd32 = !native64Bit(hw) || (hw == HW::XeHPC);
    strategy.altCRemainder |= (strategy.C.accessType == AccessType::Block) || strategy.kParallel;

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
            mod.erase(0,3);
            strategy.GRFs = std::stoi(mod);
        } else if (mod.substr(0, 3) == "dot") {
            mod.erase(0,3);
            strategy.dotVL = mod.empty() ? 1 : std::stoi(mod);
        } else if (mod == "sys")
            strategy.systolic = true;
        else if (mod == "dw")
            strategy.dpasw = true;
        else if (mod == "fs") {
            strategy.fixedSystolic = strategy.systolic = true;
            strategy.CO.base = BTS;
        } else if (mod == "ar")
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
        else if (mod == "ska")
            strategy.coopA = CoopSplit::FullK;
        else if (mod == "sn")
            strategy.coopB = CoopSplit::MN;
        else if (mod == "skb")
            strategy.coopB = CoopSplit::FullK;
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
        else if (mod == "fx")
            strategy.fmaBoustrophedon = true;
        else if (mod == "ch")
            strategy.checkAdd32 = true;
        else if (mod == "nch")
            strategy.checkAdd32 = false;
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
                strategy.CO.base = A64;
        } else if (mod == "kr")
            strategy.kParallelLocal = true;
        else if (mod == "akr")
            strategy.kParallelLocal = strategy.shrinkWGK = true;
        else if (mod == "ikr")
            strategy.kParallelLocal = strategy.kInterleave = true;
        else if (mod == "fb")
            strategy.fuseBeta = true;
        else if (mod == "afb")
            strategy.fuseBeta = strategy.altFusedBeta = true;
        else if (mod == "fp")
            strategy.fusePostOps = true;
        else if (mod == "zt")
            strategy.zeroTempC = true;
        else if (mod == "rx")
            strategy.relaxedAccumulation = true;
        else if (mod == "fg") {
            float fillGoal;
            s >> fillGoal;
            strategy.fillGoal = int(fillGoal * 16);
        } else if (mod == "au")
            strategy.C.atomic = strategy.CO.atomic = true;
        else if (mod == "nau")
            strategy.C.atomic = strategy.CO.atomic = strategy.autoatomic = false;
        else if (mod == "ff")
            strategy.forceWGUpdate = WGFixed;
        else if (mod == "wg") {
            char x;
            s >> strategy.wg[LoopM];
            s >> x;
            s >> strategy.wg[LoopN];
            strategy.wg[LoopK] = 0;
            if (s.peek() == 'x')
                s >> x >> strategy.wg[LoopK];
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
        else if (mod == "nl")
            strategy.cWalkOrder = WalkOrder::NestedLinear;
        else if (mod == "pt")
            strategy.persistent = true;
        else if (mod == "of")
            strategy.arbitrationMode = ThreadArbitrationMode::OldestFirst;
        else if (mod == "rr")
            strategy.arbitrationMode = ThreadArbitrationMode::RoundRobin;
        else if (mod == "rrs")
            strategy.arbitrationMode = ThreadArbitrationMode::RoundRobinOnStall;
        else if (mod == "l2d")
            strategy.optAlignAB2D = true;
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
            else if (mod.substr(0, 2) == "ki")
                strategy.kInterleaveChunk = stoi(mod.substr(2));
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
            else if (mod.substr(0, 2) == "cr") {
                strategy.cRepackPanel = stoi(mod.substr(2));
                if (strategy.cRepackPanel == 0)
                    strategy.cRepackPanel = 1024;   /* arbitrary large value */
            } else if (mod.substr(0, 2) == "rc") {
                strategy.repackC = stoi(mod.substr(2));
                if (strategy.cRepackPanel == 0)
                    strategy.cRepackPanel = 1024;   /* arbitrary large value */
            } else if (mod.substr(0, 2) == "wx") {
                strategy.wgPadFactor = stoi(mod.substr(2));
                strategy.forceWGUpdate = WGFixed;
            } else if (mod.substr(0, 2) == "ql") {
                strategy.skewLocalIDs = true;
            } else switch (mod[0]) {
                case 'b':
                    if (isdigit(mod[1]))
                        strategy.barrierFreq = stoi(mod.substr(1));
                    else {
                        LoopType loop;
                        switch (mod[1]) {
                            case 'm': loop = LoopM; break;
                            case 'n': loop = LoopN; break;
                            case 'k': loop = LoopK; break;
                            default: stub("Unknown strategy modifier.");
                        }
                        size_t alt;
                        strategy.blocking[loop] = stoi(mod.substr(2), &alt);
                        if (strategy.blocking[loop] == 0)
                            strategy.blocking[loop] = 16777216;
                        alt += 3;
                        if (mod.length() > alt)
                            strategy.blockingAlt[loop] = stoi(mod.substr(alt));
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
                    if (!ms.eof())
                        ms >> strategy.slmCopies;
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
                } case 'l':
                    strategy.optAlignAB = stoi(mod.substr(1));
                    break;
                default:
                    stub("Unknown strategy modifier.");
            }
        } else if (!mod.empty())
            stub("Unknown strategy modifier.");
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

    if (strategy.persistentLoop() && !strategy.linearOrder()) strategy.cWalkOrder = WalkOrder::SimpleLinear;

    // Use new LSC messages on Xe2+
    if (hw >= ngen::HW::Xe2) {
       strategy.A.newDP = strategy.A_prefetch.newDP = true;
       strategy.B.newDP = strategy.B_prefetch.newDP = true;
       strategy.C.newDP = strategy.C_prefetch.newDP = true;
    }

    size_t poCount = problem.postOps.len();
    strategy.binary.resize(poCount);
    for (auto &astrategy: strategy.binary) {
        astrategy.base = (hw >= HW::XeHPC) ? A64 : BTS;
        astrategy.newDP = strategy.C.newDP;
    }

    bool surfaceAq = (problem.quantized2DA() && !strategy.A.base.isStateless());
    bool surfaceBq = (problem.quantized2DB() && !strategy.B.base.isStateless());

    strategy.AO.base = strategy.A_scale.base = (surfaceAq ? BTS : A64);
    strategy.BO.base = strategy.B_scale.base = (surfaceBq ? BTS : A64);

    if (problem.aoPtrDims <= 2) strategy.AO.base = A64;
    if (problem.boPtrDims <= 2) strategy.BO.base = A64;

    strategy.AO.newDP = strategy.A_scale.newDP = strategy.A.newDP;
    strategy.BO.newDP = strategy.B_scale.newDP = strategy.B.newDP;
}

void adjustStrategy(HW hw, const GEMMProblem &problem, GEMMStrategy &strategy, const char *tags)
{
    auto *gemmAStrategy = &strategy.A, *gemmBStrategy = &strategy.B;
    if (problem.A.needA64) strategy.A.forceA64();
    if (problem.B.needA64) strategy.B.forceA64();
    if (problem.C.needA64) strategy.C.forceA64();

    // 2D block accesses use 2D addressing where supported.
    auto use2DAddressing = [](MatrixAddressingStrategy &astrategy) {
        astrategy.address2D |= isBlock2D(astrategy.accessType);
    };

    if (!isPacked(problem.A.layout)) {
        use2DAddressing(strategy.A);
        use2DAddressing(strategy.A_prefetch);
    }
    if (!isPacked(problem.B.layout)) {
        use2DAddressing(strategy.B);
        use2DAddressing(strategy.B_prefetch);
    }
    if (!isPacked(problem.C.layout)) {
        use2DAddressing(strategy.C);
        use2DAddressing(strategy.C_prefetch);
    }
    if (!(strategy.l3PrefetchA && isPacked(problem.A.layout)) && !(strategy.l3PrefetchB && isPacked(problem.B.layout)))
        use2DAddressing(strategy.AB_prefetchL3);

    // Notify kernel generator to downgrade block 2D prefetches if block 2D cannot be used.
    if (tags && !strategy.optAlignAB2D) {
        bool block2DA = false, block2DB = false;
        while (*tags) {
            block2DA |= (*tags == kcatalog::ReqBlock2DA);
            block2DB |= (*tags == kcatalog::ReqBlock2DB);
            tags++;
        }

        strategy.A_prefetch.preflight(hw);
        strategy.B_prefetch.preflight(hw);
        if (!block2DA && strategy.A_prefetch.address2D)
            strategy.A_prefetch.address2D = false;
        if (!block2DB && strategy.B_prefetch.address2D)
            strategy.B_prefetch.address2D = false;
    }

    // No need to use split remainder handling for 2D block accesses as there's no penalty for masking.
    if (isBlock2D(strategy.A.accessType) && (!strategy.prefetchA || isBlock2D(strategy.A_prefetch.accessType)) && !problem.quantized2DA())
        strategy.remHandling[LoopM] = RemainderHandling::General;
    if (isBlock2D(strategy.B.accessType) && (!strategy.prefetchB || isBlock2D(strategy.B_prefetch.accessType)) && !problem.quantized2DB())
        strategy.remHandling[LoopN] = RemainderHandling::General;

    // Also don't split remainder handling if padded.
    if (gemmAStrategy->padded) strategy.remHandling[LoopM] = RemainderHandling::General;
    if (gemmBStrategy->padded) strategy.remHandling[LoopN] = RemainderHandling::General;

    // But always use split remainder handling when prefetching C if it _isn't_ block 2D
    //  ... in that case there are no C prefetches on the remainder path.
    if (strategy.prefetchC && !isBlock2D(strategy.C_prefetch.accessType))
        strategy.remHandling[LoopM] = strategy.remHandling[LoopN] = RemainderHandling::Split;

    // ... and always use split remainder handling on later GPUs when panel checks are active.
    if (strategy.panelCheck && strategy.lateExit() && hw >= HW::XeHP && !strategy.fixedSystolic) {
        if (isPacked(problem.A.layout)) strategy.remHandling[LoopM] = RemainderHandling::Split;
        if (isPacked(problem.B.layout)) strategy.remHandling[LoopN] = RemainderHandling::Split;
    }

    // Finally, no remainder handling needed for GEMV-type kernels.
    if (strategy.unroll[LoopM] * strategy.wg[LoopM] == 1) strategy.remHandling[LoopM] = RemainderHandling::Ignore;
    if (strategy.unroll[LoopN] * strategy.wg[LoopN] == 1) strategy.remHandling[LoopN] = RemainderHandling::Ignore;
}

const char *parseLayout(const char *s, MatrixAddressing &atype)
{
    atype.crosspack = 1;
    atype.alignment = 0;

    switch (*s++) {
        case 'A': atype.layout = MatrixLayout::Pc; break;
        case 'B': atype.layout = MatrixLayout::Pr; break;
        case 'N': atype.layout = MatrixLayout::N;  break;
        case 'T': atype.layout = MatrixLayout::T;  break;
        case '?': atype.layout = MatrixLayout::N;  break; // Either N/T; used for fused GEMM.
        default:
            throw std::runtime_error("Unknown matrix layout requested.");
    }

    if (isdigit(*s))
        atype.crosspack = strtol(s, (char **) &s, 10);
    if (*s == '#') {
        s++;
        atype.tileR = strtol(s, (char **) &s, 10);
        if (*s != ',') throw std::runtime_error("Bad tiling syntax; expected #<rows>,<columns>.");
        s++;
        atype.tileC = strtol(s, (char **) &s, 10);
    }
    if (*s == '%') {
        s++;
        atype.packSize = strtol(s, (char **) &s, 10);
    }
    if (*s == '@') {
        s++;
        atype.alignment = strtol(s, (char **) &s, 10);
    }

    return s;
}

const char *parsePrecision(const char *s, Type &precision)
{
    if (*s) {
        precision = charPrecision(*s++);
    }
    return s;
}

const char *parsePrecisions(const char *s, Type &precision1, Type &precision2)
{
    if (*s == '[') {
        s++;
        s = parsePrecision(s, precision1);
        s = parsePrecision(s, precision2);
        if (*s++ != ']')
            throw std::runtime_error("Syntax error in precisions; expected ]");
    } else {
        s = parsePrecision(s, precision1);
        precision2 = precision1;
    }

    return s;
}

std::string unparseStrategy(HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    std::stringstream s;

    bool anyOptAlignAB = strategy.optAlignAB || strategy.optAlignAB2D;

    unparseAddressBase(s, strategy.A.base);
    unparseAccessType(s, strategy.A);
    if (anyOptAlignAB && strategy.unalignedAccA != strategy.A.accessType) {
        auto unalignedA = strategy.A;
        unalignedA.accessType = strategy.unalignedAccA;
        s << '/';
        unparseAccessType(s, unalignedA);
    }
    s << strategy.ka_load;
    if (strategy.ka_load_masked != strategy.ka_load)
        s << '/' << strategy.ka_load_masked;
    unparseTiling(s, strategy.A);
    if (strategy.A_copies > 1)
        s << 'x' << strategy.A_copies;
    unparseCaching(hw, s, strategy.A);
    if (strategy.prefetchA) {
        s << '+';
        unparseAccessType(s, strategy.A_prefetch);
        s << strategy.ka_prefetch << ',' << strategy.ka_pfStride << '@' << strategy.prefetchA;
        if (strategy.prefetchAMasked != strategy.prefetchA)
            s << '/' << strategy.prefetchAMasked;
        unparseCaching(hw, s, strategy.A_prefetch);
    }
    if (strategy.l3PrefetchA) {
        if (!strategy.prefetchA) s << '+';
        s << '+';
        unparseAccessType(s, strategy.AB_prefetchL3);
        s << strategy.ka_prefetchL3 << '@' << strategy.prefetchABL3;
        unparseCaching(hw, s, strategy.AB_prefetchL3);
    }
    s << ' ';
    unparseAddressBase(s, strategy.B.base);
    unparseAccessType(s, strategy.B);
    if (anyOptAlignAB && strategy.unalignedAccB != strategy.B.accessType) {
        auto unalignedB = strategy.B;
        unalignedB.accessType = strategy.unalignedAccB;
        s << '/';
        unparseAccessType(s, unalignedB);
    }
    s << strategy.kb_load;
    if (strategy.kb_load_masked != strategy.kb_load)
        s << '/' << strategy.kb_load_masked;
    unparseTiling(s, strategy.B);
    if (strategy.B_copies > 1)
        s << 'x' << strategy.B_copies;
    unparseCaching(hw, s, strategy.B);
    if (strategy.prefetchB) {
        s << '+';
        unparseAccessType(s, strategy.B_prefetch);
        s << strategy.kb_prefetch << ',' << strategy.kb_pfStride << '@' << strategy.prefetchB;
        if (strategy.prefetchBMasked != strategy.prefetchB)
            s << '/' << strategy.prefetchBMasked;
        unparseCaching(hw, s, strategy.B_prefetch);
    }
    if (strategy.l3PrefetchB) {
        if (!strategy.prefetchB) s << '+';
        s << '+';
        unparseAccessType(s, strategy.AB_prefetchL3);
        s << strategy.kb_prefetchL3 << '@' << strategy.prefetchABL3;
        unparseCaching(hw, s, strategy.AB_prefetchL3);
    }
    s << ' ';
    unparseAddressBase(s, strategy.C.base);
    unparseAccessType(s, strategy.C);
    unparseTiling(s, strategy.C);
    unparseCaching(hw, s, strategy.C);
    if (strategy.prefetchC) {
        s << '+';
        unparseAccessType(s, strategy.C_prefetch);
        s << '@' << strategy.prefetchC;
        unparseCaching(hw, s, strategy.C_prefetch);
    }

    if (strategy.slmA || strategy.slmB) {
        s << " c";
        if (strategy.slmA) s << 'a';
        if (strategy.slmB) s << 'b';
        s << strategy.slmBuffers;
        if (strategy.slmCopies > 1)
            s << 'x' << strategy.slmCopies;

        if (strategy.unrollKSLM != 0) {
            s << " ks" << strategy.unrollKSLM;
            if (strategy.unrollKSLMMasked > 0 && strategy.unrollKSLMMasked != strategy.unrollKSLM)
                s << '/' << strategy.unrollKSLMMasked;
        }
    }

    if (strategy.wg[LoopM] != 0) {
        s << " wg " << strategy.wg[LoopM] << 'x' << strategy.wg[LoopN];
        if (strategy.wg[LoopK] > 1)
            s << 'x' << strategy.wg[LoopK];
    }

    if (strategy.wgPadFactor > 1)           s << " wx" << strategy.wgPadFactor;
    if (strategy.forceWGUpdate == WGFixed)  s << " ff";


    if (strategy.kParallelLocal)    s << (strategy.shrinkWGK   ? " akr" :
                                          strategy.kInterleave ? " ikr" :
                                                                 " kr");
    if (strategy.fillGoal)          s << " fg" << strategy.fillGoal * (1. / 16);
    if (strategy.kInterleave)       s << " ki" << strategy.kInterleaveChunk;

    if (strategy.fixedSystolic)
        s << " fs";
    else if (strategy.systolic) {
        s << " sys";
        if (strategy.dpasw)
            s << " dw";
    }

    if (strategy.dotVL) {
        s << " dot";
        if (strategy.dotVL > 1) s << strategy.dotVL;
    }

    if (strategy.kChain > 1)                s << " kc" << strategy.kChain;

    if (strategy.atomicFMA)                 s << (strategy.extendedAtomicFMA ? " xaf" : " af");
    if (strategy.stallAfterLoad)            s << " st";
    if (strategy.fmaBoustrophedon)          s << " fx";

    s << " k" << strategy.unroll[LoopK];

    if (strategy.GRFs != 128)
        s << " grf" << strategy.GRFs;

    if (strategy.coopA == CoopSplit::MN)    s << " sm";
    if (strategy.coopA == CoopSplit::FullK) s << " ska";
    if (strategy.coopB == CoopSplit::MN)    s << " sn";
    if (strategy.coopB == CoopSplit::FullK) s << " skb";

    switch (strategy.registerScheme) {
        case GEMMStrategy::CSeparate:       s << " cs";  break;
        case GEMMStrategy::ACB:             s << " acb"; break;
        case GEMMStrategy::BCA:             s << " bca"; break;
        case GEMMStrategy::VNC:             s << " vnc"; break;
        case GEMMStrategy::ABInterleave:    s << " int"; break;
        case GEMMStrategy::NSeparate:       s << " nse"; break;
        case GEMMStrategy::VAvoid:          s << " vav"; break;
    }

    if (strategy.cAccumulators)             s << " ac";
    if (strategy.cLoadAhead)                s << " el";
    if (strategy.delayABInc)                s << " di";
    if (strategy.loadBFirst)                s << " ba";
    if (strategy.doubleMasking)             s << " dm";
    if (strategy.kDescRem)                  s << " kd";
    if (strategy.splitCopy)                 s << " sc";
    if (!strategy.slmUseIncrCopy)           s << " ni";
    if (strategy.slmEarlyKMask)             s << " ek";
    if (strategy.strictFence)               s << " sf";
    if (strategy.slmATrans)                 s << " ta";
    if (strategy.slmBTrans)                 s << " tb";
    if (!strategy.altCRemainder)            s << " sr";
    else if (!strategy.block2DCRemainder)   s << " ar";
    if (strategy.block2DCFull)              s << " bf";
    else if (strategy.block2DCRemainder)    s << " br";
    if (strategy.panelCheck)                s << " up";
    if (strategy.reverse[LoopM])            s << " rm";
    if (strategy.reverse[LoopN])            s << " rn";

    if (strategy.checkAdd32 && !strategy.emulate.emulate64) s << " ch";
    if (!strategy.checkAdd32 && strategy.emulate.emulate64) s << " nch";
    if (strategy.wgInSS)                                    s << " ws";
    if (strategy.C.smode == ScatterSIMD::Wide)              s << " wc";
    if (strategy.forceCopyC)                                s << " cc";

    if (!strategy.jointSplit)           s << " njs";
    if (strategy.mSplitThresh)          s << " ms" << strategy.mSplitThresh;
    if (strategy.nSplitThresh)          s << " ns" << strategy.nSplitThresh;

    if (strategy.kParallel)             s << " kb";
    if (strategy.kParallelVariable)     s << " kv";
    if (strategy.fuseBeta)              s << (strategy.altFusedBeta ? " afb" : " fb");
    if (strategy.fusePostOps)           s << " fp";
    if (strategy.zeroTempC)             s << " zt";
    if (strategy.relaxedAccumulation)   s << " rx";
    if (strategy.kPadding)              s << " pk" << strategy.kPadding;

    if (strategy.C.atomic && !strategy.kParallel && !strategy.kParallelVariable)
        s << " au";
    else if (!strategy.autoatomic)
        s << " nau";

    if (strategy.namedBarriers[LoopM] || strategy.namedBarriers[LoopN])
        s << " nb " << strategy.namedBarriers[LoopM] << 'x' << strategy.namedBarriers[LoopN];

    switch (strategy.cWalkOrder) {
        case WalkOrder::Hilbertlike:      s << " hi"; break;
        case WalkOrder::Boustrophedon:    s << " bo"; break;
        case WalkOrder::SimpleLinear:     s << " li"; break;
        case WalkOrder::NestedLinear:     s << " nl"; break;
        default: break;
    }

    if (strategy.persistent)
        s << " pt";

    if (strategy.loopOrder[0] != LoopM || strategy.loopOrder[1] != LoopN || strategy.loopOrder[2] != LoopK) {
        s << ' ';
        for (int l = 0; l < 3; l++)
            s << "mnk"[strategy.loopOrder[l]];
    }

    if (strategy.fused) s << " f" << "mnk"[strategy.fusedLoop];

    switch (strategy.arbitrationMode) {
        case ThreadArbitrationMode::OldestFirst:            s << " of"; break;
        case ThreadArbitrationMode::RoundRobin:             s << " rr"; break;
        case ThreadArbitrationMode::RoundRobinOnStall:      s << " rrs"; break;
        default: break;
    }

    if (strategy.optAlignAB > 0)    s << " l" << strategy.optAlignAB;
    if (strategy.optAlignAB2D)      s << " l2d";

    bool nq = false;
    for (auto &astrategy: {strategy.A, strategy.B, strategy.C, strategy.CO,
                           strategy.A_prefetch, strategy.B_prefetch, strategy.C_prefetch}) {
        nq |= astrategy.noExtraPad;
    }
    if (nq) s << " nq";

    if (strategy.barrierFreq > 0) {
        s << (strategy.splitBarrier ? " sb" : " b")
          << strategy.barrierFreq;
    }

    if (strategy.cRepackPanel)      s << " cr" << strategy.cRepackPanel;

    if (strategy.skewLocalIDs) {
        s << " ql";
    }

    bool npA = !isPacked(problem.A.layout) && !strategy.A.padded;
    bool npB = !isPacked(problem.B.layout) && !strategy.B.padded;
    bool pabA = !isPacked(problem.A.layout) && strategy.A.padded;
    bool pabB = !isPacked(problem.B.layout) && strategy.B.padded;
    bool pc = !isPacked(problem.C.layout) && strategy.C.padded;

    if (pabA || pabB)   s << " pab";
    if (npA || npB)     s << " np";
    if (pc)             s << " pc";

    for (auto loop: {LoopM, LoopN, LoopK}) if (strategy.blocking[loop] != 0) {
        s << " b" << "mnk"[loop] << strategy.blocking[loop];
        bool checkAlt = (loop == LoopK) ? (hw == HW::XeHP)
                                        : (strategy.cWalkOrder == WalkOrder::Hilbertlike);
        if (checkAlt && (strategy.blockingAlt[loop] != 0))
            s << '/' << strategy.blockingAlt[loop];
    }

    return s.str();
}


void unparseAccessType(std::ostream &s, const MatrixAddressingStrategy &astrategy)
{
    char c = ' ';
    switch (astrategy.accessType) {
        case AccessType::Block:             c = 'b'; break;
        case AccessType::PseudoBlock:       c = 'p'; break;
        case AccessType::Scattered:         c = 's'; break;
        case AccessType::ChannelScattered:  c = 'u'; break;
        case AccessType::Block2D:           c = 'm'; break;
        case AccessType::Block2DTranspose:  c = 't'; break;
        case AccessType::Block2DVNNI:       c = 'v'; break;
        case AccessType::CacheLine:         c = 'c'; break;
    }
    if (astrategy.newDP)
        c = std::toupper(c);
    s << c;
}

void unparseAddressBase(std::ostream &s, ngen::AddressBase base)
{
    switch (base.getModel()) {
        case ngen::AddressModel::ModelA64:     s << 'a'; break;
        case ngen::AddressModel::ModelCC:      s << 'c'; break;
        case ngen::AddressModel::ModelSC:      s << 'm'; break;
        case ngen::AddressModel::ModelBTS:     s << 's'; break;
        case ngen::AddressModel::ModelInvalid: s << 'r'; break;
        default: s << 'u'; break;
    }
}

void unparseCaching(HW hw, std::ostream &s, const MatrixAddressingStrategy &astrategy)
{
    auto cachingR = astrategy.cachingR;
    auto cachingW = astrategy.cachingW;

    if (!astrategy.newDP)
        return;
    if (cachingR == CacheSettingsLSC::Default && cachingW == CacheSettingsLSC::Default)
        return;

    s << '{';

    {
        switch (cachingR) {
            case CacheSettingsLSC::Default:   s << "dd"; break;
            case CacheSettingsLSC::L1UC_L3UC: s << "uu"; break;
            case CacheSettingsLSC::L1UC_L3C:  s << "uc"; break;
            case CacheSettingsLSC::L1C_L3UC:  s << "cu"; break;
            case CacheSettingsLSC::L1C_L3C:   s << "cc"; break;
            case CacheSettingsLSC::L1S_L3UC:  s << "su"; break;
            case CacheSettingsLSC::L1S_L3C:   s << "sc"; break;
            case CacheSettingsLSC::L1IAR_L3C: s << "ic"; break;
            case CacheSettingsLSC::L1C_L3CC:  s << "cn"; break;
            case CacheSettingsLSC::L1UC_L3CC: s << "un"; break;
            default:                          s << "??"; break;
        }
        if (cachingW != CacheSettingsLSC::Default) {
            s << '/';
            switch (cachingW) {
                case CacheSettingsLSC::L1UC_L3UC: s << "uu"; break;
                case CacheSettingsLSC::L1UC_L3C:  s << "ub"; break;
                case CacheSettingsLSC::L1C_L3UC:  s << "tu"; break;
                case CacheSettingsLSC::L1C_L3C:   s << "tb"; break;
                case CacheSettingsLSC::L1S_L3UC:  s << "su"; break;
                case CacheSettingsLSC::L1S_L3C:   s << "sb"; break;
                case CacheSettingsLSC::L1IAR_L3C: s << "bb"; break;
                default:                          s << "??"; break;
            }
        }
    }
    s << '}';
}

void unparseTiling(std::ostream &s, const MatrixAddressingStrategy &astrategy)
{
    if (astrategy.tileR || astrategy.tileC)
        s << '#' << int(astrategy.tileR) << ',' << int(astrategy.tileC);
}


#include "internal/namespace_end.hxx"
