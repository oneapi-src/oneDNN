/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#define BINARY_OUTPUT

#include "microkernel_provider.hpp"
#include "generator.hpp"
#include "kernel_selector.hpp"
#include "strategy_parser.hpp"
#include "npack/neo_packager.hpp"

#include "internal/namespace_start.hxx"

#define _CATALOG_ CatalogMMR
#include "selector/db/ukernel_mmr.db"
;
#undef _CATALOG_

#define _CATALOG_ CatalogLMR
#include "selector/db/ukernel_lmr.db"
;
#undef _CATALOG_

#define _CATALOG_ CatalogMLR
#include "selector/db/ukernel_mlr.db"
;
#undef _CATALOG_

using namespace ngen;
using namespace micro;

static inline bool getStrategyByHeuristics(HW hw, GEMMStrategy &strategy, bool localA, bool localB,
                                           GEMMProblem &problem, HWInformation hwInfo, SizeParams sizes,
                                           const std::vector<StrategyRequirement> &reqs);

Package selectGEMMMicrokernel(GEMMProtocol protocol, HWInformation hwInfo, SizeParams sizes,
                              const GEMMProblem &problem_, const std::vector<StrategyRequirement> &reqs_,
                              void (*strategyAdjuster)(GEMMStrategy &strategy))
{
    kcatalog::Catalog catalog;

    bool localA = protocol.options().localA;
    bool localB = protocol.options().localB;
    bool beta1 = protocol.options().addToC;
    bool slmPtr = protocol.options().slmPtr;
    bool scaleA = protocol.options().scaleA;
    bool scaleB = protocol.options().scaleB;
    bool offsetA = protocol.options().offsetA;
    bool offsetB = protocol.options().offsetB;

    bool transC = !isColMajor(problem_.C.layout);

    auto problem = problem_;
    auto reqs = reqs_;

    problem.alpha = 1;
    problem.beta = beta1 ? 1 : 0;

    problem.C.setAlignment(4);

    if (transC) {
        problem.transpose();
        std::swap(localA, localB);
        std::swap(sizes.m, sizes.n);
        std::swap(scaleA, scaleB);
        std::swap(offsetA, offsetB);
        for (auto &req: reqs)
            req.transpose();
    }

    if (scaleA != problem.aScale2D || scaleB != problem.bScale2D)
        stub("Protocol scales do not match problem description");
    if (offsetA != (problem.aoPtrDims >= 0) || offsetB != (problem.boPtrDims >= 0))
        stub("Protocol offsets do not match problem description");

    /* Get hardware information */
    auto product = npack::decodeHWIPVersion(hwInfo.gmdid);
    auto hw = getCore(product.family);
    auto stepping = hwInfo.gmdid & 0xFF;

    bool isIntegrated = getPlatformType(product.family) == PlatformType::Integrated;

    /* Create catalog matcher */
    MatchParams matchParams(hw, hwInfo.systolicAvailable, isIntegrated, problem);

    matchParams.sizes = sizes;
    matchParams.stepping = stepping;
    matchParams.nExtraReqs = int(reqs.size());
    matchParams.extraReqs = reqs.data();

    auto tags = const_cast<char *>(matchParams.tags);
    while (*tags)
        tags++;
    *tags++ = kcatalog::ReqBlock2DA;
    *tags++ = kcatalog::ReqBlock2DB;

    /* Provide information for kernel selection */
    EvaluateParams evalParams;
    evalParams.sizes = matchParams.sizes;
    evalParams.alpha = 1;
    evalParams.beta = 0;
    evalParams.euCount = hwInfo.euCount;

    /* Locate appropriate kernel catalog */
    if (localA && localB)
        stub("Unsupported protocol");

    if (localA)
        catalog = CatalogLMR;
    else if (localB)
        catalog = CatalogMLR;
    else
        catalog = CatalogMMR;

    /* Call kernel selector */
    EvaluateAuxOutput auxParams;
    auto entry = select(catalog, 1, &matchParams, evalParams, auxParams);

    GEMMStrategy strategy(hw, stepping);

    if (entry) {
        problem.A.setAlignment(std::max(problem.Ta.size(), entry->driverInfo.alignment[0]));
        problem.B.setAlignment(std::max(problem.Tb.size(), entry->driverInfo.alignment[1]));

        /* Prepare strategy parameters */
        strategy.unroll[LoopM] = entry->driverInfo.unroll[LoopM];
        strategy.unroll[LoopN] = entry->driverInfo.unroll[LoopN];
        parseStrategy(entry->strategy, hw, problem, strategy);
        adjustStrategy(hw, problem, strategy);
        modifyStrategy(strategy, auxParams);

        /* Xe2/Xe3-XeHPC compatibility logic */
        if (hw == ngen::HW::Xe2 || hw == ngen::HW::Xe3) {
            // Use XeHPC register banking on Xe2/Xe3, in order
            //   to successfully reuse XeHPC strategies.
            strategy.raHW = ngen::HW::XeHPC;

            // Bump up alignments to 16 bytes for block 2D if available.
            bool block2DA = false, block2DB = false;
            for (auto c = entry->restrictions.tags; *c; c++) {
                block2DA |= (*c == kcatalog::ReqBlock2DA);
                block2DB |= (*c == kcatalog::ReqBlock2DB);
            }
            if (block2DA && strategy.legalAAlignment(problem, 16))
                problem.A.setAlignment(std::max<int>(problem.A.alignment, 16));
            if (block2DB && strategy.legalBAlignment(problem, 16))
                problem.B.setAlignment(std::max<int>(problem.B.alignment, 16));
        }
    } else if (!getStrategyByHeuristics(hw, strategy, localA, localB, problem, hwInfo, sizes, reqs))
        throw std::runtime_error("No matching kernel");

    strategy.systolicAvailable &= hwInfo.systolicAvailable;

    /* Disable strategies not related to microkernels */
    strategy.kParallel = strategy.kParallelVariable = strategy.persistent = false;
    strategy.cWalkOrder = WalkOrder::HW2D;

    /* Adjust strategy for performance */
    if (strategy.barrierFreq > 0 && sizes.k < 4 * strategy.barrierFreq)
        strategy.barrierFreq = 0;

    /* Keep size down by only using checkAdd32 when really needed */
    strategy.checkAdd32 &= (hw != HW::XeHPC);

    /* C output in registers */
    strategy.C.base = AddressBase{};

    /* Allow caller to adjust strategy further */
    if (strategyAdjuster) strategyAdjuster(strategy);

    strategy.preflight(hw, problem);

    /* Set up arguments for microkernel */
    InterfaceHandler interface(hw);

    interface.setArgumentBase(ngen::GRF(8));
    interface.newArgument("A", localA ? ExternalArgumentType::LocalPtr : ExternalArgumentType::GlobalPtr);
    interface.newArgument("lda", DataType::d);
    interface.newArgument("B", localB ? ExternalArgumentType::LocalPtr : ExternalArgumentType::GlobalPtr);
    interface.newArgument("ldb", DataType::d);
    interface.newArgument("m", DataType::d);
    interface.newArgument("n", DataType::d);
    interface.newArgument("k", DataType::d);
    interface.newArgument("i0", DataType::d);
    interface.newArgument("j0", DataType::d);
    interface.newArgument("h0", DataType::d);
    interface.newArgument("local_id_m", DataType::d);
    interface.newArgument("local_id_n", DataType::d);
    if (slmPtr)            interface.newArgument("slm_base", ExternalArgumentType::LocalPtr);
    if (scaleA)            interface.newArgument("a_scale_ptr", ExternalArgumentType::GlobalPtr);
    if (offsetA)           interface.newArgument("ao_ptr", ExternalArgumentType::GlobalPtr);
    if (scaleA || offsetA) interface.newArgument("ldaq", DataType::d);
    if (scaleB)            interface.newArgument("b_scale_ptr", ExternalArgumentType::GlobalPtr);
    if (offsetB)           interface.newArgument("bo_ptr", ExternalArgumentType::GlobalPtr);
    if (scaleB || offsetB) interface.newArgument("ldbq", DataType::d);

    /* Update problem from strategy */
    if (isPacked(problem.A.layout))
        problem.A.packSize = strategy.unroll[LoopM];
    if (isPacked(problem.B.layout))
        problem.B.packSize = strategy.unroll[LoopN];

    /* Generate microkernel */
#define ARCH_DISPATCH(arch)                                                         \
    case HW::arch: {                                                                \
        BLASKernelGenerator<HW::arch> generator;                                    \
        generator.setStepping(stepping);                                            \
        return generator.gemmMicrokernelPackage(problem, strategy, interface,       \
                                                protocol, hwInfo.gmdid, transC);    \
    }

    switch (hw) {
        REG_GEN9_ISA(ARCH_DISPATCH(Gen9))
        REG_GEN11_ISA(ARCH_DISPATCH(Gen11))
        REG_XELP_ISA(ARCH_DISPATCH(XeLP))
        REG_XEHP_ISA(ARCH_DISPATCH(XeHP))
        REG_XEHPG_ISA(ARCH_DISPATCH(XeHPG))
        REG_XEHPC_ISA(ARCH_DISPATCH(XeHPC))
        REG_XE2_ISA(ARCH_DISPATCH(Xe2))
        REG_XE3_ISA(ARCH_DISPATCH(Xe3))
        default: throw std::runtime_error("Unsupported architecture");
    }
#undef ARCH_DISPATCH
}

static inline bool getStrategyByHeuristics(HW hw, GEMMStrategy &strategy, bool localA, bool localB,
                                           GEMMProblem &problem, HWInformation hwInfo, SizeParams sizes,
                                           const std::vector<StrategyRequirement> &reqs)
{
    if (hw < HW::XeHPG) return false;
    if (problem.C.layout == MatrixLayout::T) return false;
    if (problem.Ta.size() != 2 || problem.Tb.size() != 2) return false;

    bool systolic = hwInfo.systolicAvailable;
    bool block2DA = (hw >= HW::XeHPC) && systolic && (problem.A.alignment % 16) == 0;
    bool block2DB = (hw >= HW::XeHPC) && systolic && (problem.B.alignment % 16) == 0;

    problem.A.alignment = std::min<int>(problem.A.alignment, 16);
    problem.B.alignment = std::min<int>(problem.B.alignment, 16);

    auto &s = strategy;

    s.ka_load = s.kb_load = 16;
    if (!systolic)
        s.ka_load = s.kb_load = 4;

    if (problem.A.layout == MatrixLayout::Pc) {
        s.A.accessType = AccessType::Block;
        s.A_copies = 2;
        s.A.padded = true;
    } else if (!block2DA) {
        s.A.accessType = AccessType::Block;
        if (systolic)
            s.ka_load = (problem.A.layout == MatrixLayout::T) ? (64 / problem.Ta_ext) : 16;
        s.slmA = true;
    } else if (problem.A.layout == MatrixLayout::T) {
        s.A.accessType = AccessType::Block2DTranspose;
        s.ka_load = 64 / problem.Ta_ext;
    } else if (problem.A.layout == MatrixLayout::N) {
        s.A.accessType = AccessType::Block2DVNNI;
        s.A_copies = 2;
    }

    if (problem.B.layout == MatrixLayout::Pr) {
        s.B.accessType = AccessType::Block;
        s.B.padded = true;
        s.B_copies = 2;
    } else if (!block2DB) {
        s.B.accessType = AccessType::Block;
        if (systolic) {
            s.doubleMasking = true;
            s.kb_load = (problem.B.layout == MatrixLayout::N) ? 32 : 16;
        }
        s.slmB = true;
    } else if (problem.B.layout == MatrixLayout::T)
        s.B.accessType = AccessType::Block2DTranspose;
    else if (problem.B.layout == MatrixLayout::N) {
        s.B.accessType = AccessType::Block2D;
        s.kb_load = 32;
    }

    s.C.accessType = AccessType::Block;

    s.A.base = localA ? AddressBase::createSLM() : AddressBase::createA64(true);
    s.B.base = localB ? AddressBase::createSLM() : AddressBase::createA64(true);
    s.A.newDP = true;
    s.B.newDP = true;
    s.A.cachingR = s.B.cachingR = CacheSettingsLSC::L1C_L3C;

    s.A_prefetch = s.A;
    s.B_prefetch = s.B;
    s.A_prefetch.prefetch = s.B_prefetch.prefetch = true;

    s.AO.newDP = s.A_scale.newDP = true;
    s.BO.newDP = s.B_scale.newDP = true;

    if (!localA && block2DA) {
        if (!isPacked(problem.A.layout))
            s.A_prefetch.accessType = AccessType::Block2D;
        s.prefetchA = s.prefetchAMasked = 2 * s.ka_load;
        s.ka_pfStride = s.ka_prefetch = s.ka_load;
    }

    if (!localB && block2DB) {
        if (!isPacked(problem.B.layout))
            s.B_prefetch.accessType = AccessType::Block2D;
        s.prefetchB = s.prefetchBMasked = 2 * s.kb_load;
        s.kb_pfStride = s.kb_prefetch = s.kb_load;
    }

    s.unroll[LoopK] = 1;
    s.wg[LoopK] = 1;
    s.unroll[LoopM] = s.unroll[LoopN] = 0;
    s.wg[LoopM] = s.wg[LoopN] = 0;

    for (auto &req: reqs) switch (req.param) {
        case StrategyRequirement::UnrollM: s.unroll[LoopM] = req.value; break;
        case StrategyRequirement::UnrollN: s.unroll[LoopN] = req.value; break;
        case StrategyRequirement::WGM:         s.wg[LoopM] = req.value; break;
        case StrategyRequirement::WGN:         s.wg[LoopN] = req.value; break;
        case StrategyRequirement::WGK:         s.wg[LoopK] = req.value; break;
        default: break;
    }

    if (s.wgTile(LoopM) * s.wgTile(LoopN) == 0)
        return false;

    s.systolic = systolic;
    if (systolic && hw >= HW::XeHPC)
        s.extendedAtomicFMA = s.atomicFMA = true;
    s.registerScheme = GEMMStrategy::VAvoid;
    if (s.wgTile(LoopM) * s.wgTile(LoopN) > 512)
        s.GRFs = 256;
    if (localA && !localB)
        s.loadBFirst = true;

    if (s.slmA || s.slmB) {
        s.slmBuffers = 1;
        s.unrollKSLM = std::max(int(s.slmA) * s.ka_load, int(s.slmB) * s.kb_load);
    }

    if (hw == HW::Xe2 || hw == HW::Xe3)
        s.raHW = HW::XeHPC;

    adjustStrategy(hw, problem, strategy);

    return true;
}

#include "internal/namespace_end.hxx"
