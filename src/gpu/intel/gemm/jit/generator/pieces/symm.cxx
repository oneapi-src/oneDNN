/*******************************************************************************
* Copyright 2019 Intel Corporation
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


#include "alloc_utils.hpp"
#include "gemmstone/generator.hpp"
#include "layout_utils.hpp"
#include "map.hpp"
#include "token_alloc_utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;
using std::vector;


template <HW hw>
bool Generator<hw>::symmKLoop(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    bool left  = problem.left;
    bool lower = problem.lower;

    auto loopX            = left ? LoopM                     : LoopN;
    auto Tu               = left ? problem.Ta                : problem.Tb;
    auto Tu_ext           = left ? problem.Ta_ext            : problem.Tb_ext;
    auto Tv_ext           = left ? problem.Tb_ext            : problem.Ta_ext;
    auto &U               = left ? problem.A                 : problem.B;
    auto &U_alt           = left ? state.A_alt               : state.B_alt;
    auto &U_strategy      = left ? strategy.A                : strategy.B;
    auto &U_strategyAlt   = left ? state.A_strategyAlt       : state.B_strategyAlt;
    auto &ku_load         = left ? strategy.ka_load          : strategy.kb_load;
    auto &ku_loadRem      = left ? state.ka_loadRem          : state.kb_loadRem;
    auto ku_loadSYMM      = left ? strategy.ka_loadTRSM      : strategy.kb_loadTRSM;
    auto &V               = left ? problem.B                 : problem.A;
    auto &V_strategy      = left ? strategy.B                : strategy.A;
    auto &kv_load         = left ? strategy.kb_load          : strategy.ka_load;
    auto &kv_loadRem      = left ? state.kb_loadRem          : state.ka_loadRem;
    bool slmU             = left ? strategy.slmA             : strategy.slmB;
    bool slmV             = left ? strategy.slmB             : strategy.slmA;
    auto ldu              = left ? state.lda                 : state.ldb;
    auto &U_layout        = left ? state.A_layout            : state.B_layout;
    auto &U_layoutRem     = left ? state.A_layoutRem         : state.B_layoutRem;
    auto &U_layoutAlt     = left ? state.A_layoutAlt         : state.B_layoutAlt;
    auto &U_layoutAltRem  = left ? state.A_layoutAltRem      : state.B_layoutAltRem;
    auto &U_addrs         = left ? state.A_addrs             : state.B_addrs;
    auto &U_addrsRem      = left ? state.A_addrsRem          : state.B_addrsRem;
    auto &U_addrsAlt      = left ? state.A_addrsAlt          : state.B_addrsAlt;
    auto &U_addrsAltRem   = left ? state.A_addrsAltRem       : state.B_addrsAltRem;
    auto &Up_strategy     = left ? strategy.A_prefetch       : strategy.B_prefetch;
    auto &Up_strategyAlt  = left ? state.Ap_strategyAlt      : state.Bp_strategyAlt;
    auto &Up_layout       = left ? state.Ap_layout           : state.Bp_layout;
    auto &Up_layoutAlt    = left ? state.Ap_layoutAlt        : state.Bp_layoutAlt;
    auto &Up_addrs        = left ? state.Ap_addrs            : state.Bp_addrs;
    auto &Up_addrsAlt     = left ? state.Ap_addrsAlt         : state.Bp_addrsAlt;
    auto &ku_pfTotal      = left ? strategy.ka_prefetch      : strategy.kb_prefetch;
    auto &ku_pfAltTotal   = left ? state.ka_prefetchAltTotal : state.kb_prefetchAltTotal;
    auto &ku_prefetch     = left ? state.ka_prefetch         : state.kb_prefetch;
    auto &ku_prefetchAlt  = left ? state.ka_prefetchAlt      : state.kb_prefetchAlt;
    auto &mnu_prefetch    = left ? state.ma_prefetch         : state.nb_prefetch;
    auto &mnu_prefetchAlt = left ? state.ma_prefetchAlt      : state.nb_prefetchAlt;
    auto &effCoopU        = left ? state.effCoopA            : state.effCoopB;
    auto &effCoopUAlt     = left ? state.effCoopAAlt         : state.effCoopBAlt;
    auto ldiag            = left ? state.diagA               : -state.diagB;
    auto &effU            = left ? state.effA                : state.effB;
    auto &effUp           = left ? state.effAp               : state.effBp;
    auto &conjU           = left ? state.conjA               : state.conjB;
    auto &V_layout        = left ? state.B_layout            : state.A_layout;
    auto &V_addrs         = left ? state.B_addrs             : state.A_addrs;

    auto originalU        = U;
    auto &symmUBlock      = state.symmUBlock;
    auto unrollX          = strategy.unroll[loopX];
    bool remX             = strategy.remHandling[loopX] != RemainderHandling::Ignore;
    auto lidX             = state.lidX;
    bool barriers         = (strategy.barrierFreq > 0);
    bool globalCM         = state.C_layout.colMajor();
    bool hemm             = (problem.srcType == MatrixSrcType::Hermitian);
    bool splitComplexC    = state.C_layout.isSplitComplex();
    bool hemmUseConjU     = hemm && (splitComplexC || left != globalCM);

    Label lSkipDiagonal, lEnterSecondGEMM;


    // Various setups we don't support.
    if (slmU)                                                               stub();
    if (slmV && ((strategy.unrollKSLM * strategy.slmBuffersA) % unrollX))    stub();
    if (isPacked(U.layout))
        stub();
    if (isBlock2D(U_strategy.accessType))                                   stub();
    if (barriers && (strategy.kParallel || strategy.kParallelLocal))        stub();

    if (ku_load > unrollX)
        ku_loadSYMM = unrollX;

    // Switch between access to U and U^T.
    auto transposeU = [&] {
        std::swap(U, U_alt);
        std::swap(U_strategy, U_strategyAlt);
        std::swap(U_layout, U_layoutAlt);
        std::swap(U_layoutRem, U_layoutAltRem);
        std::swap(U_addrs, U_addrsAlt);
        std::swap(U_addrsRem, U_addrsAltRem);
        std::swap(Up_strategy, Up_strategyAlt);
        std::swap(Up_layout, Up_layoutAlt);
        std::swap(Up_addrs, Up_addrsAlt);
        std::swap(ku_pfTotal, ku_pfAltTotal);
        std::swap(ku_prefetch, ku_prefetchAlt);
        std::swap(mnu_prefetch, mnu_prefetchAlt);
        std::swap(effCoopU, effCoopUAlt);
        clearMappedTokenAllocations(hw, state);
        gemmAllocateTokens(problem, strategy, state);
    };

    auto offsetAk = [&](auto koff) {
        gemmOffsetAk(koff, state.effA, problem.A, problem, strategy, state);
    };
    auto offsetBk = [&](auto koff) {
        gemmOffsetBk(koff, state.effB, problem.B, problem, strategy, state);
    };
    auto offsetUk = [&](auto koff) {
        left ? offsetAk(koff) : offsetBk(koff);
    };
    auto offsetVk = [&](auto koff) {
        left ? offsetBk(koff) : offsetAk(koff);
    };

    auto recalcUPrefetch = [&] {
        if (effUp != effU) {
            state.ra.safeRelease(effUp);
            effUp = effU;
        }
        gemmABPrefetchAddrSetup(problem, strategy, state, left, !left);
    };

    // Decide whether A/B need address resets between loops.
    bool resetA = false, resetB = false;
    auto &resetU = left ? resetA : resetB;
    auto &resetV = left ? resetB : resetA;

    resetU = true;
    resetV = (unrollX % ku_load != 0) || (unrollX % kv_load != 0);
    bool resetV2 = resetV;

    // Periodic barrier handling.
    Subregister kNoBarrier;
    if (barriers) {
        state.splitBarrierAlways = strategy.splitBarrier;
        kNoBarrier = state.ra.alloc_sub<uint32_t>();
    }

    // First GEMM chunk. For LU/RL cases, U has already been transposed.
    auto name = hemm ? "HEMM" : "SYMM";
    status << "GEMM loop for " << name << (left ? " lower" : " upper") << " triangle" << status_stream::endl;

    if (hemmUseConjU && left != lower)
        conjU = !conjU;

    state.K = state.ra.alloc_sub<int32_t>();

    if (!kLoopSetup(problem, strategy, state)) return false;

    if (remX) resetV2 |= (kv_load > kv_loadRem);    /* reset V for second GEMM loop only? */

    if (barriers)
        emad(1 | sat, kNoBarrier, -ldiag, -lidX.w(), unrollX, strategy, state);
    min_(1 | sat, state.K.ud(), -ldiag, state.k);
    if (barriers)
        add(1 | sat, kNoBarrier, -kNoBarrier, state.K);
    if (unrollX % ku_load == 0 && unrollX % kv_load == 0)
        state.kAlign = unrollX;
    state.kNoBarrierEnd = kNoBarrier;
    kLoop(KLoop::GEMM, problem, strategy, state);
    state.kNoBarrierEnd = Subregister();
    state.kAlign = 1;

    offsetUk(-ldiag);
    if (resetV2)
        offsetVk(-ldiag);

    // Non-workgroup-coordinated GEMMs from wgX0 to x0 would go here...

    // SYMM/HEMM handling around diagonal.
    status << name << " diagonal handling" << status_stream::endl;

    bool tdiag = !isTransposing(U_strategy.accessType);
    if (tdiag) transposeU();                    /* ensure we get a scattered layout for U */

    recalcUPrefetch();

    symmUBlock = 1;

    // Trim down ku_load if requested.
    auto ku_loadSave = ku_load;
    auto kv_loadSave = kv_load;
    auto kChainSave = strategy.kChain;
    if (ku_loadSYMM > 0) {
        ku_load = ku_loadSYMM;
        strategy.trimKChain(hw, ku_loadSYMM, problem);
    }

    if (!remX) {
        // Optimized path with vectorized scattered loads. Identify vector length (symmUBlock).
        if (unrollX % ku_load) stub();
        symmUBlock = 0;
        for (auto &block: U_layout) {
            auto uBlock = (block.colMajor ? block.nc : block.nr);
            if (symmUBlock == 0)
                symmUBlock = uBlock;
            else if (uBlock != symmUBlock) stub();
        }
    } else if (ku_loadRem > 1) stub();

    // If not broadcasting individual elements of U, need to de-vectorize scattered loads.
    RegisterLayout U_layoutSYMM, U_layout0, V_layoutSYMM;
    vector<RegisterRange> U_addrsSYMM, U_addrs0, V_addrsSYMM;

    if (left == globalCM && symmUBlock > 1) {
        symmUBlock = 1;
        RegisterLayout U_layout1;
        vector<RegisterRange> U_addrs1;
        vector<RegisterBlock> U_listSYMM;

        for (int h = 0; h < ku_load; h++) {
            U_layout1 = U_layout.trySlice(U_addrs1, U_addrs, left, h, h+1, U_strategy.padded);
            bool needNewAddrs = !U_layout1.valid();
            if (needNewAddrs)
                U_layout1 = U_layout.slice(left, h, h+1, U_strategy.padded, true);

            U_listSYMM.insert(U_listSYMM.end(), U_layout1.begin(), U_layout1.end());
            if (needNewAddrs) for (int i = 0; i < U_layout1.blocks(); i++)
                U_addrsSYMM.push_back(RegisterRange());
            else
                U_addrsSYMM.insert(U_addrsSYMM.end(), U_addrs1.begin(), U_addrs1.end());
        }

        U_layoutSYMM = RegisterLayout(Tu, U, U_strategy, std::move(U_listSYMM));

        // Allocate any missing address registers.
        for (int l = 0; l < U_layoutSYMM.blocks(); l++) {
            auto &addr = U_addrsSYMM[l];
            auto &block = U_layoutSYMM[l];
            if (block.isLoadBlock() && addr.isInvalid())
                addr = state.ra.alloc_range(block.addrGRFs(U, U_strategy));
        }
    }

    // Trim down U layout (part 2), and also V if needed.
    if (symmUBlock > 1 && ku_load != ku_loadSave)
        U_layoutSYMM = U_layout.slice(U_addrsSYMM, U_addrs, left, 0, ku_loadSYMM, U_strategy.padded);

    if (!U_layoutSYMM.empty()) {
        std::swap(U_layout, U_layoutSYMM);
        std::swap(U_addrs, U_addrsSYMM);
    }

    if (kv_load > unrollX && !remX) {
        kv_load = unrollX;
        V_layoutSYMM = V_layout.slice(V_addrsSYMM, V_addrs, !left, 0, kv_load, V_strategy.padded);
        std::swap(V_layout, V_layoutSYMM);
        std::swap(V_addrs, V_addrsSYMM);
    }

    // Separate out k = 0 blocks -- these are the only ones with "normal" address setups.
    // Pass these to kLoopReset for address initialization.
    U_layout0 = U_layout.slice(U_addrs0, U_addrs, left, 0, symmUBlock, U_strategy.padded);

    auto savedU = U;
    auto savedUStrategy = U_strategy;

    bool transposeFarTriangle = (left == isColMajor(originalU.layout));
    if (transposeFarTriangle) {
        U.layout = transposeLayout(U.layout);
        U_strategy.accessType = AccessType::PseudoBlock;
        U_layout0.addressing() = U_layout.addressing() = U;
        U_layout0.addressingStrategy() = U_layout.addressingStrategy() = U_strategy;
    }

    std::swap(U_layout, U_layout0);
    std::swap(U_addrs, U_addrs0);
    kLoopReset(Subregister(), problem, strategy, state, false, resetA, resetB);
    std::swap(U_addrs, U_addrs0);
    std::swap(U_layout, U_layout0);

    cmp(1 | gt | state.flagAP, ldiag, 0);
    add(1 | sat, state.K.ud(), state.k, ldiag);
    jmpi(1 | state.flagAP, lSkipDiagonal);

    // Initialize address increment vector.
    state.addrIncUStride = state.addrIncUConsecutive = 1;
    if (U_strategy.base.isStateless() && !strategy.emulate.emulate64 && !strategy.emulate.emulate64_add32) {
        state.addrIncUStride = 2;
        state.addrIncUConsecutive = U_layout[0].extra;
    }
    state.addrIncU = chunkAlloc(GRF::bytesToGRFs(hw, unrollX * sizeof(int32_t) * state.addrIncUStride * state.addrIncUConsecutive), 2, state);
    auto ldu_ku = lookupIncrement(left ? state.ldaIncrements : state.ldbIncrements, ldu, symmUBlock, strategy, state);
    map<uint32_t>(hw, state.addrIncU, state.addrIncU, strategy, [&](int simd, GRF r, GRF _) {
        transposeFarTriangle ? mov(simd, r, (symmUBlock > 1) ? ldu_ku : ldu)
                             : mov(simd, r, symmUBlock * Tu);
    });

    U_layoutRem.addressingStrategy() = U_layout.addressingStrategy();

    // Set up addresses for h > 0 blocks.
    symmUIncrement(remX ? U_layoutRem : U_layout, remX ? U_addrsRem : U_addrs, problem, strategy, state, -1);

    // Disable periodic barriers.
    auto saveBarrierFreq = strategy.barrierFreq;
    strategy.barrierFreq = 0;

    // Restore U layout now for correct prefetch address increments.
    U_layoutRem.addressing() = U_layout.addressing() = U = savedU;

    // Emit unrolled SYMM/HEMM loop.
    state.inSYMMLoop = true;
    kLoop(KLoop::SYMM, problem, strategy, state);
    state.inSYMMLoop = false;

    // Undo modifications and clean up.
    U_layoutRem.addressingStrategy() = U_layout.addressingStrategy() = U_strategy = savedUStrategy;
    symmUBlock = 1;
    ku_load = ku_loadSave;
    kv_load = kv_loadSave;
    strategy.barrierFreq = saveBarrierFreq;
    strategy.kChain = kChainSave;

    if (!U_layoutSYMM.empty()) {
        std::swap(U_layout, U_layoutSYMM);
        std::swap(U_addrs, U_addrsSYMM);
        safeReleaseRanges(U_addrsSYMM, state);
        reclaimRanges(U_addrs, state);
    }

    if (!V_layoutSYMM.empty()) {
        std::swap(V_layout, V_layoutSYMM);
        std::swap(V_addrs, V_addrsSYMM);
    }

    safeReleaseRanges(state.addrIncU, state);
    state.raVFlag.safeRelease(state.flagXMask);

    if (tdiag) transposeU();

    resetV = resetV2;

    // Non-workgroup-coordinated GEMMs from x1 to wgX1 would go here...

    // Second GEMM chunk.
    status << "GEMM loop for " << name << (left ? " upper" : " lower") << " triangle" << status_stream::endl;

    if (hemm) {
        if (hemmUseConjU)
            conjU = !conjU;
        else
            complexConjugateC(problem, strategy, state);
    }

    transposeU();

    offsetUk(unrollX);
    if (resetV2)
        offsetVk(unrollX);
    add(1 | sat, state.K.ud(), state.K, -unrollX);
    jmpi(1, lEnterSecondGEMM);

    mark(lSkipDiagonal);
    offsetUk(ldiag);
    if (resetV2)
        offsetVk(ldiag);
    mov(1, state.K, state.k);

    mark(lEnterSecondGEMM);

    recalcUPrefetch();

    if (barriers) {
        // Prepare to line up barrier counts between threads.
        if (strategy.fixedWG(problem))
            emad(1, kNoBarrier, unrollX * (strategy.wg[loopX] - 1), -lidX.w(), unrollX, strategy, state);
        else {
            eadd3(1, kNoBarrier, state.lszX, -lidX, -1);
            mulConstant(1, kNoBarrier, kNoBarrier, unrollX);
        }
        min_(1 | sat, kNoBarrier, kNoBarrier, state.K);
        int unrollK = strategy.unroll[LoopK];
        if (unrollX % unrollK) {
            auto temp = state.ra.alloc_sub<uint32_t>();
            add(1, temp, state.K, -kNoBarrier);
            mod(temp, temp, unrollK, strategy, state);
            add(1, kNoBarrier, kNoBarrier, temp);
            alignDown(kNoBarrier, kNoBarrier, unrollK, strategy, state);
            state.ra.safeRelease(temp);
        }
    }

    state.kNoBarrierStart = kNoBarrier;
    kLoopReset(Subregister(), problem, strategy, state, false, resetA, resetB);
    kLoop(KLoop::GEMM, problem, strategy, state);
    state.kNoBarrierStart = Subregister();

    transposeU();

    state.ra.safeRelease(state.K);
    state.ra.safeRelease(kNoBarrier);
    kLoopTeardown(problem, strategy, state);

    if (!strategy.persistentLoop()) {
        state.ra.safeRelease(state.diagA);
        state.ra.safeRelease(state.diagB);
        state.ra.safeRelease(state.inputs.diagA);
        state.ra.safeRelease(state.inputs.diagB);
    }

    if (hemmUseConjU && left != lower)
        conjU = !conjU;
    if (hemm && !hemmUseConjU && left == lower) {
        if (state.haveCSwap)
            state.finalConjV = true;
        else
            complexConjugateC(problem, strategy, state);
    }

    return true;
}

template <HW hw>
void Generator<hw>::hemmPreprocessU(bool doA, int h0, const RegisterLayout &U_layout, const GRFMultirange &U_regs, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    bool left = problem.left;

    auto Tu_real = U_layout.type().real();
    int r = U_layout.rows(), c = U_layout.cols();

    int nk = left ? c : r;
    int nx = left ? r : c;

    if (Tu_real.size() % 4) stub();

    bool uSplitComplex = U_layout.isSplitComplex();
    int stride = uSplitComplex ? 1 : 2;
    int dwords = Tu_real.size() / 4;
    int nmax = (uSplitComplex ? 2 : 1) * elementsPerGRF(hw, Tu_real);

    // Conjugate far triangle.
    for (int hh = 0; hh < nk; hh++) {
        int h = h0 + hh;
        for (int x = 0; x < nx; ) {
            int n;
            auto Ui_reg = U_layout.find(left ? x : hh, left ? hh : x, U_regs, &n, nullptr, 1);
            auto Ui_region = Ui_reg.ud(dwords - 1)(stride * dwords);

            n = std::min(n, nmax);

            int nconj = std::min(x + n, h) - x;
            if (nconj <= 0) break;

            int simd = roundup_pow2(nconj);

            if (nconj < simd) {
                int prevPow2 = rounddown_pow2(nconj);
                if (is_zero_or_pow2(nconj - prevPow2)) {
                    // Split into two xors instead of loading the flag register.
                    n = nconj = simd = prevPow2;
                }
            }

            if (nconj == simd)
                xor_(nconj, Ui_region, Ui_region, 0x80000000);
            else {
                mov(1, state.flagXMask, (1u << nconj) - 1);
                xor_(simd | state.flagXMask, Ui_region, Ui_region, 0x80000000);
            }

            x += n;
        }
    }

    // Zero out diagonal.
    for (int hh = 0; hh < nk; hh++) {
        int h = h0 + hh;

        int n;
        auto Ui_reg = U_layout.find(left ? h : hh, left ? hh : h, U_regs, &n, nullptr, 1);
        mov(dwords, Ui_reg.ud()(1), 0);
    }
}

// Address increments for symmetric/Hermitian matrices.
template <HW hw>
void Generator<hw>::symmUIncrement(const RegisterLayout &layout, const vector<RegisterRange> &addrs,
                                   const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state,
                                   int h)
{
    if (layout.empty()) return;

    auto Tu = layout.type();
    const auto &U_strategy = layout.addressingStrategy();
    int U_m = layout.rows(), U_n = layout.cols();

    bool left = problem.left;
    auto symmUBlock = state.symmUBlock;
    bool transposeFarTriangle = !isTransposing(U_strategy.accessType);

    int ku_load = left ? U_n       : U_m;
    auto loopX =  left ? LoopM     : LoopN;
    auto ldu =    left ? state.lda : state.ldb;

    auto ldu_ku = lookupIncrement(left ? state.ldaIncrements : state.ldbIncrements, ldu, symmUBlock, strategy, state);

    auto RegisterBlock::* offsetMN = left ? &RegisterBlock::offsetR : &RegisterBlock::offsetC;
    auto RegisterBlock::* offsetK  = left ? &RegisterBlock::offsetC : &RegisterBlock::offsetR;

    auto unrollX = strategy.unroll[loopX];

    auto &addrIncU = state.addrIncU;
    auto addrIncUStride = state.addrIncUStride;
    auto addrIncUConsecutive = state.addrIncUConsecutive;

    if (layout[0].addrShift != 0) stub();
    if (!one_of(U_strategy.accessType, {AccessType::PseudoBlock, AccessType::Scattered, AccessType::ChannelScattered})) stub();

    bool init = h < 0;
    if (init)
        h = 0;
    else
        h = align_down(h, ku_load);

    for (int hh = 0; hh < ku_load; hh += symmUBlock) {
        // Increment all addresses in the range [hh, hh + symmUBlock).
        if (!(init && hh == 0)) for (size_t b = 0; b < layout.blocks(); b++) {
            auto &block = layout[b];
            auto &addr = addrs[b];
            auto GRFaddr = GRFRange(addrs[b].getBase(), addrs[b].getLen());

            int naddr = block.simdSize;
            int consecutive = block.extra;
            if (consecutive % addrIncUConsecutive) stub();
            consecutive /= addrIncUConsecutive;
            if (!(init && hh == 0)) for (size_t b = 0; b < layout.blocks(); b++) {
                auto &block = layout[b];
                auto &addr = addrs[b];
                int naddr = block.simdSize;
                int consecutive = block.extra;
                if (consecutive % addrIncUConsecutive) stub();
                consecutive /= addrIncUConsecutive;
                if (block.*offsetK >= hh && block.*offsetK < hh + symmUBlock) {
                    GRFRange addrSrc;
                    for (size_t b0 = 0; b0 < layout.blocks(); b0++) {
                        if (layout[b0].*offsetMN == block.*offsetMN) {
                            auto kdiff = block.*offsetK - layout[b0].*offsetK;
                            if ((kdiff - symmUBlock) % ku_load == 0) {
                                addrSrc = addrs[b0];
                                break;
                            }
                        }
                    }
                    if (!addrSrc.isValid()) stub();
                    if (U_strategy.base.getModel() == ModelA64) {
                        auto ne = elementsPerGRF(hw, Type::u64);
                        for (int ar = 0; naddr > 0; ar += 2, naddr -= 2 * ne) {
                            auto inc = addrIncU.sub(hw, (block.*offsetMN * addrIncUConsecutive + ar*ne/consecutive) * addrIncUStride, DataType::d);
                            eadd<uint64_t>(std::min(naddr, 2 * ne), addr[ar], inc(addrIncUStride, consecutive, 0), addrSrc[ar], strategy, state);
                        }
                    } else
                        add<uint32_t>(naddr, addr[0], addrIncU.sub(hw, block.*offsetMN * addrIncUStride, DataType::d)(1, consecutive, 0), addrSrc[0]);
                }
            }
        }

        // Reflect address increments on the diagonal.
        auto hReflect = h + hh;
        if (!init) hReflect += ku_load;
        if (hReflect < unrollX) {
            auto inc0 = addrIncU.sub(hw, hReflect * addrIncUStride * addrIncUConsecutive, DataType::d)(addrIncUStride);
            int simd = symmUBlock * addrIncUConsecutive;
            transposeFarTriangle ? mov(simd, inc0, symmUBlock * Tu)
                                 : mov(simd, inc0, symmUBlock > 1 ? ldu_ku : ldu);
        }
    }
}

GEMMSTONE_NAMESPACE_END
