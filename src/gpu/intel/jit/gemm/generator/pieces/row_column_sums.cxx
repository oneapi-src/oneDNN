/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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
#include "generator.hpp"
#include "layout_utils.hpp"
#include "map.hpp"
#include "ngen_object_helpers.hpp"
#include "state_utils.hpp"

using namespace ngen;
using namespace ngen::utils;
using std::vector;

#include "internal/namespace_start.hxx"


// Prepare layout for row/column sum matrices, and any needed auxiliary registers.
template <HW hw>
void BLASKernelGenerator<hw>::makeSumLayout(bool column,
                                            Type Tsrc, const vector<RegisterBlock> &srcLayout,
                                            Type Tdst, vector<RegisterBlock> &dstLayout,
                                            const CommonStrategy &strategy, CommonState &state)
{
    bool canDP4A = (hw >= HW::Gen12LP) && one_of(Tsrc, Type::s8, Type::u8) && one_of(Tdst, Type::s32, Type::u32);
    bool cm = isLayoutColMajor(srcLayout);
    bool hReduce = (column == cm);
    bool needAll1s = false;
    int m, n, cp = 1;

    getLayoutDims(srcLayout, m, n);
    auto &rdim = column ? m : n;

    if (Tsrc.bits() == Tdst.bits())
        cp = srcLayout[0].crosspack;

    if (hReduce) {
        if (canDP4A && hasFullCrosspack(srcLayout, 1)) {
            rdim /= 4;
            needAll1s = true;
            if (rdim & 1) rdim <<= 1; // Ensure dp4a dest offset is even.
        }
    } else {
        if (canDP4A && hasFullCrosspack(srcLayout, 4))
            needAll1s |= (rdim >= 4);
        rdim = 1;
        cp = 1;
    }

    bool partials = canSwizzle(hw, Tdst);
    makeUnbackedRegLayout(Tdst, dstLayout, m, n, cm, cp, 0, 0, partials);

    // Prepare all-1s immediate for dp4a.
    if (needAll1s && state.all1s.isInvalid()) {
        state.all1s = state.ra.alloc_sub(Tdst.ngen(), getHint(HintType::LongTerm, strategy));
        mov(1, state.all1s, 0x01010101);
    }
}

// Accumulate row/column sums.
template <HW hw>
void BLASKernelGenerator<hw>::accumulateSum(bool column,
                                            Type Tsrc, const GRFMultirange &srcRegs, const vector<RegisterBlock> &srcLayout,
                                            Type Tdst, const GRFMultirange &dstRegs, const vector<RegisterBlock> &dstLayout,
                                            const CommonStrategy &strategy, CommonState &state,
                                            int q0, int q1)
{
    bool canDP4A = (hw >= HW::Gen12LP) && one_of(Tsrc, Type::s8, Type::u8) && one_of(Tdst, Type::s32, Type::u32);

    bool cm = isLayoutColMajor(srcLayout);
    if (cm != isLayoutColMajor(dstLayout)) stub();

    int m, n;
    getLayoutDims(srcLayout, m, n);

    // x: consecutive dimension in src; y: strided dimension in src
    auto nx = cm ? m : n;
    auto ny = cm ? n : m;

    int x0 = 0, y0 = 0;
    int x1 = nx, y1 = ny;

    if (q1 >= 0) ((column == cm) ? x1 : y1) = q1;
    if (q0 >= 0) ((column == cm) ? x0 : y0) = q0;

    // Two cases to handle:
    //   hReduce = false:  Good case; no reduction. Sum is vector of size mx1 or 1xn.
    //   hReduce = true:   Bad case; needs reduction later, although with dp4a some reduction can be done now.
    bool hReduce = (column == cm);

    int yinc = 1;
    int reduce = (canDP4A && hReduce) ? 4 : 1;
    if (x0 % reduce || x1 % reduce) stub();

    GRFRange temp;
    Subregister imm;

    for (int y = y0; y < y1; y += yinc) {
        for (int x = x0; x < x1; ) {
            int isrc, jsrc, idst, jdst, nsrc, ndst;
            const RegisterBlock *blockSrc, *blockDst;

            isrc = cm ? x : y;
            jsrc = cm ? y : x;
            if (!hReduce) {
                idst = cm ? x : 0;
                jdst = cm ? 0 : x;
            } else {
                idst = cm ? x / reduce : y;
                jdst = cm ? y : x / reduce;
            }

            Subregister srcBase = findBlockReg(Tsrc, srcLayout, isrc, jsrc, srcRegs, nsrc, blockSrc);
            Subregister dstBase = findBlockReg(Tdst, dstLayout, idst, jdst, dstRegs, ndst, blockDst);
            nsrc = std::min(nsrc, x1 - x);
            int neMax = elementsPerGRF(hw, Tdst) * 2;
            if (Tdst == Type::f32 && Tsrc.paddedSize() < 4)
                neMax /= 2;
            auto ne = std::min({nsrc / reduce, ndst, neMax});

            auto src = srcBase(blockSrc->crosspack);
            auto dst = dstBase(blockDst->crosspack);

            bool hsMatch = (src.getHS() * Tsrc == dst.getHS() * Tdst);
            if (Tsrc == Type::bf16 && Tdst == Type::f32)
                hsMatch = (src.getHS() == 1) && (dst.getHS() == 1);

            if (!canSwizzle(hw, Tsrc) && ne > 1 && (srcBase.getOffset() != dstBase.getOffset() || !hsMatch)) {
                if (temp.isInvalid()) temp = state.ra.alloc_range(2);
                auto srcI = src;
                int tmpHS = std::max<int>(1, (blockDst->crosspack * Tdst) / Tsrc);
                if (Tsrc == Type::bf16 && Tdst == Type::f32)
                    tmpHS = blockDst->crosspack;
                auto tmpBase = temp[0].sub(dst.getByteOffset() / Tsrc.real(), src.getType());
                auto tmp = tmpBase(tmpHS);
                auto tmpI = tmp;
                moveToIntPipe(ne, srcI);
                moveToIntPipe(ne, tmpI);
                mov(ne, tmpI, srcI);
                src = tmp;
                srcBase = tmpBase;
            }

            if (Tsrc == Type::f16 && Tdst == Type::f32 && hw >= HW::Gen12LP) {
                if (temp.isInvalid()) temp = state.ra.alloc_range(2);
                if (src.getHS() < 2) stub();
                auto tmpF = temp[0].sub(src.getByteOffset() / Type::f32, DataType::f)(src.getHS() / 2);
                mov(ne, tmpF, src);
                src = tmpF;
            }

            if (canDP4A) {
                auto srcDP4A = Tsrc.isSigned() ? srcBase.d()(1) : srcBase.ud()(1);
                if (!hReduce && blockSrc->crosspack == 4) {
                    yinc = std::min(y1 - y, 4);
                    if (yinc == 4)
                        dp4a(ne, dst, dst, srcDP4A, state.all1s);
                    else if (yinc == 1)
                        add(ne, dst, srcBase(4), dst);
                    else if (hw == HW::XeHPC) {
                        // Workaround, some issue with dp4a with immediates
                        // TODO: hoist immediate out of inner-loop
                        if (imm.isInvalid()) imm = state.ra.alloc_sub(Tdst.ngen());
                        mov(1, imm, 0x01010101 & ((1 << (yinc * 8)) - 1));
                        dp4a(ne, dst, dst, srcDP4A, imm);
                    } else
                        dp4a(ne, dst, dst, srcDP4A, 0x01010101 & ((1 << (yinc * 8)) - 1));
                } else if (hReduce && blockSrc->crosspack == 1) {
                    if (Tsrc.isSigned())
                        dp4a(ne, dst, dst, srcDP4A, state.all1s);
                    else {
                        // Workaround for suspected HW issue.
                        dst.setType(DataType::ud);
                        dp4a(ne, dst, dst, srcDP4A, state.all1s.ud());
                    }
                }
            } else
                eadd(ne, dst, dst, src, strategy, state);

            x += ne * reduce;
        }
    }

    state.ra.safeRelease(temp);
    state.ra.safeRelease(imm);
}

template <HW hw>
void BLASKernelGenerator<hw>::setupTeardownAccumulateSumSystolic(bool setup, Type T, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto &sysSumAll1s = state.sysSumAll1s;

    if (setup) {
        if (sysSumAll1s.isInvalid()) {
            sysSumAll1s = state.ra.alloc();
            sysSumAll1s.setType(T.ngen());

            int ne = elementsPerGRF(hw, T);
            if (T == Type::s8 || T == Type::u8)
                mov(ne / 4, sysSumAll1s.ud(), uint32_t(0x01010101));
            else if (T == Type::bf16)
                mov(ne, sysSumAll1s.uw(), uint16_t(0x3F80));
            else
                mov(ne, sysSumAll1s.retype(T.arithmetic().ngen()), cast(T.arithmetic(), 1.0));
        }
    } else
        state.ra.safeRelease(sysSumAll1s);
}

// Horizontally add intermediate sums if needed.
template <HW hw>
void BLASKernelGenerator<hw>::horizontalAdd(bool column, Type T, const GRFMultirange &regs, vector<RegisterBlock> &layout, CommonState &state)
{
    bool cm = isLayoutColMajor(layout);
    if (cm != column)
        return;         // Nothing to do.

    int m, n, cp;
    getLayoutDims(layout, m, n);
    cp = layout[0].crosspack;

    int nx = cm ? m : n;
    int ny = cm ? n : m;
    int ne = elementsPerGRF(hw, T);
    bool swizzleOK = canSwizzle(hw, T);

    GRF tempGRF;
    if (!swizzleOK && nx > 1)
        tempGRF = state.ra.alloc();

    int nsLimit = (2 * elementsPerGRF(hw, T)) / cp;

    for (int chunk = roundup_pow2(nx) >> 1; chunk > 0; chunk >>= 1) {
        for (int y = 0; y < ny; y += cp) {
            for (int x = chunk; x < (chunk * 2) && x < nx;) {
                int i = cm ? x : y;
                int j = cm ? y : x;
                int ns, nb;
                const RegisterBlock *block;
                Subregister shifted = findBlockReg(T, layout, i, j, regs, ns, block);

                ns = std::min({ns, chunk, nsLimit});
                (cm ? i : j) -= chunk;
                Subregister base = findBlockReg(T, layout, i, j, regs, nb, block);

                auto dest = base;
                if (chunk == 1)
                    dest = regs[y / ne].sub(y % ne, T.ngen());

                int ne = ns * cp;

                if (!swizzleOK && chunk*cp > 1 && shifted.getOffset() != base.getOffset()) {
                    auto temp = tempGRF.sub(base.getOffset(), T.ngen());
                    auto tempI = temp;
                    auto shiftedI = shifted;
                    moveToIntPipe(tempI);
                    moveToIntPipe(shiftedI);
                    mov(ne, tempI(1), shiftedI(1));
                    if (base == dest)
                        add(ne, base(1), base(1), temp(1));
                    else for (int q = 0; q < ne; q++) {
                        add(1, dest, base, temp);
                        dest.setOffset(dest.getOffset() + 1);
                        base.setOffset(base.getOffset() + 1);
                        temp.setOffset(temp.getOffset() + 1);
                    }
                } else
                    add(ne, dest(1), base(1), shifted(1));

                x += ns;
            }
        }
    }

    state.ra.safeRelease(tempGRF);

    (cm ? m : n) = 1;
    makeUnbackedRegLayout(T, layout, m, n, !cm, 1);
}

// Get final A/B sums. For SLM copy kernels, this requires accumulating each thread's contributions.
template <HW hw>
bool BLASKernelGenerator<hw>::gemmFinalizeSums(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    bool doA = problem.needsASums();
    bool doB = problem.needsBSums();
    bool doASLM = state.slmASums && (strategy.wg[LoopN] > 1);
    bool doBSLM = state.slmBSums && (strategy.wg[LoopM] > 1);

    if (!doA && !doB)
        return true;

    auto Tc = problem.Tc;
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];
    bool ok = true;

    int ms = 0, ns = 0;
    if (doA) getLayoutDims(state.As_layout, ms, ns);
    bool reduceAs = (ns > 1);
    if (doB) getLayoutDims(state.Bs_layout, ms, ns);
    bool reduceBs = (ms > 1);

    if (reduceAs && doA && !doASLM)
        horizontalAdd(false, Tc, state.As_regs, state.As_layout, state);
    if (reduceBs && doB && !doBSLM)
        horizontalAdd(true,  Tc, state.Bs_regs, state.Bs_layout, state);

    if (!doASLM && !doBSLM)
        return true;

    if (state.effCoopA == CoopSplit::Linear || state.effCoopB == CoopSplit::Linear) stub();
    bool A_coopSplitM = (state.effCoopA == CoopSplit::MN);
    bool B_coopSplitN = (state.effCoopB == CoopSplit::MN);

    GRFMultirange *ABs_regs[2]           = {&state.As_regs,     &state.Bs_regs};
    bool AB_coopSplitMN[2]               = {A_coopSplitM,       B_coopSplitN};
    vector<RegisterBlock> *ABs_layout[2] = {&state.As_layout,   &state.Bs_layout};

    vector<RegisterBlock> ABs_layoutSLM[2];
    MatrixAddressing ABs_SLM[2];
    MatrixAddressingStrategy ABs_strategySLM[2];
    MatrixAddressingStrategy ABs_strategySLMAtomic[2];
    vector<GRFRange> ABs_addrs[2];
    GRF temp = state.ra.alloc();
    FlagRegister leader[2];
    Subregister ABs_base[2];

    if (state.r0_info.isARF()) stub();
    GRF r0_info{state.r0_info.getBase()};

    // Plan:
    //   1) First thread of each m/n-block (leader) stores its sums in SLM; barrier
    //   2) Remaining threads atomically add their sums to the first; barrier
    //   3) All threads read final sums
    // For scattered SLM write kernels, threads have accumulated disjoint parts
    //  of the sums, so the second step isn't needed. However, each thread needs
    //  to do a horizontal reduction first.

    // Wait for previous SLM reads to complete.
    // In the meantime, finish sum reduction if necessary.
    status << "Finalize A/B sums" << status_stream::endl;

    if (hw >= HW::Gen11) {
        slmfence(temp, r0_info);
        fencewait();
    }
    MOCK_BARRIERS activeThreadBarrierSignal(temp, r0_info, strategy);

    if (doASLM && A_coopSplitM)
        horizontalAdd(false, Tc, state.As_regs, state.As_layout, state);
    if (doBSLM && B_coopSplitN)
        horizontalAdd(true, Tc, state.Bs_regs, state.Bs_layout, state);

    MOCK_BARRIERS barrierwait();

    auto step1 = [&](bool isB, int r, int c) {
        std::vector<MaskAssignment> masks;

        ABs_SLM[isB].setAlignment(r * c * Tc);
        ABs_SLM[isB].crosspack = 1;
        ABs_SLM[isB].layout = !isB ? MatrixLayout::Pc : MatrixLayout::Pr;
        ABs_SLM[isB].packSize = r * c;
        // Use pseudoblock to share address registers between regular and atomic accesses,
        //  or for non-power-of-2 sizes.
        bool useBlock = AB_coopSplitMN[isB] && is_zero_or_pow2(isB ? c : r);
        ABs_strategySLMAtomic[isB].base = AddressBase::createSLM();
        ABs_strategySLMAtomic[isB].padded = true;
        ABs_strategySLMAtomic[isB].accessType = useBlock ? AccessType::Block
                                                         : AccessType::PseudoBlock;
        ABs_strategySLMAtomic[isB].atomic = !AB_coopSplitMN[isB];
        ABs_strategySLMAtomic[isB].newDP = (hw >= HW::XeHPG);
        ABs_strategySLM[isB] = ABs_strategySLMAtomic[isB];
        ABs_strategySLM[isB].atomic = false;

        int maxRBlock = 0;
        if (hw == HW::Gen12LP && !isB && !A_coopSplitM)
            maxRBlock = 8;      /* Workaround for Gen12LP HW bug with SIMD16 untyped SLM reads */

        ok = ok && getRegLayout(Tc, ABs_layoutSLM[isB], r, c, false, false, true, AvoidFragment,
                                maxRBlock, 0, ABs_SLM[isB], ABs_strategySLMAtomic[isB])
                && matchLayouts(Tc, ABs_layoutSLM[isB], *ABs_layout[isB])
                && assignMasks(ABs_layoutSLM[isB], LoopM, LoopN, masks, strategy, state, false);

        Subregister remainders[3] = {Subregister{}};
        loadMasks(masks, remainders, strategy, state);

        Subregister adjBase = ABs_base[isB] = state.ra.alloc_sub<uint32_t>();
        uint32_t slmOffset = (isB && doASLM) ? (unrollM * strategy.wg[LoopM] * Tc) : 0;

        !isB ? mulConstant(1, ABs_base[isB], state.lidM, unrollM * Tc)
             : mulConstant(1, ABs_base[isB], state.lidN, unrollN * Tc);

        if (strategy.kParallelLocal) {
            slmOffset *= strategy.wg[LoopK];
            int perK = !isB ? strategy.wg[LoopM] * unrollM * Tc
                            : strategy.wg[LoopN] * unrollN * Tc;
            emad(1, ABs_base[isB], ABs_base[isB], state.lidK, perK, strategy, state);
        }

        if (slmOffset != 0)
            add(1, ABs_base[isB], ABs_base[isB], slmOffset);

        if (AB_coopSplitMN[isB]) {
            adjBase = state.ra.alloc_sub<uint32_t>();
            !isB ? mulConstant(1, adjBase, state.lidN, state.ma_slm * Tc)
                 : mulConstant(1, adjBase, state.lidM, state.nb_slm * Tc);
            add(1, adjBase, adjBase, ABs_base[isB]);
        }
        makeSLMBaseRelative(adjBase, state);
        allocAddrRegs(ABs_addrs[isB], ABs_layoutSLM[isB], ABs_SLM[isB], ABs_strategySLMAtomic[isB], state);
        setupAddr(Tc, ABs_addrs[isB], adjBase, ABs_layoutSLM[isB], Subregister(), ABs_SLM[isB], ABs_strategySLMAtomic[isB], strategy, state);
        releaseMaskAssignments(masks, state);

        if (AB_coopSplitMN[isB]) state.ra.safeRelease(adjBase);

        Label labelNoStore;
        if (!AB_coopSplitMN[isB]) {
            leader[isB] = state.raVFlag.alloc();
            cmp(16 | eq | leader[isB], !isB ? state.lidN : state.lidM, 0);
            if_(16 | leader[isB], labelNoStore);
        }
        storeMatrix(*ABs_regs[isB], ABs_layoutSLM[isB], ABs_SLM[isB], ABs_strategySLM[isB], ABs_addrs[isB], strategy, state);
        if (!AB_coopSplitMN[isB]) {
            mark(labelNoStore);
            endif(16);
        }
    };

    bool barrier2 = false;
    auto step2 = [&](bool isB) {
        allocEAtomicAddRegs(hw, Tc, ABs_layoutSLM[isB], ABs_SLM[isB], ABs_strategySLMAtomic[isB], state, state.flagAP);

        Label labelNoAdd;
        if_(16 | ~leader[isB], labelNoAdd);
        atomicAddMatrix(Tc, *ABs_regs[isB], ABs_layoutSLM[isB], ABs_SLM[isB], ABs_strategySLMAtomic[isB], ABs_addrs[isB], problem, strategy, state);
        mark(labelNoAdd);
        endif(16);
        barrier2 = true;

        freeEAtomicAddRegs(state, state.flagAP);
    };

    auto step3 = [&](bool isB, int r, int c) {
        if (AB_coopSplitMN[isB]) {
            safeReleaseRanges(ABs_addrs[isB], state);
            ABs_SLM[isB].packSize = r * c;
            ABs_SLM[isB].setAlignment(r * c * Tc);
            ABs_strategySLM[isB].accessType = AccessType::Block;
            ok = ok && getRegLayout(Tc, ABs_layoutSLM[isB], r, c, false, false, false, AvoidFragment,
                                    0, 0, ABs_SLM[isB], ABs_strategySLM[isB]);

            auto nregs = getRegCount(ABs_layoutSLM[isB]);
            if (nregs > ABs_regs[isB]->getLen()) {
                safeReleaseRanges(*ABs_regs[isB], state);
                *ABs_regs[isB] = state.ra.alloc_range(nregs);
            }

            allocAddrRegs(ABs_addrs[isB], ABs_layoutSLM[isB], ABs_SLM[isB], ABs_strategySLM[isB], state);
            setupAddr(Tc, ABs_addrs[isB], ABs_base[isB], ABs_layoutSLM[isB], Subregister(), ABs_SLM[isB], ABs_strategySLM[isB], strategy, state);
        }
        loadMatrix(*ABs_regs[isB], ABs_layoutSLM[isB], ABs_SLM[isB], ABs_strategySLM[isB], ABs_addrs[isB], strategy, state);
        *ABs_layout[isB] = std::move(ABs_layoutSLM[isB]);
    };

    if (doASLM) step1(false, state.ma_slm, 1);
    if (doBSLM) step1(true,  1, state.nb_slm);

    MOCK_BARRIERS slmBarrier(temp, r0_info, strategy);

    if (doASLM && !A_coopSplitM) step2(false);
    if (doBSLM && !B_coopSplitN) step2(true);

    MOCK_BARRIERS if (barrier2)
        slmBarrier(temp, r0_info, strategy);

    if (doASLM) step3(false, unrollM, 1);
    if (doBSLM) step3(true,  1, unrollN);

    state.ra.safeRelease(temp);
    state.ra.safeRelease(ABs_base[0]);
    state.ra.safeRelease(ABs_base[1]);
    state.raVFlag.safeRelease(leader[0]);
    state.raVFlag.safeRelease(leader[1]);
    safeReleaseRanges(ABs_addrs[0], state);
    safeReleaseRanges(ABs_addrs[1], state);

    return ok;
}

#include "internal/namespace_end.hxx"
