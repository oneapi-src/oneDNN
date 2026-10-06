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


#include "gemmstone/generator.hpp"
#include "hw_utils.hpp"
#include "ngen_object_helpers.hpp"
#include "layout_utils.hpp"
#include "map.hpp"
#include "state_utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;
using std::vector;


// Initialize the interface and claim arguments.
template <HW hw>
void Generator<hw>::copyInitInterface(CopyProblem &problem, CopyStrategy &strategy, CopyState &state)
{
    initInterface(problem, strategy, state);

    if (strategy.barrierFreq > 0)
        interface.requireBarrier();

    interface.finalize();

#if defined(ASM_OUTPUT)
    interface.generateDummyCL(std::cerr);
#endif

    // Get input register assignments.
    state.inputs.S = interface.getArgumentIfExists("S");
    state.inputs.D = interface.getArgumentIfExists("D");
    state.inputs.surfaceS = interface.getArgumentSurfaceIfExists("S");
    state.inputs.surfaceD = interface.getArgumentSurfaceIfExists("D");
    state.inputs.offsetS = interface.getArgument("offset_S");
    state.inputs.offsetD = interface.getArgument("offset_D");
    state.inputs.lds = interface.getArgument("lds");
    state.inputs.ldd = interface.getArgumentIfExists("ldd");
    state.inputs.m = interface.getArgument("m");
    state.inputs.n = interface.getArgument("n");
    state.inputs.alpha_real = interface.getArgumentIfExists("alpha_real");
    state.inputs.alpha_imag = interface.getArgumentIfExists("alpha_imag");
    state.inputs.diag = interface.getArgumentIfExists("diag");
    state.inputs.blockZ = interface.getArgumentIfExists("block_z");

    state.inputs.localIDW = interface.getLocalID(0);
    state.inputs.localSizeW = interface.getLocalSize(0);
    if (strategy.zParallel) {
        state.inputs.localIDZ = interface.getLocalID(1);
        state.inputs.localSizeZ = interface.getLocalSize(1);
    }

    state.inputs.groupIDW = r0.ud(1);
    if (strategy.zParallel)
        state.inputs.groupIDZ = r0.ud(6);

    // Downgrade offset variables to 32-bit for non-A64 accesses.
    if (strategy.S.base.getModel() != ModelA64)
        state.inputs.offsetS = state.inputs.offsetS.d();
    if (strategy.D.base.getModel() != ModelA64)
        state.inputs.offsetD = state.inputs.offsetD.d();

    // For now, reinterpret m/n/ld/diag variables to 32-bit if they are 64-bit.
    state.inputs.m = state.inputs.m.d();
    state.inputs.n = state.inputs.n.d();
    state.inputs.lds = state.inputs.lds.ud();
    if (state.inputs.ldd.isValid())
        state.inputs.ldd = state.inputs.ldd.ud();
    if (state.inputs.diag.isValid())
        state.inputs.diag = state.inputs.diag.d();

    // Claim inputs.
    for (int i = 0; i < r0DWords(hw); i++)
        state.ra.claim(r0.ud(i));

    if (strategy.S.base.isStateless())
        state.ra.claim(state.inputs.S);
    if (strategy.D.base.isStateless())
        state.ra.claim(state.inputs.D);

    state.ra.claim(state.inputs.offsetS);
    state.ra.claim(state.inputs.offsetD);
    state.ra.claim(state.inputs.lds);
    if (state.inputs.ldd.isValid())
        state.ra.claim(state.inputs.ldd);
    state.ra.claim(state.inputs.m);
    state.ra.claim(state.inputs.n);
    if (state.inputs.diag.isValid())
        state.ra.claim(state.inputs.diag);
    if (!problem.alpha.fixed()) {
        state.ra.claim(state.inputs.alpha_real);
        if (problem.Td.isComplex())
            state.ra.claim(state.inputs.alpha_imag);
    }

    state.ra.claim(state.inputs.localIDW);
    state.ra.claim(state.inputs.localSizeW);
    if (strategy.zParallel) {
        state.ra.claim(state.inputs.localIDZ);
        state.ra.claim(state.inputs.localSizeZ);
    }

    if (strategy.zParallel)
        state.ra.claim(state.inputs.blockZ);
}

// Initialize the state structure.
template <HW hw>
void Generator<hw>::copyInitState(CopyProblem &problem, CopyStrategy &strategy, CopyState &state)
{
    if (!state.fusedGEMM.active) {
        initState(problem, strategy, state);
        copyInitInterface(problem, strategy, state);
        state.isNested = false;
    }

    state.effS = strategy.S.base.isStateless() ? state.inputs.S
                                              : state.inputs.offsetS.d();
    state.effD = strategy.D.base.isStateless() ? state.inputs.D
                                              : state.inputs.offsetD.d();

    state.flagAP = state.raVFlag.alloc();

    // Set aside flag registers for triangular masking. These can be used for vflags too.
    if (problem.srcType == MatrixSrcType::Triangular) {
        state.flagTri = state.raVFlag.alloc();
        state.raVFlag.unlock(state.flagTri);
        if (problem.unit || problem.trsm) {
            state.flagDiag = state.raVFlag.alloc();
            state.raVFlag.unlock(state.flagDiag);
        }
    }

    state.allocEmulate64Temp(strategy.emulate);
}

// Copy kernel generation interface.
template <HW hw>
void Generator<hw>::copy(CopyProblem problem, CopyStrategy strategy, const InterfaceHandler &interface_)
{
    interface = interface_;
    CopyState state(hw);
    copy(problem, strategy, state);
}

template <HW hw>
void Generator<hw>::copy(CopyProblem &problem, CopyStrategy &strategy, CopyState &state)
{
    auto Ts = problem.Ts, Td = problem.Td;
    bool inFused = state.fusedGEMM.active;
    auto unrollW = strategy.unrollW();

    // Check layouts.
    if (!isPacked(problem.D.layout)) stub();

    if (strategy.zParallel && problem.sum) stub();

    // By default, don't use dispatch mask.
    setDefaultNoMask();
    setDefaultAutoSWSB();

    // Set up.
#ifdef STANDALONE
    copyKernelNotes(problem, strategy);
#endif
    copyInitState(problem, strategy, state);
#ifdef ASM_OUTPUT
    std::stringstream s;
    interface.dumpAssignments(s);
    comment(s.str());
#endif

    if (!strategy.S.base.isStateless()) strategy.S.base.setIndex(state.inputs.surfaceS);
    if (!strategy.D.base.isStateless()) strategy.D.base.setIndex(state.inputs.surfaceD);

    // Prologue.
    if (!inFused)
        prologue(strategy);

    // Grab fused ID if needed.
    getFusedID(unrollW, problem, strategy, state);

    // Set up constants for inversion subroutine if needed.
    if (problem.trsm && !problem.unit)
        invertSubInit(problem.Ts, false, strategy, state);

    // Convert real offsets to complex offsets for MKL interface.
    // Also negate diag for B copies.
    if (problem.mklOffsets) {
        if (Ts.isComplex()) eshr(1, state.inputs.offsetS, state.inputs.offsetS, 1, strategy, state);
        if (Td.isComplex()) eshr(1, state.inputs.offsetD, state.inputs.offsetD, 1, strategy, state);
        if ((problem.D.layout == MatrixLayout::Pr) && (problem.srcType != MatrixSrcType::General))
            mov(1, state.inputs.diag, -state.inputs.diag);
    }

    // Calculate w0, the starting row/column for this thread.
    // This is the first x (if xloop = false) or y (xloop = true) value.
    state.w0 = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
    if (strategy.zParallel)
        state.z0 = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp1, strategy));

    auto globalIDW = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp1, strategy));
    auto globalIDZ = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));

    int idWScale = inFused ? 1 : strategy.subgroupSize;
    bool multiple = (unrollW % idWScale) == 0;

    if (strategy.wgW > 0)
        mulConstant(1, globalIDW, state.inputs.groupIDW, strategy.wgW * idWScale);
    else
        mul(1, globalIDW, state.inputs.groupIDW, state.inputs.localSizeW.uw());
    if (strategy.zParallel) {
        if (strategy.wgZ > 0)
            mulConstant(1, globalIDZ, state.inputs.groupIDZ, strategy.wgZ);
        else
            mul(1, globalIDZ, state.inputs.groupIDZ, state.inputs.localSizeZ.uw());
    }
    add(1, globalIDW, globalIDW, state.inputs.localIDW.uw(0));
    if (strategy.zParallel && (strategy.wgZ != 1))
        add(1, globalIDZ, globalIDZ, state.inputs.localIDZ.uw(0));
    if (multiple)
        mulConstant(1, state.w0, globalIDW, unrollW / idWScale);
    else {
        mulConstant(1, state.w0, globalIDW, unrollW);
        shr(1, state.w0, state.w0, ilog2(idWScale));
    }
    if (strategy.zParallel)
        emul(1, state.z0, globalIDZ, state.inputs.blockZ, strategy, state);

    state.ra.safeRelease(globalIDW);
    state.ra.safeRelease(globalIDZ);
    state.ra.safeRelease(state.inputs.localIDW);
    state.ra.safeRelease(state.inputs.localIDZ);
    state.ra.safeRelease(state.inputs.localSizeW);
    state.ra.safeRelease(state.inputs.localSizeZ);

    // Move r0 to acc0 if configured.
    moveR0(strategy, state);

    // Copy our slice.
    copySlice(problem, strategy, state);

    if (!inFused) {
        epilogue(strategy, state);

        // Diagonal inversion subroutine.
#ifdef STANDALONE
        if (!problem.readCheck)
#endif
            invertSub(Ts, false, state, true);

        padding();
    }
}

// Calculate or recalculate lds_sl/ldd_dl as needed.
template <HW hw>
void Generator<hw>::copyCalcIncrements(const CopyProblem &problem, const CopyStrategy &strategy, CopyState &state, int s_load, int d_load)
{
    // S: w0 * s_load is needed for N->Pc, T->Pr [!xLoop] N->Pr, T->Pc [xLoop]
    // D: no increment needed (always packed)    [!xLoop] ldd * d_load [xLoop]
    bool sStrided = (isColMajor(problem.S.layout) == isColMajor(problem.D.layout)) ^ strategy.xLoop;

    if (sStrided || problem.reflecting()) {
        if (s_load == 0) s_load = strategy.s_load;
        if (s_load > 1) {
            if (state.lds_sl.isInvalid()) {
                state.lds_sl = state.ra.alloc_sub<uint32_t>();
                s_load *= problem.Ts;
            }
            emulConstant(1, state.lds_sl, state.inputs.lds, s_load, strategy, state);
        }
    }

    if (strategy.xLoop) {
        if (d_load == 0) d_load = strategy.d_load;
        if (d_load > 1) {
            if (state.ldd_dl.isInvalid()) {
                state.ldd_dl = state.ra.alloc_sub<uint32_t>();
                d_load *= problem.Td;
            }
            emulConstant(1, state.ldd_dl, state.inputs.ldd, d_load, strategy, state);
        }
    }
}

// Copy kernel generation interface.
template <HW hw>
void Generator<hw>::copySlice(CopyProblem &problem, CopyStrategy &strategy, CopyState &state)
{
    auto Ts = problem.Ts, Td = problem.Td;
    Label labelExit;
    Subregister lddSrc;

    // For TRSM copy kernels, pad D panel length to multiple of panel width.
    state.D_m = state.inputs.m;
    state.D_n = state.inputs.n;
    auto &D_y = (problem.D.layout == MatrixLayout::Pc) ? state.D_n : state.D_m;
    if (problem.trsm) {
        auto D_yAligned = state.ra.alloc_sub<uint32_t>();
        alignUp(D_yAligned, D_y.ud(), problem.D.packSize, strategy, state);
        D_y = D_yAligned;
    }

    // If ldd not specified, use y.
    // Pad as necessary to meet alignment requirements and store row/column sums.
    if (state.inputs.ldd.isInvalid()) {
        state.inputs.ldd = lddSrc = D_y;
        int align = problem.trsm ? problem.D.packSize : problem.D.crosspack;
        if (align > 1 || problem.sum || strategy.zParallel) {
            state.inputs.ldd = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
            mov(1, state.inputs.ldd, lddSrc);
            lddSrc = invalid;
        }
        if (align > 1)
            alignUp(state.inputs.ldd, state.inputs.ldd, align, strategy, state);
        if (problem.sum)
            add(1, state.inputs.ldd, state.inputs.ldd, problem.Tsum.size() / problem.Td);
    }

    // Duplicate alpha if configured.
    if (strategy.duplicateAlpha) {
        duplicateScalar(state.inputs.alpha_real, state);
        if (Td.isComplex())
            duplicateScalar(state.inputs.alpha_imag, state);
    }

    if (Ts.isComplex() && problem.unit) {
        int ne = 2 * elementsPerGRF<uint32_t>(hw);
        state.complexOne = state.ra.alloc_range(2);
        if (Ts == Type::cf16 || Ts == Type::cbf16)
            mov(ne, state.complexOne[0].ud(), problem.trsm ? 0xBC00 : 0x3C00);
        else {
            mov(ne, state.complexOne[0].ud(), 0);
            if (Ts == Type::cf32 || Ts == Type::ctf32)
                mov(ne / 2, state.complexOne[0].f(0)(2), problem.trsm ? -1.0f : 1.0f);
            else if (Ts == Type::cf64)
                mov(ne / 4, state.complexOne[0].df(0)(2), problem.trsm ? -1.0 : 1.0);
            else stub();
        }
    }

    // For fused kernels, compute 2 * unrollW - fusedID for use in several places.
    Subregister unrollWRem;
    if (strategy.fused) {
        unrollWRem = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
        add(1, unrollWRem, -state.fusedID, uint16_t(2 * strategy.unrollW()));
    }

    // Align code paths.
    bool mLoop = isColMajor(problem.D.layout) == strategy.xLoop;
    auto z   = mLoop ? state.inputs.m : state.inputs.n;
    auto D_z = mLoop ? state.D_m      : state.D_n;
    Subregister z0;

    // Handle z blocking.
    if (strategy.zParallel) {
        z0 = state.z0;
        add(1 | le | f0[1], D_z, D_z, -z0);
        if (z != D_z)
            add(1 | sat, z.ud(), z, -z0);
        min_(1, D_z, D_z, state.inputs.blockZ);
        if (z != D_z)
            min_(1, z, z, state.inputs.blockZ);
        state.ra.safeRelease(state.inputs.blockZ);
    }

    // diag -= m0 and/or diag += n0
    if (problem.srcType != MatrixSrcType::General) {
        add(1, state.inputs.diag, state.inputs.diag, mLoop ? state.w0 : -state.w0);
        if (state.z0.isValid())
            add(1, state.inputs.diag, state.inputs.diag, mLoop ? -state.z0 : state.z0);
    }

    // Triangular kernel handling.
    //   lower n loop: n = min(n, unrollM - new_diag) [2x unroll on first fused thread]
    //   upper m loop: m = min(m, unrollN + new_diag)
    //   lower m loop: m0 = align_down(+new_diag, unrollM); m -= m0; diag -= m0 [adjust new_diag for fused threads]
    //   upper n loop: n0 = align_down(-new_diag, unrollN); n -= n0; diag += n0
    // For fused threads, m/n/m0/n0 must be same for both fused threads
    // For SLM copy TRMM kernels, they must be the same for all slices that are used by a threadgroup
    bool triangular = (problem.srcType == MatrixSrcType::Triangular);

    if (triangular) {
        Subregister twgW0;
        if (problem.targetWG > 1) {
            twgW0 = state.ra.alloc_sub<uint32_t>();
            mod(twgW0, state.w0, problem.targetWG * strategy.unrollX, strategy, state);
        }

        // Copy m/n to ldd before potentially altering it, if ldd not specified.
        if (lddSrc.isValid()) {
            state.inputs.ldd = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
            mov(1, state.inputs.ldd, lddSrc);
        }

        if (mLoop ^ problem.lower) {
            // Triangle at end of copy.
            Subregister temp = state.ra.alloc_sub<int32_t>(getHint(HintType::TempComp0, strategy));
            auto signedDiag = problem.lower ? -state.inputs.diag : +state.inputs.diag;
            auto alignZ = 1;

            if (strategy.fused && strategy.xLoop)
                alignZ = strategy.unrollZ() * 2;

            if (problem.targetWG > 1) {
                if (strategy.xLoop)
                    alignZ = lcm(alignZ, problem.targetWG * strategy.unrollX);
                add(1, temp, signedDiag, problem.targetWG * strategy.unrollX);
                add(1 | le | f0[1], temp, temp, -twgW0);
            } else if (strategy.fused)
                add(1 | le | f0[1], temp, signedDiag, unrollWRem);
            else
                add(1 | le | f0[1], temp, signedDiag, strategy.unrollW());
            if (alignZ > 1)
                alignUp(temp, temp, alignZ, strategy, state);
            min_(1, z, z, temp);
            if (z != D_z)
                min_(1, D_z, D_z, temp);

            state.ra.safeRelease(temp);
        } else {
            // Triangle at beginning of copy.
            auto triZ0 = state.ra.alloc_sub<int32_t>(getHint(HintType::TempComp0, strategy));
            Subregister rem = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp1, strategy));

            auto alignZ = (strategy.fused ? 2 : 1) * strategy.unrollZ();
            auto z0src = problem.lower ? +state.inputs.diag : -state.inputs.diag;

            if (problem.targetWG > 1) {
                if (strategy.xLoop)
                    alignZ = lcm(alignZ, problem.targetWG * strategy.unrollX);
                add(1 | sat, triZ0.ud(), z0src, -twgW0);
            } else if (strategy.fused)
                add(1 | sat, triZ0.ud(), z0src, -state.fusedID);
            else
                mov(1 | sat, triZ0.ud(), z0src);
            alignDown(triZ0, triZ0, alignZ, strategy, state);
            add(1 | le | f0[1], D_z, D_z, -triZ0);
            if (z != D_z)
                add(1, z, z, -triZ0);
            add(1, state.inputs.diag, state.inputs.diag, problem.lower ? -triZ0 : +triZ0);

            state.ra.safeRelease(rem);

            if (z0.isInvalid())
                z0 = triZ0;
            else {
                add(1, z0, z0, triZ0);
                state.ra.safeRelease(triZ0);
            }
        }

        state.ra.safeRelease(twgW0);
    }

    // Compute base addresses for S, D.
    //   S += w0 + z0 * lds (N->Pc, T->Pr) z0 + w0 * lds (N->Pr, T->Pc) [swapped if xLoop = true]
    bool sStrided = (isColMajor(problem.S.layout) == isColMajor(problem.D.layout)) ^ strategy.xLoop;
    auto incC = sStrided ? state.w0 : z0;
    auto incS = sStrided ? z0 : state.w0;

    if (incC.isValid())
        eadd(1, state.inputs.offsetS, state.inputs.offsetS, incC, strategy, state);
    if (incS.isValid()) {
        Subregister temp = state.ra.alloc_sub(state.inputs.offsetS.getType(), getHint(HintType::TempComp1, strategy));
        emul(1, temp, incS, state.inputs.lds, strategy, state);
        eadd(1, state.inputs.offsetS, state.inputs.offsetS, temp, strategy, state);
        state.ra.safeRelease(temp);
    }

    // Quick exit if no work to do.
    if (triangular || strategy.zParallel)
        jmpi(1 | f0[1], labelExit);

    // D += align_up(x0, unroll) * ldd + y0 * unroll + (x0 % unroll) * crosspack
    {
        Subregister temp0 = state.ra.alloc_sub(state.inputs.offsetD.getType(), getHint(HintType::TempComp0, strategy));
        Subregister temp1 = state.ra.alloc_sub(state.inputs.offsetD.getType(), getHint(HintType::TempComp1, strategy));
        Subregister temp2 = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
        auto x0 = strategy.xLoop ? z0 : state.w0;
        auto y0 = strategy.xLoop ? state.w0 : z0;
        bool splitX = strategy.unrollX < problem.D.packSize;

        if (x0.isValid()) {
            if (splitX) {
                modExt(temp2, temp1.ud(), x0, problem.D.packSize, strategy, state);
                emul(1, temp0, temp1.ud(), state.inputs.ldd, strategy, state);
                mulConstant(1, temp2, temp2, problem.D.crosspack);
            } else
                emul(1, temp0, x0, state.inputs.ldd, strategy, state);
        }
        if (y0.isValid())
            emulConstant(1, temp1, y0, problem.D.packSize, strategy, state);
        if (x0.isValid())
            eadd(1, state.inputs.offsetD, state.inputs.offsetD, temp0, strategy, state);
        if (y0.isValid())
            eadd(1, state.inputs.offsetD, state.inputs.offsetD, temp1, strategy, state);
        if (x0.isValid() && splitX)
            eadd(1, state.inputs.offsetD, state.inputs.offsetD, temp2, strategy, state);

        state.ra.safeRelease(temp0);
        state.ra.safeRelease(temp1);
        state.ra.safeRelease(temp2);
    }

    state.ra.safeRelease(z0);
    state.z0 = invalid;

    // S1 (reflected base address) = S + diag - diag * lds [N] S - diag + diag * lds [T]
    if (problem.reflecting()) {
        state.offsetS1 = state.ra.alloc_sub(state.inputs.offsetS.getType(), getHint(HintType::LongTerm, strategy));
        Subregister diag_lds = state.ra.alloc_sub(state.offsetS1.getType(), getHint(HintType::TempComp0, strategy));
        emul(1, diag_lds, state.inputs.diag, state.inputs.lds, strategy, state);
        switch (problem.S.layout) {
            case MatrixLayout::N:
                // Gen12: add3
                eadd(1, state.offsetS1, state.inputs.offsetS, state.inputs.diag, strategy, state);
                eadd(1, state.offsetS1, state.offsetS1, -diag_lds, strategy, state);
                break;
            case MatrixLayout::T:
                eadd(1, state.offsetS1, state.inputs.offsetS, -state.inputs.diag, strategy, state);
                eadd(1, state.offsetS1, state.offsetS1, diag_lds, strategy, state);
                break;
            default: stub();
        }
        state.ra.safeRelease(diag_lds);
    }

    // Calculate increments.
    copyCalcIncrements(problem, strategy, state);

    // Calculate remainders for w loop as needed.
    if (!strategy.xLoop && (strategy.remHandlingX != RemainderHandling::Ignore)) {
        auto x = (problem.D.layout == MatrixLayout::Pc) ? state.inputs.m : state.inputs.n;
        state.remainderX = state.ra.alloc_sub<uint32_t>();
        add(1 | sat, state.remainderX, -state.w0, x);
        if (strategy.remHandlingX == RemainderHandling::Split) {
            if (strategy.fused)
                cmp(1 | lt | state.flagAP, null.ud(), state.remainderX, unrollWRem);
            else
                cmp(1 | lt | state.flagAP, null.ud(), state.remainderX, strategy.unrollX);
        }
        min_(1, state.remainderX, state.remainderX, strategy.unrollX);
    }
    if (strategy.xLoop && (strategy.remHandlingY != RemainderHandling::Ignore)) {
        auto y = (problem.D.layout == MatrixLayout::Pc) ? state.inputs.n : state.inputs.m;
        state.remainderY = state.D_remainderY = state.ra.alloc_sub<uint32_t>();
        add(1 | sat, state.remainderY, -state.w0, y);
        if (D_y != y) {
            state.D_remainderY = state.ra.alloc_sub<uint32_t>();
            add(1 | sat, state.D_remainderY, -state.w0, D_y);
        }
        if (strategy.remHandlingY == RemainderHandling::Split) {
            if (strategy.fused)
                cmp(1 | lt | state.flagAP, null.ud(), state.remainderY, unrollWRem);
            else
                cmp(1 | lt | state.flagAP, null.ud(), state.remainderY, strategy.unrollY);
        }
        min_(1, state.D_remainderY, state.D_remainderY, strategy.unrollY);
        if (D_y != y)
            min_(1, state.remainderY, state.remainderY, strategy.unrollY);
    }

    // Convert lds to bytes.
    emulConstant(1, state.inputs.lds, state.inputs.lds, Ts, strategy, state);

    // Add offsets to base pointers for stateless accesses.
    if (problem.reflecting())
        emulConstant(1, state.offsetS1, state.offsetS1, Ts, strategy, state);
    emulConstant(1, state.inputs.offsetS, state.inputs.offsetS, Ts, strategy, state);
    emulConstant(1, state.inputs.offsetD, state.inputs.offsetD, Td, strategy, state);

    if (strategy.S.base.isStateless()) {
        if (problem.reflecting()) {
            state.effS1 = state.ra.alloc_sub(state.inputs.S.getType(), getHint(HintType::LongTerm, strategy));
            eadd(1, state.effS1, state.inputs.S, state.offsetS1, strategy, state);
            state.ra.safeRelease(state.offsetS1);
        }
        eadd(1, state.inputs.S, state.inputs.S, state.inputs.offsetS, strategy, state);

        state.ra.safeRelease(state.inputs.offsetS);
    } else
        state.effS1 = state.offsetS1;

    if (strategy.D.base.isStateless()) {
        eadd(1, state.inputs.D, state.inputs.D, state.inputs.offsetD, strategy, state);
        state.ra.safeRelease(state.inputs.offsetD);
    }

    state.ra.safeRelease(unrollWRem);

    if (!copyBody(problem, strategy, state)) {
        lastException ? std::rethrow_exception(lastException)
                      : stub("Could not generate kernel.");
    }

    mark(labelExit);
}

// Wrapper around copyBodyRemCheck, checking for optimally-aligned S.
template <HW hw>
bool Generator<hw>::copyBody(CopyProblem &problem, CopyStrategy &strategy, CopyState &state)
{
    if (!is_zero_or_pow2(strategy.optionalAlignS))
        stub();

    bool success;

    if (strategy.optionalAlignS == 0)
        success = copyBodyRemCheck(problem, strategy, state);
    else {
        Label labelUnaligned, labelEnd;

        status << "S alignment check" << status_stream::endl;
        and_(1 | nz | f0[1], null.uw(), state.effS.uw(), uint16_t(strategy.optionalAlignS - 1));
        and_(1 | nz | f1[1], null.uw(), state.inputs.lds.uw(), uint16_t(strategy.optionalAlignS - 1));
        ejmpi(1 | f0[1] | anyv, labelUnaligned);

        auto modProblem = problem;
        modProblem.S.setAlignment(strategy.optionalAlignS);

        status << "S aligned to " << strategy.optionalAlignS << ':' << status_stream::endl;
        success = copyBodyRemCheck(modProblem, strategy, state);

        if (state.isNested)
            jmpi(1, labelEnd);
        else
            epilogue(strategy, state);

        mark(labelUnaligned);

        status << "S unaligned" << status_stream::endl;
        success = success && copyBodyRemCheck(problem, strategy, state);

        mark(labelEnd);
    }

    return success;
}

// Wrapper around copyBodyInternal, handling split remainders.
template <HW hw>
bool Generator<hw>::copyBodyRemCheck(CopyProblem &problem, CopyStrategy &strategy, CopyState &state)
{
    auto CopyStrategy::*remHandlingW = (strategy.xLoop ? &CopyStrategy::remHandlingY : &CopyStrategy::remHandlingX);
    bool wSplit = strategy.*remHandlingW == RemainderHandling::Split;
    bool success;

    if (!wSplit)
        success = copyBodyInternal(problem, strategy, state);
    else {
        CopyStrategy modStrategy = strategy;
        Label wRemBegin, wRemEnd;
        jmpi(1 | state.flagAP, wRemBegin);

        status << "Generating " << "xy"[strategy.xLoop] << " non-remainder kernel" << status_stream::endl;
        modStrategy.*remHandlingW = RemainderHandling::Ignore;
        success = copyBodyInternal(problem, modStrategy, state);

        if (state.isNested)
            jmpi(1, wRemEnd);
        else
            epilogue(strategy, state);

        modStrategy.*remHandlingW = RemainderHandling::KnownRemainder;

        bool recalc = false;

        if (strategy.xLoop && !strategy.doubleMasking && !isTransposing(modStrategy.D.accessType) && !isLargeCrosspack(problem.Td, problem.D.crosspack)) {
            // Change D access to use scattered stores so masking is possible.
            modStrategy.D.accessType = AccessType::Scattered;
            modStrategy.S.accessType = isTransposing(modStrategy.S.accessType) ? AccessType::Block : AccessType::Scattered;
        }
        if (!strategy.xLoop && !strategy.S.padded) {
            // Check if we need to change s_load/d_load.
            if (strategy.s_load > strategy.s_load_masked) {
                status << "Downgrading s_load: " << strategy.s_load << " -> " << strategy.s_load_masked << status_stream::endl;
                modStrategy.s_load = strategy.s_load_masked;
                recalc = true;
            }
            if (strategy.d_load > strategy.d_load_masked) {
                status << "Downgrading d_load: " << strategy.d_load << " -> " << strategy.d_load_masked << status_stream::endl;
                modStrategy.d_load = strategy.d_load_masked;
                recalc = true;
            }
        }

        status << "Generating " << "xy"[strategy.xLoop] << " remainder kernel" << status_stream::endl;
        mark(wRemBegin);
        if (recalc)
            copyCalcIncrements(problem, modStrategy, state);
        success = success && copyBodyInternal(problem, modStrategy, state);
        mark(wRemEnd);
    }

    return success;
}

// Body of copy kernel.
template <HW hw>
bool Generator<hw>::copyBodyInternal(CopyProblem &problem, CopyStrategy &strategy, CopyState &state)
{
    Label lZLoopBegin, lZLoopEnd;
    constexpr auto SD_copies = 1;
    vector<MaskAssignment> masks, D0_yMasks;
    bool share;

    auto Ts = problem.Ts, Td = problem.Td, Tsum = problem.Tsum;
    const bool byColumn = isColMajor(problem.D.layout);
    const bool sStrided = (isColMajor(problem.S.layout) == isColMajor(problem.D.layout)) ^ strategy.xLoop;
    const bool xLoop = strategy.xLoop;
    const bool mLoop = isColMajor(problem.D.layout) == xLoop;

    auto unrollZ = strategy.unrollZ();

    const bool reflecting = problem.reflecting();
    const bool hermitian = (problem.srcType == MatrixSrcType::Hermitian);
    const bool triangular = (problem.srcType == MatrixSrcType::Triangular);
    const bool triEnd = triangular && (mLoop != problem.lower);
    const bool triRemOnly = triEnd && !xLoop && (problem.targetWG == 1);

    auto crosspack = problem.D.crosspack;

    // Release w0 -- no longer needed.
    state.ra.safeRelease(state.w0);

    // Get flag register for complex swizzles for XeHP+.
    if (hw >= HW::XeHP && Ts.isComplex()) {
        state.flagSwizzle = state.raVFlag.alloc();
        state.raVFlag.unlock(state.flagSwizzle);
    }

    MatrixAddressingStrategy S_strategyReflected = strategy.S;
    RegisterLayout S_layoutReflected;

    // Prepare for reflecting (symmetric/Hermitian) copies.
    S_strategyReflected.accessType = isTransposing(strategy.S.accessType) ? AccessType::PseudoBlock : AccessType::Scattered;

    if (reflecting || triangular) {
        auto s = triangular ? Ts.isComplex()
                            : ilog2(std::max<int>(Ts.size() >> 3, 1));
        state.indexVecRT = state.ra.alloc(getHint(HintType::LongTerm, strategy)).uw();
        mov(8, state.indexVecRT[0](1), Immediate::uv(0>>s,1>>s,2>>s,3>>s,4>>s,5>>s,6>>s,7>>s));
        mov(8, state.indexVecRT[8](1), Immediate::uv(8>>s,9>>s,10>>s,11>>s,12>>s,13>>s,14>>s,15>>s));

        if (reflecting)
            state.flagReflect = state.raVFlag.alloc();
    }

    // Prepare S and D addressing.
    if (reflecting) {
        if (strategy.S.accessType == AccessType::Block)
            strategy.S.accessType = AccessType::PseudoBlock;
        problem.S.alignment = Ts.paddedSize();
    }

    // Decide what remainder handling needs to be done.
    bool remainderX = (strategy.remHandlingX != RemainderHandling::Ignore);
    bool remainderY = (strategy.remHandlingY != RemainderHandling::Ignore);
    bool remainderZ = xLoop ? remainderX : remainderY;
    bool D_paddingRem = xLoop && remainderY && (state.remainderY != state.D_remainderY);
    bool D_paddingLoop = !xLoop && problem.trsm;

    bool checkYRem1 = xLoop && remainderY && strategy.unrollY == 1;
    VirtualFlag flagYRem1;

    remainderY &= !checkYRem1;

    // Find and load any needed mask registers.
    auto assignAllMasks = [&]() -> bool {
        bool ok = assignMasks(state.S_layout, LoopM, LoopN, masks, strategy, state)
               && assignMasks(state.D_layout, LoopM, LoopN, masks, strategy, state);
        if (ok && D_paddingRem) {
            ok &= state.vflagsEnabled();
            state.D0_layout = state.D_layout;
            for (auto &block: state.D0_layout)
                block.flag[byColumn].clear();
            ok &= assignMasks(state.D0_layout, byColumn ? LoopNone : LoopM,
                                               byColumn ? LoopN : LoopNone, D0_yMasks, strategy, state);
        }
        return ok;
    };

    // Get register layouts for S and D.
    int nms, nmd, nns, nnd;
    auto setup = [&](int s_load, int d_load, Subregister S_addr0, Subregister S1_addr0, Subregister D_addr0, bool handleRemZ) -> bool {
        bool remM = remainderX && (!xLoop || handleRemZ);
        bool remN = remainderY &&  (xLoop || handleRemZ);
        Subregister remainders[3]   = {state.remainderX, state.remainderY,   Subregister{}};
        Subregister D_remainders[3] = {state.remainderX, state.D_remainderY, Subregister{}};

        if (!xLoop) {
            nmd = nms = strategy.unrollX;
            nnd = d_load;
            nns = s_load;
        } else {
            nnd = nns = strategy.unrollY;
            nmd = d_load;
            nms = s_load;
        }

        if (!byColumn) {
            std::swap(nms, nns);
            std::swap(nmd, nnd);
            std::swap(remM, remN);
            std::swap(remainders[0], remainders[1]);
            std::swap(D_remainders[0], D_remainders[1]);
        }

        auto remM_S = remM && !strategy.S.padded;
        auto remN_S = remN && !strategy.S.padded;
        auto remM_D = remM && !strategy.D.padded && !byColumn;
        auto remN_D = remN && !strategy.D.padded &&  byColumn;

        bool sCMInRegs = isRegisterColMajor(Ts, problem.S, strategy.S);
        auto sMaxRBlock = (reflecting && !sCMInRegs) ? 1 : 0;
        auto sMaxCBlock = (reflecting &&  sCMInRegs) ? 1 : 0;

        if (!(state.S_layout = RegisterLayout::tryCreate(hw, Ts, nms, nns, problem.S, strategy.S, remM_S, remN_S, false, AvoidFragment, sMaxRBlock, sMaxCBlock))) return false;
        if (!(state.D_layout = RegisterLayout::tryCreate(hw, Td, nmd, nnd, problem.D, strategy.D, remM_D, remN_D, true))) return false;
        if (reflecting) {
            if (!(S_layoutReflected = RegisterLayout::tryCreate(hw, Ts, nns, nms, problem.S, S_strategyReflected, remN_S, remM_S, false, AvoidFragment, sMaxCBlock, sMaxRBlock, true))) return false;
        }

        // Check for unsupported remainder handling: fragmenting and z-masking.
        if (state.S_layout.hasFragmenting() || state.D_layout.hasFragmenting()) {
            status << "Fragmenting not supported." << status_stream::endl;
            return false;
        }

        auto RegisterBlock::* zMask = mLoop ? &RegisterBlock::rowMask : &RegisterBlock::colMask;
        auto nzs = mLoop ? nms : nns;
        auto nzd = mLoop ? nmd : nnd;
        if (nzs < unrollZ) for (const auto &block: state.S_layout)
            if (block.*zMask) return false;
        if (nzd < unrollZ) for (const auto &block: state.D_layout)
            if (block.*zMask) return false;

        bool success = true;
        if (checkYRem1) {
            flagYRem1 = state.allocVFlag(hw);
            success &= !(state.raVFlag.isVirtual(flagYRem1) && !state.vflagsEnabled());
        }

        success = success && assignAllMasks();
        if (!success && !state.vflagsEnabled()) {
            status << "Retrying with virtual flags." << status_stream::endl;
            allocVFlagStorage(strategy, state, false);
            success = assignAllMasks();
        }

        if (!success)
            return false;

        loadMasks(masks,       remainders, strategy, state);
        loadMasks(D0_yMasks, D_remainders, strategy, state);

        if (D_paddingRem) {
            // Subtract regular y masks from D0 y masks to get the parts that need zeroing.
            for (const auto &a: D0_yMasks) {
                VirtualFlag vflag;
                for (size_t i = 0; i < state.D_layout.blocks(); i++) {
                    if (state.D0_layout[i].flag[byColumn] == a.flag) {
                        vflag = state.D_layout[i].flag[byColumn];
                        break;
                    }
                }
                if (!vflag) continue;
                auto flag = getMaskFlag(hw, vflag, state);
                auto flag0 = getMaskFlag(hw, a.flag, state);
                and_(1, flag0, flag0, ~flag);
            }
        }

        if (!xLoop && strategy.remHandlingX != RemainderHandling::Ignore) {
            // Find a mask to use for destination layout for y loop remainders.
            VirtualFlag flag;
            bool found = false;
            for (auto &mask : masks)
                if (mask.var == (byColumn ? LoopM : LoopN) && mask.offset == 0)
                    flag = mask.flag, found = true;
            if (!found) stub();
            for (auto &block : state.D_layout) {
                if (block.hasFlag()) return false;
                if (block.simdSize > 16) stub();
                block.flag[0] = flag;
                block.flagAny = true;
            }
        } else if (checkYRem1) {
            // Create mask for y remainder for x-loop kernels with unrollY == 1, and
            //   apply it by hand to both source and destination.
            // If D has padding (TRSM), then there are two masks and we stuff them
            //   both in a single flag register and use any/all tricks to disambiguate.
            RegData regYRem1 = getMaskFlag(hw, flagYRem1, state);
            FlagRegister testFlag;

            testFlag = regYRem1.isARF() ? reinterpret_cast<FlagRegister &>(regYRem1) : f0[1];

            InstructionModifier modRemY;

            cmp(16 | gt | testFlag, state.remainderY, 0);
            if (D_paddingRem) {
                cmp(1 | ~testFlag | le | testFlag, state.D_remainderY, 0);
                modRemY  = modRemY  | (hw >= HW::XeHPC ? all : all16h);
            }

            for (auto &mask : masks)
                mov(1 | ~testFlag | modRemY, getMaskFlag(hw, mask.flag, state), 0);

            if (!regYRem1.isARF())
                mov(1, regYRem1, testFlag);

            for (auto &block : state.S_layout)
                if (!block.hasFlag()) {
                    block.flag[0] = flagYRem1;
                    block.flagAll = D_paddingRem;
                }
            for (auto &block : state.D_layout) {
                if (block.simdSize > 16) stub();
                block.flag[0] = flagYRem1;
                block.flagAll = D_paddingRem;
            }
            for (auto &block : state.D0_layout) {
                if (block.simdSize > 16) stub();
                block.flag[0] = flagYRem1;
                block.flagAny = D_paddingRem;
                block.flagInvert = true;
            }
        }

        // Match source layout to destination layout if possible, so that they can share registers.
        share = (Ts == Td)
             && (!Td.isComplex() || problem.alpha.fixed())
             && (s_load == d_load)
             && matchBidirectional(state.S_layout, state.D_layout);

        // Allocate address registers.
        auto S_hint = getHint(share ? HintType::DAddr : HintType::SAddr, strategy);
        allocAddrRegs(hw, state.S_addrs, state.S_layout, state, S_hint);
        if (reflecting) {
            allocAddrRegs(hw, state.S_addrSrcs[0], state.S_layout,    state, S_hint);
            allocAddrRegs(hw, state.S_addrSrcs[1], S_layoutReflected, state, S_hint);
        }
        allocAddrRegs(hw, state.D_addrs, state.D_layout, state, getHint(HintType::DAddr, strategy));

        // Set up address registers.
        if (reflecting) {
            setupAddr(state.S_addrSrcs[0], S_addr0,  state.S_layout,    state.inputs.lds, strategy, state);
            setupAddr(state.S_addrSrcs[1], S1_addr0, S_layoutReflected, state.inputs.lds, strategy, state);
        } else
            setupAddr(state.S_addrs, S_addr0, state.S_layout, state.inputs.lds, strategy, state);
        setupAddr(state.D_addrs, D_addr0, state.D_layout, state.inputs.ldd, strategy, state);

        // Allocate data registers.
        int S_regCount = state.S_layout.regs();
        int D_regCount = state.D_layout.regs();
        if (reflecting && (S_regCount != S_layoutReflected.regs())) stub();

        state.D_regs = state.ra.alloc_range(D_regCount, getHint(HintType::D, strategy));
        state.S_regs = share ? state.D_regs
                             : state.ra.alloc_range(S_regCount, getHint(HintType::S, strategy));

        if (D_paddingRem) {
            // Get enough zeros for the largest block of D0 and point all blocks to it.
            int D0_regCount = state.D0_layout.maxLoadBlock();
            state.D0_regs = state.ra.try_alloc_range(D0_regCount);
            if (state.D0_regs.isValid()) {
                for (auto &block: state.D0_layout)
                    block.offsetBytes = 0;
                zeroMatrix(state.D0_regs, strategy);
            }
        }

        // Prepare for summation.
        // Clean up previous sums if any, and try to reuse their registers.
        // Allocate and zero new sum registers as needed.
        if (problem.sum) {
            if (xLoop) stub();

            RegisterLayout Ds_layout;
            makeSumLayout(!byColumn, state.D_layout, Tsum, Ds_layout, strategy, state);

            bool alloc = state.Ds_layout.empty() || !Ds_layout.match(state.Ds_layout);
            if (!state.Ds_layout.empty() && alloc) {
                horizontalAdd(!byColumn, state.Ds_regs.back(), state.Ds_layout, state);
                alloc = !Ds_layout.match(state.Ds_layout);
            }
            if (alloc) {
                state.Ds_layout = std::move(Ds_layout);
                auto Ds_regs = state.ra.allocRange(state.Ds_layout.regs());
                zeroMatrix(Ds_regs, strategy);
                state.Ds_regs.push_back(Ds_regs);
            }
        }
#ifdef STANDALONE
        if (problem.readCheck)
            state.readFailures = state.D_regs[0].ud(0);
#endif

        return true;
    };

    auto cleanup = [&]() {
        state.raVFlag.safeRelease(flagYRem1);
        safeReleaseMaskAssignments(masks, state);
        safeReleaseRanges(state.S_addrs, state);
        safeReleaseRanges(state.D_addrs, state);

        state.ra.safeRelease(state.S_regs);
        state.ra.safeRelease(state.D_regs);
        state.ra.safeRelease(state.D0_regs);
        // Sum registers not freed here.

        state.S_layout.clear();
        state.D_layout.clear();
        state.D0_layout.clear();
        S_layoutReflected.clear();
    };

    auto doSLoad = [&](const RegisterLayout &layout, const RegisterLayout &layoutReflect, const vector<RegisterRange> &addrs, const vector<RegisterRange> (&addrSrcs)[2], int z0, int s_load, int S_copy, bool checkRem) {
        bool unlockAP = false;
        Label skipLoad;
        checkRem &= (z0 > 0);

#ifdef STANDALONE
        if (problem.mock.sLoad) return;

        if (problem.readCheck) readCheckInitMatrix(state.S_regs, strategy);
#endif

        if (checkRem) {
            zeroMatrix(state.S_regs, strategy);
            unlockAP = !state.raVFlag.lock(state.flagAP);
            state.usePhysicalFlag(state.flagAP, hw);
            cmp(1 | le | state.flagAP, state.Z, uint16_t(z0));
            jmpi(1 | state.flagAP, skipLoad);
        }

        if (reflecting)
            copyLoadReflecting(state.S_regs, layout, problem.S, strategy.S, addrs, addrSrcs, nms, nns, problem, strategy, state);
        else {
#ifdef STANDALONE
            loadMatrix(state.S_regs, layout, addrs, strategy, state, problem.readCheck);
#else
            loadMatrix(state.S_regs, layout, addrs, strategy, state);
#endif
        }

        auto addrsFixed    = reflecting ? &addrSrcs[0] : &addrs;
        auto addrsStrided  = reflecting ? &addrSrcs[1] : nullptr;
        auto layoutFixed   = &layout;
        auto layoutStrided = &layoutReflect;

        if (sStrided) {
            std::swap(addrsFixed, addrsStrided);
            std::swap(layoutFixed, layoutStrided);
        }

#ifdef STANDALONE
        if (!problem.mock.sInc)
#endif
        {
            if (addrsStrided)
                incAddr(*addrsStrided, (s_load == 1) ? state.inputs.lds : state.lds_sl, *layoutStrided, strategy, state);
            if (addrsFixed)
                incAddr(*addrsFixed, uint16_t(s_load * Ts), *layoutFixed, strategy, state);
        }
#ifdef STANDALONE
        if (problem.readCheck)
            readCheck(state.S_regs, strategy, state);
#endif
        if (checkRem) {
            if (unlockAP)
                state.raVFlag.unlock(state.flagAP);
            mark(skipLoad);
        }
    };

    auto doDStore = [&](const RegisterLayout &layout, const RegisterLayout &layout0, const vector<RegisterRange> &addrs, int d_load, int D_copy) {
#ifdef STANDALONE
        if (problem.mock.dStore) return;
#endif
        storeMatrix(state.D_regs, layout, addrs, strategy, state);
        if (!layout0.empty()) {
            auto *regs0 = &state.D0_regs;
            if (regs0->isInvalid()) {
                regs0 = &state.D_regs;
                zeroMatrix(state.D_regs, strategy);
            }
            storeMatrix(*regs0, layout0, addrs, strategy, state);
        }
        if (problem.sum)
            accumulateSum(!byColumn, state.D_regs, layout, state.Ds_regs.back(), state.Ds_layout, strategy, state);
        if (xLoop) {
            if (d_load >= strategy.unrollX)
                incAddr(addrs, state.ldd_dl, layout, strategy, state);
            else
                incAddr(addrs, uint16_t(d_load * Td), layout, strategy, state);
        } else {
            auto D_tileX = byColumn ? problem.D.tileR : problem.D.tileC;
            auto D_tileY = byColumn ? problem.D.tileC : problem.D.tileR;
            auto effPS = (d_load < D_tileY) ? D_tileX : problem.D.packSize;
            incAddr(addrs, uint16_t(d_load * effPS * Td), layout, strategy, state);
        }
    };

    // Start generating code.

    // Reuse z for the loop counter.
    // If z unroll > 1, the loop counter will be offset by (unrollZ - 1) during the main loop,
    //  unless there's no z remainder.
    // For triangular-ended copies, offset by an additional unrollW [2x unrollX if fused] to push triangular handling to remainder loop.
    state.Z = mLoop ? state.inputs.m : state.inputs.n;

    Subregister Z_padding;
    if (D_paddingLoop) {
        // ... and reuse D_z for the padding loop counter.
        auto &D_z = mLoop ? state.D_m : state.D_n;
        Z_padding = D_z;
        add(1, Z_padding, D_z, -state.Z);
    }

    auto offsetZ = (remainderZ || triRemOnly) ? (unrollZ - 1) : 0;

    auto unrollW = strategy.unrollW();
    if (triRemOnly)
        offsetZ += (strategy.fused ? (2 * strategy.unrollX) : unrollW);

    if (offsetZ == 0)
        cmp(1 | le | state.flagAP, null.d(), state.Z, int16_t(0));
    else
        add(1 | le | state.flagAP, state.Z, state.Z, int16_t(-offsetZ));

    // For conjugate copies, prepare a register with alternating 0 and 1 sign bits for negating the imaginary part.
    // Only needed when not performing complex scaling.
    // Also need this for hermitian copies.
    //   TODO: reduce register usage.
    bool alphaN1 = (problem.alpha == -1);
    auto ne32 = elementsPerGRF<uint32_t>(hw);
    if ((problem.conjugate && problem.alpha.fixed()) || alphaN1 || hermitian) {
        if (alphaN1 && hermitian) stub();
        bool conj = problem.conjugate || hermitian, neg = alphaN1;

        state.signChange = state.ra.alloc();
#ifdef STANDALONE
        if (problem.readCheck)
            mov<uint32_t>(ne32, state.signChange, 0);
        else
#endif
        switch (Ts.real()) {
            case Type::bf16:
            case Type::f16: mov<uint32_t>(ne32, state.signChange, uint32_t((conj ? 0x80000000 : 0) ^ (neg ? 0x80008000 : 0))); break;
            case Type::f32: mov<uint16_t>(ne32 * 2, state.signChange, Immediate::uv(0, neg, 0, neg ^ conj, 0, neg, 0, neg ^ conj));
                            shl<uint32_t>(ne32, state.signChange, state.signChange, 15); break;
            case Type::f64: mov<uint16_t>(ne32 * 2, state.signChange, Immediate::uv(0, 0, 0, neg, 0, 0, 0, neg ^ conj));
                            shl<uint32_t>(ne32, state.signChange, state.signChange, 15); break;
            default: stub();
        }
    }
    if (hermitian) {
        state.selectImag = state.ra.alloc();
        switch (Ts.real()) {
            case Type::bf16:
            case Type::f16: mov<uint32_t>(ne32, state.selectImag, uint32_t(0xFFFF0000)); break;
            case Type::f32: asr<int32_t>(ne32, state.selectImag, state.signChange, uint16_t(31)); break;
            case Type::f64:
                if (strategy.emulate.emulate64) {
                    asr<int32_t>(ne32 >> 1, state.selectImag[0](2), state.signChange[1](2), 31);
                    asr<int32_t>(ne32 >> 1, state.selectImag[1](2), state.signChange[1](2), 31);
                } else
                    asr<int64_t>(ne32 >> 1, state.selectImag, state.signChange, uint16_t(63));
                break;
            default: stub();
        }
    }

    // Get flag register and loop counter for barrier check if needed.
    FlagRegister flagBarrier;
    Subregister bcount;
    if (strategy.barrierFreq > 0) {
        flagBarrier = state.raVFlag.alloc();

        // Can use main loop counter if barrierFreq and unrollZ both powers of 2.
        if (!is_zero_or_pow2(strategy.barrierFreq * unrollZ)) {
            bcount = state.ra.alloc_sub<uint32_t>();
            mov(1, bcount, uint16_t(strategy.barrierFreq));
        }
    }

    // Setup for main loop.
    if (!setup(strategy.s_load, strategy.d_load, state.effS, state.effS1, state.effD, false))
        return false;

#ifdef ASM_OUTPUT
    std::stringstream s;
    state.ra.dump(s);
    comment(s.str());
#endif

    bool lateZLoopCheck = state.vflagsEnabled();
    if (lateZLoopCheck) {
        // Release flags for use by vflags. Note flagReflect is not released.
        state.raVFlag.unlock(state.flagAP);
        if (flagBarrier.isValid())
            state.raVFlag.unlock(flagBarrier);
    }

    // Bail to remainder loop if no main loops.
    jmpi(1 | state.flagAP, lZLoopEnd);

    // Loop check code.
    auto zLoopCheck = [&](int unrollZ, bool enableBarriers) {
        // Use the all-purpose flag for z loop query.
        add(1 | gt | state.flagAP, state.Z, state.Z, int16_t(-unrollZ));

        // Check for barrier if requested.
        if (enableBarriers) {
            if (bcount.isInvalid())
                and_(1 | ze | flagBarrier, null.ud(), state.Z, uint16_t(unrollZ * strategy.barrierFreq - unrollZ));
            else
                add(1 | ze | flagBarrier, bcount, bcount, int16_t(-1));
        }
    };

    // Lambdas used in zLoopBody (moved outside to w/a GCC bug)
    auto mulAlpha = [&](int esize, RegData r) {
        mul(esize, r, r, state.inputs.alpha_real.getRegAvoiding(hw, r));
    };

    auto mulAlphaFixed = [&](int esize, RegData r) {
        mul(esize, r, r, cast(Ts.real(), problem.alpha));
    };

    auto signChange = [&](int esize, RegData r) {
        auto ne = elementsPerGRF<uint32_t>(hw);
        xor_<uint32_t>(esize, r, r, (ne < esize) ? state.signChange[0](0, ne, 1)
                                                 : state.signChange[0](1));
    };

    // z loop: returns true on success.
    int S_copy = 0, D_copy = 0;
    auto zLoopBody = [&](const RegisterLayout &S_layout, const RegisterLayout &S_layoutReflected, const RegisterLayout &D_layout, const RegisterLayout &D0_layout,
                         const vector<RegisterRange> &S_addrs, const vector<RegisterRange> (&S_addrSrcs)[2], const vector<RegisterRange> &D_addrs,
                         int unrollZ, int s_load, int d_load, bool enableBarriers, bool enableTri, bool needSRem = false, bool noLoop = false) {
        int us = s_load, ud = 0;
        int uZLoopCheck = noLoop ? -1 :
                  lateZLoopCheck ? (unrollZ - 1)
                                 : 0;
        bool dMasked = D_layout.hasMasking();

        for (int u = 0; u < unrollZ; u++, us++, ud++) {
            // Maintain us (= u % s_load) and ud (= u % d_load) counters.
            bool loadS = false;
            if (us == s_load) {
                us = 0;
                loadS = true;
            }

            if (ud == d_load)
                ud = 0;
            bool storeD = ((ud + 1) == d_load);

            // Test loop counter on first iteration (lateZLoopCheck == false)
            if ((u == uZLoopCheck) && !lateZLoopCheck)
                zLoopCheck(unrollZ, enableBarriers);

            // Load S every s_load loops, and copy as necessary.
            if (loadS) {
                doSLoad(S_layout, S_layoutReflected, S_addrs, S_addrSrcs, u, s_load, S_copy, needSRem);

                // Mask/transform data for triangular copies.
                if (enableTri && (problem.srcType == MatrixSrcType::Triangular))
                    copyTriangularMask(state.S_regs, S_layout, nms, nns, mLoop, problem, strategy, state);
                else if (triEnd)
                    add(1, state.inputs.diag, state.inputs.diag, int16_t(mLoop ? -nms : +nns));

                // Copy S registers to D registers, or perform in-place scaling/transposition.
                if (!share) {
                    int dOffR = 0, dOffC = 0;
                    (byColumn ? dOffC : dOffR) = ud;

                    copyRegisters(Ts, Td, S_layout, D_layout, state.S_regs, state.D_regs, dOffR, dOffC,
                                  problem.alpha, state.inputs.alpha_real, state.inputs.alpha_imag, problem.conjugate,
                                  strategy, state);
                } else
#ifdef STANDALONE
                if (!problem.readCheck)
#endif
                {
                    if (!problem.alpha.fixed())
                        map(hw, Ts.real(), state.S_regs, S_layout, strategy, mulAlpha);
                    else if ((problem.alpha != 1) && (problem.alpha != -1))
                        map(hw, Ts.real(), state.S_regs, S_layout, strategy, mulAlphaFixed);
                    if (problem.conjugate || (problem.alpha == -1))
                        map<uint32_t>(hw, state.S_regs, S_layout, strategy, signChange);
                }

                // Advance S copy counter.
                if (++S_copy == SD_copies)
                    S_copy = 0;
            }

            // Test loop counter on last iteration (lateZLoopCheck == true) if D unmasked.
            if ((u == uZLoopCheck) && lateZLoopCheck && !dMasked)
                zLoopCheck(unrollZ, enableBarriers);

            // Store D every d_load loops.
            if (storeD) {
                doDStore(D_layout, D0_layout, D_addrs, d_load, D_copy);
                if (++D_copy == SD_copies)
                    D_copy = 0;
            }

            // Test loop counter at very end (lateZLoopCheck == true) if D masked.
            if ((u == uZLoopCheck) && lateZLoopCheck && dMasked)
                zLoopCheck(unrollZ, enableBarriers);
        }

        // Forget about active vflags.
        state.wipeActiveVFlags();

        return true;
    };

    syncall();

    mark(lZLoopBegin); {
        if (!zLoopBody(state.S_layout, S_layoutReflected, state.D_layout, state.D0_layout, state.S_addrs, state.S_addrSrcs, state.D_addrs,
                       unrollZ, strategy.s_load, strategy.d_load, strategy.barrierFreq > 0, !triRemOnly)) return false;

        if (strategy.barrierFreq == 0)
            jmpi(1 | state.flagAP, lZLoopBegin);
        else {
            jmpi(1 | ~state.flagAP, lZLoopEnd);
            jmpi(1 | ~flagBarrier, lZLoopBegin);

            auto temp = state.ra.alloc();
            if (!bcount.isInvalid())
                mov(1, bcount, uint16_t(strategy.barrierFreq));

            GRF r0_info;
            bool freeR0Info = false;

            if (state.r0_info.isARF()) {
                r0_info = state.ra.alloc();
                mov<uint32_t>(r0DWords(hw), r0_info, state.r0_info);
                freeR0Info = true;
            } else
                r0_info = GRF{state.r0_info.getBase()};

            barrier(temp, r0_info);
            state.ra.safeRelease(temp);
            if (freeR0Info)
                state.ra.safeRelease(r0_info);

            jmpi(1, lZLoopBegin);
        }
    }
    mark(lZLoopEnd);

    state.raVFlag.safeRelease(flagBarrier);
    state.ra.safeRelease(bcount);

    // z remainder loop.
    if (offsetZ) {
        // Undo offseting on the z loop counter and check for zero remainder loops.
        add(1 | le | state.flagAP, state.Z, state.Z, uint16_t(offsetZ));

        // Get the current S, D addresses.
        Subregister S_addr0, S1_addr0, D_addr0;
        int S_shift, D_shift;
        int S1_shift = 0;
        if (reflecting) {
            S_addr0 = getOriginAddr(state.S_layout, state.S_addrSrcs[0], &S_shift);
            S1_addr0 = getOriginAddr(state.S_layout, state.S_addrSrcs[1], &S1_shift);
        } else
            S_addr0 = getOriginAddr(state.S_layout, state.S_addrs, &S_shift);

        D_addr0 = getOriginAddr(state.D_layout, state.D_addrs, &D_shift);

        auto unshiftAddr0 = [&]() {
            if (S_shift) shl(1, S_addr0, S_addr0, S_shift);
            if (D_shift) shl(1, D_addr0, D_addr0, D_shift);
            if (reflecting && S1_shift) shl(1, S1_addr0, S1_addr0, S1_shift);
        };

        // Prepare for potential new layout.
        RegisterLayout S_layout1, S_layout1Reflect, D_layout1, D0_layout1;
        vector<RegisterRange> S_addrs1, S_addrSrcs1[2], D_addrs1;

        // First, try handling the whole remainder, all at once.
        bool wholeRem = false, fragmented = false;
        auto newSLoad = strategy.s_load, newDLoad = strategy.d_load;
        auto saveSStrategy = strategy.S;
        bool largeDCrosspack = isLargeCrosspack(Td, problem.D.crosspack);

        if (S_addr0.isValid() && D_addr0.isValid() && !largeDCrosspack && !D_paddingLoop) {
            auto saveState = state;
            auto saveMasks = masks;
            (xLoop ? state.remainderX : state.remainderY) = state.Z;
            pushStream(); try {
                cleanup();
                state.ra.claim(S_addr0);
                state.ra.claim(D_addr0);
                if (reflecting) state.ra.claim(S1_addr0);
                unshiftAddr0();

                wholeRem = setup(strategy.s_load, strategy.d_load, S_addr0, S1_addr0, D_addr0, true);

                state.ra.release(S_addr0);
                state.ra.release(D_addr0);
                if (reflecting) state.ra.release(S1_addr0);
            } catch (...) {}
            if (!wholeRem) {
                masks = saveMasks;
                state = saveState;
            }
            wholeRem ? appendCurrentStream() : discardStream();
        }

        // If that doesn't work, retry with minimal unroll.
        if (!wholeRem) {
            newSLoad = 1;
            newDLoad = crosspack;
            bool unshare = share && (newSLoad != newDLoad);

            // Fragment the S, D layouts, taking the first row/column of each.
            vector<int> indices;
            vector<RegisterRange> D0_addrs1;
            fragmented = !unshare && !largeDCrosspack
                       && (S_layout1 =           state.S_layout.trySlice(indices, !mLoop, 0, newSLoad, strategy.S.padded))
                       && (S_layout1Reflect = S_layoutReflected.trySlice(          mLoop, 0, newSLoad, strategy.S.padded))
                       && (D_layout1 =           state.D_layout.trySlice(D_addrs1,  state.D_addrs, !mLoop, 0, newDLoad, false))
                       && (D0_layout1 =         state.D0_layout.trySlice(D0_addrs1, state.D_addrs, !mLoop, 0, newDLoad, false));

            if (fragmented) {
                // Select source address registers from the fragments.
                for (auto b : indices)
                    S_addrs1.push_back(state.S_addrs[b]);
                if (reflecting) for (auto b : indices) {
                    S_addrSrcs1[0].push_back(state.S_addrSrcs[0][b]);
                    S_addrSrcs1[1].push_back(state.S_addrSrcs[1][b]);
                }

                // Update sizes.
                (mLoop ? nms : nns) = newSLoad;
                (mLoop ? nmd : nnd) = newDLoad;
            } else {
                // Fragmentation failed. Start fresh.
                if (S_addr0.isInvalid() || D_addr0.isInvalid())
                    return false;

                cleanup();
                state.ra.claim(S_addr0);
                state.ra.claim(D_addr0);
                if (reflecting) state.ra.claim(S1_addr0);
                unshiftAddr0();

                if (largeDCrosspack)
                    transposeAccessType(strategy.S);

                if (!setup(newSLoad, newDLoad, S_addr0, S1_addr0, D_addr0, false))
                    return false;

                state.ra.release(S_addr0);
                state.ra.release(D_addr0);
                if (reflecting) state.ra.release(S1_addr0);
            }

            if (crosspack > 1) {
                lateZLoopCheck = true;
                copyCalcIncrements(problem, strategy, state, newSLoad, newDLoad);
            }
        }

        // Emit z remainder loop.
        Label lZRemLoopBegin, lZRemLoopEnd;
        jmpi(1 | state.flagAP, lZRemLoopEnd);
        mark(lZRemLoopBegin);
          wholeRem ? zLoopBody(state.S_layout, S_layoutReflected, state.D_layout, state.D0_layout, state.S_addrs, state.S_addrSrcs, state.D_addrs, unrollZ, newSLoad, newDLoad, false, true, false, !triRemOnly) :
        fragmented ? zLoopBody(S_layout1, S_layout1Reflect, D_layout1, D0_layout1, S_addrs1, S_addrSrcs1, D_addrs1, crosspack, newSLoad, newDLoad, false, true, crosspack > 1)
                   : zLoopBody(state.S_layout, S_layoutReflected, state.D_layout, state.D0_layout, state.S_addrs, state.S_addrSrcs, state.D_addrs, crosspack, newSLoad, newDLoad, false, true, crosspack > 1);
        if (!wholeRem || triRemOnly)
            jmpi(1 | state.flagAP, lZRemLoopBegin);
        mark(lZRemLoopEnd);

        strategy.S = saveSStrategy;
    }

    // Padding for TRSM copy kernels.
    if (D_paddingLoop) {
        // Grab first row/column from the D layout.
        vector<RegisterRange> D_addrs1;
        auto D_layout1 = state.D_layout.slice(D_addrs1, state.D_addrs, byColumn, 0, 1, false);

        Label lZZeroLoopBegin, lZZeroLoopEnd;
        cmp(1 | gt | state.flagAP, Z_padding, 0);
        jmpi(1 | ~state.flagAP, lZZeroLoopEnd);
        zeroMatrix(state.D_regs, strategy);
        mark(lZZeroLoopBegin); {
            storeMatrix(state.D_regs, D_layout1, D_addrs1, strategy, state);
            add(1 | gt | state.flagAP, Z_padding, Z_padding, -1);
            incAddr(D_addrs1, uint16_t(problem.D.packSize * Td), D_layout1, strategy, state);
            jmpi(1 | state.flagAP, lZZeroLoopBegin);
        }
        mark(lZZeroLoopEnd);
    }

    // Finalize and store sums.
    if (problem.sum) {
        Label skipSumStore;
        bool simtCF = strategy.fused;

        if (remainderX) {
            cmp((simtCF ? 16 : 1) | le | state.flagAP, state.remainderX, 0);
            simtCF ? goto12(16 | state.flagAP, skipSumStore)
                   :   jmpi(1  | state.flagAP, skipSumStore);
        }

        horizontalAdd(!byColumn, state.Ds_regs.back(), state.Ds_layout, state);

        // Accumulate sums from main and remainder loops.
        for (int l = 1; l < int(state.Ds_regs.size()); l++) {
            map(hw, Tsum, state.Ds_regs[0], state.Ds_regs[l], strategy, [&](int ne, GRF r1, GRF r2) {
                add(ne, r1, r1, r2);
            });
            state.ra.safeRelease(state.Ds_regs[l]);
        }
        state.Ds_regs.resize(1);

        MatrixAddressing Ds = problem.D;
        Ds.crosspack = 1;

        MatrixAddressingStrategy Ds_strategy = strategy.D;
        Ds_strategy.accessType = AccessType::Block;

        int sr = 1, sc = 1;
        (byColumn ? sr : sc) = problem.D.packSize;

        RegisterLayout Ds_layoutOut(hw, Tsum, sr, sc, Ds, Ds_strategy, false, false, true);
        if (!Ds_layoutOut.match(state.Ds_layout))
            return false;

        vector<RegisterRange> Ds_addrs;
        allocAddrRegs(hw, Ds_addrs, Ds_layoutOut, state);

        Subregister Ds_base;
        Ds_base = state.ra.alloc_sub(state.effD.getType());

        mulConstant(1, Ds_base.ud(), state.inputs.ldd, problem.D.packSize * Td);
        add(1, Ds_base.ud(), Ds_base.ud(), -problem.D.packSize * Tsum);
        eadd(1, Ds_base, Ds_base.ud(), state.effD, strategy, state);

        setupAddr(Ds_addrs, Ds_base, Ds_layoutOut, Subregister(), strategy, state);
        storeMatrix(state.Ds_regs[0], Ds_layoutOut, Ds_addrs, strategy, state);

        state.ra.safeRelease(Ds_base);
        safeReleaseRanges(Ds_addrs, state);
        safeReleaseRanges(state.Ds_regs, state);
        state.Ds_layout.clear();
        state.ra.safeRelease(state.all1s);

        if (remainderX) {
            mark(skipSumStore);
            if (simtCF) join(16);
        }
    }

    // Done. Free address, data, and flag registers.
    cleanup();
    state.ra.safeRelease(state.signChange);
    if (lateZLoopCheck)
        state.raVFlag.lock(state.flagAP);
    if (reflecting || triangular)
        releaseIndexVec(state);
    state.raVFlag.safeRelease(state.flagReflect);
    state.raVFlag.safeRelease(state.flagSwizzle);

    return true;    /* Success! */
}

// Logic for symmetric/Hermitian matrix loads.
template <HW hw>
void Generator<hw>::copyLoadReflecting(const RegisterRange &dest, const RegisterLayout &layout, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                                       const vector<RegisterRange> &addrs, const vector<RegisterRange> (&addrSrcs)[2], int nr, int nc,
                                       const CopyProblem &problem, const CopyStrategy &strategy, CopyState &state)
{
    const bool mLoop = isColMajor(problem.D.layout) == strategy.xLoop;
    const bool hermitian = (problem.srcType == MatrixSrcType::Hermitian);
    const bool S_a64 = (strategy.S.base.getModel() == ModelA64);
    bool emulate64 = strategy.emulate.emulate64;
    emulate64 |= strategy.emulate.emulate64_logic;

    auto &flagReflect = state.flagReflect;
    auto &diag = state.inputs.diag;

    int lastR = layout[0].offsetR;
    int lastC = layout[0].offsetC;
    if (lastR != lastC)
        add(1, diag, diag, int16_t(lastC - lastR));

    auto nblocks = layout.blocks();

#ifdef STANDALONE
    bool readCheck = problem.readCheck;
#else
    bool readCheck = false;
#endif

    for (size_t b = 0; b < nblocks; b++) {
        auto &block = layout[b];
        auto &addr = addrs[b];              // Scattered addresses
        auto &addr0 = addrSrcs[0][b];       // Non-transposed addresses
        auto &addr1 = addrSrcs[1][b];       // Transposed addresses
        auto nextR = b < (nblocks - 1) ? layout[b+1].offsetR : (mLoop ? nr : 0);
        auto nextC = b < (nblocks - 1) ? layout[b+1].offsetC : (mLoop ? 0 : nc);
        auto diffR = nextR - lastR;
        auto diffC = nextC - lastC;

        auto indexVec = state.indexVecRT;
        if (!block.colMajor)
            indexVec.setMods(2);    // negative

        auto advanceDiag = [&](){
            if (diffR != diffC)
                add(1, diag, diag, int16_t(diffC - diffR));
        };

        bool simd16 = (block.simdSize == 16);
        bool qword = (block.ebytes == 8);
        int s = 64 >> GRF::log2Bytes(hw);

        auto cmpCMod = (problem.lower ? le : ge);

        cmp(block.simdSize | cmpCMod | flagReflect, null.d(), diag, indexVec);

        if (!hermitian) advanceDiag();

        if (!S_a64)
            sel<uint32_t>(block.simdSize | flagReflect, addr, addr0, addr1);
        else if (!emulate64) {
            sel<uint64_t>(8 | flagReflect, addr[0], addr0[0], addr1[0]);
            if (simd16)
                sel<uint64_t>(8 | M8 | flagReflect, addr[s], addr0[s], addr1[s]);
        } else {
            sel<uint32_t>(8 | flagReflect, addr[0][0](2), addr0[0][0](2), addr1[0][0](2));
            if (simd16)
                sel<uint32_t>(8 | M8 | flagReflect, addr[s][0](2), addr0[s][0](2), addr1[s][0](2));
            sel<uint32_t>(8 | flagReflect, addr[0][1](2), addr0[0][1](2), addr1[0][1](2));
            if (simd16)
                sel<uint32_t>(8 | M8 | flagReflect, addr[s][1](2), addr0[s][1](2), addr1[s][1](2));
        }

        if (hermitian) {
            // For Hermitian copies:
            //  - load block now;
            //  - conjugate reflected part;
            //  - set imaginary parts along diagonal to zero.

            // Load block.
            GRFRange thisDest{dest.getBase() + block.offsetReg(), block.nregs()};
            loadMatrixBlock(thisDest, block, atype, astrategy, addr, strategy, state, layout.type().real().ngen(), readCheck);

            // Conjugate reflected part.
            if (!qword)
                xor_<uint32_t>(block.simdSize | ~flagReflect, thisDest, thisDest, state.signChange[0](0,2,1));
            else if (!emulate64) {
                xor_<uint64_t>(8 | ~flagReflect, thisDest[0], thisDest[0], state.signChange[0](0,2,1));
                if (simd16)
                    xor_<uint64_t>(8 | ~flagReflect | M8, thisDest[s], thisDest[s], state.signChange[0](0,2,1));
            } else {
                xor_<uint32_t>(8 | ~flagReflect, thisDest[0][1](2), thisDest[0][1](2), state.signChange[1](0,2,2));
                if (simd16)
                    xor_<uint32_t>(8 | ~flagReflect | M8, thisDest[s][1](2), thisDest[s][1](2), state.signChange[1](0,2,2));
            }

            // Find diagonal elements.
            cmp(block.simdSize | eq | flagReflect, null.d(), diag, indexVec);

            advanceDiag();

            if (!qword)
                and_<uint32_t>(block.simdSize | flagReflect, thisDest, thisDest, ~state.selectImag[0](0,2,1));
            else if (!emulate64) {
                and_<uint64_t>(8 | flagReflect, thisDest[0], thisDest[0], ~state.selectImag[0](0,2,1));
                if (simd16)
                    and_<uint64_t>(8 | flagReflect | M8, thisDest[s], thisDest[s], ~state.selectImag[0](0,2,1));
            } else {
                and_<uint32_t>(8 | flagReflect, thisDest[0][0](2), thisDest[0][0](2), ~state.selectImag[0](0,2,2));
                if (simd16)
                    and_<uint32_t>(8 | flagReflect | M8, thisDest[s][0](2), thisDest[s][0](2), ~state.selectImag[0](0,2,2));
                and_<uint32_t>(8 | flagReflect, thisDest[0][1](2), thisDest[0][1](2), ~state.selectImag[1](0,2,2));
                if (simd16)
                    and_<uint32_t>(8 | flagReflect | M8, thisDest[s][1](2), thisDest[s][1](2), ~state.selectImag[1](0,2,2));
            }
        }

        lastR = nextR;
        lastC = nextC;
    }

    if (!hermitian)
        loadMatrix(dest, layout, addrs, strategy, state, readCheck);
}

// Logic for triangular copies.
template <HW hw>
void Generator<hw>::copyTriangularMask(RegisterRange &regs, const RegisterLayout &layout, int nr, int nc, bool mLoop,
                                       const CopyProblem &problem, const CopyStrategy &strategy, CopyState &state)
{
    auto Ts = problem.Ts;

#ifdef STANDALONE
    // Skip if read checking.
    if (problem.readCheck) return;
#endif

    // Setup.
    bool globalCM = layout.colMajor();

    RegData indexVec = state.indexVecRT;
    if (!globalCM)
        indexVec = -indexVec;
    RegData indexVecRe = indexVec;
    indexVecRe.setRegion(0, 0, 2);

    auto flagTri = state.flagTri, flagDiag = state.flagDiag;

    state.usePhysicalFlag(flagTri, hw);
    if (problem.unit || problem.trsm)
        state.usePhysicalFlag(flagDiag, hw);

    Immediate zero{0}, one{0};
    if (Ts.isInteger()) {
        one = int16_t(problem.trsm ? -1 : +1);
        zero = uint16_t(0);
    } else if (Ts.real() == Type::f16) {
        one = half(problem.trsm ? -1.0f : +1.0f);
        zero = half(0.0f);
    } else {
        one = float(problem.trsm ? -1.0f : +1.0f);
        zero = 0.0f;
    }

    // Decide whether to loop in column or row major order.
    int nx = globalCM ? nr : nc;
    int ny = globalCM ? nc : nr;

    int lastR = 0, lastC = 0;

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx;) {
            auto r = globalCM ? x : y;
            auto c = globalCM ? y : x;

            // Adjust diag.
            auto diffR = r - lastR;
            auto diffC = c - lastC;
            if (diffR != diffC)
                add(1, state.inputs.diag, state.inputs.diag, int16_t(diffC - diffR));

            // Locate register.
            int nelems;
            Subregister src = layout.find(r, c, regs, &nelems);

            // Limit SIMD to 2 GRFs, or 16 real elements, whichever is smaller.
            nelems = std::min({nelems, 2 * elementsPerGRF(hw, Ts), 16 / Ts.complexComponents()});
            int simd = nelems * Ts.complexComponents();

            // Find elements outside triangle, and those on the diagonal
            //   diag <  indexVec: below diagonal
            //   diag == indexVec: along diagonal
            //   diag >  indexVec: above diagonal
            cmp(simd | (problem.lower ? gt : lt) | flagTri, null.d(), state.inputs.diag, indexVec);
            if (problem.unit || problem.trsm) {
                if (Ts.isComplex() && problem.trsm && !problem.unit)
                    cmp((simd >> 1) | eq | flagDiag, null.d(), state.inputs.diag, indexVecRe);
                else
                    cmp(simd | eq | flagDiag, null.d(), state.inputs.diag, indexVec);
            }

            // Zero out the triangle's exterior.
            mov(simd | flagTri, src(1), zero);

            if (problem.unit) {
                // Set diagonal to 1 (or for TRSM, -1. Sign will be flipped later.)
                if (Ts.isComplex())
                    mov(simd | flagDiag, src(1), state.complexOne[0].retype(Ts.real().ngen()));
                else
                    mov(simd | flagDiag, src(1), one);
            } else if (problem.trsm) {
                if (!state.invertSub.use)
                    inv(simd | flagDiag, src(1), -src(1));
                else {
                    auto desiredSIMDSub = (simd >> int(Ts.isComplex()));
                    auto simdSub = state.invertSub.simd;
                    auto &srcR1 = state.invertSub.srcR1;
                    auto &srcI1 = state.invertSub.srcI1;
                    auto &src1 = state.invertSub.src1;

                    if (desiredSIMDSub > simdSub) stub();
                    if (desiredSIMDSub < simdSub)
                        and_(1, flagDiag, flagDiag, uint16_t((1 << desiredSIMDSub) - 1));

                    if (Ts.isComplex()) {
                        // Deinterleave real and imaginary parts in preparation for sub.
                        RegisterRegion srcR, srcI;
                        auto nd = (Ts.size() >> 3);
                        if (Ts == Type::cf32 || Ts == Type::ctf32)
                            srcR = srcI = src.ud()(2, 1, 0);
                        else if (Ts == Type::cf64)
                            srcR = srcI = src.ud()(4, 2, 1);
                        else stub();
                        srcI.setOffset(srcR.getOffset() + nd);

                        auto nsimd = (simd >> 1) * nd;
                        mov(nsimd, srcR1.ud(), srcR);
                        mov(nsimd, srcI1.ud(), srcI);
                    } else
                        mov(simd, src1, src(1));

                    state.invertSub.tempFlags[0] = flagDiag;
                    state.invertSub.tempFlags[1] = flagTri;
                    call(simdSub | flagDiag, state.invertSub.callStorage, state.invertSub.label);

                    if (Ts.isComplex()) {
                        if (Ts == Type::cf32 || Ts == Type::ctf32) {
                            mov((simd >> 1) | Switch, src.ud(0)(2), srcR1.ud());
                            mov(simd >> 1, src.ud(1)(2), srcI1.ud());
                        } else if (Ts == Type::cf64) {
                            mov((simd >> 1) | Switch, src.ud(0)(4), srcR1.ud(0)(2));
                            mov(simd >> 1, src.ud(1)(4), srcR1.ud(1)(2));
                            mov(simd >> 1, src.ud(2)(4), srcI1.ud(0)(2));
                            mov(simd >> 1, src.ud(3)(4), srcI1.ud(1)(2));
                        } else stub();
                    } else
                        mov(simd | Switch, src(1), src1);
                }
            }

            x += nelems;
            lastR = r;
            lastC = c;
        }
    }

    // Prepare diag for next block.
    auto diffR = (mLoop ? nr : 0) - lastR;
    auto diffC = (mLoop ? 0 : nc) - lastC;
    if (diffR != diffC)
        add(1, state.inputs.diag, state.inputs.diag, int16_t(diffC - diffR));
}

GEMMSTONE_NAMESPACE_END
