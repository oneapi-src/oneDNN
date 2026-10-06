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
#include "invert.hpp"
#include "kernel_queries.hpp"
#include "layout_utils.hpp"
#include "map.hpp"
#include "ngen_object_helpers.hpp"
#include "state_utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;
using std::vector;


static inline bool trsmSkipSLMVSolveSync(const GEMMProblem &problem, const GEMMStrategy &strategy);


// TRSM kernel generation interface.
template <HW hw>
void Generator<hw>::trsm(GEMMProblem problem, GEMMStrategy strategy, const InterfaceHandler &interface_)
{
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc;
    bool left = problem.left;
    bool backward = problem.backward();

    GEMMState state(hw, strategy);

    // Check for (unsupported) type conversions.
    if (Ta != problem.Ta_ext || Tb != problem.Tb_ext || Tc != problem.Tc_ext) stub();

    // Set up underlying GEMM problem/strategy.
    problem.updateType = UpdateType::Full;
    problem.beta = (problem.inplace ? 1 : 0);
    strategy.kParallel = false;
    strategy.kParallelLocal = false;
    strategy.checkAdd32 = false;
    strategy.cLoadAhead = problem.inplace;

    // By default, don't use dispatch mask.
    setDefaultNoMask();
    setDefaultAutoSWSB();

    // Set up.
    problem.autoTypeConversions(strategy.systolic);
    interface = interface_;
    if (strategy.xParallel)
        interface.requireBarrier();
    trsmInitState(problem, strategy, state);
#ifdef STANDALONE
    gemmKernelNotes(problem, strategy);
#endif
#ifdef ASM_OUTPUT
    interface.dumpAssignments(std::cout);
#endif
    if (!strategy.A.base.isStateless()) strategy.A.base.setIndex(state.inputs.surfaceA);
    if (!strategy.B.base.isStateless()) strategy.B.base.setIndex(state.inputs.surfaceB);
    if (!strategy.C.base.isStateless()) strategy.C.base.setIndex(state.inputs.surfaceC[0]);

    // Merge left and right-side logic by renaming variables:
    //  U: triangular matrix
    //  V: other matrix
    //  x: outer index in U
    //  y: outer index in V
    auto loopX =   left ? LoopM : LoopN;
    auto loopY =   left ? LoopN : LoopM;
    auto &x =      left ? state.inputs.m : state.inputs.n;
    auto &y =      left ? state.inputs.n : state.inputs.m;
    auto &ldu =    left ? state.inputs.lda : state.inputs.ldb;
    auto &pU =     left ? problem.A : problem.B;
    auto &sV =     left ? strategy.B : strategy.A;
    auto Tu =      left ? problem.Ta : problem.Tb;
    auto &effU =   left ? state.effA : state.effB;
    auto ku_load = left ? strategy.ka_load : strategy.kb_load;
    auto kv_load = left ? strategy.kb_load : strategy.ka_load;
    auto slmU    = left ? strategy.slmA : strategy.slmB;
    auto slmV    = left ? strategy.slmB : strategy.slmA;
    auto unrollX = strategy.unroll[loopX];
    auto unrollY = strategy.unroll[loopY];

    auto localIDY =    left ? state.inputs.localIDN : state.inputs.localIDM;
    auto &localSizeX = left ? state.inputs.localSizeM : state.inputs.localSizeN;
    auto &localSizeY = left ? state.inputs.localSizeN : state.inputs.localSizeM;
    auto globalIDY =   left ? state.inputs.groupIDN : state.inputs.groupIDM;

    auto &x0 = left ? state.i0 : state.j0;
    auto &y0 = left ? state.j0 : state.i0;

    bool pbackward = backward && isPacked(pU.layout);
    bool fixedWG = strategy.fixedWG(problem);
    bool wgCheck = wgRemCheck(problem, strategy);
    bool wgCheckX = wgCheck && strategy.xParallel;
    Subregister wgY0;

    // Prologue.
    prologue(strategy, state);

#ifdef STANDALONE
    // Fill SLM with NaN for debugging, if enabled.
    if (problem.mock.slmInit) {
        auto Tslm = strategy.slmA ? problem.Ta :
                    strategy.slmB ? problem.Tb : problem.Tc;
        mockInitSLM(Tslm, gemmSLMSize(hw, problem, strategy), state);
    }
#endif

    // Grab fused ID if needed.
    if (strategy.fused) {
        if (strategy.fusedLoop != loopY)
            stub("TRSM kernels must be fused in the y (rectangular) dimension");
        if ((strategy.loopOrder[0] != loopY) && strategy.xParallel)
            stub("Fused x-parallel TRSM kernels must have the y dimension innermost");
        getFusedID(unrollY, problem, strategy, state);
    }

    // Divide out subgroup size from local size 0 and local ID 0
    removeSG(problem, strategy, state);

    // Round up x to unrollX for packed backward kernels.
    Subregister xUnpadded = x;
    if (pbackward) {
        x = state.ra.alloc_sub<int32_t>(getHint(HintType::LongTerm, strategy));
        alignUp(x, xUnpadded, strategy.unroll[problem.left ? LoopM : LoopN], strategy, state);
    }

    // Check x against skewLocalID limit.
    // Skewing local IDs can improve solves at the expense of GEMM.
    if (strategy.skewLocalIDXLimit)
        cmp(1 | gt | f0[1], x, strategy.skewLocalIDXLimit);

    // Check for in-place operation for out-of-place TRSM, as an optimization.
    if (!problem.inplace && strategy.C.base.isStateless() && sV.base.isStateless()) {
        if (!state.inputs.flags.isValid()) {
            state.inputs.flags = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
            mov(1, state.inputs.flags, 0);
        }
        auto &V = left ? state.inputs.B : state.inputs.A;
        if (strategy.emulate.emulate64) {
            cmp(1 | eq | f1[0], null.ud(), state.inputs.C[0].ud(0), V.ud(0));
            cmp(1 | f1[0] | eq | f1[0], null.ud(), state.inputs.C[0].ud(1), V.ud(1));
        } else
            add(1 | eq | f1[0], null.uq(), state.inputs.C[0], -V);
        or_(1 | f1[0], state.inputs.flags, state.inputs.flags, FlagInPlace);
    }

    // Infer missing leading dimensions.
    gemmAdjustImplicitLDs(problem, strategy, state);

    // Scale inputs.
    gemmScaleInputs(problem, strategy, state);

    // Load alpha/beta from memory if needed.
    gemmLoadScalars(problem, strategy, state);

    // Local ID handling.
    if (strategy.xParallel) {
        Label skipQL;
        if (strategy.skewLocalIDXLimit) jmpi(1 | f0[1], skipQL);
        gemmReorderLocalIDs(problem, strategy, state);
        mark(skipQL);
    }

    // Batch handling.
    gemmGetBatchIDs(problem, strategy, state);

    // Compute offset for A, B, C for non-strided and strided batch.
    gemmOffsetBatchABC(problem, strategy, state);

    // Calculate y0 -- the starting y index for this thread.
    status << "Calculate y0" << status_stream::endl;
    y0 = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));

    Subregister idY = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp1, strategy));

    if (fixedWG)
        mulConstant(1, idY, globalIDY, strategy.wg[loopY]);
    else
        mul(1, idY, globalIDY, localSizeY.uw());
    if (wgCheck) {
        wgY0 = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
        mulConstant(1, wgY0, idY, unrollY);
    }
    add(1, idY, idY, localIDY.uw(0));
    mulConstant(1, y0, idY, unrollY);

    // Move r0 if configured and save local IDs/sizes if needed.
    Subregister lidX;
    moveR0(strategy, state);
    if (strategy.needsMNLocalIDs()) {
        saveMNLocalIDs(strategy, state);
        lidX = left ? state.lidM : state.lidN;
    }
    if (strategy.xParallel && !fixedWG) {
        auto storage = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        state.lszX = storage.uw(0);
        state.wgTileX = storage.uw(1);
        mov(1, state.lszX, localSizeX.uw());
        mulConstant(1, state.wgTileX, localSizeX.uw(), unrollX);
    }

    state.ra.safeRelease(idY);
    state.ra.safeRelease(state.inputs.localIDM);
    state.ra.safeRelease(state.inputs.localIDN);
    state.ra.safeRelease(localSizeY);
    state.ra.safeRelease(localSizeX);

    // Align x up to a multiple of unrollX * localSizeX.
    Subregister xAlign;

    auto calcXAlign = [&]() {
        xAlign = state.ra.alloc_sub<int32_t>();
        if (strategy.xParallel) {
            if (fixedWG)
                alignUp(xAlign, x, unrollX * strategy.wg[loopX], strategy, state);
            else {
                // WG size must be a power of 2 in x dimension. TODO: pass in xAlign, remove restriction.
                auto lg2LSX = state.ra.alloc_sub<uint32_t>();
                eadd3(1, xAlign, x, state.lszX, -1);
                fbl(1, lg2LSX, state.lszX);
                shr(1, xAlign, xAlign, lg2LSX);
                alignUp(xAlign, xAlign, unrollX, strategy, state);
                shl(1, xAlign, xAlign, lg2LSX);
                state.ra.safeRelease(lg2LSX);
            }
        } else
            alignUp(xAlign, x, unrollX, strategy, state);
    };

    // Initialize x0.
    status << "Calculate x0" << status_stream::endl;
    x0 = state.ra.alloc_sub<int32_t>(getHint(HintType::LongTerm, strategy));
    if (strategy.xParallel) {
        mulConstant(1, x0, lidX, unrollX);
        if (backward)
            eadd3(1, x0, -x0, x, -unrollX);
    } else if (backward)
        add(1, x0, x, int16_t(-unrollX));
    else
        mov(1, x0, uint16_t(0));

    auto x0Unclamped = x0;
    if (backward) {
        x0Unclamped = state.ra.alloc_sub<int32_t>(getHint(HintType::LongTerm, strategy));
        std::swap(x0, x0Unclamped);
        mov(1 | sat, x0.ud(), x0Unclamped);
    }

    // Compute starting addresses for A, B, C.
    //    Ux += x0
    //    Vy += y0
    // For LU/RL TRSM (reverse)
    //    Uk += x - ku_load [x - unrollKSLM if copied]
    //    Vk += x - kv_load [x - unrollKSLM if copied]
    // For out-of-place TRSM, need equivalent V offsets applied to C also.
    Subregister offsetUX = x0, offsetUK, offsetVK, offsetVY = y0;

    if (backward) {
        offsetUK = state.ra.alloc_sub<int32_t>(getHint(HintType::TempComp1));
        offsetVK = state.ra.alloc_sub<int32_t>(getHint(HintType::TempComp0));
        add(1, offsetUK, x, slmU ? -strategy.unrollKSLM : -ku_load);
        add(1, offsetVK, x, slmV ? -strategy.unrollKSLM : -kv_load);
    } else if (!strategy.xParallel)
        offsetUX = Subregister();           // x0 = 0; no need to offset.

    if (strategy.A.address2D || strategy.B.address2D || strategy.C.address2D) stub();

    status << "Increment A, B offsets" << status_stream::endl;
    if (left) {
        gemmOffsetABC(true, offsetUX, Subregister(), offsetUK, offsetUX, Subregister(), problem, strategy, state, true, false, false);
        gemmOffsetABC(true, Subregister(), offsetVY, offsetVK, Subregister(), offsetVY, problem, strategy, state, false, true, false);
    } else {
        gemmOffsetABC(true, offsetVY, Subregister(), offsetVK, offsetVY, Subregister(), problem, strategy, state, true, false, false);
        gemmOffsetABC(true, Subregister(), offsetUX, offsetUK, Subregister(), offsetUX, problem, strategy, state, false, true, false);
    }

    if (!problem.inplace) {
        state.offsetCV = state.ra.alloc_sub(state.offsetC[0].getType(), getHint(HintType::LongTerm, strategy));
        emov(1, state.offsetCV, state.offsetC[0], strategy, state);
        auto modProblem = problem;
        auto modStrategy = strategy;
        auto modState = state;
        trsmMoveCToV(modProblem, modStrategy, modState);
        left ? gemmOffsetABC(true, Subregister(), offsetVY, offsetVK, Subregister(), offsetVY, modProblem, modStrategy, modState, false, true, false) :
               gemmOffsetABC(true, offsetVY, Subregister(), offsetVK, offsetVY, Subregister(), modProblem, modStrategy, modState, true, false, false);
    }

    status << "Increment C offset" << status_stream::endl;
    gemmOffsetABC(true, state.i0, state.j0, Subregister(), Subregister(), Subregister(), problem, strategy, state, false, false, true);

    state.ra.safeRelease(offsetUK);
    state.ra.safeRelease(offsetVK);

    // Allocate remainder variable for x/y loops.
    // Calculate remainders for y loop: clamp(y - y0, 0, unrollY).
    bool remX = (strategy.remHandling[loopX] != RemainderHandling::Ignore);
    bool remY = (strategy.remHandling[loopY] != RemainderHandling::Ignore);
    bool checkRemY = remY && !strategy.needsBarrier();

    state.remaindersFused[loopX] = state.remainders[loopX] = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));

    if (remX) {
        int simt = strategy.fused ? 16 : 1;
        status << "X >= 0 test" << status_stream::endl;
        cmp(simt | le | f0[1], null.d(), x, int16_t(0));
    }
    if (remY) {
        status << "Y remainder" << status_stream::endl;
        state.remaindersFused[loopY] = state.remainders[loopY] = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        if (strategy.fused) {
            Subregister rfStorage;
            rfStorage = state.remaindersFused[loopY] = state.remFusedStorage = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
            cmp(16 | ge | f1[1], y0, y);
            add(1, rfStorage, -state.fusedID, uint16_t(unrollY));
            add(1 | sat, state.remainders[loopY], -y0, y);
            add(1 | sat, rfStorage, state.remainders[loopY], -rfStorage);
        } else
            add(1 | le | f1[1] | sat, state.remainders[loopY], -y0, y);
        min_(1, state.remainders[loopY], state.remainders[loopY], uint16_t(unrollY));
    }

    if (wgCheck) {
        state.remaindersWG[loopX] = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp1, strategy));
        state.remaindersWG[loopY] = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        add(1 | sat, state.remaindersWG[loopY], -wgY0, y);
    }

    state.ra.safeRelease(y0);
    state.ra.safeRelease(wgY0);

    // Add offsets to A, B, C base pointers for stateless accesses.
    status << "Offset A/B/C" << status_stream::endl;

    if (state.offsetCV.isValid()) {
        if (strategy.C.base.isStateless()) {
            state.effCV = state.ra.alloc_sub(state.inputs.C[0].getType(), getHint(HintType::LongTerm, strategy));
            eadd(1, state.effCV, state.inputs.C[0], state.offsetCV, strategy, state);
            state.ra.safeRelease(state.offsetCV);
        } else
            state.effCV = state.offsetCV;
    }

    gemmSetupABC(problem, strategy, state);

    // Early exit if nothing to do. Don't do y remainder check for x-parallelized/SLM kernels
    //  to ensure thread group stays together.
    Label leave;
    if (strategy.fused) {
        if (remX && checkRemY)
            goto12(16 | f0[1] | anyv, leave);
        else if (remX)
            goto12(16 | f0[1], leave);
        else if (checkRemY)
            goto12(16 | f1[1], leave);
    } else {
        if (remX && checkRemY)
            ejmpi(1 | f0[1] | anyv, leave);
        else if (remX)
            jmpi(1 | f0[1], leave);
        else if (checkRemY)
            jmpi(1 | f1[1], leave);
    }

    // Synthesize x loop.
    Label xLoopTop;
    syncall();
    mark(xLoopTop); {
        // Get x remainder and update (GEMM) k.
        //   forward:    k = x0
        //            remX = min(x - x0, unrollX)
        //  backward:    k = max(0, x - x0 - unrollX)
        //            remX = clamp(unrollX + x0Unclamped, 0, unrollX)
        // Also calculate workgroup x remainder if needed.
        //   forward: wgRemX = clamp(x - wgX0, 0, wgTileX)
        //  backward: wgRemX = min(wgTileX + wgX0, wgTileX)
        auto remX   = state.remainders[loopX];
        auto wgRemX = state.remaindersWG[loopX];
        Subregister temp;

        if (wgCheckX) {
            mulConstant(1, wgRemX, lidX, unrollX);
            add(1, wgRemX.d(), x0Unclamped, backward ? wgRemX : -wgRemX);
        }

        if (!backward) {
            add(1 | sat, remX, x, -x0);
            mov(1, state.k, x0);
            if (wgCheckX)
                add(1 | sat, wgRemX, x, -wgRemX.d());
            min_(1, remX, remX, unrollX);
        } else {
            if (pbackward) {
                temp = state.ra.alloc_sub<uint32_t>();
                add(1 | sat, temp, xUnpadded, -x0);
            }
            add(1, remX.d(), x0Unclamped, unrollX);
            if (wgCheckX)
                add(1 | sat, wgRemX, wgRemX.d(), unrollX);
            add(1, state.k, x, -remX.d());
            min_(1 | sat, remX, remX.d(), unrollX);
        }

        if (wgCheckX) {
            fixedWG ? min_(1, wgRemX, wgRemX, unrollX * strategy.wg[loopX])
                    : min_(1, wgRemX, wgRemX, state.wgTileX);
        } else if (wgCheck)
            mov(1, wgRemX, remX);

        if (pbackward) {
            min_(1, remX, remX, temp);
            state.ra.safeRelease(temp);
        }

        // Create the kernel body.
        if (backward) state.ra.release(x0Unclamped);
        if (!trsmMEdge(problem, strategy, state)) {
            lastException ? std::rethrow_exception(lastException)
                          : stub("Could not generate kernel.");
        }
        if (backward) state.ra.claim(x0Unclamped);

        // Global fence for x-parallelized kernels, or mixed stateless/surface accesses.
        if (strategy.xParallel || (sV.base.isStateless() != strategy.C.base.isStateless())) {
            GRF r0_info;
            auto temp = state.ra.alloc();

            if (state.r0_info.isARF()) {
                r0_info = state.ra.alloc();
                mov<uint32_t>(r0DWords(hw), r0_info, state.r0_info);
            } else
                r0_info = GRF{state.r0_info.getBase()};

            if (!slmV) {
                memfence(FenceScopeLSC::Subslice, FlushTypeLSC::None, temp, r0_info);
                fencewait();
            }
            MOCK_BARRIERS if (strategy.xParallel)
                activeThreadBarrierSignal(temp, r0_info, strategy);

            state.ra.safeRelease(temp);
            if (r0_info != state.r0_info)
                state.ra.safeRelease(r0_info);
        }

        // Increment/decrement x0 by wgX * unrollX, and do loop check.
        // Recompute xAlign if needed for loop check.
        int dxFixed = unrollX;
        Subregister dxVariable;
        if (strategy.xParallel) {
            if (fixedWG)
                dxFixed *= strategy.wg[loopX];
            else {
                dxFixed = 0;
                dxVariable = state.wgTileX;
            }
        }

        if (strategy.xParallel || backward) calcXAlign();

        if (!backward) {
            dxFixed ? add(1, x0, x0, dxFixed)
                    : add(1, x0, x0, dxVariable);
            cmp(1 | lt | state.flagAP, null.ud(), x0, strategy.xParallel ? xAlign : x);
        } else {
            auto temp = state.ra.alloc_sub<uint32_t>();
            mov(1, temp, x0);
            if (dxFixed) {
                add(1, x0Unclamped, x0, -dxFixed);
                add(1 | sat, x0.ud(), x0, -dxFixed);
            } else {
                add(1, x0Unclamped, x0, -dxVariable);
                add(1 | sat, x0.ud(), x0, -dxVariable);
            }
            dxFixed = 0;
            dxVariable = temp;
            add(1, xAlign, -xAlign, x);
            add(1, dxVariable, temp, -x0);
            cmp(1 | ge | state.flagAP, null.ud(), x0Unclamped, xAlign);
        }
        state.ra.safeRelease(xAlign);

        // Adjust pointers.
        //   Ux += wg_x * unroll_x                  [ -= if backward ]
        //   C  += wg_x * unroll_x        [left N]  [ -= if backward ]
        //         wg_x * unroll_x * ldc  [right N]
        // If backward, clamp Ux/Cx to 0.
        //   (dxVariable holds how much we need to move in x dimension)
        auto effC = state.effC[0];
        auto ldc = state.inputs.ldc[0];
        bool C_ldc = left ^ isColMajor(problem.C.layout);
        bool U_ldu = (pU.layout != (left ? MatrixLayout::N : MatrixLayout::T));

        Subregister temp0 = state.ra.alloc_sub<int32_t>(getHint(HintType::TempComp1, strategy));
        Subregister temp1 = state.ra.alloc_sub<int32_t>(getHint(HintType::TempComp0, strategy));

        auto uxFixed = dxFixed;
        auto uxVariable = dxVariable;
        if (U_ldu) {
            uxFixed ? mulConstant(1, temp0, ldu, uxFixed)
                    : mul(1, temp0, ldu, uxVariable.uw());
            uxVariable = temp0;
            uxFixed = 0;
        } else if (!uxFixed) {
            mulConstant(1, temp0, uxVariable, Tu.size());
            uxVariable = temp0;
        }

        auto cxFixed = dxFixed;
        auto cxVariable = dxVariable;
        if (C_ldc) {
            cxFixed ? mulConstant(1, temp1, ldc, cxFixed)
                    : mul(1, temp1, ldc, cxVariable.uw());
            cxVariable = temp1;
            cxFixed = 0;
        } else if (!cxFixed) {
            mulConstant(1, temp1, cxVariable, Tc.size());
            cxVariable = temp1;
        }

        int sign = backward ? -1 : +1;

        uxFixed ? eadd(1, effU, effU, sign * uxFixed * Tu, strategy, state)
                : eadd(1, effU, effU, (sign == -1) ? -uxVariable : uxVariable, strategy, state);
        cxFixed ? eadd(1, effC, effC, sign * cxFixed * Tc, strategy, state)
                : eadd(1, effC, effC, (sign == -1) ? -cxVariable : cxVariable, strategy, state);

        if (backward)
            state.ra.safeRelease(dxVariable);
        state.ra.safeRelease(temp1);
        state.ra.safeRelease(temp0);

        MOCK_BARRIERS if (strategy.xParallel)
            barrierwait();  // Wait on memory fences from other threads.

        jmpi(1 | state.flagAP, xLoopTop);
    }

    state.ra.safeRelease(x0);

    mark(leave);
    if (strategy.fused)
        join(16);

    epilogue(strategy, state);
    padding();
}

// Handle outer-level m edge cases.
template <HW hw>
bool Generator<hw>::trsmMEdge(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    if (strategy.jointSplit && strategy.remHandling[LoopM] == RemainderHandling::Split && strategy.remHandling[LoopN] == RemainderHandling::Split)
        return mnJointSplitRemainderHandling(problem, strategy, state, &Generator<hw>::trsmBody);
    else
        return mnRemainderHandling(LoopM, problem, strategy, state, &Generator<hw>::trsmNEdge);
}

// Handle outer-level n edge cases.
template <HW hw>
bool Generator<hw>::trsmNEdge(GEMMProblem problem, GEMMStrategy strategy, GEMMState state)
{
    return mnRemainderHandling(LoopN, problem, strategy, state, &Generator<hw>::trsmBody);
}

// Create body of TRSM kernel.
template <HW hw>
bool Generator<hw>::trsmBody(GEMMProblem problem, GEMMStrategy strategy, GEMMState state)
{
    auto loopX = problem.left ? LoopM : LoopN;
    state.ra.safeRelease(state.remaindersWG[loopX]);

    // Adjust for masked A/B.
    auto &ka = strategy.slmA ? strategy.unrollKSLM : strategy.ka_load;
    auto &kb = strategy.slmB ? strategy.unrollKSLM : strategy.kb_load;
    auto ka0 = ka, kb0 = kb;
    (void) gemmPrepMaskedAB(problem, strategy, state);
    auto ka_diff = ka - ka0;
    auto kb_diff = kb - kb0;
    if (problem.backward()) {
        gemmOffsetAk(-ka_diff, state.effA, problem.A, problem, strategy, state);
        gemmOffsetBk(-kb_diff, state.effB, problem.B, problem, strategy, state);
    }

    bool success;
    pushStream(); {
        // Do the real work.
        success = trsmBodyInternal(problem, strategy, state);
    }

    // Discards any code produced in case of a failure.
    success ? appendCurrentStream() : discardStream();

    if (problem.backward()) {
        gemmOffsetAk(+ka_diff, state.effA, problem.A, problem, strategy, state);
        gemmOffsetBk(+kb_diff, state.effB, problem.B, problem, strategy, state);
    }

    return success;
}

// Main body of TRSM kernel. Returns true if successful.
template <HW hw>
bool Generator<hw>::trsmBodyInternal(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    auto Ta = problem.Ta, Tb = problem.Tb;
    bool left = problem.left;
    bool lower = problem.lower;
    bool unit = problem.unit;
    bool inplace = problem.inplace;
    bool backward = problem.backward();
    bool fixedWG = strategy.fixedWG(problem);

    auto loopX = left ? LoopM : LoopN;
    auto loopY = left ? LoopN : LoopM;
    auto wgX = strategy.wg[loopX];
    auto &remX = state.remainders[loopX];
    auto &wgRemX = state.remaindersWG[loopX];
    auto unrollX = strategy.unroll[loopX];
    auto unrollY = strategy.unroll[loopY];
    auto &unrollKSLM = strategy.unrollKSLM;
    auto V_barrierFreq = strategy.V_barrierFreq;
    auto Tu =           left ? Ta                : Tb;
    auto Tv =           left ? Tb                : Ta;
    auto &U =           left ? problem.A         : problem.B;
    auto &V =           left ? problem.B         : problem.A;
    auto &V_strategy =  left ? strategy.B        : strategy.A;
    auto &V_layout =    left ? state.B_layout    : state.A_layout;
    auto &V_layoutRem = left ? state.B_layoutRem : state.A_layoutRem;
    auto &V_addrs =     left ? state.B_addrs     : state.A_addrs;
    auto &V_addrsRem =  left ? state.B_addrsRem  : state.A_addrsRem;
    auto &V_regs =      left ? state.B_regs      : state.A_regs;
    auto &Vi_layout =   left ? state.Bi_layout   : state.Ai_layout;
    auto &Vi_addrs =    left ? state.Bi_addrs    : state.Ai_addrs;
    auto &Vi_addrsRem = left ? state.Bi_addrsRem : state.Ai_addrsRem;
    auto &Vi_regs =     left ? state.Bi_regs     : state.Ai_regs;
    auto &Vi_regsRem =  left ? state.Bi_regsRem  : state.Ai_regsRem;
    auto &Vo =          left ? state.Bo          : state.Ao;
    auto &Vo_strategy = left ? state.Bo_strategy : state.Ao_strategy;
    auto &Vo_layout =   left ? state.Bo_layout   : state.Ao_layout;
    auto &Vo_addrs =    left ? state.Bo_addrs    : state.Ao_addrs;
    auto &Vo_regs =     left ? state.Bo_regs     : state.Ao_regs;
    auto &Vo_regsRem =  left ? state.Bo_regsRem  : state.Ao_regsRem;
    auto  slmU =        left ? strategy.slmA     : strategy.slmB;
    auto &slmV =        left ? strategy.slmB     : strategy.slmA;
    bool  slmVMain =    slmV;
    bool  slmVSolve =   strategy.slmVSolve;
    auto lidX =         left ? state.lidM        : state.lidN;
    auto lidY =         left ? state.lidN        : state.lidM;
    auto &x0 =          left ? state.i0          : state.j0;
    auto &x =           left ? state.inputs.m    : state.inputs.n;
    auto &ldv =         left ? state.inputs.ldb  : state.inputs.lda;
    auto &kv_load =     left ? strategy.kb_load  : strategy.ka_load;
    auto &kv_loadRem =  left ? state.kb_loadRem  : state.ka_loadRem;
    auto  kv_loadMain = kv_load;

    bool uPacked = state.packedU = isPacked(U.layout);
    auto nbX = strategy.namedBarriers[left ? 0 : 1];
    auto nbY = strategy.namedBarriers[left ? 1 : 0];

    Subregister localX0, kMain;

    // When SLM V copies only used in final GEMM loops,
    //   ensure NSeparate allocates consecutive registers.
    if (slmVSolve && !slmVMain && strategy.registerScheme == GEMMStrategy::NSeparate)
        strategy.nSeparateChunk = 8;

    // Accumulate C with panel*panel multiply.
    // Out-of-place TRSM: substitute C for V to read previously solved data.
    auto savedProblem = problem;
    auto savedStrategy = strategy;
    auto savedState = state;
    bool ok = true;

    if (!inplace)
        trsmMoveCToV(problem, strategy, state);

    gemmCalcIncrements(problem, strategy, state);
    ok = ok && gemmAccumulateCSetup(problem, strategy, state);
    ok = ok && kLoopSetup(problem, strategy, state);
    if (!ok) return false;

    // Synchronization preparations.
    bool globalUBarriers = slmU && (nbX < wgX); /* if true, SLM U copies in solve use global barriers. */
    auto &barrierY       = left ? state.barrierN       : state.barrierM;
    auto &barrierHeaderY = left ? state.barrierHeaderN : state.barrierHeaderM;
    if (V_barrierFreq && globalUBarriers) stub();

    // Prepare for trim down cases and post-GEMM-loop resets:
    //   kx_load > kx_loadTRSM: get X subblocks, free unneeded registers.
    //   unrollX < kx_load: need k-loop reset after main loop.
    //   unrollX < unrollKSLM: need k-loop reset after main loop.
    bool needMainReset = (strategy.ka_load > unrollX) || (strategy.kb_load > unrollX)
                      || ((slmU || slmV) && (unrollKSLM * std::max(strategy.slmBuffersA, strategy.slmBuffersB) > unrollX));
    bool needXPReset = trsmNeedsXPReset(problem, strategy, state);

    auto trimDownAB = [&] {
        int kOffsetA = 0, kOffsetB = 0;

        if (strategy.ka_load > strategy.ka_loadTRSM) {
            if (backward)
                kOffsetA = strategy.ka_loadTRSM - strategy.ka_load;
            vector<RegisterRange> A_addrsNew;
            auto A_layoutNew = state.A_layout.slice(A_addrsNew, state.A_addrs, true, 0, strategy.ka_loadTRSM, false);
            safeReleaseRanges(state.A_addrs, state);
            reclaimRanges(A_addrsNew, state);
            for (auto &A_regs: state.A_regs)
                releaseUnusedRanges(hw, A_regs, A_layoutNew, state);
            std::swap(state.A_layout, A_layoutNew);
            std::swap(state.A_addrs, A_addrsNew);
            strategy.ka_load = strategy.ka_loadTRSM;
            strategy.trimKChain(hw, strategy.ka_load, problem);
            gemmCalcIncrements(problem, strategy, state, 0, 0, true, false);

            if (state.ka_loadRem > strategy.ka_loadTRSM) {
                state.ka_loadRem = strategy.ka_loadTRSM;
                state.A_layoutRem = state.A_layout.slice(state.A_addrsRem, state.A_addrs, true, 0, state.ka_loadRem, strategy.A.padded);
            }
        }
        if (strategy.kb_load > strategy.kb_loadTRSM) {
            if (backward)
                kOffsetB = strategy.kb_loadTRSM - strategy.kb_load;
            vector<RegisterRange> B_addrsNew;
            auto B_layoutNew = state.B_layout.slice(B_addrsNew, state.B_addrs, false, 0, strategy.kb_loadTRSM, false);
            safeReleaseRanges(state.B_addrs, state);
            reclaimRanges(B_addrsNew, state);
            for (auto &B_regs: state.B_regs)
                releaseUnusedRanges(hw, B_regs, B_layoutNew, state);
            std::swap(state.B_layout, B_layoutNew);
            std::swap(state.B_addrs, B_addrsNew);
            strategy.kb_load = strategy.kb_loadTRSM;
            strategy.trimKChain(hw, strategy.kb_load, problem);
            gemmCalcIncrements(problem, strategy, state, 0, 0, false, true);

            if (state.kb_loadRem > strategy.kb_loadTRSM) {
                state.kb_loadRem = strategy.kb_loadTRSM;
                state.B_layoutRem = state.B_layout.slice(state.B_addrsRem, state.B_addrs, false, 0, state.kb_loadRem, strategy.B.padded);
            }
        }

        if (needMainReset) {
            // Re-setup all A/B addresses. Could skip if remainder loop never entered...
            kLoopReset(state.k, problem, strategy, state, true);
        }

        gemmAIncrement(state.A_layout, state.A_addrs, kOffsetA, problem, strategy, state);
        gemmBIncrement(state.B_layout, state.B_addrs, kOffsetB, problem, strategy, state);

        state.ra.safeRelease(kMain);
    };

    if (strategy.xParallel) {
        // Remove GEMM portion of cooperative triangle and do main GEMM loop.
        localX0 = state.ra.alloc_sub<uint32_t>();

        mulConstant(1, localX0, lidX, unrollX);

        auto doTRSMBias = [&](bool apply) {
            add(1, state.k, state.k, apply ? -localX0 : +localX0);
        };

        doTRSMBias(true);
        kLoop(KLoop::TRSMMainGEMMXParallel, problem, strategy, state);

        // Extra GEMM loops inside cooperative triangle.
        status << "Final GEMM section for x-parallelized TRSM" << status_stream::endl;

        Label lXPTop, lXPBottom, lXPFirstLoop, lXPSkipGEMM;

        auto storage = state.ra.alloc_sub<uint32_t>();
        auto xpLoops = storage.w(1);
        auto kTriangle = storage.uw(0);

        // Trim down A/B layouts if needed.
        trimDownAB();

        // Calculate workgroup x remainder if it's not available.
        if (wgRemX.isInvalid()) {
            wgRemX = state.ra.alloc_sub<uint32_t>();
            if (!backward)
                eadd3(1 | sat, wgRemX, localX0, x, -x0);
            else {
                mov(1 | sat, wgRemX, state.k);
                add(1 | sat, wgRemX, x, -wgRemX);
            }
            fixedWG ? min_(1, wgRemX, wgRemX, wgX * unrollX)
                    : min_(1, wgRemX, wgRemX, state.wgTileX);
        }

        // If needed, tear down V layout and create SLM V layout if not already copying V to SLM.
        if (slmVSolve && !slmVMain) {
            safeReleaseRanges(V_addrs, state);
            safeReleaseRanges(V_addrsRem, state);
            V.layout = left ? MatrixLayout::Pr : MatrixLayout::Pc;
            V.packSize = unrollY;
            V.panelLength = kv_load;
            V.crosspack = 1;
            V.setAlignment(unrollY * Tv);
            V.tileR = V.tileC = 0;
            V_strategy.base = SLM;
            V_strategy.accessType = AccessType::Block;
            V_strategy.address2D = false;
            V_strategy.newDP = (hw >= HW::XeHPG);
            V_strategy.padded = true;
            V_strategy.cachingR = V_strategy.cachingW = CacheSettingsLSC::Default;
            Vo = V;
            Vo_strategy = V_strategy;

            V_layout = RegisterLayout(hw, Tv, V_layout.rows(), V_layout.cols(), V, V_strategy);
            if (V_layout.regs() > V_regs[0].getLen()) stub();
            V_layoutRem.clear();

            allocAddrRegs(hw, V_addrs, V_layout, state);
            if (unrollX % kv_load) stub();   /* would need to set up remainder layout */

            auto V_addr0 = state.ra.alloc_sub<uint32_t>();
            uint32_t V_off0 = 0;
            if (backward)
                V_off0 += unrollY * (std::max(unrollX, unrollKSLM) - kv_load) * Tv;
            if (left && strategy.slmA)
                V_off0 += strategy.slmABufSize(problem);

            mulConstant(1, V_addr0, lidY, strategy.slmTRSMVBlockSize(problem));
            if (V_off0 != 0)
                add(1, V_addr0, V_addr0, V_off0);
            makeSLMBaseRelative(V_addr0, state);
            setupAddr(V_addrs, V_addr0, V_layout, Subregister(), strategy, state);
            state.ra.safeRelease(V_addr0);
        } else if (slmVSolve && slmVMain) {
            // May need to adjust SLM addresses from main loop due to larger
            // SLM blocks.
            int bufSizeDiff = left ? strategy.slmTRSMBBlockSize(problem) - strategy.slmBBufBlockSize(problem)
                                   : strategy.slmTRSMABlockSize(problem) - strategy.slmABufBlockSize(problem);
            if (bufSizeDiff > 0) {
                auto V_shift0 = state.ra.alloc_sub<int32_t>();
                mulConstant(1, V_shift0, lidY, bufSizeDiff >> V_layout[0].addrShift);
                incAddrShifted(V_addrs, V_shift0, V_layout, strategy, state);
                state.ra.safeRelease(V_shift0);
            }
        }

        // Activate SLM remainders if not active and they will be used in the
        //   extra GEMM loops, in case all GEMM loops are skipped.
        if (!state.remActiveSLM && (unrollX % unrollKSLM != 0))
            kLoopActivateSLMRemainder(true, true, problem, strategy, state);

        // Loop counter for extra GEMM chunks.
        add(1 | ge | state.flagAP, xpLoops, lidX, -1);

        // Calculate remaining k work to do.
        add(1, kTriangle, localX0, unrollX);
        doTRSMBias(false);
        min_(1, kTriangle, kTriangle, wgRemX);

        // Top of loop over extra GEMMs.
        jmpi(1 | ~state.flagAP, lXPBottom);

        auto V_layoutXP = left ? (state.remActiveB ? state.B_layoutRem : state.B_layout)
                               : (state.remActiveA ? state.A_layoutRem : state.A_layout);
        auto V_addrsXP =  left ? (state.remActiveB ? state.B_addrsRem  : state.B_addrs)
                               : (state.remActiveA ? state.A_addrsRem  : state.A_addrs);

        if (slmVSolve) {
            if (backward && !needMainReset && slmVMain) {
                left ? gemmBIncrement(V_layoutXP, V_addrsXP, unrollKSLM - unrollX, problem, strategy, state)
                     : gemmAIncrement(V_layoutXP, V_addrsXP, unrollKSLM - unrollX, problem, strategy, state);
            }
            jmpi(1, lXPFirstLoop);
        }

        mark(lXPTop); {
            // Reset V SLM addresses manually between loops (not auto-reset inside k loop as slmV flag is disabled).
            if (slmVSolve) {
                left ? gemmBIncrement(V_layoutXP, V_addrsXP, -unrollX, problem, strategy, state)
                     : gemmAIncrement(V_layoutXP, V_addrsXP, -unrollX, problem, strategy, state);
            }

            mark(lXPFirstLoop);

            // Barrier(s) to synchronize with kernels in solve step.
            MOCK_BARRIERS {
                GRF header;
                Subregister barriersVariable;

                if (globalUBarriers || !nbY)
                    header = kLoopGetBarrierHeader(strategy, state);

                if (globalUBarriers) {
                    // Calculate # of barriers performed by SLM U copies in the solve loop.
                    auto slmFreq = std::min(unrollKSLM, unrollX);
                    if (!is_zero_or_pow2(slmFreq) || (unrollX % slmFreq)) stub();

                    barriersVariable = state.ra.alloc_sub<int32_t>();
                    min_(1, barriersVariable, kTriangle, unrollX);
                    add(1, barriersVariable, barriersVariable, slmFreq - 1);
                    shr(1, barriersVariable, barriersVariable, ilog2(slmFreq));
                    mulConstant(1, barriersVariable, barriersVariable, strategy.slmBarriersPerUnroll());
                    add(1 | ge | state.flagAP, barriersVariable, barriersVariable, -1);
                }

                if (slmVSolve && !globalUBarriers && !trsmSkipSLMVSolveSync(problem, strategy)) {
                    // Notify solve that we are done reading V data.
                    trsmYBarrier(problem, strategy, state);
                }

                if (globalUBarriers) {
                    // Gratuitous barriers for U SLM copies in solve loop (x barrier).
                    if (nbX) stub();    /* need x barrier here */
                    Label lTop, lEnd;
                    jmpi(1 | ~state.flagAP, lEnd);
                    mark(lTop);
                    add(1 | ge | state.flagAP, barriersVariable, barriersVariable, -1);
                    barriermsg(state.modBarrierFence[0], header);
                    barrierwait();
                    jmpi(1 | state.flagAP, lTop);
                    mark(lEnd);
                }

                // Wait for V data from solve.
                if (!V_barrierFreq)
                    trsmYBarrier(problem, strategy, state);
                else {
                    // With fine-grained SLM barriers, if kTriangle <= unrollX, there will be no GEMM.
                    //   Manually synchronize with solve loop in that case.
                    Label lTop, lEnd, lEnteringGEMM;

                    cmp(1 | gt | state.flagAP, kTriangle, unrollX);
                    jmpi(1 | state.flagAP, lEnteringGEMM);

                    if (barriersVariable.isInvalid())
                        barriersVariable = state.ra.alloc_sub<int32_t>();
                    if (!is_zero_or_pow2(V_barrierFreq)) stub();

                    auto kTriangleAligned = kTriangle;
                    if (backward) {
                        kTriangleAligned = barriersVariable;
                        add(1, kTriangleAligned, kTriangle, V_barrierFreq - 1);
                    }
                    shr(1 | gt | state.flagAP, barriersVariable, kTriangleAligned, ilog2(V_barrierFreq));

                    jmpi(1 | ~state.flagAP, lEnd);
                    mark(lTop);
                    add(1 | gt | state.flagAP, barriersVariable, barriersVariable, -1);
                    trsmYBarrier(problem, strategy, state);
                    jmpi(1 | state.flagAP, lTop);
                    jmpi(1, lEnd);

                    mark(lEnteringGEMM);
                    trsmYBarrier(problem, strategy, state, KBarrierType::Signal);

                    mark(lEnd);
                }

                state.ra.safeRelease(barriersVariable);
            }

            // Calculate k size for this chunk.
            add(1 | sat | gt | state.flagAP, kTriangle, kTriangle, -unrollX);
            mov(1, state.K, unrollX);

            // If no remaining cooperative GEMM work (out of bounds), skip GEMM.
            jmpi(1 | ~state.flagAP, lXPSkipGEMM);

            // Strategy modifications:
            //   * disable SLM V copies as V data is placed into SLM by TRSM solve.
            //   * disable periodic barriers.
            //   * load U first for fine-grained solve/gemm parallelization.
            slmV = false;
            strategy.barrierFreq = 0;
            if (V_barrierFreq)
                strategy.loadBFirst = !left;
            if (slmVSolve)
                (left ? strategy.prefetchB : strategy.prefetchA) = false;

            // Fully unrolled GEMM loop.
            kLoop(KLoop::TRSMExtraGEMMs, problem, strategy, state);

            // Restore U/Ui/Uo/V addresses as needed.
            if (needXPReset)
                kLoopFixedReset(unrollX, problem, strategy, state);

            // Restore strategy.
            strategy.barrierFreq = savedStrategy.barrierFreq;
            strategy.loadBFirst = savedStrategy.loadBFirst;
            slmV = slmVMain;

            // Loop check.
            mark(lXPSkipGEMM);

            add(1 | ge | state.flagAP, xpLoops, xpLoops, -1);
            jmpi(1 | state.flagAP, lXPTop);
        }
        mark(lXPBottom);

        state.ra.safeRelease(storage);
    } else {
        // Non-x-parallel path -- single GEMM loop.
        kLoop(KLoop::TRSMMainGEMM, problem, strategy, state);
        trimDownAB();
    }

    // Incoming V load size.
    bool remActiveV = left ? state.remActiveB : state.remActiveA;
    auto kv_loadIncoming  = remActiveV ? kv_loadRem : kv_load;

    // Code for setting up V pointers with V source data.
    auto setupVSource = [&]() {
        auto &effV = left ? state.effB : state.effA;
        auto saveEffV = state.ra.alloc_sub(effV.getType());
        Subregister kOffset = state.k;
        bool freeKOff = false;

        if (backward) {
            int bAdjust = 0;

            auto adjKOff = state.ra.alloc_sub<int32_t>();
            freeKOff = true;
            mov(1 | sat, adjKOff.ud(), kOffset);
            kOffset = adjKOff;

            bAdjust -= (slmV ? unrollKSLM : kv_loadMain);
            bAdjust += kv_loadIncoming;
            if (bAdjust != 0)
                add(1, kOffset, kOffset, bAdjust);

            kOffset = -kOffset;
        }

        emov(1, saveEffV, effV, strategy, state);
        left ? gemmOffsetBk(kOffset, effV, problem.B, problem, strategy, state)
             : gemmOffsetAk(kOffset, effV, problem.A, problem, strategy, state);
        left ? setupAddr(state.B_addrs, effV, state.B_layout, state.inputs.ldb, strategy, state, state.B_params)
             : setupAddr(state.A_addrs, effV, state.A_layout, state.inputs.lda, strategy, state, state.A_params);
        emov(1, effV, saveEffV, strategy, state);

        state.ra.safeRelease(saveEffV);
        if (freeKOff)
            state.ra.safeRelease(kOffset);
    };

    if (!inplace) {
        trsmUnmoveCToV(problem, strategy, state, savedProblem, savedStrategy, savedState);

        if (!slmVSolve) {
            // Move V pointers to V source data (was solved data in C).
            // Can skip for in-place operation.
            Label lSkipMoveV;
            if (state.inputs.flags.isValid()) {
                and_(1 | nz | state.flagAP, null.ud(), state.inputs.flags, FlagInPlace);
                jmpi(1 | state.flagAP, lSkipMoveV);
            }
            setupVSource();
            mark(lSkipMoveV);
            gemmCalcIncrements(problem, strategy, state, strategy.ka_inc(), strategy.kb_inc(), !left, left);
        }
    }

    // Free up unused registers.
    if (state.remActiveA) {
        safeReleaseRanges(state.A_addrs, state);
        reclaimRanges(state.A_addrsRem, state);
        for (auto &A_regs: state.A_regs)
            releaseUnusedRanges(hw, A_regs, state.A_layoutRem, state);
        safeReleaseRanges(state.Ai_regs, state);
        reclaimRanges(state.Ai_regsRem, state);
        reclaimRanges(state.Ao_regsRem, state);
        state.A_addrs = state.A_addrsRem;
        state.Ai_regs = state.Ai_regsRem;
        state.A_layout = state.A_layoutRem;
        state.Ai_layout = state.Ai_layoutRem;
    }
    if (state.remActiveB) {
        safeReleaseRanges(state.B_addrs, state);
        reclaimRanges(state.B_addrsRem, state);
        for (auto &B_regs: state.B_regs)
            releaseUnusedRanges(hw, B_regs, state.B_layoutRem, state);
        safeReleaseRanges(state.Bi_regs, state);
        reclaimRanges(state.Bi_regsRem, state);
        reclaimRanges(state.Bo_regsRem, state);
        state.B_addrs = state.B_addrsRem;
        state.Bi_regs = state.Bi_regsRem;
        state.B_layout = state.B_layoutRem;
        state.Bi_layout = state.Bi_layoutRem;
    }

    // Perform TRSM solve.
    status << "TRSM "
           << char(left  ? 'L' : 'R')
           << char(lower ? 'L' : 'U')
           << " solve" << status_stream::endl;

    // # of loops for solve.
    if (backward && uPacked)
        ecsel(1, gt, f0[0], state.K, unrollX, remX, remX);
    else
        mov(1, state.K, remX);

    // Reserve a flag register for partially vectorized steps.
    bool globalCM = state.C_layout.colMajor();
    bool xVectorize = (globalCM == left);
    auto &fmaSIMD = strategy.fmaSIMD;
    int oldFMASIMD = fmaSIMD;

    if (xVectorize) {
        fmaSIMD = std::min(fmaSIMD, 32);
        state.flagXMask = state.raVFlag.alloc(div_up(fmaSIMD, 16));
    }

    int vr = 1, vc = 1;
    (left ? vc : vr) = unrollY;

    // Create layout for storing V back to global memory (inplace).
    if (inplace) {
        auto &V_layoutStore = state.V_layoutStore;
        auto &V_addrsStore = state.V_addrsStore;

        if (slmVSolve) {
            V          = left ? savedProblem.B  : savedProblem.A;
            V_strategy = left ? savedStrategy.B : savedStrategy.A;
        }

        V_layoutStore = RegisterLayout(hw, Tv, vr, vc, V, V_strategy, false, false, true);
        allocAddrRegs(hw, V_addrsStore, V_layoutStore, state);

        Subregister V_addr0, V_addr0Shifted;
        if (!slmVSolve) {
            // Get V address from V read layout.
            // It's already in the right place, more or less.
            int addrShift;
            V_addr0 = V_addr0Shifted = getOriginAddr(V_layout, V_addrs, &addrShift);

            if (V_layout[0].addrShift != 0) {
                V_addr0 = state.ra.alloc_sub(V_addr0.getType());
                shl(1, V_addr0, V_addr0Shifted, V_layout[0].addrShift);
            }

            int kAdjust = backward ? (kv_loadIncoming - 1) : 0;
            left ? gemmOffsetBk(kAdjust, V_addr0, problem.B, problem, strategy, state)
                 : gemmOffsetAk(kAdjust, V_addr0, problem.A, problem, strategy, state);
        } else {
            // Need to calculate V address from effVi, undoing SLM calculations.
            // If we were sure that no SLM remainders occurred, we could pull this from Vi addrs instead.
            if (!slmVMain) stub();
            Subregister viOff, viOffR, viOffC, kAdjust;
            auto &effVi = left ? state.effBi : state.effAi;
            left ? gemmCalcWorkshareBOffset(viOff, viOffR, viOffC, state.Bi, state.Bi_strategy, state.kb_slm, state.nb_slm, problem, strategy, state)
                 : gemmCalcWorkshareAOffset(viOff, viOffR, viOffC, state.Ai, state.Ai_strategy, state.ma_slm, state.ka_slm, problem, strategy, state);
            V_addr0 = state.ra.alloc_sub(effVi.getType());
            eadd(1, V_addr0, effVi, -viOff, strategy, state);

            if (backward) {
                kAdjust = viOff.d();
                add(1, kAdjust, -state.k, unrollKSLM - 1);
            } else
                kAdjust = state.k;
            left ? gemmOffsetBk(kAdjust, V_addr0, problem.B, problem, strategy, state)
                 : gemmOffsetAk(kAdjust, V_addr0, problem.A, problem, strategy, state);
            state.ra.safeRelease(viOff);
            state.ra.safeRelease(viOffR);
            state.ra.safeRelease(viOffC);
        }

        setupAddr(V_addrsStore, V_addr0, V_layoutStore, Subregister(), strategy, state);

        if (V_addr0 != V_addr0Shifted)
            state.ra.safeRelease(V_addr0);
    }

    // Create layout for copying V to SLM, allocate and set up addresses.
    if (slmVSolve) {
        if (!strategy.xParallel) stub();

        // Free old SLM V copy-related registers.
        safeReleaseRanges(Vi_addrs, state);
        safeReleaseRanges(Vi_addrsRem, state);
        safeReleaseRanges(Vi_regs, state);
        safeReleaseRanges(Vi_regsRem, state);
        safeReleaseRanges(Vo_addrs, state);
        safeReleaseRanges(Vo_regs, state);
        safeReleaseRanges(Vo_regsRem, state);

        for (auto *maskList: {&state.kMasksAi, &state.kMasksBi})
            for (auto &mask: *maskList)
                if (mask.var == loopY)
                    state.raVFlag.safeRelease(mask.flag);

        if (!inplace) {
            // Set up global V layout again.
            bool globalCM = state.C_layout.colMajor();

            if (!slmVMain || (globalCM != left && globalCM != Vi_layout.colMajor())) {
                // Need to change the access type to allow vectorization in y dimension.
                V_strategy.accessType = (isColMajor(V.layout) == globalCM) ? AccessType::Block :
                                             V_strategy.base.isStateless() ? AccessType::Scattered
                                                                           : AccessType::ChannelScattered;
            }

            if (left) {
                bool remN_B = !strategy.B.padded && (strategy.remHandling[LoopN] != RemainderHandling::Ignore);
                state.Tb_load = problem.Tb_ext;
                state.B_layout = RegisterLayout(hw, state.Tb_load, strategy.kb_load, strategy.unroll[LoopN], problem.B, strategy.B, false, remN_B);
            } else {
                bool remM_A = !strategy.A.padded && (strategy.remHandling[LoopM] != RemainderHandling::Ignore);
                state.Ta_load = problem.Ta_ext;
                state.A_layout = RegisterLayout(hw, state.Ta_load, strategy.unroll[LoopM], strategy.ka_load, problem.A, strategy.A, remM_A, false);
            }

            if (V_layout.regs() > V_regs[0].getLen()) stub();

            if (!matchAddrRegs(V_addrs, V_layout)) {
                safeReleaseRanges(V_addrs, state);
                allocAddrRegs(hw, V_addrs, V_layout, state);
            }

            auto nMasksOriginal = int(state.AB_masks.size());
            if (!assignMasks(V_layout, left ? LoopK : LoopM, left ? LoopN : LoopK, state.AB_masks, strategy, state, true)) stub();
            loadMasks(state.AB_masks, state.remainders, strategy, state, nMasksOriginal);

            setupVSource();
            gemmCalcIncrements(problem, strategy, state, strategy.ka_inc(), strategy.kb_inc(), !left, left);

            if (left)
                state.B_layoutRem = state.B_layout.slice(state.B_addrsRem, state.B_addrs, false, 0, state.kb_loadRem, strategy.B.padded);
            else
                state.A_layoutRem = state.A_layout.slice(state.A_addrsRem, state.A_addrs, true,  0, state.ka_loadRem, strategy.A.padded);
        }

        // Set up single-row/column SLM V copy layout for TRSM solve.
        Vo_strategy.accessType = AccessType::Block;
        Vo_layout.clear();

        Vo_layout = RegisterLayout(hw, Tv, vr, vc, Vo, Vo_strategy, false, false, true);

        int slmVBufBlockSize = strategy.slmTRSMVBlockSize(problem);
        int slmVBufOffset = left ? strategy.slmABufSize(problem) : 0;
        if (backward)
            slmVBufOffset += (std::max(unrollX, unrollKSLM) - 1) * unrollY * Tv;

        Subregister Vo_base = state.ra.alloc_sub<uint32_t>();
        mulConstant(1, Vo_base, lidY, slmVBufBlockSize);
        if (slmVBufOffset != 0)
            add(1, Vo_base, Vo_base, slmVBufOffset);

        allocAddrRegs(hw, Vo_addrs, Vo_layout, state);
        setupAddr(Vo_addrs, Vo_base, Vo_layout, Subregister(), strategy, state);

        state.ra.safeRelease(Vo_base);

        (left ? state.Bi_incrementalRem : state.Ai_incrementalRem) = false;
        (left ? state.Bi_remIncrCopy    : state.Ai_remIncrCopy)    = false;
        (left ? state.Bi_lateKRem       : state.Ai_lateKRem)       = false;
    }

    // Prepare for inversion subroutine if needed.
    bool invert = state.trsmInvert = !unit && !uPacked;
    if (invert) {
        int ku_load = left ? strategy.ka_load : strategy.kb_load;
        int invertSIMD = std::min(ku_load, maxInvertSubSIMD(hw, Tu));
        invertSubInit(Tu, true, strategy, state, true, invertSIMD);
    }

    // Prepare XOR mask for optional U conjugation.
    auto &flipSign = state.trsmSignFlip;
    bool mayConj = Tu.isComplex() && state.inputs.flags.isValid() && !uPacked;

    if (mayConj) {
        Label lNoConj;
        flipSign = state.ra.alloc();
        and_(1 | ze | state.flagAP, null.ud(), state.inputs.flags, left ? FlagConjA : FlagConjB);
        mov(elementsPerGRF<uint32_t>(hw), flipSign.ud(), 0);
        jmpi(1 | state.flagAP, lNoConj);
        int s = xVectorize ? 2 : 1;
        switch (Tu) {
            case Type::cbf16:
            case Type::cf16: mov(elementsPerGRF(hw, DataType::uw), flipSign.uw(s-1)(s), 0x8000); break;
            case Type::ctf32:
            case Type::cf32: mov(elementsPerGRF(hw, DataType::ud), flipSign.ud(s-1)(s), 0x80000000); break;
            case Type::cf64:
                if (strategy.emulate.emulate64)
                    mov(elementsPerGRF(hw, DataType::df), flipSign.df(s-1)(s), -0.);
                else
                    mov(elementsPerGRF(hw, DataType::uq), flipSign.uq(s-1)(s), 0x8000000000000000);
                break;
            default: stub();
        }
        mark(lNoConj);
    }

    // Prepare for optional unit diagonal.
    auto &jumpOffUnit = state.jumpOffUnit;
    bool checkUnit = invert && problem.checkUnit && state.inputs.flags.isValid();

    if (checkUnit) {
        jumpOffUnit = state.ra.alloc_sub<int32_t>();
        and_(1, jumpOffUnit, state.inputs.flags, FlagUnit);
        if (FlagUnit > 16)
            shr(1, jumpOffUnit, jumpOffUnit, ilog2(FlagUnit) - 4);
        else if (FlagUnit < 16)
            shl(1, jumpOffUnit, jumpOffUnit, 4 - ilog2(FlagUnit));
        if (isGen12)
            add(1, jumpOffUnit, jumpOffUnit, 16);
    }

    // SLM V copies: barrier to wait for previous SLM reads to complete.
    MOCK_BARRIERS if (slmVSolve && !globalUBarriers && !trsmSkipSLMVSolveSync(problem, strategy))
        trsmYBarrier(problem, strategy, state);

    // Backward SLM U copies with remainder.
    // We are potentially skipping a bunch of address increments, so we need to:
    //  * advance U pointers
    //        k -= (unrollX - k) % (unrollKSLM * std::max(strategy.slmBuffersA, strategy.slmBuffersB))
    //  * move Ui pointers rightward to align U copy chunk to unrollKSLM
    //        k += (unrollX - k) % unrollKSLM
    //  * advance Uo pointers to the proper buffer (slmBuffersA, slmBuffersB > 1)
    //        k += align_up(unrollX - k, unrollKSLM) % (unrollKSLM * std::max(strategy.slmBuffersA, strategy.slmBuffersB))
    if (slmU && backward) {
        uint16_t kWrap = unrollKSLM * std::max(strategy.slmBuffersA, strategy.slmBuffersB);
        auto ku_inc = state.ra.alloc_sub<uint32_t>();
        auto kui_inc = state.ra.alloc_sub<uint32_t>();
        auto kui_inc_eff = ku_inc;
        add(1 | sat, ku_inc, -state.K, unrollX);
        if (std::max(strategy.slmBuffersA, strategy.slmBuffersB) > 1 && unrollX > unrollKSLM)
            mod(kui_inc, ku_inc, unrollKSLM, strategy, state);
        else
            kui_inc_eff = ku_inc;
        if (unrollX > kWrap)
            mod(ku_inc, ku_inc, kWrap, strategy, state);

        left ? gemmAIncrement(state.A_layout, state.A_addrs, ku_inc.uw(), problem, strategy, state)
             : gemmBIncrement(state.B_layout, state.B_addrs, ku_inc.uw(), problem, strategy, state);
        left ? gemmAIncrement(state.Ai_layout, state.Ai_addrs, -kui_inc_eff.uw(), problem, strategy, state)
             : gemmBIncrement(state.Bi_layout, state.Bi_addrs, -kui_inc_eff.uw(), problem, strategy, state);

        if (std::max(strategy.slmBuffersA, strategy.slmBuffersB) > 1 && unrollX > unrollKSLM) {
            auto kuo_inc = ku_inc;
            add(1, kuo_inc, ku_inc, -kui_inc);
            left ? gemmAIncrement(state.Ao_layout, state.Ao_addrs, kuo_inc.uw(), problem, strategy, state)
                 : gemmBIncrement(state.Bo_layout, state.Bo_addrs, kuo_inc.uw(), problem, strategy, state);
        }

        // Also if needed calculate kSLMU now:
        //   kSLMU = -(unrollX - k) % unrollKSLM + hx_slm0
        state.kSLMCountUp = true;
        if (left) {
            if (state.Ai_incrementalRem) {
                state.kSLMA = state.kSLMStorage = state.ra.alloc_sub<int16_t>();
                gemmCalcKSLMA(problem, strategy, state, -kui_inc_eff);
            }
        } else {
            if (state.Bi_incrementalRem) {
                state.kSLMB = state.kSLMStorage = state.ra.alloc_sub<int16_t>();
                gemmCalcKSLMB(problem, strategy, state, -kui_inc_eff);
            }
        }

        state.ra.safeRelease(ku_inc);
        state.ra.safeRelease(kui_inc);
    }

    // Disable SLM V copies as we are using SLM V buffers for storing solved data.
    slmV = false;

    // Disable periodic barriers and k-chaining, and prefetches for remainder backward solves.
    strategy.barrierFreq = 0;
    strategy.kChain = 1;
    if (backward && (strategy.remHandling[loopX] != RemainderHandling::Ignore)) {
        strategy.prefetchA = strategy.prefetchAMasked = 0;
        strategy.prefetchB = strategy.prefetchBMasked = 0;
    }

    // Emit solve loop (completely unrolled).
    kLoop(KLoop::TRSMSolve, problem, strategy, state);

    // End of final y split barrier guarding SLM V loads.
    if (V_barrierFreq) {
        bool guaranteed = (backward && uPacked) || (strategy.remHandling[loopX] == RemainderHandling::Ignore);
        Label skip;
        if (!guaranteed) {
            cmp(1 | ge | state.flagAP, remX, backward ? 1 : V_barrierFreq);
            jmpi(1 | ~state.flagAP, skip);
        }
        trsmYBarrier(problem, strategy, state, KBarrierType::Wait);
        mark(skip);
    }

    // Undo strategy modifications.
    strategy.barrierFreq = savedStrategy.barrierFreq;
    strategy.kChain = savedStrategy.kChain;
    strategy.prefetchA = savedStrategy.prefetchA;
    strategy.prefetchB = savedStrategy.prefetchB;
    strategy.prefetchAMasked = savedStrategy.prefetchAMasked;
    strategy.prefetchBMasked = savedStrategy.prefetchBMasked;
    slmV = slmVMain;
    fmaSIMD = oldFMASIMD;

    // Free auxiliary registers.
    state.V_layoutStore.clear();
    safeReleaseRanges(state.V_addrsStore, state);
    state.ra.safeRelease(flipSign);
    state.ra.safeRelease(jumpOffUnit);
    state.raVFlag.safeRelease(state.flagXMask);

    auto oldBarrierY = barrierY;
    auto oldBarrierHeaderY = barrierHeaderY;

    // Teardown after k loops.
    Subregister saveLDV = ldv;
    if (!inplace) ldv = state.inputs.ldc[0];    /* temporary move C to V for Ai/Bi reset calculations */
    kLoopTeardown(problem, strategy, state);
    gemmAccumulateCTeardown(problem, strategy, state);
    if (!inplace) ldv = saveLDV;

    state.kSLMCountUp = false;

    // Extra barriers (TRSM sync + SLM copies) while other threads continue computation.
    // Must wait for after C update for out-of-place TRSMs if out of place and V is not copied to SLM.
    // The trick here is to figure out exactly how many are needed.
    bool earlyBarriers = inplace || slmVSolve;

    // Copy y named barrier objects (released in gemmAccumulateCTeardown), before they're trashed.
    Subregister saveBarrierY;
    if (nbY && strategy.xParallel) {
        barrierY = oldBarrierY;
        saveBarrierY = state.ra.alloc_sub<uint16_t>();
        auto newBarrierHeaderY = state.ra.alloc();
        mov(1, saveBarrierY, barrierY);
        mov<uint32_t>(elementsPerGRF<uint32_t>(hw), newBarrierHeaderY, oldBarrierHeaderY);
        barrierHeaderY = newBarrierHeaderY;
    }

    auto extraBarriers = [&]() {
        MOCK_BARRIERS if (strategy.xParallel) {
            Label loopTop, loopEnd;

            if (saveBarrierY.isValid()) {
                state.raVFlag.claim(barrierY);
                mov(1, barrierY, saveBarrierY);
                state.ra.safeRelease(saveBarrierY);
            }

            if (state.r0_info.isARF()) stub();
            GRF r0_info{state.r0_info.getBase()};
            GRF temp = state.ra.alloc();
            auto kOtherGEMM = temp.ud(0);
            auto kOtherSolve = temp.ud(1);
            auto temp0 = temp.ud(2);
            auto bcount = state.ra.alloc_sub<int32_t>();

            if (!trsmSkipSLMVSolveSync(problem, strategy)) {
                if (fixedWG)
                    add(1 | gt | state.flagAP, bcount, -lidX, wgX - 1);
                else {
                    add(1, bcount, state.lszX, -1);
                    add(1 | gt | state.flagAP, bcount, bcount, -lidX);
                }
            }

            if (globalUBarriers) {
                // Calculate solve work (in units of k) that other threads are doing:
                //     kOtherSolve = max(0, (wgRemX - localX0) - unrollX)
                eadd3(1 | sat, kOtherSolve, -localX0, wgRemX, -unrollX);

                // Now calculate GEMM work that other threads are doing.
                alignUp(kOtherGEMM, kOtherSolve, unrollX, strategy, state);

                // Calculate # of SLM barriers needed for GEMMs and solves.
                auto slmFreq = std::min(unrollKSLM, unrollX);
                if (!is_zero_or_pow2(slmFreq) || (unrollX % slmFreq)) stub();

                eadd3(1, temp0, kOtherSolve, kOtherGEMM, slmFreq - 1);
                shr(1, temp0, temp0, ilog2(slmFreq));
                emad(1, bcount, bcount, temp0, strategy.slmBarriersPerUnroll(), strategy, state);
            } else if (slmVSolve) {
                if (V_barrierFreq) {
                    if (!is_zero_or_pow2(V_barrierFreq) || (unrollX % V_barrierFreq)) stub();
                    eadd3(1 | sat, kOtherSolve, -localX0, wgRemX, -unrollX);
                    if (backward)
                        add(1, kOtherSolve, kOtherSolve, V_barrierFreq - 1);
                    if (trsmSkipSLMVSolveSync(problem, strategy))
                        shr(1 | gt | state.flagAP, bcount, kOtherSolve, ilog2(V_barrierFreq));
                    else {
                        shr(1, temp0, kOtherSolve, ilog2(V_barrierFreq));
                        add(1, bcount, bcount, temp0);
                    }
                } else
                    shl(1, bcount, bcount, 1);
            }

            jmpi(1 | ~state.flagAP, loopEnd);

            if (!slmVSolve)
                memfence(FenceScopeLSC::Subslice, FlushTypeLSC::None, temp, r0_info);
            else
                slmfence(temp, r0_info);
            fencewait();

            mark(loopTop);
            add(1 | gt | state.flagAP, bcount, bcount, -1);
            if (nbY) {
                if (globalUBarriers) stub();
                trsmYBarrier(problem, strategy, state);
            } else
                activeThreadBarrier(temp, r0_info, strategy);
            jmpi(1 | state.flagAP, loopTop);
            mark(loopEnd);

            state.ra.safeRelease(bcount);
            state.ra.safeRelease(temp);
            state.raVFlag.safeRelease(barrierY);
            state.ra.safeRelease(barrierHeaderY);
        }
    };

#ifdef STANDALONE
    // Finalize and store read check information in case remX = 0 or remY = 0.
    gemmFinalizeReadCheck(problem, strategy, state);
    gemmStoreReadCheck(problem, strategy, state);
#endif

    // Update C, just as in GEMM. Alpha scaling has already been done.
    problem.alpha = 1;
    if (earlyBarriers) extraBarriers();
    if (!gemmUpdateC(problem, strategy, state)) return false;
    if (!earlyBarriers) extraBarriers();

    state.ra.safeRelease(localX0);

    strategy.ka_load = savedStrategy.ka_load;
    strategy.kb_load = savedStrategy.kb_load;

    return true;    // Success!
}

// Invert diagonal of U and optionally conjugate it.
template <HW hw>
void Generator<hw>::trsmInvertDiagonal(bool doA, int h0, const RegisterLayout &U_layout, const GRFMultirange &U_regs, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    bool invert = !state.packedU;
    Label lInvert, lSkipInvert;

    auto Tu = U_layout.type();
    int r = U_layout.rows(), c = U_layout.cols();
    int nk = std::min(r, c);

    auto U_reg = [&](int hh, int cxComponent = -1) {
        int r0 = h0 + hh;
        int c0 = hh;
        if (!doA) std::swap(r0, c0);
        if (problem.backward()) {
            r0 = r - 1 - r0;
            c0 = c - 1 - c0;
        }
        return U_layout.find(r0, c0, U_regs, nullptr, nullptr, cxComponent);
    };

    auto applyConj = [&](int simd, Subregister dst, Subregister src) {
        simd *= getDwords(dst.getType());
        moveToIntPipe(src);
        moveToIntPipe(dst);
        if (state.trsmSignFlip.isValid())
            xor_(simd, dst(1), src(1), state.trsmSignFlip.sub(simd, dst.getType())(1));
        else if (src != dst)
            mov(simd, dst(1), src(1));
    };

    bool checkUnit = state.jumpOffUnit.isValid();
    if (checkUnit) {
        jmpi(1, state.jumpOffUnit);
        jmpi(1, lInvert);
        for (int hh = 0; hh < nk; hh++) {
            mov(1, U_reg(hh, 0), cast(Tu.real(), 1));
            if (Tu.isComplex())
                mov(1, U_reg(hh, 1), cast(Tu.real(), 0));
        }
        jmpi(1, lSkipInvert);
        mark(lInvert);
    }

    if (invert) {
        if (state.invertSub.use) for (int hh0 = 0; hh0 < nk;) {
            int chunk = rounddown_pow2(std::min(nk - hh0, state.invertSub.simd));
            for (int hh = 0; hh < chunk; hh++) {
                int hh1 = hh + hh0;
                if (Tu.isComplex()) {
                    mov(1, state.invertSub.srcR1[hh], U_reg(hh1, 0));
                    applyConj(1, state.invertSub.srcI1[hh], U_reg(hh1, 1));
                } else
                    mov(1, state.invertSub.src1Storage.sub(hw, hh, Tu.ngen()), U_reg(hh1));
            }
            if (state.emulate.flag != state.flagAP) {
                state.invertSub.tempFlags[0] = state.flagAP;
                kLoopModifiedFlagAP(state);
            }
            invertSub(Tu, true, state);
            for (int hh = 0; hh < chunk; hh++) {
                int hh1 = hh + hh0;
                if (Tu.isComplex()) {
                    mov(1, U_reg(hh1, 0), state.invertSub.srcR1[hh]);
                    mov(1, U_reg(hh1, 1), state.invertSub.srcI1[hh]);
                } else
                    mov(1, U_reg(hh1), state.invertSub.src1Storage.sub(hw, hh, Tu.ngen()));
            }
            hh0 += chunk;
        } else for (int hh = 0; hh < nk; hh++)
            inv(1, U_reg(hh), U_reg(hh));
    } else if (Tu.isComplex())
        for (int hh = 0; hh < nk; hh++)
            applyConj(1, U_reg(hh, 1), U_reg(hh, 1));

    if (checkUnit)
        mark(lSkipInvert);
}

template <HW hw>
void Generator<hw>::trsmSolve(int h, int ha, int hb, int opCount, const RegisterLayout &A_layout, const RegisterLayout &B_layout, const GRFMultirange &A_regs, const GRFMultirange &B_regs, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc;
    bool unit = problem.unit;
    bool left = problem.left;
    bool inplace = problem.inplace;
    bool backward = problem.backward();

    // Merge left- and right-side logic by renaming variables:
    //  U: triangular matrix
    //  V: other matrix
    //  x: outer index in U
    //  y: outer index in V
    auto loopX = left ? LoopM : LoopN;
    auto loopY = left ? LoopN : LoopM;
    auto unrollX = strategy.unroll[loopX];
    auto unrollY = strategy.unroll[loopY];
    auto &V_regs =      left ? B_regs            : A_regs;
    auto &U_layout =    left ? A_layout          : B_layout;
    auto &V_layout =    left ? B_layout          : A_layout;
    auto Tu =           left ? Ta                : Tb;
    auto Tv =           left ? Tb                : Ta;
    auto &V =           left ? problem.B         : problem.A;
    auto &V_strategy =  left ? strategy.B        : strategy.A;
    auto &Vo =          left ? state.Bo          : state.Ao;
    auto &Vo_strategy = left ? state.Bo_strategy : state.Ao_strategy;
    auto &Vo_layout =   left ? state.Bo_layout   : state.Ao_layout;
    auto &Vo_addrs =    left ? state.Bo_addrs    : state.Ao_addrs;
    int  hv =           left ? hb                : ha;
    bool slmVSolve =    strategy.slmVSolve;
    bool slmVSignal =   slmVSolve && strategy.V_barrierFreq
                            && ((h + 1) % strategy.V_barrierFreq == 0 || (h + 1) == unrollX);

    bool aComplex = Ta.isComplex();
    bool bComplex = Tb.isComplex();
    bool cComplex = Tc.isComplex();
    bool uComplex = Tu.isComplex();
    bool vComplex = Tv.isComplex();
    bool uSplitComplex = uComplex && U_layout.isSplitComplex();
    bool uPacked = state.packedU;
    bool alpha1 = problem.alpha1();
    if (!alpha1 && problem.alpha.fixed()) stub();
    bool alphaRMul = !alpha1 && !Tv.isComplex();    // Do real alpha multiplication?
    bool alphaCMul = !alpha1 &&  Tv.isComplex();    // Do complex alpha multiplication?
    bool alphaRMad = false;
    if (alphaRMul)
        alphaRMad = true, alphaRMul = false;
    auto &C_layout = state.C_layout;
    bool globalCM = C_layout.colMajor();
    bool xVectorize = (globalCM == left);
    int nec = elementsPerGRF(hw, Tc);
    int fmaSIMD = strategy.fmaSIMD;

    int x = backward ? (unrollX - 1 - h) : h;

    // No split-complex support, except for U with y-vectorization.
    if (cComplex && C_layout.isSplitComplex()) stub();
    if (uComplex && uSplitComplex && xVectorize) stub();
    if (vComplex && V_layout.isSplitComplex()) stub();

    // Temporary registers for complex alpha multiplication.
    GRFRange V1_regs;
    if (alphaCMul) V1_regs = state.ra.alloc_range(GRF::bytesToGRFs(hw, unrollY * Tv));

    // Temporary registers for gathering C.
    bool gatherC = false;
    GRFRange gatherCRange, gatherC1Range;

    if (xVectorize && unrollY * Tc.complexComponents() > 1) {
        gatherC = true;
        int nreg = GRF::bytesToGRFs(hw, unrollY * Tc);
        gatherCRange = state.ra.alloc_range(nreg);
        if (cComplex)
            gatherC1Range = state.ra.alloc_range(nreg);
    }

    // Conjugation handling.
    auto applyConj = [&](int simd, Subregister dst, Subregister src) {
        int d = getDwords(dst.getType());
        simd *= d;
        moveToIntPipe(src);
        moveToIntPipe(dst);
        if (state.trsmSignFlip.isValid())
            xor_(simd, dst(1), src(1), state.trsmSignFlip.sub(0, dst.getType())((2*d >= simd) ? 2*d : 0,2*d,1));
        else if (src != dst)
            mov(simd, dst(1), src(1));
    };

    // Solve: (1/U[x,x]) * (C_load +/- C_acc)   [out of place: + if packed, - if not]
    //        (1/U[x,x]) * C_acc                [in place]
#ifdef STANDALONE
    if (!problem.readCheck)
#endif
    for (int phase = 0; phase < 10; phase++) {
        for (int y = 0; y < unrollY; ) {
            Label labelInvert, labelSkipInvert;
            int na, nb, nc, c = 0, r = 0;
            int cc = Tc.complexComponents();

            (left ? c : r) = y;
            (left ? r : c) = x;
            Subregister A_reg = A_layout.find(r, ha,       A_regs,    &na, nullptr, 0);
            Subregister B_reg = B_layout.find(hb, c,       B_regs,    &nb, nullptr, 0);
            Subregister C_reg = C_layout.find(r,  c, state.C_regs[0], &nc, nullptr);
            Subregister Ai_reg, Bi_reg, C1_reg;
            if (aComplex) Ai_reg = A_layout.find(r, ha,       A_regs,    &na, nullptr, 1);
            if (bComplex) Bi_reg = B_layout.find(hb, c,       B_regs,    &nb, nullptr, 1);
            if (cComplex) C1_reg = C_layout.find(r,  c, state.C_regs[1], &nc, nullptr);

            auto &U_reg  = left ? A_reg  : B_reg;
            auto &Ui_reg = left ? Ai_reg : Bi_reg;
            auto &V_reg  = left ? B_reg  : A_reg;
            Subregister V1_reg;

            GRFRange C_range, C1_range, V_range, V1_range;

            auto nv = left ? nb : na;
            nc = std::min<int>({nc, nv, fmaSIMD});
            auto nregCV = (nc + nec - 1) / nec;

            if (gatherC || cComplex) {
                if (V_reg.getOffset() > 0) stub();
                V_range = GRFRange(V_reg.getBase(), nregCV);
            }

            if (cComplex) {
                if (C_reg.getOffset() > 0 || C1_reg.getOffset() > 0) stub();
                C_range  = GRFRange(C_reg.getBase(),  nregCV);
                C1_range = GRFRange(C1_reg.getBase(), nregCV);
                if (alphaCMul) {
                    V1_reg = V1_regs.sub(hw, y * 2, Tv.real().ngen());
                    V1_range = GRFRange(V1_reg.getBase(), nregCV);
                }
            }

            if (gatherC && phase == 0 && y == 0) {
                // Gather relevant C entries when vectorizing over x.
                for (int yy = 0; yy < unrollY; yy++, (left ? c++ : r++)) {
                    if (yy != 0) {
                        C_reg = C_layout.find(r, c, state.C_regs[0]);
                        if (cComplex)
                            C1_reg = C_layout.find(r, c, state.C_regs[1]);
                    }
                    applyConj(cc, gatherCRange.sub(hw, yy * cc, Tc.real().ngen()), C_reg);
                    if (cComplex)
                        applyConj(cc, gatherC1Range.sub(hw, yy * cc, Tc.real().ngen()), C1_reg);
                }
            }

            if (gatherC) {
                nc = rounddown_pow2(unrollY - y);
                C_reg = gatherCRange.sub(hw, y * cc, Tc.real().ngen());
                if (cComplex)
                    C1_reg = gatherC1Range.sub(hw, y * cc, Tc.real().ngen());
            }

            auto simd = nc * Tc.complexComponents();
            auto signedC = uPacked ? C_reg(1) : -C_reg(1);

            switch (phase) {
                case 0: if (alphaCMul)               mul(simd, V1_reg(1), V_reg(1), state.inputs.alpha_imag);                           break;
                case 1: if (alphaCMul || alphaRMul)  mul(simd, V_reg(1), V_reg(1), state.inputs.alpha_real);                            break;
                case 2: if (cComplex && !xVectorize) applyConj(simd, C1_reg, C1_reg);                                                   break;
                case 3: if (alphaCMul)               complexCombine(Tv, V_range, V1_range, strategy, state);                            break;
                case 4: if (cComplex)                complexCombine(Tc, C_range, C1_range, strategy, state);                            break;
                case 5: if (!inplace)    alphaRMad ? mad(simd, unit ? V_reg(1) : C_reg(1), signedC, V_reg(1), state.inputs.alpha_real)
                                                   : add(simd, unit ? V_reg(1) : C_reg(1), signedC, V_reg(1));                          break;
                case 6: if (!unit && uComplex)       mul(simd, C1_reg(1), C_reg(1), Ui_reg(0));                                         break;
                case 7: if (!unit)                   mul(simd, V_reg(1), C_reg(1), U_reg(0));                                           break;
                case 8: if (!unit && uComplex)       complexCombine(Tc, V_range, C1_range, strategy, state);                            break;
                case 9: if (!gatherC)                mov(simd, C_reg(1), V_reg(1));                                                     break;
            }

            if (phase == 9 && gatherC) {
                // Scatter back to C.
                for (int yy = 0; yy < nc; yy++, (left ? c++ : r++)) {
                    int simd = cc * getDwords(Tc.real().ngen());
                    C_reg = C_layout.find(r, c, state.C_regs[0]);
                    auto src = V_range.sub(hw, yy * cc, Tc.real().ngen());
                    moveToIntPipe(C_reg);
                    moveToIntPipe(src);
                    mov(simd, C_reg(1), src(1));
                }
            }

            y += nc;
        }
    }

    // Free temporary registers from diagonal computation.
    state.ra.safeRelease(V1_regs);
    state.ra.safeRelease(gatherCRange);
    state.ra.safeRelease(gatherC1Range);

    // Store V:
    //  - to global memory (out-of-place)
    //  - to SLM (if copying enabled for V)
    auto storeV = [&](const MatrixAddressing &Vx, const MatrixAddressingStrategy &Vx_strategy,
                      RegisterLayout Vx_layout, const vector<RegisterRange> &Vx_addrs)
    {
        auto V_layout1 = V_layout.unlinkedClone().slice(!left, hv, hv+1, true);
        for (auto &block: V_layout1)
            (left ? block.offsetR : block.offsetC) -= hv;

        if (Vx_layout.match(V_layout1))
            storeMatrix(V_regs, Vx_layout, Vx_addrs, strategy, state);
        else {
            auto Vx_regs = state.ra.alloc_range(Vx_layout.regs());
            copyRegisters(V_layout1, Vx_layout, V_regs, Vx_regs, strategy, state);
            storeMatrix(Vx_regs, Vx_layout, Vx_addrs, strategy, state);
            state.ra.safeRelease(Vx_regs);
        }
    };

    auto &V_layoutStore = state.V_layoutStore;
    auto &V_addrsStore = state.V_addrsStore;

    if (inplace)   storeV(V, V_strategy, V_layoutStore, V_addrsStore);
    if (slmVSolve) storeV(Vo, Vo_strategy, Vo_layout, Vo_addrs);
    if (slmVSignal) {
        GRF temp;
        temp = state.ra.alloc();
        useR0(state, [&](const GRF &r0_info) {
            int idx = (strategy.namedBarriers[LoopM] && strategy.namedBarriers[LoopN] && left) ? 1 : 0;
            slmfence(state.modBarrierFence[idx], temp, r0_info);
        });
        if (h >= strategy.V_barrierFreq) {
            Label lNoWait;
            bool backwardCheck = backward && (strategy.remHandling[loopX] != RemainderHandling::Ignore);
            if (backwardCheck) {
                cmp(1 | gt | state.flagAP, state.K, unrollX - align_down(h, strategy.V_barrierFreq));
                jmpi(1 | ~state.flagAP, lNoWait);
                kLoopModifiedFlagAP(state);
            }
            trsmYBarrier(problem, strategy, state, KBarrierType::Wait);
            if (backwardCheck)
                mark(lNoWait);
        }
        fencewait();
        trsmYBarrier(problem, strategy, state, KBarrierType::Signal);
        state.ra.safeRelease(temp);
    }

    // V address increments.
    if (h < unrollX - 1) {
        if (inplace && !isPacked(V.layout)) stub();
        if (inplace)   incDecAddr(V_addrsStore, unrollY * Tv, V_layoutStore, strategy, state, backward);
        if (slmVSolve) incDecAddr(Vo_addrs,     unrollY * Tv, Vo_layout,     strategy, state, backward);
    }

    // Propagate to following rows/columns.
#ifdef STANDALONE
    if (!problem.readCheck) {
#endif
    if (xVectorize) {
        int lastMaskOut = -1;

        // Vectorize in triangular (x) dimension. Vector operations need masking.
        for (int y = 0; y < unrollY; y++) {
            // Find V register and broadcast if necessary (double W/A).
            Subregister V_reg = left ? B_layout.find(hb, y, B_regs)
                                     : A_layout.find(y, ha, A_regs);
            Subregister Vr_src = V_reg;

            Subregister Vi_src = Vr_src;
            Vi_src.setOffset(Vr_src.getOffset() + 1);

            int xxMin = !backward ? h + 1 : 0;
            int xxMax = !backward ? unrollX : unrollX - h - 1;
            for (int xx = xxMin, maskOut = 0; xx < xxMax;) {
                int nc, nu, c = 0, r = 0;

                (left ? c : r) = y;
                (left ? r : c) = xx;

                Subregister U_reg = left ? A_layout.find(xx, ha, A_regs, &nu)
                                         : B_layout.find(hb, xx, B_regs, &nu);
                Subregister C_reg = C_layout.find(r, c, state.C_regs[0], &nc);
                Subregister C1_reg;
                if (vComplex)
                    C1_reg = C_layout.find(r, c, state.C_regs[1]);

                nc = std::min<int>({nc, nu, xxMax - xx, fmaSIMD - (xx % fmaSIMD)});

                int ncFull = roundup_pow2(nc);
                if (nc < ncFull) {
                    maskOut += ncFull - nc;
                    if (backward)
                        nc = ncFull;
                    else {
                        xx -= maskOut;
                        if (xx < 0) stub();
                        continue;
                    }
                }

                int simd = nc * Tc.complexComponents();
                InstructionModifier mod = simd;

                if (maskOut > 0) {
                    if (maskOut != lastMaskOut) {
                        // Prepare mask (TODO: move earlier)
                        int simdMO = maskOut * Tc.complexComponents();
                        uint32_t mask = backward ? (1 << (simd - simdMO)) - 1
                                                 : (1 << simd) - (1 << simdMO);
                        mov(1, state.flagXMask, mask);
                        lastMaskOut = maskOut;
                    }
                    mod |= state.flagXMask;
                    maskOut = 0;
                }

                mad(mod, C_reg(1), C_reg(1), U_reg(1), Vr_src(0));
                if (vComplex)
                    mad(mod, C1_reg(1), C1_reg(1), U_reg(1), Vi_src(0));

                xx += nc;
            }
        }
    } else {
        // Vectorize in rectangular (y) dimension.
        for (int xxi = h + 1; xxi < unrollX; xxi++) {
            int xx = backward ? (unrollX - 1 - xxi) : xxi;

            // Find U register and broadcast if necessary (double W/A).
            Subregister U_reg = left ? A_layout.find(xx, ha, A_regs, nullptr, nullptr, 0)
                                     : B_layout.find(hb, xx, B_regs, nullptr, nullptr, 0);
            Subregister Ur_src = U_reg, Ui_src;

            if (uComplex) {
                Ui_src = left ? A_layout.find(xx, ha, A_regs, nullptr, nullptr, 1)
                              : B_layout.find(hb, xx, B_regs, nullptr, nullptr, 1);
            }

            // Update the following rows/columns.
            for (int y = 0; y < unrollY; ) {
                int nc, nv, c = 0, r = 0;

                (left ? c : r) = y;
                (left ? r : c) = xx;

                Subregister V_reg = left ? B_layout.find(hb, y, B_regs, &nv)
                                         : A_layout.find(y, ha, A_regs, &nv);
                Subregister C_reg = C_layout.find(r, c, state.C_regs[0], &nc);
                Subregister C1_reg;
                if (uComplex)
                    C1_reg = C_layout.find(r, c, state.C_regs[1]);

                nc = std::min<int>({nc, nv, fmaSIMD});
                auto simd = nc * Tc.complexComponents();

                mad(simd, C_reg(1), C_reg(1), V_reg(1), Ur_src(0));
                if (uComplex)
                    mad(simd, C1_reg(1), C1_reg(1), V_reg(1), Ui_src(0));

                y += nc;
            }
        }
    }
#ifdef STANDALONE
    }
#endif
}

template <HW hw>
void Generator<hw>::trsmYBarrier(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, KBarrierType type)
{
    bool wait   = (type != KBarrierType::Signal);
    bool signal = (type != KBarrierType::Wait);
    bool left = problem.left;
    auto nbX = strategy.namedBarriers[left ? 0 : 1];
    auto nbY = strategy.namedBarriers[left ? 1 : 0];

    if (nbY) {
        if (signal) {
            auto barrierHeaderY = left ? state.barrierHeaderN : state.barrierHeaderM;
            auto modYBarrier = state.modBarrierFence[(nbX && nbY && left) ? 1 : 0];
            barriermsg(modYBarrier, barrierHeaderY);
        }
        if (wait) {
            auto barrierY = left ? state.barrierN : state.barrierM;
            sync.bar(barrierY);
        }
    } else {
        if (signal)
            barriermsg(state.modBarrierFence[0], kLoopGetBarrierHeader(strategy, state));
        if (wait)
            barrierwait();
    };
}

static inline bool trsmSkipSLMVSolveSync(const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    if (!strategy.V_barrierFreq) return false;

    auto unrollX = strategy.unroll[problem.left ? LoopM : LoopN];
    return (2 * strategy.V_barrierFreq < unrollX);
}

// Helper routine to substitute C for V.
template <HW hw>
void Generator<hw>::trsmMoveCToV(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    auto left = problem.left;

    (left ? problem.Tb            : problem.Ta           ) = problem.Tc;
    (left ? problem.B             : problem.A            ) = problem.C;
    (left ? strategy.B.base       : strategy.A.base      ) = strategy.C.base;
    (left ? state.inputs.B        : state.inputs.A       ) = state.inputs.C[0];
    (left ? state.offsetB         : state.offsetA        ) = state.offsetCV;
    (left ? state.inputs.ldb      : state.inputs.lda     ) = state.inputs.ldc[0];
    (left ? state.ldb             : state.lda            ) = state.inputs.ldc[0];
    (left ? state.inputs.surfaceB : state.inputs.surfaceA) = state.inputs.surfaceC[0];
    (left ? state.effB            : state.effA           ) = state.effCV;
#ifdef STANDALONE
    (left ? problem.readCheckB    : problem.readCheckA   ) = problem.readCheckC;
#endif

    if (left) for (auto *l: {&state.B_layout, &state.B_layoutRem}) {
        l->addressing() = problem.B;
        l->addressingStrategy() = strategy.B;
    } else for (auto *l: {&state.A_layout, &state.A_layoutRem}) {
        l->addressing() = problem.A;
        l->addressingStrategy() = strategy.A;
    }

    auto &effVp    = left ? state.effBp        : state.effAp;
    auto prefetchV = left ? strategy.prefetchB : strategy.prefetchA;
    effVp = prefetchV ? state.effCV : Subregister();

    if (problem.batch == BatchMode::Strided)
        for (int b = 0; b < problem.batchDims; b++)
            (left ? state.inputs.strideB[b] : state.inputs.strideA[b]) = state.inputs.strideC[b];
}

// Undo the effects of trsmMoveCToV.
template <HW hw>
void Generator<hw>::trsmUnmoveCToV(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state, const GEMMProblem &savedProblem, const GEMMStrategy &savedStrategy, const GEMMState &savedState)
{
    auto left = problem.left;
    auto Tv          = left ? &GEMMProblem::Tb             : &GEMMProblem::Ta;
    auto V           = left ? &GEMMProblem::B              : &GEMMProblem::A;
#ifdef STANDALONE
    auto readCheckV  = left ? &GEMMProblem::readCheckB     : &GEMMProblem::readCheckA;
#endif
    auto V_strategy  = left ? &GEMMStrategy::B             : &GEMMStrategy::A;
    auto offsetV     = left ? &GEMMState::offsetB          : &GEMMState::offsetA;
    auto ldv         = left ? &GEMMState::ldb              : &GEMMState::lda;
    auto effV        = left ? &GEMMState::effB             : &GEMMState::effA;
    auto effVp       = left ? &GEMMState::effBp            : &GEMMState::effAp;
    auto V_input     = left ? &GEMMState::Inputs::B        : &GEMMState::Inputs::A;
    auto ldv_input   = left ? &GEMMState::Inputs::ldb      : &GEMMState::Inputs::lda;
    auto surfaceV    = left ? &GEMMState::Inputs::surfaceB : &GEMMState::Inputs::surfaceA;
    auto strideV     = left ? &GEMMState::Inputs::strideB  : &GEMMState::Inputs::strideA;
    auto V_layout    = left ? &GEMMState::B_layout         : &GEMMState::A_layout;
    auto V_layoutRem = left ? &GEMMState::B_layoutRem      : &GEMMState::A_layoutRem;

    problem.*Tv = savedProblem.*Tv;
    problem.*V = savedProblem.*V;
#ifdef STANDALONE
    problem.*readCheckV = savedProblem.*readCheckV;
#endif
    strategy.*V_strategy = savedStrategy.*V_strategy;
    state.*offsetV = savedState.*offsetV;
    state.*ldv = savedState.*ldv;
    state.*effV = savedState.*effV;
    state.*effVp = savedState.*effVp;
    state.inputs.*V_input = savedState.inputs.*V_input;
    state.inputs.*ldv_input = savedState.inputs.*ldv_input;
    state.inputs.*surfaceV = savedState.inputs.*surfaceV;

    for (auto *l: {&(state.*V_layout), &(state.*V_layoutRem)}) {
        l->addressing() = problem.*V;
        l->addressingStrategy() = strategy.*V_strategy;
    }

    if (problem.batch == BatchMode::Strided)
        for (int b = 0; b < problem.batchDims; b++)
            (state.inputs.*strideV)[b] = (savedState.inputs.*strideV)[b];
}

template <HW hw>
void Generator<hw>::trsmInitState(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    state.allowEmptyC |= strategy.needsBarrier();
    gemmInitState(problem, strategy, state);
    state.isNested = true;
}


GEMMSTONE_NAMESPACE_END
