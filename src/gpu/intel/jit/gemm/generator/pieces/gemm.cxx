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


#include "cooperative_split.hpp"
#include "generator.hpp"
#include "hw_utils.hpp"
#include "kernel_queries.hpp"
#include "layout_utils.hpp"
#include "map.hpp"
#include "ngen_object_helpers.hpp"
#include "state_utils.hpp"

using namespace ngen;
using namespace ngen::utils;
using std::vector;

#include "internal/namespace_start.hxx"


// Create a GEMM kernel.
template <HW hw>
void BLASKernelGenerator<hw>::gemm(GEMMProblem problem, GEMMStrategy strategy, const InterfaceHandler &interface_)
{
    GEMMState state(hw);
    interface = interface_;
    gemm(problem, strategy, state);
}

template <HW hw>
void BLASKernelGenerator<hw>::gemm(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    const bool inFusedGEMM = false;
    bool anyKParallelFixed = strategy.kParallelLocal || strategy.kParallel;

    Label lKernelDone, lReentry, lLeavePersistentLoop, lPadThread;

    // By default, don't use dispatch mask.
    setDefaultNoMask();
    setDefaultAutoSWSB();

    // Set up.
    problem.autoTypeConversions(hw, strategy.systolic);
    gemmInitState(problem, strategy, state);

    // Transfer surface indices to strategy AddressBases.
    if (!strategy.A.base.isStateless()) strategy.A.base.setIndex(state.inputs.surfaceA);
    if (!strategy.B.base.isStateless()) strategy.B.base.setIndex(state.inputs.surfaceB);
    if (!strategy.C.base.isStateless()) {
        strategy.C.base.setIndex(state.inputs.surfaceC[0]);
        if (state.C_count > 1) stub();
        if (state.useTempC)
            state.tempCStrategy.base.setIndex(state.inputs.surfaceTempC);
    }
    if (problem.usesCO() && !strategy.CO.base.isStateless())
        strategy.CO.base.setIndex(state.inputs.surfaceCO);

    for (size_t i = 0; i < strategy.binary.size(); i++)
        if (!strategy.binary[i].base.isStateless())
            strategy.binary[i].base.setIndex(state.inputs.binarySurfaces[i]);

    // Prologue.
    if (!inFusedGEMM)
        prologue(strategy, state);

    // Grab fused ID if needed, and multiply by unroll.
    getFusedID(strategy.unroll[strategy.fusedLoop], problem, strategy, state);

    if (!inFusedGEMM) {
        // Divide out subgroup size from local size 0 and local ID 0, and reorder threads for fusing if needed.
        removeSG(problem, strategy, state);
        reorderFusedEUs(problem, strategy, state);

    } /* !inFusedGEMM */

    // Initialize flags if needed and not provided as a kernel argument.
    bool needsFlags = strategy.kParallelVariable || strategy.fuseBeta || strategy.fusePostOps
                    || problem.sumA || problem.sumB;

    if (needsFlags && !state.inputs.flags.isValid()) {
        state.inputs.flags = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        mov(1, state.inputs.flags, 0);
    }

    // Check for copy or compute kernel.
    auto wgY = strategy.wg[strategy.loopOrder[1]];
    auto &localIDY = (strategy.loopOrder[1] == LoopN) ? state.lidN : state.lidM;
    if (strategy.splitCopy) {
        state.isCompute = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        cmp(1 | ge | f1[1], state.isCompute, localIDY, wgY);
        if (is_zero_or_pow2(wgY))
            and_(1, localIDY, localIDY, wgY - 1);
        else
            add(1 | f1[1], localIDY, localIDY, -wgY);
    }

    // Check if this is a padding thread, and exit if so.
    if (strategy.wgPadFactor > 1) {
        cmp(1 | ge | f1[0], localIDY, wgY * (strategy.splitCopy ? 2 : 1));
        jmpi(1 | f1[0], lPadThread);
    }

    // Scale LDs/offsets.
    gemmScaleInputs(problem, strategy, state);

    // Local ID handling and saving.
    gemmReorderLocalIDs(problem, strategy, state);

    if (strategy.needsMNLocalIDs())
        saveMNLocalIDs(strategy, state);

    if (strategy.needsKLocalIDs())
        saveKLocalIDSize(strategy, state);

    // Save full k size if needed.
    bool anyAB2D = strategy.A.address2D || strategy.B.address2D
                || (strategy.prefetchA && strategy.A_prefetch.address2D)
                || (strategy.prefetchB && strategy.B_prefetch.address2D);
    if (anyKParallelFixed || strategy.kParallelVariable) {
        if (strategy.persistent || strategy.fusePostOps || anyAB2D) {
            state.fullK = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
            mov(1, state.fullK, state.inputs.k);
        }
    } else
        state.fullK = state.inputs.k;

    if (strategy.kParallelVariable)
        state.k0Rem = copySubregister(state.inputs.k0, state, getHint(HintType::LongTerm, strategy));

    // Surface handling for quantization parameters.
    auto replace0 = [&](Subregister &s) {
        if (s.isValid()) {
            state.ra.release(s);
            s = state.ra.alloc_sub<uint32_t>();
            mov(1, s, 0);
        }
    };

    state.A_offsetStrategy.base = A64;
    state.B_offsetStrategy.base = A64;
    state.A_scaleStrategy.base = A64;
    state.B_scaleStrategy.base = A64;

    if (problem.quantized2DA() && !strategy.A.base.isStateless()) {
        if (problem.aoPtrDims == 2) replace0(state.inputs.aoPtr);
        replace0(state.inputs.aScalePtr);
        state.A_offsetStrategy.base = state.A_scaleStrategy.base = AddressBase::createBTS(0);
    }
    if (problem.quantized2DB() && !strategy.B.base.isStateless()) {
        if (problem.boPtrDims == 2) replace0(state.inputs.boPtr);
        replace0(state.inputs.bScalePtr);
        state.B_offsetStrategy.base = state.B_scaleStrategy.base = AddressBase::createBTS(0);
    }

    // A/B offset pointer handling.
    bool aOffset = (problem.aOffset != ABOffset::None);
    bool bOffset = (problem.bOffset != ABOffset::None);
    if (aOffset && state.inputs.offsetAO.isValid())
        eadd(1, state.inputs.aoPtr, state.inputs.aoPtr, state.inputs.offsetAO, strategy, state);
    if (bOffset && state.inputs.offsetBO.isValid())
        eadd(1, state.inputs.boPtr, state.inputs.boPtr, state.inputs.offsetBO, strategy, state);

    state.ra.safeRelease(state.inputs.offsetAO);
    state.ra.safeRelease(state.inputs.offsetBO);

    // Load scalar ao/bo from memory as needed.
    bool aoScalarLoad = aOffset && problem.aoPtrDims == 0 && !problem.earlyDequantizeA();
    bool boScalarLoad = bOffset && problem.boPtrDims == 0 && !problem.earlyDequantizeB();
    auto Tc = problem.Tc;

    if (Tc.isInteger() && (aoScalarLoad || boScalarLoad)) {
        state.inputs.abo = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        if (aoScalarLoad) state.inputs.ao = state.inputs.abo.w(0);
        if (boScalarLoad) state.inputs.bo = state.inputs.abo.w(1);
    } else {
        if (aoScalarLoad) state.inputs.ao = state.ra.alloc_sub(Tc.ngen(), getHint(HintType::LongTerm, strategy));
        if (boScalarLoad) state.inputs.bo = state.ra.alloc_sub(Tc.ngen(), getHint(HintType::LongTerm, strategy));
    }

    auto loadABO = [&](Type T, const ngen::Subregister &xo, ngen::Subregister &xoPtr) {
        if (xoPtr.isInvalid())
            mov(1, xo, cast(Tc, 0));
        else {
            vector<Subregister> srcs;
            srcs.push_back(xoPtr);
            auto xoLoad = loadScalars(T, srcs, strategy, state);
            if (T.isInteger() && T.paddedSize() > 2)
                xoLoad = xoLoad.w();
            mov(1, xo, -xoLoad);
            state.ra.safeRelease(xoPtr);
            state.ra.safeRelease(xoLoad);
        }
    };

    if (aoScalarLoad) loadABO(problem.Tao, state.inputs.ao, state.inputs.aoPtr);
    if (boScalarLoad) loadABO(problem.Tbo, state.inputs.bo, state.inputs.boPtr);

    // 2D scale address handling.
    if (problem.aScale2D && state.inputs.offsetAScale.isValid())
        eadd(1, state.inputs.aScalePtr, state.inputs.aScalePtr, state.inputs.offsetAScale, strategy, state);
    if (problem.bScale2D && state.inputs.offsetBScale.isValid())
        eadd(1, state.inputs.bScalePtr, state.inputs.bScalePtr, state.inputs.offsetBScale, strategy, state);

    state.ra.safeRelease(state.inputs.offsetAScale);
    state.ra.safeRelease(state.inputs.offsetBScale);

    if (problem.aqGroupK == 0) problem.aqGroupK = strategy.ka_load;
    if (problem.bqGroupK == 0) problem.bqGroupK = strategy.kb_load;

    // Persistent thread preparation and re-entry.
    if (strategy.persistent) {
        if (!strategy.linearOrder()) stub();
        if (problem.batch != BatchMode::None) stub();       // need to wrangle groupIDK also

        auto newGroupIDMN = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        mov(1, newGroupIDMN, state.inputs.groupIDMN);
        state.inputs.groupIDMN = newGroupIDMN;

        if (state.effTempC == state.inputs.tempC)
            state.effTempC = state.ra.alloc_sub<uint64_t>(getHint(HintType::LongTerm, strategy));

        gemmFoldOffsets(problem, strategy, state);

        mark(lReentry);

    }

    // Variable k-slicing logic.
    if (strategy.kParallelVariable) {
        if (!strategy.persistent) stub();

        state.h0 = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        Label lNoKSlice, lAlreadySliced;

        auto slicedGroupIdx = state.ra.alloc_sub<int32_t>();
        auto slicedGroups   = state.ra.alloc_sub<uint32_t>();
        auto temp           = state.ra.alloc_sub<uint32_t>();
        auto temp2          = state.ra.alloc_sub<uint32_t>();
        auto temp3          = state.ra.alloc_sub<uint32_t>();
        auto gcMNStorage    = state.ra.alloc_sub<uint32_t>();
        auto kpad           = state.ra.alloc_sub<uint32_t>();

        // Check if we have reached the k-sliced region yet, and if so,
        //   if we need to do k-slicing computations.
        and_(1 | nz | f0[1], null.ud(), state.inputs.flags, FlagKSlicing);
        add(1 | ge | f0[0], slicedGroupIdx, state.inputs.groupIDMN, -state.inputs.kParallelStart);
        mov(1, state.h0, 0);
        mov(1 | lt | f1[1], temp, state.fullK);
        jmpi(1 | f0[1], lAlreadySliced);
        if (strategy.kParallelLocal)
            mov(1, state.inputs.k, state.fullK);
        jmpi(1 | ~f0[0], lNoKSlice);

        // Reverse ordering of slices so each WG traverses its tiles in reverse order.
        // For the alternate fused beta path, this helps ensure beta scaling completes
        //   before the other WGs need its result.
        eadd3(1, temp, state.inputs.groupStride, -slicedGroupIdx, -1);

        auto groupCountMN = state.inputs.groupCountMN;
        if (groupCountMN.isInvalid()) {
            groupCountMN = gcMNStorage;
            emul(1, groupCountMN, state.inputs.groupCountM, state.inputs.groupCountN, strategy, state);
        }

        // Split remaining GEMM work (slicedGroups' worth) among persistent workgroups.
        // Each workgroup gets a k0-sized range in k space, which may span
        //  multiple C workgroup tiles.
        // Divide (k0 * groupID) by k...
        //        h0 <- remainder
        //   groupID <- quotient + kParallelStart
        // Also save groupID - kParallelStart back in slicedGroupIdx.
        Subregister effKPad;
        if (strategy.kPadding)
            effKPad = gemmCalcKPadding(problem, strategy, state);
        alignUp(kpad, state.inputs.k, strategy.kAlign(problem), strategy, state);
        mul(1, temp, state.inputs.k0, temp.uw());
        if (strategy.kPadding)
            add(1, kpad, kpad, effKPad);
        or_(1, state.inputs.flags, state.inputs.flags, FlagKSlicing);
        divDown(slicedGroupIdx.ud(), temp, kpad, state.inputs.kRecip, f1[0], strategy, state);
        emad(1, state.h0, temp, -kpad, slicedGroupIdx.ud(), strategy, state);
        eadd3(1, state.inputs.groupIDMN, groupCountMN, -slicedGroupIdx.ud(), -1);
        if (strategy.altFusedBeta)
            add(1 | gt | f0[1], temp3.d(), state.h0, -state.inputs.k0);
        add(1 | le | f1[1], temp.d(), state.fullK, -state.h0);
        if (strategy.altFusedBeta) {
            eadd3(1 | ge | f1[0], temp2.d(), state.h0, state.inputs.k0, -kpad);
            cmp(1 | f0[1] | lt | f0[1], null.ud(), temp3.ud(), state.fullK);
        }
        add(1, slicedGroupIdx, state.inputs.groupIDMN, -state.inputs.kParallelStart);
        if (strategy.altFusedBeta) {
            // Decide if we are responsible for beta scaling.
            // If within padded region (h0 >= k, f1.1), scale if f0.1:
            //   - we're the first one in the padded region (h0 - k0 < k), and
            //   - no WG completely covers the computation for this tile (h0 - k0 > 0)
            // Otherwise (h0 < k, ~f1.1), scale only if no later WGs on this tile (h0 + k0 >= kpad, f1.0).
            mov(1 | f1[1], f1[0], f0[1]);
            or_(1 | f1[0], state.inputs.flags, state.inputs.flags, FlagDidBeta);
        }

        mark(lAlreadySliced);

        // In the tail case, it's possible that we don't have a full k0 chunk
        //  of work to do. Bail if so.
        // If bias enabled, do it if h0 = 0.
        cmp(1 | lt | f0[0], state.inputs.groupIDMN.d(), state.inputs.kParallelStart);
        if (problem.cOffset == COffset::Pre)
            cmp(1 | gt | f0[1], state.h0, 0);
        min_(1, state.inputs.k, temp.d(), state.k0Rem);     /* temp holds k - h0 */
            jmpi(1 | f0[0], lLeavePersistentLoop);
        if (problem.cOffset == COffset::Pre)
            or_(1 | f0[1], state.inputs.flags, state.inputs.flags, FlagNonfinalKBlock);

        // Update k0Rem with remaining work.
        if (strategy.kPadding) {
            auto temp4 = gemmCalcKPadding(problem, strategy, state);
            add(1, temp4, state.inputs.k, temp4);
            mov(1 | sat, state.inputs.k.ud(), state.inputs.k);  /* k may have been negative in padded region */
            add(1 | sat, state.k0Rem.ud(), state.k0Rem, -temp4);
            state.ra.safeRelease(temp4);
        } else
            add(1 | sat, state.k0Rem.ud(), state.k0Rem, -state.inputs.k);

        // With k padding, we might have a zero-size slice. Handle appropriately.
        if (strategy.kPadding) {
            if (strategy.altFusedBeta) {
                Label lContinue;
                jmpi(1 | ~f1[1], lContinue);
                jmpi(1 | ~f1[0], lKernelDone);                  // Skip if zero-size and not doing beta scaling.
                mark(lContinue);
            } else
                jmpi(1 | f1[1], lKernelDone);                   // Skip if zero-size.
        }

        // Beta/post-op fusing: check if we need to do it.
        if (strategy.fuseBeta || strategy.fusePostOps) {
            Label lNoCheck;

            cmp(1 | eq | f1[0], state.inputs.k, state.fullK);
            or_(1 | ~f1[0], state.inputs.flags, state.inputs.flags, FlagKPartitioned);
            jmpi(1 | f1[0], lNoCheck);
            gemmFusedBetaPOInit(slicedGroupIdx, problem, strategy, state);
            mark(lNoCheck);
        }

        mark(lNoKSlice);

        // Further slice k range within workgroup if requested.
        // k local size must be a power of 2.
        state.wgK = state.inputs.k;
        if (strategy.kParallelLocal) {
            if (!is_zero_or_pow2(strategy.wg[LoopK])) stub();
            if (strategy.kInterleave) stub();
            state.threadK0 = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
            if ((strategy.fuseBeta && !strategy.altFusedBeta) || strategy.fusePostOps) {
                state.wgK = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
                mov(1, state.wgK, state.inputs.k);
            }
            fbl(1, temp, state.lszK);
            eadd3(1, state.threadK0, state.inputs.k, state.lszK, -1);
            shr(1, state.threadK0, state.threadK0, temp);
            alignUp(state.threadK0, state.threadK0, strategy.kAlign(problem), strategy, state);
            mul(1, temp, state.threadK0, state.lidK.uw());
            add(1 | sat, state.inputs.k.ud(), state.inputs.k, -temp);
            add(1, state.h0, state.h0, temp);
            min_(1, state.inputs.k, state.inputs.k, state.threadK0);
        }

        state.ra.safeRelease(effKPad);
        state.ra.safeRelease(slicedGroupIdx);
        state.ra.safeRelease(slicedGroups);
        state.ra.safeRelease(kpad);
        state.ra.safeRelease(temp);
        state.ra.safeRelease(temp2);
        state.ra.safeRelease(temp3);
        state.ra.safeRelease(gcMNStorage);
    }

    if (strategy.kParallel && (strategy.fuseBeta || strategy.fusePostOps)) {
        if (!strategy.linearOrder()) stub();
        gemmFusedBetaPOInit(state.inputs.groupIDMN, problem, strategy, state);
    }

    // Group ID remapping.
    if (strategy.cWalkOrder == WalkOrder::SimpleLinear)
        gemmSimpleLinearOrder(problem, strategy, state);
    else if (strategy.cWalkOrder == WalkOrder::Hilbertlike)
        gemmHilbertlikeOrder(problem, strategy, state);
    else if (strategy.cWalkOrder == WalkOrder::Boustrophedon)
        gemmBoustrophedonOrder(problem, strategy, state);

    // Batch handling.
    gemmGetBatchIDs(problem, strategy, state);

    // Compute offset for A, B, C for non-strided and strided batch.
    gemmOffsetBatchABC(problem, strategy, state);

    // 32-bit add check. TODO: move out of persistent loop for non-batch.
    gemmCheck32(problem, strategy, state);

    // Calculate i0, j0, h0 -- the initial i/j/h indices for this thread.
    bool needH0 = anyKParallelFixed;

    state.i0 = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
    state.j0 = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp1, strategy));
    if (needH0 && state.h0.isInvalid())
        state.h0 = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));

    bool wgCheck = wgRemCheck(problem, strategy);
    bool gemmtBarriers = problem.gemmt() && strategy.needsBarrier();

    Subregister idM, idN, idK;

    idM = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp1, strategy));
    idN = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
    if (strategy.kParallel)
        idK = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));

    if (strategy.fixedWG(problem)) {
        mulConstant(1, idM, state.inputs.groupIDM, strategy.wg[LoopM]);
        mulConstant(1, idN, state.inputs.groupIDN, strategy.wg[LoopN]);
    } else {
        mul(1, idM, state.inputs.groupIDM, state.inputs.localSizeM.uw());
        mul(1, idN, state.inputs.groupIDN, state.inputs.localSizeN.uw());
    }
    if (strategy.kParallel) {
        mul(1, idK, state.inputs.groupIDK, state.inputs.localSizeK.uw());
        if (strategy.kPadding) {
            add(1 | lt | state.flagAP, idK.d(), idK, -state.inputs.localSizeK);
            mov(1 | state.flagAP, state.inputs.k0, 0);
        }
        if (strategy.fusePostOps) {
            state.wgK = state.inputs.k;
            if (strategy.kParallelLocal) {
                state.wgK = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
                auto temp = state.ra.alloc_sub<uint32_t>();
                emul(1, state.wgK, idK, state.inputs.k0, strategy, state);
                mul(1, temp, state.inputs.k0, state.inputs.localSizeK.uw());
                add(1 | sat, state.wgK, state.inputs.k, -state.wgK);
                min_(1, state.wgK, state.wgK, temp);
                state.ra.safeRelease(temp);
            }
        }
    }

    if (wgCheck || gemmtBarriers) {
        state.wgI0 = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
        state.wgJ0 = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp1, strategy));
        mulConstant(1, state.wgI0, idM, strategy.unroll[LoopM]);
        mulConstant(1, state.wgJ0, idN, strategy.unroll[LoopN]);
    }

    add(1, idM, idM, state.lidM);
    add(1, idN, idN, state.lidN);
    if (strategy.kParallel && !strategy.kInterleave)
        add(1 | gt | state.flagAP, idK, idK, state.lidK);

    mulConstant(1, state.i0, idM, strategy.unroll[LoopM]);
    mulConstant(1, state.j0, idN, strategy.unroll[LoopN]);

    if (strategy.kParallelVariable)
        noop(); /* h0 already calculated */
    else if (strategy.kParallelLocal && strategy.kInterleave) {
        mulConstant(1, state.h0, state.lidK, strategy.kInterleaveChunk);
        if (strategy.kParallel)
            emad(1, state.h0, state.h0, idK, state.inputs.k0, strategy, state);
    } else if (strategy.kParallel)
        emul(1, state.h0, idK, state.inputs.k0, strategy, state);
    else if (strategy.kParallelLocal)
        mul(1, state.h0, state.inputs.k0, state.lidK);

    if (strategy.kParallel) {
        if (strategy.kInterleave)
            add(1 | gt | state.flagAP, idK, idK, state.lidK);
        if (state.inputs.flags.isValid() && problem.cOffset == COffset::Pre)
            or_(1 | state.flagAP, state.inputs.flags, state.inputs.flags, FlagNonfinalKBlock);
    }

    gemmReverseLoops(problem, strategy, state);

    state.ra.safeRelease(idM);
    state.ra.safeRelease(idN);
    state.ra.safeRelease(idK);
    if (!strategy.persistent) {
        state.ra.safeRelease(state.inputs.localSizeM);
        state.ra.safeRelease(state.inputs.localSizeN);
    }
    if (anyKParallelFixed) {
        state.ra.safeRelease(state.inputs.localIDK);
        if (!strategy.persistent)
            state.ra.safeRelease(state.inputs.localSizeK);
    }
    if (strategy.linearOrder() || strategy.persistent) {
        state.ra.safeRelease(state.inputs.groupIDM);
        state.ra.safeRelease(state.inputs.groupIDN);
    }

    moveR0(strategy, state);

    // Adjust k range for local/global k-reduction.
    if (anyKParallelFixed && !strategy.kParallelVariable) {
        add(1 | sat, state.inputs.k.ud(), strategy.persistent ? state.fullK : state.inputs.k, -state.h0);

        if (strategy.kInterleave) {
            // k <- floor(k / (chunk * k local size)) * chunk + min(k % (chunk * k local size), chunk)
            auto chunk = strategy.kInterleaveChunk;
            auto wgChunk = chunk * strategy.wg[LoopK];
            if (!is_zero_or_pow2(wgChunk)) stub();
            auto temp1 = state.ra.alloc_sub<uint32_t>();
            auto temp2 = state.ra.alloc_sub<uint32_t>();

            shr(1, temp1, state.inputs.k.ud(), ilog2(strategy.wg[LoopK]));
            and_(1, temp2, state.inputs.k.ud(), wgChunk - 1);
            and_(1, temp1, temp1, ~uint32_t(chunk - 1));
            min_(1, temp2, temp2, chunk);
            add(1, state.inputs.k.ud(), temp1, temp2);

            state.ra.safeRelease(temp1);
            state.ra.safeRelease(temp2);
        }

        min_(1, state.inputs.k, state.inputs.k, state.inputs.k0);

        bool keepK0 = false;
        keepK0 |= strategy.kParallelLocal && (strategy.barrierFreq > 0 || strategy.slmBuffers > 0);
        keepK0 |= strategy.persistent;

        if (keepK0)
            state.threadK0 = state.inputs.k0;
        else
            state.ra.safeRelease(state.inputs.k0);
    }

    state.ra.safeRelease(state.inputs.localIDM);
    state.ra.safeRelease(state.inputs.localIDN);
    if (!strategy.needsMNLocalIDs())
        state.lidM = state.lidN = invalid;

    // Calculate workgroup remainders if needed.
    gemmCalcWGRemainders(problem, strategy, state);

    // Compute base addresses for A, B, C.
    auto &i0p = (strategy.coopA == CoopSplit::FullK) ? state.wgI0 : state.i0;
    auto &j0p = (strategy.coopB == CoopSplit::FullK) ? state.wgJ0 : state.j0;
    gemmOffsetABC(true, state.i0, state.j0, state.h0, i0p, j0p, problem, strategy, state);
    if (!(strategy.prefetchA && strategy.A_prefetch.address2D)) state.ra.safeRelease(state.wgI0);
    if (!(strategy.prefetchB && strategy.B_prefetch.address2D)) state.ra.safeRelease(state.wgJ0);

    // Fixed systolic kernels don't support checking for out-of-bounds panels.
    // Instead, move A/B pointers for out-of-bounds panels back in bounds.
    if (strategy.panelCheck && strategy.fixedSystolic) {
        bool checkA = strategy.A.base.isStateless();
        bool checkB = strategy.B.base.isStateless();
        if (checkA) cmp(2 | ge | f0[1], state.i0, state.inputs.m);
        if (checkB) cmp(2 | ge | f1[1], state.j0, state.inputs.n);
        if (checkA) emov(1 | f0[1], state.offsetA, 0, strategy, state);
        if (checkB) emov(1 | f1[1], state.offsetB, 0, strategy, state);
    }

    gemmSetupABC(problem, strategy, state);
    gemmSubkernel(problem, strategy, state);

    mark(lKernelDone);

    // Persistent thread loop. Advance group ID and re-enter kernel if there's more work to do.
    if (strategy.persistent) {
        status << "Persistent loop" << status_stream::endl;

        GRF temp;
        auto doBarrier = [&](const GRF &r0_info) {      /* GCC nested lambda bug */
            MOCK_BARRIERS activeThreadBarrier(temp, r0_info, strategy);
        };

        auto persistentRestore = [&] {
            gemmRestoreOffsets(problem, strategy, state);

            if (strategy.slmBuffers > 0) {
                temp = state.ra.alloc();
                useR0(state, doBarrier);
                state.ra.safeRelease(temp);
            }
        };

        uint32_t flagsToClear = 0;

        if (strategy.fuseBeta)
            flagsToClear |= FlagDidBeta | FlagSkipBetaCheck | FlagKPartitioned;
        if (strategy.fusePostOps)
            flagsToClear |= FlagKPartitioned;
        if (strategy.kParallelVariable && problem.cOffset == COffset::Pre)
            flagsToClear |= FlagNonfinalKBlock;
        if (problem.sumA || problem.sumB)
            flagsToClear |= FlagStoreSums;

        if (flagsToClear)
            and_(1, state.inputs.flags, state.inputs.flags, ~uint32_t(flagsToClear));

        if (state.movedR0 && state.r0_info != r0.ud())
            mov<uint32_t>(r0DWords(hw), r0, state.r0_info);

        if (strategy.kParallelVariable) {
            Label lNotKSliced;
            and_(1 | ze | f0[0], null.ud(), state.inputs.flags, FlagKSlicing);
            cmp(1 | gt | f1[0], state.k0Rem, 0);
            jmpi(1 | f0[0], lNotKSliced);
            add(1, state.inputs.groupIDMN, state.inputs.groupIDMN, -1);
            alignDown(1 | gt | f0[1], state.k0Rem.ud(), state.k0Rem.ud(), strategy.kAlign(problem), strategy, state);
            persistentRestore();
            jmpi(1 | f0[1], lReentry);
            jmpi(1, lLeavePersistentLoop);
            mark(lNotKSliced);
        }

        if (state.inputs.groupCountMN.isInvalid()) {
            state.inputs.groupCountMN = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
            emul(1, state.inputs.groupCountMN, state.inputs.groupCountM, state.inputs.groupCountN, strategy, state);
        }

        add(1, state.inputs.groupIDMN, state.inputs.groupIDMN, state.inputs.groupStride);
        cmp(1 | lt | f0[0], state.inputs.groupIDMN, state.inputs.groupCountMN);
        state.ra.safeRelease(state.inputs.groupCountMN);

        persistentRestore();
        jmpi(1 | f0[0], lReentry);

        if (strategy.kParallelVariable) {
            jmpi(1 | f1[0], lReentry);
            mark(lLeavePersistentLoop);
        }
    }

    mark(lPadThread);

    if (!inFusedGEMM) {
        epilogue(strategy, state);
        padding();
    }
}

template <HW hw>
void BLASKernelGenerator<hw>::gemmSubkernel(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState state)
{
    Label labelSubkernelDone, labelSubkernelEarlyExit;

    status << "Begin subkernel: unroll " << strategy.unroll[LoopM] << 'x' << strategy.unroll[LoopN] << status_stream::endl;

    // Calculate remainders for m/n loops: clamp(m - i0, 0, unrollM), clamp(n - j0, 0, unrollN).
    // Careful with this clamping, because unroll may change in remainder handling.
    bool remM = (strategy.remHandling[LoopM] != RemainderHandling::Ignore);
    bool remN = (strategy.remHandling[LoopN] != RemainderHandling::Ignore);
    bool fusedremM = remM && strategy.fused && (strategy.fusedLoop == LoopM);
    bool fusedremN = remN && strategy.fused && (strategy.fusedLoop == LoopN);

    state.doLateExit = strategy.lateExit();
    bool earlyExit = !state.doLateExit;

    if (fusedremM || fusedremN) {
        state.remFusedStorage = state.ra.alloc_sub<uint32_t>();
        add(1, state.remFusedStorage, -state.fusedID, uint16_t(strategy.unroll[strategy.fusedLoop]));
    }
    if (remM || !earlyExit) {
        state.remaindersFused[LoopM] = state.remainders[LoopM] = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        InstructionModifier mod = 1 | sat;
        if (!fusedremM && earlyExit)
            mod = mod | le | f0[1];
        add(mod, state.remainders[LoopM], -state.i0, state.inputs.m);
    }
    if (remN || !earlyExit) {
        state.remaindersFused[LoopN] = state.remainders[LoopN] = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        InstructionModifier mod = 1 | sat;
        if (!fusedremN && earlyExit)
            mod = mod | le | f1[1];
        add(mod, state.remainders[LoopN], -state.j0, state.inputs.n);
    }
    if (fusedremM || fusedremN) {
        state.remaindersFused[strategy.fusedLoop] = state.remFusedStorage;
        add(1 | sat, state.remFusedStorage, -state.remFusedStorage, state.remainders[strategy.fusedLoop]);
        if (earlyExit) {
            cmp(1 | le | (fusedremM ? f0[1] : f1[1]), null.d(), state.remainders[strategy.fusedLoop].d(), -state.fusedID);
            state.allowEmptyC = true;
        }
    }
    if (remM)
        min_(1, state.remainders[LoopM], state.remainders[LoopM], uint16_t(strategy.unroll[LoopM]));
    if (remN)
        min_(1, state.remainders[LoopN], state.remainders[LoopN], uint16_t(strategy.unroll[LoopN]));

    gemmCalcIncrements(problem, strategy, state);

    // Early exit if nothing to do. Keep fused threads together.
    if (earlyExit && (remM || remN)) {
        InstructionModifier cond;
        if (remM && remN)
            cond = 1 | f0[1] | anyv;
        else if (remM)
            cond = 1 | f0[1];
        else
            cond = 1 | f1[1];

        ejmpi(cond, labelSubkernelDone);
    }

    // Create the kernel body. If enabled, create two versions, one with A/B more aligned.
    bool success;
    int optAlignA = strategy.optAlignAB;
    int optAlignB = strategy.optAlignAB;

    // Handle block 2D alignment checks.
    if (strategy.optAlignAB2D) {
        optAlignA = std::max({
            optAlignA,
            block2DMinAlignment(hw, problem.A, strategy.A),
            block2DMinAlignment(hw, problem.A, strategy.A_prefetch)
        });
        optAlignB = std::max({
            optAlignB,
            block2DMinAlignment(hw, problem.B, strategy.B),
            block2DMinAlignment(hw, problem.B, strategy.B_prefetch)
        });
    }

    // If it's not possible to force the higher alignment, force the unaligned path.
    bool forceDowngrade = false;
    if (optAlignA && problem.A.layout == MatrixLayout::N)
        forceDowngrade |= ((strategy.unroll[LoopM] * problem.Ta) % optAlignA) != 0;
    if (optAlignB && problem.B.layout == MatrixLayout::T)
        forceDowngrade |= ((strategy.unroll[LoopN] * problem.Tb) % optAlignB) != 0;

    if (forceDowngrade) {
        optAlignA = optAlignB = 0;
        gemmDowngradeAccess(problem, strategy, state);
    }

    uint16_t maskA = (optAlignA - 1);
    uint16_t maskB = (optAlignB - 1);
    bool doA = (optAlignA && (problem.A.alignment & maskA));
    bool doB = (optAlignB && (problem.B.alignment & maskB));

    if (!doA && !doB)
        success = gemmMEdge(problem, strategy, state);
    else {
        // Check alignment of effA, effB, lda, and ldb.
        Label labelUnaligned;
        bool checkLDA = !isPacked(problem.A.layout);
        bool checkLDB = !isPacked(problem.B.layout);
        if (doA) {
            and_(1 | nz | f0[0], null.uw(), state.effA.uw(), maskA);
            if (checkLDA) and_(1 | nz | f1[0], null.uw(), state.inputs.lda.uw(), maskA);
        }
        if (doB) {
            and_(1 | nz | f0[1], null.uw(), state.effB.uw(), maskB);
            if (checkLDB) and_(1 | nz | f1[1], null.uw(), state.inputs.ldb.uw(), maskB);
        }
        if (doA) {
            InstructionModifier amod = checkLDA ? 1 | f0[0] | anyv : 1 | f0[0];
            ejmpi(amod, labelUnaligned);
        }
        if (doB) {
            InstructionModifier bmod = checkLDB ? 1 | f0[1] | anyv : 1 | f0[1];
            ejmpi(bmod, labelUnaligned);
        }
        if (strategy.optAlignAB2D) {
            if (doA) and_(1 | nz | f0[0], null.ud(), state.inputs.lda, 0xFF000000);
            if (doB) and_(1 | nz | f1[0], null.ud(), state.inputs.ldb, 0xFF000000);
            if (doA) jmpi(1 | f0[0], labelUnaligned);
            if (doB) jmpi(1 | f1[0], labelUnaligned);
        }

        auto alignedProblem = problem;
        if (doA) alignedProblem.A.setAlignment(std::max<int>(problem.A.alignment, optAlignA));
        if (doB) alignedProblem.B.setAlignment(std::max<int>(problem.B.alignment, optAlignB));

        status << "Aligned A/B" << status_stream::endl;
        success = gemmMEdge(alignedProblem, strategy, state);

        if (!success && lastException) std::rethrow_exception(lastException);

        state.isNested ? jmpi(1, labelSubkernelDone) : epilogue(strategy, state);

        mark(labelUnaligned);

        auto modStrategy = strategy;

        gemmDowngradeAccess(problem, modStrategy, state);

        status << "Unaligned A/B" << status_stream::endl;
        if (!gemmMEdge(problem, modStrategy, state)) {
            modStrategy.checkAdd32 = false;                     // Don't optimize additions on this (slow) path to reduce code size.
            status << "Reducing register usage" << status_stream::endl;
            success = success && modStrategy.minimize(hw, problem);

            gemmCalcIncrements(problem, modStrategy, state);    // Recalculate ld increments as they may have changed.

            success = success && gemmMEdge(problem, modStrategy, state);
        }
    }

    if (!success) lastException ? std::rethrow_exception(lastException)
                                : stub("Could not generate kernel.");

    mark(labelSubkernelDone);

    gemmFreeIncrements(problem, strategy, state);
}

// Handle outer-level m edge cases.
template <HW hw>
bool BLASKernelGenerator<hw>::gemmMEdge(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    if (strategy.jointSplit && strategy.remHandling[LoopM] == RemainderHandling::Split && strategy.remHandling[LoopN] == RemainderHandling::Split)
        return mnJointSplitRemainderHandling(problem, strategy, state, &BLASKernelGenerator<hw>::gemmBody);
    else
        return mnRemainderHandling(LoopM, problem, strategy, state, &BLASKernelGenerator<hw>::gemmNEdge);
}

// Handle outer-level n edge cases.
template <HW hw>
bool BLASKernelGenerator<hw>::gemmNEdge(GEMMProblem problem, GEMMStrategy strategy, GEMMState state)
{
    return mnRemainderHandling(LoopN, problem, strategy, state, &BLASKernelGenerator<hw>::gemmBody);
}

// Entrypoint for generating the GEMM kernel body, returning true if successful.
template <HW hw>
bool BLASKernelGenerator<hw>::gemmBody(GEMMProblem problem, GEMMStrategy strategy, GEMMState state)
{
    // Record whether we are in the first row/column for fused sum kernels.
    if (problem.sumA || problem.sumB) {
        if (problem.sumA && problem.sumB) stub();
        auto &y0 = problem.sumA ? state.j0 : state.i0;
        cmp(1 | eq | state.flagAP, y0, 0);
        or_(1 | state.flagAP, state.inputs.flags, state.inputs.flags, FlagStoreSums);
    }

    // Out-of-bounds panel checks.
    bool panelCheck = strategy.panelCheck && strategy.lateExit() && !strategy.fixedSystolic;
    if (panelCheck && isPacked(problem.A.layout) && strategy.remHandling[LoopM] != RemainderHandling::Ignore) {
        state.panelMaskA = state.raVFlag.alloc();
        cmp(16 | gt | state.panelMaskA, state.remainders[LoopM], 0);
    }

    if (panelCheck && isPacked(problem.B.layout) && strategy.remHandling[LoopN] != RemainderHandling::Ignore) {
        state.panelMaskB = state.raVFlag.alloc();
        cmp(16 | gt | state.panelMaskB, state.remainders[LoopN], 0);
    }

    // Release variables that are no longer needed.
    bool saveIJ0 = keepIJ0(problem, strategy), saveH0 = keepH0(problem, strategy);
    bool a2D = strategy.A.address2D || (strategy.prefetchA && strategy.A_prefetch.address2D);
    bool b2D = strategy.B.address2D || (strategy.prefetchB && strategy.B_prefetch.address2D);
    bool c2D = strategy.C.address2D || (strategy.prefetchC && strategy.C_prefetch.address2D);

    if (!a2D && !c2D && !saveIJ0) state.ra.safeRelease(state.i0);
    if (!b2D && !c2D && !saveIJ0) state.ra.safeRelease(state.j0);
    if (!a2D && !b2D && !saveH0)  state.ra.safeRelease(state.h0);
    if (!strategy.altCRemainder && !strategy.block2DCRemainder)
        releaseFusedRemainders(state);
    if (strategy.coopA != CoopSplit::FullK)
        state.ra.safeRelease(state.remaindersWG[LoopM]);
    if (strategy.coopB != CoopSplit::FullK)
        state.ra.safeRelease(state.remaindersWG[LoopN]);

    // If A/B are masked, check if we need to change ka_load/kb_load. If so, recalculate ld increments.
    if (gemmPrepMaskedAB(problem, strategy, state))
        gemmCalcIncrements(problem, strategy, state);

    // Disable C prefetch in remainder handling if it needs masks/fragmenting.
    if (strategy.remHandling[LoopM] != RemainderHandling::Ignore || strategy.remHandling[LoopN] != RemainderHandling::Ignore) {
        if (strategy.C.base.isStateless() && !strategy.C.padded && strategy.prefetchC && !isBlock2D(strategy.C_prefetch.accessType)) {
            status << "Auto-disabling C prefetch in masked region" << status_stream::endl;
            strategy.prefetchC = 0;
            if (state.effCp != state.effC[0]) state.ra.safeRelease(state.effCp);
        }
    }

    // Try generating kernel body with current strategy.
    bool success = false;
    pushStream(); try {
        success = gemmBodyInternal(problem, strategy, state);
    } catch (...) {
        lastException = std::current_exception();
    }
    success ? appendCurrentStream() : discardStream();

    return success;
}

// Generate the GEMM kernel body, returning true if successful.
template <HW hw>
bool BLASKernelGenerator<hw>::gemmBodyInternal(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    auto Tc = problem.Tc;

    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];
    auto &remM = state.remainders[LoopM];
    auto &remN = state.remainders[LoopN];

    // Accumulate C with panel*panel multiply.
    if (!gemmAccumulateC(problem, strategy, state)) return false;

    // Final C horizontal reduction for dot product kernels.
    gemmDotReduce(problem, strategy, state);

    // Add A/B offsets.
    gemmLoadABOffset(problem, strategy, state);
    if (!gemmFinalizeSums(problem, strategy, state)) return false;
    gemmApplyABOffset(problem, strategy, state);

    // If C is packed, update remainders and prepare to mask out border regions.
    bool remaskC_M = isPacked(problem.C.layout) && (strategy.remHandling[LoopM] != RemainderHandling::Ignore);
    bool remaskC_N = isPacked(problem.C.layout) && (strategy.remHandling[LoopN] != RemainderHandling::Ignore);

    if (remaskC_M || remaskC_N) {
        if (remaskC_M) setupTeardownRemask(Tc, 0, true, unrollM, remM, strategy, state);
        if (remaskC_N) setupTeardownRemask(Tc, 1, true, unrollN, remN, strategy, state);

        int C_mgran, C_ngran;
        getGranularities(problem.C, C_mgran, C_ngran);
        if (!remaskC_M || C_mgran == unrollM) C_mgran = 1;
        if (!remaskC_N || C_ngran == unrollN) C_ngran = 1;
        if (!is_zero_or_pow2(C_mgran)) stub();
        if (!is_zero_or_pow2(C_ngran)) stub();

        if (C_mgran > 1) add(1, remM, remM, C_mgran - 1);
        if (C_ngran > 1) add(1, remN, remN, C_ngran - 1);
        if (C_mgran > 1) and_(1, remM, remM, uint32_t(~(C_mgran - 1)));
        if (C_ngran > 1) and_(1, remN, remN, uint32_t(~(C_ngran - 1)));
    }

    // Local k reduction.
    if (strategy.kParallelLocal)
        gemmKReduce(problem, strategy, state);

    // Wait for fused beta completion (involves a barrier) before late exit.
    if (strategy.fuseBeta)
        gemmFusedBetaWaitCompletion(problem, strategy, state);

    // Late exit.
    Label labelLateExit;
    if (state.doLateExit && !strategy.fusePostOps && !(strategy.registerOutput() && outputCRange.empty()))
        gemmOOBExit(labelLateExit, strategy, state);

    // Handle fused post-ops for atomic update kernels.
    if (strategy.fusePostOps) {
        if (!gemmFusedPostOpsFinalize(labelLateExit, problem, strategy, state)) return false;
    }

    if (strategy.registerOutput()) {
        // Marshal C into output registers. The main path defines the output registers.
        if (outputCRange.empty()) {
            outputCRange = state.C_regs[0];
            outputCLayout = state.C_layout;
        } else {
            // FIXME: check that layouts are compatible, and rearrange if not.
            overlappedCopy(state.C_regs[0], outputCRange, state);
        }
    } else {
        // Regular C update into memory.
        if (!gemmUpdateCDispatch(problem, strategy, state)) return false;
    }

    // Cleanup.
    if (remaskC_M) setupTeardownRemask(Tc, 0, false, unrollM, remM, strategy, state);
    if (remaskC_N) setupTeardownRemask(Tc, 1, false, unrollN, remN, strategy, state);

    if (state.doLateExit || strategy.fusePostOps) {
        mark(labelLateExit);
        if (strategy.fused)
            join(16);
    }

    if (strategy.altFusedBeta && !strategy.fusePostOps)
        gemmFusedBetaNotifyCompletion(problem, strategy, state);

    return true;
}

// Perform the body of the GEMM computation, updating a block of C.
template <HW hw>
bool BLASKernelGenerator<hw>::gemmAccumulateC(GEMMProblem &problem_, GEMMStrategy &strategy_, GEMMState &state)
{
    if (strategy_.fixedSystolic) {
        if (problem_.sumA || problem_.sumB) stub();
        if (problem_.aOffset == ABOffset::Calc || problem_.bOffset == ABOffset::Calc) stub();

        return strategy_.splitCopy ? sysgemm2AccumulateC(problem_, strategy_, state)
                                   : sysgemmAccumulateC(problem_, strategy_, state);
    }

    auto problem = problem_;
    auto strategy = strategy_;

    if (!gemmAccumulateCSetup(problem, strategy, state))
        return false;

    // Synthesize k loop. If configured, choose between 32-bit adds and 64-bit adds.
    if (strategy.checkAdd32 && state.add64.isValid()) {
        Label loop64, done;
        bool success = true;

        cmp(1 | ne | state.flagAP, state.add64, uint16_t(0));
        jmpi(1 | state.flagAP, loop64);
        state.ra.safeRelease(state.add64);

        status << "k loop: 32-bit address update" << status_stream::endl;
        strategy.emulate.emulate64_add32 = true;
        auto substate32 = state;
        success &= gemmKLoop(problem, strategy, substate32);
        jmpi(1, done);

        mark(loop64);
        status << "k loop: 64-bit address update" << status_stream::endl;
        strategy.emulate.emulate64_add32 = false;
        success &= gemmKLoop(problem, strategy, state);

        mark(done);
        if (!success) return false;
    } else {
        state.ra.safeRelease(state.add64);
        if (!gemmKLoop(problem, strategy, state))
            return false;
    }

    gemmAccumulateCTeardown(problem, strategy, state);

    return true;
}

template <HW hw>
bool BLASKernelGenerator<hw>::gemmKLoop(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    return kLoopSingle(KLoop::GEMM, problem, strategy, state);
}


/**********************/
/* Remainder Handling */
/**********************/

// Synthesize a jump conditional on whether this C tile is completely outside the C matrix.
template <HW hw>
void BLASKernelGenerator<hw>::gemmOOBExit(Label &target, const GEMMStrategy &strategy, GEMMState &state)
{
    int simt = strategy.fused ? 16 : 1;

    cmp(simt | le | f0[0], state.remainders[LoopM], uint16_t(0));
    cmp(simt | le | f1[0], state.remainders[LoopN], uint16_t(0));

    InstructionModifier cond = simt | f0[0] | anyv;

    strategy.fused ? goto12(cond, target)
                   :  ejmpi(cond, target);
}

// Check whether all threads in a thread group should stay together in m/n remainder handling.
template <HW hw>
bool BLASKernelGenerator<hw>::wgRemCheck(const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    return (strategy.slmA && (effCoopSplitA(problem, strategy) == CoopSplit::MN) && (strategy.remHandling[LoopM] != RemainderHandling::Ignore) && !strategy.A.padded)
        || (strategy.slmB && (effCoopSplitB(problem, strategy) == CoopSplit::MN) && (strategy.remHandling[LoopN] != RemainderHandling::Ignore) && !strategy.B.padded)
        || strategy.kParallelLocal
        || ((strategy.barrierFreq > 0 || strategy.cooperativePF) && (strategy.prefetchA || strategy.prefetchB || strategy.prefetchC))
        || (strategy.coopA == CoopSplit::FullK)
        || (strategy.coopB == CoopSplit::FullK);
}

// Do outer-level m/n remainder handling.
template <HW hw>
template <typename Problem>
bool BLASKernelGenerator<hw>::mnRemainderHandling(LoopType loop, Problem &problem, GEMMStrategy &strategy, GEMMState &state,
                                                  bool (BLASKernelGenerator<hw>::*func)(Problem, GEMMStrategy, GEMMState))
{
    auto method = strategy.remHandling[loop];
    auto &unroll = strategy.unroll[loop];
    auto mn = (loop == LoopM) ? state.inputs.m : state.inputs.n;
    auto splitThresh = (loop == LoopM) ? strategy.mSplitThresh : strategy.nSplitThresh;

    Label label_done;

    auto originalCheckAdd32 = strategy.checkAdd32;

    if (method == RemainderHandling::Split) {
        Label label_remainder;

        // Jump to remainder loop if needed.
        // If threads fused in this direction, factor fused ID into calculation.
        if (wgRemCheck(problem, strategy))
            cmp(1 | lt | f0[0], null.d(), state.remaindersWG[loop], uint16_t(unroll * strategy.wg[loop]));
        else
            cmp(1 | lt | f0[0], null.d(), state.remaindersFused[loop], uint16_t(unroll));

        if (splitThresh) {
            cmp(1 | lt | f1[0], null.d(), mn, int32_t(splitThresh));
            ejmpi(1 | f0[0] | anyv, label_remainder);
        } else
            jmpi(1 | f0[0], label_remainder);

        // First generate code that ignores remainder handling.
        GEMMStrategy substrategy = strategy;
        substrategy.remHandling[loop] = RemainderHandling::Ignore;

        status << "Generating " << "MNK"[static_cast<int>(loop)] << " non-remainder kernel for unroll " << unroll << '.' << status_stream::endl;
        if (!(this->*func)(problem, substrategy, state)) {
            status << "Non-remainder kernel failed, aborting." << status_stream::endl;
            return false;
        }

        // Return, unless this is part of a larger computation, in which case jump to end.
        if (state.isNested)
            jmpi(1, label_done);
        else
            epilogue(strategy, state);

        mark(label_remainder);

        strategy.checkAdd32 = strategy.checkAdd32Rem();
    }

    // OK, great! Now try to create remainder-handling code.
    status << "Attempting to generate " << "MNK"[static_cast<int>(loop)] << " general kernel for unroll " << unroll << '.' << status_stream::endl;
    bool success = (this->*func)(problem, strategy, state);

    strategy.checkAdd32 = originalCheckAdd32;
    if (success) {
        mark(label_done);
        return true;
    }

#ifndef ALLOW_REMAINDERS
    // Disable remainder code for now.
    return false;
#else
    auto &bound  = (loop == LoopN) ? state.inputs.n : state.inputs.m;
    auto &index  = (loop == LoopN) ? state.j0       : state.i0;
    auto &remainders = state.remainders[loop];

    if (method == RemainderHandling::Ignore)
        stub("Could not generate kernel.");

    // It failed, so break up the loop into the next smaller power of 2 along this dimension,
    //  plus the remainder (recursively).
    Label label_next_rem;

    if (unroll == 1) {
        // No more splitting to do.
        // We don't know if this was originally split, so just output a warning.
        status << "NOTE: Split remainder handling is required for loop " << "MNK"[static_cast<int>(loop)] << '.' << status_stream::endl;
        return true;
    }
    int chunkSize = rounddown_pow2(unroll - 1);

    // Jump to next remainder loop if needed.
    pushStream(); {
        cmp(1 | lt | state.flagAP, null.d(), remainders, chunkSize);
        jmpi(1 | state.flagAP, label_next_rem);

        {
            GEMMStrategy substrategy = strategy;
            GEMMState substate = state;
            substrategy.remHandling[loop] = RemainderHandling::Ignore;
            substrategy.unroll[loop] = chunkSize;
            substate.isNested = true;
            status << "Generating " << "MNK"[static_cast<int>(loop)] << " remainder kernel with unroll " << chunkSize << '.' << status_stream::endl;
            if (!(this->*func)(problem, substrategy, substate)) {
                discardStream();
                return false;
            }
        }

        // Adjust remainder.
        add(1, remainders, remainders, -chunkSize);

        // Adjust pointers as needed.
        // A += i0 (N) i0 * lda (T, Pc)
        // B += j0 * ldb (N, Pr) j0 (T)
        // C += i0 + j0 * ldc (N, Pr) j0 + i0 * ldc (T, Pc)
        switch (loop) {
            case LoopM:
                if (problem.A.layout == MatrixLayout::N)
                    eadd(1, state.effA, state.effA, chunkSize * Ta, strategy, state);
                else {
                    Subregister temp = state.ra.alloc_sub<uint32_t>();
                    mulConstant(1, temp, state.inputs.lda, chunkSize);
                    eadd(1, state.effA, state.effA, temp, strategy, state);
                    state.ra.safeRelease(temp);
                }
                if (problem.C.layout == MatrixLayout::N || problem.C.layout == MatrixLayout::Pr)
                    eadd(1, state.effC, state.effC, chunkSize * transaction_safe, strategy, state);
                else {
                    Subregister temp = state.ra.alloc_sub<uint32_t>();
                    mulConstant(1, temp, state.inputs.lda, chunkSize);
                    eadd(1, state.effA, state.effA, temp, strategy, state);
                    state.ra.safeRelease(temp);
                }
                break;
            case LoopN:
                if (problem.B.layout == MatrixLayout::T)
                    eadd(1, state.effB, state.effB, chunkSize * Tb, strategy, state);
                else {
                    Subregister temp = state.ra.alloc_sub<uint32_t>();
                    mulConstant(1, temp, state.inputs.ldb, chunkSize);
                    eadd(1, state.effB, state.effB, temp, strategy, state);
                    state.ra.safeRelease(temp);
                }
                if (problem.C.layout == MatrixLayout::T || problem.C.layout == MatrixLayout::Pc)
                    eadd(1, state.effC, state.effC, chunkSize * Tc, strategy, state);
                else {
                    Subregister temp = state.ra.alloc_sub<uint32_t>();
                    mulConstant(1, temp, state.inputs.ldb, chunkSize);
                    eadd(1, state.effB, state.effB, temp, strategy, state);
                    state.ra.safeRelease(temp);
                }
                break;
        }

        mark(label_next_rem);

        // Handle the remainder recursively.
        {
            GEMMStrategy substrategy = strategy;
            substrategy.remHandling[loop] = RemainderHandling::General;
            substrategy.unroll[loop] -= chunkSize;
            if (!mnRemainderHandling(loop, problem, substrategy, state, func)) {
                discardStream();
                return false;
            }
        }
    } /* end stream */

    appendCurrentStream();

    return true; /* success */
#endif
}

// Synthesize remainder handling in m/n simultaneously.
template <HW hw>
template <typename Problem>
bool BLASKernelGenerator<hw>::mnJointSplitRemainderHandling(Problem &problem, GEMMStrategy &strategy, GEMMState &state,
                                                            bool (BLASKernelGenerator<hw>::*func)(Problem, GEMMStrategy, GEMMState))
{
    Label label_done, label_remainder;
    bool success = false;

    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];

    pushStream(); do {
        // Jump to remainder loop if needed:
        //  - if m/n below split thresholds (when enabled)
        //  - if in a remainder kernel.
        bool wgCheck = wgRemCheck(problem, strategy);

        if (strategy.mSplitThresh && strategy.nSplitThresh) {
            cmp(1 | lt | f0[0], null.d(), state.inputs.m, int32_t(strategy.mSplitThresh));
            cmp(1 | lt | f1[0], null.d(), state.inputs.n, int32_t(strategy.nSplitThresh));
            ejmpi(1 | f0[0] | anyv, label_remainder);
        } else if (strategy.mSplitThresh) {
            cmp(1 | lt | f0[0], null.d(), state.inputs.m, int32_t(strategy.mSplitThresh));
            jmpi(1 | f0[0], label_remainder);
        } else if (strategy.nSplitThresh) {
            cmp(1 | lt | f0[0], null.d(), state.inputs.n, int32_t(strategy.nSplitThresh));
            jmpi(1 | f0[0], label_remainder);
        }
        if (wgCheck) {
            cmp(1 | lt | f0[0], null.d(), state.remaindersWG[LoopM], uint16_t(unrollM * strategy.wg[LoopM]));
            cmp(1 | lt | f1[0], null.d(), state.remaindersWG[LoopN], uint16_t(unrollN * strategy.wg[LoopN]));
        } else {
            cmp(1 | lt | f0[0], null.d(), state.remaindersFused[LoopM], uint16_t(unrollM));
            cmp(1 | lt | f1[0], null.d(), state.remaindersFused[LoopN], uint16_t(unrollN));
        }
        ejmpi(1 | f0[0] | anyv, label_remainder);

        // First generate code that ignores remainder handling.
        GEMMStrategy substrategy = strategy;
        substrategy.remHandling[LoopM] = RemainderHandling::Ignore;
        substrategy.remHandling[LoopN] = RemainderHandling::Ignore;

        status << "Generating MN non-remainder kernel." << status_stream::endl;
        if (!(this->*func)(problem, substrategy, state)) {
            status << "Non-remainder kernel failed, aborting." << status_stream::endl;
            break;
        }

        // Return, unless this is part of a larger computation, in which case jump to end.
        if (state.isNested)
            jmpi(1, label_done);
        else
            epilogue(strategy, state);

        mark(label_remainder);

        // Finally, generate remainder handling kernel.
        substrategy = strategy;
        substrategy.remHandling[LoopM] = substrategy.remHandling[LoopN] =
            (wgCheck ? RemainderHandling::General : RemainderHandling::KnownRemainder);
        substrategy.checkAdd32 = substrategy.checkAdd32Rem();
        status << "Generating MN general kernel." << status_stream::endl;
        success = (this->*func)(problem, substrategy, state);

        mark(label_done);
    } while (false);

    success ? appendCurrentStream() : discardStream();

    return success;
}

#include "internal/namespace_end.hxx"
