/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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
using std::vector;

#include "internal/namespace_start.hxx"


// Perform alpha scaling and update problem to reflect it.
template <HW hw>
void BLASKernelGenerator<hw>::gemmAlphaScale(GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool cxCombine)
{
    auto Tacc = state.Tacc;
    auto  &alpha = problem.alpha;
    auto valphar = state.inputs.alpha_real;

    if (alpha == -1) {
        map(hw, Tacc.real(), state.C_regs[0], state.C_regs[0], strategy,
            [&](int esize, GRF acc, GRF _) {
                mov(esize, acc, -acc);
            }
        );
    } else if (alpha != 1) {
        map(hw, Tacc.real(), state.C_regs[0], state.C_regs[0], strategy,
            [&](int esize, GRF acc, GRF _) {
                alpha.fixed() ? mul(esize, acc, acc, cast(Tacc.real(), alpha))
                              : mul(esize, acc, acc, valphar.getRegAvoiding(hw, acc));
            }
        );
    }

    alpha = 1;
}

// Perform beta scaling, including type conversions.
template <HW hw>
void BLASKernelGenerator<hw>::gemmBetaScale(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    Label labelBetaDone;

    auto Ts = problem.Ts;
    auto beta = problem.beta;
    auto vbetar = state.inputs.beta_real;

    if (state.beta1.isValid()) {
        if (strategy.fused) {
            cmp(16 | lt | state.flagAP, null.d(), state.beta1, int16_t(0));
            goto12(16 | state.flagAP, labelBetaDone);
        } else {
            cmp(1 | lt | state.flagAP, null.d(), state.beta1, int16_t(0));
            jmpi(1 | state.flagAP, labelBetaDone);
        }
    }

    gemmConvertC(problem.Ts, problem, strategy, state);

    if (beta != 1) {
        map(hw, Ts.real(), state.C_regs[0], state.C_regs[0], strategy, [&](int esize, GRF acc, GRF _) {
            beta.fixed() ? mul(esize, acc, acc, cast(Ts.real(), beta))
                         : mul(esize, acc, acc, vbetar.getRegAvoiding(hw, acc));
        });

        if (problem.sumA || problem.sumB) {
            auto Tc = problem.Tc;
            auto &Xs_regs   = problem.sumA ? state.As_regs   : state.Bs_regs;
            auto &Xs_layout = problem.sumA ? state.As_layout : state.Bs_layout;

            int Xs_nregs = getRegCount(Xs_layout);
            auto Xs_usedRegs = Xs_regs.subrange(0, Xs_nregs);

            map(hw, Tc.real(), Xs_usedRegs, Xs_usedRegs, strategy, [&](int esize, GRF acc, GRF _) {
                beta.fixed() ? mul(esize, acc, acc, cast(Tc.real(), beta))
                             : mul(esize, acc, acc, vbetar.getRegAvoiding(hw, acc));
            });
        }
    }

    gemmConvertC(problem.Tc, problem, strategy, state);

    mark(labelBetaDone);

    if (state.beta1.isValid() && strategy.fused)
        join(16);
}

template <HW hw>
void BLASKernelGenerator<hw>::binaryOp(BinaryOp op, int simd, const RegData &dst, const RegData &src0, const RegData &src1, CommonState &state)
{
    switch (op) {
        case BinaryOp::Add: add(simd, dst, src0, src1); break;
        case BinaryOp::Sub: add(simd, dst, src0, -src1); break;
        case BinaryOp::Mul: mul(simd, dst, src0, src1); break;
        case BinaryOp::Div: stub();
        case BinaryOp::Min: min_(simd, dst, src0, src1); break;
        case BinaryOp::Max: max_(simd, dst, src0, src1); break;
        case BinaryOp::Prelu: {
            auto T = src1.getType();
            auto nRegs = GRF::bytesToGRFs(hw, simd * getBytes(T));
            auto tempRng = state.ra.alloc_range(nRegs);
            auto temp = tempRng[0].retype(T);
            mul(simd, temp, src0, src1);
            csel(simd | le, dst, temp, src0, src0);
            state.ra.release(tempRng);
            break;
        }
        default: stub();
    }
}

// Apply binary operation to C with a scalar operand.
template <HW hw>
void BLASKernelGenerator<hw>::gemmScalarBinaryOpC(BinaryOp op, const Subregister &offset,
                                                  const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto offsetTc = offset.reinterpret(0, state.Tacc.ngen());
    if (offset != offsetTc)
        emov(1, offsetTc, offset, strategy, state);
    if (op == BinaryOp::Div && one_of(state.Tacc, Type::f32, Type::f16)) {
        inv(1, offsetTc, offsetTc);
        op = BinaryOp::Mul;
    }

    map(hw, state.Tacc, state.C_regs[0], state.C_layout, strategy, [&](int simd, const RegData &r) {
        binaryOp(op, simd, r, r, offsetTc, state);
    });
}

// Apply binary operation to C with a vector operand, optionally multiplied by a scalar.
template <HW hw>
void BLASKernelGenerator<hw>::gemmVectorBinaryOpC(BinaryOp op, bool column, const GRFMultirange &offsets, const Subregister &scaleIn,
                                                  const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state,
                                                  Type Tco, vector<RegisterBlock> CO_layout, int y0, int y1)
{
    auto Tacc = state.Tacc;
    auto ne = elementsPerGRF(hw, Tacc);
    auto globalCM = isLayoutColMajor(state.C_layout);
    auto unrollX = strategy.unroll[globalCM ? LoopM : LoopN];
    auto unrollY = strategy.unroll[globalCM ? LoopN : LoopM];
    auto crosspack = CO_layout.empty() ? 1 : CO_layout[0].crosspack;
    auto stride = [&]() { return (column == globalCM) ? 0 : crosspack; };
    auto scale = scaleIn;
    const GRFMultirange *offsetsPtr = &offsets;

    if (Tco == Type::invalid) Tco = Tacc;

    if (Tacc == Type::f32 && scale.isValid() && scale.getType() != DataType::f) {
        scale = state.ra.alloc_sub<float>();
        mov(1, scale, scaleIn);
    }

    bool needRepack = (Tacc != Tco);
    needRepack |= (stride() > 1 && hw >= HW::XeHP && Tacc.isFP());

    GRFMultirange repackOffsets;
    if (needRepack) {
        // Repack data to unit stride as float pipe can't swizzle.
        vector<RegisterBlock> repackLayout;
        int r =  column ? 1 : strategy.unroll[LoopM];
        int c = !column ? 1 : strategy.unroll[LoopN];
        makeUnbackedRegLayout(Tacc, repackLayout, r, c, !column);
        repackOffsets = state.ra.alloc_range(getRegCount(repackLayout));
        copyRegisters(Tco, Tacc, CO_layout, repackLayout, offsets, repackOffsets, strategy, state);
        crosspack = 1;
        offsetsPtr = &repackOffsets;

        // Late inversion, for binary divide.
        if (op == BinaryOp::Div && one_of(Tacc, Type::f32, Type::f16)) {
            map(hw, Tacc, repackOffsets, repackOffsets, strategy, [&](int simd, GRF r, GRF) {
                inv(simd, r, r);
            });
            op = BinaryOp::Mul;
        }
    }

    if (y0 < 0) y0 = 0;
    if (y1 < 0) y1 = unrollY;

    for (int y = y0; y < y1; y++) {
        for (int x = 0; x < unrollX;) {
            auto i = globalCM ? x : y;
            auto j = globalCM ? y : x;
            int nc;
            const RegisterBlock *C_block;
            Subregister C = findBlockReg(Tacc, state.C_layout, i, j, state.C_regs[0], nc, C_block);

            nc = std::min({nc, strategy.fmaSIMD / crosspack, 2 * ne});
            auto nco = (column ? j : i) * crosspack;
            auto offBase = (*offsetsPtr)[nco / ne].sub(nco % ne, Tacc.ngen());
            if (scale.isValid()) {
                if (op != BinaryOp::Add) stub();
                mad(nc, C(1), C(1), offBase(stride()), scale);
            } else if (strategy.dotVL > 0) {
                // Dot-based kernels pack the C-tile into one register when possible, which results in
                // register region restriction complications. Split binary operations into execSize=1 pieces.
                for (int off = 0; off < nc; off++) {
                    auto val = C.offset(off)(1);
                    auto src1 = offBase.offset(off)(stride());
                    binaryOp(op, 1, val, val, src1, state);
                }
            } else
                binaryOp(op, nc, C(1), C(1), offBase(stride()), state);

            x += nc;
        }
    }

    if (scale != scaleIn)
        state.ra.safeRelease(scale);
    safeReleaseRanges(repackOffsets, state);
}

// Apply binary operation to C.
template <HW hw>
bool BLASKernelGenerator<hw>::gemmBinaryOpC(BinaryOp op, bool row, bool column,
                                            Type Tco, MatrixAddressing CO, MatrixAddressingStrategy CO_strategy,
                                            Subregister base, Subregister ld,
                                            const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    std::vector<GRFRange> CO_addrs;
    std::vector<RegisterBlock> CO_layout;
    std::vector<MaskAssignment> masks;
    auto globalCM = isLayoutColMajor(state.C_layout);

    bool recip = false;
    if (op == BinaryOp::Div && one_of(Tco, Type::f32, Type::f16)) {
        // Implement div as inv+mul for speed, especially when broadcasting.
        recip = true;
        op = BinaryOp::Mul;
    }

    bool matrix = row && column;
    if (matrix) {
        // Matrix case implemented as loop over rows/columns, depending on C's layout.
        row &= globalCM;
        column &= !globalCM;
        CO_strategy.accessType = (isColMajor(CO.layout) == row) ? AccessType::Block :
                                 CO_strategy.base.isStateless() ? AccessType::Scattered
                                                                : AccessType::ChannelScattered;
    } else {
        CO.layout = column ? MatrixLayout::T : MatrixLayout::N;
        CO_strategy.accessType = AccessType::Block;
    }

    bool coColMajor = isColMajor(CO.layout);

    auto cor  = row    ? strategy.unroll[LoopM] : 1;
    auto coc  = column ? strategy.unroll[LoopN] : 1;
    auto remR = row    && !CO_strategy.padded && strategy.remHandling[LoopM] != RemainderHandling::Ignore;
    auto remC = column && !CO_strategy.padded && strategy.remHandling[LoopN] != RemainderHandling::Ignore;

    if (!getRegLayout(Tco, CO_layout, cor, coc, remR, remC, false, AvoidFragment, 0, 0, CO, CO_strategy))
        return false;

    auto CO_regs = state.ra.alloc_range(getRegCount(CO_layout));

    allocAddrRegs(CO_addrs, CO_layout, CO, CO_strategy, state);
    setupAddr(Tco, CO_addrs, base, CO_layout, ld, CO, CO_strategy, strategy, state);

    if (!assignMasks(CO_layout, LoopM, LoopN, masks, strategy, state, true)) return false;

    loadMasks(masks, state.remainders, strategy, state);

    if (matrix) {
        auto LoopY = globalCM ? LoopN : LoopM;
        auto unrollY = strategy.unroll[LoopY];
        auto remY = state.remainders[LoopY];
        Label lDone;
        bool checkRemY = !CO_strategy.padded && strategy.remHandling[LoopY] != RemainderHandling::Ignore;
        bool simtCF = strategy.fused && (strategy.fusedLoop == LoopY);
        int simt = simtCF ? 16 : 1;

        if (checkRemY)
            cmp(simt | gt | state.flagAP, remY, 0);

        for (int y = 0; y < unrollY; y++) {
            if (checkRemY) {
                simtCF ? goto12(16 | ~state.flagAP, lDone)
                       :   jmpi(1  | ~state.flagAP, lDone);
            }
            loadMatrix(CO_regs, CO_layout, CO, CO_strategy, CO_addrs, strategy, state);
            if (recip) map(hw, Tco, CO_regs, CO_regs, strategy, [&](int simd, GRF r, GRF) {
                inv(simd, r, r);
            });
            if (checkRemY && (y + 1 < unrollY))
                cmp(simt | gt | state.flagAP, remY, y + 1);
            if (coColMajor == globalCM)
                incAddr(CO_addrs, ld, int(row), int(column), CO_layout, CO, CO_strategy, strategy, state);
            else
                incAddr(CO_addrs, Tco.size(), int(row), int(column), CO_layout, CO, CO_strategy, strategy, state);

            gemmVectorBinaryOpC(op, column, CO_regs, Subregister(), problem, strategy, state, Tco, CO_layout, y, y+1);
        }

        mark(lDone);
        if (simtCF) join(16);
    } else {
        loadMatrix(CO_regs, CO_layout, CO, CO_strategy, CO_addrs, strategy, state);
        if (recip) map(hw, Tco, CO_regs, CO_regs, strategy, [&](int simd, GRF r, GRF) {
            inv(simd, r, r);
        });

        if (!row && !column)
            gemmScalarBinaryOpC(op, CO_regs[0].sub(0, Tco.ngen()), problem, strategy, state);
        else
            gemmVectorBinaryOpC(op, column, CO_regs, Subregister(), problem, strategy, state, Tco, CO_layout);
    }

    safeReleaseMaskAssignments(masks, state);
    state.ra.safeRelease(CO_regs);
    safeReleaseRanges(CO_addrs, state);

    return true;
}

// Check kernel input for desired C offset and apply it.
template <HW hw>
bool BLASKernelGenerator<hw>::gemmApplyCOffsetDispatch(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    Label labelCOColumn, labelCORow, labelCOMatrix, labelCODone;
    bool doMatrix = problem.allowMatrixOffset();
    auto Tco = problem.Tco;
    auto &CO = problem.CO;
    auto &CO_strategy = strategy.CO;
    auto &effCO = state.effCO;
    auto &ldco = state.inputs.ldco;

    bool ok = true;

    if (state.flagSwizzle.isValid()) state.raVFlag.release(state.flagSwizzle);

    auto flagNonfinal = state.raVFlag.alloc();
    auto flagCOC = state.raVFlag.alloc();
    auto flagCOR = state.raVFlag.alloc();

    and_(1 | nz | flagNonfinal, null.ud(), state.inputs.flags, FlagNonfinalKBlock);
    and_(1 | nz | flagCOC, null.ud(), state.inputs.flags, FlagCOColumn);
    and_(1 | nz | flagCOR, null.ud(), state.inputs.flags, FlagCORow);
    jmpi(1 | flagNonfinal, labelCODone);
    jmpi(1 | flagCOC, labelCOColumn);
    jmpi(1 | flagCOR, labelCORow);

    state.raVFlag.safeRelease(flagNonfinal);
    state.raVFlag.safeRelease(flagCOC);
    state.raVFlag.safeRelease(flagCOR);

    if (state.flagSwizzle.isValid()) state.raVFlag.claim(state.flagSwizzle);

    status << "Applying fixed C offset" << status_stream::endl;
    ok = ok && gemmBinaryOpC(BinaryOp::Add, false, false, Tco, CO, CO_strategy, effCO, ldco, problem, strategy, state);
    jmpi(1, labelCODone);

    mark(labelCOColumn);
    if (doMatrix) jmpi(1 | flagCOR, labelCOMatrix);
    status << "Applying column-wise C offset" << status_stream::endl;
    ok = ok && gemmBinaryOpC(BinaryOp::Add, false, true, Tco, CO, CO_strategy, effCO, ldco, problem, strategy, state);
    jmpi(1, labelCODone);

    mark(labelCORow);
    status << "Applying row-wise C offset" << status_stream::endl;
    ok = ok && gemmBinaryOpC(BinaryOp::Add, true, false, Tco, CO, CO_strategy, effCO, ldco, problem, strategy, state);

    if (doMatrix) {
        jmpi(1, labelCODone);

        mark(labelCOMatrix);
        status << "Applying matrix C offset" << status_stream::endl;
        ok = ok && gemmBinaryOpC(BinaryOp::Add, true, true, Tco, CO, CO_strategy, effCO, ldco, problem, strategy, state);
    }

    mark(labelCODone);

    if (!strategy.persistent) {
        state.ra.safeRelease(ldco);
        state.ra.safeRelease(effCO);
    }

    return ok;
}

static inline BinaryOp dnnlToBinaryOp(alg_kind_t kind)
{
    switch (kind) {
        case alg_kind::binary_add:   return BinaryOp::Add;
        case alg_kind::binary_sub:   return BinaryOp::Sub;
        case alg_kind::binary_mul:   return BinaryOp::Mul;
        case alg_kind::binary_div:   return BinaryOp::Div;
        case alg_kind::binary_min:   return BinaryOp::Min;
        case alg_kind::binary_max:   return BinaryOp::Max;
        case alg_kind::binary_prelu: return BinaryOp::Prelu;
        default: stub();
    }
}

template <HW hw>
void BLASKernelGenerator<hw>::gemmLoadBinaryOpArgs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (hw < HW::XeHP) stub();

    std::vector<ngen::Subregister *> argList;
    argList.reserve(state.inputs.binaryOffsets.size() * 5);

    for (auto &r: state.inputs.binarySrcs)
        if (r.isValid()) argList.push_back(&r);
    for (auto &r: state.inputs.binaryOffsets)
        if (r.isValid()) argList.push_back(&r);
    for (auto &r: state.inputs.binaryLDs)
        if (r.isValid()) argList.push_back(&r);

    if (problem.batch == BatchMode::Strided) {
        for (auto &rs: state.inputs.binaryStrides)
            for (auto &r: rs)
                if (r.isValid()) argList.push_back(&r);
    }

    int loadBase = interface.getArgLoadBase().getBase();
    int nGRFs = 0;
    for (auto arg: argList) {
        int base = arg->getBase();
        if (base < loadBase) stub();
        nGRFs = std::max(nGRFs, base - loadBase + 1);
    }

    auto temp = state.ra.alloc();
    auto args = state.ra.alloc_range(nGRFs);

    if (state.r0_info.isARF() || state.r0_info.getBase() != 0) stub();
    loadargs(args, nGRFs, temp, false);

    int grfOffset = args.getBase() - loadBase;

    state.ra.release(args);

    for (auto arg: argList) {
        arg->setBase(arg->getBase() + grfOffset);
        state.ra.claim(*arg);
    }

    state.ra.safeRelease(temp);
}

template <HW hw>
void BLASKernelGenerator<hw>::gemmApplyPostOps(size_t poMin, size_t poMax, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (poMin >= poMax && !problem.cStochasticRound) return;

    Label lSkip;
    and_(1 | nz | state.flagAP, null.ud(), state.inputs.flags, FlagNonfinalKBlock);

    (void) gemmConvertC(problem.Ts, problem, strategy, state);

    jmpi(1 | state.flagAP, lSkip);

    // Binary preparations: load binary-related args + calculate starting addresses
    if (problem.hasBinaryPostOp() && state.effBinary.empty()) {
        auto &postOps = problem.postOps;
        size_t poCount = postOps.len();

        gemmLoadBinaryOpArgs(problem, strategy, state);

#define FOR_EACH_BINARY \
    for (size_t i = 0; i < poCount; i++) \
        if (postOps[i].is_binary())

        FOR_EACH_BINARY {
            const auto &ld = state.inputs.binaryLDs[i];
            auto T = problem.Tbinary[i];
            if (ld.isValid())
                emulConstant(1, ld, ld, T, strategy, state);
            emulConstant(1, state.inputs.binaryOffsets[i], state.inputs.binaryOffsets[i], T, strategy, state);
            if (problem.batch == BatchMode::Strided) for (int b = 0; b < problem.batchDims; b++) {
                const auto &stride = state.inputs.binaryStrides[i][b];
                if (stride.isValid())
                    emulConstant(1, stride, stride, T, strategy, state);
            }
        }

        for (int b = 0; b < problem.batchDims; b++) {
            FOR_EACH_BINARY {
                const auto &stride = state.inputs.binaryStrides[i][b];
                if (stride.isValid())
                    emul(1, stride, stride, state.batchID[b], strategy, state);
            }
        }

        for (int b = 0; b < problem.batchDims; b++) {
            FOR_EACH_BINARY {
                auto &offsetStride = state.inputs.binaryStrides[i][b];
                if (offsetStride.isValid())
                    eadd(1, state.inputs.binaryOffsets[i], state.inputs.binaryOffsets[i], offsetStride, strategy, state);
                state.ra.safeRelease(offsetStride);
            }
        }

        gemmOffsetABC(true, state.i0, state.j0, state.h0, Subregister(), Subregister(), problem, strategy, state, false, false, false, true);

        state.effBinary.resize(poCount);

        FOR_EACH_BINARY {
            if (strategy.binary[i].base.isStateless()) {
                state.effBinary[i] = state.inputs.binarySrcs[i];
                eadd(1, state.effBinary[i], state.inputs.binarySrcs[i], state.inputs.binaryOffsets[i], strategy, state);
                state.ra.safeRelease(state.inputs.binaryOffsets[i]);
            } else
                state.effBinary[i] = state.inputs.binaryOffsets[i];
        }
#undef FOR_EACH_BINARY
    }

    // Apply post-ops to all of C.
    int C_grfs[GRF::maxRegs()];
    int C_ngrf = state.C_regs[0].getLen();
    for (int r = 0; r < C_ngrf; r++)
        C_grfs[r] = state.C_regs[0][r].getBase();

    for (size_t i = poMin; i < poMax; i++) {
        auto &entry = problem.postOps[i];
        switch (entry.kind()) {
            case post_op::kind_t::eltwise: {
                using Injector = eltwise_injector_f32_t<hw>;
                if (state.Tacc != Type::f32) stub();

                int euCount = 0; /* only used for a DG2 W/A for conv */
                auto &ee = entry.as_eltwise();
                Injector injector{this, ee.alg, ee.alpha, ee.beta, ee.scale,
                                  euCount, GRFRange(), problem.postOpFwd};

                auto scratch = state.ra.try_alloc_range(injector.preferred_scratch_regs());
                if (scratch.isInvalid())
                    scratch = state.ra.alloc_range(injector.min_scratch_regs());

                injector.set_scratch(scratch);
                injector.prepare();
                injector.compute(C_grfs, C_ngrf);
                break;
            }
            case post_op::kind_t::binary: {
                auto &ld = state.inputs.binaryLDs[i];
                auto &eff = state.effBinary[i];
                auto op = dnnlToBinaryOp(entry.as_binary().alg);

                bool ok = gemmBinaryOpC(op, problem.binaryRow[i], problem.binaryCol[i],
                                        problem.Tbinary[i], problem.binary[i], strategy.binary[i],
                                        eff, ld, problem, strategy, state);
                if (!ok) stub();

                state.ra.safeRelease(ld);
                state.ra.safeRelease(eff);
                break;
            }
            default: stub();
        }
    }
    if(problem.cStochasticRound){
        using Injector = eltwise_injector_f32_t<hw>;
        int euCount = 0; /* only used for a DG2 W/A for conv */
        Injector injector{this, alg_kind::eltwise_stochastic_round, 0.0, 0.0, 1.0, 
                          euCount, GRFRange(), problem.postOpFwd};
        auto scratch = state.ra.try_alloc_range(injector.preferred_scratch_regs());
        if (scratch.isInvalid())
            scratch = state.ra.alloc_range(injector.min_scratch_regs());
        if (scratch.isInvalid())
            stub();
        
        injector.set_scratch(scratch);
        injector.prepare();
        injector.compute(C_grfs, C_ngrf, state.inputs.sroundSeed.getBase(), state.inputs.sroundSeed.getOffset(), problem.Tc_ext.ngen());
    }
        

    mark(lSkip);
}


// Calculate addresses of A/B sums in packed input data. Sums are stored at the end of each panel.
template <HW hw>
void BLASKernelGenerator<hw>::gemmCalcABOffsetAddrs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    bool doA = (problem.bOffset == ABOffset::Load);
    bool doB = (problem.aOffset == ABOffset::Load);

    auto &effAs = state.effAs;
    auto &effBs = state.effBs;

    auto Tc = problem.Tc;
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];

    if (doA && effAs.isInvalid()) effAs = state.ra.alloc_sub(state.effA.getType());
    if (doB && effBs.isInvalid()) effBs = state.ra.alloc_sub(state.effB.getType());

    if (doA) mulConstant(1, effAs.ud(), state.inputs.lda, unrollM);
    if (doB) mulConstant(1, effBs.ud(), state.inputs.ldb, unrollN);
    if (doA) add(1, effAs.ud(), effAs.ud(), -unrollM * Tc);
    if (doB) add(1, effBs.ud(), effBs.ud(), -unrollN * Tc);
    if (doA) eadd(1, effAs, effAs.ud(), state.effA, strategy, state);
    if (doB) eadd(1, effBs, effBs.ud(), state.effB, strategy, state);
}

// Load A/B sums from packed input data.
template <HW hw>
bool BLASKernelGenerator<hw>::gemmLoadABOffset(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    bool doA = (problem.bOffset == ABOffset::Load);
    bool doB = (problem.aOffset == ABOffset::Load);
    if (!doA && !doB)
        return true;

    auto Tc = problem.Tc;
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];

    MatrixAddressing As = problem.A, Bs = problem.B;
    As.crosspack = 1;
    Bs.crosspack = 1;
    As.tileR = As.tileC = 0;
    Bs.tileR = Bs.tileC = 0;

    MatrixAddressingStrategy As_strategy = strategy.A, Bs_strategy = strategy.B;
    As_strategy.accessType = AccessType::Block;
    Bs_strategy.accessType = AccessType::Block;
    As_strategy.tileR = As_strategy.tileC = 0;
    Bs_strategy.tileR = Bs_strategy.tileC = 0;
    As_strategy.dpasw = Bs_strategy.dpasw = false;

    state.As_layout.clear();
    state.Bs_layout.clear();

    bool ok = true;
    if (doA) ok = ok && getRegLayout(Tc, state.As_layout, unrollM, 1, false, false, false, AvoidFragment, 0, 0, As, As_strategy);
    if (doB) ok = ok && getRegLayout(Tc, state.Bs_layout, 1, unrollN, false, false, false, AvoidFragment, 0, 0, Bs, Bs_strategy);
    if (!ok) return false;

    state.As_regs = state.ra.alloc_range(getRegCount(state.As_layout));
    state.Bs_regs = state.ra.alloc_range(getRegCount(state.Bs_layout));

    vector<GRFRange> As_addrs, Bs_addrs;
    allocAddrRegs(As_addrs, state.As_layout, As, As_strategy, state);
    allocAddrRegs(Bs_addrs, state.Bs_layout, Bs, Bs_strategy, state);

    if (state.effAs.isInvalid() && state.effBs.isInvalid())
        gemmCalcABOffsetAddrs(problem, strategy, state);

    setupAddr(Tc, As_addrs, state.effAs, state.As_layout, Subregister(), As, As_strategy, strategy, state);
    setupAddr(Tc, Bs_addrs, state.effBs, state.Bs_layout, Subregister(), Bs, Bs_strategy, strategy, state);

    loadMatrix(state.As_regs, state.As_layout, As, As_strategy, As_addrs, strategy, state);
    loadMatrix(state.Bs_regs, state.Bs_layout, Bs, Bs_strategy, Bs_addrs, strategy, state);

    state.ra.safeRelease(state.effAs);
    state.ra.safeRelease(state.effBs);
    safeReleaseRanges(As_addrs, state);
    safeReleaseRanges(Bs_addrs, state);

    return true;
}

// Rank-1 update of matrix C.
template <HW hw>
void BLASKernelGenerator<hw>::gemmRank1UpdateC(const GRFMultirange &r, const GRFMultirange &c, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Tacc = state.Tacc;
    auto ne = elementsPerGRF(hw, Tacc);
    auto globalCM = isLayoutColMajor(state.C_layout);
    auto unrollX = strategy.unroll[globalCM ? LoopM : LoopN];
    auto unrollY = strategy.unroll[globalCM ? LoopN : LoopM];

    if (Tacc != problem.Tc) stub();

    for (int y = 0; y < unrollY; y++) {
        for (int x = 0; x < unrollX;) {
            auto i = globalCM ? x : y;
            auto j = globalCM ? y : x;
            int nc;
            const RegisterBlock *C_block;
            Subregister C = findBlockReg(Tacc, state.C_layout, i, j, state.C_regs[0], nc, C_block);

            nc = std::min({nc, strategy.fmaSIMD, 2 * ne});

            auto offR = r[i / ne].sub(i % ne, Tacc.ngen());
            auto offC = c[j / ne].sub(j % ne, Tacc.ngen());

            globalCM ? emad(nc, C(1), C(1), offR(1), offC(0), strategy, state)
                     : emad(nc, C(1), C(1), offC(1), offR(0), strategy, state);

            x += nc;
        }
    }
}

// Apply contributions from A/B offsets to C matrix, using previously loaded/computed
// A row sums and B column sums.
template <HW hw>
void BLASKernelGenerator<hw>::gemmApplyABOffset(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    bool aOffset = (problem.aOffset != ABOffset::None) && !problem.earlyDequantizeA();
    bool bOffset = (problem.bOffset != ABOffset::None) && !problem.earlyDequantizeB();
    if (!aOffset && !bOffset)
        return;

    auto Tao = problem.Tao, Tbo = problem.Tbo, Tc = problem.Tc;
    bool noFMA = (hw == HW::Gen9);

    bool aoVector = aOffset && (problem.aoPtrDims == 1);
    bool boVector = bOffset && (problem.boPtrDims == 1);
    GRFRange aoData, boData;

    auto temp = [&]() {
        if (!(aOffset && bOffset)) return Subregister{};

        auto ret = state.ra.alloc_sub(problem.Tc.ngen());

        if (!boVector) mul(1, ret, state.k, state.inputs.bo);
        else if (Tc.isFP()) mov(1, ret, state.k);
        else stub();

        return ret;
    }();

    // Two steps: (O = all-1s matrix)
    //   1) C += A * O * bo
    //   2) C += (O * B + bo * k) * ao
    // Separate paths, depending on whether A/B offsets are vector or scalar.

    if (aoVector || boVector) {
        // Vector offset path.
        if (noFMA) stub();

        Subregister aoBase, boBase;
        if (aoVector) {
            aoBase = state.ra.alloc_sub<uint64_t>();
            eaddScaled(1, aoBase, state.inputs.aoPtr, state.i0, Tao, strategy, state);
        }
        if (boVector) {
            boBase = state.ra.alloc_sub<uint64_t>();
            eaddScaled(1, boBase, state.inputs.boPtr, state.j0, Tbo, strategy, state);
        }

        if (aoVector) aoData = loadVector(Tao, Tc, aoBase, strategy.unroll[LoopM], state.remainders[LoopM], strategy, state);
        if (boVector) boData = loadVector(Tbo, Tc, boBase, strategy.unroll[LoopN], state.remainders[LoopN], strategy, state);

        state.ra.safeRelease(aoBase);
        state.ra.safeRelease(boBase);

        if (aoVector) map(hw, Tc, aoData, aoData, strategy, [&](int ne, RegData r, RegData) {
            mov(ne, r, -r);
        });
        if (boVector) map(hw, Tc, boData, boData, strategy, [&](int ne, RegData r, RegData) {
            mov(ne, r, -r);
        });

        if (aoVector && !hasFullCrosspack(state.Bs_layout, 1)) stub();
        if (boVector && !hasFullCrosspack(state.As_layout, 1)) stub();

        if (bOffset) {
            boVector ? gemmRank1UpdateC(state.As_regs, boData, problem, strategy, state)
                     : gemmVectorBinaryOpC(BinaryOp::Add, false, state.As_regs, state.inputs.bo,
                                           problem, strategy, state, problem.Tc, state.As_layout);
        }

        if (aOffset && bOffset) for (int r = 0; r < state.Bs_regs.getLen(); r++) {
            auto ne = elementsPerGRF(hw, Tc);
            auto Bs = state.Bs_regs[r];
            boVector ? emad(ne, Bs, Bs, boData[r].retype(Tc.ngen()), temp, strategy, state)
                     : add(ne, Bs, Bs, temp);
        };

        state.ra.safeRelease(temp);

        if (aOffset) {
            aoVector ? gemmRank1UpdateC(aoData, state.Bs_regs, problem, strategy, state)
                     : gemmVectorBinaryOpC(BinaryOp::Add, true,  state.Bs_regs, state.inputs.ao,
                                           problem, strategy, state, problem.Tc, state.Bs_layout);
        }
    } else {
        // Scalar offset path.
        // TODO: combine C adds into add3 on XeHP+.
        if (noFMA) {
            if (aOffset && bOffset) map(hw, Tc, state.Bs_regs, state.Bs_layout, strategy, [&](int ne, RegData r) {
                add(ne, r, r, temp);
            });
            if (bOffset) map(hw, Tc, state.As_regs, state.As_layout, strategy, [&](int ne, RegData r) {
                mul(ne, r, r, state.inputs.bo);
            });
            if (aOffset) map(hw, Tc, state.Bs_regs, state.Bs_layout, strategy, [&](int ne, RegData r) {
                mul(ne, r, r, state.inputs.ao);
            });
        } else if (aOffset && bOffset) {
            mul(1, temp, temp, state.inputs.ao);
            map(hw, Tc, state.Bs_regs, state.Bs_layout, strategy, [&](int ne, RegData r) {
                mad(ne, r, temp, r, state.inputs.ao);
            });
        }
        state.ra.safeRelease(temp);

        auto As_scale = noFMA              ? Subregister() : state.inputs.bo;
        auto Bs_scale = (noFMA || bOffset) ? Subregister() : state.inputs.ao;

        if (bOffset) gemmVectorBinaryOpC(BinaryOp::Add, false, state.As_regs, As_scale, problem, strategy, state, problem.Tc, state.As_layout);
        if (aOffset) gemmVectorBinaryOpC(BinaryOp::Add, true,  state.Bs_regs, Bs_scale, problem, strategy, state, problem.Tc, state.Bs_layout);
    }

    state.ra.safeRelease(aoData);
    state.ra.safeRelease(boData);
    safeReleaseRanges(state.As_regs, state);
    safeReleaseRanges(state.Bs_regs, state);
    if (!strategy.persistent) {
        state.ra.safeRelease(state.inputs.ao);
        state.ra.safeRelease(state.inputs.bo);
    }
    state.As_layout.clear();
    state.Bs_layout.clear();
}

#include "internal/namespace_end.hxx"
