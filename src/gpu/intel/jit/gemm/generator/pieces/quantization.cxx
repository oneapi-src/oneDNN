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
#include "hw_utils.hpp"
#include "layout_utils.hpp"
#include "map.hpp"
#include "ngen_object_helpers.hpp"
#include "quantization.hpp"

using namespace ngen;
using std::vector;

#include "internal/namespace_start.hxx"


// Prepare 2D dequantization layouts.
template <HW hw>
bool BLASKernelGenerator<hw>::gemmMake2DQuantizationLayouts(bool isA, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    int xoPtrDims = (isA ? problem.aoPtrDims : problem.boPtrDims);
    bool xo2D = (xoPtrDims == 2);
    bool xs2D = isA ? problem.aScale2D : problem.bScale2D;
    bool xoTo2D = !xo2D && (isA ? problem.aOffset == ABOffset::Calc && problem.earlyDequantizeA()
                                : problem.bOffset == ABOffset::Calc && problem.earlyDequantizeB());
    bool cColMajor = isRegisterColMajor(problem.Tc_ext, problem.C, strategy.C);

    if (!xo2D && !xoTo2D && !xs2D) return true;

    auto &X_strategy       = isA ? strategy.A             : strategy.B;
    auto &X_offsetStrategy = isA ? state.A_offsetStrategy : state.B_offsetStrategy;
    auto &X_scaleStrategy  = isA ? state.A_scaleStrategy  : state.B_scaleStrategy;
    auto &X_offsetLayout   = isA ? state.A_offsetLayout   : state.B_offsetLayout;
    auto &X_scaleLayout    = isA ? state.A_scaleLayout    : state.B_scaleLayout;
    auto &Xr_offsetLayout  = isA ? state.Ar_offsetLayout  : state.Br_offsetLayout;
    auto &Xr_scaleLayout   = isA ? state.Ar_scaleLayout   : state.Br_scaleLayout;

    auto &XO         = isA ? problem.AO         : problem.BO;
    auto &XS         = isA ? problem.A_scale    : problem.B_scale;
    auto Tx_ext      = isA ? problem.Ta_ext     : problem.Tb_ext;
    auto Tx          = isA ? problem.Ta         : problem.Tb;
    auto Txo         = isA ? problem.Tao        : problem.Tbo;
    auto Txs         = isA ? problem.Ta_scale   : problem.Tb_scale;
    auto &Txo_int    = isA ? state.Tao_int      : state.Tbo_int;
    auto &Txs_int    = isA ? state.Ta_scaleInt  : state.Tb_scaleInt;
    auto &Tx_scaleOp = isA ? state.Ta_scaleOp   : state.Tb_scaleOp;
    auto &lateScale  = isA ? state.lateScale2DA : state.lateScale2DB;

    bool Txs_bf = Txs == Type::bf16;
    Tx_scaleOp = (Txs_bf ? Type(Tx_ext.isInt4() ? Type::f16 : Type::f32) : Txs);
    Txo_int    = Txo.isInteger() ? Tx.asSignedInt() : Tx;
    Txs_int    = Tx;

    int cpoDiv = 1;
    if (Txo_int.isInt8()) Txo_int = Type::s16, cpoDiv = 2;

    if (xs2D && (Txs.paddedSize() > Tx.paddedSize())) {
        lateScale = true;
        Txs_int = Tx_scaleOp = problem.Tc;
    }

    bool int4SpecialPath = Tx_ext.isInt4() && one_of(Tx, Type::f16, Type::bf16, Type::f32);
    if (int4SpecialPath)
        Txo_int = Txs_int = Type::f16;

    // Get tile sizes, depending on whether A/B are copied to SLM.
    // For late scaling (after compute), scales are always applied to the whole tile.
    int r, c, k, rNoSLM, cNoSLM;
    int tileR = 0, tileC = 0;
    bool remR = false, remC = false;
    if (isA) {
        bool slmA = strategy.slmA;
        rNoSLM = strategy.unroll[LoopM];
        cNoSLM = strategy.ka_load;
        r = slmA ? state.ma_slm : rNoSLM;
        c = slmA ? state.ka_slm : cNoSLM;
        k = slmA ? strategy.unrollKSLM : cNoSLM;
        c = state.kaq = std::max(1, c / problem.aqGroupK);
        state.kaqStride = std::max(1, k / problem.aqGroupK);
        cNoSLM = state.kaqLate = std::max(1, cNoSLM / problem.aqGroupK);
        remR = (strategy.remHandling[LoopM] != RemainderHandling::Ignore);
        tileC = 1;
    } else {
        bool slmB = strategy.slmB;
        cNoSLM = strategy.unroll[LoopN];
        rNoSLM = strategy.kb_load;
        c = slmB ? state.nb_slm : cNoSLM;
        r = slmB ? state.kb_slm : rNoSLM;
        k = slmB ? strategy.unrollKSLM : rNoSLM;
        r = state.kbq = std::max(1, r / problem.bqGroupK);
        state.kbqStride = std::max(1, k / problem.bqGroupK);
        rNoSLM = state.kbqLate = std::max(1, rNoSLM / problem.bqGroupK);
        remC = (strategy.remHandling[LoopN] != RemainderHandling::Ignore);
        tileR = 1;
    }

    int rs = lateScale ? rNoSLM : r;
    int cs = lateScale ? cNoSLM : c;

    if (X_strategy.padded) {
        X_offsetStrategy.padded = true;
        remR = remC = false;
    }

    X_offsetStrategy.newDP = (hw >= HW::XeHPG);
    X_scaleStrategy = X_offsetStrategy;

    X_offsetStrategy.accessType = (isA == isColMajor(XO.layout)) ? AccessType::Block : AccessType::Scattered;
    X_scaleStrategy.accessType  = (isA == isColMajor(XS.layout)) ? AccessType::Block : AccessType::Scattered;

    if (!X_offsetStrategy.base.isStateless()) {
        X_offsetStrategy.base.setIndex(isA ? state.inputs.surfaceAO : state.inputs.surfaceBO);
        X_scaleStrategy.base.setIndex(isA ? state.inputs.surfaceAScale : state.inputs.surfaceBScale);
    }

    if (xo2D && !getRegLayout(Txo, X_offsetLayout, r,  c,  remR, remC, false, AvoidFragment, tileR, tileC, XO, X_offsetStrategy)) return false;
    if (xs2D && !getRegLayout(Txs, X_scaleLayout,  rs, cs, remR, remC, false, AvoidFragment, tileR, tileC, XS, X_scaleStrategy)) return false;

    // Quantization parameters will be upconverted to the size of A/B and duplicated to match crosspack.
    auto &lsrc = isA ? (strategy.slmA ? state.Ao_layout : !state.Ar_layout.empty() ? state.Ar_layout : state.A_layout)
                     : (strategy.slmB ? state.Bo_layout : !state.Br_layout.empty() ? state.Br_layout : state.B_layout);
    if (lsrc.empty()) stub();
    int crosspack = lsrc[0].crosspack;
    if (int4SpecialPath && Tx == Type::bf16)
        crosspack = 1;
    int cpo = div_up(crosspack, cpoDiv);

    if (isA  && problem.aqGroupK == 1) tileC = crosspack;
    if (!isA && problem.bqGroupK == 1) tileR = crosspack;

    auto makeQRepack = [&](Type Txq, Type Txq_int, vector<RegisterBlock> &repack, vector<RegisterBlock> &src, int m, int n, int cp) {
        if (cp > 1 || (cColMajor && (cp != src[0].crosspack)) || Txq != Txq_int)
            makeUnbackedRegLayout(Txq_int, repack, m, n, isA, cp, tileR, tileC, false);
    };

    if (xo2D) makeQRepack(Txo, Txo_int,    Xr_offsetLayout, X_offsetLayout, r,  c,  cpo);
    if (xs2D) makeQRepack(Txs, Tx_scaleOp, Xr_scaleLayout,  X_scaleLayout,  rs, cs, lateScale ? 1 : crosspack);

    if (xoTo2D) {
        if (xoPtrDims == 0)
            makeUnbackedRegLayout(Txo_int, Xr_offsetLayout, 1, 1, isA);
        else if (xoPtrDims == 1)
            makeUnbackedRegLayout(Txo_int, Xr_offsetLayout, r, c, isA, cpo, tileR, tileC, false);
        else stub();
    }

    return true;
}

// Convert and repack 2D grouped quantization data in preparation for dequantization.
template <HW hw>
void BLASKernelGenerator<hw>::gemmRepack2DQuantizationData(Type Ts, Type Td, const vector<RegisterBlock> &layoutSrc, const vector<RegisterBlock> &layoutDst,
                                                           const GRFMultirange &src, const GRFMultirange &dst,
                                                           const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (layoutDst.empty()) return;

    int ms, ns, md, nd;
    getLayoutDims(layoutSrc, ms, ns);
    getLayoutDims(layoutDst, md, nd);

    // Copy, broadcasting 1D to 2D data as needed.
    for (int doffR = 0; doffR < md; doffR += ms)
        for (int doffC = 0; doffC < nd; doffC += ns)
            copyRegisters(Ts, Td, layoutSrc, layoutDst, src, dst, doffR, doffC, false, strategy, state);

    // Duplicate data in padded region. TODO: do this as part of the copy.
    int cp = layoutDst[0].crosspack;
    int p0 = layoutDst[0].colMajor ? layoutDst[0].nc : layoutDst[0].nr;

    if (cp > 1) map(hw, Td, dst, layoutDst, strategy, [&](int simd, RegData r) {
        Subregister r0 = GRF(r.getBase()).sub(r.getOffset(), r.getType());
        moveToIntPipe(r0);
        auto r1 = r0;
        for (int i = p0; i < cp; i++) {
            r1.setOffset(r1.getOffset() + 1);
            mov(simd / cp, r1(cp), r0(cp));
        }
    });
}

template <HW hw>
void BLASKernelGenerator<hw>::gemmRepack2DOffsetData(Type Text, Type Ts, Type Td, const vector<RegisterBlock> &layoutSrc, const vector<RegisterBlock> &layoutDst,
                                                     const GRFMultirange &src, const GRFMultirange &dst,
                                                     const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    bool s4 = (Text == Type::s4);
    bool s8 = (Ts == Type::s8);
    bool u8 = (Ts == Type::u8);

    bool int4SpecialPath = Text.isInt4() && Td == Type::f16;
    auto tmpType = Td;

    if (int4SpecialPath) {
        if (u8) tmpType = Type::u16;
        if (s8) tmpType = Type::s16;
    }

    gemmRepack2DQuantizationData(Ts, tmpType, layoutSrc, layoutDst, src, dst, problem, strategy, state);

    if (int4SpecialPath) {
        if (s8 || u8) {
            int off = s4 ? 8 : 0;

            // Shift s8 -> u8 data.
            if (s8) {
                map(hw, Type::s16, dst, dst, strategy, [&](int esize, RegData r, RegData _) {
                    add(esize, r, r, 0x80);
                });
                off -= 0x80;
            }

            // Reinterpret as f16 and undo offsets.
            if (off != 0) map(hw, Type::f16, dst, dst, strategy, [&](int esize, RegData r, RegData _) {
                uint16_t offF16 = std::abs(off);
                if (off < 0) offF16 |= 0x8000;
                add(esize, r, r, Immediate::hf(offF16));
            });

            // Rescale into normal range. End result is 2^(-9) * intended offset.
            map(hw, Type::f16, dst, dst, strategy, [&](int esize, RegData r, RegData _) {
                mul(esize, r, r, Immediate::hf(0x7800));
            });
        } else {
            map(hw, Type::f16, dst, dst, strategy, [&](int esize, RegData r, RegData _) {
                s4 ? mad(esize, r, Immediate::hf(0x2400), r, Immediate::hf(0x1800))     // 0x2400 = 8 * 2^(-9)
                   : mul(esize, r,                        r, Immediate::hf(0x1800));    // 0x1800 = 2^(-9)
            });
        }
    }
}

// Apply a single 2D group dequantize operation (scale/multiply).
template <HW hw>
void BLASKernelGenerator<hw>::gemmDequantizeOperation(bool doA, Type T, Type To, BinaryOp op,
                                                      const std::vector<RegisterBlock> &layout, const std::vector<RegisterBlock> &qlayout,
                                                      const GRFMultirange &regs, const GRFMultirange &qregs,
                                                      int hq, const GEMMProblem &problem)
{
    int xqGroupK = doA ? problem.aqGroupK : problem.bqGroupK;

    int mq, nq;
    getLayoutDims(qlayout, mq, nq);
    bool broadcast = (mq * nq) == 1;

    for (auto &block: layout) {
        auto crosspack = block.crosspack;
        bool colMajor = block.colMajor;
        int nx = colMajor ? block.nr : block.nc;
        int ny = colMajor ? block.nc : block.nr;

        for (int y0 = 0; y0 < ny; y0 += crosspack) {
        for (int x0 = 0; x0 < nx; ) {
            auto ii0 = colMajor ? x0 : y0;
            auto jj0 = colMajor ? y0 : x0;
            auto io0 = ii0 + block.offsetR;
            auto jo0 = jj0 + block.offsetC;
            auto &ho0 = doA ? jo0 : io0;
            ho0 += hq;
            ho0 /= xqGroupK;
            if (broadcast) io0 = jo0 = 0;

            int ne, neq;
            const RegisterBlock *qblock;
            auto data = findBlockReg(T, block, ii0, jj0, regs, ne);
            auto qdata = findBlockReg(To, qlayout, io0, jo0, qregs, neq, qblock);

            int strideq = 1;
            if (broadcast)
                strideq = 0;
            else if (colMajor == doA) {
                ne = std::min(ne, neq);
                if (qblock->crosspack * To != crosspack * T) stub();
            } else {
                ne = std::min(ne, xqGroupK);
                strideq = 0;
            }

            int maxSIMD = (op == BinaryOp::Sub && T.isInt8()) ? 64 : 32;
            int simd = std::min({ne * crosspack, 2 * elementsPerGRF(hw, T), maxSIMD});
            switch (op) {
                case BinaryOp::Sub:
                    if (T.isInt8()) {
                        add(simd / 2, data(2), data(2), -qdata(strideq * 2 / To));
                        data.setOffset(data.getOffset() + 1);
                        qdata.setOffset(qdata.getOffset() + strideq / To);
                        add(simd / 2, data(2), data(2), -qdata(strideq * 2 / To));
                    } else
                        add(simd, data(1), data(1), -qdata(strideq));
                    break;
                case BinaryOp::Mul: mul(simd, data(1), data(1),  qdata(strideq)); break;
                case BinaryOp::ScaleSub:
                    if (T != Type::f16) stub();
                    mad(simd, data(1), -qdata(strideq), data(1), Immediate::hf(0x7800));  /* 0x7800 = 2^15 */
                    break;
                default: stub();
            }
            x0 += simd / crosspack;
        }
        }
    }
}

bool canDequantizeInt4(Type Tsrc, Type Tdst,
                       const vector<RegisterBlock> &layoutSrc, const vector<RegisterBlock> &layoutDst,
                       const vector<RegisterBlock> layoutOffset, const vector<RegisterBlock> layoutScale)
{
    if (!Tsrc.isInt4() || !one_of(Tdst, Type::f16, Type::bf16, Type::f32))
        return false;

    if (layoutOffset.empty() || layoutScale.empty()) {
        int m, n, md, nd;
        getLayoutDims(layoutSrc, m, n);
        getLayoutDims(layoutDst, md, nd);

        if (m < md || n < nd) return false;
    }

    return true;
}

// Shift s4 data by 8 to transfrom it into u4 data.
template <HW hw>
void BLASKernelGenerator<hw>::dequantizeInt4Shift(Type Tsrc, GRFMultirange src, const CommonStrategy &strategy)
{
    if (Tsrc != Type::s4) return;
    map(hw, Type::u16, src, src, strategy, [&](int esize, RegData r, RegData _) {
        xor_(esize, r, r, 0x8888);
    });
}

// Optimized int4 -> f16/bf16/f32 dequantization sequence.
template <HW hw>
void BLASKernelGenerator<hw>::dequantizeInt4(bool doA, Type Tsrc, Type Tdst, const vector<RegisterBlock> &layoutSrc, const vector<RegisterBlock> &layoutDst,
                                             const vector<RegisterBlock> &layoutOffset, const vector<RegisterBlock> &layoutScale,
                                             GRFMultirange src, GRFMultirange dst, GRFMultirange offset, GRFMultirange scale,
                                             Type Tscale, int offR, int offC,
                                             const GEMMProblem *problem, const CommonStrategy &strategy, CommonState &state, bool s4Shift)
{
    if (!canDequantizeInt4(Tsrc, Tdst, layoutSrc, layoutDst, layoutOffset, layoutScale))
        stub("Cannot perform dequantizeInt4");

    int m, n, md, nd;
    getLayoutDims(layoutSrc, m, n);
    getLayoutDims(layoutDst, md, nd);

    bool s4 = Tsrc.isSigned();
    bool f32 = (Tdst == Type::f32);
    bool bf16 = (Tdst == Type::bf16);

    vector<RegisterBlock> layoutDstF16;
    const vector<RegisterBlock> *effLayoutDst = &layoutDst;
    GRFMultirange dstF16;
    const GRFMultirange *effDst = &dst;
    if (f32 || bf16) {
        makeUnbackedRegLayout(Type::f16, layoutDstF16, m, n, isLayoutColMajor(layoutDst), 1);
        for (auto &block: layoutDstF16) {
            block.offsetR += layoutDst[0].offsetR;
            block.offsetC += layoutDst[0].offsetC;
        }
        dstF16 = chunkAlloc(getRegCount(layoutDstF16), 2, state);
        effLayoutDst = &layoutDstF16;
        effDst = &dstF16;
    }

    // 1) Shift s4 data to u4 data by adding 8.
    if (s4 && s4Shift)
        dequantizeInt4Shift(Tsrc, src, strategy);

    // 2) Copy u4 -> u16 data.
    copyRegisters(Type::u4, Type::u16, layoutSrc, *effLayoutDst, src, *effDst, offR, offC, false, strategy, state);

    // 3) Reinterpret u16 data as denormal f16, scale into normal range and subtract (rescaled) offsets if available.
    //     The required rescaling factor (2^24) is necessarily outside f16 range,
    //     so two multiplications are needed.
    int hab = doA ? offC : offR;
    if (!layoutOffset.empty()) {
        if (!problem) stub();
        gemmDequantizeOperation(doA, Type::f16, Type::f16, BinaryOp::ScaleSub, *effLayoutDst, layoutOffset, *effDst, offset, hab, *problem);
    } else {
        map(hw, Type::f16, *effDst, *effLayoutDst, strategy, [&](int esize, RegData r) {
            s4 ? mad(esize, r, Immediate::hf(0xA400), r, Immediate::hf(0x7800))
               : mul(esize, r, r, Immediate::hf(0x7800));
        });
    }

    // 4) Finish rescaling -- remaining factor is 2^9.
    map(hw, Type::f16, *effDst, *effLayoutDst, strategy, [&](int esize, RegData r) {
        mul(esize, r, r, Immediate::hf(0x6000));    /* 0x6000 = 2^9 */
    });

    // 5) Apply scales if present. If the scales are not too large (absolute value < 128),
    //      this could be scaled into the previous multiplication.
    if (!layoutScale.empty()) {
        if (!problem) stub();
        gemmDequantizeOperation(doA, Type::f16, Tscale, BinaryOp::Mul, *effLayoutDst, layoutScale, *effDst, scale, hab, *problem);
    }

    // 6) Convert to dst type if needed.
    if (f32 || bf16) {
        copyRegisters(Type::f16, Tdst, layoutDstF16, layoutDst, dstF16, dst, offR, offC, false, strategy, state);
        safeReleaseRanges(dstF16, state);
    }
}

// Dequantize A/B, given 2D grouped quantization data.
template <HW hw>
void BLASKernelGenerator<hw>::gemmDequantizeAB(bool doA, Type Tsrc, Type Tdst,
                                               const vector<RegisterBlock> &layoutSrc, const vector<RegisterBlock> &layoutDst0,
                                               const GRFMultirange &src, const GRFMultirange &dst0, int hab,
                                               const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state,
                                               bool s4Shift)
{
    auto Txo_int     = doA ? state.Tao_int         : state.Tbo_int;
    auto Tx_scaleInt = doA ? state.Ta_scaleInt     : state.Tb_scaleInt;
    auto Tx_scaleOp  = doA ? state.Ta_scaleOp      : state.Tb_scaleOp;
    auto &oiLayout   = doA ? state.A_offsetLayout  : state.B_offsetLayout;
    auto &orLayout   = doA ? state.Ar_offsetLayout : state.Br_offsetLayout;
    auto &oiRegs     = doA ? state.A_offsetRegs    : state.B_offsetRegs;
    auto &orRegs     = doA ? state.Ar_offsetRegs   : state.Br_offsetRegs;
    auto &siLayout   = doA ? state.A_scaleLayout   : state.B_scaleLayout;
    auto &srLayout   = doA ? state.Ar_scaleLayout  : state.Br_scaleLayout;
    auto &siRegs     = doA ? state.A_scaleRegs     : state.B_scaleRegs;
    auto &srRegs     = doA ? state.Ar_scaleRegs    : state.Br_scaleRegs;
    bool lateScale   = doA ? state.lateScale2DA    : state.lateScale2DB;

    auto &oLayout = orLayout.empty() ? oiLayout : orLayout;
    auto &oRegs   = orLayout.empty() ? oiRegs   : orRegs;
    auto &sLayout = srLayout.empty() ? siLayout : srLayout;
    auto &sRegs   = srLayout.empty() ? siRegs   : srRegs;

    bool xo2D = !oLayout.empty();
    bool xs2D = !sLayout.empty() && !lateScale;

    bool copy = !layoutDst0.empty();
    auto layoutDst = copy ? layoutDst0 : layoutSrc;
    auto dst       = copy ? dst0 : src;

    auto Tx1_int = xo2D ? Txo_int : Tx_scaleInt;
    auto Tx2_int = xs2D ? Tx_scaleInt : Tdst;

    if (xo2D && !xs2D && (Txo_int.bits() > Tdst.bits()))
        Tx1_int = Tdst;

    int offR = doA ? 0 : hab;
    int offC = doA ? hab : 0;
    int offR0 = offR, offC0 = offC;

    int ms, md, ns, nd;
    getLayoutDims(layoutSrc, ms, ns);
    getLayoutDims(layoutDst, md, nd);

    if (ms < md || ns < nd) {
        if (!copy) stub();
        makeUnbackedRegLayout(Tdst, layoutDst, ms, ns, isLayoutColMajor(layoutDst0), layoutDst[0].crosspack);
        dst = chunkAlloc(getRegCount(layoutDst), 2, state);
        offR = offC = 0;
    }

    if (canDequantizeInt4(Tsrc, Tdst, layoutSrc, layoutDst, oLayout, sLayout)) {
        dequantizeInt4(doA, Tsrc, Tdst, layoutSrc, layoutDst, oLayout, sLayout,
                       src, dst, oRegs, sRegs, Tx_scaleOp, offR, offC, &problem,
                       strategy, state, s4Shift);
    } else {
        if (copy)
            copyRegisters(Tsrc, Tx1_int, layoutSrc, layoutDst, src, dst, offR, offC, false, strategy, state);
        else if (Tsrc.asSigned() != Tx1_int.asSigned())
            convert(src, Tsrc, Tx1_int, strategy, state);

        if (xo2D) {
            gemmDequantizeOperation(doA, Tx1_int, Txo_int, BinaryOp::Sub, layoutDst, oLayout, dst, oRegs, hab, problem);
            convert(dst, Tx1_int, Tx2_int, strategy, state);
        }

        if (xs2D) {
            gemmDequantizeOperation(doA, Tx_scaleInt, Tx_scaleOp, BinaryOp::Mul, layoutDst, sLayout, dst, sRegs, hab, problem);
            convert(dst, Tx_scaleInt, Tdst, strategy, state);
        }
    }

    if (ms < md || ns < nd) {
        copyRegisters(Tdst, Tdst, layoutDst, layoutDst0, dst, dst0, offR0, offC0, false, strategy, state);
        safeReleaseRanges(dst, state);
    }
}

#include "internal/namespace_end.hxx"
