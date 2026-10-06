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
#include "hw_utils.hpp"
#include "invert.hpp"
#include "ngen_object_helpers.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using std::vector;


static inline void invertSubAllocMathTemp(HW hw, Type Ts, CommonState &state);

// Generate subroutine for (complex) inversion.
template <HW hw>
void Generator<hw>::invertSub(Type Ts, bool inlineInvert, CommonState &state, bool negate)
{
    const bool haveIEEE = hasIEEEMacro(hw);
    const bool dfInv = (Ts.real() == Type::f64) && !haveIEEE;

    if (!state.invertSub.use) return;
    if (inlineInvert)
        invertSubAllocMathTemp(hw, Ts, state);
    else
        mark(state.invertSub.label);

    auto dt = Ts.ngen();
    auto simd = state.invertSub.simd;
    auto mathSIMD = elementsPerGRF(hw, Ts.real());
    auto cfSIMD = std::min(simd, mathSIMD);
    auto srcR1 = state.invertSub.srcR1.retype(dt);
    auto srcI1 = state.invertSub.srcI1.retype(dt);
    auto src1 = state.invertSub.src1.retype(dt);
    auto r = state.invertSub.r.retype(dt);
    auto d = state.invertSub.d.retype(dt);
    auto temp = state.invertSub.temp.retype(dt);
    auto &zero = state.invertSub.zero;
    auto &one = state.invertSub.one;
    auto &mathTemp = state.invertSub.mathTemp;
    auto tempFlags = state.invertSub.tempFlags;
    auto flagStore = state.invertSub.flagStore;

    // Save off flags that will be clobbered, if needed.
    bool haveFlags = tempFlags[0].isValid();
    if (Ts.isComplex() && !tempFlags[1].isValid()) {
        // Only one flag register: squeeze everything into it if possible.
        if (simd <= 8 && mathSIMD <= 8)
            tempFlags[1] = tempFlags[0];
        else
            haveFlags = false;
    }

    if (!haveFlags) {
        mov(1, flagStore, f0.ud());
        tempFlags[0] = f0[0];
        tempFlags[1] = f0[1];
    }

    // Invert diagonal.
    setDefaultNoMask(false);
    if (Ts.isComplex()) {
        // Complex inversion. If |x| > |y|,
        //      r = y / x
        //      d = x + r * y
        // result = (1/d, -r/d). Negate.

        // Decide which component is larger.
        auto flag = tempFlags[1];
        auto Mx = ExecutionOffset(tempFlags[0] == tempFlags[1] ? 8 : 0);
        cmp(simd | gt | Mx | flag | Switch, abs(srcR1), abs(srcI1));     // Switch on first sub statement.
        sel(simd | Mx | flag, d, srcR1, srcI1);                          // Larger components.
        sel(simd | Mx | ~flag, srcI1, srcR1, srcI1);                     // Smaller components.

        if (haveIEEE)
            fdiv_ieee(mathSIMD, tempFlags[0], r, srcI1, d, zero, one, mathTemp, cfSIMD);
        else {
            dfInv ? invDF(simd, r, d, tempFlags[0], state)
                  : inv(simd, r, d);
            mul(simd, r, r, srcI1);
        }
        mad(simd, srcR1, d, r, srcI1);
        if (haveIEEE)
            inv_ieee(mathSIMD, tempFlags[0], d, srcR1, one, mathTemp, cfSIMD);
        else {
            dfInv ? invDF(simd, d, srcR1, tempFlags[0], state)
                  : inv(simd, d, srcR1);
        }

        mov(simd | Mx | flag,  srcR1, negate ? -d : d);                  // Could remove an instruction here
        mov(simd | Mx | ~flag, srcI1, negate ? d : -d);                  //  but with a longer dependency chain
        mul(simd | Mx | ~flag, srcR1, negate ? -r : r, d);
        mul(simd | Mx | flag,  srcI1, negate ? r : -r, d);
    } else {
        // Real reciprocal.
        auto srcOff = src1[0](1);
        for (int eo = 0; eo < simd; eo += mathSIMD) {
            InstructionModifier mod;
            if (eo == 0)
                mod = Switch;
            if (dfInv) {
                auto srcOffGRF = GRF(srcOff.getBase()).retype(dt);
                invDF(mathSIMD | ExecutionOffset(eo), srcOffGRF, srcOffGRF, tempFlags[0], state, negate);
            } else {
                mov(mathSIMD | ExecutionOffset(eo) | mod, temp, negate ? -srcOff : srcOff);
                inv_ieee(mathSIMD | ExecutionOffset(eo), tempFlags[0], srcOff, temp, one, mathTemp, cfSIMD | ExecutionOffset(eo));
            }
            srcOff.setBase(srcOff.getBase() + 1);
        }
    }

    setDefaultNoMask(true);

    if (!haveFlags)
        mov(1, f0.ud(), flagStore);

    if (inlineInvert)
        safeReleaseRanges(state.invertSub.mathTemp, state);
    else
        ret(simd, state.invertSub.callStorage);
}

// Prepare registers for diagonal inversion subroutine.
template <HW hw>
void Generator<hw>::invertSubInit(Type Ts, bool inlineInvert, const CommonStrategy &strategy, CommonState &state, bool saveFlags, int simd)
{
    auto Ts_real = Ts.real();
    bool isComplex = Ts.isComplex();

    bool ieeeSeq = hasIEEEMacro(hw) && (Ts_real != Type::f16);
    bool invDFSeq = !hasIEEEMacro(hw) && (Ts_real == Type::f64);
    state.invertSub.use = (isComplex || ieeeSeq || invDFSeq);

    if (state.invertSub.use) {
        state.invertSub.simd = simd ? simd : maxInvertSubSIMD(hw, Ts);

        state.invertSub.src1Storage = state.ra.alloc_range(2);
        state.invertSub.src1 = state.invertSub.src1Storage[0];
        state.invertSub.src1.setType(Ts_real.ngen());

        if (!inlineInvert)
            invertSubAllocMathTemp(hw, Ts, state);

        if (isComplex) {
            state.invertSub.srcR1 = state.invertSub.src1;
            state.invertSub.srcI1 = state.invertSub.src1Storage[1];
            state.invertSub.r = state.ra.alloc();
            state.invertSub.d = state.ra.alloc();

            state.invertSub.srcI1.setType(Ts_real.ngen());
            state.invertSub.r.setType(Ts_real.ngen());
            state.invertSub.d.setType(Ts_real.ngen());
        } else
            state.invertSub.temp = state.ra.alloc();

        if (!inlineInvert) {
            if (getEfficient64Bit())
                state.invertSub.callStorageSub = state.ra.alloc().ud(0);
            else
                state.invertSub.callStorageSub = state.ra.alloc_sub<uint64_t>();
            state.invertSub.callStorage = state.invertSub.callStorageSub.ud();
        }

        if (isComplex) {
            state.invertSub.zero = state.ra.alloc();
            state.invertSub.zero.setType(Ts.real().ngen());
            mov(elementsPerGRF<uint32_t>(hw), state.invertSub.zero.ud(), uint16_t(0));
        }

        state.invertSub.one = state.ra.alloc();
        state.invertSub.one.setType(Ts.real().ngen());
        mov(elementsPerGRF(hw, Ts.real()), state.invertSub.one, cast(Ts.real(), 1));

        if (saveFlags)
            state.invertSub.flagStore = state.ra.alloc_sub<uint32_t>();
    }
}

// Release registers used by inversion subroutine.
template <HW hw>
void Generator<hw>::invertSubRelease(CommonState &state)
{
    if (!state.invertSub.use) return;

    safeReleaseRanges(state.invertSub.mathTemp, state);
    state.ra.safeRelease(state.invertSub.temp);
    state.ra.safeRelease(state.invertSub.src1Storage);
    state.ra.safeRelease(state.invertSub.r);
    state.ra.safeRelease(state.invertSub.d);
    if (getEfficient64Bit()) {
        state.ra.release(GRF(state.invertSub.callStorageSub.getBase()));
        state.invertSub.callStorageSub = invalid;
    } else
        state.ra.safeRelease(state.invertSub.callStorageSub);
    state.ra.safeRelease(state.invertSub.zero);
    state.ra.safeRelease(state.invertSub.one);
    state.ra.safeRelease(state.invertSub.flagStore);
}

inline int maxInvertSubSIMD(HW hw, Type Ts)
{
    return std::min(16, 2 * elementsPerGRF(hw, Ts));
}

static inline void invertSubAllocMathTemp(HW hw, Type Ts, CommonState &state)
{
    auto Ts_real = Ts.real();
    bool ieeeSeq = hasIEEEMacro(hw) && (Ts_real != Type::f16);

    int nMathTemp = 0;

    if (ieeeSeq) {
        nMathTemp = !Ts.isComplex() ? 3 :
             (Ts_real == Type::f64) ? 5 : 4;
    } else if (Ts_real == Type::f64)
        nMathTemp = 4;

    for (int t = 0; t < nMathTemp; t++)
        state.invertSub.mathTemp.append(state.ra.alloc());
}

GEMMSTONE_NAMESPACE_END
