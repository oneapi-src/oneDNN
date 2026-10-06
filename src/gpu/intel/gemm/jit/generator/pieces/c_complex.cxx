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
#include "layout_utils.hpp"
#include "map.hpp"
#include "ngen_object_helpers.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using std::vector;


// Swap the real/imaginary parts of a complex matrix.
template <HW hw>
void Generator<hw>::complexSwizzle(Type T, const GRFMultirange &C1, const CommonStrategy &strategy, CommonState &state)
{
    if (T.size() == 4) {
        // Easy: rotate.
        map(hw, T.real(), C1, C1, strategy, [&](int esize, GRF c1, GRF _) {
            ror(esize >> 1, c1.ud(), c1.ud(), uint16_t(16));
        });
    } else if ((T.size() == 8) && (hw >= HW::XeHPC)) {
        map(hw, T.real(), C1, C1, strategy, [&](int esize, GRF c1, GRF _) {
            ror(esize >> 1, c1.uq(), c1.uq(), uint16_t(32));
        });
    } else if ((T.size() == 16) && (hw >= HW::Xe3p)) {
        // Shuffle with new and improved movi.
        int nreg = C1.getLen();
        mov(16, a0, Immediate::uv(8, 12, 0, 4, 8, 12, 0, 4));
        add(4, a0[4](1),  a0[4](1),  16);
        add(4, a0[8](1),  a0[8](1),  32);
        add(4, a0[12](1), a0[12](1), 48);
        for (int i = 0; i < nreg; i++)
            movi<uint32_t>(16, C1[i], indirect[a0[0]], C1[i].getBase() * GRF::bytes(hw));
    } else {
        // Grab a flag register to use. May need to handle virtual flag registers.
        bool allocFlag = false;
        auto flagSwizzle = state.flagSwizzle;
        Subregister flagSave;
        if (flagSwizzle.isValid())
            state.usePhysicalFlag(flagSwizzle, hw);
        else try {
            flagSwizzle = state.raVFlag.alloc();
            allocFlag = true;
        } catch (out_of_registers_exception &) {
            flagSwizzle = f1[1];
            flagSave = state.ra.alloc_sub<uint16_t>();
            mov(1, flagSave, flagSwizzle);
        }
        mov(1, flagSwizzle, uint16_t(0xAAAA));      // that's what I thought, too.

        // Use (integer) sel to swizzle in-place.
        switch (T.size()) {
            case 4:
                map<uint32_t>(hw, C1, C1, strategy, [&](int esize, GRF c1, GRF _) {
                    c1 = c1.uw();
                    auto esize1 = std::min(esize, 16);
                    sel(esize1 | flagSwizzle, c1, c1[0](2,2,0), c1[1](2,2,0));
                    c1++; esize -= esize1;
                    sel(esize1 | flagSwizzle, c1, c1[0](2,2,0), c1[1](2,2,0));
                });
                break;
            case 8:
                map<uint32_t>(hw, C1, C1, strategy, [&](int esize, GRF c1, GRF _) {
                    sel(esize | flagSwizzle, c1, c1[0](2,2,0), c1[1](2,2,0));
                });
                break;
            case 16:
                map<uint64_t>(hw, C1, C1, strategy, [&](int esize, GRF c1, GRF _) {
                    c1 = c1.ud();
                    sel(esize | flagSwizzle, c1[0](2), c1[0](4,2,0), c1[2](4,2,0));
                });
                map<uint64_t>(hw, C1, C1, strategy, [&](int esize, GRF c1, GRF _) {
                    c1 = c1.ud();
                    sel(esize | flagSwizzle, c1[1](2), c1[1](4,2,0), c1[3](4,2,0));
                });
                break;
            default: stub();
        }

        if (allocFlag)
            state.raVFlag.safeRelease(flagSwizzle);
        else if (flagSave.isValid()) {
            mov(1, flagSwizzle, flagSave);
            state.ra.safeRelease(flagSave);
        }
    }
}

// Setup temporary registers for complex combines (alternating +1/-1)
template <HW hw>
void Generator<hw>::setupTeardownComplexCombine(bool setup, Type T, GEMMState &state)
{
    auto &altSigns = state.alternatingSigns;

    if (setup) {
        altSigns = state.ra.alloc_range(2);
        int ne = elementsPerGRF(hw, T.real());
        {
            mov(ne, altSigns[0].sub(0, T.real().ngen())(2), cast(T.real(), +1));
            mov(ne, altSigns[0].sub(1, T.real().ngen())(2), cast(T.real(), -1));
        }
    } else
        state.ra.safeRelease(altSigns);
}

// Accumulate the two copies of C for complex C, possibly applying conjugation.
//  conjC controls conjugation within each copy of C.
//  conjH controls whether the second copy of C is negated.
template <HW hw>
void Generator<hw>::complexCombine(Type T, const GRFMultirange &C0, const GRFMultirange &C1,
                                             const GEMMStrategy &strategy, GEMMState &state,
                                             bool conjC, bool conjH, bool preswizzled)
{
    // ATS+ can't swizzle in float/long pipes. Swizzle now if it hasn't been done.
    if (!preswizzled && (hw >= HW::XeHP)) {
        complexSwizzle(T, C1, strategy, state);
        preswizzled = true;
    }

    if (preswizzled && !(hw == HW::XeHPC && T == Type::cf64)) {
        // This code-path is disabled for cf64 on PVC since it causes very intermittent accuracy failures.

        // none: (r0-r1,  i0+i1)
        //    C: (r0+r1, -i0+i1)
        //    H: (r0+r1,  i0-i1)
        //   CH: (r0-r1, -i0-i1)
        auto &altSigns = state.alternatingSigns;
        bool alloc = altSigns.isInvalid();
        if (alloc) setupTeardownComplexCombine(true, T, state);

        map(hw, T.real(), C0, C1, strategy, [&](int esize, GRF c0, GRF c1) {
            auto mc1 = conjH ? -c1 : +c1;
            auto ct =  altSigns[0].retype(T.real().ngen());
            conjC ? mad(esize, c0, mc1, c0,  ct)
                  : mad(esize, c0, c0, mc1, -ct);
        });

        if (alloc) setupTeardownComplexCombine(false, T, state);
    } else {
        map(hw, T.real(), C0, C1, strategy, [&](int esize, GRF c0, GRF c1) {
            auto src0 = c0[0](2, 1, 0);
            auto src1 = (conjC ^ conjH) ? +c1[preswizzled ? 0 : 1](2, 1, 0)
                                        : -c1[preswizzled ? 0 : 1](2, 1, 0);
            add(esize >> 1, c0[0](2), src0, src1);
        });
        map(hw, T.real(), C0, C1, strategy, [&](int esize, GRF c0, GRF c1) {
            auto src0 = conjC ? -c0[1](2, 1, 0)
                              : +c0[1](2, 1, 0);
            auto src1 = conjH ? -c1[preswizzled ? 1 : 0](2, 1, 0)
                              : +c1[preswizzled ? 1 : 0](2, 1, 0);
            add(esize >> 1, c0[1](2), src0, src1);
        });
    }
}

// Accumulate the two copies of C for complex C, possibly applying conjugation.
// One is in phase (real, imag); the other is 90 degrees out of phase (imag, -real).
//  conjC controls conjugation within each copy of C.
//  conjH controls whether the second copy of C is negated.
template <HW hw>
void Generator<hw>::complexCombineC(bool conjC, bool conjH, bool preswizzled,
                                    const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (!problem.Tc.isComplex() || !state.haveCSwap) return;

    conjC ^= state.finalConjV;

    status << "Combine C components: " << "NC"[conjC] << "NC"[conjH] << status_stream::endl;

    complexCombine(problem.Tc, state.C_regs[0], state.C_regs[1], strategy, state, conjC, conjH, preswizzled);
}

template <HW hw>
void Generator<hw>::complexConjugateC(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    status << "Conjugate C" << status_stream::endl;

    for (int q = 0; q < state.C_buffers; q++) {
        map(hw, Type::u32, state.C_regs[q], state.C_regs[q], strategy, [&](int esize, GRF c, GRF) {
            switch (problem.Tc.real()) {
                case Type::f16: xor_(esize, c, c, 0x80000000); break;
                case Type::f32: xor_(esize >> 1, c[1](2), c[1](2), 0x80000000); break;
                case Type::f64: xor_(esize >> 2, c[3](4), c[3](4), 0x80000000); break;
                default: stub();
            }
        });
    }
}

template <HW hw>
void Generator<hw>::complex3mCombineC(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    status << "Complex3m Combine C" << status_stream::endl;

    GRFMultirange C_real, C_imag;
    // T1 contains Ar * Br accumulations (real)
    // T2 contains Ai * Bi accumulations (imaginary)
    GRFMultirange T1, T2;

    getSplitComplexComponents(hw, state.Tacc, state.C_regs[0], state.C_layout, C_real, C_imag);
    getSplitComplexComponents(hw, state.Tacc, state.C_regs[1], state.C_layout, T1, T2);

    map(hw, problem.Tc, C_real, T2, strategy, [&](int simd, const RegData &c1, const RegData &c2) {
        add(simd, c1, c1, -c2);
    });
    map(hw, problem.Tc, C_imag, T2, strategy, [&](int simd, const RegData &c1, const RegData &c2) {
        add(simd, c1, c1, -c2);
    });
    map(hw, problem.Tc, C_real, T1, strategy, [&](int simd, const RegData &c1, const RegData &c2) {
        add(simd, c1, c1, c2);
    });
    map(hw, problem.Tc, C_imag, T1, strategy, [&](int simd, const RegData &c1, const RegData &c2) {
        add(simd, c1, c1, -c2);
    });
}

template <HW hw>
void Generator<hw>::complexCombineCDispatch(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
#ifdef STANDALONE
    if (problem.readCheck) return;
#endif

    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc;

    // Combine the two copies of C for interleaved complex C.
    if (Ta.isComplex() && Tb.isComplex() && state.haveCSwap && !strategy.complex3m) {
        bool preswizzled = false;

        // Swap real/imaginary parts ahead of time for ATS+.
        if (hw >= HW::XeHP) {
            complexSwizzle(Tc, state.C_regs[1], strategy, state);
            preswizzled = true;
        }

        if (state.inputs.flags.isInvalid()) {
            // No conjugation.
            complexCombineC(false, false, preswizzled, problem, strategy, state);
        } else {
            // Adjustable conjugation. Generate code for all possible conjugation combinations.
            Label labelConjC, labelConjCH, labelConjH, labelDone;
            FlagRegister flagConjC = state.flagAP;
            FlagRegister flagConjH = state.raVFlag.alloc();

            // If C is column-major, then conjC -> conjA, conjH -> conjB. Vice versa for C row-major.
            // clang-format off
            bool globalCM = state.C_layout.colMajor();
            and_(1 | nz | flagConjC, null.ud(), state.inputs.flags, uint16_t(globalCM ? FlagConjA : FlagConjB));
            and_(1 | nz | flagConjH, null.ud(), state.inputs.flags, uint16_t(globalCM ? FlagConjB : FlagConjA));
            jmpi(1 | flagConjC, labelConjC);
                jmpi(1 | flagConjH, labelConjH);
                    complexCombineC(false, false, preswizzled, problem, strategy, state);
                    jmpi(1, labelDone);
                mark(labelConjH);
                    complexCombineC(false, true, preswizzled, problem, strategy, state);
                    jmpi(1, labelDone);
            mark(labelConjC);
                jmpi(1 | flagConjH, labelConjCH);
                    complexCombineC(true, false, preswizzled, problem, strategy, state);
                    jmpi(1, labelDone);
                mark(labelConjCH);
                    complexCombineC(true, true, preswizzled, problem, strategy, state);

            state.raVFlag.safeRelease(flagConjH);
            mark(labelDone);
            // clang-format on
        }
    } else if (state.inputs.flags.isValid()) {
        if (strategy.complex3m)
            complex3mCombineC(problem, strategy, state);

        // Apply conjugation if necessary (split-complex C or mixed real-complex kernels)
        Label skipConjC;
        and_(1 | ze | state.flagAP, null.ud(), state.inputs.flags, (Ta.isComplex() && !state.didConjA) ? FlagConjA : FlagConjB);
        jmpi(1 | state.flagAP, skipConjC);
        if (state.C_layout.isSplitComplex())
            splitComplexConjugateC(problem, strategy, state);
        else
            complexConjugateC(problem, strategy, state);
        mark(skipConjC);
    }
}

template <HW hw>
void Generator<hw>::splitComplexConjugateC(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    status << "Conjugate C" << status_stream::endl;

    GRFMultirange C_real, C_imag;
    getSplitComplexComponents(hw, state.Tacc, state.C_regs[0], state.C_layout, C_real, C_imag);

    map(hw, Type::u32, C_imag, C_imag, strategy, [&](int esize, GRF c, GRF) {
        switch (problem.Tc.real()) {
            case Type::f16: xor_(esize, c, c, 0x80008000); break;
            case Type::f32: xor_(esize, c, c, 0x80000000); break;
            case Type::f64: xor_(esize >> 1, c[1](2), c[1](2), 0x80000000); break;
            default: stub();
        }
    });
}

template <HW hw>
void Generator<hw>::splitComplexScaleC(const SubregisterPair &re, const SubregisterPair &im,
                                       const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Tc_real = state.Tacc.real().ngen();

    RegisterRange storage;
    RegData effAcc0, effAcc2;
    if (strategy.moveR0 != MoveR0::Acc && AccumulatorRegister::count(hw, strategy.GRFs, Tc_real) >= 4) {
        effAcc0 = acc0.retype(Tc_real);
        effAcc2 = acc2.retype(Tc_real);
    } else {
        storage = state.ra.alloc_range(4);
        effAcc0 = storage[0].retype(Tc_real);
        effAcc2 = storage[2].retype(Tc_real);
    }

    GRFMultirange C_real, C_imag;
    getSplitComplexComponents(hw, state.Tacc, state.C_regs[0], state.C_layout, C_real, C_imag);

    map(hw, problem.Tc.real(), C_real, C_imag, strategy, [&](int esize, GRF accRe, GRF accIm) {
        auto scaleRe = re.getRegAvoiding(hw, accRe);
        auto scaleIm = im.getRegAvoiding(hw, accIm);
        mul(esize, effAcc0,        accIm, -scaleIm);
        mul(esize, effAcc2,        accRe,  scaleIm);
        mad(esize, accRe, effAcc0, accRe,  scaleRe);
        mad(esize, accIm, effAcc2, accIm,  scaleRe);
    });

    state.ra.safeRelease(storage);
}

GEMMSTONE_NAMESPACE_END
