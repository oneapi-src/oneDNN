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


#include "generator.hpp"
#include "hw_utils.hpp"

using namespace ngen;
using namespace ngen::utils;

#include "internal/namespace_start.hxx"


// Convert linear index to 2D index using column/row-major ordering.
template <HW hw>
void BLASKernelGenerator<hw>::gemmSimpleLinearOrder(const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    state.inputs.groupIDM = state.ra.alloc_sub<uint32_t>();
    state.inputs.groupIDN = state.ra.alloc_sub<uint32_t>();

    bool nmk = (strategy.loopOrder[0] == LoopN);
    auto &groupCountX = nmk ? state.inputs.groupCountN : state.inputs.groupCountM;
    auto &groupIDX    = nmk ? state.inputs.groupIDN    : state.inputs.groupIDM;
    auto &groupIDY    = nmk ? state.inputs.groupIDM    : state.inputs.groupIDN;

    divDown(groupIDY, state.inputs.groupIDMN, groupCountX, state.inputs.gcMNRecip, state.flagAP, strategy, state);
    emad(1, groupIDX, state.inputs.groupIDMN, -groupIDY, groupCountX, strategy, state);

    if (!strategy.persistent) {
        state.ra.safeRelease(state.inputs.groupCountM);
        state.ra.safeRelease(state.inputs.groupCountN);
        state.ra.safeRelease(state.inputs.gcMNRecip);
    }
}

// Convert linear index to 2D index in a Hilbert curve-like fashion.
template <HW hw>
void BLASKernelGenerator<hw>::gemmHilbertlikeOrder(const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    bool triangular = false;
    bool rectangular = !triangular && state.inputs.hilbertVD.isValid();

    auto storage = state.ra.alloc();
    auto u = storage.ud(0);
    auto v = storage.ud(1);
    auto uh = storage.ud(2);
    auto vh = storage.ud(3);
    auto a = storage.ud(4);
    auto b = storage.ud(5);
    /* auto isNormal = storage.ud(6); */    // not used directly
    auto isReversed = storage.ud(7);
    int soff = storage.getBase() * GRF::bytes(hw);

    auto storage2 = state.ra.alloc_range(2);
    auto nbu = storage2[0].ud(0);
    auto nbv = storage2[0].ud(1);
    auto np1 = storage2[0].ud(2);
    auto bv1 = storage2[0].ud(3);
    auto uv1 = storage2[0].ud(4);
    auto temp3 = storage2[0].ud(5);
    auto uo = storage2[0].ud(6);
    /* auto vo = storage2[0].ud(7); */      // not used directly
    auto temp = storage2[1].ud(0);
    auto temp2 = storage2[1].ud(1);
    auto qrem = storage2[1].ud(2);
    auto qqot = storage2[1].ud(4);
    auto q = storage2[1].ud(6);
    auto ud = storage2[1].ud(7);

    auto bu = f1[0], bv = f1[1];

    auto vd = state.inputs.hilbertVD;
    auto uvdRecip = state.inputs.hilbertUVDRecip;
    auto hilbertBail = state.inputs.hilbertBail;

    auto any8      = (hw >= HW::XeHPC) ? any : any8h;
    auto any16     = (hw >= HW::XeHPC) ? any : any16h;
    bool avoidAny2 = (hw >= HW::XeHPC);

    auto jumpAny2 = [&](InstructionModifier mod, Label &l) {
        if (avoidAny2) {
            mod.setExecSize(16);
            goto12(mod | any16, l);
        } else
            jmpi(mod | any2h, l);
    };

    Label lTriangularTop, lTriangularExit, lTriangularBypass;
    Label lRecursiveTop, lRecursiveEnd;

    // NB: Sequence assumes group counts fit in 16 bits.
    status << "Hilbert-like ordering" << status_stream::endl;
    if (avoidAny2) mov(1, f0[0], 0);
    if (rectangular)
        mov(1, f0[1], vd.uw(1));                    // High word of vd = 0xFFFF -> start splitting in x
    else if (triangular)
        cmp(1 | ne | f0[0], state.inputs.diagC, 0);
    mov(1, u, state.inputs.groupCountM);
    mov(1, v, state.inputs.groupCountN);
    mov(4, a0, Immediate::uv(4, 0, 12, 8, 0, 0, 0, 0));
    mov(1, f1.ud(0), 0);                            // bu = bv = false
    mov(1, np1, triangular ? 0xFFFFFFFF : 0);
    if (triangular)
        cmp(1 | ~f0[0] | ne | f0[0], state.inputs.m, state.inputs.n);
    else
        cmp(2 | le | f0[0], u(1), hilbertBail);
    mov(1, q, state.inputs.groupIDMN);
    add(4, a0[4](1), a0[0](1), 16);
    if (!rectangular && !triangular)
        emad(1, uv1, -1, u.uw(), v.uw(), strategy, state);
    mov(8, a.uw()(1), Immediate::uv(0x00010000));   // a = b = 0, normal = 1, reversed = 0;
    if (soff >= 512) {
        add(4, a0, a0, soff);
        soff = 0;
    }
    auto swapXY = [&](InstructionModifier mod = 8) {
        if (hw >= HW::Gen12LP)
            subdep(Operand::src0, storage);
        movi(8 | mod, storage.ud(), indirect[a0].ud(soff)(1));
    };

    if (triangular)
        jmpi(1 | f0[0], lTriangularBypass);
    else
        jumpAny2(1 | f0[0], lRecursiveEnd);

    // Rectangular partitioning step. Break dispatch into blocks of roughly desired aspect ratio.
    if (rectangular) {
        auto uvd = uv1;
        swapXY(8 | f0[1]);
        mul(1, uvd, u, vd.uw());
        divDown(nbv, q, uvd, uvdRecip, f0[0], strategy, state);
        and_(1 | ne | bv, bv1, nbv, 1);
        mul(1, temp, uvd, nbv.uw());
        mul(1, b, vd.uw(), nbv.uw());
        add(1, q, q, -temp);                        // no DWxW with source modifiers
        add(1, v, v, -b);
        avg(1, ud, u, -bv1);
        min_(1, v, v, vd.uw());
        avg(1, uh, u, 0);
        mul(1, temp, v.uw(), ud.uw());
        cmp(1 | ge | bu, nbu, q, temp);
        add(1 | bu, q, q, -temp);
        cmp(1 | ne | bu, nbu, nbu.d(), -bv1.d());   // {bu,nbu} ^= bv1
        sel(1 | bu, a, uh, 0);
        avg(1, u, u, nbu.d());
        swapXY(8 | ~bu | any8);
        cmp(2 | le | f0[0], u(1), hilbertBail);
        sel(1 | ~bu, np1, -bv1, 0);
        emad(1, uv1, -1, u.uw(), v.uw(), strategy, state);
        mov(1, f1.ud(0), 0);                        // bu = bv = false
        jumpAny2(1 | f0[0], lRecursiveEnd);
    }

    // Recursive partitioning. Each step breaks the current block
    //  into 2x2 subblocks and follows the block we are currently in.
    // Exit when one dimension is less than hilbertBail.
    mark(lRecursiveTop); {
        avg(2, uh(1), u(1), 0);
        add(1 | bv, q, uv1, -q);

        mul(1, temp, u.uw(), vh.uw());
        cmp(1 | ge | bv, nbv, q, temp);
        mov(2, uo(1), u(1));
        add(1 | bv, q, uv1, -q);
        avg(1, v, v, nbv.d());
        mul(1, temp, uh.uw(), v.uw());
        cmp(1 | ge | bu, nbu, q, temp);
        add(1 | bu, q, q, -temp);
        avg(1, u, u, nbu.d());

        xor_(2, temp(1), nbu(1), np1);
        avg(2, uo(1), uo(1), np1.d());
        xor_(1 | bv, np1, np1, ~nbu);
        and_(2, uo(1), temp(1), uo(1));
        emad(1, uv1, -1, u.uw(), v.uw(), strategy, state);
        add(2, a(1), uo(1), a(1));

        cmp(2 | le | f0[0], u(1), hilbertBail);
        swapXY(8 | ~bu | any8);

        if (avoidAny2)
            goto12(16 | ~f0[0] | any16, lRecursiveEnd, lRecursiveTop, true);
        else
            jmpi(1 | ~f0[0] | any2h, lRecursiveTop);
    }
    mark(lRecursiveEnd);
    if (avoidAny2) join(16);

    cmp(8 | ne | f0[0], isReversed, 0);
    swapXY(8 | f0[0]);

    // Regular 2D traversal over final block.
    bool nmk = (strategy.loopOrder[0] == LoopN);
    auto divisor = nmk ? v : u;

    if (hw < HW::Gen12LP) {
        irem(1, qrem, q, divisor);
        iqot(1, qqot, q, divisor);
    } else {
        auto bias = temp.f();
        auto divisorFP = temp2.f();
        auto qFP = temp3.f();
        mov(1, divisorFP, divisor);
        mov(1, qFP, q);
        mov(1, bias, -0.499996185302734375f);       // -1/2 + 2^(-18)
        einv(1, divisorFP, divisorFP, strategy, state);
        add(1, divisorFP.ud(), divisorFP.ud(), 2);
        mad(1, qqot.f(), bias, qFP, divisorFP);
        mov(1, qqot, qqot.f());
        mad(1, qrem, q, -qqot.uw(), divisor.uw());
    }

    // Reassign m/n group IDs.
    if (!strategy.persistent) {
        state.inputs.groupIDM = state.inputs.groupCountM;
        state.inputs.groupIDN = state.inputs.groupCountN;
        state.inputs.groupCountM = invalid;
        state.inputs.groupCountN = invalid;
    } else {
        state.inputs.groupIDM = state.ra.alloc_sub<uint32_t>();
        state.inputs.groupIDN = state.ra.alloc_sub<uint32_t>();
    }

    add(1, state.inputs.groupIDM, a, nmk ? qqot : qrem);
    add(1, state.inputs.groupIDN, b, nmk ? qrem : qqot);

    state.ra.safeRelease(storage);
    state.ra.safeRelease(storage2);
    if (!strategy.persistent) {
        state.ra.safeRelease(state.inputs.hilbertVD);
        state.ra.safeRelease(state.inputs.hilbertUVDRecip);
        state.ra.safeRelease(state.inputs.hilbertBail);
    }
}

// Convert linear index to 2D index in a boustrophedon pattern.
template <HW hw>
void BLASKernelGenerator<hw>::gemmBoustrophedonOrder(const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    auto storage = state.ra.alloc_range(4);
    auto u = storage[0].ud(0);
    auto s = storage[0].ud(1);
    auto v = storage[0].ud(2);
    auto s1 = storage[0].ud(3);
    auto i = storage[0].ud(4);
    auto j = storage[0].ud(5);
    auto i0 = storage[0].ud(6);
    auto two = storage[0].f(7);
    auto numFP = storage[1].f(0);
    auto islice = storage[1].ud(1);
    auto qot = storage[1].ud(2);
    auto rem = storage[1].ud(4);
    auto ithresh = storage[1].ud(6);
    auto temp0 = storage[2].ud(0);
    auto temp1 = storage[2].ud(2);
    auto temp2 = storage[2].ud(4);
    auto bias = storage[3].f(0);
    auto denomFP = storage[3].f(2);
    auto q = storage[3].ud(4);
    auto qFP = storage[3].f(6);

    auto s0 = state.inputs.bslice;          // Slice width/height in WGs. Sign interpretation:
                                            //   + means slice in m dimension, - means n dimension
    auto thresh = state.inputs.bthresh;     // Slice size adjustment threshold
                                            //   + means increase slice size by 1 starting with this row/column
                                            //   - means decrease slice size by 1 starting with this row/column

    auto &groupCountM = state.inputs.groupCountM;
    auto &groupCountN = state.inputs.groupCountN;
    auto idMN = state.inputs.groupIDMN;

    Label lBegin, lEnd, lDone, lBeginTri2, lEndTri2, lTricalc1, lTricalc2, lTricalcOut;

    auto divqot = [&](const Subregister &num, const Subregister &denom, bool large) {
        if (hw < HW::Gen12LP) {
            irem(1, rem, num, denom);
            iqot(1, qot, num, denom);
        } else if (large) {
            // denom <= 0x400000, qot < 2^16
            or_(1, cr0[0], cr0[0], 0x20);               // round toward -inf
            mov(1, denomFP, denom);
            mov(1, numFP, -num);
            einv(1, denomFP, denomFP, strategy, state);
            add(1, denomFP.ud(), denomFP.ud(), 2);
            mul(1, qot.f(), -numFP, denomFP);
            mov(1, qot, qot.f());
            mad(1 | lt | f1[1], rem.d(), num, denom, -qot.uw());
            add(1 | f1[1], rem, rem, denom);
            add(1 | f1[1], qot, qot, -1);
            and_(1, cr0[0], cr0[0], ~0x30);
        } else {
            // denom <= 0x40, qot < 2^16
            mov(1, denomFP, denom);
            mov(1, numFP, num);
            mov(1, bias, -0.499996185302734375f);       // -1/2 + 2^(-18)
            einv(1, denomFP, denomFP, strategy, state);
            add(1, denomFP.ud(), denomFP.ud(), 2);
            mad(1, qot.f(), bias, numFP, denomFP);
            mov(1, qot, qot.f());
            mad(1, rem, num, -qot.uw(), denom.uw());
        }
    };

    // NB: Sequence assumes group counts fit in 16 bits.
    status << "Boustrophedon ordering" << status_stream::endl;

    mul(1, ithresh, abs(thresh.w()), abs(s0.w()));
    cmp(1 | ge | f1[0], thresh, 0);
    ecsel(1, lt, f0[0], v, groupCountM, groupCountN, s0);
    ecsel(1, ge, f0[0], u, groupCountM, groupCountN, s0);

    emad(1, temp0, idMN, -v.uw(), ithresh.uw(), strategy, state);
    cmp(1 | ge | f0[0], temp2.d(), temp0.d(), 0);
    ecsel(1, ge, f0[1], q, temp0, idMN, temp0.d());

    if (hw >= HW::XeHPC) {
        add(1,          s1, abs(s0), 1);
        add(1 | ~f0[0], s1, abs(s0), temp2.d());
        add(1 | ~f1[0], s1, abs(s0), temp2.d());
    } else {
        add(1,                s1, abs(s0), temp2.d());
        add(1 | f0[0] | allv, s1, abs(s0), 1);
    }

    mul(1, temp1, s1.uw(), v.uw());

    divqot(q, temp1, true);

    mul(1, i0, qot.uw(), s1.uw());
    mov(1, islice, qot);
    add(1 | f0[0], i0, i0, ithresh);
    mov(1, q, rem);
    add(1 | sat, temp0, u, -i0);
    min_(1, s, s1, temp0);
    add(1 | f0[0], islice, islice, abs(thresh));

    mul(1, temp2, s.uw(), s.uw());
    emad(1, temp1, temp1, -s.uw(), s.uw(), strategy, state);

    cmp(1 | gt | f0[0], i0, 0);         // not first row?
    cmp(1 | lt | f0[1], s1, temp0);     // not last row?

    if (hw >= HW::XeHPC) {
        cmp(1 | f0[0] | lt | f0[0], q, temp2);      // beginning of row?
        cmp(1 | f0[1] | ge | f0[1], q, temp1);      // end of row?
    } else {
        cmp(1 | lt | f1[0], q, temp2);      // beginning of row?
        cmp(1 | ge | f1[1], q, temp1);      // end of row?
    }

    mov(1, two, 2.0f);
    mov(1, bias, 1.25f);

    if (hw >= HW::XeHPC) {
        jmpi(1 | f0[0], lBegin);
        jmpi(1 | f0[1], lEnd);
    } else {
        jmpi(1 | f0[0] | allv, lBegin);
        jmpi(1 | f0[1] | allv, lEnd);
    }

    {
        divqot(q, s, false);

        add(1, i, i0, rem);
        mov(1, j, qot);
    }

    jmpi(1, lDone);

    mark(lBegin);
    {
        avg(1, temp0, temp2, -s);       // s(s-1)/2
        mov(1, f1.ud(0), 0xFFFF);
        cmp(1 | lt | f0[0], q, temp0);
        jmpi(1 | ~f0[0], lBeginTri2);

        eadd3(1, q, temp0, -q, -1);
        jmpi(1, lTricalc1);

        mark(lBeginTri2);
        add(1, q, q, -temp0);
        jmpi(1, lTricalc2);
    }

    mark(lEnd);
    {
        add(1, q, q, -temp1);
        avg(1, temp0, temp2, s);        // s(s+1)/2
        mov(1, f1.ud(0), 0);
        cmp(1 | lt | f0[0], q, temp0);
        jmpi(1 | ~f0[0], lEndTri2);

        eadd3(1, q, temp0, -q, -1);
        mark(lTricalc2);
        {
            mov(1, qFP, q);
            mad(1, qFP, bias, qFP, two);
            esqt(1, qFP, qFP, strategy, state);
            if (hw == HW::Gen9)
                rnde(1, qFP, qFP);
            mov(1, j, qFP);
            mul(1, temp0, j.uw(), j.uw());
            avg(1, temp0, temp0, -j);
            add(1, j, j, -1);
            add(1, i, q, -temp0);
        }
        jmpi(1, lTricalcOut);

        mark(lEndTri2);
        add(1, q, q, -temp0);
        mark(lTricalc1);
        {
            mov(1, qFP, q);
            mad(1, qFP, bias, qFP, two);
            esqt(1, qFP, qFP, strategy, state);
            if (hw == HW::Gen9)
                rnde(1, qFP, qFP);
            mov(1, i, qFP);
            mul(1, temp0, i.uw(), i.uw());
            avg(1, temp0, temp0, -i);
            add(1, j, q, -temp0);
        }

        mark(lTricalcOut);
        eadd3(1 |  f1[0], i, s, -i, -1);
        eadd3(1 | ~f1[0], j, v, -j, -1);
        add(1, i, i, i0);
    }

    // Reassign m/n group IDs.
    mark(lDone);

    if (!strategy.persistent) {
        state.inputs.groupIDM = state.inputs.groupCountM;
        state.inputs.groupIDN = state.inputs.groupCountN;
        state.inputs.groupCountM = invalid;
        state.inputs.groupCountN = invalid;
    } else {
        state.inputs.groupIDM = state.ra.alloc_sub<uint32_t>();
        state.inputs.groupIDN = state.ra.alloc_sub<uint32_t>();
    }

    and_(1 | ne | f1[1], null.ud(), islice, 1);
    eadd3(1 | f1[1], j, v, -j, -1);
    ecsel(1, ge, f0[0], state.inputs.groupIDM, i, j, s0);
    ecsel(1, lt, f0[0], state.inputs.groupIDN, i, j, s0);

    state.ra.safeRelease(storage);
    if (!strategy.persistent) {
        state.ra.safeRelease(state.inputs.bslice);
        state.ra.safeRelease(state.inputs.bthresh);
    }
}

// Reverse m/n loops if requested.
template <HW hw>
void BLASKernelGenerator<hw>::gemmReverseLoops(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    for (LoopType l : {LoopM, LoopN}) if (strategy.reverse[l]) {
        bool fusedL = strategy.fused && (l == strategy.fusedLoop);
        auto q =  (l == LoopM) ? state.inputs.m : state.inputs.n;
        auto q0 = (l == LoopM) ? state.i0 : state.j0;
        auto q0Align = state.ra.alloc_sub<uint32_t>();
        auto temp = state.ra.alloc_sub<uint32_t>();

        add(1, q0Align, q, -1);
        if (strategy.fixedWG(problem)) {
            mod(temp, q0, strategy.wg[l] * strategy.unroll[l], strategy, state);
            alignDown(q0Align, q0Align, strategy.wg[l] * strategy.unroll[l], strategy, state);
            shl(1, temp, temp, 1);
            eadd3(1 | ge | f0[0], q0Align.d(), q0Align, -q0, temp);
            mov(1 | f0[0], q0, q0Align);
        } else if (fusedL) {
            shl(1, temp, state.fusedID, 1);
            alignDown(q0Align, q0Align, 2 * strategy.unroll[l], strategy, state);
            eadd3(1 | ge | f0[0], q0Align.d(), q0Align, -q0, temp);
            mov(1 | f0[0], q0, q0Align);
        } else {
            alignDown(q0Align, q0Align, strategy.unroll[l], strategy, state);
            cmp(1 | le | f0[0], q0, q0Align);
            add(1 | f0[0], q0, q0Align, -q0);
        }
        state.ra.safeRelease(temp);
        state.ra.safeRelease(q0Align);
    }
}

// Reorder local IDs as needed.
template <HW hw>
void BLASKernelGenerator<hw>::gemmReorderLocalIDs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (strategy.fixedSystolic)
        sysgemmReorderLocalIDs(problem, strategy, state);

    if (strategy.skewLocalIDs) {
        if (!strategy.fixedWG(problem)) stub();
        auto wgI = strategy.wg[strategy.loopOrder[0]];
        auto adjustEvery = div_up(eusPerSubslice(hw), wgI);
        bool innerM = strategy.loopOrder[0] == LoopM;
        auto lidI = innerM ? state.lidM : state.lidN;
        auto lidO = innerM ? state.lidN : state.lidM;
        auto temp = state.ra.alloc_sub<uint16_t>();
        auto slidO = lidO;

        if (adjustEvery > 1) {
            shr(1, temp, lidO, ilog2(adjustEvery));
            slidO = temp;
        }

        if (strategy.fused)
            emad(1, lidI, lidI, slidO, 2, strategy, state);
        else
            add(1, lidI, lidI, slidO);

        if (!is_zero_or_pow2(wgI)) stub();

        and_(1, lidI, lidI, wgI - 1);

        state.ra.safeRelease(temp);
    }
}

#include "internal/namespace_end.hxx"
