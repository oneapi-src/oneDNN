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

using namespace ngen;
using namespace ngen::utils;

#include "internal/namespace_start.hxx"


template <HW hw>
void BLASKernelGenerator<hw>::gemmMicrokernel(GEMMProblem problem, GEMMStrategy strategy, const ngen::InterfaceHandler &interface_)
{
    GEMMState state(hw);

    interface = interface_;

    problem.autoTypeConversions(hw, strategy.systolic);
    gemmInitState(problem, strategy, state);
    for (int q = 0; q < 2; q++)
        state.ra.safeRelease(state.emulate.temp[q]);

    outputCRange = GRFMultirange();
    outputCLayout.clear();

    strategy.forceWGUpdate = WGFixed;

    state.isNested = true;
    state.ra.claim(r0-r6);      /* Leave some space for host kernel arguments */

    state.fullK = state.inputs.k;

    bool registerC = strategy.registerOutput();

    // Locate and claim additional inputs.
    auto getAndClaim = [&](const char *name) {
        auto sub = interface.getArgument(name);
        state.ra.claim(sub);
        return sub;
    };

    state.i0 = getAndClaim("i0");
    state.j0 = getAndClaim("j0");
    state.h0 = getAndClaim("h0");

    state.lidM = getAndClaim("local_id_m").uw();
    state.lidN = getAndClaim("local_id_n").uw();

    state.allocEmulate64Temp(strategy.emulate);

    setDefaultNoMask();
    setDefaultAutoSWSB();

    // Save and modify dispatch mask as needed.
    Subregister dmaskSave;
    int minSIMD = GRF::bytes(hw) >> 2;
    if (minSIMD < state.internalSIMD()) {
        dmaskSave = state.ra.alloc_sub<uint32_t>();
        mov(1, dmaskSave, sr0[2]);
        mov(1 | SWSB<AllPipes>(1), sr0[2], uint32_t(uint64_t(1) << state.internalSIMD()) - 1);
    }

    // Synchronize and save flag registers from host kernel.
    syncall();
    Subregister flagSave[4];
    for (int i = 0; i < FlagRegister::count(hw); i++) {
        flagSave[i] = state.ra.alloc_sub<uint32_t>();
        mov(1, flagSave[i], FlagRegister(i));
    }

    // Beginning of microkernel:
    //   - check32
    //   - fused ID calculation
    //   - ld scaling
    //   - i0/j0/h0 calculations (inside WG)
    //   - A/B/C offsets
    bool wgCheck = wgRemCheck(problem, strategy);
    bool gemmtBarriers = problem.gemmt() && strategy.needsBarrier();

    auto &k = state.inputs.k;
    auto &k0 = state.inputs.k0;

    state.lid0 = (strategy.fusedLoop == LoopN) ? state.lidN : state.lidM;
    getFusedID(strategy.unroll[strategy.fusedLoop], problem, strategy, state);

    emulConstant(1, state.inputs.lda, state.inputs.lda, problem.Ta_ext.size(), strategy, state);
    emulConstant(1, state.inputs.ldb, state.inputs.ldb, problem.Tb_ext.size(), strategy, state);
    if (!registerC)
        emulConstant(1, state.inputs.ldc[0], state.inputs.ldc[0], problem.Tc_ext.size(), strategy, state);

    if (wgCheck || gemmtBarriers) {
        state.wgI0 = copySubregister(state.i0, state);
        state.wgJ0 = copySubregister(state.j0, state);
    }

    if (strategy.kParallelLocal) {
        /* Select k0 automatically -- also need to compute lidK */
        int wgK = strategy.wg[LoopK];
        if (!is_zero_or_pow2(wgK)) stub();
        k0 = state.ra.alloc_sub<uint32_t>();
        add(1, k0, k, wgK - 1);
        shr(1, k0, k0, ilog2(wgK));
        alignUp(k0, k0, strategy.kAlign(problem), strategy, state);
    }

    emad(1, state.i0, state.i0, state.lidM, strategy.unroll[LoopM], strategy, state);
    emad(1, state.j0, state.j0, state.lidN, strategy.unroll[LoopN], strategy, state);
    if (strategy.kParallelLocal) {
        emad(1, state.h0, state.h0, k0, state.lidK, strategy, state);
        add(1 | sat, k.ud(), k, -state.h0);
        min_(1, k, k, k0);
        if (strategy.barrierFreq > 0 || strategy.slmBuffers > 0)
            state.ra.safeRelease(k0);
        else
            state.threadK0 = k0;
    }

    gemmCalcWGRemainders(problem, strategy, state);
    gemmCheck32(problem, strategy, state);

    auto &i0p = (strategy.coopA == CoopSplit::FullK) ? state.wgI0 : state.i0;
    auto &j0p = (strategy.coopB == CoopSplit::FullK) ? state.wgJ0 : state.j0;

    gemmOffsetABC(false, state.i0, state.j0, state.h0, i0p, j0p, problem, strategy, state, true, true, !registerC);

    if (!(strategy.prefetchA && strategy.A_prefetch.address2D)) state.ra.safeRelease(state.wgI0);
    if (!(strategy.prefetchB && strategy.B_prefetch.address2D)) state.ra.safeRelease(state.wgJ0);

    if (strategy.prefetchA && state.effAp.isInvalid()) state.effAp = state.effA;
    if (strategy.prefetchB && state.effBp.isInvalid()) state.effBp = state.effB;

    gemmSubkernel(problem, strategy, state);

    // Restore flag registers and dispatch mask and return to host kernel.
    for (int i = 0; i < FlagRegister::count(hw); i++)
        mov(1, FlagRegister(i), flagSave[i]);
    if (dmaskSave.isValid())
        mov(1, sr0[2], dmaskSave);
    syncall();
}

static inline micro::StructuredType::Type microType(Type T);

template <HW hw>
micro::Package BLASKernelGenerator<hw>::gemmMicrokernelPackage(const GEMMProblem &problem_, const GEMMStrategy &strategy, const ngen::InterfaceHandler &interface_, micro::GEMMProtocol protocol, uint32_t gmdid, bool transposeC)
{
    using namespace micro;
    Package package;

    auto problem = problem_;
    problem.autoTypeConversions(hw, strategy.systolic);

    gemmMicrokernel(problem, strategy, interface_);

    package.protocol = protocol;
    package.gmdidCompat = gmdid;
    package.binary = this->getCode();

    for (auto parg: package.protocol.arguments()) {
        Argument arg;
        arg.name = parg.name;

        if (arg.name == "c") {
            int tileM = strategy.unroll[LoopM];
            int tileN = strategy.unroll[LoopN];
            int blockM = outputCLayout[0].nr;
            int blockN = outputCLayout[0].nc;

            for (auto &block: outputCLayout) {
                if (blockM != block.nr) stub();
                if (blockN != block.nc) stub();
            }
            if (!isLayoutColMajor(outputCLayout)) stub(); /* Swap dims and block ordering */

            int blockGRF = 8;
            for (auto &r: outputCRange.ranges)
                blockGRF = gcd(blockGRF, r.getLen());
            int maxBlock = blockGRF * elementsPerGRF(hw, problem.Tc);

            if (blockM > maxBlock) {
                if (blockM % maxBlock) stub();
                blockM = maxBlock;
                blockN = 1;
            } else if (blockM * blockN > maxBlock) {
                int split = (blockM * blockN / maxBlock);
                if (blockM * blockN % maxBlock || blockN % split) stub();
                blockN /= split;
            }

            arg.location.resize((tileM * tileN) / (blockM * blockN));
            int idx = 0;
            for (int bj = 0; bj < tileN; bj += blockN) {
                for (int bi = 0; bi < tileM; bi += blockM) {
                    const RegisterBlock *block;
                    int ne;
                    auto topLeft = findBlockReg(problem.Tc, outputCLayout, bi, bj, outputCRange, ne, block);
                    arg.location[idx].boffset = topLeft.getBase() * GRF::bytes(hw) + topLeft.getByteOffset();
                    arg.location[idx].blen    = blockM * blockN * problem.Tc;
                    idx++;
                }
            }

            arg.sizes.dims[0] = tileM;
            arg.sizes.dims[1] = tileN;
            arg.sizes.block[0] = blockM;
            arg.sizes.block[1] = blockN;
        } else {
            const char *aname = parg.name;
            if (arg.name == "a") aname = "A";
            if (arg.name == "b") aname = "B";
            if (arg.name == "slm") aname = "slm_base";
            auto reg = interface.getArgument(aname);
            arg.location.resize(1);
            arg.location[0].boffset = reg.getBase() * GRF::bytes(hw) + reg.getByteOffset();
            arg.location[0].blen = reg.getBytes();
        }

        if (arg.name == "a") arg.actualType = microType(problem.Ta_ext);
        if (arg.name == "b") arg.actualType = microType(problem.Tb_ext);
        if (arg.name == "c") arg.actualType = microType(problem.Tc);

        if (transposeC) {
            if (arg.name == "a") arg.name = "b";
            else if (arg.name == "b") arg.name = "a";
            else if (arg.name == "lda") arg.name = "ldb";
            else if (arg.name == "ldb") arg.name = "lda";
            else if (arg.name == "m") arg.name = "n";
            else if (arg.name == "n") arg.name = "m";
            else if (arg.name == "i0") arg.name = "j0";
            else if (arg.name == "j0") arg.name = "i0";
            else if (arg.name == "local_id_m") arg.name = "local_id_n";
            else if (arg.name == "local_id_n") arg.name = "local_id_m";
        }

        package.arguments.push_back(std::move(arg));
    }

    auto effLoopM = !transposeC ? LoopM : LoopN;
    auto effLoopN = !transposeC ? LoopN : LoopM;
    package.settings.push_back({"sg_tile_m", strategy.unroll[effLoopM]});
    package.settings.push_back({"sg_tile_n", strategy.unroll[effLoopN]});
    package.settings.push_back({"wg_tile_m", strategy.wgTile(effLoopM)});
    package.settings.push_back({"wg_tile_n", strategy.wgTile(effLoopN)});
    package.settings.push_back({"sg_per_wg_m", strategy.wg[effLoopM]});
    package.settings.push_back({"sg_per_wg_n", strategy.wg[effLoopN]});
    package.settings.push_back({"sg_per_wg_k", strategy.wg[LoopK]});
    package.settings.push_back({"slm_size", int(gemmSLMSize(hw, problem, strategy, true))});

    package.barrierCount = interface.getBarrierCount();

    EntranceAgent::scan(package);

    return package;
}

static inline micro::StructuredType::Type microType(Type T)
{
    using ST = micro::StructuredType::Type;
#define CASE(x) case Type::x: return ST::x;
    switch (T) {
        CASE(f64)
        CASE(f32)
        CASE(f16)
        CASE(bf16)
        CASE(s32)
        CASE(s16)
        CASE(s8)
        CASE(u32)
        CASE(u16)
        CASE(u8)
        default: stub("Unsupported type");
    }
#undef CASE
}

#include "internal/namespace_end.hxx"
