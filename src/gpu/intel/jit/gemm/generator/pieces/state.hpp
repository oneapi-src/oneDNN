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

#ifndef GEMMSTONE_GUARD_STATE_HPP
#define GEMMSTONE_GUARD_STATE_HPP

#include "internal/ngen_includes.hpp"
#include "type.hpp"
#include "register_block.hpp"
#include "allocators.hpp"
#include "emulation.hpp"
#include "driver_info.hpp"
#include "problem.hpp"
#include "strategy.hpp"
#include "grf_multirange.hpp"

#include "internal/namespace_start.hxx"

// A pair of Subregisters in opposite banks.
class SubregisterPair {
protected:
    ngen::Subregister regs[2];
    bool negative;

public:
    SubregisterPair() : SubregisterPair(ngen::Subregister()) {}
    SubregisterPair(ngen::Subregister reg0, ngen::Subregister reg1) : regs{reg0, reg1}, negative(false) {}
    explicit SubregisterPair(ngen::Subregister reg) : SubregisterPair(reg, reg) {}

    /* implicit */ operator ngen::Subregister() const { return regs[0]; }

    SubregisterPair &operator=(ngen::Subregister reg) { regs[0] = regs[1] = reg; negative = false; return *this; }

    ngen::Subregister getReg(int idx) const;
    ngen::Subregister getRegAvoiding(ngen::HW hw, const ngen::RegData &rd) const;

    bool isValid()      const { return regs[0].isValid() && regs[1].isValid(); }
    bool isInvalid()    const { return !isValid(); }
    void invalidate()         { regs[0].invalidate(); regs[1].invalidate(); }

    bool isDuplicated() const { return regs[0] != regs[1]; }

    SubregisterPair operator-() const {
        auto copy = *this;
        copy.negative = !copy.negative;
        return copy;
    }
};

// A set of subregisters representing a value bit-shifted by some factor(s).
class MultishiftSubregister {
protected:
    static constexpr int maxShift = 5;
    ngen::Subregister regs[maxShift + 1] = {ngen::Subregister()};
    bool neg = false;

public:
    MultishiftSubregister operator-() const {
        auto copy = *this;
        copy.neg = !copy.neg;
        return copy;
    }

    ngen::Subregister operator>>(int shift) const {
        ngen::RegData sub = ngen::Subregister{};
        if (shift >= 0 && shift <= maxShift)
            sub = regs[shift];
        if (neg)
            sub = -sub;
        return *reinterpret_cast<ngen::Subregister *>(&sub);
    }

    void set(int shift, ngen::Subregister reg) {
        regs[shift] = reg;
    }
};

// Cached vector of leading dimension multiples (ld*0, ld*1, ld*2, etc.), for address setup.
struct LDMultiples {
    ngen::GRFRange range;
    bool a64 = false;
    int count = 0;
};

// Matrix information for setting up block 2D accesses.
struct Address2DParams {
    ngen::Subregister rows, cols;
    ngen::Subregister offR, offC;
    ngen::Subregister remR, remC;
    int fixedRows = 0, fixedCols = 0;
};

// Cached set of leading dimension multiples for address increments.
struct LDIncrements {
    using value_type = std::pair<int, SubregisterPair>;
    LDIncrements(const MatrixAddressingStrategy &s): type(s.base.isA64() ? ngen::DataType::q : ngen::DataType::ud) {}

    std::vector<value_type>::iterator begin() { return increments.begin(); }
    std::vector<value_type>::const_iterator begin() const { return increments.begin(); }
    std::vector<value_type>::iterator end() { return increments.end(); }
    std::vector<value_type>::const_iterator end() const { return increments.end(); }
    void push_back(const value_type & v) { increments.push_back(v); }
    void clear() { increments.clear(); }

    ngen::DataType type;
    std::vector<value_type> increments;
};

// Assignment of a logical mask to an variable and virtual flag register.
struct MaskAssignment {
    MaskInfo mask;              // Associated mask.
    VirtualFlag flag;           // Index of virtual flag register to use.
    LoopType var;               // Variable to base mask off of.
    uint16_t offset;            // Amount to subtract from variable.

    bool compatible(const MaskAssignment &other) const {
        return mask == other.mask && var == other.var && offset == other.offset;
    }
    void reverse(int width) {
        offset = width - offset - mask.variable.rsize;
        mask.variable.reverse = !mask.variable.reverse;
    }
};

// State parameters shared between different kernel types.
struct CommonState {
    ngen::RegisterAllocator ra;
    ngen::GRF signChange, selectImag;
    GRFMultirange vflagStorage;
    std::array<VirtualFlag, 8> activeVFlags;
    VirtualFlagAllocator raVFlag;
    TokenAllocator tokenAllocator;
    std::vector<std::pair<uint8_t, int8_t>> tokenMap;
    ngen::Subregister readFailures;
    ngen::Subregister fusedID;
    ngen::Subregister lsDescConstant[4];
    ngen::FlagRegister flagSwizzle;
    EmulationState emulate;
    ngen::GRFRange eatomicAddRegs[2];
    ngen::GRFRange remaskRegs[3];
    VirtualFlag vflagEAtomicAdd;
    VirtualFlag blockEMask;
    ngen::Label blockDone;
    ngen::Subregister all1s;
    ngen::RegData r0_info;
    bool movedR0 = false;
    ngen::Subregister lid0;
    GRFMultirange indexVec;                         // uw
    int ivEntries = 0;
    struct {
        ngen::GRF zero, one;
        ngen::GRFRange src1Storage;
        ngen::GRF src1, srcR1, srcI1, r, d;
        GRFMultirange mathTemp;
        ngen::GRF temp;
        std::array<ngen::FlagRegister, 2> tempFlags;
        ngen::Subregister flagStore;                // ud
        ngen::Label label;
        int simd;
        ngen::Subregister callStorageSub, callStorage;
        bool use = false;
    } invertSub;

    CommonState(ngen::HW hw) : ra(hw), raVFlag(hw), tokenAllocator(hw) {}

    VirtualFlag allocVFlag(ngen::HW hw, int n = 1);
    void wipeActiveVFlags();
    bool vflagsEnabled() const                    { return !vflagStorage.empty(); }
    void usePhysicalFlag(ngen::FlagRegister flag) { activeVFlags[flag.index()] = flag; }

    void allocEmulate64Temp(const EmulationStrategy &estrategy);
};

// GEMM kernel generator state.
struct GEMMState : public CommonState {
    struct Inputs {
        ngen::Subregister A, B, C[2], CO, base, tempC;      // q
        ngen::Subregister ao, bo, abo;                      // w/w/ud
        ngen::Subregister aoPtr, boPtr;                     // q
        ngen::Subregister aScalePtr, bScalePtr;             // q
        ngen::Subregister offsetA, offsetB, offsetC[2];     // q
        ngen::Subregister offsetAO, offsetBO, offsetCO;     // d
        ngen::Subregister offsetAScale, offsetBScale;       // d
        ngen::Subregister offsetAq, offsetBq;               // d
        ngen::Subregister lda, ldb, ldc[2];                 // d
        ngen::Subregister ldao, ldbo, ldco;                 // d
        ngen::Subregister ldaScale, ldbScale;               // d
        ngen::Subregister ldaq, ldbq;                       // d
        ngen::Subregister m, n, k, k0;                      // d
        SubregisterPair alpha_real, alpha_imag;             // T_real
        SubregisterPair beta_real, beta_imag;               // T_real
        ngen::Subregister alphaPtr, betaPtr;                // q
        ngen::Subregister groupIDM, groupIDN, groupIDK;     // ud
        ngen::Subregister groupIDMN;                        // ud
        ngen::GRF localIDM, localIDN, localIDK;             // uw
        ngen::Subregister localSizeM, localSizeN, localSizeK;       // ud
        ngen::Subregister groupCountM, groupCountN, groupCountK;    // ud
        ngen::Subregister groupCountMN;                     // ud
        ngen::Subregister gcMNRecip;                        // ud
        ngen::Subregister groupStride;                      // ud
        ngen::Subregister kvConfig, kRecip;                 // ud
        ngen::Subregister kSlicedTiles, kSyncSlabs;         // uw
        ngen::Subregister hilbertVD, hilbertUVDRecip;       // ud
        ngen::Subregister hilbertBail;                      // ud
        ngen::Subregister bslice, bthresh;                  // d
        ngen::Subregister flags;                            // ud
        ngen::Subregister diagA, diagB, diagC;              // q
        ngen::Subregister statusBuffer;                     // q
        uint8_t surfaceA, surfaceAO, surfaceAScale;         // BTS indices
        uint8_t surfaceB, surfaceBO, surfaceBScale;         // BTS indices
        uint8_t surfaceC[2], surfaceCO, surfaceTempC;       // BTS
        std::vector<ngen::Subregister> strideA;             // ud, used for strided batch.
        std::vector<ngen::Subregister> strideB;             // ud
        std::vector<ngen::Subregister> strideC;             // ud
        std::vector<ngen::Subregister> batchSize;           // ud
        std::vector<ngen::Subregister> recipBatchSize;      // ud
        ngen::Subregister offsetBatch;                      // ud, used for non-strided batch.
        ngen::Subregister incr_a_array, incr_b_array;       // ud, used for non-strided variable batch.
        ngen::Subregister incr_alpha, incr_beta;            // ud, used for non-strided variable batch.
        ngen::Subregister alpha_array, beta_array;          // q, used for non-strided variable batch.
        ngen::Subregister slmBase;                          // ud
        std::vector<ngen::Subregister> binarySrcs;          // q
        std::vector<ngen::Subregister> binaryOffsets;       // q/d
        std::vector<ngen::Subregister> binaryLDs;           // d
        std::vector<std::vector<ngen::Subregister>> binaryStrides;    // d
        std::vector<uint8_t> binarySurfaces;
        ngen::Subregister sroundSeedPtr;                    // q
        ngen::Subregister sroundSeed;                       // ud
    } inputs;
    Type Ta_load, Tb_load;                                  // Current type to be loaded into A/B_regs.
    Type Tacc;                                              // Current type in accumulator registers.
    ngen::Subregister persistentGroupID;                    // ud
    ngen::Subregister batchID[4];                           // ud
    ngen::Subregister offsetA, offsetB, offsetC[2];
    ngen::Subregister offsetAp, offsetBp, offsetCp;
    ngen::Subregister offsetCO;
    ngen::Subregister saveOffsetA, saveOffsetB, saveOffsetC[2];
    ngen::Subregister saveOffsetCO;
    ngen::Subregister fullK;
    ngen::Subregister effA, effB, effC[2], effCO, effTempC; // Offsets to base of A/B/C/CO/tempC chunks for loading/storing.
    ngen::Subregister effAi, effBi;
    ngen::Subregister effAo, effBo;
    ngen::Subregister effAp, effBp, effCp;
    ngen::Subregister effAs, effBs;
    std::vector<ngen::GRFRange> A_addrs, B_addrs, C_addrs[2];
    std::vector<ngen::GRFRange> A_addrsRem, B_addrsRem;
    std::vector<ngen::GRFRange> A_addrsAlt, B_addrsAlt;
    std::vector<ngen::GRFRange> A_addrsAltRem, B_addrsAltRem;
    std::vector<ngen::GRFRange> Ai_addrs, Bi_addrs;
    std::vector<std::vector<ngen::GRFRange>> Ai_addrsK, Bi_addrsK;
    std::vector<ngen::GRFRange> Ai_addrsRem, Bi_addrsRem;
    std::vector<ngen::GRFRange> Ao_addrs, Bo_addrs;
    std::vector<ngen::GRFRange> Ap_addrs, Bp_addrs, Cp_addrs;
    std::vector<ngen::GRFRange> Ap_addrsAlt, Bp_addrsAlt;
    std::vector<ngen::GRFRange> A_offsetAddrs, B_offsetAddrs;
    std::vector<ngen::GRFRange> A_scaleAddrs, B_scaleAddrs;
    std::vector<GRFMultirange> A_regs, B_regs, C_regs;
    GRFMultirange Ar_regs, Br_regs;                         // Repacked A/B registers.
    GRFMultirange Cr_regs;                                  // C registers to be repacked.
    std::vector<GRFMultirange> Ai_regs, Bi_regs;            // Incoming data to copy to SLM.
    std::vector<GRFMultirange> Ai_regsRem, Bi_regsRem;
    GRFMultirange Ao_regs, Bo_regs;                         // Outgoing data to copy to SLM.
    GRFMultirange Ao_regsRem, Bo_regsRem;
    GRFMultirange As_regs, Bs_regs;                         // A row sums/B column sums.
    GRFMultirange Asr_regs, Bsr_regs;                       // A row sums/B column sums to be repacked.
    GRFMultirange Ap_regs, Bp_regs, Cp_regs;                // A/B/C prefetch registers.
    GRFMultirange A_offsetRegs, B_offsetRegs;               // A/B offsets (grouped).
    GRFMultirange A_scaleRegs, B_scaleRegs;                 // A/B scales (grouped).
    GRFMultirange Ar_offsetRegs, Br_offsetRegs;             // Repacked A/B offsets.
    GRFMultirange Ar_scaleRegs, Br_scaleRegs;               // Repacked A/B scales.
    std::vector<MaskAssignment> AB_masks, AB_masksCoop;
    ngen::GRFRange broadcast_regs;
    std::vector<ngen::GRFRange> tempMul_regs;
    ngen::Subregister groupCountMN, groupIDMN;              // ud
    ngen::Subregister i0, j0, h0;                           // d
    ngen::Subregister wgI0, wgJ0;                           // d
    ngen::Subregister threadK0, k0Rem, wgK;                 // ud
    ngen::Subregister remainders[3];                        // d (todo: w)
    ngen::Subregister remaindersFused[2];                   // w
    ngen::Subregister remaindersWG[2];                      // d (todo: w)
    ngen::Subregister remaindersCoop[3];                    // d
    ngen::Subregister remFusedStorage;                      // d
    ngen::Subregister diagA, diagB, diagC;                  // d
    SubregisterPair lda, ldb;
    SubregisterPair ldao, ldbo, ldaScale, ldbScale;
    LDIncrements ldaIncrements, ldbIncrements;              // Cached lda * ka, ldb * kb
    LDIncrements ldaoIncrements, ldboIncrements;
    LDIncrements ldasIncrements, ldbsIncrements;
    LDMultiples ldaMultiples, ldbMultiples, ldcMultiples[2];
    ngen::Subregister k, K;                                 // d
    ngen::Subregister kNoBarrierStart, kNoBarrierEnd;       // d
    ngen::FlagRegister flagAP;
    ngen::Subregister beta1;                                // d
    ngen::Subregister add64;                                // uw
    ngen::Subregister lidM, lidN, lidStorage;               // uw, uw, ud
    ngen::Subregister lidK, lszK, lidszKStorage;            // uw, uw, ud
    ngen::Subregister ia0_slm, jb0_slm;                     // uw
    ngen::Subregister postRemA, postRemB;                   // ud
    ngen::Subregister postRemAi, postRemBi;                 // ud
    ngen::Subregister postRemAo, postRemBo;                 // ud
    ngen::Subregister isCompute;                            // ud
    ngen::GRF sysSumAll1s;                                  // Ta/Tb
    ngen::GRF betaCheckReturn;
    ngen::Subregister statusFlagAddr;                        // uq
    bool systolicSumA = false, systolicSumB = false;
    bool lateKLoopCheck = false;
    bool splitBarrierAlways = false;
    int ka_loadRem, kb_loadRem;
    bool A_lateKRem, B_lateKRem;
    bool A_descRem, B_descRem;
    bool Ai_hasKRem, Bi_hasKRem;
    bool Ai_lateKRem, Bi_lateKRem;
    bool Ai_incrementalRem, Bi_incrementalRem;
    bool Ai_remIncrCopy, Bi_remIncrCopy;
    int ma_slm, ka_slm, kb_slm, nb_slm;
    int ma_prefetch, ka_prefetch, kb_prefetch, nb_prefetch;
    CoopSplit effCoopA = CoopSplit::K;
    CoopSplit effCoopB = CoopSplit::K;
    ngen::Subregister kSLMA, kSLMB, kSLMStorage;            // w/w/ud
    bool kSLMCountUp = false;
    int kaq = 0, kbq = 0, kaqStride, kbqStride, kaqLate = 0, kbqLate = 0;
    bool lateScale2DA = false, lateScale2DB = false;
    std::vector<RegisterBlock> A_layout, B_layout, C_layout;
    std::vector<RegisterBlock> A_layoutRem, B_layoutRem;
    std::vector<RegisterBlock> A_layoutAlt, B_layoutAlt;
    std::vector<RegisterBlock> A_layoutAltRem, B_layoutAltRem;
    std::vector<RegisterBlock> Ar_layout, Br_layout;
    std::vector<RegisterBlock> Ai_layout, Bi_layout;
    std::vector<std::vector<RegisterBlock>> Ai_layoutK, Bi_layoutK;
    std::vector<RegisterBlock> Ai_layoutRem, Bi_layoutRem;
    std::vector<RegisterBlock> Ao_layout, Bo_layout;
    std::vector<RegisterBlock> As_layout, Bs_layout;
    std::vector<RegisterBlock> Asr_layout, Bsr_layout;
    std::vector<RegisterBlock> Ap_layout, Bp_layout, Cp_layout;
    std::vector<RegisterBlock> Ap_layoutAlt, Bp_layoutAlt;
    std::vector<RegisterBlock> A_offsetLayout, B_offsetLayout;
    std::vector<RegisterBlock> A_scaleLayout, B_scaleLayout;
    std::vector<RegisterBlock> Ar_offsetLayout, Br_offsetLayout;
    std::vector<RegisterBlock> Ar_scaleLayout, Br_scaleLayout;
    std::vector<RegisterBlock> Cr_layout;
    std::vector<RegisterBlock> C_layoutReduced;
    std::vector<RegisterBlock> C_layoutExt, C_layoutExtUnmasked, C_layoutExtNonatomicUnmasked;
    Address2DParams A_params, B_params;
    Address2DParams Ai_params, Bi_params;
    Address2DParams Ap_params, Bp_params;
    int Ai_regCount = 0, Bi_regCount = 0;
    bool aioShare = false, bioShare = false;
    bool aioShareRem = false, bioShareRem = false;
    bool aoReuseA = false, boReuseB = false;
    Type Tao_int, Ta_scaleInt;
    Type Tbo_int, Tb_scaleInt;
    Type Ta_scaleOp, Tb_scaleOp;
    MatrixAddressing Ai, Bi, Ao, Bo, tempC;
    MatrixAddressingStrategy Ai_strategy, Bi_strategy;
    MatrixAddressingStrategy Ao_strategy, Bo_strategy;
    MatrixAddressingStrategy Cext_strategy, tempCStrategy;
    ngen::FlagRegister panelMaskA, panelMaskB;
    int8_t tokenBarrierFence[2];
    ngen::InstructionModifier modBarrierFence[2];
    bool barrierReady = false;
    ngen::GRF barrierHeader;
    ngen::GRF barrierHeaderM, barrierHeaderN;
    ngen::FlagRegister barrierM, barrierN;
    bool firstKLoopSegment;
    bool isNested = false;
    int C_accCount;
    bool cSwapActive = false;
    bool haveCSwap = false;
    int C_count = 1;
    int C_buffers = 1;
    bool allocedAo = false, allocedBo = false;
    bool allowEmptyC = false;
    bool copyC = false;
    bool useTempC = false;
    bool broadcast;
    bool repackA = false, repackB = false;
    bool repackARem = false, repackBRem = false;
    int ka_repack, ka_repackRem, kb_repackRem;
    int cRepackPeriod = 0;
    bool remActiveA, remActiveB, remActiveSLM;
    std::vector<MaskAssignment> kMasksA, kMasksB, kMasksAi, kMasksBi;
    int initSLMKOffset = 0;
    bool slmRemaskA = false, slmRemaskB = false;
    bool slmASums = false, slmBSums = false;
    bool doLateExit = false;
    bool needBRFallback = true;
    ngen::GRF emulate64TempSave[2];
    bool simd32KMasks = false;
    int lastThresh = 0;
    ngen::Subregister nextGroupIDM, nextGroupIDN;
    ngen::Subregister nextFlagL3PFA, nextFlagL3PFB;
    ngen::FlagRegister flagL3PFA, flagL3PFB;
    std::vector<RegisterBlock> Apl3_layout, Bpl3_layout;
    std::vector<ngen::GRFRange> Apl3_addrs, Bpl3_addrs;

    std::vector<ngen::Subregister> effBinary;

    struct {
        ngen::InstructionModifier depAddr[4];
    } sysgemm;

    GEMMState(ngen::HW hw, const GEMMStrategy& strategy) : CommonState(hw),
                                                           ldaIncrements(strategy.A),
                                                           ldbIncrements(strategy.B),
                                                           ldaoIncrements(strategy.AO),
                                                           ldboIncrements(strategy.BO),
                                                           ldasIncrements(strategy.A_scale),
                                                           ldbsIncrements(strategy.B_scale) {}

    int internalSIMD() const { return simd32KMasks ? 32 : 16; }

};

#include "internal/namespace_end.hxx"

#endif /* header guard */
