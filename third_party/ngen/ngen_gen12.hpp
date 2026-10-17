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

/*
 * Do not #include this file directly; ngen uses it internally.
 */
#ifndef NGEN_GEN12_HPP
#define NGEN_GEN12_HPP

#include "ngen_auto_swsb.hpp"
#include "ngen_gen8.hpp"

namespace NGEN_NAMESPACE {
// Gen12 binary encoding.

struct EncodingTag12 {};
struct EncodingTagXeHPC {};

template <HW hw> struct EncodingTag12Dispatch       { using tag = EncodingTag12; };
template <> struct EncodingTag12Dispatch<HW::XeHPC> { using tag = EncodingTagXeHPC; };
template <> struct EncodingTag12Dispatch<HW::Xe2>   { using tag = EncodingTagXeHPC; };
template <> struct EncodingTag12Dispatch<HW::Xe3>   { using tag = EncodingTagXeHPC; };
struct EncodingTagXe3p : public EncodingTagXeHPC {};

template <> struct EncodingTag12Dispatch<HW::Xe3p>  { using tag = EncodingTagXe3p; };

class SWSBInfo12
{
    friend class InstructionModifier;
protected:
    union {
        struct {
            unsigned dist : 3;
            unsigned pipe : 4;
            unsigned combined : 1;
        } pipeline;
        struct {
            unsigned sbid : 4;
            unsigned mode : 3;
            unsigned combined : 1;
        } scoreboard;
        struct {
            unsigned sbid : 4;
            unsigned dist : 3;
            unsigned combined : 1;
        } combined;
        uint8_t all;
    };

    constexpr SWSBInfo12(uint8_t all_, bool dummy) : all{all_} {}

    constexpr bool isPipeline() const {
        return !combined.combined && ((scoreboard.mode < 2) || (scoreboard.mode > 4));
    }

public:
    constexpr SWSBInfo12() : all{0} {}

    SWSBInfo12(SWSBInfo info, Opcode op) {
        normalizeSWSB(info);    /* token in slot 0, pipe in slot 1 */
        if (info[0] && info[1]) {
            combined.sbid = info[0].token.token;
            combined.dist = info[1].pipe.dist;
            combined.combined = true;
        } else if (info[1]) {
            combined.combined = false;
            uint8_t pipeMap[8] = {0, 1, 2, 3, 10, 0, 0, 0};
            pipeline.dist = info[1].pipe.dist;
            pipeline.pipe = pipeMap[info[1].pipe.pipe & 7];
        } else if (info[0]) {
            combined.combined = false;
            combined.sbid = info[0].token.token;
            scoreboard.mode = 1 + info[0].tokenMode();
        } else
            all = 0;
    }

    SWSBInfo decode(Opcode op, unsigned dstTypecode) const {
        if (combined.combined) {
            bool vl = trackedByToken(HW::Gen12LP, op, dstTypecode);
            auto pipe = (op == Opcode::send || op == Opcode::sendc) ? Pipe::A : Pipe::Default;
            return {SWSBItem(combined.sbid, vl, true), SWSBItem(pipe, combined.dist)};
        } else if (isPipeline()) {
            static const Pipe pipeMap[4] = {Pipe::Default, Pipe::A, Pipe::F, Pipe::I};
            auto pipe = (pipeline.pipe == 10) ? Pipe::L : pipeMap[pipeline.pipe & 3];
            return {SWSBItem(pipe, pipeline.dist)};
        } else
            return {SWSBItem(scoreboard.sbid, scoreboard.mode != 2, scoreboard.mode != 3)};
    }

    constexpr bool empty() const                              { return all == 0; }
    constexpr uint8_t raw() const                             { return all; }
    static constexpr14 SWSBInfo12 createFromRaw(uint8_t all_) { return SWSBInfo12(all_, false); }
};

class SWSBInfoXeHPC
{
    friend class InstructionModifier;
protected:
    union {
        struct {
            unsigned dist : 3;
            unsigned pipe : 4;
            unsigned sb : 1;
            unsigned mode : 2;
            unsigned : 6;
        } pipeline;
        struct {
            unsigned sbid : 5;
            unsigned type : 2;  // .dst: 0, .src: 1, .set: 2
            unsigned sb : 1;
            unsigned mode : 2;
            unsigned : 6;
        } scoreboard;
        struct {
            unsigned sbid : 5;
            unsigned dist : 3;
            unsigned mode : 2;
            unsigned : 6;
        } combined;
        uint16_t all;
    };

    constexpr SWSBInfoXeHPC(uint16_t all_, bool dummy) : all{all_} {}

    static constexpr14 unsigned combinedMode(SWSBItem itoken, SWSBItem ipipe, Opcode op) {
        if (op == Opcode::dpas) {
            if (itoken.token.src && itoken.token.dst) return 1;
            if (itoken.token.src) return 2;
            if (itoken.token.dst) return 3;
            return 0;
        }
        auto pipe = ipipe.getPipe();
        if (itoken.token.src && itoken.token.dst)
            return (pipe == Pipe::F) ? 2 : (pipe == Pipe::I) ? 3 : 1;
        if (itoken.token.src) return 2;
        if (itoken.token.dst) return (pipe == Pipe::A) ? 3 : 1;
        return 0;
    }

public:
    constexpr SWSBInfoXeHPC() : all{0} {}

    SWSBInfoXeHPC(SWSBInfo info, Opcode op) {
        normalizeSWSB(info);    /* token in slot 0, pipe in slot 1 */
        if (info[1].isNoAccSBSet())
            all = 0xF0;
        else if (info[0] && info[1]) {
            combined.sbid = info[0].token.token;
            combined.dist = info[1].pipe.dist;
            combined.mode = combinedMode(info[0], info[1], op);
        } else if (info[1]) {
            pipeline.dist = info[1].pipe.dist;
            pipeline.pipe = info[1].pipe.pipe;
            pipeline.sb = false;
            pipeline.mode = 0;
        } else if (info[0]) {
            scoreboard.sbid = info[0].token.token;
            scoreboard.type = info[0].tokenMode() - 1;
            scoreboard.sb = true;
            scoreboard.mode = 0;
        } else
            all = 0;
    }

    SWSBInfo decode(Opcode op) const {
        if (all == 0xF0)
            return {SWSBItem::createNoAccSBSet()};

        SWSBItem ipipe(pipe(op), dist());
        SWSBItem itoken;

        if (combined.mode) {
            bool src, dst;
            if (isSend(op))
                src = dst = true;
            else if (op == Opcode::dpas) {
                src = (combined.mode <= 2);
                dst = combined.mode & 1;
            } else {
                dst = combined.mode & 1;
                src = !dst;
            }
            itoken = SWSBItem(combined.sbid, src, dst);
        } else if (scoreboard.sb)
            itoken = SWSBItem(scoreboard.sbid, scoreboard.type != 0, scoreboard.type != 1);

        return {ipipe, itoken};
    }

    constexpr bool empty() const { return all == 0; }
    constexpr14 int dist() const {
        if (combined.mode)
            return combined.dist;
        else if (!scoreboard.sb)
            return pipeline.dist;
        else
            return 0;
    }
    constexpr14 Pipe pipe(Opcode op) const {
        if (combined.mode) {
            if (isSend(op))
                return (combined.mode == 1) ? Pipe::A : (combined.mode == 2) ? Pipe::F : Pipe::I;
            if (op == Opcode::dpas)
                return Pipe::Default;
            return (combined.mode == 3) ? Pipe::A : Pipe::Default;
        } else if (!scoreboard.sb) {
            const Pipe table[8] = {Pipe::Default, Pipe::A, Pipe::F, Pipe::I, Pipe::L, Pipe::M, Pipe::S, Pipe::A};
            return table[pipeline.pipe];
        } else
            return Pipe::Default;
    }

    constexpr uint16_t raw() const { return all; }
    static constexpr14 SWSBInfoXeHPC createFromRaw(uint16_t all_) { return SWSBInfoXeHPC(all_, false); }
};

// 24 bits of data common between src0 and src1 (lower 16 bits common with dst)
union BinaryOperand12 {
    uint32_t bits;
    struct {
        unsigned hs : 2;
        unsigned regFile : 1;
        unsigned subRegNum : 5;
        unsigned regNum : 8;
        unsigned addrMode : 1;          // = 0 (direct)
        unsigned width : 3;
        unsigned vs : 4;
    } direct;
    struct {
        unsigned hs : 2;
        unsigned addrOff : 10;
        unsigned addrReg : 4;
        unsigned addrMode : 1;          // = 1 (indirect)
        unsigned width : 3;
        unsigned vs : 4;
    } indirect;
    struct {
        unsigned : 20;
        unsigned vs : 3;
        unsigned subRegNum0 : 1;
    } directXeHPC;
    struct {
        unsigned : 20;
        unsigned vs : 3;
        unsigned addrOff0 : 1;
    } indirectXeHPC;
};

// 16 bits of data common between dst, src0/1/2 for 3-source instructions
union TernaryOperand12 {
    uint16_t bits;
    struct {
        unsigned hs : 2;
        unsigned regFile : 1;
        unsigned subRegNum : 5;         // mme# for math
        unsigned regNum : 8;
    } direct;
};

struct Instruction12 {
    union {
        struct {                            // Lower 35 bits are essentially common.
            unsigned opcode : 8;            // High bit reserved, used for auto-SWSB flag.
            unsigned swsb : 8;
            unsigned execSize : 3;
            unsigned execOffset : 3;
            unsigned flagReg : 2;
            unsigned predCtrl : 4;
            unsigned predInv : 1;
            unsigned cmptCtrl : 1;
            unsigned debugCtrl : 1;
            unsigned maskCtrl : 1;
            //
            unsigned atomicCtrl : 1;
            unsigned accWrCtrl : 1;
            unsigned saturate : 1;
            unsigned : 29;
            //
            unsigned : 32;
            unsigned : 32;
        } common;
        struct {
            unsigned : 8;
            unsigned swsb : 10;
            unsigned execSize : 3;
            unsigned flagReg : 3;
            unsigned execOffset : 2;
            unsigned predCtrl : 2;
            unsigned : 4;
            //
            unsigned : 1;
            unsigned dstExt : 1;    // Low bit of subRegNum [direct] or addrOff [indirect]
            unsigned : 30;
            //
            unsigned : 32;
            unsigned : 32;
        } commonXeHPC;
        struct {
            unsigned : 32;
            //
            unsigned : 3;
            unsigned dstAddrMode : 1;
            unsigned dstType : 4;
            unsigned src0Type : 4;
            unsigned src0Mods : 2;
            unsigned src0Imm : 1;
            unsigned src1Imm : 1;
            unsigned dst : 16;              // first 16 bits of BinaryOperand12
            //
            unsigned src0 : 24;             // BinaryOperand12
            unsigned src1Type : 4;
            unsigned cmod : 4;
            //
            unsigned src1 : 24;             // BinaryOperand12
            unsigned src1Mods : 2;
            unsigned _ : 6;
        } binary;
        struct {
            uint64_t _;
            uint32_t __;
            uint32_t value;
        } imm32;
        struct {
            uint64_t _;
            uint32_t high;
            uint32_t low;
        } imm64;
        struct {
            unsigned : 32;                  // common
            unsigned : 3;
            unsigned src0VS0 : 1;
            unsigned dstType : 3;
            unsigned execType : 1;
            unsigned src0Type : 3;
            unsigned src0VS1 : 1;
            unsigned src0Mods : 2;
            unsigned src0Imm : 1;
            unsigned src2Imm : 1;
            unsigned dst : 16;              // TernaryOperand12
            //
            unsigned src0 : 16;             // TernaryOperand12 or immediate
            unsigned src2Type : 3;
            unsigned src1VS0 : 1;
            unsigned src2Mods : 2;
            unsigned src1Mods : 2;
            unsigned src1Type : 3;
            unsigned src1VS1 : 1;
            unsigned cmod : 4;              // same location as binary
            //
            unsigned src1 : 16;             // TernaryOperand12
            unsigned src2 : 16;             // TernaryOperand12 or immediate
        } ternary;
        struct {
            unsigned : 32;
            unsigned : 32;
            unsigned : 20;
            unsigned bfnCtrl03 : 4;
            unsigned : 4;
            unsigned bfnCtrl47 : 4;
            unsigned : 32;
        } bfn;
        struct {
            unsigned : 32;
            //
            unsigned : 11;
            unsigned rcount : 3;
            unsigned : 2;
            unsigned sdepth : 2;
            unsigned : 14;
            //
            unsigned : 20;
            unsigned src2SubBytePrecision : 2;
            unsigned src1SubBytePrecision : 2;
            unsigned : 8;
            //
            unsigned : 32;
        } dpas;
        struct {
            unsigned : 32;
            //
            unsigned : 1;
            unsigned fusionCtrl : 1;
            unsigned eot : 1;
            unsigned exDesc11_23 : 13;
            unsigned descIsReg : 1;
            unsigned exDescIsReg : 1;
            unsigned dstRegFile : 1;
            unsigned desc20_24 : 5;
            unsigned dstReg : 8;
            //
            unsigned exDesc24_25 : 2;
            unsigned src0RegFile : 1;
            unsigned desc25_29 : 5;
            unsigned src0Reg : 8;
            unsigned : 1;
            unsigned desc0_10 : 11;
            unsigned sfid : 4;
            //
            unsigned exDesc26_27 : 2;
            unsigned src1RegFile : 1;
            unsigned exDesc6_10 : 5;
            unsigned src1Reg : 8;
            unsigned : 1;
            unsigned desc11_19 : 9;
            unsigned desc30_31 : 2;
            unsigned exDesc28_31 : 4;
        } send;
        struct {
            unsigned : 32;
            unsigned : 7;
            unsigned exBSO : 1;
            unsigned exDescReg : 3;
            unsigned : 21;
            unsigned : 32;
            unsigned : 32;
        } sendIndirect;
        struct {
            unsigned : 32;
            //
            unsigned eot : 1;
            unsigned dstRegFile : 1;
            unsigned ind0Present : 1;
            unsigned ind1Present : 1;
            unsigned desc40_41 : 2;
            unsigned ind1_desc42_46 : 5;
            unsigned ind0 : 3;
            unsigned : 2;
            unsigned desc32_39 : 8;
            unsigned dstReg : 8;
            //
            unsigned desc30_31 : 2;
            unsigned src0RegFile : 1;
            unsigned src0Len : 5;
            unsigned src0Reg : 8;
            unsigned desc16_27 : 12;
            unsigned sfid : 4;
            //
            unsigned desc28_29 : 2;
            unsigned src1RegFile : 1;
            unsigned src1Len : 5;
            unsigned src1Reg : 8;
            unsigned desc0_15 : 16;
        } sendg;
        struct {
            unsigned : 32;                  // common
            unsigned : 1;
            unsigned branchCtrl : 1;
            unsigned : 30;
            int32_t uip;
            int32_t jip;
        } branches;
        struct {
            unsigned : 32;
            //
            unsigned : 12;
            unsigned dstReg8 : 1;
            unsigned : 19;
            //
            unsigned : 32;
            unsigned : 32;
        } unaryXe3pImm;
        struct {
            unsigned : 32;
            //
            unsigned : 11;
            unsigned src0Reg8 : 1;
            unsigned : 2;
            unsigned dstReg8 : 1;
            unsigned : 17;
            //
            unsigned : 32;
            unsigned : 32;
        } binaryXe3pImm;
        struct {
            unsigned : 32;
            //
            unsigned : 7;
            unsigned execType : 1;
            unsigned : 24;
            //
            unsigned : 32;
            //
            unsigned dstReg8 : 1;
            unsigned src0Reg8 : 1;
            unsigned : 18;
            unsigned src1Scalar : 1;
            unsigned : 5;
            unsigned src1Reg8 : 1;
            unsigned : 5;
        } binaryXe3p;
        struct {
            unsigned : 32;
            //
            unsigned : 17;
            unsigned src1Scalar : 1;
            unsigned : 14;
            //
            unsigned : 19;
            unsigned src1Reg8 : 1;
            unsigned : 7;
            unsigned src2Reg8 : 1;
            unsigned : 4;
            //
            unsigned dstReg8 : 1;
            unsigned src0Reg8 : 1;
            unsigned : 30;
        } ternaryXe3p;
        struct {
            unsigned : 32;
            //
            unsigned : 12;
            unsigned dstReg8 : 1;
            unsigned src0Reg8 : 1;
            unsigned : 18;
            //
            unsigned : 32;
            unsigned : 32;
        } branchXe3p;
        struct {
            unsigned : 32;
            //
            unsigned : 1;
            unsigned dstReg8 : 1;
            unsigned : 30;
            //
            unsigned : 2;
            unsigned src0Reg8 : 1;
            unsigned : 29;
            //
            unsigned : 2;
            unsigned src1Reg8 : 1;
            unsigned : 29;
        } sendgx;
        struct {
            unsigned : 32;
            //
            unsigned : 2;
            unsigned src4RegFile : 1;
            unsigned src3RegFile : 1;
            unsigned : 7;
            unsigned src4Reg0_3 : 4;
            unsigned src3Reg0 : 1;
            unsigned : 16;
            //
            unsigned src3Reg1_2 : 2;
            unsigned : 26;
            unsigned src3Reg3_6 : 4;
            //
            unsigned : 3;
            unsigned src4SubReg3_5 : 3;
            unsigned src3SubReg4_5 : 2;
            unsigned : 8;
            unsigned src3Reg7_8 : 2;
            unsigned : 1;
            unsigned src4Reg4_8 : 5;
            unsigned : 8;
        } bdpas;
        uint64_t qword[2];
    };

    constexpr Instruction12() : qword{0,0} {};

    // Decoding routines for auto-SWSB.
    bool autoSWSB() const         { return (common.opcode & 0x80); }
    SWSBInfo swsb() const         { return SWSBInfo12::createFromRaw(common.swsb).decode(opcode(), dstTypecode()); }
    void setSWSB(SWSBInfo swsb)   { common.swsb = SWSBInfo12(swsb, opcode()).raw(); }
    void clearAutoSWSB()          { common.opcode &= 0x7F; }
    Opcode opcode() const         { return static_cast<Opcode>(common.opcode & 0x7F); }
    SyncFunction syncFC() const   { return static_cast<SyncFunction>(binary.cmod); }
    SharedFunction sfid() const   { return static_cast<SharedFunction>(send.sfid); }
    bool eot() const              { return isSend(opcode()) && (isSendg(opcode()) ? sendg.eot : send.eot); }
    bool predicated() const       { return !common.maskCtrl || (static_cast<PredCtrl>(common.predCtrl) != PredCtrl::None); }
    bool atomic() const           { return common.atomicCtrl; }
    unsigned dstTypecode() const  { return binary.dstType; }
    unsigned src0Typecode() const { return srcTypecode(0); }
    unsigned src1Typecode() const { return srcTypecode(1); }
    void shiftJIP(int32_t shift)  { branches.jip += shift; }
    void shiftUIP(int32_t shift)  { branches.uip += shift; }
    static bool is64()            { return false; }

    inline autoswsb::DestinationMask destinations(int &jip, int &uip) const;
    template <typename Tag = EncodingTag12>
    inline bool getOperandRegion(autoswsb::DependencyRegion &region, int opNum) const;
    template <typename Tag = EncodingTag12>
    inline bool getCModDepRegion(autoswsb::DependencyRegion &region) const;
    inline bool getImm32(uint32_t &imm) const;
    inline bool getSendDesc(MessageDescriptor &desc) const;
    inline bool getARFType(ARFType &arfType, int opNum, HW hw) const;
    inline int getFencedepJIP() const;
    inline SendgMessageDescriptor getSendgDesc() const;

    bool isMathMacro() const {
        if (opcode() != Opcode::math) return false;
        auto fc = static_cast<MathFunction>(binary.cmod);
        return (fc == MathFunction::invm || fc == MathFunction::rsqtm);
    }

protected:
    inline unsigned srcTypecode(int opNum) const;
};

static_assert(sizeof(Instruction12) == 16, "Internal error: Instruction12 has been padded by the compiler.");

struct InstructionXeHPC : public Instruction12 {
    SWSBInfo swsb() const        { return SWSBInfoXeHPC::createFromRaw(commonXeHPC.swsb).decode(opcode()); }
    void setSWSB(SWSBInfo swsb)  { commonXeHPC.swsb = SWSBInfoXeHPC(swsb, opcode()).raw(); }

    template <typename Tag = EncodingTagXeHPC>
    bool getOperandRegion(autoswsb::DependencyRegion &region, int opNum) const {
        return Instruction12::getOperandRegion<EncodingTagXeHPC>(region, opNum);
    }

    template <typename Tag = EncodingTagXeHPC>
    bool getCModDepRegion(autoswsb::DependencyRegion &region) const {
        return Instruction12::getCModDepRegion<EncodingTagXeHPC>(region);
    }

    bool atomic() const {
        if (isSendg(opcode())) return false;    /* no atomic field */
        return Instruction12::atomic();
    }
};

static_assert(sizeof(InstructionXeHPC) == 16, "Internal error: InstructionXeHPC has been padded by the compiler.");

struct InstructionXe3p : public InstructionXeHPC {
    template <typename Tag = EncodingTagXe3p>
    bool getOperandRegion(autoswsb::DependencyRegion &region, int opNum) const {
        return Instruction12::getOperandRegion<EncodingTagXe3p>(region, opNum);
    }
};

static_assert(sizeof(InstructionXe3p) == 16, "Internal error: InstructionXe3p has been padded by the compiler.");

// Encoding routines.

static inline unsigned getTypecode12(DataType type)
{
    static const uint8_t conversionTable[32] = {2,6,1,5,0,4,11,10,3,7,9,13,8,0,4,8,
                                                14,12,2,2,2,2,2,2,2,2,8,8,0,4,0,4};
    return conversionTable[static_cast<unsigned>(type) & 0x1F];
}

static inline unsigned encodeSubBytePrecision12(DataType type)
{
    static const uint8_t conversionTable[32] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                                0,0,0,0,0,0,0,0,0,0,2,1,1,1,2,2};
    return conversionTable[static_cast<unsigned>(type) & 0x1F];
}

static inline unsigned pow2Encode(unsigned x)
{
    return (x == 0) ? 0 : (1 + utils::log2(x));
}

template <int srcN, bool encodeHS = true>
static inline constexpr14 BinaryOperand12 encodeBinaryOperand12(const RegData &rd, EncodingTag12 tag)
{
    BinaryOperand12 op{0};

#ifdef NGEN_SAFE
    if (rd.isInvalid()) throw invalid_object_exception();
#endif

    if (rd.isIndirect()) {
        op.indirect.addrOff = rd.getOffset();
        op.indirect.addrReg = rd.getIndirectOff();
        op.indirect.addrMode = 1;
        if (srcN >= 0)
            op.indirect.vs = (rd.isVxIndirect()) ? 0xF : pow2Encode(rd.getVS());
    } else {
        op.direct.regFile = rd.getRegFile8();
        op.direct.subRegNum = rd.getByteOffset();
        op.direct.regNum = rd.getBase();
        op.direct.addrMode = 0;
        if (srcN >= 0)
            op.direct.vs = pow2Encode(rd.getVS());
    }

    if (encodeHS)
        op.direct.hs = pow2Encode(rd.getHS());

    if (srcN >= 0) op.direct.width = utils::log2(rd.getWidth());

    return op;
}

template <int srcN, bool encodeHS = true>
static inline constexpr14 BinaryOperand12 encodeBinaryOperand12(const RegData &rd, EncodingTagXeHPC tag)
{
    BinaryOperand12 op{0};

#ifdef NGEN_SAFE
    if (rd.isInvalid()) throw invalid_object_exception();
#endif

    if (rd.isIndirect()) {
        op.indirect.addrOff = (rd.getOffset() >> 1);
        op.indirect.addrReg = rd.getIndirectOff();
        op.indirect.addrMode = 1;
        if (srcN >= 0) {
            op.indirect.vs = (rd.isVxIndirect()) ? 0xF : pow2Encode(rd.getVS());
            op.indirectXeHPC.addrOff0 = (rd.getOffset() & 1);
        }
    } else {
        op.direct.regFile = rd.getRegFile8();
        op.direct.subRegNum = (rd.getByteOffset() >> 1);
        op.direct.regNum = rd.getBase();
        op.direct.addrMode = 0;
        if (srcN >= 0) {
            op.directXeHPC.vs = pow2Encode(rd.getVS());
            op.directXeHPC.subRegNum0 = rd.getByteOffset() & 1;
        }
    }

    if (encodeHS)
        op.direct.hs = pow2Encode(rd.getHS());

    if (srcN >= 0) op.direct.width = utils::log2(rd.getWidth());

    return op;
}

template <int srcN, bool encodeHS = true>
static inline constexpr14 BinaryOperand12 encodeBinaryOperand12(const RegData &rd, EncodingTagXe3p tag)
{
    BinaryOperand12 op{0};

#ifdef NGEN_SAFE
    if (rd.isInvalid()) throw invalid_object_exception();
#endif

    if (rd.isIndirect()) {
        op.indirect.addrOff = (rd.getOffset() >> 1);
        op.indirect.addrReg = rd.getIndirectOff();
        op.indirect.addrMode = 1;
        if (srcN == 0) {
            op.indirect.vs = (rd.isVxIndirect()) ? 0xFFFF : pow2Encode(rd.getVS());
            op.indirectXeHPC.addrOff0 = (rd.getOffset() & 1);
        }
    } else {
        op.direct.regFile = rd.getRegFile8();
        op.direct.subRegNum = (rd.getByteOffset() >> 1);
        op.direct.regNum = rd.getBase();
        op.direct.addrMode = 0;
        if (srcN == 0)
            op.directXeHPC.vs = pow2Encode(rd.getVS());
        if (srcN >= 0)
            op.directXeHPC.subRegNum0 = rd.getByteOffset() & 1;
    }

    if (encodeHS && srcN <= 0)
        op.direct.hs = pow2Encode(rd.getHS());

    if (srcN == 0) op.direct.width = utils::log2(rd.getWidth());

    return op;
}

template <int srcN, typename Tag>
static inline constexpr14 BinaryOperand12 encodeBinaryOperand12(const ExtendedReg &reg, Tag tag)
{
    auto op = encodeBinaryOperand12<srcN>(reg.getBase(), tag);
    op.direct.subRegNum = reg.getMMENum();

    return op;
}

template <bool dest, bool encodeHS = true>
static inline constexpr14 TernaryOperand12 encodeTernaryOperand12(const RegData &rd, EncodingTag12 tag)
{
#ifdef NGEN_SAFE
    if (rd.isInvalid()) throw invalid_object_exception();
    if (rd.isIndirect()) throw invalid_operand_exception();
#endif

    TernaryOperand12 op{0};

    if (encodeHS)
        op.direct.hs = dest ? utils::log2(rd.getHS()) : pow2Encode(rd.getHS());

    op.direct.regFile = rd.getRegFile8();
    op.direct.subRegNum = rd.getByteOffset();
    op.direct.regNum = rd.getBase();

    return op;
}

template <bool dest, bool encodeHS = true>
static inline constexpr14 TernaryOperand12 encodeTernaryOperand12(const RegData &rd, EncodingTagXeHPC tag)
{
#ifdef NGEN_SAFE
    if (rd.isInvalid()) throw invalid_object_exception();
    if (rd.isIndirect()) throw invalid_operand_exception();
#endif

    TernaryOperand12 op{0};

    if (encodeHS)
        op.direct.hs = dest ? utils::log2(rd.getHS()) : pow2Encode(rd.getHS());

    op.direct.regFile = rd.getRegFile8();
    op.direct.subRegNum = rd.getByteOffset() >> 1;
    op.direct.regNum = rd.getBase();

    return op;
}

template <bool dest, typename Tag>
static inline constexpr14 TernaryOperand12 encodeTernaryOperand12(const ExtendedReg &reg, Tag tag)
{
    auto op = encodeTernaryOperand12<dest>(reg.getBase(), tag);
    op.direct.subRegNum = reg.getMMENum();

    return op;
}

template <typename Tag>
static inline void encodeTernary512GRF(Instruction12 &i, unsigned dstReg8, unsigned src0Reg8, unsigned src1Reg8, unsigned src2Reg8, bool src1Scalar, Tag tag) {}

static inline void encodeTernary512GRF(Instruction12 &i, unsigned dstReg8, unsigned src0Reg8, unsigned src1Reg8, unsigned src2Reg8, bool src1Scalar, EncodingTagXe3p tag)
{
    i.ternaryXe3p.dstReg8 = dstReg8;
    i.ternaryXe3p.src0Reg8 = src0Reg8;
    i.ternaryXe3p.src1Reg8 = src1Reg8;
    i.ternaryXe3p.src2Reg8 = src2Reg8;
    i.ternaryXe3p.src1Scalar = src1Scalar;
}

template <typename D, typename S0, typename S1, typename S2, typename Tag>
static inline void encodeTernary512GRF(Instruction12 &i, D dst, S0 src0, S1 src1, S2 src2, Tag tag)
{
    encodeTernary512GRF(i, getHighBit(dst), getHighBit(src0), getHighBit(src1), getHighBit(src2), checkSrc1Scalar(i.opcode(), src1, dst, tag), tag);
}

static inline bool checkFlatRegion(RegData dst, RegData r) {
        int reqHS = dst.getHS() * dst.getBytes() / r.getBytes();
        return (r.getHS() == reqHS && r.getVS() == r.getWidth() * reqHS)
                || (r.getHS() == 0 && r.getWidth() == 1 && r.getVS() == reqHS);
}

template <typename Tag>
static inline bool checkSrc1Scalar(Opcode op, RegData r, RegData dst, Tag tag) { return false; }

static inline bool checkSrc1Scalar(Opcode op, RegData r, RegData dst, EncodingTagXe3p tag)
{
    switch (op) {
        case Opcode::bdpas:
        case Opcode::dpas:
        case Opcode::shfl: return false;
        default: break;
    }
    if (r.isScalar())
        return true;

#ifdef NGEN_SAFE
    if (!r.isARF()) {
        if (!checkFlatRegion(dst, r))
            throw invalid_region_exception();
    }
#endif

    return false;
}

template <typename Tag>
static inline bool checkSrc1Scalar(Opcode op, ExtendedReg r, ExtendedReg dst, Tag tag)
{
    return checkSrc1Scalar(op, r.getBase(), dst.getBase(), tag);
}

// Xe3p: a non-scalar FP source must occupy the same byte positions within a GRF as the destination.
static inline bool isXe3pFPPipeType(DataType dt)
{
    switch (dt) {
        case DataType::df:
        case DataType::f:
        case DataType::hf:
        case DataType::bf: return true;
        default: return false;
    }
}

static inline void checkXe3pFPRegionSrc(HW hw, Opcode op, const RegData &dst, const RegData &src)
{
#ifdef NGEN_SAFE
    if (hw < HW::Xe3p) return;
    switch (op) {
        case Opcode::bdpas:
        case Opcode::dpas:
        case Opcode::shfl:
        case Opcode::send:
        case Opcode::sendc:
        case Opcode::sends:
        case Opcode::sendsc: return;
        default: break;
    }
    if (dst.isNull() || dst.isIndirect() || dst.isARF()) return;
    if (src.isNull() || src.isIndirect() || src.isARF() || src.isScalar()) return;
    if (!isXe3pFPPipeType(dst.getType()) || !isXe3pFPPipeType(src.getType())) return;
    if (dst.getBytes() != src.getBytes()) return;      /* mixed-width rule not covered */

    int grfBytes = GRF::bytes(hw);
    if ((dst.getByteOffset() % grfBytes) != (src.getByteOffset() % grfBytes))
        throw invalid_region_exception();
#else
    (void) hw; (void) op; (void) dst; (void) src;
#endif
}

// Other operand types (ExtendedReg math-macro operands, Immediate, NoOperand): nothing to check.
template <typename D, typename S>
static inline void checkXe3pFPRegionSrc(HW, Opcode, const D &, const S &) {}

template <typename D>
static inline void checkXe3pFPRegion(HW, Opcode, const D &) {}

template <typename D, typename S, typename... Srcs>
static inline void checkXe3pFPRegion(HW hw, Opcode op, const D &dst, const S &src, const Srcs &... srcs)
{
    checkXe3pFPRegionSrc(hw, op, dst, src);
    checkXe3pFPRegion(hw, op, dst, srcs...);
}

static inline void encodeCommon12(Instruction12 &i, Opcode opcode, const InstructionModifier &mod, const RegData &dst, EncodingTag12 tag)
{
    Instruction12 i2;   /* separate variable to avoid gcc13 bug */
    i2.common.opcode = static_cast<unsigned>(opcode) | (mod.parts.autoSWSB << 7);
    i2.common.swsb = SWSBInfo12(mod.getSWSB(), opcode).raw();
    i2.common.execSize = mod.parts.log2ExecSize;
    i2.common.execOffset = mod.parts.chanOff;
    i2.common.flagReg = (mod.parts.flagRegNum << 1) | mod.parts.flagSubRegNum;
    i2.common.predCtrl = mod.parts.predCtrl;
    i2.common.predInv = mod.parts.predInv;
    i2.common.cmptCtrl = mod.parts.cmptCtrl;
    i2.common.debugCtrl = mod.parts.debugCtrl;
    i2.common.maskCtrl = mod.parts.maskCtrl;
    i2.common.atomicCtrl = mod.parts.threadCtrl;
    i2.common.accWrCtrl = mod.parts.accWrCtrl;
    i2.common.saturate = mod.parts.saturate;
    i.common = i2.common;
}

static inline void encodeCommon12(Instruction12 &i, Opcode opcode, const InstructionModifier &mod, const RegData &dst, EncodingTagXeHPC tag)
{
    Instruction12 i2;   /* separate variable to avoid gcc13 bug */
    i2.common.opcode = static_cast<unsigned>(opcode) | (mod.parts.autoSWSB << 7);
    i2.commonXeHPC.swsb = SWSBInfoXeHPC(mod.getSWSB(), opcode).raw();
    i2.commonXeHPC.execSize = mod.parts.log2ExecSize;
    i2.commonXeHPC.flagReg = (mod.parts.flagRegNum1 << 2) | (mod.parts.flagRegNum << 1) | mod.parts.flagSubRegNum;
    i2.commonXeHPC.execOffset = mod.parts.chanOff >> 1;
    i2.commonXeHPC.predCtrl = mod.parts.predCtrl;
    i2.common.predInv = mod.parts.predInv;
    i2.common.cmptCtrl = mod.parts.cmptCtrl;
    i2.common.debugCtrl = mod.parts.debugCtrl;
    i2.common.maskCtrl = mod.parts.maskCtrl;
    i2.common.atomicCtrl = mod.parts.threadCtrl;
    i2.commonXeHPC.dstExt = (dst.isIndirect() ? dst.getOffset() : dst.getByteOffset()) & 1;
    if (opcode == Opcode::dpas || opcode == Opcode::bdpas)
        i2.common.accWrCtrl = mod.parts.accWrCtrl;   /* {Fwd} */
    i2.common.saturate = mod.parts.saturate;
    i.common = i2.common;
}

template <typename Tag>
static inline void encodeCommon12(Instruction12 &i, Opcode opcode, const InstructionModifier &mod, const ExtendedReg &dst, Tag tag)
{
    encodeCommon12(i, opcode, mod, dst.getBase(), tag);
}

static inline unsigned encodeTernaryVS01(const RegData &rd)
{
    switch (rd.getVS()) {
        case 0: return 0;
        case 1: return 1;
        case 4: return 2;
        case 8: return 3;
        default:
#ifdef NGEN_SAFE
            if (rd.getHS() == 0)
                throw invalid_region_exception();
#endif
            return 3;
    }
}

static inline unsigned encodeTernaryVS01(const ExtendedReg &reg)
{
    return encodeTernaryVS01(reg.getBase());
}

template <typename D, typename S0, typename S1, typename S2>
static inline void encodeTernaryTypes(Instruction12 &i, D dst, S0 src0, S1 src1, S2 src2)
{
    auto dtype = getTypecode12(dst.getType());
    auto s0type = getTypecode12(src0.getType());
    auto s1type = getTypecode12(src1.getType());
    auto s2type = getTypecode12(src2.getType());

    i.ternary.execType = (dtype >> 3);
    i.ternary.dstType  = dtype;
    i.ternary.src0Type = s0type;
    i.ternary.src1Type = s1type;
    i.ternary.src2Type = s2type;

#ifdef NGEN_SAFE
    if (((dtype & s0type & s1type & s2type) ^ (dtype | s0type | s1type | s2type)) & 8)
        throw NGEN_NAMESPACE::invalid_type_exception();
#endif
}

template <typename S0, typename Tag>
static inline void encodeTernarySrc0(Instruction12 &i, S0 src0, Tag tag)
{
    auto i2 = i; /* separate variable to avoid gcc13 bug */
    i2.ternary.src0 = encodeTernaryOperand12<false>(src0, tag).bits;
    i2.ternary.src0Mods = src0.getMods();

    auto vs0 = encodeTernaryVS01(src0);

    i2.ternary.src0VS0 = vs0 & 1;
    i2.ternary.src0VS1 = vs0 >> 1;
    i = i2;
}

template <typename Tag>
static inline void encodeTernarySrc0(Instruction12 &i, const Immediate &src0, Tag tag)
{
    i.ternary.src0Imm = true;
    i.ternary.src0 = static_cast<uint64_t>(src0);
}

template <typename S1, typename Tag>
static inline void encodeTernarySrc1(Instruction12 &i, S1 src1, Tag tag)
{
    auto i2 = i; /* separate variable to avoid gcc13 bug */
    i2.ternary.src1 = encodeTernaryOperand12<false>(src1, tag).bits;
    i2.ternary.src1Mods = src1.getMods();

    auto vs1 = encodeTernaryVS01(src1);

    i2.ternary.src1VS0 = vs1 & 1;
    i2.ternary.src1VS1 = vs1 >> 1;
    i = i2;
}

template <typename S2, typename Tag>
static inline void encodeTernarySrc2(Instruction12 &i, S2 src2, Tag tag)
{
    i.ternary.src2 = encodeTernaryOperand12<false>(src2, tag).bits;
    i.ternary.src2Mods = src2.getMods();
}

template <typename Tag>
static inline void encodeTernarySrc2(Instruction12 &i, const Immediate &src2, Tag tag)
{
    i.qword[0] |= 0x800000000000; /* same as (i.ternary.src2Imm = true), update directly to avoid gcc13 bug */
    i.ternary.src2 = static_cast<uint64_t>(src2);
}

static inline void encodeSendExDesc(Instruction12 &i, uint32_t exdesc, InstructionModifier mod, int src1Length, HW hw)
{
    i.send.eot = (exdesc >> 5);
    i.send.exDesc6_10 = (exdesc >> 6);
    i.send.exDesc11_23 = (exdesc >> 11);
    i.send.exDesc24_25 = (exdesc >> 24);
    i.send.exDesc26_27 = (exdesc >> 26);
    i.send.exDesc28_31 = (exdesc >> 28);
}

static inline void encodeSendExDesc(Instruction12 &i, RegData exdesc, InstructionModifier mod, int src1Length, HW hw)
{
    bool encodeSrc1Length = mod.isExBSO();
    bool recordExBSO = mod.isExBSO();
    if (hw == HW::Xe2 && static_cast<SharedFunction>(i.send.sfid) == SharedFunction::ugm) {
        encodeSrc1Length = true;
        recordExBSO = false;
    }

#ifdef NGEN_SAFE
    // Only a0.x:ud is allowed for extended descriptor.
    if (!exdesc.isARF() || exdesc.getARFType() != ARFType::a || exdesc.getARFBase() != 0 || exdesc.getType() != DataType::ud)
        throw invalid_arf_exception();
    if (encodeSrc1Length && src1Length < 0)
        throw missing_src1_length_exception();
#endif

    i.sendIndirect.exBSO = recordExBSO;
    i.sendIndirect.exDescReg = exdesc.getOffset();
    if (encodeSrc1Length)
        i.send.exDesc6_10 = src1Length;
    i.send.exDescIsReg = true;
}

static inline void encodeSendDesc(Instruction12 &i, uint32_t desc)
{
    i.send.desc0_10 = (desc >> 0);
    i.send.desc11_19 = (desc >> 11);
    i.send.desc20_24 = (desc >> 20);
    i.send.desc25_29 = (desc >> 25);
    i.send.desc30_31 = (desc >> 30);
}

static inline void encodeSendDesc(Instruction12 &i, RegData desc)
{
#ifdef NGEN_SAFE
    // Only a0.0:ud is allowed for desc.
    if (!desc.isARF() || desc.getARFType() != ARFType::a || desc.getARFBase() != 0 || desc.getOffset() != 0)
        throw invalid_arf_exception();
#endif
    i.send.descIsReg = true;
}

static inline unsigned getHighBit(RegData r) {
    return r.isARF() ? 0 : (r.getBase() >> 8);
}

static inline unsigned getHighBit(ExtendedReg r) {
    return r.getBase().getBase() >> 8;
}

static inline unsigned getHighBit(Immediate i) {
    return 0;
}

static inline uint8_t encodeDnsclCtrl(uint8_t mode, RoundingType rnd, RegData &dst, RegData &src0, RegData &src1)
{
    auto dt = dst.getType();
    auto st = src0.getType();

#ifdef NGEN_SAFE
    if (dt != DataType::s4 && dt != DataType::e2m1 && dt != DataType::e3m0) throw invalid_type_exception();
    if (st != DataType::hf && st != DataType::bf) throw invalid_type_exception();
    if (st != src1.getType()) throw invalid_type_exception();
#endif

    dst.setOffset(dst.getByteOffset() >> 2);
    src0.setOffset(src0.getByteOffset() >> 2);
    src1.setOffset(src1.getByteOffset() >> 2);
    dst.setRegion(dst.getVS() / 8, dst.getWidth(), dst.getHS() / 8);
    src0.setRegion(src0.getVS() / 2, src0.getWidth(), src0.getHS() / 2);
    src1.setRegion(src1.getVS() / 2, src1.getWidth(), src1.getHS() / 2);
    dst.setType(DataType::ud);
    src0.setType(DataType::ud);
    src1.setType(DataType::ud);

    mode <<= 4;
    if (dt == DataType::e2m1)     mode |= 1;
    if (dt == DataType::s4)       mode |= 2;
    if (st == DataType::bf)       mode |= 4;
    if (rnd == RoundingType::rne) mode |= 8;

    return mode;
}

/*********************/
/* Decoding Routines */
/*********************/

static inline DataType decodeRegTypecode12(unsigned dt)
{
    static const DataType conversionTable[16] = {
        DataType::ub,      DataType::uw,      DataType::ud,      DataType::uq,
        DataType::b,       DataType::w,       DataType::d,       DataType::q,
        DataType::bf8,     DataType::hf,      DataType::f,       DataType::df,
        DataType::hf8,     DataType::bf,      DataType::tf32,    DataType::invalid,
    };
    return conversionTable[dt & 0xF];
}

static inline int decodeDPASTypecodeBytes12(unsigned dt)
{
    return (1 << (dt & 3));
}

inline ARFType normalizeARFType(ARFType type, HW hw)
{
    if (hw >= HW::Xe3 && type == ARFType::sp)
        type = ARFType::s;
    return type;
}

template <typename Tag>
bool Instruction12::getOperandRegion(autoswsb::DependencyRegion &region, int opNum) const
{
    using namespace autoswsb;

    constexpr bool xeHPC = !std::is_same<Tag, EncodingTag12>::value;
    constexpr bool xe3p = std::is_same<Tag, EncodingTagXe3p>::value;

    bool unaryXe3p = false;

    auto hw = region.hw;
    auto op = opcode();
    RegData rd;

    switch (op) {
        case Opcode::nop_gen12:
        case Opcode::illegal:
        case Opcode::goto_:
        case Opcode::if_:
        case Opcode::else_:
        case Opcode::endif:
        case Opcode::join:
        case Opcode::cont:
        case Opcode::break_:
        case Opcode::while_:
            return false;
        case Opcode::directive:
            switch (opNum) {
                case -1: {
                    BinaryOperand12 o;
                    o.bits = binary.dst;
                    unsigned regNum = o.direct.regNum;
                    if (xe3p) regNum |= (binaryXe3p.dstReg8 << 8);
                    region = DependencyRegion(hw, 1, GRF(regNum));
                    return true;
                }
                case 0: {
                    BinaryOperand12 o0, o1;
                    o0.bits = binary.src0;
                    o1.bits = binary.src1;
                    unsigned rn0 = o0.direct.regNum, rn1 = o1.direct.regNum;
                    if (xe3p) {
                        rn0 |= (binaryXe3p.src0Reg8 << 8);
                        rn1 |= (binaryXe3p.src1Reg8 << 8);
                    }
                    region = DependencyRegion(hw, GRF(rn0)-GRF(rn1));
                    return true;
                }
                default: return false;
            }
        case Opcode::bdpas:
        case Opcode::dpas:
        case Opcode::dpasw: {
            unsigned sdepth = 1 << dpas.sdepth;
            unsigned rcount = 1 + dpas.rcount;
            unsigned len;
            TernaryOperand12 o;
            o.bits = 0;
            unsigned regNum8 = 0;

            switch (opNum) {
                case -1: {
                    int typebytes = decodeDPASTypecodeBytes12(ternary.dstType);
                    len = (rcount * typebytes + 3) >> 2;
                    regNum8 = ternaryXe3p.dstReg8;
                    o.bits = ternary.dst;
                    break;
                }
                case 0: {
                    int typebytes = decodeDPASTypecodeBytes12(ternary.src0Type);
                    len = (rcount * typebytes + 3) >> 2;
                    regNum8 = ternaryXe3p.src0Reg8;
                    o.bits = ternary.src0;
                    break;
                }
                case 1:
                    len = sdepth;
                    o.bits = ternary.src1;
                    regNum8 = ternaryXe3p.src1Reg8;
                    break;
                case 2: {
                    if (op == Opcode::dpasw) rcount = (rcount + 1) >> 1;
                    o.bits = ternary.src2;
                    auto sr = o.direct.subRegNum;
                    regNum8 = ternaryXe3p.src2Reg8;
                    if (op == Opcode::bdpas) {
                        sr = 0;
                        rcount = 8;
                    }
                    if (xeHPC)
                        len = ((sr << 1) + sdepth * rcount * 4 + 63) >> 6;
                    else
                        len = (sr + sdepth * rcount * 4 + 31) >> 5;
                    break;
                }
                case 3: {
                    if (op != Opcode::bdpas) return false;
                    unsigned reg = bdpas.src3Reg0;
                    reg |= bdpas.src3Reg1_2 << 1;
                    reg |= bdpas.src3Reg3_6 << 3;
                    reg |= bdpas.src3Reg7_8 << 7;
                    o.direct.regFile = bdpas.src3RegFile;
                    o.direct.regNum = reg;      // low 8 bits
                    regNum8 = reg >> 8;         // bit 8 routed separately
                    len = 1;
                    break;
                }
                case 4: {
                    if (op != Opcode::bdpas) return false;
                    unsigned reg = bdpas.src4Reg0_3;
                    reg |= bdpas.src4Reg4_8 << 4;
                    o.direct.regFile = bdpas.src4RegFile;
                    o.direct.regNum = reg;      // low 8 bits
                    regNum8 = reg >> 8;         // bit 8 routed separately
                    len = 1;
                    break;
                }
                default: return false;
            }

            if (o.direct.regFile == RegFileARF) return false;    // null operand

            unsigned regNum = o.direct.regNum;
            if (xe3p) regNum |= (regNum8 << 8);
            region = DependencyRegion(hw, GRFRange(regNum, len));
            return true;
        }
        case Opcode::send:
        case Opcode::sendc: {
            int base = 0, len = 0;
            if (send.src0RegFile == RegFileARF && hw >= HW::Xe3) switch (opNum) {
                case 0:
                    region = DependencyRegion(hw);
                    return true;
                    break;
                case 1: {
                    /* report s0 dependency as if it came from src1 */
                    region = DependencyRegion(hw, send.desc25_29 & 0xF, ScalarRegister(0)[send.exDesc6_10 << 1](1));
                    return true;
                    break;
                }
                default: break;
            }
            switch (opNum) {
                case -1:
                    if (send.dstRegFile == RegFileARF) return false;
                    base = send.dstReg;
                    len = send.descIsReg ? -1 : (int)send.desc20_24;
                    if (len == 31) len++;
                    break;
                case 0:
                    if (send.src0RegFile == RegFileARF) return false;
                    base = send.src0Reg;
                    len = send.descIsReg ? -1 : (int)(send.desc25_29 & 0xF);
                    break;
                case 1:
                    if (send.src1RegFile == RegFileARF) return false;
                    base = send.src1Reg;
                    len = send.exDescIsReg ? -1 : (int)send.exDesc6_10;
                    break;
                case 2:
                case 3: // TODO: May need to track indirect acc usage
                default: return false;
            }

            if (len == 0)
                return false;
            else if (len == -1) {
                region = DependencyRegion(hw);
                region.base = base;
            } else
                region = DependencyRegion(hw, GRFRange(base, len));
            return true;
        }
        case Opcode::sendg:
        case Opcode::sendgc: {
            if (send.src0RegFile == RegFileARF && (sendg.src0Reg >> 4) == 0x6) switch (opNum) {
                case 0:
                    region = DependencyRegion(hw);
                    return true;
                case 1: {
                    /* report s0 dependency as if it came from src1 */
                    region = DependencyRegion(hw, sendg.src0Len, ScalarRegister(0)[(send.src0Reg & 0xF) << 1](1));
                    return true;
                }
                default: break;
            }
            switch (opNum) {
                case -1: {
                    if (sendg.dstRegFile == RegFileARF) return false;
                    int dstLen = getSendgDesc().dstLen(hw, 1 << common.execSize, static_cast<SharedFunction>(sendg.sfid));
                    if (dstLen == -1) {
                        region = DependencyRegion(hw);
                        region.base = sendg.dstReg;
                    } else
                        region = DependencyRegion(hw, GRFRange(sendg.dstReg, dstLen));
                    break;
                }
                case 0:
                    if (sendg.src0RegFile == RegFileARF) return false;
                    region = DependencyRegion(hw, GRFRange(sendg.src0Reg, sendg.src0Len));
                    break;
                case 1:
                    if (sendg.src1RegFile == RegFileARF) return false;
                    region = DependencyRegion(hw, GRFRange(sendg.src1Reg, sendg.src1Len));
                    break;
                // ind0/ind1 are indirect descriptor reads from scalar register s0.
                case 2:
                    if (!sendg.ind0Present) return false;
                    region = DependencyRegion(hw, 1, ScalarRegister(0).uq(sendg.ind0)(1));
                    return true;
                case 3:
                    if (!sendg.ind1Present) return false;
                    region = DependencyRegion(hw, 1, ScalarRegister(0).uq(sendg.ind1_desc42_46)(1));
                    return true;
                default: return false;
            }
            return true;
        }
        case Opcode::sendgx:
        case Opcode::sendgxc: {
            int regNum = 0, len = -1;
            switch (opNum) {
                case -1: {
                    regNum = sendg.dstReg | (sendgx.dstReg8 << 8);
                    len = getSendgDesc().dstLen(hw, 1 << common.execSize, static_cast<SharedFunction>(sendg.sfid));
                    break;
                }
                case 0:
                    regNum = sendg.src0Reg | (sendgx.src0Reg8 << 8);
                    len = sendg.src0Len;
                    break;
                case 1:
                    regNum = sendg.src1Reg | (sendgx.src1Reg8 << 8);
                    len = sendg.src1Len;
                    break;
                // ind0/ind1 are indirect descriptor reads from scalar register s0.
                case 2:
                    if (!sendg.ind0Present) return false;
                    region = DependencyRegion(hw, 1, ScalarRegister(0).uq(sendg.ind0)(1));
                    return true;
                case 3:
                    if (!sendg.ind1Present) return false;
                    region = DependencyRegion(hw, 1, ScalarRegister(0).uq(sendg.ind1_desc42_46)(1));
                    return true;
                default: return false;
            }
            if (regNum == 0x1FF)
                return false;
            else if (len == -1) {
                region = DependencyRegion(hw);
                region.base = regNum;
            } else
                region = DependencyRegion(hw, GRFRange(regNum, len));
            return true;
        }
        case Opcode::dp4a:
        case Opcode::add3:
        case Opcode::bfn:
        case Opcode::bfe_gen12:
        case Opcode::bfi2_gen12:
        case Opcode::csel_gen12:
        case Opcode::dnscl:
        case Opcode::mad:
        case Opcode::madm: {  // ternary
            TernaryOperand12 o;
            unsigned dt = 0, vs = 0;
            unsigned regNum8 = 0;
            switch (opNum) {
                case -1:
                    o.bits = ternary.dst;
                    dt = ternary.dstType;
                    regNum8 = ternaryXe3p.dstReg8;
                    break;
                case 0:
                    if (ternary.src0Imm) return false;
                    o.bits = ternary.src0;
                    dt = ternary.src0Type;
                    vs = ternary.src0VS0 + (ternary.src0VS1 * 3);
                    regNum8 = ternaryXe3p.src0Reg8;
                    break;
                case 1:
                    o.bits = ternary.src1;
                    dt = ternary.src1Type;
                    vs = ternary.src1VS0 + (ternary.src1VS1 * 3);
                    regNum8 = ternaryXe3p.src1Reg8;
                    break;
                case 2:
                    if (ternary.src2Imm) return false;
                    o.bits = ternary.src2;
                    dt = ternary.src2Type;
                    regNum8 = ternaryXe3p.src2Reg8;
                    break;
                default: return false;
            }
            dt |= (ternary.execType << 3);
            if (op == Opcode::madm) o.direct.subRegNum = 0;
            unsigned regNum = o.direct.regNum;
            if (xe3p) regNum |= (regNum8 << 8);
            auto base = GRF(regNum).retype(decodeRegTypecode12(dt));
            auto sr = o.direct.subRegNum;
            if (xeHPC) sr <<= 1;
            auto sub = base[sr / base.getBytes()];
            auto hs = (1 << o.direct.hs);
            if (opNum >= 0) hs >>= 1;
            if (xe3p && opNum == 1) {
                hs = 0;
                if (!ternaryXe3p.src1Scalar) {
                    auto dbytes = getBytes(decodeRegTypecode12((ternary.execType << 3) | ternary.dstType));
                    TernaryOperand12 odst;
                    odst.bits = ternary.dst;
                    hs = std::max(1, (dbytes << odst.direct.hs) / base.getBytes());
                }
                vs = 0;
            }
            if (opNum < 0)
                rd = sub(hs, 1, 0);
            else if (opNum == 2)
                rd = sub(hs);
            else
                rd = sub((1 << vs) >> 1, hs);

            if (o.direct.regFile == RegFileARF) {
                rd.setRegFile(RegFileARF);
                if (!autoswsb::trackableARF(normalizeARFType(rd.getARFType(), hw)))
                    return false;
            }
            break;
        }
        case Opcode::mov_gen12:
        case Opcode::not_gen12:
        case Opcode::frc:
        case Opcode::rndd:
        case Opcode::rndu:
        case Opcode::rnde:
        case Opcode::rndz:
        case Opcode::lzd:
        case Opcode::fbh:
        case Opcode::fbl:
        case Opcode::cbit:
        case Opcode::math:
            unaryXe3p = true;
            /* fall through */
        default: {    // unary/binary
            BinaryOperand12 o;
            unsigned dt;
            unsigned regNum8 = 0;

            if (op == Opcode::math) switch (static_cast<MathFunction>(binary.cmod)) {
                case MathFunction::invm:
                case MathFunction::rsqtm: unaryXe3p = false; break;
                default: break;
            }
            switch (opNum) {
                case -1:
                    o.bits = binary.dst;
                    dt = binary.dstType;
                    if (unaryXe3p)
                        regNum8 = binary.src0Imm ? unaryXe3pImm.dstReg8 : binaryXe3p.dstReg8;
                    else
                        regNum8 = binary.src1Imm ? binaryXe3pImm.dstReg8 : binaryXe3p.dstReg8;
                    break;
                case 0:
                    if (!xe3p || unaryXe3p)
                    if (binary.src0Imm) return false;
                    o.bits = binary.src0;
                    dt = binary.src0Type;
                    regNum8 = (!unaryXe3p && binary.src1Imm) ? binaryXe3pImm.src0Reg8
                                                             : binaryXe3p.src0Reg8;
                    if (xe3p && (!unaryXe3p || op == Opcode::math))
                        dt |= (binaryXe3p.execType << 3);
                    break;
                case 1:
                    if (binary.src0Imm || binary.src1Imm) return false;
                    o.bits = binary.src1;
                    dt = binary.src1Type;
                    regNum8 = binaryXe3p.src1Reg8;
                    break;
                default: return false;
            }
            if (o.direct.addrMode) { region = DependencyRegion(hw); return true; } // indirect
            if (isMathMacro())
                o.direct.subRegNum = 0;
            auto sr = xeHPC ? ((o.direct.subRegNum << 1) | o.directXeHPC.subRegNum0)
                            : o.direct.subRegNum;
            auto regNum = o.direct.regNum;
            if (xe3p) regNum |= (regNum8 << 8);
            auto base = GRF(regNum).retype(decodeRegTypecode12(dt));
            auto sub = base[sr / base.getBytes()];
            auto hs = (1 << o.direct.hs) >> 1;
            auto vs = xeHPC ? o.directXeHPC.vs : o.direct.vs;
            if (xe3p && opNum == 1) {
                hs = 0;
                if (!binaryXe3p.src1Scalar) {
                    auto dbytes = getBytes(decodeRegTypecode12(binary.dstType));
                    BinaryOperand12 odst;
                    odst.bits = binary.dst;
                    hs = std::max(1, (dbytes << (odst.direct.hs - 1)) / base.getBytes());
                }
                rd = sub(hs);
            } else
            if (opNum < 0)
                rd = sub(hs, 1, 0);
            else
                rd = sub((1 << vs) >> 1, 1 << o.direct.width, hs);

            if (o.direct.regFile == RegFileARF) {
                rd.setRegFile(RegFileARF);
                if (!autoswsb::trackableARF(normalizeARFType(rd.getARFType(), hw)))
                    return false;
            }
            break;
        }
    }

    auto esize = 1 << ((hw >= HW::XeHPC) ? commonXeHPC.execSize : common.execSize);
    rd.fixup(hw, esize, 0, DataType::invalid, opNum, 2);
    region = DependencyRegion(hw, esize, rd);
    if (op == Opcode::mullh) region.duplicateLH();
    return true;
}

template <typename Tag>
bool Instruction12::getCModDepRegion(autoswsb::DependencyRegion &region) const
{
    /* Skip instructions with no cmod fields */
    auto op = opcode();
    switch (op) {
        case Opcode::directive:
        case Opcode::sync:
        case Opcode::illegal:
        case Opcode::bfn:
        case Opcode::send:
        case Opcode::sendc:
        case Opcode::dpas:
        case Opcode::dpasw:
        case Opcode::math:
        case Opcode::bdpas:
        case Opcode::sendg:
        case Opcode::sendgc:
        case Opcode::sendgx:
        case Opcode::sendgxc:
            return false;
        default:
            if (isBranch(op))
                return false;
            break;
    }

    auto cmod = static_cast<ConditionModifier>(binary.cmod);
    if (cmod == ConditionModifier::none)
        return false;

    /* Decode cmod flag */
    constexpr bool xeHPC = !std::is_same<Tag, EncodingTag12>::value;
    unsigned flagn = xeHPC ? commonXeHPC.flagReg : common.flagReg;
    unsigned lg2ES = xeHPC ? commonXeHPC.execSize : common.execSize;

    auto flag = FlagRegister::createFromIndex(flagn);
    if (lg2ES >= 5)
        flag.setType(DataType::ud);
    region = autoswsb::DependencyRegion(region.hw, 1, flag);

    return true;
}

unsigned Instruction12::srcTypecode(int opNum) const
{
    auto op = opcode();

    switch (op) {
        case Opcode::nop_gen12:
        case Opcode::illegal:
        case Opcode::send:
        case Opcode::sendc:
        case Opcode::dp4a:
            return 0;
        case Opcode::dpas:
        case Opcode::dpasw:
            // This method is only used for checking for long pipe types.
            return 0;
        case Opcode::add3:
        case Opcode::bfn:
        case Opcode::bfe_gen12:
        case Opcode::bfi2_gen12:
        case Opcode::csel_gen12:
        case Opcode::mad:
        case Opcode::madm: // ternary
            switch (opNum) {
                case 0: return ternary.src0Type | (ternary.execType << 3);
                case 1: return ternary.src1Type | (ternary.execType << 3);
                case 2: return ternary.src2Type | (ternary.execType << 3);
                default: return 0;
            }
        default: // unary/binary
            switch (opNum) {
                case 0: return binary.src0Type;
                case 1: return binary.src1Type;
                default: return 0;
            }
    }

    return 0;
}

bool Instruction12::getImm32(uint32_t &imm) const
{
    // Only need to support sync.allrd/wr.
    if (binary.src0Imm)
        imm = imm32.value;
    return binary.src0Imm;
}

bool Instruction12::getSendDesc(MessageDescriptor &desc) const
{
    if (!send.descIsReg)
        desc.all = send.desc0_10 | (send.desc11_19 << 11) | (send.desc20_24 << 20)
                                 | (send.desc25_29 << 25) | (send.desc30_31 << 30);
    return !send.descIsReg;
}

int Instruction12::getFencedepJIP() const
{
    return int32_t(imm32.value) / sizeof(Instruction12);
}

SendgMessageDescriptor Instruction12::getSendgDesc() const
{
    SendgMessageDescriptor desc;
    desc.all =  uint64_t(sendg.desc0_15)
             | (uint64_t(sendg.desc16_27) << 16)
             | (uint64_t(sendg.desc28_29) << 28)
             | (uint64_t(sendg.desc30_31) << 30)
             | (uint64_t(sendg.desc32_39) << 32)
             | (uint64_t(sendg.desc40_41) << 40);
    if (!sendg.ind1Present)
        desc.all |= (uint64_t(sendg.ind1_desc42_46) << 42);
    return desc;
}

bool Instruction12::getARFType(ARFType &arfType, int opNum, HW hw) const
{
    if (opNum > 1) return false;

    if (isSend(opcode()))
        return false;

    // Only need to support unary/binary, for detecting ce/cr/sr usage.
    switch (opcode()) {
        case Opcode::nop:
        case Opcode::illegal:
        case Opcode::bfe:
        case Opcode::bfi2:
        case Opcode::csel:
        case Opcode::mad:
        case Opcode::madm:
        case Opcode::dp4a:
        case Opcode::add3:
        case Opcode::bfn:
        case Opcode::dpas:
        case Opcode::dpasw:
            return false;
        default: {
            BinaryOperand12 o;
            switch (opNum) {
                case -1:
                    o.bits = binary.dst;
                    break;
                case 0:
                    if (binary.src0Imm) return false;
                    o.bits = binary.src0;
                    break;
                case 1:
                    if (binary.src0Imm || binary.src1Imm) return false;
                    o.bits = binary.src1;
                    break;
                default: return false;
            }
            if (o.direct.addrMode) return false;
            if (o.direct.regFile != RegFileARF) return false;
            arfType = normalizeARFType(static_cast<ARFType>(o.direct.regNum >> 4), hw);
            return true;
        }
    }
}

autoswsb::DestinationMask Instruction12::destinations(int &jip, int &uip) const
{
    using namespace autoswsb;

    if (!isBranch(opcode())) {
        if (eot() && !predicated())
            return DestNone;
        return DestNextIP;
    }

    DestinationMask mask = DestNextIP;
    switch (opcode()) {
        case Opcode::ret:
        case Opcode::endif:
        case Opcode::while_:
        case Opcode::call:
        case Opcode::calla:
        case Opcode::join:
        case Opcode::jmpi:
        case Opcode::brd:
            mask = binary.src0Imm ? (DestNextIP | DestJIP) : DestUnknown; break;
        case Opcode::goto_:
        case Opcode::if_:
        case Opcode::else_:
        case Opcode::break_:
        case Opcode::cont:
        case Opcode::halt:
        case Opcode::brc:
            mask = binary.src0Imm ? (DestNextIP | DestJIP | DestUIP) : DestUnknown; break;
        default: break;
    }

    if ((opcode() == Opcode::jmpi) && !predicated())
        mask &= ~DestNextIP;

    if (mask & DestJIP) jip = branches.jip / sizeof(Instruction12);
    if (mask & DestUIP) uip = branches.uip / sizeof(Instruction12);

    return mask;
}

}
#endif
