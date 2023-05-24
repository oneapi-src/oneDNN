/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

// Gen12 binary encoding.

struct EncodingTag12 {};
struct EncodingTagXeHPC {};
template <HW hw> struct EncodingTag12Dispatch { using tag = EncodingTag12; };
template <> struct EncodingTag12Dispatch<HW::XeHPC> { using tag = EncodingTagXeHPC; };

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
        if (info.hasDist() && info.hasToken()) {
            combined.sbid = info.parts.token;
            combined.dist = info.parts.dist;
            combined.combined = true;
        } else if (info.hasDist()) {
            combined.combined = false;
            uint8_t pipeMap[8] = {0, 1, 2, 3, 10, 0, 0, 0};
            pipeline.dist = info.parts.dist;
            pipeline.pipe = pipeMap[info.parts.pipe & 7];
        } else if (info.hasToken()) {
            combined.combined = false;
            combined.sbid = info.parts.token;
            scoreboard.mode = 1 + info.tokenMode();
        } else
            all = 0;
    }

    SWSBInfo decode(Opcode op, unsigned dstTypecode) const {
        if (combined.combined) {
            bool vl = trackedByToken(HW::Gen12LP, op, dstTypecode);
            auto pipe = (op == Opcode::send || op == Opcode::sendc) ? Pipe::A : Pipe::Default;
            return SWSBInfo(combined.sbid, vl, true) | SWSBInfo(pipe, combined.dist);
        } else if (isPipeline()) {
            static const Pipe pipeMap[4] = {Pipe::Default, Pipe::A, Pipe::F, Pipe::I};
            auto pipe = (pipeline.pipe == 10) ? Pipe::L : pipeMap[pipeline.pipe & 3];
            return SWSBInfo(pipe, pipeline.dist);
        } else
            return SWSBInfo(scoreboard.sbid, scoreboard.mode != 2, scoreboard.mode != 3);
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

    static constexpr14 unsigned combinedMode(SWSBInfo info, Opcode op) {
        auto pipe = info.getPipe();
        if (info.parts.src && info.parts.dst)
            return (pipe == Pipe::F) ? 2 : (pipe == Pipe::I) ? 3 : 1;
        if (info.parts.src) return 2;
        if (info.parts.dst) return (pipe == Pipe::A || op == Opcode::dpas) ? 3 : 1;
        return 0;
    }

public:
    constexpr SWSBInfoXeHPC() : all{0} {}

    SWSBInfoXeHPC(SWSBInfo info, Opcode op) {
        if (info.hasDist() && info.hasToken()) {
            combined.sbid = info.parts.token;
            combined.dist = info.parts.dist;
            combined.mode = combinedMode(info, op);
        } else if (info.hasDist()) {
            pipeline.dist = info.parts.dist;
            pipeline.pipe = info.parts.pipe;
            pipeline.sb = false;
            pipeline.mode = 0;
        } else if (info.hasToken()) {
            scoreboard.sbid = info.parts.token;
            scoreboard.type = info.tokenMode() - 1;
            scoreboard.sb = true;
            scoreboard.mode = 0;
        } else if (info.parts.noacc)
            all = 0xF0;
        else
            all = 0;
    }

    SWSBInfo decode(Opcode op) const {
        if (all == 0xF0)
            return SWSBInfo::createNoAccSBSet();

        auto result = SWSBInfo(pipe(op), dist());
        if (combined.mode) {
            bool src, dst;
            if (op == Opcode::send || op == Opcode::sendc)
                src = dst = true;
            else if (op == Opcode::dpas) {
                src = (combined.mode <= 2);
                dst = combined.mode & 1;
            } else {
                dst = combined.mode & 1;
                src = !dst;
            }
            result = result | SWSBInfo(combined.sbid, src, dst);
        } else if (scoreboard.sb)
            result = result | SWSBInfo(scoreboard.sbid, scoreboard.type != 0, scoreboard.type != 1);

        return result;
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
            if (op == Opcode::send || op == Opcode::sendc)
                return (combined.mode == 1) ? Pipe::A : (combined.mode == 2) ? Pipe::F : Pipe::I;
            if (op == Opcode::dpas)
                return Pipe::Default;
            return (combined.mode == 3) ? Pipe::A : Pipe::Default;
        } else if (!scoreboard.sb) {
            const Pipe table[8] = {Pipe::Default, Pipe::A, Pipe::F, Pipe::I, Pipe::L, Pipe::M, Pipe::A, Pipe::A};
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
            unsigned dst : 16;              // TernaryOperand12 or immediate
            //
            unsigned src0 : 16;
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
            unsigned : 8;
            unsigned exDescReg : 3;
            unsigned : 21;
            unsigned : 32;
            unsigned : 32;
        } sendIndirect;
        struct {
            unsigned : 32;                  // common
            unsigned : 1;
            unsigned branchCtrl : 1;
            unsigned : 30;
            int32_t uip;
            int32_t jip;
        } branches;
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
    bool eot() const              { return (opcode() == Opcode::send || opcode() == Opcode::sendc) && send.eot; }
    bool predicated() const       { return !common.maskCtrl || (static_cast<PredCtrl>(common.predCtrl) != PredCtrl::None); }
    bool atomic() const           { return common.atomicCtrl; }
    unsigned dstTypecode() const  { return binary.dstType; }
    unsigned src0Typecode() const { return srcTypecode(0); }
    unsigned src1Typecode() const { return srcTypecode(1); }
    void shiftJIP(int32_t shift) { branches.jip += shift * sizeof(Instruction12); }
    void shiftUIP(int32_t shift) { branches.uip += shift * sizeof(Instruction12); }

    inline autoswsb::DestinationMask destinations(int &jip, int &uip) const;
    template <bool xeHPC = false>
    inline bool getOperandRegion(autoswsb::DependencyRegion &region, int opNum) const;
    inline bool getImm32(uint32_t &imm) const;
    inline bool getSendDesc(MessageDescriptor &desc) const;
    inline bool getARFType(ARFType &arfType, int opNum) const;

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

    template <bool xeHPC = true>
    bool getOperandRegion(autoswsb::DependencyRegion &region, int opNum) const {
        return Instruction12::getOperandRegion<true>(region, opNum);
    }
};

static_assert(sizeof(InstructionXeHPC) == 16, "Internal error: InstructionXeHPC has been padded by the compiler.");

// Encoding routines.

static inline unsigned getTypecode12(DataType type)
{
    static const uint8_t conversionTable[32] = {2,6,1,5,0,4,11,10,3,7,9,13,8,0,4,8,
                                                14,2,2,2,2,2,2,2,2,2,2,2,0,4,0,4};
    return conversionTable[static_cast<unsigned>(type) & 0x1F];
}

static inline unsigned encodeSubBytePrecision12(DataType type)
{
    static const uint8_t conversionTable[32] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                                0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,2};
    return conversionTable[static_cast<unsigned>(type) & 0x1F];
}

static inline unsigned pow2Encode(unsigned x)
{
    return (x == 0) ? 0 : (1 + utils::log2(x));
}

template <bool dest, bool encodeHS = true>
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
        if (!dest)
            op.indirect.vs = (rd.isVxIndirect()) ? 0xFFFF : pow2Encode(rd.getVS());
    } else {
        op.direct.regFile = getRegFile(rd);
        op.direct.subRegNum = rd.getByteOffset();
        op.direct.regNum = rd.getBase();
        op.direct.addrMode = 0;
        if (!dest)
            op.direct.vs = pow2Encode(rd.getVS());
    }

    if (encodeHS)
        op.direct.hs = pow2Encode(rd.getHS());

    if (!dest) op.direct.width = utils::log2(rd.getWidth());

    return op;
}

template <bool dest, bool encodeHS = true>
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
        if (!dest) {
            op.indirect.vs = (rd.isVxIndirect()) ? 0xFFFF : pow2Encode(rd.getVS());
            op.indirectXeHPC.addrOff0 = (rd.getOffset() & 1);
        }
    } else {
        op.direct.regFile = getRegFile(rd);
        op.direct.subRegNum = (rd.getByteOffset() >> 1);
        op.direct.regNum = rd.getBase();
        op.direct.addrMode = 0;
        if (!dest) {
            op.directXeHPC.vs = pow2Encode(rd.getVS());
            op.directXeHPC.subRegNum0 = rd.getByteOffset() & 1;
        }
    }

    if (encodeHS)
        op.direct.hs = pow2Encode(rd.getHS());

    if (!dest) op.direct.width = utils::log2(rd.getWidth());

    return op;
}

template <bool dest, typename Tag>
static inline constexpr14 BinaryOperand12 encodeBinaryOperand12(const ExtendedReg &reg, Tag tag)
{
    auto op = encodeBinaryOperand12<dest>(reg.getBase(), tag);
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

    op.direct.regFile = getRegFile(rd);
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

    op.direct.regFile = getRegFile(rd);
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

static inline void encodeCommon12(Instruction12 &i, Opcode opcode, const InstructionModifier &mod, const RegData &dst, EncodingTag12 tag)
{
    i.common.opcode = static_cast<unsigned>(opcode) | (mod.parts.autoSWSB << 7);
    i.common.swsb = SWSBInfo12(mod.getSWSB(), opcode).raw();
    i.common.execSize = mod.parts.eSizeField;
    i.common.execOffset = mod.parts.chanOff;
    i.common.flagReg = (mod.parts.flagRegNum << 1) | mod.parts.flagSubRegNum;
    i.common.predCtrl = mod.parts.predCtrl;
    i.common.predInv = mod.parts.predInv;
    i.common.cmptCtrl = mod.parts.cmptCtrl;
    i.common.debugCtrl = mod.parts.debugCtrl;
    i.common.maskCtrl = mod.parts.maskCtrl;
    i.common.atomicCtrl = mod.parts.threadCtrl;
    i.common.accWrCtrl = mod.parts.accWrCtrl;
    i.common.saturate = mod.parts.saturate;
}

static inline void encodeCommon12(Instruction12 &i, Opcode opcode, const InstructionModifier &mod, const RegData &dst, EncodingTagXeHPC tag)
{
    i.common.opcode = static_cast<unsigned>(opcode) | (mod.parts.autoSWSB << 7);
    i.commonXeHPC.swsb = SWSBInfoXeHPC(mod.getSWSB(), opcode).raw();
    i.commonXeHPC.execSize = mod.parts.eSizeField;
    i.commonXeHPC.flagReg = (mod.parts.flagRegNum1 << 2) | (mod.parts.flagRegNum << 1) | mod.parts.flagSubRegNum;
    i.commonXeHPC.execOffset = mod.parts.chanOff >> 1;
    i.commonXeHPC.predCtrl = mod.parts.predCtrl;
    i.common.predInv = mod.parts.predInv;
    i.common.cmptCtrl = mod.parts.cmptCtrl;
    i.common.debugCtrl = mod.parts.debugCtrl;
    i.common.maskCtrl = mod.parts.maskCtrl;
    i.common.atomicCtrl = mod.parts.threadCtrl;
    i.commonXeHPC.dstExt = (dst.isIndirect() ? dst.getOffset() : dst.getByteOffset()) & 1;
    i.common.saturate = mod.parts.saturate;
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
        throw ngen::invalid_type_exception();
#endif
}

template <typename S0, typename Tag>
static inline void encodeTernarySrc0(Instruction12 &i, S0 src0, Tag tag)
{
    i.ternary.src0 = encodeTernaryOperand12<false>(src0, tag).bits;
    i.ternary.src0Mods = src0.getMods();

    auto vs0 = encodeTernaryVS01(src0);

    i.ternary.src0VS0 = vs0;
    i.ternary.src0VS1 = vs0 >> 1;
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
    i.ternary.src1 = encodeTernaryOperand12<false>(src1, tag).bits;
    i.ternary.src1Mods = src1.getMods();

    auto vs1 = encodeTernaryVS01(src1);

    i.ternary.src1VS0 = vs1;
    i.ternary.src1VS1 = vs1 >> 1;
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
    i.ternary.src2Imm = true;
    i.ternary.src2 = static_cast<uint64_t>(src2);
}

static inline void encodeSendExDesc(Instruction12 &i, uint32_t exdesc)
{
    i.send.eot = (exdesc >> 5);
    i.send.exDesc6_10 = (exdesc >> 6);
    i.send.exDesc11_23 = (exdesc >> 11);
    i.send.exDesc24_25 = (exdesc >> 24);
    i.send.exDesc26_27 = (exdesc >> 26);
    i.send.exDesc28_31 = (exdesc >> 28);
}

static inline void encodeSendExDesc(Instruction12 &i, RegData exdesc)
{
#ifdef NGEN_SAFE
    // Only a0.x:ud is allowed for extended descriptor.
    if (!exdesc.isARF() || exdesc.getARFType() != ARFType::a || exdesc.getARFBase() != 0 || exdesc.getType() != DataType::ud)
        throw invalid_arf_exception();
#endif
    i.sendIndirect.exDescReg = exdesc.getOffset();
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

/*********************/
/* Decoding Routines */
/*********************/

static inline DataType decodeRegTypecode12(unsigned dt)
{
    static const DataType conversionTable[16] = {
        DataType::ub,      DataType::uw,      DataType::ud,      DataType::uq,
        DataType::b,       DataType::w,       DataType::d,       DataType::q,
        DataType::invalid, DataType::hf,      DataType::f,       DataType::df,
        DataType::invalid, DataType::bf,      DataType::tf32,    DataType::bf8
    };
    return conversionTable[dt & 0xF];
}

static inline int decodeDPASTypecodeBytes12(unsigned dt)
{
    return (1 << (dt & 3));
}

template <bool xeHPC>
bool Instruction12::getOperandRegion(autoswsb::DependencyRegion &region, int opNum) const
{
    using namespace autoswsb;

    auto hw = region.hw;
    auto op = opcode();
    RegData rd;

    switch (op) {
        case Opcode::nop_gen12:
        case Opcode::illegal:
            return false;
        case Opcode::wrdep:
            if (opNum != 0) return false;
            BinaryOperand12 o0, o1;
            o0.bits = binary.src0;
            o1.bits = binary.src1;
            region = DependencyRegion(hw, GRF(o0.direct.regNum)-GRF(o1.direct.regNum));
            return true;
        case Opcode::dpas:
        case Opcode::dpasw: {
            unsigned sdepth = 1 << dpas.sdepth;
            unsigned rcount = 1 + dpas.rcount;
            unsigned len;
            TernaryOperand12 o;

            switch (opNum) {
                case -1: {
                    int typebytes = decodeDPASTypecodeBytes12(ternary.dstType);
                    len = (rcount * typebytes + 3) >> 2;
                    o.bits = ternary.dst;
                    break;
                }
                case 0: {
                    int typebytes = decodeDPASTypecodeBytes12(ternary.src0Type);
                    len = (rcount * typebytes + 3) >> 2;
                    o.bits = ternary.src0;
                    break;
                }
                case 1:  len = sdepth; o.bits = ternary.src1; break;
                case 2: {
                    if (op == Opcode::dpasw) rcount = (rcount + 1) >> 1;
                    o.bits = ternary.src2;
                    auto sr = o.direct.subRegNum;
                    if (xeHPC)
                        len = ((sr << 1) + sdepth * rcount * 4 + 63) >> 6;
                    else
                        len = (sr + sdepth * rcount * 4 + 31) >> 5;
                    break;
                }
                default: return false;
            }

            region = DependencyRegion(hw, GRFRange(o.direct.regNum, len));
            return true;
        }
        case Opcode::send:
        case Opcode::sendc: {
            int base = 0, len = 0;
            switch (opNum) {
                case -1:
                    if (send.dstRegFile == RegFileARF) return false;
                    base = send.dstReg;
                    len = send.descIsReg ? -1 : send.desc20_24;
                    if (len == 31) len++;
                    break;
                case 0:
                    if (send.src0RegFile == RegFileARF) return false;
                    base = send.src0Reg;
                    len = send.descIsReg ? -1 : (send.desc25_29 & 0xF);
                    break;
                case 1:
                    if (send.src1RegFile == RegFileARF) return false;
                    base = send.src1Reg;
                    len = send.exDescIsReg ? -1 : send.exDesc6_10;
                    break;
                case 2:
                case 3: // TODO: May need to track indirect acc usage
                default: return false;
            }

            if (len == 0)
                return false;
            else if (len == -1)
                region = DependencyRegion(hw);
            else
                region = DependencyRegion(hw, GRFRange(base, len));
            return true;
        }
        case Opcode::dp4a:
        case Opcode::add3:
        case Opcode::bfn:
        case Opcode::bfe_gen12:
        case Opcode::bfi2_gen12:
        case Opcode::csel_gen12:
        case Opcode::mad:
        case Opcode::madm: {  // ternary
            TernaryOperand12 o;
            unsigned dt = 0, vs = 0;
            switch (opNum) {
                case -1:
                    o.bits = ternary.dst;
                    dt = ternary.dstType;
                    break;
                case 0:
                    if (ternary.src0Imm) return false;
                    o.bits = ternary.src0;
                    dt = ternary.src0Type;
                    vs = ternary.src0VS0 + (ternary.src0VS1 * 3);
                    break;
                case 1:
                    o.bits = ternary.src1;
                    dt = ternary.src1Type;
                    vs = ternary.src1VS0 + (ternary.src1VS1 * 3);
                    break;
                case 2:
                    if (ternary.src2Imm) return false;
                    o.bits = ternary.src2;
                    dt = ternary.src2Type;
                    break;
                default: return false;
            }
            dt |= (ternary.execType << 3);
            if (op == Opcode::madm) o.direct.subRegNum = 0;
            auto base = GRF(o.direct.regNum).retype(decodeRegTypecode12(dt));
            auto sr = o.direct.subRegNum;
            if (xeHPC) sr <<= 1;
            auto sub = base[sr / getBytes(base.getType())];
            auto hs = (1 << o.direct.hs);
            if (opNum >= 0) hs >>= 1;
            if ((opNum < 0) || (opNum == 2))
                rd = sub(hs);
            else
                rd = sub((1 << vs) >> 1, hs);

            if (o.direct.regFile == RegFileARF) {
                rd.setARF(true);
                if (!autoswsb::trackableARF(rd.getARFType()))
                    return false;
            }
            break;
        }
        default: {    // unary/binary
            BinaryOperand12 o;
            unsigned dt;
            switch (opNum) {
                case -1:
                    o.bits = binary.dst;
                    dt = binary.dstType;
                    break;
                case 0:
                    if (binary.src0Imm) return false;
                    o.bits = binary.src0;
                    dt = binary.src0Type;
                    break;
                case 1:
                    if (binary.src0Imm || binary.src1Imm) return false;
                    o.bits = binary.src1;
                    dt = binary.src1Type;
                    break;
                default: return false;
            }
            if (o.direct.addrMode) { region = DependencyRegion(hw); return true; } // indirect
            if (isMathMacro())
                o.direct.subRegNum = 0;
            auto sr = xeHPC ? ((o.direct.subRegNum << 1) | o.directXeHPC.subRegNum0)
                            : o.direct.subRegNum;
            auto vs = xeHPC ? o.directXeHPC.vs : o.direct.vs;
            auto base = GRF(o.direct.regNum).retype(decodeRegTypecode12(dt));
            auto sub = base[sr / getBytes(base.getType())];
            auto hs = (1 << o.direct.hs) >> 1;
            if (opNum < 0)
                rd = sub(hs);
            else
                rd = sub((1 << vs) >> 1, 1 << o.direct.width, hs);

            if (o.direct.regFile == RegFileARF) {
                rd.setARF(true);
                if (!autoswsb::trackableARF(rd.getARFType()))
                    return false;
            }
            break;
        }
    }

    auto esize = 1 << ((hw >= HW::XeHPC) ? commonXeHPC.execSize : common.execSize);
    rd.fixup(hw, esize, DataType::invalid, opNum, 2);
    region = DependencyRegion(hw, esize, rd);
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

bool Instruction12::getARFType(ARFType &arfType, int opNum) const
{
    if (opNum > 1) return false;

    // Only need to support unary/binary, for detecting ce/cr/sr usage.
    switch (opcode()) {
        case Opcode::nop:
        case Opcode::illegal:
        case Opcode::send:
        case Opcode::sendc:
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
            arfType = static_cast<ARFType>(o.direct.regNum >> 4);
            return true;
        }
    }
}

autoswsb::DestinationMask Instruction12::destinations(int &jip, int &uip) const
{
    using namespace autoswsb;

    if (!isBranch(opcode())) {
        if (opcode() == Opcode::send || opcode() == Opcode::sendc)
            if (send.eot)
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
