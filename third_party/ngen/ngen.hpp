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

// nGEN: a C++ library for runtime Gen assembly generation.
//
// Macros that control nGEN's interface:
//    NGEN_SAFE             if defined, enables run-time safety checks. Exceptions will be thrown if checks fail.
//    NGEN_SHORT_NAMES      if defined, enables some short names (r[...] for indirect addressing, W for NoMask)
//    NGEN_GLOBAL_REGS      if defined, register names and instruction modifiers (r7, cr0, Switch, etc.) are
//                           global variables in the ngen namespace. Otherwise, they are members of the code
//                           generator classes
//    NGEN_CPP11            if defined, ngen is C++11-compatible (C++17 not required)

#ifndef NGEN_HPP
#define NGEN_HPP

#ifdef ENABLE_LLVM_WCONVERSION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#endif

#include "ngen_config.hpp"

#include <array>
#include <cstring>
#include <type_traits>
#include <vector>

#include "ngen_core.hpp"
#include "ngen_auto_swsb.hpp"
#include "ngen_debuginfo.hpp"

// -----------------------------------------------------------------------
// Binary formats, split between pre-Gen12 and post-Gen12.
#include "ngen_gen8.hpp"
#include "ngen_gen12.hpp"
// -----------------------------------------------------------------------

#ifdef NGEN_ASM
#include "ngen_asm.hpp"
#endif

namespace NGEN_NAMESPACE {

// Forward declarations.
template <HW hw> class BinaryCodeGenerator;
template <HW hw> class ELFCodeGenerator;

// MSVC v140 workaround for enum comparison in template arguments.
static constexpr bool hwLT(HW hw1, HW hw2) { return hw1 < hw2; }
static constexpr bool hwLE(HW hw1, HW hw2) { return hw1 <= hw2; }
static constexpr bool hwGE(HW hw1, HW hw2) { return hw1 >= hw2; }
static constexpr bool hwGT(HW hw1, HW hw2) { return hw1 > hw2; }

class LabelFixup {
public:
    uint32_t labelID;
    int32_t anchor;
    int32_t offset;

    LabelFixup(uint32_t labelID_, int32_t offset_) : labelID(labelID_), anchor(0), offset(offset_) {}

    static constexpr auto JIPOffset = 12;
    static constexpr auto JIPOffsetJMPI = -4;
    static constexpr auto UIPOffset = 8;
};

#if defined(NGEN_GLOBAL_REGS) && !defined(NGEN_GLOBAL_REGS_DEFINED)
#define NGEN_GLOBAL_REGS_DEFINED
#include "ngen_registers.hpp"
#endif

template <HW hw>
class BinaryCodeGenerator
{
    friend class ELFCodeGenerator<hw>;

public:
    static constexpr HW hardware = hw;
    static constexpr HW getHardware() { return hardware; }

protected:
    class InstructionStream {
        friend class BinaryCodeGenerator;

        std::vector<LabelFixup> fixups;
        std::vector<uint32_t> labels;
        std::vector<uint64_t> code;
        bool appended = false;

        int length() const { return int(code.size() * sizeof(uint64_t)); }

        void db(const Instruction8 &i) {
            code.push_back(i.qword[0]);
            code.push_back(i.qword[1]);
        }

        void db(const Instruction12 &i) {
            code.push_back(i.qword[0]);
            code.push_back(i.qword[1]);
        }

        void addFixup(LabelFixup fixup) {
            fixup.anchor = length();
            fixups.push_back(fixup);
        }

        void mark(Label &label, LabelManager &man) {
            uint32_t id = label.getID(man);

            man.setTarget(id, length());
            labels.push_back(id);
        }

        void fixLabels(LabelManager &man) {
            for (const auto &fixup : fixups) {
                int32_t target = man.getTarget(fixup.labelID);
                uint8_t *field = ((uint8_t *) code.data()) + fixup.anchor + fixup.offset;
                *((int32_t *) field) = target - fixup.anchor;
            }
        }

        void append(InstructionStream &other, LabelManager &man) {
            auto offset = length();
            auto sz = code.size();

            code.resize(sz + other.code.size());
            std::copy(other.code.begin(), other.code.end(), code.begin() + sz);

            sz = labels.size();
            labels.resize(sz + other.labels.size());
            std::copy(other.labels.begin(), other.labels.end(), labels.begin() + sz);

            for (LabelFixup fixup : other.fixups) {
                fixup.anchor += offset;
                fixups.push_back(fixup);
            }

#ifdef NGEN_SAFE
            if (other.appended && !other.labels.empty())
                throw multiple_label_exception();
#endif

            for (uint32_t id : other.labels)
                man.offsetTarget(id, offset);

            other.appended = true;
        }

        InstructionStream() {}
    };

    class Program {
        friend class BinaryCodeGenerator;
        using Instruction = typename Instruction12Dispatch<hw>::type;
        std::vector<uint64_t> &code;

        Program(InstructionStream &stream) : code(stream.code) {};

    public:
        size_t size() const                               { return code.size() >> 1; }
        Instruction &operator[](size_t index)             { return *reinterpret_cast<Instruction *>(&code[index * 2]); }
        const Instruction &operator[](size_t index) const { return *reinterpret_cast<Instruction *>(&code[index * 2]); }
    };

    static constexpr bool isGen12 = (hw >= HW::Gen12LP);
    Product product;
    int declaredGRFs = 128;

    Label _labelLocalIDsLoaded;
    Label _labelArgsLoaded;
    Label _lastFenceLabel;
    RegData _lastFenceDst;

    DebugLine debugLine;

private:
    InstructionModifier defaultModifier;

    LabelManager labelManager;
    InstructionStream rootStream;
    std::vector<InstructionStream*> streamStack;

    void db(const Instruction8 &i)  { streamStack.back()->db(i); }
    void db(const Instruction12 &i) { streamStack.back()->db(i); }
    void addFixup(LabelFixup fixup) { streamStack.back()->addFixup(fixup); }

    template <bool forceWE = false, typename D, typename S0, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, SourceLocation loc);
    template <bool forceWE = false, typename D, typename S0, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, SourceLocation loc);
    template <bool forceWE = false, typename D, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, const Immediate &src0, SourceLocation loc);
    template <bool forceWE = false, typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, const Immediate &src0, SourceLocation loc);

    template <bool forceWE = false, typename D, typename S0, typename S1, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, SourceLocation loc);
    template <bool forceWE = false, typename D, typename S0, typename S1, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, SourceLocation loc);
    template <bool forceWE = false, typename D, typename S0, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, const Immediate &src1, SourceLocation loc);
    template <bool forceWE = false, typename D, typename S0, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, const Immediate &src1, SourceLocation loc);

    template <HW hw_ = hw>
    typename std::enable_if<hwLE(hw_, HW::Gen9)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, RegData dst, RegData src0, RegData src1, RegData src2, SourceLocation loc);
    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, Align16Operand dst, Align16Operand src0, Align16Operand src1, Align16Operand src2, SourceLocation loc);
    template <typename D, typename S0, typename S1, typename S2, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2, SourceLocation loc);
    template <typename D, typename S0, typename S1, typename S2, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2, SourceLocation loc);

    template <typename DS0>
    void opMath(Opcode op, DataType defaultType, const InstructionModifier &mod, MathFunction fc, DS0 dst, DS0 src0, SourceLocation loc);
    template <typename DS0, typename S1>
    void opMath(Opcode op, DataType defaultType, const InstructionModifier &mod, MathFunction fc, DS0 dst, DS0 src0, S1 src1, SourceLocation loc);

    template <typename D, typename S0, typename S2>
    void opBfn(Opcode op, DataType defaultType, const InstructionModifier &mod, int bfnCtrl, D dst, S0 src0, RegData src1, S2 src2, SourceLocation loc);
    void opDpas(Opcode op, DataType defaultType, const InstructionModifier &mod, int sdepth, int rcount, RegData dst, RegData src0, RegData src1, RegData src2, SourceLocation loc);

    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, int src1Length, uint32_t exdesc, D desc, SourceLocation loc);
    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, int src1Length, const RegData &exdesc, D desc, SourceLocation loc);
    template <typename ED, typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, int src1Length, ED exdesc, D desc, SourceLocation loc);

    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc, SourceLocation loc);
    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc, SourceLocation loc);
    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, D desc, SourceLocation loc);

    template <typename ED, typename D, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, ED exdesc, D desc, SourceLocation loc);
    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, D desc, SourceLocation loc);
    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, RegData exdesc, D desc, SourceLocation loc);

    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, int32_t uip, SourceLocation loc);
    template <HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, int32_t uip, SourceLocation loc);
    template <bool forceWE = false, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, SourceLocation loc);
    template <bool forceWE = false, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, SourceLocation loc);
    template <bool forceWE = false, bool small12 = true, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc);
    template <bool forceWE = false, bool small12 = true, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc);

    void opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip, Label &uip, SourceLocation loc);
    template <bool forceWE = false>
    void opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip, SourceLocation loc);
    void opCall(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip, SourceLocation loc);

    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, RegData src0, uint32_t jip, SourceLocation loc);
    template <HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, RegData src0, uint32_t jip, SourceLocation loc);
    void opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, Label &jip, SourceLocation loc);

    void opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, SourceLocation loc);
    void opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, RegData src0, SourceLocation loc);
    void opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, const Immediate &src0, SourceLocation loc);

    void opNop(Opcode op, SourceLocation loc);

    inline void unsupported();

#include "ngen_compiler_fix.hpp"

public:
    explicit BinaryCodeGenerator(Product product_, DebugConfig debugConfig = {})
        : product{product_}, debugLine(debugConfig), defaultModifier{}, labelManager{},

                                                     sync{this}, load{this}, store{this}, atomic{this}
    {
        _workaround_();
        pushStream(rootStream);
    }

    explicit BinaryCodeGenerator(int stepping_ = 0, DebugConfig debugConfig = {}) : BinaryCodeGenerator({genericProductFamily(hw), stepping_}, debugConfig) {}

    ~BinaryCodeGenerator() {
        for (size_t sn = 1; sn < streamStack.size(); sn++)
            delete streamStack[sn];
    }

    std::vector<uint8_t> getCode();
    size_t getRootStreamLength() const { return rootStream.length(); }

    Product getProduct() const { return product; }
    ProductFamily getProductFamily() const { return product.family; }
    int getStepping() const { return product.stepping; }

    void setProduct(Product product_) { product = product_; }
    void setProductFamily(ProductFamily family_) { product.family = family_; }
    void setStepping(int stepping_) { product.stepping = stepping_; }

protected:
    // Configuration.
    void setDefaultNoMask(bool def = true)          { defaultModifier.setWrEn(def); }
    void setDefaultAutoSWSB(bool def = true)        { defaultModifier.setAutoSWSB(def); }
    bool getDefaultNoMask() const                   { return defaultModifier.isWrEn(); }
    bool getDefaultAutoSWSB() const                 { return defaultModifier.isAutoSWSB(); }

    // Stream handling.
    void pushStream()                               { pushStream(new InstructionStream()); }
    void pushStream(InstructionStream *s)           { streamStack.push_back(s); }
    void pushStream(InstructionStream &s)           { pushStream(&s); }

    InstructionStream *popStream();

    void appendStream(InstructionStream *s)         { appendStream(*s); }
    void appendStream(InstructionStream &s)         { streamStack.back()->append(s, labelManager); }
    void appendCurrentStream()                      { InstructionStream *s = popStream(); appendStream(s); delete s; }

    void discardStream()                            { delete popStream(); }

    template <typename String>
    void comment(String)                            {}

    void requireGRF(int grfs)                       { declaredGRFs = grfs; }

    // Registers.
#ifndef NGEN_GLOBAL_REGS
#include "ngen_registers.hpp"
#endif

    // Labels.
    inline void mark(Label &label)          { streamStack.back()->mark(label, labelManager); }

    // Instructions.
    template <typename DT = void>
    void add(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::add, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void add(const InstructionModifier &mod, const RegData &dst,
             const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::add, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst,
              const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::addc, getDataType<DT>(), mod | AccWrEn, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::addc, getDataType<DT>(), mod | AccWrEn, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void and_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::and_gen12 : Opcode::and_, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void and_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::and_gen12 : Opcode::and_, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void and(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        and_<DT>(mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void and(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        and_<DT>(mod, dst, src0, src1, loc);
    }
#endif
    template <typename DT = void>
    void asr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::asr_gen12 : Opcode::asr, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void asr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::asr_gen12 : Opcode::asr, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void avg(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::avg, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void avg(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::avg, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfe_gen12 : Opcode::bfe, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfe_gen12 : Opcode::bfe, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfe_gen12 : Opcode::bfe, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfe_gen12 : Opcode::bfe, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfi1(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfi1_gen12 : Opcode::bfi1, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void bfi1(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfi1_gen12 : Opcode::bfi1, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
        opBfn(Opcode::bfn, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
        opBfn(Opcode::bfn, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
        opBfn(Opcode::bfn, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
        opBfn(Opcode::bfn, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfrev(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfrev_gen12 : Opcode::bfrev, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void bfrev(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfrev_gen12 : Opcode::bfrev, getDataType<DT>(), mod, dst, src0, loc);
    }
    void brc(const InstructionModifier &mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        opBranch(Opcode::brc, mod, isGen12 ? null.ud() : ip.d(), jip, uip, loc);
    }
    void brc(const InstructionModifier &mod, RegData src0, SourceLocation loc = {}) {
        src0.setRegion(2, 2, 1);
        opBranch<true, true>(Opcode::brc, mod, isGen12 ? null.ud() : ip.d(), src0, loc);
    }
    void brd(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        opBranch(Opcode::brd, mod, isGen12 ? null.ud() : ip.d(), jip, loc);
    }
    void brd(const InstructionModifier &mod, RegData src0, SourceLocation loc = {}) {
        src0.setRegion(2, 2, 1);
        opBranch<true, true>(Opcode::brd, mod, isGen12 ? null.ud() : ip.d(), src0, loc);
    }
    void break_(const InstructionModifier &mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        opBranch(Opcode::break_, mod, null, jip, uip, loc);
    }
    void call(const InstructionModifier &mod, const RegData &dst, Label &jip, SourceLocation loc = {}) {
        opCall(Opcode::call, mod, dst, jip, loc);
    }
    void call(const InstructionModifier &mod, const RegData &dst, RegData jip, SourceLocation loc = {}) {
        if (isGen12)
            opBranch<true, true>(Opcode::call, mod, dst, jip, loc);
        else {
            jip.setRegion(0, 1, 0);
            opX<true>(Opcode::call, DataType::d, mod, dst, null.ud(0)(0, 1, 0), jip, loc);
        }
    }
    void calla(const InstructionModifier &mod, const RegData &dst, int32_t jip, SourceLocation loc = {}) {
        if (isGen12)
            opBranch<true>(Opcode::calla, mod, dst, jip, loc);
        else
            opX<true>(Opcode::calla, DataType::d, mod, dst, (hw <= HW::Gen9) ? null.ud(0)(2,2,1) : null.ud(0)(0,1,0), Immediate::d(jip), loc);
    }
    void calla(const InstructionModifier &mod, const RegData &dst, RegData jip, SourceLocation loc = {}) {
        if (isGen12)
            opBranch<true, true>(Opcode::calla, mod, dst, jip, loc);
        else {
            jip.setRegion(0, 1, 0);
            opX<true>(Opcode::calla, DataType::d, mod, dst, null.ud(0)(0, 1, 0), jip, loc);
        }
    }
    template <typename DT = void>
    void cbit(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::cbit, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void cbit(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(Opcode::cbit, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void cmp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::cmp_gen12 : Opcode::cmp, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void cmp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::cmp_gen12 : Opcode::cmp, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void cmpn(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::cmpn_gen12 : Opcode::cmpn, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void csel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::csel_gen12 : Opcode::csel, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void csel(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::csel_gen12 : Opcode::csel, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void csel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::csel_gen12 : Opcode::csel, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void csel(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::csel_gen12 : Opcode::csel, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    void cont(const InstructionModifier &mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        opBranch(Opcode::cont, mod, null, jip, uip, loc);
    }
    template <typename DT = void>
    void dp2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::dp2, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void dp2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::dp2, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void dp3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::dp3, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void dp3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::dp3, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void dp4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::dp4, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void dp4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::dp4, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        if (hw < HW::Gen12LP) unsupported();
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        if (hw < HW::Gen12LP) unsupported();
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        if (hw < HW::Gen12LP) unsupported();
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        if (hw < HW::Gen12LP) unsupported();
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void dpas(const InstructionModifier &mod, uint8_t sdepth, uint8_t rcount, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opDpas(Opcode::dpas, getDataType<DT>(), mod, sdepth, rcount, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void dpasw(const InstructionModifier &mod, uint8_t sdepth, uint8_t rcount, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opDpas(Opcode::dpasw, getDataType<DT>(), mod, sdepth, rcount, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void dph(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::dph, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void dph(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::dph, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    void else_(InstructionModifier mod, Label &jip, Label &uip, bool branchCtrl, SourceLocation loc = {}) {
        mod.setBranchCtrl(branchCtrl);
        opBranch(Opcode::else_, mod, null, jip, uip, loc);
    }
    void else_(InstructionModifier mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        else_(mod, jip, uip, false, loc);
    }
    void else_(InstructionModifier mod, Label &jip, SourceLocation loc = {}) {
        else_(mod, jip, jip, false, loc);
    }
    void endif(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        opBranch(Opcode::endif, mod, null, jip, loc);
    }
    void endif(const InstructionModifier &mod, SourceLocation loc = {}) {
        opBranch(Opcode::endif, mod, null, sizeof(Instruction8), loc);
    }
    template <typename DT = void>
    void fbh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::fbh, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void fbh(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(Opcode::fbh, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void fbl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::fbl, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void fbl(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(Opcode::fbl, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void frc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::frc, getDataType<DT>(), mod, dst, src0, loc);
    }
    void goto_(InstructionModifier mod, Label &jip, Label &uip, bool branchCtrl, SourceLocation loc = {}) {
        mod.setBranchCtrl(branchCtrl);
        opBranch(Opcode::goto_, mod, null, jip, uip, loc);
    }
    void goto_(const InstructionModifier &mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        goto_(mod, jip, uip, false, loc);
    }
    void goto_(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        goto_(mod, jip, jip, loc);
    }
    void halt(const InstructionModifier &mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        opBranch(Opcode::halt, mod, null, jip, uip, loc);
    }
    void halt(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        halt(mod, jip, jip, loc);
    }
    void if_(InstructionModifier mod, Label &jip, Label &uip, bool branchCtrl = false, SourceLocation loc = {}) {
        mod.setBranchCtrl(branchCtrl);
        opBranch(Opcode::if_, mod, null, jip, uip, loc);
    }
    void if_(const InstructionModifier &mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        if_(mod, jip, uip, false, loc);
    }
    void if_(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        if_(mod, jip, jip, false, loc);
    }
    void illegal(SourceLocation loc = {}) {
        opX(Opcode::illegal, DataType::invalid, InstructionModifier(), null, null, null, loc);
    }
    void join(InstructionModifier mod, Label &jip, SourceLocation loc = {}) {
        opBranch(Opcode::join, mod, null, jip, loc);
    }
    void join(InstructionModifier mod, SourceLocation loc = {}) {
        opBranch(Opcode::join, mod, null, sizeof(Instruction8), loc);
    }
    void jmpi(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        auto dst = isGen12 ? ARF(null) : ARF(ip);
        opJmpi(Opcode::jmpi, mod, dst, dst, jip, loc);
    }
    void jmpi(const InstructionModifier &mod, const RegData &jip, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (!isGen12 && jip.getType() != DataType::d && jip.getType() != DataType::invalid)
            throw invalid_type_exception();
#endif
        if (isGen12)
            opBranch<true, false>(Opcode::jmpi, mod, null, jip, loc);
        else
            opX(Opcode::jmpi, DataType::d, mod, ip, ip, jip, loc);
    }
    template <typename DT = void>
    void line(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        if (hw >= HW::Gen11) unsupported();
        opX(Opcode::line, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void line(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        if (hw >= HW::Gen11) unsupported();
        opX(Opcode::line, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void lrp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(Opcode::lrp, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void lzd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::lzd, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void lzd(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(Opcode::lzd, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void mac(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::mac, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mac(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::mac, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mach(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::mach, getDataType<DT>(), (hw >= HW::XeHPC) ? mod : (mod | AccWrEn), dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mach(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::mach, getDataType<DT>(), (hw >= HW::XeHPC) ? mod : (mod | AccWrEn), dst, src0, src1, loc);
    }
    template <typename DT = void>
    void macl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (hw < HW::Gen10) unsupported();
#endif
        opX((hw >= HW::XeHPC) ? Opcode::macl : Opcode::mach, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void macl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (hw < HW::Gen10) unsupported();
#endif
        opX((hw >= HW::XeHPC) ? Opcode::macl : Opcode::mach, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwLE(hw_, HW::Gen9)>::type
    madm(const InstructionModifier &mod, const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1, const ExtendedReg &src2, SourceLocation loc = {}) {
        opX(Opcode::madm, getDataType<DT>(), mod, extToAlign16(dst), extToAlign16(src0), extToAlign16(src1), extToAlign16(src2), loc);
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwGT(hw_, HW::Gen9)>::type
    madm(const InstructionModifier &mod, const ExtendedReg &dst, ExtendedReg src0, ExtendedReg src1, const ExtendedReg &src2, SourceLocation loc = {}) {
        src0.getBase().setRegion(4,4,1);
        src1.getBase().setRegion(4,4,1);
        opX(Opcode::madm, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (mathArgCount(hw, fc) != 1) throw invalid_operand_count_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0, loc);
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (mathArgCount(hw, fc) != 2) throw invalid_operand_count_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (fc == MathFunction::invm || fc == MathFunction::rsqtm) throw invalid_operand_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0, src1.forceInt32(), loc);
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen11)>::type
    math(const InstructionModifier &mod, MathFunction fc, const ExtendedReg &dst, const ExtendedReg &src0, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::rsqtm) throw invalid_operand_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, extToAlign16(dst), extToAlign16(src0), loc);
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen11)>::type
    math(const InstructionModifier &mod, MathFunction fc, const ExtendedReg &dst, ExtendedReg src0, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::rsqtm) throw invalid_operand_exception();
#endif
        if (hw == HW::Gen11)
            src0.getBase().setRegion(2,2,1);
        else
            src0.getBase().setRegion(1,1,0);
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0, loc);
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen11)>::type
    math(const InstructionModifier &mod, MathFunction fc, const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::invm) throw invalid_operand_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, extToAlign16(dst), extToAlign16(src0), extToAlign16(src1), loc);
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen11)>::type
    math(const InstructionModifier &mod, MathFunction fc, const ExtendedReg &dst, ExtendedReg src0, ExtendedReg src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::invm) throw invalid_operand_exception();
#endif
        if (hw == HW::Gen11) {
            src0.getBase().setRegion(2,2,1);
            src1.getBase().setRegion(2,2,1);
        } else {
            src0.getBase().setRegion(1,1,0);
            src1.getBase().setRegion(1,1,0);
        }
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mov(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::mov_gen12 : Opcode::mov, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void mov(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::mov_gen12 : Opcode::mov, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void movi(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (!src0.isIndirect()) throw invalid_address_mode_exception();
#endif
        if (hardware >= HW::Gen10)
            movi<DT>(mod, dst, src0, null.ud(0)(1,1,0));
        else
            opX(isGen12 ? Opcode::movi_gen12 : Opcode::movi, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void movi(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (hardware < HW::Gen10) throw unsupported_instruction();
        if (!src0.isIndirect()) throw invalid_address_mode_exception();
#endif
        opX(isGen12 ? Opcode::movi_gen12 : Opcode::movi, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void movi(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (hardware < HW::Gen10) throw unsupported_instruction();
#endif
        opX(isGen12 ? Opcode::movi_gen12 : Opcode::movi, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mul(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::mul, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mul(const InstructionModifier &mod, const RegData &dst, const RegData &src0, Immediate src1, SourceLocation loc = {}) {
        if (dst.getBytes() == 8)
            src1 = src1.forceInt32();
        opX(Opcode::mul, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    void nop(SourceLocation loc = {}) {
        opNop(isGen12 ? Opcode::nop_gen12 : Opcode::nop, loc);
    }
    void nop(const InstructionModifier &mod, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::nop_gen12 : Opcode::nop, DataType::invalid, mod, null, null, null, loc);
    }
    template <typename DT = void>
    void not_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::not_gen12 : Opcode::not_, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void not_(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::not_gen12 : Opcode::not_, getDataType<DT>(), mod, dst, src0, loc);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void not(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        not_<DT>(mod, dst, src0, loc);
    }
    template <typename DT = void>
    void not(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        not_<DT>(mod, dst, src0, loc);
    }
#endif
    template <typename DT = void>
    void or_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::or_gen12 : Opcode::or_, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void or_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::or_gen12 : Opcode::or_, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void or(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        or_<DT>(mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void or(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        or_<DT>(mod, dst, src0, src1, loc);
    }
#endif
    template <typename DT = void>
    void pln(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        if (hw >= HW::Gen11) unsupported();
        opX(Opcode::pln, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    void ret(const InstructionModifier &mod, RegData src0, SourceLocation loc = {}) {
        src0.setRegion(2,2,1);
        if (isGen12)
            opBranch<true, true>(Opcode::ret, mod, null, src0, loc);
        else
            opX<true>(Opcode::ret, DataType::ud, mod, null, src0, loc);
    }
    template <typename DT = void>
    void rndd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::rndd, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rndd(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(Opcode::rndd, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rnde(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::rnde, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rnde(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(Opcode::rnde, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rndu(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::rndu, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rndu(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(Opcode::rndu, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rndz(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::rndz, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rndz(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(Opcode::rndz, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rol(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::rol_gen12 : Opcode::rol, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void rol(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::rol_gen12 : Opcode::rol, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void ror(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::ror_gen12 : Opcode::ror, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void ror(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::ror_gen12 : Opcode::ror, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sad2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        if (hw >= HW::Gen12LP) unsupported();
        opX(Opcode::sad2, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sad2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        if (hw >= HW::Gen12LP) unsupported();
        opX(Opcode::sad2, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sada2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        if (hw >= HW::Gen12LP) unsupported();
        opX(Opcode::sada2, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sada2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        if (hw >= HW::Gen12LP) unsupported();
        opX(Opcode::sada2, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::sel_gen12 : Opcode::sel, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::sel_gen12 : Opcode::sel, getDataType<DT>(), mod, dst, src0, src1, loc);
    }

    /* Gen12-style sends */
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, src1, -1, exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, src1, -1, exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const GRFRange &src1, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, src1[0], src1.getLen(), exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, src1, -1, exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, src1, -1, exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const GRFRange &src1, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, src1[0], src1.getLen(), exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1, -1, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1, -1, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const GRFRange &src1, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1[0], src1.getLen(), exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1, -1, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1, -1, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const GRFRange &src1, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1[0], src1.getLen(), exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc, loc);
    }
    /* Pre-Gen12-style sends; also supported on Gen12. */
    void send(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, dst, src0, exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, dst, src0, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, dst, src0, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, dst, src0, exdesc, desc, loc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSends(Opcode::sends, mod, dst, src0, src1, exdesc, desc, loc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSends(Opcode::sends, mod, dst, src0, src1, exdesc, desc, loc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSends(Opcode::sends, mod, dst, src0, src1, exdesc, desc, loc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSends(Opcode::sends, mod, dst, src0, src1, exdesc, desc, loc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSends(Opcode::sendsc, mod, dst, src0, src1, exdesc, desc, loc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSends(Opcode::sendsc, mod, dst, src0, src1, exdesc, desc, loc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSends(Opcode::sendsc, mod, dst, src0, src1, exdesc, desc, loc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSends(Opcode::sendsc, mod, dst, src0, src1, exdesc, desc, loc);
    }

    template <typename DT = void>
    void shl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::shl_gen12 : Opcode::shl, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void shl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::shl_gen12 : Opcode::shl, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void shr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::shr_gen12 : Opcode::shr, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void shr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::shr_gen12 : Opcode::shr, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void smov(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::smov_gen12 : Opcode::smov, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void srnd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::srnd, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void srnd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::srnd, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::subb, getDataType<DT>(), mod | AccWrEn, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::subb, getDataType<DT>(), mod | AccWrEn, dst, src0, src1, loc);
    }
    void wait(const InstructionModifier &mod, const RegData &nreg, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (!nreg.isARF() || nreg.getARFType() != ARFType::n) throw invalid_arf_exception();
#endif
        opX(Opcode::wait, DataType::invalid, mod, nreg, nreg, loc);
    }
    void while_(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        opBranch(Opcode::while_, mod, null, jip, loc);
    }
    template <typename DT = void>
    void xor_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::xor_gen12 : Opcode::xor_, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void xor_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::xor_gen12 : Opcode::xor_, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void xor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        xor_<DT>(mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void xor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        xor_<DT>(mod, dst, src0, src1, loc);
    }
#endif

private:
    struct Sync {
        BinaryCodeGenerator<hw> &parent;

        Sync(BinaryCodeGenerator<hw> *parent_) : parent(*parent_) {}

        void operator()(SyncFunction fc, const InstructionModifier &mod = InstructionModifier(), SourceLocation loc = {}) {
            parent.opSync(Opcode::sync, fc, mod, loc);
        }
        void operator()(SyncFunction fc, const RegData &src0, SourceLocation loc) {
            this->operator()(fc, InstructionModifier(), src0, loc);
        }
        void operator()(SyncFunction fc, const InstructionModifier &mod, const RegData &src0, SourceLocation loc) {
            parent.opSync(Opcode::sync, fc, mod, src0, loc);
        }
        void operator()(SyncFunction fc, int src0, SourceLocation loc) {
            this->operator()(fc, InstructionModifier(), src0, loc);
        }
        void operator()(SyncFunction fc, const InstructionModifier &mod, uint32_t src0, SourceLocation loc) {
            parent.opSync(Opcode::sync, fc, mod, Immediate::ud(src0), loc);
        }
        void allrd(SourceLocation loc = {}) {
            allrd(null.ud(0)(0, 1, 1), loc);
        }
        void allrd(const InstructionModifier &mod, SourceLocation loc = {}) {
            allrd(mod, null.ud(0)(0, 1, 1), loc);
        }
        void allrd(const RegData &src0, SourceLocation loc = {}) {
            allrd(InstructionModifier(), src0, loc);
        }
        void allrd(const InstructionModifier &mod, const RegData &src0, SourceLocation loc = {}) {
            this->operator()(SyncFunction::allrd, mod, src0, loc);
        }
        void allrd(uint32_t src0, SourceLocation loc = {}) {
            allrd(InstructionModifier(), src0, loc);
        }
        void allrd(const InstructionModifier &mod, uint32_t src0, SourceLocation loc = {}) {
            this->operator()(SyncFunction::allrd, mod, src0, loc);
        }
        void allwr(SourceLocation loc = {}) {
            allwr(null, loc);
        }
        void allwr(const InstructionModifier &mod, SourceLocation loc = {}) {
            allwr(mod, null, loc);
        }
        void allwr(const RegData &src0, SourceLocation loc = {}) {
            allwr(InstructionModifier(), src0, loc);
        }
        void allwr(const InstructionModifier &mod, const RegData &src0, SourceLocation loc = {}) {
            this->operator()(SyncFunction::allwr, mod, src0, loc);
        }
        void allwr(uint32_t src0, SourceLocation loc = {}) {
            allwr(InstructionModifier(), src0, loc);
        }
        void allwr(const InstructionModifier &mod, uint32_t src0, SourceLocation loc = {}) {
            this->operator()(SyncFunction::allwr, mod, src0, loc);
        }
        void bar(const InstructionModifier &mod = InstructionModifier(), SourceLocation loc = {}) {
            this->operator()(SyncFunction::bar, mod, loc);
        }
        void bar(const InstructionModifier &mod, uint32_t src0, SourceLocation loc = {}) {
            this->operator()(SyncFunction::bar, mod, src0, loc);
        }
        void bar(const InstructionModifier &mod, const RegData &src0, SourceLocation loc = {}) {
            this->operator()(SyncFunction::bar, mod, src0, loc);
        }
        void bar(uint32_t src0, SourceLocation loc = {}) {
            this->operator()(SyncFunction::bar, InstructionModifier(), src0, loc);
        }
        void bar(const RegData &src0, SourceLocation loc = {}) {
            this->operator()(SyncFunction::bar, InstructionModifier(), src0, loc);
        }
        void flush(SourceLocation loc = {}) {
            flush(InstructionModifier(), loc);
        }
        void flush(const InstructionModifier &mod, SourceLocation loc = {}) {
            this->operator()(SyncFunction::flush, InstructionModifier(), null, loc);
        }
        void host(const InstructionModifier &mod = InstructionModifier(), SourceLocation loc = {}) {
            this->operator()(SyncFunction::host, mod, loc);
        }
        void nop(const InstructionModifier &mod = InstructionModifier(), SourceLocation loc = {}) {
            this->operator()(SyncFunction::nop, mod, loc);
        }
    };
public:
    Sync sync;

    void ignoredep(Operand op, SourceLocation loc = {}) {
        if (hw >= HW::Gen12LP)
            opX(Opcode::directive, DataType::ud, InstructionModifier(), GRF(static_cast<int>(op)), NullRegister(), NullRegister(), loc);
    }
    void subdep(Operand op, const GRFRange &r, SourceLocation loc) {
        if (op == Operand::dst && !r.isEmpty()) {
#ifdef NGEN_SAFE
            if (r.getLen() > 32) throw invalid_directive_exception();
#endif
            opX(Opcode::directive, DataType::ud, InstructionModifier::createAutoSWSB(), GRF(static_cast<int>(Directive::subdep_dst)), r[0], r[r.getLen() - 1], loc);
        } else {
            ignoredep(op, loc);
            wrdep(r, loc);
        }
    }
    void subdep(Operand op, const GRF &r, SourceLocation loc = {}) {
        subdep(op, r-r, loc);
    }
    void wrdep(const GRFRange &r, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (hw < HW::Gen12LP) throw unsupported_instruction();
#endif
        int len = r.getLen();
        for (int o = 0; o < len; o += 32) {
            int thisLen = std::min(len - o, 32);
            opX(Opcode::directive, DataType::ud, InstructionModifier::createAutoSWSB(), GRF(static_cast<int>(Directive::wrdep)), r[o], r[o + thisLen - 1], loc);
        }
    }
    void wrdep(const GRF &r, SourceLocation loc = {}) {
        wrdep(r-r, loc);
    }
    void fencedep(Label &fenceLocation, SourceLocation loc) {
        addFixup(LabelFixup(fenceLocation.getID(labelManager), LabelFixup::JIPOffset));
        opX(Opcode::directive, DataType::ud, InstructionModifier::createAutoSWSB(), GRF(static_cast<int>(Directive::fencedep)), Immediate::ud(0), loc);
    }
    void disablePVCWARWA(SourceLocation loc) {
        opX(Opcode::directive, DataType::ud, InstructionModifier::createAutoSWSB(), GRF(static_cast<int>(Directive::pvcwarwa)), NullRegister(), loc);
    }

    using _self = BinaryCodeGenerator<hw>;
#include "ngen_pseudo.hpp"
};

#define NGEN_FORWARD(hw) NGEN_FORWARD_NAMESPACE(NGEN_NAMESPACE::BinaryCodeGenerator<hw>)

#define NGEN_FORWARD_NAMESPACE(ns) \
NGEN_FORWARD_NAMESPACE_NO_ELF_OVERRIDES(ns) \
void requireGRF(int grfs) { ns::requireGRF(grfs); }

#define NGEN_NILARY_OP(op, ns) void op(NGEN_NAMESPACE::SourceLocation loc = {}) {ns::op(loc);}
#define NGEN_UNARY_OP(op, ns) template <typename A0> void op(A0 &&a0, NGEN_NAMESPACE::SourceLocation loc = {}) {ns::op(std::forward<A0>(a0), loc);}
#define NGEN_BINARY_OP(op, ns) template <typename A0, typename A1> void op(A0 &&a0, A1 &&a1, NGEN_NAMESPACE::SourceLocation loc = {}) {ns::op(std::forward<A0>(a0), std::forward<A1>(a1), loc);}
#define NGEN_TERNARY_OP(op, ns) template <typename A0, typename A1, typename A2> void op(A0 &&a0, A1 &&a1, A2 &&a2, NGEN_NAMESPACE::SourceLocation loc = {}) {ns::op(std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), loc);}
#define NGEN_QUADRARY_OP(op, ns) template <typename A0, typename A1, typename A2, typename A3> void op(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, NGEN_NAMESPACE::SourceLocation loc = {}) {ns::op(std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), loc);}
#define NGEN_PENTARY_OP(op, ns) template <typename A0, typename A1, typename A2, typename A3, typename A4> void op(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, NGEN_NAMESPACE::SourceLocation loc = {}) {ns::op(std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), std::forward<A4>(a4), loc);}
#define NGEN_HEXARY_OP(op, ns) template <typename A0, typename A1, typename A2, typename A3, typename A4, typename A5> void op(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5, NGEN_NAMESPACE::SourceLocation loc = {}) {ns::op(std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), std::forward<A4>(a4), std::forward<A5>(a5), loc);}
#define NGEN_SEPTARY_OP(op, ns) template <typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6> void op(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5, A6 &&a6, NGEN_NAMESPACE::SourceLocation loc = {}) {ns::op(std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), std::forward<A4>(a4), std::forward<A5>(a5), std::forward<A6>(a6), loc);}

#define NGEN_FORWARD_NAMESPACE_OP(op, ns) \
    NGEN_UNARY_OP(op, ns)       \
    NGEN_BINARY_OP(op, ns)      \
    NGEN_TERNARY_OP(op, ns)     \
    NGEN_QUADRARY_OP(op, ns)    \
    NGEN_PENTARY_OP(op, ns)     \
    NGEN_HEXARY_OP(op, ns)      \
    NGEN_SEPTARY_OP(op, ns)     \

#define NGEN_BINARY_DT_OP(op, ns) template <typename DT = void, typename A0, typename A1> void op(const NGEN_NAMESPACE::InstructionModifier &mod, A0 &&a0, A1 &&a1, NGEN_NAMESPACE::SourceLocation loc = {}) {ns::template op<DT>(mod, std::forward<A0>(a0), std::forward<A1>(a1), loc);}
#define NGEN_TERNARY_DT_OP(op, ns) template <typename DT = void, typename A0, typename A1, typename A2> void op(const NGEN_NAMESPACE::InstructionModifier &mod, A0 &&a0, A1 &&a1, A2 &&a2, NGEN_NAMESPACE::SourceLocation loc = {}) {ns::template op<DT>(mod, std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), loc);}
#define NGEN_QUADRARY_DT_OP(op, ns) template <typename DT = void, typename A0, typename A1, typename A2, typename A3> void op(const NGEN_NAMESPACE::InstructionModifier &mod, A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, NGEN_NAMESPACE::SourceLocation loc = {}) {ns::template op<DT>(mod, std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), loc);}
#define NGEN_PENTARY_DT_OP(op, ns) template <typename DT = void, typename A0, typename A1, typename A2, typename A3, typename A4> void op(const NGEN_NAMESPACE::InstructionModifier &mod, A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, NGEN_NAMESPACE::SourceLocation loc = {}) {ns::template op<DT>(mod, std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), std::forward<A4>(a4), loc);}
#define NGEN_HEXARY_DT_OP(op, ns) template <typename DT = void, typename A0, typename A1, typename A2, typename A3, typename A4, typename A5> void op(const NGEN_NAMESPACE::InstructionModifier &mod, A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5, NGEN_NAMESPACE::SourceLocation loc = {}) {ns::template op<DT>(mod, std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), std::forward<A4>(a4), std::forward<A5>(a5), loc);}
#define NGEN_OCTARY_DT_OP(op, ns) template <typename DT = void, typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7> void op(const NGEN_NAMESPACE::InstructionModifier &mod, A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5, A6 &&a6, A7 &&a7, NGEN_NAMESPACE::SourceLocation loc = {}) {ns::template op<DT>(mod, std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), std::forward<A4>(a4), std::forward<A5>(a5), std::forward<A6>(a6), std::forward<A7>(a7), loc);}

#define NGEN_FORWARD_NAMESPACE_DT_OP(op, ns) \
    NGEN_BINARY_DT_OP(op, ns)      \
    NGEN_TERNARY_DT_OP(op, ns)     \
    NGEN_QUADRARY_DT_OP(op, ns)    \
    NGEN_PENTARY_DT_OP(op, ns)     \
    NGEN_HEXARY_DT_OP(op, ns)      \
    NGEN_OCTARY_DT_OP(op, ns)      \

#define NGEN_FORWARD_NAMESPACE_NO_ELF_OVERRIDES(ns)            \
using ns::isGen12; \
constexpr NGEN_NAMESPACE::HW getHardware() const { return ns::getHardware(); } \
NGEN_FORWARD_NAMESPACE_DT_OP(add, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(addc, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(add3, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(and_, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(asr, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(avg, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(bfe, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(bfi1, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(bfi2, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(bfn, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(bfrev, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(cbit, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(cmp, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(cmpn, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(csel, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(dp2, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(dp3, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(dp4, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(dp4a, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(dpas, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(dpasw, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(dph, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(fbh, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(fbl, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(frc, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(line, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(lrp, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(lzd, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(mac, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(macl, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(mach, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(mad, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(madm, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(math, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(mov, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(movi, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(mul, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(not_, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(or_, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(pln, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(rndd, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(rnde, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(rndu, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(rndz, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(rol, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(ror, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(sad2, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(sada2, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(sel, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(shl, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(shr, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(smov, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(subb, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(xor_, ns) \
NGEN_FORWARD_NAMESPACE_OP(brc, ns) \
NGEN_FORWARD_NAMESPACE_OP(brd, ns) \
NGEN_FORWARD_NAMESPACE_OP(break_, ns) \
NGEN_FORWARD_NAMESPACE_OP(call, ns) \
NGEN_FORWARD_NAMESPACE_OP(calla, ns) \
NGEN_FORWARD_NAMESPACE_OP(cont, ns) \
NGEN_FORWARD_NAMESPACE_OP(else_, ns) \
NGEN_FORWARD_NAMESPACE_OP(endif, ns) \
NGEN_FORWARD_NAMESPACE_OP(goto_, ns) \
NGEN_FORWARD_NAMESPACE_OP(halt, ns) \
NGEN_FORWARD_NAMESPACE_OP(if_, ns) \
NGEN_NILARY_OP(illegal, ns) \
NGEN_FORWARD_NAMESPACE_OP(join, ns) \
NGEN_FORWARD_NAMESPACE_OP(jmpi, ns) \
NGEN_NILARY_OP(nop, ns) \
NGEN_FORWARD_NAMESPACE_OP(ret, ns) \
NGEN_FORWARD_NAMESPACE_OP(send, ns) \
NGEN_FORWARD_NAMESPACE_OP(sendc, ns) \
NGEN_FORWARD_NAMESPACE_OP(sends, ns) \
NGEN_FORWARD_NAMESPACE_OP(sendsc, ns) \
using ns::sync; \
NGEN_FORWARD_NAMESPACE_OP(wait, ns) \
NGEN_FORWARD_NAMESPACE_OP(while_, ns) \
NGEN_FORWARD_NAMESPACE_OP(ignoredep, ns) \
NGEN_FORWARD_NAMESPACE_OP(subdep, ns) \
NGEN_FORWARD_NAMESPACE_OP(wrdep, ns) \
NGEN_FORWARD_NAMESPACE_OP(fencedep, ns) \
NGEN_NILARY_OP(disablePVCWARWA, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(min_, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(max_, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(bfi, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(cos, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(exp, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(fdiv, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(idiv, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(inv, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(invm, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(iqot, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(irem, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(log, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(pow, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(rsqt, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(rsqtm, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(sin, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(sqt, ns) \
template <typename DT = void, typename... Targs> void fdiv_ieee(Targs&&... args) { ns::template fdiv_ieee<DT>(std::forward<Targs>(args)...); } \
NGEN_FORWARD_NAMESPACE_DT_OP(inv_ieee, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(sqt_ieee, ns) \
NGEN_FORWARD_NAMESPACE_OP(threadend, ns) \
template <typename... Targs> void barrierheader(Targs&&... args) { ns::barrierheader(std::forward<Targs>(args)...); } \
NGEN_FORWARD_NAMESPACE_OP(barriermsg, ns)                                           \
template <typename... Targs> void barriersignal(Targs&&... args) { ns::barriersignal(std::forward<Targs>(args)...); } \
NGEN_NILARY_OP(barrierwait, ns) \
NGEN_FORWARD_NAMESPACE_OP(barrierwait, ns) \
template <typename... Targs> void barrier(Targs&&... args) { ns::barrier(std::forward<Targs>(args)...); } \
using ns::load; \
using ns::store; \
using ns::atomic; \
template <typename... Targs> void memfence(Targs&&... args) { ns::memfence(std::forward<Targs>(args)...); } \
template <typename... Targs> void slmfence(Targs&&... args) { ns::slmfence(std::forward<Targs>(args)...); } \
NGEN_NILARY_OP(fencewait, ns) \
template <typename... Targs> void loadlid(Targs&&... args) { ns::loadlid(std::forward<Targs>(args)...); } \
template <typename... Targs> void loadargs(Targs&&... args) { ns::loadargs(std::forward<Targs>(args)...); } \
template <typename... Targs> void epilogue(int GRFCount, bool hasSLM, const NGEN_NAMESPACE::RegData &r0_info) { ns::epilogue(GRFCount, hasSLM, r0_info); } \
template <typename... Targs> void pushStream(Targs&&... args) { ns::pushStream(std::forward<Targs>(args)...); } \
template <typename... Targs> void appendStream(Targs&&... args) { ns::appendStream(std::forward<Targs>(args)...); } \
template <typename... Targs> void appendCurrentStream(Targs&&... args) { ns::appendCurrentStream(std::forward<Targs>(args)...); } \
template <typename... Targs> void discardStream(Targs&&... args) { ns::discardStream(std::forward<Targs>(args)...); } \
template <typename... Targs> void mark(Targs&&... args) { ns::mark(std::forward<Targs>(args)...); } \
template <typename... Targs> void comment(Targs&&... args) { ns::comment(std::forward<Targs>(args)...); } \
template <typename... Targs> void setDefaultNoMask(Targs&&... args) { ns::setDefaultNoMask(std::forward<Targs>(args)...); } \
template <typename... Targs> void setDefaultAutoSWSB(Targs&&... args) { ns::setDefaultAutoSWSB(std::forward<Targs>(args)...); } \
bool getDefaultNoMask() const { return ns::getDefaultNoMask(); } \
bool getDefaultAutoSWSB() const { return ns::getDefaultAutoSWSB(); } \
using ns::product; \
NGEN_NAMESPACE::Product getProduct() const { return ns::getProduct(); } \
NGEN_NAMESPACE::ProductFamily getProductFamily() const { return ns::getProductFamily(); } \
int getStepping() const { return ns::getStepping(); } \
void setProduct(NGEN_NAMESPACE::Product product_) { ns::setProduct(product_); } \
void setProductFamily(NGEN_NAMESPACE::ProductFamily family_) { ns::setProductFamily(family_); } \
void setStepping(int stepping_) { ns::setStepping(stepping_); } \
NGEN_FORWARD_NAMESPACE_OP_NAMES(ns) \
NGEN_FORWARD_NAMESPACE_MIN_MAX(ns) \
NGEN_FORWARD_NAMESPACE_REGISTERS(ns)

#ifdef NGEN_NO_OP_NAMES
#define NGEN_FORWARD_NAMESPACE_OP_NAMES(ns)
#else
#define NGEN_FORWARD_NAMESPACE_OP_NAMES(ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(and, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(not, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(or, ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(xor, ns)
#endif

#ifdef NGEN_WINDOWS_COMPAT
#define NGEN_FORWARD_NAMESPACE_MIN_MAX(ns)
#else
#define NGEN_FORWARD_NAMESPACE_MIN_MAX(ns) \
NGEN_FORWARD_NAMESPACE_DT_OP(min, ns)     \
NGEN_FORWARD_NAMESPACE_DT_OP(max, ns)
#endif

#ifdef NGEN_GLOBAL_REGS
#define NGEN_FORWARD_NAMESPACE_REGISTERS(ns)
#else
#define NGEN_FORWARD_NAMESPACE_REGISTERS_BASE(ns) \
using ns::indirect; \
using ns::r0; using ns::r1; using ns::r2; using ns::r3; \
using ns::r4; using ns::r5; using ns::r6; using ns::r7; \
using ns::r8; using ns::r9; using ns::r10; using ns::r11; \
using ns::r12; using ns::r13; using ns::r14; using ns::r15; \
using ns::r16; using ns::r17; using ns::r18; using ns::r19; \
using ns::r20; using ns::r21; using ns::r22; using ns::r23; \
using ns::r24; using ns::r25; using ns::r26; using ns::r27; \
using ns::r28; using ns::r29; using ns::r30; using ns::r31; \
using ns::r32; using ns::r33; using ns::r34; using ns::r35; \
using ns::r36; using ns::r37; using ns::r38; using ns::r39; \
using ns::r40; using ns::r41; using ns::r42; using ns::r43; \
using ns::r44; using ns::r45; using ns::r46; using ns::r47; \
using ns::r48; using ns::r49; using ns::r50; using ns::r51; \
using ns::r52; using ns::r53; using ns::r54; using ns::r55; \
using ns::r56; using ns::r57; using ns::r58; using ns::r59; \
using ns::r60; using ns::r61; using ns::r62; using ns::r63; \
using ns::r64; using ns::r65; using ns::r66; using ns::r67; \
using ns::r68; using ns::r69; using ns::r70; using ns::r71; \
using ns::r72; using ns::r73; using ns::r74; using ns::r75; \
using ns::r76; using ns::r77; using ns::r78; using ns::r79; \
using ns::r80; using ns::r81; using ns::r82; using ns::r83; \
using ns::r84; using ns::r85; using ns::r86; using ns::r87; \
using ns::r88; using ns::r89; using ns::r90; using ns::r91; \
using ns::r92; using ns::r93; using ns::r94; using ns::r95; \
using ns::r96; using ns::r97; using ns::r98; using ns::r99; \
using ns::r100; using ns::r101; using ns::r102; using ns::r103; \
using ns::r104; using ns::r105; using ns::r106; using ns::r107; \
using ns::r108; using ns::r109; using ns::r110; using ns::r111; \
using ns::r112; using ns::r113; using ns::r114; using ns::r115; \
using ns::r116; using ns::r117; using ns::r118; using ns::r119; \
using ns::r120; using ns::r121; using ns::r122; using ns::r123; \
using ns::r124; using ns::r125; using ns::r126; using ns::r127; \
using ns::r128; using ns::r129; using ns::r130; using ns::r131; \
using ns::r132; using ns::r133; using ns::r134; using ns::r135; \
using ns::r136; using ns::r137; using ns::r138; using ns::r139; \
using ns::r140; using ns::r141; using ns::r142; using ns::r143; \
using ns::r144; using ns::r145; using ns::r146; using ns::r147; \
using ns::r148; using ns::r149; using ns::r150; using ns::r151; \
using ns::r152; using ns::r153; using ns::r154; using ns::r155; \
using ns::r156; using ns::r157; using ns::r158; using ns::r159; \
using ns::r160; using ns::r161; using ns::r162; using ns::r163; \
using ns::r164; using ns::r165; using ns::r166; using ns::r167; \
using ns::r168; using ns::r169; using ns::r170; using ns::r171; \
using ns::r172; using ns::r173; using ns::r174; using ns::r175; \
using ns::r176; using ns::r177; using ns::r178; using ns::r179; \
using ns::r180; using ns::r181; using ns::r182; using ns::r183; \
using ns::r184; using ns::r185; using ns::r186; using ns::r187; \
using ns::r188; using ns::r189; using ns::r190; using ns::r191; \
using ns::r192; using ns::r193; using ns::r194; using ns::r195; \
using ns::r196; using ns::r197; using ns::r198; using ns::r199; \
using ns::r200; using ns::r201; using ns::r202; using ns::r203; \
using ns::r204; using ns::r205; using ns::r206; using ns::r207; \
using ns::r208; using ns::r209; using ns::r210; using ns::r211; \
using ns::r212; using ns::r213; using ns::r214; using ns::r215; \
using ns::r216; using ns::r217; using ns::r218; using ns::r219; \
using ns::r220; using ns::r221; using ns::r222; using ns::r223; \
using ns::r224; using ns::r225; using ns::r226; using ns::r227; \
using ns::r228; using ns::r229; using ns::r230; using ns::r231; \
using ns::r232; using ns::r233; using ns::r234; using ns::r235; \
using ns::r236; using ns::r237; using ns::r238; using ns::r239; \
using ns::r240; using ns::r241; using ns::r242; using ns::r243; \
using ns::r244; using ns::r245; using ns::r246; using ns::r247; \
using ns::r248; using ns::r249; using ns::r250; using ns::r251; \
using ns::r252; using ns::r253; using ns::r254; using ns::r255; \
using ns::null; \
using ns::a0; \
using ns::acc0; using ns::acc1; using ns::acc2; using ns::acc3; \
using ns::acc4; using ns::acc5; using ns::acc6; using ns::acc7; \
using ns::acc8; using ns::acc9; \
using ns::mme0; using ns::mme1; using ns::mme2; using ns::mme3; \
using ns::mme4; using ns::mme5; using ns::mme6; using ns::mme7; \
using ns::noacc; using ns::nomme; \
using ns::f0; using ns::f1; using ns::f2; using ns::f3; \
using ns::f0_0; using ns::f0_1; using ns::f1_0; using ns::f1_1; \
using ns::ce0; using ns::sp; using ns::sr0; using ns::sr1; \
using ns::cr0; using ns::n0; using ns::ip; using ns::tdr0; \
using ns::tm0; using ns::tm1; using ns::tm2; using ns::tm3; \
using ns::tm4; using ns::pm0; using ns::tp0; using ns::dbg0; \
using ns::fc0; using ns::fc1; using ns::fc2; using ns::fc3; \
using ns::NoDDClr; using ns::NoDDChk; \
using ns::AccWrEn; using ns::NoSrcDepSet; using ns::Breakpoint; using ns::sat; \
using ns::NoMask; \
using ns::ExBSO; \
using ns::Serialize; using ns::EOT; \
using ns::Atomic; using ns::Switch; using ns::NoPreempt; \
using ns::anyv; using ns::allv; using ns::any2h; using ns::all2h; \
using ns::any4h; using ns::all4h; using ns::any8h; using ns::all8h; \
using ns::any16h; using ns::all16h; using ns::any32h; using ns::all32h; \
using ns::any; using ns::all; \
using ns::x_repl; using ns::y_repl; using ns::z_repl; using ns::w_repl; \
using ns::ze; using ns::eq; using ns::nz; using ns::ne; \
using ns::gt; using ns::ge; using ns::lt; using ns::le; \
using ns::ov; using ns::un; using ns::eo; \
using ns::M0; using ns::M4; using ns::M8; using ns::M12; \
using ns::M16; using ns::M20; using ns::M24; using ns::M28; \
using ns::sb0; using ns::sb1; using ns::sb2; using ns::sb3; \
using ns::sb4; using ns::sb5; using ns::sb6; using ns::sb7; \
using ns::sb8; using ns::sb9; using ns::sb10; using ns::sb11; \
using ns::sb12; using ns::sb13; using ns::sb14; using ns::sb15; \
using ns::sb16; using ns::sb17; using ns::sb18; using ns::sb19; \
using ns::sb20; using ns::sb21; using ns::sb22; using ns::sb23; \
using ns::sb24; using ns::sb25; using ns::sb26; using ns::sb27; \
using ns::sb28; using ns::sb29; using ns::sb30; using ns::sb31; \
using ns::NoAccSBSet; \
using ns::A32; using ns::A32NC; using ns::A64; using ns::A64NC; \
using ns::SLM; \
template <typename... Targs> NGEN_NAMESPACE::InstructionModifier ExecutionOffset(Targs&&... args) { return ns::ExecutionOffset(std::forward<Targs>(args)...); } \
template <typename... Targs> NGEN_NAMESPACE::AddressBase Surface(Targs&&... args) { return ns::Surface(std::forward<Targs>(args)...); } \
template <typename... Targs> NGEN_NAMESPACE::AddressBase CC(Targs&&... args) { return ns::CC(std::forward<Targs>(args)...); } \
template <typename... Targs> NGEN_NAMESPACE::AddressBase SC(Targs&&... args) { return ns::SC(std::forward<Targs>(args)...); } \
using ns::D8; using ns::D16; using ns::D32; using ns::D64; \
using ns::D8U32; using ns::D16U32; \
using ns::D8T; using ns::D16T; using ns::D32T; using ns::D64T; \
using ns::D8U32T; using ns::D16U32T; \
using ns::V1; using ns::V2; using ns::V3; using ns::V4; \
using ns::V8; using ns::V16; using ns::V32; using ns::V64; \
using ns::V1T; using ns::V2T; using ns::V3T; using ns::V4T; \
using ns::V8T; using ns::V16T; using ns::V32T; using ns::V64T; \
using ns::transpose; \
using ns::vnni; \
using ns::L1UC_L3UC; using ns::L1UC_L3C; using ns::L1C_L3UC; using ns::L1C_L3C; \
using ns::L1S_L3UC; using ns::L1S_L3C; using ns::L1IAR_L3C; using ns::L1UC_L3WB; \
using ns::L1WT_L3UC; using ns::L1WT_L3WB; using ns::L1S_L3WB; using ns::L1WB_L3WB; \
using ns::L1C_L3CC; using ns::L1UC_L3CC;
#define NGEN_FORWARD_NAMESPACE_REGISTERS_EXTRA1(ns) \
using ns::s0;
#define NGEN_FORWARD_NAMESPACE_REGISTERS_EXTRA2(ns)
#define NGEN_FORWARD_NAMESPACE_REGISTERS_EXTRA3(ns)
#define NGEN_FORWARD_NAMESPACE_REGISTERS(ns) NGEN_FORWARD_NAMESPACE_REGISTERS_BASE(ns) NGEN_FORWARD_NAMESPACE_REGISTERS_EXTRA1(ns) NGEN_FORWARD_NAMESPACE_REGISTERS_EXTRA2(ns) NGEN_FORWARD_NAMESPACE_REGISTERS_EXTRA3(ns)
#endif

template <HW hw>
inline void BinaryCodeGenerator<hw>::unsupported()
{
#ifdef NGEN_SAFE
    throw unsupported_instruction();
#endif
}

template <HW hw>
typename BinaryCodeGenerator<hw>::InstructionStream *BinaryCodeGenerator<hw>::popStream()
{
#ifdef NGEN_SAFE
    if (streamStack.size() <= 1) throw stream_stack_underflow();
#endif

    InstructionStream *result = streamStack.back();
    streamStack.pop_back();
    return result;
}

template <HW hw>
static inline Instruction12 encodeSyncInsertion(autoswsb::SyncInsertion &si)
{
    Instruction12 i;

    i.common.opcode = static_cast<int>(Opcode::sync);
    i.common.swsb = (hw >= HW::XeHPC) ? SWSBInfoXeHPC(si.swsb, Opcode::sync).raw()
                                      :    SWSBInfo12(si.swsb, Opcode::sync).raw();
    i.common.maskCtrl = true;
    i.binary.cmod = static_cast<int>(si.fc);

    if (si.mask) {
        i.binary.src0Type = getTypecode12(DataType::ud);
        i.binary.src0Imm = true;
        i.imm32.value = si.mask;
    }
    i.binary.dst = 1;

    return i;
}

template <HW hw>
static inline Instruction12 encodeDummyMovInsertion(autoswsb::DummyMovInsertion &mi)
{
    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};

    i.common.opcode = static_cast<int>(Opcode::mov_gen12);
    i.common.swsb = (hw >= HW::XeHPC) ? SWSBInfoXeHPC(mi.swsb, Opcode::sync).raw()
                                      :    SWSBInfo12(mi.swsb, Opcode::sync).raw();
    i.common.maskCtrl = true;
    i.binary.dst = 1;
    i.binary.dstType = i.binary.src0Type = getTypecode12(DataType::ud);

    if (mi.constant) {
        i.binary.src0Imm = true;
        i.imm32.value = 0;
    } else
        i.binary.src0 = encodeBinaryOperand12<0>(GRF(mi.grf).ud(0), tag).bits;

    return i;
}

template <HW hw>
std::vector<uint8_t> BinaryCodeGenerator<hw>::getCode()
{
#ifdef NGEN_SAFE
    if (streamStack.size() > 1) throw unfinished_stream_exception();
#endif
    rootStream.fixLabels(labelManager);

    Program program(rootStream);
    autoswsb::BasicBlockList analysis = autoswsb::autoSWSB(hw, declaredGRFs, program);
    std::vector<uint8_t> result;

    if (analysis.empty()) {
        result.resize(rootStream.length());
        std::memmove(result.data(), rootStream.code.data(), rootStream.length());
    } else {
        std::multimap<int32_t, autoswsb::SyncInsertion*> syncs;
        std::multimap<int32_t, autoswsb::DummyMovInsertion*> movs;

        for (auto &bb : analysis) {
            for (auto &sync : bb.syncs)
                syncs.insert(std::make_pair(sync.inum, &sync));
            for (auto &mov : bb.movs)
                movs.insert(std::make_pair(mov.inum, &mov));
        }

        result.resize(rootStream.length() + (syncs.size() + movs.size()) * sizeof(Instruction12));

        auto *psrc_start = reinterpret_cast<const Instruction12 *>(rootStream.code.data());
        auto *psrc = psrc_start;
        auto *pdst_start = reinterpret_cast<Instruction12 *>(result.data());
        auto *pdst = pdst_start;
        auto &srcLines = debugLine.srcLines;

        auto nextSync = syncs.begin();
        auto nextMov = movs.begin();

        for (uint32_t isrc = 0; isrc < program.size(); isrc++, psrc++) {
            if (psrc->opcode() == Opcode::directive)
                continue;
            while ((nextSync != syncs.end()) && (nextSync->second->inum == isrc))
                *pdst++ = encodeSyncInsertion<hw>(*(nextSync++)->second);
            while ((nextMov != movs.end()) && (nextMov->second->inum == isrc))
                *pdst++ = encodeDummyMovInsertion<hw>(*(nextMov++)->second);

            if(!srcLines.empty())
                srcLines[psrc - psrc_start].address = sizeof(*pdst) * (pdst - pdst_start);
            *pdst++ = *psrc;
        }

        result.resize(reinterpret_cast<uint8_t *>(pdst) - result.data());
    }

    return result;
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType()});
    dst.fixup(hw, emod.getExecSize(), ewidth, defaultType, -1, 1);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 1);

    encodeCommon8(i, op, emod);
    i.common.accessMode = std::is_base_of<Align16Operand, D>::value;

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;

    if (dst.isIndirect())  i.binary.dstAddrImm9 = dst.getOffset() >> 9;
    if (src0.isIndirect()) i.binary.src0AddrImm9 = src0.getOffset() >> 9;

    i.binary.dstType = getTypecode<hw>(dst.getType());
    i.binary.src0Type = getTypecode<hw>(src0.getType());

    i.binary.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);
    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};

    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType()});
    dst.fixup(hw, emod.getExecSize(), ewidth, defaultType, -1, 1);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 1);

    encodeCommon12(i, op, emod, dst, tag);

    i.binary.dst  = encodeBinaryOperand12<-1>(dst, tag).bits;
    i.binary.src0 = encodeBinaryOperand12<0>(src0, tag).bits;

    i.binary.dstAddrMode = dst.isIndirect();
    i.binary.dstType  = getTypecode12(dst.getType());
    i.binary.src0Type = getTypecode12(src0.getType());

    i.binary.src0Mods = src0.getMods();

    i.binary.cmod = static_cast<int>(mod.getCMod());

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, const Immediate &src0, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType()});
    dst.fixup(hw, emod.getExecSize(), ewidth, defaultType, -1, 1);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 1);

    encodeCommon8(i, op, emod);
    i.common.accessMode = std::is_base_of<Align16Operand, D>::value;

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;

    i.binary.dstType = getTypecode<hw>(dst.getType());
    i.binary.src0Type = getImmediateTypecode<hw>(src0.getType());

    i.binary.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);

    if (dst.isIndirect())  i.binary.dstAddrImm9 = dst.getOffset() >> 9;

    if (getBytes(src0.getType()) == 8)
        i.imm64.value = static_cast<uint64_t>(src0);
    else
        i.imm32.value = static_cast<uint64_t>(src0);

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, const Immediate &src0, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};

    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType()});
    dst.fixup(hw, emod.getExecSize(), ewidth, defaultType, -1, 1);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 1);

    encodeCommon12(i, op, emod, dst, tag);

    i.binary.dst = encodeBinaryOperand12<-1>(dst, tag).bits;

    i.binary.dstAddrMode = dst.isIndirect();

    i.binary.dstType  = getTypecode12(dst.getType());
    i.binary.src0Type = getTypecode12(src0.getType());

    i.binary.src0Imm = true;

    i.binary.cmod = static_cast<int>(mod.getCMod());

    auto val = static_cast<uint64_t>(src0);
    i.imm32.value = uint32_t(val);
    if (getBytes(src0.getType()) == 8) {
#ifdef NGEN_SAFE
        if (mod.getCMod() != ConditionModifier::none) throw invalid_modifiers_exception();
#endif
        i.imm64.high = val >> 32;
    }

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, typename S1, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    Instruction8 i{};

    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType(), src1.getType()});
    dst.fixup(hw, emod.getExecSize(),  ewidth, defaultType, -1, 2);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 2);
    src1.fixup(hw, emod.getExecSize(), ewidth, defaultType, 1, 2);

    encodeCommon8(i, op, emod);
    i.common.accessMode = std::is_base_of<Align16Operand, D>::value;

    i.binary.dst  = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;
    i.binary.src1 = encodeBinaryOperand8<false>(src1).bits;

    if (dst.isIndirect())  i.binary.dstAddrImm9 = dst.getOffset() >> 9;
    if (src0.isIndirect()) i.binary.src0AddrImm9 = src0.getOffset() >> 9;
    if (src1.isIndirect()) i.binary.src1AddrImm9 = src1.getOffset() >> 9;

    i.binary.dstType  = getTypecode<hw>(dst.getType());
    i.binary.src0Type = getTypecode<hw>(src0.getType());
    i.binary.src1Type = getTypecode<hw>(src1.getType());

    i.binary.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);
    i.binary.src1RegFile = getRegFile(src1);

#ifdef NGEN_SAFE
    if (src1.isARF() && op != Opcode::illegal && op != Opcode::movi && op != Opcode::directive)
        throw grf_expected_exception();
#endif

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, typename S1, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};

    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType(), src1.getType()});
    dst.fixup(hw,  emod.getExecSize(), ewidth, defaultType, -1, 2);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 2);
    src1.fixup(hw, emod.getExecSize(), ewidth, defaultType, 1, 2);

    encodeCommon12(i, op, emod, dst, tag);

    i.binary.dst  = encodeBinaryOperand12<-1>(dst, tag).bits;
    i.binary.src0 = encodeBinaryOperand12<0>(src0, tag).bits;
    i.binary.src1 = encodeBinaryOperand12<1>(src1, tag).bits;

    i.binary.dstAddrMode = dst.isIndirect();
    i.binary.dstType  = getTypecode12(dst.getType());
    i.binary.src0Type = getTypecode12(src0.getType());
    i.binary.src1Type = getTypecode12(src1.getType());

    i.binary.src0Mods = src0.getMods();
    i.binary.src1Mods = src1.getMods();

    i.binary.cmod = static_cast<int>(mod.getCMod());

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, const Immediate &src1, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType(), src1.getType()});
    dst.fixup(hw,  emod.getExecSize(), ewidth, defaultType, -1, 2);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 2);
    src1.fixup(hw, emod.getExecSize(), ewidth, defaultType, 1, 2);

    encodeCommon8(i, op, emod);
    i.common.accessMode = std::is_base_of<Align16Operand, D>::value;

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;

    if (dst.isIndirect())  i.binary.dstAddrImm9 = dst.getOffset() >> 9;
    if (src0.isIndirect()) i.binary.src0AddrImm9 = src0.getOffset() >> 9;

    i.binary.dstType = getTypecode<hw>(dst.getType());
    i.binary.src0Type = getTypecode<hw>(src0.getType());
    i.binary.src1Type = getImmediateTypecode<hw>(src1.getType());

    i.binary.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);
    i.binary.src1RegFile = getRegFile(src1);

    i.imm32.value = static_cast<uint64_t>(src1);

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, const Immediate &src1, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};

    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType(), src1.getType()});
    dst.fixup(hw,  emod.getExecSize(), ewidth, defaultType, -1, 2);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 2);
    src1.fixup(hw, emod.getExecSize(), ewidth, defaultType, 1, 2);

    encodeCommon12(i, op, emod, dst, tag);

    i.binary.dst  = encodeBinaryOperand12<-1>(dst, tag).bits;
    i.binary.src0 = encodeBinaryOperand12<0>(src0, tag).bits;
    i.binary.src1 = static_cast<uint64_t>(src1);

    i.binary.dstAddrMode = dst.isIndirect();
    i.binary.dstType  = getTypecode12(dst.getType());
    i.binary.src0Type = getTypecode12(src0.getType());
    i.binary.src1Type = getTypecode12(src1.getType());

    i.binary.src0Mods = src0.getMods();

    i.binary.cmod = static_cast<int>(mod.getCMod());

    i.binary.src1Imm = true;
    i.imm32.value = uint32_t(static_cast<uint64_t>(src1));

    db(i);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLE(hw_, HW::Gen9)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, RegData dst, RegData src0, RegData src1, RegData src2, SourceLocation loc)
{
    opX(op, defaultType, mod, emulateAlign16Dst(dst),  emulateAlign16Src(src0),
                              emulateAlign16Src(src1), emulateAlign16Src(src2), loc);
}


template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, Align16Operand dst, Align16Operand src0, Align16Operand src1, Align16Operand src2, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

#ifdef NGEN_SAFE
    if (dst.getReg().isARF())  throw grf_expected_exception();
    if (src0.getReg().isARF()) throw grf_expected_exception();
    if (src1.getReg().isARF()) throw grf_expected_exception();
    if (src2.getReg().isARF()) throw grf_expected_exception();
#endif

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier | Align16;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType(), src1.getType(), src2.getType()});
    dst.getReg().fixup(hw,  emod.getExecSize(), ewidth, defaultType, -1, 3);
    src0.getReg().fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 3);
    src1.getReg().fixup(hw, emod.getExecSize(), ewidth, defaultType, 1, 3);
    src2.getReg().fixup(hw, emod.getExecSize(), ewidth, defaultType, 2, 3);

    encodeCommon8(i, op, emod);

    i.ternary16.dstChanEn = dst.getChanEn();
    i.ternary16.dstRegNum = dst.getReg().getBase();
    i.ternary16.dstSubregNum2_4 = dst.getReg().getByteOffset() >> 2;
    i.ternary16.dstType = getTernary16Typecode8(dst.getReg().getType());

    i.ternary16.srcType = getTernary16Typecode8(src0.getReg().getType());

    bool isFOrHF = (src0.getReg().getType() == DataType::f
                 || src0.getReg().getType() == DataType::hf);

    i.ternary16.src1Type = isFOrHF && (src1.getReg().getType() == DataType::hf);
    i.ternary16.src2Type = isFOrHF && (src1.getReg().getType() == DataType::hf);

    encodeTernaryCommon8(i, src0, src1, src2);

    db(i);
}

template <HW hw>
template <typename D, typename S0, typename S1, typename S2, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2, SourceLocation loc)
{
    if (hw < HW::Gen10)
        unsupported();

    debugLine.add(rootStream.length(), loc);

#ifdef NGEN_SAFE
    if (src0.isARF()) throw grf_expected_exception();
    if (src2.isARF()) throw grf_expected_exception();
#endif

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType(), src1.getType(), src2.getType()});
    dst.fixup(hw,  emod.getExecSize(), ewidth, defaultType, -1, 3);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 3);
    src1.fixup(hw, emod.getExecSize(), ewidth, defaultType, 1, 3);
    src2.fixup(hw, emod.getExecSize(), ewidth, defaultType, 2, 3);

    encodeCommon8(i, op, emod);

    i.ternary1.src0RegFile = std::is_base_of<Immediate, S0>::value;
    i.ternary1.src1RegFile = src1.isARF();
    i.ternary1.src2RegFile = std::is_base_of<Immediate, S2>::value;

    encodeTernaryCommon8(i, src0, src1, src2);
    encodeTernary1Dst10(i, dst);

    db(i);
}

template <HW hw>
template <typename D, typename S0,typename S1, typename S2, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType(), src1.getType(), src2.getType()});
    dst.fixup(hw,  emod.getExecSize(), ewidth, defaultType, -1, 3);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 3);
    src1.fixup(hw, emod.getExecSize(), ewidth, defaultType, 1, 3);
    src2.fixup(hw, emod.getExecSize(), ewidth, defaultType, 2, 3);

    encodeCommon12(i, op, emod, dst, tag);

    i.ternary.dst  = encodeTernaryOperand12<true>(dst, tag).bits;
    encodeTernarySrc0(i, src0, tag);
    encodeTernarySrc1(i, src1, tag);
    encodeTernarySrc2(i, src2, tag);
    encodeTernaryTypes(i, dst, src0, src1, src2);

    i.ternary.cmod = static_cast<int>(mod.getCMod());

    db(i);
}

template <HW hw>
template <typename DS0>
void BinaryCodeGenerator<hw>::opMath(Opcode op, DataType defaultType, const InstructionModifier &mod, MathFunction fc, DS0 dst, DS0 src0, SourceLocation loc)
{
    InstructionModifier mmod = mod;

    mmod.setCMod(static_cast<ConditionModifier>(fc));
    opX(op, defaultType, mmod, dst, src0, loc);
}

template <HW hw>
template <typename DS0, typename S1>
void BinaryCodeGenerator<hw>::opMath(Opcode op, DataType defaultType, const InstructionModifier &mod, MathFunction fc, DS0 dst, DS0 src0, S1 src1, SourceLocation loc)
{
    InstructionModifier mmod = mod;

    mmod.setCMod(static_cast<ConditionModifier>(fc));
    opX(op, defaultType, mmod, dst, src0, src1, loc);
}

template <HW hw>
template <typename D, typename S0, typename S2>
void BinaryCodeGenerator<hw>::opBfn(Opcode op, DataType defaultType, const InstructionModifier &mod, int bfnCtrl, D dst, S0 src0, RegData src1, S2 src2, SourceLocation loc)
{
    if (hw < HW::XeHP)
        unsupported();

    debugLine.add(rootStream.length(), loc);

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType(), src1.getType(), src2.getType()});
    dst.fixup(hw,  emod.getExecSize(), ewidth, defaultType, -1, 3);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 3);
    src1.fixup(hw, emod.getExecSize(), ewidth, defaultType, 1, 3);
    src2.fixup(hw, emod.getExecSize(), ewidth, defaultType, 2, 3);

    encodeCommon12(i, op, emod, dst, tag);

    i.ternary.dst  = encodeTernaryOperand12<true>(dst, tag).bits;
    encodeTernarySrc0(i, src0, tag);
    encodeTernarySrc1(i, src1, tag);
    encodeTernarySrc2(i, src2, tag);
    encodeTernaryTypes(i, dst, src0, src1, src2);

    i.ternary.cmod = static_cast<int>(mod.getCMod());

    i.bfn.bfnCtrl03 = (bfnCtrl >> 0);
    i.bfn.bfnCtrl47 = (bfnCtrl >> 4);

    db(i);
}

template <HW hw>
static inline void encodeDPAS(Instruction12 &i, Opcode op, DataType defaultType, const InstructionModifier &emod, int sdepth, int rcount, RegData dst, RegData src0, RegData src1, RegData src2)
{
    typename EncodingTag12Dispatch<hw>::tag tag;

    dst.fixup(hw, emod.getExecSize(), 0, defaultType, -1, 3);
    src0.fixup(hw, emod.getExecSize(), 0, defaultType, 0, 3);
    src1.fixup(hw, emod.getExecSize(), 0, defaultType, 1, 3);
    src2.fixup(hw, emod.getExecSize(), 0, defaultType, 2, 3);

    encodeCommon12(i, op, emod, dst, tag);

    i.ternary.dst  = encodeTernaryOperand12<true,  false>(dst,  tag).bits;
    i.ternary.src0 = encodeTernaryOperand12<false, false>(src0, tag).bits;
    i.ternary.src1 = encodeTernaryOperand12<false, false>(src1, tag).bits;
    i.ternary.src2 = encodeTernaryOperand12<false, false>(src2, tag).bits;

    encodeTernaryTypes(i, dst, src0, src1, src2);

    i.dpas.rcount = rcount - 1;
    i.dpas.sdepth = utils::log2(sdepth);

    i.dpas.src1SubBytePrecision = encodeSubBytePrecision12(src1.getType());
    i.dpas.src2SubBytePrecision = encodeSubBytePrecision12(src2.getType());

    i.ternary.cmod = static_cast<int>(emod.getCMod());
}

template <HW hw>
void BinaryCodeGenerator<hw>::opDpas(Opcode op, DataType defaultType, const InstructionModifier &mod, int sdepth, int rcount, RegData dst, RegData src0, RegData src1, RegData src2, SourceLocation loc)
{
    if (hw < HW::XeHP)
        unsupported();

    debugLine.add(rootStream.length(), loc);

    Instruction12 i{};
    encodeDPAS<hw>(i, op, defaultType, mod | defaultModifier, sdepth, rcount, dst, src0, src1, src2);
    db(i);
}

template <HW hw>
template <typename D, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, int src1Length, uint32_t exdesc, D desc, SourceLocation loc)
{
    exdesc |= uint32_t(static_cast<uint8_t>(sfid));
    opSends(static_cast<Opcode>(static_cast<uint8_t>(op) | 2), mod, dst, src0, src1, exdesc, desc, loc);
}

template <HW hw>
template <typename D, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, int src1Length, const RegData &exdesc, D desc, SourceLocation loc)
{
    opSends(static_cast<Opcode>(static_cast<uint8_t>(op) | 2), mod, dst, src0, src1, exdesc, desc, loc);
}

template <HW hw>
template <typename ED, typename D, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0_, const RegData &src1, int src1Length, ED exdesc, D desc, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    auto src0 = src0_;
    const bool src0Indirect = (hw >= HW::Xe3 && src0.isIndirect());
    if (src0Indirect)
        src0 = src0.getIndirectReg();

    encodeCommon12(i, op, emod, dst, tag);

    i.send.fusionCtrl = emod.isSerialized();

    i.send.dstReg = dst.getBase();
    i.send.src0Reg = src0.getBase();
    i.send.src1Reg = src1.getBase();

    i.send.dstRegFile = getRegFile(dst);
    i.send.src0RegFile = getRegFile(src0);
    i.send.src1RegFile = getRegFile(src1);

    i.send.sfid = static_cast<int>(sfid) & 0xF;

    if (src1.isNull())
        src1Length = 0;

    encodeSendDesc(i, desc);
    encodeSendExDesc(i, exdesc, mod, src1Length, hw);

    if (src0Indirect)
        i.send.exDesc6_10 = src0.getOffset() >> 1;

    db(i);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon8(i, op, emod);

    i.binary.dst  = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;

    i.sendsGen9.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);
    i.binary.src1RegFile = RegFileIMM;

    i.binary.dstType = getTypecode<hw>(dst.getType());

    i.sendsGen9.sfid = exdesc & 0xF;
    i.sendGen8.zero = 0;
    i.sendGen8.exDesc16_19 = (exdesc >> 16) & 0xF;
    i.sendGen8.exDesc20_23 = (exdesc >> 20) & 0xF;
    i.sendGen8.exDesc24_27 = (exdesc >> 24) & 0xF;
    i.sendGen8.exDesc28_31 = (exdesc >> 28) & 0xF;
    i.sendsGen9.desc = desc;

    i.sendsGen9.eot = (exdesc >> 5) & 1;
    if (dst.isIndirect()) i.sendsGen9.dstAddrImm9 = dst.getOffset() >> 9;

    db(i);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

#ifdef NGEN_SAFE
    // Only a0.0:ud is allowed for desc.
    if (!desc.isARF() || desc.getARFType() != ARFType::a || desc.getARFBase() != 0 || desc.getOffset() != 0)
        throw invalid_arf_exception();
#endif
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;
    i.binary.src1 = encodeBinaryOperand8<false>(desc).bits;

    i.sendsGen9.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);
    i.binary.src1RegFile = getRegFile(desc);
    i.binary.src1Type = getTypecode<hw>(desc.getType());

    i.sendsGen9.sfid = exdesc & 0xF;
    i.sendGen8.zero = 0;
    i.sendGen8.exDesc16_19 = (exdesc >> 16) & 0xF;
    i.sendGen8.exDesc20_23 = (exdesc >> 20) & 0xF;
    i.sendGen8.exDesc24_27 = (exdesc >> 24) & 0xF;
    i.sendGen8.exDesc28_31 = (exdesc >> 28) & 0xF;

    i.sendsGen9.eot = (exdesc >> 5) & 1;
    if (dst.isIndirect()) i.sendsGen9.dstAddrImm9 = dst.getOffset() >> 9;

    db(i);
}

template <HW hw>
template <typename D, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, D desc, SourceLocation loc)
{
    opSends(op, mod, dst, src0, null, exdesc, desc, loc);
}

template <HW hw>
template <typename ED, typename D, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, ED exdesc, D desc, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;

    i.binary.src0RegFile = 0;                   // ?
    i.sendsGen9.dstRegFile = getRegFile(dst);
    i.sendsGen9.src1RegFile = getRegFile(src1);
    i.sendsGen9.src1RegNum = src1.getBase();

    if (dst.isIndirect())  i.sendsGen9.dstAddrImm9  =  dst.getOffset() >> 9;
    if (src0.isIndirect()) i.sendsGen9.src0AddrImm9 = src0.getOffset() >> 9;

    encodeSendsDesc(i, desc);
    encodeSendsExDesc(i, exdesc);

    db(i);
}

template <HW hw>
template <typename D, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, RegData exdesc, D desc, SourceLocation loc)
{
#ifdef NGEN_SAFE
    throw sfid_needed_exception();
#endif
}

template <HW hw>
template <typename D, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, D desc, SourceLocation loc)
{
    Opcode mop = static_cast<Opcode>(static_cast<int>(op) & ~2);
    opSend(mop, mod, static_cast<SharedFunction>(exdesc & 0x1F), dst, src0, src1, -1, exdesc, desc, loc);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, int32_t uip, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.dstRegFile = getRegFile(dst);
    i.binary.dstType = getTypecode<hw>(dst.getType());
    i.binary.src0RegFile = getRegFile(Immediate());
    i.binary.src0Type = getTypecode<hw>(DataType::d);
    i.branches.jip = jip;
    i.branches.uip = uip;

    db(i);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, int32_t uip, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon12(i, op, emod, dst, tag);

    i.branches.branchCtrl = emod.getBranchCtrl();

    i.binary.dst = encodeBinaryOperand12<-1, false>(dst, tag).bits;

    i.binary.src0Imm = true;
    i.binary.src1Imm = true;

    i.branches.jip = jip;
    i.branches.uip = uip;

    db(i);
}

template <HW hw>
template <bool forceWE, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.dstRegFile = getRegFile(dst);
    i.binary.dstType = getTypecode<hw>(dst.getType());
    i.binary.src1RegFile = RegFileIMM;
    i.binary.src1Type = getTypecode<hw>(DataType::d);
    i.branches.jip = jip;

    db(i);
}

template <HW hw>
template <bool forceWE, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    encodeCommon12(i, op, emod, dst, tag);

    i.branches.branchCtrl = emod.getBranchCtrl();

    i.binary.dst = encodeBinaryOperand12<-1, false>(dst, tag).bits;
    i.binary.src0Imm = true;
    i.branches.jip = jip;

    db(i);
}

template <HW hw>
template <bool forceWE, bool small12, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.dstRegFile = getRegFile(dst);
    i.binary.dstType = getTypecode<hw>(DataType::d);
    i.binary.src0RegFile = getRegFile(src0);
    i.binary.src0Type = getTypecode<hw>(DataType::d);
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;

    db(i);
}

template <HW hw>
template <bool forceWE, bool small12, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    encodeCommon12(i, op, emod, dst, tag);

    i.branches.branchCtrl = emod.getBranchCtrl();

    i.binary.dst = encodeBinaryOperand12<-1, false>(dst, tag).bits;
    i.binary.src0 = encodeBinaryOperand12<0, false>(src0, tag).bits;
    if (small12)
        i.binary.src0 &= 0xFFFF;


    db(i);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip, Label &uip, SourceLocation loc)
{
    addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffset));
    addFixup(LabelFixup(uip.getID(labelManager), LabelFixup::UIPOffset));
    opBranch(op, mod, dst, 0, 0, loc);
}

template <HW hw>
template <bool forceWE>
void BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip, SourceLocation loc)
{
    addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffset));
    opBranch<forceWE>(op, mod, dst, 0, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opCall(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip, SourceLocation loc)
{
    addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffset));
    if (isGen12)
        opBranch<true>(op, mod, dst, 0, loc);
    else
        opX<true>(op, DataType::d, mod, dst, null.ud(0)(0, 1, 0), Immediate::d(0), loc);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, RegData src0, uint32_t jip, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier | NoMask;

    encodeCommon8(i, op, emod);

    src0.fixup(hw, emod.getExecSize(), 0, DataType::d, 0, 2);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;
    i.binary.src0RegFile = getRegFile(src0);
    i.binary.src1RegFile = RegFileIMM;
    i.binary.src1Type = getTypecode<hw>(DataType::d);

    i.branches.jip = jip;

    db(i);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, RegData src0, uint32_t jip, SourceLocation loc)
{
    opBranch<true>(op, mod, dst, jip, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, Label &jip, SourceLocation loc)
{
    if (hw >= HW::Gen12LP)
        addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffset));
    opJmpi(op, mod, dst, src0, 0, loc);
    if (hw < HW::Gen12LP)
        addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffsetJMPI));
}

template <HW hw>
void BinaryCodeGenerator<hw>::opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, SourceLocation loc)
{
    if (hw < HW::Gen12LP)
        unsupported();

    debugLine.add(rootStream.length(), loc);

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon12(i, op, emod, null, tag);

    i.binary.dst = 0x1;
    i.binary.cmod = static_cast<int>(fc);

    db(i);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, RegData src0, SourceLocation loc)
{
    typename EncodingTag12Dispatch<hw>::tag tag;
    if (hw < HW::Gen12LP)
        unsupported();

    debugLine.add(rootStream.length(), loc);

    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon12(i, op, emod, null, tag);

    i.binary.dst = 0x1;
    if (!src0.isNull()) {
        src0.setRegion(0, 1, 0);
        i.binary.src0 = encodeBinaryOperand12<0>(src0, tag).bits;
        i.binary.src0Type = getTypecode12(src0.getType());
    }
    i.binary.cmod = static_cast<int>(fc);

    db(i);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, const Immediate &src0, SourceLocation loc)
{
    if (hw < HW::Gen12LP)
        unsupported();

    debugLine.add(rootStream.length(), loc);

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon12(i, op, emod, null, tag);

    i.binary.dst = 0x1;
    i.binary.src0Type = getTypecode12(src0.getType());
    i.binary.src0Imm = true;
    i.binary.cmod = static_cast<int>(fc);

    i.imm32.value = uint32_t(static_cast<uint64_t>(src0));

    db(i);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opNop(Opcode op, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    Instruction8 i{};

    i.qword[0] = static_cast<int>(op);
    i.qword[1] = 0;

    db(i);
}

} /* namespace NGEN_NAMESPACE */

#ifdef ENABLE_LLVM_WCONVERSION
#pragma clang diagnostic pop
#endif

#endif /* header guard */
