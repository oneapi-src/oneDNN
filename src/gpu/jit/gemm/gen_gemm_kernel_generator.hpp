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

#ifndef GPU_JIT_GEMM_GEN_GEMM_KERNEL_GENERATOR_HPP
#define GPU_JIT_GEMM_GEN_GEMM_KERNEL_GENERATOR_HPP

/* Embargo support */

#define STANDALONE 0

#include "common/math_utils.hpp"
#include "common/utils.hpp"
#include "gpu/jit/gemm/gen_gemm_kernel_common.hpp"
#include "gpu/jit/gemm/utils.hpp"
#include "gpu/jit/jit_generator.hpp"
#include "gpu/jit/jit_post_op_injector.hpp"
#include "gpu/serialization.hpp"

#if defined(ZEBIN_OUTPUT)
#include "../ngen/ngen_elf.hpp"
#else
#include "../ngen/ngen_opencl.hpp"

#endif
#include "../ngen/ngen_register_allocator.hpp"

#include "gpu/jit/gemm/emulation.hpp"

#include <array>
#include <complex>
#include <cstdint>
#include <exception>
#include <iostream>
#include <sstream>
#include <vector>

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

struct RegisterBlock;

class Type {
public:
    enum _Type : uint32_t {
        invalid = 0,
        f16 = 0x01000201,
        f32 = 0x01010402,
        u8 = 0x01840100,
        s8 = 0x01850100,
        u16 = 0x01860201,
        s16 = 0x01870201,
        u32 = 0x01880402,
        s32 = 0x01890402,
        u64 = 0x018A0803,
        s64 = 0x018B0803,
        bf16 = 0x010C0201,
        tf32 = 0x010D0402,
    };

private:
    _Type val;

public:
    constexpr Type() : Type(f32) {}
    constexpr Type(_Type val_) : val(val_) {}
    constexpr operator _Type() const { return val; }

    constexpr Type real() const { return *this; }
    constexpr bool isComplex() const { return false; }
    constexpr int complexComponents() const { return 1; }
    constexpr int components() const { return 1; }
    constexpr bool isInteger() const { return uint32_t(val) & 0x800000; }
    constexpr bool isFP() const { return !isInteger(); }
    constexpr bool isSigned() const {
        return (uint32_t(val) & 0x810000) != 0x800000;
    }
    constexpr int log2Size() const { return uint32_t(val) & 0xFF; }
    constexpr int size() const { return (uint32_t(val) >> 8) & 0xFF; }

    constexpr Type arithmetic() const {
        return (val == tf32) ? Type(f32) : real();
    }
    data_type_t get_dnnl_type() const {
        switch (val) {
            case Type::f32: return data_type::f32;
            case Type::f16: return data_type::f16;
            case Type::s32: return data_type::s32;
            case Type::u8: return data_type::u8;
            case Type::s8: return data_type::s8;
            default: assert(!"Unsupported type"); return data_type::undef;
        }
    }
    constexpr Type baseType() const { return *this; }

    template <typename U>
    constexpr friend int operator*(U a, Type t) {
        return int(a << t.log2Size());
    }
    template <typename U>
    constexpr friend int operator/(U a, Type t) {
        return int(a >> t.log2Size());
    }

    ngen::DataType ngen() const {
        using namespace ngen;
        static const DataType table[16] = {DataType::hf, DataType::f,
                DataType::df, DataType::invalid, DataType::ub, DataType::b,
                DataType::uw, DataType::w, DataType::ud, DataType::d,
                DataType::uq, DataType::q, DataType::bf, DataType::tf32,
                DataType::invalid, DataType::invalid};
        return table[(uint32_t(val) >> 16) & 0xF];
    }

    bool isSubsetOf(Type T) const;
};

enum class MatrixLayout : uint8_t {
    N = 0,
    Nontranspose = 0,
    T = 1,
    Transpose = 1,
    Pc = 2,
    PackedColumns = 2,
    Pr = 3,
    PackedRows = 3
};

static inline bool isPacked(MatrixLayout l) {
    return (l == MatrixLayout::PackedRows)
            || (l == MatrixLayout::PackedColumns);
}

static inline bool isColMajor(MatrixLayout l) {
    return (l == MatrixLayout::N || l == MatrixLayout::Pc);
}

static inline bool isLargeCrosspack(int sizeofT, int crosspack) {
    return (crosspack * sizeofT > 4) && (crosspack > 1);
}

static inline bool isLargeCrosspack(Type T, int crosspack) {
    return isLargeCrosspack(T.size(), crosspack);
}

static inline MatrixLayout transposeLayout(MatrixLayout l) {
    return static_cast<MatrixLayout>(static_cast<uint8_t>(l) ^ 0x1);
}

// Information on scalar arguments (alpha/beta)
class Scalar {
public:
    enum ScalarType { Fixed, Variable, Pointer, RealPointer };

private:
    int value;
    ScalarType type;

public:
    Scalar() : Scalar(Variable) {}
    explicit Scalar(ScalarType type_) : value(0), type(type_) {}
    explicit Scalar(int value_) : value(value_), type(Fixed) {}

    Scalar &operator=(int value_) {
        type = Fixed;
        value = value_;
        return *this;
    }
    Scalar &operator=(ScalarType type_) {
        type = type_;
        value = 0;
        return *this;
    }

    template <typename U>
    bool operator==(U value_) const {
        return fixed() && (value == value_);
    }
    bool operator==(ScalarType type_) const { return (type == type_); }
    template <typename U>
    bool operator!=(U value_) const {
        return !operator==(value_);
    }

    operator int() const {
        if (!fixed()) throw std::runtime_error("Scalar is not fixed.");
        return value;
    }
    operator double() const { return int(*this); }

    bool fixed() const { return (type == Fixed); }
    bool pointer() const { return (type == Pointer) || (type == RealPointer); }
    ScalarType getType() const { return type; }
};

enum class AccessType : uint8_t {
    Scattered, // Use scattered accesses
    ChannelScattered, // Use untyped surface reads
    Block, // Use block messages
    PseudoBlock, // Use scattered accesses to emulate block accesses
    Block2D, // Use 2D block messages
    Block2DTranspose, // Use 2D block messages with transposition
    Block2DVNNI, // Use 2D block messages with VNNI transform
};

static inline bool isBlocklike(AccessType t) {
    return (t == AccessType::Block || t == AccessType::PseudoBlock);
}

static inline bool isBlock2D(AccessType t) {
    return (t == AccessType::Block2D || t == AccessType::Block2DTranspose
            || t == AccessType::Block2DVNNI);
}

enum class RemainderHandling : uint8_t {
    Ignore, // Assume no remainder, or handled by hardware bounds checking.
    General, // Handle all remainder cases.
    Split, // Generate copies of the kernel with and without remainder handling.
    KnownRemainder, // Assume remainder case; don't create special code for non-remainder case.
};

enum class KernelScheduling : uint8_t {
    Static,
    EUStatic,
    Dynamic,
};

// Preferences for remainder handling.
enum RemainderOptions : uint8_t {
    AvoidFragment
    = 0, // Avoid/allow making blocks that will need to be broken up
    AllowFragment = 1, //  ("fragmented") during remainder handling.
    AllowDescriptors
    = 2, // Allow indirect send descriptor-based remainder handling.
    AllowFragDesc = 3, // Allow fragmentation and descriptors.
};

// Preferences for using scattered accesses.
enum class ScatterSIMD : uint8_t {
    Default,
    Wide, // Prefer wider SIMD (more scattered lanes)
    Narrow // Prefer narrower SIMD (more consecutive access)
};

struct GRFMultirange {
    std::vector<ngen::GRFRange> ranges;

    GRFMultirange() {}
    GRFMultirange(ngen::GRFRange range) : ranges {1, range} {}

    ngen::GRF operator[](int idx) const {
        for (auto &r : ranges) {
            if (idx < r.getLen()) return r[idx];
            idx -= r.getLen();
        }
        throw std::runtime_error("Index out of bounds");
    }

    ngen::Subregister sub(ngen::HW hw, int offset, ngen::DataType type) const {
        const int lg2Len = ngen::GRF::log2Bytes(hw) - ngen::getLog2Bytes(type);
        return (*this)[offset >> lg2Len].sub(
                offset - ((offset >> lg2Len) << lg2Len), type);
    }

    GRFMultirange subrange(int start, int count) const {
        GRFMultirange result;
        for (auto &r : ranges) {
            if (start < r.getLen()) {
                auto got = std::min(count, r.getLen() - start);
                result.ranges.push_back(
                        ngen::GRFRange {r.getBase() + start, got});
                count -= got;
                start = 0;
                if (count <= 0) break;
            } else
                start -= r.getLen();
        }
        return result;
    }

    GRFMultirange subrange(
            ngen::HW hw, Type T, const RegisterBlock &block) const;

    bool contiguous(int start, int count) const {
        for (auto &r : ranges) {
            if (start < r.getLen()) return (start + count) <= r.getLen();
            start -= r.getLen();
        }
        return false;
    }

    void append(ngen::GRF r) { append(r - r); }

    void append(ngen::GRFRange r) {
        if (!ranges.empty()) {
            auto &rend = ranges.back();
            if (rend.getBase() + rend.getLen() == r.getBase()) {
                rend = ngen::GRFRange(
                        rend.getBase(), rend.getLen() + r.getLen());
                return;
            }
        }
        ranges.push_back(r);
    }

    void append(const GRFMultirange &r) {
        for (auto &rr : r.ranges)
            append(rr);
    }

    uint8_t getLen() const {
        uint8_t len = 0;
        for (auto &r : ranges)
            len += r.getLen();
        return len;
    }

    bool empty() const {
        for (auto &r : ranges)
            if (r.getLen() > 0) return false;
        return true;
    }
    void clear() { ranges.clear(); }
};

// A pair of Subregisters in opposite banks.
class SubregisterPair {
protected:
    ngen::Subregister regs[2];
    bool negative;

public:
    SubregisterPair() : SubregisterPair(ngen::Subregister()) {}
    SubregisterPair(ngen::Subregister reg0, ngen::Subregister reg1)
        : regs {reg0, reg1}, negative(false) {}
    explicit SubregisterPair(ngen::Subregister reg)
        : SubregisterPair(reg, reg) {}

    void serialize(serialized_data_t &s) const {
        s.append(regs);
        s.append(negative);
    }
    /* implicit */ operator ngen::Subregister() const { return regs[0]; }

    SubregisterPair &operator=(ngen::Subregister reg) {
        regs[0] = regs[1] = reg;
        negative = false;
        return *this;
    }

    ngen::Subregister getReg(int idx) const;
    ngen::Subregister getRegAvoiding(
            ngen::HW hw, const ngen::RegData &rd) const;

    bool isValid() const { return regs[0].isValid() && regs[1].isValid(); }
    bool isInvalid() const { return !isValid(); }
    void invalidate() {
        regs[0].invalidate();
        regs[1].invalidate();
    }

    bool isDuplicated() const { return regs[0] != regs[1]; }

    SubregisterPair operator-() const {
        auto copy = *this;
        copy.negative = !copy.negative;
        return copy;
    }
};

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
        ngen::RegData sub = ngen::Subregister {};
        if (shift >= 0 && shift <= maxShift) sub = regs[shift];
        if (neg) sub = -sub;
        return *reinterpret_cast<ngen::Subregister *>(&sub);
    }

    void set(int shift, ngen::Subregister reg) { regs[shift] = reg; }
};

struct MatrixAddressing {
    MatrixLayout layout; // Layout type (N/T/Pr/Pc)
    uint8_t packSize; // # of elements in a packed row/column for packed layouts.
    uint8_t crosspack; // Crosspack for packed layouts.
    uint8_t alignment; // Alignment for all addresses, offsets, and leading dimensions.
    uint8_t tileR = 0, tileC = 0; // Tiling (0 if none) for packed layouts.
    uint8_t panelLength
            = 0; // Length of the panel for packed layouts = #cols/rows for Pc/Pr respectively.

    void setAlignment(int align) { alignment = sanitizeAlign(align); }
    int defaultAlignment(Type T) const {
        return sanitizeAlign(
                T.size() * (isPacked(layout) ? (packSize * crosspack) : 1));
    }

private:
    static int sanitizeAlign(int align) {
        return std::min(128, largest_pow2_divisor(align));
    }
};

struct MatrixAddressingStrategy {
    ngen::AddressBase base; // Base for addressing (A64/BTS/...)
    AccessType accessType = AccessType::Block; // Block/scattered/etc. access
    uint8_t tileR = 0, tileC = 0; // Desired tiling (0 if none) in registers.
    ScatterSIMD smode
            = ScatterSIMD::Default; // SIMD selection for scattered accesses.
    uint8_t padded : 1; // Allow read/write overruns?
    uint8_t atomic : 1; // Atomic access? (only relevant for C)
    uint8_t address2D : 1; // Use 2D addressing? (media block-style loads)
    uint8_t prefetch : 1; // Prefetch only?
    uint8_t newDP : 1; // Use new dataport messages? (XeHPG+)
    uint8_t dpasw : 1; // DPASW half layout?
    uint8_t noExtraPad : 1; // Avoid extra padding?
    uint8_t bitfield_padding : 1;
    ngen::CacheSettingsLSC cachingR // Cache policies for LSC reads.
            = ngen::CacheSettingsLSC::Default;
    ngen::CacheSettingsLSC cachingW // Cache policies for LSC writes.
            = ngen::CacheSettingsLSC::Default;
    uint8_t pad0[1] = {};

    MatrixAddressingStrategy()
        : padded(false)
        , atomic(false)
        , address2D(false)
        , prefetch(false)
        , newDP(false)
        , dpasw(false)
        , noExtraPad(false)
        , bitfield_padding(false) {}

    void preflight(ngen::HW hw);
    void forceA64();

    ngen::GlobalAccessType getGlobalAccessType() const {
        return base.isStateless() ? ngen::GlobalAccessType::Stateless
                                  : ngen::GlobalAccessType::Surface;
    }
};

struct VirtualFlag {
    uint8_t idx : 6;
    uint8_t n : 2;

    constexpr VirtualFlag() : idx(0), n(0) {}
    /* implicit */ VirtualFlag(const ngen::FlagRegister &flag)
        : idx(flag.index()), n(flag.getBytes() >> 1) {}
    explicit constexpr VirtualFlag(int idx_, int n_ = 1) : idx(idx_), n(n_) {}

    ngen::FlagRegister toPhysical() const;

    friend inline bool operator==(VirtualFlag vf1, VirtualFlag vf2) {
        return vf1.idx == vf2.idx && vf1.n == vf2.n;
    }
    friend inline bool operator!=(VirtualFlag vf1, VirtualFlag vf2) {
        return !(vf1 == vf2);
    }

    bool operator!() const { return (idx == 0) && (n == 0); }
    explicit operator bool() const { return !!*this; }

    void clear() { *this = VirtualFlag(); }

    int getBytes() const { return n << 1; }
};

struct MaskInfo {
    union {
        struct {
            uint8_t isFixed : 1; // = false (variable mask)
            uint8_t reverse : 1; // True to reverse mask.
            uint8_t rdivide : 6; // Amount by which to divide index before forming mask. Fractions are rounded up.
                    // Note maskRep * bitRep * (rsize >> rshift) = # mask bits.
            uint8_t rsize; // Maximum remainder value. (e.g. 16 if we need the last 4 bits of the index).
            uint8_t maskRep; // # of repetitions of mask pattern.
            uint8_t bitRep; // # of times each mask bit is repeated.
        } variable;
        struct {
            uint8_t isFixed : 1; // = true (fixed mask)
            uint8_t _ : 7;
            uint8_t rsize; // Maximum remainder value.
            uint16_t value; // Mask value.
        } fixed;
        uint32_t raw;
    };

    MaskInfo() : fixed {true, 0, 0, 0xFFFF} {}

    bool operator!() const { return fixed.isFixed && fixed.value == 0xFFFF; }
    explicit operator bool() const { return !!*this; }

    static MaskInfo None() { return MaskInfo(); }

    friend bool operator==(const MaskInfo &i1, const MaskInfo &i2) {
        return i1.raw == i2.raw;
    }
    friend bool operator!=(const MaskInfo &i1, const MaskInfo &i2) {
        return !(i1 == i2);
    }
};

struct MaskAssignment {
    MaskInfo mask; // Associated mask
    LoopType var; // Variable to base mask off of
    uint8_t offset; // Amount to subtract from variable.
    VirtualFlag flag; // Index of virtual flag register to use.

    bool compatible(const MaskAssignment &other) const {
        return mask == other.mask && var == other.var && offset == other.offset;
    }
    void reverse(int width) {
        offset = width - offset - mask.variable.rsize;
        mask.variable.reverse = !mask.variable.reverse;
    }
};

struct RegisterBlock {
    /* Register layout information. */
    uint8_t nr, nc; // Size of this block.
    uint8_t ld; // Leading dimension, in elements.
    uint8_t offsetR, offsetC; // Row and column offset within matrix block.
    uint8_t colMajor : 1; // Is this block column-major? (columns stored consecutively inside each register)
    uint8_t splitComplex : 1; // True if complex data split into successive real and imaginary parts.
    uint8_t : 6;
    uint8_t crosspack; // Crosspack for this block (1 if none).
    uint8_t component : 6; // Component # for this block.
    int8_t cxComponent : 2; // Complex component # for this block (-1 if not complex or interleaved).
    uint16_t bytes; // # of bytes in this block.
    uint16_t offsetBytes; // Byte offset within register block.

    /* Load/store information. */
    uint8_t remainderR : 1; // Row remaindering enabled?
    uint8_t remainderC : 1; // Column remaindering enabled?
    uint8_t noRowsOK : 1; // Can handle no rows (in mask/descriptor)?
    uint8_t noColsOK : 1; // Can handle no columns (in mask/descriptor)?
    uint8_t descRemR : 1; // Row remainders can be handled by changing the descriptor?
    uint8_t descRemC : 1; // Column remainders can be handled by changing the descriptor?
    uint8_t descAssigned : 1; // True if address registers have been assigned for this block's descriptors.
    uint8_t writable : 1; // True if block is set up for writing.

    uint8_t ebytes; // Size of element in bytes, e.g. 4 for scattered_dword, 16 for block_hword
    uint8_t count; // Element count.
    uint8_t extra; // Extra info. For block accesses, 1 means aligned OWord, 0 unaligned. For scattered accesses, # of consecutive elements.
    uint8_t simdSize; // SIMD size for load/stores (0 indicating no separate load/store needs to be done.)
    uint8_t msgRegs; // Underlying register count for load/store operation (may be different from nregs()).
    std::array<VirtualFlag, 2> flag;
    // Assigned flag register indices ([0] -> row, [1] -> column)
    uint8_t flagAny : 1; // Use .anyh?
    uint8_t flagAll : 1; // Use .allh?
    uint8_t flagInvert : 1; // Invert flag?
    uint8_t hasNoLoad : 1; // Does this load/store cover additional (no-load) RegisterBlocks? (packed layouts)
    uint8_t : 4;
    uint8_t sfid; // SFID for this block.
    uint8_t rowFragment; // If this block needs fragmenting to support row/column remainders, the maximum block size (power of 2) to fragment down to.
    uint8_t colFragment; //     Zero if no fragmenting needed.
    uint8_t addrShift; // log2(address units). e.g. 0 if byte addresses should be used, 4 if oword addresses should be used.
    uint8_t log2GRFBytes; // log2(bytes per GRF).

    MaskInfo rowMask; // Row mask for this block.
    MaskInfo colMask; // Column mask for this block.

    static constexpr int8_t Interleaved
            = -1; // Value for cxComponent indicating interleaved real/imaginary data.

    void calcBytes(Type T); // Auto-calculate # of registers.
    void calcBytes(Type T, const MatrixAddressingStrategy &astrategy);

    bool hasFlag() const { return flag[0] || flag[1]; }
    void clearFlag() {
        flag[0].clear();
        flag[1].clear();
        flagAll = flagAny = flagInvert = false;
    }
    void eraseMask() {
        clearFlag();
        rowMask = MaskInfo();
        colMask = MaskInfo();
    }

    bool isLoadBlock() const { return simdSize > 0; }

    int nregs() const;
    int offsetReg() const;

    void simplify(Type T);
    void compact(Type T);
};

struct Address2DParams {
    ngen::Subregister rows, cols;
    ngen::Subregister offR, offC;
    ngen::Subregister remR, remC;
    int fixedRows = 0, fixedCols = 0;
};

class VirtualFlagAllocator {
public:
    VirtualFlagAllocator(ngen::HW hw)
        : free(~uint64_t(0)), nflag(ngen::FlagRegister::subcount(hw)) {}

    VirtualFlag allocVirtual(int n = 1);
    ngen::FlagRegister alloc(int n = 1);
    ngen::FlagRegister allocSubreg0();
    ngen::FlagRegister tryAlloc(int n = 1);

    void claim(VirtualFlag vflag) { free &= ~mask(vflag); }
    void release(VirtualFlag vflag) { free |= mask(vflag); }
    void release(const ngen::FlagRegister &reg) {
        release(VirtualFlag(reg));
        unlock(reg);
    }
    void safeRelease(VirtualFlag &vflag) {
        if (vflag) release(vflag);
        vflag.clear();
    }
    void safeRelease(ngen::FlagRegister &reg) {
        if (reg.isValid()) release(reg);
        reg.invalidate();
    }

    bool isVirtual(VirtualFlag vflag) { return (vflag.idx >= nflag); }

    bool lock(VirtualFlag vflag) {
        bool wasLocked = isLocked(vflag);
        locked |= mask(vflag);
        return wasLocked;
    }
    void unlock(VirtualFlag vflag) { locked &= ~mask(vflag); }
    bool isLocked(VirtualFlag vflag) const { return !(~locked & mask(vflag)); }
    bool canLock(int n = 1) const;

    ngen::FlagRegister assignPhysical(VirtualFlag vflag);

protected:
    uint64_t free;
    uint8_t locked = 0;
    uint8_t nextPhys = 0;
    uint8_t nflag;

    static uint64_t mask(VirtualFlag vflag) { return mask(vflag.idx, vflag.n); }
    static uint64_t mask(int idx, int n) {
        return (uint64_t(1) << (idx + n)) - (uint64_t(1) << idx);
    }
};

class TokenAllocator {
public:
    TokenAllocator(ngen::HW hw, int grfCount = 128);

    int8_t tryAlloc();
    void release(int8_t token) { free |= (1u << token); }
    void safeRelease(int8_t &token) {
        if (token >= 0) release(token);
        token = -1;
    }

protected:
    uint32_t free;
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
    ngen::GRFRange remaskRegs[2];
    VirtualFlag vflagEAtomicAdd;
    VirtualFlag blockEMask;
    ngen::Label blockDone;
    ngen::Subregister all1s;
    ngen::RegData r0_info;
    bool movedR0 = false;
    ngen::Subregister lid0;
    GRFMultirange indexVec; // uw
    int ivEntries = 0;
    struct {
        ngen::GRF zero, one;
        ngen::GRFRange src1Storage;
        ngen::GRF src1, srcR1, srcI1, r, d;
        GRFMultirange mathTemp;
        ngen::GRF temp;
        std::array<ngen::FlagRegister, 2> tempFlags;
        ngen::Subregister flagStore; // ud
        ngen::Label label;
        int simd;
        ngen::Subregister callStorageSub, callStorage;
        bool use = false;
    } invertSub;

    CommonState(ngen::HW hw) : ra(hw), raVFlag(hw), tokenAllocator(hw) {}

    VirtualFlag allocVFlag(ngen::HW hw, int n = 1);

    void wipeActiveVFlags() {
        for (int i = 0; i < int(activeVFlags.size()); i++)
            if (!raVFlag.isLocked(VirtualFlag(i))) activeVFlags[i].clear();
    }

    bool vflagsEnabled() const { return !vflagStorage.empty(); }

    void usePhysicalFlag(ngen::FlagRegister flag) {
        activeVFlags[flag.index()] = flag;
    }

    void allocEmulate64Temp(const EmulationStrategy &estrategy) {
        int ntemp = 0;
        if (estrategy.emulate64) ntemp = std::max(ntemp, 2);
        if (estrategy.emulate64_mul) ntemp = std::max(ntemp, 2);
        if (estrategy.emulateDWxDW) ntemp = std::max(ntemp, 1);

        for (int q = 0; q < ntemp; q++)
            emulate.temp[q] = ra.alloc();
    }
};

// Places to store r0 information.
enum class MoveR0 { None, Acc, Addr, GRF };

// Problem parameters shared between kernel types.
struct CommonProblem {
    bool nonuniformWGs = false; // Support nonuniform workgroups?
    bool gtpinSupport = false; // Support GT-Pin?
};

// Strategy parameters shared between different kernel types.
struct CommonStrategy {
    int subgroupSize = 8; // Subgroup size provided to OpenCL runtime.
    bool fused = false; // Fused EU handling enabled?
    bool dualGRF = true; // Enable two-GRF instructions.
    bool ieeeDenormals = true; // Enable IEEE-compliant denormals.
    bool spf = true; // Enable Single Program Flow (SPF) mode in EUs.
    MoveR0 moveR0 = MoveR0::Acc; // Where to store r0 information.
    bool sipR0WA = false; // Avoid using r0 to avoid clobbering by SIP.
    bool readSuppressionWA
            = true; // Workaround for HW issue with read suppression after fused sends.
    bool multitile = true; // Enable multitile (implicit scaling) support?
    bool wgInSS
            = false; // Pretend to use barriers so that each WG belongs to 1 SS/DSS.
    int GRFs = 128; // # of GRFs to use.
    bool finalFence = false; // Issue global memory fence before EOT.
    uint8_t pad1[3] = {};
    int pauseCycles
            = 0x0100; // Number of cycles to pause when waiting in a spin-loop.
    bool simulation = false; // For use in simulator?
    bool systolicAvailable = false; // True if systolic array present.
    uint8_t pad2[2] = {};
    ngen::HW raHW = ngen::HW::
            Unknown; // Pretend to be a different GPU for register allocation purposes.
    ngen::ThreadArbitrationMode arbitrationMode = ngen::ThreadArbitrationMode::
            Default; // Thread arbitration policy to use.

    EmulationStrategy emulate;
    uint8_t pad3[2] = {};

    CommonStrategy() {}
    CommonStrategy(ngen::HW hw, int stepping = 0);
    void preflight(ngen::HW hw, const CommonProblem &problem);
};

// Types of updates for GEMM kernels.
enum class UpdateType {
    Full,
    UpperTriangle,
    UpperTriangleHermitian,
    LowerTriangle,
    LowerTriangleHermitian
};

// A/B offset mode.
enum class ABOffset {
    None, // No A/B offsets.
    Calc, // Calculate A/B row/column sums in kernel.
    Load, // Use precalculated row/column sums.
};

// C offset mode.
enum class COffset {
    None, // No C offsets.
    Post, // C offset after all other updates.
    Pre, // C offset before all other updates (bias).
};

// Batch mode.
enum class BatchMode { None, Strided, Nonstrided, Variable };

// Binary operations.
enum class BinaryOp { Add, Sub, Mul, Div, Min, Max };

// GEMM kernel problem description.
struct GEMMProblem : public CommonProblem {
    Type Ta, Tb, Tc, Tco, Ts; // Types for A/B/C/C offsets/scalars in registers.
    Type Ta_ext, Tb_ext, Tc_ext; // Types for A/B/C data in memory.

    Scalar alpha, beta; // Scaling factors for A*B and C, respectively.
    MatrixAddressing A, B, C, CO; // Addressing information for matrices.
    bool checkBeta0 = true; // If true, check for beta = 0 and handle specially.
    ABOffset abOffset = ABOffset::None; // A/B offset mode.
    int aoPtrDims = -1,
        boPtrDims
            = -1; // A/B offset dimensionality (-1: none; 0: scalar; 1: vector) -- currently ignored.
    COffset cOffset = COffset::None; // C offset mode.
    BatchMode batch = BatchMode::None; // Batch mode.
    int batchDims = 0; // # of batch dimensions (strided batch only).
    bool sumA = false,
         sumB
            = false; // If true, calculate A row sums/B column sums and store in CO.
    bool postOpFwd = true; // Eltwise parameters

    post_ops_t postOps; // Fused post operations to apply

    // The following data is derived from the postOps and does not need
    // considered for equality/hashing purposes
    std::vector<MatrixAddressing> binary; // Binary postop data
    std::vector<Type> Tbinary; // Binary types
    std::vector<bool> binaryRow; // Dimensionality of binary data
    std::vector<bool>
            binaryCol; //    (false means broadcast in the given dimension)
    std::vector<bool> binaryBatch;

    bool hasPostOp() const { return postOps.len() > 0; }
    bool hasNonSum1PostOp() const {
        for (const auto &e : postOps.entry_)
            if (!e.is_sum()) return true;
        return false;
    }
    bool hasBinaryPostOp() const {
        for (int idx = 0; idx < postOps.len(); idx++)
            if (postOps.entry_[idx].is_binary()) return true;
        return false;
    }
    bool hasSum1PostOpAtEnd() const {
        return postOps.len() > 0 && postOps.entry_[postOps.len() - 1].is_sum();
    }
    void removeFinalSumPostOp() {
        if (postOps.len() > 0) {
            auto &lastPO = postOps.entry_[postOps.len() - 1];
            if (lastPO.kind == primitive_kind::sum)
                postOps.entry_.resize(postOps.len() - 1);
        }
    }

    bool beta0() const { return (beta == 0); }
    bool beta1() const { return (beta == 1); }
    bool alpha1() const { return (alpha == 1); }
    bool alphaM1() const { return (alpha == -1); }

    bool needsTsConvert() const {
        if (!(alpha1() || alphaM1())) return true;
        if (!(beta0() || beta1())) return true;
        if (beta1() && !Tc_ext.isSubsetOf(Tc)) return true;
        if (hasNonSum1PostOp()) return true;
        return false;
    }

    bool gemmt() const { return false; }
    bool backward() const { return false; }

    bool needsASums() const { return (abOffset == ABOffset::Calc) || sumA; }
    bool needsBSums() const { return (abOffset == ABOffset::Calc) || sumB; }
    bool usesCO() const { return (cOffset != COffset::None) || sumA || sumB; }
    bool allowMatrixOffset() const { return (cOffset == COffset::Pre); }

    /* Kernel cache helpers. */
    void serialize(serialized_data_t &s) const {
        s.append(Ta, Tb, Tc, Tco, Ts);
        s.append(Ta_ext, Tb_ext, Tc_ext);
        s.append(alpha);
        s.append(beta);
        s.append(A, B, C, CO);
        s.append(checkBeta0);
        s.append(abOffset);
        s.append(aoPtrDims, boPtrDims);
        s.append(cOffset);
        s.append(batch);
        s.append(batchDims);
        s.append(sumA, sumB);
        s.append(postOpFwd);
        s.append(postOps);
    }
};

struct GEMMState;

// How to split A/B amongst threads in a workgroup.
enum class CoopSplit {
    K, // Split in k dimension
    MN, // Split in m/n dimensions
    Linear, // Split in linear index order
};

// Methods for traversing a matrix.
enum class WalkOrder : uint8_t {
    HW2D, // Rely on HW thread dispatch for ordering
    SimpleLinear, // Simple 1D->2D mapping in column-major/row-major order
    Hilbertlike, // Cache-oblivious Hilbert curve-based order
    Boustrophedon, // Cache-aware panel boustrophedon walk order
};

// Strategy parameters for GEMM kernels.
struct GEMMStrategyPOD : public CommonStrategy {
    void serialize(serialized_data_t &s) const {
        // Explicitly maintain zero padding to keep the implementation simple and
        // robust
        s.append(*this);
    }
    int blocking[3] = {
            0}; // Recommended block size in each dimension (m/n/k) -- for driver.
    int blockingAlt[3] = {
            0}; // Alternate block size in each dimension (m/n/k) -- for driver.
    //     m/n alternates are for Hilbert-ordered kernels when Hilbert ordering disabled.
    //     k alternate is for multi-tile execution with implicit scaling.
    int unroll[3]; // Unrolls in each dimension (m/n/k), indexed by LoopType.
    int unrollK_masked = 0; // k unroll to use when masking.
    int extraKAlign = 1; // Additional k alignment when blocking.
    LoopType loopOrder[3] = {LoopM, LoopN,
            LoopK}; // Expected order of loops in driver code (in order from innermost to outermost).
    LoopType fusedLoop = LoopM; // Direction of fusing if threads fused.
    WalkOrder cWalkOrder = WalkOrder::HW2D; // Order for traversing tiles of C
    bool persistent = false; // Use persistent thread model?
    bool reverse[2] = {false, false}; // Reverse m/n walk order?
    int fmaSIMD = 0; // Vector length for FMA (0 = default = 2 GRFs).
    int kChain = 1; // # of FMAs to chain in k dimension.
    int wg[3] = {0, 0,
            0}; // m/n/k workgroup sizes, 0 if unconstrained. Indexed by LoopType.
    WGType forceWGUpdate = WGDynamic; // Force work group update type.
    uint8_t pad1[3] = {};
    MatrixAddressingStrategy A, B, C,
            CO; // Strategies for accessing A/B/C/C offsets.
    int ka_load, kb_load; // How much of A/B is loaded at once, in k dimension
    int ka_load_masked = 0,
        kb_load_masked
            = 0; // Same as above, when masking m/n (0 = default = same as ka/kb_load)
    bool loadBFirst = false; // If true, load B before A (default A then B).
    bool doubleMasking = false; // Allow A/B to be masked in both dimensions.
    bool kDescRem
            = false; // Allow descriptor-based k remainder handling for A/B.
    bool slmA = false, slmB = false; // Whether to copy A/B to SLM.
    bool splitCopy = false; // Separate SLM copy and compute threads?
    uint8_t pad2[2] = {};
    int slmBuffers = 0; // # of A/B SLM buffers, 0 for none.
    int unrollKSLM
            = 0; // k unroll for SLM copies (0 = auto = unroll[LoopK]/slmCopies)
    int unrollKSLMMasked
            = 0; //   Alternate value to use with masking (0 = same as unrollKSLM)
    bool slmATrans = false,
         slmBTrans
            = false; // Whether A/B SLM data should be completely crosspacked (transposed).
    uint8_t pad3[2] = {};
    int A_copies = 1,
        B_copies = 1; // # of copies of A/B matrices, for latency absorption
    int slmCopies = 1; // # of copies of loaded A/B matrices for SLM copies.
    bool slmRepackAhead = false; // Repack SLM data ahead of stores?
    uint8_t pad4[3] = {};
    int optAlignAB
            = 0; // Optional alignment for A/B. If > 0, create two versions of k loop, one for A/B aligned to this value, one not.
    AccessType unalignedAccA,
            unalignedAccB; // Access types to use for A/B on unaligned path.
    uint8_t pad5[2] = {};
    int ka_prefetch = 0, kb_prefetch = 0; // Chunk size for prefetching A/B.
    int ka_pfStride = 0, kb_pfStride = 0; // k stride between A/B prefetches.
    bool cooperativePF = true; // Enable WG-cooperative A/B prefetches.
    uint8_t pad6[3] = {};
    int prefetchA = 0, prefetchB = 0,
        prefetchC = 0; // Prefetch distances, in units of unrollK.
    int prefetchAMasked = 0,
        prefetchBMasked = 0; // Same as above, when masking m/n.
    MatrixAddressingStrategy A_prefetch, B_prefetch,
            C_prefetch; // Strategies for prefetching A/B/C.
    enum {
        CSeparate, // C stored in its own bundle, A/B in the other bundle.
        ACB, // A, then C, then B
        BCA, // B, then C, then A
        VNC, // A/B (broadcast matrix second), then C
        ABInterleave, // A/B interleaved, then C
        NSeparate, // Broadcast input stored in its own bundle(s)
        VAvoid, // C registers allocated to avoid non-broadcast inputs
    } registerScheme
            = CSeparate; // Register layout scheme.
    int nSeparateChunk
            = 0; // If > 0, chunk size for NSeparate, to facilitate switching layouts.
    bool avoidIncConflicts
            = true; // If true, duplicate some increment values across banks to avoid bundle conflicts.
    bool kParallel
            = false; // If true, generate k-parallelized kernel using global memory reduction.
    bool kParallelLocal
            = false; // If true, generate k-parallelized kernel using local memory reduction.
    bool shrinkWGK
            = false; //   Shrink wgK automatically to try to fit dispatch in 1 wave?
    bool kParallelVariable
            = false; // If true, generate kernel that uses variable k-parallelization for load balancing.
    bool fuseBeta
            = false; //   Fuse beta scaling into kernel? (kParallel/kParallelVariable, requires linear ordering)
    bool fusePostOps
            = false; //   Fuse post-operations into kernel? (kParallel/kParallelVariable, requires linear ordering)
    bool altFusedBeta
            = false; //   Enable alternate beta fusion implementation? (requires sequential dispatch)
    int kPadding
            = 32; //   Pad k dimension when load balancing (kParallel/kParallelVariable)
    bool doubleWA
            = false; // Use explicit double broadcast instructions? (Gen9 only)
    uint8_t pad8[3] = {};
    int barrierFreq
            = 0; // If > 0, set a periodic barrier every barrierFreq k loops to keep threads together.
    bool splitBarrier
            = false; //   Use split barriers for these periodic barriers?
    bool altCRemainder = false; // Use alternative double-loop C remainder code?
    bool block2DCRemainder = false; // Generate block 2D C remainder path?
    bool block2DCFull
            = false; //   Use block 2D C remainder path even for full tiles?
    bool cAccumulators
            = false; // Use accumulator registers for part of C (to save a few registers)?
    bool cLoadAhead = false; // Load C before doing FMAs?
    bool autoatomic = true; // Automatically use C atomics for beta = 1 kernels?
    bool forceCopyC = false; // Force C to be copied before the update step?
    bool noJumpTables = false; // Disallow jump tables?
    RemainderHandling remHandling[3] = {
            // m, n, k remainder handling.
            RemainderHandling::Split,
            RemainderHandling::Split,
            RemainderHandling::General,
    };
    bool jointSplit
            = true; // Use remainder kernel for both m and n dimensions if both are split.
    uint8_t pad9[3] = {};
    int mSplitThresh = 0,
        nSplitThresh
            = 0; // m/n minimum thresholds for using split remainder handling. 0 means always use split.
    bool atomicFMA = false; // Use {Atomic} FMA chains.
    bool extendedAtomicFMA = false; // Use longer {Atomic} FMA chains.
    bool stallAfterLoad = false; // Insert stalls after load operations.
    bool checkAdd32
            = false; // Check inside kernel if inner loop additions can be done in 32-bit.
    bool delayABInc
            = true; // Delay A/B increment a few outer products in the k loop.
    uint8_t pad10[3] = {};
    CoopSplit coopA = CoopSplit::
            K; // How to split SLM copies, cooperative prefetches amongst threads in a workgroup
    CoopSplit coopB = CoopSplit::K;
    bool slmEarlyKMask
            = false; // Prepare A/B reads to use k-masking (when applicable) in main loop, instead of waiting for remainder.
    bool slmUseIncrCopy = true; // Use incremental SLM copies if needed.
    bool slmAltBarriers = false; // Alternate fenceless SLM buffering algorithm.
    bool strictFence
            = false; // Add extra SLM fences that are not usually required on HW.
    bool skipFence
            = false; // Skip SLM fences that theoretically should be required but HW doesn't need.
    bool slmFenceWARWA
            = false; // Work around buggy SLM fence that doesn't protect against WAR hazards.
    bool systolic = false; // Use systolic array if applicable.
    bool dpasw = false; // Use DPASW for fused EU architectures.
    bool fixedSystolic
            = false; // Use hardcoded systolic inner loop for 32x32 or 32x48 unrolls.
    uint8_t pad11[3] = {};
    int namedBarriers[2] = {0,
            0}; // # of named barriers in m, n dimensions (0 to use regular barriers).
    bool skewLocalIDs
            = false; // Remap local IDs for large workgroups so that threads on the same EU don't depend on the same data.
    bool xParallel = false; // TRSM: parallelize in x dimension.
    bool checkBeta1
            = false; // If true, check for beta = 1 and handle specially.
    bool panelCheck = false; // If true, check for out-of-bounds panel reads.
    bool insideSK = false; // Inside a superkernel?
    uint8_t pad12[3] = {};

    GEMMStrategyPOD() {}
    GEMMStrategyPOD(ngen::HW hw, int stepping = 0)
        : CommonStrategy(hw, stepping) {}
};

struct GEMMStrategy : public GEMMStrategyPOD {
    std::vector<MatrixAddressingStrategy>
            binary; // Strategies for binary postop data

    GEMMStrategy() {}
    GEMMStrategy(ngen::HW hw, int stepping = 0)
        : GEMMStrategyPOD(hw, stepping) {}

    void preflight(ngen::HW hw, const GEMMProblem &problem);
    bool minimize(ngen::HW hw, const GEMMProblem &problem);

    int wgTile(LoopType l) const { return unroll[l] * wg[l]; }

    bool lateExit() const {
        return (slmBuffers > 0) || barrierFreq || kParallelLocal || fuseBeta
                || fusePostOps || cooperativePF;
    }

    int slmABufBlockSize(const GEMMProblem &problem) const {
        return fixedSystolic ? 1152
                             : int(slmA) * problem.Ta * problem.Ta.components()
                        * unroll[LoopM] * unrollKSLM;
    }
    int slmBBufBlockSize(const GEMMProblem &problem) const {
        return fixedSystolic ? 1536
                             : int(slmB) * problem.Tb * problem.Tb.components()
                        * unroll[LoopN] * unrollKSLM;
    }
    int slmGEMMABufSize(const GEMMProblem &problem) const {
        return slmABufBlockSize(problem) * wg[LoopM] * wg[LoopK] * slmBuffers;
    }
    int slmGEMMBBufSize(const GEMMProblem &problem) const {
        return slmBBufBlockSize(problem) * wg[LoopN] * wg[LoopK] * slmBuffers;
    }
    int slmABufSize(const GEMMProblem &problem) const {
        return slmGEMMABufSize(problem);
    }
    int slmBBufSize(const GEMMProblem &problem) const {
        return slmGEMMBBufSize(problem);
    }
    int slmSysgemmBlockSize() const {
        return 1152 * wg[LoopM] + 1536 * wg[LoopN];
    }
    bool variableSLM() const { return kParallelLocal; }
    int slmBarriersPerUnroll() const {
        return (slmBuffers == 0) ? 0 : (slmBuffers == 1) ? 2 : 1;
    }

    int ka_inc() const { return slmA ? unrollKSLM : ka_load; }
    int kb_inc() const { return slmB ? unrollKSLM : kb_load; }

    bool needsMNLocalIDs() const {
        return xParallel || (slmBuffers > 0) || cooperativePF || kParallelLocal
                || persistent || namedBarriers[LoopM] || namedBarriers[LoopN]
                || (dpasw && !fixedSystolic);
    }
    bool needsKLocalIDs() const { return kParallelLocal || persistent; }
    bool needsKLoopBarrier() const {
        return (barrierFreq > 0) || (slmBuffers > 0);
    }
    bool needsBarrier() const {
        return needsKLoopBarrier() || xParallel || kParallelLocal || fuseBeta
                || fusePostOps;
    }

    bool needsUnnamedBarrier(const GEMMProblem &problem) const;
    bool needsNamedBarriersM(const GEMMProblem &problem) const;
    bool needsNamedBarriersN(const GEMMProblem &problem) const;

    bool fusedM() const { return fused && (fusedLoop == LoopM); }
    bool fusedN() const { return fused && (fusedLoop == LoopN); }

    WGType getWGType(const GEMMProblem &problem) const {
        if (forceWGUpdate == WGFixed) return WGFixed;
        if ((slmBuffers > 0) || (forceWGUpdate == WGFixed)
                || namedBarriers[LoopM] || namedBarriers[LoopN])
            return WGFixed;
        if (cooperativePF)
            return WGFixed; /* until flexible cooperative PF enabled */
        if (forceWGUpdate == WGShrinkable)
            return WGShrinkable;
        else
            return WGDynamic;
    }

    bool fixedWG(const GEMMProblem &problem) const {
        return (getWGType(problem) == WGFixed);
    }
    bool linearOrder() const { return cWalkOrder != WalkOrder::HW2D; }

    int kAlign(const GEMMProblem &problem) const;

    int statusFlagStride() const {
        return 64 * (int(fuseBeta) + int(fusePostOps));
    }
    bool needsTempC(const GEMMProblem &problem) const;

    void serialize(serialized_data_t &s) const {
        GEMMStrategyPOD::serialize(s);
        for (const auto &astrategy : binary)
            s.append(astrategy);
    }
};

struct LDMultiples {
    ngen::GRFRange range;
    bool a64 = false;
};

// State parameters for GEMM kernels.
struct GEMMState : public CommonState {
    struct Inputs {
        ngen::Subregister A, B, C[2], CO, base, tempC; // q
        ngen::Subregister ao, bo, abo; // w/w/ud
        ngen::Subregister aoPtr, boPtr; // q
        ngen::Subregister offsetA, offsetB, offsetC[2]; // q
        ngen::Subregister offsetCO; // d
        ngen::Subregister lda, ldb, ldc[2], ldco; // d
        ngen::Subregister m, n, k, k0; // d
        SubregisterPair alpha_real, alpha_imag; // T_real
        SubregisterPair beta_real, beta_imag; // T_real
        ngen::Subregister alphaPtr, betaPtr; // q
        ngen::Subregister groupIDM, groupIDN, groupIDK; // ud
        ngen::Subregister groupIDMN; // ud
        ngen::GRF localIDM, localIDN, localIDK; // uw
        ngen::Subregister localSizeM, localSizeN, localSizeK; // ud
        ngen::Subregister groupCountM, groupCountN, groupCountK; // ud
        ngen::Subregister groupCountMN; // ud
        ngen::Subregister gcMNRecip; // ud
        ngen::Subregister groupStride; // ud
        ngen::Subregister kParallelStart, kRecip, k0Recip; // ud
        ngen::Subregister hilbertVD, hilbertUVDRecip; // ud
        ngen::Subregister hilbertBail; // ud
        ngen::Subregister bslice, bthresh; // d
        ngen::Subregister flags; // ud
        ngen::Subregister diagA, diagB, diagC; // q
        ngen::Subregister statusBuffer; // q
        uint8_t surfaceA, surfaceB; // BTS indices
        uint8_t surfaceC[2], surfaceCO, surfaceTempC; // BTS indices
        ngen::Subregister strideA[2], strideB[2],
                strideC[2]; // ud, used for strided batch.
        ngen::Subregister batchSize1, recipBatchSize1; // ud, 2D strided batch
        ngen::Subregister offsetBatch; // ud, used for non-strided batch.
        ngen::Subregister incr_a_array,
                incr_b_array; // ud, used for non-strided variable batch.
        ngen::Subregister incr_alpha,
                incr_beta; // ud, used for non-strided variable batch.
        ngen::Subregister alpha_array,
                beta_array; // q, used for non-strided variable batch.
        std::vector<ngen::Subregister> binarySrcs; // q
        std::vector<ngen::Subregister> binaryOffsets; // q/d
        std::vector<ngen::Subregister> binaryLDs; // d
        std::vector<std::array<ngen::Subregister, 2>> binaryStrides; // d
        std::vector<uint8_t> binarySurfaces;
    } inputs;
    Type Ta_load, Tb_load; // Current type to be loaded into A/B_regs.
    Type Tacc; // Current type in accumulator registers.
    ngen::Subregister persistentGroupID; // ud
    ngen::Subregister batchID[2]; // ud
    ngen::Subregister offsetA, offsetB, offsetC[2];
    ngen::Subregister offsetAp, offsetBp, offsetCp;
    ngen::Subregister offsetCO;
    ngen::Subregister saveOffsetA, saveOffsetB, saveOffsetC[2];
    ngen::Subregister saveOffsetCO;
    ngen::Subregister fullK;
    ngen::Subregister effA, effB, effC[2], effCO,
            effTempC; // Offsets to base of A/B/C/CO/tempC chunks for loading/storing.
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
    std::vector<GRFMultirange> A_regs, B_regs, C_regs;
    GRFMultirange Ar_regs, Br_regs; // Repacked A/B registers.
    std::vector<GRFMultirange> Ai_regs,
            Bi_regs; // Incoming data to copy to SLM.
    std::vector<GRFMultirange> Ai_regsRem, Bi_regsRem;
    GRFMultirange Ao_regs, Bo_regs; // Outgoing data to copy to SLM.
    GRFMultirange Ao_regsRem, Bo_regsRem;
    GRFMultirange As_regs, Bs_regs; // A row sums/B column sums.
    GRFMultirange Ap_regs, Bp_regs, Cp_regs; // A/B/C prefetch registers.
    std::vector<MaskAssignment> AB_masks, AB_masksCoop;
    ngen::GRFRange broadcast_regs;
    std::vector<ngen::GRFRange> tempMul_regs;
    ngen::Subregister i0, j0, h0; // d
    ngen::Subregister threadK0, k0Rem, wgK; // ud
    ngen::Subregister remainders[3]; // d (todo: w)
    ngen::Subregister remaindersFused[2]; // w
    ngen::Subregister remaindersWG[2]; // d (todo: w)
    ngen::Subregister remaindersCoop[3]; // d
    ngen::Subregister remFusedStorage; // d
    ngen::Subregister diagA, diagB, diagC; // d
    SubregisterPair lda, ldb;
    SubregisterPair lda_ka, ldb_kb; // Cached lda * ka, ldb * kb
    SubregisterPair lda_ka_prefetch,
            ldb_kb_prefetch; // Cached lda * ka_pfStride, ldb * kb_pfStride
    LDMultiples ldaMultiples, ldbMultiples, ldcMultiples[2];
    int ka_cached = 0, kb_cached = 0; // Multipliers for lda_ka/ldb_kb.
    ngen::Subregister k, K; // d
    ngen::Subregister kNoBarrierStart, kNoBarrierEnd; // d
    ngen::FlagRegister flagAP;
    ngen::Subregister beta1; // d
    ngen::Subregister add64; // uw
    ngen::Subregister lidM, lidN, lidStorage; // uw, uw, ud
    ngen::Subregister lidK, lszK, lidszKStorage; // uw, uw, ud
    ngen::Subregister ia0_slm, jb0_slm; // uw
    ngen::Subregister postRemA, postRemB; // ud
    ngen::Subregister postRemAi, postRemBi; // ud
    ngen::Subregister postRemAo, postRemBo; // ud
    ngen::Subregister isCompute; // ud
    ngen::GRF sysSumAll1s; // Ta/Tb
    ngen::GRF betaCheckReturn;
    ngen::Subregister statusFlagAddr; // uq
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
    ngen::Subregister kSLMA, kSLMB, kSLMStorage; // w/w/ud
    bool kSLMCountUp = false;
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
    std::vector<RegisterBlock> Ap_layout, Bp_layout, Cp_layout;
    std::vector<RegisterBlock> Ap_layoutAlt, Bp_layoutAlt;
    std::vector<RegisterBlock> C_layoutExt, C_layoutExtUnmasked,
            C_layoutExtNonatomicUnmasked;
    Address2DParams A_params, B_params;
    Address2DParams Ai_params, Bi_params;
    Address2DParams Ap_params, Bp_params;
    int Ai_regCount = 0, Bi_regCount = 0;
    bool aioShare, bioShare;
    bool aioShareRem, bioShareRem;
    bool aoReuseA = false, boReuseB = false;
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
    int ka_repackRem, kb_repackRem;
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

    std::vector<ngen::Subregister> effBinary;

    struct {
        bool active = false;
        uint8_t surfacePlan;
        ngen::Subregister plan;
        ngen::Subregister slotA, slotB;
        ngen::Subregister localIDFlat;
        ngen::FlagRegister needLateGEMMDone;
    } fusedGEMM;

    struct {
        ngen::InstructionModifier depAddr[4];
    } sysgemm;

    GEMMState(ngen::HW hw) : CommonState(hw) {}
};

// GEMM superkernel problem.
struct GEMMSuperkernelProblem : public GEMMProblem {};

// GEMM superkernel strategy parameters.
struct GEMMSuperkernelStrategy {
    std::vector<GEMMStrategy> substrategies;
    KernelScheduling schedule;
    bool multiM, multiN;
    bool persistent = false;

    void preflight(ngen::HW hw, const GEMMProblem &problem);
    int subgroupSize() const { return substrategies[0].subgroupSize; }
};

// GEMM superkernel state.
struct GEMMSuperkernelState : public GEMMState {
    struct {
        uint8_t surfacePlan;
        ngen::Subregister planCount;
        ngen::GRF localID;
        ngen::Subregister localSize;
    } inputsSK;
    ngen::Subregister last_i0, last_j0, last_h0;

    GEMMSuperkernelState(ngen::HW hw) : GEMMState(hw) {}
};

// Copy kernel problem description: D <- alpha*S
struct CopyProblem : public CommonProblem {
    Type Ts, Td, Tsum;
    Scalar alpha;
    MatrixAddressing S, D;
    bool conjugate = false;
    bool lower;
    bool unit;
    bool trsm = false;
    bool sum = false;
    int targetWG = 1;

    bool reflecting() const { return false; }
};

// Strategy parameters for copy kernels.
struct CopyStrategy : public CommonStrategy {
    MatrixAddressingStrategy S, D;
    RemainderHandling remHandlingX,
            remHandlingY; // Remainder handling for X dimension (packed dimension) and Y dimension (length of panel)
    int s_load, d_load; // # of rows/columns to load from S/store to D at once
    int s_load_masked = 0,
        d_load_masked
            = 0; // Same as s_load/d_load, for use when masking (0 = default = same as {s,d}_load)
    int wgW = 0, wgZ = 0; // Fixed workgroup sizes (0 if variable).

    int unrollX, unrollY; // Unrolls for each dimension.
    bool xLoop
            = false; // True to loop over x, false to loop over y within a kernel

    bool zParallel = false; // Kernel parallelized in z dimension?

    int barrierFreq = 0; // If > 0, set a barrier every barrierFreq loops
    int optionalAlignS
            = 0; // If > 0, generate code to check if S is aligned to this #elements and branch to specific code for that case.
    bool doubleMasking = false; // Allow S to be masked in both dimensions

    bool duplicateAlpha
            = true; // True to make two copies of alpha, one for each register bank

    CopyStrategy() {}
    CopyStrategy(ngen::HW hw, int stepping = 0)
        : CommonStrategy(hw, stepping) {}

    void preflight(ngen::HW hw, const CopyProblem &problem);

    int unrollW() const { return xLoop ? unrollY : unrollX; }
    int unrollZ() const { return xLoop ? unrollX : unrollY; }
};

// State parameters for copy kernels.
struct CopyState : public CommonState {
    struct {
        ngen::Subregister S, D; // q
        ngen::Subregister offsetS, offsetD; // q
        ngen::Subregister lds, ldd; // d
        ngen::Subregister m, n; // d
        SubregisterPair alpha_real; // T_real
        SubregisterPair alpha_imag; // T_real
        ngen::Subregister groupIDW, groupIDZ; // ud
        ngen::GRF localIDW, localIDZ; // uw
        ngen::Subregister localSizeW, localSizeZ; // ud
        ngen::Subregister diag; // d
        ngen::Subregister blockZ; // ud
        uint8_t surfaceS, surfaceD; // DTS indices
    } inputs;
    ngen::Subregister D_m, D_n; // d
    ngen::Subregister w0, z0; // ud
    ngen::Subregister effS,
            effD; // Offsets to base of S/D chunks for loading/storing.
    ngen::Subregister offsetS1,
            effS1; // Reflected variants of offsetS/effS for symmetric/Hermitian.
    std::vector<ngen::GRFRange> S_addrs, D_addrs;
    std::vector<ngen::GRFRange> S_addrSrcs[2];
    ngen::GRFRange S_regs, D_regs, D0_regs;
    std::vector<ngen::GRFRange> Ds_regs;
    ngen::Subregister lds_sl; // d
    ngen::Subregister ldd_dl; // d
    ngen::Subregister Z; // d
    ngen::FlagRegister flagAP, flagTri, flagDiag;
    ngen::FlagRegister flagReflect;
    std::vector<RegisterBlock> S_layout, D_layout;
    std::vector<RegisterBlock> D0_layout, Ds_layout;
    ngen::Subregister remainderX, remainderY; // ud
    ngen::Subregister D_remainderY; // ud
    ngen::GRFRange complexOne; // T_real
    ngen::GRF indexVecRT; // uw

    bool isNested;

    struct {
        bool active = false;
    } fusedGEMM;

    CopyState(ngen::HW hw) : CommonState(hw) {}

    void dump();
};

template <ngen::HW hw>
class gemm_kernel_generator_t : public jit_generator<hw> {
public:
    using super = ngen::OpenCLCodeGenerator<hw>;
    gemm_kernel_generator_t() {}

    NGEN_FORWARD_OPENCL(hw);

    using Injector = jit_post_op_injector<hw>;
    std::unique_ptr<Injector> postOpInjector;

    static bool supportedBinaryOp(alg_kind_t alg) {
        using namespace alg_kind;
        return utils::one_of(alg, binary_add, binary_sub, binary_mul,
                binary_div, binary_min, binary_max);
    }

    void gemm(GEMMProblem problem, GEMMStrategy strategy,
            const ngen::InterfaceHandler &interface_);
    void gemmSuperkernel(GEMMSuperkernelProblem problem,
            GEMMSuperkernelStrategy strategy,
            const ngen::InterfaceHandler &interface_);
    void copy(CopyProblem problem, CopyStrategy strategy,
            const ngen::InterfaceHandler &interface_);

    static CommonDriverInfo driverInfo(
            const GEMMProblem &problem, const GEMMStrategy &strategy);
    static CommonDriverInfo driverInfo(const GEMMSuperkernelProblem &problem,
            const GEMMStrategy &strategy);
    static CommonDriverInfo driverInfo(
            const CopyProblem &problem, const CopyStrategy &strategy);

protected:
    ngen::InterfaceHandler
            &interface = ngen::OpenCLCodeGenerator<hw>::interface_;

    std::exception_ptr lastException;

    std::ostream &getOutStream() const { return std::cerr; }

    std::ostream &noteStream() const { return getOutStream(); }

    class status_stream {
    protected:
        char cc;
        std::stringstream line;
        bool lineStart = true;

        gemm_kernel_generator_t<hw> &parent;

        friend class gemm_kernel_generator_t<hw>;

    public:
        status_stream(gemm_kernel_generator_t<hw> &parent_, int color = 1)
            : cc(color + '0'), parent(parent_) {}

        static constexpr struct Endl {
        } endl {};

        template <typename T>
        status_stream &operator<<(const T &obj) {
            return *this;
        }

        status_stream &operator<<(const Endl &e) { return *this; }
    } status {*this};

#ifdef SHOW_DISCARDS
    void discardStream() {
        InstructionStream *s = popStream();
        auto oldCC = status.cc;
        status.cc = '4';
        status << "------- \x1B[32mBEGIN\x1B[34m discarded stream -------"
               << status_stream::endl;
        auto &sbuffer = *reinterpret_cast<std::ostringstream *>(s->getBuffer());
        auto str = sbuffer.str();
        bool lastNL = false;
        for (int l = 0; l < str.length(); l++) {
            char c = str[l];

            if (c == '\n') {
                if (lastNL) status << "//";
                status << status_stream::endl;
                lastNL = true;
            } else {
                status << c;
                lastNL = false;
            }
        }
        status << "-------  \x1B[32mEND\x1B[34m discarded stream  -------"
               << status_stream::endl;
        status.cc = status.cc;
        delete s;
    }
#endif

    enum class HintType {
        Bank0,
        Bank1,
        TempComp0,
        TempComp1,
        LongTerm,
        LongTerm0,
        LongTerm1,
        R0Info,
        A0,
        A0Broadcast,
        A1,
        A1Broadcast,
        B0,
        B0Broadcast,
        B1,
        B1Broadcast,
        C,
        C1,
        CLoad,
        S,
        D,
        SAddr,
        DAddr
    };
    enum class StdCRemType { Ignore, Mask, Descriptor };
    enum class COperation { Load, Update, UpdateStore, Store };
    enum class AccessClass { Read, Write, Atomic };
    enum class KLoop {
        GEMM,
    };
    enum class KBarrierType { Normal, Signal, Wait };

    friend std::ostream &operator<<(std::ostream &s, StdCRemType rt) {
        const char *names[3] = {"ignore", "mask", "custom descriptor"};
        return (s << names[static_cast<int>(rt)]);
    }

    ngen::FlagRegister getPhysicalFlag(VirtualFlag vflag, CommonState &state);
    void allocVFlagStorage(const CommonStrategy &strategy, CommonState &state,
            bool saveCurrent = true);
    void deallocVFlagStorage(CommonState &state, bool saveCurrent = true);

    ngen::Bundle getHint(HintType type);
    ngen::Bundle getHint(HintType type, const CommonStrategy &strategy);
    ngen::Bundle getHint(HintType type, const GEMMStrategy &strategy);
    ngen::Bundle getHint(HintType type, const CopyStrategy &strategy);

    void goto12(const ngen::InstructionModifier &mod, ngen::Label &jip) {
        goto12(mod, jip, jip);
    }
    void goto12(const ngen::InstructionModifier &mod, ngen::Label &jip,
            ngen::Label &uip, bool branchCtrl = false);

    template <typename DT = void>
    void mulConstant(const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0, int32_t src1);

    friend struct EmulationImplementation;
    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, const CommonStrategy &strategy,
            CommonState &state);
    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::Immediate src0, const CommonStrategy &strategy,
            CommonState &state) {
        EmulationImplementation::emov<DT>(
                *this, mod, dst, src0, strategy.emulate);
    }
    template <typename DT = void>
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1,
            const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void>
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, ngen::Immediate src1,
            const CommonStrategy &strategy, const CommonState &state) {
        EmulationImplementation::eadd<DT>(
                *this, mod, dst, src0, src1, strategy.emulate, state.emulate);
    }
    template <typename DT = void>
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1,
            const CommonStrategy &strategy, const CommonState &state) {
        EmulationImplementation::emul<DT>(
                *this, mod, dst, src0, src1, strategy.emulate, state.emulate);
    }
    template <typename DT = void>
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, ngen::Immediate src1,
            const CommonStrategy &strategy, const CommonState &state) {
        EmulationImplementation::emul<DT>(
                *this, mod, dst, src0, src1, strategy.emulate, state.emulate);
    }
    template <typename DT = void>
    void eshl(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, uint16_t src1, const CommonStrategy &strategy,
            const CommonState &state) {
        EmulationImplementation::eshl<DT>(
                *this, mod, dst, src0, src1, strategy.emulate, state.emulate);
    }
    template <typename DT = void>
    void eshr(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, uint16_t src1, const CommonStrategy &strategy,
            const CommonState &state) {
        EmulationImplementation::eshr<DT>(
                *this, mod, dst, src0, src1, strategy.emulate, state.emulate);
    }
    template <typename DT = void>
    void emulConstant(const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0, int32_t src1,
            const CommonStrategy &strategy, const CommonState &state) {
        EmulationImplementation::emulConstant<DT>(
                *this, mod, dst, src0, src1, strategy.emulate, state.emulate);
    }
    template <typename S1>
    void emul32High(const ngen::InstructionModifier &mod,
            const ngen::RegData &dstHi, const ngen::RegData &src0,
            const S1 &src1) {
        EmulationImplementation::emul32High(*this, mod, dstHi, src0, src1);
    }

    template <typename S0, typename S2>
    void emad(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const S0 &src0, const ngen::RegData &src1, const S2 &src2,
            const CommonStrategy &strategy, CommonState &state, bool sub);
    template <typename S0>
    void emad(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const S0 &src0, const ngen::RegData &src1,
            const ngen::Immediate &src2, const CommonStrategy &strategy,
            CommonState &state);
    template <typename S0>
    void emad(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const S0 &src0, ngen::RegData src1, ngen::RegData src2,
            const CommonStrategy &strategy, CommonState &state);
    template <typename S0>
    void emad(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const S0 &src0, const ngen::RegData &src1, int32_t src2,
            const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void, typename S0, typename S2>
    void eadd3(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const S0 &src0, const ngen::RegData &src1, const S2 &src2);
    template <typename S0>
    void ecsel(const ngen::InstructionModifier &mod,
            const ngen::InstructionModifier &cmod,
            const ngen::FlagRegister &flag, const ngen::RegData &dst,
            const S0 &src0, const ngen::RegData &src1,
            const ngen::RegData &src2);

    template <typename DT = void>
    void emath(const ngen::InstructionModifier &mod, ngen::MathFunction fc,
            const ngen::RegData &dst, const ngen::RegData &src0,
            const GEMMStrategy &strategy, CommonState &state);
    template <typename DT = void>
    void einv(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const GEMMStrategy &strategy,
            CommonState &state) {
        emath<DT>(mod, ngen::MathFunction::inv, dst, src0, strategy, state);
    }
    template <typename DT = void>
    void esqt(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const GEMMStrategy &strategy,
            CommonState &state) {
        emath<DT>(mod, ngen::MathFunction::sqt, dst, src0, strategy, state);
    }

    void ejmpi(ngen::InstructionModifier mod, ngen::Label &dst);

    void cmp0(const ngen::InstructionModifier &mod, ngen::RegData src0);
    void syncall();

    void wrdepRanges(const std::vector<GRFMultirange> &rrs) {
        for (auto &rr : rrs)
            for (auto &r : rr.ranges)
                wrdep(r);
    }

    void addScaled(const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, int src0, const ngen::RegData &src1,
            int numerator, int denominator, CommonState &state,
            bool exact = false);
    void addScaled(const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0,
            const ngen::RegData &src1, int numerator, int denominator,
            CommonState &state, bool exact = false);
    void addScaled(const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0, int src1,
            int numerator, int denominator, CommonState &state,
            bool exact = false);

    template <typename DT = void>
    void mod(const ngen::Subregister &dst, const ngen::Subregister &src,
            uint16_t modulus, const CommonStrategy &strategy,
            CommonState &state);
    template <typename DT = void>
    void modExt(const ngen::Subregister &dstMod,
            const ngen::Subregister &dstMultiple, const ngen::Subregister &src,
            uint16_t modulus, const CommonStrategy &strategy,
            CommonState &state);
    template <typename DT = void>
    void alignDown(const ngen::Subregister &dst, const ngen::Subregister &src,
            uint16_t align, const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void>
    void alignDown(const ngen::InstructionModifier &mod,
            const ngen::Subregister &dst, const ngen::Subregister &src,
            uint16_t align, const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void>
    void alignUp(const ngen::Subregister &dst, const ngen::Subregister &src,
            uint16_t align, const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void>
    void divDown(const ngen::Subregister &dst, const ngen::Subregister &src,
            uint16_t divisor, const CommonStrategy &strategy,
            CommonState &state);
    template <typename DT = void>
    void divDown(const ngen::Subregister &dst, const ngen::Subregister &src0,
            const ngen::Subregister &src1, const ngen::Subregister &src1Recip,
            const ngen::FlagRegister &flag, const CommonStrategy &strategy,
            CommonState &state);
    template <typename DT = void>
    void divUp(const ngen::Subregister &dst, const ngen::Subregister &src0,
            const ngen::Subregister &src1, const ngen::Subregister &src1Recip,
            const ngen::FlagRegister &flag, const CommonStrategy &strategy,
            CommonState &state);

    void simtDoWhileLoop(
            const ngen::InstructionModifier &mod, ngen::Label &dest);
    void slmBarrier(const ngen::GRF &temp, const ngen::GRF &r0_info = r0);
    void globalMemFence(const ngen::GRF &temp, const ngen::GRF &r0_info,
            const CommonStrategy &strategy);
    void globalMemBarrier(const ngen::GRF &temp, const ngen::GRF &r0_info,
            const CommonStrategy &strategy);
    void pause(const CommonStrategy &strategy);

    void duplicateScalar(SubregisterPair &val, CommonState &state);
    void deduplicateScalar(SubregisterPair &val, CommonState &state);
    MultishiftSubregister multishift(const ngen::Subregister &reg,
            unsigned shifts, const CommonStrategy &strategy, CommonState &state,
            ngen::Bundle hint = ngen::Bundle());

    void getFusedID(int scale, const CommonProblem &problem,
            const CommonStrategy &strategy, CommonState &state);
    void moveR0(const CommonStrategy &strategy, CommonState &state);
    void moveR0(const GEMMStrategy &strategy, GEMMState &state);
    template <typename F>
    void useR0(CommonState &state, F f);
    template <typename F>
    void useTempAndR0(CommonState &state, F f);
    void removeSG(const CommonProblem &problem, const CommonStrategy &strategy,
            const CommonState &state);
    void reorderFusedEUs(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    ngen::Subregister copySubregister(const ngen::Subregister &reg,
            CommonState &state,
            ngen::Bundle hint = ngen::Bundle(ngen::Bundle::any, 0));
    void zeroMatrix(const GRFMultirange &r, const CommonStrategy &strategy);
    void releaseFusedRemainders(GEMMState &state);
    void saveMNLocalIDs(const GEMMStrategy &strategy, GEMMState &state);
    void saveKLocalIDSize(const GEMMStrategy &strategy, GEMMState &state);
    void releaseSavedMNLocalIDs(GEMMState &state);

    void doReadSuppressionWA(
            const CommonStrategy &strategy, CommonState &state);

    bool getBlockInfo(Type T, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy, int r, int c,
            bool remainderR, bool remainderC, bool writable,
            RemainderOptions remOpts, int maxRBlock, int maxCBlock, int &rblock,
            int &cblock, RegisterBlock &layout);
    bool getSubblock(Type T, RegisterBlock &blockDst,
            const RegisterBlock &blockSrc, bool column, int x1, int x2,
            int x1Unclamped, int x2Unclamped, bool overrunOK,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool getSubblocks(Type T, std::vector<RegisterBlock> &sublayout,
            const std::vector<RegisterBlock> &layout, bool column, int x1,
            int x2, bool overrunOK, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool getSubblocks(Type T, std::vector<RegisterBlock> &sublayout,
            std::vector<ngen::GRFRange> *subaddrs, std::vector<int> *indices,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> *addrs, bool column, int x1,
            int x2, bool overrunOK, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool getSubblocks(Type T, std::vector<RegisterBlock> &sublayout,
            std::vector<ngen::GRFRange> &subaddrs,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, bool column, int x1,
            int x2, bool overrunOK, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool getSubblocks(Type T, std::vector<RegisterBlock> &sublayout,
            std::vector<int> &indices, const std::vector<RegisterBlock> &layout,
            bool column, int x1, int x2, bool overrunOK,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool reblockLayout(Type Tdst, std::vector<int32_t> &blockMap,
            std::vector<RegisterBlock> &layoutDst,
            const std::vector<RegisterBlock> &layoutRef,
            const std::vector<RegisterBlock> &layoutSrc,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);

    bool tryAddRemainder(Type T, RegisterBlock &block, bool remainderR,
            bool remainderC, RemainderOptions remOpts,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool tryAddRemainder(Type T, std::vector<RegisterBlock> &layout,
            bool remainderR, bool remainderC, RemainderOptions remOpts,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    void addRemainder(Type T, std::vector<RegisterBlock> &layout,
            bool remainderR, bool remainderC, RemainderOptions remOpts,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    void addRemainder(Type T, std::vector<RegisterBlock> &layout,
            std::vector<ngen::GRFRange> &addrs, const ngen::Subregister &ld,
            bool remainderR, bool remainderC, RemainderOptions remOpts,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state,
            int dataRegs = -1);
    int checkDescriptorRemainder(Type T, int r, int c, bool column,
            bool writable, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    void updateBlock2DSizes(ngen::GRF addr, const RegisterBlock &dst,
            const RegisterBlock &src, const MatrixAddressing &atype);
    void adjustSubblockAddrs(Type T,
            const std::vector<RegisterBlock> &sublayout,
            const std::vector<ngen::GRFRange> &subaddrs,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, const CommonState &state);

    bool addToRegLayout(Type T, std::vector<RegisterBlock> &layout, int r,
            int c, int roff, int coff, bool remainderR, bool remainderC,
            bool writable, RemainderOptions remOpts, int maxRBlock,
            int maxCBlock, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool add1DBlockToRegLayout(Type T, std::vector<RegisterBlock> &layout,
            int r, int c, bool writable, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool getRegLayout(Type T, std::vector<RegisterBlock> &layout, int r, int c,
            bool remainderR, bool remainderC, bool writable,
            RemainderOptions remOpts, int maxRBlock, int maxCBlock,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            bool reverseOrder = false);
    void makeUnbackedRegLayout(Type T, std::vector<RegisterBlock> &layout,
            int r, int c, bool colMajor, int crosspack = 1, int tileR = 0,
            int tileC = 0, bool allowPartialRegs = true,
            bool fullySplitCx = false);
    bool upgradeLayoutToBlock2D(Type T,
            const std::vector<RegisterBlock> &layoutSrc,
            std::vector<RegisterBlock> &layout2D, bool remainderR,
            bool remainderC, bool writable, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);

    void setupTeardownLoadStoreDesc(bool setup,
            const std::vector<RegisterBlock> &layout,
            const CommonStrategy &strategy, CommonState &state);
    void loadLoadStoreDescriptors(bool load, bool store, RegisterBlock &block,
            ngen::Subregister count, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state,
            bool clamp = false, int offset = 0);

    static ngen::DataSpecLSC getDataSpecLSC(
            AccessType access, const RegisterBlock &block);
    static ngen::DataSpecLSC getDataSpecLSC(const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const RegisterBlock &block, AccessClass aclass);
    void startDoubleMask(VirtualFlag vflag, CommonState &state);
    void prepareSeriesRegisterBlockDoubleMasking(
            const std::vector<RegisterBlock> &layout, CommonState &state,
            int start);
    void prepareSeriesRegisterBlockMasking(
            const std::vector<RegisterBlock> &layout, CommonState &state,
            int start);
    ngen::InstructionModifier registerBlockMasking(const RegisterBlock &block,
            CommonState &state, ngen::FlagRegister *outFlag = nullptr);
    void finishRegisterBlockMasking(CommonState &state);
    void loadMatrixBlock(const ngen::Register &dest,
            const RegisterBlock &layout, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const ngen::GRFRange &addr, const CommonStrategy &strategy,
            CommonState &state, bool readCheck = false, bool series = false);
    void loadMatrix(const GRFMultirange &dest,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const std::vector<ngen::GRFRange> &addrs,
            const CommonStrategy &strategy, CommonState &state,
            bool readCheck = false);
    void prefetchMatrix(const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const std::vector<ngen::GRFRange> &addrs,
            const CommonStrategy &strategy, CommonState &state);
    void storeMatrixBlock(const ngen::GRF &src, const RegisterBlock &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const ngen::GRFRange &addr, const CommonStrategy &strategy,
            CommonState &state, bool series = false);
    void storeMatrix(const GRFMultirange &src,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const std::vector<ngen::GRFRange> &addrs,
            const CommonStrategy &strategy, CommonState &state);
    void atomicAddMatrixBlock(Type T, const ngen::GRF &src,
            const RegisterBlock &layout, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const ngen::GRFRange &addr, const CommonProblem &problem,
            const CommonStrategy &strategy, CommonState &state,
            bool series = false);
    void atomicAddMatrix(Type T, const GRFMultirange &src,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const std::vector<ngen::GRFRange> &addrs,
            const CommonProblem &problem, const CommonStrategy &strategy,
            CommonState &state);

    bool assignMasks(std::vector<RegisterBlock> &layout, LoopType rloop,
            LoopType cloop, std::vector<MaskAssignment> &assignments,
            const CommonStrategy &strategy, CommonState &state,
            bool retryVirtual = false,
            const std::vector<MaskAssignment> *existing = nullptr);
    void loadMask(MaskAssignment assignment, ngen::Subregister index,
            const CommonStrategy &strategy, CommonState &state, int offset = 0);
    void loadMasks(const std::vector<MaskAssignment> &assignments,
            ngen::Subregister (&indices)[3], const CommonStrategy &strategy,
            CommonState &state, int start = 0);
    void loadMasks(const std::vector<MaskAssignment> &assignments,
            ngen::Subregister (&indices)[3], int (&offsets)[3],
            const CommonStrategy &strategy, CommonState &state, int start = 0);

    void setupTeardownRemask(Type T, int index, bool setup, int nq,
            const ngen::Subregister &remQ, const CommonStrategy &strategy,
            CommonState &state, int fixedOffQ = 0,
            const ngen::Subregister &variableOffQ = ngen::Subregister());
    void remaskLayout(Type T, int index, bool column,
            const std::vector<RegisterBlock> &layout, const GRFMultirange &regs,
            const CommonStrategy &strategy, CommonState &state, int offset = 0);

    void setAddrRemainder(Type T, const ngen::GRFRange &addr,
            const RegisterBlock &block, const ngen::Subregister &remR,
            const ngen::Subregister &remC, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    void setAddrRemainder(Type T, const std::vector<ngen::GRFRange> &addr,
            const std::vector<RegisterBlock> &layout,
            const ngen::Subregister &remR, const ngen::Subregister &remC,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);

    ngen::Subregister startShift(
            const MultishiftSubregister &ptr, int shift, CommonState &state);
    SubregisterPair startShift(
            const SubregisterPair &ptr, int shift, CommonState &state);
    template <typename BO>
    typename std::enable_if<!std::is_base_of<ngen::RegData, BO>::value,
            BO>::type
    startShift(const BO &ptr, int shift, CommonState &state);
    template <typename BO>
    typename std::enable_if<std::is_base_of<ngen::RegData, BO>::value, BO>::type
    startShift(const BO &ptr, int shift, CommonState &state);
    template <typename BO, typename BI>
    typename std::enable_if<!std::is_base_of<ngen::RegData, BO>::value>::type
    doneShift(
            const BO &ptr, const BI &ptrShifted, int shift, CommonState &state);
    template <typename BO, typename BI>
    typename std::enable_if<std::is_base_of<ngen::RegData, BO>::value>::type
    doneShift(
            const BO &ptr, const BI &ptrShifted, int shift, CommonState &state);
    void doneShift(const SubregisterPair &ptr,
            const SubregisterPair &ptrShifted, int shift, CommonState &state);

    void offsetAddr(const ngen::GRFRange &addrDst,
            const ngen::GRFRange &addrSrc, const RegisterBlock &blockDst,
            const RegisterBlock &blockSrc, int offsetFixed, int offsetLD,
            const ngen::Subregister &ld, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state,
            const LDMultiples &ldMultiples = {});
    void setupAddrRel(Type T, const ngen::GRFRange &addrDst,
            const ngen::GRFRange &addrSrc, const RegisterBlock &blockDst,
            const RegisterBlock &blockSrc,
            const std::vector<RegisterBlock> &layout,
            const ngen::Subregister &ld, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state,
            const LDMultiples &ldMultiples = {});
    template <typename BO>
    void setupAddr(const ngen::GRFRange &addr, const BO &ptr,
            const RegisterBlock &layout, const ngen::Subregister &ld,
            int sizeofT, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state,
            const Address2DParams &params = {}, LDMultiples ldMultiples = {});
    template <typename BO>
    void setupAddr(Type T, const std::vector<ngen::GRFRange> &addr,
            const BO &ptr, const std::vector<RegisterBlock> &layout,
            const ngen::Subregister &ld, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state,
            const Address2DParams &params = {},
            const LDMultiples &ldMultiples = {});
    template <typename I, typename Ir, typename Ic>
    void incAddrShifted(const ngen::GRFRange &addrDst,
            const ngen::GRFRange &addrSrc, I inc, Ir incR, Ic incC,
            const RegisterBlock &layoutDst, const RegisterBlock &layoutSrc,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename I, typename Ir, typename Ic>
    void incAddrShifted(const std::vector<ngen::GRFRange> &addr, I inc, Ir incR,
            Ic incC, const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename I>
    void incAddrShifted(const std::vector<ngen::GRFRange> &addr, I inc,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename I, typename Ir, typename Ic>
    void incAddr(const ngen::GRFRange &addrDst, const ngen::GRFRange &addrSrc,
            I inc, Ir incR, Ic incC, const RegisterBlock &layoutDst,
            const RegisterBlock &layoutSrc, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename I, typename Ir, typename Ic>
    void incAddr(const std::vector<ngen::GRFRange> &addr, I inc, Ir incR,
            Ic incC, const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename I>
    void incAddr(const ngen::GRFRange &addrDst, const ngen::GRFRange &addrSrc,
            I inc, const RegisterBlock &layoutDst,
            const RegisterBlock &layoutSrc, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename I>
    void incAddr(const std::vector<ngen::GRFRange> &addr, I inc,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename A, typename I, typename Ir, typename Ic>
    void incDecAddr(const A &addr, I inc, Ir incR, Ic incC,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state, bool decrement);
    template <typename A, typename I>
    void incDecAddr(const A &addr, I inc,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state, bool decrement);

    void extendIndexVec(int n, CommonState &state);
    ngen::Subregister accessIndexVec(int n, CommonState &state);

    LDMultiples createLDMultiples(bool a64, int nmultiples,
            const ngen::Subregister &ld, const CommonStrategy &strategy,
            CommonState &state);
    ngen::Subregister findLDMultiple(const LDMultiples &multiples, bool a64,
            int n, const CommonStrategy &strategy, CommonState &state);

    void setupCAddr0(ngen::GRFRange (&C_addr0)[2],
            ngen::GRFRange (&C_addr0Unmasked)[2],
            const std::vector<RegisterBlock> &C_layout,
            const std::vector<RegisterBlock> &C_layoutUnmasked, int C_count,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state, const Address2DParams *params = nullptr);

    void outerProductGen9IGEMM(int ha, int hb,
            const std::vector<RegisterBlock> &A_layout,
            const std::vector<RegisterBlock> &B_layout,
            const GRFMultirange &A_regs, const GRFMultirange &B_regs,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void outerProductSystolic(int h, int ha, int hb,
            const std::vector<RegisterBlock> &A_layout,
            const std::vector<RegisterBlock> &B_layout,
            const GRFMultirange &A_regs, const GRFMultirange &B_regs,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void outerProduct(int h, int ha, int hb, int opCount,
            const std::vector<RegisterBlock> &A_layout,
            const std::vector<RegisterBlock> &B_layout,
            const GRFMultirange &A_regs, const GRFMultirange &B_regs,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void setupTeardownAccumulateSumSystolic(bool setup, Type Tother,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);

    void updateC(const GRFMultirange &C_acc, const GRFMultirange &C_accSwap,
            const GRFMultirange &C_load, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void updateCLayout(const std::vector<RegisterBlock> &layoutExt,
            const ngen::GRFRange (&C_addr0)[2], const RegisterBlock &C_block0,
            COperation op, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    bool doStdCRemainder(std::vector<RegisterBlock> &layoutExt,
            std::vector<RegisterBlock> &layoutExtUnmasked, bool inside,
            bool columns[2], StdCRemType remTypes[2], bool fragments[2],
            bool fragPositives[2], int fragSizes[2],
            const ngen::GRFRange (&C_addr0)[2],
            const ngen::GRFRange (&C_addr0Unmasked)[2], COperation op,
            std::vector<MaskAssignment> &masks, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState state,
            RegisterBlock *C_block0 = nullptr,
            RegisterBlock *C_blockUnmasked0 = nullptr);
    void doAlternateCRemainder(COperation op, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);

    void accumulateSum(bool column, Type Tsrc, const GRFMultirange &srcRegs,
            const std::vector<RegisterBlock> &srcLayout, Type Tdst,
            const GRFMultirange &dstRegs,
            const std::vector<RegisterBlock> &dstLayout,
            const CommonStrategy &strategy, CommonState &state, int q0 = -1,
            int q1 = -1);
    void makeSumLayout(bool column, Type Tsrc,
            const std::vector<RegisterBlock> &srcLayout, Type Tdst,
            std::vector<RegisterBlock> &dstLayout,
            const CommonStrategy &strategy, CommonState &state);
    void horizontalAdd(bool column, Type T, const GRFMultirange &regs,
            std::vector<RegisterBlock> &layout, CommonState &state);
    bool gemmFinalizeSums(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);

    CoopSplit effCoopSplitA(
            const GEMMProblem &problem, const GEMMStrategy &strategy);
    CoopSplit effCoopSplitB(
            const GEMMProblem &problem, const GEMMStrategy &strategy);

    void convert(const GRFMultirange &range, Type Told, Type Tnew,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    bool gemmConvertC(Type Tnew, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmAlphaScale(GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state, bool cxCombine = true);
    void gemmBetaScale(const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void binaryOp(BinaryOp op, int simd, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1);
    void gemmScalarBinaryOpC(BinaryOp op, const ngen::Subregister &offset,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void gemmVectorBinaryOpC(BinaryOp op, bool column,
            const GRFMultirange &offsets, const ngen::Subregister &scale,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state, Type Tco = Type::invalid,
            std::vector<RegisterBlock> CO_layout = std::vector<RegisterBlock>(),
            int y0 = -1, int y1 = -1);
    void gemmCalcABOffsetAddrs(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    bool gemmLoadABOffset(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmApplyABOffset(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmAccessSums(COperation op, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    bool gemmBinaryOpC(BinaryOp op, bool row, bool column, Type Tco,
            MatrixAddressing CO, MatrixAddressingStrategy CO_strategy,
            ngen::Subregister base, ngen::Subregister ld,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    bool gemmApplyCOffsetDispatch(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmKReduce(const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void gemmPrefetchC(const GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state);

    void gemmApplyPostOps(int poMin, int poMax, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmLoadBinaryOpArgs(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);

    void gemmAllocRegs(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    void gemmAllocAoBoRegs(const GEMMStrategy &strategy, GEMMState &state);
    void gemmAIncrementInternal(Type Ta,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &A,
            const MatrixAddressingStrategy &A_strategy, int ka_inc,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state, int ha = 0);
    void gemmAIncrementInternal(Type Ta,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &A,
            const MatrixAddressingStrategy &A_strategy,
            const MultishiftSubregister &ka_inc, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int ha = 0);
    void gemmAIncrementInternal(Type Ta,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &A,
            const MatrixAddressingStrategy &A_strategy,
            const ngen::Subregister &ka_inc, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int ha = 0);
    template <typename I>
    void gemmAIncrement(Type Ta, const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &A,
            const MatrixAddressingStrategy &A_strategy, I ka_inc,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state, int ha = 0, int h = 0);
    void gemmALoad(const GRFMultirange &regs,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &A,
            const MatrixAddressingStrategy &A_strategy,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    template <typename I>
    void gemmALoadInc(Type Ta, const GRFMultirange &regs,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &A,
            const MatrixAddressingStrategy &A_strategy, I ka_inc,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void gemmBIncrementInternal(Type Tb,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &B,
            const MatrixAddressingStrategy &B_strategy, int kb_inc,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state, int hb = 0);
    void gemmBIncrementInternal(Type Tb,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &B,
            const MatrixAddressingStrategy &B_strategy,
            const MultishiftSubregister &kb_inc, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int hb = 0);
    void gemmBIncrementInternal(Type Tb,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &B,
            const MatrixAddressingStrategy &B_strategy,
            const ngen::Subregister &kb_inc, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int hb = 0);
    template <typename I>
    void gemmBIncrement(Type Tb, const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &B,
            const MatrixAddressingStrategy &B_strategy, I kb_inc,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state, int hb = 0, int h = 0);
    void gemmBLoad(const GRFMultirange &regs,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &B,
            const MatrixAddressingStrategy &B_strategy,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    template <typename I>
    void gemmBLoadInc(Type Tb, const GRFMultirange &regs,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &B,
            const MatrixAddressingStrategy &B_strategy, I kb_inc,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    template <bool doA>
    void gemmAiBiRemLoadInc(bool incremental, bool incrementalCopy,
            bool keepAddrTogether, bool willRemask,
            const ngen::Subregister &kSLMX, const GRFMultirange &Xi_regs,
            const std::vector<RegisterBlock> &Xi_layout,
            const std::vector<ngen::GRFRange> &Xi_addrs,
            const std::vector<std::vector<RegisterBlock>> &Xi_layoutK,
            const std::vector<std::vector<ngen::GRFRange>> &Xi_addrsK,
            const GRFMultirange &Xo_regs,
            const std::vector<RegisterBlock> &Xo_layout,
            const MatrixAddressing &Xi,
            const MatrixAddressingStrategy &Xi_strategy,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    SubregisterPair allocIncrement(
            const GEMMStrategy &strategy, CommonState &state);
    void gemmCalcIncrements(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int ka_load = 0,
            int kb_load = 0, bool doA = true, bool doB = true);
    void gemmCalcWorkshareAOffset(ngen::Subregister &off,
            ngen::Subregister &offR, ngen::Subregister &offC,
            const MatrixAddressing &A,
            const MatrixAddressingStrategy &A_strategy, int ma, int ka,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void gemmCalcWorkshareBOffset(ngen::Subregister &off,
            ngen::Subregister &offR, ngen::Subregister &offC,
            const MatrixAddressing &B,
            const MatrixAddressingStrategy &B_strategy, int kb, int nb,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    bool gemmPrepMaskedAB(const GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state);
    void gemmSLMRemask(bool remaskA, bool remaskB, GRFMultirange &Ao_regs,
            GRFMultirange &Bo_regs, int kOffset, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);

    void gemmCalcKLoopBarrierCount(ngen::Subregister &count,
            const ngen::Subregister &k, int cooldown,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void gemmCalcKSLM(const ngen::Subregister &kSLM,
            const ngen::Subregister &lid, int kgran, int kdiv, int krep,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state, ngen::Subregister kBase = ngen::Subregister());
    void gemmCalcKSLMA(const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state, ngen::Subregister kBase = ngen::Subregister());
    void gemmCalcKSLMB(const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state, ngen::Subregister kBase = ngen::Subregister());
    void kLoopAllocBarrierHeader(GEMMState &state);
    ngen::GRF kLoopGetBarrierHeader(GEMMState &state);
    void kLoop(KLoop type, const GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state);
    bool kLoopSetup(const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void kLoopTeardown(const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    bool kLoopSingle(KLoop type, const GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state);
    void kLoopActivateABRemainder(bool active, bool doA, bool doB,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state, int kOffset = 0);
    void kLoopActivateSLMRemainder(bool active, bool preactivate,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state, int kOffset = 0);
    bool gemmKLoop(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    void gemmAllocateTokens(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmABPrefetchAddrSetup(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, bool doA = true,
            bool doB = true);
    bool gemmAccumulateC(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    bool gemmAccumulateCSetup(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    void gemmAccumulateCTeardown(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    bool gemmAccessC(COperation op, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    bool gemmUpdateC(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    bool gemmBody(GEMMProblem problem, GEMMStrategy strategy, GEMMState state);
    bool gemmBodyInternal(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    bool wgRemCheck(const GEMMProblem &problem, const GEMMStrategy &strategy);
    template <typename Problem>
    bool mnRemainderHandling(LoopType loop, Problem &problem,
            GEMMStrategy &strategy, GEMMState &state,
            bool (gemm_kernel_generator_t<hw>::*func)(
                    Problem, GEMMStrategy, GEMMState));
    template <typename Problem>
    bool mnJointSplitRemainderHandling(Problem &problem, GEMMStrategy &strategy,
            GEMMState &state,
            bool (gemm_kernel_generator_t<hw>::*func)(
                    Problem, GEMMStrategy, GEMMState));
    bool gemmMEdge(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    bool gemmNEdge(GEMMProblem problem, GEMMStrategy strategy, GEMMState state);

    void gemmSimpleLinearOrder(const GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state);
    void gemmHilbertlikeOrder(const GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state);
    void gemmBoustrophedonOrder(const GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state);
    void gemmReorderLocalIDs(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);

    ngen::Subregister gemmCalcKPadding(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void broadcastToWG(ngen::FlagRegister leaderFlag, ngen::GRF value,
            CommonState &state, int slmOffset = 0);
    void gemmFusedBetaPOInit(const ngen::Subregister &groupID,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void gemmFusedBetaScale(
            GEMMProblem problem, GEMMStrategy strategy, GEMMState &state);
    void gemmFusedBetaCalcWGCount(const ngen::Subregister &count,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void gemmFusedBetaNotifyCompletion(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmFusedBetaWaitCompletion(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmRedirectToTempC(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    void gemmCheck32(const GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state);
    void gemmGetBatchIDs(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmReleaseBatchIDs(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmOffsetAk(int h, const ngen::Subregister &effA,
            const MatrixAddressing &globalA, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmOffsetAk(const ngen::Subregister &h, const ngen::Subregister &effA,
            const MatrixAddressing &globalA, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmOffsetBk(int h, const ngen::Subregister &effB,
            const MatrixAddressing &globalB, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmOffsetBk(const ngen::Subregister &h, const ngen::Subregister &effB,
            const MatrixAddressing &globalB, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmFoldOffsets(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmRestoreOffsets(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmOffsetABC(bool initial, ngen::Subregister i0, ngen::Subregister j0,
            ngen::Subregister h0, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, bool doA = true,
            bool doB = true, bool doC = true, bool doBinary = false);
    void gemmOffsetBatchABC(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmSetupABC(const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void gemmCacheLDABMultiples(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, bool doA = true,
            bool doB = true);
    void gemmCacheLDCMultiples(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state,
            bool prefetch = false);
    void gemmScaleInputs(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmReverseLoops(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmDowngradeAccess(const GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state);
    void gemmSubkernel(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState state);
    static size_t gemmSLMSize(
            const GEMMProblem &problem, const GEMMStrategy &strategy);
    static size_t gemmPerKSLMSize(
            const GEMMProblem &problem, const GEMMStrategy &strategy);
    void gemmInitInterface(GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state, bool inSK = false);
    void gemmInitState(GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state, bool inSK = false);
    void gemmAutoTypeConversions(
            GEMMProblem &problem, const GEMMStrategy &strategy);
    void gemm(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    void gemmSuperkernelInitState(GEMMSuperkernelProblem &problem,
            GEMMSuperkernelStrategy &strategy, GEMMSuperkernelState &state);

    bool sysgemmAccumulateC(GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void sysgemmKLoop(const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void sysgemmKLoop4(const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state, bool oddB);
    void sysgemmStoreSignal(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state,
            bool forceFence = false);
    void sysgemmCopyLoad(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int storeBuffer,
            bool useC = false);
    void sysgemmCopyLoad4(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int storeBuffer,
            bool loadB, int useC = 0,
            ngen::RegData flagLoadB = ngen::RegData());
    void sysgemmCopyStore(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int storeBuffer,
            bool first = false);
    void sysgemmCopyStore4(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int storeBuffer,
            bool storeB, int useC = 0, int useC_B = 0);
    void sysgemmMultiply(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int buffer,
            bool lastMultiply = false);
    void sysgemmMultiply4(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int buffer,
            bool firstMultiply = false,
            ngen::RegData flagWaitLoad = ngen::RegData(),
            ngen::RegData flagSignal = ngen::RegData(),
            ngen::Label *labelDone = nullptr);
    void sysgemmMultiplyChunk(const GEMMProblem &problem,
            const GEMMStrategy &strategy, bool first, int ao, int i0,
            bool waitB, bool prepB,
            const ngen::InstructionModifier &swsb0
            = ngen::InstructionModifier(),
            const ngen::InstructionModifier &swsbEnd
            = ngen::InstructionModifier());
    void sysgemmBarrierPrep(
            const ngen::InstructionModifier &swsb, const ngen::GRF &header);
    void sysgemmReorderLocalIDs(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);

    bool sysgemm2AccumulateC(GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void sysgemm2KLoopCompute(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void sysgemm2KLoopCopy(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void sysgemm2Multiply(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int buffer,
            bool cooldown = false,
            ngen::FlagRegister flagWaitLoad = ngen::FlagRegister(),
            ngen::FlagRegister flagSignal = ngen::FlagRegister());
    void sysgemm2MultiplyX32(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int buffer,
            bool cooldown = false,
            ngen::FlagRegister flagWaitLoad = ngen::FlagRegister(),
            ngen::FlagRegister flagSignal = ngen::FlagRegister());
    void sysgemm2MultiplyX48(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int buffer,
            bool cooldown = false,
            ngen::FlagRegister flagWaitLoad = ngen::FlagRegister(),
            ngen::FlagRegister flagSignal = ngen::FlagRegister());
    void sysgemm2MultiplyChunkX32(const GEMMProblem &problem,
            const GEMMStrategy &strategy, int chunkA, bool odd);
    void sysgemm2MultiplyChunkX48(const GEMMProblem &problem,
            const GEMMStrategy &strategy, int chunkA);

    bool copyRegisterBlock(Type Ts, Type Td, const RegisterBlock &blockSrc,
            const RegisterBlock &blockDst, const GRFMultirange &src,
            const GRFMultirange &dst, int dOffR, int dOffC,
            const CommonStrategy &strategy, CommonState &state,
            bool preserveSrc = false);
    bool copyRegisters(Type Ts, Type Td,
            const std::vector<RegisterBlock> &layoutSrc,
            const std::vector<RegisterBlock> &layoutDst,
            const GRFMultirange &src, const GRFMultirange &dst, int dOffR,
            int dOffC, bool conjugate, const CommonStrategy &strategy,
            CommonState &state, bool preserveSrc = false);
    bool copyRegisters(Type Ts, Type Td,
            const std::vector<RegisterBlock> &layoutSrc,
            const std::vector<RegisterBlock> &layoutDst,
            const GRFMultirange &src, const GRFMultirange &dst, int dOffR,
            int dOffC, const Scalar &alpha, const SubregisterPair &alpha_real,
            const SubregisterPair &alpha_imag, bool conjugate,
            const CommonStrategy &strategy, CommonState &state,
            bool preserveSrc = false);

    bool copyBody(
            CopyProblem &problem, CopyStrategy &strategy, CopyState &state);
    bool copyBodyRemCheck(
            CopyProblem &problem, CopyStrategy &strategy, CopyState &state);
    bool copyBodyInternal(
            CopyProblem &problem, CopyStrategy &strategy, CopyState &state);
    void copySlice(
            CopyProblem &problem, CopyStrategy &strategy, CopyState &state);

    void copyCalcIncrements(const CopyProblem &problem,
            const CopyStrategy &strategy, CopyState &state, int s_load = 0,
            int d_load = 0);

    void copyInitInterface(
            CopyProblem &problem, CopyStrategy &strategy, CopyState &state);
    void copyInitState(
            CopyProblem &problem, CopyStrategy &strategy, CopyState &state);
    void copy(CopyProblem &problem, CopyStrategy &strategy, CopyState &state);

    void prologue(const CommonStrategy &strategy, int internalSIMD = 16);
    void prologue(const GEMMStrategy &strategy, GEMMState &state);
    void epilogue(const CommonStrategy &strategy, CommonState &state);
    void padding();
    void initInterface(const CommonProblem &problem,
            const CommonStrategy &strategy, CommonState &state);
    void initState(const CommonProblem &problem, const CommonStrategy &strategy,
            CommonState &state);
};

inline char precisionChar(Type T) {
    switch (T.baseType()) {
        case Type::f16: return 'H';
        case Type::f32: return 'S';
        case Type::u8: return 'o';
        case Type::s8: return 'O';
        case Type::u16: return 'w';
        case Type::s16: return 'W';
        case Type::u32: return 'i';
        case Type::s32: return 'I';
        case Type::u64: return 'l';
        case Type::s64: return 'L';
        case Type::bf16: return 'B';
        case Type::tf32: return 'T';
        default: return '?';
    }
}

static inline Type charPrecision(char c) {
    switch (c) {
        case 'H': return Type::f16;
        case 'S': return Type::f32;
        case 'o': return Type::u8;
        case 'O': return Type::s8;
        case 'w': return Type::u16;
        case 'W': return Type::s16;
        case 'i': return Type::u32;
        case 'I': return Type::s32;
        case 'B': return Type::bf16;
        case 'T': return Type::tf32;
        default: return Type::invalid;
    }
}

inline char layoutChar(MatrixLayout layout) {
    switch (layout) {
        case MatrixLayout::N: return 'N';
        case MatrixLayout::T: return 'T';
        case MatrixLayout::Pc: return 'A';
        case MatrixLayout::Pr: return 'B';
        default: return '?';
    }
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif /* header guard */
