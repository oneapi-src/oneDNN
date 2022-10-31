/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

static inline bool isLargeCrosspack(size_t sizeofT, int crosspack) {
    return (crosspack * sizeofT > 4) && (crosspack > 1);
}

static inline bool isLargeCrosspack(Type T, int crosspack) {
    return isLargeCrosspack(T.size(), crosspack);
}

enum class AccessType : uint8_t {
    Scattered, // Use scattered accesses
    ChannelScattered, // Use untyped surface reads
    Block, // Use block messages
    PseudoBlock, // Use scattered accesses to emulate block accesses
    Block2D, // Use 2D block messages
    Block2DTranspose, // Use 2D block messages with transposition
    Block2DVNNI, // Use 2D block messages with VNNI transform
};

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

// Preferences for using scattered accesses.
enum class ScatterSIMD {
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

    SubregisterPair operator-() const {
        auto copy = *this;
        copy.negative = !copy.negative;
        return copy;
    }
};

template <typename T>
class Scalar {
protected:
    bool fixed_value;
    union {
        SubregisterPair subs;
        T value;
    };

public:
    Scalar() : Scalar(ngen::Subregister()) {}
    explicit Scalar(T value_) : fixed_value(true), value(value_) {}
    Scalar(ngen::Subregister reg0, ngen::Subregister reg1)
        : fixed_value(false), subs {reg0, reg1} {}
    explicit Scalar(ngen::Subregister reg) : Scalar(reg, reg) {}

    Scalar &operator=(T value_) {
        fixed_value = true;
        value = value_;
        return *this;
    }
    Scalar &operator=(ngen::Subregister reg) {
        fixed_value = false;
        subs = reg;
        return *this;
    }

    template <typename U>
    friend inline bool operator==(const Scalar<T> &scalar, const U &val) {
        return scalar.fixed_value && (val == scalar.value);
    }
    template <typename U>
    friend inline bool operator==(const U &val, const Scalar<T> &scalar) {
        return scalar == val;
    }

    template <typename U>
    friend inline bool operator!=(const Scalar<T> &scalar, const U &val) {
        return !(scalar == val);
    }
    template <typename U>
    friend inline bool operator!=(const U &val, const Scalar<T> &scalar) {
        return !(scalar == val);
    }

    operator T() const {
        if (!fixed_value) throw std::runtime_error("Scalar is not fixed.");
        return value;
    }

    operator SubregisterPair() const {
        if (fixed_value) throw std::runtime_error("Scalar is fixed.");
        return subs;
    }

    SubregisterPair &getPair() {
        if (fixed_value) throw std::runtime_error("Scalar is fixed.");
        return subs;
    }

    bool fixed() const { return fixed_value; }

    ngen::Subregister getReg(int idx) const {
        return SubregisterPair(*this).getReg(idx);
    }
    ngen::Subregister getRegAvoiding(
            ngen::HW hw, const ngen::RegData &rd) const {
        return SubregisterPair(*this).getRegAvoiding(hw, rd);
    };
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
    unsigned padded : 1; // Allow read/write overruns?
    unsigned atomic : 1; // Atomic access? (only relevant for C)
    unsigned address2D : 1; // Use 2D addressing? (media block-style loads)
    unsigned prefetch : 1; // Prefetch only?
    unsigned newDP : 1; // Use new dataport messages? (XeHPG+)
    unsigned dpasw : 1; // DPASW half layout?
    ngen::CacheSettingsLSC cachingR // Cache policies for LSC reads.
            = ngen::CacheSettingsLSC::Default;
    ngen::CacheSettingsLSC cachingW // Cache policies for LSC writes.
            = ngen::CacheSettingsLSC::Default;

    MatrixAddressingStrategy()
        : padded(false)
        , atomic(false)
        , address2D(false)
        , prefetch(false)
        , newDP(false)
        , dpasw(false) {}

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
};

struct MaskInfo {
    union {
        struct {
            uint8_t isFixed : 1; // = false (variable mask)
            uint8_t reverse : 1; // True to reverse mask.
            uint8_t : 6;
            uint8_t rsize; // Maximum remainder value. (e.g. 16 if we need the last 4 bits of the index).
            uint8_t maskRep; // # of repetitions of mask pattern.
            uint8_t bitRep : 5; // # of times each mask bit is repeated.
            uint8_t rdivide : 3; // Amount by which to divide index before forming mask. Fractions are rounded up.
                    // Note maskRep * bitRep * (rsize >> rshift) = # mask bits.
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
    uint8_t component; // Component # for this block.
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
    VirtualFlag flag; // Assigned flag register index and modifiers, if any.
    uint8_t flagAny : 1; // Use .anyh?
    uint8_t flagAll : 1; // Use .allh?
    uint8_t hasNoLoad : 1; // Does this load/store cover additional (no-load) RegisterBlocks? (packed layouts)
    uint8_t : 5;
    uint8_t sfid; // SFID for this block.
    uint8_t rowFragment; // If this block needs fragmenting to support row/column remainders, the maximum block size (power of 2) to fragment down to.
    uint8_t colFragment; //     Zero if no fragmenting needed.
    uint8_t addrShift; // log2(address units). e.g. 0 if byte addresses should be used, 4 if oword addresses should be used.
    uint8_t log2GRFBytes; // log2(bytes per GRF).

    MaskInfo rowMask; // Row mask for this block.
    MaskInfo colMask; // Column mask for this block.

    void calcBytes(Type T); // Auto-calculate # of registers.
    void calcBytes(Type T, const MatrixAddressingStrategy &astrategy);

    void clearFlag() {
        flag.clear();
        flagAll = flagAny = false;
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
        : free((1ul << (ngen::GRF::bytes(hw) >> 1)) - 1)
        , nflag(ngen::FlagRegister::subcount(hw)) {}

    VirtualFlag allocVirtual(int n = 1);
    ngen::FlagRegister alloc(int n = 1);
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
    bool isLocked(VirtualFlag vflag) const { return (locked & mask(vflag)); }

    ngen::FlagRegister assignPhysical(VirtualFlag vflag);

    static int getBase(int idx) { return idx & 0x1F; }
    static int getN(int idx) { return idx >> 5; }
    static int makeIndex(int base, int n) { return base | (n << 5); }

protected:
    uint32_t free;
    uint8_t locked = 0;
    uint8_t nextPhys = 0;
    uint8_t nflag;

    static uint32_t mask(VirtualFlag vflag) { return mask(vflag.idx, vflag.n); }
    static uint32_t mask(int idx, int n) {
        return (1ul << (idx + n)) - (1ul << idx);
    }
};

class TokenAllocator {
public:
    TokenAllocator(ngen::HW hw);

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
    ngen::GRF vflagStorage;
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
        ngen::GRFRange mathTemp;
        ngen::GRF temp;
        std::array<ngen::FlagRegister, 2> tempFlags;
        ngen::Subregister flagStore; // ud
        ngen::Label label;
        int simd;
        ngen::Subregister callStorageSub, callStorage;
        bool use = false;
    } invertSub;

    CommonState(ngen::HW hw) : ra(hw), raVFlag(hw), tokenAllocator(hw) {}

    void wipeActiveVFlags() {
        for (int i = 0; i < int(activeVFlags.size()); i++)
            if (!raVFlag.isLocked(VirtualFlag(i))) activeVFlags[i].clear();
    }

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
    bool wgInSS
            = false; // Pretend to use barriers so that each WG belongs to 1 SS/DSS.
    int GRFs = 128; // # of GRFs to use.
    bool finalFence = false; // Issue global memory fence before EOT.
    int pauseCycles
            = 0x0200; // Number of cycles to pause when waiting in a spin-loop.
    bool simulation = false; // For use in simulator?

    EmulationStrategy emulate;

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

    Scalar<double> alpha_real, alpha_imag; // Alpha value, if fixed.
    Scalar<double> beta_real, beta_imag; // Beta value, if fixed.
    MatrixAddressing A, B, C, CO; // Addressing information for matrices.
    bool checkBeta0 = true; // If true, check for beta = 0 and handle specially.
    ABOffset abOffset = ABOffset::None; // A/B offset mode.
    COffset cOffset = COffset::None; // C offset mode.
    BatchMode batch = BatchMode::None; // Batch mode.
    int batchDims = 0; // # of batch dimensions (strided batch only).
    bool sumA = false,
         sumB
            = false; // If true, calculate A row sums/B column sums and store in CO.
    post_ops_t post_ops; // Fused post operations to apply
    bool postOpFwd = true; // Eltwise parameters
    std::vector<MatrixAddressing> binary; // Binary postop data
    std::vector<Type> Tbinary; // Binary types
    std::vector<bool> binaryRow; // Dimensionality of binary data
    std::vector<bool>
            binaryCol; //    (false means broadcast in the given dimension)

    bool hasPostOp() const { return post_ops.len() > 0; }
    bool hasEltwisePostOp() const {
        for (int idx = 0; idx < post_ops.len(); idx++)
            if (post_ops.entry_[idx].is_eltwise()) return true;
        return false;
    }
    int binaryPOCount() const {
        int count = 0;
        for (int idx = 0; idx < post_ops.len(); idx++)
            count += int(post_ops.entry_[idx].is_binary());
        return count;
    }

    bool beta0() const {
        return (beta_real == 0) && (!Tc.isComplex() || (beta_imag == 0));
    }
    bool beta1() const {
        return (beta_real == 1) && (!Tc.isComplex() || (beta_imag == 0));
    }
    bool alpha1() const {
        return (alpha_real == 1) && (!Tc.isComplex() || (alpha_imag == 0));
    }
    bool alphaM1() const {
        return (alpha_real == -1) && (!Tc.isComplex() || (alpha_imag == 0));
    }

    bool needsTsConvert() const {
        if (!(alpha1() || alphaM1())) return true;
        if (!(beta0() || beta1())) return true;
        if (beta1() && !Tc_ext.isSubsetOf(Tc)) return true;
        if (hasPostOp()) return true;
        return false;
    }

    bool gemmt() const { return false; }
    bool backward() const { return false; }

    bool needsASums() const { return (abOffset == ABOffset::Calc) || sumA; }
    bool needsBSums() const { return (abOffset == ABOffset::Calc) || sumB; }
    bool usesCO() const { return (cOffset != COffset::None) || sumA || sumB; }
    bool allowMatrixOffset() const { return (cOffset == COffset::Pre); }
};

struct GEMMState;

// How to split A/B amongst threads in a workgroup.
enum class CoopSplit {
    K, // Split in k dimension
    MN, // Split in m/n dimensions
    Linear, // Split in linear index order
};

// Strategy parameters for GEMM kernels.
struct GEMMStrategy : public CommonStrategy {
    int blocking[3] = {
            0}; // Recommended block size in each dimension (m/n/k) -- for driver.
    int blockingAlt[3] = {
            0}; // Alternate block size in each dimension (m/n/k) -- for driver.
    //     m/n alternates are for Hilbert-ordered kernels when Hilbert ordering disabled.
    //     k alternate is for multi-tile execution with implicit scaling.
    int unroll[3]; // Unrolls in each dimension (m/n/k), indexed by LoopType.
    int unrollK_masked = 0; // k unroll to use when masking.
    LoopType loopOrder[3] = {LoopM, LoopN,
            LoopK}; // Expected order of loops in driver code (in order from innermost to outermost).
    LoopType fusedLoop = LoopM; // Direction of fusing if threads fused.
    bool hilbertOrder = false; // Use Hilbert-like walk order in C?
    bool boustrophedon = false; // Use panel-boustrophedon walk order in C?
    bool persistent = false; // Use persistent thread model?
    bool reverse[2] = {false, false}; // Reverse m/n walk order?
    int fmaSIMD = 0; // Vector length for FMA (0 = default = 2 GRFs).
    int kChain = 1; // # of FMAs to chain in k dimension.
    int wg[3] = {0, 0,
            0}; // m/n/k workgroup sizes, 0 if unconstrained. Indexed by LoopType.
    WGType forceWGUpdate = WGDynamic; // Force work group update type.
    MatrixAddressingStrategy A, B, C,
            CO; // Strategies for accessing A/B/C/C offsets.
    int ka_load, kb_load; // How much of A/B is loaded at once, in k dimension
    int ka_load_masked = 0,
        kb_load_masked
            = 0; // Same as above, when masking m/n (0 = default = same as ka/kb_load)
    bool slmA = false, slmB = false; // Whether to copy A/B to SLM.
    bool splitCopy = false; // Separate SLM copy and compute threads?
    int slmBuffers = 0; // # of A/B SLM buffers, 0 for none.
    int unrollKSLM
            = 0; // k unroll for SLM copies (0 = auto = unroll[LoopK]/slmCopies)
    int unrollKSLMMasked
            = 0; //   Alternate value to use with masking (0 = same as unrollKSLM)
    bool slmATrans = false,
         slmBTrans
            = false; // Whether A/B SLM data should be completely crosspacked (transposed).
    int A_copies = 1,
        B_copies = 1; // # of copies of A/B matrices, for latency absorption
    int slmCopies = 1; // # of copies of loaded A/B matrices for SLM copies.
    bool slmRepackAhead = false; // Repack SLM data ahead of stores?
    int optAlignAB
            = 0; // Optional alignment for A/B. If > 0, create two versions of k loop, one for A/B aligned to this value, one not.
    AccessType unalignedAccA,
            unalignedAccB; // Access types to use for A/B on unaligned path.
    int ka_prefetch = 0, kb_prefetch = 0; // Chunk size for prefetching A/B.
    int ka_pfStride = 0, kb_pfStride = 0; // k stride between A/B prefetches.
    bool cooperativePF = true; // Enable WG-cooperative A/B prefetches.
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
    bool avoidIncConflicts
            = true; // If true, duplicate some increment values across banks to avoid bundle conflicts.
    bool kParallel
            = false; // If true, generate k-parallelized kernel using global memory reduction.
    bool kParallelLocal
            = false; // If true, generate k-parallelized kernel using local memory reduction.
    bool doubleWA
            = false; // Use explicit double broadcast instructions? (Gen9 only)
    int barrierFreq
            = 0; // If > 0, set a periodic barrier every barrierFreq k loops to keep threads together.
    bool splitBarrier
            = false; //   Use split barriers for these periodic barriers?
    bool altCRemainder = false; // Use alternative double-loop C remainder code?
    bool block2DCRemainder = false; // Generate block 2D C remainder path?
    bool cAccumulators
            = false; // Use accumulator registers for part of C (to save a few registers)?
    bool cLoadAhead = false; // Load C before doing FMAs?
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
    int namedBarriers[2] = {0,
            0}; // # of named barriers in m, n dimensions (0 to use regular barriers).
    bool skewLocalIDs
            = false; // Remap local IDs for large workgroups so that threads on the same EU don't depend on the same data.
    bool xParallel = false; // TRSM: parallelize in x dimension.
    bool checkBeta1
            = false; // If true, check for beta = 1 and handle specially.
    std::vector<MatrixAddressingStrategy>
            binary; // Strategies for binary postop data

    bool insideSK = false; // Inside a superkernel?

    GEMMStrategy() {}
    GEMMStrategy(ngen::HW hw, int stepping = 0)
        : CommonStrategy(hw, stepping) {}

    void preflight(ngen::HW hw, const GEMMProblem &problem);
    bool minimize(ngen::HW hw, const GEMMProblem &problem);

    bool lateExit() const {
        return (slmBuffers > 0) || barrierFreq || kParallelLocal
                || (cooperativePF && (prefetchA || prefetchB));
    }

    int maxKSLM(const GEMMProblem &problem, bool isA) const;
    int slmABufBlockSize(const GEMMProblem &problem) const {
        return fixedSystolic ? 1152
                             : int(slmA) * problem.Ta * problem.Ta.components()
                        * unroll[LoopM] * maxKSLM(problem, true);
    }
    int slmBBufBlockSize(const GEMMProblem &problem) const {
        return fixedSystolic ? 1536
                             : int(slmB) * problem.Tb * problem.Tb.components()
                        * unroll[LoopN] * maxKSLM(problem, false);
    }
    int slmABufSize(const GEMMProblem &problem) const {
        return slmABufBlockSize(problem) * wg[LoopM] * wg[LoopK] * slmBuffers;
    }
    int slmBBufSize(const GEMMProblem &problem) const {
        return slmBBufBlockSize(problem) * wg[LoopN] * wg[LoopK] * slmBuffers;
    }
    int slmSysgemmBlockSize() const {
        return 1152 * wg[LoopM] + 1536 * wg[LoopN];
    }
    bool variableSLM() const { return kParallelLocal; }

    int ka_inc() const { return slmA ? unrollKSLM : ka_load; }
    int kb_inc() const { return slmB ? unrollKSLM : kb_load; }

    bool needsMNLocalIDs() const {
        return xParallel || (slmBuffers > 0) || cooperativePF || kParallelLocal
                || persistent || namedBarriers[0] || (dpasw && !fixedSystolic);
    }
    bool needsKLocalIDs() const { return kParallelLocal || persistent; }
    bool needsBarrier() const {
        return (barrierFreq > 0) || (slmBuffers > 0) || xParallel
                || kParallelLocal;
    }

    bool fusedM() const { return fused && (fusedLoop == LoopM); }
    bool fusedN() const { return fused && (fusedLoop == LoopN); }

    WGType getWGType(const GEMMProblem &problem) const {
        if ((slmBuffers > 0) || (forceWGUpdate == WGFixed)
                || (barrierFreq && namedBarriers[0]))
            return WGFixed;
        else
            return WGDynamic;
    }

    bool fixedWG(const GEMMProblem &problem) const {
        return (getWGType(problem) == WGFixed);
    }
    bool linearOrder() const { return hilbertOrder || boustrophedon; }
};

struct LDMultiples {
    ngen::GRFRange range;
    bool a64 = false;
};

// State parameters for GEMM kernels.
struct GEMMState : public CommonState {
    struct Inputs {
        ngen::Subregister A, B, C[2], CO, base; // q
        ngen::Subregister ao, bo, abo; // w/w/ud
        ngen::Subregister offsetA, offsetB, offsetC[2]; // q
        ngen::Subregister offsetCO; // d
        ngen::Subregister lda, ldb, ldc[2], ldco; // d
        ngen::Subregister m, n, k, k0; // d
        ngen::Subregister alpha_real, alpha_imag; // T_real
        ngen::Subregister beta_real, beta_imag; // T_real
        ngen::Subregister groupIDM, groupIDN, groupIDK; // ud
        ngen::Subregister groupIDMN; // ud
        ngen::GRF localIDM, localIDN, localIDK; // uw
        ngen::Subregister localSizeM, localSizeN, localSizeK; // ud
        ngen::Subregister groupCountM, groupCountN; // ud
        ngen::Subregister groupCountMN; // ud
        ngen::Subregister groupStride; // ud
        ngen::Subregister hilbertVD, hilbertUVDRecip; // ud
        ngen::Subregister hilbertBail; // ud
        ngen::Subregister bslice, bthresh; // d
        ngen::Subregister flags; // ud
        ngen::Subregister diagA, diagB, diagC; // q
        uint8_t surfaceA, surfaceB; // BTS indices
        uint8_t surfaceC[2], surfaceCO; // BTS indices
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
    ngen::Subregister effA, effB, effC[2],
            effCO; // Offsets to base of A/B/C/CO chunks for loading/storing.
    ngen::Subregister effAi, effBi;
    ngen::Subregister effAo, effBo;
    ngen::Subregister effAp, effBp, effCp;
    ngen::Subregister effAs, effBs;
    std::vector<ngen::GRFRange> A_addrs, B_addrs, C_addrs[2];
    std::vector<ngen::GRFRange> A_addrsRem, B_addrsRem;
    std::vector<ngen::GRFRange> Ai_addrs, Bi_addrs;
    std::vector<std::vector<ngen::GRFRange>> Ai_addrsK, Bi_addrsK;
    std::vector<ngen::GRFRange> Ai_addrsRem, Bi_addrsRem;
    std::vector<ngen::GRFRange> Ao_addrs, Bo_addrs;
    std::vector<ngen::GRFRange> Ap_addrs, Bp_addrs, Cp_addrs;
    std::vector<GRFMultirange> A_regs, B_regs, C_regs;
    GRFMultirange Ar_regs, Br_regs; // Repacked A/B registers.
    std::vector<GRFMultirange> Ai_regs,
            Bi_regs; // Incoming data to copy to SLM.
    std::vector<GRFMultirange> Ai_regsRem, Bi_regsRem;
    GRFMultirange Ao_regs, Bo_regs; // Outgoing data to copy to SLM.
    GRFMultirange Ao_regsRem, Bo_regsRem;
    GRFMultirange As_regs, Bs_regs; // A row sums/B column sums.
    GRFMultirange Ap_regs, Bp_regs, Cp_regs; // A/B/C prefetch registers.
    std::vector<MaskAssignment> AB_masks;
    ngen::GRFRange broadcast_regs;
    std::vector<ngen::GRFRange> tempMul_regs;
    ngen::Subregister i0, j0, h0; // d
    ngen::Subregister remainders[3]; // d (todo: w)
    ngen::Subregister remaindersFused[2]; // w
    ngen::Subregister remaindersWG[2]; // d (todo: w)
    ngen::Subregister remFusedStorage; // d
    ngen::Subregister diagC; // d
    SubregisterPair lda, ldb;
    SubregisterPair lda_ka, ldb_kb; // Cached lda * ka, ldb * kb
    SubregisterPair lda_ka_prefetch,
            ldb_kb_prefetch; // Cached lda * ka_pfStride, ldb * kb_pfStride
    LDMultiples ldaMultiples, ldbMultiples, ldcMultiples[2];
    int ka_cached = 0, kb_cached = 0; // Multipliers for lda_ka/ldb_kb.
    ngen::Subregister k, K; // d
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
    bool systolicSumA = false, systolicSumB = false;
    bool lateKLoopCheck = false;
    int ka_loadRem, kb_loadRem;
    bool Ai_hasKRem, Bi_hasKRem;
    bool Ai_lateKRem, Bi_lateKRem;
    bool Ai_incrementalRem, Bi_incrementalRem;
    bool Ai_remIncrCopy, Bi_remIncrCopy;
    int ma_slm, ka_slm, kb_slm, nb_slm;
    int ma_prefetch, ka_prefetch, kb_prefetch, nb_prefetch;
    CoopSplit effCoopA = CoopSplit::K;
    CoopSplit effCoopB = CoopSplit::K;
    std::vector<RegisterBlock> A_layout, B_layout, C_layout;
    std::vector<RegisterBlock> A_layoutRem, B_layoutRem;
    std::vector<RegisterBlock> Ar_layout, Br_layout;
    std::vector<RegisterBlock> Ai_layout, Bi_layout;
    std::vector<std::vector<RegisterBlock>> Ai_layoutK, Bi_layoutK;
    std::vector<RegisterBlock> Ai_layoutRem, Bi_layoutRem;
    std::vector<RegisterBlock> Ao_layout, Bo_layout;
    std::vector<RegisterBlock> As_layout, Bs_layout;
    std::vector<RegisterBlock> Ap_layout, Bp_layout, Cp_layout;
    std::vector<RegisterBlock> C_layoutExt, C_layoutExtUnmasked;
    Address2DParams A_params, B_params;
    Address2DParams Ai_params, Bi_params;
    Address2DParams Ap_params, Bp_params;
    int Ai_regCount = 0, Bi_regCount = 0;
    bool aioShare, bioShare;
    bool aioShareRem, bioShareRem;
    bool aoReuseA = false, boReuseB = false;
    MatrixAddressing Ai, Bi, Ao, Bo;
    MatrixAddressingStrategy Ai_strategy, Bi_strategy;
    MatrixAddressingStrategy Ao_strategy, Bo_strategy;
    MatrixAddressingStrategy Cext_strategy;
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
    int C_count = 1;
    int C_buffers = 1;
    bool allocedAo = false, allocedBo = false;
    bool allowEmptyC = false;
    bool copyC = false;
    bool broadcast;
    bool repackA = false, repackB = false;
    bool repackARem = false, repackBRem = false;
    int ka_repackRem, kb_repackRem;
    bool remActiveA, remActiveB, remActiveSLM;
    std::vector<MaskAssignment> kMasksSLM;
    bool slmRemaskA = false, slmRemaskB = false;
    bool slmASums = false, slmBSums = false;
    bool doLateExit = false;
    ngen::GRF emulate64TempSave[2];

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
    Scalar<double> alpha_real, alpha_imag;
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
    bool duplicateAlpha
            = true; // True to make two copies of alpha, one for each register bank
    bool xLoop
            = false; // True to loop over x, false to loop over y within a kernel

    bool zParallel = false; // Kernel parallelized in z dimension?

    int barrierFreq = 0; // If > 0, set a barrier every barrierFreq loops
    int optionalAlignS
            = 0; // If > 0, generate code to check if S is aligned to this #elements and branch to specific code for that case.

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
        ngen::Subregister alpha_real; // T_real
        ngen::Subregister alpha_imag; // T_real
        ngen::Subregister groupIDW, groupIDZ; // ud
        ngen::GRF localIDW, localIDZ; // uw
        ngen::Subregister localSizeW, localSizeZ; // ud
        ngen::Subregister diag; // d
        ngen::Subregister blockZ; // ud
        uint8_t surfaceS, surfaceD; // DTS indices
    } inputs;
    ngen::Subregister w0, z0; // ud
    ngen::Subregister effS,
            effD; // Offsets to base of S/D chunks for loading/storing.
    ngen::Subregister offsetS1,
            effS1; // Reflected variants of offsetS/effS for symmetric/Hermitian.
    std::vector<ngen::GRFRange> S_addrs, D_addrs;
    std::vector<ngen::GRFRange> S_addrSrcs[2];
    ngen::GRFRange S_regs, D_regs;
    std::vector<ngen::GRFRange> Ds_regs;
    ngen::Subregister lds_sl; // d
    ngen::Subregister ldd_dl; // d
    ngen::Subregister Z; // d
    ngen::FlagRegister flagAP, flagTri, flagDiag;
    ngen::FlagRegister flagReflect;
    std::vector<RegisterBlock> S_layout, D_layout;
    std::vector<RegisterBlock> Ds_layout;
    ngen::Subregister remainderX, remainderY; // ud
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
    enum class COperation { Load, Update, UpdateStore };
    enum class KLoop {
        GEMM,
    };

    friend std::ostream &operator<<(std::ostream &s, StdCRemType rt) {
        const char *names[3] = {"ignore", "mask", "custom descriptor"};
        return (s << names[static_cast<int>(rt)]);
    }

    ngen::FlagRegister getPhysicalFlag(VirtualFlag vflag, CommonState &state);
    void allocVFlagStorage(const CommonStrategy &strategy, CommonState &state);

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
            const CommonStrategy &strategy, CommonState &state);
    template <typename S0>
    void emad(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const S0 &src0, const ngen::RegData &src1, int32_t src2,
            const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void, typename S0, typename S2>
    void eadd3(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const S0 &src0, const ngen::RegData &src1, const S2 &src2);

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
    void alignUp(const ngen::Subregister &dst, const ngen::Subregister &src,
            uint16_t align, const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void>
    void divDown(const ngen::Subregister &dst, const ngen::Subregister &src0,
            const ngen::Subregister &src1, const ngen::Subregister &src1Recip,
            const ngen::FlagRegister &flag, const CommonStrategy &strategy,
            CommonState &state);
    template <typename DT = void>
    void divDown(const ngen::Subregister &dst, const ngen::Subregister &src,
            uint16_t divisor, const CommonStrategy &strategy,
            CommonState &state);

    void simtDoWhileLoop(
            const ngen::InstructionModifier &mod, ngen::Label &dest);
    void slmBarrier(const ngen::GRF &temp, const ngen::GRF &r0_info = r0);
    void globalMemBarrier(const ngen::GRF &temp, const ngen::GRF &r0_info = r0);
    void pause(const CommonStrategy &strategy);

    void duplicateScalar(SubregisterPair &val, CommonState &state);
    void deduplicateScalar(SubregisterPair &val, CommonState &state);
    template <typename T>
    void duplicateScalar(Scalar<T> &val, CommonState &state);
    MultishiftSubregister multishift(const ngen::Subregister &reg,
            unsigned shifts, const CommonStrategy &strategy, CommonState &state,
            ngen::Bundle hint = ngen::Bundle());

    void getFusedID(int scale, const CommonProblem &problem,
            const CommonStrategy &strategy, CommonState &state);
    void moveR0(const CommonStrategy &strategy, CommonState &state);
    void moveR0(const GEMMStrategy &strategy, GEMMState &state);
    template <typename F>
    void useR0(CommonState &state, F f);
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
            bool remainderR, bool remainderC, bool writable, bool avoidFragment,
            int maxRBlock, int maxCBlock, int &rblock, int &cblock,
            RegisterBlock &layout);
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

    bool tryAddMasking(Type T, RegisterBlock &block, bool remainderR,
            bool remainderC, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool tryAddMasking(Type T, std::vector<RegisterBlock> &layout,
            bool remainderR, bool remainderC, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    void addMasking(Type T, std::vector<RegisterBlock> &layout, bool remainderR,
            bool remainderC, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    void addMasking(Type T, std::vector<RegisterBlock> &layout,
            std::vector<ngen::GRFRange> &addrs, const ngen::Subregister &ld,
            bool remainderR, bool remainderC, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state,
            int dataRegs = -1);
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
            bool writable, bool avoidFragment, int maxRBlock, int maxCBlock,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool add1DBlockToRegLayout(Type T, std::vector<RegisterBlock> &layout,
            int r, int c, bool writable, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool getRegLayout(Type T, std::vector<RegisterBlock> &layout, int r, int c,
            bool remainderR, bool remainderC, bool writable, bool avoidFragment,
            int maxRBlock, int maxCBlock, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            bool reverseOrder = false);
    void makeUnbackedRegLayout(Type T, std::vector<RegisterBlock> &layout,
            int r, int c, bool colMajor, int crosspack = 1, int tileR = 0,
            int tileC = 0, bool allowPartialRegs = true);
    bool upgradeLayoutToBlock2D(Type T,
            const std::vector<RegisterBlock> &layoutSrc,
            std::vector<RegisterBlock> &layout2D, bool remainderR,
            bool remainderC, bool writable, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);

    void setupTeardownLoadStoreDesc(bool setup,
            const std::vector<RegisterBlock> &layout,
            const CommonStrategy &strategy, CommonState &state);
    void loadLoadStoreDescriptors(bool load, bool store, RegisterBlock &block,
            const ngen::Subregister &count, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);

    static ngen::DataSpecLSC getDataSpecLSC(
            AccessType access, const RegisterBlock &block);
    static ngen::DataSpecLSC getDataSpecLSC(const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const RegisterBlock &block, bool write);
    ngen::InstructionModifier getRegisterBlockMask(
            const RegisterBlock &block, CommonState &state);
    void loadMatrixBlock(const ngen::Register &dest,
            const RegisterBlock &layout, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const ngen::GRFRange &addr, const CommonStrategy &strategy,
            CommonState &state, bool zeroMask = false);
    void loadMatrix(const GRFMultirange &dest,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const std::vector<ngen::GRFRange> &addrs,
            const CommonStrategy &strategy, CommonState &state,
            bool zeroMask = false);
    void prefetchMatrix(const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const std::vector<ngen::GRFRange> &addrs,
            const CommonStrategy &strategy, CommonState &state);
    void storeMatrixBlock(const ngen::GRF &src, const RegisterBlock &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const ngen::GRFRange &addr, const CommonStrategy &strategy,
            CommonState &state);
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
            const CommonStrategy &strategy, CommonState &state);
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
            bool retryVirtual = false);
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
            size_t sizeofT, const MatrixAddressing &atype,
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
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state,
            const Address2DParams *params = nullptr);

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
            const GRFMultirange &C_load, GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state);
    void updateCLayout(const std::vector<RegisterBlock> &layoutExt,
            const ngen::GRFRange (&C_addr0)[2], const RegisterBlock &C_block0,
            COperation op, GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state);
    bool doStdCRemainder(std::vector<RegisterBlock> &layoutExt,
            std::vector<RegisterBlock> &layoutExtUnmasked, bool inside,
            bool columns[2], StdCRemType remTypes[2], bool fragments[2],
            bool fragPositives[2], int fragSizes[2],
            const ngen::GRFRange (&C_addr0)[2],
            const ngen::GRFRange (&C_addr0Unmasked)[2], COperation op,
            std::vector<MaskAssignment> &masks, GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState state);
    void doAlternateCRemainder(COperation op, GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state);

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
    void gemmBetaScale(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
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
    void gemmUpdateSums(const GEMMProblem &problem,
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

    void gemmApplyBinaryOps(const GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state);
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
            GEMMState &state, int ha = 0);
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
            GEMMState &state, int hb = 0);
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
            GEMMState &state);
    void kLoopAllocBarrierHeader(GEMMState &state);
    ngen::GRF kLoopGetBarrierHeader(GEMMState &state);
    void kLoop(KLoop type, const GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state);
    bool kLoopSetup(const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    template <typename I>
    void kLoopReset(const I &kOffset, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void kLoopTeardown(const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    bool kLoopSingle(KLoop type, const GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state);
    bool gemmKLoop(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    bool gemmAccumulateC(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    bool gemmAccumulateCSetup(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    void gemmAccumulateCTeardown(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    bool gemmAccessC(COperation op, GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state);
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

    void gemmHilbertlikeOrder(const GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state);
    void gemmBoustrophedonOrder(const GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state);
    void gemmReorderLocalIDs(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);

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
            const GEMMStrategy &strategy, GEMMState &state);
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
            int dOffC, const Scalar<double> &alpha_real,
            const Scalar<double> &alpha_imag, bool conjugate,
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

    void prologue(const CommonStrategy &strategy);
    void epilogue(const CommonStrategy &strategy, const CommonState &state);
    void padding();
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
