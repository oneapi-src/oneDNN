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

#include <array>
#include <cstddef>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "common/impl_registration.hpp"
#include "gpu/jit/gemm/gen_gemm_kernel_generator.hpp"
#include "gpu/jit/gemm/loop_sequencer.hpp"
#include "gpu/jit/gemm/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using namespace ngen;
using namespace ngen::utils;
using dnnl::impl::utils::one_of;
using ngen::utils::log2;

using std::complex;
using std::vector;

class need_vflag : public std::runtime_error {
public:
    need_vflag() : std::runtime_error("Need virtual flag registers") {}
};

class stub_exception : public std::runtime_error {
public:
    stub_exception()
        : std::runtime_error("Functionality not yet implemented") {}
};

class hw_unsupported_exception : public std::runtime_error {
public:
    hw_unsupported_exception()
        : std::runtime_error("Unsupported in hardware") {}
};

[[noreturn]] static void hw_unsupported() {
    throw hw_unsupported_exception();
}

[[noreturn]] static void stub() {
    throw stub_exception();
}

// Helpers
template <typename U>
static inline Immediate cast(Type T, U val) {
    switch (T) {
        case Type::f16: return half(val);
        case Type::f32: return float(val);
        case Type::u8: return uint8_t(val);
        case Type::s8: return int8_t(val);
        case Type::u16: return uint16_t(val);
        case Type::s16: return int16_t(val);
        case Type::u32: return uint32_t(val);
        case Type::s32: return int32_t(val);
        case Type::u64: return uint64_t(val);
        case Type::s64: return int64_t(val);
        case Type::bf16:
        case Type::tf32:
        default: stub();
    }
}

static inline Immediate cast(Type T, Scalar<double> val) {
    return cast(T, double(val));
}

bool Type::isSubsetOf(Type T) const {
    if (*this == T) return true;

    if (isInteger() && T == bf16) return false;

    return (size() < T.size());
}

constexpr bool operator==(const RegData &rd, int i) {
    return false;
}
constexpr bool operator==(const RegData &rd, const Immediate &i) {
    return false;
}
constexpr bool operator!=(const RegData &rd, int i) {
    return true;
}
constexpr bool operator!=(const RegData &rd, const Immediate &i) {
    return true;
}

void noop() {}

static inline constexpr bool isGen9IGEMM(HW hw, Type Ta, Type Tb, Type Tc) {
    return (hw < HW::Gen12LP && Ta.size() == 1 && Tb.size() == 1
            && Tc.size() == 4);
}

template <typename T>
static inline constexpr int elementsPerGRF(HW hw) {
    return GRF::bytes(hw) / sizeof(T);
}

static inline constexpr int elementsPerGRF(HW hw, Type T) {
    return GRF::bytes(hw) / T;
}

static inline constexpr int elementsPerGRF(HW hw, DataType dt) {
    return GRF::bytes(hw) / getBytes(dt);
}

static inline bool hasNativeAtomicAdd(HW hw, Type T,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    bool floatAtomics
            = (astrategy.base.getModel() == ModelA64 || astrategy.newDP);
    if (T.isInteger())
        return true;
    else if (T == Type::f32)
        return floatAtomics && (hw >= HW::XeHP);
    else
        return false;
}

static inline int slmCapacity(HW hw) {
    switch (hw) {
        case HW::Gen9:
        case HW::Gen11: return 65536;
        case HW::Gen12LP:
        case HW::XeHP:
        case HW::XeHPG:
        case HW::XeHPC: return 131072;
        default: return 0;
    }
}

static inline int threadsPerEU(HW hw, const CommonStrategy &strategy) {
    if (hw >= HW::XeHP)
        return (strategy.GRFs > 128) ? 4 : 8;
    else
        return 7;
}

static inline int eusPerSubslice(HW hw) {
    switch (hw) {
        case HW::Gen9:
        case HW::Gen11:
        case HW::XeHPC: return 8;
        case HW::Gen12LP:
        case HW::XeHP:
        case HW::XeHPG: return 16;
        default: return 0;
    }
}

void RegisterBlock::calcBytes(
        Type T, const MatrixAddressingStrategy &astrategy) {
    if (astrategy.newDP && astrategy.prefetch)
        bytes = 0;
    else
        calcBytes(T);
}

void RegisterBlock::calcBytes(Type T) {
    bytes = align_up(colMajor ? nc : nr, crosspack) * ld * T;
    if (isLoadBlock() && msgRegs == 0)
        msgRegs = (bytes + (1 << log2GRFBytes) - 1) >> log2GRFBytes;
}

int RegisterBlock::nregs() const {
    auto grfBytes = (1 << log2GRFBytes);
    if (offsetBytes & (grfBytes - 1)) stub();
    return (bytes + grfBytes - 1) >> log2GRFBytes;
}

int RegisterBlock::offsetReg() const {
    auto grfBytes = (1 << log2GRFBytes);
    if (offsetBytes & (grfBytes - 1)) stub();
    return offsetBytes >> log2GRFBytes;
}

void RegisterBlock::simplify(Type T) {
    // If block is completely crosspacked, convert to equivalent layout without crosspack.
    if (crosspack == (colMajor ? nc : nr) && isLargeCrosspack(T, crosspack)) {
        auto od = colMajor ? nr : nc;
        if (ld == od) {
            colMajor = !colMajor;
            ld = crosspack;
            crosspack = 1;
        }
    }
}

GRFMultirange GRFMultirange::subrange(
        HW hw, Type T, const RegisterBlock &block) const {
    int ne = elementsPerGRF(hw, T);
    int ldGRFs = div_up(block.ld, ne);
    int ldUsedGRFs = div_up(block.colMajor ? block.nr : block.nc, ne);
    int td = block.colMajor ? block.nc : block.nr;

    if (ldUsedGRFs >= ldGRFs)
        return subrange(block.offsetReg(), block.nregs());
    else {
        int offReg = block.offsetReg();
        GRFMultirange result = subrange(offReg, ldUsedGRFs);
        for (int y = 1; y < td; y++) {
            offReg += ldGRFs;
            result.append(subrange(offReg, ldUsedGRFs));
        }
        return result;
    }
}

// Make a RegisterBlock smaller by contracting the leading dimension, if possible.
void RegisterBlock::compact(Type T) {
    auto newLD = std::max<int>(
            roundup_pow2(colMajor ? nr : nc), (1 << log2GRFBytes) / T);
    if (newLD < ld) {
        ld = newLD;
        calcBytes(T);
    }
}

static inline bool isTransposing(AccessType atype) {
    if (atype == AccessType::Scattered) return true;
    if (atype == AccessType::ChannelScattered) return true;
    if (atype == AccessType::Block2DTranspose) return true;
    return false;
}

Subregister SubregisterPair::getReg(int idx) const {
    auto r = regs[idx & 1];
    if (negative) r = -r;
    return r;
}

Subregister SubregisterPair::getRegAvoiding(HW hw, const RegData &rd) const {
    if (Bundle::same_bank(hw, rd, regs[0]))
        return getReg(1);
    else
        return getReg(0);
}

inline namespace {
template <typename T>
struct ACHelper {
    static T avoidConflict(HW hw, const T &x, const RegData &other) {
        return x;
    }
};
template <>
struct ACHelper<SubregisterPair> {
    static Subregister avoidConflict(
            HW hw, const SubregisterPair &x, const RegData &other) {
        return x.getRegAvoiding(hw, other);
    }
};
template <typename T>
struct ACHelper<Scalar<T>> {
    static Subregister avoidConflict(
            HW hw, const Scalar<T> &x, const RegData &other) {
        return x.getRegAvoiding(hw, other);
    }
};
} // namespace
template <typename T>
decltype(ACHelper<T>::avoidConflict(HW::Unknown, std::declval<T>(), RegData()))
avoidConflict(HW hw, const T &x, const RegData &other) {
    return ACHelper<T>::avoidConflict(hw, x, other);
}

FlagRegister VirtualFlag::toPhysical() const {
    if (n == 2)
        return FlagRegister(idx >> 1);
    else
        return FlagRegister::createFromIndex(idx);
}

VirtualFlag VirtualFlagAllocator::allocVirtual(int n) {
    if (!free) throw out_of_registers_exception();
    if (n > 2) stub();

    uint32_t bmask = free;
    if (n == 2) bmask = (bmask & (bmask >> 1)) & 0x55555555;
    int base = bsf(bmask);

    VirtualFlag vflag {base, n};
    claim(vflag);

    return vflag;
}

FlagRegister VirtualFlagAllocator::alloc(int n) {
    auto vflag = allocVirtual(n);
    if (isVirtual(vflag)) throw out_of_registers_exception();

    lock(vflag);

    return vflag.toPhysical();
}

FlagRegister VirtualFlagAllocator::assignPhysical(VirtualFlag vflag) {
    VirtualFlag pflag;

    // Is it already a physical flag register?
    if (!isVirtual(vflag)) {
        pflag = vflag;
    } else {
        // It's virtual. Starting at nextPhys, find an unlocked flag register.
        for (int i = nextPhys; i < nextPhys + nflag; i++) {
            if (i & (vflag.n - 1)) continue;
            auto idx = i & (nflag - 1);
            if (!(locked & mask(idx, vflag.n))) {
                nextPhys = (idx + vflag.n) & (nflag - 1);
                pflag = VirtualFlag {idx, vflag.n};
                break;
            }
        }
    }

    if (!pflag) throw out_of_registers_exception();

    return pflag.toPhysical();
}

static inline RegData getMaskFlag(VirtualFlag vflag, CommonState &state) {
    if (state.vflagStorage.isValid())
        return state.vflagStorage[vflag.idx].reinterpret(
                0, vflag.n == 2 ? DataType::ud : DataType::uw);
    else if (!state.raVFlag.isVirtual(vflag)) {
        auto pflag = vflag.toPhysical();
        state.usePhysicalFlag(pflag);
        return pflag;
    } else
        throw need_vflag();
}

template <HW hw>
FlagRegister gemm_kernel_generator_t<hw>::getPhysicalFlag(
        VirtualFlag vflag, CommonState &state) {
    VirtualFlag pflag;

    if (state.vflagStorage.isValid()) {
        // Check if virtual flag is currently active.
        int pidx = -1;
        for (int i = 0; i < FlagRegister::subcount(hw); i++)
            if (state.activeVFlags[i] == vflag) pidx = i;

        // If flag is not currently active, load it into a physical flag.
        if (pidx == -1) {
            auto freg = state.raVFlag.assignPhysical(vflag);
            pidx = freg.index();
            mov(1, freg, getMaskFlag(vflag, state));
            for (int i = 0; i < int(vflag.n); i++)
                state.activeVFlags[pidx + i] = vflag;
        }

        pflag = VirtualFlag {pidx, vflag.n};
    } else {
        if (state.raVFlag.isVirtual(vflag)) throw need_vflag();

        pflag = vflag;
    }

    return pflag.toPhysical();
}

template <HW hw>
void gemm_kernel_generator_t<hw>::allocVFlagStorage(
        const CommonStrategy &strategy, CommonState &state) {
    state.vflagStorage
            = state.ra.alloc(getHint(HintType::LongTerm, strategy)).uw();
}

TokenAllocator::TokenAllocator(HW hw) {
    free = (1ull << tokenCount(hw)) - 1;
}

int8_t TokenAllocator::tryAlloc() {
    if (free) {
        int8_t token = bsf(free);
        free &= ~(1 << token);
        return token;
    } else
        return -1;
}

/************************/
/* Pseudo-instructions. */
/************************/

// goto instruction with Gen12 semantics.
template <HW hw>
void gemm_kernel_generator_t<hw>::goto12(const InstructionModifier &mod,
        Label &jip, Label &uip, bool branchCtrl) {
    InstructionModifier mmod = mod;
    if (!isGen12 && !branchCtrl) {
        if (mmod.getPredCtrl() == PredCtrl::None) stub();
        mmod.setPredInv(!mmod.isPredInv());
    }
    goto_(mmod, jip, uip, branchCtrl);
}

// Compare to zero.
template <HW hw>
void gemm_kernel_generator_t<hw>::cmp0(
        const InstructionModifier &mod, RegData src0) {
    mov(mod, null.retype(src0.getType()), abs(src0));
}

// Scale then add: dst <- src0 + src1 * (numerator / denominator), rounding up.
// If exact = true, ensure src1 * num / denom is integral if src1 immediate.
template <HW hw>
void gemm_kernel_generator_t<hw>::addScaled(const InstructionModifier &mod,
        const RegData &dst, int src0, const RegData &src1, int numerator,
        int denominator, CommonState &state, bool exact) {
    if (!is_zero_or_pow2(numerator)) stub();
    if (!is_zero_or_pow2(denominator)) stub();

    if (numerator == denominator) {
        (src0 != 0) ? add(mod, dst, src1, src0)
                    : (src1 != dst) ? mov(mod, dst, src1) : noop();
    } else if (numerator > denominator) {
        (src0 == 0) ? mulConstant(mod, dst, src1, numerator / denominator)
                    : mad(mod, dst, src0, src1, numerator / denominator);
    } else if ((numerator * 2) == denominator)
        avg(mod, dst, src1, src0 * 2);
    else {
        add(mod, dst, src1, ((src0 + 1) * denominator / numerator) - 1);
        asr(mod, dst, dst, log2(denominator) - log2(numerator));
    }
}

template <HW hw>
void gemm_kernel_generator_t<hw>::addScaled(const InstructionModifier &mod,
        const RegData &dst, const RegData &src0, const RegData &src1,
        int numerator, int denominator, CommonState &state, bool exact) {
    if (!is_zero_or_pow2(numerator)) stub();
    if (!is_zero_or_pow2(denominator)) stub();

    if (numerator == denominator)
        add(mod, dst, src1, src0);
    else if (numerator > denominator)
        mad(mod, dst, src0, src1, numerator / denominator);
    else {
        auto temp = state.ra.alloc_sub(src1.getType());
        if (exact)
            asr(mod, temp, src1, log2(denominator) - log2(numerator));
        else {
            add(mod, temp, src1, (denominator / numerator) - 1);
            asr(mod, temp, temp, log2(denominator) - log2(numerator));
        }
        add(mod, dst, temp, src0);
        state.ra.safeRelease(temp);
    }
}

template <HW hw>
void gemm_kernel_generator_t<hw>::addScaled(const InstructionModifier &mod,
        const RegData &dst, const RegData &src0, int src1, int numerator,
        int denominator, CommonState &state, bool exact) {
    if (!is_zero_or_pow2(numerator)) stub();
    if (!is_zero_or_pow2(denominator)) stub();
    if (exact && ((numerator * src1) % denominator))
        throw std::runtime_error("Misaligned immediate value.");
    add(mod, dst, src0, (numerator * src1) / denominator);
}

// Synchronize on all pipes and OOO operations.
template <HW hw>
void gemm_kernel_generator_t<hw>::syncall() {
    if (hw == HW::Gen12LP)
        sync.allwr(SWSB(1));
    else if (hw >= HW::XeHP)
        sync.allwr(SWSB<AllPipes>(1));
}

// Multiply by a constant, optimizing for power-of-2 constants.
template <HW hw>
template <typename DT>
void gemm_kernel_generator_t<hw>::mulConstant(const InstructionModifier &mod,
        const RegData &dst, const RegData &src0, int32_t src1) {
    if (src1 == 0)
        mov<DT>(mod, dst, uint16_t(0));
    else if (src1 == 1) {
        if (dst != src0) mov<DT>(mod, dst, src0);
    } else if (src1 == -1)
        mov<DT>(mod, dst, -src0);
    else if (is_zero_or_pow2(src1))
        shl<DT>(mod, dst, src0, uint16_t(log2(src1)));
    else if (src1 >= 0x10000)
        mul<DT>(mod, dst, src0, uint32_t(src1));
    else if (src1 < -0x8000)
        mul<DT>(mod, dst, src0, int32_t(src1));
    else if (src1 > 0)
        mul<DT>(mod, dst, src0, uint16_t(src1));
    else
        mul<DT>(mod, dst, src0, int16_t(src1));
}

// Three-argument add.
template <HW hw>
template <typename DT, typename S0, typename S2>
void gemm_kernel_generator_t<hw>::eadd3(const InstructionModifier &mod,
        const RegData &dst, const S0 &src0, const RegData &src1,
        const S2 &src2) {
    if ((hw >= HW::XeHP) && !(dst.getOffset() & 1))
        add3<DT>(mod, dst, src0, src1, src2);
    else {
        add<DT>(mod, dst, src1, src0);
        add<DT>(mod, dst, dst, src2);
    }
}

template <HW hw>
template <typename DT>
void gemm_kernel_generator_t<hw>::emov(const ngen::InstructionModifier &mod,
        ngen::RegData dst, ngen::RegData src0, const CommonStrategy &strategy,
        CommonState &state) {
    EmulationImplementation::applyDefaultType<DT>(dst);
    EmulationImplementation::applyDefaultType<DT>(src0);

    if (dst.getType() == DataType::tf32 && src0.getType() == DataType::tf32) {
        dst.setType(DataType::f);
        src0.setType(DataType::f);
    }

    if (hw < HW::XeHP && dst.getType() == DataType::f
            && src0.getType() == DataType::bf) {
        dst.setType(DataType::ud);
        src0.setType(DataType::uw);
        shl(mod, dst, src0, 16);
    } else
        EmulationImplementation::emov(*this, mod, dst, src0, strategy.emulate);
}

template <HW hw>
template <typename S0, typename S2>
void gemm_kernel_generator_t<hw>::emad(const InstructionModifier &mod,
        const RegData &dst, const S0 &src0, const RegData &src1, const S2 &src2,
        const CommonStrategy &strategy, CommonState &state) {
    auto dstType = dst.getType();
    if ((hw >= HW::Gen10 && !(dst.getByteOffset() & 7)
                && !one_of(dstType, DataType::q, DataType::uq)
                && !one_of(src2.getType(), DataType::d, DataType::ud))
            || one_of(dstType, DataType::hf, DataType::f, DataType::df)) {
        mad(mod, dst, src0, src1, src2);
    } else {
        auto ttype = (isSigned(src1.getType()) || isSigned(src2.getType()))
                ? DataType::d
                : DataType::ud;
        auto temp = state.ra.alloc_sub(ttype);
        emul(mod, temp, src1, src2, strategy, state);
        eadd(mod, dst, temp, src0, strategy, state);
        state.ra.safeRelease(temp);
    }
}

template <HW hw>
template <typename S0>
void gemm_kernel_generator_t<hw>::emad(const InstructionModifier &mod,
        const RegData &dst, const S0 &src0, const RegData &src1, int32_t src2,
        const CommonStrategy &strategy, CommonState &state) {
    auto dstType = dst.getType();
    if (src2 == 0)
        emov(mod, dst, src0, strategy, state);
    else if (src2 == 1)
        eadd(mod, dst, src1, src0, strategy, state);
    else if (hw >= HW::Gen10 && !(dst.getByteOffset() & 7)
            && (src2 >= -0x8000 && src2 < 0x10000)
            && !one_of(dstType, DataType::q, DataType::uq)) {
        mad(mod, dst, src0, src1, src2);
    } else {
        auto ttype = isSigned(src1.getType()) ? DataType::d : DataType::ud;
        Subregister tempScalar;
        GRFRange tempGRFs;
        RegData temp;
        if (mod.getExecSize() == 1)
            temp = tempScalar = state.ra.alloc_sub(ttype);
        else {
            tempGRFs = state.ra.alloc_range(2);
            temp = tempGRFs[0].retype(ttype);
        }
        emulConstant(mod, temp, src1, src2, strategy, state);
        eadd(mod, dst, temp, src0, strategy, state);
        state.ra.safeRelease(tempScalar);
        state.ra.safeRelease(tempGRFs);
    }
}

template <HW hw>
template <typename DT>
void gemm_kernel_generator_t<hw>::emath(const InstructionModifier &mod,
        MathFunction fc, const RegData &dst, const RegData &src0,
        const GEMMStrategy &strategy, CommonState &state) {
    if (hw == HW::XeHP && strategy.systolic && mod.getExecSize() <= 8) {
        // Workaround for DPAS + SIMD8 EM hang: use SIMD16 arithmetic.
        auto mod16 = mod;
        mod16.setExecSize(16);

        auto temp = state.ra.alloc_range(2);
        auto tt = temp[0].retype(src0.getType());

        mov(mod.getExecSize(), tt, src0);
        math(mod16, fc, tt, tt);
        mov(mod.getExecSize(), dst, tt);

        state.ra.safeRelease(temp);
    } else
        math(mod, fc, dst, src0);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::ejmpi(InstructionModifier mod, Label &dst) {
    if (hw == HW::XeHPC && mod.getPredCtrl() == PredCtrl::anyv
            && !mod.isPredInv()) {
        mod.setPredCtrl(PredCtrl::Normal);
        jmpi(mod, dst);
        auto flag = mod.getFlagReg();
        flag.setBase(flag.getBase() ^ 1);
        mod.setFlagReg(flag);
        jmpi(mod, dst);
    } else
        jmpi(mod, dst);
}

/********************/
/* Utility routines */
/********************/

// Modulo by constant value.
template <HW hw>
template <typename DT>
void gemm_kernel_generator_t<hw>::mod(const Subregister &dst,
        const Subregister &src, uint16_t modulus,
        const CommonStrategy &strategy, CommonState &state) {
    if (is_zero_or_pow2(modulus))
        and_<DT>(1, dst, src, modulus - 1);
    else if (strategy.emulate.emulate64 && (hw <= HW::Gen12LP))
        math<DT>(1, MathFunction::irem, dst, src, modulus);
    else {
        alignDown<DT>(dst, src, modulus, strategy, state);
        add<DT>(1, dst, src, -dst);
    }
}

// Return both (a % b) and a - (a % b).
template <HW hw>
template <typename DT>
void gemm_kernel_generator_t<hw>::modExt(const Subregister &dstMod,
        const Subregister &dstMultiple, const Subregister &src,
        uint16_t modulus, const CommonStrategy &strategy, CommonState &state) {
    if (is_zero_or_pow2(modulus)) {
        and_<DT>(1, dstMultiple, src, ~uint32_t(modulus - 1));
        and_<DT>(1, dstMod, src, modulus - 1);
    } else if (strategy.emulate.emulate64 && (hw <= HW::Gen12LP)) {
        math<DT>(1, MathFunction::irem, dstMod, src, modulus);
        add<DT>(1, dstMultiple, src, -dstMod);
    } else {
        alignDown<DT>(dstMultiple, src, modulus, strategy, state);
        add<DT>(1, dstMod, src, -dstMultiple);
    }
}

// Divide an unsigned value by a constant, rounding down.
template <HW hw>
template <typename DT>
void gemm_kernel_generator_t<hw>::divDown(const ngen::Subregister &dst,
        const ngen::Subregister &src, uint16_t divisor,
        const CommonStrategy &strategy, CommonState &state) {
    if (is_zero_or_pow2(divisor))
        shr<DT>(1, dst, src, log2(divisor));
    else if (strategy.emulate.emulate64 && (hw <= HW::Gen12LP))
        math<DT>(1, MathFunction::iqot, dst, src, uint32_t(divisor));
    else {
        // Replace integer division with multiplication by reciprocal + shift.
        // Valid for numerators <= 2^31.
        int shift = ngen::utils::bsr(divisor);
        uint32_t recip32
                = ((uint64_t(0x100000000) << shift) + divisor - 1) / divisor;
        emul32High(1, dst, src, recip32);
        shr(1, dst, dst, shift);
    }
}

// Align an unsigned value down to a multiple of align.
template <HW hw>
template <typename DT>
void gemm_kernel_generator_t<hw>::alignDown(const Subregister &dst,
        const Subregister &src, uint16_t align, const CommonStrategy &strategy,
        CommonState &state) {
    if (is_zero_or_pow2(align))
        and_<DT>(1, dst, src, uint32_t(-align));
    else {
        divDown(dst, src, align, strategy, state);
        mul(1, dst, dst, align);
    }
}

// Align an unsigned value up to a multiple of align.
template <HW hw>
template <typename DT>
void gemm_kernel_generator_t<hw>::alignUp(const Subregister &dst,
        const Subregister &src, uint16_t align, const CommonStrategy &strategy,
        CommonState &state) {
    add<DT>(1, dst, src, uint16_t(align - 1));
    alignDown<DT>(dst, dst, align, strategy, state);
}

// Non-constant integer division.
// Requires an auxiliary constant: ceiling(2^(32 + s) / denom), where s = floor(log2(denom)).
template <HW hw>
template <typename DT>
void gemm_kernel_generator_t<hw>::divDown(const Subregister &dst,
        const Subregister &src0, const Subregister &src1,
        const Subregister &src1Recip, const FlagRegister &flag,
        const CommonStrategy &strategy, CommonState &state) {
    auto shift = state.ra.alloc_sub<uint32_t>();
    auto pop = state.ra.alloc_sub<uint16_t>();
    cbit(1, pop, src1);
    fbh(1, shift, src1);
    cmp(1 | gt | flag, pop, 1);
    add(1, shift, -shift, 31);
    emul32High(1 | flag, dst, src0, src1Recip);
    shr(1 | ~flag, dst, src0, shift);
    shr(1 | flag, dst, dst, shift);
    state.ra.safeRelease(shift);
    state.ra.safeRelease(pop);
}

// Simple do-while loop macro for the backward conditional branch at end of loop.
template <HW hw>
void gemm_kernel_generator_t<hw>::simtDoWhileLoop(
        const InstructionModifier &mod, Label &dest) {
    Label next;

    goto12(mod, next, dest, true);
    mark(next);
    join(mod.getExecSize());
}

// Barrier with SLM fence.
template <HW hw>
void gemm_kernel_generator_t<hw>::slmBarrier(
        const GRF &temp, const GRF &r0_info) {
    if (hw >= HW::Gen11) {
        slmfence(temp, r0_info);
        if (hw < HW::Gen12LP) mov<uint32_t>(8, null, temp);
    }
    barrier(temp, r0_info);
}

// Barrier with global memory fence.
template <HW hw>
void gemm_kernel_generator_t<hw>::globalMemBarrier(
        const GRF &temp, const GRF &r0_info) {
    memfence(temp, r0_info);
    if (hw < HW::Gen12LP) mov<uint32_t>(8, null, temp);
    barrier(temp, r0_info);
}

// Pause for a short period of time.
template <HW hw>
void gemm_kernel_generator_t<hw>::pause(const CommonStrategy &strategy) {
    if (hw >= HW::Gen11)
        mov(1 | Switch, tm0[4], strategy.pauseCycles);
    else
        for (int i = 0; i < 8; i++)
            mov<uint32_t>(8 | Switch, null, acc0);
}

// Create a copy of a SubregisterPair in the other bank.
template <HW hw>
void gemm_kernel_generator_t<hw>::duplicateScalar(
        SubregisterPair &val, CommonState &state) {
    auto reg0 = val.getReg(0);

    if (reg0 != val.getReg(1)) return;

    auto bundle = Bundle::locate(hw, reg0);
    auto reg1 = state.ra.alloc_sub(
            reg0.getType(), Bundle(bundle.bank_id ^ 1, Bundle::any));

    mov(1, reg1, reg0);
    val = SubregisterPair(reg0, reg1);
}

// Create a copy of a scalar subregister in the other bank.
template <HW hw>
template <typename T>
void gemm_kernel_generator_t<hw>::duplicateScalar(
        Scalar<T> &val, CommonState &state) {
    if (!val.fixed()) duplicateScalar(val.getPair(), state);
}

// Create multiple versions of the input subregister reg, shifted by amounts specified by the shifts bitmask.
// The input subregister is used for one of the versions.
template <HW hw>
MultishiftSubregister gemm_kernel_generator_t<hw>::multishift(
        const Subregister &reg, unsigned int shifts,
        const CommonStrategy &strategy, CommonState &state, Bundle hint) {
    MultishiftSubregister ms;

    while (shifts != 0) {
        int shift = bsr(shifts);
        shifts &= ~(1 << shift);

        if (shifts != 0) {
            Subregister s = state.ra.alloc_sub(reg.getType(), hint);
            ms.set(shift, s);
            eshr(1, s, reg, shift, strategy, state);
        } else {
            ms.set(shift, reg);
            if (shift > 0) eshr(1, reg, reg, shift, strategy, state);
        }
    }

    return ms;
}

// Get ID of fused thread (0/1), multiplied by a scaling factor. Assumes r1 has not been overwritten,
//  or state.lid0 is set to a subregister containing local ID 0 (divided by the subgroup size).
template <HW hw>
void gemm_kernel_generator_t<hw>::getFusedID(int scale,
        const CommonProblem &problem, const CommonStrategy &strategy,
        CommonState &state) {
    if (strategy.fused) {
        state.fusedID = state.ra.alloc_sub<uint16_t>(
                getHint(HintType::LongTerm, strategy));
        if (state.lid0.isValid()) {
            if (is_zero_or_pow2(scale) && scale > 1
                    && (state.fusedID.getOffset() & 3) == 0)
                bfi2(1, state.fusedID, scale, state.lid0, 0);
            else {
                and_(1, state.fusedID, state.lid0, 1);
                mulConstant(1, state.fusedID, state.fusedID, scale);
            }
        } else if (is_zero_or_pow2(scale)) {
            int shift = log2(scale) - log2(strategy.subgroupSize);
            Subregister lid0 = r1.uw(0);

            if (shift > 0)
                shl(1, state.fusedID, lid0, uint16_t(shift));
            else if (shift < 0)
                shr(1, state.fusedID, lid0, uint16_t(-shift));

            and_(1, state.fusedID, (shift == 0) ? lid0 : state.fusedID,
                    uint16_t(scale));
        } else {
            shr(1, state.fusedID, r1.uw(0),
                    uint16_t(log2(strategy.subgroupSize)));
            and_(1, state.fusedID, state.fusedID, uint16_t(1));
            mulConstant(1, state.fusedID, state.fusedID, uint16_t(scale));
        }
    }
}

// Move r0 information to another register if configured.
template <HW hw>
void gemm_kernel_generator_t<hw>::moveR0(
        const CommonStrategy &strategy, CommonState &state) {
    if (state.movedR0) return;
    if (state.r0_info.isInvalid()) {
        switch (strategy.moveR0) {
            case MoveR0::None:
                state.r0_info = r0.ud();
                state.movedR0 = true;
                return;
            case MoveR0::Acc: state.r0_info = acc0.ud(); break;
            case MoveR0::Addr: state.r0_info = a0.ud(); break;
            case MoveR0::GRF:
                state.r0_info
                        = state.ra.alloc(getHint(HintType::R0Info, strategy));
                break;
        }
    }

    mov<uint32_t>(8, state.r0_info, r0);

    if (!strategy.sipR0WA) state.ra.release(r0);

    state.movedR0 = true;
}

template <HW hw>
void gemm_kernel_generator_t<hw>::moveR0(
        const GEMMStrategy &strategy, GEMMState &state) {
    if (state.movedR0) return;
    if (strategy.moveR0 == MoveR0::GRF) {
        if (strategy.registerScheme == GEMMStrategy::ACB
                || strategy.registerScheme == GEMMStrategy::BCA) {
            state.r0_info = r127;
            state.ra.claim(r127);
        }
    }
    moveR0(static_cast<CommonStrategy>(strategy), state);
}

// Call a functor needing the r0 header in a GRF.
template <HW hw>
template <typename F>
void gemm_kernel_generator_t<hw>::useR0(CommonState &state, F f) {
    if (state.r0_info.isARF()) {
        auto r0_info = state.ra.alloc();
        mov<uint32_t>(8, r0_info, state.r0_info);
        f(r0_info);
        state.ra.safeRelease(r0_info);
    } else
        f(GRF {state.r0_info.getBase()});
}

// Divide out subgroup size from x local size and local ID.
template <HW hw>
void gemm_kernel_generator_t<hw>::removeSG(const CommonProblem &problem,
        const CommonStrategy &strategy, const CommonState &state) {
    uint16_t sss = log2(strategy.subgroupSize);

    auto localSize0 = interface.getLocalSize(0);
    auto localID0 = interface.getLocalID(0);

    shr(1, localSize0, localSize0, sss);
    shr(1, localID0.uw(0), localID0.uw(0), sss);
}

// Swap bit 0 of local ID x and y if needed so that threads are ordered according to specified EU fusion.
template <HW hw>
void gemm_kernel_generator_t<hw>::reorderFusedEUs(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state) {
    if (!strategy.fused) return;

    if (strategy.loopOrder[0] != strategy.fusedLoop) {
        auto temp = state.ra.alloc_sub<uint32_t>();
        and_(1, temp, state.inputs.localIDN.ud(), uint16_t(1));
        bfi2(1, state.inputs.localIDN.ud(), uint16_t(1),
                state.inputs.localIDM.ud(), state.inputs.localIDN.ud());
        bfi2(1, state.inputs.localIDM.ud(), uint16_t(1), temp,
                state.inputs.localIDM.ud());
        state.ra.safeRelease(temp);
    }
}

template <HW hw>
Subregister gemm_kernel_generator_t<hw>::copySubregister(
        const Subregister &reg, CommonState &state, Bundle hint) {
    auto copy = state.ra.alloc_sub(reg.getType(), hint);
    mov(1, copy, reg);
    return copy;
}

static inline bool canDualGRF(
        HW hw, DataType dt, const CommonStrategy &strategy) {
    return (strategy.dualGRF && (elementsPerGRF(hw, dt) < 32));
}

// Perform a binary register-wise operation.
template <typename F>
static inline void map(HW hw, DataType dt, const GRFMultirange &r1,
        const GRFMultirange &r2, const CommonStrategy &strategy, F f) {
    int ne = elementsPerGRF(hw, dt);
    int rstride = canDualGRF(hw, dt, strategy) ? 2 : 1;
    int len = r1.getLen();

    for (int rr = 0; rr < len;) {
        int nr = std::min<int>(len - rr, rstride);
        if (!r1.contiguous(rr, nr) || !r2.contiguous(rr, nr)) nr = 1;
        f(nr * ne, r1[rr].retype(dt), r2[rr].retype(dt));
        rr += nr;
    }
}

// Perform a ternary register-wise operation.
template <typename F>
static inline void map(HW hw, DataType dt, const GRFMultirange &r1,
        const GRFMultirange &r2, const GRFMultirange &r3,
        const CommonStrategy &strategy, F f) {
    int ne = elementsPerGRF(hw, dt);
    int rstride = canDualGRF(hw, dt, strategy) ? 2 : 1;
    int len = r1.getLen();

    for (int rr = 0; rr < len;) {
        int nr = std::min<int>(len - rr, rstride);
        if (!r1.contiguous(rr, nr) || !r2.contiguous(rr, nr)
                || !r3.contiguous(rr, nr))
            nr = 1;
        f(nr * ne, r1[rr].retype(dt), r2[rr].retype(dt), r3[rr].retype(dt));
        rr += nr;
    }
}

// Perform a quaternary register-wise operation.
template <typename F>
static inline void map(HW hw, DataType dt, const GRFMultirange &r1,
        const GRFMultirange &r2, const GRFMultirange &r3,
        const GRFMultirange &r4, const CommonStrategy &strategy, F f) {
    int ne = elementsPerGRF(hw, dt);
    int rstride = canDualGRF(hw, dt, strategy) ? 2 : 1;
    int len = r1.getLen();

    for (int rr = 0; rr < len;) {
        int nr = std::min<int>(len - rr, rstride);
        if (!r1.contiguous(rr, nr) || !r2.contiguous(rr, nr)
                || !r3.contiguous(rr, nr) || !r4.contiguous(rr, nr))
            nr = 1;
        f(nr * ne, r1[rr].retype(dt), r2[rr].retype(dt), r3[rr].retype(dt),
                r4[rr].retype(dt));
        rr += nr;
    }
}

// Perform a unary register-wise operation on a register block.
template <typename F>
static inline void map(HW hw, DataType dt, const GRFMultirange &regs,
        const vector<RegisterBlock> &layout, const CommonStrategy &strategy,
        F f) {
    int curReg = 0, curOff = 0, curBytes = 0;
    auto ebytes = getBytes(dt);

    auto map1 = [&]() {
        curOff &= -ebytes;
        curBytes &= -ebytes;
        while (curBytes) {
            int maxBytes;
            if (curOff & (GRF::bytes(hw) - 1))
                maxBytes = GRF::bytes(hw) - curOff;
            else
                maxBytes = (canDualGRF(hw, dt, strategy) ? 2 : 1)
                        * GRF::bytes(hw);

            auto nbytes = rounddown_pow2(std::min(maxBytes, curBytes));
            auto ne = std::min<int>(32, nbytes / ebytes);
            nbytes = ne * ebytes;

            auto reg = regs[curOff >> GRF::log2Bytes(hw)].sub(
                    (curOff & (GRF::bytes(hw) - 1)) / ebytes, dt)(1);

            f(ne, reg);

            curBytes -= nbytes;
            curOff += nbytes;
        }
    };

    for (auto &block : layout) {
        int endReg
                = (curOff + curBytes + block.bytes - 1) >> GRF::log2Bytes(hw);
        if ((block.offsetBytes == curOff + curBytes)
                && regs.contiguous(curReg, endReg - curReg + 1))
            curBytes += block.bytes;
        else {
            map1();
            curOff = block.offsetBytes;
            curReg = curOff >> GRF::log2Bytes(hw);
            curBytes = block.bytes;
        }
    }

    map1();
}

template <typename T, typename F>
static inline void map(HW hw, const GRFMultirange &r1, const GRFMultirange &r2,
        const CommonStrategy &strategy, F f) {
    map(hw, getDataType<T>(), r1, r2, strategy, f);
}

template <typename T, typename F>
static inline void map(HW hw, const GRFMultirange &r1, const GRFMultirange &r2,
        const GRFMultirange &r3, const CommonStrategy &strategy, F f) {
    map(hw, getDataType<T>(), r1, r2, r3, strategy, f);
}

template <typename T, typename F>
static inline void map(HW hw, const GRFMultirange &regs,
        const vector<RegisterBlock> &layout, const CommonStrategy &strategy,
        F f) {
    map(hw, getDataType<T>(), regs, layout, strategy, f);
}

template <typename... Targs>
static inline void map(HW hw, Type T, Targs... args) {
    map(hw, T.ngen(), args...);
}

// Move subregister to another pipe.
static inline void movePipes(Subregister &s, bool sizeCanChange = true) {
    DataType type = s.getType();

    switch (type) {
        case DataType::bf:
        case DataType::hf: type = DataType::uw; break;
        case DataType::tf32:
        case DataType::f: type = DataType::ud; break;
        case DataType::df:
            if (sizeCanChange) type = DataType::ud;
            break;
        case DataType::w:
        case DataType::uw: type = DataType::hf; break;
        case DataType::d:
        case DataType::ud: type = DataType::f; break;
        case DataType::q:
        case DataType::uq:
            if (sizeCanChange) type = DataType::f;
            break;
        default: break;
    }

    s = s.reinterpret(0, type);
}

// Move register region to integer pipe.
static inline void moveToIntPipe(int esize, RegData &s) {
    switch (s.getType()) {
        case DataType::bf:
        case DataType::hf: s.setType(DataType::uw); break;
        case DataType::q:
        case DataType::uq:
        case DataType::tf32:
        case DataType::f: s.setType(DataType::ud); break;
        case DataType::df:
            s.setType(DataType::uq);
            EmulationImplementation::makeDWPair(s, esize);
            break;
        default: break;
    }
}

// Set a matrix to zero.
template <HW hw>
void gemm_kernel_generator_t<hw>::zeroMatrix(
        const GRFMultirange &r, const CommonStrategy &strategy) {
    map<uint32_t>(hw, r, r, strategy,
            [&](int esize, GRF reg, GRF _) { mov(esize, reg, uint16_t(0)); });
}

// Release fused remainder-related state variables.
template <HW hw>
void gemm_kernel_generator_t<hw>::releaseFusedRemainders(GEMMState &state) {
    state.ra.safeRelease(state.remFusedStorage);
    state.remaindersFused[LoopM] = Subregister {};
    state.remaindersFused[LoopN] = Subregister {};
}

template <HW hw>
void gemm_kernel_generator_t<hw>::saveMNLocalIDs(
        const GEMMStrategy &strategy, GEMMState &state) {
    state.lidStorage = state.ra.alloc_sub<uint32_t>(
            getHint(HintType::LongTerm, strategy));
    state.lidM = state.lidStorage.uw(0);
    state.lidN = state.lidStorage.uw(1);
    mov(1, state.lidM, state.inputs.localIDM);
    mov(1, state.lidN, state.inputs.localIDN);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::saveKLocalIDSize(
        const GEMMStrategy &strategy, GEMMState &state) {
    state.lidszKStorage = state.ra.alloc_sub<uint32_t>(
            getHint(HintType::LongTerm, strategy));
    state.lidK = state.lidszKStorage.uw(0);
    state.lszK = state.lidszKStorage.uw(1);
    mov(1, state.lidK, state.inputs.localIDK);
    mov(1, state.lszK, state.inputs.localSizeK);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::releaseSavedMNLocalIDs(GEMMState &state) {
    state.ra.safeRelease(state.lidStorage);
    state.lidStorage = invalid;
    state.lidM = invalid;
    state.lidN = invalid;
}

// Clear read suppresion data on ALU pipes.
template <HW hw>
void gemm_kernel_generator_t<hw>::doReadSuppressionWA(
        const CommonStrategy &strategy, CommonState &state) {
    GRF temp;
    bool freeTemp = false;

    if (!strategy.readSuppressionWA) return;

    if (state.r0_info.isValid() && !state.r0_info.isARF())
        temp = GRF(state.r0_info.getBase());
    else {
        temp = state.ra.try_alloc();
        if (temp.isValid())
            freeTemp = true;
        else
            temp = r0;
    }

    csel<int16_t>(8, temp, temp, temp, temp);
    csel<float>(8, temp, temp, temp, temp);

    if (freeTemp) state.ra.safeRelease(temp);
}

// Get minimum row/column granularity for a matrix in memory.
static void getGranularities(
        const MatrixAddressing &atype, int &rgran, int &cgran) {
    auto &xgran = isColMajor(atype.layout) ? cgran : rgran;
    auto &ygran = isColMajor(atype.layout) ? rgran : cgran;
    rgran = std::max<int>(atype.tileR, 1);
    cgran = std::max<int>(atype.tileC, 1);
    xgran = std::max<int>(xgran, atype.crosspack);
    if (isPacked(atype.layout)) ygran = std::max<int>(ygran, atype.packSize);
}

// Common register allocator hints.
template <HW hw>
Bundle gemm_kernel_generator_t<hw>::getHint(HintType type) {
    switch (type) {
        case HintType::Bank0: return Bundle(0, Bundle::any);
        case HintType::Bank1: return Bundle(1, Bundle::any);
        default: break;
    }

    switch (hw) {
        case HW::Gen9:
        case HW::Gen10:
        case HW::Gen11:
            switch (type) {
                case HintType::TempComp0: return Bundle(0, 1);
                case HintType::TempComp1: return Bundle(1, 1);
                case HintType::LongTerm: return Bundle(Bundle::any, 0);
                case HintType::LongTerm0: return Bundle(0, 0);
                case HintType::LongTerm1: return Bundle(1, 0);
                default: break;
            }
            break;
        case HW::Gen12LP:
        case HW::XeHP:
        case HW::XeHPG:
        case HW::XeHPC:
            switch (type) {
                case HintType::LongTerm0: return Bundle(0, Bundle::any);
                case HintType::LongTerm1: return Bundle(1, Bundle::any);
                default: break;
            }
        default: break;
    }

    return Bundle();
}

template <HW hw>
Bundle gemm_kernel_generator_t<hw>::getHint(
        HintType type, const CommonStrategy &strategy) {
    return getHint(type);
}

// GEMM register allocation hints.
template <HW hw>
Bundle gemm_kernel_generator_t<hw>::getHint(
        HintType type, const GEMMStrategy &strategy) {
    switch (hw) {
        case HW::Gen9:
        case HW::Gen10:
        case HW::Gen11:
            switch (strategy.registerScheme) {
                case GEMMStrategy::CSeparate:
                    switch (type) {
                        case HintType::A0Broadcast:
                        case HintType::A0: return Bundle(1, 0);
                        case HintType::A1Broadcast:
                        case HintType::A1: return Bundle(0, 0);
                        case HintType::B0Broadcast:
                        case HintType::B0: return Bundle(0, 0);
                        case HintType::B1Broadcast:
                        case HintType::B1: return Bundle(1, 0);
                        case HintType::C: return Bundle(0, 1);
                        case HintType::CLoad: return Bundle(1, 0);
                        default: break;
                    }
                    break;
                case GEMMStrategy::ACB:
                    switch (type) {
                        case HintType::A0Broadcast:
                        case HintType::A0: return Bundle(1, 0);
                        case HintType::A1Broadcast:
                        case HintType::A1: return Bundle(0, 0);
                        case HintType::B0Broadcast:
                        case HintType::B0: return Bundle(0, 1);
                        case HintType::B1Broadcast:
                        case HintType::B1: return Bundle(1, 1);
                        case HintType::C: return Bundle(0, 0);
                        case HintType::CLoad: return Bundle();
                        default: break;
                    }
                    break;
                case GEMMStrategy::BCA:
                    switch (type) {
                        case HintType::A0Broadcast:
                        case HintType::A0: return Bundle(0, 1);
                        case HintType::A1Broadcast:
                        case HintType::A1: return Bundle(1, 1);
                        case HintType::B0Broadcast:
                        case HintType::B0: return Bundle(1, 0);
                        case HintType::B1Broadcast:
                        case HintType::B1: return Bundle(0, 0);
                        case HintType::C: return Bundle(0, 0);
                        case HintType::CLoad: return Bundle();
                        default: break;
                    }
                    break;
                default: break;
            }
            break;
        case HW::Gen12LP:
        case HW::XeHP:
        case HW::XeHPG:
        case HW::XeHPC:
            switch (strategy.registerScheme) {
                case GEMMStrategy::CSeparate:
                    switch (type) {
                        case HintType::A0Broadcast:
                        case HintType::A0: return Bundle(1, Bundle::any);
                        case HintType::A1Broadcast:
                        case HintType::A1: return Bundle(0, Bundle::any);
                        case HintType::B0Broadcast:
                        case HintType::B0: return Bundle(0, Bundle::any);
                        case HintType::B1Broadcast:
                        case HintType::B1: return Bundle(1, Bundle::any);
                        case HintType::C: return Bundle(0, 0);
                        case HintType::CLoad: return Bundle(1, Bundle::any);
                        default: break;
                    }
                    break;
                case GEMMStrategy::ACB:
                case GEMMStrategy::BCA:
                    if (strategy.systolic) switch (type) {
                            case HintType::A0:
                            case HintType::B0: return Bundle(0, Bundle::any);
                            case HintType::A1:
                            case HintType::B1: return Bundle(1, Bundle::any);
                            case HintType::A0Broadcast:
                            case HintType::B0Broadcast:
                                return Bundle(1, Bundle::any);
                            case HintType::A1Broadcast:
                            case HintType::B1Broadcast:
                                return Bundle(0, Bundle::any);
                            case HintType::C: return Bundle(0, Bundle::any);
                            default: break;
                        }
                    /* else fall through */
                case GEMMStrategy::VNC:
                    switch (type) {
                        case HintType::A0:
                        case HintType::B0: return Bundle(1, Bundle::any);
                        case HintType::A1:
                        case HintType::B1: return Bundle(0, Bundle::any);
                        case HintType::A0Broadcast:
                        case HintType::B0Broadcast:
                            return Bundle(0, Bundle::any);
                        case HintType::A1Broadcast:
                        case HintType::B1Broadcast:
                            return Bundle(1, Bundle::any);
                        case HintType::C: return Bundle(0, Bundle::any);
                        default: break;
                    }
                    break;
                case GEMMStrategy::ABInterleave:
                    switch (type) {
                        case HintType::A0:
                        case HintType::A1:
                        case HintType::A0Broadcast:
                        case HintType::A1Broadcast: return Bundle(1, 0);
                        case HintType::B0:
                        case HintType::B1:
                        case HintType::B0Broadcast:
                        case HintType::B1Broadcast: return Bundle(1, 4);
                        case HintType::C: return Bundle(0, Bundle::any);
                        default: break;
                    }
                    break;
                case GEMMStrategy::NSeparate:
                    switch (type) {
                        case HintType::A0:
                        case HintType::B0: return Bundle(1, Bundle::any);
                        case HintType::A1:
                        case HintType::B1: return Bundle(0, Bundle::any);
                        case HintType::A0Broadcast:
                        case HintType::B0Broadcast:
                        case HintType::A1Broadcast:
                        case HintType::B1Broadcast: return Bundle();
                        case HintType::C: return Bundle(0, Bundle::any);
                        case HintType::C1: return Bundle(1, Bundle::any);
                        default: break;
                    }
                    break;
                case GEMMStrategy::VAvoid:
                    switch (type) {
                        case HintType::A0:
                        case HintType::B0: return Bundle(0, Bundle::any);
                        case HintType::A1:
                        case HintType::B1: return Bundle(1, Bundle::any);
                        case HintType::A0Broadcast:
                        case HintType::B0Broadcast:
                        case HintType::A1Broadcast:
                        case HintType::B1Broadcast:
                            return Bundle(1, Bundle::any);
                        case HintType::C: return Bundle(0, Bundle::any);
                        case HintType::C1: return Bundle(1, Bundle::any);
                        default: break;
                    }
                    break;
            }
            break;
        default: break;
    }

    return getHint(type);
}

// Copy kernel register allocation hints.
template <HW hw>
Bundle gemm_kernel_generator_t<hw>::getHint(
        HintType type, const CopyStrategy &strategy) {
    switch (hw) {
        case HW::Gen9:
        case HW::Gen10:
        case HW::Gen11:
        case HW::Gen12LP:
        case HW::XeHP:
        case HW::XeHPG:
        case HW::XeHPC:
            switch (type) {
                case HintType::S: return Bundle();
                case HintType::D: return Bundle();
                case HintType::SAddr: return Bundle();
                case HintType::DAddr: return Bundle();
                default: break;
            }
            break;
        default: break;
    }

    return getHint(type);
}

static inline void safeReleaseRanges(
        vector<GRFRange> &ranges, CommonState &state) {
    for (auto &a : ranges)
        state.ra.safeRelease(a);
    ranges.clear();
}

static inline void releaseRanges(
        const vector<GRFRange> &ranges, CommonState &state) {
    for (auto &a : ranges)
        state.ra.release(a);
}

static inline void reclaimRanges(
        const vector<GRFRange> &ranges, CommonState &state) {
    for (auto &a : ranges)
        state.ra.claim(a);
}

static inline void safeRelease(SubregisterPair &pair, CommonState &state) {
    state.ra.release(pair.getReg(0));
    state.ra.release(pair.getReg(1));
    pair.invalidate();
}

static inline void safeReleaseRanges(
        GRFMultirange &ranges, CommonState &state) {
    safeReleaseRanges(ranges.ranges, state);
    ranges.ranges.clear();
}

static inline void safeReleaseRanges(
        vector<GRFMultirange> &ranges, CommonState &state) {
    for (auto &a : ranges)
        safeReleaseRanges(a, state);
    ranges.clear();
}

static inline void releaseRanges(
        const GRFMultirange &ranges, CommonState &state) {
    releaseRanges(ranges.ranges, state);
}

static inline void releaseRanges(
        const vector<GRFMultirange> &ranges, CommonState &state) {
    for (auto &a : ranges)
        releaseRanges(a, state);
}

static inline void reclaimRanges(
        const GRFMultirange &ranges, CommonState &state) {
    reclaimRanges(ranges.ranges, state);
}

// Reclaim a list of GRF multiranges.
static inline void reclaimRanges(
        const vector<GRFMultirange> &ranges, CommonState &state) {
    for (auto &a : ranges)
        reclaimRanges(a, state);
}

/***********************\
|* Load/store support. *|
\***********************/

static bool needsPseudoblock(HW hw, Type T, int r, int c,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy, bool writable, bool masked) {
    auto consecutive
            = (isColMajor(atype.layout) ^ isLargeCrosspack(T, atype.crosspack))
            ? r
            : c;
    bool dwAligned = (atype.alignment & 0x3) == 0;
    bool owAligned = (atype.alignment & 0xF) == 0;
    bool pseudo = !dwAligned || ((consecutive * T) & 0x3)
            || (writable && !owAligned)
            || (writable && masked && (T.size() & 3))
            || (masked && !owAligned
                    && (hw >= HW::XeHP
                            || astrategy.base.getModel() != ModelA64))
            || (hw >= HW::XeHPC && masked)
            || (hw >= HW::XeHPC && !astrategy.padded && !astrategy.newDP
                    && ((r * c * T) & 0xF))
            || astrategy.atomic
            || (isColMajor(atype.layout) ? c : r) % atype.crosspack
            || ((astrategy.base.getModel() == ModelSLM)
                    && (hw < HW::Gen11 || !owAligned));

    return pseudo;
}

static bool pseudoblockUseSurface(const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy, const RegisterBlock &block) {
    return (astrategy.base.getModel() == ModelSLM) && (block.ebytes == 4);
}

static bool downgradeBlock2D(AccessType type, const MatrixAddressing &atype) {
    if (!isBlock2D(type)) return false;
    return (atype.alignment & 0x3F);
}

// Get effective access type to use when setting up addresses.
static AccessType effectiveAccessType(const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy, const RegisterBlock &block) {
    auto type = astrategy.accessType;
    if (!block.isLoadBlock()) return type;
    if (downgradeBlock2D(type, atype))
        type = (type == AccessType::Block2DTranspose) ? AccessType::Scattered
                                                      : AccessType::Block;
    if (type == AccessType::Block && block.ebytes < 16 && block.extra)
        type = AccessType::PseudoBlock;
    else if (type == AccessType::Scattered
            && astrategy.base.getModel() == ModelSLM && block.ebytes == 4
            && !astrategy.newDP)
        type = AccessType::ChannelScattered;
    else if (type == AccessType::ChannelScattered && block.ebytes != 4)
        type = AccessType::Scattered;
    return type;
}

// Get effective access type to use when performing loads/stores.
static AccessType implAccessType(const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy, const RegisterBlock &block) {
    auto type = effectiveAccessType(atype, astrategy, block);
    if (type == AccessType::PseudoBlock)
        type = pseudoblockUseSurface(atype, astrategy, block)
                ? AccessType::ChannelScattered
                : AccessType::Scattered;
    return type;
}

// Count the number of address/header GRFs required by a RegisterBlock.
static inline int addrGRFCount(const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy, const RegisterBlock &block) {
    // Non-load blocks don't get address registers.
    if (!block.isLoadBlock()) return 0;

    switch (effectiveAccessType(atype, astrategy, block)) {
        case AccessType::Scattered:
        case AccessType::ChannelScattered:
        case AccessType::PseudoBlock: {
            auto bytesPerAddr = (astrategy.base.getModel() == ModelA64) ? 8 : 4;
            auto baseSIMD = std::max<int>(block.simdSize, 8);
            auto log2Bytes = block.log2GRFBytes;
            return (bytesPerAddr * baseSIMD + (1 << log2Bytes) - 1)
                    >> log2Bytes;
        }
        case AccessType::Block: return 1;
        case AccessType::Block2D:
        case AccessType::Block2DTranspose:
        case AccessType::Block2DVNNI: return 1;
    }
    throw std::runtime_error("Invalid addressing.");
}

// Attempt to allocate address registers for a layout. Returns true if successful.
static bool tryAllocAddrRegs(vector<GRFRange> &addrRegs,
        const vector<RegisterBlock> &layout, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy, CommonState &state,
        Bundle hint = Bundle()) {
    auto nblocks = int(layout.size());
    bool ok = true;

    addrRegs.resize(nblocks);

    for (int l = 0; l < nblocks && ok; l++) {
        addrRegs[l] = state.ra.try_alloc_range(
                addrGRFCount(atype, astrategy, layout[l]), hint);
        ok &= addrRegs[l].isValid();
    }

    if (!ok) {
        for (auto &regs : addrRegs)
            state.ra.safeRelease(regs);
        addrRegs.clear();
    }

    return ok;
}

// Allocate address registers for a layout.
static void allocAddrRegs(vector<GRFRange> &addrRegs,
        const vector<RegisterBlock> &layout, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy, CommonState &state,
        Bundle hint = Bundle()) {
    if (!tryAllocAddrRegs(addrRegs, layout, atype, astrategy, state, hint))
        throw out_of_registers_exception();
}

// Check if a layout is completely column-major.
static inline bool isLayoutColMajor(const vector<RegisterBlock> &layout) {
    if (layout.size() == 0) throw std::runtime_error("Empty layout.");
    return layout[0]
            .colMajor; // All layouts we create are homogeneous currently.
}

// Get the matrix size represented by a layout.
static inline void getLayoutDims(
        const vector<RegisterBlock> &layout, int &m, int &n) {
    // For now all layouts are sorted so last block is in lower-right corner.
    if (layout.size() == 0) throw std::runtime_error("Empty layout.");
    auto &last = layout[layout.size() - 1];
    m = last.offsetR + last.nr;
    n = last.offsetC + last.nc;
}

// Check if every block in a layout has the given crosspack, with no padding.
static inline bool hasFullCrosspack(
        const vector<RegisterBlock> &layout, int crosspack) {
    if (layout.size() == 0) return true;
    if (layout[0].crosspack
            != crosspack) // Only need to check first block of layout currently.
        return false;
    for (const auto &block : layout)
        if ((block.colMajor ? block.nc : block.nr) % crosspack) return false;
    return true;
}

// Check if the layout is tiled with the given tiling.
static inline bool hasTiling(
        const vector<RegisterBlock> &layout, int tileR, int tileC) {
    for (auto &block : layout) {
        if (tileR > 0)
            if (block.offsetR / tileR != (block.offsetR + block.nr - 1) / tileR)
                return false;
        if (tileC > 0)
            if (block.offsetC / tileC != (block.offsetC + block.nc - 1) / tileC)
                return false;
    }
    return true;
}

// Check if a layout has row fragmenting.
static bool hasRowFragmenting(const vector<RegisterBlock> &layout) {
    for (auto &block : layout)
        if (block.rowFragment) return true;
    return false;
}

// Check if a layout has column fragmenting.
static bool hasColumnFragmenting(const vector<RegisterBlock> &layout) {
    for (auto &block : layout)
        if (block.colFragment) return true;
    return false;
}

// Check if a layout has remainders enabled.
static bool hasRemainders(const vector<RegisterBlock> &layout,
        bool remainderR = true, bool remainderC = true) {
    for (auto &block : layout)
        if ((remainderR && block.remainderR)
                || (remainderC && block.remainderC))
            return true;
    return false;
}

// Check if a layout has any kind of fragmenting.
static bool hasFragmenting(const vector<RegisterBlock> &layout) {
    for (auto &block : layout)
        if (block.rowFragment || block.colFragment) return true;
    return false;
}

// Check if a layout has any masking.
static bool hasMasking(const vector<RegisterBlock> &layout) {
    for (auto &block : layout)
        if (block.rowMask || block.colMask || block.flag) return true;
    return false;
}

// Check if a layout has any flag registers assigned.
static bool hasFlags(const vector<RegisterBlock> &layout) {
    for (auto &block : layout)
        if (block.flag) return true;
    return false;
}

// Find the maximum block size in a layout, in registers.
static inline int getMaxLoadBlock(const vector<RegisterBlock> &layout) {
    int result = 0;
    for (auto &l : layout)
        result = std::max<int>(result, l.msgRegs);
    return result;
}

// Count the number of registers needed by a register layout.
static inline int getRegCount(const vector<RegisterBlock> &layout) {
    if (layout.empty()) return 0;

    int lastByte = 0;
    for (auto &l : layout)
        lastByte = std::max(lastByte, l.offsetBytes + l.bytes);

    int log2Bytes = layout[0].log2GRFBytes;
    return (lastByte + (1 << log2Bytes) - 1) >> log2Bytes;
}

static int getAddr0Offset(const RegisterBlock &block,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    if (astrategy.newDP) return 0;
    if (astrategy.base.getModel() == ModelA64) return 0;
    if (effectiveAccessType(atype, astrategy, block) == AccessType::Block)
        return 2;
    return 0;
}

// Get a subregister containing the (shifted) address of the (0,0) entry of a layout.
static Subregister getOriginAddr(const vector<RegisterBlock> &layout,
        const vector<GRFRange> &addrRegs, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy, int *shiftOut = nullptr) {
    bool a64 = (astrategy.base.getModel() == ModelA64);

    for (size_t b = 0; b < layout.size(); b++) {
        const auto &block = layout[b];
        if ((block.offsetR != 0) || (block.offsetC != 0)) continue;

        int off = getAddr0Offset(block, atype, astrategy);

        if (shiftOut) *shiftOut = block.addrShift;
        return addrRegs[b][0].sub(off, a64 ? DataType::uq : DataType::ud);
    }

    if (shiftOut) *shiftOut = 0;
    return Subregister();
}

static inline int maxScatteredSIMD(
        HW hw, const MatrixAddressingStrategy &astrategy) {
    if (astrategy.newDP) return GRF::bytes(hw) >> 1;
    return 16;
}

static inline int minScatteredSIMD(
        HW hw, const MatrixAddressingStrategy &astrategy) {
    if (hw == HW::XeHPC) return 16;
    return maxScatteredSIMD(hw, astrategy) >> 1;
}

// Get width and height parameters for underlying 2D block load message.
static void getBlock2DWH(int &w, int &h, const MatrixAddressing &atype,
        const RegisterBlock &block, int *outMultiX = nullptr) {
    int multiX = 1;
    w = isColMajor(atype.layout) ? block.nr : block.nc;
    h = isColMajor(atype.layout) ? block.nc : block.nr;
    w = (w * block.extra) / block.ebytes;
    if (isPacked(atype.layout)) {
        int maxW = 64 / block.ebytes;
        multiX = div_up(w, maxW);
        w /= multiX;
        h *= multiX;
    }
    if (outMultiX) *outMultiX = multiX;
}

static bool isRegisterColMajor(Type T, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    return isColMajor(atype.layout) ^ isTransposing(astrategy.accessType)
            ^ isLargeCrosspack(T, atype.crosspack);
}

// Set up a RegisterBlock structure.
template <HW hw>
bool gemm_kernel_generator_t<hw>::getBlockInfo(Type T,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy, int r, int c,
        bool remainderR, bool remainderC, bool writable, bool avoidFragment,
        int maxRBlock, int maxCBlock, int &rblock, int &cblock,
        RegisterBlock &block) {
    bool prefetch = astrategy.prefetch;
    int R = rounddown_pow2(r);
    int C = rounddown_pow2(c);

    if (maxRBlock == 0) maxRBlock = r;
    if (maxCBlock == 0) maxCBlock = c;

    // Set default parameters.
    block.colMajor = isColMajor(atype.layout);
    block.splitComplex = false;
    block.crosspack = 1;
    block.rowMask = MaskInfo::None();
    block.colMask = MaskInfo::None();
    block.rowFragment = 0;
    block.colFragment = 0;
    block.remainderR = remainderR;
    block.remainderC = remainderC;
    block.noRowsOK = false;
    block.noColsOK = false;
    block.descRemR = false;
    block.descRemC = false;
    block.descAssigned = false;
    block.addrShift = 0;
    block.writable = writable;
    block.clearFlag();
    block.log2GRFBytes = GRF::log2Bytes(hw);
    block.msgRegs = 0;
    block.bytes = 0;
    block.hasNoLoad = false;

    auto &vrmask = block.rowMask.variable;
    auto &vcmask = block.colMask.variable;

    vrmask.rsize = 0;
    vcmask.rsize = 0;

    auto accessType = astrategy.accessType;

    if (downgradeBlock2D(accessType, atype)) accessType = AccessType::Block;

    switch (accessType) {
        case AccessType::ChannelScattered:
        case AccessType::Scattered: {
            bool channelScattered
                    = (accessType == AccessType::ChannelScattered);

            // Detect large crosspack case.
            bool largeCP = isLargeCrosspack(T, atype.crosspack);
            int effCP = largeCP ? 1 : atype.crosspack;

            // Scattered read/write messages effectively transpose DW/QW matrices.
            block.colMajor = !block.colMajor ^ largeCP;

            // Let X be the contiguous dimension, Y the scattered dimension (in memory).
            int *xblock, *yblock;
            int maxXBlock, maxYBlock;
            int X, Y;
            bool remainderX, remainderY;
            int tileX, tileY;
            auto &vxmask = block.colMajor ? vcmask : vrmask;
            auto &vymask = block.colMajor ? vrmask : vcmask;
            auto &fragment
                    = block.colMajor ? block.colFragment : block.rowFragment;
            auto smode = astrategy.smode;

            if (block.colMajor) {
                Y = R;
                X = C;
                yblock = &rblock;
                xblock = &cblock;
                maxYBlock = maxRBlock;
                maxXBlock = maxCBlock;
                remainderY = remainderR;
                remainderX = remainderC;
                tileY = atype.tileR;
                tileX = atype.tileC;
            } else {
                X = R;
                Y = C;
                xblock = &rblock;
                yblock = &cblock;
                maxXBlock = maxRBlock;
                maxYBlock = maxCBlock;
                remainderX = remainderR;
                remainderY = remainderC;
                tileX = atype.tileR;
                tileY = atype.tileC;
            }

            // Allowed accesses:
            //   A64             Essentially max 256 bytes.
            //                    8 slots x (1,2,4,8) dwords [Gen12/surface: 1,2,4]
            //                    8 slots x (1,2,4) qwords
            //                   16 slots x (1,2,4) dwords
            //                   16 slots x (1,2) qwords
            //   Others           8 slots x 1 dword
            //                   16 slots x 1 dword
            // Slot counts doubled for 64-byte GRFs.

            // Native (col major in memory) matrix block sizes, as a result:
            //   SIMD8:          1x8  2x4 4x2 8x1      (count 1)  2x8  4x8  8x8  [others]
            //   SIMD16:         1x16 2x8 4x4 8x2 16x1 (count 1)  2x16 4x16
            // Other layouts are possible too but require dummy (non-load) blocks.
            // Only kx8 and kx16 are supported for now for {4,8}-byte types.
            // For 16-byte types, only 1x4 and 1x8 are supported.

            auto maxSIMD = maxScatteredSIMD(hw, astrategy);
            auto minSIMD = minScatteredSIMD(hw, astrategy);

            auto Xc = (avoidFragment && remainderX) ? 1 : X;
            bool byte = (atype.alignment < 4) || (Xc * T < 4);
            bool a64 = (astrategy.base.getModel() == ModelA64);

            channelScattered |= byte;

            bool qword = (T.size() >= 8 && !channelScattered && !prefetch
                    && (a64 || astrategy.newDP));
            if (astrategy.atomic
                    && hasNativeAtomicAdd(hw, T.real(), atype, astrategy))
                qword &= (T.real().size() >= 8);
            int width = qword ? 8 : 4;
            block.ebytes = byte ? 1 : width;
            block.crosspack = std::max<int>(1, width / T);
            int consecutive = std::max<int>(1, T.size() / width);

            if (prefetch) consecutive = 1;

            if (block.ebytes == 4 && astrategy.base.getModel() == ModelSLM
                    && !astrategy.newDP)
                channelScattered = true;

            bool simd1 = !a64 && !channelScattered;
            simd1 &= !astrategy.newDP;

            // Handle source crosspack.
            int uncrosspack = 1;
            if (effCP > 1) {
                if (effCP == block.crosspack) {
                    block.crosspack = 1;
                    uncrosspack = effCP;
                } else
                    stub();
            }

            // Try to fit a native matrix block size to X and Y.
            auto slots = std::min(Y, maxYBlock) * consecutive / uncrosspack;
            if (prefetch) {
                // Prefetch only: maximize Y usage.
                block.simdSize = maxSIMD;
            } else if (smode == ScatterSIMD::Narrow
                    || (smode == ScatterSIMD::Default
                            && block.ebytes * minSIMD > GRF::bytes(hw))) {
                // Maximize X usage because we always have at least 2 consecutive GRFs.
                block.simdSize
                        = (slots >= maxSIMD && X <= 2) ? maxSIMD : minSIMD;
            } else {
                // Otherwise, try to maximize Y usage (larger SIMD, worse memory access).
                block.simdSize = maxSIMD;
            }
            block.simdSize
                    = std::min<int>(block.simdSize, rounddown_pow2(slots));

            bool no8x8DW = isGen12;
            bool no16x4QW = false;

            no8x8DW &= !astrategy.newDP;
            if (hw == HW::XeHPG && astrategy.newDP)
                no8x8DW = no16x4QW
                        = true; // Not supported on 512 EU A0. OK on later steppings.
            no16x4QW |= (!astrategy.newDP && GRF::bytes(hw) == 64);

            int hwMaxXBlock;

            if (prefetch)
                hwMaxXBlock = 64 / T;
            else if (consecutive > 1)
                hwMaxXBlock = 1;
            else if (byte)
                hwMaxXBlock = remainderX ? 1 : block.crosspack;
            else if (simd1)
                hwMaxXBlock = block.crosspack;
            else if (a64 && astrategy.atomic)
                hwMaxXBlock = block.crosspack;
            else if (channelScattered || (block.ebytes == 4 && no8x8DW)
                    || (block.ebytes == 8 && no16x4QW)
                    || (block.simdSize == maxSIMD))
                hwMaxXBlock = 16 / T;
            else
                hwMaxXBlock = 32 / T;

            maxXBlock = std::min(maxXBlock, hwMaxXBlock);

            if (tileX > 0) maxXBlock = std::min(maxXBlock, tileX);

            *xblock = std::min<int>(X, maxXBlock);
            block.count = *xblock;

            *yblock = block.simdSize * uncrosspack / consecutive;
            if (tileY > 0 && tileY < *yblock) stub();

            if (prefetch)
                block.count = 1;
            else if (byte)
                block.count *= T.size();
            else
                block.count = std::max<int>(1, block.count / block.crosspack);

            // LD is determined by actual # of SIMD slots in HW. But for X = 1 we may
            //  shrink the LD to avoid allocating unnecessary registers.
            auto ldSIMD = block.simdSize;
            if (*xblock > 1 || (minSIMD * block.ebytes <= GRF::bytes(hw)))
                ldSIMD = std::max<int>(ldSIMD, minSIMD);
            block.ld = ldSIMD * uncrosspack / consecutive;

            // Handle remainder. Masking handles Y remainders.
            if (remainderY) {
                vymask.isFixed = false;
                vymask.bitRep = consecutive;
                vymask.maskRep = 1;
                vymask.rsize = *yblock;
                vymask.rdivide = 1;
            }

            // X remainders require fragmenting. Channel scattered float doesn't need complete fragmenting.
            //   (ditto for regular scattered float with new dataport messages.)
            //  Otherwise, fragment 2 is possible for DWord+ types but not implemented.
            if (remainderX && !prefetch) {
                if (avoidFragment && !remainderY) {
                    vxmask.isFixed = false;
                    vxmask.bitRep = 16;
                    vxmask.maskRep = 1;
                    vxmask.rsize = 1;
                    vxmask.rdivide = 1;
                } else if ((channelScattered || astrategy.newDP)
                        && block.crosspack == 1 && block.ebytes == T.size()) {
                    fragment = std::min(*xblock, 4);
                    if (block.colMajor) // Clang can't handle the ternary operator equivalent of this.
                        block.descRemC = true;
                    else
                        block.descRemR = true;
                } else
                    fragment = 1;
            }

            block.extra = consecutive;

            // BTS scattered accesses are addressed by elements.
            if (!astrategy.newDP && !channelScattered
                    && !astrategy.base.isStateless())
                block.addrShift = log2(block.ebytes);

            break;
        }
        case AccessType::Block:
        case AccessType::PseudoBlock: {
            // Three types of block messages:
            //    block_oword: 16 byte align, BLK masking (= dw except ow channel on R Gen9 only -- silently ignore, can't fault)
            //  aligned_oword:  4 byte align, no masking, read only
            //    block_hword: [Gen9-12LP] A64; 4 byte align R, BLKCM masking (= dw but can do ow channel on Gen9 only)
            //                             A64; 16 byte align W
            //                 [XeHP]   A64/BTS; 32 byte align R/W
            // New dataport messages support {DW, QW}x{1...64} with DW/QW alignment, no masking.
            //
            // Prefer block_hword in all cases. When block_hword can't be used:
            //   Use oword if alignment can be assured (i.e. packed row/column layout, or oword-sized scalar)
            //   Otherwise, use aligned oword. load/storeMatrixBlock will emit an error if masking/stores attempted.
            //
            // Pseudoblock messages have similar layouts, but are limited to
            //  {8,16}x{dw,qw} sizes, so lengths 8,16 allowed for float, 4,8,16 for double.

            bool colMajor = block.colMajor;
            bool effCM = colMajor ^ isLargeCrosspack(T, atype.crosspack);
            auto consecutive = (effCM ? R : C);
            bool masking = (effCM ? remainderR : remainderC);
            bool bytePartialCP
                    = (T.size() & 3) && ((colMajor ? C : R) % atype.crosspack);
            bool byte = (atype.alignment & 3) || (consecutive * T & 3)
                    || bytePartialCP || ((T.size() & 3) && writable && masking);
            bool byte1PerSlot = byte && (bytePartialCP || masking);
            bool pseudo = (accessType == AccessType::PseudoBlock)
                    | needsPseudoblock(
                            hw, T, R, C, atype, astrategy, writable, masking);
            int maxElements = 0;
            int maskGranularity = 1;
            int maxSIMD = maxScatteredSIMD(hw, astrategy);
            bool oword = false, aoword = false;
            int npack = 0;
            bool canQW = false, mustQW = false;

            bool a32 = (astrategy.base.getModel() == ModelA32);
            bool a64 = (astrategy.base.getModel() == ModelA64);
            bool sc = (astrategy.base.getModel() == ModelSC);
            bool slm = (astrategy.base.getModel() == ModelSLM);

            if (!pseudo && byte) return false;

            if (astrategy.newDP && !pseudo) {
                bool qword = ((atype.alignment | (R * C * T)) % 8 == 0);
                block.ebytes = qword ? 8 : 4;
                maxElements = (64 * block.ebytes) / T;
            } else if (!pseudo) {
                int maxCount = 8;
                oword = !a64;
                aoword = ((atype.alignment & 0xF) != 0) || sc;
                if (hw > HW::Gen12LP) {
                    oword |= (atype.alignment & 0x1F) != 0;
                    if (slm) maxCount = 16;
                }
                block.ebytes = oword ? 16 : 32;
                maxElements = maxCount * block.ebytes / T;
                maskGranularity = 4; // Block accesses mask by dwords
            } else {
                canQW = ((R * C * T | atype.alignment) % 8 == 0);
                if (!astrategy.newDP) canQW &= !byte && a64;
                if (remainderR || remainderC) canQW &= (T.size() % 8 == 0);
                if (astrategy.atomic
                        && hasNativeAtomicAdd(hw, T.real(), atype, astrategy))
                    canQW = mustQW = (T.real().size() >= 8);
                auto stride = canQW ? 8 : 4;
                auto maxNPack = byte1PerSlot ? 1 : std::max<int>(1, stride / T);
                maxElements = maxSIMD * maxNPack;
                if (T.size() > stride) maxElements = maxElements * stride / T;
            }

            auto maxABlock = maxElements / (byte1PerSlot ? 1 : atype.crosspack);

            switch (atype.layout) {
                case MatrixLayout::Pc:
                    rblock = std::min<int>(maxABlock, R);

                    if (atype.tileR)
                        rblock = std::min<int>(rblock, atype.tileR);
                    if ((atype.tileR ? atype.tileR : atype.packSize)
                            == rblock) {
                        cblock = std::min<int>(maxElements / rblock, C);
                        if (cblock < atype.crosspack
                                && isLargeCrosspack(T, atype.crosspack)) {
                            cblock = atype.crosspack;
                            rblock = std::min<int>(
                                    rblock, maxElements / cblock);
                        }
                    } else
                        cblock = atype.crosspack; // Remainder loop: no longer packed in memory

                    block.crosspack = atype.crosspack;
                    C = div_up(C, atype.crosspack);
                    break;
                case MatrixLayout::N:
                    if (atype.crosspack > 1) stub();
                    if (atype.tileR == R && R <= maxElements) {
                        cblock = std::min<int>(maxElements / R, C);
                        rblock = R;
                    } else {
                        cblock = 1;
                        rblock = std::min<int>(maxElements, R);
                    }
                    break;
                case MatrixLayout::Pr:
                    cblock = std::min<int>(maxABlock, C);

                    if (atype.tileC)
                        cblock = std::min<int>(cblock, atype.tileC);
                    if ((atype.tileC ? atype.tileC : atype.packSize)
                            == cblock) {
                        rblock = std::min<int>(maxElements / C, R);
                        if (rblock < atype.crosspack
                                && isLargeCrosspack(T, atype.crosspack)) {
                            rblock = atype.crosspack;
                            cblock = std::min<int>(
                                    cblock, maxElements / rblock);
                        }
                    } else
                        rblock = atype.crosspack;

                    block.crosspack = atype.crosspack;
                    R = div_up(R, atype.crosspack);
                    break;
                case MatrixLayout::T:
                    if (atype.crosspack > 1) stub();
                    if (atype.tileC == C && C <= maxElements) {
                        rblock = std::min<int>(maxElements / cblock, R);
                        cblock = C;
                    } else {
                        rblock = 1;
                        cblock = std::min<int>(maxElements, C);
                    }
                    break;
            }

            rblock = std::min(rblock, maxRBlock);
            cblock = std::min(cblock, maxCBlock);

            if (pseudo) {
                bool qword = mustQW
                        || (canQW && (rblock * cblock * T >= 4 * maxSIMD));
                npack = std::max<int>(1, (qword ? 8 : 4) / T);
                if (byte1PerSlot) {
                    if (isLargeCrosspack(T, block.crosspack)) {
                        if (block.crosspack == (colMajor ? cblock : rblock))
                            block.colMajor = colMajor = effCM;
                        else
                            stub();
                    }
                    block.crosspack = npack;
                    npack = 1;
                    (effCM ? cblock : rblock) = 1;
                }
                maskGranularity = qword ? 8 : byte1PerSlot ? T.size() : 4;
            }

            if (remainderR) {
                if (effCM) {
                    // rblock cannot be more than 16 dwords = 64 bytes for masking
                    //  except for pseudo-block
                    int rblockLimit = pseudo ? rblock : 64 / T;

                    if (avoidFragment)
                        rblock = std::min<int>(rblock, rblockLimit);
                    if (rblock > rblockLimit)
                        block.rowFragment = rblockLimit;
                    else {
                        // For sizeof(T) < maskGranularity, this is a bit of a cheat.
                        //
                        // As long as we do not need to write to this matrix, we can read
                        // in maskGranularity-sized chunks knowing we will never cross a page boundary.

                        if (writable && (T.size() & (maskGranularity - 1)))
                            return false;
                        if (!pseudo && oword && aoword) hw_unsupported();

                        if (!pseudo
                                && !(isPacked(atype.layout)
                                        && (atype.packSize == rblock)))
                            cblock = 1;

                        vrmask.isFixed = false;
                        vrmask.rsize = rblock;
                        vrmask.bitRep
                                = std::max<int>(T.size() / maskGranularity, 1);
                        vrmask.maskRep = cblock;
                        vrmask.rdivide = std::max<int>(maskGranularity / T, 1);
                    }
                } else {
                    if (avoidFragment && !remainderC) {
                        // No native masking in this dimension. One mask/row.
                        rblock = 1;
                        vrmask.isFixed = false;
                        vrmask.bitRep = 16;
                        vrmask.maskRep = 1;
                        vrmask.rdivide = 1;
                        vrmask.rsize = 1;
                    } else {
                        // Fragment it. Could actually handle rowFragment = 2 by changing descriptor.
                        block.rowFragment = 1;
                    }
                }
            }

            if (remainderC) {
                if (!effCM) {
                    // cblock cannot be more than 16 dwords = 64 bytes except for pseudo-block
                    int cblockLimit = pseudo ? cblock : 64 / T;

                    if (avoidFragment)
                        cblock = std::min<int>(cblock, cblockLimit);
                    if (cblock > cblockLimit)
                        block.colFragment = cblockLimit;
                    else {
                        if (writable && (T.size() & (maskGranularity - 1)))
                            return false;
                        if (!pseudo && oword && aoword) hw_unsupported();

                        if (!pseudo
                                && !(isPacked(atype.layout)
                                        && (atype.packSize == cblock)))
                            rblock = 1;

                        vcmask.isFixed = false;
                        vcmask.rsize = cblock;
                        vcmask.bitRep
                                = std::max<int>(T.size() / maskGranularity, 1);
                        vcmask.maskRep = rblock;
                        vcmask.rdivide = std::max<int>(maskGranularity / T, 1);
                    }
                } else {
                    if (avoidFragment && !remainderR) {
                        // No native masking in this dimension. One mask/column.
                        cblock = 1;
                        vcmask.isFixed = false;
                        vcmask.bitRep = 16;
                        vcmask.maskRep = 1;
                        vcmask.rdivide = 1;
                        vcmask.rsize = 1;
                    } else {
                        // Fragment it. Could actually handle colFragment = 2 by changing descriptor.
                        block.colFragment = 1;
                    }
                }
            }

            int nbytes = (rblock * cblock) * T;
            block.simdSize
                    = clamp(roundup_pow2(nbytes) / maskGranularity, 1, maxSIMD);
            block.ld = colMajor ? rblock : cblock;
            if (!pseudo) {
                if (astrategy.newDP) block.simdSize = 1;
                block.count = div_up(nbytes, block.ebytes);
                block.extra = aoword;
                if (block.ebytes == 16 && !(a32 || a64)
                        && !aoword) // BTS/SLM oword loads are oword-addressed.
                    block.addrShift = 4;
            } else {
                block.count = byte ? std::min(nbytes, npack * T) : 1;
                block.ebytes = byte ? 1 : maskGranularity;
                block.extra = 1;
                if (!(a32 || a64
                            || pseudoblockUseSurface(atype, astrategy, block)
                            || astrategy.atomic))
                    block.addrShift = log2(block.ebytes);
            }
            if (astrategy.newDP) block.addrShift = 0;
            break;
        }
        case AccessType::Block2D:
        case AccessType::Block2DTranspose:
        case AccessType::Block2DVNNI: {
            // bytes * array length <= 8
            // width * array length <= 64 bytes
            //  => width <= 1 GRF
            // height <= 32 (load) 8 (store)
            // array length = 1 for store, transpose
            //
            // normal: width >= 4 bytes
            // transpose: d32 only
            // vnni: d8/d16 only, height >= 4 bytes
            bool transpose = (accessType == AccessType::Block2DTranspose);
            bool vnni = (accessType == AccessType::Block2DVNNI);

            bool memCM = block.colMajor;
            block.colMajor ^= transpose;
            auto X = memCM ? R : C;
            auto Y = memCM ? C : R;
            auto &xblock = memCM ? rblock : cblock;
            auto &yblock = memCM ? cblock : rblock;
            auto maxXBlock = memCM ? maxRBlock : maxCBlock;
            auto maxYBlock = memCM ? maxCBlock : maxRBlock;

            if (hw != HW::XeHPC || !astrategy.newDP) hw_unsupported();
            if (atype.alignment % 16) hw_unsupported();

            // Choose underlying type.
            auto Tblock = T;
            if (transpose) {
                if (Tblock.size() > 4) hw_unsupported();
                Tblock = Type::u32;
                maxXBlock = std::min(maxXBlock, (8 * Tblock) / T);
            } else if (vnni) {
                if (Tblock.size() >= 4) hw_unsupported();
                if ((Y * Tblock) % 4) hw_unsupported();
                maxXBlock = std::min(maxXBlock, 16);
            } else {
                if (Tblock.size() > 8) Tblock = Type::u64;
                block.crosspack = atype.crosspack;
            }
            if ((X * T) % 4) hw_unsupported();

            // Reinterpret X/maxXBlock to underlying type.
            maxXBlock = (maxXBlock * T) / Tblock;
            auto X_logical = X;
            X = (X * T) / Tblock;

            // Carve out a maximal allowed block size.
            xblock = std::min(X, 64 / Tblock);
            xblock = std::max(xblock, 4 / Tblock);
            int yblockLimit = writable ? 8 : 32;

            if (isPacked(atype.layout) && 2 * xblock <= X
                    && X_logical == atype.packSize) {
                // Split logical x dimension into multiple spans to accomodate width restriction.
                if (astrategy.address2D) stub();
                int multiX = X / xblock;
                xblock *= multiX;
                yblockLimit /= multiX;
            }

            yblock = std::min({maxYBlock, Y, yblockLimit});

            // Choose # of blocks. In postprocessLayout, this RegisterBlock will be
            //  split into one RegisterBlock for each block in the array.
            int count = 1;
            if (!(writable || transpose)) {
                count = rounddown_pow2(xblock / maxXBlock);
                count = std::min({count, 8 / Tblock, 64 / xblock});
                count = std::max(count, 1);
            }
            xblock = std::min(xblock, maxXBlock * count);

            // Crosspack calculation.
            int crosspack = (transpose || vnni) ? std::max(1, 4 / T) : 1;
            if (atype.crosspack == 1)
                block.crosspack = crosspack;
            else if (atype.crosspack == crosspack)
                block.crosspack = 1;
            else
                return false;

            // Convert size from underlying type to our actual type.
            xblock = (xblock * Tblock) / T;

            block.simdSize = 1;
            block.ld = roundup_pow2(transpose ? yblock : xblock);
            block.ebytes = Tblock.size();
            block.count = count;
            block.extra = T.size();
            auto bytes = align_up((block.colMajor ? cblock : rblock) / count,
                                 block.crosspack)
                    * block.ld * count * T;
            block.msgRegs = GRF::bytesToGRFs(hw, bytes);
            break;
        }
    }

    // The mask moduli are almost always rblock/cblock.
    // Also, clamp mask reps to ensure mask length does not exceed SIMD size.
    if (block.rowMask && !block.rowMask.fixed.isFixed) {
        if (vrmask.rsize == 0) vrmask.rsize = rblock;
        vrmask.maskRep = std::min<int>(vrmask.maskRep,
                std::max<int>(1,
                        vrmask.rdivide * block.simdSize
                                / (vrmask.bitRep * vrmask.rsize)));
        block.noRowsOK = true; // All-zero masks are always OK.
    }
    if (block.colMask && !block.colMask.fixed.isFixed) {
        if (vcmask.rsize == 0) vcmask.rsize = cblock;
        vcmask.maskRep = std::min<int>(vcmask.maskRep,
                std::max<int>(1,
                        vcmask.rdivide * block.simdSize
                                / (vcmask.bitRep * vcmask.rsize)));
        block.noColsOK = true;
    }

    return true;
}

template <HW hw>
bool gemm_kernel_generator_t<hw>::tryAddMasking(Type T, RegisterBlock &block,
        bool remainderR, bool remainderC, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    auto blockNew = block;
    blockNew.remainderR |= remainderR;
    blockNew.remainderC |= remainderC;

    auto curAccessType = implAccessType(atype, astrategy, block);

    if (curAccessType == AccessType::Block) {
        if (astrategy.newDP) return false;
        if (hw >= HW::XeHPC) return false;
    }

    bool remChanged = (block.colMajor ? (remainderR && !block.remainderR)
                                      : (remainderC && !block.remainderC));

    if (remChanged && !isBlock2D(curAccessType)) {
        int rblock, cblock;
        if (!getBlockInfo(T, atype, astrategy, block.nr, block.nc,
                    blockNew.remainderR, blockNew.remainderC, block.writable,
                    true, 0, 0, rblock, cblock, blockNew))
            return false;
        if (rblock != block.nr || cblock != block.nc) return false;
        if (implAccessType(atype, astrategy, blockNew) != curAccessType)
            return false;
        if (curAccessType != AccessType::Block) {
            if (blockNew.ebytes != block.ebytes) return false;
            if (blockNew.ebytes == 1 && blockNew.count != block.count)
                return false;
        }
    }

    block = blockNew;
    return true;
}

template <HW hw>
bool gemm_kernel_generator_t<hw>::tryAddMasking(Type T,
        vector<RegisterBlock> &layout, bool remainderR, bool remainderC,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    auto layoutNew = layout;
    for (auto &block : layoutNew) {
        if (!tryAddMasking(T, block, remainderR, remainderC, atype, astrategy))
            return false;
    }
    std::swap(layout, layoutNew);
    return true;
}

template <HW hw>
void gemm_kernel_generator_t<hw>::addMasking(Type T,
        vector<RegisterBlock> &layout, bool remainderR, bool remainderC,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    for (auto &block : layout)
        if (!tryAddMasking(T, block, remainderR, remainderC, atype, astrategy))
            stub();
}

template <HW hw>
void gemm_kernel_generator_t<hw>::addMasking(Type T,
        vector<RegisterBlock> &layout, vector<GRFRange> &addrs,
        const Subregister &ld, bool remainderR, bool remainderC,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const CommonStrategy &strategy, CommonState &state) {
    // Check if masking can be trivially enabled without changing the layout.
    if (tryAddMasking(T, layout, remainderR, remainderC, atype, astrategy))
        return;

    // If not, tear down the old layout and create a new one in its place, recalculating address registers.
    vector<RegisterBlock> layoutNew;
    int r, c;
    bool remR = remainderR || hasRemainders(layout, true, false);
    bool remC = remainderC || hasRemainders(layout, false, true);
    getLayoutDims(layout, r, c);
    if (!getRegLayout(T, layoutNew, r, c, remR, remC, false, true, 0, 0, atype,
                astrategy))
        stub();
    if (getRegCount(layoutNew) != getRegCount(layout)) stub();
    if (isLayoutColMajor(layoutNew) != isLayoutColMajor(layout)) stub();

    int shift = 0;
    auto addr0 = getOriginAddr(layout, addrs, atype, astrategy, &shift);
    std::swap(layout, layoutNew);
    if (shift > 0) shl(1, addr0, addr0, shift);
    safeReleaseRanges(addrs, state);
    state.ra.claim(addr0);

    Address2DParams params2D {};
    if (astrategy.address2D) stub();
    allocAddrRegs(addrs, layout, atype, astrategy, state);
    setupAddr(T, addrs, addr0, layout, ld, atype, astrategy, strategy, state,
            params2D);

    state.ra.safeRelease(addr0);
}

template <HW hw>
bool gemm_kernel_generator_t<hw>::getSubblock(Type T, RegisterBlock &blockDst,
        const RegisterBlock &blockSrc, bool column, int x1, int x2,
        int x1Unclamped, int x2Unclamped, bool overrunOK,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    auto effAccessType = effectiveAccessType(atype, astrategy, blockSrc);
    blockDst = blockSrc;

    auto &ns = (column ? blockDst.nc : blockDst.nr);
    auto &nt = (column ? blockDst.nr : blockDst.nc);
    int oldNS = ns;

    (column ? blockDst.offsetC : blockDst.offsetR) += x1;
    ns = x2 - x1;

    if ((ns == oldNS) && (overrunOK || !blockSrc.hasNoLoad)) return true;

    if (blockSrc.colMajor == column) {
        if (x1 % blockSrc.crosspack) return false;

        blockDst.offsetBytes += (x1 * blockSrc.bytes) / oldNS;

        if (blockSrc.isLoadBlock()) switch (effAccessType) {
                case AccessType::Scattered:
                case AccessType::ChannelScattered:
                    blockDst.count = x2 - x1;
                    if (blockDst.ebytes == 1)
                        blockDst.count *= T.size();
                    else if (blockDst.splitComplex)
                        blockDst.count *= 2;
                    else if (T.size() < blockDst.ebytes) {
                        // Extra alignment path with small types.
                        // Check to see if we can still use this element size,
                        //  if not downgrade to scattered byte.
                        // Note for surface accesses this requires shifting the addresses back.
                        auto bcount = blockDst.count * T;
                        if (bcount % 4) {
                            blockDst.ebytes = 1;
                            blockDst.addrShift = 0;
                            blockDst.count = bcount;
                            if (blockDst.count > 4) stub();
                        } else
                            blockDst.count = bcount >> 2;
                    }
                    break;
                case AccessType::Block:
                case AccessType::PseudoBlock: {
                    auto offBytes = x1 * nt * T;
                    if (offBytes % blockDst.ebytes) return false;
                    auto reqBytes = (x2 - x1) * nt * T;
                    auto align = std::min<int>(
                            blockDst.ebytes, blockDst.simdSize * 4);
                    if (!overrunOK && (reqBytes & (align - 1))) return false;
                    auto ncount = div_up(reqBytes, blockDst.ebytes);
                    auto count = roundup_pow2(ncount);
                    if (!overrunOK && (count != ncount)) return false;
                    if (effAccessType == AccessType::Block)
                        blockDst.count = count;
                    else
                        blockDst.simdSize = count / blockDst.count;
                    break;
                }
                case AccessType::Block2D: break;
                case AccessType::Block2DTranspose:
                case AccessType::Block2DVNNI:
                    int crosspack = std::max(1, 4 / blockDst.ebytes);
                    if (x1 % crosspack || x2 % crosspack) return false;
                    break;
            }

        blockDst.calcBytes(T, astrategy);
    } else {
        blockDst.offsetBytes += x1 * T * blockSrc.crosspack;

        if (blockSrc.isLoadBlock()) switch (effAccessType) {
                case AccessType::Block:
                case AccessType::PseudoBlock: {
                    // Update count and mask information.
                    // Beware, cheat: with DW-aligned sub-DW types, true block may be downgraded to byte PseudoBlock,
                    //                which requires 2 address registers, though only 1 is used, and only 1 may be allocated.
                    int rblock, cblock;
                    (void)getBlockInfo(T, atype, astrategy, blockDst.nr,
                            blockDst.nc, blockDst.remainderR,
                            blockDst.remainderC, blockDst.writable, false, 0, 0,
                            rblock, cblock, blockDst);
                    blockDst.simplify(T);
                    break;
                }
                case AccessType::Scattered:
                case AccessType::ChannelScattered: {
                    if (T.size() > blockDst.ebytes) return false;
                    if (x1 != 0) return false;
                    if (!is_zero_or_pow2(x2)) return false;

                    blockDst.simdSize = div_up(ns * T, blockDst.ebytes);

                    auto minSIMD = minScatteredSIMD(hw, astrategy);
                    if (blockDst.simdSize <= minSIMD
                            && blockSrc.simdSize > minSIMD) {
                        if (blockDst.count > 1 && blockDst.ebytes > 1)
                            return false;
                        blockDst.ld >>= 1;
                    }
                    break;
                }
                case AccessType::Block2D:
                case AccessType::Block2DTranspose:
                case AccessType::Block2DVNNI:
                    if (ns != oldNS)
                        stub(); // Can do this, but not implemented.
                    if (blockDst.simdSize != 0) // Recompute block array length.
                        blockDst.count = div_up(x2Unclamped,
                                isColMajor(atype.layout) ? blockDst.nr
                                                         : blockDst.nc);
                    // TODO: need to recompute ld
                    break;
            }

        blockDst.calcBytes(T, astrategy);
    }

    return true;
}

// Get list of subblocks intersecting rows/columns [x1, x2).
template <HW hw>
bool gemm_kernel_generator_t<hw>::getSubblocks(Type T,
        vector<RegisterBlock> &sublayout, const vector<RegisterBlock> &layout,
        bool column, int x1, int x2, bool overrunOK,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    auto RegisterBlock::*nq = column ? &RegisterBlock::nc : &RegisterBlock::nr;
    auto RegisterBlock::*offsetQ
            = column ? &RegisterBlock::offsetC : &RegisterBlock::offsetR;

    sublayout.clear();

    for (auto &block : layout) {
        int qq1Unclamped = x1 - block.*offsetQ;
        int qq2Unclamped = x2 - block.*offsetQ;
        int qq1 = clamp<int>(qq1Unclamped, 0, block.*nq);
        int qq2 = clamp<int>(qq2Unclamped, 0, block.*nq);
        if (qq2 > qq1) {
            RegisterBlock subblock;
            if (!getSubblock(T, subblock, block, column, qq1, qq2, qq1Unclamped,
                        qq2Unclamped, overrunOK, atype, astrategy)) {
                status << "Could not make subblock." << status_stream::endl;
                return false;
            }
            sublayout.push_back(subblock);
        }
    }
    return true;
}

// Get list of subblocks intersecting rows/columns [x1, x2), and associated address registers and/or indices.
// Returns false if fragmenting failed, or an address register doesn't match a previous one.
template <HW hw>
bool gemm_kernel_generator_t<hw>::getSubblocks(Type T,
        vector<RegisterBlock> &sublayout, vector<GRFRange> *subaddrs,
        vector<int> *indices, const vector<RegisterBlock> &layout,
        const vector<GRFRange> *addrs, bool column, int x1, int x2,
        bool overrunOK, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    auto RegisterBlock::*nq = column ? &RegisterBlock::nc : &RegisterBlock::nr;
    auto RegisterBlock::*offsetQ
            = column ? &RegisterBlock::offsetC : &RegisterBlock::offsetR;

    if (subaddrs) subaddrs->clear();
    if (indices) indices->clear();
    sublayout.clear();

    for (int b = 0; b < int(layout.size()); b++) {
        auto &block = layout[b];
        int qq1Unclamped = x1 - block.*offsetQ;
        int qq2Unclamped = x2 - block.*offsetQ;
        int qq1 = clamp<int>(qq1Unclamped, 0, block.*nq);
        int qq2 = clamp<int>(qq2Unclamped, 0, block.*nq);
        if (qq2 > qq1) {
            RegisterBlock subblock;
            if (!getSubblock(T, subblock, block, column, qq1, qq2, qq1Unclamped,
                        qq2Unclamped, overrunOK, atype, astrategy)) {
                status << "Could not make subblock." << status_stream::endl;
                return false;
            }
            if (subblock.offsetR != block.offsetR
                    || subblock.offsetC != block.offsetC) {
                status << "Subblock is not aligned to parent block."
                       << status_stream::endl;
                return false;
            }
            if (subaddrs) subaddrs->push_back((*addrs)[b]);
            if (indices) indices->push_back(int(b));
            sublayout.push_back(subblock);
        }
    }
    return true;
}

// Get list of subblocks intersecting rows/columns [x1, x2), and associated address registers.
// Returns false if fragmenting failed, or an address register doesn't match a previous one.
template <HW hw>
bool gemm_kernel_generator_t<hw>::getSubblocks(Type T,
        vector<RegisterBlock> &sublayout, vector<GRFRange> &subaddrs,
        const vector<RegisterBlock> &layout, const vector<GRFRange> &addrs,
        bool column, int x1, int x2, bool overrunOK,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    return getSubblocks(T, sublayout, &subaddrs, nullptr, layout, &addrs,
            column, x1, x2, overrunOK, atype, astrategy);
}

// Get list of subblocks intersecting rows/columns [x1, x2), and indices of associated address registers.
// Returns false if fragmenting failed, or an address register doesn't match a previous one.
template <HW hw>
bool gemm_kernel_generator_t<hw>::getSubblocks(Type T,
        vector<RegisterBlock> &sublayout, vector<int> &indices,
        const vector<RegisterBlock> &layout, bool column, int x1, int x2,
        bool overrunOK, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    return getSubblocks(T, sublayout, nullptr, &indices, layout, nullptr,
            column, x1, x2, overrunOK, atype, astrategy);
}

// Adjust address registers as needed for a newly-created subblock.
template <HW hw>
void gemm_kernel_generator_t<hw>::adjustSubblockAddrs(Type T,
        const vector<RegisterBlock> &sublayout,
        const vector<GRFRange> &subaddrs, const vector<RegisterBlock> &layout,
        const vector<GRFRange> &addrs, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const CommonStrategy &strategy, const CommonState &state) {
    bool a64 = (astrategy.base.getModel() == ModelA64);

    auto nsubs = int(sublayout.size());
    auto nblocks = int(layout.size());

    for (int isub = 0; isub < nsubs; isub++) {
        // Find parent block by comparing address registers.
        auto &subaddr = subaddrs[isub];
        const RegisterBlock *pptr = nullptr;
        for (int i = 0; i < nblocks; i++) {
            if (addrs[i].getBase() == subaddr.getBase()) {
                pptr = &layout[i];
                break;
            }
        }
        if (!pptr) stub();

        auto &block = *pptr;
        auto &subblock = sublayout[isub];

        auto off = getAddr0Offset(block, atype, astrategy);
        auto suboff = getAddr0Offset(subblock, atype, astrategy);

        // Perform any necessary shifts/moves. Moves are only for non-A64 block->pseudoblock settings.
        if (suboff != off) {
            if (subblock.simdSize != 1)
                stub(); // Need to prepare more pseudoblock addresses.
            mov<uint32_t>(1, subaddr[0][suboff], subaddr[0][off]);
        }
        if (subblock.addrShift != block.addrShift) {
            map(hw, a64 ? Type::u64 : Type::u32, subaddr, subaddr, strategy,
                    [&](int simd, GRF r, GRF _) {
                        auto shift = block.addrShift - subblock.addrShift;
                        (shift > 0) ? eshl(simd, r, r, +shift, strategy, state)
                                    : eshr(simd, r, r, -shift, strategy, state);
                    });
        }

        if (isBlock2D(astrategy.accessType)) {
            // Adjust 2D block header as needed.
            int bw, bh;
            bool memCM = isColMajor(atype.layout);
            auto RegisterBlock::*nw
                    = memCM ? &RegisterBlock::nr : &RegisterBlock::nc;
            auto RegisterBlock::*nh
                    = memCM ? &RegisterBlock::nc : &RegisterBlock::nr;
            bool remW = memCM ? subblock.remainderR : subblock.remainderC;
            bool remH = memCM ? subblock.remainderC : subblock.remainderR;
            getBlock2DWH(bw, bh, atype, subblock);

            if (!astrategy.address2D) {
                if (subblock.*nw != block.*nw
                        || subblock.count != block.count) {
                    int newW = bw * subblock.count * subblock.ebytes - 1;
                    remW ? min_(1, subaddr[0].ud(2), subaddr[0].ud(2), newW)
                         : mov(1, subaddr[0].ud(2), newW);
                }
                if (subblock.*nh != block.*nh) {
                    int newH = bh * subblock.ebytes - 1;
                    remH ? min_(1, subaddr[0].ud(3), subaddr[0].ud(3), newH)
                         : mov(1, subaddr[0].ud(3), newH);
                }
            }
            if (subblock.nr != block.nr || subblock.nc != block.nc
                    || subblock.count != block.count)
                mov(1, subaddr[0].ud(7),
                        (bw - 1) | ((bh - 1) << 8)
                                | ((subblock.count - 1) << 16));
        }
    }
}

// Split 2D block array loads into multiple blocks.
static inline void postprocessLayout2D(vector<RegisterBlock> &layout,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    if (!isBlock2D(astrategy.accessType)) return;

    int maxCount = 1;
    for (auto &block : layout)
        maxCount = std::max(maxCount, int(block.count));
    if (maxCount == 1) return;

    vector<RegisterBlock> xlayout;
    xlayout.reserve(layout.size() * maxCount);

    bool memCM = isColMajor(atype.layout);
    auto RegisterBlock::*nx = memCM ? &RegisterBlock::nr : &RegisterBlock::nc;
    auto RegisterBlock::*offsetX
            = memCM ? &RegisterBlock::offsetR : &RegisterBlock::offsetC;

    for (auto &block : layout) {
        auto nblock = block;
        nblock.*nx /= block.count;
        if (!isTransposing(astrategy.accessType)) nblock.ld /= block.count;

        for (int i = 0; i < block.count; i++) {
            xlayout.push_back(nblock);
            nblock.*offsetX += nblock.*nx;
            nblock.simdSize = 0; // Blocks > 0 do not need loads.
        }
    }

    std::swap(layout, xlayout);
}

// Split large crosspack blocks into smaller pieces so that they can be transposed.
static inline void postprocessLayoutLargeCP(Type T,
        vector<RegisterBlock> &layout, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    if (!isLargeCrosspack(T, atype.crosspack)) return;

    bool haveLargeCP = false;
    for (const auto &block : layout) {
        haveLargeCP |= isLargeCrosspack(T, block.crosspack);
        if (haveLargeCP) break;
    }

    if (!haveLargeCP) return;

    vector<RegisterBlock> xlayout;
    xlayout.reserve(layout.size());

    for (const auto &block : layout) {
        if (!isLargeCrosspack(T, block.crosspack))
            xlayout.push_back(block);
        else {
            auto ny = block.colMajor ? &RegisterBlock::nc : &RegisterBlock::nr;
            auto offsetY = block.colMajor ? &RegisterBlock::offsetC
                                          : &RegisterBlock::offsetR;

            if (block.*ny % block.crosspack) return;
            int blocks = (block.*ny / block.crosspack);
            auto nblock = block;
            nblock.*ny = block.crosspack;
            nblock.simplify(T);
            for (int i = 0; i < blocks; i++) {
                xlayout.push_back(nblock);
                nblock.simdSize = 0;
                nblock.*offsetY += nblock.*ny;
            }
        }
    }

    std::swap(layout, xlayout);
}

static inline void postprocessLayout(Type T, vector<RegisterBlock> &layout,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    postprocessLayout2D(layout, atype, astrategy);
    postprocessLayoutLargeCP(T, layout, atype, astrategy);
}

// Add a submatrix to a register layout.
template <HW hw>
bool gemm_kernel_generator_t<hw>::addToRegLayout(Type T,
        std::vector<RegisterBlock> &layout, int nr, int nc, int roff, int coff,
        bool remainderR, bool remainderC, bool writable, bool avoidFragment,
        int maxRBlock, int maxCBlock, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    int rblock, cblock;
    RegisterBlock blockTemplate;
    if (!getBlockInfo(T, atype, astrategy, nr, nc, remainderR, remainderC,
                writable, avoidFragment, maxRBlock, maxCBlock, rblock, cblock,
                blockTemplate))
        return false; /* Cannot handle requested block and remainder. */

    if (rblock == 0 || cblock == 0) return false;

    blockTemplate.nr = rblock;
    blockTemplate.nc = cblock;

    if (isColMajor(atype.layout)) {
        // Order blocks in column-major fashion.
        for (int c = 0; c + cblock <= nc; c += cblock) {
            for (int r = 0; r + rblock <= nr; r += rblock) {
                auto thisBlock = blockTemplate;

                thisBlock.offsetR = r + roff;
                thisBlock.offsetC = c + coff;

                layout.push_back(thisBlock);
            }
        }
    } else {
        // Order blocks in row-major fashion.
        for (int r = 0; r + rblock <= nr; r += rblock) {
            for (int c = 0; c + cblock <= nc; c += cblock) {
                auto thisBlock = blockTemplate;

                thisBlock.offsetR = r + roff;
                thisBlock.offsetC = c + coff;

                layout.push_back(thisBlock);
            }
        }
    }

    // Handle remainder recursively, checking for infinite recursion.
    int rrem = nr % rblock;
    int crem = nc % cblock;

    status << "Register layout: " << nr << 'x' << nc << " -> blocks " << rblock
           << 'x' << cblock << " remainder " << rrem << 'x' << crem
           << status_stream::endl;

    bool success = true;
    if (rrem || crem) {
        if ((nr == rrem || rrem == 0) && (nc == crem || crem == 0)) {
            status << "Cannot load/store requested matrix block size."
                   << status_stream::endl;
            success = false;
        } else {
            if (rrem)
                success &= addToRegLayout(T, layout, rrem, nc - crem, nr - rrem,
                        0, remainderR, remainderC, writable, avoidFragment,
                        maxRBlock, maxCBlock, atype, astrategy);
            if (crem)
                success &= addToRegLayout(T, layout, nr, crem, 0, nc - crem,
                        remainderR, remainderC, writable, avoidFragment,
                        maxRBlock, maxCBlock, atype, astrategy);
        }
    }
    return success;
}

// Add a submatrix (contiguous in memory) to a block-accessed register layout.
template <HW hw>
bool gemm_kernel_generator_t<hw>::add1DBlockToRegLayout(Type T,
        vector<RegisterBlock> &layout, int r, int c, bool writable,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    // Skip pseudoblock cases (possible to support though)
    if (needsPseudoblock(hw, T, r, c, atype, astrategy, writable, false))
        return false;

    // Get total number of bytes to load. No masking supported, so stub if
    //  number of bytes not divisible by 16 (1 oword).
    int nbytes = r * c * T;
    int align = 16;
    if (astrategy.newDP) align = 4;

    if (nbytes & (align - 1)) return false;

    // Get block info.
    int maxBBytes = 0;
    int ebytes = 0;
    int extra = 0;
    int addrShift = 0;
    int maxSIMD = 1;

    if (astrategy.newDP) {
        bool qword = (nbytes | atype.alignment) % 8 == 0;
        ebytes = qword ? 8 : 4;
        maxBBytes = ebytes * 64;
    } else {
        bool a64 = (astrategy.base.getModel() == ModelA64);
        bool oword = !a64;
        bool aoword = (astrategy.base.getModel()
                == ModelSC); // SC only does aligned oword
        if (hw >= HW::XeHP) oword |= ((atype.alignment & 0x1F) != 0);

        extra = aoword;
        ebytes = oword ? 16 : 32;
        maxBBytes = oword ? 128 : 256;
        if (astrategy.base.getModel() == ModelSLM && hw >= HW::XeHP)
            maxBBytes = 256;
        addrShift = (!a64 && oword && !aoword) ? 4 : 0;
        maxSIMD = 16;
    }

    // Get normalized dimensions.
    bool colMajor = isColMajor(atype.layout);
    int x = colMajor ? r : c;
    auto crosspack = atype.crosspack;

    // Counters for current x and y positions.
    int cx = 0, cy = 0;

    while (nbytes > 0) {
        // Carve out the largest chunk possible.
        int bbytes = std::min<int>(maxBBytes, rounddown_pow2(nbytes));
        int belems = bbytes / T;

        // Create a true load block for first (possibly partial) row/column.
        // Then, create additional no-load blocks for any further (possible partial)
        //   rows/columns until block is exhausted.
        bool first = true;
        while (belems > 0) {
            int nxRem = belems / crosspack;
            int nx = std::min<int>(nxRem, x - cx);
            if (nx <= 0) stub();
            if (cy % crosspack) return false;

            RegisterBlock block;

            block.ld = nx;
            (colMajor ? block.nr : block.nc) = nx;
            (colMajor ? block.nc : block.nr) = crosspack;
            (colMajor ? block.offsetR : block.offsetC) = cx;
            (colMajor ? block.offsetC : block.offsetR) = cy;
            block.colMajor = colMajor;
            block.splitComplex = false;

            if (first) {
                block.ebytes = ebytes;
                block.count = div_up(bbytes, ebytes);
                block.simdSize = std::min(maxSIMD, roundup_pow2(bbytes) >> 2);
            } else
                block.ebytes = block.count = block.simdSize = 0;

            block.extra = extra;
            block.clearFlag();
            block.colMask = MaskInfo::None();
            block.rowMask = MaskInfo::None();
            block.colFragment = 0;
            block.rowFragment = 0;
            block.log2GRFBytes = GRF::log2Bytes(hw);

            block.crosspack = crosspack;
            block.remainderR = false;
            block.remainderC = false;
            block.noRowsOK = false;
            block.noColsOK = false;
            block.descRemR = false;
            block.descRemC = false;
            block.descAssigned = false;
            block.addrShift = addrShift;
            block.hasNoLoad = false;
            block.msgRegs = std::max(1, bbytes >> GRF::log2Bytes(hw));

            if (first && cx == 0 && (nxRem % x) == 0) {
                // Shortcut: one register block can represent this block access.
                int ny = belems / x;
                (colMajor ? block.nc : block.nr) = ny;
                cy += ny;
                belems = 0;
            } else {
                cx += nx;
                belems -= nx * crosspack;
                if (cx == x) {
                    cy += crosspack;
                    cx = 0;
                }
                block.hasNoLoad = first && (belems > 0);
                first = false;
            }

            layout.push_back(block);
        }

        nbytes -= bbytes;
    }

    return true;
}

static inline int getPartialCrosspack(size_t sizeofT,
        const MatrixAddressing &atype, const RegisterBlock &block) {
    if (block.ebytes == 1 && !isLargeCrosspack(sizeofT, atype.crosspack))
        return div_up(atype.crosspack, block.colMajor ? block.nc : block.nr);
    else
        return 1;
}

// Get linear element offset in tiled layout (both register and memory)
static int untile(const MatrixAddressing &atype, int i, int j, int r, int c,
        int tileR, int tileC, bool reverse = false) {
    bool cm = isColMajor(atype.layout) ^ reverse;

    if (isPacked(atype.layout)) (cm ? r : c) = atype.packSize;

    int cpR = cm ? 1 : atype.crosspack;
    int cpC = cm ? atype.crosspack : 1;

    if (tileR == 0) tileR = r;
    if (tileC == 0) tileC = c;

    int rstride = cm ? tileC : c;
    int cstride = cm ? r : tileR;
    int rtstride = cm ? cpC : tileC;
    int ctstride = cm ? tileR : cpR;

    int iTile = i % tileR;
    int jTile = j % tileC;
    i -= iTile;
    j -= jTile;
    int iCP = iTile % cpR;
    int jCP = jTile % cpC;
    iTile -= iCP;
    jTile -= jCP;
    int idx = i * rstride + j * cstride + iTile * rtstride + jTile * ctstride
            + iCP + jCP;
    return idx;
}

static int untile(const MatrixAddressing &atype, const RegisterBlock &block,
        int r, int c, int tileR, int tileC, bool reverse = false) {
    return untile(
            atype, block.offsetR, block.offsetC, r, c, tileR, tileC, reverse);
}

static int untile(const MatrixAddressing &atype, const RegisterBlock &block,
        int r, int c, bool reverse = false) {
    return untile(atype, block, r, c, atype.tileR, atype.tileC, reverse);
}

static int untile(const MatrixAddressing &atype, int i, int j, int r, int c,
        bool reverse = false) {
    return untile(atype, i, j, r, c, atype.tileR, atype.tileC, reverse);
}

// Split A/B matrix between threads.
static inline void coopSplit(bool isA, int &splitR, int &splitC, int r, int c,
        CoopSplit stype, int threads, const MatrixAddressing &atype) {
    auto &mn = isA ? r : c;
    auto &k = isA ? c : r;
    auto &splitMN = isA ? splitR : splitC;
    auto &splitK = isA ? splitC : splitR;
    auto tileMN = isA ? atype.tileR : atype.tileC;
    auto tileK = isA ? atype.tileC : atype.tileR;

    bool ok = false;

    switch (stype) {
        case CoopSplit::K:
            ok = (k % threads == 0);
            splitMN = mn;
            splitK = k / threads;
            break;
        case CoopSplit::MN:
            ok = (mn % threads == 0);
            splitMN = mn / threads;
            splitK = k;
            break;
        case CoopSplit::Linear: {
            int elems = r * c;
            ok = (elems % threads == 0);
            int selems = elems / threads;
            int cp = atype.crosspack;

            if (!tileK) tileK = k;
            if (!tileMN) tileMN = mn;

            // First try splitting into tiles in k dimension.
            if (selems >= (tileK * mn)) {
                ok &= (selems % (tileK * mn) == 0);
                splitMN = mn;
                splitK = k / threads;
                break;
            }

            ok &= (threads % (k / tileK) == 0);
            if (!ok) break;
            threads /= (k / tileK);

            // Then try splitting into tiles in m/n dimensions as well.
            if (selems >= (tileK * tileMN)) {
                ok &= (selems % (tileK * tileMN) == 0);
                splitMN = mn / threads;
                splitK = tileK;
                break;
            }

            ok &= (threads % (mn / tileMN) == 0);
            if (!ok) break;
            threads /= (mn / tileMN);

            // Then try splitting each tile in the k dimension.
            if (selems >= (cp * tileMN)) {
                ok &= (selems % (cp * tileMN) == 0);
                splitMN = tileMN;
                splitK = tileK / threads;
                break;
            }

            ok &= (threads % (tileK / cp) == 0);
            if (!ok) break;
            threads /= (tileK / cp);

            // Finally try splitting in the m/n dimensions.
            ok &= (selems % cp == 0);
            splitMN = tileMN / threads;
            splitK = cp;
            break;
        }
    }

    if (!ok)
        throw std::runtime_error(
                "Cooperative operation cannot be split evenly between "
                "threads.");
}

// Re-order a layout so that registers appear in appropriate order
//  (row or column major)
static void sortRegLayout(vector<RegisterBlock> &layout, int r, int c,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy, bool reverse = false) {
    auto order = [=](const RegisterBlock &block) {
        return untile(
                atype, block, r, c, astrategy.tileR, astrategy.tileC, reverse);
    };

    std::sort(layout.begin(), layout.end(),
            [&](const RegisterBlock &b1, const RegisterBlock &b2) {
                return (order(b1) < order(b2));
            });
}

// Create a register layout for a matrix.
template <HW hw>
bool gemm_kernel_generator_t<hw>::getRegLayout(Type T,
        vector<RegisterBlock> &layout, int r, int c, bool remainderR,
        bool remainderC, bool writable, bool avoidFragment, int maxRBlock,
        int maxCBlock, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy, bool reverseOrder) {
    bool success = false;

    layout.clear();

    // Tiling handling.
    if (astrategy.tileR > 0)
        maxRBlock = (maxRBlock == 0) ? astrategy.tileR
                                     : gcd(int(astrategy.tileR), maxRBlock);
    if (astrategy.tileC > 0)
        maxCBlock = (maxCBlock == 0) ? astrategy.tileC
                                     : gcd(int(astrategy.tileC), maxRBlock);

    // Two separate strategies for creating register layout:
    //    - standard 2D partitioning
    //    - special 1D partitioning for block access to packed inputs.
    if (((atype.layout == MatrixLayout::Pc && atype.packSize == r)
                || (atype.layout == MatrixLayout::Pr && atype.packSize == c))
            && (astrategy.accessType == AccessType::Block) && !remainderR
            && !remainderC && !atype.tileR && !atype.tileC
            && (maxRBlock >= r || maxRBlock == 0)
            && (maxCBlock >= c || maxCBlock == 0)) {
        success = add1DBlockToRegLayout(
                T, layout, r, c, writable, atype, astrategy);
    }
    if (!success) {
        success = addToRegLayout(T, layout, r, c, 0, 0, remainderR, remainderC,
                writable, avoidFragment, maxRBlock, maxCBlock, atype,
                astrategy);
        sortRegLayout(layout, r, c, atype, astrategy, reverseOrder);
        postprocessLayout(T, layout, atype, astrategy);
    }
    if (!success) return false;

    int offsetBytes = 0;
    for (auto &block : layout) {
        if (block.isLoadBlock() || isBlock2D(astrategy.accessType))
            offsetBytes = alignup_pow2(offsetBytes, GRF::bytes(hw));
        block.calcBytes(T, astrategy);
        block.offsetBytes = offsetBytes;
        offsetBytes += block.bytes;
        block.simplify(T);
    }

    return true;
}

// Create a register layout for a uniform matrix not backed by memory.
template <HW hw>
void gemm_kernel_generator_t<hw>::makeUnbackedRegLayout(Type T,
        vector<RegisterBlock> &layout, int r, int c, bool colMajor,
        int crosspack, int tileR, int tileC) {
    auto block = RegisterBlock();

    if ((colMajor ? c : r) % crosspack) stub();
    layout.clear();

    if (tileR <= 0) tileR = r;
    if (tileC <= 0) tileC = c;

    int offsetBytes = 0;

    for (int i = 0; i < r; i += tileR) {
        for (int j = 0; j < c; j += tileC) {
            block.log2GRFBytes = GRF::log2Bytes(hw);
            block.nr = std::min(r - i, tileR);
            block.nc = std::min(c - j, tileC);
            block.ld = colMajor ? tileR : tileC;
            block.offsetR = i;
            block.offsetC = j;
            block.colMajor = colMajor;
            block.crosspack = crosspack;
            block.offsetBytes = offsetBytes;
            block.splitComplex = false;

            block.calcBytes(T);
            offsetBytes += block.bytes;

            block.remainderR = false;
            block.remainderC = false;
            block.simdSize = 0; // Not backed by memory.

            layout.push_back(block);
        }
    }
}

// Find the subregister in a RegisterBlock corresponding to element at offset (rr,cc),
//  as well as the contiguous elements following it (nelems).
static Subregister findBlockReg(Type T, const RegisterBlock &block, int rr,
        int cc, const GRFMultirange &regs, int &nelems, int component = -1) {
    auto Te = T;
    const int ne = (1 << block.log2GRFBytes) / Te;

    if (rr < 0 || rr >= block.nr || cc < 0 || cc >= block.nc)
        throw std::runtime_error("Requested out-of-bounds element.");

    int crosspack = block.crosspack;
    int elFixed, elLD;
    if (block.colMajor) {
        int ccx = cc % crosspack;
        elFixed = ccx + (rr * crosspack);
        elLD = cc - ccx;
        nelems = block.nr - rr;
    } else {
        int rrx = rr % crosspack;
        elFixed = rrx + (cc * crosspack);
        elLD = (rr - rrx);
        nelems = block.nc - cc;
    }

    int el = elFixed + elLD * block.ld;
    el += block.offsetBytes / Te;
    int reg = el / ne;
    int subreg = el % ne;

    return regs[reg].sub(subreg, Te.ngen());
}

// Find the subregister in a layout corresponding to element (r,c), as well as the
//  associated block, and the number of contiguous elements following it (nelems).
static Subregister findBlockReg(Type T, const vector<RegisterBlock> &layout,
        int r, int c, const GRFMultirange &regs, int &nelems,
        const RegisterBlock *&block, int component = -1) {
    for (auto &l : layout) {
        int rr = r - l.offsetR;
        int cc = c - l.offsetC;
        if (rr >= 0 && rr < l.nr && cc >= 0 && cc < l.nc) {
            block = &l;
            return findBlockReg(T, l, rr, cc, regs, nelems, component);
        }
    }

    throw std::runtime_error(
            "Could not find requested matrix element in layout.");
}

// Match the register offsets in one register layout to another, reference layout.
// Returns true if successful. If not successful, the layout is unchanged.
static bool matchLayouts(Type T, vector<RegisterBlock> &layout,
        const vector<RegisterBlock> &layoutRef) {
    vector<RegisterBlock> nlayout = layout;

    if (getRegCount(layoutRef) >= 256) return false;

    for (auto &nblock : nlayout) {
        int nelems;
        const RegisterBlock *blockRef;
        auto sr = findBlockReg(T, layoutRef, nblock.offsetR, nblock.offsetC,
                GRFRange(0, 254), nelems, blockRef);

        // Check:
        //  1. Does this register block's offset match the reference block's offset?
        if (sr.getByteOffset()
                != (nblock.offsetBytes & ((1 << nblock.log2GRFBytes) - 1)))
            return false;

        //  2. Is there any free space in the register block?
        if (nblock.nr * nblock.nc * T != nblock.bytes) return false;

        //  3. Does this register block's data layout match the reference block's layout?
        if (blockRef->colMajor != nblock.colMajor) return false;
        if (blockRef->crosspack != nblock.crosspack) return false;

        //  4. Does this register block fit inside the reference block?
        auto RegisterBlock::*nx
                = nblock.colMajor ? &RegisterBlock::nr : &RegisterBlock::nc;
        auto RegisterBlock::*ny
                = nblock.colMajor ? &RegisterBlock::nc : &RegisterBlock::nr;

        if (nblock.*nx < blockRef->*nx) {
            if (nblock.*ny > 1) return false;
        } else if (nblock.*nx == blockRef->*nx) {
            if (nblock.*ny > blockRef->*ny) return false;
        } else
            return false;

        if (nblock.*ny > 1 && (nblock.ld != blockRef->ld)) return false;

        // It's compatible. Point this register block where it belongs.
        nblock.offsetBytes
                = (sr.getBase() << nblock.log2GRFBytes) + sr.getByteOffset();
    }

    std::swap(nlayout, layout);
    return true;
}

// Like matchLayouts but allows either layout to change to match the other.
static bool matchLayoutsBidirectional(Type T, vector<RegisterBlock> &layout1,
        vector<RegisterBlock> &layout2) {
    return matchLayouts(T, layout1, layout2)
            || matchLayouts(T, layout2, layout1);
}

static bool allocateTokens(const vector<RegisterBlock> &layout,
        const GRFMultirange &regs, CommonState &state,
        const vector<GRFRange> &addrs = vector<GRFRange>()) {
    bool success = true;
    size_t origSize = state.tokenMap.size();
    auto saveTA = state.tokenAllocator;

    for (size_t l = 0; l < layout.size(); l++) {
        auto token = state.tokenAllocator.tryAlloc();
        if (token < 0)
            success = false;
        else {
            auto regKey = !regs.empty() ? regs[layout[l].offsetReg()].getBase()
                                        : addrs[l].getBase();
            state.tokenMap.push_back(std::make_pair(regKey, token));
        }
    }

    if (!success) {
        state.tokenAllocator = saveTA;
        state.tokenMap.resize(origSize);
    }

    return success;
}

static void clearTokenAllocations(HW hw, CommonState &state) {
    state.tokenMap.clear();
    state.tokenAllocator = TokenAllocator(hw);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::setupTeardownLoadStoreDesc(bool setup,
        const vector<RegisterBlock> &layout, const CommonStrategy &strategy,
        CommonState &state) {
    if (strategy.emulate.emulateDWxDW) {
        auto nconstants = (hw >= HW::XeHPG) ? 3 : 2;

        if (setup)
            for (int s = 0; s < nconstants; s++) {
                state.lsDescConstant[s] = state.ra.alloc_sub<uint32_t>();
                mov(1, state.lsDescConstant[s], uint32_t(0x00100040 << s));
            }
        else
            for (int s = 0; s < nconstants; s++)
                state.ra.safeRelease(state.lsDescConstant[s]);
    }
}

// Output code for loading address register(s) with load/store message descriptors for remainders.
template <HW hw>
void gemm_kernel_generator_t<hw>::loadLoadStoreDescriptors(bool load,
        bool store, RegisterBlock &block, const Subregister &count,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const CommonStrategy &strategy, CommonState &state) {
    MessageDescriptor descLoad; // a0.0:ud
    MessageDescriptor descStore; // a0.2 (a0.0 if no loads)
    ExtendedMessageDescriptor exdescLoad;
    ExtendedMessageDescriptor exdescStore; // a0.1

    Subregister t1 = state.ra.alloc_sub<uint32_t>();
    Subregister t2 = state.ra.alloc_sub<uint32_t>();

    if (astrategy.newDP) switch (astrategy.accessType) {
            case AccessType::ChannelScattered:
            case AccessType::Scattered: {
                bool channel = (astrategy.accessType
                        == AccessType::ChannelScattered);

                encodeLoadDescriptors(hw, descLoad, exdescLoad, block.simdSize,
                        r0, getDataSpecLSC(atype, astrategy, block, false),
                        astrategy.base, null);
                encodeStoreDescriptors(hw, descStore, exdescStore,
                        block.simdSize,
                        getDataSpecLSC(atype, astrategy, block, true),
                        astrategy.base, null);
                descLoad.cmask.cmask = 0; // also vectSize
                descStore.cmask.cmask = 0;
                exdescStore.parts.extMessageLen = 0;
                descLoad.parts.responseLen = 0;

                int underlyingSIMD = std::max<int>(
                        block.simdSize, maxScatteredSIMD(hw, astrategy) >> 1);
                int log2GRFs = log2(underlyingSIMD * block.ebytes)
                        - GRF::log2Bytes(hw);
                int log2Components = int(block.splitComplex);

                if (channel) mov(1, t2, 0x1000 << log2Components);
                mul(1, t1, state.lsDescConstant[log2GRFs + log2Components],
                        count.uw());
                channel ? shl(1, t2, t2, count)
                        : shl(1, t2, count, 12 + log2Components);
                if (store) or_(1, a0.ud(1), t1.uw(0), exdescStore.all);
                add(1, t1.uw(0), t2, -0x1000);
                if (load) or_(1, a0.ud(0), t1, descLoad.all);
                if (store) or_(1, a0.ud(load ? 2 : 0), t1.uw(0), descStore.all);
                break;
            }
            default: hw_unsupported();
        }
    else
        switch (astrategy.accessType) {
            case AccessType::ChannelScattered: {
                encodeLoadDescriptors(hw, descLoad, exdescLoad, block.simdSize,
                        r0, surface_dword(ChannelMask::rgba), astrategy.base,
                        null);
                encodeStoreDescriptors(hw, descStore, exdescStore,
                        block.simdSize, surface_dword(ChannelMask::rgba),
                        astrategy.base, null);
                descLoad.surface.cmask = 0; //
                descStore.surface.cmask = 0; // Fields to fill in.
                exdescStore.parts.extMessageLen = 0; //
                descLoad.parts.responseLen = 0;

                int log2Components = int(block.splitComplex);
                int shift = int(block.simdSize == 16) + log2Components;
                auto bitmask = uint16_t(0x0F00 << log2Components);

                if (strategy.emulate.emulateDWxDW)
                    mul(1, t1, state.lsDescConstant[shift], count.uw());
                else
                    mul(1, t1, count, uint32_t(0x00100040) << shift);
                mov(1, t2, bitmask);
                if (store) or_(1, a0.ud(1), t1.uw(0), exdescStore.all);
                shl(1, t2, t2, count);
                and_(1, t1.uw(0), t2, bitmask);
                if (load) or_(1, a0.ud(0), t1, descLoad.all);
                if (store) or_(1, a0.ud(load ? 2 : 0), t1.uw(0), descStore.all);
                break;
            }
            default: hw_unsupported();
        }

    state.ra.safeRelease(t1);
    state.ra.safeRelease(t2);
    block.sfid = exdescLoad.all;
}

template <HW hw>
InstructionModifier gemm_kernel_generator_t<hw>::getRegisterBlockMask(
        const RegisterBlock &block, CommonState &state) {
    InstructionModifier result;

    if (block.flag) {
        result |= getPhysicalFlag(block.flag, state);
        if (hw == HW::XeHPC) {
            if (block.flagAll) result |= all;
            if (block.flagAny) result |= any;
        } else if (block.flagAll)
            result |= (block.simdSize > 8) ? all16h : all8h;
        else if (block.flagAny)
            result |= (block.simdSize > 8) ? any16h : any8h;
    }

    return result;
}

// Check if a block occupies a contiguous portion of registers in the given GRFMultirange.
// If so, return index of the block's first register in the range.
static inline int contiguityCheck(
        HW hw, const RegisterBlock &block, const GRFMultirange &range) {
    auto offsetBytes = block.offsetBytes;
    if (offsetBytes & (GRF::bytes(hw) - 1))
        if (block.isLoadBlock()) stub();
    auto offsetReg = offsetBytes >> GRF::log2Bytes(hw);
    auto lastReg = GRF::bytesToGRFs(hw, offsetBytes + block.bytes);
    if (!range.contiguous(offsetReg, lastReg - offsetReg)) stub();

    return offsetReg;
}

static DataSizeLSC getDataSizeLSC(int ebytes, bool pad32) {
    switch (ebytes) {
        case 8: return DataSizeLSC::D64;
        case 4: return DataSizeLSC::D32;
        case 2: return pad32 ? DataSizeLSC::D16U32 : DataSizeLSC::D16;
        case 1: return pad32 ? DataSizeLSC::D8U32 : DataSizeLSC::D8;
    }
    throw std::runtime_error("Invalid data size");
}

template <HW hw>
DataSpecLSC gemm_kernel_generator_t<hw>::getDataSpecLSC(
        AccessType access, const RegisterBlock &block) {
    switch (access) {
        case AccessType::ChannelScattered: {
            static const ChannelMask cmasks[4] = {ChannelMask::r,
                    ChannelMask::rg, ChannelMask::rgb, ChannelMask::rgba};
            if (block.ebytes != 4) hw_unsupported();
            return D32 | cmasks[block.count - 1];
        }
        case AccessType::Scattered:
            if (block.ebytes == 8) return D64(block.count);
            if (block.ebytes == 4) return D32(block.count);
            if (block.ebytes == 1) return getDataSizeLSC(block.count, true);
            hw_unsupported();
        case AccessType::Block:
            if (block.ebytes == 8) return D64T(block.count);
            if (block.ebytes == 4) return D32T(block.count);
            hw_unsupported();
        default: stub();
    }
}

template <HW hw>
DataSpecLSC gemm_kernel_generator_t<hw>::getDataSpecLSC(
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy, const RegisterBlock &block,
        bool write) {
    return getDataSpecLSC(implAccessType(atype, astrategy, block), block)
            | (write ? astrategy.cachingW : astrategy.cachingR);
}

// Output code for prefetching a matrix chunk (XeHPG+).
template <HW hw>
void gemm_kernel_generator_t<hw>::prefetchMatrix(
        const vector<RegisterBlock> &layout, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const vector<GRFRange> &addrs, const CommonStrategy &strategy,
        CommonState &state) {
    auto nblocks = int(layout.size());

    for (int l = 0; l < nblocks; l++)
        loadMatrixBlock(null, layout[l], atype, astrategy, addrs[l], strategy,
                state, false);
}

// Output code for loading a matrix chunk into registers.
template <HW hw>
void gemm_kernel_generator_t<hw>::loadMatrix(const GRFMultirange &dest,
        const vector<RegisterBlock> &layout, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const vector<GRFRange> &addrs, const CommonStrategy &strategy,
        CommonState &state, bool zeroMask) {
    auto nblocks = int(layout.size());

    if (astrategy.prefetch && astrategy.newDP) {
        prefetchMatrix(layout, atype, astrategy, addrs, strategy, state);
        return;
    }

    if (strategy.readSuppressionWA && (hasFlags(layout) || !getDefaultNoMask()))
        doReadSuppressionWA(strategy, state);

    for (int l = 0; l < nblocks; l++) {
        auto offsetReg = contiguityCheck(hw, layout[l], dest);
        loadMatrixBlock(dest[offsetReg], layout[l], atype, astrategy, addrs[l],
                strategy, state, zeroMask);
    }
}

// Output code for loading a single matrix block into registers.
template <HW hw>
void gemm_kernel_generator_t<hw>::loadMatrixBlock(const Register &dest,
        const RegisterBlock &block, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy, const GRFRange &addr,
        const CommonStrategy &strategy, CommonState &state, bool zeroMask) {
    InstructionModifier maskMod;
    InstructionModifier mod = block.simdSize;

    // Zero SIMD size blocks are filled as part of another load. Skip them.
    if (!block.isLoadBlock()) return;

    // Get mask to apply, if any.
    auto mask = getRegisterBlockMask(block, state);
    maskMod |= mask;
    mod |= mask;

    // Look up preassigned token.
    for (auto &entry : state.tokenMap) {
        if (entry.first == dest.getBase() || entry.first == addr.getBase()) {
            mod |= SBID(entry.second);
            break;
        }
    }

    if (astrategy.newDP) switch (implAccessType(atype, astrategy, block)) {
            case AccessType::Block:
            case AccessType::Scattered:
            case AccessType::ChannelScattered: {
                auto spec = getDataSpecLSC(atype, astrategy, block, false);
                if (block.descAssigned) {
                    MessageDescriptor desc;
                    ExtendedMessageDescriptor exdesc;
                    encodeLoadDescriptors(hw, desc, exdesc, block.simdSize, r0,
                            spec, astrategy.base, null);
                    send(mod, static_cast<SharedFunction>(block.sfid), dest,
                            addr, null, exdesc.all, a0[0]);
                } else {
                    load(mod, dest, spec, astrategy.base, addr[0]);
                }
                break;
            }
            case AccessType::Block2D:
            case AccessType::Block2DTranspose:
            case AccessType::Block2DVNNI: {
                int w = 0, h = 0;
                getBlock2DWH(w, h, atype, block);
                auto spec = block_2d(getDataSizeLSC(block.ebytes, false), w, h,
                                    block.count)
                        | astrategy.cachingR;
                if (astrategy.accessType == AccessType::Block2DTranspose)
                    spec |= transpose;
                if (astrategy.accessType == AccessType::Block2DVNNI)
                    spec |= vnni;
                load(mod, dest, spec, astrategy.base, addr);
                break;
            }
            default: stub();
        }
    else if (block.descAssigned)
        send(mod, static_cast<SharedFunction>(block.sfid), dest, addr, null,
                block.sfid, a0[0]);
    else
        switch (implAccessType(atype, astrategy, block)) {
            case AccessType::ChannelScattered: {
                static const ChannelMask cmasks[4] = {ChannelMask::r,
                        ChannelMask::rg, ChannelMask::rgb, ChannelMask::rgba};
                if (block.ebytes != 4) stub();
                load(mod, dest, surface_dword(cmasks[block.count - 1]),
                        astrategy.base, addr);
                break;
            }
            case AccessType::Scattered:
                if (block.ebytes == 8)
                    load(mod, dest, scattered_qword(block.count),
                            astrategy.base, addr);
                else if (block.ebytes == 4)
                    load(mod, dest, scattered_dword(block.count),
                            astrategy.base, addr);
                else if (block.ebytes == 1)
                    load(mod, dest, scattered_byte(block.count), astrategy.base,
                            addr);
                else
                    hw_unsupported();
                break;
            case AccessType::Block:
                if (block.ebytes == 32)
                    load(mod, dest, block_hword(block.count), astrategy.base,
                            addr);
                else if (block.ebytes == 16 && !block.extra)
                    load(mod, dest, block_oword(block.count), astrategy.base,
                            addr);
                else if (block.ebytes == 16)
                    load(mod, dest, aligned_block_oword(block.count),
                            astrategy.base, addr);
                else
                    hw_unsupported();
                if (zeroMask && (astrategy.base.getModel() == ModelBTS)) {
                    if (block.flag)
                        mov<uint32_t>(block.simdSize | ~maskMod, dest, 0);
                    if (block.simdSize <= 2) mov<uint32_t>(2, dest[2](1), 0);
                    if (block.simdSize <= 1) mov<uint32_t>(1, dest[1], 0);
                }
                break;
            default: stub();
        }
}

// Output code for storing a matrix chunk from registers.
template <HW hw>
void gemm_kernel_generator_t<hw>::storeMatrix(const GRFMultirange &src,
        const vector<RegisterBlock> &layout, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const vector<GRFRange> &addrs, const CommonStrategy &strategy,
        CommonState &state) {
    auto nblocks = int(layout.size());

    for (int l = 0; l < nblocks; l++) {
        auto offsetReg = contiguityCheck(hw, layout[l], src);
        storeMatrixBlock(src[offsetReg], layout[l], atype, astrategy, addrs[l],
                strategy, state);
    }
}

// Output code for storing a matrix block from registers.
template <HW hw>
void gemm_kernel_generator_t<hw>::storeMatrixBlock(const GRF &src,
        const RegisterBlock &block, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy, const GRFRange &addr,
        const CommonStrategy &strategy, CommonState &state) {
    InstructionModifier mod = block.simdSize;
    ;

    // Zero SIMD size blocks are filled as part of another store. Skip them.
    if (!block.isLoadBlock()) return;

    // Get mask to apply, if any.
    mod |= getRegisterBlockMask(block, state);

    // Look up preassigned token.
    for (auto &entry : state.tokenMap) {
        if (entry.first == src.getBase()) {
            mod |= SBID(entry.second);
            break;
        }
    }

    if (block.descAssigned)
        send(mod, static_cast<SharedFunction>(block.sfid), null, addr, src,
                a0.ud(1), a0.ud(0));
    else if (astrategy.newDP)
        switch (implAccessType(atype, astrategy, block)) {
            case AccessType::Block:
            case AccessType::Scattered:
            case AccessType::ChannelScattered: {
                auto spec = getDataSpecLSC(atype, astrategy, block, true);
                store(mod, spec, astrategy.base, addr[0], src);
                break;
            }
            case AccessType::Block2D:
            case AccessType::Block2DTranspose:
            case AccessType::Block2DVNNI: {
                int w = 0, h = 0;
                getBlock2DWH(w, h, atype, block);
                auto spec = block_2d(getDataSizeLSC(block.ebytes, false), w, h,
                                    block.count)
                        | astrategy.cachingW;
                if (astrategy.accessType == AccessType::Block2DTranspose)
                    spec |= transpose;
                if (astrategy.accessType == AccessType::Block2DVNNI)
                    spec |= vnni;
                store(mod, spec, astrategy.base, addr, src);
                break;
            }
            default: stub();
        }
    else
        switch (implAccessType(atype, astrategy, block)) {
            case AccessType::ChannelScattered: {
                static const ChannelMask cmasks[4] = {ChannelMask::r,
                        ChannelMask::rg, ChannelMask::rgb, ChannelMask::rgba};
                if (block.ebytes != 4) stub();
                store(mod, surface_dword(cmasks[block.count - 1]),
                        astrategy.base, addr, src);
                break;
            }
            case AccessType::Scattered:
                if (block.ebytes == 8)
                    store(mod, scattered_qword(block.count), astrategy.base,
                            addr, src);
                else if (block.ebytes == 4)
                    store(mod, scattered_dword(block.count), astrategy.base,
                            addr, src);
                else if (block.ebytes == 1)
                    store(mod, scattered_byte(block.count), astrategy.base,
                            addr, src);
                else
                    hw_unsupported();
                break;
            case AccessType::Block:
                if (block.ebytes == 32)
                    store(mod, block_hword(block.count), astrategy.base, addr,
                            src);
                else if (block.ebytes == 16 && !block.extra)
                    store(mod, block_oword(block.count), astrategy.base, addr,
                            src);
                else
                    hw_unsupported();
                break;
            default: stub();
        }
}

// Atomic addition of a matrix in registers.
template <HW hw>
void gemm_kernel_generator_t<hw>::atomicAddMatrix(Type T,
        const GRFMultirange &src, const vector<RegisterBlock> &layout,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const vector<GRFRange> &addrs, const CommonProblem &problem,
        const CommonStrategy &strategy, CommonState &state) {
    auto nblocks = int(layout.size());

    if (strategy.readSuppressionWA && (hasFlags(layout) || !getDefaultNoMask()))
        doReadSuppressionWA(strategy, state);

    for (int l = 0; l < nblocks; l++) {
        auto offsetReg = contiguityCheck(hw, layout[l], src);
        atomicAddMatrixBlock(T, src[offsetReg], layout[l], atype, astrategy,
                addrs[l], problem, strategy, state);
    }
}

template <HW hw>
void gemm_kernel_generator_t<hw>::atomicAddMatrixBlock(Type T, const GRF &src,
        const RegisterBlock &block, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy, const GRFRange &addr,
        const CommonProblem &problem, const CommonStrategy &strategy,
        CommonState &state) {
    InstructionModifier maskMod;

    if (!block.isLoadBlock()) return;
    if (block.descAssigned) stub();

    maskMod |= getRegisterBlockMask(block, state);

    // SIMD16 A64 atomics are emulated with 2x SIMD8.
    bool a64 = (astrategy.base.getModel() == ModelA64);
    int hsize = a64 ? 2 : 1;
    int simd = block.simdSize;
    if (!astrategy.newDP && a64) simd = std::min(simd, 8);
    if (hw >= HW::XeHPC && block.ebytes < 8 && block.simdSize == 16
            && simd == 8)
        stub(); // Can't split data GRFs.
    auto nreg = block.nregs();
    auto nregReal = (nreg * simd) / block.simdSize;

    auto specLSC = D32;
    if (astrategy.newDP)
        specLSC = getDataSpecLSC(atype, astrategy, block, true);

    switch (implAccessType(atype, astrategy, block)) {
        case AccessType::Scattered:
        case AccessType::ChannelScattered:
            if (hasNativeAtomicAdd(hw, T.real(), atype, astrategy)) {
                auto curSrc = src;
                for (int eoff = 0, hoff = 0; eoff < block.simdSize;
                        eoff += simd, hoff += hsize, curSrc += nregReal) {
                    auto mod = simd | maskMod | ExecutionOffset(eoff);
                    if (block.ebytes != T.real().size()) stub();
                    if (astrategy.newDP)
                        atomic(T.isFP() ? AtomicOp::fadd : AtomicOp::add, mod,
                                specLSC, astrategy.base, addr[hoff], curSrc);
                    else
                        switch (T.real()) {
                            case Type::f32:
                                atomic(AtomicOp::fadd, mod, scattered_dword(),
                                        astrategy.base, addr[hoff], curSrc);
                                break;
                            case Type::u64:
                            case Type::s64:
                                atomic(AtomicOp::add, mod, scattered_qword(),
                                        astrategy.base, addr[hoff], curSrc);
                                break;
                            case Type::u32:
                            case Type::s32:
                                atomic(AtomicOp::add, mod, scattered_dword(),
                                        astrategy.base, addr[hoff], curSrc);
                                break;
                            case Type::u16:
                            case Type::s16:
                                if (hw < HW::Gen12LP) hw_unsupported();
                                atomic(AtomicOp::add, mod, scattered_word(),
                                        astrategy.base, addr[hoff], curSrc);
                                break;
                            default: stub();
                        }
                }
            } else {
                // Emulated atomic addition with a compare-and-swap loop.
                auto rOldNew = state.eatomicAddRegs[0];
                auto rSave = state.eatomicAddRegs[1];
                auto rOld = rOldNew[0];
                auto rNew = rOldNew[nregReal];
                auto flagToDo = getPhysicalFlag(state.vflagEAtomicAdd, state);

                if (block.simdSize > 16) stub(); // Need 32 channels.
                if (astrategy.newDP)
                    load(block.simdSize | maskMod, rOld, specLSC,
                            astrategy.base, addr[0]);
                else if (astrategy.base.getModel() == ModelA64) {
                    if (block.ebytes == 2)
                        load(block.simdSize | maskMod, rOld, scattered_byte(2),
                                astrategy.base, addr);
                    else if (block.ebytes == 4)
                        load(block.simdSize | maskMod, rOld, scattered_dword(),
                                astrategy.base, addr);
                    else if (block.ebytes == 8)
                        load(block.simdSize | maskMod, rOld, scattered_qword(),
                                astrategy.base, addr);
                } else {
                    if (block.ebytes == 2)
                        load(block.simdSize | maskMod, rOld, scattered_byte(2),
                                astrategy.base, addr);
                    else if (block.ebytes == 4)
                        load(block.simdSize | maskMod, rOld,
                                surface_dword(ChannelMask::r), astrategy.base,
                                addr);
                    else if (block.ebytes == 8)
                        stub(); // needs cmpwr2
                }
                Label labelMask;

                // Save off high half of data when emulating SIMD16.
                if (block.simdSize > simd)
                    mov<uint32_t>(nregReal * 8, rOld.advance(nreg),
                            rOld.advance(nregReal));

                if (block.flag) {
                    if_(16 | getPhysicalFlag(block.flag, state), labelMask);
                    setDefaultNoMask(false);
                }

                and_(1 | NoMask, flagToDo, ce0,
                        uint16_t((1 << block.simdSize) - 1));

                auto curSrc = src;

                for (int eoff = 0, hoff = 0; eoff < block.simdSize;
                        eoff += simd, hoff += hsize) {
                    auto eoMod = ExecutionOffset(eoff);

                    Label labelCmpXchgLoop;
                    mark(labelCmpXchgLoop);

                    auto dt = T.ngen();
                    add(int(simd * block.ebytes / T.real()) | eoMod | NoMask,
                            rNew.retype(dt), rOld.retype(dt),
                            curSrc.retype(dt));
                    mov<uint32_t>((simd * block.ebytes / 4) | eoMod | NoMask,
                            rSave, rOld);

                    auto atomicMod = simd | flagToDo | eoMod;
                    auto cmpMod = simd | flagToDo | ne | flagToDo | eoMod;

                    if (astrategy.newDP)
                        atomic(AtomicOp::cmpwr, atomicMod, rOld, specLSC,
                                astrategy.base, addr[hoff], rOld);
                    else
                        switch (block.ebytes) {
                            case 2:
                                if (hw < HW::Gen12LP) hw_unsupported();
                                atomic(AtomicOp::cmpwr, atomicMod, rOld,
                                        scattered_word(), astrategy.base,
                                        addr[hoff], rOld);
                                break;
                            case 4:
                                atomic(AtomicOp::cmpwr, atomicMod, rOld,
                                        scattered_dword(), astrategy.base,
                                        addr[hoff], rOld);
                                break;
                            case 8:
                                atomic(AtomicOp::cmpwr, atomicMod, rOld,
                                        scattered_qword(), astrategy.base,
                                        addr[hoff], rOld);
                                break;
                            default: stub();
                        }

                    if (block.ebytes == 2)
                        cmp<uint16_t>(cmpMod, rSave[0][0](2), rOld[0](2));
                    else if (block.ebytes == 4)
                        cmp<uint32_t>(cmpMod, rSave, rOld);
                    else if (block.ebytes == 8) {
                        if (strategy.emulate.emulate64) {
                            cmp<uint32_t>(simd | ne | flagToDo | eoMod,
                                    rSave[0][0](2), rOld[0](2));
                            cmp<uint32_t>(
                                    simd | ~flagToDo | ne | flagToDo | eoMod,
                                    rSave[0][1](2), rOld[1](2));
                        } else
                            cmp<uint64_t>(cmpMod, rSave, rOld);
                    } else
                        stub();

                    strategy.fused ? simtDoWhileLoop(
                            16 | flagToDo | any16h, labelCmpXchgLoop)
                                   : (hw == HW::XeHPC)
                                    ? jmpi(1 | flagToDo | any, labelCmpXchgLoop)
                                    : (eoff == 0 && simd == 8)
                                            ? jmpi(1 | flagToDo | any8h,
                                                    labelCmpXchgLoop)
                                            : jmpi(1 | flagToDo | any16h,
                                                    labelCmpXchgLoop);

                    rOld += 2 * nregReal;
                    rNew += 2 * nregReal;
                    curSrc += nregReal;
                }

                if (block.flag) {
                    mark(labelMask);
                    setDefaultNoMask(true);
                    endif(16);
                }
            }
            break;
        default: hw_unsupported();
    }
}

// Allocate temporary registers for emulating atomic addition.
static inline void allocEAtomicAddRegs(HW hw, Type T,
        const vector<RegisterBlock> &layout, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy, CommonState &state,
        const FlagRegister &flag = FlagRegister()) {
    if (hasNativeAtomicAdd(hw, T.real(), atype, astrategy)) return;

    int maxNReg = 0;
    for (const auto &block : layout)
        maxNReg = std::max(maxNReg, block.nregs());

    if (maxNReg == 0) return;

    state.eatomicAddRegs[0] = state.ra.alloc_range(maxNReg * 2);
    state.eatomicAddRegs[1] = state.ra.alloc_range(maxNReg);
    state.vflagEAtomicAdd
            = flag.isValid() ? flag : state.raVFlag.allocVirtual();
}

// Free temporary registers for emulating atomic addition.
static inline void freeEAtomicAddRegs(
        CommonState &state, const FlagRegister &flag = FlagRegister()) {
    state.ra.safeRelease(state.eatomicAddRegs[0]);
    state.ra.safeRelease(state.eatomicAddRegs[1]);
    if (flag.isInvalid()) state.raVFlag.release(state.vflagEAtomicAdd);
}

static inline void releaseMaskAssignments(vector<MaskAssignment> &assignments,
        CommonState &state, int start = 0) {
    for (size_t an = start; an < assignments.size(); an++)
        state.raVFlag.release(assignments[an].flag);

    state.wipeActiveVFlags();
}

static inline void reclaimMaskAssignments(vector<MaskAssignment> &assignments,
        CommonState &state, int start = 0) {
    for (size_t an = start; an < assignments.size(); an++)
        state.raVFlag.claim(assignments[an].flag);
}

// Release all masks in a mask assignment. If 'start' is specified, only the masks
//  at index 'start' and above will be released.
static inline void safeReleaseMaskAssignments(
        vector<MaskAssignment> &assignments, CommonState &state,
        int start = 0) {
    releaseMaskAssignments(assignments, state, start);
    assignments.resize(start);
}

// Assign mask registers to a register layout.
// The assignments parameter is both input and output:
//     existing assignments will be reused if compatible, and new assignments
//     created as necessary.
template <HW hw>
bool gemm_kernel_generator_t<hw>::assignMasks(
        std::vector<RegisterBlock> &layout, LoopType rloop, LoopType cloop,
        vector<MaskAssignment> &assignments, CommonState &state) {
    auto nassignOriginal = int(assignments.size());
    bool outOfRegs = false;

    // Loop through layout, collecting masks.
    //  - For each unique mask+loop+offset, allocate an index (flag reg)
    //  - Store new assignment if unique and update flag reg in layout.
    //  - For now, simultaneous row and column masks are not supported.
    for (RegisterBlock &l : layout) {
        MaskAssignment thisAssignment;

        if (l.rowMask) {
            if (l.colMask) stub();

            thisAssignment.mask = l.rowMask;
            thisAssignment.offset = l.offsetR;
            thisAssignment.var = rloop;
        } else if (l.colMask) {
            thisAssignment.mask = l.colMask;
            thisAssignment.offset = l.offsetC;
            thisAssignment.var = cloop;
        } else {
            l.clearFlag();
            continue;
        }

        // Look for compatible mask.
        bool gotMask = false;
        for (auto &a : assignments) {
            if (a.compatible(thisAssignment)) {
                l.flag = a.flag;
                gotMask = true;
                break;
            }
        }

        if (!gotMask) {
            // No compatible mask, so make a new assignment.
            thisAssignment.flag
                    = state.raVFlag.allocVirtual((l.simdSize + 0xF) >> 4);
            assignments.push_back(thisAssignment);
            if (state.raVFlag.isVirtual(thisAssignment.flag)
                    && state.vflagStorage.isInvalid()) {
                outOfRegs = true;
                break;
            }
            l.flag = thisAssignment.flag;
        }
    }

    if (outOfRegs) {
        // Not enough (virtual) flag registers! Free any masks we added to the list.
        safeReleaseMaskAssignments(assignments, state, nassignOriginal);
        status << "Not enough flag registers available." << status_stream::endl;
        return false;
    }

    return true;
}

// Output code for loading a mask into a flag register.
template <HW hw>
void gemm_kernel_generator_t<hw>::loadMask(MaskAssignment assignment,
        Subregister index, const CommonStrategy &strategy, CommonState &state,
        int offset) {
    auto flagIdx = assignment.flag;
    RegData flag = getMaskFlag(flagIdx, state);

    if (assignment.mask.fixed.isFixed) {
        // Load fixed mask. Easy.
        mov(1, flag, uint16_t(assignment.mask.fixed.value));
    } else {
        // Load a variable mask, which requires some minor bit-twiddling.
        auto &vmask = assignment.mask.variable;

        uint32_t rsizeScaled = vmask.rsize / vmask.rdivide;
        uint32_t fullMask
                = (1ul << (vmask.bitRep * vmask.maskRep * rsizeScaled)) - 1;
        uint32_t rep1Mask = (1ul << (vmask.bitRep * rsizeScaled)) - 1;
        uint32_t repMultiplier = fullMask / rep1Mask;

        auto flagType = flag.getType();
        auto mask0Type = getBytes(flagType) >= 4 ? DataType::uq : flagType;

        auto temp = state.ra.alloc_sub(flagType, getHint(HintType::Bank0));
        auto mask0 = state.ra.alloc_sub(mask0Type, getHint(HintType::Bank1));
        auto mask = mask0.reinterpret(0, flagType);
        auto mindex = index;

        if (vmask.rdivide > 1) {
            if (!is_zero_or_pow2(vmask.rdivide)) stub();
            add(1, temp, mindex, -offset + vmask.rdivide - 1);
            shr(1, temp, temp, uint16_t(log2(vmask.rdivide)));
            mindex = temp;
            offset = 0;
        }
        if (vmask.bitRep > 1) {
            if (offset > 0) {
                add(1, temp, mindex, -offset);
                mindex = temp;
                offset = 0;
            }
            mulConstant(1, temp, mindex, vmask.bitRep);
            mindex = temp;
        }
        uint16_t tshift = vmask.bitRep
                * (rsizeScaled
                        + div_up(assignment.offset + offset, vmask.rdivide));
        add(1 | sat, temp, -mindex, tshift);
        if (tshift >= 32)
            min_(1, temp, temp,
                    vmask.bitRep
                            * rsizeScaled); // Ensure shift count doesn't overflow.
        emov(1, mask0, rep1Mask, strategy, state);
        if (vmask.maskRep == 1) {
            bool twoStage = (!flag.isARF() && getBytes(mask0Type) > 4);
            auto flag1 = twoStage ? mask0 : flag;
            vmask.reverse ? shl(1, flag1, mask0, temp)
                          : shr(1, flag1, mask0, temp);
            if (twoStage) mov(1, flag, mask);
        } else {
            vmask.reverse ? stub() // need shl + and
                          : shr(1, mask0, mask0, temp);
            if (repMultiplier & 0x10000) mov(1, mask.uw(1), mask.uw(0));
            mul(1, flag, mask, uint16_t(repMultiplier));
        }

        state.ra.safeRelease(temp);
        state.ra.safeRelease(mask0);
    }
}

// Output code for loading all masks in a mask assignment to flag registers.
template <HW hw>
void gemm_kernel_generator_t<hw>::loadMasks(
        const vector<MaskAssignment> &assignments, Subregister (&indices)[3],
        const CommonStrategy &strategy, CommonState &state, int start) {
    for (size_t an = start; an < assignments.size(); an++) {
        auto &a = assignments[an];
        auto av = static_cast<int>(a.var);
        loadMask(a, indices[av], strategy, state);
    }
}

template <HW hw>
void gemm_kernel_generator_t<hw>::loadMasks(
        const vector<MaskAssignment> &assignments, Subregister (&indices)[3],
        int (&offsets)[3], const CommonStrategy &strategy, CommonState &state,
        int start) {
    for (size_t an = start; an < assignments.size(); an++) {
        auto &a = assignments[an];
        auto av = static_cast<int>(a.var);
        loadMask(a, indices[av], strategy, state, offsets[av]);
    }
}

template <HW hw>
void gemm_kernel_generator_t<hw>::extendIndexVec(int n, CommonState &state) {
    auto &indexVec = state.indexVec;
    auto &ivEntries = state.ivEntries;

    if (n > ivEntries) {
        int simd = GRF::bytes(hw) >> 1;
        int nregs = div_up(n, simd);
        int cregs = indexVec.getLen();
        if (nregs > cregs)
            indexVec.ranges.push_back(state.ra.alloc_range(nregs - cregs));
        if (ivEntries == 0) {
            mov<uint16_t>(8, indexVec[0][0](1),
                    Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
            ivEntries = 8;
        }
        if (n > 8 && ivEntries < 16) {
            mov<uint16_t>(8, indexVec[0][8](1),
                    Immediate::uv(8, 9, 10, 11, 12, 13, 14, 15));
            ivEntries = 16;
        }
        if (GRF::bytes(hw) > 32 && n > 16 && ivEntries < 32) {
            add<uint16_t>(16, indexVec[0][16](1), indexVec[0].uw(0)(1), 16);
            ivEntries = 32;
        }
        if (n > ivEntries) {
            for (int e = std::max(cregs, 1); e < nregs; e++)
                add<uint16_t>(simd, indexVec[e], indexVec[0], simd * e);
            ivEntries = nregs * simd;
        }
    }
}

template <HW hw>
Subregister gemm_kernel_generator_t<hw>::accessIndexVec(
        int n, CommonState &state) {
    if (n >= state.ivEntries) extendIndexVec(n, state);

    int simd = GRF::bytes(hw) >> 1;
    return state.indexVec[n / simd].uw(n % simd);
}

static inline void releaseIndexVec(CommonState &state) {
    safeReleaseRanges(state.indexVec, state);
    state.ivEntries = 0;
}

template <HW hw>
LDMultiples gemm_kernel_generator_t<hw>::createLDMultiples(bool a64,
        int nmultiples, const Subregister &ld, const CommonStrategy &strategy,
        CommonState &state) {
    int simd = GRF::bytes(hw) >> (a64 ? 3 : 2);
    int nregs = div_up(nmultiples, simd);
    auto r = state.ra.try_alloc_range(nregs);

    GRF tempHi = state.emulate.temp[0], tempLo = state.emulate.temp[1];
    bool freeTempHi = false, freeTempLo = false;
    if (a64) {
        if (tempHi.isInvalid()) {
            tempHi = state.ra.alloc();
            freeTempHi = true;
        }
        if (tempLo.isInvalid()) {
            tempLo = state.ra.alloc();
            freeTempLo = true;
        }
    }

    if (r.isValid()) {
        extendIndexVec(nmultiples, state);
        for (int i = 0; i < nregs; i += 2) {
            auto thisSIMD = simd * std::min(nregs - i, 2);
            auto iv = accessIndexVec(simd * i, state)(1);
            if (a64) {
                mul<uint32_t>(thisSIMD, acc0, ld, iv);
                mach<uint32_t>(thisSIMD, tempHi, ld, Immediate::ud(0));
                mov<uint32_t>(thisSIMD, tempLo, acc0);
                mov<uint32_t>(thisSIMD, r[i][1](2), tempHi);
                mov<uint32_t>(thisSIMD, r[i][0](2), tempLo);
            } else
                mul<uint32_t>(thisSIMD, r[i], ld, iv);
        }
    }

    if (freeTempHi) state.ra.safeRelease(tempHi);
    if (freeTempLo) state.ra.safeRelease(tempLo);

    LDMultiples result;
    result.range = r;
    result.a64 = a64;

    return result;
}

template <HW hw>
Subregister gemm_kernel_generator_t<hw>::findLDMultiple(
        const LDMultiples &multiples, bool a64, int n,
        const CommonStrategy &strategy, CommonState &state) {
    int simd = GRF::bytes(hw) >> (multiples.a64 ? 3 : 2);
    int off = (n / simd), sub = (n % simd);

    if (multiples.range.isInvalid()) return Subregister();
    if (off < 0 || off >= multiples.range.getLen()) return Subregister();
    if (a64 && !multiples.a64) return Subregister();

    return !multiples.a64 ? multiples.range[off].ud(sub)
                          : a64 ? multiples.range[off].uq(sub)
                                : multiples.range[off].ud(2 * sub);
}

static inline void releaseLDMultiples(
        LDMultiples &multiples, CommonState &state) {
    state.ra.safeRelease(multiples.range);
    multiples.a64 = false;
}

// Ugly helpers handling address shifts. constexpr if would clean this all up.
template <HW hw>
template <typename BO>
typename std::enable_if<!std::is_base_of<RegData, BO>::value, BO>::type
gemm_kernel_generator_t<hw>::startShift(
        const BO &ptr, int shift, CommonState &state) {
    return ptr >> shift;
}

template <HW hw>
Subregister gemm_kernel_generator_t<hw>::startShift(
        const MultishiftSubregister &ptr, int shift, CommonState &state) {
    return ptr >> shift;
}

template <HW hw>
SubregisterPair gemm_kernel_generator_t<hw>::startShift(
        const SubregisterPair &ptr, int shift, CommonState &state) {
    if (shift == 0)
        return ptr;
    else
        return SubregisterPair(startShift(ptr.getReg(0), shift, state));
}

template <HW hw>
template <typename BO>
typename std::enable_if<std::is_base_of<RegData, BO>::value, BO>::type
gemm_kernel_generator_t<hw>::startShift(
        const BO &ptr, int shift, CommonState &state) {
    BO ptrShifted = ptr;

    // Shift pointer as necessary.
    if (shift > 0) {
        ptrShifted = state.ra.alloc_sub(ptr.getType());
        shr(1, ptrShifted, ptr, shift);
    }

    return ptrShifted;
}

template <HW hw>
template <typename BO, typename BI>
typename std::enable_if<!std::is_base_of<RegData, BO>::value>::type
gemm_kernel_generator_t<hw>::doneShift(
        const BO &ptr, const BI &ptrShifted, int shift, CommonState &state) {}

template <HW hw>
template <typename BO, typename BI>
typename std::enable_if<std::is_base_of<RegData, BO>::value>::type
gemm_kernel_generator_t<hw>::doneShift(
        const BO &ptr, const BI &ptrShifted, int shift, CommonState &state) {
    if (shift > 0) state.ra.release(ptrShifted);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::doneShift(const SubregisterPair &ptr,
        const SubregisterPair &ptrShifted, int shift, CommonState &state) {
    if (shift > 0) doneShift(ptr.getReg(0), ptrShifted.getReg(0), shift, state);
}

static inline bool canIncAddr(const RegisterBlock &blockSrc,
        const RegisterBlock &blockDst, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    if (!blockSrc.isLoadBlock() || !blockDst.isLoadBlock()) return false;
    if (effectiveAccessType(atype, astrategy, blockDst) == AccessType::Block
            && effectiveAccessType(atype, astrategy, blockSrc)
                    == AccessType::Block)
        return true;
    if (isBlock2D(astrategy.accessType))
        return (blockSrc.nr == blockDst.nr && blockSrc.nc == blockDst.nc);
    return (blockSrc.simdSize >= blockDst.simdSize);
}

// Output code for setting up address/header GRFs for a single block, given
//  the base pointer (a Subregister, MultishiftSubregister or integer) and leading dimension.
template <HW hw>
template <typename BO>
void gemm_kernel_generator_t<hw>::setupAddr(const GRFRange &addr, const BO &ptr,
        const RegisterBlock &block, const Subregister &bld, size_t sizeofT,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const CommonStrategy &strategy, CommonState &state,
        const Address2DParams &params, LDMultiples ldMultiples) {
    bool a64 = astrategy.base.getModel() == ModelA64;

    // Nothing to do for non-load blocks.
    if (!block.isLoadBlock()) return;

    auto effAccessType = effectiveAccessType(atype, astrategy, block);
    switch (effAccessType) {
        case AccessType::Scattered:
        case AccessType::ChannelScattered:
        case AccessType::PseudoBlock: {
            int simdSize = block.simdSize;
            auto consecutive = block.extra;
            bool pseudo = (effAccessType == AccessType::PseudoBlock);
            auto Tptr = a64 ? DataType::uq : DataType::ud;
            int ne = elementsPerGRF(hw, Tptr);
            int preshift = 0;

            auto oldIndexVec = state.indexVec;
            auto oldIVEntries = state.ivEntries;

            if (!pseudo && !isPacked(atype.layout)) {
                // Get pointers to successive rows/columns, strided by ld.
                bool allocLDMultiples = false;
                if (ldMultiples.range.isInvalid()) {
                    ldMultiples = createLDMultiples(
                            a64, simdSize, bld, strategy, state);
                    allocLDMultiples = true;
                } else
                    (void)findLDMultiple(
                            ldMultiples, a64, simdSize - 1, strategy, state);

                for (int r = 0; r < addr.getLen(); r += 2) {
                    int nr = std::min(2, addr.getLen() - r);
                    int simd = nr * ne;
                    auto ld0 = findLDMultiple(ldMultiples, a64,
                            r * ne / consecutive, strategy, state);
                    auto ldStride = (ldMultiples.a64 && !a64) ? 2 : 1;
                    auto ldR = ld0(ldStride, consecutive, 0);
                    auto addrR = addr[r].retype(Tptr);
                    if (a64 && consecutive > 1 && hw >= HW::XeHP
                            && !strategy.emulate
                                        .emulate64) { /* no swizzle in L pipe */
                        mov(simd, addr[r].ud(0)(2),
                                ld0.ud(0)(ldStride * 2, consecutive, 0));
                        mov(simd, addr[r].ud(1)(2),
                                ld0.ud(1)(ldStride * 2, consecutive, 0));
                        if (ptr != 0) add(simd, addrR, addrR, ptr);
                    } else if (ptr != 0)
                        eadd(simd, addrR, ptr, ldR, strategy, state);
                    else
                        emov(simd, addrR, ldR, strategy, state);
                }
                if (allocLDMultiples) releaseLDMultiples(ldMultiples, state);
            } else {
                // Get pointers to successive elements, with constant stride.
                extendIndexVec(simdSize, state);
                auto iv = accessIndexVec(0, state)(1, consecutive, 0);
                uint16_t stride;
                preshift = block.addrShift;
                auto ptrShifted = startShift(ptr, block.addrShift, state);

                if (pseudo)
                    stride = (block.ebytes * block.count
                                     * getPartialCrosspack(
                                             sizeofT, atype, block))
                            >> preshift;
                else {
                    int psElems = (isLargeCrosspack(sizeofT, atype.crosspack)
                                                  ? 1
                                                  : atype.packSize)
                            * atype.crosspack;
                    stride = uint16_t(psElems * sizeofT) >> preshift;
                }

                if (a64) {
                    int udStride = (hw >= HW::XeHP) ? 2 : 1;
                    int simd1 = std::min(2 * ne, simdSize);
                    int simd2 = simdSize - simd1;
                    if (udStride == 2 && simd2) {
                        auto iv2 = accessIndexVec(simd1 / consecutive, state)(
                                1, consecutive, 0);
                        mulConstant(
                                simd2, addr[2].ud(0)(udStride), iv2, stride);
                        mulConstant(simd1, addr[0].ud(0)(udStride), iv, stride);
                    } else
                        mulConstant(
                                simdSize, addr[0].ud(0)(udStride), iv, stride);
                    if (simd2)
                        eadd(simd2, addr[2].uq(), ptrShifted,
                                addr[udStride].ud(0)(udStride), strategy,
                                state);
                    eadd(simd1, addr[0].uq(), ptrShifted,
                            addr[0].ud(0)(udStride), strategy, state);
                } else if (ptrShifted != 0) {
                    if (consecutive > 1) {
                        mulConstant<uint32_t>(simdSize, addr, iv, stride);
                        add<uint32_t>(simdSize, addr, addr, ptrShifted);
                    } else
                        emad(simdSize, addr[0].ud(), ptrShifted, iv,
                                int32_t(stride), strategy, state);
                } else
                    mulConstant<uint32_t>(simdSize, addr, iv, stride);

                doneShift(ptr, ptrShifted, block.addrShift, state);
            }

            // Add offsets for consecutive elements in scattered accesses.
            if (consecutive > 1) {
                if ((consecutive - 1) * block.ebytes >= 0x10) stub();
                if (consecutive > 4) stub();
                uint8_t incs[4];
                for (int idx = 0; idx < 4; idx++)
                    incs[idx]
                            = (block.ebytes * (idx % consecutive)) >> preshift;

                if (!a64) {
                    auto incImm = Immediate::uv(
                            incs[0], 0, incs[1], 0, incs[2], 0, incs[3], 0);
                    add<uint32_t>(simdSize, addr, addr, incImm);
                } else {
                    if (consecutive > 2) stub();
                    auto incImm
                            = Immediate::uv(incs[0], 0, 0, 0, incs[1], 0, 0, 0);
                    auto temp = state.ra.alloc_range(2);
                    mov<uint32_t>(
                            2 * elementsPerGRF<uint32_t>(hw), temp, incImm);
                    map(hw, Tptr, addr, addr, strategy,
                            [&](int simd, GRF r1, GRF _) {
                                eadd<uint64_t>(simd, r1, r1, temp[0].ud(0)(2),
                                        strategy, state);
                            });
                    state.ra.safeRelease(temp);
                }
            }

            // Scale if needed.
            if (block.addrShift > preshift)
                shr<uint32_t>(simdSize, addr, addr, block.addrShift - preshift);

            // Restore original cached index vector in case we extended it.
            releaseRanges(state.indexVec, state);
            state.indexVec = oldIndexVec;
            state.ivEntries = oldIVEntries;
            reclaimRanges(state.indexVec, state);
            break;
        }
        case AccessType::Block:
            if (astrategy.base.getModel() == ModelA64) {
                emov(1, addr[0].uq(0), ptr, strategy, state);
                // Disable OWord channel mode on SKL.
                if (block.ebytes == 32 && hw < HW::Gen10)
                    mov(1, addr[0].ud(5), uint32_t(0x80000000));
            } else if (astrategy.newDP) {
                mov(1, addr[0].ud(0), ptr);
            } else if (block.addrShift > 0)
                shr(1, addr[0].ud(2), ptr, block.addrShift);
            else
                mov(1, addr[0].ud(2), ptr);
            break;
        case AccessType::Block2D:
        case AccessType::Block2DTranspose:
        case AccessType::Block2DVNNI:
            if (astrategy.base.getModel() != ModelA64) hw_unsupported();
            int bw, bh, multiX;
            bool memCM = isColMajor(atype.layout);
            auto iremR = params.remR, iremC = params.remC;
            if (!block.remainderR) iremR.invalidate();
            if (!block.remainderC) iremC.invalidate();
            auto remW = memCM ? iremR : iremC;
            auto remH = memCM ? iremC : iremR;
            getBlock2DWH(bw, bh, atype, block, &multiX);
            if (!astrategy.address2D) mov(4, addr[0].ud(4)(1), 0u);
            emov(1, addr[0].uq(0), ptr, strategy, state);
            if (astrategy.address2D) {
                if (params.rows.isInvalid() && params.fixedRows == 0)
                    throw std::runtime_error("Unknown matrix size.");
                auto &nx = memCM ? params.rows : params.cols;
                auto &ny = memCM ? params.cols : params.rows;
                auto fixedX = memCM ? params.fixedRows : params.fixedCols;
                auto fixedY = memCM ? params.fixedCols : params.fixedRows;
                auto &offX = memCM ? params.offR : params.offC;
                auto &offY = memCM ? params.offC : params.offR;
                auto boffX = memCM ? block.offsetR : block.offsetC;
                auto boffY = memCM ? block.offsetC : block.offsetR;

                boffX *= uint8_t(sizeofT);
                if (boffX % block.ebytes) stub();
                boffX /= block.ebytes;

                nx.isValid() ? mad(1, addr[0].ud(2), -1, nx, sizeofT)
                             : mov(1, addr[0].ud(2), fixedX * sizeofT - 1);
                ny.isValid() ? add(1, addr[0].ud(3), ny, -1)
                             : mov(1, addr[0].ud(3), fixedY - 1);
                offX.isValid() ? addScaled(1, addr[0].ud(5), boffX, offX,
                        int(sizeofT), block.ebytes, state)
                               : mov(1, addr[0].ud(5), boffX);
                offY.isValid() ? add(1, addr[0].ud(6), offY, boffY)
                               : mov(1, addr[0].ud(6), boffY);
                if (sizeofT < 4)
                    or_(1, addr[0].ud(2), addr[0].ud(2),
                            3); // Width must be 4-byte-aligned.
            } else if (remW.isInvalid() && remH.isInvalid())
                emov(1, addr[0].uq(1),
                        uint64_t(bw * block.count * block.ebytes - 1)
                                | (uint64_t(bh * block.ebytes - 1) << 32),
                        strategy, state);
            else {
                if (remW.isValid() && multiX > 1) stub();
                remW.isValid() ? mad(1, addr[0].ud(2), -1, remW.uw(), sizeofT)
                               : mov(1, addr[0].ud(2),
                                       bw * block.count * block.ebytes - 1);
                remH.isValid() ? mad(1, addr[0].ud(3), -1, remH.uw(), multiX)
                               : mov(1, addr[0].ud(3), bh - 1);
                if (remW.isValid() && sizeofT < 4)
                    or_(1, addr[0].ud(2), addr[0].ud(2), 3);
            }
            if (isPacked(atype.layout)) {
                auto pitch = bw * block.count * block.ebytes;
                if (pitch < 64 || pitch & 0xF) hw_unsupported();
                mov(1, addr[0].ud(4), pitch - 1);
            } else
                add(1, addr[0].ud(4), bld, -1);
            mov(1, addr[0].ud(7),
                    (bw - 1) | ((bh - 1) << 8) | ((block.count - 1) << 16));
            break;
    }
}

// Shift an address block by a combination of a fixed and LD offset.
template <HW hw>
void gemm_kernel_generator_t<hw>::offsetAddr(const GRFRange &addrDst,
        const GRFRange &addrSrc, const RegisterBlock &blockDst,
        const RegisterBlock &blockSrc, int offsetFixed, int offsetLD,
        const Subregister &ld, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const CommonStrategy &strategy, CommonState &state,
        const LDMultiples &ldMultiples) {
    bool a64 = (astrategy.base.getModel() == ModelA64);

    if (astrategy.address2D) stub();

    if (offsetLD == 0) {
        if (offsetFixed != 0)
            incAddr(addrDst, addrSrc, offsetFixed, blockDst, blockSrc, atype,
                    astrategy, strategy, state);
    } else {
        // Reuse ld * offsetLD calculation if available.
        auto ldInc
                = findLDMultiple(ldMultiples, a64, offsetLD, strategy, state);

        if (ldInc.isValid() && offsetFixed == 0)
            incAddr(addrDst, addrSrc, (offsetLD == 1) ? ld : ldInc, blockDst,
                    blockSrc, atype, astrategy, strategy, state);
        else {
            Subregister incAlloc
                    = state.ra.alloc_sub(a64 ? DataType::uq : DataType::ud);
            auto inc = incAlloc;

            if (ldInc.isInvalid()) {
                if (offsetLD == 1)
                    ldInc = ld;
                else {
                    emulConstant(1, inc, ld, offsetLD, strategy, state);
                    ldInc = inc;
                }
            }
            if (offsetFixed != 0)
                eadd(1, inc, ldInc, offsetFixed, strategy, state);
            else
                inc = ldInc;
            incAddr(addrDst, addrSrc, inc, blockDst, blockSrc, atype, astrategy,
                    strategy, state);

            state.ra.safeRelease(incAlloc);
        }
    }
}

// Output code for initializing address/header GRFs for one block based on another block's headers.
template <HW hw>
void gemm_kernel_generator_t<hw>::setupAddrRel(const GRFRange &addrDst,
        const GRFRange &addrSrc, const RegisterBlock &blockDst,
        const RegisterBlock &blockSrc, const vector<RegisterBlock> &layout,
        size_t sizeofT, const Subregister &ld, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const CommonStrategy &strategy, CommonState &state,
        const LDMultiples &ldMultiples) {
    int deltaR = blockDst.offsetR - blockSrc.offsetR;
    int deltaC = blockDst.offsetC - blockSrc.offsetC;

    if (astrategy.address2D)
        incAddr(addrDst, addrSrc, Subregister(), deltaR, deltaC, blockDst,
                blockSrc, atype, astrategy, strategy, state);
    else {
        int offsetFixed = 0, offsetLD = 0, r = 0, c = 0;

        if (isPacked(atype.layout)) getLayoutDims(layout, r, c);

        switch (atype.layout) {
            case MatrixLayout::N:
                offsetFixed = deltaR;
                offsetLD = deltaC;
                break;
            case MatrixLayout::T:
                offsetFixed = deltaC;
                offsetLD = deltaR;
                break;
            case MatrixLayout::Pc:
            case MatrixLayout::Pr:
                offsetFixed = untile(atype, blockDst, r, c)
                        - untile(atype, blockSrc, r, c);
                break;
        }

        offsetFixed *= int(sizeofT);

        offsetAddr(addrDst, addrSrc, blockDst, blockSrc, offsetFixed, offsetLD,
                ld, atype, astrategy, strategy, state, ldMultiples);
    }
}

static inline int findBaseBlock(const RegisterBlock &block,
        const vector<RegisterBlock> &layout, int start, int end,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    int bbase = -1;
    for (int bb = start; bb < end; bb++) {
        auto &other = layout[bb];
        if (canIncAddr(other, block, atype, astrategy)) {
            if (bbase < 0) bbase = bb;
            if (other.offsetR == block.offsetR
                    || other.offsetC == block.offsetC)
                return bb; // "Best fit"
        }
    }
    return bbase;
}

// Output code for initializing address/header GRFs for an entire register layout.
//  ptr is an integer, Subregister, or MultishiftSubregister holding the base pointer/offset.
template <HW hw>
template <typename BO>
void gemm_kernel_generator_t<hw>::setupAddr(Type T,
        const vector<GRFRange> &addr, const BO &ptr,
        const vector<RegisterBlock> &layout, const Subregister &ld,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const CommonStrategy &strategy, CommonState &state,
        const Address2DParams &params, const LDMultiples &ldMultiples) {
    auto nblocks = int(layout.size());

    for (int b = 0; b < nblocks; b++) {
        auto &block = layout[b];

        // Skip non-load blocks.
        if (!block.isLoadBlock()) continue;

        auto bparams = params;
        Subregister tempRem;
        if (isBlock2D(astrategy.accessType) && !astrategy.address2D) {
            tempRem = state.ra.alloc_sub<uint32_t>();
            if (bparams.remR.isValid()) bparams.remR = tempRem.uw(0);
            if (bparams.remC.isValid()) bparams.remC = tempRem.uw(1);
            if (bparams.remR.isValid() && block.offsetR)
                add(1 | sat, bparams.remR, params.remR, -block.offsetR);
            if (bparams.remC.isValid() && block.offsetC)
                add(1 | sat, bparams.remC, params.remC, -block.offsetC);
            if (bparams.remR.isValid())
                min_(1, bparams.remR,
                        block.offsetR ? bparams.remR : params.remR, block.nr);
            if (bparams.remC.isValid())
                min_(1, bparams.remC,
                        block.offsetC ? bparams.remC : params.remC, block.nc);
        }
        // Look for a block to base this one off of.
        int bbase = findBaseBlock(block, layout, 0, b, atype, astrategy);

        if (bbase < 0) {
            // No base address, set up a new base address.
            setupAddr(addr[b], ptr, block, ld, T.size(), atype, astrategy,
                    strategy, state, bparams, ldMultiples);
            state.ra.safeRelease(tempRem);
        }

        // Increment as appropriate.
        if (bbase >= 0) {
            setupAddrRel(addr[b], addr[bbase], block, layout[bbase], layout,
                    T.size(), ld, atype, astrategy, strategy, state,
                    ldMultiples);
        } else if (!astrategy.address2D) {
            int offsetFixed = 0, offsetLD = 0, r = 0, c = 0;
            if (isPacked(atype.layout)) getLayoutDims(layout, r, c);
            switch (atype.layout) {
                case MatrixLayout::N:
                    offsetFixed = block.offsetR;
                    offsetLD = block.offsetC;
                    break;
                case MatrixLayout::T:
                    offsetFixed = block.offsetC;
                    offsetLD = block.offsetR;
                    break;
                case MatrixLayout::Pc:
                case MatrixLayout::Pr:
                    offsetFixed = untile(atype, block, r, c);
                    break;
            }

            offsetFixed *= T.size();

            offsetAddr(addr[b], addr[b], block, block, offsetFixed, offsetLD,
                    ld, atype, astrategy, strategy, state, ldMultiples);
        }
    }
}

// Output code for incrementing the pointers for a given block by a specified # of bytes.
// The amount may be an immediate, Subregister, or MultishiftSubregister.
template <HW hw>
template <typename I, typename Ir, typename Ic>
void gemm_kernel_generator_t<hw>::incAddr(const GRFRange &addrDst,
        const GRFRange &addrSrc, I inc, Ir incR, Ic incC,
        const RegisterBlock &layoutDst, const RegisterBlock &layoutSrc,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const CommonStrategy &strategy, CommonState &state) {
    auto incShifted = startShift(inc, layoutDst.addrShift, state);

    incAddrShifted(addrDst, addrSrc, incShifted, incR, incC, layoutDst,
            layoutSrc, atype, astrategy, strategy, state);

    doneShift(inc, incShifted, layoutDst.addrShift, state);
}

template <HW hw>
template <typename I>
void gemm_kernel_generator_t<hw>::incAddr(const GRFRange &addrDst,
        const GRFRange &addrSrc, I inc, const RegisterBlock &layoutDst,
        const RegisterBlock &layoutSrc, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const CommonStrategy &strategy, CommonState &state) {
    if (astrategy.address2D) stub();
    incAddr(addrDst, addrSrc, inc, Subregister(), Subregister(), layoutDst,
            layoutSrc, atype, astrategy, strategy, state);
}

template <HW hw>
template <typename I, typename Ir, typename Ic>
void gemm_kernel_generator_t<hw>::incAddrShifted(const GRFRange &addrDst,
        const GRFRange &addrSrc, I inc, Ir incR, Ic incC,
        const RegisterBlock &layoutDst, const RegisterBlock &layoutSrc,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const CommonStrategy &strategy, CommonState &state) {
    // Handle non-load blocks.
    if (!layoutDst.isLoadBlock()) return;
    if (!layoutSrc.isLoadBlock()) stub();

    if (layoutDst.addrShift != layoutSrc.addrShift) stub();

    auto cinc = avoidConflict(hw, inc, addrSrc[0]);
    auto cincR = avoidConflict(hw, incR, addrSrc[0]);
    auto cincC = avoidConflict(hw, incC, addrSrc[0]);

    switch (effectiveAccessType(atype, astrategy, layoutSrc)) {
        case AccessType::PseudoBlock:
            if (layoutSrc.ebytes != layoutDst.ebytes) stub();
            // fall through
        case AccessType::ChannelScattered:
        case AccessType::Scattered: {
            int naddrDst = layoutDst.simdSize;
            int naddrSrc = layoutSrc.simdSize;
            if (naddrDst > naddrSrc) stub();
            if (astrategy.base.getModel() == ModelA64) {
                auto simd = 2 * elementsPerGRF(hw, Type::u64);
                for (int ar = 0; naddrDst > 0; ar += 2, naddrDst -= simd)
                    eadd<uint64_t>(std::min(naddrDst, simd), addrDst[ar],
                            addrSrc[ar], avoidConflict(hw, inc, addrSrc[ar]),
                            strategy, state);
            } else
                add<uint32_t>(naddrDst, addrDst[0], addrSrc[0], cinc);
            break;
        }
        case AccessType::Block:
            if (astrategy.base.getModel() == ModelA64) {
                eadd(1, addrDst[0].uq(0), addrSrc[0].uq(0), cinc, strategy,
                        state);
                if (addrDst != addrSrc && layoutDst.ebytes == 32
                        && hw < HW::Gen10)
                    mov(1, addrDst[0].ud(5),
                            uint32_t(
                                    0x80000000)); // Disable OWord channel mode on SKL.
            } else if (astrategy.newDP) {
                add(1, addrDst[0].ud(0), addrSrc[0].ud(0), cinc);
            } else
                add(1, addrDst[0].ud(2), addrSrc[0].ud(2), cinc);
            break;
        case AccessType::Block2D:
        case AccessType::Block2DTranspose:
        case AccessType::Block2DVNNI:
            if (addrDst != addrSrc) mov<uint32_t>(8, addrDst[0], addrSrc[0]);
            if (astrategy.address2D) {
                if (isColMajor(atype.layout)) {
                    if (cincR != 0)
                        addScaled(1, addrDst[0].d(5), addrDst[0].d(5), cincR,
                                layoutDst.extra, layoutDst.ebytes, state, true);
                    if (cincC != 0)
                        add(1, addrDst[0].d(6), addrDst[0].d(6), cincC);
                } else {
                    if (cincC != 0)
                        addScaled(1, addrDst[0].d(5), addrDst[0].d(5), cincC,
                                layoutDst.extra, layoutDst.ebytes, state, true);
                    if (cincR != 0)
                        add(1, addrDst[0].d(6), addrDst[0].d(6), cincR);
                }
            } else
                eadd(1, addrDst[0].uq(0), addrSrc[0].uq(0), cinc, strategy,
                        state);
            break;
    }
}

// Output code for incrementing all pointers for a register layout by a specified # of bytes.
// The amount may be an immediate or a subregister.
template <HW hw>
template <typename I, typename Ir, typename Ic>
void gemm_kernel_generator_t<hw>::incAddr(const vector<GRFRange> &addr, I inc,
        Ir incR, Ic incC, const vector<RegisterBlock> &layout,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const CommonStrategy &strategy, CommonState &state) {
    auto nblocks = int(layout.size());

    for (int b = 0; b < nblocks; b++)
        incAddr(addr[b], addr[b], inc, incR, incC, layout[b], layout[b], atype,
                astrategy, strategy, state);
}

template <HW hw>
template <typename I>
void gemm_kernel_generator_t<hw>::incAddr(const vector<GRFRange> &addr, I inc,
        const vector<RegisterBlock> &layout, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const CommonStrategy &strategy, CommonState &state) {
    if (astrategy.address2D) stub();
    incAddr(addr, inc, Subregister(), Subregister(), layout, atype, astrategy,
            strategy, state);
}

template <HW hw>
template <typename I, typename Ir, typename Ic>
void gemm_kernel_generator_t<hw>::incAddrShifted(const vector<GRFRange> &addr,
        I inc, Ir incR, Ic incC, const vector<RegisterBlock> &layout,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const CommonStrategy &strategy, CommonState &state) {
    auto nblocks = int(layout.size());

    for (int b = 0; b < nblocks; b++)
        incAddrShifted(addr[b], addr[b], inc, incR, incC, layout[b], layout[b],
                atype, astrategy, strategy, state);
}

template <HW hw>
template <typename I>
void gemm_kernel_generator_t<hw>::incAddrShifted(const vector<GRFRange> &addr,
        I inc, const vector<RegisterBlock> &layout,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const CommonStrategy &strategy, CommonState &state) {
    if (astrategy.address2D) stub();
    incAddrShifted(addr, inc, Subregister(), Subregister(), layout, atype,
            astrategy, strategy, state);
}

template <typename T>
struct NegativeType {
    typedef T type;
};
template <>
struct NegativeType<uint8_t> {
    typedef int8_t type;
};
template <>
struct NegativeType<uint16_t> {
    typedef int16_t type;
};
template <>
struct NegativeType<uint32_t> {
    typedef int32_t type;
};
template <>
struct NegativeType<int> {
    typedef int32_t type;
};
template <>
struct NegativeType<int64_t> {
    typedef int32_t type;
};

// Output code for incrementing or decrementing all pointers for a register layout by a specified # of bytes.
// The amount may be an immediate or a MultishiftSubregister.
template <HW hw>
template <typename A, typename I, typename Ir, typename Ic>
void gemm_kernel_generator_t<hw>::incDecAddr(const A &addr, I inc, Ir incR,
        Ic incC, const vector<RegisterBlock> &layout,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const CommonStrategy &strategy, CommonState &state, bool decrement) {
    typename NegativeType<I>::type signedInc = decrement ? -inc : inc;
    typename NegativeType<Ir>::type signedIncR = decrement ? -incR : incR;
    typename NegativeType<Ic>::type signedIncC = decrement ? -incC : incC;

    incAddr(addr, signedInc, signedIncR, signedIncC, layout, atype, astrategy,
            strategy, state);
}

template <HW hw>
template <typename A, typename I>
void gemm_kernel_generator_t<hw>::incDecAddr(const A &addr, I inc,
        const vector<RegisterBlock> &layout, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const CommonStrategy &strategy, CommonState &state, bool decrement) {
    if (astrategy.address2D) stub();
    incDecAddr(addr, inc, Subregister(), Subregister(), layout, atype,
            astrategy, strategy, state, decrement);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::setAddrRemainder(Type T, const GRFRange &addr,
        const RegisterBlock &block, const Subregister &remR,
        const Subregister &remC, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const CommonStrategy &strategy, CommonState &state) {
    if (!isBlock2D(astrategy.accessType) || astrategy.address2D) return;

    auto tempRem = state.ra.alloc_sub<uint32_t>();
    Subregister thisRemR = remR, thisRemC = remC;

    auto memCM = isColMajor(atype.layout);
    auto &remW = memCM ? thisRemR : thisRemC;
    auto &remH = memCM ? thisRemC : thisRemR;
    int bw, bh, multiX;
    getBlock2DWH(bw, bh, atype, block, &multiX);

    if (!block.remainderR) thisRemR.invalidate();
    if (!block.remainderC) thisRemC.invalidate();
    if (thisRemR.isValid()) thisRemR = tempRem.uw(0);
    if (thisRemC.isValid()) thisRemC = tempRem.uw(1);
    if (thisRemR.isValid() && block.offsetR)
        add(1 | sat, thisRemR, remR, -block.offsetR);
    if (thisRemC.isValid() && block.offsetC)
        add(1 | sat, thisRemC, remC, -block.offsetC);
    if (thisRemR.isValid())
        min_(1, thisRemR, block.offsetR ? thisRemR : remR, block.nr);
    if (thisRemC.isValid())
        min_(1, thisRemC, block.offsetC ? thisRemC : remC, block.nc);

    if (remW.isValid()) {
        if (block.count > 1 || multiX > 1) stub();
        mad(1, addr[0].ud(2), -1, remW.uw(), T.size());
    }
    if (remH.isValid()) mad(1, addr[0].ud(3), -1, remH.uw(), T.size() * multiX);
    if (remW.isValid() && T.size() < 4) or_(1, addr[0].ud(2), addr[0].ud(2), 3);

    state.ra.safeRelease(tempRem);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::setAddrRemainder(Type T,
        const vector<GRFRange> &addr, const vector<RegisterBlock> &layout,
        const Subregister &remR, const Subregister &remC,
        const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy,
        const CommonStrategy &strategy, CommonState &state) {
    auto nblocks = int(layout.size());

    for (int b = 0; b < nblocks; b++)
        setAddrRemainder(T, addr[b], layout[b], remR, remC, atype, astrategy,
                strategy, state);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::setupTeardownRemask(Type T, int index,
        bool setup, int nq, const Subregister &remQ,
        const CommonStrategy &strategy, CommonState &state, int fixedOffQ,
        const Subregister &variableOffQ) {
    if (setup) {
        auto masks = state.remaskRegs[index] = state.ra.alloc_range(
                div_up(T.size(), 2) * div_up(nq * 2, GRF::bytes(hw)));
        int ne16 = elementsPerGRF(hw, Type::u16);
        int n16 = std::min(nq, ne16);
        int ne = elementsPerGRF(hw, T);

        auto effRemQ = remQ;
        bool freeEffRemQ = false;
        bool haveVariableOff = variableOffQ.isValid();
        bool haveFixedOff = (fixedOffQ != 0);

        if (haveVariableOff || haveFixedOff) {
            freeEffRemQ = true;
            effRemQ = state.ra.alloc_sub<uint32_t>();

            if (haveVariableOff && haveFixedOff)
                eadd3(1, effRemQ, remQ, -variableOffQ, -fixedOffQ);
            else if (haveVariableOff)
                add(1, effRemQ, remQ, -variableOffQ);
            else
                add(1, effRemQ, remQ, -fixedOffQ);
        }

        mov<uint16_t>(8, masks[0][0](1), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
        if (nq > 8)
            mov<uint16_t>(8, masks[0][8](1),
                    Immediate::uv(8, 9, 10, 11, 12, 13, 14, 15));
        if (GRF::bytes(hw) > 32 && nq > 16)
            add<uint16_t>(16, masks[0][16](1), masks[0][0](1), 16);
        add<uint16_t>(n16, masks[0], masks[0], -effRemQ.w());
        for (int q0 = n16; q0 < nq; q0 += n16)
            add<uint16_t>(n16, masks[q0 / n16], masks[0], q0);

        switch (T.size()) {
            case 1:
                if (nq >= 256) stub();
                for (int q0 = 0; q0 < nq; q0 += n16)
                    mov(n16, masks[q0 / ne].ub(q0 % ne)(1),
                            masks[q0 / n16].ub(1)(2));
                break;
            case 2:
                map(hw, Type::s16, masks, masks, strategy,
                        [=](int simd, const RegData &r1, const RegData &) {
                            asr(simd, r1, r1, 15);
                        });
                break;
            case 4:
                for (int qq0 = div_up(nq, ne16) - 1; qq0 >= 1; qq0--)
                    asr(ne16, masks[qq0 * 2].d(), masks[qq0].w(), 15);
                if (nq > (ne16 / 2))
                    asr(ne16 / 2, masks[1].d(), masks[0].w(ne16 / 2)(1), 15);
                asr(ne16 / 2, masks[0].d(), masks[0].w(), 15);
                break;
            default: stub();
        }

        if (freeEffRemQ) state.ra.safeRelease(effRemQ);
    } else
        state.ra.safeRelease(state.remaskRegs[index]);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::remaskLayout(Type T, int index, bool column,
        const std::vector<RegisterBlock> &layout, const GRFMultirange &regs,
        const CommonStrategy &strategy, CommonState &state, int offset) {
    for (auto &block : layout) {
        auto crosspack = block.crosspack;
        bool colMajor = block.colMajor;
        int nx = colMajor ? block.nr : block.nc;
        int ny = colMajor ? block.nc : block.nr;

        for (int y0 = 0; y0 < ny; y0 += crosspack) {
            for (int x0 = 0; x0 < nx;) {
                auto ii0 = colMajor ? x0 : y0;
                auto jj0 = colMajor ? y0 : x0;
                auto i0 = ii0 + block.offsetR;
                auto j0 = jj0 + block.offsetC;

                int ne;
                auto sub = findBlockReg(T, block, ii0, jj0, regs, ne);

                auto necp = ne * crosspack;
                necp = std::min(necp, 2 * elementsPerGRF(hw, T));
                if ((necp * T) & 3) stub();

                int moff = (offset + (column ? j0 : i0)) * T / 4;
                int mreg = moff / elementsPerGRF<uint32_t>(hw);
                int msub = moff % elementsPerGRF<uint32_t>(hw);

                int mstride;
                if (colMajor != column && crosspack == 1)
                    mstride = 1;
                else if (colMajor == column && crosspack == 4 / T)
                    mstride = 0;
                else
                    stub();

                and_<uint32_t>((necp * T) / 4, sub.ud()(1), sub.ud()(1),
                        state.remaskRegs[index][mreg][msub](mstride));
                x0 += necp / crosspack;
            }
        }
    }
}

static bool needsRemask(Type T, bool column, const RegisterBlock &block,
        const MatrixAddressingStrategy &astrategy, bool ignoreMasks = false) {
    if (!ignoreMasks)
        if (column ? !block.remainderC : !block.remainderR) return false;

    int maskGranularity = block.ebytes;
    if (block.ebytes >= 16) maskGranularity = 4;
    if (isBlock2D(astrategy.accessType))
        maskGranularity = std::max(maskGranularity, 4);
    if (ignoreMasks) maskGranularity = 256;

    return (T.size() < maskGranularity);
}

static bool needsRemask(Type T, bool column,
        const vector<RegisterBlock> &layout,
        const MatrixAddressingStrategy &astrategy, bool ignoreMasks = false) {
    for (auto &block : layout)
        if (needsRemask(T, column, block, astrategy, ignoreMasks)) return true;
    return false;
}

// The systolic array performs a series of GEMVs with a single fixed-size matrix.
// The size of the matrix is osys x ksys with vectors of size ksys x 1.
// The number of GEMVs (with same matrix) is given by the (variable) repeat count.
struct SystolicParams {
    int opsPerChan; // # of FMAs/stage
    int sdepth; // Number of stages (systolic depth)
    int rcountMax; // Maximum repeat count (# of RHS)
    int ksys; // Total number of FMAs
    int osys; // Output vector length
};

static inline SystolicParams systolicParams(
        HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy) {
    SystolicParams params;
    params.opsPerChan = std::max(1, std::min(4 / problem.Ta, 4 / problem.Tb));
    params.sdepth = 8;
    params.ksys = params.sdepth * params.opsPerChan;
    params.osys = GRF::bytes(hw) / std::max(problem.Tc.size(), 4);
    params.rcountMax = 8;

    if (hw == HW::XeHPC) {
        // Workaround for src2 read suppression bug (TODO PVC-B: remove WA)
        bool cColMajor = isRegisterColMajor(problem.Tc, problem.C, strategy.C);
        if (strategy.unroll[cColMajor ? LoopN : LoopM] == 8)
            params.rcountMax = 4;
    }

    return params;
}

// Return # of outer products performed at once.
static inline int minOuterProductCount(
        HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy) {
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc;
    if (strategy.systolic) {
        auto params = systolicParams(hw, problem, strategy);
        return params.ksys;
    }
    if (Ta.real().size() == 1 && Tb.real().size() == 1 && Tc.real().size() == 4
            && (hw >= HW::Gen12LP))
        return 4;
    return 1;
}

// Return # of outer products performed at once.
static inline int outerProductCount(
        HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy) {
    return minOuterProductCount(hw, problem, strategy) * strategy.kChain;
}

// Get the A and B crosspacks needed by the kernel. 0 indicates any crosspack is OK.
static std::tuple<int, int> targetKernelCrosspack(
        HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy) {
    int opBatch = minOuterProductCount(hw, problem, strategy);
    bool aColMajor = isRegisterColMajor(problem.Ta, problem.A, strategy.A);
    bool bColMajor = isRegisterColMajor(problem.Tb, problem.B, strategy.B);
    bool cColMajor = isRegisterColMajor(problem.Tc, problem.C, strategy.C);

    if (strategy.systolic) {
        return cColMajor
                ? std::make_tuple(std::max(1, 4 / problem.Ta.size()), 1)
                : std::make_tuple(1, std::max(1, 4 / problem.Tb.size()));
    }
    if (opBatch == 1) {
        return cColMajor ? std::make_tuple(1, 0) : std::make_tuple(0, 1);
    } else {
        bool bcastOK = cColMajor ? bColMajor : !aColMajor;

        return cColMajor ? std::make_tuple(opBatch, bcastOK ? 1 : opBatch)
                         : std::make_tuple(bcastOK ? 1 : opBatch, opBatch);
    }
}

// Get the A and B crosspacks to use for SLM data.
static std::tuple<int, int> targetSLMCrosspack(
        HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy) {
    int opBatch = minOuterProductCount(hw, problem, strategy);

    if (strategy.systolic) {
        bool cColMajor = isRegisterColMajor(problem.Tc, problem.C, strategy.C);
        return cColMajor
                ? std::make_tuple(std::max(1, 4 / problem.Ta.size()), opBatch)
                : std::make_tuple(opBatch, std::max(1, 4 / problem.Tb.size()));
    }
    return std::make_tuple(opBatch, opBatch);
}

// Get the A and B tiling needed by the kernel.
// Return value is in the format {A_tileR, A_tileC, B_tileR, B_tileC}.
static std::tuple<int, int, int, int> targetKernelTiling(
        HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy) {
    if (strategy.systolic) {
        auto params = systolicParams(hw, problem, strategy);
        bool cColMajor = isRegisterColMajor(problem.Tc, problem.C, strategy.C);
        auto tileO_V = params.osys;
        auto tileI_N = params.ksys;
        if (strategy.unroll[cColMajor ? LoopN : LoopM] == 1) tileI_N = 0;
        return cColMajor ? std::make_tuple(tileO_V, 0, tileI_N, 0)
                         : std::make_tuple(0, tileI_N, 0, tileO_V);
    }
    return std::make_tuple(0, 0, 0, 0);
}

// Do one outer product (k = 1 slice) of A*B, updating C. ha and hb are the
//  k indices within the A and B chunks, respectively. A_copy, B_copy are the
//  indices of the A, B copies to use.
template <HW hw>
void gemm_kernel_generator_t<hw>::outerProduct(int h, int ha, int hb,
        int opCount, const vector<RegisterBlock> &A_layout,
        const vector<RegisterBlock> &B_layout, const GRFMultirange &A_regs,
        const GRFMultirange &B_regs, GEMMProblem &problem,
        GEMMStrategy &strategy, GEMMState &state) {
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc;

    if (strategy.systolic) {
        outerProductSystolic(h, ha, hb, A_layout, B_layout, A_regs, B_regs,
                problem, strategy, state);
        return;
    }
    if (isGen9IGEMM(hw, Ta, Tb, Tc)) {
        outerProductGen9IGEMM(ha, hb, A_layout, B_layout, A_regs, B_regs,
                problem, strategy, state);
        return;
    }

    bool mixedMode = ((Tc.real() == Type::f32)
            && (Ta.real() != Type::f32 || Tb.real() != Type::f32));
    bool useDP4A = (Ta.size() == 1 && Tb.size() == 1 && Tc.size() == 4
            && hw >= HW::Gen12LP);

    int minOPCount = minOuterProductCount(hw, problem, strategy);
    int kChain = std::min(strategy.kChain, opCount);
    int aCP, bCP;
    std::tie(aCP, bCP) = targetKernelCrosspack(hw, problem, strategy);

    int accNum = 0;
    Subregister Clast;
    int nec = elementsPerGRF(hw, Tc);
    bool globalCM = isLayoutColMajor(state.C_layout);
    int fmaSIMD = strategy.fmaSIMD;

    bool csplit = false, mixedRC = false;
    int icompCount = 1, ocompCount = 1, ivcompCount = 1, ovcompCount = 1;

    bool bfloat16WA = (Tc.real() == Type::f32)
            && ((globalCM ? Tb : Ta).real() == Type::bf16);

    // Emit an FMA instruction.
    auto outputFMA = [&](const InstructionModifier &mod, const Subregister &A,
                             const Subregister &B, const Subregister &C,
                             const RegData &bcastSrc, bool colMajor, int hh,
                             bool ivfirst, bool ivlast) {
        auto Cacc = AccumulatorRegister(accNum).sub(0, Tc.real().ngen());
        auto Csrc = (hh == 0 && ivfirst) ? C : Cacc;
        auto Cdst = (hh == opCount - minOPCount && ivlast) ? C : Cacc;
        if (useDP4A) {
            auto Ar = A.reinterpret(
                    0, isSigned(A.getType()) ? DataType::d : DataType::ud);
            auto Br = B.reinterpret(
                    0, isSigned(B.getType()) ? DataType::d : DataType::ud);

            colMajor ? dp4a(mod, Cdst(1), Csrc(1), Ar(1), Br(0))
                     : dp4a(mod, Cdst(1), Csrc(1), Br(1), Ar(0));
        } else if (C.isARF() && hw < HW::XeHP) {
            colMajor ? mac(mod, C(1), A(1), bcastSrc)
                     : mac(mod, C(1), bcastSrc, B(1));
        } else {
            // On Gen12, always put broadcast in src2 for better bank conflict avoidance.
            colMajor ? mad(mod, Cdst(1), Csrc(1), A(1), bcastSrc)
                     : (hw < HW::Gen12LP)
                            ? mad(mod, Cdst(1), Csrc(1), bcastSrc, B(1))
                            : mad(mod, Cdst(1), Csrc(1), B(1), bcastSrc);
        }
    };

    ha = align_down(ha, opCount);
    hb = align_down(hb, opCount);

    // Decide whether to loop in column or row major order.
    //   x = vectorized dimension
    //   y = non-vectorized dimension
    int nx = globalCM ? strategy.unroll[LoopM] : strategy.unroll[LoopN];
    int ny = globalCM ? strategy.unroll[LoopN] : strategy.unroll[LoopM];
    int nx1 = (mixedMode || state.broadcast) ? nx : fmaSIMD;

    // Prepare for chaining FMAs through accumulator registers.
    int necAcc = nec * (csplit ? 2 : 1);
    int accCount = AccumulatorRegister::count(hw, strategy.GRFs, Tc.ngen());
    int accPerFMA = div_up(std::min(nx, fmaSIMD), necAcc);
    int minAccPerFMA = Tc.isFP() ? 1 : 2;
    accPerFMA = std::max(accPerFMA, minAccPerFMA);
    int independentAccs = div_up(accCount, accPerFMA);

    int nx1i = 1, ny1 = 1;
    if (kChain > 1) {
        if (independentAccs < icompCount) hw_unsupported();
        int indepAccComp = div_up(independentAccs, icompCount);

        nx1i = std::min(nx1, indepAccComp * fmaSIMD);
        ny1 = div_up(indepAccComp, div_up(nx1i, fmaSIMD));
    }

    GRFRange broadcastRegs = state.broadcast_regs;
    Subregister lastBcastBase;

    // Last A/B blocks found;
    const RegisterBlock *A_blockLast = nullptr, *B_blockLast = nullptr;

    for (int x0 = 0; x0 < nx; x0 += nx1) {
        for (int ovcomp = 0; ovcomp < ovcompCount; ovcomp++) {
            for (int ocomp = 0; ocomp < ocompCount; ocomp++) {
                for (int y0 = 0; y0 < ny; y0 += ny1) {
                    for (int x1 = 0; x1 < nx1 && (x0 + x1) < nx;) {
                        int x1New = x1;
                        for (int ivcomp = 0; ivcomp < ivcompCount; ivcomp++) {
                            for (int hh = 0; hh < opCount; hh += minOPCount) {
                                accNum = 0;
                                for (int y1 = 0; y1 < ny1 && y0 + y1 < ny;
                                        y1++) {
                                    for (int x1i = x1; (x1i < x1 + nx1i)
                                            && (x0 + x1i < nx);) {
                                        auto x = x0 + x1i;
                                        auto y = y0 + y1;
                                        auto i = globalCM ? x : y;
                                        auto j = globalCM ? y : x;
                                        auto hha = ha + hh;
                                        auto hhb = hb + hh;

                                        int fmaCount = 1;

                                        for (int icomp = 0; icomp < icompCount;
                                                icomp++) {
                                            // Find the appropriate A and B registers.
                                            int na, nb;
                                            int vcomp = ivcomp + ovcomp;
                                            int ncomp = (vcomp ^ ocomp) + icomp;
                                            int compA
                                                    = globalCM ? vcomp : ncomp;
                                            int compB
                                                    = globalCM ? ncomp : vcomp;

                                            if (compA >= Ta.components()
                                                    || compB >= Tb.components())
                                                continue;

                                            const RegisterBlock *A_block,
                                                    *B_block;
                                            Subregister A = findBlockReg(Ta,
                                                    A_layout, i, hha, A_regs,
                                                    na, A_block, compA);
                                            Subregister B = findBlockReg(Tb,
                                                    B_layout, hhb, j, B_regs,
                                                    nb, B_block, compB);

                                            // Check for expected crosspack.
                                            if (globalCM ? (aCP
                                                        && A_block->crosspack
                                                                != aCP)
                                                         : (bCP
                                                                 && B_block->crosspack
                                                                         != bCP))
                                                stub();

                                            // Check if we should specify {Atomic}.
                                            bool atomic = (strategy.atomicFMA
                                                    && (A_block == A_blockLast)
                                                    && (B_block
                                                            == B_blockLast));
                                            A_blockLast = A_block;
                                            B_blockLast = B_block;

                                            // Find the appropriate C register.
                                            int C_buffer = csplit
                                                    ? 0
                                                    : (icomp + ocomp);
                                            int compC = csplit ? ocomp : 0;
                                            int nc;
                                            const RegisterBlock *C_block;
                                            Subregister C = findBlockReg(Tc,
                                                    state.C_layout, i, j,
                                                    state.C_regs[C_buffer], nc,
                                                    C_block, compC);
                                            if (C_block->crosspack > 1) stub();

                                            // Swap out C register for an accumulator, if necessary.
                                            if (strategy.cAccumulators) {
                                                auto C_roff = C.getBase()
                                                        - state.C_regs[0]
                                                                  .ranges[0]
                                                                  .getBase();
                                                if (C_roff < state.C_accCount)
                                                    C = AccumulatorRegister(
                                                            C_roff)
                                                                .sub(C.getOffset(),
                                                                        Tc.ngen());
                                            }

                                            InstructionModifier mod;

                                            // Use requested execution size if possible, but limited to available elements.
                                            // Decide on kernel type based on register block layouts.
                                            bool canColMajor
                                                    = (A_block->colMajor
                                                            && globalCM);
                                            bool canRowMajor
                                                    = (!B_block->colMajor
                                                            && !globalCM);
                                            bool colMajor = globalCM;

                                            if (!canColMajor && !canRowMajor)
                                                fmaCount = 1;
                                            else if (canColMajor)
                                                fmaCount = rounddown_pow2(
                                                        std::min({fmaSIMD, na,
                                                                nc}));
                                            else
                                                fmaCount = rounddown_pow2(
                                                        std::min({fmaSIMD, nb,
                                                                nc}));

                                            int simdSize = fmaCount;
                                            if (!csplit)
                                                simdSize *= Tc.components();

                                            // Crosspacked kernels: ensure broadcast matrix is contiguous in k.
                                            if (minOPCount > 1) {
                                                bool nativeDir = (globalCM
                                                                ? B_block->colMajor
                                                                : !A_block->colMajor);
                                                auto bcastCrosspack
                                                        = (globalCM ? B_block
                                                                    : A_block)
                                                                  ->crosspack;
                                                if (nativeDir) {
                                                    if ((globalCM ? nb : na)
                                                            < minOPCount)
                                                        stub();
                                                    if (bcastCrosspack > 1)
                                                        stub();
                                                } else {
                                                    if (bcastCrosspack
                                                            % minOPCount)
                                                        stub();
                                                }
                                            }

                                            // Add Atomic if appropriate.
                                            if (atomic) mod |= Atomic;

                                            // Handle broadcast duties.
                                            Subregister bcastSrcSub
                                                    = colMajor ? B : A;
                                            RegData bcastSrc = bcastSrcSub;

                                            if (state.broadcast) {

                                                // Broadcast if necessary: pair of doubles (doubleWA) or single elements.
                                                int nbcast = strategy.doubleWA
                                                        ? 2
                                                        : 1;
                                                int hs = strategy.doubleWA
                                                        ? 0
                                                        : nbcast;

                                                auto bcastType
                                                        = bcastSrc.getType();
                                                Subregister bcastBase
                                                        = bcastSrcSub;
                                                bcastBase.setOffset(
                                                        bcastBase.getOffset()
                                                        & ~(nbcast - 1));

                                                if (bcastBase
                                                        != lastBcastBase) {
                                                    auto bcastRegion = bcastBase(
                                                            0, nbcast,
                                                            (nbcast > 1) ? 1
                                                                         : 0);
                                                    if (bfloat16WA) {
                                                        // Upconvert to f32 during broadcast.
                                                        bcastRegion.setType(
                                                                DataType::uw);
                                                        shl(strategy.fmaSIMD
                                                                        * Tc.components(),
                                                                broadcastRegs[0]
                                                                        .ud(),
                                                                bcastRegion,
                                                                16);
                                                    } else {
                                                        moveToIntPipe(
                                                                strategy.fmaSIMD
                                                                        * Tc.components(),
                                                                bcastRegion);
                                                        mov(strategy.fmaSIMD
                                                                        * Tc.components()
                                                                        * bcastSrc.getBytes()
                                                                        / bcastRegion
                                                                                  .getBytes(),
                                                                broadcastRegs[0].retype(
                                                                        bcastRegion
                                                                                .getType()),
                                                                bcastRegion);
                                                    }
                                                }
                                                if (bfloat16WA)
                                                    bcastType = DataType::f;
                                                bcastSrc = broadcastRegs[0].sub(
                                                        bcastSrc.getOffset()
                                                                & (nbcast - 1),
                                                        bcastType)(hs);
                                                lastBcastBase = bcastBase;
                                            }

                                            bool ivfirst
                                                    = mixedRC || (ivcomp == 0);
                                            bool ivlast = mixedRC
                                                    || (ivcomp
                                                            == ivcompCount - 1);

                                            // Finally, perform the long-awaited FMA.
                                            outputFMA(simdSize | mod, A, B, C,
                                                    bcastSrc, colMajor, hh,
                                                    ivfirst, ivlast);
                                            Clast = C;

                                            if (kChain > 1
                                                    && accNum >= accCount)
                                                stub();
                                            accNum += std::max(minAccPerFMA,
                                                    div_up(fmaCount, necAcc));
                                        } /* icomp */

                                        x1i += fmaCount;
                                        x1New = x1i;
                                    } /* x1i */
                                } /* y1 */
                            } /* hh */
                        } /* ivcomp */
                        x1 = x1New;
                    } /* x1 */
                } /* y0 */
            } /* ocomp */
        } /* ovcomp */
    } /* x0 */
}

template <HW hw>
void gemm_kernel_generator_t<hw>::outerProductGen9IGEMM(int ha, int hb,
        const vector<RegisterBlock> &A_layout,
        const vector<RegisterBlock> &B_layout, const GRFMultirange &A_regs,
        const GRFMultirange &B_regs, GEMMProblem &problem,
        GEMMStrategy &strategy, GEMMState &state) {
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc;
    DataType tempType
            = (Ta.isSigned() || Tb.isSigned()) ? DataType::w : DataType::uw;

    struct AddItem {
        int simd;
        RegData dest, src0, src1;
    };
    std::vector<AddItem> adds;

    auto replayAdds = [&]() {
        for (auto &item : adds)
            add(item.simd, item.dest, item.src0, item.src1);
        adds.clear();
    };

    bool globalCM = isLayoutColMajor(state.C_layout);

    // Decide whether to loop in column or row major order.
    int nx = globalCM ? strategy.unroll[LoopM] : strategy.unroll[LoopN];
    int ny = globalCM ? strategy.unroll[LoopN] : strategy.unroll[LoopM];

    int tidx = 0;
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx;) {
            auto i = globalCM ? x : y;
            auto j = globalCM ? y : x;

            int fmaCount;

            // Find the appropriate A and B registers.
            int na, nb;
            const RegisterBlock *A_block, *B_block;
            Subregister A
                    = findBlockReg(Ta, A_layout, i, ha, A_regs, na, A_block);
            Subregister B
                    = findBlockReg(Tb, B_layout, hb, j, B_regs, nb, B_block);

            // Find the appropriate C register. Todo: remainders.
            int nc;
            const RegisterBlock *C_block;
            Subregister C = findBlockReg(
                    Tc, state.C_layout, i, j, state.C_regs[0], nc, C_block);

            // No C crosspack support.
            auto cpA = A_block->crosspack, cpB = B_block->crosspack;
            if (C_block->crosspack > 1) stub();

            // Swap out C register for an accumulator, if necessary.
            auto C_roff = C.getBase() - state.C_regs[0].ranges[0].getBase();
            if (C_roff < state.C_accCount)
                C = AccumulatorRegister(C_roff).sub(C.getOffset(), Tc.ngen());

            // Use requested execution size if possible, but limited to available elements.
            // Decide the kernel type based on register block layouts.
            bool canColMajor = (A_block->colMajor && C_block->colMajor);
            bool canRowMajor = (!B_block->colMajor && !C_block->colMajor);
            bool colMajor;

            if (!canColMajor && !canRowMajor) {
                colMajor = true;
                fmaCount = 1;
            } else if (canColMajor) {
                colMajor = true;
                fmaCount = na;
            } else {
                colMajor = false;
                fmaCount = nb;
            }
            fmaCount = rounddown_pow2(std::min(
                    {strategy.fmaSIMD, nc, elementsPerGRF<int16_t>(hw)}));

            auto temp = state.tempMul_regs[tidx++];

            if (C.isARF()) {
                if (colMajor)
                    mac(fmaCount, C(1), A(cpA), B(0));
                else
                    mac(fmaCount, C(1), A(0), B(cpB));
            } else {
                if (colMajor)
                    mul(fmaCount, temp[0].sub(0, tempType)(2), A(cpA), B(0));
                else
                    mul(fmaCount, temp[0].sub(0, tempType)(2), A(0), B(cpB));

                adds.push_back(
                        {fmaCount, C(1), C(1), temp[0].sub(0, tempType)(2)});
            }

            if (tidx >= int(state.tempMul_regs.size())) {
                tidx = 0;
                replayAdds();
            }

            x += fmaCount;
        }
    }

    replayAdds();

    // A4B4 outer product (4 temporary GRFs per 2 C registers) - 2/3 SP
    //
    // mul (32) temp0.0:w<1> A.0:b<32;16,2> B.0:b<32;16,2>   - EM
    // mul (32) temp2.0:w<1> A.1:b<32;16,2> B.1:b<32;16,2>   - FPU
    // add (16) C.0:d<1> C.0:d<8;8,1> temp0.0:w<16;8,2>      - EM
    // add (16) C.0:d<1> C.0:d<8;8,1> temp0.1:w<16;8,2>      - FPU
    // add (16) C.0:d<1> C.0:d<8;8,1> temp2.0:w<16;8,2>      - EM
    // add (16) C.0:d<1> C.0:d<8;8,1> temp2.1:w<16;8,2>      - FPU

    // Faster A4B4 outer product a la non-VNNI (4 temporary GRFs per 2 C registers) - 4/5 SP
    //
    // mul (32) temp0.0:w<1> A.0:b<32;16,2> B.0:b<32;16,2>   - EM
    // mul (32) temp2.0:w<1> A.1:b<32;16,2> B.1:b<32;16,2>   - FPU
    // add (32) (sat) temp0.0:w<1> temp0.0:w<1> temp2.0:w<1> - EM/FPU
    // add (16) C.0:d<1> C.0:d<8;8,1> temp0.0:w<16;8,2>      - EM
    // add (16) C.0:d<1> C.0:d<8;8,1> temp0.1:w<16;8,2>      - FPU
}

static int elementDiff(HW hw, const RegData &r1, const RegData &r2) {
    return elementsPerGRF(hw, r1.getType()) * (r1.getBase() - r2.getBase())
            + (r1.getOffset() - r2.getOffset());
}

// Accumulate multiple outer products using the systolic array.
template <HW hw>
void gemm_kernel_generator_t<hw>::outerProductSystolic(int h, int ha, int hb,
        const vector<RegisterBlock> &A_layout,
        const vector<RegisterBlock> &B_layout, const GRFMultirange &A_regs,
        const GRFMultirange &B_regs, GEMMProblem &problem,
        GEMMStrategy &strategy, GEMMState &state) {
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc;
    bool globalCM = isLayoutColMajor(state.C_layout);
    auto params = systolicParams(hw, problem, strategy);
    auto ksys = params.ksys;
    auto osys = params.osys;
    auto sdepth = params.sdepth;
    auto rcountMax = params.rcountMax;
    bool rsFix = (strategy.readSuppressionWA
            && hasMasking(globalCM ? A_layout : B_layout));
    bool canAtomicNon8x8
            = (hw >= HW::XeHPC) && (getStepping() >= SteppingPVCXTB0);

    // dpas processes ksys outer products at once.
    ha = align_down(ha, ksys);
    hb = align_down(hb, ksys);

    // Decide whether to loop in column or row major order, to facilitate macro sequences.
    int nx = strategy.unroll[globalCM ? LoopN : LoopM];
    int ny = strategy.unroll[globalCM ? LoopM : LoopN];

    for (int y = 0; y < ny; y += osys) {
        Subregister A0, B0, C0;
        int rcount = 0, x0 = 0;

        auto issueDPAS = [&](bool last) {
            while (rcount > 0) {
                InstructionModifier mod = osys;

                auto rc2 = rounddown_pow2(rcount);
                auto &V0 = globalCM ? A0 : B0;
                auto &N0 = globalCM ? B0 : A0;

                if (rsFix) {
                    GRF v0GRF {V0.getBase()};
                    mov<uint32_t>(8, v0GRF, v0GRF);
                    rsFix = false;
                }

                if (strategy.atomicFMA)
                    if (!(last && (rc2 == rcount)))
                        if (rc2 == 8 || canAtomicNon8x8) mod |= Atomic;

                dpas(mod, sdepth, rc2, C0, C0, V0, N0);

                rcount -= rc2;
                x0 += rc2;
                N0.setBase(N0.getBase() + rc2);
                C0.setBase(C0.getBase() + rc2);
            }
        };

        for (int x = 0; x < nx; x++) {
            auto i = globalCM ? y : x;
            auto j = globalCM ? x : y;

            // Find the appropriate A and B registers.
            int na, nb, nc;
            const RegisterBlock *A_block, *B_block, *C_block;
            Subregister A
                    = findBlockReg(Ta, A_layout, i, ha, A_regs, na, A_block);
            Subregister B
                    = findBlockReg(Tb, B_layout, hb, j, B_regs, nb, B_block);
            Subregister C = findBlockReg(
                    Tc, state.C_layout, i, j, state.C_regs[0], nc, C_block);

            int nv = globalCM ? na : nb;
            int nn = globalCM ? nb : na;

            // Verify DPAS requirements.
            if (globalCM) {
                if (A_block->crosspack * problem.Ta.size()
                        != std::max(4, problem.Ta.size()))
                    stub();
                if (B_block->crosspack > 1) stub();
            } else {
                if (B_block->crosspack * problem.Tb.size()
                        != std::max(4, problem.Tb.size()))
                    stub();
                if (A_block->crosspack > 1) stub();
            }
            if (A_block->colMajor != globalCM || B_block->colMajor != globalCM)
                stub();
            if (C_block->crosspack > 1) stub();

            if (nv != osys) stub();
            if (nn < ksys) stub();

            // Check if current DPAS can be fused with the previous one.
            bool chain = false;
            if (A0.isValid()) {
                chain = globalCM ? (elementDiff(hw, B, B0) == (x - x0) * ksys)
                                 : (elementDiff(hw, A, A0) == (x - x0) * ksys);
                chain = chain && (elementDiff(hw, C, C0) == (x - x0) * osys);
                chain = chain && (rcount < rcountMax);
            }

            if (chain)
                rcount++;
            else {
                if (A0.isValid()) issueDPAS(false);
                A0 = A;
                B0 = B;
                C0 = C;
                rcount = 1;
                A0.setType(problem.Ta.ngen());
                B0.setType(problem.Tb.ngen());
                C0.setType(problem.Tc.ngen());
                x0 = x;
            }
        }

        bool finishChain = !strategy.extendedAtomicFMA || (y + osys >= ny);
        issueDPAS(finishChain);
    }
}

// Perform C update operation on C_acc, given original C data in C_load.
// All inputs and outputs are assumed to be of type problem.Ts.
template <HW hw>
void gemm_kernel_generator_t<hw>::updateC(const GRFMultirange &C_acc,
        const GRFMultirange &C_accSwap, const GRFMultirange &C_load,
        GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state) {
    auto &alphar = problem.alpha_real;
    auto &betar = problem.beta_real;
    bool alpha1 = (alphar == 1);
    bool alphaM1 = (alphar == -1);
    bool beta1 = (betar == 1);
    bool beta0 = (betar == 0);
    bool betaM1 = (betar == -1);

#define FOR_EACH_C(f) \
    do { \
        map(hw, state.Tacc.real(), C_load, C_acc, strategy, \
                [&](int esize, GRF loaded, GRF acc) { f; }); \
    } while (false)

#define FOR_EACH_C_CX(f) \
    do { \
        map(hw, state.Tacc.real(), C_load, C_acc, C_accSwap, strategy, \
                [&](int esize, GRF loaded, GRF acc, GRF accswap) { f; }); \
    } while (false)

    if (!beta0) {
        if (alpha1 || alphaM1) {
            if (beta1)
                FOR_EACH_C(add(esize, acc, loaded, alpha1 ? acc : -acc));
            else if (betaM1)
                FOR_EACH_C(add(esize, acc, -loaded, alpha1 ? acc : -acc));
            else if (betar.fixed())
                stub(); // beta should be put in a register first.
            else {
                if (!strategy.doubleWA)
                    FOR_EACH_C(mad(esize, acc, alpha1 ? acc : -acc, loaded,
                            betar.getRegAvoiding(hw, loaded)));
                else {
                    FOR_EACH_C(mul(esize, loaded, loaded,
                            betar.getRegAvoiding(hw, loaded)));
                    FOR_EACH_C(add(esize, acc, loaded, alpha1 ? acc : -acc));
                }
            }
        } else {
            bool neg = false;
            if (!beta1) {
                if (betaM1)
                    neg = true;
                else if (!betar.fixed())
                    FOR_EACH_C(mul(esize, loaded, loaded,
                            betar.getRegAvoiding(hw, acc)));
                else
                    stub();
            }
            if (alphar.fixed())
                stub(); // alpha should be put in a register first.
            else {
                if (!strategy.doubleWA)
                    FOR_EACH_C(mad(esize, acc, neg ? -loaded : loaded, acc,
                            alphar.getRegAvoiding(hw, acc)));
                else {
                    FOR_EACH_C(mul(
                            esize, acc, acc, alphar.getRegAvoiding(hw, acc)));
                    FOR_EACH_C(add(esize, acc, neg ? -loaded : loaded, acc));
                }
            }
        }
    } else if (alphaM1)
        FOR_EACH_C(mov(esize, acc, -acc));
    else if (alpha1)
        /* no op */;
    else if (alphar.fixed())
        stub(); // alpha should be put in a register first.
    else {
        FOR_EACH_C(mul(esize, acc, acc, alphar.getRegAvoiding(hw, acc)));
    }

    if (problem.hasPostOp()) {
        Label labelPostOpDone;
        bool allocFlag = state.flagAP.isInvalid();
        auto flagNonfinal = allocFlag ? state.raVFlag.alloc() : state.flagAP;
        and_(1 | nz | flagNonfinal, null.ud(), state.inputs.flags,
                FlagNonfinalKBlock);
        jmpi(1 | flagNonfinal, labelPostOpDone);
        if (allocFlag) state.raVFlag.safeRelease(flagNonfinal);
        if (state.Tacc != Type::f32 || !postOpInjector) stub();
        for (const auto &range : C_acc.ranges)
            postOpInjector->compute(range);
        mark(labelPostOpDone);
    }

#undef FOR_EACH_C
#undef FOR_EACH_C_CX
}

template <HW hw>
bool gemm_kernel_generator_t<hw>::reblockLayout(Type Tdst,
        vector<int32_t> &blockMap, vector<RegisterBlock> &layoutDst,
        const vector<RegisterBlock> &layoutRef,
        const vector<RegisterBlock> &layoutSrc, const MatrixAddressing &atype,
        const MatrixAddressingStrategy &astrategy) {
    auto nblockRef = layoutRef.size();
    layoutDst.clear();
    layoutDst.reserve(nblockRef);
    blockMap.clear();
    blockMap.reserve(nblockRef + 1);
    blockMap.push_back(0);
    for (auto &blockRef : layoutRef) {
        RegisterBlock blockDst, blockMid;
        for (auto &blockSrc : layoutSrc) {
            int rr1 = blockRef.offsetR - blockSrc.offsetR,
                rr2 = rr1 + blockRef.nr;
            int cc1 = blockRef.offsetC - blockSrc.offsetC,
                cc2 = cc1 + blockRef.nc;
            if (rr1 >= blockSrc.nr || rr2 <= 0) continue;
            if (cc1 >= blockSrc.nc || cc2 <= 0) continue;
            rr1 = std::max(rr1, 0);
            cc1 = std::max(cc1, 0);
            rr2 = std::min(rr2, int(blockSrc.nr));
            cc2 = std::min(cc2, int(blockSrc.nc));
            if (!getSubblock(Tdst, blockMid, blockSrc, false, rr1, rr2, rr1,
                        rr2, true, atype, astrategy))
                return false;
            if (!getSubblock(Tdst, blockDst, blockMid, true, cc1, cc2, cc1, cc2,
                        true, atype, astrategy))
                return false;
            layoutDst.push_back(blockDst);
        }
        blockMap.push_back(int32_t(layoutDst.size()));
    }
    return true;
}

// Update an entire C layout.
template <HW hw>
void gemm_kernel_generator_t<hw>::updateCLayout(
        const vector<RegisterBlock> &layoutExt, const GRFRange (&C_addr0)[2],
        const RegisterBlock &C_block0, COperation op, GEMMProblem &problem,
        GEMMStrategy &strategy, GEMMState &state) {
#define FOR_EACH_C for (int q = 0; q < C_count; q++)
    auto Tc = problem.Tc, Tc_ext = problem.Tc_ext, Ts = problem.Ts;
    bool loadOnly = (op == COperation::Load);
    bool beta0 = problem.beta0();
    bool needLoad = (!beta0 && !loadOnly);
    bool copyC = state.copyC;
    int C_count = (op == COperation::UpdateStore) ? state.C_count : 1;

    auto nblocks = int(layoutExt.size());
    bool haveDescs = layoutExt[0].descAssigned;

    vector<GRFRange>(&C_addrs)[2] = state.C_addrs;
    GRFMultirange C_extRange, C_copyRange;
    GRFMultirange &C_accRange = state.C_regs[0];
    auto &C_extRegs = C_extRange.ranges;
    auto &C_copyRegs = C_copyRange.ranges;
    vector<GRFRange> C_convertRegs;

    for (int q = 0; q < C_count; q++)
        C_addrs[0].clear();

    // Map layout to blocks in internal C layout as needed.
    vector<RegisterBlock> layout;
    vector<int> blockMap;
    if (copyC) {
        if (!reblockLayout(Tc, blockMap, layout, layoutExt, state.C_layout,
                    problem.C, strategy.C))
            stub();
    } else {
        layout = layoutExt;
        blockMap.resize(nblocks + 1);
        for (int i = 0; i <= nblocks; i++)
            blockMap[i] = i;
    }

    // Prepare for late C conversion.
    bool lateCConvert = (!loadOnly && !strategy.C.atomic
            && problem.needsTsConvert() && state.Tacc != Ts);
    bool copyCLoad = needLoad && (copyC || lateCConvert);
    if (lateCConvert && Tc.isComplex()) stub();

    // Load as much of C as is possible at a time, given register space.
    for (int lstart = 0; lstart < nblocks;) {
        int lend;

        // Allocate address and data registers for C updating. If allocator chokes,
        //  proceed with the registers we were able to allocate.
        //
        // At the same time, build up three layouts for this chunk of C:
        //   sublayoutExt:   C data to be loaded/stored
        //   sublayoutCopy:  copied C data
        //   sublayoutAcc:   C data in accumulators
        bool allocOK = true;
        auto tryAlloc = [&](int regs, Bundle hint = Bundle()) {
            auto range = state.ra.try_alloc_range(regs, hint);
            allocOK &= range.isValid();
            return range;
        };

        vector<RegisterBlock> sublayoutExt, sublayoutCopy, sublayoutAcc;
        size_t sublayoutCopySize = 0;
        int bytes = 0, bytesConvert = 0;
        int tokens = 0, maxTokens = 256;
        if (needLoad && hw >= HW::Gen12LP) maxTokens = tokenCount(hw);

        for (lend = lstart; (lend < nblocks) && (tokens < maxTokens);
                lend++, tokens++) {
            auto li0 = blockMap[lend], li1 = blockMap[lend + 1];
            int expand
                    = lateCConvert ? div_up(Ts.size(), state.Tacc.size()) : 1;

            if (copyCLoad)
                for (int li = li0; li < li1; li++) {
                    auto block = layout[li];
                    block.compact(state.Tacc);
                    block.offsetBytes = bytesConvert;
                    bytesConvert += block.nregs() * expand * GRF::bytes(hw);
                    sublayoutCopy.push_back(block);
                }

            auto blockExt = layoutExt[lend];
            auto naddr = addrGRFCount(problem.C, strategy.C, blockExt);
            FOR_EACH_C C_addrs[q].push_back(
                    (blockExt.offsetR == 0 && blockExt.offsetC == 0)
                            ? C_addr0[q]
                            : tryAlloc(naddr));
            if (needLoad || copyC)
                C_extRegs.push_back(tryAlloc(
                        blockExt.nregs(), getHint(HintType::CLoad, strategy)));
            if (copyCLoad)
                for (int li = li0; li < li1; li++)
                    C_copyRegs.push_back(tryAlloc(
                            sublayoutCopy[li - li0 + sublayoutCopySize].nregs()
                                    * expand,
                            getHint(HintType::CLoad, strategy)));
            if (lateCConvert)
                for (int li = li0; li < li1; li++)
                    C_convertRegs.push_back(
                            tryAlloc(layout[li].nregs() * expand));
            if (!allocOK) break;

            blockExt.offsetBytes = bytes;
            bytes += blockExt.nregs() * GRF::bytes(hw);
            sublayoutExt.push_back(blockExt);

            sublayoutCopySize = sublayoutCopy.size();
        }

        sublayoutCopy.resize(sublayoutCopySize);

        int listart = blockMap[lstart];
        int liend = blockMap[lend];

        sublayoutAcc.reserve(liend - listart);
        for (int l = listart; l < liend; l++)
            sublayoutAcc.push_back(layout[l]);

        // Set up C addresses relative to prior blocks.
        for (int l = lstart; l < lend; l++) {
            auto &block = sublayoutExt[l - lstart];
            int bbase = findBaseBlock(
                    block, sublayoutExt, 0, l - lstart, problem.C, strategy.C);
            FOR_EACH_C {
                auto &blockSrc = (bbase >= 0) ? sublayoutExt[bbase] : C_block0;
                auto &addrSrc = (bbase >= 0) ? C_addrs[q][bbase] : C_addr0[q];
                setupAddrRel(C_addrs[q][l - lstart], addrSrc, block, blockSrc,
                        state.C_layout, Tc_ext.size(), state.inputs.ldc[q],
                        problem.C, strategy.C, strategy, state,
                        state.ldcMultiples[q]);
            }
        }

        if (strategy.C.atomic) {
            // Atomic update.
            // Alpha scaling is done earlier; beta scaling isn't supported.
            if (!problem.alpha1() || !problem.beta1()) stub();
            if (copyC)
                if (!copyRegisters(state.Tacc, Tc_ext, sublayoutAcc,
                            sublayoutExt, C_accRange, C_extRange, 0, 0, false,
                            strategy, state))
                    stub();

            auto &sublayoutSrc = copyC ? sublayoutExt : sublayoutAcc;
            auto &C_srcRange = copyC ? C_extRange : C_accRange;
            FOR_EACH_C atomicAddMatrix(Tc_ext, C_srcRange, sublayoutSrc,
                    problem.C, strategy.C, C_addrs[q], problem, strategy,
                    state);
        } else {
            // Data types before and after scaling phase.
            auto Tacc_final = Tc;
            if (op == COperation::UpdateStore && copyC) Tacc_final = state.Tacc;

            // Regular update.
            auto Tload = Tc_ext;
            if (!beta0 || loadOnly) {
                // Set up a0.0 descriptor for loads if needed.
                if (lstart > 0 && haveDescs) mov(1, a0.ud(0), a0.ud(3));

                // Load C data.
                auto &sublayoutLoad
                        = (loadOnly && !copyC) ? sublayoutAcc : sublayoutExt;
                auto &C_loadRange
                        = (loadOnly && !copyC) ? C_accRange : C_extRange;
                loadMatrix(C_loadRange, sublayoutLoad, problem.C, strategy.C,
                        C_addrs[0], strategy, state);

                // Set up a0.0 descriptor for stores (and save load descriptors) if needed.
                if (haveDescs && !loadOnly) {
                    if (lend < nblocks) mov(1, a0.ud(3), a0.ud(0));
                    mov(1, a0.ud(0), a0.ud(2));
                }

                // Copy loaded data as needed.
                if (copyCLoad) {
                    auto &sublayoutDst
                            = loadOnly ? sublayoutAcc : sublayoutCopy;
                    auto &C_dstRange = loadOnly ? C_accRange : C_copyRange;
                    Tload = lateCConvert ? Ts : state.Tacc;
                    if (!copyRegisters(Tc_ext, Tload, sublayoutExt,
                                sublayoutDst, C_extRange, C_dstRange, 0, 0,
                                false, strategy, state))
                        stub();
                }
            }

            // Late C conversion.
            auto originalTacc = state.Tacc;
            if (lateCConvert) {
                for (int li = listart; li < liend; li++) {
                    auto C_acc = state.C_regs[0].subrange(
                            hw, state.Tacc, layout[li]);
                    copyRegisterBlock(state.Tacc, Ts, layout[li], layout[li],
                            C_acc, C_convertRegs[li - listart], 0, 0, strategy,
                            state);
                }
                state.Tacc = Ts;
            }

            // Alpha/beta scaling and optional fp32<->int32 conversion.
            if (!loadOnly)
                for (int phase = 0; phase < 3; phase++) {
                    vector<GRFMultirange> C_accs, C_accSwaps, C_loads;
                    C_accs.reserve(liend - listart);
                    C_accSwaps.reserve(liend - listart);
                    C_loads.reserve(liend - listart);

                    for (int li = listart; li < liend; li++) {
                        GRFMultirange C_acc0 = state.C_regs[0].subrange(
                                hw, state.Tacc, layout[li]);
                        GRFMultirange C_acc = lateCConvert
                                ? C_convertRegs[li - listart]
                                : C_acc0;
                        GRFMultirange C_accSwap;
                        GRFMultirange C_load = beta0
                                ? C_acc
                                : copyCLoad ? C_copyRegs[li - listart]
                                            : C_extRegs[li - listart];
                        switch (phase) {
                            case 0:
                                if (!beta0)
                                    convert(C_load, Tload, state.Tacc, problem,
                                            strategy, state);
                                break;
                            case 1: {
                                C_accs.push_back(C_acc);
                                C_accSwaps.push_back(C_accSwap);
                                C_loads.push_back(C_load);
                            } break;
                            case 2:
                                if (lateCConvert)
                                    copyRegisterBlock(state.Tacc, Tacc_final,
                                            layout[li], layout[li], C_acc,
                                            C_acc0, 0, 0, strategy, state);
                                else
                                    convert(C_acc, state.Tacc, Tacc_final,
                                            problem, strategy, state);
                                break;
                        }
                    }

                    if (phase == 1) {
                        std::vector<int> order(liend - listart);
                        std::iota(order.begin(), order.end(), 0);
                        std::sort(
                                order.begin(), order.end(), [&](int a, int b) {
                                    return C_accs[a][0].getBase()
                                            < C_accs[b][0].getBase();
                                });
                        GRFMultirange C_accsSorted, C_accSwapsSorted,
                                C_loadsSorted;
                        std::vector<RegisterBlock> C_accSortedLayout;

                        bool remaskC_M = isPacked(problem.C.layout)
                                && (strategy.remHandling[LoopM]
                                        != RemainderHandling::Ignore);
                        bool remaskC_N = isPacked(problem.C.layout)
                                && (strategy.remHandling[LoopN]
                                        != RemainderHandling::Ignore);

                        for (int i = 0; i < (liend - listart); i++) {
                            if (remaskC_M || remaskC_N) {
                                auto block = layout[listart + order[i]];
                                block.offsetBytes = C_accsSorted.getLen()
                                        << GRF::log2Bytes(hw);
                                C_accSortedLayout.push_back(block);
                            }

                            C_accsSorted.append(C_accs[order[i]]);
                            C_accSwapsSorted.append(C_accSwaps[order[i]]);
                            C_loadsSorted.append(C_loads[order[i]]);
                        }

                        updateC(C_accsSorted, C_accSwapsSorted, C_loadsSorted,
                                problem, strategy, state);

                        if (remaskC_M)
                            remaskLayout(state.Tacc, 0, false,
                                    C_accSortedLayout, C_accsSorted, strategy,
                                    state);
                        if (remaskC_N)
                            remaskLayout(state.Tacc, 1, true, C_accSortedLayout,
                                    C_accsSorted, strategy, state);
                    }
                }

            state.Tacc = Tacc_final;

            // Store updated data.
            if (op == COperation::UpdateStore) {
                if (copyC)
                    if (!copyRegisters(state.Tacc, Tc_ext, sublayoutAcc,
                                sublayoutExt, C_accRange, C_extRange, 0, 0,
                                false, strategy, state))
                        stub();

                auto &sublayoutSrc = copyC ? sublayoutExt : sublayoutAcc;
                auto &C_srcRange = copyC ? C_extRange : C_accRange;
                FOR_EACH_C storeMatrix(C_srcRange, sublayoutSrc, problem.C,
                        strategy.C, C_addrs[q], strategy, state);
            }

            state.Tacc = originalTacc;
        }

        // Free address and data registers, including C accumulators that are no longer used...
        //  ... except C_addr0. I need that!
        FOR_EACH_C safeReleaseRanges(C_addrs[q], state);
        safeReleaseRanges(C_extRange, state);
        safeReleaseRanges(C_copyRange, state);
        safeReleaseRanges(C_convertRegs, state);
        if (op == COperation::UpdateStore)
            for (int li = listart; li < liend; li++)
                for (int b = 0; b < state.C_buffers; b++)
                    releaseRanges(state.C_regs[b].subrange(
                                          hw, state.Tacc, layout[li]),
                            state);
        FOR_EACH_C state.ra.claim(C_addr0[q]);

        // Check for forward progress.
        if (lend == lstart) throw out_of_registers_exception();
        lstart = lend;
    }

    // Re-claim all the C registers we freed, so as not to disturb the caller's RegisterAllocator.
    reclaimRanges(state.C_regs[0], state);
#undef FOR_EACH_C
}

// Assign runtime-computed descriptor information to all blocks in this layout.
// Returns true if successful; false if not all blocks in layout are compatible.
static inline bool assignAllDescs(vector<RegisterBlock> &layout) {
    for (auto &block : layout) {
        if (block.simdSize != layout[0].simdSize) return false;
        block.descAssigned = true;
        block.sfid = layout[0].sfid;
    }

    return true;
}

// Output code for standard C remainder handling.
template <HW hw>
bool gemm_kernel_generator_t<hw>::doStdCRemainder(
        vector<RegisterBlock> &layoutExt,
        vector<RegisterBlock> &layoutExtUnmasked, bool inside, bool columns[2],
        StdCRemType remTypes[2], bool fragments[2], bool fragPositives[2],
        int fragSizes[2], const GRFRange (&C_addr0)[2],
        const GRFRange (&C_addr0Unmasked)[2], COperation op,
        vector<MaskAssignment> &masks, GEMMProblem &problem,
        GEMMStrategy &strategy, GEMMState state) {
    auto Tc_ext = problem.Tc_ext;
    auto column = columns[inside];
    LoopType loop = column ? LoopN : LoopM;
    auto remType = remTypes[loop];
    auto fragment = fragments[loop];
    auto fragPositive = fragPositives[loop];
    auto fragSize = fragSizes[loop];
    auto unroll = strategy.unroll[loop];
    auto remainder = state.remainders[loop];

    bool canEOT = !state.isNested && (op == COperation::UpdateStore);

    Label lEnd;

    // The "q" dimension is the one whose remainder we are currently handling.
    auto RegisterBlock::*nq = column ? &RegisterBlock::nc : &RegisterBlock::nr;
    auto RegisterBlock::*offsetQ
            = column ? &RegisterBlock::offsetC : &RegisterBlock::offsetR;

    // Status message.
    status << "C remainder handling (" << char('m' + column) << ") " << remType;
    if (fragment) status << ", fragment";
    if (fragPositive) status << ", no empty accesses";
    status << status_stream::endl;

    // Allocate temporaries for emulated atomic addition if needed.
    if (!inside && strategy.C.atomic)
        allocEAtomicAddRegs(
                hw, Tc_ext, layoutExt, problem.C, strategy.C, state);

    // Handle a subproblem. Return true if successful.
    auto descend = [&](vector<RegisterBlock> &sublayoutExt,
                           vector<RegisterBlock> &sublayoutExtUnmasked,
                           bool full = false) -> bool {
        bool success = true;
        auto nMasksOriginal = int(masks.size());

        if (remType == StdCRemType::Mask) {
            if (!full) {
                // Assign and load any extra masks needed.
                if (!assignMasks(sublayoutExt, LoopM, LoopN, masks, state))
                    return false;
                loadMasks(masks, state.remainders, strategy, state,
                        nMasksOriginal);
                sublayoutExtUnmasked.clear();
            } else {
                // Clear out mask assignments in this dimension.
                for (auto &block : layoutExt)
                    block.clearFlag();
            }
        }

        // Recursively handle subproblem.
        if (!inside)
            success = doStdCRemainder(sublayoutExt, sublayoutExtUnmasked, true,
                    columns, remTypes, fragments, fragPositives, fragSizes,
                    C_addr0, C_addr0Unmasked, op, masks, problem, strategy,
                    state);
        else if (sublayoutExtUnmasked.empty())
            updateCLayout(sublayoutExt, C_addr0, state.C_layoutExt[0], op,
                    problem, strategy, state);
        else
            updateCLayout(sublayoutExtUnmasked, C_addr0Unmasked,
                    state.C_layoutExtUnmasked[0], op, problem, strategy, state);

        // Free any new masks.
        if (remType == StdCRemType::Mask)
            safeReleaseMaskAssignments(masks, state, nMasksOriginal);
        return success;
    };

    // Exit remainder handling.
    auto done = [&]() {
        if (!canEOT)
            jmpi(1, lEnd);
        else
            epilogue(strategy, state);
    };

    // Main code.
    bool success = false;
    pushStream();

    if (!fragment) {
        // If descriptor-based remainders requested, all blocks should be smaller than fragSize.
        // Load descriptors based on total remainder in this (rare) case.
        if (remType == StdCRemType::Descriptor) {
            loadLoadStoreDescriptors(!problem.beta0(), true, layoutExt[0],
                    remainder, problem.C, strategy.C, strategy, state);
            if (!assignAllDescs(layoutExt)
                    || !assignAllDescs(layoutExtUnmasked))
                goto failed;
        }
        if (inside && !layoutExtUnmasked.empty()
                && layoutExt.size() == state.C_layoutExt.size()) {
            // If unmasked layout is available, implement full remainder case specially.
            const bool useSIMTFlow = strategy.fused
                    && (strategy.fusedLoop == loop
                            || strategy.fusedLoop == LoopAny);
            Label labelRem, labelDone;

            if (useSIMTFlow) {
                cmp(16 | ge | state.flagAP, remainder, unroll);
                if_(16 | state.flagAP, labelRem, labelDone);
            } else if (strategy.fused) {
                cmp(1 | ge | state.flagAP, remainder, unroll);
                jmpi(1 | ~state.flagAP, labelRem);
            } else {
                // No flag registers guaranteed -- use a jump table.
                auto tempQ = state.ra.alloc_sub<uint64_t>();
                auto temp = tempQ.ud(0);

                add(1 | sat, temp, remainder, -unroll + 1);
                isGen12 ? mad(1, temp, 16, temp, 16) : shl(1, temp, temp, 4);
                jmpi(1, temp.d());
                jmpi(1, labelRem);

                state.ra.safeRelease(tempQ);
            }

            status << "Code for full " << char('m' + column) << " remainder"
                   << status_stream::endl;
            if (!descend(layoutExt, layoutExtUnmasked, true)) goto failed;

            useSIMTFlow ? else_(16, labelDone) : jmpi(1, labelDone);
            mark(labelRem);

            status << "Code for generic " << char('m' + column) << " remainder"
                   << status_stream::endl;
            if (!descend(layoutExt, layoutExtUnmasked)) goto failed;

            mark(labelDone);
            if (useSIMTFlow) endif(16);
        } else {
            // Otherwise, nothing else to do: go down a level.
            if (!descend(layoutExt, layoutExtUnmasked)) goto failed;
        }
    } else {
        // Use SIMT control flow if remainders could be different between fused threads or if jump tables disabled.
        const bool useSIMTFlow = strategy.noJumpTables
                || (strategy.fused
                        && (strategy.fusedLoop == loop
                                || strategy.fusedLoop == LoopAny));

        // Fix up fragment size (fragSize).
        //  - Check that every block starts at a multiple of fragSize; if not fall back on fragSize 1.
        //  - Max fragment size is 16.
        //  - Should check unmasked layout, but it will have the same kind of fragmenting as the masked layout.
        fragSize = std::min<int>(fragSize, 16);
        for (auto &block : layoutExt) {
            if (block.*offsetQ % fragSize) {
                fragSize = 1;
                break;
            }
        }

        // There are two strategies for fragmenting for remainder handling:
        //    fragSize = 1:  Try to get the largest blocks as possible. These are always fragPositive.
        //    fragSize > 1:  Always use blocks of size fragSize in the q dimension.
        if (fragSize == 1) {
            if (!useSIMTFlow) {
                // SIMD control flow, using a jump table.
                Subregister temp = state.ra.alloc_sub<uint32_t>();
                vector<Label> rlabels(unroll);

                // Generate jump table.
                shl(1, temp, remainder,
                        uint16_t(4)); // Multiply by instruction length.
                if (isGen12) // Gen12+ jmpi is relative to current IP.
                    add(1, temp, temp, uint16_t(16));
                jmpi(1, temp.d()); // Indexed jump into jump table.
                for (int r = 0; r < unroll; r++)
                    jmpi(1, rlabels[r]);

                // Full remainder case: continue downward.
                status << "Code for full " << char('m' + column) << " remainder"
                       << status_stream::endl;
                if (!descend(layoutExt, layoutExtUnmasked, true)) goto failed;
                inside ? jmpi(1, rlabels[0]) : done();

                // Remainder handling.
                vector<bool> qdone(unroll, false);
                qdone[0] = true;
                int qnext = 0;
                for (int nqtodo = unroll - 2; nqtodo >= 0; nqtodo--) {
                    // Decide which q to do.
                    int q;
                    if (qnext > 0)
                        q = qnext;
                    else {
                        for (q = unroll - 1; q >= 0; q--)
                            if (!qdone[q]) break;
                    }

                    status << "Code for " << char('m' + column) << " remainder "
                           << q << status_stream::endl;

                    mark(rlabels[q]);

                    // Figure out how many rows/columns to take.
                    int chunkSize = q & ~(q - 1); // = 1 << lowest set bit

                    // Look through all blocks in this row/column, and reduce chunk size if appropriate.
                    for (auto &block : layoutExt) {
                        if (!block.isLoadBlock())
                            stub(); // Dummy blocks should be replaced by real ones...
                        int qq = q
                                - block.*offsetQ; // Note q = 1 + last row/column.
                        if (qq > 0 && qq <= block.*nq)
                            chunkSize = std::min<int>(chunkSize, qq);
                    }

                    // With chunk size chosen, get rows/columns [q - chunkSize, q) of intersecting blocks.
                    vector<RegisterBlock> C_subblocksExt,
                            C_subblocksExtUnmasked;
                    if (!getSubblocks(Tc_ext, C_subblocksExt, layoutExt, column,
                                q - chunkSize, q, false, problem.C, strategy.C))
                        goto failed;
                    if (!layoutExtUnmasked.empty())
                        if (!getSubblocks(Tc_ext, C_subblocksExtUnmasked,
                                    layoutExtUnmasked, column, q - chunkSize, q,
                                    false, problem.C, strategy.C))
                            goto failed;

                    // Perform the requested update.
                    if (!descend(C_subblocksExt, C_subblocksExtUnmasked))
                        goto failed;

                    // Go to next remainder handler, or return.
                    qdone[q] = true;
                    qnext = q - chunkSize;
                    if (nqtodo > 0) {
                        if (qnext == 0 && canEOT)
                            epilogue(strategy, state);
                        else if (qdone[qnext]) {
                            jmpi(1, rlabels[qnext]);
                            qnext = 0;
                        }
                    }
                }
                mark(rlabels[0]);

                state.ra.safeRelease(temp);
            } else {
                // SIMT control flow: massively nested if-else.

                // Handle remainder in the range [q0, q1).
                std::function<bool(int, int)> handleRemainder
                        = [&](int q0, int q1) -> bool {
                    Label labelElse, labelEndif;

                    int qChunk = rounddown_pow2(q1 - q0 - 1);

                    if (qChunk == 0) qChunk = 1;

                    status << "Code for " << char('m' + column)
                           << " remainders " << q0 << " - " << (q1 - 1)
                           << status_stream::endl;

                    if (q1 - q0 > 1) {
                        cmp(16 | ge | state.flagAP, remainder,
                                uint16_t(q0 + qChunk));
                        if_(16 | state.flagAP,
                                (qChunk > 1) ? labelElse : labelEndif,
                                labelEndif);
                    }

                    vector<RegisterBlock> C_subblocksExt,
                            C_subblocksExtUnmasked;
                    if (!getSubblocks(Tc_ext, C_subblocksExt, layoutExt, column,
                                q0, q0 + qChunk, false, problem.C, strategy.C))
                        return false;
                    if (!layoutExtUnmasked.empty())
                        if (!getSubblocks(Tc_ext, C_subblocksExtUnmasked,
                                    layoutExtUnmasked, column, q0, q0 + qChunk,
                                    false, problem.C, strategy.C))
                            return false;

                    if (!descend(C_subblocksExt, C_subblocksExtUnmasked))
                        return false;

                    if (q1 - q0 > 1) {
                        if (qChunk > 1) {
                            if (!handleRemainder(q0 + qChunk, q1)) return false;

                            else_(16, labelEndif);
                            mark(labelElse);

                            if (!handleRemainder(q0, q0 + qChunk)) return false;
                        }

                        mark(labelEndif);
                        endif(16);
                    }

                    return true;
                };

                Label labelRem, labelRemDone, labelDone;

                cmp(16 | ge | state.flagAP, remainder, uint16_t(unroll));
                if_(16 | state.flagAP, labelRem, labelDone);

                status << "Code for " << char('m' + column) << " full remainder"
                       << status_stream::endl;
                if (!descend(layoutExt, layoutExtUnmasked, true)) goto failed;

                else_(16, labelDone);
                mark(labelRem);

                if (!handleRemainder(0, unroll)) goto failed;

                mark(labelDone);
                endif(16);
                setDefaultNoMask(true);
            }
        } else {
            auto handleRemainderFP = [&](int q0, int q1) -> bool {
                // Get rows/columns [q0, q1) of intersecting blocks.
                vector<RegisterBlock> C_subblocksExt, C_subblocksExtUnmasked;
                if (!getSubblocks(Tc_ext, C_subblocksExt, layoutExt, column, q0,
                            q1, false, problem.C, strategy.C))
                    return false;
                if (!layoutExtUnmasked.empty())
                    if (!getSubblocks(Tc_ext, C_subblocksExtUnmasked,
                                layoutExtUnmasked, column, q0, q1, false,
                                problem.C, strategy.C))
                        return false;

                if (remType == StdCRemType::Descriptor) {
                    // Load address registers for subsequent loads and stores.
                    Subregister rcount = state.ra.alloc_sub<uint32_t>();
                    Subregister mremainder = remainder;

                    if (q0 != 0) {
                        add(1 | sat, rcount, mremainder, int16_t(-q0));
                        mremainder = rcount;
                    }
                    if (q1 < unroll) {
                        min_(1, rcount, mremainder, uint16_t(fragSize));
                        mremainder = rcount;
                    }

                    loadLoadStoreDescriptors(!problem.beta0(), true,
                            C_subblocksExt[0], mremainder, problem.C,
                            strategy.C, strategy, state);
                    if (!assignAllDescs(C_subblocksExt)
                            || !assignAllDescs(C_subblocksExtUnmasked))
                        return false;

                    state.ra.safeRelease(rcount);
                }

                // Perform the requested update.
                return descend(C_subblocksExt, C_subblocksExtUnmasked);
            };

            if (!useSIMTFlow) {
                // SIMD control flow, possibly using a jump table.
                int N = div_up(unroll, fragSize);
                vector<Label> rlabels(N); // Targets for jump table.
                Label rdone;

                // Create a jump table, if needed.
                if (fragPositive) {
                    Subregister t1 = state.ra.alloc_sub<uint32_t>();
                    Subregister t2 = state.ra.alloc_sub<uint32_t>();

                    add(1 | sat, t2, remainder, int16_t(-unroll + 1));
                    add(1, t1, remainder,
                            int16_t(-1 + (isGen12 ? fragSize : 0)));
                    add(1, t1, t1,
                            t2); // Increment index if remainder == unroll.
                    if (fragSize < 16) // Precondition: fragSize <= 16.
                        mulConstant(1, t1, t1,
                                16 / fragSize); // Multiply by instruction length (16b/uncompacted instruction)
                    and_(1, t1, t1,
                            uint16_t(0xFFF0)); // Mask off unwanted bits.
                    jmpi(1, t1.d()); // Indexed jump into jump table.
                    for (int r = 0; r < N; r++)
                        jmpi(1, rlabels[r]);

                    state.ra.safeRelease(t2);
                    state.ra.safeRelease(t1);
                }

                // Full loop.
                status << "Code for " << char('m' + column) << " full remainder"
                       << status_stream::endl;
                if (!descend(layoutExt, layoutExtUnmasked, true)) goto failed;
                inside ? jmpi(1, rdone) : done();

                // Remainder handling.
                for (int r = N - 1; r >= 0; r--) {
                    int q0 = r * fragSize;
                    int q1 = std::min<int>(q0 + fragSize, unroll);

                    status << "Code for " << char('m' + column)
                           << " remainders " << q0 + 1 << " - " << q1
                           << status_stream::endl;

                    mark(rlabels[r]);

                    if (!handleRemainderFP(q0, q1)) goto failed;
                }

                if (inside) mark(rdone);
            } else {
                // SIMT control flow version.
                Label labelRem, labelRemDone, labelDone;

                cmp(16 | ge | state.flagAP, remainder, uint16_t(unroll));
                if_(16 | state.flagAP, labelRem, labelDone);

                status << "Code for " << char('m' + column) << " full remainder"
                       << status_stream::endl;
                if (!descend(layoutExt, layoutExtUnmasked, true)) goto failed;

                else_(16, labelDone);
                mark(labelRem);

                for (int q0 = 0; q0 < unroll; q0 += fragSize) {
                    int q1 = std::min<int>(q0 + fragSize, unroll);

                    cmp(16 | le | state.flagAP, remainder, uint16_t(q0));
                    goto12(16 | state.flagAP, labelRemDone);
                    status << "Code for " << char('m' + column)
                           << " remainders " << q0 + 1 << " - " << q1
                           << status_stream::endl;

                    if (!handleRemainderFP(q0, q1)) goto failed;
                }

                mark(labelRemDone);
                join(16);

                mark(labelDone);
                endif(16);
            }
        }
    }

    // Success!
    success = true;
failed:

    mark(lEnd);
    success ? appendCurrentStream() : discardStream();

    if (!inside && strategy.C.atomic) freeEAtomicAddRegs(state);

    return success;
}

// Alternate code path for C remainder handling, based on a simple double loop
//  and indirect addressing.
template <HW hw>
void gemm_kernel_generator_t<hw>::doAlternateCRemainder(COperation op,
        GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state) {
    auto Tc = problem.Tc, Tc_ext = problem.Tc_ext;
    int C_count = (op == COperation::UpdateStore) ? state.C_count : 1;
#define FOR_EACH_C for (int q = 0; q < C_count; q++)
#define FOR_EACH_C_REV for (int q = C_count - 1; q >= 0; q--)

    bool lateYLoopCheck = false;

    bool surface = !strategy.C.base.isStateless();
    bool loadOnly = (op == COperation::Load);

    // Vector length in inner loop.
    const auto nbytes = 64;
    auto nec = nbytes / Tc;

    // 1- and 2-byte types must be padded to 4 bytes.
    bool byte_access = (Tc_ext.size() < 4);
    if (byte_access) nec = nbytes >> 2;

    // 8-byte+ types can use scattered qword. Only atomic for now.
    bool nativeAtomic = strategy.C.atomic
            && hasNativeAtomicAdd(hw, Tc_ext.real(), problem.C, strategy.C);
    bool qword = !((nativeAtomic ? Tc_ext.real() : Tc_ext).size() & 7)
            && strategy.C.atomic;
    int rshift = qword ? 3 : 2; // log2(data stride in regs)
    int rsimd = 64 >> rshift;

    auto &block0 = state.C_layout[0];
    bool cColMajorMem = isColMajor(problem.C.layout);
    bool cColMajorReg = block0.colMajor;
    bool transpose = (cColMajorReg != cColMajorMem);
    if (isPacked(problem.C.layout)) stub();

    // x is the contiguous dimension (in registers), y is the other dimension.
    auto LoopX = cColMajorReg ? LoopM : LoopN;
    auto LoopY = cColMajorReg ? LoopN : LoopM;
    int unrollX = strategy.unroll[LoopX];
    int unrollY = strategy.unroll[LoopY];

    // Check the layout:
    //  - C is a contiguous block of registers.
    //  - nx must be divisible by 2 (unpacked) GRFs, unless x unroll is < 2 GRFs,
    //      or there's an extra GRF at the end of C.
    //  - register offsets must be in a uniform 2D grid
    //  - all blocks must share same ordering (row/column major).
    // Otherwise use non-uniform path, and indirectly load GRFs.

    auto Tcx = Tc;
    bool uniform = true;
    int16_t xByteInc = 0, yByteInc = 0;
    bool cAtEnd = (state.C_regs[0][state.C_regs[0].getLen() - 1].getBase() + 1)
            >= strategy.GRFs;

    if (state.C_regs[0].ranges.size() != 1) uniform = false;

    for (auto &block : state.C_layout) {
        if (block.colMajor != block0.colMajor) stub();

        int nx = cColMajorReg ? block.nr : block.nc;
        int ny = cColMajorReg ? block.nc : block.nr;
        int ox = cColMajorReg ? block.offsetR : block.offsetC;
        int oy = cColMajorReg ? block.offsetC : block.offsetR;

        ox /= nec;

        if ((nx & (nec - 1)) && cAtEnd) uniform = false;

        if (xByteInc == 0 && nx > nec) xByteInc = nec * Tcx;
        if (yByteInc == 0 && ny > 1) yByteInc = block.ld * Tc;

        if (block.offsetBytes != ox * xByteInc + oy * yByteInc) {
            if (xByteInc == 0 && ox > 0)
                xByteInc = (block.offsetBytes - oy * yByteInc) / ox;
            else if (yByteInc == 0 && oy > 0)
                yByteInc = (block.offsetBytes - ox * xByteInc) / oy;
            else
                uniform = false;
        }
    }

    GRFRange bases;
    bool nonuniformSubs = false;

    if (!uniform) {
        uint8_t baseIndices[256] = {0};
        uint16_t offIndices[256] = {0};

        if (state.Tacc.size() == 1) stub();

        xByteInc = div_up(nec * Tcx, GRF::bytes(hw));
        int nec1 = nec / xByteInc;
        yByteInc = div_up(unrollX, nec1);

        for (int y = 0; y < unrollY; y++) {
            for (int xx = 0; xx < yByteInc; xx++) {
                auto x = xx * nec1;
                auto i = cColMajorReg ? x : y;
                auto j = cColMajorReg ? y : x;
                const RegisterBlock *blockPtr;
                int ne;
                auto sub = findBlockReg(Tc, state.C_layout, i, j,
                        state.C_regs[0], ne, blockPtr, 0);
                nonuniformSubs |= (sub.getOffset() != 0);
                if (ne < std::min(nec1, unrollX - x)) stub();
                baseIndices[y * yByteInc + xx] = sub.getBase();
                offIndices[y * yByteInc + xx]
                        = sub.getByteOffset() + sub.getBase() * GRF::bytes(hw);
            }
        }

        if (nonuniformSubs) {
            xByteInc *= 2;
            yByteInc *= 2;
        }

        bases = state.ra.alloc_range(
                div_up(unrollY * yByteInc, GRF::bytes(hw)));
        bool haveDF = !strategy.emulate.emulate64;
        haveDF |= (hw == HW::XeHPC);
        if (haveDF) {
            for (int i = 0; i < unrollY * yByteInc; i += 8) {
                auto sub = bases[i / GRF::bytes(hw)].df(
                        (i % GRF::bytes(hw)) / 8);
                auto data = nonuniformSubs
                        ? reinterpret_cast<double *>(&offIndices[i / 2])
                        : reinterpret_cast<double *>(&baseIndices[i]);
                mov(1, sub, *data);
            }
        } else {
            for (int i = 0; i < unrollY * yByteInc; i += 4) {
                auto sub = bases[i / GRF::bytes(hw)].ud(
                        (i % GRF::bytes(hw)) / 4);
                auto data = nonuniformSubs
                        ? reinterpret_cast<uint32_t *>(&offIndices[i / 2])
                        : reinterpret_cast<uint32_t *>(&baseIndices[i]);
                mov(1, sub, *data);
            }
        }
    }

    // Claim flags.
    auto saveFlagAP = state.flagAP;
    state.raVFlag.safeRelease(state.flagAP);
    state.raVFlag.claim(f0[0]);
    state.raVFlag.claim(f0[1]);
    state.raVFlag.claim(f1[0]);

    // Clear f0[1] for any16h trick.
    if (strategy.fused && !lateYLoopCheck) mov(1, f0[1], uint16_t(0));

    // Update C with scattered accesses.
    // Get mask and set up header.
    GRFRange header[2];
    auto hregs = (surface ? 1 : 2) * (qword ? 1 : 2);
    FOR_EACH_C header[q] = state.ra.alloc_range(hregs);
    Subregister temp = state.ra.alloc_sub<uint32_t>();
    Subregister mask = state.ra.alloc_sub<uint32_t>();
    Subregister xIndex = state.remainders[LoopX];

    GRF indexVec, ivContig, ivScatter;

    indexVec = state.ra.alloc();
    indexVec.setType(DataType::w);
    mov(8, indexVec[0](1), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
    if (rsimd > 8)
        mov(8, indexVec[8](1), Immediate::uv(8, 9, 10, 11, 12, 13, 14, 15));

    auto oshift = std::min<int>(rshift, Tc_ext.log2Size());

    // Prepare x mask in f1.0 and prepare header for loads/stores.
    if (Tc_ext.size() > 4) {
        mulConstant(1, temp, xIndex, uint16_t(Tc_ext.size() >> rshift));
        xIndex = temp;
    }

    ivScatter = indexVec;
    bool splitScatter = transpose && (Tc_ext.log2Size() > rshift);
    if (splitScatter) {
        ivContig = state.ra.alloc();
        ivContig.setType(DataType::w);
        auto shift = Tc_ext.log2Size() - rshift;
        auto m = (1 << shift) - 1;

        asr(16, ivScatter, indexVec, uint16_t(shift));
        mov(16, ivContig,
                Immediate::uv((0 & m) << rshift, (1 & m) << rshift,
                        (2 & m) << rshift, (3 & m) << rshift, (4 & m) << rshift,
                        (5 & m) << rshift, (6 & m) << rshift,
                        (7 & m) << rshift));
    }

    add(1, temp, xIndex, int16_t(-1));
    FOR_EACH_C transpose
            ? mul(rsimd, header[q][0].d(), state.inputs.ldc[q], ivScatter)
            : shl(rsimd, header[q][0].d(), indexVec, uint16_t(oshift));
    FOR_EACH_C if (splitScatter)
            add(rsimd, header[q][0].d(), header[q][0].d(), ivContig);

    int hs = 1;
    bool header4 = !qword && !surface;
    int neq = elementsPerGRF(hw, DataType::uq);

    header4 &= (GRF::bytes(hw) < 64);
    if (hw >= HW::XeHP && !surface) {
        if (header4)
            FOR_EACH_C mov<uint32_t>(2 * neq, header[q][2][0](2), header[q][1]);
        FOR_EACH_C mov<uint32_t>(neq, header[q][1][0](2), header[q][0][neq](1));
        FOR_EACH_C mov<uint32_t>(neq, header[q][0][0](2), header[q][0][0](1));
        hs = 2;
    }

    and_(1, temp, ~temp, uint16_t(rsimd - 1));
    FOR_EACH_C surface
            ? add(rsimd, header[q][0].d(), header[q][0].d(), state.effC[q])
            : header4 ? eadd(8, header[q][2].uq(), header[q][hs].d(0)(hs),
                      state.effC[q], strategy, state)
                      : noop();
    mov(1, mask, uint16_t((1 << rsimd) - 1));
    FOR_EACH_C if (!surface) eadd(2 * neq, header[q][0].uq(),
            header[q][0].d(0)(hs), state.effC[q], strategy, state);
    shr(1, f1[0], mask, temp);

    state.ra.safeRelease(mask);
    state.ra.safeRelease(temp);
    state.ra.safeRelease(ivContig);

    // Synthesize double loop updating 2 GRFs (indirectly addressed) at a time.
    GRF ix = state.ra.alloc();
    Subregister ix_init = state.ra.alloc_sub<uint16_t>();
    Subregister iy = state.ra.alloc_sub<int16_t>();
    Subregister cXInc[2], cYInc[2];
    FOR_EACH_C cYInc[q] = state.ra.alloc_sub<int32_t>();
    Label yLoop, xLoop;
    GRFRange Cacc = state.ra.alloc_range(2);
    GRFRange CaccSwap {};
    GRFRange Cload
            = state.ra.alloc_range(2, getHint(HintType::CLoad, strategy));

    if (transpose) FOR_EACH_C {
            cXInc[q] = state.ra.alloc_sub<int32_t>();
            mulConstant(1, cXInc[q], state.inputs.ldc[q], nec);
        }

    add(1, ix_init, state.remainders[LoopX], int16_t(-1));
    mov(1, iy, state.remainders[LoopY]);
    shr(1, ix_init, ix_init, uint16_t(log2(nec)));

    if (uniform)
        mov(1, a0[0], state.C_regs[0][0].getBase() * GRF::bytes(hw));
    else
        mov(1, a0[0], bases.getBase() * GRF::bytes(hw));

    add(1, cYInc[0], ix_init, uint16_t(1));
    mulConstant(1, cYInc[0], cYInc[0],
            uint16_t(nec * (!transpose ? Tc_ext.size() : 1)));
    if (!transpose)
        FOR_EACH_C_REV add(1, cYInc[q], -cYInc[0], state.inputs.ldc[q]);
    else {
        FOR_EACH_C_REV mul(1, cYInc[q], state.inputs.ldc[q], cYInc[0].w());
        FOR_EACH_C_REV add(1, cYInc[q], -cYInc[q], uint16_t(Tc_ext.size()));
    }

    mark(yLoop);
    mov<uint16_t>(16, ix, ix_init);
    if (!lateYLoopCheck) add(1 | gt | f0[1], iy, iy, int16_t(-1));
    mov(1, a0[1], a0[0]);

    mark(xLoop);
    add<int16_t>(16 | ge | f0[0], ix, ix, int16_t(-1));

    // Update. The anyv is a trick to use the generated m mask (f1.0) on the last
    //  iteration of the loop, and no mask (0xFFFF) on the other iterations.
    InstructionModifier mod;
    mod = mod | f0[0] | anyv;

    // Alas, no anyv on PVC.
    if (hw == HW::XeHPC) {
        mov(1 | ~f0[0], f0[0], f1[0]);
        mod = InstructionModifier() | f0[0];
    }

    if (!uniform) {
        nonuniformSubs ? mov(xByteInc, a0[2](1), indirect[a0[1]].uw())
                       : shl(xByteInc, a0[2](1), indirect[a0[1]].ub(),
                               GRF::log2Bytes(hw));
    }

    if (!loadOnly) {
        if (uniform) switch (state.Tacc.size()) {
                case 1: mov<uint32_t>(16, Cacc, indirect[a0[1]].ub()); break;
                case 2: mov<uint32_t>(16, Cacc, indirect[a0[1]].uw()); break;
                default: mov<uint32_t>(16, Cacc, indirect[a0[1]]); break;
            }
        else if (xByteInc == 1)
            switch (state.Tacc.size()) {
                case 2: mov<uint32_t>(16, Cacc, indirect[a0[2]].uw()); break;
                default: mov<uint32_t>(16, Cacc, indirect[a0[2]]); break;
            }
        else
            switch (state.Tacc.size()) {
                case 2:
                    mov<uint32_t>(
                            16, Cacc, indirect[a0[2]].uw(0)(16 / xByteInc, 1));
                    break;
                default:
                    mov<uint32_t>(
                            16, Cacc, indirect[a0[2]].ud(0)(16 / xByteInc, 1));
                    break;
            }
    }

    if (strategy.C.atomic) {
        // Atomic update. Requires beta = 1, alpha prescaled.
        if (!problem.alpha1() && !problem.beta1()) stub();
        if (C_count > 1) stub();
        if (op != COperation::UpdateStore) stub();

        std::vector<RegisterBlock> layout {1};
        auto &block = layout[0];
        block.ebytes = qword ? 8 : Tc_ext.real().size();
        block.simdSize = rsimd;
        block.clearFlag();
        block.bytes = 64;
        block.extra = 1;
        block.count = 1;
        block.log2GRFBytes = GRF::log2Bytes(hw);

        allocEAtomicAddRegs(
                hw, Tc_ext, layout, problem.C, strategy.C, state, f1[1]);

        Label labelEndAtomic;
        if_(16 | mod, labelEndAtomic);
        setDefaultNoMask(false);
        atomicAddMatrixBlock(Tc_ext, Cacc, block, problem.C, strategy.C,
                header[0], problem, strategy, state);
        setDefaultNoMask(true);
        mark(labelEndAtomic);
        endif(16);

        freeEAtomicAddRegs(state, f1[1]);
    } else {
        // Late C conversion, if needed.
        auto originalTacc = state.Tacc;
        if (problem.needsTsConvert() && state.Tacc != problem.Ts) {
            convert(Cacc, state.Tacc, problem.Ts, problem, strategy, state);
            state.Tacc = problem.Ts;
        }

        // Regular update.
        if (loadOnly || !problem.beta0()) {
            doReadSuppressionWA(strategy, state);
            if (strategy.C.newDP) {
                !byte_access ? load(16 | mod, Cload, D32 | strategy.C.cachingR,
                        strategy.C.base, header[0])
                             : (Tc_ext.size() == 2)
                                ? load(16 | mod, Cload,
                                        D16U32 | strategy.C.cachingR,
                                        strategy.C.base, header[0])
                                : load(16 | mod, Cload,
                                        D8U32 | strategy.C.cachingR,
                                        strategy.C.base, header[0]);
            } else {
                byte_access
                        ? load(16 | mod, Cload, scattered_byte(Tc_ext.size()),
                                strategy.C.base, header[0])
                        : !surface ? load(16 | mod, Cload, scattered_dword(),
                                  strategy.C.base, header[0])
                                   : load(16 | mod, Cload,
                                           surface_dword(ChannelMask::r),
                                           strategy.C.base, header[0]);
            }
        }

        if (!loadOnly) {
            auto Tc_out = (op == COperation::UpdateStore) ? problem.Tc_ext
                                                          : problem.Tc;
            if (!problem.beta0())
                convert(Cload, problem.Tc_ext, state.Tacc, problem, strategy,
                        state);
            updateC(Cacc, CaccSwap, Cload, problem, strategy, state);
            convert(Cacc, state.Tacc, Tc_out, problem, strategy, state);
        }

        if (op != COperation::UpdateStore) {
            auto src = (op == COperation::Load) ? Cload : Cacc;
            if (uniform) switch (Tc.size()) {
                    case 1:
                        mov<uint32_t>(16 | mod, indirect[a0[1]].ub(), src);
                        break;
                    case 2:
                        mov<uint32_t>(16 | mod, indirect[a0[1]].uw(), src);
                        break;
                    default:
                        mov<uint32_t>(16 | mod, indirect[a0[1]], src);
                        break;
                }
            else if (xByteInc == 1)
                switch (state.Tacc.size()) {
                    case 2:
                        mov<uint32_t>(16 | mod, indirect[a0[2]].uw(), src);
                        break;
                    default:
                        mov<uint32_t>(16 | mod, indirect[a0[2]], src);
                        break;
                }
            else
                switch (state.Tacc.size()) {
                    case 2:
                        mov<uint32_t>(16 | mod,
                                indirect[a0[2]].uw(0)(16 / xByteInc, 1), src);
                        break;
                    default:
                        mov<uint32_t>(16 | mod,
                                indirect[a0[2]].ud(0)(16 / xByteInc, 1), src);
                        break;
                }
        } else
            FOR_EACH_C {
                if (strategy.C.newDP) {
                    !byte_access ? store(16 | mod, D32 | strategy.C.cachingW,
                            strategy.C.base, header[q], Cacc)
                                 : (Tc_ext.size() == 2)
                                    ? store(16 | mod,
                                            D16U32 | strategy.C.cachingW,
                                            strategy.C.base, header[q], Cacc)
                                    : store(16 | mod,
                                            D8U32 | strategy.C.cachingW,
                                            strategy.C.base, header[q], Cacc);
                } else {
                    byte_access ? store(16 | mod, scattered_byte(Tc_ext.size()),
                            strategy.C.base, header[q], Cacc)
                                : !surface
                                    ? store(16 | mod, scattered_dword(),
                                            strategy.C.base, header[q], Cacc)
                                    : store(16 | mod,
                                            surface_dword(ChannelMask::r),
                                            strategy.C.base, header[q], Cacc);
                }
            }

        state.Tacc = originalTacc;
    }

    if (hw == HW::XeHPC) cmp<int16_t>(1 | ge | f0[0], ix, 0);

    add(1, a0[1], a0[1], xByteInc);
    if (!transpose) {
        uint16_t inc = nec * Tc_ext;
        if (!surface) {
            FOR_EACH_C eadd<uint64_t>(std::min(2 * neq, rsimd), header[q][0],
                    header[q][0], inc, strategy, state);
            if (header4)
                FOR_EACH_C eadd<uint64_t>(
                        8, header[q][2], header[q][2], inc, strategy, state);
        } else
            FOR_EACH_C add<uint32_t>(rsimd, header[q][0], header[q][0], inc);
    } else {
        if (!surface) {
            FOR_EACH_C eadd<uint64_t>(std::min(2 * neq, rsimd), header[q][0],
                    header[q][0], cXInc[q], strategy, state);
            if (header4)
                FOR_EACH_C eadd<uint64_t>(8, header[q][2], header[q][2],
                        cXInc[q], strategy, state);
        } else
            FOR_EACH_C add<uint32_t>(
                    rsimd, header[q][0], header[q][0], cXInc[q]);
    }

    // Bottom of x loop.
    //  Fused threads must use SIMT control flow instructions.
    strategy.fused ? simtDoWhileLoop(16 | f0[0], xLoop)
                   : jmpi(1 | f0[0], xLoop);

    if (lateYLoopCheck) add(1 | gt | f0[1], iy, iy, int16_t(-1));
    add(1, a0[0], a0[0], yByteInc);
    if (!surface) {
        FOR_EACH_C eadd<uint64_t>(std::min(2 * neq, rsimd), header[q][0],
                header[q][0], cYInc[q], strategy, state);
        if (header4)
            FOR_EACH_C eadd<uint64_t>(
                    8, header[q][2], header[q][2], cYInc[q], strategy, state);
    } else
        FOR_EACH_C add<uint32_t>(rsimd, header[q][0], header[q][0], cYInc[q]);

    // Bottom of y loop.
    //  The any16h is a trick: only the lowest bit of f0[1] is updated when decrementing iy,
    //  but we want to apply it to all channels.
    strategy.fused ? simtDoWhileLoop(16 | f0[1] | any16h, yLoop)
                   : jmpi(1 | f0[1], yLoop);

    // Cleanup.
    state.raVFlag.release(f0[0]);
    state.raVFlag.release(f0[1]);
    state.raVFlag.release(f1[0]);
    state.ra.safeRelease(bases);

    state.ra.safeRelease(indexVec);
    state.ra.safeRelease(Cload);
    state.ra.safeRelease(CaccSwap);
    state.ra.safeRelease(Cacc);
    FOR_EACH_C state.ra.safeRelease(cXInc[q]);
    FOR_EACH_C state.ra.safeRelease(cYInc[q]);
    state.ra.safeRelease(iy);
    state.ra.safeRelease(ix);
    state.ra.safeRelease(ix_init);
    FOR_EACH_C state.ra.safeRelease(header[q]);

    state.flagAP = saveFlagAP;
    if (state.flagAP.isValid()) state.raVFlag.claim(state.flagAP);

#undef FOR_EACH_C
}

// Prepare for GEMM k loop with m/n masked A/B accesses. Returns true if ka_lda/kb_ldb need recalculating.
template <HW hw>
bool gemm_kernel_generator_t<hw>::gemmPrepMaskedAB(
        const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state) {
    bool recalc = false;
    bool shrinkUK = false;
    if (!strategy.A.padded
            && (strategy.remHandling[LoopM] != RemainderHandling::Ignore)) {
        shrinkUK = true;
        if (strategy.ka_load > strategy.ka_load_masked) {
            status << "Downgrading ka_load: " << strategy.ka_load << " -> "
                   << strategy.ka_load_masked << status_stream::endl;
            strategy.ka_load = strategy.ka_load_masked;
            strategy.kChain = gcd(strategy.kChain, strategy.ka_load);
            recalc = true;
        }
        // Avoid access patterns that can't be handled by masking.
        if (isBlock2D(strategy.A.accessType))
            noop();
        else if (problem.A.layout == MatrixLayout::T
                && !isTransposing(strategy.A.accessType))
            strategy.A.accessType = strategy.A.base.isStateless()
                    ? AccessType::Scattered
                    : AccessType::ChannelScattered;
        else if (problem.A.layout != MatrixLayout::T
                && isTransposing(strategy.A.accessType))
            strategy.A.accessType = AccessType::Block;
        strategy.slmATrans = false;
        strategy.prefetchA = strategy.prefetchAMasked;
    }
    if (!strategy.B.padded
            && (strategy.remHandling[LoopN] != RemainderHandling::Ignore)) {
        shrinkUK = true;
        if (strategy.kb_load > strategy.kb_load_masked) {
            status << "Downgrading kb_load: " << strategy.kb_load << " -> "
                   << strategy.kb_load_masked << status_stream::endl;
            strategy.kb_load = strategy.kb_load_masked;
            strategy.kChain = gcd(strategy.kChain, strategy.kb_load);
            recalc = true;
        }
        // Avoid access patterns that can't be handled by masking.
        if (isBlock2D(strategy.B.accessType))
            noop();
        else if (problem.B.layout == MatrixLayout::N
                && !isTransposing(strategy.B.accessType))
            strategy.B.accessType = strategy.B.base.isStateless()
                    ? AccessType::Scattered
                    : AccessType::ChannelScattered;
        else if (problem.B.layout != MatrixLayout::N
                && isTransposing(strategy.B.accessType))
            strategy.B.accessType = AccessType::Block;
        strategy.slmBTrans = false;
        strategy.prefetchB = strategy.prefetchBMasked;
    }
    if (shrinkUK && (strategy.unrollK_masked > 0)
            && (strategy.unroll[LoopK] > strategy.unrollK_masked)) {
        status << "Downgrading k unroll: " << strategy.unroll[LoopK] << " -> "
               << strategy.unrollK_masked << status_stream::endl;
        strategy.unroll[LoopK] = strategy.unrollK_masked;
    }
    if (shrinkUK && (strategy.unrollKSLMMasked > 0)
            && (strategy.unrollKSLM > strategy.unrollKSLMMasked)) {
        status << "Downgrading SLM k chunk size: " << strategy.unrollKSLM
               << " -> " << strategy.unrollKSLMMasked << status_stream::endl;
        strategy.unrollKSLM = strategy.unrollKSLMMasked;
    }
    return recalc;
}

// Generate the GEMM kernel body. If it fails (due to excessive masking, say), return false.
template <HW hw>
bool gemm_kernel_generator_t<hw>::gemmBody(
        GEMMProblem problem, GEMMStrategy strategy, GEMMState state) {
    bool a2D = strategy.A.address2D;
    bool b2D = strategy.B.address2D;
    bool c2D = strategy.C.address2D;

    // Release variables that are no longer needed.
    if (!a2D && !c2D) state.ra.safeRelease(state.i0);
    if (!b2D && !c2D) state.ra.safeRelease(state.j0);
    if (!a2D && !b2D) state.ra.safeRelease(state.h0);
    if (!strategy.altCRemainder) releaseFusedRemainders(state);
    state.ra.safeRelease(state.remaindersWG[LoopM]);
    state.ra.safeRelease(state.remaindersWG[LoopN]);

    // If A/B are masked, check if we need to change ka_load/kb_load. If so, recalculate lda_ka/ldb_kb.
    if (gemmPrepMaskedAB(problem, strategy, state))
        gemmCalcIncrements(problem, strategy, state);

    // Disable C prefetch in remainder handling if it needs masks/fragmenting.
    if (strategy.remHandling[LoopM] != RemainderHandling::Ignore
            || strategy.remHandling[LoopN] != RemainderHandling::Ignore) {
        if (strategy.C.base.isStateless() && !strategy.C.padded
                && strategy.prefetchC
                && !isBlock2D(strategy.C_prefetch.accessType)) {
            status << "Auto-disabling C prefetch in masked region"
                   << status_stream::endl;
            strategy.prefetchC = 0;
            if (state.effCp != state.effC[0]) state.ra.safeRelease(state.effCp);
        }
    }

    // Try generating kernel body with current strategy.
    bool success = false;
    pushStream();
    try {
        success = gemmBodyInternal(problem, strategy, state);
    } catch (...) { lastException = std::current_exception(); }
    success ? appendCurrentStream() : discardStream();

    return success;
}

// Allocate nreg registers in chunks of a given size.
static inline GRFMultirange chunkAlloc(int nreg, int chunk, Bundle hint,
        BundleGroup mask, CommonState &state) {
    GRFMultirange r;
    for (; nreg > 0; nreg -= chunk) {
        auto nr = std::min(nreg, chunk);
        r.ranges.push_back(state.ra.alloc_range(nr, hint, mask));
    }
    return r;
}

static inline GRFMultirange chunkAlloc(
        int nreg, int chunk, Bundle hint, CommonState &state) {
    return chunkAlloc(nreg, chunk, hint, BundleGroup::AllBundles(), state);
}

// Allocate register layout in individual chunks.
static inline GRFMultirange trySplitAlloc(HW hw, Type T,
        const vector<RegisterBlock> &layout, std::array<Bundle, 2> hints,
        BundleGroup mask, CommonState &state, int copies = 1) {
    auto oddHint = Bundle(0, 0).group_size(hw) * elementsPerGRF(hw, T);

    GRFMultirange r;
    struct Request {
        int length, offset, index, hint;
    };
    vector<Request> requests;
    requests.reserve(layout.size());

    for (auto &block : layout) {
        if (block.isLoadBlock()) {
            int hint = ((block.colMajor ? block.offsetR : block.offsetC)
                               & oddHint)
                    != 0;
            requests.push_back({block.msgRegs, block.offsetReg(), 0, hint});
        }
    }

    // Figure out which order the ranges belong in.
    std::sort(requests.begin(), requests.end(),
            [](const Request &r1, const Request &r2) {
                return (r1.offset < r2.offset);
            });
    for (size_t i = 0; i < requests.size(); i++)
        requests[i].index = int(i);

    // Sort again and allocate largest to smallest.
    std::sort(requests.begin(), requests.end(),
            [](const Request &r1, const Request &r2) {
                return (r1.length > r2.length)
                        || (r1.length == r2.length && r1.offset < r2.offset);
            });
    r.ranges.resize(requests.size() * copies);

    bool ok = true;
    for (size_t i = 0; i < requests.size(); i++) {
        for (int c = 0; c < copies; c++) {
            auto newRange = state.ra.try_alloc_range(
                    requests[i].length, hints[requests[i].hint], mask);
            r.ranges[requests[i].index + c * requests.size()] = newRange;
            ok &= newRange.isValid();
        }
    }

    if (!ok) {
        for (auto &rr : r.ranges)
            state.ra.release(rr);
        r.ranges.clear();
    }

    return r;
}

static inline GRFMultirange splitAlloc(HW hw, Type T,
        const vector<RegisterBlock> &layout, std::array<Bundle, 2> hints,
        BundleGroup mask, CommonState &state, int copies = 1) {
    auto r = trySplitAlloc(hw, T, layout, hints, mask, state, copies);
    if (r.empty() && !layout.empty()) throw out_of_registers_exception();
    return r;
}

// Allocate register ranges for A/B/C.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmAllocRegs(
        GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state) {
    // Summary: order of allocations is important.
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc;

    auto A_copies = strategy.A_copies;
    auto B_copies = strategy.B_copies;
    int A_regCount = getRegCount(state.A_layout);
    int Ar_regCount = getRegCount(state.Ar_layout);
    int B_regCount = getRegCount(state.B_layout);
    int Br_regCount = getRegCount(state.Br_layout);
    int C_regCountPerBuffer = getRegCount(state.C_layout);
    int C_regCount = state.C_buffers * C_regCountPerBuffer;
    GRFMultirange C_regs;

    bool globalCM = isLayoutColMajor(state.C_layout);

    auto hintA0 = globalCM ? HintType::A0 : HintType::A0Broadcast;
    auto hintB0 = !globalCM ? HintType::B0 : HintType::B0Broadcast;

    auto Tv = globalCM ? Ta : Tb;
    auto Tn = !globalCM ? Ta : Tb;

    auto &V_layout = globalCM ? state.A_layout : state.B_layout;
    auto &Vr_layout = globalCM ? state.Ar_layout : state.Br_layout;
    auto &V_regs = globalCM ? state.A_regs : state.B_regs;
    auto &Vr_regs = globalCM ? state.Ar_regs : state.Br_regs;
    auto V_copies = globalCM ? A_copies : B_copies;
    auto V_regCount = globalCM ? A_regCount : B_regCount;
    auto Vr_regCount = globalCM ? Ar_regCount : Br_regCount;
    auto &N_layout = !globalCM ? state.A_layout : state.B_layout;
    auto &Nr_layout = !globalCM ? state.Ar_layout : state.Br_layout;
    auto &N_regs = !globalCM ? state.A_regs : state.B_regs;
    auto &Nr_regs = !globalCM ? state.Ar_regs : state.Br_regs;
    auto N_copies = !globalCM ? A_copies : B_copies;
    auto N_regCount = !globalCM ? A_regCount : B_regCount;
    auto Nr_regCount = !globalCM ? Ar_regCount : Br_regCount;

    const auto &C_layout = state.C_layout;
    const auto &C_layoutExt = state.C_layoutExtUnmasked.empty()
            ? state.C_layoutExt
            : state.C_layoutExtUnmasked;

    int C_chunk = state.copyC ? 1 : getMaxLoadBlock(C_layoutExt);
    C_chunk = alignup_pow2(C_chunk, Bundle(0, 0).group_size(hw) * 2);
    if (strategy.systolic) C_chunk = std::max(C_chunk, 8);

    state.C_accCount = strategy.cAccumulators
            ? AccumulatorRegister::count(hw, strategy.GRFs, Tc.ngen())
            : 0;

    state.A_regs.resize(A_copies);
    state.B_regs.resize(B_copies);

    switch (strategy.registerScheme) {
        case GEMMStrategy::CSeparate: {
            // Standard allocation (Gen9-11). A and B allocated together in lower half of registers.
            // Interleave allocation of A and B to minimize wasted registers. Test the waters to find out
            //  whether to try bank 0 or 1 first.
            int bases[2];
            for (int bank = 0; bank < 2; bank++) {
                auto r = state.ra.alloc_range(4, Bundle(bank, Bundle::any));
                bases[bank] = r.getBase();
                state.ra.safeRelease(r);
            }

            // Order of the banks.
            int banks[2];
            banks[0] = (bases[1] < bases[0]) ? 1 : 0;
            banks[1] = 1 - banks[0];

            // Allocate all the registers needed from bank 0, then all the registers needed from bank 1.
            for (int bank : banks) {
                if (getHint(hintA0, strategy).bank_id == bank) {
                    for (int copy = 0; copy < A_copies; copy++)
                        state.A_regs[copy] = state.ra.alloc_range(
                                A_regCount, getHint(hintA0, strategy));
                    if (state.broadcast && !globalCM)
                        state.broadcast_regs = state.ra.alloc_range(
                                2, getHint(hintA0, strategy));
                    if (Ar_regCount > 0)
                        state.Ar_regs = state.ra.alloc_range(
                                Ar_regCount, getHint(hintA0, strategy));
                }

                if (getHint(hintB0, strategy).bank_id == bank) {
                    for (int copy = 0; copy < B_copies; copy++)
                        state.B_regs[copy] = state.ra.alloc_range(
                                B_regCount, getHint(hintB0, strategy));
                    if (state.broadcast && globalCM)
                        state.broadcast_regs = state.ra.alloc_range(
                                2, getHint(hintB0, strategy));
                    if (Br_regCount > 0)
                        state.Br_regs = state.ra.alloc_range(
                                Br_regCount, getHint(hintB0, strategy));
                }
            }

            C_regs = state.ra.alloc_range(C_regCount - state.C_accCount,
                    getHint(HintType::C, strategy));
            break;
        }
        case GEMMStrategy::ACB:
            if (state.broadcast && !globalCM)
                state.broadcast_regs
                        = state.ra.alloc_range(2, getHint(hintA0, strategy));

            for (int copy = 0; copy < A_copies; copy++)
                state.A_regs[copy] = state.ra.alloc_range(
                        A_regCount, getHint(hintA0, strategy));
            if (Ar_regCount > 0)
                state.Ar_regs = state.ra.alloc_range(
                        Ar_regCount, getHint(hintA0, strategy));

            C_regs = state.ra.alloc_range(C_regCount - state.C_accCount,
                    getHint(HintType::C, strategy));

            for (int copy = 0; copy < B_copies; copy++)
                state.B_regs[copy] = state.ra.alloc_range(
                        B_regCount, getHint(hintB0, strategy));
            if (Br_regCount > 0)
                state.Br_regs = state.ra.alloc_range(
                        Br_regCount, getHint(hintB0, strategy));

            if (state.broadcast && globalCM)
                state.broadcast_regs
                        = state.ra.alloc_range(2, getHint(hintB0, strategy));
            break;
        case GEMMStrategy::BCA:
            if (state.broadcast && !globalCM)
                state.broadcast_regs
                        = state.ra.alloc_range(2, getHint(hintA0, strategy));

            for (int copy = 0; copy < B_copies; copy++)
                state.B_regs[copy] = state.ra.alloc_range(
                        B_regCount, getHint(hintB0, strategy));
            if (Br_regCount > 0)
                state.Br_regs = state.ra.alloc_range(
                        Br_regCount, getHint(hintB0, strategy));

            C_regs = state.ra.alloc_range(C_regCount - state.C_accCount,
                    getHint(HintType::C, strategy));

            for (int copy = 0; copy < A_copies; copy++)
                state.A_regs[copy] = state.ra.alloc_range(
                        A_regCount, getHint(hintA0, strategy));
            if (Ar_regCount > 0)
                state.Ar_regs = state.ra.alloc_range(
                        Ar_regCount, getHint(hintA0, strategy));

            if (state.broadcast && globalCM)
                state.broadcast_regs
                        = state.ra.alloc_range(2, getHint(hintB0, strategy));
            break;
        case GEMMStrategy::VNC: {
            if (hw < HW::Gen12LP) stub();

            // Gen12+. Assign non-broadcast input matrix (V), then broadcast input matrix (N), then C.
            auto unrollVBytes
                    = strategy.unroll[globalCM ? LoopM : LoopN] * Tv.size();
            auto unrollNBytes
                    = strategy.unroll[globalCM ? LoopN : LoopM] * Tn.size();
            auto regUnrollV = div_up(unrollVBytes, GRF::bytes(hw));
            auto regUnrollN = div_up(unrollNBytes, GRF::bytes(hw));
            auto hintV = getHint(HintType::A0, strategy);
            auto hintN = getHint(
                    (regUnrollN == 1) ? HintType::A0 : HintType::A0Broadcast,
                    strategy); // Put V and N in same bundle if we can avoid N<->C conflicts.
            auto hintC = getHint(HintType::C, strategy);
            GRFRange tempPadding;

            for (int copy = 0; copy < V_copies; copy++)
                V_regs[copy] = state.ra.alloc_range(V_regCount, hintV);
            if (Vr_regCount > 0)
                Vr_regs = state.ra.alloc_range(Vr_regCount, hintV);

            N_regs[0] = state.ra.alloc_range(N_regCount, hintN);

            // Check if A * B outer product 0 has a bank conflict. If so, move N to avoid this.
            auto stride = Bundle(0, 0).stride(hw);
            auto offN = (N_regs[0][0].getBase() - V_regs[0][0].getBase())
                    & (stride - 1);
            auto offNMin = offN - ((regUnrollV - 1) & ~1);
            auto offNMax = offN + regUnrollN - 1;
            if (offNMax >= stride) offNMax -= stride, offNMin -= stride;
            if (offNMin <= 0) {
                unsigned obAlign = Bundle(0, 0).group_size(hw);
                if (hintN.bank_id != Bundle::any) obAlign *= 2;
                offNMax = alignup_pow2(offNMax, obAlign);
                safeReleaseRanges(N_regs[0], state);
                tempPadding = state.ra.alloc_range(offNMax, hintN);
                N_regs[0] = state.ra.alloc_range(N_regCount, hintN);
            }

            for (int copy = 1; copy < N_copies; copy++)
                N_regs[copy] = state.ra.alloc_range(N_regCount, hintN);
            if (Nr_regCount > 0)
                Nr_regs = state.ra.alloc_range(Nr_regCount, hintN);

            state.ra.safeRelease(tempPadding);

            C_regs = state.ra.alloc_range(C_regCount - state.C_accCount, hintC);
            break;
        }
        case GEMMStrategy::ABInterleave: {
            // Gen12+. Interleave A and B, place C afterward.
            if (hw < HW::Gen12LP) stub();
            auto chunk = Bundle(0, 0).stride(hw) >> 1;

            // Test allocation. Put A earlier if it has more registers.
            int A_regTotal = A_regCount * A_copies + Ar_regCount;
            int B_regTotal = B_regCount * B_copies + Br_regCount;
            auto hintA = getHint(HintType::A0, strategy);
            auto hintB = getHint(HintType::B0, strategy);
            auto hintC = getHint(HintType::C, strategy);
            auto testA = state.ra.alloc_range(8, hintA);
            auto testB = state.ra.alloc_range(8, hintB);
            if ((testA.getBase() < testB.getBase())
                    == (A_regTotal < B_regTotal))
                std::swap(hintA, hintB);
            state.ra.safeRelease(testA);
            state.ra.safeRelease(testB);

            for (int copy = 0; copy < A_copies; copy++)
                state.A_regs[copy]
                        = chunkAlloc(A_regCount, chunk, hintA, state);
            if (Ar_regCount > 0)
                state.Ar_regs = chunkAlloc(Ar_regCount, chunk, hintA, state);
            for (int copy = 0; copy < B_copies; copy++)
                state.B_regs[copy]
                        = chunkAlloc(B_regCount, chunk, hintB, state);
            if (Br_regCount > 0)
                state.Br_regs = chunkAlloc(Br_regCount, chunk, hintB, state);
            C_regs = state.ra.alloc_range(C_regCount - state.C_accCount, hintC);
            break;
        }
        case GEMMStrategy::NSeparate: {
            // Broadcast matrix (N) has dedicated bundle(s) (both banks)
            // V and C start in opposite banks in other bundles.
            if (hw < HW::Gen12LP) stub();
            if (state.C_accCount > 0) stub();

            int bundles = Bundle::bundle_count(hw) * Bundle::bank_count(hw);
            int bregsConsecutive = Bundle(0, 0).group_size(hw);
            int bregs = strategy.GRFs / bundles;
            int N_chunk = getMaxLoadBlock(N_layout);
            int N_nregs = Nr_regCount + N_regCount * N_copies;
            int N_nbundles = std::max(
                    div_up(N_chunk, bregsConsecutive), div_up(N_nregs, bregs));
            BundleGroup N_bundles(hw), VC_bundles(hw);

            auto hintV0 = getHint(HintType::A0, strategy);
            auto hintV1 = getHint(HintType::A1, strategy);
            auto hintN = getHint(HintType::A0Broadcast, strategy);
            auto hintC0 = getHint(HintType::C, strategy);
            auto hintC1 = getHint(HintType::C1, strategy);

            // Give bundles starting at the end to broadcast matrix.
            for (int bundle = Bundle::bundle_count(hw) - 1; bundle >= 0;
                    bundle--) {
                for (int bank = Bundle::bank_count(hw) - 1; bank >= 0; bank--) {
                    if (N_nbundles-- > 0)
                        N_bundles |= Bundle(bank, bundle);
                    else
                        VC_bundles |= Bundle(bank, bundle);
                }
            }

            for (int copy = 0; copy < V_copies; copy++)
                V_regs[copy] = splitAlloc(
                        hw, Tv, V_layout, {hintV0, hintV1}, VC_bundles, state);
            if (Vr_regCount > 0)
                Vr_regs = splitAlloc(
                        hw, Tv, Vr_layout, {hintV0, hintV1}, VC_bundles, state);
            if (!strategy.systolic)
                C_regs = trySplitAlloc(hw, Tc, C_layout, {hintC0, hintC1},
                        VC_bundles, state, state.C_buffers);
            if (C_regs.empty())
                C_regs = chunkAlloc(
                        C_regCount, C_chunk, hintC0, VC_bundles, state);
            for (int copy = 0; copy < N_copies; copy++)
                N_regs[copy] = splitAlloc(
                        hw, Tn, N_layout, {hintN, hintN}, N_bundles, state);
            if (Nr_regCount > 0)
                Nr_regs = splitAlloc(
                        hw, Tn, Nr_layout, {hintN, hintN}, N_bundles, state);
            break;
        }
        case GEMMStrategy::VAvoid: {
            // Broadcast matrix (N) has dedicated starting bank.
            // V and C share starting banks, but C allocations chosen to miss matching V allocations.
            auto hintV = getHint(HintType::A0, strategy);
            auto hintN = getHint(HintType::A0Broadcast, strategy);
            auto hintC = getHint(HintType::C, strategy);

            for (int copy = 0; copy < N_copies; copy++)
                N_regs[copy] = state.ra.alloc_range(N_regCount, hintN);
            if (Nr_regCount > 0)
                Nr_regs = state.ra.alloc_range(Nr_regCount, hintN);

            for (int copy = 0; copy < V_copies; copy++)
                V_regs[copy] = state.ra.alloc_range(V_regCount, hintV);
            if (Vr_regCount > 0)
                Vr_regs = state.ra.alloc_range(Vr_regCount, hintV);

            int nv;
            const RegisterBlock *V_block;
            int V_rows, V_cols;
            getLayoutDims(
                    Vr_regCount > 0 ? Vr_layout : V_layout, V_rows, V_cols);
            int kv = globalCM ? V_cols : V_rows;

            int minOPCount = minOuterProductCount(hw, problem, strategy);
            int lastMN0 = -1;
            int sliceRegs = 0;
            BundleGroup V_bundles(hw);

            vector<GRFMultirange> C_extra(state.C_buffers - 1);
            auto allocSlice = [&]() {
                if (sliceRegs <= 0) return;
                auto C_bundles = ~V_bundles;

                C_regs.append(chunkAlloc(
                        sliceRegs, C_chunk, hintC, C_bundles, state));
                for (int copy = 1; copy < state.C_buffers; copy++)
                    C_extra[copy - 1].append(chunkAlloc(
                            sliceRegs, C_chunk, hintC, C_bundles, state));

                sliceRegs = 0;
            };

            for (const auto &block : C_layout) {
                int mn0 = globalCM ? block.offsetR : block.offsetC;
                if (mn0 == lastMN0) {
                    sliceRegs += block.nregs();
                    continue;
                }

                allocSlice();

                V_bundles = BundleGroup(hw);
                for (int h0 = 0; h0 < kv; h0 += minOPCount) {
                    int r = globalCM ? mn0 : h0;
                    int c = globalCM ? h0 : mn0;
                    if (Vr_regCount == 0)
                        for (int copy = 0; copy < V_copies; copy++) {
                            auto V0 = findBlockReg(Tv, V_layout, r, c,
                                    V_regs[copy], nv, V_block, 0);
                            V_bundles |= Bundle::locate(hw, V0);
                        }
                    else {
                        auto V0 = findBlockReg(
                                Tv, Vr_layout, r, c, Vr_regs, nv, V_block, 0);
                        V_bundles |= Bundle::locate(hw, V0);
                    }
                }

                lastMN0 = mn0;
                sliceRegs = block.nregs();
            }

            allocSlice();

            for (int copy = 1; copy < state.C_buffers; copy++)
                C_regs.append(C_extra[copy - 1]);
        }
    }

    // Assign C_regs, adding in GRFs (in place of accumulators) to use later.
    // Also split into two halves (regular and swapped real/imag parts) for complex.
    state.C_regs.resize(state.C_buffers);

    auto it = C_regs.ranges.begin();
    int off = -state.C_accCount;
    for (int buf = 0; buf < state.C_buffers; buf++) {
        for (int todo = C_regCountPerBuffer; todo > 0;) {
            if (it == C_regs.ranges.end())
                throw std::runtime_error("Not enough C registers allocated.");
            int left = it->getLen() - off;
            int take = std::min(left, todo);
            state.C_regs[buf].ranges.push_back(
                    GRFRange(it->getBase() + off, take));
            todo -= take;
            off += take;
            if (off >= it->getLen()) off = 0, it++;
        }
    }

    // Allocate registers for SLM copies.
    state.Ai_regs.resize(strategy.slmCopies);
    state.Bi_regs.resize(strategy.slmCopies);
    if (strategy.slmA)
        for (int q = 0; q < strategy.slmCopies; q++)
            state.Ai_regs[q]
                    = state.ra.alloc_range(getRegCount(state.Ai_layout));
    if (strategy.slmB)
        for (int q = 0; q < strategy.slmCopies; q++)
            state.Bi_regs[q]
                    = state.ra.alloc_range(getRegCount(state.Bi_layout));

    // Allocate registers for A/B sums.
    state.As_regs = state.ra.alloc_range(getRegCount(state.As_layout));
    state.Bs_regs = state.ra.alloc_range(getRegCount(state.Bs_layout));

    // Allocate registers for A/B prefetch.
    state.Ap_regs = state.ra.alloc_range(getRegCount(state.Ap_layout));
    state.Bp_regs = state.ra.alloc_range(getRegCount(state.Bp_layout));

    // Allocate multiplication temporaries for Gen9 IGEMM, in pairs.
    if (isGen9IGEMM(hw, Ta, Tb, Tc)) {
        auto &temps = state.tempMul_regs;
        for (int ntemp = 0; ntemp < 2; ntemp++) {
            auto range = state.ra.try_alloc_range(2);
            if (range.isValid())
                temps.push_back(range);
            else if (temps.empty())
                throw out_of_registers_exception();
            else
                break;
        }
    }
}

template <HW hw>
void gemm_kernel_generator_t<hw>::gemmAllocAoBoRegs(
        const GEMMStrategy &strategy, GEMMState &state) {
    bool allocAo = false, allocBo = false;

    if (strategy.slmA && state.Ao_regs.empty() && !state.aioShare) {
        allocAo = true;
        if (strategy.slmRepackAhead == 0 && strategy.A_copies == 1) {
            auto nreg = getRegCount(state.Ao_layout);
            auto &defaultRegs = state.A_regs[0];
            allocAo = (defaultRegs.getLen() < nreg);

            if (!allocAo) {
                state.Ao_regs = defaultRegs;
                state.aoReuseA = true;
            }
        }
    }

    if (strategy.slmB && state.Bo_regs.empty() && !state.bioShare) {
        allocBo = true;
        if (strategy.slmRepackAhead == 0 && strategy.B_copies == 1) {
            auto nreg = getRegCount(state.Bo_layout);
            auto &defaultRegs = state.B_regs[0];
            allocBo = (defaultRegs.getLen() < nreg);

            if (!allocBo) {
                state.Bo_regs = defaultRegs;
                state.boReuseB = true;
            }
        }
    }

    if (allocAo && !state.allocedAo) {
        state.allocedAo = true;
        state.Ao_regs = state.ra.alloc_range(getRegCount(state.Ao_layout));
    }

    if (allocBo && !state.allocedBo) {
        state.allocedBo = true;
        state.Bo_regs = state.ra.alloc_range(getRegCount(state.Bo_layout));
    }
}

// Prepare layout for row/column sum matrices, and any needed auxiliary registers.
template <HW hw>
void gemm_kernel_generator_t<hw>::makeSumLayout(bool column, Type Tsrc,
        const vector<RegisterBlock> &srcLayout, Type Tdst,
        vector<RegisterBlock> &dstLayout, const CommonStrategy &strategy,
        CommonState &state) {
    bool canDP4A = (hw >= HW::Gen12LP) && one_of(Tsrc, Type::s8, Type::u8)
            && one_of(Tdst, Type::s32, Type::u32);
    bool cm = isLayoutColMajor(srcLayout);
    bool hReduce = (column == cm);
    bool needAll1s = false;
    int m, n;

    getLayoutDims(srcLayout, m, n);
    auto &rdim = column ? m : n;

    if (hReduce) {
        if (canDP4A && hasFullCrosspack(srcLayout, 1)) {
            rdim /= 4;
            needAll1s = true;
            if (rdim & 1) rdim <<= 1; // Ensure dp4a dest offset is even.
        }
    } else {
        if (canDP4A && hasFullCrosspack(srcLayout, 4)) needAll1s |= (rdim >= 4);
        rdim = 1;
    }

    makeUnbackedRegLayout(Tdst, dstLayout, m, n, cm, 1);

    // Prepare all-1s immediate for dp4a.
    if (needAll1s && state.all1s.isInvalid()) {
        state.all1s = state.ra.alloc_sub(
                Tdst.ngen(), getHint(HintType::LongTerm, strategy));
        mov(1, state.all1s, 0x01010101);
    }
}

// Accumulate row/column sums.
template <HW hw>
void gemm_kernel_generator_t<hw>::accumulateSum(bool column, Type Tsrc,
        const GRFMultirange &srcRegs, const vector<RegisterBlock> &srcLayout,
        Type Tdst, const GRFMultirange &dstRegs,
        const vector<RegisterBlock> &dstLayout, const CommonStrategy &strategy,
        CommonState &state, int q0, int q1) {
    bool canDP4A = (hw >= HW::Gen12LP) && one_of(Tsrc, Type::s8, Type::u8)
            && one_of(Tdst, Type::s32, Type::u32);

    bool cm = isLayoutColMajor(srcLayout);
    if (cm != isLayoutColMajor(dstLayout)) stub();

    int m, n;
    getLayoutDims(srcLayout, m, n);

    // x: consecutive dimension in src; y: strided dimension in src
    auto nx = cm ? m : n;
    auto ny = cm ? n : m;

    int x0 = 0, y0 = 0;
    int x1 = nx, y1 = ny;

    if (q1 >= 0) ((column == cm) ? x1 : y1) = q1;
    if (q0 >= 0) ((column == cm) ? x0 : y0) = q0;

    // Two cases to handle:
    //   hReduce = false:  Good case; no reduction. Sum is vector of size mx1 or 1xn.
    //   hReduce = true:   Bad case; needs reduction later, although with dp4a some reduction can be done now.
    bool hReduce = (column == cm);

    int yinc = 1;
    int reduce = (canDP4A && hReduce) ? 4 : 1;
    if (x0 % reduce || x1 % reduce) stub();

    for (int y = y0; y < y1; y += yinc) {
        for (int x = x0; x < x1;) {
            int isrc, jsrc, idst, jdst, nsrc, ndst;
            const RegisterBlock *blockSrc, *blockDst;

            isrc = cm ? x : y;
            jsrc = cm ? y : x;
            if (!hReduce) {
                idst = cm ? x : 0;
                jdst = cm ? 0 : x;
            } else {
                idst = cm ? x / reduce : y;
                jdst = cm ? y : x / reduce;
            }

            Subregister srcBase = findBlockReg(
                    Tsrc, srcLayout, isrc, jsrc, srcRegs, nsrc, blockSrc);
            Subregister dstBase = findBlockReg(
                    Tdst, dstLayout, idst, jdst, dstRegs, ndst, blockDst);
            auto ne = std::min(
                    {nsrc / reduce, ndst, elementsPerGRF(hw, Tdst) * 2});

            auto src = srcBase(blockSrc->crosspack);
            auto dst = dstBase(blockDst->crosspack);

            if (canDP4A) {
                auto srcDP4A
                        = Tsrc.isSigned() ? srcBase.d()(1) : srcBase.ud()(1);
                if (!hReduce && blockSrc->crosspack == 4) {
                    yinc = std::min(y1 - y, 4);
                    if (yinc == 4)
                        dp4a(ne, dst, dst, srcDP4A, state.all1s);
                    else if (yinc == 1)
                        add(ne, dst, srcBase(4), dst);
                    else
                        dp4a(ne, dst, dst, srcDP4A,
                                0x01010101 & ((1 << (yinc * 8)) - 1));
                } else if (hReduce && blockSrc->crosspack == 1) {
                    if (Tsrc.isSigned())
                        dp4a(ne, dst, dst, srcDP4A, state.all1s);
                    else {
                        // Workaround for suspected HW issue.
                        dst.setType(DataType::ud);
                        dp4a(ne, dst, dst, srcDP4A, state.all1s.ud());
                    }
                }
            } else
                add(ne, dst, dst, src);

            x += ne * reduce;
        }
    }
}

// Horizontally add intermediate sums if needed.
template <HW hw>
void gemm_kernel_generator_t<hw>::horizontalAdd(bool column, Type T,
        const GRFMultirange &regs, vector<RegisterBlock> &layout) {
    bool cm = isLayoutColMajor(layout);
    if (cm != column) return; // Nothing to do.

    int m, n;
    getLayoutDims(layout, m, n);

    int nx = cm ? m : n;
    int ny = cm ? n : m;
    int ne = elementsPerGRF(hw, T);

    for (int chunk = roundup_pow2(nx) >> 1; chunk > 0; chunk >>= 1) {
        for (int y = 0; y < ny; y++) {
            for (int x = chunk; x < (chunk * 2) && x < nx;) {
                int i = cm ? x : y;
                int j = cm ? y : x;
                int ns, nb;
                const RegisterBlock *block;
                Subregister shifted
                        = findBlockReg(T, layout, i, j, regs, ns, block);

                ns = std::min(ns, chunk);
                (cm ? i : j) -= chunk;
                Subregister base
                        = findBlockReg(T, layout, i, j, regs, nb, block);

                auto dest = base;
                if (chunk == 1) dest = regs[y / ne].sub(y % ne, T.ngen());

                add(ns, dest(1), base(1), shifted(1));
                x += ns;
            }
        }
    }

    (cm ? m : n) = 1;
    makeUnbackedRegLayout(T, layout, m, n, !cm, 1);
}

// Get final A/B sums. For SLM copy kernels, this requires accumulating each thread's contributions.
template <HW hw>
bool gemm_kernel_generator_t<hw>::gemmFinalizeSums(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state) {
    if (problem.abOffset != ABOffset::Calc) return true;

    auto Tc = problem.Tc;
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];
    bool ok = true;

    int ms, ns;
    getLayoutDims(state.As_layout, ms, ns);
    bool reduceAs = (ns > 1);
    getLayoutDims(state.Bs_layout, ms, ns);
    bool reduceBs = (ms > 1);

    if (!strategy.slmA && reduceAs)
        horizontalAdd(false, Tc, state.As_regs, state.As_layout);
    if (!strategy.slmB && reduceBs)
        horizontalAdd(true, Tc, state.Bs_regs, state.Bs_layout);

    if (!strategy.slmA && !strategy.slmB) return true;

    if (state.effCoopA == CoopSplit::Linear
            || state.effCoopB == CoopSplit::Linear)
        stub();
    bool A_coopSplitM = (state.effCoopA == CoopSplit::MN);
    bool B_coopSplitN = (state.effCoopB == CoopSplit::MN);

    GRFMultirange *ABs_regs[2] = {&state.As_regs, &state.Bs_regs};
    bool AB_coopSplitMN[2] = {A_coopSplitM, B_coopSplitN};
    vector<RegisterBlock> *ABs_layout[2] = {&state.As_layout, &state.Bs_layout};

    vector<RegisterBlock> ABs_layoutSLM[2];
    MatrixAddressing ABs_SLM[2];
    MatrixAddressingStrategy ABs_strategySLM[2];
    MatrixAddressingStrategy ABs_strategySLMAtomic[2];
    vector<GRFRange> ABs_addrs[2];
    GRF temp = state.ra.alloc();
    FlagRegister leader[2];
    Subregister ABs_base[2];

    if (state.r0_info.isARF()) stub();
    GRF r0_info {state.r0_info.getBase()};

    // Plan:
    //   1) First thread of each m/n-block (leader) stores its sums in SLM; barrier
    //   2) Remaining threads atomically add their sums to the first; barrier
    //   3) All threads read final sums
    // For scattered SLM write kernels, threads have accumulated disjoint parts
    //  of the sums, so the second step isn't needed. However, each thread needs
    //  to do a horizontal reduction first.

    // Wait for previous SLM reads to complete.
    // In the meantime, finish sum reduction if necessary.
    status << "Finalize A/B sums" << status_stream::endl;

    if (hw >= HW::Gen11) slmfence(temp, r0_info);
    barriersignal(temp, r0_info);

    if (strategy.slmA && A_coopSplitM)
        horizontalAdd(false, Tc, state.As_regs, state.As_layout);
    if (strategy.slmB && B_coopSplitN)
        horizontalAdd(true, Tc, state.Bs_regs, state.Bs_layout);

    barrierwait();

    auto step1 = [&](bool isB, int r, int c) {
        ABs_SLM[isB].setAlignment(r * c * Tc);
        ABs_SLM[isB].crosspack = 1;
        ABs_SLM[isB].layout = !isB ? MatrixLayout::Pc : MatrixLayout::Pr;
        ABs_SLM[isB].packSize = r * c;
        // Use pseudoblock to share address registers between regular and atomic accesses.
        ABs_strategySLMAtomic[isB].base = AddressBase::createSLM();
        ABs_strategySLMAtomic[isB].padded = true;
        ABs_strategySLMAtomic[isB].accessType = AB_coopSplitMN[isB]
                ? AccessType::Block
                : AccessType::PseudoBlock;
        ABs_strategySLMAtomic[isB].atomic = !AB_coopSplitMN[isB];
        ABs_strategySLM[isB] = ABs_strategySLMAtomic[isB];
        ABs_strategySLM[isB].atomic = false;

        ok = ok
                && getRegLayout(Tc, ABs_layoutSLM[isB], r, c, false, false,
                        true, true, 0, 0, ABs_SLM[isB], ABs_strategySLM[isB])
                && matchLayouts(Tc, ABs_layoutSLM[isB], *ABs_layout[isB]);

        Subregister adjBase = ABs_base[isB] = state.ra.alloc_sub<uint32_t>();
        uint16_t slmOffset = (isB && strategy.slmA)
                ? (unrollM * strategy.wg[LoopM] * Tc)
                : 0;

        !isB ? mulConstant(1, ABs_base[isB], state.lidM, unrollM * Tc)
             : mulConstant(1, ABs_base[isB], state.lidN, unrollN * Tc);

        if (slmOffset != 0) add(1, ABs_base[isB], ABs_base[isB], slmOffset);

        if (AB_coopSplitMN[isB]) {
            adjBase = state.ra.alloc_sub<uint32_t>();
            !isB ? mulConstant(1, adjBase, state.lidN, state.ma_slm * Tc)
                 : mulConstant(1, adjBase, state.lidM, state.nb_slm * Tc);
            add(1, adjBase, adjBase, ABs_base[isB]);
        }

        allocAddrRegs(ABs_addrs[isB], ABs_layoutSLM[isB], ABs_SLM[isB],
                ABs_strategySLM[isB], state);
        setupAddr(Tc, ABs_addrs[isB], adjBase, ABs_layoutSLM[isB],
                Subregister(), ABs_SLM[isB], ABs_strategySLM[isB], strategy,
                state);

        if (AB_coopSplitMN[isB]) state.ra.safeRelease(adjBase);

        Label labelNoStore;
        if (!AB_coopSplitMN[isB]) {
            leader[isB] = state.raVFlag.alloc();
            cmp(16 | eq | leader[isB], !isB ? state.lidN : state.lidM, 0);
            if_(16 | leader[isB], labelNoStore);
        }
        storeMatrix(*ABs_regs[isB], ABs_layoutSLM[isB], ABs_SLM[isB],
                ABs_strategySLM[isB], ABs_addrs[isB], strategy, state);
        if (!AB_coopSplitMN[isB]) {
            mark(labelNoStore);
            endif(16);
        }
    };

    bool barrier2 = false;
    auto step2 = [&](bool isB) {
        Label labelNoAdd;
        if_(16 | ~leader[isB], labelNoAdd);
        atomicAddMatrix(Tc, *ABs_regs[isB], ABs_layoutSLM[isB], ABs_SLM[isB],
                ABs_strategySLMAtomic[isB], ABs_addrs[isB], problem, strategy,
                state);
        mark(labelNoAdd);
        endif(16);
        barrier2 = true;
    };

    auto step3 = [&](bool isB, int r, int c) {
        if (AB_coopSplitMN[isB]) {
            safeReleaseRanges(ABs_addrs[isB], state);
            ABs_SLM[isB].packSize = r * c;
            ABs_SLM[isB].setAlignment(r * c * Tc);
            ABs_strategySLM[isB].accessType = AccessType::Block;
            ok = ok
                    && getRegLayout(Tc, ABs_layoutSLM[isB], r, c, false, false,
                            false, true, 0, 0, ABs_SLM[isB],
                            ABs_strategySLM[isB]);

            auto nregs = getRegCount(ABs_layoutSLM[isB]);
            if (nregs > ABs_regs[isB]->getLen()) {
                safeReleaseRanges(*ABs_regs[isB], state);
                *ABs_regs[isB] = state.ra.alloc_range(nregs);
            }

            allocAddrRegs(ABs_addrs[isB], ABs_layoutSLM[isB], ABs_SLM[isB],
                    ABs_strategySLM[isB], state);
            setupAddr(Tc, ABs_addrs[isB], ABs_base[isB], ABs_layoutSLM[isB],
                    Subregister(), ABs_SLM[isB], ABs_strategySLM[isB], strategy,
                    state);
        }
        loadMatrix(*ABs_regs[isB], ABs_layoutSLM[isB], ABs_SLM[isB],
                ABs_strategySLM[isB], ABs_addrs[isB], strategy, state);
        *ABs_layout[isB] = std::move(ABs_layoutSLM[isB]);
    };

    if (strategy.slmA) step1(false, state.ma_slm, 1);
    if (strategy.slmB) step1(true, 1, state.nb_slm);

    slmBarrier(temp, r0_info);

    if (strategy.slmA && !A_coopSplitM) step2(false);
    if (strategy.slmB && !B_coopSplitN) step2(true);

    if (barrier2) slmBarrier(temp, r0_info);

    if (strategy.slmA) step3(false, unrollM, 1);
    if (strategy.slmB) step3(true, 1, unrollN);

    state.ra.safeRelease(temp);
    state.ra.safeRelease(ABs_base[0]);
    state.ra.safeRelease(ABs_base[1]);
    state.raVFlag.safeRelease(leader[0]);
    state.raVFlag.safeRelease(leader[1]);
    safeReleaseRanges(ABs_addrs[0], state);
    safeReleaseRanges(ABs_addrs[1], state);

    return ok;
}

// Convert register range to a new type.
// If types are different sizes, we assume that the smaller type's stride is the width
//  of the larger type.
template <HW hw>
void gemm_kernel_generator_t<hw>::convert(const GRFMultirange &range, Type Told,
        Type Tnew, const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
    if (Told == Tnew) return;

    if (hw == HW::Gen9 && Told == Type::f32 && !Tnew.isFP()) {
        // Gen9: round to nearest before downconvert (not done by mov).
        map(hw, Told, range, range, strategy,
                [&](int esize, GRF r, GRF _) { rnde(esize, r.f(), r.f()); });
    }

    int maxLS = std::max(Told.log2Size(), Tnew.log2Size());
    int hsOld = 1 << (maxLS - Told.log2Size());
    int hsNew = 1 << (maxLS - Tnew.log2Size());
    auto Tmax = (Told.size() < Tnew.size()) ? Tnew : Told;

    InstructionModifier mod;
    if (Told != Tnew && Tnew.isInteger() && Tnew.size() <= Told.size())
        mod = mod | sat;

    map(hw, Tmax, range, range, strategy, [&](int esize, GRF r, GRF _) {
        emov(esize | mod, r.sub(0, Tnew.ngen())(hsNew),
                r.sub(0, Told.ngen())(hsOld), strategy, state);
    });
}

// Convert C accumulator registers to a new type. Returns true if successful, or false if old and new type are different sizes.
template <HW hw>
bool gemm_kernel_generator_t<hw>::gemmConvertC(Type Tnew,
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
    auto Told = state.Tacc;
    int ncomp = (problem.Tc.isComplex() && state.C_buffers == 2
                        && state.cSwapActive)
            ? 2
            : 1;

    if (Tnew.size() != Told.size()) return false;

    for (int comp = 0; comp < ncomp; comp++)
        convert(state.C_regs[comp], Told, Tnew, problem, strategy, state);

    state.Tacc = Tnew;

    return true;
}

// Perform beta scaling.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmBetaScale(
        GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state) {
    Label labelBetaDone;

    auto Ts = problem.Ts;
    auto &betar = problem.beta_real;

    if (state.beta1.isValid()) {
        if (strategy.fused) {
            cmp(16 | lt | state.flagAP, null.d(), state.beta1, int16_t(0));
            goto12(16 | state.flagAP, labelBetaDone);
        } else {
            cmp(1 | lt | state.flagAP, null.d(), state.beta1, int16_t(0));
            jmpi(1 | state.flagAP, labelBetaDone);
        }
    }

    gemmConvertC(problem.Ts, problem, strategy, state);

    if (betar != 1) {
        map(hw, Ts.real(), state.C_regs[0], state.C_regs[0], strategy,
                [&](int esize, GRF acc, GRF _) {
                    betar.fixed() ? mul(esize, acc, acc, cast(Ts.real(), betar))
                                  : mul(esize, acc, acc,
                                          betar.getRegAvoiding(hw, acc));
                });
    }

    gemmConvertC(problem.Tc, problem, strategy, state);

    mark(labelBetaDone);

    if (state.beta1.isValid() && strategy.fused) join(16);
}

// Add fixed offset to C.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmFixedOffsetC(const Subregister &offset,
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
    auto offsetTc = offset.reinterpret(0, problem.Tc.ngen());
    if (problem.Tc != problem.Tco) emov(1, offsetTc, offset, strategy, state);

    map(hw, problem.Tc, state.C_regs[0], state.C_layout, strategy,
            [&](int simd, const RegData &r) { add(simd, r, r, offsetTc); });
}

// Add row-wise or column-wise offsets to C, possibly multiplying by a scalar.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmVariableOffsetC(bool column,
        const GRFMultirange &offsets, const Subregister &scale,
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state, Type Tco, vector<RegisterBlock> CO_layout) {
    auto Tc = problem.Tc;
    auto ne = elementsPerGRF(hw, Tc);
    auto globalCM = isLayoutColMajor(state.C_layout);
    auto unrollX = strategy.unroll[globalCM ? LoopM : LoopN];
    auto unrollY = strategy.unroll[globalCM ? LoopN : LoopM];
    auto crosspack = CO_layout.empty() ? 1 : CO_layout[0].crosspack;
    auto stride = [&]() { return (column == globalCM) ? 0 : crosspack; };
    const GRFMultirange *offsetsPtr = &offsets;

    if (Tco == Type::invalid) Tco = Tc;

    bool needRepack = (Tc != Tco);
    needRepack |= (stride() > 1 && hw >= HW::XeHP && Tc.isFP());

    GRFMultirange repackOffsets;
    if (needRepack) {
        // Repack data to unit stride as float pipe can't swizzle.
        vector<RegisterBlock> repackLayout;
        int r = column ? 1 : strategy.unroll[LoopM];
        int c = !column ? 1 : strategy.unroll[LoopN];
        makeUnbackedRegLayout(Tc, repackLayout, r, c, !column);
        repackOffsets = state.ra.alloc_range(getRegCount(repackLayout));
        copyRegisters(Tco, Tc, CO_layout, repackLayout, offsets, repackOffsets,
                0, 0, false, strategy, state);
        crosspack = 1;
        offsetsPtr = &repackOffsets;
    }

    for (int y = 0; y < unrollY; y++) {
        for (int x = 0; x < unrollX;) {
            auto i = globalCM ? x : y;
            auto j = globalCM ? y : x;
            int nc;
            const RegisterBlock *C_block;
            Subregister C = findBlockReg(
                    Tc, state.C_layout, i, j, state.C_regs[0], nc, C_block);

            nc = std::min({nc, strategy.fmaSIMD / crosspack, 2 * ne});
            auto nco = (column ? j : i) * crosspack;
            auto offBase = (*offsetsPtr)[nco / ne].sub(nco % ne, Tc.ngen());
            if (scale.isValid())
                mad(nc, C(1), C(1), offBase(stride()), scale);
            else
                add(nc, C(1), C(1), offBase(stride()));

            x += nc;
        }
    }

    safeReleaseRanges(repackOffsets, state);
}

// Apply fixed/row-wise/column-wise C offset.
template <HW hw>
bool gemm_kernel_generator_t<hw>::gemmApplyCOffset(bool row, bool column,
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
    auto Tco = problem.Tco;
    auto cor = row ? strategy.unroll[LoopM] : 1;
    auto coc = column ? strategy.unroll[LoopN] : 1;

    auto CO = problem.CO;
    auto CO_strategy = strategy.CO;
    std::vector<GRFRange> CO_addrs;
    std::vector<RegisterBlock> CO_layout;
    std::vector<MaskAssignment> masks;
    CO_strategy.accessType = AccessType::Block;

    CO.layout = column ? MatrixLayout::T : MatrixLayout::N;

    auto remR = row && !strategy.CO.padded;
    auto remC = column && !strategy.CO.padded;

    if (!getRegLayout(Tco, CO_layout, cor, coc, remR, remC, false, true, 0, 0,
                CO, CO_strategy))
        return false;

    auto CO_regs = state.ra.alloc_range(getRegCount(CO_layout));

    allocAddrRegs(CO_addrs, CO_layout, CO, CO_strategy, state);
    setupAddr(Tco, CO_addrs, state.effCO, CO_layout, Subregister(), CO,
            CO_strategy, strategy, state);

    if (!assignMasks(CO_layout, LoopM, LoopN, masks, state)) {
        status << "Retrying with virtual flags." << status_stream::endl;
        allocVFlagStorage(strategy, state);
        if (!assignMasks(CO_layout, LoopM, LoopN, masks, state)) return false;
    }

    loadMasks(masks, state.remainders, strategy, state);
    loadMatrix(CO_regs, CO_layout, CO, CO_strategy, CO_addrs, strategy, state);
    safeReleaseMaskAssignments(masks, state);

    if (row && column)
        stub();
    else if (!row && !column)
        gemmFixedOffsetC(
                CO_regs[0].sub(0, Tco.ngen()), problem, strategy, state);
    else
        gemmVariableOffsetC(column, CO_regs, Subregister(), problem, strategy,
                state, problem.Tco, CO_layout);

    state.ra.safeRelease(CO_regs);
    safeReleaseRanges(CO_addrs, state);

    return true;
}

// Check kernel input for desired C offset and apply it.
template <HW hw>
bool gemm_kernel_generator_t<hw>::gemmApplyCOffsetDispatch(
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
    Label labelCOColumn, labelCORow, labelCODone;
    bool ok = true;

    if (state.flagSwizzle.isValid()) state.raVFlag.release(state.flagSwizzle);

    auto flagNonfinal = state.raVFlag.alloc();
    auto flagCOC = state.raVFlag.alloc();
    auto flagCOR = state.raVFlag.alloc();

    and_(1 | nz | flagNonfinal, null.ud(), state.inputs.flags,
            FlagNonfinalKBlock);
    and_(1 | nz | flagCOC, null.ud(), state.inputs.flags, FlagCOColumn);
    and_(1 | nz | flagCOR, null.ud(), state.inputs.flags, FlagCORow);
    jmpi(1 | flagNonfinal, labelCODone);
    jmpi(1 | flagCOC, labelCOColumn);
    jmpi(1 | flagCOR, labelCORow);

    state.raVFlag.safeRelease(flagNonfinal);
    state.raVFlag.safeRelease(flagCOC);
    state.raVFlag.safeRelease(flagCOR);

    if (state.flagSwizzle.isValid()) state.raVFlag.claim(state.flagSwizzle);

    status << "Applying fixed C offset" << status_stream::endl;
    ok = ok && gemmApplyCOffset(false, false, problem, strategy, state);
    jmpi(1, labelCODone);

    mark(labelCOColumn);
    status << "Applying column-wise C offset" << status_stream::endl;
    ok = ok && gemmApplyCOffset(false, true, problem, strategy, state);
    jmpi(1, labelCODone);

    mark(labelCORow);
    status << "Applying row-wise C offset" << status_stream::endl;
    ok = ok && gemmApplyCOffset(true, false, problem, strategy, state);

    mark(labelCODone);

    return ok;
}

// Calculate addresses of A/B sums in packed input data. Sums are stored at the end of each panel.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmCalcABOffsetAddrs(
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
    auto &effAs = state.effAs;
    auto &effBs = state.effBs;

    auto Tc = problem.Tc;
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];

    if (effAs.isInvalid()) effAs = state.ra.alloc_sub(state.effA.getType());
    if (effBs.isInvalid()) effBs = state.ra.alloc_sub(state.effB.getType());

    mulConstant(1, effAs.ud(), state.inputs.lda, unrollM);
    mulConstant(1, effBs.ud(), state.inputs.ldb, unrollN);
    add(1, effAs.ud(), effAs.ud(), -unrollM * Tc);
    add(1, effBs.ud(), effBs.ud(), -unrollN * Tc);
    eadd(1, effAs, effAs.ud(), state.effA, strategy, state);
    eadd(1, effBs, effBs.ud(), state.effB, strategy, state);
}

// Load A/B sums from packed input data.
template <HW hw>
bool gemm_kernel_generator_t<hw>::gemmLoadABOffset(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state) {
    if (problem.abOffset != ABOffset::Load) return true;

    auto Tc = problem.Tc;
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];

    MatrixAddressing As = problem.A, Bs = problem.B;
    As.crosspack = 1;
    Bs.crosspack = 1;
    As.tileR = As.tileC = 0;
    Bs.tileR = Bs.tileC = 0;

    MatrixAddressingStrategy As_strategy = strategy.A, Bs_strategy = strategy.B;
    As_strategy.accessType = AccessType::Block;
    Bs_strategy.accessType = AccessType::Block;

    bool ok = true;
    ok = ok
            && getRegLayout(Tc, state.As_layout, unrollM, 1, false, false,
                    false, true, 0, 0, As, As_strategy);
    ok = ok
            && getRegLayout(Tc, state.Bs_layout, 1, unrollN, false, false,
                    false, true, 0, 0, Bs, Bs_strategy);
    if (!ok) return false;

    state.As_regs = state.ra.alloc_range(getRegCount(state.As_layout));
    state.Bs_regs = state.ra.alloc_range(getRegCount(state.Bs_layout));

    vector<GRFRange> As_addrs, Bs_addrs;
    allocAddrRegs(As_addrs, state.As_layout, As, As_strategy, state);
    allocAddrRegs(Bs_addrs, state.Bs_layout, Bs, Bs_strategy, state);

    if (state.effAs.isInvalid())
        gemmCalcABOffsetAddrs(problem, strategy, state);

    setupAddr(Tc, As_addrs, state.effAs, state.As_layout, Subregister(), As,
            As_strategy, strategy, state);
    setupAddr(Tc, Bs_addrs, state.effBs, state.Bs_layout, Subregister(), Bs,
            Bs_strategy, strategy, state);

    loadMatrix(state.As_regs, state.As_layout, As, As_strategy, As_addrs,
            strategy, state);
    loadMatrix(state.Bs_regs, state.Bs_layout, Bs, Bs_strategy, Bs_addrs,
            strategy, state);

    state.ra.safeRelease(state.effAs);
    state.ra.safeRelease(state.effBs);
    safeReleaseRanges(As_addrs, state);
    safeReleaseRanges(Bs_addrs, state);

    return true;
}

// Apply contributions from A/B offsets to C matrix, using previously loaded/computed
// A row sums and B column sums.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmApplyABOffset(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state) {
    if (problem.abOffset == ABOffset::None) return;

    // Two steps: (O = all-1s matrix)
    //   1) C += A * O * bo
    //   2) C += (O * B + bo * k) * ao
    // TODO: combine C adds into add3 on XeHP+.
    auto temp = state.ra.alloc_sub(problem.Tc.ngen());
    mul(1, temp, state.k, state.inputs.bo);

    bool noFMA = (hw == HW::Gen9);
    if (noFMA) {
        map(hw, problem.Tc, state.Bs_regs, state.Bs_layout, strategy,
                [&](int ne, RegData r) { add(ne, r, r, temp); });
        map(hw, problem.Tc, state.As_regs, state.As_layout, strategy,
                [&](int ne, RegData r) { mul(ne, r, r, state.inputs.bo); });
        map(hw, problem.Tc, state.Bs_regs, state.Bs_layout, strategy,
                [&](int ne, RegData r) { mul(ne, r, r, state.inputs.ao); });
    } else {
        mul(1, temp, temp, state.inputs.ao);
        map(hw, problem.Tc, state.Bs_regs, state.Bs_layout, strategy,
                [&](int ne, RegData r) {
                    mad(ne, r, temp, r, state.inputs.ao);
                });
    }
    state.ra.safeRelease(temp);

    gemmVariableOffsetC(false, state.As_regs,
            noFMA ? Subregister() : state.inputs.bo, problem, strategy, state,
            problem.Tc, state.As_layout);
    gemmVariableOffsetC(true, state.Bs_regs, Subregister(), problem, strategy,
            state, problem.Tc, state.Bs_layout);

    safeReleaseRanges(state.As_regs, state);
    safeReleaseRanges(state.Bs_regs, state);
    if (!strategy.persistent) {
        state.ra.safeRelease(state.inputs.ao);
        state.ra.safeRelease(state.inputs.bo);
    }
    state.As_layout.clear();
    state.Bs_layout.clear();
}

// Generate code for summing C across k dimension through SLM.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmKReduce(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state) {
    auto Tc = problem.Tc;
    Label lDone;

    // Early exit if nothing to do. All branching scalar since no fusing in k dimension.
    cmp(1 | le | state.flagAP, state.lszK, 1);
    jmpi(1 | state.flagAP, lDone);

    status << "k reduction through SLM" << status_stream::endl;
    cmp(1 | eq | state.flagAP, state.lidK, 0);

    // In general SLM isn't large enough to do the reduction in one step.
    // Slice C into pieces that will fit.
    int maxMNThreads = strategy.wg[LoopM] * strategy.wg[LoopN];
    if (maxMNThreads <= 0)
        throw std::runtime_error("Max workgroup size not specified");

    int regs = state.C_regs[0].getLen();
    int sliceRegs = int(gemmPerKSLMSize(problem, strategy)
            / (maxMNThreads * GRF::bytes(hw)));
    if (sliceRegs <= 0)
        throw std::runtime_error("Not enough SLM for k reduction");

    // Temporaries.
    auto kt = state.ra.alloc_sub<int32_t>();
    auto flagKTLoop = state.raVFlag.alloc();
    auto barrierTemp = state.ra.alloc();

    if (state.r0_info.isARF()) stub();
    GRF r0_info {state.r0_info.getBase()};

    bool initialBarrier = (strategy.slmBuffers > 0 || strategy.persistent);
    if (initialBarrier) barriersignal(barrierTemp, r0_info);

    // Set up addressing.
    auto addr0 = state.ra.alloc_sub<uint32_t>();
    emad(1, addr0, state.lidM, state.lidN, strategy.wg[LoopM], strategy, state);
    emad(1, addr0, addr0, state.lidK, strategy.wg[LoopM] * strategy.wg[LoopN],
            strategy, state);
    mulConstant(1, addr0, addr0, sliceRegs * GRF::bytes(hw));

    int unrollKSLMStride = strategy.wg[LoopM] * strategy.wg[LoopN] * sliceRegs
            * GRF::bytes(hw);
    Subregister unrollKSLMReturn = state.ra.alloc_sub<int32_t>();

    mulConstant(1, unrollKSLMReturn, -state.lszK, unrollKSLMStride);

    MatrixAddressing C_slm;
    MatrixAddressingStrategy C_slmStrategy;

    C_slm.layout = MatrixLayout::Pc;
    C_slm.packSize = elementsPerGRF(hw, Tc);
    C_slm.crosspack = 1;
    C_slm.setAlignment(GRF::bytes(hw));

    C_slmStrategy.base = SLM;
    C_slmStrategy.accessType = AccessType::Block;
    C_slmStrategy.padded = true;
    if (hw >= HW::XeHPG) C_slmStrategy.newDP = true;

    GRFRange C_load;
    vector<RegisterBlock> C_slmLayout;
    vector<GRFRange> C_slmAddrs;

    // Find maximum # registers of C we can transfer to/from SLM at once.
    int maxContig = rounddown_pow2(regs);
    for (; maxContig > 1; maxContig >>= 1) {
        bool ok = true;
        for (int offsetReg = 0; offsetReg < regs; offsetReg += maxContig) {
            int nr = std::min(regs - offsetReg, maxContig);
            if (!state.C_regs[0].contiguous(offsetReg, nr)) {
                ok = false;
                break;
            }
        }
        if (ok) break;
    }

    // Allocate address and data registers, automatically shrinking sliceRegs if
    //  there are not enough registers.
    for (; sliceRegs > 0; sliceRegs = rounddown_pow2(sliceRegs - 1)) {
        bool ok = true;

        C_load = state.ra.try_alloc_range(sliceRegs);
        ok = ok && C_load.isValid();

        if (!getRegLayout(Tc, C_slmLayout, elementsPerGRF(hw, Tc), sliceRegs,
                    false, false, true, true, 0, maxContig, C_slm,
                    C_slmStrategy))
            stub();
        ok = ok
                && tryAllocAddrRegs(
                        C_slmAddrs, C_slmLayout, C_slm, C_slmStrategy, state);

        if (ok) break;

        state.ra.safeRelease(C_load);
    }

    if (sliceRegs <= 0) throw out_of_registers_exception();

    setupAddr(Tc, C_slmAddrs, addr0, C_slmLayout, Subregister(), C_slm,
            C_slmStrategy, strategy, state);

    if (initialBarrier) barrierwait();

    // Loop over slices.
    for (int rr = 0; rr < regs; rr += sliceRegs) {
        Label lSkipWrite, lSkipReduce, lTop;

        int nreg = std::min(sliceRegs, regs - rr);
        auto C_range = state.C_regs[0].subrange(rr, nreg);

        if (rr > 0) slmBarrier(barrierTemp, r0_info);

        // Trim down SLM layout for final loop.
        if (nreg < sliceRegs) {
            vector<RegisterBlock> sublayout;
            vector<GRFRange> subaddrs;
            if (!getSubblocks(Tc, sublayout, subaddrs, C_slmLayout, C_slmAddrs,
                        true, 0, nreg, true, C_slm, C_slmStrategy))
                stub();
            std::swap(sublayout, C_slmLayout);
            std::swap(subaddrs, C_slmAddrs);
        }

        // Non-leaders write to SLM.
        jmpi(1 | state.flagAP, lSkipWrite);
        storeMatrix(C_range, C_slmLayout, C_slm, C_slmStrategy, C_slmAddrs,
                strategy, state);
        mark(lSkipWrite);

        slmBarrier(barrierTemp, r0_info);

        // Leader reads SLM data and accumulates C.
        jmpi(1 | ~state.flagAP, lSkipReduce);
        add(1, kt, state.lszK, -1);
        incAddr(C_slmAddrs, unrollKSLMStride, C_slmLayout, C_slm, C_slmStrategy,
                strategy, state);

        mark(lTop);
        add(1 | gt | flagKTLoop, kt, kt, -1);
        loadMatrix(C_load, C_slmLayout, C_slm, C_slmStrategy, C_slmAddrs,
                strategy, state);
        incAddr(C_slmAddrs, unrollKSLMStride, C_slmLayout, C_slm, C_slmStrategy,
                strategy, state);
        map(hw, Tc.real(), C_range, C_load, strategy,
                [&](int simd, GRF r1, GRF r2) { add(simd, r1, r1, r2); });
        jmpi(1 | flagKTLoop, lTop);

        if (rr + nreg < regs)
            incAddr(C_slmAddrs, unrollKSLMReturn, C_slmLayout, C_slm,
                    C_slmStrategy, strategy, state);

        mark(lSkipReduce);
    }

    // Followers will not update C.
    mov(1 | ~state.flagAP, state.remainders[LoopM], 0);
    mov(1 | ~state.flagAP, state.remainders[LoopN], 0);

    state.raVFlag.safeRelease(flagKTLoop);
    state.ra.safeRelease(C_load);
    state.ra.safeRelease(kt);
    state.ra.safeRelease(unrollKSLMReturn);
    state.ra.safeRelease(addr0);
    state.ra.safeRelease(barrierTemp);
    safeReleaseRanges(C_slmAddrs, state);

    mark(lDone);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::gemmPrefetchC(
        const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state) {
    auto Tc_ext = problem.Tc_ext;
    bool checkBeta0 = problem.checkBeta0 && !problem.beta_real.fixed();
    bool checkIDK = strategy.kParallelLocal;

    releaseRanges(state.Ap_regs, state);
    releaseRanges(state.Bp_regs, state);

    status << "Prefetch C" << status_stream::endl;

    if (checkBeta0) {
        cmp0(1 | eq | state.flagAP, problem.beta_real.getReg(0));
    }

    Address2DParams Cp_params;
    if (strategy.C.address2D) {
        Cp_params.rows = state.inputs.m;
        Cp_params.cols = state.inputs.n;
        Cp_params.offR = state.i0;
        Cp_params.offC = state.j0;
    } else {
        Cp_params.rows = state.remainders[LoopM];
        Cp_params.cols = state.remainders[LoopN];
    }
    Cp_params.remR = state.remainders[LoopM];
    Cp_params.remC = state.remainders[LoopN];

    bool oldAdd32 = strategy.emulate.emulate64_add32;
    strategy.emulate.emulate64_add32 = false;

    gemmCacheLDCMultiples(problem, strategy, state, 1);

    if (checkIDK) {
        if (checkBeta0)
            cmp(1 | ~state.flagAP | gt | state.flagAP, state.lidK, 0);
        else
            cmp(1 | gt | state.flagAP, state.lidK, 0);
    }

    allocAddrRegs(state.Cp_addrs, state.Cp_layout, problem.C,
            strategy.C_prefetch, state);
    setupAddr(Tc_ext, state.Cp_addrs, state.effCp, state.Cp_layout,
            state.inputs.ldc[0], problem.C, strategy.C_prefetch, strategy,
            state, Cp_params, state.ldcMultiples[0]);

    Label lSkipPrefetchC;
    if (checkBeta0 || checkIDK) jmpi(1 | state.flagAP, lSkipPrefetchC);

    state.Cp_regs = state.ra.alloc_range(getRegCount(state.Cp_layout));

    loadMatrix(state.Cp_regs, state.Cp_layout, problem.C, strategy.C_prefetch,
            state.Cp_addrs, strategy, state);

    safeReleaseRanges(state.Cp_regs, state);
    safeReleaseRanges(state.Cp_addrs, state);
    if (state.effCp != state.effC[0]) state.ra.safeRelease(state.effCp);

    releaseLDMultiples(state.ldcMultiples[0], state);
    releaseIndexVec(state);

    if (checkBeta0 || checkIDK) mark(lSkipPrefetchC);

    strategy.emulate.emulate64_add32 = oldAdd32;

    reclaimRanges(state.Ap_regs, state);
    reclaimRanges(state.Bp_regs, state);
}

// Generate code for checking whether 32-bit address arithmetic can be used inside k loop.
// Assumes leading dimensions have not been shifted yet.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmCheck32(
        const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state) {
    if (!strategy.checkAdd32) return;

    bool checkA = (strategy.A.base.getModel() == ModelA64);
    bool checkB = (strategy.B.base.getModel() == ModelA64);
    if (!checkA && !checkB) return;

    auto &m = state.inputs.m;
    auto &n = state.inputs.n;
    auto &k = state.fullK.isValid() ? state.fullK : state.inputs.k;
    auto &lda = state.inputs.lda;
    auto &ldb = state.inputs.ldb;
    auto temp1GRF = state.ra.alloc();
    auto temp2GRF = state.ra.alloc();
    auto temp1 = temp1GRF.ud(
            0); // Only need one :ud subregister. But GRF-align it for mach.
    auto temp2 = temp2GRF.ud(0);
    auto temp3 = temp2GRF.ud(4);
    auto flag = state.raVFlag.alloc();

    if (checkA) {
        add(1, temp2, state.effA.ud(), state.offsetA.ud());
        switch (problem.A
                        .layout) { // Conservatively estimate upper bound for size of A.
            case MatrixLayout::N: emul32High(1, temp1, lda, k); break;
            case MatrixLayout::T: emul32High(1, temp1, lda, m); break;
            case MatrixLayout::Pc: {
                if (strategy.fixedWG(problem))
                    add(1, temp3, m,
                            uint16_t(strategy.wg[LoopM] * strategy.unroll[LoopM]
                                    - 1));
                else
                    emad(1, temp3, m, state.inputs.localSizeM,
                            strategy.unroll[LoopM], strategy, state);
                emul32High(1, temp1, lda, temp3);
                break;
            }
            default: stub();
        }
        add(1 | ov | flag, temp2, acc0.ud(0), temp2);
        cmp(1 | ~flag | ne | flag, temp1, uint16_t(0));
    }

    if (checkB) {
        add(1, temp2, state.effB.ud(), state.offsetB.ud());
        switch (problem.B.layout) {
            case MatrixLayout::T: emul32High(1, temp1, ldb, k); break;
            case MatrixLayout::N: emul32High(1, temp1, ldb, n); break;
            case MatrixLayout::Pr: {
                if (strategy.fixedWG(problem))
                    add(1, temp3, n,
                            uint16_t(strategy.wg[LoopN] * strategy.unroll[LoopN]
                                    - 1));
                else
                    emad(1, temp3, n, state.inputs.localSizeN,
                            strategy.unroll[LoopN], strategy, state);
                emul32High(1, temp1, ldb, temp3);
                break;
            }
            default: stub();
        }
        InstructionModifier mod = 1;
        if (checkA) mod |= ~flag;
        add(mod | ov | flag, temp2, acc0.ud(0), temp2);
        cmp(1 | ~flag | ne | flag, temp1, uint16_t(0));
    }

    state.add64 = state.ra.alloc_sub<uint16_t>();
    and_(1, state.add64, flag, 1u);
    state.raVFlag.safeRelease(flag);

    state.ra.safeRelease(temp1GRF);
    temp1 = invalid;
    state.ra.safeRelease(temp2GRF);
    temp2 = invalid;
    temp3 = invalid;
}

// Increment A pointer after load, inside GEMM k loop.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmAIncrementInternal(Type Ta,
        const std::vector<RegisterBlock> &layout,
        const std::vector<GRFRange> &addrs, const MatrixAddressing &A,
        const MatrixAddressingStrategy &A_strategy, int ka_inc,
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state, int ha) {
    if (ka_inc == 0)
        /* no-op */;
    else if (A_strategy.address2D)
        incDecAddr(addrs, Subregister(), 0, ka_inc, layout, A, A_strategy,
                strategy, state, problem.backward());
    else if (A.layout == MatrixLayout::N) {
        SubregisterPair lda_ka;
        bool release = false;
        // Use cached lda * ka_inc if available, otherwise calculate on the fly.
        if (ka_inc == 1)
            lda_ka = state.lda;
        else if (state.lda_ka.isValid() && ka_inc == state.ka_cached)
            lda_ka = state.lda_ka;
        else if (state.lda_ka_prefetch.isValid()
                && ka_inc == strategy.ka_pfStride)
            lda_ka = state.lda_ka_prefetch;
        else {
            lda_ka = state.ra.alloc_sub<int32_t>();
            emulConstant(1, lda_ka, state.inputs.lda, ka_inc, strategy, state);
            release = true;
        }
        incDecAddr(addrs, lda_ka, layout, A, A_strategy, strategy, state,
                problem.backward());
        if (release) state.ra.safeRelease(lda_ka);
    } else {
        int incA;
        switch (A.layout) {
            case MatrixLayout::Pc:
                incA = untile(A, 0, ha + ka_inc, A.packSize,
                               strategy.unrollKSLM)
                        - untile(A, 0, ha, A.packSize, strategy.unrollKSLM);
                break;
            case MatrixLayout::T: incA = ka_inc; break;
            default: stub();
        }
        incDecAddr(addrs, uint16_t(incA * Ta), layout, A, A_strategy, strategy,
                state, problem.backward());
    }
}

template <HW hw>
void gemm_kernel_generator_t<hw>::gemmAIncrementInternal(Type Ta,
        const std::vector<RegisterBlock> &layout,
        const std::vector<GRFRange> &addrs, const MatrixAddressing &A,
        const MatrixAddressingStrategy &A_strategy,
        const MultishiftSubregister &ka_inc, const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state, int ha) {
    incDecAddr(addrs, ka_inc, layout, A, A_strategy, strategy, state,
            problem.backward());
}

template <HW hw>
void gemm_kernel_generator_t<hw>::gemmAIncrementInternal(Type Ta,
        const std::vector<RegisterBlock> &layout,
        const std::vector<GRFRange> &addrs, const MatrixAddressing &A,
        const MatrixAddressingStrategy &A_strategy, const Subregister &ka_inc,
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state, int ha) {
    incDecAddr(addrs, ka_inc, 0, ka_inc, layout, A, A_strategy, strategy, state,
            problem.backward());
}

template <HW hw>
template <typename I>
void gemm_kernel_generator_t<hw>::gemmAIncrement(Type Ta,
        const std::vector<RegisterBlock> &layout,
        const std::vector<GRFRange> &addrs, const MatrixAddressing &A,
        const MatrixAddressingStrategy &A_strategy, I ka_inc,
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state, int ha) {
    gemmAIncrementInternal(Ta, layout, addrs, A, A_strategy, ka_inc, problem,
            strategy, state, ha);
}

// A load for GEMM k loop.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmALoad(const GRFMultirange &regs,
        const std::vector<RegisterBlock> &layout,
        const std::vector<GRFRange> &addrs, const MatrixAddressing &A,
        const MatrixAddressingStrategy &A_strategy, const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state) {
    loadMatrix(regs, layout, A, A_strategy, addrs, strategy, state);
}

template <HW hw>
template <typename I>
void gemm_kernel_generator_t<hw>::gemmALoadInc(Type Ta,
        const GRFMultirange &regs, const std::vector<RegisterBlock> &layout,
        const std::vector<GRFRange> &addrs, const MatrixAddressing &A,
        const MatrixAddressingStrategy &A_strategy, I ka_inc,
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
    gemmALoad(regs, layout, addrs, A, A_strategy, problem, strategy, state);
    gemmAIncrement(
            Ta, layout, addrs, A, A_strategy, ka_inc, problem, strategy, state);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::gemmBIncrementInternal(Type Tb,
        const std::vector<RegisterBlock> &layout,
        const std::vector<GRFRange> &addrs, const MatrixAddressing &B,
        const MatrixAddressingStrategy &B_strategy, int kb_inc,
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state, int hb) {
    if (kb_inc == 0)
        /* no-op */;
    else if (B_strategy.address2D)
        incDecAddr(addrs, Subregister(), kb_inc, 0, layout, B, B_strategy,
                strategy, state, problem.backward());
    else if (B.layout == MatrixLayout::T) {
        SubregisterPair ldb_kb;
        bool release = false;
        if (kb_inc == 1)
            ldb_kb = state.ldb;
        else if (state.ldb_kb.isValid() && kb_inc == state.kb_cached)
            ldb_kb = state.ldb_kb;
        else if (state.ldb_kb_prefetch.isValid()
                && kb_inc == strategy.kb_pfStride)
            ldb_kb = state.ldb_kb_prefetch;
        else {
            ldb_kb = state.ra.alloc_sub<int32_t>();
            emulConstant(1, ldb_kb, state.inputs.ldb, kb_inc, strategy, state);
            release = true;
        }
        incDecAddr(addrs, ldb_kb, layout, B, B_strategy, strategy, state,
                problem.backward());
        if (release) state.ra.safeRelease(ldb_kb);
    } else {
        int incB;
        switch (B.layout) {
            case MatrixLayout::Pr:
                incB = untile(B, hb + kb_inc, 0, strategy.unrollKSLM,
                               B.packSize)
                        - untile(B, hb, 0, strategy.unrollKSLM, B.packSize);
                break;
            case MatrixLayout::N: incB = kb_inc; break;
            default: stub();
        }
        incDecAddr(addrs, uint16_t(incB * Tb), layout, B, B_strategy, strategy,
                state, problem.backward());
    }
}

template <HW hw>
void gemm_kernel_generator_t<hw>::gemmBIncrementInternal(Type Tb,
        const std::vector<RegisterBlock> &layout,
        const std::vector<GRFRange> &addrs, const MatrixAddressing &B,
        const MatrixAddressingStrategy &B_strategy,
        const MultishiftSubregister &kb_inc, const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state, int hb) {
    incDecAddr(addrs, kb_inc, layout, B, B_strategy, strategy, state,
            problem.backward());
}

template <HW hw>
void gemm_kernel_generator_t<hw>::gemmBIncrementInternal(Type Tb,
        const std::vector<RegisterBlock> &layout,
        const std::vector<GRFRange> &addrs, const MatrixAddressing &B,
        const MatrixAddressingStrategy &B_strategy, const Subregister &kb_inc,
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state, int hb) {
    incDecAddr(addrs, kb_inc, kb_inc, 0, layout, B, B_strategy, strategy, state,
            problem.backward());
}

template <HW hw>
template <typename I>
void gemm_kernel_generator_t<hw>::gemmBIncrement(Type Tb,
        const std::vector<RegisterBlock> &layout,
        const std::vector<GRFRange> &addrs, const MatrixAddressing &B,
        const MatrixAddressingStrategy &B_strategy, I kb_inc,
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state, int hb) {
    gemmBIncrementInternal(Tb, layout, addrs, B, B_strategy, kb_inc, problem,
            strategy, state, hb);
}

// B load for GEMM k loop.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmBLoad(const GRFMultirange &regs,
        const std::vector<RegisterBlock> &layout,
        const std::vector<GRFRange> &addrs, const MatrixAddressing &B,
        const MatrixAddressingStrategy &B_strategy, const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state) {
    loadMatrix(regs, layout, B, B_strategy, addrs, strategy, state);
}

template <HW hw>
template <typename I>
void gemm_kernel_generator_t<hw>::gemmBLoadInc(Type Tb,
        const GRFMultirange &regs, const std::vector<RegisterBlock> &layout,
        const std::vector<GRFRange> &addrs, const MatrixAddressing &B,
        const MatrixAddressingStrategy &B_strategy, I kb_inc,
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
    gemmBLoad(regs, layout, addrs, B, B_strategy, problem, strategy, state);
    gemmBIncrement(
            Tb, layout, addrs, B, B_strategy, kb_inc, problem, strategy, state);
}

template <HW hw>
template <bool doA>
void gemm_kernel_generator_t<hw>::gemmAiBiRemLoadInc(bool incremental,
        bool incrementalCopy, bool keepAddrTogether, bool willRemask,
        const Subregister &kSLMX, const GRFMultirange &Xi_regs,
        const vector<RegisterBlock> &Xi_layout,
        const vector<GRFRange> &Xi_addrs,
        const vector<vector<RegisterBlock>> &Xi_layoutK,
        const vector<vector<GRFRange>> &Xi_addrsK, const GRFMultirange &Xo_regs,
        const vector<RegisterBlock> &Xo_layout, const MatrixAddressing &Xi,
        const MatrixAddressingStrategy &Xi_strategy, const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state) {
    auto T = doA ? problem.Ta : problem.Tb;
    auto T_ext = doA ? problem.Ta_ext : problem.Tb_ext;
    auto kx_slm = doA ? state.ka_slm : state.kb_slm;

    auto unrollKSLM = strategy.unrollKSLM;

    bool prezero = !willRemask
            && ((problem.abOffset == ABOffset::Calc)
                    || (minOuterProductCount(hw, problem, strategy) > 1));

    if (!incremental) {
        if (prezero) zeroMatrix(Xi_regs, strategy);
        doA ? gemmALoad(Xi_regs, Xi_layout, Xi_addrs, Xi, Xi_strategy, problem,
                strategy, state)
            : gemmBLoad(Xi_regs, Xi_layout, Xi_addrs, Xi, Xi_strategy, problem,
                    strategy, state);
    } else {
        bool simtCF = strategy.fused
                && (strategy.fusedLoop == (doA ? LoopN : LoopM));
        int simt = simtCF ? 16 : 1;
        Label done;

        keepAddrTogether &= (Xi_addrsK.size() > 1);

        if (problem.backward() && incrementalCopy && kx_slm > 1) {
            // Adjust pointer from main loop.
            doA ? gemmAIncrement(T_ext, Xi_layoutK[0], Xi_addrsK[0], Xi,
                    Xi_strategy, 1 - kx_slm, problem, strategy, state)
                : gemmBIncrement(T_ext, Xi_layoutK[0], Xi_addrsK[0], Xi,
                        Xi_strategy, 1 - kx_slm, problem, strategy, state);
        }

        cmp(simt | gt | state.flagAP, kSLMX, 0);
        add(1, kSLMX, kSLMX, (kx_slm > 1) ? -1 : -unrollKSLM);

        if (prezero) zeroMatrix(incrementalCopy ? Xo_regs : Xi_regs, strategy);

        for (int hh = 0; hh < kx_slm; hh++) {
            int hhRem = kx_slm - hh - 1;

            simtCF ? goto12(16 | ~state.flagAP, done)
                   : jmpi(1 | ~state.flagAP, done);

            if (hhRem > 0) {
                cmp(simt | gt | state.flagAP, kSLMX, 0);
                add(1, kSLMX, kSLMX,
                        (hhRem == 1) ? -(unrollKSLM - kx_slm + 1) : -1);
            }

            int hh_eff = problem.backward() ? (kx_slm - 1 - hh) : hh;
            int hh_layout = hh_eff;
            int hh_addr = hh_eff;

            if (Xi_layoutK.size() == 1) hh_layout = 0;
            if (Xi_addrsK.size() == 1) hh_addr = 0;

            // OPTIMIZEME: delay inc if kx_slm = 1
            auto kx_inc = (Xi_addrsK.size() > 1)
                    ? unrollKSLM
                    : ((hh + 1) != kx_slm) ? 1 : (unrollKSLM - kx_slm + 1);

            if (keepAddrTogether) kx_inc = 0;

            doA ? gemmALoadInc(T_ext, Xi_regs, Xi_layoutK[hh_layout],
                    Xi_addrsK[hh_addr], Xi, Xi_strategy, kx_inc, problem,
                    strategy, state)
                : gemmBLoadInc(T_ext, Xi_regs, Xi_layoutK[hh_layout],
                        Xi_addrsK[hh_addr], Xi, Xi_strategy, kx_inc, problem,
                        strategy, state);

            if (incrementalCopy) {
                int rr_eff = doA ? 0 : hh_eff;
                int cc_eff = doA ? hh_eff : 0;
                copyRegisters(T_ext, T, Xi_layoutK[hh_layout], Xo_layout,
                        Xi_regs, Xo_regs, rr_eff, cc_eff, false, strategy,
                        state);
            }
        }

        mark(done);
        if (simtCF) join(16);

        if (keepAddrTogether) {
            doA ? gemmAIncrement(T_ext, Xi_layout, Xi_addrs, Xi, Xi_strategy,
                    unrollKSLM, problem, strategy, state)
                : gemmBIncrement(T_ext, Xi_layout, Xi_addrs, Xi, Xi_strategy,
                        unrollKSLM, problem, strategy, state);
        }
    }
}

// Calculate A offset for SLM copies or cooperative prefetches for this local ID.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmCalcWorkshareAOffset(Subregister &off,
        Subregister &offR, Subregister &offC, const MatrixAddressing &A,
        const MatrixAddressingStrategy &A_strategy, int ma, int ka,
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
    bool splitM = (state.effCoopA == CoopSplit::MN);
    bool splitLinear = (state.effCoopA == CoopSplit::Linear);

    if (A_strategy.address2D) {
        if (splitLinear) stub();
        if (splitM) {
            offR = state.ra.alloc_sub<uint32_t>(
                    getHint(HintType::TempComp0, strategy));
            mulConstant(1, offR, state.lidN, ma);
        } else {
            offC = state.ra.alloc_sub<uint32_t>(
                    getHint(HintType::TempComp0, strategy));
            mulConstant(1, offC, state.lidN, ka);
        }
    } else {
        auto Ta_ext = problem.Ta_ext;
        off = state.ra.alloc_sub<uint32_t>(
                getHint(HintType::TempComp0, strategy));

        switch (A.layout) {
            case MatrixLayout::Pc:
                mulConstant(1, off, state.lidN, ma * ka * Ta_ext);
                break;
            case MatrixLayout::T:
                if (splitLinear) stub();
                if (splitM) {
                    mul(1, off, state.inputs.lda, state.lidN);
                    mulConstant(1, off, off, ma);
                } else
                    mulConstant(1, off, state.lidN, ka * Ta_ext);
                break;
            case MatrixLayout::N:
                if (splitLinear) stub();
                if (splitM)
                    mulConstant(1, off, state.lidN, ma * Ta_ext);
                else {
                    mul(1, off, state.inputs.lda, state.lidN);
                    mulConstant(1, off, off, ka);
                }
                break;
            default: stub();
        }
    }
}

// Calculate B offset for SLM copies or cooperative prefetches for this local ID.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmCalcWorkshareBOffset(Subregister &off,
        Subregister &offR, Subregister &offC, const MatrixAddressing &B,
        const MatrixAddressingStrategy &B_strategy, int kb, int nb,
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
    bool splitN = (state.effCoopB == CoopSplit::MN);
    bool splitLinear = (state.effCoopB == CoopSplit::Linear);

    if (B_strategy.address2D) {
        if (splitLinear) stub();
        if (splitN) {
            offC = state.ra.alloc_sub<uint32_t>(
                    getHint(HintType::TempComp0, strategy));
            mulConstant(1, offC, state.lidM, nb);
        } else {
            offR = state.ra.alloc_sub<uint32_t>(
                    getHint(HintType::TempComp0, strategy));
            mulConstant(1, offR, state.lidM, kb);
        }
    } else {
        auto Tb_ext = problem.Tb_ext;
        off = state.ra.alloc_sub<uint32_t>(
                getHint(HintType::TempComp0, strategy));

        switch (B.layout) {
            case MatrixLayout::Pr:
                mulConstant(1, off, state.lidM, nb * kb * Tb_ext);
                break;
            case MatrixLayout::N:
                if (splitLinear) stub();
                if (splitN) {
                    mul(1, off, state.inputs.ldb, state.lidM);
                    mulConstant(1, off, off, nb);
                } else
                    mulConstant(1, off, state.lidM, kb * Tb_ext);
                break;
            case MatrixLayout::T:
                if (splitLinear) stub();
                if (splitN)
                    mulConstant(1, off, state.lidM, nb * Tb_ext);
                else {
                    mul(1, off, state.inputs.ldb, state.lidM);
                    mulConstant(1, off, off, kb);
                }
                break;
            default: stub();
        }
    }
}

// Remask incoming global data for SLM copies.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmSLMRemask(bool remaskA, bool remaskB,
        GRFMultirange &Ao_regs, GRFMultirange &Bo_regs, int kOffset,
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
    if (problem.backward()) stub();

    auto Ta = problem.Ta, Tb = problem.Tb;

    bool oremaskA = remaskA && (state.effCoopA == CoopSplit::K);
    bool oremaskB = remaskB && (state.effCoopB == CoopSplit::K);
    bool noshareRemask = (oremaskA || oremaskB)
            || (remaskA && remaskB && Ta.size() != Tb.size());
    int aRemaskLen = state.ka_slm;
    int bRemaskLen = state.kb_slm;

    Subregister offK_A, offK_B;
    if (oremaskA) {
        offK_A = state.ra.alloc_sub<uint32_t>();
        mulConstant(1, offK_A, state.lidN, state.ka_slm);
    }

    if (oremaskB) {
        offK_B = state.ra.alloc_sub<uint32_t>();
        mulConstant(1, offK_B, state.lidM, state.kb_slm);
    }

    if (!noshareRemask && remaskA && remaskB)
        aRemaskLen = bRemaskLen = std::max(aRemaskLen, bRemaskLen);

    if (remaskA) {
        setupTeardownRemask(Ta, 1, true, aRemaskLen, state.K, strategy, state,
                kOffset, offK_A);
        remaskLayout(Ta, 1, true, state.Ao_layout, Ao_regs, strategy, state);
        if (noshareRemask || !remaskB)
            setupTeardownRemask(Ta, 1, false, aRemaskLen, state.K, strategy,
                    state, kOffset, offK_A);
    }

    if (remaskB) {
        if (noshareRemask || !remaskA)
            setupTeardownRemask(Tb, 1, true, bRemaskLen, state.K, strategy,
                    state, kOffset, offK_B);
        remaskLayout(Tb, 1, false, state.Bo_layout, Bo_regs, strategy, state);
        setupTeardownRemask(Tb, 1, false, bRemaskLen, state.K, strategy, state,
                kOffset, offK_B);
    }
}

// Calculate kSLMA/kSLMB -- countdown variables for SLM copies.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmCalcKSLM(const Subregister &kSLM,
        const Subregister &lid, int kgran, int kdiv, int krep,
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
    if (kdiv == 1)
        mov(1, kSLM, state.K);
    else {
        auto modLID = lid;
        if (krep > 1) {
            if (!is_zero_or_pow2(krep)) stub();
            modLID = state.ra.alloc_sub<uint16_t>();
            shr(1, modLID, lid, log2(krep));
        }
        if (!problem.backward())
            emad(1, kSLM, state.K.w(), -modLID, kgran, strategy, state);
        else {
            emad(1, kSLM, strategy.unrollKSLM - kgran, -modLID, kgran, strategy,
                    state);
            add(1, kSLM, state.K, -kSLM);
        }
        if (krep > 1) state.ra.safeRelease(modLID);
    }
}

// Calculate barrier count for a k loop.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmCalcKLoopBarrierCount(Subregister &count,
        const Subregister &k, int cooldown, const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state) {
    int barrierFreq = strategy.barrierFreq;
    int unrollK = strategy.unroll[LoopK];
    int unrollKSLM = strategy.unrollKSLM;

    if (count.isInvalid()) count = state.ra.alloc_sub<uint32_t>();

    if (barrierFreq > 0) {
        if (!is_zero_or_pow2(barrierFreq)) stub();

        if (strategy.splitBarrier && cooldown > 0)
            cmp(1 | ge | state.flagAP, k, cooldown);
        add(1 | sat, count, k, barrierFreq - cooldown - unrollK);
        shr(1, count, count, uint16_t(log2(barrierFreq)));
        if (strategy.splitBarrier) {
            (cooldown > 0) ? add(1 | state.flagAP, count, count, 1)
                           : add(1, count, count, 1);
        }
    } else if (strategy.slmBuffers > 0) {
        if (!is_zero_or_pow2(unrollKSLM)) stub();

        if (strategy.slmBuffers == 1) {
            add(1 | sat, count, k, unrollKSLM - 1);
            if (unrollKSLM == 2)
                and_(1, count, count, ~uint32_t(1));
            else {
                shr(1, count, count, uint16_t(log2(unrollKSLM)));
                shl(1, count, count, 1);
            }
        } else {
            add(1 | sat, count, k, unrollKSLM - 1);
            shr(1, count, count, uint16_t(log2(unrollKSLM)));
        }
    } else
        mov(1, count, 0);
}

int maxExtraKLoopRemBarriers(const GEMMStrategy &strategy) {
    if (strategy.slmBuffers == 2)
        return div_up(strategy.unroll[LoopK], strategy.unrollKSLM);
    return 0;
}

static void makeAiBiKCloneLayout(HW hw, bool isA,
        vector<RegisterBlock> &Xi_layout,
        vector<vector<RegisterBlock>> &Xi_layoutK,
        vector<GRFMultirange> &Xi_regsRem, int kx_slm,
        const GEMMStrategy &strategy, GEMMState &state) {
    auto regCountK = getRegCount(Xi_layoutK[0]);
    auto regCount = regCountK * kx_slm;
    auto offsetK = isA ? &RegisterBlock::offsetC : &RegisterBlock::offsetR;

    Xi_layout = Xi_layoutK[0];

    for (int h1 = 1; h1 < kx_slm; h1++) {
        Xi_layoutK[h1] = Xi_layoutK[h1 - 1];
        for (auto &block : Xi_layoutK[h1]) {
            block.offsetBytes += regCountK * GRF::bytes(hw);

            auto oblock = block;
            oblock.*offsetK += h1;
            Xi_layout.push_back(std::move(oblock));
        }
    }

    int extraRegs = regCount - Xi_regsRem[0].getLen();
    if (extraRegs > 0) {
        for (int q = 0; q < strategy.slmCopies; q++)
            Xi_regsRem[q].append(state.ra.alloc_range(extraRegs));
    }
}

// Perform the body of the GEMM computation, updating a block of C.
template <HW hw>
bool gemm_kernel_generator_t<hw>::gemmKLoop(bool lateKLoopCheck,
        GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state) {
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc;
    auto Ta_ext = problem.Ta_ext, Tb_ext = problem.Tb_ext;
    auto Ta_load = state.Ta_load, Tb_load = state.Tb_load;

    bool cLoadAhead = strategy.cLoadAhead;
    auto opCountMain = outerProductCount(hw, problem, strategy);
    auto minOPCount = minOuterProductCount(hw, problem, strategy);
    auto opCountRem = minOPCount;

    auto A_copies = strategy.A_copies;
    auto B_copies = strategy.B_copies;
    auto slmCopies = strategy.slmCopies;
    auto slmBuffers = strategy.slmBuffers;
    auto ka_loadMain = strategy.ka_load;
    auto kb_loadMain = strategy.kb_load;
    auto ka_pfStride = strategy.ka_pfStride;
    auto kb_pfStride = strategy.kb_pfStride;
    bool slmA = strategy.slmA;
    bool slmB = strategy.slmB;
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];
    auto unrollK = strategy.unroll[LoopK];
    auto unrollKSLM = strategy.unrollKSLM;
    auto ka_slm = state.ka_slm;
    auto kb_slm = state.kb_slm;
    bool calcABSums = (problem.abOffset == ABOffset::Calc);

    bool needBarrier = (slmA || slmB || strategy.barrierFreq > 0);
    bool needSLMReset = false;

    int curPhase;
    int lastThresh = 0;

    // Get r0 information where needed.
    GRF r0_info;
    if (needBarrier) {
        if (state.r0_info.isARF()) stub();
        r0_info = GRF {state.r0_info.getBase()};
    }

    // Get A/B named barrier IDs.
    GRF barrierHeaderM, barrierHeaderN;
    FlagRegister barrierM, barrierN;
    bool nbM = (strategy.slmA || strategy.barrierFreq)
            && strategy.namedBarriers[LoopM];
    bool nbN = (strategy.slmB || strategy.barrierFreq)
            && strategy.namedBarriers[LoopN];

    if (nbM) {
        barrierHeaderM = state.ra.alloc();

        // Get fN.0 subregister for use with sync.bar.
        barrierM = state.raVFlag.alloc(2); // TODO: unlock these flag registers.
        state.raVFlag.release(FlagRegister {barrierM.getARFBase(), 1});
        barrierM = FlagRegister {barrierM.getARFBase(), 0};

        if (!is_zero_or_pow2(strategy.wg[LoopM])
                || !is_zero_or_pow2(strategy.namedBarriers[LoopM]))
            stub();
        shr(1, barrierHeaderM.uw(4), state.lidM,
                log2(strategy.wg[LoopM]) - log2(strategy.namedBarriers[LoopM]));
    }
    if (nbN) {
        barrierHeaderN = state.ra.alloc();
        barrierN = state.raVFlag.alloc(2);
        state.raVFlag.release(FlagRegister {barrierN.getARFBase(), 1});
        barrierN = FlagRegister {barrierN.getARFBase(), 0};

        if (!is_zero_or_pow2(strategy.wg[LoopN])
                || !is_zero_or_pow2(strategy.namedBarriers[LoopN]))
            stub();
        shr(1, barrierHeaderN.uw(4), state.lidN,
                log2(strategy.wg[LoopN]) - log2(strategy.namedBarriers[LoopN]));
    }
    if (nbM) {
        int threadsPerMBar = strategy.wg[LoopM] * strategy.wg[LoopN]
                / strategy.namedBarriers[LoopM];
        mov(1, barrierHeaderM.uw(5), threadsPerMBar | (threadsPerMBar << 8));
    }
    if (nbN) {
        int threadsPerNBar = strategy.wg[LoopM] * strategy.wg[LoopN]
                / strategy.namedBarriers[LoopN];
        mov(1, barrierHeaderN.uw(5), threadsPerNBar | (threadsPerNBar << 8));
    }
    if (nbM && nbN)
        add(1, barrierHeaderN.uw(4), barrierHeaderN.uw(4),
                strategy.namedBarriers[LoopM]);
    if (nbM) mov(1, barrierM, barrierHeaderM.uw(4));
    if (nbN) mov(1, barrierN, barrierHeaderN.uw(4));

    // Get tokens for barriers/fences.
    int tokenBarrierFence[2] = {-1, -1};
    InstructionModifier modBarrierFence[2];

    if (hw >= HW::Gen12LP) {
        if (strategy.needsBarrier())
            tokenBarrierFence[0] = state.tokenAllocator.tryAlloc();
        if (nbM && nbN) tokenBarrierFence[1] = state.tokenAllocator.tryAlloc();
        for (int q = 0; q < 2; q++)
            if (tokenBarrierFence[q] >= 0)
                modBarrierFence[q] = SBID(tokenBarrierFence[q]);
    }

    // Unified barrier handling for k loop.
    GRF barrierHeader;
    bool barrierReady = false;

    auto allocBarrierHeader = [&]() {
        if (barrierHeader.isInvalid()) {
            barrierHeader = state.ra.alloc();
            barrierReady = false;
        }
    };

    auto getBarrierHeader = [&]() {
        allocBarrierHeader();
        if (!barrierReady) {
            barrierheader(barrierHeader, r0_info);
            barrierReady = true;
        }

        return barrierHeader;
    };

    auto getFenceTemp = [&]() {
        auto temp = state.ra.try_alloc();
        if (temp.isValid()) return temp;
        if (barrierHeader.isValid()) {
            barrierReady = false;
            return barrierHeader;
        }
        throw ngen::out_of_registers_exception();
    };

    auto releaseFenceTemp = [&](GRF temp) {
        if (temp != barrierHeader) state.ra.release(temp);
    };

    // Unified SLM fence handling.
    GRF slmFenceTemp;
    auto slmFenceIssue = [&]() {
        if (hw >= HW::Gen11) {
            slmFenceTemp = getFenceTemp();
            slmfence(modBarrierFence[0], slmFenceTemp, r0_info);
            releaseFenceTemp(slmFenceTemp);
        }
    };

    auto slmFenceWait = [&]() {
        if (hw >= HW::Gen12LP)
            wrdep(slmFenceTemp);
        else if (hw >= HW::Gen11)
            mov<uint32_t>(8, null, slmFenceTemp);
    };

    enum class KBarrierType { Normal, Signal, Wait };
    auto kLoopBarrier = [&](bool withSLMFence,
                                KBarrierType type = KBarrierType::Normal) {
        withSLMFence &= (hw >= HW::Gen11); // No SLM fences needed on Gen9.

        if (withSLMFence && type == KBarrierType::Wait) {
            auto temp = getFenceTemp();
            slmfence(modBarrierFence[0], temp, r0_info);
            (hw >= HW::Gen12LP) ? wrdep(temp) : mov<uint32_t>(8, null, temp);
            releaseFenceTemp(temp);
        }

        if (!nbM && !nbN) {
            if (type != KBarrierType::Wait) {
                allocBarrierHeader();
                auto temp = getFenceTemp();
                if (withSLMFence) {
                    slmfence(modBarrierFence[0], temp, r0_info);
                    (hw >= HW::Gen12LP) ? wrdep(temp)
                                        : mov<uint32_t>(8, null, temp);
                }
                auto header = getBarrierHeader();
                barriermsg(modBarrierFence[0], header);
                releaseFenceTemp(temp);
            }
            if (type != KBarrierType::Signal) barrierwait();
        } else {
            if (type != KBarrierType::Wait) {
                if (withSLMFence) {
                    auto temp = getFenceTemp();
                    slmfence(temp, r0_info);
                    wrdep(temp);
                    releaseFenceTemp(temp);
                }
                if (nbM) barriermsg(modBarrierFence[0], barrierHeaderM);
                if (nbN)
                    barriermsg(modBarrierFence[nbM ? 1 : 0], barrierHeaderN);
            }
            if (type != KBarrierType::Signal) {
                if (nbM) sync.bar(barrierM);
                if (nbN) sync.bar(barrierN);
            }
        }
    };

    // Remainder load preparations.
    vector<RegisterBlock> A_layoutRem, B_layoutRem, Ai_layoutRem, Bi_layoutRem;
    vector<vector<RegisterBlock>> Ai_layoutK, Bi_layoutK;
    vector<GRFRange> A_addrsRem, B_addrsRem, Ai_addrsRem, Bi_addrsRem;
    vector<vector<GRFRange>> Ai_addrsK, Bi_addrsK;
    vector<GRFMultirange> Ai_regsRem, Bi_regsRem;
    int ka_loadRem = 1, kb_loadRem = 1;

    // For packed layouts, extend remainder loads to encompass a full logical block.
    int ignore;
    getGranularities(problem.A, ignore, ka_loadRem);
    getGranularities(problem.B, kb_loadRem, ignore);

    // With 2D block loads, extend k unroll to at least a full block (array).
    bool a2D = isBlock2D(strategy.A.accessType);
    bool b2D = isBlock2D(strategy.B.accessType);
    bool ai2D = strategy.slmA && isBlock2D(state.Ai_strategy.accessType);
    bool bi2D = strategy.slmB && isBlock2D(state.Bi_strategy.accessType);
    if (a2D || ai2D) {
        ka_loadRem = state.A_layout[0].nc;
        if (!isColMajor(problem.A.layout))
            ka_loadRem *= state.A_layout[0].count;
    }
    if (b2D || bi2D) {
        kb_loadRem = state.B_layout[0].nr;
        if (isColMajor(problem.B.layout)) kb_loadRem *= state.B_layout[0].count;
    }

    // Fragment the A, B layouts into smaller blocks (usually 1 row/column) for remainder loads.
    if (!getSubblocks(Ta_load, A_layoutRem, A_addrsRem, state.A_layout,
                state.A_addrs, true, 0, ka_loadRem, strategy.A.padded,
                problem.A, strategy.A))
        return false;
    if (!getSubblocks(Tb_load, B_layoutRem, B_addrsRem, state.B_layout,
                state.B_addrs, false, 0, kb_loadRem, strategy.B.padded,
                problem.B, strategy.B))
        return false;

    // Add k masking.
    if (a2D && (ka_loadRem > 1))
        addMasking(Ta_load, A_layoutRem, false, true, problem.A, strategy.A);
    if (b2D && (kb_loadRem > 1))
        addMasking(Tb_load, B_layoutRem, true, false, problem.B, strategy.B);

    // Ai/Bi remainders.
    Ai_layoutRem = state.Ai_layout;
    Bi_layoutRem = state.Bi_layout;
    Ai_addrsRem = state.Ai_addrs;
    Bi_addrsRem = state.Bi_addrs;
    Ai_regsRem = state.Ai_regs;
    Bi_regsRem = state.Bi_regs;
    bool Ai_hasKRem = false, Ai_lateKRem = false;
    bool Bi_hasKRem = false, Bi_lateKRem = false;
    bool Ai_remIncrCopy = false, Bi_remIncrCopy = false;
    auto Ao_regsRem = state.Ao_regs;
    auto Bo_regsRem = state.Bo_regs;

    if (ai2D && (ka_loadRem > 1) && state.Ai_strategy.address2D) {
        Ai_hasKRem = true;
        addMasking(
                Ta_ext, Ai_layoutRem, false, true, state.Ai, state.Ai_strategy);
    }

    if (bi2D && (kb_loadRem > 1) && state.Bi_strategy.address2D) {
        Bi_hasKRem = true;
        addMasking(
                Tb_ext, Bi_layoutRem, true, false, state.Bi, state.Bi_strategy);
    }

    if (strategy.slmA && !Ai_hasKRem)
        Ai_lateKRem |= !isRegisterColMajor(Ta_ext, state.Ai, state.Ai_strategy);
    if (strategy.slmB && !Bi_hasKRem)
        Bi_lateKRem |= isRegisterColMajor(Tb_ext, state.Bi, state.Bi_strategy);

    bool Ai_incrementalRem = strategy.slmA && !Ai_hasKRem && !Ai_lateKRem;
    bool Bi_incrementalRem = strategy.slmB && !Bi_hasKRem && !Bi_lateKRem;
    bool aioShareRem = state.aioShare;
    bool bioShareRem = state.bioShare;

    if (Ai_incrementalRem) {
        // Prepare to split Ai layout in k dimension. If it's not possible to do in-place, then
        // either redo the layout or copy Ai->Ao incrementally.
        Ai_layoutK.resize(ka_slm);
        Ai_addrsK.resize(ka_slm);
        for (int h = 0; h < ka_slm; h++) {
            bool success = false;

            if (h < int(Ai_addrsK.size())) {
                success = getSubblocks(Ta_ext, Ai_layoutK[h], Ai_addrsK[h],
                        Ai_layoutRem, state.Ai_addrs, true, h, h + 1,
                        state.Ai_strategy.padded, state.Ai, state.Ai_strategy);
            }

            if (!success && h == 0) stub();

            if (!success) {
                // Maybe the subblock is OK, but we didn't get an address register. Try again without
                //  asking for address registers.
                Ai_addrsK.resize(1);
                success = getSubblocks(Ta_ext, Ai_layoutK[h], Ai_layoutRem,
                        true, h, h + 1, state.Ai_strategy.padded, state.Ai,
                        state.Ai_strategy);
            }

            if (!success) {
                // Can't make a subblock. Will need a new layout or an incremental copy.
                if (strategy.slmUseIncrCopy) {
                    Ai_remIncrCopy = true;
                    Ai_layoutK.resize(1);
                } else
                    makeAiBiKCloneLayout(hw, true, Ai_layoutRem, Ai_layoutK,
                            Ai_regsRem, ka_slm, strategy, state);

                aioShareRem = false;
                if (state.aioShare || state.aoReuseA)
                    Ao_regsRem = state.ra.alloc_range(
                            getRegCount(state.Ao_layout));
                break;
            }
        }
    }

    if (Bi_incrementalRem) {
        Bi_layoutK.resize(kb_slm);
        Bi_addrsK.resize(kb_slm);
        for (int h = 0; h < kb_slm; h++) {
            bool success = false;

            if (h < int(Bi_addrsK.size())) {
                success = getSubblocks(Tb_ext, Bi_layoutK[h], Bi_addrsK[h],
                        Bi_layoutRem, state.Bi_addrs, false, h, h + 1,
                        state.Bi_strategy.padded, state.Bi, state.Bi_strategy);
            }

            if (!success && h == 0) stub();

            if (!success) {
                Bi_addrsK.resize(1);
                success = getSubblocks(Tb_ext, Bi_layoutK[h], Bi_layoutRem,
                        false, h, h + 1, state.Bi_strategy.padded, state.Bi,
                        state.Bi_strategy);
            }

            if (!success) {
                if (strategy.slmUseIncrCopy) {
                    Bi_remIncrCopy = true;
                    Bi_layoutK.resize(1);
                } else
                    makeAiBiKCloneLayout(hw, false, Bi_layoutRem, Bi_layoutK,
                            Bi_regsRem, kb_slm, strategy, state);

                bioShareRem = false;
                if (state.bioShare || state.boReuseB)
                    Bo_regsRem = state.ra.alloc_range(
                            getRegCount(state.Bo_layout));
                break;
            }
        }
    }

    // Allocate repack registers if we need to assemble multiple loads for
    //  each outer product calculation.
    // TODO: allow allocation to overlap unneeded A/B registers.
    bool repackARem = state.repackA;
    bool repackBRem = state.repackB;
    int ka_repackRem = state.repackA ? ka_loadRem : 0;
    int kb_repackRem = state.repackB ? kb_loadRem : 0;
    if (minOPCount > 1) {
        int crosspackA, crosspackB, tileM_A, tileK_A, tileK_B, tileN_B;
        std::tie(crosspackA, crosspackB)
                = targetKernelCrosspack(hw, problem, strategy);
        std::tie(tileM_A, tileK_A, tileK_B, tileN_B)
                = targetKernelTiling(hw, problem, strategy);

        if (ka_loadRem < minOPCount) {
            ka_repackRem = minOPCount;
            if (!repackARem) {
                makeUnbackedRegLayout(Ta, state.Ar_layout, unrollM,
                        ka_repackRem, isLayoutColMajor(state.A_layout),
                        crosspackA, tileM_A, tileK_A);
                state.Ar_regs
                        = state.ra.alloc_range(getRegCount(state.Ar_layout),
                                getHint(HintType::A0, strategy));
                repackARem = true;
            }
        }
        if (kb_loadRem < minOPCount) {
            kb_repackRem = minOPCount;
            if (!repackBRem) {
                makeUnbackedRegLayout(Tb, state.Br_layout, kb_repackRem,
                        unrollN, isLayoutColMajor(state.B_layout), crosspackB,
                        tileK_B, tileN_B);
                state.Br_regs
                        = state.ra.alloc_range(getRegCount(state.Br_layout),
                                getHint(HintType::B0, strategy));
                repackBRem = true;
            }
        }
    }

    bool mustActivateRemainder = false;

    bool remActiveA = false, remActiveB = false;
    auto activateABRemainder = [&](bool active, bool doA, bool doB) {
        if (remActiveA == active) doA = false;
        if (remActiveB == active) doB = false;
        if (!active && ((doA && remActiveA) || (doB && remActiveB))) stub();
        if (!doA && !doB) return;

        if (doA) remActiveA = active;
        if (doB) remActiveB = active;

        // Adjust A/B/Ai/Bi addresses if needed.
        if (doA)
            adjustSubblockAddrs(Ta_load, A_layoutRem, A_addrsRem,
                    state.A_layout, state.A_addrs, problem.A, strategy.A,
                    strategy, state);
        if (doB)
            adjustSubblockAddrs(Tb_load, B_layoutRem, B_addrsRem,
                    state.B_layout, state.B_addrs, problem.B, strategy.B,
                    strategy, state);

        if (doA && strategy.slmA && (state.effCoopA == CoopSplit::K) && !ai2D) {
            vector<RegisterBlock> tempLayout;
            vector<GRFRange> tempAddrs;
            if (!getSubblocks(Ta_ext, tempLayout, tempAddrs, state.Ai_layout,
                        state.Ai_addrs, true, 0, 1, state.Ai_strategy.padded,
                        state.Ai, state.Ai_strategy))
                stub();
            adjustSubblockAddrs(Ta_ext, tempLayout, tempAddrs, state.Ai_layout,
                    state.Ai_addrs, state.Ai, state.Ai_strategy, strategy,
                    state);
        }
        if (doB && strategy.slmB && (state.effCoopB == CoopSplit::K) && !bi2D) {
            vector<RegisterBlock> tempLayout;
            vector<GRFRange> tempAddrs;
            if (!getSubblocks(Tb_ext, tempLayout, tempAddrs, state.Bi_layout,
                        state.Bi_addrs, false, 0, 1, state.Bi_strategy.padded,
                        state.Bi, state.Bi_strategy))
                stub();
            adjustSubblockAddrs(Tb_ext, tempLayout, tempAddrs, state.Bi_layout,
                    state.Bi_addrs, state.Bi, state.Bi_strategy, strategy,
                    state);
        }

        if (doA && a2D && (ka_loadRem > 1))
            setAddrRemainder(Ta_load, A_addrsRem, A_layoutRem, Subregister(),
                    state.K, problem.A, strategy.A, strategy, state);
        if (doB && b2D && (kb_loadRem > 1))
            setAddrRemainder(Tb_load, B_addrsRem, B_layoutRem, state.K,
                    Subregister(), problem.B, strategy.B, strategy, state);

        // Recalculate lda_ka/ldb_kb if needed.
        gemmCalcIncrements(
                problem, strategy, state, ka_loadRem, kb_loadRem, doA, doB);
    };

    bool remActiveSLM = false;
    vector<MaskAssignment> kMasks;
    Subregister kSLMStorage;
    Subregister kSLMA, kSLMB; // k remainders for k-split SLM loads
    bool slmRemaskA = false, slmRemaskB = false;

    auto resetKSLM = [&]() {
        state.ra.safeRelease(kSLMStorage);
        kSLMA = kSLMB = invalid;
    };

    auto activateSLMRemainder = [&](bool active, int kOffset = 0) {
        // Calculate or recalculate SLM k remainders as needed.
        if (active && kSLMStorage.isInvalid()) {
            if (Ai_incrementalRem || Bi_incrementalRem)
                kSLMStorage = state.ra.alloc_sub<uint32_t>();

            if (Ai_incrementalRem) {
                kSLMA = kSLMStorage.w(0);
                int kgran, kdiv, krep;
                switch (state.effCoopA) {
                    case CoopSplit::MN:
                        kgran = unrollKSLM;
                        kdiv = 1;
                        krep = strategy.wg[LoopN];
                        break;
                    case CoopSplit::K:
                        kgran = state.ka_slm;
                        kdiv = strategy.wg[LoopN];
                        krep = 1;
                        break;
                    case CoopSplit::Linear:
                        kgran = std::max(state.Ai.crosspack, state.Ai.tileC);
                        kdiv = unrollKSLM / kgran;
                        krep = strategy.wg[LoopN] / kdiv;
                        break;
                    default: stub();
                }
                gemmCalcKSLM(kSLMA, state.lidN, kgran, kdiv, krep, problem,
                        strategy, state);
            }

            if (Bi_incrementalRem) {
                kSLMB = kSLMStorage.w(1);
                int kgran, kdiv, krep;
                switch (state.effCoopB) {
                    case CoopSplit::MN:
                        kgran = unrollKSLM;
                        kdiv = 1;
                        krep = strategy.wg[LoopM];
                        break;
                    case CoopSplit::K:
                        kgran = state.kb_slm;
                        kdiv = strategy.wg[LoopM];
                        krep = 1;
                        break;
                    case CoopSplit::Linear:
                        kgran = std::max(state.Bi.crosspack, state.Bi.tileR);
                        kdiv = unrollKSLM / kgran;
                        krep = strategy.wg[LoopM] / kdiv;
                        break;
                    default: stub();
                }
                gemmCalcKSLM(kSLMB, state.lidM, kgran, kdiv, krep, problem,
                        strategy, state);
            }

            if ((Ai_incrementalRem || Bi_incrementalRem) && kOffset != 0)
                add(2, kSLMStorage.w()(1), kSLMStorage.w()(1), kOffset);
        }

        // k mask information.
        Subregister rems[3]
                = {state.remainders[LoopM], state.remainders[LoopN], state.K};
        int offsets[3] = {0, 0, -kOffset};

        // If not changing between main loop and remainder, update k masks as needed and return.
        if (remActiveSLM == active) {
            if (active) {
                state.wipeActiveVFlags();
                loadMasks(kMasks, rems, offsets, strategy, state);
            }
            return;
        }

        // Not possible to deactivate remainder path with late k remainder.
        if (!active && remActiveSLM && (Ai_lateKRem || Bi_lateKRem)) stub();
        remActiveSLM = active;

        // Start using k masks if needed.
        if (Ai_lateKRem && !state.Ai_strategy.padded) {
            Ai_layoutRem = state.Ai_layout;
            Ai_addrsRem = state.Ai_addrs;
            addMasking(Ta_ext, Ai_layoutRem, Ai_addrsRem, state.inputs.lda,
                    false, true, state.Ai, state.Ai_strategy, strategy, state);
            if (!assignMasks(Ai_layoutRem, LoopM, LoopK, kMasks, state)) stub();
            if (state.aioShare && Ao_regsRem.empty()
                    && Ai_layoutRem[0].crosspack
                            != state.Ai_layout[0].crosspack) {
                aioShareRem = false;
                Ao_regsRem = state.ra.alloc_range(getRegCount(state.Ao_layout));
            }
        }
        if (Bi_lateKRem && !state.Bi_strategy.padded) {
            Bi_layoutRem = state.Bi_layout;
            Bi_addrsRem = state.Bi_addrs;
            addMasking(Tb_ext, Bi_layoutRem, Bi_addrsRem, state.inputs.ldb,
                    true, false, state.Bi, state.Bi_strategy, strategy, state);
            if (!assignMasks(Bi_layoutRem, LoopK, LoopN, kMasks, state)) stub();
            if (state.bioShare && Bo_regsRem.empty()
                    && Bi_layoutRem[0].crosspack
                            != state.Bi_layout[0].crosspack) {
                bioShareRem = false;
                Bo_regsRem = state.ra.alloc_range(getRegCount(state.Bo_layout));
            }
        }

        if (problem.backward())
            for (auto &mask : kMasks)
                mask.reverse(unrollKSLM);

        loadMasks(kMasks, rems, offsets, strategy, state);

        bool asIfMaskedAi = Ai_lateKRem && state.Ai_strategy.padded;
        bool asIfMaskedBi = Bi_lateKRem && state.Bi_strategy.padded;
        slmRemaskA = slmA && (minOPCount > 1) && !Ai_remIncrCopy
                && needsRemask(Ta_ext, true, Ai_layoutRem, state.Ai_strategy,
                        asIfMaskedAi);
        slmRemaskB = slmB && (minOPCount > 1) && !Bi_remIncrCopy
                && needsRemask(Tb_ext, false, Bi_layoutRem, state.Bi_strategy,
                        asIfMaskedBi);
    };

    // Reuse k/k0 for the loop counter, unless nested.
    //  - If k unroll > 1, the loop counter will offset by (unrollK - 1) during the main loop.
    // Then, unless we are assured positive loop count, check for zero main loop count.
    auto kInput = state.k;
    bool matchBarriers = (strategy.kParallelLocal && needBarrier);
    bool saveK = state.isNested || (problem.abOffset != ABOffset::None)
            || matchBarriers;

    state.K = saveK ? state.ra.alloc_sub<int32_t>() : kInput;

    if (saveK) mov(1, state.K, kInput);

    // Zero out A/B sums if needed.
    if (calcABSums) {
        zeroMatrix(state.As_regs, strategy);
        zeroMatrix(state.Bs_regs, strategy);
    }

    // Zero out C, if not loading ahead of time.
    if (!cLoadAhead) {
        for (int i = 0; i < state.C_accCount; i += 2)
            mov<uint32_t>(2 * elementsPerGRF<uint32_t>(hw),
                    AccumulatorRegister(i), uint16_t(0));

        for (int buf = 0; buf < state.C_buffers; buf++)
            zeroMatrix(state.C_regs[buf], strategy);
    }

    LoopSequencer ls;
    using namespace loop_sequencer;

    int slmBufferLA = 0;
    switch (slmBuffers) {
        case 0:
        case 1: slmBufferLA = 0; break;
        case 2:
        case 3: slmBufferLA = 1; break;
        case 4: slmBufferLA = 2; break;
        default: stub();
    }

    int lookaheadALoad = ka_loadMain * (A_copies - 1);
    int lookaheadBLoad = kb_loadMain * (B_copies - 1);
    int lookaheadALoadRem = ka_loadRem * (A_copies - 1);
    int lookaheadBLoadRem = kb_loadRem * (B_copies - 1);
    int lookaheadSLMLoad = unrollKSLM * (slmCopies - 1) + unrollKSLM - 1;
    int lookaheadSLMStore = unrollKSLM * slmBufferLA + 1;

    if (slmA && slmB) {
        if (lookaheadALoad != lookaheadBLoad) stub();
        if (lookaheadALoadRem != lookaheadBLoadRem) stub();
        if (ka_loadMain != kb_loadMain && lookaheadALoad != lookaheadALoadRem)
            stub();
    }

    int lookaheadSLMReload = slmA ? lookaheadALoad : lookaheadBLoad;
    int lookaheadSLMReloadRem = slmA ? lookaheadALoadRem : lookaheadBLoadRem;
    int durationSLMMainLoad = std::max(slmA * ka_loadMain, slmB * kb_loadMain);

    auto A_remActive = [&](Iteration h) {
        return (h.remaining() < ka_loadMain - (h % ka_loadMain));
    };
    auto B_remActive = [&](Iteration h) {
        return (h.remaining() < kb_loadMain - (h % kb_loadMain));
    };
    auto slmRemActive = [&](Iteration h) {
        return (h.remaining() < unrollKSLM - (h % unrollKSLM));
    };
    auto opRemActive = [&](Iteration h) {
        return (h.remaining() < opCountMain - (h % opCountMain));
    };
    auto repackA = [&](Iteration h) {
        return A_remActive(h) ? repackARem : state.repackA;
    };
    auto repackB = [&](Iteration h) {
        return B_remActive(h) ? repackBRem : state.repackB;
    };
    auto ka_load = [&](Iteration h) {
        return A_remActive(h) ? ka_loadRem : ka_loadMain;
    };
    auto kb_load = [&](Iteration h) {
        return B_remActive(h) ? kb_loadRem : kb_loadMain;
    };
    auto A_copy = [&](Iteration h) { return (h / ka_load(h)) % A_copies; };
    auto B_copy = [&](Iteration h) { return (h / kb_load(h)) % B_copies; };
    auto A_regs = [&](Iteration h) -> GRFMultirange & {
        return state.A_regs[A_copy(h)];
    };
    auto B_regs = [&](Iteration h) -> GRFMultirange & {
        return state.B_regs[B_copy(h)];
    };
    auto A_layout = [&](Iteration h) -> vector<RegisterBlock> & {
        return A_remActive(h) ? A_layoutRem : state.A_layout;
    };
    auto B_layout = [&](Iteration h) -> vector<RegisterBlock> & {
        return B_remActive(h) ? B_layoutRem : state.B_layout;
    };
    auto slmCopy = [&](Iteration h) { return (h / unrollKSLM) % slmCopies; };
    auto slmBuffer = [&](Iteration h) { return (h / unrollKSLM) % slmBuffers; };
    auto Ai_layout = [&](Iteration h) -> vector<RegisterBlock> & {
        return slmRemActive(h) ? Ai_layoutRem : state.Ai_layout;
    };
    auto Bi_layout = [&](Iteration h) -> vector<RegisterBlock> & {
        return slmRemActive(h) ? Bi_layoutRem : state.Bi_layout;
    };
    auto Ai_addrs = [&](Iteration h) -> vector<GRFRange> & {
        return slmRemActive(h) ? Ai_addrsRem : state.Ai_addrs;
    };
    auto Bi_addrs = [&](Iteration h) -> vector<GRFRange> & {
        return slmRemActive(h) ? Bi_addrsRem : state.Bi_addrs;
    };
    auto Ai_allRegs = [&](Iteration h) -> vector<GRFMultirange> & {
        return slmRemActive(h) ? Ai_regsRem : state.Ai_regs;
    };
    auto Bi_allRegs = [&](Iteration h) -> vector<GRFMultirange> & {
        return slmRemActive(h) ? Bi_regsRem : state.Bi_regs;
    };
    auto Ai_regs = [&](Iteration h) -> GRFMultirange & {
        return Ai_allRegs(h)[slmCopy(h)];
    };
    auto Bi_regs = [&](Iteration h) -> GRFMultirange & {
        return Bi_allRegs(h)[slmCopy(h)];
    };
    auto Ao_regs = [&](Iteration h) -> GRFMultirange & {
        return slmRemActive(h) ? Ao_regsRem : state.Ao_regs;
    };
    auto Bo_regs = [&](Iteration h) -> GRFMultirange & {
        return slmRemActive(h) ? Bo_regsRem : state.Bo_regs;
    };
    auto effAo_regs = [&](Iteration h) -> GRFMultirange & {
        return Ao_regs(h).empty() ? Ai_regs(h) : Ao_regs(h);
    };
    auto effBo_regs = [&](Iteration h) -> GRFMultirange & {
        return Bo_regs(h).empty() ? Bi_regs(h) : Bo_regs(h);
    };
    auto aioShare = [&](Iteration h) {
        return slmRemActive(h) ? aioShareRem : state.aioShare;
    };
    auto bioShare = [&](Iteration h) {
        return slmRemActive(h) ? bioShareRem : state.bioShare;
    };
    auto opCount = [&](Iteration h) {
        return opRemActive(h) ? opCountRem : opCountMain;
    };
    auto nothing = [&](Iteration h) {};

    // Dummy task to extend k unroll if needed.
    ls.schedule(every(unrollK) | checkOptional(), nothing);

    // A prefetch.
    auto reqPFA = every(ka_pfStride)
            | duration(
                    strategy.cooperativePF ? ka_pfStride : strategy.ka_prefetch)
            | lookahead(strategy.prefetchA);

    if (strategy.prefetchA && !strategy.slmA) {
        ls.schedule(reqPFA, [&](Iteration h) {
            auto &A_global = strategy.slmA ? state.Ai : problem.A;
            gemmALoad(state.Ap_regs, state.Ap_layout, state.Ap_addrs, A_global,
                    strategy.A_prefetch, problem, strategy, state);
        });
    }

    // B prefetch.
    auto reqPFB = every(kb_pfStride)
            | duration(
                    strategy.cooperativePF ? kb_pfStride : strategy.kb_prefetch)
            | lookahead(strategy.prefetchB);

    if (strategy.prefetchB && !strategy.slmB) {
        ls.schedule(reqPFB, [&](Iteration h) {
            auto &B_global = strategy.slmB ? state.Bi : problem.B;
            gemmBLoad(state.Bp_regs, state.Bp_layout, state.Bp_addrs, B_global,
                    strategy.B_prefetch, problem, strategy, state);
        });
    }

    // SLM loads.
    auto reqSLMLoad = every(unrollKSLM) | variants(slmCopies)
            | lookahead(
                    lookaheadSLMLoad + lookaheadSLMStore + lookaheadSLMReload);
    auto reqSLMLoadABRem = every(unrollKSLM) | variants(slmCopies)
            | lookahead(lookaheadSLMLoad + lookaheadSLMStore
                    + lookaheadSLMReloadRem);
    auto reqSLMStore = every(unrollKSLM) | variants(slmCopies)
            | lookahead(lookaheadSLMStore + lookaheadSLMReload)
            | duration(durationSLMMainLoad);
    auto reqSLMStoreABRem = every(unrollKSLM) | variants(slmCopies)
            | lookahead(lookaheadSLMStore + lookaheadSLMReloadRem);

    if ((slmA || slmB) && mustActivateRemainder) {
        ls.schedule({{reqSLMLoad | duration(unrollKSLM), nothing},
                {reqSLMLoad | unconditional(), [&](Iteration h) {
                     activateSLMRemainder(true, h.counterOffset());
                 }}});
    }

    auto doSLMRemLoad = [&](Iteration h) {
        activateSLMRemainder(true, h.counterOffset());
        if (slmA)
            gemmAiBiRemLoadInc<true>(Ai_incrementalRem, Ai_remIncrCopy,
                    needSLMReset, slmRemaskA, kSLMA, Ai_regs(h), Ai_layoutRem,
                    Ai_addrsRem, Ai_layoutK, Ai_addrsK, Ao_regsRem,
                    state.Ao_layout, state.Ai, state.Ai_strategy, problem,
                    strategy, state);
        if (slmB)
            gemmAiBiRemLoadInc<false>(Bi_incrementalRem, Bi_remIncrCopy,
                    needSLMReset, slmRemaskB, kSLMB, Bi_regs(h), Bi_layoutRem,
                    Bi_addrsRem, Bi_layoutK, Bi_addrsK, Bo_regsRem,
                    state.Bo_layout, state.Bi, state.Bi_strategy, problem,
                    strategy, state);
        if (Ai_incrementalRem || Bi_incrementalRem) lastThresh = 0;
    };

    if (slmA || slmB) {
        ls.schedule({{reqSLMLoad | duration(unrollKSLM),
                             [&](Iteration h) {
                                 activateSLMRemainder(false);
                                 if (slmA)
                                     gemmALoad(Ai_regs(h), state.Ai_layout,
                                             state.Ai_addrs, state.Ai,
                                             state.Ai_strategy, problem,
                                             strategy, state);
                                 if (slmB)
                                     gemmBLoad(Bi_regs(h), state.Bi_layout,
                                             state.Bi_addrs, state.Bi,
                                             state.Bi_strategy, problem,
                                             strategy, state);
                             }},
                {reqSLMLoad | duration(durationSLMMainLoad), doSLMRemLoad},
                {reqSLMLoadABRem, doSLMRemLoad}});
    }

    // Read suppression W/A for fused EU architectures.
    bool rswaA = strategy.readSuppressionWA && (A_copies == 1)
            && ((ka_loadMain <= opCountMain) || state.repackA)
            && hasMasking(state.A_layout);
    bool rswaB = strategy.readSuppressionWA && (B_copies == 1)
            && ((kb_loadMain <= opCountMain) || state.repackB)
            && hasMasking(state.B_layout);
    bool rswaARem = strategy.readSuppressionWA && (A_copies == 1)
            && ((ka_loadRem <= opCountRem) || repackARem)
            && hasMasking(A_layoutRem);
    bool rswaBRem = strategy.readSuppressionWA && (B_copies == 1)
            && ((kb_loadRem <= opCountRem) || repackBRem)
            && hasMasking(B_layoutRem);

    Iteration A_lastRSWA;
    bool haveA_lastRSWA = false;

    bool saveRSWA;
    auto disableRSWA = [&]() {
        saveRSWA = strategy.readSuppressionWA;
        strategy.readSuppressionWA = false;
    };
    auto restoreRSWA = [&]() { strategy.readSuppressionWA = saveRSWA; };

    auto doRSWA_A = [&](Iteration h) {
        A_lastRSWA = h;
        haveA_lastRSWA = true;
        doReadSuppressionWA(strategy, state);
    };

    auto doRSWA_B = [&](Iteration h) {
        if (!(haveA_lastRSWA && A_lastRSWA == h))
            doReadSuppressionWA(strategy, state);
        haveA_lastRSWA = false;
    };

    // A/B load scheduling.
    auto reqLoadA = every(ka_loadMain) | duration(ka_loadMain)
            | variants(A_copies) | lookahead(lookaheadALoad);
    auto reqLoadARem = every(ka_loadRem) | variants(A_copies)
            | lookahead(lookaheadALoadRem);
    auto reqLoadAPrezero = every(minOPCount) | variants(A_copies)
            | lookahead(repackARem ? 0 : lookaheadALoadRem);

    auto reqLoadB = every(kb_loadMain) | duration(kb_loadMain)
            | variants(B_copies) | lookahead(lookaheadBLoad);
    auto reqLoadBRem = every(kb_loadRem) | variants(B_copies)
            | lookahead(lookaheadBLoadRem);
    auto reqLoadBPrezero = every(minOPCount) | variants(B_copies)
            | lookahead(repackBRem ? 0 : lookaheadBLoadRem);

    // A/B prezeroing for partial remainder loads with multi-k outer products.
    bool prezeroARem = !slmA && (ka_loadRem < minOPCount);
    bool prezeroBRem = !slmB && (kb_loadRem < minOPCount);

    if (prezeroARem && prezeroBRem && Ta.isInteger() && Tb.isInteger()
            && !calcABSums) {
        // Only need to pre-zero one operand for integer A/B. Choose the smaller one.
        if (unrollM >= unrollN)
            prezeroARem = false;
        else
            prezeroBRem = false;
    }

    if (prezeroARem)
        ls.schedule({{reqLoadA, nothing},
                {reqLoadAPrezero, [&](Iteration h) {
                     zeroMatrix(
                             repackARem ? state.Ar_regs : A_regs(h), strategy);
                 }}});

    if (prezeroBRem)
        ls.schedule({{reqLoadB, nothing},
                {reqLoadBPrezero, [&](Iteration h) {
                     zeroMatrix(
                             repackBRem ? state.Br_regs : B_regs(h), strategy);
                 }}});

    // A/B enforced remainder preparations.
    if (mustActivateRemainder) {
        ls.schedule({{reqLoadA, nothing},
                {reqLoadARem | unconditional(), [&](Iteration h) {
                     activateABRemainder(true, true, false);
                 }}});
        ls.schedule({{reqLoadB, nothing},
                {reqLoadBRem | unconditional(), [&](Iteration h) {
                     activateABRemainder(true, false, true);
                 }}});
    }

    // A loads.
    ls.schedule({{reqLoadA,
                         [&](Iteration h) {
                             if (rswaA) doRSWA_A(h);
                             disableRSWA();
                             activateABRemainder(false, true, false);
                             gemmALoad(A_regs(h), state.A_layout, state.A_addrs,
                                     problem.A, strategy.A, problem, strategy,
                                     state);
                             restoreRSWA();
                         }},
            {reqLoadARem, [&](Iteration h) {
                 if (rswaARem) doRSWA_A(h);
                 disableRSWA();
                 activateABRemainder(true, true, false);
                 gemmALoad(A_regs(h), A_layoutRem, A_addrsRem, problem.A,
                         strategy.A, problem, strategy, state);
                 restoreRSWA();
             }}});

    // B loads.
    ls.schedule({{reqLoadB,
                         [&](Iteration h) {
                             if (rswaB) doRSWA_B(h);
                             disableRSWA();
                             activateABRemainder(false, false, true);
                             gemmBLoad(B_regs(h), state.B_layout, state.B_addrs,
                                     problem.B, strategy.B, problem, strategy,
                                     state);
                             restoreRSWA();
                         }},
            {reqLoadBRem, [&](Iteration h) {
                 if (rswaBRem) doRSWA_B(h);
                 disableRSWA();
                 activateABRemainder(true, false, true);
                 gemmBLoad(B_regs(h), B_layoutRem, B_addrsRem, problem.B,
                         strategy.B, problem, strategy, state);
                 restoreRSWA();
             }}});

    // Stalls to promote thread switches.
    auto reqStall = every(lcm(ka_loadMain, kb_loadMain)) | checkOptional();

    if (strategy.stallAfterLoad)
        ls.schedule(reqStall, [&](Iteration h) {
            if (hw < HW::Gen12LP)
                mov<uint32_t>(1 | Switch, null, 0);
            else if (Tc.isInteger()) {
                mov<float>(1, null, 0.0f);
                sync.nop(SWSB<float>(1));
            } else {
                mov<uint32_t>(1, null, 0);
                sync.nop(SWSB<uint32_t>(1));
            }
        });

    // k decrement and loop check.
    auto reqLoopCheck = every(unrollK) | duration(unrollK);

    if (lateKLoopCheck)
        reqLoopCheck = reqLoopCheck.delay(
                unrollK - std::min(ka_loadMain, kb_loadMain));

    ls.schedule_if(
            reqLoopCheck,
            [&](Iteration h) {
                add(1 | gt | f0[0], state.K, state.K, -unrollK);
            },
            [&](Iteration h) {
                return (curPhase == LoopSequencer::PhaseMainLoop);
            });

    // SLM store address increments.
    auto doSLMStoreInc = [&](Iteration h) {
        int kIncSLMStore
                = (slmBuffer(h) == slmBuffers - 1) ? -(slmBuffers - 1) : +1;
        kIncSLMStore *= unrollKSLM;
        if (slmA)
            gemmAIncrement(Ta, state.Ao_layout, state.Ao_addrs, state.Ao,
                    state.Ao_strategy, kIncSLMStore, problem, strategy, state);
        if (slmB)
            gemmBIncrement(Tb, state.Bo_layout, state.Bo_addrs, state.Bo,
                    state.Bo_strategy, kIncSLMStore, problem, strategy, state);
    };

    if (strategy.slmBuffers >= 2) {
        ls.schedule({{(reqSLMStore | duration(durationSLMMainLoad)).delay(1),
                             doSLMStoreInc},
                {reqSLMStoreABRem.delay(1), doSLMStoreInc}});
    }

    // SLM load address increments.
    int delaySLMInc = strategy.delayABInc ? (unrollKSLM >> 1) : 0;

    auto doSLMLoadInc = [&](Iteration h) {
        bool fullLoad = (h.remaining() >= (unrollKSLM - delaySLMInc));
        if (slmA && (fullLoad || !Ai_incrementalRem))
            gemmAIncrement(Ta_ext, Ai_layout(h), Ai_addrs(h), state.Ai,
                    state.Ai_strategy, unrollKSLM, problem, strategy, state);
        if (slmB && (fullLoad || !Bi_incrementalRem))
            gemmBIncrement(Tb_ext, Bi_layout(h), Bi_addrs(h), state.Bi,
                    state.Bi_strategy, unrollKSLM, problem, strategy, state);
    };

    auto checkSLMLoadInc = [&](Iteration h) {
        bool fullLoad = (h.remaining() >= (unrollKSLM - delaySLMInc));
        return (slmA && (fullLoad || !Ai_incrementalRem))
                || (slmB && (fullLoad || !Bi_incrementalRem));
    };

    if (slmA || slmB) {
        ls.schedule_if({{(reqSLMLoad | duration(durationSLMMainLoad))
                                        .delay(delaySLMInc),
                                doSLMLoadInc, checkSLMLoadInc},
                {reqSLMLoadABRem.delay(delaySLMInc), doSLMLoadInc,
                        checkSLMLoadInc}});
    }

    // A prefetch address increment.
    int delayAPFInc = strategy.delayABInc ? (ka_pfStride >> 1) : 0;

    if (strategy.prefetchA && !slmA) {
        ls.schedule(reqPFA.delay(delayAPFInc), [&](Iteration h) {
            gemmAIncrement(Ta_ext, state.Ap_layout, state.Ap_addrs, problem.A,
                    strategy.A_prefetch, ka_pfStride, problem, strategy, state);
        });
    }

    // B prefetch address increment.
    int delayBPFInc = strategy.delayABInc ? (kb_pfStride >> 1) : 0;

    if (strategy.prefetchB && !slmB) {
        ls.schedule(reqPFB.delay(delayBPFInc), [&](Iteration h) {
            gemmBIncrement(Tb_ext, state.Bp_layout, state.Bp_addrs, problem.B,
                    strategy.B_prefetch, kb_pfStride, problem, strategy, state);
        });
    }

    // A address increment.
    int delayAInc
            = (strategy.delayABInc && A_copies > 1) ? (ka_loadMain >> 1) : 0;

    auto ka_inc = [&](Iteration h) {
        auto inc = ka_load(h);
        if (slmA) {
            int kWraparound = unrollKSLM * slmBuffers;
            if ((h + inc) % kWraparound < inc) inc -= kWraparound;
        }
        return inc;
    };

    ls.schedule({{reqLoadA.delay(delayAInc),
                         [&](Iteration h) {
                             gemmAIncrement(Ta_load, state.A_layout,
                                     state.A_addrs, problem.A, strategy.A,
                                     ka_inc(h), problem, strategy, state);
                         }},
            {reqLoadARem, [&](Iteration h) {
                 gemmAIncrement(Ta_load, A_layoutRem, A_addrsRem, problem.A,
                         strategy.A, ka_inc(h), problem, strategy, state,
                         h % unrollKSLM);
             }}});

    // B address increment.
    int delayBInc
            = (strategy.delayABInc && B_copies > 1) ? (kb_loadMain >> 1) : 0;

    auto kb_inc = [&](Iteration h) {
        auto inc = kb_load(h);
        if (slmB) {
            int kWraparound = unrollKSLM * slmBuffers;
            if ((h + inc) % kWraparound < inc) inc -= kWraparound;
        }
        return inc;
    };

    ls.schedule({{reqLoadB.delay(delayBInc),
                         [&](Iteration h) {
                             gemmBIncrement(Tb_load, state.B_layout,
                                     state.B_addrs, problem.B, strategy.B,
                                     kb_inc(h), problem, strategy, state);
                         }},
            {reqLoadBRem, [&](Iteration h) {
                 gemmBIncrement(Tb_load, B_layoutRem, B_addrsRem, problem.B,
                         strategy.B, kb_inc(h), problem, strategy, state,
                         h % unrollKSLM);
             }}});

    // A/B remasking in k dimension, during remainder handling.
    bool remaskA = !strategy.slmA && (minOPCount > 1)
            && needsRemask(Ta_load, true, A_layoutRem, strategy.A);
    bool remaskB = !strategy.slmB && (minOPCount > 1)
            && needsRemask(Tb_load, false, B_layoutRem, strategy.B);

    if (remaskA && remaskB && Ta.isInteger() && Tb.isInteger() && !calcABSums) {
        // Only need to remask one operand for integer A/B. Choose the smaller one.
        if (unrollM >= unrollN)
            remaskA = false;
        else
            remaskB = false;
    }

    auto Tremask = remaskA ? Ta_load : Tb_load;
    if (remaskA && remaskB && Ta_load.size() != Tb_load.size()) stub();
    if ((remaskA || remaskB) && problem.backward()) stub();

    int remaskPeriod = lcm(ka_loadRem, kb_loadRem);
    auto reqRemaskSetup = every(remaskPeriod);
    auto reqRemaskA = every(ka_loadRem) | variants(A_copies);
    auto reqRemaskB = every(kb_loadRem) | variants(B_copies);

    if (remaskA || remaskB)
        ls.schedule({{reqRemaskSetup | duration(remaskPeriod), nothing},
                {reqRemaskSetup, [&](Iteration h) {
                     setupTeardownRemask(Tremask, 0, false, remaskPeriod,
                             state.K, strategy, state);
                     setupTeardownRemask(Tremask, 0, true, remaskPeriod,
                             state.K, strategy, state, -h.counterOffset());
                 }}});

    if (remaskA)
        ls.schedule({{reqLoadA, nothing},
                {reqRemaskA, [&](Iteration h) {
                     remaskLayout(Ta_load, 0, true, A_layoutRem, A_regs(h),
                             strategy, state, h % remaskPeriod);
                 }}});

    if (remaskB)
        ls.schedule({{reqLoadB, nothing},
                {reqRemaskB, [&](Iteration h) {
                     remaskLayout(Tb_load, 0, false, B_layoutRem, B_regs(h),
                             strategy, state, h % remaskPeriod);
                 }}});

    // A/B repacking.
    auto reqRepackA = every(ka_loadMain) | variants(A_copies);
    auto reqRepackARem = every(ka_loadRem) | variants(A_copies);
    bool convertA = (Ta != Ta_load) && (Ta.size() == Ta_load.size());

    if (state.repackA || repackARem || convertA)
        ls.schedule({{reqRepackA,
                             [&](Iteration h) {
                                 if (state.repackA)
                                     copyRegisters(Ta_load, Ta, state.A_layout,
                                             state.Ar_layout, A_regs(h),
                                             state.Ar_regs, 0, 0, false,
                                             strategy, state);
                                 else if (convertA)
                                     convert(A_regs(h), Ta_load, Ta, problem,
                                             strategy, state);
                             }},
                {reqRepackARem, [&](Iteration h) {
                     if (repackARem)
                         copyRegisters(Ta_load, Ta, A_layoutRem,
                                 state.Ar_layout, A_regs(h), state.Ar_regs, 0,
                                 h % ka_repackRem, false, strategy, state);
                     else if (convertA)
                         convert(A_regs(h), Ta_load, Ta, problem, strategy,
                                 state);
                 }}});

    auto reqRepackB = every(kb_loadMain) | variants(B_copies);
    auto reqRepackBRem = every(kb_loadRem) | variants(B_copies);
    bool convertB = (Tb != Tb_load) && (Tb.size() == Tb_load.size());

    if (state.repackB || repackBRem || convertB)
        ls.schedule({{reqRepackB,
                             [&](Iteration h) {
                                 if (state.repackB)
                                     copyRegisters(Tb_load, Tb, state.B_layout,
                                             state.Br_layout, B_regs(h),
                                             state.Br_regs, 0, 0, false,
                                             strategy, state);
                                 else if (convertB)
                                     convert(B_regs(h), Tb_load, Tb, problem,
                                             strategy, state);
                             }},
                {reqRepackBRem, [&](Iteration h) {
                     if (repackBRem)
                         copyRegisters(Tb_load, Tb, B_layoutRem,
                                 state.Br_layout, B_regs(h), state.Br_regs,
                                 h % kb_repackRem, 0, false, strategy, state);
                     else if (convertB)
                         convert(B_regs(h), Tb_load, Tb, problem, strategy,
                                 state);
                 }}});

    // Outer product(s).
    // If outer products batched across k (dp4a/dpas/k-chaining), trigger every opCount loops.
    auto reqOP = every(minOPCount) | lookahead(-(minOPCount - 1));

    ls.schedule(reqOP, [&](Iteration h) {
        auto oc = opCount(h);
        auto hNext = h + minOPCount;
        if (hNext % oc != 0) return;

        int ka = ka_load(h), kb = kb_load(h);
        int ha = h % ka;
        int hb = h % kb;
        if (problem.backward()) {
            ha = ka - 1 - ha;
            hb = kb - 1 - hb;
        }

        auto layoutA = &A_layout(h);
        auto layoutB = &B_layout(h);
        auto regsA = &A_regs(h);
        auto regsB = &B_regs(h);

        if (repackA(h)) {
            layoutA = &state.Ar_layout;
            regsA = &state.Ar_regs;
        }
        if (repackB(h)) {
            layoutB = &state.Br_layout;
            regsB = &state.Br_regs;
        }

        outerProduct(h, ha, hb, oc, *layoutA, *layoutB, *regsA, *regsB, problem,
                strategy, state);

        if (calcABSums) {
            if (!slmA)
                accumulateSum(false, Ta, *regsA, *layoutA, Tc, state.As_regs,
                        state.As_layout, strategy, state, ha, ha + oc);
            if (!slmB)
                accumulateSum(true, Tb, *regsB, *layoutB, Tc, state.Bs_regs,
                        state.Bs_layout, strategy, state, hb, hb + oc);
        }
    });

    // SLM data repacking and remasking.
    auto reqSLMRepack = every(unrollKSLM) | variants(slmCopies)
            | lookahead(lookaheadSLMStore + lookaheadSLMReload
                    + strategy.slmRepackAhead)
            | duration(durationSLMMainLoad);
    auto reqSLMRepackABRem = every(unrollKSLM) | variants(slmCopies)
            | lookahead(lookaheadSLMStore + lookaheadSLMReloadRem
                    + strategy.slmRepackAhead);

    auto slmConvertA = [&](Iteration h) {
        return slmA && aioShare(h) && (Ta != Ta_ext)
                && (Ta.size() == Ta_ext.size());
    };
    auto slmConvertB = [&](Iteration h) {
        return slmB && bioShare(h) && (Tb != Tb_ext)
                && (Tb.size() == Tb_ext.size());
    };

    auto doSLMRepack = [&](Iteration h) {
        if (slmA && !aioShare(h) && !(slmRemActive(h) && Ai_remIncrCopy))
            copyRegisters(Ta_ext, Ta, Ai_layout(h), state.Ao_layout, Ai_regs(h),
                    Ao_regs(h), 0, 0, false, strategy, state);
        else if (slmConvertA(h))
            convert(Ai_regs(h), Ta_ext, Ta, problem, strategy, state);

        if (slmB && !bioShare(h) && !(slmRemActive(h) && Bi_remIncrCopy))
            copyRegisters(Tb_ext, Tb, Bi_layout(h), state.Bo_layout, Bi_regs(h),
                    Bo_regs(h), 0, 0, false, strategy, state);
        else if (slmConvertB(h))
            convert(Bi_regs(h), Tb_ext, Tb, problem, strategy, state);

        if (slmRemActive(h) && (slmRemaskA || slmRemaskB)) {
            releaseMaskAssignments(
                    kMasks, state); // Not in use -- can temporarily free these.
            gemmSLMRemask(slmRemaskA, slmRemaskB, effAo_regs(h), effBo_regs(h),
                    -h.counterOffset(), problem, strategy, state);
            reclaimMaskAssignments(kMasks, state);
        }
    };

    auto checkSLMRepack = [&](Iteration h) {
        return (slmA && !aioShare(h) && !(slmRemActive(h) && Ai_remIncrCopy))
                || (slmB && !bioShare(h)
                        && !(slmRemActive(h) && Bi_remIncrCopy))
                || (slmRemActive(h) && (slmRemaskA || slmRemaskB))
                || slmConvertA(h) || slmConvertB(h);
    };

    if (slmA || slmB) {
        ls.schedule_if({{reqSLMRepack, doSLMRepack, checkSLMRepack},
                {reqSLMRepackABRem, doSLMRepack, checkSLMRepack}});
    }

    // SLM stores and synchronization.
    auto reqSLMAfterStore = every(unrollKSLM) | variants(slmCopies)
            | lookahead(lookaheadSLMStore + lookaheadSLMReload - unrollKSLM)
            | duration(durationSLMMainLoad);
    auto reqSLMAfterStore2 = every(unrollKSLM) | variants(slmCopies)
            | lookahead(lookaheadSLMStore + lookaheadSLMReload - 2 * unrollKSLM)
            | duration(durationSLMMainLoad);
    auto reqSLMAfterStoreABRem = every(unrollKSLM) | variants(slmCopies)
            | lookahead(lookaheadSLMStore + lookaheadSLMReloadRem - unrollKSLM);
    auto reqSLMAfterStoreABRem2 = every(unrollKSLM) | variants(slmCopies)
            | lookahead(
                    lookaheadSLMStore + lookaheadSLMReloadRem - 2 * unrollKSLM);

    auto slm1x2xFencedBarrier = [&]() {
        // For DG2+, before 1x/2x buffered stores, we must ensure prior SLM reads are complete.
        // Use a fence for >2x global buffering.
        // For 2x global buffering, use SWSB since loaded data will be used shortly.
        // For 1x global buffering, loaded data has already been consumed.
        if (hw < HW::XeHPG && !strategy.strictFence)
            kLoopBarrier(false);
        else if (A_copies > 2 || B_copies > 2)
            kLoopBarrier(true);
        else {
            if (slmA && A_copies > 1) wrdepRanges(state.A_regs);
            if (slmB && B_copies > 1) wrdepRanges(state.B_regs);
            kLoopBarrier(false);
        }
    };

    auto doSLMAfterStore2 = [&](Iteration h) {
        switch (slmBuffers) {
            case 1:
            case 2:
            case 3: break;
            case 4: kLoopBarrier(false, KBarrierType::Wait); break;
            default: stub();
        }
    };

    auto doSLMAfterStore = [&](Iteration h) {
        switch (slmBuffers) {
            case 1: break;
            case 2: slm1x2xFencedBarrier(); break;
            case 3: kLoopBarrier(false, KBarrierType::Wait); break;
            case 4:
                // TEMP: move me earlier.
                slmFenceIssue();
                //
                slmFenceWait();
                kLoopBarrier(false, KBarrierType::Signal);
                break;
        }
    };

    auto doSLMStore = [&](Iteration h) {
        if (!slmA && !slmB) return;

        switch (slmBuffers) {
            case 1: slm1x2xFencedBarrier(); break;
            case 2:
            case 3:
            case 4: break;
            default: stub();
        }

        if (slmA)
            storeMatrix(effAo_regs(h), state.Ao_layout, state.Ao,
                    state.Ao_strategy, state.Ao_addrs, strategy, state);
        if (slmB)
            storeMatrix(effBo_regs(h), state.Bo_layout, state.Bo,
                    state.Bo_strategy, state.Bo_addrs, strategy, state);

        if (calcABSums) {
            if (slmA)
                accumulateSum(false, Ta, effAo_regs(h), state.Ao_layout, Tc,
                        state.As_regs, state.As_layout, strategy, state);
            if (slmB)
                accumulateSum(true, Tb, effBo_regs(h), state.Bo_layout, Tc,
                        state.Bs_regs, state.Bs_layout, strategy, state);
        }

        switch (slmBuffers) {
            case 1: kLoopBarrier(true); break;
            case 2:
                slmFenceIssue();
                slmFenceWait();
                break;
            case 3: kLoopBarrier(true, KBarrierType::Signal); break;
            case 4: break;
            default: stub();
        }
    };

    if (slmBuffers > 0) {
        if (slmBuffers >= 4)
            ls.schedule({{reqSLMAfterStore2, doSLMAfterStore2},
                    {reqSLMAfterStoreABRem2, doSLMAfterStore2}});

        if (slmBuffers >= 2)
            ls.schedule({{reqSLMAfterStore, doSLMAfterStore},
                    {reqSLMAfterStoreABRem, doSLMAfterStore}});

        ls.schedule(
                {{reqSLMStore, doSLMStore}, {reqSLMStoreABRem, doSLMStore}});
    }

    // Save pre-loop state.
    auto statePreLoop = state;

    using CT = LoopSequencer::CallbackType;

    Label lTop, lBottom;
    std::vector<Label> labels;

    ls.analyze();

    if (ls.getUnroll() != unrollK)
        stub(); // Auto-calculated unroll should match unrollK from strategy.

    // Prepare to save off loops for periodic barriers, if needed.
    Subregister outerK;
    if (strategy.barrierFreq > 0) outerK = state.ra.alloc_sub<uint32_t>();

    // Prepare to peel loops for C prefetch, if needed.
    int prefetchCPeelLoops = -1;
    Subregister pfCPeelK;
    if (strategy.prefetchC > 0) {
        prefetchCPeelLoops = div_up(
                std::max(0, strategy.prefetchC - ls.getCooldown()), unrollK);
        if (prefetchCPeelLoops > 0) pfCPeelK = state.ra.alloc_sub<uint32_t>();
    }

    auto resetForNewLoop = [&]() {
        resetKSLM();
        lastThresh = 0;
        haveA_lastRSWA = false;
        state.ra.safeRelease(barrierHeader);
        setupTeardownRemask(
                Tremask, 0, false, remaskPeriod, state.K, strategy, state);
    };

    // Main events in lifetime of loop.
    ls.setCallback(CT::OffsetCounter,
            [&](int offset, int) { add(1, state.K, state.K, offset); });
    ls.setCallback(CT::LoopStart, [&](int unroll, int) {
        cmp(1 | le | state.flagAP, state.K, 0);
        if (prefetchCPeelLoops > 0) {
            min_(1, pfCPeelK, state.K, prefetchCPeelLoops * unrollK);
            add(1, state.K, state.K, -pfCPeelK);
        }
        if (strategy.barrierFreq > 0) {
            add(1 | sat, outerK, state.K, -strategy.barrierFreq);
            min_(1, state.K, state.K, strategy.barrierFreq);
            if (strategy.splitBarrier)
                kLoopBarrier(false, KBarrierType::Signal);
        }
        if (hw >= HW::Gen12LP) sync.nop(SWSB(Pipe::A, 1));
        jmpi(1 | state.flagAP, lBottom);
        mark(lTop);
        state.wipeActiveVFlags();
    });
    ls.setCallback(CT::LoopEnd, [&](int, int) {
        jmpi(1 | state.flagAP, lTop);
        if (strategy.barrierFreq > 0) {
            add(1, state.K, state.K, outerK);
            add(1 | sat, outerK, outerK, int16_t(-strategy.barrierFreq));
            add(1 | gt | state.flagAP, state.K, state.K, -outerK);
            if (strategy.splitBarrier) {
                kLoopBarrier(false, KBarrierType::Wait);
                kLoopBarrier(false, KBarrierType::Signal);
            } else
                kLoopBarrier(false);
            jmpi(1 | state.flagAP, lTop);
        }
        if (prefetchCPeelLoops > 0) {
            add(1 | gt | state.flagAP, state.K, state.K, pfCPeelK);
            mov(1, pfCPeelK, 0);
            gemmPrefetchC(problem, strategy, state);
            jmpi(1 | state.flagAP, lTop);
        }
        mark(lBottom);
        state.wipeActiveVFlags();
    });
    ls.setCallback(CT::JumpIfLT, [&](int thresh, int label) {
        if (size_t(label) >= labels.size()) labels.resize(label + 1);
        if (thresh != lastThresh) cmp(1 | lt | state.flagAP, state.K, thresh);
        jmpi(1 | state.flagAP, labels[label]);
        lastThresh = thresh;
    });
    ls.setCallback(CT::JumpTarget, [&](int label, int) {
        mark(labels[label]);
        state.wipeActiveVFlags();
    });
    ls.setCallback(CT::Jump, [&](int label, int) {
        if (size_t(label) >= labels.size()) labels.resize(label + 1);
        jmpi(1, labels[label]);
    });
    ls.setCallback(CT::NotifyPhase, [&](int phase, int) {
        curPhase = phase;
        switch (phase) {
            case LoopSequencer::PhaseWarmup:
                status << "k loop warmup" << status_stream::endl;
                break;
            case LoopSequencer::PhaseMainLoop:
                status << "Main k loop" << status_stream::endl;
                break;
            case LoopSequencer::PhaseMainPathEnd:
                if (strategy.barrierFreq > 0 && strategy.splitBarrier)
                    kLoopBarrier(false, KBarrierType::Wait);
                break;
            case LoopSequencer::PhaseCooldown:
                if (prefetchCPeelLoops == 0)
                    gemmPrefetchC(problem, strategy, state);
                if (lateKLoopCheck) state.raVFlag.lock(state.flagAP);
                haveA_lastRSWA = false;
                status << "k loop cooldown" << status_stream::endl;
                break;
            case LoopSequencer::PhaseShortLoop:
                if (strategy.prefetchC > 0)
                    gemmPrefetchC(problem, strategy, state);
                status << "Short k loop" << status_stream::endl;
                remActiveA = remActiveB = remActiveSLM = false;
                resetForNewLoop();
                state = statePreLoop;
                break;
            case LoopSequencer::PhaseRemainder:
                status << "k loop remainder" << status_stream::endl;
                break;
            default: break;
        }
    });

    // Early C prefetch.
    if (strategy.prefetchC < 0) gemmPrefetchC(problem, strategy, state);

    if (lateKLoopCheck) state.raVFlag.unlock(state.flagAP);

    // Avoid unnecessary SWSB dependencies in main loop.
    syncall();

    // Generate k loop.
    ls.materialize();

    // Release barrier header from short k loop.
    state.ra.safeRelease(barrierHeader);

    // Additional barriers to match other threads' barrier count, if other threads might have different k.
    if (matchBarriers) {
        status << "Match barrier counts between threads" << status_stream::endl;
        Subregister myBarriers, k0Barriers;
        Label lSkipExtraBarriers, lExtraBarrierLoop;
        int maxExtraBarriers = maxExtraKLoopRemBarriers(strategy);

        if (strategy.barrierFreq > 0 && prefetchCPeelLoops > 0) stub();

        gemmCalcKLoopBarrierCount(k0Barriers, state.inputs.k0, ls.getCooldown(),
                problem, strategy, state);
        gemmCalcKLoopBarrierCount(myBarriers, state.k, ls.getCooldown(),
                problem, strategy, state);
        if (maxExtraBarriers > 0)
            add(1, k0Barriers, k0Barriers, maxExtraBarriers);
        add(1 | sat | le | state.flagAP, myBarriers.ud(), k0Barriers,
                -myBarriers);
        (void)getBarrierHeader();
        jmpi(1 | state.flagAP, lSkipExtraBarriers);

        mark(lExtraBarrierLoop);
        {
            add(1 | gt | state.flagAP, myBarriers, myBarriers, -1);
            kLoopBarrier(false);
            jmpi(1 | state.flagAP, lExtraBarrierLoop);
        }
        mark(lSkipExtraBarriers);

        state.ra.safeRelease(myBarriers);
        state.ra.safeRelease(k0Barriers);
        if (!strategy.persistent) state.ra.safeRelease(state.inputs.k0);
    }

    // Free resources that are no longer needed.
    if (state.K != state.k) state.ra.safeRelease(state.K);
    state.ra.safeRelease(outerK);
    state.ra.safeRelease(pfCPeelK);
    state.ra.safeRelease(barrierHeaderM);
    state.ra.safeRelease(barrierHeaderN);
    state.ra.safeRelease(barrierHeader);
    state.raVFlag.safeRelease(barrierM);
    state.raVFlag.safeRelease(barrierN);
    state.tokenAllocator.release(tokenBarrierFence[0]);
    state.tokenAllocator.release(tokenBarrierFence[1]);
    safeReleaseMaskAssignments(kMasks, state);
    setupTeardownRemask(
            Tremask, 0, false, remaskPeriod, state.K, strategy, state);

    return true;
}

template <HW hw>
bool gemm_kernel_generator_t<hw>::gemmKLoopDispatch(bool lateKLoopCheck,
        GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state) {
    return gemmKLoop(lateKLoopCheck, problem, strategy, state);
}

// Decide whether C layout needs m/n remainder handling.
static inline void getCRemainders(const GEMMProblem &problem,
        const GEMMStrategy &strategy, bool &remM_C, bool &remN_C) {
    bool remainderM
            = (strategy.remHandling[LoopM] != RemainderHandling::Ignore);
    bool remainderN
            = (strategy.remHandling[LoopN] != RemainderHandling::Ignore);

    int C_mgran, C_ngran;
    getGranularities(problem.C, C_mgran, C_ngran);

    remM_C = remainderM && !strategy.C.padded && !strategy.altCRemainder
            && (C_mgran < strategy.unroll[LoopM]);
    remN_C = remainderN && !strategy.C.padded && !strategy.altCRemainder
            && (C_ngran < strategy.unroll[LoopN]);
}

// Perform the body of the GEMM computation, updating a block of C.
template <HW hw>
bool gemm_kernel_generator_t<hw>::gemmAccumulateC(
        GEMMProblem &problem_, GEMMStrategy &strategy, GEMMState &state) {
    if (strategy.fixedSystolic) {
        return strategy.splitCopy
                ? sysgemm2AccumulateC(problem_, strategy, state)
                : sysgemmAccumulateC(problem_, strategy, state);
    }

    auto problem = problem_;
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc;
    auto Ta_ext = problem.Ta_ext, Tb_ext = problem.Tb_ext,
         Tc_ext = problem.Tc_ext;
    auto &Ta_load = state.Ta_load, &Tb_load = state.Tb_load;

    bool lateKLoopCheck = false;
    bool cLoadAhead = strategy.cLoadAhead;
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];

    // Decide what remainder handling needs to be done.
    bool remainderM = strategy.remHandling[LoopM] != RemainderHandling::Ignore;
    bool remainderN = strategy.remHandling[LoopN] != RemainderHandling::Ignore;
    bool remainderK = strategy.remHandling[LoopK] != RemainderHandling::Ignore;
    bool remM_A = remainderM && !strategy.A.padded;
    bool remK_A = false;
    bool remK_B = false;
    bool remN_B = remainderN && !strategy.B.padded;
    bool remM_C, remN_C;
    getCRemainders(problem, strategy, remM_C, remN_C);
    bool remM_Ce = remM_C;
    bool remN_Ce = remN_C;

    if (state.copyC) remM_C = remN_C = false;

    auto globalA = problem.A;
    auto globalB = problem.B;

    // 2D addressing parameters.
    Address2DParams A_params, B_params;
    A_params.rows = state.inputs.m;
    A_params.cols = state.fullK;
    A_params.offR = state.i0;
    A_params.offC = state.h0;
    A_params.remR = state.remainders[LoopM];
    B_params.rows = state.fullK;
    B_params.cols = state.inputs.n;
    B_params.offR = state.h0;
    B_params.offC = state.j0;
    B_params.remC = state.remainders[LoopN];
    auto Ai_params = A_params, Bi_params = B_params;
    auto Ap_params = A_params, Bp_params = B_params;

    // Decide which dimensions to split for WG-cooperative operations (SLM copy, cooperative PF).
    state.effCoopA = effCoopSplitA(problem, strategy);
    state.effCoopB = effCoopSplitB(problem, strategy);

    if (strategy.slmA && (state.effCoopA != CoopSplit::K) && remM_A
            && !isBlock2D(strategy.A.accessType)) {
        strategy.A.accessType = isColMajor(problem.A.layout)
                ? AccessType::Block
                : AccessType::Scattered;
        state.effCoopA = CoopSplit::K;
    }

    if (strategy.slmB && (state.effCoopB != CoopSplit::K) && remN_B
            && !isBlock2D(strategy.B.accessType)) {
        strategy.B.accessType = !isColMajor(problem.B.layout)
                ? AccessType::Block
                : AccessType::Scattered;
        state.effCoopB = CoopSplit::K;
    }

    // Prepare layouts for prefetch.
    bool remM_Cp = remM_C && strategy.C.base.isStateless();
    bool remN_Cp = remN_C && strategy.C.base.isStateless();

    state.ma_prefetch = state.ka_prefetch = state.kb_prefetch
            = state.nb_prefetch = 0;
    if (strategy.prefetchA)
        coopSplit(true, state.ma_prefetch, state.ka_prefetch, unrollM,
                strategy.ka_prefetch, state.effCoopA, strategy.wg[LoopN],
                problem.A);
    if (strategy.prefetchB)
        coopSplit(false, state.kb_prefetch, state.nb_prefetch,
                strategy.kb_prefetch, unrollN, state.effCoopB,
                strategy.wg[LoopM], problem.B);

    if (strategy.prefetchA
            && !getRegLayout(Ta_ext, state.Ap_layout, state.ma_prefetch,
                    state.ka_prefetch, remM_A, remK_A, false, true, 0, 0,
                    problem.A, strategy.A_prefetch))
        return false;
    if (strategy.prefetchB
            && !getRegLayout(Tb_ext, state.Bp_layout, state.kb_prefetch,
                    state.nb_prefetch, remK_B, remN_B, false, true, 0, 0,
                    problem.B, strategy.B_prefetch))
        return false;
    if (strategy.prefetchC
            && !getRegLayout(Tc_ext, state.Cp_layout, unrollM, unrollN, remM_Cp,
                    remN_Cp, false, true, 0, 0, problem.C, strategy.C_prefetch))
        return false;

    if (hasMasking(state.Cp_layout) || hasFragmenting(state.Cp_layout)) stub();

    // Prepare addresses for prefetch.
    if (strategy.cooperativePF && strategy.prefetchA) {
        Subregister offAp;
        gemmCalcWorkshareAOffset(offAp, Ap_params.offR, Ap_params.offC,
                problem.A, strategy.A_prefetch, state.ma_prefetch,
                state.ka_prefetch, problem, strategy, state);
        if (!strategy.A_prefetch.address2D) {
            auto inEffAp = state.effAp;
            if (state.effA == state.effAp)
                state.effAp = state.ra.alloc_sub(state.effA.getType());
            eadd(1, state.effAp, inEffAp, offAp, strategy, state);
        }
        state.ra.safeRelease(offAp);
    }
    if (strategy.cooperativePF && strategy.prefetchB) {
        Subregister offBp;
        gemmCalcWorkshareBOffset(offBp, Bp_params.offR, Bp_params.offC,
                problem.B, strategy.B_prefetch, state.kb_prefetch,
                state.nb_prefetch, problem, strategy, state);
        if (!strategy.B_prefetch.address2D) {
            auto inEffBp = state.effBp;
            if (state.effB == state.effBp)
                state.effBp = state.ra.alloc_sub(state.effB.getType());
            eadd(1, state.effBp, inEffBp, offBp, strategy, state);
        }
        state.ra.safeRelease(offBp);
    }

    // Prepare layouts and starting addresses for SLM copies and adjust problem.
    auto saveAStrategy = strategy.A, saveBStrategy = strategy.B;
    if (strategy.slmBuffers > 0) {
        int A_slmCP, B_slmCP;
        int A_tileR, A_tileC, B_tileR, B_tileC;
        std::tie(A_slmCP, B_slmCP) = targetSLMCrosspack(hw, problem, strategy);
        std::tie(A_tileR, A_tileC, B_tileR, B_tileC)
                = targetKernelTiling(hw, problem, strategy);
        auto opCount = outerProductCount(hw, problem, strategy);

        if (strategy.slmA) {
            coopSplit(true, state.ma_slm, state.ka_slm, unrollM,
                    strategy.unrollKSLM, state.effCoopA, strategy.wg[LoopN],
                    problem.A);

            if (state.ma_slm < unrollM) {
                remM_A = false;
                remK_A = remainderK && strategy.slmEarlyKMask;
            }
            if (strategy.slmATrans) {
                A_slmCP = state.ka_slm;
                if (strategy.ka_load % A_slmCP)
                    throw std::runtime_error(
                            "ka_load must be a multiple of ka_slm");
            }
            bool splitCP = (state.ka_slm < A_slmCP);
            if (splitCP && (strategy.unrollKSLM != A_slmCP))
                throw std::runtime_error(
                        "ka_slm must be a multiple of crosspack, or unrollKSLM "
                        "= crosspack.");

            // Layout in from memory...
            state.Ai = problem.A;
            state.Ai_strategy = strategy.A;

            // ... layout out to SLM.
            state.Ao.layout = MatrixLayout::Pc;
            state.Ao.packSize = unrollM;
            state.Ao.crosspack = A_slmCP;
            state.Ao.setAlignment(state.Ao.packSize * Ta);
            state.Ao.tileR = A_tileR;
            state.Ao.tileC = (A_tileC || !A_tileR)
                    ? A_tileC
                    : std::max(opCount, strategy.ka_load);

            bool colMajorIn
                    = isRegisterColMajor(Ta_ext, state.Ai, state.Ai_strategy);
            bool colMajorSLM = !isLargeCrosspack(Ta, A_slmCP);
            state.Ao_strategy.base = SLM;
            state.Ao_strategy.accessType = (colMajorIn == colMajorSLM)
                    ? AccessType::Block
                    : AccessType::Scattered;
            state.Ao_strategy.smode = ScatterSIMD::Default;

            if (state.Ai.layout == MatrixLayout::N
                    && state.Ai_strategy.accessType == AccessType::Block2DVNNI
                    && isLargeCrosspack(Ta, A_slmCP)) {
                state.Ao_strategy.accessType = AccessType::ChannelScattered;
                state.Ao_strategy.smode = ScatterSIMD::Narrow;
            }
            state.Ao_strategy.padded = true;
            state.Ao_strategy.atomic = false;
            state.Ao_strategy.address2D = false;
            state.Ao_strategy.newDP = (hw >= HW::XeHPG);
            state.Ao_strategy.cachingW = CacheSettingsLSC::Default;

            // Layout in from memory...
            if (!getRegLayout(Ta_ext, state.Ai_layout, state.ma_slm,
                        state.ka_slm, remM_A, remK_A, false, true, 0, 0,
                        state.Ai, state.Ai_strategy))
                return false;

            // ... layout out to SLM...
            remM_A = remK_A = false;
            if (!getRegLayout(Ta, state.Ao_layout, state.ma_slm, state.ka_slm,
                        remM_A, remK_A, true, true, 0, 0, state.Ao,
                        state.Ao_strategy))
                return false;

            // ... and layout back from SLM.
            problem.A = state.Ao;
            strategy.A.base = SLM;
            strategy.A.accessType = AccessType::Block;
            strategy.A.address2D = false;
            strategy.A.newDP = (hw >= HW::XeHPG);
            strategy.A.cachingR = CacheSettingsLSC::Default;
            Ta_load = Ta;
            state.aioShare = Ta.size() == Ta_ext.size()
                    && matchLayoutsBidirectional(
                            Ta, state.Ai_layout, state.Ao_layout);

            // Offset A addresses in and out.
            state.effAi = state.effA;
            state.effA = state.ra.alloc_sub<uint32_t>(
                    getHint(HintType::LongTerm, strategy));
            state.effAo = state.ra.alloc_sub<uint32_t>(
                    getHint(HintType::LongTerm, strategy));

            auto temp = state.ra.alloc_sub<uint32_t>(
                    getHint(HintType::TempComp0, strategy));
            Subregister temp2;

            uint32_t noff, noffTile, tileSplit = 1;

            switch (state.effCoopA) {
                case CoopSplit::Linear:
                    noff = state.ma_slm * state.ka_slm;
                    break;
                case CoopSplit::MN:
                    noff = untile(state.Ao, state.ma_slm, 0, state.Ao.packSize,
                            strategy.unrollKSLM);
                    if (state.ma_slm < state.Ao.tileR
                            && state.Ao.tileR < state.Ao.packSize) {
                        // m division splits tiles -- starting offsets no longer a linear sequence.
                        if (state.Ao.tileR % state.ma_slm) stub();
                        tileSplit = state.Ao.tileR / state.ma_slm;
                        noffTile = untile(state.Ao, state.Ao.tileR, 0,
                                state.Ao.packSize, strategy.unrollKSLM);
                    }
                    break;
                case CoopSplit::K:
                    noff = untile(state.Ao, 0, state.ka_slm, state.Ao.packSize,
                            strategy.unrollKSLM);
                    if (state.ka_slm < state.Ao.tileC
                            && state.Ao.tileC < strategy.unrollKSLM) {
                        // k division splits tiles -- starting offsets no longer a linear sequence.
                        if (state.Ao.tileC % state.ka_slm) stub();
                        tileSplit = state.Ao.tileC / state.ka_slm;
                        noffTile = untile(state.Ao, 0, state.Ao.tileC,
                                state.Ao.packSize, strategy.unrollKSLM);
                    }
                    break;
                default: stub();
            }

            int32_t A_slmStride
                    = strategy.slmABufBlockSize(problem) * strategy.slmBuffers;

            if (tileSplit > 1) {
                if (!is_zero_or_pow2(tileSplit)) stub();
                shr(1, temp, state.lidN, log2(tileSplit));
            }
            gemmCalcWorkshareAOffset(temp2, Ai_params.offR, Ai_params.offC,
                    state.Ai, state.Ai_strategy, state.ma_slm, state.ka_slm,
                    problem, strategy, state);
            if (tileSplit > 1) {
                mulConstant(1, temp, temp, (noffTile - noff * tileSplit) * Ta);
                emad(1, temp, temp, state.lidN, noff * Ta, strategy, state);
            } else
                mulConstant(1, temp, state.lidN, noff * Ta);
            mulConstant(1, state.effA, state.lidM, A_slmStride);
            if (strategy.wg[LoopK] > 1)
                emad(1, state.effA, state.effA, state.lidK,
                        A_slmStride * strategy.wg[LoopM], strategy, state);
            if (state.Ai_strategy.address2D) {
                if (Ai_params.offR != A_params.offR && A_params.offR.isValid())
                    add(1, Ai_params.offR, Ai_params.offR, A_params.offR);
                if (Ai_params.offC != A_params.offC && A_params.offC.isValid())
                    add(1, Ai_params.offC, Ai_params.offC, A_params.offC);
            } else
                eadd(1, state.effAi, state.effAi, temp2, strategy, state);
            add(1, state.effAo, state.effA, temp);
            if (problem.backward())
                add(1, state.effA, state.effA,
                        (strategy.unrollKSLM - strategy.ka_load) * unrollM
                                * Ta);

            state.ra.safeRelease(temp2);
            state.ra.safeRelease(temp);
        }
        if (strategy.slmB) {
            coopSplit(false, state.kb_slm, state.nb_slm, strategy.unrollKSLM,
                    unrollN, state.effCoopB, strategy.wg[LoopM], problem.B);

            if (state.nb_slm < unrollN) {
                remN_B = false;
                remK_B = remainderK && strategy.slmEarlyKMask;
            }
            if (strategy.slmBTrans) {
                B_slmCP = state.kb_slm;
                if (strategy.kb_load % B_slmCP)
                    throw std::runtime_error(
                            "kb_load must be a multiple of kb_slm");
            }
            bool splitCP = (state.kb_slm < B_slmCP);
            if (splitCP && (strategy.unrollKSLM != B_slmCP))
                throw std::runtime_error(
                        "kb_slm must be a multiple of crosspack, or unrollKSLM "
                        "= crosspack.");

            // Layout in from memory...
            state.Bi = problem.B;
            state.Bi_strategy = strategy.B;

            // ... layout out to SLM.
            state.Bo.layout = MatrixLayout::Pr;
            state.Bo.packSize = unrollN;
            state.Bo.crosspack = B_slmCP;
            state.Bo.setAlignment(state.Bo.packSize * Tb);
            state.Bo.tileR = (B_tileR || !B_tileC)
                    ? B_tileR
                    : std::max(opCount, strategy.kb_load);
            state.Bo.tileC = B_tileC;

            bool colMajorIn
                    = isRegisterColMajor(Tb_ext, state.Bi, state.Bi_strategy);
            bool colMajorSLM = isLargeCrosspack(Tb, B_slmCP);
            state.Bo_strategy.base = SLM;
            state.Bo_strategy.accessType = (colMajorIn == colMajorSLM)
                    ? AccessType::Block
                    : AccessType::Scattered;
            state.Bo_strategy.smode = ScatterSIMD::Default;

            if (state.Bi.layout == MatrixLayout::T
                    && state.Bi_strategy.accessType == AccessType::Block2DVNNI
                    && isLargeCrosspack(Tb, B_slmCP)) {
                state.Bo_strategy.accessType = AccessType::ChannelScattered;
                state.Bo_strategy.smode = ScatterSIMD::Narrow;
            }
            state.Bo_strategy.padded = true;
            state.Bo_strategy.atomic = false;
            state.Bo_strategy.address2D = false;
            state.Bo_strategy.newDP = (hw >= HW::XeHPG);
            state.Bo_strategy.cachingW = CacheSettingsLSC::Default;

            // Layout in from memory...
            if (!getRegLayout(Tb_ext, state.Bi_layout, state.kb_slm,
                        state.nb_slm, remK_B, remN_B, false, true, 0, 0,
                        state.Bi, state.Bi_strategy))
                return false;

            // ... layout out to SLM...
            remK_B = remN_B = false;
            if (!getRegLayout(Tb, state.Bo_layout, state.kb_slm, state.nb_slm,
                        remK_B, remN_B, true, true, 0, 0, state.Bo,
                        state.Bo_strategy))
                return false;

            // ... and layout back from SLM.
            problem.B = state.Bo;
            strategy.B.base = SLM;
            strategy.B.accessType = AccessType::Block;
            strategy.B.address2D = false;
            strategy.B.newDP = (hw >= HW::XeHPG);
            strategy.B.cachingR = CacheSettingsLSC::Default;
            Tb_load = Tb;
            state.bioShare = Tb.size() == Tb_ext.size()
                    && matchLayoutsBidirectional(
                            Tb, state.Bi_layout, state.Bo_layout);

            // Offset B addresses in and out.
            state.effBi = state.effB;
            state.effB = state.ra.alloc_sub<uint32_t>(
                    getHint(HintType::LongTerm, strategy));
            state.effBo = state.ra.alloc_sub<uint32_t>(
                    getHint(HintType::LongTerm, strategy));

            auto temp = state.ra.alloc_sub<uint32_t>(
                    getHint(HintType::TempComp0, strategy));
            Subregister temp2;

            uint32_t moff, moffTile, tileSplit = 1;

            switch (state.effCoopB) {
                case CoopSplit::Linear:
                    moff = state.kb_slm * state.nb_slm;
                    break;
                case CoopSplit::MN:
                    moff = untile(state.Bo, 0, state.nb_slm,
                            strategy.unrollKSLM, state.Bo.packSize);
                    if (state.nb_slm < state.Bo.tileC
                            && state.Bo.tileC < state.Bo.packSize) {
                        if (state.Bo.tileC % state.nb_slm) stub();
                        tileSplit = state.Bo.tileC / state.nb_slm;
                        moffTile = untile(state.Bo, 0, state.Bo.tileC,
                                strategy.unrollKSLM, state.Bo.packSize);
                    }
                    break;
                case CoopSplit::K:
                    moff = untile(state.Bo, state.kb_slm, 0,
                            strategy.unrollKSLM, state.Bo.packSize);
                    if (state.kb_slm < state.Bo.tileR) {
                        if (state.Bo.tileR % state.kb_slm) stub();
                        tileSplit = state.Bo.tileR / state.kb_slm;
                        moffTile = untile(state.Bo, state.Bo.tileR, 0,
                                strategy.unrollKSLM, state.Bo.packSize);
                    }
                    break;
                default: stub();
            }

            int32_t B_slmStride
                    = strategy.slmBBufBlockSize(problem) * strategy.slmBuffers;

            if (tileSplit > 1) {
                if (!is_zero_or_pow2(tileSplit)) stub();
                shr(1, temp, state.lidM, log2(tileSplit));
            }
            gemmCalcWorkshareBOffset(temp2, Bi_params.offR, Bi_params.offC,
                    state.Bi, state.Bi_strategy, state.kb_slm, state.nb_slm,
                    problem, strategy, state);
            if (tileSplit > 1) {
                mulConstant(1, temp, temp, (moffTile - moff * tileSplit) * Tb);
                emad(1, temp, temp, state.lidM, moff * Tb, strategy, state);
            } else
                mulConstant(1, temp, state.lidM, moff * Tb);
            mulConstant(1, state.effB, state.lidN, B_slmStride);
            if (strategy.wg[LoopK] > 1)
                emad(1, state.effB, state.effB, state.lidK,
                        B_slmStride * strategy.wg[LoopN], strategy, state);
            if (state.Bi_strategy.address2D) {
                if (Bi_params.offR != B_params.offR && B_params.offR.isValid())
                    add(1, Bi_params.offR, Bi_params.offR, B_params.offR);
                if (Bi_params.offC != B_params.offC && B_params.offC.isValid())
                    add(1, Bi_params.offC, Bi_params.offC, B_params.offC);
            } else
                eadd(1, state.effBi, state.effBi, temp2, strategy, state);
            if (strategy.slmABufSize(problem) > 0)
                add(1, state.effB, state.effB, strategy.slmABufSize(problem));
            add(1, state.effBo, state.effB, temp);
            if (problem.backward())
                add(1, state.effB, state.effB,
                        (strategy.unrollKSLM - strategy.kb_load) * unrollN
                                * Tb);

            state.ra.safeRelease(temp2);
            state.ra.safeRelease(temp);
        }

        if (!(remainderK || (problem.abOffset == ABOffset::Calc)
                    || strategy.kParallelLocal || strategy.persistent))
            releaseSavedMNLocalIDs(state);
    }

    // Get register layouts for A/B/C.
    if (!getRegLayout(Ta_load, state.A_layout, unrollM, strategy.ka_load,
                remM_A, remK_A, false, true, 0, 0, problem.A, strategy.A))
        return false;
    if (!getRegLayout(Tb_load, state.B_layout, strategy.kb_load, unrollN,
                remK_B, remN_B, false, true, 0, 0, problem.B, strategy.B))
        return false;

    if (state.copyC) {
        bool cColMajor = isRegisterColMajor(Tc_ext, problem.C, strategy.C);
        makeUnbackedRegLayout(Tc, state.C_layout, unrollM, unrollN, cColMajor,
                1, strategy.C.tileR, strategy.C.tileC);
        if (!getRegLayout(Tc_ext, state.C_layoutExt, unrollM, unrollN, remM_Ce,
                    remN_Ce, true, false, 0, 0, problem.C, state.Cext_strategy))
            return false;
    } else {
        if (!getRegLayout(Tc, state.C_layout, unrollM, unrollN, remM_C, remN_C,
                    true, false, 0, 0, problem.C, strategy.C))
            return false;
    }

    if (!strategy.altCRemainder && (remM_Ce || remN_Ce)) {
        // Try preparing C layout without masking (may reduce memory accesses).
        // Only use it if compatible with the masked layout, and saves on send instructions.
        auto &layoutExt = state.copyC ? state.C_layoutExt : state.C_layout;
        (void)getRegLayout(Tc_ext, state.C_layoutExtUnmasked, unrollM, unrollN,
                false, false, true, false, 0, 0, problem.C,
                state.Cext_strategy);
        if (state.C_layoutExtUnmasked.size() == layoutExt.size()
                || (!state.copyC
                        && !matchLayouts(
                                Tc, layoutExt, state.C_layoutExtUnmasked)))
            state.C_layoutExtUnmasked.clear();
    }

    if (!state.copyC) state.C_layoutExt = state.C_layout;

    if (hasRowFragmenting(state.A_layout)
            || hasColumnFragmenting(state.B_layout)) {
        status << "Can't fragment A or B.\n";
        return false;
    }

    // Prepare to repack A/B if needed.
    int crosspackA, crosspackB, tileM_A, tileK_A, tileK_B, tileN_B;
    std::tie(crosspackA, crosspackB)
            = targetKernelCrosspack(hw, problem, strategy);
    std::tie(tileM_A, tileK_A, tileK_B, tileN_B)
            = targetKernelTiling(hw, problem, strategy);

    state.repackA
            |= (crosspackA && !hasFullCrosspack(state.A_layout, crosspackA))
            || !hasTiling(state.A_layout, tileM_A, tileK_A);
    state.repackB
            |= (crosspackB && !hasFullCrosspack(state.B_layout, crosspackB))
            || !hasTiling(state.B_layout, tileK_B, tileN_B);

    state.repackA |= (Ta.size() != Ta_ext.size() && !strategy.slmA);
    state.repackB |= (Tb.size() != Tb_ext.size() && !strategy.slmB);

    if (crosspackA == 0) crosspackA = 1;
    if (crosspackB == 0) crosspackB = 1;

    if (state.repackA)
        makeUnbackedRegLayout(Ta, state.Ar_layout, unrollM, strategy.ka_load,
                isLayoutColMajor(state.A_layout), crosspackA, tileM_A, tileK_A);
    if (state.repackB)
        makeUnbackedRegLayout(Tb, state.Br_layout, strategy.kb_load, unrollN,
                isLayoutColMajor(state.B_layout), crosspackB, tileK_B, tileN_B);

    // Prepare layouts for row/column sum calculation.
    if (problem.abOffset == ABOffset::Calc) {
        auto As_srcLayout = strategy.slmA
                ? state.Ao_layout
                : state.repackA ? state.Ar_layout : state.A_layout;
        auto Bs_srcLayout = strategy.slmB
                ? state.Bo_layout
                : state.repackB ? state.Br_layout : state.B_layout;
        makeSumLayout(
                false, Ta, As_srcLayout, Tc, state.As_layout, strategy, state);
        makeSumLayout(
                true, Tb, Bs_srcLayout, Tc, state.Bs_layout, strategy, state);
    }

    // Round up needed A/B flag registers; hold off on C.
    // Try first without virtual flags and retry if needed.
    // m/n cooperative SLM copies use k masking, so skip those masks for now.
    vector<MaskAssignment> masks;

    auto assignAllMasks = [&]() {
        return assignMasks(state.A_layout, LoopM, LoopK, masks, state)
                && assignMasks(state.Ap_layout, LoopM, LoopK, masks, state)
                && assignMasks(state.B_layout, LoopK, LoopN, masks, state)
                && assignMasks(state.Bp_layout, LoopK, LoopN, masks, state)
                && ((state.effCoopA != CoopSplit::K)
                        || assignMasks(
                                state.Ai_layout, LoopM, LoopK, masks, state))
                && ((state.effCoopB != CoopSplit::K)
                        || assignMasks(
                                state.Bi_layout, LoopK, LoopN, masks, state));
    };

    bool success = assignAllMasks();
    if (!success && state.vflagStorage.isInvalid()) {
        status << "Retrying with virtual flags." << status_stream::endl;
        allocVFlagStorage(strategy, state);
        success = assignAllMasks();
        lateKLoopCheck = true;
    }

    if (!success) return false;

    loadMasks(masks, state.remainders, strategy, state);

    // Temporary: move add64 out of the way (later: general cramming).
    if (state.add64.isValid()) {
        auto oldAdd64 = state.add64;
        state.ra.safeRelease(state.add64);
        state.add64 = state.ra.alloc_sub<uint32_t>();
        if (oldAdd64 != state.add64) mov(1, state.add64, oldAdd64);
    }

    // Allocate data registers.
    gemmAllocRegs(problem, strategy, state);
    gemmAllocAoBoRegs(strategy, state);

    // Allocate address registers for A/B loads. We don't need C addresses yet.
    allocAddrRegs(state.A_addrs, state.A_layout, problem.A, strategy.A, state);
    allocAddrRegs(state.B_addrs, state.B_layout, problem.B, strategy.B, state);
    allocAddrRegs(state.Ap_addrs, state.Ap_layout, globalA, strategy.A_prefetch,
            state);
    allocAddrRegs(state.Bp_addrs, state.Bp_layout, globalB, strategy.B_prefetch,
            state);
    allocAddrRegs(state.Ai_addrs, state.Ai_layout, state.Ai, state.Ai_strategy,
            state);
    allocAddrRegs(state.Bi_addrs, state.Bi_layout, state.Bi, state.Bi_strategy,
            state);
    allocAddrRegs(state.Ao_addrs, state.Ao_layout, state.Ao, state.Ao_strategy,
            state);
    allocAddrRegs(state.Bo_addrs, state.Bo_layout, state.Bo, state.Bo_strategy,
            state);

    // Free up some C registers temporarily for use in address calculations.
    releaseRanges(state.C_regs, state);

    // Set up address registers.
    gemmCacheLDABMultiples(problem, strategy, state);
    setupAddr(Ta_ext, state.Ap_addrs, state.effAp, state.Ap_layout,
            state.inputs.lda, globalA, strategy.A_prefetch, strategy, state,
            A_params, state.ldaMultiples);
    setupAddr(Tb_ext, state.Bp_addrs, state.effBp, state.Bp_layout,
            state.inputs.ldb, globalB, strategy.B_prefetch, strategy, state,
            B_params, state.ldbMultiples);
    setupAddr(Ta_ext, state.Ai_addrs, state.effAi, state.Ai_layout,
            state.inputs.lda, state.Ai, state.Ai_strategy, strategy, state,
            Ai_params, state.ldaMultiples);
    setupAddr(Tb_ext, state.Bi_addrs, state.effBi, state.Bi_layout,
            state.inputs.ldb, state.Bi, state.Bi_strategy, strategy, state,
            Bi_params, state.ldbMultiples);
    setupAddr(Ta, state.Ao_addrs, state.effAo, state.Ao_layout, Subregister(),
            state.Ao, state.Ao_strategy, strategy, state);
    setupAddr(Tb, state.Bo_addrs, state.effBo, state.Bo_layout, Subregister(),
            state.Bo, state.Bo_strategy, strategy, state);
    setupAddr(Ta_load, state.A_addrs, state.effA, state.A_layout,
            state.inputs.lda, problem.A, strategy.A, strategy, state, A_params,
            state.ldaMultiples);
    setupAddr(Tb_load, state.B_addrs, state.effB, state.B_layout,
            state.inputs.ldb, problem.B, strategy.B, strategy, state, B_params,
            state.ldbMultiples);

    // Free unneeded registers after address setup.
    if (!state.isNested) {
        state.ra.safeRelease(state.h0);
        if (strategy.A.address2D
                && (!strategy.prefetchA || strategy.A_prefetch.address2D))
            state.ra.safeRelease(state.inputs.lda);
        if (strategy.B.address2D
                && (!strategy.prefetchB || strategy.B_prefetch.address2D))
            state.ra.safeRelease(state.inputs.ldb);
        if (!strategy.C.address2D
                && (!strategy.prefetchC || !strategy.C_prefetch.address2D)) {
            state.ra.safeRelease(state.i0);
            state.ra.safeRelease(state.j0);
        }
    }
    if (state.Ai_strategy.address2D) {
        if (Ai_params.offR != A_params.offR)
            state.ra.safeRelease(Ai_params.offR);
        if (Ai_params.offC != A_params.offC)
            state.ra.safeRelease(Ai_params.offC);
    }
    if (state.Bi_strategy.address2D) {
        if (Bi_params.offR != B_params.offR)
            state.ra.safeRelease(Bi_params.offR);
        if (Bi_params.offC != B_params.offC)
            state.ra.safeRelease(Bi_params.offC);
    }
    if (strategy.A_prefetch.address2D) {
        if (Ap_params.offR != A_params.offR)
            state.ra.safeRelease(Ap_params.offR);
        if (Ap_params.offC != A_params.offC)
            state.ra.safeRelease(Ap_params.offC);
    }
    if (strategy.B_prefetch.address2D) {
        if (Bp_params.offR != B_params.offR)
            state.ra.safeRelease(Bp_params.offR);
        if (Bp_params.offC != B_params.offC)
            state.ra.safeRelease(Bp_params.offC);
    }

    if (!one_of(state.effAp, state.effA, state.effAi))
        state.ra.safeRelease(state.effAp);
    if (!one_of(state.effBp, state.effB, state.effBi))
        state.ra.safeRelease(state.effBp);

    releaseLDMultiples(state.ldaMultiples, state);
    releaseLDMultiples(state.ldbMultiples, state);
    releaseIndexVec(state);

    reclaimRanges(state.C_regs, state);

    // Allocate tokens.
    success = true;
    for (int q = 0; q < strategy.A_copies; q++)
        success &= allocateTokens(state.A_layout, state.A_regs[q], state);
    for (int q = 0; q < strategy.B_copies; q++)
        success &= allocateTokens(state.B_layout, state.B_regs[q], state);
    for (int q = 0; q < strategy.slmCopies; q++) {
        if (strategy.slmA)
            success &= allocateTokens(state.Ai_layout, state.Ai_regs[q], state);
        if (strategy.slmB)
            success &= allocateTokens(state.Bi_layout, state.Bi_regs[q], state);
    }
    if (!state.aioShare)
        success &= allocateTokens(state.Ao_layout, state.Ao_regs, state);
    if (!state.bioShare)
        success &= allocateTokens(state.Bo_layout, state.Bo_regs, state);
    success &= allocateTokens(
            state.Ap_layout, state.Ap_regs, state, state.Ap_addrs);
    success &= allocateTokens(
            state.Bp_layout, state.Bp_regs, state, state.Bp_addrs);
    if (!success) {
        if (hw >= HW::Gen12LP)
            status << "Not enough tokens for k loop." << status_stream::endl;
        clearTokenAllocations(hw, state);
    }

    // Load C now if configured.
    //  - temporarily free A/B data regs to use as C headers
    //  - do beta scaling
    if (cLoadAhead) {
        if (problem.checkBeta0 && !problem.beta_real.fixed()) stub();
        if (state.C_accCount > 0) stub();
        if (strategy.kParallelLocal) stub();

        releaseRanges(state.A_regs, state);
        releaseRanges(state.B_regs, state);
        if (!state.Ar_regs.empty()) releaseRanges(state.Ar_regs, state);
        if (!state.Br_regs.empty()) releaseRanges(state.Br_regs, state);

        status << "Loading C" << status_stream::endl;
        gemmAccessC(COperation::Load, problem, strategy, state);

        gemmBetaScale(problem, strategy, state);
        if (!state.Br_regs.empty()) reclaimRanges(state.Br_regs, state);
        if (!state.Ar_regs.empty()) reclaimRanges(state.Ar_regs, state);
        reclaimRanges(state.B_regs, state);
        reclaimRanges(state.A_regs, state);
    }

    for (int q = 0; q < state.C_count; q++)
        releaseLDMultiples(state.ldcMultiples[q], state);
    releaseIndexVec(state);

    // Release 64-bit emulation registers as they aren't needed in the inner loop.
    // Could also move r0 to acc here.
    GRF emulate64Temp[2];
    if (state.emulate.temp[0].isValid()) {
        for (int q = 0; q < 2; q++) {
            emulate64Temp[q] = state.emulate.temp[q];
            state.ra.safeRelease(state.emulate.temp[q]);
        }
        if (GRF::bytes(hw) == 64) {
            // Need a whole flag register to do emulated SIMD16 arithmetic.
            state.emulate.flag = state.raVFlag.alloc();
            state.emulate.flagOffset = 0;
        } else {
            state.emulate.flag = state.flagAP;
            state.emulate.flagOffset = 8;
            lateKLoopCheck = false;
        }
    }

    // Synthesize k loop. If configured, choose between 32-bit adds and 64-bit adds.
    if (strategy.checkAdd32 && state.add64.isValid()) {
        Label loop64, done;
        bool success = true;

        cmp(1 | ne | state.flagAP, state.add64, uint16_t(0));
        jmpi(1 | state.flagAP, loop64);
        state.ra.safeRelease(state.add64);

        status << "k loop: 32-bit address update" << status_stream::endl;
        strategy.emulate.emulate64_add32 = true;
        auto substate32 = state;
        success &= gemmKLoopDispatch(
                lateKLoopCheck, problem, strategy, substate32);
        jmpi(1, done);

        mark(loop64);
        status << "k loop: 64-bit address update" << status_stream::endl;
        strategy.emulate.emulate64_add32 = false;
        success &= gemmKLoopDispatch(lateKLoopCheck, problem, strategy, state);

        mark(done);
        if (!success) return false;
    } else {
        state.ra.safeRelease(state.add64);
        if (!gemmKLoopDispatch(lateKLoopCheck, problem, strategy, state))
            return false;
    }

    // We're done with A and B. Free their address, data, and flag registers.
    // Also done with loop counter.
    safeReleaseMaskAssignments(masks, state);
    safeReleaseRanges(state.A_addrs, state);
    safeReleaseRanges(state.B_addrs, state);
    safeReleaseRanges(state.Ai_addrs, state);
    safeReleaseRanges(state.Bi_addrs, state);
    safeReleaseRanges(state.Ao_addrs, state);
    safeReleaseRanges(state.Bo_addrs, state);
    safeReleaseRanges(state.Ap_addrs, state);
    safeReleaseRanges(state.Bp_addrs, state);

    safeReleaseRanges(state.A_regs, state);
    safeReleaseRanges(state.Ar_regs, state);
    safeReleaseRanges(state.Ai_regs, state);
    safeReleaseRanges(state.Ao_regs, state);
    safeReleaseRanges(state.Ap_regs, state);
    safeReleaseRanges(state.B_regs, state);
    safeReleaseRanges(state.Br_regs, state);
    safeReleaseRanges(state.Bi_regs, state);
    safeReleaseRanges(state.Bo_regs, state);
    safeReleaseRanges(state.Bp_regs, state);
    state.ra.safeRelease(state.broadcast_regs);
    safeReleaseRanges(state.tempMul_regs, state);
    clearTokenAllocations(hw, state);

    state.A_layout.clear();
    state.B_layout.clear();
    state.Ai_layout.clear();
    state.Bi_layout.clear();
    state.Ao_layout.clear();
    state.Bo_layout.clear();
    state.Ar_layout.clear();
    state.Br_layout.clear();
    state.Ap_layout.clear();
    state.Bp_layout.clear();
    state.Cp_layout.clear();

    // Restore A/B addresses and strategies that were modified by SLM copies.
    if (strategy.slmA) {
        state.ra.safeRelease(state.effA);
        state.ra.safeRelease(state.effAo);
        state.effA = state.effAi;
        state.effAi = invalid;
        strategy.A = saveAStrategy;
    }
    if (strategy.slmB) {
        state.ra.safeRelease(state.effB);
        state.ra.safeRelease(state.effBo);
        state.effB = state.effBi;
        state.effBi = invalid;
        strategy.B = saveBStrategy;
    }

    // Put accumulators with the rest of C.
    if (state.C_accCount > 0) {
        // Reclaim the bottom registers of C.
        reclaimRanges(state.C_regs[0], state);

        auto e = elementsPerGRF<uint32_t>(hw);
        for (int i = 0; i < state.C_accCount; i += 2)
            mov<uint32_t>(2 * e, state.C_regs[0][i], AccumulatorRegister(i));
    }

    // Restore emulation registers.
    if (emulate64Temp[0].isValid()) {
        for (int q = 0; q < 2; q++) {
            state.emulate.temp[q] = emulate64Temp[q];
            if (emulate64Temp[q].isValid()) state.ra.claim(emulate64Temp[q]);
        }
        if (GRF::bytes(hw) == 64) state.raVFlag.release(state.emulate.flag);
        state.emulate.flag = invalid;
        state.emulate.flagOffset = 0;
    }

    return true;
}

template <HW hw>
void gemm_kernel_generator_t<hw>::setupCAddr0(GRFRange (&C_addr0)[2],
        GRFRange (&C_addr0Unmasked)[2], const vector<RegisterBlock> &C_layout,
        const vector<RegisterBlock> &C_layoutUnmasked, int C_count,
        GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state) {
    Address2DParams params;
    params.rows = state.inputs.m;
    params.cols = state.inputs.n;
    params.offR = state.i0;
    params.offC = state.j0;
    params.remR = state.remainders[LoopM];
    params.remC = state.remainders[LoopN];
    for (int q = 0; q < C_count; q++) {
        C_addr0[q] = state.ra.alloc_range(
                addrGRFCount(problem.C, strategy.C, C_layout[0]));
        setupAddr(C_addr0[q], state.effC[q], C_layout[0], state.inputs.ldc[q],
                problem.Tc.size(), problem.C, strategy.C, strategy, state,
                params, state.ldcMultiples[q]);
    }
    if (!C_layoutUnmasked.empty())
        for (int q = 0; q < C_count; q++) {
            C_addr0Unmasked[q] = state.ra.alloc_range(
                    addrGRFCount(problem.C, strategy.C, C_layoutUnmasked[0]));
            setupAddr(C_addr0Unmasked[q], state.effC[q], C_layoutUnmasked[0],
                    state.inputs.ldc[q], problem.Tc.size(), problem.C,
                    strategy.C, strategy, state, params, state.ldcMultiples[q]);
        }
}

template <HW hw>
bool gemm_kernel_generator_t<hw>::gemmUpdateC(
        GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state) {

    auto Ts = problem.Ts;

    status << "C update" << status_stream::endl;

    auto &alphar = problem.alpha_real;
    auto &betar = problem.beta_real;

    if (strategy.cLoadAhead) {
        betar = 0;
        if (!problem.alpha1()) stub();
    }

    // C early offset.
    if (problem.cOffset == COffset::Pre)
        if (!gemmApplyCOffsetDispatch(problem, strategy, state)) return false;

    // Prepare postop injector if configured.
    GRFRange postOpScratch;
    if (problem.hasPostOp()) {
        postOpInjector.reset(new Injector(this, problem.Ts.get_dnnl_type(),
                problem.post_ops, GRFRange(), problem.postOpFwd));
        if (!postOpInjector) stub();

        postOpScratch = state.ra.try_alloc_range(
                postOpInjector->preferred_scratch_regs());
        if (postOpScratch.isInvalid())
            postOpScratch
                    = state.ra.alloc_range(postOpInjector->min_scratch_regs());
        postOpInjector->set_scratch(postOpScratch);
    }

    // Convert C to the type of alpha/beta if needed and if possible (no data size change).
    // If not possible, must be done at a lower level during C update.
    bool successfulConvert = true;

    if (problem.needsTsConvert())
        successfulConvert = gemmConvertC(Ts, problem, strategy, state);

    // Scale by alpha now if alpha and beta are both nontrivial. Todo: move above beta = 0 check,
    //  handle double precision correctly (load alpha to register first).
    // Also scale if atomically updating C or for split-complex.
    bool nontrivialAlpha = !problem.alpha1() && !problem.alphaM1();
    bool forceScale = !problem.alpha1() && strategy.C.atomic;

    if (successfulConvert
            && ((nontrivialAlpha && (!problem.beta1() || strategy.doubleWA))
                    || forceScale)) {

        if (alphar != 1)
            map(hw, Ts.real(), state.C_regs[0], state.C_regs[0], strategy,
                    [&](int esize, GRF acc, GRF _) {
                        alphar.fixed()
                                ? mul(esize, acc, acc, cast(Ts.real(), alphar))
                                : mul(esize, acc, acc,
                                        alphar.getRegAvoiding(hw, acc));
                    });

        alphar = 1;
    }

    // Do the actual updating.
    if (!gemmAccessC(COperation::UpdateStore, problem, strategy, state))
        return false;

    // Postop cleanup.
    if (problem.hasPostOp()) {
        postOpInjector.reset();
        state.ra.safeRelease(postOpScratch);
    }

    // Free C data and layout.
    safeReleaseRanges(state.C_regs, state);
    state.C_layout.clear();
    state.C_layoutExt.clear();

    state.raVFlag.safeRelease(state.flagSwizzle);

    // Success!
    return true;
}

// Load from, update, and/or store to C, with complete remainder handling.
// If op == COperation::Load, only load C.
// If op == COperation::Update, load and update C.
// If op == COperation::UpdateStore, perform full C update with alpha/beta scaling. Unless state.isNested == true, assumed
//   to be the conclusion of the kernel.
template <HW hw>
bool gemm_kernel_generator_t<hw>::gemmAccessC(COperation op,
        GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state) {
    Label labelAltCRemainder, labelAltCRemDone, labelSkip;

    int C_count = (op == COperation::UpdateStore) ? state.C_count : 1;
    bool remainderM
            = (strategy.remHandling[LoopM] != RemainderHandling::Ignore);
    bool remainderN
            = (strategy.remHandling[LoopN] != RemainderHandling::Ignore);
    bool remM_C, remN_C;
    getCRemainders(problem, strategy, remM_C, remN_C);
    bool altCRemainder = strategy.altCRemainder && !strategy.C.padded
            && (remainderM || remainderN || problem.gemmt());
    bool stdCRemainder = !(altCRemainder
            && (strategy.remHandling[LoopM]
                    == RemainderHandling::KnownRemainder)
            && (strategy.remHandling[LoopN]
                    == RemainderHandling::KnownRemainder));

    if ((op != COperation::UpdateStore) && strategy.C.atomic) stub();

    if (state.allowEmptyC && (remainderM || remainderN)) {
        if (!state.isNested) stub();
        int simt = strategy.fused ? 16 : 1;
        cmp(simt | le | f0[0], null.ud(), state.remainders[LoopM], 0);
        cmp(simt | le | f1[0], null.ud(), state.remainders[LoopN], 0);
        strategy.fused ? goto12(16 | f0[0] | anyv, labelSkip)
                       : ejmpi(1 | f0[0] | anyv, labelSkip);
    }

    if (op == COperation::UpdateStore && problem.cOffset == COffset::Post) {
        // C postoffset is implemented by splitting the update and store steps.
        bool ok = true;
        bool oldAllowEmptyC = state.allowEmptyC;
        state.allowEmptyC = false;

        if (!(problem.alpha1() && problem.beta0()))
            ok = ok
                    && gemmAccessC(
                            COperation::Update, problem, strategy, state);
        auto storeProblem = problem;
        storeProblem.cOffset = COffset::None;
        storeProblem.alpha_real = 1;
        storeProblem.alpha_imag = 0;
        storeProblem.beta_real = 0;
        storeProblem.beta_imag = 0;
        gemmConvertC(problem.Tc, problem, strategy, state);
        ok = ok && gemmApplyCOffsetDispatch(problem, strategy, state);
        ok = ok
                && gemmAccessC(
                        COperation::UpdateStore, storeProblem, strategy, state);

        state.allowEmptyC = oldAllowEmptyC;
        if (ok && state.allowEmptyC && (remainderM || remainderN)) {
            mark(labelSkip);
            if (strategy.fused) join(16);
        }
        return ok;
    }

    if (stdCRemainder) {
        // Check to see if we should jump to alternate C remainder handling path, when enabled:
        //  - if this a remainder kernel
        //  - for triangular updates, if the diagonal crosses this block.
        //       When fusing, check diagonal for thread 0 for (fused in n) upper/m lower, thread 1 for n lower/m upper.
        if (altCRemainder) {
            if (remainderM || remainderN) {
                cmp(1 | lt | f0[0], null.ud(), state.remaindersFused[LoopM],
                        strategy.unroll[LoopM]);
                cmp(1 | lt | f1[0], null.ud(), state.remaindersFused[LoopN],
                        strategy.unroll[LoopN]);
            }

            if (remainderM || remainderN)
                ejmpi(1 | f0[0] | anyv, labelAltCRemainder);
        }

        // Release the all-purpose flag temporarily to free up flag registers if it won't be needed.
        auto saveFlagAP = state.flagAP;
        if (!problem.hasPostOp())
            if (!strategy.fused && !strategy.noJumpTables
                    && state.emulate.flag != state.flagAP)
                state.raVFlag.safeRelease(state.flagAP);

        // Decide on the C remainder handling strategy.
        bool fragments[2] = {false, false};
        bool fragPositives[2] = {true, true};
        int fragSizes[2] = {1 << 16, 1 << 16};

        // Check for fragmenting.
        auto &C_layoutExt = state.C_layoutExt;
        auto &C_layoutExtUnmasked = state.C_layoutExtUnmasked;
        bool remDescs[2] = {false, false};
        bool remMasks[2] = {false, false};

        // Loop over rows (rc = 0) and columns (rc = 1)
        for (int rc = 0; rc < 2; rc++) {
            if (!(rc ? remN_C : remM_C))
                continue; // Skip if not doing remainder handling in this dimension.

            for (auto &l : C_layoutExt) {
                auto qFragment = rc ? l.colFragment : l.rowFragment;
                bool qZeroOK = rc ? l.noColsOK : l.noRowsOK;
                bool qMasked = rc ? (bool)l.colMask : (bool)l.rowMask;
                bool qDescRem = rc ? l.descRemC : l.descRemR;

                if (qFragment > 0) {
                    fragments[rc] = true;
                    fragSizes[rc] = std::min<int>(fragSizes[rc], qFragment);
                    if (qZeroOK) fragPositives[rc] = false;

                    if (qFragment > 1) {
                        remDescs[rc] |= qDescRem;
                        remMasks[rc] |= !qDescRem;
                    }
                } else
                    remMasks[rc] |= qMasked;
            }
        }

        // Disable fragmentation if fragment size is bigger than unroll.
        fragments[0] &= fragSizes[0] < strategy.unroll[LoopM];
        fragments[1] &= fragSizes[1] < strategy.unroll[LoopN];

        // Sanity check the requirements.
        if ((remDescs[0] && remMasks[0]) || (remDescs[1] && remMasks[1])) {
            status << "Different remainder types mixed in C layout."
                   << status_stream::endl;
            return false;
        }
        if (remMasks[0] && remMasks[1]) {
            status << "Both dimensions are masked (not supported)."
                   << status_stream::endl;
            return false;
        }
        if (remDescs[0] && remDescs[1]) {
            status << "Both dimensions use descriptors (not supported)."
                   << status_stream::endl;
            return false;
        }

        // Set remainder handling types.
        StdCRemType remTypes[2] = {StdCRemType::Ignore, StdCRemType::Ignore};
        for (int rc = 0; rc < 2; rc++) {
            if (remDescs[rc])
                remTypes[rc] = StdCRemType::Descriptor;
            else if (remMasks[rc])
                remTypes[rc] = StdCRemType::Mask;
        }

        // Decide whether to do m or n first. Criteria, in order of priority:
        //   - Do an ignored dimension first.
        //   - Do a fragmented dimension first.
        //   - Do descriptors first.
        //   - Do whichever dimension of C is strided first.
        bool nFirst;
        if (remTypes[0] == StdCRemType::Ignore
                || remTypes[1] == StdCRemType::Ignore)
            nFirst = (remTypes[1] == StdCRemType::Ignore);
        else if (fragments[0] != fragments[1])
            nFirst = fragments[1];
        else if (remDescs[0] || remDescs[1])
            nFirst = remDescs[1];
        else
            nFirst = (problem.C.layout == MatrixLayout::N);

        // Cache ldc multiples.
        gemmCacheLDCMultiples(problem, strategy, state);

        // Prepare for load/store descriptor generation.
        if (remDescs[0] || remDescs[1])
            setupTeardownLoadStoreDesc(true, C_layoutExt, strategy, state);

        // Set up address for the beginning of C.
        GRFRange C_addr0[2], C_addr0Unmasked[2];
        setupCAddr0(C_addr0, C_addr0Unmasked, C_layoutExt, C_layoutExtUnmasked,
                C_count, problem, strategy, state);

        // Try to load C masks. If that fails, fragment the masked dimension down to the size of current blocks.
        vector<MaskAssignment> masks;
        if (!assignMasks(C_layoutExt, LoopM, LoopN, masks, state)) {
            for (int rc = 0; rc < 2; rc++) {
                if (remMasks[rc]) {
                    fragments[rc] = true;
                    fragSizes[rc] = rc ? C_layoutExt[0].nc : C_layoutExt[0].nr;
                }
            }
        } else
            loadMasks(masks, state.remainders, strategy, state);

        // Call the remainder handling routine. If it fails, try again, switching M and N.
        // If that still fails, then try again with complete fragmentation if partial
        //  fragmentation attempted the first time.
        bool columns[2] = {nFirst, !nFirst};
        bool switchedColumns[2] = {!nFirst, nFirst};
        do {
            if (doStdCRemainder(C_layoutExt, C_layoutExtUnmasked, false,
                        columns, remTypes, fragments, fragPositives, fragSizes,
                        C_addr0, C_addr0Unmasked, op, masks, problem, strategy,
                        state))
                break;
            if (doStdCRemainder(C_layoutExt, C_layoutExtUnmasked, false,
                        switchedColumns, remTypes, fragments, fragPositives,
                        fragSizes, C_addr0, C_addr0Unmasked, op, masks, problem,
                        strategy, state))
                break;

            if ((fragments[0] && (fragSizes[0] > 1))
                    || (fragments[1] && (fragSizes[1] > 1))) {
                fragSizes[0] = fragSizes[1] = 1;

                if (doStdCRemainder(C_layoutExt, C_layoutExtUnmasked, false,
                            columns, remTypes, fragments, fragPositives,
                            fragSizes, C_addr0, C_addr0Unmasked, op, masks,
                            problem, strategy, state))
                    break;
                if (doStdCRemainder(C_layoutExt, C_layoutExtUnmasked, false,
                            switchedColumns, remTypes, fragments, fragPositives,
                            fragSizes, C_addr0, C_addr0Unmasked, op, masks,
                            problem, strategy, state))
                    break;
            }
            return false;
        } while (false);

        // Free cached ldc multiples.
        for (int q = 0; q < state.C_count; q++)
            releaseLDMultiples(state.ldcMultiples[q], state);
        releaseIndexVec(state);

        // Free address header for block 0.
        for (int q = 0; q < C_count; q++)
            state.ra.safeRelease(C_addr0[q]);

        // Free C mask registers.
        safeReleaseMaskAssignments(masks, state);

        // Clean up after load/store descriptor generation.
        if (remDescs[0] || remDescs[1])
            setupTeardownLoadStoreDesc(false, C_layoutExt, strategy, state);

        // Restore all-purpose flag.
        state.flagAP = saveFlagAP;
        state.raVFlag.claim(saveFlagAP);
    }

    // Do alternate C remainder handling if enabled.
    if (altCRemainder) {
        if (stdCRemainder) {
            if (state.isNested || (op != COperation::UpdateStore))
                jmpi(1, labelAltCRemDone);
            else
                epilogue(strategy, state);
        }
        mark(labelAltCRemainder);
        doAlternateCRemainder(op, problem, strategy, state);
        mark(labelAltCRemDone);
    }

    // C accumulators were converted back to the regular C type.
    state.Tacc = problem.Tc;

    if (state.allowEmptyC && (remainderM || remainderN)) {
        mark(labelSkip);
        if (strategy.fused) join(16);
    }

    return true; /* Successful! */
}

template <HW hw>
bool gemm_kernel_generator_t<hw>::gemmBodyInternal(
        GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state) {
    auto Tc = problem.Tc;

    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];
    auto &remM = state.remainders[LoopM];
    auto &remN = state.remainders[LoopN];

    // Accumulate C with panel*panel multiply.
    if (!gemmAccumulateC(problem, strategy, state)) return false;

    // Add A/B offsets.
    gemmLoadABOffset(problem, strategy, state);
    if (!gemmFinalizeSums(problem, strategy, state)) return false;
    gemmApplyABOffset(problem, strategy, state);

    // If C is packed, update remainders and prepare to mask out border regions.
    bool remaskC_M = isPacked(problem.C.layout)
            && (strategy.remHandling[LoopM] != RemainderHandling::Ignore);
    bool remaskC_N = isPacked(problem.C.layout)
            && (strategy.remHandling[LoopN] != RemainderHandling::Ignore);

    if (remaskC_M || remaskC_N) {
        if (remaskC_M)
            setupTeardownRemask(Tc, 0, true, unrollM, remM, strategy, state);
        if (remaskC_N)
            setupTeardownRemask(Tc, 1, true, unrollN, remN, strategy, state);

        int C_mgran, C_ngran;
        getGranularities(problem.C, C_mgran, C_ngran);
        if (!remaskC_M || C_mgran == unrollM) C_mgran = 1;
        if (!remaskC_N || C_ngran == unrollN) C_ngran = 1;
        if (!is_zero_or_pow2(C_mgran)) stub();
        if (!is_zero_or_pow2(C_ngran)) stub();

        if (C_mgran > 1) add(1, remM, remM, C_mgran - 1);
        if (C_ngran > 1) add(1, remN, remN, C_ngran - 1);
        if (C_mgran > 1) and_(1, remM, remM, uint32_t(~(C_mgran - 1)));
        if (C_ngran > 1) and_(1, remN, remN, uint32_t(~(C_ngran - 1)));
    }

    // Local k reduction.
    if (strategy.kParallelLocal) gemmKReduce(problem, strategy, state);

    // Late exit.
    bool lateExit = strategy.lateExit();
    Label labelLateExit;

    if (lateExit) {
        int simt = strategy.fused ? 16 : 1;

        cmp(simt | le | f0[0], state.remainders[LoopM], uint16_t(0));
        cmp(simt | le | f1[0], state.remainders[LoopN], uint16_t(0));

        InstructionModifier cond = simt | f0[0] | anyv;

        strategy.fused ? goto12(cond, labelLateExit)
                       : ejmpi(cond, labelLateExit);
    }

    // Update C. If configured, choose between regular beta and beta = 0 or beta = 1 updates now.
    bool checkBeta0 = problem.checkBeta0 && !problem.beta_real.fixed();
    bool checkBeta1 = strategy.checkBeta1 && !problem.beta_real.fixed();
    bool checkTRMMBeta1 = state.beta1.isValid();

    if (checkTRMMBeta1 && (checkBeta0 || checkBeta1)) stub();

    if (!checkBeta0 && !checkBeta1 && !checkTRMMBeta1) {
        if (!gemmUpdateC(problem, strategy, state)) return false;
    } else {
        Label labelBeta0, labelBeta1, labelBetaDone;
        InstructionModifier mod0 = 1 | f0[0];
        InstructionModifier mod1 = 1 | f0[1];
        bool simtCF1 = false;

        if (checkBeta0) { cmp0(1 | eq | f0[0], problem.beta_real.getReg(0)); }

        if (checkBeta1) {
            cmp(1 | eq | f0[1], problem.beta_real.getReg(0),
                    cast(problem.Ts, 1.0));
        }

        if (checkBeta0) jmpi(mod0, labelBeta0);

        if (checkBeta1 || checkTRMMBeta1) {
            simtCF1 ? if_(mod1, labelBeta1, labelBetaDone)
                    : jmpi(mod1, labelBeta1);
        }

        // Regular update.
        {
            auto subproblem = problem;
            auto substrategy = strategy;
            auto substate = state;

            if (strategy.C.atomic && !strategy.C.base.isStateless()
                    && !strategy.C.newDP)
                stub(); /* need to shift addresses */
            substrategy.C.atomic = false;

            if (!gemmUpdateC(subproblem, substrategy, substate)) return false;
        }

        simtCF1 ? else_(16, labelBetaDone)
                : state.isNested ? jmpi(1, labelBetaDone)
                                 : epilogue(strategy, state);

        // beta = 1 update.
        if (checkBeta1 || checkTRMMBeta1) {
            status << "Special path: beta = 1" << status_stream::endl;
            mark(labelBeta1);

            auto subproblem = problem;
            auto substate = state;

            subproblem.beta_real = 1;
            subproblem.beta_imag = 0;

            if (!gemmUpdateC(subproblem, strategy, substate)) return false;

            if (checkBeta0) {
                (simtCF1 || state.isNested) ? jmpi(1, labelBetaDone)
                                            : epilogue(strategy, state);
            }
        }

        // beta = 0 update.
        if (checkBeta0) {
            status << "Special path: beta = 0" << status_stream::endl;
            mark(labelBeta0);

            auto subproblem = problem;
            auto substrategy = strategy;
            auto substate = state;

            subproblem.beta_real = 0;
            subproblem.beta_imag = 0;

            substrategy.C.atomic = false;

            if (!gemmUpdateC(subproblem, substrategy, substate)) return false;
        }

        mark(labelBetaDone);
        if (simtCF1) endif(16);
    }

    // Cleanup.
    if (remaskC_M)
        setupTeardownRemask(Tc, 0, false, unrollM, remM, strategy, state);
    if (remaskC_N)
        setupTeardownRemask(Tc, 1, false, unrollN, remN, strategy, state);

    if (lateExit) {
        mark(labelLateExit);
        if (strategy.fused) join(16);
    }

    return true;
}

template <HW hw>
CoopSplit gemm_kernel_generator_t<hw>::effCoopSplitA(
        const GEMMProblem &problem, const GEMMStrategy &strategy) {
    if (isPacked(problem.A.layout))
        return CoopSplit::Linear;
    else if (!isRegisterColMajor(problem.Ta_ext, problem.A, strategy.A)
            && (strategy.unroll[LoopM] % strategy.wg[LoopN] == 0)
            && !isBlock2D(strategy.A.accessType))
        return CoopSplit::MN;
    else
        return strategy.coopA;
}

template <HW hw>
CoopSplit gemm_kernel_generator_t<hw>::effCoopSplitB(
        const GEMMProblem &problem, const GEMMStrategy &strategy) {
    if (isPacked(problem.B.layout))
        return CoopSplit::Linear;
    else if (isRegisterColMajor(problem.Tb_ext, problem.B, strategy.B)
            && (strategy.unroll[LoopN] % strategy.wg[LoopM] == 0)
            && !isBlock2D(strategy.B.accessType))
        return CoopSplit::MN;
    else
        return strategy.coopB;
}

// Check whether all threads in a thread group should stay together in m/n remainder handling.
template <HW hw>
bool gemm_kernel_generator_t<hw>::wgRemCheck(
        const GEMMProblem &problem, const GEMMStrategy &strategy) {
    return (strategy.slmA && (effCoopSplitA(problem, strategy) != CoopSplit::K)
                   && (strategy.remHandling[LoopM] != RemainderHandling::Ignore)
                   && !strategy.A.padded)
            || (strategy.slmB
                    && (effCoopSplitB(problem, strategy) != CoopSplit::K)
                    && (strategy.remHandling[LoopN]
                            != RemainderHandling::Ignore)
                    && !strategy.B.padded)
            || strategy.kParallelLocal
            || ((strategy.barrierFreq > 0 || strategy.cooperativePF)
                    && (strategy.prefetchA || strategy.prefetchB
                            || strategy.prefetchC));
}

// Do outer-level m/n remainder handling.
template <HW hw>
template <typename Problem>
bool gemm_kernel_generator_t<hw>::mnRemainderHandling(LoopType loop,
        Problem &problem, GEMMStrategy &strategy, GEMMState &state,
        bool (gemm_kernel_generator_t<hw>::*func)(
                Problem, GEMMStrategy, GEMMState)) {
    auto method = strategy.remHandling[loop];
    auto &unroll = strategy.unroll[loop];
    auto mn = (loop == LoopM) ? state.inputs.m : state.inputs.n;
    auto splitThresh
            = (loop == LoopM) ? strategy.mSplitThresh : strategy.nSplitThresh;

    Label label_done;

    auto originalCheckAdd32 = strategy.checkAdd32;

    if (method == RemainderHandling::Split) {
        Label label_remainder;

        // Jump to remainder loop if needed.
        // If threads fused in this direction, factor fused ID into calculation.
        if (wgRemCheck(problem, strategy))
            cmp(1 | lt | f0[0], null.d(), state.remaindersWG[loop],
                    uint16_t(unroll * strategy.wg[loop]));
        else
            cmp(1 | lt | f0[0], null.d(), state.remaindersFused[loop],
                    uint16_t(unroll));

        if (splitThresh) {
            cmp(1 | lt | f1[0], null.d(), mn, int32_t(splitThresh));
            ejmpi(1 | f0[0] | anyv, label_remainder);
        } else
            jmpi(1 | f0[0], label_remainder);

        // First generate code that ignores remainder handling.
        GEMMStrategy substrategy = strategy;
        substrategy.remHandling[loop] = RemainderHandling::Ignore;

        status << "Generating "
               << "MNK"[static_cast<int>(loop)]
               << " non-remainder kernel for unroll " << unroll << '.'
               << status_stream::endl;
        if (!(this->*func)(problem, substrategy, state)) {
            status << "Non-remainder kernel failed, aborting."
                   << status_stream::endl;
            return false;
        }

        // Return, unless this is part of a larger computation, in which case jump to end.
        if (state.isNested)
            jmpi(1, label_done);
        else
            epilogue(strategy, state);

        mark(label_remainder);

        strategy.checkAdd32 = false;
    }

    // OK, great! Now try to create remainder-handling code.
    status << "Attempting to generate "
           << "MNK"[static_cast<int>(loop)] << " general kernel for unroll "
           << unroll << '.' << status_stream::endl;
    bool success = (this->*func)(problem, strategy, state);

    strategy.checkAdd32 = originalCheckAdd32;
    if (success) {
        mark(label_done);
        return true;
    }

#ifndef ALLOW_REMAINDERS
    // Disable remainder code for now.
    return false;
#else
    auto &bound = (loop == LoopN) ? state.inputs.n : state.inputs.m;
    auto &index = (loop == LoopN) ? state.j0 : state.i0;
    auto &remainders = state.remainders[loop];

    if (method == RemainderHandling::Ignore)
        throw std::runtime_error("Could not generate kernel.");

    // It failed, so break up the loop into the next smaller power of 2 along this dimension,
    //  plus the remainder (recursively).
    Label label_next_rem;

    if (unroll == 1) {
        // No more splitting to do.
        // We don't know if this was originally split, so just output a warning.
        status << "NOTE: Split remainder handling is required for loop "
               << "MNK"[static_cast<int>(loop)] << '.' << status_stream::endl;
        return true;
    }
    int chunkSize = rounddown_pow2(unroll - 1);

    // Jump to next remainder loop if needed.
    pushStream();
    {
        cmp(1 | lt | state.flagAP, null.d(), remainders, chunkSize);
        jmpi(1 | state.flagAP, label_next_rem);

        {
            GEMMStrategy substrategy = strategy;
            GEMMState substate = state;
            substrategy.remHandling[loop] = RemainderHandling::Ignore;
            substrategy.unroll[loop] = chunkSize;
            substate.isNested = true;
            status << "Generating "
                   << "MNK"[static_cast<int>(loop)]
                   << " remainder kernel with unroll " << chunkSize << '.'
                   << status_stream::endl;
            if (!(this->*func)(problem, substrategy, substate)) {
                discardStream();
                return false;
            }
        }

        // Adjust remainder.
        add(1, remainders, remainders, -chunkSize);

        // Adjust pointers as needed.
        // A += i0 (N) i0 * lda (T, Pc)
        // B += j0 * ldb (N, Pr) j0 (T)
        // C += i0 + j0 * ldc (N, Pr) j0 + i0 * ldc (T, Pc)
        switch (loop) {
            case LoopM:
                if (problem.A.layout == MatrixLayout::N)
                    eadd(1, state.effA, state.effA, chunkSize * Ta, strategy,
                            state);
                else {
                    Subregister temp = state.ra.alloc_sub<uint32_t>();
                    mulConstant(1, temp, state.inputs.lda, chunkSize);
                    eadd(1, state.effA, state.effA, temp, strategy, state);
                    state.ra.safeRelease(temp);
                }
                if (problem.C.layout == MatrixLayout::N
                        || problem.C.layout == MatrixLayout::Pr)
                    eadd(1, state.effC, state.effC,
                            chunkSize * transaction_safe, strategy, state);
                else {
                    Subregister temp = state.ra.alloc_sub<uint32_t>();
                    mulConstant(1, temp, state.inputs.lda, chunkSize);
                    eadd(1, state.effA, state.effA, temp, strategy, state);
                    state.ra.safeRelease(temp);
                }
                break;
            case LoopN:
                if (problem.B.layout == MatrixLayout::T)
                    eadd(1, state.effB, state.effB, chunkSize * Tb, strategy,
                            state);
                else {
                    Subregister temp = state.ra.alloc_sub<uint32_t>();
                    mulConstant(1, temp, state.inputs.ldb, chunkSize);
                    eadd(1, state.effB, state.effB, temp, strategy, state);
                    state.ra.safeRelease(temp);
                }
                if (problem.C.layout == MatrixLayout::T
                        || problem.C.layout == MatrixLayout::Pc)
                    eadd(1, state.effC, state.effC, chunkSize * Tc, strategy,
                            state);
                else {
                    Subregister temp = state.ra.alloc_sub<uint32_t>();
                    mulConstant(1, temp, state.inputs.ldb, chunkSize);
                    eadd(1, state.effB, state.effB, temp, strategy, state);
                    state.ra.safeRelease(temp);
                }
                break;
        }

        mark(label_next_rem);

        // Handle the remainder recursively.
        {
            GEMMStrategy substrategy = strategy;
            substrategy.remHandling[loop] = RemainderHandling::General;
            substrategy.unroll[loop] -= chunkSize;
            if (!mnRemainderHandling(loop, problem, substrategy, state, func)) {
                discardStream();
                return false;
            }
        }
    } /* end stream */

    appendCurrentStream();

    return true; /* success */
#endif
}

template <HW hw>
template <typename Problem>
bool gemm_kernel_generator_t<hw>::mnJointSplitRemainderHandling(
        Problem &problem, GEMMStrategy &strategy, GEMMState &state,
        bool (gemm_kernel_generator_t<hw>::*func)(
                Problem, GEMMStrategy, GEMMState)) {
    Label label_done, label_remainder;
    bool success = false;

    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];

    pushStream();
    do {
        // Jump to remainder loop if needed:
        //  - if m/n below split thresholds (when enabled)
        //  - if in a remainder kernel.
        bool wgCheck = wgRemCheck(problem, strategy);

        if (strategy.mSplitThresh && strategy.nSplitThresh) {
            cmp(1 | lt | f0[0], null.d(), state.inputs.m,
                    int32_t(strategy.mSplitThresh));
            cmp(1 | lt | f1[0], null.d(), state.inputs.n,
                    int32_t(strategy.nSplitThresh));
            ejmpi(1 | f0[0] | anyv, label_remainder);
        } else if (strategy.mSplitThresh) {
            cmp(1 | lt | f0[0], null.d(), state.inputs.m,
                    int32_t(strategy.mSplitThresh));
            jmpi(1 | f0[0], label_remainder);
        } else if (strategy.nSplitThresh) {
            cmp(1 | lt | f0[0], null.d(), state.inputs.n,
                    int32_t(strategy.nSplitThresh));
            jmpi(1 | f0[0], label_remainder);
        }
        if (wgCheck) {
            cmp(1 | lt | f0[0], null.d(), state.remaindersWG[LoopM],
                    uint16_t(unrollM * strategy.wg[LoopM]));
            cmp(1 | lt | f1[0], null.d(), state.remaindersWG[LoopN],
                    uint16_t(unrollN * strategy.wg[LoopN]));
        } else {
            cmp(1 | lt | f0[0], null.d(), state.remaindersFused[LoopM],
                    uint16_t(unrollM));
            cmp(1 | lt | f1[0], null.d(), state.remaindersFused[LoopN],
                    uint16_t(unrollN));
        }
        ejmpi(1 | f0[0] | anyv, label_remainder);

        // First generate code that ignores remainder handling.
        GEMMStrategy substrategy = strategy;
        substrategy.remHandling[LoopM] = RemainderHandling::Ignore;
        substrategy.remHandling[LoopN] = RemainderHandling::Ignore;

        status << "Generating MN non-remainder kernel." << status_stream::endl;
        if (!(this->*func)(problem, substrategy, state)) {
            status << "Non-remainder kernel failed, aborting."
                   << status_stream::endl;
            break;
        }

        // Return, unless this is part of a larger computation, in which case jump to end.
        if (state.isNested)
            jmpi(1, label_done);
        else
            epilogue(strategy, state);

        mark(label_remainder);

        // Finally, generate remainder handling kernel.
        substrategy = strategy;
        substrategy.remHandling[LoopM] = substrategy.remHandling[LoopN]
                = (wgCheck ? RemainderHandling::General
                           : RemainderHandling::KnownRemainder);
        substrategy.checkAdd32 = false;
        status << "Generating MN general kernel." << status_stream::endl;
        success = (this->*func)(problem, substrategy, state);

        mark(label_done);
    } while (false);

    success ? appendCurrentStream() : discardStream();

    return success;
}

// Handle outer-level m edge cases.
template <HW hw>
bool gemm_kernel_generator_t<hw>::gemmMEdge(
        GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state) {
    if (strategy.jointSplit
            && strategy.remHandling[LoopM] == RemainderHandling::Split
            && strategy.remHandling[LoopN] == RemainderHandling::Split)
        return mnJointSplitRemainderHandling(problem, strategy, state,
                &gemm_kernel_generator_t<hw>::gemmBody);
    else
        return mnRemainderHandling(LoopM, problem, strategy, state,
                &gemm_kernel_generator_t<hw>::gemmNEdge);
}

// Handle outer-level n edge cases.
template <HW hw>
bool gemm_kernel_generator_t<hw>::gemmNEdge(
        GEMMProblem problem, GEMMStrategy strategy, GEMMState state) {
    return mnRemainderHandling(LoopN, problem, strategy, state,
            &gemm_kernel_generator_t<hw>::gemmBody);
}

// Initialize the interface.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmInitInterface(GEMMProblem &problem,
        GEMMStrategy &strategy, GEMMState &state, bool inSK) {
    Subregister localSize[3];
    GRF localID[3];
    Subregister tgids[3]
            = {r0.ud(1), r0.ud(6), r0.ud(7)}; // X, Y, Z threadgroup IDs

    if (strategy.systolic) interface.requireDPAS();
    if (strategy.C.atomic) interface.requireGlobalAtomics();
    if (strategy.barrierFreq > 0) interface.requireBarrier();

    auto slmSize = gemmSLMSize(problem, strategy);
    auto slmPerK = gemmPerKSLMSize(problem, strategy);
    if (slmSize > 0 || slmPerK > 0) {
        status << "SLM usage: " << slmSize / 1024. << 'k';
        if (slmPerK) status << " (" << slmPerK / 1024. << "k per-k)";
        status << status_stream::endl;
        if (!slmPerK) interface.requireSLM(slmSize);
        interface.requireBarrier();
    }

    if (strategy.fixedWG(problem)) {
        auto wgX = strategy.wg[strategy.loopOrder[0]];
        auto wgY = strategy.wg[strategy.loopOrder[1]];
        auto wgZ = strategy.wg[strategy.loopOrder[2]];
        if (strategy.splitCopy) wgY *= 2;
        if (wgZ <= 1)
            interface.requireWorkgroup(strategy.subgroupSize * wgX, wgY, wgZ);
    }
    interface.requireWalkOrder(0, 1, 2);
    interface.requireStatelessWrites(strategy.C.base.isStateless());

    int nb = int(strategy.slmA || strategy.barrierFreq)
                    * strategy.namedBarriers[LoopM]
            + int(strategy.slmB || strategy.barrierFreq)
                    * strategy.namedBarriers[LoopN];
    if (nb) interface.requireBarriers(nb);

    interface.finalize();

    for (int dim = 0; dim < 3; dim++) {
        localID[dim] = interface.getLocalID(dim);
        localSize[dim] = interface.getLocalSize(dim);
    }

    // Get input arguments.
    state.inputs.base = interface.getArgumentIfExists("base");
    auto baseSurface = interface.getArgumentSurfaceIfExists("base");
    if (state.inputs.base.isInvalid()
            && baseSurface == InterfaceHandler::noSurface) {
        state.inputs.A = interface.getArgumentIfExists("A");
        state.inputs.B = interface.getArgumentIfExists("B");
        state.inputs.C[1] = interface.getArgumentIfExists("P");
        state.inputs.surfaceA = interface.getArgumentSurfaceIfExists("A");
        state.inputs.surfaceB = interface.getArgumentSurfaceIfExists("B");
        state.inputs.surfaceC[1] = interface.getArgumentSurfaceIfExists("P");
    } else {
        state.inputs.A = state.inputs.B = state.inputs.base;
        state.inputs.surfaceA = state.inputs.surfaceB = baseSurface;
        if (interface.getArgumentIfExists("offset_P").isValid()) {
            state.inputs.C[1] = state.inputs.base;
            state.inputs.surfaceC[1] = state.inputs.surfaceA;
        }
    }

    state.inputs.C[0] = interface.getArgumentIfExists("C");
    state.inputs.surfaceC[0] = interface.getArgumentSurfaceIfExists("C");
    state.C_count = state.inputs.C[1].isValid() ? 2 : 1;
    if (problem.cOffset != COffset::None) {
        state.inputs.CO = interface.getArgumentIfExists("CO");
        state.inputs.surfaceCO = interface.getArgumentSurfaceIfExists("CO");
    }

    if (problem.abOffset != ABOffset::None) {
        state.inputs.abo = interface.getArgumentIfExists("abo");
        if (state.inputs.abo.isValid()) {
            // A/B offset are two words packed into a single dword argument.
            state.inputs.ao = state.inputs.abo.w(0);
            state.inputs.bo = state.inputs.abo.w(1);
        } else {
            state.inputs.ao = interface.getArgumentIfExists("ao");
            state.inputs.bo = interface.getArgumentIfExists("bo");
        }
    }
    state.inputs.offsetA = interface.getArgumentIfExists("offset_A");
    state.inputs.offsetB = interface.getArgumentIfExists("offset_B");
    state.inputs.offsetC[0] = interface.getArgumentIfExists("offset_C");
    state.inputs.offsetC[1] = interface.getArgumentIfExists("offset_P");
    state.inputs.offsetCO = interface.getArgumentIfExists("offset_CO");
    if (problem.batch == BatchMode::Strided) {
        state.inputs.strideA[0] = interface.getArgumentIfExists("stride_A");
        state.inputs.strideB[0] = interface.getArgumentIfExists("stride_B");
        state.inputs.strideC[0] = interface.getArgumentIfExists("stride_C");
        if (problem.batchDims > 1) {
            state.inputs.strideA[1]
                    = interface.getArgumentIfExists("stride_A1");
            state.inputs.strideB[1]
                    = interface.getArgumentIfExists("stride_B1");
            state.inputs.strideC[1]
                    = interface.getArgumentIfExists("stride_C1");
            state.inputs.batchSize1
                    = interface.getArgumentIfExists("batch_size1");
            state.inputs.recipBatchSize1
                    = interface.getArgumentIfExists("recip_batch_size1");
        }
    } else if (problem.batch == BatchMode::Nonstrided)
        state.inputs.offsetBatch
                = interface.getArgumentIfExists("offset_batch");
    else if (problem.batch == BatchMode::Variable) {
        state.inputs.incr_a_array
                = interface.getArgumentIfExists("incr_a_array");
        state.inputs.incr_b_array
                = interface.getArgumentIfExists("incr_b_array");
    }
    state.inputs.lda = interface.getArgumentIfExists("lda");
    state.inputs.ldb = interface.getArgumentIfExists("ldb");
    state.inputs.ldc[0] = interface.getArgumentIfExists("ldc");
    state.inputs.ldc[1] = interface.getArgumentIfExists("ldp");
    state.inputs.m = interface.getArgumentIfExists("m");
    state.inputs.n = interface.getArgumentIfExists("n");
    state.inputs.k = interface.getArgumentIfExists("k");
    state.inputs.k0 = interface.getArgumentIfExists("k0");
    state.inputs.alpha_real = interface.getArgumentIfExists("alpha_real");
    state.inputs.alpha_imag = interface.getArgumentIfExists("alpha_imag");
    state.inputs.beta_real = interface.getArgumentIfExists("beta_real");
    state.inputs.beta_imag = interface.getArgumentIfExists("beta_imag");
    if (problem.batch == BatchMode::Variable) {
        state.inputs.alpha_array = interface.getArgumentIfExists("alpha_array");
        state.inputs.beta_array = interface.getArgumentIfExists("beta_array");
        state.inputs.incr_alpha = interface.getArgumentIfExists("incr_alpha");
        state.inputs.incr_beta = interface.getArgumentIfExists("incr_beta");
    }
    state.inputs.diagA = interface.getArgumentIfExists("diag_A");
    state.inputs.diagB = interface.getArgumentIfExists("diag_B");
    state.inputs.diagC = interface.getArgumentIfExists("diag_C");
    state.inputs.flags = interface.getArgumentIfExists("flags");

    if (strategy.linearOrder()) {
        state.inputs.groupCountM = interface.getArgument("group_count_m");
        state.inputs.groupCountN = interface.getArgument("group_count_n");
    }
    if (strategy.hilbertOrder) {
        state.inputs.hilbertVD = interface.getArgumentIfExists("hilbert_vd");
        state.inputs.hilbertUVDRecip
                = interface.getArgumentIfExists("hilbert_uvd_recip");
        state.inputs.hilbertBail
                = interface.getArgumentIfExists("hilbert_bail");
    } else if (strategy.boustrophedon) {
        state.inputs.bslice = interface.getArgument("bslice");
        state.inputs.bthresh = interface.getArgument("bthresh");
    }
    if (strategy.persistent) {
        state.inputs.groupCountMN
                = interface.getArgumentIfExists("group_count");
        state.inputs.groupStride = interface.getArgument("group_stride");
    }

    Subregister tgids_reordered[3];
    GRF lids_reordered[3];
    Subregister lszs_reordered[3];

    for (int l = 0; l < 3; l++) {
        int i = static_cast<int>(strategy.loopOrder[l]);
        tgids_reordered[i] = tgids[l];
        lids_reordered[i] = localID[l];
        lszs_reordered[i] = localSize[l];
    }
    state.inputs.groupIDM = tgids_reordered[0];
    state.inputs.groupIDN = tgids_reordered[1];
    state.inputs.groupIDK = tgids_reordered[2];
    state.inputs.localIDM = lids_reordered[0];
    state.inputs.localIDN = lids_reordered[1];
    state.inputs.localIDK = lids_reordered[2];
    state.inputs.localSizeM = lszs_reordered[0];
    state.inputs.localSizeN = lszs_reordered[1];
    state.inputs.localSizeK = lszs_reordered[2];

    if (strategy.linearOrder()) {
        state.inputs.groupIDMN = tgids[0];
        state.inputs.groupIDM = invalid;
        state.inputs.groupIDN = invalid;
    }

    // Downgrade offsets to 32 bits for non-A64 accesses.
    if (strategy.A.base.getModel() != ModelA64)
        state.inputs.offsetA = state.inputs.offsetA.d();
    if (strategy.B.base.getModel() != ModelA64)
        state.inputs.offsetB = state.inputs.offsetB.d();
    if (strategy.C.base.getModel() != ModelA64)
        for (int q = 0; q < state.C_count; q++)
            state.inputs.offsetC[q] = state.inputs.offsetC[q].d();
    if (problem.cOffset != COffset::None
            && strategy.CO.base.getModel() != ModelA64)
        state.inputs.offsetCO = state.inputs.offsetCO.d();

    // For now, reinterpret m/n/k/ld/diag variables to 32-bit if they are 64-bit.
    state.inputs.m = state.inputs.m.d();
    state.inputs.n = state.inputs.n.d();
    state.inputs.k = state.inputs.k.d();
    state.inputs.lda = state.inputs.lda.ud();
    state.inputs.ldb = state.inputs.ldb.ud();
    for (int q = 0; q < state.C_count; q++)
        state.inputs.ldc[q] = state.inputs.ldc[q].ud();
    state.inputs.diagA = state.inputs.diagA.d();
    state.inputs.diagB = state.inputs.diagB.d();
    state.inputs.diagC = state.inputs.diagC.d();

    // Claim registers.
    for (int i = 0; i < 4; i++)
        state.ra.claim(r0.uq(i));

    if (strategy.A.base.isStateless()) state.ra.claim(state.inputs.A);
    if (strategy.B.base.isStateless()) state.ra.claim(state.inputs.B);
    if (strategy.C.base.isStateless())
        for (int q = 0; q < state.C_count; q++)
            state.ra.claim(state.inputs.C[q]);

    if (problem.abOffset != ABOffset::None) {
        state.ra.claim(state.inputs.ao);
        state.ra.claim(state.inputs.bo);
    }

    if (problem.cOffset != COffset::None) {
        if (strategy.CO.base.isStateless()) state.ra.claim(state.inputs.CO);
        state.ra.claim(state.inputs.offsetCO);
    }

    state.ra.claim(state.inputs.offsetA);
    state.ra.claim(state.inputs.offsetB);
    for (int q = 0; q < state.C_count; q++)
        state.ra.claim(state.inputs.offsetC[q]);
    state.ra.claim(state.inputs.lda);
    state.ra.claim(state.inputs.ldb);
    for (int q = 0; q < state.C_count; q++)
        state.ra.claim(state.inputs.ldc[q]);
    state.ra.claim(state.inputs.m);
    state.ra.claim(state.inputs.n);
    state.ra.claim(state.inputs.k);
    if (strategy.kParallel || strategy.kParallelLocal)
        state.ra.claim(state.inputs.k0);

    if (!problem.alpha_real.fixed()) state.ra.claim(state.inputs.alpha_real);
    if (!problem.beta_real.fixed()) state.ra.claim(state.inputs.beta_real);

    if (!inSK) {
        state.ra.claim(state.inputs.localIDM);
        state.ra.claim(state.inputs.localIDN);
        if (!strategy.fixedWG(problem)) {
            state.ra.claim(state.inputs.localSizeM);
            state.ra.claim(state.inputs.localSizeN);
        } else
            state.inputs.localSizeM = state.inputs.localSizeN = invalid;
        if (strategy.kParallel || strategy.kParallelLocal) {
            state.ra.claim(state.inputs.localIDK);
            state.ra.claim(state.inputs.localSizeK);
        }
    }

    if (state.inputs.flags.isValid()) state.ra.claim(state.inputs.flags);

    if (problem.batch == BatchMode::Strided) {
        for (int i = 0; i < problem.batchDims; i++) {
            state.ra.claim(state.inputs.strideA[i]);
            state.ra.claim(state.inputs.strideB[i]);
            state.ra.claim(state.inputs.strideC[i]);
        }
        if (problem.batchDims > 1) {
            state.ra.claim(state.inputs.batchSize1);
            state.ra.claim(state.inputs.recipBatchSize1);
        }
        state.ra.claim(state.inputs.groupIDK);
    } else if (problem.batch == BatchMode::Nonstrided) {
        state.ra.claim(state.inputs.offsetBatch);
        state.ra.claim(state.inputs.groupIDK);
    } else if (problem.batch == BatchMode::Variable) {
        state.ra.claim(state.inputs.incr_a_array);
        state.ra.claim(state.inputs.incr_b_array);
        state.ra.claim(state.inputs.alpha_array);
        state.ra.claim(state.inputs.beta_array);
        state.ra.claim(state.inputs.incr_alpha);
        state.ra.claim(state.inputs.incr_beta);
        state.ra.claim(state.inputs.groupIDK);
    }

    if (strategy.linearOrder()) {
        state.ra.claim(state.inputs.groupCountM);
        state.ra.claim(state.inputs.groupCountN);
    }

    if (strategy.hilbertOrder) {
        {
            state.ra.claim(state.inputs.hilbertVD);
            state.ra.claim(state.inputs.hilbertUVDRecip);
        }
        state.ra.claim(state.inputs.hilbertBail);
    }

    if (strategy.boustrophedon) {
        state.ra.claim(state.inputs.bslice);
        state.ra.claim(state.inputs.bthresh);
    }

    if (strategy.persistent) {
        state.ra.claim(state.inputs.groupStride);
        if (state.inputs.groupCountMN.isValid())
            state.ra.claim(state.inputs.groupCountMN);
    }
}

// Return amount of SLM needed by a GEMM kernel.
template <HW hw>
size_t gemm_kernel_generator_t<hw>::gemmSLMSize(
        const GEMMProblem &problem, const GEMMStrategy &strategy) {
    // Space needed by SLM copies.
    size_t slmSize
            = strategy.slmABufSize(problem) + strategy.slmBBufSize(problem);

    // Space needed for row/column sum reduction/sharing.
    if (problem.abOffset == ABOffset::Calc) {
        slmSize = std::max<size_t>(slmSize,
                (strategy.unroll[LoopM] * strategy.wg[LoopM]
                        + strategy.unroll[LoopN] * strategy.wg[LoopN])
                        * problem.Tc);
    }

    return slmSize;
}

// Return amount of per-k SLM needed by a GEMM kernel.
template <HW hw>
size_t gemm_kernel_generator_t<hw>::gemmPerKSLMSize(
        const GEMMProblem &problem, const GEMMStrategy &strategy) {
    size_t slmSize = 0;

    // Space needed for local k reduction (as much as possible).
    if (strategy.kParallelLocal) {
        // Calculate max SLM usage that doesn't reduce thread count.
        int mnThreads = strategy.wg[LoopM] * strategy.wg[LoopN];
        if (mnThreads <= 0) stub();
        int concurrentK = std::max(
                1, threadsPerEU(hw, strategy) * eusPerSubslice(hw) / mnThreads);
        slmSize = rounddown_pow2(slmCapacity(hw) / concurrentK);
    }

    return slmSize;
}

// Initialize the state structure.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmInitState(GEMMProblem &problem,
        GEMMStrategy &strategy, GEMMState &state, bool inSK) {
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc;

    if (!state.fusedGEMM.active) {
        initState(problem, strategy, state);
        gemmInitInterface(problem, strategy, state, inSK);
        state.isNested |= strategy.fused;
        state.isNested |= strategy.persistent;
    }

    state.effA = strategy.A.base.isStateless() ? state.inputs.A
                                               : state.inputs.offsetA.d();
    state.effB = strategy.B.base.isStateless() ? state.inputs.B
                                               : state.inputs.offsetB.d();
    for (int q = 0; q < state.C_count; q++) {
        state.effC[q] = strategy.C.base.isStateless()
                ? state.inputs.C[q]
                : state.inputs.offsetC[q].d();
    }
    if (problem.cOffset != COffset::None) {
        state.effCO = strategy.CO.base.isStateless()
                ? state.inputs.CO
                : state.inputs.offsetCO.d();
    }

    if (!problem.alpha_real.fixed())
        problem.alpha_real = state.inputs.alpha_real;
    if (!problem.beta_real.fixed()) problem.beta_real = state.inputs.beta_real;

    state.offsetA = state.inputs.offsetA;
    state.offsetB = state.inputs.offsetB;
    for (int q = 0; q < state.C_count; q++)
        state.offsetC[q] = state.inputs.offsetC[q];
    state.offsetCO = state.inputs.offsetCO;

    state.flagAP = state.raVFlag.alloc();

    state.allocEmulate64Temp(strategy.emulate);

    state.Ta_load = problem.Ta_ext;
    state.Tb_load = problem.Tb_ext;

    state.Tacc = problem.Tc;
    state.copyC = (problem.Tc != problem.Tc_ext)
            || (!strategy.altCRemainder && (Tc.size() < 4))
            || strategy.forceCopyC;

    state.broadcast = strategy.doubleWA;

    bool cColMajor = isRegisterColMajor(problem.Tc, problem.C, strategy.C);
    state.broadcast |= (Tc == Type::f32 && (cColMajor ? Tb : Ta) == Type::bf16);

    state.Cext_strategy = strategy.C;
    state.Cext_strategy.tileR = state.Cext_strategy.tileC = 0;

    state.lidM = state.inputs.localIDM[0];
    state.lidN = state.inputs.localIDN[0];
    if (strategy.kParallel || strategy.kParallelLocal)
        state.lidK = state.inputs.localIDK[0];

    state.diagC = state.inputs.diagC;
    state.k = state.inputs.k;

    state.lda = state.inputs.lda;
    state.ldb = state.inputs.ldb;
}

// Offset A pointer in k dimension by a constant value.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmOffsetAk(int h,
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
    auto Ta_ext = problem.Ta_ext;
    if (strategy.A.address2D) stub();
    if (h) switch (problem.A.layout) {
            case MatrixLayout::T:
                eadd(1, state.effA, state.effA, h * Ta_ext, strategy, state);
                break;
            case MatrixLayout::Pc:
                eadd(1, state.effA, state.effA, h * problem.A.packSize * Ta_ext,
                        strategy, state);
                break;
            case MatrixLayout::N:
                emad(1, state.effA, state.effA, state.inputs.lda,
                        Immediate::w(h), strategy, state);
                break;
            default: stub();
        }
}

// Offset B pointer in k dimension by a constant value.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmOffsetBk(int h,
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
    auto Tb_ext = problem.Tb_ext;
    if (strategy.B.address2D) stub();
    if (h) switch (problem.B.layout) {
            case MatrixLayout::N:
                eadd(1, state.effB, state.effB, h * Tb_ext, strategy, state);
                break;
            case MatrixLayout::Pr:
                eadd(1, state.effB, state.effB, h * problem.B.packSize * Tb_ext,
                        strategy, state);
                break;
            case MatrixLayout::T:
                emad(1, state.effB, state.effB, state.inputs.ldb,
                        Immediate::w(h), strategy, state);
                break;
            default: stub();
        }
}

// Adjust A, B, C to start at (i0, j0).
//  initial is true to adjust offset_{A,B,C}, false to adjust A,B,C pointers.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmOffsetABC(bool initial, Subregister i0,
        Subregister j0, Subregister h0, const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state, bool doA, bool doB,
        bool doC) {
    auto Ta_ext = problem.Ta_ext, Tb_ext = problem.Tb_ext,
         Tc_ext = problem.Tc_ext, Tco = problem.Tco;
    auto &offsetA = initial ? state.offsetA : state.effA;
    auto &offsetB = initial ? state.offsetB : state.effB;
    auto &offsetC0 = initial ? state.offsetC[0] : state.effC[0];
    auto &offsetAp = initial ? state.offsetAp : state.effAp;
    auto &offsetBp = initial ? state.offsetBp : state.effBp;
    auto &offsetCp = initial ? state.offsetCp : state.effCp;
    auto offsetCO = initial ? state.offsetCO : state.effCO;
    bool doCO = doC && (problem.cOffset != COffset::None);

    Subregister tempQ0 = state.ra.alloc_sub<int64_t>(
            getHint(HintType::TempComp0, strategy));
    Subregister tempQ1 = state.ra.alloc_sub<int64_t>(
            getHint(HintType::TempComp1, strategy));

    bool a2D = strategy.A.address2D;
    bool b2D = strategy.B.address2D;
    bool c2D = strategy.C.address2D;
    bool ap2D = strategy.prefetchA ? strategy.A_prefetch.address2D : a2D;
    bool bp2D = strategy.prefetchB ? strategy.B_prefetch.address2D : b2D;
    bool cp2D = strategy.prefetchC ? strategy.C_prefetch.address2D : c2D;

    if (a2D && ap2D) doA = false;
    if (b2D && bp2D) doB = false;
    if (c2D && cp2D) doC = false;

    if (doA && (a2D != ap2D)) {
        if (!initial) stub();
        if (offsetAp.isInvalid()) {
            offsetAp = state.ra.alloc_sub(
                    offsetA.getType(), getHint(HintType::LongTerm, strategy));
            emov(1, state.offsetAp, offsetA, strategy, state);
        } else if (a2D && !ap2D)
            std::swap(offsetA, offsetAp);
    }
    if (doB && (b2D != bp2D)) {
        if (!initial) stub();
        if (offsetBp.isInvalid()) {
            offsetBp = state.ra.alloc_sub(
                    offsetB.getType(), getHint(HintType::LongTerm, strategy));
            emov(1, offsetBp, offsetB, strategy, state);
        } else if (b2D && !bp2D)
            std::swap(offsetB, offsetBp);
    }
    if (doC && (c2D != cp2D)) {
        if (!initial) stub();
        if (offsetCp.isInvalid()) {
            offsetCp = state.ra.alloc_sub(
                    offsetC0.getType(), getHint(HintType::LongTerm, strategy));
            emov(1, offsetCp, offsetC0, strategy, state);
        } else if (c2D && !cp2D)
            std::swap(offsetC0, offsetCp);
    }

    // To do: interleave code.
    // A += i0 (N) i0 * lda (T, Pc)
    // B += j0 * ldb (N, Pr) j0 (T)
    // C += i0 + j0 * ldc (N, Pr) j0 + i0 * ldc (T, Pc)
    // CO += i0 (row offsets) j0 (col offsets)
    if (doA && i0.isValid()) {
        if (problem.A.layout == MatrixLayout::Nontranspose)
            emad(1, offsetA, offsetA, i0, Ta_ext.size(), strategy, state);
        else {
            emul(1, tempQ1, i0, state.inputs.lda, strategy, state);
            eadd(1, offsetA, offsetA, tempQ1.reinterpret(0, offsetA.getType()),
                    strategy, state);
        }
    }

    if (doB && j0.isValid()) {
        if (problem.B.layout == MatrixLayout::Transpose)
            emad(1, offsetB, offsetB, j0, Tb_ext.size(), strategy, state);
        else {
            emul(1, tempQ0, j0, state.inputs.ldb, strategy, state);
            eadd(1, offsetB, offsetB, tempQ0.reinterpret(0, offsetB.getType()),
                    strategy, state);
        }
    }

    FlagRegister flagCOR, flagCOC;
    if (doCO) {
        flagCOR = state.raVFlag.alloc();
        flagCOC = state.raVFlag.alloc();
        and_(1 | nz | flagCOC, null.ud(), state.inputs.flags, FlagCOColumn);
        and_(1 | nz | flagCOR, null.ud(), state.inputs.flags, FlagCORow);
    }
    if (doC) {
        for (int q = 0; q < state.C_count; q++) {
            auto offsetC = initial ? state.offsetC[q] : state.effC[q];

            Subregister x, y;
            int xstride = Tc_ext.size();
            switch (problem.C.layout) {
                case MatrixLayout::Pr:
                    xstride *= strategy.unroll[LoopN]; /* fall through */
                case MatrixLayout::N:
                    x = i0;
                    y = j0;
                    break;
                case MatrixLayout::Pc:
                    xstride *= strategy.unroll[LoopM]; /* fall through */
                case MatrixLayout::T:
                    x = j0;
                    y = i0;
                    break;
            }
            emad(1, offsetC, offsetC, x, xstride, strategy, state);
            emul(1, tempQ0, y, state.inputs.ldc[q], strategy, state);
            eadd(1, offsetC, offsetC, tempQ0.reinterpret(0, offsetC.getType()),
                    strategy, state); // Gen12: Use add3.
        }
    }
    if (doCO) {
        emad(1 | flagCOC, offsetCO, offsetCO, j0, Tco.size(), strategy, state);
        emad(1 | flagCOR, offsetCO, offsetCO, i0, Tco.size(), strategy, state);
        state.raVFlag.safeRelease(flagCOR);
        state.raVFlag.safeRelease(flagCOC);
    }

    // When k blocking (or certain triangular source kernels)
    //   A += h0 * lda (N) h0 (T) h0 * mb (Pc)
    //   B += h0 (N) h0 * ldb (T) h0 * nb (Pr)
    if (!h0.isInvalid()) {
        if (!initial) stub();
        if (doA) switch (problem.A.layout) {
                case MatrixLayout::Nontranspose:
                    emul(1, tempQ1, h0, state.inputs.lda, strategy, state);
                    eadd(1, offsetA, offsetA,
                            tempQ1.reinterpret(0, offsetA.getType()), strategy,
                            state);
                    break;
                case MatrixLayout::Transpose:
                    emad(1, offsetA, offsetA, h0, Ta_ext.size(), strategy,
                            state);
                    break;
                case MatrixLayout::PackedColumns:
                    emad(1, offsetA, offsetA, h0,
                            strategy.unroll[LoopM] * Ta_ext, strategy, state);
                    break;
                default: stub();
            }
        if (doB) switch (problem.B.layout) {
                case MatrixLayout::Nontranspose:
                    emad(1, offsetB, offsetB, h0, Tb_ext.size(), strategy,
                            state);
                    break;
                case MatrixLayout::Transpose:
                    emul(1, tempQ0, h0, state.inputs.ldb, strategy, state);
                    eadd(1, offsetB, offsetB,
                            tempQ0.reinterpret(0, offsetB.getType()), strategy,
                            state);
                    break;
                case MatrixLayout::PackedRows:
                    emad(1, offsetB, offsetB, h0,
                            strategy.unroll[LoopN] * Tb_ext, strategy, state);
                    break;
                default: stub();
            }
    }

    state.ra.safeRelease(tempQ0);
    state.ra.safeRelease(tempQ1);

    if (doA && a2D && !ap2D) std::swap(offsetA, offsetAp);
    if (doB && b2D && !bp2D) std::swap(offsetB, offsetBp);
    if (doC && c2D && !cp2D) std::swap(offsetC0, offsetCp);
}

template <ngen::HW hw>
void gemm_kernel_generator_t<hw>::gemmOffsetBatchABC(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state) {
    auto Ts = problem.Ts;

    // Non-strided batch support.
    if (problem.batch == BatchMode::Nonstrided) {
        auto temp = state.ra.alloc().uq();
        auto boffset = state.ra.alloc_sub<uint32_t>();

        add(1, boffset, state.inputs.offsetBatch, state.inputs.groupIDK);
        mov(1, state.flagAP, 0x7);
        shl(1, boffset, boffset, uint16_t(3));

        eadd(1, temp[0], state.inputs.A, boffset, strategy, state);
        eadd(1, temp[1], state.inputs.B, boffset, strategy, state);
        eadd(1, temp[2], state.inputs.C[0], boffset, strategy, state);

        load(4 | state.flagAP, temp, scattered_qword(1), strategy.A.base, temp);

        emov(1, state.effA, temp[0], strategy, state);
        emov(1, state.effB, temp[1], strategy, state);
        emov(1, state.effC[0], temp[2], strategy, state);

        state.ra.safeRelease(temp);
        state.ra.safeRelease(boffset);
        if (!strategy.persistent)
            state.ra.safeRelease(state.inputs.offsetBatch);
    }

    // Strided batch support.
    if (problem.batch == BatchMode::Strided) {
        for (int b = 0; b < problem.batchDims; b++) {
            emul(1, state.inputs.strideA[b], state.inputs.strideA[b],
                    state.batchID[b], strategy, state);
            emul(1, state.inputs.strideB[b], state.inputs.strideB[b],
                    state.batchID[b], strategy, state);
            emul(1, state.inputs.strideC[b], state.inputs.strideC[b],
                    state.batchID[b], strategy, state);
        }

        for (int b = 0; b < problem.batchDims; b++) {
            eadd(1, state.offsetA, state.offsetA, state.inputs.strideA[b],
                    strategy, state);
            if (!strategy.persistent)
                state.ra.safeRelease(state.inputs.strideA[b]);
        }

        for (int b = 0; b < problem.batchDims; b++) {
            eadd(1, state.offsetB, state.offsetB, state.inputs.strideB[b],
                    strategy, state);
            if (!strategy.persistent)
                state.ra.safeRelease(state.inputs.strideB[b]);
        }

        for (int q = 0; q < state.C_count; q++) {
            auto offsetC = state.offsetC[q];
            for (int b = 0; b < problem.batchDims; b++)
                eadd(1, offsetC, offsetC, state.inputs.strideC[b], strategy,
                        state);
        }

        if (!strategy.persistent)
            for (int b = 0; b < problem.batchDims; b++)
                state.ra.safeRelease(state.inputs.strideC[b]);
    }

    // Non-strided variable batch support.
    if (problem.batch == BatchMode::Variable) {
        auto tempA = state.ra.alloc().uq();
        auto tempB = state.ra.alloc().uq();
        auto tempC = state.ra.alloc().uq();
        auto tempIDK = state.ra.alloc().ud();
        auto offset_scalar = state.ra.alloc().uq();
        auto offset_pointer = state.ra.alloc().uq();
        auto tempAlphaReal = state.ra.alloc().uq();
        auto tempAlphaImag = state.ra.alloc().uq();
        auto tempBetaReal = state.ra.alloc().uq();
        auto tempBetaImag = state.ra.alloc().uq();

        eshl(1, tempIDK, state.inputs.groupIDK, uint16_t(log2(Ts.size())),
                strategy, state);

        // load and set alpha
        emul(1, offset_scalar, tempIDK, state.inputs.incr_alpha.uw(), strategy,
                state);
        eadd(1, tempAlphaReal[0], state.inputs.alpha_array, offset_scalar,
                strategy, state);
        if (Ts.isComplex()) {
            eadd(1, tempAlphaImag[0], tempAlphaReal[0], Ts.real().size(),
                    strategy, state);
        }
        if (Ts.real().size() == 4) {
            load(1, tempAlphaReal, scattered_dword(1), A64, tempAlphaReal);
            if (Ts.isComplex()) {
                load(1, tempAlphaImag, scattered_dword(1), A64, tempAlphaImag);
            }
        } else if (Ts.real().size() == 2) {
            load(1, tempAlphaReal, scattered_byte(2), A64, tempAlphaReal);
            if (Ts.isComplex()) {
                load(1, tempAlphaImag, scattered_byte(2), A64, tempAlphaImag);
            }
        } else {
            load(1, tempAlphaReal, scattered_qword(1), A64, tempAlphaReal);
            if (Ts.isComplex()) {
                load(1, tempAlphaImag, scattered_qword(1), A64, tempAlphaImag);
            }
        }
        mov(1, state.inputs.alpha_real, tempAlphaReal.sub(0, Ts.real().ngen()));
        if (Ts.isComplex())
            mov(1, state.inputs.alpha_imag,
                    tempAlphaImag.sub(0, Ts.real().ngen()));
        // end load and set alpha

        // load and set beta
        emul(1, offset_scalar, tempIDK, state.inputs.incr_beta.uw(), strategy,
                state);
        eadd(1, tempBetaReal[0], state.inputs.beta_array, offset_scalar,
                strategy, state);
        if (Ts.isComplex()) {
            eadd(1, tempBetaImag[0], tempBetaReal[0], Ts.real().size(),
                    strategy, state);
        }
        if (Ts.real().size() == 4) {
            load(1, tempBetaReal, scattered_dword(1), A64, tempBetaReal);
            if (Ts.isComplex()) {
                load(1, tempBetaImag, scattered_dword(1), A64, tempBetaImag);
            }
        } else if (Ts.real().size() == 2) {
            load(1, tempBetaReal, scattered_byte(2), A64, tempBetaReal);
            if (Ts.isComplex()) {
                load(1, tempBetaImag, scattered_byte(2), A64, tempBetaImag);
            }
        } else {
            load(1, tempBetaReal, scattered_qword(1), A64, tempBetaReal);
            if (Ts.isComplex()) {
                load(1, tempBetaImag, scattered_qword(1), A64, tempBetaImag);
            }
        }
        mov(1, state.inputs.beta_real, tempBetaReal.sub(0, Ts.real().ngen()));
        if (Ts.isComplex())
            mov(1, state.inputs.beta_imag,
                    tempBetaImag.sub(0, Ts.real().ngen()));
        // end load and set beta

        eshl(1, tempIDK, state.inputs.groupIDK, uint16_t(3), strategy, state);
        emul(1, offset_pointer, tempIDK, state.inputs.incr_a_array.uw(),
                strategy, state);
        eadd(1, tempA[0], state.inputs.A, offset_pointer, strategy, state);
        load(1, tempA, scattered_qword(1), strategy.A.base, tempA);

        emul(1, offset_pointer, tempIDK, state.inputs.incr_b_array.uw(),
                strategy, state);
        eadd(1, tempB[0], state.inputs.B, offset_pointer, strategy, state);
        load(1, tempB, scattered_qword(1), strategy.B.base, tempB);

        eadd(1, tempC[0], state.inputs.C[0], tempIDK, strategy, state);
        load(1, tempC, scattered_qword(1), strategy.C.base, tempC);

        emov(1, state.effA, tempA, strategy, state);
        emov(1, state.effB, tempB, strategy, state);
        emov(1, state.effC[0], tempC, strategy, state);

        state.ra.safeRelease(tempA);
        state.ra.safeRelease(tempB);
        state.ra.safeRelease(tempC);
        state.ra.safeRelease(tempIDK);
        state.ra.safeRelease(offset_scalar);
        state.ra.safeRelease(offset_pointer);
        state.ra.safeRelease(tempAlphaReal);
        state.ra.safeRelease(tempAlphaImag);
        state.ra.safeRelease(tempBetaReal);
        state.ra.safeRelease(tempBetaImag);
        if (!strategy.persistent) {
            state.ra.safeRelease(state.inputs.incr_a_array);
            state.ra.safeRelease(state.inputs.incr_b_array);
            state.ra.safeRelease(state.inputs.incr_alpha);
            state.ra.safeRelease(state.inputs.incr_beta);
            state.ra.safeRelease(state.inputs.alpha_array);
            state.ra.safeRelease(state.inputs.beta_array);
        }
    }
}

// Prepare for persistent GEMM by folding offsets into A/B/C pointers (if stateless),
//  or saving offsets (if stateful)
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmFoldOffsets(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state) {
    auto foldOrSave
            = [&](const MatrixAddressingStrategy &sX, Subregister &inputX,
                      Subregister &offsetX, const Subregister &inputOffsetX,
                      Subregister &saveOffsetX, bool newInput = false) {
                  if (sX.base.isStateless()) {
                      auto oldInputX = inputX;
                      if (newInput)
                          inputX = state.ra.alloc_sub(DataType::uq,
                                  getHint(HintType::LongTerm, strategy));
                      eadd(1, inputX, oldInputX, offsetX, strategy, state);
                      if (getBytes(offsetX.getType()) < 8) {
                          state.ra.safeRelease(offsetX);
                          offsetX = state.ra.alloc_sub(DataType::uq,
                                  getHint(HintType::LongTerm, strategy));
                      }
                      emov(1, offsetX, 0, strategy, state);
                  } else {
                      offsetX = state.ra.alloc_sub(offsetX.getType(),
                              getHint(HintType::LongTerm, strategy));
                      mov(1, offsetX, inputOffsetX);
                  }
                  saveOffsetX = offsetX;
              };

    bool deduplicateAB = (state.inputs.A == state.inputs.B);

    foldOrSave(strategy.A, state.inputs.A, state.offsetA, state.inputs.offsetA,
            state.saveOffsetA, deduplicateAB);
    foldOrSave(strategy.B, state.inputs.B, state.offsetB, state.inputs.offsetB,
            state.saveOffsetB);
    for (int q = 0; q < state.C_count; q++)
        foldOrSave(strategy.C, state.inputs.C[q], state.offsetC[q],
                state.inputs.offsetC[q],
                state.saveOffsetC[q]); // todo init for hpl
    if (problem.cOffset != COffset::None)
        foldOrSave(strategy.CO, state.inputs.CO, state.offsetCO,
                state.inputs.offsetCO, state.saveOffsetCO);

    if (deduplicateAB) state.effA = state.inputs.A;
}

// Restore input offsets from saved copies, for persistent GEMM.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmRestoreOffsets(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state) {
    auto zeroOrRestore = [&](const MatrixAddressingStrategy &sX,
                                 const Subregister &offsetX,
                                 const Subregister &inputOffsetX) {
        if (sX.base.isStateless())
            emov(1, offsetX, 0, strategy, state);
        else
            mov(1, offsetX, inputOffsetX);
    };

    zeroOrRestore(strategy.A, state.saveOffsetA, state.inputs.offsetA);
    zeroOrRestore(strategy.B, state.saveOffsetB, state.inputs.offsetB);
    for (int q = 0; q < state.C_count; q++)
        zeroOrRestore(
                strategy.C, state.saveOffsetC[q], state.inputs.offsetC[q]);
    if (problem.cOffset != COffset::None)
        zeroOrRestore(strategy.CO, state.saveOffsetCO, state.inputs.offsetCO);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::gemmSetupABC(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state) {
    if (strategy.persistent) {
        state.effA = state.offsetA;
        state.effB = state.offsetB;
        for (int q = 0; q < state.C_count; q++)
            state.effC[q] = state.offsetC[q];
        state.effCO = state.offsetCO;
    }

    // Add offsets to A, B, C base pointers for stateless accesses.
    if (strategy.C.base.isStateless()) {
        for (int q = 0; q < state.C_count; q++) {
            auto Csrc = state.inputs.C[q];
            if ((q > 0) && strategy.C.base.isStateless()
                    && state.inputs.base.isValid())
                state.effC[q] = state.inputs.C[q]
                        = state.ra.alloc_sub<uint64_t>(
                                getHint(HintType::LongTerm, strategy));

            eadd(1, state.effC[q], Csrc, state.offsetC[q], strategy, state);
            if (strategy.persistent)
                state.offsetC[q] = invalid;
            else
                state.ra.safeRelease(state.offsetC[q]);
        }
    }

    if ((problem.cOffset != COffset::None) && strategy.CO.base.isStateless()) {
        eadd(1, state.effCO, state.inputs.CO, state.offsetCO, strategy, state);
        if (strategy.persistent)
            state.offsetCO = invalid;
        else
            state.ra.safeRelease(state.offsetCO);
    }

    if (state.offsetAp.isValid()) {
        if (strategy.A.base.isStateless()) {
            state.effAp = state.ra.alloc_sub<uint64_t>(
                    getHint(HintType::LongTerm, strategy));
            eadd(1, state.effAp, state.inputs.A, state.offsetAp, strategy,
                    state);
            state.ra.safeRelease(state.offsetAp);
        } else
            state.effAp = state.offsetAp;
    }

    if (state.offsetBp.isValid()) {
        if (strategy.B.base.isStateless()) {
            state.effBp = state.ra.alloc_sub<uint64_t>(
                    getHint(HintType::LongTerm, strategy));
            eadd(1, state.effBp, state.inputs.B, state.offsetBp, strategy,
                    state);
            state.ra.safeRelease(state.offsetBp);
        } else
            state.effBp = state.offsetBp;
    }

    if (state.offsetCp.isValid()) {
        if (strategy.C.base.isStateless()) {
            state.effCp = state.ra.alloc_sub<uint64_t>(
                    getHint(HintType::LongTerm, strategy));
            eadd(1, state.effCp, state.inputs.C[0], state.offsetCp, strategy,
                    state);
            state.ra.safeRelease(state.offsetCp);
        } else
            state.effCp = state.offsetCp;
    }

    if (strategy.A.base.isStateless()) {
        auto Asrc = state.inputs.A;
        if (strategy.B.base.isStateless() && (state.effA == state.effB))
            state.effA = state.inputs.A = state.ra.alloc_sub<uint64_t>(
                    getHint(HintType::LongTerm, strategy));

        eadd(1, state.effA, Asrc, state.offsetA, strategy, state);
        if (strategy.persistent)
            state.offsetA = invalid;
        else
            state.ra.safeRelease(state.offsetA);
    }

    if (strategy.B.base.isStateless()) {
        eadd(1, state.effB, state.inputs.B, state.offsetB, strategy, state);
        if (strategy.persistent)
            state.offsetB = invalid;
        else
            state.ra.safeRelease(state.offsetB);
    }

    if (strategy.prefetchA && state.effAp.isInvalid()) state.effAp = state.effA;
    if (strategy.prefetchB && state.effBp.isInvalid()) state.effBp = state.effB;
    if (strategy.prefetchC && state.effCp.isInvalid())
        state.effCp = state.effC[0];
}

// Get (possibly multidimensional) batch IDs.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmGetBatchIDs(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state) {
    switch (problem.batchDims) {
        case 0: break;
        case 1: state.batchID[0] = state.inputs.groupIDK; break;
        case 2: {
            state.batchID[0] = state.ra.alloc_sub<uint32_t>();
            state.batchID[1] = state.ra.alloc_sub<uint32_t>();
            divDown(state.batchID[1], state.inputs.groupIDK,
                    state.inputs.batchSize1, state.inputs.recipBatchSize1,
                    state.flagAP, strategy, state);
            emul(1, state.batchID[0], state.batchID[1], state.inputs.batchSize1,
                    strategy, state);
            add(1, state.batchID[0], -state.batchID[0], state.inputs.groupIDK);
            if (!strategy.persistent) {
                state.ra.safeRelease(state.inputs.batchSize1);
                state.ra.safeRelease(state.inputs.recipBatchSize1);
            }
            break;
        }
        default: stub();
    }
}

template <HW hw>
void gemm_kernel_generator_t<hw>::gemmReleaseBatchIDs(
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
    if (problem.batch != BatchMode::Strided) return;
    if (problem.batchDims == 1 && state.r0_info == r0) return;
    for (int b = 0; b < problem.batchDims; b++)
        state.ra.safeRelease(state.batchID[b]);
}

// Convert linear index to 2D index in a Hilbert curve-like fashion.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmHilbertlikeOrder(
        const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state) {
    bool triangular = false;
    bool rectangular = !triangular && state.inputs.hilbertVD.isValid();

    auto storage = state.ra.alloc();
    auto u = storage.ud(0);
    auto v = storage.ud(1);
    auto uh = storage.ud(2);
    auto vh = storage.ud(3);
    auto a = storage.ud(4);
    auto b = storage.ud(5);
    /* auto isNormal = storage.ud(6); */ // not used directly
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
    /* auto vo = storage2[0].ud(7); */ // not used directly
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

    auto any8 = (hw == HW::XeHPC) ? any : any8h;
    auto any16 = (hw == HW::XeHPC) ? any : any16h;
    bool avoidAny2 = (hw == HW::XeHPC);

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
        mov(1, f0[1],
                vd.uw(1)); // High word of vd = 0xFFFF -> start splitting in x
    else if (triangular)
        cmp(1 | ne | f0[0], state.inputs.diagC, 0);
    mov(1, u, state.inputs.groupCountM);
    mov(1, v, state.inputs.groupCountN);
    mov(4, a0, Immediate::uv(4, 0, 12, 8, 0, 0, 0, 0));
    mov(1, f1.ud(0), 0); // bu = bv = false
    mov(1, np1, triangular ? 0xFFFFFFFF : 0);
    if (triangular)
        cmp(1 | ~f0[0] | ne | f0[0], state.inputs.m, state.inputs.n);
    else
        cmp(2 | le | f0[0], u(1), hilbertBail);
    mov(1, q, state.inputs.groupIDMN);
    add(4, a0[4](1), a0[0](1), 16);
    if (!rectangular && !triangular)
        emad(1, uv1, -1, u.uw(), v.uw(), strategy, state);
    mov(8, a.uw()(1),
            Immediate::uv(0x00010000)); // a = b = 0, normal = 1, reversed = 0;
    if (soff >= 512) {
        add(4, a0, a0, soff);
        soff = 0;
    }
    if (triangular)
        jmpi(1 | f0[0], lTriangularBypass);
    else
        jumpAny2(1 | f0[0], lRecursiveEnd);

    // Rectangular partitioning step. Break dispatch into blocks of roughly desired aspect ratio.
    if (rectangular) {
        auto uvd = uv1;
        movi(8 | f0[1], storage.ud(), indirect[a0].ud(soff)(1));
        mul(1, uvd, u, vd.uw());
        divDown(nbv, q, uvd, uvdRecip, f0[0], strategy, state);
        and_(1 | ne | bv, bv1, nbv, 1);
        mul(1, temp, uvd, nbv.uw());
        mul(1, b, vd.uw(), nbv.uw());
        add(1, q, q, -temp); // no DWxW with source modifiers
        add(1, v, v, -b);
        avg(1, ud, u, -bv1);
        min_(1, v, v, vd.uw());
        avg(1, uh, u, 0);
        mul(1, temp, v.uw(), ud.uw());
        cmp(1 | ge | bu, nbu, q, temp);
        add(1 | bu, q, q, -temp);
        cmp(1 | ne | bu, nbu, nbu.d(), -bv1.d()); // {bu,nbu} ^= bv1
        sel(1 | bu, a, uh, 0);
        avg(1, u, u, nbu.d());
        movi(8 | ~bu | any8, storage.ud(), indirect[a0].ud(soff)(1));
        cmp(2 | le | f0[0], u(1), hilbertBail);
        sel(1 | ~bu, np1, -bv1, 0);
        emad(1, uv1, -1, u.uw(), v.uw(), strategy, state);
        mov(1, f1.ud(0), 0); // bu = bv = false
        jumpAny2(1 | f0[0], lRecursiveEnd);
    }

    // Recursive partitioning. Each step breaks the current block
    //  into 2x2 subblocks and follows the block we are currently in.
    // Exit when one dimension is less than hilbertBail.
    mark(lRecursiveTop);
    {
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
        and_(2, uo(1), uo(1), temp(1));
        emad(1, uv1, -1, u.uw(), v.uw(), strategy, state);
        add(2, a(1), a(1), uo(1));

        cmp(2 | le | f0[0], u(1), hilbertBail);
        movi(8 | ~bu | any8, storage.ud(), indirect[a0].ud(soff)(1));

        if (avoidAny2)
            goto12(16 | ~f0[0] | any16, lRecursiveEnd, lRecursiveTop, true);
        else
            jmpi(1 | ~f0[0] | any2h, lRecursiveTop);
    }
    mark(lRecursiveEnd);
    if (avoidAny2) join(16);

    cmp(8 | ne | f0[0], isReversed, 0);
    movi(8 | f0[0], storage.ud(), indirect[a0].ud(soff)(1));

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
        mov(1, bias, -0.499996185302734375f); // -1/2 + 2^(-17)
        einv(1, divisorFP, divisorFP, strategy, state);
        add(1, divisorFP.ud(), divisorFP.ud(), 1);
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
void gemm_kernel_generator_t<hw>::gemmBoustrophedonOrder(
        const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state) {
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

    auto s0 = state.inputs
                      .bslice; // Slice width/height in WGs. Sign interpretation:
    //   + means slice in m dimension, - means n dimension
    auto thresh = state.inputs.bthresh; // Slice size adjustment threshold
            //   + means increase slice size by 1 starting with this row/column
            //   - means decrease slice size by 1 starting with this row/column

    auto &groupCountM = state.inputs.groupCountM;
    auto &groupCountN = state.inputs.groupCountN;
    auto idMN = state.inputs.groupIDMN;

    Label lBegin, lEnd, lDone, lBeginTri2, lEndTri2, lTricalc1, lTricalc2,
            lTricalcOut;

    auto divqot = [&](const Subregister &num, const Subregister &denom) {
        if (hw < HW::Gen12LP) {
            irem(1, rem, num, denom);
            iqot(1, qot, num, denom);
        } else {
            mov(1, denomFP, denom);
            mov(1, numFP, num);
            mov(1, bias, -0.499996185302734375f); // -1/2 + 2^(-17)
            einv(1, denomFP, denomFP, strategy, state);
            add(1, denomFP.ud(), denomFP.ud(), 1);
            mad(1, qot.f(), bias, numFP, denomFP);
            mov(1, qot, qot.f());
            mad(1, rem, q, -qot.uw(), denom.uw());
        }
    };

    auto ecsel
            = [&](const InstructionModifier &mod,
                      const InstructionModifier &cmod, const FlagRegister &flag,
                      const RegData &dst, const RegData &src0,
                      const RegData &src1, const RegData &src2) {
                  if (hw == HW::Gen9 || dst.getByteOffset() & 7) {
                      cmp(mod | cmod | flag, src2, 0);
                      sel(mod | flag, dst, src0, src1);
                  } else
                      csel(mod | cmod | flag, dst, src0, src1, src2);
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

    if (hw == HW::XeHPC) {
        add(1, s1, abs(s0), 1);
        add(1 | ~f0[0], s1, abs(s0), temp2.d());
        add(1 | ~f1[0], s1, abs(s0), temp2.d());
    } else {
        add(1, s1, abs(s0), temp2.d());
        add(1 | f0[0] | allv, s1, abs(s0), 1);
    }

    mul(1, temp1, s1.uw(), v.uw());

    divqot(q, temp1);

    mul(1, i0, qot.uw(), s1.uw());
    mov(1, islice, qot);
    add(1 | f0[0], i0, i0, ithresh);
    mov(1, q, rem);
    add(1 | sat, temp0, u, -i0);
    min_(1, s, s1, temp0);
    add(1 | f0[0], islice, islice, abs(thresh));

    mul(1, temp2, s.uw(), s.uw());
    emad(1, temp1, temp1, -s.uw(), s.uw(), strategy, state);

    cmp(1 | gt | f0[0], i0, 0); // not first row?
    cmp(1 | lt | f0[1], s1, temp0); // not last row?

    if (hw == HW::XeHPC) {
        cmp(1 | f0[0] | lt | f0[0], q, temp2); // beginning of row?
        cmp(1 | f0[1] | ge | f0[1], q, temp1); // end of row?
    } else {
        cmp(1 | lt | f1[0], q, temp2); // beginning of row?
        cmp(1 | ge | f1[1], q, temp1); // end of row?
    }

    mov(1, two, 2.0f);
    mov(1, bias, 1.25f);

    if (hw == HW::XeHPC) {
        jmpi(1 | f0[0], lBegin);
        jmpi(1 | f0[1], lEnd);
    } else {
        jmpi(1 | f0[0] | allv, lBegin);
        jmpi(1 | f0[1] | allv, lEnd);
    }

    {
        divqot(q, s);

        add(1, i, i0, rem);
        mov(1, j, qot);
    }

    jmpi(1, lDone);

    mark(lBegin);
    {
        avg(1, temp0, temp2, -s); // s(s-1)/2
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
        avg(1, temp0, temp2, s); // s(s+1)/2
        mov(1, f1.ud(0), 0);
        cmp(1 | lt | f0[0], q, temp0);
        jmpi(1 | ~f0[0], lEndTri2);

        eadd3(1, q, temp0, -q, -1);
        mark(lTricalc2);
        {
            mov(1, qFP, q);
            mad(1, qFP, bias, qFP, two);
            esqt(1, qFP, qFP, strategy, state);
            if (hw == HW::Gen9) rnde(1, qFP, qFP);
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
            if (hw == HW::Gen9) rnde(1, qFP, qFP);
            mov(1, i, qFP);
            mul(1, temp0, i.uw(), i.uw());
            avg(1, temp0, temp0, -i);
            add(1, j, q, -temp0);
        }

        mark(lTricalcOut);
        eadd3(1 | f1[0], i, s, -i, -1);
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
void gemm_kernel_generator_t<hw>::gemmReverseLoops(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state) {
    for (LoopType l : {LoopM, LoopN})
        if (strategy.reverse[l]) {
            bool fusedL = strategy.fused && (l == strategy.fusedLoop);
            auto q = (l == LoopM) ? state.inputs.m : state.inputs.n;
            auto q0 = (l == LoopM) ? state.i0 : state.j0;
            auto q0Align = state.ra.alloc_sub<uint32_t>();
            auto temp = state.ra.alloc_sub<uint32_t>();

            add(1, q0Align, q, -1);
            if (strategy.fixedWG(problem)) {
                mod(temp, q0, strategy.wg[l] * strategy.unroll[l], strategy,
                        state);
                alignDown(q0Align, q0Align, strategy.wg[l] * strategy.unroll[l],
                        strategy, state);
                shl(1, temp, temp, 1);
                eadd3(1 | ge | f0[0], q0Align.d(), q0Align, -q0, temp);
                mov(1 | f0[0], q0, q0Align);
            } else if (fusedL) {
                shl(1, temp, state.fusedID, 1);
                alignDown(q0Align, q0Align, 2 * strategy.unroll[l], strategy,
                        state);
                eadd3(1 | ge | f0[0], q0Align.d(), q0Align, -q0, temp);
                mov(1 | f0[0], q0, q0Align);
            } else {
                alignDown(
                        q0Align, q0Align, strategy.unroll[l], strategy, state);
                cmp(1 | le | f0[0], q0, q0Align);
                add(1 | f0[0], q0, q0Align, -q0);
            }
            state.ra.safeRelease(temp);
            state.ra.safeRelease(q0Align);
        }
}

// Reorder local IDs as needed.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmReorderLocalIDs(
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
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
            shr(1, temp, lidO, log2(adjustEvery));
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

// Convert leading dimension and offset inputs to bytes.
template <ngen::HW hw>
void gemm_kernel_generator_t<hw>::gemmScaleInputs(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state) {
    auto Ta_ext = problem.Ta_ext, Tb_ext = problem.Tb_ext,
         Tc_ext = problem.Tc_ext, Tco = problem.Tco;

    {
        emulConstant(1, state.inputs.lda, state.inputs.lda, Ta_ext.size(),
                strategy, state);
        if (state.inputs.ldb != state.inputs.lda)
            emulConstant(1, state.inputs.ldb, state.inputs.ldb, Tb_ext.size(),
                    strategy, state);
    }
    for (int q = 0; q < state.C_count; q++)
        emulConstant(1, state.inputs.ldc[q], state.inputs.ldc[q], Tc_ext.size(),
                strategy, state);

    {
        emulConstant(1, state.inputs.offsetA, state.inputs.offsetA,
                Ta_ext.size(), strategy, state);
        emulConstant(1, state.inputs.offsetB, state.inputs.offsetB,
                Tb_ext.size(), strategy, state);
        for (int q = 0; q < state.C_count; q++)
            emulConstant(1, state.inputs.offsetC[q], state.inputs.offsetC[q],
                    Tc_ext.size(), strategy, state);
        if (problem.cOffset != COffset::None)
            emulConstant(1, state.inputs.offsetCO, state.inputs.offsetCO,
                    Tco.size(), strategy, state);
    }

    if (problem.batch == BatchMode::Strided)
        for (int b = 0; b < problem.batchDims; b++) {
            emulConstant(1, state.inputs.strideA[b], state.inputs.strideA[b],
                    Ta_ext.size(), strategy, state);
            emulConstant(1, state.inputs.strideB[b], state.inputs.strideB[b],
                    Tb_ext.size(), strategy, state);
            emulConstant(1, state.inputs.strideC[b], state.inputs.strideC[b],
                    Tc_ext.size(), strategy, state);
        }
}

// Cache multiples of lda/ldb for later address calculations.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmCacheLDABMultiples(
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
    int na = 0, nb = 0;

    if (!strategy.A.address2D) switch (problem.A.layout) {
            case MatrixLayout::N:
                na = std::max(strategy.ka_load, strategy.ka_prefetch);
                break;
            case MatrixLayout::T:
                na = strategy.unroll[LoopM];
                if (isTransposing(strategy.A.accessType))
                    na = std::min(na, maxScatteredSIMD(hw, strategy.A));
                break;
            default: break;
        }

    if (!strategy.B.address2D) switch (problem.B.layout) {
            case MatrixLayout::T:
                nb = std::max(strategy.kb_load, strategy.kb_prefetch);
                break;
            case MatrixLayout::N:
                nb = strategy.unroll[LoopN];
                if (isTransposing(strategy.B.accessType))
                    nb = std::min(nb, maxScatteredSIMD(hw, strategy.B));
                break;
            default: break;
        }

    if (na <= 2) na = 0;
    if (nb <= 2) nb = 0;

    if (na || nb) extendIndexVec(std::max(na, nb), state);

    if (na) {
        bool a64 = (strategy.A.base.getModel() == ModelA64);
        state.ldaMultiples
                = createLDMultiples(a64, na, state.lda, strategy, state);
    }

    if (nb) {
        bool a64 = (strategy.B.base.getModel() == ModelA64);
        state.ldbMultiples
                = createLDMultiples(a64, nb, state.ldb, strategy, state);
    }
}

// Cache multiples of ldc for later address calculations.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmCacheLDCMultiples(
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state, bool prefetch) {
    if ((prefetch ? strategy.C_prefetch : strategy.C).address2D) return;

    int nc = 0;
    switch (problem.C.layout) {
        case MatrixLayout::N: nc = strategy.unroll[LoopN]; break;
        case MatrixLayout::T: nc = strategy.unroll[LoopM]; break;
        default: break;
    }

    if (nc <= 2) return;

    bool a64 = (strategy.C.base.getModel() == ModelA64);
    int C_count = prefetch ? 1 : state.C_count;
    for (int q = 0; q < C_count; q++)
        state.ldcMultiples[q] = createLDMultiples(
                a64, nc, state.inputs.ldc[q], strategy, state);
}

// GEMM kernel generation interface.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemm(GEMMProblem problem,
        GEMMStrategy strategy, const InterfaceHandler &interface_) {
    GEMMState state(hw);
    interface = interface_;
    gemm(problem, strategy, state);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::gemm(
        GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state) {
    bool inFusedGEMM = state.fusedGEMM.active;
    bool anyKParallel = strategy.kParallelLocal || strategy.kParallel;

    Label labelKernelDone, labelReentry;

    // By default, don't use dispatch mask.
    setDefaultNoMask();
    setDefaultAutoSWSB();

    // Set up.
    gemmInitState(problem, strategy, state);

    // Transfer surface indices to strategy AddressBases.
    if (!strategy.A.base.isStateless())
        strategy.A.base.setIndex(state.inputs.surfaceA);
    if (!strategy.B.base.isStateless())
        strategy.B.base.setIndex(state.inputs.surfaceB);
    if (!strategy.C.base.isStateless()) {
        strategy.C.base.setIndex(state.inputs.surfaceC[0]);
        if (state.C_count > 1) stub();
    }
    if ((problem.cOffset != COffset::None) && !strategy.CO.base.isStateless())
        strategy.CO.base.setIndex(state.inputs.surfaceCO);

    // Prologue.
    if (!inFusedGEMM) prologue(strategy);

    // Grab fused ID if needed, and multiply by unroll.
    getFusedID(strategy.unroll[strategy.fusedLoop], problem, strategy, state);

    if (!inFusedGEMM) {
        // Divide out subgroup size from local size 0 and local ID 0, and reorder threads for fusing if needed.
        removeSG(problem, strategy, state);
        reorderFusedEUs(problem, strategy, state);

    } /* !inFusedGEMM */

    // Check for copy or compute kernel.
    if (strategy.splitCopy) {
        state.isCompute = state.ra.alloc_sub<uint32_t>(
                getHint(HintType::LongTerm, strategy));
        auto localIDY = (strategy.loopOrder[1] == LoopN)
                ? state.inputs.localIDN
                : state.inputs.localIDM;
        auto wgY = strategy.wg[strategy.loopOrder[1]];
        cmp(1 | ge | f1[1], state.isCompute, localIDY, wgY);
        if (is_zero_or_pow2(wgY))
            and_(1, localIDY, localIDY, wgY - 1);
        else
            add(1 | f1[1], localIDY, localIDY, -wgY);
    }

    // Scale LDs/offsets.
    gemmScaleInputs(problem, strategy, state);

    // Local ID handling and saving.
    gemmReorderLocalIDs(problem, strategy, state);

    if (strategy.needsMNLocalIDs()) saveMNLocalIDs(strategy, state);

    if (strategy.needsKLocalIDs()) saveKLocalIDSize(strategy, state);

    // Save full k size if needed.
    bool anyAB2D = strategy.A.address2D || strategy.B.address2D
            || (strategy.prefetchA && strategy.A_prefetch.address2D)
            || (strategy.prefetchB && strategy.B_prefetch.address2D);
    if (anyKParallel) {
        if (strategy.persistent || anyAB2D) {
            state.fullK = state.ra.alloc_sub<uint32_t>(
                    getHint(HintType::LongTerm, strategy));
            mov(1, state.fullK, state.inputs.k);
        }
    } else
        state.fullK = state.inputs.k;

    // Persistent thread preparation and re-entry.
    if (strategy.persistent) {
        if (!strategy.linearOrder()) stub();
        if (problem.batch != BatchMode::None)
            stub(); // need to wrangle groupIDK also

        auto newGroupIDMN = state.ra.alloc_sub<uint32_t>(
                getHint(HintType::LongTerm, strategy));
        mov(1, newGroupIDMN, state.inputs.groupIDMN);
        state.inputs.groupIDMN = newGroupIDMN;

        gemmFoldOffsets(problem, strategy, state);

        mark(labelReentry);
    }

    // Group ID remapping.
    if (strategy.hilbertOrder)
        gemmHilbertlikeOrder(problem, strategy, state);
    else if (strategy.boustrophedon)
        gemmBoustrophedonOrder(problem, strategy, state);

    // Batch handling.
    gemmGetBatchIDs(problem, strategy, state);

    // Compute offset for A, B, C for non-strided and strided batch.
    gemmOffsetBatchABC(problem, strategy, state);

    // 32-bit add check. TODO: move out of persistent loop for non-batch.
    gemmCheck32(problem, strategy, state);

    // Calculate i0, j0, h0 -- the initial i/j/h indices for this thread.
    bool needH0 = anyKParallel;

    state.i0 = state.ra.alloc_sub<uint32_t>(
            getHint(HintType::TempComp0, strategy));
    state.j0 = state.ra.alloc_sub<uint32_t>(
            getHint(HintType::TempComp1, strategy));
    if (needH0)
        state.h0 = state.ra.alloc_sub<uint32_t>(
                getHint(HintType::TempComp0, strategy));

    bool wgCheck = wgRemCheck(problem, strategy);
    bool gemmtBarriers = problem.gemmt() && strategy.needsBarrier();

    Subregister idM, idN, idK;
    Subregister wgI0, wgJ0;

    idM = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp1, strategy));
    idN = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
    if (strategy.kParallel)
        idK = state.ra.alloc_sub<uint32_t>(
                getHint(HintType::TempComp0, strategy));

    if (strategy.fixedWG(problem)) {
        mulConstant(1, idM, state.inputs.groupIDM, strategy.wg[LoopM]);
        mulConstant(1, idN, state.inputs.groupIDN, strategy.wg[LoopN]);
        if (strategy.kParallel)
            mulConstant(1, idK, state.inputs.groupIDK, strategy.wg[LoopK]);
    } else {
        mul(1, idM, state.inputs.groupIDM, state.inputs.localSizeM.uw());
        mul(1, idN, state.inputs.groupIDN, state.inputs.localSizeN.uw());
        if (strategy.kParallel)
            mul(1, idK, state.inputs.groupIDK, state.inputs.localSizeK.uw());
    }

    if (wgCheck || gemmtBarriers) {
        wgI0 = state.ra.alloc_sub<uint32_t>(
                getHint(HintType::TempComp0, strategy));
        wgJ0 = state.ra.alloc_sub<uint32_t>(
                getHint(HintType::TempComp1, strategy));
        mulConstant(1, wgI0, idM, strategy.unroll[LoopM]);
        mulConstant(1, wgJ0, idN, strategy.unroll[LoopN]);
    }

    add(1, idM, idM, state.lidM);
    add(1, idN, idN, state.lidN);
    if (strategy.kParallel) add(1, idK, idK, state.lidK);

    mulConstant(1, state.i0, idM, strategy.unroll[LoopM]);
    mulConstant(1, state.j0, idN, strategy.unroll[LoopN]);

    if (strategy.kParallel)
        emul(1, state.h0, idK, state.inputs.k0, strategy, state);
    else if (strategy.kParallelLocal)
        mul(1, state.h0, state.inputs.k0, state.lidK);

    gemmReverseLoops(problem, strategy, state);

    state.ra.safeRelease(idM);
    state.ra.safeRelease(idN);
    state.ra.safeRelease(idK);
    state.ra.safeRelease(state.inputs.localIDM);
    state.ra.safeRelease(state.inputs.localIDN);
    if (!strategy.needsMNLocalIDs()) state.lidM = state.lidN = invalid;
    if (!strategy.persistent) {
        state.ra.safeRelease(state.inputs.localSizeM);
        state.ra.safeRelease(state.inputs.localSizeN);
    }
    if (anyKParallel) {
        state.ra.safeRelease(state.inputs.localIDK);
        if (!strategy.persistent) state.ra.safeRelease(state.inputs.localSizeK);
    }
    if (strategy.linearOrder() || strategy.persistent) {
        state.ra.safeRelease(state.inputs.groupIDM);
        state.ra.safeRelease(state.inputs.groupIDN);
    }

    moveR0(strategy, state);

    // Adjust k range as needed.
    if (anyKParallel) {
        add(1, state.inputs.k,
                strategy.persistent ? state.fullK : state.inputs.k, -state.h0);
        min_(1, state.inputs.k, state.inputs.k, state.inputs.k0);

        bool keepK0 = false;
        keepK0 |= strategy.kParallelLocal
                && (strategy.barrierFreq > 0 || strategy.slmBuffers > 0);
        keepK0 |= strategy.persistent;

        if (!keepK0) state.ra.safeRelease(state.inputs.k0);
    }

    // Compute workgroup remainders if needed.
    if (wgCheck) {
        state.remaindersWG[LoopM] = state.ra.alloc_sub<uint32_t>(
                getHint(HintType::TempComp1, strategy));
        state.remaindersWG[LoopN] = state.ra.alloc_sub<uint32_t>(
                getHint(HintType::TempComp0, strategy));
        add(1 | sat, state.remaindersWG[LoopM], -wgI0, state.inputs.m);
        add(1 | sat, state.remaindersWG[LoopN], -wgJ0, state.inputs.n);
    }
    state.ra.safeRelease(wgI0);
    state.ra.safeRelease(wgJ0);

    // Compute base addresses for A, B, C.
    gemmOffsetABC(true, state.i0, state.j0, state.h0, problem, strategy, state);

    gemmSetupABC(problem, strategy, state);
    gemmSubkernel(problem, strategy, state);

    mark(labelKernelDone);

    // Persistent thread loop. Advance group ID and re-enter kernel if there's more work to do.
    if (strategy.persistent) {
        status << "Persistent loop" << status_stream::endl;
        if (state.inputs.groupCountMN.isInvalid()) {
            state.inputs.groupCountMN = state.ra.alloc_sub<uint32_t>(
                    getHint(HintType::LongTerm, strategy));
            emul(1, state.inputs.groupCountMN, state.inputs.groupCountM,
                    state.inputs.groupCountN, strategy, state);
        }

        add(1, state.inputs.groupIDMN, state.inputs.groupIDMN,
                state.inputs.groupStride);
        cmp(1 | lt | state.flagAP, state.inputs.groupIDMN,
                state.inputs.groupCountMN);

        state.ra.safeRelease(state.inputs.groupCountMN);
        gemmRestoreOffsets(problem, strategy, state);

        if (strategy.slmBuffers > 0) {
            auto temp = state.ra.alloc();
            useR0(state, [&](const GRF &r0_info) { barrier(temp, r0_info); });
            state.ra.safeRelease(temp);
        }

        jmpi(1 | state.flagAP, labelReentry);
    }

    if (!inFusedGEMM) {
        epilogue(strategy, state);
        padding();
    }
}

template <HW hw>
SubregisterPair gemm_kernel_generator_t<hw>::allocIncrement(
        const GEMMStrategy &strategy, CommonState &state) {
    if (strategy.avoidIncConflicts)
        return SubregisterPair(state.ra.alloc_sub<uint32_t>(
                                       getHint(HintType::LongTerm0, strategy)),
                state.ra.alloc_sub<uint32_t>(
                        getHint(HintType::LongTerm1, strategy)));
    else
        return SubregisterPair(state.ra.alloc_sub<uint32_t>(
                getHint(HintType::LongTerm, strategy)));
}

// Calculate and cache lda_ka (= lda * ka) and ldb_kb (= ldb * kb) as necessary.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmCalcIncrements(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state, int ka_load,
        int kb_load, bool doA, bool doB) {
    int nr = strategy.avoidIncConflicts ? 2 : 1;

    if (ka_load == 0) ka_load = strategy.ka_inc();
    if (kb_load == 0) kb_load = strategy.kb_inc();

    // If A is nontranspose, we need lda * ka_load * elementSize.
    if (doA && (problem.A.layout == MatrixLayout::N)) {
        if (!strategy.A.address2D) {
            if (ka_load > 1) {
                if (state.lda_ka.isInvalid())
                    state.lda_ka = allocIncrement(strategy, state);
                for (int i = 0; i < nr; i++)
                    emulConstant(1, state.lda_ka.getReg(i), state.inputs.lda,
                            ka_load, strategy, state);
                state.ka_cached = ka_load;
            } else if (strategy.avoidIncConflicts)
                duplicateScalar(state.lda, state);
        }
        if (strategy.prefetchA && !strategy.A_prefetch.address2D
                && (strategy.ka_pfStride != ka_load || strategy.A.address2D)) {
            if (strategy.ka_pfStride > 1) {
                if (state.lda_ka_prefetch.isInvalid())
                    state.lda_ka_prefetch = allocIncrement(strategy, state);
                for (int i = 0; i < nr; i++)
                    emulConstant(1, state.lda_ka_prefetch.getReg(i),
                            state.inputs.lda, strategy.ka_pfStride, strategy,
                            state);
            } else if (strategy.avoidIncConflicts)
                duplicateScalar(state.lda, state);
        }
    }

    // Similarly for B if it's transpose.
    if (doB && (problem.B.layout == MatrixLayout::T)) {
        if (!strategy.B.address2D) {
            if (kb_load > 1) {
                if (state.ldb_kb.isInvalid())
                    state.ldb_kb = allocIncrement(strategy, state);
                for (int i = 0; i < nr; i++)
                    emulConstant(1, state.ldb_kb.getReg(i), state.inputs.ldb,
                            kb_load, strategy, state);
                state.kb_cached = kb_load;
            } else if (strategy.avoidIncConflicts)
                duplicateScalar(state.ldb, state);
        }
        if (strategy.prefetchB && !strategy.B_prefetch.address2D
                && (strategy.kb_pfStride != kb_load || strategy.B.address2D)) {
            if (strategy.kb_pfStride > 1) {
                if (state.ldb_kb_prefetch.isInvalid())
                    state.ldb_kb_prefetch = allocIncrement(strategy, state);
                for (int i = 0; i < nr; i++)
                    emulConstant(1, state.ldb_kb_prefetch.getReg(i),
                            state.inputs.ldb, strategy.kb_pfStride, strategy,
                            state);
            } else if (strategy.avoidIncConflicts)
                duplicateScalar(state.ldb, state);
        }
    }
}

template <HW hw>
void gemm_kernel_generator_t<hw>::gemmSubkernel(
        GEMMProblem &problem, GEMMStrategy &strategy, GEMMState state) {
    Label labelSubkernelDone, labelSubkernelEarlyExit;

    status << "Begin subkernel: unroll " << strategy.unroll[LoopM] << 'x'
           << strategy.unroll[LoopN] << status_stream::endl;

    // Calculate remainders for m/n loops: clamp(m - i0, 0, unrollM), clamp(n - j0, 0, unrollN).
    // Careful with this clamping, because unroll may change in remainder handling.
    bool remM = (strategy.remHandling[LoopM] != RemainderHandling::Ignore);
    bool remN = (strategy.remHandling[LoopN] != RemainderHandling::Ignore);
    bool fusedremM = remM && strategy.fused && (strategy.fusedLoop == LoopM);
    bool fusedremN = remN && strategy.fused && (strategy.fusedLoop == LoopN);
    bool earlyExit = !strategy.lateExit();

    if (fusedremM || fusedremN) {
        state.remFusedStorage = state.ra.alloc_sub<uint32_t>();
        add(1, state.remFusedStorage, -state.fusedID,
                uint16_t(strategy.unroll[strategy.fusedLoop]));
    }
    if (remM || !earlyExit) {
        state.remaindersFused[LoopM] = state.remainders[LoopM]
                = state.ra.alloc_sub<uint32_t>(
                        getHint(HintType::LongTerm, strategy));
        InstructionModifier mod = 1 | sat;
        if (!fusedremM && earlyExit) mod = mod | le | f0[1];
        add(mod, state.remainders[LoopM], -state.i0, state.inputs.m);
    }
    if (remN || !earlyExit) {
        state.remaindersFused[LoopN] = state.remainders[LoopN]
                = state.ra.alloc_sub<uint32_t>(
                        getHint(HintType::LongTerm, strategy));
        InstructionModifier mod = 1 | sat;
        if (!fusedremN && earlyExit) mod = mod | le | f1[1];
        add(mod, state.remainders[LoopN], -state.j0, state.inputs.n);
    }
    if (fusedremM || fusedremN) {
        state.remaindersFused[strategy.fusedLoop] = state.remFusedStorage;
        add(1 | sat, state.remFusedStorage, -state.remFusedStorage,
                state.remainders[strategy.fusedLoop]);
        if (earlyExit) {
            cmp(1 | le | (fusedremM ? f0[1] : f1[1]), null.d(),
                    state.remainders[strategy.fusedLoop].d(), -state.fusedID);
            state.allowEmptyC = true;
        }
    }
    if (remM)
        min_(1, state.remainders[LoopM], state.remainders[LoopM],
                uint16_t(strategy.unroll[LoopM]));
    if (remN)
        min_(1, state.remainders[LoopN], state.remainders[LoopN],
                uint16_t(strategy.unroll[LoopN]));

    gemmCalcIncrements(problem, strategy, state);

    // Early exit if nothing to do. Keep fused threads together.
    if (earlyExit && (remM || remN)) {
        InstructionModifier cond;
        if (remM && remN)
            cond = 1 | f0[1] | anyv;
        else if (remM)
            cond = 1 | f0[1];
        else
            cond = 1 | f1[1];

        if (state.fusedGEMM.active)
            and_(16 | nz | state.fusedGEMM.needLateGEMMDone, null.uw(),
                    state.inputs.flags.uw(), FlagEarlyFusedGEMMDone);

        auto &label = state.fusedGEMM.active ? labelSubkernelEarlyExit
                                             : labelSubkernelDone;

        ejmpi(cond, label);
    }

    // Create the kernel body. If enabled, create two versions, one with A/B more aligned.
    bool success;
    if (!strategy.optAlignAB)
        success = gemmMEdge(problem, strategy, state);
    else {
        // Check alignment of effA, effB, lda, and ldb.
        Label labelUnaligned;
        uint16_t mask = (strategy.optAlignAB - 1);
        bool check_lda = !isPacked(problem.A.layout);
        bool check_ldb = !isPacked(problem.B.layout);
        if (problem.A.alignment & mask) {
            and_(1 | nz | f0[0], null.uw(), state.effA.uw(), mask);
            if (check_lda)
                and_(1 | nz | f1[0], null.uw(), state.inputs.lda.uw(), mask);
        }
        if (problem.B.alignment & mask) {
            and_(1 | nz | f0[1], null.uw(), state.effB.uw(), mask);
            if (check_ldb)
                and_(1 | nz | f1[1], null.uw(), state.inputs.ldb.uw(), mask);
        }
        if (problem.A.alignment & mask) {
            InstructionModifier amod = check_lda ? 1 | f0[0] | anyv : 1 | f0[0];
            ejmpi(amod, labelUnaligned);
        }
        if (problem.B.alignment & mask) {
            InstructionModifier bmod = check_ldb ? 1 | f0[1] | anyv : 1 | f0[1];
            ejmpi(bmod, labelUnaligned);
        }

        auto alignedProblem = problem;
        alignedProblem.A.setAlignment(
                std::max<int>(problem.A.alignment, strategy.optAlignAB));
        alignedProblem.B.setAlignment(
                std::max<int>(problem.B.alignment, strategy.optAlignAB));

        status << "Aligned A/B" << status_stream::endl;
        success = gemmMEdge(alignedProblem, strategy, state);

        if (!success && lastException) std::rethrow_exception(lastException);

        state.isNested ? jmpi(1, labelSubkernelDone)
                       : epilogue(strategy, state);

        mark(labelUnaligned);

        status << "Unaligned A/B" << status_stream::endl;
        if (!gemmMEdge(problem, strategy, state)) {
            auto modStrategy = strategy;

            modStrategy.checkAdd32
                    = false; // Don't optimize additions on this (slow) path to reduce code size.
            status << "Reducing register usage" << status_stream::endl;
            success = success && modStrategy.minimize(hw, problem);

            gemmCalcIncrements(problem, modStrategy,
                    state); // Recalculate lda_ka/ldb_kb as they have changed.

            success = success && gemmMEdge(problem, modStrategy, state);
        }
    }

    if (!success)
        lastException ? std::rethrow_exception(lastException)
                      : throw std::runtime_error("Could not generate kernel.");

    mark(labelSubkernelDone);

    if (state.fusedGEMM.active) {
        mov(1, state.fusedGEMM.needLateGEMMDone, 0);
        mark(labelSubkernelEarlyExit);
    }

    safeRelease(state.lda_ka, state);
    safeRelease(state.ldb_kb, state);
    safeRelease(state.lda_ka_prefetch, state);
    safeRelease(state.ldb_kb_prefetch, state);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::gemmSuperkernelInitState(
        GEMMSuperkernelProblem &problem, GEMMSuperkernelStrategy &strategy,
        GEMMSuperkernelState &state) {
    if (strategy.persistent) interface.requireGlobalAtomics();

    gemmInitState(problem, strategy.substrategies[0], state, true);

    state.isNested |= strategy.persistent;

    state.inputsSK.surfacePlan = interface.getArgumentSurface("plan");
    state.inputsSK.planCount = interface.getArgument("plan_count");
    state.inputsSK.localID = interface.getLocalID(0);
    state.inputsSK.localSize = interface.getLocalSize(0);

    state.ra.claim(state.inputsSK.localID);
    state.ra.claim(state.inputsSK.localSize);
    state.ra.claim(state.inputsSK.planCount);
}

// Create a GEMM superkernel.
template <HW hw>
void gemm_kernel_generator_t<hw>::gemmSuperkernel(
        GEMMSuperkernelProblem problem, GEMMSuperkernelStrategy strategy,
        const InterfaceHandler &interface_) {
    auto &strategy0 = strategy.substrategies[0];
    bool persistent = strategy.persistent;

    GEMMSuperkernelState state(hw);

    // Set up.
    setDefaultNoMask();
    setDefaultAutoSWSB();
    interface = interface_;
    gemmSuperkernelInitState(problem, strategy, state);
    state.ra.safeRelease(state.inputs.localIDN);
    state.ra.safeRelease(state.inputs.localSizeN);

    for (auto &ss : strategy.substrategies) {
        if (!ss.A.base.isStateless()) ss.A.base.setIndex(state.inputs.surfaceA);
        if (!ss.B.base.isStateless()) ss.B.base.setIndex(state.inputs.surfaceB);
        if (!ss.C.base.isStateless())
            ss.C.base.setIndex(state.inputs.surfaceC[0]);
    }

    // Prevent unhelpful layouts.
    if (problem.A.layout == MatrixLayout::PackedRows) stub();
    if (problem.B.layout == MatrixLayout::PackedColumns) stub();

    Label loopSK, loopSKEnd;

    // Prologue.
    prologue(strategy0);

    // Grab fused ID if needed.
    getFusedID(1, problem, strategy0, state);

    // Get my plan ID and convert to offset in plan.
    auto idX = r0.ud(1);
    auto header = state.ra.alloc();
    auto poff = header.ud(2);
    constexpr uint16_t eltSz = 8;

    auto temp = state.ra.alloc_sub<uint32_t>();

    mulConstant(1, temp, state.inputsSK.planCount, strategy.subgroupSize());
    mul(1, poff, idX, state.inputsSK.localSize);
    add(1, poff, poff, state.inputsSK.localID.uw(0));
    cmp<uint32_t>(1 | ge | f0[0], poff, temp);
    if (eltSz < strategy.subgroupSize())
        shr(1, poff, poff, log2(strategy.subgroupSize() / eltSz));
    else if (eltSz > strategy.subgroupSize())
        mulConstant(1, poff, poff, eltSz / strategy.subgroupSize());

    state.ra.safeRelease(temp);
    state.ra.safeRelease(state.inputsSK.localID);
    state.ra.safeRelease(state.inputsSK.localSize);

    if (persistent) add(1, poff, poff, eltSz);

    // Move r0 to acc0 if configured.
    moveR0(strategy0, state);

    // Quick exit for extra threads (uniform WG).
    jmpi(1 | f0[0], loopSKEnd);

    // Retrieve plan element.
    auto pdata = state.ra.alloc(getHint(HintType::TempComp0, strategy0));
    load(8, pdata, aligned_block_oword(1), Surface(state.inputsSK.surfacePlan),
            header);
    state.ra.safeRelease(header);

    gemmScaleInputs(problem, strategy0, state); // Scale inputs while waiting.

    state.i0 = pdata.d(0);
    state.j0 = pdata.d(1);

    state.ra.safeRelease(pdata);
    state.ra.claim(state.i0);
    state.ra.claim(state.j0);

    auto flagKID0 = f1[0];
    auto flagKID1 = f1[1];

    if (strategy.multiM) cmp(1 | lt | flagKID0, null.d(), state.i0, 0);
    if (strategy.multiN) cmp(1 | lt | flagKID1, null.d(), state.j0, 0);
    and_(2, state.i0.ud()(1), state.i0.ud()(1), uint32_t(0x7FFFFFFF));

    // Initial offset of A/B/C.
    gemmOffsetABC(
            true, state.i0, state.j0, Subregister(), problem, strategy0, state);
    gemmSetupABC(problem, strategy0, state);

    // Save i0, j0 for later.
    state.last_i0 = state.ra.alloc_sub<int32_t>(
            getHint(HintType::LongTerm, strategy0));
    state.last_j0 = state.ra.alloc_sub<int32_t>(
            getHint(HintType::LongTerm, strategy0));
    mov(1, state.last_i0, state.i0);
    mov(1, state.last_j0, state.j0);

    // Top of superkernel loop.
    status << "Begin superkernel loop" << status_stream::endl;
    mark(loopSK);
    {
        // Dispatch appropriate kernel, supporting up to 4 subkernels.
        int kidx = 0;
        Label labelM1, labelM0N1, labelM1N1, labelKernelDone;
        if (strategy.multiM) jmpi(1 | flagKID0, labelM1);
        if (strategy.multiN) jmpi(1 | flagKID1, labelM0N1);

        gemmSubkernel(problem, strategy.substrategies[kidx++], state);

        if (strategy.multiN) {
            jmpi(1, labelKernelDone);
            mark(labelM0N1);
            gemmSubkernel(problem, strategy.substrategies[kidx++], state);
        }

        if (strategy.multiM) {
            jmpi(1, labelKernelDone);

            mark(labelM1);
            if (strategy.multiN) jmpi(1 | flagKID1, labelM1N1);

            gemmSubkernel(problem, strategy.substrategies[kidx++], state);

            if (strategy.multiN) {
                jmpi(1, labelKernelDone);
                mark(labelM1N1);
                gemmSubkernel(problem, strategy.substrategies[kidx++], state);
            }
        }

        mark(labelKernelDone);

        if (persistent) {
            // Get next plan element via atomic increment of plan ID counter.
            auto header = state.ra.alloc();
            auto nextID
                    = state.ra.alloc(getHint(HintType::TempComp1, strategy0));
            auto pdata
                    = state.ra.alloc(getHint(HintType::TempComp0, strategy0));

            mov<uint32_t>(8, header, uint16_t(0));
            atomic(AtomicOp::inc, 1, nextID, scattered_dword(),
                    Surface(state.inputsSK.surfacePlan), header);

            // Load next plan element, or exit if no more work.
            mulConstant<uint32_t>(1, header[2], nextID[0], eltSz);
            cmp<uint32_t>(
                    1 | ge | f0[0], null, nextID[0], state.inputsSK.planCount);
            add<uint32_t>(1, header[2], header[2], eltSz);

            jmpi(1 | f0[0], loopSKEnd);

            load(8, pdata, aligned_block_oword(1),
                    Surface(state.inputsSK.surfacePlan), header);
            state.ra.safeRelease(header);
            state.ra.safeRelease(nextID);

            // Load next (i0, j0) and kernel IDs.
            auto in_i0 = pdata.d(0);
            auto in_j0 = pdata.d(1);

            if (strategy.multiM) cmp(1 | lt | flagKID0, null.d(), in_i0, 0);
            if (strategy.multiN) cmp(1 | lt | flagKID1, null.d(), in_j0, 0);
            and_(1, state.i0.ud(), in_i0.ud(), uint32_t(0x7FFFFFFF));
            and_(1, state.j0.ud(), in_j0.ud(), uint32_t(0x7FFFFFFF));

            // Get difference in i0 and j0...
            add(1, in_i0, state.i0, -state.last_i0);
            add(1, in_j0, state.j0, -state.last_j0);

            // ... save current (i0, j0) for later...
            mov(1, state.last_i0, state.i0);
            mov(1, state.last_j0, state.j0);

            // ...and offset A, B, C appropriately.
            gemmOffsetABC(false, in_i0, in_j0, Subregister(), problem,
                    strategy0, state);

            state.ra.safeRelease(pdata);

            state.ra.safeRelease(state.i0);
            state.ra.safeRelease(state.j0);

            // Ready for the next kernel.
            jmpi(1, loopSK);
        }
    }
    mark(loopSKEnd);

    epilogue(strategy.substrategies[0], state);
    padding();
}

// Get driver information from this strategy.
template <HW hw>
CommonDriverInfo gemm_kernel_generator_t<hw>::driverInfo(
        const GEMMProblem &problem, const GEMMStrategy &strategy) {
    CommonDriverInfo info;

    info.subgroupSize = strategy.subgroupSize;
    info.fusedLoop = strategy.fused ? strategy.fusedLoop : LoopNone;
    info.grfCount = strategy.GRFs;
    for (int d = 0; d < 3; d++) {
        info.loopOrder[d] = strategy.loopOrder[d];
        info.blocking[d] = strategy.blocking[d];
        info.blockingAlt[d] = strategy.blockingAlt[d];
        info.unroll[d] = strategy.unroll[d];
        info.wg[d] = strategy.wg[d];
    }
    info.wgExpand = strategy.splitCopy ? 2 : 1;
    if (strategy.hilbertOrder) {
        info.loopOrder[0] = (info.loopOrder[0] == LoopN) ? LoopMNHilbertNMK
                                                         : LoopMNHilbertMNK;
        info.loopOrder[1] = LoopNone;
    } else if (strategy.boustrophedon) {
        info.loopOrder[0] = (info.loopOrder[0] == LoopN)
                ? LoopMNBoustrophedonNMK
                : LoopMNBoustrophedonMNK;
        info.loopOrder[1] = LoopNone;
    }
    if (strategy.persistent)
        info.loopOrder[0]
                = static_cast<LoopType>(info.loopOrder[0] | LoopPersistent);
    if (problem.batch == BatchMode::None && !strategy.kParallelLocal)
        info.loopOrder[2] = LoopNone;
    info.wgUpdate = strategy.getWGType(problem);
    info.kRemainderHandling
            = (strategy.remHandling[LoopK] != RemainderHandling::Ignore);
    info.kParallel = strategy.kParallel;
    info.kParallelLocal = strategy.kParallelLocal;
    info.slm = int(gemmSLMSize(problem, strategy));
    info.perKSLM = int(gemmPerKSLMSize(problem, strategy));
    info.alignment[0] = problem.A.alignment;
    info.alignment[1] = problem.B.alignment;
    info.alignment[2] = problem.C.alignment;
    info.support4GB[0] = (strategy.A.base.getModel() == ModelA64);
    info.support4GB[1] = (strategy.B.base.getModel() == ModelA64);
    info.support4GB[2] = (strategy.C.base.getModel() == ModelA64);

    return info;
}

template <HW hw>
CommonDriverInfo gemm_kernel_generator_t<hw>::driverInfo(
        const GEMMSuperkernelProblem &problem, const GEMMStrategy &strategy) {
    auto info = driverInfo(static_cast<GEMMProblem>(problem), strategy);
    return info;
}

// Return the maximum possible k size for copied SLM data.
int GEMMStrategy::maxKSLM(const GEMMProblem &problem, bool isA) const {
    return unrollKSLM;
}

// Validate a GEMM strategy, correcting settings as necessary.
void GEMMStrategy::preflight(HW hw, const GEMMProblem &problem) {
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc;
    auto Ta_real = Ta.real();
    auto Tb_real = Tb.real();
    auto Tc_real = Tc.real();

    // Addressing preflight.

    if (C.atomic && !C.base.isStateless() && !C.newDP) C.forceA64();

    slmA &= (slmBuffers > 0);
    slmB &= (slmBuffers > 0);

    A.preflight(hw);
    B.preflight(hw);
    C.preflight(hw);
    A_prefetch.preflight(hw);
    B_prefetch.preflight(hw);
    C_prefetch.preflight(hw);

    bool globalCM = isRegisterColMajor(problem.Tc, problem.C, C);

    // Default SIMD setting.
    if (fmaSIMD == 0) {
        fmaSIMD = std::min(32,
                2 * GRF::bytes(hw)
                        / std::max<int>({Ta.size(), Tb.size(), Tc.size()}));
        if (hw == HW::Gen9 && Ta_real.size() == 1 && Tb_real.size() == 1
                && Tc_real.size() == 4)
            fmaSIMD = 32;
    }

    if (problem.batch != BatchMode::None) {
        persistent = false;
        kParallel = false;
    }

    if (coopA == CoopSplit::K && slmATrans) coopA = CoopSplit::MN;
    if (coopB == CoopSplit::K && slmBTrans) coopB = CoopSplit::MN;

    checkBeta1 |= C.atomic && !problem.beta1();

    // Fixed systolic kernel handling.
    if (fixedSystolic) {
        if (wg[LoopM] == 0) wg[LoopM] = 4;
        if (wg[LoopN] == 0) wg[LoopN] = 4;
        bool doubleM = (wg[LoopM] == 8);

        slmCopies = (slmCopies == 3) ? 3 : 1;
        slmBuffers = (splitCopy || doubleM) ? 4 : 3;
        slmA = slmB = true;
        GRFs = 256;
        altCRemainder = false;
        loopOrder[0] = LoopM;
        loopOrder[1] = LoopN;
        loopOrder[2] = LoopK;
        A.accessType = B.accessType = AccessType::Block;
        ka_load = kb_load = 32 / Ta_real;
    }
    dpasw = fixedSystolic;

    // Accumulator usage: 64-bit emulation, or k chaining, or extra C registers, or storage for r0 header.
    // Priority: k chaining > extra C registers > r0 header storage.
    //                         64-bit emulation > r0 header storage.
    if (hw <= HW::Gen9) kChain = 1;
    cAccumulators &= (kChain == 1);

    bool emulateNeedsAcc = emulate.emulate64 || emulate.emulateDWxDW;
    if (moveR0 == MoveR0::Acc)
        if (cAccumulators || emulateNeedsAcc || xParallel || (kChain > 1)
                || barrierFreq)
            moveR0 = MoveR0::None;

    // Mixed mode restrictions:
    //  - mixed hf/f is max SIMD 8 on Gen9
    //  - mixed hf/f is not allowed on Gen12
    //  - mixed bf/f is max SIMD 8 on ATS+
    if ((Tc_real == Type::f32)
            && (Ta_real != Type::f32 || Tb_real != Type::f32))
        fmaSIMD = std::min(fmaSIMD, GRF::bytes(hw) >> 2);

    // No jump table paths use SIMT control flow. Also atomic reductions.
    spf &= !noJumpTables;
    spf &= !C.atomic;

    checkAdd32 &= !emulate.emulate64_add32;

    int opCount = outerProductCount(hw, problem, *this);
    int minOPCount = minOuterProductCount(hw, problem, *this);
    int ukAlign = opCount;

    if (kParallelLocal) moveR0 = MoveR0::None;

    // SLM copy logic.
    int slmVersions = std::max(1, lcm(slmCopies, slmBuffers));
    if (slmBuffers > 0) {
        moveR0 = MoveR0::None;
        barrierFreq = 0;
        if (wg[LoopM] <= 0 || wg[LoopN] <= 0)
            throw std::runtime_error("Workgroup sizes required.");
        if (slmA) ukAlign = lcm(ukAlign, wg[LoopN] * slmVersions);
        if (slmB) ukAlign = lcm(ukAlign, wg[LoopM] * slmVersions);
        slmUseIncrCopy &= (slmCopies == 1);
    }

    // ka/kb_load wranging.
    if (ka_load_masked == 0) ka_load_masked = ka_load;
    if (kb_load_masked == 0) kb_load_masked = kb_load;

    if (!slmA) {
        ka_load = align_up(ka_load, opCount);
        ka_load_masked = align_up(ka_load_masked, minOPCount);
    }
    if (!slmB) {
        kb_load = align_up(kb_load, opCount);
        kb_load_masked = align_up(kb_load_masked, minOPCount);
    }

    // Systolic handling.
    if (systolic) {
        auto params = systolicParams(hw, problem, *this);

        ukAlign = lcm(ukAlign, params.ksys);
        (globalCM ? C.tileR : C.tileC) = params.osys;
        if (unroll[globalCM ? LoopM : LoopN] > params.osys) forceCopyC = true;
    }

    // Prefetch handling.
    cooperativePF &= (prefetchA || prefetchB);

    if (problem.beta0()) prefetchC = 0;

    // Propagate tiling requests to strategy.
    int tileM_A, tileK_A, tileK_B, tileN_B;
    std::tie(tileM_A, tileK_A, tileK_B, tileN_B)
            = targetKernelTiling(hw, problem, *this);
    if (A.accessType != AccessType::Block) {
        if (tileM_A && !A.tileR) A.tileR = tileM_A;
        if (tileK_A && !A.tileC) A.tileC = tileK_A;
    }
    if (B.accessType != AccessType::Block) {
        if (tileK_B && !B.tileR) B.tileR = tileK_B;
        if (tileN_B && !B.tileC) B.tileC = tileN_B;
    }

    // Always use 1D addressing for packed inputs.
    A.address2D &= !isPacked(problem.A.layout);
    B.address2D &= !isPacked(problem.B.layout);

    // k unroll wrangling.
    ukAlign = lcm(ukAlign, A_copies * ka_load);
    ukAlign = lcm(ukAlign, B_copies * kb_load);
    if (slmCopies > 1) {
        ukAlign = lcm(ukAlign, slmCopies * ka_load);
        ukAlign = lcm(ukAlign, slmCopies * kb_load);
    }
    if (ka_pfStride) ukAlign = lcm(ukAlign, ka_pfStride);
    if (kb_pfStride) ukAlign = lcm(ukAlign, kb_pfStride);

    int minUnrollKSLM = 1;
    if (unrollKSLM > 0)
        minUnrollKSLM = unrollKSLM;
    else {
        if (slmA) minUnrollKSLM = lcm(minUnrollKSLM, ka_load);
        if (slmB) minUnrollKSLM = lcm(minUnrollKSLM, kb_load);
    }

    ukAlign = align_up(ukAlign, minUnrollKSLM * slmVersions);

    unroll[LoopK] = align_up(unroll[LoopK], ukAlign);
    barrierFreq = align_up(barrierFreq, unroll[LoopK]);

    if (unrollKSLM == 0) unrollKSLM = unroll[LoopK] / slmVersions;

    if (fixedSystolic) unroll[LoopK] = unrollKSLM = 32 / Ta_real;

    barrierFreq = align_up(barrierFreq, unroll[LoopK]);

    int kChunkA = (problem.A.tileC ? problem.A.tileC : problem.A.crosspack);
    int kChunkB = (problem.B.tileR ? problem.B.tileR : problem.B.crosspack);
    if (unroll[LoopK] <= std::min(kChunkA, kChunkB))
        remHandling[LoopK] = RemainderHandling::Ignore;

    // Default blocking.
    bool isZ = problem.Tc.size() >= 16;
    auto defaultMBlock = isZ ? 2048 : 4096;
    if (hw >= HW::XeHP) defaultMBlock *= 2;
    auto defaultNBlock = defaultMBlock;
    auto defaultMNBlockNonHilbert = defaultMBlock;

    /* No more than (2^16 - 1) workgroups in m/n dimensions for linear orders, plus a huge safety margin. */
    if (linearOrder()) {
        defaultMBlock = 16384 * unroll[LoopM];
        defaultNBlock = 16384 * unroll[LoopN];
    }

    if (blocking[LoopM] <= 0) blocking[LoopM] = defaultMBlock;
    if (blocking[LoopN] <= 0) blocking[LoopN] = defaultNBlock;
    if (blocking[LoopK] <= 0) {
        int points = 1;
        if (slmA || (problem.A.layout != MatrixLayout::T)) points++;
        if (slmB || (problem.B.layout != MatrixLayout::N)) points++;
        blocking[LoopK] = std::min(2048, (2048 * points) / problem.Ta);
    }

    auto defaultBlockAltK = blocking[LoopK];
    if (hw < HW::XeHPC)
        if (hw >= HW::XeHP) defaultBlockAltK = std::min(defaultBlockAltK, 1024);

    if (blockingAlt[LoopM] <= 0) blockingAlt[LoopM] = defaultMNBlockNonHilbert;
    if (blockingAlt[LoopN] <= 0) blockingAlt[LoopN] = defaultMNBlockNonHilbert;
    if (blockingAlt[LoopK] <= 0) blockingAlt[LoopK] = defaultBlockAltK;

    // Default workgroups.
    auto defaultWGX = 2, defaultWGY = 8;

    if (wg[loopOrder[0]] <= 0) wg[loopOrder[0]] = defaultWGX;
    if (wg[loopOrder[1]] <= 0) wg[loopOrder[1]] = defaultWGY;
    if (wg[LoopK] <= 0) {
        if (kParallelLocal)
            wg[LoopK] = (threadsPerEU(hw, *this) * eusPerSubslice(hw))
                    / (wg[LoopM] * wg[LoopN]);
        else
            wg[LoopK] = 1;
    }

    kParallelLocal &= (wg[LoopK] > 1);

    skewLocalIDs &= (wg[LoopM] * wg[LoopN] > eusPerSubslice(hw));

    if (skewLocalIDs) forceWGUpdate = WGFixed;

    avoidIncConflicts &= (hw >= HW::XeHP);

    CommonStrategy::preflight(hw, problem);
}

// Reduce register pressure. Returns true if successful.
bool GEMMStrategy::minimize(HW hw, const GEMMProblem &problem) {
    bool better = false;
    auto minOPCount = minOuterProductCount(hw, problem, *this);
    auto ka_load_best_min = std::max<int>({1, 4 / problem.Ta, minOPCount});
    auto kb_load_best_min = std::max<int>({1, 4 / problem.Tb, minOPCount});

    // Reduce ka/b_load down to suggested minimums (not requiring crosspack)
    if (ka_load > ka_load_best_min) {
        ka_load = ka_load_best_min;
        better = true;
    }
    if (kb_load > kb_load_best_min) {
        kb_load = kb_load_best_min;
        better = true;
    }

    // Reduce A/B copies.
    A_copies = B_copies = 1;

    // Remove k chaining.
    kChain = 1;

    // Reduce k unroll for SLM copies.
    if (slmA || slmB) {
        auto oldUK = unroll[LoopK];
        unroll[LoopK] = 1;
        unrollKSLM = 0;
        preflight(hw, problem);
        better |= (unroll[LoopK] < oldUK);
    }

    if (better) return better;

    // Reduce ka/b_load to absolute minimum if that failed.
    if (ka_load > minOPCount) {
        ka_load = minOPCount;
        better = true;
    }
    if (kb_load > minOPCount) {
        kb_load = minOPCount;
        better = true;
    }

    return better;
}

// Validate a GEMM superkernel strategy, correcting settings as necessary.
void GEMMSuperkernelStrategy::preflight(HW hw, const GEMMProblem &problem) {
    if (substrategies.size() <= 0)
        throw std::runtime_error("No substrategies for superkernel.");
    auto subgroupSize = substrategies[0].subgroupSize;
    for (auto &ss : substrategies) {
        ss.insideSK = true;
        ss.preflight(hw, problem);
        if (ss.subgroupSize != subgroupSize)
            throw std::runtime_error("Incompatible subgroup sizes.");
    }
}

void MatrixAddressingStrategy::preflight(HW hw) {
    newDP |= isBlock2D(accessType);
    if (prefetch && newDP && cachingR == CacheSettingsLSC::Default)
        cachingR = CacheSettingsLSC::L1C_L3C;

    if (accessType == AccessType::ChannelScattered && base.isStateless()
            && !newDP)
        base = AddressBase::createBTS(0);
}

void MatrixAddressingStrategy::forceA64() {
    base = AddressBase::createA64(true);
    if (accessType == AccessType::ChannelScattered && !newDP)
        accessType = AccessType::Scattered;
}

/**********************************************************************/
/*                 Fixed Systolic GEMM (XeHP/XeHPG)                   */
/**********************************************************************/
namespace sysgemm {
static GRFRange A_copy0 = GRF(40) - GRF(47);
static GRFRange B_copy0 = GRF(2) - GRF(13);
static GRFRange A_regs = GRF(48) - GRF(63);
static GRFRange B_regs = GRF(14) - GRF(37);
static GRFRange C_regs = GRF(64) - GRF(255);
static GRFRange A_copy1 = GRF(96) - GRF(103);
static GRFRange B_copy1 = GRF(104) - GRF(111);
static GRFRange A_copy2 = GRF(144) - GRF(151);
static GRFRange B_copy2 = GRF(152) - GRF(159);
static GRFRange A_copy[3] = {A_copy0, A_copy1, A_copy2};
static GRFRange B_copy[3] = {B_copy0, B_copy1, B_copy2};
static GRF addr0 = GRF(1);
static GRF addr1 = GRF(38);
static GRF addr2 = GRF(39);
static GRF addr3 = GRF(0);
static Subregister A_ptr64 = addr1.uq(3);
static Subregister B_ptr64 = addr2.uq(3);
static Subregister C_ptr64 = addr2.uq(2);
static Subregister slmAOffsetLoad = addr1.uw(8); // offsets in OWords
static Subregister slmBOffsetLoad = addr1.uw(9);
static Subregister slmAOffsetStore = addr1.uw(10);
static Subregister slmBOffsetStore = addr1.uw(11);
static Subregister slmAOffsetLoadInit = addr1.uw(6);
static Subregister slmBOffsetLoadInit = addr1.uw(7);
static Subregister slmAOffsetStoreInit = addr2.uw(6);
static Subregister slmBOffsetStoreInit = addr2.uw(7);
static Subregister kCounter = AccumulatorRegister(2).d(0);
static Subregister barrierVal = AddressRegister(0).ud(0);
static constexpr int accStride = 48;
} // namespace sysgemm

template <HW hw>
bool gemm_kernel_generator_t<hw>::sysgemmAccumulateC(
        GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state) {
    using namespace sysgemm;
    auto params = systolicParams(hw, problem, strategy);
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];
    auto wgM = strategy.wg[LoopM];
    auto wgN = strategy.wg[LoopN];
    auto localIDM = state.lidM;
    auto localIDN = state.lidN;
    bool doubleM = (wgM == 8);
    bool surfaceAB = !strategy.A.base.isStateless();
    bool surfaceC = !strategy.C.base.isStateless();

    if (unrollM != 32) stub();
    if (unrollN != 32 && unrollN != 48) stub();
    if (wgM != 4 && wgM != 8) stub();
    if (wgN != 4) stub();
    if (strategy.A.base.getModel() != strategy.B.base.getModel()) stub();
    if (problem.A.layout != MatrixLayout::Pc) stub();
    if (problem.A.crosspack != params.opsPerChan) stub();
    if (problem.A.tileR != params.osys) stub();
    if (problem.A.tileC != params.ksys) stub();
    if (problem.B.layout != MatrixLayout::Pr) stub();
    if (problem.B.crosspack != params.ksys) stub();
    if (problem.B.tileR != 0 || problem.B.tileC != 0) stub();

    state.ra.claim(C_regs);

    // Adjust A/B addresses and SLM offsets.
    auto tempStorage = C_regs[0];
    auto suboffsetA = tempStorage.ud(0);
    auto suboffsetB = tempStorage.ud(1);
    auto tempA = tempStorage.ud(2);
    auto wlidM = tempStorage.uw(6);
    auto tempB = tempStorage.ud(4);
    auto suboffsetBl = tempStorage.ud(5);

    if (doubleM) {
        and_(1, wlidM, localIDM, 3);
        and_(1 | ne | f1[1], null.uw(), localIDM, 4);
    }
    and_(1 | ne | state.flagAP, null.uw(), localIDM, 1);
    mulConstant(1, suboffsetA, localIDN, unrollM * (32 / 4));
    if (doubleM) {
        mulConstant(1, suboffsetB, wlidM, unrollN * (32 / 4));
        mulConstant(1, suboffsetBl, localIDM, unrollN * (32 / 4));
    } else {
        mulConstant(1, suboffsetB, localIDM, unrollN * (32 / 4));
        suboffsetBl = suboffsetB;
    }

    auto A_ptr = A_ptr64, B_ptr = B_ptr64, C_ptr = C_ptr64;
    if (surfaceAB) {
        if (!strategy.A.newDP || !strategy.B.newDP) stub();
        A_ptr = A_ptr.ud();
        B_ptr = B_ptr.ud();
    }
    if (surfaceC) C_ptr = C_ptr.ud();

    eadd(1, A_ptr, state.effA, suboffsetA, strategy, state);
    eadd(1, B_ptr, state.effB, suboffsetBl, strategy, state);
    emov(1, C_ptr, state.effC[0], strategy, state);

    shr(2, suboffsetA(1), suboffsetA(1), 4);

    mul(1, tempA, localIDM, (unrollM * 36) / 16);
    mad(1, tempB, (wgM * unrollM * 36) / 16, localIDN, (unrollN * 32) / 16);

    mov(1, slmAOffsetLoadInit.uw(), tempA.uw());
    add(1 | state.flagAP, slmBOffsetLoadInit.uw(), tempB.uw(),
            (unrollN / 2) * (32 / 16));
    mov(1 | ~state.flagAP, slmBOffsetLoadInit.uw(), tempB.uw());
    add(1, slmAOffsetStoreInit.uw(), tempA.uw(), suboffsetA.uw());
    add(1, slmBOffsetStoreInit.uw(), tempB.uw(), suboffsetB.uw());
    mov(2, slmAOffsetLoad(1), slmAOffsetLoadInit(1));

    // Marshal data needed later into acc2 for safekeeping.
    auto saveData = state.ra.alloc_range(2);
    auto kLoops = saveData[0].d(0);
    auto ldc = saveData[0].ud(1);
    auto flags = saveData[0].ud(2);
    auto k = saveData[0].ud(3);
    auto remM = saveData[0].uw(8);
    auto remN = saveData[0].uw(9);
    auto abo = saveData[0].ud(5);
    auto ao = saveData[0].w(10);
    auto bo = saveData[0].w(11);
    auto alpha = saveData[0].ud(6).reinterpret(0, problem.Ts.ngen());
    auto beta = saveData[0].ud(7).reinterpret(0, problem.Ts.ngen());
    auto remFusedStorage = saveData[1].ud(0);
    auto diagC = saveData[1].ud(1);
    auto effCO = saveData[1].uq(1);
    auto slotAB = saveData[1].ud(4);
    auto effAs = saveData[1].uq(2).reinterpret(0, state.effA.getType());
    auto effBs = saveData[1].uq(3).reinterpret(0, state.effB.getType());

    if (state.r0_info != acc0.ud()) mov<uint32_t>(8, acc0, state.r0_info);

    add(1, kLoops, state.k, params.ksys - 1);
    mov(1, ldc, state.inputs.ldc[0]);
    if (state.inputs.flags.isValid()) mov(1, flags, state.inputs.flags);
    mov(1, k, state.k);
    if (state.remainders[LoopM].isValid())
        mov(1, remM, state.remainders[LoopM]);
    if (state.remainders[LoopN].isValid())
        mov(1, remN, state.remainders[LoopN]);
    if (state.inputs.abo.isValid())
        mov(1, abo, state.inputs.abo);
    else {
        if (state.inputs.ao.isValid()) mov(1, ao, state.inputs.ao);
        if (state.inputs.bo.isValid()) mov(1, bo, state.inputs.bo);
    }
    if (state.inputs.alpha_real.isValid())
        mov(1, alpha, state.inputs.alpha_real);
    if (state.inputs.beta_real.isValid()) mov(1, beta, state.inputs.beta_real);
    shr(1, kLoops, kLoops, log2(params.ksys));
    if (state.remFusedStorage.isValid())
        mov(1, remFusedStorage, state.remFusedStorage);
    if (state.diagC.isValid()) mov(1, diagC, state.diagC);
    if (state.effCO.isValid()) {
        effCO = effCO.reinterpret(0, state.effCO.getType());
        emov(1, effCO, state.effCO, strategy, state);
    }
    if (problem.abOffset != ABOffset::None) {
        state.effAs = effAs;
        state.effBs = effBs;
        gemmCalcABOffsetAddrs(problem, strategy, state);
    }
    if (state.fusedGEMM.slotA.isValid()) {
        if (problem.abOffset != ABOffset::None)
            stub(); // Not enough room in acc2.
        mov(1, slotAB, state.fusedGEMM.slotA.ud());
    }

    releaseSavedMNLocalIDs(state);
    state.ra.safeRelease(state.effA);
    state.ra.safeRelease(state.effB);
    state.ra.safeRelease(state.effC[0]);
    state.ra.safeRelease(state.inputs.lda);
    state.ra.safeRelease(state.inputs.ldb);

    state.ra.release(state.inputs.ldc[0]);
    state.ra.release(state.k);
    state.ra.release(state.remainders[LoopM]);
    state.ra.release(state.remainders[LoopN]);
    state.ra.release(state.inputs.abo);
    state.ra.release(state.inputs.ao);
    state.ra.release(state.inputs.bo);
    state.ra.release(state.inputs.alpha_real);
    state.ra.release(state.inputs.beta_real);
    state.ra.release(state.remFusedStorage);
    state.ra.release(state.diagC);
    state.ra.release(state.effCO);
    state.ra.release(state.fusedGEMM.slotA);
    state.ra.release(state.fusedGEMM.slotB);

    if (state.r0_info.isARF()) stub();
    GRF r0_info {state.r0_info.getBase()};
    if (hw >= HW::XeHPG) {
        mov(1, barrierVal.uw(0), Immediate::uw(0));
        mov(2, barrierVal.ub(2)(1), r0_info.ub(11)(0));
    } else
        and_(1, barrierVal, r0_info.ud(2), 0x7F000000);

    mov<float>(16, acc2, saveData[0]);

    sync.nop(SWSB<AllPipes>(1));

    if (!doubleM)
        sysgemmKLoop(problem, strategy, state);
    else {
        Label oddB, done;
        jmpi(1 | f1[1], oddB);
        sysgemmKLoop4(problem, strategy, state, false);
        jmpi(1, done);
        mark(oddB);
        sysgemmKLoop4(problem, strategy, state, true);
        mark(done);
    }

    mov<float>(16, saveData[0], acc2);

    state.effC[0] = C_ptr;
    state.inputs.ldc[0] = ldc;
    if (state.inputs.flags.isValid()) state.inputs.flags = flags;
    state.k = k;
    if (state.remainders[LoopM].isValid()) state.remainders[LoopM] = remM;
    if (state.remainders[LoopN].isValid()) state.remainders[LoopN] = remN;
    if (state.inputs.abo.isValid()) state.inputs.abo = abo;
    if (state.inputs.ao.isValid()) state.inputs.ao = ao;
    if (state.inputs.bo.isValid()) state.inputs.bo = bo;
    if (state.inputs.alpha_real.isValid()) {
        state.inputs.alpha_real = alpha;
        if (!problem.alpha_real.fixed()) problem.alpha_real = alpha;
    }
    if (state.inputs.beta_real.isValid()) {
        state.inputs.beta_real = beta;
        if (!problem.beta_real.fixed()) problem.beta_real = beta;
    }
    if (state.remFusedStorage.isValid()) {
        state.remFusedStorage = remFusedStorage;
        state.remaindersFused[LoopM] = state.remainders[LoopM];
        state.remaindersFused[LoopN] = state.remainders[LoopN];
        state.remaindersFused[strategy.fusedLoop] = remFusedStorage;
    }
    if (state.diagC.isValid()) state.diagC = diagC;
    if (state.effCO.isValid()) state.effCO = effCO;
    if (state.fusedGEMM.slotA.isValid()) {
        state.fusedGEMM.slotA = slotAB.uw(0);
        state.fusedGEMM.slotB = slotAB.uw(1);
    }

    state.ra.claim(C_ptr);

    // Set up C internal layout and registers.
    state.C_regs.resize(1);
    state.C_regs[0] = C_regs;
    state.C_layout.clear();
    state.C_layout.reserve((unrollM / 8) * (unrollN / 4));
    for (int j0 = 0; j0 < unrollN; j0 += 4) {
        for (int i0 = 0; i0 < unrollM; i0 += 8) {
            RegisterBlock block;
            block.log2GRFBytes = GRF::log2Bytes(hw);
            block.colMajor = true;
            block.splitComplex = false;
            block.nr = block.ld = 8;
            block.nc = 4;
            block.offsetR = i0;
            block.offsetC = j0;
            block.crosspack = 1;
            block.bytes = 8 * 4 * problem.Tc.size();
            block.simdSize = 0;

            int j0Interleaved = j0 << 1;
            if (j0Interleaved >= unrollN) j0Interleaved += 4 - unrollN;

            block.offsetBytes
                    = (accStride * i0 / 8 + j0Interleaved) * GRF::bytes(hw);
            state.C_layout.push_back(block);
        }
    }

    // Set up C external layout.
    state.copyC = true;
    bool remM_Ce, remN_Ce;
    getCRemainders(problem, strategy, remM_Ce, remN_Ce);

    if (!getRegLayout(problem.Tc_ext, state.C_layoutExt, unrollM, unrollN,
                remM_Ce, remN_Ce, true, false, 0, 0, problem.C,
                state.Cext_strategy))
        return false;
    if (remM_Ce || remN_Ce)
        (void)getRegLayout(problem.Tc_ext, state.C_layoutExtUnmasked, unrollM,
                unrollN, false, false, true, false, 0, 0, problem.C,
                state.Cext_strategy);

    if (state.r0_info != acc0.ud()) mov<uint32_t>(8, state.r0_info, acc0);

    return true; // Success!
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemmKLoop(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state) {
    using namespace sysgemm;
    Label top, bottom, skipMain, remTop, remBottom;

    auto nbBarrierWait = [&]() {
        if (!strategy.slmAltBarriers) barrierwait();
    };
    auto nbStoreSignal = [&](bool forceFence = false) {
        if (!strategy.slmAltBarriers)
            sysgemmStoreSignal(problem, strategy, state, forceFence);
    };
    auto storeSignal = [&](bool forceFence = false) {
        sysgemmStoreSignal(problem, strategy, state, forceFence);
    };
    auto copyLoad = [&](int storeBuffer, bool useC = false) {
        sysgemmCopyLoad(problem, strategy, state, storeBuffer, useC);
    };
    auto copyStore = [&](int storeBuffer, bool first = false) {
        sysgemmCopyStore(problem, strategy, state, storeBuffer, first);
    };
    auto multiply = [&](int buffer, bool lastMultiply = false) {
        sysgemmMultiply(problem, strategy, state, buffer, lastMultiply);
    };

    bool oldDefaultAutoSWSB = getDefaultAutoSWSB();
    setDefaultAutoSWSB(false);

    if (strategy.slmCopies == 1) {
        cmp(1 | lt | f1[1], kCounter, 3);
        add(1 | le | f0[1], kCounter, kCounter, -5);

        jmpi(1 | f1[1], skipMain);

        copyLoad(0, true); // L0 -> C
        copyLoad(1); // L1
        copyStore(0, true); // S0 <- C
        storeSignal(true); // Signal 0 ready
        zeroMatrix(C_regs, strategy);
        sync.nop(SWSB<AllPipes>(1));
        copyStore(1); // S1

        nbBarrierWait(); // Wait 0 ready
        nbStoreSignal(); // Signal 1 ready

        jmpi(1 | f0[1], bottom); // Zero-trip loop check

        mark(top);
        add(1 | gt | f0[1], kCounter, kCounter, -3);

        copyLoad(2); // L2
        multiply(0); // M0
        nbBarrierWait(); // Wait 0 ready
        copyStore(2); // S2
        nbStoreSignal(); // Signal 2 ready

        copyLoad(0); // L0
        multiply(1); // M1
        nbBarrierWait(); // Wait 2 ready
        copyStore(0); // S0
        nbStoreSignal(); // Signal 0 ready

        copyLoad(1); // L1
        multiply(2); // M2
        nbBarrierWait(); // Wait 0 ready
        copyStore(1); // S1
        nbStoreSignal(); // Signal 1 ready

        jmpi(1 | f0[1], top);
        mark(bottom);

        copyLoad(2); // L2
        multiply(0); // M0
        nbBarrierWait(); // Wait 1 ready
        copyStore(2); // S2
        nbStoreSignal(); // Signal 2 ready

        multiply(1); // M1

        nbBarrierWait(); // Wait 2 ready

        multiply(2, true); // M2

        add(1 | le | f0[1], kCounter, kCounter, 2);
        jmpi(1 | f0[1], remBottom);
        jmpi(1, remTop);

        mark(skipMain);

        zeroMatrix(C_regs, strategy);
        add(1, kCounter, kCounter, 5);

        mov(2, slmAOffsetStore(1), slmAOffsetStoreInit(1));
        sync.nop(SWSB<AllPipes>(1));

        mark(remTop);

        cmp(1 | lt | f0[1], kCounter, 2);
        copyLoad(0);
        copyStore(0);
        storeSignal(true);
        nbBarrierWait();
        multiply(0, true);

        jmpi(1 | f0[1], remBottom);
        copyLoad(1);
        copyStore(1);
        storeSignal(true);
        nbBarrierWait();
        multiply(1, true);

        mark(remBottom);
    } else if (strategy.slmCopies == 3) {
        // Triple-buffered global memory load + SLM pipeline.
        cmp(1 | lt | f1[1], kCounter, 4);
        add(1 | le | f0[1], kCounter, kCounter, -6);

        jmpi(1 | f1[1], skipMain);

        copyLoad(0); // L0
        copyLoad(1); // L1
        copyLoad(2); // L2
        copyStore(0, true); // S0
        storeSignal(true); // Signal 0 ready
        zeroMatrix(C_regs, strategy);
        copyLoad(0); // L0
        sync.nop(SWSB<uint32_t>(1));
        copyStore(1); // S1

        nbBarrierWait(); // Wait 0 ready
        nbStoreSignal(); // Signal 1 ready

        jmpi(1 | f0[1], bottom); // Zero-trip loop check

        mark(top);
        add(1 | gt | f0[1], kCounter, kCounter, -3);

        copyLoad(1); // L1
        multiply(0); // M0
        nbBarrierWait(); // Wait 0 ready
        copyStore(2); // S2
        nbStoreSignal(); // Signal 2 ready

        copyLoad(2); // L2
        multiply(1); // M1
        nbBarrierWait(); // Wait 2 ready
        copyStore(0); // S0
        nbStoreSignal(); // Signal 0 ready

        copyLoad(0); // L0
        multiply(2); // M2
        nbBarrierWait(); // Wait 0 ready
        copyStore(1); // S1
        nbStoreSignal(); // Signal 1 ready

        jmpi(1 | f0[1], top);
        mark(bottom);

        multiply(0); // M0
        nbBarrierWait(); // Wait 1 ready
        copyStore(2); // S2
        nbStoreSignal(); // Signal 2 ready

        multiply(1); // M1
        nbBarrierWait(); // Wait 2 ready
        copyStore(0); // S0
        nbStoreSignal(); // Signal 0 ready

        multiply(2); // M2

        nbBarrierWait(); // Wait 0 ready

        multiply(0, true); // M0

        add(1 | le | f0[1], kCounter, kCounter, 2);
        jmpi(1 | f0[1], remBottom);
        jmpi(1, remTop);

        mark(skipMain);

        zeroMatrix(C_regs, strategy);
        add(1 | le | f0[1], kCounter, kCounter, 5);

        mov(2, slmAOffsetStore(1), slmAOffsetStoreInit(1));
        sync.nop(SWSB<uint32_t>(1));

        copyLoad(0);
        copyStore(0);
        storeSignal(true);
        nbBarrierWait();
        multiply(0, true);

        jmpi(1 | f0[1], remBottom);

        mark(remTop);

        cmp(1 | lt | f0[1], kCounter, 2);

        copyLoad(1);
        copyStore(1);
        storeSignal(true);
        nbBarrierWait();
        multiply(1, true);

        jmpi(1 | f0[1], remBottom);

        copyLoad(2);
        copyStore(2);
        storeSignal(true);
        nbBarrierWait();
        multiply(2, true);

        mark(remBottom);
    } else
        stub();

    sync.allwr();
    setDefaultAutoSWSB(oldDefaultAutoSWSB);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemmKLoop4(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state, bool oddB) {
    using namespace sysgemm;
    auto &depAddr = state.sysgemm.depAddr;

    Label top, bottom, skipMain, done;
    Label skipLoad0, skipLoad1, skipLoad2;
    Label skipStore0, skipStore1, skipStore2;
    Label sskipLoad12, sskipStore1, sskipStore2Load3, sskipStore3;

    auto clearDepAddr = [&]() {
        for (int i = 0; i < 4; i++)
            depAddr[i] = InstructionModifier();
    };
    auto storeSignal
            = [&]() { sysgemmStoreSignal(problem, strategy, state, true); };
    auto copyLoad = [&](int storeBuffer, int useC = 0, bool forceLoadB = false,
                            RegData flagLoadB = RegData()) {
        sysgemmCopyLoad4(problem, strategy, state, storeBuffer,
                ((storeBuffer & 1) != oddB) || forceLoadB, useC, flagLoadB);
    };
    auto copyLoadRem = [&](int storeBuffer, RegData flagLoadB) {
        sysgemmCopyLoad4(problem, strategy, state, storeBuffer,
                (storeBuffer & 1) != oddB, 0, flagLoadB);
    };
    auto copyStore = [&](int storeBuffer, int useC = 0, int useC_B = 0) {
        sysgemmCopyStore4(problem, strategy, state, storeBuffer,
                (storeBuffer & 1) == oddB, useC, useC_B);
    };
    auto multiply = [&](int buffer, bool firstMultiply = false) {
        sysgemmMultiply4(problem, strategy, state, buffer, firstMultiply);
    };
    auto multiplyRem = [&](int buffer, RegData flagWaitLoad, RegData flagSignal,
                               bool firstMultiply = false) {
        sysgemmMultiply4(problem, strategy, state, buffer, firstMultiply,
                flagWaitLoad, flagSignal, &done);
    };
    auto slmRead = [&]() {
        mov(1 | depAddr[0], addr0.ud(2), slmAOffsetLoad);
        mov(1 | depAddr[1], addr1.ud(2), slmBOffsetLoad);
        add(1 | depAddr[2], addr2.ud(2), slmBOffsetLoad, 8 * 32 / 16);
        add(1 | depAddr[3], addr3.ud(2), slmBOffsetLoad, 16 * 32 / 16);

        load(16 | SWSB<AllPipes>(sb3, 4), A_regs[0], block_oword(16), SLM,
                addr0);
        load(16 | SWSB<AllPipes>(sb0, 3), B_regs[0], block_oword(16), SLM,
                addr1);
        load(16 | SWSB<AllPipes>(sb1, 2), B_regs[8], block_oword(16), SLM,
                addr2);
        load(16 | SWSB<AllPipes>(sb2, 1), B_regs[16], block_oword(16), SLM,
                addr3);
        depAddr[0] = sb3.src;
        depAddr[1] = sb0.src;
        depAddr[2] = sb1.src;
        depAddr[3] = sb2.src;

        add(1 | depAddr[0], addr0.ud(2), slmAOffsetLoad, 8 * 32 / 16);
        load(16 | SWSB<AllPipes>(sb4, 1), A_regs[8], block_oword(16), SLM,
                addr0);
        depAddr[0] = sb4.src;
    };

    bool oldDefaultAutoSWSB = getDefaultAutoSWSB();
    setDefaultAutoSWSB(false);

    clearDepAddr();
    mov(1, f1.ud(), 0);
    mov(1, f0.ud(), 0);
    cmp(1 | lt | f1[1], kCounter, 4);
    add(1 | le | f0[1], kCounter, kCounter, -7);

    jmpi(1 | f1[1], skipMain);

    status << "Main path, " << (oddB ? "odd B" : "even B")
           << status_stream::endl;

    copyLoad(0, 1, true); // L0 -> C1
    copyLoad(1, 2); // L1 -> C2
    copyLoad(2); // L2
    copyStore(0, 1, 1); // S0 <- C1
    storeSignal();
    copyStore(1, 2, 1); // S1 <- C2
    barrierwait();
    slmRead();
    storeSignal();
    copyStore(2, 0, 2); // S2
    if (!oddB) sync.allrd(0x3000);
    zeroMatrix(C_regs, strategy);
    sync.allrd(SWSB<AllPipes>(1));

    jmpi(1 | f0[1], bottom); // Zero-trip loop check

    depAddr[0] = sb8.src;
    depAddr[1] = !oddB ? sb9.src : sb0.src;
    depAddr[2] = !oddB ? sb10.src : sb4.src;
    depAddr[3] = sb3.src;

    mark(top);
    add(1 | gt | f0[1], kCounter, kCounter, -4);

    copyLoad(3);
    multiply(0);
    copyStore(3);

    copyLoad(0);
    multiply(1);
    copyStore(0);

    copyLoad(1);
    multiply(2);
    copyStore(1);

    copyLoad(2);
    multiply(3);
    copyStore(2);

    jmpi(1 | f0[1], top);
    mark(bottom);

    cmp(1 | gt | f0[0], kCounter, -4 + 1);
    cmp(1 | gt | f0[1], kCounter, -4 + 2);
    cmp(1 | gt | f1[0], kCounter, -4 + 3);

    copyLoadRem(3, f0[0]);
    multiply(0);
    copyStore(3);

    sync.allrd();
    jmpi(1 | ~f0[0], skipLoad0);
    copyLoadRem(0, f0[1]);
    mark(skipLoad0);
    multiply(1);
    sync.allrd();
    jmpi(1 | ~f0[0], skipStore0);
    copyStore(0);
    mark(skipStore0);

    sync.allrd();
    jmpi(1 | ~f0[1], skipLoad1);
    copyLoadRem(1, f1[0]);
    mark(skipLoad1);
    multiplyRem(2, FlagRegister(), f0[0]);
    sync.allrd();
    jmpi(1 | ~f0[1], skipStore1);
    copyStore(1);
    mark(skipStore1);

    sync.allrd();
    jmpi(1 | ~f1[0], skipLoad2);
    copyLoadRem(2, null);
    mark(skipLoad2);
    multiplyRem(3, f0[0], f0[1]);
    sync.allrd();
    jmpi(1 | ~f1[0], skipStore2);
    copyStore(2);
    mark(skipStore2);

    multiplyRem(0, f0[1], f1[0]);
    multiplyRem(1, f1[0], null);
    multiplyRem(2, null, null);

    jmpi(1, done);

    status << "Small-k path, " << (oddB ? "odd B" : "even B")
           << status_stream::endl;

    clearDepAddr();
    mark(skipMain);

    // Short loops: special case for 1-4 unrolls
    cmp(1 | gt | f0[0], kCounter, -7 + 1);
    cmp(1 | gt | f0[1], kCounter, -7 + 2);
    cmp(1 | gt | f1[0], kCounter, -7 + 3);

    auto flagLoadB0 = oddB ? f0[0] : FlagRegister();
    copyLoad(0, 1, true, flagLoadB0);
    sync.allrd();
    jmpi(1 | ~f0[0], sskipLoad12);
    copyLoad(1, 2, false, f0[1]);
    sync.allrd();
    jmpi(1 | ~f0[1], sskipLoad12);
    copyLoadRem(2, f1[0]);
    mark(sskipLoad12);
    copyStore(0, 1, 1);
    storeSignal();
    sync.allrd();
    jmpi(1 | ~f0[0], sskipStore1);
    copyStore(1, 2, 1);
    mark(sskipStore1);
    barrierwait();
    slmRead();
    sync.allrd();
    jmpi(1 | ~f0[0], sskipStore2Load3);
    storeSignal();
    sync.allrd();
    jmpi(1 | ~f0[1], sskipStore2Load3);
    copyStore(2, 0, 2);
    sync.allrd();
    jmpi(1 | ~f1[0], sskipStore2Load3);
    copyLoadRem(3, null);
    mark(sskipStore2Load3);
    multiplyRem(0, f0[0], f0[1], true);
    jmpi(1 | ~f0[0], done);
    sync.allrd();
    jmpi(1 | ~f1[0], sskipStore3);
    copyStore(3);
    mark(sskipStore3);
    multiplyRem(1, f0[1], f1[0]);
    multiplyRem(2, f1[0], null);
    multiplyRem(3, null, null);

    mark(done);

    sync.allwr();
    setDefaultAutoSWSB(oldDefaultAutoSWSB);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemmStoreSignal(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state, bool forceFence) {
    using namespace sysgemm;
    auto &depAddr = state.sysgemm.depAddr;

    if (!strategy.slmAltBarriers || forceFence) {
        // Signal SLM data ready once memory fence returns, asynchronously
        sync.nop(depAddr[0]);
        sysgemmBarrierPrep(depAddr[3], addr3);

        slmfence(SWSB<AllPipes>(sb15, 1), addr0);
        barriermsg(sb15, addr3);
        depAddr[0] = InstructionModifier();
        depAddr[3] = sb15.src;
    } else {
        sysgemmBarrierPrep(depAddr[3], addr3);
        barriermsg(SWSB<AllPipes>(sb15, 1), addr3);
        depAddr[3] = sb15.src;
    }
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemmCopyLoad(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state, int storeBuffer,
        bool useC) {
    using namespace sysgemm;
    auto &depAddr = state.sysgemm.depAddr;

    bool surface = !strategy.A.base.isStateless();
    bool emulate64 = strategy.emulate.emulate64;
    int unrollM = strategy.unroll[LoopM];
    int unrollN = strategy.unroll[LoopN];

    auto A_ptr = A_ptr64, B_ptr = B_ptr64;
    if (surface) {
        A_ptr = A_ptr.ud();
        B_ptr = B_ptr.ud();
        emulate64 = false;
    }

    // Load new A and B and increment load pointers.
    if (surface) {
        sync(SyncFunction::nop, SWSB<uint32_t>(1));
        mov(1 | depAddr[0], addr0.ud(0), A_ptr);
        mov(1 | depAddr[1], addr1.ud(0), B_ptr);
        add(1 | depAddr[2], addr2.ud(0), B_ptr, 8 * 32);
    } else if (!emulate64) {
        sync(SyncFunction::nop, SWSB<uint64_t>(1));
        mov(1 | depAddr[0], addr0.uq(0), A_ptr);
        mov(1 | depAddr[1], addr1.uq(0), B_ptr);
        add(1 | depAddr[2], addr2.uq(0), B_ptr, 8 * 32);
    } else {
        sync(SyncFunction::nop, SWSB<uint32_t>(1));
        mov(1 | depAddr[2], addr2.ud(1), B_ptr.ud(1));
        add(1 | ov | f1[1], addr2.ud(0), B_ptr.ud(0), 8 * 32);
        mov(2 | depAddr[0], addr0.ud(0)(1), A_ptr.ud()(1));
        mov(2 | depAddr[1], addr1.ud(0)(1), B_ptr.ud()(1));
        add(1 | f1[1] | SWSB(4), addr2.ud(1), addr2.ud(1), 1);
    }

    if (useC) {
        if (surface) {
            load(1 | SWSB<AllPipes>(sb11, 3), C_regs[0],
                    D64T(32) | strategy.A.cachingR, strategy.A.base, addr0);
            load(1 | SWSB<AllPipes>(sb12, 2), C_regs[8],
                    D64T(32) | strategy.B.cachingR, strategy.B.base, addr1);
            if (strategy.unroll[LoopN] > 32)
                load(1 | SWSB<AllPipes>(sb13, 1), C_regs[16],
                        D64T(16) | strategy.B.cachingR, strategy.B.base, addr2);
        } else {
            load(16 | SWSB<AllPipes>(sb11, 3), C_regs[0], block_hword(8), A64,
                    addr0);
            load(16 | SWSB<AllPipes>(sb12, 2), C_regs[8], block_hword(8), A64,
                    addr1);
            if (strategy.unroll[LoopN] > 32)
                load(16 | SWSB<AllPipes>(sb13, 1), C_regs[16], block_hword(4),
                        A64, addr2);
        }
        depAddr[0] = sb11.src;
        depAddr[1] = sb12.src;
        if (strategy.unroll[LoopN] > 32) depAddr[2] = sb13.src;
        if (strategy.simulation) sync.allrd(0x3000);
    } else {
        // Stronger than necessary dependencies... can load as soon as prev. store inputs are read.
        int loadBuffer = (strategy.slmCopies == 3) ? storeBuffer : 0;
        int t0 = 8 + loadBuffer * 2;
        SBID token0 {t0}, token1 {t0 + 1}, token2 {t0 + 2};

        if (surface) {
            load(1 | SWSB<AllPipes>(token0, 3), A_copy[loadBuffer][0],
                    D64T(32) | strategy.A.cachingR, strategy.A.base, addr0);
            load(1 | SWSB<AllPipes>(token1, 2), B_copy[loadBuffer][0],
                    D64T(32) | strategy.B.cachingR, strategy.B.base, addr1);
            if (strategy.unroll[LoopN] > 32)
                load(1 | SWSB<AllPipes>(token2, 1), B_copy[loadBuffer][8],
                        D64T(16) | strategy.B.cachingR, strategy.B.base, addr2);
        } else {
            load(16 | SWSB<AllPipes>(token0, 3), A_copy[loadBuffer][0],
                    block_hword(8), A64, addr0);
            load(16 | SWSB<AllPipes>(token1, 2), B_copy[loadBuffer][0],
                    block_hword(8), A64, addr1);
            if (strategy.unroll[LoopN] > 32)
                load(16 | SWSB<AllPipes>(token2, 1), B_copy[loadBuffer][8],
                        block_hword(4), A64, addr2);
        }
        depAddr[0] = token0.src;
        depAddr[1] = token1.src;
        if (strategy.unroll[LoopN] > 32) depAddr[2] = token2.src;
        if (strategy.simulation) sync.allrd(0x6 << t0);
    }

    if (!emulate64) {
        add(1 | SWSB(3), A_ptr, A_ptr, unrollM * 32);
        add(1 | SWSB(3), B_ptr, B_ptr, unrollN * 32);
    } else {
        add(1 | ov | f1[0] | SWSB(3), A_ptr.ud(0), A_ptr.ud(0), unrollM * 32);
        add(1 | ov | f1[1] | SWSB(3), B_ptr.ud(0), B_ptr.ud(0), unrollN * 32);
        add(1 | f1[0], A_ptr.ud(1), A_ptr.ud(1), 1);
        add(1 | f1[1], B_ptr.ud(1), B_ptr.ud(1), 1);
    }
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemmCopyLoad4(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state, int storeBuffer,
        bool loadB, int useC, RegData flagLoadB) {
    using namespace sysgemm;
    auto &depAddr = state.sysgemm.depAddr;

    bool emulate64 = strategy.emulate.emulate64;
    bool surface = !strategy.A.base.isStateless();
    int unrollM = strategy.unroll[LoopM];
    int unrollN = strategy.unroll[LoopN];
    int x48 = (unrollN > 32);
    InstructionModifier loadBMod;
    loadB &= !(flagLoadB.isValid() && flagLoadB.isNull());
    if (flagLoadB.isValid())
        loadBMod = loadBMod | static_cast<FlagRegister &>(flagLoadB) | any16h;

    // Load new A and B and increment load pointers.
    auto A_ptr = A_ptr64, B_ptr = B_ptr64;
    if (surface) {
        A_ptr = A_ptr.ud();
        B_ptr = B_ptr.ud();
        emulate64 = false;
    }

    if (surface) {
        sync.nop(SWSB(Pipe::I, 1));
        mov(1 | depAddr[0], addr0.ud(0), A_ptr);
        if (loadB) {
            mov(1 | depAddr[1], addr1.ud(0), B_ptr);
            add(1 | depAddr[2], addr2.ud(0), B_ptr, 8 * 32);
        }
    } else if (!emulate64) {
        sync.nop(SWSB(Pipe::L, 1));
        mov(1 | depAddr[0], addr0.uq(0), A_ptr);
        if (loadB) {
            mov(1 | depAddr[1], addr1.uq(0), B_ptr);
            add(1 | depAddr[2], addr2.uq(0), B_ptr, 8 * 32);
        }
    } else {
        sync.nop(SWSB(Pipe::I, 1));
        if (loadB) {
            mov(1 | depAddr[2], addr2.ud(1), B_ptr.ud(1));
            add(1 | ov | f1[1], addr2.ud(0), B_ptr.ud(0), 8 * 32);
        }
        mov(2 | depAddr[0], addr0.ud(0)(1), A_ptr.ud()(1));
        if (loadB) {
            mov(2 | depAddr[1], addr1.ud(0)(1), B_ptr.ud()(1));
            add(1 | f1[1], addr2.ud(1), addr2.ud(1), 1);
        }
    }

    SBID tokenA(0), tokenB0(0), tokenB1(0);
    GRF dstA, dstB0, dstB1;

    if (useC) {
        tokenA = SBID((useC == 1) ? 5 : 11);
        tokenB0 = SBID((useC == 1) ? 6 : 12);
        tokenB1 = SBID((useC == 1) ? 7 : 13);
        int C_off = (useC == 1) ? 0 : 20;
        dstA = C_regs[C_off + 0];
        dstB0 = C_regs[C_off + 8];
        dstB1 = C_regs[C_off + 16];
    } else {
        // Stronger than necessary dependencies... can load as soon as prev. store inputs are read.
        int loadBuffer = (strategy.slmCopies == 3) ? storeBuffer : 0;
        int t0 = 8 + loadBuffer * 2;
        tokenA = SBID(t0 + 0);
        tokenB0 = SBID(t0 + 1);
        tokenB1 = SBID(t0 + 2);
        dstA = A_copy[loadBuffer][0];
        dstB0 = B_copy[loadBuffer][0];
        dstB1 = B_copy[loadBuffer][8];
    }

    if (surface) {
        load(1 | tokenA | SWSB<AllPipes>(1 + loadB * (1 + x48)), dstA,
                D64T(32) | strategy.A.cachingR, strategy.A.base, addr0);
        if (loadB) {
            load(1 | tokenB0 | loadBMod | SWSB<AllPipes>(1 + x48), dstB0,
                    D64T(32) | strategy.B.cachingR, strategy.B.base, addr1);
            if (x48)
                load(1 | tokenB1 | loadBMod | SWSB<AllPipes>(1), dstB1,
                        D64T(16) | strategy.B.cachingR, strategy.B.base, addr2);
        }
    } else {
        load(16 | tokenA | SWSB<AllPipes>(1 + loadB * (1 + x48)), dstA,
                block_hword(8), A64, addr0);
        if (loadB) {
            load(16 | tokenB0 | loadBMod | SWSB<AllPipes>(1 + x48), dstB0,
                    block_hword(8), A64, addr1);
            if (x48)
                load(16 | tokenB1 | loadBMod | SWSB<AllPipes>(1), dstB1,
                        block_hword(4), A64, addr2);
        }
    }
    depAddr[0] = tokenA.src;
    if (loadB) {
        depAddr[1] = tokenB0.src;
        if (x48) depAddr[2] = tokenB1.src;
    }
    if (strategy.simulation) {
        uint16_t tmask = (1 << tokenA.getID());
        if (loadB) tmask |= (1 << tokenB0.getID()) | (1 << tokenB1.getID());
        sync.allrd(tmask);
    }

    if (!emulate64) {
        add(1, A_ptr, A_ptr, unrollM * 32);
        if (loadB) add(1, B_ptr, B_ptr, 2 * unrollN * 32);
    } else {
        add(1 | ov | f1[1], A_ptr.ud(0), A_ptr.ud(0), unrollM * 32);
        if (loadB)
            add(1 | ov | f1[1] | M8, B_ptr.ud(0), B_ptr.ud(0),
                    2 * unrollN * 32);
        add(1 | f1[1], A_ptr.ud(1), A_ptr.ud(1), 1);
        if (loadB) add(1 | f1[1] | M8, B_ptr.ud(1), B_ptr.ud(1), 1);
    }
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemmCopyStore(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state, int storeBuffer,
        bool first) {
    using namespace sysgemm;
    auto &depAddr = state.sysgemm.depAddr;

    auto aoffset = first ? slmAOffsetStoreInit : slmAOffsetStore;
    auto boffset = first ? slmBOffsetStoreInit : slmBOffsetStore;

    // Store A and B and advance store pointers to next buffer.
    mov(1 | depAddr[0], addr0.ud(2), aoffset);
    mov(1 | depAddr[1], addr1.ud(2), boffset);
    add(1 | depAddr[2], addr2.ud(2), boffset, 8 * 32 / 16);

    if (first && strategy.slmCopies == 1) {
        store(16 | SWSB<AllPipes>(sb11, 3), block_oword(16), SLM, addr0,
                C_regs[0]);
        store(16 | SWSB<AllPipes>(sb12, 2), block_oword(16), SLM, addr1,
                C_regs[8]);
        if (strategy.unroll[LoopN] > 32)
            store(16 | SWSB<AllPipes>(sb13, 1), block_oword(8), SLM, addr2,
                    C_regs[16]);
        depAddr[0] = sb11.src;
        depAddr[1] = sb12.src;
        if (strategy.unroll[LoopN] > 32) depAddr[2] = sb13.src;
        if (strategy.simulation) sync.allrd(0x3000);
    } else {
        int loadBuffer = (strategy.slmCopies == 3) ? storeBuffer : 0;
        int t0 = 8 + loadBuffer * 2;
        SBID token0 {t0}, token1 {t0 + 1}, token2 {t0 + 2};

        store(16 | SWSB<AllPipes>(token0, 3), block_oword(16), SLM, addr0,
                A_copy[loadBuffer][0]);
        store(16 | SWSB<AllPipes>(token1, 2), block_oword(16), SLM, addr1,
                B_copy[loadBuffer][0]);
        if (strategy.unroll[LoopN] > 32)
            store(16 | SWSB<AllPipes>(token2, 1), block_oword(8), SLM, addr2,
                    B_copy[loadBuffer][8]);
        depAddr[0] = token0.src;
        depAddr[1] = token1.src;
        if (strategy.unroll[LoopN] > 32) depAddr[2] = token2.src;
        if (strategy.simulation) sync.allrd(0x6 << t0);
    }

    if (storeBuffer == 2)
        mov(2, slmAOffsetStore(1), slmAOffsetStoreInit(1));
    else
        add(2, slmAOffsetStore(1), aoffset(1),
                strategy.slmSysgemmBlockSize() / 16);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemmCopyStore4(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state, int storeBuffer,
        bool storeB, int useC, int useC_B) {
    using namespace sysgemm;
    auto &depAddr = state.sysgemm.depAddr;
    bool first = (useC == 1);
    bool x48 = (strategy.unroll[LoopN] > 32);

    auto aoffset = first ? slmAOffsetStoreInit : slmAOffsetStore;
    auto boffset = first ? slmBOffsetStoreInit : slmBOffsetStore;

    // Store A and B and advance store pointers to next buffer.
    mov(1 | depAddr[0], addr0.ud(2), aoffset);
    if (storeB) {
        mov(1 | depAddr[1], addr1.ud(2), boffset);
        if (x48) add(1 | depAddr[2], addr2.ud(2), boffset, 8 * 32 / 16);
    }

    int loadBuffer = (strategy.slmCopies == 3) ? storeBuffer : 0;
    int t0 = 8 + loadBuffer * 2;
    auto tokenA = SBID(t0 + 0);
    auto tokenB0 = SBID(t0 + 1);
    auto tokenB1 = SBID(t0 + 2);
    auto srcA = A_copy[loadBuffer][0];
    auto srcB0 = B_copy[loadBuffer][0];
    auto srcB1 = B_copy[loadBuffer][8];

    if (useC) {
        tokenA = SBID((useC == 1) ? 5 : 11);
        int C_off = (useC == 1) ? 0 : 20;
        srcA = C_regs[C_off + 0];
    }

    if (useC_B) {
        tokenB0 = SBID((useC_B == 1) ? 6 : 12);
        tokenB1 = SBID((useC_B == 1) ? 7 : 13);
        int C_off = (useC_B == 1) ? 0 : 20;
        srcB0 = C_regs[C_off + 8];
        srcB1 = C_regs[C_off + 16];
    }

    store(16 | tokenA | SWSB<AllPipes>(1 + storeB * (1 + x48)), block_oword(16),
            SLM, addr0, srcA);
    if (storeB) {
        store(16 | tokenB0 | SWSB<AllPipes>(1 + x48), block_oword(16), SLM,
                addr1, srcB0);
        if (x48)
            store(16 | tokenB1 | SWSB<AllPipes>(1), block_oword(8), SLM, addr2,
                    srcB1);
    }

    depAddr[0] = tokenA.src;
    if (storeB) {
        depAddr[1] = tokenB0.src;
        if (x48) depAddr[2] = tokenB1.src;
    }
    if (strategy.simulation) {
        uint16_t tmask = (1 << tokenA.getID());
        if (storeB) tmask |= (1 << tokenB0.getID()) | (1 << tokenB1.getID());
        sync.allrd(tmask);
    }

    if (storeBuffer == 3)
        mov(2, slmAOffsetStore(1), slmAOffsetStoreInit(1));
    else
        add(2, slmAOffsetStore(1), aoffset(1),
                strategy.slmSysgemmBlockSize() / 16);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemmMultiply(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state, int buffer,
        bool lastMultiply) {
    using namespace sysgemm;
    auto &depAddr = state.sysgemm.depAddr;

    // Load half of A (16x32) -- hopefully broadcast from SLM to this row -- and half of B, interleaved.
    InstructionModifier swsb = lastMultiply ? SWSB(1) : depAddr[0];

    mov(1 | swsb, addr0.ud(2), slmAOffsetLoad);
    mov(1 | depAddr[1], addr1.ud(2), slmBOffsetLoad);
    add(1 | depAddr[2], addr2.ud(2), slmBOffsetLoad, 8 * 32 / 16);
    add(1 | depAddr[3], addr3.ud(2), slmBOffsetLoad, 16 * 32 / 16);

    if (strategy.slmAltBarriers) barrierwait();

    if (strategy.simulation) sync(SyncFunction::nop, SWSB<int64_t>(1));
    sync.nop(sb5.src);
    load(16 | SWSB<AllPipes>(sb3, 4), A_regs[0], block_oword(16), SLM, addr0);
    load(16 | SWSB<AllPipes>(sb0, 3), B_regs[0], block_oword(16), SLM, addr1);
    load(16 | SWSB<AllPipes>(sb1, 2), B_regs[8], block_oword(16), SLM, addr2);
    if (strategy.unroll[LoopN] > 32)
        load(16 | SWSB<AllPipes>(sb2, 1), B_regs[16], block_oword(16), SLM,
                addr3);

    add(1 | sb3.src, addr0.ud(2), slmAOffsetLoad, 8 * 32 / 16);
    add(1 | sb0.src, addr1.ud(2), slmAOffsetLoad, 16 * 32 / 16);
    add(1 | sb1.src, addr2.ud(2), slmAOffsetLoad, 24 * 32 / 16);
    load(16 | SWSB<AllPipes>(sb4, 3), A_regs[8], block_oword(16), SLM, addr0);

    // Wait for A data to load.
    sync.allwr(0x18);

    if (strategy.slmAltBarriers && !lastMultiply) {
        sysgemmBarrierPrep(sb2.src, addr3);
        barriermsg(SWSB<AllPipes>(sb15, 1), addr3);
    }

    // Rows 0-7
    sysgemmMultiplyChunk(
            problem, strategy, false, 0, 0, true, false, sb0.dst, sb3);

    // Rows 8-15
    sysgemmMultiplyChunk(problem, strategy, false, 8, 8, false, false,
            InstructionModifier(), sb4);

    // Load third quarter of A (8x32)
    load(16 | SWSB<AllPipes>(sb3, 2), A_regs[0], block_oword(16), SLM, addr1);

    // Rows 16-23
    sysgemmMultiplyChunk(
            problem, strategy, false, 0, 16, false, false, sb3.dst);

    // Load last quarter of A (8x32)
    load(16 | SWSB<AllPipes>(sb4, 1), A_regs[8], block_oword(16), SLM, addr2);

    // Increment A and B to next buffer.
    swsb = strategy.simulation ? InstructionModifier(sb3.src)
                               : InstructionModifier();
    if (buffer == 2)
        mov(2 | swsb, slmAOffsetLoad(1), slmAOffsetLoadInit(1));
    else
        add(2 | swsb, slmAOffsetLoad(1), slmAOffsetLoad(1),
                strategy.slmSysgemmBlockSize() / 16);

    // Rows 24-31
    sysgemmMultiplyChunk(
            problem, strategy, false, 8, 24, false, false, sb4.dst, sb5);

    // Remember dependencies for address registers.
    depAddr[0] = InstructionModifier {};
    depAddr[1] = sb3.src;
    depAddr[2] = sb4.src;
    depAddr[3] = strategy.slmAltBarriers ? sb15.src : sb2.src;
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemmMultiply4(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state, int buffer,
        bool firstMultiply, RegData flagWaitLoad, RegData flagSignal,
        Label *labelDone) {
    using namespace sysgemm;
    auto &depAddr = state.sysgemm.depAddr;
    bool x48 = (strategy.unroll[LoopN] > 32);
    uint16_t slmStride = strategy.slmSysgemmBlockSize() / 16;
    int16_t slmAdvance = ((buffer == 3) ? -3 : 1) * slmStride;

    InstructionModifier loadMod {}, signalMod {};
    bool cooldownWaitLoad = flagWaitLoad.isValid();
    bool cooldownSignal = flagSignal.isValid();
    bool doWaitLoad = !cooldownWaitLoad || !flagWaitLoad.isNull();
    bool doSignal = !cooldownSignal || !flagSignal.isNull();
    auto fWaitLoad = static_cast<FlagRegister &>(flagWaitLoad);
    auto fSignal = static_cast<FlagRegister &>(flagSignal);
    if (doWaitLoad && cooldownWaitLoad) loadMod = loadMod | fWaitLoad | any16h;
    if (doSignal && cooldownSignal) signalMod = signalMod | fSignal | any16h;

    // Fence.
    if (doSignal) {
        sync.nop(depAddr[0]);
        slmfence(sb15 | signalMod, addr0);
        depAddr[0] = sb15.dst;
    }

    // Rows 0-7. Upper half of A (16x32) is already loaded.
    sync.nop(sb3.dst);
    depAddr[3] = InstructionModifier();
    sysgemmMultiplyChunk(
            problem, strategy, firstMultiply, 0, 0, true, false, sb0.dst, sb3);

    // Prepare addresses for loading lower half of A, and part of B
    add(1 | depAddr[1], addr1.ud(2), slmAOffsetLoad, 16 * 32 / 16);
    add(1 | depAddr[2], addr2.ud(2), slmAOffsetLoad, 24 * 32 / 16);
    sysgemmBarrierPrep(depAddr[3], addr3);

    // Rows 8-15.
    sysgemmMultiplyChunk(
            problem, strategy, firstMultiply, 8, 8, false, false, sb4.dst, sb4);

    // Load lower half of A (16x32) -- hopefully broadcast from SLM to this row.
    load(16 | SWSB<AllPipes>(sb3, 3), A_regs[0], block_oword(16), SLM, addr1);
    load(16 | SWSB<AllPipes>(sb4, 2), A_regs[8], block_oword(16), SLM, addr2);
    depAddr[1] = sb3.src;
    depAddr[2] = sb4.src;

    // Rows 16-23.
    sysgemmMultiplyChunk(problem, strategy, firstMultiply, 0, 16, false, false,
            sb3.dst, sb3);
    depAddr[1] = InstructionModifier();

    // Address prep, part 2.
    add(1 | depAddr[1], addr1.ud(2), slmBOffsetLoad, slmAdvance + 0 * 32 / 16);
    add(1 | depAddr[2], addr2.ud(2), slmBOffsetLoad, slmAdvance + 8 * 32 / 16);
    if (x48)
        add(1 | depAddr[0], addr0.ud(2), slmBOffsetLoad,
                slmAdvance
                        + 16 * 32
                                / 16); // consider moving after next dpasw block

    // Rows 24-31.
    sysgemmMultiplyChunk(
            problem, strategy, firstMultiply, 8, 24, false, true, sb4.dst, sb2);

    if (doWaitLoad) {
        if (cooldownWaitLoad) jmpi(1 | ~fWaitLoad, *labelDone);

        // Split barrier.
        barrierwait();
        if (doSignal) {
            barriermsg(SWSB<AllPipes>(sb15, x48 ? 4 : 3) | signalMod, addr3);
            depAddr[3] = sb15.src;
        }

        // Load next B data and upper 16x32 of A.
        load(16 | SWSB<AllPipes>(sb0, x48 ? 3 : 2), B_regs[0], block_oword(16),
                SLM, addr1);
        load(16 | SWSB<AllPipes>(sb1, x48 ? 2 : 1), B_regs[8], block_oword(16),
                SLM, addr2);
        if (x48)
            load(16 | SWSB<AllPipes>(sb2, 1), B_regs[16], block_oword(16), SLM,
                    addr0);
        depAddr[1] = sb0.src;
        depAddr[2] = sb1.src;
        if (x48) depAddr[0] = sb2.src;

        add(1 | depAddr[3], addr3.ud(2), slmAOffsetLoad,
                slmAdvance + 0 * 32 / 16);
        add(1 | depAddr[2], addr2.ud(2), slmAOffsetLoad,
                slmAdvance + 8 * 32 / 16);
        InstructionModifier swsb;
        if (strategy.simulation) swsb = sb2.src;
        if (buffer == 3)
            mov(2 | swsb, slmAOffsetLoad(1), slmAOffsetLoadInit(1));
        else
            add(2 | swsb, slmAOffsetLoad(1), slmAOffsetLoad(1), slmAdvance);

        load(16 | SWSB<AllPipes>(sb3, 3), A_regs[0], block_oword(16), SLM,
                addr3);
        load(16 | SWSB<AllPipes>(sb4, 2), A_regs[8], block_oword(16), SLM,
                addr2);
        depAddr[3] = sb3.src;
        depAddr[2] = sb4.src;
    }
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemmMultiplyChunk(
        const GEMMProblem &problem, const GEMMStrategy &strategy, bool first,
        int ao, int i0, bool waitB, bool prepB,
        const InstructionModifier &swsb0, const InstructionModifier &swsbEnd) {
    using namespace sysgemm;
    int co = i0 * 6;

    auto dpaswTyped
            = [&](InstructionModifier mod, uint8_t sdepth, uint8_t rcount,
                      const GRF &cReg, const GRF &aReg, const GRF &bReg) {
                  auto A = aReg.retype(problem.Ta.ngen());
                  auto B = bReg.retype(problem.Tb.ngen());
                  auto C = cReg.retype(problem.Tc.ngen());
                  first ? dpasw(mod, sdepth, rcount, C,
                          null.retype(problem.Tc.ngen()), A, B)
                        : dpasw(mod, sdepth, rcount, C, C, A, B);
              };

    if (strategy.unroll[LoopN] > 32) {
        if (waitB) {
            dpaswTyped(8 | swsb0 | Atomic, 8, 8, C_regs[co], A_regs[ao],
                    B_regs[0]);
            dpaswTyped(8, 8, 8, C_regs[co + 8], A_regs[ao], B_regs[4]);
            dpaswTyped(8 | sb1.dst | Atomic, 8, 8, C_regs[co + 16], A_regs[ao],
                    B_regs[8]);
            dpaswTyped(8, 8, 8, C_regs[co + 24], A_regs[ao], B_regs[12]);
            dpaswTyped(8 | sb2.dst | Atomic, 8, 8, C_regs[co + 32], A_regs[ao],
                    B_regs[16]);
            dpaswTyped(
                    8 | swsbEnd, 8, 8, C_regs[co + 40], A_regs[ao], B_regs[20]);
        } else if (prepB) {
            dpaswTyped(8 | swsb0 | Atomic, 8, 8, C_regs[co], A_regs[ao],
                    B_regs[0]);
            dpaswTyped(8 | sb0, 8, 8, C_regs[co + 8], A_regs[ao], B_regs[4]);
            dpaswTyped(
                    8 | Atomic, 8, 8, C_regs[co + 16], A_regs[ao], B_regs[8]);
            dpaswTyped(8 | sb1, 8, 8, C_regs[co + 24], A_regs[ao], B_regs[12]);
            dpaswTyped(
                    8 | Atomic, 8, 8, C_regs[co + 32], A_regs[ao], B_regs[16]);
            dpaswTyped(
                    8 | swsbEnd, 8, 8, C_regs[co + 40], A_regs[ao], B_regs[20]);
        } else {
            dpaswTyped(8 | swsb0 | Atomic, 8, 8, C_regs[co], A_regs[ao],
                    B_regs[0]);
            dpaswTyped(8 | Atomic, 8, 8, C_regs[co + 8], A_regs[ao], B_regs[4]);
            dpaswTyped(
                    8 | Atomic, 8, 8, C_regs[co + 16], A_regs[ao], B_regs[8]);
            dpaswTyped(
                    8 | Atomic, 8, 8, C_regs[co + 24], A_regs[ao], B_regs[12]);
            dpaswTyped(
                    8 | Atomic, 8, 8, C_regs[co + 32], A_regs[ao], B_regs[16]);
            dpaswTyped(
                    8 | swsbEnd, 8, 8, C_regs[co + 40], A_regs[ao], B_regs[20]);
        }
    } else {
        if (waitB) {
            dpaswTyped(8 | swsb0 | Atomic, 8, 8, C_regs[co], A_regs[ao],
                    B_regs[0]);
            dpaswTyped(8, 8, 8, C_regs[co + 8], A_regs[ao], B_regs[4]);
            dpaswTyped(8 | sb1.dst | Atomic, 8, 8, C_regs[co + 16], A_regs[ao],
                    B_regs[8]);
            dpaswTyped(
                    8 | swsbEnd, 8, 8, C_regs[co + 24], A_regs[ao], B_regs[12]);
        } else if (prepB) {
            dpaswTyped(8 | swsb0 | Atomic, 8, 8, C_regs[co], A_regs[ao],
                    B_regs[0]);
            dpaswTyped(8 | sb0, 8, 8, C_regs[co + 8], A_regs[ao], B_regs[4]);
            dpaswTyped(
                    8 | Atomic, 8, 8, C_regs[co + 16], A_regs[ao], B_regs[8]);
            dpaswTyped(
                    8 | swsbEnd, 8, 8, C_regs[co + 24], A_regs[ao], B_regs[12]);
        } else {
            dpaswTyped(8 | swsb0 | Atomic, 8, 8, C_regs[co], A_regs[ao],
                    B_regs[0]);
            dpaswTyped(8 | Atomic, 8, 8, C_regs[co + 8], A_regs[ao], B_regs[4]);
            dpaswTyped(
                    8 | Atomic, 8, 8, C_regs[co + 16], A_regs[ao], B_regs[8]);
            dpaswTyped(
                    8 | swsbEnd, 8, 8, C_regs[co + 24], A_regs[ao], B_regs[12]);
        }
    }
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemmBarrierPrep(
        const InstructionModifier &swsb, const GRF &header) {
    using namespace sysgemm;
    mov<uint32_t>(1 | swsb, header[2], barrierVal);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemmReorderLocalIDs(
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
    if (strategy.splitCopy) return;
    if (strategy.wg[LoopM] != 8) return;

    auto storage = state.ra.alloc_sub<uint64_t>();
    auto temp = storage.uw(0);
    auto temp2 = storage.uw(2);
    auto lidM = state.inputs.localIDM;
    auto lidN = state.inputs.localIDN;

    // Remap local IDs so that upper 4x4 threads come first, then lower 4x4 threads.
    bfi2(1, temp, 0x08, lidN, lidM);
    shr(1, temp2, lidN, 1);
    shr(1, lidN, temp, 2);
    bfi2(1, lidM, 0x04, temp2, lidM);

    state.ra.safeRelease(storage);
}

namespace sysgemm2 {
namespace x48 {
static GRFRange A_regs = GRF(32) - GRF(63);
static GRFRange B_regs = GRF(2) - GRF(25);
static GRFRange C_regs = GRF(64) - GRF(255);
static Subregister B_addr[3] = {GRF(26).ud(2), GRF(27).ud(2), GRF(1).ud(2)};
static Subregister A_addr[4]
        = {GRF(28).ud(2), GRF(29).ud(2), GRF(30).ud(2), GRF(31).ud(2)};
static GRF headerTemp = GRF(0);
} // namespace x48

namespace x32 {
static GRFRange A_regs[2] = {GRF(32) - GRF(63), GRF(96) - GRF(127)};
static GRFRange B_regs[2] = {GRF(2) - GRF(17), GRF(66) - GRF(81)};
static GRFRange C_regs = GRF(128) - GRF(255);
static Subregister B_addr[2][2]
        = {{GRF(26).ud(2), GRF(27).ud(2)}, {GRF(90).ud(2), GRF(91).ud(2)}};
static Subregister A_addr[2][4]
        = {{GRF(28).ud(2), GRF(29).ud(2), GRF(30).ud(2), GRF(31).ud(2)},
                {GRF(92).ud(2), GRF(93).ud(2), GRF(94).ud(2), GRF(95).ud(2)}};
static GRF barrierHeader = GRF(0);
static GRF fenceHeader = GRF(64);
} // namespace x32

static GRFRange copyInputs = GRF(254) - GRF(255);
static Subregister A_copyLoadAddr0 = GRF(254).uq(0);
static Subregister A_copyLoadAddrSurf0 = GRF(254).ud(2);
static Subregister slmAOff = GRF(254).d(4);
static Subregister lda = GRF(254).ud(6);
static Subregister B_copyLoadAddr0 = GRF(255).uq(0);
static Subregister B_copyLoadAddrSurf0 = GRF(255).ud(2);
static Subregister slmBOff[2] = {GRF(255).d(4), GRF(255).d(5)};
static Subregister ldb = GRF(255).ud(6);

static Subregister kCounter = AccumulatorRegister(2).d(0);
static Subregister barrierVal = AddressRegister(0).ud(0);
} // namespace sysgemm2

template <HW hw>
bool gemm_kernel_generator_t<hw>::sysgemm2AccumulateC(
        GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state) {
    using namespace sysgemm2;
    auto params = systolicParams(hw, problem, strategy);
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];
    auto localIDM = state.lidM;
    auto localIDN = state.lidN;
    auto C_regs = (unrollN == 48) ? x48::C_regs : x32::C_regs;

    if (unrollM != 16 && unrollM != 32) stub();
    if (unrollN != 32 && unrollN != 48) stub();
    if (isPacked(problem.A.layout)) {
        if (problem.A.crosspack != params.opsPerChan) stub();
        if (problem.A.tileR != params.osys) stub();
        if (problem.A.tileC != params.ksys) stub();
    }
    if (isPacked(problem.B.layout)) {
        if (problem.B.crosspack != params.ksys) stub();
        if (problem.B.tileR != 0 || problem.B.tileC != 0) stub();
    }

    state.ra.claim(C_regs);

    // Check whether this thread will do copy or compute. Copy threads get priority (lower thread #).
    auto flagCompute = f1[1];
    mov(1, flagCompute, state.isCompute.uw());
    state.ra.safeRelease(state.isCompute);

    // Calculate A/B addresses and SLM offsets.
    auto tempStorage = C_regs[0];
    auto suboffsetA = tempStorage.ud(0);
    auto suboffsetB = tempStorage.ud(1);
    auto tempB = tempStorage.ud(2);
    auto ldaUnrollM4 = tempStorage.ud(3);
    auto aInc = tempStorage.ud(4);
    auto ldbUnrollN4 = tempStorage.ud(5);
    auto bInc = tempStorage.ud(6);

    if (problem.A.layout == MatrixLayout::T)
        mulConstant(1, ldaUnrollM4, state.inputs.lda, unrollM / 4);
    if (problem.B.layout == MatrixLayout::N)
        mulConstant(1, ldbUnrollN4, state.inputs.ldb, unrollN / 4);
    if (!isPacked(problem.A.layout)) mov(1, lda, state.inputs.lda);
    if (!isPacked(problem.B.layout)) mov(1, ldb, state.inputs.ldb);

    and_(1 | ne | state.flagAP, null.uw(), localIDM, 1);

    switch (problem.A.layout) {
        case MatrixLayout::Pc:
            mulConstant(1, aInc, localIDN, unrollM * (32 / 4));
            break;
        case MatrixLayout::N:
            mulConstant(1, aInc, localIDN, unrollM * problem.Ta / 4);
            break;
        case MatrixLayout::T: mul(1, aInc, ldaUnrollM4, localIDN.uw()); break;
        default: stub();
    }
    switch (problem.B.layout) {
        case MatrixLayout::Pr:
            mulConstant(1, bInc, localIDM, unrollN * (32 / 4));
            break;
        case MatrixLayout::N: mul(1, bInc, ldbUnrollN4, localIDM.uw()); break;
        case MatrixLayout::T:
            mulConstant(1, bInc, localIDM, unrollN * problem.Tb / 4);
            break;
        default: stub();
    }

    mulConstant(1, suboffsetA, localIDN, unrollM * (32 / 4) / 16);
    mulConstant(1, suboffsetB, localIDM, unrollN * (32 / 4) / 16);

    if (strategy.A.base.isStateless())
        eadd(1, A_copyLoadAddr0, state.effA, aInc, strategy, state);
    else
        add(1, A_copyLoadAddrSurf0, state.effA, aInc);

    if (strategy.B.base.isStateless())
        eadd(1, B_copyLoadAddr0, state.effB, bInc, strategy, state);
    else
        add(1, B_copyLoadAddrSurf0, state.effB, bInc);

    mad(1, tempB, (4 * unrollM * 36) / 16, localIDN, (unrollN * 32) / 16);

    mul(1, x48::A_addr[0], localIDM, (unrollM * 36) / 16);
    add(1 | state.flagAP, x48::B_addr[0], tempB, (unrollN / 2) * (32 / 16));
    mov(1 | ~state.flagAP, x48::B_addr[0], tempB);

    add(1, slmAOff, x48::A_addr[0], suboffsetA);
    add(1, slmBOff[0], tempB, suboffsetB);
    add3(1, slmBOff[1], tempB, suboffsetB, 8 * 32 / 16);

    // Marshal data needed later into acc2 for safekeeping.
    auto saveData = state.ra.alloc_range(2);
    auto kLoops = saveData[0].d(0);
    auto ldc = saveData[0].ud(1);
    auto flags = saveData[0].ud(2);
    auto k = saveData[0].ud(3);
    auto remM = saveData[0].uw(8);
    auto remN = saveData[0].uw(9);
    auto abo = saveData[0].ud(5);
    auto ao = saveData[0].w(10);
    auto bo = saveData[0].w(11);
    auto alpha = saveData[0].ud(6).reinterpret(0, problem.Ts.ngen());
    auto beta = saveData[0].ud(7).reinterpret(0, problem.Ts.ngen());
    auto remFusedStorage = saveData[1].ud(0);
    auto diagC = saveData[1].ud(1);
    auto effCO = saveData[1].uq(1);
    auto C_ptr = saveData[1].uq(2);
    auto slotAB = saveData[1].ud(6);
    auto effAs = a0.ud(4); // dwords 4-5
    auto effBs = a0.ud(6); // dwords 6-7

    if (state.r0_info != acc0.ud()) mov<uint32_t>(8, acc0, state.r0_info);

    add(1, kLoops, state.k, params.ksys - 1);
    mov(1, ldc, state.inputs.ldc[0]);
    emov(1, C_ptr, state.effC[0], strategy, state);
    if (state.inputs.flags.isValid()) mov(1, flags, state.inputs.flags);
    mov(1, k, state.k);
    if (state.remainders[LoopM].isValid())
        mov(1, remM, state.remainders[LoopM]);
    if (state.remainders[LoopN].isValid())
        mov(1, remN, state.remainders[LoopN]);
    if (state.inputs.abo.isValid())
        mov(1, abo, state.inputs.abo);
    else {
        if (state.inputs.ao.isValid()) mov(1, ao, state.inputs.ao);
        if (state.inputs.bo.isValid()) mov(1, bo, state.inputs.bo);
    }
    if (state.inputs.alpha_real.isValid())
        mov(1, alpha, state.inputs.alpha_real);
    if (state.inputs.beta_real.isValid()) mov(1, beta, state.inputs.beta_real);
    shr(1, kLoops, kLoops, log2(params.ksys));
    if (state.remFusedStorage.isValid())
        mov(1, remFusedStorage, state.remFusedStorage);
    if (state.diagC.isValid()) mov(1, diagC, state.diagC);
    if (state.effCO.isValid()) {
        effCO = effCO.reinterpret(0, state.effCO.getType());
        emov(1, effCO, state.effCO, strategy, state);
    }
    if (problem.abOffset != ABOffset::None) {
        GRF temp = state.ra.alloc();
        state.effAs = temp.uq(0).reinterpret(0, state.effA.getType());
        state.effBs = temp.uq(1).reinterpret(0, state.effB.getType());
        gemmCalcABOffsetAddrs(problem, strategy, state);
        mov<uint32_t>(4, effAs(1), temp);
        state.ra.safeRelease(temp);
    }
    if (state.fusedGEMM.slotA.isValid())
        mov(1, slotAB, state.fusedGEMM.slotA.ud());

    if (state.isNested) {
        // To do: replace with sel
        mov(2 | ~flagCompute, remM(1), 0);
        mov(1 | ~flagCompute, remFusedStorage, 0);
    }

    releaseSavedMNLocalIDs(state);
    state.ra.safeRelease(state.effA);
    state.ra.safeRelease(state.effB);
    state.ra.safeRelease(state.effC[0]);
    state.ra.safeRelease(state.inputs.lda);
    state.ra.safeRelease(state.inputs.ldb);

    state.ra.release(state.inputs.ldc[0]);
    state.ra.release(state.k);
    state.ra.release(state.remainders[LoopM]);
    state.ra.release(state.remainders[LoopN]);
    state.ra.release(state.inputs.abo);
    state.ra.release(state.inputs.ao);
    state.ra.release(state.inputs.bo);
    state.ra.release(state.inputs.alpha_real);
    state.ra.release(state.inputs.beta_real);
    state.ra.release(state.remFusedStorage);
    state.ra.release(state.diagC);
    state.ra.release(state.effCO);
    state.ra.release(state.fusedGEMM.slotA);
    state.ra.release(state.fusedGEMM.slotB);

    if (state.r0_info.isARF()) stub();
    GRF r0_info {state.r0_info.getBase()};
    if (hw >= HW::XeHPG) {
        mov(1, barrierVal.uw(0), Immediate::uw(0));
        mov(2, barrierVal.ub(2)(1), r0_info.ub(11)(0));
    } else
        and_(1, barrierVal, r0_info.ud(2), 0x7F000000);

    mov<float>(16, acc2, saveData[0]);

    Label labelCompute, labelDone;

    jmpi(1 | f1[1], labelCompute);
    sysgemm2KLoopCopy(problem, strategy, state);
    if (state.isNested) {
        jmpi(1, labelDone);
    } else
        epilogue(strategy, state);
    mark(labelCompute);
    sysgemm2KLoopCompute(problem, strategy, state);
    mark(labelDone);

    mov<float>(16, saveData[0], acc2);

    state.effC[0] = C_ptr;
    state.inputs.ldc[0] = ldc;
    if (state.inputs.flags.isValid()) state.inputs.flags = flags;
    state.k = k;
    if (state.remainders[LoopM].isValid()) state.remainders[LoopM] = remM;
    if (state.remainders[LoopN].isValid()) state.remainders[LoopN] = remN;
    if (state.inputs.abo.isValid()) state.inputs.abo = abo;
    if (state.inputs.ao.isValid()) state.inputs.ao = ao;
    if (state.inputs.bo.isValid()) state.inputs.bo = bo;
    if (state.inputs.alpha_real.isValid()) {
        state.inputs.alpha_real = alpha;
        if (!problem.alpha_real.fixed()) problem.alpha_real = alpha;
    }
    if (state.inputs.beta_real.isValid()) {
        state.inputs.beta_real = beta;
        if (!problem.beta_real.fixed()) problem.beta_real = beta;
    }
    if (state.remFusedStorage.isValid()) {
        state.remFusedStorage = remFusedStorage;
        state.remaindersFused[LoopM] = state.remainders[LoopM];
        state.remaindersFused[LoopN] = state.remainders[LoopN];
        state.remaindersFused[strategy.fusedLoop] = remFusedStorage;
    }
    if (state.diagC.isValid()) state.diagC = diagC;
    if (state.effCO.isValid()) state.effCO = effCO;
    if (state.fusedGEMM.slotA.isValid()) {
        state.fusedGEMM.slotA = slotAB.uw(0);
        state.fusedGEMM.slotB = slotAB.uw(1);
    }
    if (problem.abOffset != ABOffset::None) {
        auto tas = state.effAs.getType();
        auto tbs = state.effBs.getType();
        state.effAs = state.ra.alloc_sub(tas);
        state.effBs = state.ra.alloc_sub(tbs);
        mov<uint32_t>(getDwords(tas), state.effAs.ud()(1), effAs(1));
        mov<uint32_t>(getDwords(tbs), state.effBs.ud()(1), effBs(1));
    }

    // Set up C internal layout and registers.
    state.C_regs.resize(1);
    state.C_regs[0] = C_regs;
    state.C_layout.clear();
    state.C_layout.reserve((unrollM / 8) * (unrollN / 4));
    for (int j0 = 0; j0 < unrollN; j0 += 4) {
        for (int i0 = 0; i0 < unrollM; i0 += 8) {
            RegisterBlock block;
            block.log2GRFBytes = GRF::log2Bytes(hw);
            block.colMajor = true;
            block.splitComplex = false;
            block.nr = block.ld = 8;
            block.nc = 4;
            block.offsetR = i0;
            block.offsetC = j0;
            block.crosspack = 1;
            block.bytes = 8 * 4 * problem.Tc.size();
            block.simdSize = 0;

            int j0Interleaved = j0 << 1;
            if (j0Interleaved >= unrollN) j0Interleaved += 4 - unrollN;

            block.offsetBytes
                    = (unrollN * i0 / 8 + j0Interleaved) * GRF::bytes(hw);
            state.C_layout.push_back(block);
        }
    }

    // Set up C external layout.
    state.copyC = true;
    bool remM_Ce, remN_Ce;
    getCRemainders(problem, strategy, remM_Ce, remN_Ce);

    if (!getRegLayout(problem.Tc_ext, state.C_layoutExt, unrollM, unrollN,
                remM_Ce, remN_Ce, true, false, 0, 0, problem.C,
                state.Cext_strategy))
        return false;
    if (remM_Ce || remN_Ce)
        (void)getRegLayout(problem.Tc_ext, state.C_layoutExtUnmasked, unrollM,
                unrollN, false, false, true, false, 0, 0, problem.C,
                state.Cext_strategy);

    if (state.r0_info != acc0.ud()) mov<uint32_t>(8, state.r0_info, acc0);

    return true; // Success!
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemm2KLoopCopy(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state) {
    using namespace sysgemm2;

    Label top, bottom, smallK, skipSmallK, reenterSmallK, done;

    bool surfaceA = !strategy.A.base.isStateless();
    bool surfaceB = !strategy.B.base.isStateless();
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];
    bool _32x = (unrollM == 32);
    bool x48 = (unrollN == 48);
    bool int8 = problem.Ta.size() == 1;

    int globalBuffers = strategy.A_copies;
    auto slmStride = strategy.slmSysgemmBlockSize() / 16;

    if (globalBuffers < 2) stub();
    if (globalBuffers > 5) stub();

    auto saveRA = state.ra;
    state.ra.release(r0 - r127);
    state.ra.release(r128 - r255);
    state.ra.claim(copyInputs);

    // Register and token allocation.
    int aTokens = 0, aLoadAddrs = 0, aStoreAddrs = 1;
    int bTokens = 0, bLoadAddrs = 0, bStoreAddrs = 1 + x48;
    int aiRegCount = unrollM / 4, biRegCount = unrollN / 4;
    bool aRepack = false, bRepack = false;

    if (problem.A.alignment & 3 || problem.B.alignment & 3) stub();

    switch (problem.A.layout) {
        case MatrixLayout::Pc:
            aTokens = aLoadAddrs = (surfaceA && _32x) ? 2 : 1;
            break;
        case MatrixLayout::N:
            if (!surfaceA) stub();
            aTokens = int8 ? 2 : 1;
            aLoadAddrs = aTokens * 2;
            aRepack = true;
            break;
        case MatrixLayout::T:
            if (!surfaceA) stub();
            aTokens = aLoadAddrs = 2;
            break;
        default: stub();
    }

    switch (problem.B.layout) {
        case MatrixLayout::Pr:
            bTokens = bLoadAddrs = (surfaceB ? 2 : 1) + x48;
            break;
        case MatrixLayout::N:
            if (!surfaceB) stub();
            bTokens = 2;
            bLoadAddrs = x48 ? 4 : 2;
            if (x48) biRegCount = 16;
            bRepack = true;
            break;
        case MatrixLayout::T:
            if (!surfaceB) stub();
            bTokens = (int8 || x48) ? 2 : 1;
            bLoadAddrs = bTokens * 2;
            bRepack = true;
            break;
        default: stub();
    }

    int tokenStride = aTokens + bTokens;
    if (tokenStride * globalBuffers > 15)
        throw std::runtime_error("Not enough tokens available.");

    auto &Ai_regs = state.Ai_regs;
    auto &Bi_regs = state.Bi_regs;
    auto &Ao_regs = state.Ao_regs;
    auto &Bo_regs = state.Bo_regs;
    auto &Ai_addrs = state.Ai_addrs;
    auto &Bi_addrs = state.Bi_addrs;
    auto &Ao_addrs = state.Ao_addrs;
    auto &Bo_addrs = state.Bo_addrs;
    GRFRange ldaMultiples, ldbMultiples;
    FlagRegister flag12;
    GRF A_swizzle, B_swizzle;
    Subregister lda16, ldb16, ldaK, ldbK;
    Subregister Ai_advance, Bi_advance;
    GRF copyBarrierHeader = state.ra.alloc();
    GRF copyFenceHeader = state.ra.alloc();
    GRF slmBase = state.ra.alloc().d();
    GRF temp = state.ra.alloc().d();

    Ai_regs.reserve(globalBuffers);
    Bi_regs.reserve(globalBuffers);
    Ai_addrs.reserve(globalBuffers);
    Bi_addrs.reserve(globalBuffers);
    for (int i = 0; i < globalBuffers; i++) {
        Ai_regs.push_back(state.ra.alloc_range(aiRegCount));
        Bi_regs.push_back(state.ra.alloc_range(biRegCount));
        Ai_addrs.push_back(state.ra.alloc_range(aLoadAddrs));
        Bi_addrs.push_back(state.ra.alloc_range(bLoadAddrs));
    }

    if (aRepack) Ao_regs = state.ra.alloc_range(8);
    if (bRepack) Bo_regs = state.ra.alloc_range(unrollN / 4);
    Ao_addrs.push_back(state.ra.alloc_range(aStoreAddrs));
    Bo_addrs.push_back(state.ra.alloc_range(bStoreAddrs));

    if (state.emulate.temp[0].isValid()) {
        for (int q = 0; q < 2; q++)
            state.ra.safeRelease(state.emulate.temp[q]);
        state.emulate.flag = f1[1];
        state.emulate.flagOffset = 8;
    }

    // Address initialization.
    if (surfaceA && isPacked(problem.A.layout))
        shr(1, A_copyLoadAddrSurf0, A_copyLoadAddrSurf0, 4);
    if (surfaceB && isPacked(problem.B.layout))
        shr(1, B_copyLoadAddrSurf0, B_copyLoadAddrSurf0, 4);

    mov(1, slmBase[0], 0);
    mov(1, slmBase[1], -4 * slmStride);

    auto makeLDMultiples = [&](GRFRange &multiples, const Subregister &ld,
                                   int n) {
        multiples = state.ra.alloc_range(n / 8);
        mov<uint16_t>(8, multiples[0], Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
        if (n > 8)
            mov<uint16_t>(8, multiples[1],
                    Immediate::uv(8, 9, 10, 11, 12, 13, 14, 15));
        mul<uint32_t>(8, multiples[0], ld, multiples[0].uw());
        if (n > 8) mul<uint32_t>(8, multiples[1], ld, multiples[1].uw());
    };

    switch (problem.A.layout) {
        case MatrixLayout::N:
            lda16 = state.ra.alloc_sub<uint32_t>();
            Ai_advance = state.ra.alloc_sub<uint32_t>();
            if (!int8) {
                A_swizzle = state.ra.alloc().uw();
                mov(4, A_swizzle[0](1), Immediate::uv(0, 4, 2, 6, 0, 0, 0, 0));
                add(4, A_swizzle[4](1), A_swizzle[0](1), 64);
                add(8, A_swizzle[8](1), A_swizzle[0](1), 128);
            }
            makeLDMultiples(ldaMultiples, lda, 16);
            mulConstant(1, lda16, lda, 16);
            if (int8) {
                ldaK = state.ra.alloc_sub<uint32_t>();
                mulConstant(1, ldaK, lda, 32);
            } else
                ldaK = lda16;
            mulConstant(1, Ai_advance, lda, (int8 ? 32 : 16) * globalBuffers);
            break;
        case MatrixLayout::T: makeLDMultiples(ldaMultiples, lda, 8); break;
        default: break;
    }

    switch (problem.B.layout) {
        case MatrixLayout::N:
            B_swizzle = state.ra.alloc().uw();
            mov(8, B_swizzle[0](1), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
            if (x48) {
                flag12 = state.raVFlag.alloc();
                mov(1, flag12, 0x0FFF);
            }
            mad(8, B_swizzle[8](1), 4, B_swizzle[0](1), x48 ? 64 : 32);
            mulConstant(8, B_swizzle[0](1), B_swizzle[0](1), x48 ? 64 : 32);
            makeLDMultiples(ldbMultiples, ldb, x48 ? 16 : 8);
            break;
        case MatrixLayout::T:
            makeLDMultiples(ldbMultiples, ldb, 16);
            if (int8) {
                ldb16 = state.ra.alloc_sub<uint32_t>();
                mulConstant(1, ldb16, ldb, 16);
            }
            Bi_advance = state.ra.alloc_sub<uint32_t>();
            ldbK = state.ra.alloc_sub<uint32_t>();
            mulConstant(1, Bi_advance, ldb, (int8 ? 32 : 16) * globalBuffers);
            mulConstant(1, ldbK, ldb, int8 ? 32 : 16);
        default: break;
    }

    for (int i = 0; i < globalBuffers; i++)
        switch (problem.A.layout) {
            case MatrixLayout::Pc:
                if (surfaceA) {
                    add(1, Ai_addrs[i][0].ud(2), A_copyLoadAddrSurf0,
                            i * unrollM * 32 / 16);
                    if (_32x)
                        add(1, Ai_addrs[i][1].ud(2), A_copyLoadAddrSurf0,
                                (i * unrollM * 32 + 4 * 32) / 16);
                } else
                    eadd(1, Ai_addrs[i][0].uq(0), A_copyLoadAddr0,
                            i * unrollM * 32, strategy, state);
                break;
            case MatrixLayout::N:
                if (i == 0) {
                    add<uint32_t>(16, Ai_addrs[i][0], A_copyLoadAddrSurf0,
                            ldaMultiples);
                    if (int8)
                        add3<uint32_t>(16, Ai_addrs[i][2], A_copyLoadAddrSurf0,
                                ldaMultiples, lda16);
                } else {
                    add<uint32_t>(16, Ai_addrs[i][0], Ai_addrs[i - 1][0], ldaK);
                    if (int8)
                        add<uint32_t>(
                                16, Ai_addrs[i][2], Ai_addrs[i - 1][2], ldaK);
                }
                break;
            case MatrixLayout::T:
                add3<uint32_t>(8, Ai_addrs[i][0], A_copyLoadAddrSurf0,
                        ldaMultiples, i * 32 + 0);
                add3<uint32_t>(8, Ai_addrs[i][1], A_copyLoadAddrSurf0,
                        ldaMultiples, i * 32 + 16);
                break;
            default: stub();
        }

    for (int i = 0; i < globalBuffers; i++)
        switch (problem.B.layout) {
            case MatrixLayout::Pr:
                if (surfaceB) {
                    add(1, Bi_addrs[i][0].ud(2), B_copyLoadAddrSurf0,
                            i * unrollN * 32 / 16);
                    add(1, Bi_addrs[i][1].ud(2), B_copyLoadAddrSurf0,
                            (i * unrollN * 32 + 4 * 32) / 16);
                    if (x48)
                        add(1, Bi_addrs[i][2].ud(2), B_copyLoadAddrSurf0,
                                (i * unrollN * 32 + 8 * 32) / 16);
                } else {
                    eadd(1, Bi_addrs[i][0].uq(0), B_copyLoadAddr0,
                            i * unrollN * 32, strategy, state);
                    if (x48)
                        eadd(1, Bi_addrs[i][1].uq(0), B_copyLoadAddr0,
                                i * unrollN * 32 + 8 * 32, strategy, state);
                }
                break;
            case MatrixLayout::N:
                add3<uint32_t>(x48 ? 16 : 8, Bi_addrs[i][0],
                        B_copyLoadAddrSurf0, ldbMultiples, i * 32 + 0);
                add3<uint32_t>(x48 ? 16 : 8, Bi_addrs[i][x48 ? 2 : 1],
                        B_copyLoadAddrSurf0, ldbMultiples, i * 32 + 16);
                break;
            case MatrixLayout::T:
                if (i == 0) {
                    add<uint32_t>(16, Bi_addrs[i][0], B_copyLoadAddrSurf0,
                            ldbMultiples);
                    if (int8)
                        add3<uint32_t>(16, Bi_addrs[i][2], B_copyLoadAddrSurf0,
                                ldbMultiples, ldb16);
                    else if (x48)
                        add3<uint32_t>(16, Bi_addrs[i][2], B_copyLoadAddrSurf0,
                                ldbMultiples, 16);
                } else {
                    add<uint32_t>(16, Bi_addrs[i][0], Bi_addrs[i - 1][0], ldbK);
                    if (int8 || x48)
                        add<uint32_t>(
                                16, Bi_addrs[i][2], Bi_addrs[i - 1][2], ldbK);
                }
                break;
            default: stub();
        }

    sysgemmBarrierPrep(InstructionModifier(), copyBarrierHeader);

    mov(2, slmBase[4](1), slmBase[0](1));

    // Main logic.
    auto copyLoad = [&](int buffer) {
        int atbase = tokenStride * buffer;
        int btbase = atbase + aTokens;
        switch (problem.A.layout) {
            case MatrixLayout::Pc:
                if (surfaceA) {
                    load(16 | SBID(atbase + 0), Ai_regs[buffer][0],
                            block_oword(8), strategy.A.base,
                            Ai_addrs[buffer][0]);
                    if (_32x)
                        load(16 | SBID(atbase + 1), Ai_regs[buffer][4],
                                block_oword(8), strategy.A.base,
                                Ai_addrs[buffer][1]);
                } else
                    load(16 | SBID(atbase + 0), Ai_regs[buffer][0],
                            block_hword(_32x ? 8 : 4), strategy.A.base,
                            Ai_addrs[buffer]);
                break;
            case MatrixLayout::N:
                if (int8) {
                    load(16 | SBID(atbase + 0), Ai_regs[buffer][0],
                            surface_dword(ChannelMask::rg), strategy.A.base,
                            Ai_addrs[buffer][0]);
                    load(16 | SBID(atbase + 1), Ai_regs[buffer][4],
                            surface_dword(ChannelMask::rg), strategy.A.base,
                            Ai_addrs[buffer][2]);
                } else
                    load(16 | SBID(atbase + 0), Ai_regs[buffer][0],
                            surface_dword(ChannelMask::rgba), strategy.A.base,
                            Ai_addrs[buffer][0]);
                break;
            case MatrixLayout::T:
                load(8 | SBID(atbase + 0), Ai_regs[buffer][0],
                        surface_dword(ChannelMask::rgba), strategy.A.base,
                        Ai_addrs[buffer][0]);
                load(8 | SBID(atbase + 1), Ai_regs[buffer][4],
                        surface_dword(ChannelMask::rgba), strategy.A.base,
                        Ai_addrs[buffer][1]);
                break;
            default: stub();
        }

        switch (problem.B.layout) {
            case MatrixLayout::Pr:
                if (surfaceB) {
                    load(16 | SBID(btbase + 0), Bi_regs[buffer][0],
                            block_oword(8), strategy.B.base,
                            Bi_addrs[buffer][0]);
                    load(16 | SBID(btbase + 1), Bi_regs[buffer][4],
                            block_oword(8), strategy.B.base,
                            Bi_addrs[buffer][1]);
                    if (x48)
                        load(16 | SBID(btbase + 2), Bi_regs[buffer][8],
                                block_oword(8), strategy.B.base,
                                Bi_addrs[buffer][2]);
                } else {
                    load(16 | SBID(btbase + 0), Bi_regs[buffer][0],
                            block_hword(8), strategy.B.base,
                            Bi_addrs[buffer][0]);
                    if (x48)
                        load(16 | SBID(btbase + 1), Bi_regs[buffer][8],
                                block_hword(4), strategy.B.base,
                                Bi_addrs[buffer][1]);
                }
                break;
            case MatrixLayout::N:
                if (x48) {
                    load(16 | SBID(btbase + 0) | flag12 | any4h,
                            Bi_regs[buffer][0],
                            surface_dword(ChannelMask::rgba), strategy.B.base,
                            Bi_addrs[buffer][0]);
                    load(16 | SBID(btbase + 1) | flag12 | any4h,
                            Bi_regs[buffer][8],
                            surface_dword(ChannelMask::rgba), strategy.B.base,
                            Bi_addrs[buffer][2]);
                } else {
                    load(8 | SBID(btbase + 0), Bi_regs[buffer][0],
                            surface_dword(ChannelMask::rgba), strategy.B.base,
                            Bi_addrs[buffer][0]);
                    load(8 | SBID(btbase + 1), Bi_regs[buffer][4],
                            surface_dword(ChannelMask::rgba), strategy.B.base,
                            Bi_addrs[buffer][1]);
                }
                break;
            case MatrixLayout::T:
                if (int8) {
                    auto cmask = x48 ? ChannelMask::rgb : ChannelMask::rg;
                    load(16 | SBID(btbase + 0), Bi_regs[buffer][0],
                            surface_dword(cmask), strategy.B.base,
                            Bi_addrs[buffer][0]);
                    load(16 | SBID(btbase + 1), Bi_regs[buffer][x48 ? 6 : 4],
                            surface_dword(cmask), strategy.B.base,
                            Bi_addrs[buffer][2]);
                } else {
                    load(16 | SBID(btbase + 0), Bi_regs[buffer][0],
                            surface_dword(ChannelMask::rgba), strategy.B.base,
                            Bi_addrs[buffer][0]);
                    if (x48)
                        load(16 | SBID(btbase + 1), Bi_regs[buffer][8],
                                surface_dword(ChannelMask::rg), strategy.B.base,
                                Bi_addrs[buffer][2]);
                }
                break;
            default: stub();
        }
    };

    auto copyRepack = [&](int buffer) {
        int atbase = tokenStride * buffer;
        int btbase = atbase + aTokens;

        switch (problem.A.layout) {
            case MatrixLayout::N:
                if (int8) {
                    for (int j = 0; j < 4; j++) {
                        int reg = (j >> 1);
                        int sub = (j & 1) << 4;
                        mov<uint8_t>(16, Ao_regs[j + 0][0](1),
                                Ai_regs[buffer][reg + 0][sub](1, 4, 4));
                        mov<uint8_t>(16, Ao_regs[j + 4][0](1),
                                Ai_regs[buffer][reg + 4][sub](1, 4, 4));
                        mov<uint8_t>(16, Ao_regs[j + 0][16](1),
                                Ai_regs[buffer][reg + 2][sub](1, 4, 4));
                        mov<uint8_t>(16, Ao_regs[j + 4][16](1),
                                Ai_regs[buffer][reg + 6][sub](1, 4, 4));
                    }
                } else {
                    // a0: 0 4 2 6 64 68 66 70...
                    add(16, a0, A_swizzle, Ai_regs[buffer][0].getBase() * 32);
                    setDefaultAutoSWSB(false);
                    sync.allwr(0b1 << atbase);
                    for (int j = 0; j < 8; j++)
                        mov<uint16_t>(
                                16, Ao_regs[j], indirect[a0].uw(j * 8)(1, 0));
                    setDefaultAutoSWSB(true);
                }
                break;
            default: break;
        }

        switch (problem.B.layout) {
            case MatrixLayout::N:
                // a0 (x32): 0 32 64 96 128 160 192 228 4 36 68...
                // a0 (x48): 0 64 128 192 256 320 384 448 4 68 132 196...
                add(16, a0, B_swizzle, Bi_regs[buffer][0].getBase() * 32);
                setDefaultAutoSWSB(false);
                sync.allwr(0b11 << btbase);
                for (int j = 0; j < unrollN / 4; j += 2) // 2 cols at a time
                    mov<uint32_t>(16, Bo_regs[j], indirect[a0].ud(j * 4)(1, 0));
                setDefaultAutoSWSB(true);
                break;
            case MatrixLayout::T:
                if (int8) {
                    for (int j = 0; j < unrollN / 4; j++)
                        mov<uint8_t>(16, Bo_regs[j][0](1),
                                Bi_regs[buffer][(j & ~3) >> 1][j & 3](4));
                    for (int j = 0; j < unrollN / 4; j++)
                        mov<uint8_t>(16, Bo_regs[j][16](1),
                                Bi_regs[buffer][(x48 ? 6 : 4) + ((j & ~3) >> 1)]
                                       [j & 3](4));
                } else {
                    for (int j = 0; j < unrollN / 4; j++)
                        mov<uint16_t>(16, Bo_regs[j],
                                Bi_regs[buffer][j & ~1][j & 1](2));
                }
                break;
            default: break;
        }
    };

    auto copyStore = [&](int buffer) {
        int atbase = tokenStride * buffer;
        int btbase = atbase + aTokens;

        auto A_regs = aRepack ? Ao_regs : Ai_regs[buffer];
        auto B_regs = bRepack ? Bo_regs : Bi_regs[buffer];

        copyRepack(buffer);

        auto b1 = (surfaceB && isPacked(problem.B.layout)) ? 2 : 1;

        store(16 | SBID(atbase + 0), block_oword(_32x ? 16 : 8), SLM,
                Ao_addrs[0][0], A_regs[0]);
        store(16 | SBID(btbase + 0), block_oword(16), SLM, Bo_addrs[0][0],
                B_regs[0]);
        if (x48)
            store(16 | SBID(btbase + b1), block_oword(8), SLM, Bo_addrs[0][1],
                    B_regs[8]);
    };

    auto advanceLoad = [&](int buffer) {
        switch (problem.A.layout) {
            case MatrixLayout::Pc:
                if (surfaceA) {
                    add(1, Ai_addrs[buffer][0].ud(2), Ai_addrs[buffer][0].ud(2),
                            globalBuffers * 32 * unrollM / 16);
                    if (_32x)
                        add(1, Ai_addrs[buffer][1].ud(2),
                                Ai_addrs[buffer][1].ud(2),
                                globalBuffers * 32 * unrollM / 16);
                } else
                    eadd(1, Ai_addrs[buffer][0].uq(0),
                            Ai_addrs[buffer][0].uq(0),
                            globalBuffers * 32 * unrollM, strategy, state);
                break;
            case MatrixLayout::N:
                add<uint32_t>(16, Ai_addrs[buffer][0], Ai_addrs[buffer][0],
                        Ai_advance);
                if (int8)
                    add<uint32_t>(16, Ai_addrs[buffer][2], Ai_addrs[buffer][2],
                            Ai_advance);
                break;
            case MatrixLayout::T:
                add<uint32_t>(8, Ai_addrs[buffer][0], Ai_addrs[buffer][0],
                        32 * globalBuffers);
                add<uint32_t>(8, Ai_addrs[buffer][1], Ai_addrs[buffer][1],
                        32 * globalBuffers);
                break;
            default: stub();
        }

        switch (problem.B.layout) {
            case MatrixLayout::Pr:
                if (surfaceB) {
                    add(1, Bi_addrs[buffer][0].ud(2), Bi_addrs[buffer][0].ud(2),
                            globalBuffers * 32 * unrollN / 16);
                    add(1, Bi_addrs[buffer][1].ud(2), Bi_addrs[buffer][1].ud(2),
                            globalBuffers * 32 * unrollN / 16);
                    if (x48)
                        add(1, Bi_addrs[buffer][2].ud(2),
                                Bi_addrs[buffer][2].ud(2),
                                globalBuffers * 32 * unrollN / 16);
                } else {
                    eadd(1, Bi_addrs[buffer][0].uq(0),
                            Bi_addrs[buffer][0].uq(0),
                            globalBuffers * 32 * unrollN, strategy, state);
                    if (x48)
                        eadd(1, Bi_addrs[buffer][1].uq(0),
                                Bi_addrs[buffer][1].uq(0),
                                globalBuffers * 32 * unrollN, strategy, state);
                }
                break;
            case MatrixLayout::N:
                add<uint32_t>(16, Bi_addrs[buffer][0], Bi_addrs[buffer][0],
                        32 * globalBuffers);
                if (x48)
                    add<uint32_t>(16, Bi_addrs[buffer][2], Bi_addrs[buffer][2],
                            32 * globalBuffers);
                break;
            case MatrixLayout::T:
                add<uint32_t>(16, Bi_addrs[buffer][0], Bi_addrs[buffer][0],
                        Bi_advance);
                if (int8 || x48)
                    add<uint32_t>(16, Bi_addrs[buffer][2], Bi_addrs[buffer][2],
                            Bi_advance);
                break;
            default: stub();
        }
    };

    auto advanceStore = [&](int buffer = -1) {
        add(2, temp, slmBase, slmStride);
        add(1, Ao_addrs[0][0].ud(2), slmBase, slmAOff);
        add(1, Bo_addrs[0][0].ud(2), slmBase, slmBOff[0]);
        if (x48) add(1, Bo_addrs[0][1].ud(2), slmBase, slmBOff[1]);

        csel(2 | ge | f0[0], slmBase, slmBase[4](1, 1, 0), temp[0](1, 1, 0),
                temp[1]);
    };

    auto fence = [&]() { slmfence(sb15, copyFenceHeader, copyFenceHeader); };

    auto splitBarrier = [&]() {
        barrierwait();
        barriermsg(sb15, copyBarrierHeader);
    };

    // Warmup.
    if (globalBuffers > 1) cmp(1 | gt | f0[0], kCounter, 1);
    if (globalBuffers > 2) cmp(1 | gt | f0[1], kCounter, 2);
    if (globalBuffers > 3) cmp(1 | gt | f1[0], kCounter, 3);
    if (globalBuffers > 4) cmp(1 | gt | f1[1], kCounter, 4);
    if (globalBuffers > 1) {
        copyLoad(0);
        jmpi(1 | ~f0[0], smallK);
    }
    if (globalBuffers > 2) {
        copyLoad(1);
        jmpi(1 | ~f0[1], smallK);
    }
    if (globalBuffers > 3) {
        copyLoad(2);
        jmpi(1 | ~f1[0], smallK);
    }
    if (globalBuffers > 4) {
        copyLoad(3);
        jmpi(1 | ~f1[1], smallK);
    }

    auto flagLast = FlagRegister::createFromIndex(globalBuffers - 2);
    cmp(1 | le | flagLast, kCounter, globalBuffers);
    copyLoad(globalBuffers - 1);
    jmpi(1 | flagLast, smallK);

    add(1 | gt | f0[0], kCounter, kCounter, -2 * globalBuffers);

    advanceStore();
    advanceLoad(0);
    if (globalBuffers > 1) advanceLoad(1);
    if (globalBuffers > 2) advanceLoad(2);
    if (globalBuffers > 3) advanceLoad(3);

    copyStore(0);
    if (globalBuffers > 4) advanceLoad(4);

    fence();
    advanceStore(0);
    copyLoad(0);
    barriermsg(sb15, copyBarrierHeader);
    copyStore(1);

    advanceLoad(0);

    jmpi(1 | ~f0[0], bottom);
    sync.nop(SWSB<AllPipes>(1));
    mark(top);
    {
        add(1 | gt | f0[0], kCounter, kCounter, -globalBuffers);
        for (int i = 0; i < globalBuffers; i++) {
            fence();
            advanceStore((i + 1) % globalBuffers);
            copyLoad((i + 1) % globalBuffers); // move after barrier?
            splitBarrier();
            copyStore((i + 2) % globalBuffers);
            advanceLoad((i + 1) % globalBuffers);
        }
    }
    jmpi(1 | f0[0], top);
    mark(bottom);

    if (globalBuffers > 1) cmp(1 | gt | f0[0], kCounter, -globalBuffers + 1);
    if (globalBuffers > 2) cmp(1 | gt | f0[1], kCounter, -globalBuffers + 2);
    if (globalBuffers > 3) cmp(1 | gt | f1[0], kCounter, -globalBuffers + 3);
    if (globalBuffers > 4) cmp(1 | gt | f1[1], kCounter, -globalBuffers + 4);

    // Cooldown loop. All buffers but #1 loaded.
    for (int i = 1; i < globalBuffers; i++) {
        Label skipLoad;
        fence();
        advanceStore(i);
        jmpi(1 | ~FlagRegister::createFromIndex(i - 1), skipLoad);
        copyLoad(i);
        mark(skipLoad);
        splitBarrier();
        copyStore((i + 1) % globalBuffers);
        advanceLoad(i);
    }

    jmpi(1, skipSmallK);
    mark(smallK);

    advanceStore();
    copyStore(0);

    fence();
    advanceStore(0);
    barriermsg(sb15, copyBarrierHeader);
    jmpi(1, reenterSmallK);

    mark(skipSmallK);

    for (int i = 0; i < globalBuffers - 1; i++) {
        fence();
        advanceStore(i);
        splitBarrier();
        if (i == 0) mark(reenterSmallK);
        jmpi(1 | ~FlagRegister::createFromIndex(i), done);
        copyStore(i + 1);
    }

    fence();
    splitBarrier();

    mark(done);
    barrierwait();

    state.ra = saveRA;
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemm2KLoopCompute(
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state) {
    using namespace sysgemm2;

    Label top, remainder, done;
    bool _32x = (strategy.unroll[LoopM] == 32);
    bool x48 = (strategy.unroll[LoopN] == 48);
    bool keepBarHdr = strategy.skipFence || !x48;
    auto slmStride = strategy.slmSysgemmBlockSize() / 16;
    auto barrierHeader = x32::barrierHeader;

    mov(1, f0.ud(0), 0);
    mov(1, f1.ud(0), 0);
    if (x48) {
        using namespace x48;
        add(1, A_addr[1], A_addr[0], 8 * 32 / 16);
        if (_32x) {
            add(1, A_addr[2], A_addr[0], 16 * 32 / 16);
            add(1, A_addr[3], A_addr[0], 24 * 32 / 16);
        }
        add(1, B_addr[1], B_addr[0], 8 * 32 / 16);
        add(1, B_addr[2], B_addr[0], 16 * 32 / 16);
    } else {
        using namespace x32;
        add(1, A_addr[0][1], A_addr[0][0], 8 * 32 / 16);
        if (_32x) {
            add(1, A_addr[0][2], A_addr[0][0], 16 * 32 / 16);
            add(1, A_addr[0][3], A_addr[0][0], 24 * 32 / 16);
        }
        add(1, A_addr[1][0], A_addr[0][0], 0 * 32 / 16 + slmStride);
        add(1, A_addr[1][1], A_addr[0][0], 8 * 32 / 16 + slmStride);
        if (_32x) {
            add(1, A_addr[1][2], A_addr[0][0], 16 * 32 / 16 + slmStride);
            add(1, A_addr[1][3], A_addr[0][0], 24 * 32 / 16 + slmStride);
        }
        add(1, B_addr[0][1], B_addr[0][0], 8 * 32 / 16);
        add(1, B_addr[1][0], B_addr[0][0], 0 * 32 / 16 + slmStride);
        add(1, B_addr[1][1], B_addr[0][0], 8 * 32 / 16 + slmStride);
    }

    if (keepBarHdr) sysgemmBarrierPrep(InstructionModifier(), barrierHeader);

    // Warmup: signal, split barrier, load
    cmp(1 | gt | f1[1], kCounter, 1);
    add(1 | gt | f0[0], kCounter, kCounter, -5);

    if (!keepBarHdr) sysgemmBarrierPrep(InstructionModifier(), barrierHeader);
    barriermsg(sb15, barrierHeader);
    barrierwait();
    barriermsg(sb15 | f1[1], barrierHeader);

    bool oldDefaultAutoSWSB = getDefaultAutoSWSB();
    setDefaultAutoSWSB(false);
    sync.nop(SWSB<AllPipes>(1));

    load(16 | sb0, x48::A_regs[0], block_oword(16), SLM, x48::A_addr[0]);
    load(16 | sb1, x48::A_regs[8], block_oword(16), SLM, x48::A_addr[1]);
    if (_32x) {
        load(16 | sb2, x48::A_regs[16], block_oword(16), SLM, x48::A_addr[2]);
        load(16 | sb3, x48::A_regs[24], block_oword(16), SLM, x48::A_addr[3]);
    }
    load(16 | sb4, x48::B_regs[0], block_oword(16), SLM, x48::B_addr[0]);
    load(16 | sb5, x48::B_regs[8], block_oword(16), SLM, x48::B_addr[1]);
    if (x48)
        load(16 | sb6, x48::B_regs[16], block_oword(16), SLM, x48::B_addr[2]);

    zeroMatrix(x48 ? x48::C_regs : x32::C_regs, strategy);

    jmpi(1 | ~f0[0], remainder);

    mark(top);
    {
        add(1 | gt | f0[0], kCounter, kCounter, -4);
        sysgemm2Multiply(problem, strategy, state, 0);
        sysgemm2Multiply(problem, strategy, state, 1);
        sysgemm2Multiply(problem, strategy, state, 2);
        sysgemm2Multiply(problem, strategy, state, 3);
    }
    jmpi(1 | f0[0], top);

    mark(remainder);

    cmp(1 | gt | f0[0], kCounter, 1 - 5);
    cmp(1 | gt | f0[1], kCounter, 2 - 5);
    cmp(1 | gt | f1[0], kCounter, 3 - 5);
    cmp(1 | gt | f1[1], kCounter, 4 - 5);
    sysgemm2Multiply(problem, strategy, state, 0, true, f0[0], f0[1]);
    jmpi(1 | ~f0[0], done);

    sysgemm2Multiply(problem, strategy, state, 1, true, f0[1], f1[0]);
    jmpi(1 | ~f0[1], done);

    sysgemm2Multiply(problem, strategy, state, 2, true, f1[0], f1[1]);
    jmpi(1 | ~f1[0], done);

    sysgemm2Multiply(problem, strategy, state, 3, true, f1[1]);
    jmpi(1 | ~f1[1], done);

    sysgemm2Multiply(problem, strategy, state, 0, true);

    mark(done);

    setDefaultAutoSWSB(oldDefaultAutoSWSB);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemm2Multiply(const GEMMProblem &problem,
        const GEMMStrategy &strategy, GEMMState &state, int slmBuffer,
        bool cooldown, FlagRegister flagWaitLoad, FlagRegister flagSignal) {
    if (strategy.unroll[LoopN] == 48)
        sysgemm2MultiplyX48(problem, strategy, state, slmBuffer, cooldown,
                flagWaitLoad, flagSignal);
    else
        sysgemm2MultiplyX32(problem, strategy, state, slmBuffer, cooldown,
                flagWaitLoad, flagSignal);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemm2MultiplyX48(
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state, int slmBuffer, bool cooldown,
        FlagRegister flagWaitLoad, FlagRegister flagSignal) {
    using namespace sysgemm2;
    using namespace sysgemm2::x48;

    auto slmStride = strategy.slmSysgemmBlockSize() / 16;
    int16_t advance = ((slmBuffer == 3) ? -3 : 1) * slmStride;
    InstructionModifier loadMod {}, signalMod {};
    bool doWaitLoad = !cooldown || flagWaitLoad.isValid();
    bool doSignal = !cooldown || flagSignal.isValid();
    if (cooldown) {
        if (doWaitLoad) loadMod = loadMod | flagWaitLoad | any16h;
        if (doSignal) signalMod = signalMod | flagSignal | any16h;
    }

    if (strategy.unroll[LoopM] != 32) stub();

    if (doWaitLoad) {
        Label skipWait;
        if (cooldown) jmpi(1 | ~flagWaitLoad, skipWait);

        // SLM fence
        if (!strategy.skipFence && doSignal)
            slmfence(sb15 | signalMod, headerTemp, headerTemp);

        // Barrier wait
        barrierwait();

        // Barrier signal
        if (doSignal) {
            if (!strategy.skipFence) {
                sysgemmBarrierPrep(sb15.dst | signalMod, headerTemp);
                barriermsg(sb15 | SWSB<AllPipes>(1) | signalMod, headerTemp);
            } else
                barriermsg(sb15 | signalMod, headerTemp);
        }

        if (cooldown) mark(skipWait);
    }

    // Advance A0 address (note dst)
    if (doWaitLoad) add(1 | sb0.dst, A_addr[0], A_addr[0], advance);

    // Rows 0-7
    sysgemm2MultiplyChunkX48(problem, strategy, 0);

    // Advance A1 address
    if (doWaitLoad) add(1 | sb1.src, A_addr[1], A_addr[1], advance);

    // Rows 8-15
    sysgemm2MultiplyChunkX48(problem, strategy, 1);

    if (doWaitLoad) {
        // Load new A0
        load(16 | sb0 | loadMod, A_regs[0], block_oword(16), SLM, A_addr[0]);

        // Advance B, A2, A3 addresses
        add(1 | sb4.src, B_addr[0], B_addr[0], advance);
        add(1 | sb5.src, B_addr[1], B_addr[1], advance);
        add(1 | sb6.src, B_addr[2], B_addr[2], advance);

        add(1 | sb2.src, A_addr[2], A_addr[2], advance);
        add(1 | sb3.src, A_addr[3], A_addr[3], advance);
    }

    // Rows 16-23
    sysgemm2MultiplyChunkX48(problem, strategy, 2);

    // Load new A1
    if (doWaitLoad)
        load(16 | sb1 | loadMod, A_regs[8], block_oword(16), SLM, A_addr[1]);

    // Rows 24-31
    sysgemm2MultiplyChunkX48(problem, strategy, 3);

    if (doWaitLoad) {
        // Load new B data
        load(16 | sb4 | loadMod, B_regs[0], block_oword(16), SLM, B_addr[0]);
        load(16 | sb5 | loadMod, B_regs[8], block_oword(16), SLM, B_addr[1]);
        load(16 | sb6 | loadMod, B_regs[16], block_oword(16), SLM, B_addr[2]);

        // Load new A2,A3
        load(16 | sb2 | loadMod, A_regs[16], block_oword(16), SLM, A_addr[2]);
        load(16 | sb3 | loadMod, A_regs[24], block_oword(16), SLM, A_addr[3]);
    }
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemm2MultiplyX32(
        const GEMMProblem &problem, const GEMMStrategy &strategy,
        GEMMState &state, int slmBuffer, bool cooldown,
        FlagRegister flagWaitLoad, FlagRegister flagSignal) {
    using namespace sysgemm2;
    using namespace sysgemm2::x32;

    auto slmStride = strategy.slmSysgemmBlockSize() / 16;
    int16_t advance = ((slmBuffer >= 2) ? -2 : 2) * slmStride;
    InstructionModifier loadMod {}, signalMod {};
    bool doWaitLoad = !cooldown || flagWaitLoad.isValid();
    bool doSignal = !cooldown || flagSignal.isValid();
    if (cooldown) {
        if (doWaitLoad) loadMod = loadMod | flagWaitLoad | any16h;
        if (doSignal) signalMod = signalMod | flagSignal | any16h;
    }
    bool _32x = (strategy.unroll[LoopM] == 32);
    bool odd = (slmBuffer & 1);
    int tokenBase = odd ? 8 : 0;
    int otokenBase = odd ? 0 : 8;

    if (doWaitLoad) {
        Label skipWait;
        if (cooldown) jmpi(1 | ~flagWaitLoad, skipWait);

        // SLM fence
        if (!strategy.skipFence && doSignal)
            slmfence(sb15 | signalMod, fenceHeader, fenceHeader);

        add(1 | SBID(tokenBase + 0).src, A_addr[odd][0], A_addr[odd][0],
                advance); // TODO: reuse src0
        add(1 | SBID(tokenBase + 4).src, B_addr[odd][0], B_addr[odd][0],
                advance);
        add(1 | SBID(tokenBase + 5).src, B_addr[odd][1], B_addr[odd][1],
                advance);
        add(1 | SBID(tokenBase + 1).src, A_addr[odd][1], A_addr[odd][1],
                advance);
        if (_32x) {
            add(1 | SBID(tokenBase + 2).src, A_addr[odd][2], A_addr[odd][2],
                    advance);
            add(1 | SBID(tokenBase + 3).src, A_addr[odd][3], A_addr[odd][3],
                    advance);
        }

        // Barrier wait
        barrierwait();

        if (hw >= HW::XeHPG) {
            // Wait for SLM loads to return before signaling.
            sync.allwr(0x3F << tokenBase);
        }

        // Barrier signal
        if (doSignal) barriermsg(sb15 | signalMod, barrierHeader);

        if (cooldown) mark(skipWait);
    }

    if (doWaitLoad) {
        load(16 | SBID(otokenBase + 0) | loadMod, A_regs[!odd][0],
                block_oword(16), SLM, A_addr[!odd][0]);
        load(16 | SBID(otokenBase + 4) | loadMod, B_regs[!odd][0],
                block_oword(16), SLM, B_addr[!odd][0]);
        load(16 | SBID(otokenBase + 5) | loadMod, B_regs[!odd][8],
                block_oword(16), SLM, B_addr[!odd][1]);
        load(16 | SBID(otokenBase + 1) | loadMod, A_regs[!odd][8],
                block_oword(16), SLM, A_addr[!odd][1]);
        if (_32x) {
            load(16 | SBID(otokenBase + 2) | loadMod, A_regs[!odd][16],
                    block_oword(16), SLM, A_addr[!odd][2]);
            load(16 | SBID(otokenBase + 3) | loadMod, A_regs[!odd][24],
                    block_oword(16), SLM, A_addr[!odd][3]);
        }
    }

    sysgemm2MultiplyChunkX32(problem, strategy, 0, odd);
    sysgemm2MultiplyChunkX32(problem, strategy, 1, odd);
    if (_32x) {
        sysgemm2MultiplyChunkX32(problem, strategy, 2, odd);
        sysgemm2MultiplyChunkX32(problem, strategy, 3, odd);
    }
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemm2MultiplyChunkX48(
        const GEMMProblem &problem, const GEMMStrategy &strategy, int chunkA) {
    using namespace sysgemm2;
    using namespace sysgemm2::x48;
    int ao = chunkA * 8;
    int co = ao * 6;
    bool waitB = (chunkA == 0);
    bool prepB = (chunkA == 3);
    SBID sbA(chunkA);

    auto dpaswTyped = [&](InstructionModifier mod, uint8_t sdepth,
                              uint8_t rcount, const GRF &cReg, const GRF &aReg,
                              const GRF &bReg) {
        dpasw(mod, sdepth, rcount, cReg.retype(problem.Tc.ngen()),
                cReg.retype(problem.Tc.ngen()), aReg.retype(problem.Ta.ngen()),
                bReg.retype(problem.Tb.ngen()));
    };

    if (waitB) {
        /* sync.nop(sbA.dst); */ // arranged by caller
        dpaswTyped(
                8 | sb4.dst | Atomic, 8, 8, C_regs[co], A_regs[ao], B_regs[0]);
        dpaswTyped(8, 8, 8, C_regs[co + 8], A_regs[ao], B_regs[4]);
        dpaswTyped(8 | sb5.dst | Atomic, 8, 8, C_regs[co + 16], A_regs[ao],
                B_regs[8]);
        dpaswTyped(8, 8, 8, C_regs[co + 24], A_regs[ao], B_regs[12]);
        dpaswTyped(8 | sb6.dst | Atomic, 8, 8, C_regs[co + 32], A_regs[ao],
                B_regs[16]);
        dpaswTyped(8 | sbA, 8, 8, C_regs[co + 40], A_regs[ao], B_regs[20]);
    } else if (prepB) {
        dpaswTyped(
                8 | sbA.dst | Atomic, 8, 8, C_regs[co], A_regs[ao], B_regs[0]);
        dpaswTyped(8 | sb4, 8, 8, C_regs[co + 8], A_regs[ao], B_regs[4]);
        dpaswTyped(8 | Atomic, 8, 8, C_regs[co + 16], A_regs[ao], B_regs[8]);
        dpaswTyped(8 | sb5, 8, 8, C_regs[co + 24], A_regs[ao], B_regs[12]);
        dpaswTyped(8 | Atomic, 8, 8, C_regs[co + 32], A_regs[ao], B_regs[16]);
        dpaswTyped(8 | sb6, 8, 8, C_regs[co + 40], A_regs[ao], B_regs[20]);
    } else {
        dpaswTyped(
                8 | sbA.dst | Atomic, 8, 8, C_regs[co], A_regs[ao], B_regs[0]);
        dpaswTyped(8 | Atomic, 8, 8, C_regs[co + 8], A_regs[ao], B_regs[4]);
        dpaswTyped(8 | Atomic, 8, 8, C_regs[co + 16], A_regs[ao], B_regs[8]);
        dpaswTyped(8 | Atomic, 8, 8, C_regs[co + 24], A_regs[ao], B_regs[12]);
        dpaswTyped(8 | Atomic, 8, 8, C_regs[co + 32], A_regs[ao], B_regs[16]);
        dpaswTyped(8 | sbA, 8, 8, C_regs[co + 40], A_regs[ao], B_regs[20]);
    }
}

template <HW hw>
void gemm_kernel_generator_t<hw>::sysgemm2MultiplyChunkX32(
        const GEMMProblem &problem, const GEMMStrategy &strategy, int chunkA,
        bool odd) {
    using namespace sysgemm2;
    using namespace sysgemm2::x32;
    int ao = chunkA * 8;
    int co = ao * 4;
    int nchunks = strategy.unroll[LoopM] / 8;
    bool waitB = (chunkA == 0);
    bool prepB = (chunkA == nchunks - 1);
    int tokenBase = odd ? 8 : 0;
    SBID sbA(tokenBase + chunkA);
    SBID sbB0(tokenBase + 4);
    SBID sbB1(tokenBase + 5);

    auto dpaswTyped = [&](InstructionModifier mod, uint8_t sdepth,
                              uint8_t rcount, const GRF &cReg, const GRF &aReg,
                              const GRF &bReg) {
        dpasw(mod, sdepth, rcount, cReg.retype(problem.Tc.ngen()),
                cReg.retype(problem.Tc.ngen()), aReg.retype(problem.Ta.ngen()),
                bReg.retype(problem.Tb.ngen()));
    };

    if (waitB) {
        sync.nop(sbA.dst);
        dpaswTyped(8 | sbB0.dst | Atomic, 8, 8, C_regs[co], A_regs[odd][ao],
                B_regs[odd][0]);
        dpaswTyped(8, 8, 8, C_regs[co + 8], A_regs[odd][ao], B_regs[odd][4]);
        dpaswTyped(8 | sbB1.dst | Atomic, 8, 8, C_regs[co + 16],
                A_regs[odd][ao], B_regs[odd][8]);
        dpaswTyped(8 | sbA, 8, 8, C_regs[co + 24], A_regs[odd][ao],
                B_regs[odd][12]);
    } else if (prepB) {
        dpaswTyped(8 | sbA.dst | Atomic, 8, 8, C_regs[co], A_regs[odd][ao],
                B_regs[odd][0]);
        dpaswTyped(8 | sbB0, 8, 8, C_regs[co + 8], A_regs[odd][ao],
                B_regs[odd][4]);
        dpaswTyped(8 | Atomic, 8, 8, C_regs[co + 16], A_regs[odd][ao],
                B_regs[odd][8]);
        dpaswTyped(8 | sbB1, 8, 8, C_regs[co + 24], A_regs[odd][ao],
                B_regs[odd][12]);
    } else {
        dpaswTyped(8 | sbA.dst | Atomic, 8, 8, C_regs[co], A_regs[odd][ao],
                B_regs[odd][0]);
        dpaswTyped(8 | Atomic, 8, 8, C_regs[co + 8], A_regs[odd][ao],
                B_regs[odd][4]);
        dpaswTyped(8 | Atomic, 8, 8, C_regs[co + 16], A_regs[odd][ao],
                B_regs[odd][8]);
        dpaswTyped(8 | sbA, 8, 8, C_regs[co + 24], A_regs[odd][ao],
                B_regs[odd][12]);
    }
}

/**********************************************************************/
/*                             Copy Kernels                           */
/**********************************************************************/

// Initialize the interface and claim arguments.
template <HW hw>
void gemm_kernel_generator_t<hw>::copyInitInterface(
        CopyProblem &problem, CopyStrategy &strategy, CopyState &state) {
    if (strategy.barrierFreq > 0) interface.requireBarrier();

    interface.finalize();

    // Get input register assignments.
    state.inputs.S = interface.getArgumentIfExists("S");
    state.inputs.D = interface.getArgumentIfExists("D");
    state.inputs.surfaceS = interface.getArgumentSurfaceIfExists("S");
    state.inputs.surfaceD = interface.getArgumentSurfaceIfExists("D");
    state.inputs.offsetS = interface.getArgument("offset_S");
    state.inputs.offsetD = interface.getArgument("offset_D");
    state.inputs.lds = interface.getArgument("lds");
    state.inputs.ldd = interface.getArgumentIfExists("ldd");
    state.inputs.m = interface.getArgument("m");
    state.inputs.n = interface.getArgument("n");
    state.inputs.alpha_real = interface.getArgumentIfExists("alpha_real");
    state.inputs.alpha_imag = interface.getArgumentIfExists("alpha_imag");
    state.inputs.diag = interface.getArgumentIfExists("diag");
    state.inputs.blockZ = interface.getArgumentIfExists("block_z");

    state.inputs.localIDW = interface.getLocalID(0);
    state.inputs.localSizeW = interface.getLocalSize(0);
    if (strategy.zParallel) {
        state.inputs.localIDZ = interface.getLocalID(1);
        state.inputs.localSizeZ = interface.getLocalSize(1);
    }

    state.inputs.groupIDW = r0.ud(1);
    if (strategy.zParallel) state.inputs.groupIDZ = r0.ud(6);

    // Downgrade offset variables to 32-bit for non-A64 accesses.
    if (strategy.S.base.getModel() != ModelA64)
        state.inputs.offsetS = state.inputs.offsetS.d();
    if (strategy.D.base.getModel() != ModelA64)
        state.inputs.offsetD = state.inputs.offsetD.d();

    // For now, reinterpret m/n/ld/diag variables to 32-bit if they are 64-bit.
    state.inputs.m = state.inputs.m.d();
    state.inputs.n = state.inputs.n.d();
    state.inputs.lds = state.inputs.lds.ud();
    if (state.inputs.ldd.isValid()) state.inputs.ldd = state.inputs.ldd.ud();
    if (state.inputs.diag.isValid()) state.inputs.diag = state.inputs.diag.d();

    // Claim inputs.
    for (int i = 0; i < 4; i++)
        state.ra.claim(r0.uq(i));

    if (strategy.S.base.isStateless()) state.ra.claim(state.inputs.S);
    if (strategy.D.base.isStateless()) state.ra.claim(state.inputs.D);

    state.ra.claim(state.inputs.offsetS);
    state.ra.claim(state.inputs.offsetD);
    state.ra.claim(state.inputs.lds);
    if (state.inputs.ldd.isValid()) state.ra.claim(state.inputs.ldd);
    state.ra.claim(state.inputs.m);
    state.ra.claim(state.inputs.n);
    if (state.inputs.diag.isValid()) state.ra.claim(state.inputs.diag);
    if (!problem.alpha_real.fixed()) state.ra.claim(state.inputs.alpha_real);
    if (problem.Td.isComplex() && !problem.alpha_imag.fixed())
        state.ra.claim(state.inputs.alpha_imag);

    state.ra.claim(state.inputs.localIDW);
    state.ra.claim(state.inputs.localSizeW);
    if (strategy.zParallel) {
        state.ra.claim(state.inputs.localIDZ);
        state.ra.claim(state.inputs.localSizeZ);
    }

    if (strategy.zParallel) state.ra.claim(state.inputs.blockZ);
}

// Initialize the state structure.
template <HW hw>
void gemm_kernel_generator_t<hw>::copyInitState(
        CopyProblem &problem, CopyStrategy &strategy, CopyState &state) {
    if (!state.fusedGEMM.active) {
        initState(problem, strategy, state);
        copyInitInterface(problem, strategy, state);
        state.isNested = false;
    }

    state.effS = strategy.S.base.isStateless() ? state.inputs.S
                                               : state.inputs.offsetS.d();
    state.effD = strategy.D.base.isStateless() ? state.inputs.D
                                               : state.inputs.offsetD.d();

    if (!problem.alpha_real.fixed())
        problem.alpha_real = state.inputs.alpha_real;
    if (problem.Td.isComplex() && !problem.alpha_imag.fixed())
        problem.alpha_imag = state.inputs.alpha_imag;

    state.flagAP = state.raVFlag.alloc();

    state.allocEmulate64Temp(strategy.emulate);
}

// Copy kernel generation interface.
template <HW hw>
void gemm_kernel_generator_t<hw>::copy(CopyProblem problem,
        CopyStrategy strategy, const InterfaceHandler &interface_) {
    interface = interface_;
    CopyState state(hw);
    copy(problem, strategy, state);
}

template <HW hw>
void gemm_kernel_generator_t<hw>::copy(
        CopyProblem &problem, CopyStrategy &strategy, CopyState &state) {
    bool inFused = state.fusedGEMM.active;
    auto unrollW = strategy.unrollW();

    // Check layouts.
    if (!isPacked(problem.D.layout)) stub();

    if (strategy.zParallel && problem.sum) stub();

    // By default, don't use dispatch mask.
    setDefaultNoMask();
    setDefaultAutoSWSB();

    // Set up.
    copyInitState(problem, strategy, state);

    if (!strategy.S.base.isStateless())
        strategy.S.base.setIndex(state.inputs.surfaceS);
    if (!strategy.D.base.isStateless())
        strategy.D.base.setIndex(state.inputs.surfaceD);

    // Prologue.
    if (!inFused) prologue(strategy);

    // Grab fused ID if needed.
    getFusedID(unrollW, problem, strategy, state);

    // Calculate w0, the starting row/column for this thread.
    // This is the first x (if xloop = false) or y (xloop = true) value.
    state.w0 = state.ra.alloc_sub<uint32_t>(
            getHint(HintType::TempComp0, strategy));
    if (strategy.zParallel)
        state.z0 = state.ra.alloc_sub<uint32_t>(
                getHint(HintType::TempComp1, strategy));

    auto globalIDW = state.ra.alloc_sub<uint32_t>(
            getHint(HintType::TempComp1, strategy));
    auto globalIDZ = state.ra.alloc_sub<uint32_t>(
            getHint(HintType::TempComp0, strategy));

    int idWScale = inFused ? 1 : strategy.subgroupSize;
    bool multiple = (unrollW % idWScale) == 0;

    if (strategy.wgW > 0)
        mulConstant(
                1, globalIDW, state.inputs.groupIDW, strategy.wgW * idWScale);
    else
        mul(1, globalIDW, state.inputs.groupIDW, state.inputs.localSizeW.uw());
    if (strategy.zParallel) {
        if (strategy.wgZ > 0)
            mulConstant(1, globalIDZ, state.inputs.groupIDZ, strategy.wgZ);
        else
            mul(1, globalIDZ, state.inputs.groupIDZ,
                    state.inputs.localSizeZ.uw());
    }
    add(1, globalIDW, globalIDW, state.inputs.localIDW.uw(0));
    if (strategy.zParallel && (strategy.wgZ != 1))
        add(1, globalIDZ, globalIDZ, state.inputs.localIDZ.uw(0));
    if (multiple)
        mulConstant(1, state.w0, globalIDW, unrollW / idWScale);
    else {
        mulConstant(1, state.w0, globalIDW, unrollW);
        shr(1, state.w0, state.w0, log2(idWScale));
    }
    if (strategy.zParallel)
        emul(1, state.z0, globalIDZ, state.inputs.blockZ, strategy, state);

    state.ra.safeRelease(globalIDW);
    state.ra.safeRelease(globalIDZ);
    state.ra.safeRelease(state.inputs.localIDW);
    state.ra.safeRelease(state.inputs.localIDZ);
    state.ra.safeRelease(state.inputs.localSizeW);
    state.ra.safeRelease(state.inputs.localSizeZ);

    // Move r0 to acc0 if configured.
    moveR0(strategy, state);

    // Copy our slice.
    copySlice(problem, strategy, state);

    if (!inFused) {
        epilogue(strategy, state);

        padding();
    }
}

// Calculate or recalculate lds_sl/ldd_dl as needed.
template <HW hw>
void gemm_kernel_generator_t<hw>::copyCalcIncrements(const CopyProblem &problem,
        const CopyStrategy &strategy, CopyState &state, int s_load,
        int d_load) {
    // S: w0 * s_load is needed for N->Pc, T->Pr [!xLoop] N->Pr, T->Pc [xLoop]
    // D: no increment needed (always packed)    [!xLoop] ldd * d_load [xLoop]
    bool sStrided
            = (isColMajor(problem.S.layout) == isColMajor(problem.D.layout))
            ^ strategy.xLoop;

    if (sStrided || problem.reflecting()) {
        if (s_load == 0) s_load = strategy.s_load;
        if (s_load > 1) {
            if (state.lds_sl.isInvalid()) {
                state.lds_sl = state.ra.alloc_sub<uint32_t>();
                s_load *= problem.Ts.size();
            }
            emulConstant(
                    1, state.lds_sl, state.inputs.lds, s_load, strategy, state);
        }
    }

    if (strategy.xLoop) {
        if (d_load == 0) d_load = strategy.d_load;
        if (d_load > 1) {
            if (state.ldd_dl.isInvalid()) {
                state.ldd_dl = state.ra.alloc_sub<uint32_t>();
                d_load *= problem.Td.size();
            }
            emulConstant(
                    1, state.ldd_dl, state.inputs.ldd, d_load, strategy, state);
        }
    }
}

// Copy kernel generation interface.
template <HW hw>
void gemm_kernel_generator_t<hw>::copySlice(
        CopyProblem &problem, CopyStrategy &strategy, CopyState &state) {
    auto Ts = problem.Ts, Td = problem.Td;
    Label labelExit;
    Subregister lddSrc;

    // If ldd not specified, use y.
    if (state.inputs.ldd.isInvalid()) {
        state.inputs.ldd = lddSrc = (problem.D.layout == MatrixLayout::Pc)
                ? state.inputs.n
                : state.inputs.m;
        if (problem.D.crosspack > 1 || problem.sum) {
            state.inputs.ldd = state.ra.alloc_sub<uint32_t>(
                    getHint(HintType::LongTerm, strategy));
            mov(1, state.inputs.ldd, lddSrc);
            lddSrc = invalid;
        }
        if (problem.D.crosspack > 1) {
            add(1, state.inputs.ldd, state.inputs.ldd, problem.D.crosspack - 1);
            and_(1, state.inputs.ldd, state.inputs.ldd,
                    ~uint32_t(problem.D.crosspack - 1));
        }
        if (problem.sum)
            add(1, state.inputs.ldd, state.inputs.ldd,
                    problem.Tsum.size() / problem.Td.size());
    }

    // Duplicate alpha if configured.
    if (strategy.duplicateAlpha) { duplicateScalar(problem.alpha_real, state); }

    // For fused kernels, compute 2 * unrollW - fusedID for use in several places.
    Subregister unrollWRem;
    if (strategy.fused) {
        unrollWRem = state.ra.alloc_sub<uint32_t>(
                getHint(HintType::TempComp0, strategy));
        add(1, unrollWRem, -state.fusedID, uint16_t(2 * strategy.unrollW()));
    }

    // Align code paths.
    bool mLoop = isColMajor(problem.D.layout) == strategy.xLoop;
    auto z = mLoop ? state.inputs.m : state.inputs.n;
    Subregister z0;

    // Handle z blocking.
    if (strategy.zParallel) {
        z0 = state.z0;
        add(1 | le | f0[1], z, z, -z0);
        min_(1, z, z, state.inputs.blockZ);
        state.ra.safeRelease(state.inputs.blockZ);
    }

    // Compute base addresses for S, D.
    //   S += w0 + z0 * lds (N->Pc, T->Pr) z0 + w0 * lds (N->Pr, T->Pc) [swapped if xLoop = true]
    bool sStrided
            = (isColMajor(problem.S.layout) == isColMajor(problem.D.layout))
            ^ strategy.xLoop;
    auto incC = sStrided ? state.w0 : z0;
    auto incS = sStrided ? z0 : state.w0;

    if (incC.isValid())
        eadd(1, state.inputs.offsetS, state.inputs.offsetS, incC, strategy,
                state);
    if (incS.isValid()) {
        Subregister temp = state.ra.alloc_sub(state.inputs.offsetS.getType(),
                getHint(HintType::TempComp1, strategy));
        emul(1, temp, incS, state.inputs.lds, strategy, state);
        eadd(1, state.inputs.offsetS, state.inputs.offsetS, temp, strategy,
                state);
        state.ra.safeRelease(temp);
    }

    // Quick exit if no work to do.
    if (strategy.zParallel) jmpi(1 | f0[1], labelExit);

    // D += align_up(x0, unroll) * ldd + y0 * unroll + (x0 % unroll) * crosspack
    {
        Subregister temp0 = state.ra.alloc_sub(state.inputs.offsetD.getType(),
                getHint(HintType::TempComp0, strategy));
        Subregister temp1 = state.ra.alloc_sub(state.inputs.offsetD.getType(),
                getHint(HintType::TempComp1, strategy));
        Subregister temp2 = state.ra.alloc_sub<uint32_t>(
                getHint(HintType::TempComp0, strategy));
        auto x0 = strategy.xLoop ? z0 : state.w0;
        auto y0 = strategy.xLoop ? state.w0 : z0;
        bool splitX = strategy.unrollX < problem.D.packSize;

        if (x0.isValid()) {
            if (splitX) {
                modExt(temp2, temp1.ud(), x0, problem.D.packSize, strategy,
                        state);
                emul(1, temp0, temp1.ud(), state.inputs.ldd, strategy, state);
                mulConstant(1, temp2, temp2, problem.D.crosspack);
            } else
                emul(1, temp0, x0, state.inputs.ldd, strategy, state);
        }
        if (y0.isValid())
            emulConstant(1, temp1, y0, problem.D.packSize, strategy, state);
        if (x0.isValid())
            eadd(1, state.inputs.offsetD, state.inputs.offsetD, temp0, strategy,
                    state);
        if (y0.isValid())
            eadd(1, state.inputs.offsetD, state.inputs.offsetD, temp1, strategy,
                    state);
        if (x0.isValid() && splitX)
            eadd(1, state.inputs.offsetD, state.inputs.offsetD, temp2, strategy,
                    state);

        state.ra.safeRelease(temp0);
        state.ra.safeRelease(temp1);
        state.ra.safeRelease(temp2);
    }

    state.ra.safeRelease(z0);
    state.z0 = invalid;

    // Calculate increments.
    copyCalcIncrements(problem, strategy, state);

    // Calculate remainders for w loop as needed.
    if (!strategy.xLoop
            && (strategy.remHandlingX != RemainderHandling::Ignore)) {
        auto x = (problem.D.layout == MatrixLayout::Pc) ? state.inputs.m
                                                        : state.inputs.n;
        state.remainderX = state.ra.alloc_sub<uint32_t>();
        add(1 | sat, state.remainderX, -state.w0, x);
        if (strategy.remHandlingX == RemainderHandling::Split) {
            if (strategy.fused)
                cmp(1 | lt | state.flagAP, null.ud(), state.remainderX,
                        unrollWRem);
            else
                cmp(1 | lt | state.flagAP, null.ud(), state.remainderX,
                        strategy.unrollX);
            mov(1 | ~state.flagAP, state.remainderX, strategy.unrollX);
        } else
            min_(1, state.remainderX, state.remainderX, strategy.unrollX);
    }
    if (strategy.xLoop
            && (strategy.remHandlingY != RemainderHandling::Ignore)) {
        auto y = (problem.D.layout == MatrixLayout::Pc) ? state.inputs.n
                                                        : state.inputs.m;
        state.remainderY = state.ra.alloc_sub<uint32_t>();
        add(1 | sat, state.remainderY, -state.w0, y);
        if (strategy.remHandlingY == RemainderHandling::Split) {
            if (strategy.fused)
                cmp(1 | lt | state.flagAP, null.ud(), state.remainderY,
                        unrollWRem);
            else
                cmp(1 | lt | state.flagAP, null.ud(), state.remainderY,
                        strategy.unrollY);
            mov(1 | ~state.flagAP, state.remainderY, strategy.unrollY);
        } else
            min_(1, state.remainderY, state.remainderY, strategy.unrollY);
    }

    // Convert lds to bytes.
    emulConstant(
            1, state.inputs.lds, state.inputs.lds, Ts.size(), strategy, state);

    // Add offsets to base pointers for stateless accesses.
    emulConstant(1, state.inputs.offsetS, state.inputs.offsetS, Ts.size(),
            strategy, state);
    emulConstant(1, state.inputs.offsetD, state.inputs.offsetD, Td.size(),
            strategy, state);

    if (strategy.S.base.isStateless()) {
        eadd(1, state.inputs.S, state.inputs.S, state.inputs.offsetS, strategy,
                state);

        state.ra.safeRelease(state.inputs.offsetS);
    } else
        state.effS1 = state.offsetS1;

    if (strategy.D.base.isStateless()) {
        eadd(1, state.inputs.D, state.inputs.D, state.inputs.offsetD, strategy,
                state);
        state.ra.safeRelease(state.inputs.offsetD);
    }

    state.ra.safeRelease(unrollWRem);

    if (!copyBody(problem, strategy, state)) {
        lastException ? std::rethrow_exception(lastException)
                      : throw std::runtime_error("Could not generate kernel.");
    }

    mark(labelExit);
}

// Wrapper around copyBodyRemCheck, checking for optimally-aligned S.
template <HW hw>
bool gemm_kernel_generator_t<hw>::copyBody(
        CopyProblem &problem, CopyStrategy &strategy, CopyState &state) {
    if (!is_zero_or_pow2(strategy.optionalAlignS)) stub();

    bool success;

    if (strategy.optionalAlignS == 0)
        success = copyBodyRemCheck(problem, strategy, state);
    else {
        Label labelUnaligned, labelEnd;

        status << "S alignment check" << status_stream::endl;
        and_(1 | nz | f0[1], null.uw(), state.effS.uw(),
                uint16_t(strategy.optionalAlignS - 1));
        and_(1 | nz | f1[1], null.uw(), state.inputs.lds.uw(),
                uint16_t(strategy.optionalAlignS - 1));
        ejmpi(1 | f0[1] | anyv, labelUnaligned);

        auto modProblem = problem;
        modProblem.S.setAlignment(strategy.optionalAlignS);

        status << "S aligned to " << strategy.optionalAlignS << ':'
               << status_stream::endl;
        success = copyBodyRemCheck(modProblem, strategy, state);

        if (state.isNested)
            jmpi(1, labelEnd);
        else
            epilogue(strategy, state);

        mark(labelUnaligned);

        status << "S unaligned" << status_stream::endl;
        success = success && copyBodyRemCheck(problem, strategy, state);

        mark(labelEnd);
    }

    return success;
}

// Wrapper around copyBodyInternal, handling split remainders.
template <HW hw>
bool gemm_kernel_generator_t<hw>::copyBodyRemCheck(
        CopyProblem &problem, CopyStrategy &strategy, CopyState &state) {
    auto CopyStrategy::*remHandlingW
            = (strategy.xLoop ? &CopyStrategy::remHandlingY
                              : &CopyStrategy::remHandlingX);
    bool wSplit = strategy.*remHandlingW == RemainderHandling::Split;
    bool success;

    if (!wSplit)
        success = copyBodyInternal(problem, strategy, state);
    else {
        CopyStrategy modStrategy = strategy;
        Label wRemBegin, wRemEnd;
        jmpi(1 | state.flagAP, wRemBegin);

        status << "Generating "
               << "xy"[strategy.xLoop] << " non-remainder kernel"
               << status_stream::endl;
        modStrategy.*remHandlingW = RemainderHandling::Ignore;
        success = copyBodyInternal(problem, modStrategy, state);

        if (state.isNested)
            jmpi(1, wRemEnd);
        else
            epilogue(strategy, state);

        modStrategy.*remHandlingW = RemainderHandling::KnownRemainder;

        bool recalc = false;

        if (strategy.xLoop && !isTransposing(modStrategy.D.accessType)
                && !isLargeCrosspack(problem.Td, problem.D.crosspack)) {
            // Change D access to use scattered stores so masking is possible.
            modStrategy.D.accessType = AccessType::Scattered;
            modStrategy.S.accessType = isTransposing(modStrategy.S.accessType)
                    ? AccessType::Block
                    : AccessType::Scattered;
        }
        if (!strategy.xLoop && !strategy.S.padded) {
            // Check if we need to change s_load/d_load.
            if (strategy.s_load > strategy.s_load_masked) {
                status << "Downgrading s_load: " << strategy.s_load << " -> "
                       << strategy.s_load_masked << status_stream::endl;
                modStrategy.s_load = strategy.s_load_masked;
                recalc = true;
            }
            if (strategy.d_load > strategy.d_load_masked) {
                status << "Downgrading d_load: " << strategy.d_load << " -> "
                       << strategy.d_load_masked << status_stream::endl;
                modStrategy.d_load = strategy.d_load_masked;
                recalc = true;
            }
        }

        status << "Generating "
               << "xy"[strategy.xLoop] << " remainder kernel"
               << status_stream::endl;
        mark(wRemBegin);
        if (recalc) copyCalcIncrements(problem, modStrategy, state);
        success = success && copyBodyInternal(problem, modStrategy, state);
        mark(wRemEnd);
    }

    return success;
}

// Body of copy kernel.
template <HW hw>
bool gemm_kernel_generator_t<hw>::copyBodyInternal(
        CopyProblem &problem, CopyStrategy &strategy, CopyState &state) {
    Label lZLoopBegin, lZLoopEnd;
    constexpr auto SD_copies = 1;
    vector<MaskAssignment> masks;
    bool share;

    auto Ts = problem.Ts, Td = problem.Td, Tsum = problem.Tsum;
    const bool byColumn = isColMajor(problem.D.layout);
    const bool sStrided
            = (isColMajor(problem.S.layout) == isColMajor(problem.D.layout))
            ^ strategy.xLoop;
    const bool mLoop = isColMajor(problem.D.layout) == strategy.xLoop;

    const bool reflecting = false;
    const bool triRemOnly = false;

    auto crosspack = problem.D.crosspack;

    // Release w0 -- no longer needed.
    state.ra.safeRelease(state.w0);

    // Get flag register for complex swizzles for XeHP+.
    if (hw >= HW::XeHP && Ts.isComplex()) {
        state.flagSwizzle = state.raVFlag.alloc();
        state.raVFlag.unlock(state.flagSwizzle);
    }

    MatrixAddressingStrategy S_strategyReflected = strategy.S;
    vector<RegisterBlock> S_layoutReflected;

    // Decide what remainder handling needs to be done.
    bool remainderX = (strategy.remHandlingX != RemainderHandling::Ignore);
    bool remainderY = (strategy.remHandlingY != RemainderHandling::Ignore);
    bool remainderZ = strategy.xLoop ? remainderX : remainderY;

    bool checkYRem1 = strategy.xLoop && remainderY && strategy.unrollY == 1;
    VirtualFlag flagYRem1;

    remainderY &= !checkYRem1;

    // Get register layouts for S and D.
    int nms, nmd, nns, nnd;
    auto setup = [&](int s_load, int d_load, Subregister S_addr0,
                         Subregister S1_addr0, Subregister D_addr0,
                         bool handleRemZ) -> bool {
        bool remM = remainderX && (!strategy.xLoop || handleRemZ);
        bool remN = remainderY && (strategy.xLoop || handleRemZ);
        Subregister remainders[3]
                = {state.remainderX, state.remainderY, Subregister {}};

        if (!strategy.xLoop) {
            nmd = nms = strategy.unrollX;
            nnd = d_load;
            nns = s_load;
        } else {
            nnd = nns = strategy.unrollY;
            nmd = d_load;
            nms = s_load;
        }

        if (!byColumn) {
            std::swap(nms, nns);
            std::swap(nmd, nnd);
            std::swap(remM, remN);
            std::swap(remainders[0], remainders[1]);
        }

        auto remM_S = remM && !strategy.S.padded;
        auto remN_S = remN && !strategy.S.padded;
        auto remM_D = remM && !strategy.D.padded && !byColumn;
        auto remN_D = remN && !strategy.D.padded && byColumn;

        auto sMaxRBlock = 0;
        auto sMaxCBlock = 0;

        if (!getRegLayout(Ts, state.S_layout, nms, nns, remM_S, remN_S, false,
                    true, sMaxRBlock, sMaxCBlock, problem.S, strategy.S))
            return false;
        if (!getRegLayout(Td, state.D_layout, nmd, nnd, remM_D, remN_D, true,
                    true, 0, 0, problem.D, strategy.D))
            return false;

        if (hasFragmenting(state.S_layout) || hasFragmenting(state.D_layout)) {
            status << "Fragmenting not supported." << status_stream::endl;
            return false;
        }

        bool success = true;
        if (checkYRem1) {
            flagYRem1 = state.raVFlag.allocVirtual();
            success &= !(state.raVFlag.isVirtual(flagYRem1)
                    && state.vflagStorage.isInvalid());
        }

        // Find and load any needed mask registers.
        success = success
                && assignMasks(state.S_layout, LoopM, LoopN, masks, state)
                && assignMasks(state.D_layout, LoopM, LoopN, masks, state);

        if (!success && state.vflagStorage.isInvalid()) {
            status << "Retrying with virtual flags." << status_stream::endl;
            allocVFlagStorage(strategy, state);
            success = assignMasks(state.S_layout, LoopM, LoopN, masks, state)
                    && assignMasks(state.D_layout, LoopM, LoopN, masks, state);
        }

        if (!success) return false;

        loadMasks(masks, remainders, strategy, state);

        if (!strategy.xLoop && !remM_D && !remN_D
                && strategy.remHandlingX != RemainderHandling::Ignore) {
            // Find a mask to use for destination layout for y loop remainders.
            VirtualFlag flag;
            bool found = false;
            for (auto &mask : masks)
                if (mask.var == (byColumn ? LoopM : LoopN) && mask.offset == 0)
                    flag = mask.flag, found = true;
            if (!found) stub();
            for (auto &block : state.D_layout) {
                if (block.simdSize > 16) stub();
                block.flag = flag;
                block.flagAny = true;
            }
        } else if (checkYRem1) {
            // Create mask for y remainder for x-loop kernels with unrollY == 1, and
            // apply it by hand to both source and destination.
            RegData regYRem1 = getMaskFlag(flagYRem1, state);
            FlagRegister testFlag;

            testFlag = regYRem1.isARF()
                    ? reinterpret_cast<FlagRegister &>(regYRem1)
                    : f0[1];

            cmp(16 | gt | testFlag, state.remainderY, 0);

            for (auto &mask : masks)
                mov(1 | ~testFlag, getMaskFlag(mask.flag, state), 0);
            if (!regYRem1.isARF()) mov(1, regYRem1, testFlag);

            for (auto &block : state.S_layout)
                if (!block.flag) block.flag = flagYRem1;
            for (auto &block : state.D_layout) {
                if (block.simdSize > 16) stub();
                block.flag = flagYRem1;
            }
        }

        // Match source layout to destination layout if possible, so that they can share registers.
        share = (Ts == Td) && (s_load == d_load)
                && matchLayoutsBidirectional(
                        Ts, state.S_layout, state.D_layout);

        // Allocate address registers.
        allocAddrRegs(state.S_addrs, state.S_layout, problem.S, strategy.S,
                state,
                getHint(share ? HintType::DAddr : HintType::SAddr, strategy));
        allocAddrRegs(state.D_addrs, state.D_layout, problem.D, strategy.D,
                state, getHint(HintType::DAddr, strategy));

        // Set up address registers.
        setupAddr(Ts, state.S_addrs, S_addr0, state.S_layout, state.inputs.lds,
                problem.S, strategy.S, strategy, state);
        setupAddr(Td, state.D_addrs, D_addr0, state.D_layout, state.inputs.ldd,
                problem.D, strategy.D, strategy, state);

        // Allocate data registers.
        int S_regCount = getRegCount(state.S_layout);
        int D_regCount = getRegCount(state.D_layout);

        state.D_regs = state.ra.alloc_range(
                D_regCount, getHint(HintType::D, strategy));
        state.S_regs = share ? state.D_regs
                             : state.ra.alloc_range(S_regCount,
                                     getHint(HintType::S, strategy));

        // Prepare for summation.
        // Clean up previous sums if any, and try to reuse their registers.
        // Allocate and zero new sum registers as needed.
        if (problem.sum) {
            if (strategy.xLoop) stub();

            vector<RegisterBlock> Ds_layout;
            makeSumLayout(!byColumn, Td, state.D_layout, Tsum, Ds_layout,
                    strategy, state);

            bool alloc = state.Ds_layout.empty()
                    || !matchLayouts(Tsum, Ds_layout, state.Ds_layout);
            if (!state.Ds_layout.empty() && alloc) {
                horizontalAdd(
                        !byColumn, Tsum, state.Ds_regs.back(), state.Ds_layout);
                alloc = !matchLayouts(Tsum, Ds_layout, state.Ds_layout);
            }
            if (alloc) {
                state.Ds_layout = std::move(Ds_layout);
                auto Ds_regs
                        = state.ra.alloc_range(getRegCount(state.Ds_layout));
                zeroMatrix(Ds_regs, strategy);
                state.Ds_regs.push_back(Ds_regs);
            }
        }

        return true;
    };

    auto cleanup = [&]() {
        state.raVFlag.safeRelease(flagYRem1);
        safeReleaseMaskAssignments(masks, state);
        safeReleaseRanges(state.S_addrs, state);
        safeReleaseRanges(state.D_addrs, state);

        state.ra.safeRelease(state.S_regs);
        state.ra.safeRelease(state.D_regs);
        // Sum registers not freed here.

        state.S_layout.clear();
        state.D_layout.clear();
    };

    auto doSLoad = [&](const vector<RegisterBlock> &layout,
                           const vector<RegisterBlock> &layoutReflect,
                           const vector<GRFRange> &addrs,
                           const vector<GRFRange>(&addrSrcs)[2], int z0,
                           int s_load, int S_copy, bool checkRem) {
        bool unlockAP = false;
        Label skipLoad;
        checkRem &= (z0 > 0);

        if (checkRem) {
            zeroMatrix(state.S_regs, strategy);
            unlockAP = !state.raVFlag.lock(state.flagAP);
            state.usePhysicalFlag(state.flagAP);
            cmp(1 | le | state.flagAP, state.Z, uint16_t(z0));
            jmpi(1 | state.flagAP, skipLoad);
        }

        {
            loadMatrix(state.S_regs, layout, problem.S, strategy.S, addrs,
                    strategy, state);
        }

        auto addrsFixed = reflecting ? &addrSrcs[0] : &addrs;
        auto addrsStrided = reflecting ? &addrSrcs[1] : nullptr;
        auto layoutFixed = &layout;
        auto layoutStrided = &layoutReflect;

        if (sStrided) {
            std::swap(addrsFixed, addrsStrided);
            std::swap(layoutFixed, layoutStrided);
        }

        {
            if (addrsStrided)
                incAddr(*addrsStrided,
                        (s_load == 1) ? state.inputs.lds : state.lds_sl,
                        *layoutStrided, problem.S, strategy.S, strategy, state);
            if (addrsFixed)
                incAddr(*addrsFixed, uint16_t(s_load * Ts), *layoutFixed,
                        problem.S, strategy.S, strategy, state);
        }
        if (checkRem) {
            if (unlockAP) state.raVFlag.unlock(state.flagAP);
            mark(skipLoad);
        }
    };

    auto doDStore = [&](const vector<RegisterBlock> &layout,
                            const vector<GRFRange> &addrs, int d_load,
                            int D_copy) {
        storeMatrix(state.D_regs, layout, problem.D, strategy.D, addrs,
                strategy, state);
        if (problem.sum)
            accumulateSum(!byColumn, Td, state.D_regs, layout, Tsum,
                    state.Ds_regs.back(), state.Ds_layout, strategy, state);
        if (strategy.xLoop) {
            if (d_load >= strategy.unrollX)
                incAddr(addrs, state.ldd_dl, layout, problem.D, strategy.D,
                        strategy, state);
            else
                incAddr(addrs, uint16_t(d_load * Td), layout, problem.D,
                        strategy.D, strategy, state);
        } else {
            auto D_tileX = byColumn ? problem.D.tileR : problem.D.tileC;
            auto D_tileY = byColumn ? problem.D.tileC : problem.D.tileR;
            auto effPS = (d_load < D_tileY) ? D_tileX : problem.D.packSize;
            incAddr(addrs, uint16_t(d_load * effPS * Td), layout, problem.D,
                    strategy.D, strategy, state);
        }
    };

    // Start generating code.

    // Reuse z for the loop counter.
    // If z unroll > 1, the loop counter will be offset by (unrollZ - 1) during the main loop,
    //  unless there's no z remainder.
    // For triangular-ended copies, offset by an additional unrollW [2x unrollX if fused] to push triangular handling to remainder loop.
    state.Z = mLoop ? state.inputs.m : state.inputs.n;

    auto unrollZ = strategy.unrollZ();
    auto offsetZ = (remainderZ || triRemOnly) ? (unrollZ - 1) : 0;

    if (offsetZ == 0)
        cmp(1 | le | state.flagAP, null.d(), state.Z, int16_t(0));
    else
        add(1 | le | state.flagAP, state.Z, state.Z, int16_t(-offsetZ));

    // Get flag register and loop counter for barrier check if needed.
    FlagRegister flagBarrier;
    Subregister bcount;
    if (strategy.barrierFreq > 0) {
        flagBarrier = state.raVFlag.alloc();

        // Can use main loop counter if barrierFreq and unrollZ both powers of 2.
        if (!is_zero_or_pow2(strategy.barrierFreq * unrollZ)) {
            bcount = state.ra.alloc_sub<uint32_t>();
            mov(1, bcount, uint16_t(strategy.barrierFreq));
        }
    }

    // Setup for main loop.
    if (!setup(strategy.s_load, strategy.d_load, state.effS, state.effS1,
                state.effD, false))
        return false;

    bool lateZLoopCheck = state.vflagStorage.isValid();
    if (lateZLoopCheck) {
        // Release flags for use by vflags. Note flagReflect is not released.
        state.raVFlag.unlock(state.flagAP);
        if (flagBarrier.isValid()) state.raVFlag.unlock(flagBarrier);
    }

    // Bail to remainder loop if no main loops.
    jmpi(1 | state.flagAP, lZLoopEnd);

    // Loop check code.
    auto zLoopCheck = [&](int unrollZ, bool enableBarriers) {
        // Use the all-purpose flag for z loop query.
        add(1 | gt | state.flagAP, state.Z, state.Z, int16_t(-unrollZ));

        // Check for barrier if requested.
        if (enableBarriers) {
            if (bcount.isInvalid())
                and_(1 | ze | flagBarrier, null.ud(), state.Z,
                        uint16_t(unrollZ * strategy.barrierFreq - unrollZ));
            else
                add(1 | ze | flagBarrier, bcount, bcount, int16_t(-1));
        }
    };

    // Lambdas used in zLoopBody (moved outside to w/a GCC bug)
    auto mulAlphaFixed = [&](int esize, RegData r) {
        mul(esize, r, r, problem.alpha_real.getRegAvoiding(hw, r));
    };

    auto mulAlpha = [&](int esize, RegData r) {
        mul(esize, r, r, cast(Ts.real(), problem.alpha_real));
    };

    auto signChange = [&](int esize, RegData r) {
        auto ne = elementsPerGRF<uint32_t>(hw);
        xor_<uint32_t>(esize, r, r,
                (ne < esize) ? state.signChange[0](0, ne, 1)
                             : state.signChange[0](1));
    };

    // z loop: returns true on success.
    int S_copy = 0, D_copy = 0;
    auto zLoopBody = [&](const vector<RegisterBlock> &S_layout,
                             const vector<RegisterBlock> &S_layoutReflected,
                             const vector<RegisterBlock> &D_layout,
                             const vector<GRFRange> &S_addrs,
                             const vector<GRFRange>(&S_addrSrcs)[2],
                             const vector<GRFRange> &D_addrs, int unrollZ,
                             int s_load, int d_load, bool enableBarriers,
                             bool enableTri, bool needSRem = false,
                             bool noLoop = false) {
        int us = s_load, ud = 0;
        int uZLoopCheck = noLoop ? -1 : lateZLoopCheck ? (unrollZ - 1) : 0;
        bool dMasked = hasMasking(D_layout);

        for (int u = 0; u < unrollZ; u++, us++, ud++) {
            // Maintain us (= u % s_load) and ud (= u % d_load) counters.
            bool loadS = false;
            if (us == s_load) {
                us = 0;
                loadS = true;
            }

            if (ud == d_load) ud = 0;
            bool storeD = ((ud + 1) == d_load);

            // Test loop counter on first iteration (lateZLoopCheck == false)
            if ((u == uZLoopCheck) && !lateZLoopCheck)
                zLoopCheck(unrollZ, enableBarriers);

            // Load S every s_load loops, and copy as necessary.
            if (loadS) {
                doSLoad(S_layout, S_layoutReflected, S_addrs, S_addrSrcs, u,
                        s_load, S_copy, needSRem);

                // Copy S registers to D registers, or perform in-place scaling/transposition.
                if (!share) {
                    int dOffR = 0, dOffC = 0;
                    (byColumn ? dOffC : dOffR) = ud;

                    if (!copyRegisters(Ts, Td, S_layout, D_layout, state.S_regs,
                                state.D_regs, dOffR, dOffC, problem.alpha_real,
                                problem.alpha_imag, problem.conjugate, strategy,
                                state))
                        return false;
                } else {
                    if (!problem.alpha_real.fixed())
                        map(hw, Ts.real(), state.S_regs, S_layout, strategy,
                                mulAlphaFixed);
                    else if ((problem.alpha_real != 1)
                            && (problem.alpha_real != -1))
                        map(hw, Ts.real(), state.S_regs, S_layout, strategy,
                                mulAlpha);
                    if (problem.conjugate || (problem.alpha_real == -1))
                        map<uint32_t>(hw, state.S_regs, S_layout, strategy,
                                signChange);
                }

                // Advance S copy counter.
                if (++S_copy == SD_copies) S_copy = 0;
            }

            // Test loop counter on last iteration (lateZLoopCheck == true) if D unmasked.
            if ((u == uZLoopCheck) && lateZLoopCheck && !dMasked)
                zLoopCheck(unrollZ, enableBarriers);

            // Store D every d_load loops.
            if (storeD) {
                doDStore(D_layout, D_addrs, d_load, D_copy);
                if (++D_copy == SD_copies) D_copy = 0;
            }

            // Test loop counter at very end (lateZLoopCheck == true) if D masked.
            if ((u == uZLoopCheck) && lateZLoopCheck && dMasked)
                zLoopCheck(unrollZ, enableBarriers);
        }

        // Forget about active vflags.
        state.wipeActiveVFlags();

        return true;
    };

    syncall();

    mark(lZLoopBegin);
    {
        if (!zLoopBody(state.S_layout, S_layoutReflected, state.D_layout,
                    state.S_addrs, state.S_addrSrcs, state.D_addrs, unrollZ,
                    strategy.s_load, strategy.d_load, strategy.barrierFreq > 0,
                    !triRemOnly))
            return false;

        if (strategy.barrierFreq == 0)
            jmpi(1 | state.flagAP, lZLoopBegin);
        else {
            jmpi(1 | ~state.flagAP, lZLoopEnd);
            jmpi(1 | ~flagBarrier, lZLoopBegin);

            auto temp = state.ra.alloc();
            if (!bcount.isInvalid())
                mov(1, bcount, uint16_t(strategy.barrierFreq));

            GRF r0_info;
            bool freeR0Info = false;

            if (state.r0_info.isARF()) {
                r0_info = state.ra.alloc();
                mov<uint32_t>(8, r0_info, state.r0_info);
                freeR0Info = true;
            } else
                r0_info = GRF {state.r0_info.getBase()};

            barrier(temp, r0_info);
            state.ra.safeRelease(temp);
            if (freeR0Info) state.ra.safeRelease(r0_info);

            jmpi(1, lZLoopBegin);
        }
    }
    mark(lZLoopEnd);

    state.raVFlag.safeRelease(flagBarrier);
    state.ra.safeRelease(bcount);

    // z remainder loop.
    if (offsetZ) {
        // Undo offseting on the z loop counter and check for zero remainder loops.
        add(1 | le | state.flagAP, state.Z, state.Z, uint16_t(offsetZ));

        // Get the current S, D addresses.
        Subregister S_addr0, S1_addr0, D_addr0;
        int S_shift, D_shift;
        S_addr0 = getOriginAddr(
                state.S_layout, state.S_addrs, problem.S, strategy.S, &S_shift);

        D_addr0 = getOriginAddr(
                state.D_layout, state.D_addrs, problem.D, strategy.D, &D_shift);

        auto unshiftAddr0 = [&]() {
            if (S_shift) shl(1, S_addr0, S_addr0, S_shift);
            if (D_shift) shl(1, D_addr0, D_addr0, D_shift);
        };

        // Prepare for potential new layout.
        vector<RegisterBlock> S_layout1, S_layout1Reflect, D_layout1;
        vector<GRFRange> S_addrs1, S_addrSrcs1[2], D_addrs1;

        // First, try handling the whole remainder, all at once.
        bool wholeRem = false, fragmented = false;
        auto newSLoad = strategy.s_load, newDLoad = strategy.d_load;
        auto saveSStrategy = strategy.S;
        bool largeDCrosspack = isLargeCrosspack(Td, problem.D.crosspack);

        if (S_addr0.isValid() && D_addr0.isValid() && !largeDCrosspack) {
            auto saveState = state;
            auto saveMasks = masks;
            (strategy.xLoop ? state.remainderX : state.remainderY) = state.Z;
            pushStream();
            try {
                cleanup();
                state.ra.claim(S_addr0);
                state.ra.claim(D_addr0);
                unshiftAddr0();

                wholeRem = setup(strategy.s_load, strategy.d_load, S_addr0,
                        S1_addr0, D_addr0, true);

                state.ra.release(S_addr0);
                state.ra.release(D_addr0);
            } catch (...) {}
            if (!wholeRem) {
                masks = saveMasks;
                state = saveState;
            }
            wholeRem ? appendCurrentStream() : discardStream();
        }

        // If that doesn't work, retry with minimal unroll.
        if (!wholeRem) {
            newSLoad = 1;
            newDLoad = crosspack;
            bool unshare = share && (newSLoad != newDLoad);

            // Fragment the S, D layouts, taking the first row/column of each.
            vector<int> indices;
            fragmented = (!unshare && !largeDCrosspack
                    && getSubblocks(Ts, S_layout1, indices, state.S_layout,
                            !mLoop, 0, newSLoad, strategy.S.padded, problem.S,
                            strategy.S)
                    && getSubblocks(Ts, S_layout1Reflect, S_layoutReflected,
                            mLoop, 0, newSLoad, strategy.S.padded, problem.S,
                            S_strategyReflected)
                    && getSubblocks(Td, D_layout1, D_addrs1, state.D_layout,
                            state.D_addrs, !mLoop, 0, newDLoad, false,
                            problem.D, strategy.D));

            if (fragmented) {
                // Select source address registers from the fragments.
                for (auto b : indices)
                    S_addrs1.push_back(state.S_addrs[b]);
                // Update sizes.
                (mLoop ? nms : nns) = newSLoad;
                (mLoop ? nmd : nnd) = newDLoad;
            } else {
                // Fragmentation failed. Start fresh.
                if (S_addr0.isInvalid() || D_addr0.isInvalid()) return false;

                cleanup();
                state.ra.claim(S_addr0);
                state.ra.claim(D_addr0);
                unshiftAddr0();

                if (largeDCrosspack) {
                    strategy.S.accessType = isTransposing(strategy.S.accessType)
                            ? AccessType::Block
                            : strategy.S.base.isStateless()
                                    ? AccessType::Scattered
                                    : AccessType::ChannelScattered;
                }

                if (!setup(newSLoad, newDLoad, S_addr0, S1_addr0, D_addr0,
                            false))
                    return false;

                state.ra.release(S_addr0);
                state.ra.release(D_addr0);
            }

            if (crosspack > 1) {
                lateZLoopCheck = true;
                copyCalcIncrements(
                        problem, strategy, state, newSLoad, newDLoad);
            }
        }

        // Emit z remainder loop.
        Label lZRemLoopBegin, lZRemLoopEnd;
        jmpi(1 | state.flagAP, lZRemLoopEnd);
        mark(lZRemLoopBegin);
        wholeRem ? zLoopBody(state.S_layout, S_layoutReflected, state.D_layout,
                state.S_addrs, state.S_addrSrcs, state.D_addrs, unrollZ,
                newSLoad, newDLoad, false, true, false, !triRemOnly)
                 : fragmented
                        ? zLoopBody(S_layout1, S_layout1Reflect, D_layout1,
                                S_addrs1, S_addrSrcs1, D_addrs1, crosspack,
                                newSLoad, newDLoad, false, true, crosspack > 1)
                        : zLoopBody(state.S_layout, S_layoutReflected,
                                state.D_layout, state.S_addrs, state.S_addrSrcs,
                                state.D_addrs, crosspack, newSLoad, newDLoad,
                                false, true, crosspack > 1);
        if (!wholeRem || triRemOnly) jmpi(1 | state.flagAP, lZRemLoopBegin);
        mark(lZRemLoopEnd);

        strategy.S = saveSStrategy;
    }

    // Finalize and store sums.
    if (problem.sum) {
        Label skipSumStore;
        bool simtCF = strategy.fused;

        if (remainderX) {
            cmp((simtCF ? 16 : 1) | le | state.flagAP, state.remainderX, 0);
            simtCF ? goto12(16 | state.flagAP, skipSumStore)
                   : jmpi(1 | state.flagAP, skipSumStore);
        }

        horizontalAdd(!byColumn, Tsum, state.Ds_regs.back(), state.Ds_layout);

        // Accumulate sums from main and remainder loops.
        for (int l = 1; l < int(state.Ds_regs.size()); l++) {
            map(hw, Tsum, state.Ds_regs[0], state.Ds_regs[l], strategy,
                    [&](int ne, GRF r1, GRF r2) { add(ne, r1, r1, r2); });
            state.ra.safeRelease(state.Ds_regs[l]);
        }
        state.Ds_regs.resize(1);

        MatrixAddressing Ds = problem.D;
        Ds.crosspack = 1;

        MatrixAddressingStrategy Ds_strategy = strategy.D;
        Ds_strategy.accessType = AccessType::Block;

        int sr = 1, sc = 1;
        (byColumn ? sr : sc) = problem.D.packSize;

        vector<RegisterBlock> Ds_layoutOut;
        bool ok = getRegLayout(Tsum, Ds_layoutOut, sr, sc, false, false, true,
                          true, 0, 0, Ds, Ds_strategy)
                && matchLayouts(Tsum, Ds_layoutOut, state.Ds_layout);
        if (!ok) return false;

        vector<GRFRange> Ds_addrs;
        allocAddrRegs(Ds_addrs, Ds_layoutOut, Ds, Ds_strategy, state);

        Subregister Ds_base;
        Ds_base = state.ra.alloc_sub(state.effD.getType());

        mulConstant(1, Ds_base.ud(), state.inputs.ldd, problem.D.packSize * Td);
        add(1, Ds_base.ud(), Ds_base.ud(), -problem.D.packSize * Tsum);
        eadd(1, Ds_base, Ds_base.ud(), state.effD, strategy, state);

        setupAddr(Tsum, Ds_addrs, Ds_base, Ds_layoutOut, Subregister(), Ds,
                Ds_strategy, strategy, state);
        storeMatrix(state.Ds_regs[0], Ds_layoutOut, Ds, Ds_strategy, Ds_addrs,
                strategy, state);

        state.ra.safeRelease(Ds_base);
        safeReleaseRanges(Ds_addrs, state);
        safeReleaseRanges(state.Ds_regs, state);
        state.Ds_layout.clear();
        state.ra.safeRelease(state.all1s);

        if (remainderX) {
            mark(skipSumStore);
            if (simtCF) join(16);
        }
    }

    // Done. Free address, data, and flag registers.
    cleanup();
    state.ra.safeRelease(state.signChange);
    if (lateZLoopCheck) state.raVFlag.lock(state.flagAP);
    state.raVFlag.safeRelease(state.flagReflect);
    state.raVFlag.safeRelease(state.flagSwizzle);

    return true; /* Success! */
}

// Register-to-register copy of a single block, ignoring register offsets in the block.
template <HW hw>
bool gemm_kernel_generator_t<hw>::copyRegisterBlock(Type Ts, Type Td,
        const RegisterBlock &blockSrc, const RegisterBlock &blockDst,
        const GRFMultirange &src, const GRFMultirange &dst, int dOffR,
        int dOffC, const CommonStrategy &strategy, CommonState &state,
        bool preserveSrc) {
    std::vector<RegisterBlock> modSrc {1, blockSrc}, modDst {1, blockDst};
    modSrc[0].offsetBytes %= GRF::bytes(hw);
    modDst[0].offsetBytes %= GRF::bytes(hw);
    return copyRegisters(Ts, Td, modSrc, modDst, src, dst, dOffR, dOffC, false,
            strategy, state, preserveSrc);
}

// Register-to-register copy, with no scaling.
template <HW hw>
bool gemm_kernel_generator_t<hw>::copyRegisters(Type Ts, Type Td,
        const vector<RegisterBlock> &layoutSrc,
        const vector<RegisterBlock> &layoutDst, const GRFMultirange &src,
        const GRFMultirange &dst, int dOffR, int dOffC, bool conjugate,
        const CommonStrategy &strategy, CommonState &state, bool preserveSrc) {
    return copyRegisters(Ts, Td, layoutSrc, layoutDst, src, dst, dOffR, dOffC,
            Scalar<double>(1.), Scalar<double>(0.), conjugate, strategy, state,
            preserveSrc);
}

// Register-to-register copy, with scaling.
template <HW hw>
bool gemm_kernel_generator_t<hw>::copyRegisters(Type Ts, Type Td,
        const vector<RegisterBlock> &layoutSrc,
        const vector<RegisterBlock> &layoutDst, const GRFMultirange &src,
        const GRFMultirange &dst, int dOffR, int dOffC,
        const Scalar<double> &alpha_real, const Scalar<double> &alpha_imag,
        bool conjugate, const CommonStrategy &strategy, CommonState &state,
        bool preserveSrc) {
    int nphases = 1;

    bool preswizzle = (hw >= HW::XeHP);
    GRFRange copyTemp;

    auto allocTemp = [&]() {
        if (preswizzle && copyTemp.isInvalid())
            copyTemp = state.ra.alloc_range(2);
    };

    int srcM, srcN;
    getLayoutDims(layoutSrc, srcM, srcN);
    bool vectorCopy = (srcM == 1 || srcN == 1);

    for (int phase = -1; phase < nphases; phase++) {
        for (auto &sblock : layoutSrc) {
            auto RegisterBlock::*nx
                    = sblock.colMajor ? &RegisterBlock::nr : &RegisterBlock::nc;
            auto RegisterBlock::*ny
                    = sblock.colMajor ? &RegisterBlock::nc : &RegisterBlock::nr;

            for (int eoffY = 0; eoffY < sblock.*ny; eoffY++) {
                for (int eoffX = 0; eoffX < sblock.*nx;) {
                    auto eoffR = sblock.colMajor ? eoffX : eoffY;
                    auto eoffC = sblock.colMajor ? eoffY : eoffX;

                    int selems, delems;
                    const RegisterBlock *sblockPtr, *dblockPtr;

                    // Locate source and destination register.
                    auto sreg = findBlockReg(Ts, layoutSrc,
                            sblock.offsetR + eoffR, sblock.offsetC + eoffC, src,
                            selems, sblockPtr);
                    auto dreg = findBlockReg(Td, layoutDst,
                            sblock.offsetR + eoffR + dOffR,
                            sblock.offsetC + eoffC + dOffC, dst, delems,
                            dblockPtr);

                    auto scrosspack = sblock.crosspack;
                    auto dcrosspack = dblockPtr->crosspack;

                    if (sblock.colMajor != dblockPtr->colMajor) {
                        bool sLargeCP = isLargeCrosspack(Ts, scrosspack);
                        bool dLargeCP = isLargeCrosspack(Td, dcrosspack);
                        bool sEffCM = sblock.colMajor ^ sLargeCP;
                        bool dEffCM = dblockPtr->colMajor ^ dLargeCP;
                        if (sEffCM == dEffCM) {
                            if (sLargeCP)
                                selems = std::min<int>(selems, scrosspack);
                            if (dLargeCP)
                                delems = std::min<int>(delems, dcrosspack);
                        } else {
                            if (!vectorCopy)
                                stub(); // No in-register matrix transposes.
                            selems = delems = 1;
                        }
                    }

                    // Find out how many consecutive elements we can copy.
                    auto nGRFs = (strategy.dualGRF ? 2 : 1);
                    auto nGRFs_d = (dreg.getOffset() >= dcrosspack)
                            ? 1
                            : nGRFs; // Don't cross destination GRF boundaries for efficiency.
                    auto selems_real = selems * Ts.components();
                    auto delems_real = delems * Td.components();
                    auto selems_limit
                            = div_up(nGRFs * elementsPerGRF(hw, Ts.real())
                                            - sreg.getOffset(),
                                    scrosspack);
                    auto delems_limit
                            = div_up(nGRFs_d * elementsPerGRF(hw, Td.real())
                                            - dreg.getOffset(),
                                    dcrosspack);
                    selems_real = std::min({selems_real, selems_limit});
                    delems_real = std::min({delems_real, delems_limit});
                    auto nelems_real = std::min(selems_real, delems_real);
                    nelems_real = rounddown_pow2(nelems_real);

                    if (Ts == Type::f32 && Td != Type::f32 && dcrosspack == 1)
                        nelems_real = std::min(nelems_real,
                                elementsPerGRF(hw,
                                        Ts)); // Special case: mixed mode packed downconversion limited to SIMD8.

                    // Check if separate conversions are needed due to size changes.
                    auto sconvertCP = (Ts.size() / Td.size());
                    bool sconvert = (Td.size() == 1 && Ts.size() > 1
                            && dcrosspack != sconvertCP);
                    if (sconvert && preserveSrc) stub();
                    auto sregConverted = sconvert
                            ? sreg.reinterpret(0, Td.real().ngen())(sconvertCP)
                            : sreg(scrosspack);

                    auto dconvertCP = (Td.size() / Ts.size());
                    bool dconvert = (Ts.size() == 1 && Td.size() > 1
                            && scrosspack != dconvertCP);
                    auto dregConverted = dconvert
                            ? dreg.reinterpret(0, Ts.real().ngen())(dconvertCP)
                            : dreg(dcrosspack);

                    InstructionModifier modMov, mmodMov;
                    if (Ts != Td && Td.isInteger() && Td.size() <= Ts.size()) {
                        modMov = modMov | sat;
                        if (!sconvert && !dconvert) mmodMov = mmodMov | sat;
                    }

                    // Finally, copy, with any necessary conjugation and scaling. If doing a raw copy, use another pipe.
                    switch (phase) {
                        case -1:
                            if (hw == HW::Gen9 && Ts == Type::f32
                                    && !Td.isFP()) {
                                // Gen9: round to nearest before downconvert (not done by mov).
                                rnde(nelems_real, sreg(scrosspack),
                                        sreg(scrosspack));
                            }
                            if (sconvert)
                                mov(nelems_real | modMov, sregConverted,
                                        sreg(scrosspack));
                            break;
                        case 0:
                            if (alpha_real == 1 || alpha_real == -1) {
                                if (Ts.real() == Td.real()) {
                                    movePipes(sreg, scrosspack == 1);
                                    movePipes(dreg, scrosspack == 1);
                                    if (!sconvert)
                                        sregConverted = sreg(scrosspack);
                                    if (!dconvert)
                                        dregConverted = dreg(dcrosspack);
                                }
                                int telems = nelems_real * Ts.real()
                                        / sreg.getBytes();
                                if (alpha_real == -1) {
                                    auto wd = elementsPerGRF(
                                            hw, sreg.getType());
                                    auto base = state.signChange.sub(
                                            0, dreg.getType());
                                    xor_(telems, dreg(1), sreg(1),
                                            (wd >= telems) ? base(1)
                                                           : base(0, wd, 1));
                                } else
                                    emov(telems | mmodMov, dregConverted,
                                            sregConverted, strategy, state);
                            } else {
                                auto realDst = dreg(dcrosspack);
                                auto effDst = realDst;
                                if (preswizzle && (Ts.isFP() || Td.isFP())) {
                                    allocTemp();
                                    if ((sreg.getOffset() != dreg.getOffset())
                                            || (scrosspack != dcrosspack))
                                        effDst = copyTemp[0].sub(
                                                sreg.getOffset(),
                                                sreg.getType())(scrosspack);
                                }

                                if (alpha_real.fixed())
                                    mul(nelems_real, effDst, sregConverted,
                                            cast(Ts.real(), alpha_real));
                                else
                                    mul(nelems_real, effDst, sregConverted,
                                            alpha_real.getRegAvoiding(
                                                    hw, sreg));

                                if (effDst != realDst) {
                                    moveToIntPipe(nelems_real, realDst);
                                    moveToIntPipe(nelems_real, effDst);
                                    int nelems_real_int = nelems_real * Td
                                            / getBytes(effDst.getType());
                                    emov(nelems_real_int, realDst, effDst,
                                            strategy, state);
                                    dconvert = false;
                                }
                            }
                            break;
                        case 1:
                            if (dconvert)
                                mov(nelems_real | modMov, dreg(dcrosspack),
                                        dregConverted);
                            break;
                    }

                    eoffX += nelems_real / Ts.components();
                }
            }
        }
    }

    state.ra.safeRelease(copyTemp);
    return true; // Success
}

// Get driver information from this strategy.
template <HW hw>
CommonDriverInfo gemm_kernel_generator_t<hw>::driverInfo(
        const CopyProblem &problem, const CopyStrategy &strategy) {
    CommonDriverInfo info;
    bool isA = (problem.D.layout == MatrixLayout::Pc);

    for (int d = 0; d < 3; d++) {
        info.blocking[d] = info.blockingAlt[d] = info.unroll[d] = 0;
        info.wg[d] = 1;
        info.loopOrder[d] = LoopNone;
    }

    info.subgroupSize = strategy.subgroupSize;
    info.grfCount = strategy.GRFs;
    info.unroll[0] = isA ? strategy.unrollX : strategy.unrollY;
    info.unroll[1] = isA ? strategy.unrollY : strategy.unrollX;
    info.kRemainderHandling
            = (strategy.remHandlingY != RemainderHandling::Ignore);
    info.loopOrder[0] = (isA ^ strategy.xLoop) ? LoopM : LoopN;
    if (strategy.zParallel)
        info.loopOrder[1] = (isA ^ strategy.xLoop) ? LoopN : LoopM;
    info.fusedLoop = strategy.fused ? info.loopOrder[0] : LoopNone;
    info.wg[0] = 16;
    info.wgExpand = 1;
    info.wgUpdate = WGDynamic;
    info.kRemainderHandling = true;
    info.kParallel = strategy.zParallel;
    info.kParallelLocal = false;
    info.slm = info.perKSLM = 0;
    info.alignment[0] = problem.S.alignment;
    info.alignment[1] = problem.D.alignment;
    info.alignment[2] = 0;
    info.support4GB[0] = (strategy.S.base.getModel() == ModelA64);
    info.support4GB[1] = (strategy.D.base.getModel() == ModelA64);
    info.support4GB[2] = false;

    return info;
}

// Validate a copy strategy, correcting settings as necessary.
void CopyStrategy::preflight(HW hw, const CopyProblem &problem) {
    bool cm = isColMajor(problem.D.layout);

    S.preflight(hw);
    D.preflight(hw);

    s_load = std::max(s_load, 1);
    d_load = std::max(d_load, 1);
    if (s_load_masked == 0) s_load_masked = s_load;
    if (d_load_masked == 0) d_load_masked = d_load;
    unrollX = std::max(unrollX, 1);
    unrollY = std::max(unrollY, 1);
    unrollY = align_up(unrollY, problem.D.crosspack);

    // Ensure d_load is a multiple of s_load and crosspack, and unrollZ a multiple of both.
    // For x loop kernels, ensure s_load is a multiple of the packing size.
    // For y loop kernels, ensure all d_loads are multiples of y tile size if any.
    if (xLoop) {
        s_load = align_up(s_load, problem.D.packSize);
        s_load_masked = align_up(s_load_masked, problem.D.packSize);
    } else {
        auto D_tileY = cm ? problem.D.tileC : problem.D.tileR;
        if (D_tileY > 0) d_load_masked = align_up(d_load_masked, D_tileY);
        d_load_masked = align_up(d_load_masked, problem.D.crosspack);
    }
    d_load = align_up(d_load, s_load);
    d_load_masked = align_up(d_load_masked, s_load_masked);
    d_load = align_up(d_load, d_load_masked);

    if (xLoop)
        unrollX = align_up(unrollX, d_load);
    else
        unrollY = align_up(unrollY, d_load);

    if (unrollY == 1 && remHandlingY == RemainderHandling::Split)
        remHandlingY = RemainderHandling::General;

    spf &= !problem.trsm; // TRSM copies use SIMT control flow.

    CommonStrategy::preflight(hw, problem);
}

/**********************************************************************/
/*                      Common Kernel Functions                       */
/**********************************************************************/

// Generate the kernel prologue.
template <HW hw>
void gemm_kernel_generator_t<hw>::prologue(const CommonStrategy &strategy) {
    uint16_t cr0Enable;

    interface.generatePrologue(*this);

    cr0Enable = 0x1000; // IEEE float->int rounding.
    if (strategy.ieeeDenormals) cr0Enable |= 0x4C0; // Enable hf|f|df denormals.
    if (strategy.spf) cr0Enable |= 0x4; // Enable single program flow.

    or_(1, cr0, cr0, cr0Enable);

    InstructionModifier imod = 1;
    if (hw < HW::Gen12LP) imod |= Switch;

    if (interface.getSIMD() < 16) mov(imod, sr0[2], uint16_t(0xFFFF));
}

// Generate the kernel epilogue.
template <HW hw>
void gemm_kernel_generator_t<hw>::epilogue(
        const CommonStrategy &strategy, const CommonState &state) {
    auto r0_info = state.r0_info;

    if (r0_info.getBase() < 112) {
        mov<uint32_t>(8, r127, r0_info);
        r0_info = r127;
    }

    if (strategy.finalFence) {
        memfence(r124, r0_info);
        mov<uint32_t>(8, null, r124);
    }

    threadend(r0_info);
}

// Pad the end of the kernel to accommodate instruction prefetching.
template <HW hw>
void gemm_kernel_generator_t<hw>::padding() {
    for (int q = 0; q < 8; q++)
        nop();
}

// Common state initialization code.
template <HW hw>
void gemm_kernel_generator_t<hw>::initState(const CommonProblem &problem,
        const CommonStrategy &strategy, CommonState &state) {
    interface.requireLocalID(3);
    interface.requireLocalSize();
    if (problem.nonuniformWGs) interface.requireNonuniformWGs();

    if (strategy.wgInSS) interface.requireBarrier();

    interface.requireSIMD(strategy.subgroupSize);

    if (!strategy.sipR0WA) interface.requireNoPreemption();

    interface.requireGRF(strategy.GRFs);
    state.ra.setRegisterCount(strategy.GRFs);

    if (problem.gtpinSupport) interface.requireScratch(128);

    for (int i = 0; i < FlagRegister::subcount(hw); i++)
        state.activeVFlags[i].clear();
}

CommonStrategy::CommonStrategy(HW hw, int stepping) : emulate(hw, stepping) {
    fused = one_of(hw, HW::Gen12LP, HW::XeHP, HW::XeHPG);
}

void CommonStrategy::preflight(HW hw, const CommonProblem &problem) {
    subgroupSize = std::max(subgroupSize, GRF::bytes(hw) >> 2);
    sipR0WA &= (hw == HW::Gen9);
    if (sipR0WA && (moveR0 == MoveR0::None)) moveR0 = MoveR0::GRF;
    readSuppressionWA &= fused;

    bool emulateNeedsAcc = emulate.emulate64 || emulate.emulateDWxDW
            || emulate.emulate64_mul;
    if (moveR0 == MoveR0::Acc && emulateNeedsAcc) moveR0 = MoveR0::None;

    spf &= !fused;
}

template <HW hw>
constexpr typename gemm_kernel_generator_t<hw>::status_stream::Endl
        gemm_kernel_generator_t<hw>::status_stream::endl;

template class gemm_kernel_generator_t<HW::Gen9>;
template class gemm_kernel_generator_t<HW::Gen12LP>;
template class gemm_kernel_generator_t<HW::XeHP>;
template class gemm_kernel_generator_t<HW::XeHPG>;
template class gemm_kernel_generator_t<HW::XeHPC>;

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
