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

#ifndef NGEN_REGISTER_ALLOCATOR_HPP
#define NGEN_REGISTER_ALLOCATOR_HPP

#ifdef ENABLE_LLVM_WCONVERSION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#endif

#include "ngen.hpp"
#include <cstdint>
#include <stdexcept>

#ifdef NGEN_ENABLE_RA_DUMP
#include <iostream>
#include <iomanip>
#endif

namespace NGEN_NAMESPACE {

// Gen registers are organized in banks of bundles.
// Each bundle is modeled as groups of contiguous registers separated by a stride.
struct Bundle {
    static const int8_t any = -1;

    int8_t bundle_id;
    int8_t bank_id;

    Bundle() : bundle_id(any), bank_id(any) {}
    Bundle(int8_t bank_id_, int8_t bundle_id_) : bundle_id(bundle_id_), bank_id(bank_id_) {}

    // Number of bundles in each bank (per thread).
    static constexpr14 int bundleCount(HW hw) {
        if (hw >= HW::Xe2) return 8;
        if (hw >= HW::XeHP) return 16;
        if (hw == HW::Gen12LP) return 8;
        return 2;
    }

    // Number of banks.
    static constexpr int bankCount(HW hw)      { return 2; }

    static inline Bundle locate(HW hw, RegData reg);

    inline int firstReg(HW hw) const;                   // The first register in the bundle.
    inline int groupSize(HW hw) const;                  // Number of registers in each contiguous group of the bundle.
    inline int stride(HW hw) const;                     // Stride between register groups of the bundle.

    inline uint64_t regMask(HW hw, int offset) const;   // Get register mask for this bundle, for registers [64*offset, 64*(offset+1)).

    friend constexpr bool operator==(const Bundle &b1, const Bundle &b2) {
        return b1.bundle_id == b2.bundle_id && b1.bank_id == b2.bank_id;
    }

    static bool conflicts(HW hw, RegData r1, RegData r2) {
        return !r1.isNull() && !r2.isNull() && (locate(hw, r1) == locate(hw, r2));
    }

    static bool sameBank(HW hw, RegData r1, RegData r2) {
        return !r1.isNull() && !r2.isNull() && (locate(hw, r1).bank_id == locate(hw, r2).bank_id);
    }

    // Deprecated snake case APIs.
    static constexpr14 int bundle_count(HW hw)              { return bundleCount(hw); }
    static constexpr   int bank_count(HW hw)                { return bankCount(hw); }
    int first_reg(HW hw) const                              { return firstReg(hw); }
    int group_size(HW hw) const                             { return groupSize(hw); }
    uint64_t reg_mask(HW hw, int offset) const              { return regMask(hw, offset); }
    static bool same_bank(HW hw, RegData r1, RegData r2)    { return sameBank(hw, r1, r2); }
};

// A group of register bundles.
struct BundleGroup {
    explicit BundleGroup(HW hw_) : hw(hw_) {}

    static BundleGroup AllBundles() {
        BundleGroup bg{HW::Gen9};
        for (auto &m: bg.reg_masks)
            m = ~uint64_t(0);
        return bg;
    }

    friend BundleGroup operator|(BundleGroup lhs, Bundle rhs) { lhs |= rhs; return lhs; }
    BundleGroup &operator|=(Bundle rhs) {
        for (size_t rchunk = 0; rchunk < reg_masks.size(); rchunk++)
            reg_masks[rchunk] |= rhs.reg_mask(hw, int(rchunk));
        return *this;
    }

    BundleGroup operator~() {
        auto result = *this;
        for (auto &m: result.reg_masks)
            m = ~m;
        return result;
    }

    uint64_t regMask(int rchunk) const {
        auto i = size_t(rchunk);
        return (i < reg_masks.size()) ? reg_masks[i] : 0;
    }

    // Deprecated snake-case API.
    uint64_t reg_mask(int rchunk) const { return regMask(rchunk); }

private:
    HW hw;
    std::array<uint64_t, GRF::maxRegs() / 64> reg_masks{};
};

// Gen register allocator.
class RegisterAllocator {
public:
    explicit RegisterAllocator(HW hw_) : hw(hw_) { init(); }

    HW hardware() const { return hw; }

    // Allocation functions: sub-GRFs, full GRFs, and GRF ranges.
    inline GRFRange allocRange(int nregs, Bundle baseBundle = Bundle(),
                               BundleGroup bundleMask = BundleGroup::AllBundles());
    GRF alloc(Bundle bundle = Bundle()) { return allocRange(1, bundle)[0]; }

    inline Subregister allocSub(DataType type, Bundle bundle = Bundle());
    template <typename T>
    Subregister allocSub(Bundle bundle = Bundle()) { return allocSub(getDataType<T>(), bundle); }

    // Allocate flag registers.
    //   sub = true (default):  a 16-bit subregister   (fX.Y:uw)
    //   sub = false:           a full 32-bit register (fX.0:ud)
    inline FlagRegister allocFlag(bool sub = true);

    // Attempted allocation. Return value is invalid if allocation failed.
    inline GRFRange tryAllocRange(int nregs, Bundle baseBundle = Bundle(),
                                  BundleGroup bundleMask = BundleGroup::AllBundles());
    inline GRF tryAlloc(Bundle bundle = Bundle());

    inline Subregister tryAllocSub(DataType type, Bundle bundle = Bundle());
    template <typename T>
    Subregister tryAllocSub(Bundle bundle = Bundle()) { return tryAllocSub(getDataType<T>(), bundle); }

    inline FlagRegister tryAllocFlag(bool sub = true);

    // Release a previous allocation or claim.
    inline void release(GRF reg);
    inline void release(GRFRange range);
    inline void release(Subregister subreg);
    inline void release(FlagRegister flag);

    template <typename RD>
    void safeRelease(RD &reg) { release(reg); reg.invalidate(); }

    // Claim specific registers.
    inline void claim(GRF reg);
    inline void claim(GRFRange range);
    inline void claim(Subregister subreg);
    inline void claim(FlagRegister flag);

    // Set register count.
    inline void setRegisterCount(int rcount);
    inline int getRegisterCount() const { return regCount; }
    inline int countAllocedRegisters() const;

    // Check availability.
    inline bool isFree(GRF reg) const;
    inline bool isFree(GRFRange range) const;
    inline bool isFree(Subregister subreg) const;

#ifdef NGEN_ENABLE_RA_DUMP
    inline void dump(std::ostream &str);
#endif

    // Deprecated snake case APIs.
    GRFRange alloc_range(int nregs, Bundle base_bundle = Bundle(),
                         BundleGroup bundle_mask = BundleGroup::AllBundles()) { return allocRange(nregs, base_bundle, bundle_mask); }
    Subregister alloc_sub(DataType type, Bundle bundle = Bundle())            { return allocSub(type, bundle); }
    template <typename T> Subregister alloc_sub(Bundle bundle = Bundle())     { return allocSub(getDataType<T>(), bundle); }
    FlagRegister alloc_flag(bool sub = true)                                  { return allocFlag(sub); }

    GRFRange try_alloc_range(int nregs, Bundle base_bundle = Bundle(), BundleGroup bundle_mask = BundleGroup::AllBundles()) {
        return tryAllocRange(nregs, base_bundle, bundle_mask);
    }
    GRF try_alloc(Bundle bundle = Bundle())                                   { return tryAlloc(bundle); }
    Subregister try_alloc_sub(DataType type, Bundle bundle = Bundle())        { return tryAllocSub(type, bundle); }
    template <typename T>
    Subregister try_alloc_sub(Bundle bundle = Bundle())                       { return tryAllocSub(getDataType<T>(), bundle); }
    FlagRegister try_alloc_flag(bool sub = true)                              { return tryAllocFlag(sub); }

protected:
    using mtype = uint16_t;

    HW hw;                                      // HW generation.
    uint8_t freeWhole[GRF::maxRegs() / 8];      // Bitmap of free whole GRFs.
    mtype freeSub[GRF::maxRegs()];              // Bitmap of free partial GRFs, at dword granularity.
    uint16_t regCount;                          // # of registers.
    uint8_t freeFlag;                           // Bitmap of free flag registers.
    mtype fullSubMask;

    inline void init();
    inline void claimSub(int r, int o, int dw);
};


// Exceptions.
class out_of_registers_exception : public std::runtime_error {
public:
    out_of_registers_exception() : std::runtime_error("Insufficient registers in requested bundle") {}
};



// -----------------------------------------
//  High-level register allocator functions.
// -----------------------------------------

int Bundle::firstReg(HW hw) const
{
    int bundle0 = (bundle_id == any) ? 0 : bundle_id;
    int bank0 = (bank_id == any) ? 0 : bank_id;

    switch (hw) {
    case HW::Gen9:
    case HW::Gen10:
        return (bundle0 << 8) | bank0;
    case HW::Gen11:
        return (bundle0 << 8) | (bank0 << 1);
    case HW::Gen12LP:
    case HW::XeHPC:
    case HW::Xe2:
    case HW::Xe3:
        return (bundle0 << 1) | bank0;
    case HW::XeHP:
    case HW::XeHPG:
        return (bundle0 << 2) | (bank0 << 1);
    default:
        return 0;
    }
}

int Bundle::groupSize(HW hw) const
{
    if (bundle_id == any && bank_id == any)
        return 128;
    else switch (hw) {
    case HW::Gen11:
    case HW::XeHP:
    case HW::XeHPG:
        return 2;
    default:
        return 1;
    }
}

int Bundle::stride(HW hw) const
{
    if (bundle_id == any && bank_id == any)
        return 128;
    else switch (hw) {
    case HW::Gen9:
    case HW::Gen10:
        return 2;
    case HW::Gen11:
        return 4;
    case HW::Gen12LP:
    case HW::Xe2:
    case HW::Xe3:
        return 16;
    case HW::XeHP:
    case HW::XeHPG:
        return 64;
    case HW::XeHPC:
        return 32;
    default:
        return 128;
    }
}

uint64_t Bundle::regMask(HW hw, int offset) const
{
    uint64_t bundle_mask = -1, bank_mask = -1, base_mask = -1;
    int bundle0 = (bundle_id == any) ? 0 : bundle_id;
    int bank0   = (bank_id == any)   ? 0 : bank_id;

    switch (hw) {
    case HW::Gen9:
    case HW::Gen10:
        if (bundle_id != any && bundle_id != offset)    bundle_mask = 0;
        if (bank_id != any)                             bank_mask = 0x5555555555555555 << bank_id;
        return bundle_mask & bank_mask;
    case HW::Gen11:
        if (bundle_id != any && bundle_id != offset)    bundle_mask = 0;
        if (bank_id != any)                             bank_mask = 0x3333333333333333 << (bank_id << 1);
        return bundle_mask & bank_mask;
    case HW::Gen12LP:
    case HW::Xe2:
    case HW::Xe3:
        if (bundle_id != any)                           base_mask  = 0x0003000300030003;
        if (bank_id != any)                             base_mask &= 0x5555555555555555;
        return base_mask << (bank0 + (bundle0 << 1));
    case HW::XeHP:
    case HW::XeHPG:
        if (bundle_id != any)                           base_mask  = 0x000000000000000F;
        if (bank_id != any)                             base_mask &= 0x3333333333333333;
        return base_mask << ((bank0 << 1) + (bundle0 << 2));
    case HW::XeHPC:
        if (bundle_id != any)                           base_mask  = 0x0000000300000003;
        if (bank_id != any)                             base_mask &= 0x5555555555555555;
        return base_mask << (bank0 + (bundle0 << 1));
    default:
        return -1;
    }
}

Bundle Bundle::locate(HW hw, RegData reg)
{
    int base = reg.getBase();

    switch (hw) {
        case HW::Gen9:
        case HW::Gen10:
            return Bundle(base & 1, base >> 6);
        case HW::Gen11:
            return Bundle((base >> 1) & 1, base >> 6);
        case HW::Gen12LP:
        case HW::Xe2:
        case HW::Xe3:
            return Bundle(base & 1, (base >> 1) & 7);
        case HW::XeHP:
        case HW::XeHPG:
            return Bundle((base >> 1) & 1, (base >> 2) & 0xF);
        case HW::XeHPC:
            return Bundle(base & 1, (base >> 1) & 0xF);
        default:
            return Bundle();
    }
}

// -----------------------------------------
//  Low-level register allocator functions.
// -----------------------------------------

void RegisterAllocator::init()
{
    constexpr int maxRegs = GRF::maxRegs();

    fullSubMask = (1u << (GRF::bytes(hw) >> 2)) - 1;
    for (int r = 0; r < maxRegs; r++)
        freeSub[r] = fullSubMask;
    for (int r_whole = 0; r_whole < (maxRegs >> 3); r_whole++)
        freeWhole[r_whole] = 0xFF;

    freeFlag = (1u << FlagRegister::subcount(hw)) - 1;
    regCount = maxRegs;

    if (hw < HW::XeHP)
        setRegisterCount(128);
}

void RegisterAllocator::claim(GRF reg)
{
    int r = reg.getBase();

    freeSub[r] = 0x00;
    freeWhole[r >> 3] &= ~(1 << (r & 7));
}

void RegisterAllocator::claim(GRFRange range)
{
    for (int i = 0; i < range.getLen(); i++)
        claim(range[i]);
}

void RegisterAllocator::claim(Subregister subreg)
{
    int r = subreg.getBase();
    int dw = subreg.getDwords();
    int o = (subreg.getByteOffset()) >> 2;

    claimSub(r, o, dw);
}

void RegisterAllocator::claimSub(int r, int o, int dw)
{
    freeSub[r]        &= ~((1 << (o + dw)) - (1 << o));
    freeWhole[r >> 3] &= ~(1 << (r & 7));
}

void RegisterAllocator::claim(FlagRegister flag)
{
    freeFlag &= ~(1 << flag.index());
    if (flag.getType() == DataType::ud)
        freeFlag &= ~(1 << (flag.index() + 1));
}

void RegisterAllocator::setRegisterCount(int rcount)
{
    constexpr int maxRegs = GRF::maxRegs();

    if (rcount < regCount) {
        for (int r = rcount; r < maxRegs; r++)
            freeSub[r] = 0x00;
        for (int rr = (rcount + 7) >> 3; rr < (maxRegs >> 3); rr++)
            freeWhole[rr] = 0x00;
        if ((rcount & 7) && (rcount < maxRegs))
            freeWhole[rcount >> 3] &= ~((1 << (rcount & 7)) - 1);
    } else if (rcount > regCount) {
        for (int r = regCount; r < std::min(rcount, maxRegs); r++)
            release(GRF(r));
    }
    regCount = rcount;
}

inline int RegisterAllocator::countAllocedRegisters() const {
   int register_count = 0;
   int group_size = 8 * sizeof(this->freeWhole[0]);
   int register_groups = this->regCount / group_size;
   for (int group = 0; group < register_groups; group++) {
       for (int subgroup = 0; subgroup < group_size; subgroup++) {
           if ((this->freeWhole[group] & (1 << subgroup)) == 0)
               register_count++;
       }
   }
   return register_count;
}

void RegisterAllocator::release(GRF reg)
{
    if (reg.isInvalid()) return;
    int r = reg.getBase();

    freeSub[r] = fullSubMask;
    freeWhole[r >> 3] |= (1 << (r & 7));
}

void RegisterAllocator::release(GRFRange range)
{
    if (range.isInvalid()) return;
    for (int i = 0; i < range.getLen(); i++)
        release(range[i]);
}

void RegisterAllocator::release(Subregister subreg)
{
    if (subreg.isInvalid()) return;
    int r = subreg.getBase();
    int dw = subreg.getDwords();
    int o = (subreg.getByteOffset()) >> 2;

    freeSub[r] |= (1 << (o + dw)) - (1 << o);
    if (freeSub[r] == fullSubMask)
        freeWhole[r >> 3] |= (1 << (r & 7));
}

void RegisterAllocator::release(FlagRegister flag)
{
    if (flag.isInvalid()) return;
    freeFlag |= (1 << flag.index());
    if (flag.getType() == DataType::ud)
        freeFlag |= (1 << (flag.index() + 1));
}

bool RegisterAllocator::isFree(GRF reg) const
{
    if (reg.isInvalid()) return true;
    return freeSub[reg.getBase()] == fullSubMask;
}

bool RegisterAllocator::isFree(GRFRange range) const
{
    if (range.isInvalid()) return true;
    for (int i = 0; i < range.getLen(); i++)
        if (!isFree(range[i]))
            return false;
    return true;
}

bool RegisterAllocator::isFree(Subregister subreg) const
{
    if (subreg.isInvalid()) return true;
    int r = subreg.getBase();
    int dw = subreg.getDwords();
    int o = (subreg.getByteOffset()) >> 2;
    auto m = (1 << (o + dw)) - (1 << o);
    return (~freeSub[r] & m) == 0;
}

// -------------------------------------------
//  High-level register allocation functions.
// -------------------------------------------

GRFRange RegisterAllocator::allocRange(int nregs, Bundle baseBundle, BundleGroup bundleMask)
{
    auto result = tryAllocRange(nregs, baseBundle, bundleMask);
    if (result.isInvalid())
        throw out_of_registers_exception();
    return result;
}

Subregister RegisterAllocator::allocSub(DataType type, Bundle bundle)
{
    auto result = tryAllocSub(type, bundle);
    if (result.isInvalid())
        throw out_of_registers_exception();
    return result;
}

FlagRegister RegisterAllocator::allocFlag(bool sub)
{
    auto result = tryAllocFlag(sub);
    if (result.isInvalid())
        throw out_of_registers_exception();
    return result;
}

GRFRange RegisterAllocator::tryAllocRange(int nregs, Bundle baseBundle, BundleGroup bundleMask)
{
    if (nregs == 0) return GRFRange(0, 0);

    uint64_t freeWhole64[sizeof(freeWhole) / sizeof(uint64_t)];
    std::memcpy(freeWhole64, freeWhole, sizeof(freeWhole));
    bool ok = false;
    int r_base = -1;

    for (int rchunk = 0; rchunk < (GRF::maxRegs() >> 6); rchunk++) {
        uint64_t free = freeWhole64[rchunk] & bundleMask.regMask(rchunk);
        uint64_t free_base = free & baseBundle.regMask(hw, rchunk);

        while (free_base) {
            // Find the first free base register.
            int first_bit = utils::bsf(free_base);
            r_base = first_bit + (rchunk << 6);

            // Check if required # of registers are available.
            int last_bit = first_bit + nregs;
            if (last_bit <= 64) {
                // Range to check doesn't cross 64-GRF boundary. Fast check using bitmasks.
                uint64_t mask = ((uint64_t(1) << (last_bit - 1)) << 1) - (uint64_t(1) << first_bit);
                ok = !(mask & ~free);
            } else {
                // Range to check crosses 64-GRF boundary. Check first part using bitmasks,
                // Check the rest using a loop (ho hum).
                uint64_t mask = ~uint64_t(0) << first_bit;
                ok = !(mask & ~free) && (r_base + nregs <= (int)(sizeof(freeSub) / sizeof(freeSub[0])));
                if (ok) for (int rr = 64 - first_bit; rr < nregs; rr++) {
                    if (freeSub[r_base + rr] != fullSubMask) {
                        ok = false;
                        break;
                    }
                }
            }

            if (ok) {
                // Create and claim GRF range.
                GRFRange result(r_base, nregs);
                claim(result);

                return result;
            }

            // Not enough consecutive registers. Save time when looking for next base
            //  register by clearing the entire range of registers we just considered.
            uint64_t clear_mask = free + (uint64_t(1) << first_bit);
            free &= clear_mask;
            free_base &= clear_mask;
        }
    }

    return GRFRange();
}

GRF RegisterAllocator::tryAlloc(Bundle bundle)
{
    auto range = tryAllocRange(1, bundle);
    return range.isInvalid() ? GRF() : range[0];
}

Subregister RegisterAllocator::tryAllocSub(DataType type, Bundle bundle)
{
    int dwords = getDwords(type);
    int r_alloc = 0, o_alloc = 0;

    auto find_alloc_sub = [&,bundle,dwords](bool search_full_grf) -> bool {
        static const uint16_t alloc_patterns[4] = {0b1111111111111111, 0b0101010101010101, 0, 0b0001000100010001};
        auto alloc_pattern = alloc_patterns[(dwords - 1) & 3];
        uint64_t freeWhole64[sizeof(freeWhole) / sizeof(uint64_t)];
        std::memcpy(freeWhole64, freeWhole, sizeof(freeWhole));

        for (int rchunk = 0; rchunk < (GRF::maxRegs() >> 6); rchunk++) {
            uint64_t free = search_full_grf ? freeWhole64[rchunk] : -1;
            free &= bundle.reg_mask(hw, rchunk);

            while (free) {
                int rr = utils::bsf(free);
                int r = rr + (rchunk << 6);
                free &= ~(uint64_t(1) << rr);

                if (search_full_grf || freeSub[r] != fullSubMask) {
                    int subfree = freeSub[r];
                    for (int dw = 1; dw < dwords; dw++)
                        subfree &= (subfree >> dw);
                    subfree &= alloc_pattern;

                    if (subfree) {
                        r_alloc = r;
                        o_alloc = utils::bsf(subfree);
                        return true;
                    }
                }
            }
        }

        return false;
    };

    // First try to find room in a partially allocated register; fall back to
    //  completely empty registers if unsuccessful.
    bool success = find_alloc_sub(false)
                || find_alloc_sub(true);

    if (!success)
        return Subregister();

    claimSub(r_alloc, o_alloc, dwords);

    return Subregister(GRF(r_alloc), (o_alloc << 2) / getBytes(type), type);
}

FlagRegister RegisterAllocator::tryAllocFlag(bool sub)
{
    if (!freeFlag) return FlagRegister();

    if (sub) {
        int idx = utils::bsf(freeFlag);
        freeFlag &= (freeFlag - 1);               // clear lowest bit.

        return FlagRegister::createFromIndex(idx);
    }
    for (int r = 0; r < FlagRegister::count(hw); r++) {
        uint8_t mask = (0b11 << 2 * r);
        if ((freeFlag & mask) == mask) {
            freeFlag &= ~mask;
            return FlagRegister(r);
        }
    }
    return FlagRegister();
}

#ifdef NGEN_ENABLE_RA_DUMP
void RegisterAllocator::dump(std::ostream &str)
{
    str << "\n// Flag registers: ";
    for (int r = 0; r < FlagRegister::subcount(hw); r++)
        str << char((freeFlag & (1 << r)) ? '.' : 'x');

    for (int r = 0; r < regCount; r++) {
        if (!(r & 0x1F)) {
            str << "\n//\n// " << std::left;
            str << 'r' << std::setw(3) << r;
            str << " - r" << std::setw(3) << r+0x1F;
            str << "  ";
        }
        if (!(r & 0xF))  str << ' ';
        if (!(r & 0x3))  str << ' ';

        if (freeSub[r] == 0x00)             str << 'x';
        else if (freeSub[r] == fullSubMask) str << '.';
        else                                 str << '/';
    }

    str << "\n//\n";

    for (int r = 0; r < GRF::maxRegs(); r++) {
        int rr = r >> 3, rb = 1 << (r & 7);
        if ((freeSub[r] == fullSubMask) != bool(freeWhole[rr] & rb))
            str << "// Inconsistent bitmaps at r" << r << std::endl;
        if (freeSub[r] != 0x00 && freeSub[r] != fullSubMask) {
            str << "//  r" << std::setw(3) << r << "   ";
            for (int s = 0; s < (GRF::bytes(hw) >> 2); s++)
                str << char((freeSub[r] & (1 << s)) ? '.' : 'x');
            str << std::endl;
        }
    }

    str << std::endl;
}
#endif /* NGEN_ENABLE_RA_DUMP */

} /* namespace NGEN_NAMESPACE */

#ifdef ENABLE_LLVM_WCONVERSION
#pragma clang diagnostic pop
#endif

#endif /* include guard */
