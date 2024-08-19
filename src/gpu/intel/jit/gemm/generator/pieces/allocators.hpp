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

#ifndef GEMMSTONE_GUARD_ALLOCATORS_HPP
#define GEMMSTONE_GUARD_ALLOCATORS_HPP

#include <cstdint>

#include "internal/ngen_includes.hpp"

#include "internal/namespace_start.hxx"

// Allocator for SWSB tokens.
class TokenAllocator {
public:
    TokenAllocator(ngen::HW hw, int grfCount = 128);

    int8_t tryAlloc();
    void release(int8_t token)                      { free |= (1u << token); }
    void safeRelease(int8_t &token)                 { if (token >= 0) release(token); token = -1; }

protected:
    uint32_t free;
};

// Virtualized flag register.
struct VirtualFlag {
    uint8_t idx : 6;
    uint8_t n : 2;

    constexpr VirtualFlag() : idx(0), n(0) {}
    /* implicit */ VirtualFlag(const ngen::FlagRegister &flag) : idx(flag.index()), n(flag.getBytes() >> 1) {}
    explicit constexpr VirtualFlag(int idx_, int n_ = 1) : idx(idx_), n(n_) {}

    ngen::FlagRegister toPhysical() const;

    friend inline bool operator==(VirtualFlag vf1, VirtualFlag vf2) { return vf1.idx == vf2.idx && vf1.n == vf2.n; }
    friend inline bool operator!=(VirtualFlag vf1, VirtualFlag vf2) { return !(vf1 == vf2); }

    bool operator!() const          { return (idx == 0) && (n == 0); }
    explicit operator bool() const  { return !!*this; }

    void clear()                    { *this = VirtualFlag(); }

    int getBytes() const            { return n << 1; }
};

// Allocator for virtual flag registers.
class VirtualFlagAllocator {
public:
    VirtualFlagAllocator(ngen::HW hw) : free(~uint64_t(0)),
                                        nflag(ngen::FlagRegister::subcount(hw)) {}

    VirtualFlag allocVirtual(int n = 1);
    ngen::FlagRegister alloc(int n = 1);
    ngen::FlagRegister allocSubreg0();
    ngen::FlagRegister tryAlloc(int n = 1);

    void claim(VirtualFlag vflag)                   { free &= ~mask(vflag); }
    void release(VirtualFlag vflag)                 { free |= mask(vflag); }
    void release(const ngen::FlagRegister &reg)     { release(VirtualFlag(reg)); unlock(reg); }
    void safeRelease(VirtualFlag &vflag)            { if (vflag) release(vflag); vflag.clear(); }
    void safeRelease(ngen::FlagRegister &reg)       { if (reg.isValid()) release(reg); reg.invalidate(); }
    bool isFree(VirtualFlag vflag)            const { return !(~free & mask(vflag)); }

    bool isVirtual(VirtualFlag vflag)               { return (vflag.idx >= nflag); }

    bool lock(VirtualFlag vflag)                    { bool wasLocked = isLocked(vflag); locked |= mask(vflag); return wasLocked; }
    void unlock(VirtualFlag vflag)                  { locked &= ~mask(vflag); }
    bool isLocked(VirtualFlag vflag)          const { return !(~locked & mask(vflag)); }
    bool canLock(int n = 1) const;
    void freeUnlocked();

    ngen::FlagRegister assignPhysical(VirtualFlag vflag);

protected:
    uint64_t free;
    uint8_t locked = 0;
    uint8_t nextPhys = 0;
    uint8_t nflag;

    static uint64_t mask(VirtualFlag vflag)         { return mask(vflag.idx, vflag.n); }
    static uint64_t mask(int idx, int n)            { return (uint64_t(1) << (idx + n)) - (uint64_t(1) << idx); }
};

#include "internal/namespace_end.hxx"

#endif
