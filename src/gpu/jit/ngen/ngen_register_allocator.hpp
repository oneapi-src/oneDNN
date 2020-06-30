/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef _GEN_REGISTER_ALLOCATOR_HPP__
#define _GEN_REGISTER_ALLOCATOR_HPP__

#include "ngen.hpp"
#include <cstdint>
#include <stdexcept>

namespace ngen {

// Gen registers are organized in banks of bundles.
// Each bundle is modeled as groups of contiguous registers separated by a stride.
struct Bundle {
    static const int8_t any = -1;

    int8_t bundle_id;
    int8_t bank_id;

    Bundle() : bundle_id(any), bank_id(any) {}
    Bundle(int8_t bank_id_, int8_t bundle_id_) : bundle_id(bundle_id_), bank_id(bank_id_) {}

    // Number of bundles in each bank (per thread).
    static constexpr int bundle_count(ngen::HW hw)    { return (hw == ngen::HW::Gen12LP) ? 8 : 2; }
    // Number of banks.
    static constexpr int bank_count(ngen::HW hw)      { return 2; }

    static Bundle locate(ngen::HW hw, ngen::RegData reg);

    int first_reg(ngen::HW hw) const;                  // The first register in the bundle.
    int group_size(ngen::HW hw) const;                 // Number of registers in each contiguous group of the bundle.
    int stride(ngen::HW hw) const;                     // Stride between register groups of the bundle.

    int64_t reg_mask(ngen::HW hw, int offset) const;   // Get register mask for this bundle, for registers [64*offset, 64*(offset+1)).

    friend constexpr bool operator==(const Bundle &b1, const Bundle &b2) {
        return b1.bundle_id == b2.bundle_id && b1.bank_id == b2.bank_id;
    }

    static bool conflicts(ngen::HW hw, ngen::RegData r1, ngen::RegData r2) {
        return !r1.isNull() && !r2.isNull() && (locate(hw, r1) == locate(hw, r2));
    }

    static bool same_bank(ngen::HW hw, ngen::RegData r1, ngen::RegData r2) {
        return !r1.isNull() && !r2.isNull() && (locate(hw, r1).bank_id == locate(hw, r2).bank_id);
    }
};

// Gen register allocator.
template <int register_count = 128>
class RegisterAllocator {
public:
    explicit RegisterAllocator(ngen::HW hw_) : hw(hw_) { init(); }

    // Allocation functions: sub-GRFs, full GRFs, and GRF ranges.
    ngen::GRFRange alloc_range(int nregs, Bundle base_bundle = Bundle());
    ngen::GRF alloc(Bundle bundle = Bundle()) { return alloc_range(1, bundle)[0]; }

    ngen::Subregister alloc_sub(ngen::DataType type, Bundle bundle = Bundle());
    template <typename T>
    ngen::Subregister alloc_sub(Bundle bundle = Bundle()) { return alloc_sub(ngen::getDataType<T>(), bundle); }

    ngen::FlagRegister alloc_flag();

    void release(ngen::GRF reg);
    void release(ngen::GRFRange range);
    void release(ngen::Subregister subreg);
    void release(ngen::FlagRegister flag);

    template <typename RD>
    void safeRelease(RD &reg) { if (!reg.isInvalid()) release(reg); reg.invalidate(); }

    // Claim specific registers.
    void claim(ngen::GRF reg);
    void claim(ngen::GRFRange range);
    void claim(ngen::Subregister subreg);
    void claim(ngen::FlagRegister flag);

    void dump(std::ostream &str);

protected:
    ngen::HW hw;                            // HW generation.
    uint8_t free_whole[register_count / 8]; // Bitmap of free whole GRFs.
    uint8_t free_sub[register_count];       // Bitmap of free partial GRFs, at dword granularity.
    uint8_t free_flag;                      // Bitmap of free flag registers.

    void init();
    void claim_sub(int r, int o, int dw);
};


// Exceptions.
class out_of_registers_exception : public std::runtime_error {
public:
    out_of_registers_exception() : std::runtime_error("Insufficient registers in requested bundle") {}
};

} /* namespace ngen */

#endif /* include guard */
