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

#ifndef GEMMSTONE_GUARD_REGISTER_BLOCK_HPP
#define GEMMSTONE_GUARD_REGISTER_BLOCK_HPP

#include <array>
#include <cstdint>

#include "internal/ngen_includes.hpp"
#include "type.hpp"
#include "allocators.hpp"

#include "internal/namespace_start.hxx"

struct MatrixAddressingStrategy;

// MaskInfo: logical description of a message mask, used to ensure in-bounds matrix accesses.
struct MaskInfo {
    union {
        struct {
            uint8_t isFixed : 1;  // = false (variable mask)
            uint8_t reverse : 1;  // True to reverse mask.
            uint8_t rshift : 6;   // Power of 2 by which to divide index before forming mask. Fractions are rounded up.
                                  // Note maskRep * bitRep * (rsize >> rshift) = # mask bits.
            uint8_t rsize;        // Maximum remainder value. (e.g. 16 if we need the last 4 bits of the index).
            uint8_t maskRep;      // # of repetitions of mask pattern.
            uint8_t bitRep;       // # of times each mask bit is repeated.
        } variable;
        struct {
            uint8_t isFixed : 1;  // = true (fixed mask)
            uint8_t _ : 7;
            uint8_t rsize;        // Maximum remainder value.
            uint16_t value;       // Mask value.
        } fixed;
        uint32_t raw;
    };

    MaskInfo() : fixed{true,0,0,0xFFFF} {}

    bool operator!()         const { return fixed.isFixed && fixed.value == 0xFFFF; }
    explicit operator bool() const { return !!*this; }

    static MaskInfo None() { return MaskInfo(); }

    friend bool operator==(const MaskInfo &i1, const MaskInfo &i2) {
        return i1.raw == i2.raw;
    }
    friend bool operator!=(const MaskInfo &i1, const MaskInfo &i2) {
        return !(i1 == i2);
    }
};

// RegisterBlock encapsulates a single matrix tile resident in registers,
//   along with information needed to move it to/from memory.
// Generally speaking RegisterBlocks map 1-1 to individual load/store/atomic instructions,
//   but some instructions may have multiple RegisterBlocks, e.g. 2D block arrays.
// It is also possible for RegisterBlocks not to be backed by memory, indicated by
//   a zero simdSize field.
struct RegisterBlock {
    /* Register layout information. */
    uint16_t nr, nc;            // Size of this block.
    uint16_t ld;                // Leading dimension, in elements.
    uint16_t offsetR, offsetC;  // Row and column offset within matrix block.
    uint8_t colMajor : 1;       // Is this block column-major? (columns stored consecutively inside each register)
    uint8_t splitComplex : 1;   // True if complex data split into successive real and imaginary parts.
    uint8_t byteGlue : 1;       // True if strided sub-byte data is unit stride within each byte.
    uint8_t : 5;
    uint8_t crosspack;          // Crosspack for this block (1 if none).
    uint8_t component;          // Component # for this block.
    int8_t cxComponent;         // Complex component # for this block (-1 if not complex or interleaved).
    uint16_t bytes;             // # of bytes in this block.
    uint16_t offsetBytes;       // Byte offset within register block.

    /* Load/store information. */
    uint8_t remainderR : 1;     // Row remaindering enabled?
    uint8_t remainderC : 1;     // Column remaindering enabled?
    uint8_t noRowsOK : 1;       // Can handle no rows (in mask/descriptor)?
    uint8_t noColsOK : 1;       // Can handle no columns (in mask/descriptor)?
    uint8_t descRemR : 1;       // Row remainders can be handled by changing the descriptor?
    uint8_t descRemC : 1;       // Column remainders can be handled by changing the descriptor?
    uint8_t descAssigned : 1;   // True if address registers have been assigned for this block's descriptors.
    uint8_t writable : 1;       // True if block is set up for writing.

    uint8_t ebytes;             // Size of element in bytes, e.g. 4 for scattered_dword, 16 for block_hword
    uint8_t count;              // Element count.
    uint8_t extra;              // Extra info. For block accesses, 1 means aligned OWord, 0 unaligned. For scattered accesses, # of consecutive elements.
    uint8_t simdSize;           // SIMD size for load/stores (0 indicating no associated load/store.)
    uint8_t msgRegs;            // Underlying register count for load/store operation (may be different from nregs()).
    std::array<VirtualFlag, 2> flag;
                                // Assigned flag register indices ([0] -> row, [1] -> column)
    uint8_t flagAny : 1;        // Use .anyh?
    uint8_t flagAll : 1;        // Use .allh?
    uint8_t flagInvert : 1;     // Invert flag?
    uint8_t hasNoLoad : 1;      // Does this load/store cover additional (no-load) RegisterBlocks? (packed layouts)
    uint8_t : 4;
    uint8_t sfid;               // SFID for this block.
    uint8_t rowFragment;        // If this block needs fragmenting to support row/column remainders, the maximum block size (power of 2) to fragment down to.
    uint8_t colFragment;        //     Zero if no fragmenting needed.
    uint8_t addrShift;          // log2(address units). e.g. 0 if byte addresses should be used, 4 if oword addresses should be used.
    uint8_t log2GRFBytes;       // log2(bytes per GRF).

    MaskInfo rowMask;           // Row mask for this block.
    MaskInfo colMask;           // Column mask for this block.

    int32_t offsetAddr;         // Address offset, for sharing address registers. For 2D addressing, contains x/y offsets in low/high words.

    static constexpr int8_t Interleaved = -1;     // Value for cxComponent indicating interleaved real/imaginary data.

    void calcBytes(Type T);     // Auto-calculate # of registers.
    void calcBytes(Type T, const MatrixAddressingStrategy &astrategy);

    bool hasFlag() const     { return flag[0] || flag[1]; }
    void clearFlag()         { flag[0].clear(); flag[1].clear(); flagAll = flagAny = flagInvert = false; }
    void eraseMask()         { clearFlag(); rowMask = MaskInfo(); colMask = MaskInfo(); }

    bool isLoadBlock() const { return simdSize > 0; }

    bool grfAligned() const  { return (offsetBytes & ((1 << log2GRFBytes) - 1)) == 0; }
    int nregs() const;
    int offsetReg() const;

    void simplify(Type T);
    void compact(Type T);

    ngen::Offset2D offset2D() const { return ngen::Offset2D(int16_t(offsetAddr & 0xFFFF), int16_t(offsetAddr >> 16)); }
    void set2DOffset(int16_t x, int16_t y) { offsetAddr = uint16_t(x) | (uint32_t(uint16_t(y)) << 16); }
    void subAddrOffset(int32_t aoff, bool is2D) {
        if (is2D)
            set2DOffset((offsetAddr - aoff) & 0xFFFF, (offsetAddr >> 16) - (aoff >> 16));
        else
            offsetAddr -= aoff;
    }

    bool isSplitComplex() const { return (splitComplex || cxComponent != Interleaved); }
};

// Preferences for remainder handling.
enum RemainderOptions : uint8_t {
    AvoidFragment = 0,      // Avoid/allow making blocks that will need to be broken up
    AllowFragment = 1,      //  ("fragmented") during remainder handling.
    AllowDescriptors = 2,   // Allow indirect send descriptor-based remainder handling.
    AllowFragDesc = 3,      // Allow fragmentation and descriptors.
    NoFixedMasks = 4,       // Do not allow fixed masks.
    AllowFragDescNFM = 7,   // Allow fragmentation and descriptors, but no fixed masks
};

// Detect crosspacked cases that should be converted to equivalent transposed layouts.
static inline bool isLargeCrosspack(Type T, int crosspack) {
    return (crosspack * T > 4) && (crosspack > 1);
}

#include "internal/namespace_end.hxx"

#endif /* header guard */
