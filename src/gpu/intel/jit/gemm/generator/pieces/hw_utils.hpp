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


#ifndef GEMMSTONE_GUARD_HW_UTILS_HPP
#define GEMMSTONE_GUARD_HW_UTILS_HPP

#include "internal/ngen_includes.hpp"
#include "register_block.hpp"
#include "strategy.hpp"

#include "internal/namespace_start.hxx"

template <typename T>
static inline constexpr int elementsPerGRF(ngen::HW hw)
{
    return ngen::GRF::bytes(hw) / sizeof(T);
}

static inline constexpr int elementsPerGRF(ngen::HW hw, Type T)
{
    return ngen::GRF::bytes(hw) / T;
}

static inline constexpr int elementsPerGRF(ngen::HW hw, ngen::DataType dt)
{
    return (ngen::GRF::bytes(hw) >> getLog2Bytes(dt)) * elementsPerByte(dt);
}

static inline bool canSwizzle(ngen::HW hw, ngen::DataType dt)
{
    using namespace ngen;

    if (hw < HW::XeHP) return true;

    switch (dt) {
        case DataType::b:
        case DataType::ub:
        case DataType::w:
        case DataType::uw:
        case DataType::d:
        case DataType::ud: return true;
        case DataType::q:
        case DataType::uq: return (hw >= HW::XeHPC);
        default:           return false;
    }
}

static inline bool canSwizzle(ngen::HW hw, Type T)
{
    return canSwizzle(hw, T.ngen());
}

static inline bool hasNativeAtomicAdd(ngen::HW hw, Type T, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    using namespace ngen;

    bool floatAtomics = (astrategy.base.getModel() == ModelA64);
    if (astrategy.newDP)
        floatAtomics |= (astrategy.base.getModel() != ModelSLM);

    if (T.isInt4())
        return false;
    if (T.isInteger() && T.size() >= (astrategy.newDP ? 2 : 4))
        return true;
    else if (T == Type::f32)
        return floatAtomics && (hw >= HW::XeHP);
    else if (T == Type::f64)
        return floatAtomics && (hw >= HW::XeHPC);
    else
        return false;
}

static inline size_t slmCapacity(ngen::HW hw)
{
    using namespace ngen;
    switch (hw) {
        case HW::Gen9:
        case HW::Gen11:     return 65536;
        case HW::Gen12LP:
        case HW::XeHP:
        case HW::XeHPG:
        case HW::XeHPC:     return 131072;
        case HW::Xe2:       return 131072;
        default:
            return 0;
    }
}

static inline size_t maxSLMPerWG(ngen::HW hw, int grfCount)
{
    auto slmMax = slmCapacity(hw);
    if (hw <= ngen::HW::XeHPG && grfCount <= 128)
        slmMax = std::min<size_t>(slmMax, 65536);
    return slmMax;
}

static inline int threadsPerEU(ngen::HW hw, const CommonStrategy &strategy)
{
    if (hw >= ngen::HW::XeHP)
        return (strategy.GRFs > 128) ? 4 : 8;
    else
        return 7;
}

static inline int eusPerSubslice(ngen::HW hw)
{
    using namespace ngen;
    switch (hw) {
        case HW::Gen9:
        case HW::Gen11:
        case HW::XeHPC:
        case HW::Xe2:
            return 8;
        case HW::Gen12LP:
        case HW::XeHP:
        case HW::XeHPG:
            return 16;
        default:
            return 0;
    }
}

// Maximum SIMD width for a scattered message.
static inline int maxScatteredSIMD(ngen::HW hw, const MatrixAddressingStrategy &astrategy)
{
    if (astrategy.newDP)
        return ngen::GRF::bytes(hw) >> 1;
    return 16;
}

// Minimum native SIMD width for a scattered message.
static inline int minScatteredSIMD(ngen::HW hw, const MatrixAddressingStrategy &astrategy)
{
    if (hw >= ngen::HW::XeHPC)
        return 16;
    return maxScatteredSIMD(hw, astrategy) >> 1;
}

// Minimum stride alignment for a block 2D message.
static inline int block2DMinAlignment(ngen::HW hw, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, bool asIfBlock2D = false)
{
    using namespace ngen;
    if (!isBlock2D(astrategy.accessType) && !asIfBlock2D) return 0;
    if (hw == HW::Xe2) return 16;
    return (isTransposing(astrategy.accessType) || astrategy.prefetch) ? 4 : 8;
}

// Minimum base address alignment for block 2D messages.
static inline int block2DBaseAlignment(ngen::HW hw, int stepping)
{
    using namespace ngen;
    if (hw == HW::XeHPC && stepping < SteppingPVCXTB4)
        return 128;
    return 64;
}

// Width alignment for a block 2D message.
static inline int block2DWidthAlignment(Type T, const RegisterBlock &block, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    // Block 2D width must be DW-aligned, but generally use QW alignment for better performance for reads.
    return ((astrategy.noExtraPad || block.writable || atype.alignment % 8) ? 4 : 8);
}

#include "internal/namespace_end.hxx"

#endif /* header guard */
