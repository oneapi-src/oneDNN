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


#include "layout_utils.hpp"
#include "hw_utils.hpp"

using std::vector;
using namespace ngen;

#include "internal/namespace_start.hxx"


int untile(Type T, const MatrixAddressing &atype, int component, int i, int j, int r, int c, int tileR, int tileC, bool reverse)
{
    bool cm = isColMajor(atype.layout) ^ reverse;

    if (isPacked(atype.layout)) {
        (cm ? r : c) = atype.packSize;

        auto &pl = (cm ? c : r);
        if (atype.panelLength)
            pl = atype.panelLength;
    }

    int cpR = cm ? 1 : atype.crosspack;
    int cpC = cm ? atype.crosspack : 1;

    if (tileR == 0) tileR = r;
    if (tileC == 0) tileC = c;

    int rstride  = cm ? tileC : c;
    int cstride  = cm ? r : tileR;
    int rtstride = cm ? cpC : tileC;
    int ctstride = cm ? tileR : cpR;

    rstride *= T.components();
    cstride *= T.components();

    if (tileR == 0) tileR = 1;    /* arbitrary value */
    if (tileC == 0) tileC = 1;

    int iTile = i % tileR;
    int jTile = j % tileC;
    i -= iTile; j -= jTile;
    int iCP = iTile % cpR;
    int jCP = jTile % cpC;
    iTile -= iCP; jTile -= jCP;
    int idx = i * rstride + j * cstride + tileR * tileC * component + iTile * rtstride + jTile * ctstride + iCP + jCP;
    return idx;
}

int consecutiveElements(int r, int c, const MatrixAddressing &atype)
{
    int x = isColMajor(atype.layout) ? r : c;
    int y = isColMajor(atype.layout) ? c : r;

    if (isPacked(atype.layout)) {
        int effTileX = (atype.layout == MatrixLayout::Pc) ? atype.tileR : atype.tileC;
        int effTileY = (atype.layout == MatrixLayout::Pc) ? atype.tileC : atype.tileR;
        if (!effTileX) effTileX = atype.packSize;
        if (!effTileY) effTileY = atype.crosspack;

        if (y % effTileY == 0) {
            if (x == atype.packSize)
                return x * y;
            else if (x % effTileX == 0)
                return x * effTileY;
        }
        if (y % atype.crosspack == 0)
            return std::min(x, effTileX) * atype.crosspack;
    }

    return x;
}

void getGranularities(const MatrixAddressing &atype, int &rgran, int &cgran)
{
    auto &xgran = isColMajor(atype.layout) ? cgran : rgran;
    auto &ygran = isColMajor(atype.layout) ? rgran : cgran;
    rgran = std::max<int>(atype.tileR, 1);
    cgran = std::max<int>(atype.tileC, 1);
    xgran = std::max<int>(xgran, atype.crosspack);
    if (isPacked(atype.layout))
        ygran = std::max<int>(ygran, atype.packSize);
}

bool needsPseudoblock(HW hw, Type T, int r, int c,
                      const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                      bool writable, bool masked)
{
    if (astrategy.accessType == AccessType::PseudoBlock) return true;
    if (astrategy.accessType != AccessType::Block) return false;

    auto consecutive = consecutiveElements(r, c, atype);
    bool dwAligned = (atype.alignment & 0x3) == 0;
    bool owAligned = (atype.alignment & 0xF) == 0;
    bool pseudo = !dwAligned
               || ((consecutive * T) & 0x3)
               || (writable && ((consecutive * T) & 0xF) && !astrategy.newDP)
               || (writable && !owAligned && !astrategy.newDP)
               || (writable && masked && (T.paddedSize() & 3))
               || (masked && !owAligned && (hw >= HW::XeHP || astrategy.base.getModel() != ModelA64))
               || (astrategy.newDP && masked)
               || (hw >= HW::XeHPC && masked)
               || (hw >= HW::XeHPC && !astrategy.padded && !astrategy.newDP && ((r * c * T) & 0xF))
               || astrategy.atomic
               || (isColMajor(atype.layout) ? c : r) % atype.crosspack
               || ((astrategy.base.getModel() == ModelSLM) && (hw < HW::Gen11 || !(owAligned || astrategy.newDP)));

    return pseudo;
}

bool pseudoblockUseChannelScattered(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const RegisterBlock &block)
{
    return (astrategy.base.getModel() == ModelSLM) && (block.ebytes == 4) && !astrategy.atomic;
}

AccessType effectiveAccessType(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const RegisterBlock &block)
{
    auto type = astrategy.accessType;
    if (!block.isLoadBlock())
        return type;
    if (type == AccessType::Block && block.ebytes < 16 && block.extra)
        type = AccessType::PseudoBlock;
    else if (type == AccessType::Scattered && astrategy.base.getModel() == ModelSLM && block.ebytes == 4 && !astrategy.newDP)
        type = AccessType::ChannelScattered;
    else if (type == AccessType::ChannelScattered && (block.ebytes != 4 || astrategy.atomic))
        type = AccessType::Scattered;
    return type;
}

AccessType implAccessType(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const RegisterBlock &block)
{
    auto type = effectiveAccessType(atype, astrategy, block);
    if (type == AccessType::PseudoBlock)
        type = pseudoblockUseChannelScattered(atype, astrategy, block) ? AccessType::ChannelScattered : AccessType::Scattered;
    else if (type == AccessType::CacheLine)
        type = AccessType::Scattered;
    return type;
}

void getBlock2DWH(int &w, int &h, int &count, const MatrixAddressing &atype, const RegisterBlock &block, int *outMultiX)
{
    int multiX = 1;
    bool transpose = (isColMajor(atype.layout) != block.colMajor);
    w = isColMajor(atype.layout) ? block.nr : block.nc;
    h = isColMajor(atype.layout) ? block.nc : block.nr;
    w = (w * block.extra) / (block.ebytes * 8);     /* block.extra: #bits in logical data type */
    if (isPacked(atype.layout)) {
        int maxW = 64 / block.ebytes;
        multiX = div_up(w, maxW);
        w /= multiX;
        h *= multiX;
    }
    count = block.count;
    if (transpose) {
        h *= count;
        count = 1;
    }
    if (outMultiX) *outMultiX = multiX;
}

int addrGRFCount(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const RegisterBlock &block)
{
    // Non-load blocks don't get address registers.
    if (!block.isLoadBlock())
        return 0;

    // Offset blocks don't either -- they will share existing address registers.
    if (block.offsetAddr != 0)
        return 0;

    switch (effectiveAccessType(atype, astrategy, block)) {
        case AccessType::Scattered:
        case AccessType::ChannelScattered:
        case AccessType::PseudoBlock:
        case AccessType::CacheLine: {
            auto bytesPerAddr = (astrategy.base.getModel() == ModelA64) ? 8 : 4;
            auto baseSIMD = std::max<int>(block.simdSize, 8);
            auto log2Bytes = block.log2GRFBytes;
            return (bytesPerAddr * baseSIMD + (1 << log2Bytes) - 1) >> log2Bytes;
        }
        case AccessType::Block:
        case AccessType::Block2D:
        case AccessType::Block2DTranspose:
        case AccessType::Block2DVNNI:
            return 1;
    }
    stub("Invalid addressing.");
}

bool tryAllocAddrRegs(vector<GRFRange> &addrRegs, const vector<RegisterBlock> &layout,
                      const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                      CommonState &state, Bundle hint)
{
    auto nblocks = int(layout.size());
    bool ok = true;

    addrRegs.resize(nblocks);

    GRFRange last;
    for (int l = 0; l < nblocks && ok; l++) {
        if (layout[l].offsetAddr == 0) {
            auto count = addrGRFCount(atype, astrategy, layout[l]);
            if (count < 1) continue;
            last = state.ra.try_alloc_range(count, hint);
            ok &= last.isValid();
        }
        addrRegs[l] = last;
    }

    if (!ok) {
        for (auto &regs: addrRegs) state.ra.safeRelease(regs);
        addrRegs.clear();
    }

    return ok;
}

void allocAddrRegs(vector<GRFRange> &addrRegs, const vector<RegisterBlock> &layout,
    const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
    CommonState &state, Bundle hint)
{
    if (!tryAllocAddrRegs(addrRegs, layout, atype, astrategy, state, hint))
        throw out_of_registers_exception();
}

bool isLayoutColMajor(const vector<RegisterBlock> &layout)
{
    if (layout.size() == 0)
        stub("Empty layout.");
    return layout[0].colMajor;              // All layouts we create are homogeneous currently.
}

void getLayoutDims(const vector<RegisterBlock> &layout, int &m, int &n)
{
    // For now all layouts are sorted so last block is in lower-right corner.
    if (layout.size() == 0)
        stub("Empty layout.");
    auto &first = layout[0];
    auto &last = layout[layout.size() - 1];
    m = last.offsetR + last.nr - first.offsetR;
    n = last.offsetC + last.nc - first.offsetC;
}

bool hasFullCrosspack(const vector<RegisterBlock> &layout, int crosspack)
{
    if (layout.size() == 0)
        return true;
    if (layout[0].crosspack != crosspack)  // Only need to check first block of layout currently.
        return false;
    for (const auto &block: layout)
        if ((block.colMajor ? block.nc : block.nr) % crosspack)
            return false;
    return true;
}

bool hasTiling(const vector<RegisterBlock> &layout, int tileR, int tileC)
{
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

bool hasRemainders(const vector<RegisterBlock> &layout, bool remainderR, bool remainderC)
{
    for (auto &block : layout)
        if ((remainderR && block.remainderR) || (remainderC && block.remainderC))
            return true;
    return false;
}

bool hasFragmenting(const vector<RegisterBlock> &layout, bool ignoreWholeFragR, bool ignoreWholeFragC)
{
    if (layout.empty()) return false;

    int r = 0, c = 0;
    if (ignoreWholeFragR || ignoreWholeFragC)
        getLayoutDims(layout, r, c);

    for (auto &block : layout) {
        if (block.rowFragment && !(ignoreWholeFragR && block.rowFragment >= r))
            return true;
        if (block.colFragment && !(ignoreWholeFragC && block.colFragment >= c))
            return true;
    }
    return false;
}

bool hasMasking(const vector<RegisterBlock> &layout)
{
    for (auto &block : layout)
        if (block.rowMask || block.colMask || block.hasFlag())
            return true;
    return false;
}

bool hasFlags(const vector<RegisterBlock> &layout)
{
    for (auto &block : layout)
        if (block.hasFlag())
            return true;
    return false;
}

int getMaxLoadBlock(const vector<RegisterBlock> &layout)
{
    int result = 0;
    for (auto &l: layout)
        result = std::max<int>(result, l.msgRegs);
    return result;
}

int getRegCount(const vector<RegisterBlock> &layout)
{
    if (layout.empty()) return 0;

    int lastByte = 0;
    for (auto &l : layout)
        lastByte = std::max(lastByte, l.offsetBytes + l.bytes);

    int log2Bytes = layout[0].log2GRFBytes;
    return (lastByte + (1 << log2Bytes) - 1) >> log2Bytes;
}

Subregister findBlockReg(Type T, const RegisterBlock &block, int rr, int cc, const GRFMultirange &regs,
                         int &nelems, int cxComponent, int component)
{
    auto Te = T;

    if (rr < 0 || rr >= block.nr || cc < 0 || cc >= block.nc || component != block.component || !one_of(block.cxComponent, -1, cxComponent))
        stub("Requested out-of-bounds element.");

    int crosspack = block.crosspack;
    int xx = block.colMajor ? rr : cc;
    int yy = block.colMajor ? cc : rr;
    int nx = block.colMajor ? block.nr : block.nc;
    nelems = nx - xx;

    int yyx = yy % crosspack;
    yy -= yyx;

    if (block.byteGlue) {
        int xxx = xx & (T.perByte() - 1);
        yyx = yyx * T.perByte() + xxx;
        xx -= xxx;
        nelems = 1;
    }

    int elFixed = yyx + (xx * crosspack);
    int elLD = yy;

    int el = elFixed + elLD * block.ld;
    el += block.offsetBytes / Te;

    int consecutive;
    auto result = regs.sub(block.log2GRFBytes, el, Te.ngen(), &consecutive);

    nelems = std::min(nelems, div_up(consecutive, crosspack));
    return result;
}

Subregister findBlockReg(Type T, const vector<RegisterBlock> &layout, int r, int c, const GRFMultirange &regs,
                         int &nelems, const RegisterBlock *&block,
                         int cxComponent, int component)
{
    int ecomponent = component;
    for (auto &l : layout) {
        int rr = r - l.offsetR;
        int cc = c - l.offsetC;
        if (rr >= 0 && rr < l.nr && cc >= 0 && cc < l.nc && ecomponent == l.component && one_of(l.cxComponent, cxComponent, RegisterBlock::Interleaved)) {
            block = &l;
            return findBlockReg(T, l, rr, cc, regs, nelems, cxComponent, component);
        }
    }

    stub("Could not find requested matrix element in layout.");
}

static RegisterRegion blockRegion(Type T, const Subregister &reg, const RegisterBlock &block,
                                  int rr, int cc, int &nelems, int cxComponent, bool allow2D)
{
    auto cp = block.crosspack;

    if (block.byteGlue && allow2D && T.bits() < 8) {
        nelems = block.colMajor ? (block.nr - rr) : (block.nc - cc);
        return reg(cp / T, 1 / T, 1);
    } else
        return reg(cp);
}

RegisterRegion findBlockRegion(Type T, const RegisterBlock &block, int rr, int cc,
                               const GRFMultirange &regs, int &nelems,
                               int cxComponent, int component, bool allow2D)
{
    auto reg = findBlockReg(T, block, rr, cc, regs, nelems, cxComponent, component);
    return blockRegion(T, reg, block, rr, cc, nelems, cxComponent, allow2D);
}

RegisterRegion findBlockRegion(Type T, const std::vector<RegisterBlock> &layout, int r, int c,
                               const GRFMultirange &regs, int &nelems, const RegisterBlock *&block,
                               int cxComponent, int component, bool allow2D)
{
    auto reg = findBlockReg(T, layout, r, c, regs, nelems, block, cxComponent, component);
    return blockRegion(T, reg, *block, r - block->offsetR, c - block->offsetC, nelems, cxComponent, allow2D);
}

int getAddr0Offset(const RegisterBlock &block, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    if (astrategy.newDP) return 0;
    if (astrategy.base.getModel() == ModelA64) return 0;
    if (effectiveAccessType(atype, astrategy, block) == AccessType::Block) return 2;
    return 0;
}

Subregister getOriginAddr(const vector<RegisterBlock> &layout, const vector<GRFRange> &addrRegs,
                          const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, int *shiftOut)
{
    bool a64 = (astrategy.base.getModel() == ModelA64);

    for (size_t b = 0; b < layout.size(); b++) {
        const auto &block = layout[b];
        if ((block.offsetR != 0) || (block.offsetC != 0))
            continue;

        int off = getAddr0Offset(block, atype, astrategy);

        if (shiftOut) *shiftOut = block.addrShift;
        return addrRegs[b][0].sub(off, a64 ? DataType::uq : DataType::ud);
    }

    if (shiftOut) *shiftOut = 0;
    return Subregister();
}

int contiguityCheck(HW hw, const RegisterBlock &block, const GRFMultirange &range)
{
    auto offsetBytes = block.offsetBytes;
    if (offsetBytes & (GRF::bytes(hw) - 1))
        if (block.isLoadBlock())
            stub();
    auto offsetReg = offsetBytes >> GRF::log2Bytes(hw);
    auto lastReg = GRF::bytesToGRFs(hw, offsetBytes + block.bytes);
    if (!range.contiguous(offsetReg, lastReg - offsetReg)) stub();

    return offsetReg;
}

GRFMultirange subrange(GRFMultirange r, HW hw, Type T, const RegisterBlock &block)
{
    int ne = elementsPerGRF(hw, T);
    int ldGRFs = div_up(block.ld, ne);
    int ldUsedGRFs = div_up(block.colMajor ? block.nr : block.nc, ne);
    int td = block.colMajor ? block.nc : block.nr;

    if (ldUsedGRFs >= ldGRFs)
        return r.subrange(block.offsetReg(), block.nregs());
    else {
        int offReg = block.offsetReg();
        GRFMultirange result = r.subrange(offReg, ldUsedGRFs);
        for (int y = 1; y < td; y++) {
            offReg += ldGRFs;
            result.append(r.subrange(offReg, ldUsedGRFs));
        }
        return result;
    }
}

void unlinkFromMemory(vector<RegisterBlock> &layout)
{
    for (auto &block: layout)
        block.simdSize = 0;
}

void sortRegLayout(Type T, vector<RegisterBlock> &layout, int r, int c,
                   const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, bool reverse)
{
    auto order = [=](const RegisterBlock &block) {
        return untile(T, atype, block, r, c, astrategy.tileR, astrategy.tileC, reverse);
    };

    std::sort(layout.begin(), layout.end(), [&](const RegisterBlock &b1, const RegisterBlock &b2) {
        return (order(b1) < order(b2));
    });
}


bool matchLayouts(Type T, vector<RegisterBlock> &layout, const vector<RegisterBlock> &layoutRef)
{
    vector<RegisterBlock> nlayout = layout;

    if (getRegCount(layoutRef) >= GRF::maxRegs()) return false;

    int lastByteAdjust = 0;

    for (auto &nblock : nlayout) {
        int nelems;
        const RegisterBlock *blockRef;
        auto sr = findBlockReg(T, layoutRef, nblock.offsetR, nblock.offsetC, GRFRange(0, GRF::maxRegs() - 2), nelems, blockRef);

        // Check:
        //  1. Does this register block's offset match the reference block's offset?
        if (sr.getByteOffset() != (nblock.offsetBytes & ((1 << nblock.log2GRFBytes) - 1))) return false;

        //  2. Is there any free space in the register block?
        if (nblock.nr * nblock.nc * T != nblock.bytes) return false;

        //  3. Does this register block's data layout match the reference block's layout?
        if (blockRef->colMajor != nblock.colMajor) return false;
        if (blockRef->crosspack != nblock.crosspack) return false;

        //  4. Does this register block fit inside the reference block?
        auto RegisterBlock::* nx = nblock.colMajor ? &RegisterBlock::nr : &RegisterBlock::nc;
        auto RegisterBlock::* ny = nblock.colMajor ? &RegisterBlock::nc : &RegisterBlock::nr;

        if (nblock.*nx > blockRef->*nx) return false;
        if (nblock.*ny > blockRef->*ny) return false;

        //  5. Are the leading dimensions and padding compatible?
        if (nblock.*nx < blockRef->*nx) {
            if (nblock.*ny > nblock.crosspack) return false;
            if (nblock.*ny < nblock.crosspack && nblock.*ny < blockRef->*ny) return false;
        }

        if (nblock.*ny > nblock.crosspack && (nblock.ld != blockRef->ld))
            return false;

        // Point this register block where it belongs.
        auto newOffsetBytes = (sr.getBase() << nblock.log2GRFBytes) + sr.getByteOffset();
        auto byteAdjust = newOffsetBytes - nblock.offsetBytes;

        // No-load blocks need to stay with their parent blocks.
        if (nblock.simdSize == 0 && byteAdjust != lastByteAdjust)
            return false;

        nblock.offsetBytes = newOffsetBytes;
        lastByteAdjust = byteAdjust;
    }

    std::swap(nlayout, layout);
    return true;
}

void assignUniformMask(vector<RegisterBlock> &layout, FlagRegister flag, int idx)
{
    for (auto &block: layout) {
        if (block.flag[idx]) stub();     /* Already has a flag? */
        block.flag[idx] = flag;
    }
}

bool assignAllDescs(vector<RegisterBlock> &layout)
{
    for (auto &block : layout) {
        if (!block.descRemR && !block.descRemC)
            continue;
        if (block.simdSize != layout[0].simdSize)
            return false;
        block.descAssigned = true;
        block.sfid = layout[0].sfid;
    }

    return true;
}

// Split 2D block array loads into multiple blocks.
void postprocessLayout2D(vector<RegisterBlock> &layout, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    if (!isBlock2D(astrategy.accessType)) return;

    int maxCount = 1;
    for (auto &block : layout)
        maxCount = std::max(maxCount, int(block.count));
    if (maxCount == 1) return;

    vector<RegisterBlock> xlayout;
    xlayout.reserve(layout.size() * maxCount);

    for (auto &block : layout) {
        bool cm = block.colMajor;
        auto RegisterBlock::* nx      = cm ? &RegisterBlock::nr      : &RegisterBlock::nc;
        auto RegisterBlock::* offsetX = cm ? &RegisterBlock::offsetR : &RegisterBlock::offsetC;

        auto nblock = block;
        nblock.*nx /= block.count;
        nblock.ld /= block.count;

        for (int i = 0; i < block.count; i++) {
            xlayout.push_back(nblock);
            nblock.*offsetX += nblock.*nx;
            nblock.simdSize = 0;           // Blocks > 0 do not need loads.
        }
    }

    std::swap(layout, xlayout);
}

// Split blocks that span multiple tiles. Requires each tile to be contained within a single block.
void postprocessLayoutMultitile(Type T, vector<RegisterBlock> &layout, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    if (!atype.tileR || !atype.tileC) return;
    if (isLargeCrosspack(T, atype.crosspack)) return;

    bool needToSplit = false;
    for (const auto &block: layout)
        needToSplit |= (block.colMajor ? (block.nr > atype.tileR) : (block.nc > atype.tileC));

    if (!needToSplit) return;

    vector<RegisterBlock> xlayout;
    xlayout.reserve(layout.size());

    for (const auto &block: layout) {
        auto nx      = block.colMajor ? &RegisterBlock::nr      : &RegisterBlock::nc;
        auto ny      = block.colMajor ? &RegisterBlock::nc      : &RegisterBlock::nr;
        auto offsetX = block.colMajor ? &RegisterBlock::offsetR : &RegisterBlock::offsetC;
        auto offsetY = block.colMajor ? &RegisterBlock::offsetC : &RegisterBlock::offsetR;
        auto tileX   = block.colMajor ? atype.tileR             : atype.tileC;
        auto tileY   = block.colMajor ? atype.tileC             : atype.tileR;

        if (block.*nx == tileX) {
            xlayout.push_back(block);
            continue;
        }

        if (block.*nx % tileX || block.*offsetX % tileX || block.*ny % tileY || block.*offsetY % tileY) stub();
        if (isTransposing(astrategy.accessType)) stub();

        auto nblock = block;
        nblock.*nx = tileX;
        nblock.*ny = tileY;
        nblock.ld = tileX;

        for (int j = 0; j < block.*ny / tileY; j++) {
            for (int i = 0; i < block.*nx / tileX; i++) {
                nblock.*offsetX = block.*offsetX + i * tileX;
                nblock.*offsetY = block.*offsetY + j * tileY;
                xlayout.push_back(nblock);
                nblock.simdSize = 0;
            }
        }
    }

    std::swap(layout, xlayout);
}

// Split large crosspack blocks into smaller pieces so that they can be transposed.
void postprocessLayoutLargeCP(Type T, vector<RegisterBlock> &layout, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    if (!isLargeCrosspack(T, atype.crosspack))
        return;

    bool haveLargeCP = false;
    for (const auto &block: layout) {
        haveLargeCP |= isLargeCrosspack(T, block.crosspack);
        if (haveLargeCP) break;
    }

    if (!haveLargeCP) return;

    vector<RegisterBlock> xlayout;
    xlayout.reserve(layout.size());

    for (const auto &block: layout) {
        if (!isLargeCrosspack(T, block.crosspack))
            xlayout.push_back(block);
        else {
            auto ny      = block.colMajor ? &RegisterBlock::nc      : &RegisterBlock::nr;
            auto offsetY = block.colMajor ? &RegisterBlock::offsetC : &RegisterBlock::offsetR;

            if (block.*ny % block.crosspack)
                return;
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
// Remove unneeded blocks from a dpasw src2 layout.
void postprocessLayoutDPASW(vector<RegisterBlock> &layout, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    if (!astrategy.dpasw)
        return;

    vector<RegisterBlock> nlayout;
    nlayout.reserve(layout.size() / 2);

    bool cm = isLayoutColMajor(layout);
    auto tile = cm ? astrategy.tileC : astrategy.tileR;
    auto offsetX = cm ? &RegisterBlock::offsetC : &RegisterBlock::offsetR;

    for (const auto &block: layout)
        if ((block.*offsetX % (2 * tile)) < tile)
            nlayout.push_back(block);

    layout = std::move(nlayout);
}

void postprocessLayout(Type T, vector<RegisterBlock> &layout, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    postprocessLayout2D(layout, atype, astrategy);
    postprocessLayoutMultitile(T, layout, atype, astrategy);
    postprocessLayoutLargeCP(T, layout, atype, astrategy);
    postprocessLayoutDPASW(layout, atype, astrategy);
}

void finalizeLayout(HW hw, Type T, vector<RegisterBlock> &layout, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    int offsetBytes = 0;
    for (auto &block : layout) {
        if (block.isLoadBlock() || isBlock2D(astrategy.accessType))
            offsetBytes = ngen::utils::alignup_pow2(offsetBytes, GRF::bytes(hw));
        block.calcBytes(T, astrategy);
        block.offsetBytes = offsetBytes;
        offsetBytes += block.bytes;
        block.simplify(T);
    }
}

// Return maximum immediate address offset for a send message.
int maxOffsetAddr(Type T, const MatrixAddressingStrategy &astrategy)
{
    switch (astrategy.base.getModel()) {
        case ModelA64:
        case ModelSLM: return 1 << 19;
        case ModelA32:
        case ModelBTS: return 1 << 11;
        default: return 0;
    }
}

// Identify and combine block address registers that differ only by constant offsets.
void coalesceAddrs(HW hw, Type T, vector<RegisterBlock> &layout, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    if (hw < HW::Xe2) return;
    if (!astrategy.newDP) return;
    if (layout.empty()) return;
    if (astrategy.noCoalesce) return;

    RegisterBlock *anchor = &layout[0];
    int max = maxOffsetAddr(T, astrategy);

    for (auto &block: layout) {
        int dr = block.offsetR - anchor->offsetR;
        int dc = block.offsetC - anchor->offsetC;

        auto accessType = implAccessType(atype, astrategy, block);

        if (isBlock2D(accessType)) {
            if (block.nr == anchor->nr && block.nc == anchor->nc && block.count == anchor->count) {
                int ox, oy;
                switch (atype.layout) {
                    case MatrixLayout::N: ox = dr; oy = dc; break;
                    case MatrixLayout::T: ox = dc; oy = dr; break;
                    default: return;
                }
                block.set2DOffset(ox * T / block.ebytes, oy);
            } else {
                // No match. Make this block the new anchor.
                anchor = &block;
            }
        } else {
            switch (atype.layout) {
                case MatrixLayout::N: if (dc == 0) block.offsetAddr = dr; break;
                case MatrixLayout::T: if (dr == 0) block.offsetAddr = dc; break;
                case MatrixLayout::Pr:
                case MatrixLayout::Pc:
                    auto offsetX = (atype.layout == MatrixLayout::Pc) ? &RegisterBlock::offsetR
                                                                      : &RegisterBlock::offsetC;
                    if (block.*offsetX / atype.packSize == anchor->*offsetX / atype.packSize)
                        block.offsetAddr = untile(T, atype, block);
                    break;
            }

            block.offsetAddr *= T;
            if (block.offsetAddr >= max || block.offsetAddr < -max)
                block.offsetAddr = 0;
            if (one_of(accessType, AccessType::Scattered, AccessType::ChannelScattered))
                if (block.simdSize > anchor->simdSize)
                    block.offsetAddr = 0;
            if (block.offsetAddr & 0x3)
                block.offsetAddr = 0;

            if (block.offsetAddr == 0)
                anchor = &block;
        }
    }
}

bool needsRemask(Type T, bool column, const RegisterBlock &block,
                        const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, bool ignoreMasks = false)
{
    if (!ignoreMasks)
        if (column ? !block.remainderC : !block.remainderR)
            return false;

    bool block2DRemask = isBlock2D(astrategy.accessType)
                      && ((block.colMajor ^ isTransposing(astrategy.accessType)) != column);

    int maskGranularity = block.ebytes;
    if (block.ebytes >= 16)
        maskGranularity = 4;
    if (block2DRemask)
        maskGranularity = std::max(maskGranularity, block2DWidthAlignment(T, block, atype, astrategy));
    if (ignoreMasks && !(block2DRemask && astrategy.address2D))
        maskGranularity = 256;

    return (T.paddedSize() < maskGranularity);
}

bool needsRemask(Type T, bool column, const vector<RegisterBlock> &layout,
                 const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, bool ignoreMasks)
{
    for (auto &block: layout)
        if (needsRemask(T, column, block, atype, astrategy, ignoreMasks))
            return true;
    return false;
}

#include "internal/namespace_end.hxx"
